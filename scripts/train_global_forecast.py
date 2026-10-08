"""Train or evaluate UtoMe using one YAML configuration."""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utome.data import GlobalForecastDataModule, TimestepForecastDataModule
from utome.engine import GlobalForecastModule
from utome.models import UtoMeModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, help="Resume training or load weights for testing.")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.test_only and args.ckpt is None:
        parser.error("--test-only requires --ckpt")
    if args.ckpt is not None and not args.ckpt.is_file():
        parser.error(f"Checkpoint not found: {args.ckpt}")

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    pl.seed_everything(config.get("seed", 42), workers=True)
    model_cfg = config["model"]
    utome_cfg = model_cfg.get("utome", {})
    model = UtoMeModel(**model_cfg)
    data_cfg = dict(config["data"])
    data_format = data_cfg.pop("format", "npz")
    if args.test_only and model.method in {"utome_p", "utome_s", "utome_plus", "utome_pp"}:
        data_cfg.update(uq_root_dir=None, strict_uq=False)
    if data_format == "npz":
        data = GlobalForecastDataModule(**data_cfg)
    elif data_format == "timestep":
        data = TimestepForecastDataModule(**data_cfg)
    else:
        raise ValueError(f"Unsupported data.format: {data_format}")

    module = GlobalForecastModule(
        net=model,
        **config["optimization"],
        **config["loss"],
        teacher_target_variables=tuple(utome_cfg.get("target_variables", [])),
        teacher_beta=utome_cfg.get("teacher_beta", 1.0),
        teacher_mode=utome_cfg.get("teacher_mode", "dynamics_minus_uq"),
        teacher_variable_scope=utome_cfg.get("teacher_variable_scope", "target"),
        teacher_uq_source=utome_cfg.get("teacher_uq_source", "eda"),
        merge_warmup_epochs=utome_cfg.get("merge_warmup_epochs", 0),
    )
    module.set_lat_lon(*data.get_lat_lon())
    module.set_denormalization(data.output_denormalizer)
    module.set_pred_range(data_cfg["predict_range"])
    module.set_val_clim(data.val_clim)
    module.set_test_clim(data.test_clim)

    trainer_cfg = dict(config["trainer"])
    if "SLURM_JOB_NUM_NODES" in os.environ:
        trainer_cfg["num_nodes"] = int(os.environ["SLURM_JOB_NUM_NODES"])
    output_dir = Path(trainer_cfg.get("default_root_dir", "outputs/utome"))
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    monitor = "val/w_rmse" if trainer_cfg.get("limit_val_batches", 1.0) else None
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint],
        **trainer_cfg,
    )
    if args.test_only:
        trainer.test(module, datamodule=data, ckpt_path=str(args.ckpt))
    else:
        trainer.fit(module, datamodule=data, ckpt_path=str(args.ckpt) if args.ckpt else None)
        if config.get("run_test_after_fit", False):
            trainer.test(module, datamodule=data, ckpt_path=checkpoint.best_model_path or None)
    if trainer.is_global_zero:
        metrics = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                   if isinstance(v, torch.Tensor) and v.numel() == 1}
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / ("test_metrics.yaml" if args.test_only else "metrics.yaml")).open("w") as handle:
            yaml.safe_dump(metrics, handle)


if __name__ == "__main__":
    main()
