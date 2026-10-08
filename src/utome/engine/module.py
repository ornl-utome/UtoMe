from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from utome.models.importance import build_utome_teacher, normalized_mse_loss, pairwise_rank_loss
from utome.models.strategies import get_utome_strategy
from utome.models.uq_controls import apply_uq_control
from utome.models.utome import UtoMeModel
from utome.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utome.utils.metrics import lat_weighted_acc, lat_weighted_mse, lat_weighted_mse_val, lat_weighted_rmse, mse
from utome.utils.rebuttal_metrics import ExtendedMetricAccumulator


class GlobalForecastModule(LightningModule):
    def __init__(
        self,
        net: UtoMeModel,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        importance_weight: float = 0.1,
        importance_type: str = "pairwise_rank",
        teacher_target_variables: tuple[str, ...] = (),
        teacher_beta: float = 1.0,
        teacher_mode: str = "dynamics_minus_uq",
        teacher_variable_scope: str = "target",
        teacher_uq_source: str = "eda",
        merge_warmup_epochs: int = 0,
        uq_control: str = "none",
        uq_control_seed: int = 0,
        extended_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        self.lat: np.ndarray | None = None
        self.denormalize = None
        self.pred_range: int | None = None
        self.val_clim: torch.Tensor | None = None
        self.test_clim: torch.Tensor | None = None
        self._extended: ExtendedMetricAccumulator | None = None
        self.extended_results: dict[str, float] = {}
        self._uq_generator: torch.Generator | None = None

    def set_lat_lon(self, lat: np.ndarray | None, lon: np.ndarray | None) -> None:
        del lon
        self.lat = lat

    def set_denormalization(self, transform) -> None:
        self.denormalize = transform

    def set_pred_range(self, value: int) -> None:
        self.pred_range = value

    def set_val_clim(self, clim: torch.Tensor | None) -> None:
        self.val_clim = clim

    def set_test_clim(self, clim: torch.Tensor | None) -> None:
        self.test_clim = clim

    def _control_uq(self, uq: torch.Tensor | None) -> torch.Tensor | None:
        control = getattr(self.hparams, "uq_control", "none")
        if uq is None or control in (None, "", "none"):
            return uq
        if self._uq_generator is None or self._uq_generator.device != uq.device:
            self._uq_generator = torch.Generator(device=uq.device)
            self._uq_generator.manual_seed(int(getattr(self.hparams, "uq_control_seed", 0)))
        return apply_uq_control(uq, control, generator=self._uq_generator)

    def _build_teacher(self, batch: dict[str, Any]):
        targets = self.hparams.teacher_target_variables
        if self.hparams.teacher_variable_scope == "input":
            targets = tuple(batch["variables"])
        elif self.hparams.teacher_variable_scope == "uq":
            targets = tuple(batch["uq_variables"] or ())
        elif not targets:
            targets = tuple(batch["out_variables"])
        elif self.hparams.teacher_variable_scope != "target":
            raise ValueError(f"Unsupported teacher_variable_scope: {self.hparams.teacher_variable_scope}")
        return build_utome_teacher(
            inputs=batch["x"],
            variables=batch["variables"],
            uq=self._control_uq(batch["uq"]),
            uq_variables=batch["uq_variables"],
            target_variables=tuple(targets),
            patch_size=self.net.patch_size,
            beta=self.hparams.teacher_beta,
            mode=self.hparams.teacher_mode,
            uq_source=self.hparams.teacher_uq_source,
        )

    def _teacher_merge_importance(self, teacher, stage: str):
        if teacher is None:
            return None
        strategy = get_utome_strategy(self.net.method)
        if strategy is not None and strategy.uses_direct_teacher_for_merge:
            return teacher.importance
        if (
            stage in {"train", "val"}
            and strategy is not None
            and strategy.learns_uq_predictor
            and self.current_epoch < self.hparams.merge_warmup_epochs
        ):
            return teacher.importance
        return None

    def _importance_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.hparams.importance_type == "pairwise_rank":
            return pairwise_rank_loss(predicted, target)
        if self.hparams.importance_type == "mse":
            return normalized_mse_loss(predicted, target)
        raise ValueError(f"Unsupported importance_type: {self.hparams.importance_type}")

    def _forecast_loss(self, preds: torch.Tensor, target: torch.Tensor, out_variables: tuple[str, ...]) -> dict[str, torch.Tensor]:
        if self.lat is None:
            return mse(preds, target, out_variables)
        return lat_weighted_mse(preds, target, out_variables, self.lat)

    def _eval_metrics(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        out_variables: tuple[str, ...],
        clim: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.denormalize is not None:
            preds_denorm = self.denormalize(preds)
            target_denorm = self.denormalize(target)
        else:
            preds_denorm = preds
            target_denorm = target

        if self.lat is None:
            values = {"w_rmse": F.mse_loss(preds_denorm, target_denorm).sqrt()}
            return values

        metrics = {}
        metrics.update(lat_weighted_mse_val(preds, target, out_variables, self.lat))
        metrics.update(lat_weighted_rmse(preds_denorm, target_denorm, out_variables, self.lat))
        metrics.update(lat_weighted_acc(preds_denorm, target_denorm, out_variables, self.lat, clim=clim))
        return metrics

    def _shared_step(self, batch: dict[str, Any], stage: str) -> tuple[torch.Tensor, dict[str, torch.Tensor | None], Any]:
        teacher = self._build_teacher(batch)
        strategy = get_utome_strategy(self.net.method)
        if strategy is not None and strategy.needs_external_uq and teacher is None:
            raise ValueError(f"{self.net.method} requires external UQ during {stage}")
        merge_importance = self._teacher_merge_importance(teacher, stage)
        if strategy is not None and strategy.uses_direct_teacher_for_merge and merge_importance is None:
            raise ValueError(f"{self.net.method} requires UQ inputs or proxy-UQ to build an uncertainty signal")

        outputs = self.net(
            x=batch["x"],
            lead_times=batch["lead_times"],
            variables=batch["variables"],
            out_variables=batch["out_variables"],
            merge_importance=merge_importance,
        )

        loss_dict = self._forecast_loss(outputs["preds"], batch["y"], batch["out_variables"])
        total_loss = loss_dict["loss"]
        batch_size = batch["x"].shape[0]
        self.log(
            f"{stage}/forecast_loss",
            total_loss,
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        predicted_importance = outputs["predicted_importance"]
        if predicted_importance is not None and teacher is not None:
            importance_loss = self._importance_loss(predicted_importance, teacher.importance)
            total_loss = total_loss + self.hparams.importance_weight * importance_loss
            self.log(
                f"{stage}/importance_loss",
                importance_loss,
                prog_bar=False,
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        return total_loss, outputs, teacher

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, _, _ = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, outputs, _ = self._shared_step(batch, stage="val")
        metrics = self._eval_metrics(outputs["preds"], batch["y"], batch["out_variables"], self.val_clim)
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == "w_rmse"),
                sync_dist=True,
                batch_size=batch["x"].shape[0],
            )
        return loss

    def on_test_epoch_start(self) -> None:
        self.extended_results = {}
        if self.hparams.extended_metrics and self.lat is not None:
            self._extended = ExtendedMetricAccumulator(variables=(), lat=self.lat)
        else:
            self._extended = None

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        teacher = self._build_teacher(batch)
        merge_importance = self._teacher_merge_importance(teacher, "test")
        outputs = self.net(
            x=batch["x"],
            lead_times=batch["lead_times"],
            variables=batch["variables"],
            out_variables=batch["out_variables"],
            merge_importance=merge_importance,
        )
        metrics = self._eval_metrics(outputs["preds"], batch["y"], batch["out_variables"], self.test_clim)
        for name, value in metrics.items():
            self.log(f"test/{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["x"].shape[0])

        if self._extended is not None:
            if not self._extended.variables:
                self._extended.variables = tuple(batch["out_variables"])
            preds_denorm = self.denormalize(outputs["preds"]) if self.denormalize is not None else outputs["preds"]
            target_denorm = self.denormalize(batch["y"]) if self.denormalize is not None else batch["y"]
            self._extended.update(
                preds=preds_denorm,
                target=target_denorm,
                clim=self.test_clim,
                uq_patch=None if teacher is None else teacher.uq_patch,
                dynamics_patch=None if teacher is None else teacher.dynamics_patch,
                patch_size=self.net.patch_size,
            )
        return metrics["w_rmse"]

    def on_test_epoch_end(self) -> None:
        if self._extended is None:
            return
        # Kept out of ``self.log`` on purpose: these are epoch-level scalars that the
        # launcher writes to ``metrics.json`` next to the run directory.
        self.extended_results = self._extended.compute()
        self._extended.reset()

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in ("var_embed", "pos_embed", "var_query")):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0.0,
                },
            ]
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.hparams.warmup_steps,
            max_steps=self.hparams.max_steps,
            warmup_start_lr=self.hparams.warmup_start_lr,
            eta_min=self.hparams.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
