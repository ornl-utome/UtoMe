"""Exercise the public CLI with small synthetic weather fields on CPU."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ["geopotential_500", "temperature_850", "2m_temperature"]


def write_data(root: Path, data_format: str) -> None:
    rng = np.random.default_rng(7)
    for folder in (root / "data", root / "uq"):
        folder.mkdir(parents=True)
        (folder / "variable_names.json").write_text(json.dumps(VARIABLES))
    np.save(root / "data/lat.npy", np.linspace(-80, 80, 8))
    np.save(root / "data/lon.npy", np.linspace(0, 360, 16, endpoint=False))
    np.savez(root / "data/normalize_mean.npz", **{v: np.zeros(1) for v in VARIABLES})
    np.savez(root / "data/normalize_std.npz", **{v: np.ones(1) for v in VARIABLES})
    for split in ("train", "val", "test"):
        frames = rng.normal(size=(8, len(VARIABLES), 8, 16)).astype("float32")
        spread = rng.uniform(0.1, 1, frames.shape).astype("float32")
        for name, array in (("data", frames), ("uq", spread)):
            folder = root / name / split
            folder.mkdir()
            if data_format == "npz":
                np.savez(folder / "1979_0.npz", **{v: array[:, i:i+1] for i, v in enumerate(VARIABLES)})
            else:
                for step, frame in enumerate(array):
                    np.save(folder / f"1979_0_{step:04d}.npy", frame)
        if split != "train":
            np.savez(root / "data" / split / "climatology.npz",
                     **{v: np.zeros((1, 8, 16)) for v in VARIABLES})


class TrainingSmokeTest(unittest.TestCase):
    def run_cli(self, config: Path, *args: str) -> None:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_global_forecast.py"),
             "--config", str(config), *args],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_train_checkpoint_and_test_for_all_variants(self) -> None:
        for data_format in ("npz", "timestep"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_data(root, data_format)
                for method in ("utome_e", "utome_p", "utome_s", "baseline"):
                    with self.subTest(data_format=data_format, method=method):
                        cfg = yaml.safe_load((ROOT / "configs/5p625/utome_e_25.yaml").read_text())
                        cfg["model"].update(default_vars=VARIABLES, img_size=[8, 16],
                                            patch_size=2, embed_dim=16, depth=4,
                                            decoder_depth=1, num_heads=2, drop_path=0, drop_rate=0)
                        cfg["model"]["utome"].update(method=method, probe_layer=0,
                            importance_hidden_dim=8, target_variables=VARIABLES,
                            teacher_uq_source="proxy" if method == "utome_s" else "eda")
                        cfg["trainer"].update(accelerator="cpu", devices=1, num_nodes=1,
                            strategy="auto", precision="32-true", max_epochs=1, max_steps=1,
                            check_val_every_n_epoch=1, limit_train_batches=1, limit_val_batches=1,
                            limit_test_batches=1, num_sanity_val_steps=0, log_every_n_steps=1,
                            enable_progress_bar=False, enable_model_summary=False,
                            default_root_dir=str(root / method))
                        cfg["optimization"].update(warmup_steps=0, max_steps=2)
                        external = method in {"utome_e", "utome_p"}
                        cfg["data"].update(root_dir=str(root / "data"),
                            uq_root_dir=str(root / "uq") if external else None,
                            variables=VARIABLES, out_variables=VARIABLES, uq_variables=VARIABLES,
                            format=data_format, predict_range=2, batch_size=2, num_workers=0,
                            strict_uq=external)
                        cfg["run_test_after_fit"] = True
                        config = root / f"{method}.yaml"
                        config.write_text(yaml.safe_dump(cfg))
                        self.run_cli(config)
                        ckpt = root / method / "checkpoints/last.ckpt"
                        self.assertTrue(ckpt.is_file())
                        metrics = yaml.safe_load((root / method / "metrics.yaml").read_text())
                        self.assertTrue(math.isfinite(metrics["test/w_rmse"]))
                        if method == "utome_p":
                            # Deployment must work without any external uncertainty directory.
                            cfg["data"]["uq_root_dir"] = str(root / "missing_uq")
                            config.write_text(yaml.safe_dump(cfg))
                        self.run_cli(config, "--test-only", "--ckpt", str(ckpt))
                        loaded = yaml.safe_load((root / method / "test_metrics.yaml").read_text())
                        self.assertAlmostEqual(loaded["test/w_rmse"], metrics["test/w_rmse"], places=5)


if __name__ == "__main__":
    unittest.main()
