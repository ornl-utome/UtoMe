import json
import os
import tempfile
import unittest
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch.nn as nn

from utome.data.dataset import TimestepForecastDataset


def _write_fake_timesteps(root: Path, partition: str = "train", n_steps: int = 6) -> None:
    (root / partition).mkdir(parents=True, exist_ok=True)
    with (root / "variable_names.json").open("w", encoding="utf-8") as handle:
        json.dump(["a", "b", "c"], handle)
    for step in range(n_steps):
        frame = np.stack(
            [
                np.full((2, 3), step, dtype=np.float16),
                np.full((2, 3), step + 10, dtype=np.float16),
                np.full((2, 3), step + 20, dtype=np.float16),
            ],
            axis=0,
        )
        np.save(root / partition / f"1979_0_{step:04d}.npy", frame)


def _build_dataset(root: Path, cache_dir: Path, rank: int = 0) -> tuple[int, tuple[int, ...], tuple[int, ...], float]:
    previous_rank = os.environ.get("RANK")
    os.environ["RANK"] = str(rank)
    try:
        dataset = TimestepForecastDataset(
            root_dir=root,
            variables=["a", "c"],
            out_variables=["b"],
            predict_range=2,
            hrs_each_step=1,
            transforms=nn.Identity(),
            output_transforms=nn.Identity(),
            partition="train",
            index_cache_dir=cache_dir,
        )
        sample = dataset[0]
        return len(dataset), tuple(sample["x"].shape), tuple(sample["y"].shape), float(sample["lead_times"])
    finally:
        if previous_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = previous_rank


def _worker(args: tuple[str, str, int]) -> tuple[int, tuple[int, ...], tuple[int, ...], float]:
    root_s, cache_s, rank = args
    return _build_dataset(Path(root_s), Path(cache_s), rank)


class TimestepForecastDatasetTest(unittest.TestCase):
    def test_builds_expected_pairs_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            cache = Path(tmp) / "cache"
            _write_fake_timesteps(root)

            length, x_shape, y_shape, lead_time = _build_dataset(root, cache)

            self.assertEqual(length, 4)
            self.assertEqual(x_shape, (2, 2, 3))
            self.assertEqual(y_shape, (1, 2, 3))
            self.assertAlmostEqual(lead_time, 0.02)

    def test_recovers_from_corrupt_cache_on_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            cache = Path(tmp) / "cache"
            cache.mkdir(parents=True)
            _write_fake_timesteps(root)
            (cache / "train_index_pr2_ss1.json").write_text("", encoding="utf-8")

            length, _, _, _ = _build_dataset(root, cache)

            self.assertEqual(length, 4)
            with (cache / "train_index_pr2_ss1.json").open("r", encoding="utf-8") as handle:
                self.assertEqual(len(json.load(handle)), 4)

    def test_parallel_ranks_share_atomic_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            cache = Path(tmp) / "cache"
            _write_fake_timesteps(root, n_steps=8)

            ctx = get_context("spawn")
            with ctx.Pool(processes=4) as pool:
                results = pool.map(_worker, [(str(root), str(cache), rank) for rank in range(4)])

            for length, x_shape, y_shape, lead_time in results:
                self.assertEqual(length, 6)
                self.assertEqual(x_shape, (2, 2, 3))
                self.assertEqual(y_shape, (1, 2, 3))
                self.assertAlmostEqual(lead_time, 0.02)


if __name__ == "__main__":
    unittest.main()
