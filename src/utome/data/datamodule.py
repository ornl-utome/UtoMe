from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from utome.data.dataset import (
    ChannelAffine,
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
    TimestepForecastDataset,
    collate_forecast_batch,
)


def _load_stats(path: Path, variables: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    mean_path = path / "normalize_mean.npz"
    std_path = path / "normalize_std.npz"
    if not mean_path.exists() or not std_path.exists():
        return None, None

    mean_npz = dict(np.load(mean_path))
    std_npz = dict(np.load(std_path))
    mean = np.concatenate([mean_npz[name] if name != "total_precipitation" else np.array([0.0]) for name in variables])
    std = np.concatenate([std_npz[name] for name in variables])
    return mean, std


class GlobalForecastDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        variables: list[str],
        buffer_size: int,
        out_variables: list[str] | None = None,
        uq_root_dir: str | None = None,
        uq_variables: list[str] | None = None,
        predict_range: int = 6,
        hrs_each_step: int = 1,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        strict_uq: bool = False,
        require_uq_variables_match_inputs: bool = False,
        train_shuffle_files: bool = True,
    ) -> None:
        super().__init__()
        if uq_root_dir is not None and uq_variables is None:
            uq_variables = list(variables)
        if require_uq_variables_match_inputs and uq_root_dir is not None and list(uq_variables or []) != list(variables):
            raise ValueError(
                "UQ variables must match input variables when require_uq_variables_match_inputs=True. "
                f"Got {len(uq_variables or [])} UQ variables and {len(variables)} input variables."
            )
        self.save_hyperparameters(logger=False)

        self.root_dir = Path(root_dir)
        self.transforms = ChannelAffine(*_load_stats(self.root_dir, variables))
        output_vars = out_variables if out_variables is not None else variables
        self.output_transforms = ChannelAffine(*_load_stats(self.root_dir, output_vars))
        out_mean, out_std = _load_stats(self.root_dir, output_vars)
        self.output_denormalizer = ChannelAffine(out_mean, out_std, inverse=True)

        self.val_clim = self._get_climatology("val", output_vars)
        self.test_clim = self._get_climatology("test", output_vars)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def _list_partition(self, partition: str) -> list[Path]:
        partition_dir = self.root_dir / partition
        if not partition_dir.exists():
            raise FileNotFoundError(f"Missing partition directory: {partition_dir}")
        return sorted(partition_dir.glob("*.npz"))

    def _get_climatology(self, partition: str, variables: list[str]) -> torch.Tensor | None:
        path = self.root_dir / partition / "climatology.npz"
        if not path.exists():
            return None
        clim = np.load(path)
        return torch.from_numpy(np.concatenate([clim[name] for name in variables])).float()

    def get_lat_lon(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        lat_path = self.root_dir / "lat.npy"
        lon_path = self.root_dir / "lon.npy"
        if not lat_path.exists() or not lon_path.exists():
            return None, None
        return np.load(lat_path), np.load(lon_path)

    def setup(self, stage: str | None = None) -> None:
        if self.data_train is not None:
            return

        common = {
            "variables": self.hparams.variables,
            "out_variables": self.hparams.out_variables,
            "uq_root_dir": self.hparams.uq_root_dir,
            "uq_variables": self.hparams.uq_variables,
            "strict_uq": self.hparams.strict_uq,
        }

        # ``train_shuffle_files=False`` keeps each dataloader worker on the same shards in
        # every epoch. Sample order is still fully shuffled by ShuffleIterableDataset, whose
        # buffer holds a whole worker-epoch, but the file-to-worker assignment becomes
        # stable, so after the first epoch the shards are served from the page cache instead
        # of being re-read from the shared filesystem. On a busy cluster this is the
        # difference between compute-bound and I/O-bound training.
        train_dataset = NpyReader(
            self._list_partition("train"),
            shuffle=self.hparams.train_shuffle_files,
            **common,
        )
        val_dataset = NpyReader(self._list_partition("val"), shuffle=False, **common)
        test_dataset = NpyReader(self._list_partition("test"), shuffle=False, **common)

        self.data_train = ShuffleIterableDataset(
            IndividualForecastDataIter(
                Forecast(train_dataset, predict_range=self.hparams.predict_range, hrs_each_step=self.hparams.hrs_each_step),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            ),
            buffer_size=self.hparams.buffer_size,
        )
        self.data_val = IndividualForecastDataIter(
            Forecast(val_dataset, predict_range=self.hparams.predict_range, hrs_each_step=self.hparams.hrs_each_step),
            transforms=self.transforms,
            output_transforms=self.output_transforms,
        )
        self.data_test = IndividualForecastDataIter(
            Forecast(test_dataset, predict_range=self.hparams.predict_range, hrs_each_step=self.hparams.hrs_each_step),
            transforms=self.transforms,
            output_transforms=self.output_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_forecast_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_forecast_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_forecast_batch,
        )


class TimestepForecastDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        variables: list[str],
        buffer_size: int = 0,
        out_variables: list[str] | None = None,
        predict_range: int = 6,
        hrs_each_step: int = 1,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_stride: int = 1,
        stats_root_dir: str | None = None,
        index_cache_dir: str | None = None,
        pre_normalized: bool = False,
        uq_root_dir: str | None = None,
        uq_variables: list[str] | None = None,
        strict_uq: bool = False,
        uq_steps_per_shard: int = 1095,
        **unused,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["unused"], logger=False)
        self.root_dir = Path(root_dir)
        self.stats_root_dir = Path(stats_root_dir) if stats_root_dir is not None else self.root_dir

        self.transforms = ChannelAffine(None, None) if pre_normalized else ChannelAffine(*_load_stats(self.stats_root_dir, variables))
        output_vars = out_variables if out_variables is not None else variables
        self.output_transforms = ChannelAffine(None, None) if pre_normalized else ChannelAffine(*_load_stats(self.stats_root_dir, output_vars))
        out_mean, out_std = _load_stats(self.stats_root_dir, output_vars)
        self.output_denormalizer = ChannelAffine(out_mean, out_std, inverse=True)

        self.val_clim = self._get_climatology("val", output_vars)
        self.test_clim = self._get_climatology("test", output_vars)
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def _get_climatology(self, partition: str, variables: list[str]) -> torch.Tensor | None:
        for root in (self.stats_root_dir, self.root_dir):
            path = root / partition / "climatology.npz"
            if path.exists():
                clim = np.load(path)
                return torch.from_numpy(np.concatenate([clim[name] for name in variables])).float()
        return None

    def get_lat_lon(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        for root in (self.stats_root_dir, self.root_dir):
            lat_path = root / "lat.npy"
            lon_path = root / "lon.npy"
            if lat_path.exists() and lon_path.exists():
                return np.load(lat_path), np.load(lon_path)
        return None, None

    def setup(self, stage: str | None = None) -> None:
        if self.data_train is not None:
            return

        common = {
            "root_dir": self.root_dir,
            "variables": self.hparams.variables,
            "out_variables": self.hparams.out_variables,
            "predict_range": self.hparams.predict_range,
            "hrs_each_step": self.hparams.hrs_each_step,
            "transforms": self.transforms,
            "output_transforms": self.output_transforms,
            "sample_stride": self.hparams.sample_stride,
            "index_cache_dir": self.hparams.index_cache_dir,
            "uq_root_dir": self.hparams.uq_root_dir,
            "uq_variables": self.hparams.uq_variables,
            "strict_uq": self.hparams.strict_uq,
            "uq_steps_per_shard": self.hparams.uq_steps_per_shard,
        }
        self.data_train = TimestepForecastDataset(partition="train", **common)
        self.data_val = TimestepForecastDataset(partition="val", **common)
        self.data_test = TimestepForecastDataset(partition="test", **common)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=collate_forecast_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=collate_forecast_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=collate_forecast_batch,
        )
