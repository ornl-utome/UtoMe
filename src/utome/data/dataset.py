import math
import random
import json
import os
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset


class ChannelAffine(nn.Module):
    def __init__(self, mean: np.ndarray | None, std: np.ndarray | None, inverse: bool = False) -> None:
        super().__init__()
        self.inverse = inverse

        if mean is None or std is None:
            self.register_buffer("mean", torch.empty(0), persistent=False)
            self.register_buffer("std", torch.empty(0), persistent=False)
            self.enabled = False
        else:
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
            self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        if self.inverse:
            x = x * self.std + self.mean
        else:
            x = (x - self.mean) / self.std.clamp_min(1e-6)
        return x.squeeze(0) if squeeze else x


class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list: Iterable[Path],
        variables: list[str],
        out_variables: list[str] | None,
        uq_root_dir: str | None = None,
        uq_variables: list[str] | None = None,
        strict_uq: bool = False,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.file_list = [Path(path) for path in file_list if "climatology" not in str(path)]
        self.variables = list(variables)
        self.out_variables = list(out_variables) if out_variables is not None else list(variables)
        self.uq_root_dir = Path(uq_root_dir) if uq_root_dir else None
        self.uq_variables = list(uq_variables) if uq_variables is not None else None
        self.strict_uq = strict_uq
        self.shuffle = shuffle

    def _resolve_uq_path(self, data_path: Path) -> Path | None:
        if self.uq_root_dir is None:
            return None
        partition = data_path.parent.name
        candidate = self.uq_root_dir / partition / data_path.name
        if candidate.exists():
            return candidate
        if self.strict_uq:
            raise FileNotFoundError(f"Missing UQ file for {data_path}: expected {candidate}")
        return None

    def __iter__(self):
        file_list = list(self.file_list)
        if self.shuffle:
            random.shuffle(file_list)

        worker_info = torch.utils.data.get_worker_info()
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id_local = worker_info.id if worker_info is not None else 0
        num_shards = max(1, num_workers * world_size)
        shard_id = rank * num_workers + worker_id_local

        indices = list(range(shard_id, len(file_list), num_shards))

        for idx in indices:
            data_path = file_list[idx]
            data = np.load(data_path)
            uq_path = self._resolve_uq_path(data_path)
            uq_data = np.load(uq_path) if uq_path is not None else None

            uq = None
            if uq_data is not None and self.uq_variables is not None:
                missing = [name for name in self.uq_variables if name not in uq_data]
                if missing:
                    raise KeyError(f"UQ file {uq_path} is missing variables: {missing}")
                uq = {name: uq_data[name] for name in self.uq_variables}

            yield {
                "data": {name: data[name] for name in self.variables},
                "uq": uq,
                "variables": tuple(self.variables),
                "out_variables": tuple(self.out_variables),
                "uq_variables": None if self.uq_variables is None else tuple(self.uq_variables),
            }


class Forecast(IterableDataset):
    def __init__(self, dataset: NpyReader, predict_range: int, hrs_each_step: int = 1) -> None:
        super().__init__()
        self.dataset = dataset
        self.predict_range = predict_range
        self.hrs_each_step = hrs_each_step

    def __iter__(self):
        for sample in self.dataset:
            data = sample["data"]
            x = np.concatenate([data[name].astype(np.float32) for name in sample["variables"]], axis=1)
            y = np.concatenate([data[name].astype(np.float32) for name in sample["out_variables"]], axis=1)

            inputs = torch.from_numpy(x[:-self.predict_range])
            targets = torch.from_numpy(y)
            predict_ranges = torch.full((inputs.shape[0],), self.predict_range, dtype=torch.long)
            output_ids = torch.arange(inputs.shape[0], dtype=torch.long) + predict_ranges
            outputs = targets[output_ids]
            lead_times = (self.hrs_each_step * predict_ranges / 100).to(torch.float32)

            uq_inputs = None
            if sample["uq"] is not None:
                uq = np.concatenate([sample["uq"][name].astype(np.float32) for name in sample["uq_variables"]], axis=1)
                uq_inputs = torch.from_numpy(uq[:-self.predict_range])

            yield {
                "inputs": inputs,
                "targets": outputs,
                "lead_times": lead_times,
                "variables": sample["variables"],
                "out_variables": sample["out_variables"],
                "uq": uq_inputs,
                "uq_variables": sample["uq_variables"],
            }


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset: Forecast, transforms: nn.Module, output_transforms: nn.Module) -> None:
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms

    def __iter__(self):
        for sample in self.dataset:
            num_steps = sample["inputs"].shape[0]
            for step in range(num_steps):
                uq = None if sample["uq"] is None else sample["uq"][step]
                yield {
                    "x": self.transforms(sample["inputs"][step]),
                    "y": self.output_transforms(sample["targets"][step]),
                    "lead_times": sample["lead_times"][step],
                    "variables": sample["variables"],
                    "out_variables": sample["out_variables"],
                    "uq": uq,
                    "uq_variables": sample["uq_variables"],
                }


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, buffer_size: int) -> None:
        super().__init__()
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer: list[dict] = []
        for item in self.dataset:
            if len(buffer) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buffer[idx]
                buffer[idx] = item
            else:
                buffer.append(item)

        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()


def collate_forecast_batch(batch: list[dict]) -> dict:
    uq = None
    if batch[0]["uq"] is not None:
        uq = torch.stack([item["uq"] for item in batch], dim=0)

    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "lead_times": torch.stack([item["lead_times"] for item in batch], dim=0),
        "variables": batch[0]["variables"],
        "out_variables": batch[0]["out_variables"],
        "uq": uq,
        "uq_variables": batch[0]["uq_variables"],
    }


class TimestepForecastDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        variables: list[str],
        out_variables: list[str] | None,
        predict_range: int,
        hrs_each_step: int,
        transforms: nn.Module,
        output_transforms: nn.Module,
        partition: str,
        sample_stride: int = 1,
        index_cache_dir: str | Path | None = None,
        uq_root_dir: str | Path | None = None,
        uq_variables: list[str] | None = None,
        strict_uq: bool = False,
        uq_steps_per_shard: int = 1095,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.partition_dir = self.root_dir / partition
        self.uq_root_dir = Path(uq_root_dir) if uq_root_dir else None
        self.uq_partition_dir = None if self.uq_root_dir is None else self.uq_root_dir / partition
        self.variables = tuple(variables)
        self.out_variables = tuple(out_variables) if out_variables is not None else tuple(variables)
        self.uq_variables = tuple(uq_variables) if uq_variables is not None else None
        self.strict_uq = strict_uq
        self.uq_steps_per_shard = uq_steps_per_shard
        self._uq_year_cache: dict[str, np.ndarray] = {}
        self.predict_range = predict_range
        self.hrs_each_step = hrs_each_step
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.sample_stride = sample_stride
        if sample_stride <= 0:
            raise ValueError("sample_stride must be positive")

        var_path = self.root_dir / "variable_names.json"
        if not var_path.exists():
            raise FileNotFoundError(f"Missing timestep variable index: {var_path}")
        with var_path.open("r", encoding="utf-8") as handle:
            all_variables = json.load(handle)
        var_to_idx = {name: idx for idx, name in enumerate(all_variables)}
        missing_inputs = [name for name in self.variables if name not in var_to_idx]
        missing_outputs = [name for name in self.out_variables if name not in var_to_idx]
        if missing_inputs or missing_outputs:
            raise KeyError(
                f"Converted timestep data is missing variables. "
                f"inputs={missing_inputs}, outputs={missing_outputs}"
            )
        self.input_indices = np.asarray([var_to_idx[name] for name in self.variables], dtype=np.int64)
        self.output_indices = np.asarray([var_to_idx[name] for name in self.out_variables], dtype=np.int64)
        self.uq_indices = None
        if self.uq_root_dir is not None:
            uq_var_path = self.uq_root_dir / "variable_names.json"
            if not uq_var_path.exists():
                raise FileNotFoundError(f"Missing timestep UQ variable index: {uq_var_path}")
            with uq_var_path.open("r", encoding="utf-8") as handle:
                all_uq_variables = json.load(handle)
            if self.uq_variables is None:
                self.uq_variables = tuple(self.variables)
            uq_var_to_idx = {name: idx for idx, name in enumerate(all_uq_variables)}
            missing_uq = [name for name in self.uq_variables if name not in uq_var_to_idx]
            if missing_uq:
                raise KeyError(f"Converted timestep UQ data is missing variables: {missing_uq}")
            self.uq_indices = np.asarray([uq_var_to_idx[name] for name in self.uq_variables], dtype=np.int64)

        cache_dir = Path(index_cache_dir) if index_cache_dir is not None else self.root_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = cache_dir / f"{partition}_index_pr{predict_range}_ss{sample_stride}.json"
        self.samples = self._load_or_build_index()

    def _read_index(self) -> list[tuple[str, str]] | None:
        if not self.index_path.exists() or self.index_path.stat().st_size == 0:
            return None
        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                return [(item[0], item[1]) for item in json.load(handle)]
        except (JSONDecodeError, OSError, IndexError, TypeError):
            return None

    def _wait_for_index(self, timeout_s: int = 600) -> list[tuple[str, str]] | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            samples = self._read_index()
            if samples is not None:
                return samples
            time.sleep(2)
        return None

    def _load_or_build_index(self) -> list[tuple[str, str]]:
        cached = self._read_index()
        if cached is not None:
            return cached

        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            cached = self._wait_for_index()
            if cached is not None:
                return cached

        groups: dict[str, dict[int, str]] = {}
        for path in sorted(self.partition_dir.glob("*.npy")):
            stem = path.stem
            prefix, step_s = stem.rsplit("_", 1)
            if not step_s.isdigit():
                continue
            groups.setdefault(prefix, {})[int(step_s)] = path.name

        samples: list[tuple[str, str]] = []
        for prefix in sorted(groups):
            step_map = groups[prefix]
            if not step_map:
                continue
            max_step = max(step_map)
            for step in range(0, max_step - self.predict_range + 1, self.sample_stride):
                target_step = step + self.predict_range
                if step in step_map and target_step in step_map:
                    samples.append((step_map[step], step_map[target_step]))

        if rank == 0:
            tmp_path = self.index_path.with_name(f"{self.index_path.name}.{os.getpid()}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(samples, handle)
            tmp_path.replace(self.index_path)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_packed_uq(self, input_name: str) -> torch.Tensor | None:
        if self.uq_partition_dir is None or self.uq_indices is None:
            return None
        parts = Path(input_name).stem.split("_")
        if len(parts) < 3 or not parts[-3].isdigit() or not parts[-2].isdigit() or not parts[-1].isdigit():
            return None
        year, shard, step = parts[-3], int(parts[-2]), int(parts[-1])
        packed_path = self.uq_partition_dir / f"{year}.npy"
        if not packed_path.exists():
            return None
        if year not in self._uq_year_cache:
            self._uq_year_cache[year] = np.load(packed_path, mmap_mode="r")
        timestep = shard * self.uq_steps_per_shard + step
        uq_np = self._uq_year_cache[year][timestep, self.uq_indices]
        return torch.from_numpy(np.asarray(uq_np, dtype=np.float32))

    def __getitem__(self, index: int) -> dict:
        input_name, target_name = self.samples[index]
        x_np = np.load(self.partition_dir / input_name, mmap_mode="r")[self.input_indices]
        y_np = np.load(self.partition_dir / target_name, mmap_mode="r")[self.output_indices]
        uq = None
        if self.uq_partition_dir is not None and self.uq_indices is not None:
            uq_path = self.uq_partition_dir / input_name
            if uq_path.exists():
                uq_np = np.load(uq_path, mmap_mode="r")[self.uq_indices]
                uq = torch.from_numpy(np.asarray(uq_np, dtype=np.float32))
            else:
                uq = self._load_packed_uq(input_name)
            if uq is None and self.strict_uq:
                packed_hint = self.uq_partition_dir / f"{Path(input_name).stem.split('_')[0]}.npy"
                raise FileNotFoundError(f"Missing timestep UQ file: {uq_path} or packed file {packed_hint}")
        x = self.transforms(torch.from_numpy(np.asarray(x_np, dtype=np.float32)))
        y = self.output_transforms(torch.from_numpy(np.asarray(y_np, dtype=np.float32)))
        return {
            "x": x,
            "y": y,
            "lead_times": torch.tensor(self.hrs_each_step * self.predict_range / 100, dtype=torch.float32),
            "variables": self.variables,
            "out_variables": self.out_variables,
            "uq": uq,
            "uq_variables": self.uq_variables,
        }
