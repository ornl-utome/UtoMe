#!/usr/bin/env python3
"""Convert EDA spread NetCDF files into per-timestep UQ .npy files.

The output mirrors the timestep ERA5 layout:

  <out>/<partition>/<year>.npy

Each file contains a float16 array with shape (T, C, H, W). We generate only
the UQ variables requested by the merge teacher, which keeps the 1.40625 degree
experiments tractable while preserving the external-UQ signal used by UtoMe-E/P.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_UQ_VARIABLES = [
    "geopotential_500",
    "temperature_850",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

SINGLE_MAP = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}

PRESSURE_MAP = {
    "geopotential": ("geopotential", "z"),
    "temperature": ("temperature", "t"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "specific_humidity": ("specific_humidity", "q"),
}


def parse_variable(name: str) -> tuple[str, int] | None:
    for prefix in PRESSURE_MAP:
        token = f"{prefix}_"
        if name.startswith(token):
            return prefix, int(name[len(token):])
    return None


def resample_lat(arr: np.ndarray, src_lat: np.ndarray, dst_lat: np.ndarray) -> np.ndarray:
    order = np.argsort(src_lat)
    src = src_lat[order]
    arr_sorted = np.take(arr, order, axis=-2)
    idx = np.searchsorted(src, dst_lat).clip(1, len(src) - 1)
    x0 = src[idx - 1]
    x1 = src[idx]
    w = ((dst_lat - x0) / (x1 - x0)).astype(arr.dtype)
    lo = np.take(arr_sorted, idx - 1, axis=-2)
    hi = np.take(arr_sorted, idx, axis=-2)
    w_shape = (1,) * (arr.ndim - 2) + (dst_lat.shape[0], 1)
    return lo + w.reshape(w_shape) * (hi - lo)


def to_hourly(arr: np.ndarray, target_steps: int) -> np.ndarray:
    hourly = np.repeat(arr, 3, axis=0)
    if hourly.shape[0] > target_steps:
        return hourly[:target_steps]
    if hourly.shape[0] < target_steps:
        pad = np.repeat(hourly[-1:], target_steps - hourly.shape[0], axis=0)
        return np.concatenate([hourly, pad], axis=0)
    return hourly


def load_single_variable(eda_root: Path, year: int, variable: str, dst_lat: np.ndarray, target_steps: int) -> np.ndarray:
    path = eda_root / "single_level" / f"eda_single_4var_{year}.nc"
    if not path.exists():
        alt = eda_root / "single_level" / f"eda_single_{year}.nc"
        path = alt if alt.exists() else path
    src_name = SINGLE_MAP[variable]
    with xr.open_dataset(path) as ds:
        if src_name not in ds:
            raise KeyError(f"{path} is missing {src_name} for {variable}")
        src_lat = np.asarray(ds.latitude.values, dtype=np.float32)
        arr = np.asarray(ds[src_name].values, dtype=np.float32)
    arr = resample_lat(arr, src_lat, dst_lat)
    return to_hourly(arr, target_steps)


def load_pressure_variable(eda_root: Path, year: int, variable: str, dst_lat: np.ndarray, target_steps: int) -> np.ndarray:
    parsed = parse_variable(variable)
    if parsed is None:
        raise ValueError(f"Not a pressure-level variable: {variable}")
    prefix, level = parsed
    file_key, data_key = PRESSURE_MAP[prefix]
    path = eda_root / "pressure_level" / f"eda_plev_{file_key}_{year}.nc"
    with xr.open_dataset(path) as ds:
        if data_key not in ds:
            raise KeyError(f"{path} is missing data variable {data_key}")
        levels = [int(v) for v in ds.pressure_level.values.tolist()]
        if level not in levels:
            raise KeyError(f"{path} is missing pressure level {level}")
        src_lat = np.asarray(ds.latitude.values, dtype=np.float32)
        arr = np.asarray(ds[data_key].isel(pressure_level=levels.index(level)).values, dtype=np.float32)
    arr = resample_lat(arr, src_lat, dst_lat)
    return to_hourly(arr, target_steps)


def load_variable(eda_root: Path, year: int, variable: str, dst_lat: np.ndarray, target_steps: int) -> np.ndarray:
    if variable in SINGLE_MAP:
        return load_single_variable(eda_root, year, variable, dst_lat, target_steps)
    return load_pressure_variable(eda_root, year, variable, dst_lat, target_steps)


def convert_year(
    eda_root: Path,
    source_partition_dir: Path,
    output_partition_dir: Path,
    year: int,
    variables: list[str],
    dst_lat: np.ndarray,
    overwrite: bool,
) -> int:
    source_files = sorted(source_partition_dir.glob(f"{year}_*.npy"))
    if not source_files:
        raise FileNotFoundError(f"No timestep source files found for {year} in {source_partition_dir}")
    output_partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_partition_dir / f"{year}.npy"
    if output_path.exists() and not overwrite:
        return 0

    target_steps = len(source_files)
    arrays = [load_variable(eda_root, year, name, dst_lat, target_steps) for name in variables]
    packed = np.stack(arrays, axis=1).astype(np.float16)
    np.save(output_path, packed)
    return target_steps


def copy_metadata(source_root: Path, output_root: Path, variables: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "variable_names.json").open("w", encoding="utf-8") as handle:
        json.dump(variables, handle, indent=2)
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "EDA ensemble spread",
                "variables": variables,
                "dtype": "float16",
                "layout": "partition/year.npy with shape (T,C,H,W)",
            },
            handle,
            indent=2,
        )
    for name in ("lat.npy", "lon.npy"):
        src = source_root / name
        dst = output_root / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variables", nargs="+", default=DEFAULT_UQ_VARIABLES)
    parser.add_argument("--train-years", nargs="+", type=int, default=[2010, 2011, 2012, 2013, 2014])
    parser.add_argument("--val-years", nargs="+", type=int, default=[2015])
    parser.add_argument("--test-years", nargs="+", type=int, default=[2016])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dst_lat = np.load(args.source_root / "lat.npy").astype(np.float32)
    copy_metadata(args.source_root, args.output_root, list(args.variables))
    specs = [("train", args.train_years), ("val", args.val_years), ("test", args.test_years)]
    for partition, years in specs:
        for year in years:
            print(f"[{partition}] converting UQ {year}", flush=True)
            written = convert_year(
                args.eda_root,
                args.source_root / partition,
                args.output_root / partition,
                year,
                list(args.variables),
                dst_lat,
                args.overwrite,
            )
            print(f"[{partition}] {year}: wrote {written} files", flush=True)


if __name__ == "__main__":
    main()
