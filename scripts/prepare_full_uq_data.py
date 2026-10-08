"""Convert downloaded EDA spread NetCDF files to UtoMe UQ npz shards.

The output mirrors the ClimaX-style ERA5 layout:

  <out>/<partition>/<year>_<shard>.npz

For each configured input variable, the output contains one UQ array with the
same key. Static fields do not have EDA spread, so they are emitted as zeros.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


LEVELS = [50, 250, 500, 600, 700, 850, 925]

DEFAULT_INPUT_VARIABLES = [
    "land_sea_mask",
    "orography",
    "lattitude",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential_50",
    "geopotential_250",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "u_component_of_wind_50",
    "u_component_of_wind_250",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "v_component_of_wind_50",
    "v_component_of_wind_250",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "temperature_50",
    "temperature_250",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "specific_humidity_50",
    "specific_humidity_250",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
]

SINGLE_MAP = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "total_precipitation": "tp",
    "sea_surface_temperature": "sst",
}

PRESSURE_MAP = {
    "geopotential": ("geopotential", "z"),
    "temperature": ("temperature", "t"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "specific_humidity": ("specific_humidity", "q"),
    "relative_humidity": ("relative_humidity", "r"),
}

STATIC_VARIABLES = {"land_sea_mask", "orography", "lattitude"}


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


def load_single(path: Path, dst_lat: np.ndarray, target_steps: int) -> dict[str, np.ndarray]:
    out = {}
    with xr.open_dataset(path) as ds:
        src_lat = np.asarray(ds.latitude.values, dtype=np.float32)
        for dst_name, src_name in SINGLE_MAP.items():
            if src_name not in ds:
                continue
            arr = np.asarray(ds[src_name].values, dtype=np.float32)
            arr = resample_lat(arr, src_lat, dst_lat)
            out[dst_name] = to_hourly(arr, target_steps)[:, None]
    return out


def load_pressure(root: Path, year: int, dst_lat: np.ndarray, target_steps: int) -> dict[str, np.ndarray]:
    out = {}
    for prefix, (file_key, data_key) in PRESSURE_MAP.items():
        path = root / "pressure_level" / f"eda_plev_{file_key}_{year}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            src_lat = np.asarray(ds.latitude.values, dtype=np.float32)
            available_levels = [int(v) for v in ds.pressure_level.values.tolist()]
            for level in LEVELS:
                if level not in available_levels:
                    raise KeyError(f"{path} is missing pressure level {level}")
                level_idx = available_levels.index(level)
                arr = np.asarray(ds[data_key].isel(pressure_level=level_idx).values, dtype=np.float32)
                arr = resample_lat(arr, src_lat, dst_lat)
                out[f"{prefix}_{level}"] = to_hourly(arr, target_steps)[:, None]
    return out


def load_year(eda_root: Path, year: int, dst_lat: np.ndarray, target_steps: int, variables: list[str]) -> dict[str, np.ndarray]:
    single = load_single(eda_root / "single_level" / f"eda_single_{year}.nc", dst_lat, target_steps)
    pressure = load_pressure(eda_root, year, dst_lat, target_steps)
    source = {**single, **pressure}

    zero = np.zeros((target_steps, 1, dst_lat.shape[0], 64), dtype=np.float32)
    out = {}
    for name in variables:
        if name in STATIC_VARIABLES:
            out[name] = zero
        elif name in source:
            out[name] = source[name]
        else:
            raise KeyError(f"No EDA spread source for requested variable: {name}")
    return out


def write_shards(year_data: dict[str, np.ndarray], out_dir: Path, year: int, steps_per_shard: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = next(iter(year_data.values())).shape[0]
    for shard_idx in range(total // steps_per_shard):
        start = shard_idx * steps_per_shard
        end = start + steps_per_shard
        np.savez(out_dir / f"{year}_{shard_idx}.npz", **{k: v[start:end] for k, v in year_data.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda-root", type=Path, required=True)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-years", nargs="*", type=int, default=[2010, 2011, 2012, 2013, 2014])
    parser.add_argument("--val-years", nargs="*", type=int, default=[2015])
    parser.add_argument("--test-years", nargs="*", type=int, default=[2016])
    parser.add_argument("--steps-per-year", type=int, default=8760)
    parser.add_argument("--steps-per-shard", type=int, default=1095)
    args = parser.parse_args()

    dst_lat = np.load(args.era5_root / "lat.npy").astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)

    for aux in ("lat.npy", "lon.npy"):
        src = args.era5_root / aux
        dst = args.out / aux
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    specs = [(args.train_years, "train"), (args.val_years, "val"), (args.test_years, "test")]
    for years, partition in specs:
        for year in years:
            print(f"[{partition}] processing {year}")
            year_data = load_year(args.eda_root, year, dst_lat, args.steps_per_year, DEFAULT_INPUT_VARIABLES)
            write_shards(year_data, args.out / partition, year, args.steps_per_shard)
            print(f"  wrote {len(year_data)} variables")


if __name__ == "__main__":
    main()
