#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected (T,1,H,W) or (T,H,W), got {array.shape}")


def load_stats(input_dir: Path, variables: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mean_path = input_dir / "normalize_mean.npz"
    std_path = input_dir / "normalize_std.npz"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Missing normalization stats under {input_dir}")
    with np.load(mean_path) as mean_npz, np.load(std_path) as std_npz:
        mean = {name: mean_npz[name].astype(np.float32).reshape(-1, 1, 1) for name in variables}
        std = {name: std_npz[name].astype(np.float32).reshape(-1, 1, 1) for name in variables}
    return mean, std


def convert_one(args: tuple[str, str, list[str], bool, bool, dict[str, np.ndarray], dict[str, np.ndarray]]) -> tuple[str, int]:
    input_path_s, output_dir_s, variables, overwrite, do_normalize, mean, std = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    prefix = input_path.stem

    with np.load(input_path) as npz:
        missing = [name for name in variables if name not in npz]
        if missing:
            raise KeyError(f"{input_path} is missing variables: {missing}")

        arrays = []
        for name in variables:
            array = normalize_array(npz[name]).astype(np.float32)
            if do_normalize:
                array = (array - mean[name]) / np.maximum(std[name], 1e-6)
            arrays.append(array)
        n_steps = arrays[0].shape[0]
        if any(array.shape[0] != n_steps for array in arrays):
            shapes = {name: array.shape for name, array in zip(variables, arrays)}
            raise ValueError(f"Inconsistent time shapes in {input_path}: {shapes}")

        for step in range(n_steps):
            output_path = output_dir / f"{prefix}_{step:04d}.npy"
            if output_path.exists() and not overwrite:
                continue
            frame = np.stack([array[step] for array in arrays], axis=0).astype(np.float16)
            np.save(output_path, frame)

    return input_path.name, n_steps


def copy_metadata(input_dir: Path, output_dir: Path) -> None:
    for name in ("lat.npy", "lon.npy", "normalize_mean.npz", "normalize_std.npz"):
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    for split in ("train", "val", "test"):
        src = input_dir / split / "climatology.npz"
        if src.exists():
            dst_dir = output_dir / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / "climatology.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ClimaX-style npz shards to per-timestep npy files.")
    parser.add_argument("--config", type=Path, required=True, help="Training config whose data.variables define channel order.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Source npz root. Defaults to config data.root_dir.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Converted timestep data root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Store raw values. Not recommended with float16 because geopotential can overflow.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    input_dir = args.input_dir or Path(data_cfg["root_dir"])
    variables = list(dict.fromkeys(list(data_cfg["variables"]) + list(data_cfg.get("out_variables") or [])))
    do_normalize = not args.no_normalize
    mean, std = load_stats(input_dir, variables) if do_normalize else ({}, {})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "variable_names.json").open("w", encoding="utf-8") as handle:
        json.dump(variables, handle, indent=2)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_dir": str(input_dir),
                "variables": variables,
                "dtype": "float16",
                "normalized": do_normalize,
                "layout": "partition/stem_step.npy with shape (C,H,W)",
            },
            handle,
            indent=2,
        )
    copy_metadata(input_dir, args.output_dir)

    started = time.time()
    for split in args.splits:
        input_split = input_dir / split
        output_split = args.output_dir / split
        output_split.mkdir(parents=True, exist_ok=True)
        files = sorted(path for path in input_split.glob("*.npz") if path.name != "climatology.npz")
        if not files:
            print(f"[convert] split={split}: no npz files at {input_split}", flush=True)
            continue

        print(f"[convert] split={split}: {len(files)} input files -> {output_split}", flush=True)
        work = [(str(path), str(output_split), variables, args.overwrite, do_normalize, mean, std) for path in files]
        total = 0
        with Pool(processes=args.num_workers) as pool:
            for filename, n_steps in pool.imap_unordered(convert_one, work):
                total += n_steps
                print(f"[convert] {split}/{filename}: {n_steps} timesteps", flush=True)
        print(f"[convert] split={split}: wrote/checked {total} timesteps", flush=True)

    elapsed = time.time() - started
    print(f"[convert] done in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[convert] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
