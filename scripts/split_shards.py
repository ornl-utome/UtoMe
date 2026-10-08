"""Split each ERA5/UQ shard into K sub-shards (K=8 by default).

This produces enough shards (e.g. train: 40 -> 320, val: 8 -> 64) that file-level
DDP sharding is balanced for any reasonable GPU count up to 64.

Outputs to a new directory tree mirroring the source layout:
  <out>/{train,val,test}/<year>_<orig_shard>_<sub_shard>.npz
Plus symlinks for lat.npy, lon.npy, normalize_*.npz, and climatology.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def split_one(src: Path, dst_dir: Path, k: int) -> None:
    data = np.load(src)
    keys = list(data.keys())
    arrs = {key: np.asarray(data[key]) for key in keys}
    n_steps = arrs[keys[0]].shape[0]
    chunk = n_steps // k
    extras = n_steps - chunk * k

    pos = 0
    for sub in range(k):
        size = chunk + (1 if sub < extras else 0)
        end = pos + size
        out_path = dst_dir / f"{src.stem}_{sub}.npz"
        if out_path.exists():
            pos = end
            continue
        payload = {key: arr[pos:end] for key, arr in arrs.items()}
        np.savez(out_path, **payload)
        pos = end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--partitions", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--copy-aux", action="store_true", help="symlink aux files (lat/lon/norm/climatology)")
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    if args.copy_aux:
        for aux in ("lat.npy", "lon.npy", "normalize_mean.npz", "normalize_std.npz"):
            src = args.src / aux
            dst = args.dst / aux
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

    for part in args.partitions:
        in_dir = args.src / part
        out_dir = args.dst / part
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.copy_aux:
            clim = in_dir / "climatology.npz"
            if clim.exists():
                target = out_dir / "climatology.npz"
                if not target.exists():
                    target.symlink_to(clim.resolve())
        files = sorted(p for p in in_dir.glob("*.npz") if "climatology" not in p.name)
        for idx, src_file in enumerate(files):
            print(f"[{part}] {idx + 1}/{len(files)}: {src_file.name}")
            split_one(src_file, out_dir, args.k)


if __name__ == "__main__":
    main()
