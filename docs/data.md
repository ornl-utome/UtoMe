# Data setup

Set the ERA5 and EDA paths in the configuration before running. Weather and uncertainty fields must use matching times, grids, and variable names.

- **ERA5 NPZ:** `train/`, `val/`, and `test/` folders containing shards keyed by variable name; each field has shape `(time, 1, height, width)`.
- **Statistics:** `normalize_mean.npz`, `normalize_std.npz`, `lat.npy`, and `lon.npy`; `val/climatology.npz` and `test/climatology.npz` for ACC.
- **EDA UQ:** corresponding partitions and shards containing the variables listed in `data.uq_variables`. UtoMe-S uses an input-derived proxy and does not require EDA files.

For timestep NPY, set `data.format: timestep`: store `(channels, height, width)` arrays under each split as `year_shard_step.npy`, with channel order in `variable_names.json`. Use `data.pre_normalized: true` when the arrays are already normalized (the ERA5 converter does this by default). Conversion helpers are in `scripts/`; install their optional dependencies with `pip install -e ".[data]"` and use `--help` for the required input paths.

For a local run, set `trainer.devices`, `trainer.num_nodes`, and `data.batch_size` to suit your machine. Use `trainer.max_steps` to set a fixed training duration; `optimization.max_steps` controls the learning-rate schedule.
