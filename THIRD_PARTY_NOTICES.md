# Third-party notices

This distribution uses CC BY-NC 4.0 for UtoMe code and configurations. The paper
and figure assets are separate scholarly materials and retain their own terms.
The following upstream notices remain in effect for the corresponding code.

## Token Merging (ToMe)

`src/utome/models/tome.py` adapts the matching, merge/unmerge and schedule utilities
from [ToMe](https://github.com/facebookresearch/ToMe).

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Licensed under CC BY-NC 4.0; see [LICENSE](LICENSE).
UtoMe modifications add uncertainty-based matching penalties, protected anchors,
random matching and token-reduction schedules.

## MAE positional embeddings

`src/utome/models/pos_embed.py` adapts the sinusoidal embedding utilities from
[MAE](https://github.com/facebookresearch/mae), also used by ClimaX.

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Licensed under CC BY-NC 4.0; see [LICENSE](LICENSE).
UtoMe modifications support rectangular grids and updated NumPy types.

## ClimaX

The weather-transformer backbone, forecasting data pipeline, and latitude-weighted
metrics build on [ClimaX](https://github.com/microsoft/ClimaX).
Its MIT notice is retained in [licenses/ClimaX-MIT.txt](licenses/ClimaX-MIT.txt).
UtoMe adds token merging, uncertainty inputs, an importance predictor and timestep
data loading.

## Earlier UtoMe distribution

The earlier MIT notice, Copyright (c) 2026 Yunbei Zhang, is retained in
[licenses/UtoMe-legacy-MIT.txt](licenses/UtoMe-legacy-MIT.txt). This release does not
revoke permissions already granted for that earlier material or change the
licenses of third-party components.
