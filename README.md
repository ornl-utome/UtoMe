<h1 align="center">UtoMe: Observation-Uncertainty-Guided Token Merging<br>for Weather Foundation Models</h1>

<p align="center">
  Yunbei Zhang<sup>1,*,†</sup> · Janet Wang<sup>1,*</sup> · Xi Xiao<sup>2</sup> · Jihun Hamm<sup>1</sup> · Xiao Wang<sup>3,†</sup>
</p>

<p align="center">
  <sup>1</sup>Tulane University &nbsp; <sup>2</sup>University of Alabama at Birmingham &nbsp; <sup>3</sup>Oak Ridge National Laboratory
</p>

<p align="center">
  <sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding authors
</p>

<p align="center">
  <a href="https://ornl-utome.github.io"><img src="https://img.shields.io/badge/Project-Page-00662C?style=flat-square" alt="Project page"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2026-84B641?style=flat-square" alt="NeurIPS 2026">
  <img src="https://img.shields.io/badge/Paper-coming_soon-5B6A62?style=flat-square" alt="Paper coming soon">
  <img src="https://img.shields.io/badge/Code-coming_soon-5B6A62?style=flat-square" alt="Code coming soon">
</p>

> This project was carried out during **Yunbei Zhang**'s internship at **Oak Ridge National Laboratory (ORNL)**.

**Code will be released soon.**

**Paper will be released soon.**

## Abstract

Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does not account for this heterogeneous input uncertainty. We introduce **UtoMe**, an observation-uncertainty-guided token merging framework that combines dynamics saliency with ensemble data assimilation spread as a relative input-reliability proxy. The resulting importance score protects selected tokens as anchors and calibrates bipartite matching between the remaining tokens. Three variants use external uncertainty at both training and inference, during training only, or neither stage. In ERA5 72-hour forecasting at two spatial resolutions, UtoMe improves Overall error over train-time ToMe at matched token budgets. At 1.40625° and 50% token reduction, UtoMe-E lowers Overall error from 67.32 to 53.00 at the same 0.55× encoder FLOPs. Small reductions remain close to the full-token baseline; larger reductions incur measurable accuracy losses. Component ablations and spatial uncertainty controls support combining dynamics with aligned uncertainty when allocating a weather transformer's token budget.

## Method

UtoMe uses observation uncertainty to guide training-time token merging for transformer-based weather forecasting. It turns weather-state saliency and low observation uncertainty into a patch-level token score, then uses that score to protect forecast-sensitive anchors and merge lower-importance redundant tokens.

![UtoMe overview](assets/overview.png)

| Variant | External uncertainty |
|---|---|
| **UtoMe-E** | Used at both training and inference |
| **UtoMe-P** | Used during training only, through a learned importance predictor |
| **UtoMe-S** | Not used; learns from a local spatial-inconsistency proxy |

## Results

Main results on 1.40625° ERA5 72-hour forecasting. Values are mean ± standard deviation over three seeds. Lower is better for Overall and per-variable wRMSE; higher is better for ACC. Compute is the encoder FLOPs ratio relative to no merging. Bold and underlined values are best and second-best within each budget.

| Method | Overall ↓ | ACC ↑ | Z500 ↓ | T850 ↓ | T2m ↓ | U10 ↓ | V10 ↓ | Compute ↓ |
|---|---|---|---|---|---|---|---|---|
| No merging | 32.41 ±0.28 | 0.964 ±0.004 | 156.76 ±1.34 | 1.299 ±0.007 | 1.269 ±0.006 | 1.339 ±0.008 | 1.360 ±0.009 | 1.00× |
| *25% token reduction* | | | | | | | | |
| ToMe | 42.45 ±0.24 | 0.949 ±0.003 | 206.02 ±1.17 | 1.452 ±0.006 | 1.483 ±0.007 | 1.572 ±0.008 | 1.724 ±0.009 | 0.86× |
| UtoMe-E | **39.59** ±0.21 | **0.952** ±0.003 | **191.93** ±1.05 | **1.413** ±0.005 | **1.454** ±0.006 | **1.527** ±0.007 | **1.645** ±0.008 | 0.86× |
| UtoMe-P | <u>39.60</u> ±0.01 | **0.952** ±0.000 | <u>191.98</u> ±0.03 | 1.425 ±0.001 | 1.459 ±0.000 | <u>1.530</u> ±0.000 | <u>1.648</u> ±0.001 | 0.87× |
| UtoMe-S | 39.84 ±0.27 | <u>0.951</u> ±0.004 | 193.14 ±1.31 | <u>1.416</u> ±0.007 | <u>1.458</u> ±0.008 | 1.533 ±0.009 | 1.656 ±0.010 | 0.87× |
| *50% token reduction* | | | | | | | | |
| ToMe | 67.32 ±0.43 | 0.905 ±0.005 | 328.21 ±2.11 | 1.846 ±0.011 | 2.056 ±0.015 | 2.136 ±0.014 | <u>2.347</u> ±0.018 | 0.55× |
| UtoMe-E | **53.00** ±0.36 | **0.913** ±0.004 | **256.72** ±1.76 | <u>1.759</u> ±0.009 | <u>2.051</u> ±0.012 | <u>2.038</u> ±0.013 | 2.433 ±0.015 | 0.55× |
| UtoMe-P | <u>53.14</u> ±0.01 | 0.907 ±0.000 | <u>257.37</u> ±0.05 | 1.762 ±0.001 | 2.056 ±0.001 | 2.045 ±0.001 | 2.444 ±0.001 | 0.57× |
| UtoMe-S | 57.01 ±0.04 | <u>0.908</u> ±0.000 | 276.99 ±0.18 | **1.752** ±0.001 | **1.968** ±0.004 | **2.009** ±0.002 | **2.335** ±0.002 | 0.57× |

## Case studies

![5.625° case summary](assets/pipeline_5p625.png)

*5.625° case summary: weather state, observation uncertainty, dynamics saliency, importance score, patch-level token score, and the final anchor or merge-candidate overlay.*

![1.40625° high-resolution case study](assets/case_1p406.png)

*1.40625° high-resolution case study: Z500 input, observation uncertainty, importance score, and patch-level token score.*

## Acknowledgements

This research was supported by ORNL's AI Initiative, sponsored by the Director's Research and Development Program at ORNL. Computing time was provided through the Oak Ridge Leadership Computing Facility. The manuscript was authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of Energy (DOE).

## Citation

```bibtex
@inproceedings{zhang2026utome,
  title     = {UtoMe: Observation-Uncertainty-Guided Token Merging for Weather Foundation Models},
  author    = {Zhang, Yunbei and Wang, Janet and Xiao, Xi and Hamm, Jihun and Wang, Xiao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
