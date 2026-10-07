# UtoMe: Observation-Uncertainty-Guided Token Merging for Weather Foundation Models

[![Project Page](https://img.shields.io/badge/Project-Page-111827?style=flat-square)](https://ornl-utome.github.io)

**Code will be released soon.**

**Paper will be released soon.**

## Abstract

Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does not account for this heterogeneous input uncertainty. We introduce **UtoMe**, an observation-uncertainty-guided token merging framework that combines dynamics saliency with ensemble data assimilation spread as a relative input-reliability proxy. The resulting importance score protects selected tokens as anchors and calibrates bipartite matching between the remaining tokens. Three variants use external uncertainty at both training and inference, during training only, or neither stage. In ERA5 72-hour forecasting at two spatial resolutions, UtoMe improves Overall error over train-time ToMe at matched token budgets. At 1.40625° and 50% token reduction, UtoMe-E lowers Overall error from 67.32 to 53.00 at the same 0.55× encoder FLOPs. Small reductions remain close to the full-token baseline; larger reductions incur measurable accuracy losses. Component ablations and spatial uncertainty controls support combining dynamics with aligned uncertainty when allocating a weather transformer's token budget.
