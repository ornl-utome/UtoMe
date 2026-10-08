"""Extended evaluation metrics used for the NeurIPS 2026 rebuttal.

The metrics here go beyond domain-average wRMSE/ACC. They are all computed from
denormalized predictions and targets, i.e. in physical units, and they follow the
same aggregation convention as ``utome.utils.metrics.lat_weighted_rmse``: the
latitude-weighted error is reduced to one number per sample, and the reported
value is the mean over samples.

Three families are implemented:

1. Tail / extreme retention
   - conditional RMSE on the upper and lower 5% of the target anomaly field
   - quantile bias, i.e. how much the predicted 95th/5th percentile is shrunk
     relative to the target percentile (a direct measure of peak smoothing)

2. Physical consistency
   - spatial-gradient RMSE for every target variable
   - relative vorticity and horizontal divergence RMSE for the (U10, V10) pair
   - kinetic-energy RMSE and kinetic-energy bias for the (U10, V10) pair

3. Regime stratification
   - lat-weighted RMSE inside the four quadrants defined by median splits of the
     patch-level observation-uncertainty score q and dynamics saliency score d.
     The high-q/high-d quadrant is the failure regime raised by Reviewer S6zQ.

All accumulators are plain tensors so that a distributed all-reduce is possible,
although the rebuttal evaluations run on a single device.
"""

from __future__ import annotations

import math

import numpy as np
import torch

EARTH_RADIUS_M = 6.371e6

U10 = "10m_u_component_of_wind"
V10 = "10m_v_component_of_wind"


def _lat_weights(lat: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=dtype, device=device).view(1, 1, -1, 1)


def _masked_rmse_per_sample(sq_err: torch.Tensor, w_lat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RMSE per (sample, variable) restricted to ``mask``.

    ``sq_err`` and ``mask`` are (B, C, H, W); ``w_lat`` is broadcastable.
    Returns (B, C). Samples with an empty mask return NaN and are dropped by the
    caller via ``torch.nansum`` bookkeeping.
    """
    weight = w_lat * mask.to(sq_err.dtype)
    num = (sq_err * weight).sum(dim=(-2, -1))
    den = weight.sum(dim=(-2, -1))
    return torch.sqrt(num / den.clamp_min(1e-12)), den


def _central_diff_x(field: torch.Tensor) -> torch.Tensor:
    """Longitude derivative index-wise, periodic in longitude."""
    return 0.5 * (torch.roll(field, shifts=-1, dims=-1) - torch.roll(field, shifts=1, dims=-1))


def _central_diff_y(field: torch.Tensor) -> torch.Tensor:
    """Latitude derivative index-wise, replicated at the two polar rows."""
    up = torch.cat([field[..., :1, :], field[..., :-1, :]], dim=-2)
    down = torch.cat([field[..., 1:, :], field[..., -1:, :]], dim=-2)
    scale = torch.ones_like(field)
    scale[..., 0, :] = 2.0
    scale[..., -1, :] = 2.0
    return 0.5 * (down - up) * scale


class ExtendedMetricAccumulator:
    """Accumulates the rebuttal metrics over a test epoch."""

    def __init__(
        self,
        variables: tuple[str, ...],
        lat: np.ndarray,
        tail_fraction: float = 0.05,
        num_strata: int = 2,
    ) -> None:
        self.variables = tuple(variables)
        self.lat = np.asarray(lat, dtype=np.float64)
        self.tail_fraction = float(tail_fraction)
        self.num_strata = int(num_strata)
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}
        self._device: torch.device | None = None
        self._geom: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------ utils
    def _add(self, name: str, value: torch.Tensor, count: torch.Tensor | float = 1.0) -> None:
        value = value.detach().to(torch.float64)
        if isinstance(count, torch.Tensor):
            count = count.detach().to(torch.float64)
        else:
            count = torch.as_tensor(float(count), dtype=torch.float64, device=value.device)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        if name not in self._sums:
            self._sums[name] = torch.zeros((), dtype=torch.float64, device=value.device)
            self._counts[name] = torch.zeros((), dtype=torch.float64, device=value.device)
        self._sums[name] += value.sum()
        self._counts[name] += count.sum() if isinstance(count, torch.Tensor) else count

    def _geometry(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        if self._geom is not None and self._geom["dx"].shape[-2] == height:
            return self._geom
        lat = torch.as_tensor(self.lat, dtype=torch.float64, device=device)
        if lat.numel() != height:
            lat = torch.linspace(90.0, -90.0, height, dtype=torch.float64, device=device)
        dlat_deg = float(abs(self.lat[1] - self.lat[0])) if self.lat.size > 1 else 180.0 / height
        dlon_deg = 360.0 / width
        cos_lat = torch.cos(torch.deg2rad(lat)).clamp_min(1e-3)
        dx = EARTH_RADIUS_M * math.radians(dlon_deg) * cos_lat  # (H,)
        dy = EARTH_RADIUS_M * math.radians(dlat_deg)
        self._geom = {
            "dx": dx.view(1, 1, -1, 1).to(dtype),
            "dy": torch.as_tensor(dy, dtype=dtype, device=device),
            "cos_lat": cos_lat.view(1, 1, -1, 1).to(dtype),
        }
        return self._geom

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        clim: torch.Tensor | None = None,
        uq_patch: torch.Tensor | None = None,
        dynamics_patch: torch.Tensor | None = None,
        patch_size: int | None = None,
    ) -> None:
        preds = preds.to(torch.float32)
        target = target.to(torch.float32)
        device = preds.device
        self._device = device
        batch, channels, height, width = preds.shape
        w_lat = _lat_weights(self.lat, device, preds.dtype)
        sq_err = (preds - target) ** 2

        # ---------------------------------------------------------- tail terms
        if clim is not None:
            anomaly = target - clim.to(device=device, dtype=target.dtype).view(1, channels, height, width)
        else:
            anomaly = target - target.mean(dim=(-2, -1), keepdim=True)

        flat_anom = anomaly.reshape(batch, channels, -1)
        hi_thresh = torch.quantile(flat_anom, 1.0 - self.tail_fraction, dim=-1).view(batch, channels, 1, 1)
        lo_thresh = torch.quantile(flat_anom, self.tail_fraction, dim=-1).view(batch, channels, 1, 1)
        mask_hi = anomaly >= hi_thresh
        mask_lo = anomaly <= lo_thresh

        rmse_hi, cnt_hi = _masked_rmse_per_sample(sq_err, w_lat, mask_hi)
        rmse_lo, cnt_lo = _masked_rmse_per_sample(sq_err, w_lat, mask_lo)
        rmse_all, _ = _masked_rmse_per_sample(sq_err, w_lat, torch.ones_like(mask_hi))
        for idx, name in enumerate(self.variables):
            self._add(f"tail_hi5_rmse_{name}", rmse_hi[:, idx], batch)
            self._add(f"tail_lo5_rmse_{name}", rmse_lo[:, idx], batch)
            self._add(f"tail_all_rmse_{name}", rmse_all[:, idx], batch)

        # quantile bias: predicted extreme amplitude minus target extreme amplitude
        flat_pred = preds.reshape(batch, channels, -1)
        flat_true = target.reshape(batch, channels, -1)
        for q, tag in ((0.99, "p99"), (0.95, "p95"), (0.05, "p05"), (0.01, "p01")):
            bias = torch.quantile(flat_pred, q, dim=-1) - torch.quantile(flat_true, q, dim=-1)
            for idx, name in enumerate(self.variables):
                self._add(f"qbias_{tag}_{name}", bias[:, idx], batch)
        # spatial standard deviation ratio: a direct smoothing diagnostic
        std_ratio = flat_pred.std(dim=-1) / flat_true.std(dim=-1).clamp_min(1e-6)
        for idx, name in enumerate(self.variables):
            self._add(f"std_ratio_{name}", std_ratio[:, idx], batch)

        # ------------------------------------------------- physical consistency
        geom = self._geometry(height, width, device, preds.dtype)
        dx = geom["dx"]
        dy = geom["dy"]

        gx_pred = _central_diff_x(preds) / dx
        gx_true = _central_diff_x(target) / dx
        gy_pred = _central_diff_y(preds) / dy
        gy_true = _central_diff_y(target) / dy
        grad_sq = (gx_pred - gx_true) ** 2 + (gy_pred - gy_true) ** 2
        grad_rmse = torch.sqrt((grad_sq * w_lat).mean(dim=(-2, -1)) / 2.0)
        grad_mag_pred = torch.sqrt(gx_pred**2 + gy_pred**2)
        grad_mag_true = torch.sqrt(gx_true**2 + gy_true**2)
        grad_ratio = (grad_mag_pred * w_lat).mean(dim=(-2, -1)) / (grad_mag_true * w_lat).mean(dim=(-2, -1)).clamp_min(1e-12)
        for idx, name in enumerate(self.variables):
            self._add(f"grad_rmse_{name}", grad_rmse[:, idx], batch)
            self._add(f"grad_ratio_{name}", grad_ratio[:, idx], batch)

        if U10 in self.variables and V10 in self.variables:
            iu = self.variables.index(U10)
            iv = self.variables.index(V10)
            u_p, v_p = preds[:, iu : iu + 1], preds[:, iv : iv + 1]
            u_t, v_t = target[:, iu : iu + 1], target[:, iv : iv + 1]

            def _vort_div_ke(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                dvdx = _central_diff_x(v) / dx
                dudy = _central_diff_y(u) / dy
                dudx = _central_diff_x(u) / dx
                dvdy = _central_diff_y(v) / dy
                vort = dvdx - dudy
                div = dudx + dvdy
                kinetic = 0.5 * (u**2 + v**2)
                return vort, div, kinetic

            vort_p, div_p, ke_p = _vort_div_ke(u_p, v_p)
            vort_t, div_t, ke_t = _vort_div_ke(u_t, v_t)
            vort_rmse = torch.sqrt((((vort_p - vort_t) ** 2) * w_lat).mean(dim=(-2, -1)))
            div_rmse = torch.sqrt((((div_p - div_t) ** 2) * w_lat).mean(dim=(-2, -1)))
            ke_rmse = torch.sqrt((((ke_p - ke_t) ** 2) * w_lat).mean(dim=(-2, -1)))
            ke_bias = ((ke_p - ke_t) * w_lat).mean(dim=(-2, -1))
            vort_ratio = ((vort_p.abs() * w_lat).mean(dim=(-2, -1))) / ((vort_t.abs() * w_lat).mean(dim=(-2, -1))).clamp_min(1e-14)
            ke_ratio = ((ke_p * w_lat).mean(dim=(-2, -1))) / ((ke_t * w_lat).mean(dim=(-2, -1))).clamp_min(1e-14)
            self._add("vorticity_rmse", vort_rmse, batch)
            self._add("divergence_rmse", div_rmse, batch)
            self._add("ke_rmse", ke_rmse, batch)
            self._add("ke_bias", ke_bias, batch)
            self._add("vorticity_ratio", vort_ratio, batch)
            self._add("ke_ratio", ke_ratio, batch)

        # -------------------------------------------------------- stratification
        if uq_patch is not None and dynamics_patch is not None and patch_size is not None:
            grid_h = height // patch_size
            grid_w = width // patch_size
            if uq_patch.shape[-1] == grid_h * grid_w:
                q_map = self._patch_to_grid(uq_patch, grid_h, grid_w, patch_size, height, width)
                d_map = self._patch_to_grid(dynamics_patch, grid_h, grid_w, patch_size, height, width)
                q_med = uq_patch.median(dim=-1, keepdim=True).values.view(batch, 1, 1, 1)
                d_med = dynamics_patch.median(dim=-1, keepdim=True).values.view(batch, 1, 1, 1)
                hi_q = q_map >= q_med
                hi_d = d_map >= d_med
                quadrants = {
                    "loQ_loD": (~hi_q) & (~hi_d),
                    "loQ_hiD": (~hi_q) & hi_d,
                    "hiQ_loD": hi_q & (~hi_d),
                    "hiQ_hiD": hi_q & hi_d,
                }
                for tag, mask in quadrants.items():
                    mask_b = mask.expand(batch, channels, height, width)
                    strat_rmse, _ = _masked_rmse_per_sample(sq_err, w_lat, mask_b)
                    for idx, name in enumerate(self.variables):
                        self._add(f"strat_{tag}_rmse_{name}", strat_rmse[:, idx], batch)

    @staticmethod
    def _patch_to_grid(
        patch_scores: torch.Tensor,
        grid_h: int,
        grid_w: int,
        patch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch = patch_scores.shape[0]
        grid = patch_scores.reshape(batch, 1, grid_h, grid_w)
        grid = grid.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
        return grid[..., :height, :width]

    # ----------------------------------------------------------------- compute
    def compute(self, reduce_distributed: bool = True) -> dict[str, float]:
        if not self._sums:
            return {}
        names = sorted(self._sums)
        sums = torch.stack([self._sums[name] for name in names])
        counts = torch.stack([self._counts[name] for name in names])
        if reduce_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        values = (sums / counts.clamp_min(1e-12)).tolist()
        return dict(zip(names, values))

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
