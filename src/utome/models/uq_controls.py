"""Causal controls on the external uncertainty field (Reviewer XWbY, question 2).

The controls below transform the EDA-spread tensor before it is turned into the
patch-level score ``q``. They are designed to separate three possible sources of
the reported UtoMe-E gain:

``none``            true EDA spread (reference).
``shuffle``         spatially shuffled spread: the marginal distribution of the
                    spread values is preserved exactly, but the spatial field is
                    destroyed. If the gain survives this, the gain does not come
                    from where the uncertainty actually is.
``roll``            the spread field is translated by a quarter of the domain in
                    both directions: coherent spatial structure is preserved but
                    is no longer aligned with the weather state.
``constant``        a spatially constant spread. The score collapses to the
                    dynamics term, so this isolates "generic importance ranking".
``rank_uniform``    each channel's spread values are replaced by their ranks
                    mapped to [0, 1]. The ordering is preserved exactly and all
                    magnitude information is destroyed, which tests whether the
                    policy is rank-driven or magnitude-driven.
``sqrt`` / ``square``
                    strictly monotone rescalings of the spread. Rank-preserving
                    like ``rank_uniform`` but with different magnitude spacing.
``noise``           a random field with the same per-channel mean and standard
                    deviation as the true spread.
``members:M``       a simulation of an EDA product estimated from ``M`` ensemble
                    members instead of the full ensemble. For Gaussian members,
                    the sample variance satisfies s_M^2 ~ sigma^2 chi^2_{M-1}/(M-1),
                    so the control multiplies the reference spread by
                    sqrt(chi^2_{M-1}/(M-1)) drawn independently per grid point.
                    This is a simulation of sampling noise, not a re-derivation
                    from raw ensemble members, and must be reported as such.
"""

from __future__ import annotations

import torch

CONTROL_NAMES = (
    "none",
    "shuffle",
    "roll",
    "constant",
    "rank_uniform",
    "sqrt",
    "square",
    "noise",
)


def _rank_uniform(x: torch.Tensor) -> torch.Tensor:
    batch, channels = x.shape[0], x.shape[1]
    flat = x.reshape(batch, channels, -1)
    order = flat.argsort(dim=-1)
    ranks = torch.empty_like(order)
    arange = torch.arange(flat.shape[-1], device=x.device).expand_as(order)
    ranks.scatter_(-1, order, arange)
    uniform = ranks.to(x.dtype) / max(1.0, float(flat.shape[-1] - 1))
    return uniform.view_as(x)


def apply_uq_control(
    uq: torch.Tensor | None,
    control: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Apply a causal control to the raw uncertainty tensor ``(B, C, H, W)``."""
    if uq is None or control in (None, "", "none"):
        return uq

    batch, channels, height, width = uq.shape
    device = uq.device

    if control == "shuffle":
        perm = torch.stack(
            [torch.randperm(height * width, device=device, generator=generator) for _ in range(batch)],
            dim=0,
        )
        flat = uq.reshape(batch, channels, -1)
        index = perm.unsqueeze(1).expand(batch, channels, height * width)
        return flat.gather(-1, index).view_as(uq)

    if control == "roll":
        return torch.roll(uq, shifts=(height // 4, width // 4), dims=(-2, -1))

    if control == "constant":
        return torch.ones_like(uq)

    if control == "rank_uniform":
        return _rank_uniform(uq)

    if control == "sqrt":
        return uq.abs().sqrt()

    if control == "square":
        return uq.abs().square()

    if control == "noise":
        flat = uq.reshape(batch, channels, -1)
        mean = flat.mean(dim=-1, keepdim=True)
        std = flat.std(dim=-1, keepdim=True)
        noise = torch.randn(flat.shape, device=device, dtype=uq.dtype, generator=generator)
        return (mean + std * noise).abs().view_as(uq)

    if control.startswith("members:"):
        members = int(control.split(":", 1)[1])
        if members < 2:
            raise ValueError("members:M requires M >= 2")
        dof = members - 1
        # chi^2_dof drawn as a sum of dof squared standard normals
        normals = torch.randn((dof,) + tuple(uq.shape), device=device, dtype=uq.dtype, generator=generator)
        chi2 = normals.square().sum(dim=0)
        return uq * (chi2 / dof).sqrt()

    raise ValueError(f"Unsupported UQ control: {control}")
