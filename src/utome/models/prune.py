"""Matched-budget token pruning baselines for the NeurIPS 2026 rebuttal.

Reviewer 4Bny asked for token-pruning baselines at the same token budget as
UtoMe/ToMe so that the gain can be attributed to uncertainty-aware *merging*
rather than to generic importance ranking or to a shorter sequence.

Implementation. At every layer where the ToMe schedule removes ``r`` tokens, the
pruning baseline removes the ``r`` lowest-scoring tokens from the sequence and
freezes their hidden states. The frozen states bypass all remaining transformer
blocks and are written back into their original positions just before the final
norm and prediction head. Consequently

* the per-layer sequence lengths, and therefore the encoder FLOPs, are exactly
  the same as in the corresponding merging run, and
* the model still produces a dense forecast for every patch, so wRMSE/ACC stay
  comparable, with pruned patches predicted from features that stopped being
  refined at the pruning layer.

Zero-filling pruned tokens would have made the baseline trivially bad, so this
"freeze and reinsert" variant is deliberately the strongest reasonable pruning
baseline at matched compute.
"""

from __future__ import annotations

import torch


def init_prune_state(state: dict, batch_size: int, num_tokens: int, channels: int, device, dtype) -> None:
    state["orig_index"] = torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, num_tokens).contiguous()
    state["frozen"] = torch.zeros(batch_size, num_tokens, channels, device=device, dtype=dtype)


def prune_tokens(x: torch.Tensor, r: int, state: dict) -> torch.Tensor:
    """Drop the ``r`` lowest-importance tokens and freeze their hidden states."""
    batch, num_tokens, channels = x.shape
    r = min(int(r), num_tokens - 1)
    if r <= 0:
        return x

    importance = state.get("importance")
    if importance is None:
        importance = torch.rand(batch, num_tokens, device=x.device, dtype=torch.float32)

    protect_mask = state.get("protect_mask")
    if protect_mask is not None:
        # protected tokens are never pruned
        importance = importance.masked_fill(protect_mask.bool(), float("inf"))

    order = importance.argsort(dim=-1, descending=True)
    keep = order[..., : num_tokens - r].sort(dim=-1).values
    drop = order[..., num_tokens - r :]

    keep_exp = keep.unsqueeze(-1).expand(batch, keep.shape[1], channels)
    drop_exp = drop.unsqueeze(-1).expand(batch, r, channels)

    drop_orig = state["orig_index"].gather(1, drop)
    state["frozen"] = state["frozen"].scatter(
        1,
        drop_orig.unsqueeze(-1).expand(batch, r, channels),
        x.gather(1, drop_exp).to(state["frozen"].dtype),
    )

    state["orig_index"] = state["orig_index"].gather(1, keep)
    if state.get("importance") is not None:
        state["importance"] = state["importance"].gather(1, keep)
    if protect_mask is not None:
        state["protect_mask"] = protect_mask.gather(1, keep)
    if state.get("size") is not None:
        state["size"] = state["size"].gather(1, keep.unsqueeze(-1).expand(batch, keep.shape[1], 1))

    return x.gather(1, keep_exp)


def restore_pruned(x: torch.Tensor, state: dict) -> torch.Tensor:
    """Write the surviving tokens back into the full-length frozen buffer."""
    batch, num_tokens, channels = x.shape
    frozen = state["frozen"].to(x.dtype)
    index = state["orig_index"].unsqueeze(-1).expand(batch, num_tokens, channels)
    return frozen.scatter(1, index, x)
