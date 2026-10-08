# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Adapted for UtoMe; see THIRD_PARTY_NOTICES.md. Licensed under CC BY-NC 4.0.

from typing import Callable, List, Tuple, Union

import torch


def do_nothing(x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
    return x


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    inflect = 0.0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    if isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / max(1, num_layers - 1)
    return [int(min_val + step * idx) for idx in range(num_layers)]


def ratio_to_decreasing_r(num_layers: int, num_tokens: int, ratio: float) -> List[int]:
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError("Token reduction ratio must be in [0, 1).")
    total_remove = int(round(num_tokens * ratio))
    if total_remove <= 0:
        return [0] * num_layers

    weights = torch.arange(num_layers - 1, -1, -1, dtype=torch.float64)
    if float(weights.sum()) == 0.0:
        return [total_remove]
    raw = total_remove * weights / weights.sum()
    schedule = torch.floor(raw).to(torch.int64)
    remainder = total_remove - int(schedule.sum().item())
    if remainder > 0:
        order = torch.argsort(raw - schedule.to(raw.dtype), descending=True)
        schedule[order[:remainder]] += 1
    return [int(x) for x in schedule.tolist()]


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    protect_mask: torch.Tensor | None = None,
    importance: torch.Tensor | None = None,
    importance_lambda: float = 0.0,
    importance_eta: float = 0.0,
) -> tuple[Callable, Callable, int]:
    protected = int(class_token) + int(distill_token)
    total_tokens = metric.shape[1]
    r = min(r, (total_tokens - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing, 0

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        src_metric, dst_metric = metric[..., ::2, :], metric[..., 1::2, :]
        scores = src_metric @ dst_metric.transpose(-1, -2)

        if importance is not None:
            src_imp = importance[..., ::2]
            dst_imp = importance[..., 1::2]
            summed_importance = src_imp[..., :, None] + dst_imp[..., None, :]
            importance_gap = (src_imp[..., :, None] - dst_imp[..., None, :]).abs()
            scores = scores - importance_lambda * summed_importance - importance_eta * importance_gap

        if protect_mask is not None:
            src_protect = protect_mask[..., ::2].bool()
            dst_protect = protect_mask[..., 1::2].bool()
            invalid = src_protect[..., :, None] | dst_protect[..., None, :]
            scores = scores.masked_fill(invalid, -torch.inf)

        if class_token:
            scores[..., 0, :] = -torch.inf
        if distill_token:
            scores[..., :, 0] = -torch.inf

        node_max, node_idx = scores.max(dim=-1)
        valid_rows = torch.isfinite(node_max)
        valid_count = int(valid_rows.sum(dim=-1).min().item())
        r = min(r, valid_count)

        if r <= 0:
            return do_nothing, do_nothing, 0

        node_max = node_max.masked_fill(~valid_rows, -torch.inf)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        batch_size, src_tokens, channels = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(batch_size, src_tokens - r, channels))
        src = src.gather(dim=-2, index=src_idx.expand(batch_size, r, channels))
        dst = dst.scatter_reduce(-2, dst_idx.expand(batch_size, r, channels), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        batch_size, _, channels = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(batch_size, r, channels))
        out = torch.zeros(batch_size, metric.shape[1], channels, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(batch_size, unm_len, channels), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(batch_size, r, channels), src=src)
        return out

    return merge, unmerge, r


def random_bipartite_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> tuple[Callable, Callable, int]:
    protected = int(class_token) + int(distill_token)
    total_tokens = metric.shape[1]
    r = min(r, (total_tokens - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing, 0

    with torch.no_grad():
        src_tokens = metric[..., ::2, :].shape[1]
        dst_tokens = metric[..., 1::2, :].shape[1]
        r = min(r, src_tokens, dst_tokens)

        if r <= 0:
            return do_nothing, do_nothing, 0

        batch_size = metric.shape[0]
        src_order = torch.rand(batch_size, src_tokens, device=metric.device).argsort(dim=-1)
        src_idx = src_order[..., :r, None]
        unm_idx = src_order[..., r:, None]
        dst_idx = torch.randint(0, dst_tokens, (batch_size, r, 1), device=metric.device)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        batch_size, src_tokens, channels = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(batch_size, src_tokens - r, channels))
        src = src.gather(dim=-2, index=src_idx.expand(batch_size, r, channels))
        dst = dst.scatter_reduce(-2, dst_idx.expand(batch_size, r, channels), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        batch_size, _, channels = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(batch_size, r, channels))
        out = torch.zeros(batch_size, metric.shape[1], channels, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(batch_size, unm_len, channels), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(batch_size, r, channels), src=src)
        return out

    return merge, unmerge, r


def merge_weighted_average(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if size is None:
        size = torch.ones_like(x[..., :1])

    merged_x = merge(x * size, mode="sum")
    merged_size = merge(size, mode="sum")
    merged_x = merged_x / merged_size.clamp_min(1e-6)
    return merged_x, merged_size


def merge_reduce(merge: Callable, x: torch.Tensor, reduce: str = "amax") -> torch.Tensor:
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(-1)
        squeeze = True
    x = merge(x, mode=reduce)
    return x.squeeze(-1) if squeeze else x
