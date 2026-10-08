from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TeacherSignal:
    importance: torch.Tensor
    uq_patch: torch.Tensor
    dynamics_patch: torch.Tensor


class TokenImportancePredictor(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256, head_type: str = "conv") -> None:
        super().__init__()
        self.head_type = head_type

        if head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        elif head_type == "conv":
            self.net = nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, tokens: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        batch_size, num_tokens, channels = tokens.shape
        grid_h, grid_w = grid_shape
        if num_tokens != grid_h * grid_w:
            raise ValueError("Token count does not match grid shape")

        if self.head_type == "mlp":
            return self.net(tokens).squeeze(-1)

        x = tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)
        x = self.net(x)
        return x.flatten(1)


def pairwise_rank_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred_diff = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    label_diff = labels.unsqueeze(2) - labels.unsqueeze(1)
    target = torch.sign(label_diff)
    mask = ~torch.eye(predictions.shape[1], device=predictions.device, dtype=torch.bool).unsqueeze(0)
    loss = F.softplus(-target * pred_diff)
    return loss.masked_select(mask).mean()


def normalized_mse_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred = zscore(predictions)
    target = zscore(labels)
    return F.mse_loss(pred, target)


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(eps)


def spatial_channel_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(2)
    flat = (flat - flat.mean(dim=-1, keepdim=True)) / flat.std(dim=-1, keepdim=True).clamp_min(eps)
    return flat.view_as(x)


def topk_protect_mask(importance: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0:
        return torch.zeros_like(importance, dtype=torch.bool)
    tokens = importance.shape[-1]
    topk = max(1, int(round(tokens * fraction)))
    indices = importance.topk(topk, dim=-1).indices
    mask = torch.zeros_like(importance, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    return mask


def _select_variables(
    tensor: torch.Tensor,
    names: tuple[str, ...],
    selected: tuple[str, ...],
) -> torch.Tensor:
    indices = [names.index(name) for name in selected if name in names]
    if not indices:
        return tensor
    return tensor[:, indices]


def _patch_reduce(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    x = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
    return x.flatten(1)


def _dynamic_saliency(x: torch.Tensor) -> torch.Tensor:
    dx = torch.diff(x, dim=-1, prepend=x[..., :1])
    dy = torch.diff(x, dim=-2, prepend=x[..., :, :1, :])
    grad = dx.abs() + dy.abs()
    centered = (x - x.mean(dim=(-2, -1), keepdim=True)).abs()
    return grad.mean(dim=1) + 0.5 * centered.mean(dim=1)


def _proxy_uncertainty(x: torch.Tensor) -> torch.Tensor:
    """Estimate analysis uncertainty from local spatial inconsistency.

    This is used only when ensemble spread is unavailable. It measures how hard
    each grid point is to reconstruct from its local 3x3 neighborhood, averaged
    over variables. The resulting map is not a meteorological diagnostic; it is
    an ensemble-free pseudo target for the uncertainty predictor.
    """
    local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    residual = (x - local_mean).abs()
    return residual.mean(dim=1)


def build_utome_teacher(
    inputs: torch.Tensor,
    variables: tuple[str, ...],
    uq: torch.Tensor | None,
    uq_variables: tuple[str, ...] | None,
    target_variables: tuple[str, ...],
    patch_size: int,
    beta: float,
    mode: str = "dynamics_minus_uq",
    uq_source: str = "eda",
) -> TeacherSignal | None:
    if uq_source not in {"eda", "proxy", "auto"}:
        raise ValueError(f"Unsupported UToMe UQ source: {uq_source}")
    use_proxy = uq_source == "proxy" or (uq is None and uq_source == "auto")
    if uq is None and not use_proxy:
        return None

    x_target = _select_variables(inputs, variables, target_variables)
    if use_proxy:
        uq_map = _proxy_uncertainty(x_target)
    elif uq_variables is None:
        uq_target = spatial_channel_zscore(uq.abs())
        uq_map = uq_target.mean(dim=1)
    else:
        uq_target = _select_variables(uq, uq_variables, target_variables)
        uq_target = spatial_channel_zscore(uq_target.abs())
        uq_map = uq_target.mean(dim=1)

    dynamics_map = _dynamic_saliency(x_target)

    dynamics_patch = _patch_reduce(dynamics_map.unsqueeze(1), patch_size)
    uq_patch = _patch_reduce(uq_map.unsqueeze(1), patch_size)

    dynamics_score = zscore(dynamics_patch)
    uq_score = zscore(uq_patch)
    if mode == "dynamics_minus_uq":
        importance = dynamics_score - beta * uq_score
    elif mode == "dynamics_plus_uq":
        importance = dynamics_score + beta * uq_score
    elif mode == "uq_only":
        importance = uq_score
    elif mode == "negative_uq_only":
        importance = -uq_score
    elif mode == "dynamics_only":
        importance = dynamics_score
    else:
        raise ValueError(f"Unsupported UToMe teacher mode: {mode}")
    return TeacherSignal(importance=importance, uq_patch=uq_patch, dynamics_patch=dynamics_patch)
