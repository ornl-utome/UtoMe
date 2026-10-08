import numpy as np
import torch


def mse(pred: torch.Tensor, target: torch.Tensor, variables: tuple[str, ...]) -> dict[str, torch.Tensor]:
    error = (pred - target) ** 2
    loss_dict: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(variables):
        loss_dict[name] = error[:, idx].mean()
    loss_dict["loss"] = error.mean()
    return loss_dict


def lat_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    variables: tuple[str, ...],
    lat: np.ndarray,
) -> dict[str, torch.Tensor]:
    error = (pred - target) ** 2
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    w_lat = torch.as_tensor(weights, dtype=error.dtype, device=error.device).view(1, 1, -1, 1)

    loss_dict: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(variables):
        loss_dict[name] = (error[:, idx : idx + 1] * w_lat).mean()
    loss_dict["loss"] = (error * w_lat).mean()
    return loss_dict


def lat_weighted_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    variables: tuple[str, ...],
    lat: np.ndarray,
) -> dict[str, torch.Tensor]:
    error = (pred - target) ** 2
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    w_lat = torch.as_tensor(weights, dtype=error.dtype, device=error.device).view(1, 1, -1, 1)

    values: dict[str, torch.Tensor] = {}
    rmse_terms = []
    for idx, name in enumerate(variables):
        rmse = torch.sqrt((error[:, idx : idx + 1] * w_lat).mean(dim=(-2, -1))).mean()
        values[f"w_rmse_{name}"] = rmse
        rmse_terms.append(rmse)
    values["w_rmse"] = torch.stack(rmse_terms).mean()
    return values


def lat_weighted_mse_val(
    pred: torch.Tensor,
    target: torch.Tensor,
    variables: tuple[str, ...],
    lat: np.ndarray,
) -> dict[str, torch.Tensor]:
    error = (pred - target) ** 2
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    w_lat = torch.as_tensor(weights, dtype=error.dtype, device=error.device).view(1, 1, -1, 1)

    values: dict[str, torch.Tensor] = {}
    mse_terms = []
    for idx, name in enumerate(variables):
        w_mse = (error[:, idx : idx + 1] * w_lat).mean()
        values[f"w_mse_{name}"] = w_mse
        mse_terms.append(w_mse)
    values["w_mse"] = torch.stack(mse_terms).mean()
    return values


def lat_weighted_acc(
    pred: torch.Tensor,
    target: torch.Tensor,
    variables: tuple[str, ...],
    lat: np.ndarray,
    clim: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    w_lat = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device).view(1, 1, -1, 1)

    if clim is not None:
        clim = clim.to(device=pred.device, dtype=pred.dtype).view(1, pred.shape[1], pred.shape[2], pred.shape[3])
        pred = pred - clim
        target = target - clim

    values: dict[str, torch.Tensor] = {}
    acc_terms = []
    for idx, name in enumerate(variables):
        pred_i = pred[:, idx : idx + 1]
        target_i = target[:, idx : idx + 1]
        pred_prime = pred_i - pred_i.mean()
        target_prime = target_i - target_i.mean()
        numerator = torch.sum(w_lat * pred_prime * target_prime)
        denominator = torch.sqrt(torch.sum(w_lat * pred_prime.square()) * torch.sum(w_lat * target_prime.square()))
        acc = numerator / denominator.clamp_min(1e-6)
        values[f"acc_{name}"] = acc
        acc_terms.append(acc)
    values["acc"] = torch.stack(acc_terms).mean()
    return values
