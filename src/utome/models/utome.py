from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp, trunc_normal_
from timm.models.vision_transformer import PatchEmbed

from utome.models.importance import TokenImportancePredictor, topk_protect_mask
from utome.models.pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from utome.models.prune import init_prune_state, prune_tokens, restore_pruned
from utome.models.strategies import PRUNE_METHODS, get_utome_strategy
from utome.models.tome import (
    bipartite_soft_matching,
    merge_reduce,
    merge_weighted_average,
    parse_r,
    random_bipartite_matching,
    ratio_to_decreasing_r,
)


@dataclass
class ToMeConfig:
    enabled: bool = True
    r: int | list[int] | tuple[int, float] = 8
    ratio: float | None = None
    prop_attn: bool = False
    start_layer: int = 0


@dataclass
class UtoMeConfig:
    method: str = "baseline"
    probe_layer: int = 3
    target_variables: tuple[str, ...] = ()
    importance_head_type: str = "conv"
    importance_hidden_dim: int = 256
    teacher_beta: float = 1.0
    teacher_mode: str = "dynamics_minus_uq"
    teacher_uq_source: str = "eda"
    teacher_variable_scope: str = "target"
    protect_fraction: float = 0.25
    importance_lambda: float = 0.05
    importance_eta: float = 0.05
    use_predicted_for_merge: bool = True
    merge_warmup_epochs: int = 0


class UtoMeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, size: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(dim=1)


class UtoMeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = UtoMeAttention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, layer_idx: int, state: dict) -> torch.Tensor:
        attn_size = state["size"] if state["prop_attn"] else None
        attn_out, metric = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path1(attn_out)

        r = state["r_schedule"][layer_idx]
        if r > 0 and state["merge_mode"] == "prune":
            x = prune_tokens(x, r, state)
        elif r > 0:
            protect_mask = state.get("protect_mask")
            importance = state.get("importance")
            if state["merge_mode"] == "bias":
                merge_importance = importance
            else:
                merge_importance = None

            if state["merge_mode"] == "random":
                merge, unmerge, r_eff = random_bipartite_matching(metric, r=r)
            else:
                merge, unmerge, r_eff = bipartite_soft_matching(
                    metric,
                    r=r,
                    protect_mask=protect_mask,
                    importance=merge_importance,
                    importance_lambda=state["importance_lambda"],
                    importance_eta=state["importance_eta"],
                )
            if r_eff > 0:
                x, state["size"] = merge_weighted_average(merge, x, state["size"])
                state["unmerges"].append(unmerge)
                if importance is not None:
                    state["importance"] = merge_reduce(merge, importance, reduce="amax")
                if protect_mask is not None:
                    merged_mask = merge_reduce(merge, protect_mask.float(), reduce="amax")
                    state["protect_mask"] = merged_mask.bool()

        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class UtoMeModel(nn.Module):
    def __init__(
        self,
        default_vars: list[str],
        img_size: list[int] = [32, 64],
        patch_size: int = 2,
        embed_dim: int = 768,
        depth: int = 8,
        decoder_depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        tome: dict | None = None,
        utome: dict | None = None,
        variable_aggregation: bool = True,
    ) -> None:
        super().__init__()
        self.default_vars = list(default_vars)
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        # ``variable_aggregation=False`` selects the second, architecturally distinct
        # patch-transformer backbone used for the architecture-transfer experiment:
        # a plain ViT that embeds all input channels with one joint patch-embedding
        # convolution and has no per-variable embedding or variable-aggregation
        # attention. Everything downstream (token merging, head, unpatchify) is shared.
        self.variable_aggregation = bool(variable_aggregation)

        self.tome_cfg = ToMeConfig(**(tome or {}))
        self.utome_cfg = UtoMeConfig(**(utome or {}))
        self.method = self.utome_cfg.method
        self.utome_strategy = get_utome_strategy(self.method)

        if self.variable_aggregation:
            self.token_embeds = nn.ModuleList([PatchEmbed(self.img_size, patch_size, 1, embed_dim) for _ in self.default_vars])
            self.num_patches = self.token_embeds[0].num_patches
            self.var_embed, self.var_map = self._create_var_embedding(embed_dim)
            self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        else:
            self.joint_embed = PatchEmbed(self.img_size, patch_size, len(self.default_vars), embed_dim)
            self.num_patches = self.joint_embed.num_patches
            self.var_map = {name: idx for idx, name in enumerate(self.default_vars)}

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.lead_time_embed = nn.Linear(1, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                UtoMeBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[idx],
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.importance_head = None
        if self.utome_strategy is not None and self.utome_strategy.learns_uq_predictor:
            self.importance_head = TokenImportancePredictor(
                embed_dim=embed_dim,
                hidden_dim=self.utome_cfg.importance_hidden_dim,
                head_type=self.utome_cfg.importance_head_type,
            )

        head = []
        for _ in range(decoder_depth):
            head.extend([nn.Linear(embed_dim, embed_dim), nn.GELU()])
        head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size * patch_size))
        self.head = nn.Sequential(*head)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        grid_h = self.img_size[0] // self.patch_size
        grid_w = self.img_size[1] // self.patch_size
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_h, grid_w)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.variable_aggregation:
            var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
            self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

            for token_embed in self.token_embeds:
                weight = token_embed.proj.weight.data
                trunc_normal_(weight.view(weight.shape[0], -1), std=0.02)
        else:
            weight = self.joint_embed.proj.weight.data
            trunc_normal_(weight.view(weight.shape[0], -1), std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _create_var_embedding(self, dim: int) -> tuple[nn.Parameter, dict[str, int]]:
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim))
        var_map = {name: idx for idx, name in enumerate(self.default_vars)}
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, variables: tuple[str, ...], device: torch.device) -> torch.Tensor:
        indices = np.array([self.var_map[name] for name in variables])
        return torch.from_numpy(indices).to(device)

    def get_var_emb(self, var_emb: torch.Tensor, variables: tuple[str, ...]) -> torch.Tensor:
        return var_emb[:, self.get_var_ids(variables, var_emb.device), :]

    def aggregate_variables(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_tokens, _ = x.shape
        x = torch.einsum("bvld->blvd", x).flatten(0, 1)
        query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(query, x, x)
        x = x.squeeze(1)
        return x.unflatten(dim=0, sizes=(batch_size, num_tokens))

    def unpatchify(self, x: torch.Tensor, out_height: int | None = None, out_width: int | None = None) -> torch.Tensor:
        patch = self.patch_size
        channels = len(self.default_vars)
        out_height = self.img_size[0] if out_height is None else out_height
        out_width = self.img_size[1] if out_width is None else out_width
        grid_h = out_height // patch
        grid_w = out_width // patch

        x = x.reshape(x.shape[0], grid_h, grid_w, patch, patch, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], channels, grid_h * patch, grid_w * patch)

    def _build_r_schedule(self) -> list[int]:
        if not self.tome_cfg.enabled:
            return [0] * self.depth
        start_layer = max(0, min(self.depth, int(self.tome_cfg.start_layer)))
        if start_layer > 0:
            active_layers = self.depth - start_layer
            if active_layers <= 0:
                return [0] * self.depth
            if self.tome_cfg.ratio is not None:
                active_schedule = ratio_to_decreasing_r(active_layers, self.num_patches, self.tome_cfg.ratio)
                return [0] * start_layer + active_schedule
            schedule = parse_r(active_layers, self.tome_cfg.r)
            return [0] * start_layer + schedule
        if self.importance_head is not None:
            cutoff = max(0, min(self.depth, self.utome_cfg.probe_layer + 1))
            active_layers = self.depth - cutoff
            if self.tome_cfg.ratio is not None:
                active_schedule = ratio_to_decreasing_r(active_layers, self.num_patches, self.tome_cfg.ratio)
                return [0] * cutoff + active_schedule
            schedule = parse_r(self.depth, self.tome_cfg.r)
            schedule[:cutoff] = [0] * cutoff
            return schedule
        if self.tome_cfg.ratio is not None:
            return ratio_to_decreasing_r(self.depth, self.num_patches, self.tome_cfg.ratio)
        schedule = parse_r(self.depth, self.tome_cfg.r)
        return schedule

    def _activate_predicted_importance(self, state: dict, predicted_importance: torch.Tensor) -> None:
        if not self.utome_cfg.use_predicted_for_merge:
            return

        self._activate_importance_for_merge(state, predicted_importance)

    def _activate_importance_for_merge(self, state: dict, importance: torch.Tensor) -> None:
        if self.utome_strategy is not None and self.utome_strategy.learns_uq_predictor:
            state["importance"] = importance
            state["protect_mask"] = topk_protect_mask(importance, self.utome_cfg.protect_fraction)
            state["merge_mode"] = "bias"

    def forward(
        self,
        x: torch.Tensor,
        lead_times: torch.Tensor,
        variables: tuple[str, ...] | list[str],
        out_variables: tuple[str, ...] | list[str],
        merge_importance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if isinstance(variables, list):
            variables = tuple(variables)
        if isinstance(out_variables, list):
            out_variables = tuple(out_variables)

        if self.variable_aggregation:
            embeddings = []
            var_ids = self.get_var_ids(variables, x.device)
            for idx, var_id in enumerate(var_ids):
                embeddings.append(self.token_embeds[int(var_id.item())](x[:, idx : idx + 1]))
            tokens = torch.stack(embeddings, dim=1)

            var_embed = self.get_var_emb(self.var_embed, variables)
            tokens = tokens + var_embed.unsqueeze(2)
            tokens = self.aggregate_variables(tokens)
        else:
            if tuple(variables) != tuple(self.default_vars):
                raise ValueError(
                    "The joint patch-embedding backbone expects the full default variable list in order."
                )
            tokens = self.joint_embed(x)

        tokens = tokens + self.pos_embed
        tokens = tokens + self.lead_time_embed(lead_times.unsqueeze(-1)).unsqueeze(1)
        tokens = self.pos_drop(tokens)

        grid_shape = (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size)
        state = {
            "r_schedule": self._build_r_schedule(),
            "size": None,
            "prop_attn": self.tome_cfg.prop_attn,
            "importance": None,
            "protect_mask": None,
            "merge_mode": "none",
            "importance_lambda": self.utome_cfg.importance_lambda,
            "importance_eta": self.utome_cfg.importance_eta,
            "unmerges": [],
        }

        if self.method in PRUNE_METHODS:
            state["merge_mode"] = "prune"
            init_prune_state(state, tokens.shape[0], tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
            if self.method == "prune":
                if merge_importance is None:
                    raise ValueError("prune requires merge_importance")
                state["importance"] = merge_importance
                if self.utome_cfg.protect_fraction > 0:
                    state["protect_mask"] = topk_protect_mask(merge_importance, self.utome_cfg.protect_fraction)
        elif self.utome_strategy is not None and self.utome_strategy.uses_direct_teacher_for_merge:
            if merge_importance is None:
                raise ValueError(f"{self.method} requires merge_importance")
            state["importance"] = merge_importance
            state["protect_mask"] = topk_protect_mask(merge_importance, self.utome_cfg.protect_fraction)
            state["merge_mode"] = "bias"
        elif self.method == "baseline":
            state["merge_mode"] = "none"
        elif self.method == "random_merge":
            state["merge_mode"] = "random"
        elif self.utome_strategy is None:
            raise ValueError(f"Unsupported UtoMe method: {self.method}")

        use_external_teacher = (
            self.utome_strategy is not None
            and self.utome_strategy.learns_uq_predictor
            and merge_importance is not None
        )
        predicted_importance = None
        if self.importance_head is not None and self.utome_cfg.probe_layer < 0:
            predicted_importance = self.importance_head(tokens, grid_shape)
            if use_external_teacher:
                self._activate_importance_for_merge(state, merge_importance)
            else:
                self._activate_predicted_importance(state, predicted_importance)

        for idx, block in enumerate(self.blocks):
            tokens = block(tokens, idx, state)
            if self.importance_head is not None and idx == self.utome_cfg.probe_layer:
                predicted_importance = self.importance_head(tokens, grid_shape)
                if use_external_teacher:
                    self._activate_importance_for_merge(state, merge_importance)
                else:
                    self._activate_predicted_importance(state, predicted_importance)

        if state["merge_mode"] == "prune":
            tokens = restore_pruned(tokens, state)

        tokens = self.norm(tokens)
        preds = self.head(tokens)
        for unmerge in reversed(state["unmerges"]):
            preds = unmerge(preds)
        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(out_variables, preds.device)
        preds = preds[:, out_var_ids]

        return {
            "preds": preds,
            "predicted_importance": predicted_importance,
            "merge_importance": state["importance"],
        }
