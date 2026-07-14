from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass(frozen=True)
class AdapterStats:
    parameter_count: int
    trainable_parameter_count: int


class FeatureGateAdapter(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        camera_count: int = 6,
        camera_embed_dim: int = 8,
        hidden_dim: int = 64,
        channel_wise: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.camera_count = int(camera_count)
        self.channel_wise = bool(channel_wise)
        self.camera_embedding = nn.Embedding(camera_count, camera_embed_dim)
        out_dim = channels if channel_wise else 1
        in_dim = channels * 4 + camera_embed_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward_pooled(
        self,
        current_pooled: torch.Tensor,
        memory_pooled: torch.Tensor,
        degradation_mask: torch.Tensor,
        camera_ids: torch.Tensor,
        memory_age: torch.Tensor,
        agreement_proxy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diff = current_pooled - memory_pooled
        abs_diff = diff.abs()
        if agreement_proxy is None:
            agreement_proxy = torch.zeros_like(degradation_mask)
        cam_embed = self.camera_embedding(camera_ids.long())
        features = torch.cat(
            [
                current_pooled,
                memory_pooled,
                diff,
                abs_diff,
                cam_embed,
                degradation_mask,
                memory_age,
            ],
            dim=-1,
        )
        logits = self.mlp(features)
        alpha = torch.sigmoid(logits)
        if not self.channel_wise:
            alpha = alpha.expand(-1, -1, self.channels)
        alpha = alpha * degradation_mask
        repaired = alpha * memory_pooled + (1.0 - alpha) * current_pooled
        return alpha, repaired

    def infer_alpha_map(
        self,
        current_feat_level0: torch.Tensor,
        memory_feat_level0: torch.Tensor,
        degradation_mask: torch.Tensor,
        camera_ids: torch.Tensor,
        memory_age: torch.Tensor,
        agreement_proxy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current_pooled = current_feat_level0.mean(dim=(-1, -2))
        memory_pooled = memory_feat_level0.mean(dim=(-1, -2))
        alpha, _ = self.forward_pooled(
            current_pooled=current_pooled,
            memory_pooled=memory_pooled,
            degradation_mask=degradation_mask,
            camera_ids=camera_ids,
            memory_age=memory_age,
            agreement_proxy=agreement_proxy,
        )
        return alpha[..., None, None]

    def apply_to_levels(
        self,
        current_levels: list[torch.Tensor],
        memory_levels: list[torch.Tensor],
        degradation_mask: torch.Tensor,
        camera_ids: torch.Tensor,
        memory_age: torch.Tensor,
        agreement_proxy: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if not current_levels:
            return [], torch.zeros((0,), dtype=torch.float32, device=degradation_mask.device)
        alpha_map = self.infer_alpha_map(
            current_feat_level0=current_levels[0],
            memory_feat_level0=memory_levels[0],
            degradation_mask=degradation_mask,
            camera_ids=camera_ids,
            memory_age=memory_age,
            agreement_proxy=agreement_proxy,
        )
        repaired_levels: list[torch.Tensor] = []
        for current_feat, memory_feat in zip(current_levels, memory_levels):
            alpha_level = alpha_map
            if alpha_level.shape[2] != current_feat.shape[2]:
                alpha_level = alpha_level[:, :, :1]
            repaired_levels.append(alpha_level * memory_feat + (1.0 - alpha_level) * current_feat)
        return repaired_levels, alpha_map


def count_parameters(module: nn.Module) -> AdapterStats:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return AdapterStats(parameter_count=int(total), trainable_parameter_count=int(trainable))


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def checkpoint_payload(adapter: FeatureGateAdapter, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dict": adapter.state_dict(),
        "meta": meta,
    }
