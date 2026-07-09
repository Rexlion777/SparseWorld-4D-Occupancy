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


class ChannelGateAdapter(nn.Module):
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
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
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
        return repaired_levels, {"alpha_level0": alpha_map}


class SpatialGateBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.act = nn.SiLU()
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.depthwise(x)
        x = self.act(x)
        return self.out(x)


class SpatialGateAdapter(nn.Module):
    def __init__(
        self,
        channels_per_level: list[int],
        camera_count: int = 6,
        hidden_channels: int = 32,
        camera_embed_dim: int = 4,
        use_channel_spatial_gate: bool = False,
    ) -> None:
        super().__init__()
        self.channels_per_level = [int(c) for c in channels_per_level]
        self.camera_count = int(camera_count)
        self.hidden_channels = int(hidden_channels)
        self.use_channel_spatial_gate = bool(use_channel_spatial_gate)
        self.camera_embedding = nn.Embedding(camera_count, camera_embed_dim)
        self.blocks = nn.ModuleList()
        self.scale_mlp = nn.ModuleList()
        for channels in self.channels_per_level:
            in_channels = channels * 3 + camera_embed_dim + 2
            self.blocks.append(SpatialGateBlock(in_channels, hidden_channels))
            if self.use_channel_spatial_gate:
                self.scale_mlp.append(nn.Conv2d(hidden_channels, channels, kernel_size=1))
            else:
                self.scale_mlp.append(nn.Identity())

    def _feature_map(
        self,
        current: torch.Tensor,
        memory: torch.Tensor,
        degradation_mask: torch.Tensor,
        camera_ids: torch.Tensor,
        memory_age: torch.Tensor,
        level_idx: int,
        agreement_proxy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, cam_count, channels, height, width = current.shape
        diff = current - memory
        abs_diff = diff.abs()
        if agreement_proxy is None:
            agreement_proxy = torch.zeros_like(degradation_mask)
        cam_embed = self.camera_embedding(camera_ids.long()).to(current.dtype)
        cam_embed = cam_embed[:, :, :, None, None].expand(-1, -1, -1, height, width)
        deg = degradation_mask[:, :, :, None, None].expand(-1, -1, 1, height, width)
        age = memory_age[:, :, :, None, None].expand(-1, -1, 1, height, width)
        feat = torch.cat([current, memory, abs_diff, cam_embed, deg, age], dim=2)
        feat = feat.reshape(bsz * cam_count, feat.shape[2], height, width)
        return feat

    def apply_to_levels(
        self,
        current_levels: list[torch.Tensor],
        memory_levels: list[torch.Tensor],
        degradation_mask: torch.Tensor,
        camera_ids: torch.Tensor,
        memory_age: torch.Tensor,
        agreement_proxy: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        repaired: list[torch.Tensor] = []
        alpha_debug: dict[str, torch.Tensor] = {}
        for level_idx, (current, memory) in enumerate(zip(current_levels, memory_levels)):
            bsz, cam_count, channels, height, width = current.shape
            features = self._feature_map(current, memory, degradation_mask, camera_ids, memory_age, level_idx, agreement_proxy)
            hidden_logits = self.blocks[level_idx](features)
            if self.use_channel_spatial_gate:
                alpha_map = torch.sigmoid(self.scale_mlp[level_idx](hidden_logits))
                alpha_map = alpha_map.reshape(bsz, cam_count, channels, height, width)
            else:
                alpha_map = torch.sigmoid(hidden_logits).reshape(bsz, cam_count, 1, height, width)
            alpha_map = alpha_map * degradation_mask[:, :, :, None, None]
            repaired_level = alpha_map * memory + (1.0 - alpha_map) * current
            repaired.append(repaired_level)
            alpha_debug[f"alpha_level{level_idx}"] = alpha_map
        return repaired, alpha_debug


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def count_parameters(module: nn.Module) -> AdapterStats:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return AdapterStats(parameter_count=int(total), trainable_parameter_count=int(trainable))


def checkpoint_payload(adapter: nn.Module, meta: dict[str, Any]) -> dict[str, Any]:
    return {"state_dict": adapter.state_dict(), "meta": meta}
