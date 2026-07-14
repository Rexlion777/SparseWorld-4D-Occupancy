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
FRONT_TRIPLET = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]
REAR_CAMERAS = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAMERA_TO_INDEX = {name: idx for idx, name in enumerate(CAMERA_NAMES)}
FRONT_TRIPLET_INDICES = [CAMERA_TO_INDEX[name] for name in FRONT_TRIPLET]
REAR_CAMERA_INDICES = [CAMERA_TO_INDEX[name] for name in REAR_CAMERAS]


@dataclass(frozen=True)
class AdapterStats:
    parameter_count: int
    trainable_parameter_count: int


def _check_feature_levels(current_levels: list[torch.Tensor], memory_tminus1_levels: list[torch.Tensor]) -> None:
    if len(current_levels) != len(memory_tminus1_levels):
        raise ValueError("current and memory feature level counts differ")
    if not current_levels:
        raise ValueError("feature levels are empty")
    for level_idx, (current, memory) in enumerate(zip(current_levels, memory_tminus1_levels)):
        if current.shape != memory.shape:
            raise ValueError(f"feature shape mismatch at level {level_idx}: {tuple(current.shape)} vs {tuple(memory.shape)}")
        if current.ndim != 5:
            raise ValueError(f"expected [B,6,C,H,W] feature at level {level_idx}, got ndim={current.ndim}")
        if current.shape[1] != len(CAMERA_NAMES):
            raise ValueError(f"expected 6 cameras at level {level_idx}, got {current.shape[1]}")


def front_triplet_degradation_mask(
    degradation_id: str,
    *,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, len(CAMERA_NAMES), 1), device=device, dtype=dtype)
    if degradation_id == "A10_drop_front_triplet":
        mask[:, FRONT_TRIPLET_INDICES, :] = 1.0
    return mask


def build_sw13_r8_base_repair(
    current_levels: list[torch.Tensor],
    memory_tminus1_levels: list[torch.Tensor],
    degradation_id: str,
) -> list[torch.Tensor]:
    """SW13 R8 hard repair: replace A10 front triplet camera features by t-1 memory."""
    _check_feature_levels(current_levels, memory_tminus1_levels)
    base_levels = [level.clone() for level in current_levels]
    if degradation_id != "A10_drop_front_triplet":
        return base_levels
    for base, memory in zip(base_levels, memory_tminus1_levels):
        base[:, FRONT_TRIPLET_INDICES] = memory[:, FRONT_TRIPLET_INDICES].to(device=base.device, dtype=base.dtype)
    return base_levels


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        camera_embed_dim: int,
        residual_limit: float,
        gate_bias_init: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.residual_limit = float(residual_limit)
        in_channels = channels * 5 + camera_embed_dim + 1
        self.reduce = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.act = nn.SiLU()
        self.delta = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.reset_parameters(gate_bias_init=gate_bias_init)

    def reset_parameters(self, gate_bias_init: float) -> None:
        nn.init.kaiming_uniform_(self.reduce.weight, a=5**0.5)
        nn.init.zeros_(self.reduce.bias)
        nn.init.kaiming_uniform_(self.depthwise.weight, a=5**0.5)
        nn.init.zeros_(self.depthwise.bias)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.act(self.depthwise(self.reduce(x)))
        residual_delta = torch.tanh(self.delta(hidden)) * self.residual_limit
        residual_gate = torch.sigmoid(self.gate(hidden))
        return residual_delta, residual_gate


class SpatialResidualAdapter(nn.Module):
    def __init__(
        self,
        channels_per_level: list[int],
        *,
        camera_count: int = 6,
        hidden_channels: int = 32,
        camera_embed_dim: int = 4,
        residual_limit: float = 0.10,
        gate_bias_init: float = -4.0,
    ) -> None:
        super().__init__()
        if camera_count != len(CAMERA_NAMES):
            raise ValueError("SW14C residual adapter assumes the SparseWorld 6-camera order")
        self.channels_per_level = [int(channels) for channels in channels_per_level]
        self.hidden_channels = int(hidden_channels)
        self.camera_embed_dim = int(camera_embed_dim)
        self.residual_limit = float(residual_limit)
        self.gate_bias_init = float(gate_bias_init)
        self.camera_embedding = nn.Embedding(camera_count, camera_embed_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    camera_embed_dim=camera_embed_dim,
                    residual_limit=residual_limit,
                    gate_bias_init=gate_bias_init,
                )
                for channels in self.channels_per_level
            ]
        )

    def _block_input(
        self,
        current: torch.Tensor,
        memory: torch.Tensor,
        base: torch.Tensor,
        degradation_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, cam_count, _channels, height, width = current.shape
        camera_ids = torch.arange(cam_count, device=current.device)[None].expand(bsz, -1)
        camera_embed = self.camera_embedding(camera_ids.long()).to(dtype=current.dtype)
        camera_embed = camera_embed[:, :, :, None, None].expand(-1, -1, -1, height, width)
        deg = degradation_mask.to(device=current.device, dtype=current.dtype)[:, :, :, None, None].expand(-1, -1, 1, height, width)
        base_current = base - current
        features = torch.cat([current, memory, base, base_current, base_current.abs(), camera_embed, deg], dim=2)
        return features.reshape(bsz * cam_count, features.shape[2], height, width)

    def apply_to_levels(
        self,
        current_levels: list[torch.Tensor],
        memory_tminus1_levels: list[torch.Tensor],
        base_repaired_levels: list[torch.Tensor],
        degradation_mask: torch.Tensor,
        *,
        gamma: float = 0.10,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        _check_feature_levels(current_levels, memory_tminus1_levels)
        _check_feature_levels(current_levels, base_repaired_levels)
        repaired_levels: list[torch.Tensor] = []
        debug: dict[str, torch.Tensor] = {}
        for level_idx, (current, memory, base) in enumerate(zip(current_levels, memory_tminus1_levels, base_repaired_levels)):
            bsz, cam_count, channels, height, width = current.shape
            block_input = self._block_input(current, memory, base, degradation_mask)
            delta_flat, gate_flat = self.blocks[level_idx](block_input)
            delta = delta_flat.reshape(bsz, cam_count, channels, height, width)
            gate = gate_flat.reshape(bsz, cam_count, 1, height, width)
            mask = degradation_mask.to(device=current.device, dtype=current.dtype)[:, :, :, None, None]
            delta = delta * mask
            gate = gate * mask
            final = base + float(gamma) * gate * delta
            repaired_levels.append(final)
            debug[f"residual_delta_level{level_idx}"] = delta
            debug[f"residual_gate_level{level_idx}"] = gate
            debug[f"final_level{level_idx}"] = final
        debug["degradation_mask"] = degradation_mask
        return repaired_levels, debug


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def count_parameters(module: nn.Module) -> AdapterStats:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return AdapterStats(parameter_count=int(total), trainable_parameter_count=int(trainable))


def checkpoint_payload(adapter: nn.Module, meta: dict[str, Any]) -> dict[str, Any]:
    return {"state_dict": adapter.state_dict(), "meta": meta}
