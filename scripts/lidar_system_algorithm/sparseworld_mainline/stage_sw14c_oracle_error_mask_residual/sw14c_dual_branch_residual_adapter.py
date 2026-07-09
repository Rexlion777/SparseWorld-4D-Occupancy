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
CAMERA_TO_INDEX = {name: idx for idx, name in enumerate(CAMERA_NAMES)}
FRONT_TRIPLET_INDICES = [CAMERA_TO_INDEX[name] for name in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]]
REAR_CAMERA_INDICES = [CAMERA_TO_INDEX[name] for name in ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]]


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


def count_parameters(module: nn.Module) -> AdapterStats:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return AdapterStats(parameter_count=int(total), trainable_parameter_count=int(trainable))


class DualResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        camera_embed_dim: int,
        residual_limit: float,
        gate_bias_init: float,
    ) -> None:
        super().__init__()
        in_channels = channels * 5 + camera_embed_dim + 1
        self.residual_limit = float(residual_limit)
        self.reduce = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.act = nn.SiLU()
        self.delta = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.reset_parameters(gate_bias_init)

    def reset_parameters(self, gate_bias_init: float) -> None:
        nn.init.kaiming_uniform_(self.reduce.weight, a=5**0.5)
        nn.init.zeros_(self.reduce.bias)
        nn.init.kaiming_uniform_(self.depthwise.weight, a=5**0.5)
        nn.init.zeros_(self.depthwise.bias)
        nn.init.normal_(self.delta.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.act(self.depthwise(self.reduce(x)))
        return torch.tanh(self.delta(hidden)) * self.residual_limit, torch.sigmoid(self.gate(hidden))


class DualBranchSpatialResidualAdapter(nn.Module):
    """Dual add/suppress feature residual adapter.

    The oracle teacher-error mask is voxel-space in this codebase. There is no
    feature-to-voxel projection available here, so this module applies structural
    camera/degradation masks at feature level and the oracle masks are enforced
    by output-space losses/audits.
    """

    def __init__(
        self,
        channels_per_level: list[int],
        *,
        camera_count: int = 6,
        hidden_channels: int = 64,
        camera_embed_dim: int = 4,
        residual_limit: float = 0.05,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        if camera_count != len(CAMERA_NAMES):
            raise ValueError("dual branch adapter assumes SparseWorld 6-camera order")
        self.channels_per_level = [int(v) for v in channels_per_level]
        self.hidden_channels = int(hidden_channels)
        self.camera_embed_dim = int(camera_embed_dim)
        self.residual_limit = float(residual_limit)
        self.gate_bias_init = float(gate_bias_init)
        self.camera_embedding = nn.Embedding(camera_count, camera_embed_dim)
        self.add_blocks = nn.ModuleList(
            [DualResidualBlock(ch, hidden_channels, camera_embed_dim, residual_limit, gate_bias_init) for ch in self.channels_per_level]
        )
        self.sup_blocks = nn.ModuleList(
            [DualResidualBlock(ch, hidden_channels, camera_embed_dim, residual_limit, gate_bias_init) for ch in self.channels_per_level]
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
        gamma_add: float = 0.10,
        gamma_sup: float = 0.10,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        _check_feature_levels(current_levels, memory_tminus1_levels)
        _check_feature_levels(current_levels, base_repaired_levels)
        repaired_levels: list[torch.Tensor] = []
        debug: dict[str, torch.Tensor] = {}
        for level_idx, (current, memory, base) in enumerate(zip(current_levels, memory_tminus1_levels, base_repaired_levels)):
            bsz, cam_count, channels, height, width = current.shape
            block_input = self._block_input(current, memory, base, degradation_mask)
            add_delta_flat, add_gate_flat = self.add_blocks[level_idx](block_input)
            sup_delta_flat, sup_gate_flat = self.sup_blocks[level_idx](block_input)
            add_delta = add_delta_flat.reshape(bsz, cam_count, channels, height, width)
            sup_delta = sup_delta_flat.reshape(bsz, cam_count, channels, height, width)
            add_gate = add_gate_flat.reshape(bsz, cam_count, 1, height, width)
            sup_gate = sup_gate_flat.reshape(bsz, cam_count, 1, height, width)
            mask = degradation_mask.to(device=current.device, dtype=current.dtype)[:, :, :, None, None]
            add_delta = add_delta * mask
            sup_delta = sup_delta * mask
            add_gate = add_gate * mask
            sup_gate = sup_gate * mask
            final = base + float(gamma_add) * add_gate * add_delta - float(gamma_sup) * sup_gate * sup_delta
            repaired_levels.append(final)
            debug[f"add_delta_level{level_idx}"] = add_delta
            debug[f"sup_delta_level{level_idx}"] = sup_delta
            debug[f"add_gate_level{level_idx}"] = add_gate
            debug[f"sup_gate_level{level_idx}"] = sup_gate
            debug[f"final_level{level_idx}"] = final
        debug["degradation_mask"] = degradation_mask
        return repaired_levels, debug

    def apply_to_levels_gamma_batch(
        self,
        current_levels: list[torch.Tensor],
        memory_tminus1_levels: list[torch.Tensor],
        base_repaired_levels: list[torch.Tensor],
        degradation_mask: torch.Tensor,
        *,
        gamma_add: torch.Tensor,
        gamma_sup: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        """Apply one prepared sample expanded over a gamma-pair batch.

        This keeps the expensive SparseWorld validation path flatter by letting
        one model forward evaluate several gamma pairs. The feature tensors are
        already expanded along batch dimension by the caller.
        """
        _check_feature_levels(current_levels, memory_tminus1_levels)
        _check_feature_levels(current_levels, base_repaired_levels)
        batch_size = int(current_levels[0].shape[0])
        gamma_add_t = gamma_add.to(device=current_levels[0].device, dtype=current_levels[0].dtype).reshape(batch_size, 1, 1, 1, 1)
        gamma_sup_t = gamma_sup.to(device=current_levels[0].device, dtype=current_levels[0].dtype).reshape(batch_size, 1, 1, 1, 1)
        repaired_levels: list[torch.Tensor] = []
        debug: dict[str, torch.Tensor] = {}
        for level_idx, (current, memory, base) in enumerate(zip(current_levels, memory_tminus1_levels, base_repaired_levels)):
            bsz, cam_count, channels, height, width = current.shape
            if bsz != batch_size:
                raise ValueError(f"gamma batch size mismatch at level {level_idx}: {bsz} vs {batch_size}")
            block_input = self._block_input(current, memory, base, degradation_mask)
            add_delta_flat, add_gate_flat = self.add_blocks[level_idx](block_input)
            sup_delta_flat, sup_gate_flat = self.sup_blocks[level_idx](block_input)
            add_delta = add_delta_flat.reshape(bsz, cam_count, channels, height, width)
            sup_delta = sup_delta_flat.reshape(bsz, cam_count, channels, height, width)
            add_gate = add_gate_flat.reshape(bsz, cam_count, 1, height, width)
            sup_gate = sup_gate_flat.reshape(bsz, cam_count, 1, height, width)
            mask = degradation_mask.to(device=current.device, dtype=current.dtype)[:, :, :, None, None]
            add_delta = add_delta * mask
            sup_delta = sup_delta * mask
            add_gate = add_gate * mask
            sup_gate = sup_gate * mask
            final = base + gamma_add_t * add_gate * add_delta - gamma_sup_t * sup_gate * sup_delta
            repaired_levels.append(final)
            debug[f"add_delta_level{level_idx}"] = add_delta
            debug[f"sup_delta_level{level_idx}"] = sup_delta
            debug[f"add_gate_level{level_idx}"] = add_gate
            debug[f"sup_gate_level{level_idx}"] = sup_gate
            debug[f"final_level{level_idx}"] = final
        debug["degradation_mask"] = degradation_mask
        return repaired_levels, debug


def build_dual_branch_adapter(
    *,
    hidden_channels: int = 64,
    gate_bias_init: float = -2.0,
    residual_limit: float = 0.05,
) -> DualBranchSpatialResidualAdapter:
    adapter = DualBranchSpatialResidualAdapter(
        channels_per_level=[256, 256, 256, 256],
        hidden_channels=hidden_channels,
        camera_embed_dim=4,
        residual_limit=residual_limit,
        gate_bias_init=gate_bias_init,
    )
    return adapter.cuda() if torch.cuda.is_available() else adapter


def checkpoint_payload(adapter: nn.Module, meta: dict[str, Any]) -> dict[str, Any]:
    return {"state_dict": adapter.state_dict(), "meta": meta}
