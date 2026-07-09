from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sw14c_residual_adapter_modules import (
    FRONT_TRIPLET_INDICES,
    REAR_CAMERA_INDICES,
    SpatialResidualAdapter,
    build_sw13_r8_base_repair,
    count_parameters,
    front_triplet_degradation_mask,
)


def build_round2b_adapter(
    *,
    gate_bias_init: float = -3.0,
    residual_bound_value: float = 0.05,
    residual_delta_init: str = "small",
) -> SpatialResidualAdapter:
    adapter = SpatialResidualAdapter(
        channels_per_level=[256, 256, 256, 256],
        hidden_channels=32,
        camera_embed_dim=4,
        residual_limit=float(residual_bound_value),
        gate_bias_init=float(gate_bias_init),
    )
    if residual_delta_init == "small":
        for block in adapter.blocks:
            nn.init.normal_(block.delta.weight, mean=0.0, std=1.0e-4)
            nn.init.zeros_(block.delta.bias)
    elif residual_delta_init == "zero":
        for block in adapter.blocks:
            nn.init.zeros_(block.delta.weight)
            nn.init.zeros_(block.delta.bias)
    else:
        raise ValueError(f"unsupported residual_delta_init={residual_delta_init}")
    return adapter.cuda() if torch.cuda.is_available() else adapter


def adapter_config_payload() -> dict[str, Any]:
    return {
        "round2b_gate_bias_init": -3.0,
        "round2b_residual_delta_init": "small",
        "round2b_residual_bound": "enabled",
        "round2b_residual_bound_value": 0.05,
        "round2b_front_triplet_only": True,
        "round2b_rear_zero": True,
        "round2b_clean_zero": True,
        "target_front_gate_mean_after_training": [0.005, 0.05],
        "target_front_gate_p95_max": 0.30,
        "target_rear_gate_mean": 0.0,
        "target_clean_gate_mean": 0.0,
    }


def audit_adapter_init(adapter: SpatialResidualAdapter) -> dict[str, Any]:
    stats = count_parameters(adapter)
    device = next(adapter.parameters()).device
    current = [torch.randn(1, 6, 256, 2, 2, device=device) for _ in range(4)]
    memory = [level + 1.0 for level in current]
    base = build_sw13_r8_base_repair(current, memory, "A10_drop_front_triplet")
    mask = front_triplet_degradation_mask("A10_drop_front_triplet", batch_size=1, device=device)
    clean_mask = front_triplet_degradation_mask("A0_clean", batch_size=1, device=device)
    clean_base = build_sw13_r8_base_repair(current, memory, "A0_clean")
    with torch.inference_mode():
        final_gamma0, debug_gamma0 = adapter.apply_to_levels(current, memory, base, mask, gamma=0.0)
        final_gamma1, debug_gamma1 = adapter.apply_to_levels(current, memory, base, mask, gamma=0.1)
        clean_final, clean_debug = adapter.apply_to_levels(current, memory, clean_base, clean_mask, gamma=0.1)
    gamma0_max_abs = max(float((final - base_level).abs().max().item()) for final, base_level in zip(final_gamma0, base))
    gamma1_max_abs = max(float((final - base_level).abs().max().item()) for final, base_level in zip(final_gamma1, base))
    rear_max = max(float((final[:, REAR_CAMERA_INDICES] - base_level[:, REAR_CAMERA_INDICES]).abs().max().item()) for final, base_level in zip(final_gamma1, base))
    clean_max = max(float((final - base_level).abs().max().item()) for final, base_level in zip(clean_final, clean_base))
    front_gate_mean = float(torch.stack([debug_gamma1[f"residual_gate_level{i}"][:, FRONT_TRIPLET_INDICES].mean() for i in range(4)]).mean().item())
    rear_gate_mean = float(torch.stack([debug_gamma1[f"residual_gate_level{i}"][:, REAR_CAMERA_INDICES].mean() for i in range(4)]).mean().item())
    clean_gate_mean = float(torch.stack([clean_debug[f"residual_gate_level{i}"].mean() for i in range(4)]).mean().item())
    return {
        "decision": "ADAPTER_INIT_R1_READY"
        if gamma0_max_abs <= 1e-12 and rear_max <= 1e-12 and clean_max <= 1e-12 and front_gate_mean > 0.0
        else "ADAPTER_INIT_R2_PROTOCOL_RISK",
        "adapter_trainable_params": int(stats.trainable_parameter_count),
        "adapter_total_params": int(stats.parameter_count),
        "gamma0_equals_sw13_base_max_abs": gamma0_max_abs,
        "gamma0_equals_sw13_base": bool(gamma0_max_abs <= 1e-12),
        "gamma0_debug_gate_mean": float(torch.stack([debug_gamma0[f"residual_gate_level{i}"][:, FRONT_TRIPLET_INDICES].mean() for i in range(4)]).mean().item()),
        "gamma0_debug_delta_mean_abs": float(torch.stack([debug_gamma0[f"residual_delta_level{i}"][:, FRONT_TRIPLET_INDICES].abs().mean() for i in range(4)]).mean().item()),
        "gamma0p1_close_to_base_max_abs": gamma1_max_abs,
        "rear_residual_max_abs": rear_max,
        "clean_residual_max_abs": clean_max,
        "front_gate_nonzero_possible": bool(front_gate_mean > 0.0),
        "front_gate_init_mean": front_gate_mean,
        "rear_gate_init_mean": rear_gate_mean,
        "clean_gate_init_mean": clean_gate_mean,
    }
