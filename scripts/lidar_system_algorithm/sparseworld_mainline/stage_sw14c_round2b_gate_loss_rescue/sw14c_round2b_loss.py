from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


EMPTY_IDX = 17


@dataclass(frozen=True)
class Round2BLossConfig:
    lambda_fn: float = 4.0
    lambda_fp: float = 4.0
    lambda_corr_occ: float = 1.0
    lambda_corr_free: float = 0.5
    lambda_density: float = 4.0
    lambda_front_proxy: float = 2.0
    lambda_res_l1: float = 0.05
    lambda_gate_sparse: float = 0.02
    lambda_res_smooth: float = 0.05
    lambda_clean: float = 2.0
    lambda_rear: float = 2.0
    lambda_bound: float = 2.0
    density_margin: float = 0.02
    density_max_delta: float = 0.08
    front_local_target: float = 1.30
    fp_delta_margin: float = 0.01
    residual_bound_value: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LOSS_TERMS_SCHEMA = [
    {"term": "L_teacher_FN_recover", "region": "M_recover", "weight_key": "lambda_fn"},
    {"term": "L_teacher_FP_suppress", "region": "M_suppress", "weight_key": "lambda_fp"},
    {"term": "L_correct_occ_preserve", "region": "teacher_correct_occ_region", "weight_key": "lambda_corr_occ"},
    {"term": "L_correct_free_preserve", "region": "teacher_correct_free_region", "weight_key": "lambda_corr_free"},
    {"term": "L_density_hinge", "region": "dense occupancy proxy", "weight_key": "lambda_density"},
    {"term": "L_front_local_proxy_hinge", "region": "front sector dense occupancy proxy", "weight_key": "lambda_front_proxy"},
    {"term": "L_residual_l1", "region": "masked residual deltas", "weight_key": "lambda_res_l1"},
    {"term": "L_gate_sparse", "region": "masked residual gates", "weight_key": "lambda_gate_sparse"},
    {"term": "L_residual_smooth", "region": "masked residual deltas", "weight_key": "lambda_res_smooth"},
    {"term": "L_clean_consistency", "region": "clean mask structural no-op", "weight_key": "lambda_clean"},
    {"term": "L_rear_consistency", "region": "rear camera structural no-op", "weight_key": "lambda_rear"},
    {"term": "L_residual_bound", "region": "residual delta bound", "weight_key": "lambda_bound"},
]


def safe_mean(values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device)
    return torch.stack(values).mean()


def masked_bce(prob: torch.Tensor, mask: torch.Tensor, target: float) -> torch.Tensor:
    mask = mask.to(device=prob.device, dtype=torch.bool)
    if not bool(mask.any().detach().item()):
        return prob.new_zeros(())
    target_tensor = torch.full_like(prob[mask], float(target))
    return F.binary_cross_entropy(prob[mask].clamp(1e-4, 1.0 - 1e-4), target_tensor)


def smoothness_loss(feature_map: torch.Tensor) -> torch.Tensor:
    dh = (feature_map[..., 1:, :] - feature_map[..., :-1, :]).abs().mean() if feature_map.shape[-2] > 1 else feature_map.new_zeros(())
    dw = (feature_map[..., :, 1:] - feature_map[..., :, :-1]).abs().mean() if feature_map.shape[-1] > 1 else feature_map.new_zeros(())
    return dh + dw


def residual_terms(debug: dict[str, torch.Tensor], config: Round2BLossConfig, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    deltas = [value for key, value in debug.items() if key.startswith("residual_delta_level")]
    gates = [value for key, value in debug.items() if key.startswith("residual_gate_level")]
    if not deltas:
        zero = torch.zeros((), device=device)
        tensors = {
            "residual_l1": zero,
            "residual_smooth": zero,
            "gate_sparse": zero,
            "clean_loss": zero,
            "rear_loss": zero,
            "bound_loss": zero,
        }
        return tensors, {key: 0.0 for key in tensors}
    residual_l1 = torch.stack([delta.abs().mean() for delta in deltas]).mean()
    residual_smooth = torch.stack([smoothness_loss(delta) for delta in deltas]).mean()
    gate_sparse = torch.stack([gate.mean() for gate in gates]).mean() if gates else torch.zeros((), device=device)
    rear_loss = torch.stack([delta[:, 3:].abs().mean() for delta in deltas]).mean()
    clean_loss = torch.zeros((), device=device)
    bound_loss = torch.stack([torch.relu(delta.abs() - float(config.residual_bound_value)).pow(2).mean() for delta in deltas]).mean()
    tensors = {
        "residual_l1": residual_l1,
        "residual_smooth": residual_smooth,
        "gate_sparse": gate_sparse,
        "clean_loss": clean_loss,
        "rear_loss": rear_loss,
        "bound_loss": bound_loss,
    }
    floats = {key: float(value.detach().item()) for key, value in tensors.items()}
    return tensors, floats


def compute_round2b_loss(
    per_h: dict[int, dict[str, torch.Tensor]],
    debug: dict[str, torch.Tensor],
    masks_by_h: dict[int, dict[str, torch.Tensor]],
    teacher_bundle: dict[str, Any],
    sectors: dict[str, torch.Tensor],
    config: Round2BLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(iter(per_h.values()))["dense_scores"].device
    fn_losses: list[torch.Tensor] = []
    fp_losses: list[torch.Tensor] = []
    corr_occ_losses: list[torch.Tensor] = []
    corr_free_losses: list[torch.Tensor] = []
    density_losses: list[torch.Tensor] = []
    front_losses: list[torch.Tensor] = []
    fp_hinge_losses: list[torch.Tensor] = []
    front_mask_cpu = sectors["front"].bool()
    for horizon_s, entry in per_h.items():
        dense_scores = entry["dense_scores"].float()
        occ_prob = dense_scores.max(dim=-1).values.clamp(1e-4, 1.0 - 1.0e-4)
        masks = masks_by_h[int(horizon_s)]
        gt_occ = masks["gt_occ"].to(device=device, dtype=torch.bool)
        gt_free = masks["gt_free"].to(device=device, dtype=torch.bool)
        teacher_h = teacher_bundle["by_horizon"][int(horizon_s)]
        teacher_eval = teacher_h["teacher_meta"]["final_eval"]
        teacher_density_delta = float(
            teacher_eval.get(
                "pred_gt_density_delta",
                teacher_eval.get("pred_gt_occupied_ratio_delta", teacher_eval.get("pred_gt_occupied_ratio", 0.0)),
            )
        )
        target_delta = min(teacher_density_delta + float(config.density_margin), float(config.density_max_delta))
        target_density = gt_occ.float().mean() + occ_prob.new_tensor(target_delta)
        density_losses.append(torch.relu(occ_prob.mean() - target_density).pow(2))
        front_mask = front_mask_cpu.to(device=device, dtype=torch.bool)
        native_front_occ = (teacher_h["native_semantic"].long() != EMPTY_IDX) & front_mask_cpu
        native_front_count = max(1.0, float(native_front_occ.sum().item()))
        front_proxy = occ_prob[front_mask].sum() / occ_prob.new_tensor(native_front_count)
        front_losses.append(torch.relu(front_proxy - float(config.front_local_target)).pow(2))
        teacher_fp_delta = float(teacher_eval.get("false_positive_delta", 0.0))
        fp_ratio = occ_prob[gt_free].mean() if bool(gt_free.any().item()) else occ_prob.new_zeros(())
        fp_hinge_losses.append(torch.relu(fp_ratio - occ_prob.new_tensor(teacher_fp_delta + float(config.fp_delta_margin))).pow(2))
        fn_losses.append(masked_bce(occ_prob, masks["M_recover"], 1.0))
        fp_losses.append(masked_bce(occ_prob, masks["M_suppress"], 0.0))
        corr_occ_losses.append(masked_bce(occ_prob, masks["teacher_correct_occ_region"], 1.0))
        corr_free_losses.append(masked_bce(occ_prob, masks["teacher_correct_free_region"], 0.0))
    loss_tensors: dict[str, torch.Tensor] = {
        "teacher_FN_recover_loss": safe_mean(fn_losses, device),
        "teacher_FP_suppress_loss": safe_mean(fp_losses, device),
        "correct_occ_preserve_loss": safe_mean(corr_occ_losses, device),
        "correct_free_preserve_loss": safe_mean(corr_free_losses, device),
        "density_loss": safe_mean(density_losses, device),
        "front_proxy_loss": safe_mean(front_losses, device),
        "fp_hinge_loss": safe_mean(fp_hinge_losses, device),
    }
    reg_tensors, reg_stats = residual_terms(debug, config, device)
    loss_tensors.update(reg_tensors)
    total = (
        config.lambda_fn * loss_tensors["teacher_FN_recover_loss"]
        + config.lambda_fp * loss_tensors["teacher_FP_suppress_loss"]
        + config.lambda_corr_occ * loss_tensors["correct_occ_preserve_loss"]
        + config.lambda_corr_free * loss_tensors["correct_free_preserve_loss"]
        + config.lambda_density * loss_tensors["density_loss"]
        + config.lambda_front_proxy * loss_tensors["front_proxy_loss"]
        + config.lambda_fp * loss_tensors["fp_hinge_loss"]
        + config.lambda_res_l1 * loss_tensors["residual_l1"]
        + config.lambda_gate_sparse * loss_tensors["gate_sparse"]
        + config.lambda_res_smooth * loss_tensors["residual_smooth"]
        + config.lambda_clean * loss_tensors["clean_loss"]
        + config.lambda_rear * loss_tensors["rear_loss"]
        + config.lambda_bound * loss_tensors["bound_loss"]
    )
    stats = {key: float(value.detach().item()) for key, value in loss_tensors.items()}
    stats.update(reg_stats)
    stats["total_loss"] = float(total.detach().item())
    return total, stats
