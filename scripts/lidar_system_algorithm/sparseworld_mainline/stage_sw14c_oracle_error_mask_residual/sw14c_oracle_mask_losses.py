from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


EMPTY_IDX = 17


@dataclass(frozen=True)
class OracleMaskLossConfig:
    lambda_add_fn: float = 4.0
    lambda_sup_fp: float = 6.0
    lambda_correct_occ: float = 1.0
    lambda_correct_free: float = 0.7
    lambda_mask_out: float = 4.0
    lambda_density: float = 4.0
    lambda_fp: float = 4.0
    lambda_res_bound: float = 2.0
    lambda_res_l1_inmask: float = 0.02
    lambda_smooth: float = 0.05
    lambda_clean: float = 2.0
    lambda_rear: float = 2.0
    residual_bound_value: float = 0.05
    density_margin: float = 0.02
    front_local_target: float = 1.30
    fp_delta_margin: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LOSS_TERMS_SCHEMA = [
    {"term": "L_add_branch_FN_recover", "region": "mask_add_{variant}", "weight_key": "lambda_add_fn"},
    {"term": "L_suppress_branch_FP_suppress", "region": "mask_sup_{variant}", "weight_key": "lambda_sup_fp"},
    {"term": "L_correct_occ_preserve", "region": "teacher_correct_occ_region", "weight_key": "lambda_correct_occ"},
    {"term": "L_correct_free_preserve", "region": "teacher_correct_free_region", "weight_key": "lambda_correct_free"},
    {"term": "L_mask_out_output_preserve", "region": "outside oracle/proxy mask", "weight_key": "lambda_mask_out"},
    {"term": "L_density_hinge", "region": "dense occupancy proxy", "weight_key": "lambda_density"},
    {"term": "L_fp_penalty", "region": "GT free", "weight_key": "lambda_fp"},
    {"term": "L_residual_bound", "region": "feature residual", "weight_key": "lambda_res_bound"},
    {"term": "L_residual_l1_inside_mask_proxy", "region": "front triplet masked features", "weight_key": "lambda_res_l1_inmask"},
    {"term": "L_residual_smooth", "region": "feature residual", "weight_key": "lambda_smooth"},
    {"term": "L_clean_consistency", "region": "clean A0 structural no-op", "weight_key": "lambda_clean"},
    {"term": "L_rear_consistency", "region": "rear cameras structural no-op", "weight_key": "lambda_rear"},
]


def masked_bce(prob: torch.Tensor, mask: torch.Tensor, target: float) -> torch.Tensor:
    mask = mask.to(device=prob.device, dtype=torch.bool)
    if not bool(mask.any().detach().item()):
        return prob.new_zeros(())
    return F.binary_cross_entropy(prob[mask].clamp(1e-4, 1.0 - 1e-4), torch.full_like(prob[mask], float(target)))


def smoothness_loss(feature_map: torch.Tensor) -> torch.Tensor:
    dh = (feature_map[..., 1:, :] - feature_map[..., :-1, :]).abs().mean() if feature_map.shape[-2] > 1 else feature_map.new_zeros(())
    dw = (feature_map[..., :, 1:] - feature_map[..., :, :-1]).abs().mean() if feature_map.shape[-1] > 1 else feature_map.new_zeros(())
    return dh + dw


def debug_residual_terms(debug: dict[str, torch.Tensor], config: OracleMaskLossConfig, device: torch.device) -> dict[str, torch.Tensor]:
    add_deltas = [v for k, v in debug.items() if k.startswith("add_delta_level")]
    sup_deltas = [v for k, v in debug.items() if k.startswith("sup_delta_level")]
    add_gates = [v for k, v in debug.items() if k.startswith("add_gate_level")]
    sup_gates = [v for k, v in debug.items() if k.startswith("sup_gate_level")]
    if not add_deltas or not sup_deltas:
        z = torch.zeros((), device=device)
        return {"res_l1_inmask": z, "smooth": z, "rear_loss": z, "clean_loss": z, "res_bound": z}
    all_delta = add_deltas + sup_deltas
    res_l1 = torch.stack([d.abs().mean() for d in all_delta]).mean()
    smooth = torch.stack([smoothness_loss(d) for d in all_delta]).mean()
    rear = torch.stack([d[:, 3:].abs().mean() for d in all_delta]).mean()
    bound = torch.stack([torch.relu(d.abs() - float(config.residual_bound_value)).pow(2).mean() for d in all_delta]).mean()
    clean = torch.zeros((), device=device)
    gate_mean = torch.stack([g.mean() for g in add_gates + sup_gates]).mean() if add_gates and sup_gates else torch.zeros((), device=device)
    return {"res_l1_inmask": res_l1, "smooth": smooth, "rear_loss": rear, "clean_loss": clean, "res_bound": bound, "gate_mean": gate_mean}


def compute_oracle_mask_loss(
    per_h: dict[int, dict[str, torch.Tensor]],
    debug: dict[str, torch.Tensor],
    masks_by_h: dict[int, dict[str, torch.Tensor]],
    teacher_bundle: dict[str, Any],
    sectors: dict[str, torch.Tensor],
    config: OracleMaskLossConfig,
    *,
    mask_variant: str = "dilated",
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(iter(per_h.values()))["dense_scores"].device
    add_losses: list[torch.Tensor] = []
    sup_losses: list[torch.Tensor] = []
    corr_occ_losses: list[torch.Tensor] = []
    corr_free_losses: list[torch.Tensor] = []
    mask_out_losses: list[torch.Tensor] = []
    density_losses: list[torch.Tensor] = []
    fp_losses: list[torch.Tensor] = []
    front_losses: list[torch.Tensor] = []
    for horizon_s, entry in per_h.items():
        masks = masks_by_h[int(horizon_s)]
        dense_scores = entry["dense_scores"].float()
        occ_prob = dense_scores.max(dim=-1).values.clamp(1e-4, 1.0 - 1.0e-4)
        gt_occ = masks["gt_occ"].to(device=device, dtype=torch.bool)
        gt_free = masks["gt_free"].to(device=device, dtype=torch.bool)
        teacher_occ = masks["teacher_occ"].to(device=device, dtype=torch.bool)
        if mask_variant == "strict":
            add_mask = masks["mask_add_strict"]
            sup_mask = masks["mask_sup_strict"]
            allowed = masks["oracle_error_mask_strict"]
        elif mask_variant == "proxy":
            add_mask = masks["mask_add_proxy"]
            sup_mask = masks["mask_sup_proxy"]
            allowed = masks["proxy_error_mask_no_gt"]
        else:
            add_mask = masks["mask_add_dilated"]
            sup_mask = masks["mask_sup_dilated"]
            allowed = masks["oracle_error_mask_dilated"]
        add_losses.append(masked_bce(occ_prob, add_mask, 1.0))
        sup_losses.append(masked_bce(occ_prob, sup_mask, 0.0))
        corr_occ_losses.append(masked_bce(occ_prob, masks["teacher_correct_occ_region"], 1.0))
        corr_free_losses.append(masked_bce(occ_prob, masks["teacher_correct_free_region"], 0.0))
        outside = ~allowed.to(device=device, dtype=torch.bool)
        if bool(outside.any().detach().item()):
            target = teacher_occ.float()
            mask_out_losses.append(F.binary_cross_entropy(occ_prob[outside], target[outside]))
        teacher_h = teacher_bundle["by_horizon"][int(horizon_s)]
        teacher_eval = teacher_h["teacher_meta"]["final_eval"]
        teacher_density_delta = float(teacher_eval.get("pred_gt_density_delta", teacher_eval.get("pred_gt_occupied_ratio_delta", 0.0)))
        target_density = gt_occ.float().mean() + min(teacher_density_delta + float(config.density_margin), 0.08)
        density_losses.append(torch.relu(occ_prob.mean() - target_density).pow(2))
        if bool(gt_free.any().detach().item()):
            fp_losses.append(occ_prob[gt_free].mean())
        front_mask = sectors["front"].to(device=device, dtype=torch.bool)
        native_front = (teacher_h["native_semantic"].long() != EMPTY_IDX) & sectors["front"].bool()
        front_proxy = occ_prob[front_mask].sum() / occ_prob.new_tensor(max(1.0, float(native_front.sum().item())))
        front_losses.append(torch.relu(front_proxy - float(config.front_local_target)).pow(2))
    def mean(xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(xs).mean() if xs else torch.zeros((), device=device)
    reg = debug_residual_terms(debug, config, device)
    terms = {
        "add_fn_loss": mean(add_losses),
        "sup_fp_loss": mean(sup_losses),
        "correct_occ_loss": mean(corr_occ_losses),
        "correct_free_loss": mean(corr_free_losses),
        "mask_out_loss": mean(mask_out_losses),
        "density_loss": mean(density_losses),
        "fp_loss": mean(fp_losses),
        "front_proxy_loss": mean(front_losses),
        **reg,
    }
    total = (
        config.lambda_add_fn * terms["add_fn_loss"]
        + config.lambda_sup_fp * terms["sup_fp_loss"]
        + config.lambda_correct_occ * terms["correct_occ_loss"]
        + config.lambda_correct_free * terms["correct_free_loss"]
        + config.lambda_mask_out * terms["mask_out_loss"]
        + config.lambda_density * terms["density_loss"]
        + config.lambda_fp * terms["fp_loss"]
        + config.lambda_res_bound * terms["res_bound"]
        + config.lambda_res_l1_inmask * terms["res_l1_inmask"]
        + config.lambda_smooth * terms["smooth"]
        + config.lambda_clean * terms["clean_loss"]
        + config.lambda_rear * terms["rear_loss"]
    )
    stats = {k: float(v.detach().item()) for k, v in terms.items()}
    stats["total_loss"] = float(total.detach().item())
    return total, stats
