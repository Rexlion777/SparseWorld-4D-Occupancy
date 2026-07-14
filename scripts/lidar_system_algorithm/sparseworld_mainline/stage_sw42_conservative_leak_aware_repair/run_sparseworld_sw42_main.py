"""Stage SW-4.2 conservative leak-aware aggregation repair.

Conservative non-oracle get_occ repair ablations built on top of SW-4.1.
This stage keeps the corrected all48 support premise and searches for
Pareto-safer inference-time repair candidates with explicit false-positive
guards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_SW41_DIR = SCRIPT_DIR.parent / "stage_sw41_exact_contributor_soft_aggregation"
STAGE_SW4_DIR = SCRIPT_DIR.parent / "stage_sw4_covered_ff_semantic_activation"
STAGE_SW2_DIR = SCRIPT_DIR.parent / "stage_sw2_temporal_query_diagnosis"
sys.path.insert(0, str(STAGE_SW41_DIR))
sys.path.insert(0, str(STAGE_SW4_DIR))
sys.path.insert(0, str(STAGE_SW2_DIR))

import run_sparseworld_sw41_main as sw41  # type: ignore
import get_occ_replay_and_ablation as sw41ab  # type: ignore

PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair"

SW41_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"

EMPTY_IDX = sw41ab.EMPTY_IDX
SMALL_OBJECT_IDS = sw41ab.SMALL_OBJECT_IDS
DYNAMIC_IDS = sw41ab.DYNAMIC_IDS

STRICT_SAFE = {
    "false_free_rate_delta_max": -0.0200,
    "false_occupied_rate_delta_max": +0.0060,
    "occupied_iou_delta_min": -0.0030,
    "semantic_miou_delta_min": -0.0030,
    "pred_gt_occupied_ratio_delta_max": +0.1000,
}
MODERATE_SAFE = {
    "false_free_rate_delta_max": -0.0300,
    "false_occupied_rate_delta_max": +0.0100,
    "occupied_iou_delta_min": -0.0060,
    "semantic_miou_delta_min": -0.0050,
    "pred_gt_occupied_ratio_delta_max": +0.1600,
}
REJECT_CRITERIA = {
    "false_occupied_rate_delta_min": +0.0150,
    "occupied_iou_delta_max": -0.0100,
    "pred_gt_occupied_ratio_delta_min": +0.2500,
    "semantic_miou_delta_max": -0.0100,
}
BONUS = {
    "small_object_false_free_delta_max": -0.0300,
    "new_visible_recall_delta_min": +0.0300,
    "dynamic_false_free_delta_max": -0.0200,
}


@dataclass
class ConservativeSpec:
    name: str
    gate_cfg: sw41ab.GateConfig
    assign_cfg: sw41ab.AssignConfig
    agg_cfg: sw41ab.AggregateConfig
    neighbor_cap: float | None = None
    require_target_free: bool = False
    boundary_neighbor_min: int | None = None
    boundary_kernel: int = 3
    max_target_contrib: int | None = None
    confidence_thr: float | None = None
    margin_min: float | None = None
    margin_max: float | None = None
    entropy_min: float | None = None
    entropy_max: float | None = None
    density_min: int | None = None
    density_max: int | None = None
    future_only: bool = False
    horizon_cap_mode: str = "constant"
    class_filter: str = "all"
    topk_per_point: int | None = None
    require_geo_support: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage SW-4.2 conservative leak-aware repair")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--repo-root", default=str(sw41.DEFAULT_REPO_ROOT))
    parser.add_argument("--config", default=str(sw41.DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(sw41.DEFAULT_CHECKPOINT))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--horizons", default="0,1,2,3,4,5,6")
    parser.add_argument("--run-baseline", action="store_true", default=True)
    parser.add_argument("--run-k7", action="store_true", default=True)
    parser.add_argument("--run-conservative", action="store_true", default=True)
    parser.add_argument("--run-validation", action="store_true", default=True)
    parser.add_argument("--save-debug", action="store_true", default=True)
    parser.add_argument("--save-figures", action="store_true", default=True)
    parser.add_argument("--diag-samples", type=int, default=5)
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[SW42] {msg}", flush=True)


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, LOGS_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_horizons(text: str) -> list[int]:
    return sw41.parse_horizons(text)


def load_records(num_samples: int, sample_indices: list[int] | None = None) -> list[dict[str, Any]]:
    return sw41.load_records(num_samples, sample_indices)


def build_bev_neighbor_count(mask_xyz: torch.Tensor, kernel_xy: int = 3) -> torch.Tensor:
    bev = mask_xyz.any(dim=2).float().T.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, kernel_xy, kernel_xy), device=bev.device, dtype=bev.dtype)
    pad = kernel_xy // 2
    out = F.conv2d(bev, kernel, padding=pad).squeeze(0).squeeze(0).T
    return out


def build_3d_neighbor_count(mask_xyz: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    x = mask_xyz.float().permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    w = torch.ones((1, 1, kernel, kernel, kernel), device=x.device, dtype=x.dtype)
    pad = kernel // 2
    out = F.conv3d(x, w, padding=pad).squeeze(0).squeeze(0).permute(2, 1, 0)
    return out


def build_geo_mask(head: Any, point_info: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = torch.zeros(tuple(int(x.item()) for x in head.voxel_num), dtype=torch.bool, device=point_info["decoded_points"].device)
    valid = point_info["valid_range_mask_flat"]
    vox = point_info["pre_voxel_index_flat"][valid].long()
    if vox.numel():
        mask[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    return mask


def k7_variant() -> tuple[sw41ab.GateConfig, sw41ab.AssignConfig, sw41ab.AggregateConfig]:
    lib = sw41ab.get_default_variant_library()
    gate = lib["G1_margin_020"][0]
    assign = lib["V2_bev_soft_splat_r2"][1]
    agg = lib["C2_small_object_temperature_scaling_20"][2]
    return gate, assign, agg


def to_cpu_report_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().tolist()
        else:
            out[k] = v
    return out


def build_baseline_context(head: Any, pred_dict: dict[str, torch.Tensor], horizon_s: int) -> dict[str, Any]:
    gate_cfg, assign_cfg, agg_cfg = sw41ab.get_default_variant_library()["V0_hard_baseline"]
    replay = sw41ab.replay_get_occ_variant(head, pred_dict, horizon_s, gate_cfg, assign_cfg, agg_cfg)
    pred_occ = replay["output"]["occ_pred"] != EMPTY_IDX
    geo_mask = build_geo_mask(head, replay["point_info"])
    ctx = {
        "baseline_replay": replay,
        "pred_occ": pred_occ,
        "contributor_count_dense": replay["output"]["contributor_count_dense"],
        "bev_neighbor_count3": build_bev_neighbor_count(pred_occ, kernel_xy=3),
        "bev_neighbor_count5": build_bev_neighbor_count(pred_occ, kernel_xy=5),
        "neighbor_count3d": build_3d_neighbor_count(pred_occ, kernel=3),
        "geo_mask": geo_mask,
    }
    return ctx


def _topk_class_mask(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "all":
        return torch.ones(scores.shape[0], device=scores.device, dtype=torch.bool)
    topk = torch.topk(scores, k=min(3, scores.shape[-1]), dim=-1).indices
    mask = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.bool)
    if mode == "small":
        for cid in SMALL_OBJECT_IDS:
            mask |= (topk == cid).any(dim=-1)
    elif mode == "dynamic_small":
        for cid in sorted(set(SMALL_OBJECT_IDS + DYNAMIC_IDS)):
            mask |= (topk == cid).any(dim=-1)
    else:
        raise ValueError(f"unknown class filter: {mode}")
    return mask


def _neighbor_cap_for_horizon(spec: ConservativeSpec, horizon_s: int) -> float | None:
    if spec.neighbor_cap is None:
        return None
    if spec.horizon_cap_mode == "constant":
        return spec.neighbor_cap
    if spec.horizon_cap_mode == "future_decay":
        if horizon_s <= 0:
            return spec.neighbor_cap
        return max(0.0, spec.neighbor_cap * (1.0 - 0.08 * horizon_s))
    if spec.horizon_cap_mode == "future_increase":
        if horizon_s <= 0:
            return spec.neighbor_cap
        return spec.neighbor_cap * (1.0 + 0.05 * horizon_s)
    return spec.neighbor_cap


def _apply_topk_per_point(assignment: dict[str, torch.Tensor], keep_mask: torch.Tensor, exact_mask: torch.Tensor, topk: int) -> torch.Tensor:
    if topk is None or topk <= 0:
        return keep_mask
    extra_mask = (~exact_mask) & keep_mask
    if not bool(extra_mask.any().item()):
        return keep_mask
    point_ids = assignment["point_flat_index"][extra_mask]
    weights = assignment["point_weight"][extra_mask]
    keep_extra = torch.zeros_like(point_ids, dtype=torch.bool)
    unique_ids = torch.unique(point_ids)
    for uid in unique_ids.tolist():
        idx = torch.nonzero(point_ids == uid, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        k = min(topk, idx.numel())
        chosen = idx[torch.topk(weights[idx], k=k).indices]
        keep_extra[chosen] = True
    out = keep_mask.clone()
    out[extra_mask] = keep_extra
    return out


def apply_conservative_postprocess(
    head: Any,
    assignment: dict[str, torch.Tensor],
    spec: ConservativeSpec,
    baseline_ctx: dict[str, Any],
    horizon_s: int,
) -> dict[str, torch.Tensor]:
    if spec.name in {"K0_baseline", "K7_all_combined"}:
        return assignment

    if assignment["target_voxel_indices"].numel() == 0:
        return assignment

    target = assignment["target_voxel_indices"].long()
    point_metric = assignment["point_metric"]
    scores = assignment["point_scores"]
    best_score = assignment["point_best_score"]
    margin = assignment["point_margin"]
    entropy = assignment["point_entropy"]
    density = assignment["point_local_density"]
    cont_xyz = (point_metric - head.pc_range[:3]) / head.voxel_size
    base_vox = torch.floor(cont_xyz).long()
    exact_mask = (target == base_vox).all(dim=-1)
    keep = torch.ones(target.shape[0], device=target.device, dtype=torch.bool)

    if spec.future_only and horizon_s < 1:
        keep &= exact_mask

    if spec.require_target_free:
        pred_occ = baseline_ctx["pred_occ"]
        keep &= exact_mask | (~pred_occ[target[:, 0], target[:, 1], target[:, 2]])

    if spec.boundary_neighbor_min is not None:
        counts = baseline_ctx["bev_neighbor_count3"] if spec.boundary_kernel == 3 else baseline_ctx["bev_neighbor_count5"]
        target_xy = counts[target[:, 0], target[:, 1]]
        keep &= exact_mask | (target_xy >= spec.boundary_neighbor_min)

    if spec.max_target_contrib is not None:
        contrib = baseline_ctx["contributor_count_dense"][target[:, 0], target[:, 1], target[:, 2]]
        keep &= exact_mask | (contrib <= spec.max_target_contrib)

    if spec.confidence_thr is not None:
        keep &= exact_mask | (best_score >= spec.confidence_thr)
    if spec.margin_min is not None:
        keep &= exact_mask | (margin >= spec.margin_min)
    if spec.margin_max is not None:
        keep &= exact_mask | (margin <= spec.margin_max)
    if spec.entropy_min is not None:
        keep &= exact_mask | (entropy >= spec.entropy_min)
    if spec.entropy_max is not None:
        keep &= exact_mask | (entropy <= spec.entropy_max)
    if spec.density_min is not None:
        keep &= exact_mask | (density >= spec.density_min)
    if spec.density_max is not None:
        keep &= exact_mask | (density <= spec.density_max)

    if spec.class_filter != "all":
        cls_mask = _topk_class_mask(scores, spec.class_filter)
        keep &= exact_mask | cls_mask

    if spec.require_geo_support:
        geo = baseline_ctx["geo_mask"]
        keep &= exact_mask | geo[target[:, 0], target[:, 1], target[:, 2]]

    weights = assignment["point_weight"].clone()
    neighbor_cap = _neighbor_cap_for_horizon(spec, horizon_s)
    if neighbor_cap is not None:
        neighbor_mask = ~exact_mask
        weights[neighbor_mask] = torch.clamp(weights[neighbor_mask], max=neighbor_cap)

    keep = _apply_topk_per_point(assignment, keep, exact_mask, spec.topk_per_point or 0)

    out: dict[str, torch.Tensor] = {}
    for key, value in assignment.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == keep.shape[0]:
            out[key] = value[keep]
        else:
            out[key] = value
    if "point_weight" in out:
        out["point_weight"] = weights[keep]
    return out


def replay_conservative_variant(
    head: Any,
    pred_dict: dict[str, torch.Tensor],
    horizon_s: int,
    spec: ConservativeSpec,
    baseline_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    point_info = sw41ab.decode_support_points(head, pred_dict)
    gate_info = sw41ab.compute_gate_mask(head, point_info, spec.gate_cfg, horizon_s)
    flat = sw41ab._flatten_points(point_info, gate_info)
    assignment = sw41ab.assign_points_to_voxels_soft(head, flat, spec.assign_cfg)
    if baseline_ctx is not None:
        assignment = apply_conservative_postprocess(head, assignment, spec, baseline_ctx, horizon_s)
    agg = sw41ab.aggregate_voxel_scores_hard(head, assignment, spec.agg_cfg)
    occ = sw41ab.produce_semantic_occ(head, agg)
    return {
        "point_info": point_info,
        "gate_info": gate_info,
        "assignment": assignment,
        "aggregation": agg,
        "output": occ,
        "variant_meta": {
            "name": spec.name,
            "gate": asdict(spec.gate_cfg),
            "assign": asdict(spec.assign_cfg),
            "aggregate": asdict(spec.agg_cfg),
            "conservative": {
                "neighbor_cap": spec.neighbor_cap,
                "require_target_free": spec.require_target_free,
                "boundary_neighbor_min": spec.boundary_neighbor_min,
                "boundary_kernel": spec.boundary_kernel,
                "max_target_contrib": spec.max_target_contrib,
                "confidence_thr": spec.confidence_thr,
                "margin_min": spec.margin_min,
                "margin_max": spec.margin_max,
                "entropy_min": spec.entropy_min,
                "entropy_max": spec.entropy_max,
                "density_min": spec.density_min,
                "density_max": spec.density_max,
                "future_only": spec.future_only,
                "horizon_cap_mode": spec.horizon_cap_mode,
                "class_filter": spec.class_filter,
                "topk_per_point": spec.topk_per_point,
                "require_geo_support": spec.require_geo_support,
            },
        },
    }


def conservative_variant_library() -> dict[str, ConservativeSpec]:
    lib0 = sw41ab.get_default_variant_library()
    hard_gate, hard_assign, hard_agg = lib0["V0_hard_baseline"]
    v2_gate, v2_assign, _ = lib0["V2_bev_soft_splat_r2"]
    v1_gate, v1_assign, _ = lib0["V1_bev_soft_splat_r1"]
    v3_gate, v3_assign, _ = lib0["V3_xyz_soft_splat_3x3x3"]
    v4_gate, v4_assign, _ = lib0["V4_z_soft_only"]
    c2_gate, _, c2_agg = lib0["C2_small_object_temperature_scaling_20"]

    lib: dict[str, ConservativeSpec] = {}
    lib["K0_baseline"] = ConservativeSpec("K0_baseline", hard_gate, hard_assign, hard_agg)
    lib["K7_all_combined"] = ConservativeSpec("K7_all_combined", *k7_variant())

    # Phase 3 conservative soft
    lib["S0_hard_baseline"] = ConservativeSpec("S0_hard_baseline", hard_gate, hard_assign, hard_agg)
    lib["S1_bev_r1_cap015"] = ConservativeSpec("S1_bev_r1_cap015", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15)
    lib["S2_bev_r1_cap025"] = ConservativeSpec("S2_bev_r1_cap025", hard_gate, v1_assign, hard_agg, neighbor_cap=0.25)
    lib["S3_bev_r1_cap035"] = ConservativeSpec("S3_bev_r1_cap035", hard_gate, v1_assign, hard_agg, neighbor_cap=0.35)
    lib["S4_bev_r2_cap010"] = ConservativeSpec("S4_bev_r2_cap010", hard_gate, sw41ab.AssignConfig(name="S4_bev_r2_cap010_assign", mode="bev_soft", radius_xy=2, sigma_xy=0.65), hard_agg, neighbor_cap=0.10)
    lib["S5_bev_r2_cap015"] = ConservativeSpec("S5_bev_r2_cap015", hard_gate, sw41ab.AssignConfig(name="S5_bev_r2_cap015_assign", mode="bev_soft", radius_xy=2, sigma_xy=0.65), hard_agg, neighbor_cap=0.15)
    lib["S6_z_soft_cap020"] = ConservativeSpec("S6_z_soft_cap020", hard_gate, sw41ab.AssignConfig(name="S6_z_soft_cap020_assign", mode="z_soft_only", radius_z=1, sigma_z=0.75), hard_agg, neighbor_cap=0.20)
    lib["S7_xyz_r1_cap015"] = ConservativeSpec("S7_xyz_r1_cap015", hard_gate, sw41ab.AssignConfig(name="S7_xyz_r1_cap015_assign", mode="xyz_soft", radius_xy=1, radius_z=1, sigma_xy=0.75, sigma_z=0.75), hard_agg, neighbor_cap=0.15)
    lib["S8_top1_neighbor_only"] = ConservativeSpec("S8_top1_neighbor_only", hard_gate, sw41ab.AssignConfig(name="S8_top1_neighbor_only_assign", mode="bev_soft", radius_xy=1, sigma_xy=0.75), hard_agg, neighbor_cap=0.20, topk_per_point=1)
    lib["S9_top2_neighbor_cap015"] = ConservativeSpec("S9_top2_neighbor_cap015", hard_gate, sw41ab.AssignConfig(name="S9_top2_neighbor_cap015_assign", mode="bev_soft", radius_xy=1, sigma_xy=0.75), hard_agg, neighbor_cap=0.15, topk_per_point=2)

    # Phase 4 leak-aware
    lib["L1_boundary_r1_cap015"] = ConservativeSpec("L1_boundary_r1_cap015", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0)
    lib["L2_boundary_r1_cap025"] = ConservativeSpec("L2_boundary_r1_cap025", hard_gate, v1_assign, hard_agg, neighbor_cap=0.25, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0)
    lib["L3_boundary_r2_cap010"] = ConservativeSpec("L3_boundary_r2_cap010", hard_gate, sw41ab.AssignConfig(name="L3_boundary_r2_cap010_assign", mode="bev_soft", radius_xy=2, sigma_xy=0.65), hard_agg, neighbor_cap=0.10, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0)
    lib["L4_neighbor_leak_reassign_top1"] = ConservativeSpec("L4_neighbor_leak_reassign_top1", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, topk_per_point=1)
    lib["L5_neighbor_leak_reassign_top2"] = ConservativeSpec("L5_neighbor_leak_reassign_top2", hard_gate, v1_assign, hard_agg, neighbor_cap=0.10, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, topk_per_point=2)
    lib["L6_hole_fill_guarded"] = ConservativeSpec("L6_hole_fill_guarded", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=3, max_target_contrib=0)
    lib["L7_confidence_guarded_boundary_04"] = ConservativeSpec("L7_confidence_guarded_boundary_04", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, confidence_thr=0.4)
    lib["L7_confidence_guarded_boundary_05"] = ConservativeSpec("L7_confidence_guarded_boundary_05", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, confidence_thr=0.5)
    lib["L7_confidence_guarded_boundary_06"] = ConservativeSpec("L7_confidence_guarded_boundary_06", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, confidence_thr=0.6)
    lib["L7_confidence_guarded_boundary_07"] = ConservativeSpec("L7_confidence_guarded_boundary_07", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, confidence_thr=0.7)
    lib["L8_entropy_guarded_boundary"] = ConservativeSpec("L8_entropy_guarded_boundary", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, entropy_min=1.5)
    lib["L9_dynamic_boundary_only"] = ConservativeSpec("L9_dynamic_boundary_only", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, class_filter="dynamic_small")

    # Phase 5 false-positive guards
    base_guard_assign = sw41ab.AssignConfig(name="FPG_base_assign", mode="bev_soft", radius_xy=1, sigma_xy=0.75)
    lib["FPG1_neighbor_count_guard_2"] = ConservativeSpec("FPG1_neighbor_count_guard_2", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0)
    lib["FPG1_neighbor_count_guard_3"] = ConservativeSpec("FPG1_neighbor_count_guard_3", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=3, max_target_contrib=0)
    lib["FPG1_neighbor_count_guard_4"] = ConservativeSpec("FPG1_neighbor_count_guard_4", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=4, max_target_contrib=0)
    lib["FPG2_confidence_guard_04"] = ConservativeSpec("FPG2_confidence_guard_04", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, confidence_thr=0.4)
    lib["FPG2_confidence_guard_05"] = ConservativeSpec("FPG2_confidence_guard_05", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, confidence_thr=0.5)
    lib["FPG2_confidence_guard_06"] = ConservativeSpec("FPG2_confidence_guard_06", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, confidence_thr=0.6)
    lib["FPG3_margin_guard"] = ConservativeSpec("FPG3_margin_guard", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, margin_min=0.05, margin_max=0.40)
    lib["FPG4_density_guard"] = ConservativeSpec("FPG4_density_guard", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, density_min=1, density_max=12)
    lib["FPG5_horizon_guard"] = ConservativeSpec("FPG5_horizon_guard", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, future_only=False, horizon_cap_mode="future_decay")
    lib["FPG6_geo_guard"] = ConservativeSpec("FPG6_geo_guard", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, require_geo_support=True)
    lib["FPG7_combined_guard"] = ConservativeSpec("FPG7_combined_guard", hard_gate, base_guard_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=3, max_target_contrib=0, confidence_thr=0.5, density_min=1, density_max=10)

    # Phase 6 small-object local
    lib["SO1_small_topk_boundary_cap015"] = ConservativeSpec("SO1_small_topk_boundary_cap015", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, class_filter="small")
    lib["SO2_small_topk_boundary_cap025"] = ConservativeSpec("SO2_small_topk_boundary_cap025", hard_gate, v1_assign, hard_agg, neighbor_cap=0.25, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, class_filter="small")
    lib["SO3_small_topk_local_r1"] = ConservativeSpec("SO3_small_topk_local_r1", hard_gate, v1_assign, hard_agg, neighbor_cap=0.18, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, class_filter="small", topk_per_point=1)
    lib["SO4_small_topk_class_preserve_plus_guard"] = ConservativeSpec("SO4_small_topk_class_preserve_plus_guard", hard_gate, v1_assign, sw41ab.AggregateConfig(name="SO4_small_preserve", mode="small_preserve", preserve_small_topk=3), neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, class_filter="small", confidence_thr=0.5, density_min=1, density_max=10)
    lib["SO5_small_object_future_only"] = ConservativeSpec("SO5_small_object_future_only", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, class_filter="small", future_only=True)
    lib["SO6_small_object_no_open_space"] = ConservativeSpec("SO6_small_object_no_open_space", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, class_filter="small", confidence_thr=0.5)

    # Phase 7 new-visible future
    lib["NV1_future_r1_cap015"] = ConservativeSpec("NV1_future_r1_cap015", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, max_target_contrib=0, future_only=True)
    lib["NV2_future_boundary_cap015"] = ConservativeSpec("NV2_future_boundary_cap015", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, future_only=True)
    lib["NV3_future_leak_aware_cap015"] = ConservativeSpec("NV3_future_leak_aware_cap015", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, future_only=True, topk_per_point=1)
    lib["NV4_future_confidence_guard"] = ConservativeSpec("NV4_future_confidence_guard", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, future_only=True, confidence_thr=0.5)
    lib["NV5_future_geo_guard"] = ConservativeSpec("NV5_future_geo_guard", hard_gate, v1_assign, hard_agg, neighbor_cap=0.15, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, future_only=True, require_geo_support=True)
    lib["NV6_future_horizon_decay"] = ConservativeSpec("NV6_future_horizon_decay", hard_gate, v1_assign, hard_agg, neighbor_cap=0.20, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, future_only=True, horizon_cap_mode="future_decay")
    lib["NV7_future_horizon_increase"] = ConservativeSpec("NV7_future_horizon_increase", hard_gate, v1_assign, hard_agg, neighbor_cap=0.10, require_target_free=True, boundary_neighbor_min=2, max_target_contrib=0, future_only=True, horizon_cap_mode="future_increase")
    return lib


def run_named_variant_set(
    head: Any,
    records: list[dict[str, Any]],
    horizons: list[int],
    variant_names: list[str],
    spec_lib: dict[str, ConservativeSpec],
    phase_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(records) * len(horizons) * len(variant_names)
    done = 0
    with torch.inference_mode():
        for record in records:
            for horizon_s in horizons:
                pred_dict = sw41.extract_pred_dict_for_horizon(record["query_artifact"], horizon_s)
                baseline_ctx = build_baseline_context(head, pred_dict, horizon_s)
                for name in variant_names:
                    spec = spec_lib[name]
                    start = time.perf_counter()
                    if name == "K0_baseline":
                        replay = baseline_ctx["baseline_replay"]
                    else:
                        replay = replay_conservative_variant(head, pred_dict, horizon_s, spec, baseline_ctx=baseline_ctx)
                    replay["head"] = head
                    runtime_ms = (time.perf_counter() - start) * 1000.0
                    rows.append(sw41.build_variant_metric_row(record, horizon_s, name, replay, runtime_ms))
                    done += 1
                    if done % 10 == 0 or done == total:
                        log(f"{phase_name}: {done}/{total} variant sample-horizons complete")
    return rows


def metric_row_map(agg_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["variant_name"]): r for r in agg_rows}


def check_replay_equivalence(baseline_rows: list[dict[str, Any]], k7_rows: list[dict[str, Any]], expected_full_samples: int) -> dict[str, Any]:
    sw41_decision = json.loads((SW41_REPORTS_DIR / "sw41_repair_candidate_decision.json").read_text(encoding="utf-8"))
    sw41_k7 = sw41_decision["validation_row"]
    k7_agg = sw41.aggregate_rows(sw41.attach_deltas(k7_rows, "K0_baseline"))
    k7_map = metric_row_map(k7_agg)
    base_agg = sw41.aggregate_rows(sw41.attach_deltas(baseline_rows, "K0_baseline"))
    base_map = metric_row_map(base_agg)
    k0 = base_map["K0_baseline"]
    k7 = k7_map["K7_all_combined"]
    checks = {
        "baseline_mean_occupied_iou": float(k0["mean_occupied_iou"]),
        "k7_mean_false_free_rate_delta": float(k7["mean_false_free_rate_delta"]),
        "k7_mean_false_occupied_rate_delta": float(k7["mean_false_occupied_rate_delta"]),
        "k7_mean_occupied_iou_delta": float(k7["mean_occupied_iou_delta"]),
        "k7_mean_pred_gt_occupied_ratio_delta": float(k7["mean_pred_gt_occupied_ratio_delta"]),
        "sw41_k7_reference": {
            "false_free_rate_delta": float(sw41_k7["mean_false_free_rate_delta"]),
            "false_occupied_rate_delta": float(sw41_k7["mean_false_occupied_rate_delta"]),
            "occupied_iou_delta": float(sw41_k7["mean_occupied_iou_delta"]),
            "pred_gt_occupied_ratio_delta": float(sw41_k7["mean_pred_gt_occupied_ratio_delta"]),
        },
    }
    strict_reference_check = expected_full_samples >= 20 and len({int(r["sample_index"]) for r in baseline_rows}) >= expected_full_samples
    if strict_reference_check:
        passed = (
            abs(checks["k7_mean_false_free_rate_delta"] - checks["sw41_k7_reference"]["false_free_rate_delta"]) <= 0.01
            and abs(checks["k7_mean_false_occupied_rate_delta"] - checks["sw41_k7_reference"]["false_occupied_rate_delta"]) <= 0.006
            and abs(checks["k7_mean_occupied_iou_delta"] - checks["sw41_k7_reference"]["occupied_iou_delta"]) <= 0.006
            and abs(checks["k7_mean_pred_gt_occupied_ratio_delta"] - checks["sw41_k7_reference"]["pred_gt_occupied_ratio_delta"]) <= 0.08
        )
    else:
        passed = (
            math.isfinite(checks["baseline_mean_occupied_iou"])
            and math.isfinite(checks["k7_mean_false_free_rate_delta"])
            and math.isfinite(checks["k7_mean_false_occupied_rate_delta"])
            and math.isfinite(checks["k7_mean_occupied_iou_delta"])
            and math.isfinite(checks["k7_mean_pred_gt_occupied_ratio_delta"])
        )
    checks["strict_reference_check"] = strict_reference_check
    checks["passed"] = passed
    return checks


def pareto_label(row: dict[str, Any]) -> str:
    ff = float(row.get("mean_false_free_rate_delta", 0.0))
    fo = float(row.get("mean_false_occupied_rate_delta", 0.0))
    oi = float(row.get("mean_occupied_iou_delta", 0.0))
    sm = float(row.get("mean_semantic_miou_delta", 0.0))
    pr = float(row.get("mean_pred_gt_occupied_ratio_delta", 0.0))
    if ff <= STRICT_SAFE["false_free_rate_delta_max"] and fo <= STRICT_SAFE["false_occupied_rate_delta_max"] and oi >= STRICT_SAFE["occupied_iou_delta_min"] and sm >= STRICT_SAFE["semantic_miou_delta_min"] and pr <= STRICT_SAFE["pred_gt_occupied_ratio_delta_max"]:
        return "strict-safe"
    if ff <= MODERATE_SAFE["false_free_rate_delta_max"] and fo <= MODERATE_SAFE["false_occupied_rate_delta_max"] and oi >= MODERATE_SAFE["occupied_iou_delta_min"] and sm >= MODERATE_SAFE["semantic_miou_delta_min"] and pr <= MODERATE_SAFE["pred_gt_occupied_ratio_delta_max"]:
        return "moderate-safe"
    if fo > REJECT_CRITERIA["false_occupied_rate_delta_min"] or oi < REJECT_CRITERIA["occupied_iou_delta_max"] or pr > REJECT_CRITERIA["pred_gt_occupied_ratio_delta_min"] or sm < REJECT_CRITERIA["semantic_miou_delta_max"]:
        return "reject"
    return "gray-zone"


def candidate_score(row: dict[str, Any]) -> float:
    ff_gain = max(0.0, -float(row.get("mean_false_free_rate_delta", 0.0)))
    fo_pen = max(0.0, float(row.get("mean_false_occupied_rate_delta", 0.0))) * 4.0
    iou = float(row.get("mean_occupied_iou_delta", 0.0)) * 3.0
    sem = float(row.get("mean_semantic_miou_delta", 0.0)) * 2.0
    pred_pen = max(0.0, float(row.get("mean_pred_gt_occupied_ratio_delta", 0.0)) - 0.08) * 2.5
    small_bonus = max(0.0, -float(row.get("mean_small_object_false_free_delta", 0.0))) * 0.6
    nv_bonus = max(0.0, float(row.get("mean_new_visible_recall_delta", 0.0))) * 0.6
    runtime_pen = max(0.0, float(row.get("mean_runtime_ms", 0.0)) - 60.0) / 150.0
    label_bonus = {"strict-safe": 0.30, "moderate-safe": 0.15, "gray-zone": 0.0, "reject": -0.50}[pareto_label(row)]
    return ff_gain + iou + sem + small_bonus + nv_bonus - fo_pen - pred_pen - runtime_pen + label_bonus


def select_best(rows: list[dict[str, Any]], prefer_metric: str | None = None) -> dict[str, Any] | None:
    if not rows:
        return None
    if prefer_metric is not None:
        rows = sorted(rows, key=lambda r: float(r.get(prefer_metric, 0.0)), reverse=True)
    return sorted(rows, key=candidate_score, reverse=True)[0]


def promote_rows(agg_rows: list[dict[str, Any]], phase: str) -> list[str]:
    promoted: list[str] = []
    for row in agg_rows:
        name = str(row["variant_name"])
        if name.endswith("baseline"):
            continue
        ff = float(row.get("mean_false_free_rate_delta", 0.0))
        fo = float(row.get("mean_false_occupied_rate_delta", 0.0))
        oi = float(row.get("mean_occupied_iou_delta", 0.0))
        if phase == "soft":
            if ff <= -0.015 and fo <= +0.010 and oi >= -0.006:
                promoted.append(name)
        else:
            if ff <= -0.012 and fo <= +0.010 and oi >= -0.006:
                promoted.append(name)
    return promoted


def save_group_outputs(base_name: str, rows5: list[dict[str, Any]], rows20: list[dict[str, Any]] | None = None, summary_text: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agg5 = sw41.aggregate_rows(rows5)
    write_csv(REPORTS_DIR / f"{base_name}_5sample.csv", rows5)
    write_json(REPORTS_DIR / f"{base_name}_5sample.json", {"rows": rows5, "aggregate": agg5})
    if rows20 is None:
        rows20 = []
    agg20 = sw41.aggregate_rows(rows20) if rows20 else []
    if rows20:
        write_csv(REPORTS_DIR / f"{base_name}_20sample.csv", rows20)
        write_json(REPORTS_DIR / f"{base_name}_20sample.json", {"rows": rows20, "aggregate": agg20})
    write_md(REPORTS_DIR / f"{base_name.replace('_metrics','')}_summary.md", summary_text)
    return agg5, agg20


def plot_pareto(fig_path: Path, agg_rows: list[dict[str, Any]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in agg_rows:
        x = float(row.get("mean_false_occupied_rate_delta", 0.0))
        y = float(row.get("mean_false_free_rate_delta", 0.0))
        ax.scatter([x], [y], label=row["variant_name"])
        ax.text(x, y, row["variant_name"], fontsize=7)
    ax.axvline(0.006, color="green", linestyle="--", alpha=0.4)
    ax.axvline(0.010, color="orange", linestyle="--", alpha=0.4)
    ax.axhline(-0.020, color="green", linestyle="--", alpha=0.4)
    ax.axhline(-0.030, color="orange", linestyle="--", alpha=0.4)
    ax.set_xlabel("false_occupied_rate_delta")
    ax.set_ylabel("false_free_rate_delta")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def write_acceptance_files() -> None:
    payload = {
        "strict_safe": STRICT_SAFE,
        "moderate_safe": MODERATE_SAFE,
        "reject": REJECT_CRITERIA,
        "bonus": BONUS,
    }
    write_json(REPORTS_DIR / "sw42_pareto_acceptance_criteria.json", payload)
    write_md(
        REPORTS_DIR / "sw42_pareto_acceptance_criteria.md",
        "# SW-4.2 Pareto-safe acceptance criteria\n\n"
        f"- strict_safe: {json.dumps(STRICT_SAFE, ensure_ascii=False)}\n"
        f"- moderate_safe: {json.dumps(MODERATE_SAFE, ensure_ascii=False)}\n"
        f"- reject: {json.dumps(REJECT_CRITERIA, ensure_ascii=False)}\n"
        f"- bonus: {json.dumps(BONUS, ensure_ascii=False)}\n",
    )


def find_row(agg_rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in agg_rows:
        if str(row["variant_name"]) == name:
            return row
    return None


def rows_to_summary_lines(rows: list[dict[str, Any]], names: list[str]) -> str:
    parts = []
    for name in names:
        row = find_row(rows, name)
        if row is None:
            continue
        parts.append(
            f"- {name}: ff_delta={float(row.get('mean_false_free_rate_delta', 0.0)):.4f}, "
            f"fo_delta={float(row.get('mean_false_occupied_rate_delta', 0.0)):.4f}, "
            f"occ_iou_delta={float(row.get('mean_occupied_iou_delta', 0.0)):.4f}, "
            f"pred_ratio_delta={float(row.get('mean_pred_gt_occupied_ratio_delta', 0.0)):.4f}, "
            f"label={pareto_label(row)}"
        )
    return "\n".join(parts)


def choose_case(best_safe: dict[str, Any] | None, best_overall: dict[str, Any] | None, k7_row: dict[str, Any]) -> tuple[str, str]:
    if best_safe is not None and pareto_label(best_safe) in {"strict-safe", "moderate-safe"}:
        return "C1", "conservative repair accepted"
    if best_overall is not None:
        ff = float(best_overall.get("mean_false_free_rate_delta", 0.0))
        fo = float(best_overall.get("mean_false_occupied_rate_delta", 0.0))
        k7_fo = float(k7_row.get("mean_false_occupied_rate_delta", 0.0))
        if ff <= -0.02 and fo < k7_fo and pareto_label(best_overall) != "reject":
            return "C2", "repair direction valid but needs guard tuning"
        safeish = fo <= 0.010 and float(best_overall.get("mean_pred_gt_occupied_ratio_delta", 0.0)) <= 0.160
        if safeish and ff > -0.015:
            return "C3", "all safe guards too weak"
        if ff <= -0.03 and fo > 0.015:
            return "C4", "all effective candidates produce false-positive explosion"
    return "C5", "deeper model-level or training-level repair required"


def main() -> int:
    args = parse_args()
    ensure_dirs()
    horizons = parse_horizons(args.horizons)
    sample_indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
    write_acceptance_files()

    log("loading SW-4.1 subset")
    records20 = load_records(args.num_samples, sample_indices or None)
    records5 = records20[: min(args.diag_samples, len(records20))]

    log("building SparseWorld model/head once for SW-4.2")
    cfg, model, checkpoint = sw41.build_sparseworld_model(Path(args.repo_root), Path(args.config), Path(args.checkpoint))
    head = sw41.get_pts_bbox_head(model)

    spec_lib = conservative_variant_library()
    sample_manifest_rows: list[dict[str, Any]] = []
    for record in records20:
        for horizon_s in horizons:
            sample_manifest_rows.append(
                {
                    "sample_index": record["sample_index"],
                    "sample_token": record["sample_token"],
                    "scene_token": record["scene_token"],
                    "horizon_s": horizon_s,
                    "baseline_status": "pending",
                    "k7_status": "pending",
                    "conservative_status": "pending",
                    "validation_status": "pending",
                    "failure_traceback": "",
                }
            )

    # Phase 1 baseline/K7 replay
    log("Phase 1: baseline/K7 replay")
    baseline_rows = run_named_variant_set(head, records20, horizons, ["K0_baseline"], spec_lib, "baseline")
    k7_rows = sw41.attach_deltas(run_named_variant_set(head, records20, horizons, ["K0_baseline", "K7_all_combined"], spec_lib, "k7"), "K0_baseline")
    replay_eq = check_replay_equivalence(baseline_rows, k7_rows, expected_full_samples=args.num_samples)
    write_json(REPORTS_DIR / "baseline_k7_replay_equivalence.json", replay_eq)
    if not replay_eq["passed"]:
        log("baseline/K7 replay equivalence failed; aborting SW-4.2")
        return 2

    for row in sample_manifest_rows:
        row["baseline_status"] = "success"
        row["k7_status"] = "success"

    # Phase groups on first 5 samples
    soft_keys = ["S0_hard_baseline", "S1_bev_r1_cap015", "S2_bev_r1_cap025", "S3_bev_r1_cap035", "S4_bev_r2_cap010", "S5_bev_r2_cap015", "S6_z_soft_cap020", "S7_xyz_r1_cap015", "S8_top1_neighbor_only", "S9_top2_neighbor_cap015"]
    leak_keys = ["L1_boundary_r1_cap015", "L2_boundary_r1_cap025", "L3_boundary_r2_cap010", "L4_neighbor_leak_reassign_top1", "L5_neighbor_leak_reassign_top2", "L6_hole_fill_guarded", "L7_confidence_guarded_boundary_04", "L7_confidence_guarded_boundary_05", "L7_confidence_guarded_boundary_06", "L7_confidence_guarded_boundary_07", "L8_entropy_guarded_boundary", "L9_dynamic_boundary_only"]
    guard_keys = ["FPG1_neighbor_count_guard_2", "FPG1_neighbor_count_guard_3", "FPG1_neighbor_count_guard_4", "FPG2_confidence_guard_04", "FPG2_confidence_guard_05", "FPG2_confidence_guard_06", "FPG3_margin_guard", "FPG4_density_guard", "FPG5_horizon_guard", "FPG6_geo_guard", "FPG7_combined_guard"]
    small_keys = ["SO1_small_topk_boundary_cap015", "SO2_small_topk_boundary_cap025", "SO3_small_topk_local_r1", "SO4_small_topk_class_preserve_plus_guard", "SO5_small_object_future_only", "SO6_small_object_no_open_space"]
    nv_keys = ["NV1_future_r1_cap015", "NV2_future_boundary_cap015", "NV3_future_leak_aware_cap015", "NV4_future_confidence_guard", "NV5_future_geo_guard", "NV6_future_horizon_decay", "NV7_future_horizon_increase"]

    log("Phase 3: conservative soft splat")
    soft_rows5 = sw41.attach_deltas(run_named_variant_set(head, records5, horizons, soft_keys, spec_lib, "soft5"), "S0_hard_baseline")
    soft_agg5 = sw41.aggregate_rows(soft_rows5)
    soft_promoted = promote_rows(soft_agg5, "soft")
    soft_rows20 = sw41.attach_deltas(run_named_variant_set(head, records20, horizons, soft_promoted, spec_lib, "soft20"), "S0_hard_baseline") if soft_promoted else []
    soft_agg20 = sw41.aggregate_rows(soft_rows20) if soft_rows20 else []
    write_csv(REPORTS_DIR / "conservative_soft_splat_metrics_5sample.csv", soft_rows5)
    write_json(REPORTS_DIR / "conservative_soft_splat_metrics_5sample.json", {"rows": soft_rows5, "aggregate": soft_agg5})
    if soft_rows20:
        write_csv(REPORTS_DIR / "conservative_soft_splat_metrics_20sample.csv", soft_rows20)
        write_json(REPORTS_DIR / "conservative_soft_splat_metrics_20sample.json", {"rows": soft_rows20, "aggregate": soft_agg20})
    write_md(REPORTS_DIR / "conservative_soft_splat_summary.md", "# Conservative soft splat summary\n\n" + rows_to_summary_lines(soft_agg5, soft_keys))
    sw41.plot_metric_panel(FIGURES_DIR / "conservative_soft_splat_metric_panel.png", soft_agg5, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "Conservative soft splat")
    plot_pareto(FIGURES_DIR / "soft_splat_pareto_false_free_vs_false_occupied.png", soft_agg5, "Conservative soft splat pareto")

    log("Phase 4: leak-aware local repair")
    leak_rows5 = sw41.attach_deltas(run_named_variant_set(head, records5, horizons, leak_keys, spec_lib, "leak5"), "K0_baseline")
    leak_agg5 = sw41.aggregate_rows(leak_rows5)
    leak_promoted = promote_rows(leak_agg5, "leak")
    leak_rows20 = sw41.attach_deltas(run_named_variant_set(head, records20, horizons, leak_promoted, spec_lib, "leak20"), "K0_baseline") if leak_promoted else []
    leak_agg20 = sw41.aggregate_rows(leak_rows20) if leak_rows20 else []
    write_csv(REPORTS_DIR / "leak_aware_repair_metrics_5sample.csv", leak_rows5)
    write_json(REPORTS_DIR / "leak_aware_repair_metrics_5sample.json", {"rows": leak_rows5, "aggregate": leak_agg5})
    if leak_rows20:
        write_csv(REPORTS_DIR / "leak_aware_repair_metrics_20sample.csv", leak_rows20)
        write_json(REPORTS_DIR / "leak_aware_repair_metrics_20sample.json", {"rows": leak_rows20, "aggregate": leak_agg20})
    write_md(REPORTS_DIR / "leak_aware_repair_summary.md", "# Leak-aware repair summary\n\n" + rows_to_summary_lines(leak_agg5, leak_keys))
    sw41.plot_metric_panel(FIGURES_DIR / "leak_aware_pareto_panel.png", leak_agg5, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "Leak-aware repair")

    log("Phase 5: false-positive guards")
    guard_rows5 = sw41.attach_deltas(run_named_variant_set(head, records5, horizons, guard_keys, spec_lib, "guard5"), "K0_baseline")
    guard_agg5 = sw41.aggregate_rows(guard_rows5)
    write_csv(REPORTS_DIR / "false_positive_guard_metrics.csv", guard_rows5)
    write_json(REPORTS_DIR / "false_positive_guard_metrics.json", {"rows": guard_rows5, "aggregate": guard_agg5})
    write_md(REPORTS_DIR / "false_positive_guard_summary.md", "# False-positive guard summary\n\n" + rows_to_summary_lines(guard_agg5, guard_keys))
    sw41.plot_metric_panel(FIGURES_DIR / "false_positive_guard_tradeoff.png", guard_agg5, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("pred_gt_occupied_ratio", "pred/gt ratio")], "False-positive guards")

    log("Phase 6: small-object local repair")
    small_rows5 = sw41.attach_deltas(run_named_variant_set(head, records5, horizons, small_keys, spec_lib, "small5"), "K0_baseline")
    small_agg5 = sw41.aggregate_rows(small_rows5)
    write_csv(REPORTS_DIR / "small_object_local_repair_metrics.csv", small_rows5)
    write_json(REPORTS_DIR / "small_object_local_repair_metrics.json", {"rows": small_rows5, "aggregate": small_agg5})
    write_md(REPORTS_DIR / "small_object_local_repair_summary.md", "# Small-object local repair summary\n\n" + rows_to_summary_lines(small_agg5, small_keys))
    sw41.plot_metric_panel(FIGURES_DIR / "small_object_local_repair_panel.png", small_agg5, [("small_object_false_free", "small-object FF"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "Small-object local repair")

    log("Phase 7: new-visible conservative future repair")
    nv_rows5 = sw41.attach_deltas(run_named_variant_set(head, records5, horizons, nv_keys, spec_lib, "nv5"), "K0_baseline")
    nv_agg5 = sw41.aggregate_rows(nv_rows5)
    write_csv(REPORTS_DIR / "new_visible_conservative_repair_metrics.csv", nv_rows5)
    write_json(REPORTS_DIR / "new_visible_conservative_repair_metrics.json", {"rows": nv_rows5, "aggregate": nv_agg5})
    write_md(REPORTS_DIR / "new_visible_conservative_repair_summary.md", "# New-visible conservative repair summary\n\n" + rows_to_summary_lines(nv_agg5, nv_keys))
    sw41.plot_metric_panel(FIGURES_DIR / "new_visible_horizon_repair_panel.png", nv_agg5, [("new_visible_recall", "new-visible recall"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "New-visible conservative repair")

    # Best candidates from groups
    soft_best = select_best(soft_agg20 or soft_agg5)
    leak_best = select_best(leak_agg20 or leak_agg5)
    guard_best = select_best(guard_agg5)
    small_best = select_best(small_agg5, prefer_metric="mean_new_visible_recall_delta")
    small_best = select_best(small_agg5) if small_best is None else small_best
    nv_best = select_best(nv_agg5, prefer_metric="mean_new_visible_recall_delta")

    log("Phase 8: combined conservative candidates")
    combo_specs: dict[str, ConservativeSpec] = {
        "C0_baseline": spec_lib["K0_baseline"],
    }

    def clone_spec(name: str, base: ConservativeSpec, **overrides: Any) -> ConservativeSpec:
        data = {
            "name": name,
            "gate_cfg": base.gate_cfg,
            "assign_cfg": base.assign_cfg,
            "agg_cfg": base.agg_cfg,
            "neighbor_cap": base.neighbor_cap,
            "require_target_free": base.require_target_free,
            "boundary_neighbor_min": base.boundary_neighbor_min,
            "boundary_kernel": base.boundary_kernel,
            "max_target_contrib": base.max_target_contrib,
            "confidence_thr": base.confidence_thr,
            "margin_min": base.margin_min,
            "margin_max": base.margin_max,
            "entropy_min": base.entropy_min,
            "entropy_max": base.entropy_max,
            "density_min": base.density_min,
            "density_max": base.density_max,
            "future_only": base.future_only,
            "horizon_cap_mode": base.horizon_cap_mode,
            "class_filter": base.class_filter,
            "topk_per_point": base.topk_per_point,
            "require_geo_support": base.require_geo_support,
        }
        data.update(overrides)
        return ConservativeSpec(**data)

    if soft_best is not None:
        combo_specs["C1_best_conservative_soft"] = clone_spec("C1_best_conservative_soft", spec_lib[str(soft_best["variant_name"])])
    if leak_best is not None:
        combo_specs["C2_best_leak_aware"] = clone_spec("C2_best_leak_aware", spec_lib[str(leak_best["variant_name"])])
    if guard_best is not None:
        combo_specs["C3_best_guarded_soft"] = clone_spec("C3_best_guarded_soft", spec_lib[str(guard_best["variant_name"])])
    if small_best is not None:
        combo_specs["C4_best_small_object_local"] = clone_spec("C4_best_small_object_local", spec_lib[str(small_best["variant_name"])])
    if nv_best is not None:
        combo_specs["C5_best_new_visible_future"] = clone_spec("C5_best_new_visible_future", spec_lib[str(nv_best["variant_name"])])
    if leak_best is not None and guard_best is not None:
        base = spec_lib[str(leak_best["variant_name"])]
        guard = spec_lib[str(guard_best["variant_name"])]
        combo_specs["C6_leak_aware_plus_guard"] = clone_spec(
            "C6_leak_aware_plus_guard",
            base,
            confidence_thr=max(base.confidence_thr or 0.0, guard.confidence_thr or 0.0) or None,
            boundary_neighbor_min=max(base.boundary_neighbor_min or 0, guard.boundary_neighbor_min or 0) or None,
            density_min=guard.density_min if guard.density_min is not None else base.density_min,
            density_max=guard.density_max if guard.density_max is not None else base.density_max,
            require_geo_support=base.require_geo_support or guard.require_geo_support,
        )
    if leak_best is not None and small_best is not None:
        base = spec_lib[str(leak_best["variant_name"])]
        combo_specs["C7_leak_aware_plus_small"] = clone_spec("C7_leak_aware_plus_small", base, class_filter="small", agg_cfg=sw41ab.AggregateConfig(name="C7_small_preserve", mode="small_preserve", preserve_small_topk=3))
    if leak_best is not None and nv_best is not None:
        base = spec_lib[str(leak_best["variant_name"])]
        combo_specs["C8_leak_aware_plus_new_visible"] = clone_spec("C8_leak_aware_plus_new_visible", base, future_only=True, horizon_cap_mode="future_decay")
    if guard_best is not None and small_best is not None:
        base = spec_lib[str(guard_best["variant_name"])]
        combo_specs["C9_guarded_soft_plus_small"] = clone_spec("C9_guarded_soft_plus_small", base, class_filter="small", agg_cfg=sw41ab.AggregateConfig(name="C9_small_preserve", mode="small_preserve", preserve_small_topk=3))
    if guard_best is not None and nv_best is not None:
        base = spec_lib[str(guard_best["variant_name"])]
        combo_specs["C10_guarded_soft_plus_new_visible"] = clone_spec("C10_guarded_soft_plus_new_visible", base, future_only=True, horizon_cap_mode="future_decay")
    if leak_best is not None and guard_best is not None and small_best is not None and nv_best is not None:
        base = spec_lib[str(leak_best["variant_name"])]
        combo_specs["C11_best_all_conservative"] = clone_spec(
            "C11_best_all_conservative",
            base,
            confidence_thr=max(base.confidence_thr or 0.0, spec_lib[str(guard_best["variant_name"])].confidence_thr or 0.0) or None,
            boundary_neighbor_min=max(base.boundary_neighbor_min or 0, spec_lib[str(guard_best["variant_name"])].boundary_neighbor_min or 0) or None,
            class_filter="small",
            future_only=True,
            horizon_cap_mode="future_decay",
            agg_cfg=sw41ab.AggregateConfig(name="C11_small_preserve", mode="small_preserve", preserve_small_topk=3),
        )

    combo_rows5 = sw41.attach_deltas(run_named_variant_set(head, records5, horizons, list(combo_specs.keys()), combo_specs, "combo5"), "C0_baseline")
    combo_agg5 = sw41.aggregate_rows(combo_rows5)
    combo_promoted = [str(r["variant_name"]) for r in combo_agg5 if str(r["variant_name"]) != "C0_baseline" and float(r.get("mean_false_occupied_rate_delta", 0.0)) <= 0.012]
    combo_rows20 = sw41.attach_deltas(run_named_variant_set(head, records20, horizons, ["C0_baseline"] + combo_promoted, combo_specs, "combo20"), "C0_baseline")
    combo_agg20 = sw41.aggregate_rows(combo_rows20)
    write_csv(REPORTS_DIR / "combined_conservative_candidate_metrics_5sample.csv", combo_rows5)
    write_json(REPORTS_DIR / "combined_conservative_candidate_metrics_5sample.json", {"rows": combo_rows5, "aggregate": combo_agg5})
    write_csv(REPORTS_DIR / "combined_conservative_candidate_metrics_20sample.csv", combo_rows20)
    write_json(REPORTS_DIR / "combined_conservative_candidate_metrics_20sample.json", {"rows": combo_rows20, "aggregate": combo_agg20})
    sw41.plot_metric_panel(FIGURES_DIR / "combined_conservative_metric_panel.png", combo_agg20, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "Combined conservative candidates")
    plot_pareto(FIGURES_DIR / "combined_conservative_pareto_panel.png", combo_agg20, "Combined conservative pareto")

    combo_nonbase = [r for r in combo_agg20 if str(r["variant_name"]) != "C0_baseline"]
    combo_nonbase.sort(key=candidate_score, reverse=True)
    best_safe_candidate = next((r for r in combo_nonbase if pareto_label(r) in {"strict-safe", "moderate-safe"}), None)
    best_overall_candidate = combo_nonbase[0] if combo_nonbase else None
    best_small_candidate = max(combo_nonbase, key=lambda r: max(0.0, -float(r.get("mean_small_object_false_free_delta", 0.0))), default=None)
    best_nv_candidate = max(combo_nonbase, key=lambda r: float(r.get("mean_new_visible_recall_delta", 0.0)), default=None)
    rejected = [str(r["variant_name"]) for r in combo_nonbase if pareto_label(r) == "reject"]
    write_md(
        REPORTS_DIR / "combined_conservative_candidate_selection.md",
        "# Combined conservative candidate selection\n\n"
        f"- best_safe_candidate: {best_safe_candidate['variant_name'] if best_safe_candidate else 'none'}\n"
        f"- best_overall_candidate: {best_overall_candidate['variant_name'] if best_overall_candidate else 'none'}\n"
        f"- best_small_object_candidate: {best_small_candidate['variant_name'] if best_small_candidate else 'none'}\n"
        f"- best_new_visible_candidate: {best_nv_candidate['variant_name'] if best_nv_candidate else 'none'}\n"
        f"- rejected_unsafe_candidates: {rejected}\n",
    )

    # Phase 9 20-sample Pareto validation
    log("Phase 9: 20-sample Pareto validation")
    validation_candidates = ["K0_baseline", "K7_all_combined"]
    for row in [best_safe_candidate, best_overall_candidate, best_small_candidate, best_nv_candidate]:
        if row is not None and str(row["variant_name"]) not in validation_candidates:
            validation_candidates.append(str(row["variant_name"]))
    validation_spec_lib = {**spec_lib, **combo_specs}
    val_rows = sw41.attach_deltas(run_named_variant_set(head, records20, horizons, validation_candidates, validation_spec_lib, "pareto20"), "K0_baseline")
    val_agg = sw41.aggregate_rows(val_rows)
    val_map = metric_row_map(val_agg)
    k7_row = val_map["K7_all_combined"]
    validation_json = {"rows": val_rows, "aggregate": val_agg}
    write_csv(REPORTS_DIR / "sw42_20sample_pareto_validation.csv", val_rows)
    write_json(REPORTS_DIR / "sw42_20sample_pareto_validation.json", validation_json)
    write_md(REPORTS_DIR / "sw42_20sample_pareto_validation_summary.md", "# SW-4.2 20-sample Pareto validation\n\n" + rows_to_summary_lines(val_agg, validation_candidates))
    plot_pareto(FIGURES_DIR / "pareto_false_free_vs_false_occupied.png", [r for r in val_agg if r["variant_name"] != "K0_baseline"], "SW-4.2 20-sample Pareto")
    sw41.plot_metric_panel(FIGURES_DIR / "best_candidate_20sample_metric_panel.png", val_agg, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "SW-4.2 20-sample validation")

    for row in val_agg:
        if row["variant_name"] != "K7_all_combined":
            row["ff_delta_vs_k7"] = float(row.get("mean_false_free_rate", 0.0) - k7_row.get("mean_false_free_rate", 0.0))
            row["fo_delta_vs_k7"] = float(row.get("mean_false_occupied_rate", 0.0) - k7_row.get("mean_false_occupied_rate", 0.0))
            row["occ_iou_delta_vs_k7"] = float(row.get("mean_occupied_iou", 0.0) - k7_row.get("mean_occupied_iou", 0.0))
            row["pred_ratio_delta_vs_k7"] = float(row.get("mean_pred_gt_occupied_ratio", 0.0) - k7_row.get("mean_pred_gt_occupied_ratio", 0.0))

    best_safe_row = val_map.get(best_safe_candidate["variant_name"]) if best_safe_candidate is not None else None
    best_overall_row = val_map.get(best_overall_candidate["variant_name"]) if best_overall_candidate is not None else None
    case, reason = choose_case(best_safe_row, best_overall_row, k7_row)
    decision = {
        "case": case,
        "reason": reason,
        "best_safe_candidate": best_safe_row["variant_name"] if best_safe_row else None,
        "best_overall_candidate": best_overall_row["variant_name"] if best_overall_row else None,
        "best_small_object_candidate": best_small_candidate["variant_name"] if best_small_candidate else None,
        "best_new_visible_candidate": best_nv_candidate["variant_name"] if best_nv_candidate else None,
        "best_safe_row": best_safe_row,
        "best_overall_row": best_overall_row,
        "k7_reference_row": k7_row,
        "rejected_unsafe_candidates": rejected,
    }
    write_json(REPORTS_DIR / "sw42_conservative_repair_decision.json", decision)
    write_md(
        REPORTS_DIR / "sw42_conservative_repair_decision.md",
        "# SW-4.2 conservative repair decision\n\n"
        f"- case: {case}\n"
        f"- reason: {reason}\n"
        f"- best_safe_candidate: {decision['best_safe_candidate']}\n"
        f"- best_overall_candidate: {decision['best_overall_candidate']}\n"
        f"- rejected_unsafe_candidates: {rejected}\n",
    )

    # manifests and report
    write_json(
        REPORTS_DIR / "sw42_run_manifest.json",
        {
            "effective_samples": len(records20),
            "diagnostic_samples": len(records5),
            "effective_horizons": horizons,
            "baseline_k7_replay_passed": replay_eq["passed"],
            "soft_promoted": soft_promoted,
            "leak_promoted": leak_promoted,
            "combo_promoted": combo_promoted,
            "case": case,
        },
    )
    write_csv(REPORTS_DIR / "sw42_sample_manifest.csv", sample_manifest_rows)

    report = {
        "stage": "SW-4.2",
        "effective_samples": len(records20),
        "diagnostic_samples": len(records5),
        "effective_horizons": horizons,
        "baseline_k7_replay_equivalence": replay_eq,
        "pareto_acceptance": {
            "strict_safe": STRICT_SAFE,
            "moderate_safe": MODERATE_SAFE,
            "reject": REJECT_CRITERIA,
            "bonus": BONUS,
        },
        "best_conservative_candidate": decision["best_safe_candidate"] or decision["best_overall_candidate"],
        "decision": decision,
        "safe_claims": [
            "subset diagnostic repair ablation",
            "non-oracle inference-time candidate",
            "not official benchmark",
            "no training",
            "not deployment-ready",
            "K7 aggressive is rejected",
        ],
        "next_unique_action": (
            "Stage SW-4.3 implement best candidate as clean patch + rerun stable subset diagnosis"
            if case == "C1"
            else (
                "Stage SW-4.3 guard tuning / threshold sweep"
                if case == "C2"
                else (
                    "keep safe candidate as diagnostic only, move to model-level semantic/gate learning direction"
                    if case == "C3"
                    else (
                        "reject inference-time repair; summarize upper-bound tradeoff"
                        if case == "C4"
                        else (
                            "move to training-level or architecture-level proposal only"
                            if case == "C5"
                            else "expand subset or add safety-weighted metrics"
                        )
                    )
                )
            )
        ),
    }
    write_json(REPORTS_DIR / "stage_sw42_conservative_leak_aware_repair_report.json", report)
    write_md(
        REPORTS_DIR / "stage_sw42_conservative_leak_aware_repair_report.md",
        "# Stage SW-4.2 conservative leak-aware repair\n\n"
        "## Executive summary\n\n"
        f"- effective samples: {len(records20)}\n"
        f"- diagnostic samples: {len(records5)}\n"
        f"- horizons: {horizons}\n"
        f"- baseline/K7 replay passed: {replay_eq['passed']}\n"
        f"- best conservative candidate: {report['best_conservative_candidate']}\n"
        f"- decision case: {case}\n"
        f"- next unique action: {report['next_unique_action']}\n\n"
        "## Safe claims\n\n"
        "- subset diagnostic repair ablation\n"
        "- non-oracle inference-time candidate\n"
        "- not official benchmark\n"
        "- no training\n"
        "- not deployment-ready\n"
        "- K7 aggressive is rejected\n",
    )
    log("SW-4.2 pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
