"""Stage SW-4.0 covered false-free semantic activation diagnosis.

This stage assumes the SW-3.5 corrected support conclusion:
- refine_pts is the true semantic occupancy support tensor
- support definition is decoded_metric + xyz + noflip + all48
- global support absence is no longer the dominant explanation

The script avoids rerunning the full SparseWorld image backbone. Instead it:
1. replays saved SW-2 query outputs
2. instruments OPUSHead.get_occ() offline
3. diagnoses covered false-free voxels, gate/filter, voxel aggregation, and
   semantic activation weakness

Safe-claim boundary:
- subset diagnostic only
- no training
- no official benchmark
- oracle interventions are diagnostic only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from instrument_get_occ import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_REPO_ROOT,
    DEFAULT_SW2_REPORTS,
    build_sparseworld_model,
    compare_tensors,
    extract_pred_dict_for_horizon,
    get_occ_debug,
    get_pts_bbox_head,
    write_json,
)

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"

EMPTY_IDX = 17
OCC_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]
LABEL_TO_ID = {name: idx for idx, name in enumerate(OCC_NAMES)}
STATIC_BACKGROUND = [
    LABEL_TO_ID["driveable_surface"],
    LABEL_TO_ID["other_flat"],
    LABEL_TO_ID["sidewalk"],
    LABEL_TO_ID["terrain"],
    LABEL_TO_ID["manmade"],
    LABEL_TO_ID["vegetation"],
]
DYNAMIC_VEHICLE = [
    LABEL_TO_ID["car"],
    LABEL_TO_ID["truck"],
    LABEL_TO_ID["bus"],
    LABEL_TO_ID["trailer"],
    LABEL_TO_ID["construction_vehicle"],
]
DYNAMIC_VULNERABLE = [
    LABEL_TO_ID["pedestrian"],
    LABEL_TO_ID["bicycle"],
    LABEL_TO_ID["motorcycle"],
]
ALL_DYNAMIC = sorted(set(DYNAMIC_VEHICLE + DYNAMIC_VULNERABLE))
SMALL_OBJECT = [
    LABEL_TO_ID["pedestrian"],
    LABEL_TO_ID["bicycle"],
    LABEL_TO_ID["motorcycle"],
    LABEL_TO_ID["traffic_cone"],
    LABEL_TO_ID["barrier"],
]
LARGE_DYNAMIC = [
    LABEL_TO_ID["car"],
    LABEL_TO_ID["truck"],
    LABEL_TO_ID["bus"],
    LABEL_TO_ID["trailer"],
    LABEL_TO_ID["construction_vehicle"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage SW-4.0 covered false-free semantic activation diagnosis")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--oracle-samples", type=int, default=5)
    parser.add_argument("--horizons", default="0,1,2,3,4,5,6")
    parser.add_argument("--primary-radius", type=int, default=2)
    parser.add_argument("--sample-cap-per-region", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_parent(path)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def to_cpu_artifact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_cpu_artifact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu_artifact(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_cpu_artifact(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    return obj


def load_success_rows(sw2_reports_dir: Path, num_samples: int) -> list[dict[str, Any]]:
    rows = list(csv.DictReader((sw2_reports_dir / "sw2_sample_manifest.csv").open(encoding="utf-8")))
    rows = [row for row in rows if row["forward_status"] == "success"]
    return rows[:num_samples]


def load_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": int(row["sample_index"]),
        "sample_token": row["sample_token"],
        "scene_token": row["scene_token"],
        "scene_name": row["scene_name"],
        "timestamp": int(row["timestamp"]),
        "query_artifact": torch.load(row["query_output_path"], map_location="cpu", weights_only=False),
        "pred_temporal": torch.load(row["standard_pred_occ_temporal_path"], map_location="cpu", weights_only=False).long(),
        "gt_temporal": torch.load(row["standard_gt_occ_temporal_path"], map_location="cpu", weights_only=False).long(),
    }


def group_mask(grid: torch.Tensor, class_ids: list[int]) -> torch.Tensor:
    mask = torch.zeros_like(grid, dtype=torch.bool)
    for class_id in class_ids:
        mask |= grid == int(class_id)
    return mask


def classify_regions(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, torch.Tensor]:
    gt_occ = gt != EMPTY_IDX
    pred_occ = pred != EMPTY_IDX
    return {
        "tp_occupied": gt_occ & pred_occ,
        "false_free": gt_occ & (~pred_occ),
        "false_occupied": (~gt_occ) & pred_occ,
        "true_free": (~gt_occ) & (~pred_occ),
    }


def dilate3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    t = mask.float()[None, None]
    out = torch.nn.functional.max_pool3d(t, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return out[0, 0] > 0


def sample_mask_coords(mask: torch.Tensor, limit: int, seed: int) -> torch.Tensor:
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.shape[0] <= limit:
        return coords
    g = torch.Generator(device=coords.device)
    g.manual_seed(seed)
    perm = torch.randperm(coords.shape[0], generator=g, device=coords.device)[:limit]
    return coords[perm]


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float32))) if values else 0.0


def safe_p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), 95)) if values else 0.0


def entropy_from_scores(scores: torch.Tensor) -> torch.Tensor:
    p = scores.clamp(min=1e-8)
    return -(p * torch.log(p)).sum(dim=-1)


def compute_class_ranks(scores: torch.Tensor) -> torch.Tensor:
    # scores: [N, C] -> ranks: [N, C], 1=best
    return (scores.unsqueeze(1) > scores.unsqueeze(2)).sum(dim=-1) + 1


def gather_point_cache(debug: dict[str, Any]) -> dict[str, Any]:
    geometric_voxels = debug["geometric_voxels"].long()
    geometric_scores = debug["geometric_scores"].float()
    valid_gate_mask = debug["gate_mask"].reshape(-1)[debug["valid_range_mask"]].bool()
    point_top1 = geometric_scores.argmax(dim=-1)
    point_best = geometric_scores.max(dim=-1).values
    point_entropy = entropy_from_scores(geometric_scores)
    class_ranks = compute_class_ranks(geometric_scores)
    return {
        "point_voxels": geometric_voxels,
        "point_scores": geometric_scores,
        "point_gate": valid_gate_mask,
        "point_top1": point_top1,
        "point_best": point_best,
        "point_entropy": point_entropy,
        "class_ranks": class_ranks,
    }


def aggregate_point_stats_for_region(
    sample_coords: torch.Tensor,
    gt: torch.Tensor,
    cache: dict[str, Any],
    radii: list[int],
    chunk_size: int,
) -> list[dict[str, Any]]:
    if sample_coords.numel() == 0:
        return []
    point_voxels = cache["point_voxels"]
    point_scores = cache["point_scores"]
    point_gate = cache["point_gate"]
    point_top1 = cache["point_top1"]
    point_best = cache["point_best"]
    point_entropy = cache["point_entropy"]
    class_ranks = cache["class_ranks"]
    gt_labels = gt[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]].long()
    valid_gt_global = (gt_labels >= 0) & (gt_labels < point_scores.shape[1])

    rows: list[dict[str, Any]] = []
    for radius in radii:
        support_exists_vals: list[float] = []
        support_count_vals: list[float] = []
        gt_score_max_vals: list[float] = []
        gt_score_mean_vals: list[float] = []
        best_fg_max_vals: list[float] = []
        best_fg_mean_vals: list[float] = []
        gt_rank_best_vals: list[float] = []
        gt_rank_mean_vals: list[float] = []
        entropy_mean_vals: list[float] = []
        gate_pass_point_ratio_vals: list[float] = []
        wrong_top1_ratio_vals: list[float] = []
        top1_all: list[int] = []
        valid_voxel_count = 0

        for start in range(0, sample_coords.shape[0], chunk_size):
            coords_chunk = sample_coords[start : start + chunk_size]
            labels_chunk = gt_labels[start : start + chunk_size]
            valid_gt_chunk = valid_gt_global[start : start + chunk_size]
            labels_safe = labels_chunk.clamp(0, point_scores.shape[1] - 1)
            near = (torch.abs(coords_chunk[:, None, :] - point_voxels[None, :, :]) <= radius).all(dim=-1)
            count = near.sum(dim=1)
            valid = count > 0
            support_exists_vals.extend(valid.float().cpu().tolist())
            support_count_vals.extend(count.float().cpu().tolist())
            if not bool(valid.any().item()):
                continue
            valid_voxel_count += int(valid.sum().item())

            gt_scores_chunk = point_scores[:, labels_safe].T
            gt_ranks_chunk = class_ranks[:, labels_safe].T.float()
            best_fg_chunk = point_best[None, :].expand(coords_chunk.shape[0], -1)
            entropy_chunk = point_entropy[None, :].expand(coords_chunk.shape[0], -1)
            gate_chunk = point_gate[None, :].expand(coords_chunk.shape[0], -1)
            top1_chunk = point_top1[None, :].expand(coords_chunk.shape[0], -1)
            wrong_chunk = (top1_chunk != labels_safe[:, None]) & valid_gt_chunk[:, None]
            near_float = near.float()
            count_float = count.clamp_min(1).float()

            gt_max = gt_scores_chunk.masked_fill(~near, -1e9).max(dim=1).values
            best_fg_max = best_fg_chunk.masked_fill(~near, -1e9).max(dim=1).values
            gt_mean = (gt_scores_chunk * near_float).sum(dim=1) / count_float
            best_fg_mean = (best_fg_chunk * near_float).sum(dim=1) / count_float
            gt_rank_best = gt_ranks_chunk.masked_fill(~near, 1e9).min(dim=1).values
            gt_rank_mean = (gt_ranks_chunk * near_float).sum(dim=1) / count_float
            entropy_mean = (entropy_chunk * near_float).sum(dim=1) / count_float
            gate_ratio = (gate_chunk.float() * near_float).sum(dim=1) / count_float
            wrong_ratio = (wrong_chunk.float() * near_float).sum(dim=1) / count_float

            gt_valid_and_supported = valid & valid_gt_chunk
            gt_score_max_vals.extend(gt_max[gt_valid_and_supported].detach().cpu().tolist())
            gt_score_mean_vals.extend(gt_mean[gt_valid_and_supported].detach().cpu().tolist())
            best_fg_max_vals.extend(best_fg_max[valid].detach().cpu().tolist())
            best_fg_mean_vals.extend(best_fg_mean[valid].detach().cpu().tolist())
            gt_rank_best_vals.extend(gt_rank_best[gt_valid_and_supported].detach().cpu().tolist())
            gt_rank_mean_vals.extend(gt_rank_mean[gt_valid_and_supported].detach().cpu().tolist())
            entropy_mean_vals.extend(entropy_mean[valid].detach().cpu().tolist())
            gate_pass_point_ratio_vals.extend(gate_ratio[valid].detach().cpu().tolist())
            wrong_top1_ratio_vals.extend(wrong_ratio[gt_valid_and_supported].detach().cpu().tolist())

            for row_idx in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                local_classes = point_top1[near[row_idx]].detach().cpu().tolist()
                top1_all.extend(int(x) for x in local_classes)

        top1_dist = json.dumps(dict(sorted(Counter(top1_all).items())), ensure_ascii=False)
        rows.append(
            {
                "coverage_radius_cells": radius,
                "sampled_voxel_count": int(sample_coords.shape[0]),
                "nearby_support_exists_ratio": safe_mean(support_exists_vals),
                "nearby_support_count_mean": safe_mean(support_count_vals),
                "gt_class_score_max_mean": safe_mean(gt_score_max_vals),
                "gt_class_score_mean_mean": safe_mean(gt_score_mean_vals),
                "best_fg_score_max_mean": safe_mean(best_fg_max_vals),
                "best_fg_score_mean_mean": safe_mean(best_fg_mean_vals),
                "gt_class_rank_best_mean": safe_mean(gt_rank_best_vals),
                "gt_class_rank_mean_mean": safe_mean(gt_rank_mean_vals),
                "semantic_entropy_mean": safe_mean(entropy_mean_vals),
                "gate_pass_point_ratio_mean": safe_mean(gate_pass_point_ratio_vals),
                "wrong_top1_ratio_mean": safe_mean(wrong_top1_ratio_vals),
                "top1_support_class_distribution": top1_dist,
                "valid_voxel_count": valid_voxel_count,
            }
        )
    return rows


def aggregate_gate_stats_for_region(
    sample_coords: torch.Tensor,
    debug: dict[str, Any],
    radius: int,
    chunk_size: int,
) -> dict[str, Any]:
    if sample_coords.numel() == 0:
        return {}
    pre_vox = debug["pre_gate_voxel_index"].long()
    valid_range_mask = debug["valid_range_mask"].bool()
    gate_mask = debug["gate_mask"].reshape(-1).bool()
    in_range_vox = pre_vox[valid_range_mask]
    gate_in_range_vox = pre_vox[valid_range_mask & gate_mask]
    unique_voxels = debug["unique_voxels"].long()
    semantic_active_mask = debug["semantic_active_mask"]

    out = {
        "support_exists_before_gate_ratio": [],
        "support_passes_valid_range_ratio": [],
        "support_passes_gate_ratio": [],
        "correct_voxel_assignment_ratio": [],
        "neighbor_voxel_assignment_ratio": [],
        "aggregation_contribution_ratio": [],
    }

    for start in range(0, sample_coords.shape[0], chunk_size):
        coords_chunk = sample_coords[start : start + chunk_size]
        near_pre = (torch.abs(coords_chunk[:, None, :] - pre_vox[None, :, :]) <= radius).all(dim=-1) if pre_vox.numel() else torch.zeros((coords_chunk.shape[0], 0), device=coords_chunk.device, dtype=torch.bool)
        near_valid = (torch.abs(coords_chunk[:, None, :] - in_range_vox[None, :, :]) <= radius).all(dim=-1) if in_range_vox.numel() else torch.zeros((coords_chunk.shape[0], 0), device=coords_chunk.device, dtype=torch.bool)
        near_gate = (torch.abs(coords_chunk[:, None, :] - gate_in_range_vox[None, :, :]) <= radius).all(dim=-1) if gate_in_range_vox.numel() else torch.zeros((coords_chunk.shape[0], 0), device=coords_chunk.device, dtype=torch.bool)
        exact_assign = (coords_chunk[:, None, :] == gate_in_range_vox[None, :, :]).all(dim=-1) if gate_in_range_vox.numel() else torch.zeros((coords_chunk.shape[0], 0), device=coords_chunk.device, dtype=torch.bool)
        near_unique = (torch.abs(coords_chunk[:, None, :] - unique_voxels[None, :, :]) <= radius).all(dim=-1) if unique_voxels.numel() else torch.zeros((coords_chunk.shape[0], 0), device=coords_chunk.device, dtype=torch.bool)

        out["support_exists_before_gate_ratio"].extend((near_pre.any(dim=1).float().cpu().tolist()))
        out["support_passes_valid_range_ratio"].extend((near_valid.any(dim=1).float().cpu().tolist()))
        out["support_passes_gate_ratio"].extend((near_gate.any(dim=1).float().cpu().tolist()))
        out["correct_voxel_assignment_ratio"].extend((exact_assign.any(dim=1).float().cpu().tolist()))
        out["neighbor_voxel_assignment_ratio"].extend((near_unique.any(dim=1).float().cpu().tolist()))
        contribution = semantic_active_mask[coords_chunk[:, 0], coords_chunk[:, 1], coords_chunk[:, 2]].float().cpu().tolist()
        out["aggregation_contribution_ratio"].extend(contribution)

    return {k: safe_mean(v) for k, v in out.items()}


def aggregate_voxel_stats_for_region(sample_coords: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, debug: dict[str, Any]) -> dict[str, Any]:
    if sample_coords.numel() == 0:
        return {}
    dense_scores = debug["dense_occ_after_padding"]
    semantic_active_mask = debug["semantic_active_mask"]
    geometric_mask = debug["geometric_mask"]
    contributor_count = debug["contributor_count_dense"].float()

    voxel_scores = dense_scores[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]].float()
    gt_labels = gt[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]].long()
    pred_labels = pred[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]].long()
    valid_gt = (gt_labels >= 0) & (gt_labels < voxel_scores.shape[1])
    gt_labels_safe = gt_labels.clamp(0, voxel_scores.shape[1] - 1)

    top2_vals, top2_idx = torch.topk(voxel_scores, k=min(2, voxel_scores.shape[-1]), dim=-1)
    top1_score = top2_vals[:, 0]
    top1_class = top2_idx[:, 0]
    top2_score = top2_vals[:, 1] if top2_vals.shape[1] > 1 else torch.zeros_like(top1_score)
    top1_margin = top1_score - top2_score
    gt_score = voxel_scores.gather(1, gt_labels_safe[:, None]).squeeze(1)
    best_fg = voxel_scores.max(dim=-1).values
    entropy = entropy_from_scores(voxel_scores.clamp(min=1e-8))
    contributor_vals = contributor_count[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]]
    exact_geom = geometric_mask[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]].float()
    exact_sem = semantic_active_mask[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]].float()

    wrong_class_ratio = float(((pred_labels != gt_labels) & (pred_labels != EMPTY_IDX)).float().mean().item())
    free_pred_ratio = float((pred_labels == EMPTY_IDX).float().mean().item())
    top1_dist = json.dumps(dict(sorted(Counter(top1_class.detach().cpu().tolist()).items())), ensure_ascii=False)

    return {
        "sampled_voxel_count": int(sample_coords.shape[0]),
        "gt_class_logit_mean": float(gt_score[valid_gt].mean().item()) if bool(valid_gt.any().item()) else 0.0,
        "best_fg_score_mean": float(best_fg.mean().item()) if best_fg.numel() else 0.0,
        "occupied_activation_score_mean": float(best_fg.mean().item()) if best_fg.numel() else 0.0,
        "top1_margin_mean": float(top1_margin.mean().item()) if top1_margin.numel() else 0.0,
        "semantic_entropy_mean": float(entropy.mean().item()) if entropy.numel() else 0.0,
        "contributor_count_mean": float(contributor_vals.mean().item()) if contributor_vals.numel() else 0.0,
        "contributor_count_median": float(contributor_vals.median().item()) if contributor_vals.numel() else 0.0,
        "exact_geometric_support_ratio": float(exact_geom.mean().item()) if exact_geom.numel() else 0.0,
        "exact_semantic_active_ratio": float(exact_sem.mean().item()) if exact_sem.numel() else 0.0,
        "wrong_class_ratio": wrong_class_ratio,
        "free_pred_ratio": free_pred_ratio,
        "top1_class_distribution": top1_dist,
    }


def build_temporal_region_masks(gt0: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, geometric_cover_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    regions = classify_regions(gt, pred)
    small_mask = group_mask(gt, SMALL_OBJECT)
    dynamic_mask = group_mask(gt, ALL_DYNAMIC)
    large_dynamic_mask = group_mask(gt, LARGE_DYNAMIC)
    gt0_occ = gt0 != EMPTY_IDX
    gt_occ = gt != EMPTY_IDX
    new_visible = (~gt0_occ) & gt_occ
    persistent = gt0_occ & gt_occ
    return {
        "tp_occupied": regions["tp_occupied"],
        "false_free": regions["false_free"],
        "false_occupied": regions["false_occupied"],
        "true_free": regions["true_free"],
        "covered_false_free": regions["false_free"] & geometric_cover_mask,
        "uncovered_false_free": regions["false_free"] & (~geometric_cover_mask),
        "covered_tp": regions["tp_occupied"] & geometric_cover_mask,
        "covered_false_occupied": regions["false_occupied"] & geometric_cover_mask,
        "small_object_covered_false_free": regions["false_free"] & geometric_cover_mask & small_mask,
        "dynamic_covered_false_free": regions["false_free"] & geometric_cover_mask & dynamic_mask,
        "new_visible_covered_false_free": regions["false_free"] & geometric_cover_mask & new_visible,
        "persistent_covered_false_free": regions["false_free"] & geometric_cover_mask & persistent,
        "large_dynamic_tp": regions["tp_occupied"] & geometric_cover_mask & large_dynamic_mask,
        "persistent_covered_tp": regions["tp_occupied"] & geometric_cover_mask & persistent,
    }


def summarize_mask_sizes(sample_index: int, horizon_s: int, masks: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for name, mask in masks.items():
        rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "region_name": name,
                "voxel_count": int(mask.sum().item()),
            }
        )
    return rows


def plot_hist_compare(fig_path: Path, title: str, labels: list[str], values: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_line_by_horizon(fig_path: Path, title: str, rows: list[dict[str, Any]], x_key: str, y_key: str, group_key: str) -> None:
    groups = sorted({row[group_key] for row in rows})
    horizons = sorted({int(row[x_key]) for row in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for group in groups:
        ys = []
        for h in horizons:
            sub = [float(r[y_key]) for r in rows if r[group_key] == group and int(r[x_key]) == h]
            ys.append(safe_mean(sub))
        ax.plot(horizons, ys, marker="o", label=str(group))
    ax.set_title(title)
    ax.set_xlabel("Horizon (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def point_target_mask_from_region(debug: dict[str, Any], region_mask: torch.Tensor) -> torch.Tensor:
    if debug["geometric_voxels"].numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=region_mask.device)
    vox = debug["geometric_voxels"].long()
    return region_mask[vox[:, 0], vox[:, 1], vox[:, 2]]


def build_pred_dict_copy(pred_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"cls_scores": pred_dict["cls_scores"].clone(), "refine_pts": pred_dict["refine_pts"].clone()}


def apply_gt_class_boost(
    pred_dict: dict[str, torch.Tensor],
    debug: dict[str, Any],
    gt: torch.Tensor,
    target_region: torch.Tensor,
    delta: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    pred_copy = build_pred_dict_copy(pred_dict)
    cls_flat = pred_copy["cls_scores"].reshape(-1, pred_copy["cls_scores"].shape[-1])
    point_mask = point_target_mask_from_region(debug, target_region)
    if point_mask.numel() == 0 or not bool(point_mask.any().item()):
        return pred_copy, {"target_point_count": 0}
    target_vox = debug["geometric_voxels"][point_mask].long()
    target_flat = debug["geometric_flat_indices"][point_mask].long()
    gt_classes = gt[target_vox[:, 0], target_vox[:, 1], target_vox[:, 2]].long()
    valid = (gt_classes >= 0) & (gt_classes < cls_flat.shape[-1])
    target_flat = target_flat[valid]
    gt_classes = gt_classes[valid]
    if target_flat.numel() == 0:
        return pred_copy, {"target_point_count": 0}
    cls_flat[target_flat, gt_classes] += float(delta)
    return pred_copy, {"target_point_count": int(target_flat.numel())}


def apply_foreground_boost(
    pred_dict: dict[str, torch.Tensor],
    debug: dict[str, Any],
    target_region: torch.Tensor,
    delta: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    pred_copy = build_pred_dict_copy(pred_dict)
    cls_flat = pred_copy["cls_scores"].reshape(-1, pred_copy["cls_scores"].shape[-1])
    point_mask = point_target_mask_from_region(debug, target_region)
    if point_mask.numel() == 0 or not bool(point_mask.any().item()):
        return pred_copy, {"target_point_count": 0}
    target_flat = debug["geometric_flat_indices"][point_mask].long()
    scores = debug["geometric_scores"][point_mask]
    top_cls = scores.argmax(dim=-1)
    cls_flat[target_flat, top_cls] += float(delta)
    return pred_copy, {"target_point_count": int(target_flat.numel())}


def apply_gate_relaxation(debug: dict[str, Any], target_region: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    flat_mask = torch.zeros_like(debug["gate_mask"].reshape(-1), dtype=torch.bool)
    point_mask = point_target_mask_from_region(debug, target_region)
    if point_mask.numel() == 0 or not bool(point_mask.any().item()):
        return flat_mask, {"target_point_count": 0}
    flat_mask[debug["geometric_flat_indices"][point_mask].long()] = True
    return flat_mask, {"target_point_count": int(point_mask.sum().item())}


def apply_local_duplication(
    pred_dict: dict[str, torch.Tensor],
    debug: dict[str, Any],
    target_region: torch.Tensor,
    max_copies: int = 256,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    pred_copy = build_pred_dict_copy(pred_dict)
    cls_flat = pred_copy["cls_scores"].reshape(-1, pred_copy["cls_scores"].shape[-1])
    pts_flat = pred_copy["refine_pts"].reshape(-1, 3)
    point_mask = point_target_mask_from_region(debug, target_region)
    source_idx = debug["geometric_flat_indices"][point_mask].long()
    if source_idx.numel() == 0:
        return pred_copy, {"target_point_count": 0, "copied_count": 0}
    source_idx = source_idx[:max_copies]
    raw_scores = pred_copy["cls_scores"].reshape(-1, pred_copy["cls_scores"].shape[-1]).sigmoid().max(dim=-1).values
    available = torch.ones_like(raw_scores, dtype=torch.bool)
    available[source_idx] = False
    dest_idx = torch.argsort(raw_scores[available])[: source_idx.numel()]
    available_idx = torch.nonzero(available, as_tuple=False).flatten()
    dest_idx = available_idx[dest_idx]
    copy_count = min(source_idx.numel(), dest_idx.numel())
    if copy_count == 0:
        return pred_copy, {"target_point_count": int(source_idx.numel()), "copied_count": 0}
    cls_flat[dest_idx[:copy_count]] = cls_flat[source_idx[:copy_count]]
    pts_flat[dest_idx[:copy_count]] = pts_flat[source_idx[:copy_count]]
    return pred_copy, {"target_point_count": int(source_idx.numel()), "copied_count": int(copy_count)}


def compute_basic_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    gt_occ = gt != EMPTY_IDX
    pred_occ = pred != EMPTY_IDX
    tp = torch.logical_and(gt_occ, pred_occ).sum().item()
    gt_count = gt_occ.sum().item()
    pred_count = pred_occ.sum().item()
    union = torch.logical_or(gt_occ, pred_occ).sum().item()
    ff = torch.logical_and(gt_occ, ~pred_occ).sum().item()
    fo = torch.logical_and(~gt_occ, pred_occ).sum().item()
    return {
        "pred_occupied_count": float(pred_count),
        "gt_occupied_count": float(gt_count),
        "occupied_iou": float(tp / union) if union else 0.0,
        "false_free_rate": float(ff / gt_count) if gt_count else 0.0,
        "false_occupied_rate": float(fo / pred_count) if pred_count else 0.0,
    }


def run_oracle_interventions(
    head: Any,
    oracle_records: list[dict[str, Any]],
    primary_radius: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deltas = [1.0, 2.0, 4.0, 8.0]
    for rec_idx, record in enumerate(oracle_records, start=1):
        gt0 = record["gt_temporal"][0].cuda(non_blocking=False)
        for horizon_s in range(record["pred_temporal"].shape[0]):
            gt = record["gt_temporal"][horizon_s].cuda(non_blocking=False)
            pred = record["pred_temporal"][horizon_s].cuda(non_blocking=False)
            pred_dict = extract_pred_dict_for_horizon(record["query_artifact"], horizon_s)
            _, debug_list = get_occ_debug(head, pred_dict, capture_dense=False)
            debug = debug_list[0]
            geometric_cover = dilate3d(debug["geometric_mask"], primary_radius)
            masks = build_temporal_region_masks(gt0, gt, pred, geometric_cover)
            base_metrics = compute_basic_metrics(pred, gt)
            small_ff_mask = masks["small_object_covered_false_free"]
            new_visible_mask = masks["new_visible_covered_false_free"]
            covered_ff_mask = masks["covered_false_free"]
            target_sets = {
                "A_gt_class_boost_covered_ff": covered_ff_mask,
                "B_foreground_boost_covered_ff": covered_ff_mask,
                "C_gate_relax_covered_ff": covered_ff_mask,
                "D_duplicate_support_covered_ff": covered_ff_mask,
                "E_small_object_class_oracle": small_ff_mask,
                "F_new_visible_future_semantics_boost": new_visible_mask if horizon_s > 0 else torch.zeros_like(covered_ff_mask),
            }
            for name, target_region in target_sets.items():
                for delta in deltas:
                    gate_force_mask = None
                    meta: dict[str, Any] = {}
                    mod_pred_dict = pred_dict
                    if name == "A_gt_class_boost_covered_ff":
                        mod_pred_dict, meta = apply_gt_class_boost(pred_dict, debug, gt, target_region, delta)
                    elif name == "B_foreground_boost_covered_ff":
                        mod_pred_dict, meta = apply_foreground_boost(pred_dict, debug, target_region, delta)
                    elif name == "C_gate_relax_covered_ff":
                        gate_force_mask, meta = apply_gate_relaxation(debug, target_region)
                    elif name == "D_duplicate_support_covered_ff":
                        mod_pred_dict, meta = apply_local_duplication(pred_dict, debug, target_region, max_copies=int(32 * delta))
                    elif name == "E_small_object_class_oracle":
                        mod_pred_dict, meta = apply_gt_class_boost(pred_dict, debug, gt, target_region, delta)
                    elif name == "F_new_visible_future_semantics_boost":
                        mod_pred_dict, meta = apply_gt_class_boost(pred_dict, debug, gt, target_region, delta)
                    if meta.get("target_point_count", 0) == 0 and name != "D_duplicate_support_covered_ff":
                        rows.append(
                            {
                                "sample_index": record["sample_index"],
                                "horizon_s": horizon_s,
                                "intervention_name": name,
                                "delta": delta,
                                "status": "skipped_no_target_points",
                                **base_metrics,
                            }
                        )
                        continue
                    with torch.no_grad():
                        mod_pred, _ = get_occ_debug(head, mod_pred_dict, capture_dense=False, gate_force_flat_mask=gate_force_mask)
                    mod_pred0 = mod_pred[0]
                    metrics = compute_basic_metrics(mod_pred0, gt)
                    small_mask = group_mask(gt, SMALL_OBJECT)
                    new_visible = ((gt0 == EMPTY_IDX) & (gt != EMPTY_IDX))
                    small_ff = torch.logical_and(small_mask, mod_pred0 == EMPTY_IDX)
                    small_gt = small_mask.sum().item()
                    new_ff = torch.logical_and(new_visible, mod_pred0 == EMPTY_IDX)
                    new_gt = new_visible.sum().item()
                    rows.append(
                        {
                            "sample_index": record["sample_index"],
                            "sample_token": record["sample_token"],
                            "scene_token": record["scene_token"],
                            "horizon_s": horizon_s,
                            "intervention_name": name,
                            "delta": delta,
                            "status": "executed",
                            "target_point_count": int(meta.get("target_point_count", 0)),
                            "copied_count": int(meta.get("copied_count", 0)),
                            "false_free_delta": float(metrics["false_free_rate"] - base_metrics["false_free_rate"]),
                            "pred_occupied_delta": float(metrics["pred_occupied_count"] - base_metrics["pred_occupied_count"]),
                            "small_object_false_free_rate": float(small_ff.sum().item() / small_gt) if small_gt else 0.0,
                            "new_visible_false_free_rate": float(new_ff.sum().item() / new_gt) if new_gt else 0.0,
                            **metrics,
                        }
                    )
        print(f"[SW4] oracle interventions progress: sample {rec_idx}/{len(oracle_records)} index={record['sample_index']}", flush=True)
    return rows


def build_case_galleries(figures_dir: Path, sample0_visuals: dict[str, np.ndarray]) -> None:
    if not sample0_visuals:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    items = list(sample0_visuals.items())[:4]
    for ax, (title, img) in zip(axes.flatten(), items):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flatten()[len(items) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "oracle_intervention_bev_before_after.png", dpi=150)
    plt.close(fig)


def run_phase1_equivalence_with_existing_head(
    head: Any,
    sample0_record: dict[str, Any],
) -> dict[str, Any]:
    checks = []
    for horizon_s in [0, 6]:
        pred_dict = extract_pred_dict_for_horizon(sample0_record["query_artifact"], horizon_s)
        with torch.no_grad():
            original = head.get_occ(pred_dict)
            debug_pred, debug_list = get_occ_debug(head, pred_dict, capture_dense=True)
        cmp = compare_tensors(original, debug_pred)
        checks.append({"sample_index": sample0_record["sample_index"], "horizon_s": horizon_s, **cmp})
        torch.save(
            to_cpu_artifact(debug_list[0]),
            ARTIFACTS_DIR / f"get_occ_debug_sample{sample0_record['sample_index']}_t{horizon_s}.pt",
        )
    manifest = {
        "repo_root": str(DEFAULT_REPO_ROOT),
        "config_path": str(DEFAULT_CONFIG),
        "checkpoint_path": str(DEFAULT_CHECKPOINT),
        "head_class": type(head).__name__,
        "get_occ_source_file": str((DEFAULT_REPO_ROOT / "mmdet3d/models/sparsedetectors/opus_head.py").resolve()),
        "captured_nodes": [
            "decoded_points_metric",
            "mask_dist",
            "mask_score",
            "gate_mask",
            "valid_range_mask",
            "unique_voxels",
            "agg_scores_sparse",
            "dense_occ_before_padding",
            "dense_occ_after_padding",
            "occ_pred",
        ],
    }
    eq_payload = {
        "sample_index": sample0_record["sample_index"],
        "checks": checks,
        "all_exact_equal": all(row["exact_equal"] for row in checks),
    }
    write_json(REPORTS_DIR / "get_occ_instrumentation_manifest.json", manifest)
    write_json(REPORTS_DIR / "get_occ_output_equivalence_check.json", eq_payload)
    return {"manifest": manifest, "equivalence": eq_payload}


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    for path in [REPORTS_DIR, LOGS_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    success_rows = load_success_rows(DEFAULT_SW2_REPORTS, args.num_samples)
    oracle_rows = success_rows[: min(args.oracle_samples, len(success_rows))]
    records = [load_record(row) for row in success_rows]
    oracle_records = [load_record(row) for row in oracle_rows]

    print(f"[SW4] loading SparseWorld head from {repo_root}", flush=True)
    _cfg, model, _checkpoint = build_sparseworld_model(repo_root, config_path, checkpoint_path, use_fp16=False)
    head = get_pts_bbox_head(model)

    # Phase 1 get_occ instrumentation + equivalence.
    print("[SW4] phase1 get_occ equivalence", flush=True)
    phase1 = run_phase1_equivalence_with_existing_head(head, records[0])

    sampling_rows: list[dict[str, Any]] = []
    support_score_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    voxel_rows: list[dict[str, Any]] = []
    sample_payload: dict[str, Any] = {}
    sample0_visuals: dict[str, np.ndarray] = {}

    radii = [1, 2, 3, 5]
    for rec_idx, record in enumerate(records, start=1):
        gt0 = record["gt_temporal"][0].cuda(non_blocking=False)
        sample_payload[str(record["sample_index"])] = {}
        for horizon_s in horizons:
            gt = record["gt_temporal"][horizon_s].cuda(non_blocking=False)
            pred = record["pred_temporal"][horizon_s].cuda(non_blocking=False)
            pred_dict = extract_pred_dict_for_horizon(record["query_artifact"], horizon_s)
            with torch.no_grad():
                _pred_dbg, debug_list = get_occ_debug(head, pred_dict, capture_dense=True)
            debug = debug_list[0]
            geometric_cover = dilate3d(debug["geometric_mask"], args.primary_radius)
            masks = build_temporal_region_masks(gt0, gt, pred, geometric_cover)
            cache = gather_point_cache(debug)

            sampling_rows.extend(summarize_mask_sizes(record["sample_index"], horizon_s, masks))
            sample_payload[str(record["sample_index"])][f"h{horizon_s}"] = {
                name: {
                    "coords": sample_mask_coords(mask, args.sample_cap_per_region, seed=record["sample_index"] * 100 + horizon_s * 10 + idx).cpu()
                }
                for idx, (name, mask) in enumerate(masks.items())
                if int(mask.sum().item()) > 0
            }

            for region_name, region_info in sample_payload[str(record["sample_index"])][f"h{horizon_s}"].items():
                sample_coords = region_info["coords"].cuda(non_blocking=False)
                if sample_coords.numel() == 0:
                    continue
                phase3_rows = aggregate_point_stats_for_region(sample_coords, gt, cache, radii, args.chunk_size)
                for row in phase3_rows:
                    row.update(
                        {
                            "sample_index": record["sample_index"],
                            "sample_token": record["sample_token"],
                            "scene_token": record["scene_token"],
                            "scene_name": record["scene_name"],
                            "horizon_s": horizon_s,
                            "region_name": region_name,
                        }
                    )
                    support_score_rows.append(row)

                gate_row = aggregate_gate_stats_for_region(sample_coords, debug, args.primary_radius, args.chunk_size)
                if gate_row:
                    gate_row.update(
                        {
                            "sample_index": record["sample_index"],
                            "sample_token": record["sample_token"],
                            "scene_token": record["scene_token"],
                            "scene_name": record["scene_name"],
                            "horizon_s": horizon_s,
                            "region_name": region_name,
                            "sampled_voxel_count": int(sample_coords.shape[0]),
                        }
                    )
                    gate_rows.append(gate_row)

                voxel_row = aggregate_voxel_stats_for_region(sample_coords, gt, pred, debug)
                if voxel_row:
                    voxel_row.update(
                        {
                            "sample_index": record["sample_index"],
                            "sample_token": record["sample_token"],
                            "scene_token": record["scene_token"],
                            "scene_name": record["scene_name"],
                            "horizon_s": horizon_s,
                            "region_name": region_name,
                        }
                    )
                    voxel_rows.append(voxel_row)

            if record["sample_index"] == 0 and horizon_s in (0, 1, 6):
                gt_occ = (gt != EMPTY_IDX).any(dim=-1).detach().cpu().numpy().astype(np.uint8)
                pred_occ = (pred != EMPTY_IDX).any(dim=-1).detach().cpu().numpy().astype(np.uint8)
                geo_occ = debug["geometric_mask"].any(dim=-1).detach().cpu().numpy().astype(np.uint8)
                sample0_visuals[f"GT_occ_h{horizon_s}"] = np.repeat(gt_occ[..., None] * 255, 3, axis=-1)
                sample0_visuals[f"Pred_occ_h{horizon_s}"] = np.repeat(pred_occ[..., None] * 255, 3, axis=-1)
                sample0_visuals[f"Support_h{horizon_s}"] = np.repeat(geo_occ[..., None] * 255, 3, axis=-1)
        print(f"[SW4] phases2-7 progress: sample {rec_idx}/{len(records)} index={record['sample_index']}", flush=True)

    torch.save(to_cpu_artifact(sample_payload), ARTIFACTS_DIR / "covered_ff_voxel_samples.pt")

    # Phase 3 outputs
    write_csv(REPORTS_DIR / "covered_ff_voxel_sampling_summary.csv", sampling_rows)
    write_json(REPORTS_DIR / "covered_ff_voxel_sampling_summary.json", {"rows": sampling_rows})
    write_csv(REPORTS_DIR / "support_level_semantic_scores.csv", support_score_rows)
    write_json(REPORTS_DIR / "support_level_semantic_scores.json", {"rows": support_score_rows})
    write_md(
        REPORTS_DIR / "support_level_semantic_score_summary.md",
        "# Support-level semantic score summary\n\n"
        f"- rows: `{len(support_score_rows)}`\n"
        f"- samples: `{len(records)}`\n"
        f"- radii: `{radii}`\n",
    )

    # Phase 4 outputs
    write_csv(REPORTS_DIR / "get_occ_gate_filter_analysis.csv", gate_rows)
    write_json(REPORTS_DIR / "get_occ_gate_filter_analysis.json", {"rows": gate_rows})
    write_md(
        REPORTS_DIR / "get_occ_gate_filter_summary.md",
        "# get_occ gate/filter summary\n\n"
        f"- rows: `{len(gate_rows)}`\n"
        f"- primary radius: `{args.primary_radius}`\n",
    )

    # Phase 5 outputs
    write_csv(REPORTS_DIR / "voxel_aggregation_logit_decomposition.csv", voxel_rows)
    write_json(REPORTS_DIR / "voxel_aggregation_logit_decomposition.json", {"rows": voxel_rows})
    write_md(
        REPORTS_DIR / "voxel_aggregation_summary.md",
        "# Voxel aggregation/logit decomposition summary\n\n"
        f"- rows: `{len(voxel_rows)}`\n",
    )

    # Phase 6 small-object diagnosis
    small_rows = [row for row in support_score_rows if row["region_name"] in {"small_object_covered_false_free", "large_dynamic_tp"} and row["coverage_radius_cells"] == args.primary_radius]
    write_csv(REPORTS_DIR / "small_object_semantic_activation_diagnosis.csv", small_rows)
    write_json(REPORTS_DIR / "small_object_semantic_activation_diagnosis.json", {"rows": small_rows})
    small_ff_cov = safe_mean([row["nearby_support_exists_ratio"] for row in small_rows if row["region_name"] == "small_object_covered_false_free"])
    small_ff_gt_score = safe_mean([row["gt_class_score_max_mean"] for row in small_rows if row["region_name"] == "small_object_covered_false_free"])
    large_dyn_gt_score = safe_mean([row["gt_class_score_max_mean"] for row in small_rows if row["region_name"] == "large_dynamic_tp"])
    small_rank = safe_mean([row["gt_class_rank_best_mean"] for row in small_rows if row["region_name"] == "small_object_covered_false_free"])
    large_rank = safe_mean([row["gt_class_rank_best_mean"] for row in small_rows if row["region_name"] == "large_dynamic_tp"])
    write_md(
        REPORTS_DIR / "small_object_semantic_activation_summary.md",
        "# Small-object semantic activation summary\n\n"
        f"- mean small-object nearby support exists ratio @r{args.primary_radius}: `{small_ff_cov}`\n"
        f"- mean small-object GT-class score max: `{small_ff_gt_score}`\n"
        f"- mean large-dynamic GT-class score max: `{large_dyn_gt_score}`\n"
        f"- mean small-object GT-class best rank: `{small_rank}`\n"
        f"- mean large-dynamic GT-class best rank: `{large_rank}`\n",
    )

    # Phase 7 new-visible diagnosis
    new_visible_rows = [row for row in support_score_rows if row["region_name"] in {"new_visible_covered_false_free", "persistent_covered_false_free", "persistent_covered_tp"} and row["coverage_radius_cells"] == args.primary_radius]
    write_csv(REPORTS_DIR / "new_visible_semantic_activation_diagnosis.csv", new_visible_rows)
    write_json(REPORTS_DIR / "new_visible_semantic_activation_diagnosis.json", {"rows": new_visible_rows})
    new_visible_gt_score = safe_mean([row["gt_class_score_max_mean"] for row in new_visible_rows if row["region_name"] == "new_visible_covered_false_free"])
    persistent_ff_gt_score = safe_mean([row["gt_class_score_max_mean"] for row in new_visible_rows if row["region_name"] == "persistent_covered_false_free"])
    persistent_tp_gt_score = safe_mean([row["gt_class_score_max_mean"] for row in new_visible_rows if row["region_name"] == "persistent_covered_tp"])
    write_md(
        REPORTS_DIR / "new_visible_semantic_activation_summary.md",
        "# New-visible semantic activation summary\n\n"
        f"- mean new-visible GT-class score max: `{new_visible_gt_score}`\n"
        f"- mean persistent FF GT-class score max: `{persistent_ff_gt_score}`\n"
        f"- mean persistent TP GT-class score max: `{persistent_tp_gt_score}`\n",
    )

    # Phase 8 oracle interventions
    print("[SW4] phase8 oracle interventions", flush=True)
    oracle_rows = run_oracle_interventions(head, oracle_records, args.primary_radius)
    write_csv(REPORTS_DIR / "semantic_activation_oracle_interventions.csv", oracle_rows)
    write_json(REPORTS_DIR / "semantic_activation_oracle_interventions.json", {"rows": oracle_rows})
    write_md(
        REPORTS_DIR / "semantic_activation_oracle_summary.md",
        "# Semantic activation oracle summary\n\n"
        "- Oracle interventions are diagnostic only.\n"
        f"- rows: `{len(oracle_rows)}`\n",
    )

    # Figures
    tp_rows = [row for row in support_score_rows if row["region_name"] == "covered_tp" and row["coverage_radius_cells"] == args.primary_radius]
    ff_rows = [row for row in support_score_rows if row["region_name"] == "covered_false_free" and row["coverage_radius_cells"] == args.primary_radius]
    fo_rows = [row for row in support_score_rows if row["region_name"] == "covered_false_occupied" and row["coverage_radius_cells"] == args.primary_radius]
    plot_hist_compare(FIGURES_DIR / "support_gt_class_score_tp_vs_ff.png", "Support GT-class score: TP vs covered FF", ["TP", "covered_FF"], [safe_mean([r["gt_class_score_max_mean"] for r in tp_rows]), safe_mean([r["gt_class_score_max_mean"] for r in ff_rows])])
    plot_hist_compare(FIGURES_DIR / "support_best_fg_score_tp_vs_ff.png", "Support best-FG score: TP vs covered FF", ["TP", "covered_FF"], [safe_mean([r["best_fg_score_max_mean"] for r in tp_rows]), safe_mean([r["best_fg_score_max_mean"] for r in ff_rows])])
    plot_hist_compare(FIGURES_DIR / "support_gt_class_rank_tp_vs_ff.png", "Support GT-class best rank: TP vs covered FF", ["TP", "covered_FF"], [safe_mean([r["gt_class_rank_best_mean"] for r in tp_rows]), safe_mean([r["gt_class_rank_best_mean"] for r in ff_rows])])
    plot_hist_compare(FIGURES_DIR / "support_semantic_entropy_tp_vs_ff.png", "Support semantic entropy: TP vs covered FF", ["TP", "covered_FF"], [safe_mean([r["semantic_entropy_mean"] for r in tp_rows]), safe_mean([r["semantic_entropy_mean"] for r in ff_rows])])
    plot_hist_compare(FIGURES_DIR / "small_object_support_score_profile.png", "Small-object vs large-dynamic GT-class score", ["small_FF", "large_dyn_TP"], [small_ff_gt_score, large_dyn_gt_score])
    plot_hist_compare(FIGURES_DIR / "new_visible_support_score_profile.png", "New-visible vs persistent GT-class score", ["new_visible_FF", "persistent_TP"], [new_visible_gt_score, persistent_tp_gt_score])

    gate_tp = [row for row in gate_rows if row["region_name"] == "covered_tp"]
    gate_ff = [row for row in gate_rows if row["region_name"] == "covered_false_free"]
    plot_hist_compare(FIGURES_DIR / "gate_pass_rate_by_error_type.png", "Gate pass ratio by error type", ["covered_TP", "covered_FF", "covered_FO"], [safe_mean([r["support_passes_gate_ratio"] for r in gate_tp]), safe_mean([r["support_passes_gate_ratio"] for r in gate_ff]), safe_mean([r["support_passes_gate_ratio"] for r in gate_rows if r["region_name"] == "covered_false_occupied"])])
    plot_hist_compare(FIGURES_DIR / "voxelization_assignment_rate_by_error_type.png", "Correct voxel assignment ratio", ["covered_TP", "covered_FF"], [safe_mean([r["correct_voxel_assignment_ratio"] for r in gate_tp]), safe_mean([r["correct_voxel_assignment_ratio"] for r in gate_ff])])
    plot_hist_compare(FIGURES_DIR / "aggregation_contribution_rate_by_error_type.png", "Aggregation contribution ratio", ["covered_TP", "covered_FF"], [safe_mean([r["aggregation_contribution_ratio"] for r in gate_tp]), safe_mean([r["aggregation_contribution_ratio"] for r in gate_ff])])
    plot_hist_compare(FIGURES_DIR / "small_object_gate_drop_profile.png", "Small-object gate pass ratio", ["small_FF", "large_dyn_TP"], [safe_mean([r["gate_pass_point_ratio_mean"] for r in small_rows if r["region_name"] == "small_object_covered_false_free"]), safe_mean([r["gate_pass_point_ratio_mean"] for r in small_rows if r["region_name"] == "large_dynamic_tp"])])
    plot_hist_compare(FIGURES_DIR / "new_visible_gate_drop_profile.png", "New-visible gate pass ratio", ["new_visible_FF", "persistent_TP"], [safe_mean([r["gate_pass_point_ratio_mean"] for r in new_visible_rows if r["region_name"] == "new_visible_covered_false_free"]), safe_mean([r["gate_pass_point_ratio_mean"] for r in new_visible_rows if r["region_name"] == "persistent_covered_tp"])])

    voxel_tp = [row for row in voxel_rows if row["region_name"] == "covered_tp"]
    voxel_ff = [row for row in voxel_rows if row["region_name"] == "covered_false_free"]
    plot_hist_compare(FIGURES_DIR / "voxel_gt_class_logit_tp_vs_ff.png", "Voxel GT-class logit: TP vs covered FF", ["covered_TP", "covered_FF"], [safe_mean([r["gt_class_logit_mean"] for r in voxel_tp]), safe_mean([r["gt_class_logit_mean"] for r in voxel_ff])])
    plot_hist_compare(FIGURES_DIR / "voxel_occupied_activation_tp_vs_ff.png", "Voxel occupied activation: TP vs covered FF", ["covered_TP", "covered_FF"], [safe_mean([r["occupied_activation_score_mean"] for r in voxel_tp]), safe_mean([r["occupied_activation_score_mean"] for r in voxel_ff])])
    plot_hist_compare(FIGURES_DIR / "voxel_contributor_count_tp_vs_ff.png", "Voxel contributor count: TP vs covered FF", ["covered_TP", "covered_FF"], [safe_mean([r["contributor_count_mean"] for r in voxel_tp]), safe_mean([r["contributor_count_mean"] for r in voxel_ff])])
    plot_hist_compare(FIGURES_DIR / "voxel_top1_margin_tp_vs_ff.png", "Voxel top1 margin: TP vs covered FF", ["covered_TP", "covered_FF"], [safe_mean([r["top1_margin_mean"] for r in voxel_tp]), safe_mean([r["top1_margin_mean"] for r in voxel_ff])])
    plot_hist_compare(FIGURES_DIR / "voxel_semantic_confusion_matrix_covered_regions.png", "Wrong-class ratio in covered regions", ["covered_TP", "covered_FF", "covered_FO"], [safe_mean([r["wrong_class_ratio"] for r in voxel_tp]), safe_mean([r["wrong_class_ratio"] for r in voxel_ff]), safe_mean([r["wrong_class_ratio"] for r in voxel_rows if r["region_name"] == "covered_false_occupied"])])

    plot_hist_compare(FIGURES_DIR / "small_object_class_rank_distribution.png", "GT-class best rank: small-object vs large-dynamic", ["small_FF", "large_dyn_TP"], [small_rank, large_rank])
    plot_hist_compare(FIGURES_DIR / "small_object_gt_score_vs_large_dynamic.png", "GT-class score: small-object vs large-dynamic", ["small_FF", "large_dyn_TP"], [small_ff_gt_score, large_dyn_gt_score])
    plot_hist_compare(FIGURES_DIR / "small_object_support_top1_class_hist.png", "Small-object wrong-top1 ratio", ["small_FF", "large_dyn_TP"], [safe_mean([r["wrong_top1_ratio_mean"] for r in small_rows if r["region_name"] == "small_object_covered_false_free"]), safe_mean([r["wrong_top1_ratio_mean"] for r in small_rows if r["region_name"] == "large_dynamic_tp"])])
    plot_hist_compare(FIGURES_DIR / "small_object_contributor_competition.png", "Small-object contributor count", ["small_FF", "large_dyn_TP"], [safe_mean([r["contributor_count_mean"] for r in voxel_rows if r["region_name"] == "small_object_covered_false_free"]), safe_mean([r["contributor_count_mean"] for r in voxel_rows if r["region_name"] == "large_dynamic_tp"])])
    plot_hist_compare(FIGURES_DIR / "new_visible_gt_score_over_horizon.png", "New-visible GT-class score vs persistent", ["new_visible_FF", "persistent_TP"], [new_visible_gt_score, persistent_tp_gt_score])
    plot_hist_compare(FIGURES_DIR / "persistent_vs_new_visible_semantic_score.png", "New-visible semantic score profile", ["new_visible_FF", "persistent_FF", "persistent_TP"], [new_visible_gt_score, persistent_ff_gt_score, persistent_tp_gt_score])
    plot_line_by_horizon(FIGURES_DIR / "new_visible_gate_pass_over_horizon.png", "New-visible gate pass over horizon", [r for r in new_visible_rows if r["region_name"] in {"new_visible_covered_false_free", "persistent_covered_tp"}], "horizon_s", "gate_pass_point_ratio_mean", "region_name")
    plot_line_by_horizon(FIGURES_DIR / "new_visible_forecast_semantic_decay.png", "New-visible GT score over horizon", [r for r in new_visible_rows if r["region_name"] in {"new_visible_covered_false_free", "persistent_covered_tp"}], "horizon_s", "gt_class_score_max_mean", "region_name")

    oracle_valid = [row for row in oracle_rows if row["status"] == "executed"]
    plot_line_by_horizon(FIGURES_DIR / "oracle_delta_vs_false_free.png", "Oracle FF delta over delta", [{"horizon_s": int(row["delta"]), "false_free_delta": row["false_free_delta"], "intervention_name": row["intervention_name"]} for row in oracle_valid if row["intervention_name"] in {"A_gt_class_boost_covered_ff", "B_foreground_boost_covered_ff", "C_gate_relax_covered_ff", "D_duplicate_support_covered_ff"}], "horizon_s", "false_free_delta", "intervention_name")
    plot_line_by_horizon(FIGURES_DIR / "oracle_small_object_recovery.png", "Small-object oracle recovery", [{"horizon_s": int(row["delta"]), "small_object_false_free_rate": row["small_object_false_free_rate"], "intervention_name": row["intervention_name"]} for row in oracle_valid if row["intervention_name"] == "E_small_object_class_oracle"], "horizon_s", "small_object_false_free_rate", "intervention_name")
    plot_line_by_horizon(FIGURES_DIR / "oracle_new_visible_recovery.png", "New-visible oracle recovery", [{"horizon_s": int(row["delta"]), "new_visible_false_free_rate": row["new_visible_false_free_rate"], "intervention_name": row["intervention_name"]} for row in oracle_valid if row["intervention_name"] == "F_new_visible_future_semantics_boost"], "horizon_s", "new_visible_false_free_rate", "intervention_name")
    build_case_galleries(FIGURES_DIR, sample0_visuals)
    # placeholder simple galleries
    for placeholder_name in ["small_object_failure_case_gallery.png", "new_visible_failure_case_gallery.png"]:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(next(iter(sample0_visuals.values())))
        ax.set_title(placeholder_name.replace(".png", ""))
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / placeholder_name, dpi=150)
        plt.close(fig)

    # Phase 9 decision
    covered_ff_gate = safe_mean([r["support_passes_gate_ratio"] for r in gate_rows if r["region_name"] == "covered_false_free"])
    covered_tp_gate = safe_mean([r["support_passes_gate_ratio"] for r in gate_rows if r["region_name"] == "covered_tp"])
    covered_ff_gt_score = safe_mean([r["gt_class_score_max_mean"] for r in ff_rows])
    covered_tp_gt_score = safe_mean([r["gt_class_score_max_mean"] for r in tp_rows])
    covered_ff_wrong = safe_mean([r["wrong_top1_ratio_mean"] for r in ff_rows])
    covered_tp_wrong = safe_mean([r["wrong_top1_ratio_mean"] for r in tp_rows])
    covered_ff_contrib = safe_mean([r["contributor_count_mean"] for r in voxel_ff])
    covered_tp_contrib = safe_mean([r["contributor_count_mean"] for r in voxel_tp])

    oracle_by_name = defaultdict(list)
    for row in oracle_valid:
        oracle_by_name[row["intervention_name"]].append(row)
    oracle_summary = {
        name: {
            "mean_false_free_delta": safe_mean([r["false_free_delta"] for r in rows]),
            "mean_small_object_false_free_rate": safe_mean([r["small_object_false_free_rate"] for r in rows]),
            "mean_new_visible_false_free_rate": safe_mean([r["new_visible_false_free_rate"] for r in rows]),
            "mean_pred_occupied_delta": safe_mean([r["pred_occupied_delta"] for r in rows]),
        }
        for name, rows in oracle_by_name.items()
    }

    primary_case = "S8"
    next_action = "deeper get_occ instrumentation"
    primary_reason = "instrumentation limitation"
    if covered_ff_gt_score + 0.10 < covered_tp_gt_score and oracle_summary.get("A_gt_class_boost_covered_ff", {}).get("mean_false_free_delta", 0.0) < -0.01:
        primary_case = "S1"
        next_action = "semantic activation calibration / class-score repair"
        primary_reason = "covered FF support exists but GT-class scores are weaker than TP"
    elif covered_ff_wrong > covered_tp_wrong + 0.2:
        primary_case = "S2"
        next_action = "semantic activation calibration / class-score repair"
        primary_reason = "covered FF has higher wrong-class competition"
    elif covered_ff_gate + 0.15 < covered_tp_gate and oracle_summary.get("C_gate_relax_covered_ff", {}).get("mean_false_free_delta", 0.0) < -0.01:
        primary_case = "S3"
        next_action = "gate/filter repair"
        primary_reason = "support exists but gate/filter suppresses covered FF more than TP"
    elif covered_ff_contrib + 0.5 < covered_tp_contrib and oracle_summary.get("D_duplicate_support_covered_ff", {}).get("mean_false_free_delta", 0.0) < -0.01:
        primary_case = "S7"
        next_action = "aggregation/local support density repair"
        primary_reason = "local contributor density is lower in covered FF"
    elif safe_mean([r["gt_class_logit_mean"] for r in voxel_ff]) + 0.08 < safe_mean([r["gt_class_logit_mean"] for r in voxel_tp]):
        primary_case = "S4"
        next_action = "aggregation/local support density repair"
        primary_reason = "support enters covered regions but voxel aggregation suppresses GT class"

    small_case = "S8"
    small_reason = "unresolved"
    if small_ff_gt_score + 0.10 < large_dyn_gt_score and small_rank > large_rank + 2.0:
        small_case = "S5"
        small_reason = "small-object support exists but class rank and GT-class score are weak"
    elif small_ff_cov < 0.5:
        small_case = "S7"
        small_reason = "small-object local support density insufficient"

    new_visible_case = "S8"
    new_visible_reason = "unresolved"
    if new_visible_gt_score + 0.08 < persistent_tp_gt_score and oracle_summary.get("F_new_visible_future_semantics_boost", {}).get("mean_false_free_delta", 0.0) < -0.005:
        new_visible_case = "S6"
        new_visible_reason = "future/new-visible support exists but forecast semantics are weaker"
    elif safe_mean([r["support_passes_gate_ratio"] for r in gate_rows if r["region_name"] == "new_visible_covered_false_free"]) + 0.10 < safe_mean([r["support_passes_gate_ratio"] for r in gate_rows if r["region_name"] == "persistent_covered_tp"]):
        new_visible_case = "S3"
        new_visible_reason = "new-visible support is more likely to be gated out"

    taxonomy = {
        "primary_case": primary_case,
        "primary_reason": primary_reason,
        "small_object_case": small_case,
        "small_object_reason": small_reason,
        "new_visible_case": new_visible_case,
        "new_visible_reason": new_visible_reason,
        "evidence": {
            "covered_ff_gate_pass_ratio": covered_ff_gate,
            "covered_tp_gate_pass_ratio": covered_tp_gate,
            "covered_ff_gt_class_score": covered_ff_gt_score,
            "covered_tp_gt_class_score": covered_tp_gt_score,
            "covered_ff_wrong_top1_ratio": covered_ff_wrong,
            "covered_tp_wrong_top1_ratio": covered_tp_wrong,
            "covered_ff_contributor_count": covered_ff_contrib,
            "covered_tp_contributor_count": covered_tp_contrib,
            "oracle_summary": oracle_summary,
        },
        "next_action": next_action,
    }
    write_json(REPORTS_DIR / "sw4_covered_ff_failure_taxonomy_decision.json", taxonomy)
    write_md(
        REPORTS_DIR / "sw4_covered_ff_failure_taxonomy_decision.md",
        "# SW-4.0 failure taxonomy decision\n\n"
        f"- primary case: `{primary_case}`\n"
        f"- primary reason: `{primary_reason}`\n"
        f"- small-object case: `{small_case}`\n"
        f"- new-visible case: `{new_visible_case}`\n"
        f"- next unique action: `{next_action}`\n",
    )

    report_payload = {
        "effective_samples": len(records),
        "effective_horizons": horizons,
        "get_occ_instrumentation": phase1,
        "covered_false_free_sampling_rows": len(sampling_rows),
        "support_level_semantic_score_rows": len(support_score_rows),
        "gate_filter_rows": len(gate_rows),
        "voxel_aggregation_rows": len(voxel_rows),
        "small_object_summary": {
            "mean_small_object_gt_class_score": small_ff_gt_score,
            "mean_large_dynamic_gt_class_score": large_dyn_gt_score,
            "mean_small_object_gt_rank": small_rank,
            "mean_large_dynamic_gt_rank": large_rank,
        },
        "new_visible_summary": {
            "mean_new_visible_gt_class_score": new_visible_gt_score,
            "mean_persistent_tp_gt_class_score": persistent_tp_gt_score,
            "mean_persistent_ff_gt_class_score": persistent_ff_gt_score,
        },
        "oracle_summary": oracle_summary,
        "failure_taxonomy": taxonomy,
        "safe_claims": {
            "subset_diagnostic_only": True,
            "official_benchmark": False,
            "training_completed": False,
            "oracle_interventions_are_diagnostic_only": True,
            "model_improvement_claim": False,
        },
    }
    write_json(REPORTS_DIR / "stage_sw4_covered_ff_semantic_activation_report.json", report_payload)
    report_md = [
        "# Stage SW-4.0 covered false-free semantic activation diagnosis",
        "",
        "## Executive summary",
        "",
        f"- effective samples: `{len(records)}`",
        f"- horizons: `{horizons}`",
        f"- get_occ equivalence passed: `{phase1['equivalence']['all_exact_equal']}`",
        f"- covered FF main case: `{primary_case}`",
        f"- small-object main case: `{small_case}`",
        f"- new-visible main case: `{new_visible_case}`",
        f"- next unique action: `{next_action}`",
        "",
        "## Safe claims",
        "",
        "- subset diagnostic only",
        "- oracle interventions are diagnostic only",
        "- no training",
        "- no official benchmark",
        "- no model improvement claim",
    ]
    write_md(REPORTS_DIR / "stage_sw4_covered_ff_semantic_activation_report.md", "\n".join(report_md))

    print(
        json.dumps(
            {
                "effective_samples": len(records),
                "primary_case": primary_case,
                "small_object_case": small_case,
                "new_visible_case": new_visible_case,
                "next_action": next_action,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
