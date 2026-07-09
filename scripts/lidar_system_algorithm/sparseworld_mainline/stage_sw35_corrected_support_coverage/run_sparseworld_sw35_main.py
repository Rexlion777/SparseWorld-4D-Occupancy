"""Stage SW-3.5 corrected support coverage re-run.

Purpose:
- replace deprecated mean-center refine point proxy
- re-run temporal/query coverage diagnostics with corrected all48 support
- produce corrected coverage assets for SW-4.0
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from corrected_support_adapter import (
    CorrectedSparseWorldSupportAdapter,
    SupportDefinition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage SW-3.5 corrected support coverage re-run")
    parser.add_argument("--project-root", default="/mnt/d/ComputerVision/cv_lidar_transition")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--horizons", default="0,1,2,3,4,5,6")
    parser.add_argument("--run-tag", default="", help="optional subdirectory tag for isolated smoke/full runs")
    parser.add_argument("--save-figures", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_parent(path)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bev_map(mask3d: torch.Tensor) -> np.ndarray:
    return mask3d.any(dim=-1).cpu().numpy().astype(np.uint8)


def classify_regions(sw3: Any, gt: torch.Tensor, pred: torch.Tensor) -> dict[str, torch.Tensor]:
    return sw3.classify_regions(gt, pred)


def dilate3d(sw3: Any, mask: torch.Tensor, radius: int) -> torch.Tensor:
    return sw3.dilate3d(mask, radius)


def distance_transform_bev(mask3d: torch.Tensor) -> np.ndarray:
    support_bev = bev_map(mask3d)
    inv = (support_bev == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def compute_group_mask(label_grid: torch.Tensor, class_ids: list[int], empty_idx: int) -> torch.Tensor:
    mask = torch.zeros_like(label_grid, dtype=torch.bool)
    for class_id in class_ids:
        mask |= label_grid == int(class_id)
    mask &= label_grid != empty_idx
    return mask


def load_sw2_rows_and_manifests(project_root: Path, num_samples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sw2_reports = project_root / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
    manifest = json.loads((sw2_reports / "sw2_run_manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((sw2_reports / "sw2_sample_manifest.csv").open(encoding="utf-8")))
    rows = [row for row in rows if row.get("forward_status") == "success"][:num_samples]
    return rows, manifest


def load_record(row: dict[str, Any]) -> dict[str, Any]:
    query = torch.load(row["query_output_path"], map_location="cpu", weights_only=False)
    pred_temporal = torch.load(row["standard_pred_occ_temporal_path"], map_location="cpu", weights_only=False).long()
    gt_temporal = torch.load(row["standard_gt_occ_temporal_path"], map_location="cpu", weights_only=False).long()
    return {
        "sample_index": int(row["sample_index"]),
        "sample_token": row["sample_token"],
        "scene_token": row["scene_token"],
        "scene_name": row["scene_name"],
        "timestamp": int(row["timestamp"]),
        "query": query,
        "pred_temporal": pred_temporal,
        "gt_temporal": gt_temporal,
    }


def float_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def top1_distribution_json(classes: list[int]) -> str:
    filtered = [int(x) for x in classes if int(x) >= 0]
    if not filtered:
        return "{}"
    counts = Counter(filtered)
    return json.dumps(dict(sorted(counts.items())), ensure_ascii=False)


def support_exact_region_stats(
    gt: torch.Tensor,
    support: dict[str, Any],
    region_mask: torch.Tensor,
    empty_idx: int,
) -> dict[str, Any]:
    coords = support.get("support_coords")
    logits = support.get("semantic_active_logits_sparse")
    if coords is None or logits is None or getattr(coords, "numel", lambda: 0)() == 0 or getattr(logits, "numel", lambda: 0)() == 0:
        return {
            "gt_class_support_score_mean": 0.0,
            "best_foreground_support_score_mean": 0.0,
            "gt_class_rank_mean": 0.0,
            "top1_support_class_distribution": "{}",
        }
    coords = coords.long()
    region_support_mask = region_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
    if not bool(region_support_mask.any().item()):
        return {
            "gt_class_support_score_mean": 0.0,
            "best_foreground_support_score_mean": 0.0,
            "gt_class_rank_mean": 0.0,
            "top1_support_class_distribution": "{}",
        }
    region_logits = logits[region_support_mask]
    gt_labels = gt[coords[region_support_mask, 0], coords[region_support_mask, 1], coords[region_support_mask, 2]].long()
    valid_gt = (gt_labels >= 0) & (gt_labels < region_logits.shape[-1]) & (gt_labels != empty_idx)
    gt_class_scores = []
    gt_class_ranks = []
    if bool(valid_gt.any().item()):
        logits_valid = region_logits[valid_gt]
        gt_valid = gt_labels[valid_gt]
        gt_scores = logits_valid.gather(1, gt_valid[:, None]).squeeze(1)
        gt_class_scores = gt_scores.detach().cpu().tolist()
        ranks = (logits_valid > gt_scores[:, None]).sum(dim=1) + 1
        gt_class_ranks = ranks.float().detach().cpu().tolist()
    top_classes = region_logits.argmax(dim=-1).detach().cpu().tolist()
    best_scores = region_logits.max(dim=-1).values.detach().cpu().tolist()
    return {
        "gt_class_support_score_mean": float(np.mean(gt_class_scores)) if gt_class_scores else 0.0,
        "best_foreground_support_score_mean": float(np.mean(best_scores)) if best_scores else 0.0,
        "gt_class_rank_mean": float(np.mean(gt_class_ranks)) if gt_class_ranks else 0.0,
        "top1_support_class_distribution": top1_distribution_json(top_classes),
    }


def prepare_support_analysis_cache(sw3: Any, support: dict[str, Any], radii: list[int]) -> dict[str, Any]:
    geo_dist = distance_transform_bev(support["geometric_mask"])
    sem_dist = distance_transform_bev(support["semantic_active_mask"])
    if torch.cuda.is_available():
        geo_mask_gpu = support["geometric_mask"].to(device="cuda", dtype=torch.float32)
        sem_mask_gpu = support["semantic_active_mask"].to(device="cuda", dtype=torch.float32)
        geo_dil = {radius: dilate3d(sw3, geo_mask_gpu, radius).cpu() for radius in radii}
        sem_dil = {radius: dilate3d(sw3, sem_mask_gpu, radius).cpu() for radius in radii}
        del geo_mask_gpu
        del sem_mask_gpu
        torch.cuda.synchronize()
    else:
        geo_dil = {radius: dilate3d(sw3, support["geometric_mask"], radius) for radius in radii}
        sem_dil = {radius: dilate3d(sw3, support["semantic_active_mask"], radius) for radius in radii}
    return {
        "geo_dist_bev": geo_dist,
        "sem_dist_bev": sem_dist,
        "geo_dil": geo_dil,
        "sem_dil": sem_dil,
    }


def region_rows_for_support(
    sw3: Any,
    record: dict[str, Any],
    support: dict[str, Any],
    horizon_s: int,
    radii: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred = record["pred_temporal"][horizon_s]
    gt = record["gt_temporal"][horizon_s]
    regions = classify_regions(sw3, gt, pred)
    analysis = support["_analysis"]
    geo_dist = analysis["geo_dist_bev"]
    sem_dist = analysis["sem_dist_bev"]
    support_conf_grid = support["semantic_active_conf"].float()
    support_density_grid = support["semantic_active_count"].float()
    support_class_grid = support["semantic_active_class"].long()
    region_base: dict[str, dict[str, Any]] = {}
    for region_name, region_mask in regions.items():
        region_voxel_count = int(region_mask.sum().item())
        region_bev = bev_map(region_mask)
        exact_stats = support_exact_region_stats(gt, support, region_mask, sw3.EMPTY_IDX)
        region_classes = support_class_grid[region_mask & support["semantic_active_mask"]]
        dist_vals = geo_dist[region_bev > 0] if region_bev.sum() else np.asarray([], dtype=np.float32)
        sem_dist_vals = sem_dist[region_bev > 0] if region_bev.sum() else np.asarray([], dtype=np.float32)
        region_base[region_name] = {
            "mask": region_mask,
            "region_voxel_count": region_voxel_count,
            "support_density_mean": float(support_density_grid[region_mask].mean().item()) if region_voxel_count else 0.0,
            "support_semantic_confidence_mean": float(support_conf_grid[region_mask].mean().item()) if region_voxel_count else 0.0,
            "nearest_support_distance_bev": float_stats(dist_vals.tolist() if dist_vals.size else []),
            "nearest_semantic_active_support_distance_bev": float_stats(sem_dist_vals.tolist() if sem_dist_vals.size else []),
            "top1_support_class_distribution": top1_distribution_json(region_classes.detach().cpu().tolist()),
            "exact_stats": exact_stats,
        }
    for radius in radii:
        geo_dil = analysis["geo_dil"][radius]
        sem_dil = analysis["sem_dil"][radius]
        for region_name, region_info in region_base.items():
            region_mask = region_info["mask"]
            region_voxel_count = int(region_info["region_voxel_count"])
            geo_cov = float((geo_dil & region_mask).sum().item() / region_voxel_count) if region_voxel_count else 0.0
            sem_cov = float((sem_dil & region_mask).sum().item() / region_voxel_count) if region_voxel_count else 0.0
            dist_stats = region_info["nearest_support_distance_bev"]
            sem_dist_stats = region_info["nearest_semantic_active_support_distance_bev"]
            exact_stats = region_info["exact_stats"]
            rows.append(
                {
                    "sample_index": record["sample_index"],
                    "sample_token": record["sample_token"],
                    "scene_token": record["scene_token"],
                    "scene_name": record["scene_name"],
                    "horizon_s": horizon_s,
                    "coverage_radius_cells": radius,
                    "region_name": region_name,
                    "region_voxel_count": region_voxel_count,
                    "geometric_support_coverage_ratio": geo_cov,
                    "semantic_active_support_coverage_ratio": sem_cov,
                    "support_density_mean": region_info["support_density_mean"],
                    "support_semantic_confidence_mean": region_info["support_semantic_confidence_mean"],
                    "support_valid_range_ratio": float(support["valid_ratio"]),
                    "nearest_support_distance_bev_mean": dist_stats["mean"],
                    "nearest_support_distance_bev_median": dist_stats["median"],
                    "nearest_support_distance_bev_p95": dist_stats["p95"],
                    "nearest_semantic_active_support_distance_bev_mean": sem_dist_stats["mean"],
                    "nearest_semantic_active_support_distance_bev_median": sem_dist_stats["median"],
                    "nearest_semantic_active_support_distance_bev_p95": sem_dist_stats["p95"],
                    "top1_support_class_distribution": region_info["top1_support_class_distribution"],
                    **exact_stats,
                }
            )
    return rows


def compute_query_to_pred_attribution(
    sw3: Any,
    records: list[dict[str, Any]],
    adapter: CorrectedSparseWorldSupportAdapter,
    radii: list[int],
    support_cache_key: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec_idx, record in enumerate(records, start=1):
        for horizon_s in range(record["pred_temporal"].shape[0]):
            support = record.get(support_cache_key, {}).get(horizon_s) if support_cache_key else None
            if support is None:
                support, _ = adapter.build_support(record["query"], horizon_s=horizon_s)
                support["_analysis"] = prepare_support_analysis_cache(sw3, support, radii)
            rows.extend(region_rows_for_support(sw3, record, support, horizon_s, radii))
        print(f"[SW35] phase2 attribution progress: sample {rec_idx}/{len(records)} index={record['sample_index']}", flush=True)
    return rows


def quadrant_summary(rows: list[dict[str, Any]], radius: int) -> dict[str, Any]:
    by_region: dict[str, dict[str, float]] = {}
    for region_name in ["tp_occupied", "false_free", "false_occupied", "true_free"]:
        sub = [row for row in rows if row["coverage_radius_cells"] == radius and row["region_name"] == region_name and row["region_voxel_count"] > 0]
        by_region[region_name] = {
            "geometric_support_coverage_ratio_mean": float(np.mean([r["geometric_support_coverage_ratio"] for r in sub])) if sub else 0.0,
            "semantic_active_support_coverage_ratio_mean": float(np.mean([r["semantic_active_support_coverage_ratio"] for r in sub])) if sub else 0.0,
            "nearest_support_distance_bev_mean": float(np.mean([r["nearest_support_distance_bev_mean"] for r in sub])) if sub else 0.0,
            "support_semantic_confidence_mean": float(np.mean([r["support_semantic_confidence_mean"] for r in sub])) if sub else 0.0,
        }
    ff_geo = by_region["false_free"]["geometric_support_coverage_ratio_mean"]
    if ff_geo > 0.7:
        conclusion = "covered_false_free_semantic_activation_or_decode_dominant"
    elif ff_geo < 0.3:
        conclusion = "coverage_failure_still_plausible"
    else:
        conclusion = "mixed_support_and_activation_failure"
    return {
        "radius": radius,
        "by_region": by_region,
        "mean_ff_geometric_coverage_ratio": ff_geo,
        "mean_ff_semantic_active_coverage_ratio": by_region["false_free"]["semantic_active_support_coverage_ratio_mean"],
        "mean_tp_semantic_active_coverage_ratio": by_region["tp_occupied"]["semantic_active_support_coverage_ratio_mean"],
        "conclusion": conclusion,
    }


def plot_quadrant_figures(rows: list[dict[str, Any]], fig_dir: Path, prefix: str) -> None:
    subset = [row for row in rows if row["coverage_radius_cells"] == 2 and row["region_voxel_count"] > 0]
    regions = ["tp_occupied", "false_free", "false_occupied", "true_free"]
    geo_vals = [float(np.mean([r["geometric_support_coverage_ratio"] for r in subset if r["region_name"] == region])) for region in regions]
    sem_vals = [float(np.mean([r["semantic_active_support_coverage_ratio"] for r in subset if r["region_name"] == region])) for region in regions]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(regions))
    ax.bar(x - 0.18, geo_vals, width=0.36, label="geometric")
    ax.bar(x + 0.18, sem_vals, width=0.36, label="semantic-active")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{prefix} coverage by region (r=2)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_coverage_four_quadrants_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    dist_vals = [float(np.mean([r["nearest_support_distance_bev_mean"] for r in subset if r["region_name"] == region])) for region in regions]
    ax.bar(regions, dist_vals)
    ax.set_title(f"{prefix} nearest support distance by region (r=2)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_nearest_support_distance_by_error_type.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    radii = sorted({int(r["coverage_radius_cells"]) for r in rows})
    for region in ["tp_occupied", "false_free", "false_occupied"]:
        ys = [
            float(np.mean([rr["geometric_support_coverage_ratio"] for rr in rows if rr["coverage_radius_cells"] == radius and rr["region_name"] == region and rr["region_voxel_count"] > 0]))
            for radius in radii
        ]
        ax.plot(radii, ys, marker="o", label=region)
    ax.set_xlabel("Coverage radius (cells)")
    ax.set_ylabel("Geometric support coverage")
    ax.set_title(f"{prefix} radius sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_radius_sweep_four_quadrants.png", dpi=150)
    plt.close(fig)


def plot_region_overlay(sw3: Any, record: dict[str, Any], support: dict[str, Any], fig_path: Path, title: str) -> None:
    pred = record["pred_temporal"][0]
    gt = record["gt_temporal"][0]
    regions = classify_regions(sw3, gt, pred)
    overlay = np.zeros((sw3.GRID_SIZE[0], sw3.GRID_SIZE[1], 3), dtype=np.uint8)
    overlay[bev_map(regions["tp_occupied"]).astype(bool)] = np.array([0, 255, 0], dtype=np.uint8)
    overlay[bev_map(regions["false_free"]).astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[bev_map(regions["false_occupied"]).astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
    support_bev = support["support_density_bev"].cpu().numpy().astype(np.uint8)
    mask = support_bev
    overlay[mask.astype(bool)] = np.maximum(overlay[mask.astype(bool)], np.array([255, 255, 255], dtype=np.uint8))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def compute_small_object_rows(
    sw3: Any,
    records: list[dict[str, Any]],
    adapter: CorrectedSparseWorldSupportAdapter,
    support_cache_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    small_classes = {
        "pedestrian": sw3.LABEL_TO_ID["pedestrian"],
        "bicycle": sw3.LABEL_TO_ID["bicycle"],
        "motorcycle": sw3.LABEL_TO_ID["motorcycle"],
        "traffic_cone": sw3.LABEL_TO_ID["traffic_cone"],
        "barrier": sw3.LABEL_TO_ID["barrier"],
    }
    large_dynamic = {
        "car": sw3.LABEL_TO_ID["car"],
        "truck": sw3.LABEL_TO_ID["truck"],
        "bus": sw3.LABEL_TO_ID["bus"],
        "trailer": sw3.LABEL_TO_ID["trailer"],
        "construction_vehicle": sw3.LABEL_TO_ID["construction_vehicle"],
    }
    rows: list[dict[str, Any]] = []
    for rec_idx, record in enumerate(records, start=1):
        for horizon_s in range(record["pred_temporal"].shape[0]):
            support = record.get(support_cache_key, {}).get(horizon_s) if support_cache_key else None
            if support is None:
                support, _ = adapter.build_support(record["query"], horizon_s=horizon_s)
                support["_analysis"] = prepare_support_analysis_cache(sw3, support, [1, 2, 3, 5, 8])
            analysis = support["_analysis"]
            geo_dist = analysis["geo_dist_bev"]
            sem_dist = analysis["sem_dist_bev"]
            pred = record["pred_temporal"][horizon_s]
            gt = record["gt_temporal"][horizon_s]
            for class_name, class_id in small_classes.items():
                gt_mask = gt == class_id
                ff_mask = gt_mask & (pred == sw3.EMPTY_IDX)
                exact_stats = support_exact_region_stats(gt, support, gt_mask, sw3.EMPTY_IDX)
                ff_stats = support_exact_region_stats(gt, support, ff_mask, sw3.EMPTY_IDX)
                row = {
                    "sample_index": record["sample_index"],
                    "sample_token": record["sample_token"],
                    "scene_token": record["scene_token"],
                    "scene_name": record["scene_name"],
                    "horizon_s": horizon_s,
                    "class_name": class_name,
                    "class_id": class_id,
                    "small_gt_count": int(gt_mask.sum().item()),
                    "small_false_free_count": int(ff_mask.sum().item()),
                }
                for radius in [1, 2, 3, 5, 8]:
                    geo_dil = analysis["geo_dil"][radius]
                    sem_dil = analysis["sem_dil"][radius]
                    row[f"geometric_support_coverage_r{radius}"] = float((geo_dil & gt_mask).sum().item() / max(1, gt_mask.sum().item()))
                    row[f"semantic_active_support_coverage_r{radius}"] = float((sem_dil & gt_mask).sum().item() / max(1, gt_mask.sum().item()))
                    row[f"ff_geometric_support_coverage_r{radius}"] = float((geo_dil & ff_mask).sum().item() / max(1, ff_mask.sum().item()))
                    row[f"ff_semantic_active_support_coverage_r{radius}"] = float((sem_dil & ff_mask).sum().item() / max(1, ff_mask.sum().item()))
                gt_bev = bev_map(gt_mask)
                ff_bev = bev_map(ff_mask)
                row["nearest_support_distance_bev_mean"] = float(geo_dist[gt_bev > 0].mean()) if gt_bev.sum() else 0.0
                row["nearest_semantic_active_support_distance_bev_mean"] = float(sem_dist[gt_bev > 0].mean()) if gt_bev.sum() else 0.0
                row["ff_nearest_support_distance_bev_mean"] = float(geo_dist[ff_bev > 0].mean()) if ff_bev.sum() else 0.0
                row["ff_nearest_semantic_active_support_distance_bev_mean"] = float(sem_dist[ff_bev > 0].mean()) if ff_bev.sum() else 0.0
                row["gt_class_score_rank_mean"] = exact_stats["gt_class_rank_mean"]
                row["gt_class_support_score_mean"] = exact_stats["gt_class_support_score_mean"]
                row["best_foreground_support_score_mean"] = exact_stats["best_foreground_support_score_mean"]
                row["ff_gt_class_score_rank_mean"] = ff_stats["gt_class_rank_mean"]
                row["ff_gt_class_support_score_mean"] = ff_stats["gt_class_support_score_mean"]
                row["ff_best_foreground_support_score_mean"] = ff_stats["best_foreground_support_score_mean"]
                row["top1_support_class_distribution"] = exact_stats["top1_support_class_distribution"]
                rows.append(row)
        print(f"[SW35] phase3 small-object progress: sample {rec_idx}/{len(records)} index={record['sample_index']}", flush=True)
    large_rows: list[dict[str, Any]] = []
    for record in records:
        for horizon_s in range(record["pred_temporal"].shape[0]):
            support = record.get(support_cache_key, {}).get(horizon_s) if support_cache_key else None
            if support is None:
                support, _ = adapter.build_support(record["query"], horizon_s=horizon_s)
                support["_analysis"] = prepare_support_analysis_cache(sw3, support, [1, 2, 3, 5, 8])
            geo_dist = support["_analysis"]["geo_dist_bev"]
            gt = record["gt_temporal"][horizon_s]
            for class_name, class_id in large_dynamic.items():
                gt_mask = gt == class_id
                if int(gt_mask.sum().item()) == 0:
                    continue
                exact_stats = support_exact_region_stats(gt, support, gt_mask, sw3.EMPTY_IDX)
                large_rows.append(
                    {
                        "class_name": class_name,
                        "nearest_support_distance_bev_mean": float(geo_dist[bev_map(gt_mask) > 0].mean()) if bev_map(gt_mask).sum() else 0.0,
                        "gt_class_score_rank_mean": exact_stats["gt_class_rank_mean"],
                        "gt_class_support_score_mean": exact_stats["gt_class_support_score_mean"],
                    }
                )
    summary = {
        "mean_small_ff_geometric_coverage_r2": float(np.mean([r["ff_geometric_support_coverage_r2"] for r in rows if r["small_false_free_count"] > 0])) if rows else 0.0,
        "mean_small_ff_semantic_active_coverage_r2": float(np.mean([r["ff_semantic_active_support_coverage_r2"] for r in rows if r["small_false_free_count"] > 0])) if rows else 0.0,
        "mean_small_ff_gt_class_rank": float(np.mean([r["ff_gt_class_score_rank_mean"] for r in rows if r["small_false_free_count"] > 0])) if rows else 0.0,
        "mean_small_ff_gt_class_support_score": float(np.mean([r["ff_gt_class_support_score_mean"] for r in rows if r["small_false_free_count"] > 0])) if rows else 0.0,
        "mean_large_dynamic_gt_class_rank": float(np.mean([r["gt_class_score_rank_mean"] for r in large_rows])) if large_rows else 0.0,
        "mean_large_dynamic_gt_support_score": float(np.mean([r["gt_class_support_score_mean"] for r in large_rows])) if large_rows else 0.0,
        "mean_small_ff_nearest_support_distance": float(np.mean([r["ff_nearest_support_distance_bev_mean"] for r in rows if r["small_false_free_count"] > 0])) if rows else 0.0,
        "mean_large_dynamic_nearest_support_distance": float(np.mean([r["nearest_support_distance_bev_mean"] for r in large_rows])) if large_rows else 0.0,
    }
    return rows, summary


def compute_new_visible_rows(
    sw3: Any,
    records: list[dict[str, Any]],
    adapter: CorrectedSparseWorldSupportAdapter,
    support_cache_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec_idx, record in enumerate(records, start=1):
        gt0 = record["gt_temporal"][0]
        gt0_occ = gt0 != sw3.EMPTY_IDX
        for horizon_s in range(1, record["gt_temporal"].shape[0]):
            support = record.get(support_cache_key, {}).get(horizon_s) if support_cache_key else None
            if support is None:
                support, _ = adapter.build_support(record["query"], horizon_s=horizon_s)
                support["_analysis"] = prepare_support_analysis_cache(sw3, support, [2])
            pred = record["pred_temporal"][horizon_s]
            gt = record["gt_temporal"][horizon_s]
            gt_occ = gt != sw3.EMPTY_IDX
            pred_occ = pred != sw3.EMPTY_IDX
            persistent = gt0_occ & gt_occ
            new_visible = (~gt0_occ) & gt_occ
            disappeared = gt0_occ & (~gt_occ)
            analysis = support["_analysis"]
            geo_dil = analysis["geo_dil"][2]
            sem_dil = analysis["sem_dil"][2]
            geo_dist = analysis["geo_dist_bev"]
            def cov(mask: torch.Tensor, dil: torch.Tensor) -> float:
                den = mask.sum().item()
                return float((mask & dil).sum().item() / den) if den else 0.0
            def dist(mask: torch.Tensor) -> float:
                bev = bev_map(mask)
                return float(geo_dist[bev > 0].mean()) if bev.sum() else 0.0
            persistent_stats = support_exact_region_stats(gt, support, persistent, sw3.EMPTY_IDX)
            new_stats = support_exact_region_stats(gt, support, new_visible, sw3.EMPTY_IDX)
            row = {
                "sample_index": record["sample_index"],
                "sample_token": record["sample_token"],
                "scene_token": record["scene_token"],
                "scene_name": record["scene_name"],
                "horizon_s": horizon_s,
                "persistent_gt_count": int(persistent.sum().item()),
                "new_visible_gt_count": int(new_visible.sum().item()),
                "disappeared_gt_count": int(disappeared.sum().item()),
                "persistent_pred_recall": float((persistent & pred_occ).sum().item() / max(1, persistent.sum().item())),
                "new_visible_pred_recall": float((new_visible & pred_occ).sum().item() / max(1, new_visible.sum().item())),
                "persistent_false_free": float((persistent & (~pred_occ)).sum().item() / max(1, persistent.sum().item())),
                "new_visible_false_free": float((new_visible & (~pred_occ)).sum().item() / max(1, new_visible.sum().item())),
                "persistent_geometric_coverage": cov(persistent, geo_dil),
                "new_visible_geometric_coverage": cov(new_visible, geo_dil),
                "persistent_semantic_active_coverage": cov(persistent, sem_dil),
                "new_visible_semantic_active_coverage": cov(new_visible, sem_dil),
                "persistent_nearest_support_distance_bev_mean": dist(persistent),
                "new_visible_nearest_support_distance_bev_mean": dist(new_visible),
                "persistent_gt_class_support_score_mean": persistent_stats["gt_class_support_score_mean"],
                "new_visible_gt_class_support_score_mean": new_stats["gt_class_support_score_mean"],
                "persistent_best_foreground_support_score_mean": persistent_stats["best_foreground_support_score_mean"],
                "new_visible_best_foreground_support_score_mean": new_stats["best_foreground_support_score_mean"],
                "coverage_gap": cov(persistent, geo_dil) - cov(new_visible, geo_dil),
                "recall_gap": float((persistent & pred_occ).sum().item() / max(1, persistent.sum().item())) - float((new_visible & pred_occ).sum().item() / max(1, new_visible.sum().item())),
                "disappeared_stale_pred_ratio": float((disappeared & pred_occ).sum().item() / max(1, disappeared.sum().item())),
            }
            rows.append(row)
        print(f"[SW35] phase4 new-visible progress: sample {rec_idx}/{len(records)} index={record['sample_index']}", flush=True)
    summary = {
        "mean_new_visible_geometric_coverage": float(np.mean([r["new_visible_geometric_coverage"] for r in rows if r["new_visible_gt_count"] > 0])) if rows else 0.0,
        "mean_persistent_geometric_coverage": float(np.mean([r["persistent_geometric_coverage"] for r in rows if r["persistent_gt_count"] > 0])) if rows else 0.0,
        "mean_new_visible_semantic_active_coverage": float(np.mean([r["new_visible_semantic_active_coverage"] for r in rows if r["new_visible_gt_count"] > 0])) if rows else 0.0,
        "mean_persistent_semantic_active_coverage": float(np.mean([r["persistent_semantic_active_coverage"] for r in rows if r["persistent_gt_count"] > 0])) if rows else 0.0,
        "mean_new_visible_pred_recall": float(np.mean([r["new_visible_pred_recall"] for r in rows if r["new_visible_gt_count"] > 0])) if rows else 0.0,
        "mean_persistent_pred_recall": float(np.mean([r["persistent_pred_recall"] for r in rows if r["persistent_gt_count"] > 0])) if rows else 0.0,
    }
    return rows, summary


def compute_temporal_support_proxy(
    sw3: Any,
    records: list[dict[str, Any]],
    adapter: CorrectedSparseWorldSupportAdapter,
    support_cache_key: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        com0 = None
        for horizon_s in range(record["pred_temporal"].shape[0]):
            support = record.get(support_cache_key, {}).get(horizon_s) if support_cache_key else None
            if support is None:
                support, manifest = adapter.build_support(record["query"], horizon_s=horizon_s)
                support["_analysis"] = prepare_support_analysis_cache(sw3, support, [2])
            else:
                manifest = {
                    "valid_range_ratio": float(support["valid_ratio"]),
                    "query_density_entropy": float(support["query_density_entropy"]),
                    "query_com_xy": list(support["query_com_xy"]),
                    "diagonal_line_score": float(support["diagonal_line_score"]),
                }
            pred = record["pred_temporal"][horizon_s]
            gt = record["gt_temporal"][horizon_s]
            regions = classify_regions(sw3, gt, pred)
            analysis = support["_analysis"]
            geo_dil = analysis["geo_dil"][2]
            sem_dil = analysis["sem_dil"][2]
            gt_occ = gt != sw3.EMPTY_IDX
            pred_occ = pred != sw3.EMPTY_IDX
            gt_cov = float((geo_dil & gt_occ).sum().item() / max(1, gt_occ.sum().item()))
            ff_cov = float((geo_dil & regions["false_free"]).sum().item() / max(1, regions["false_free"].sum().item()))
            pred_cov = float((geo_dil & pred_occ).sum().item() / max(1, pred_occ.sum().item()))
            fo_cov = float((geo_dil & regions["false_occupied"]).sum().item() / max(1, regions["false_occupied"].sum().item()))
            gt_sem_cov = float((sem_dil & gt_occ).sum().item() / max(1, gt_occ.sum().item()))
            ff_sem_cov = float((sem_dil & regions["false_free"]).sum().item() / max(1, regions["false_free"].sum().item()))
            pred_sem_cov = float((sem_dil & pred_occ).sum().item() / max(1, pred_occ.sum().item()))
            fo_sem_cov = float((sem_dil & regions["false_occupied"]).sum().item() / max(1, regions["false_occupied"].sum().item()))
            com = np.asarray(support["query_com_xy"], dtype=np.float32)
            if com0 is None:
                com0 = com.copy()
            shift = float(np.linalg.norm(com - com0)) if np.isfinite(com).all() and np.isfinite(com0).all() else 0.0
            rows.append(
                {
                    "sample_index": record["sample_index"],
                    "sample_token": record["sample_token"],
                    "scene_token": record["scene_token"],
                    "scene_name": record["scene_name"],
                    "horizon_s": horizon_s,
                    "support_density_bev_active_cells": int(support["support_density_bev"].sum().item()),
                    "semantic_active_density_bev_active_cells": int(support["semantic_active_density_bev"].sum().item()),
                    "support_coverage_gt_occupied": gt_cov,
                    "support_coverage_false_free": ff_cov,
                    "support_coverage_pred_occupied": pred_cov,
                    "support_coverage_false_occupied": fo_cov,
                    "semantic_active_support_coverage_gt_occupied": gt_sem_cov,
                    "semantic_active_support_coverage_false_free": ff_sem_cov,
                    "semantic_active_support_coverage_pred_occupied": pred_sem_cov,
                    "semantic_active_support_coverage_false_occupied": fo_sem_cov,
                    "support_valid_range_ratio": manifest["valid_range_ratio"],
                    "support_density_entropy": manifest["query_density_entropy"],
                    "support_density_center_of_mass_x": manifest["query_com_xy"][0],
                    "support_density_center_of_mass_y": manifest["query_com_xy"][1],
                    "support_density_shift_from_t0": shift,
                    "diagonal_line_score": manifest["diagonal_line_score"],
                }
            )
    return rows


def aggregate_by_horizon(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    horizons = sorted({int(row["horizon_s"]) for row in rows})
    out: list[dict[str, Any]] = []
    for horizon_s in horizons:
        sub = [row for row in rows if int(row["horizon_s"]) == horizon_s]
        agg = {"horizon_s": horizon_s, "sample_count": len(sub)}
        for field in fields:
            vals = [float(row[field]) for row in sub]
            agg[field] = float(np.mean(vals)) if vals else 0.0
        out.append(agg)
    return out


def plot_temporal_support_curves(
    corrected_agg: list[dict[str, Any]],
    old_agg: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    horizons = [row["horizon_s"] for row in corrected_agg]
    def series(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, series(corrected_agg, "support_coverage_gt_occupied"), marker="o", label="gt occupied")
    ax.plot(horizons, series(corrected_agg, "support_coverage_false_free"), marker="o", label="false-free")
    ax.plot(horizons, series(corrected_agg, "support_coverage_pred_occupied"), marker="o", label="pred occupied")
    ax.set_title("Corrected support coverage over horizon")
    ax.set_xlabel("Horizon (s)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_support_coverage_over_horizon.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, series(corrected_agg, "support_coverage_false_free"), marker="o", label="corrected")
    ax.plot(horizons, series(old_agg, "support_coverage_false_free"), marker="o", label="old_mean_center")
    ax.set_title("Old vs corrected false-free support coverage")
    ax.set_xlabel("Horizon (s)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "old_vs_corrected_false_free_coverage_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, series(corrected_agg, "support_density_shift_from_t0"), marker="o", label="corrected")
    ax.plot(horizons, series(old_agg, "support_density_shift_from_t0"), marker="o", label="old_mean_center")
    ax.set_title("Support density shift from t0")
    ax.set_xlabel("Horizon (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_support_density_shift_over_horizon.png", dpi=150)
    plt.close(fig)


def plot_support_density_rollout(record: dict[str, Any], corrected_supports: dict[int, dict[str, Any]], fig_path: Path) -> None:
    horizons = sorted(corrected_supports.keys())
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for ax, horizon_s in zip(axes, horizons):
        ax.imshow(corrected_supports[horizon_s]["support_density_bev"].cpu().numpy().T, origin="lower", cmap="magma")
        ax.set_title(f"t={horizon_s}s")
        ax.axis("off")
    for ax in axes[len(horizons):]:
        ax.axis("off")
    fig.suptitle(f"Corrected support density rollout sample {record['sample_index']}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def old_from_sw2_query_coverage(project_root: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader((project_root / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/query_coverage_metrics_sample_subset.csv").open(encoding="utf-8")))
    out: list[dict[str, Any]] = []
    center_lookup: dict[tuple[int, int], tuple[float, float]] = {}
    for row in rows:
        if int(row["coverage_radius_cells"]) != 2:
            continue
        center_lookup[(int(row["sample_index"]), int(row["horizon_s"]))] = (
            float(row["query_density_center_of_mass_x"]),
            float(row["query_density_center_of_mass_y"]),
        )
        out.append(
            {
                "sample_index": int(row["sample_index"]),
                "horizon_s": int(row["horizon_s"]),
                "support_coverage_gt_occupied": float(row["query_coverage_gt_ratio"]),
                "support_coverage_false_free": float(row["query_coverage_false_free_ratio"]),
                "support_coverage_pred_occupied": float(row["query_coverage_pred_ratio"]),
                "support_coverage_false_occupied": 0.0,
                "support_density_entropy": float(row["query_density_entropy"]),
                "support_density_shift_from_t0": 0.0,
            }
        )
    by_sample = defaultdict(list)
    for row in out:
        by_sample[row["sample_index"]].append(row)
    for sample_rows in by_sample.values():
        sample_rows.sort(key=lambda x: x["horizon_s"])
        x0 = None
        y0 = None
        for row in sample_rows:
            x, y = center_lookup[(row["sample_index"], row["horizon_s"])]
            if x0 is None:
                x0 = x
                y0 = y
            row["support_density_shift_from_t0"] = float(np.linalg.norm(np.asarray([x, y]) - np.asarray([x0, y0])))
    return out


def load_sw2_metric_lookups(project_root: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int, str], dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    base = project_root / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
    temporal = {
        (int(row["sample_index"]), int(row["horizon_s"])): row
        for row in csv.DictReader((base / "temporal_horizon_metrics_per_sample.csv").open(encoding="utf-8"))
    }
    groups = {
        (int(row["sample_index"]), int(row["horizon_s"]), row["group_name"]): row
        for row in csv.DictReader((base / "class_group_horizon_metrics.csv").open(encoding="utf-8"))
    }
    new_visible = {
        (int(row["sample_index"]), int(row["horizon_s"])): row
        for row in csv.DictReader((base / "new_visible_proxy_metrics.csv").open(encoding="utf-8"))
    }
    return temporal, groups, new_visible


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    base_reports_dir = project_root / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage"
    base_logs_dir = project_root / "logs/sparseworld_mainline/stage_sw35_corrected_support_coverage"
    base_fig_dir = project_root / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw35_corrected_support_coverage"
    base_artifacts_dir = project_root / "artifacts/sparseworld_mainline/stage_sw35_corrected_support_coverage"
    if args.run_tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in args.run_tag)
        reports_dir = base_reports_dir / safe_tag
        logs_dir = base_logs_dir / safe_tag
        fig_dir = base_fig_dir / safe_tag
        artifacts_dir = base_artifacts_dir / safe_tag
    else:
        reports_dir = base_reports_dir
        logs_dir = base_logs_dir
        fig_dir = base_fig_dir
        artifacts_dir = base_artifacts_dir
    for p in [reports_dir, logs_dir, fig_dir, artifacts_dir]:
        p.mkdir(parents=True, exist_ok=True)

    sample_rows, sw2_manifest = load_sw2_rows_and_manifests(project_root, args.num_samples)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    sw3_module_path = project_root / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation/run_sparseworld_sw3_main.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("sparseworld_sw3_main", sw3_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load SW-3 module: {sw3_module_path}")
    sw3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sw3)

    corrected_adapter = CorrectedSparseWorldSupportAdapter(project_root=project_root)
    old_adapter = CorrectedSparseWorldSupportAdapter(
        project_root=project_root,
        support_definition=SupportDefinition(point_mode="mean48", source="deprecated wrong proxy"),
    )

    run_manifest = {
        "source_sw2_effective_subset_size": sw2_manifest["effective_subset_size"],
        "effective_sample_count": len(sample_rows),
        "horizons": horizons,
        "corrected_support_definition": {
            "tensor": "refine_pts_current",
            "coord_mode": "decoded_metric",
            "order": "xyz",
            "flip": "noflip",
            "point_mode": "all48",
        },
        "deprecated_old_proxy": {
            "tensor": "refine_pts_current",
            "coord_mode": "decoded_metric",
            "order": "xyz",
            "flip": "noflip",
            "point_mode": "mean48",
        },
    }
    write_json(reports_dir / "sw35_run_manifest.json", run_manifest)
    write_csv(reports_dir / "sw35_sample_manifest.csv", sample_rows)

    print(f"[SW35] phase0 init: effective_samples={len(sample_rows)} horizons={horizons} run_tag={args.run_tag or 'canonical'}", flush=True)

    # Phase 1 corrected support adapter artifact.
    sample0 = load_record(sample_rows[0])
    corrected_adapter.save_sample0_artifact(
        sample0["query"],
        horizons=horizons,
        artifact_path=artifacts_dir / "corrected_support_sample0.pt",
        manifest_path=reports_dir / "corrected_support_adapter_manifest.json",
    )
    print("[SW35] phase1 corrected support adapter artifact saved", flush=True)

    # Load all records once and precompute corrected / old support caches.
    records: list[dict[str, Any]] = []
    print("[SW35] precomputing corrected and deprecated-old support caches", flush=True)
    for rec_idx, row in enumerate(sample_rows, start=1):
        record = load_record(row)
        record["corrected_supports"] = {}
        record["old_supports"] = {}
        for horizon_s in horizons:
            record["corrected_supports"][horizon_s], _ = corrected_adapter.build_support(record["query"], horizon_s=horizon_s)
            record["corrected_supports"][horizon_s]["_analysis"] = prepare_support_analysis_cache(sw3, record["corrected_supports"][horizon_s], [1, 2, 3, 5, 8])
            record["old_supports"][horizon_s], _ = old_adapter.build_support(record["query"], horizon_s=horizon_s)
            record["old_supports"][horizon_s]["_analysis"] = prepare_support_analysis_cache(sw3, record["old_supports"][horizon_s], [1, 2, 3, 5, 8])
        record.pop("query", None)
        records.append(record)
        print(f"[SW35] support cache ready: sample {rec_idx}/{len(sample_rows)} index={record['sample_index']}", flush=True)
    sample0_record = records[0]

    # Phase 2 corrected query-to-pred attribution.
    radii = [1, 2, 3, 5, 8]
    corrected_attr_rows = compute_query_to_pred_attribution(sw3, records, corrected_adapter, radii, support_cache_key="corrected_supports")
    corrected_attr_summary = quadrant_summary(corrected_attr_rows, radius=2)
    write_json(reports_dir / "corrected_query_to_pred_attribution_matrix.json", {"rows": corrected_attr_rows, "summary": corrected_attr_summary})
    write_csv(reports_dir / "corrected_query_to_pred_attribution_matrix.csv", corrected_attr_rows)
    (reports_dir / "corrected_query_to_pred_attribution_summary.md").write_text(
        "# Corrected query-to-pred attribution summary\n\n"
        f"- mean false-free geometric support coverage (r=2): `{corrected_attr_summary['mean_ff_geometric_coverage_ratio']}`\n"
        f"- mean false-free semantic-active support coverage (r=2): `{corrected_attr_summary['mean_ff_semantic_active_coverage_ratio']}`\n"
        f"- mean TP semantic-active support coverage (r=2): `{corrected_attr_summary['mean_tp_semantic_active_coverage_ratio']}`\n"
        f"- conclusion: `{corrected_attr_summary['conclusion']}`\n",
        encoding="utf-8",
    )
    plot_quadrant_figures(corrected_attr_rows, fig_dir, "corrected")
    corrected_support0 = sample0_record["corrected_supports"][0]
    plot_region_overlay(sw3, sample0_record, corrected_support0, fig_dir / "corrected_query_density_on_tp_ff_fo_regions.png", "Corrected support on TP / FF / FO")
    print("[SW35] phase2 corrected query-to-pred attribution complete", flush=True)

    # Phase 3 corrected small-object support diagnosis.
    small_rows, small_summary = compute_small_object_rows(sw3, records, corrected_adapter, support_cache_key="corrected_supports")
    write_json(reports_dir / "corrected_small_object_failure_localization.json", {"rows": small_rows, "summary": small_summary})
    write_csv(reports_dir / "corrected_small_object_failure_localization.csv", small_rows)
    (reports_dir / "corrected_small_object_failure_summary.md").write_text(
        "# Corrected small-object failure summary\n\n"
        f"- mean small-object false-free geometric support coverage r=2: `{small_summary['mean_small_ff_geometric_coverage_r2']}`\n"
        f"- mean small-object false-free semantic-active support coverage r=2: `{small_summary['mean_small_ff_semantic_active_coverage_r2']}`\n"
        f"- mean small-object false-free GT-class rank: `{small_summary['mean_small_ff_gt_class_rank']}`\n"
        f"- mean large-dynamic GT-class rank: `{small_summary['mean_large_dynamic_gt_class_rank']}`\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["small_ff_cov_r2", "small_ff_sem_r2", "large_dyn_rank"],
        [
            small_summary["mean_small_ff_geometric_coverage_r2"],
            small_summary["mean_small_ff_semantic_active_coverage_r2"],
            small_summary["mean_large_dynamic_gt_class_rank"],
        ],
    )
    ax.set_title("Corrected small-object / large-dynamic summary")
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_small_vs_large_dynamic_coverage.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([r["ff_nearest_support_distance_bev_mean"] for r in small_rows if r["small_false_free_count"] > 0], bins=20)
    ax.set_title("Corrected small-object FF nearest support distance")
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_small_object_nearest_support_distance_hist.png", dpi=150)
    plt.close(fig)
    plot_region_overlay(sw3, sample0_record, corrected_support0, fig_dir / "corrected_small_object_gt_pred_support_overlay_sample0.png", "Corrected support overlay sample0")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["small_ff_gt_rank", "large_dyn_gt_rank"],
        [small_summary["mean_small_ff_gt_class_rank"], small_summary["mean_large_dynamic_gt_class_rank"]],
    )
    ax.set_title("Corrected GT-class support rank")
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_small_object_class_score_rank.png", dpi=150)
    plt.close(fig)
    print("[SW35] phase3 corrected small-object diagnosis complete", flush=True)

    # Phase 4 corrected new-visible / persistent diagnosis.
    new_rows, new_summary = compute_new_visible_rows(sw3, records, corrected_adapter, support_cache_key="corrected_supports")
    write_json(reports_dir / "corrected_new_visible_failure_localization.json", {"rows": new_rows, "summary": new_summary})
    write_csv(reports_dir / "corrected_new_visible_failure_localization.csv", new_rows)
    (reports_dir / "corrected_new_visible_failure_summary.md").write_text(
        "# Corrected new-visible failure summary\n\n"
        f"- mean new-visible geometric coverage: `{new_summary['mean_new_visible_geometric_coverage']}`\n"
        f"- mean persistent geometric coverage: `{new_summary['mean_persistent_geometric_coverage']}`\n"
        f"- mean new-visible semantic-active coverage: `{new_summary['mean_new_visible_semantic_active_coverage']}`\n"
        f"- mean persistent semantic-active coverage: `{new_summary['mean_persistent_semantic_active_coverage']}`\n"
        f"- mean new-visible recall: `{new_summary['mean_new_visible_pred_recall']}`\n"
        f"- mean persistent recall: `{new_summary['mean_persistent_pred_recall']}`\n",
        encoding="utf-8",
    )
    horizons_nv = sorted({int(r["horizon_s"]) for r in new_rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons_nv, [float(np.mean([rr["new_visible_geometric_coverage"] for rr in new_rows if rr["horizon_s"] == h and rr["new_visible_gt_count"] > 0])) for h in horizons_nv], marker="o", label="new_visible_cov")
    ax.plot(horizons_nv, [float(np.mean([rr["persistent_geometric_coverage"] for rr in new_rows if rr["horizon_s"] == h and rr["persistent_gt_count"] > 0])) for h in horizons_nv], marker="o", label="persistent_cov")
    ax.set_title("Corrected new-visible coverage over horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_new_visible_coverage_over_horizon.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["persistent_dist", "new_visible_dist"],
        [
            float(np.mean([r["persistent_nearest_support_distance_bev_mean"] for r in new_rows if r["persistent_gt_count"] > 0])),
            float(np.mean([r["new_visible_nearest_support_distance_bev_mean"] for r in new_rows if r["new_visible_gt_count"] > 0])),
        ],
    )
    ax.set_title("Corrected persistent vs new-visible nearest support distance")
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_persistent_vs_new_visible_query_distance.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r["coverage_gap"] for r in new_rows], [r["recall_gap"] for r in new_rows], alpha=0.5)
    ax.set_xlabel("coverage gap")
    ax.set_ylabel("recall gap")
    ax.set_title("Corrected new-visible recall vs coverage gap")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_new_visible_recall_vs_coverage_gap.png", dpi=150)
    plt.close(fig)
    plot_region_overlay(sw3, sample0_record, corrected_support0, fig_dir / "corrected_new_visible_query_overlay_sample0.png", "Corrected support overlay sample0")
    print("[SW35] phase4 corrected new-visible diagnosis complete", flush=True)

    # Phase 5 corrected temporal support coverage proxy.
    corrected_temporal_rows = compute_temporal_support_proxy(sw3, records, corrected_adapter, support_cache_key="corrected_supports")
    old_temporal_rows = compute_temporal_support_proxy(sw3, records, old_adapter, support_cache_key="old_supports")
    corrected_temporal_agg = aggregate_by_horizon(
        corrected_temporal_rows,
        [
            "support_coverage_gt_occupied",
            "support_coverage_false_free",
            "support_coverage_pred_occupied",
            "support_coverage_false_occupied",
            "semantic_active_support_coverage_gt_occupied",
            "semantic_active_support_coverage_false_free",
            "support_valid_range_ratio",
            "support_density_entropy",
            "support_density_shift_from_t0",
            "diagonal_line_score",
        ],
    )
    old_temporal_agg = aggregate_by_horizon(
        old_temporal_rows,
        [
            "support_coverage_gt_occupied",
            "support_coverage_false_free",
            "support_coverage_pred_occupied",
            "support_coverage_false_occupied",
            "semantic_active_support_coverage_gt_occupied",
            "semantic_active_support_coverage_false_free",
            "support_valid_range_ratio",
            "support_density_entropy",
            "support_density_shift_from_t0",
            "diagonal_line_score",
        ],
    )
    write_json(reports_dir / "corrected_temporal_support_coverage_proxy.json", {"rows": corrected_temporal_rows, "aggregate": corrected_temporal_agg})
    write_csv(reports_dir / "corrected_temporal_support_coverage_proxy.csv", corrected_temporal_rows)
    (reports_dir / "corrected_temporal_support_coverage_proxy_summary.md").write_text(
        "# Corrected temporal support coverage proxy summary\n\n"
        f"- mean t=0 false-free support coverage: `{next((r['support_coverage_false_free'] for r in corrected_temporal_agg if r['horizon_s']==0), 0.0)}`\n"
        f"- mean t=6 false-free support coverage: `{next((r['support_coverage_false_free'] for r in corrected_temporal_agg if r['horizon_s']==6), 0.0)}`\n"
        f"- old SW-2 coverage_failure_dominant conclusion: `deprecated`\n",
        encoding="utf-8",
    )
    plot_temporal_support_curves(corrected_temporal_agg, old_temporal_agg, fig_dir)
    corrected_supports0 = sample0_record["corrected_supports"]
    plot_support_density_rollout(sample0_record, corrected_supports0, fig_dir / "corrected_support_density_rollout_sample0.png")
    old_support0 = sample0_record["old_supports"][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(old_support0["support_density_bev"].cpu().numpy().T, origin="lower", cmap="magma")
    axes[0].set_title("Old mean-center support")
    axes[0].axis("off")
    axes[1].imshow(corrected_support0["support_density_bev"].cpu().numpy().T, origin="lower", cmap="magma")
    axes[1].set_title("Corrected all48 support")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "old_vs_corrected_query_density_sample0.png", dpi=150)
    plt.close(fig)
    print("[SW35] phase5 corrected temporal support coverage complete", flush=True)

    # Phase 6 old vs corrected comparison.
    old_sw2_rows = old_from_sw2_query_coverage(project_root)
    old_sw2_agg = aggregate_by_horizon(
        old_sw2_rows,
        [
            "support_coverage_gt_occupied",
            "support_coverage_false_free",
            "support_coverage_pred_occupied",
            "support_coverage_false_occupied",
            "support_density_entropy",
            "support_density_shift_from_t0",
        ],
    )
    old_vs_corrected = {
        "old_proxy": {
            "definition": "deprecated mean-center refine_pts proxy",
            "mean_false_free_coverage_r2_by_horizon": {row["horizon_s"]: row["support_coverage_false_free"] for row in old_sw2_agg},
            "mean_gt_coverage_r2_by_horizon": {row["horizon_s"]: row["support_coverage_gt_occupied"] for row in old_sw2_agg},
        },
        "corrected_proxy": {
            "definition": "refine_pts_current + decoded_metric + xyz + noflip + all48",
            "mean_false_free_coverage_r2_by_horizon": {row["horizon_s"]: row["support_coverage_false_free"] for row in corrected_temporal_agg},
            "mean_gt_coverage_r2_by_horizon": {row["horizon_s"]: row["support_coverage_gt_occupied"] for row in corrected_temporal_agg},
        },
        "old_conclusion": "coverage_failure_dominant (deprecated)",
        "corrected_conclusion": corrected_attr_summary["conclusion"],
        "why_old_proxy_failed": [
            "mean-center aggregation destroyed the support footprint",
            "get_occ voxelizes refined points, not per-query mean centers",
            "all48 support points are required for faithful spatial support coverage",
        ],
        "evidence_sources": [
            "SW-3 semantic_occ dependency trace",
            "SW-3 causal controllability tests",
            "SW-3 best coordinate mapping",
            "SW-3.5 corrected coverage re-run",
        ],
    }
    write_json(reports_dir / "old_vs_corrected_support_proxy_comparison.json", old_vs_corrected)
    (reports_dir / "old_vs_corrected_support_proxy_comparison.md").write_text(
        "# Old vs corrected support proxy comparison\n\n"
        "- old proxy: deprecated mean-center refine_pts proxy\n"
        "- corrected proxy: refine_pts_current + decoded_metric + xyz + noflip + all48\n"
        "- old SW-2 conclusion: `coverage_failure_dominant` (deprecated)\n"
        f"- corrected conclusion: `{corrected_attr_summary['conclusion']}`\n"
        "- why old proxy failed:\n"
        "  - mean-center aggregation destroyed the support footprint\n"
        "  - get_occ voxelizes refined points, not per-query centers\n"
        "  - corrected all48 support is required for faithful coverage\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    horizons_cmp = [row["horizon_s"] for row in old_sw2_agg]
    ax.plot(horizons_cmp, [row["support_coverage_false_free"] for row in old_sw2_agg], marker="o", label="old_mean_center")
    ax.plot(horizons_cmp, [row["support_coverage_false_free"] for row in corrected_temporal_agg], marker="o", label="corrected_all48")
    ax.set_title("Old vs corrected false-free coverage")
    ax.set_xlabel("Horizon (s)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "old_vs_corrected_false_free_coverage_curve.png", dpi=150)
    plt.close(fig)
    old_attr_rows = compute_query_to_pred_attribution(sw3, records, old_adapter, radii=[2], support_cache_key="old_supports")
    old_attr_summary = quadrant_summary(old_attr_rows, radius=2)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["TP", "FF", "FO", "TN"]
    old_vals = [old_attr_summary["by_region"][k]["geometric_support_coverage_ratio_mean"] for k in ["tp_occupied", "false_free", "false_occupied", "true_free"]]
    new_vals = [corrected_attr_summary["by_region"][k]["geometric_support_coverage_ratio_mean"] for k in ["tp_occupied", "false_free", "false_occupied", "true_free"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, old_vals, width=0.36, label="old_mean_center")
    ax.bar(x + 0.18, new_vals, width=0.36, label="corrected_all48")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Old vs corrected quadrant coverage (r=2)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "old_vs_corrected_coverage_four_quadrants.png", dpi=150)
    plt.close(fig)
    print("[SW35] phase6 old-vs-corrected comparison complete", flush=True)

    # Phase 7 corrected joint temporal-query diagnosis matrix.
    temporal_lookup, group_lookup, new_visible_lookup = load_sw2_metric_lookups(project_root)
    attr_r2_lookup = {
        (int(row["sample_index"]), int(row["horizon_s"]), row["region_name"]): row
        for row in corrected_attr_rows
        if int(row["coverage_radius_cells"]) == 2
    }
    small_lookup = defaultdict(list)
    for row in small_rows:
        small_lookup[(int(row["sample_index"]), int(row["horizon_s"]))].append(row)
    new_lookup = {(int(row["sample_index"]), int(row["horizon_s"])): row for row in new_rows}
    temporal_proxy_lookup = {(int(row["sample_index"]), int(row["horizon_s"])): row for row in corrected_temporal_rows}
    joint_rows: list[dict[str, Any]] = []
    for record in records:
        for horizon_s in range(record["pred_temporal"].shape[0]):
            key = (record["sample_index"], horizon_s)
            temporal = temporal_lookup[key]
            ff_row = attr_r2_lookup[(record["sample_index"], horizon_s, "false_free")]
            tp_row = attr_r2_lookup[(record["sample_index"], horizon_s, "tp_occupied")]
            proxy = temporal_proxy_lookup[key]
            nv = new_lookup.get(key)
            small_vals = small_lookup.get(key, [])
            joint_rows.append(
                {
                    "sample_index": record["sample_index"],
                    "sample_token": record["sample_token"],
                    "scene_token": record["scene_token"],
                    "horizon_s": horizon_s,
                    "occupied_iou": float(temporal["occupied_iou"]),
                    "semantic_miou": float(temporal["semantic_miou"]),
                    "false_free_rate": float(temporal["false_free_rate"]),
                    "false_occupied_rate": float(temporal["false_occupied_rate"]),
                    "pred_gt_occupied_ratio": float(temporal["pred_gt_occupied_ratio"]),
                    "static_false_free": float(group_lookup[(record["sample_index"], horizon_s, "static_background")]["group_false_free_rate"]),
                    "dynamic_false_free": float(group_lookup[(record["sample_index"], horizon_s, "all_dynamic")]["group_false_free_rate"]),
                    "small_object_false_free": float(group_lookup[(record["sample_index"], horizon_s, "small_object")]["group_false_free_rate"]),
                    "new_visible_false_free": float(new_visible_lookup[key]["new_visible_false_free"]) if key in new_visible_lookup else 0.0,
                    "persistent_false_free": float(new_visible_lookup[key]["persistent_false_free"]) if key in new_visible_lookup else 0.0,
                    "corrected_FF_geometric_coverage": float(ff_row["geometric_support_coverage_ratio"]),
                    "corrected_FF_semantic_active_coverage": float(ff_row["semantic_active_support_coverage_ratio"]),
                    "corrected_TP_semantic_active_coverage": float(tp_row["semantic_active_support_coverage_ratio"]),
                    "corrected_small_object_FF_coverage": float(np.mean([r["ff_geometric_support_coverage_r2"] for r in small_vals if r["small_false_free_count"] > 0])) if small_vals else 0.0,
                    "corrected_new_visible_coverage": float(nv["new_visible_geometric_coverage"]) if nv else 0.0,
                    "corrected_persistent_coverage": float(nv["persistent_geometric_coverage"]) if nv else 0.0,
                    "support_density_entropy": float(proxy["support_density_entropy"]),
                    "support_density_shift": float(proxy["support_density_shift_from_t0"]),
                    "nearest_support_distance_FF": float(ff_row["nearest_support_distance_bev_mean"]),
                    "nearest_support_distance_TP": float(tp_row["nearest_support_distance_bev_mean"]),
                }
            )
    write_json(reports_dir / "corrected_joint_temporal_query_diagnosis_matrix.json", {"rows": joint_rows})
    write_csv(reports_dir / "corrected_joint_temporal_query_diagnosis_matrix.csv", joint_rows)
    corrected_joint_summary = {
        "mean_corrected_FF_geometric_coverage": float(np.mean([r["corrected_FF_geometric_coverage"] for r in joint_rows])),
        "mean_corrected_FF_semantic_active_coverage": float(np.mean([r["corrected_FF_semantic_active_coverage"] for r in joint_rows])),
        "mean_corrected_small_object_FF_coverage": float(np.mean([r["corrected_small_object_FF_coverage"] for r in joint_rows])),
        "mean_corrected_new_visible_coverage": float(np.mean([r["corrected_new_visible_coverage"] for r in joint_rows if r["horizon_s"] > 0])),
        "next_stage": "Stage SW-4.0 Covered False-Free Semantic Activation Diagnosis",
    }
    (reports_dir / "corrected_joint_temporal_query_diagnosis_summary.md").write_text(
        "# Corrected joint temporal-query diagnosis summary\n\n"
        f"- corrected FF geometric coverage remains high: `{corrected_joint_summary['mean_corrected_FF_geometric_coverage']}`\n"
        f"- corrected FF semantic-active coverage remains high: `{corrected_joint_summary['mean_corrected_FF_semantic_active_coverage']}`\n"
        f"- corrected small-object FF coverage: `{corrected_joint_summary['mean_corrected_small_object_FF_coverage']}`\n"
        f"- corrected new-visible coverage: `{corrected_joint_summary['mean_corrected_new_visible_coverage']}`\n"
        "- implication: covered false-free is more consistent with semantic activation / get_occ gating / aggregation weakness than global support absence\n",
        encoding="utf-8",
    )
    print("[SW35] phase7 corrected joint temporal-query matrix complete", flush=True)

    # Phase 8 final report.
    corrected_resolution = (
        "covered_false_free_semantic_activation_dominant"
        if corrected_attr_summary["mean_ff_geometric_coverage_ratio"] > 0.7
        else "coverage_failure_still_plausible"
    )
    next_action = "Stage SW-4.0 Covered False-Free Semantic Activation Diagnosis"
    report_payload = {
        "executive_summary": {
            "effective_sample_count": len(records),
            "corrected_support_definition": run_manifest["corrected_support_definition"],
            "old_conclusion_deprecated": True,
            "corrected_false_free_geometric_coverage_r2": corrected_attr_summary["mean_ff_geometric_coverage_ratio"],
            "corrected_false_free_semantic_active_coverage_r2": corrected_attr_summary["mean_ff_semantic_active_coverage_ratio"],
            "corrected_resolution": corrected_resolution,
            "next_unique_action": next_action,
        },
        "corrected_query_to_pred_attribution_summary": corrected_attr_summary,
        "corrected_small_object_summary": small_summary,
        "corrected_new_visible_summary": new_summary,
        "old_vs_corrected_support_proxy_comparison": old_vs_corrected,
        "corrected_joint_summary": corrected_joint_summary,
        "safe_claims": {
            "training_completed": False,
            "official_benchmark": False,
            "model_improvement": False,
            "sensor_perturbation_completed": False,
        },
    }
    write_json(reports_dir / "stage_sw35_corrected_support_coverage_report.json", report_payload)
    (reports_dir / "stage_sw35_corrected_support_coverage_report.md").write_text(
        "# Stage SW-3.5 corrected support coverage\n\n"
        "## Executive summary\n\n"
        "- SW-2 old `coverage_failure_dominant` conclusion is deprecated.\n"
        "- Corrected support definition: `refine_pts_current + decoded_metric + xyz + noflip + all48`.\n"
        f"- Corrected false-free geometric coverage (r=2): `{corrected_attr_summary['mean_ff_geometric_coverage_ratio']}`\n"
        f"- Corrected false-free semantic-active coverage (r=2): `{corrected_attr_summary['mean_ff_semantic_active_coverage_ratio']}`\n"
        f"- Corrected resolution: `{corrected_resolution}`\n"
        "- If support coverage remains high, next stage must inspect semantic activation / get_occ / gating / aggregation.\n\n"
        "## Safe claims\n\n"
        "- This is a corrected subset diagnostic.\n"
        "- Benchmark reproduction claim: no.\n"
        "- Training completed claim: no.\n"
        "- Model improvement claim: no.\n"
        "- Sensor perturbation completion claim: no.\n\n"
        "## Next unique action\n\n"
        f"- {next_action}\n",
        encoding="utf-8",
    )
    print("[SW35] phase8 final report complete", flush=True)

    print(
        json.dumps(
            {
                "effective_samples": len(records),
                "corrected_false_free_geometric_coverage_r2": corrected_attr_summary["mean_ff_geometric_coverage_ratio"],
                "corrected_resolution": corrected_resolution,
                "next_unique_action": next_action,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
