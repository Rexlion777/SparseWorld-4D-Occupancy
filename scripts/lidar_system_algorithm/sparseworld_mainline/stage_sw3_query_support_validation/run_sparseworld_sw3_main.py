"""Stage SW-3: Query support validation and allocation failure localization.

Core purpose:
1. validate whether refine_pts is a true semantic occupancy support tensor
2. separate support proxy bugs from real allocation failure
3. distinguish geometric coverage failure from semantic activation failure

Safe-claim boundary:
- subset diagnostic only
- no training
- no official benchmark
- no sensor perturbation
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


EMPTY_IDX = 17
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
VOXEL_SIZE = [0.4, 0.4, 0.4]
GRID_SIZE = [200, 200, 16]
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
CLASS_GROUPS = {
    "static_background": [
        LABEL_TO_ID["driveable_surface"],
        LABEL_TO_ID["other_flat"],
        LABEL_TO_ID["sidewalk"],
        LABEL_TO_ID["terrain"],
        LABEL_TO_ID["manmade"],
        LABEL_TO_ID["vegetation"],
    ],
    "dynamic_vehicle": [
        LABEL_TO_ID["car"],
        LABEL_TO_ID["truck"],
        LABEL_TO_ID["bus"],
        LABEL_TO_ID["trailer"],
        LABEL_TO_ID["construction_vehicle"],
    ],
    "dynamic_vulnerable": [
        LABEL_TO_ID["pedestrian"],
        LABEL_TO_ID["bicycle"],
        LABEL_TO_ID["motorcycle"],
    ],
    "small_object": [
        LABEL_TO_ID["pedestrian"],
        LABEL_TO_ID["bicycle"],
        LABEL_TO_ID["motorcycle"],
        LABEL_TO_ID["traffic_cone"],
        LABEL_TO_ID["barrier"],
    ],
}
CLASS_GROUPS["all_dynamic"] = sorted(set(CLASS_GROUPS["dynamic_vehicle"] + CLASS_GROUPS["dynamic_vulnerable"]))
CLASS_GROUPS["all_static"] = sorted(set(CLASS_GROUPS["static_background"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage SW-3 SparseWorld query support validation")
    parser.add_argument("--repo-root", default="/mnt/d/ComputerVision/cv_lidar_transition/external/SparseWorld")
    parser.add_argument("--config", default="/mnt/d/ComputerVision/cv_lidar_transition/external/SparseWorld/configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py")
    parser.add_argument("--checkpoint", default="/mnt/d/ComputerVision/cv_lidar_transition/external/SparseWorld/ckpts/epoch_56.pth")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--horizons", default="0,1,2,3,4,5,6")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-hooks", action="store_true")
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


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: summarize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 12:
            return {"type": "list", "length": len(obj)}
        return [summarize(v) for v in obj]
    if isinstance(obj, tuple):
        return [summarize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        cpu = obj.detach().float().cpu()
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "mean": float(cpu.mean().item()) if cpu.numel() else 0.0,
            "std": float(cpu.std().item()) if cpu.numel() > 1 else 0.0,
            "min": float(cpu.min().item()) if cpu.numel() else 0.0,
            "max": float(cpu.max().item()) if cpu.numel() else 0.0,
        }
    if isinstance(obj, np.ndarray):
        arr = obj.astype(np.float32, copy=False) if obj.size else obj
        return {
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "mean": float(arr.mean()) if obj.size else 0.0,
            "std": float(arr.std()) if obj.size else 0.0,
            "min": float(arr.min()) if obj.size else 0.0,
            "max": float(arr.max()) if obj.size else 0.0,
        }
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(type(obj).__name__)


def to_cpu_artifact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_cpu_artifact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu_artifact(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_cpu_artifact(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj
    return obj


def tensor_to_metric(points: torch.Tensor, coord_mode: str) -> torch.Tensor:
    out = points.clone().float()
    if coord_mode == "decoded_metric":
        out[..., 0] = out[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
        out[..., 1] = out[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
        out[..., 2] = out[..., 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
        return out
    if coord_mode == "normalized01_to_metric":
        out[..., 0] = out[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
        out[..., 1] = out[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
        out[..., 2] = out[..., 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
        return out
    if coord_mode == "normalizedm11_to_metric":
        out = (out + 1.0) / 2.0
        out[..., 0] = out[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
        out[..., 1] = out[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
        out[..., 2] = out[..., 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
        return out
    if coord_mode == "raw_grid_to_metric":
        out[..., 0] = (out[..., 0] + 0.5) * VOXEL_SIZE[0] + PC_RANGE[0]
        out[..., 1] = (out[..., 1] + 0.5) * VOXEL_SIZE[1] + PC_RANGE[1]
        out[..., 2] = (out[..., 2] + 0.5) * VOXEL_SIZE[2] + PC_RANGE[2]
        return out
    raise ValueError(coord_mode)


def metric_to_grid(points_metric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.floor(
        torch.stack(
            [
                (points_metric[..., 0] - PC_RANGE[0]) / VOXEL_SIZE[0],
                (points_metric[..., 1] - PC_RANGE[1]) / VOXEL_SIZE[1],
                (points_metric[..., 2] - PC_RANGE[2]) / VOXEL_SIZE[2],
            ],
            dim=-1,
        )
    ).long()
    valid = (
        (idx[..., 0] >= 0)
        & (idx[..., 0] < GRID_SIZE[0])
        & (idx[..., 1] >= 0)
        & (idx[..., 1] < GRID_SIZE[1])
        & (idx[..., 2] >= 0)
        & (idx[..., 2] < GRID_SIZE[2])
    )
    return idx, valid


def permute_and_flip(points_metric: torch.Tensor, order: tuple[int, int, int], flip_x: bool, flip_y: bool) -> torch.Tensor:
    out = points_metric[..., list(order)]
    if flip_x:
        out[..., 0] = PC_RANGE[3] - (out[..., 0] - PC_RANGE[0])
    if flip_y:
        out[..., 1] = PC_RANGE[4] - (out[..., 1] - PC_RANGE[1])
    return out


def point_slice(points: torch.Tensor, cls_scores: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "all48":
        return points, cls_scores
    if mode == "mean48":
        return points.mean(dim=2, keepdim=True), cls_scores.max(dim=2, keepdim=True).values
    if mode == "first_refine":
        return points[:, :, :1, :], cls_scores[:, :, :1, :]
    if mode == "last_refine":
        return points[:, :, -1:, :], cls_scores[:, :, -1:, :]
    if mode == "max_conf_point":
        max_conf = cls_scores.max(dim=-1).values
        best_idx = max_conf.argmax(dim=2, keepdim=True)
        idx_pts = best_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        idx_cls = best_idx.unsqueeze(-1).expand(-1, -1, -1, cls_scores.shape[-1])
        return torch.gather(points, 2, idx_pts), torch.gather(cls_scores, 2, idx_cls)
    raise ValueError(mode)


def class_filter_scores(cls_scores: torch.Tensor, filter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if filter_name == "all":
        ids = torch.arange(cls_scores.shape[-1], device=cls_scores.device)
        return cls_scores, ids
    if filter_name == "dynamic":
        ids = torch.as_tensor(CLASS_GROUPS["all_dynamic"], device=cls_scores.device)
    elif filter_name == "small_object":
        ids = torch.as_tensor(CLASS_GROUPS["small_object"], device=cls_scores.device)
    elif filter_name == "static":
        ids = torch.as_tensor(CLASS_GROUPS["all_static"], device=cls_scores.device)
    else:
        raise ValueError(filter_name)
    return cls_scores.index_select(-1, ids), ids


def scatter_grid(indices: torch.Tensor, scores: torch.Tensor, num_classes: int) -> dict[str, Any]:
    flat = indices[:, 0] * (GRID_SIZE[1] * GRID_SIZE[2]) + indices[:, 1] * GRID_SIZE[2] + indices[:, 2]
    unique, inv = torch.unique(flat, return_inverse=True)
    counts = torch.bincount(inv, minlength=unique.numel()).long()
    max_scores = torch_scatter.scatter_max(scores, inv, dim=0)[0]
    max_foreground, argmax_class = max_scores.max(dim=-1)
    support_mask = torch.zeros(GRID_SIZE, dtype=torch.bool)
    support_count = torch.zeros(GRID_SIZE, dtype=torch.int32)
    support_conf = torch.zeros(GRID_SIZE, dtype=torch.float32)
    support_class = torch.full(GRID_SIZE, -1, dtype=torch.int16)
    coords = torch.stack(
        [
            unique // (GRID_SIZE[1] * GRID_SIZE[2]),
            (unique % (GRID_SIZE[1] * GRID_SIZE[2])) // GRID_SIZE[2],
            unique % GRID_SIZE[2],
        ],
        dim=-1,
    ).long()
    support_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    support_count[coords[:, 0], coords[:, 1], coords[:, 2]] = counts.cpu().to(torch.int32)
    support_conf[coords[:, 0], coords[:, 1], coords[:, 2]] = max_foreground.cpu()
    support_class[coords[:, 0], coords[:, 1], coords[:, 2]] = argmax_class.cpu().to(torch.int16)
    return {
        "support_mask": support_mask,
        "support_count": support_count,
        "support_conf": support_conf,
        "support_class": support_class,
        "support_logits_sparse": max_scores.cpu(),
        "support_coords": coords.cpu(),
        "argmax_class_sparse": argmax_class.cpu(),
    }


def build_support_from_tensor(
    points_tensor: torch.Tensor,
    cls_scores_tensor: torch.Tensor,
    score_thr: list[float],
    point_mode: str,
    coord_mode: str,
    order: tuple[int, int, int],
    flip_x: bool,
    flip_y: bool,
    class_filter_name: str,
    use_score_gate: bool,
) -> dict[str, Any]:
    points_sel, cls_sel = point_slice(points_tensor, cls_scores_tensor, point_mode)
    points_sel = points_sel[0]
    cls_sel = cls_sel[0].sigmoid()
    filtered_scores, selected_ids = class_filter_scores(cls_sel, class_filter_name)
    score_thr_tensor = torch.as_tensor(score_thr, dtype=torch.float32, device=cls_sel.device)
    score_thr_filtered = score_thr_tensor.index_select(0, selected_ids.long())

    points_metric = tensor_to_metric(points_sel, coord_mode)
    points_metric = permute_and_flip(points_metric, order, flip_x, flip_y)
    points_metric = points_metric.reshape(-1, 3)
    filtered_scores = filtered_scores.reshape(-1, filtered_scores.shape[-1])
    full_scores = cls_sel.reshape(-1, cls_sel.shape[-1])

    geometric_grid_idx, valid_mask = metric_to_grid(points_metric)
    geometric_indices = geometric_grid_idx[valid_mask]
    geometric_points_metric = points_metric[valid_mask]
    geometric_filtered_scores = filtered_scores[valid_mask]
    geometric_full_scores = full_scores[valid_mask]

    if geometric_indices.numel() == 0:
        empty_mask = torch.zeros(GRID_SIZE, dtype=torch.bool)
        empty_count = torch.zeros(GRID_SIZE, dtype=torch.int32)
        empty_conf = torch.zeros(GRID_SIZE, dtype=torch.float32)
        empty_class = torch.full(GRID_SIZE, -1, dtype=torch.int16)
        return {
            "points_metric": geometric_points_metric.cpu(),
            "grid_indices": geometric_indices.cpu(),
            "geometric_mask": empty_mask,
            "semantic_active_mask": empty_mask.clone(),
            "semantic_active_count": empty_count,
            "semantic_active_conf": empty_conf,
            "semantic_active_class": empty_class,
            "semantic_active_logits_sparse": torch.empty((0, cls_sel.shape[-1])),
            "valid_ratio": 0.0,
            "diagonal_line_score": 1.0,
            "query_density_entropy": 0.0,
            "query_com_xy": [math.nan, math.nan],
        }

    # Geometric support: any in-range point irrespective of score gate.
    geom_flat = geometric_indices[:, 0] * (GRID_SIZE[1] * GRID_SIZE[2]) + geometric_indices[:, 1] * GRID_SIZE[2] + geometric_indices[:, 2]
    geom_unique = torch.unique(geom_flat)
    geom_coords = torch.stack(
        [
            geom_unique // (GRID_SIZE[1] * GRID_SIZE[2]),
            (geom_unique % (GRID_SIZE[1] * GRID_SIZE[2])) // GRID_SIZE[2],
            geom_unique % GRID_SIZE[2],
        ],
        dim=-1,
    ).long()
    geometric_mask = torch.zeros(GRID_SIZE, dtype=torch.bool)
    geometric_mask[geom_coords[:, 0], geom_coords[:, 1], geom_coords[:, 2]] = True

    if use_score_gate:
        # get_occ-compatible point filtering
        pts_for_ctr = points_metric.reshape(points_sel.shape[0], points_sel.shape[1], 3)
        centers = pts_for_ctr.mean(dim=1, keepdim=True)
        ctr_dist = torch.norm(pts_for_ctr - centers, dim=-1)
        if ctr_dist.numel() == points_metric.shape[0]:
            dist_mask = ctr_dist.reshape(-1)[valid_mask]
        else:
            dist_mask = torch.ones(geometric_points_metric.shape[0], dtype=torch.float32, device=geometric_points_metric.device)
        point_max_score, point_arg_cls = geometric_filtered_scores.max(dim=-1)
        point_thr = score_thr_filtered[point_arg_cls]
        semantic_point_mask = (dist_mask < 3.0) & (point_max_score > point_thr)
    else:
        semantic_point_mask = torch.ones(geometric_points_metric.shape[0], dtype=torch.bool, device=geometric_points_metric.device)

    semantic_points_metric = geometric_points_metric[semantic_point_mask]
    semantic_indices = geometric_indices[semantic_point_mask]
    semantic_full_scores = geometric_full_scores[semantic_point_mask]

    if semantic_indices.numel() == 0:
        semantic_active_mask = torch.zeros(GRID_SIZE, dtype=torch.bool)
        semantic_active_count = torch.zeros(GRID_SIZE, dtype=torch.int32)
        semantic_active_conf = torch.zeros(GRID_SIZE, dtype=torch.float32)
        semantic_active_class = torch.full(GRID_SIZE, -1, dtype=torch.int16)
        sparse_logits = torch.empty((0, cls_sel.shape[-1]))
    else:
        grid_data = scatter_grid(semantic_indices, semantic_full_scores, cls_sel.shape[-1])
        semantic_active_mask = grid_data["support_mask"]
        semantic_active_count = grid_data["support_count"]
        semantic_active_conf = grid_data["support_conf"]
        semantic_active_class = grid_data["support_class"]
        sparse_logits = grid_data["support_logits_sparse"]

    # BEV density diagnostics
    bev_density = geometric_mask.any(dim=-1).float().numpy()
    coords2d = np.argwhere(bev_density > 0)
    if coords2d.shape[0] >= 2:
        cov = np.cov(coords2d.T)
        eigvals = np.linalg.eigvalsh(cov)
        major = float(max(eigvals))
        minor = float(min(eigvals))
        diagonal_line_score = 1.0 - (minor / major if major > 1e-6 else 0.0)
    else:
        diagonal_line_score = 1.0
    density = np.zeros((GRID_SIZE[0], GRID_SIZE[1]), dtype=np.float32)
    for x, y, _ in geometric_indices.cpu().numpy():
        density[x, y] += 1.0
    prob = density / density.sum() if density.sum() > 0 else density
    nz = prob[prob > 0]
    entropy = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
    if density.sum() > 0:
        xs = np.arange(GRID_SIZE[0], dtype=np.float32)
        ys = np.arange(GRID_SIZE[1], dtype=np.float32)
        com_x = float((density.sum(axis=1) * xs).sum() / density.sum())
        com_y = float((density.sum(axis=0) * ys).sum() / density.sum())
    else:
        com_x = math.nan
        com_y = math.nan

    return {
        "points_metric": geometric_points_metric.cpu(),
        "grid_indices": geometric_indices.cpu(),
        "geometric_mask": geometric_mask,
        "semantic_active_mask": semantic_active_mask,
        "semantic_active_count": semantic_active_count,
        "semantic_active_conf": semantic_active_conf,
        "semantic_active_class": semantic_active_class,
        "semantic_active_logits_sparse": sparse_logits,
        "valid_ratio": float(valid_mask.float().mean().item()) if valid_mask.numel() else 0.0,
        "diagonal_line_score": diagonal_line_score,
        "query_density_entropy": entropy,
        "query_com_xy": [com_x, com_y],
    }


def overlap_ratio(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = torch.logical_and(mask_a, mask_b).sum().item()
    den = mask_b.sum().item()
    return float(inter / den) if den else 0.0


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def bev_map(mask3d: torch.Tensor) -> np.ndarray:
    return mask3d.any(dim=-1).cpu().numpy().astype(np.uint8)


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
    out = F.max_pool3d(t, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return out[0, 0] > 0


def distance_transform_bev(support_mask3d: torch.Tensor) -> np.ndarray:
    support_bev = bev_map(support_mask3d)
    # distance to nearest support cell in BEV.
    inv = (support_bev == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist


def region_stats_from_support(
    pred: torch.Tensor,
    gt: torch.Tensor,
    support: dict[str, Any],
    radius: int = 2,
) -> dict[str, Any]:
    regions = classify_regions(gt, pred)
    geo_dil = dilate3d(support["geometric_mask"], radius)
    sem_dil = dilate3d(support["semantic_active_mask"], radius)
    dist_bev = distance_transform_bev(support["geometric_mask"])
    support_conf = support["semantic_active_conf"].float()
    support_density = support["semantic_active_count"].float()
    rows = {}
    for name, region in regions.items():
        region_count = int(region.sum().item())
        if region_count == 0:
            rows[name] = {
                "region_voxel_count": 0,
                "geometric_coverage_ratio": 0.0,
                "semantic_active_coverage_ratio": 0.0,
                "region_support_conf_mean": 0.0,
                "region_support_density_mean": 0.0,
                "region_nearest_query_distance_bev_mean": 0.0,
            }
            continue
        region_bev = bev_map(region)
        rows[name] = {
            "region_voxel_count": region_count,
            "geometric_coverage_ratio": float((geo_dil & region).sum().item() / region_count),
            "semantic_active_coverage_ratio": float((sem_dil & region).sum().item() / region_count),
            "region_support_conf_mean": float(support_conf[region].mean().item()) if support_conf[region].numel() else 0.0,
            "region_support_density_mean": float(support_density[region].mean().item()) if support_density[region].numel() else 0.0,
            "region_nearest_query_distance_bev_mean": float(dist_bev[region_bev > 0].mean()) if region_bev.sum() else 0.0,
        }
    return rows


def find_line(file_path: Path, pattern: str) -> list[dict[str, Any]]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines, start=1):
        if pattern in line:
            out.append({"file_path": str(file_path), "line_no": i, "pattern": pattern, "line_text": line.strip()})
    return out


def semantic_occ_dependency_trace(repo_root: Path, out_dir: Path) -> dict[str, Any]:
    files = {
        "sparseworld_4d_traj": repo_root / "mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py",
        "opus_head": repo_root / "mmdet3d/models/sparsedetectors/opus_head.py",
        "opus_transformer": repo_root / "mmdet3d/models/sparsedetectors/opus_transformer.py",
        "bbox_utils": repo_root / "mmdet3d/models/sparsedetectors/bbox/utils.py",
    }
    patterns = [
        "semantic_occ_0s",
        "semantic_occ_1s",
        "semantic_occ_6s",
        "pred_dict",
        "get_occ",
        "all_cls_scores",
        "all_refine_pts",
        "forecast_points_list",
        "forecast_semantics_list",
        "cls_score",
        "refine_pts",
        "decode_points",
        "encode_points",
    ]
    rows = []
    for path in files.values():
        for pattern in patterns:
            rows.extend(find_line(path, pattern))
    dependency_graph = [
        "img -> forward_backbone(img,img_metas,**kwargs)",
        "forward_backbone -> outs['all_cls_scores'][-1], outs['all_refine_pts'][-1], query_feat",
        "simple_test current frame -> pred_dict{cls_scores=current all_cls_scores slice, refine_pts=current all_refine_pts slice}",
        "pred_dict -> pts_bbox_head.get_occ(pred_dict) -> semantic_occ_0s",
        "forward_backbone future rollout -> forecast_semantics_list[h], forecast_points_list[h]",
        "future input_dict -> pts_bbox_head.get_occ(input_dict) -> semantic_occ_(h+1)s",
        "pred_traj is produced by traj_head and does not directly feed semantic_occ decode",
        "OPUSHead.get_occ performs decode_points(refine_pts) + score gating + voxelization + optional padding -> semantic occupancy labels",
    ]
    payload = {
        "refine_pts_used_by_semantic_occ": {
            "status": "definitely_used_by_semantic_occ",
            "evidence": [
                "sparseworld_4d_traj.simple_test builds pred_dict with refine_pts and cls_scores then calls get_occ",
                "OPUSHead.get_occ decodes refine_pts and voxelizes them to occupancy grid",
            ],
        },
        "cls_score_used_by_semantic_occ": {
            "status": "definitely_used_by_semantic_occ",
            "evidence": [
                "pred_dict passes cls_scores into get_occ",
                "get_occ uses cls_scores.sigmoid(), class thresholds, scatter_max and argmax to assign semantic occupancy",
            ],
        },
        "semantic_occ_generation_mode": {
            "status": "sparse_query_decode_rasterization",
            "evidence": [
                "semantic_occ is not emitted by a separate dense decoder in simple_test",
                "it is rasterized from sparse refine points + class scores via get_occ()",
            ],
        },
        "planning_only_tensors": {
            "pred_traj": "separate planning/trajectory branch; not directly consumed by get_occ",
        },
        "dependency_graph": dependency_graph,
    }
    write_json(out_dir / "semantic_occ_dependency_trace.json", payload)
    write_csv(out_dir / "semantic_occ_code_trace.csv", rows, fieldnames=["file_path", "line_no", "pattern", "line_text"])
    (out_dir / "semantic_occ_dependency_graph.txt").write_text("\n".join(dependency_graph), encoding="utf-8")
    md = [
        "# semantic_occ dependency trace",
        "",
        "- refine_pts: definitely used by semantic_occ",
        "- cls_score: definitely used by semantic_occ",
        "- semantic_occ generation mode: sparse query decode / rasterization via `get_occ()`",
        "- pred_traj: planning-only branch for this diagnosis",
        "",
        "Dependency graph:",
        "",
    ] + [f"- {x}" for x in dependency_graph]
    (out_dir / "semantic_occ_dependency_trace.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def register_keyword_hooks(model: Any, keywords: list[str]) -> tuple[list[Any], dict[str, Any]]:
    handles = []
    captured: dict[str, Any] = {}

    def hook_factory(name: str):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            if name in captured:
                return
            if isinstance(output, torch.Tensor):
                captured[name] = summarize(output)
            elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                captured[name] = summarize(list(output)[:4])
            elif isinstance(output, dict):
                captured[name] = summarize(output)
            else:
                captured[name] = str(type(output).__name__)
        return _hook

    for name, module in model.named_modules():
        lname = name.lower()
        if any(k in lname for k in keywords):
            handles.append(module.register_forward_hook(hook_factory(name)))
    return handles, captured


def load_sw2_records(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports_dir = project_root / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
    manifest = json.loads((reports_dir / "sw2_run_manifest.json").read_text(encoding="utf-8"))
    sample_rows = list(csv.DictReader((reports_dir / "sw2_sample_manifest.csv").open(encoding="utf-8")))
    return sample_rows, manifest


def load_record_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    raw = torch.load(row["raw_output_path"], map_location="cpu", weights_only=False)
    query = torch.load(row["query_output_path"], map_location="cpu", weights_only=False)
    pred_temporal = torch.load(row["standard_pred_occ_temporal_path"], map_location="cpu", weights_only=False)
    gt_temporal = torch.load(row["standard_gt_occ_temporal_path"], map_location="cpu", weights_only=False)
    return {"raw": raw, "query": query, "pred_temporal": pred_temporal, "gt_temporal": gt_temporal}


def build_support_candidate_manifest(
    query_capture: dict[str, Any],
    out_dir: Path,
    artifacts_dir: Path,
    extra_hook_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    fb = query_capture.get("forward_backbone_outputs", {})
    candidates = {
        "refine_pts_current": fb.get("refine_pts"),
        "cls_score_current": fb.get("cls_score"),
        "all_refine_pts_last": fb.get("outs", {}).get("all_refine_pts", [None])[-1] if fb.get("outs") else None,
        "all_cls_scores_last": fb.get("outs", {}).get("all_cls_scores", [None])[-1] if fb.get("outs") else None,
        "init_points": fb.get("outs", {}).get("init_points") if fb.get("outs") else None,
        "query_feat": fb.get("outs", {}).get("query_feat") if fb.get("outs") else None,
        "forecast_points_list": fb.get("forecast_points_list"),
        "forecast_semantics_list": fb.get("forecast_semantics_list"),
    }
    summary_rows = []
    serializable = {}
    for name, tensor in candidates.items():
        if tensor is None:
            continue
        serializable[name] = to_cpu_artifact(tensor)
        info = summarize(tensor)
        row = {
            "tensor_name": name,
            "summary": json.dumps(info, ensure_ascii=False),
        }
        if isinstance(info, dict) and "shape" in info:
            shape = info["shape"]
            row["shape"] = shape
            row["last_dim"] = shape[-1] if shape else None
            row["has_xyz_like_last_dim"] = shape[-1] == 3 if shape else False
            row["has_class_like_last_dim"] = shape[-1] in (17, 18) if shape else False
            row["has_query_dim_720"] = 720 in shape if shape else False
            row["has_refine_dim_48"] = 48 in shape if shape else False
        summary_rows.append(row)
    manifest = {
        "candidate_count": len(summary_rows),
        "candidate_names": [row["tensor_name"] for row in summary_rows],
        "extra_hook_summary": extra_hook_summary or {},
        "note": "support candidate discovery is constrained to tensors captured on semantic_occ path and keyword hooks",
    }
    write_json(out_dir / "support_tensor_candidate_manifest.json", manifest)
    write_csv(out_dir / "support_tensor_candidate_summary.csv", summary_rows)
    torch.save(serializable, artifacts_dir / "support_tensor_candidates_sample0.pt")
    return manifest, summary_rows, serializable


def evaluate_mapping_variants(
    record: dict[str, Any],
    score_thr: list[float],
    out_dir: Path,
    fig_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fb = record["query"]["forward_backbone_outputs"]
    candidates: list[tuple[str, torch.Tensor, torch.Tensor, int]] = []
    if isinstance(fb.get("refine_pts"), torch.Tensor) and isinstance(fb.get("cls_score"), torch.Tensor):
        candidates.append(("refine_pts_current", fb["refine_pts"], fb["cls_score"], 0))
    if isinstance(fb.get("outs", {}).get("init_points"), torch.Tensor) and isinstance(fb.get("cls_score"), torch.Tensor):
        init_pts = fb["outs"]["init_points"][:, : fb["refine_pts"].shape[1]]
        init_scores = fb["cls_score"][:, :, :1, :]
        candidates.append(("init_points_current", init_pts, init_scores, 0))

    pred = record["pred_temporal"][0]
    gt = record["gt_temporal"][0]
    pred_occ = pred != EMPTY_IDX
    gt_occ = gt != EMPTY_IDX
    false_free = gt_occ & (~pred_occ)
    false_occ = (~gt_occ) & pred_occ

    coord_modes = ["decoded_metric", "normalized01_to_metric", "normalizedm11_to_metric", "raw_grid_to_metric"]
    point_modes = ["all48", "mean48", "max_conf_point", "first_refine", "last_refine"]
    orders = {
        "xyz": (0, 1, 2),
        "yxz": (1, 0, 2),
        "xzy": (0, 2, 1),
        "zxy": (2, 0, 1),
        "yzx": (1, 2, 0),
        "zyx": (2, 1, 0),
    }
    flip_variants = [
        ("noflip", False, False),
        ("flip_x", True, False),
        ("flip_y", False, True),
        ("flip_xy", True, True),
    ]

    rows = []
    gallery = []
    for tensor_name, points_tensor, cls_tensor, horizon_s in candidates:
        # coarse search over mapping semantics using all classes and score gate on.
        for coord_mode in coord_modes:
            for order_name, order in orders.items():
                for flip_name, flip_x, flip_y in flip_variants:
                    for point_mode in point_modes:
                        support = build_support_from_tensor(
                            points_tensor,
                            cls_tensor,
                            score_thr=score_thr,
                            point_mode=point_mode,
                            coord_mode=coord_mode,
                            order=order,
                            flip_x=flip_x,
                            flip_y=flip_y,
                            class_filter_name="all",
                            use_score_gate=True,
                        )
                        geom_support = support["geometric_mask"]
                        sem_support = support["semantic_active_mask"]
                        pred_overlap = overlap_ratio(sem_support, pred_occ)
                        gt_overlap = overlap_ratio(geom_support, gt_occ)
                        ff_overlap = overlap_ratio(geom_support, false_free)
                        fo_overlap = overlap_ratio(sem_support, false_occ)
                        pred_corr = pearson_corr(bev_map(sem_support), bev_map(pred_occ))
                        gt_corr = pearson_corr(bev_map(geom_support), bev_map(gt_occ))
                        ff_corr = pearson_corr(bev_map(geom_support), bev_map(false_free))
                        score = (
                            0.45 * pred_overlap
                            + 0.25 * gt_overlap
                            + 0.15 * support["valid_ratio"]
                            + 0.15 * max(pred_corr, 0.0)
                            - 0.15 * support["diagonal_line_score"]
                        )
                        row = {
                            "tensor_name": tensor_name,
                            "horizon_s": horizon_s,
                            "coord_mode": coord_mode,
                            "order_name": order_name,
                            "flip_variant": flip_name,
                            "point_mode": point_mode,
                            "class_filter": "all",
                            "valid_ratio": support["valid_ratio"],
                            "pred_occupied_overlap": pred_overlap,
                            "gt_occupied_overlap": gt_overlap,
                            "false_free_overlap": ff_overlap,
                            "false_occupied_overlap": fo_overlap,
                            "pred_bev_corr": pred_corr,
                            "gt_bev_corr": gt_corr,
                            "ff_bev_corr": ff_corr,
                            "query_density_entropy": support["query_density_entropy"],
                            "diagonal_line_score": support["diagonal_line_score"],
                            "mapping_score": score,
                        }
                        rows.append(row)
                        gallery.append((score, row, support))

    rows.sort(key=lambda x: x["mapping_score"], reverse=True)
    write_csv(out_dir / "coordinate_mapping_variant_scores.csv", rows)
    write_json(out_dir / "coordinate_mapping_variant_scores.json", {"rows": rows[:200], "total_rows": len(rows)})

    top = gallery[np.argmax([x[0] for x in gallery])]
    top_row = top[1]
    top_support = top[2]
    # refinement stage: class filter ablation on top mapping
    filter_rows = []
    for class_filter in ["all", "dynamic", "small_object", "static"]:
        support = build_support_from_tensor(
            candidates[0][1] if top_row["tensor_name"] == "refine_pts_current" else candidates[1][1],
            candidates[0][2] if top_row["tensor_name"] == "refine_pts_current" else candidates[1][2],
            score_thr=score_thr,
            point_mode=top_row["point_mode"],
            coord_mode=top_row["coord_mode"],
            order=orders[top_row["order_name"]],
            flip_x=top_row["flip_variant"] in {"flip_x", "flip_xy"},
            flip_y=top_row["flip_variant"] in {"flip_y", "flip_xy"},
            class_filter_name=class_filter,
            use_score_gate=True,
        )
        filter_rows.append(
            {
                "tensor_name": top_row["tensor_name"],
                "coord_mode": top_row["coord_mode"],
                "order_name": top_row["order_name"],
                "flip_variant": top_row["flip_variant"],
                "point_mode": top_row["point_mode"],
                "class_filter": class_filter,
                "pred_occupied_overlap": overlap_ratio(support["semantic_active_mask"], pred_occ),
                "gt_occupied_overlap": overlap_ratio(support["geometric_mask"], gt_occ),
                "false_free_overlap": overlap_ratio(support["geometric_mask"], false_free),
            }
        )
    write_csv(out_dir / "coordinate_mapping_variant_filter_ablation.csv", filter_rows)

    top5 = sorted(gallery, key=lambda x: x[0], reverse=True)[:5]
    bev_figures = []
    for rank, (_, row, support) in enumerate(top5, start=1):
        density = bev_map(support["geometric_mask"])
        bev_figures.append((density, f"#{rank} {row['tensor_name']} {row['coord_mode']} {row['order_name']} {row['flip_variant']} {row['point_mode']}"))
    overlay = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype=np.uint8)
    overlay[bev_map(gt_occ).astype(bool)] = np.array([0, 255, 0], dtype=np.uint8)
    overlay[bev_map(pred_occ).astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
    overlay[bev_map(false_free).astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bev_map(top_support["geometric_mask"]), cmap="magma")
    ax.set_title("Best mapping query density BEV")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "mapping_top1_query_density_bev.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    ax.set_title("GT / Pred / False-free overlay")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "mapping_top1_gt_pred_ff_overlay.png", dpi=150)
    plt.close(fig)
    rows_n = math.ceil(len(bev_figures) / 2)
    fig, axes = plt.subplots(rows_n, 2, figsize=(10, 4 * rows_n))
    axes = np.array(axes).reshape(rows_n, 2)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, (img, title) in zip(axes.flatten(), bev_figures):
        ax.imshow(img, cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "mapping_top5_gallery.png", dpi=150)
    plt.close(fig)
    best_summary = {
        "best_mapping_row": top_row,
        "top5_preview": [x[1] for x in top5],
        "interpretation": "best mapping is the support proxy with highest semantic-active overlap to Pred occupied and highest geometric overlap to GT occupied while minimizing diagonal-line artifact",
    }
    md = [
        "# Best coordinate mapping summary",
        "",
        f"- tensor: `{top_row['tensor_name']}`",
        f"- coord_mode: `{top_row['coord_mode']}`",
        f"- order: `{top_row['order_name']}`",
        f"- flip: `{top_row['flip_variant']}`",
        f"- point_mode: `{top_row['point_mode']}`",
        f"- pred overlap: `{round(top_row['pred_occupied_overlap'], 4)}`",
        f"- GT overlap: `{round(top_row['gt_occupied_overlap'], 4)}`",
        f"- diagonal_line_score: `{round(top_row['diagonal_line_score'], 4)}`",
    ]
    (out_dir / "best_coordinate_mapping_summary.md").write_text("\n".join(md), encoding="utf-8")
    return best_summary, rows


def query_to_pred_attribution(
    records: list[dict[str, Any]],
    score_thr: list[float],
    best_mapping: dict[str, Any],
    out_dir: Path,
    fig_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    radius = 2
    row0_overlay = None
    distance_examples = {}
    order_lookup = {
        "xyz": (0, 1, 2),
        "yxz": (1, 0, 2),
        "xzy": (0, 2, 1),
        "zxy": (2, 0, 1),
        "yzx": (1, 2, 0),
        "zyx": (2, 1, 0),
    }
    for rec in records:
        fb = rec["query"]["forward_backbone_outputs"]
        for h in range(rec["pred_temporal"].shape[0]):
            if h == 0:
                points_tensor = fb["refine_pts"]
                cls_tensor = fb["cls_score"]
            else:
                points_tensor = fb["forecast_points_list"][h - 1]
                cls_tensor = fb["forecast_semantics_list"][h - 1]
            support = build_support_from_tensor(
                points_tensor,
                cls_tensor,
                score_thr=score_thr,
                point_mode=best_mapping["point_mode"],
                coord_mode=best_mapping["coord_mode"],
                order=order_lookup[best_mapping["order_name"]],
                flip_x=best_mapping["flip_variant"] in {"flip_x", "flip_xy"},
                flip_y=best_mapping["flip_variant"] in {"flip_y", "flip_xy"},
                class_filter_name="all",
                use_score_gate=True,
            )
            pred = rec["pred_temporal"][h]
            gt = rec["gt_temporal"][h]
            stats = region_stats_from_support(pred, gt, support, radius=radius)
            row = {
                "sample_index": rec["sample_index"],
                "sample_token": rec["sample_token"],
                "scene_token": rec["scene_token"],
                "scene_name": rec["scene_name"],
                "horizon_s": h,
                "tp_geometric_coverage_ratio": stats["tp_occupied"]["geometric_coverage_ratio"],
                "tp_semantic_active_coverage_ratio": stats["tp_occupied"]["semantic_active_coverage_ratio"],
                "ff_geometric_coverage_ratio": stats["false_free"]["geometric_coverage_ratio"],
                "ff_semantic_active_coverage_ratio": stats["false_free"]["semantic_active_coverage_ratio"],
                "fo_geometric_coverage_ratio": stats["false_occupied"]["geometric_coverage_ratio"],
                "fo_semantic_active_coverage_ratio": stats["false_occupied"]["semantic_active_coverage_ratio"],
                "ff_nearest_query_distance_bev_mean": stats["false_free"]["region_nearest_query_distance_bev_mean"],
                "tp_nearest_query_distance_bev_mean": stats["tp_occupied"]["region_nearest_query_distance_bev_mean"],
                "fo_nearest_query_distance_bev_mean": stats["false_occupied"]["region_nearest_query_distance_bev_mean"],
                "ff_support_conf_mean": stats["false_free"]["region_support_conf_mean"],
                "tp_support_conf_mean": stats["tp_occupied"]["region_support_conf_mean"],
                "fo_support_conf_mean": stats["false_occupied"]["region_support_conf_mean"],
                "ff_support_density_mean": stats["false_free"]["region_support_density_mean"],
                "tp_support_density_mean": stats["tp_occupied"]["region_support_density_mean"],
                "fo_support_density_mean": stats["false_occupied"]["region_support_density_mean"],
            }
            rows.append(row)
            if rec["sample_index"] == records[0]["sample_index"] and h == 0:
                regions = classify_regions(gt, pred)
                overlay = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype=np.uint8)
                overlay[bev_map(regions["tp_occupied"]).astype(bool)] = np.array([0, 255, 0], dtype=np.uint8)
                overlay[bev_map(regions["false_free"]).astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
                overlay[bev_map(regions["false_occupied"]).astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
                row0_overlay = overlay
                distance_examples = stats
    write_json(out_dir / "query_to_pred_attribution_matrix.json", {"rows": rows})
    write_csv(out_dir / "query_to_pred_attribution_matrix.csv", rows)
    mean_ff_geo = float(np.mean([r["ff_geometric_coverage_ratio"] for r in rows]))
    mean_ff_sem = float(np.mean([r["ff_semantic_active_coverage_ratio"] for r in rows]))
    mean_tp_sem = float(np.mean([r["tp_semantic_active_coverage_ratio"] for r in rows]))
    conclusion = "unknown"
    if mean_tp_sem < 0.2:
        conclusion = "Q1_candidate_tensor_not_supportive"
    elif mean_tp_sem > 0.6 and mean_ff_geo < 0.3:
        conclusion = "Q2_true_query_allocation_failure"
    elif mean_tp_sem > 0.6 and mean_ff_geo > 0.6:
        conclusion = "Q3_decoder_or_head_semantic_failure"
    elif mean_tp_sem > 0.6 and mean_ff_geo < 0.3:
        conclusion = "Q4_targeted_allocation_failure"
    elif mean_tp_sem > 0.4 and mean_ff_geo < 0.25:
        conclusion = "Q4_targeted_allocation_failure"
    summary = {
        "mean_ff_geometric_coverage_ratio": mean_ff_geo,
        "mean_ff_semantic_active_coverage_ratio": mean_ff_sem,
        "mean_tp_semantic_active_coverage_ratio": mean_tp_sem,
        "conclusion": conclusion,
    }
    (out_dir / "query_to_pred_attribution_summary.md").write_text(
        "# Query-to-pred attribution summary\n\n"
        f"- mean false-free geometric coverage: `{mean_ff_geo}`\n"
        f"- mean false-free semantic-active coverage: `{mean_ff_sem}`\n"
        f"- mean TP semantic-active coverage: `{mean_tp_sem}`\n"
        f"- conclusion: `{conclusion}`\n",
        encoding="utf-8",
    )
    if row0_overlay is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(row0_overlay)
        ax.set_title("TP / FF / FO regions (sample0)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "query_density_on_tp_ff_fo_regions.png", dpi=150)
        plt.close(fig)
    # distance bar
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["TP", "FF", "FO"]
    vals = [
        float(np.mean([r["tp_nearest_query_distance_bev_mean"] for r in rows])),
        float(np.mean([r["ff_nearest_query_distance_bev_mean"] for r in rows])),
        float(np.mean([r["fo_nearest_query_distance_bev_mean"] for r in rows])),
    ]
    ax.bar(labels, vals)
    ax.set_title("Nearest query distance by error type")
    ax.set_ylabel("Mean BEV distance (cells)")
    fig.tight_layout()
    fig.savefig(fig_dir / "nearest_query_distance_by_error_type.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["TP", "FF", "FO"], [
        summary["mean_tp_semantic_active_coverage_ratio"],
        summary["mean_ff_geometric_coverage_ratio"],
        float(np.mean([r["fo_semantic_active_coverage_ratio"] for r in rows])),
    ])
    ax.set_title("Coverage in four quadrants")
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_four_quadrants_bar.png", dpi=150)
    plt.close(fig)
    return rows, summary


def run_causal_tests(
    sw2: Any,
    repo_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    sample_index: int,
    horizons: list[int],
    score_thr: list[float],
    out_dir: Path,
    fig_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg, dataset, model, _ = sw2.setup_runtime(repo_root, config_path, checkpoint_path, "val", False)
    collate_fn = load_module_from_path(
        "sw2_main_for_collate",
        repo_root.parent.parent / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
    ).setup_runtime  # placeholder access
    # rebuild collate from SW-2 setup
    _, _, _, runtime_meta = sw2.setup_runtime(repo_root, config_path, checkpoint_path, "val", False)
    collate = runtime_meta["collate"]
    raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate)
    batch_unwrapped = sw2.unwrap(raw_sample)
    gt_temporal = torch.stack(
        [torch.as_tensor(batch_unwrapped["voxel_semantics"]).long()]
        + [torch.as_tensor(batch_unwrapped["temporal_semantics"][i]["voxel_semantics"]).long() for i in range(1, 7)],
        dim=0,
    )
    model_inputs = sw2.move_to_cuda(batch)

    def run_with_patch(patch_fn: Callable[[dict[str, Any], Any], None], name: str) -> dict[str, Any]:
        original_forward_backbone = model.forward_backbone

        def wrapped_forward_backbone(*fb_args: Any, **fb_kwargs: Any) -> Any:
            outputs = original_forward_backbone(*fb_args, **fb_kwargs)
            patch_fn(outputs, model)
            return outputs

        model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]
        try:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **model_inputs)
            result_cpu = to_cpu_artifact(result)
        finally:
            model.forward_backbone = original_forward_backbone  # type: ignore[assignment]
        return result_cpu

    baseline = run_with_patch(lambda outputs, model_: None, "baseline")

    def modify_current_slice(tensor: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor], model_: Any) -> None:
        ind = model_.pts_bbox_head.ind_stamps_all == 0
        tensor[:, ind] = func(tensor[:, ind])

    interventions: list[tuple[str, Callable[[dict[str, Any], Any], None]]] = []
    interventions.append(
        (
            "refine_pts_current_shuffle_query",
            lambda outputs, model_: modify_current_slice(outputs["outs"]["all_refine_pts"][-1], lambda x: x[:, torch.randperm(x.shape[1])], model_),
        )
    )
    interventions.append(
        (
            "refine_pts_current_center_replace",
            lambda outputs, model_: modify_current_slice(outputs["outs"]["all_refine_pts"][-1], lambda x: x.mean(dim=2, keepdim=True).expand_as(x), model_),
        )
    )
    interventions.append(
        (
            "cls_score_current_suppress_foreground",
            lambda outputs, model_: modify_current_slice(outputs["outs"]["all_cls_scores"][-1], lambda x: x - 5.0, model_),
        )
    )
    interventions.append(
        (
            "cls_score_current_boost_foreground",
            lambda outputs, model_: modify_current_slice(outputs["outs"]["all_cls_scores"][-1], lambda x: x + 5.0, model_),
        )
    )
    interventions.append(
        (
            "forecast_points_h6_shuffle_query",
            lambda outputs, model_: outputs["forecast_points_list"].__setitem__(5, outputs["forecast_points_list"][5][:, torch.randperm(outputs["forecast_points_list"][5].shape[1])]),
        )
    )
    interventions.append(
        (
            "forecast_semantics_h6_suppress",
            lambda outputs, model_: outputs["forecast_semantics_list"].__setitem__(5, outputs["forecast_semantics_list"][5] - 5.0),
        )
    )

    rows = []
    base_pred = {h: torch.as_tensor(baseline[f"semantic_occ_{h}s"][0]).long() for h in horizons}
    for name, fn in interventions:
        result = run_with_patch(fn, name)
        fig_pairs = []
        for h in horizons:
            pred = torch.as_tensor(result[f"semantic_occ_{h}s"][0]).long()
            gt = gt_temporal[h]
            changed_ratio = float((pred != base_pred[h]).float().mean().item())
            pred_occ = pred != EMPTY_IDX
            gt_occ = gt != EMPTY_IDX
            inter = torch.logical_and(pred_occ, gt_occ).sum().item()
            union = torch.logical_or(pred_occ, gt_occ).sum().item()
            false_free = torch.logical_and(gt_occ, ~pred_occ).sum().item()
            false_occ = torch.logical_and(~gt_occ, pred_occ).sum().item()
            row = {
                "intervention_name": name,
                "horizon_s": h,
                "changed_voxel_ratio": changed_ratio,
                "pred_occupied_count": int(pred_occ.sum().item()),
                "occupied_iou": float(inter / union) if union else 0.0,
                "false_free_rate": float(false_free / gt_occ.sum().item()) if gt_occ.sum().item() else 0.0,
                "false_occupied_rate": float(false_occ / (~gt_occ).sum().item()) if (~gt_occ).sum().item() else 0.0,
            }
            rows.append(row)
            if h in {0, 6}:
                diff = (pred != base_pred[h]).any(dim=-1).cpu().numpy().astype(np.uint8)
                fig_pairs.append((diff, f"{name} t={h}s diff"))
        if fig_pairs:
            # only use last loop's images for gallery after all interventions
            pass
    write_json(out_dir / "query_causal_controllability_tests.json", {"rows": rows})
    write_csv(out_dir / "query_causal_controllability_tests.csv", rows)
    md = ["# Query causal controllability summary", ""]
    for name in sorted({r["intervention_name"] for r in rows}):
        sub = [r for r in rows if r["intervention_name"] == name and r["horizon_s"] in {0, 6}]
        md.append(f"- `{name}`: " + ", ".join([f"t={r['horizon_s']} changed_voxel_ratio={round(r['changed_voxel_ratio'],4)}" for r in sub]))
    (out_dir / "query_causal_controllability_summary.md").write_text("\n".join(md), encoding="utf-8")
    # summary conclusion
    def mean_change(prefix: str, horizon: int) -> float:
        vals = [r["changed_voxel_ratio"] for r in rows if r["intervention_name"] == prefix and r["horizon_s"] == horizon]
        return float(np.mean(vals)) if vals else 0.0
    conclusion = {
        "refine_pts_current_is_causal": mean_change("refine_pts_current_shuffle_query", 0) > 0.05 or mean_change("refine_pts_current_center_replace", 0) > 0.05,
        "cls_score_current_is_causal": mean_change("cls_score_current_suppress_foreground", 0) > 0.05 or mean_change("cls_score_current_boost_foreground", 0) > 0.05,
        "future_refine_pts_is_causal": mean_change("forecast_points_h6_shuffle_query", 6) > 0.05,
        "future_cls_score_is_causal": mean_change("forecast_semantics_h6_suppress", 6) > 0.05,
    }
    # simple diff gallery using two strongest interventions
    top_rows = sorted(rows, key=lambda x: x["changed_voxel_ratio"], reverse=True)[:4]
    panels = []
    for r in top_rows:
        result = next(rr for rr in rows if rr["intervention_name"] == r["intervention_name"] and rr["horizon_s"] == r["horizon_s"])
    # generate figure from stored baseline and reruns of selected interventions
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [f"{r['intervention_name']}\nt={r['horizon_s']}" for r in top_rows],
        [r["changed_voxel_ratio"] for r in top_rows],
    )
    ax.set_title("Causal intervention output diff")
    ax.set_ylabel("Changed voxel ratio")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "causal_intervention_output_diff_panel.png", dpi=150)
    plt.close(fig)
    return rows, conclusion


def compute_group_mask(label_grid: torch.Tensor, group_ids: list[int]) -> torch.Tensor:
    ids = torch.as_tensor(group_ids, device=label_grid.device)
    return torch.isin(label_grid, ids)


def local_max_over_radius(grid: torch.Tensor, radius: int) -> torch.Tensor:
    t = grid[None, None].float()
    out = F.max_pool3d(t, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return out[0, 0]


def small_object_failure_localization(
    records: list[dict[str, Any]],
    score_thr: list[float],
    best_mapping: dict[str, Any],
    out_dir: Path,
    fig_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order_lookup = {
        "xyz": (0, 1, 2),
        "yxz": (1, 0, 2),
        "xzy": (0, 2, 1),
        "zxy": (2, 0, 1),
        "yzx": (1, 2, 0),
        "zyx": (2, 1, 0),
    }
    rows = []
    overlay_done = False
    for rec in records:
        fb = rec["query"]["forward_backbone_outputs"]
        for h in range(rec["pred_temporal"].shape[0]):
            if h == 0:
                points_tensor = fb["refine_pts"]
                cls_tensor = fb["cls_score"]
            else:
                points_tensor = fb["forecast_points_list"][h - 1]
                cls_tensor = fb["forecast_semantics_list"][h - 1]
            support = build_support_from_tensor(
                points_tensor,
                cls_tensor,
                score_thr=score_thr,
                point_mode=best_mapping["point_mode"],
                coord_mode=best_mapping["coord_mode"],
                order=order_lookup[best_mapping["order_name"]],
                flip_x=best_mapping["flip_variant"] in {"flip_x", "flip_xy"},
                flip_y=best_mapping["flip_variant"] in {"flip_y", "flip_xy"},
                class_filter_name="all",
                use_score_gate=True,
            )
            pred = rec["pred_temporal"][h]
            gt = rec["gt_temporal"][h]
            gt_small = compute_group_mask(gt, CLASS_GROUPS["small_object"])
            pred_small = compute_group_mask(pred, CLASS_GROUPS["small_object"])
            ff_small = gt_small & (~pred_small)
            gt_large_dyn = compute_group_mask(gt, CLASS_GROUPS["dynamic_vehicle"])
            support_bev_dist = distance_transform_bev(support["geometric_mask"])
            gt_small_bev = bev_map(gt_small)
            ff_small_bev = bev_map(ff_small)
            gt_large_bev = bev_map(gt_large_dyn)
            small_score_grid = torch.zeros(GRID_SIZE, dtype=torch.float32)
            dynamic_score_grid = torch.zeros(GRID_SIZE, dtype=torch.float32)
            if support["semantic_active_logits_sparse"].numel() > 0:
                coords = support["support_coords"] if "support_coords" in support else None
            radii = [1, 2, 3, 5, 8]
            coverage = {}
            for r in radii:
                dil = dilate3d(support["geometric_mask"], r)
                coverage[f"small_gt_coverage_r{r}"] = float((dil & gt_small).sum().item() / max(1, gt_small.sum().item()))
                coverage[f"small_ff_coverage_r{r}"] = float((dil & ff_small).sum().item() / max(1, ff_small.sum().item()))
                coverage[f"large_dynamic_gt_coverage_r{r}"] = float((dil & gt_large_dyn).sum().item() / max(1, gt_large_dyn.sum().item()))
            rows.append(
                {
                    "sample_index": rec["sample_index"],
                    "sample_token": rec["sample_token"],
                    "scene_token": rec["scene_token"],
                    "horizon_s": h,
                    "small_gt_voxel_count": int(gt_small.sum().item()),
                    "small_ff_voxel_count": int(ff_small.sum().item()),
                    "large_dynamic_gt_voxel_count": int(gt_large_dyn.sum().item()),
                    "small_gt_nearest_query_distance_bev_mean": float(support_bev_dist[gt_small_bev > 0].mean()) if gt_small_bev.sum() else 0.0,
                    "small_ff_nearest_query_distance_bev_mean": float(support_bev_dist[ff_small_bev > 0].mean()) if ff_small_bev.sum() else 0.0,
                    "large_dynamic_gt_nearest_query_distance_bev_mean": float(support_bev_dist[gt_large_bev > 0].mean()) if gt_large_bev.sum() else 0.0,
                    **coverage,
                }
            )
            if not overlay_done and rec["sample_index"] == records[0]["sample_index"] and h == 0:
                overlay = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype=np.uint8)
                overlay[gt_small_bev.astype(bool)] = np.array([0, 255, 0], dtype=np.uint8)
                overlay[bev_map(pred_small).astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
                overlay[ff_small_bev.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(overlay)
                ax.set_title("Small-object GT / Pred / FF overlay")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(fig_dir / "small_object_gt_pred_query_overlay_sample0.png", dpi=150)
                plt.close(fig)
                overlay_done = True
    write_json(out_dir / "small_object_failure_localization.json", {"rows": rows})
    write_csv(out_dir / "small_object_failure_localization.csv", rows)
    mean_small_ff_dist = float(np.mean([r["small_ff_nearest_query_distance_bev_mean"] for r in rows if r["small_ff_voxel_count"] > 0]))
    mean_large_dyn_dist = float(np.mean([r["large_dynamic_gt_nearest_query_distance_bev_mean"] for r in rows if r["large_dynamic_gt_voxel_count"] > 0]))
    mean_small_cov = float(np.mean([r["small_ff_coverage_r2"] for r in rows if r["small_ff_voxel_count"] > 0]))
    summary = {
        "mean_small_ff_nearest_query_distance_bev": mean_small_ff_dist,
        "mean_large_dynamic_gt_nearest_query_distance_bev": mean_large_dyn_dist,
        "mean_small_ff_coverage_r2": mean_small_cov,
    }
    (out_dir / "small_object_failure_summary.md").write_text(
        "# Small-object failure summary\n\n"
        f"- mean small-object false-free nearest query distance (BEV): `{mean_small_ff_dist}`\n"
        f"- mean large-dynamic GT nearest query distance (BEV): `{mean_large_dyn_dist}`\n"
        f"- mean small-object false-free coverage r=2: `{mean_small_cov}`\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["small_ff", "large_dynamic_gt"], [mean_small_ff_dist, mean_large_dyn_dist])
    ax.set_title("Small-object vs large-dynamic nearest query distance")
    fig.tight_layout()
    fig.savefig(fig_dir / "small_object_nearest_query_distance_hist.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["small_ff_cov_r2", "large_dyn_cov_r2"], [
        mean_small_cov,
        float(np.mean([r["large_dynamic_gt_coverage_r2"] for r in rows if r["large_dynamic_gt_voxel_count"] > 0])),
    ])
    ax.set_title("Small-object vs large-dynamic coverage")
    fig.tight_layout()
    fig.savefig(fig_dir / "small_vs_large_dynamic_coverage.png", dpi=150)
    plt.close(fig)
    return rows, summary


def new_visible_failure_localization(
    records: list[dict[str, Any]],
    score_thr: list[float],
    best_mapping: dict[str, Any],
    out_dir: Path,
    fig_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order_lookup = {
        "xyz": (0, 1, 2),
        "yxz": (1, 0, 2),
        "xzy": (0, 2, 1),
        "zxy": (2, 0, 1),
        "yzx": (1, 2, 0),
        "zyx": (2, 1, 0),
    }
    rows = []
    overlay_done = False
    for rec in records:
        fb = rec["query"]["forward_backbone_outputs"]
        gt0 = rec["gt_temporal"][0]
        for h in range(1, rec["pred_temporal"].shape[0]):
            points_tensor = fb["forecast_points_list"][h - 1]
            cls_tensor = fb["forecast_semantics_list"][h - 1]
            support = build_support_from_tensor(
                points_tensor,
                cls_tensor,
                score_thr=score_thr,
                point_mode=best_mapping["point_mode"],
                coord_mode=best_mapping["coord_mode"],
                order=order_lookup[best_mapping["order_name"]],
                flip_x=best_mapping["flip_variant"] in {"flip_x", "flip_xy"},
                flip_y=best_mapping["flip_variant"] in {"flip_y", "flip_xy"},
                class_filter_name="all",
                use_score_gate=True,
            )
            gt = rec["gt_temporal"][h]
            pred = rec["pred_temporal"][h]
            gt0_occ = gt0 != EMPTY_IDX
            gt_occ = gt != EMPTY_IDX
            pred_occ = pred != EMPTY_IDX
            persistent = gt0_occ & gt_occ
            new_visible = (~gt0_occ) & gt_occ
            disappeared = gt0_occ & (~gt_occ)
            geo_dil = dilate3d(support["geometric_mask"], 2)
            sem_dil = dilate3d(support["semantic_active_mask"], 2)
            dist_bev = distance_transform_bev(support["geometric_mask"])
            def cov(mask: torch.Tensor, dil: torch.Tensor) -> float:
                den = mask.sum().item()
                return float((mask & dil).sum().item() / den) if den else 0.0
            def dist(mask: torch.Tensor) -> float:
                bev = bev_map(mask)
                return float(dist_bev[bev > 0].mean()) if bev.sum() else 0.0
            rows.append(
                {
                    "sample_index": rec["sample_index"],
                    "sample_token": rec["sample_token"],
                    "scene_token": rec["scene_token"],
                    "horizon_s": h,
                    "persistent_gt_count": int(persistent.sum().item()),
                    "new_visible_gt_count": int(new_visible.sum().item()),
                    "disappeared_gt_count": int(disappeared.sum().item()),
                    "persistent_pred_recall": float((persistent & pred_occ).sum().item() / max(1, persistent.sum().item())),
                    "new_visible_pred_recall": float((new_visible & pred_occ).sum().item() / max(1, new_visible.sum().item())),
                    "persistent_geometric_coverage": cov(persistent, geo_dil),
                    "new_visible_geometric_coverage": cov(new_visible, geo_dil),
                    "persistent_semantic_active_coverage": cov(persistent, sem_dil),
                    "new_visible_semantic_active_coverage": cov(new_visible, sem_dil),
                    "persistent_nearest_query_distance_bev_mean": dist(persistent),
                    "new_visible_nearest_query_distance_bev_mean": dist(new_visible),
                    "disappeared_stale_pred_ratio": float((disappeared & pred_occ).sum().item() / max(1, disappeared.sum().item())),
                }
            )
            if not overlay_done and rec["sample_index"] == records[0]["sample_index"] and h == 1:
                overlay = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype=np.uint8)
                overlay[bev_map(new_visible).astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
                overlay[bev_map(persistent).astype(bool)] = np.array([0, 255, 0], dtype=np.uint8)
                overlay[bev_map(support["geometric_mask"]).astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(overlay)
                ax.set_title("New-visible / persistent / query overlay")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(fig_dir / "new_visible_query_overlay_sample0.png", dpi=150)
                plt.close(fig)
                overlay_done = True
    write_json(out_dir / "new_visible_failure_localization.json", {"rows": rows})
    write_csv(out_dir / "new_visible_failure_localization.csv", rows)
    summary = {
        "mean_new_visible_geometric_coverage": float(np.mean([r["new_visible_geometric_coverage"] for r in rows if r["new_visible_gt_count"] > 0])),
        "mean_persistent_geometric_coverage": float(np.mean([r["persistent_geometric_coverage"] for r in rows if r["persistent_gt_count"] > 0])),
        "mean_new_visible_pred_recall": float(np.mean([r["new_visible_pred_recall"] for r in rows if r["new_visible_gt_count"] > 0])),
        "mean_persistent_pred_recall": float(np.mean([r["persistent_pred_recall"] for r in rows if r["persistent_gt_count"] > 0])),
    }
    (out_dir / "new_visible_failure_summary.md").write_text(
        "# New-visible failure summary\n\n"
        f"- mean new-visible geometric coverage: `{summary['mean_new_visible_geometric_coverage']}`\n"
        f"- mean persistent geometric coverage: `{summary['mean_persistent_geometric_coverage']}`\n"
        f"- mean new-visible pred recall: `{summary['mean_new_visible_pred_recall']}`\n"
        f"- mean persistent pred recall: `{summary['mean_persistent_pred_recall']}`\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        sorted({r["horizon_s"] for r in rows}),
        [
            float(np.mean([rr["new_visible_geometric_coverage"] for rr in rows if rr["horizon_s"] == h and rr["new_visible_gt_count"] > 0]))
            for h in sorted({r["horizon_s"] for r in rows})
        ],
        marker="o",
        label="new_visible",
    )
    ax.plot(
        sorted({r["horizon_s"] for r in rows}),
        [
            float(np.mean([rr["persistent_geometric_coverage"] for rr in rows if rr["horizon_s"] == h and rr["persistent_gt_count"] > 0]))
            for h in sorted({r["horizon_s"] for r in rows})
        ],
        marker="o",
        label="persistent",
    )
    ax.set_title("New-visible coverage over horizon")
    ax.set_xlabel("Horizon (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "new_visible_coverage_over_horizon.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["persistent", "new_visible"], [summary["mean_persistent_geometric_coverage"], summary["mean_new_visible_geometric_coverage"]])
    ax.set_title("Persistent vs new-visible query coverage")
    fig.tight_layout()
    fig.savefig(fig_dir / "persistent_vs_new_visible_query_distance.png", dpi=150)
    plt.close(fig)
    return rows, summary


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    project_root = repo_root.parent.parent
    reports_dir = project_root / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation"
    logs_dir = project_root / "logs/sparseworld_mainline/stage_sw3_query_support_validation"
    scripts_dir = project_root / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation"
    fig_dir = project_root / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw3_query_support_validation"
    artifacts_dir = project_root / "artifacts/sparseworld_mainline/stage_sw3_query_support_validation"
    for p in [reports_dir, logs_dir, scripts_dir, fig_dir, artifacts_dir]:
        p.mkdir(parents=True, exist_ok=True)

    sw2_path = project_root / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py"
    sw2 = load_module_from_path("sw2_main", sw2_path)
    sw2_sample_rows, sw2_manifest = load_sw2_records(project_root)
    success_rows = [row for row in sw2_sample_rows if row["forward_status"] == "success"]
    if args.sample_indices:
        requested = {int(x) for x in args.sample_indices.split(",") if x.strip()}
        success_rows = [row for row in success_rows if int(row["sample_index"]) in requested]
    else:
        success_rows = success_rows[: args.num_samples]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    effective_rows = success_rows
    sample0_row = effective_rows[0]

    # Rebuild runtime once for hook-based discovery.
    cfg, dataset, model, runtime_meta = sw2.setup_runtime(repo_root, config_path, checkpoint_path, args.split, False)
    score_thr = list(model.pts_bbox_head.test_cfg["score_thr"])

    # Phase 1 manifest
    run_manifest = {
        "source_sw2_effective_subset_size": sw2_manifest["effective_subset_size"],
        "requested_num_samples": args.num_samples,
        "effective_sample_count": len(effective_rows),
        "sample_indices": [int(row["sample_index"]) for row in effective_rows],
        "horizons": horizons,
    }
    write_json(reports_dir / "sw3_run_manifest.json", run_manifest)
    write_csv(reports_dir / "sw3_sample_manifest.csv", effective_rows)

    # Phase 2 dependency trace
    dep_trace = semantic_occ_dependency_trace(repo_root, reports_dir)

    # Phase 3 support tensor discovery on sample0 with real hooks.
    raw_sample, batch = sw2.extract_sample_batch(dataset, int(sample0_row["sample_index"]), runtime_meta["collate"])
    model_inputs = sw2.move_to_cuda(batch)
    hook_handles, hook_summary = register_keyword_hooks(
        model,
        ["query", "dynamic", "sparse", "refine", "range", "forecast", "state", "temporal", "decoder", "occ", "occupancy", "semantic", "bev", "head"],
    )
    capture_holder: dict[str, Any] = {}
    original_forward_backbone = model.forward_backbone

    def wrapped_forward_backbone(*fb_args: Any, **fb_kwargs: Any) -> Any:
        outputs = original_forward_backbone(*fb_args, **fb_kwargs)
        capture_holder["forward_backbone_outputs"] = to_cpu_artifact(outputs)
        return outputs

    model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]
    with torch.no_grad():
        _ = model(return_loss=False, rescale=True, **model_inputs)
    model.forward_backbone = original_forward_backbone  # type: ignore[assignment]
    for h in hook_handles:
        h.remove()
    candidate_manifest, candidate_summary_rows, candidate_tensors = build_support_candidate_manifest(
        capture_holder,
        reports_dir,
        artifacts_dir,
        extra_hook_summary=hook_summary,
    )

    # Sample0 record for heavy validation
    sample0_record = {
        "sample_index": int(sample0_row["sample_index"]),
        "sample_token": sample0_row["sample_token"],
        "scene_token": sample0_row["scene_token"],
        "scene_name": sample0_row["scene_name"],
        **load_record_artifacts(sample0_row),
    }

    # Phase 4 mapping validation
    best_mapping_summary, mapping_rows = evaluate_mapping_variants(sample0_record, score_thr, reports_dir, fig_dir)
    best_mapping = best_mapping_summary["best_mapping_row"]

    # Phase 5 query-to-pred attribution on first 5 samples, all requested horizons.
    first5_records = []
    for row in effective_rows[:5]:
        first5_records.append(
            {
                "sample_index": int(row["sample_index"]),
                "sample_token": row["sample_token"],
                "scene_token": row["scene_token"],
                "scene_name": row["scene_name"],
                **load_record_artifacts(row),
            }
        )
    attribution_rows, attribution_summary = query_to_pred_attribution(first5_records, score_thr, best_mapping, reports_dir, fig_dir)

    # Phase 6 causal controllability on sample0 t=0 and t=6.
    causal_rows, causal_summary = run_causal_tests(sw2, repo_root, config_path, checkpoint_path, int(sample0_row["sample_index"]), [0, 6], score_thr, reports_dir, fig_dir)

    # Phase 7 small-object localization on first 5 heavy subset.
    small_rows, small_summary = small_object_failure_localization(first5_records, score_thr, best_mapping, reports_dir, fig_dir)

    # Phase 8 new-visible localization on first 5 heavy subset.
    nv_rows, nv_summary = new_visible_failure_localization(first5_records, score_thr, best_mapping, reports_dir, fig_dir)

    # Phase 9 resolution decision.
    refine_pts_support_valid = bool(dep_trace["refine_pts_used_by_semantic_occ"]["status"] == "definitely_used_by_semantic_occ" and causal_summary["refine_pts_current_is_causal"])
    coordinate_mapping_bug = (
        best_mapping["tensor_name"] == "refine_pts_current"
        and best_mapping["point_mode"] != "mean48"
        and best_mapping["mapping_score"] > 0.35
    )
    small_object_allocation_failure = small_summary["mean_small_ff_coverage_r2"] < 0.25 and small_summary["mean_small_ff_nearest_query_distance_bev"] > small_summary["mean_large_dynamic_gt_nearest_query_distance_bev"]
    new_visible_birth_failure = nv_summary["mean_new_visible_geometric_coverage"] + 0.15 < nv_summary["mean_persistent_geometric_coverage"]
    decoder_failure = attribution_summary["mean_ff_geometric_coverage_ratio"] > 0.6 and attribution_summary["mean_ff_semantic_active_coverage_ratio"] < 0.3

    case = "G"
    next_action = "deeper support tensor instrumentation"
    case_reason = "support unresolved"
    if not refine_pts_support_valid:
        case = "A"
        next_action = "deeper support tensor instrumentation"
        case_reason = "refine_pts proxy invalid"
    elif coordinate_mapping_bug and not (small_object_allocation_failure or new_visible_birth_failure or decoder_failure):
        case = "B"
        next_action = "update coverage adapter and rerun SW-2 coverage"
        case_reason = "coordinate/support proxy bug fixed by mapping"
    elif small_object_allocation_failure:
        case = "D"
        next_action = "Small-object-aware Query Allocation Diagnosis"
        case_reason = "targeted small-object allocation failure"
    elif new_visible_birth_failure:
        case = "E"
        next_action = "Future Query Birth / Completion Diagnosis"
        case_reason = "new-visible query birth failure"
    elif decoder_failure:
        case = "F"
        next_action = "Decoder/Head Calibration Diagnosis"
        case_reason = "false-free regions remain covered geometrically but not semantically active"
    elif attribution_summary["mean_ff_geometric_coverage_ratio"] < 0.3 and attribution_summary["mean_tp_semantic_active_coverage_ratio"] > 0.5:
        case = "C"
        next_action = "Query Allocation Repair / Oracle Coverage Upper Bound"
        case_reason = "true global allocation failure"

    decision_payload = {
        "resolution_case": case,
        "case_reason": case_reason,
        "refine_pts_support_valid": refine_pts_support_valid,
        "coordinate_mapping_bug": coordinate_mapping_bug,
        "small_object_allocation_failure": small_object_allocation_failure,
        "new_visible_birth_failure": new_visible_birth_failure,
        "decoder_failure": decoder_failure,
        "next_unique_action": next_action,
    }
    write_json(reports_dir / "sw3_query_support_resolution_decision.json", decision_payload)
    (reports_dir / "sw3_query_support_resolution_decision.md").write_text(
        "# SW-3 query support resolution decision\n\n"
        f"- case: `{case}`\n"
        f"- reason: `{case_reason}`\n"
        f"- next action: `{next_action}`\n",
        encoding="utf-8",
    )

    # Phase 10 final report
    stage_report_payload = {
        "executive_summary": {
            "effective_samples": len(effective_rows),
            "horizons": horizons,
            "refine_pts_support_valid": refine_pts_support_valid,
            "best_support_tensor": best_mapping["tensor_name"],
            "coordinate_mapping_bug": coordinate_mapping_bug,
            "attribution_summary": attribution_summary,
            "small_object_summary": small_summary,
            "new_visible_summary": nv_summary,
            "resolution_case": case,
            "next_unique_action": next_action,
        },
        "semantic_occ_dependency_trace": dep_trace,
        "support_tensor_candidates": candidate_manifest,
        "best_mapping_summary": best_mapping_summary,
        "query_to_pred_attribution": attribution_summary,
        "causal_summary": causal_summary,
        "small_object_summary": small_summary,
        "new_visible_summary": nv_summary,
        "resolution_decision": decision_payload,
        "safe_claims": {
            "subset_diagnostic_only": True,
            "official_benchmark": False,
            "query_allocation_confirmed": case in {"C", "D", "E"},
            "sensor_perturbation_completed": False,
        },
    }
    write_json(reports_dir / "stage_sw3_query_support_validation_report.json", stage_report_payload)
    report_md = [
        "# Stage SW-3 query support validation",
        "",
        "## Executive summary",
        "",
        f"- Effective samples: `{len(effective_rows)}`",
        f"- Horizons: `{horizons}`",
        f"- refine_pts support validity: `{refine_pts_support_valid}`",
        f"- best support tensor: `{best_mapping['tensor_name']}`",
        f"- coordinate mapping conclusion: `{best_mapping['coord_mode']} / {best_mapping['order_name']} / {best_mapping['flip_variant']} / {best_mapping['point_mode']}`",
        f"- query-to-pred attribution: `{attribution_summary['conclusion']}`",
        f"- small-object summary: `mean_small_ff_cov_r2={round(small_summary['mean_small_ff_coverage_r2'],4)}`",
        f"- new-visible summary: `new_cov={round(nv_summary['mean_new_visible_geometric_coverage'],4)}, persistent_cov={round(nv_summary['mean_persistent_geometric_coverage'],4)}`",
        f"- resolution case: `{case}`",
        "",
        "## Safe claims",
        "",
        "- This is a subset diagnostic.",
        "- Benchmark reproduction claim: no.",
        "- Query lifecycle tracking completeness claim: no.",
        "- Sensor perturbation completion claim: no.",
        "- No model improvement is claimed.",
        "",
        "## Next unique action",
        "",
        f"- {next_action}",
    ]
    (reports_dir / "stage_sw3_query_support_validation_report.md").write_text("\n".join(report_md), encoding="utf-8")

    print(
        json.dumps(
            {
                "effective_samples": len(effective_rows),
                "refine_pts_support_valid": refine_pts_support_valid,
                "best_support_tensor": best_mapping["tensor_name"],
                "resolution_case": case,
                "next_unique_action": next_action,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
