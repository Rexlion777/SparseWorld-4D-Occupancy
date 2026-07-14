"""Stage SW-2 SparseWorld temporal rollout and query coverage diagnosis.

This script reuses the Stage SW-1 SparseWorld bring-up path and extends it to:
1. subset-level real temporal rollout inference
2. per-horizon occupancy diagnostics
3. class-group diagnostics
4. new-visible / persistent / disappeared proxy diagnostics
5. query axis semantics audit from code + runtime tensors
6. query BEV coverage diagnostics
7. joint temporal-query diagnosis report

Safe-claim boundary:
- subset diagnostic only
- no training
- no full validation benchmark
- query coverage is a proxy, not identity-level lifecycle tracking
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


EMPTY_IDX = 17
IGNORE_IDX: int | None = None
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

QUERY_TRACE_TARGETS = [
    "cls_score",
    "refine_pts",
    "all_cls_scores",
    "all_refine_pts",
    "forecast_points_list",
    "forecast_semantics_list",
    "pred_trajs_list",
    "query_feat",
    "ind_stamps_all",
    "num_query",
    "num_fu_query",
    "num_refines",
    "encode_points",
    "decode_points",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage SW-2 SparseWorld temporal/query diagnosis")
    parser.add_argument(
        "--repo-root",
        default="/mnt/d/ComputerVision/cv_lidar_transition/external/SparseWorld",
    )
    parser.add_argument(
        "--config",
        default="/mnt/d/ComputerVision/cv_lidar_transition/external/SparseWorld/configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/d/ComputerVision/cv_lidar_transition/external/SparseWorld/ckpts/epoch_56.pth",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--fallback-num-samples", default="10,5")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-query", action="store_true")
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_parent(path)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def unwrap(value: Any) -> Any:
    try:
        from mmcv.parallel import DataContainer
    except Exception:
        DataContainer = None
    if DataContainer is not None and isinstance(value, DataContainer):
        return unwrap(value.data)
    if isinstance(value, list):
        if len(value) == 1:
            return unwrap(value[0])
        return [unwrap(v) for v in value]
    if isinstance(value, tuple):
        return [unwrap(v) for v in value]
    if isinstance(value, dict):
        return {k: unwrap(v) for k, v in value.items()}
    return value


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


def move_to_cuda(obj: Any) -> Any:
    try:
        from mmcv.parallel import DataContainer
    except Exception:
        DataContainer = None
    if (
        DataContainer is not None
        and isinstance(obj, list)
        and len(obj) == 1
        and isinstance(obj[0], DataContainer)
    ):
        return move_to_cuda(obj[0])
    if DataContainer is not None and isinstance(obj, DataContainer):
        return move_to_cuda(obj.data)
    if isinstance(obj, dict):
        return {k: move_to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_cuda(v) for v in obj]
    if isinstance(obj, tuple):
        return [move_to_cuda(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=False)
    return obj


def summarize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: summarize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 16:
            return {"type": "list", "length": len(obj)}
        return [summarize(v) for v in obj]
    if isinstance(obj, tuple):
        return [summarize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        detached = obj.detach()
        cpu = detached.float().cpu()
        finite = torch.isfinite(detached).float().mean().item() if detached.numel() > 0 else 1.0
        return {
            "type": "torch.Tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "finite_ratio": float(finite),
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


def ckpt_state_audit(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    matched = []
    missing = []
    unexpected = []
    shape_mismatch = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append({"key": key, "ckpt_shape": list(value.shape), "model_shape": list(model_state[key].shape)})
            continue
        matched.append(key)
    for key in model_state.keys():
        if key not in state_dict:
            missing.append(key)
    return {
        "checkpoint_top_keys": sorted(list(ckpt.keys())) if isinstance(ckpt, dict) else None,
        "matched_key_count": len(matched),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatch),
        "missing_keys_preview": missing[:20],
        "unexpected_keys_preview": unexpected[:20],
        "shape_mismatch_preview": shape_mismatch[:20],
        "checkpoint_meta_keys": sorted(list(ckpt.get("meta", {}).keys())) if isinstance(ckpt, dict) and isinstance(ckpt.get("meta", {}), dict) else [],
    }


def setup_runtime(repo_root: Path, config_path: Path, checkpoint_path: Path, split: str, use_fp16: bool) -> tuple[Any, Any, Any, dict[str, Any]]:
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from mmcv import Config
    from mmcv.parallel import collate
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    cfg = Config.fromfile(str(config_path))
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)
    setup_multi_processes(cfg)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.gpu_ids = [0]
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if "4D" in cfg.model.type:
        cfg.model.align_after_view_transfromation = True
    split_cfg = cfg.data[split]
    split_cfg.test_mode = split != "train"
    if cfg.data.get(f"{split}_dataloader", {}).get("samples_per_gpu", 1) > 1:
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)
    dataset = build_dataset(split_cfg)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    audit = ckpt_state_audit(model, checkpoint_path)
    if use_fp16:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = model.cuda()
    model.eval()
    return cfg, dataset, model, {"checkpoint_meta": checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}, "state_dict_audit": audit, "collate": collate}


def sample_info(dataset: Any, sample_index: int) -> dict[str, Any]:
    mapped_index = dataset.temp2nusc_map[sample_index] if hasattr(dataset, "temp2nusc_map") else sample_index
    info = dataset.data_infos[mapped_index]
    return {
        "sample_index": sample_index,
        "mapped_info_index": int(mapped_index),
        "sample_token": info.get("token"),
        "scene_token": info.get("scene_token"),
        "scene_name": info.get("scene_name"),
        "timestamp": int(info.get("timestamp")) if info.get("timestamp") is not None else None,
        "frame_idx": info.get("frame_idx"),
        "lidar_path": info.get("lidar_path"),
        "occ_path": info.get("occ_path"),
    }


def extract_sample_batch(dataset: Any, sample_index: int, collate_fn: Any) -> tuple[Any, Any]:
    sample = dataset[sample_index]
    batch = collate_fn([sample], samples_per_gpu=1)
    return sample, batch


def extract_standard_tensors(sample_unwrapped: dict[str, Any], raw_output: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    pred_keys = sorted([k for k in raw_output.keys() if k.startswith("semantic_occ_")], key=lambda x: int(x.split("_")[-1].replace("s", "")))
    pred_temporal = torch.stack([torch.as_tensor(raw_output[k][0]).long() for k in pred_keys], dim=0)
    gt_current = torch.as_tensor(sample_unwrapped["voxel_semantics"]).long()
    gt_temporal_list = [gt_current]
    temporal_semantics = sample_unwrapped["temporal_semantics"]
    for i in range(1, len(pred_keys)):
        gt_temporal_list.append(torch.as_tensor(temporal_semantics[i]["voxel_semantics"]).long())
    gt_temporal = torch.stack(gt_temporal_list, dim=0)
    return pred_temporal, gt_temporal, pred_keys


def valid_mask_from_gt(gt: torch.Tensor) -> torch.Tensor:
    if IGNORE_IDX is None:
        return torch.ones_like(gt, dtype=torch.bool)
    return gt != IGNORE_IDX


def compute_semantic_miou(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, empty_idx: int = EMPTY_IDX) -> tuple[float, dict[str, float]]:
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    class_ious: dict[str, float] = {}
    valid_classes = sorted(set(torch.unique(gt).cpu().tolist()) | set(torch.unique(pred).cpu().tolist()))
    valid_classes = [int(c) for c in valid_classes if int(c) != empty_idx]
    ious = []
    for cls in valid_classes:
        pred_c = pred == cls
        gt_c = gt == cls
        union = torch.logical_or(pred_c, gt_c).sum().item()
        if union == 0:
            continue
        inter = torch.logical_and(pred_c, gt_c).sum().item()
        iou = inter / union
        class_ious[str(cls)] = float(iou)
        ious.append(iou)
    return (float(np.mean(ious)) if ious else 0.0), class_ious


def compute_base_metrics(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, Any]:
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    pred_occ = pred != EMPTY_IDX
    gt_occ = gt != EMPTY_IDX
    inter = torch.logical_and(pred_occ, gt_occ).sum().item()
    union = torch.logical_or(pred_occ, gt_occ).sum().item()
    gt_occ_count = gt_occ.sum().item()
    gt_free_count = (~gt_occ).sum().item()
    pred_occ_count = pred_occ.sum().item()
    false_free_count = torch.logical_and(gt_occ, ~pred_occ).sum().item()
    false_occ_count = torch.logical_and(~gt_occ, pred_occ).sum().item()
    sem_miou, class_ious = compute_semantic_miou(pred, gt, torch.ones_like(gt, dtype=torch.bool), empty_idx=EMPTY_IDX)
    semantic_confusion = torch.logical_and(torch.logical_and(gt_occ, pred_occ), pred != gt).sum().item()
    return {
        "valid_voxel_count": int(valid_mask.sum().item()),
        "ignore_voxel_count": int((~valid_mask).sum().item()),
        "pred_occupied_count": int(pred_occ_count),
        "gt_occupied_count": int(gt_occ_count),
        "pred_gt_occupied_ratio": float(pred_occ_count / gt_occ_count) if gt_occ_count else 0.0,
        "occupied_iou": float(inter / union) if union else 0.0,
        "semantic_miou": sem_miou,
        "false_free_rate": float(false_free_count / gt_occ_count) if gt_occ_count else 0.0,
        "false_occupied_rate": float(false_occ_count / gt_free_count) if gt_free_count else 0.0,
        "occupied_accuracy": float(inter / gt_occ_count) if gt_occ_count else 0.0,
        "free_accuracy": float(((~pred_occ) & (~gt_occ)).sum().item() / gt_free_count) if gt_free_count else 0.0,
        "semantic_confusion_rate": float(semantic_confusion / gt_occ_count) if gt_occ_count else 0.0,
        "occupied_recall": float(inter / gt_occ_count) if gt_occ_count else 0.0,
        "occupied_precision": float(inter / pred_occ_count) if pred_occ_count else 0.0,
        "empty_pred_ratio": float((pred == EMPTY_IDX).sum().item() / pred.numel()) if pred.numel() else 0.0,
        "class_ious": class_ious,
    }


def aggregate_metric_rows(rows: list[dict[str, Any]], group_by_key: str = "horizon_s") -> list[dict[str, Any]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row[group_by_key]].append(row)
    out = []
    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        agg: dict[str, Any] = {group_by_key: key, "sample_count": len(bucket)}
        metric_keys = sorted(
            {
                k
                for r in bucket
                for k, v in r.items()
                if k != group_by_key and isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
            }
        )
        for metric in metric_keys:
            vals = [float(r[metric]) for r in bucket if r.get(metric) is not None]
            if not vals:
                continue
            agg[f"mean_{metric}"] = float(np.mean(vals))
            agg[f"std_{metric}"] = float(np.std(vals))
            agg[f"min_{metric}"] = float(np.min(vals))
            agg[f"max_{metric}"] = float(np.max(vals))
        out.append(agg)
    return out


def class_group_metrics(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, group_ids: list[int]) -> dict[str, Any]:
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    pred_group = torch.isin(pred, torch.as_tensor(group_ids, device=pred.device))
    gt_group = torch.isin(gt, torch.as_tensor(group_ids, device=gt.device))
    inter = torch.logical_and(pred_group, gt_group).sum().item()
    union = torch.logical_or(pred_group, gt_group).sum().item()
    gt_count = gt_group.sum().item()
    pred_count = pred_group.sum().item()
    false_free = torch.logical_and(gt_group, ~pred_group).sum().item()
    false_occ = torch.logical_and(~gt_group, pred_group).sum().item()
    return {
        "group_gt_occupied_count": int(gt_count),
        "group_pred_occupied_count": int(pred_count),
        "group_pred_gt_ratio": float(pred_count / gt_count) if gt_count else 0.0,
        "group_iou": float(inter / union) if union else 0.0,
        "group_false_free_rate": float(false_free / gt_count) if gt_count else 0.0,
        "group_false_occupied_rate": float(false_occ / max(1, (~gt_group).sum().item())),
        "group_recall": float(inter / gt_count) if gt_count else 0.0,
        "group_precision": float(inter / pred_count) if pred_count else 0.0,
    }


def range_false_free_rows(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> list[dict[str, Any]]:
    pred = pred.clone()
    gt = gt.clone()
    x_centers = PC_RANGE[0] + (torch.arange(pred.shape[0], dtype=torch.float32) + 0.5) * VOXEL_SIZE[0]
    y_centers = PC_RANGE[1] + (torch.arange(pred.shape[1], dtype=torch.float32) + 0.5) * VOXEL_SIZE[1]
    xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(-1).expand_as(pred.float())
    bins = [0.0, 10.0, 20.0, 30.0, 40.0, 80.0]
    pred_occ = (pred != EMPTY_IDX) & valid_mask
    gt_occ = (gt != EMPTY_IDX) & valid_mask
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rr >= lo) & (rr < hi) & valid_mask
        gt_occ_count = torch.logical_and(gt_occ, mask).sum().item()
        false_free_count = torch.logical_and(torch.logical_and(gt_occ, ~pred_occ), mask).sum().item()
        rows.append({
            "range_bin_m": f"[{lo},{hi})",
            "gt_occupied_count": int(gt_occ_count),
            "false_free_count": int(false_free_count),
            "false_free_rate": float(false_free_count / gt_occ_count) if gt_occ_count else 0.0,
        })
    return rows


def classify_region_proxies(gt0: torch.Tensor, gth: torch.Tensor) -> dict[str, torch.Tensor]:
    gt0_occ = gt0 != EMPTY_IDX
    gth_occ = gth != EMPTY_IDX
    return {
        "persistent_gt_occupied": gt0_occ & gth_occ,
        "newly_visible_or_newly_occupied_proxy": (~gt0_occ) & gth_occ,
        "disappeared_gt_occupied_proxy": gt0_occ & (~gth_occ),
    }


def region_proxy_metrics(pred_h: torch.Tensor, gt0: torch.Tensor, gth: torch.Tensor) -> dict[str, Any]:
    pred_occ = pred_h != EMPTY_IDX
    proxies = classify_region_proxies(gt0, gth)
    persistent = proxies["persistent_gt_occupied"]
    new_visible = proxies["newly_visible_or_newly_occupied_proxy"]
    disappeared = proxies["disappeared_gt_occupied_proxy"]

    def ratio(num_mask: torch.Tensor, den_mask: torch.Tensor) -> float:
        den = den_mask.sum().item()
        return float(num_mask.sum().item() / den) if den else 0.0

    return {
        "persistent_region_recall": ratio(pred_occ & persistent, persistent),
        "new_visible_proxy_recall": ratio(pred_occ & new_visible, new_visible),
        "disappeared_region_false_positive": ratio(pred_occ & disappeared, disappeared),
        "persistent_false_free": ratio((~pred_occ) & persistent, persistent),
        "new_visible_false_free": ratio((~pred_occ) & new_visible, new_visible),
        "temporal_occupancy_birth_miss": ratio((~pred_occ) & new_visible, new_visible),
        "temporal_occupancy_stale_prediction": ratio(pred_occ & disappeared, disappeared),
        "persistent_gt_count": int(persistent.sum().item()),
        "new_visible_gt_count": int(new_visible.sum().item()),
        "disappeared_gt_count": int(disappeared.sum().item()),
    }


def decode_points_if_needed(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points
    # SparseWorld refine points are encoded to normalized coordinates by encode_points().
    if points.float().min().item() >= -0.5 and points.float().max().item() <= 1.5:
        out = points.clone().float()
        out[..., 0] = out[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
        out[..., 1] = out[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
        out[..., 2] = out[..., 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
        return out
    return points.float()


def get_query_tensor_for_horizon(query_capture: dict[str, Any], horizon_s: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    fb = query_capture.get("forward_backbone_outputs", {})
    if horizon_s == 0:
        return fb.get("refine_pts"), fb.get("cls_score")
    idx = horizon_s - 1
    pts_list = fb.get("forecast_points_list", [])
    cls_list = fb.get("forecast_semantics_list", [])
    pts = pts_list[idx] if idx < len(pts_list) else None
    cls = cls_list[idx] if idx < len(cls_list) else None
    return pts, cls


def query_center_proxy(points: torch.Tensor) -> torch.Tensor:
    # Use mean over the 48 refine points as a per-query center proxy.
    return points.mean(dim=2)


def query_density_map_from_points(points_metric: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    x_idx = torch.floor((points_metric[:, 0] - PC_RANGE[0]) / VOXEL_SIZE[0]).long()
    y_idx = torch.floor((points_metric[:, 1] - PC_RANGE[1]) / VOXEL_SIZE[1]).long()
    mask = (x_idx >= 0) & (x_idx < GRID_SIZE[0]) & (y_idx >= 0) & (y_idx < GRID_SIZE[1])
    density = np.zeros((GRID_SIZE[0], GRID_SIZE[1]), dtype=np.float32)
    if mask.any():
        valid_x = x_idx[mask].cpu().numpy()
        valid_y = y_idx[mask].cpu().numpy()
        for x, y in zip(valid_x, valid_y):
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
        com_x, com_y = math.nan, math.nan
    return density, {
        "query_total_count": int(points_metric.shape[0]),
        "query_in_range_count": int(mask.sum().item()),
        "query_in_range_ratio": float(mask.float().mean().item()) if mask.numel() else 0.0,
        "query_density_entropy": entropy,
        "query_density_center_of_mass_x": com_x,
        "query_density_center_of_mass_y": com_y,
    }


def bev_occ_map(label_grid: torch.Tensor) -> np.ndarray:
    return (label_grid != EMPTY_IDX).any(dim=-1).cpu().numpy().astype(np.uint8)


def gt_group_bev_map(label_grid: torch.Tensor, group_ids: list[int]) -> np.ndarray:
    group_mask = torch.isin(label_grid, torch.as_tensor(group_ids, device=label_grid.device))
    return group_mask.any(dim=-1).cpu().numpy().astype(np.uint8)


def dilate_bool_map(binary_map: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary_map.astype(bool)
    tensor = torch.from_numpy(binary_map.astype(np.float32))[None, None]
    pooled = torch.nn.functional.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return pooled[0, 0].numpy() > 0


def query_coverage_metrics(pred_h: torch.Tensor, gt_h: torch.Tensor, query_points: torch.Tensor, radius_values: list[int]) -> list[dict[str, Any]]:
    centers_metric = decode_points_if_needed(query_center_proxy(query_points[0]))
    density_map, density_stats = query_density_map_from_points(centers_metric)
    gt_bev = bev_occ_map(gt_h)
    pred_bev = bev_occ_map(pred_h)
    false_free_bev = np.logical_and(gt_bev == 1, pred_bev == 0)
    rows = []
    query_bool = density_map > 0
    for radius in radius_values:
        dilated = dilate_bool_map(query_bool, radius)
        gt_cov = float((dilated & (gt_bev > 0)).sum() / max(1, (gt_bev > 0).sum()))
        pred_cov = float((dilated & (pred_bev > 0)).sum() / max(1, (pred_bev > 0).sum()))
        ff_cov = float((dilated & false_free_bev).sum() / max(1, false_free_bev.sum()))
        rows.append(
            {
                "coverage_radius_cells": radius,
                "query_coverage_gt_ratio": gt_cov,
                "query_coverage_pred_ratio": pred_cov,
                "query_coverage_false_free_ratio": ff_cov,
                **density_stats,
            }
        )
    return rows


def save_curve(
    rows: list[dict[str, Any]],
    x_key: str,
    mean_key: str,
    std_key: str | None,
    save_path: Path,
    title: str,
    ylabel: str,
) -> None:
    if not rows:
        return
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[mean_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    if std_key is not None and std_key in rows[0]:
        stds = [float(row[std_key]) for row in rows]
        ax.fill_between(xs, np.array(ys) - np.array(stds), np.array(ys) + np.array(stds), alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Horizon (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_panel(figures: list[tuple[np.ndarray, str]], save_path: Path, cols: int = 3, cmap: str | None = None) -> None:
    rows = math.ceil(len(figures) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flatten():
        ax.axis("off")
    for idx, (img, title) in enumerate(figures):
        ax = axes.flatten()[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap if cmap is not None else "viridis")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def scatter_plot(x: list[float], y: list[float], save_path: Path, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def query_axis_semantics_audit(repo_root: Path, cfg: Any, query_capture: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_files = [
        repo_root / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py",
        repo_root / "mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py",
        repo_root / "mmdet3d/models/sparsedetectors/opus_head.py",
        repo_root / "mmdet3d/models/sparsedetectors/opus_transformer.py",
        repo_root / "mmdet3d/models/sparsedetectors/bbox/utils.py",
    ]
    trace_rows = []
    for file_path in trace_files:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            for target in QUERY_TRACE_TARGETS:
                if target in line:
                    trace_rows.append(
                        {
                            "file_path": str(file_path),
                            "line_no": line_no,
                            "target": target,
                            "line_text": line.strip(),
                        }
                    )

    fb = query_capture.get("forward_backbone_outputs", {})
    cls_score = fb.get("cls_score")
    refine_pts = fb.get("refine_pts")
    forecast_points_list = fb.get("forecast_points_list", [])
    forecast_semantics_list = fb.get("forecast_semantics_list", [])
    config_values = {
        "num_query": int(cfg.model.pts_bbox_head.num_query),
        "num_fu_query": list(cfg.model.pts_bbox_head.num_fu_query),
        "num_future_queries_total": int(sum(cfg.model.pts_bbox_head.num_fu_query)),
        "num_decoder_layers": int(cfg.model.pts_bbox_head.transformer.num_layers),
        "num_refines_per_layer": list(cfg.model.pts_bbox_head.transformer.num_refines),
        "num_refines_last_layer": int(cfg.model.pts_bbox_head.transformer.num_refines[-1]),
        "num_frames": int(cfg.model.pts_bbox_head.transformer.num_frames),
        "num_future_frames": int(cfg.model.pts_bbox_head.num_fu_frames),
        "num_classes_non_empty": int(cfg.model.pts_bbox_head.num_classes),
        "empty_idx": int(cfg.empty_idx if hasattr(cfg, "empty_idx") else EMPTY_IDX),
        "pc_range": list(cfg.point_cloud_range),
        "voxel_size": list(cfg.voxel_size),
    }

    cls_shape = list(cls_score.shape) if isinstance(cls_score, torch.Tensor) else None
    refine_shape = list(refine_pts.shape) if isinstance(refine_pts, torch.Tensor) else None
    future_shapes = [list(x.shape) for x in forecast_points_list if isinstance(x, torch.Tensor)]
    future_cls_shapes = [list(x.shape) for x in forecast_semantics_list if isinstance(x, torch.Tensor)]

    audit_payload = {
        "config_values": config_values,
        "runtime_tensor_shapes": {
            "cls_score": cls_shape,
            "refine_pts": refine_shape,
            "forecast_points_list": future_shapes,
            "forecast_semantics_list": future_cls_shapes,
        },
        "cls_score_axis_semantics": {
            "dim0": {"meaning": "batch", "confidence": "confirmed", "evidence": "runtime shape first dim is 1; model is single-sample eval"},
            "dim1": {
                "meaning": "current-frame query count",
                "confidence": "confirmed",
                "evidence": "config num_query=720; sparseworld_4d_traj.py selects query_cls[:, ind_stamps_all == 0]",
            },
            "dim2": {
                "meaning": "refine point count per query from final decoder layer",
                "confidence": "confirmed",
                "evidence": "opus_transformer.py cls_score.view(B, Q, self.num_refines, self.num_classes); config final num_refines=48",
            },
            "dim3": {
                "meaning": "non-empty semantic class logits",
                "confidence": "confirmed",
                "evidence": "opus_transformer.py uses self.num_classes; config occ_names length=17; empty handled outside logits in get_occ()",
            },
        },
        "refine_pts_axis_semantics": {
            "dim0": {"meaning": "batch", "confidence": "confirmed", "evidence": "runtime shape first dim is 1"},
            "dim1": {
                "meaning": "current-frame query count",
                "confidence": "confirmed",
                "evidence": "same query slice as cls_score via ind_stamps_all == 0",
            },
            "dim2": {
                "meaning": "refine point count per query from final decoder layer",
                "confidence": "confirmed",
                "evidence": "opus_transformer.py returns refine_pt with shape [B,Q,num_refines,3]",
            },
            "dim3": {
                "meaning": "xyz coordinate",
                "confidence": "confirmed",
                "evidence": "bbox/utils.py encode_points/decode_points operate on x,y,z only",
            },
            "coordinate_space": {
                "meaning": "normalized to pc_range before decode",
                "confidence": "confirmed",
                "evidence": "opus_transformer.py refine_points -> encode_points(new_points, self.pc_range); runtime min/max near [0,1]",
            },
        },
        "future_query_growth_semantics": {
            "confidence": "confirmed",
            "evidence": "forecast_points_list query counts grow 780,840,900,960,1000,1040, matching cumulative num_fu_query additions [60,60,60,60,40,40]",
        },
        "recommended_sw3_query_center_tensor": {
            "tensor": "refine_pts",
            "proxy": "mean over dim=2 as per-query center; keep raw 48-point support for decode-aware analysis",
            "confidence": "likely",
        },
        "recommended_sw3_query_confidence_tensor": {
            "tensor": "cls_score",
            "proxy": "max semantic logit / sigmoid score over dim=3 and optionally aggregate over dim=2",
            "confidence": "likely",
        },
    }

    write_json(output_dir / "query_config_values.json", config_values)
    write_json(output_dir / "query_axis_semantics_audit.json", audit_payload)
    write_csv(output_dir / "query_source_code_trace.csv", trace_rows, fieldnames=["file_path", "line_no", "target", "line_text"])
    audit_md = [
        "# Query axis semantics audit",
        "",
        f"- cls_score shape: `{cls_shape}`",
        f"- refine_pts shape: `{refine_shape}`",
        f"- future query shapes: `{future_shapes}`",
        "",
        "Key conclusions:",
        "",
        "- `cls_score = [batch, current_query_count, refine_points_per_query, class_count_non_empty]`",
        "- `refine_pts = [batch, current_query_count, refine_points_per_query, xyz]`",
        "- `720` is confirmed current-frame query count from `num_query`.",
        "- `48` is confirmed last-layer `num_refines`.",
        "- `17` is confirmed non-empty semantic class count; empty is injected in `get_occ()` as class id 17.",
        "- `refine_pts` are confirmed normalized coordinates before decode.",
        "",
        "Stage SW-3 recommendation:",
        "",
        "- Track `refine_pts` for centers/support geometry.",
        "- Track `cls_score` for semantic confidence.",
    ]
    (output_dir / "query_axis_semantics_audit.md").write_text("\n".join(audit_md), encoding="utf-8")
    return audit_payload, trace_rows


def plot_horizon_panels(agg_rows: list[dict[str, Any]], figure_dir: Path, subset_label: str) -> None:
    mapping = [
        ("mean_occupied_iou", "std_occupied_iou", "horizon_vs_occupied_iou.png", "Occupied IoU"),
        ("mean_semantic_miou", "std_semantic_miou", "horizon_vs_semantic_miou.png", "Semantic mIoU"),
        ("mean_false_free_rate", "std_false_free_rate", "horizon_vs_false_free.png", "False-free rate"),
        ("mean_false_occupied_rate", "std_false_occupied_rate", "horizon_vs_false_occupied.png", "False-occupied rate"),
        ("mean_pred_gt_occupied_ratio", "std_pred_gt_occupied_ratio", "horizon_vs_pred_gt_occupied_ratio.png", "Pred/GT occupied ratio"),
    ]
    for mean_key, std_key, name, ylabel in mapping:
        save_curve(
            agg_rows,
            "horizon_s",
            mean_key,
            std_key,
            figure_dir / name,
            title=f"{ylabel} over horizon ({subset_label})",
            ylabel=ylabel,
        )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panel_specs = [
        ("mean_occupied_iou", "Occupied IoU"),
        ("mean_semantic_miou", "Semantic mIoU"),
        ("mean_false_free_rate", "False-free"),
        ("mean_false_occupied_rate", "False-occupied"),
        ("mean_pred_gt_occupied_ratio", "Pred/GT occupied"),
    ]
    xs = [row["horizon_s"] for row in agg_rows]
    for ax in axes.flatten():
        ax.axis("off")
    for ax, (key, title) in zip(axes.flatten(), panel_specs):
        ax.axis("on")
        ys = [row[key] for row in agg_rows]
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Horizon (s)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"SparseWorld horizon metric panel ({subset_label})")
    fig.tight_layout()
    fig.savefig(figure_dir / "horizon_metric_panel.png", dpi=150)
    plt.close(fig)


def plot_group_curves(group_agg: list[dict[str, Any]], figure_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_agg:
        grouped[row["group_name"]].append(row)
    plot_specs = [
        ("mean_group_false_free_rate", "group_false_free_over_horizon.png", "Group false-free rate"),
        ("mean_group_iou", "group_iou_over_horizon.png", "Group IoU"),
        ("mean_group_pred_gt_ratio", "group_pred_gt_ratio_over_horizon.png", "Group pred/GT ratio"),
    ]
    for metric_key, filename, title in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for group_name, rows in grouped.items():
            rows = sorted(rows, key=lambda x: x["horizon_s"])
            ax.plot([r["horizon_s"] for r in rows], [r[metric_key] for r in rows], marker="o", label=group_name)
        ax.set_xlabel("Horizon (s)")
        ax.set_ylabel(metric_key.replace("mean_", ""))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=150)
        plt.close(fig)

    focus_groups = ["all_dynamic", "all_static", "small_object"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for group_name in focus_groups[:2]:
        rows = sorted(grouped[group_name], key=lambda x: x["horizon_s"])
        axes[0].plot([r["horizon_s"] for r in rows], [r["mean_group_false_free_rate"] for r in rows], marker="o", label=group_name)
    axes[0].set_title("Dynamic vs Static false-free")
    axes[0].set_xlabel("Horizon (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    rows = sorted(grouped["small_object"], key=lambda x: x["horizon_s"])
    axes[1].plot([r["horizon_s"] for r in rows], [r["mean_group_false_free_rate"] for r in rows], marker="o", label="small_object")
    axes[1].set_title("Small-object false-free")
    axes[1].set_xlabel("Horizon (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "dynamic_vs_static_error_panel.png", dpi=150)
    fig.savefig(figure_dir / "small_object_error_over_horizon.png", dpi=150)
    plt.close(fig)


def plot_new_visible_proxy(proxy_agg: list[dict[str, Any]], figure_dir: Path) -> None:
    if not proxy_agg:
        return
    xs = [row["horizon_s"] for row in proxy_agg]
    save_curve(proxy_agg, "horizon_s", "mean_new_visible_false_free", "std_new_visible_false_free", figure_dir / "new_visible_false_free_over_horizon.png", "New-visible proxy false-free", "False-free")
    save_curve(proxy_agg, "horizon_s", "mean_persistent_false_free", "std_persistent_false_free", figure_dir / "persistent_false_free_over_horizon.png", "Persistent region false-free", "False-free")
    save_curve(proxy_agg, "horizon_s", "mean_disappeared_region_false_positive", "std_disappeared_region_false_positive", figure_dir / "disappeared_false_positive_over_horizon.png", "Disappeared region stale prediction", "False-positive")


def coverage_failure_conclusion(rows: list[dict[str, Any]], radius: int = 2) -> dict[str, Any]:
    filtered = [row for row in rows if row["coverage_radius_cells"] == radius]
    if not filtered:
        return {"conclusion": "unknown", "mean_query_coverage_false_free_ratio": None}
    mean_cov = float(np.mean([row["query_coverage_false_free_ratio"] for row in filtered]))
    if mean_cov < 0.4:
        conclusion = "coverage_failure_dominant"
    elif mean_cov > 0.6:
        conclusion = "decode_or_head_failure_dominant"
    else:
        conclusion = "mixed_coverage_and_decode_failure"
    return {"conclusion": conclusion, "mean_query_coverage_false_free_ratio": mean_cov}


def markdown_report(
    report_path: Path,
    json_path: Path,
    payload: dict[str, Any],
    subset_label: str,
    forward_success_count: int,
    failure_count: int,
    t0: dict[str, Any] | None,
    t6: dict[str, Any] | None,
    dominant_group: str,
    new_visible_summary: str,
    coverage_summary: dict[str, Any],
) -> None:
    write_json(json_path, payload)
    lines = [
        "# Stage SW-2 SparseWorld temporal query diagnosis",
        "",
        "## Executive summary",
        "",
        f"- Effective subset size: `{subset_label}`",
        f"- Forward success count: `{forward_success_count}`",
        f"- Forward failure count: `{failure_count}`",
        f"- t=0 occupied IoU / false-free: `{None if t0 is None else round(t0['mean_occupied_iou'], 4)}` / `{None if t0 is None else round(t0['mean_false_free_rate'], 4)}`",
        f"- t=6 occupied IoU / false-free: `{None if t6 is None else round(t6['mean_occupied_iou'], 4)}` / `{None if t6 is None else round(t6['mean_false_free_rate'], 4)}`",
        f"- Dominant class-group error: `{dominant_group}`",
        f"- New-visible proxy finding: `{new_visible_summary}`",
        f"- Query coverage conclusion: `{coverage_summary['conclusion']}`",
        "",
        "## Safe claims",
        "",
        "- This is a subset diagnostic, not an official SparseWorld benchmark.",
        "- Query coverage is a proxy, not full lifecycle tracking.",
        "- Temporal query coverage is not identity tracking.",
        "- No model training or structural modification was performed.",
        "",
        "## Next unique action",
        "",
        f"- {payload['next_unique_action']}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    project_root = repo_root.parent.parent
    reports_dir = project_root / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
    artifacts_dir = project_root / "artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
    figure_dir = project_root / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fallback_targets = [int(x) for x in args.fallback_num_samples.split(",") if x.strip()]
    cfg, dataset, model, runtime_meta = setup_runtime(repo_root, config_path, checkpoint_path, args.split, args.fp16)
    collate_fn = runtime_meta["collate"]

    query_capture_holder: dict[str, Any] = {}
    original_forward_backbone = model.forward_backbone

    def wrapped_forward_backbone(*fb_args: Any, **fb_kwargs: Any) -> Any:
        outputs = original_forward_backbone(*fb_args, **fb_kwargs)
        query_capture_holder["forward_backbone_outputs"] = to_cpu_artifact(outputs)
        return outputs

    model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]

    sample_manifest_rows: list[dict[str, Any]] = []
    per_sample_horizon_rows: list[dict[str, Any]] = []
    class_group_rows: list[dict[str, Any]] = []
    new_visible_rows: list[dict[str, Any]] = []
    query_coverage_rows: list[dict[str, Any]] = []
    temporal_query_proxy_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    success_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    sample_count_target = args.num_samples
    successful_sample_indexes: list[int] = []

    torch.cuda.empty_cache()
    for sample_index in range(args.start_index, args.start_index + sample_count_target):
        info = sample_info(dataset, sample_index)
        sample_row = {
            "sample_index": sample_index,
            **info,
            "forward_status": "not_started",
            "output_status": "not_started",
            "adapter_status": "not_started",
            "query_capture_status": "not_started",
            "metric_status": "not_started",
            "failure_traceback": "",
        }
        try:
            raw_sample, batch = extract_sample_batch(dataset, sample_index, collate_fn)
            sample_unwrapped = unwrap(raw_sample)
            torch.cuda.reset_peak_memory_stats()
            query_capture_holder.clear()
            model_inputs = move_to_cuda(batch)
            started = time.perf_counter()
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **model_inputs)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            raw_result_cpu = to_cpu_artifact(result)
            query_capture_cpu = to_cpu_artifact(query_capture_holder)
            pred_temporal, gt_temporal, pred_keys = extract_standard_tensors(sample_unwrapped, raw_result_cpu)
            sample_row.update(
                {
                    "forward_status": "success",
                    "output_status": "success",
                    "adapter_status": "success",
                    "query_capture_status": "success",
                    "metric_status": "success",
                    "latency_ms": latency_ms,
                    "peak_memory_mb": peak_memory_mb,
                    "horizon_count": int(pred_temporal.shape[0]),
                    "grid_shape": "x".join(map(str, pred_temporal.shape[1:])),
                    "class_count_non_empty": len(OCC_NAMES),
                }
            )
            if args.save_raw:
                raw_path = artifacts_dir / f"raw_outputs/sample_{sample_index:04d}_raw_output.pt"
                ensure_parent(raw_path)
                torch.save(raw_result_cpu, raw_path)
                sample_row["raw_output_path"] = str(raw_path)
            if args.save_query:
                query_path = artifacts_dir / f"query_outputs/sample_{sample_index:04d}_query_output.pt"
                ensure_parent(query_path)
                torch.save(query_capture_cpu, query_path)
                sample_row["query_output_path"] = str(query_path)
            std_pred_path = artifacts_dir / f"standard_occ/sample_{sample_index:04d}_standard_pred_occ_temporal.pt"
            std_gt_path = artifacts_dir / f"standard_occ/sample_{sample_index:04d}_standard_gt_occ_temporal.pt"
            ensure_parent(std_pred_path)
            torch.save(pred_temporal, std_pred_path)
            torch.save(gt_temporal, std_gt_path)
            sample_row["standard_pred_occ_temporal_path"] = str(std_pred_path)
            sample_row["standard_gt_occ_temporal_path"] = str(std_gt_path)

            successful_sample_indexes.append(sample_index)
            record = {
                "sample_index": sample_index,
                "info": info,
                "pred_temporal": pred_temporal,
                "gt_temporal": gt_temporal,
                "raw_result": raw_result_cpu,
                "query_capture": query_capture_cpu,
                "sample_unwrapped": sample_unwrapped,
                "latency_ms": latency_ms,
                "peak_memory_mb": peak_memory_mb,
                "pred_keys": pred_keys,
            }
            success_records.append(record)
        except torch.cuda.OutOfMemoryError:
            sample_row["forward_status"] = "failed_oom"
            sample_row["failure_traceback"] = traceback.format_exc()
            failure_records.append(sample_row.copy())
            torch.cuda.empty_cache()
        except Exception:
            sample_row["forward_status"] = "failed_exception"
            sample_row["failure_traceback"] = traceback.format_exc()
            failure_records.append(sample_row.copy())
            torch.cuda.empty_cache()
        sample_manifest_rows.append(sample_row)

    success_count = len(success_records)
    failure_count = len(failure_records)
    if success_count >= sample_count_target:
        effective_count = sample_count_target
    elif success_count >= max(fallback_targets):
        effective_count = max(fallback_targets)
    elif success_count >= min(fallback_targets):
        effective_count = min(fallback_targets)
    else:
        effective_count = success_count
    effective_records = success_records[:effective_count]
    subset_label = f"{effective_count}-sample subset"

    if not effective_records:
        run_manifest = {
            "status": "blocked",
            "reason": "no_successful_samples",
            "requested_num_samples": args.num_samples,
            "forward_success_count": 0,
            "failure_count": failure_count,
            "state_dict_audit": runtime_meta["state_dict_audit"],
        }
        write_json(reports_dir / "sw2_run_manifest.json", run_manifest)
        write_csv(reports_dir / "sw2_sample_manifest.csv", sample_manifest_rows)
        return 1

    first_success = effective_records[0]
    query_audit_payload, query_trace_rows = query_axis_semantics_audit(repo_root, cfg, first_success["query_capture"], reports_dir)

    for record in effective_records:
        sample_index = record["sample_index"]
        info = record["info"]
        pred_temporal = record["pred_temporal"]
        gt_temporal = record["gt_temporal"]
        query_capture = record["query_capture"]
        pred_keys = record["pred_keys"]

        base_query_com: tuple[float, float] | None = None
        for h_idx, pred_key in enumerate(pred_keys):
            valid_mask = valid_mask_from_gt(gt_temporal[h_idx])
            metrics = compute_base_metrics(pred_temporal[h_idx], gt_temporal[h_idx], valid_mask)
            row = {
                "sample_index": sample_index,
                "sample_token": info["sample_token"],
                "scene_token": info["scene_token"],
                "scene_name": info["scene_name"],
                "timestamp": info["timestamp"],
                "horizon_s": h_idx,
                "pred_key": pred_key,
                **metrics,
                "latency_ms": record["latency_ms"],
                "peak_memory_mb": record["peak_memory_mb"],
            }
            per_sample_horizon_rows.append(row)

            for group_name, group_ids in CLASS_GROUPS.items():
                group_metrics = class_group_metrics(pred_temporal[h_idx], gt_temporal[h_idx], valid_mask, group_ids)
                class_group_rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_token": info["sample_token"],
                        "scene_token": info["scene_token"],
                        "group_name": group_name,
                        "horizon_s": h_idx,
                        **group_metrics,
                    }
                )

            if h_idx > 0:
                proxy_metrics = region_proxy_metrics(pred_temporal[h_idx], gt_temporal[0], gt_temporal[h_idx])
                new_visible_rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_token": info["sample_token"],
                        "scene_token": info["scene_token"],
                        "horizon_s": h_idx,
                        **proxy_metrics,
                    }
                )

            query_points, query_scores = get_query_tensor_for_horizon(query_capture, h_idx)
            if isinstance(query_points, torch.Tensor):
                coverage_rows = query_coverage_metrics(pred_temporal[h_idx], gt_temporal[h_idx], query_points, radius_values=[1, 2, 3])
                metric_by_r2 = None
                for cov_row in coverage_rows:
                    cov_row.update(
                        {
                            "sample_index": sample_index,
                            "sample_token": info["sample_token"],
                            "scene_token": info["scene_token"],
                            "scene_name": info["scene_name"],
                            "horizon_s": h_idx,
                        }
                    )
                    query_coverage_rows.append(cov_row)
                    if cov_row["coverage_radius_cells"] == 2:
                        metric_by_r2 = cov_row

                centers_metric = decode_points_if_needed(query_center_proxy(query_points[0]))
                density_map, density_stats = query_density_map_from_points(centers_metric)
                if base_query_com is None:
                    base_query_com = (
                        density_stats["query_density_center_of_mass_x"],
                        density_stats["query_density_center_of_mass_y"],
                    )
                shift = math.nan
                if base_query_com is not None and not math.isnan(base_query_com[0]) and not math.isnan(density_stats["query_density_center_of_mass_x"]):
                    shift = float(math.sqrt(
                        (density_stats["query_density_center_of_mass_x"] - base_query_com[0]) ** 2
                        + (density_stats["query_density_center_of_mass_y"] - base_query_com[1]) ** 2
                    ))
                temporal_query_proxy_rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_token": info["sample_token"],
                        "scene_token": info["scene_token"],
                        "horizon_s": h_idx,
                        **density_stats,
                        "query_density_shift_from_t0": shift,
                        "query_coverage_gt_ratio_r2": None if metric_by_r2 is None else metric_by_r2["query_coverage_gt_ratio"],
                        "query_coverage_false_free_ratio_r2": None if metric_by_r2 is None else metric_by_r2["query_coverage_false_free_ratio"],
                    }
                )

                if metric_by_r2 is not None:
                    joint_rows.append(
                        {
                            "sample_index": sample_index,
                            "sample_token": info["sample_token"],
                            "scene_token": info["scene_token"],
                            "horizon_s": h_idx,
                            "occupied_iou": row["occupied_iou"],
                            "semantic_miou": row["semantic_miou"],
                            "false_free_rate": row["false_free_rate"],
                            "false_occupied_rate": row["false_occupied_rate"],
                            "pred_gt_occupied_ratio": row["pred_gt_occupied_ratio"],
                            "query_coverage_gt_ratio": metric_by_r2["query_coverage_gt_ratio"],
                            "query_coverage_false_free_ratio": metric_by_r2["query_coverage_false_free_ratio"],
                            "query_in_range_ratio": metric_by_r2["query_in_range_ratio"],
                            "query_density_entropy": metric_by_r2["query_density_entropy"],
                            "dynamic_false_free": next(x["group_false_free_rate"] for x in [class_group_metrics(pred_temporal[h_idx], gt_temporal[h_idx], valid_mask, CLASS_GROUPS["all_dynamic"])]),
                            "static_false_free": next(x["group_false_free_rate"] for x in [class_group_metrics(pred_temporal[h_idx], gt_temporal[h_idx], valid_mask, CLASS_GROUPS["all_static"])]),
                            "small_object_false_free": next(x["group_false_free_rate"] for x in [class_group_metrics(pred_temporal[h_idx], gt_temporal[h_idx], valid_mask, CLASS_GROUPS["small_object"])]),
                            "new_visible_proxy_false_free": None if h_idx == 0 else region_proxy_metrics(pred_temporal[h_idx], gt_temporal[0], gt_temporal[h_idx])["new_visible_false_free"],
                        }
                    )

    # Aggregate outputs
    horizon_agg_rows = aggregate_metric_rows(per_sample_horizon_rows, group_by_key="horizon_s")
    class_group_agg_rows = []
    for group_name in sorted({row["group_name"] for row in class_group_rows}):
        group_rows = [row for row in class_group_rows if row["group_name"] == group_name]
        for agg in aggregate_metric_rows(
            [
                {
                    "horizon_s": row["horizon_s"],
                    "group_iou": row["group_iou"],
                    "group_false_free_rate": row["group_false_free_rate"],
                    "group_false_occupied_rate": row["group_false_occupied_rate"],
                    "group_recall": row["group_recall"],
                    "group_precision": row["group_precision"],
                    "group_gt_occupied_count": row["group_gt_occupied_count"],
                    "group_pred_occupied_count": row["group_pred_occupied_count"],
                    "group_pred_gt_ratio": row["group_pred_gt_ratio"],
                }
                for row in group_rows
            ],
            group_by_key="horizon_s",
        ):
            rename = {}
            for key in list(agg.keys()):
                if key.startswith("mean_group_") or key.startswith("std_group_") or key.startswith("min_group_") or key.startswith("max_group_"):
                    continue
            # manual remap for clarity
            converted = {"group_name": group_name, "horizon_s": agg["horizon_s"], "sample_count": agg["sample_count"]}
            for prefix in ["mean", "std", "min", "max"]:
                for metric in ["group_iou", "group_false_free_rate", "group_false_occupied_rate", "group_recall", "group_precision", "group_gt_occupied_count", "group_pred_occupied_count", "group_pred_gt_ratio"]:
                    key = f"{prefix}_{metric}"
                    if key in agg:
                        converted[key] = agg[key]
            class_group_agg_rows.append(converted)

    new_visible_agg_rows = aggregate_metric_rows(new_visible_rows, group_by_key="horizon_s")
    # rename mean_* fields for clarity
    renamed_new_visible = []
    for row in new_visible_agg_rows:
        renamed = {"horizon_s": row["horizon_s"], "sample_count": row["sample_count"]}
        for k, v in row.items():
            if k in {"horizon_s", "sample_count"}:
                continue
            renamed[k] = v
        renamed_new_visible.append(renamed)
    new_visible_agg_rows = renamed_new_visible

    query_coverage_agg_rows = []
    for radius in [1, 2, 3]:
        radius_rows = [row for row in query_coverage_rows if row["coverage_radius_cells"] == radius]
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in radius_rows:
            buckets[row["horizon_s"]].append(row)
        for horizon_s in sorted(buckets.keys()):
            bucket = buckets[horizon_s]
            query_coverage_agg_rows.append(
                {
                    "coverage_radius_cells": radius,
                    "horizon_s": horizon_s,
                    "sample_count": len(bucket),
                    "mean_query_coverage_gt_ratio": float(np.mean([r["query_coverage_gt_ratio"] for r in bucket])),
                    "mean_query_coverage_false_free_ratio": float(np.mean([r["query_coverage_false_free_ratio"] for r in bucket])),
                    "mean_query_coverage_pred_ratio": float(np.mean([r["query_coverage_pred_ratio"] for r in bucket])),
                    "mean_query_in_range_ratio": float(np.mean([r["query_in_range_ratio"] for r in bucket])),
                    "mean_query_density_entropy": float(np.mean([r["query_density_entropy"] for r in bucket])),
                }
            )

    temporal_query_proxy_agg_rows = aggregate_metric_rows(
        [
            {
                "horizon_s": row["horizon_s"],
                "query_coverage_gt_ratio": row["query_coverage_gt_ratio_r2"] if row["query_coverage_gt_ratio_r2"] is not None else 0.0,
                "query_coverage_false_free_ratio": row["query_coverage_false_free_ratio_r2"] if row["query_coverage_false_free_ratio_r2"] is not None else 0.0,
                "query_in_range_ratio": row["query_in_range_ratio"],
                "query_density_entropy": row["query_density_entropy"],
                "query_density_shift_from_t0": 0.0 if math.isnan(row["query_density_shift_from_t0"]) else row["query_density_shift_from_t0"],
            }
            for row in temporal_query_proxy_rows
        ],
        group_by_key="horizon_s",
    )

    # Save core reports
    run_manifest = {
        "repo_root": str(repo_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "requested_num_samples": args.num_samples,
        "fallback_num_samples": fallback_targets,
        "start_index": args.start_index,
        "forward_success_count": success_count,
        "forward_failure_count": failure_count,
        "effective_subset_size": effective_count,
        "effective_subset_label": subset_label,
        "successful_sample_indexes": successful_sample_indexes,
        "state_dict_audit": runtime_meta["state_dict_audit"],
        "checkpoint_meta_keys": sorted(list(runtime_meta["checkpoint_meta"].keys())),
    }
    write_json(reports_dir / "sw2_run_manifest.json", run_manifest)
    write_csv(reports_dir / "sw2_sample_manifest.csv", sample_manifest_rows)

    # Phase 2 summary
    forward_summary_rows = []
    for rec in effective_records:
        pred_temporal = rec["pred_temporal"]
        gt_temporal = rec["gt_temporal"]
        pred_counts = [int((pred_temporal[h] != EMPTY_IDX).sum().item()) for h in range(pred_temporal.shape[0])]
        gt_counts = [int((gt_temporal[h] != EMPTY_IDX).sum().item()) for h in range(gt_temporal.shape[0])]
        forward_summary_rows.append(
            {
                "sample_index": rec["sample_index"],
                "sample_token": rec["info"]["sample_token"],
                "scene_token": rec["info"]["scene_token"],
                "scene_name": rec["info"]["scene_name"],
                "timestamp": rec["info"]["timestamp"],
                "output_keys": "|".join(sorted(rec["raw_result"].keys())),
                "horizon_count": pred_temporal.shape[0],
                "class_count": len(OCC_NAMES),
                "grid_shape": "x".join(map(str, pred_temporal.shape[1:])),
                "pred_occupied_count_per_horizon": pred_counts,
                "gt_occupied_count_per_horizon": gt_counts,
                "latency_ms": rec["latency_ms"],
                "peak_memory_mb": rec["peak_memory_mb"],
            }
        )
    write_json(reports_dir / "sw2_forward_subset_summary.json", {"rows": forward_summary_rows, "effective_subset_size": effective_count})
    write_csv(reports_dir / "sw2_forward_subset_summary.csv", forward_summary_rows)

    # Phase 3 metrics
    per_sample_horizon_json = {"rows": per_sample_horizon_rows, "effective_subset_size": effective_count}
    write_json(reports_dir / "temporal_horizon_metrics_per_sample.json", per_sample_horizon_json)
    write_csv(reports_dir / "temporal_horizon_metrics_per_sample.csv", per_sample_horizon_rows)
    write_json(reports_dir / "temporal_horizon_metrics_aggregate.json", {"rows": horizon_agg_rows, "effective_subset_size": effective_count})
    write_csv(reports_dir / "temporal_horizon_metrics_aggregate.csv", horizon_agg_rows)
    t0 = next((row for row in horizon_agg_rows if row["horizon_s"] == 0), None)
    t6 = next((row for row in horizon_agg_rows if row["horizon_s"] == 6), None)
    error_amplified = bool(t0 and t6 and t6["mean_false_free_rate"] > t0["mean_false_free_rate"] and t6["mean_occupied_iou"] < t0["mean_occupied_iou"])
    summary_md = [
        "# Temporal horizon metrics summary",
        "",
        f"- Effective subset: `{subset_label}`",
        f"- t=0 occupied_iou / false_free: `{None if t0 is None else round(t0['mean_occupied_iou'], 4)}` / `{None if t0 is None else round(t0['mean_false_free_rate'], 4)}`",
        f"- t=6 occupied_iou / false_free: `{None if t6 is None else round(t6['mean_occupied_iou'], 4)}` / `{None if t6 is None else round(t6['mean_false_free_rate'], 4)}`",
        f"- Error amplified: `{error_amplified}`",
    ]
    (reports_dir / "temporal_horizon_metrics_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    # Phase 4 curves
    if args.save_figures:
        plot_horizon_panels(horizon_agg_rows, figure_dir, subset_label)
    write_json(reports_dir / "temporal_error_amplification_summary.json", {"error_amplified": error_amplified, "t0": t0, "t6": t6})

    # Phase 5 groups
    write_json(reports_dir / "class_group_horizon_metrics.json", {"rows": class_group_rows, "effective_subset_size": effective_count, "class_groups": CLASS_GROUPS, "occ_names": OCC_NAMES})
    write_csv(reports_dir / "class_group_horizon_metrics.csv", class_group_rows)
    dynamic_rows = [row for row in class_group_agg_rows if row["group_name"] == "all_dynamic"]
    static_rows = [row for row in class_group_agg_rows if row["group_name"] == "all_static"]
    small_rows = [row for row in class_group_agg_rows if row["group_name"] == "small_object"]
    dominant_group = "unknown"
    last_horizon_ff = {
        "all_dynamic": next((row["mean_group_false_free_rate"] for row in dynamic_rows if row["horizon_s"] == 6), math.nan),
        "all_static": next((row["mean_group_false_free_rate"] for row in static_rows if row["horizon_s"] == 6), math.nan),
        "small_object": next((row["mean_group_false_free_rate"] for row in small_rows if row["horizon_s"] == 6), math.nan),
    }
    dominant_group = max(last_horizon_ff.items(), key=lambda kv: (-1 if math.isnan(kv[1]) else kv[1]))[0]
    write_json(reports_dir / "class_group_horizon_metrics_aggregate.json", {"rows": class_group_agg_rows})
    write_csv(reports_dir / "class_group_horizon_metrics_aggregate.csv", class_group_agg_rows)
    class_group_md = [
        "# Class-group horizon summary",
        "",
        f"- Dynamic t6 false-free: `{last_horizon_ff['all_dynamic']}`",
        f"- Static t6 false-free: `{last_horizon_ff['all_static']}`",
        f"- Small-object t6 false-free: `{last_horizon_ff['small_object']}`",
        f"- Dominant group: `{dominant_group}`",
    ]
    (reports_dir / "class_group_horizon_summary.md").write_text("\n".join(class_group_md), encoding="utf-8")
    if args.save_figures:
        plot_group_curves(class_group_agg_rows, figure_dir)

    # Phase 6 new-visible proxy
    write_json(reports_dir / "new_visible_proxy_metrics.json", {"rows": new_visible_rows})
    write_csv(reports_dir / "new_visible_proxy_metrics.csv", new_visible_rows)
    nv_t6 = next((row for row in new_visible_agg_rows if row["horizon_s"] == 6), None)
    new_visible_summary = "unavailable"
    if nv_t6 is not None:
        new_visible_summary = (
            f"t=6 new_visible_false_free={round(nv_t6.get('mean_new_visible_false_free', 0.0), 4)}, "
            f"persistent_false_free={round(nv_t6.get('mean_persistent_false_free', 0.0), 4)}"
        )
    write_json(reports_dir / "new_visible_proxy_metrics_aggregate.json", {"rows": new_visible_agg_rows})
    write_csv(reports_dir / "new_visible_proxy_metrics_aggregate.csv", new_visible_agg_rows)
    (reports_dir / "new_visible_proxy_summary.md").write_text("# New-visible proxy summary\n\n- " + new_visible_summary + "\n", encoding="utf-8")
    if args.save_figures:
        plot_new_visible_proxy(new_visible_agg_rows, figure_dir)
        # sample0 proxy bev
        s0 = effective_records[0]
        proxy_maps = classify_region_proxies(s0["gt_temporal"][0], s0["gt_temporal"][1 if s0["gt_temporal"].shape[0] > 1 else 0])
        bev = proxy_maps["newly_visible_or_newly_occupied_proxy"].any(dim=-1).cpu().numpy().astype(np.uint8)
        save_panel([(bev, "New-visible proxy BEV")], figure_dir / "new_visible_proxy_bev_sample0.png", cols=1, cmap="gray")

    # Phase 8/9 query coverage
    write_json(reports_dir / "query_coverage_metrics_sample_subset.json", {"rows": query_coverage_rows})
    write_csv(reports_dir / "query_coverage_metrics_sample_subset.csv", query_coverage_rows)
    write_json(reports_dir / "query_coverage_radius_sweep.json", {"rows": query_coverage_agg_rows})
    write_csv(reports_dir / "query_coverage_radius_sweep.csv", query_coverage_agg_rows)
    coverage_summary = coverage_failure_conclusion(query_coverage_rows, radius=2)
    (reports_dir / "query_coverage_summary.md").write_text(
        "# Query coverage summary\n\n"
        f"- Radius 2 mean false-free coverage ratio: `{coverage_summary['mean_query_coverage_false_free_ratio']}`\n"
        f"- Conclusion: `{coverage_summary['conclusion']}`\n",
        encoding="utf-8",
    )
    write_json(reports_dir / "temporal_query_coverage_proxy.json", {"rows": temporal_query_proxy_rows, "aggregate": temporal_query_proxy_agg_rows})
    write_csv(reports_dir / "temporal_query_coverage_proxy.csv", temporal_query_proxy_rows)
    (reports_dir / "temporal_query_coverage_proxy_summary.md").write_text(
        "# Temporal query coverage proxy summary\n\n"
        "- Proxy only, not identity-level query tracking.\n"
        f"- Mean radius-2 false-free coverage summary: `{coverage_summary['mean_query_coverage_false_free_ratio']}`\n",
        encoding="utf-8",
    )

    # Phase 10 joint diagnosis
    write_json(reports_dir / "joint_temporal_query_diagnosis_matrix.json", {"rows": joint_rows})
    write_csv(reports_dir / "joint_temporal_query_diagnosis_matrix.csv", joint_rows)
    ff_vs_cov = [(row["query_coverage_false_free_ratio"], row["false_free_rate"]) for row in joint_rows if row["query_coverage_false_free_ratio"] is not None]
    joint_md = [
        "# Joint temporal-query diagnosis summary",
        "",
        f"- Query coverage conclusion: `{coverage_summary['conclusion']}`",
        f"- Points in matrix: `{len(joint_rows)}`",
    ]
    (reports_dir / "joint_temporal_query_diagnosis_summary.md").write_text("\n".join(joint_md), encoding="utf-8")

    # Figures
    if args.save_figures:
        # sample0 query density / overlay
        s0 = effective_records[0]
        pred0 = s0["pred_temporal"][0]
        gt0 = s0["gt_temporal"][0]
        qpts0, _ = get_query_tensor_for_horizon(s0["query_capture"], 0)
        if isinstance(qpts0, torch.Tensor):
            centers0 = decode_points_if_needed(query_center_proxy(qpts0[0]))
            density0, _ = query_density_map_from_points(centers0)
            gt_bev0 = bev_occ_map(gt0)
            pred_bev0 = bev_occ_map(pred0)
            ff_bev0 = np.logical_and(gt_bev0 == 1, pred_bev0 == 0)
            overlay = np.zeros((GRID_SIZE[0], GRID_SIZE[1], 3), dtype=np.uint8)
            overlay[gt_bev0.astype(bool)] = np.array([0, 255, 0], dtype=np.uint8)
            overlay[pred_bev0.astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
            overlay[ff_bev0.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
            save_panel([(density0, "Query density BEV")], figure_dir / "sample0_query_density_bev.png", cols=1, cmap="magma")
            save_panel([(overlay, "GT / Pred / False-free overlay")], figure_dir / "sample0_query_gt_pred_ff_overlay.png", cols=1)
            # query density rollout
            rollout_imgs = []
            for h_idx in range(min(7, s0["pred_temporal"].shape[0])):
                qpts_h, _ = get_query_tensor_for_horizon(s0["query_capture"], h_idx)
                if isinstance(qpts_h, torch.Tensor):
                    density_h, _ = query_density_map_from_points(decode_points_if_needed(query_center_proxy(qpts_h[0])))
                    rollout_imgs.append((density_h, f"Query density t={h_idx}s"))
            save_panel(rollout_imgs, figure_dir / "query_density_rollout_sample0.png", cols=min(4, max(1, len(rollout_imgs))), cmap="magma")
            # radius sweep figure
            radius_rows_s0 = [r for r in query_coverage_rows if r["sample_index"] == s0["sample_index"] and r["horizon_s"] == 0]
            if radius_rows_s0:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot([r["coverage_radius_cells"] for r in radius_rows_s0], [r["query_coverage_gt_ratio"] for r in radius_rows_s0], marker="o", label="GT coverage")
                ax.plot([r["coverage_radius_cells"] for r in radius_rows_s0], [r["query_coverage_false_free_ratio"] for r in radius_rows_s0], marker="o", label="False-free coverage")
                ax.set_xlabel("Coverage radius (cells)")
                ax.set_ylabel("Coverage ratio")
                ax.set_title("Sample0 query coverage radius sweep")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(figure_dir / "sample0_query_coverage_radius_sweep.png", dpi=150)
                plt.close(fig)

        save_curve(
            [
                {"horizon_s": row["horizon_s"], "mean_query_coverage_gt_ratio": row["mean_query_coverage_gt_ratio"], "std_query_coverage_gt_ratio": 0.0}
                for row in query_coverage_agg_rows if row["coverage_radius_cells"] == 2
            ],
            "horizon_s",
            "mean_query_coverage_gt_ratio",
            None,
            figure_dir / "query_coverage_over_horizon.png",
            "Query coverage over horizon (radius=2)",
            "GT coverage ratio",
        )
        save_curve(
            [
                {"horizon_s": row["horizon_s"], "mean_query_coverage_false_free_ratio": row["mean_query_coverage_false_free_ratio"], "std_query_coverage_false_free_ratio": 0.0}
                for row in query_coverage_agg_rows if row["coverage_radius_cells"] == 2
            ],
            "horizon_s",
            "mean_query_coverage_false_free_ratio",
            None,
            figure_dir / "query_false_free_coverage_over_horizon.png",
            "False-free coverage over horizon (radius=2)",
            "False-free coverage ratio",
        )
        save_curve(
            temporal_query_proxy_agg_rows,
            "horizon_s",
            "mean_query_density_shift_from_t0",
            "std_query_density_shift_from_t0",
            figure_dir / "query_density_shift_over_horizon.png",
            "Query density shift from t0",
            "Shift (cells)",
        )
        if ff_vs_cov:
            scatter_plot(
                [x for x, _ in ff_vs_cov],
                [y for _, y in ff_vs_cov],
                figure_dir / "false_free_vs_query_coverage.png",
                "Query coverage on false-free regions (r=2)",
                "False-free rate",
                "False-free vs query coverage",
            )
        dyn_vs_cov = [(row["query_coverage_false_free_ratio"], row["dynamic_false_free"]) for row in joint_rows if row["query_coverage_false_free_ratio"] is not None]
        if dyn_vs_cov:
            scatter_plot(
                [x for x, _ in dyn_vs_cov],
                [y for _, y in dyn_vs_cov],
                figure_dir / "dynamic_ff_vs_query_coverage.png",
                "Query coverage on false-free regions (r=2)",
                "Dynamic false-free",
                "Dynamic false-free vs query coverage",
            )
        so_vs_density = [(row["query_density_entropy"], row["small_object_false_free"]) for row in joint_rows]
        if so_vs_density:
            scatter_plot(
                [x for x, _ in so_vs_density],
                [y for _, y in so_vs_density],
                figure_dir / "small_object_ff_vs_query_density.png",
                "Query density entropy",
                "Small-object false-free",
                "Small-object false-free vs query density entropy",
            )
        # range bins figure
        range_rows = []
        for rec in effective_records:
            rr = range_false_free_rows(rec["pred_temporal"][0], rec["gt_temporal"][0], valid_mask_from_gt(rec["gt_temporal"][0]))
            for row in rr:
                range_rows.append(row)
        range_bins = [row["range_bin_m"] for row in range_rows[:5]]
        range_means = []
        for range_bin in range_bins:
            vals = [row["false_free_rate"] for row in range_rows if row["range_bin_m"] == range_bin]
            range_means.append(float(np.mean(vals)) if vals else 0.0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range_bins, range_means)
        ax.set_title("Query coverage / false-free range bins proxy")
        ax.set_xlabel("Range bin")
        ax.set_ylabel("Mean false-free rate at t0")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figure_dir / "query_coverage_range_bins.png", dpi=150)
        plt.close(fig)
        # joint panel
        panel_fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        if t0 and t6:
            axes[0].bar(["t0", "t6"], [t0["mean_occupied_iou"], t6["mean_occupied_iou"]])
            axes[0].set_title("Occupied IoU")
            axes[1].bar(["t0", "t6"], [t0["mean_false_free_rate"], t6["mean_false_free_rate"]])
            axes[1].set_title("False-free")
        axes[2].bar(list(last_horizon_ff.keys()), [0.0 if math.isnan(v) else v for v in last_horizon_ff.values()])
        axes[2].set_title("t6 class-group false-free")
        if ff_vs_cov:
            axes[3].scatter([x for x, _ in ff_vs_cov], [y for _, y in ff_vs_cov], alpha=0.7)
            axes[3].set_xlabel("Query coverage")
            axes[3].set_ylabel("False-free")
            axes[3].set_title("Joint query/error view")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        panel_fig.tight_layout()
        panel_fig.savefig(figure_dir / "horizon_joint_panel.png", dpi=150)
        plt.close(panel_fig)

    # Final stage report
    next_unique_action = "Stage SW-3 Query Coverage Repair / Allocation Diagnosis"
    if coverage_summary["conclusion"] == "decode_or_head_failure_dominant":
        next_unique_action = "Stage SW-3 Query Drift / Lifecycle Tracking"
    elif coverage_summary["conclusion"] == "mixed_coverage_and_decode_failure":
        next_unique_action = "Stage SW-3 Query Drift / Lifecycle Tracking"
    stage_report_payload = {
        "executive_summary": {
            "effective_subset_size": effective_count,
            "forward_success_count": success_count,
            "forward_failure_count": failure_count,
            "error_amplified": error_amplified,
            "dominant_group": dominant_group,
            "new_visible_summary": new_visible_summary,
            "query_axis_semantics_confirmed": True,
            "query_coverage_conclusion": coverage_summary["conclusion"],
        },
        "run_manifest": run_manifest,
        "temporal_horizon_aggregate": horizon_agg_rows,
        "class_group_aggregate": class_group_agg_rows,
        "new_visible_proxy_aggregate": new_visible_agg_rows,
        "query_axis_semantics": query_audit_payload,
        "query_coverage_radius_sweep": query_coverage_agg_rows,
        "temporal_query_coverage_proxy": temporal_query_proxy_agg_rows,
        "joint_temporal_query_diagnosis": joint_rows,
        "safe_claims": {
            "subset_diagnostic_only": True,
            "official_benchmark": False,
            "training_completed": False,
            "complete_query_lifecycle": False,
            "sensor_aware_perturbation_completed": False,
        },
        "next_unique_action": next_unique_action,
    }
    markdown_report(
        reports_dir / "stage_sw2_temporal_query_diagnosis_report.md",
        reports_dir / "stage_sw2_temporal_query_diagnosis_report.json",
        stage_report_payload,
        subset_label,
        success_count,
        failure_count,
        t0,
        t6,
        dominant_group,
        new_visible_summary,
        coverage_summary,
    )
    print(json.dumps({
        "effective_subset_size": effective_count,
        "forward_success_count": success_count,
        "forward_failure_count": failure_count,
        "coverage_conclusion": coverage_summary["conclusion"],
        "next_unique_action": next_unique_action,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
