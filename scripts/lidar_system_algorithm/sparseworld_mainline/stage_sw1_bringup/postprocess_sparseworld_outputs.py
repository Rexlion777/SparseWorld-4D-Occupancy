"""Postprocess SparseWorld forward outputs into standard occupancy artifacts.

This stage does three bounded tasks:
1. build standard occupancy tensors for current frame + temporal rollout
2. compute OpenOCC-style diagnostic metrics
3. summarize query tensor candidates captured during forward
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess SparseWorld outputs")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--raw-output-pt", required=True)
    parser.add_argument("--query-output-pt", required=True)
    parser.add_argument("--output-adapter-json", required=True)
    parser.add_argument("--output-diagnosis-sample-json", required=True)
    parser.add_argument("--output-diagnosis-temporal-json", required=True)
    parser.add_argument("--output-query-manifest-json", required=True)
    parser.add_argument("--output-query-summary-csv", required=True)
    parser.add_argument("--standard-pred-sample-pt", required=True)
    parser.add_argument("--standard-gt-sample-pt", required=True)
    parser.add_argument("--standard-pred-temporal-pt", required=True)
    parser.add_argument("--standard-gt-temporal-pt", required=True)
    parser.add_argument("--figure-dir", required=True)
    return parser.parse_args()


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


def summarize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: summarize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 16:
            return {"type": "list", "length": len(obj)}
        return [summarize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        cpu = obj.float().cpu()
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
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


def compute_semantic_miou(pred: torch.Tensor, gt: torch.Tensor, empty_idx: int = 17) -> dict[str, Any]:
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
    return {"semantic_miou": float(np.mean(ious)) if ious else 0.0, "class_ious": class_ious}


def occupancy_metrics(pred: torch.Tensor, gt: torch.Tensor, empty_idx: int = 17) -> dict[str, Any]:
    pred_occ = pred != empty_idx
    gt_occ = gt != empty_idx
    inter = torch.logical_and(pred_occ, gt_occ).sum().item()
    union = torch.logical_or(pred_occ, gt_occ).sum().item()
    gt_occ_count = gt_occ.sum().item()
    gt_free_count = (~gt_occ).sum().item()
    pred_occ_count = pred_occ.sum().item()
    false_free = torch.logical_and(gt_occ, ~pred_occ).sum().item()
    false_occ = torch.logical_and(~gt_occ, pred_occ).sum().item()
    sem = compute_semantic_miou(pred, gt, empty_idx=empty_idx)
    return {
        "pred_occupied_count": int(pred_occ_count),
        "gt_occupied_count": int(gt_occ_count),
        "occupied_iou": float(inter / union) if union else 0.0,
        "false_free_rate": float(false_free / gt_occ_count) if gt_occ_count else 0.0,
        "false_occupied_rate": float(false_occ / gt_free_count) if gt_free_count else 0.0,
        "occupied_accuracy": float(inter / gt_occ_count) if gt_occ_count else 0.0,
        "free_accuracy": float(((~pred_occ) & (~gt_occ)).sum().item() / gt_free_count) if gt_free_count else 0.0,
        "semantic_miou": sem["semantic_miou"],
        "class_ious": sem["class_ious"],
    }


def temporal_consistency(pred_t: torch.Tensor, empty_idx: int = 17) -> dict[str, Any]:
    pred_occ = pred_t != empty_idx
    step_scores = []
    for i in range(pred_occ.shape[0] - 1):
        a = pred_occ[i]
        b = pred_occ[i + 1]
        union = torch.logical_or(a, b).sum().item()
        inter = torch.logical_and(a, b).sum().item()
        step_scores.append(inter / union if union else 1.0)
    return {
        "temporal_occ_iou_mean": float(np.mean(step_scores)) if step_scores else 1.0,
        "temporal_occ_iou_per_step": [float(x) for x in step_scores],
    }


def range_false_free(pred: torch.Tensor, gt: torch.Tensor, x_min: float = -40.0, y_min: float = -40.0, voxel: float = 0.4, empty_idx: int = 17) -> list[dict[str, Any]]:
    pred_occ = pred != empty_idx
    gt_occ = gt != empty_idx
    x_centers = x_min + (torch.arange(pred.shape[0], dtype=torch.float32) + 0.5) * voxel
    y_centers = y_min + (torch.arange(pred.shape[1], dtype=torch.float32) + 0.5) * voxel
    xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(-1).expand_as(pred.float())
    bins = [0.0, 10.0, 20.0, 30.0, 40.0, 80.0]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rr >= lo) & (rr < hi)
        gt_occ_count = torch.logical_and(gt_occ, mask).sum().item()
        false_free_count = torch.logical_and(torch.logical_and(gt_occ, ~pred_occ), mask).sum().item()
        rows.append(
            {
                "range_bin_m": f"[{lo},{hi})",
                "gt_occupied_count": int(gt_occ_count),
                "false_free_count": int(false_free_count),
                "false_free_rate": float(false_free_count / gt_occ_count) if gt_occ_count else 0.0,
            }
        )
    return rows


def save_triptych(gt: torch.Tensor, pred: torch.Tensor, save_path: Path, empty_idx: int = 17) -> None:
    gt_occ = (gt != empty_idx).any(dim=-1).cpu().numpy().astype(np.uint8)
    pred_occ = (pred != empty_idx).any(dim=-1).cpu().numpy().astype(np.uint8)
    false_free = ((gt != empty_idx) & (pred == empty_idx)).any(dim=-1).cpu().numpy().astype(np.uint8)
    false_occ = ((gt == empty_idx) & (pred != empty_idx)).any(dim=-1).cpu().numpy().astype(np.uint8)
    error = np.zeros((*gt_occ.shape, 3), dtype=np.uint8)
    error[false_free.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
    error[false_occ.astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
    both = np.logical_and(gt_occ == 1, pred_occ == 1)
    error[both] = np.array([0, 255, 0], dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_occ, cmap="gray")
    axes[0].set_title("GT occupied BEV")
    axes[1].imshow(pred_occ, cmap="gray")
    axes[1].set_title("Pred occupied BEV")
    axes[2].imshow(error)
    axes[2].set_title("Error map BEV")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_temporal_rollout(pred_t: torch.Tensor, save_path: Path, empty_idx: int = 17) -> None:
    t = pred_t.shape[0]
    fig, axes = plt.subplots(1, t, figsize=(3 * t, 3))
    if t == 1:
        axes = [axes]
    for i in range(t):
        bev = (pred_t[i] != empty_idx).any(dim=-1).cpu().numpy().astype(np.uint8)
        axes[i].imshow(bev, cmap="gray")
        axes[i].set_title(f"t={i}s")
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    for out_path in [
        args.output_adapter_json,
        args.output_diagnosis_sample_json,
        args.output_diagnosis_temporal_json,
        args.output_query_manifest_json,
        args.output_query_summary_csv,
        args.standard_pred_sample_pt,
        args.standard_gt_sample_pt,
        args.standard_pred_temporal_pt,
        args.standard_gt_temporal_pt,
    ]:
        Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from mmcv import Config
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import patch_config

    cfg = Config.fromfile(str(config_path))
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)
    setup_multi_processes(cfg)
    split_cfg = cfg.data[args.split]
    split_cfg.test_mode = args.split != "train"
    dataset = build_dataset(split_cfg)
    sample = unwrap(dataset[args.sample_index])

    raw = torch.load(str(Path(args.raw_output_pt).resolve()), map_location="cpu", weights_only=False)
    query = torch.load(str(Path(args.query_output_pt).resolve()), map_location="cpu", weights_only=False)

    pred_keys = sorted([k for k in raw.keys() if k.startswith("semantic_occ_")], key=lambda x: int(x.split("_")[-1].replace("s", "")))
    pred_temporal = torch.stack([torch.as_tensor(raw[k][0]).long() for k in pred_keys], dim=0)
    pred_current = pred_temporal[0]

    gt_current = torch.as_tensor(sample["voxel_semantics"]).long()
    gt_temporal_list = [gt_current]
    temporal_semantics = sample["temporal_semantics"]
    for i in range(1, len(pred_keys)):
        gt_temporal_list.append(torch.as_tensor(temporal_semantics[i]["voxel_semantics"]).long())
    gt_temporal = torch.stack(gt_temporal_list, dim=0)

    torch.save(pred_current, Path(args.standard_pred_sample_pt).resolve())
    torch.save(gt_current, Path(args.standard_gt_sample_pt).resolve())
    torch.save(pred_temporal, Path(args.standard_pred_temporal_pt).resolve())
    torch.save(gt_temporal, Path(args.standard_gt_temporal_pt).resolve())

    adapter_payload = {
        "sample_index": args.sample_index,
        "pred_keys": pred_keys,
        "pred_current_summary": summarize(pred_current),
        "gt_current_summary": summarize(gt_current),
        "pred_temporal_summary": summarize(pred_temporal),
        "gt_temporal_summary": summarize(gt_temporal),
        "empty_idx": 17,
        "standard_pred_sample_pt": str(Path(args.standard_pred_sample_pt).resolve()),
        "standard_gt_sample_pt": str(Path(args.standard_gt_sample_pt).resolve()),
        "standard_pred_temporal_pt": str(Path(args.standard_pred_temporal_pt).resolve()),
        "standard_gt_temporal_pt": str(Path(args.standard_gt_temporal_pt).resolve()),
    }
    Path(args.output_adapter_json).write_text(json.dumps(adapter_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_metrics = occupancy_metrics(pred_current, gt_current, empty_idx=17)
    sample_metrics["range_false_free"] = range_false_free(pred_current, gt_current, empty_idx=17)
    Path(args.output_diagnosis_sample_json).write_text(json.dumps(sample_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    temporal_metrics = {
        "per_timestep": [occupancy_metrics(pred_temporal[i], gt_temporal[i], empty_idx=17) for i in range(pred_temporal.shape[0])],
        **temporal_consistency(pred_temporal, empty_idx=17),
    }
    Path(args.output_diagnosis_temporal_json).write_text(json.dumps(temporal_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    triptych_path = figure_dir / "sparseworld_sample0_gt_pred_error.png"
    temporal_path = figure_dir / "sparseworld_temporal_rollout.png"
    save_triptych(gt_current, pred_current, triptych_path, empty_idx=17)
    save_temporal_rollout(pred_temporal, temporal_path, empty_idx=17)

    forward_outputs = query.get("forward_backbone_outputs", {})
    candidates = []
    for key, value in forward_outputs.items():
        summary = summarize(value)
        row = {
            "name": key,
            "summary": summary,
        }
        if isinstance(summary, dict) and "shape" in summary:
            shape = summary["shape"]
            row["shape"] = shape
            row["query_count"] = shape[1] if len(shape) >= 2 else None
            row["embed_dim"] = shape[-1] if len(shape) >= 1 else None
        candidates.append(row)

    query_payload = {
        "available_keys": sorted(list(forward_outputs.keys())) if isinstance(forward_outputs, dict) else [],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "raw_query_artifact": str(Path(args.query_output_pt).resolve()),
    }
    Path(args.output_query_manifest_json).write_text(json.dumps(query_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with Path(args.output_query_summary_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "shape", "query_count", "embed_dim"])
        writer.writeheader()
        for row in candidates:
            writer.writerow(
                {
                    "name": row["name"],
                    "shape": row.get("shape"),
                    "query_count": row.get("query_count"),
                    "embed_dim": row.get("embed_dim"),
                }
            )

    print(json.dumps({
        "occupied_iou": sample_metrics["occupied_iou"],
        "semantic_miou": sample_metrics["semantic_miou"],
        "pred_occupied_count": sample_metrics["pred_occupied_count"],
        "temporal_occ_iou_mean": temporal_metrics["temporal_occ_iou_mean"],
        "query_candidate_count": len(candidates),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
