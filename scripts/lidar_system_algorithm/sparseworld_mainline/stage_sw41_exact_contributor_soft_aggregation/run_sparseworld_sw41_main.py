"""Stage SW-4.1 exact contributor and soft aggregation repair ablation.

This stage reuses SparseWorld Stage SW-2 / SW-3.5 / SW-4.0 artifacts and
replays `get_occ()` offline from saved query outputs. It does not retrain or
rerun the image backbone. The focus is:

1. radius coverage -> exact contributor bottleneck decomposition
2. neighbor leakage / offset diagnosis
3. hard get_occ reconstruction equivalence
4. non-training ablations:
   - soft splatting
   - adaptive gate relax
   - class-aware aggregation
5. 20-sample lightweight validation of the best non-oracle candidates

Safe-claim boundary:
- subset diagnostic repair ablation only
- no training
- no official SparseWorld benchmark
- oracle variants remain diagnostic-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_SW4_DIR = SCRIPT_DIR.parent / "stage_sw4_covered_ff_semantic_activation"
STAGE_SW2_DIR = SCRIPT_DIR.parent / "stage_sw2_temporal_query_diagnosis"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(STAGE_SW4_DIR))
sys.path.insert(0, str(STAGE_SW2_DIR))

from instrument_get_occ import (  # type: ignore
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
from get_occ_replay_and_ablation import (  # type: ignore
    AggregateConfig,
    AssignConfig,
    GateConfig,
    get_default_variant_library,
    replay_get_occ_variant,
    save_replay_debug,
    tensor_to_cpu,
)
from run_sparseworld_sw2_main import compute_base_metrics, valid_mask_from_gt  # type: ignore
import run_sparseworld_sw4_main as sw4  # type: ignore

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"

EMPTY_IDX = 17
PRIMARY_RADIUS = 2
DIAG_SAMPLE_CAP = 512
NEAREST_CHUNK = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage SW-4.1 exact contributor and soft aggregation repair ablation")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--horizons", default="0,1,2,3,4,5,6")
    parser.add_argument("--run-baseline", action="store_true", default=True)
    parser.add_argument("--run-diagnostics", action="store_true", default=True)
    parser.add_argument("--run-ablation", action="store_true", default=True)
    parser.add_argument("--save-debug", action="store_true", default=True)
    parser.add_argument("--save-figures", action="store_true", default=True)
    parser.add_argument("--oracle-samples", type=int, default=5)
    parser.add_argument("--diag-samples", type=int, default=5)
    parser.add_argument("--primary-radius", type=int, default=PRIMARY_RADIUS)
    parser.add_argument("--sample-cap-per-region", type=int, default=DIAG_SAMPLE_CAP)
    parser.add_argument("--nearest-chunk", type=int, default=NEAREST_CHUNK)
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[SW41] {msg}", flush=True)


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, LOGS_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_horizons(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    return sorted(set(vals))


def load_records(num_samples: int, sample_indices: list[int] | None = None) -> list[dict[str, Any]]:
    rows = sw4.load_success_rows(DEFAULT_SW2_REPORTS, 10000)
    if sample_indices:
        wanted = set(sample_indices)
        rows = [row for row in rows if int(row["sample_index"]) in wanted]
    rows = rows[:num_samples]
    return [sw4.load_record(row) for row in rows]


def build_geometric_mask(head: Any, point_info: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = torch.zeros(tuple(int(x.item()) for x in head.voxel_num), dtype=torch.bool, device=point_info["decoded_points"].device)
    valid = point_info["valid_range_mask_flat"]
    vox = point_info["pre_voxel_index_flat"][valid].long()
    if vox.numel():
        mask[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    return mask


def safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_extra_metrics(pred: torch.Tensor, gt: torch.Tensor, gt0: torch.Tensor) -> dict[str, float]:
    pred_occ = pred != EMPTY_IDX
    gt_occ = gt != EMPTY_IDX
    gt0_occ = gt0 != EMPTY_IDX
    tp = torch.logical_and(pred_occ, gt_occ)
    small_gt = sw4.group_mask(gt, sw4.SMALL_OBJECT)
    dynamic_gt = sw4.group_mask(gt, sw4.ALL_DYNAMIC)
    static_gt = sw4.group_mask(gt, sw4.STATIC_BACKGROUND) & gt_occ
    new_visible = (~gt0_occ) & gt_occ
    persistent = gt0_occ & gt_occ
    return {
        "small_object_false_free": safe_rate(torch.logical_and(small_gt, ~pred_occ).sum().item(), small_gt.sum().item()),
        "dynamic_false_free": safe_rate(torch.logical_and(dynamic_gt, ~pred_occ).sum().item(), dynamic_gt.sum().item()),
        "static_false_free": safe_rate(torch.logical_and(static_gt, ~pred_occ).sum().item(), static_gt.sum().item()),
        "new_visible_recall": safe_rate(torch.logical_and(tp, new_visible).sum().item(), new_visible.sum().item()),
        "persistent_recall": safe_rate(torch.logical_and(tp, persistent).sum().item(), persistent.sum().item()),
    }


def build_variant_metric_row(
    record: dict[str, Any],
    horizon_s: int,
    variant_name: str,
    replay: dict[str, Any],
    runtime_ms: float,
) -> dict[str, Any]:
    pred = replay["output"]["occ_pred"].detach().cpu()
    gt = record["gt_temporal"][horizon_s].cpu()
    gt0 = record["gt_temporal"][0].cpu()
    base = compute_base_metrics(pred, gt, valid_mask_from_gt(gt))
    extra = compute_extra_metrics(pred, gt, gt0)
    pred_occ = pred != EMPTY_IDX
    gt_occ = gt != EMPTY_IDX
    contributor_dense = replay["output"]["contributor_count_dense"].detach().cpu()
    point_info = replay["point_info"]
    geometric_mask = build_geometric_mask(replay["head"], point_info).detach().cpu()
    cover = sw4.dilate3d(geometric_mask, PRIMARY_RADIUS)
    masks = sw4.build_temporal_region_masks(gt0, gt, pred, cover)
    covered_ff = masks["covered_false_free"]
    covered_tp = masks["covered_tp"]
    gate_mask = replay["gate_info"]["gate_mask"].detach().cpu()
    row = {
        "variant_name": variant_name,
        "sample_index": record["sample_index"],
        "sample_token": record["sample_token"],
        "scene_token": record["scene_token"],
        "scene_name": record["scene_name"],
        "timestamp": record["timestamp"],
        "horizon_s": horizon_s,
        "runtime_ms": float(runtime_ms),
        "gate_pass_ratio": float(gate_mask.float().mean().item()) if gate_mask.numel() else 0.0,
        "mean_contributor_count_gt": float(contributor_dense[gt_occ].float().mean().item()) if bool(gt_occ.any().item()) else 0.0,
        "mean_contributor_count_pred_occ": float(contributor_dense[pred_occ].float().mean().item()) if bool(pred_occ.any().item()) else 0.0,
        "covered_ff_contributor_ratio": float((contributor_dense[covered_ff] > 0).float().mean().item()) if bool(covered_ff.any().item()) else 0.0,
        "covered_tp_contributor_ratio": float((contributor_dense[covered_tp] > 0).float().mean().item()) if bool(covered_tp.any().item()) else 0.0,
    }
    row.update(base)
    row.update(extra)
    return row


def attach_deltas(rows: list[dict[str, Any]], baseline_variant: str) -> list[dict[str, Any]]:
    base_map: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row["variant_name"] == baseline_variant:
            base_map[(int(row["sample_index"]), int(row["horizon_s"]))] = row
    out = []
    for row in rows:
        key = (int(row["sample_index"]), int(row["horizon_s"]))
        base = base_map.get(key)
        row = dict(row)
        if base is not None:
            for metric in [
                "occupied_iou",
                "semantic_miou",
                "false_free_rate",
                "false_occupied_rate",
                "pred_occupied_count",
                "pred_gt_occupied_ratio",
                "small_object_false_free",
                "new_visible_recall",
                "persistent_recall",
                "gate_pass_ratio",
                "covered_ff_contributor_ratio",
            ]:
                row[f"{metric}_delta"] = float(row[metric] - base[metric])
        out.append(row)
    return out


def aggregate_rows(rows: list[dict[str, Any]], group_key: str = "variant_name") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    out: list[dict[str, Any]] = []
    metric_keys = [
        "occupied_iou",
        "semantic_miou",
        "false_free_rate",
        "false_occupied_rate",
        "pred_occupied_count",
        "pred_gt_occupied_ratio",
        "small_object_false_free",
        "dynamic_false_free",
        "static_false_free",
        "new_visible_recall",
        "persistent_recall",
        "runtime_ms",
        "gate_pass_ratio",
        "covered_ff_contributor_ratio",
        "covered_tp_contributor_ratio",
        "mean_contributor_count_gt",
    ]
    delta_keys = [f"{m}_delta" for m in metric_keys if f"{m}_delta" in rows[0]] if rows else []
    for key, items in groups.items():
        row: dict[str, Any] = {group_key: key, "sample_horizon_count": len(items)}
        for metric in metric_keys + delta_keys:
            vals = [float(x[metric]) for x in items if metric in x]
            row[f"mean_{metric}"] = float(np.mean(vals)) if vals else 0.0
            row[f"std_{metric}"] = float(np.std(vals)) if vals else 0.0
        out.append(row)
    out.sort(key=lambda x: str(x[group_key]))
    return out


def plot_metric_panel(fig_path: Path, agg_rows: list[dict[str, Any]], metric_pairs: list[tuple[str, str]], title: str) -> None:
    fig, axes = plt.subplots(1, len(metric_pairs), figsize=(5 * len(metric_pairs), 4))
    if len(metric_pairs) == 1:
        axes = [axes]
    labels = [row["variant_name"] for row in agg_rows]
    for ax, (metric, ylabel) in zip(axes, metric_pairs):
        ys = [row.get(f"mean_{metric}", 0.0) for row in agg_rows]
        ax.bar(labels, ys)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def build_voxel_hash(vox: torch.Tensor, voxel_num: torch.Tensor) -> torch.Tensor:
    mul_y = int(voxel_num[0].item())
    mul_z = int(voxel_num[0].item() * voxel_num[1].item())
    return vox[:, 0].long() + vox[:, 1].long() * mul_y + vox[:, 2].long() * mul_z


def membership_ratio(target: torch.Tensor, source: torch.Tensor, voxel_num: torch.Tensor, dims: str = "xyz") -> torch.Tensor:
    if target.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=target.device)
    if source.numel() == 0:
        return torch.zeros((target.shape[0],), dtype=torch.bool, device=target.device)
    if dims == "xyz":
        lhs = build_voxel_hash(target, voxel_num)
        rhs = build_voxel_hash(source, voxel_num)
    elif dims == "xy":
        mul_y = int(voxel_num[0].item())
        lhs = target[:, 0].long() + target[:, 1].long() * mul_y
        rhs = source[:, 0].long() + source[:, 1].long() * mul_y
    else:
        raise ValueError(f"unsupported dims: {dims}")
    return torch.isin(lhs, rhs)


def nearest_offsets(
    target: torch.Tensor,
    support: torch.Tensor,
    chunk: int,
) -> dict[str, torch.Tensor]:
    device = target.device
    n = target.shape[0]
    if n == 0:
        empty = torch.zeros((0,), device=device)
        return {
            "min_l1": empty,
            "min_l2": empty,
            "min_linf": empty,
            "nearest_dx": empty,
            "nearest_dy": empty,
            "nearest_dz": empty,
        }
    if support.numel() == 0:
        inf = torch.full((n,), float("inf"), device=device)
        return {
            "min_l1": inf,
            "min_l2": inf,
            "min_linf": inf,
            "nearest_dx": inf,
            "nearest_dy": inf,
            "nearest_dz": inf,
        }
    out_l1: list[torch.Tensor] = []
    out_l2: list[torch.Tensor] = []
    out_linf: list[torch.Tensor] = []
    out_dx: list[torch.Tensor] = []
    out_dy: list[torch.Tensor] = []
    out_dz: list[torch.Tensor] = []
    support_f = support.float()
    for start in range(0, n, chunk):
        q = target[start : start + chunk].float()
        diff = support_f.unsqueeze(0) - q.unsqueeze(1)
        absdiff = diff.abs()
        l2 = diff.square().sum(dim=-1)
        argmin = l2.argmin(dim=1)
        nearest = support[argmin]
        off = nearest - target[start : start + chunk]
        out_dx.append(off[:, 0].float())
        out_dy.append(off[:, 1].float())
        out_dz.append(off[:, 2].float())
        out_l2.append(l2.gather(1, argmin[:, None]).sqrt().squeeze(1))
        out_l1.append(absdiff.sum(dim=-1).min(dim=1).values)
        out_linf.append(absdiff.amax(dim=-1).min(dim=1).values)
        del diff, absdiff, l2, argmin, nearest, off
    return {
        "min_l1": torch.cat(out_l1),
        "min_l2": torch.cat(out_l2),
        "min_linf": torch.cat(out_linf),
        "nearest_dx": torch.cat(out_dx),
        "nearest_dy": torch.cat(out_dy),
        "nearest_dz": torch.cat(out_dz),
    }


def sampled_coords(mask: torch.Tensor, limit: int, seed: int) -> torch.Tensor:
    return sw4.sample_mask_coords(mask, limit=limit, seed=seed)


def summarize_exact_stages(
    head: Any,
    coords: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    replay: dict[str, Any],
    primary_radius: int,
    nearest_chunk: int,
) -> dict[str, Any]:
    device = replay["point_info"]["decoded_points"].device
    coords_gpu = coords.to(device=device, dtype=torch.long)
    point_info = replay["point_info"]
    gate_info = replay["gate_info"]
    assignment = replay["assignment"]
    output = replay["output"]

    all_pre = point_info["pre_voxel_index_flat"].long()
    valid_pre = all_pre[point_info["valid_range_mask_flat"]]
    gated_pre = all_pre[point_info["valid_range_mask_flat"] & gate_info["gate_mask"].reshape(-1)]
    assigned = assignment["target_voxel_indices"].long()
    voxel_num = head.voxel_num.long()
    contributor_dense = output["contributor_count_dense"]

    nearest_valid = nearest_offsets(coords_gpu, valid_pre, nearest_chunk)
    nearest_gated = nearest_offsets(coords_gpu, gated_pre, nearest_chunk)

    same_xy_gated = membership_ratio(coords_gpu, gated_pre, voxel_num, dims="xy")
    exact_gated = membership_ratio(coords_gpu, gated_pre, voxel_num, dims="xyz")
    exact_assigned = membership_ratio(coords_gpu, assigned, voxel_num, dims="xyz")

    contrib_ratio = (contributor_dense[coords_gpu[:, 0], coords_gpu[:, 1], coords_gpu[:, 2]] > 0).float()
    pred_occ_ratio = (pred.to(device)[coords_gpu[:, 0], coords_gpu[:, 1], coords_gpu[:, 2]] != EMPTY_IDX).float()
    correct_ratio = (
        (pred.to(device)[coords_gpu[:, 0], coords_gpu[:, 1], coords_gpu[:, 2]] == gt.to(device)[coords_gpu[:, 0], coords_gpu[:, 1], coords_gpu[:, 2]])
        & (gt.to(device)[coords_gpu[:, 0], coords_gpu[:, 1], coords_gpu[:, 2]] != EMPTY_IDX)
    ).float()

    stage = {
        "sampled_voxel_count": int(coords.shape[0]),
        "support_r2_ratio": float((nearest_valid["min_linf"] <= 2).float().mean().item()) if coords.shape[0] else 0.0,
        "support_r3_ratio": float((nearest_valid["min_linf"] <= 3).float().mean().item()) if coords.shape[0] else 0.0,
        "support_r5_ratio": float((nearest_valid["min_linf"] <= 5).float().mean().item()) if coords.shape[0] else 0.0,
        "valid_range_support_ratio": float((nearest_valid["min_linf"] <= primary_radius).float().mean().item()) if coords.shape[0] else 0.0,
        "gate_pass_support_ratio": float((nearest_gated["min_linf"] <= primary_radius).float().mean().item()) if coords.shape[0] else 0.0,
        "exact_bev_match_ratio": float(same_xy_gated.float().mean().item()) if coords.shape[0] else 0.0,
        "exact_z_match_ratio": float(exact_gated.float().mean().item()) if coords.shape[0] else 0.0,
        "exact_voxel_assignment_ratio": float(exact_assigned.float().mean().item()) if coords.shape[0] else 0.0,
        "aggregation_candidate_ratio": float(exact_assigned.float().mean().item()) if coords.shape[0] else 0.0,
        "final_contributor_ratio": float(contrib_ratio.mean().item()) if coords.shape[0] else 0.0,
        "final_occupied_ratio": float(pred_occ_ratio.mean().item()) if coords.shape[0] else 0.0,
        "final_correct_semantic_ratio": float(correct_ratio.mean().item()) if coords.shape[0] else 0.0,
        "mean_nearest_dx": float(nearest_valid["nearest_dx"].mean().item()) if coords.shape[0] else 0.0,
        "mean_nearest_dy": float(nearest_valid["nearest_dy"].mean().item()) if coords.shape[0] else 0.0,
        "mean_nearest_dz": float(nearest_valid["nearest_dz"].mean().item()) if coords.shape[0] else 0.0,
        "mean_nearest_l1": float(nearest_valid["min_l1"].mean().item()) if coords.shape[0] else 0.0,
        "mean_nearest_l2": float(nearest_valid["min_l2"].mean().item()) if coords.shape[0] else 0.0,
        "mean_nearest_linf": float(nearest_valid["min_linf"].mean().item()) if coords.shape[0] else 0.0,
    }
    return stage


def summarize_neighbor_leakage(
    head: Any,
    coords: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    replay: dict[str, Any],
    nearest_chunk: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    device = replay["point_info"]["decoded_points"].device
    coords_gpu = coords.to(device=device, dtype=torch.long)
    point_info = replay["point_info"]
    assignment = replay["assignment"]
    valid_pre = point_info["pre_voxel_index_flat"][point_info["valid_range_mask_flat"]].long()
    assigned = assignment["target_voxel_indices"].long()
    nearest = nearest_offsets(coords_gpu, valid_pre, nearest_chunk)
    contrib = replay["output"]["contributor_count_dense"]
    pred_occ = pred.to(device) != EMPTY_IDX
    gt_occ = gt.to(device) != EMPTY_IDX
    fo_mask = (~gt_occ) & pred_occ

    neighbor_contrib = []
    neighbor_pred_occ = []
    neighbor_fo = []
    for coord in coords_gpu:
        lo = torch.clamp(coord - 1, min=0)
        hi = torch.minimum(coord + 1, head.voxel_num.long() - 1)
        sx = slice(int(lo[0].item()), int(hi[0].item()) + 1)
        sy = slice(int(lo[1].item()), int(hi[1].item()) + 1)
        sz = slice(int(lo[2].item()), int(hi[2].item()) + 1)
        sub_contrib = contrib[sx, sy, sz] > 0
        sub_pred_occ = pred_occ[sx, sy, sz]
        sub_fo = fo_mask[sx, sy, sz]
        cx, cy, cz = int(coord[0].item() - lo[0].item()), int(coord[1].item() - lo[1].item()), int(coord[2].item() - lo[2].item())
        sub_contrib = sub_contrib.clone()
        sub_pred_occ = sub_pred_occ.clone()
        sub_fo = sub_fo.clone()
        sub_contrib[cx, cy, cz] = False
        sub_pred_occ[cx, cy, cz] = False
        sub_fo[cx, cy, cz] = False
        neighbor_contrib.append(bool(sub_contrib.any().item()))
        neighbor_pred_occ.append(bool(sub_pred_occ.any().item()))
        neighbor_fo.append(bool(sub_fo.any().item()))
    detail = {
        "dx": nearest["nearest_dx"].detach().cpu(),
        "dy": nearest["nearest_dy"].detach().cpu(),
        "dz": nearest["nearest_dz"].detach().cpu(),
    }
    row = {
        "sampled_voxel_count": int(coords.shape[0]),
        "same_bev_cell_ratio": float(((nearest["nearest_dx"] == 0) & (nearest["nearest_dy"] == 0)).float().mean().item()) if coords.shape[0] else 0.0,
        "same_z_bin_ratio": float((nearest["nearest_dz"] == 0).float().mean().item()) if coords.shape[0] else 0.0,
        "same_exact_voxel_ratio": float(((nearest["nearest_dx"] == 0) & (nearest["nearest_dy"] == 0) & (nearest["nearest_dz"] == 0)).float().mean().item()) if coords.shape[0] else 0.0,
        "mean_abs_dx": float(nearest["nearest_dx"].abs().mean().item()) if coords.shape[0] else 0.0,
        "mean_abs_dy": float(nearest["nearest_dy"].abs().mean().item()) if coords.shape[0] else 0.0,
        "mean_abs_dz": float(nearest["nearest_dz"].abs().mean().item()) if coords.shape[0] else 0.0,
        "neighbor_contributor_ratio": float(np.mean(neighbor_contrib)) if neighbor_contrib else 0.0,
        "neighbor_pred_occupied_ratio": float(np.mean(neighbor_pred_occ)) if neighbor_pred_occ else 0.0,
        "neighbor_false_occupied_ratio": float(np.mean(neighbor_fo)) if neighbor_fo else 0.0,
    }
    return row, detail


def run_variant_set(
    head: Any,
    records: list[dict[str, Any]],
    horizons: list[int],
    variant_keys: list[str],
    variant_lib: dict[str, tuple[GateConfig, AssignConfig, AggregateConfig]],
    phase_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(records) * len(horizons) * len(variant_keys)
    done = 0
    with torch.inference_mode():
        for record in records:
            for horizon_s in horizons:
                pred_dict = extract_pred_dict_for_horizon(record["query_artifact"], horizon_s)
                for variant_name in variant_keys:
                    gate_cfg, assign_cfg, aggregate_cfg = variant_lib[variant_name]
                    start = time.perf_counter()
                    replay = replay_get_occ_variant(head, pred_dict, horizon_s, gate_cfg, assign_cfg, aggregate_cfg)
                    replay["head"] = head
                    runtime_ms = (time.perf_counter() - start) * 1000.0
                    rows.append(build_variant_metric_row(record, horizon_s, variant_name, replay, runtime_ms))
                    done += 1
                    if done % 10 == 0 or done == total:
                        log(f"{phase_name}: {done}/{total} variant sample-horizons complete")
    return rows


def best_variant_from_family(agg_rows: list[dict[str, Any]], allowed: set[str]) -> str:
    candidates = [row for row in agg_rows if row["variant_name"] in allowed and row["variant_name"] not in {"V0_hard_baseline", "G0_baseline_gate", "C0_baseline_aggregation"}]
    if not candidates:
        return next(iter(sorted(allowed)))
    def score(row: dict[str, Any]) -> float:
        ff = -row.get("mean_false_free_rate_delta", 0.0)
        fo_pen = max(0.0, row.get("mean_false_occupied_rate_delta", 0.0) - 0.01) * 4.0
        iou = row.get("mean_occupied_iou_delta", 0.0) * 2.0
        small = -row.get("mean_small_object_false_free_delta", 0.0)
        newv = row.get("mean_new_visible_recall_delta", 0.0)
        runtime_pen = max(0.0, (row.get("mean_runtime_ms_delta", 0.0) if "mean_runtime_ms_delta" in row else 0.0)) / 1000.0
        return ff + iou + 0.8 * small + 0.8 * newv - fo_pen - runtime_pen
    candidates.sort(key=score, reverse=True)
    return str(candidates[0]["variant_name"])


def combine_variant(
    name: str,
    soft_key: str | None,
    gate_key: str | None,
    class_key: str | None,
    variant_lib: dict[str, tuple[GateConfig, AssignConfig, AggregateConfig]],
) -> tuple[GateConfig, AssignConfig, AggregateConfig]:
    gate = GateConfig()
    assign = AssignConfig()
    agg = AggregateConfig()
    if gate_key is not None:
        gate = variant_lib[gate_key][0]
    if soft_key is not None:
        assign = variant_lib[soft_key][1]
    if class_key is not None:
        agg = variant_lib[class_key][2]
    gate = GateConfig(**{**gate.__dict__, "name": f"{name}_gate"})
    assign = AssignConfig(**{**assign.__dict__, "name": f"{name}_assign"})
    agg = AggregateConfig(**{**agg.__dict__, "name": f"{name}_agg"})
    return gate, assign, agg


def select_case(best_row: dict[str, Any] | None, family: str | None) -> tuple[str, str]:
    if best_row is None:
        return "R6", "no repair candidate effective"
    ff_delta = float(best_row.get("mean_false_free_rate_delta", 0.0))
    fo_delta = float(best_row.get("mean_false_occupied_rate_delta", 0.0))
    occ_iou_delta = float(best_row.get("mean_occupied_iou_delta", 0.0))
    if ff_delta < -0.005 and fo_delta <= 0.01 and occ_iou_delta >= -0.002:
        if family == "soft":
            return "R1", "soft splatting effective"
        if family == "gate":
            return "R2", "adaptive gate effective"
        if family == "class":
            return "R3", "class-aware aggregation effective"
        if family == "combined":
            return "R4", "combined repair effective"
    if ff_delta < -0.005 and fo_delta > 0.01:
        return "R5", "repair tradeoff unacceptable"
    return "R7", "diagnostic-only improvement"


def main() -> int:
    args = parse_args()
    ensure_dirs()
    horizons = parse_horizons(args.horizons)
    sample_indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
    sw41_log = LOGS_DIR / "phase1_sw41_main.log"
    phase4_log = LOGS_DIR / "phase4_get_occ_replay_decomposition.log"

    log("loading Stage SW-2 success subset")
    records20 = load_records(args.num_samples, sample_indices or None)
    records5 = records20[: min(args.diag_samples, len(records20))]
    sample_manifest_rows: list[dict[str, Any]] = []

    log("building SparseWorld model/head once for offline get_occ replay")
    cfg, model, checkpoint = build_sparseworld_model(Path(args.repo_root), Path(args.config), Path(args.checkpoint))
    head = get_pts_bbox_head(model)

    variant_lib = get_default_variant_library()
    soft_keys = [
        "V0_hard_baseline",
        "V1_bev_soft_splat_r1",
        "V2_bev_soft_splat_r2",
        "V3_xyz_soft_splat_3x3x3",
        "V4_z_soft_only",
        "V5_topk_nearest_support_aggregate_k1",
        "V5_topk_nearest_support_aggregate_k3",
        "V5_topk_nearest_support_aggregate_k5",
        "V6_distance_weighted_neighbor_aggregate",
    ]
    gate_keys = [
        "G0_baseline_gate",
        "G1_margin_005",
        "G1_margin_010",
        "G1_margin_020",
        "G1_margin_040",
        "G2_entropy_20",
        "G2_entropy_25",
        "G3_near_thr_005",
        "G3_near_thr_010",
        "G3_near_thr_020",
        "G4_future_relax_005",
        "G5_low_density_relax",
        "G6_small_object_relax",
    ]
    class_keys = [
        "C0_baseline_aggregation",
        "C1_small_object_topk_preserve",
        "C1_small_object_topk_preserve_5",
        "C2_small_object_temperature_scaling_12",
        "C2_small_object_temperature_scaling_15",
        "C2_small_object_temperature_scaling_20",
        "C3_dynamic_class_reweight_12",
        "C4_class_balanced_aggregate",
        "C5_small_object_local_soft_splat",
    ]

    # Phase 1 baseline replay equivalence
    log("Phase 1: baseline replay equivalence")
    equivalence_checks: list[dict[str, Any]] = []
    sample0_debug: dict[str, Any] = {}
    with torch.inference_mode():
        for horizon_s in [0, 6]:
            pred_dict = extract_pred_dict_for_horizon(records5[0]["query_artifact"], horizon_s)
            orig_pred, orig_debug = get_occ_debug(head, pred_dict)
            replay = replay_get_occ_variant(head, pred_dict, horizon_s, *variant_lib["V0_hard_baseline"])
            cmp = compare_tensors(orig_pred, replay["output"]["occ_pred"].unsqueeze(0))
            equivalence_checks.append(
                {
                    "sample_index": records5[0]["sample_index"],
                    "horizon_s": horizon_s,
                    **cmp,
                }
            )
            sample0_debug[f"h{horizon_s}"] = tensor_to_cpu(replay)
    saved_pred_checks: list[dict[str, Any]] = []
    with torch.inference_mode():
        for record in records5:
            for horizon_s in horizons:
                pred_dict = extract_pred_dict_for_horizon(record["query_artifact"], horizon_s)
                replay = replay_get_occ_variant(head, pred_dict, horizon_s, *variant_lib["V0_hard_baseline"])
                cmp = compare_tensors(record["pred_temporal"][horizon_s].to(replay["output"]["occ_pred"].device), replay["output"]["occ_pred"])
                saved_pred_checks.append(
                    {
                        "sample_index": record["sample_index"],
                        "horizon_s": horizon_s,
                        **cmp,
                    }
                )
                sample_manifest_rows.append(
                    {
                        "sample_index": record["sample_index"],
                        "sample_token": record["sample_token"],
                        "scene_token": record["scene_token"],
                        "horizon_s": horizon_s,
                        "baseline_status": "success" if cmp["exact_equal"] else "mismatch",
                        "debug_status": "pending",
                        "ablation_status": "pending",
                        "failure_traceback": "",
                    }
                )
    baseline_equivalence = {
        "phase": "SW-4.1 baseline replay",
        "original_get_occ_checks": equivalence_checks,
        "saved_pred_checks": saved_pred_checks,
        "passed": bool(all(x["exact_equal"] for x in equivalence_checks) and all(x["exact_equal"] for x in saved_pred_checks)),
    }
    write_json(REPORTS_DIR / "baseline_replay_equivalence.json", baseline_equivalence)
    save_replay_debug(ARTIFACTS_DIR / "get_occ_replay_debug_sample0.pt", sample0_debug)
    with phase4_log.open("w", encoding="utf-8") as f:
        f.write(json.dumps(baseline_equivalence, indent=2))
    if not baseline_equivalence["passed"]:
        log("baseline replay equivalence failed; aborting SW-4.1")
        return 2

    # Phase 2/3/4 diagnostics on first 5 samples
    log("Phase 2/3/4: exact contributor bottleneck and neighbor leakage diagnostics")
    waterfall_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    sample0_neighbor_details: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for record in records5:
            log(f"diagnostics: sample {record['sample_index']}")
            for horizon_s in horizons:
                pred_dict = extract_pred_dict_for_horizon(record["query_artifact"], horizon_s)
                replay = replay_get_occ_variant(head, pred_dict, horizon_s, *variant_lib["V0_hard_baseline"])
                replay["head"] = head
                gt = record["gt_temporal"][horizon_s].cpu()
                pred = replay["output"]["occ_pred"].detach().cpu()
                gt0 = record["gt_temporal"][0].cpu()
                geometric_mask = build_geometric_mask(head, replay["point_info"]).detach().cpu()
                geometric_cover = sw4.dilate3d(geometric_mask, args.primary_radius)
                masks = sw4.build_temporal_region_masks(gt0, gt, pred, geometric_cover)
                region_order = {
                    "covered_tp": masks["covered_tp"],
                    "covered_false_free": masks["covered_false_free"],
                    "covered_false_occupied": masks["covered_false_occupied"],
                    "small_object_covered_false_free": masks["small_object_covered_false_free"],
                    "new_visible_covered_false_free": masks["new_visible_covered_false_free"],
                    "persistent_covered_tp": masks["persistent_covered_tp"],
                }
                for region_name, mask in region_order.items():
                    coords = sampled_coords(mask, limit=args.sample_cap_per_region, seed=record["sample_index"] * 100 + horizon_s)
                    if coords.numel() == 0:
                        continue
                    stage = summarize_exact_stages(head, coords, gt, pred, replay, args.primary_radius, args.nearest_chunk)
                    stage.update(
                        {
                            "sample_index": record["sample_index"],
                            "sample_token": record["sample_token"],
                            "scene_token": record["scene_token"],
                            "horizon_s": horizon_s,
                            "region_name": region_name,
                        }
                    )
                    waterfall_rows.append(stage)
                    neigh_row, detail = summarize_neighbor_leakage(head, coords, gt, pred, replay, args.nearest_chunk)
                    neigh_row.update(
                        {
                            "sample_index": record["sample_index"],
                            "sample_token": record["sample_token"],
                            "scene_token": record["scene_token"],
                            "horizon_s": horizon_s,
                            "region_name": region_name,
                        }
                    )
                    neighbor_rows.append(neigh_row)
                    if record["sample_index"] == records5[0]["sample_index"] and region_name in {"covered_tp", "covered_false_free"}:
                        sample0_neighbor_details[f"{region_name}_h{horizon_s}"] = detail
                for row in sample_manifest_rows:
                    if int(row["sample_index"]) == int(record["sample_index"]) and int(row["horizon_s"]) == horizon_s:
                        row["debug_status"] = "success"

    write_csv(REPORTS_DIR / "exact_contributor_waterfall.csv", waterfall_rows)
    write_json(REPORTS_DIR / "exact_contributor_waterfall.json", {"rows": waterfall_rows})
    write_csv(REPORTS_DIR / "nearest_support_offset_by_error_type.csv", neighbor_rows)
    write_json(REPORTS_DIR / "nearest_support_offset_by_error_type.json", {"rows": neighbor_rows})
    write_csv(REPORTS_DIR / "neighbor_leakage_analysis.csv", neighbor_rows)
    write_json(REPORTS_DIR / "neighbor_leakage_analysis.json", {"rows": neighbor_rows})

    def region_mean(rows: list[dict[str, Any]], key: str, region: str) -> float:
        vals = [float(r[key]) for r in rows if r["region_name"] == region]
        return float(np.mean(vals)) if vals else 0.0

    waterfall_summary = {
        "covered_ff_support_r2": region_mean(waterfall_rows, "support_r2_ratio", "covered_false_free"),
        "covered_ff_exact_voxel_assignment": region_mean(waterfall_rows, "exact_voxel_assignment_ratio", "covered_false_free"),
        "covered_ff_final_contributor": region_mean(waterfall_rows, "final_contributor_ratio", "covered_false_free"),
        "covered_tp_support_r2": region_mean(waterfall_rows, "support_r2_ratio", "covered_tp"),
        "covered_tp_exact_voxel_assignment": region_mean(waterfall_rows, "exact_voxel_assignment_ratio", "covered_tp"),
        "covered_tp_final_contributor": region_mean(waterfall_rows, "final_contributor_ratio", "covered_tp"),
    }
    write_md(
        REPORTS_DIR / "exact_contributor_waterfall_summary.md",
        "# Exact contributor waterfall summary\n\n"
        f"- covered FF support@r2: {waterfall_summary['covered_ff_support_r2']:.4f}\n"
        f"- covered FF exact voxel assignment: {waterfall_summary['covered_ff_exact_voxel_assignment']:.4f}\n"
        f"- covered FF final contributor: {waterfall_summary['covered_ff_final_contributor']:.4f}\n"
        f"- covered TP support@r2: {waterfall_summary['covered_tp_support_r2']:.4f}\n"
        f"- covered TP exact voxel assignment: {waterfall_summary['covered_tp_exact_voxel_assignment']:.4f}\n"
        f"- covered TP final contributor: {waterfall_summary['covered_tp_final_contributor']:.4f}\n",
    )
    write_md(
        REPORTS_DIR / "neighbor_leakage_summary.md",
        "# Neighbor leakage summary\n\n"
        f"- covered FF same exact voxel ratio: {region_mean(neighbor_rows, 'same_exact_voxel_ratio', 'covered_false_free'):.4f}\n"
        f"- covered FF neighbor contributor ratio: {region_mean(neighbor_rows, 'neighbor_contributor_ratio', 'covered_false_free'):.4f}\n"
        f"- covered FF neighbor false-occupied ratio: {region_mean(neighbor_rows, 'neighbor_false_occupied_ratio', 'covered_false_free'):.4f}\n"
        f"- covered TP same exact voxel ratio: {region_mean(neighbor_rows, 'same_exact_voxel_ratio', 'covered_tp'):.4f}\n",
    )

    plot_metric_panel(
        FIGURES_DIR / "gate_pass_to_contributor_waterfall.png",
        [
            {"variant_name": "covered_tp", "mean_support_r2_ratio": region_mean(waterfall_rows, "support_r2_ratio", "covered_tp"), "mean_exact_voxel_assignment_ratio": region_mean(waterfall_rows, "exact_voxel_assignment_ratio", "covered_tp"), "mean_final_contributor_ratio": region_mean(waterfall_rows, "final_contributor_ratio", "covered_tp")},
            {"variant_name": "covered_false_free", "mean_support_r2_ratio": region_mean(waterfall_rows, "support_r2_ratio", "covered_false_free"), "mean_exact_voxel_assignment_ratio": region_mean(waterfall_rows, "exact_voxel_assignment_ratio", "covered_false_free"), "mean_final_contributor_ratio": region_mean(waterfall_rows, "final_contributor_ratio", "covered_false_free")},
        ],
        [("support_r2_ratio", "support@r2"), ("exact_voxel_assignment_ratio", "exact voxel"), ("final_contributor_ratio", "final contributor")],
        "Waterfall: support -> exact voxel -> contributor",
    )
    plot_metric_panel(
        FIGURES_DIR / "contributor_stage_drop_by_error_type.png",
        [
            {"variant_name": "covered_tp", "mean_gate_pass_support_ratio": region_mean(waterfall_rows, "gate_pass_support_ratio", "covered_tp"), "mean_exact_bev_match_ratio": region_mean(waterfall_rows, "exact_bev_match_ratio", "covered_tp"), "mean_final_contributor_ratio": region_mean(waterfall_rows, "final_contributor_ratio", "covered_tp")},
            {"variant_name": "covered_false_free", "mean_gate_pass_support_ratio": region_mean(waterfall_rows, "gate_pass_support_ratio", "covered_false_free"), "mean_exact_bev_match_ratio": region_mean(waterfall_rows, "exact_bev_match_ratio", "covered_false_free"), "mean_final_contributor_ratio": region_mean(waterfall_rows, "final_contributor_ratio", "covered_false_free")},
            {"variant_name": "covered_false_occupied", "mean_gate_pass_support_ratio": region_mean(waterfall_rows, "gate_pass_support_ratio", "covered_false_occupied"), "mean_exact_bev_match_ratio": region_mean(waterfall_rows, "exact_bev_match_ratio", "covered_false_occupied"), "mean_final_contributor_ratio": region_mean(waterfall_rows, "final_contributor_ratio", "covered_false_occupied")},
        ],
        [("gate_pass_support_ratio", "gate pass"), ("exact_bev_match_ratio", "exact BEV"), ("final_contributor_ratio", "final contributor")],
        "Contributor stage drop by error type",
    )
    plot_metric_panel(
        FIGURES_DIR / "exact_contributor_ratio_vs_radius_coverage.png",
        [
            {"variant_name": "covered_tp", "mean_support_r2_ratio": region_mean(waterfall_rows, "support_r2_ratio", "covered_tp"), "mean_support_r3_ratio": region_mean(waterfall_rows, "support_r3_ratio", "covered_tp"), "mean_support_r5_ratio": region_mean(waterfall_rows, "support_r5_ratio", "covered_tp")},
            {"variant_name": "covered_false_free", "mean_support_r2_ratio": region_mean(waterfall_rows, "support_r2_ratio", "covered_false_free"), "mean_support_r3_ratio": region_mean(waterfall_rows, "support_r3_ratio", "covered_false_free"), "mean_support_r5_ratio": region_mean(waterfall_rows, "support_r5_ratio", "covered_false_free")},
        ],
        [("support_r2_ratio", "support@r2"), ("support_r3_ratio", "support@r3"), ("support_r5_ratio", "support@r5")],
        "Radius coverage vs exact contributor",
    )
    plot_metric_panel(
        FIGURES_DIR / "small_object_waterfall.png",
        [
            {"variant_name": "small_object_FF", "mean_support_r2_ratio": region_mean(waterfall_rows, "support_r2_ratio", "small_object_covered_false_free"), "mean_exact_voxel_assignment_ratio": region_mean(waterfall_rows, "exact_voxel_assignment_ratio", "small_object_covered_false_free"), "mean_final_contributor_ratio": region_mean(waterfall_rows, "final_contributor_ratio", "small_object_covered_false_free")},
        ],
        [("support_r2_ratio", "support@r2"), ("exact_voxel_assignment_ratio", "exact voxel"), ("final_contributor_ratio", "final contributor")],
        "Small-object covered FF waterfall",
    )
    plot_metric_panel(
        FIGURES_DIR / "new_visible_waterfall.png",
        [
            {"variant_name": "new_visible_FF", "mean_support_r2_ratio": region_mean(waterfall_rows, "support_r2_ratio", "new_visible_covered_false_free"), "mean_exact_voxel_assignment_ratio": region_mean(waterfall_rows, "exact_voxel_assignment_ratio", "new_visible_covered_false_free"), "mean_final_contributor_ratio": region_mean(waterfall_rows, "final_contributor_ratio", "new_visible_covered_false_free")},
        ],
        [("support_r2_ratio", "support@r2"), ("exact_voxel_assignment_ratio", "exact voxel"), ("final_contributor_ratio", "final contributor")],
        "New-visible covered FF waterfall",
    )

    if args.save_figures and sample0_neighbor_details:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, name in zip(axes, ["covered_tp_h0", "covered_false_free_h0"]):
            detail = sample0_neighbor_details.get(name)
            if detail is None or detail["dx"].numel() == 0:
                continue
            ax.hist(detail["dx"].numpy(), bins=np.arange(-5, 6) - 0.5, alpha=0.7, label="dx")
            ax.hist(detail["dy"].numpy(), bins=np.arange(-5, 6) - 0.5, alpha=0.5, label="dy")
            ax.set_title(name)
            ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "nearest_support_offset_hist_by_error_type.png", dpi=150)
        plt.close(fig)

        ff = sample0_neighbor_details.get("covered_false_free_h0")
        tp = sample0_neighbor_details.get("covered_tp_h0")
        if ff is not None and tp is not None and ff["dx"].numel() and tp["dx"].numel():
            heat_lim = 5
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, detail, title in zip(axes, [ff, tp], ["FF", "TP"]):
                heat = np.zeros((2 * heat_lim + 1, 2 * heat_lim + 1), dtype=np.float32)
                dx = detail["dx"].numpy().astype(int)
                dy = detail["dy"].numpy().astype(int)
                for x, y in zip(dx, dy):
                    if -heat_lim <= x <= heat_lim and -heat_lim <= y <= heat_lim:
                        heat[y + heat_lim, x + heat_lim] += 1
                ax.imshow(heat, origin="lower")
                ax.set_title(title)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "dx_dy_offset_heatmap_ff_vs_tp.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ff["dz"].numpy(), bins=np.arange(-5, 6) - 0.5, alpha=0.7, label="FF")
            ax.hist(tp["dz"].numpy(), bins=np.arange(-5, 6) - 0.5, alpha=0.7, label="TP")
            ax.legend()
            ax.set_title("dz offset FF vs TP")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "dz_offset_hist_ff_vs_tp.png", dpi=150)
            plt.close(fig)

    # Phase 4 decomposition manifest
    decomp_manifest = {
        "functions": [
            "decode_support_points",
            "compute_gate_mask",
            "assign_points_to_voxels_hard",
            "aggregate_voxel_scores_hard",
            "produce_semantic_occ",
            "replay_get_occ_variant",
        ],
        "variant_library_size": len(variant_lib),
        "debug_artifact": str(ARTIFACTS_DIR / "get_occ_replay_debug_sample0.pt"),
    }
    write_json(REPORTS_DIR / "get_occ_replay_decomposition_manifest.json", decomp_manifest)
    write_json(REPORTS_DIR / "get_occ_hard_mode_equivalence.json", baseline_equivalence)

    # Phase 5/6/7 family ablations on first 5 samples
    log("Phase 5: soft splatting ablations")
    soft_rows = attach_deltas(run_variant_set(head, records5, horizons, soft_keys, variant_lib, "soft"), "V0_hard_baseline")
    soft_agg = aggregate_rows(soft_rows)
    write_csv(REPORTS_DIR / "soft_splat_ablation_metrics.csv", soft_rows)
    write_json(REPORTS_DIR / "soft_splat_ablation_metrics.json", {"rows": soft_rows, "aggregate": soft_agg})
    write_md(REPORTS_DIR / "soft_splat_ablation_summary.md", "# Soft splat ablation summary\n\n" + "\n".join(f"- {row['variant_name']}: ff_delta={row.get('mean_false_free_rate_delta', 0.0):.4f}, fo_delta={row.get('mean_false_occupied_rate_delta', 0.0):.4f}" for row in soft_agg))
    plot_metric_panel(FIGURES_DIR / "soft_splat_ablation_metrics_panel.png", soft_agg, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "Soft splat ablations")

    log("Phase 6: adaptive gate ablations")
    gate_rows = attach_deltas(run_variant_set(head, records5, horizons, gate_keys, variant_lib, "gate"), "G0_baseline_gate")
    gate_agg = aggregate_rows(gate_rows)
    write_csv(REPORTS_DIR / "adaptive_gate_ablation_metrics.csv", gate_rows)
    write_json(REPORTS_DIR / "adaptive_gate_ablation_metrics.json", {"rows": gate_rows, "aggregate": gate_agg})
    write_md(REPORTS_DIR / "adaptive_gate_ablation_summary.md", "# Adaptive gate ablation summary\n\n" + "\n".join(f"- {row['variant_name']}: ff_delta={row.get('mean_false_free_rate_delta', 0.0):.4f}, fo_delta={row.get('mean_false_occupied_rate_delta', 0.0):.4f}" for row in gate_agg))
    plot_metric_panel(FIGURES_DIR / "adaptive_gate_false_free_vs_false_occupied.png", gate_agg, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("covered_ff_contributor_ratio", "covered FF contrib")], "Adaptive gate ablations")

    log("Phase 7: class-aware aggregation ablations")
    class_rows = attach_deltas(run_variant_set(head, records5, horizons, class_keys, variant_lib, "class"), "C0_baseline_aggregation")
    class_agg = aggregate_rows(class_rows)
    write_csv(REPORTS_DIR / "class_aware_aggregation_ablation_metrics.csv", class_rows)
    write_json(REPORTS_DIR / "class_aware_aggregation_ablation_metrics.json", {"rows": class_rows, "aggregate": class_agg})
    write_md(REPORTS_DIR / "class_aware_aggregation_summary.md", "# Class-aware aggregation summary\n\n" + "\n".join(f"- {row['variant_name']}: ff_delta={row.get('mean_false_free_rate_delta', 0.0):.4f}, small_ff_delta={row.get('mean_small_object_false_free_delta', 0.0):.4f}" for row in class_agg))
    plot_metric_panel(FIGURES_DIR / "class_aware_small_object_false_free.png", class_agg, [("small_object_false_free", "small-object FF"), ("false_occupied_rate", "false_occupied"), ("new_visible_recall", "new-visible recall")], "Class-aware aggregation ablations")

    # Phase 8 combined candidates
    best_soft = best_variant_from_family(soft_agg, set(soft_keys))
    best_gate = best_variant_from_family(gate_agg, set(gate_keys))
    best_class = best_variant_from_family(class_agg, set(class_keys))
    log(f"best family variants: soft={best_soft}, gate={best_gate}, class={best_class}")
    combo_lib = {
        "K0_baseline": combine_variant("K0_baseline", "V0_hard_baseline", "G0_baseline_gate", "C0_baseline_aggregation", variant_lib),
        "K1_best_soft_only": combine_variant("K1_best_soft_only", best_soft, "G0_baseline_gate", "C0_baseline_aggregation", variant_lib),
        "K2_best_gate_only": combine_variant("K2_best_gate_only", "V0_hard_baseline", best_gate, "C0_baseline_aggregation", variant_lib),
        "K3_best_class_only": combine_variant("K3_best_class_only", "V0_hard_baseline", "G0_baseline_gate", best_class, variant_lib),
        "K4_soft_plus_gate": combine_variant("K4_soft_plus_gate", best_soft, best_gate, "C0_baseline_aggregation", variant_lib),
        "K5_gate_plus_class": combine_variant("K5_gate_plus_class", "V0_hard_baseline", best_gate, best_class, variant_lib),
        "K6_soft_plus_class": combine_variant("K6_soft_plus_class", best_soft, "G0_baseline_gate", best_class, variant_lib),
        "K7_all_combined": combine_variant("K7_all_combined", best_soft, best_gate, best_class, variant_lib),
    }
    log("Phase 8: combined candidate ablations")
    combo_rows = attach_deltas(run_variant_set(head, records5, horizons, list(combo_lib.keys()), combo_lib, "combined"), "K0_baseline")
    combo_agg = aggregate_rows(combo_rows)
    write_csv(REPORTS_DIR / "combined_repair_candidate_metrics.csv", combo_rows)
    write_json(REPORTS_DIR / "combined_repair_candidate_metrics.json", {"rows": combo_rows, "aggregate": combo_agg})
    plot_metric_panel(FIGURES_DIR / "combined_candidate_metric_panel.png", combo_agg, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "Combined repair candidates")

    def find_row(agg_rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        for row in agg_rows:
            if row["variant_name"] == name:
                return row
        return None

    def candidate_score(row: dict[str, Any]) -> float:
        ff_gain = -float(row.get("mean_false_free_rate_delta", 0.0))
        fo_pen = max(0.0, float(row.get("mean_false_occupied_rate_delta", 0.0)) - 0.01) * 4.0
        occ_iou = float(row.get("mean_occupied_iou_delta", 0.0)) * 2.0
        sem_iou = float(row.get("mean_semantic_miou_delta", 0.0))
        small_gain = -float(row.get("mean_small_object_false_free_delta", 0.0)) * 0.8
        newv_gain = float(row.get("mean_new_visible_recall_delta", 0.0)) * 0.8
        runtime_pen = max(0.0, float(row.get("mean_runtime_ms_delta", 0.0))) / 1000.0
        return ff_gain + occ_iou + sem_iou + small_gain + newv_gain - fo_pen - runtime_pen

    best_combo_candidates = [row for row in combo_agg if row["variant_name"] != "K0_baseline"]
    best_combo_candidates.sort(key=candidate_score, reverse=True)
    best_overall_candidate = best_combo_candidates[0]["variant_name"] if best_combo_candidates else "K0_baseline"
    best_safe_candidate = best_overall_candidate
    best_small_object_candidate = max(best_combo_candidates, key=lambda r: -float(r.get("mean_small_object_false_free_delta", 0.0)))["variant_name"] if best_combo_candidates else best_overall_candidate
    best_new_visible_candidate = max(best_combo_candidates, key=lambda r: r.get("mean_new_visible_recall_delta", 0.0))["variant_name"] if best_combo_candidates else best_overall_candidate
    write_md(
        REPORTS_DIR / "combined_repair_candidate_selection.md",
        "# Combined repair candidate selection\n\n"
        f"- best_overall_candidate: {best_overall_candidate}\n"
        f"- best_safe_candidate: {best_safe_candidate}\n"
        f"- best_small_object_candidate: {best_small_object_candidate}\n"
        f"- best_new_visible_candidate: {best_new_visible_candidate}\n",
    )
    write_json(
        REPORTS_DIR / "combined_repair_candidate_selection.json",
        {
            "best_overall_candidate": best_overall_candidate,
            "best_safe_candidate": best_safe_candidate,
            "best_small_object_candidate": best_small_object_candidate,
            "best_new_visible_candidate": best_new_visible_candidate,
        },
    )

    # Phase 9 20-sample validation
    log("Phase 9: 20-sample lightweight validation of selected candidates")
    validation_keys = ["K0_baseline", best_safe_candidate]
    if best_overall_candidate not in validation_keys:
        validation_keys.append(best_overall_candidate)
    if best_small_object_candidate not in validation_keys:
        validation_keys.append(best_small_object_candidate)
    if best_new_visible_candidate not in validation_keys:
        validation_keys.append(best_new_visible_candidate)
    validation_rows = attach_deltas(run_variant_set(head, records20, horizons, validation_keys, combo_lib, "validation"), "K0_baseline")
    validation_agg = aggregate_rows(validation_rows)
    write_csv(REPORTS_DIR / "best_candidate_20sample_validation.csv", validation_rows)
    write_json(REPORTS_DIR / "best_candidate_20sample_validation.json", {"rows": validation_rows, "aggregate": validation_agg})
    write_md(
        REPORTS_DIR / "best_candidate_20sample_validation_summary.md",
        "# 20-sample validation summary\n\n"
        + "\n".join(
            f"- {row['variant_name']}: ff_delta={row.get('mean_false_free_rate_delta', 0.0):.4f}, fo_delta={row.get('mean_false_occupied_rate_delta', 0.0):.4f}, occ_iou_delta={row.get('mean_occupied_iou_delta', 0.0):.4f}"
            for row in validation_agg
        ),
    )
    plot_metric_panel(FIGURES_DIR / "best_candidate_20sample_metric_panel.png", validation_agg, [("false_free_rate", "false_free"), ("false_occupied_rate", "false_occupied"), ("occupied_iou", "occupied_iou")], "20-sample validation")

    best_validation_row = find_row(validation_agg, best_overall_candidate)
    case, case_reason = select_case(best_validation_row, "combined")
    decision = {
        "case": case,
        "reason": case_reason,
        "best_soft_variant": best_soft,
        "best_gate_variant": best_gate,
        "best_class_variant": best_class,
        "best_overall_candidate": best_overall_candidate,
        "best_safe_candidate": best_safe_candidate,
        "best_small_object_candidate": best_small_object_candidate,
        "best_new_visible_candidate": best_new_visible_candidate,
        "validation_row": best_validation_row,
    }
    write_json(REPORTS_DIR / "sw41_repair_candidate_decision.json", decision)
    write_md(
        REPORTS_DIR / "sw41_repair_candidate_decision.md",
        "# SW-4.1 repair candidate decision\n\n"
        f"- case: {case}\n"
        f"- reason: {case_reason}\n"
        f"- best_soft_variant: {best_soft}\n"
        f"- best_gate_variant: {best_gate}\n"
        f"- best_class_variant: {best_class}\n"
        f"- best_overall_candidate: {best_overall_candidate}\n",
    )

    # Final manifests and report
    write_json(
        REPORTS_DIR / "sw41_run_manifest.json",
        {
            "effective_samples": len(records20),
            "diagnostic_samples": len(records5),
            "effective_horizons": horizons,
            "baseline_replay_passed": baseline_equivalence["passed"],
            "best_soft_variant": best_soft,
            "best_gate_variant": best_gate,
            "best_class_variant": best_class,
            "best_overall_candidate": best_overall_candidate,
            "case": case,
        },
    )
    write_csv(REPORTS_DIR / "sw41_sample_manifest.csv", sample_manifest_rows)

    report = {
        "stage": "SW-4.1",
        "effective_samples": len(records20),
        "diagnostic_samples": len(records5),
        "effective_horizons": horizons,
        "baseline_equivalence": baseline_equivalence,
        "contributor_bottleneck": {
            "covered_ff_support_r2": waterfall_summary["covered_ff_support_r2"],
            "covered_ff_exact_voxel_assignment": waterfall_summary["covered_ff_exact_voxel_assignment"],
            "covered_ff_final_contributor": waterfall_summary["covered_ff_final_contributor"],
            "covered_tp_support_r2": waterfall_summary["covered_tp_support_r2"],
            "covered_tp_exact_voxel_assignment": waterfall_summary["covered_tp_exact_voxel_assignment"],
            "covered_tp_final_contributor": waterfall_summary["covered_tp_final_contributor"],
        },
        "neighbor_leakage": {
            "covered_ff_neighbor_contributor_ratio": region_mean(neighbor_rows, "neighbor_contributor_ratio", "covered_false_free"),
            "covered_ff_same_exact_voxel_ratio": region_mean(neighbor_rows, "same_exact_voxel_ratio", "covered_false_free"),
            "covered_ff_neighbor_false_occupied_ratio": region_mean(neighbor_rows, "neighbor_false_occupied_ratio", "covered_false_free"),
        },
        "best_variants": {
            "soft": best_soft,
            "gate": best_gate,
            "class": best_class,
            "combined": best_overall_candidate,
        },
        "decision": decision,
        "safe_claims": [
            "subset diagnostic repair ablation",
            "no training",
            "not official SparseWorld benchmark",
            "oracle variants diagnostic only",
            "non-oracle candidate only for next-stage patch consideration",
        ],
        "next_unique_action": "Stage SW-4.2 implement best non-oracle repair patch and rerun subset diagnosis" if case in {"R1", "R2", "R3", "R4"} else ("tune conservative version / add false-positive guard" if case == "R5" else ("deeper get_occ instrumentation or training-level approach" if case == "R6" else "do not claim repair; use as diagnostic insight only")),
    }
    write_json(REPORTS_DIR / "stage_sw41_exact_contributor_soft_aggregation_report.json", report)
    write_md(
        REPORTS_DIR / "stage_sw41_exact_contributor_soft_aggregation_report.md",
        "# Stage SW-4.1 exact contributor and soft aggregation repair ablation\n\n"
        "## Executive summary\n\n"
        f"- effective samples: {len(records20)}\n"
        f"- diagnostic samples: {len(records5)}\n"
        f"- horizons: {horizons}\n"
        f"- covered FF support@r2: {waterfall_summary['covered_ff_support_r2']:.4f}\n"
        f"- covered FF exact voxel assignment: {waterfall_summary['covered_ff_exact_voxel_assignment']:.4f}\n"
        f"- covered FF final contributor: {waterfall_summary['covered_ff_final_contributor']:.4f}\n"
        f"- best soft: {best_soft}\n"
        f"- best gate: {best_gate}\n"
        f"- best class-aware: {best_class}\n"
        f"- best combined: {best_overall_candidate}\n"
        f"- decision case: {case}\n"
        f"- next unique action: {report['next_unique_action']}\n\n"
        "## Safe claims\n\n"
        "- subset diagnostic repair ablation\n"
        "- no training\n"
        "- not official SparseWorld benchmark\n"
        "- not deployment-ready\n"
        "- oracle variants diagnostic only\n",
    )

    sw41_log.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("SW-4.1 pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
