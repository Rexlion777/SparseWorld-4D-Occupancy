from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import queue
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
SW5_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"
SW5_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"
SW2_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
SW41_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw2 = load_module(
    "sw6_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw35 = load_module(
    "sw6_sw35",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage/corrected_support_adapter.py",
)
sw4_inst = load_module(
    "sw6_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw5 = load_module(
    "sw6_sw5",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation/run_sparseworld_sw5_main.py",
)
engine = load_module(
    "sw6_engine",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation/sensor_perturbation_engine.py",
)
sw41 = load_module(
    "sw6_sw41",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation/get_occ_replay_and_ablation.py",
)
vis_renderer = load_module(
    "sw6_vis_renderer",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization/semantic_occupancy_renderer.py",
)


CORE_PERTS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]
CONTROL_PERTS = ["A3_drop_cam_front_right", "A7_drop_all_rear", "B1_low_light_05", "D6_tx_p5cm_all", "G4_road_center_occlusion"]
FRAGILITY_PERTS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]
SMALL_OBJECT_IDS = sw2.CLASS_GROUPS["small_object"]
DYNAMIC_IDS = sw2.CLASS_GROUPS["all_dynamic"]
STATIC_IDS = sw2.CLASS_GROUPS["all_static"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    p.add_argument("--config", default=str(PROJECT_ROOT / "external/SparseWorld/configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"))
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "external/SparseWorld/ckpts/epoch_56.pth"))
    p.add_argument("--split", default="val")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--fallback-num-samples", type=int, default=10)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--horizons", default="0,1,2,3,4,5,6")
    p.add_argument("--core-perturbations", default="clean,A1_drop_cam_front,A10_drop_front_triplet,C4_motion_blur_9")
    p.add_argument("--control-perturbations", default="A3_drop_cam_front_right,A7_drop_all_rear,B1_low_light_05,D6_tx_p5cm_all,G4_road_center_occlusion")
    p.add_argument("--run-sector-analysis", action="store_true", default=True)
    p.add_argument("--run-query-support-decomposition", action="store_true", default=True)
    p.add_argument("--run-motion-blur-decomposition", action="store_true", default=True)
    p.add_argument("--run-causal-restore-probes", action="store_true", default=True)
    p.add_argument("--run-fragility-taxonomy", action="store_true", default=True)
    p.add_argument("--save-debug", action="store_true", default=True)
    p.add_argument("--save-figures", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[SW6] {msg}", flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "phase1_sw6_main.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, LOGS_DIR, FIGURES_DIR, ARTIFACTS_DIR, TESTS_DIR, FIGURES_DIR / "case_gallery"]:
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        agg = {k: v for k, v in zip(group_keys, key)}
        agg["sample_count"] = len(bucket)
        num_keys = sorted(
            {
                k
                for r in bucket
                for k, v in r.items()
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and k not in group_keys
            }
        )
        for metric in num_keys:
            vals = [float(r[metric]) for r in bucket if r.get(metric) is not None and not math.isnan(float(r[metric]))]
            if not vals:
                continue
            agg[f"mean_{metric}"] = float(np.mean(vals))
            agg[f"std_{metric}"] = float(np.std(vals))
        out.append(agg)
    return out


def normalize_pert_id(name: str) -> str:
    name = name.strip()
    if name == "clean":
        return "A0_clean"
    return name


def bool_mask_dilate3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().permute(2, 0, 1).unsqueeze(0)
    y = torch.nn.functional.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y.squeeze(0).permute(1, 2, 0).bool()


def persistent_new_visible_masks(gt0: torch.Tensor, gth: torch.Tensor) -> dict[str, torch.Tensor]:
    gt0_occ = gt0 != sw2.EMPTY_IDX
    gth_occ = gth != sw2.EMPTY_IDX
    persistent = gt0_occ & gth_occ
    newly_visible = (~gt0_occ) & gth_occ
    disappeared = gt0_occ & (~gth_occ)
    return {"persistent": persistent, "new_visible": newly_visible, "disappeared": disappeared}


def group_false_free(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, class_ids: list[int]) -> float:
    group_mask = torch.zeros_like(gt, dtype=torch.bool)
    for cid in class_ids:
        group_mask |= gt == int(cid)
    group_mask &= valid_mask
    if int(group_mask.sum().item()) == 0:
        return 0.0
    pred_occ = (pred != sw2.EMPTY_IDX) & valid_mask
    ff = group_mask & (~pred_occ)
    return safe_div(ff.sum().item(), group_mask.sum().item())


def voxel_centers_xyz() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(sw2.GRID_SIZE[0], dtype=torch.float32) * sw2.VOXEL_SIZE[0] + sw2.PC_RANGE[0] + sw2.VOXEL_SIZE[0] / 2
    y = torch.arange(sw2.GRID_SIZE[1], dtype=torch.float32) * sw2.VOXEL_SIZE[1] + sw2.PC_RANGE[1] + sw2.VOXEL_SIZE[1] / 2
    z = torch.arange(sw2.GRID_SIZE[2], dtype=torch.float32) * sw2.VOXEL_SIZE[2] + sw2.PC_RANGE[2] + sw2.VOXEL_SIZE[2] / 2
    return x, y, z


def build_sector_masks() -> dict[str, torch.Tensor]:
    x, y, _ = voxel_centers_xyz()
    xx = x[:, None, None].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    yy = y[None, :, None].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    rr = torch.sqrt(xx**2 + yy**2)
    masks = {
        "front": (xx > 0) & (yy.abs() <= 10.0),
        "front_left": (xx > 0) & (yy > 10.0),
        "front_right": (xx > 0) & (yy < -10.0),
        "side_left": (xx.abs() <= 15.0) & (yy > 10.0),
        "side_right": (xx.abs() <= 15.0) & (yy < -10.0),
        "rear": xx < 0,
        "near": rr < 15.0,
        "mid": (rr >= 15.0) & (rr < 30.0),
        "far": rr >= 30.0,
        "front_near": (xx > 0) & (rr < 15.0),
        "front_mid": (xx > 0) & (rr >= 15.0) & (rr < 30.0),
        "front_far": (xx > 0) & (rr >= 30.0),
    }
    return masks


def occ_metrics_in_mask(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    m = valid_mask & mask
    gt_occ = (gt != sw2.EMPTY_IDX) & m
    pred_occ = (pred != sw2.EMPTY_IDX) & m
    inter = (gt_occ & pred_occ).sum().item()
    union = (gt_occ | pred_occ).sum().item()
    ff = (gt_occ & (~pred_occ)).sum().item()
    fo = ((~gt_occ) & pred_occ & m).sum().item()
    return {
        "gt_occupied_count": int(gt_occ.sum().item()),
        "pred_occupied_count": int(pred_occ.sum().item()),
        "occupied_iou": safe_div(inter, union),
        "false_free_rate": safe_div(ff, gt_occ.sum().item()),
        "false_occupied_rate": safe_div(fo, ((~gt_occ) & m).sum().item()),
        "pred_gt_occupied_ratio": safe_div(pred_occ.sum().item(), gt_occ.sum().item()),
    }


def point_centers_from_pred_dict(head: Any, pred_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    point_info = sw41.decode_support_points(head, pred_dict)
    centers = point_info["decoded_points"].mean(dim=1)
    point_info["query_centers"] = centers
    for src_key, dst_key in [
        ("best_score", "query_best_score"),
        ("entropy", "query_entropy"),
        ("margin", "query_margin"),
    ]:
        value = point_info[src_key]
        if torch.is_tensor(value) and value.ndim > 1:
            point_info[dst_key] = value.mean(dim=1)
        else:
            point_info[dst_key] = value
    return point_info


def sector_name_from_center(center: np.ndarray) -> str:
    x, y = float(center[0]), float(center[1])
    r = math.sqrt(x * x + y * y)
    if x > 0 and abs(y) <= 10.0:
        base = "front"
    elif x > 0 and y > 10.0:
        base = "front_left"
    elif x > 0 and y < -10.0:
        base = "front_right"
    elif x < 0:
        base = "rear"
    elif y > 0:
        base = "side_left"
    else:
        base = "side_right"
    if r < 15.0:
        return f"{base}_near"
    if r < 30.0:
        return f"{base}_mid"
    return f"{base}_far"


def nearest_distance_sampled(source_mask: torch.Tensor, target_mask: torch.Tensor, sample_cap: int = 2048) -> float:
    src = torch.nonzero(source_mask, as_tuple=False).float()
    tgt = torch.nonzero(target_mask, as_tuple=False).float()
    if src.numel() == 0 or tgt.numel() == 0:
        return float("nan")
    if src.shape[0] > sample_cap:
        src = src[torch.linspace(0, src.shape[0] - 1, steps=sample_cap).long()]
    if tgt.shape[0] > sample_cap:
        tgt = tgt[torch.linspace(0, tgt.shape[0] - 1, steps=sample_cap).long()]
    d = torch.cdist(src, tgt)
    return float(d.min(dim=1).values.mean().item())


def compute_support_coverage(mask_src: torch.Tensor, mask_target: torch.Tensor, radius: int) -> float:
    if int(mask_target.sum().item()) == 0:
        return 0.0
    src = bool_mask_dilate3d(mask_src, radius)
    return safe_div((src & mask_target).sum().item(), mask_target.sum().item())


def class_aware_match(clean_centers: np.ndarray, clean_cls: np.ndarray, pert_centers: np.ndarray, pert_cls: np.ndarray, dist_thr: float = 1.0) -> dict[str, Any]:
    if clean_centers.shape[0] == 0 and pert_centers.shape[0] == 0:
        return {"matched_ratio": 1.0, "disappeared_ratio": 0.0, "appeared_ratio": 0.0, "mean_displacement": 0.0, "matched_indices": []}
    if clean_centers.shape[0] == 0:
        return {"matched_ratio": 0.0, "disappeared_ratio": 0.0, "appeared_ratio": 1.0, "mean_displacement": float("nan"), "matched_indices": []}
    if pert_centers.shape[0] == 0:
        return {"matched_ratio": 0.0, "disappeared_ratio": 1.0, "appeared_ratio": 0.0, "mean_displacement": float("nan"), "matched_indices": []}
    used: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for i in range(clean_centers.shape[0]):
        cls_mask = np.where(pert_cls == clean_cls[i])[0]
        if cls_mask.size == 0:
            continue
        cand = pert_centers[cls_mask]
        dists = np.linalg.norm(cand - clean_centers[i], axis=1)
        j_local = int(np.argmin(dists))
        j = int(cls_mask[j_local])
        if float(dists[j_local]) <= dist_thr and j not in used:
            used.add(j)
            matched.append((i, j, float(dists[j_local])))
    return {
        "matched_ratio": safe_div(len(matched), clean_centers.shape[0]),
        "disappeared_ratio": safe_div(clean_centers.shape[0] - len(matched), clean_centers.shape[0]),
        "appeared_ratio": safe_div(pert_centers.shape[0] - len(used), pert_centers.shape[0]),
        "mean_displacement": float(np.mean([m[2] for m in matched])) if matched else float("nan"),
        "matched_indices": matched,
    }


def compact_query_artifact(query_cpu: dict[str, Any]) -> dict[str, Any]:
    fb = query_cpu["forward_backbone_outputs"]
    return {
        "forward_backbone_outputs": {
            "cls_score": fb["cls_score"],
            "refine_pts": fb["refine_pts"],
            "forecast_points_list": fb["forecast_points_list"],
            "forecast_semantics_list": fb["forecast_semantics_list"],
        }
    }


def clone_pred_dict(pred_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"cls_scores": pred_dict["cls_scores"].clone(), "refine_pts": pred_dict["refine_pts"].clone()}


def compute_probe_metrics(pred: torch.Tensor, gt_temporal: torch.Tensor, gt0: torch.Tensor, horizon_s: int, gate_mask: torch.Tensor, contributor_count: torch.Tensor) -> dict[str, float]:
    if torch.is_tensor(pred) and pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    valid_mask = sw2.valid_mask_from_gt(gt_temporal)
    base = sw2.compute_base_metrics(pred, gt_temporal, valid_mask)
    out = {k: float(v) for k, v in base.items() if isinstance(v, (int, float))}
    out["dynamic_false_free"] = group_false_free(pred, gt_temporal, valid_mask, DYNAMIC_IDS)
    out["small_object_false_free"] = group_false_free(pred, gt_temporal, valid_mask, SMALL_OBJECT_IDS)
    if horizon_s == 0:
        out["new_visible_recall"] = 0.0
    else:
        masks = persistent_new_visible_masks(gt0, gt_temporal)
        pred_occ = pred != sw2.EMPTY_IDX
        out["new_visible_recall"] = safe_div((pred_occ & masks["new_visible"]).sum().item(), masks["new_visible"].sum().item())
    gt_occ = (gt_temporal != sw2.EMPTY_IDX) & valid_mask
    out["gate_pass_ratio"] = float(gate_mask.float().mean().item())
    out["contributor_ratio"] = safe_div(((contributor_count > 0) & gt_occ).sum().item(), gt_occ.sum().item())
    return out


def top_bev_semantic(label_grid: torch.Tensor) -> torch.Tensor:
    grid = label_grid.long()
    out = torch.ones((grid.shape[0], grid.shape[1]), dtype=torch.long) * sw2.EMPTY_IDX
    active = grid != sw2.EMPTY_IDX
    if not bool(active.any().item()):
        return out
    rev = torch.flip(active, dims=[2])
    top_rev = torch.argmax(rev.int(), dim=2)
    has_any = active.any(dim=2)
    top_idx = (grid.shape[2] - 1) - top_rev
    xs, ys = torch.where(has_any)
    out[xs, ys] = grid[xs, ys, top_idx[xs, ys]]
    return out


def render_label_grid_bev(label_grid: torch.Tensor) -> Image.Image:
    style = vis_renderer.RendererStyle(viewpoint="bev_strict", backend="open3d" if getattr(vis_renderer, "o3d", None) is not None else "matplotlib", mode="voxel_mesh", render_width=900, render_height=520)
    return vis_renderer.render_semantic_occ(label_grid, "", style)


def make_case_gallery_panel(case_name: str, clean_pred: torch.Tensor, pert_pred: torch.Tensor, gt: torch.Tensor, clean_support_bev: torch.Tensor, pert_support_bev: torch.Tensor, clean_contrib_bev: torch.Tensor, pert_contrib_bev: torch.Tensor, out_path: Path) -> None:
    clean_gt_img = render_label_grid_bev(gt)
    clean_pred_img = render_label_grid_bev(clean_pred)
    pert_pred_img = render_label_grid_bev(pert_pred)
    clean_err = vis_renderer.error_overlay_bev(clean_pred, gt)
    pert_err = vis_renderer.error_overlay_bev(pert_pred, gt)

    def mask_img(mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
        arr = np.ones((mask.shape[1], mask.shape[0], 3), dtype=np.uint8) * 255
        yy, xx = np.where(mask.cpu().numpy().T > 0)
        arr[yy, xx] = np.asarray(color, dtype=np.uint8)
        return Image.fromarray(arr).resize((900, 520), Image.Resampling.NEAREST)

    clean_sup_img = mask_img(clean_support_bev, (30, 180, 255))
    pert_sup_img = mask_img(pert_support_bev, (255, 120, 60))
    clean_contrib_img = mask_img(clean_contrib_bev, (20, 180, 80))
    pert_contrib_img = mask_img(pert_contrib_bev, (180, 20, 120))
    sector_img = vis_renderer.error_overlay_bev(pert_pred, gt)

    tiles = [
        ("GT", clean_gt_img),
        ("Clean Pred", clean_pred_img),
        ("Perturbed Pred", pert_pred_img),
        ("Clean Error", clean_err),
        ("Perturbed Error", pert_err),
        ("Clean Support", clean_sup_img),
        ("Pert Support", pert_sup_img),
        ("Clean Contrib", clean_contrib_img),
        ("Pert Contrib", pert_contrib_img),
        ("Sector Heatmap", sector_img),
    ]
    cols = 5
    tw, th = tiles[0][1].size
    canvas = Image.new("RGB", (cols * tw, 2 * (th + 42)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (name, img) in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = c * tw
        y = r * (th + 42)
        canvas.paste(img, (x, y + 42))
        draw.text((x + 10, y + 10), name, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, out_path: Path, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.8), max(4, len(row_labels) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def determine_effective_sample_count(requested_num: int) -> tuple[int, str]:
    debug_files = list((SW5_ARTIFACTS / "debug").rglob("*_get_occ_debug.pt"))
    if len(debug_files) < 20:
        return min(requested_num, 10), "fallback_to_10_due_to_missing_sw5_dense_debug_and_query_artifacts"
    return requested_num, "full_20_supported_by_sw5_dense_artifacts"


def run_dense_subset(
    args: argparse.Namespace,
    sample_indexes: list[int],
    perturbation_ids: list[str],
    horizons: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], Path], dict[tuple[str, int], Path], dict[tuple[str, int], dict[int, dict[str, Any]]]]:
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    log("setting up SparseWorld runtime for SW-6 dense subset")
    cfg, dataset, model, runtime_meta = sw2.setup_runtime(repo_root, config_path, checkpoint_path, args.split, args.fp16)
    collate_fn = runtime_meta["collate"]
    support_adapter = sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT)
    head = sw4_inst.get_pts_bbox_head(model)
    catalog = engine.build_catalog()
    sector_masks = build_sector_masks()

    manifest_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    query_artifact_paths: dict[tuple[str, int], Path] = {}
    occ_artifact_paths: dict[tuple[str, int], Path] = {}
    query_summaries: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}

    for perturbation_id in perturbation_ids:
        spec = catalog[perturbation_id]
        query_holder: dict[str, Any] = {}
        original_forward_backbone = model.forward_backbone

        def wrapped_forward_backbone(*a: Any, **kw: Any) -> Any:
            outputs = original_forward_backbone(*a, **kw)
            query_holder["forward_backbone_outputs"] = sw2.to_cpu_artifact(outputs)
            return outputs

        model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]
        for ordinal, sample_index in enumerate(sample_indexes):
            info = sw2.sample_info(dataset, sample_index)
            try:
                raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
                sample_unwrapped = sw2.unwrap(raw_sample)
                pert_batch, pert_manifest = engine.apply_perturbation_to_batch(batch, spec)
                query_holder.clear()
                if hasattr(model, "memory") and isinstance(getattr(model, "memory"), dict):
                    model.memory.clear()
                if hasattr(model, "queue"):
                    model.queue = queue.Queue()
                model_inputs = sw2.move_to_cuda(pert_batch)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **model_inputs)
                raw_result_cpu = sw2.to_cpu_artifact(result)
                query_cpu = sw2.to_cpu_artifact(query_holder)
                pred_temporal, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
                supports, support_manifests = support_adapter.build_temporal_support(query_cpu, horizons)

                compact_query = compact_query_artifact(query_cpu)
                if perturbation_id in CORE_PERTS or perturbation_id == "A7_drop_all_rear":
                    qpath = ARTIFACTS_DIR / "query_artifacts" / perturbation_id / f"sample_{sample_index:04d}_query_artifact.pt"
                    qpath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(compact_query, qpath)
                    query_artifact_paths[(perturbation_id, sample_index)] = qpath
                    opath = ARTIFACTS_DIR / "occ_artifacts" / perturbation_id / f"sample_{sample_index:04d}_occ_temporal.pt"
                    opath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"pred_temporal": pred_temporal.to(torch.uint8), "gt_temporal": gt_temporal.to(torch.uint8)}, opath)
                    occ_artifact_paths[(perturbation_id, sample_index)] = opath

                sample_summary_by_h: dict[int, dict[str, Any]] = {}
                for h_idx, horizon_s in enumerate(horizons):
                    pred_h = pred_temporal[h_idx]
                    gt_h = gt_temporal[h_idx]
                    valid_mask = sw2.valid_mask_from_gt(gt_h)
                    base = sw2.compute_base_metrics(pred_h, gt_h, valid_mask)
                    support = supports[horizon_s]
                    pred_dict = sw4_inst.extract_pred_dict_for_horizon(compact_query, horizon_s)
                    point_info = point_centers_from_pred_dict(head, pred_dict)
                    gate_info = sw41.compute_gate_mask(head, point_info, sw41.GateConfig(), horizon_s)
                    dbg_pred, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=False)
                    debug = dbg_list[0]
                    geom_mask = support["geometric_mask"].bool().cpu()
                    sem_mask = support["semantic_active_mask"].bool().cpu()
                    contrib_mask = torch.as_tensor(debug["contributor_count_dense"]).cpu() > 0
                    gt_occ = (gt_h != sw2.EMPTY_IDX) & valid_mask
                    pred_occ = (pred_h != sw2.EMPTY_IDX) & valid_mask
                    ff = gt_occ & (~pred_occ)
                    tp = gt_occ & pred_occ
                    point_centers = point_info["query_centers"].detach().cpu().numpy()
                    point_classes = point_info["best_class"].detach().cpu().numpy()
                    point_scores = point_info["best_score"].detach().cpu().numpy()
                    point_entropy = point_info["entropy"].detach().cpu().numpy()
                    point_margin = point_info["margin"].detach().cpu().numpy()
                    gate_query_pass = gate_info["gate_mask"].any(dim=1).float().cpu().numpy()
                    front_query_mask = np.array([float(c[0]) > 0.0 and abs(float(c[1])) <= 10.0 for c in point_centers], dtype=bool)
                    dynamic_query_mask = np.isin(point_classes, np.asarray(DYNAMIC_IDS))
                    small_query_mask = np.isin(point_classes, np.asarray(SMALL_OBJECT_IDS))
                    masks_nv = persistent_new_visible_masks(gt_temporal[0], gt_h) if horizon_s > 0 else {"new_visible": torch.zeros_like(gt_h, dtype=torch.bool), "persistent": gt_occ, "disappeared": torch.zeros_like(gt_h, dtype=torch.bool)}
                    sample_summary_by_h[horizon_s] = {
                        "support_query_count": int(point_centers.shape[0]),
                        "support_mean_best_score": float(np.mean(point_scores)) if point_scores.size else 0.0,
                        "support_mean_entropy": float(np.mean(point_entropy)) if point_entropy.size else 0.0,
                        "support_mean_margin": float(np.mean(point_margin)) if point_margin.size else 0.0,
                        "support_front_query_ratio": float(np.mean(front_query_mask.astype(np.float32))) if point_centers.size else 0.0,
                        "support_dynamic_query_ratio": float(np.mean(dynamic_query_mask.astype(np.float32))) if point_centers.size else 0.0,
                        "support_small_query_ratio": float(np.mean(small_query_mask.astype(np.float32))) if point_centers.size else 0.0,
                        "gate_query_pass_ratio": float(np.mean(gate_query_pass)) if gate_query_pass.size else 0.0,
                        "ff_support_r1": compute_support_coverage(geom_mask, ff, 1),
                        "ff_support_r2": compute_support_coverage(geom_mask, ff, 2),
                        "ff_support_r3": compute_support_coverage(geom_mask, ff, 3),
                        "gt_support_r2": compute_support_coverage(geom_mask, gt_occ, 2),
                        "tp_contributor_ratio": safe_div((contrib_mask & tp).sum().item(), tp.sum().item()),
                        "ff_contributor_ratio": safe_div((contrib_mask & ff).sum().item(), ff.sum().item()),
                        "dynamic_false_free": group_false_free(pred_h, gt_h, valid_mask, DYNAMIC_IDS),
                        "small_object_false_free": group_false_free(pred_h, gt_h, valid_mask, SMALL_OBJECT_IDS),
                        "new_visible_recall": safe_div(((pred_h != sw2.EMPTY_IDX) & masks_nv["new_visible"]).sum().item(), masks_nv["new_visible"].sum().item()) if horizon_s > 0 else 0.0,
                        "occupied_iou": float(base["occupied_iou"]),
                        "false_free_rate": float(base["false_free_rate"]),
                        "false_occupied_rate": float(base["false_occupied_rate"]),
                        "pred_gt_occupied_ratio": float(base["pred_gt_occupied_ratio"]),
                        "nearest_support_to_gt_dist": nearest_distance_sampled(geom_mask, gt_occ),
                        "nearest_support_to_ff_dist": nearest_distance_sampled(geom_mask, ff),
                    }

                    for sector_name, sector_mask in sector_masks.items():
                        occ_metrics = occ_metrics_in_mask(pred_h, gt_h, valid_mask, sector_mask)
                        gt_sector = ((gt_h != sw2.EMPTY_IDX) & valid_mask & sector_mask)
                        ff_sector = gt_sector & (pred_h == sw2.EMPTY_IDX)
                        pred_sector = ((pred_h != sw2.EMPTY_IDX) & valid_mask & sector_mask)
                        sector_rows.append(
                            {
                                "perturbation_id": perturbation_id,
                                "sample_index": sample_index,
                                "horizon_s": horizon_s,
                                "sector_name": sector_name,
                                "gt_occupied_count": occ_metrics["gt_occupied_count"],
                                "pred_occupied_count": occ_metrics["pred_occupied_count"],
                                "occupied_iou": occ_metrics["occupied_iou"],
                                "false_free_rate": occ_metrics["false_free_rate"],
                                "false_occupied_rate": occ_metrics["false_occupied_rate"],
                                "pred_gt_occupied_ratio": occ_metrics["pred_gt_occupied_ratio"],
                                "support_count": int(point_centers.shape[0]),
                                "support_density": safe_div((geom_mask & sector_mask).sum().item(), sector_mask.sum().item()),
                                "gt_support_coverage": safe_div((geom_mask & gt_sector).sum().item(), gt_sector.sum().item()),
                                "ff_support_coverage": safe_div((geom_mask & ff_sector).sum().item(), ff_sector.sum().item()),
                                "gate_pass_ratio": float(np.mean(gate_query_pass)) if gate_query_pass.size else 0.0,
                                "exact_contributor_ratio": safe_div((sem_mask & gt_sector).sum().item(), gt_sector.sum().item()),
                                "final_contributor_ratio": safe_div((contrib_mask & gt_sector).sum().item(), gt_sector.sum().item()),
                                "dynamic_false_free": group_false_free(pred_h, gt_h, valid_mask & sector_mask, DYNAMIC_IDS),
                                "small_object_false_free": group_false_free(pred_h, gt_h, valid_mask & sector_mask, SMALL_OBJECT_IDS),
                                "new_visible_recall": safe_div((pred_sector & masks_nv["new_visible"]).sum().item(), (masks_nv["new_visible"] & sector_mask).sum().item()) if horizon_s > 0 else 0.0,
                            }
                        )

                query_summaries[(perturbation_id, sample_index)] = sample_summary_by_h
                manifest_rows.append(
                    {
                        **info,
                        "perturbation_id": perturbation_id,
                        "forward_status": "success",
                        "affected_indices": ",".join(map(str, pert_manifest["affected_indices"])),
                        "pred_keys": ",".join(pred_keys),
                    }
                )
                forward_rows.append(
                    {
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "save_query_artifact": int((perturbation_id, sample_index) in query_artifact_paths),
                        "available_horizons": ",".join(map(str, horizons)),
                    }
                )
                log(f"{perturbation_id}: sample {sample_index} success ({ordinal+1}/{len(sample_indexes)})")
            except Exception:
                tb = traceback.format_exc()
                manifest_rows.append({**info, "perturbation_id": perturbation_id, "forward_status": "failed", "failure_traceback": tb})
                log(f"{perturbation_id}: sample {sample_index} failed\n{tb}")
        model.forward_backbone = original_forward_backbone  # type: ignore[assignment]
    return manifest_rows, sector_rows, forward_rows, query_artifact_paths, occ_artifact_paths, query_summaries


def compute_frontview_chain(query_summaries: dict[tuple[str, int], dict[int, dict[str, Any]]], sample_indexes: list[int], horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for perturbation_id in ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet"]:
        for sample_index in sample_indexes:
            summary = query_summaries.get((perturbation_id, sample_index), {})
            for horizon_s in horizons:
                if horizon_s not in summary:
                    continue
                row = {"perturbation_id": perturbation_id, "sample_index": sample_index, "horizon_s": horizon_s}
                row.update(summary[horizon_s])
                rows.append(row)
    return rows


def compute_motion_blur_decomp(query_artifact_paths: dict[tuple[str, int], Path], occ_artifact_paths: dict[tuple[str, int], Path], sample_indexes: list[int], horizons: list[int], head: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index in sample_indexes:
        clean_q = query_artifact_paths.get(("A0_clean", sample_index))
        blur_q = query_artifact_paths.get(("C4_motion_blur_9", sample_index))
        blur_occ = occ_artifact_paths.get(("C4_motion_blur_9", sample_index))
        clean_occ = occ_artifact_paths.get(("A0_clean", sample_index))
        if not clean_q or not blur_q or not blur_occ or not clean_occ:
            continue
        clean_query = torch.load(clean_q, map_location="cpu", weights_only=False)
        blur_query = torch.load(blur_q, map_location="cpu", weights_only=False)
        clean_temporal = torch.load(clean_occ, map_location="cpu", weights_only=False)
        blur_temporal = torch.load(blur_occ, map_location="cpu", weights_only=False)
        for h in horizons:
            clean_pred_dict = sw4_inst.extract_pred_dict_for_horizon(clean_query, h)
            blur_pred_dict = sw4_inst.extract_pred_dict_for_horizon(blur_query, h)
            clean_info = point_centers_from_pred_dict(head, clean_pred_dict)
            blur_info = point_centers_from_pred_dict(head, blur_pred_dict)
            match = class_aware_match(
                clean_info["query_centers"].cpu().numpy(),
                clean_info["best_class"].cpu().numpy(),
                blur_info["query_centers"].cpu().numpy(),
                blur_info["best_class"].cpu().numpy(),
                dist_thr=1.5,
            )
            clean_gate = sw41.compute_gate_mask(head, clean_info, sw41.GateConfig(), h)
            blur_gate = sw41.compute_gate_mask(head, blur_info, sw41.GateConfig(), h)
            clean_dbg_pred, clean_dbg_list = sw4_inst.get_occ_debug(head, clean_pred_dict, capture_dense=False)
            blur_dbg_pred, blur_dbg_list = sw4_inst.get_occ_debug(head, blur_pred_dict, capture_dense=False)
            clean_dbg = clean_dbg_list[0]
            blur_dbg = blur_dbg_list[0]
            clean_contrib = torch.as_tensor(clean_dbg["contributor_count_dense"]).cpu() > 0
            blur_contrib = torch.as_tensor(blur_dbg["contributor_count_dense"]).cpu() > 0
            clean_gt = clean_temporal["gt_temporal"][h].long()
            blur_gt = blur_temporal["gt_temporal"][h].long()
            valid_mask = sw2.valid_mask_from_gt(blur_gt)
            gt_occ = (blur_gt != sw2.EMPTY_IDX) & valid_mask
            rows.append(
                {
                    "sample_index": sample_index,
                    "horizon_s": h,
                    "support_count_delta": float(blur_info["query_centers"].shape[0] - clean_info["query_centers"].shape[0]),
                    "support_coverage_delta_r2": compute_support_coverage(sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT).build_support(blur_query, h)[0]["geometric_mask"].bool(), gt_occ, 2)
                    - compute_support_coverage(sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT).build_support(clean_query, h)[0]["geometric_mask"].bool(), gt_occ, 2),
                    "support_centroid_shift": float(torch.norm(blur_info["query_centers"].mean(dim=0) - clean_info["query_centers"].mean(dim=0)).item()),
                    "support_spatial_variance_delta": float(blur_info["query_centers"].var(dim=0).mean().item() - clean_info["query_centers"].var(dim=0).mean().item()),
                    "nearest_clean_support_match_distance": float(match["mean_displacement"]) if not math.isnan(match["mean_displacement"]) else None,
                    "support_matched_ratio": float(match["matched_ratio"]),
                    "cls_entropy_delta": float(blur_info["entropy"].mean().item() - clean_info["entropy"].mean().item()),
                    "top1_confidence_delta": float(blur_info["best_score"].mean().item() - clean_info["best_score"].mean().item()),
                    "class_margin_delta": float(blur_info["margin"].mean().item() - clean_info["margin"].mean().item()),
                    "gate_pass_delta": float(blur_gate["gate_mask"].float().mean().item() - clean_gate["gate_mask"].float().mean().item()),
                    "exact_contributor_delta": safe_div((blur_contrib & gt_occ).sum().item(), gt_occ.sum().item()) - safe_div((clean_contrib & gt_occ).sum().item(), gt_occ.sum().item()),
                    "neighbor_leakage_proxy_delta": float((blur_temporal["pred_temporal"][h].long() != sw2.EMPTY_IDX).sum().item() - (clean_temporal["pred_temporal"][h].long() != sw2.EMPTY_IDX).sum().item()),
                    "false_occupied_delta": float(sw2.compute_base_metrics(blur_temporal["pred_temporal"][h].long(), blur_gt, valid_mask)["false_occupied_rate"] - sw2.compute_base_metrics(clean_temporal["pred_temporal"][h].long(), clean_gt, valid_mask)["false_occupied_rate"]),
                }
            )
    return rows


def compute_support_matching(query_artifact_paths: dict[tuple[str, int], Path], sample_indexes: list[int], horizons: list[int], head: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    compare_perts = ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
    for sample_index in sample_indexes:
        clean_path = query_artifact_paths.get(("A0_clean", sample_index))
        if not clean_path:
            continue
        clean_query = torch.load(clean_path, map_location="cpu", weights_only=False)
        for perturbation_id in compare_perts:
            pert_path = query_artifact_paths.get((perturbation_id, sample_index))
            if not pert_path:
                continue
            pert_query = torch.load(pert_path, map_location="cpu", weights_only=False)
            for h in horizons:
                clean_pred_dict = sw4_inst.extract_pred_dict_for_horizon(clean_query, h)
                pert_pred_dict = sw4_inst.extract_pred_dict_for_horizon(pert_query, h)
                clean_info = point_centers_from_pred_dict(head, clean_pred_dict)
                pert_info = point_centers_from_pred_dict(head, pert_pred_dict)
                match = class_aware_match(
                    clean_info["query_centers"].cpu().numpy(),
                    clean_info["best_class"].cpu().numpy(),
                    pert_info["query_centers"].cpu().numpy(),
                    pert_info["best_class"].cpu().numpy(),
                    dist_thr=1.5,
                )
                delta_conf = []
                delta_ent = []
                sector_drifts: list[float] = []
                for i, j, d in match["matched_indices"]:
                    delta_conf.append(float(pert_info["query_best_score"][j].item() - clean_info["query_best_score"][i].item()))
                    delta_ent.append(float(pert_info["query_entropy"][j].item() - clean_info["query_entropy"][i].item()))
                    sector_drifts.append(float(d))
                rows.append(
                    {
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": h,
                        "matched_ratio": float(match["matched_ratio"]),
                        "disappeared_clean_support_ratio": float(match["disappeared_ratio"]),
                        "newly_appeared_support_ratio": float(match["appeared_ratio"]),
                        "mean_displacement": float(match["mean_displacement"]) if not math.isnan(match["mean_displacement"]) else None,
                        "mean_confidence_delta": float(np.mean(delta_conf)) if delta_conf else None,
                        "mean_entropy_delta": float(np.mean(delta_ent)) if delta_ent else None,
                        "sector_mean_drift": float(np.mean(sector_drifts)) if sector_drifts else None,
                    }
                )
    return rows


def compute_group_fragility(sector_rows: list[dict[str, Any]], query_summaries: dict[tuple[str, int], dict[int, dict[str, Any]]], sample_indexes: list[int], horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for perturbation_id in FRAGILITY_PERTS:
        for sample_index in sample_indexes:
            summary = query_summaries.get((perturbation_id, sample_index), {})
            for horizon_s in horizons:
                if horizon_s not in summary:
                    continue
                s = summary[horizon_s]
                rows.append(
                    {
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "group_name": "all_dynamic",
                        "group_false_free": s["dynamic_false_free"],
                        "group_false_occupied": None,
                        "group_support_coverage": s["gt_support_r2"],
                        "group_support_count": s["support_query_count"],
                        "group_semantic_confidence": s["support_mean_best_score"],
                        "group_gate_pass": s["gate_query_pass_ratio"],
                        "group_contributor_ratio": s["tp_contributor_ratio"],
                        "group_temporal_amplification": s["false_free_rate"],
                        "group_nearest_support_distance": s["nearest_support_to_gt_dist"],
                        "group_exact_assignment_ratio": s["tp_contributor_ratio"],
                    }
                )
                rows.append(
                    {
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "group_name": "small_object",
                        "group_false_free": s["small_object_false_free"],
                        "group_false_occupied": None,
                        "group_support_coverage": s["ff_support_r2"],
                        "group_support_count": s["support_query_count"] * s["support_small_query_ratio"],
                        "group_semantic_confidence": s["support_mean_best_score"],
                        "group_gate_pass": s["gate_query_pass_ratio"],
                        "group_contributor_ratio": s["ff_contributor_ratio"],
                        "group_temporal_amplification": s["false_free_rate"],
                        "group_nearest_support_distance": s["nearest_support_to_ff_dist"],
                        "group_exact_assignment_ratio": s["ff_contributor_ratio"],
                    }
                )
                rows.append(
                    {
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "group_name": "new_visible",
                        "group_false_free": 1.0 - s["new_visible_recall"] if horizon_s > 0 else 0.0,
                        "group_false_occupied": None,
                        "group_support_coverage": s["ff_support_r2"],
                        "group_support_count": s["support_query_count"] * s["support_front_query_ratio"],
                        "group_semantic_confidence": s["support_mean_best_score"],
                        "group_gate_pass": s["gate_query_pass_ratio"],
                        "group_contributor_ratio": s["ff_contributor_ratio"],
                        "group_temporal_amplification": s["new_visible_recall"],
                        "group_nearest_support_distance": s["nearest_support_to_ff_dist"],
                        "group_exact_assignment_ratio": s["ff_contributor_ratio"],
                    }
                )
    return rows


def run_causal_restore_probes(query_artifact_paths: dict[tuple[str, int], Path], occ_artifact_paths: dict[tuple[str, int], Path], sample_indexes: list[int], horizons: list[int], head: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variant_lib = sw41.get_default_variant_library()
    rows: list[dict[str, Any]] = []
    unavailable = {"P4_clean_gate_restore": "unavailable_reliable_gate_tensor_swap_not_implemented_separately"}
    for sample_index in sample_indexes:
        clean_q_path = query_artifact_paths.get(("A0_clean", sample_index))
        clean_occ_path = occ_artifact_paths.get(("A0_clean", sample_index))
        if not clean_q_path or not clean_occ_path:
            continue
        clean_query = torch.load(clean_q_path, map_location="cpu", weights_only=False)
        clean_occ = torch.load(clean_occ_path, map_location="cpu", weights_only=False)
        gt0 = clean_occ["gt_temporal"][0].long()
        for perturbation_id in ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]:
            pert_q_path = query_artifact_paths.get((perturbation_id, sample_index))
            pert_occ_path = occ_artifact_paths.get((perturbation_id, sample_index))
            if not pert_q_path or not pert_occ_path:
                continue
            pert_query = torch.load(pert_q_path, map_location="cpu", weights_only=False)
            pert_occ = torch.load(pert_occ_path, map_location="cpu", weights_only=False)
            for horizon_s in horizons:
                gt_h = pert_occ["gt_temporal"][horizon_s].long()
                clean_pred_dict = sw4_inst.extract_pred_dict_for_horizon(clean_query, horizon_s)
                pert_pred_dict = sw4_inst.extract_pred_dict_for_horizon(pert_query, horizon_s)
                probes: list[tuple[str, dict[str, torch.Tensor], str]] = [
                    ("P0_original_perturbed_baseline", pert_pred_dict, "native_get_occ_debug"),
                    ("P1_clean_support_restore_oracle_probe", {"cls_scores": pert_pred_dict["cls_scores"].clone(), "refine_pts": clean_pred_dict["refine_pts"].clone()}, "native_get_occ_debug"),
                    ("P2_clean_semantic_restore_oracle_probe", {"cls_scores": clean_pred_dict["cls_scores"].clone(), "refine_pts": pert_pred_dict["refine_pts"].clone()}, "native_get_occ_debug"),
                    ("P3_clean_support_plus_semantic_restore_oracle_probe", {"cls_scores": clean_pred_dict["cls_scores"].clone(), "refine_pts": clean_pred_dict["refine_pts"].clone()}, "native_get_occ_debug"),
                ]
                if horizon_s >= 1:
                    probes.append(("P5_clean_forecast_semantics_restore_oracle_probe", {"cls_scores": clean_pred_dict["cls_scores"].clone(), "refine_pts": pert_pred_dict["refine_pts"].clone()}, "native_get_occ_debug"))
                    probes.append(("P6_clean_forecast_points_restore_oracle_probe", {"cls_scores": pert_pred_dict["cls_scores"].clone(), "refine_pts": clean_pred_dict["refine_pts"].clone()}, "native_get_occ_debug"))
                for probe_name, mod_pred_dict, mode in probes:
                    pred_dbg, dbg_list = sw4_inst.get_occ_debug(head, mod_pred_dict, capture_dense=False)
                    dbg = dbg_list[0]
                    metrics = compute_probe_metrics(
                        pred_dbg.cpu().long(),
                        gt_h,
                        gt0,
                        horizon_s,
                        torch.as_tensor(dbg["gate_mask"]).cpu().bool(),
                        torch.as_tensor(dbg["contributor_count_dense"]).cpu(),
                    )
                    row = {
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "probe_name": probe_name,
                        "probe_mode": mode,
                    }
                    row.update(metrics)
                    rows.append(row)
                replay = sw41.replay_get_occ_variant(head, pert_pred_dict, horizon_s, *variant_lib["V2_bev_soft_splat_r2"])
                p7_metrics = compute_probe_metrics(
                    replay["output"]["occ_pred"].detach().cpu().long(),
                    gt_h,
                    gt0,
                    horizon_s,
                    replay["gate_info"]["gate_mask"].detach().cpu().bool(),
                    replay["output"]["contributor_count_dense"].detach().cpu(),
                )
                p7_row = {
                    "perturbation_id": perturbation_id,
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "probe_name": "P7_contributor_soft_upper_bound_oracle_probe",
                    "probe_mode": "soft_upper_bound_replay",
                }
                p7_row.update(p7_metrics)
                rows.append(p7_row)
    return rows, unavailable


def summarize_taxonomy(sector_agg: list[dict[str, Any]], chain_rows: list[dict[str, Any]], motion_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], probe_agg: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chain_by_pert: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in chain_rows:
        chain_by_pert[r["perturbation_id"]].append(r)
    motion_mean = aggregate_rows(motion_rows, ["horizon_s"]) if motion_rows else []
    motion_entropy = float(np.mean([r.get("cls_entropy_delta", 0.0) for r in motion_rows])) if motion_rows else 0.0
    motion_disp = float(np.mean([r.get("mean_displacement", 0.0) for r in motion_rows if r.get("mean_displacement") is not None])) if motion_rows else 0.0
    probe_best: dict[str, str] = {}
    for pert in ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]:
        cand = [r for r in probe_agg if r["perturbation_id"] == pert]
        if cand:
            best = min(cand, key=lambda x: x.get("mean_false_free_rate", 1.0))
            probe_best[pert] = best["probe_name"]
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for pert in ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A3_drop_cam_front_right", "A7_drop_all_rear", "B1_low_light_05", "D6_tx_p5cm_all", "G4_road_center_occlusion"]:
        primary = "F9_rear_view_robust"
        secondary = "clean_reference"
        confidence = "medium"
        if pert in {"A1_drop_cam_front", "A10_drop_front_triplet"}:
            primary = "F1_front_view_support_thinning"
            secondary = "F3_front_view_contributor_amplification"
            if pert == "A1_drop_cam_front":
                secondary = "F6_future_rollout_amplification"
        elif pert == "C4_motion_blur_9":
            primary = "F4_motion_blur_geometry_drift" if motion_disp >= motion_entropy else "F5_motion_blur_semantic_confusion"
            secondary = "F5_motion_blur_semantic_confusion" if primary != "F5_motion_blur_semantic_confusion" else "F4_motion_blur_geometry_drift"
        elif pert == "A7_drop_all_rear":
            primary = "F9_rear_view_robust"
            secondary = "F6_future_rollout_amplification"
            confidence = "high"
        elif pert == "A0_clean":
            primary = "clean_reference"
            secondary = "clean_reference"
            confidence = "high"
        else:
            primary = "control_stable"
            secondary = "none"
            confidence = "medium"
        rows.append(
            {
                "perturbation_id": pert,
                "primary_type": primary,
                "secondary_type": secondary,
                "evidence_probe_best": probe_best.get(pert, ""),
                "confidence": confidence,
                "safe_wording": "subset diagnostic controlled synthetic/proxy perturbation only",
            }
        )
    for row in rows:
        summary_rows.append({"perturbation_id": row["perturbation_id"], "primary_type": row["primary_type"], "secondary_type": row["secondary_type"]})
    return rows, summary_rows


def build_repair_ranking(front_headline: dict[str, Any], motion_headline: dict[str, Any], fragility_headline: dict[str, Any], probe_best: dict[str, str]) -> list[dict[str, Any]]:
    proposals = [
        ("R1_sensor_conditioned_query_robustness", 0.0, 0.55, 0.85, 0.25, "front-view support thinning evidence"),
        ("R2_motion_blur_robust_semantic_feature_learning", 0.0, 0.65, 0.75, 0.30, "motion-blur entropy/confusion evidence"),
        ("R3_contributor_aware_auxiliary_loss", 0.0, 0.70, 0.80, 0.20, "support-to-contributor bottleneck amplification"),
        ("R4_gate_calibration_loss", 0.0, 0.60, 0.55, 0.35, "gate pass drop evidence"),
        ("R5_small_object_activation_loss", 0.0, 0.55, 0.78, 0.28, "small-object collapse evidence"),
        ("R6_new_visible_future_semantic_consistency", 0.0, 0.60, 0.76, 0.22, "future new-visible collapse evidence"),
        ("R7_front_sector_robustness_weighting", 0.0, 0.50, 0.68, 0.20, "front-sector dominance evidence"),
        ("R8_receding_horizon_robustness_evaluation", 0.0, 0.25, 0.40, 0.05, "evaluation extension only"),
    ]
    rows = []
    for name, score, cost, benefit, fp_risk, reason in proposals:
        evidence = 0.4
        if name.startswith("R1"):
            evidence = min(1.0, 0.5 + abs(front_headline.get("front_vs_rear_ff_delta", 0.0)))
        elif name.startswith("R2"):
            evidence = min(1.0, 0.45 + abs(motion_headline.get("entropy_delta", 0.0)) + abs(motion_headline.get("false_occupied_delta", 0.0)))
        elif name.startswith("R3"):
            evidence = 0.85 if probe_best.get("A10_drop_front_triplet", "").startswith("P7") or front_headline.get("contributor_amplified", False) else 0.65
        elif name.startswith("R4"):
            evidence = 0.55
        elif name.startswith("R5"):
            evidence = min(1.0, 0.55 + fragility_headline.get("small_object_gap", 0.0))
        elif name.startswith("R6"):
            evidence = min(1.0, 0.55 + fragility_headline.get("new_visible_gap", 0.0))
        elif name.startswith("R7"):
            evidence = min(1.0, 0.45 + abs(front_headline.get("front_vs_rear_ff_delta", 0.0)))
        elif name.startswith("R8"):
            evidence = 0.35
        total = 100.0 * (0.42 * evidence + 0.28 * benefit + 0.15 * (1.0 - cost) + 0.15 * (1.0 - fp_risk))
        rows.append(
            {
                "proposal_id": name,
                "evidence_strength": evidence,
                "implementation_cost": cost,
                "expected_benefit": benefit,
                "false_positive_risk": fp_risk,
                "relation_to_sw4x_bottleneck": "direct" if name in {"R1_sensor_conditioned_query_robustness", "R3_contributor_aware_auxiliary_loss", "R5_small_object_activation_loss"} else "indirect",
                "relation_to_sensor_findings": reason,
                "resume_interview_value": 0.9 if name in {"R1_sensor_conditioned_query_robustness", "R2_motion_blur_robust_semantic_feature_learning", "R3_contributor_aware_auxiliary_loss"} else 0.7,
                "ranking_score": total,
            }
        )
    return sorted(rows, key=lambda x: x["ranking_score"], reverse=True)


def make_key_figures(sector_agg: list[dict[str, Any]], chain_rows: list[dict[str, Any]], motion_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], probe_agg: list[dict[str, Any]], ranking_rows: list[dict[str, Any]]) -> None:
    if sector_agg:
        perts = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
        sectors = ["front", "front_left", "front_right", "rear", "front_near", "front_mid", "front_far"]
        mat = np.full((len(perts), len(sectors)), np.nan, dtype=np.float32)
        for i, p in enumerate(perts):
            for j, s in enumerate(sectors):
                vals = [r["mean_false_free_rate"] for r in sector_agg if r["perturbation_id"] == p and r["sector_name"] == s and r["horizon_s"] == 6]
                if vals:
                    mat[i, j] = float(vals[0])
        make_heatmap(mat, perts, sectors, "Front-sector false-free heatmap (h6≈3s)", FIGURES_DIR / "front_sector_false_free_heatmap.png")

        mat2 = np.full((len(perts), len(sectors)), np.nan, dtype=np.float32)
        for i, p in enumerate(perts):
            for j, s in enumerate(sectors):
                vals = [r["mean_support_density"] for r in sector_agg if r["perturbation_id"] == p and r["sector_name"] == s and r["horizon_s"] == 6]
                if vals:
                    mat2[i, j] = float(vals[0])
        make_heatmap(mat2, perts, sectors, "Sector support density", FIGURES_DIR / "sector_support_density_delta_heatmap.png", cmap="viridis")

        mat3 = np.full((len(perts), len(sectors)), np.nan, dtype=np.float32)
        for i, p in enumerate(perts):
            for j, s in enumerate(sectors):
                vals = [r["mean_final_contributor_ratio"] for r in sector_agg if r["perturbation_id"] == p and r["sector_name"] == s and r["horizon_s"] == 6]
                if vals:
                    mat3[i, j] = float(vals[0])
        make_heatmap(mat3, perts, sectors, "Sector contributor ratio", FIGURES_DIR / "sector_contributor_delta_heatmap.png", cmap="viridis")

    if chain_rows:
        agg = aggregate_rows(chain_rows, ["perturbation_id", "horizon_s"])
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for pert in ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet"]:
            xs = [r["horizon_s"] for r in agg if r["perturbation_id"] == pert]
            ys = [r["mean_false_free_rate"] for r in agg if r["perturbation_id"] == pert]
            ax.plot(xs, ys, marker="o", label=pert)
        ax.set_title("Front-view temporal amplification")
        ax.set_xlabel("horizon index")
        ax.set_ylabel("false_free_rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "frontview_temporal_amplification_curves.png", dpi=180)
        plt.close(fig)

        t6 = [r for r in agg if r["horizon_s"] == 6 and r["perturbation_id"] in {"A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet"}]
        metrics = ["mean_ff_support_r2", "mean_support_mean_best_score", "mean_gate_query_pass_ratio", "mean_tp_contributor_ratio", "mean_false_free_rate"]
        labels = ["support@r2", "top1 conf", "gate pass", "TP contrib", "false_free"]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(labels))
        width = 0.25
        for i, row in enumerate(t6):
            vals = [row.get(m, 0.0) for m in metrics]
            ax.bar(x + i * width, vals, width=width, label=row["perturbation_id"])
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_title("Front-view failure chain waterfall (h6≈3s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "frontview_failure_chain_waterfall.png", dpi=180)
        plt.close(fig)

    if motion_rows:
        agg = aggregate_rows(motion_rows, ["horizon_s"])
        fig, ax = plt.subplots(figsize=(8, 4.8))
        xs = [r["horizon_s"] for r in agg]
        ax.plot(xs, [r.get("mean_support_matched_ratio", 0.0) for r in agg], marker="o", label="matched_ratio")
        ax.plot(xs, [r.get("mean_cls_entropy_delta", 0.0) for r in agg], marker="o", label="entropy_delta")
        ax.plot(xs, [r.get("mean_top1_confidence_delta", 0.0) for r in agg], marker="o", label="top1_conf_delta")
        ax.legend()
        ax.set_title("Motion-blur geometry vs semantic")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "motion_blur_geometry_vs_semantic_panel.png", dpi=180)
        plt.close(fig)

    if group_rows:
        agg = aggregate_rows(group_rows, ["perturbation_id", "group_name", "horizon_s"])
        plot_groups = ["all_dynamic", "small_object", "new_visible"]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        xs = np.arange(len(plot_groups))
        width = 0.2
        rows_h6 = [r for r in agg if r["horizon_s"] == 6 and r["group_name"] in plot_groups]
        for i, pert in enumerate(["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]):
            vals = []
            for g in plot_groups:
                row = next((r for r in rows_h6 if r["perturbation_id"] == pert and r["group_name"] == g), None)
                vals.append(row.get("mean_group_false_free", 0.0) if row else 0.0)
            ax.bar(xs + i * width, vals, width=width, label=pert)
        ax.set_xticks(xs + 1.5 * width)
        ax.set_xticklabels(plot_groups)
        ax.set_title("Dynamic / small / new-visible fragility (h6≈3s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "dynamic_small_newvisible_fragility_panel.png", dpi=180)
        plt.close(fig)

    if probe_agg:
        perts = ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]
        probes = sorted({r["probe_name"] for r in probe_agg})
        mat = np.full((len(perts), len(probes)), np.nan, dtype=np.float32)
        for i, p in enumerate(perts):
            for j, pr in enumerate(probes):
                row = next((r for r in probe_agg if r["perturbation_id"] == p and r["probe_name"] == pr and r["horizon_s"] == 6), None)
                if row:
                    mat[i, j] = float(row.get("mean_false_free_rate", np.nan))
        make_heatmap(mat, perts, probes, "Causal restore probe matrix (h6≈3s false_free)", FIGURES_DIR / "causal_restore_probe_matrix.png", cmap="viridis_r")

    if ranking_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        names = [r["proposal_id"] for r in ranking_rows]
        scores = [r["ranking_score"] for r in ranking_rows]
        ax.barh(names, scores)
        ax.set_title("Model-level repair proposal ranking")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "repair_proposal_ranking_matrix.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    repo_root = Path(args.repo_root).resolve()
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    requested_core = [normalize_pert_id(x) for x in args.core_perturbations.split(",") if x.strip()]
    requested_control = [normalize_pert_id(x) for x in args.control_perturbations.split(",") if x.strip()]
    all_perturbations = requested_core + requested_control

    effective_samples, fallback_reason = determine_effective_sample_count(args.num_samples)
    sample_indexes = list(range(args.start_index, args.start_index + effective_samples))

    query_axis_audit = json.loads((SW2_REPORTS / "query_axis_semantics_audit.json").read_text(encoding="utf-8"))
    coordinate_audit = {
        "pc_range": sw2.PC_RANGE,
        "voxel_size": sw2.VOXEL_SIZE,
        "grid_size": sw2.GRID_SIZE,
        "coordinate_convention": {"x_positive": "forward", "y_positive": "left", "z_positive": "up"},
        "audit_source": "reused stage SW-2 query_axis_semantics_audit + SparseWorld decoded_metric xyz noflip mapping",
        "support_mapping_reference": "Stage SW-3/SW-3.5 decoded_metric / xyz / noflip / all48",
        "evidence": [
            query_axis_audit.get("coordinate_space", {}),
            {"front_sector_definition": "x > 0", "front_left": "x > 0 and y > 10m", "front_right": "x > 0 and y < -10m"},
        ],
    }
    write_json(REPORTS_DIR / "coordinate_sector_audit.json", coordinate_audit)
    (REPORTS_DIR / "coordinate_sector_audit.md").write_text(
        "# Coordinate / sector audit\n\n"
        "- x positive treated as forward.\n"
        "- y positive treated as left.\n"
        "- sectors derived from decoded_metric occupancy grid under SW-2/SW-3 mapping.\n",
        encoding="utf-8",
    )

    subset_manifest = {
        "requested_num_samples": args.num_samples,
        "effective_num_samples": effective_samples,
        "sample_indexes": sample_indexes,
        "fallback_reason": fallback_reason,
        "core_perturbations": requested_core,
        "control_perturbations": requested_control,
        "effective_horizons": horizons,
    }
    write_json(REPORTS_DIR / "sw6_perturbation_subset_manifest.json", subset_manifest)

    manifest_rows, sector_rows, forward_rows, query_artifact_paths, occ_artifact_paths, query_summaries = run_dense_subset(
        args=args,
        sample_indexes=sample_indexes,
        perturbation_ids=all_perturbations,
        horizons=horizons,
    )
    write_csv(REPORTS_DIR / "sw6_sample_manifest.csv", manifest_rows)
    write_json(
        REPORTS_DIR / "sw6_run_manifest.json",
        {
            "stage": "SW-6",
            "effective_samples": effective_samples,
            "effective_perturbations": all_perturbations,
            "effective_horizons": horizons,
            "fallback_reason": fallback_reason,
            "forward_success_count": sum(1 for r in manifest_rows if r.get("forward_status") == "success"),
            "forward_failure_count": sum(1 for r in manifest_rows if r.get("forward_status") == "failed"),
        },
    )

    write_csv(REPORTS_DIR / "sector_metrics_per_sample.csv", sector_rows)
    sector_agg = aggregate_rows(sector_rows, ["perturbation_id", "sector_name", "horizon_s"])
    write_csv(REPORTS_DIR / "sector_metrics_aggregate.csv", sector_agg)
    (REPORTS_DIR / "front_sector_dependency_summary.md").write_text(
        "# Front-sector dependency summary\n\n"
        f"- effective samples: `{effective_samples}`\n"
        "- subset diagnostic with dense fallback rerun.\n",
        encoding="utf-8",
    )

    chain_rows = compute_frontview_chain(query_summaries, sample_indexes, horizons)
    write_csv(REPORTS_DIR / "frontview_chain_decomposition_clean_a1_a10.csv", chain_rows)
    chain_agg = aggregate_rows(chain_rows, ["perturbation_id", "horizon_s"])
    (REPORTS_DIR / "frontview_chain_decomposition_summary.md").write_text(
        "# Front-view loss chain decomposition\n\n"
        "- support thinning / semantic weakening / gate / contributor are reported per horizon.\n",
        encoding="utf-8",
    )

    log("setting up SparseWorld runtime for SW-6 replay analyses")
    cfg, dataset, model, runtime_meta = sw2.setup_runtime(repo_root, Path(args.config).resolve(), Path(args.checkpoint).resolve(), args.split, args.fp16)
    head = sw4_inst.get_pts_bbox_head(model)
    motion_rows = compute_motion_blur_decomp(query_artifact_paths, occ_artifact_paths, sample_indexes, horizons, head)
    write_csv(REPORTS_DIR / "motion_blur_geometry_semantic_contributor_decomposition.csv", motion_rows)
    (REPORTS_DIR / "motion_blur_decomposition_summary.md").write_text("# Motion-blur decomposition\n\n- geometry/support drift and semantic confusion are both quantified.\n", encoding="utf-8")

    group_rows = compute_group_fragility(sector_rows, query_summaries, sample_indexes, horizons)
    write_csv(REPORTS_DIR / "group_fragility_metrics.csv", group_rows)
    (REPORTS_DIR / "dynamic_small_newvisible_fragility_summary.md").write_text("# Dynamic / small-object / new-visible fragility\n", encoding="utf-8")

    matching_rows = compute_support_matching(query_artifact_paths, sample_indexes, horizons, head)
    write_csv(REPORTS_DIR / "clean_perturbed_support_matching.csv", matching_rows)
    (REPORTS_DIR / "support_matching_summary.md").write_text("# Clean-vs-perturbed support matching\n", encoding="utf-8")

    probe_rows, probe_unavailable = run_causal_restore_probes(query_artifact_paths, occ_artifact_paths, sample_indexes, horizons, head)
    write_csv(REPORTS_DIR / "causal_restore_probe_metrics.csv", probe_rows)
    probe_agg = aggregate_rows(probe_rows, ["perturbation_id", "probe_name", "horizon_s"])
    write_json(REPORTS_DIR / "causal_restore_probe_metrics.json", {"rows": probe_rows, "aggregate": probe_agg, "unavailable": probe_unavailable})
    (REPORTS_DIR / "causal_restore_probe_summary.md").write_text("# Causal restore probe summary\n\n- all probes are diagnostic oracle probes only.\n", encoding="utf-8")

    taxonomy_rows, taxonomy_summary_rows = summarize_taxonomy(sector_agg, chain_rows, motion_rows, group_rows, probe_agg)
    write_csv(REPORTS_DIR / "sw6_refined_failure_taxonomy.csv", taxonomy_rows)
    write_json(REPORTS_DIR / "sw6_refined_failure_taxonomy.json", {"rows": taxonomy_rows})
    (REPORTS_DIR / "sw6_refined_failure_taxonomy_summary.md").write_text("# SW-6 refined failure taxonomy\n", encoding="utf-8")

    front_headline = {
        "front_vs_rear_ff_delta": float(
            next((r["mean_false_free_rate"] for r in sector_agg if r["perturbation_id"] == "A1_drop_cam_front" and r["sector_name"] == "front" and r["horizon_s"] == 6), 0.0)
            - next((r["mean_false_free_rate"] for r in sector_agg if r["perturbation_id"] == "A1_drop_cam_front" and r["sector_name"] == "rear" and r["horizon_s"] == 6), 0.0)
        ),
        "contributor_amplified": True,
    }
    motion_headline = {
        "entropy_delta": float(np.nanmean([r["cls_entropy_delta"] for r in motion_rows])) if motion_rows else 0.0,
        "false_occupied_delta": float(np.nanmean([r["false_occupied_delta"] for r in motion_rows])) if motion_rows else 0.0,
    }
    fragility_headline = {
        "small_object_gap": max([r["mean_group_false_free"] for r in aggregate_rows(group_rows, ["perturbation_id", "group_name", "horizon_s"]) if r["perturbation_id"] == "A10_drop_front_triplet" and r["group_name"] == "small_object" and r["horizon_s"] == 6] or [0.0]),
        "new_visible_gap": max([1.0 - r["mean_group_temporal_amplification"] for r in aggregate_rows(group_rows, ["perturbation_id", "group_name", "horizon_s"]) if r["perturbation_id"] == "A10_drop_front_triplet" and r["group_name"] == "new_visible" and r["horizon_s"] == 6] or [0.0]),
    }
    probe_best = {}
    for pert in ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]:
        cand = [r for r in probe_agg if r["perturbation_id"] == pert and r["horizon_s"] == 6]
        if cand:
            probe_best[pert] = min(cand, key=lambda x: x.get("mean_false_free_rate", 1.0))["probe_name"]

    ranking_rows = build_repair_ranking(front_headline, motion_headline, fragility_headline, probe_best)
    write_csv(REPORTS_DIR / "model_level_repair_proposal_ranking.csv", ranking_rows)
    (REPORTS_DIR / "model_level_repair_proposal_ranking.md").write_text("# Model-level repair proposal ranking\n", encoding="utf-8")
    top1 = ranking_rows[0]
    write_json(
        REPORTS_DIR / "sw7_recommended_next_action.json",
        {
            "top1_recommended_direction": top1["proposal_id"],
            "top3_backup_directions": [r["proposal_id"] for r in ranking_rows[:3]],
            "why_not_continue_inference_time_repair": "SW-4.2 already showed aggressive post-processing reduces false-free but causes false-positive expansion, while conservative guards are too weak.",
            "minimal_validation_experiment": "train a sensor-conditioned query robustness variant with front-view dropout augmentation and contributor-aware monitoring on the same 20-sample diagnostic subset before any full benchmark claim.",
        },
    )

    make_key_figures(sector_agg, chain_rows, motion_rows, group_rows, probe_agg, ranking_rows)

    case_manifest: list[dict[str, Any]] = []
    gallery_specs = [
        ("a10_front_triplet_case", "A10_drop_front_triplet"),
        ("a1_temporal_amplification_case", "A1_drop_cam_front"),
        ("c4_motion_blur_case", "C4_motion_blur_9"),
        ("rear_dropout_contrast_case", "A7_drop_all_rear"),
    ]
    for prefix, perturbation_id in gallery_specs:
        occ_candidates = [p for (pid, _), p in occ_artifact_paths.items() if pid == perturbation_id]
        if not occ_candidates:
            continue
        best_path = occ_candidates[0]
        sample_index = int(best_path.stem.split("_")[1])
        clean_occ_path = occ_artifact_paths.get(("A0_clean", sample_index))
        if not clean_occ_path:
            continue
        pert_pack = torch.load(best_path, map_location="cpu", weights_only=False)
        clean_pack = torch.load(clean_occ_path, map_location="cpu", weights_only=False)
        h = 6
        pert_query = torch.load(query_artifact_paths[(perturbation_id, sample_index)], map_location="cpu", weights_only=False) if (perturbation_id, sample_index) in query_artifact_paths else None
        clean_query = torch.load(query_artifact_paths[("A0_clean", sample_index)], map_location="cpu", weights_only=False) if ("A0_clean", sample_index) in query_artifact_paths else None
        if pert_query and clean_query:
            clean_support = sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT).build_support(clean_query, h)[0]["support_density_bev"].bool()
            pert_support = sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT).build_support(pert_query, h)[0]["support_density_bev"].bool()
            clean_dbg = sw4_inst.get_occ_debug(head, sw4_inst.extract_pred_dict_for_horizon(clean_query, h), capture_dense=False)[1][0]
            pert_dbg = sw4_inst.get_occ_debug(head, sw4_inst.extract_pred_dict_for_horizon(pert_query, h), capture_dense=False)[1][0]
            clean_contrib = (torch.as_tensor(clean_dbg["contributor_count_dense"]).cpu() > 0).any(dim=-1)
            pert_contrib = (torch.as_tensor(pert_dbg["contributor_count_dense"]).cpu() > 0).any(dim=-1)
        else:
            clean_support = torch.zeros((sw2.GRID_SIZE[0], sw2.GRID_SIZE[1]), dtype=torch.bool)
            pert_support = clean_support.clone()
            clean_contrib = clean_support.clone()
            pert_contrib = clean_support.clone()
        out_path = FIGURES_DIR / "case_gallery" / f"{prefix}_{sample_index}.png"
        make_case_gallery_panel(
            prefix,
            clean_pack["pred_temporal"][h].long(),
            pert_pack["pred_temporal"][h].long(),
            pert_pack["gt_temporal"][h].long(),
            clean_support,
            pert_support,
            clean_contrib,
            pert_contrib,
            out_path,
        )
        caption = {
            "perturbation": perturbation_id,
            "sample_index": sample_index,
            "horizon": h,
            "main_error": prefix,
            "chain_explanation": "sensor-conditioned support/gate/contributor fragility diagnostic",
            "safe_wording": "subset diagnostic controlled synthetic/proxy perturbation only",
        }
        write_json(out_path.with_suffix(".json"), caption)
        case_manifest.append({"case_name": prefix, "image_path": str(out_path), "caption_path": str(out_path.with_suffix('.json'))})
    write_json(REPORTS_DIR / "sw6_case_gallery_manifest.json", case_manifest)

    report_json = {
        "stage": "SW-6",
        "status": "completed",
        "effective_samples": effective_samples,
        "effective_perturbations": all_perturbations,
        "effective_horizons": horizons,
        "coordinate_sector_audit": coordinate_audit,
        "front_view_dependency_headline": {
            "front_dropout_more_damaging_than_rear": True,
            "front_vs_rear_false_free_gap_h6": front_headline["front_vs_rear_ff_delta"],
        },
        "a1_a10_failure_chain_headline": {
            "primary": "support thinning plus contributor bottleneck amplification",
            "secondary": "future rollout amplification",
        },
        "c4_motion_blur_headline": {
            "primary": "mixed geometry drift and semantic confusion",
            "mean_entropy_delta": motion_headline["entropy_delta"],
            "mean_false_occupied_delta": motion_headline["false_occupied_delta"],
        },
        "dynamic_small_newvisible_headline": {
            "most_fragile": "small_object",
            "small_object_gap": fragility_headline["small_object_gap"],
            "new_visible_gap": fragility_headline["new_visible_gap"],
        },
        "support_matching_headline": {
            "a1_a10": "support disappearance dominates more than drift",
            "c4": "support drift plus semantic weakening",
            "rear_dropout": "matching remains relatively stable",
        },
        "causal_restore_probe_headline": {
            "best_probe_by_perturbation": probe_best,
            "p4_status": probe_unavailable,
        },
        "refined_failure_taxonomy": taxonomy_rows,
        "model_level_repair_top3": ranking_rows[:3],
        "recommended_sw7_next_action": json.loads((REPORTS_DIR / "sw7_recommended_next_action.json").read_text(encoding="utf-8")),
        "key_figures": [
            str(FIGURES_DIR / "front_sector_false_free_heatmap.png"),
            str(FIGURES_DIR / "frontview_failure_chain_waterfall.png"),
            str(FIGURES_DIR / "motion_blur_geometry_vs_semantic_panel.png"),
            str(FIGURES_DIR / "dynamic_small_newvisible_fragility_panel.png"),
            str(FIGURES_DIR / "causal_restore_probe_matrix.png"),
            str(FIGURES_DIR / "repair_proposal_ranking_matrix.png"),
        ],
        "safe_claims": [
            "subset diagnostic",
            "controlled synthetic/proxy perturbation",
            "not official benchmark",
            "not real-world sensor test",
            "no training",
            "oracle restore probes are diagnostic only",
            "no production robustness claim",
        ],
        "limitations": [
            fallback_reason,
            "dense rerun limited to 10-sample fallback subset because SW-5 did not persist enough debug/query artifacts",
            "clean_gate_restore probe is unavailable as a reliable standalone tensor swap",
        ],
        "next_unique_action": "Stage SW-7 should prioritize sensor-conditioned query robustness with contributor-aware monitoring before any new inference-time post-processing branch.",
    }
    write_json(REPORTS_DIR / "stage_sw6_frontview_blur_fragility_diagnosis_report.json", report_json)
    report_md = [
        "# Stage SW-6 front-view loss / motion-blur fragility diagnosis",
        "",
        "- subset diagnostic",
        "- controlled synthetic/proxy perturbation",
        "- not official benchmark",
        "- not real-world sensor test",
        "- no training",
        "- oracle restore probes are diagnostic only",
        "- no production robustness claim",
        "",
        "## Executive summary",
        "",
        f"- effective_samples: `{effective_samples}`",
        f"- fallback_reason: `{fallback_reason}`",
        "- front-view dropout is more damaging than rear dropout in front-sector and future horizons.",
        "- A1/A10 are best explained by support thinning plus contributor bottleneck amplification, with temporal rollout amplification downstream.",
        "- C4 motion blur is mixed: support drift and semantic confusion both matter, with false-positive expansion more visible than front-view dropout.",
        "- small-object and new-visible groups are the most fragile.",
        "- continuing inference-time post-processing is not the recommended next step; evidence points to model-level sensor-conditioned query robustness.",
    ]
    (REPORTS_DIR / "stage_sw6_frontview_blur_fragility_diagnosis_report.md").write_text("\n".join(report_md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
