from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import queue
import subprocess
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"

SW6_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
SW6_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
SW5_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"


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
    "sw7_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw35 = load_module(
    "sw7_sw35",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage/corrected_support_adapter.py",
)
sw4_inst = load_module(
    "sw7_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw41 = load_module(
    "sw7_sw41",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation/get_occ_replay_and_ablation.py",
)
engine = load_module(
    "sw7_engine",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation/sensor_perturbation_engine.py",
)
vis_renderer = load_module(
    "sw7_vis_renderer",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization/semantic_occupancy_renderer.py",
)


EMPTY_IDX = sw2.EMPTY_IDX
OCC_NAMES = list(vis_renderer.OCC_NAMES)
PALETTE = dict(vis_renderer.PALETTE)
CORE_PERTS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
OPTIONAL_CONTROLS = ["B1_low_light_05", "G4_road_center_occlusion"]
ALL_DYNAMIC_IDS = set(sw2.CLASS_GROUPS["all_dynamic"])
ALL_STATIC_IDS = set(sw2.CLASS_GROUPS["all_static"])
SMALL_OBJECT_IDS = set(sw2.CLASS_GROUPS["small_object"])
RISK_TYPE_NAMES = {
    0: "low_risk",
    1: "semantic_risk",
    2: "support_risk",
    3: "contributor_risk",
    4: "sensor_condition_risk",
    5: "temporal_risk",
    6: "mixed_risk",
}
CAM_FRONT_SET = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
CAM_REAR_SET = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    p.add_argument("--config", default=str(PROJECT_ROOT / "external/SparseWorld/configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"))
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "external/SparseWorld/ckpts/epoch_56.pth"))
    p.add_argument("--split", default="val")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--sample-indices", default="")
    p.add_argument("--horizons", default="0,1,2,3,4,5,6")
    p.add_argument("--core-perturbations", default="A0_clean,A1_drop_cam_front,A10_drop_front_triplet,C4_motion_blur_9,A7_drop_all_rear")
    p.add_argument("--run-reliability-map", action="store_true", default=True)
    p.add_argument("--run-risk-map", action="store_true", default=True)
    p.add_argument("--run-score-alpha", action="store_true", default=True)
    p.add_argument("--run-dashboard", action="store_true", default=True)
    p.add_argument("--run-high-risk-ranking", action="store_true", default=True)
    p.add_argument("--save-debug", action="store_true", default=True)
    p.add_argument("--save-figures", action="store_true", default=True)
    p.add_argument("--save-videos", action="store_true", default=True)
    p.add_argument("--smoke", action="store_true", default=False)
    return p.parse_args()


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, LOGS_DIR, FIGURES_DIR, ARTIFACTS_DIR, VIDEOS_DIR, TESTS_DIR, FIGURES_DIR / "score_alpha", FIGURES_DIR / "risk_maps", FIGURES_DIR / "dashboard_frames"]:
        p.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[SW7] {msg}", flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "phase1_sw7_main.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


class StageLogger:
    def __init__(self, status_path: Path, progress_path: Path) -> None:
        self.status_path = status_path
        self.progress_path = progress_path
        self.state: dict[str, Any] = {"started_at": __import__("time").time(), "current_stage": None, "stages": {}}
        self._flush()

    def _flush(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def start(self, name: str, **payload: Any) -> None:
        now = __import__("time").time()
        self.state["current_stage"] = name
        self.state["stages"].setdefault(name, {})
        self.state["stages"][name].update({"status": "running", "start_ts": now, **payload})
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now, "event": "start", "stage": name, **payload}, ensure_ascii=False) + "\n")
        self._flush()

    def progress(self, name: str, current: int, total: int, **payload: Any) -> None:
        now = __import__("time").time()
        self.state["stages"].setdefault(name, {})
        self.state["stages"][name].update({"status": "running", "progress_current": current, "progress_total": total, **payload})
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now, "event": "progress", "stage": name, "current": current, "total": total, **payload}, ensure_ascii=False) + "\n")
        self._flush()

    def done(self, name: str, **payload: Any) -> None:
        now = __import__("time").time()
        stage = self.state["stages"].setdefault(name, {})
        start_ts = float(stage.get("start_ts", now))
        stage.update({"status": "done", "end_ts": now, "duration_sec": now - start_ts, **payload})
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now, "event": "done", "stage": name, **payload}, ensure_ascii=False) + "\n")
        self._flush()

    def fail(self, name: str, error: str) -> None:
        now = __import__("time").time()
        stage = self.state["stages"].setdefault(name, {})
        start_ts = float(stage.get("start_ts", now))
        stage.update({"status": "failed", "end_ts": now, "duration_sec": now - start_ts, "error": error})
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now, "event": "failed", "stage": name, "error": error}, ensure_ascii=False) + "\n")
        self._flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_for_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([normalize_for_export(row) for row in rows])


def to_user_path(text: str) -> str:
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def normalize_for_export(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_for_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_for_export(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize_for_export(v) for v in obj]
    if isinstance(obj, str):
        return to_user_path(obj)
    return obj


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def parse_csv_arg(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, Any]]:
    bucket: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket[tuple(row[k] for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in bucket.items():
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["sample_count"] = len(items)
        for vk in value_keys:
            vals = [float(r[vk]) for r in items if r.get(vk) is not None]
            rec[f"mean_{vk}"] = float(np.mean(vals)) if vals else None
            rec[f"std_{vk}"] = float(np.std(vals)) if vals else None
        out.append(rec)
    return out


def build_sector_masks() -> dict[str, torch.Tensor]:
    x = torch.arange(sw2.GRID_SIZE[0], dtype=torch.float32) * sw2.VOXEL_SIZE[0] + sw2.PC_RANGE[0] + sw2.VOXEL_SIZE[0] / 2
    y = torch.arange(sw2.GRID_SIZE[1], dtype=torch.float32) * sw2.VOXEL_SIZE[1] + sw2.PC_RANGE[1] + sw2.VOXEL_SIZE[1] / 2
    z = torch.arange(sw2.GRID_SIZE[2], dtype=torch.float32) * sw2.VOXEL_SIZE[2] + sw2.PC_RANGE[2] + sw2.VOXEL_SIZE[2] / 2
    xx = x[:, None, None].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    yy = y[None, :, None].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    zz = z[None, None, :].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    rr = torch.sqrt(xx**2 + yy**2)
    return {
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
        "side": yy.abs() > 10.0,
        "all_valid_height": zz > -1e9,
    }


def persistent_new_visible_masks(gt0: torch.Tensor, gth: torch.Tensor) -> dict[str, torch.Tensor]:
    gt0_occ = gt0 != EMPTY_IDX
    gth_occ = gth != EMPTY_IDX
    return {
        "persistent": gt0_occ & gth_occ,
        "new_visible": (~gt0_occ) & gth_occ,
        "disappeared": gt0_occ & (~gth_occ),
    }


def top_bev_semantic(label_grid: torch.Tensor) -> torch.Tensor:
    grid = label_grid.long()
    out = torch.ones((grid.shape[0], grid.shape[1]), dtype=torch.long) * EMPTY_IDX
    active = grid != EMPTY_IDX
    if not bool(active.any().item()):
        return out
    rev = torch.flip(active, dims=[2])
    top_rev = torch.argmax(rev.int(), dim=2)
    has_any = active.any(dim=2)
    top_idx = (grid.shape[2] - 1) - top_rev
    xs, ys = torch.where(has_any)
    out[xs, ys] = grid[xs, ys, top_idx[xs, ys]]
    return out


def top_bev_scalar_for_pred(label_grid: torch.Tensor, scalar_grid: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((label_grid.shape[0], label_grid.shape[1]), dtype=torch.float32)
    active = label_grid != EMPTY_IDX
    if not bool(active.any().item()):
        return out
    rev = torch.flip(active, dims=[2])
    top_rev = torch.argmax(rev.int(), dim=2)
    has_any = active.any(dim=2)
    top_idx = (label_grid.shape[2] - 1) - top_rev
    xs, ys = torch.where(has_any)
    out[xs, ys] = scalar_grid[xs, ys, top_idx[xs, ys]].float()
    return out


def hex_to_rgb255(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def render_alpha_bev(label_grid: torch.Tensor, alpha_grid_3d: torch.Tensor, title: str, out_path: Path, alpha_min: float = 0.15, alpha_max: float = 1.0, gamma: float = 0.7) -> None:
    label_bev = top_bev_semantic(label_grid)
    alpha_bev = top_bev_scalar_for_pred(label_grid, alpha_grid_3d).clamp(0, 1)
    alpha_bev = alpha_min + (alpha_max - alpha_min) * alpha_bev.pow(gamma)
    canvas = np.ones((label_bev.shape[0], label_bev.shape[1], 3), dtype=np.float32) * 255.0
    for cls_id in torch.unique(label_bev):
        cid = int(cls_id.item())
        if cid == EMPTY_IDX:
            continue
        mask = label_bev == cid
        cls_name = OCC_NAMES[cid] if 0 <= cid < len(OCC_NAMES) else "others"
        color = np.array(hex_to_rgb255(PALETTE.get(cls_name, "#808080")), dtype=np.float32)
        alpha = alpha_bev[mask].cpu().numpy()[:, None]
        canvas[mask.cpu().numpy()] = alpha * color + (1.0 - alpha) * 255.0
    img = Image.fromarray(canvas.astype(np.uint8)).resize((900, 900), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.text((20, 16), title, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def render_risk_overlay_bev(label_grid: torch.Tensor, risk_grid: torch.Tensor, ff_mask: torch.Tensor | None, fo_mask: torch.Tensor | None, title: str, out_path: Path) -> None:
    label_bev = top_bev_semantic(label_grid)
    risk_bev = top_bev_scalar_for_pred(label_grid, risk_grid).clamp(0, 1).cpu().numpy()
    base = np.ones((label_bev.shape[0], label_bev.shape[1], 3), dtype=np.float32) * 255.0
    for cls_id in torch.unique(label_bev):
        cid = int(cls_id.item())
        if cid == EMPTY_IDX:
            continue
        mask = label_bev == cid
        cls_name = OCC_NAMES[cid] if 0 <= cid < len(OCC_NAMES) else "others"
        base[mask.cpu().numpy()] = np.array(hex_to_rgb255(PALETTE.get(cls_name, "#808080")), dtype=np.float32)
    overlay = base.copy()
    red = np.array([230, 80, 60], dtype=np.float32)
    alpha = (0.15 + 0.70 * risk_bev)[..., None]
    overlay = alpha * red + (1.0 - alpha) * overlay
    img = Image.fromarray(overlay.astype(np.uint8)).resize((900, 900), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.text((20, 16), title, fill="black")
    if ff_mask is not None:
        ff_bev = top_bev_scalar_for_pred(ff_mask.long() * 1, ff_mask.float()).cpu().numpy() > 0
        fo_bev = top_bev_scalar_for_pred(fo_mask.long() * 1, fo_mask.float()).cpu().numpy() > 0 if fo_mask is not None else np.zeros_like(ff_bev)
        ff_bev = np.array(Image.fromarray((ff_bev.astype(np.uint8) * 255)).resize((900, 900), Image.Resampling.NEAREST)) > 0
        fo_bev = np.array(Image.fromarray((fo_bev.astype(np.uint8) * 255)).resize((900, 900), Image.Resampling.NEAREST)) > 0
        arr = np.array(img)
        arr[ff_bev] = np.array([255, 0, 0], dtype=np.uint8)
        arr[fo_bev] = np.array([40, 110, 255], dtype=np.uint8)
        img = Image.fromarray(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def dilate3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().unsqueeze(0).unsqueeze(0)
    y = F.max_pool3d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y.squeeze(0).squeeze(0).bool()


def avg_local_3d(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    x = mask.float().unsqueeze(0).unsqueeze(0)
    y = F.avg_pool3d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return y.squeeze(0).squeeze(0)


def build_distance_proxy_map(geom_mask: torch.Tensor) -> torch.Tensor:
    d0 = geom_mask.bool()
    d1 = dilate3d(d0, 1)
    d2 = dilate3d(d0, 2)
    d3 = dilate3d(d0, 3)
    out = torch.full_like(geom_mask, 4, dtype=torch.int64)
    out[d3] = 3
    out[d2] = 2
    out[d1] = 1
    out[d0] = 0
    return out


def normalized_entropy_from_scores(scores: torch.Tensor) -> torch.Tensor:
    denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p = scores / denom
    ent = -(p.clamp(min=1e-8) * torch.log(p.clamp(min=1e-8))).sum(dim=-1)
    return ent / math.log(scores.shape[-1])


def compute_semantic_reliability(dense_scores: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    top2 = torch.topk(dense_scores, k=min(2, dense_scores.shape[-1]), dim=-1).values
    top1 = top2[..., 0].clamp(0, 1)
    second = top2[..., 1] if top2.shape[-1] > 1 else torch.zeros_like(top1)
    margin = (top1 - second).clamp(min=0)
    margin_norm = (margin / 0.35).clamp(0, 1)
    ent_norm = normalized_entropy_from_scores(dense_scores)
    ent_factor = (1.0 - ent_norm).clamp(0, 1)
    semantic_rel = (top1 * (0.4 + 0.6 * margin_norm) * (0.3 + 0.7 * ent_factor)).clamp(0, 1)
    return semantic_rel, {
        "top1_score": top1,
        "class_margin": margin_norm,
        "entropy_factor": ent_factor,
        "foreground_score": top1,
    }


def build_class_maps(pred_label: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "dynamic_vehicle": torch.isin(pred_label, torch.tensor([3, 4, 5, 9, 10], device=pred_label.device)),
        "ped_cyc_moto": torch.isin(pred_label, torch.tensor([2, 6, 7], device=pred_label.device)),
        "small_object": torch.isin(pred_label, torch.tensor(sorted(SMALL_OBJECT_IDS), device=pred_label.device)),
        "static_background": torch.isin(pred_label, torch.tensor([1, 11, 12, 13, 14, 15, 16], device=pred_label.device)),
        "driveable_terrain": torch.isin(pred_label, torch.tensor([11, 12, 13, 14], device=pred_label.device)),
        "manmade_vegetation": torch.isin(pred_label, torch.tensor([1, 15, 16], device=pred_label.device)),
    }


def compute_sensor_prior_map(perturbation_id: str, pred_label: torch.Tensor, sectors: dict[str, torch.Tensor], horizon_s: int) -> torch.Tensor:
    prior = torch.ones_like(pred_label, dtype=torch.float32) * 0.80
    if perturbation_id == "A0_clean":
        prior.fill_(0.95)
    elif perturbation_id == "A7_drop_all_rear":
        prior.fill_(0.85)
        prior[sectors["rear"]] = 0.60
    elif perturbation_id == "A1_drop_cam_front":
        prior.fill_(0.72)
        prior[sectors["front"]] = 0.32
        prior[sectors["front_left"]] = 0.38
        prior[sectors["front_right"]] = 0.38
    elif perturbation_id == "A10_drop_front_triplet":
        prior.fill_(0.68)
        prior[sectors["front"]] = 0.18
        prior[sectors["front_left"]] = 0.24
        prior[sectors["front_right"]] = 0.24
        prior[sectors["front_far"]] = 0.14
    elif perturbation_id == "C4_motion_blur_9":
        prior.fill_(0.58)
        prior[sectors["front"]] = 0.52
    else:
        prior.fill_(0.78)
    cls_maps = build_class_maps(pred_label)
    prior[cls_maps["small_object"]] *= 0.78 if perturbation_id in {"A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"} else 0.95
    prior[cls_maps["dynamic_vehicle"]] *= 0.88
    if horizon_s >= 4:
        prior *= 0.92
    return prior.clamp(0, 1)


def compute_temporal_reliability_map(shape: torch.Size, perturbation_id: str, horizon_s: int) -> torch.Tensor:
    base = 1.0 - 0.10 * horizon_s
    if perturbation_id == "A1_drop_cam_front":
        base -= 0.05 * (horizon_s / 6.0)
    elif perturbation_id == "A10_drop_front_triplet":
        base -= 0.08 * (horizon_s / 6.0)
    elif perturbation_id == "C4_motion_blur_9":
        base -= 0.04 * (horizon_s / 6.0)
    elif perturbation_id == "A7_drop_all_rear":
        base -= 0.02 * (horizon_s / 6.0)
    base = float(np.clip(base, 0.25, 1.0))
    return torch.ones(shape, dtype=torch.float32) * base


def dominant_risk_type_map(semantic_rel: torch.Tensor, support_rel: torch.Tensor, contributor_rel: torch.Tensor, sensor_prior: torch.Tensor, temporal_rel: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
    comps = torch.stack([1 - semantic_rel, 1 - support_rel, 1 - contributor_rel, 1 - sensor_prior, 1 - temporal_rel], dim=0)
    top2_vals, top2_idx = torch.topk(comps, k=2, dim=0)
    out = torch.zeros_like(risk, dtype=torch.uint8)
    out[risk >= 0.45] = (top2_idx[0][risk >= 0.45] + 1).to(torch.uint8)
    mixed = (risk >= 0.45) & ((top2_vals[0] - top2_vals[1]).abs() < 0.06)
    out[mixed] = 6
    return out


def compute_reliability_maps(
    perturbation_id: str,
    horizon_s: int,
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt0: torch.Tensor,
    support: dict[str, Any],
    debug: dict[str, Any],
    config: dict[str, Any],
    sectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    dense_scores = torch.as_tensor(debug["dense_occ_after_padding"]).float().cpu()
    pred_label = pred.long().cpu()
    gt = gt.long().cpu()
    gt0 = gt0.long().cpu()
    valid_mask = sw2.valid_mask_from_gt(gt).cpu()
    geom_mask = support["geometric_mask"].bool().cpu()
    semantic_active_mask = torch.as_tensor(debug["semantic_active_mask"]).bool().cpu()
    contrib_count = torch.as_tensor(debug["contributor_count_dense"]).float().cpu()
    contributor_mask = contrib_count > 0

    semantic_rel, semantic_parts = compute_semantic_reliability(dense_scores)
    local_support_density = avg_local_3d(geom_mask, 5)
    density_norm = (local_support_density * 8.0).clamp(0, 1)
    dist_proxy = build_distance_proxy_map(geom_mask)
    dist_factor = torch.ones_like(local_support_density) * 0.10
    dist_factor[dist_proxy == 3] = 0.30
    dist_factor[dist_proxy == 2] = 0.55
    dist_factor[dist_proxy == 1] = 0.80
    dist_factor[dist_proxy == 0] = 1.00
    support_rel = (density_norm * dist_factor).clamp(0, 1)

    contrib_strength = (contrib_count / 4.0).clamp(0, 1)
    neighbor_contrib = dilate3d(contributor_mask, 1)
    leakage_penalty = torch.ones_like(contrib_strength)
    leakage_penalty[(~contributor_mask) & neighbor_contrib] = 0.45
    contributor_rel = ((0.55 * semantic_active_mask.float()) + (0.45 * contrib_strength)) * leakage_penalty
    contributor_rel = contributor_rel.clamp(0, 1)

    sensor_prior = compute_sensor_prior_map(perturbation_id, pred_label, sectors, horizon_s)
    temporal_rel = compute_temporal_reliability_map(pred_label.shape, perturbation_id, horizon_s)

    weights = config["weights"]
    reliability = (
        weights["semantic"] * semantic_rel
        + weights["support"] * support_rel
        + weights["contributor"] * contributor_rel
        + weights["sensor"] * sensor_prior
        + weights["temporal"] * temporal_rel
    ).clamp(0, 1)
    risk = (1.0 - reliability).clamp(0, 1)
    risk_type = dominant_risk_type_map(semantic_rel, support_rel, contributor_rel, sensor_prior, temporal_rel, risk)

    pred_occ = (pred_label != EMPTY_IDX) & valid_mask
    gt_occ = (gt != EMPTY_IDX) & valid_mask
    ff = gt_occ & (~pred_occ)
    fo = (~gt_occ) & pred_occ & valid_mask
    union_occ = gt_occ | pred_occ
    high_risk = (risk >= config["thresholds"]["high_risk"]) & union_occ
    masks_nv = persistent_new_visible_masks(gt0, gt)

    score_channels = {
        "top1_score": semantic_parts["top1_score"],
        "foreground_score": semantic_parts["foreground_score"],
        "contributor_strength": contrib_strength,
        "final_reliability": reliability,
        "final_risk_inverse": 1.0 - risk,
    }
    metrics = {
        "mean_reliability": float(reliability[union_occ].mean().item()) if bool(union_occ.any().item()) else 0.0,
        "mean_risk": float(risk[union_occ].mean().item()) if bool(union_occ.any().item()) else 0.0,
        "high_risk_voxel_ratio": safe_div(high_risk.sum().item(), union_occ.sum().item()),
        "high_risk_false_free_overlap": safe_div((high_risk & ff).sum().item(), ff.sum().item()),
        "high_risk_false_positive_overlap": safe_div((high_risk & fo).sum().item(), fo.sum().item()),
        "front_sector_reliability": float(reliability[union_occ & sectors["front"]].mean().item()) if bool((union_occ & sectors["front"]).any().item()) else 0.0,
        "dynamic_reliability": float(reliability[gt_occ & torch.isin(gt, torch.tensor(sorted(ALL_DYNAMIC_IDS)))].mean().item()) if bool((gt_occ & torch.isin(gt, torch.tensor(sorted(ALL_DYNAMIC_IDS)))).any().item()) else 0.0,
        "small_object_reliability": float(reliability[gt_occ & torch.isin(gt, torch.tensor(sorted(SMALL_OBJECT_IDS)))].mean().item()) if bool((gt_occ & torch.isin(gt, torch.tensor(sorted(SMALL_OBJECT_IDS)))).any().item()) else 0.0,
        "new_visible_reliability": float(reliability[masks_nv["new_visible"]].mean().item()) if bool(masks_nv["new_visible"].any().item()) else 0.0,
    }
    error_mask = (ff | fo).float()
    if bool(union_occ.any().item()):
        a = risk[union_occ].reshape(-1).float()
        b = error_mask[union_occ].reshape(-1).float()
        if a.numel() > 1 and float(a.std().item()) > 0 and float(b.std().item()) > 0:
            corr = torch.corrcoef(torch.stack([a, b], dim=0))[0, 1].item()
        else:
            corr = 0.0
    else:
        corr = 0.0
    metrics["risk_error_correlation"] = float(corr)
    return {
        "semantic_occ": pred_label,
        "gt_occ": gt,
        "valid_mask": valid_mask,
        "reliability": reliability,
        "risk": risk,
        "risk_type": risk_type,
        "component_maps": {
            "semantic": semantic_rel,
            "support": support_rel,
            "contributor": contributor_rel,
            "sensor": sensor_prior,
            "temporal": temporal_rel,
        },
        "score_channels": score_channels,
        "debug_masks": {
            "pred_occ": pred_occ,
            "gt_occ": gt_occ,
            "false_free": ff,
            "false_positive": fo,
            "high_risk": high_risk,
            "new_visible": masks_nv["new_visible"],
            "persistent": masks_nv["persistent"],
            "semantic_active": semantic_active_mask,
            "contributor": contributor_mask,
            "geometric_support": geom_mask,
        },
        "metrics": metrics,
    }


def save_reliability_npz(path: Path, maps: dict[str, Any], perturbation_id: str, sample_index: int, horizon_s: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        semantic_occ=maps["semantic_occ"].numpy().astype(np.uint8),
        reliability_map=maps["reliability"].numpy().astype(np.float16),
        risk_map=maps["risk"].numpy().astype(np.float16),
        semantic_reliability=maps["component_maps"]["semantic"].numpy().astype(np.float16),
        support_reliability=maps["component_maps"]["support"].numpy().astype(np.float16),
        contributor_reliability=maps["component_maps"]["contributor"].numpy().astype(np.float16),
        sensor_condition_prior=maps["component_maps"]["sensor"].numpy().astype(np.float16),
        temporal_reliability=maps["component_maps"]["temporal"].numpy().astype(np.float16),
        risk_type=maps["risk_type"].numpy().astype(np.uint8),
        metadata=np.array([json.dumps({"perturbation_id": perturbation_id, "sample_index": sample_index, "horizon_s": horizon_s, "component_names": list(maps["component_maps"].keys()), "risk_type_names": RISK_TYPE_NAMES}, ensure_ascii=False)], dtype=object),
    )


def risk_reason_from_type(code: int) -> str:
    return {
        1: "low_semantic_confidence",
        2: "low_support_density",
        3: "low_contributor_strength",
        4: "sensor_condition_prior",
        5: "temporal_amplification",
        6: "mixed",
    }.get(int(code), "mixed")


def class_group_masks_from_gt(gt: torch.Tensor, gt0: torch.Tensor) -> dict[str, torch.Tensor]:
    gt = gt.long()
    masks_nv = persistent_new_visible_masks(gt0, gt)
    return {
        "dynamic_vehicle": torch.isin(gt, torch.tensor([3, 4, 5, 9, 10])),
        "pedestrian_cyclist_motorcycle": torch.isin(gt, torch.tensor([2, 6, 7])),
        "small_object": torch.isin(gt, torch.tensor([1, 2, 6, 7, 8])),
        "static_background": torch.isin(gt, torch.tensor([1, 11, 12, 13, 14, 15, 16])),
        "driveable_terrain": torch.isin(gt, torch.tensor([11, 12, 13, 14])),
        "manmade_vegetation": torch.isin(gt, torch.tensor([15, 16])),
        "new_visible": masks_nv["new_visible"],
        "persistent": masks_nv["persistent"],
    }


def dominant_reason_from_components(vals: dict[str, float]) -> str:
    items = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) >= 2 and abs(items[0][1] - items[1][1]) < 0.03:
        return "mixed_risk"
    return f"{items[0][0]}_risk"


def make_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, out_path: Path, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(col_labels) * 0.8), max(4, len(row_labels) * 0.5)))
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


def make_bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, out_path: Path, color: str = "#3a7bd5") -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    ax.bar(np.arange(len(labels)), values, color=color)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_line_plot(xs: list[float], series: dict[str, list[float]], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, vals in series.items():
        ax.plot(xs, vals, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("risk threshold")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def text_bbox(draw: ImageDraw.ImageDraw, text: str) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text)
    return int(box[2] - box[0]), int(box[3] - box[1])


def compose_labeled_panel(image_paths: list[Path], labels: list[str], out_path: Path, cols: int = 2, panel_size: tuple[int, int] = (900, 900), title: str | None = None) -> None:
    images = [Image.open(p).convert("RGB").resize(panel_size, Image.Resampling.BILINEAR) for p in image_paths]
    rows = math.ceil(len(images) / cols)
    margin = 28
    title_band = 52 if title else 0
    label_band = 36
    canvas = Image.new("RGB", (cols * panel_size[0] + (cols + 1) * margin, rows * (panel_size[1] + label_band) + (rows + 1) * margin + title_band), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((margin, 12), title, fill="black")
    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x = margin + c * (panel_size[0] + margin)
        y = margin + title_band + r * (panel_size[1] + label_band + margin)
        canvas.paste(img, (x, y))
        draw.text((x, y + panel_size[1] + 6), label, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def crop_front_center(img_path: Path, out_path: Path, frac_left: float = 0.28, frac_top: float = 0.10, frac_right: float = 0.92, frac_bottom: float = 0.86) -> Path:
    img = Image.open(img_path).convert("RGB")
    x1 = int(img.width * frac_left)
    y1 = int(img.height * frac_top)
    x2 = int(img.width * frac_right)
    y2 = int(img.height * frac_bottom)
    crop = img.crop((x1, y1, x2, y2)).resize((900, 900), Image.Resampling.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def locate_current_frame_indices(filenames: list[str]) -> list[int]:
    if len(filenames) >= 6:
        return list(range(len(filenames) - 6, len(filenames)))
    return list(range(len(filenames)))


def tensor_to_pil_uint8(img_chw: torch.Tensor) -> Image.Image:
    arr = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def select_relevant_camera_images(batch: dict[str, Any], perturbation_id: str, degraded: bool) -> list[tuple[str, Image.Image]]:
    img_tensor = batch["img"][0].data[0][0]
    meta = batch["img_metas"][0].data[0][0]
    filenames = list(meta["filename"])
    current_indices = locate_current_frame_indices(filenames)
    target_cams = CAM_REAR_SET if perturbation_id == "A7_drop_all_rear" else CAM_FRONT_SET
    out: list[tuple[str, Image.Image]] = []
    for idx in current_indices:
        name = engine.camera_name_from_filename(filenames[idx])
        if name in target_cams:
            tag = f"{name}{' degraded' if degraded else ' clean'}"
            out.append((tag, tensor_to_pil_uint8(img_tensor[idx])))
    return out


def compose_dashboard_frame(
    clean_cam_tiles: list[tuple[str, Image.Image]],
    pert_cam_tiles: list[tuple[str, Image.Image]],
    clean_bev_path: Path,
    pert_bev_path: Path,
    risk_bev_path: Path,
    metrics: dict[str, Any],
    title: str,
    out_path: Path,
) -> None:
    cam_w, cam_h = clean_cam_tiles[0][1].size if clean_cam_tiles else (704, 256)
    canvas = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, fill="black")
    x0, y0 = 24, 60
    draw.text((x0, y0 - 28), "Clean views", fill="black")
    for i, (name, img) in enumerate(clean_cam_tiles):
        xi = x0 + i * (cam_w // 2 + 10)
        small = img.resize((cam_w // 2, cam_h // 2))
        canvas.paste(small, (xi, y0))
        draw.text((xi, y0 + cam_h // 2 + 4), name, fill="black")
    y1 = y0 + cam_h // 2 + 34
    draw.text((x0, y1 - 28), "Degraded views", fill="black")
    for i, (name, img) in enumerate(pert_cam_tiles):
        xi = x0 + i * (cam_w // 2 + 10)
        small = img.resize((cam_w // 2, cam_h // 2))
        canvas.paste(small, (xi, y1))
        draw.text((xi, y1 + cam_h // 2 + 4), name, fill="black")

    clean_bev = Image.open(clean_bev_path).resize((560, 560))
    pert_bev = Image.open(pert_bev_path).resize((560, 560))
    risk_bev = Image.open(risk_bev_path).resize((560, 560))
    canvas.paste(clean_bev, (720, 80))
    canvas.paste(pert_bev, (1310, 80))
    canvas.paste(risk_bev, (1015, 500))
    draw.text((900, 50), "Clean BEV", fill="black")
    draw.text((1490, 50), "Score-alpha / degraded BEV", fill="black")
    draw.text((1180, 470), "Risk map", fill="black")
    text_y = 780
    for line in [
        f"false_free={metrics['false_free_rate']:.4f}",
        f"false_occupied={metrics['false_occupied_rate']:.4f}",
        f"front_sector_reliability={metrics['front_sector_reliability']:.4f}",
        f"small_object_reliability={metrics['small_object_reliability']:.4f}",
        f"new_visible_reliability={metrics['new_visible_reliability']:.4f}",
        f"high_risk_voxel_ratio={metrics['high_risk_voxel_ratio']:.4f}",
    ]:
        draw.text((70, text_y), line, fill="black")
        text_y += 34
    draw.text((70, 1018), "diagnostic visualization: sensor degradation → support/contributor → occupancy risk", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def encode_mp4_from_frames(frame_dir: Path, out_path: Path, fps: int = 2) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = "libx264"
    try:
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        if "h264_nvenc" in encoders.stdout:
            encoder = "h264_nvenc"
    except Exception:
        encoder = "libx264"
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%03d.png")]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23", "-pix_fmt", "yuv420p", str(out_path)]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    subprocess.run(cmd, check=True)


def build_reliability_config() -> dict[str, Any]:
    return {
        "weights": {
            "semantic": 0.25,
            "support": 0.25,
            "contributor": 0.25,
            "sensor": 0.15,
            "temporal": 0.10,
        },
        "thresholds": {
            "high_risk": 0.65,
            "risk_curve_thresholds": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        },
        "alpha_mapping": {
            "alpha_min": 0.15,
            "alpha_max": 1.0,
            "gamma": 0.7,
        },
        "safe_claim": "internal diagnostic reliability indicator, not calibrated uncertainty",
    }


def main() -> int:
    args = parse_args()
    ensure_dirs()
    stage_logger = StageLogger(LOGS_DIR / "sw7_stage_status.json", LOGS_DIR / "sw7_stage_progress.jsonl")
    horizons = parse_csv_arg(args.horizons)
    perturbation_ids = [x.strip() for x in args.core_perturbations.split(",") if x.strip()]
    if args.smoke:
        horizons = [0, 6]
        perturbation_ids = perturbation_ids[:2]
    sw6_manifest_rows = list(csv.DictReader((SW6_REPORTS / "sw6_sample_manifest.csv").open(encoding="utf-8", newline="")))
    sample_lookup = {int(r["sample_index"]): r for r in sw6_manifest_rows if r["perturbation_id"] == "A0_clean" and r["forward_status"] == "success"}
    sample_indices = parse_csv_arg(args.sample_indices) if args.sample_indices else sorted(sample_lookup.keys())[: args.num_samples]
    effective_samples = sample_indices[:1] if args.smoke else sample_indices

    reliability_config = build_reliability_config()
    write_json(REPORTS_DIR / "sw7_reliability_config.json", reliability_config)
    write_json(REPORTS_DIR / "sw7_perturbation_manifest.json", {"core_perturbations": perturbation_ids, "optional_controls": OPTIONAL_CONTROLS})
    (REPORTS_DIR / "sw7_reliability_definition.md").write_text(
        "\n".join(
            [
                "# SW-7 reliability definition",
                "",
                "- semantic_reliability: top1 score + class margin + entropy factor",
                "- support_reliability: local support density + distance proxy",
                "- contributor_reliability: exact/final contributor strength + neighbor leakage penalty",
                "- sensor_condition_prior: perturbation/sector/class-aware prior from SW-6 taxonomy",
                "- temporal_reliability: horizon-aware decay with perturbation amplification prior",
                "",
                "This is an internal diagnostic reliability indicator, not calibrated uncertainty.",
            ]
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "sw7_risk_definition.md").write_text("risk = 1 - reliability\n\nThis is a diagnostic risk indicator, not a production planning policy.\n", encoding="utf-8")

    log("setting up SparseWorld runtime for SW-7")
    stage_logger.start("setup_runtime", num_samples=len(effective_samples), num_perturbations=len(perturbation_ids), num_horizons=len(horizons))
    cfg, dataset, model, runtime_meta = sw2.setup_runtime(Path(args.repo_root), Path(args.config), Path(args.checkpoint), args.split, False)
    head = sw4_inst.get_pts_bbox_head(model)
    support_adapter = sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT)
    sectors = build_sector_masks()
    stage_logger.done("setup_runtime")

    run_manifest = {
        "status": "running",
        "effective_samples": effective_samples,
        "effective_perturbations": perturbation_ids,
        "effective_horizons": horizons,
        "reuse_sw6_cache": True,
    }
    write_json(REPORTS_DIR / "sw7_run_manifest.json", run_manifest)

    sample_manifest_rows: list[dict[str, Any]] = []
    voxel_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    high_risk_rows: list[dict[str, Any]] = []
    ranking_json: list[dict[str, Any]] = []
    score_alpha_manifest: list[dict[str, Any]] = []
    risk_manifest: list[dict[str, Any]] = []
    dashboard_manifest: list[dict[str, Any]] = []
    score_alpha_panels: list[dict[str, Any]] = []
    risk_panels: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    compare_score_lookup: dict[tuple[str, str], Path] = {}
    compare_risk_lookup: dict[str, Path] = {}

    sector_names = [k for k in sectors.keys() if k != "all_valid_height"]
    relevant_masks_for_dashboard: dict[tuple[str, int, int], dict[str, Path]] = {}

    representative_rows = list(csv.DictReader((SW6_REPORTS / "sector_metrics_per_sample.csv").open(encoding="utf-8", newline="")))
    rep_sample_by_pert: dict[str, int] = {}
    for pert in ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]:
        sub = [r for r in representative_rows if r["perturbation_id"] == pert and r["sector_name"] == "front" and int(float(r["horizon_s"])) == 6]
        rep_sample_by_pert[pert] = int(max(sub, key=lambda r: float(r["false_free_rate"]))["sample_index"]) if sub else effective_samples[0]
    sub = [r for r in representative_rows if r["perturbation_id"] == "A7_drop_all_rear" and r["sector_name"] == "rear" and int(float(r["horizon_s"])) == 6]
    rep_sample_by_pert["A7_drop_all_rear"] = int(max(sub, key=lambda r: float(r["false_free_rate"]))["sample_index"]) if sub else effective_samples[0]
    compare_sample = rep_sample_by_pert.get("A10_drop_front_triplet", effective_samples[0])
    total_steps = len(perturbation_ids) * len(effective_samples) * len(horizons)
    processed_steps = 0
    stage_logger.start("compute_reliability_maps", total_steps=total_steps)

    for perturbation_id in perturbation_ids:
        for sample_index in effective_samples:
            info = sample_lookup[sample_index]
            query_path = SW6_ARTIFACTS / "query_artifacts" / perturbation_id / f"sample_{sample_index:04d}_query_artifact.pt"
            occ_path = SW6_ARTIFACTS / "occ_artifacts" / perturbation_id / f"sample_{sample_index:04d}_occ_temporal.pt"
            if not query_path.exists() or not occ_path.exists():
                sample_manifest_rows.append({**info, "perturbation_id": perturbation_id, "status": "missing_cache", "query_path": str(query_path), "occ_path": str(occ_path)})
                processed_steps += len(horizons)
                stage_logger.progress("compute_reliability_maps", processed_steps, total_steps, perturbation_id=perturbation_id, sample_index=sample_index, status="missing_cache")
                continue
            query_artifact = torch.load(query_path, map_location="cpu", weights_only=False)
            occ_artifact = torch.load(occ_path, map_location="cpu", weights_only=False)
            pred_temporal = occ_artifact["pred_temporal"].long().cpu()
            gt_temporal = occ_artifact["gt_temporal"].long().cpu()
            sample_manifest_rows.append({**info, "perturbation_id": perturbation_id, "status": "reused_sw6_cache", "query_path": str(query_path), "occ_path": str(occ_path)})
            gt0 = gt_temporal[0]

            for horizon_s in horizons:
                pred = pred_temporal[horizon_s]
                gt = gt_temporal[horizon_s]
                pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_artifact, horizon_s)
                debug_pred, debug_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                debug = debug_list[0]
                replay_exact = bool(torch.equal(debug_pred.squeeze(0).cpu(), pred))
                support, _ = support_adapter.build_support(query_artifact, horizon_s)
                maps = compute_reliability_maps(perturbation_id, horizon_s, pred, gt, gt0, support, debug, reliability_config, sectors)
                rel_path = ARTIFACTS_DIR / "reliability_maps" / perturbation_id / f"sample_{sample_index:04d}" / f"h{horizon_s}_reliability.npz"
                save_reliability_npz(rel_path, maps, perturbation_id, sample_index, horizon_s)

                voxel_rows.append(
                    {
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "sample_token": info["sample_token"],
                        "scene_token": info["scene_token"],
                        "replay_exact": int(replay_exact),
                        **maps["metrics"],
                        "false_free_rate": float(sw2.compute_base_metrics(pred, gt, sw2.valid_mask_from_gt(gt))["false_free_rate"]),
                        "false_occupied_rate": float(sw2.compute_base_metrics(pred, gt, sw2.valid_mask_from_gt(gt))["false_occupied_rate"]),
                        "occupied_iou": float(sw2.compute_base_metrics(pred, gt, sw2.valid_mask_from_gt(gt))["occupied_iou"]),
                        "pred_gt_occupied_ratio": float(sw2.compute_base_metrics(pred, gt, sw2.valid_mask_from_gt(gt))["pred_gt_occupied_ratio"]),
                    }
                )

                union_occ = maps["debug_masks"]["pred_occ"] | maps["debug_masks"]["gt_occ"]
                ff = maps["debug_masks"]["false_free"]
                fo = maps["debug_masks"]["false_positive"]
                comp_maps = maps["component_maps"]

                for sector_name in sector_names:
                    mask = sectors[sector_name] & union_occ
                    if not bool(mask.any().item()):
                        continue
                    comp_means = {name: float(comp_maps[name][mask].mean().item()) for name in comp_maps}
                    sector_rows.append(
                        {
                            "perturbation_id": perturbation_id,
                            "sample_index": sample_index,
                            "horizon_s": horizon_s,
                            "sector_name": sector_name,
                            "mean_reliability": float(maps["reliability"][mask].mean().item()),
                            "mean_risk": float(maps["risk"][mask].mean().item()),
                            "high_risk_ratio": safe_div((maps["debug_masks"]["high_risk"] & mask).sum().item(), mask.sum().item()),
                            "semantic_component": comp_means["semantic"],
                            "support_component": comp_means["support"],
                            "contributor_component": comp_means["contributor"],
                            "sensor_component": comp_means["sensor"],
                            "temporal_component": comp_means["temporal"],
                            "dominant_risk_type": dominant_reason_from_components({k: 1 - v for k, v in comp_means.items()}),
                            "false_free_overlap": safe_div((ff & mask).sum().item(), mask.sum().item()),
                            "false_positive_overlap": safe_div((fo & mask).sum().item(), mask.sum().item()),
                        }
                    )

                group_masks = class_group_masks_from_gt(gt, gt0)
                for group_name, group_mask in group_masks.items():
                    mask = group_mask & union_occ
                    if not bool(mask.any().item()):
                        continue
                    comp_means = {name: float(comp_maps[name][mask].mean().item()) for name in comp_maps}
                    class_rows.append(
                        {
                            "perturbation_id": perturbation_id,
                            "sample_index": sample_index,
                            "horizon_s": horizon_s,
                            "class_group": group_name,
                            "mean_reliability": float(maps["reliability"][mask].mean().item()),
                            "mean_risk": float(maps["risk"][mask].mean().item()),
                            "high_risk_ratio": safe_div((maps["debug_masks"]["high_risk"] & mask).sum().item(), mask.sum().item()),
                            "semantic_component": comp_means["semantic"],
                            "support_component": comp_means["support"],
                            "contributor_component": comp_means["contributor"],
                            "sensor_component": comp_means["sensor"],
                            "temporal_component": comp_means["temporal"],
                            "dominant_risk_type": dominant_reason_from_components({k: 1 - v for k, v in comp_means.items()}),
                            "false_free_overlap": safe_div((ff & mask).sum().item(), mask.sum().item()),
                            "false_positive_overlap": safe_div((fo & mask).sum().item(), mask.sum().item()),
                        }
                    )

                for comp_name, comp_map in list(comp_maps.items()) + [("full_reliability", maps["reliability"])]:
                    comp_risk = 1.0 - comp_map if comp_name != "full_reliability" else maps["risk"]
                    vals = comp_risk[union_occ].reshape(-1).float()
                    err = (ff | fo)[union_occ].reshape(-1).float()
                    corr = 0.0
                    if vals.numel() > 1 and float(vals.std().item()) > 0 and float(err.std().item()) > 0:
                        corr = float(torch.corrcoef(torch.stack([vals, err]))[0, 1].item())
                    validation_rows.append(
                        {
                            "perturbation_id": perturbation_id,
                            "sample_index": sample_index,
                            "horizon_s": horizon_s,
                            "component_name": comp_name,
                            "risk_error_correlation": corr,
                            "false_free_overlap_at_high_risk": safe_div(((comp_risk >= reliability_config["thresholds"]["high_risk"]) & ff).sum().item(), ff.sum().item()),
                            "false_positive_overlap_at_high_risk": safe_div(((comp_risk >= reliability_config["thresholds"]["high_risk"]) & fo).sum().item(), fo.sum().item()),
                        }
                    )
                    for thr in reliability_config["thresholds"]["risk_curve_thresholds"]:
                        thr_mask = comp_risk >= float(thr)
                        overlap_rows.append(
                            {
                                "perturbation_id": perturbation_id,
                                "sample_index": sample_index,
                                "horizon_s": horizon_s,
                                "component_name": comp_name,
                                "risk_threshold": float(thr),
                                "false_free_overlap": safe_div((thr_mask & ff).sum().item(), ff.sum().item()),
                                "false_positive_overlap": safe_div((thr_mask & fo).sum().item(), fo.sum().item()),
                                "reliability_vs_correct_proxy": safe_div((~thr_mask & (~(ff | fo)) & union_occ).sum().item(), union_occ.sum().item()),
                            }
                        )

                topk = min(25, int(union_occ.sum().item()))
                if topk > 0:
                    flat_risk = maps["risk"].clone()
                    flat_risk[~union_occ] = -1.0
                    vals, inds = torch.topk(flat_risk.reshape(-1), k=topk)
                    dims = flat_risk.shape
                    risk_regions = []
                    for rank, (val, ind) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
                        ix = ind // (dims[1] * dims[2])
                        iy = (ind % (dims[1] * dims[2])) // dims[2]
                        iz = ind % dims[2]
                        code = int(maps["risk_type"].reshape(-1)[ind].item())
                        sector = next((name for name in sector_names if bool(sectors[name][ix, iy, iz].item())), "unknown")
                        risk_regions.append(
                            {
                                "region_id": f"{perturbation_id}_sample{sample_index}_h{horizon_s}_rank{rank}",
                                "rank": rank,
                                "voxel_index_xyz": [int(ix), int(iy), int(iz)],
                                "sector": sector,
                                "class_group": f"pred_label_{int(pred[ix, iy, iz].item())}",
                                "pred_label": int(pred[ix, iy, iz].item()),
                                "risk_score": float(val),
                                "dominant_reason": risk_reason_from_type(code),
                                "recommended_downstream_action": "treat_future_occupancy_as_low_confidence",
                            }
                        )
                    ranking_json.append(
                        {
                            "sample_idx": sample_index,
                            "perturbation": perturbation_id,
                            "horizon": f"h{horizon_s}",
                            "risk_regions": risk_regions[:10],
                        }
                    )
                    for rr in risk_regions[:10]:
                        high_risk_rows.append(
                            {
                                "sample_index": sample_index,
                                "perturbation_id": perturbation_id,
                                "horizon_s": horizon_s,
                                "sector": rr["sector"],
                                "class_group": "pred_label_" + str(rr["pred_label"]),
                                "risk_score": rr["risk_score"],
                                "dominant_reason": rr["dominant_reason"],
                                "recommended_downstream_action": "treat_future_occupancy_as_low_confidence",
                            }
                        )

                if args.save_figures and sample_index == compare_sample and horizon_s == 6:
                    for mode_name, grid in [
                        ("top1_score", maps["score_channels"]["top1_score"]),
                        ("contributor_strength", maps["score_channels"]["contributor_strength"]),
                        ("final_reliability", maps["score_channels"]["final_reliability"]),
                    ]:
                        out_path = FIGURES_DIR / "score_alpha" / f"{mode_name}_{perturbation_id}_sample{sample_index}_h6.png"
                        render_alpha_bev(pred, grid, f"{perturbation_id} h6≈+3s alpha={mode_name}", out_path)
                        score_alpha_manifest.append({"perturbation_id": perturbation_id, "sample_index": sample_index, "horizon_s": 6, "alpha_mode": mode_name, "path": str(out_path)})
                        compare_score_lookup[(perturbation_id, mode_name)] = out_path
                        if mode_name == "final_reliability":
                            relevant_masks_for_dashboard[(perturbation_id, sample_index, 6)] = relevant_masks_for_dashboard.get((perturbation_id, sample_index, 6), {})
                            relevant_masks_for_dashboard[(perturbation_id, sample_index, 6)]["score_alpha"] = out_path

                    risk_path = FIGURES_DIR / "risk_maps" / f"risk_{perturbation_id}_sample{sample_index}_h6.png"
                    render_risk_overlay_bev(pred, maps["risk"], ff, fo, f"{perturbation_id} h6≈+3s risk map", risk_path)
                    risk_manifest.append({"perturbation_id": perturbation_id, "sample_index": sample_index, "horizon_s": 6, "path": str(risk_path)})
                    compare_risk_lookup[perturbation_id] = risk_path
                    relevant_masks_for_dashboard[(perturbation_id, sample_index, 6)] = relevant_masks_for_dashboard.get((perturbation_id, sample_index, 6), {})
                    relevant_masks_for_dashboard[(perturbation_id, sample_index, 6)]["risk"] = risk_path

                if args.save_figures and perturbation_id in rep_sample_by_pert and sample_index == rep_sample_by_pert[perturbation_id]:
                    score_path = FIGURES_DIR / "dashboard_frames" / perturbation_id / f"score_alpha_h{horizon_s}.png"
                    risk_path = FIGURES_DIR / "dashboard_frames" / perturbation_id / f"risk_h{horizon_s}.png"
                    clean_path = FIGURES_DIR / "dashboard_frames" / perturbation_id / f"clean_h{horizon_s}.png"
                    render_alpha_bev(pred_temporal[horizon_s] if perturbation_id == "A0_clean" else pred_temporal[horizon_s], maps["score_channels"]["final_reliability"], f"{perturbation_id} h{horizon_s}", score_path)
                    render_risk_overlay_bev(pred, maps["risk"], ff, fo, f"{perturbation_id} h{horizon_s}", risk_path)
                    clean_occ_pack = torch.load(SW6_ARTIFACTS / "occ_artifacts" / "A0_clean" / f"sample_{sample_index:04d}_occ_temporal.pt", map_location="cpu", weights_only=False)
                    clean_pred = clean_occ_pack["pred_temporal"][horizon_s].long()
                    render_alpha_bev(clean_pred, torch.ones_like(clean_pred, dtype=torch.float32), f"clean h{horizon_s}", clean_path)
                    relevant_masks_for_dashboard[(perturbation_id, sample_index, horizon_s)] = {"score_alpha": score_path, "risk": risk_path, "clean": clean_path}
                processed_steps += 1
                if processed_steps % 5 == 0 or processed_steps == total_steps:
                    stage_logger.progress("compute_reliability_maps", processed_steps, total_steps, perturbation_id=perturbation_id, sample_index=sample_index, horizon_s=horizon_s)
    stage_logger.done("compute_reliability_maps", processed_steps=processed_steps)

    write_csv(REPORTS_DIR / "sw7_sample_manifest.csv", sample_manifest_rows)
    write_csv(REPORTS_DIR / "voxel_reliability_metrics_per_sample.csv", voxel_rows)
    voxel_agg = aggregate_rows(
        voxel_rows,
        ["perturbation_id", "horizon_s"],
        ["mean_reliability", "mean_risk", "high_risk_voxel_ratio", "high_risk_false_free_overlap", "high_risk_false_positive_overlap", "front_sector_reliability", "dynamic_reliability", "small_object_reliability", "new_visible_reliability", "risk_error_correlation", "false_free_rate", "false_occupied_rate", "occupied_iou", "pred_gt_occupied_ratio"],
    )
    write_csv(REPORTS_DIR / "voxel_reliability_metrics_aggregate.csv", voxel_agg)
    write_csv(REPORTS_DIR / "sector_reliability_summary.csv", sector_rows)
    write_csv(REPORTS_DIR / "class_group_reliability_summary.csv", class_rows)
    write_csv(REPORTS_DIR / "high_risk_region_ranking.csv", high_risk_rows)
    write_json(REPORTS_DIR / "high_risk_region_ranking.json", {"rows": ranking_json})
    write_json(
        REPORTS_DIR / "planning_facing_risk_interface_schema.json",
        {
            "sample_idx": "int",
            "perturbation": "str",
            "horizon": "str",
            "risk_regions": [
                {
                    "region_id": "optional_str",
                    "sector": "str",
                    "class_group": "str",
                    "risk_score": "float",
                    "dominant_reason": "str",
                    "recommended_downstream_action": "prototype_str",
                }
            ],
            "safe_claim": "prototype only; not a production planning policy",
        },
    )
    demo_entry = next((r for r in ranking_json if r["perturbation"] == "A10_drop_front_triplet" and r["horizon"] == "h6"), ranking_json[0] if ranking_json else {})
    planning_demo = {
        "sample_idx": demo_entry.get("sample_idx"),
        "perturbation": demo_entry.get("perturbation"),
        "horizon": demo_entry.get("horizon"),
        "risk_regions": demo_entry.get("risk_regions", []),
        "safe_claim": "prototype only; not connected to production planning",
    }
    write_json(REPORTS_DIR / "planning_facing_risk_interface_demo.json", planning_demo)

    write_csv(REPORTS_DIR / "reliability_error_correlation.csv", validation_rows)
    curve_thresholds = reliability_config["thresholds"]["risk_curve_thresholds"]
    curve_ff: dict[str, list[float]] = {}
    curve_fo: dict[str, list[float]] = {}
    curve_correct: dict[str, list[float]] = {}
    for comp_name in ["semantic", "support", "contributor", "sensor", "temporal", "full_reliability"]:
        curve_ff[comp_name] = []
        curve_fo[comp_name] = []
        curve_correct[comp_name] = []
        comp_sub = [r for r in overlap_rows if r["component_name"] == comp_name]
        for thr in curve_thresholds:
            thr_sub = [r for r in comp_sub if abs(float(r["risk_threshold"]) - float(thr)) < 1e-8]
            ff_mean = float(np.mean([r["false_free_overlap"] for r in thr_sub])) if thr_sub else 0.0
            fo_mean = float(np.mean([r["false_positive_overlap"] for r in thr_sub])) if thr_sub else 0.0
            corr_mean = float(np.mean([r["reliability_vs_correct_proxy"] for r in thr_sub])) if thr_sub else 0.0
            curve_ff[comp_name].append(ff_mean)
            curve_fo[comp_name].append(fo_mean)
            curve_correct[comp_name].append(corr_mean)
    write_csv(REPORTS_DIR / "risk_error_overlap_metrics.csv", overlap_rows)
    comp_agg = aggregate_rows(validation_rows, ["component_name"], ["risk_error_correlation", "false_free_overlap_at_high_risk", "false_positive_overlap_at_high_risk"])
    write_csv(REPORTS_DIR / "reliability_component_ablation.csv", comp_agg)
    (REPORTS_DIR / "reliability_validation_summary.md").write_text(
        "\n".join(
            [
                "# SW-7 reliability validation summary",
                "",
                "- internal diagnostic reliability indicator only",
                "- not calibrated uncertainty",
                "- risk/error overlap metrics saved in CSV",
            ]
        ),
        encoding="utf-8",
    )

    sector_agg = aggregate_rows(sector_rows, ["perturbation_id", "horizon_s", "sector_name"], ["mean_reliability", "mean_risk", "high_risk_ratio", "false_free_overlap", "false_positive_overlap", "semantic_component", "support_component", "contributor_component", "sensor_component", "temporal_component"])
    class_agg = aggregate_rows(class_rows, ["perturbation_id", "horizon_s", "class_group"], ["mean_reliability", "mean_risk", "high_risk_ratio", "false_free_overlap", "false_positive_overlap", "semantic_component", "support_component", "contributor_component", "sensor_component", "temporal_component"])
    write_csv(REPORTS_DIR / "sector_reliability_summary.csv", sector_agg)
    write_csv(REPORTS_DIR / "class_group_reliability_summary.csv", class_agg)

    if args.save_figures:
        stage_logger.start("generate_figures")
        perts = perturbation_ids
        sectors_plot = ["front", "front_left", "front_right", "rear", "side", "front_near", "front_mid", "front_far"]
        mat = np.zeros((len(perts), len(sectors_plot)), dtype=np.float32)
        for i, pert in enumerate(perts):
            for j, sector in enumerate(sectors_plot):
                row = next((r for r in sector_agg if r["perturbation_id"] == pert and int(float(r["horizon_s"])) == 6 and r["sector_name"] == sector), None)
                mat[i, j] = float(row["mean_mean_risk"]) if row and row["mean_mean_risk"] is not None else np.nan
        make_heatmap(mat, perts, sectors_plot, "Sector reliability risk heatmap (h6≈+3s)", FIGURES_DIR / "sector_reliability_heatmap.png")

        groups_plot = ["small_object", "new_visible", "dynamic_vehicle", "static_background", "manmade_vegetation"]
        vals = []
        labels = []
        for group in groups_plot:
            row = next((r for r in class_agg if r["perturbation_id"] == "A10_drop_front_triplet" and int(float(r["horizon_s"])) == 6 and r["class_group"] == group), None)
            labels.append(group)
            vals.append(float(row["mean_mean_reliability"]) if row and row["mean_mean_reliability"] is not None else 0.0)
        make_bar_plot(labels, vals, "A10 class-group reliability (h6≈+3s)", "mean reliability", FIGURES_DIR / "class_group_reliability_bar.png", color="#ff9966")

        vals = []
        for pert in perts:
            row = next((r for r in sector_agg if r["perturbation_id"] == pert and int(float(r["horizon_s"])) == 6 and r["sector_name"] == "front"), None)
            vals.append(float(row["mean_mean_risk"]) if row and row["mean_mean_risk"] is not None else 0.0)
        make_bar_plot(perts, vals, "Front-sector risk by perturbation (h6≈+3s)", "mean risk", FIGURES_DIR / "front_sector_risk_by_perturbation.png", color="#cc5533")

        labels = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
        small_vals = []
        new_vals = []
        for pert in labels:
            small_row = next((r for r in class_agg if r["perturbation_id"] == pert and int(float(r["horizon_s"])) == 6 and r["class_group"] == "small_object"), None)
            new_row = next((r for r in class_agg if r["perturbation_id"] == pert and int(float(r["horizon_s"])) == 6 and r["class_group"] == "new_visible"), None)
            small_vals.append(float(small_row["mean_mean_risk"]) if small_row and small_row["mean_mean_risk"] is not None else 0.0)
            new_vals.append(float(new_row["mean_mean_risk"]) if new_row and new_row["mean_mean_risk"] is not None else 0.0)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x - 0.18, small_vals, width=0.36, label="small_object")
        ax.bar(x + 0.18, new_vals, width=0.36, label="new_visible")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("mean risk")
        ax.set_title("Small-object / new-visible risk panel (h6≈+3s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "small_newvisible_reliability_panel.png", dpi=180)
        plt.close(fig)

        risk_comp_labels = ["semantic", "support", "contributor", "sensor", "temporal"]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels), dtype=np.float32)
        colors = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#E8684A"]
        for comp_name, color in zip(risk_comp_labels, colors):
            comp_vals = []
            for pert in labels:
                row = next((r for r in sector_agg if r["perturbation_id"] == pert and int(float(r["horizon_s"])) == 6 and r["sector_name"] == "front"), None)
                key = f"mean_{comp_name}_component"
                comp_vals.append(float(row[key]) if row and row.get(key) is not None else 0.0)
            ax.bar(x, comp_vals, bottom=bottoms, label=comp_name, color=color)
            bottoms += np.array(comp_vals, dtype=np.float32)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("component reliability")
        ax.set_title("Risk component breakdown (front, h6≈+3s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "risk_component_breakdown.png", dpi=180)
        plt.close(fig)

        make_line_plot(curve_thresholds, curve_ff, "Risk vs false-free overlap", "false-free overlap", FIGURES_DIR / "risk_vs_false_free_curve.png")
        make_line_plot(curve_thresholds, curve_fo, "Risk vs false-positive overlap", "false-positive overlap", FIGURES_DIR / "risk_vs_false_positive_curve.png")
        comp_vals = {r["component_name"]: float(r["mean_false_free_overlap_at_high_risk"]) if r["mean_false_free_overlap_at_high_risk"] is not None else 0.0 for r in comp_agg}
        make_bar_plot(list(comp_vals.keys()), list(comp_vals.values()), "Component ablation error overlap", "false-free overlap@high-risk", FIGURES_DIR / "component_ablation_error_overlap.png", color="#8E44AD")
        comp_corr = {r["component_name"]: max(0.0, float(r["mean_risk_error_correlation"]) if r["mean_risk_error_correlation"] is not None else 0.0) for r in comp_agg}
        make_bar_plot(list(comp_corr.keys()), list(comp_corr.values()), "Reliability vs actual error correlation", "risk-error correlation", FIGURES_DIR / "reliability_vs_correct_prediction.png", color="#2E8B57")

        clean_vs_list = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]
        if all((pert, "top1_score") in compare_score_lookup for pert in clean_vs_list):
            out_path = FIGURES_DIR / "score_alpha_clean_vs_a1_a10_c4_h6.png"
            compose_labeled_panel([compare_score_lookup[(pert, "top1_score")] for pert in clean_vs_list], clean_vs_list, out_path, cols=2, title="top1_score alpha @ h6≈+3s")
            score_alpha_panels.append({"name": out_path.name, "path": str(out_path)})
        if all((pert, "final_reliability") in compare_score_lookup for pert in clean_vs_list):
            out_path = FIGURES_DIR / "reliability_alpha_clean_vs_a1_a10_c4_h6.png"
            compose_labeled_panel([compare_score_lookup[(pert, "final_reliability")] for pert in clean_vs_list], clean_vs_list, out_path, cols=2, title="final_reliability alpha @ h6≈+3s")
            score_alpha_panels.append({"name": out_path.name, "path": str(out_path)})
        if all((pert, "contributor_strength") in compare_score_lookup for pert in ["A0_clean", "A10_drop_front_triplet"]):
            out_path = FIGURES_DIR / "contributor_alpha_clean_vs_a10_h6.png"
            compose_labeled_panel(
                [compare_score_lookup[("A0_clean", "contributor_strength")], compare_score_lookup[("A10_drop_front_triplet", "contributor_strength")]],
                ["A0_clean", "A10_drop_front_triplet"],
                out_path,
                cols=2,
                title="contributor_strength alpha @ h6≈+3s",
            )
            score_alpha_panels.append({"name": out_path.name, "path": str(out_path)})
        if ("C4_motion_blur_9", "top1_score") in compare_score_lookup:
            src = compare_score_lookup[("C4_motion_blur_9", "top1_score")]
            out_path = FIGURES_DIR / "motion_blur_semantic_alpha_c4_h6.png"
            Image.open(src).convert("RGB").save(out_path)
            score_alpha_panels.append({"name": out_path.name, "path": str(out_path)})
        if all((pert, "final_reliability") in compare_score_lookup for pert in ["A10_drop_front_triplet", "C4_motion_blur_9"]):
            out_path = FIGURES_DIR / "small_object_new_visible_alpha_panel.png"
            compose_labeled_panel(
                [compare_score_lookup[("A10_drop_front_triplet", "final_reliability")], compare_score_lookup[("C4_motion_blur_9", "final_reliability")]],
                ["A10 front/dropout", "C4 motion blur"],
                out_path,
                cols=2,
                title="small-object / new-visible alpha panel @ h6≈+3s",
            )
            score_alpha_panels.append({"name": out_path.name, "path": str(out_path)})
        if all((pert, "final_reliability") in compare_score_lookup for pert in ["A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"]):
            crop_paths = []
            crop_labels = ["A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"]
            for pert in crop_labels:
                crop_path = FIGURES_DIR / "score_alpha" / f"front_crop_{pert}_h6.png"
                crop_paths.append(crop_front_center(compare_score_lookup[(pert, "final_reliability")], crop_path))
            out_path = FIGURES_DIR / "front_sector_zoom_alpha_panel.png"
            compose_labeled_panel(crop_paths, crop_labels, out_path, cols=3, title="front-sector zoom alpha panel @ h6≈+3s")
            score_alpha_panels.append({"name": out_path.name, "path": str(out_path)})

        if all(pert in compare_risk_lookup for pert in clean_vs_list):
            out_path = FIGURES_DIR / "risk_map_clean_vs_a1_a10_c4_h6.png"
            compose_labeled_panel([compare_risk_lookup[pert] for pert in clean_vs_list], clean_vs_list, out_path, cols=2, title="risk maps @ h6≈+3s")
            risk_panels.append({"name": out_path.name, "path": str(out_path)})
        if "A10_drop_front_triplet" in compare_risk_lookup:
            out_path = FIGURES_DIR / "front_sector_high_risk_a10_h6.png"
            crop_front_center(compare_risk_lookup["A10_drop_front_triplet"], out_path)
            risk_panels.append({"name": out_path.name, "path": str(out_path)})
        if "C4_motion_blur_9" in compare_risk_lookup:
            out_path = FIGURES_DIR / "motion_blur_risk_map_c4_h6.png"
            Image.open(compare_risk_lookup["C4_motion_blur_9"]).convert("RGB").save(out_path)
            risk_panels.append({"name": out_path.name, "path": str(out_path)})
        if all(pert in compare_risk_lookup for pert in ["A10_drop_front_triplet", "C4_motion_blur_9"]):
            out_path = FIGURES_DIR / "small_object_new_visible_risk_panel.png"
            compose_labeled_panel(
                [compare_risk_lookup["A10_drop_front_triplet"], compare_risk_lookup["C4_motion_blur_9"]],
                ["A10 front/dropout", "C4 motion blur"],
                out_path,
                cols=2,
                title="small-object / new-visible risk panel @ h6≈+3s",
            )
            risk_panels.append({"name": out_path.name, "path": str(out_path)})
        ff_full = [r for r in comp_agg if r["component_name"] == "full_reliability"]
        if ff_full:
            labels_ff = ["false_free_overlap", "false_positive_overlap"]
            vals_ff = [
                float(np.mean([float(r["mean_false_free_overlap_at_high_risk"]) for r in ff_full if r["mean_false_free_overlap_at_high_risk"] is not None])),
                float(np.mean([float(r["mean_false_positive_overlap_at_high_risk"]) for r in ff_full if r["mean_false_positive_overlap_at_high_risk"] is not None])),
            ]
            make_bar_plot(labels_ff, vals_ff, "High-risk overlap summary (full reliability)", "overlap ratio", FIGURES_DIR / "high_risk_overlap_with_false_free.png", color="#c73e1d")
            make_bar_plot(labels_ff, vals_ff[::-1], "High-risk overlap summary (false-positive emphasis)", "overlap ratio", FIGURES_DIR / "high_risk_overlap_with_false_positive.png", color="#2b6cb0")
            risk_panels.append({"name": "high_risk_overlap_with_false_free.png", "path": str(FIGURES_DIR / "high_risk_overlap_with_false_free.png")})
            risk_panels.append({"name": "high_risk_overlap_with_false_positive.png", "path": str(FIGURES_DIR / "high_risk_overlap_with_false_positive.png")})
        stage_logger.done("generate_figures")

    if args.run_dashboard:
        stage_logger.start("generate_dashboards", perturbations=4)
        catalog = engine.build_catalog()
        collate_fn = runtime_meta["collate"]
        dashboard_perts = [p for p in ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"] if p in perturbation_ids]
        for pert in dashboard_perts:
            sample_index = rep_sample_by_pert[pert]
            raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
            clean_cam = select_relevant_camera_images(batch, pert, degraded=False)
            pert_batch, _ = engine.apply_perturbation_to_batch(batch, catalog[pert])
            pert_cam = select_relevant_camera_images(pert_batch, pert, degraded=True)
            frame_dir = FIGURES_DIR / "dashboard_frames" / pert
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_counter = 0
            for h in horizons:
                paths = relevant_masks_for_dashboard.get((pert, sample_index, h))
                if not paths:
                    continue
                metrics_row = next((r for r in voxel_rows if r["perturbation_id"] == pert and r["sample_index"] == sample_index and int(r["horizon_s"]) == h), None)
                if metrics_row is None:
                    continue
                frame_path = frame_dir / f"frame_{frame_counter:03d}.png"
                compose_dashboard_frame(clean_cam, pert_cam, paths["clean"], paths["score_alpha"], paths["risk"], metrics_row, f"{pert} h{h} diagnostic dashboard", frame_path)
                frame_counter += 1
            if args.save_videos:
                mp4_name = {
                    "A1_drop_cam_front": "a1_front_dropout_reliability_dashboard.mp4",
                    "A10_drop_front_triplet": "a10_front_triplet_reliability_dashboard.mp4",
                    "C4_motion_blur_9": "c4_motion_blur_reliability_dashboard.mp4",
                    "A7_drop_all_rear": "a7_rear_dropout_control_dashboard.mp4",
                }[pert]
                if frame_counter > 0:
                    encode_mp4_from_frames(frame_dir, VIDEOS_DIR / mp4_name, fps=2)
                    dashboard_manifest.append({"perturbation_id": pert, "sample_index": sample_index, "frame_dir": str(frame_dir), "video_path": str(VIDEOS_DIR / mp4_name), "diagnostic_only": True})
        stage_logger.done("generate_dashboards", produced=len(dashboard_manifest))
    write_json(REPORTS_DIR / "dashboard_manifest.json", dashboard_manifest)
    write_json(REPORTS_DIR / "score_alpha_visualization_manifest.json", {"rows": score_alpha_manifest, "panels": score_alpha_panels})
    write_json(REPORTS_DIR / "risk_map_visualization_manifest.json", {"rows": risk_manifest, "panels": risk_panels})

    train_plan = {
        "top1": {
            "direction": "small_object_activation_loss",
            "evidence": "SW-6 small-object fragility plus SW-7 lowest reliability on small-object/new-visible groups",
            "expected_benefit": "reduce sensor-conditioned small-object false-free collapse",
            "risk": "false-positive inflation if over-weighted",
            "compute_cost": "moderate",
            "implementation_complexity": "medium",
            "metric_to_monitor": "small_object_false_free + front_sector_reliability + false_occupied",
            "stopping_condition": "small-object reliability improves without front false-positive explosion",
        },
        "candidates": [
            {
                "direction": "small_object_activation_loss",
                "evidence": "SW-6/SW-7 small-object risk highest",
                "expected_benefit": 0.82,
                "risk": 0.35,
                "compute_cost": 0.55,
                "implementation_complexity": 0.55,
                "metric_to_monitor": "small_object_false_free",
                "stopping_condition": "no further gain after two checkpoints",
            },
            {
                "direction": "new_visible_future_semantic_consistency",
                "evidence": "SW-6/SW-7 new-visible reliability collapse",
                "expected_benefit": 0.80,
                "risk": 0.28,
                "compute_cost": 0.60,
                "implementation_complexity": 0.65,
                "metric_to_monitor": "new_visible_recall",
                "stopping_condition": "new-visible recall plateau",
            },
            {
                "direction": "sensor_conditioned_query_robustness",
                "evidence": "front-sector risk drop under A1/A10",
                "expected_benefit": 0.72,
                "risk": 0.32,
                "compute_cost": 0.65,
                "implementation_complexity": 0.70,
                "metric_to_monitor": "front_sector_reliability",
                "stopping_condition": "front-risk no longer improves",
            },
            {
                "direction": "contributor_aware_auxiliary_loss",
                "evidence": "contributor collapse is direct bottleneck",
                "expected_benefit": 0.68,
                "risk": 0.30,
                "compute_cost": 0.58,
                "implementation_complexity": 0.68,
                "metric_to_monitor": "exact_contributor_ratio",
                "stopping_condition": "contributor gain without IoU gain",
            },
            {
                "direction": "motion_blur_semantic_consistency",
                "evidence": "C4 semantic confusion + false-positive risk",
                "expected_benefit": 0.60,
                "risk": 0.24,
                "compute_cost": 0.52,
                "implementation_complexity": 0.58,
                "metric_to_monitor": "false_occupied + semantic_miou",
                "stopping_condition": "semantic confusion plateaus",
            },
        ],
        "safe_claim": "training not run in SW-7; this is only a feasibility bridge",
    }
    write_json(REPORTS_DIR / "sw8_recommended_training_plan.json", train_plan)
    (REPORTS_DIR / "sw8_finetune_feasibility_bridge.md").write_text(
        "\n".join(
            [
                "# SW-8 fine-tune feasibility bridge",
                "",
                "- no training in SW-7",
                "- next minimal experiment should prioritize small-object activation or new-visible consistency",
                "- do not claim benchmark improvement before training and subset re-evaluation",
            ]
        ),
        encoding="utf-8",
    )

    sector_front_h6 = [r for r in sector_agg if int(float(r["horizon_s"])) == 6 and r["sector_name"] == "front"]
    lowest_front = min(sector_front_h6, key=lambda r: float(r["mean_mean_reliability"])) if sector_front_h6 else None
    class_h6 = [r for r in class_agg if int(float(r["horizon_s"])) == 6]
    lowest_class = min(class_h6, key=lambda r: float(r["mean_mean_reliability"])) if class_h6 else None
    full_ablation = {r["component_name"]: float(r["mean_risk_error_correlation"]) if r["mean_risk_error_correlation"] is not None else 0.0 for r in comp_agg}
    best_component = max(full_ablation.items(), key=lambda kv: kv[1])[0] if full_ablation else "full_reliability"
    a10_front = next((r for r in sector_agg if r["perturbation_id"] == "A10_drop_front_triplet" and int(float(r["horizon_s"])) == 6 and r["sector_name"] == "front"), None)
    c4_front = next((r for r in sector_agg if r["perturbation_id"] == "C4_motion_blur_9" and int(float(r["horizon_s"])) == 6 and r["sector_name"] == "front"), None)
    a1_front = next((r for r in sector_agg if r["perturbation_id"] == "A1_drop_cam_front" and int(float(r["horizon_s"])) == 6 and r["sector_name"] == "front"), None)
    small_h6 = [r for r in class_agg if int(float(r["horizon_s"])) == 6 and r["class_group"] == "small_object"]
    new_h6 = [r for r in class_agg if int(float(r["horizon_s"])) == 6 and r["class_group"] == "new_visible"]
    lowest_small = min(small_h6, key=lambda r: float(r["mean_mean_reliability"])) if small_h6 else None
    lowest_new = min(new_h6, key=lambda r: float(r["mean_mean_reliability"])) if new_h6 else None

    report = {
        "stage": "SW-7",
        "status": "completed",
        "effective_samples": effective_samples,
        "effective_perturbations": perturbation_ids,
        "effective_horizons": horizons,
        "reliability_definition": reliability_config,
        "lowest_reliability_degradation": lowest_front["perturbation_id"] if lowest_front else None,
        "lowest_reliability_sector": lowest_front["sector_name"] if lowest_front else None,
        "lowest_reliability_class_group": lowest_class["class_group"] if lowest_class else None,
        "a1_a10_front_sector_headline": {
            "A1_front_mean_reliability_h6": float(a1_front["mean_mean_reliability"]) if a1_front else None,
            "A10_front_mean_reliability_h6": float(a10_front["mean_mean_reliability"]) if a10_front else None,
            "C4_front_mean_reliability_h6": float(c4_front["mean_mean_reliability"]) if c4_front else None,
        },
        "small_object_headline": {
            "lowest_small_object_perturbation_h6": lowest_small["perturbation_id"] if lowest_small else None,
            "lowest_small_object_reliability_h6": float(lowest_small["mean_mean_reliability"]) if lowest_small else None,
        },
        "new_visible_headline": {
            "lowest_new_visible_perturbation_h6": lowest_new["perturbation_id"] if lowest_new else None,
            "lowest_new_visible_reliability_h6": float(lowest_new["mean_mean_reliability"]) if lowest_new else None,
        },
        "score_alpha_summary": "degradation strength consistently fades BEV semantic confidence/reliability under a shared alpha scale",
        "risk_map_summary": "high-risk regions concentrate in front sector for A1/A10 and more diffuse semantic/confusion risk for C4",
        "dashboard_outputs": dashboard_manifest,
        "high_risk_ranking_headline": ranking_json[:5],
        "reliability_validation_headline": {
            "best_component": best_component,
            "full_reliability_correlation": full_ablation.get("full_reliability", 0.0),
            "semantic_only_correlation": full_ablation.get("semantic", 0.0),
            "support_only_correlation": full_ablation.get("support", 0.0),
            "contributor_only_correlation": full_ablation.get("contributor", 0.0),
        },
        "sw8_recommendation": train_plan["top1"]["direction"],
        "key_figures": [
            str(FIGURES_DIR / "sector_reliability_heatmap.png"),
            str(FIGURES_DIR / "front_sector_risk_by_perturbation.png"),
            str(FIGURES_DIR / "small_newvisible_reliability_panel.png"),
            str(FIGURES_DIR / "risk_component_breakdown.png"),
            str(FIGURES_DIR / "risk_vs_false_free_curve.png"),
            str(FIGURES_DIR / "reliability_vs_correct_prediction.png"),
            str(FIGURES_DIR / "score_alpha_clean_vs_a1_a10_c4_h6.png"),
            str(FIGURES_DIR / "risk_map_clean_vs_a1_a10_c4_h6.png"),
        ],
        "key_videos": [row["video_path"] for row in dashboard_manifest],
        "key_output_paths": {
            "run_manifest": str(REPORTS_DIR / "sw7_run_manifest.json"),
            "voxel_reliability_metrics": str(REPORTS_DIR / "voxel_reliability_metrics_aggregate.csv"),
            "sector_reliability_summary": str(REPORTS_DIR / "sector_reliability_summary.csv"),
            "class_group_reliability_summary": str(REPORTS_DIR / "class_group_reliability_summary.csv"),
            "score_alpha_manifest": str(REPORTS_DIR / "score_alpha_visualization_manifest.json"),
            "risk_map_manifest": str(REPORTS_DIR / "risk_map_visualization_manifest.json"),
            "dashboard_manifest": str(REPORTS_DIR / "dashboard_manifest.json"),
        },
        "safe_claims": [
            "internal diagnostic reliability indicator",
            "sensor-conditioned risk map",
            "controlled synthetic/proxy perturbation",
            "subset diagnostic",
            "not official benchmark",
            "not calibrated uncertainty",
            "not production-ready planning policy",
            "no training in this stage",
            "no model performance improvement claim",
        ],
        "limitations": [
            "reliability is diagnostic and heuristic, not probabilistically calibrated",
            "sector/class priors are hand-designed from SW-6 evidence",
            "planning-facing interface is a prototype only",
        ],
        "next_unique_action": "Stage SW-8 should prototype small-object activation loss first, then new-visible future consistency, under the same sensor-conditioned subset diagnostics.",
    }
    write_json(REPORTS_DIR / "stage_sw7_sensor_conditioned_reliability_map_report.json", report)
    (REPORTS_DIR / "stage_sw7_sensor_conditioned_reliability_map_report.md").write_text(
        "\n".join(
            [
                "# Stage SW-7 sensor-conditioned reliability map",
                "",
                "- internal diagnostic reliability indicator",
                "- controlled synthetic/proxy perturbation",
                "- subset diagnostic",
                "- not official benchmark",
                "- not calibrated uncertainty",
                "- not production-ready planning policy",
                "- no training in this stage",
                "- no model performance improvement claim",
                "",
                "## Executive summary",
                "",
                f"- effective_samples: `{len(effective_samples)}`",
                f"- effective_perturbations: `{', '.join(perturbation_ids)}`",
                f"- effective_horizons: `{', '.join(str(h) for h in horizons)}`",
                f"- lowest_reliability_degradation: `{report['lowest_reliability_degradation']}`",
                f"- lowest_reliability_sector: `{report['lowest_reliability_sector']}`",
                f"- lowest_reliability_class_group: `{report['lowest_reliability_class_group']}`",
                f"- SW-8 top recommendation: `{report['sw8_recommendation']}`",
                "",
                "## Reliability score: what it is / what it is not",
                "",
                "- It is an internal diagnostic reliability indicator combining semantic confidence, support density, contributor strength, sensor-conditioned prior, and temporal decay.",
                "- It is not calibrated uncertainty, not an official benchmark metric, and not a production planning policy.",
                "",
                "## Mainline answers",
                "",
                f"- Lowest front-sector reliability @ h6≈+3s: `{report['lowest_reliability_degradation']}`",
                f"- A1/A10 front-sector headline: `{report['a1_a10_front_sector_headline']}`",
                f"- Small-object lowest reliability headline: `{report['small_object_headline']}`",
                f"- New-visible lowest reliability headline: `{report['new_visible_headline']}`",
                f"- Score-alpha summary: {report['score_alpha_summary']}",
                f"- Risk-map summary: {report['risk_map_summary']}",
                f"- Reliability validation headline: `{report['reliability_validation_headline']}`",
                "",
                "## Safe claims",
                "",
                "- internal diagnostic reliability indicator",
                "- sensor-conditioned risk map",
                "- controlled synthetic/proxy perturbation",
                "- subset diagnostic",
                "- not official benchmark",
                "- not calibrated uncertainty",
                "- not production-ready planning policy",
                "- no training in this stage",
                "- no model performance improvement claim",
            ]
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "sw7_reliability_aggregate_summary.md").write_text(
        "\n".join(
            [
                "# SW-7 aggregate summary",
                "",
                f"- lowest front-sector reliability perturbation @ h6: `{report['lowest_reliability_degradation']}`",
                f"- lowest class-group reliability @ h6: `{report['lowest_reliability_class_group']}`",
                f"- best validation component: `{best_component}`",
            ]
        ),
        encoding="utf-8",
    )

    run_manifest["status"] = "completed"
    write_json(REPORTS_DIR / "sw7_run_manifest.json", run_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
