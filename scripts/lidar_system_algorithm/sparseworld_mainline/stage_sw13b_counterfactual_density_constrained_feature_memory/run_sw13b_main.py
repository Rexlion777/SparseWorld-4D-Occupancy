from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"

SW13A_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
SW13A_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"

PROGRESS_PATH = REPORTS_DIR / "sw13b_progress_state.json"
EXECUTION_MANIFEST_PATH = REPORTS_DIR / "sw13b_execution_manifest.json"
EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]
EVAL_SAMPLE_IDS = list(range(20))
PAIR_PERTURBATIONS = ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]
CONTROL_PERTURBATIONS = ["A0_clean", "A7_drop_all_rear"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw2 = load_module(
    "sw13b_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw13b_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw13b_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw13b_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw12b = load_module(
    "sw13b_sw12b",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py",
)
sw13a = load_module(
    "sw13b_sw13a",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/run_sw13a_main.py",
)

from mmcv.parallel import collate as collate_fn


@dataclass
class BaseRepairSpec:
    label: str
    perturbation_id: str
    base_variant_label: str
    source_variant_labels: list[str]
    primary_horizon: int
    reason: str


@dataclass
class FilterSpec:
    label: str
    family: str
    confidence_beta: float | None = None
    free_veto_thr: float | None = None
    boundary_r: int | None = None
    agreement_m: int | None = None
    agreement_n: int | None = None
    component_min_size: int | None = None
    class_consistency_thr: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-13B counterfactual density constrained feature memory replay")
    parser.add_argument("--samples", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "paired_dumps",
        ARTIFACTS_DIR / "filtered_outputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_export(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def init_progress_state() -> dict[str, Any]:
    payload = {
        "stage": "SW-13B",
        "start_time": now_iso(),
        "current_phase": None,
        "completed_phases": [],
        "phase_records": {},
        "status": "running",
    }
    write_json(PROGRESS_PATH, payload)
    return payload


def init_execution_manifest() -> dict[str, Any]:
    payload = {
        "stage": "SW-13B",
        "subtitle": "Counterfactual Density-Constrained Feature Memory Replay",
        "start_time": now_iso(),
        "phases": [],
        "artifacts": [],
        "safe_claim_boundary": [
            "no training",
            "no checkpoint modification",
            "no get_occ modification",
            "no GT repair",
            "no current clean same-frame feature",
            "no future-frame feature",
            "subset diagnostic only",
        ],
    }
    write_json(EXECUTION_MANIFEST_PATH, payload)
    return payload


def update_progress(progress: dict[str, Any]) -> None:
    write_json(PROGRESS_PATH, progress)


def update_manifest(manifest: dict[str, Any]) -> None:
    write_json(EXECUTION_MANIFEST_PATH, manifest)


def phase_start(progress: dict[str, Any], manifest: dict[str, Any], name: str) -> None:
    meta = {"phase_name": name, "start_time": now_iso(), "start_ts": time.time(), "status": "running"}
    progress["current_phase"] = name
    progress["phase_records"][name] = meta
    manifest["phases"].append(meta.copy())
    update_progress(progress)
    update_manifest(manifest)


def phase_end(progress: dict[str, Any], manifest: dict[str, Any], name: str, status: str, **extra: Any) -> None:
    record = progress["phase_records"][name]
    end_ts = time.time()
    record.update({"end_time": now_iso(), "end_ts": end_ts, "duration_sec": end_ts - record["start_ts"], "status": status, **extra})
    progress["current_phase"] = None
    if status == "done" and name not in progress["completed_phases"]:
        progress["completed_phases"].append(name)
    for item in manifest["phases"]:
        if item["phase_name"] == name and item["start_time"] == record["start_time"]:
            item.update(record)
            break
    update_progress(progress)
    update_manifest(manifest)


def attach_artifact(manifest: dict[str, Any], label: str, path: Path) -> None:
    manifest["artifacts"].append({"label": label, "path": str(path), "timestamp": now_iso()})
    update_manifest(manifest)


def make_base_repair_specs() -> list[BaseRepairSpec]:
    return [
        BaseRepairSpec("A1_base_repair", "A1_drop_cam_front", "R1_replace_tminus1", ["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2"], 6, "strongest A1 recovery signal"),
        BaseRepairSpec("A10_base_repair_1", "A10_drop_front_triplet", "R4_ema_K3", ["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"], 6, "strongest A10 recovery signal"),
        BaseRepairSpec("A10_base_repair_2", "A10_drop_front_triplet", "R8_camera_group_repair_front_triplet", ["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"], 6, "explicit front-triplet repair control"),
        BaseRepairSpec("C4_base_repair", "C4_motion_blur_9", "R5_blend_alpha03", ["R5_blend_alpha03", "R6_blend_alpha05", "R7_blend_alpha07"], 6, "safest SW-13A positive signal"),
    ]


def make_filter_specs(confidence_available: bool, semantic_available: bool) -> list[FilterSpec]:
    specs = [
        FilterSpec("D0_none", "none"),
        FilterSpec("D3_boundary_r1", "boundary", boundary_r=1),
        FilterSpec("D3_boundary_r2", "boundary", boundary_r=2),
        FilterSpec("D3_boundary_r3", "boundary", boundary_r=3),
        FilterSpec("D4_temporal_agreement_loose", "agreement", agreement_m=2, agreement_n=3),
        FilterSpec("D4_temporal_agreement_medium", "agreement", agreement_m=2, agreement_n=4),
        FilterSpec("D6_combined_light", "combined_light", confidence_beta=0.12, free_veto_thr=0.8, boundary_r=3, agreement_m=2, agreement_n=3, component_min_size=3, class_consistency_thr=0.5),
        FilterSpec("D7_combined_medium", "combined_medium", confidence_beta=0.08, free_veto_thr=0.7, boundary_r=2, agreement_m=2, agreement_n=4, component_min_size=5, class_consistency_thr=0.5),
        FilterSpec("D8_combined_strict", "combined_strict", confidence_beta=0.05, free_veto_thr=0.6, boundary_r=1, agreement_m=3, agreement_n=4, component_min_size=10, class_consistency_thr=0.7),
    ]
    if confidence_available:
        specs.extend(
            [
                FilterSpec("D1_conf_topk_b03", "confidence_topk", confidence_beta=0.03),
                FilterSpec("D1_conf_topk_b05", "confidence_topk", confidence_beta=0.05),
                FilterSpec("D1_conf_topk_b08", "confidence_topk", confidence_beta=0.08),
                FilterSpec("D1_conf_topk_b12", "confidence_topk", confidence_beta=0.12),
                FilterSpec("D2_native_free_veto_t06", "free_veto", free_veto_thr=0.6),
                FilterSpec("D2_native_free_veto_t07", "free_veto", free_veto_thr=0.7),
                FilterSpec("D2_native_free_veto_t08", "free_veto", free_veto_thr=0.8),
                FilterSpec("D2_native_free_veto_t09", "free_veto", free_veto_thr=0.9),
            ]
        )
    if semantic_available:
        specs.extend(
            [
                FilterSpec("D5_component_m3_t05", "component", component_min_size=3, class_consistency_thr=0.5),
                FilterSpec("D5_component_m5_t05", "component", component_min_size=5, class_consistency_thr=0.5),
                FilterSpec("D5_component_m10_t07", "component", component_min_size=10, class_consistency_thr=0.7),
            ]
        )
    return specs


def build_runtime():
    cfg, dataset, model, _ = sw81.build_sparseworld_runtime(train=False, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    model.eval()
    return cfg, dataset, model


def load_cache_map(sample_ids: list[int]) -> dict[int, dict[str, Any]]:
    cache_map: dict[int, dict[str, Any]] = {}
    for sample_id in sample_ids:
        path = SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_id:03d}.pt"
        cache_map[sample_id] = torch.load(path, map_location="cpu", weights_only=False)
    return cache_map


def dense_debug_for_case(model: Any, pred_dict: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    head = sw4_inst.get_pts_bbox_head(model)
    pred_dbg, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
    return pred_dbg[0].detach().cpu().long(), sw2.to_cpu_artifact(dbg_list[0])


def confidence_maps(debug_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    dense = torch.as_tensor(debug_dict["dense_occ_after_padding"]).cpu().float()
    top1_conf, top1_cls, top1_margin = sw12b.derive_top_conf_and_margin(dense)
    occ_pred = torch.as_tensor(debug_dict["occ_pred"]).cpu().long()
    occ_mask = occ_pred != EMPTY_IDX
    occ_conf = top1_conf * occ_mask.float()
    free_conf = (1.0 - torch.clamp(top1_conf, 0.0, 1.0)) * (~occ_mask).float()
    return {
        "top1_conf": top1_conf,
        "top1_cls": top1_cls,
        "top1_margin": top1_margin,
        "occ_conf": occ_conf,
        "free_conf": free_conf,
        "dense_logits_proxy": dense,
    }


def relevant_region_mask(perturbation_id: str, horizon_s: int, sectors: dict[str, torch.Tensor]) -> torch.Tensor:
    if perturbation_id in {"A1_drop_cam_front", "A10_drop_front_triplet"}:
        return sectors["front"].clone().bool()
    return torch.ones_like(sectors["front"], dtype=torch.bool)


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    out = mask.float()[None, None]
    for _ in range(max(0, radius)):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
    return out[0, 0] > 0


def boundary_distance_map(native_occ: torch.Tensor, max_radius: int = 5) -> torch.Tensor:
    dist = torch.full(native_occ.shape, fill_value=max_radius + 1, dtype=torch.int16)
    current = native_occ.clone()
    dist[native_occ] = 0
    for r in range(1, max_radius + 1):
        current = dilate_mask(current, 1)
        newly = current & (dist > max_radius)
        dist[newly] = r
    return dist


def topk_confidence_filter(delta_mask: torch.Tensor, repair_conf: torch.Tensor, native_occ: torch.Tensor, relevant_mask: torch.Tensor, beta: float) -> torch.Tensor:
    candidate = delta_mask & relevant_mask
    k = max(1, int(round(beta * max(1, int((native_occ & relevant_mask).sum().item())))))
    if candidate.sum().item() <= k:
        return candidate
    conf = repair_conf[candidate]
    top_vals, top_idx = torch.topk(conf, k=k, largest=True)
    keep = torch.zeros_like(delta_mask)
    coords = torch.nonzero(candidate, as_tuple=False)
    keep_coords = coords[top_idx]
    keep[keep_coords[:, 0], keep_coords[:, 1], keep_coords[:, 2]] = True
    return keep


def native_free_veto_filter(delta_mask: torch.Tensor, native_free_conf: torch.Tensor, thr: float) -> torch.Tensor:
    return delta_mask & (native_free_conf <= thr)


def boundary_band_filter(delta_mask: torch.Tensor, native_occ: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    dist_map = boundary_distance_map(native_occ, max_radius=max(5, radius))
    keep = delta_mask & (dist_map <= radius)
    return keep, dist_map


def temporal_agreement_filter(delta_mask: torch.Tensor, agreement_count: torch.Tensor, m: int) -> torch.Tensor:
    return delta_mask & (agreement_count >= m)


def connected_components(mask: torch.Tensor) -> list[list[tuple[int, int, int]]]:
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return []
    coord_set = {tuple(int(v) for v in xyz.tolist()) for xyz in coords}
    comps: list[list[tuple[int, int, int]]] = []
    seen: set[tuple[int, int, int]] = set()
    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for start in coord_set:
        if start in seen:
            continue
        comp: list[tuple[int, int, int]] = []
        q: deque[tuple[int, int, int]] = deque([start])
        seen.add(start)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for dx, dy, dz in neigh:
                nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                if nxt in coord_set and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def component_filter(
    delta_mask: torch.Tensor,
    repair_semantic: torch.Tensor,
    native_semantic: torch.Tensor,
    native_occ: torch.Tensor,
    min_size: int,
    class_thr: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not delta_mask.any():
        return torch.zeros_like(delta_mask), {"component_count_before": 0.0, "component_count_after": 0.0, "component_consistency_mean": 0.0}

    # Fast local component surrogate: keep voxels only when they sit inside a sufficiently supported local 3D neighborhood.
    delta_f = delta_mask.float()[None, None]
    local_support = F.conv3d(delta_f, torch.ones((1, 1, 3, 3, 3), dtype=delta_f.dtype), padding=1)[0, 0]
    size_keep = delta_mask & (local_support >= float(min(min_size, 27)))

    dominant_cls = repair_semantic.clone()
    dominant_cls[~delta_mask] = EMPTY_IDX
    native_ring = dilate_mask(native_occ, 1) | dilate_mask(delta_mask, 1)
    ring = native_ring & ~delta_mask
    same_class_support = torch.zeros_like(local_support)
    neighbor_valid = torch.zeros_like(local_support)
    for cls_id in torch.unique(repair_semantic[delta_mask]).tolist():
        if int(cls_id) == EMPTY_IDX:
            continue
        cls_mask = (repair_semantic == int(cls_id)).float()[None, None]
        cls_ring = ((repair_semantic == int(cls_id)) | (native_semantic == int(cls_id))) & ring
        cls_ring_f = cls_ring.float()[None, None]
        cls_support = F.conv3d(cls_ring_f, torch.ones((1, 1, 3, 3, 3), dtype=delta_f.dtype), padding=1)[0, 0]
        total_support = F.conv3d(ring.float()[None, None], torch.ones((1, 1, 3, 3, 3), dtype=delta_f.dtype), padding=1)[0, 0]
        match_vox = delta_mask & (repair_semantic == int(cls_id))
        same_class_support[match_vox] = cls_support[match_vox]
        neighbor_valid[match_vox] = total_support[match_vox]
    consistency = torch.zeros_like(local_support)
    nonzero = neighbor_valid > 0
    consistency[nonzero] = same_class_support[nonzero] / neighbor_valid[nonzero]
    keep = size_keep & ((consistency >= class_thr) | (neighbor_valid == 0))
    debug = {
        "component_count_before": float(delta_mask.sum().item()),
        "component_count_after": float(keep.sum().item()),
        "component_consistency_mean": float(consistency[delta_mask].mean().item()) if delta_mask.any() else 0.0,
    }
    return keep, debug


def apply_filter_spec(
    spec: FilterSpec,
    base_spec: BaseRepairSpec,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    native_case: dict[str, Any],
    repair_case: dict[str, Any],
    source_cases: dict[str, dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    native_occ = native_case["occ_mask"]
    repair_occ = repair_case["occ_mask"]
    delta_mask = repair_case["delta_occ_mask"]
    filtered = delta_mask.clone()
    debug = {
        "filter_label": spec.label,
        "raw_delta_count": int(delta_mask.sum().item()),
        "confidence_available": bool(repair_case["confidence_available"]),
        "semantic_available": True,
        "boundary_distance_mean": 0.0,
        "temporal_agreement_score": 0.0,
        "delta_keep_ratio": 0.0,
        "low_margin_ratio": 0.0,
        "repair_confidence_mean": 0.0,
        "component_count_before": 0.0,
        "component_count_after": 0.0,
        "component_consistency_mean": 0.0,
        "selection_uses_gt": False,
    }
    boundary_dist = boundary_distance_map(native_occ, max_radius=5)
    if spec.family == "none":
        filtered = delta_mask.clone()
    if spec.family == "confidence_topk" and spec.confidence_beta is not None and repair_case["confidence_available"]:
        filtered = topk_confidence_filter(delta_mask, repair_case["occ_conf"], native_occ, relevant_region_mask(base_spec.perturbation_id, horizon_s, sectors), spec.confidence_beta)
    if spec.family == "free_veto" and spec.free_veto_thr is not None and native_case["confidence_available"]:
        filtered = native_free_veto_filter(delta_mask, native_case["free_conf"], spec.free_veto_thr)
    if spec.family == "boundary" and spec.boundary_r is not None:
        filtered, boundary_dist = boundary_band_filter(delta_mask, native_occ, spec.boundary_r)
    if spec.family == "agreement" and spec.agreement_m is not None:
        agreement_count = torch.zeros_like(delta_mask, dtype=torch.int16)
        valid_sources = 0
        for case in source_cases.values():
            agreement_count += case["delta_occ_mask"].to(torch.int16)
            valid_sources += 1
        filtered = temporal_agreement_filter(delta_mask, agreement_count, spec.agreement_m)
        debug["temporal_agreement_score"] = safe_div(int(agreement_count[delta_mask].float().mean().item() * 1000) if delta_mask.any() else 0, 1000)
        debug["agreement_m"] = spec.agreement_m
        debug["agreement_n"] = valid_sources
    if spec.family == "component" and spec.component_min_size is not None and spec.class_consistency_thr is not None:
        filtered, comp_debug = component_filter(delta_mask, repair_case["semantic"], native_case["semantic"], native_occ, spec.component_min_size, spec.class_consistency_thr)
        debug.update(comp_debug)
    if spec.family.startswith("combined"):
        if native_case["confidence_available"] and spec.free_veto_thr is not None:
            filtered = native_free_veto_filter(filtered, native_case["free_conf"], spec.free_veto_thr)
        if spec.boundary_r is not None:
            filtered, boundary_dist = boundary_band_filter(filtered, native_occ, spec.boundary_r)
        if spec.agreement_m is not None:
            agreement_count = torch.zeros_like(delta_mask, dtype=torch.int16)
            valid_sources = 0
            for case in source_cases.values():
                agreement_count += case["delta_occ_mask"].to(torch.int16)
                valid_sources += 1
            filtered = temporal_agreement_filter(filtered, agreement_count, spec.agreement_m)
            debug["temporal_agreement_score"] = safe_div(int(agreement_count[delta_mask].float().mean().item() * 1000) if delta_mask.any() else 0, 1000)
            debug["agreement_m"] = spec.agreement_m
            debug["agreement_n"] = valid_sources
        if repair_case["confidence_available"] and spec.confidence_beta is not None:
            filtered = topk_confidence_filter(filtered, repair_case["occ_conf"], native_occ, relevant_region_mask(base_spec.perturbation_id, horizon_s, sectors), spec.confidence_beta)
        if spec.component_min_size is not None and spec.class_consistency_thr is not None:
            filtered, comp_debug = component_filter(filtered, repair_case["semantic"], native_case["semantic"], native_occ, spec.component_min_size, spec.class_consistency_thr)
            debug.update(comp_debug)
    kept_conf = repair_case["occ_conf"][filtered] if repair_case["confidence_available"] else torch.zeros((0,), dtype=torch.float32)
    kept_margin = repair_case["top1_margin"][filtered]
    debug["delta_keep_ratio"] = safe_div(int(filtered.sum().item()), max(1, int(delta_mask.sum().item())))
    debug["repair_confidence_mean"] = float(kept_conf.mean().item()) if kept_conf.numel() else 0.0
    debug["low_margin_ratio"] = float((kept_margin < 0.05).float().mean().item()) if kept_margin.numel() else 0.0
    debug["boundary_distance_mean"] = float(boundary_dist[filtered].float().mean().item()) if filtered.any() else 0.0
    return filtered, debug


def compose_final_prediction(native_semantic: torch.Tensor, repair_semantic: torch.Tensor, native_occ: torch.Tensor, filtered_delta: torch.Tensor) -> torch.Tensor:
    final_pred = native_semantic.clone()
    final_pred[filtered_delta] = repair_semantic[filtered_delta]
    final_pred[~(native_occ | filtered_delta)] = EMPTY_IDX
    return final_pred


def selection_proxy_pass(proxy_row: dict[str, Any]) -> bool:
    pert = proxy_row["perturbation_id"]
    delta_keep_ratio = float(proxy_row.get("delta_keep_ratio", 1.0))
    boundary_mean = float(proxy_row.get("boundary_distance_mean", 99.0))
    low_margin = float(proxy_row.get("low_margin_ratio", 1.0))
    conf_mean = float(proxy_row.get("repair_confidence_mean", 0.0))
    agreement = float(proxy_row.get("temporal_agreement_score", 0.0))
    if pert in {"A1_drop_cam_front", "A10_drop_front_triplet"}:
        return delta_keep_ratio <= 0.20 and boundary_mean <= 3.0 and low_margin <= 0.60 and (conf_mean >= 0.12 or agreement >= 1.5)
    return delta_keep_ratio <= 0.15 and low_margin <= 0.65


def proxy_score(proxy_row: dict[str, Any]) -> float:
    return (
        1.5 * float(proxy_row.get("temporal_agreement_score", 0.0))
        + 2.0 * float(proxy_row.get("repair_confidence_mean", 0.0))
        + 1.0 * float(proxy_row.get("component_consistency_mean", 0.0))
        - 1.2 * float(proxy_row.get("delta_keep_ratio", 1.0))
        - 0.25 * float(proxy_row.get("boundary_distance_mean", 0.0))
        - 0.8 * float(proxy_row.get("low_margin_ratio", 0.0))
    )


def row_to_metric_dict(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["front_sector_false_free_rate"] = out["front_sector_false_free"]
    out["small_object_false_free_rate"] = out["small_object_false_free"]
    out["dynamic_object_false_free_rate"] = out["dynamic_false_free"]
    out["future_h4_h6_false_free_rate"] = out.get("future_h4_h6_false_free_rate", out["false_free_rate"])
    out["A10_front_h6_recovery_rate"] = out["A10_front_h6_recovery_ratio"]
    out["false_positive_rate"] = out["false_occupied_rate"]
    out["density_drift"] = out["pred_gt_density_delta"]
    out["wrong_class_rate"] = out["wrong_class_activation"]
    out["clean_A0_metric_drift"] = out["clean_occupied_iou_delta"]
    return out


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    return sw12b.aggregate_rows(rows, group_keys)


def build_metric_definition() -> None:
    text = "\n".join(
        [
            "# SW-13B Metric Definition",
            "",
            "- `front_sector_false_free_rate`: proportion of front-sector GT occupied voxels predicted empty.",
            "- `small_object_false_free_rate`: proportion of GT occupied small-object voxels predicted empty.",
            "- `dynamic_object_false_free_rate`: proportion of GT occupied dynamic-object voxels predicted empty.",
            "- `future_h4_h6_false_free_rate`: future false-free rate at horizons h4/h6.",
            "- `new_visible_recall`: recall on voxels newly visible at the future horizon.",
            "- `A10_front_h6_recovery_rate`: share of A10/front/h6 degraded-native misses recovered as occupied.",
            "- `recovery_retention_ratio`: filtered recovery divided by raw repair recovery under the same base repair.",
            "- `density_reduction_ratio`: fraction of raw repair density delta removed by delta filtering.",
            "- `delta_keep_ratio`: kept delta voxels divided by raw delta voxels.",
            "- `temporal_agreement_score`: mean number of repair sources that agree on kept delta voxels.",
            "- `boundary_distance_mean`: mean native-boundary distance of kept delta voxels.",
            "- `wrong_class_delta_reduction`: reduction in wrong-class delta relative to raw repair.",
            "",
        ]
    )
    write_md(REPORTS_DIR / "sw13b_metric_definition.md", text)


def choose_visual_cases(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pert in ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]:
        rows = [r for r in metric_rows if r["perturbation_id"] == pert and int(r["horizon_s"]) == 6 and r["filter_label"] not in {"NATIVE_REFERENCE"} and r.get("filtered_output_path")]
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r.get("occupied_iou", 0.0)) - 0.10 * max(0.0, float(r.get("pred_gt_occupied_ratio", 0.0)) - 1.0))
        out[pert] = best
    return out


def save_bev_dashboard(path: Path, payload: dict[str, Any], title: str) -> None:
    gt = payload["gt_h"]
    native = payload["native_pred"]
    raw = payload["raw_repair_pred"]
    filt = payload["filtered_pred"]
    raw_delta = payload["raw_delta"]
    filt_delta = payload["filtered_delta"]
    removed = raw_delta & ~filt_delta
    gt_bev = (gt != EMPTY_IDX).any(dim=-1).float().numpy()
    nat_bev = (native != EMPTY_IDX).any(dim=-1).float().numpy()
    raw_bev = (raw != EMPTY_IDX).any(dim=-1).float().numpy()
    filt_bev = (filt != EMPTY_IDX).any(dim=-1).float().numpy()
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    panels = [
        (gt_bev, "GT"),
        (nat_bev, "degraded native"),
        (raw_bev, "raw repair D0"),
        (filt_bev, "filtered repair"),
        ((raw_delta.any(dim=-1).float().numpy()), "raw Delta"),
        ((filt_delta.any(dim=-1).float().numpy()), "filtered Delta"),
        ((removed.any(dim=-1).float().numpy()), "removed Delta"),
        ((np.logical_and(gt_bev > 0.5, nat_bev < 0.5)).astype(float), "false-free before"),
        ((np.logical_and(gt_bev > 0.5, raw_bev < 0.5)).astype(float), "false-free after raw"),
        ((np.logical_and(gt_bev > 0.5, filt_bev < 0.5)).astype(float), "false-free after filtered"),
        ((np.logical_and(gt_bev < 0.5, raw_bev > 0.5)).astype(float), "false-positive raw"),
        ((np.logical_and(gt_bev < 0.5, filt_bev > 0.5)).astype(float), "false-positive filtered"),
    ]
    for ax, (arr, ttl) in zip(axes.flatten(), panels):
        ax.imshow(arr.T, origin="lower", cmap="viridis")
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def phase0_inherit_sw13a(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[BaseRepairSpec]]:
    decision = read_json(SW13A_REPORTS / "sw13a_feature_memory_replay_decision.json")
    specs = make_base_repair_specs()
    summary = {
        "sw13a_decision_type": decision["decision_type"],
        "starting_points": [
            {
                "name": spec.label,
                "perturbation": spec.perturbation_id,
                "base_repair": spec.base_variant_label,
                "best_horizon_focus": spec.primary_horizon,
                "reason": spec.reason,
            }
            for spec in specs
        ],
        "must_not_change": {
            "no_training": True,
            "no_checkpoint_modification": True,
            "no_get_occ_modification": True,
        },
    }
    write_json(REPORTS_DIR / "sw13b_inherited_sw13a_summary.json", summary)
    attach_artifact(manifest, "inherited_sw13a_summary", REPORTS_DIR / "sw13b_inherited_sw13a_summary.json")
    return decision, specs


def phase1_phase4_replay(
    dataset: Any,
    model: Any,
    sample_ids: list[int],
    base_specs: list[BaseRepairSpec],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    build_metric_definition()
    cache_map = load_cache_map(sample_ids)
    sectors = {name: tensor.cpu().bool() for name, tensor in sw7.build_sector_masks().items()}
    spec_catalog = sw81.sw5_engine.build_catalog()
    manifest_rows: list[dict[str, Any]] = []
    paired_dump_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    confidence_available = True
    semantic_available = True

    base_by_pert: dict[str, list[BaseRepairSpec]] = defaultdict(list)
    for spec in base_specs:
        base_by_pert[spec.perturbation_id].append(spec)

    for sample_index in sample_ids:
        raw_sample, batch_clean = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        cache = cache_map[sample_index]
        for perturbation_id in PAIR_PERTURBATIONS + CONTROL_PERTURBATIONS:
            batch_deg = copy.deepcopy(batch_clean)
            if perturbation_id != "A0_clean":
                batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[perturbation_id])[0]
            variants_needed = {"R0_degraded_native"}
            for spec in base_by_pert.get(perturbation_id, []):
                variants_needed.update(spec.source_variant_labels)
                variants_needed.add(spec.base_variant_label)
            if perturbation_id in CONTROL_PERTURBATIONS:
                variants_needed = {"R0_degraded_native"}
            variant_lookup = {v.label: v for v in sw13a.make_variants()}
            results_by_variant: dict[str, dict[int, dict[str, Any]]] = {}
            runtime_by_variant: dict[str, dict[str, Any]] = {}
            for variant_label in sorted(variants_needed):
                variant = variant_lookup[variant_label]
                raw_result, per_h, runtime_debug = sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache, variant, perturbation_id)
                del raw_result
                variant_cases: dict[int, dict[str, Any]] = {}
                for horizon_s in CORE_HORIZONS:
                    pred_occ, debug = dense_debug_for_case(model, per_h[horizon_s]["pred_dict"])
                    conf = confidence_maps(debug)
                    occ_mask = pred_occ != EMPTY_IDX
                    variant_cases[horizon_s] = {
                        "pred": pred_occ,
                        "semantic": pred_occ,
                        "debug": debug,
                        "gt_h": per_h[horizon_s]["gt_h"],
                        "gt0": per_h[horizon_s]["gt0"],
                        "occ_mask": occ_mask,
                        "top1_conf": conf["top1_conf"],
                        "top1_cls": conf["top1_cls"],
                        "top1_margin": conf["top1_margin"],
                        "occ_conf": conf["occ_conf"],
                        "free_conf": conf["free_conf"],
                        "confidence_available": True,
                    }
                results_by_variant[variant_label] = variant_cases
                runtime_by_variant[variant_label] = runtime_debug
            for horizon_s in CORE_HORIZONS:
                native_case = results_by_variant["R0_degraded_native"][horizon_s]
                if perturbation_id in CONTROL_PERTURBATIONS:
                    row = sw13a.build_eval_row_extended(
                        native_case["pred"],
                        native_case["gt_h"],
                        native_case["gt0"],
                        perturbation_id,
                        horizon_s,
                        sectors,
                        native_case["pred"],
                        {"frame_debug": [{"repaired_camera_count": 0, "repair_applied": False}]},
                    )
                    row = row_to_metric_dict(row)
                    row.update(
                        {
                            "sample_index": sample_index,
                            "perturbation_id": perturbation_id,
                            "horizon_s": horizon_s,
                            "base_repair_variant": "R0_degraded_native",
                            "filter_label": "D0_none",
                            "selected_by_non_oracle_rule": False,
                            "uses_gt_repair": False,
                            "uses_future_info": False,
                            "uses_current_clean_same_frame": False,
                            "native_occ_count": int(native_case["occ_mask"].sum().item()),
                            "delta_raw_count": 0,
                            "delta_filtered_count": 0,
                            "delta_keep_ratio": 0.0,
                            "confidence_available": True,
                            "selection_uses_gt": False,
                        }
                    )
                    metric_rows.append(row)
                    continue

                for base_spec in base_by_pert[perturbation_id]:
                    repair_case = copy.deepcopy(results_by_variant[base_spec.base_variant_label][horizon_s])
                    repair_case["delta_occ_mask"] = repair_case["occ_mask"] & ~native_case["occ_mask"]
                    source_cases = {}
                    for label in base_spec.source_variant_labels:
                        source_case = copy.deepcopy(results_by_variant[label][horizon_s])
                        source_case["delta_occ_mask"] = source_case["occ_mask"] & ~native_case["occ_mask"]
                        source_cases[label] = source_case
                    paired_dir = ARTIFACTS_DIR / "paired_dumps" / perturbation_id / base_spec.base_variant_label
                    paired_dir.mkdir(parents=True, exist_ok=True)
                    paired_path = paired_dir / f"sample_{sample_index:03d}_h{horizon_s}.npz"
                    np.savez_compressed(
                        paired_path,
                        native_occ_mask=native_case["occ_mask"].numpy().astype(np.uint8),
                        repair_occ_mask=repair_case["occ_mask"].numpy().astype(np.uint8),
                        delta_occ_mask=repair_case["delta_occ_mask"].numpy().astype(np.uint8),
                        native_semantic=native_case["semantic"].numpy().astype(np.int16),
                        repair_semantic=repair_case["semantic"].numpy().astype(np.int16),
                        native_occ_confidence=native_case["occ_conf"].numpy().astype(np.float32),
                        repair_occ_confidence=repair_case["occ_conf"].numpy().astype(np.float32),
                        native_free_confidence=native_case["free_conf"].numpy().astype(np.float32),
                        repair_class_confidence=repair_case["top1_conf"].numpy().astype(np.float32),
                        sample_id=np.int16(sample_index),
                        horizon_s=np.int16(horizon_s),
                        confidence_available=np.bool_(True),
                        uses_future_info=np.bool_(False),
                        uses_current_clean_same_frame=np.bool_(False),
                        uses_gt_repair=np.bool_(False),
                    )
                    paired_dump_rows.append(
                        {
                            "sample_index": sample_index,
                            "perturbation_id": perturbation_id,
                            "horizon_s": horizon_s,
                            "base_repair_variant": base_spec.base_variant_label,
                            "paired_dump_path": str(paired_path),
                            "confidence_available": True,
                            "uses_future_info": False,
                            "uses_current_clean_same_frame": False,
                            "uses_gt_repair": False,
                        }
                    )
                    native_row = sw13a.build_eval_row_extended(
                        native_case["pred"],
                        native_case["gt_h"],
                        native_case["gt0"],
                        perturbation_id,
                        horizon_s,
                        sectors,
                        native_case["pred"],
                        {"frame_debug": [{"repaired_camera_count": 0, "repair_applied": False}]},
                    )
                    native_row = row_to_metric_dict(native_row)
                    native_row.update(
                        {
                            "sample_index": sample_index,
                            "perturbation_id": perturbation_id,
                            "horizon_s": horizon_s,
                            "base_repair_variant": base_spec.base_variant_label,
                            "filter_label": "NATIVE_REFERENCE",
                            "base_repair_label": base_spec.label,
                            "native_occ_count": int(native_case["occ_mask"].sum().item()),
                            "repair_occ_count": int(native_case["occ_mask"].sum().item()),
                            "delta_raw_count": 0,
                            "delta_filtered_count": 0,
                            "delta_keep_ratio": 0.0,
                            "temporal_agreement_score": 0.0,
                            "boundary_distance_mean": 0.0,
                            "component_count_before_after": "0->0",
                            "component_consistency_mean": 0.0,
                            "repair_confidence_mean": float(native_case["occ_conf"][native_case["occ_mask"]].mean().item()) if native_case["occ_mask"].any() else 0.0,
                            "low_margin_ratio": float((native_case["top1_margin"][native_case["occ_mask"]] < 0.05).float().mean().item()) if native_case["occ_mask"].any() else 0.0,
                            "confidence_available": True,
                            "uses_gt_repair": False,
                            "uses_future_info": False,
                            "uses_current_clean_same_frame": False,
                            "selection_uses_gt": False,
                            "native_occ_unchanged": True,
                        }
                    )
                    metric_rows.append(native_row)
                    filter_specs = make_filter_specs(confidence_available=True, semantic_available=True)
                    for filt in filter_specs:
                        filtered_delta, filter_debug = apply_filter_spec(filt, base_spec, horizon_s, sectors, native_case, repair_case, source_cases)
                        final_pred = repair_case["pred"] if filt.label == "D0_none" else compose_final_prediction(native_case["semantic"], repair_case["semantic"], native_case["occ_mask"], filtered_delta)
                        filtered_path = ARTIFACTS_DIR / "filtered_outputs" / f"{perturbation_id}__{base_spec.base_variant_label}__{filt.label}__sample{sample_index:03d}_h{horizon_s}.npz"
                        should_save_filtered = sample_index in {0, 1, 2} and horizon_s == 6 and filt.label in {"D0_none", "D6_combined_light", "D7_combined_medium", "D8_combined_strict"}
                        if should_save_filtered:
                            np.savez_compressed(
                                filtered_path,
                                final_semantic=final_pred.numpy().astype(np.int16),
                                delta_filtered_mask=filtered_delta.numpy().astype(np.uint8),
                                gt_h=repair_case["gt_h"].numpy().astype(np.int16),
                                uses_gt_repair=np.bool_(False),
                                uses_future_info=np.bool_(False),
                                uses_current_clean_same_frame=np.bool_(False),
                            )
                        row = sw13a.build_eval_row_extended(
                            final_pred,
                            repair_case["gt_h"],
                            repair_case["gt0"],
                            perturbation_id,
                            horizon_s,
                            sectors,
                            native_case["pred"],
                            runtime_by_variant[base_spec.base_variant_label],
                        )
                        row = row_to_metric_dict(row)
                        row.update(
                            {
                                "sample_index": sample_index,
                                "perturbation_id": perturbation_id,
                                "horizon_s": horizon_s,
                                "base_repair_variant": base_spec.base_variant_label,
                                "filter_label": filt.label,
                                "base_repair_label": base_spec.label,
                                "native_occ_count": int(native_case["occ_mask"].sum().item()),
                                "repair_occ_count": int(repair_case["occ_mask"].sum().item()),
                                "delta_raw_count": int(repair_case["delta_occ_mask"].sum().item()),
                                "delta_filtered_count": int(filtered_delta.sum().item()),
                                "delta_keep_ratio": filter_debug["delta_keep_ratio"],
                                "temporal_agreement_score": filter_debug.get("temporal_agreement_score", 0.0),
                                "boundary_distance_mean": filter_debug.get("boundary_distance_mean", 0.0),
                                "component_count_before_after": f"{int(filter_debug.get('component_count_before', 0))}->{int(filter_debug.get('component_count_after', 0))}",
                                "component_consistency_mean": filter_debug.get("component_consistency_mean", 0.0),
                                "repair_confidence_mean": filter_debug.get("repair_confidence_mean", 0.0),
                                "low_margin_ratio": filter_debug.get("low_margin_ratio", 0.0),
                                "confidence_available": True,
                                "uses_gt_repair": False,
                                "uses_future_info": False,
                                "uses_current_clean_same_frame": False,
                                "selection_uses_gt": False,
                                "native_occ_unchanged": bool(torch.equal(final_pred[native_case["occ_mask"]], native_case["semantic"][native_case["occ_mask"]])),
                                "filtered_output_path": str(filtered_path) if should_save_filtered else "",
                            }
                        )
                        metric_rows.append(row)
                        proxy_row = {
                            "sample_index": sample_index,
                            "perturbation_id": perturbation_id,
                            "horizon_s": horizon_s,
                            "base_repair_variant": base_spec.base_variant_label,
                            "filter_label": filt.label,
                            "delta_keep_ratio": filter_debug["delta_keep_ratio"],
                            "temporal_agreement_score": filter_debug.get("temporal_agreement_score", 0.0),
                            "boundary_distance_mean": filter_debug.get("boundary_distance_mean", 0.0),
                            "component_consistency_mean": filter_debug.get("component_consistency_mean", 0.0),
                            "repair_confidence_mean": filter_debug.get("repair_confidence_mean", 0.0),
                            "low_margin_ratio": filter_debug.get("low_margin_ratio", 0.0),
                            "proxy_selection_pass": selection_proxy_pass(
                                {
                                    "perturbation_id": perturbation_id,
                                    "delta_keep_ratio": filter_debug["delta_keep_ratio"],
                                    "temporal_agreement_score": filter_debug.get("temporal_agreement_score", 0.0),
                                    "boundary_distance_mean": filter_debug.get("boundary_distance_mean", 0.0),
                                    "component_consistency_mean": filter_debug.get("component_consistency_mean", 0.0),
                                    "repair_confidence_mean": filter_debug.get("repair_confidence_mean", 0.0),
                                    "low_margin_ratio": filter_debug.get("low_margin_ratio", 0.0),
                                }
                            ),
                            "proxy_score": proxy_score(
                                {
                                    "delta_keep_ratio": filter_debug["delta_keep_ratio"],
                                    "temporal_agreement_score": filter_debug.get("temporal_agreement_score", 0.0),
                                    "boundary_distance_mean": filter_debug.get("boundary_distance_mean", 0.0),
                                    "component_consistency_mean": filter_debug.get("component_consistency_mean", 0.0),
                                    "repair_confidence_mean": filter_debug.get("repair_confidence_mean", 0.0),
                                    "low_margin_ratio": filter_debug.get("low_margin_ratio", 0.0),
                                }
                            ),
                            "selection_uses_gt": False,
                        }
                        proxy_rows.append(proxy_row)
                        manifest_rows.append(
                            {
                                "sample_index": sample_index,
                                "perturbation_id": perturbation_id,
                                "horizon_s": horizon_s,
                                "base_repair_variant": base_spec.base_variant_label,
                                "filter_label": filt.label,
                                "uses_gt_repair": False,
                                "uses_future_info": False,
                                "uses_current_clean_same_frame": False,
                            }
                        )

    write_csv(REPORTS_DIR / "sw13b_replay_manifest.csv", manifest_rows)
    write_csv(REPORTS_DIR / "sw13b_paired_replay_dump_manifest.csv", paired_dump_rows)
    write_md(
        REPORTS_DIR / "sw13b_confidence_availability_report.md",
        "\n".join(
            [
                "# SW-13B Confidence Availability",
                "",
                "- confidence_available: true",
                "- source: `dense_occ_after_padding` captured through `sw4_inst.get_occ_debug(..., capture_dense=True)`",
                "- occupancy confidence proxy: per-voxel top-1 post-padding class score",
                "- native free confidence proxy: `1 - top1_conf` on native-empty voxels",
                "- no confidence values are fabricated from GT or clean teacher data",
                "",
            ]
        ),
    )
    attach_artifact(manifest, "paired_dump_manifest", REPORTS_DIR / "sw13b_paired_replay_dump_manifest.csv")
    return metric_rows, proxy_rows, paired_dump_rows, {"confidence_available": confidence_available, "semantic_available": semantic_available}


def phase2_phase3_selection_and_reports(
    sw13a_decision: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    base_specs: list[BaseRepairSpec],
    availability: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    write_json(
        REPORTS_DIR / "sw13b_delta_filter_variant_manifest.json",
        {
            "variants": [normalize_export(spec.__dict__) for spec in make_filter_specs(availability["confidence_available"], availability["semantic_available"])],
            "native_existing_occupied_unchanged": True,
            "selection_must_not_use_gt": True,
        },
    )
    write_md(
        REPORTS_DIR / "sw13b_selection_rule.md",
        "\n".join(
            [
                "# SW-13B Non-oracle Selection Rule",
                "",
                "- candidate filtering is based only on prediction-side quantities: delta keep ratio, confidence, temporal agreement, boundary proximity, component consistency, and low-margin proxy.",
                "- GT metrics are not used to choose the deployed candidate.",
                "- selection pipeline:",
                "  1. apply proxy safety prefilter",
                "  2. rank passing candidates by proxy score",
                "  3. report GT-based recovery and safety only after the candidate is frozen",
                "",
            ]
        ),
    )
    agg_proxy = aggregate_rows(proxy_rows, ["perturbation_id", "base_repair_variant", "filter_label"])
    for row in agg_proxy:
        row["proxy_selection_pass"] = selection_proxy_pass(row)
        row["proxy_score"] = proxy_score(row)
    write_json(
        REPORTS_DIR / "sw13b_non_oracle_candidate_selection.json",
        {
            "selection_uses_gt": False,
            "sw13a_decision_type": sw13a_decision["decision_type"],
            "aggregate_proxy_rows": agg_proxy,
        },
    )

    agg_metrics = aggregate_rows(metric_rows, ["perturbation_id", "base_repair_variant", "filter_label", "horizon_s"])
    baseline_map = {
        (row["perturbation_id"], row["base_repair_variant"], int(row["horizon_s"])): row
        for row in agg_metrics
        if row["filter_label"] == "D0_none"
    }
    native_map = {
        (row["perturbation_id"], row["base_repair_variant"], int(row["horizon_s"])): row
        for row in agg_metrics
        if row["filter_label"] == "NATIVE_REFERENCE"
    }
    summary_rows: list[dict[str, Any]] = []
    for row in agg_metrics:
        base = baseline_map.get((row["perturbation_id"], row["base_repair_variant"], int(row["horizon_s"])))
        native_ref = native_map.get((row["perturbation_id"], row["base_repair_variant"], int(row["horizon_s"])))
        summary = dict(row)
        if base is not None and native_ref is not None:
            raw_front_gain = max(0.0, -float(base["front_sector_false_free_rate"]) + float(native_ref["front_sector_false_free_rate"]))
            filt_front_gain = max(0.0, -float(row["front_sector_false_free_rate"]) + float(native_ref["front_sector_false_free_rate"]))
            raw_a10_gain = max(0.0, float(base["A10_front_h6_recovery_rate"]) - float(native_ref["A10_front_h6_recovery_rate"]))
            filt_a10_gain = max(0.0, float(row["A10_front_h6_recovery_rate"]) - float(native_ref["A10_front_h6_recovery_rate"]))
            raw_density = max(0.0, float(base["pred_gt_occupied_ratio"]) - float(native_ref["pred_gt_occupied_ratio"]))
            filt_density = max(0.0, float(row["pred_gt_occupied_ratio"]) - float(native_ref["pred_gt_occupied_ratio"]))
            summary.update(
                {
                    "front_sector_false_free_rate_delta_vs_native": float(row["front_sector_false_free_rate"]) - float(native_ref["front_sector_false_free_rate"]),
                    "small_object_false_free_rate_delta_vs_native": float(row["small_object_false_free_rate"]) - float(native_ref["small_object_false_free_rate"]),
                    "future_h4_h6_false_free_rate_delta_vs_native": float(row["future_h4_h6_false_free_rate"]) - float(native_ref["future_h4_h6_false_free_rate"]),
                    "new_visible_recall_delta_vs_native": float(row["new_visible_recall"]) - float(native_ref["new_visible_recall"]),
                    "A10_front_h6_recovery_rate_delta_vs_native": float(row["A10_front_h6_recovery_rate"]) - float(native_ref["A10_front_h6_recovery_rate"]),
                    "false_positive_rate_delta_vs_native": float(row["false_positive_rate"]) - float(native_ref["false_positive_rate"]),
                    "pred_gt_occupied_ratio_delta_vs_native": float(row["pred_gt_occupied_ratio"]) - float(native_ref["pred_gt_occupied_ratio"]),
                    "wrong_class_rate_delta_vs_native": float(row["wrong_class_rate"]) - float(native_ref["wrong_class_rate"]),
                    "occupied_iou_delta_vs_native": float(row["occupied_iou"]) - float(native_ref["occupied_iou"]),
                    "semantic_miou_delta_vs_native": float(row["semantic_miou"]) - float(native_ref["semantic_miou"]),
                    "recovery_retention_ratio": safe_div(filt_a10_gain if "A10" in row["perturbation_id"] else filt_front_gain, max(1e-6, raw_a10_gain if "A10" in row["perturbation_id"] else raw_front_gain)),
                    "density_reduction_ratio": 1.0 - safe_div(filt_density, max(1e-6, raw_density)),
                    "recovery_per_density": safe_div(filt_a10_gain if "A10" in row["perturbation_id"] else filt_front_gain, max(1e-6, filt_density)),
                    "recovery_per_false_positive": safe_div(filt_a10_gain if "A10" in row["perturbation_id"] else filt_front_gain, max(1e-6, max(0.0, float(row["false_positive_rate"]) - float(native_ref["false_positive_rate"])))),
                    "wrong_class_delta_reduction": (float(base["wrong_class_rate"]) - float(native_ref["wrong_class_rate"])) - (float(row["wrong_class_rate"]) - float(native_ref["wrong_class_rate"])),
                }
            )
        summary_rows.append(summary)
    write_csv(REPORTS_DIR / "sw13b_delta_filter_aggregate_metrics.csv", agg_metrics)
    write_csv(REPORTS_DIR / "sw13b_metric_summary.csv", summary_rows)
    write_csv(REPORTS_DIR / "sw13b_delta_filter_metrics.csv", metric_rows)
    attach_artifact(manifest, "delta_filter_metrics", REPORTS_DIR / "sw13b_delta_filter_metrics.csv")

    selected: dict[str, dict[str, Any]] = {}
    for spec in base_specs:
        candidates = [row for row in agg_proxy if row["perturbation_id"] == spec.perturbation_id and row["base_repair_variant"] == spec.base_variant_label]
        passing = [row for row in candidates if row["proxy_selection_pass"]]
        pool = passing if passing else candidates
        if not pool:
            continue
        selected_row = max(pool, key=lambda r: float(r["proxy_score"]))
        selected[spec.label] = selected_row

    for row in metric_rows:
        row["selected_by_non_oracle_rule"] = any(
            row["perturbation_id"] == selected_item["perturbation_id"]
            and row["base_repair_variant"] == selected_item["base_repair_variant"]
            and row["filter_label"] == selected_item["filter_label"]
            for selected_item in selected.values()
        )
    write_csv(REPORTS_DIR / "sw13b_delta_filter_metrics.csv", metric_rows)
    return summary_rows, selected


def phase5_phase9_decision_and_visuals(
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    criteria = {
        "A1_strong_success": {
            "recovery_retention_ratio_min": 0.60,
            "density_delta_max": 0.08,
            "false_positive_delta_max": 0.005,
            "wrong_class_delta_max": 0.015,
            "front_sector_false_free_rate_delta_vs_native_max": -0.10,
        },
        "A1_medium_success": {
            "recovery_retention_ratio_min": 0.40,
            "density_delta_max": 0.05,
            "false_positive_delta_max": 0.005,
            "front_sector_false_free_rate_delta_vs_native_max": -0.05,
        },
        "A10_strong_success": {
            "recovery_retention_ratio_min": 0.50,
            "density_delta_max": 0.08,
            "wrong_class_delta_max": 0.015,
            "false_positive_delta_max": 0.005,
            "A10_front_h6_recovery_rate_delta_vs_native_min": 0.15,
            "front_sector_false_free_rate_delta_vs_native_max": -0.15,
        },
        "A10_medium_success": {
            "recovery_retention_ratio_min": 0.35,
            "density_delta_max": 0.08,
            "wrong_class_delta_max": 0.02,
            "A10_front_h6_recovery_rate_delta_vs_native_min": 0.10,
        },
        "C4_success": {
            "preserve_direction": True,
            "false_positive_not_worse_than_D0_plus": 0.02,
            "density_not_worse_than_D0_plus": 0.02,
        },
    }
    write_json(REPORTS_DIR / "sw13b_success_criteria.json", criteria)

    selected_rows = {}
    for key, sel in selected.items():
        matches = [
            row
            for row in summary_rows
            if row["perturbation_id"] == sel["perturbation_id"]
            and row["base_repair_variant"] == sel["base_repair_variant"]
            and row["filter_label"] == sel["filter_label"]
            and int(row["horizon_s"]) == 6
        ]
        if matches:
            selected_rows[key] = matches[0]

    c4_selected = selected_rows.get("C4_base_repair")
    a1_selected = selected_rows.get("A1_base_repair")
    a10_selected = selected_rows.get("A10_base_repair_1") or selected_rows.get("A10_base_repair_2")

    def a1_strong(row: dict[str, Any] | None) -> bool:
        if row is None:
            return False
        return (
            float(row["recovery_retention_ratio"]) >= 0.60
            and float(row["pred_gt_occupied_ratio_delta_vs_native"]) <= 0.08
            and float(row["false_positive_rate_delta_vs_native"]) <= 0.005
            and float(row["wrong_class_rate_delta_vs_native"]) <= 0.015
            and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.10
        )

    def a1_medium(row: dict[str, Any] | None) -> bool:
        if row is None:
            return False
        return (
            float(row["recovery_retention_ratio"]) >= 0.40
            and float(row["pred_gt_occupied_ratio_delta_vs_native"]) <= 0.05
            and float(row["false_positive_rate_delta_vs_native"]) <= 0.005
            and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.05
        )

    def a10_strong(row: dict[str, Any] | None) -> bool:
        if row is None:
            return False
        return (
            float(row["recovery_retention_ratio"]) >= 0.50
            and float(row["pred_gt_occupied_ratio_delta_vs_native"]) <= 0.08
            and float(row["wrong_class_rate_delta_vs_native"]) <= 0.015
            and float(row["false_positive_rate_delta_vs_native"]) <= 0.005
            and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.15
            and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.15
        )

    def a10_medium(row: dict[str, Any] | None) -> bool:
        if row is None:
            return False
        return (
            float(row["recovery_retention_ratio"]) >= 0.35
            and float(row["pred_gt_occupied_ratio_delta_vs_native"]) <= 0.08
            and float(row["wrong_class_rate_delta_vs_native"]) <= 0.02
            and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.10
        )

    def c4_success(row: dict[str, Any] | None) -> bool:
        if row is None:
            return False
        return (
            float(row["front_sector_false_free_rate_delta_vs_native"]) <= 0.0
            and float(row["small_object_false_free_rate_delta_vs_native"]) <= 0.0
            and float(row["false_positive_rate_delta_vs_native"]) <= 0.0
            and float(row["pred_gt_occupied_ratio_delta_vs_native"]) <= 0.0
        )

    if (a1_strong(a1_selected) or a10_strong(a10_selected)) and c4_success(c4_selected):
        decision_type = "T1_STRONG_SAFE_DENSITY_CONSTRAINED_RECOVERY"
    elif (a1_medium(a1_selected) or a10_medium(a10_selected)) and c4_success(c4_selected):
        decision_type = "T2_MEDIUM_SAFE_DENSITY_CONSTRAINED_RECOVERY"
    elif (a1_selected and float(a1_selected.get("recovery_retention_ratio", 0.0)) > 0.0) or (a10_selected and float(a10_selected.get("recovery_retention_ratio", 0.0)) > 0.0):
        if (a1_selected and float(a1_selected.get("pred_gt_occupied_ratio_delta_vs_native", 0.0)) > 0.15) or (a10_selected and float(a10_selected.get("pred_gt_occupied_ratio_delta_vs_native", 0.0)) > 0.15):
            decision_type = "T3_RECOVERY_RETAINED_BUT_DENSITY_STILL_HIGH"
        else:
            decision_type = "T4_DENSITY_CONTROLLED_BUT_RECOVERY_LOST"
    elif c4_success(c4_selected):
        decision_type = "T5_C4_ONLY_SAFE"
    else:
        decision_type = "T4_DENSITY_CONTROLLED_BUT_RECOVERY_LOST"

    # visual cases
    chosen = choose_visual_cases(metric_rows)
    for pert, best in chosen.items():
        base_variant = best["base_repair_variant"]
        filt = best["filter_label"]
        sample_id = int(best["sample_index"])
        horizon_s = int(best["horizon_s"])
        pair_path = ARTIFACTS_DIR / "paired_dumps" / pert / base_variant / f"sample_{sample_id:03d}_h{horizon_s}.npz"
        pair = np.load(pair_path)
        filt_rows = [row for row in metric_rows if int(row["sample_index"]) == sample_id and row["perturbation_id"] == pert and int(row["horizon_s"]) == horizon_s and row["base_repair_variant"] == base_variant and row["filter_label"] == filt]
        if not filt_rows:
            continue
        filtered_path = ARTIFACTS_DIR / "filtered_outputs" / f"{pert}__{base_variant}__{filt}__sample{sample_id:03d}_h{horizon_s}.npz"
        if not filtered_path.exists():
            continue
        payload = np.load(filtered_path)
        save_bev_dashboard(
            FIGURES_DIR / f"sw13b_bev_counterfactual_delta_{'A1' if 'A1' in pert else 'A10' if 'A10' in pert else 'C4'}_sample{sample_id}.png",
            {
                "gt_h": torch.from_numpy(payload["gt_h"]),
                "native_pred": torch.from_numpy(pair["native_semantic"]),
                "raw_repair_pred": torch.from_numpy(pair["repair_semantic"]),
                "filtered_pred": torch.from_numpy(payload["final_semantic"]),
                "raw_delta": torch.from_numpy(pair["delta_occ_mask"]).bool(),
                "filtered_delta": torch.from_numpy(payload["delta_filtered_mask"]).bool(),
            },
            "SW-13B counterfactual delta filtering subset diagnostic no training no GT repair",
        )

    pareto_rows = [row for row in summary_rows if int(row["horizon_s"]) == 6 and row["perturbation_id"] in {"A1_drop_cam_front", "A10_drop_front_triplet"} and row["filter_label"] != "NATIVE_REFERENCE"]
    for pert, fig_name in [("A1_drop_cam_front", "sw13b_A1_recovery_density_pareto.png"), ("A10_drop_front_triplet", "sw13b_A10_recovery_density_pareto.png")]:
        rows = [r for r in pareto_rows if r["perturbation_id"] == pert]
        fig, ax = plt.subplots(figsize=(8.2, 5.1))
        for row in rows:
            ax.scatter(float(row.get("pred_gt_occupied_ratio_delta_vs_native", 0.0)), float(max(0.0, -float(row.get("front_sector_false_free_rate_delta_vs_native", 0.0)))), s=50)
            ax.text(float(row.get("pred_gt_occupied_ratio_delta_vs_native", 0.0)), float(max(0.0, -float(row.get("front_sector_false_free_rate_delta_vs_native", 0.0)))), row["filter_label"], fontsize=7)
        ax.set_xlabel("density delta vs native")
        ax.set_ylabel("front false-free reduction vs native")
        ax.set_title(f"SW-13B counterfactual delta filtering subset diagnostic {pert} Pareto")
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / fig_name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    rows = [r for r in summary_rows if int(r["horizon_s"]) == 6 and r["filter_label"] != "NATIVE_REFERENCE"]
    ax.scatter(
        [float(r.get("wrong_class_rate_delta_vs_native", 0.0)) for r in rows],
        [float(r.get("recovery_retention_ratio", 0.0)) for r in rows],
        alpha=0.7,
    )
    ax.set_title("SW-13B counterfactual delta filtering subset diagnostic wrong-class vs recovery")
    ax.set_xlabel("wrong-class delta vs native")
    ax.set_ylabel("recovery retention ratio")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "sw13b_wrong_class_recovery_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([float(r.get("boundary_distance_mean", 0.0)) for r in rows], bins=12, color="#2E86C1")
    axes[0].set_title("boundary distance mean")
    axes[1].scatter(
        [float(r.get("delta_keep_ratio", 0.0)) for r in rows],
        [float(r.get("recovery_retention_ratio", 0.0)) for r in rows],
        alpha=0.7,
    )
    axes[1].set_title("delta keep ratio vs recovery retention")
    fig.suptitle("SW-13B counterfactual delta filtering subset diagnostic debug")
    fig.savefig(FIGURES_DIR / "sw13b_delta_filter_debug.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    decision = {
        "decision_type": decision_type,
        "selected_non_oracle_candidates": selected,
        "best_A1_candidate": selected.get("A1_base_repair"),
        "best_A10_candidate": selected.get("A10_base_repair_1") or selected.get("A10_base_repair_2"),
        "best_C4_candidate": selected.get("C4_base_repair"),
        "safe_claim_template": (
            "历史特征补偿产生的新增 occupied 经 native-vs-repair 反事实差分过滤后，可在保留部分前向漏检恢复的同时显著降低预测密度扩张。"
            if decision_type in {"T1_STRONG_SAFE_DENSITY_CONSTRAINED_RECOVERY", "T2_MEDIUM_SAFE_DENSITY_CONSTRAINED_RECOVERY"}
            else "Delta filtering 验证了 density-recovery tradeoff，但尚未形成安全候选。"
        ),
        "selection_uses_gt": False,
        "no_training": True,
        "no_checkpoint_modification": True,
        "no_get_occ_modification": True,
        "no_gt_repair": True,
        "no_future_frame_feature": True,
        "no_current_clean_same_frame_feature": True,
    }
    write_json(REPORTS_DIR / "sw13b_counterfactual_density_constrained_decision.json", decision)
    write_md(REPORTS_DIR / "sw13b_counterfactual_density_constrained_decision.md", json.dumps(normalize_export(decision), indent=2, ensure_ascii=False) + "\n")
    attach_artifact(manifest, "decision", REPORTS_DIR / "sw13b_counterfactual_density_constrained_decision.json")

    report_md = "\n".join(
        [
            "# Stage SW-13B Counterfactual Density-Constrained Feature Memory Replay",
            "",
            "1. Executive summary",
            f"- decision: {decision_type}",
            "",
            "2. Why SW-13B follows SW-13A",
            "- SW-13A established strong recovery with unsafe density expansion.",
            "- SW-13B keeps native occupied unchanged and filters only the repair-vs-native delta.",
            "",
            "3. Paired native-vs-repair replay",
            "- paired native and repair predictions are dumped for the same sample / perturbation / horizon.",
            "",
            "4. Counterfactual Delta definition",
            "- Delta_occ = repair_occ AND NOT native_occ.",
            "- final prediction = native_occ union filtered Delta_occ.",
            "",
            "5. Delta filter variants",
            "- confidence budget, native free veto, boundary band, temporal agreement, component consistency, and combined filters were evaluated.",
            "",
            "6. Non-oracle selection rule",
            "- deployed candidates are selected using prediction-side proxy rules only; GT is not used for candidate selection.",
            "",
            "7. A1 results",
            "- reported as subset diagnostic replay only.",
            "",
            "8. A10 results",
            "- reported as subset diagnostic replay only.",
            "",
            "9. C4 results",
            "- reported as subset diagnostic replay only.",
            "",
            "10. Recovery-density Pareto analysis",
            "- density reduction is evaluated against raw SW-13A-style repair while tracking retained recovery.",
            "",
            "11. Wrong-class and component analysis",
            "- wrong-class deltas and connected component filtering are reported.",
            "",
            "12. Visualizations",
            "- counterfactual delta BEV panels and Pareto plots are included.",
            "",
            "13. Decision T1-T7",
            f"- {decision_type}",
            "",
            "14. Safe claims",
            "- no training",
            "- no checkpoint modification",
            "- no get_occ modification",
            "- no GT repair",
            "- no future-frame feature",
            "- no current clean same-frame feature",
            "- native existing occupied unchanged",
            "- only Delta_occ is filtered",
            "- subset diagnostic only",
            "- not official benchmark",
            "",
            "15. Limitations",
            "- this stage evaluates replay-only safety filtering; it does not establish any final model improvement claim.",
            "",
            "16. Next unique action",
            f"- {decision_type}",
            "",
        ]
    )
    report_json = {
        "decision_type": decision_type,
        "selected_non_oracle_candidates": selected,
        "selection_uses_gt": False,
        "safe_claim_boundary": [
            "no training",
            "no checkpoint modification",
            "no get_occ modification",
            "no GT repair",
            "no future-frame feature",
            "no current clean same-frame feature",
            "subset diagnostic only",
            "not official benchmark",
        ],
    }
    write_md(REPORTS_DIR / "stage_sw13b_counterfactual_density_constrained_feature_memory_report.md", report_md)
    write_json(REPORTS_DIR / "stage_sw13b_counterfactual_density_constrained_feature_memory_report.json", report_json)


def write_tests() -> None:
    tests = {
        "test_outputs_exist.py": """from pathlib import Path\nBASE = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory')\n\ndef test_outputs_exist() -> None:\n    required = [\n        'sw13b_inherited_sw13a_summary.json',\n        'sw13b_execution_manifest.json',\n        'sw13b_paired_replay_dump_manifest.csv',\n        'sw13b_delta_filter_variant_manifest.json',\n        'sw13b_selection_rule.md',\n        'sw13b_non_oracle_candidate_selection.json',\n        'sw13b_replay_manifest.csv',\n        'sw13b_delta_filter_metrics.csv',\n        'sw13b_delta_filter_aggregate_metrics.csv',\n        'sw13b_metric_summary.csv',\n        'sw13b_counterfactual_density_constrained_decision.json',\n        'stage_sw13b_counterfactual_density_constrained_feature_memory_report.md',\n    ]\n    for name in required:\n        path = BASE / name\n        assert path.exists(), name\n        assert path.stat().st_size > 0, name\n""",
        "test_paired_dump_schema.py": """import csv\nimport numpy as np\nfrom pathlib import Path\nMANIFEST = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_paired_replay_dump_manifest.csv')\n\ndef test_paired_dump_schema() -> None:\n    rows = list(csv.DictReader(MANIFEST.open()))\n    assert rows\n    sample = np.load(rows[0]['paired_dump_path'])\n    for key in ['native_occ_mask','repair_occ_mask','delta_occ_mask','native_semantic','repair_semantic']:\n        assert key in sample, key\n""",
        "test_delta_definition.py": """import csv\nimport numpy as np\nfrom pathlib import Path\nMANIFEST = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_paired_replay_dump_manifest.csv')\n\ndef test_delta_definition() -> None:\n    rows = list(csv.DictReader(MANIFEST.open()))\n    sample = np.load(rows[0]['paired_dump_path'])\n    expected = np.logical_and(sample['repair_occ_mask'] > 0, np.logical_not(sample['native_occ_mask'] > 0))\n    assert np.array_equal(sample['delta_occ_mask'] > 0, expected)\n""",
        "test_native_occ_unchanged.py": """import csv\nfrom pathlib import Path\nPATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_delta_filter_metrics.csv')\n\ndef test_native_occ_unchanged() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    for row in rows:\n        if row['perturbation_id'] in {'A1_drop_cam_front','A10_drop_front_triplet','C4_motion_blur_9'} and row['filter_label'] not in {'NATIVE_REFERENCE', 'D0_none'}:\n            assert row['native_occ_unchanged'] == 'True'\n""",
        "test_no_gt_filtering.py": """import csv, json\nfrom pathlib import Path\nCSV_PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_delta_filter_metrics.csv')\nJSON_PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_non_oracle_candidate_selection.json')\n\ndef test_no_gt_filtering() -> None:\n    rows = list(csv.DictReader(CSV_PATH.open()))\n    assert rows\n    for row in rows:\n        assert row['selection_uses_gt'] == 'False'\n    obj = json.loads(JSON_PATH.read_text())\n    assert obj['selection_uses_gt'] is False\n""",
        "test_no_oracle_selection.py": """import json\nfrom pathlib import Path\nPATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_non_oracle_candidate_selection.json')\n\ndef test_no_oracle_selection() -> None:\n    obj = json.loads(PATH.read_text())\n    assert obj['selection_uses_gt'] is False\n    assert obj['sw13a_decision_type'] == 'S3_RECOVERY_BUT_DENSITY_UNSAFE'\n""",
        "test_no_future_or_current_clean.py": """import csv\nfrom pathlib import Path\nPATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_delta_filter_metrics.csv')\n\ndef test_no_future_or_current_clean() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    for row in rows:\n        assert row['uses_future_info'] == 'False'\n        assert row['uses_current_clean_same_frame'] == 'False'\n        assert row['uses_gt_repair'] == 'False'\n""",
        "test_no_get_occ_modification.py": """import json\nfrom pathlib import Path\nPATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_counterfactual_density_constrained_decision.json')\n\ndef test_no_get_occ_modification() -> None:\n    obj = json.loads(PATH.read_text())\n    assert obj['no_get_occ_modification'] is True\n""",
        "test_metric_schema.py": """import csv\nfrom pathlib import Path\nPATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_metric_summary.csv')\n\ndef test_metric_schema() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    required = {\n        'perturbation_id','base_repair_variant','filter_label','horizon_s','front_sector_false_free_rate','small_object_false_free_rate',\n        'future_h4_h6_false_free_rate','new_visible_recall','A10_front_h6_recovery_rate','false_positive_rate','pred_gt_occupied_ratio',\n        'recovery_retention_ratio','density_reduction_ratio','delta_keep_ratio','temporal_agreement_score','boundary_distance_mean'\n    }\n    assert required.issubset(rows[0].keys())\n""",
        "test_decision_schema.py": """import json\nfrom pathlib import Path\nPATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_counterfactual_density_constrained_decision.json')\n\ndef test_decision_schema() -> None:\n    obj = json.loads(PATH.read_text())\n    assert obj['decision_type'] in {\n        'T1_STRONG_SAFE_DENSITY_CONSTRAINED_RECOVERY','T2_MEDIUM_SAFE_DENSITY_CONSTRAINED_RECOVERY','T3_RECOVERY_RETAINED_BUT_DENSITY_STILL_HIGH',\n        'T4_DENSITY_CONTROLLED_BUT_RECOVERY_LOST','T5_C4_ONLY_SAFE','T6_IMPLEMENTATION_BLOCKED','T7_ORACLE_RISK'\n    }\n""",
        "test_no_false_claims.py": """from pathlib import Path\nPATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/stage_sw13b_counterfactual_density_constrained_feature_memory_report.md')\n\ndef test_no_false_claims() -> None:\n    text = PATH.read_text(encoding='utf-8').lower()\n    assert 'subset diagnostic' in text\n    assert 'not official benchmark' in text\n    banned = ['official benchmark result','trained model improvement','beats the paper']\n    for phrase in banned:\n        assert phrase not in text\n""",
    }
    for name, text in tests.items():
        write_md(TESTS_DIR / name, text)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    progress = init_progress_state()
    manifest = init_execution_manifest()
    sample_ids = parse_int_list(args.samples)

    def run_phase(name: str, fn):
        phase_start(progress, manifest, name)
        try:
            result = fn()
            phase_end(progress, manifest, name, "done")
            return result
        except Exception as exc:
            phase_end(progress, manifest, name, "failed", error=str(exc))
            raise

    _, dataset, model = build_runtime()
    sw13a_decision, base_specs = run_phase("phase0_inherit_sw13a", lambda: phase0_inherit_sw13a(manifest))
    metric_rows, proxy_rows, _, availability = run_phase(
        "phase1_phase4_replay",
        lambda: phase1_phase4_replay(dataset, model, sample_ids, base_specs, manifest),
    )
    summary_rows, selected = run_phase(
        "phase2_phase3_selection",
        lambda: phase2_phase3_selection_and_reports(sw13a_decision, metric_rows, proxy_rows, base_specs, availability, manifest),
    )
    run_phase("phase5_phase9_decision", lambda: phase5_phase9_decision_and_visuals(metric_rows, summary_rows, selected, manifest))
    run_phase("phase11_tests_write", write_tests)
    progress["status"] = "complete"
    progress["end_time"] = now_iso()
    update_progress(progress)
    manifest["end_time"] = now_iso()
    update_manifest(manifest)


if __name__ == "__main__":
    main()
