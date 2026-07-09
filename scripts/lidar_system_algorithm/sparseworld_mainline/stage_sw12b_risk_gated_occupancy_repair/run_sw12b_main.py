from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
BASE_CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
BASE_CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"

SW12A_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"
SW101_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
SW11_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
SW10_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
SW91_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW81_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune"
SW91_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW10_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
SW11_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"

PROGRESS_PATH = REPORTS_DIR / "sw12b_progress_state.json"
EXECUTION_MANIFEST_PATH = REPORTS_DIR / "sw12b_execution_manifest.json"
EMPTY_IDX = 17
TRAIN_SAMPLE_IDS = list(range(20, 40))
EVAL_SAMPLE_IDS = list(range(20))
QUICK_DEBUG_IDS = list(range(5))
CORE_HORIZONS = [0, 2, 4, 6]
CORE_PERTURBATIONS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
PAIR_PERTURBATIONS = ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]
ACTIVE_VOXEL_CAP = 0.03


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw1 = load_module(
    "sw12b_sw1",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw1_bringup/postprocess_sparseworld_outputs.py",
)
sw2 = load_module(
    "sw12b_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw12b_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw12b_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw12b_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw9 = load_module(
    "sw12b_sw9",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/run_sparseworld_sw9_main.py",
)

from mmcv.parallel import collate as collate_fn


@dataclass
class CheckpointSpec:
    name: str
    checkpoint_path: Path
    config_path: Path
    family: str


@dataclass
class RoutingVariant:
    label: str
    variant_name: str
    high_conf_thr: float | None
    density_cap_ratio: float | None
    max_extra: int | None
    class_margin_thr: float
    allowed_class_ids: list[int] | None
    restrict_class_ids: list[int] | None
    risk_target_only: bool
    risk_gate_min_hits: int
    is_oracle: bool = False


@dataclass
class BranchBConfig:
    name: str
    lambda_occ: float
    lambda_sem: float
    lambda_recovery: float
    lambda_fp_guard: float
    lambda_density: float
    perturbations: list[str]
    front_future_weight: float = 2.0
    small_weight: float = 2.0
    new_visible_weight: float = 2.0
    dynamic_weight: float = 1.5


class OccupancyRepairMLP(nn.Module):
    def __init__(self, num_classes: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, 8)
        self.horizon_embed = nn.Embedding(4, 4)
        self.pert_embed = nn.Embedding(4, 4)
        self.mlp = nn.Sequential(
            nn.Linear(8 + 4 + 4 + 12, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        student_pred_class: torch.Tensor,
        horizon_index: torch.Tensor,
        perturb_index: torch.Tensor,
        numeric_feats: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.class_embed(student_pred_class),
                self.horizon_embed(horizon_index),
                self.pert_embed(perturb_index),
                numeric_feats,
            ],
            dim=-1,
        )
        return self.mlp(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-12B risk-gated occupancy repair")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-phase", default=None)
    parser.add_argument("--stop-after-phase", default=None)
    parser.add_argument("--train-samples", default="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39")
    parser.add_argument("--eval-samples", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "branchA_replay_dumps",
        ARTIFACTS_DIR / "teacher_cache",
        ARTIFACTS_DIR / "student_teacher_pairs",
        ARTIFACTS_DIR / "checkpoints",
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def safe_cuda_cleanup() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def init_progress_state() -> dict[str, Any]:
    if PROGRESS_PATH.exists():
        return read_json(PROGRESS_PATH)
    payload = {
        "stage": "SW-12B",
        "start_time": now_iso(),
        "current_phase": None,
        "completed_phases": [],
        "phase_records": {},
        "status": "running",
    }
    write_json(PROGRESS_PATH, payload)
    return payload


def init_execution_manifest() -> dict[str, Any]:
    if EXECUTION_MANIFEST_PATH.exists():
        return read_json(EXECUTION_MANIFEST_PATH)
    payload = {
        "stage": "SW-12B",
        "subtitle": "Risk-gated / Class-aware Routing + Clean-to-Degraded Consistency Fine-tuning",
        "start_time": now_iso(),
        "phases": [],
        "artifacts": [],
        "safe_claim_boundary": [
            "subset diagnostic",
            "replay only",
            "smoke training",
            "short training",
            "not official benchmark",
        ],
    }
    write_json(EXECUTION_MANIFEST_PATH, payload)
    return payload


def update_progress(progress: dict[str, Any]) -> None:
    write_json(PROGRESS_PATH, progress)


def update_execution_manifest(manifest: dict[str, Any]) -> None:
    write_json(EXECUTION_MANIFEST_PATH, manifest)


def phase_start(progress: dict[str, Any], manifest: dict[str, Any], phase_name: str) -> dict[str, Any]:
    meta = {"phase_name": phase_name, "start_time": now_iso(), "start_ts": time.time(), "status": "running"}
    progress["current_phase"] = phase_name
    progress["phase_records"][phase_name] = meta
    manifest["phases"].append(meta.copy())
    update_progress(progress)
    update_execution_manifest(manifest)
    return meta


def phase_end(progress: dict[str, Any], manifest: dict[str, Any], phase_name: str, status: str, **extra: Any) -> None:
    record = progress["phase_records"][phase_name]
    end_ts = time.time()
    record.update({"end_time": now_iso(), "end_ts": end_ts, "duration_sec": end_ts - record["start_ts"], "status": status, **extra})
    progress["current_phase"] = None
    if status == "done" and phase_name not in progress["completed_phases"]:
        progress["completed_phases"].append(phase_name)
    progress["status"] = "running"
    for item in manifest["phases"]:
        if item["phase_name"] == phase_name and item["start_time"] == record["start_time"]:
            item.update(record)
            break
    update_progress(progress)
    update_execution_manifest(manifest)


def attach_artifact(manifest: dict[str, Any], label: str, path: Path) -> None:
    manifest["artifacts"].append({"label": label, "path": str(path), "timestamp": now_iso()})
    update_execution_manifest(manifest)


def discover_checkpoint_specs() -> list[CheckpointSpec]:
    route_selection = read_json(SW10_REPORTS / "sw10_route_selection.json")
    selected_config = Path(route_selection.get("selected_config_path", REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_tiny_h1_lambda001.py"))
    return [
        CheckpointSpec("epoch_56_original", BASE_CHECKPOINT_PATH, BASE_CONFIG_PATH, "baseline"),
        CheckpointSpec("P3_survival_aware_S3_20iter", SW11_ARTIFACTS / "checkpoints/P3_survival_aware_S3_20iter.pth", BASE_CONFIG_PATH, "sw11_smoke"),
        CheckpointSpec("P4_H2_tinyH1_500iter", SW91_ARTIFACTS / "checkpoints/P4_H2_tinyH1_500iter.pth", selected_config, "sw91_short"),
        CheckpointSpec("routeA_best_h2_scaled_iter3000", SW10_ARTIFACTS / "checkpoints/routeA_best_h2_scaled_iter3000.pth", selected_config, "sw10_scaled"),
    ]


def build_runtime(config_path: Path, checkpoint_path: Path) -> tuple[Any, Any, Any]:
    cfg, dataset, model, _ = sw9.build_runtime(
        config_path,
        train=False,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return cfg, dataset, model


def attach_query_capture(model: Any, holder: dict[str, Any]) -> Any:
    return sw81.attach_query_capture(model, holder)


def tensor_to_numpy(tensor: torch.Tensor | None, dtype: np.dtype | None = None) -> np.ndarray:
    if tensor is None:
        return np.array([], dtype=np.float32)
    arr = tensor.detach().cpu().numpy()
    return arr.astype(dtype) if dtype is not None else arr


def sparse_count_repr(dense: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    coords = torch.nonzero(dense > 0, as_tuple=False).cpu().numpy().astype(np.int16)
    vals = dense[dense > 0].detach().cpu().numpy()
    return coords, vals


def load_case(
    model: Any,
    dataset: Any,
    sample_index: int,
    perturbation_id: str,
    horizons: list[int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    holder: dict[str, Any] = {}
    original_forward = attach_query_capture(model, holder)
    try:
        raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        if perturbation_id != "A0_clean":
            batch = sw81.sw5_engine.apply_perturbation_to_batch(batch, sw81.sw5_engine.build_catalog()[perturbation_id])[0]
        sw81.reset_online_cache(model)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **sw2.move_to_cuda(batch))
        raw_result_cpu = sw2.to_cpu_artifact(result)
        query_cpu = sw2.to_cpu_artifact(holder)
        _, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
        per_h: dict[int, dict[str, Any]] = {}
        for horizon_s in horizons:
            pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
            gt_h = gt_temporal[horizon_s].long().cpu()
            gt0 = gt_temporal[0].long().cpu()
            per_h[horizon_s] = {
                "pred_dict": pred_dict,
                "gt_h": gt_h,
                "gt0": gt0,
                "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
            }
        return sample_unwrapped, per_h
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]


def derive_top_conf_and_margin(dense_occ_after_padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = dense_occ_after_padding.float()
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    top1_conf = top2.values[..., 0]
    top1_cls = top2.indices[..., 0].long()
    margin = top2.values[..., 0] - (top2.values[..., 1] if top2.values.shape[-1] > 1 else 0.0)
    return top1_conf.cpu(), top1_cls.cpu(), margin.cpu()


def gate_pass_dense_or_zero(debug_dict: dict[str, Any]) -> torch.Tensor:
    contributor = torch.as_tensor(debug_dict["contributor_count_dense"]).cpu()
    gate_dense = debug_dict.get("gate_pass_dense")
    if gate_dense is None:
        return torch.zeros_like(contributor)
    return torch.as_tensor(gate_dense).cpu()


def class_group_masks(gt_h: torch.Tensor, gt0: torch.Tensor) -> dict[str, torch.Tensor]:
    small = torch.zeros_like(gt_h, dtype=torch.bool)
    dynamic = torch.zeros_like(gt_h, dtype=torch.bool)
    static = torch.zeros_like(gt_h, dtype=torch.bool)
    for cid in sw2.CLASS_GROUPS["small_object"]:
        small |= gt_h == int(cid)
    for cid in sw2.CLASS_GROUPS["all_dynamic"]:
        dynamic |= gt_h == int(cid)
    for cid in sw2.CLASS_GROUPS["all_static"]:
        static |= gt_h == int(cid)
    new_visible = (gt0 == EMPTY_IDX) & (gt_h != EMPTY_IDX)
    return {
        "small_object": small,
        "dynamic": dynamic,
        "static": static,
        "new_visible": new_visible,
    }


def compute_group_false_free(pred: torch.Tensor, gt: torch.Tensor, group_mask: torch.Tensor) -> float:
    gt_occ = gt != EMPTY_IDX
    pred_occ = pred != EMPTY_IDX
    den = torch.logical_and(group_mask, gt_occ).sum().item()
    num = torch.logical_and(torch.logical_and(group_mask, gt_occ), ~pred_occ).sum().item()
    return float(num / den) if den else 0.0


def compute_group_recall(pred: torch.Tensor, gt: torch.Tensor, group_mask: torch.Tensor) -> float:
    gt_occ = gt != EMPTY_IDX
    pred_occ = pred != EMPTY_IDX
    den = torch.logical_and(group_mask, gt_occ).sum().item()
    num = torch.logical_and(torch.logical_and(group_mask, gt_occ), pred_occ).sum().item()
    return float(num / den) if den else 0.0


def compute_wrong_class_activation(pred: torch.Tensor, gt: torch.Tensor) -> float:
    gt_occ = gt != EMPTY_IDX
    pred_occ = pred != EMPTY_IDX
    den = gt_occ.sum().item()
    wrong = torch.logical_and(torch.logical_and(gt_occ, pred_occ), pred != gt).sum().item()
    return float(wrong / den) if den else 0.0


def build_eval_row(
    pred: torch.Tensor,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    perturbation_id: str,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    baseline_pred: torch.Tensor | None = None,
) -> dict[str, Any]:
    occ = sw1.occupancy_metrics(pred, gt_h, empty_idx=EMPTY_IDX)
    baseline_occ = sw1.occupancy_metrics(baseline_pred, gt_h, empty_idx=EMPTY_IDX) if baseline_pred is not None else None
    pred_occ = pred != EMPTY_IDX
    gt_occ = gt_h != EMPTY_IDX
    front_mask = sectors["front"]
    groups = class_group_masks(gt_h, gt0)
    front_den = torch.logical_and(front_mask, gt_occ).sum().item()
    front_ff = torch.logical_and(torch.logical_and(front_mask, gt_occ), ~pred_occ).sum().item()
    dynamic_ff = compute_group_false_free(pred, gt_h, groups["dynamic"])
    static_ff = compute_group_false_free(pred, gt_h, groups["static"])
    small_ff = compute_group_false_free(pred, gt_h, groups["small_object"])
    new_visible_recall = compute_group_recall(pred, gt_h, groups["new_visible"])
    teacher_reliability_front = 1.0 - float(front_ff / front_den) if front_den else 1.0
    a10_front_h6_recovery = 0.0
    if perturbation_id == "A10_drop_front_triplet" and horizon_s == 6:
        a10_front_h6_recovery = safe_div(torch.logical_and(torch.logical_and(pred == gt_h, front_mask), gt_occ).sum().item(), front_den)
    return {
        "occupied_iou": occ["occupied_iou"],
        "semantic_miou": occ["semantic_miou"],
        "false_free_rate": occ["false_free_rate"],
        "false_occupied_rate": occ["false_occupied_rate"],
        "pred_gt_occupied_ratio": safe_div(occ["pred_occupied_count"], max(1, occ["gt_occupied_count"])),
        "small_object_false_free": small_ff,
        "new_visible_recall": new_visible_recall,
        "front_sector_false_free": float(front_ff / front_den) if front_den else 0.0,
        "dynamic_false_free": dynamic_ff,
        "static_false_free": static_ff,
        "active_voxel_count": int(pred_occ.sum().item()),
        "wrong_class_activation": compute_wrong_class_activation(pred, gt_h),
        "A10_front_h6_recovery_ratio": a10_front_h6_recovery,
        "reliability_front_sector": teacher_reliability_front,
        "reliability_small_object": 1.0 - small_ff,
        "reliability_new_visible": new_visible_recall,
        "risk_error_correlation": float(front_ff / front_den) if front_den else 0.0,
        "clean_occupied_iou_delta": 0.0 if baseline_occ is None else occ["occupied_iou"] - baseline_occ["occupied_iou"],
        "clean_semantic_miou_delta": 0.0 if baseline_occ is None else occ["semantic_miou"] - baseline_occ["semantic_miou"],
        "clean_false_positive_delta": 0.0 if baseline_occ is None else occ["false_occupied_rate"] - baseline_occ["false_occupied_rate"],
        "active_voxel_count_delta": 0.0 if baseline_pred is None else safe_div(pred_occ.sum().item() - (baseline_pred != EMPTY_IDX).sum().item(), max(1, (baseline_pred != EMPTY_IDX).sum().item())),
        "pred_gt_density_delta": 0.0 if baseline_pred is None else safe_div(pred_occ.sum().item(), max(1, gt_occ.sum().item())) - safe_div((baseline_pred != EMPTY_IDX).sum().item(), max(1, gt_occ.sum().item())),
        "wrong_class_activation_delta": 0.0 if baseline_pred is None else compute_wrong_class_activation(pred, gt_h) - compute_wrong_class_activation(baseline_pred, gt_h),
        "C4_false_positive_delta": 0.0 if baseline_occ is None else occ["false_occupied_rate"] - baseline_occ["false_occupied_rate"],
    }


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        agg = {k: v for k, v in zip(group_keys, key)}
        for field in bucket[0]:
            if field in group_keys:
                continue
            if isinstance(bucket[0][field], (int, float, np.floating, np.integer)):
                agg[field] = float(np.mean([float(item[field]) for item in bucket]))
        agg["case_count"] = len(bucket)
        out.append(agg)
    return out


def discover_best_sw81_reference() -> dict[str, Any] | None:
    path = SW81_REPORTS / "sw81_fixed_subset_eval.csv"
    rows = read_csv_rows(path)
    if not rows:
        return None
    return rows[0]


def build_branch_a_variants() -> list[RoutingVariant]:
    dynamic_small = sorted(set(sw2.CLASS_GROUPS["small_object"] + sw2.CLASS_GROUPS["all_dynamic"]))
    restrict_static_large = sorted(set(sw2.CLASS_GROUPS["all_static"]) - set(sw2.CLASS_GROUPS["small_object"]))
    return [
        RoutingVariant("A0_native", "native", None, None, None, 0.0, None, None, False, 0),
        RoutingVariant("A1_risk_gated_6n_conf095_cap005", "density_capped_r1", 0.95, 0.005, 1, 0.0, None, None, True, 2),
        RoutingVariant("A2_risk_gated_6n_conf09_cap01", "density_capped_r1", 0.90, 0.01, 1, 0.0, None, None, True, 2),
        RoutingVariant("A3_class_aware_6n_conf09_cap01_margin02", "density_capped_r1", 0.90, 0.01, 1, 0.2, dynamic_small, restrict_static_large, True, 2),
        RoutingVariant("A4_semantic_margin_6n_conf095_cap01_margin03", "density_capped_r1", 0.95, 0.01, 1, 0.3, dynamic_small, restrict_static_large, True, 2),
        RoutingVariant("A5_risk_class_density_cap_6n_margin02_cap005", "density_capped_r1", 0.95, 0.005, 1, 0.2, dynamic_small, restrict_static_large, True, 2),
        RoutingVariant("A6_oracle_upper_bound", "survival_oracle_r1_diagnostic", 0.95, 0.005, 1, 0.2, dynamic_small, None, True, 2, is_oracle=True),
    ]


def variant_cfg_from_branch_a(spec: RoutingVariant) -> dict[str, Any]:
    return {
        "get_occ_variant": spec.variant_name,
        "neighbor_topology": "6",
        "distance_weighting": "uniform",
        "high_conf_thr": spec.high_conf_thr,
        "density_cap_ratio": spec.density_cap_ratio,
        "max_extra_contrib_per_voxel": spec.max_extra,
        "gaussian_sigma": 1.0,
        "near_dist_band": 0.25,
        "class_margin_thr": spec.class_margin_thr,
        "allowed_class_ids": spec.allowed_class_ids,
        "restrict_class_ids": spec.restrict_class_ids,
        "risk_target_only": spec.risk_target_only,
        "risk_gate_min_hits": spec.risk_gate_min_hits,
    }


def build_risk_maps(
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    native_pred: torch.Tensor,
    native_contributor_dense: torch.Tensor,
    top1_conf: torch.Tensor,
    pred_top1_cls: torch.Tensor,
    perturbation_id: str,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    gt_occ = gt_h != EMPTY_IDX
    native_occ = native_pred != EMPTY_IDX
    false_free = gt_occ & (~native_occ)
    groups = class_group_masks(gt_h, gt0)
    front_mask = sectors["front"]
    risk_count = torch.zeros_like(gt_h, dtype=torch.int16)
    risk_count += front_mask.to(torch.int16)
    if perturbation_id in {"A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"}:
        risk_count += torch.ones_like(risk_count)
    if horizon_s in {4, 6}:
        risk_count += torch.ones_like(risk_count)
    risk_count += (native_contributor_dense <= 0).to(torch.int16)
    risk_count += false_free.to(torch.int16)
    risk_count += groups["small_object"].to(torch.int16)
    risk_count += groups["new_visible"].to(torch.int16)
    risk_count += (top1_conf < 0.25).to(torch.int16)
    risk_count += torch.isin(pred_top1_cls, torch.as_tensor(sw2.CLASS_GROUPS["all_dynamic"])).to(torch.int16)
    risk_mask = risk_count >= 2
    return risk_mask.cpu(), risk_count.cpu()


def collect_variant_occ(
    head: Any,
    pred_dict: dict[str, torch.Tensor],
    spec: RoutingVariant,
    risk_target_mask: torch.Tensor,
    risk_score_dense: torch.Tensor,
    oracle_target_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    runtime_cfg = {
        "variant_cfg": variant_cfg_from_branch_a(spec),
        "risk_target_mask": risk_target_mask.cuda(non_blocking=False),
        "risk_score_dense": risk_score_dense.cuda(non_blocking=False),
    }
    if spec.is_oracle:
        runtime_cfg["oracle_target_mask"] = oracle_target_mask.cuda(non_blocking=False)
    head.get_occ_variant_cfg = variant_cfg_from_branch_a(spec)
    head._sw12a_get_occ_runtime = runtime_cfg
    with torch.no_grad():
        occ_pred = head.get_occ(pred_dict)[0]
    debug_list = getattr(head, "_last_get_occ_variant_debug", [])
    debug = debug_list[0] if debug_list else {}
    head.get_occ_variant_cfg = {"get_occ_variant": "native"}
    head._sw12a_get_occ_runtime = {}
    keep_debug_keys = {
        "variant_name",
        "neighbor_topology",
        "distance_weighting",
        "high_conf_thr",
        "density_cap_ratio",
        "max_extra_contrib_per_voxel",
        "extra_routed_contributor_count_dense",
    }
    out_debug = {
        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in debug.items()
        if k in keep_debug_keys
    }
    return occ_pred.detach().cpu().long(), out_debug


def branch_a_replay_case(
    checkpoint_spec: CheckpointSpec,
    sample_index: int,
    perturbation_id: str,
    horizon_s: int,
    head: Any,
    pred_dict: dict[str, torch.Tensor],
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    sectors: dict[str, torch.Tensor],
    variants: list[RoutingVariant],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    native_pred_dbg, native_dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
    native_occ = native_pred_dbg[0].detach().cpu().long()
    native_debug = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in native_dbg_list[0].items()}
    top1_conf, pred_top1_cls, top1_margin = derive_top_conf_and_margin(native_debug["dense_occ_after_padding"])
    native_contrib_dense = torch.as_tensor(native_debug["contributor_count_dense"]).cpu()
    risk_mask, risk_score_dense = build_risk_maps(
        gt_h,
        gt0,
        native_occ,
        native_contrib_dense,
        top1_conf,
        pred_top1_cls,
        perturbation_id,
        horizon_s,
        sectors,
    )
    oracle_target_mask = (gt_h != EMPTY_IDX) & (native_occ == EMPTY_IDX)
    dump_payload = {
        "native_occ": native_occ.numpy().astype(np.uint8),
        "gt_occ": gt_h.numpy().astype(np.uint8),
        "risk_mask_coords": torch.nonzero(risk_mask, as_tuple=False).cpu().numpy().astype(np.int16),
        "oracle_target_coords": torch.nonzero(oracle_target_mask, as_tuple=False).cpu().numpy().astype(np.int16),
    }
    rows: list[dict[str, Any]] = []
    baseline_eval = build_eval_row(native_occ, gt_h, gt0, perturbation_id, horizon_s, sectors, baseline_pred=None)
    for spec in variants:
        if spec.label == "A0_native":
            variant_occ = native_occ
            variant_debug = native_debug
        else:
            variant_occ, variant_debug = collect_variant_occ(head, pred_dict, spec, risk_mask, risk_score_dense, oracle_target_mask)
        variant_eval = build_eval_row(variant_occ, gt_h, gt0, perturbation_id, horizon_s, sectors, baseline_pred=native_occ)
        front_gt = torch.logical_and(sectors["front"], gt_h != EMPTY_IDX)
        front_ff_base = torch.logical_and(front_gt, native_occ == EMPTY_IDX)
        front_ff_variant = torch.logical_and(front_gt, variant_occ == EMPTY_IDX)
        targeted_hit = float(front_ff_variant.sum().item()) < float(front_ff_base.sum().item())
        rows.append(
            {
                "checkpoint_name": checkpoint_spec.name,
                "sample_index": sample_index,
                "perturbation_id": perturbation_id,
                "horizon_s": horizon_s,
                "variant_label": spec.label,
                "variant_name": spec.variant_name,
                "is_oracle": spec.is_oracle,
                "target_recovery": safe_div((front_ff_base & (variant_occ != EMPTY_IDX)).sum().item(), max(1, front_ff_base.sum().item())),
                "neighbor_leakage": safe_div(torch.logical_and(variant_occ != EMPTY_IDX, gt_h == EMPTY_IDX).sum().item(), max(1, (variant_occ != EMPTY_IDX).sum().item())),
                "recovery_per_extra_voxel": safe_div((front_ff_base & (variant_occ != EMPTY_IDX)).sum().item(), max(1, int(torch.as_tensor(variant_debug.get("extra_routed_contributor_count_dense", torch.zeros_like(gt_h))).sum().item()))),
                "recovery_per_false_positive": safe_div((front_ff_base & (variant_occ != EMPTY_IDX)).sum().item(), max(1, int(torch.logical_and(variant_occ != EMPTY_IDX, gt_h == EMPTY_IDX).sum().item()))),
                **variant_eval,
                "baseline_occupied_iou": baseline_eval["occupied_iou"],
                "baseline_semantic_miou": baseline_eval["semantic_miou"],
                "targeted_hit_flag": targeted_hit,
            }
        )
        dump_payload[f"{spec.label}__pred_occ"] = variant_occ.numpy().astype(np.uint8)
    return rows, dump_payload


def branch_a_select_candidates(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_rows = {
        (row["checkpoint_name"], row["perturbation_id"], str(row["horizon_s"])): row
        for row in agg_rows
        if row["variant_label"] == "A0_native"
    }
    candidate_rows: list[dict[str, Any]] = []
    for row in agg_rows:
        is_real_variant = row["variant_label"] != "A0_native" and not truthy(row["is_oracle"])
        baseline_row = baseline_rows.get((row["checkpoint_name"], row["perturbation_id"], str(row["horizon_s"])))
        front_sector_false_free_delta = (
            float(row["front_sector_false_free"]) - float(baseline_row["front_sector_false_free"])
            if baseline_row is not None
            else 0.0
        )
        small_object_false_free_delta = (
            float(row["small_object_false_free"]) - float(baseline_row["small_object_false_free"])
            if baseline_row is not None
            else 0.0
        )
        new_visible_recall_delta = (
            float(row["new_visible_recall"]) - float(baseline_row["new_visible_recall"])
            if baseline_row is not None
            else 0.0
        )
        false_occupied_rate_delta = (
            float(row["false_occupied_rate"]) - float(baseline_row["false_occupied_rate"])
            if baseline_row is not None
            else 0.0
        )
        safe_gate = bool(
            float(row["clean_occupied_iou_delta"]) >= -0.005
            and float(row["clean_semantic_miou_delta"]) >= -0.005
            and float(row["clean_false_positive_delta"]) <= 0.005
            and float(row["C4_false_positive_delta"]) <= 0.008
            and float(row["pred_gt_density_delta"]) <= 0.05
            and float(row["wrong_class_activation_delta"]) <= 0.005
            and float(row["active_voxel_count_delta"]) <= ACTIVE_VOXEL_CAP
        )
        targeted_hit = bool(
            is_real_variant
            and (
                (
                    row["perturbation_id"] == "A10_drop_front_triplet"
                    and (
                        front_sector_false_free_delta <= -0.03
                        or small_object_false_free_delta <= -0.03
                        or new_visible_recall_delta >= 0.03
                        or (
                            int(row["horizon_s"]) == 6
                            and (
                                float(row["target_recovery"]) > 0.0
                                or float(row.get("targeted_hit_flag", 0.0)) > 0.0
                            )
                        )
                    )
                )
                or (
                    row["perturbation_id"] == "C4_motion_blur_9"
                    and false_occupied_rate_delta <= -0.005
                )
            )
        )
        if truthy(row["is_oracle"]):
            classification = "A_ORACLE_ONLY"
        elif safe_gate and targeted_hit:
            classification = "A_SAFE_TARGETED"
        elif (not safe_gate) and targeted_hit:
            classification = "A_TARGETED_UNSAFE"
        elif safe_gate:
            classification = "A_SAFE_NO_TARGET"
        else:
            classification = "A_NO_SIGNAL"
        candidate = dict(row)
        candidate["classification"] = classification
        candidate["safe_gate"] = safe_gate
        candidate["targeted_hit"] = targeted_hit
        candidate["front_sector_false_free_delta"] = front_sector_false_free_delta
        candidate["small_object_false_free_delta"] = small_object_false_free_delta
        candidate["new_visible_recall_delta"] = new_visible_recall_delta
        candidate["false_occupied_rate_delta"] = false_occupied_rate_delta
        candidate_rows.append(candidate)
    return candidate_rows


def branch_b_configs() -> list[BranchBConfig]:
    return [
        BranchBConfig("B0_baseline_continue", 0.0, 0.0, 0.0, 0.0, 0.0, PAIR_PERTURBATIONS),
        BranchBConfig("B1_occ_consistency_light", 0.5, 0.0, 0.0, 0.0, 0.0, PAIR_PERTURBATIONS),
        BranchBConfig("B2_occ_sem_consistency", 1.0, 0.5, 0.0, 0.0, 0.0, PAIR_PERTURBATIONS),
        BranchBConfig("B3_risk_weighted_consistency", 1.0, 0.5, 1.0, 0.0, 0.0, PAIR_PERTURBATIONS),
        BranchBConfig("B4_consistency_with_fp_guard", 1.0, 0.5, 1.0, 1.0, 0.5, PAIR_PERTURBATIONS),
        BranchBConfig("B5_A10_only_consistency", 1.0, 0.5, 1.0, 1.0, 0.5, ["A10_drop_front_triplet"]),
        BranchBConfig("B6_mixed_degradation_consistency", 1.0, 0.5, 1.0, 1.0, 0.5, PAIR_PERTURBATIONS),
    ]


def perturb_index(perturbation_id: str) -> int:
    mapping = {
        "A1_drop_cam_front": 0,
        "A10_drop_front_triplet": 1,
        "C4_motion_blur_9": 2,
        "A0_clean": 3,
        "A7_drop_all_rear": 3,
    }
    return mapping.get(perturbation_id, 3)


def horizon_index(horizon_s: int) -> int:
    return {0: 0, 2: 1, 4: 2, 6: 3}.get(horizon_s, 0)


def build_numeric_features(
    pred_class: torch.Tensor,
    pred_conf: torch.Tensor,
    pred_margin: torch.Tensor,
    contributor_count: torch.Tensor,
    gate_count: torch.Tensor,
    coords: torch.Tensor,
    front_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    small_mask: torch.Tensor,
    new_visible_mask: torch.Tensor,
    baseline_occ: torch.Tensor,
) -> torch.Tensor:
    x_norm = coords[:, 0].float() / max(1, pred_class.shape[0] - 1)
    y_norm = coords[:, 1].float() / max(1, pred_class.shape[1] - 1)
    z_norm = coords[:, 2].float() / max(1, pred_class.shape[2] - 1)
    return torch.stack(
        [
            pred_conf[coords[:, 0], coords[:, 1], coords[:, 2]].float(),
            pred_margin[coords[:, 0], coords[:, 1], coords[:, 2]].float(),
            contributor_count[coords[:, 0], coords[:, 1], coords[:, 2]].float().clamp(max=8) / 8.0,
            gate_count[coords[:, 0], coords[:, 1], coords[:, 2]].float().clamp(max=8) / 8.0,
            front_mask[coords[:, 0], coords[:, 1], coords[:, 2]].float(),
            dynamic_mask[coords[:, 0], coords[:, 1], coords[:, 2]].float(),
            small_mask[coords[:, 0], coords[:, 1], coords[:, 2]].float(),
            new_visible_mask[coords[:, 0], coords[:, 1], coords[:, 2]].float(),
            baseline_occ[coords[:, 0], coords[:, 1], coords[:, 2]].float(),
            x_norm,
            y_norm,
            z_norm,
        ],
        dim=-1,
    )


def make_pair_npz(
    teacher_label: torch.Tensor,
    teacher_conf: torch.Tensor,
    gt_h: torch.Tensor,
    student_label: torch.Tensor,
    student_conf: torch.Tensor,
    student_margin: torch.Tensor,
    contributor_count: torch.Tensor,
    gate_count: torch.Tensor,
    front_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    small_mask: torch.Tensor,
    new_visible_mask: torch.Tensor,
    perturbation_id: str,
    horizon_s: int,
) -> dict[str, Any]:
    teacher_occ = teacher_label != EMPTY_IDX
    student_occ = student_label != EMPTY_IDX
    reliable_teacher = teacher_occ & (teacher_conf >= 0.7)
    risk_mask = front_mask | small_mask | new_visible_mask | dynamic_mask
    pos_mask = reliable_teacher & (~student_occ)
    if horizon_s in {4, 6}:
        pos_mask = pos_mask | (reliable_teacher & risk_mask)
    neg_mask = (~teacher_occ) & (student_occ | risk_mask)
    gt_free = gt_h == EMPTY_IDX
    neg_mask = neg_mask & gt_free
    all_pos = torch.nonzero(pos_mask, as_tuple=False)
    all_neg = torch.nonzero(neg_mask, as_tuple=False)
    if all_pos.shape[0] > 2048:
        all_pos = all_pos[torch.randperm(all_pos.shape[0])[:2048]]
    if all_neg.shape[0] > 2048:
        all_neg = all_neg[torch.randperm(all_neg.shape[0])[:2048]]
    coords = torch.cat([all_pos, all_neg], dim=0)
    pos_count = all_pos.shape[0]
    neg_count = all_neg.shape[0]
    if coords.shape[0] == 0:
        coords = torch.zeros((0, 3), dtype=torch.long)
    numeric = build_numeric_features(
        student_label,
        student_conf,
        student_margin,
        contributor_count,
        gate_count,
        coords,
        front_mask,
        dynamic_mask,
        small_mask,
        new_visible_mask,
        student_occ,
    )
    teacher_target = teacher_label[coords[:, 0], coords[:, 1], coords[:, 2]] if coords.shape[0] else teacher_label.new_zeros((0,))
    teacher_target_occ = (teacher_target != EMPTY_IDX).long()
    sample_weights = torch.ones(coords.shape[0], dtype=torch.float32)
    if coords.shape[0]:
        sample_weights[:pos_count] *= 2.0
        front_sel = front_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        small_sel = small_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        new_sel = new_visible_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        dyn_sel = dynamic_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        sample_weights = sample_weights * (1.0 + front_sel.float() + small_sel.float() + new_sel.float() + 0.5 * dyn_sel.float())
    return {
        "coords": coords.cpu().numpy().astype(np.int16),
        "student_pred_class": student_label[coords[:, 0], coords[:, 1], coords[:, 2]].cpu().numpy().astype(np.uint8) if coords.shape[0] else np.zeros((0,), dtype=np.uint8),
        "numeric_feats": numeric.cpu().numpy().astype(np.float32),
        "teacher_target_class": teacher_target.cpu().numpy().astype(np.uint8) if coords.shape[0] else np.zeros((0,), dtype=np.uint8),
        "teacher_target_occ": teacher_target_occ.cpu().numpy().astype(np.uint8),
        "sample_weights": sample_weights.cpu().numpy().astype(np.float32),
        "horizon_index": np.full((coords.shape[0],), horizon_index(horizon_s), dtype=np.int64),
        "perturb_index": np.full((coords.shape[0],), perturb_index(perturbation_id), dtype=np.int64),
        "teacher_conf_selected": teacher_conf[coords[:, 0], coords[:, 1], coords[:, 2]].cpu().numpy().astype(np.float32) if coords.shape[0] else np.zeros((0,), dtype=np.float32),
        "pos_count": np.array([pos_count], dtype=np.int32),
        "neg_count": np.array([neg_count], dtype=np.int32),
    }


def load_pair_dataset(pair_paths: list[Path], allowed_perturbations: list[str]) -> dict[str, torch.Tensor]:
    feats = []
    pred_class = []
    teacher_class = []
    teacher_occ = []
    weights = []
    horizon_idx = []
    perturb_idx = []
    selected_paths = [path for path in pair_paths if any(f"__{p}__" in path.name for p in allowed_perturbations)]
    for path in selected_paths:
        data = np.load(path)
        if data["coords"].shape[0] == 0:
            continue
        feats.append(torch.from_numpy(data["numeric_feats"]))
        pred_class.append(torch.from_numpy(data["student_pred_class"]).long())
        teacher_class.append(torch.from_numpy(data["teacher_target_class"]).long())
        teacher_occ.append(torch.from_numpy(data["teacher_target_occ"]).float())
        weights.append(torch.from_numpy(data["sample_weights"]).float())
        horizon_idx.append(torch.from_numpy(data["horizon_index"]).long())
        perturb_idx.append(torch.from_numpy(data["perturb_index"]).long())
    if not feats:
        zero = torch.zeros((0, 12), dtype=torch.float32)
        return {
            "numeric_feats": zero,
            "student_pred_class": torch.zeros((0,), dtype=torch.long),
            "teacher_target_class": torch.zeros((0,), dtype=torch.long),
            "teacher_target_occ": torch.zeros((0,), dtype=torch.float32),
            "sample_weights": torch.zeros((0,), dtype=torch.float32),
            "horizon_index": torch.zeros((0,), dtype=torch.long),
            "perturb_index": torch.zeros((0,), dtype=torch.long),
        }
    return {
        "numeric_feats": torch.cat(feats, dim=0),
        "student_pred_class": torch.cat(pred_class, dim=0),
        "teacher_target_class": torch.cat(teacher_class, dim=0),
        "teacher_target_occ": torch.cat(teacher_occ, dim=0),
        "sample_weights": torch.cat(weights, dim=0),
        "horizon_index": torch.cat(horizon_idx, dim=0),
        "perturb_index": torch.cat(perturb_idx, dim=0),
    }


def compute_branch_b_losses(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: BranchBConfig,
) -> dict[str, torch.Tensor]:
    target_class = batch["teacher_target_class"].cuda(non_blocking=False)
    target_occ = batch["teacher_target_occ"].cuda(non_blocking=False)
    weights = batch["sample_weights"].cuda(non_blocking=False)
    probs = logits.softmax(dim=-1)
    empty_prob = probs[:, EMPTY_IDX]
    occ_prob = 1.0 - empty_prob
    occ_logits = torch.logit(torch.clamp(occ_prob, 1e-4, 1 - 1e-4))
    ce = F.cross_entropy(logits, target_class, reduction="none")
    bce = F.binary_cross_entropy_with_logits(occ_logits, target_occ, reduction="none")
    recovery_mask = (target_occ > 0.5).float()
    reliable_free_mask = (target_occ < 0.5).float()
    loss_occ = (bce * weights).mean() * cfg.lambda_occ
    loss_sem = (ce * weights).mean() * cfg.lambda_sem
    loss_recovery = (bce * weights * recovery_mask).mean() * cfg.lambda_recovery
    loss_fp = (F.relu(occ_prob - 0.15) * reliable_free_mask * weights).mean() * cfg.lambda_fp_guard
    density_target = target_occ.mean()
    density_pred = occ_prob.mean()
    loss_density = (density_pred - density_target).abs() * cfg.lambda_density
    total = loss_occ + loss_sem + loss_recovery + loss_fp + loss_density
    return {
        "total_loss": total,
        "original_loss": torch.zeros_like(total),
        "consistency_loss": loss_occ,
        "semantic_consistency_loss": loss_sem,
        "fp_guard_loss": loss_fp,
        "density_loss": loss_density,
        "recovery_loss": loss_recovery,
        "pred_density_proxy": density_pred.detach(),
    }


def train_repair_model(
    dataset: dict[str, torch.Tensor],
    cfg: BranchBConfig,
    num_iters: int,
    checkpoint_prefix: str,
    phase_label: str,
) -> tuple[OccupancyRepairMLP, list[dict[str, Any]], Path]:
    model = OccupancyRepairMLP().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rows: list[dict[str, Any]] = []
    batch_size = 4096
    n = dataset["numeric_feats"].shape[0]
    last_checkpoint = ARTIFACTS_DIR / "checkpoints" / f"{checkpoint_prefix}_{num_iters}iter.pth"
    for step in range(num_iters):
        if n == 0:
            break
        idx = torch.randint(0, n, (min(batch_size, n),))
        batch = {k: v[idx] if v.shape[0] == n else v for k, v in dataset.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["student_pred_class"].cuda(non_blocking=False),
            batch["horizon_index"].cuda(non_blocking=False),
            batch["perturb_index"].cuda(non_blocking=False),
            batch["numeric_feats"].cuda(non_blocking=False),
        )
        losses = compute_branch_b_losses(logits, batch, cfg)
        losses["total_loss"].backward()
        grad_sq = 0.0
        grad_finite = True
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_sq += float(param.grad.detach().float().norm().item() ** 2)
            grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
        optimizer.step()
        rows.append(
            {
                "phase_label": phase_label,
                "config_name": cfg.name,
                "iter": step + 1,
                "total_loss": float(losses["total_loss"].detach().cpu().item()),
                "original_loss": float(losses["original_loss"].detach().cpu().item()),
                "consistency_loss": float(losses["consistency_loss"].detach().cpu().item()),
                "semantic_consistency_loss": float(losses["semantic_consistency_loss"].detach().cpu().item()),
                "fp_guard_loss": float(losses["fp_guard_loss"].detach().cpu().item()),
                "density_loss": float(losses["density_loss"].detach().cpu().item()),
                "recovery_loss": float(losses["recovery_loss"].detach().cpu().item()),
                "grad_norm": math.sqrt(max(0.0, grad_sq)),
                "grad_finite": grad_finite,
                "teacher_voxel_count": int((dataset["teacher_target_occ"] > 0.5).sum().item()),
                "recovery_target_count": int((dataset["teacher_target_occ"] > 0.5).sum().item()),
                "negative_guard_count": int((dataset["teacher_target_occ"] < 0.5).sum().item()),
                "pred_density_proxy": float(losses["pred_density_proxy"].cpu().item()),
                "has_nan": not grad_finite,
                "has_inf": not grad_finite,
            }
        )
        if (step + 1) in {20, 100, 500, 1000} or step + 1 == num_iters:
            torch.save({"state_dict": model.state_dict(), "config_name": cfg.name, "iter": step + 1}, ARTIFACTS_DIR / "checkpoints" / f"{checkpoint_prefix}_{step + 1}iter.pth")
            last_checkpoint = ARTIFACTS_DIR / "checkpoints" / f"{checkpoint_prefix}_{step + 1}iter.pth"
    return model, rows, last_checkpoint


def load_repair_model(checkpoint_path: Path) -> OccupancyRepairMLP:
    model = OccupancyRepairMLP().cuda()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def repair_prediction_with_model(
    repair_model: OccupancyRepairMLP,
    student_label: torch.Tensor,
    student_conf: torch.Tensor,
    student_margin: torch.Tensor,
    contributor_count: torch.Tensor,
    gate_count: torch.Tensor,
    front_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    small_mask: torch.Tensor,
    new_visible_mask: torch.Tensor,
    horizon_s: int,
    perturbation_id: str,
) -> torch.Tensor:
    student_occ = student_label != EMPTY_IDX
    candidate_mask = front_mask | student_occ | (contributor_count > 0) | (gate_count > 0)
    if horizon_s in {4, 6}:
        candidate_mask = candidate_mask | dynamic_mask | small_mask | new_visible_mask
    coords = torch.nonzero(candidate_mask, as_tuple=False)
    repaired = student_label.clone()
    if coords.shape[0] == 0:
        return repaired
    numeric = build_numeric_features(
        student_label,
        student_conf,
        student_margin,
        contributor_count,
        gate_count,
        coords,
        front_mask,
        dynamic_mask,
        small_mask,
        new_visible_mask,
        student_occ,
    )
    pred_class = student_label[coords[:, 0], coords[:, 1], coords[:, 2]].long()
    h_idx = torch.full((coords.shape[0],), horizon_index(horizon_s), dtype=torch.long)
    p_idx = torch.full((coords.shape[0],), perturb_index(perturbation_id), dtype=torch.long)
    chunk = 32768
    out_labels = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk):
            end = min(coords.shape[0], start + chunk)
            logits = repair_model(
                pred_class[start:end].cuda(non_blocking=False),
                h_idx[start:end].cuda(non_blocking=False),
                p_idx[start:end].cuda(non_blocking=False),
                numeric[start:end].cuda(non_blocking=False),
            )
            out_labels.append(logits.argmax(dim=-1).cpu())
    new_labels = torch.cat(out_labels, dim=0)
    repaired[coords[:, 0], coords[:, 1], coords[:, 2]] = new_labels
    return repaired


def run_phase_if_needed(
    progress: dict[str, Any],
    manifest: dict[str, Any],
    phase_name: str,
    fn,
    force_phase: str | None,
    stop_after_phase: str | None,
) -> Any:
    if phase_name in progress["completed_phases"] and force_phase is None:
        return None
    meta = phase_start(progress, manifest, phase_name)
    try:
        result = fn()
        phase_end(progress, manifest, phase_name, "done")
        if stop_after_phase == phase_name:
            progress["status"] = "stopped_after_phase"
            update_progress(progress)
            raise SystemExit(0)
        return result
    except Exception as exc:
        phase_end(progress, manifest, phase_name, "failed", failure_reason=str(exc))
        raise


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    progress = init_progress_state()
    manifest = init_execution_manifest()
    sectors = {name: tensor.cpu() for name, tensor in sw7.build_sector_masks().items()}
    train_ids = parse_int_list(args.train_samples)
    eval_ids = parse_int_list(args.eval_samples)
    checkpoints = discover_checkpoint_specs()

    def phase1() -> None:
        decision = read_json(SW12A_REPORTS / "sw12a_soft_neighbor_routing_decision.json")
        sweep = read_csv_rows(SW12A_REPORTS / "sw12a_variant_sweep_metrics.csv")
        pareto = read_csv_rows(SW12A_REPORTS / "sw12a_pareto_candidates.csv")
        survival = read_csv_rows(SW12A_REPORTS / "sw12a_survival_chain_before_after.csv")
        report = read_json(SW12A_REPORTS / "stage_sw12a_soft_neighbor_getocc_routing_report.json")
        digest = {
            "decision": decision,
            "best_real_candidate": decision["best_real_candidate"],
            "pareto_class_counts": dict(Counter(row["classification"] for row in pareto)),
            "survival_rows": survival,
            "report_headline": report["digest_headline"],
        }
        write_json(REPORTS_DIR / "sw12a_digest_for_sw12b.json", digest)
        write_md(
            REPORTS_DIR / "sw12b_objective.md",
            "\n".join(
                [
                    "1. SW-12A decision = C3_targeted_but_unsafe.",
                    "2. The best real candidate recovered part of A10/front/h6 false-free by moving native exact assignment and final contributor from 0 to non-zero.",
                    "3. It remained unsafe because A0 clean active voxels, A0 false-positive, C4 false-positive, pred_gt density, neighbor leakage, and wrong-class activation all increased sharply.",
                    "4. SW-12B therefore splits into Branch A risk-gated / class-aware guarded routing and Branch B clean-to-degraded consistency feasibility training.",
                    "5. Branch A continues the routing mechanism line.",
                    "6. Branch B seeks real trainable recovery under explicit clean drift / false-positive / density guards.",
                    "",
                ]
            ),
        )
        attach_artifact(manifest, "sw12a_digest_for_sw12b", REPORTS_DIR / "sw12a_digest_for_sw12b.json")

    def phase2() -> None:
        protocol = {
            "subset_name": "eval_core_20",
            "quick_subset": "quick_debug_5",
            "samples_eval_core_20": eval_ids,
            "samples_quick_debug_5": QUICK_DEBUG_IDS,
            "perturbations": CORE_PERTURBATIONS,
            "horizons": CORE_HORIZONS,
            "core_target_regions": [
                "A10 front-sector false-free",
                "A1 front-sector false-free",
                "small-object false-free",
                "new-visible false-free",
                "dynamic false-free",
                "h4/h6 future false-free",
                "C4 false-positive / density risk",
                "A0 clean control",
            ],
            "metrics": [
                "occupied_iou",
                "semantic_miou",
                "false_free_rate",
                "false_occupied_rate",
                "pred_gt_occupied_ratio",
                "small_object_false_free",
                "new_visible_recall",
                "front_sector_false_free",
                "dynamic_false_free",
                "static_false_free",
                "active_voxel_count_delta",
                "wrong_class_activation_delta",
                "A10_front_h6_recovery_ratio",
                "C4_false_positive_delta",
                "clean_occupied_iou_delta",
                "clean_false_positive_delta",
                "reliability_front_sector",
                "reliability_small_object",
                "reliability_new_visible",
                "risk_error_correlation",
            ],
            "safety_gates": {
                "clean_occupied_iou_delta_min": -0.005,
                "clean_semantic_miou_delta_min": -0.005,
                "A0_false_positive_delta_max": 0.005,
                "C4_false_positive_delta_max": 0.008,
                "pred_gt_density_delta_max": 0.05,
                "wrong_class_activation_delta_max": 0.005,
                "active_voxel_count_delta_max": 0.03,
            },
            "targeted_hit_rules": [
                "A10 front_sector_false_free_delta <= -0.03",
                "A10 small_object_false_free_delta <= -0.03",
                "A10 new_visible_recall_delta >= +0.03",
                "A10_front_h6_recovery_ratio > 0",
                "C4 false_positive_delta <= -0.005 without clean drift",
            ],
            "claim_boundary": ["subset diagnostic", "replay only", "smoke training", "short training", "not official benchmark"],
        }
        write_json(REPORTS_DIR / "sw12b_eval_protocol.json", protocol)
        write_md(REPORTS_DIR / "sw12b_eval_protocol.md", json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
        attach_artifact(manifest, "sw12b_eval_protocol", REPORTS_DIR / "sw12b_eval_protocol.json")

    def phaseA1() -> None:
        variants = build_branch_a_variants()
        write_json(
            REPORTS_DIR / "branchA_routing_variant_manifest.json",
            {"variants": [normalize_export(v.__dict__) for v in variants], "native_default_unchanged": True},
        )
        write_md(
            REPORTS_DIR / "branchA_routing_code_diff_summary.md",
            "\n".join(
                [
                    "- extended flag-controlled get_occ variant routing with risk-target-only gating",
                    "- added class margin gate, allowed/restricted class ids, and risk hit filtering",
                    "- native get_occ default path remains unchanged",
                    "- Branch A continues replay only; no routing training in SW-12B",
                    "",
                ]
            ),
        )

    def phaseA2A3() -> None:
        variants = build_branch_a_variants()
        quick_metrics_path = REPORTS_DIR / "branchA_routing_quick_metrics.csv"
        quick_agg_path = REPORTS_DIR / "branchA_routing_quick_aggregate_metrics.csv"
        quick_candidates_path = REPORTS_DIR / "branchA_routing_quick_candidates.csv"
        selected_variants_path = REPORTS_DIR / "branchA_selected_variants_by_checkpoint.json"
        replay_manifest_path = REPORTS_DIR / "branchA_routing_replay_manifest.csv"

        quick_rows: list[dict[str, Any]]
        replay_manifest: list[dict[str, Any]]
        selected_variants_by_checkpoint: dict[str, list[str]]
        checkpoint_allow_full: dict[str, bool] = {}

        if quick_metrics_path.exists() and quick_agg_path.exists() and quick_candidates_path.exists() and selected_variants_path.exists():
            quick_rows = read_csv_rows(quick_metrics_path)
            quick_agg = read_csv_rows(quick_agg_path)
            candidate_rows = read_csv_rows(quick_candidates_path)
            selected_variants_by_checkpoint = {
                str(k): list(dict.fromkeys(str(x) for x in v))
                for k, v in read_json(selected_variants_path).get("selected_variants_by_checkpoint", {}).items()
            }
            replay_manifest = read_csv_rows(replay_manifest_path) if replay_manifest_path.exists() else []
        else:
            quick_rows = []
            replay_manifest = []
            selected_variants_by_checkpoint = {}
            for checkpoint_spec in checkpoints[:3]:
                cfg, dataset, model = build_runtime(checkpoint_spec.config_path, checkpoint_spec.checkpoint_path)
                head = sw4_inst.get_pts_bbox_head(model)
                try:
                    for perturbation_id in CORE_PERTURBATIONS:
                        for sample_index in QUICK_DEBUG_IDS:
                            sample_unwrapped, per_h = load_case(model, dataset, sample_index, perturbation_id, CORE_HORIZONS)
                            for horizon_s in CORE_HORIZONS:
                                case = per_h[horizon_s]
                                rows, dump_payload = branch_a_replay_case(
                                    checkpoint_spec,
                                    sample_index,
                                    perturbation_id,
                                    horizon_s,
                                    head,
                                    case["pred_dict"],
                                    case["gt_h"],
                                    case["gt0"],
                                    sectors,
                                    variants,
                                )
                                for row in rows:
                                    row["phase"] = "quick_debug_5"
                                quick_rows.extend(rows)
                                dump_dir = ARTIFACTS_DIR / "branchA_replay_dumps" / checkpoint_spec.name / perturbation_id
                                dump_dir.mkdir(parents=True, exist_ok=True)
                                dump_path = dump_dir / f"sample_{sample_index}_h{horizon_s}.npz"
                                np.savez_compressed(dump_path, **dump_payload)
                                replay_manifest.append(
                                    {
                                        "phase": "quick_debug_5",
                                        "checkpoint_name": checkpoint_spec.name,
                                        "sample_index": sample_index,
                                        "perturbation_id": perturbation_id,
                                        "horizon_s": horizon_s,
                                        "dump_path": str(dump_path),
                                    }
                                )
                finally:
                    del model, dataset, cfg
                    safe_cuda_cleanup()
            quick_agg = aggregate_rows(
                quick_rows,
                ["checkpoint_name", "variant_label", "variant_name", "is_oracle", "perturbation_id", "horizon_s"],
            )
            candidate_rows = branch_a_select_candidates(quick_agg)
            top_real_by_checkpoint: dict[str, list[str]] = defaultdict(list)
            for checkpoint_name in {row["checkpoint_name"] for row in candidate_rows}:
                real = [row for row in candidate_rows if row["checkpoint_name"] == checkpoint_name and row["variant_label"] != "A0_native" and not row["is_oracle"]]
                if real:
                    real = sorted(real, key=lambda x: (x["classification"] == "A_SAFE_TARGETED", float(x["target_recovery"])), reverse=True)
                    top_real_by_checkpoint[checkpoint_name] = [x["variant_label"] for x in real[:2]]
                else:
                    top_real_by_checkpoint[checkpoint_name] = []
                selected_variants_by_checkpoint[checkpoint_name] = list(
                    dict.fromkeys(["A0_native"] + top_real_by_checkpoint[checkpoint_name] + ["A6_oracle_upper_bound"])
                )
            write_csv(quick_metrics_path, quick_rows)
            write_csv(quick_agg_path, quick_agg)
            write_csv(quick_candidates_path, candidate_rows)
            write_json(selected_variants_path, {"selected_variants_by_checkpoint": selected_variants_by_checkpoint})
            write_csv(replay_manifest_path, replay_manifest)

        for checkpoint_name, labels in selected_variants_by_checkpoint.items():
            quick_real = [
                row for row in candidate_rows
                if row["checkpoint_name"] == checkpoint_name and row["variant_label"] in labels and row["variant_label"] != "A0_native" and not truthy(row["is_oracle"])
            ]
            checkpoint_allow_full[checkpoint_name] = any(
                row["classification"] == "A_SAFE_TARGETED"
                or (
                    truthy(row["targeted_hit"])
                    and float(row["clean_false_positive_delta"]) <= 0.01
                    and float(row["pred_gt_density_delta"]) <= 0.08
                    and float(row["active_voxel_count_delta"]) <= 0.05
                )
                for row in quick_real
            )

        full_rows: list[dict[str, Any]] = []
        variant_lookup = {v.label: v for v in variants}
        if any(checkpoint_allow_full.values()):
            for checkpoint_spec in checkpoints[:3]:
                if not checkpoint_allow_full.get(checkpoint_spec.name, False):
                    continue
                cfg, dataset, model = build_runtime(checkpoint_spec.config_path, checkpoint_spec.checkpoint_path)
                head = sw4_inst.get_pts_bbox_head(model)
                try:
                    chosen_labels = list(dict.fromkeys(selected_variants_by_checkpoint[checkpoint_spec.name]))
                    chosen = [variant_lookup[label] for label in chosen_labels]
                    for perturbation_id in CORE_PERTURBATIONS:
                        for sample_index in eval_ids:
                            sample_unwrapped, per_h = load_case(model, dataset, sample_index, perturbation_id, CORE_HORIZONS)
                            for horizon_s in CORE_HORIZONS:
                                case = per_h[horizon_s]
                                rows, _ = branch_a_replay_case(
                                    checkpoint_spec,
                                    sample_index,
                                    perturbation_id,
                                    horizon_s,
                                    head,
                                    case["pred_dict"],
                                    case["gt_h"],
                                    case["gt0"],
                                    sectors,
                                    chosen,
                                )
                                for row in rows:
                                    row["phase"] = "eval_core_20"
                                full_rows.extend(rows)
                finally:
                    del model, dataset, cfg
                    safe_cuda_cleanup()
        all_rows = quick_rows + full_rows
        aggregate_rows_final = aggregate_rows(
            all_rows,
            ["phase", "checkpoint_name", "variant_label", "variant_name", "is_oracle", "perturbation_id", "horizon_s"],
        )
        final_candidate_source = [
            row for row in aggregate_rows_final
            if row["phase"] == "eval_core_20" and row["perturbation_id"] in {"A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"} and int(row["horizon_s"]) in {0, 6}
        ]
        if not final_candidate_source:
            final_candidate_source = [
                row for row in quick_agg
                if row["perturbation_id"] in {"A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"} and int(row["horizon_s"]) in {0, 6}
            ]
        candidate_rows_final = branch_a_select_candidates(final_candidate_source)
        write_csv(REPORTS_DIR / "branchA_routing_replay_manifest.csv", replay_manifest)
        write_csv(REPORTS_DIR / "branchA_routing_replay_metrics.csv", all_rows)
        write_csv(REPORTS_DIR / "branchA_routing_aggregate_metrics.csv", aggregate_rows_final)
        write_csv(REPORTS_DIR / "branchA_routing_candidates.csv", candidate_rows_final)
        write_md(
            REPORTS_DIR / "branchA_routing_candidate_summary.md",
            "\n".join(
                [f"- {row['checkpoint_name']} / {row['variant_label']}: {row['classification']}" for row in candidate_rows_final]
                + [f"- full_eval_enabled_checkpoints: {json.dumps(checkpoint_allow_full, ensure_ascii=False)}"]
            ) + "\n",
        )
        # figures
        fig_rows = [row for row in candidate_rows_final if row["variant_label"] != "A6_oracle_upper_bound"]
        if fig_rows:
            fig, ax = plt.subplots(figsize=(8.6, 5.0))
            xs = [float(row["pred_gt_density_delta"]) for row in fig_rows]
            ys = [float(row["target_recovery"]) for row in fig_rows]
            cs = ["#117A65" if row["classification"] == "A_SAFE_TARGETED" else "#CA6F1E" if row["classification"] == "A_TARGETED_UNSAFE" else "#5D6D7E" for row in fig_rows]
            ax.scatter(xs, ys, c=cs)
            for row, x, y in zip(fig_rows, xs, ys):
                ax.text(x, y, row["variant_label"], fontsize=7)
            ax.set_title("SW-12B Branch A subset diagnostic replay tradeoff curve")
            ax.set_xlabel("pred_gt_density_delta")
            ax.set_ylabel("target_recovery")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "branchA_routing_tradeoff_curve.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.6, 5.0))
            xs = [float(row["clean_false_positive_delta"]) for row in fig_rows]
            ys = [float(row["target_recovery"]) for row in fig_rows]
            cs = ["#117A65" if row["safe_gate"] else "#922B21" for row in fig_rows]
            ax.scatter(xs, ys, c=cs)
            for row, x, y in zip(fig_rows, xs, ys):
                ax.text(x, y, row["variant_label"], fontsize=7)
            ax.set_title("SW-12B Branch A subset diagnostic replay Pareto scatter")
            ax.set_xlabel("clean_false_positive_delta")
            ax.set_ylabel("target_recovery")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "branchA_pareto_scatter.png", dpi=180)
            plt.close(fig)

    def phaseB1() -> None:
        teacher_rows: list[dict[str, Any]] = []
        checkpoint = checkpoints[0]
        cfg, dataset, model = build_runtime(checkpoint.config_path, checkpoint.checkpoint_path)
        head = sw4_inst.get_pts_bbox_head(model)
        try:
            for split_name, sample_ids in [("train", train_ids), ("eval", eval_ids)]:
                for sample_index in sample_ids:
                    sample_unwrapped, per_h = load_case(model, dataset, sample_index, "A0_clean", CORE_HORIZONS)
                    for horizon_s in CORE_HORIZONS:
                        case = per_h[horizon_s]
                        pred_dbg, dbg_list = sw4_inst.get_occ_debug(head, case["pred_dict"], capture_dense=True)
                        pred = pred_dbg[0].detach().cpu().long()
                        dbg = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in dbg_list[0].items()}
                        top1_conf, top1_cls, top1_margin = derive_top_conf_and_margin(dbg["dense_occ_after_padding"])
                        groups = class_group_masks(case["gt_h"], case["gt0"])
                        out = {
                            "teacher_label": pred.numpy().astype(np.uint8),
                            "teacher_conf": top1_conf.numpy().astype(np.float16),
                            "teacher_cls": top1_cls.numpy().astype(np.uint8),
                            "teacher_margin": top1_margin.numpy().astype(np.float16),
                            "gt_label": case["gt_h"].numpy().astype(np.uint8),
                            "front_mask": sectors["front"].numpy().astype(np.uint8),
                            "small_mask": groups["small_object"].numpy().astype(np.uint8),
                            "dynamic_mask": groups["dynamic"].numpy().astype(np.uint8),
                            "new_visible_mask": groups["new_visible"].numpy().astype(np.uint8),
                            "valid_mask": np.ones_like(case["gt_h"].numpy(), dtype=np.uint8),
                            "teacher_contributor_count": tensor_to_numpy(torch.as_tensor(dbg["contributor_count_dense"]), dtype=np.int16),
                        }
                        cache_path = ARTIFACTS_DIR / "teacher_cache" / f"{split_name}__sample{sample_index:03d}__h{horizon_s}.npz"
                        np.savez_compressed(cache_path, **out)
                        teacher_rows.append(
                            {
                                "split": split_name,
                                "sample_index": sample_index,
                                "horizon_s": horizon_s,
                                "cache_path": str(cache_path),
                                "teacher_conf_thr_05_count": int((top1_conf >= 0.5).sum().item()),
                                "teacher_conf_thr_07_count": int((top1_conf >= 0.7).sum().item()),
                                "teacher_conf_thr_09_count": int((top1_conf >= 0.9).sum().item()),
                                "teacher_occupied_count": int((pred != EMPTY_IDX).sum().item()),
                            }
                        )
        finally:
            del model, dataset, cfg
            safe_cuda_cleanup()
        write_csv(REPORTS_DIR / "branchB_teacher_cache_manifest.csv", teacher_rows)
        write_md(REPORTS_DIR / "branchB_teacher_cache_summary.md", f"generated {len(teacher_rows)} teacher cache NPZ files for subset diagnostic training/eval splits\n")

    def phaseB2() -> None:
        checkpoint = checkpoints[0]
        teacher_manifest = read_csv_rows(REPORTS_DIR / "branchB_teacher_cache_manifest.csv")
        teacher_lookup = {(row["split"], int(row["sample_index"]), int(row["horizon_s"])): Path(row["cache_path"]) for row in teacher_manifest}
        pair_manifest: list[dict[str, Any]] = []
        pair_dist_rows: list[dict[str, Any]] = []
        cfg, dataset, model = build_runtime(checkpoint.config_path, checkpoint.checkpoint_path)
        head = sw4_inst.get_pts_bbox_head(model)
        try:
            for split_name, sample_ids in [("train", train_ids), ("eval", eval_ids)]:
                for perturbation_id in PAIR_PERTURBATIONS:
                    for sample_index in sample_ids:
                        sample_unwrapped, per_h = load_case(model, dataset, sample_index, perturbation_id, CORE_HORIZONS)
                        for horizon_s in CORE_HORIZONS:
                            case = per_h[horizon_s]
                            teacher_npz = np.load(teacher_lookup[(split_name, sample_index, horizon_s)])
                            pred_dbg, dbg_list = sw4_inst.get_occ_debug(head, case["pred_dict"], capture_dense=True)
                            pred = pred_dbg[0].detach().cpu().long()
                            dbg = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in dbg_list[0].items()}
                            student_conf, student_cls, student_margin = derive_top_conf_and_margin(dbg["dense_occ_after_padding"])
                            pair = make_pair_npz(
                                torch.from_numpy(teacher_npz["teacher_label"]).long(),
                                torch.from_numpy(teacher_npz["teacher_conf"]).float(),
                                case["gt_h"],
                                pred,
                                student_conf,
                                student_margin,
                                torch.as_tensor(dbg["contributor_count_dense"]).cpu(),
                                gate_pass_dense_or_zero(dbg),
                                torch.from_numpy(teacher_npz["front_mask"]).bool(),
                                torch.from_numpy(teacher_npz["dynamic_mask"]).bool(),
                                torch.from_numpy(teacher_npz["small_mask"]).bool(),
                                torch.from_numpy(teacher_npz["new_visible_mask"]).bool(),
                                perturbation_id,
                                horizon_s,
                            )
                            pair_path = ARTIFACTS_DIR / "student_teacher_pairs" / f"{split_name}__sample{sample_index:03d}__{perturbation_id}__h{horizon_s}.npz"
                            np.savez_compressed(pair_path, **pair)
                            pair_manifest.append(
                                {
                                    "split": split_name,
                                    "sample_index": sample_index,
                                    "perturbation_id": perturbation_id,
                                    "horizon_s": horizon_s,
                                    "pair_path": str(pair_path),
                                    "selected_voxel_count": int(pair["coords"].shape[0]),
                                    "positive_count": int(pair["pos_count"][0]),
                                    "negative_count": int(pair["neg_count"][0]),
                                }
                            )
                            pair_dist_rows.append(
                                {
                                    "split": split_name,
                                    "perturbation_id": perturbation_id,
                                    "horizon_s": horizon_s,
                                    "selected_voxel_count": int(pair["coords"].shape[0]),
                                    "positive_count": int(pair["pos_count"][0]),
                                    "negative_count": int(pair["neg_count"][0]),
                                }
                            )
        finally:
            del model, dataset, cfg
            safe_cuda_cleanup()
        write_csv(REPORTS_DIR / "branchB_pair_builder_manifest.csv", pair_manifest)
        write_csv(REPORTS_DIR / "branchB_pair_distribution.csv", aggregate_rows(pair_dist_rows, ["split", "perturbation_id", "horizon_s"]))

    def phaseB3() -> None:
        write_md(
            REPORTS_DIR / "branchB_consistency_loss_design.md",
            "\n".join(
                [
                    "- L_occ_consistency: BCE on occupied vs empty from teacher reliable voxels",
                    "- L_semantic_consistency: CE to teacher class label on selected voxels",
                    "- L_risk_weighted_recovery: extra weight on front / small / new-visible / h4-h6",
                    "- L_false_positive_guard: penalize occupied activation where teacher says reliable free",
                    "- L_density_regularizer: penalize occupied density drift relative to teacher target",
                    "",
                ]
            ),
        )
        write_json(REPORTS_DIR / "branchB_config_manifest.json", {"configs": [normalize_export(cfg.__dict__) for cfg in branch_b_configs()]})

    def phaseB4() -> None:
        pair_paths = sorted((ARTIFACTS_DIR / "student_teacher_pairs").glob("train__*.npz"))
        grad_rows: list[dict[str, Any]] = []
        smoke_rows: list[dict[str, Any]] = []
        for cfg in [x for x in branch_b_configs() if x.name in {"B1_occ_consistency_light", "B2_occ_sem_consistency", "B3_risk_weighted_consistency", "B4_consistency_with_fp_guard"}]:
            dataset = load_pair_dataset(pair_paths, cfg.perturbations)
            model = OccupancyRepairMLP().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            n = dataset["numeric_feats"].shape[0]
            batch_n = min(4096, max(1, n))
            idx = torch.randint(0, n, (batch_n,)) if n > 0 else torch.zeros((0,), dtype=torch.long)
            batch = {k: v[idx] if v.shape[0] == n else v for k, v in dataset.items()}
            if n > 0:
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    batch["student_pred_class"].cuda(non_blocking=False),
                    batch["horizon_index"].cuda(non_blocking=False),
                    batch["perturb_index"].cuda(non_blocking=False),
                    batch["numeric_feats"].cuda(non_blocking=False),
                )
                losses = compute_branch_b_losses(logits, batch, cfg)
                losses["total_loss"].backward()
                grad_sq = 0.0
                grad_finite = True
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    grad_sq += float(param.grad.detach().float().norm().item() ** 2)
                    grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
                optimizer.step()
            else:
                losses = {
                    "total_loss": torch.tensor(0.0),
                    "original_loss": torch.tensor(0.0),
                    "consistency_loss": torch.tensor(0.0),
                    "semantic_consistency_loss": torch.tensor(0.0),
                    "fp_guard_loss": torch.tensor(0.0),
                    "density_loss": torch.tensor(0.0),
                    "pred_density_proxy": torch.tensor(0.0),
                }
                grad_sq = 0.0
                grad_finite = False
            grad_rows.append(
                {
                    "config_name": cfg.name,
                    "total_loss": float(losses["total_loss"].detach().cpu().item()),
                    "original_loss": float(losses["original_loss"].detach().cpu().item()),
                    "consistency_loss": float(losses["consistency_loss"].detach().cpu().item()),
                    "semantic_consistency_loss": float(losses["semantic_consistency_loss"].detach().cpu().item()),
                    "fp_guard_loss": float(losses["fp_guard_loss"].detach().cpu().item()),
                    "density_loss": float(losses["density_loss"].detach().cpu().item()),
                    "grad_norm": math.sqrt(max(0.0, grad_sq)),
                    "grad_finite": grad_finite,
                    "has_nan": not grad_finite,
                    "has_inf": not grad_finite,
                    "memory_mb": 0.0,
                    "teacher_voxel_count": int((dataset["teacher_target_occ"] > 0.5).sum().item()),
                    "recovery_target_count": int((dataset["teacher_target_occ"] > 0.5).sum().item()),
                    "negative_guard_count": int((dataset["teacher_target_occ"] < 0.5).sum().item()),
                    "pred_density_proxy": float(losses["pred_density_proxy"].detach().cpu().item()),
                }
            )
            trained_model, smoke_cfg_rows, last_ckpt = train_repair_model(dataset, cfg, 20, f"branchB_{cfg.name}", "smoke")
            smoke_rows.extend(smoke_cfg_rows)
            del trained_model
            safe_cuda_cleanup()
        write_csv(REPORTS_DIR / "branchB_gradient_check.csv", grad_rows)
        write_csv(REPORTS_DIR / "branchB_20iter_smoke_metrics.csv", smoke_rows)
        if smoke_rows:
            fig, ax = plt.subplots(figsize=(8.6, 5.0))
            for name in sorted({row["config_name"] for row in smoke_rows}):
                rows = [row for row in smoke_rows if row["config_name"] == name]
                ax.plot([int(r["iter"]) for r in rows], [float(r["total_loss"]) for r in rows], label=name)
            ax.set_title("SW-12B Branch B subset diagnostic smoke loss curves")
            ax.set_xlabel("iter")
            ax.set_ylabel("total_loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "branchB_loss_smoke_curves.png", dpi=180)
            plt.close(fig)

    def phaseB5B6CDE() -> None:
        pair_paths = sorted((ARTIFACTS_DIR / "student_teacher_pairs").glob("train__*.npz"))
        grad_rows = read_csv_rows(REPORTS_DIR / "branchB_gradient_check.csv")
        smoke_rows = read_csv_rows(REPORTS_DIR / "branchB_20iter_smoke_metrics.csv")
        short_rows: list[dict[str, Any]] = []
        completed_branchb_checkpoints: list[dict[str, Any]] = []
        promising_cfgs = []
        for cfg in [x for x in branch_b_configs() if x.name in {"B4_consistency_with_fp_guard", "B3_risk_weighted_consistency", "B2_occ_sem_consistency"}]:
            smoke_cfg_rows = [row for row in smoke_rows if row["config_name"] == cfg.name]
            if not smoke_cfg_rows:
                continue
            loss_head = np.mean([float(row["total_loss"]) for row in smoke_cfg_rows[:5]])
            loss_tail = np.mean([float(row["total_loss"]) for row in smoke_cfg_rows[-5:]])
            density_peak = max(float(row["pred_density_proxy"]) for row in smoke_cfg_rows)
            grad_ok = any(row["config_name"] == cfg.name and row["grad_finite"] in {True, "True", "true"} for row in grad_rows)
            if grad_ok and loss_tail <= loss_head and density_peak <= 0.85:
                promising_cfgs.append(cfg)
        for cfg in promising_cfgs:
            dataset = load_pair_dataset(pair_paths, cfg.perturbations)
            model, rows500, ckpt = train_repair_model(dataset, cfg, 500, f"branchB_{cfg.name}", "short_train")
            short_rows.extend(rows500)
            completed_branchb_checkpoints.append({"config_name": cfg.name, "checkpoint_path": str(ckpt)})
            del model
            safe_cuda_cleanup()
        if short_rows:
            write_csv(REPORTS_DIR / "branchB_short_train_metrics.csv", short_rows)

        eval_rows: list[dict[str, Any]] = []
        brancha_candidates = read_csv_rows(REPORTS_DIR / "branchA_routing_candidates.csv")
        best_brancha = next(
            (
                row
                for row in brancha_candidates
                if row["classification"] == "A_SAFE_TARGETED" and row["variant_label"] != "A0_native"
            ),
            None,
        )
        checkpoint = checkpoints[0]
        cfg, dataset, model = build_runtime(checkpoint.config_path, checkpoint.checkpoint_path)
        head = sw4_inst.get_pts_bbox_head(model)
        try:
            repair_models = {item["config_name"]: load_repair_model(Path(item["checkpoint_path"])) for item in completed_branchb_checkpoints}
            for sample_index in eval_ids:
                for perturbation_id in CORE_PERTURBATIONS:
                    sample_unwrapped, per_h = load_case(model, dataset, sample_index, perturbation_id, CORE_HORIZONS)
                    for horizon_s in CORE_HORIZONS:
                        case = per_h[horizon_s]
                        pred_dbg, dbg_list = sw4_inst.get_occ_debug(head, case["pred_dict"], capture_dense=True)
                        base_pred = pred_dbg[0].detach().cpu().long()
                        dbg = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in dbg_list[0].items()}
                        student_conf, student_cls, student_margin = derive_top_conf_and_margin(dbg["dense_occ_after_padding"])
                        groups = class_group_masks(case["gt_h"], case["gt0"])
                        eval_rows.append(
                            {
                                "model_name": "epoch_56_baseline",
                                "sample_index": sample_index,
                                "perturbation_id": perturbation_id,
                                "horizon_s": horizon_s,
                                **build_eval_row(base_pred, case["gt_h"], case["gt0"], perturbation_id, horizon_s, sectors, baseline_pred=base_pred if perturbation_id == "A0_clean" else None),
                            }
                        )
                        for config_name, repair_model in repair_models.items():
                            repaired = repair_prediction_with_model(
                                repair_model,
                                student_cls,
                                student_conf,
                                student_margin,
                                torch.as_tensor(dbg["contributor_count_dense"]).cpu(),
                                gate_pass_dense_or_zero(dbg),
                                sectors["front"],
                                groups["dynamic"],
                                groups["small_object"],
                                groups["new_visible"],
                                horizon_s,
                                perturbation_id,
                            )
                            eval_rows.append(
                                {
                                    "model_name": config_name,
                                    "sample_index": sample_index,
                                    "perturbation_id": perturbation_id,
                                    "horizon_s": horizon_s,
                                    **build_eval_row(repaired, case["gt_h"], case["gt0"], perturbation_id, horizon_s, sectors, baseline_pred=base_pred),
                                }
                            )
        finally:
            del model, dataset, cfg
            safe_cuda_cleanup()
        if eval_rows:
            write_csv(REPORTS_DIR / "branchB_fixed_subset_eval.csv", eval_rows)
            eval_summary = aggregate_rows(eval_rows, ["model_name", "perturbation_id", "horizon_s"])
            write_md(REPORTS_DIR / "branchB_fixed_subset_eval_summary.md", f"aggregated {len(eval_summary)} subset diagnostic eval rows\n")
            fig_rows = [row for row in eval_summary if row["perturbation_id"] == "A10_drop_front_triplet" and int(row["horizon_s"]) == 6]
            if fig_rows:
                fig, ax = plt.subplots(figsize=(8.6, 5.0))
                xs = np.arange(len(fig_rows))
                ax.bar(xs - 0.15, [float(r["front_sector_false_free"]) for r in fig_rows], width=0.3, label="front_sector_false_free")
                ax.bar(xs + 0.15, [float(r["false_occupied_rate"]) for r in fig_rows], width=0.3, label="false_occupied_rate")
                ax.set_xticks(xs)
                ax.set_xticklabels([r["model_name"] for r in fig_rows], rotation=20)
                ax.set_title("SW-12B Branch B subset diagnostic metric deltas")
                ax.legend(fontsize=7)
                ax.grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / "branchB_metric_delta_bar.png", dpi=180)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(8.6, 5.0))
                xs2 = [float(r["false_occupied_rate"]) for r in fig_rows]
                ys2 = [1.0 - float(r["front_sector_false_free"]) for r in fig_rows]
                ax.scatter(xs2, ys2)
                for row, x, y in zip(fig_rows, xs2, ys2):
                    ax.text(x, y, row["model_name"], fontsize=7)
                ax.set_title("SW-12B Branch B subset diagnostic false-free / false-positive tradeoff")
                ax.set_xlabel("false_occupied_rate")
                ax.set_ylabel("front_sector_recall_proxy")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / "branchB_falsefree_falsepositive_tradeoff.png", dpi=180)
                plt.close(fig)

        brancha_candidates = read_csv_rows(REPORTS_DIR / "branchA_routing_candidates.csv")
        brancha_safe = [
            row
            for row in brancha_candidates
            if row["classification"] == "A_SAFE_TARGETED" and row["variant_label"] != "A0_native"
        ]
        brancha_targeted_unsafe = [
            row
            for row in brancha_candidates
            if row["classification"] == "A_TARGETED_UNSAFE" and row["variant_label"] != "A0_native"
        ]
        eval_summary_rows = aggregate_rows(read_csv_rows(REPORTS_DIR / "branchB_fixed_subset_eval.csv"), ["model_name", "perturbation_id", "horizon_s"]) if (REPORTS_DIR / "branchB_fixed_subset_eval.csv").exists() else []
        branchb_promising = []
        for row in eval_summary_rows:
            if row["model_name"] == "epoch_56_baseline":
                continue
            if row["perturbation_id"] != "A10_drop_front_triplet" or int(row["horizon_s"]) != 6:
                continue
            safe_gate = (
                float(row["clean_occupied_iou_delta"]) >= -0.005
                and float(row["clean_semantic_miou_delta"]) >= -0.005
                and float(row["clean_false_positive_delta"]) <= 0.005
                and float(row["C4_false_positive_delta"]) <= 0.008
                and float(row["pred_gt_density_delta"]) <= 0.05
                and float(row["wrong_class_activation_delta"]) <= 0.005
                and float(row["active_voxel_count_delta"]) <= 0.03
            )
            targeted_hit = float(row["A10_front_h6_recovery_ratio"]) > 0.0 or float(row["new_visible_recall"]) > 0.03 or float(row["small_object_false_free"]) < 0.0
            if safe_gate and targeted_hit:
                branchb_promising.append(row)

        comparison_rows = [
            {
                "branch": "A",
                "status": "safe_targeted" if brancha_safe else "targeted_unsafe" if brancha_targeted_unsafe else "no_signal",
                "best_candidate": brancha_safe[0]["variant_label"] if brancha_safe else brancha_targeted_unsafe[0]["variant_label"] if brancha_targeted_unsafe else None,
            },
            {
                "branch": "B",
                "status": "promising" if branchb_promising else "no_signal",
                "best_candidate": branchb_promising[0]["model_name"] if branchb_promising else None,
            },
        ]
        write_csv(REPORTS_DIR / "sw12b_branch_comparison.csv", comparison_rows)

        if branchb_promising and not brancha_safe:
            decision_type = "D1_branchB_promising_mainline" if not brancha_targeted_unsafe else "D3_branchA_targeted_but_unsafe_branchB_promising"
            next_route = "scale_branchB_consistency_training"
            summary = "Branch B produced the stronger safe targeted subset signal while Branch A remained unsafe or not ready."
        elif brancha_safe and not branchb_promising:
            decision_type = "D2_branchA_safe_candidate"
            next_route = "train_and_eval_branchA_safe_candidate"
            summary = "Branch A found a safe replay candidate while Branch B did not yet show a stronger safe subset gain."
        elif brancha_safe and branchb_promising:
            decision_type = "D4_both_promising"
            next_route = "combine_risk_gated_routing_with_consistency_training"
            summary = "Both Branch A and Branch B produced promising subset diagnostic signals."
        elif short_rows and any(float(row["pred_density_proxy"]) > 0.9 for row in short_rows):
            decision_type = "D6_branchB_overfits_or_density_explodes"
            next_route = "strengthen_fp_guard_and_teacher_filter"
            summary = "Branch B reduced some false-free targets but density / false-positive behavior became unstable."
        elif not read_csv_rows(REPORTS_DIR / "branchB_teacher_cache_manifest.csv"):
            decision_type = "D7_teacher_quality_blocked"
            next_route = "tighten_teacher_reliability_and_gt_filtering"
            summary = "Teacher cache or teacher-student pair quality was not reliable enough for a clean Branch B conclusion."
        else:
            decision_type = "D5_both_no_signal"
            next_route = "stop_architecture_tinkering_or_move_to_larger_redesign"
            summary = "Neither Branch A nor Branch B produced a safe targeted subset candidate."

        decision_payload = {
            "decision_type": decision_type,
            "summary": summary,
            "branchA_safe_candidates": brancha_safe,
            "branchA_targeted_unsafe": brancha_targeted_unsafe[:3],
            "branchB_promising": branchb_promising[:3],
            "next_unique_action": next_route,
        }
        write_json(REPORTS_DIR / "sw12b_risk_gated_occupancy_repair_decision.json", decision_payload)
        write_md(REPORTS_DIR / "sw12b_risk_gated_occupancy_repair_decision.md", json.dumps(normalize_export(decision_payload), indent=2, ensure_ascii=False) + "\n")

        # figures and report
        fig, ax = plt.subplots(figsize=(10, 4.0))
        ax.axis("off")
        ax.text(0.5, 0.6, "SW-12B subset diagnostic pipeline", ha="center", va="center", fontsize=18)
        ax.text(0.5, 0.35, "Branch A replay + Branch B teacher-student smoke / short training", ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sw12b_overall_pipeline.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.0))
        ax.axis("off")
        ax.text(0.5, 0.6, "Clean teacher -> degraded student pairs -> repair head training", ha="center", va="center", fontsize=16)
        ax.set_title("SW-12B Branch B subset diagnostic teacher-student pipeline")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "branchB_teacher_student_pipeline.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        labels = [row["branch"] for row in comparison_rows]
        vals = [1.0 if row["status"] in {"safe_targeted", "promising"} else 0.5 if "unsafe" in row["status"] else 0.0 for row in comparison_rows]
        ax.bar(labels, vals, color=["#117A65", "#CA6F1E"])
        ax.set_title("SW-12B subset diagnostic branch comparison matrix")
        ax.set_ylim(0.0, 1.1)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "branch_comparison_matrix.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.axis("off")
        ax.text(0.5, 0.5, decision_type, ha="center", va="center", fontsize=18)
        ax.set_title("SW-12B subset diagnostic decision flow")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sw12b_decision_flow.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.axis("off")
        ax.text(0.5, 0.6, "subset diagnostic qualitative placeholder", ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.4, "clean vs degraded vs teacher vs student repair", ha="center", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "qualitative_bev_clean_degraded_teacher_student_sample0.png", dpi=180)
        plt.close(fig)

        report_json = {
            "executive_summary": "SW-12B executed Branch A replay and Branch B teacher-student feasibility training under explicit clean drift / false-positive / density guards.",
            "sw12a_digest_headline": "SW-12A showed that soft neighbor routing can recover contributor signal but remained targeted-but-unsafe.",
            "unified_eval_protocol": read_json(REPORTS_DIR / "sw12b_eval_protocol.json"),
            "branchA_status": "replay only",
            "branchB_status": "feasibility training",
            "branch_comparison": comparison_rows,
            "decision": decision_payload,
        }
        report_md = "\n".join(
            [
                "# Stage SW-12B Risk-gated Occupancy Repair Feasibility",
                "",
                "1. Executive summary",
                f"- {report_json['executive_summary']}",
                "",
                "2. Why SW-12B follows SW-12A",
                "- SW-12B follows SW-12A because SW-12A proved routing signal exists but is unsafe without stronger guards.",
                "",
                "3. SW-12A evidence digest",
                f"- {report_json['sw12a_digest_headline']}",
                "",
                "4. Unified eval protocol",
                "- Branch A and Branch B share the same subset diagnostic protocol over A0/A1/A10/C4/A7 and h0/h2/h4/h6.",
                "",
                "5. Branch A: risk-gated / class-aware routing replay",
                "- Branch A is replay only unless explicitly stated otherwise.",
                "",
                "6. Branch A results and safety analysis",
                "- Branch A routing candidates are reported with false-positive / density / clean drift tradeoffs.",
                "",
                "7. Branch B: clean-to-degraded teacher consistency design",
                "- Branch B is feasibility training, not an official benchmark.",
                "",
                "8. Teacher cache and pair builder",
                "- teacher prediction is not GT; teacher reliability filters are reported separately.",
                "",
                "9. Branch B gradient check and smoke",
                "- 20-iter smoke is a mechanism test, not a final performance claim.",
                "",
                "10. Branch B short training and fixed subset eval",
                "- any reported gains are subset diagnostic only and must be read together with false-positive / density / clean drift.",
                "",
                "11. Branch comparison",
                f"- {json.dumps(comparison_rows, ensure_ascii=False)}",
                "",
                "12. Decision D1-D7",
                f"- {decision_payload['decision_type']}: {decision_payload['summary']}",
                "",
                "13. Safe claims",
                "- Branch A is replay only unless otherwise stated",
                "- Branch B is feasibility training, not official benchmark",
                "- no claim of surpassing prior paper",
                "- no production claim",
                "- no calibrated uncertainty claim",
                "- all improvements are subset diagnostic only",
                "- teacher prediction is not GT",
                "- oracle or teacher-derived targets must not be overclaimed",
                "",
                "14. Limitations",
                "- Branch A remains a replay-only routing study in this stage",
                "- Branch B trains an occupancy repair feasibility head under subset supervision rather than claiming a final model-level gain",
                "",
                "15. Next unique action",
                f"- {decision_payload['next_unique_action']}",
                "",
            ]
        )
        write_md(REPORTS_DIR / "stage_sw12b_risk_gated_occupancy_repair_report.md", report_md)
        write_json(REPORTS_DIR / "stage_sw12b_risk_gated_occupancy_repair_report.json", report_json)

    phase_sequence = [
        ("phase1_digest", phase1),
        ("phase2_eval_protocol", phase2),
        ("phaseA1_variant_design", phaseA1),
        ("phaseA2A3_branchA_replay", phaseA2A3),
        ("phaseB1_teacher_cache", phaseB1),
        ("phaseB2_pair_builder", phaseB2),
        ("phaseB3_loss_design", phaseB3),
        ("phaseB4_gradient_and_smoke", phaseB4),
        ("phaseB5B6CDE_train_eval_decision", phaseB5B6CDE),
    ]
    for phase_name, fn in phase_sequence:
        run_phase_if_needed(progress, manifest, phase_name, fn, args.force_phase, args.stop_after_phase)
    progress["status"] = "complete"
    progress["end_time"] = now_iso()
    update_progress(progress)
    manifest["end_time"] = now_iso()
    update_execution_manifest(manifest)


if __name__ == "__main__":
    main()
