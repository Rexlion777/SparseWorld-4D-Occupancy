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
import torch.nn.functional as F


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
BASE_CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
BASE_CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair"

SW12B_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"
SW12B_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"
SW91_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"

PROGRESS_PATH = REPORTS_DIR / "sw12c_progress_state.json"
EXECUTION_MANIFEST_PATH = REPORTS_DIR / "sw12c_execution_manifest.json"
EMPTY_IDX = 17
QUICK_DEBUG_IDS = list(range(5))
EVAL_IDS = list(range(20))
TRAIN_IDS = list(range(20, 40))
CORE_HORIZONS = [0, 2, 4, 6]
CORE_PERTURBATIONS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
PAIR_PERTURBATIONS = ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]
DEFAULT_LR = 2e-5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw12b = load_module(
    "sw12c_sw12b",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py",
)
sw91 = load_module(
    "sw12c_sw91",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/run_sparseworld_sw91_main.py",
)

sw1 = sw12b.sw1
sw2 = sw12b.sw2
sw4_inst = sw12b.sw4_inst
sw7 = sw12b.sw7
sw81 = sw12b.sw81
sw9 = sw12b.sw9
OccupancyRepairMLP = sw12b.OccupancyRepairMLP


@dataclass
class BranchACheckpoint:
    name: str
    path: Path
    config_path: Path


@dataclass
class ResidualConfig:
    name: str
    conf_thr: float
    pos_topk: int
    neg_ratio: int
    front_h46_only: bool = False
    new_occ_budget: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-12C safe routing smoke and residual repair audit")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-gpu", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=0)
    parser.add_argument("--routing-smoke-iters", type=int, default=100)
    parser.add_argument("--branchb-smoke-iters", type=int, default=20)
    parser.add_argument("--force-phase", default=None)
    return parser.parse_args()


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
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "checkpoints",
        ARTIFACTS_DIR / "branchA_eval_dumps",
        ARTIFACTS_DIR / "branchB_residual_pairs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_cuda_cleanup() -> None:
    sw12b.safe_cuda_cleanup()


def safe_div(num: float | int, den: float | int) -> float:
    return sw12b.safe_div(num, den)


def init_execution_manifest() -> dict[str, Any]:
    manifest = {
        "stage": "SW-12C",
        "start_time": now_iso(),
        "phases": [],
        "artifacts": [],
        "safe_claim_boundary": ["subset diagnostic", "smoke only", "not official benchmark", "no final model improvement"],
    }
    write_json(EXECUTION_MANIFEST_PATH, manifest)
    return manifest


def init_progress_state() -> dict[str, Any]:
    state = {
        "stage": "SW-12C",
        "start_time": now_iso(),
        "current_phase": None,
        "completed_phases": [],
        "phase_records": {},
        "status": "running",
    }
    write_json(PROGRESS_PATH, state)
    return state


def phase_start(progress: dict[str, Any], manifest: dict[str, Any], phase_name: str) -> None:
    record = {"phase_name": phase_name, "start_time": now_iso(), "start_ts": time.time(), "status": "running"}
    progress["current_phase"] = phase_name
    progress["phase_records"][phase_name] = record
    manifest["phases"].append(record.copy())
    write_json(PROGRESS_PATH, progress)
    write_json(EXECUTION_MANIFEST_PATH, manifest)


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
    write_json(PROGRESS_PATH, progress)
    write_json(EXECUTION_MANIFEST_PATH, manifest)


def attach_artifact(manifest: dict[str, Any], label: str, path: Path) -> None:
    manifest["artifacts"].append({"label": label, "path": str(path), "timestamp": now_iso()})
    write_json(EXECUTION_MANIFEST_PATH, manifest)


def set_runtime_globals(config_path: Path, checkpoint_path: Path) -> tuple[Path, Path]:
    old_cfg = sw81.CONFIG_PATH
    old_ckpt = sw81.CHECKPOINT_PATH
    sw81.CONFIG_PATH = config_path
    sw81.CHECKPOINT_PATH = checkpoint_path
    return old_cfg, old_ckpt


def restore_runtime_globals(old_cfg: Path, old_ckpt: Path) -> None:
    sw81.CONFIG_PATH = old_cfg
    sw81.CHECKPOINT_PATH = old_ckpt


def build_runtime_from_checkpoint(config_path: Path, checkpoint_path: Path, train: bool, cfg_overrides: dict[str, Any] | None = None):
    old_cfg, old_ckpt = set_runtime_globals(config_path, checkpoint_path)
    try:
        return sw81.build_sparseworld_runtime(train=train, cfg_overrides=cfg_overrides)
    finally:
        restore_runtime_globals(old_cfg, old_ckpt)


def load_case(model: Any, dataset: Any, sample_index: int, perturbation_id: str, horizons: list[int]):
    return sw12b.load_case(model, dataset, sample_index, perturbation_id, horizons)


def build_branch_a_variant_a2():
    variants = {v.label: v for v in sw12b.build_branch_a_variants()}
    return variants["A2_risk_gated_6n_conf09_cap01"], variants["A0_native"]


def p4_config_path() -> Path:
    return sw91.CONFIG_PATHS["H2_TINY_H1_L001"]


def p4_checkpoint_path() -> Path:
    return SW91_ARTIFACTS / "checkpoints/P4_H2_tinyH1_500iter.pth"


def branch_a_smoke_train(
    config_path: Path,
    start_checkpoint: Path,
    target_iters: int,
    seed: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> tuple[list[dict[str, Any]], dict[int, Path], str]:
    cfg, dataset, model, checkpoint = build_runtime_from_checkpoint(
        config_path,
        start_checkpoint,
        train=True,
        cfg_overrides={"data.samples_per_gpu": samples_per_gpu, "data.workers_per_gpu": workers_per_gpu, "optimizer.lr": DEFAULT_LR},
    )
    dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu, shuffle=True, seed=seed)
    from mmcv.runner import build_optimizer

    optimizer = build_optimizer(model, cfg.optimizer)
    base_batch = next(iter(dataloader))
    iter_rows: list[dict[str, Any]] = []
    checkpoint_paths: dict[int, Path] = {}
    start_epoch = 56
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
        start_epoch = int(checkpoint["meta"].get("epoch", start_epoch))
    if hasattr(model, "set_epoch"):
        model.set_epoch(start_epoch)
    stop_reason = "completed"
    try:
        for iter_idx in range(1, target_iters + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model_inputs = sw81.move_train_batch_to_cuda(base_batch)
            losses = model(return_loss=True, **model_inputs)
            total_loss, log_vars = sw81.parse_losses(losses)
            if not bool(torch.isfinite(total_loss).item()):
                stop_reason = "non_finite_loss"
                break
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).detach().cpu().item())
            optimizer.step()
            row = {
                "iter": iter_idx,
                "iter_time_sec": time.perf_counter() - started,
                "peak_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                "grad_norm": grad_norm,
                **log_vars,
            }
            iter_rows.append(row)
            if iter_idx in {20, 50, 100}:
                ckpt_path = ARTIFACTS_DIR / "checkpoints" / f"branchA_A2_{iter_idx}iter.pth"
                sw81.save_checkpoint(
                    ckpt_path,
                    model,
                    {
                        "experiment_id": "branchA_A2_smoke",
                        "epoch": start_epoch,
                        "iter": iter_idx,
                        "planned_iters": target_iters,
                        "optimizer_lr": DEFAULT_LR,
                        "config_path": str(config_path),
                        "source_checkpoint": str(start_checkpoint),
                    },
                )
                checkpoint_paths[iter_idx] = ckpt_path
            if iter_idx % 10 == 0 or iter_idx == target_iters:
                with (LOGS_DIR / "branchA_A2_smoke.log").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(normalize_export(row), ensure_ascii=False) + "\n")
    finally:
        del model, dataset, cfg
        safe_cuda_cleanup()
    return iter_rows, checkpoint_paths, stop_reason


def evaluate_branch_a_checkpoint(
    checkpoint_name: str,
    checkpoint_path: Path,
    config_path: Path,
    subset_name: str,
    sample_ids: list[int],
    include_native: bool,
) -> list[dict[str, Any]]:
    a2_spec, native_spec = build_branch_a_variant_a2()
    variants = [native_spec, a2_spec] if include_native else [a2_spec]
    cfg, dataset, model, _ = build_runtime_from_checkpoint(
        config_path,
        checkpoint_path,
        train=False,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    head = sw4_inst.get_pts_bbox_head(model)
    sectors = {name: tensor.cpu() for name, tensor in sw7.build_sector_masks().items()}
    rows: list[dict[str, Any]] = []
    try:
        for perturbation_id in CORE_PERTURBATIONS:
            for sample_index in sample_ids:
                _, per_h = load_case(model, dataset, sample_index, perturbation_id, CORE_HORIZONS)
                for horizon_s in CORE_HORIZONS:
                    case = per_h[horizon_s]
                    case_rows, dump_payload = sw12b.branch_a_replay_case(
                        sw12b.CheckpointSpec(checkpoint_name, checkpoint_path, config_path, "sw12c"),
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
                    for row in case_rows:
                        row["phase"] = subset_name
                    rows.extend(case_rows)
                    dump_dir = ARTIFACTS_DIR / "branchA_eval_dumps" / checkpoint_name / subset_name / perturbation_id
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dump_dir / f"sample_{sample_index}_h{horizon_s}.npz", **dump_payload)
    finally:
        del model, dataset, cfg
        safe_cuda_cleanup()
    return rows


def branch_a_eval_and_decide(iter_rows: list[dict[str, Any]], checkpoint_paths: dict[int, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_path = p4_config_path()
    baseline_rows = evaluate_branch_a_checkpoint("P4_native_baseline", p4_checkpoint_path(), config_path, "quick_debug_5", QUICK_DEBUG_IDS, include_native=True)
    baseline_rows += evaluate_branch_a_checkpoint("P4_native_baseline", p4_checkpoint_path(), config_path, "eval_core_20", EVAL_IDS, include_native=True)
    eval_rows = list(baseline_rows)
    completed_iters = sorted(checkpoint_paths)
    for iter_idx in completed_iters:
        subset_rows = evaluate_branch_a_checkpoint(f"branchA_A2_{iter_idx}iter", checkpoint_paths[iter_idx], config_path, "quick_debug_5", QUICK_DEBUG_IDS, include_native=True)
        eval_rows.extend(subset_rows)
        subset_rows = evaluate_branch_a_checkpoint(f"branchA_A2_{iter_idx}iter", checkpoint_paths[iter_idx], config_path, "eval_core_20", EVAL_IDS, include_native=True)
        eval_rows.extend(subset_rows)
    agg_rows = sw12b.aggregate_rows(
        eval_rows,
        ["phase", "checkpoint_name", "variant_label", "variant_name", "is_oracle", "perturbation_id", "horizon_s"],
    )
    write_csv(REPORTS_DIR / "branchA_A2_100iter_eval.csv", agg_rows)
    sw12b_rows = read_csv_rows(SW12B_REPORTS / "branchA_routing_candidates.csv")
    sw12b_a2_baseline = next(
        row
        for row in sw12b_rows
        if row["checkpoint_name"] == "P4_H2_tinyH1_500iter"
        and row["variant_label"] == "A2_risk_gated_6n_conf09_cap01"
        and row["perturbation_id"] == "A10_drop_front_triplet"
        and row["horizon_s"] == "6"
    )
    baseline_target_recovery = float(sw12b_a2_baseline["target_recovery"])
    baseline_a10_recovery = float(sw12b_a2_baseline["A10_front_h6_recovery_ratio"])
    candidate_rows = [
        row
        for row in agg_rows
        if row["checkpoint_name"].startswith("branchA_A2_")
        and row["variant_label"] == "A2_risk_gated_6n_conf09_cap01"
        and row["phase"] == "eval_core_20"
        and row["perturbation_id"] == "A10_drop_front_triplet"
        and str(row["horizon_s"]) == "6"
    ]
    unsafe_rows = []
    amplified_rows = []
    for row in candidate_rows:
        unsafe = (
            float(row["clean_false_positive_delta"]) > 0.005
            or float(row["C4_false_positive_delta"]) > 0.008
            or float(row["pred_gt_density_delta"]) > 0.05
            or float(row["wrong_class_activation_delta"]) > 0.005
            or float(row["clean_occupied_iou_delta"]) < -0.005
        )
        if unsafe:
            unsafe_rows.append(row)
        if float(row["target_recovery"]) > baseline_target_recovery or float(row["A10_front_h6_recovery_ratio"]) > baseline_a10_recovery:
            amplified_rows.append(row)
    if unsafe_rows:
        decision_type = "A2_TRAIN_UNSAFE_DRIFT"
    elif amplified_rows:
        decision_type = "A2_TRAIN_SIGNAL_AMPLIFIED_SAFE"
    elif candidate_rows:
        decision_type = "A2_TRAIN_NO_AMPLIFICATION"
    else:
        decision_type = "A2_TRAIN_INSTRUMENTATION_FAILED"
    best_row = max(candidate_rows, key=lambda r: float(r["target_recovery"]), default=None)
    decision = {
        "decision_type": decision_type,
        "baseline_replay_target_recovery": baseline_target_recovery,
        "baseline_replay_A10_front_h6_recovery_ratio": baseline_a10_recovery,
        "best_candidate_row": best_row,
        "candidate_rows": candidate_rows,
        "unsafe_rows": unsafe_rows,
        "amplified_rows": amplified_rows,
    }
    write_json(REPORTS_DIR / "branchA_A2_100iter_decision.json", decision)
    return agg_rows, decision


def residual_configs() -> list[ResidualConfig]:
    return [
        ResidualConfig("R1_residual_conf09_K128_neg2", 0.90, 128, 2, False, 0.1),
        ResidualConfig("R2_residual_conf09_K256_neg2", 0.90, 256, 2, False, 0.1),
        ResidualConfig("R3_residual_conf095_K128_neg2", 0.95, 128, 2, False, 0.1),
        ResidualConfig("R4_residual_conf095_K256_neg4", 0.95, 256, 4, False, 0.2),
        ResidualConfig("R5_residual_front_h46_only_conf095_K128_neg4", 0.95, 128, 4, True, 0.2),
    ]


def high_risk_mask(
    teacher_label: torch.Tensor,
    gt_label: torch.Tensor,
    front_mask: torch.Tensor,
    small_mask: torch.Tensor,
    new_visible_mask: torch.Tensor,
    perturbation_id: str,
    horizon_s: int,
    conf_thr_mask: torch.Tensor | None = None,
    front_h46_only: bool = False,
) -> torch.Tensor:
    mask = front_mask | small_mask | new_visible_mask
    if horizon_s in {4, 6}:
        mask = mask | torch.ones_like(mask, dtype=torch.bool)
    if perturbation_id in {"A1_drop_cam_front", "A10_drop_front_triplet"}:
        mask = mask | front_mask
    if front_h46_only:
        mask = front_mask & torch.ones_like(mask, dtype=torch.bool)
        if horizon_s not in {4, 6}:
            mask = torch.zeros_like(mask)
    return mask


def make_residual_pair_npz(
    cfg: ResidualConfig,
    teacher_npz: dict[str, np.ndarray],
    gt_h: torch.Tensor,
    student_label: torch.Tensor,
    student_conf: torch.Tensor,
    student_margin: torch.Tensor,
    contributor_count: torch.Tensor,
    gate_count: torch.Tensor,
    perturbation_id: str,
    horizon_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    teacher_label = torch.from_numpy(teacher_npz["teacher_label"]).long()
    teacher_conf = torch.from_numpy(teacher_npz["teacher_conf"]).float()
    front_mask = torch.from_numpy(teacher_npz["front_mask"]).bool()
    small_mask = torch.from_numpy(teacher_npz["small_mask"]).bool()
    new_visible_mask = torch.from_numpy(teacher_npz["new_visible_mask"]).bool()
    dynamic_mask = torch.from_numpy(teacher_npz["dynamic_mask"]).bool()
    valid_mask = torch.from_numpy(teacher_npz["valid_mask"]).bool()
    teacher_occupied = teacher_label != EMPTY_IDX
    teacher_reliable_occupied = teacher_occupied & (teacher_conf >= cfg.conf_thr) & valid_mask
    teacher_reliable_free = (teacher_label == EMPTY_IDX) & (teacher_conf >= cfg.conf_thr) & valid_mask
    student_native_occupied = (student_label != EMPTY_IDX) & valid_mask
    risk_mask = high_risk_mask(teacher_label, gt_h, front_mask, small_mask, new_visible_mask, perturbation_id, horizon_s, front_h46_only=cfg.front_h46_only)
    residual_positive_raw = teacher_reliable_occupied & (~student_native_occupied)
    positive_repair_mask = residual_positive_raw & risk_mask
    negative_guard_raw = (teacher_reliable_free | ((gt_h == EMPTY_IDX) & valid_mask) | ((student_native_occupied) & (gt_h == EMPTY_IDX))) & valid_mask
    if perturbation_id == "C4_motion_blur_9":
        negative_guard_raw = negative_guard_raw | ((teacher_label == EMPTY_IDX) & (~front_mask) & valid_mask)
    raw_positive_already_student_occupied_count = int((teacher_reliable_occupied & student_native_occupied & risk_mask).sum().item())
    raw_positive_not_high_risk_count = int((residual_positive_raw & (~risk_mask)).sum().item())
    overlap_mask = positive_repair_mask & negative_guard_raw
    raw_overlap_removed_count = int(overlap_mask.sum().item())
    positive_repair_mask = positive_repair_mask & (~overlap_mask)
    negative_guard_mask = negative_guard_raw & (~positive_repair_mask)

    pos_coords = torch.nonzero(positive_repair_mask, as_tuple=False)
    neg_coords = torch.nonzero(negative_guard_mask, as_tuple=False)
    selected_positive_count = min(cfg.pos_topk, int(pos_coords.shape[0]))
    if selected_positive_count > 0:
        pos_scores = teacher_conf[pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2]]
        pos_order = torch.argsort(pos_scores, descending=True)[:selected_positive_count]
        pos_coords = pos_coords[pos_order]
    else:
        pos_coords = pos_coords[:0]
    selected_negative_count = min(int(neg_coords.shape[0]), cfg.neg_ratio * selected_positive_count)
    if selected_negative_count > 0:
        neg_scores = student_conf[neg_coords[:, 0], neg_coords[:, 1], neg_coords[:, 2]]
        neg_order = torch.argsort(neg_scores, descending=True)[:selected_negative_count]
        neg_coords = neg_coords[neg_order]
    else:
        neg_coords = neg_coords[:0]

    coords = torch.cat([pos_coords, neg_coords], dim=0) if pos_coords.shape[0] + neg_coords.shape[0] > 0 else torch.zeros((0, 3), dtype=torch.long)
    numeric = sw12b.build_numeric_features(
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
        student_native_occupied,
    )
    teacher_target = teacher_label[coords[:, 0], coords[:, 1], coords[:, 2]] if coords.shape[0] else teacher_label.new_zeros((0,))
    teacher_target_occ = (teacher_target != EMPTY_IDX).long()
    sample_weights = torch.ones(coords.shape[0], dtype=torch.float32)
    if coords.shape[0]:
        sample_weights[: selected_positive_count] *= 2.0
        sample_weights[selected_positive_count:] *= 1.5
    selected_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    if coords.shape[0]:
        selected_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    ignored_count = int((valid_mask & (~selected_mask)).sum().item())
    audit = {
        "teacher_occupied_count_raw": int((teacher_occupied & valid_mask).sum().item()),
        "teacher_reliable_occupied_count_raw": int(teacher_reliable_occupied.sum().item()),
        "student_native_occupied_count_raw": int(student_native_occupied.sum().item()),
        "residual_positive_count_raw": int(residual_positive_raw.sum().item()),
        "high_risk_count_raw": int((risk_mask & valid_mask).sum().item()),
        "positive_repair_count_raw": int(positive_repair_mask.sum().item()),
        "selected_positive_count": selected_positive_count,
        "selected_negative_count": selected_negative_count,
        "ignored_count": ignored_count,
        "teacher_reliable_free_count_raw": int(teacher_reliable_free.sum().item()),
        "negative_guard_count_raw": int(negative_guard_mask.sum().item()),
        "positive_negative_overlap_count": int((positive_repair_mask & negative_guard_mask).sum().item()),
        "positive_already_student_occupied_count": int((positive_repair_mask & student_native_occupied).sum().item()),
        "positive_not_high_risk_count": int((positive_repair_mask & (~risk_mask)).sum().item()),
        "raw_overlap_removed_count": raw_overlap_removed_count,
        "raw_positive_already_student_occupied_count": raw_positive_already_student_occupied_count,
        "raw_positive_not_high_risk_count": raw_positive_not_high_risk_count,
        "positive_conf_mean": float(teacher_conf[pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2]].mean().item()) if selected_positive_count > 0 else 0.0,
        "negative_conf_mean": float(teacher_conf[neg_coords[:, 0], neg_coords[:, 1], neg_coords[:, 2]].mean().item()) if selected_negative_count > 0 else 0.0,
        "ignored_reason_summary": "valid_voxel_not_selected_by_strict_residual_or_negative_guard",
    }
    pair = {
        "coords": coords.cpu().numpy().astype(np.int16),
        "student_pred_class": student_label[coords[:, 0], coords[:, 1], coords[:, 2]].cpu().numpy().astype(np.uint8) if coords.shape[0] else np.zeros((0,), dtype=np.uint8),
        "numeric_feats": numeric.cpu().numpy().astype(np.float32),
        "teacher_target_class": teacher_target.cpu().numpy().astype(np.uint8) if coords.shape[0] else np.zeros((0,), dtype=np.uint8),
        "teacher_target_occ": teacher_target_occ.cpu().numpy().astype(np.uint8),
        "sample_weights": sample_weights.cpu().numpy().astype(np.float32),
        "horizon_index": np.full((coords.shape[0],), sw12b.horizon_index(horizon_s), dtype=np.int64),
        "perturb_index": np.full((coords.shape[0],), sw12b.perturb_index(perturbation_id), dtype=np.int64),
        "teacher_conf_selected": teacher_conf[coords[:, 0], coords[:, 1], coords[:, 2]].cpu().numpy().astype(np.float32) if coords.shape[0] else np.zeros((0,), dtype=np.float32),
        "pos_count": np.array([selected_positive_count], dtype=np.int32),
        "neg_count": np.array([selected_negative_count], dtype=np.int32),
        "new_occ_budget": np.array([cfg.new_occ_budget], dtype=np.float32),
    }
    return pair, audit


def load_residual_pair_dataset(pair_paths: list[Path]) -> dict[str, torch.Tensor]:
    feats = []
    pred_class = []
    teacher_class = []
    teacher_occ = []
    weights = []
    horizon_idx = []
    perturb_idx = []
    budgets = []
    for path in pair_paths:
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
        budgets.append(torch.from_numpy(np.repeat(data["new_occ_budget"], data["coords"].shape[0])).float())
    if not feats:
        zero_feats = torch.zeros((0, 12), dtype=torch.float32)
        zero_long = torch.zeros((0,), dtype=torch.long)
        zero_float = torch.zeros((0,), dtype=torch.float32)
        return {
            "numeric_feats": zero_feats,
            "student_pred_class": zero_long,
            "teacher_target_class": zero_long,
            "teacher_target_occ": zero_float,
            "sample_weights": zero_float,
            "horizon_index": zero_long,
            "perturb_index": zero_long,
            "new_occ_budget": zero_float,
        }
    return {
        "numeric_feats": torch.cat(feats, dim=0),
        "student_pred_class": torch.cat(pred_class, dim=0),
        "teacher_target_class": torch.cat(teacher_class, dim=0),
        "teacher_target_occ": torch.cat(teacher_occ, dim=0),
        "sample_weights": torch.cat(weights, dim=0),
        "horizon_index": torch.cat(horizon_idx, dim=0),
        "perturb_index": torch.cat(perturb_idx, dim=0),
        "new_occ_budget": torch.cat(budgets, dim=0),
    }


def compute_residual_losses(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_class = batch["teacher_target_class"].cuda(non_blocking=False)
    target_occ = batch["teacher_target_occ"].cuda(non_blocking=False)
    weights = batch["sample_weights"].cuda(non_blocking=False)
    new_occ_budget = batch["new_occ_budget"].cuda(non_blocking=False)
    probs = logits.softmax(dim=-1)
    empty_prob = probs[:, EMPTY_IDX]
    occ_prob = 1.0 - empty_prob
    occ_logits = torch.logit(torch.clamp(occ_prob, 1e-4, 1 - 1e-4))
    pos_mask = target_occ > 0.5
    neg_mask = target_occ < 0.5
    native_occ = batch["numeric_feats"][:, 8].cuda(non_blocking=False)
    loss_occ = torch.tensor(0.0, device=logits.device)
    loss_sem = torch.tensor(0.0, device=logits.device)
    loss_neg = torch.tensor(0.0, device=logits.device)
    if bool(pos_mask.any().item()):
        loss_occ = F.binary_cross_entropy_with_logits(occ_logits[pos_mask], torch.ones_like(occ_logits[pos_mask]), reduction="none")
        loss_occ = (loss_occ * weights[pos_mask]).mean()
        loss_sem = F.cross_entropy(logits[pos_mask], target_class[pos_mask], reduction="none")
        loss_sem = (loss_sem * weights[pos_mask]).mean()
    if bool(neg_mask.any().item()):
        loss_neg = F.binary_cross_entropy_with_logits(occ_logits[neg_mask], torch.zeros_like(occ_logits[neg_mask]), reduction="none")
        loss_neg = (loss_neg * weights[neg_mask]).mean()
    new_occ_proxy = torch.clamp(occ_prob - native_occ, min=0.0).mean()
    budget = new_occ_budget.mean() if new_occ_budget.numel() else torch.tensor(0.1, device=logits.device)
    loss_density = F.relu(new_occ_proxy - budget)
    total = loss_occ + 0.3 * loss_sem + 2.0 * loss_neg + 2.0 * loss_density
    return {
        "total_loss": total,
        "L_residual_occ_repair": loss_occ,
        "L_residual_semantic_repair": 0.3 * loss_sem,
        "L_negative_empty_guard": 2.0 * loss_neg,
        "L_density_budget": 2.0 * loss_density,
        "pred_density_proxy_on_sampled": occ_prob.mean().detach(),
        "positive_occ_prob_mean": occ_prob[pos_mask].mean().detach() if bool(pos_mask.any().item()) else torch.tensor(0.0, device=logits.device),
        "negative_occ_prob_mean": occ_prob[neg_mask].mean().detach() if bool(neg_mask.any().item()) else torch.tensor(0.0, device=logits.device),
        "native_student_density_proxy": native_occ.mean().detach() if native_occ.numel() else torch.tensor(0.0, device=logits.device),
        "new_occ_budget_proxy": new_occ_proxy.detach(),
        "selected_positive_count": int(pos_mask.sum().item()),
        "selected_negative_count": int(neg_mask.sum().item()),
    }


def branch_b_gradient_check(dataset: dict[str, torch.Tensor], cfg: ResidualConfig) -> dict[str, Any]:
    model = OccupancyRepairMLP().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    n = dataset["numeric_feats"].shape[0]
    if n == 0:
        safe_cuda_cleanup()
        return {
            "config_name": cfg.name,
            "total_loss": 0.0,
            "L_residual_occ_repair": 0.0,
            "L_residual_semantic_repair": 0.0,
            "L_negative_empty_guard": 0.0,
            "L_density_budget": 0.0,
            "grad_norm": 0.0,
            "has_nan": True,
            "has_inf": True,
            "selected_positive_count": 0,
            "selected_negative_count": 0,
            "pred_density_proxy_on_sampled": 0.0,
            "positive_occ_prob_mean": 0.0,
            "negative_occ_prob_mean": 0.0,
            "native_student_density_proxy": 0.0,
            "new_occ_budget_proxy": 0.0,
            "pass": False,
            "failure_reason": "empty_dataset",
        }
    idx = torch.randint(0, n, (min(4096, n),))
    batch = {k: v[idx] if v.shape[0] == n else v for k, v in dataset.items()}
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        batch["student_pred_class"].cuda(non_blocking=False),
        batch["horizon_index"].cuda(non_blocking=False),
        batch["perturb_index"].cuda(non_blocking=False),
        batch["numeric_feats"].cuda(non_blocking=False),
    )
    losses = compute_residual_losses(logits, batch)
    total = losses["total_loss"]
    total.backward()
    grad_sq = 0.0
    grad_finite = True
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_sq += float(param.grad.detach().float().norm().item() ** 2)
        grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
    optimizer.step()
    pass_flag = (
        grad_finite
        and int(losses["selected_positive_count"]) > 0
        and int(losses["selected_negative_count"]) >= 2 * max(1, int(losses["selected_positive_count"])) or int(losses["selected_negative_count"]) > 0
    )
    row = {
        "config_name": cfg.name,
        "total_loss": float(total.detach().cpu().item()),
        "L_residual_occ_repair": float(losses["L_residual_occ_repair"].detach().cpu().item()),
        "L_residual_semantic_repair": float(losses["L_residual_semantic_repair"].detach().cpu().item()),
        "L_negative_empty_guard": float(losses["L_negative_empty_guard"].detach().cpu().item()),
        "L_density_budget": float(losses["L_density_budget"].detach().cpu().item()),
        "ratio_occ": safe_div(float(losses["L_residual_occ_repair"].detach().cpu().item()), max(1e-6, float(total.detach().cpu().item()))),
        "ratio_sem": safe_div(float(losses["L_residual_semantic_repair"].detach().cpu().item()), max(1e-6, float(total.detach().cpu().item()))),
        "ratio_neg": safe_div(float(losses["L_negative_empty_guard"].detach().cpu().item()), max(1e-6, float(total.detach().cpu().item()))),
        "ratio_density": safe_div(float(losses["L_density_budget"].detach().cpu().item()), max(1e-6, float(total.detach().cpu().item()))),
        "grad_norm": math.sqrt(max(0.0, grad_sq)),
        "has_nan": not grad_finite,
        "has_inf": not grad_finite,
        "selected_positive_count": int(losses["selected_positive_count"]),
        "selected_negative_count": int(losses["selected_negative_count"]),
        "pred_density_proxy_on_sampled": float(losses["pred_density_proxy_on_sampled"].detach().cpu().item()),
        "positive_occ_prob_mean": float(losses["positive_occ_prob_mean"].detach().cpu().item()),
        "negative_occ_prob_mean": float(losses["negative_occ_prob_mean"].detach().cpu().item()),
        "native_student_density_proxy": float(losses["native_student_density_proxy"].detach().cpu().item()),
        "new_occ_budget_proxy": float(losses["new_occ_budget_proxy"].detach().cpu().item()),
        "pass": bool(
            grad_finite
            and int(losses["selected_positive_count"]) > 0
            and int(losses["selected_negative_count"]) >= min(int(losses["selected_positive_count"]) * 2, int(losses["selected_negative_count"]))
            and float(losses["pred_density_proxy_on_sampled"].detach().cpu().item()) < 0.85
            and float(losses["negative_occ_prob_mean"].detach().cpu().item()) < float(losses["positive_occ_prob_mean"].detach().cpu().item())
            and float(losses["new_occ_budget_proxy"].detach().cpu().item()) <= cfg.new_occ_budget + 1e-6
        ),
    }
    del model
    safe_cuda_cleanup()
    return row


def train_residual_smoke(dataset: dict[str, torch.Tensor], cfg: ResidualConfig, iters: int) -> list[dict[str, Any]]:
    model = OccupancyRepairMLP().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    n = dataset["numeric_feats"].shape[0]
    rows: list[dict[str, Any]] = []
    if n == 0:
        del model
        safe_cuda_cleanup()
        return rows
    for iter_idx in range(1, iters + 1):
        idx = torch.randint(0, n, (min(4096, n),))
        batch = {k: v[idx] if v.shape[0] == n else v for k, v in dataset.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["student_pred_class"].cuda(non_blocking=False),
            batch["horizon_index"].cuda(non_blocking=False),
            batch["perturb_index"].cuda(non_blocking=False),
            batch["numeric_feats"].cuda(non_blocking=False),
        )
        losses = compute_residual_losses(logits, batch)
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
                "config_name": cfg.name,
                "iter": iter_idx,
                "total_loss": float(losses["total_loss"].detach().cpu().item()),
                "L_residual_occ": float(losses["L_residual_occ_repair"].detach().cpu().item()),
                "L_residual_sem": float(losses["L_residual_semantic_repair"].detach().cpu().item()),
                "L_negative_empty": float(losses["L_negative_empty_guard"].detach().cpu().item()),
                "L_density_budget": float(losses["L_density_budget"].detach().cpu().item()),
                "grad_norm": math.sqrt(max(0.0, grad_sq)),
                "pred_density_proxy_on_sampled": float(losses["pred_density_proxy_on_sampled"].detach().cpu().item()),
                "positive_occ_prob_mean": float(losses["positive_occ_prob_mean"].detach().cpu().item()),
                "negative_occ_prob_mean": float(losses["negative_occ_prob_mean"].detach().cpu().item()),
                "native_student_density_proxy": float(losses["native_student_density_proxy"].detach().cpu().item()),
                "new_occ_budget_proxy": float(losses["new_occ_budget_proxy"].detach().cpu().item()),
                "selected_positive_count": int(losses["selected_positive_count"]),
                "selected_negative_count": int(losses["selected_negative_count"]),
                "has_nan": not grad_finite,
                "has_inf": not grad_finite,
            }
        )
    del model
    safe_cuda_cleanup()
    return rows


def phase1_digest(manifest: dict[str, Any]) -> None:
    decision = read_json(SW12B_REPORTS / "sw12b_risk_gated_occupancy_repair_decision.json")
    brancha = read_csv_rows(SW12B_REPORTS / "branchA_routing_candidates.csv")
    branchb_smoke = read_csv_rows(SW12B_REPORTS / "branchB_20iter_smoke_metrics.csv")
    branchb_grad = read_csv_rows(SW12B_REPORTS / "branchB_gradient_check.csv")
    digest = {
        "decision": decision,
        "branchA_safe_candidates": [row for row in brancha if row["classification"] == "A_SAFE_TARGETED"],
        "branchB_gradient_rows": branchb_grad,
        "branchB_smoke_rows_head": branchb_smoke[:10],
    }
    write_json(REPORTS_DIR / "sw12b_digest_for_sw12c.json", digest)
    write_md(
        REPORTS_DIR / "sw12c_objective.md",
        "\n".join(
            [
                "1. Branch A A2 is a weak but safe candidate, not a strong gain.",
                "2. Branch B did not enter short training because sampled density_proxy stayed above the 0.85 gate.",
                "3. Branch B in SW-12B was not pure residual repair.",
                "4. SW-12C therefore limits work to A2 100-iter smoke and a strict residual Branch B rebuild with audit first and smoke only.",
                "",
            ]
        ),
    )
    attach_artifact(manifest, "sw12b_digest_for_sw12c", REPORTS_DIR / "sw12b_digest_for_sw12c.json")


def phase_branch_a(manifest: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iter_rows, checkpoint_paths, stop_reason = branch_a_smoke_train(
        p4_config_path(),
        p4_checkpoint_path(),
        args.routing_smoke_iters,
        args.seed,
        args.samples_per_gpu,
        args.workers_per_gpu,
    )
    write_csv(REPORTS_DIR / "branchA_A2_100iter_train_metrics.csv", iter_rows)
    attach_artifact(manifest, "branchA_A2_100iter_train_metrics", REPORTS_DIR / "branchA_A2_100iter_train_metrics.csv")
    agg_rows, decision = branch_a_eval_and_decide(iter_rows, checkpoint_paths)
    if iter_rows:
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        xs = [row["iter"] for row in iter_rows]
        ys = [row["loss_total"] for row in iter_rows]
        ax.plot(xs, ys, color="#117A65")
        ax.set_title("SW-12C subset diagnostic Branch A A2 100-iter smoke tradeoff")
        ax.set_xlabel("iter")
        ax.set_ylabel("loss_total")
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / "branchA_A2_100iter_tradeoff.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return agg_rows, {"stop_reason": stop_reason, **decision}


def phase_branch_b(manifest: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    teacher_manifest = read_csv_rows(SW12B_REPORTS / "branchB_teacher_cache_manifest.csv")
    teacher_lookup = {(row["split"], int(row["sample_index"]), int(row["horizon_s"])): Path(row["cache_path"]) for row in teacher_manifest}
    write_md(
        REPORTS_DIR / "branchB_residual_mask_definition.md",
        "\n".join(
            [
                "- positive_repair_mask = teacher_occupied & teacher_conf>=thr & (~student_native_occupied) & high_risk_region",
                "- no reliable_teacher & risk_mask union is allowed",
                "- negative_guard_mask uses teacher_reliable_free / GT_free / student_native_false_positive / C4 free background",
                "- positive and negative must be mutually exclusive",
                "- ignore mask is explicit via ignored_count",
                "",
            ]
        ),
    )
    cfgs = residual_configs()
    write_json(REPORTS_DIR / "branchB_residual_config_manifest.json", {"configs": [cfg.__dict__ for cfg in cfgs]})

    audit_rows: list[dict[str, Any]] = []
    pair_dist_rows: list[dict[str, Any]] = []
    all_pair_paths: dict[str, list[Path]] = defaultdict(list)

    base_cfg, base_dataset, base_model, _ = build_runtime_from_checkpoint(
        BASE_CONFIG_PATH,
        BASE_CHECKPOINT_PATH,
        train=False,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    head = sw4_inst.get_pts_bbox_head(base_model)
    try:
        for split_name, sample_ids in [("train", TRAIN_IDS), ("eval", EVAL_IDS)]:
            for perturbation_id in PAIR_PERTURBATIONS:
                for sample_index in sample_ids:
                    _, per_h = load_case(base_model, base_dataset, sample_index, perturbation_id, CORE_HORIZONS)
                    for horizon_s in CORE_HORIZONS:
                        case = per_h[horizon_s]
                        teacher_npz = dict(np.load(teacher_lookup[(split_name, sample_index, horizon_s)]))
                        pred_dbg, dbg_list = sw4_inst.get_occ_debug(head, case["pred_dict"], capture_dense=True)
                        pred = pred_dbg[0].detach().cpu().long()
                        dbg = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in dbg_list[0].items()}
                        student_conf, student_cls, student_margin = sw12b.derive_top_conf_and_margin(dbg["dense_occ_after_padding"])
                        contributor_count = torch.as_tensor(dbg["contributor_count_dense"]).cpu()
                        gate_count = sw12b.gate_pass_dense_or_zero(dbg)
                        for residual_cfg in cfgs:
                            pair, audit = make_residual_pair_npz(
                                residual_cfg,
                                teacher_npz,
                                case["gt_h"],
                                pred,
                                student_conf,
                                student_margin,
                                contributor_count,
                                gate_count,
                                perturbation_id,
                                horizon_s,
                            )
                            pair_path = ARTIFACTS_DIR / "branchB_residual_pairs" / residual_cfg.name / f"{split_name}__sample{sample_index:03d}__{perturbation_id}__h{horizon_s}.npz"
                            pair_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(pair_path, **pair)
                            all_pair_paths[residual_cfg.name].append(pair_path)
                            audit_row = {
                                "config_name": residual_cfg.name,
                                "split": split_name,
                                "sample_index": sample_index,
                                "perturbation_id": perturbation_id,
                                "horizon_s": horizon_s,
                                **audit,
                            }
                            audit_rows.append(audit_row)
                            pair_dist_rows.append(
                                {
                                    "config_name": residual_cfg.name,
                                    "split": split_name,
                                    "perturbation_id": perturbation_id,
                                    "horizon_s": horizon_s,
                                    "selected_positive_count": audit["selected_positive_count"],
                                    "selected_negative_count": audit["selected_negative_count"],
                                }
                            )
    finally:
        del base_model, base_dataset, base_cfg
        safe_cuda_cleanup()

    write_csv(REPORTS_DIR / "branchB_residual_mask_audit.csv", audit_rows)
    write_csv(REPORTS_DIR / "branchB_residual_pair_distribution.csv", sw12b.aggregate_rows(pair_dist_rows, ["config_name", "split", "perturbation_id", "horizon_s"]))
    failed_audit = [
        row
        for row in audit_rows
        if int(row["positive_negative_overlap_count"]) != 0
        or int(row["positive_already_student_occupied_count"]) != 0
        or int(row["selected_positive_count"]) > next(cfg.pos_topk for cfg in cfgs if cfg.name == row["config_name"])
        or int(row["selected_negative_count"]) > next(cfg.neg_ratio * cfg.pos_topk for cfg in cfgs if cfg.name == row["config_name"])
    ]
    summary_lines = [
        f"- audit_rows: {len(audit_rows)}",
        f"- failed_rows: {len(failed_audit)}",
        f"- strict residual configs: {', '.join(cfg.name for cfg in cfgs)}",
    ]
    write_md(REPORTS_DIR / "branchB_residual_mask_audit_summary.md", "\n".join(summary_lines) + "\n")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    cfg_names = [cfg.name for cfg in cfgs]
    pos_means = []
    neg_means = []
    for cfg in cfgs:
        rows = [row for row in audit_rows if row["config_name"] == cfg.name]
        pos_means.append(float(np.mean([int(r["selected_positive_count"]) for r in rows])) if rows else 0.0)
        neg_means.append(float(np.mean([int(r["selected_negative_count"]) for r in rows])) if rows else 0.0)
    xs = np.arange(len(cfg_names))
    ax.bar(xs - 0.15, pos_means, width=0.3, label="selected_positive_count")
    ax.bar(xs + 0.15, neg_means, width=0.3, label="selected_negative_count")
    ax.set_xticks(xs)
    ax.set_xticklabels(cfg_names, rotation=20)
    ax.set_title("SW-12C subset diagnostic residual mask distribution")
    ax.legend(fontsize=7)
    fig.savefig(FIGURES_DIR / "branchB_residual_mask_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    write_md(
        REPORTS_DIR / "branchB_residual_loss_design.md",
        "\n".join(
            [
                "- L_residual_occ_repair: BCE only on positive_repair_mask",
                "- L_residual_semantic_repair: CE only on positive_repair_mask",
                "- L_negative_empty_guard: BCE only on negative_guard_mask",
                "- L_density_budget: penalize new occupied probability above fixed budget relative to native student",
                "- no full teacher consistency; no global teacher imitation across the reliable teacher volume",
                "",
            ]
        ),
    )

    if failed_audit:
        grad_rows = [{"config_name": "none", "failure_reason": "mask_audit_failed"}]
        write_csv(REPORTS_DIR / "branchB_residual_gradient_check.csv", grad_rows)
        write_csv(REPORTS_DIR / "branchB_residual_20iter_smoke_metrics.csv", [{"config_name": "none", "skipped_reason": "mask_audit_failed"}])
        decision = {"decision_type": "B_RESIDUAL_MASK_AUDIT_FAILED", "failed_rows": failed_audit[:10]}
        write_json(REPORTS_DIR / "branchB_residual_smoke_decision.json", decision)
        return audit_rows, decision, grad_rows

    grad_rows = []
    passing_cfgs: list[ResidualConfig] = []
    for residual_cfg in cfgs:
        pair_paths = [path for path in all_pair_paths[residual_cfg.name] if path.name.startswith("train__")]
        dataset = load_residual_pair_dataset(pair_paths)
        row = branch_b_gradient_check(dataset, residual_cfg)
        grad_rows.append(row)
        if truthy(row.get("pass")):
            passing_cfgs.append(residual_cfg)
    write_csv(REPORTS_DIR / "branchB_residual_gradient_check.csv", grad_rows)
    if not passing_cfgs:
        write_csv(REPORTS_DIR / "branchB_residual_20iter_smoke_metrics.csv", [{"config_name": "none", "skipped_reason": "all_gradient_checks_failed"}])
        decision = {"decision_type": "B_RESIDUAL_NUMERIC_FAILED", "gradient_rows": grad_rows}
        write_json(REPORTS_DIR / "branchB_residual_smoke_decision.json", decision)
        return audit_rows, decision, grad_rows

    ranked_cfgs = sorted(
        passing_cfgs,
        key=lambda cfg: next(float(row["pred_density_proxy_on_sampled"]) for row in grad_rows if row["config_name"] == cfg.name),
    )[:2]
    smoke_rows: list[dict[str, Any]] = []
    for residual_cfg in ranked_cfgs:
        pair_paths = [path for path in all_pair_paths[residual_cfg.name] if path.name.startswith("train__")]
        dataset = load_residual_pair_dataset(pair_paths)
        smoke_rows.extend(train_residual_smoke(dataset, residual_cfg, args.branchb_smoke_iters))
    write_csv(REPORTS_DIR / "branchB_residual_20iter_smoke_metrics.csv", smoke_rows)
    if smoke_rows:
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        for name in sorted({row["config_name"] for row in smoke_rows}):
            rows = [row for row in smoke_rows if row["config_name"] == name]
            ax.plot([int(r["iter"]) for r in rows], [float(r["pred_density_proxy_on_sampled"]) for r in rows], label=name)
        ax.set_title("SW-12C subset diagnostic residual density curve")
        ax.set_xlabel("iter")
        ax.set_ylabel("pred_density_proxy_on_sampled")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / "branchB_residual_density_curve.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        for name in sorted({row["config_name"] for row in smoke_rows}):
            rows = [row for row in smoke_rows if row["config_name"] == name]
            ax.plot([int(r["iter"]) for r in rows], [float(r["total_loss"]) for r in rows], label=name)
        ax.set_title("SW-12C subset diagnostic residual loss curve")
        ax.set_xlabel("iter")
        ax.set_ylabel("total_loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / "branchB_residual_loss_curve.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    by_cfg = defaultdict(list)
    for row in smoke_rows:
        by_cfg[row["config_name"]].append(row)
    passed = []
    still_high = []
    for cfg_name, rows in by_cfg.items():
        rows = sorted(rows, key=lambda r: int(r["iter"]))
        head_loss = float(np.mean([float(r["total_loss"]) for r in rows[:5]]))
        tail_loss = float(np.mean([float(r["total_loss"]) for r in rows[-5:]]))
        density_head = float(np.mean([float(r["pred_density_proxy_on_sampled"]) for r in rows[:5]]))
        density_tail = float(np.mean([float(r["pred_density_proxy_on_sampled"]) for r in rows[-5:]]))
        density_peak = float(max(float(r["pred_density_proxy_on_sampled"]) for r in rows))
        pos_tail = float(np.mean([float(r["positive_occ_prob_mean"]) for r in rows[-5:]]))
        neg_tail = float(np.mean([float(r["negative_occ_prob_mean"]) for r in rows[-5:]]))
        budget_peak = float(max(float(r["new_occ_budget_proxy"]) for r in rows))
        cfg_budget = next(cfg.new_occ_budget for cfg in cfgs if cfg.name == cfg_name)
        ok = (
            tail_loss <= head_loss
            and density_peak <= 0.85
            and density_tail <= density_head
            and neg_tail < pos_tail
            and budget_peak <= cfg_budget + 1e-6
            and not any(truthy(r["has_nan"]) or truthy(r["has_inf"]) for r in rows)
        )
        summary = {
            "config_name": cfg_name,
            "loss_head": head_loss,
            "loss_tail": tail_loss,
            "density_head": density_head,
            "density_tail": density_tail,
            "density_peak": density_peak,
            "positive_occ_prob_tail": pos_tail,
            "negative_occ_prob_tail": neg_tail,
            "budget_peak": budget_peak,
            "budget_limit": cfg_budget,
        }
        if ok:
            passed.append(summary)
        else:
            still_high.append(summary)
    if passed:
        decision = {"decision_type": "B_RESIDUAL_SMOKE_PASS", "passed_configs": passed, "other_configs": still_high}
    elif any(row["selected_positive_count"] == 0 for row in grad_rows):
        decision = {"decision_type": "B_RESIDUAL_NO_POSITIVE_TARGET", "gradient_rows": grad_rows}
    else:
        decision = {"decision_type": "B_RESIDUAL_DENSITY_STILL_HIGH", "failed_configs": still_high, "gradient_rows": grad_rows}
    write_json(REPORTS_DIR / "branchB_residual_smoke_decision.json", decision)
    return audit_rows, decision, grad_rows


def phase_decision_and_report(branch_a_decision: dict[str, Any], branch_b_decision: dict[str, Any], branch_a_eval_rows: list[dict[str, Any]], branch_b_grad_rows: list[dict[str, Any]]) -> None:
    labels = []
    if branch_a_decision["decision_type"] == "A2_TRAIN_SIGNAL_AMPLIFIED_SAFE":
        labels.append("C1_A2_SAFE_TRAIN_AMPLIFIES")
    elif branch_a_decision["decision_type"] == "A2_TRAIN_NO_AMPLIFICATION":
        labels.append("C2_A2_SAFE_BUT_NO_AMPLIFICATION")
    elif branch_a_decision["decision_type"] == "A2_TRAIN_UNSAFE_DRIFT":
        labels.append("C3_A2_UNSAFE")
    if branch_b_decision["decision_type"] == "B_RESIDUAL_SMOKE_PASS":
        labels.append("C4_BRANCHB_RESIDUAL_SMOKE_PASS")
    elif branch_b_decision["decision_type"] in {"B_RESIDUAL_DENSITY_STILL_HIGH", "B_RESIDUAL_NUMERIC_FAILED"}:
        labels.append("C5_BRANCHB_MASK_BUG_FIXED_BUT_DENSITY_HIGH")
    elif branch_b_decision["decision_type"] == "B_RESIDUAL_NO_POSITIVE_TARGET":
        labels.append("C6_BRANCHB_NO_REAL_RESIDUAL_TARGET")
    if not labels:
        labels.append("C7_BOTH_WEAK")
    elif set(labels) == {"C2_A2_SAFE_BUT_NO_AMPLIFICATION", "C5_BRANCHB_MASK_BUG_FIXED_BUT_DENSITY_HIGH"}:
        labels.append("C7_BOTH_WEAK")

    decision = {
        "decision_labels": labels,
        "branchA_decision": branch_a_decision,
        "branchB_decision": branch_b_decision,
    }
    write_json(REPORTS_DIR / "sw12c_decision.json", decision)
    write_md(REPORTS_DIR / "sw12c_decision.md", json.dumps(normalize_export(decision), indent=2, ensure_ascii=False) + "\n")
    report_json = {
        "executive_summary": "SW-12C limited Branch A to A2 100-iter smoke and forced Branch B into strict residual teacher repair audit plus smoke only.",
        "branchA_decision": branch_a_decision,
        "branchB_decision": branch_b_decision,
        "decision_labels": labels,
    }
    report_md = "\n".join(
        [
            "# Stage SW-12C Safe Routing Smoke + Strict Residual Teacher Repair Audit",
            "",
            "1. Executive summary",
            f"- {report_json['executive_summary']}",
            "",
            "2. Why SW-12C follows SW-12B",
            "- A2 is a weak safe candidate, not a strong gain.",
            "- Branch B in SW-12B was not pure residual repair and is corrected here before any further smoke.",
            "",
            "3. Branch A A2 100-iter smoke",
            "- Branch A is subset diagnostic smoke only.",
            f"- decision: {branch_a_decision['decision_type']}",
            "",
            "4. Branch A safety/recovery analysis",
            "- Branch A remains judged by clean drift / false-positive / density / wrong-class gates.",
            "",
            "5. Branch B strict residual mask definition",
            "- Branch B positive repair mask requires teacher occupied, high confidence, student native free, and high-risk region.",
            "- No full teacher consistency and no reliable_teacher & risk_mask union is allowed.",
            "",
            "6. Branch B residual mask audit",
            f"- decision: {branch_b_decision['decision_type']}",
            "",
            "7. Branch B residual gradient check",
            "- Residual repair uses smoke-only gradient checks before any smoke training.",
            "",
            "8. Branch B residual 20-iter smoke",
            "- Branch B smoke only, not training gain and not official benchmark.",
            "",
            "9. Decision",
            f"- {labels}",
            "",
            "10. Safe claims",
            "- A2 is a weak safe candidate, not a strong gain",
            "- Branch B original implementation was not pure residual and is force-corrected here",
            "- Branch B smoke only, not training gain",
            "- not official benchmark",
            "- no claim of model improvement",
            "",
            "11. Limitations",
            "- SW-12C does not run 500/1000-iter training",
            "- Branch B remains an audit plus 20-iter smoke stage only",
            "",
            "12. Next unique action",
            f"- {labels[-1]}",
            "",
        ]
    )
    write_md(REPORTS_DIR / "stage_sw12c_safe_routing_and_residual_repair_report.md", report_md)
    write_json(REPORTS_DIR / "stage_sw12c_safe_routing_and_residual_repair_report.json", report_json)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.5, " + ".join(labels), ha="center", va="center", fontsize=16)
    ax.set_title("SW-12C subset diagnostic decision flow")
    fig.savefig(FIGURES_DIR / "sw12c_decision_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_tests() -> None:
    tests = {
        "test_sw12c_outputs_exist.py": '''from pathlib import Path\n\nBASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair")\n\n\ndef test_outputs_exist() -> None:\n    required = [\n        "sw12c_execution_manifest.json",\n        "sw12c_progress_state.json",\n        "sw12b_digest_for_sw12c.json",\n        "branchA_A2_100iter_train_metrics.csv",\n        "branchA_A2_100iter_eval.csv",\n        "branchA_A2_100iter_decision.json",\n        "branchB_residual_mask_audit.csv",\n        "branchB_residual_pair_distribution.csv",\n        "branchB_residual_gradient_check.csv",\n        "branchB_residual_20iter_smoke_metrics.csv",\n        "branchB_residual_smoke_decision.json",\n        "sw12c_decision.json",\n        "stage_sw12c_safe_routing_and_residual_repair_report.md",\n    ]\n    for name in required:\n        path = BASE / name\n        assert path.exists(), name\n        assert path.stat().st_size > 0, name\n''',
        "test_branchA_A2_schema.py": '''import csv\nfrom pathlib import Path\n\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchA_A2_100iter_eval.csv")\n\n\ndef test_branchA_schema() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    required = {"phase", "checkpoint_name", "variant_label", "perturbation_id", "horizon_s", "target_recovery", "A10_front_h6_recovery_ratio", "clean_false_positive_delta", "pred_gt_density_delta"}\n    assert required.issubset(rows[0].keys())\n''',
        "test_residual_mask_audit_schema.py": '''import csv\nfrom pathlib import Path\n\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_mask_audit.csv")\n\n\ndef test_residual_mask_audit_schema() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    required = {\n        "config_name", "split", "sample_index", "perturbation_id", "horizon_s",\n        "teacher_occupied_count_raw", "teacher_reliable_occupied_count_raw", "student_native_occupied_count_raw",\n        "residual_positive_count_raw", "high_risk_count_raw", "positive_repair_count_raw",\n        "selected_positive_count", "selected_negative_count", "ignored_count",\n        "teacher_reliable_free_count_raw", "negative_guard_count_raw",\n        "positive_negative_overlap_count", "positive_already_student_occupied_count", "positive_not_high_risk_count"\n    }\n    assert required.issubset(rows[0].keys())\n''',
        "test_residual_masks_are_strict.py": '''import csv\nfrom pathlib import Path\n\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_mask_audit.csv")\nLIMITS = {\n    "R1_residual_conf09_K128_neg2": (128, 2),\n    "R2_residual_conf09_K256_neg2": (256, 2),\n    "R3_residual_conf095_K128_neg2": (128, 2),\n    "R4_residual_conf095_K256_neg4": (256, 4),\n    "R5_residual_front_h46_only_conf095_K128_neg4": (128, 4),\n}\n\n\ndef test_strict_masks() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    for row in rows:\n        k, ratio = LIMITS[row["config_name"]]\n        pos = int(row["selected_positive_count"])\n        neg = int(row["selected_negative_count"])\n        assert int(row["positive_already_student_occupied_count"]) == 0\n        assert int(row["positive_negative_overlap_count"]) == 0\n        assert pos <= k\n        assert neg <= ratio * max(1, pos) if pos > 0 else neg == 0\n        if int(row["horizon_s"]) in {4, 6}:\n            assert int(row["positive_not_high_risk_count"]) == 0\n''',
        "test_no_full_teacher_consistency.py": '''from pathlib import Path\n\nMASK_DEF = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_mask_definition.md")\nLOSS_DEF = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_loss_design.md")\n\n\ndef test_no_full_teacher_consistency() -> None:\n    mask_text = MASK_DEF.read_text(encoding="utf-8").lower()\n    loss_text = LOSS_DEF.read_text(encoding="utf-8").lower()\n    assert "no reliable_teacher & risk_mask union" in mask_text\n    assert "no full teacher consistency" in loss_text\n    assert "all reliable teacher voxels" not in loss_text\n    assert "all selected voxels" not in loss_text\n''',
        "test_residual_smoke_schema.py": '''import csv\nfrom pathlib import Path\n\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_20iter_smoke_metrics.csv")\n\n\ndef test_residual_smoke_schema() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    required = {"config_name", "iter", "total_loss", "L_residual_occ", "L_residual_sem", "L_negative_empty", "L_density_budget", "pred_density_proxy_on_sampled", "positive_occ_prob_mean", "negative_occ_prob_mean", "new_occ_budget_proxy"}\n    if "skipped_reason" not in rows[0]:\n        assert required.issubset(rows[0].keys())\n''',
        "test_decision_schema.py": '''import json\nfrom pathlib import Path\n\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/sw12c_decision.json")\n\n\ndef test_decision_schema() -> None:\n    obj = json.loads(PATH.read_text(encoding="utf-8"))\n    assert "decision_labels" in obj\n    assert "branchA_decision" in obj\n    assert "branchB_decision" in obj\n    assert obj["decision_labels"]\n''',
        "test_no_false_claims.py": '''from pathlib import Path\n\nREPORT = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/stage_sw12c_safe_routing_and_residual_repair_report.md")\n\n\ndef test_no_false_claims() -> None:\n    text = REPORT.read_text(encoding="utf-8").lower()\n    assert "subset diagnostic" in text\n    assert "not official benchmark" in text\n    banned = ["official benchmark result", "final model improvement", "beats the paper", "production ready"]\n    for phrase in banned:\n        assert phrase not in text\n''',
    }
    for name, text in tests.items():
        write_md(TESTS_DIR / name, text)


def main() -> None:
    args = parse_args()
    np.Inf = np.inf
    ensure_dirs()
    seed_everything(args.seed)
    progress = init_progress_state()
    manifest = init_execution_manifest()

    def run_phase(name: str, fn):
        phase_start(progress, manifest, name)
        try:
            result = fn()
            phase_end(progress, manifest, name, "done")
            return result
        except Exception as exc:
            phase_end(progress, manifest, name, "failed", error=str(exc))
            raise

    run_phase("phase1_sw12b_digest", lambda: phase1_digest(manifest))
    branch_a_eval_rows, branch_a_decision = run_phase("phaseA_A2_100iter_smoke", lambda: phase_branch_a(manifest, args))
    branch_b_audit_rows, branch_b_decision, branch_b_grad_rows = run_phase("phaseB_residual_audit_and_smoke", lambda: phase_branch_b(manifest, args))
    run_phase("phaseC_decision_and_report", lambda: phase_decision_and_report(branch_a_decision, branch_b_decision, branch_a_eval_rows, branch_b_grad_rows))
    run_phase("phaseE_tests_write", write_tests)
    progress["status"] = "complete"
    progress["end_time"] = now_iso()
    write_json(PROGRESS_PATH, progress)
    manifest["end_time"] = now_iso()
    write_json(EXECUTION_MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
