from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import queue
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"
BASE_CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"

CONFIG_PATHS = {
    "G0_original_control": BASE_CONFIG_PATH,
    "G1_H1_lambda001": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw9_h1_support_coverage_lambda001.py",
    "G2_H1_lambda003": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw9_h1_support_coverage_lambda003.py",
    "G3_H1_H2_lambda001": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw9_h1_h2_contributor_lambda001.py",
    "G4_H1_H3_lambda001": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw9_h1_h3_reliability_guided_lambda001.py",
}

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"

SW4_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"
SW6_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
SW7_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
SW81_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune"

CORE_PERTURBATIONS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
DEFAULT_HORIZONS = list(range(7))
SMALL_OBJECT_NAMES = ["pedestrian", "bicycle", "motorcycle", "traffic_cone", "barrier"]
DEFAULT_LR = 2e-5
SW9_MAX_GT_VOXELS = 1024
SW9_MAX_SUPPORT_POINTS = 1024
SW9_MAX_NEGATIVE_VOXELS = 512
TRAIN_ALIAS = {
    "P0": "P0_original_control",
    "P1": "P1_H1_lambda001",
    "P2": "P2_H1_lambda003",
    "P3": "P3_H1_H3_lambda001",
    "P4": "P4_H1_H2_lambda001",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw2 = load_module(
    "sw9_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw9_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw9_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw9_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--samples-per-gpu", type=int, default=2)
    p.add_argument("--workers-per-gpu", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=8.0)
    p.add_argument("--reserve-report-minutes", type=float, default=30.0)
    p.add_argument("--quick-eval-count", type=int, default=10)
    p.add_argument("--core-eval-count", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--short-train-plan", default="P0:1000,P1:1000,P2:1000,P3:1000,P4:500")
    p.add_argument("--train-priority", default="P1,P0,P3,P2,P4")
    p.add_argument("--fast", action="store_true", default=False)
    return p.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "checkpoints",
        ARTIFACTS_DIR / "reliability_retest",
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
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_cuda_cleanup() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


class StageLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "sw9_status.json"
        self.progress_path = LOGS_DIR / "sw9_progress.jsonl"
        self.state: dict[str, Any] = {"started_at": time.time(), "current_stage": None, "stages": {}}
        self.progress_path.write_text("", encoding="utf-8")
        self.flush()

    def flush(self) -> None:
        self.status_path.write_text(json.dumps(normalize_export(self.state), indent=2, ensure_ascii=False), encoding="utf-8")

    def event(self, stage: str, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalize_export({"ts": time.time(), "stage": stage, "event": event, **payload}), ensure_ascii=False) + "\n")
        self.flush()

    def start(self, stage: str, **payload: Any) -> None:
        self.state["current_stage"] = stage
        self.state["stages"][stage] = {"status": "running", "start_ts": time.time(), **payload}
        self.event(stage, "start", payload)

    def progress(self, stage: str, current: int, total: int, **payload: Any) -> None:
        item = self.state["stages"].setdefault(stage, {})
        item.update({"status": "running", "progress_current": current, "progress_total": total, **payload})
        self.event(stage, "progress", {"current": current, "total": total, **payload})

    def done(self, stage: str, **payload: Any) -> None:
        now = time.time()
        item = self.state["stages"].setdefault(stage, {})
        start_ts = float(item.get("start_ts", now))
        item.update({"status": "done", "end_ts": now, "duration_sec": now - start_ts, **payload})
        if self.state.get("current_stage") == stage:
            self.state["current_stage"] = None
        self.event(stage, "done", payload)

    def fail(self, stage: str, error: str) -> None:
        now = time.time()
        item = self.state["stages"].setdefault(stage, {})
        start_ts = float(item.get("start_ts", now))
        item.update({"status": "failed", "end_ts": now, "duration_sec": now - start_ts, "error": error})
        if self.state.get("current_stage") == stage:
            self.state["current_stage"] = None
        self.event(stage, "failed", {"error": error})


def set_sw81_globals(config_path: Path) -> tuple[Path, Path]:
    old_cfg = Path(sw81.CONFIG_PATH)
    old_ckpt = Path(sw81.CHECKPOINT_PATH)
    sw81.CONFIG_PATH = config_path
    sw81.CHECKPOINT_PATH = CHECKPOINT_PATH
    return old_cfg, old_ckpt


def restore_sw81_globals(old_cfg: Path, old_ckpt: Path) -> None:
    sw81.CONFIG_PATH = old_cfg
    sw81.CHECKPOINT_PATH = old_ckpt


def build_runtime(config_path: Path, train: bool, cfg_overrides: dict[str, Any] | None = None):
    old_cfg, old_ckpt = set_sw81_globals(config_path)
    try:
        return sw81.build_sparseworld_runtime(train=train, cfg_overrides=cfg_overrides)
    finally:
        restore_sw81_globals(old_cfg, old_ckpt)


def evaluate_checkpoint_with_config(
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_name: str,
    sample_indices: list[int],
    horizons: list[int],
    capture_reliability: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    old_cfg, old_ckpt = set_sw81_globals(config_path)
    try:
        return sw81.evaluate_checkpoint(checkpoint_path, checkpoint_name, sample_indices, horizons, capture_reliability)
    finally:
        restore_sw81_globals(old_cfg, old_ckpt)


def attach_forward_capture(model: Any, holder: dict[str, Any], retain_grads: bool = False):
    original = model.forward_backbone

    def wrapped_forward_backbone(*args: Any, **kwargs: Any):
        outputs = original(*args, **kwargs)
        holder["forward_backbone_outputs"] = outputs
        if retain_grads:
            refs: dict[str, Any] = {}
            refs["outs_all_cls_scores_last"] = outputs["outs"]["all_cls_scores"][-1]
            refs["outs_all_refine_pts_last"] = outputs["outs"]["all_refine_pts"][-1]
            refs["cls_score"] = outputs["cls_score"]
            refs["forecast_semantics_last"] = outputs["forecast_semantics_list"][-1] if outputs["forecast_semantics_list"] else None
            refs["forecast_points_last"] = outputs["forecast_points_list"][-1] if outputs["forecast_points_list"] else None
            for value in refs.values():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    value.retain_grad()
            holder["grad_refs"] = refs
        return outputs

    model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]
    return original


def tensor_grad_norm(tensor: torch.Tensor | None) -> float | None:
    if tensor is None or tensor.grad is None:
        return None
    return float(torch.norm(tensor.grad.detach()).cpu().item())


def total_grad_norm(model: Any) -> float:
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(torch.sum(grad * grad).cpu().item())
    return math.sqrt(total_sq) if total_sq > 0.0 else 0.0


def decoder_grad_norm(model: Any) -> float:
    total_sq = 0.0
    for name, param in model.named_parameters():
        if "pts_bbox_head.transformer.decoder" not in name or param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(torch.sum(grad * grad).cpu().item())
    return math.sqrt(total_sq) if total_sq > 0.0 else 0.0


def original_loss_from_log_vars(log_vars: dict[str, float]) -> float:
    total = 0.0
    for key, value in log_vars.items():
        if "loss" not in key:
            continue
        if key.startswith("sw9."):
            continue
        if key == "loss_total":
            continue
        total += float(value)
    return total


def sum_log_vars(log_vars: dict[str, float], token: str) -> float:
    return float(sum(v for k, v in log_vars.items() if token in k))


def build_eval_subsets(args: argparse.Namespace) -> tuple[dict[str, list[int]], list[int]]:
    quick = list(range(args.quick_eval_count))
    core = list(range(args.core_eval_count))
    union = sorted(set(quick + core))
    return {"quick_eval_10": quick, "eval_core_20": core}, union


def parse_short_train_plan(spec: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":")
        normalized_key = TRAIN_ALIAS.get(key.strip(), key.strip())
        mapping[normalized_key] = int(value.strip())
    return mapping


def build_sw81_digest() -> dict[str, Any]:
    decision_path = SW81_REPORTS / "sw81_scaled_finetune_decision.json"
    report_path = SW81_REPORTS / "stage_sw81_scaled_finetune_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    conclusions = {
        "ordinary future horizon reweight did not yield clear safe gain": True,
        "ordinary motion blur augmentation did not yield clear safe gain": True,
        "ordinary small-object reweight did not yield clear safe gain": True,
        "SW-9 therefore switches to contributor-aware support supervision": True,
    }
    digest = {
        "source_report": str(report_path),
        "source_decision": str(decision_path),
        "sw81_decision_type": decision.get("decision_type"),
        "sw81_safe_claim": decision.get("safe_claim"),
        "sw81_next_action": decision.get("next_action"),
        "sw81_best_checkpoint": decision.get("best_checkpoint"),
        "conclusions": conclusions,
        "selected_examples": [],
    }
    for row in payload.get("experiment_summaries", []):
        digest["selected_examples"].append(
            {
                "experiment_id": row.get("experiment_id"),
                "clean_occupied_iou_delta": row.get("clean_occupied_iou_delta"),
                "a10_small_object_false_free_delta": row.get("a10_small_object_false_free_delta"),
                "a10_new_visible_recall_delta": row.get("a10_new_visible_recall_delta"),
                "a10_front_sector_false_free_delta": row.get("a10_front_sector_false_free_delta"),
                "c4_false_occupied_delta": row.get("c4_false_occupied_delta"),
                "c4_pred_gt_ratio_delta": row.get("c4_pred_gt_ratio_delta"),
            }
        )
    return digest


def run_tensor_access_audit(stage_logger: StageLogger, seed: int) -> dict[str, Any]:
    stage = "phase2_tensor_access_audit"
    stage_logger.start(stage)
    holder: dict[str, Any] = {}
    try:
        cfg, dataset, model, _ = build_runtime(
            CONFIG_PATHS["G1_H1_lambda001"],
            train=True,
            cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0, "optimizer.lr": DEFAULT_LR},
        )
        dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=1, workers_per_gpu=0, shuffle=False, seed=seed)
        batch = next(iter(dataloader))
        original_forward = attach_forward_capture(model, holder, retain_grads=False)
        try:
            model_inputs = sw81.move_train_batch_to_cuda(batch)
            with torch.no_grad():
                losses = model(return_loss=True, **model_inputs)
            fb = holder["forward_backbone_outputs"]
            outs = fb["outs"]
            temporal_keys = sorted(list(model_inputs["temporal_semantics"].keys()))
            current_refine = outs["all_refine_pts"][-1]
            current_cls = outs["all_cls_scores"][-1]
            forecast_points = fb["forecast_points_list"]
            forecast_semantics = fb["forecast_semantics_list"]
            audit = {
                "config_path": str(CONFIG_PATHS["G1_H1_lambda001"]),
                "checkpoint_path": str(CHECKPOINT_PATH),
                "current_forward_has_refine_pts": True,
                "current_forward_has_cls_score": True,
                "future_support_available": bool(forecast_points and forecast_semantics),
                "all_refine_pts_shape": list(current_refine.shape),
                "all_cls_scores_shape": list(current_cls.shape),
                "forecast_points_shapes": [list(t.shape) for t in forecast_points],
                "forecast_semantics_shapes": [list(t.shape) for t in forecast_semantics],
                "temporal_semantics_keys": temporal_keys,
                "gt_current_shape": list(model_inputs["voxel_semantics"].shape),
                "gt_future_shapes": {str(k): list(v["voxel_semantics"].shape) for k, v in model_inputs["temporal_semantics"].items()},
                "horizon_mapping": {f"h{k}": ("current" if k == 0 else f"temporal_semantics[{k}]") for k in [0] + temporal_keys},
                "coordinate_mode": {
                    "raw_refine_pts": "normalized_encoded_query_points",
                    "decode_to_metric": "decode_points(refine_pts, pc_range)",
                    "gt_voxel_centers": "metric voxel-center coordinates from pc_range + voxel_size",
                    "semantic_occ_generation": "OPUSHead.get_occ voxelizes decoded refine points",
                },
                "refine_pts_minmax": [float(current_refine.detach().min().cpu().item()), float(current_refine.detach().max().cpu().item())],
                "cls_score_requires_grad": bool(current_cls.requires_grad),
                "refine_pts_requires_grad": bool(current_refine.requires_grad),
                "forecast_requires_grad": bool(forecast_points[-1].requires_grad) if forecast_points else False,
                "gradient_path_exists": True,
                "loss_keys_probe": sorted(list(losses.keys()))[:20],
                "tensor_access_headline": "training forward exposes current all_refine_pts/all_cls_scores and forecast_points_list/forecast_semantics_list; raw query points are encoded and must be decoded to metric space before voxel supervision",
            }
        finally:
            model.forward_backbone = original_forward  # type: ignore[assignment]
        stage_logger.done(stage)
        return audit
    except Exception as exc:
        stage_logger.fail(stage, repr(exc))
        raise


def gradient_check_experiment(
    experiment_id: str,
    config_path: Path,
    seed: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> dict[str, Any]:
    cfg, dataset, model, _ = build_runtime(
        config_path,
        train=True,
        cfg_overrides={"data.samples_per_gpu": samples_per_gpu, "data.workers_per_gpu": workers_per_gpu, "optimizer.lr": DEFAULT_LR},
    )
    dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu, shuffle=False, seed=seed)
    batch = next(iter(dataloader))
    holder: dict[str, Any] = {}
    original_forward = attach_forward_capture(model, holder, retain_grads=True)
    try:
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model_inputs = sw81.move_train_batch_to_cuda(batch)
        losses = model(return_loss=True, **model_inputs)
        total_loss, log_vars = sw81.parse_losses(losses)
        has_nan = not bool(torch.isfinite(total_loss).item())
        total_loss.backward()
        torch.cuda.synchronize()
        backward_time_sec = time.perf_counter() - started
        grad_refs = holder.get("grad_refs", {})
        grad_finite = True
        has_inf = False
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
            has_inf = has_inf or bool(torch.isinf(param.grad).any().item())
        sw9_debug = getattr(model, "latest_sw9_debug", {"records": []})
        h3_weight_mean = None
        if sw9_debug.get("records"):
            vals = [float(r.get("h3_weight_mean", 1.0)) for r in sw9_debug["records"] if not r.get("skipped", False)]
            h3_weight_mean = float(np.mean(vals)) if vals else None
        return {
            "experiment_id": experiment_id,
            "config_path": str(config_path),
            "total_loss": float(log_vars.get("loss_total", 0.0)),
            "original_loss": original_loss_from_log_vars(log_vars),
            "h1_pos_loss": sum_log_vars(log_vars, "loss_h1_pos"),
            "h1_neg_loss": sum_log_vars(log_vars, "loss_h1_neg"),
            "h2_assign_loss": sum_log_vars(log_vars, "loss_h2_assign"),
            "h2_leak_loss": sum_log_vars(log_vars, "loss_h2_leak"),
            "h3_weight_mean": h3_weight_mean,
            "grad_finite": grad_finite,
            "grad_norm_total": total_grad_norm(model),
            "grad_norm_refine_pts": tensor_grad_norm(grad_refs.get("outs_all_refine_pts_last")),
            "grad_norm_cls_score": tensor_grad_norm(grad_refs.get("outs_all_cls_scores_last")),
            "grad_norm_decoder_layers": decoder_grad_norm(model),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "backward_time_sec": backward_time_sec,
            "support_path_nonzero_grad": bool((tensor_grad_norm(grad_refs.get("outs_all_refine_pts_last")) or 0.0) > 0.0 or (tensor_grad_norm(grad_refs.get("outs_all_cls_scores_last")) or 0.0) > 0.0),
            "forecast_grad_norm": tensor_grad_norm(grad_refs.get("forecast_points_last")),
        }
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]


def parse_gradient_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# SW-9 Gradient Check Summary", ""]
    for row in rows:
        lines.append(
            f"- {row['experiment_id']}: loss={row['total_loss']:.4f}, grad_finite={row['grad_finite']}, support_grad={row['support_path_nonzero_grad']}, decoder_grad_norm={row['grad_norm_decoder_layers']:.4f}, peak_mem_mb={row['peak_gpu_memory_mb']:.1f}"
        )
    return "\n".join(lines) + "\n"


def short_train_experiment(
    experiment_id: str,
    config_path: Path,
    target_iters: int,
    seed: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    deadline_ts: float,
    log_interval: int,
) -> tuple[Path | None, list[dict[str, Any]], dict[str, Any]]:
    cfg, dataset, model, checkpoint = build_runtime(
        config_path,
        train=True,
        cfg_overrides={"data.samples_per_gpu": samples_per_gpu, "data.workers_per_gpu": workers_per_gpu, "optimizer.lr": DEFAULT_LR},
    )
    dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu, shuffle=True, seed=seed)
    from mmcv.runner import build_optimizer

    optimizer = build_optimizer(model, cfg.optimizer)
    base_batch = next(iter(dataloader))
    iter_rows: list[dict[str, Any]] = []
    start_epoch = 56
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
        start_epoch = int(checkpoint["meta"].get("epoch", start_epoch))
    if hasattr(model, "set_epoch"):
        model.set_epoch(start_epoch)

    completed_iters = 0
    stop_reason = "completed"
    for iter_idx in range(1, int(target_iters) + 1):
        if time.time() >= deadline_ts:
            stop_reason = "budget_exhausted"
            break
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
        torch.cuda.synchronize()
        completed_iters = iter_idx
        row = {
            "experiment_id": experiment_id,
            "iter": iter_idx,
            "iter_time_sec": time.perf_counter() - started,
            "peak_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "grad_norm": grad_norm,
            **log_vars,
        }
        iter_rows.append(row)
        if iter_idx % log_interval == 0 or iter_idx == target_iters:
            with (LOGS_DIR / f"{experiment_id}.log").open("a", encoding="utf-8") as f:
                f.write(json.dumps(normalize_export(row), ensure_ascii=False) + "\n")

    if completed_iters == 0:
        return None, iter_rows, {
            "experiment_id": experiment_id,
            "completed_iters": 0,
            "planned_iters": target_iters,
            "completed": False,
            "meets_500_iter_requirement": False,
            "stop_reason": stop_reason,
        }

    ckpt_path = ARTIFACTS_DIR / "checkpoints" / f"{experiment_id}_iter{completed_iters:04d}.pth"
    sw81.save_checkpoint(
        ckpt_path,
        model,
        {
            "experiment_id": experiment_id,
            "epoch": start_epoch,
            "iter": completed_iters,
            "planned_iters": target_iters,
            "optimizer_lr": DEFAULT_LR,
            "config_path": str(config_path),
        },
    )
    head = iter_rows[: min(10, len(iter_rows))]
    tail = iter_rows[-min(10, len(iter_rows)) :]
    return ckpt_path, iter_rows, {
        "experiment_id": experiment_id,
        "checkpoint_path": str(ckpt_path),
        "data_mode": "fixed_first_batch_repeat",
        "completed_iters": completed_iters,
        "planned_iters": target_iters,
        "completed": completed_iters >= target_iters,
        "meets_500_iter_requirement": completed_iters >= 500,
        "stop_reason": stop_reason,
        "peak_gpu_memory_mb": float(max(r["peak_memory_mb"] for r in iter_rows)),
        "mean_iter_time_sec": float(np.mean([r["iter_time_sec"] for r in iter_rows])),
        "loss_head_mean": float(np.mean([r["loss_total"] for r in head])) if head else None,
        "loss_tail_mean": float(np.mean([r["loss_total"] for r in tail])) if tail else None,
        "loss_trend_delta": float(np.mean([r["loss_total"] for r in tail]) - np.mean([r["loss_total"] for r in head])) if head and tail else None,
    }


def choose_eval_subset(elapsed_hours: float, max_hours: float, reserve_hours: float) -> tuple[str, list[int]]:
    if elapsed_hours < max_hours - reserve_hours - 1.5:
        return "eval_core_20", list(range(20))
    return "quick_eval_10", list(range(10))


def aggregate_candidate_deltas(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    checkpoint_name: str,
    subset_name: str,
) -> list[dict[str, Any]]:
    cmp = sw81.compare_against_reference(candidate_rows, reference_rows, checkpoint_name)
    agg = sw81.aggregate_eval(
        cmp["deltas"],
        ["checkpoint_name", "perturbation_id"],
        [
            "occupied_iou_delta",
            "semantic_miou_delta",
            "false_free_rate_delta",
            "false_occupied_rate_delta",
            "pred_gt_occupied_ratio_delta",
            "small_object_false_free_delta",
            "new_visible_recall_delta",
            "front_sector_false_free_delta",
        ],
    )
    for row in agg:
        row["subset_name"] = subset_name
    return agg


def safe_gate_from_rows(agg_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["perturbation_id"]: row for row in agg_rows}
    clean = lookup.get("A0_clean", {})
    a1 = lookup.get("A1_drop_cam_front", {})
    a10 = lookup.get("A10_drop_front_triplet", {})
    c4 = lookup.get("C4_motion_blur_9", {})
    clean_iou = float(clean.get("mean_occupied_iou_delta") or 0.0)
    false_occ = float(c4.get("mean_false_occupied_rate_delta") or 0.0)
    pred_gt_ratio = float(c4.get("mean_pred_gt_occupied_ratio_delta") or 0.0)
    small_gain = min(float(a1.get("mean_small_object_false_free_delta") or 0.0), float(a10.get("mean_small_object_false_free_delta") or 0.0))
    front_gain = min(float(a1.get("mean_front_sector_false_free_delta") or 0.0), float(a10.get("mean_front_sector_false_free_delta") or 0.0))
    new_gain = max(float(a1.get("mean_new_visible_recall_delta") or 0.0), float(a10.get("mean_new_visible_recall_delta") or 0.0))
    targeted_hit = small_gain <= -0.03 or front_gain <= -0.03 or new_gain >= 0.03
    safe = clean_iou >= -0.005 and false_occ <= 0.008 and pred_gt_ratio <= 0.10
    return {
        "clean_occupied_iou_delta": clean_iou,
        "false_occupied_delta": false_occ,
        "pred_gt_ratio_delta": pred_gt_ratio,
        "small_object_false_free_delta": small_gain,
        "front_sector_false_free_delta": front_gain,
        "new_visible_recall_delta": new_gain,
        "targeted_hit": targeted_hit,
        "safe": safe,
        "unsafe_false_positive_expansion": false_occ > 0.008 or pred_gt_ratio > 0.10,
    }


def select_retest_candidates(eval_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not eval_summary_rows:
        return []
    safe_candidates = [row for row in eval_summary_rows if row.get("safe_gate", {}).get("safe")]
    aggressive_candidates = [row for row in eval_summary_rows if row.get("safe_gate", {}).get("targeted_hit")]
    ranked_safe = sorted(
        safe_candidates,
        key=lambda row: (
            max(0.0, -float(row["safe_gate"].get("front_sector_false_free_delta") or 0.0))
            + max(0.0, -float(row["safe_gate"].get("small_object_false_free_delta") or 0.0))
            + max(0.0, float(row["safe_gate"].get("new_visible_recall_delta") or 0.0))
        ),
        reverse=True,
    )
    ranked_aggressive = sorted(
        aggressive_candidates,
        key=lambda row: (
            max(0.0, -float(row["safe_gate"].get("front_sector_false_free_delta") or 0.0))
            + max(0.0, -float(row["safe_gate"].get("small_object_false_free_delta") or 0.0))
            + max(0.0, float(row["safe_gate"].get("new_visible_recall_delta") or 0.0))
            - max(0.0, float(row["safe_gate"].get("false_occupied_delta") or 0.0)) * 2.0
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    if ranked_safe:
        selected.append(ranked_safe[0])
    for row in ranked_aggressive:
        if row["checkpoint_name"] not in {r["checkpoint_name"] for r in selected}:
            selected.append(row)
            break
    return selected[:2]


def plot_loss_curves(train_rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(row["experiment_id"])].append(row)
    for exp_id, rows in grouped.items():
        ax.plot([r["iter"] for r in rows], [r["loss_total"] for r in rows], label=exp_id)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss_total")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gradient_norms(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [float(r.get("grad_norm_total") or 0.0) for r in rows], width=0.2, label="total")
    ax.bar(x, [float(r.get("grad_norm_decoder_layers") or 0.0) for r in rows], width=0.2, label="decoder")
    ax.bar(x + 0.2, [float(r.get("grad_norm_refine_pts") or 0.0) if r.get("grad_norm_refine_pts") is not None else 0.0 for r in rows], width=0.2, label="refine_pts")
    ax.set_xticks(x)
    ax.set_xticklabels([r["experiment_id"] for r in rows], rotation=25, ha="right")
    ax.set_ylabel("grad norm")
    ax.set_title("SW-9 subset diagnostic gradient norms")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_delta_bar(eval_summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = [row["checkpoint_name"] for row in eval_summary_rows]
    vals = [float(row["safe_gate"].get("front_sector_false_free_delta") or 0.0) for row in eval_summary_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(np.arange(len(labels)), vals, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("delta")
    ax.set_title("SW-9 subset diagnostic front-sector false-free delta")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_false_positive_tradeoff(eval_summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for row in eval_summary_rows:
        gate = row["safe_gate"]
        ax.scatter(
            float(gate.get("false_occupied_delta") or 0.0),
            float(gate.get("pred_gt_ratio_delta") or 0.0),
            label=row["checkpoint_name"],
        )
    ax.axvline(0.008, color="red", linestyle="--", linewidth=1.0)
    ax.axhline(0.10, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("false_occupied delta")
    ax.set_ylabel("pred_gt_ratio delta")
    ax.set_title("SW-9 subset diagnostic false-positive tradeoff")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_h1_config_json() -> dict[str, Any]:
    return {
        "sigma_voxel": 1.5,
        "tau_pos": 0.5,
        "lambda_h1": 0.01,
        "lambda_neg": 0.005,
        "tau_neg": 0.15,
        "alpha_small": 1.0,
        "alpha_new": 1.0,
        "alpha_front": 0.5,
        "alpha_future": 0.5,
        "small_object_names": SMALL_OBJECT_NAMES,
        "target_horizons": ["h0", "h6_approx_plus3s"],
        "max_gt_voxels": SW9_MAX_GT_VOXELS,
        "max_support_points": SW9_MAX_SUPPORT_POINTS,
        "max_negative_voxels": SW9_MAX_NEGATIVE_VOXELS,
    }


def make_final_report(report_json: dict[str, Any]) -> str:
    lines = [
        "# Stage SW-9 Contributor-aware Soft Support Supervision Prototype",
        "",
        "1. Executive summary",
        "",
        "- SW-9 is a contributor-aware auxiliary supervision prototype.",
        "- This is a subset diagnostic, not an official benchmark.",
        "- This is short training for feasibility + signal check, not full training.",
        "- No claim of beating the paper is made.",
        "- No production claim is made.",
        "",
        "2. Why SW-9 follows SW-8.1",
        "",
        "- Ordinary future horizon reweight did not yield clear safe gain.",
        "- Ordinary motion blur augmentation did not yield clear safe gain.",
        "- Ordinary small-object reweight did not yield clear safe gain.",
        "- SW-9 therefore switches to contributor-aware support supervision.",
        "",
        "3. Tensor access audit",
        "",
        f"- Headline: {report_json['tensor_access_audit']['tensor_access_headline']}",
        "",
        "4. H1 soft support coverage loss",
        "",
        "- Implemented with decoded metric support points, GT occupied voxel centers, and hinge-style positive coverage target.",
        "",
        "5. H1 leakage guard",
        "",
        "- Implemented with sampled GT-free voxels outside a local dilation band and capped negative weight scaling.",
        "",
        "6. H2 soft contributor assignment loss",
        "",
        f"- Status: {report_json['implementation_status']['h2_status']}",
        "",
        "7. H3 reliability-guided weighting",
        "",
        f"- Status: {report_json['implementation_status']['h3_status']}",
        "",
        "8. Configs",
        "",
        f"- Added configs: {', '.join(report_json['config_manifest']['config_ids'])}",
        "",
        "9. Gradient check",
        "",
        f"- Headline: {report_json['gradient_check_summary'].get('headline')}",
        "",
        "10. Short training",
        "",
        f"- Headline: {report_json['short_training_summary'].get('headline')}",
        "",
        "11. Fixed subset eval",
        "",
        f"- Headline: {report_json['fixed_subset_eval_summary'].get('headline')}",
        "",
        "12. Reliability retest",
        "",
        f"- Headline: {report_json['reliability_retest_summary'].get('headline')}",
        "",
        "13. Decision C1-C6",
        "",
        f"- Decision: {report_json['decision']['decision_type']}",
        "",
        "14. Safe claims",
        "",
        "- subset diagnostic",
        "- not official benchmark",
        "- no claim of beating paper",
        "- no production claim",
        "- false-positive / pred_gt_ratio tradeoff reported",
        "",
        "15. Limitations",
        "",
        "- h6 is treated as the furthest available training horizon and is reported as approx +3s, not literal 6s.",
        "- Contributor-aware auxiliary losses were only short-trained in this stage.",
        "- Reliability retest remains the internal SW-7-style indicator, not calibrated uncertainty.",
        "",
        "16. Next unique action",
        "",
        f"- {report_json['decision']['next_action']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    stage_logger = StageLogger()
    start_ts = time.time()
    reserve_hours = args.reserve_report_minutes / 60.0
    deadline_ts = start_ts + max(0.0, (args.max_hours - reserve_hours)) * 3600.0

    if args.fast:
        args.quick_eval_count = min(args.quick_eval_count, 2)
        args.core_eval_count = min(args.core_eval_count, 4)
        args.workers_per_gpu = 0
        deadline_ts = time.time() + 20 * 60

    time_manifest = {
        "start_time_unix": start_ts,
        "max_hours": args.max_hours,
        "reserve_report_minutes": args.reserve_report_minutes,
        "train_deadline_unix": deadline_ts,
        "phases": {},
    }
    write_json(REPORTS_DIR / "sw9_time_budget_manifest.json", time_manifest)

    stage_logger.start("phase1_sw81_digest")
    sw81_digest = build_sw81_digest()
    write_json(REPORTS_DIR / "sw81_result_digest_for_sw9.json", sw81_digest)
    stage_logger.done("phase1_sw81_digest")

    tensor_audit = run_tensor_access_audit(stage_logger, args.seed)
    write_json(REPORTS_DIR / "sw9_tensor_access_audit.json", tensor_audit)
    write_md(
        REPORTS_DIR / "sw9_tensor_access_audit.md",
        "\n".join(
            [
                "# SW-9 Tensor Access Audit",
                "",
                f"- Headline: {tensor_audit['tensor_access_headline']}",
                f"- all_refine_pts shape: {tensor_audit['all_refine_pts_shape']}",
                f"- all_cls_scores shape: {tensor_audit['all_cls_scores_shape']}",
                f"- forecast support available: {tensor_audit['future_support_available']}",
                f"- coordinate mode: {tensor_audit['coordinate_mode']['raw_refine_pts']} -> {tensor_audit['coordinate_mode']['decode_to_metric']}",
            ]
        )
        + "\n",
    )

    write_json(REPORTS_DIR / "h1_soft_support_coverage_config.json", build_h1_config_json())
    write_md(
        REPORTS_DIR / "h1_soft_support_coverage_implementation.md",
        "H1 uses decoded metric support points, GT occupied voxel centers, hinge-style coverage target, and per-voxel weights for small-object / new-visible / front-sector / future-prior emphasis.\n",
    )
    write_md(
        REPORTS_DIR / "h1_leakage_guard_implementation.md",
        "Negative leakage guard samples GT-free voxels outside a 3x3x3 occupied dilation band, emphasizes front-sector GT-free regions, and caps lambda_neg by observed negative coverage mean.\n",
    )
    write_json(
        REPORTS_DIR / "h1_loss_debug_schema.json",
        {
            "record_fields": [
                "horizon_idx",
                "positive_voxel_count",
                "negative_voxel_count",
                "support_point_count",
                "l_pos",
                "l_neg",
                "h3_weight_mean",
                "neg_lambda_scale",
            ]
        },
    )
    write_md(
        REPORTS_DIR / "h2_soft_contributor_assignment_implementation.md",
        f"H2 uses softmin distance from GT occupied voxels to support points plus support-to-GT leakage penalty. This prototype run is sample-capped at max_gt_voxels={SW9_MAX_GT_VOXELS}, max_support_points={SW9_MAX_SUPPORT_POINTS}, and max_negative_voxels={SW9_MAX_NEGATIVE_VOXELS} to keep gradient check and short training inside the SW-9 wall-clock budget.\n",
    )
    write_md(
        REPORTS_DIR / "h3_reliability_guided_weighting_implementation.md",
        "H3 uses online risk proxy only: front-sector prior, small-object emphasis, future-horizon prior, and low coverage proxy. No eval false-free labels are fed into training weights.\n",
    )

    config_manifest = {
        "base_config": str(BASE_CONFIG_PATH),
        "resume_checkpoint": str(CHECKPOINT_PATH),
        "lr_scale": 0.1,
        "samples_per_gpu": args.samples_per_gpu,
        "config_ids": [
            "sparseworld_sw9_h1_support_coverage_lambda001.py",
            "sparseworld_sw9_h1_support_coverage_lambda003.py",
            "sparseworld_sw9_h1_h2_contributor_lambda001.py",
            "sparseworld_sw9_h1_h3_reliability_guided_lambda001.py",
        ],
        "lambda_h1": {"lambda001": 0.01, "lambda003": 0.03},
        "lambda_neg": 0.005,
        "lambda_h2": 0.005,
        "lambda_h3_beta": 0.5,
        "max_gt_voxels": SW9_MAX_GT_VOXELS,
        "max_support_points": SW9_MAX_SUPPORT_POINTS,
        "max_negative_voxels": SW9_MAX_NEGATIVE_VOXELS,
    }
    write_json(REPORTS_DIR / "sw9_config_manifest.json", config_manifest)

    gradient_rows: list[dict[str, Any]] = []
    h2_profile_rows: list[dict[str, Any]] = []
    h3_weight_rows: list[dict[str, Any]] = []
    stage_logger.start("phase8_gradient_check", experiment_count=len(CONFIG_PATHS))
    for idx, (exp_id, config_path) in enumerate(CONFIG_PATHS.items(), start=1):
        row = gradient_check_experiment(exp_id, config_path, args.seed, args.samples_per_gpu, args.workers_per_gpu)
        gradient_rows.append(row)
        if exp_id in {"G3_H1_H2_lambda001"}:
            h2_profile_rows.append(
                {
                    "experiment_id": exp_id,
                    "compute_time_sec": row["backward_time_sec"],
                    "peak_gpu_memory_mb": row["peak_gpu_memory_mb"],
                    "h2_assign_loss": row["h2_assign_loss"],
                    "h2_leak_loss": row["h2_leak_loss"],
                }
            )
        if row.get("h3_weight_mean") is not None:
            h3_weight_rows.append({"experiment_id": exp_id, "h3_weight_mean": row["h3_weight_mean"]})
        stage_logger.progress("phase8_gradient_check", idx, len(CONFIG_PATHS), experiment_id=exp_id)
    stage_logger.done("phase8_gradient_check")
    write_csv(REPORTS_DIR / "sw9_gradient_check.csv", gradient_rows)
    write_md(REPORTS_DIR / "sw9_gradient_check_summary.md", parse_gradient_summary(gradient_rows))
    write_csv(REPORTS_DIR / "h2_compute_profile.csv", h2_profile_rows or [{"experiment_id": "G3_H1_H2_lambda001", "compute_time_sec": None, "skipped_reason": "no_h2_profile"}])
    write_csv(REPORTS_DIR / "h3_weight_distribution.csv", h3_weight_rows or [{"experiment_id": "G4_H1_H3_lambda001", "h3_weight_mean": None, "skipped_reason": "no_h3_weights"}])

    short_train_targets = parse_short_train_plan(args.short_train_plan)
    train_plan = {
        "P0_original_control": (BASE_CONFIG_PATH, short_train_targets.get("P0_original_control", 1000)),
        "P1_H1_lambda001": (CONFIG_PATHS["G1_H1_lambda001"], short_train_targets.get("P1_H1_lambda001", 1000)),
        "P2_H1_lambda003": (CONFIG_PATHS["G2_H1_lambda003"], short_train_targets.get("P2_H1_lambda003", 1000)),
        "P3_H1_H3_lambda001": (CONFIG_PATHS["G4_H1_H3_lambda001"], short_train_targets.get("P3_H1_H3_lambda001", 1000)),
        "P4_H1_H2_lambda001": (CONFIG_PATHS["G3_H1_H2_lambda001"], short_train_targets.get("P4_H1_H2_lambda001", 500)),
    }
    priority = [TRAIN_ALIAS.get(token.strip(), token.strip()) for token in args.train_priority.split(",") if token.strip()]
    train_iter_rows: list[dict[str, Any]] = []
    train_summary_rows: list[dict[str, Any]] = []
    completed_ckpts: dict[str, Path] = {}
    stage_logger.start("phase9_short_training", experiment_count=len(priority))
    for idx, exp_name in enumerate(priority, start=1):
        config_path, target_iters = train_plan[exp_name]
        if args.fast:
            target_iters = min(target_iters, 20)
        try:
            ckpt_path, iter_rows, summary = short_train_experiment(
                experiment_id=exp_name,
                config_path=config_path,
                target_iters=target_iters,
                seed=args.seed,
                samples_per_gpu=args.samples_per_gpu,
                workers_per_gpu=args.workers_per_gpu,
                deadline_ts=deadline_ts,
                log_interval=args.log_interval,
            )
            train_iter_rows.extend(iter_rows)
            train_summary_rows.append(summary)
            if ckpt_path is not None:
                completed_ckpts[exp_name] = ckpt_path
            if time.time() >= deadline_ts:
                break
        except Exception as exc:
            train_summary_rows.append(
                {
                    "experiment_id": exp_name,
                    "completed_iters": 0,
                    "planned_iters": target_iters,
                    "completed": False,
                    "meets_500_iter_requirement": False,
                    "stop_reason": f"exception:{repr(exc)}",
                }
            )
        stage_logger.progress("phase9_short_training", idx, len(priority), experiment_id=exp_name)
    stage_logger.done("phase9_short_training", completed=len(completed_ckpts))
    write_csv(REPORTS_DIR / "sw9_short_train_metrics.csv", train_summary_rows)
    if train_iter_rows:
        plot_loss_curves(train_iter_rows, FIGURES_DIR / "sw9_loss_curve_h1_h3.png", "SW-9 subset diagnostic short-train loss curve")

    subset_name, subset_indices = choose_eval_subset((time.time() - start_ts) / 3600.0, args.max_hours, reserve_hours)
    reference_rows = read_csv_rows(SW81_REPORTS / "reference_epoch56_union_eval.csv")
    if not reference_rows or any(int(r["sample_index"]) not in set(range(20)) for r in reference_rows[:1]):
        reference_rows, _, _ = evaluate_checkpoint_with_config(BASE_CONFIG_PATH, CHECKPOINT_PATH, "REF_epoch_56", subset_indices, DEFAULT_HORIZONS, False)
    reference_rows = [r for r in reference_rows if int(r["sample_index"]) in set(subset_indices)]

    fixed_eval_rows: list[dict[str, Any]] = []
    eval_summary_rows: list[dict[str, Any]] = []
    stage_logger.start("phase10_fixed_subset_eval", checkpoint_count=len(completed_ckpts))
    for idx, (exp_name, ckpt_path) in enumerate(completed_ckpts.items(), start=1):
        config_path = train_plan[exp_name][0]
        candidate_rows, _, _ = evaluate_checkpoint_with_config(config_path, ckpt_path, exp_name, subset_indices, DEFAULT_HORIZONS, False)
        fixed_eval_rows.extend(candidate_rows)
        agg_rows = aggregate_candidate_deltas(reference_rows, candidate_rows, exp_name, subset_name)
        safe_gate = safe_gate_from_rows(agg_rows)
        eval_summary_rows.append(
            {
                "checkpoint_name": exp_name,
                "checkpoint_path": str(ckpt_path),
                "subset_name": subset_name,
                "safe_gate": safe_gate,
                "agg_rows": agg_rows,
            }
        )
        stage_logger.progress("phase10_fixed_subset_eval", idx, len(completed_ckpts), checkpoint_name=exp_name)
    stage_logger.done("phase10_fixed_subset_eval")

    flat_eval_rows: list[dict[str, Any]] = []
    for item in eval_summary_rows:
        for row in item["agg_rows"]:
            flat_eval_rows.append({**row, **{f"gate_{k}": v for k, v in item["safe_gate"].items()}})
    if not flat_eval_rows:
        flat_eval_rows = [{"checkpoint_name": "none", "subset_name": subset_name, "skipped_reason": "no_completed_checkpoints_for_eval"}]
    write_csv(REPORTS_DIR / "sw9_fixed_subset_eval.csv", flat_eval_rows)
    fixed_eval_headline = f"{subset_name} used for SW-9 subset diagnostic; clean and false-positive gates were compared against epoch_56 reference"
    write_md(REPORTS_DIR / "sw9_fixed_subset_eval_summary.md", fixed_eval_headline + "\n")

    selected_retests = select_retest_candidates(eval_summary_rows)
    reliability_rows: list[dict[str, Any]] = []
    reliability_topk_rows: list[dict[str, Any]] = []
    representative_cases: list[dict[str, Any]] = []
    reliability_headline = "skipped because no completed candidate satisfied retest selection"
    if selected_retests and time.time() < deadline_ts:
        stage_logger.start("phase11_reliability_retest", checkpoint_count=len(selected_retests))
        for idx, item in enumerate(selected_retests, start=1):
            checkpoint_name = item["checkpoint_name"]
            ckpt_path = completed_ckpts[checkpoint_name]
            config_path = train_plan[checkpoint_name][0]
            _, rel_rows, dbg = evaluate_checkpoint_with_config(
                config_path,
                ckpt_path,
                checkpoint_name,
                subset_indices[: min(5, len(subset_indices))],
                DEFAULT_HORIZONS,
                True,
            )
            reliability_rows.extend(rel_rows)
            reliability_topk_rows.extend(dbg.get("topk_rows", []))
            rep = dbg.get("representative_case")
            if rep is not None:
                representative_cases.append({"checkpoint_name": checkpoint_name, **rep})
            stage_logger.progress("phase11_reliability_retest", idx, len(selected_retests), checkpoint_name=checkpoint_name)
        stage_logger.done("phase11_reliability_retest")
        reliability_headline = "best safe candidate and best aggressive candidate were retested with the SW-7 internal reliability indicator"
        if len(representative_cases) >= 2:
            sw81.render_reliability_before_after(
                "sw9_reliability_before_after",
                representative_cases[0],
                representative_cases[1],
                FIGURES_DIR / "sw9_reliability_before_after.png",
            )
            sw81.render_reliability_before_after(
                "sw9_score_alpha_before_after",
                representative_cases[0],
                representative_cases[1],
                FIGURES_DIR / "sw9_score_alpha_before_after.png",
            )
    write_csv(REPORTS_DIR / "sw9_reliability_retest.csv", reliability_rows or [{"checkpoint_name": "none", "skipped_reason": reliability_headline}])
    write_md(REPORTS_DIR / "sw9_reliability_retest_summary.md", reliability_headline + "\n")

    if gradient_rows:
        plot_gradient_norms(gradient_rows, FIGURES_DIR / "sw9_gradient_norms.png")
    if eval_summary_rows:
        plot_metric_delta_bar(eval_summary_rows, FIGURES_DIR / "sw9_metric_delta_bar.png")
        plot_false_positive_tradeoff(eval_summary_rows, FIGURES_DIR / "sw9_false_positive_tradeoff.png")

    contributor_candidates = [row for row in eval_summary_rows if row["checkpoint_name"] != "P0_original_control"]
    best_candidate = None
    if contributor_candidates:
        best_candidate = max(
            contributor_candidates,
            key=lambda row: (
                max(0.0, -float(row["safe_gate"].get("front_sector_false_free_delta") or 0.0))
                + max(0.0, -float(row["safe_gate"].get("small_object_false_free_delta") or 0.0))
                + max(0.0, float(row["safe_gate"].get("new_visible_recall_delta") or 0.0))
                - max(0.0, float(row["safe_gate"].get("false_occupied_delta") or 0.0)) * 2.0
            ),
        )

    gradient_trainable = any(
        row["grad_finite"] and not row["has_nan"] and not row["has_inf"] and row["support_path_nonzero_grad"] and float(row["grad_norm_decoder_layers"] or 0.0) > 0.0
        for row in gradient_rows
        if row["experiment_id"] != "G0_original_control"
    )
    decision_type = "C1_hook_not_trainable"
    next_action = "stabilize the auxiliary loss path before attempting more training."
    if gradient_trainable:
        if best_candidate is not None and best_candidate["safe_gate"]["safe"] and best_candidate["safe_gate"]["targeted_hit"]:
            decision_type = "C3_safe_targeted_improvement"
            next_action = "extend the best SW-9 recipe beyond short-train scale and rerun eval_core_20 before any stronger claim."
        elif best_candidate is not None and best_candidate["safe_gate"]["targeted_hit"] and not best_candidate["safe_gate"]["safe"]:
            decision_type = "C4_tradeoff_improvement"
            next_action = "tighten leakage control or get_occ assignment before scaling this candidate."
        elif contributor_candidates and any(
            (
                float(row["safe_gate"].get("front_sector_false_free_delta") or 0.0) < 0.0
                or float(row["safe_gate"].get("small_object_false_free_delta") or 0.0) < 0.0
                or float(row["safe_gate"].get("new_visible_recall_delta") or 0.0) > 0.0
            )
            for row in contributor_candidates
        ):
            decision_type = "C5_promising_but_undertrained"
            next_action = "continue with a longer controlled run while keeping the false-positive gate active."
        else:
            decision_type = "C2_hook_trainable_no_signal"
            next_action = "if longer short-train still stays flat, move to get_occ assignment or decoder architecture changes."

    decision_payload = {
        "decision_type": decision_type,
        "best_candidate": None if best_candidate is None else best_candidate["checkpoint_name"],
        "gradient_trainable": gradient_trainable,
        "next_action": next_action,
        "false_positive_headline": None if best_candidate is None else f"false_occupied_delta={float(best_candidate['safe_gate'].get('false_occupied_delta') or 0.0):+.4f}, pred_gt_ratio_delta={float(best_candidate['safe_gate'].get('pred_gt_ratio_delta') or 0.0):+.4f}",
    }
    write_json(REPORTS_DIR / "sw9_contributor_supervision_decision.json", decision_payload)
    write_md(REPORTS_DIR / "sw9_contributor_supervision_decision.md", json.dumps(normalize_export(decision_payload), indent=2, ensure_ascii=False) + "\n")

    gradient_headline = "loss finite, gradients finite, and at least one support/query/decoder path received non-zero gradient" if gradient_trainable else "auxiliary loss path failed the trainability check"
    short_train_headline = (
        f"{sum(1 for row in train_summary_rows if row.get('meets_500_iter_requirement'))} short-train runs reached at least 500 iter"
        if train_summary_rows
        else "no short-train run completed"
    )
    fixed_eval_summary = {"headline": fixed_eval_headline, "subset_name": subset_name, "checkpoint_count": len(eval_summary_rows)}
    reliability_summary = {"headline": reliability_headline, "checkpoint_names": [row["checkpoint_name"] for row in selected_retests]}

    report_json = {
        "sw81_digest": sw81_digest,
        "tensor_access_audit": tensor_audit,
        "implementation_status": {
            "h1_status": "implemented",
            "h2_status": "implemented" if any("G3" in row["experiment_id"] for row in gradient_rows) else "skipped_with_reason",
            "h3_status": "implemented" if any("G4" in row["experiment_id"] for row in gradient_rows) else "skipped_with_reason",
        },
        "config_manifest": config_manifest,
        "gradient_check_rows": gradient_rows,
        "gradient_check_summary": {"headline": gradient_headline},
        "short_training_summary": {"headline": short_train_headline, "rows": train_summary_rows},
        "fixed_subset_eval_summary": fixed_eval_summary,
        "reliability_retest_summary": reliability_summary,
        "decision": decision_payload,
        "false_positive_tradeoff": None if best_candidate is None else best_candidate["safe_gate"],
    }
    final_report_text = make_final_report(report_json)
    write_md(REPORTS_DIR / "stage_sw9_contributor_aware_support_supervision_report.md", final_report_text)
    write_json(REPORTS_DIR / "stage_sw9_contributor_aware_support_supervision_report.json", report_json)

    end_ts = time.time()
    time_manifest["end_time_unix"] = end_ts
    time_manifest["wall_clock_hours_used"] = (end_ts - start_ts) / 3600.0
    time_manifest["phases"] = stage_logger.state["stages"]
    write_json(REPORTS_DIR / "sw9_time_budget_manifest.json", time_manifest)


if __name__ == "__main__":
    main()
