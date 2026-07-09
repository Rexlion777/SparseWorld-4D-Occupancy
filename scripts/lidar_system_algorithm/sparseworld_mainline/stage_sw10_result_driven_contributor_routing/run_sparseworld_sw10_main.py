from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
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

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"

SW91_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW91_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"

DEFAULT_LR = 2e-5
ROUTE_A_EXPERIMENT_ID = "routeA_best_h2_scaled_iter3000"
CORE_PERTURBATIONS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
RETEST_PERTURBATIONS = ["A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"]
DEFAULT_HORIZONS = list(range(7))
RETEST_HORIZONS = [2, 4, 6]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    parent = str(path.parent)
    inserted = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        spec.loader.exec_module(mod)
    finally:
        if inserted and sys.path and sys.path[0] == parent:
            sys.path.pop(0)
    return mod


sw91 = load_module(
    "sw10_sw91",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/run_sparseworld_sw91_main.py",
)
sw9 = sw91.sw9
sw81 = sw91.sw81


ROUTE_CONFIGS = {
    "P1_H2_only_low_1000iter": sw91.CONFIG_PATHS["H2_ONLY_L001"],
    "P2_H2_warmup_1000iter": sw91.CONFIG_PATHS["H2_WARMUP_L001"],
    "P3_H2_H3_1000iter": sw91.CONFIG_PATHS["H2_H3_L001_B03"],
    "P4_H2_tinyH1_500iter": sw91.CONFIG_PATHS["H2_TINY_H1_L001"],
    "P5_H2_stronger_500iter": sw91.CONFIG_PATHS["H2_ONLY_L001"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-gpu", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=6)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--reserve-report-minutes", type=float, default=30.0)
    parser.add_argument("--routeA-total-iters", type=int, default=3000)
    parser.add_argument("--routeA-optional-iters", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--quick-eval-count", type=int, default=10)
    parser.add_argument("--core-eval-count", type=int, default=20)
    parser.add_argument("--fast", action="store_true", default=False)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "checkpoints",
        ARTIFACTS_DIR / "routeA_train_logs",
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
                seen.add(key)
                fieldnames.append(key)
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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class StageLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "sw10_status.json"
        self.progress_path = LOGS_DIR / "sw10_progress.jsonl"
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
        now_ts = time.time()
        item = self.state["stages"].setdefault(stage, {})
        start_ts = float(item.get("start_ts", now_ts))
        item.update({"status": "done", "end_ts": now_ts, "duration_sec": now_ts - start_ts, **payload})
        if self.state.get("current_stage") == stage:
            self.state["current_stage"] = None
        self.event(stage, "done", payload)

    def fail(self, stage: str, error: str) -> None:
        now_ts = time.time()
        item = self.state["stages"].setdefault(stage, {})
        start_ts = float(item.get("start_ts", now_ts))
        item.update({"status": "failed", "end_ts": now_ts, "duration_sec": now_ts - start_ts, "error": error})
        if self.state.get("current_stage") == stage:
            self.state["current_stage"] = None
        self.event(stage, "failed", {"error": error})


def phase_block(time_manifest: dict[str, Any], time_manifest_path: Path, name: str) -> dict[str, Any]:
    phase_meta = {"phase_name": name, "start_time": now_iso(), "start_ts": time.time()}
    time_manifest["phases"].append(phase_meta)
    write_json(time_manifest_path, time_manifest)
    return phase_meta


def end_phase(time_manifest: dict[str, Any], time_manifest_path: Path, phase_meta: dict[str, Any], status: str, **extra: Any) -> None:
    end_ts = time.time()
    phase_meta.update(
        {
            "end_time": now_iso(),
            "end_ts": end_ts,
            "duration_sec": end_ts - float(phase_meta.get("start_ts", end_ts)),
            "status": status,
            **extra,
        }
    )
    write_json(time_manifest_path, time_manifest)


def find_first(patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(SW91_REPORTS_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"missing SW-9.1 artifact for patterns: {patterns}")


def read_sw91_inputs() -> dict[str, Any]:
    decision_path = find_first(["*decision*.json"])
    fixed_eval_path = find_first(["*fixed_subset_eval*.csv"])
    contributor_path = find_first(["*contributor*retest*.csv"])
    reliability_path = find_first(["*reliability*retest*.csv"])
    assignment_path = find_first(["*assignment*diagnostics*.csv"])
    payload = {
        "decision_path": str(decision_path),
        "fixed_subset_eval_path": str(fixed_eval_path),
        "contributor_retest_path": str(contributor_path),
        "reliability_retest_path": str(reliability_path),
        "assignment_diagnostics_path": str(assignment_path),
        "decision": json.loads(decision_path.read_text(encoding="utf-8")),
        "fixed_subset_eval_rows": read_csv_rows(fixed_eval_path),
        "contributor_retest_rows": read_csv_rows(contributor_path),
        "reliability_retest_rows": read_csv_rows(reliability_path),
        "assignment_diagnostics_rows": read_csv_rows(assignment_path),
    }
    return payload


def summarize_sw91_digest(sw91_inputs: dict[str, Any]) -> dict[str, Any]:
    decision = sw91_inputs["decision"]
    fixed_rows = sw91_inputs["fixed_subset_eval_rows"]
    best_candidate = decision.get("best_candidate")
    candidate_rows = [row for row in fixed_rows if row.get("checkpoint_name") == best_candidate]
    a0 = next((row for row in candidate_rows if row.get("perturbation_id") == "A0_clean"), {})
    a10 = next((row for row in candidate_rows if row.get("perturbation_id") == "A10_drop_front_triplet"), {})
    c4 = next((row for row in candidate_rows if row.get("perturbation_id") == "C4_motion_blur_9"), {})
    digest = {
        "sw91_decision_type": decision.get("decision_type"),
        "sw91_summary": decision.get("summary"),
        "gradient_trainable": bool(decision.get("gradient_trainable")),
        "proxy_improved": bool(decision.get("proxy_improved")),
        "best_candidate": best_candidate,
        "false_positive_pred_gt_tradeoff": decision.get("false_positive_pred_gt_tradeoff"),
        "next_unique_action": decision.get("next_unique_action"),
        "best_candidate_subset_metrics": {
            "clean_occupied_iou_delta": float(a0.get("mean_occupied_iou_delta") or 0.0),
            "clean_semantic_miou_delta": float(a0.get("mean_semantic_miou_delta") or 0.0),
            "a10_small_object_false_free_delta": float(a10.get("mean_small_object_false_free_delta") or 0.0),
            "a10_front_sector_false_free_delta": float(a10.get("mean_front_sector_false_free_delta") or 0.0),
            "a10_new_visible_recall_delta": float(a10.get("mean_new_visible_recall_delta") or 0.0),
            "c4_false_occupied_delta": float(c4.get("mean_false_occupied_rate_delta") or 0.0),
        },
    }
    return digest


def select_route(sw91_digest: dict[str, Any], sw91_inputs: dict[str, Any]) -> dict[str, Any]:
    decision_type = sw91_digest["sw91_decision_type"]
    route = "A"
    route_reason = (
        f"SW-9.1 decision={decision_type}; at least one H2-centered candidate was safe and hit a targeted subset gate, "
        "so SW-10 must first test whether that signal survives longer training."
    )
    primary_candidate = sw91_digest["best_candidate"] or "P2_H2_warmup_1000iter"
    selected_config = ROUTE_CONFIGS.get(primary_candidate, ROUTE_CONFIGS["P2_H2_warmup_1000iter"])
    selected_checkpoint = SW91_ARTIFACTS_DIR / "checkpoints" / f"{primary_candidate}.pth"
    if not selected_checkpoint.exists():
        fallback = "P2_H2_warmup_1000iter" if (SW91_ARTIFACTS_DIR / "checkpoints/P2_H2_warmup_1000iter.pth").exists() else "P1_H2_only_low_1000iter"
        primary_candidate = fallback
        selected_config = ROUTE_CONFIGS[primary_candidate]
        selected_checkpoint = SW91_ARTIFACTS_DIR / "checkpoints" / f"{primary_candidate}.pth"
    evidence = {
        "decision_type": decision_type,
        "best_candidate": primary_candidate,
        "best_candidate_checkpoint_exists": selected_checkpoint.exists(),
        "best_candidate_config": str(selected_config),
        "clean_tradeoff": sw91_digest["best_candidate_subset_metrics"],
        "sw91_next_action": sw91_digest["next_unique_action"],
    }
    return {
        "selected_route": route,
        "summary": "Route A selected: scale the best safe SW-9.1 H2-centered candidate before any get_occ routing change.",
        "evidence": evidence,
        "selected_candidate": primary_candidate,
        "selected_config_path": str(selected_config),
        "selected_checkpoint_path": str(selected_checkpoint),
        "helper_check": "re-evaluate the starting SW-9.1 checkpoint alongside the scaled checkpoint under the same eval_core_20 protocol",
        "route_reason": route_reason,
    }


def build_eval_subsets(args: argparse.Namespace) -> tuple[dict[str, list[int]], list[int]]:
    quick = list(range(args.quick_eval_count))
    core = list(range(args.core_eval_count))
    union = sorted(set(quick + core))
    return {"quick_eval_10": quick, "eval_core_20": core}, union


def checkpoint_iter(path: Path) -> int:
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    return int(meta.get("iter", 0))


def plot_route_selection(route_payload: dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axis("off")
    boxes = [
        (0.05, 0.35, 0.22, 0.3, "SW-9.1\ndecision"),
        (0.39, 0.35, 0.22, 0.3, "Route A\nselected"),
        (0.73, 0.35, 0.22, 0.3, "Scale safe\nH2 candidate"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, fc="#E8F1F2", ec="#1F3A5F", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(0.39, 0.5), xytext=(0.27, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#1F3A5F"})
    ax.annotate("", xy=(0.73, 0.5), xytext=(0.61, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#1F3A5F"})
    ax.set_title("SW-10 subset diagnostic route selection flow")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def routeA_train(
    experiment_id: str,
    config_path: Path,
    resume_checkpoint_path: Path,
    target_total_iters: int,
    seed: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    deadline_ts: float,
    log_interval: int,
) -> tuple[Path | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg = dataset = model = dataloader = optimizer = checkpoint = None
    iter_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    dataloader_workers = workers_per_gpu
    stop_reason = "completed"
    failure_trace = None
    start_total_iter = 0
    completed_total_iter = 0
    ckpt_iter3000_path: Path | None = None
    try:
        cfg, dataset, model, checkpoint = sw9.build_runtime(
            config_path,
            train=True,
            cfg_overrides={
                "data.samples_per_gpu": samples_per_gpu,
                "data.workers_per_gpu": dataloader_workers,
                "optimizer.lr": DEFAULT_LR,
            },
        )
        if resume_checkpoint_path.exists():
            resume_state = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(resume_state["state_dict"], strict=False)
            meta = resume_state.get("meta", {})
            start_total_iter = int(meta.get("iter", 0))
        from mmcv.runner import build_optimizer

        dataloader = sw81.build_dataloader_for_cfg(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=dataloader_workers,
            shuffle=True,
            seed=seed,
        )
        try:
            first_batch = next(iter(dataloader))
            del first_batch
        except (OSError, RuntimeError) as exc:
            memory_fetch_error = "cannot allocate memory" in str(exc).lower() or "dataloader worker process" in str(exc).lower()
            if dataloader_workers > 0 and memory_fetch_error:
                dataloader_workers = 0
                dataloader = sw81.build_dataloader_for_cfg(
                    dataset,
                    samples_per_gpu=samples_per_gpu,
                    workers_per_gpu=0,
                    shuffle=True,
                    seed=seed,
                )
            else:
                raise
        optimizer = build_optimizer(model, cfg.optimizer)
        data_iter = iter(dataloader)
        additional_iters = max(0, target_total_iters - start_total_iter)
        if additional_iters == 0:
            stop_reason = "already_at_target"
        for local_iter in range(1, additional_iters + 1):
            current_total_iter = start_total_iter + local_iter
            if time.time() >= deadline_ts:
                stop_reason = "budget_exhausted"
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            try:
                optimizer.zero_grad(set_to_none=True)
                sw91.set_h2_progress(model, current_total_iter)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                model_inputs = sw81.move_train_batch_to_cuda(batch)
                losses = model(return_loss=True, **model_inputs)
                total_loss, log_vars = sw81.parse_losses(losses)
                if not bool(torch.isfinite(total_loss).item()):
                    stop_reason = "non_finite_loss"
                    break
                total_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).detach().cpu().item())
                optimizer.step()
                torch.cuda.synchronize()
                completed_total_iter = current_total_iter
                diag = sw91.aggregate_sw9_debug(getattr(model, "latest_sw9_debug", {"records": []}))
                row = {
                    "experiment_id": experiment_id,
                    "iter": current_total_iter,
                    "local_iter": local_iter,
                    "iter_time_sec": time.perf_counter() - started,
                    "peak_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                    "grad_norm": grad_norm,
                    **log_vars,
                    **diag,
                }
                if current_total_iter % log_interval == 0 or current_total_iter == target_total_iters:
                    iter_rows.append(row)
                    with (ARTIFACTS_DIR / "routeA_train_logs" / f"{experiment_id}.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(normalize_export(row), ensure_ascii=False) + "\n")
                if current_total_iter % 100 == 0 or current_total_iter == target_total_iters:
                    diag_rows.append({"experiment_id": experiment_id, **diag, "iter": current_total_iter})
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    stop_reason = "oom"
                    safe_cuda_cleanup()
                    break
                stop_reason = "runtime_error"
                failure_trace = traceback.format_exc()
                break
        if completed_total_iter == 0 and start_total_iter > 0:
            completed_total_iter = start_total_iter
        if completed_total_iter >= target_total_iters:
            ckpt_iter3000_path = ARTIFACTS_DIR / "checkpoints" / f"{experiment_id}.pth"
            sw81.save_checkpoint(
                ckpt_iter3000_path,
                model,
                {
                    "experiment_id": experiment_id,
                    "iter": completed_total_iter,
                    "planned_total_iters": target_total_iters,
                    "resume_from": str(resume_checkpoint_path),
                    "config_path": str(config_path),
                    "optimizer_lr": DEFAULT_LR,
                },
            )
        summary = {
            "experiment_id": experiment_id,
            "config_path": str(config_path),
            "resume_checkpoint_path": str(resume_checkpoint_path),
            "start_total_iter": start_total_iter,
            "target_total_iter": target_total_iters,
            "completed_total_iter": completed_total_iter,
            "completed_additional_iters": max(0, completed_total_iter - start_total_iter),
            "effective_workers_per_gpu": dataloader_workers,
            "stop_reason": stop_reason,
            "completed": completed_total_iter >= target_total_iters,
            "checkpoint_path": None if ckpt_iter3000_path is None else str(ckpt_iter3000_path),
            "peak_gpu_memory_mb": None if not iter_rows else float(max(row["peak_memory_mb"] for row in iter_rows)),
            "mean_iter_time_sec": None if not iter_rows else float(np.mean([row["iter_time_sec"] for row in iter_rows])),
            "loss_head_mean": None if not iter_rows else float(np.mean([row["loss_total"] for row in iter_rows[: min(10, len(iter_rows))]])),
            "loss_tail_mean": None if not iter_rows else float(np.mean([row["loss_total"] for row in iter_rows[-min(10, len(iter_rows)) :]])),
            "loss_trend_delta": None
            if not iter_rows
            else float(np.mean([row["loss_total"] for row in iter_rows[-min(10, len(iter_rows)) :]]) - np.mean([row["loss_total"] for row in iter_rows[: min(10, len(iter_rows))]])),
            "failure_trace": failure_trace,
            **(diag_rows[-1] if diag_rows else {}),
        }
        return ckpt_iter3000_path, iter_rows, diag_rows, summary
    finally:
        del optimizer, dataloader, checkpoint, model, dataset, cfg
        safe_cuda_cleanup()


def evaluate_routeA_candidates(
    candidate_items: list[tuple[str, Path, Path]],
    sample_indices: list[int],
    subset_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reference_rows, _, _ = sw91.evaluate_checkpoint_detailed(
        BASE_CONFIG_PATH,
        CHECKPOINT_PATH,
        "REF_epoch_56",
        sample_indices,
        DEFAULT_HORIZONS,
        capture_reliability=False,
    )
    eval_flat_rows: list[dict[str, Any]] = []
    eval_summary_rows: list[dict[str, Any]] = []
    gate_lookup: dict[str, Any] = {}
    for checkpoint_name, checkpoint_path, config_path in candidate_items:
        rows, _, _ = sw91.evaluate_checkpoint_detailed(
            config_path,
            checkpoint_path,
            checkpoint_name,
            sample_indices,
            DEFAULT_HORIZONS,
            capture_reliability=False,
        )
        cmp_payload = sw91.compare_against_reference(rows, reference_rows, checkpoint_name)
        agg_rows = sw91.aggregate_candidate_deltas(cmp_payload, subset_name)
        gate = sw91.safe_gate_from_rows(agg_rows)
        for row in agg_rows:
            row["gate_safe"] = gate["safe"]
            row["gate_targeted_hit"] = gate["targeted_hit"]
            row["safe_gate_json"] = json.dumps(gate, ensure_ascii=False)
        eval_flat_rows.extend(agg_rows)
        eval_summary_rows.append({"checkpoint_name": checkpoint_name, "safe_gate": gate, "aggregate_rows": agg_rows})
        gate_lookup[checkpoint_name] = gate
    return eval_flat_rows, eval_summary_rows, gate_lookup


def plot_loss_curve(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([int(r["iter"]) for r in rows], [float(r["loss_total"]) for r in rows], color="#C0392B", linewidth=1.8)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss_total")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_routeA_metric_bar(eval_rows: list[dict[str, Any]], out_path: Path) -> None:
    focus_rows = [row for row in eval_rows if row["perturbation_id"] == "A10_drop_front_triplet"]
    labels = [row["checkpoint_name"] for row in focus_rows]
    vals = [float(row.get("mean_front_sector_false_free_delta") or 0.0) for row in focus_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, vals, color="#1F618D")
    ax.set_ylabel("delta")
    ax.set_title("SW-10 subset diagnostic A10 front-sector false-free delta")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_false_positive_tradeoff(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for row in summary_rows:
        gate = row["safe_gate"]
        ax.scatter(float(gate["pred_gt_ratio_delta"]), float(gate["false_occupied_delta"]), label=row["checkpoint_name"], s=80)
    ax.axvline(0.10, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.008, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("pred_gt_ratio_delta")
    ax.set_ylabel("false_occupied_delta")
    ax.set_title("SW-10 subset diagnostic false-positive / density tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def aggregate_reliability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    value_keys = [
        "front_sector_reliability",
        "small_object_reliability",
        "new_visible_reliability",
        "high_risk_voxel_ratio",
        "risk_error_correlation",
        "pred_gt_occupied_ratio",
        "false_occupied_rate",
        "false_free_rate",
    ]
    return sw91.aggregate_eval(rows, ["checkpoint_name"], value_keys)


def make_final_report(report_json: dict[str, Any]) -> str:
    lines = [
        "# Stage SW-10 Result-driven Contributor Routing and Native get_occ Alignment",
        "",
        "1. Executive summary",
        f"- {report_json['executive_summary']}",
        "",
        "2. Why SW-10 follows SW-9.1",
        "- result-driven execution based on SW-9.1",
        "- subset diagnostic, not official benchmark",
        "- no claim of surpassing prior paper results",
        "- no production claim and no calibrated uncertainty claim",
        "",
        "3. SW-9.1 result digest",
        f"- SW-9.1 decision: {report_json['sw91_digest']['sw91_decision_type']}",
        "",
        "4. Route selection",
        f"- {report_json['route_selection']['summary']}",
        "",
        "5. Executed route details",
        f"- {report_json['route_execution']['headline']}",
        "",
        "6. Training / diagnostic / variant results",
        f"- {report_json['route_execution']['result_headline']}",
        "",
        "7. Contributor diagnostic retest",
        f"- {report_json['contributor_retest']['headline']}",
        "",
        "8. Reliability retest",
        f"- {report_json['reliability_retest']['headline']}",
        "",
        "9. False-positive / density tradeoff",
        f"- false_occupied_delta={report_json['decision']['best_tradeoff']['false_occupied_delta']}",
        f"- pred_gt_ratio_delta={report_json['decision']['best_tradeoff']['pred_gt_ratio_delta']}",
        "",
        "10. Final decision R1-R9",
        f"- {report_json['decision']['decision_type']}: {report_json['decision']['summary']}",
        "",
        "11. Safe claims",
        "- subset diagnostic only",
        "- not official benchmark",
        "- no claim of surpassing prior paper results",
        "- no full validation claim",
        "- no model improvement claim unless subset safety gates support it",
        "",
        "12. Limitations",
        "- Route A scaled only one main candidate within budget",
        "- eval_stress_20 is optional and may be skipped when report/test reserve must be protected",
        "",
        "13. Next unique action",
        f"- {report_json['decision']['next_unique_action']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    stage_logger = StageLogger()
    started_at = time.time()
    deadline_ts = started_at + args.max_hours * 3600.0
    time_manifest = {
        "stage": "SW-10",
        "start_time": now_iso(),
        "max_hours": args.max_hours,
        "reserve_report_minutes": args.reserve_report_minutes,
        "phases": [],
        "route_execution": [],
    }
    time_manifest_path = REPORTS_DIR / "sw10_time_budget_manifest.json"
    write_json(time_manifest_path, time_manifest)

    # Phase 1
    phase = phase_block(time_manifest, time_manifest_path, "phase1_sw91_read_and_route_selection")
    stage_logger.start("phase1_sw91_read_and_route_selection")
    try:
        sw91_inputs = read_sw91_inputs()
    except FileNotFoundError as exc:
        blocked = {
            "status": "blocked",
            "reason": str(exc),
            "wait_window_minutes": 10,
        }
        write_json(REPORTS_DIR / "sw91_result_digest_for_sw10.json", blocked)
        write_json(REPORTS_DIR / "sw10_route_selection.json", blocked)
        write_md(REPORTS_DIR / "sw10_route_selection.md", f"Blocked: {exc}\n")
        write_json(REPORTS_DIR / "sw10_result_driven_decision.json", {"decision_type": "R7_trainability_blocked", "summary": "SW-9.1 inputs missing; route execution was blocked."})
        write_md(REPORTS_DIR / "sw10_result_driven_decision.md", "SW-10 blocked because SW-9.1 outputs were missing.\n")
        stage_logger.fail("phase1_sw91_read_and_route_selection", str(exc))
        end_phase(time_manifest, time_manifest_path, phase, "blocked", reason=str(exc))
        time_manifest["end_time"] = now_iso()
        time_manifest["wall_clock_sec"] = time.time() - started_at
        write_json(time_manifest_path, time_manifest)
        return
    sw91_digest = summarize_sw91_digest(sw91_inputs)
    route_selection = select_route(sw91_digest, sw91_inputs)
    write_json(REPORTS_DIR / "sw91_result_digest_for_sw10.json", sw91_digest)
    write_json(REPORTS_DIR / "sw10_route_selection.json", route_selection)
    write_md(REPORTS_DIR / "sw10_route_selection.md", route_selection["route_reason"] + "\n")
    plot_route_selection(route_selection, FIGURES_DIR / "sw10_route_selection_flow.png")
    stage_logger.done("phase1_sw91_read_and_route_selection", selected_route=route_selection["selected_route"])
    end_phase(time_manifest, time_manifest_path, phase, "done", selected_route=route_selection["selected_route"])

    # Route A
    phase = phase_block(time_manifest, time_manifest_path, "routeA_scaled_training_and_eval")
    stage_logger.start("routeA_scaled_training_and_eval")
    selected_config_path = Path(route_selection["selected_config_path"])
    selected_checkpoint_path = Path(route_selection["selected_checkpoint_path"])
    start_iter_hint = checkpoint_iter(selected_checkpoint_path)
    target_total_iters = (start_iter_hint + 40) if args.fast else args.routeA_total_iters
    train_ckpt_path, train_iter_rows, train_diag_rows, train_summary = routeA_train(
        ROUTE_A_EXPERIMENT_ID,
        selected_config_path,
        selected_checkpoint_path,
        target_total_iters,
        args.seed,
        args.samples_per_gpu,
        args.workers_per_gpu,
        deadline_ts - args.reserve_report_minutes * 60.0,
        args.log_interval,
    )
    write_csv(REPORTS_DIR / "routeA_scaled_h2_train_metrics.csv", [train_summary])
    if train_iter_rows:
        plot_loss_curve(train_iter_rows, FIGURES_DIR / "routeA_scaled_h2_loss_curve.png", "SW-10 subset diagnostic Route A scaled H2 loss curve")

    eval_subsets, _ = build_eval_subsets(args)
    subset_name = "quick_eval_10" if args.fast else "eval_core_20"
    sample_indices = eval_subsets[subset_name]
    eval_rows: list[dict[str, Any]] = []
    eval_summary_rows: list[dict[str, Any]] = []
    gate_lookup: dict[str, Any] = {}
    candidate_items: list[tuple[str, Path, Path]] = []
    if selected_checkpoint_path.exists():
        candidate_items.append((f"routeA_start_{route_selection['selected_candidate']}", selected_checkpoint_path, selected_config_path))
    if train_ckpt_path is not None and train_ckpt_path.exists():
        candidate_items.append((ROUTE_A_EXPERIMENT_ID, train_ckpt_path, selected_config_path))
    if candidate_items:
        eval_rows, eval_summary_rows, gate_lookup = evaluate_routeA_candidates(candidate_items, sample_indices, subset_name)
    write_csv(REPORTS_DIR / "routeA_scaled_h2_eval_metrics.csv", eval_rows or [{"checkpoint_name": "none", "skipped_reason": "no routeA checkpoint available"}])
    if eval_rows:
        plot_routeA_metric_bar(eval_rows, FIGURES_DIR / "sw10_metric_delta_bar.png")
        plot_false_positive_tradeoff(eval_summary_rows, FIGURES_DIR / "sw10_false_positive_density_tradeoff.png")

    best_scaled_gate = gate_lookup.get(ROUTE_A_EXPERIMENT_ID)
    start_gate = gate_lookup.get(f"routeA_start_{route_selection['selected_candidate']}")
    route_result_lines = [
        f"Route A resumed {route_selection['selected_candidate']} from SW-9.1 and trained to total_iter={train_summary['completed_total_iter']}.",
        f"subset_name={subset_name}; eval_stress_20 skipped to protect the report/test reserve.",
        f"start_gate={json.dumps(normalize_export(start_gate), ensure_ascii=False) if start_gate else 'missing'}",
        f"scaled_gate={json.dumps(normalize_export(best_scaled_gate), ensure_ascii=False) if best_scaled_gate else 'missing'}",
    ]
    write_md(REPORTS_DIR / "routeA_scaled_h2_summary.md", "\n".join(route_result_lines) + "\n")
    stage_logger.done("routeA_scaled_training_and_eval", completed_total_iter=train_summary["completed_total_iter"])
    end_phase(
        time_manifest,
        time_manifest_path,
        phase,
        "done",
        completed_total_iter=train_summary["completed_total_iter"],
        checkpoint_path=None if train_ckpt_path is None else str(train_ckpt_path),
    )

    # Contributor retest
    phase = phase_block(time_manifest, time_manifest_path, "common_phase_contributor_retest")
    stage_logger.start("common_phase_contributor_retest")
    contributor_rows: list[dict[str, Any]] = []
    contributor_headline = "skipped because Route A did not produce a checkpoint"
    contributor_summary_rows: list[dict[str, Any]] = []
    if train_ckpt_path is not None and train_ckpt_path.exists():
        retest_sample_indices = sample_indices[: min(5, len(sample_indices))]
        contributor_rows.extend(
            sw91.run_contributor_diagnostic(
                BASE_CONFIG_PATH,
                CHECKPOINT_PATH,
                "REF_epoch_56",
                retest_sample_indices,
                RETEST_HORIZONS,
                RETEST_PERTURBATIONS,
            )
        )
        contributor_rows.extend(
            sw91.run_contributor_diagnostic(
                selected_config_path,
                selected_checkpoint_path,
                f"routeA_start_{route_selection['selected_candidate']}",
                retest_sample_indices,
                RETEST_HORIZONS,
                RETEST_PERTURBATIONS,
            )
        )
        contributor_rows.extend(
            sw91.run_contributor_diagnostic(
                selected_config_path,
                train_ckpt_path,
                ROUTE_A_EXPERIMENT_ID,
                retest_sample_indices,
                RETEST_HORIZONS,
                RETEST_PERTURBATIONS,
            )
        )
        contributor_summary_rows = sw91.aggregate_contributor_rows(contributor_rows)
        if contributor_summary_rows:
            contributor_headline = "starting SW-9.1 checkpoint and the scaled SW-10 Route A checkpoint were retested with subset contributor diagnostics"
            sw91.plot_contributor_waterfall(contributor_summary_rows, FIGURES_DIR / "sw10_contributor_waterfall_before_after.png")
    write_csv(REPORTS_DIR / "sw10_contributor_diagnostic_retest.csv", contributor_rows or [{"checkpoint_name": "none", "skipped_reason": contributor_headline}])
    write_md(REPORTS_DIR / "sw10_contributor_diagnostic_summary.md", contributor_headline + "\n")
    stage_logger.done("common_phase_contributor_retest")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    # Reliability retest
    phase = phase_block(time_manifest, time_manifest_path, "common_phase_reliability_retest")
    stage_logger.start("common_phase_reliability_retest")
    reliability_rows: list[dict[str, Any]] = []
    reliability_headline = "skipped because Route A did not produce a checkpoint"
    if train_ckpt_path is not None and train_ckpt_path.exists():
        retest_sample_indices = sample_indices[: min(5, len(sample_indices))]
        _, ref_rel_rows, ref_dbg = sw91.evaluate_checkpoint_detailed(
            BASE_CONFIG_PATH,
            CHECKPOINT_PATH,
            "REF_epoch_56",
            retest_sample_indices,
            RETEST_HORIZONS,
            capture_reliability=True,
        )
        _, start_rel_rows, start_dbg = sw91.evaluate_checkpoint_detailed(
            selected_config_path,
            selected_checkpoint_path,
            f"routeA_start_{route_selection['selected_candidate']}",
            retest_sample_indices,
            RETEST_HORIZONS,
            capture_reliability=True,
        )
        _, scaled_rel_rows, scaled_dbg = sw91.evaluate_checkpoint_detailed(
            selected_config_path,
            train_ckpt_path,
            ROUTE_A_EXPERIMENT_ID,
            retest_sample_indices,
            RETEST_HORIZONS,
            capture_reliability=True,
        )
        reliability_rows.extend(ref_rel_rows)
        reliability_rows.extend(start_rel_rows)
        reliability_rows.extend(scaled_rel_rows)
        if ref_dbg.get("representative_case") and scaled_dbg.get("representative_case"):
            sw81.render_reliability_before_after(
                "sw10_reliability",
                ref_dbg["representative_case"],
                scaled_dbg["representative_case"],
                FIGURES_DIR / "sw10_reliability_before_after.png",
            )
            sw81.render_reliability_before_after(
                "sw10_score_alpha",
                ref_dbg["representative_case"],
                scaled_dbg["representative_case"],
                FIGURES_DIR / "sw10_score_alpha_before_after.png",
            )
        reliability_headline = "subset SW-7 reliability retest completed for the Route A starting checkpoint and the scaled Route A checkpoint"
    write_csv(REPORTS_DIR / "sw10_reliability_retest.csv", reliability_rows or [{"checkpoint_name": "none", "skipped_reason": reliability_headline}])
    write_md(REPORTS_DIR / "sw10_reliability_retest_summary.md", reliability_headline + "\n")
    stage_logger.done("common_phase_reliability_retest")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    # Decision
    phase = phase_block(time_manifest, time_manifest_path, "common_phase_final_decision")
    stage_logger.start("common_phase_final_decision")
    scaled_safe = bool(best_scaled_gate and best_scaled_gate["safe"])
    scaled_targeted = bool(best_scaled_gate and best_scaled_gate["targeted_hit"])
    start_safe = bool(start_gate and start_gate["safe"])
    start_targeted = bool(start_gate and start_gate["targeted_hit"])
    if scaled_safe and scaled_targeted:
        decision_type = "R1_scale_h2_success"
        decision_summary = "Route A longer training preserved subset safety gates and retained a targeted gain signal."
        next_unique_action = "extend the same Route A recipe to 5000 iter and rerun eval_core_20 before considering any architecture change."
    elif start_safe and start_targeted and not scaled_safe:
        decision_type = "R8_architecture_bottleneck_strengthened"
        decision_summary = "The SW-9.1 short safe signal did not remain stable after Route A scaling, which strengthens the evidence for an architecture-side bottleneck."
        next_unique_action = "prepare the minimal native get_occ routing prototype next instead of another blind long run."
    else:
        decision_type = "R8_architecture_bottleneck_strengthened"
        decision_summary = "Route A did not produce a new safe targeted signal in the available budget, so scaling alone is not enough evidence."
        next_unique_action = "move to the smallest native get_occ contributor-routing prototype under a dedicated next stage."
    best_tradeoff = best_scaled_gate or start_gate or {"false_occupied_delta": None, "pred_gt_ratio_delta": None}
    decision_payload = {
        "selected_route": route_selection["selected_route"],
        "decision_type": decision_type,
        "summary": decision_summary,
        "best_candidate": ROUTE_A_EXPERIMENT_ID if best_scaled_gate is not None else route_selection["selected_candidate"],
        "best_tradeoff": {
            "false_occupied_delta": best_tradeoff.get("false_occupied_delta"),
            "pred_gt_ratio_delta": best_tradeoff.get("pred_gt_ratio_delta"),
        },
        "scale_not_stable": bool(start_safe and start_targeted and not scaled_safe),
        "next_unique_action": next_unique_action,
    }
    write_json(REPORTS_DIR / "sw10_result_driven_decision.json", decision_payload)
    write_md(REPORTS_DIR / "sw10_result_driven_decision.md", json.dumps(normalize_export(decision_payload), indent=2, ensure_ascii=False) + "\n")
    stage_logger.done("common_phase_final_decision", decision_type=decision_type)
    end_phase(time_manifest, time_manifest_path, phase, "done", decision_type=decision_type)

    # Final report
    phase = phase_block(time_manifest, time_manifest_path, "final_report")
    stage_logger.start("final_report")
    final_report_json = {
        "executive_summary": route_selection["summary"],
        "sw91_digest": sw91_digest,
        "route_selection": route_selection,
        "route_execution": {
            "headline": f"Route A resumed {route_selection['selected_candidate']} and aimed for total_iter={target_total_iters}.",
            "result_headline": f"Route A completed_total_iter={train_summary['completed_total_iter']} with stop_reason={train_summary['stop_reason']}.",
            "train_summary": train_summary,
        },
        "contributor_retest": {"headline": contributor_headline},
        "reliability_retest": {"headline": reliability_headline},
        "decision": decision_payload,
    }
    final_report_text = make_final_report(final_report_json)
    write_md(REPORTS_DIR / "stage_sw10_result_driven_contributor_routing_report.md", final_report_text)
    write_json(REPORTS_DIR / "stage_sw10_result_driven_contributor_routing_report.json", final_report_json)
    stage_logger.done("final_report")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    time_manifest["end_time"] = now_iso()
    time_manifest["wall_clock_sec"] = time.time() - started_at
    write_json(time_manifest_path, time_manifest)


if __name__ == "__main__":
    main()
