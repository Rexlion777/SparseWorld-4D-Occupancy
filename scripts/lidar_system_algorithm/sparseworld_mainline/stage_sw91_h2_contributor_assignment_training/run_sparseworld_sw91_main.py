from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
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

CONFIG_PATHS = {
    "H2_ONLY_L0005": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_only_lambda0005.py",
    "H2_ONLY_L001": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_only_lambda001.py",
    "H2_WARMUP_L001": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_warmup_lambda001.py",
    "H2_H3_L001_B03": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_h3_lambda001_beta03.py",
    "H2_TINY_H1_L001": REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_tiny_h1_lambda001.py",
}

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"

SW4_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"
SW7_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
SW81_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune"
SW9_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"
SW9_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"
SW81_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw81_scaled_finetune"

CORE_PERTURBATIONS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
RETEST_PERTURBATIONS = ["A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"]
DEFAULT_HORIZONS = list(range(7))
RETEST_HORIZONS = [2, 4, 6]
DEFAULT_LR = 2e-5

TRAIN_ALIASES = {
    "P0": "P0_control_1000iter",
    "P1": "P1_H2_only_low_1000iter",
    "P2": "P2_H2_warmup_1000iter",
    "P3": "P3_H2_H3_1000iter",
    "P4": "P4_H2_tinyH1_500iter",
    "P5": "P5_H2_stronger_500iter",
}


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


sw2 = load_module(
    "sw91_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4 = load_module(
    "sw91_sw4",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/run_sparseworld_sw4_main.py",
)
sw4_inst = load_module(
    "sw91_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw91_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw41 = load_module(
    "sw91_sw41",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation/run_sparseworld_sw41_main.py",
)
sw81 = load_module(
    "sw91_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw9 = load_module(
    "sw91_sw9",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/run_sparseworld_sw9_main.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-gpu", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=6)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--reserve-report-minutes", type=float, default=30.0)
    parser.add_argument("--quick-eval-count", type=int, default=10)
    parser.add_argument("--core-eval-count", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--short-train-plan", default="P0:1000,P1:1000,P2:1000,P3:1000,P4:500,P5:500")
    parser.add_argument("--train-priority", default="P0,P1,P2,P3,P4,P5")
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


def safe_div(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class StageLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "sw91_status.json"
        self.progress_path = LOGS_DIR / "sw91_progress.jsonl"
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
        mapping[TRAIN_ALIASES.get(key.strip(), key.strip())] = int(value.strip())
    return mapping


def set_h2_progress(model: Any, current_iter: int) -> None:
    supervisor = getattr(model, "sw9_support_supervision", None)
    if supervisor is not None and hasattr(supervisor, "set_training_progress"):
        supervisor.set_training_progress(current_iter)


def aggregate_sw9_debug(debug_payload: dict[str, Any]) -> dict[str, float | None]:
    records = [r for r in debug_payload.get("records", []) if not r.get("skipped", False)]
    if not records:
        return {
            "mean_h2_assign_loss": None,
            "mean_h2_leak_loss": None,
            "mean_soft_assign_distance_pos": None,
            "mean_support_to_gt_distance": None,
            "positive_voxel_coverage_proxy": None,
            "negative_leakage_proxy": None,
            "support_confidence_mean": None,
            "support_density_front_sector": None,
            "support_density_small_object_region": None,
            "active_lambda_h2": None,
            "active_lambda_leak": None,
            "h2_warmup_scale": None,
        }
    def avg(key: str) -> float | None:
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "mean_h2_assign_loss": avg("l_assign"),
        "mean_h2_leak_loss": avg("l_support_leak"),
        "mean_soft_assign_distance_pos": avg("assign_distance_mean"),
        "mean_support_to_gt_distance": avg("support_to_gt_distance_mean"),
        "positive_voxel_coverage_proxy": avg("positive_voxel_coverage_proxy"),
        "negative_leakage_proxy": avg("negative_leakage_proxy"),
        "support_confidence_mean": avg("support_conf_mean"),
        "support_density_front_sector": avg("support_density_front_sector"),
        "support_density_small_object_region": avg("support_density_small_object_region"),
        "active_lambda_h2": avg("active_lambda_h2"),
        "active_lambda_leak": avg("active_lambda_leak"),
        "h2_warmup_scale": avg("h2_warmup_scale"),
    }


def compute_train_budget(deadline_ts: float, reserve_minutes: float) -> tuple[float, float]:
    reserve_sec = reserve_minutes * 60.0
    remaining_sec = max(0.0, deadline_ts - time.time())
    return remaining_sec, reserve_sec


def choose_eval_subset(elapsed_hours: float, max_hours: float, reserve_hours: float) -> tuple[str, list[int]]:
    if elapsed_hours + reserve_hours <= max_hours:
        return "eval_core_20", list(range(20))
    return "quick_eval_10", list(range(10))


def run_time_manifest_update(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def compute_eval_row_sw91(
    checkpoint_name: str,
    perturbation_id: str,
    sample_index: int,
    horizon_s: int,
    pred_h: torch.Tensor,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    sector_masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    valid_mask = sw2.valid_mask_from_gt(gt_h)
    base = sw2.compute_base_metrics(pred_h, gt_h, valid_mask)
    region = sw2.region_proxy_metrics(pred_h, gt0, gt_h)
    front_ff = sw81.front_sector_false_free(pred_h, gt_h, valid_mask, sector_masks["front"])
    dyn = sw2.class_group_metrics(pred_h, gt_h, valid_mask, sw2.CLASS_GROUPS["all_dynamic"])
    sta = sw2.class_group_metrics(pred_h, gt_h, valid_mask, sw2.CLASS_GROUPS["all_static"])
    row = {
        "checkpoint_name": checkpoint_name,
        "perturbation_id": perturbation_id,
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        **base,
        **region,
        "small_object_false_free": sw81.small_object_false_free(pred_h, gt_h, valid_mask),
        "new_visible_recall": float(region["new_visible_proxy_recall"]),
        "front_sector_false_free": float(front_ff),
        "dynamic_false_free": float(dyn["group_false_free_rate"]),
        "static_false_free": float(sta["group_false_free_rate"]),
    }
    return row


def evaluate_checkpoint_detailed(
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_name: str,
    sample_indices: list[int],
    horizons: list[int],
    capture_reliability: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg = dataset = model = None
    original_forward = None
    cfg, dataset, model, _ = sw9.build_runtime(
        config_path,
        train=False,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    from mmcv.parallel import collate as collate_fn

    head = sw4_inst.get_pts_bbox_head(model)
    catalog = sw81.sw5_engine.build_catalog()
    support_adapter = sw81.sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT)
    sector_masks = sw7.build_sector_masks()
    rel_cfg = sw7.build_reliability_config()
    rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, query_holder)
    debug_examples: dict[str, Any] = {"topk_rows": []}
    try:
        for perturbation_id in CORE_PERTURBATIONS:
            spec = catalog[perturbation_id]
            for sample_index in sample_indices:
                raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
                sample_unwrapped = sw2.unwrap(raw_sample)
                if perturbation_id != "A0_clean":
                    batch, _ = sw81.sw5_engine.apply_perturbation_to_batch(batch, spec)
                query_holder.clear()
                sw81.reset_online_cache(model)
                model_inputs = sw2.move_to_cuda(batch)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **model_inputs)
                raw_result_cpu = sw2.to_cpu_artifact(result)
                query_cpu = sw2.to_cpu_artifact(query_holder)
                pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
                gt0 = gt_temporal[0]
                supports = support_adapter.build_temporal_support(query_cpu, horizons)[0] if capture_reliability else {}
                for horizon_s in horizons:
                    pred_h = pred_temporal[horizon_s].long().cpu()
                    gt_h = gt_temporal[horizon_s].long().cpu()
                    rows.append(compute_eval_row_sw91(checkpoint_name, perturbation_id, sample_index, horizon_s, pred_h, gt_h, gt0, sector_masks))
                    if capture_reliability:
                        pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
                        _, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                        debug = dbg_list[0]
                        maps = sw7.compute_reliability_maps(
                            perturbation_id=perturbation_id,
                            horizon_s=horizon_s,
                            pred=pred_h,
                            gt=gt_h,
                            gt0=gt0,
                            support=supports[horizon_s],
                            debug=debug,
                            config=rel_cfg,
                            sectors=sector_masks,
                        )
                        rel_rows.append(
                            {
                                "checkpoint_name": checkpoint_name,
                                "perturbation_id": perturbation_id,
                                "sample_index": sample_index,
                                "horizon_s": horizon_s,
                                **maps["metrics"],
                                "false_free_rate": rows[-1]["false_free_rate"],
                                "false_occupied_rate": rows[-1]["false_occupied_rate"],
                                "occupied_iou": rows[-1]["occupied_iou"],
                                "pred_gt_occupied_ratio": rows[-1]["pred_gt_occupied_ratio"],
                            }
                        )
                        if perturbation_id == "A10_drop_front_triplet" and sample_index == sample_indices[0] and horizon_s == 6:
                            debug_examples["representative_case"] = {
                                "pred": pred_h,
                                "maps": maps,
                            }
    finally:
        if model is not None and original_forward is not None:
            model.forward_backbone = original_forward  # type: ignore[assignment]
        model = None
        dataset = None
        cfg = None
        safe_cuda_cleanup()
    return rows, rel_rows, debug_examples


def aggregate_eval(rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        record = {k: v for k, v in zip(group_keys, key)}
        record["count"] = len(items)
        for value_key in value_keys:
            vals = [float(item[value_key]) for item in items if item.get(value_key) is not None]
            record[f"mean_{value_key}"] = float(np.mean(vals)) if vals else None
        out.append(record)
    return out


def compare_against_reference(candidate_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], checkpoint_name: str) -> dict[str, Any]:
    def eval_key(row: dict[str, Any]) -> tuple[str, int, int]:
        return (str(row["perturbation_id"]), int(row["sample_index"]), int(row["horizon_s"]))

    ref_map = {eval_key(row): row for row in reference_rows}
    deltas: list[dict[str, Any]] = []
    metric_names = [
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
    ]
    for row in candidate_rows:
        key = eval_key(row)
        ref = ref_map.get(key)
        if ref is None:
            continue
        record = {
            "checkpoint_name": checkpoint_name,
            "perturbation_id": row["perturbation_id"],
            "sample_index": row["sample_index"],
            "horizon_s": row["horizon_s"],
        }
        for metric_name in metric_names:
            record[f"{metric_name}_delta"] = float(row[metric_name]) - float(ref[metric_name])
        deltas.append(record)
    aggregate = aggregate_eval(deltas, ["checkpoint_name", "perturbation_id"], [f"{m}_delta" for m in metric_names])
    return {"deltas": deltas, "aggregate": aggregate}


def aggregate_candidate_deltas(cmp_payload: dict[str, Any], subset_name: str) -> list[dict[str, Any]]:
    rows = cmp_payload["aggregate"]
    for row in rows:
        row["subset_name"] = subset_name
    return rows


def safe_gate_from_rows(agg_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["perturbation_id"]: row for row in agg_rows}
    clean = lookup.get("A0_clean", {})
    a10 = lookup.get("A10_drop_front_triplet", {})
    c4 = lookup.get("C4_motion_blur_9", {})
    max_false_occ = max(float(row.get("mean_false_occupied_rate_delta") or 0.0) for row in agg_rows) if agg_rows else 0.0
    max_pred_gt = max(float(row.get("mean_pred_gt_occupied_ratio_delta") or 0.0) for row in agg_rows) if agg_rows else 0.0
    targeted_hit = (
        float(a10.get("mean_small_object_false_free_delta") or 0.0) <= -0.03
        or float(a10.get("mean_front_sector_false_free_delta") or 0.0) <= -0.03
        or float(a10.get("mean_new_visible_recall_delta") or 0.0) >= 0.03
        or (float(c4.get("mean_false_occupied_rate_delta") or 0.0) <= -0.005 and float(clean.get("mean_occupied_iou_delta") or 0.0) >= -0.005)
    )
    safe = (
        float(clean.get("mean_occupied_iou_delta") or 0.0) >= -0.005
        and float(clean.get("mean_semantic_miou_delta") or 0.0) >= -0.005
        and max_false_occ <= 0.008
        and max_pred_gt <= 0.10
    )
    return {
        "clean_occupied_iou_delta": float(clean.get("mean_occupied_iou_delta") or 0.0),
        "clean_semantic_miou_delta": float(clean.get("mean_semantic_miou_delta") or 0.0),
        "false_occupied_delta": max_false_occ,
        "pred_gt_ratio_delta": max_pred_gt,
        "a10_small_object_false_free_delta": float(a10.get("mean_small_object_false_free_delta") or 0.0),
        "a10_front_sector_false_free_delta": float(a10.get("mean_front_sector_false_free_delta") or 0.0),
        "a10_new_visible_recall_delta": float(a10.get("mean_new_visible_recall_delta") or 0.0),
        "c4_false_occupied_delta": float(c4.get("mean_false_occupied_rate_delta") or 0.0),
        "targeted_hit": targeted_hit,
        "safe": safe,
    }


def plot_gradient_sweep(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    labels = [row["experiment_id"] for row in rows]
    x = np.arange(len(rows))
    ax1.bar(x - 0.2, [float(row.get("grad_norm_total") or 0.0) for row in rows], width=0.4, label="grad_norm_total", color="#2E86AB")
    ax2 = ax1.twinx()
    ax2.plot(x, [float(row.get("h2_assign_loss") or 0.0) + float(row.get("h2_leak_loss") or 0.0) for row in rows], marker="o", color="#E67E22", label="h2_aux_loss")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("grad norm")
    ax2.set_ylabel("H2 aux loss")
    ax1.set_title(title)
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_curves(train_rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(row["experiment_id"])].append(row)
    for exp_id, rows in grouped.items():
        ax.plot([int(r["iter"]) for r in rows], [float(r["loss_total"]) for r in rows], label=exp_id)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss_total")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_h2_curve(rows: list[dict[str, Any]], metric: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment_id"])].append(row)
    for exp_id, items in grouped.items():
        xs = [int(r["iter"]) for r in items if r.get(metric) is not None]
        ys = [float(r[metric]) for r in items if r.get(metric) is not None]
        if xs and ys:
            ax.plot(xs, ys, label=exp_id)
    ax.set_xlabel("iter")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_delta_bar(eval_summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    focus_rows = [row for row in eval_summary_rows if row["perturbation_id"] == "A10_drop_front_triplet"]
    labels = [row["checkpoint_name"] for row in focus_rows]
    vals = [float(row.get("mean_front_sector_false_free_delta") or 0.0) for row in focus_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, vals, color="#C0392B")
    ax.set_ylabel("delta")
    ax.set_title("SW-9.1 subset diagnostic A10 front-sector false-free delta")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_false_positive_tradeoff(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    candidates = [row for row in summary_rows if row["checkpoint_name"] != "REF_epoch_56"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for row in candidates:
        gate = row["safe_gate"]
        ax.scatter(float(gate["pred_gt_ratio_delta"]), float(gate["false_occupied_delta"]), label=row["checkpoint_name"], s=70)
    ax.axvline(0.10, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.008, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("pred_gt_ratio_delta")
    ax.set_ylabel("false_occupied_delta")
    ax.set_title("SW-9.1 subset diagnostic false-positive / density tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_contributor_waterfall(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    metrics = ["radius_support_coverage", "exact_voxel_assignment_ratio", "final_contributor_ratio"]
    labels = [row["checkpoint_name"] for row in summary_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.22
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, [float(row.get(metric) or 0.0) for row in summary_rows], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("ratio")
    ax.set_title("SW-9.1 subset diagnostic contributor waterfall before/after")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def gradient_check_experiment(
    experiment_id: str,
    config_path: Path,
    seed: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_overrides = {"data.samples_per_gpu": samples_per_gpu, "data.workers_per_gpu": workers_per_gpu, "optimizer.lr": DEFAULT_LR}
    if cfg_overrides:
        merged_overrides.update(cfg_overrides)
    cfg, dataset, model, _ = sw9.build_runtime(config_path, train=True, cfg_overrides=merged_overrides)
    del cfg
    dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu, shuffle=False, seed=seed)
    batch = next(iter(dataloader))
    holder: dict[str, Any] = {}
    original_forward = sw9.attach_forward_capture(model, holder, retain_grads=True)
    try:
        set_h2_progress(model, 1)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model_inputs = sw81.move_train_batch_to_cuda(batch)
        losses = model(return_loss=True, **model_inputs)
        total_loss, log_vars = sw81.parse_losses(losses)
        has_nan = not bool(torch.isfinite(total_loss).item())
        total_loss.backward()
        torch.cuda.synchronize()
        grad_finite = True
        has_inf = False
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
            has_inf = has_inf or bool(torch.isinf(param.grad).any().item())
        grad_refs = holder.get("grad_refs", {})
        return {
            "experiment_id": experiment_id,
            "config_path": str(config_path),
            "total_loss": float(log_vars.get("loss_total", 0.0)),
            "original_loss": sw9.original_loss_from_log_vars(log_vars),
            "h2_assign_loss": sw9.sum_log_vars(log_vars, "loss_h2_assign"),
            "h2_leak_loss": sw9.sum_log_vars(log_vars, "loss_h2_leak"),
            "grad_finite": grad_finite,
            "grad_norm_total": sw9.total_grad_norm(model),
            "grad_norm_decoder": sw9.decoder_grad_norm(model),
            "grad_norm_forecast_path": sw9.tensor_grad_norm(grad_refs.get("forecast_points_last")),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "backward_time_sec": time.perf_counter() - started,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]


def run_h2_lambda_gradient_sweep(seed: int, samples_per_gpu: int) -> list[dict[str, Any]]:
    cfg = dataset = model = None
    original_forward = None
    supervisor = None
    base_state: dict[str, Any] = {}
    cfg, dataset, model, _ = sw9.build_runtime(
        CONFIG_PATHS["H2_ONLY_L001"],
        train=True,
        cfg_overrides={"data.samples_per_gpu": samples_per_gpu, "data.workers_per_gpu": 0, "optimizer.lr": DEFAULT_LR},
    )
    dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=0, shuffle=False, seed=seed)
    batch = next(iter(dataloader))
    holder: dict[str, Any] = {}
    original_forward = sw9.attach_forward_capture(model, holder, retain_grads=True)
    supervisor = getattr(model, "sw9_support_supervision", None)
    if supervisor is None:
        raise RuntimeError("SW-9.1 sweep expects sw9_support_supervision to be enabled")
    base_state = {
        "h1_enabled": bool(supervisor.h1_enabled),
        "h2_enabled": bool(supervisor.h2_enabled),
        "h3_enabled": bool(supervisor.h3_enabled),
        "lambda_h2": float(supervisor.lambda_h2),
        "lambda_leak": float(supervisor.lambda_leak),
        "h2_warmup_iters": int(supervisor.h2_warmup_iters),
        "h2_warmup_start_iter": int(supervisor.h2_warmup_start_iter),
    }
    sweep_specs = [
        ("G2a_H2_only_lambda0005", 0.0005, 0, 1),
        ("G2b_H2_only_lambda001", 0.0010, 0, 1),
        ("G2c_H2_only_lambda002", 0.0020, 0, 1),
        ("G2d_H2_only_lambda005", 0.0050, 0, 1),
        ("G2e_H2_warmup_target001_preview", 0.0010, 200, 1),
    ]
    rows: list[dict[str, Any]] = []
    try:
        for experiment_id, lambda_h2, warmup_iters, current_iter in sweep_specs:
            supervisor.h1_enabled = False
            supervisor.h2_enabled = True
            supervisor.h3_enabled = False
            supervisor.lambda_h2 = float(lambda_h2)
            supervisor.lambda_leak = 0.0005
            supervisor.h2_warmup_iters = int(warmup_iters)
            supervisor.h2_warmup_start_iter = 0
            set_h2_progress(model, current_iter)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model_inputs = sw81.move_train_batch_to_cuda(batch)
            losses = model(return_loss=True, **model_inputs)
            total_loss, log_vars = sw81.parse_losses(losses)
            has_nan = not bool(torch.isfinite(total_loss).item())
            total_loss.backward()
            torch.cuda.synchronize()
            grad_finite = True
            has_inf = False
            for param in model.parameters():
                if param.grad is None:
                    continue
                grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
                has_inf = has_inf or bool(torch.isinf(param.grad).any().item())
            grad_refs = holder.get("grad_refs", {})
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "config_path": str(CONFIG_PATHS["H2_ONLY_L001"] if warmup_iters == 0 else CONFIG_PATHS["H2_WARMUP_L001"]),
                    "total_loss": float(log_vars.get("loss_total", 0.0)),
                    "original_loss": sw9.original_loss_from_log_vars(log_vars),
                    "h2_assign_loss": sw9.sum_log_vars(log_vars, "loss_h2_assign"),
                    "h2_leak_loss": sw9.sum_log_vars(log_vars, "loss_h2_leak"),
                    "grad_finite": grad_finite,
                    "grad_norm_total": sw9.total_grad_norm(model),
                    "grad_norm_decoder": sw9.decoder_grad_norm(model),
                    "grad_norm_forecast_path": sw9.tensor_grad_norm(grad_refs.get("forecast_points_last")),
                    "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                    "backward_time_sec": time.perf_counter() - started,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                }
            )
    finally:
        if supervisor is not None:
            for key, value in base_state.items():
                setattr(supervisor, key, value)
        if model is not None and original_forward is not None:
            model.forward_backbone = original_forward  # type: ignore[assignment]
        supervisor = None
        model = None
        dataset = None
        cfg = None
        safe_cuda_cleanup()
    return rows


def choose_h2_lambda(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["experiment_id"]: row for row in rows}
    row_0005 = lookup["G2a_H2_only_lambda0005"]
    row_001 = lookup["G2b_H2_only_lambda001"]
    row_002 = lookup["G2c_H2_only_lambda002"]
    stable_001 = row_001["grad_finite"] and not row_001["has_nan"] and not row_001["has_inf"] and float(row_001["h2_assign_loss"]) + float(row_001["h2_leak_loss"]) <= float(row_001["original_loss"])
    stable_002 = row_002["grad_finite"] and not row_002["has_nan"] and not row_002["has_inf"] and float(row_002["h2_assign_loss"]) + float(row_002["h2_leak_loss"]) <= float(row_002["original_loss"]) * 1.1
    selected = 0.001 if stable_001 else 0.0005
    stronger = 0.002 if stable_002 else None
    reason = "lambda=0.001 stayed finite and did not overpower original loss" if selected == 0.001 else "lambda=0.001 looked too strong; fallback to 0.0005"
    return {
        "selected_lambda_h2": selected,
        "stronger_lambda_candidate": stronger,
        "reason": reason,
        "selected_config_key": "H2_ONLY_L001" if selected == 0.001 else "H2_ONLY_L0005",
    }


def build_sw9_reinterpretation(gradient_rows: list[dict[str, Any]], fixed_eval_rows: list[dict[str, Any]], decision_payload: dict[str, Any], tensor_audit: dict[str, Any]) -> dict[str, Any]:
    h1_row = next((row for row in fixed_eval_rows if row.get("checkpoint_name") == "P1_H1_lambda001" and row.get("perturbation_id") == "A0_clean"), {})
    g3 = next((row for row in gradient_rows if row.get("experiment_id") == "G3_H1_H2_lambda001"), {})
    other_forecast = [float(row.get("forecast_grad_norm") or 0.0) for row in gradient_rows if row.get("experiment_id") != "G3_H1_H2_lambda001"]
    reinterpretation = {
        "headline": "H1-only 500 iter was negative, but SW-9 never validated H2/H3 with 500/1000 iter short training; SW-9.1 therefore shifts to H2-centered contributor assignment training.",
        "h1_only_negative_signal": {
            "clean_occupied_iou_delta": float(h1_row.get("mean_occupied_iou_delta") or 0.0),
            "a10_small_object_false_free_delta": float(next((row.get("mean_small_object_false_free_delta") for row in fixed_eval_rows if row.get("checkpoint_name") == "P1_H1_lambda001" and row.get("perturbation_id") == "A10_drop_front_triplet"), 0.0) or 0.0),
            "a10_new_visible_recall_delta": float(next((row.get("mean_new_visible_recall_delta") for row in fixed_eval_rows if row.get("checkpoint_name") == "P1_H1_lambda001" and row.get("perturbation_id") == "A10_drop_front_triplet"), 0.0) or 0.0),
            "c4_false_free_rate_delta": float(next((row.get("mean_false_free_rate_delta") for row in fixed_eval_rows if row.get("checkpoint_name") == "P1_H1_lambda001" and row.get("perturbation_id") == "C4_motion_blur_9"), 0.0) or 0.0),
        },
        "cannot_rule_out_h2_h3": True,
        "why_cannot_rule_out_h2_h3": "SW-9 only short-trained H1-only for 500 iter; H2/H3 were gradient-checked but not validated with 500/1000 iter training.",
        "g3_h1_h2_gradient_signal": {
            "forecast_grad_norm": float(g3.get("forecast_grad_norm") or 0.0),
            "other_experiment_max_forecast_grad_norm": max(other_forecast) if other_forecast else 0.0,
            "support_forecast_path_strong_gradient": float(g3.get("forecast_grad_norm") or 0.0) >= max(other_forecast or [0.0]),
        },
        "tensor_access_headline": tensor_audit.get("tensor_access_headline"),
        "sw91_direction": "H2-centered contributor assignment training",
        "do_not_continue_h1_only": True,
        "do_not_modify_get_occ_mainline_in_sw91": True,
        "prior_decision_type": decision_payload.get("decision_type"),
    }
    return reinterpretation


def short_train_experiment(
    experiment_id: str,
    config_path: Path,
    target_iters: int,
    seed: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    deadline_ts: float,
    log_interval: int,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[Path | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    merged_overrides = {"data.samples_per_gpu": samples_per_gpu, "data.workers_per_gpu": workers_per_gpu, "optimizer.lr": DEFAULT_LR}
    if cfg_overrides:
        merged_overrides.update(cfg_overrides)
    cfg = dataset = model = checkpoint = dataloader = optimizer = base_batch = None
    iter_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    completed_iters = 0
    stop_reason = "completed"
    failure_trace = None
    start_epoch = 56
    ckpt_path: Path | None = None
    summary: dict[str, Any]
    dataloader_workers = workers_per_gpu
    try:
        cfg, dataset, model, checkpoint = sw9.build_runtime(config_path, train=True, cfg_overrides=merged_overrides)
        dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=dataloader_workers, shuffle=True, seed=seed)
        from mmcv.runner import build_optimizer

        optimizer = build_optimizer(model, cfg.optimizer)
        try:
            base_batch = next(iter(dataloader))
        except (OSError, RuntimeError) as exc:
            memory_fetch_error = "cannot allocate memory" in str(exc).lower() or "dataloader worker process" in str(exc).lower()
            if dataloader_workers > 0 and memory_fetch_error:
                dataloader_workers = 0
                dataloader = sw81.build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=0, shuffle=True, seed=seed)
                base_batch = next(iter(dataloader))
            else:
                raise
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
            start_epoch = int(checkpoint["meta"].get("epoch", start_epoch))
        if hasattr(model, "set_epoch"):
            model.set_epoch(start_epoch)
        for iter_idx in range(1, int(target_iters) + 1):
            if time.time() >= deadline_ts:
                stop_reason = "budget_exhausted"
                break
            try:
                optimizer.zero_grad(set_to_none=True)
                set_h2_progress(model, iter_idx)
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
                sw9_debug = getattr(model, "latest_sw9_debug", {"records": []})
                diag = aggregate_sw9_debug(sw9_debug)
                row = {
                    "experiment_id": experiment_id,
                    "iter": iter_idx,
                    "iter_time_sec": time.perf_counter() - started,
                    "peak_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                    "grad_norm": grad_norm,
                    **log_vars,
                    **diag,
                }
                iter_rows.append(row)
                if iter_idx % 100 == 0 or iter_idx == target_iters:
                    diag_rows.append(
                        {
                            "experiment_id": experiment_id,
                            "iter": iter_idx,
                            **diag,
                        }
                    )
                if iter_idx % log_interval == 0 or iter_idx == target_iters:
                    with (LOGS_DIR / f"{experiment_id}.log").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(normalize_export(row), ensure_ascii=False) + "\n")
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    stop_reason = "oom"
                    safe_cuda_cleanup()
                    break
                stop_reason = "runtime_error"
                failure_trace = traceback.format_exc()
                break
        if completed_iters == 0:
            summary = {
                "experiment_id": experiment_id,
                "config_path": str(config_path),
                "completed_iters": 0,
                "planned_iters": target_iters,
                "completed": False,
                "meets_500_iter_requirement": False,
                "effective_workers_per_gpu": dataloader_workers,
                "stop_reason": stop_reason,
                "failure_trace": failure_trace,
            }
            return None, iter_rows, diag_rows, summary
        ckpt_path = ARTIFACTS_DIR / "checkpoints" / f"{experiment_id}.pth"
        sw81.save_checkpoint(
            ckpt_path,
            model,
            {
                "experiment_id": experiment_id,
                "epoch": start_epoch,
                "iter": completed_iters,
                "planned_iters": target_iters,
                "config_path": str(config_path),
                "optimizer_lr": DEFAULT_LR,
            },
        )
        head = iter_rows[: min(10, len(iter_rows))]
        tail = iter_rows[-min(10, len(iter_rows)) :]
        last_diag = aggregate_sw9_debug(getattr(model, "latest_sw9_debug", {"records": []}))
        summary = {
            "experiment_id": experiment_id,
            "config_path": str(config_path),
            "checkpoint_path": str(ckpt_path),
            "data_mode": "fixed_first_batch_repeat",
            "completed_iters": completed_iters,
            "planned_iters": target_iters,
            "completed": completed_iters >= target_iters,
            "meets_500_iter_requirement": completed_iters >= 500,
            "effective_workers_per_gpu": dataloader_workers,
            "stop_reason": stop_reason,
            "peak_gpu_memory_mb": float(max(row["peak_memory_mb"] for row in iter_rows)),
            "mean_iter_time_sec": float(np.mean([row["iter_time_sec"] for row in iter_rows])),
            "loss_head_mean": float(np.mean([row["loss_total"] for row in head])) if head else None,
            "loss_tail_mean": float(np.mean([row["loss_total"] for row in tail])) if tail else None,
            "loss_trend_delta": float(np.mean([row["loss_total"] for row in tail]) - np.mean([row["loss_total"] for row in head])) if head and tail else None,
            **last_diag,
        }
        return ckpt_path, iter_rows, diag_rows, summary
    finally:
        del base_batch, optimizer, dataloader, checkpoint, model, dataset, cfg
        safe_cuda_cleanup()


def rank_candidate(row: dict[str, Any]) -> float:
    gate = row["safe_gate"]
    return (
        max(0.0, -float(gate["a10_front_sector_false_free_delta"]))
        + max(0.0, -float(gate["a10_small_object_false_free_delta"]))
        + max(0.0, float(gate["a10_new_visible_recall_delta"]))
        - max(0.0, float(gate["false_occupied_delta"])) * 4.0
        - max(0.0, -float(gate["clean_occupied_iou_delta"])) * 6.0
    )


def run_contributor_diagnostic(
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_name: str,
    sample_indices: list[int],
    horizons: list[int],
    perturbations: list[str],
) -> list[dict[str, Any]]:
    cfg = dataset = model = None
    original_forward = None
    cfg, dataset, model, _ = sw9.build_runtime(config_path, train=False, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    from mmcv.parallel import collate as collate_fn

    head = sw4_inst.get_pts_bbox_head(model)
    sector_masks = sw7.build_sector_masks()
    catalog = sw81.sw5_engine.build_catalog()
    rows: list[dict[str, Any]] = []
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, query_holder)
    try:
        for perturbation_id in perturbations:
            spec = catalog[perturbation_id]
            for sample_index in sample_indices:
                raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
                sample_unwrapped = sw2.unwrap(raw_sample)
                if perturbation_id != "A0_clean":
                    batch, _ = sw81.sw5_engine.apply_perturbation_to_batch(batch, spec)
                query_holder.clear()
                sw81.reset_online_cache(model)
                model_inputs = sw2.move_to_cuda(batch)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **model_inputs)
                raw_result_cpu = sw2.to_cpu_artifact(result)
                query_cpu = sw2.to_cpu_artifact(query_holder)
                pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
                gt0 = gt_temporal[0].long().cpu()
                for horizon_s in horizons:
                    pred_h = pred_temporal[horizon_s].long().cpu()
                    gt_h = gt_temporal[horizon_s].long().cpu()
                    pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
                    _, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=False)
                    dbg = dbg_list[0]
                    geometric_mask = dbg["geometric_mask"].cpu()
                    semantic_active = dbg["semantic_active_mask"].cpu()
                    final_occ = dbg["occ_pred"].cpu() != sw2.EMPTY_IDX
                    gt_occ = gt_h != sw2.EMPTY_IDX
                    cover_r2 = sw4.dilate3d(geometric_mask, 2)
                    region_masks = sw4.build_temporal_region_masks(gt0, gt_h, pred_h, cover_r2)
                    front_gt = gt_occ & sector_masks["front"].cpu()
                    small_gt = gt_occ & sw4.group_mask(gt_h, sw4.SMALL_OBJECT)
                    new_visible_gt = gt_occ & ((gt0 == sw2.EMPTY_IDX) & gt_occ)
                    neighbor_region = sw4.dilate3d(gt_occ, 1) & (~gt_occ)
                    rows.append(
                        {
                            "checkpoint_name": checkpoint_name,
                            "perturbation_id": perturbation_id,
                            "sample_index": sample_index,
                            "horizon_s": horizon_s,
                            "radius_support_coverage": safe_div((cover_r2 & gt_occ).sum().item(), gt_occ.sum().item()),
                            "exact_voxel_assignment_ratio": safe_div((semantic_active & gt_occ).sum().item(), gt_occ.sum().item()),
                            "final_contributor_ratio": safe_div((final_occ & gt_occ).sum().item(), gt_occ.sum().item()),
                            "neighbor_leakage_ratio": safe_div((final_occ & neighbor_region).sum().item(), neighbor_region.sum().item()),
                            "front_sector_contributor_ratio": safe_div((final_occ & front_gt).sum().item(), front_gt.sum().item()),
                            "small_object_contributor_ratio": safe_div((final_occ & small_gt).sum().item(), small_gt.sum().item()),
                            "new_visible_contributor_ratio": safe_div((final_occ & new_visible_gt).sum().item(), new_visible_gt.sum().item()),
                            "covered_false_free_contributor_count": int((semantic_active & region_masks["covered_false_free"]).sum().item()),
                            "tp_contributor_count": int((semantic_active & region_masks["covered_tp"]).sum().item()),
                        }
                    )
    finally:
        if model is not None and original_forward is not None:
            model.forward_backbone = original_forward  # type: ignore[assignment]
        model = None
        dataset = None
        cfg = None
        safe_cuda_cleanup()
    return rows


def aggregate_contributor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    value_keys = [
        "radius_support_coverage",
        "exact_voxel_assignment_ratio",
        "final_contributor_ratio",
        "neighbor_leakage_ratio",
        "front_sector_contributor_ratio",
        "small_object_contributor_ratio",
        "new_visible_contributor_ratio",
        "covered_false_free_contributor_count",
        "tp_contributor_count",
    ]
    return aggregate_eval(rows, ["checkpoint_name"], value_keys)


def make_final_report(report_json: dict[str, Any]) -> str:
    lines = [
        "# Stage SW-9.1 H2-centered Contributor Assignment Training",
        "",
        "1. Executive summary",
        f"- {report_json['executive_summary']}",
        "",
        "2. Why SW-9.1 follows SW-9",
        "- SW-9.1 is H2-centered contributor assignment training.",
        "- This is a subset diagnostic, not an official benchmark.",
        "- This is not full training and does not claim to surpass prior paper results.",
        "",
        "3. SW-9 reinterpretation",
        f"- {report_json['sw9_reinterpretation']['headline']}",
        "",
        "4. H2 lambda gradient sweep",
        f"- {report_json['h2_lambda_sweep_summary']['headline']}",
        "",
        "5. Configs and training setup",
        f"- Selected lambda_h2={report_json['config_manifest']['selected_lambda_h2']}",
        "",
        "6. Short training results",
        f"- {report_json['short_train_summary']['headline']}",
        "",
        "7. H2 assignment diagnostics",
        f"- {report_json['h2_assignment_summary']['headline']}",
        "",
        "8. Fixed subset eval",
        f"- {report_json['fixed_subset_eval_summary']['headline']}",
        "",
        "9. Contributor diagnostic retest",
        f"- {report_json['contributor_retest_summary']['headline']}",
        "",
        "10. SW-7 reliability retest",
        f"- {report_json['reliability_retest_summary']['headline']}",
        "",
        "11. Decision D1-D7",
        f"- {report_json['decision']['decision_type']}: {report_json['decision']['summary']}",
        "",
        "12. Safe claims",
        "- not official benchmark",
        "- not full training",
        "- no claim of surpassing prior paper results",
        "- 1000 iter subset result is not a final performance conclusion",
        "- false-positive / pred_gt_ratio tradeoff is reported explicitly",
        "",
        "13. Limitations",
        "- contributor/reliability retests are subset diagnostics, not full validation",
        "- if eval stays flat while H2 proxy improves, proxy-to-native get_occ alignment remains incomplete",
        "",
        "14. Next unique action",
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
    time_manifest: dict[str, Any] = {
        "stage": "SW-9.1",
        "start_time": now_iso(),
        "max_hours": args.max_hours,
        "reserve_report_minutes": args.reserve_report_minutes,
        "phases": [],
        "experiments": [],
    }
    time_manifest_path = REPORTS_DIR / "sw91_time_budget_manifest.json"
    run_time_manifest_update(time_manifest_path, time_manifest)

    def phase_block(name: str):
        phase_meta = {"phase_name": name, "start_time": now_iso(), "start_ts": time.time()}
        time_manifest["phases"].append(phase_meta)
        run_time_manifest_update(time_manifest_path, time_manifest)
        return phase_meta

    def end_phase(phase_meta: dict[str, Any], status: str, **extra: Any) -> None:
        end_ts = time.time()
        phase_meta.update({"end_time": now_iso(), "end_ts": end_ts, "duration_sec": end_ts - float(phase_meta.get("start_ts", end_ts)), "status": status, **extra})
        run_time_manifest_update(time_manifest_path, time_manifest)

    # Phase 1
    phase = phase_block("phase1_sw9_reinterpretation")
    stage_logger.start("phase1_sw9_reinterpretation")
    sw9_gradient_rows = read_csv_rows(SW9_REPORTS / "sw9_gradient_check.csv")
    sw9_fixed_rows = read_csv_rows(SW9_REPORTS / "sw9_fixed_subset_eval.csv")
    sw9_decision = json.loads((SW9_REPORTS / "sw9_contributor_supervision_decision.json").read_text(encoding="utf-8"))
    sw9_tensor_audit = json.loads((SW9_REPORTS / "sw9_tensor_access_audit.json").read_text(encoding="utf-8"))
    reinterpretation = build_sw9_reinterpretation(sw9_gradient_rows, sw9_fixed_rows, sw9_decision, sw9_tensor_audit)
    write_json(REPORTS_DIR / "sw9_result_reinterpretation_for_sw91.json", reinterpretation)
    write_md(REPORTS_DIR / "sw9_result_reinterpretation_for_sw91.md", reinterpretation["headline"] + "\n")
    stage_logger.done("phase1_sw9_reinterpretation")
    end_phase(phase, "done")

    # Phase 2
    phase = phase_block("phase2_h2_lambda_sweep")
    stage_logger.start("phase2_h2_lambda_sweep")
    gradient_sweep_rows = run_h2_lambda_gradient_sweep(args.seed, args.samples_per_gpu)
    lambda_choice = choose_h2_lambda(gradient_sweep_rows)
    write_csv(REPORTS_DIR / "h2_lambda_gradient_sweep.csv", gradient_sweep_rows)
    write_md(
        REPORTS_DIR / "h2_lambda_gradient_sweep_summary.md",
        f"Selected lambda_h2={lambda_choice['selected_lambda_h2']:.4f}; {lambda_choice['reason']}.\n",
    )
    plot_gradient_sweep(gradient_sweep_rows, FIGURES_DIR / "h2_lambda_vs_grad_norm.png", "SW-9.1 subset diagnostic H2 lambda sweep")
    plot_gradient_sweep(gradient_sweep_rows, FIGURES_DIR / "sw91_h2_lambda_gradient_sweep.png", "SW-9.1 subset diagnostic H2 lambda sweep")
    stage_logger.done("phase2_h2_lambda_sweep", selected_lambda=lambda_choice["selected_lambda_h2"])
    end_phase(phase, "done", selected_lambda=lambda_choice["selected_lambda_h2"])

    # Phase 3
    phase = phase_block("phase3_config_manifest")
    stage_logger.start("phase3_config_manifest")
    config_manifest = {
        "base_config": str(BASE_CONFIG_PATH),
        "resume_checkpoint": str(CHECKPOINT_PATH),
        "lr_scale": 0.1,
        "samples_per_gpu": 2,
        "workers_per_gpu": 6,
        "selected_lambda_h2": lambda_choice["selected_lambda_h2"],
        "stronger_lambda_candidate": lambda_choice["stronger_lambda_candidate"],
        "lambda_leak": 0.0005,
        "r_assign_voxel": 1.0,
        "r_leak_voxel": 2.5,
        "temperature": 1.0,
        "beta_h3": 0.3,
        "max_gt_voxels": 4096,
        "max_support_points": 4096,
        "sampling_seed": 17,
        "config_ids": {key: str(value) for key, value in CONFIG_PATHS.items()},
    }
    write_json(REPORTS_DIR / "sw91_config_manifest.json", config_manifest)
    write_md(REPORTS_DIR / "sw91_config_summary.md", f"SW-9.1 uses H2-centered configs with selected lambda_h2={lambda_choice['selected_lambda_h2']:.4f}.\n")
    stage_logger.done("phase3_config_manifest")
    end_phase(phase, "done")

    # Phase 4/5 training
    phase = phase_block("phase4_phase5_short_training")
    stage_logger.start("phase4_phase5_short_training")
    short_plan = parse_short_train_plan(args.short_train_plan)
    selected_h2_cfg = CONFIG_PATHS[lambda_choice["selected_config_key"]]
    train_plan: dict[str, tuple[Path, int, dict[str, Any] | None]] = {
        "P0_control_1000iter": (BASE_CONFIG_PATH, short_plan.get("P0_control_1000iter", 1000), None),
        "P1_H2_only_low_1000iter": (selected_h2_cfg, short_plan.get("P1_H2_only_low_1000iter", 1000), None),
        "P2_H2_warmup_1000iter": (CONFIG_PATHS["H2_WARMUP_L001"], short_plan.get("P2_H2_warmup_1000iter", 1000), None),
        "P3_H2_H3_1000iter": (CONFIG_PATHS["H2_H3_L001_B03"], short_plan.get("P3_H2_H3_1000iter", 1000), None),
        "P4_H2_tinyH1_500iter": (CONFIG_PATHS["H2_TINY_H1_L001"], short_plan.get("P4_H2_tinyH1_500iter", 500), None),
    }
    if lambda_choice["stronger_lambda_candidate"] is not None:
        train_plan["P5_H2_stronger_500iter"] = (
            CONFIG_PATHS["H2_ONLY_L001"],
            short_plan.get("P5_H2_stronger_500iter", 500),
            {"model.sw9_support_supervision.lambda_h2": lambda_choice["stronger_lambda_candidate"]},
        )
    train_priority = [TRAIN_ALIASES.get(item.strip(), item.strip()) for item in args.train_priority.split(",") if item.strip()]
    if args.fast:
        train_priority = [name for name in train_priority if name in {"P0_control_1000iter", "P1_H2_only_low_1000iter", "P2_H2_warmup_1000iter", "P3_H2_H3_1000iter"}]
    train_summaries: list[dict[str, Any]] = []
    train_iter_rows: list[dict[str, Any]] = []
    h2_diag_rows: list[dict[str, Any]] = []
    completed_checkpoints: dict[str, tuple[Path, Path]] = {}
    for idx, exp_name in enumerate(train_priority, start=1):
        if exp_name not in train_plan:
            continue
        config_path, target_iters, overrides = train_plan[exp_name]
        actual_target = min(target_iters, 20) if args.fast else target_iters
        exp_start = {"experiment_id": exp_name, "start_time": now_iso(), "planned_iters": actual_target}
        time_manifest["experiments"].append(exp_start)
        run_time_manifest_update(time_manifest_path, time_manifest)
        ckpt_path, iter_rows, diag_rows, summary = short_train_experiment(
            exp_name,
            config_path,
            actual_target,
            args.seed,
            args.samples_per_gpu,
            args.workers_per_gpu,
            deadline_ts - args.reserve_report_minutes * 60.0,
            args.log_interval,
            overrides,
        )
        train_summaries.append(summary)
        train_iter_rows.extend(iter_rows)
        h2_diag_rows.extend(diag_rows)
        exp_start.update({"end_time": now_iso(), "status": summary["stop_reason"], "completed_iters": summary["completed_iters"]})
        if ckpt_path is not None:
            completed_checkpoints[exp_name] = (ckpt_path, config_path)
            exp_start["checkpoint_path"] = str(ckpt_path)
        run_time_manifest_update(time_manifest_path, time_manifest)
        stage_logger.progress("phase4_phase5_short_training", idx, len(train_priority), experiment_id=exp_name, stop_reason=summary["stop_reason"])
    write_csv(REPORTS_DIR / "sw91_short_train_metrics.csv", train_summaries)
    write_csv(REPORTS_DIR / "h2_assignment_diagnostics.csv", h2_diag_rows or [{"experiment_id": "none", "skipped_reason": "no_h2_diagnostics"}])
    if train_iter_rows:
        plot_loss_curves(train_iter_rows, FIGURES_DIR / "sw91_h2_loss_curves.png", "SW-9.1 subset diagnostic short-train loss curves")
        plot_h2_curve(train_iter_rows, "mean_h2_assign_loss", FIGURES_DIR / "h2_assign_loss_curve.png", "SW-9.1 subset diagnostic H2 assign loss")
        plot_h2_curve(train_iter_rows, "mean_h2_leak_loss", FIGURES_DIR / "h2_leakage_curve.png", "SW-9.1 subset diagnostic H2 leakage loss")
        plot_h2_curve(train_iter_rows, "mean_support_to_gt_distance", FIGURES_DIR / "h2_support_distance_curve.png", "SW-9.1 subset diagnostic support-to-GT distance")
    stage_logger.done("phase4_phase5_short_training", completed_count=len(completed_checkpoints))
    end_phase(phase, "done", completed_count=len(completed_checkpoints))

    # Phase 6 eval
    phase = phase_block("phase6_fixed_subset_eval")
    stage_logger.start("phase6_fixed_subset_eval")
    eval_subsets, eval_union = build_eval_subsets(args)
    if args.fast:
        subset_name, sample_indices = "quick_eval_10", eval_subsets["quick_eval_10"]
    else:
        elapsed_hours = (time.time() - started_at) / 3600.0
        subset_name, sample_indices = choose_eval_subset(elapsed_hours, args.max_hours, args.reserve_report_minutes / 60.0)
    reference_rows, _, _ = evaluate_checkpoint_detailed(BASE_CONFIG_PATH, CHECKPOINT_PATH, "REF_epoch_56", sample_indices, DEFAULT_HORIZONS, capture_reliability=False)
    eval_flat_rows: list[dict[str, Any]] = []
    eval_summary_rows: list[dict[str, Any]] = []
    baseline_candidates: list[tuple[str, Path, Path]] = []
    if "P0_control_1000iter" in completed_checkpoints:
        baseline_candidates.append(("P0_control_1000iter", *completed_checkpoints["P0_control_1000iter"]))
    if not args.fast:
        h1_sw9_ckpt = SW9_ARTIFACTS / "checkpoints/P1_H1_lambda001_iter0500.pth"
        h1_sw9_cfg = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw9_h1_support_coverage_lambda001.py"
        if h1_sw9_ckpt.exists():
            baseline_candidates.append(("SW9_P1_H1_lambda001", h1_sw9_ckpt, h1_sw9_cfg))
        for path_name in [
            "E4_future_horizon_reweight_lr0.1x_iter1000.pth",
            "E8_motion_blur_aug_light_lr0.1x_iter1000.pth",
            "E1_small_object_reweight_light_lr0.1x_iter1000.pth",
        ]:
            ckpt = SW81_ARTIFACTS / f"checkpoints/{path_name}"
            if ckpt.exists():
                baseline_candidates.append((path_name.replace(".pth", ""), ckpt, BASE_CONFIG_PATH))
    seen_eval_names: set[str] = set()
    candidate_list: list[tuple[str, Path, Path]] = []
    for item in baseline_candidates + [(name, ckpt, cfg) for name, (ckpt, cfg) in completed_checkpoints.items()]:
        if item[0] in seen_eval_names:
            continue
        seen_eval_names.add(item[0])
        candidate_list.append(item)
    for idx, (name, ckpt_path, config_path) in enumerate(candidate_list, start=1):
        rows, _, _ = evaluate_checkpoint_detailed(config_path, ckpt_path, name, sample_indices, DEFAULT_HORIZONS, capture_reliability=False)
        cmp_payload = compare_against_reference(rows, reference_rows, name)
        agg_rows = aggregate_candidate_deltas(cmp_payload, subset_name)
        gate = safe_gate_from_rows(agg_rows)
        for row in agg_rows:
            row["gate_safe"] = gate["safe"]
            row["gate_targeted_hit"] = gate["targeted_hit"]
            row["safe_gate_json"] = json.dumps(gate, ensure_ascii=False)
        eval_flat_rows.extend(agg_rows)
        eval_summary_rows.append({"checkpoint_name": name, "safe_gate": gate, "aggregate_rows": agg_rows})
        stage_logger.progress("phase6_fixed_subset_eval", idx, len(candidate_list), checkpoint_name=name)
    if eval_flat_rows:
        write_csv(REPORTS_DIR / "sw91_fixed_subset_eval.csv", eval_flat_rows)
    else:
        write_csv(REPORTS_DIR / "sw91_fixed_subset_eval.csv", [{"checkpoint_name": "none", "skipped_reason": "no checkpoints available"}])
    fixed_eval_headline = f"{subset_name} used for SW-9.1 subset diagnostic; clean drift and false-positive / pred_gt_ratio gates were applied against epoch_56 reference"
    write_md(REPORTS_DIR / "sw91_fixed_subset_eval_summary.md", fixed_eval_headline + "\n")
    if eval_flat_rows:
        plot_metric_delta_bar(eval_flat_rows, FIGURES_DIR / "sw91_metric_delta_bar.png")
        plot_false_positive_tradeoff(eval_summary_rows, FIGURES_DIR / "sw91_false_positive_density_tradeoff.png")
    stage_logger.done("phase6_fixed_subset_eval", checkpoint_count=len(eval_summary_rows))
    end_phase(phase, "done", subset_name=subset_name)

    # Phase 7 contributor retest
    phase = phase_block("phase7_contributor_retest")
    stage_logger.start("phase7_contributor_retest")
    contributor_rows: list[dict[str, Any]] = []
    contributor_headline = "skipped because no SW-9.1 candidate checkpoint completed"
    contributor_summary_rows: list[dict[str, Any]] = []
    if completed_checkpoints:
        ranked_eval = [row for row in eval_summary_rows if row["checkpoint_name"] in completed_checkpoints]
        best_h2 = max((row for row in ranked_eval if row["checkpoint_name"] in {"P1_H2_only_low_1000iter", "P2_H2_warmup_1000iter", "P4_H2_tinyH1_500iter", "P5_H2_stronger_500iter"}), key=rank_candidate, default=None)
        best_h2_h3 = next((row for row in ranked_eval if row["checkpoint_name"] == "P3_H2_H3_1000iter"), None)
        selected = []
        if best_h2 is not None:
            selected.append(best_h2["checkpoint_name"])
        if best_h2_h3 is not None and best_h2_h3["checkpoint_name"] not in selected:
            selected.append(best_h2_h3["checkpoint_name"])
        selected = selected[:2]
        retest_count = 2 if args.fast else 5
        baseline_contrib = run_contributor_diagnostic(BASE_CONFIG_PATH, CHECKPOINT_PATH, "REF_epoch_56", sample_indices[: min(retest_count, len(sample_indices))], RETEST_HORIZONS, RETEST_PERTURBATIONS)
        contributor_rows.extend(baseline_contrib)
        for checkpoint_name in selected:
            ckpt_path, cfg_path = completed_checkpoints[checkpoint_name]
            contributor_rows.extend(run_contributor_diagnostic(cfg_path, ckpt_path, checkpoint_name, sample_indices[: min(retest_count, len(sample_indices))], RETEST_HORIZONS, RETEST_PERTURBATIONS))
        contributor_summary_rows = aggregate_contributor_rows(contributor_rows)
        if contributor_summary_rows:
            contributor_headline = "best H2 candidate and best H2+H3 candidate were retested with subset contributor / exact-assignment diagnostics"
            plot_contributor_waterfall(contributor_summary_rows, FIGURES_DIR / "sw91_contributor_waterfall_before_after.png")
    write_csv(REPORTS_DIR / "sw91_contributor_diagnostic_retest.csv", contributor_rows or [{"checkpoint_name": "none", "skipped_reason": contributor_headline}])
    write_md(REPORTS_DIR / "sw91_contributor_diagnostic_summary.md", contributor_headline + "\n")
    stage_logger.done("phase7_contributor_retest")
    end_phase(phase, "done")

    # Phase 8 reliability
    phase = phase_block("phase8_reliability_retest")
    stage_logger.start("phase8_reliability_retest")
    reliability_rows: list[dict[str, Any]] = []
    reliability_headline = "skipped because no SW-9.1 candidate checkpoint completed"
    if completed_checkpoints:
        retest_count = 2 if args.fast else 5
        reference_eval_rows, reference_rel_rows, reference_dbg = evaluate_checkpoint_detailed(
            BASE_CONFIG_PATH,
            CHECKPOINT_PATH,
            "REF_epoch_56",
            sample_indices[: min(retest_count, len(sample_indices))],
            RETEST_HORIZONS,
            capture_reliability=True,
        )
        _ = reference_eval_rows
        reliability_rows.extend(reference_rel_rows)
        selected_for_rel = []
        if "P2_H2_warmup_1000iter" in completed_checkpoints:
            selected_for_rel.append("P2_H2_warmup_1000iter")
        if "P3_H2_H3_1000iter" in completed_checkpoints:
            selected_for_rel.append("P3_H2_H3_1000iter")
        if not selected_for_rel:
            selected_for_rel = list(completed_checkpoints.keys())[:2]
        for checkpoint_name in selected_for_rel[:2]:
            ckpt_path, cfg_path = completed_checkpoints[checkpoint_name]
            _, rel_rows, dbg = evaluate_checkpoint_detailed(cfg_path, ckpt_path, checkpoint_name, sample_indices[: min(retest_count, len(sample_indices))], RETEST_HORIZONS, capture_reliability=True)
            reliability_rows.extend(rel_rows)
            if reference_dbg.get("representative_case") and dbg.get("representative_case"):
                sw81.render_reliability_before_after(
                    "sw91_reliability",
                    reference_dbg["representative_case"],
                    dbg["representative_case"],
                    FIGURES_DIR / "sw91_reliability_before_after.png",
                )
                sw81.render_reliability_before_after(
                    "sw91_score_alpha",
                    reference_dbg["representative_case"],
                    dbg["representative_case"],
                    FIGURES_DIR / "sw91_score_alpha_before_after.png",
                )
                break
        if reliability_rows:
            reliability_headline = "subset SW-7 reliability retest completed for the best safe H2 candidate and the H2+H3 candidate"
    write_csv(REPORTS_DIR / "sw91_reliability_retest.csv", reliability_rows or [{"checkpoint_name": "none", "skipped_reason": reliability_headline}])
    write_md(REPORTS_DIR / "sw91_reliability_retest_summary.md", reliability_headline + "\n")
    stage_logger.done("phase8_reliability_retest")
    end_phase(phase, "done")

    # Phase 9 decision
    phase = phase_block("phase9_decision")
    stage_logger.start("phase9_decision")
    gradient_trainable = any(row["grad_finite"] and not row["has_nan"] and not row["has_inf"] for row in gradient_sweep_rows)
    best_candidate_eval = max((row for row in eval_summary_rows if row["checkpoint_name"] in completed_checkpoints), key=rank_candidate, default=None)
    best_h2h3_eval = next((row for row in eval_summary_rows if row["checkpoint_name"] == "P3_H2_H3_1000iter"), None)
    contributor_lookup = {row["checkpoint_name"]: row for row in contributor_summary_rows}
    reference_contrib = contributor_lookup.get("REF_epoch_56")
    proxy_improved = False
    if best_candidate_eval is not None and reference_contrib is not None and best_candidate_eval["checkpoint_name"] in contributor_lookup:
        candidate_contrib = contributor_lookup[best_candidate_eval["checkpoint_name"]]
        proxy_improved = (
            float(candidate_contrib.get("mean_radius_support_coverage") or 0.0) >= float(reference_contrib.get("mean_radius_support_coverage") or 0.0)
            and float(candidate_contrib.get("mean_exact_voxel_assignment_ratio") or 0.0) > float(reference_contrib.get("mean_exact_voxel_assignment_ratio") or 0.0)
        )
    if not gradient_trainable:
        decision_type = "D1_h2_not_trainable"
        decision_summary = "H2 gradient or memory behavior was not stable enough for short training."
        next_action = "stabilize H2 numerical range before any longer run."
    elif best_h2h3_eval is not None and best_candidate_eval is not None and rank_candidate(best_h2h3_eval) > rank_candidate(best_candidate_eval) and best_h2h3_eval["safe_gate"]["safe"]:
        decision_type = "D6_h2_h3_best_promising"
        decision_summary = "H2+H3 outperformed the best H2-only candidate under the subset diagnostic gates."
        next_action = "extend H2+H3 with the same low-lambda regime before considering routing changes."
    elif best_candidate_eval is not None and best_candidate_eval["safe_gate"]["safe"] and best_candidate_eval["safe_gate"]["targeted_hit"]:
        decision_type = "D4_h2_safe_targeted_improvement"
        decision_summary = "At least one H2-centered candidate improved target metrics while staying inside clean / false-positive / density gates."
        next_action = "scale the best safe H2 candidate beyond 1000 iter and rerun eval_core_20."
    elif best_candidate_eval is not None and best_candidate_eval["safe_gate"]["targeted_hit"] and not best_candidate_eval["safe_gate"]["safe"]:
        decision_type = "D5_h2_tradeoff"
        decision_summary = "Target metrics moved in the right direction, but clean drift or false-positive / density expansion exceeded the gate."
        next_action = "tighten leakage or routing before longer training."
    elif proxy_improved and best_candidate_eval is not None:
        decision_type = "D3_h2_proxy_improves_no_metric_gain"
        decision_summary = "Contributor / assignment proxy improved, but final occupancy metrics stayed flat under the subset diagnostic."
        next_action = "run a longer H2-centered training window before changing get_occ routing."
    elif best_candidate_eval is not None and not proxy_improved:
        decision_type = "D2_h2_trainable_no_proxy_change"
        decision_summary = "H2 trained numerically, but the assignment proxy did not move enough to support the route."
        next_action = "audit H2 alignment first; if repeated, prepare SW-10 get_occ routing changes."
    else:
        decision_type = "D7_move_to_SW10_getocc_routing"
        decision_summary = "H2-centered short training did not show proxy or final-metric signal in the available budget."
        next_action = "prepare SW-10 get_occ assignment / decoder routing modification."
    decision_payload = {
        "decision_type": decision_type,
        "summary": decision_summary,
        "gradient_trainable": gradient_trainable,
        "proxy_improved": proxy_improved,
        "best_candidate": None if best_candidate_eval is None else best_candidate_eval["checkpoint_name"],
        "false_positive_pred_gt_tradeoff": None if best_candidate_eval is None else {
            "false_occupied_delta": best_candidate_eval["safe_gate"]["false_occupied_delta"],
            "pred_gt_ratio_delta": best_candidate_eval["safe_gate"]["pred_gt_ratio_delta"],
        },
        "next_unique_action": next_action,
    }
    write_json(REPORTS_DIR / "sw91_h2_contributor_assignment_decision.json", decision_payload)
    write_md(REPORTS_DIR / "sw91_h2_contributor_assignment_decision.md", json.dumps(normalize_export(decision_payload), indent=2, ensure_ascii=False) + "\n")
    stage_logger.done("phase9_decision", decision_type=decision_type)
    end_phase(phase, "done", decision_type=decision_type)

    # Phase 11 final report
    phase = phase_block("phase11_final_report")
    stage_logger.start("phase11_final_report")
    short_train_headline = f"{sum(1 for row in train_summaries if row.get('completed_iters', 0) >= 500)} short-train runs reached at least 500 iter"
    h2_assignment_headline = "H2 diagnostics were recorded every 100 iter / final checkpoint to separate proxy movement from final occupancy movement"
    final_report_json = {
        "executive_summary": reinterpretation["headline"],
        "sw9_reinterpretation": reinterpretation,
        "h2_lambda_sweep_summary": {
            "headline": f"selected lambda_h2={lambda_choice['selected_lambda_h2']:.4f}; stronger candidate={lambda_choice['stronger_lambda_candidate']}",
            "rows": gradient_sweep_rows,
        },
        "config_manifest": config_manifest,
        "short_train_summary": {"headline": short_train_headline, "rows": train_summaries},
        "h2_assignment_summary": {"headline": h2_assignment_headline},
        "fixed_subset_eval_summary": {"headline": fixed_eval_headline, "rows": eval_flat_rows},
        "contributor_retest_summary": {"headline": contributor_headline},
        "reliability_retest_summary": {"headline": reliability_headline},
        "decision": decision_payload,
    }
    final_report_text = make_final_report(final_report_json)
    write_md(REPORTS_DIR / "stage_sw91_h2_contributor_assignment_training_report.md", final_report_text)
    write_json(REPORTS_DIR / "stage_sw91_h2_contributor_assignment_training_report.json", final_report_json)
    stage_logger.done("phase11_final_report")
    end_phase(phase, "done")

    time_manifest["end_time"] = now_iso()
    time_manifest["wall_clock_sec"] = time.time() - started_at
    run_time_manifest_update(time_manifest_path, time_manifest)


if __name__ == "__main__":
    main()
