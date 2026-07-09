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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
BASE_CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
BASE_CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"

SW101_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
SW91_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW10_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
SW91_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW10_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"

CORE_PERTURBATIONS = ["A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"]
CORE_HORIZONS = [0, 2, 4, 6]
CORE_SAMPLES = [0, 1, 2, 3, 4]
EMPTY_IDX = 17
TOPK_CLASS_CHECKS = [1, 3, 5]
DEFAULT_LR = 2e-5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw2 = load_module(
    "sw11_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw11_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw11_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw11_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw9 = load_module(
    "sw11_sw9",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/run_sparseworld_sw9_main.py",
)
sw101 = load_module(
    "sw11_sw101",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment/run_sw101_alignment_audit.py",
)

from mmcv.parallel import collate as collate_fn


@dataclass
class CheckpointSpec:
    name: str
    checkpoint_path: Path
    config_path: Path
    family: str


@dataclass
class SamplerMode:
    sampler_mode: str
    positive_mix: dict[str, float]
    negative_mix: dict[str, float]
    fallback_pool: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-11 risk-targeted H2 resampling")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", default="0,1,2,3,4")
    parser.add_argument("--perturbations", default="A0_clean,A10_drop_front_triplet,C4_motion_blur_9")
    parser.add_argument("--horizons", default="0,2,4,6")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--reserve-report-minutes", type=float, default=30.0)
    parser.add_argument("--smoke-iters", type=int, default=20)
    parser.add_argument("--workers-per-gpu", type=int, default=0)
    parser.add_argument("--samples-per-gpu", type=int, default=1)
    parser.add_argument("--max-positive-targets", type=int, default=512)
    parser.add_argument("--max-negative-targets", type=int, default=512)
    parser.add_argument("--resume-from", choices=["start", "phase7"], default="start")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "target_pools",
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
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    mode = "a" if exists else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
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


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


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


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


class StageLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "sw11_status.json"
        self.progress_path = LOGS_DIR / "sw11_progress.jsonl"
        self.state: dict[str, Any] = {"started_at": time.time(), "current_stage": None, "stages": {}}
        self.progress_path.write_text("", encoding="utf-8")
        self.flush()

    def flush(self) -> None:
        self.status_path.write_text(json.dumps(normalize_export(self.state), indent=2, ensure_ascii=False), encoding="utf-8")

    def event(self, stage: str, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalize_export({"ts": time.time(), "stage": stage, "event": event, **payload}), ensure_ascii=False) + "\n")
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


def phase_block(time_manifest: dict[str, Any], path: Path, name: str) -> dict[str, Any]:
    meta = {"phase_name": name, "start_time": now_iso(), "start_ts": time.time()}
    time_manifest["phases"].append(meta)
    write_json(path, time_manifest)
    return meta


def end_phase(time_manifest: dict[str, Any], path: Path, meta: dict[str, Any], status: str, **extra: Any) -> None:
    end_ts = time.time()
    meta.update(
        {
            "end_time": now_iso(),
            "end_ts": end_ts,
            "duration_sec": end_ts - float(meta.get("start_ts", end_ts)),
            "status": status,
            **extra,
        }
    )
    write_json(path, time_manifest)


def find_first(base_dir: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"missing file in {base_dir} for patterns {patterns}")


def discover_baseline_specs() -> list[CheckpointSpec]:
    route_selection = read_json(find_first(SW10_REPORTS, ["sw10_route_selection.json"]))
    selected_config = Path(route_selection.get("selected_config_path", REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_tiny_h1_lambda001.py"))
    p4_ckpt = find_first(SW91_ARTIFACTS / "checkpoints", ["P4_H2_tinyH1_500iter.pth"])
    routea_ckpt = find_first(SW10_ARTIFACTS / "checkpoints", ["routeA_best_h2_scaled_iter3000.pth"])
    return [
        CheckpointSpec("epoch_56_original", BASE_CHECKPOINT_PATH, BASE_CONFIG_PATH, "baseline"),
        CheckpointSpec("P4_H2_tinyH1_500iter", p4_ckpt, selected_config, "sw91_short_safe"),
        CheckpointSpec("routeA_best_h2_scaled_iter3000", routea_ckpt, selected_config, "sw10_scaled"),
    ]


def load_h2_cfg() -> Any:
    audit = read_json(SW101_REPORTS / "native_getocc_path_audit.json")
    cfg_payload = audit["h2_proxy_alignment_audit"]["h2_audit_config_source"]
    return sw101.H2AuditConfig(**cfg_payload)


def digest_sw101() -> tuple[dict[str, Any], str]:
    decision = read_json(find_first(SW101_REPORTS, ["*decision*.json"]))
    waterfall_rows = read_csv_rows(find_first(SW101_REPORTS, ["*waterfall*.csv"]))
    mismatch_rows = read_csv_rows(find_first(SW101_REPORTS, ["*mismatch*taxonomy*.csv"]))
    align_rows = read_csv_rows(find_first(SW101_REPORTS, ["*assignment*alignment*metrics*.csv"]))
    evolution_rows = read_csv_rows(find_first(SW101_REPORTS, ["checkpoint_alignment_evolution.csv"]))

    def avg_metric(checkpoint: str, scenario: str, metric: str) -> float | None:
        vals = [
            float(row[metric])
            for row in waterfall_rows
            if row.get("checkpoint_name") == checkpoint and row.get("scenario_name") == scenario and row.get(metric, "") != ""
        ]
        return float(np.mean(vals)) if vals else None

    digest = {
        "sw101_decision": decision,
        "a10_front_falsefree_has_h2_high_score_ratio": {
            "epoch_56_original": avg_metric("epoch_56_original", "front_sector_false_free", "has_h2_high_score_ratio"),
            "P4_H2_tinyH1_500iter": avg_metric("P4_H2_tinyH1_500iter", "front_sector_false_free", "has_h2_high_score_ratio"),
            "routeA_best_h2_scaled_iter3000": avg_metric("routeA_best_h2_scaled_iter3000", "front_sector_false_free", "has_h2_high_score_ratio"),
        },
        "a10_front_falsefree_native_exact_assignment_ratio": {
            "epoch_56_original": avg_metric("epoch_56_original", "front_sector_false_free", "has_native_exact_assignment_ratio"),
            "P4_H2_tinyH1_500iter": avg_metric("P4_H2_tinyH1_500iter", "front_sector_false_free", "has_native_exact_assignment_ratio"),
            "routeA_best_h2_scaled_iter3000": avg_metric("routeA_best_h2_scaled_iter3000", "front_sector_false_free", "has_native_exact_assignment_ratio"),
        },
        "a10_front_falsefree_final_contributor_ratio": {
            "epoch_56_original": avg_metric("epoch_56_original", "front_sector_false_free", "has_final_contributor_ratio"),
            "P4_H2_tinyH1_500iter": avg_metric("P4_H2_tinyH1_500iter", "front_sector_false_free", "has_final_contributor_ratio"),
            "routeA_best_h2_scaled_iter3000": avg_metric("routeA_best_h2_scaled_iter3000", "front_sector_false_free", "has_final_contributor_ratio"),
        },
        "alignment_row_count": len(align_rows),
        "mismatch_row_count": len(mismatch_rows),
        "evolution_rows": evolution_rows,
    }
    headline = (
        "SW-10.1 concluded A5_h2_target_not_focusing_failure: H2 was not stably focusing failure voxels, "
        "P4 briefly raised A10 front false-free H2 activation, but native exact assignment and final contributor stayed at zero, "
        "so SW-11 must test risk-targeted H2 resampling plus native contributor survival."
    )
    return digest, headline


def build_runtime(config_path: Path, checkpoint_path: Path, train: bool) -> tuple[Any, Any, Any]:
    cfg, dataset, model, _ = sw9.build_runtime(
        config_path,
        train=train,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0, "optimizer.lr": DEFAULT_LR},
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    if train:
        model.train()
    else:
        model.eval()
    return cfg, dataset, model


def unwrap_sample(sample: Any) -> dict[str, Any]:
    return sw2.unwrap(sample)


def apply_perturbation(batch: Any, perturbation_id: str) -> Any:
    if perturbation_id == "A0_clean":
        return batch
    spec = sw81.sw5_engine.build_catalog()[perturbation_id]
    batch_out, _ = sw81.sw5_engine.apply_perturbation_to_batch(batch, spec)
    return batch_out


def gt_in_topk(dense_scores: torch.Tensor, gt_h: torch.Tensor, k: int) -> torch.Tensor:
    topk = torch.topk(dense_scores, k=min(k, dense_scores.shape[-1]), dim=-1).indices
    return (topk == gt_h.unsqueeze(-1)).any(dim=-1)


def class_group_name(label: int) -> str:
    if label in sw2.CLASS_GROUPS["small_object"]:
        return "small_object"
    if label in sw2.CLASS_GROUPS["all_dynamic"]:
        return "dynamic"
    return "static"


def primary_pool_name(pool_hits: list[str]) -> str:
    priority = ["F6_semantic_survival_failure", "F5_native_contributor_missing", "F1_A10_front_false_free", "F2_small_object_false_free", "F3_new_visible_false_free", "F4_future_risk_false_free"]
    for item in priority:
        if item in pool_hits:
            return item
    return pool_hits[0] if pool_hits else "none"


def neighbor_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    return sw101.dilate3d(mask, radius)


def compute_pool_masks(
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    final_occ: torch.Tensor,
    debug: dict[str, Any],
    h2_proxy: dict[str, Any],
    perturbation_id: str,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    gt_occ = gt_h != EMPTY_IDX
    pred_free = final_occ == EMPTY_IDX
    native_exact = debug["semantic_active_mask"].cpu().bool()
    contributor_count = debug["contributor_count_dense"].cpu()
    geometric_mask = debug["geometric_mask"].cpu().bool()
    small_mask = torch.zeros_like(gt_occ)
    for cid in sw2.CLASS_GROUPS["small_object"]:
        small_mask |= gt_h == int(cid)
    new_visible = (gt0 == EMPTY_IDX) & gt_occ if horizon_s > 0 else torch.zeros_like(gt_occ)
    dense_after = debug["dense_occ_after_padding"].cpu().float()
    gt_top3 = gt_in_topk(dense_after, gt_h, 3)
    gt_top5 = gt_in_topk(dense_after, gt_h, 5)
    gt_score_top1 = dense_after.argmax(dim=-1)
    c4_fp_region = (gt_h == EMPTY_IDX) & (final_occ != EMPTY_IDX) if perturbation_id == "C4_motion_blur_9" else torch.zeros_like(gt_occ)

    f1 = (perturbation_id == "A10_drop_front_triplet") and horizon_s in {2, 4, 6}
    f1_mask = gt_occ & pred_free & sectors["front"] & (torch.ones_like(gt_occ, dtype=torch.bool) if f1 else torch.zeros_like(gt_occ, dtype=torch.bool))
    f2_mask = gt_occ & pred_free & small_mask & torch.ones_like(gt_occ, dtype=torch.bool)
    f3_mask = new_visible & pred_free & (torch.ones_like(gt_occ, dtype=torch.bool) if horizon_s in {2, 4, 6} else torch.zeros_like(gt_occ, dtype=torch.bool))
    f4_mask = gt_occ & pred_free & (torch.ones_like(gt_occ, dtype=torch.bool) if horizon_s in {4, 6} else torch.zeros_like(gt_occ, dtype=torch.bool)) & ((~geometric_mask) | (contributor_count == 0))
    f5_mask = gt_occ & (~native_exact) & (final_occ == EMPTY_IDX)
    f6_mask = native_exact & gt_occ & ((final_occ != gt_h) | (~gt_top3) | (~gt_top5) | (final_occ == EMPTY_IDX))
    n1_mask = c4_fp_region | ((gt_h == EMPTY_IDX) & (perturbation_id == "C4_motion_blur_9") & (h2_proxy["negative_sample_mask_dense"].cpu().bool()))
    return {
        "F1_A10_front_false_free": f1_mask,
        "F2_small_object_false_free": f2_mask,
        "F3_new_visible_false_free": f3_mask,
        "F4_future_risk_false_free": f4_mask,
        "F5_native_contributor_missing": f5_mask,
        "F6_semantic_survival_failure": f6_mask,
        "N1_hard_negative": n1_mask,
        "general_gt_occupied_fallback": gt_occ,
        "general_gt_free_random": gt_h == EMPTY_IDX,
    }


def coords_from_mask(mask: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(mask, as_tuple=False)
    return coords.long()


def pool_stats(mask: torch.Tensor, gt_h: torch.Tensor, gt0: torch.Tensor, final_occ: torch.Tensor, sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    coords = torch.nonzero(mask, as_tuple=False)
    class_counter: Counter[str] = Counter()
    sector_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()
    for coord in coords:
        x, y, z = [int(v) for v in coord.tolist()]
        label = int(gt_h[x, y, z].item())
        if label != EMPTY_IDX:
            class_counter[class_group_name(label)] += 1
        else:
            class_counter["free"] += 1
        if bool(sectors["front"][x, y, z].item()):
            sector_counter["front"] += 1
        elif bool(sectors["rear"][x, y, z].item()):
            sector_counter["rear"] += 1
        elif bool(sectors["side"][x, y, z].item()):
            sector_counter["side"] += 1
        else:
            sector_counter["other"] += 1
        gt_occ = label != EMPTY_IDX
        pred_label = int(final_occ[x, y, z].item())
        if gt_occ and pred_label == EMPTY_IDX:
            error_counter["false_free"] += 1
        elif (not gt_occ) and pred_label != EMPTY_IDX:
            error_counter["false_positive"] += 1
        elif gt_occ and pred_label == label:
            error_counter["tp"] += 1
        elif gt_occ:
            error_counter["wrong_class"] += 1
        else:
            error_counter["free"] += 1
    return {
        "voxel_count": int(coords.shape[0]),
        "sector_distribution": dict(sector_counter),
        "class_group_distribution": dict(class_counter),
        "error_type_distribution": dict(error_counter),
        "enough_samples": bool(coords.shape[0] > 0),
    }


def save_pool_npz(path: Path, pool_masks: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: coords_from_mask(mask).cpu().numpy().astype(np.int16) for name, mask in pool_masks.items()}
    np.savez_compressed(path, **payload)


def prepare_query_capture_inference(model: Any, holder: dict[str, Any]) -> Any:
    return sw81.attach_query_capture(model, holder)


def eval_case(
    model: Any,
    dataset: Any,
    sample_index: int,
    perturbation_id: str,
    horizons: list[int],
    h2_cfg: Any,
    sectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    holder: dict[str, Any] = {}
    original_forward = prepare_query_capture_inference(model, holder)
    try:
        raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = unwrap_sample(raw_sample)
        batch = apply_perturbation(batch, perturbation_id)
        sw81.reset_online_cache(model)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **sw2.move_to_cuda(batch))
        raw_result_cpu = sw2.to_cpu_artifact(result)
        query_cpu = sw2.to_cpu_artifact(holder)
        _, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
        head = sw4_inst.get_pts_bbox_head(model)
        per_horizon: dict[int, Any] = {}
        for horizon_s in horizons:
            pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
            _pred, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
            debug = dbg_list[0]
            gt_h = gt_temporal[horizon_s].long().cpu()
            gt0 = gt_temporal[0].long().cpu()
            final_occ = debug["occ_pred"].detach().cpu().long()
            debug_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in debug.items()}
            h2_proxy = sw101.build_h2_proxy_audit(debug_cpu, gt_h, None if horizon_s == 0 else gt0, horizon_s, h2_cfg)
            pool_masks = compute_pool_masks(gt_h, gt0, final_occ, debug_cpu, h2_proxy, perturbation_id, horizon_s, sectors)
            per_horizon[horizon_s] = {
                "gt_h": gt_h,
                "gt0": gt0,
                "final_occ": final_occ,
                "debug": debug_cpu,
                "h2_proxy": {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in h2_proxy.items()},
                "pool_masks": pool_masks,
                "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
            }
        return {
            "sample_index": sample_index,
            "perturbation_id": perturbation_id,
            "sample_info": sw2.sample_info(dataset, sample_index),
            "per_horizon": per_horizon,
        }
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]


def build_survival_rows_for_case(
    checkpoint_name: str,
    sample_index: int,
    perturbation_id: str,
    horizon_s: int,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    final_occ: torch.Tensor,
    debug: dict[str, Any],
    h2_proxy: dict[str, Any],
    pool_masks: dict[str, torch.Tensor],
    sectors: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    target_union = torch.zeros_like(gt_h, dtype=torch.bool)
    for pool_name in ["F1_A10_front_false_free", "F2_small_object_false_free", "F3_new_visible_false_free", "F4_future_risk_false_free", "F5_native_contributor_missing", "F6_semantic_survival_failure"]:
        target_union |= pool_masks[pool_name]
    rows: list[dict[str, Any]] = []
    coords = torch.nonzero(target_union, as_tuple=False)
    positive_coord_map = {tuple(int(v) for v in coord): idx for idx, coord in enumerate(h2_proxy["positive_coords_all"].tolist())}
    gate_flat = debug["gate_mask"].reshape(-1).cpu().bool()
    max_score_flat = debug["max_score"].reshape(-1).cpu().float()
    cls_scores_flat = debug["cls_scores_sigmoid"].reshape(-1, debug["cls_scores_sigmoid"].shape[-1]).cpu().float()
    voxel_index_flat = debug["pre_gate_voxel_index"].cpu().long()
    valid_range_flat = debug["valid_range_mask"].cpu().bool()
    dense_before = debug["dense_occ_before_padding"].cpu().float()
    dense_after = debug["dense_occ_after_padding"].cpu().float()
    native_exact = debug["semantic_active_mask"].cpu().bool()
    contributor_count = debug["contributor_count_dense"].cpu().long()
    for coord in coords:
        x, y, z = [int(v) for v in coord.tolist()]
        coord_key = (x, y, z)
        pool_hits = [name for name, mask in pool_masks.items() if name.startswith("F") and bool(mask[x, y, z].item())]
        if not pool_hits:
            continue
        gt_label = int(gt_h[x, y, z].item())
        class_name = class_group_name(gt_label)
        if bool(sectors["front"][x, y, z].item()):
            sector_name = "front"
        elif bool(sectors["rear"][x, y, z].item()):
            sector_name = "rear"
        elif bool(sectors["side"][x, y, z].item()):
            sector_name = "side"
        else:
            sector_name = "other"
        idx = positive_coord_map.get(coord_key)
        h2_score = float(h2_proxy["h2_score_dense"][x, y, z].item())
        h2_distance = float(h2_proxy["h2_distance_dense"][x, y, z].item()) if not torch.isnan(h2_proxy["h2_distance_dense"][x, y, z]) else -1.0
        if idx is not None:
            nearest_support_idx = int(h2_proxy["nearest_support_idx_all"][idx].item())
            nearest_support_distance = h2_distance
            support_conf = float(h2_proxy["support_conf"][nearest_support_idx].item())
            h2_rank = int((h2_proxy["h2_score_all"] > h2_proxy["h2_score_all"][idx]).sum().item()) + 1
        else:
            nearest_support_distance = -1.0
            support_conf = 0.0
            h2_rank = -1
        exact_support_mask = valid_range_flat & torch.all(voxel_index_flat == coord[None, :], dim=-1)
        neighbor_support_mask = valid_range_flat & (torch.max((voxel_index_flat - coord[None, :]).abs(), dim=-1).values <= 1) & (~exact_support_mask)
        gate_pass = bool((exact_support_mask & gate_flat).any().item())
        valid_pass = bool(exact_support_mask.any().item())
        neighbor_assign = bool(neighbor_support_mask.any().item())
        floor_match = bool(exact_support_mask.any().item())
        foreground_score = float(max_score_flat[exact_support_mask].max().item()) if bool(exact_support_mask.any().item()) else 0.0
        semantic_score = float(cls_scores_flat[exact_support_mask, gt_label].max().item()) if bool(exact_support_mask.any().item()) else 0.0
        dense_before_voxel = dense_before[x, y, z]
        dense_after_voxel = dense_after[x, y, z]
        top_order = torch.argsort(dense_after_voxel, descending=True)
        gt_rank = int(torch.nonzero(top_order == gt_label, as_tuple=False)[0].item()) + 1
        top1 = bool(top_order[0].item() == gt_label)
        top3 = bool((top_order[:3] == gt_label).any().item())
        top5 = bool((top_order[:5] == gt_label).any().item())
        gt_score = float(dense_after_voxel[gt_label].item())
        top1_score = float(dense_after_voxel[top_order[0]].item())
        second_score = float(dense_after_voxel[top_order[1]].item()) if dense_after_voxel.shape[0] > 1 else 0.0
        semantic_margin = gt_score - (second_score if top1 else top1_score)
        final_label = int(final_occ[x, y, z].item())
        if final_label == EMPTY_IDX:
            error_type = "false_free"
        elif final_label == gt_label:
            error_type = "tp"
        else:
            error_type = "wrong_class"
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "sample_index": sample_index,
                "perturbation_id": perturbation_id,
                "horizon_s": horizon_s,
                "is_selected_by_sampler": True,
                "pool_name": primary_pool_name(pool_hits),
                "pool_hits": json.dumps(pool_hits, ensure_ascii=False),
                "class_group": class_name,
                "sector_name": sector_name,
                "error_type": error_type,
                "gt_label": gt_label,
                "voxel_coord": json.dumps([x, y, z]),
                "h2_score": h2_score,
                "h2_distance": h2_distance,
                "h2_rank": h2_rank,
                "h2_high_score_thr03": h2_score > 0.3,
                "h2_high_score_thr05": h2_score > 0.5,
                "h2_high_score_thr07": h2_score > 0.7,
                "nearest_support_distance": nearest_support_distance,
                "support_confidence": support_conf,
                "native_exact_assignment_exists": bool(native_exact[x, y, z].item()),
                "native_contributor_count": int(contributor_count[x, y, z].item()),
                "native_voxel_index_match": floor_match,
                "floor_voxel_index": json.dumps(voxel_index_flat[exact_support_mask][:3].tolist() if bool(exact_support_mask.any().item()) else []),
                "neighbor_voxel_assignment": neighbor_assign,
                "foreground_score": foreground_score,
                "semantic_score": semantic_score,
                "native_score_gate_pass": gate_pass,
                "valid_mask_pass": valid_pass,
                "range_mask_pass": valid_pass,
                "survives_scatter_max": bool(dense_before_voxel.argmax().item() == gt_label and native_exact[x, y, z].item()),
                "post_padding_active": final_label != EMPTY_IDX,
                "final_contributor_exists": final_label != EMPTY_IDX,
                "contributor_class_logits": json.dumps(dense_after_voxel.tolist()),
                "gt_class_rank": gt_rank,
                "gt_class_top1": top1,
                "gt_class_top3": top3,
                "gt_class_top5": top5,
                "semantic_margin": semantic_margin,
                "final_semantic_occ_class": final_label,
                "final_error_type": error_type,
            }
        )
    return rows


def aggregate_survival_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_rows: list[dict[str, Any]] = []
    for scenario_name, scenario_filter in [
        ("A10_front_false_free", lambda r: r["perturbation_id"] == "A10_drop_front_triplet" and r["pool_name"] == "F1_A10_front_false_free"),
        ("small_object_false_free", lambda r: r["pool_name"] == "F2_small_object_false_free"),
        ("new_visible_false_free", lambda r: r["pool_name"] == "F3_new_visible_false_free"),
        ("future_false_free", lambda r: r["pool_name"] == "F4_future_risk_false_free"),
        ("native_missing", lambda r: r["pool_name"] == "F5_native_contributor_missing"),
        ("semantic_survival_failure", lambda r: r["pool_name"] == "F6_semantic_survival_failure"),
    ]:
        buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if scenario_filter(row):
                buckets[(row["checkpoint_name"], row["perturbation_id"], int(row["horizon_s"]))].append(row)
        for (checkpoint_name, perturbation_id, horizon_s), items in buckets.items():
            count = len(items)
            if count == 0:
                continue
            pred_gt_proxy = safe_div(sum(1 for item in items if item["final_contributor_exists"]), count)
            scenario_rows.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "perturbation_id": perturbation_id,
                    "horizon_s": horizon_s,
                    "scenario_name": scenario_name,
                    "voxel_count": count,
                    "has_h2_high_score_ratio": safe_div(sum(1 for item in items if item["h2_high_score_thr05"]), count),
                    "H2_native_IoU": safe_div(sum(1 for item in items if item["h2_high_score_thr05"] and item["native_exact_assignment_exists"]), sum(1 for item in items if item["h2_high_score_thr05"] or item["native_exact_assignment_exists"])),
                    "native_contributor_but_low_H2_score_ratio": safe_div(sum(1 for item in items if item["native_exact_assignment_exists"] and not item["h2_high_score_thr05"]), sum(1 for item in items if item["native_exact_assignment_exists"])),
                    "has_native_exact_assignment_ratio": safe_div(sum(1 for item in items if item["native_exact_assignment_exists"]), count),
                    "native_score_gate_pass_ratio": safe_div(sum(1 for item in items if item["native_score_gate_pass"]), count),
                    "has_final_contributor_ratio": safe_div(sum(1 for item in items if item["final_contributor_exists"]), count),
                    "GT_class_top3_survival_ratio": safe_div(sum(1 for item in items if item["gt_class_top3"]), count),
                    "GT_class_top5_survival_ratio": safe_div(sum(1 for item in items if item["gt_class_top5"]), count),
                    "final_semantic_occ_TP_recovery_ratio": safe_div(sum(1 for item in items if item["final_error_type"] == "tp"), count),
                    "pred_gt_density_proxy": pred_gt_proxy,
                    "neighbor_leakage_ratio": safe_div(sum(1 for item in items if item["neighbor_voxel_assignment"] and not item["native_exact_assignment_exists"]), count),
                }
            )
    return scenario_rows


def plot_target_pool_distribution(manifest_rows: list[dict[str, Any]], out_path: Path) -> None:
    counts: Counter[str] = Counter()
    for row in manifest_rows:
        counts[row["pool_name"]] += int(row["voxel_count"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = list(counts.keys())
    vals = [counts[label] for label in labels]
    ax.bar(labels, vals, color="#1F618D")
    ax.set_title("SW-11 subset diagnostic target pool distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gradient_norms(rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = [row["experiment_id"] for row in rows]
    total = [float(row["grad_norm_total"]) for row in rows]
    decoder = [float(row["grad_norm_decoder"]) for row in rows]
    forecast = [float(row.get("grad_norm_forecast_path") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, total, width=w, label="total")
    ax.bar(x, decoder, width=w, label="decoder")
    ax.bar(x + w, forecast, width=w, label="forecast")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_title("SW-11 subset diagnostic gradient norms")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_survival_before_after(rows: list[dict[str, Any]], out_path: Path, scenario_name: str) -> None:
    subset = [row for row in rows if row["scenario_name"] == scenario_name]
    if not subset:
        subset = rows
    if not subset:
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        grouped[row["checkpoint_name"]].append(row)
    checkpoints = list(grouped.keys())
    metrics = [
        "has_h2_high_score_ratio",
        "has_native_exact_assignment_ratio",
        "has_final_contributor_ratio",
        "GT_class_top3_survival_ratio",
        "final_semantic_occ_TP_recovery_ratio",
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(checkpoints))
    w = 0.14
    for idx, metric in enumerate(metrics):
        vals = [float(np.mean([float(item[metric]) for item in grouped[checkpoint]])) for checkpoint in checkpoints]
        ax.bar(x + (idx - 2) * w, vals, width=w, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=25)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"SW-11 subset diagnostic / 20-iter smoke survival chain: {scenario_name}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mismatch_before_after(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = [row["mismatch_type"] for row in summary_rows]
    before = [float(row.get("baseline_count", 0.0)) for row in summary_rows]
    after = [float(row.get("smoke_count", 0.0)) for row in summary_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.2, before, width=0.4, label="baseline")
    ax.bar(x + 0.2, after, width=0.4, label="smoke")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_title("SW-11 subset diagnostic / 20-iter smoke mismatch taxonomy before vs after")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_decision_flow(decision_type: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    boxes = [
        (0.04, 0.38, 0.18, 0.22, "risk-targeted\nH2 resampling"),
        (0.30, 0.38, 0.18, 0.22, "H2 high-score\ncoverage check"),
        (0.56, 0.38, 0.18, 0.22, "native survival\nchain check"),
        (0.82, 0.38, 0.14, 0.22, decision_type),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, ec="#154360", fc="#EAF2F8", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][0], 0.49), xytext=(boxes[i][0] + boxes[i][2], 0.49), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#154360"})
    ax.set_title("SW-11 subset diagnostic / 20-iter smoke decision flow")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def original_loss_from_log_vars(log_vars: dict[str, float]) -> float:
    return sw9.original_loss_from_log_vars(log_vars)


def build_sampler_modes() -> dict[str, SamplerMode]:
    return {
        "S0_original_h2_sampling": SamplerMode(
            sampler_mode="S0_original_h2_sampling",
            positive_mix={"general_gt_occupied_fallback": 1.0},
            negative_mix={"general_gt_free_random": 1.0},
            fallback_pool="general_gt_occupied_fallback",
        ),
        "S1_risk_targeted_h2_sampling": SamplerMode(
            sampler_mode="S1_risk_targeted_h2_sampling",
            positive_mix={
                "F1_A10_front_false_free": 0.35,
                "F2_small_object_false_free": 0.20,
                "F3_new_visible_false_free": 0.20,
                "F4_future_risk_false_free": 0.15,
                "general_gt_occupied_fallback": 0.10,
            },
            negative_mix={"N1_hard_negative": 0.50, "general_gt_free_random": 0.50},
            fallback_pool="general_gt_occupied_fallback",
        ),
        "S2_native_missing_h2_sampling": SamplerMode(
            sampler_mode="S2_native_missing_h2_sampling",
            positive_mix={
                "F5_native_contributor_missing": 0.50,
                "F1_A10_front_false_free": 0.20,
                "F2_small_object_false_free": 0.15,
                "F3_new_visible_false_free": 0.15,
            },
            negative_mix={"N1_hard_negative": 0.50, "general_gt_free_random": 0.50},
            fallback_pool="general_gt_occupied_fallback",
        ),
        "S3_survival_aware_h2_sampling": SamplerMode(
            sampler_mode="S3_survival_aware_h2_sampling",
            positive_mix={
                "F6_semantic_survival_failure": 0.30,
                "F1_A10_front_false_free": 0.30,
                "F2_small_object_false_free": 0.20,
                "F3_new_visible_false_free": 0.20,
            },
            negative_mix={"N1_hard_negative": 0.50, "general_gt_free_random": 0.50},
            fallback_pool="general_gt_occupied_fallback",
        ),
        "G4_risk_targeted_with_hard_negative": SamplerMode(
            sampler_mode="G4_risk_targeted_with_hard_negative",
            positive_mix={
                "F1_A10_front_false_free": 0.35,
                "F2_small_object_false_free": 0.20,
                "F3_new_visible_false_free": 0.20,
                "F4_future_risk_false_free": 0.15,
                "general_gt_occupied_fallback": 0.10,
            },
            negative_mix={"N1_hard_negative": 0.50, "general_gt_free_random": 0.50},
            fallback_pool="general_gt_occupied_fallback",
        ),
    }


class RiskTargetedH2Loss:
    def __init__(self, h2_cfg: Any, seed: int, max_positive_targets: int, max_negative_targets: int) -> None:
        self.h2_cfg = h2_cfg
        self.seed = int(seed)
        self.max_positive_targets = int(max_positive_targets)
        self.max_negative_targets = int(max_negative_targets)

    def centers_from_coords(self, coords: torch.Tensor, device: torch.device) -> torch.Tensor:
        if coords.numel() == 0:
            return torch.zeros((0, 3), device=device, dtype=torch.float32)
        pc = torch.tensor(self.h2_cfg.pc_range[:3], device=device, dtype=torch.float32)
        voxel = torch.tensor(self.h2_cfg.voxel_size, device=device, dtype=torch.float32)
        return (coords.float().to(device) + 0.5) * voxel + pc

    def support_from_tensors(self, refine_pts: torch.Tensor, cls_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from mmdet3d.models.sparsedetectors.bbox.utils import decode_points

        points = decode_points(refine_pts.reshape(1, -1, 3), torch.as_tensor(self.h2_cfg.pc_range, device=refine_pts.device, dtype=torch.float32)).squeeze(0)
        conf = cls_logits.sigmoid().amax(dim=-1).reshape(-1)
        if points.shape[0] > self.h2_cfg.max_support_points:
            topk = torch.topk(conf, k=self.h2_cfg.max_support_points, largest=True).indices
            points = points[topk]
            conf = conf[topk]
        return points, conf

    def _select_from_mask(self, mask: torch.Tensor, desired_count: int, seed_offset: int) -> torch.Tensor:
        coords = coords_from_mask(mask)
        if coords.shape[0] <= desired_count:
            return coords
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + seed_offset)
        perm = torch.randperm(coords.shape[0], generator=gen)[:desired_count]
        return coords[perm]

    def _weight_for_positive(self, pool_name: str, gt_label: int, perturbation_id: str, horizon_s: int) -> float:
        weight = 1.0
        if pool_name in {"F1_A10_front_false_free", "F2_small_object_false_free", "F3_new_visible_false_free"}:
            weight *= 2.0
        if pool_name == "F4_future_risk_false_free":
            weight *= 1.5
        if gt_label in sw2.CLASS_GROUPS["small_object"]:
            weight *= 2.0
        if horizon_s in {4, 6}:
            weight *= 1.5
        if perturbation_id == "A0_clean":
            weight *= 0.5
        return weight

    def sample_case(
        self,
        sampler_mode: SamplerMode,
        pool_masks: dict[str, torch.Tensor],
        gt_h: torch.Tensor,
        perturbation_id: str,
        horizon_s: int,
        seed_offset: int,
    ) -> dict[str, Any]:
        enough_f6 = int(pool_masks["F6_semantic_survival_failure"].sum().item()) > 0
        effective_mode = sampler_mode
        if sampler_mode.sampler_mode == "S3_survival_aware_h2_sampling" and not enough_f6:
            effective_mode = build_sampler_modes()["S1_risk_targeted_h2_sampling"]
        positive_chunks: list[torch.Tensor] = []
        positive_meta: list[tuple[str, int]] = []
        fallback_count = 0
        distribution: dict[str, int] = {}
        for idx, (pool_name, ratio) in enumerate(effective_mode.positive_mix.items()):
            desired = max(1, int(round(self.max_positive_targets * ratio))) if ratio > 0 else 0
            if desired <= 0:
                continue
            source_mask = pool_masks.get(pool_name, torch.zeros_like(gt_h, dtype=torch.bool))
            picked = self._select_from_mask(source_mask, desired, seed_offset + idx)
            if picked.shape[0] < desired and pool_name != effective_mode.fallback_pool:
                fallback_needed = desired - picked.shape[0]
                fallback = self._select_from_mask(pool_masks[effective_mode.fallback_pool], fallback_needed, seed_offset + 100 + idx)
                if fallback.shape[0] > 0:
                    picked = torch.cat([picked, fallback], dim=0)
                    fallback_count += int(fallback.shape[0])
            if picked.shape[0] == 0:
                continue
            positive_chunks.append(picked)
            distribution[pool_name] = int(picked.shape[0])
            positive_meta.extend([(pool_name, int(i)) for i in range(picked.shape[0])])
        if positive_chunks:
            positive_coords = torch.cat(positive_chunks, dim=0)
        else:
            positive_coords = self._select_from_mask(pool_masks["general_gt_occupied_fallback"], min(64, self.max_positive_targets), seed_offset + 500)
            positive_meta = [("general_gt_occupied_fallback", int(i)) for i in range(positive_coords.shape[0])]
            distribution["general_gt_occupied_fallback"] = int(positive_coords.shape[0])
            fallback_count = int(positive_coords.shape[0])
        if positive_coords.shape[0] > self.max_positive_targets:
            positive_coords = positive_coords[: self.max_positive_targets]
            positive_meta = positive_meta[: self.max_positive_targets]

        negative_chunks: list[torch.Tensor] = []
        negative_distribution: dict[str, int] = {}
        for idx, (pool_name, ratio) in enumerate(effective_mode.negative_mix.items()):
            desired = max(1, int(round(self.max_negative_targets * ratio))) if ratio > 0 else 0
            if desired <= 0:
                continue
            picked = self._select_from_mask(pool_masks.get(pool_name, torch.zeros_like(gt_h, dtype=torch.bool)), desired, seed_offset + 700 + idx)
            if picked.shape[0] == 0 and pool_name != "general_gt_free_random":
                fallback = self._select_from_mask(pool_masks["general_gt_free_random"], desired, seed_offset + 760 + idx)
                picked = fallback
            if picked.shape[0] > 0:
                negative_chunks.append(picked)
                negative_distribution[pool_name] = int(picked.shape[0])
        negative_coords = torch.cat(negative_chunks, dim=0) if negative_chunks else self._select_from_mask(pool_masks["general_gt_free_random"], min(64, self.max_negative_targets), seed_offset + 900)
        if negative_coords.shape[0] > self.max_negative_targets:
            negative_coords = negative_coords[: self.max_negative_targets]

        positive_weights = []
        pool_names = []
        for coord, (pool_name, _dummy) in zip(positive_coords.tolist(), positive_meta):
            label = int(gt_h[coord[0], coord[1], coord[2]].item())
            positive_weights.append(self._weight_for_positive(pool_name, label, perturbation_id, horizon_s))
            pool_names.append(pool_name)
        return {
            "effective_mode": effective_mode.sampler_mode,
            "positive_coords": positive_coords,
            "negative_coords": negative_coords,
            "positive_weights": torch.tensor(positive_weights, dtype=torch.float32),
            "positive_pool_names": pool_names,
            "pool_distribution": distribution,
            "negative_distribution": negative_distribution,
            "fallback_ratio": safe_div(fallback_count, max(1, positive_coords.shape[0])),
            "enough_f6": enough_f6,
        }

    def compute_aux_loss_for_case(
        self,
        sampler_mode: SamplerMode,
        refine_pts: torch.Tensor,
        cls_logits: torch.Tensor,
        gt_h: torch.Tensor,
        pool_masks: dict[str, torch.Tensor],
        perturbation_id: str,
        horizon_s: int,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sampled = self.sample_case(sampler_mode, pool_masks, gt_h, perturbation_id, horizon_s, seed_offset)
        support_points, support_conf = self.support_from_tensors(refine_pts, cls_logits)
        positive_centers = self.centers_from_coords(sampled["positive_coords"], support_points.device)
        negative_centers = self.centers_from_coords(sampled["negative_coords"], support_points.device)
        pos_weights = sampled["positive_weights"].to(support_points.device)
        temperature_m = self.h2_cfg.h2_temperature_voxel * self.h2_cfg.voxel_size[0]
        assign_thr_m = self.h2_cfg.h2_r_assign_voxel * self.h2_cfg.voxel_size[0]
        if positive_centers.numel():
            assign_dist = torch.cdist(positive_centers, support_points, p=2.0)
            soft_assign = -temperature_m * torch.logsumexp(-assign_dist / max(temperature_m, 1e-6), dim=-1)
            loss_assign = (pos_weights * torch.relu(soft_assign - assign_thr_m).pow(2)).mean()
        else:
            soft_assign = torch.zeros((0,), device=support_points.device)
            loss_assign = support_points.sum() * 0.0
        if negative_centers.numel():
            neg_dist = torch.cdist(negative_centers, support_points, p=2.0)
            neg_soft = -temperature_m * torch.logsumexp(-neg_dist / max(temperature_m, 1e-6), dim=-1)
            neg_score = torch.sigmoid((assign_thr_m - neg_soft) / max(temperature_m, 1e-6))
            loss_leak = 0.5 * neg_score.pow(2).mean()
        else:
            neg_soft = torch.zeros((0,), device=support_points.device)
            neg_score = torch.zeros((0,), device=support_points.device)
            loss_leak = support_points.sum() * 0.0
        metrics = {
            "positive_target_count": int(sampled["positive_coords"].shape[0]),
            "negative_target_count": int(sampled["negative_coords"].shape[0]),
            "pool_distribution": sampled["pool_distribution"],
            "negative_distribution": sampled["negative_distribution"],
            "fallback_ratio": float(sampled["fallback_ratio"]),
            "target_pool_not_empty": bool(sampled["positive_coords"].shape[0] > 0),
            "mean_assign_distance": float(soft_assign.detach().mean().item()) if soft_assign.numel() else 0.0,
            "mean_negative_score": float(neg_score.detach().mean().item()) if neg_score.numel() else 0.0,
            "effective_mode": sampled["effective_mode"],
        }
        return loss_assign, loss_leak, metrics


def build_train_batch(dataset: Any, sample_index: int, perturbation_id: str) -> Any:
    sample = dataset[sample_index]
    batch = collate_fn([sample], samples_per_gpu=1)
    if "img" in batch and not isinstance(batch["img"], list):
        batch["img"] = [batch["img"]]
    if "img_metas" in batch and not isinstance(batch["img_metas"], list):
        batch["img_metas"] = [batch["img_metas"]]
    return apply_perturbation(batch, perturbation_id)


def attach_forward_capture(model: Any, holder: dict[str, Any], retain_grads: bool = False):
    return sw9.attach_forward_capture(model, holder, retain_grads=retain_grads)


def extract_train_tensors(
    holder: dict[str, Any],
    model_inputs: dict[str, Any],
    horizons: list[int],
    h2_cfg: Any,
    perturbation_id: str,
    sectors: dict[str, torch.Tensor],
) -> dict[int, Any]:
    fb = holder["forward_backbone_outputs"]
    outs = fb["outs"]
    model_ref = holder["model_ref"]
    ind = model_ref.pts_bbox_head.ind_stamps_all
    current_refine = outs["all_refine_pts"][-1]
    current_cls = outs["all_cls_scores"][-1]
    if current_refine.shape[1] == ind.shape[0]:
        current_refine = current_refine[:, ind == 0]
        current_cls = current_cls[:, ind == 0]
    temporal_semantics = model_inputs["temporal_semantics"]
    head = sw4_inst.get_pts_bbox_head(model_ref)
    case_by_h: dict[int, Any] = {}
    for horizon_s in horizons:
        if horizon_s == 0:
            refine_tensor = current_refine
            cls_tensor = current_cls
            gt_h = model_inputs["voxel_semantics"]
            gt0 = model_inputs["voxel_semantics"]
        else:
            future_idx = horizon_s - 1
            if future_idx >= len(fb["forecast_points_list"]) or future_idx not in temporal_semantics:
                continue
            refine_tensor = fb["forecast_points_list"][future_idx]
            cls_tensor = fb["forecast_semantics_list"][future_idx]
            gt_h = temporal_semantics[future_idx]["voxel_semantics"]
            gt0 = model_inputs["voxel_semantics"]
        pred_dict = {"cls_scores": cls_tensor.detach(), "refine_pts": refine_tensor.detach()}
        _pred, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
        debug = dbg_list[0]
        gt_h_cpu = gt_h[0].detach().cpu().long()
        gt0_cpu = gt0[0].detach().cpu().long()
        final_occ = debug["occ_pred"].detach().cpu().long()
        debug_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in debug.items()}
        h2_proxy = sw101.build_h2_proxy_audit(debug_cpu, gt_h_cpu, None if horizon_s == 0 else gt0_cpu, horizon_s, h2_cfg)
        pool_masks = compute_pool_masks(gt_h_cpu, gt0_cpu, final_occ, debug_cpu, h2_proxy, perturbation_id, horizon_s, sectors)
        case_by_h[horizon_s] = {
            "refine_tensor": refine_tensor[0],
            "cls_tensor": cls_tensor[0],
            "gt_h": gt_h_cpu,
            "gt0": gt0_cpu,
            "debug": debug_cpu,
            "final_occ": final_occ,
            "h2_proxy": {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in h2_proxy.items()},
            "pool_masks": pool_masks,
        }
    return case_by_h


def compute_external_sampler_losses(
    sampler: RiskTargetedH2Loss,
    sampler_mode: SamplerMode,
    holder: dict[str, Any],
    model_inputs: dict[str, Any],
    perturbation_id: str,
    horizons: list[int],
    h2_cfg: Any,
    sectors: dict[str, torch.Tensor],
    iter_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    case_by_h = extract_train_tensors(holder, model_inputs, horizons, h2_cfg, perturbation_id, sectors)
    assign_losses: list[torch.Tensor] = []
    leak_losses: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for idx, horizon_s in enumerate(horizons):
        case = case_by_h.get(horizon_s)
        if case is None:
            continue
        loss_assign, loss_leak, metrics = sampler.compute_aux_loss_for_case(
            sampler_mode=sampler_mode,
            refine_pts=case["refine_tensor"],
            cls_logits=case["cls_tensor"],
            gt_h=case["gt_h"],
            pool_masks=case["pool_masks"],
            perturbation_id=perturbation_id,
            horizon_s=horizon_s,
            seed_offset=iter_seed * 100 + idx,
        )
        assign_losses.append(loss_assign)
        leak_losses.append(loss_leak)
        metrics.update({"horizon_s": horizon_s})
        records.append(metrics)
    if assign_losses:
        assign_total = torch.stack(assign_losses).mean()
        leak_total = torch.stack(leak_losses).mean()
    else:
        device = next(iter(holder["forward_backbone_outputs"]["outs"]["all_cls_scores"][-1].flatten())).device
        zero = torch.zeros((), device=device)
        assign_total = zero
        leak_total = zero
    return assign_total, leak_total, records


def total_grad_norm(model: Any) -> float:
    return sw9.total_grad_norm(model)


def decoder_grad_norm(model: Any) -> float:
    return sw9.decoder_grad_norm(model)


def forecast_grad_norm(holder: dict[str, Any]) -> float | None:
    grad_refs = holder.get("grad_refs", {})
    return sw9.tensor_grad_norm(grad_refs.get("forecast_points_last"))


def gradient_check(
    experiment_id: str,
    sampler_mode: SamplerMode,
    h2_cfg: Any,
    sectors: dict[str, torch.Tensor],
    sampler: RiskTargetedH2Loss,
) -> dict[str, Any]:
    cfg, dataset, model = build_runtime(BASE_CONFIG_PATH, BASE_CHECKPOINT_PATH, train=True)
    batch = build_train_batch(dataset, 0, "A10_drop_front_triplet")
    holder: dict[str, Any] = {"model_ref": model}
    original_forward = attach_forward_capture(model, holder, retain_grads=True)
    try:
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model_inputs = sw81.move_train_batch_to_cuda(batch)
        losses = model(return_loss=True, **model_inputs)
        original_total, log_vars = sw81.parse_losses(losses)
        ext_assign, ext_leak, records = compute_external_sampler_losses(
            sampler,
            sampler_mode,
            holder,
            model_inputs,
            perturbation_id="A10_drop_front_triplet",
            horizons=CORE_HORIZONS,
            h2_cfg=h2_cfg,
            sectors=sectors,
            iter_seed=0,
        )
        total_loss = original_total + (0.001 * ext_assign) + (0.0005 * ext_leak)
        has_nan = not bool(torch.isfinite(total_loss).item())
        total_loss.backward()
        torch.cuda.synchronize()
        backward_time_sec = time.perf_counter() - started
        grad_finite = True
        has_inf = False
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
            has_inf = has_inf or bool(torch.isinf(param.grad).any().item())
        target_pool_not_empty = any(record["target_pool_not_empty"] for record in records) if records else False
        fallback_ratio = float(np.mean([record["fallback_ratio"] for record in records])) if records else 1.0
        scaled_aux_loss = float((0.001 * ext_assign + 0.0005 * ext_leak).detach().cpu().item())
        row = {
            "experiment_id": experiment_id,
            "sampler_mode": sampler_mode.sampler_mode,
            "total_loss": float(total_loss.detach().cpu().item()),
            "original_loss": original_loss_from_log_vars(log_vars),
            "h2_assign_loss": float(ext_assign.detach().cpu().item()),
            "h2_leak_loss": float(ext_leak.detach().cpu().item()),
            "positive_target_count": int(sum(record["positive_target_count"] for record in records)),
            "negative_target_count": int(sum(record["negative_target_count"] for record in records)),
            "pool_distribution": json.dumps({k: int(sum(record["pool_distribution"].get(k, 0) for record in records)) for k in {name for record in records for name in record["pool_distribution"].keys()}}, ensure_ascii=False),
            "fallback_ratio": fallback_ratio,
            "scaled_aux_loss": scaled_aux_loss,
            "grad_finite": grad_finite,
            "grad_norm_total": total_grad_norm(model),
            "grad_norm_decoder": decoder_grad_norm(model),
            "grad_norm_forecast_path": forecast_grad_norm(holder),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "backward_time_sec": backward_time_sec,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "target_pool_not_empty": target_pool_not_empty,
            "eligible_for_smoke": bool(grad_finite and (not has_nan) and (not has_inf) and fallback_ratio <= 0.5 and target_pool_not_empty and scaled_aux_loss <= 0.25 * max(original_loss_from_log_vars(log_vars), 1e-6)),
        }
        return row
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]
        del model, dataset, cfg
        safe_cuda_cleanup()


def smoke_schedule() -> list[tuple[str, int]]:
    return [
        ("A10_drop_front_triplet", 0),
        ("A10_drop_front_triplet", 1),
        ("A10_drop_front_triplet", 2),
        ("C4_motion_blur_9", 0),
        ("C4_motion_blur_9", 1),
        ("A10_drop_front_triplet", 3),
        ("A10_drop_front_triplet", 4),
        ("C4_motion_blur_9", 2),
        ("C4_motion_blur_9", 3),
        ("A0_clean", 0),
    ]


def smoke_train(
    experiment_id: str,
    sampler_mode: SamplerMode,
    h2_cfg: Any,
    sectors: dict[str, torch.Tensor],
    sampler: RiskTargetedH2Loss,
    smoke_iters: int,
) -> tuple[Path | None, list[dict[str, Any]], dict[str, Any]]:
    cfg, dataset, model = build_runtime(BASE_CONFIG_PATH, BASE_CHECKPOINT_PATH, train=True)
    from mmcv.runner import build_optimizer

    optimizer = build_optimizer(model, cfg.optimizer)
    schedule = smoke_schedule()
    iter_rows: list[dict[str, Any]] = []
    stop_reason = "completed"
    ckpt_path: Path | None = None
    try:
        for iter_idx in range(1, smoke_iters + 1):
            perturbation_id, sample_index = schedule[(iter_idx - 1) % len(schedule)]
            batch = build_train_batch(dataset, sample_index, perturbation_id)
            holder: dict[str, Any] = {"model_ref": model}
            original_forward = attach_forward_capture(model, holder, retain_grads=True)
            try:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                model_inputs = sw81.move_train_batch_to_cuda(batch)
                losses = model(return_loss=True, **model_inputs)
                original_total, log_vars = sw81.parse_losses(losses)
                ext_assign, ext_leak, records = compute_external_sampler_losses(
                    sampler,
                    sampler_mode,
                    holder,
                    model_inputs,
                    perturbation_id=perturbation_id,
                    horizons=CORE_HORIZONS,
                    h2_cfg=h2_cfg,
                    sectors=sectors,
                    iter_seed=iter_idx,
                )
                total_loss = original_total + (0.001 * ext_assign) + (0.0005 * ext_leak)
                if not bool(torch.isfinite(total_loss).item()):
                    stop_reason = "non_finite_total_loss"
                    break
                total_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).detach().cpu().item())
                optimizer.step()
                torch.cuda.synchronize()
                iter_time_sec = time.perf_counter() - started
                pool_distribution = {k: int(sum(record["pool_distribution"].get(k, 0) for record in records)) for k in {name for record in records for name in record["pool_distribution"].keys()}}
                fallback_ratio = float(np.mean([record["fallback_ratio"] for record in records])) if records else 1.0
                iter_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "iter": iter_idx,
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "loss_total": float(total_loss.detach().cpu().item()),
                        "original_loss": original_loss_from_log_vars(log_vars),
                        "h2_assign_loss": float(ext_assign.detach().cpu().item()),
                        "h2_leak_loss": float(ext_leak.detach().cpu().item()),
                        "pool_distribution": json.dumps(pool_distribution, ensure_ascii=False),
                        "fallback_ratio": fallback_ratio,
                        "grad_norm_total": grad_norm,
                        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                        "iter_time_sec": iter_time_sec,
                    }
                )
            finally:
                model.forward_backbone = original_forward  # type: ignore[assignment]
            if iter_idx % 5 == 0:
                safe_cuda_cleanup()
        ckpt_path = ARTIFACTS_DIR / "checkpoints" / f"{experiment_id}.pth"
        torch.save({"state_dict": model.state_dict(), "meta": {"iter": len(iter_rows), "experiment_id": experiment_id}}, ckpt_path)
        summary = {
            "experiment_id": experiment_id,
            "sampler_mode": sampler_mode.sampler_mode,
            "completed_iters": len(iter_rows),
            "stop_reason": stop_reason,
            "checkpoint_path": None if ckpt_path is None else str(ckpt_path),
            "mean_loss_total": float(np.mean([row["loss_total"] for row in iter_rows])) if iter_rows else None,
            "mean_fallback_ratio": float(np.mean([row["fallback_ratio"] for row in iter_rows])) if iter_rows else None,
            "peak_gpu_memory_mb": float(max(row["peak_gpu_memory_mb"] for row in iter_rows)) if iter_rows else None,
        }
        return ckpt_path, iter_rows, summary
    finally:
        del optimizer, model, dataset, cfg
        safe_cuda_cleanup()


def baseline_or_smoke_specs(smoke_checkpoint_rows: list[dict[str, Any]]) -> list[CheckpointSpec]:
    specs = discover_baseline_specs()
    for row in smoke_checkpoint_rows:
        if row.get("checkpoint_path"):
            specs.append(CheckpointSpec(row["experiment_id"], Path(row["checkpoint_path"]), BASE_CONFIG_PATH, "sw11_smoke"))
    return specs


def summarize_taxonomy_rows(rows: list[dict[str, Any]], baseline_names: set[str], smoke_names: set[str]) -> list[dict[str, Any]]:
    counter_baseline: Counter[str] = Counter()
    counter_smoke: Counter[str] = Counter()
    for row in rows:
        if row["checkpoint_name"] in baseline_names:
            counter_baseline[row["mismatch_type"]] += 1
        if row["checkpoint_name"] in smoke_names:
            counter_smoke[row["mismatch_type"]] += 1
    all_keys = sorted(set(counter_baseline.keys()) | set(counter_smoke.keys()))
    return [{"mismatch_type": key, "baseline_count": counter_baseline[key], "smoke_count": counter_smoke[key]} for key in all_keys]


def summarize_alignment_survival(rows: list[dict[str, Any]]) -> str:
    focus = [
        row
        for row in rows
        if row["scenario_name"] == "A10_front_false_free"
        and row["perturbation_id"] == "A10_drop_front_triplet"
        and int(row["horizon_s"]) == 6
    ]
    if not focus:
        return "subset retest summary unavailable because no A10 front h6 rows were produced"
    lookup = {row["checkpoint_name"]: row for row in focus}
    base = lookup.get("epoch_56_original")
    p1 = lookup.get("P1_risk_targeted_S1_20iter")
    p2 = lookup.get("P2_native_missing_S2_20iter")
    if not base:
        return "subset retest summary unavailable because epoch_56 baseline row was missing"
    chunks = [
        f"epoch_56 A10/front/h6 has_h2_high_score_ratio={float(base['has_h2_high_score_ratio']):.4f}",
    ]
    if p1:
        chunks.append(f"P1 risk-targeted={float(p1['has_h2_high_score_ratio']):.4f} / native_exact={float(p1['has_native_exact_assignment_ratio']):.4f} / final_contributor={float(p1['has_final_contributor_ratio']):.4f}")
    if p2:
        chunks.append(f"P2 native-missing={float(p2['has_h2_high_score_ratio']):.4f} / native_exact={float(p2['has_native_exact_assignment_ratio']):.4f} / final_contributor={float(p2['has_final_contributor_ratio']):.4f}")
    return "; ".join(chunks) + "."


def reconstruct_smoke_summaries() -> list[dict[str, Any]]:
    metrics_path = REPORTS_DIR / "sw11_20iter_smoke_train_metrics.csv"
    rows = read_csv_rows(metrics_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("experiment_id"):
            grouped[row["experiment_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    for experiment_id, items in grouped.items():
        ckpt_path = ARTIFACTS_DIR / "checkpoints" / f"{experiment_id}.pth"
        summaries.append(
            {
                "experiment_id": experiment_id,
                "sampler_mode": items[0].get("sampler_mode", ""),
                "completed_iters": len(items),
                "stop_reason": "completed" if ckpt_path.exists() else "missing_checkpoint",
                "checkpoint_path": str(ckpt_path) if ckpt_path.exists() else None,
                "mean_loss_total": float(np.mean([float(item["loss_total"]) for item in items])) if items else None,
                "mean_fallback_ratio": float(np.mean([float(item["fallback_ratio"]) for item in items if item.get("fallback_ratio", "") != ""])) if items else None,
                "peak_gpu_memory_mb": float(np.max([float(item["peak_gpu_memory_mb"]) for item in items])) if items else None,
            }
        )
    return summaries


def load_existing_pool_manifest_rows() -> list[dict[str, Any]]:
    return read_csv_rows(REPORTS_DIR / "sw11_failure_target_pool_manifest.csv")


def load_sw101_baseline_alignment_rows() -> list[dict[str, Any]]:
    waterfall_rows = read_csv_rows(SW101_REPORTS / "h2_to_semantic_occ_waterfall.csv")
    alignment_rows = read_csv_rows(SW101_REPORTS / "h2_native_assignment_alignment_metrics.csv")
    iou_lookup: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for row in alignment_rows:
        if row.get("h2_score_threshold") != "0.5":
            continue
        checkpoint = row["checkpoint_name"]
        perturbation = row["perturbation_id"]
        horizon = int(row["horizon_s"])
        sector = row["sector_name"]
        class_group = row["class_group"]
        error_type = row["error_type"]
        if error_type != "false_free":
            continue
        key = (checkpoint, perturbation, horizon, f"{sector}:{class_group}")
        bucket = iou_lookup.setdefault(key, {"count": 0.0, "iou": 0.0, "native_low_h2": 0.0})
        bucket["count"] += 1.0
        bucket["iou"] += float(row["H2_native_IoU"])
        bucket["native_low_h2"] += float(row["native_contributor_but_low_H2_score_ratio"])

    scenario_map = {
        "front_sector_false_free": ("A10_front_false_free", "front:all"),
        "small_object_false_free": ("small_object_false_free", "all:small_object"),
        "new_visible_false_free": ("new_visible_false_free", "all:all"),
    }
    exported: list[dict[str, Any]] = []
    for row in waterfall_rows:
        if row["checkpoint_name"] not in {"epoch_56_original", "P4_H2_tinyH1_500iter", "routeA_best_h2_scaled_iter3000"}:
            continue
        mapped = scenario_map.get(row["scenario_name"])
        if mapped is None:
            continue
        scenario_name, iou_key_name = mapped
        key = (row["checkpoint_name"], row["perturbation_id"], int(row["horizon_s"]), iou_key_name)
        aux = iou_lookup.get(key, {"count": 0.0, "iou": 0.0, "native_low_h2": 0.0})
        exported.append(
            {
                "checkpoint_name": row["checkpoint_name"],
                "perturbation_id": row["perturbation_id"],
                "horizon_s": int(row["horizon_s"]),
                "scenario_name": scenario_name,
                "voxel_count": int(row["voxel_count"]),
                "has_h2_high_score_ratio": float(row["has_h2_high_score_ratio"]),
                "H2_native_IoU": safe_div(aux["iou"], aux["count"]),
                "native_contributor_but_low_H2_score_ratio": safe_div(aux["native_low_h2"], aux["count"]),
                "has_native_exact_assignment_ratio": float(row["has_native_exact_assignment_ratio"]),
                "native_score_gate_pass_ratio": float(row["passes_native_gate_ratio"]),
                "has_final_contributor_ratio": float(row["has_final_contributor_ratio"]),
                "GT_class_top3_survival_ratio": float(row["gt_class_topk_contains_target_ratio"]),
                "GT_class_top5_survival_ratio": float(row["gt_class_topk_contains_target_ratio"]),
                "final_semantic_occ_TP_recovery_ratio": float(row["tp_ratio"]),
                "pred_gt_density_proxy": float(row["has_final_contributor_ratio"]),
                "neighbor_leakage_ratio": 0.0,
            }
        )
    exported.sort(key=lambda row: (row["checkpoint_name"], row["perturbation_id"], int(row["horizon_s"]), row["scenario_name"]))
    return exported


def load_sw101_baseline_taxonomy_counters() -> Counter[str]:
    rows = read_csv_rows(SW101_REPORTS / "h2_native_mismatch_taxonomy.csv")
    counter: Counter[str] = Counter()
    for row in rows:
        checkpoint = row.get("checkpoint_name", "")
        if checkpoint in {"epoch_56_original", "P4_H2_tinyH1_500iter", "routeA_best_h2_scaled_iter3000"}:
            counter[row["mismatch_type"]] += 1
    return counter


def update_survival_aggregate(agg: dict[tuple[str, str, int, str], dict[str, float]], rows: list[dict[str, Any]]) -> None:
    def has_pool(row: dict[str, Any], pool_name: str) -> bool:
        if row.get("pool_name") == pool_name:
            return True
        try:
            hits = json.loads(row.get("pool_hits", "[]"))
        except Exception:
            hits = []
        return pool_name in hits

    for scenario_name, scenario_filter in [
        ("A10_front_false_free", lambda r: r["perturbation_id"] == "A10_drop_front_triplet" and has_pool(r, "F1_A10_front_false_free")),
        ("small_object_false_free", lambda r: has_pool(r, "F2_small_object_false_free")),
        ("new_visible_false_free", lambda r: has_pool(r, "F3_new_visible_false_free")),
        ("future_false_free", lambda r: has_pool(r, "F4_future_risk_false_free")),
        ("native_missing", lambda r: has_pool(r, "F5_native_contributor_missing")),
        ("semantic_survival_failure", lambda r: has_pool(r, "F6_semantic_survival_failure")),
    ]:
        filtered = [row for row in rows if scenario_filter(row)]
        if not filtered:
            continue
        key = (filtered[0]["checkpoint_name"], filtered[0]["perturbation_id"], int(filtered[0]["horizon_s"]), scenario_name)
        bucket = agg.setdefault(
            key,
            {
                "voxel_count": 0.0,
                "h2_high": 0.0,
                "native_exact": 0.0,
                "h2_native_intersection": 0.0,
                "union_h2_native": 0.0,
                "native_low_h2": 0.0,
                "gate_pass": 0.0,
                "final_contributor": 0.0,
                "gt_top3": 0.0,
                "gt_top5": 0.0,
                "tp_recovery": 0.0,
                "neighbor_leak": 0.0,
            },
        )
        for item in filtered:
            bucket["voxel_count"] += 1.0
            h2_high = 1.0 if item["h2_high_score_thr05"] else 0.0
            native_exact = 1.0 if item["native_exact_assignment_exists"] else 0.0
            bucket["h2_high"] += h2_high
            bucket["native_exact"] += native_exact
            bucket["h2_native_intersection"] += 1.0 if (h2_high and native_exact) else 0.0
            bucket["union_h2_native"] += 1.0 if (h2_high or native_exact) else 0.0
            bucket["native_low_h2"] += 1.0 if (native_exact and not h2_high) else 0.0
            bucket["gate_pass"] += 1.0 if item["native_score_gate_pass"] else 0.0
            bucket["final_contributor"] += 1.0 if item["final_contributor_exists"] else 0.0
            bucket["gt_top3"] += 1.0 if item["gt_class_top3"] else 0.0
            bucket["gt_top5"] += 1.0 if item["gt_class_top5"] else 0.0
            bucket["tp_recovery"] += 1.0 if item["final_error_type"] == "tp" else 0.0
            bucket["neighbor_leak"] += 1.0 if (item["neighbor_voxel_assignment"] and not item["native_exact_assignment_exists"]) else 0.0


def finalize_survival_aggregate(agg: dict[tuple[str, str, int, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (checkpoint_name, perturbation_id, horizon_s, scenario_name), bucket in agg.items():
        count = max(bucket["voxel_count"], 1.0)
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "perturbation_id": perturbation_id,
                "horizon_s": horizon_s,
                "scenario_name": scenario_name,
                "voxel_count": int(bucket["voxel_count"]),
                "has_h2_high_score_ratio": bucket["h2_high"] / count,
                "H2_native_IoU": bucket["h2_native_intersection"] / max(bucket["union_h2_native"], 1.0),
                "native_contributor_but_low_H2_score_ratio": bucket["native_low_h2"] / max(bucket["native_exact"], 1.0),
                "has_native_exact_assignment_ratio": bucket["native_exact"] / count,
                "native_score_gate_pass_ratio": bucket["gate_pass"] / count,
                "has_final_contributor_ratio": bucket["final_contributor"] / count,
                "GT_class_top3_survival_ratio": bucket["gt_top3"] / count,
                "GT_class_top5_survival_ratio": bucket["gt_top5"] / count,
                "final_semantic_occ_TP_recovery_ratio": bucket["tp_recovery"] / count,
                "pred_gt_density_proxy": bucket["final_contributor"] / count,
                "neighbor_leakage_ratio": bucket["neighbor_leak"] / count,
            }
        )
    rows.sort(key=lambda row: (row["checkpoint_name"], row["perturbation_id"], int(row["horizon_s"]), row["scenario_name"]))
    return rows


def decide(rows: list[dict[str, Any]], smoke_metrics_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["checkpoint_name"] in {"P1_risk_targeted_S1_20iter", "P2_native_missing_S2_20iter", "P3_survival_aware_S3_20iter"}
        and row["scenario_name"] == "A10_front_false_free"
        and row["perturbation_id"] == "A10_drop_front_triplet"
        and int(row["horizon_s"]) == 6
    ]
    if not candidates:
        decision = {
            "decision_type": "B7_instrumentation_blocked",
            "summary": "No smoke survival rows were available for A10 front h6, so the survival chain could not be compared reliably.",
            "best_candidate": None,
            "next_unique_action": "extend instrumentation before any further training",
        }
        return decision, {"plan_type": "SW-12E_instrumentation_or_target_redesign_plan", "next_route": "instrumentation_or_target_redesign"}
    best = max(candidates, key=lambda row: float(row["has_h2_high_score_ratio"]))
    base = next(
        row
        for row in rows
        if row["checkpoint_name"] == "epoch_56_original"
        and row["scenario_name"] == "A10_front_false_free"
        and row["perturbation_id"] == "A10_drop_front_triplet"
        and int(row["horizon_s"]) == 6
    )
    h2_up = float(best["has_h2_high_score_ratio"]) > float(base["has_h2_high_score_ratio"]) + 0.01
    native_up = float(best["has_native_exact_assignment_ratio"]) > float(base["has_native_exact_assignment_ratio"]) + 0.01
    final_up = float(best["has_final_contributor_ratio"]) > float(base["has_final_contributor_ratio"]) + 0.01
    semantic_up = float(best["GT_class_top3_survival_ratio"]) > float(base["GT_class_top3_survival_ratio"]) + 0.01
    density_risk = any((row.get("mean_fallback_ratio") is not None and float(row["mean_fallback_ratio"]) > 0.5) for row in smoke_metrics_rows)
    if not h2_up:
        decision_type = "B1_targeting_failed"
        summary = "Risk-targeted sampling did not materially raise H2 high-score coverage on the failure voxels."
        next_route = "SW-12E_instrumentation_or_target_redesign_plan"
        plan_type = "SW-12E_instrumentation_or_target_redesign_plan"
    elif h2_up and not native_up:
        decision_type = "B2_h2_lights_up_but_no_native_assignment"
        summary = "H2 high-score coverage increased, but native exact assignment stayed at or near zero on the failure voxels."
        next_route = "SW-12A_getocc_soft_neighbor_routing_plan"
        plan_type = "SW-12A_getocc_soft_neighbor_routing_plan"
    elif native_up and not final_up:
        decision_type = "B3_native_assignment_improves_but_no_final_contributor"
        summary = "Native exact assignment improved, but final contributor survival did not follow."
        next_route = "SW-12B_aggregation_repair_plan"
        plan_type = "SW-12B_aggregation_repair_plan"
    elif final_up and not semantic_up:
        decision_type = "B4_final_contributor_improves_but_semantic_fails"
        summary = "Final contributor survival improved, but GT top-k semantic survival did not."
        next_route = "SW-12C_semantic_gate_coupling_plan"
        plan_type = "SW-12C_semantic_gate_coupling_plan"
    elif h2_up and native_up and final_up and semantic_up:
        if density_risk:
            decision_type = "B6_density_or_false_positive_risk"
            summary = "The survival chain improved, but the smoke run showed fallback or density-risk pressure that needs guarding first."
            next_route = "SW-12A_getocc_soft_neighbor_routing_plan"
            plan_type = "SW-12A_getocc_soft_neighbor_routing_plan"
        else:
            decision_type = "B5_full_survival_improves"
            summary = "H2 high score, native exact assignment, final contributor, and GT top-k survival all improved in the subset smoke."
            next_route = "SW-12D_targeted_training_plan"
            plan_type = "SW-12D_targeted_training_plan"
    else:
        decision_type = "B7_instrumentation_blocked"
        summary = "The smoke run moved some survival-chain metrics but not in a way that cleanly isolates the next bottleneck."
        next_route = "SW-12E_instrumentation_or_target_redesign_plan"
        plan_type = "SW-12E_instrumentation_or_target_redesign_plan"
    decision = {
        "decision_type": decision_type,
        "summary": summary,
        "best_candidate": best["checkpoint_name"],
        "best_candidate_metrics": best,
        "baseline_metrics": base,
        "next_unique_action": next_route,
    }
    plan = {"plan_type": plan_type, "next_route": next_route}
    return decision, plan


def make_sw12_plan(plan: dict[str, Any], decision: dict[str, Any]) -> tuple[dict[str, Any], str]:
    plan_type = plan["plan_type"]
    if plan_type == "SW-12A_getocc_soft_neighbor_routing_plan":
        payload = {
            "plan_type": plan_type,
            "focus": "soft neighbor contributor routing",
            "items": [
                "keep native get_occ as default and add a diagnostic soft-neighbor flag",
                "radius <= 1 voxel",
                "strict confidence gate and density cap",
                "diagnostic replay before any training",
            ],
        }
    elif plan_type == "SW-12B_aggregation_repair_plan":
        payload = {
            "plan_type": plan_type,
            "focus": "scatter_max / post-padding / aggregation repair",
            "items": [
                "audit scatter_max conflicts",
                "measure post-padding survival drop",
                "test contributor survival cap and conflict resolution in replay",
            ],
        }
    elif plan_type == "SW-12C_semantic_gate_coupling_plan":
        payload = {
            "plan_type": plan_type,
            "focus": "semantic gate coupling",
            "items": [
                "GT class top-k margin loss",
                "foreground gate survival loss",
                "class-aware contributor / semantic coupling in replay before training",
            ],
        }
    elif plan_type == "SW-12D_targeted_training_plan":
        payload = {
            "plan_type": plan_type,
            "focus": "500/1000 iter targeted training",
            "items": [
                "500 iter first, then optional 1000 iter",
                "eval_core_20 only",
                "retain clean / false-positive safety gates",
            ],
        }
    else:
        payload = {
            "plan_type": "SW-12E_instrumentation_or_target_redesign_plan",
            "focus": "instrumentation or target redesign",
            "items": [
                "redefine failure-target pools or pool weights",
                "expose more native gate / aggregation tensors",
                "avoid long training before the mechanism is clear",
            ],
        }
    md = "\n".join(
        [
            f"recommended route: {payload['plan_type']}",
            f"triggered by decision: {decision['decision_type']}",
            *[f"- {item}" for item in payload["items"]],
            "",
        ]
    )
    return payload, md


def make_final_report(report_json: dict[str, Any]) -> str:
    lines = [
        "# Stage SW-11 Risk-targeted H2 Resampling with Native Contributor Survival Check",
        "",
        "1. Executive summary",
        f"- {report_json['executive_summary']}",
        "",
        "2. Why SW-11 follows SW-10.1",
        "- SW-11 is not long training.",
        "- SW-11 is not an official benchmark.",
        "- SW-11 does not claim model improvement.",
        "",
        "3. SW-10.1 evidence digest",
        f"- {report_json['digest_headline']}",
        "",
        "4. Failure target pool construction",
        f"- {report_json['pool_headline']}",
        "",
        "5. Risk-targeted H2 sampler implementation",
        f"- {report_json['sampler_headline']}",
        "",
        "6. Native contributor survival instrumentation",
        f"- {report_json['survival_headline']}",
        "",
        "7. Gradient check",
        f"- {report_json['gradient_headline']}",
        "",
        "8. 20-iter smoke",
        f"- {report_json['smoke_headline']}",
        "",
        "9. Alignment survival retest",
        f"- {report_json['retest_headline']}",
        "",
        "10. Mismatch taxonomy update",
        f"- {report_json['taxonomy_headline']}",
        "",
        "11. Decision B1-B7",
        f"- {report_json['decision']['decision_type']}: {report_json['decision']['summary']}",
        "",
        "12. Recommended SW-12 route",
        f"- {report_json['sw12_plan']['plan_type']}",
        "",
        "13. Safe claims",
        "- subset diagnostic only",
        "- 20-iter smoke is only a mechanism test",
        "- no get_occ mainline modification in this stage",
        "- no official benchmark claim",
        "- no model improvement claim",
        "",
        "14. Limitations",
        "- no long training in SW-11",
        "- no full validation benchmark",
        "- no native get_occ mainline replacement",
        "",
        "15. Next unique action",
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
        "stage": "SW-11",
        "start_time": now_iso(),
        "max_hours": args.max_hours,
        "reserve_report_minutes": args.reserve_report_minutes,
        "phases": [],
    }
    time_manifest_path = REPORTS_DIR / "sw11_time_budget_manifest.json"
    write_json(time_manifest_path, time_manifest)

    samples = parse_int_list(args.samples)
    perturbations = [item.strip() for item in args.perturbations.split(",") if item.strip()]
    horizons = parse_int_list(args.horizons)
    sectors = {name: tensor.cpu() for name, tensor in sw7.build_sector_masks().items()}
    h2_cfg = load_h2_cfg()

    digest, digest_headline = digest_sw101()
    baseline_specs = discover_baseline_specs()
    sampler_modes = build_sampler_modes()
    sampler = RiskTargetedH2Loss(h2_cfg, args.seed, args.max_positive_targets, args.max_negative_targets)
    pool_manifest_rows: list[dict[str, Any]] = []

    if args.resume_from == "start":
        phase = phase_block(time_manifest, time_manifest_path, "phase1_sw101_digest")
        stage_logger.start("phase1_sw101_digest")
        write_json(REPORTS_DIR / "sw101_digest_for_sw11.json", digest)
        write_md(
            REPORTS_DIR / "sw11_objective.md",
            "\n".join(
                [
                    digest_headline,
                    "",
                    "1. SW-10.1 decision = A5_h2_target_not_focusing_failure.",
                    "2. H2 currently does not stably focus false-free risk voxels.",
                    "3. On A10 front false-free, P4 briefly raised has_h2_high_score_ratio, but native exact assignment and final contributor stayed at 0.",
                    "4. P4 short-term signal did not come with native exact assignment or final contributor improvement.",
                    "5. SW-11 therefore redefines H2 target sampling and runs a native contributor survival check instead of long training.",
                    "",
                ]
            ),
        )
        stage_logger.done("phase1_sw101_digest")
        end_phase(time_manifest, time_manifest_path, phase, "done")

        phase = phase_block(time_manifest, time_manifest_path, "phase2_pool_and_baseline_survival")
        stage_logger.start("phase2_pool_and_baseline_survival")
        pool_manifest_rows: list[dict[str, Any]] = []
        baseline_survival_rows: list[dict[str, Any]] = []
        completed = 0
        total_cases = len(baseline_specs) * len(samples) * len(perturbations) * len(horizons)
        for spec in baseline_specs:
            cfg, dataset, model = build_runtime(spec.config_path, spec.checkpoint_path, train=False)
            try:
                for perturbation_id in perturbations:
                    for sample_index in samples:
                        if time.time() >= deadline_ts - args.reserve_report_minutes * 60.0:
                            raise TimeoutError("time budget reached before finishing baseline replay")
                        case = eval_case(model, dataset, sample_index, perturbation_id, horizons, h2_cfg, sectors)
                        for horizon_s in horizons:
                            item = case["per_horizon"].get(horizon_s)
                            if item is None:
                                continue
                            completed += 1
                            dump_path = ARTIFACTS_DIR / "target_pools" / spec.name / perturbation_id / f"sample_{sample_index}_h{horizon_s}.npz"
                            save_pool_npz(dump_path, item["pool_masks"])
                            for pool_name, mask in item["pool_masks"].items():
                                if pool_name.startswith("general_"):
                                    continue
                                stats = pool_stats(mask, item["gt_h"], item["gt0"], item["final_occ"], sectors)
                                pool_manifest_rows.append(
                                    {
                                        "pool_name": pool_name,
                                        "checkpoint_source": spec.name,
                                        "perturbation": perturbation_id,
                                        "sample_id": sample_index,
                                        "horizon": horizon_s,
                                        "voxel_count": stats["voxel_count"],
                                        "sector_distribution": json.dumps(stats["sector_distribution"], ensure_ascii=False),
                                        "class_group_distribution": json.dumps(stats["class_group_distribution"], ensure_ascii=False),
                                        "error_type_distribution": json.dumps(stats["error_type_distribution"], ensure_ascii=False),
                                        "enough_samples": stats["enough_samples"],
                                        "insufficiency_reason": "" if stats["enough_samples"] else "pool_empty_for_case",
                                        "pool_npz_path": str(dump_path),
                                    }
                                )
                            baseline_survival_rows.extend(
                                build_survival_rows_for_case(
                                    spec.name,
                                    sample_index,
                                    perturbation_id,
                                    horizon_s,
                                    item["gt_h"],
                                    item["gt0"],
                                    item["final_occ"],
                                    item["debug"],
                                    item["h2_proxy"],
                                    item["pool_masks"],
                                    sectors,
                                )
                            )
                            stage_logger.progress("phase2_pool_and_baseline_survival", completed, total_cases, checkpoint_name=spec.name, perturbation_id=perturbation_id, sample_index=sample_index, horizon_s=horizon_s)
            finally:
                del model, dataset, cfg
                safe_cuda_cleanup()
        write_csv(REPORTS_DIR / "sw11_failure_target_pool_manifest.csv", pool_manifest_rows)
        write_md(
            REPORTS_DIR / "sw11_failure_target_pool_summary.md",
            f"Failure-target pools were built for {completed} subset cases across baseline checkpoints; empty pools were recorded rather than synthesized.\n",
        )
        write_json(
            REPORTS_DIR / "sw11_native_survival_schema.json",
            {
                "stages": {
                    "A": ["is_selected_by_sampler", "pool_name", "sample_index", "perturbation_id", "horizon_s", "class_group", "sector_name", "error_type"],
                    "B": ["h2_score", "h2_distance", "h2_rank", "h2_high_score_thr03", "h2_high_score_thr05", "h2_high_score_thr07", "nearest_support_distance", "support_confidence"],
                    "C": ["native_exact_assignment_exists", "native_contributor_count", "native_voxel_index_match", "floor_voxel_index", "neighbor_voxel_assignment"],
                    "D": ["foreground_score", "semantic_score", "native_score_gate_pass", "valid_mask_pass", "range_mask_pass"],
                    "E": ["survives_scatter_max", "post_padding_active", "final_contributor_exists", "contributor_class_logits"],
                    "F": ["gt_class_rank", "gt_class_top1", "gt_class_top3", "gt_class_top5", "semantic_margin", "final_semantic_occ_class", "final_error_type"],
                }
            },
        )
        write_csv(REPORTS_DIR / "sw11_native_survival_check_baseline.csv", baseline_survival_rows)
        plot_target_pool_distribution(pool_manifest_rows, FIGURES_DIR / "sw11_target_pool_distribution.png")
        stage_logger.done("phase2_pool_and_baseline_survival", completed_cases=completed)
        end_phase(time_manifest, time_manifest_path, phase, "done", completed_cases=completed)

        phase = phase_block(time_manifest, time_manifest_path, "phase3_sampler_and_phase5_gradient")
        stage_logger.start("phase3_sampler_and_phase5_gradient")
        sampler_manifest = {
            "seed": args.seed,
            "max_positive_targets": args.max_positive_targets,
            "max_negative_targets": args.max_negative_targets,
            "front_false_free_weight": 2.0,
            "small_object_weight": 2.0,
            "new_visible_weight": 2.0,
            "future_h46_weight": 1.5,
            "clean_control_weight": 0.5,
            "negative_leakage_weight_relative": 0.5,
            "modes": {name: normalize_export(mode.__dict__) for name, mode in sampler_modes.items()},
        }
        write_json(REPORTS_DIR / "sw11_sampler_config_manifest.json", sampler_manifest)
        write_md(REPORTS_DIR / "sw11_sampler_implementation_summary.md", "SW-11 keeps the original H2 sampler as S0 and adds S1/S2/S3 risk-targeted modes outside get_occ mainline, with deterministic pool fallback logging.\n")
        gradient_rows = [
            gradient_check("G0_original_H2_sampler", sampler_modes["S0_original_h2_sampling"], h2_cfg, sectors, sampler),
            gradient_check("G1_risk_targeted_S1", sampler_modes["S1_risk_targeted_h2_sampling"], h2_cfg, sectors, sampler),
            gradient_check("G2_native_missing_S2", sampler_modes["S2_native_missing_h2_sampling"], h2_cfg, sectors, sampler),
            gradient_check("G3_survival_aware_S3", sampler_modes["S3_survival_aware_h2_sampling"], h2_cfg, sectors, sampler),
            gradient_check("G4_risk_targeted_with_hard_negative", sampler_modes["G4_risk_targeted_with_hard_negative"], h2_cfg, sectors, sampler),
        ]
        write_csv(REPORTS_DIR / "sw11_gradient_check.csv", gradient_rows)
        write_md(REPORTS_DIR / "sw11_gradient_check_summary.md", "\n".join([f"- {row['experiment_id']}: grad_finite={row['grad_finite']}, fallback_ratio={row['fallback_ratio']:.4f}, eligible_for_smoke={row['eligible_for_smoke']}" for row in gradient_rows]) + "\n")
        plot_gradient_norms(gradient_rows, FIGURES_DIR / "sw11_gradient_norms.png")
        stage_logger.done("phase3_sampler_and_phase5_gradient")
        end_phase(time_manifest, time_manifest_path, phase, "done")

        phase = phase_block(time_manifest, time_manifest_path, "phase6_smoke")
        stage_logger.start("phase6_smoke")
        smoke_train_rows: list[dict[str, Any]] = []
        smoke_summary_rows: list[dict[str, Any]] = []
        gradient_lookup = {row["experiment_id"]: row for row in gradient_rows}
        smoke_experiments = [
            ("P0_original_H2_sampling_20iter", "G0_original_H2_sampler", sampler_modes["S0_original_h2_sampling"]),
            ("P1_risk_targeted_S1_20iter", "G1_risk_targeted_S1", sampler_modes["S1_risk_targeted_h2_sampling"]),
            ("P2_native_missing_S2_20iter", "G2_native_missing_S2", sampler_modes["S2_native_missing_h2_sampling"]),
        ]
        if gradient_lookup["G3_survival_aware_S3"]["eligible_for_smoke"]:
            smoke_experiments.append(("P3_survival_aware_S3_20iter", "G3_survival_aware_S3", sampler_modes["S3_survival_aware_h2_sampling"]))
        for experiment_id, gradient_id, sampler_mode in smoke_experiments:
            if not gradient_lookup[gradient_id]["eligible_for_smoke"]:
                smoke_summary_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "sampler_mode": sampler_mode.sampler_mode,
                        "completed_iters": 0,
                        "stop_reason": "skipped_by_gradient_gate",
                        "checkpoint_path": None,
                        "mean_loss_total": None,
                        "mean_fallback_ratio": None,
                        "peak_gpu_memory_mb": None,
                    }
                )
                continue
            ckpt_path, iter_rows, summary = smoke_train(experiment_id, sampler_mode, h2_cfg, sectors, sampler, args.smoke_iters)
            smoke_train_rows.extend(iter_rows)
            smoke_summary_rows.append(summary)
        write_csv(REPORTS_DIR / "sw11_20iter_smoke_train_metrics.csv", smoke_train_rows if smoke_train_rows else [{"skipped_reason": "no_eligible_smoke_experiments"}])
        stage_logger.done("phase6_smoke")
        end_phase(time_manifest, time_manifest_path, phase, "done")
    else:
        gradient_rows = read_csv_rows(REPORTS_DIR / "sw11_gradient_check.csv")
        smoke_summary_rows = reconstruct_smoke_summaries()
        pool_manifest_rows = load_existing_pool_manifest_rows()

    phase = phase_block(time_manifest, time_manifest_path, "phase7_retest_and_phase8_taxonomy")
    stage_logger.start("phase7_retest_and_phase8_taxonomy")
    smoke_retest_rows = [
        row
        for row in smoke_summary_rows
        if row.get("checkpoint_path")
        and row["experiment_id"] in {"P1_risk_targeted_S1_20iter", "P2_native_missing_S2_20iter", "P3_survival_aware_S3_20iter"}
    ]
    all_specs = [CheckpointSpec(row["experiment_id"], Path(row["checkpoint_path"]), BASE_CONFIG_PATH, "sw11_smoke") for row in smoke_retest_rows]
    survival_agg: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for base_row in load_sw101_baseline_alignment_rows():
        voxel_count = float(base_row["voxel_count"])
        h2_high_count = float(base_row["has_h2_high_score_ratio"]) * voxel_count
        native_exact_count = float(base_row["has_native_exact_assignment_ratio"]) * voxel_count
        iou = float(base_row["H2_native_IoU"])
        intersection = safe_div(iou * (h2_high_count + native_exact_count), 1.0 + iou)
        union_count = max(1.0, h2_high_count + native_exact_count - intersection)
        key = (
            base_row["checkpoint_name"],
            base_row["perturbation_id"],
            int(base_row["horizon_s"]),
            base_row["scenario_name"],
        )
        survival_agg[key] = {
            "voxel_count": voxel_count,
            "h2_high": h2_high_count,
            "native_exact": native_exact_count,
            "h2_native_intersection": intersection,
            "union_h2_native": union_count,
            "native_low_h2": float(base_row["native_contributor_but_low_H2_score_ratio"]) * max(
                1.0, native_exact_count
            ),
            "gate_pass": float(base_row["native_score_gate_pass_ratio"]) * voxel_count,
            "final_contributor": float(base_row["has_final_contributor_ratio"]) * voxel_count,
            "gt_top3": float(base_row["GT_class_top3_survival_ratio"]) * voxel_count,
            "gt_top5": float(base_row["GT_class_top5_survival_ratio"]) * voxel_count,
            "tp_recovery": float(base_row["final_semantic_occ_TP_recovery_ratio"]) * voxel_count,
            "neighbor_leak": float(base_row["neighbor_leakage_ratio"]) * voxel_count,
        }
    retest_taxonomy_path = REPORTS_DIR / "sw11_mismatch_taxonomy_after_resampling.csv"
    if retest_taxonomy_path.exists():
        retest_taxonomy_path.unlink()
    taxonomy_counter_baseline: Counter[str] = load_sw101_baseline_taxonomy_counters()
    taxonomy_counter_smoke: Counter[str] = Counter()
    completed = 0
    total_cases = len(all_specs) * len(samples) * len(perturbations) * len(horizons)
    for spec in all_specs:
        cfg, dataset, model = build_runtime(spec.config_path, spec.checkpoint_path, train=False)
        try:
            for perturbation_id in perturbations:
                for sample_index in samples:
                    case = eval_case(model, dataset, sample_index, perturbation_id, horizons, h2_cfg, sectors)
                    for horizon_s in horizons:
                        item = case["per_horizon"].get(horizon_s)
                        if item is None:
                            continue
                        completed += 1
                        survival_rows_case = build_survival_rows_for_case(
                            spec.name,
                            sample_index,
                            perturbation_id,
                            horizon_s,
                            item["gt_h"],
                            item["gt0"],
                            item["final_occ"],
                            item["debug"],
                            item["h2_proxy"],
                            item["pool_masks"],
                            sectors,
                        )
                        update_survival_aggregate(survival_agg, survival_rows_case)
                        taxonomy_rows_case = sw101.compute_mismatch_taxonomy_for_case(
                            spec.name,
                            perturbation_id,
                            sample_index,
                            horizon_s,
                            item["gt_h"],
                            item["gt0"],
                            item["final_occ"],
                            item["debug"],
                            item["h2_proxy"],
                            h2_cfg,
                            sectors,
                        )
                        append_csv_rows(retest_taxonomy_path, taxonomy_rows_case)
                        for row in taxonomy_rows_case:
                            if spec.family == "sw11_smoke":
                                taxonomy_counter_smoke[row["mismatch_type"]] += 1
                            else:
                                taxonomy_counter_baseline[row["mismatch_type"]] += 1
                        stage_logger.progress("phase7_retest_and_phase8_taxonomy", completed, total_cases, checkpoint_name=spec.name, perturbation_id=perturbation_id, sample_index=sample_index, horizon_s=horizon_s)
                        del survival_rows_case
                        del taxonomy_rows_case
                        safe_cuda_cleanup()
        finally:
            del model, dataset, cfg
            safe_cuda_cleanup()
    alignment_survival_rows = finalize_survival_aggregate(survival_agg)
    write_csv(REPORTS_DIR / "sw11_alignment_survival_retest.csv", alignment_survival_rows)
    retest_headline = summarize_alignment_survival(alignment_survival_rows)
    write_md(REPORTS_DIR / "sw11_alignment_survival_retest_summary.md", retest_headline + "\n")
    mismatch_summary_rows = [
        {"mismatch_type": key, "baseline_count": taxonomy_counter_baseline[key], "smoke_count": taxonomy_counter_smoke[key]}
        for key in sorted(set(taxonomy_counter_baseline.keys()) | set(taxonomy_counter_smoke.keys()))
    ]
    dominant_after = max(mismatch_summary_rows, key=lambda row: row["smoke_count"]) if mismatch_summary_rows else {"mismatch_type": "none"}
    taxonomy_headline = f"After risk-targeted 20-iter smoke, dominant smoke mismatch type was {dominant_after['mismatch_type']}."
    write_md(REPORTS_DIR / "sw11_mismatch_taxonomy_after_resampling_summary.md", taxonomy_headline + "\n")
    plot_survival_before_after([row for row in alignment_survival_rows if row["scenario_name"] == "A10_front_false_free" and int(row["horizon_s"]) == 6], FIGURES_DIR / "sw11_a10_front_falsefree_survival.png", "A10_front_false_free")
    plot_survival_before_after([row for row in alignment_survival_rows if row["scenario_name"] in {"small_object_false_free", "new_visible_false_free"} and int(row["horizon_s"]) == 6], FIGURES_DIR / "sw11_small_newvisible_survival.png", "small_object_false_free")
    plot_survival_before_after([row for row in alignment_survival_rows if row["scenario_name"] == "A10_front_false_free" and int(row["horizon_s"]) == 6], FIGURES_DIR / "sw11_survival_chain_before_after.png", "A10_front_false_free")
    plot_mismatch_before_after(mismatch_summary_rows, FIGURES_DIR / "sw11_mismatch_taxonomy_before_after.png")
    stage_logger.done("phase7_retest_and_phase8_taxonomy", completed_cases=completed)
    end_phase(time_manifest, time_manifest_path, phase, "done", completed_cases=completed)

    phase = phase_block(time_manifest, time_manifest_path, "phase9_decision_and_phase10_sw12_plan")
    stage_logger.start("phase9_decision_and_phase10_sw12_plan")
    decision, plan_stub = decide(alignment_survival_rows, smoke_summary_rows)
    write_json(REPORTS_DIR / "sw11_risk_targeted_h2_decision.json", decision)
    write_md(REPORTS_DIR / "sw11_risk_targeted_h2_decision.md", json.dumps(normalize_export(decision), indent=2, ensure_ascii=False) + "\n")
    sw12_plan, sw12_md = make_sw12_plan(plan_stub, decision)
    write_json(REPORTS_DIR / "sw12_recommended_route_plan.json", sw12_plan)
    write_md(REPORTS_DIR / "sw12_recommended_route_plan.md", sw12_md)
    plot_decision_flow(decision["decision_type"], FIGURES_DIR / "sw11_decision_flow.png")
    stage_logger.done("phase9_decision_and_phase10_sw12_plan", decision_type=decision["decision_type"])
    end_phase(time_manifest, time_manifest_path, phase, "done", decision_type=decision["decision_type"])

    phase = phase_block(time_manifest, time_manifest_path, "phase12_final_report")
    stage_logger.start("phase12_final_report")
    report_json = {
        "executive_summary": "SW-11 executed a risk-targeted H2 resampling mechanism test with native contributor survival tracing and 20-iter smoke only.",
        "digest_headline": digest_headline,
        "pool_headline": f"Failure-target pools were built from {len(pool_manifest_rows)} pool-case rows without synthesizing missing voxels.",
        "sampler_headline": "S0 preserved original H2 sampling, while S1/S2/S3 introduced deterministic failure-pool resampling and hard-negative tracking outside get_occ mainline.",
        "survival_headline": "Baseline and smoke checkpoints were replayed through the same native survival chain from H2 score to native exact assignment, final contributor, and GT top-k survival.",
        "gradient_headline": "\n".join([f"{row['experiment_id']} eligible={row['eligible_for_smoke']}" for row in gradient_rows]),
        "smoke_headline": f"Smoke runs completed for {', '.join(row['experiment_id'] for row in smoke_summary_rows)}.",
        "retest_headline": retest_headline,
        "taxonomy_headline": taxonomy_headline,
        "decision": decision,
        "sw12_plan": sw12_plan,
    }
    write_md(REPORTS_DIR / "stage_sw11_risk_targeted_h2_resampling_report.md", make_final_report(report_json))
    write_json(REPORTS_DIR / "stage_sw11_risk_targeted_h2_resampling_report.json", report_json)
    stage_logger.done("phase12_final_report")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    time_manifest["end_time"] = now_iso()
    time_manifest["wall_clock_sec"] = time.time() - started_at
    write_json(time_manifest_path, time_manifest)


if __name__ == "__main__":
    main()
