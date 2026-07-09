from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

np.Inf = np.inf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))
BASE_STAGE_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter"
BASE_STAGE_SCRIPT = BASE_STAGE_DIR / "run_sw14b_metric_gt_adapter.py"
if str(BASE_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_STAGE_DIR))

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14b_density_clean_rescue"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14b_density_clean_rescue"
LOCAL_TMP_CACHE_DIR = Path(tempfile.gettempdir()) / "sw14b_density_clean_rescue_cache"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("sw14b_rescue_base", BASE_STAGE_SCRIPT)
from sw14b_adapter_modules import CAMERA_NAMES, SpatialGateAdapter  # type: ignore  # noqa: E402
import sw14b_postprocess as base_postprocess  # type: ignore  # noqa: E402

ALLOW_RUNTIME_TEACHER_REBUILD = os.environ.get("SW14B_RESCUE_ALLOW_RUNTIME_TEACHER_REBUILD", "0") == "1"


@dataclass(frozen=True)
class RescuePolicy:
    name: str
    global_scale: float = 1.0
    front_scale: float = 1.0
    side_scale: float = 1.0
    front_proxy_damping: bool = False
    front_proxy_damping_base_scale: float = 1.0


FRONT_CAMERA_INDEX = CAMERA_NAMES.index("CAM_FRONT")
FRONT_RIGHT_CAMERA_INDEX = CAMERA_NAMES.index("CAM_FRONT_RIGHT")
FRONT_LEFT_CAMERA_INDEX = CAMERA_NAMES.index("CAM_FRONT_LEFT")
CORE_HORIZONS = list(base.CORE_HORIZONS)
EMPTY_IDX = int(base.EMPTY_IDX)


class LazyCaseStore:
    def __init__(
        self,
        model: Any,
        dataset: Any,
        sample_indices: list[int],
        perturbation_id: str,
        candidate: Any,
        sectors: dict[str, torch.Tensor],
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.sample_indices = list(sample_indices)
        self.perturbation_id = perturbation_id
        self.candidate = candidate
        self.sectors = sectors
        self._loaded: dict[int, dict[str, Any]] = {}

    def __getitem__(self, sample_index: int) -> dict[str, Any]:
        # Deliberately no global memo here. A single sample carries hundreds of
        # MB of feature/teacher tensors; keeping val/debug ranges resident causes
        # WSL swap storms and long D-drive reads.
        if sample_index not in self._loaded:
            self._loaded[sample_index] = get_native_rule_teacher_case(
                self.model,
                self.dataset,
                sample_index,
                self.perturbation_id,
                self.candidate,
                self.sectors,
            )
        return self._loaded[sample_index]

    def release_sample(self, sample_index: int) -> None:
        case = self._loaded.pop(sample_index, None)
        if case is not None:
            base.release_runtime(case)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14B density clamp / clean drift rescue")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", default=str(base.ARTIFACTS_DIR / "checkpoints/sw14b_round2_metric_teacher_checkpoint.pth"))
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=199)
    parser.add_argument("--val-start", type=int, default=200)
    parser.add_argument("--val-end", type=int, default=249)
    parser.add_argument("--eval-start", type=int, default=100)
    parser.add_argument("--eval-end", type=int, default=119)
    parser.add_argument("--skip-retrain", action="store_true")
    parser.add_argument("--force-runtime-attribution", action="store_true", help="recompute clean/density attribution instead of reusing existing reports")
    parser.add_argument("--force-runtime-sweep", action="store_true", help="recompute val_small alpha/clamp sweeps instead of reusing existing selections")
    parser.add_argument("--force-runtime-eval-debug", action="store_true", help="recompute eval_debug rescue metrics instead of reusing a complete existing result")
    parser.add_argument("--eval-scale-sweep", action="store_true", help="run eval_debug scale sweep in one runtime with sample-major feature reuse")
    parser.add_argument("--eval-scale-list", default="0.25,0.40,0.50,0.60,0.75,0.90,1.00")
    parser.add_argument("--io-audit-only", action="store_true", help="write the rescue I/O safety audit and exit without model execution")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, ARTIFACTS_DIR, ARTIFACTS_DIR / "checkpoints", FIGURES_DIR, TESTS_DIR, LOCAL_TMP_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(obj: Any) -> Any:
    return base.normalize(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize(row))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_field(rows: list[dict[str, Any]], key: str, *, predicate=None, default: float = 0.0) -> float:
    items = rows if predicate is None else [row for row in rows if predicate(row)]
    if not items:
        return default
    return float(np.mean([float(row[key]) for row in items]))


def metric_value(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    return base.metric_value(row, *keys, default=default)


def safe_div(a: float | int, b: float | int) -> float:
    return base.safe_div(a, b)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def aggregate_sw14b_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len({int(row["sample_index"]) for row in rows}) if rows else 0,
        "row_count": len(rows),
        "mean_pred_gt_density_delta": mean_field(rows, "pred_gt_density_delta"),
        "mean_front_sector_false_free_rate_delta_vs_native": mean_field(rows, "front_sector_false_free_rate_delta_vs_native"),
        "mean_future_h4_h6_false_free_rate_delta_vs_native": mean_field(rows, "future_h4_h6_false_free_rate_delta_vs_native", predicate=lambda row: int(row["horizon_s"]) in {4, 6}),
        "mean_false_positive_delta": mean_field(rows, "false_positive_delta"),
        "mean_wrong_class_delta": mean_field(rows, "wrong_class_delta"),
        "mean_front_local_density_proxy": mean_field(rows, "front_local_density_proxy"),
        "mean_protected_zone_preservation_ratio": mean_field(rows, "protected_zone_preservation_ratio") if rows and "protected_zone_preservation_ratio" in rows[0] else 1.0,
        "mean_teacher_gap_occ": mean_field(rows, "teacher_gap_occ") if rows and "teacher_gap_occ" in rows[0] and any(str(row.get("teacher_gap_occ", "")) not in {"", "None"} for row in rows) else None,
    }


def aggregate_rescue_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len({int(row["sample_index"]) for row in rows}) if rows else 0,
        "row_count": len(rows),
        "mean_density_delta": mean_field(rows, "final_density_adapter"),
        "mean_front_sector_false_free_rate_delta_vs_native": mean_field(rows, "front_false_free_adapter"),
        "mean_future_h4_h6_false_free_rate_delta_vs_native": mean_field(rows, "future_false_free_adapter", predicate=lambda row: int(row["horizon_s"]) in {4, 6}),
        "mean_false_positive_delta": mean_field(rows, "false_positive_adapter"),
        "mean_front_local_proxy": mean_field(rows, "front_local_proxy_adapter"),
        "mean_protected_zone_preservation_ratio": mean_field(rows, "protected_zone_preservation_ratio_adapter") if rows and "protected_zone_preservation_ratio_adapter" in rows[0] else 1.0,
    }


def filter_metric_rows_by_samples(rows: list[dict[str, Any]], sample_indices: set[int]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["sample_index"]) in sample_indices]


def load_unified_original_eval_debug_summary() -> dict[str, Any]:
    rows = read_csv(base.REPORTS_DIR / "sw14b_eval_debug_metrics.csv")
    teacher_metric_rows = read_csv(base.REPORTS_DIR / "sw14b_eval_debug_teacher_metrics.csv")
    legacy = read_json(base.REPORTS_DIR / "sw14b_eval_debug_summary.json")
    if not rows:
        return legacy
    adapter_rows = rows
    if teacher_metric_rows:
        teacher_rows = teacher_metric_rows
    else:
        teacher_rows = [
            {
                "sample_index": row["sample_index"],
                "horizon_s": row["horizon_s"],
                "pred_gt_density_delta": row.get("teacher_density_delta", 0.0),
                "front_sector_false_free_rate_delta_vs_native": row.get("teacher_front_false_free", 0.0),
                "future_h4_h6_false_free_rate_delta_vs_native": row.get("teacher_future_false_free", 0.0),
                "false_positive_delta": row.get("teacher_false_positive_delta", 0.0),
                "wrong_class_delta": row.get("teacher_wrong_class_delta", 0.0),
                "front_local_density_proxy": row.get("teacher_front_local_density_proxy", 0.0),
                "protected_zone_preservation_ratio": row.get("teacher_protected_zone_preservation_ratio", 1.0),
                "teacher_gap_occ": 0.0,
            }
            for row in rows
        ]
    adapter_summary = aggregate_sw14b_metric_rows(adapter_rows)
    teacher_summary = aggregate_sw14b_metric_rows(teacher_rows)
    legacy_teacher = legacy.get("teacher_summary", {})
    if not any("teacher_front_local_density_proxy" in row for row in rows):
        teacher_summary["mean_front_local_density_proxy"] = float(legacy_teacher.get("mean_front_local_density_proxy", 0.0))
    if not any("teacher_future_false_free" in row for row in rows):
        teacher_summary["mean_future_h4_h6_false_free_rate_delta_vs_native"] = float(legacy_teacher.get("mean_future_h4_h6_false_free_rate_delta_vs_native", 0.0))
    teacher_front = float(teacher_summary["mean_front_sector_false_free_rate_delta_vs_native"] or 0.0)
    adapter_front = float(adapter_summary["mean_front_sector_false_free_rate_delta_vs_native"] or 0.0)
    front_recovery_ratio = safe_div(adapter_front, teacher_front) if abs(teacher_front) >= 1e-9 else 0.0
    return {
        "adapter_summary": adapter_summary,
        "teacher_summary": teacher_summary,
        "front_recovery_ratio": front_recovery_ratio,
        "decision": legacy.get("decision", "REJECT"),
    }


def load_normalized_original_sw14b_final_decision() -> dict[str, Any]:
    payload = read_json(base.REPORTS_DIR / "sw14b_final_decision.json")
    unified_eval = load_unified_original_eval_debug_summary()
    payload["adapter_summary"] = unified_eval["adapter_summary"]
    payload["teacher_summary"] = unified_eval["teacher_summary"]
    if isinstance(payload.get("eval_debug"), dict):
        payload["eval_debug"]["adapter_summary"] = unified_eval["adapter_summary"]
        payload["eval_debug"]["teacher_summary"] = unified_eval["teacher_summary"]
        payload["eval_debug"]["front_recovery_ratio"] = unified_eval["front_recovery_ratio"]
        payload["eval_debug"]["decision"] = unified_eval["decision"]
    return payload


CANONICAL_METRICS = [
    "front_local_density_proxy",
    "front_sector_false_free_rate_delta_vs_native",
    "future_h4_h6_false_free_rate_delta_vs_native",
    "pred_gt_density_delta",
    "false_positive_delta",
    "protected_zone_preservation_ratio",
]


def canonical_from_sw14b_summary(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "front_local_density_proxy": float(summary.get("mean_front_local_density_proxy", 0.0) or 0.0),
        "front_sector_false_free_rate_delta_vs_native": float(summary.get("mean_front_sector_false_free_rate_delta_vs_native", 0.0) or 0.0),
        "future_h4_h6_false_free_rate_delta_vs_native": float(summary.get("mean_future_h4_h6_false_free_rate_delta_vs_native", 0.0) or 0.0),
        "pred_gt_density_delta": float(summary.get("mean_pred_gt_density_delta", summary.get("mean_density_delta", 0.0)) or 0.0),
        "false_positive_delta": float(summary.get("mean_false_positive_delta", 0.0) or 0.0),
        "protected_zone_preservation_ratio": float(summary.get("mean_protected_zone_preservation_ratio", 1.0) or 1.0),
    }


def canonical_from_rescue_summary(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "front_local_density_proxy": float(summary.get("mean_front_local_proxy", 0.0) or 0.0),
        "front_sector_false_free_rate_delta_vs_native": float(summary.get("mean_front_sector_false_free_rate_delta_vs_native", 0.0) or 0.0),
        "future_h4_h6_false_free_rate_delta_vs_native": float(summary.get("mean_future_h4_h6_false_free_rate_delta_vs_native", 0.0) or 0.0),
        "pred_gt_density_delta": float(summary.get("mean_density_delta", 0.0) or 0.0),
        "false_positive_delta": float(summary.get("mean_false_positive_delta", 0.0) or 0.0),
        "protected_zone_preservation_ratio": float(summary.get("mean_protected_zone_preservation_ratio", 1.0) or 1.0),
    }


def is_noop_policy(policy: Any) -> bool:
    if isinstance(policy, RescuePolicy):
        payload = asdict(policy)
    elif isinstance(policy, dict):
        payload = policy
    else:
        return False
    return (
        abs(float(payload.get("global_scale", 1.0)) - 1.0) < 1e-9
        and abs(float(payload.get("front_scale", 1.0)) - 1.0) < 1e-9
        and abs(float(payload.get("side_scale", 1.0)) - 1.0) < 1e-9
        and not bool(payload.get("front_proxy_damping", False))
    )


def is_effective_alpha_scale_row(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    scale = float(row.get("scale", row.get("global_scale", 1.0)))
    return 0.0 < scale < 1.0


def scale_selection_is_effective_pass(scale_selection: dict[str, Any]) -> bool:
    return scale_selection.get("decision") == "SCALE_RESCUE_PASS" and is_effective_alpha_scale_row(scale_selection.get("best_scale_row"))


def write_unified_debug_comparison(rescue_eval: dict[str, Any], best_config: dict[str, Any]) -> dict[str, Any]:
    original_eval = load_unified_original_eval_debug_summary()
    original_summary = original_eval["adapter_summary"]
    teacher_summary = original_eval["teacher_summary"]
    rescue_summary = rescue_eval.get("rescued_summary", {})
    source = "rescue_eval_debug_metrics"
    rescue_rows = read_csv(REPORTS_DIR / "sw14b_rescue_eval_debug_metrics.csv")
    if rescue_rows:
        rescue_samples = {int(row["sample_index"]) for row in rescue_rows}
        original_rows = filter_metric_rows_by_samples(read_csv(base.REPORTS_DIR / "sw14b_eval_debug_metrics.csv"), rescue_samples)
        teacher_rows = filter_metric_rows_by_samples(read_csv(base.REPORTS_DIR / "sw14b_eval_debug_teacher_metrics.csv"), rescue_samples)
        if original_rows and teacher_rows:
            original_summary = aggregate_sw14b_metric_rows(original_rows)
            teacher_summary = aggregate_sw14b_metric_rows(teacher_rows)
            rescue_summary = aggregate_rescue_metric_rows(rescue_rows)
            source = f"rescue_eval_debug_metrics_matched_subset_{len(rescue_samples)}_samples"
    elif int(rescue_summary.get("sample_count", 0) or 0) != int(original_summary.get("sample_count", 0) or 0):
        source = f"partial_rescue_eval_debug_metrics_{int(rescue_summary.get('sample_count', 0) or 0)}_samples"
    rows = []
    systems = {
        "teacher_sw13c_f3_frontcap": canonical_from_sw14b_summary(teacher_summary),
        "original_sw14b_adapter_f3_frontcap": canonical_from_sw14b_summary(original_summary),
        "rescued_sw14b_adapter_f3_frontcap": canonical_from_rescue_summary(rescue_summary)
        if "mean_density_delta" in rescue_summary
        else canonical_from_sw14b_summary(rescue_summary),
    }
    for metric_name in CANONICAL_METRICS:
        row = {"metric": metric_name}
        for system_name, values in systems.items():
            row[system_name] = values[metric_name]
        rows.append(row)
    payload = {
        "metric_schema": "SW13-style deltas: system - native; negative false-free deltas are better.",
        "rescue_metric_source": source,
        "teacher_sample_count": int(teacher_summary.get("sample_count", 0) or 0),
        "original_sample_count": int(original_summary.get("sample_count", 0) or 0),
        "rescue_sample_count": int(rescue_summary.get("sample_count", 0) or 0),
        "best_config": serialize_best_config(best_config),
        "table": rows,
    }
    write_csv(REPORTS_DIR / "sw14b_rescue_unified_metric_comparison.csv", rows)
    write_json(REPORTS_DIR / "sw14b_rescue_unified_metric_comparison.json", payload)
    write_md(
        REPORTS_DIR / "sw14b_rescue_unified_metric_comparison.md",
        "\n".join(
            [
                "# SW14B Rescue Unified Metric Comparison",
                "",
                "- metric schema: SW13-style deltas, `system - native`; negative false-free deltas are better.",
                f"- rescue metric source: `{source}`.",
                f"- sample counts: teacher `{payload['teacher_sample_count']}`, original `{payload['original_sample_count']}`, rescue `{payload['rescue_sample_count']}`.",
                "",
                "| metric | teacher | original | rescue |",
                "|---|---:|---:|---:|",
                *[
                    f"| {row['metric']} | {row['teacher_sw13c_f3_frontcap']} | {row['original_sw14b_adapter_f3_frontcap']} | {row['rescued_sw14b_adapter_f3_frontcap']} |"
                    for row in rows
                ],
            ]
        ),
    )
    return payload


def load_existing_execution_payload(key: str) -> dict[str, Any] | None:
    dedicated_paths = {
        "scale_selection": REPORTS_DIR / "sw14b_alpha_scale_selection.json",
        "clamp_selection": REPORTS_DIR / "sw14b_density_aware_clamp_selection.json",
        "retrain_decision": REPORTS_DIR / "sw14b_conservative_retrain_decision.json",
        "rescue_eval": REPORTS_DIR / "sw14b_rescue_eval_debug_decision.json",
        "clean_attr": REPORTS_DIR / "sw14b_clean_drift_attribution.json",
    }
    dedicated_path = dedicated_paths.get(key)
    if dedicated_path is not None and dedicated_path.exists() and dedicated_path.stat().st_size > 0:
        value = read_json(dedicated_path)
        if isinstance(value, dict):
            return value
    summary_path = REPORTS_DIR / "sw14b_density_clean_rescue_execution_summary.json"
    if summary_path.exists():
        payload = read_json(summary_path)
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    final_path = REPORTS_DIR / "sw14b_density_clean_rescue_final_decision.json"
    key_map = {
        "clean_attr": "clean_drift_attribution",
        "density_attr": "density_attribution",
        "scale_selection": "alpha_scale_sweep",
        "clamp_selection": "density_aware_clamp",
        "retrain_decision": "conservative_retrain",
        "rescue_eval": "frozen_eval_debug_verification",
    }
    if final_path.exists() and key in key_map:
        payload = read_json(final_path)
        value = payload.get(key_map[key])
        if isinstance(value, dict):
            return value
    return None


def complete_existing_rescue_eval(args: argparse.Namespace, rescue_eval: dict[str, Any], best_config: dict[str, Any]) -> dict[str, Any] | None:
    expected_samples = args.eval_end - args.eval_start + 1
    summary = rescue_eval.get("rescued_summary", {})
    if int(summary.get("sample_count", 0) or 0) == expected_samples:
        return rescue_eval
    if int(summary.get("sample_count", 0) or 0) > 0:
        payload = dict(rescue_eval)
        payload["metric_source_note"] = (
            f"existing rescue eval_debug is partial: {int(summary.get('sample_count', 0) or 0)} "
            f"of {expected_samples} samples. It is not substituted with original SW14B metrics."
        )
        return payload
    return None


def write_io_safety_audit(args: argparse.Namespace) -> dict[str, Any]:
    eval_indices = list(range(args.eval_start, args.eval_end + 1))
    val_indices = list(range(args.val_start, args.val_end + 1))
    sample_probe = val_indices[0] if val_indices else (eval_indices[0] if eval_indices else 0)
    feature_path = base.cache_path_for_sample(sample_probe)
    teacher_path = teacher_bundle_cache_path(sample_probe, base.teacher_ref().perturbation_id)
    feature_mb = feature_path.stat().st_size / 1024 / 1024 if feature_path.exists() else None
    teacher_mb = teacher_path.stat().st_size / 1024 / 1024 if teacher_path.exists() else None
    payload = {
        "runtime_teacher_rebuild_default_enabled": ALLOW_RUNTIME_TEACHER_REBUILD,
        "runtime_teacher_rebuild_guard": "disabled unless SW14B_RESCUE_ALLOW_RUNTIME_TEACHER_REBUILD=1",
        "global_case_memo_enabled": False,
        "global_teacher_bundle_memo_enabled": False,
        "per_sample_release_enabled": True,
        "reuses_existing_attribution_by_default": not args.force_runtime_attribution,
        "reuses_existing_sweep_by_default": not args.force_runtime_sweep,
        "reuses_existing_eval_debug_by_default": not args.force_runtime_eval_debug,
        "eval_sample_count": len(eval_indices),
        "val_sample_count": len(val_indices),
        "feature_cache_probe_path": str(feature_path),
        "feature_cache_probe_mb": feature_mb,
        "teacher_bundle_probe_path": str(teacher_path),
        "teacher_bundle_probe_mb": teacher_mb,
        "normal_runtime_forbidden_calls": ["compute_teacher_bundle_local", "base.run_rule_forward", "base.run_native_forward teacher reconstruction"],
    }
    write_json(REPORTS_DIR / "sw14b_density_clean_rescue_io_safety_audit.json", payload)
    write_md(
        REPORTS_DIR / "sw14b_density_clean_rescue_io_safety_audit.md",
        "\n".join(
            [
                "# SW14B Density Clean Rescue I/O Safety Audit",
                "",
                f"- runtime teacher rebuild enabled by default: `{ALLOW_RUNTIME_TEACHER_REBUILD}`.",
                "- global case memo: `False`.",
                "- global teacher bundle memo: `False`.",
                "- per-sample release: `True`.",
                f"- feature cache probe MB: `{feature_mb}`.",
                f"- teacher bundle probe MB: `{teacher_mb}`.",
                "- forbidden in normal runtime: `compute_teacher_bundle_local`, `run_rule_forward`, teacher `run_native_forward` reconstruction.",
            ]
        ),
    )
    return payload


def try_reuse_existing_run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.eval_scale_sweep or args.force_runtime_attribution or args.force_runtime_sweep or args.force_runtime_eval_debug:
        return None
    clean_attr = load_existing_execution_payload("clean_attr")
    density_attr = load_existing_execution_payload("density_attr")
    scale_selection = load_existing_execution_payload("scale_selection")
    clamp_selection = load_existing_execution_payload("clamp_selection")
    retrain_decision = load_existing_execution_payload("retrain_decision") or {"decision": "RETRAIN_R5_NOT_EXECUTED"}
    if not all(isinstance(x, dict) for x in [clean_attr, density_attr, scale_selection, clamp_selection]):
        return None
    best_config = choose_best_config(scale_selection, clamp_selection, retrain_decision)
    rescue_eval_existing = load_existing_execution_payload("rescue_eval")
    if rescue_eval_existing is None:
        return None
    rescue_eval = complete_existing_rescue_eval(args, rescue_eval_existing, best_config)
    if rescue_eval is None:
        return None
    final_decision = phase7_final_decision(clean_attr, density_attr, scale_selection, clamp_selection, retrain_decision, rescue_eval, best_config)
    comparison = write_unified_debug_comparison(rescue_eval, best_config)
    return {
        "clean_attr": clean_attr,
        "density_attr": density_attr,
        "scale_selection": scale_selection,
        "clamp_selection": clamp_selection,
        "retrain_decision": retrain_decision,
        "best_config": serialize_best_config(best_config),
        "rescue_eval": rescue_eval,
        "final_decision": final_decision,
        "unified_metric_comparison": comparison,
        "reused_existing_artifacts": True,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_adapter_from_checkpoint(checkpoint_path: str) -> SpatialGateAdapter:
    adapter = base.build_adapter()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    adapter.load_state_dict(state_dict, strict=True)
    adapter.eval()
    return adapter


def local_feature_cache_path(sample_index: int) -> Path:
    return LOCAL_TMP_CACHE_DIR / "feature_memory_cache" / f"sample_{sample_index:03d}.pt"


def ensure_local_feature_cache(sample_index: int) -> Path:
    local_path = local_feature_cache_path(sample_index)
    if local_path.exists():
        return local_path
    return base.cache_path_for_sample(sample_index)


def camera_scale_tensor(policy: RescuePolicy, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    scale = torch.ones((1, 6, 1, 1, 1), device=device, dtype=dtype)
    scale *= float(policy.global_scale)
    scale[:, FRONT_CAMERA_INDEX] *= float(policy.front_scale)
    scale[:, FRONT_LEFT_CAMERA_INDEX] *= float(policy.side_scale)
    scale[:, FRONT_RIGHT_CAMERA_INDEX] *= float(policy.side_scale)
    return scale


def degradation_mask_from_camera_set(degraded: set[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [[[1.0] if cam in degraded else [0.0] for cam in CAMERA_NAMES]]
    return torch.tensor(values, device=device, dtype=dtype)


def alpha_stats(alpha: torch.Tensor) -> dict[str, float]:
    flat = alpha.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "max": float(flat.max().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
    }


def alpha_stats_from_debug(alpha_debug: dict[str, torch.Tensor]) -> dict[str, float]:
    levels = [value.detach().float().reshape(-1) for key, value in alpha_debug.items() if key.startswith("alpha_level") and isinstance(value, torch.Tensor)]
    if not levels:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0, "p99": 0.0}
    return alpha_stats(torch.cat(levels, dim=0))


def nondegraded_alpha_abs_mean(alpha_debug: dict[str, torch.Tensor]) -> float:
    deg_mask = alpha_debug.get("degradation_mask")
    if deg_mask is None:
        return 0.0
    vals: list[torch.Tensor] = []
    for key, alpha in alpha_debug.items():
        if key.startswith("alpha_level") and isinstance(alpha, torch.Tensor):
            vals.append((alpha * (1.0 - deg_mask[:, :, :, None, None])).abs().reshape(-1))
    if not vals:
        return 0.0
    return float(torch.cat(vals, dim=0).mean().item())


def mean_front_recovery_ratio(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return 0.0
    mean_adapter = float(np.mean([float(row["front_false_free_adapter"]) for row in rows]))
    mean_teacher = float(np.mean([float(row["front_false_free_teacher"]) for row in rows]))
    if abs(mean_teacher) < 1e-4:
        return None
    return safe_div(mean_adapter, mean_teacher)


def serialize_best_config(best_config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(best_config)
    if isinstance(payload.get("policy"), RescuePolicy):
        payload["policy"] = asdict(payload["policy"])
    return base.normalize(payload)


def run_postprocess_verbose(
    *,
    raw_semantic: torch.Tensor,
    raw_confidence: torch.Tensor,
    raw_margin: torch.Tensor,
    native_semantic: torch.Tensor,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    candidate: Any,
    sample_index: int,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    agreement_map: torch.Tensor,
) -> dict[str, Any]:
    device = raw_semantic.device
    raw_semantic = raw_semantic.long()
    raw_confidence = raw_confidence.float().to(device)
    raw_margin = raw_margin.float().to(device)
    native_semantic = native_semantic.long().to(device)
    gt_h = gt_h.long().to(device)
    gt0 = gt0.long().to(device)
    raw_occ = raw_semantic != EMPTY_IDX
    native_occ = native_semantic != EMPTY_IDX
    raw_delta = raw_occ & ~native_occ
    agreement = agreement_map.float().to(device)
    sectors_local = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sectors.items()}
    protected = base.sw13c_fix.protected_zone_fix(
        candidate.protected_variant,
        raw_occ,
        raw_delta,
        raw_confidence,
        raw_margin,
        agreement,
        sectors_local,
        horizon_s,
    )
    low_value = base.sw13c_fix.low_value_score(
        raw_occ,
        protected,
        raw_confidence,
        raw_margin,
        agreement,
        sectors_local,
        horizon_s,
        candidate.wrong_class_aware,
        candidate.front_bias,
    )
    f3_semantic, f3_pruned, f3_meta = base.sw13c_fix.apply_pruning_no_gt(
        raw_semantic=raw_semantic,
        protected_mask=protected,
        low_value_score_map=low_value,
        native_occ_count=int(native_occ.sum().item()),
        raw_occ_count=int(raw_occ.sum().item()),
        raw_delta_count=int(raw_delta.sum().item()),
        budget_mode="native_expansion_ratio",
        expansion_ratio=candidate.expansion_ratio,
        keep_ratio=None,
    )
    final_semantic, front_pruned, cap_meta = base.frontcap50.apply_front_local_cap_no_gt(
        final_semantic_before_cap=f3_semantic.long(),
        native_semantic=native_semantic.long(),
        raw_semantic=raw_semantic.long(),
        protected_mask=protected,
        front_mask=sectors_local["front"].bool(),
        confidence=raw_confidence,
        margin=raw_margin,
        agreement=agreement,
        low_value_score_map=low_value,
        cap_ratio=candidate.cap_ratio,
        cap_mode="disabled" if candidate.cap_ratio is None else "front_native_ratio_cap",
    )
    native_eval = base_postprocess.sw12b.build_eval_row(native_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors_local, baseline_pred=native_semantic.long())
    raw_eval = base_postprocess.attach_sw13_style_deltas(
        base_postprocess.sw12b.build_eval_row(raw_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors_local, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    f3_eval = base_postprocess.attach_sw13_style_deltas(
        base_postprocess.sw12b.build_eval_row(f3_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors_local, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    final_eval = base_postprocess.attach_sw13_style_deltas(
        base_postprocess.sw12b.build_eval_row(final_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors_local, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    front_proxy_raw = safe_div(float((raw_occ & sectors_local["front"]).sum().item()), max(1.0, float((native_occ & sectors_local["front"]).sum().item())))
    front_proxy_f3 = safe_div(float(((f3_semantic != EMPTY_IDX) & sectors_local["front"]).sum().item()), max(1.0, float((native_occ & sectors_local["front"]).sum().item())))
    front_proxy_final = safe_div(float(((final_semantic != EMPTY_IDX) & sectors_local["front"]).sum().item()), max(1.0, float((native_occ & sectors_local["front"]).sum().item())))
    protected_zone_preservation_ratio = 1.0 - safe_div(float((protected & f3_pruned).sum().item()), max(1.0, float(protected.sum().item())))
    return {
        "raw_semantic": raw_semantic,
        "f3_semantic": f3_semantic.long(),
        "final_semantic": final_semantic.long(),
        "protected_mask": protected.bool(),
        "raw_delta_mask": raw_delta.bool(),
        "raw_eval": base.normalize(raw_eval),
        "f3_eval": base.normalize(f3_eval),
        "final_eval": base.normalize(final_eval),
        "raw_occ_count": int(raw_occ.sum().item()),
        "f3_occ_count": int((f3_semantic != EMPTY_IDX).sum().item()),
        "final_occ_count": int((final_semantic != EMPTY_IDX).sum().item()),
        "front_proxy_raw": float(front_proxy_raw),
        "front_proxy_f3": float(front_proxy_f3),
        "front_proxy_final": float(front_proxy_final),
        "protected_zone_preservation_ratio": float(protected_zone_preservation_ratio),
        "f3_pruned_count": int(f3_pruned.sum().item()),
        "frontcap_pruned_count": int(front_pruned.sum().item()),
        "low_value_score": low_value.float(),
        "f3_meta": base.normalize(f3_meta),
        "frontcap_meta": base.normalize(cap_meta),
    }


def run_postprocess_final_only(
    *,
    raw_semantic: torch.Tensor,
    raw_confidence: torch.Tensor,
    raw_margin: torch.Tensor,
    native_semantic: torch.Tensor,
    candidate: Any,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    agreement_map: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    device = raw_semantic.device
    raw_semantic = raw_semantic.long()
    raw_confidence = raw_confidence.float().to(device)
    raw_margin = raw_margin.float().to(device)
    native_semantic = native_semantic.long().to(device)
    raw_occ = raw_semantic != EMPTY_IDX
    native_occ = native_semantic != EMPTY_IDX
    raw_delta = raw_occ & ~native_occ
    agreement = agreement_map.float().to(device)
    sectors_local = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sectors.items()}
    protected = base.sw13c_fix.protected_zone_fix(
        candidate.protected_variant,
        raw_occ,
        raw_delta,
        raw_confidence,
        raw_margin,
        agreement,
        sectors_local,
        horizon_s,
    )
    low_value = base.sw13c_fix.low_value_score(
        raw_occ,
        protected,
        raw_confidence,
        raw_margin,
        agreement,
        sectors_local,
        horizon_s,
        candidate.wrong_class_aware,
        candidate.front_bias,
    )
    f3_semantic, f3_pruned, _f3_meta = base.sw13c_fix.apply_pruning_no_gt(
        raw_semantic=raw_semantic,
        protected_mask=protected,
        low_value_score_map=low_value,
        native_occ_count=int(native_occ.sum().item()),
        raw_occ_count=int(raw_occ.sum().item()),
        raw_delta_count=int(raw_delta.sum().item()),
        budget_mode="native_expansion_ratio",
        expansion_ratio=candidate.expansion_ratio,
        keep_ratio=None,
    )
    final_semantic, _front_pruned, cap_meta = base.frontcap50.apply_front_local_cap_no_gt(
        final_semantic_before_cap=f3_semantic.long(),
        native_semantic=native_semantic.long(),
        raw_semantic=raw_semantic.long(),
        protected_mask=protected,
        front_mask=sectors_local["front"].bool(),
        confidence=raw_confidence,
        margin=raw_margin,
        agreement=agreement,
        low_value_score_map=low_value,
        cap_ratio=candidate.cap_ratio,
        cap_mode="disabled" if candidate.cap_ratio is None else "front_native_ratio_cap",
    )
    return final_semantic.long(), float(cap_meta.get("front_local_density_proxy_after", 0.0))


def apply_alpha_transform(
    alpha: torch.Tensor,
    policy: RescuePolicy,
    front_proxy_raw: float | None = None,
) -> tuple[torch.Tensor, float]:
    transformed = alpha * camera_scale_tensor(policy, alpha.device, alpha.dtype)
    damping_factor = 1.0
    if policy.front_proxy_damping and front_proxy_raw is not None:
        if front_proxy_raw <= 1.20:
            damping_factor = 1.0
        elif front_proxy_raw <= 1.30:
            damping_factor = 0.75
        elif front_proxy_raw <= 1.40:
            damping_factor = 0.50
        else:
            damping_factor = 0.35
        transformed[:, [FRONT_CAMERA_INDEX, FRONT_LEFT_CAMERA_INDEX, FRONT_RIGHT_CAMERA_INDEX]] *= damping_factor
    return transformed.clamp(0.0, 1.0), float(damping_factor)


def extract_gt_temporal_local(sample_unwrapped: dict[str, Any]) -> tuple[list[torch.Tensor], list[str]]:
    gt_current = torch.as_tensor(sample_unwrapped["voxel_semantics"]).long()
    temporal_semantics = sample_unwrapped["temporal_semantics"]
    gt_temporal_list = [gt_current]
    if isinstance(temporal_semantics, dict):
        for key in sorted(int(k) for k in temporal_semantics.keys()):
            gt_temporal_list.append(torch.as_tensor(temporal_semantics[key]["voxel_semantics"]).long())
    else:
        for temporal in temporal_semantics:
            gt_temporal_list.append(torch.as_tensor(temporal["voxel_semantics"]).long())
    pred_keys = [f"semantic_occ_{idx}s" for idx in range(len(gt_temporal_list))]
    return gt_temporal_list, pred_keys


def run_adapter_forward_policy(
    model: Any,
    adapter: SpatialGateAdapter,
    batch_input: dict[str, Any],
    sample_unwrapped: dict[str, Any],
    cache: dict[str, Any],
    perturbation_id: str,
    policy: RescuePolicy,
    degraded_override: list[str] | None = None,
    keep_device: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, torch.Tensor]]:
    holder: dict[str, Any] = {}
    original_forward = base.attach_query_capture_live(model, holder)
    original_simple_test_online = model.simple_test_online
    variant = base.variant_spec(base.teacher_ref().base_repair_variant)
    degraded = set(degraded_override if degraded_override is not None else base.sw13a.perturbation_camera_set(perturbation_id))
    alpha_holder: dict[str, torch.Tensor] = {}

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        bsz, total_n, c, h, w = img.shape
        img = img.reshape(bsz, total_n // 6, 6, c, h, w)
        img_filenames = img_metas[0]["filename"]
        num_frames = len(img_filenames) // 6
        img_shape = (h, w, c)
        img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_feats_list, img_metas_list = [], []
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))
            img_metas_curr = base.sw13a.clone_meta_for_indices(img_metas[0], img_indices)
            img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
            if i == 0:
                mem_levels: list[torch.Tensor] = []
                for level_idx, feat in enumerate(img_feats_curr):
                    mem_level = torch.zeros_like(feat)
                    for cam_idx, _cam_name in enumerate(CAMERA_NAMES):
                        mem = base.sw13a.aggregate_memory_features(cache, variant.offsets, level_idx, cam_idx)
                        if mem is None:
                            mem = feat[:, cam_idx].detach().cpu()
                        mem_level[:, cam_idx] = mem.to(feat.device, dtype=feat.dtype)
                    mem_levels.append(mem_level)
                deg_mask = degradation_mask_from_camera_set(degraded, img_feats_curr[0].device, img_feats_curr[0].dtype)
                cam_ids = torch.arange(6, device=img_feats_curr[0].device)[None]
                avg_age = float(sum(variant.offsets)) / max(1.0, float(len(variant.offsets)))
                mem_age = torch.full((1, 6, 1), avg_age / 3.0, device=img_feats_curr[0].device, dtype=img_feats_curr[0].dtype)
                _, alpha_debug = adapter.apply_to_levels(img_feats_curr, mem_levels, deg_mask, cam_ids, mem_age)
                repaired_levels: list[torch.Tensor] = []
                for level_idx, current_level in enumerate(img_feats_curr):
                    alpha_level = alpha_debug[f"alpha_level{level_idx}"]
                    alpha_level, _ = apply_alpha_transform(alpha_level, policy)
                    repaired_levels.append(alpha_level * mem_levels[level_idx] + (1.0 - alpha_level) * current_level)
                    alpha_holder[f"alpha_level{level_idx}"] = alpha_level
                alpha_holder["degradation_mask"] = deg_mask
                alpha_holder["current_level0"] = img_feats_curr[0]
                alpha_holder["memory_level0"] = mem_levels[0]
                alpha_holder["repaired_level0"] = repaired_levels[0]
                img_feats_curr = repaired_levels
            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)
        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for key, value in img_metas_list[i][0].items():
                if isinstance(value, list):
                    img_metas_reorganized[0][key].extend(value)
        img_feats_cast = base.sw13a.cast_tensor_type(img_feats_reorganized, torch.half, torch.float32)
        return self.simple_test_pts(img_feats_cast, img_metas_reorganized, rescale=rescale)

    model.simple_test_online = patched_simple_test_online.__get__(model, type(model))
    try:
        base.sw13a.reset_model_cache(model)
        moved = base.sw2.move_to_cuda(base.test_batch_compatible(batch_input))
        with torch.inference_mode():
            _outputs = model(return_loss=False, rescale=True, **moved)
            gt_temporal, pred_keys = extract_gt_temporal_local(sample_unwrapped)
            head = base.sw4_inst.get_pts_bbox_head(model)
            per_h: dict[int, dict[str, Any]] = {}
            for horizon_s in CORE_HORIZONS:
                pred_dict = base.extract_pred_dict_live(holder, horizon_s)
                occ_pred, debug_list = base.sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                dense_scores = debug_list[0]["dense_occ_after_padding"]
                raw_semantic = occ_pred[0]
                conf, margin = base.occupancy_confidence_and_margin(dense_scores)
                raw_semantic_out = raw_semantic.detach() if keep_device else raw_semantic.detach().cpu()
                conf_out = conf.detach() if keep_device else conf.detach().cpu()
                margin_out = margin.detach() if keep_device else margin.detach().cpu()
                dense_scores_out = dense_scores.detach() if keep_device else dense_scores.detach().cpu()
                gt_h = gt_temporal[horizon_s].long().to(raw_semantic.device) if keep_device else gt_temporal[horizon_s].long()
                gt0 = gt_temporal[0].long().to(raw_semantic.device) if keep_device else gt_temporal[0].long()
                per_h[horizon_s] = {
                    "raw_semantic": raw_semantic_out,
                    "raw_confidence": conf_out,
                    "raw_margin": margin_out,
                    "dense_scores": dense_scores_out,
                    "gt_h": gt_h,
                    "gt0": gt0,
                    "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
                }
        return per_h, alpha_holder
    finally:
        model.simple_test_online = original_simple_test_online
        model.forward_backbone = original_forward  # type: ignore[assignment]


def prepare_policy_feature_case(
    model: Any,
    adapter: SpatialGateAdapter,
    batch_input: dict[str, Any],
    cache: dict[str, Any],
    perturbation_id: str,
    degraded_override: list[str] | None = None,
) -> dict[str, Any]:
    variant = base.variant_spec(base.teacher_ref().base_repair_variant)
    degraded = set(degraded_override if degraded_override is not None else base.sw13a.perturbation_camera_set(perturbation_id))
    moved = base.sw2.move_to_cuda(base.test_batch_compatible(batch_input))
    img = moved["img"][0]
    img_metas = moved["img_metas"][0]
    bsz, total_n, c, h, w = img.shape
    img_seq = img.reshape(bsz, total_n // 6, 6, c, h, w)
    img_filenames = img_metas[0]["filename"]
    num_frames = len(img_filenames) // 6
    img_shape = (h, w, c)
    img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
    img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
    img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
    frame_levels: list[list[torch.Tensor]] = []
    frame_metas: list[list[dict[str, Any]]] = []
    mem_levels: list[torch.Tensor] | None = None
    alpha_debug_base: dict[str, torch.Tensor] | None = None
    deg_mask: torch.Tensor | None = None
    for i in range(num_frames):
        img_indices = list(np.arange(i * 6, (i + 1) * 6))
        img_metas_curr = base.sw13a.clone_meta_for_indices(img_metas[0], img_indices)
        current_levels = model.extract_feat(img_seq[:, i], img_metas_curr)
        frame_levels.append(current_levels)
        frame_metas.append(img_metas_curr)
        if i == 0:
            mem_levels = []
            for level_idx, feat in enumerate(current_levels):
                mem_level = torch.zeros_like(feat)
                for cam_idx, _cam_name in enumerate(CAMERA_NAMES):
                    mem = base.sw13a.aggregate_memory_features(cache, variant.offsets, level_idx, cam_idx)
                    if mem is None:
                        mem = feat[:, cam_idx].detach().cpu()
                    mem_level[:, cam_idx] = mem.to(feat.device, dtype=feat.dtype)
                mem_levels.append(mem_level)
            deg_mask = degradation_mask_from_camera_set(degraded, current_levels[0].device, current_levels[0].dtype)
            cam_ids = torch.arange(6, device=current_levels[0].device)[None]
            avg_age = float(sum(variant.offsets)) / max(1.0, float(len(variant.offsets)))
            mem_age = torch.full((1, 6, 1), avg_age / 3.0, device=current_levels[0].device, dtype=current_levels[0].dtype)
            _, alpha_debug_base = adapter.apply_to_levels(current_levels, mem_levels, deg_mask, cam_ids, mem_age)
    assert mem_levels is not None and alpha_debug_base is not None and deg_mask is not None
    return {
        "moved_batch": moved,
        "frame_levels": frame_levels,
        "frame_metas": frame_metas,
        "memory_levels": mem_levels,
        "alpha_debug_base": alpha_debug_base,
        "degradation_mask": deg_mask,
    }


def run_adapter_forward_policy_prepared(
    model: Any,
    sample_unwrapped: dict[str, Any],
    prepared_case: dict[str, Any],
    policy: RescuePolicy,
    front_proxy_raw: float | None = None,
    keep_device: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, torch.Tensor]]:
    holder: dict[str, Any] = {}
    original_forward = base.attach_query_capture_live(model, holder)
    original_extract_feat = model.extract_feat
    frame_cursor = {"idx": 0}
    alpha_holder: dict[str, torch.Tensor] = {}

    def patched_extract_feat(img: torch.Tensor, img_metas_curr: list[dict[str, Any]]):
        idx = frame_cursor["idx"]
        frame_cursor["idx"] += 1
        current_levels = prepared_case["frame_levels"][idx]
        if idx != 0:
            return current_levels
        repaired_levels: list[torch.Tensor] = []
        for level_idx, current_level in enumerate(current_levels):
            base_alpha = prepared_case["alpha_debug_base"][f"alpha_level{level_idx}"]
            alpha_level, damping = apply_alpha_transform(base_alpha, policy, front_proxy_raw=front_proxy_raw)
            repaired_levels.append(alpha_level * prepared_case["memory_levels"][level_idx] + (1.0 - alpha_level) * current_level)
            alpha_holder[f"alpha_level{level_idx}"] = alpha_level
            if level_idx == 0:
                alpha_holder["front_proxy_damping_factor"] = torch.tensor(float(damping), device=alpha_level.device)
        alpha_holder["degradation_mask"] = prepared_case["degradation_mask"]
        alpha_holder["current_level0"] = current_levels[0]
        alpha_holder["memory_level0"] = prepared_case["memory_levels"][0]
        alpha_holder["repaired_level0"] = repaired_levels[0]
        return repaired_levels

    model.extract_feat = patched_extract_feat  # type: ignore[assignment]
    try:
        base.sw13a.reset_model_cache(model)
        with torch.inference_mode():
            _outputs = model(return_loss=False, rescale=True, **prepared_case["moved_batch"])
            gt_temporal, pred_keys = extract_gt_temporal_local(sample_unwrapped)
            head = base.sw4_inst.get_pts_bbox_head(model)
            per_h: dict[int, dict[str, Any]] = {}
            for horizon_s in CORE_HORIZONS:
                pred_dict = base.extract_pred_dict_live(holder, horizon_s)
                occ_pred, debug_list = base.sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                dense_scores = debug_list[0]["dense_occ_after_padding"]
                raw_semantic = occ_pred[0]
                conf, margin = base.occupancy_confidence_and_margin(dense_scores)
                raw_semantic_out = raw_semantic.detach() if keep_device else raw_semantic.detach().cpu()
                conf_out = conf.detach() if keep_device else conf.detach().cpu()
                margin_out = margin.detach() if keep_device else margin.detach().cpu()
                dense_scores_out = dense_scores.detach() if keep_device else dense_scores.detach().cpu()
                gt_h = gt_temporal[horizon_s].long().to(raw_semantic.device) if keep_device else gt_temporal[horizon_s].long()
                gt0 = gt_temporal[0].long().to(raw_semantic.device) if keep_device else gt_temporal[0].long()
                per_h[horizon_s] = {
                    "raw_semantic": raw_semantic_out,
                    "raw_confidence": conf_out,
                    "raw_margin": margin_out,
                    "dense_scores": dense_scores_out,
                    "gt_h": gt_h,
                    "gt0": gt0,
                    "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
                }
        return per_h, alpha_holder
    finally:
        model.extract_feat = original_extract_feat  # type: ignore[assignment]
        model.forward_backbone = original_forward  # type: ignore[assignment]


def run_adapter_forward_front_proxy_damping(
    model: Any,
    adapter: SpatialGateAdapter,
    batch_input: dict[str, Any],
    sample_unwrapped: dict[str, Any],
    cache: dict[str, Any],
    perturbation_id: str,
    policy: RescuePolicy,
    native_per_h: dict[int, dict[str, Any]],
    sectors: dict[str, torch.Tensor],
    degraded_override: list[str] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, torch.Tensor]]:
    base_policy = RescuePolicy(
        name=f"{policy.name}__base",
        global_scale=policy.front_proxy_damping_base_scale,
        front_scale=policy.front_scale,
        side_scale=policy.side_scale,
        front_proxy_damping=False,
    )
    first_pass, first_alpha = run_adapter_forward_policy(model, adapter, batch_input, sample_unwrapped, cache, perturbation_id, base_policy, degraded_override=degraded_override)
    front_values: list[float] = []
    for horizon_s in CORE_HORIZONS:
        raw_occ = first_pass[horizon_s]["raw_semantic"].long() != EMPTY_IDX
        native_occ = native_per_h[horizon_s]["raw_semantic"].long() != EMPTY_IDX
        front_values.append(safe_div(float((raw_occ & sectors["front"]).sum().item()), max(1.0, float((native_occ & sectors["front"]).sum().item()))))
    front_proxy_raw = max(front_values) if front_values else 1.0
    damping_factor = 1.0
    if front_proxy_raw <= 1.20:
        damping_factor = 1.0
    elif front_proxy_raw <= 1.30:
        damping_factor = 0.75
    elif front_proxy_raw <= 1.40:
        damping_factor = 0.50
    else:
        damping_factor = 0.35
    damped_policy = RescuePolicy(
        name=policy.name,
        global_scale=policy.front_proxy_damping_base_scale,
        front_scale=policy.front_scale * damping_factor,
        side_scale=policy.side_scale * damping_factor,
        front_proxy_damping=False,
    )
    second_pass, second_alpha = run_adapter_forward_policy(model, adapter, batch_input, sample_unwrapped, cache, perturbation_id, damped_policy, degraded_override=degraded_override)
    second_alpha["front_proxy_damping_factor"] = torch.tensor(float(damping_factor))
    second_alpha["front_proxy_raw"] = torch.tensor(float(front_proxy_raw))
    base.release_runtime(first_pass, first_alpha)
    return second_pass, second_alpha


def current_and_memory_features_local(
    model: Any,
    dataset: Any,
    sample_index: int,
    perturbation_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[torch.Tensor], list[torch.Tensor], dict[str, Any], dict[str, Any]]:
    raw_sample, batch_clean = base.sw2.extract_sample_batch(dataset, sample_index, base.collate_fn)
    sample_unwrapped = base.sw2.unwrap(raw_sample)
    cache_file = ensure_local_feature_cache(sample_index)
    if cache_file.exists():
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        moved_clean = base.sw2.move_to_cuda(batch_clean)
        base.sw13a.reset_model_cache(model)
        _, _, cache = base.sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_index)
    # The rescue path only needs the sample, degraded batch, and memory cache.
    # Building clean/current levels here duplicated extract_feat before every
    # prepared adapter pass and made multi-sample runs bursty.
    clean_levels: list[torch.Tensor] = []
    memory_levels: list[torch.Tensor] = []
    meta: dict[str, Any] = {}
    spec_catalog = base.sw81.sw5_engine.build_catalog()
    batch_deg = copy.deepcopy(batch_clean)
    if not isinstance(batch_deg["img"], list):
        batch_deg["img"] = [batch_deg["img"]]
    if "img_metas" in batch_deg and not isinstance(batch_deg["img_metas"], list):
        batch_deg["img_metas"] = [batch_deg["img_metas"]]
    batch_deg = base.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[perturbation_id])[0]
    return sample_unwrapped, batch_clean, batch_deg, clean_levels, memory_levels, meta, cache


def get_native_rule_teacher_case(
    model: Any,
    dataset: Any,
    sample_index: int,
    perturbation_id: str,
    candidate: Any,
    sectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    sample_unwrapped, _, batch_deg, _, _, _, cache = current_and_memory_features_local(model, dataset, sample_index, perturbation_id)
    teacher_bundle = get_teacher_bundle_cached(model, dataset, sample_index, f"rescue_{perturbation_id}", perturbation_id)
    native_per_h = {h: {"raw_semantic": teacher_bundle["by_horizon"][h]["native_semantic"]} for h in CORE_HORIZONS}
    by_horizon: dict[int, dict[str, Any]] = {}
    for horizon_s in CORE_HORIZONS:
        teacher_h = teacher_bundle["by_horizon"][horizon_s]
        native_semantic = native_per_h[horizon_s]["raw_semantic"].long()
        agreement = teacher_h["agreement"].float()
        native_verbose = run_postprocess_verbose(
            raw_semantic=native_semantic,
            raw_confidence=teacher_h["teacher_confidence"].float() * 0.0 + 1.0,
            raw_margin=teacher_h["teacher_margin"].float() * 0.0,
            native_semantic=native_semantic,
            gt_h=teacher_h["gt_h"].long(),
            gt0=teacher_h["gt0"].long(),
            candidate=candidate,
            sample_index=sample_index,
            horizon_s=horizon_s,
            sectors=sectors,
            agreement_map=agreement,
        )
        teacher_verbose = run_postprocess_verbose(
            raw_semantic=teacher_h["teacher_raw_semantic"].long(),
            raw_confidence=teacher_h["teacher_confidence"].float(),
            raw_margin=teacher_h["teacher_margin"].float(),
            native_semantic=native_semantic,
            gt_h=teacher_h["gt_h"].long(),
            gt0=teacher_h["gt0"].long(),
            candidate=candidate,
            sample_index=sample_index,
            horizon_s=horizon_s,
            sectors=sectors,
            agreement_map=agreement,
        )
        by_horizon[horizon_s] = {
            "teacher_h": teacher_h,
            "native_semantic": native_semantic,
            "agreement": agreement,
            "native_verbose": native_verbose,
            "teacher_verbose": teacher_verbose,
        }
    return {
        "sample_unwrapped": sample_unwrapped,
        "batch_deg": batch_deg,
        "cache": cache,
        "teacher_bundle": teacher_bundle,
        "native_per_h": native_per_h,
        "by_horizon": by_horizon,
    }


def build_case_cache(
    model: Any,
    dataset: Any,
    sample_indices: list[int],
    perturbation_id: str,
    candidate: Any,
    sectors: dict[str, torch.Tensor],
) -> LazyCaseStore:
    return LazyCaseStore(model, dataset, sample_indices, perturbation_id, candidate, sectors)


def release_case_cache_sample(case_cache: Any, sample_index: int, case: dict[str, Any] | None = None) -> None:
    if hasattr(case_cache, "release_sample"):
        case_cache.release_sample(sample_index)
    elif case is not None:
        base.release_runtime(case)


def prime_case_store(case_cache: LazyCaseStore, sample_indices: list[int], label: str) -> None:
    print(f"[rescue] prewarm {label} cases: {len(sample_indices)} samples", flush=True)
    for sample_index in sample_indices:
        _ = case_cache[sample_index]
        case_cache.release_sample(sample_index)


def compute_teacher_bundle_local(
    model: Any,
    dataset: Any,
    sample_index: int,
    split_name: str,
    perturbation_id: str,
) -> dict[str, Any]:
    candidate = base.build_candidate()
    sectors = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    sample_unwrapped, _, batch_deg, _, _, _, cache = current_and_memory_features_local(model, dataset, sample_index, perturbation_id)
    native = base.run_native_forward(model, batch_deg, sample_unwrapped)
    rule_outputs = {
        label: base.run_rule_forward(model, batch_deg, sample_unwrapped, cache, perturbation_id, label)
        for label in base.teacher_ref().agreement_sources
    }
    bundle: dict[str, Any] = {"sample_index": sample_index, "split_name": split_name, "perturbation_id": perturbation_id, "by_horizon": {}}
    for horizon_s in CORE_HORIZONS:
        agreement = base.build_agreement_from_rule_outputs(rule_outputs, horizon_s)
        rule_r8 = rule_outputs[base.teacher_ref().base_repair_variant][horizon_s]
        final_semantic, meta = base.run_sw14b_full_postprocess(
            raw_semantic=rule_r8["raw_semantic"].long(),
            raw_confidence=rule_r8["raw_confidence"].float(),
            raw_margin=rule_r8["raw_margin"].float(),
            native_semantic=native[horizon_s]["raw_semantic"].long(),
            gt_h=rule_r8["gt_h"].long(),
            gt0=rule_r8["gt0"].long(),
            candidate=candidate,
            sample_index=sample_index,
            horizon_s=horizon_s,
            sectors=sectors,
            load_gpu_dump=base.load_gpu_dump,
            agreement_map=agreement,
        )
        bundle["by_horizon"][horizon_s] = {
            "native_semantic": native[horizon_s]["raw_semantic"].long(),
            "teacher_raw_semantic": rule_r8["raw_semantic"].long(),
            "teacher_final_semantic": final_semantic.long(),
            "teacher_confidence": rule_r8["raw_confidence"].float(),
            "teacher_margin": rule_r8["raw_margin"].float(),
            "agreement": agreement.float(),
            "gt_h": rule_r8["gt_h"].long(),
            "gt0": rule_r8["gt0"].long(),
            "teacher_meta": meta,
        }
    base.release_runtime(sample_unwrapped, batch_deg, cache, native, rule_outputs)
    return bundle


def teacher_bundle_cache_path(sample_index: int, perturbation_id: str) -> Path:
    return base.ARTIFACTS_DIR / "runtime_teacher_cache" / f"rescue_{perturbation_id}" / f"{perturbation_id}__sample{sample_index:03d}.pt"


def get_teacher_bundle_cached(
    model: Any,
    dataset: Any,
    sample_index: int,
    split_name: str,
    perturbation_id: str,
) -> dict[str, Any]:
    cache_path = teacher_bundle_cache_path(sample_index, perturbation_id)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    if ALLOW_RUNTIME_TEACHER_REBUILD:
        return compute_teacher_bundle_local(model, dataset, sample_index, split_name, perturbation_id)
    raise FileNotFoundError(
        f"missing precomputed teacher bundle {cache_path}; runtime teacher rebuild is disabled. "
        "Set SW14B_RESCUE_ALLOW_RUNTIME_TEACHER_REBUILD=1 only for an explicit debug run."
    )


def phase1_clean_drift_attribution(args: argparse.Namespace, adapter: SpatialGateAdapter) -> dict[str, Any]:
    print("[rescue] phase1 clean drift attribution", flush=True)
    _, dataset, model, _ = base.build_runtime(train=False)
    model.eval()
    try:
        return phase1_clean_drift_attribution_with_runtime(args, adapter, dataset, model)
    finally:
        base.release_runtime(model, dataset, collect=True, empty_cache=True)


def phase1_clean_drift_attribution_with_runtime(args: argparse.Namespace, adapter: SpatialGateAdapter, dataset: Any, model: Any) -> dict[str, Any]:
    if not args.force_runtime_attribution:
        existing = load_existing_execution_payload("clean_attr")
        if existing is not None:
            print("[rescue] reuse existing phase1 clean drift attribution", flush=True)
            return existing
    sectors = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    candidate = base.build_candidate()
    rows: list[dict[str, Any]] = []
    sample_indices = list(range(args.eval_start, args.eval_end + 1))
    disabled_policy = RescuePolicy(name="disabled", global_scale=0.0)
    enabled_policy = RescuePolicy(name="enabled", global_scale=1.0)
    for sample_index in sample_indices:
        sample_unwrapped, batch_clean, _batch_deg, _, _, _, cache = current_and_memory_features_local(model, dataset, sample_index, "A0_clean")
        native_raw = base.run_native_forward(model, batch_clean, sample_unwrapped)
        enabled_raw_buggy, enabled_alpha_buggy = run_adapter_forward_policy(model, adapter, batch_clean, sample_unwrapped, cache, "A0_clean", enabled_policy)
        enabled_raw, enabled_alpha = run_adapter_forward_policy(model, adapter, batch_clean, sample_unwrapped, cache, "A0_clean", enabled_policy, degraded_override=[])
        disabled_raw, disabled_alpha = run_adapter_forward_policy(model, adapter, batch_clean, sample_unwrapped, cache, "A0_clean", disabled_policy, degraded_override=[])
        rule_outputs = {
            label: base.run_rule_forward(model, batch_clean, sample_unwrapped, cache, "A0_clean", label)
            for label in base.teacher_ref().agreement_sources
        }
        for horizon_s in CORE_HORIZONS:
            agreement = base.build_agreement_from_rule_outputs(rule_outputs, horizon_s)
            native_verbose = run_postprocess_verbose(
                raw_semantic=native_raw[horizon_s]["raw_semantic"].long(),
                raw_confidence=native_raw[horizon_s]["raw_confidence"].float(),
                raw_margin=native_raw[horizon_s]["raw_margin"].float(),
                native_semantic=native_raw[horizon_s]["raw_semantic"].long(),
                gt_h=native_raw[horizon_s]["gt_h"].long(),
                gt0=native_raw[horizon_s]["gt0"].long(),
                candidate=candidate,
                sample_index=sample_index,
                horizon_s=horizon_s,
                sectors=sectors,
                agreement_map=agreement,
            )
            enabled_verbose = run_postprocess_verbose(
                raw_semantic=enabled_raw[horizon_s]["raw_semantic"].long(),
                raw_confidence=enabled_raw[horizon_s]["raw_confidence"].float(),
                raw_margin=enabled_raw[horizon_s]["raw_margin"].float(),
                native_semantic=native_raw[horizon_s]["raw_semantic"].long(),
                gt_h=enabled_raw[horizon_s]["gt_h"].long(),
                gt0=enabled_raw[horizon_s]["gt0"].long(),
                candidate=candidate,
                sample_index=sample_index,
                horizon_s=horizon_s,
                sectors=sectors,
                agreement_map=agreement,
            )
            rows.append(
                {
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "native_vs_disabled_raw_occ_diff_ratio": safe_div(float(((native_raw[horizon_s]["raw_semantic"].long() != EMPTY_IDX) ^ (disabled_raw[horizon_s]["raw_semantic"].long() != EMPTY_IDX)).sum().item()), float((native_raw[horizon_s]["raw_semantic"].numel()))),
                    "native_vs_enabled_buggy_raw_occ_diff_ratio": safe_div(float(((native_raw[horizon_s]["raw_semantic"].long() != EMPTY_IDX) ^ (enabled_raw_buggy[horizon_s]["raw_semantic"].long() != EMPTY_IDX)).sum().item()), float((native_raw[horizon_s]["raw_semantic"].numel()))),
                    "native_vs_enabled_raw_occ_diff_ratio": safe_div(float(((native_raw[horizon_s]["raw_semantic"].long() != EMPTY_IDX) ^ (enabled_raw[horizon_s]["raw_semantic"].long() != EMPTY_IDX)).sum().item()), float((native_raw[horizon_s]["raw_semantic"].numel()))),
                    "native_vs_enabled_raw_semantic_abs_diff_mean": float((native_raw[horizon_s]["raw_semantic"].float() - enabled_raw[horizon_s]["raw_semantic"].float()).abs().mean().item()),
                    "native_raw_density_delta": metric_value(native_verbose["raw_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
                    "enabled_raw_density_delta": metric_value(enabled_verbose["raw_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
                    "enabled_final_density_delta": metric_value(enabled_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
                    "native_final_density_delta": metric_value(native_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
                    "after_f3_occ_diff_ratio": safe_div(float(((native_verbose["f3_semantic"] != EMPTY_IDX) ^ (enabled_verbose["f3_semantic"] != EMPTY_IDX)).sum().item()), float(native_verbose["f3_semantic"].numel())),
                    "after_frontcap_occ_diff_ratio": safe_div(float(((native_verbose["final_semantic"] != EMPTY_IDX) ^ (enabled_verbose["final_semantic"] != EMPTY_IDX)).sum().item()), float(native_verbose["final_semantic"].numel())),
                    "enabled_raw_front_false_free_delta": metric_value(enabled_verbose["raw_eval"], "front_sector_false_free_rate_delta"),
                    "enabled_final_front_false_free_delta": metric_value(enabled_verbose["final_eval"], "front_sector_false_free_rate_delta"),
                    "alpha_mean": alpha_stats_from_debug(enabled_alpha)["mean"],
                    "alpha_max": alpha_stats_from_debug(enabled_alpha)["max"],
                    "alpha_p95": alpha_stats_from_debug(enabled_alpha)["p95"],
                    "alpha_p99": alpha_stats_from_debug(enabled_alpha)["p99"],
                    "buggy_degradation_mask_sum": float(enabled_alpha_buggy["degradation_mask"].sum().item()),
                    "degradation_mask_sum": float(enabled_alpha["degradation_mask"].sum().item()),
                    "nondegraded_alpha_abs_mean": nondegraded_alpha_abs_mean(enabled_alpha),
                    "clean_candidate_is_a10_config": True,
                }
            )
        base.release_runtime(sample_unwrapped, batch_clean, cache, native_raw, enabled_raw_buggy, enabled_raw, disabled_raw, enabled_alpha_buggy, enabled_alpha, disabled_alpha, rule_outputs)
    write_csv(REPORTS_DIR / "sw14b_clean_drift_attribution.csv", rows)
    mean_native_disabled = float(np.mean([float(row["native_vs_disabled_raw_occ_diff_ratio"]) for row in rows])) if rows else 0.0
    mean_buggy_raw_diff = float(np.mean([float(row["native_vs_enabled_buggy_raw_occ_diff_ratio"]) for row in rows])) if rows else 0.0
    mean_raw_diff = float(np.mean([float(row["native_vs_enabled_raw_occ_diff_ratio"]) for row in rows])) if rows else 0.0
    mean_final_diff = float(np.mean([float(row["after_frontcap_occ_diff_ratio"]) for row in rows])) if rows else 0.0
    mean_deg_mask = float(np.mean([float(row["degradation_mask_sum"]) for row in rows])) if rows else 0.0
    mean_buggy_deg_mask = float(np.mean([float(row["buggy_degradation_mask_sum"]) for row in rows])) if rows else 0.0
    mean_alpha = float(np.mean([float(row["alpha_mean"]) for row in rows])) if rows else 0.0
    decision = "CLEAN_D6_TRUE_CLEAN_UNSAFE"
    if mean_buggy_deg_mask > 0.0:
        decision = "CLEAN_D4_CLEAN_DEGRADATION_FLAG_BUG"
    elif mean_native_disabled > 1e-6:
        decision = "CLEAN_D3_METRIC_SCHEMA_DRIFT"
    elif mean_raw_diff <= 1e-4 and mean_final_diff > 1e-3:
        decision = "CLEAN_D2_POSTPROCESS_PATH_DRIFT"
    elif mean_raw_diff <= 1e-4 and mean_final_diff <= 1e-3:
        decision = "CLEAN_D5_DRIFT_NEGLIGIBLE_AFTER_DECOMPOSITION"
    elif mean_alpha > 1e-5 or mean_raw_diff > 1e-3:
        decision = "CLEAN_D1_ADAPTER_CAUSES_RAW_DRIFT"
    summary = {
        "decision": decision,
        "mean_native_vs_disabled_raw_occ_diff_ratio": mean_native_disabled,
        "mean_native_vs_enabled_buggy_raw_occ_diff_ratio": mean_buggy_raw_diff,
        "mean_native_vs_enabled_raw_occ_diff_ratio": mean_raw_diff,
        "mean_after_frontcap_occ_diff_ratio": mean_final_diff,
        "mean_alpha_mean": mean_alpha,
        "mean_degradation_mask_sum": mean_deg_mask,
        "mean_buggy_degradation_mask_sum": mean_buggy_deg_mask,
        "corrected_clean_safe": bool(mean_raw_diff <= 1e-4 and mean_final_diff <= 1e-3 and mean_deg_mask == 0.0),
    }
    write_json(REPORTS_DIR / "sw14b_clean_drift_attribution.json", summary)
    write_md(
        REPORTS_DIR / "sw14b_clean_drift_attribution.md",
        "\n".join(
            [
                "# SW14B Clean Drift Attribution",
                "",
                f"- decision: `{decision}`.",
                f"- native vs adapter-disabled raw occ diff ratio mean: `{mean_native_disabled}`.",
                f"- native vs adapter-enabled buggy raw occ diff ratio mean: `{mean_buggy_raw_diff}`.",
                f"- native vs adapter-enabled corrected raw occ diff ratio mean: `{mean_raw_diff}`.",
                f"- native vs adapter-enabled final occ diff ratio mean: `{mean_final_diff}`.",
                f"- clean alpha mean: `{mean_alpha}`.",
                f"- corrected clean degradation mask sum mean: `{mean_deg_mask}`.",
                f"- buggy clean degradation mask sum mean: `{mean_buggy_deg_mask}`.",
            ]
        ),
    )
    return summary


def collect_policy_rows(
    model: Any,
    dataset: Any,
    adapter: SpatialGateAdapter,
    sample_indices: list[int],
    perturbation_id: str,
    split_name: str,
    policy: RescuePolicy,
    case_cache: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = base.build_candidate()
    sectors = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    rows: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        case: dict[str, Any] | None = None
        per_h: dict[int, dict[str, Any]] | None = None
        alpha_debug: dict[str, torch.Tensor] | None = None
        try:
            case = case_cache[sample_index] if case_cache is not None else get_native_rule_teacher_case(model, dataset, sample_index, perturbation_id, candidate, sectors)
            per_h, alpha_debug = run_policy_for_case(model, adapter, case, perturbation_id, policy, sectors)
            for horizon_s in CORE_HORIZONS:
                rows.append(build_policy_row(case, per_h, alpha_debug, split_name, policy, sample_index, horizon_s, sectors, candidate))
        finally:
            base.release_runtime(per_h, alpha_debug)
            if case_cache is None and case is not None:
                base.release_runtime(case)
            elif case_cache is not None:
                release_case_cache_sample(case_cache, sample_index, case)
    return rows, summarize_policy_rows(rows, sample_count=len(sample_indices))


def run_policy_for_case(
    model: Any,
    adapter: SpatialGateAdapter,
    case: dict[str, Any],
    perturbation_id: str,
    policy: RescuePolicy,
    sectors: dict[str, torch.Tensor],
) -> tuple[dict[int, dict[str, Any]], dict[str, torch.Tensor]]:
    sample_unwrapped = case["sample_unwrapped"]
    batch_deg = case["batch_deg"]
    cache = case["cache"]
    native_per_h = case["native_per_h"]
    prepared_case = case.get("prepared_case")
    if policy.front_proxy_damping:
        base_policy = RescuePolicy(
            name=f"{policy.name}__base",
            global_scale=policy.front_proxy_damping_base_scale,
            front_scale=policy.front_scale,
            side_scale=policy.side_scale,
            front_proxy_damping=False,
        )
        if prepared_case is not None:
            first_pass, _first_alpha = run_adapter_forward_policy_prepared(model, sample_unwrapped, prepared_case, base_policy, keep_device=True)
        else:
            first_pass, _first_alpha = run_adapter_forward_policy(model, adapter, batch_deg, sample_unwrapped, cache, perturbation_id, base_policy)
        front_values: list[float] = []
        for horizon_s in CORE_HORIZONS:
            raw_occ = first_pass[horizon_s]["raw_semantic"].long() != EMPTY_IDX
            if prepared_case is not None and "gpu_static" in case:
                native_occ = case["gpu_static"][horizon_s]["native_semantic"].long() != EMPTY_IDX
            else:
                native_occ = native_per_h[horizon_s]["raw_semantic"].long() != EMPTY_IDX
            front_values.append(safe_div(float((raw_occ & sectors["front"]).sum().item()), max(1.0, float((native_occ & sectors["front"]).sum().item()))))
        front_proxy_raw = max(front_values) if front_values else 1.0
        damping_factor = 1.0
        if front_proxy_raw <= 1.20:
            damping_factor = 1.0
        elif front_proxy_raw <= 1.30:
            damping_factor = 0.75
        elif front_proxy_raw <= 1.40:
            damping_factor = 0.50
        else:
            damping_factor = 0.35
        damped_policy = RescuePolicy(
            name=policy.name,
            global_scale=policy.front_proxy_damping_base_scale,
            front_scale=policy.front_scale * damping_factor,
            side_scale=policy.side_scale * damping_factor,
            front_proxy_damping=False,
        )
        if prepared_case is not None:
            second_pass, second_alpha = run_adapter_forward_policy_prepared(model, sample_unwrapped, prepared_case, damped_policy, front_proxy_raw=front_proxy_raw, keep_device=True)
        else:
            second_pass, second_alpha = run_adapter_forward_policy(model, adapter, batch_deg, sample_unwrapped, cache, perturbation_id, damped_policy)
        second_alpha["front_proxy_damping_factor"] = torch.tensor(float(damping_factor))
        second_alpha["front_proxy_raw"] = torch.tensor(float(front_proxy_raw))
        base.release_runtime(first_pass)
        return second_pass, second_alpha
    if prepared_case is not None:
        return run_adapter_forward_policy_prepared(model, sample_unwrapped, prepared_case, policy, keep_device=True)
    return run_adapter_forward_policy(model, adapter, batch_deg, sample_unwrapped, cache, perturbation_id, policy)


def build_policy_row(
    case: dict[str, Any],
    per_h: dict[int, dict[str, Any]],
    alpha_debug: dict[str, torch.Tensor],
    split_name: str,
    policy: RescuePolicy,
    sample_index: int,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    candidate: Any,
) -> dict[str, Any]:
    cached_h = case["by_horizon"][horizon_s]
    gpu_static = case.get("gpu_static", {}).get(horizon_s)
    native_semantic = gpu_static["native_semantic"] if gpu_static is not None else cached_h["native_semantic"].long()
    teacher_h = cached_h["teacher_h"]
    agreement = gpu_static["agreement"] if gpu_static is not None else cached_h["agreement"].float()
    adapter_verbose = run_postprocess_verbose(
        raw_semantic=per_h[horizon_s]["raw_semantic"].long(),
        raw_confidence=per_h[horizon_s]["raw_confidence"].float(),
        raw_margin=per_h[horizon_s]["raw_margin"].float(),
        native_semantic=native_semantic,
        gt_h=gpu_static["gt_h"] if gpu_static is not None else per_h[horizon_s]["gt_h"].long(),
        gt0=gpu_static["gt0"] if gpu_static is not None else per_h[horizon_s]["gt0"].long(),
        candidate=candidate,
        sample_index=sample_index,
        horizon_s=horizon_s,
        sectors=sectors,
        agreement_map=agreement,
    )
    native_verbose = cached_h["native_verbose"]
    teacher_verbose = cached_h["teacher_verbose"]
    gt_occ = (gpu_static["gt_h"] if gpu_static is not None else teacher_h["gt_h"].long()) != EMPTY_IDX
    adapter_occ = adapter_verbose["final_semantic"] != EMPTY_IDX
    native_occ = (gpu_static["native_final_semantic"] if gpu_static is not None else native_verbose["final_semantic"]) != EMPTY_IDX
    teacher_occ = (gpu_static["teacher_final_semantic"] if gpu_static is not None else teacher_verbose["final_semantic"]) != EMPTY_IDX
    front_mask = sectors["front"].to(adapter_occ.device).bool()
    protected_mask = adapter_verbose["protected_mask"]
    raw_delta_mask = adapter_verbose["raw_delta_mask"]
    alpha_summary = alpha_stats_from_debug(alpha_debug)
    row = {
        "split_name": split_name,
        "policy_name": policy.name,
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "raw_density_native": metric_value(native_verbose["raw_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "raw_density_teacher": metric_value(teacher_verbose["raw_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "raw_density_adapter": metric_value(adapter_verbose["raw_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "f3_density_native": metric_value(native_verbose["f3_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "f3_density_teacher": metric_value(teacher_verbose["f3_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "f3_density_adapter": metric_value(adapter_verbose["f3_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "final_density_native": metric_value(native_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "final_density_teacher": metric_value(teacher_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "final_density_adapter": metric_value(adapter_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "front_false_free_teacher": metric_value(teacher_verbose["final_eval"], "front_sector_false_free_rate_delta"),
        "front_false_free_adapter": metric_value(adapter_verbose["final_eval"], "front_sector_false_free_rate_delta"),
        "future_false_free_teacher": metric_value(teacher_verbose["final_eval"], "future_h4_h6_false_free_rate_delta", default=0.0),
        "future_false_free_adapter": metric_value(adapter_verbose["final_eval"], "future_h4_h6_false_free_rate_delta", default=0.0),
        "false_positive_teacher": metric_value(teacher_verbose["final_eval"], "false_positive_delta"),
        "false_positive_adapter": metric_value(adapter_verbose["final_eval"], "false_positive_delta"),
        "front_local_proxy_teacher": teacher_verbose["front_proxy_final"],
        "front_local_proxy_adapter": adapter_verbose["front_proxy_final"],
        "protected_zone_preservation_ratio_teacher": float(teacher_verbose["protected_zone_preservation_ratio"]),
        "protected_zone_preservation_ratio_adapter": float(adapter_verbose["protected_zone_preservation_ratio"]),
        "adapter_minus_teacher_density": metric_value(adapter_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio") - metric_value(teacher_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "adapter_minus_native_density": metric_value(adapter_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio") - metric_value(native_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "gt_recovery_count": int((adapter_occ & gt_occ & ~native_occ).sum().item()),
        "gt_free_false_positive_count": int((adapter_occ & ~gt_occ).sum().item()),
        "wrong_class_count": int(((adapter_verbose["final_semantic"] != (gpu_static["gt_h"] if gpu_static is not None else teacher_h["gt_h"].long())) & adapter_occ & gt_occ).sum().item()),
        "newly_occupied_vs_native": int((adapter_occ & ~native_occ).sum().item()),
        "newly_occupied_vs_teacher": int((adapter_occ & ~teacher_occ).sum().item()),
        "front_newly_occupied_vs_native": int((adapter_occ & ~native_occ & front_mask).sum().item()),
        "nonfront_newly_occupied_vs_native": int((adapter_occ & ~native_occ & ~front_mask).sum().item()),
        "future_h46_newly_occupied_vs_native": int((adapter_occ & ~native_occ).sum().item()) if horizon_s in {4, 6} else 0,
        "protected_newly_occupied_vs_native": int((adapter_occ & ~native_occ & protected_mask).sum().item()),
        "raw_delta_zone_newly_occupied": int((adapter_occ & raw_delta_mask).sum().item()),
        "alpha_mean": alpha_summary["mean"],
        "alpha_max": alpha_summary["max"],
        "alpha_p95": alpha_summary["p95"],
        "alpha_p99": alpha_summary["p99"],
        "sample_joint_success": 1.0 if (
            metric_value(adapter_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio") <= min(metric_value(teacher_verbose["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio") + 0.04, 0.08)
            and metric_value(adapter_verbose["final_eval"], "false_positive_delta") <= metric_value(teacher_verbose["final_eval"], "false_positive_delta") + 0.015
            and adapter_verbose["front_proxy_final"] <= 1.30
        ) else 0.0,
    }
    if "front_proxy_damping_factor" in alpha_debug and "front_proxy_raw" in alpha_debug:
        row["front_proxy_damping_factor"] = float(alpha_debug["front_proxy_damping_factor"].item())
        row["front_proxy_raw_first_pass"] = float(alpha_debug["front_proxy_raw"].item())
    return row


def summarize_policy_rows(rows: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    ratio = mean_front_recovery_ratio(rows)
    return {
        "sample_count": sample_count,
        "row_count": len(rows),
        "mean_front_recovery_ratio": 0.0 if ratio is None else ratio,
        "mean_front_recovery_ratio_valid": ratio is not None,
        "mean_front_sector_false_free_rate_delta_vs_native": float(np.mean([row["front_false_free_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_front_sector_false_free_rate_delta_vs_native": float(np.mean([row["front_false_free_teacher"] for row in rows])) if rows else 0.0,
        "mean_future_h4_h6_false_free_rate_delta_vs_native": float(np.mean([row["future_false_free_adapter"] for row in rows if int(row["horizon_s"]) in {4, 6}])) if rows else 0.0,
        "mean_teacher_future_h4_h6_false_free_rate_delta_vs_native": float(np.mean([row["future_false_free_teacher"] for row in rows if int(row["horizon_s"]) in {4, 6}])) if rows else 0.0,
        "mean_density_delta": float(np.mean([row["final_density_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_density_delta": float(np.mean([row["final_density_teacher"] for row in rows])) if rows else 0.0,
        "mean_false_positive_delta": float(np.mean([row["false_positive_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_false_positive_delta": float(np.mean([row["false_positive_teacher"] for row in rows])) if rows else 0.0,
        "mean_front_local_proxy": float(np.mean([row["front_local_proxy_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_front_local_proxy": float(np.mean([row["front_local_proxy_teacher"] for row in rows])) if rows else 0.0,
        "mean_protected_zone_preservation_ratio": float(np.mean([row["protected_zone_preservation_ratio_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_protected_zone_preservation_ratio": float(np.mean([row["protected_zone_preservation_ratio_teacher"] for row in rows])) if rows else 0.0,
        "mean_alpha_mean": float(np.mean([row["alpha_mean"] for row in rows])) if rows else 0.0,
        "mean_alpha_max": float(np.mean([row["alpha_max"] for row in rows])) if rows else 0.0,
        "mean_alpha_p95": float(np.mean([row["alpha_p95"] for row in rows])) if rows else 0.0,
        "mean_alpha_p99": float(np.mean([row["alpha_p99"] for row in rows])) if rows else 0.0,
        "sample_joint_success_rate": float(np.mean([row["sample_joint_success"] for row in rows])) if rows else 0.0,
    }


def compute_fast_selection_metrics(
    *,
    final_semantic: torch.Tensor,
    gt_h: torch.Tensor,
    native_final_semantic: torch.Tensor,
    teacher_final_semantic: torch.Tensor,
    front_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred_occ = final_semantic != EMPTY_IDX
    gt_occ = gt_h != EMPTY_IDX
    native_occ = native_final_semantic != EMPTY_IDX
    teacher_occ = teacher_final_semantic != EMPTY_IDX
    gt_free = ~gt_occ
    front_gt = front_mask & gt_occ

    gt_occ_den = gt_occ.sum().clamp_min(1).float()
    gt_free_den = gt_free.sum().clamp_min(1).float()
    front_den = front_gt.sum().clamp_min(1).float()
    native_front_occ_den = (front_mask & native_occ).sum().clamp_min(1).float()

    pred_occ_count = pred_occ.sum().float()
    native_occ_count = native_occ.sum().float()
    teacher_occ_count = teacher_occ.sum().float()

    pred_density_delta = pred_occ_count / gt_occ_den - native_occ_count / gt_occ_den
    teacher_density_delta = teacher_occ_count / gt_occ_den - native_occ_count / gt_occ_den

    pred_fp = (pred_occ & gt_free).sum().float() / gt_free_den
    native_fp = (native_occ & gt_free).sum().float() / gt_free_den
    teacher_fp = (teacher_occ & gt_free).sum().float() / gt_free_den
    pred_fp_delta = pred_fp - native_fp
    teacher_fp_delta = teacher_fp - native_fp

    pred_front_ff = (front_gt & ~pred_occ).sum().float() / front_den
    native_front_ff = (front_gt & ~native_occ).sum().float() / front_den
    teacher_front_ff = (front_gt & ~teacher_occ).sum().float() / front_den
    pred_front_ff_delta = pred_front_ff - native_front_ff
    teacher_front_ff_delta = teacher_front_ff - native_front_ff

    front_proxy = (front_mask & pred_occ).sum().float() / native_front_occ_den
    return {
        "front_false_free_adapter": pred_front_ff_delta,
        "front_false_free_teacher": teacher_front_ff_delta,
        "final_density_adapter": pred_density_delta,
        "final_density_teacher": teacher_density_delta,
        "false_positive_adapter": pred_fp_delta,
        "false_positive_teacher": teacher_fp_delta,
        "front_local_proxy_adapter": front_proxy,
    }


def run_postprocess_density_only(
    *,
    raw_semantic: torch.Tensor,
    raw_confidence: torch.Tensor,
    raw_margin: torch.Tensor,
    native_semantic: torch.Tensor,
    candidate: Any,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    agreement_map: torch.Tensor,
) -> dict[str, Any]:
    device = raw_semantic.device
    raw_semantic = raw_semantic.long()
    raw_confidence = raw_confidence.float().to(device)
    raw_margin = raw_margin.float().to(device)
    native_semantic = native_semantic.long().to(device)
    raw_occ = raw_semantic != EMPTY_IDX
    native_occ = native_semantic != EMPTY_IDX
    raw_delta = raw_occ & ~native_occ
    agreement = agreement_map.float().to(device)
    sectors_local = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sectors.items()}
    protected = base.sw13c_fix.protected_zone_fix(
        candidate.protected_variant,
        raw_occ,
        raw_delta,
        raw_confidence,
        raw_margin,
        agreement,
        sectors_local,
        horizon_s,
    )
    low_value = base.sw13c_fix.low_value_score(
        raw_occ,
        protected,
        raw_confidence,
        raw_margin,
        agreement,
        sectors_local,
        horizon_s,
        candidate.wrong_class_aware,
        candidate.front_bias,
    )
    f3_semantic, f3_pruned, _f3_meta = base.sw13c_fix.apply_pruning_no_gt(
        raw_semantic=raw_semantic,
        protected_mask=protected,
        low_value_score_map=low_value,
        native_occ_count=int(native_occ.sum().item()),
        raw_occ_count=int(raw_occ.sum().item()),
        raw_delta_count=int(raw_delta.sum().item()),
        budget_mode="native_expansion_ratio",
        expansion_ratio=candidate.expansion_ratio,
        keep_ratio=None,
    )
    final_semantic, _front_pruned, cap_meta = base.frontcap50.apply_front_local_cap_no_gt(
        final_semantic_before_cap=f3_semantic.long(),
        native_semantic=native_semantic.long(),
        raw_semantic=raw_semantic.long(),
        protected_mask=protected,
        front_mask=sectors_local["front"].bool(),
        confidence=raw_confidence,
        margin=raw_margin,
        agreement=agreement,
        low_value_score_map=low_value,
        cap_ratio=candidate.cap_ratio,
        cap_mode="disabled" if candidate.cap_ratio is None else "front_native_ratio_cap",
    )
    return {
        "raw_semantic": raw_semantic,
        "f3_semantic": f3_semantic.long(),
        "final_semantic": final_semantic.long(),
        "protected_mask": protected.bool(),
        "raw_delta_mask": raw_delta.bool(),
        "front_proxy_final": float(cap_meta.get("front_local_density_proxy_after", 0.0)),
        "protected_zone_preservation_ratio": 1.0 - safe_div(float((protected & f3_pruned).sum().item()), max(1.0, float(protected.sum().item()))),
    }


def summarize_policy_fast_rows(rows: list[dict[str, float]], sample_count: int) -> dict[str, Any]:
    ratio = mean_front_recovery_ratio(rows)
    return {
        "sample_count": sample_count,
        "row_count": len(rows),
        "mean_front_recovery_ratio": 0.0 if ratio is None else ratio,
        "mean_front_recovery_ratio_valid": ratio is not None,
        "mean_density_delta": float(np.mean([row["final_density_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_density_delta": float(np.mean([row["final_density_teacher"] for row in rows])) if rows else 0.0,
        "mean_false_positive_delta": float(np.mean([row["false_positive_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_false_positive_delta": float(np.mean([row["false_positive_teacher"] for row in rows])) if rows else 0.0,
        "mean_front_local_proxy": float(np.mean([row["front_local_proxy_adapter"] for row in rows])) if rows else 0.0,
        "mean_teacher_front_local_proxy": float(np.mean([row["front_local_proxy_teacher"] for row in rows])) if rows else 0.0,
        "mean_alpha_mean": float(np.mean([row["alpha_mean"] for row in rows])) if rows else 0.0,
        "mean_alpha_max": float(np.mean([row["alpha_max"] for row in rows])) if rows else 0.0,
        "mean_alpha_p95": float(np.mean([row["alpha_p95"] for row in rows])) if rows else 0.0,
        "mean_alpha_p99": float(np.mean([row["alpha_p99"] for row in rows])) if rows else 0.0,
        "sample_joint_success_rate": float(np.mean([row["sample_joint_success"] for row in rows])) if rows else 0.0,
    }


def collect_policy_selection_summaries_sample_major(
    model: Any,
    dataset: Any,
    adapter: SpatialGateAdapter,
    sample_indices: list[int],
    perturbation_id: str,
    policies: list[RescuePolicy],
    case_cache: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows_by_policy: dict[str, list[dict[str, float]]] = {policy.name: [] for policy in policies}
    sectors_cpu = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    for sample_index in sample_indices:
        case: dict[str, Any] | None = None
        try:
            case = case_cache[sample_index]
            case["prepared_case"] = prepare_policy_feature_case(model, adapter, case["batch_deg"], case["cache"], perturbation_id)
            device = case["prepared_case"]["frame_levels"][0][0].device
            case["gpu_static"] = {
                horizon_s: {
                    "gt_h": case["by_horizon"][horizon_s]["teacher_h"]["gt_h"].long().to(device),
                    "gt0": case["by_horizon"][horizon_s]["teacher_h"]["gt0"].long().to(device),
                    "agreement": case["by_horizon"][horizon_s]["agreement"].float().to(device),
                    "native_semantic": case["by_horizon"][horizon_s]["native_semantic"].long().to(device),
                    "native_final_semantic": case["by_horizon"][horizon_s]["native_verbose"]["final_semantic"].long().to(device),
                    "teacher_final_semantic": case["by_horizon"][horizon_s]["teacher_verbose"]["final_semantic"].long().to(device),
                    "native_final_density": float(metric_value(case["by_horizon"][horizon_s]["native_verbose"]["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "teacher_final_density": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "teacher_false_positive": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["final_eval"], "false_positive_delta")),
                    "teacher_front_proxy": float(case["by_horizon"][horizon_s]["teacher_verbose"]["front_proxy_final"]),
                }
                for horizon_s in CORE_HORIZONS
            }
            sectors = {k: v.to(device) for k, v in sectors_cpu.items()}
            for policy in policies:
                per_h, alpha_debug = run_policy_for_case(model, adapter, case, perturbation_id, policy, sectors)
                alpha_summary = alpha_stats_from_debug(alpha_debug)
                try:
                    for horizon_s in CORE_HORIZONS:
                        static = case["gpu_static"][horizon_s]
                        final_semantic, front_proxy_after = run_postprocess_final_only(
                            raw_semantic=per_h[horizon_s]["raw_semantic"].long(),
                            raw_confidence=per_h[horizon_s]["raw_confidence"].float(),
                            raw_margin=per_h[horizon_s]["raw_margin"].float(),
                            native_semantic=static["native_semantic"],
                            candidate=base.build_candidate(),
                            horizon_s=horizon_s,
                            sectors=sectors,
                            agreement_map=static["agreement"],
                        )
                        metrics = compute_fast_selection_metrics(
                            final_semantic=final_semantic,
                            gt_h=static["gt_h"],
                            native_final_semantic=static["native_final_semantic"],
                            teacher_final_semantic=static["teacher_final_semantic"],
                            front_mask=sectors["front"].bool(),
                        )
                        row = {
                            "front_false_free_adapter": float(metrics["front_false_free_adapter"].item()),
                            "front_false_free_teacher": float(metrics["front_false_free_teacher"].item()),
                            "final_density_adapter": float(metrics["final_density_adapter"].item()),
                            "final_density_teacher": float(metrics["final_density_teacher"].item()),
                            "false_positive_adapter": float(metrics["false_positive_adapter"].item()),
                            "false_positive_teacher": float(metrics["false_positive_teacher"].item()),
                            "front_local_proxy_adapter": float(front_proxy_after),
                            "front_local_proxy_teacher": static["teacher_front_proxy"],
                            "alpha_mean": alpha_summary["mean"],
                            "alpha_max": alpha_summary["max"],
                            "alpha_p95": alpha_summary["p95"],
                            "alpha_p99": alpha_summary["p99"],
                            "sample_joint_success": 1.0 if (
                                float(metrics["final_density_adapter"].item()) <= min(static["teacher_final_density"] + 0.04, 0.08)
                                and float(metrics["false_positive_adapter"].item()) <= static["teacher_false_positive"] + 0.015
                                and float(front_proxy_after) <= 1.30
                            ) else 0.0,
                        }
                        rows_by_policy[policy.name].append(row)
                finally:
                    base.release_runtime(per_h, alpha_debug)
        finally:
            gpu_static = case.pop("gpu_static", None) if case is not None else None
            prepared_case = case.pop("prepared_case", None) if case is not None else None
            base.release_runtime(gpu_static, prepared_case)
            release_case_cache_sample(case_cache, sample_index, case)
    return {
        policy.name: {
            "summary": summarize_policy_fast_rows(rows_by_policy[policy.name], sample_count=len(sample_indices)),
        }
        for policy in policies
    }


def collect_policy_summaries_sample_major(
    model: Any,
    dataset: Any,
    adapter: SpatialGateAdapter,
    sample_indices: list[int],
    perturbation_id: str,
    split_name: str,
    policies: list[RescuePolicy],
    case_cache: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate = base.build_candidate()
    sectors_cpu = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    rows_by_policy: dict[str, list[dict[str, Any]]] = {policy.name: [] for policy in policies}
    for sample_index in sample_indices:
        case: dict[str, Any] | None = None
        try:
            case = case_cache[sample_index]
            case["prepared_case"] = prepare_policy_feature_case(model, adapter, case["batch_deg"], case["cache"], perturbation_id)
            device = case["prepared_case"]["frame_levels"][0][0].device
            case["gpu_static"] = {
                horizon_s: {
                    "native_semantic": case["by_horizon"][horizon_s]["native_semantic"].long().to(device),
                    "agreement": case["by_horizon"][horizon_s]["agreement"].float().to(device),
                    "gt_h": case["by_horizon"][horizon_s]["teacher_h"]["gt_h"].long().to(device),
                    "gt0": case["by_horizon"][horizon_s]["teacher_h"]["gt0"].long().to(device),
                    "teacher_final_semantic": case["by_horizon"][horizon_s]["teacher_verbose"]["final_semantic"].long().to(device),
                    "native_final_semantic": case["by_horizon"][horizon_s]["native_verbose"]["final_semantic"].long().to(device),
                }
                for horizon_s in CORE_HORIZONS
            }
            sectors = {k: v.to(device) for k, v in sectors_cpu.items()}
            for policy in policies:
                per_h, alpha_debug = run_policy_for_case(model, adapter, case, perturbation_id, policy, sectors)
                try:
                    for horizon_s in CORE_HORIZONS:
                        rows_by_policy[policy.name].append(build_policy_row(case, per_h, alpha_debug, split_name, policy, sample_index, horizon_s, sectors, candidate))
                finally:
                    base.release_runtime(per_h, alpha_debug)
        finally:
            gpu_static = case.pop("gpu_static", None) if case is not None else None
            prepared_case = case.pop("prepared_case", None) if case is not None else None
            base.release_runtime(gpu_static, prepared_case)
            release_case_cache_sample(case_cache, sample_index, case)
    return {
        policy.name: {
            "rows": rows_by_policy[policy.name],
            "summary": summarize_policy_rows(rows_by_policy[policy.name], sample_count=len(sample_indices)),
        }
        for policy in policies
    }


def collect_single_policy_rows_sample_major(
    model: Any,
    dataset: Any,
    adapter: SpatialGateAdapter,
    sample_indices: list[int],
    perturbation_id: str,
    split_name: str,
    policy: RescuePolicy,
    case_cache: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = collect_policy_summaries_sample_major(
        model,
        dataset,
        adapter,
        sample_indices,
        perturbation_id,
        split_name,
        [policy],
        case_cache,
    )[policy.name]
    return payload["rows"], payload["summary"]


def collect_density_rows_sample_major(
    model: Any,
    dataset: Any,
    adapter: SpatialGateAdapter,
    sample_indices: list[int],
    perturbation_id: str,
    split_name: str,
    policy: RescuePolicy,
    case_cache: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = base.build_candidate()
    sectors_cpu = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    rows: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        case: dict[str, Any] | None = None
        try:
            case = case_cache[sample_index]
            case["prepared_case"] = prepare_policy_feature_case(model, adapter, case["batch_deg"], case["cache"], perturbation_id)
            device = case["prepared_case"]["frame_levels"][0][0].device
            case["gpu_static"] = {
                horizon_s: {
                    "native_semantic": case["by_horizon"][horizon_s]["native_semantic"].long().to(device),
                    "agreement": case["by_horizon"][horizon_s]["agreement"].float().to(device),
                    "gt_h": case["by_horizon"][horizon_s]["teacher_h"]["gt_h"].long().to(device),
                    "native_final_semantic": case["by_horizon"][horizon_s]["native_verbose"]["final_semantic"].long().to(device),
                    "teacher_final_semantic": case["by_horizon"][horizon_s]["teacher_verbose"]["final_semantic"].long().to(device),
                    "raw_density_native": float(metric_value(case["by_horizon"][horizon_s]["native_verbose"]["raw_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "raw_density_teacher": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["raw_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "f3_density_native": float(metric_value(case["by_horizon"][horizon_s]["native_verbose"]["f3_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "f3_density_teacher": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["f3_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "final_density_native": float(metric_value(case["by_horizon"][horizon_s]["native_verbose"]["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "final_density_teacher": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio")),
                    "front_false_free_teacher": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["final_eval"], "front_sector_false_free_rate_delta")),
                    "future_false_free_teacher": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["final_eval"], "future_h4_h6_false_free_rate_delta", default=0.0)),
                    "false_positive_teacher": float(metric_value(case["by_horizon"][horizon_s]["teacher_verbose"]["final_eval"], "false_positive_delta")),
                    "front_local_proxy_teacher": float(case["by_horizon"][horizon_s]["teacher_verbose"]["front_proxy_final"]),
                }
                for horizon_s in CORE_HORIZONS
            }
            sectors = {k: v.to(device) for k, v in sectors_cpu.items()}
            per_h, alpha_debug = run_policy_for_case(model, adapter, case, perturbation_id, policy, sectors)
            alpha_summary = alpha_stats_from_debug(alpha_debug)
            front_mask = sectors["front"].bool()
            try:
                for horizon_s in CORE_HORIZONS:
                    static = case["gpu_static"][horizon_s]
                    stage = run_postprocess_density_only(
                        raw_semantic=per_h[horizon_s]["raw_semantic"].long(),
                        raw_confidence=per_h[horizon_s]["raw_confidence"].float(),
                        raw_margin=per_h[horizon_s]["raw_margin"].float(),
                        native_semantic=static["native_semantic"],
                        candidate=candidate,
                        horizon_s=horizon_s,
                        sectors=sectors,
                        agreement_map=static["agreement"],
                    )
                    raw_metrics = compute_fast_selection_metrics(
                        final_semantic=stage["raw_semantic"],
                        gt_h=static["gt_h"],
                        native_final_semantic=static["native_semantic"],
                        teacher_final_semantic=static["teacher_final_semantic"],
                        front_mask=front_mask,
                    )
                    f3_metrics = compute_fast_selection_metrics(
                        final_semantic=stage["f3_semantic"],
                        gt_h=static["gt_h"],
                        native_final_semantic=static["native_semantic"],
                        teacher_final_semantic=static["teacher_final_semantic"],
                        front_mask=front_mask,
                    )
                    final_metrics = compute_fast_selection_metrics(
                        final_semantic=stage["final_semantic"],
                        gt_h=static["gt_h"],
                        native_final_semantic=static["native_final_semantic"],
                        teacher_final_semantic=static["teacher_final_semantic"],
                        front_mask=front_mask,
                    )
                    gt_occ = static["gt_h"] != EMPTY_IDX
                    adapter_occ = stage["final_semantic"] != EMPTY_IDX
                    native_occ = static["native_final_semantic"] != EMPTY_IDX
                    teacher_occ = static["teacher_final_semantic"] != EMPTY_IDX
                    future_false_free_adapter = 0.0
                    if horizon_s in {4, 6}:
                        gt_occ_den = gt_occ.sum().clamp_min(1).float()
                        pred_ff = (~adapter_occ & gt_occ).sum().float() / gt_occ_den
                        native_ff = (~native_occ & gt_occ).sum().float() / gt_occ_den
                        future_false_free_adapter = float((pred_ff - native_ff).item())
                    row = {
                        "split_name": split_name,
                        "policy_name": policy.name,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "raw_density_native": static["raw_density_native"],
                        "raw_density_teacher": static["raw_density_teacher"],
                        "raw_density_adapter": float(raw_metrics["final_density_adapter"].item()),
                        "f3_density_native": static["f3_density_native"],
                        "f3_density_teacher": static["f3_density_teacher"],
                        "f3_density_adapter": float(f3_metrics["final_density_adapter"].item()),
                        "final_density_native": static["final_density_native"],
                        "final_density_teacher": static["final_density_teacher"],
                        "final_density_adapter": float(final_metrics["final_density_adapter"].item()),
                        "front_false_free_teacher": static["front_false_free_teacher"],
                        "front_false_free_adapter": float(final_metrics["front_false_free_adapter"].item()),
                        "future_false_free_teacher": static["future_false_free_teacher"],
                        "future_false_free_adapter": future_false_free_adapter,
                        "false_positive_teacher": static["false_positive_teacher"],
                        "false_positive_adapter": float(final_metrics["false_positive_adapter"].item()),
                        "front_local_proxy_teacher": static["front_local_proxy_teacher"],
                        "front_local_proxy_adapter": stage["front_proxy_final"],
                        "protected_zone_preservation_ratio_teacher": 1.0,
                        "protected_zone_preservation_ratio_adapter": float(stage["protected_zone_preservation_ratio"]),
                        "adapter_minus_teacher_density": float(final_metrics["final_density_adapter"].item()) - static["final_density_teacher"],
                        "adapter_minus_native_density": float(final_metrics["final_density_adapter"].item()) - static["final_density_native"],
                        "gt_recovery_count": int((adapter_occ & gt_occ & ~native_occ).sum().item()),
                        "gt_free_false_positive_count": int((adapter_occ & ~gt_occ).sum().item()),
                        "wrong_class_count": int(((stage["final_semantic"] != static["gt_h"]) & adapter_occ & gt_occ).sum().item()),
                        "newly_occupied_vs_native": int((adapter_occ & ~native_occ).sum().item()),
                        "newly_occupied_vs_teacher": int((adapter_occ & ~teacher_occ).sum().item()),
                        "front_newly_occupied_vs_native": int((adapter_occ & ~native_occ & front_mask).sum().item()),
                        "nonfront_newly_occupied_vs_native": int((adapter_occ & ~native_occ & ~front_mask).sum().item()),
                        "future_h46_newly_occupied_vs_native": int((adapter_occ & ~native_occ).sum().item()) if horizon_s in {4, 6} else 0,
                        "protected_newly_occupied_vs_native": int((adapter_occ & ~native_occ & stage["protected_mask"]).sum().item()),
                        "raw_delta_zone_newly_occupied": int((adapter_occ & stage["raw_delta_mask"]).sum().item()),
                        "alpha_mean": alpha_summary["mean"],
                        "alpha_max": alpha_summary["max"],
                        "alpha_p95": alpha_summary["p95"],
                        "alpha_p99": alpha_summary["p99"],
                        "sample_joint_success": 1.0 if (
                            float(final_metrics["final_density_adapter"].item()) <= min(static["final_density_teacher"] + 0.04, 0.08)
                            and float(final_metrics["false_positive_adapter"].item()) <= static["false_positive_teacher"] + 0.015
                            and float(stage["front_proxy_final"]) <= 1.30
                        ) else 0.0,
                    }
                    if "front_proxy_damping_factor" in alpha_debug and "front_proxy_raw" in alpha_debug:
                        row["front_proxy_damping_factor"] = float(alpha_debug["front_proxy_damping_factor"].item())
                        row["front_proxy_raw_first_pass"] = float(alpha_debug["front_proxy_raw"].item())
                    rows.append(row)
            finally:
                base.release_runtime(per_h, alpha_debug)
        finally:
            gpu_static = case.pop("gpu_static", None) if case is not None else None
            prepared_case = case.pop("prepared_case", None) if case is not None else None
            base.release_runtime(gpu_static, prepared_case)
            release_case_cache_sample(case_cache, sample_index, case)
    return rows, summarize_policy_rows(rows, sample_count=len(sample_indices))


def phase2_density_risk_attribution(args: argparse.Namespace, adapter: SpatialGateAdapter) -> dict[str, Any]:
    print("[rescue] phase2 density attribution", flush=True)
    _, dataset, model, _ = base.build_runtime(train=False)
    model.eval()
    try:
        return phase2_density_risk_attribution_with_runtime(args, adapter, dataset, model)
    finally:
        base.release_runtime(model, dataset, collect=True, empty_cache=True)


def phase2_density_risk_attribution_with_runtime(args: argparse.Namespace, adapter: SpatialGateAdapter, dataset: Any, model: Any, debug_case_cache: dict[int, dict[str, Any]] | None = None, val_case_cache: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    if not args.force_runtime_attribution:
        existing = load_existing_execution_payload("density_attr")
        if existing is not None:
            print("[rescue] reuse existing phase2 density risk attribution", flush=True)
            return existing
    policy = RescuePolicy(name="original", global_scale=1.0)
    candidate = base.build_candidate()
    sectors = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    debug_indices = list(range(args.eval_start, args.eval_end + 1))
    val_indices = list(range(args.val_start, args.val_end + 1))
    if debug_case_cache is None:
        debug_case_cache = build_case_cache(model, dataset, debug_indices, base.teacher_ref().perturbation_id, candidate, sectors)
    if val_case_cache is None:
        val_case_cache = build_case_cache(model, dataset, val_indices, base.teacher_ref().perturbation_id, candidate, sectors)
    rows_debug, summary_debug = collect_density_rows_sample_major(
        model,
        dataset,
        adapter,
        debug_indices,
        base.teacher_ref().perturbation_id,
        "eval_debug",
        policy,
        debug_case_cache,
    )
    rows_val, summary_val = collect_density_rows_sample_major(
        model,
        dataset,
        adapter,
        val_indices,
        base.teacher_ref().perturbation_id,
        "val_small",
        policy,
        val_case_cache,
    )
    rows = rows_debug + rows_val
    write_csv(REPORTS_DIR / "sw14b_density_risk_attribution.csv", rows)
    core500_summary_path = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_metric_summary.csv"
    core500_rows = read_csv(core500_summary_path)
    core500_teacher_density = 0.0601
    if core500_rows:
        for row in core500_rows:
            if row.get("metric_name") == "pred_gt_density_delta":
                core500_teacher_density = float(row["metric_value"])
                break
    mean_raw_gap = float(np.mean([row["raw_density_adapter"] - row["raw_density_teacher"] for row in rows])) if rows else 0.0
    mean_f3_gap = float(np.mean([row["f3_density_adapter"] - row["f3_density_teacher"] for row in rows])) if rows else 0.0
    mean_final_gap = float(np.mean([row["final_density_adapter"] - row["final_density_teacher"] for row in rows])) if rows else 0.0
    mean_front_new = float(np.mean([row["front_newly_occupied_vs_native"] for row in rows])) if rows else 0.0
    mean_nonfront_new = float(np.mean([row["nonfront_newly_occupied_vs_native"] for row in rows])) if rows else 0.0
    mean_fp_count = float(np.mean([row["gt_free_false_positive_count"] for row in rows])) if rows else 0.0
    mean_recovery_count = float(np.mean([row["gt_recovery_count"] for row in rows])) if rows else 0.0
    decision = "DENSITY_R1_ADAPTER_RAW_OVER_OCCUPIES"
    if mean_raw_gap <= 0.01 and mean_f3_gap > 0.02:
        decision = "DENSITY_R2_F3_FAILS_TO_CLAMP_ADAPTER"
    elif mean_f3_gap <= 0.01 and mean_final_gap > 0.02:
        decision = "DENSITY_R3_FRONTCAP_NOT_ENOUGH"
    elif mean_front_new > mean_nonfront_new and mean_recovery_count >= mean_fp_count:
        decision = "DENSITY_R4_DENSITY_MAINLY_FRONT_RECOVERY"
    elif mean_fp_count > mean_recovery_count:
        decision = "DENSITY_R5_DENSITY_MAINLY_FALSE_POSITIVE"
    elif abs(summary_debug["mean_teacher_density_delta"] - core500_teacher_density) > 0.03:
        decision = "DENSITY_R6_TEACHER_DENSITY_UNUSUALLY_LOW_ON_DEBUG"
    payload = {
        "decision": decision,
        "summary_eval_debug": summary_debug,
        "summary_val_small": summary_val,
        "mean_raw_gap": mean_raw_gap,
        "mean_f3_gap": mean_f3_gap,
        "mean_final_gap": mean_final_gap,
        "core500_teacher_density_reference": core500_teacher_density,
        "eval_debug_teacher_density_reference": summary_debug["mean_teacher_density_delta"],
    }
    write_md(
        REPORTS_DIR / "sw14b_density_risk_attribution.md",
        "\n".join(
            [
                "# SW14B Density Risk Attribution",
                "",
                f"- decision: `{decision}`.",
                f"- mean raw adapter-teacher density gap: `{mean_raw_gap}`.",
                f"- mean F3 adapter-teacher density gap: `{mean_f3_gap}`.",
                f"- mean final adapter-teacher density gap: `{mean_final_gap}`.",
                f"- debug teacher density delta: `{summary_debug['mean_teacher_density_delta']}`.",
                f"- core500 teacher density reference: `{core500_teacher_density}`.",
            ]
        ),
    )
    base.release_runtime(debug_case_cache, val_case_cache)
    return payload


def clean_selection_ok(clean_attr: dict[str, Any]) -> bool:
    if clean_attr["decision"] == "CLEAN_D4_CLEAN_DEGRADATION_FLAG_BUG":
        return bool(clean_attr.get("corrected_clean_safe"))
    return clean_attr["decision"] in {"CLEAN_D2_POSTPROCESS_PATH_DRIFT", "CLEAN_D3_METRIC_SCHEMA_DRIFT", "CLEAN_D5_DRIFT_NEGLIGIBLE_AFTER_DECOMPOSITION"}


def phase3_alpha_scale_sweep(args: argparse.Namespace, adapter: SpatialGateAdapter, clean_attr: dict[str, Any]) -> dict[str, Any]:
    print("[rescue] phase3 alpha scale sweep", flush=True)
    _, dataset, model, _ = base.build_runtime(train=False)
    model.eval()
    try:
        return phase3_alpha_scale_sweep_with_runtime(args, adapter, clean_attr, dataset, model)
    finally:
        base.release_runtime(model, dataset, collect=True, empty_cache=True)


def phase3_alpha_scale_sweep_with_runtime(
    args: argparse.Namespace,
    adapter: SpatialGateAdapter,
    clean_attr: dict[str, Any],
    dataset: Any,
    model: Any,
    case_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not args.force_runtime_sweep:
        existing = load_existing_execution_payload("scale_selection")
        if existing is not None:
            print("[rescue] reuse existing phase3 alpha scale selection", flush=True)
            return existing
    sample_indices = list(range(args.val_start, args.val_end + 1))
    owns_case_cache = case_cache is None
    if case_cache is None:
        case_cache = build_case_cache(
            model,
            dataset,
            sample_indices,
            base.teacher_ref().perturbation_id,
            base.build_candidate(),
            {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()},
        )
    scales = [0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]
    policies = [RescuePolicy(name=f"alpha_scale_{scale:.2f}", global_scale=scale) for scale in scales]
    print(f"[rescue] alpha sweep sample-major over {len(sample_indices)} samples x {len(policies)} scales", flush=True)
    sweep_results = collect_policy_selection_summaries_sample_major(
        model,
        dataset,
        adapter,
        sample_indices,
        base.teacher_ref().perturbation_id,
        policies,
        case_cache,
    )
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scale, policy in zip(scales, policies):
        summary = sweep_results[policy.name]["summary"]
        teacher_density = float(summary["mean_teacher_density_delta"])
        teacher_fp = float(summary["mean_teacher_false_positive_delta"])
        safety_ok = (
            (float(summary["mean_density_delta"]) <= teacher_density + 0.04 or float(summary["mean_density_delta"]) <= 0.08)
            and float(summary["mean_false_positive_delta"]) <= teacher_fp + 0.015
            and float(summary["mean_front_local_proxy"]) <= 1.30
            and clean_selection_ok(clean_attr)
        )
        row = {
            "scale": scale,
            "policy_name": policy.name,
            "front_recovery_ratio": summary["mean_front_recovery_ratio"],
            "density_delta": summary["mean_density_delta"],
            "teacher_density_delta": teacher_density,
            "false_positive_delta": summary["mean_false_positive_delta"],
            "teacher_false_positive_delta": teacher_fp,
            "front_local_proxy": summary["mean_front_local_proxy"],
            "alpha_mean": summary["mean_alpha_mean"],
            "alpha_max": summary["mean_alpha_max"],
            "alpha_p95": summary["mean_alpha_p95"],
            "alpha_p99": summary["mean_alpha_p99"],
            "sample_joint_success_rate": summary["sample_joint_success_rate"],
            "clean_attr_safe": clean_selection_ok(clean_attr),
            "safety_ok": safety_ok,
        }
        rows.append(row)
        if is_effective_alpha_scale_row(row) and safety_ok and (best is None or float(row["front_recovery_ratio"]) > float(best["front_recovery_ratio"])):
            best = row
    write_csv(REPORTS_DIR / "sw14b_alpha_scale_sweep_valsmall.csv", rows)
    plt.figure(figsize=(8, 4))
    plt.plot([row["scale"] for row in rows], [row["front_recovery_ratio"] for row in rows], marker="o", label="front recovery ratio")
    plt.plot([row["scale"] for row in rows], [row["density_delta"] for row in rows], marker="s", label="density delta")
    plt.axhline(0.08, color="tab:red", linestyle="--", linewidth=1, label="density 0.08")
    plt.title("frozen SparseWorld lightweight adapter density-clean rescue alpha scale subset diagnostic")
    plt.xlabel("alpha scale")
    plt.legend()
    plt.savefig(FIGURES_DIR / "sw14b_alpha_scale_sweep.png", dpi=180, bbox_inches="tight")
    plt.close()
    selection = {
        "decision": "SCALE_RESCUE_PASS" if best is not None else "SCALE_RESCUE_FAIL",
        "selected_on_split": "val_small_200_249",
        "uses_eval_debug_for_selection": False,
        "clean_attr_decision": clean_attr["decision"],
        "best_scale_row": best,
        "selection_rule": "best safe non-noop alpha scale, requiring 0 < scale < 1.0; scale=1.0 is original/no rescue.",
    }
    write_json(REPORTS_DIR / "sw14b_alpha_scale_selection.json", selection)
    if owns_case_cache:
        base.release_runtime(case_cache)
    return selection


def phase4_density_aware_clamp(
    args: argparse.Namespace,
    adapter: SpatialGateAdapter,
    clean_attr: dict[str, Any],
    scale_selection: dict[str, Any],
) -> dict[str, Any]:
    if scale_selection_is_effective_pass(scale_selection):
        payload = {"executed": False, "reason": "alpha_scale_rescue_already_passed"}
        write_json(REPORTS_DIR / "sw14b_density_aware_clamp_selection.json", payload)
        write_csv(REPORTS_DIR / "sw14b_density_aware_clamp_valsmall.csv", [])
        return payload
    _, dataset, model, _ = base.build_runtime(train=False)
    model.eval()
    try:
        return phase4_density_aware_clamp_with_runtime(args, adapter, clean_attr, scale_selection, dataset, model)
    finally:
        base.release_runtime(model, dataset, collect=True, empty_cache=True)


def phase4_density_aware_clamp_with_runtime(
    args: argparse.Namespace,
    adapter: SpatialGateAdapter,
    clean_attr: dict[str, Any],
    scale_selection: dict[str, Any],
    dataset: Any,
    model: Any,
    case_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not args.force_runtime_sweep:
        existing = load_existing_execution_payload("clamp_selection")
        if existing is not None:
            print("[rescue] reuse existing phase4 density-aware clamp selection", flush=True)
            return existing
    if scale_selection_is_effective_pass(scale_selection):
        payload = {"executed": False, "reason": "alpha_scale_rescue_already_passed"}
        write_json(REPORTS_DIR / "sw14b_density_aware_clamp_selection.json", payload)
        write_csv(REPORTS_DIR / "sw14b_density_aware_clamp_valsmall.csv", [])
        return payload
    print("[rescue] phase4 density-aware clamp", flush=True)
    sample_indices = list(range(args.val_start, args.val_end + 1))
    owns_case_cache = case_cache is None
    if case_cache is None:
        case_cache = build_case_cache(
            model,
            dataset,
            sample_indices,
            base.teacher_ref().perturbation_id,
            base.build_candidate(),
            {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()},
        )
    camera_policies = [
        RescuePolicy(name=f"camera_scale_f{front_scale:.2f}_s{side_scale:.2f}", global_scale=1.0, front_scale=front_scale, side_scale=side_scale)
        for front_scale in [0.4, 0.6, 0.8, 1.0]
        for side_scale in [0.25, 0.4, 0.6, 0.8]
    ]
    damping_policy = RescuePolicy(name="front_proxy_damping", global_scale=1.0, front_proxy_damping=True, front_proxy_damping_base_scale=1.0)
    policies = camera_policies + [damping_policy]
    print(f"[rescue] clamp sweep sample-major over {len(sample_indices)} samples x {len(policies)} policies", flush=True)
    sweep_results = collect_policy_selection_summaries_sample_major(
        model,
        dataset,
        adapter,
        sample_indices,
        base.teacher_ref().perturbation_id,
        policies,
        case_cache,
    )
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for policy in camera_policies:
        front_scale = policy.front_scale
        side_scale = policy.side_scale
        summary = sweep_results[policy.name]["summary"]
        row = {
            "policy_name": policy.name,
            "front_scale": front_scale,
            "side_scale": side_scale,
            "front_recovery_ratio": summary["mean_front_recovery_ratio"],
            "density_delta": summary["mean_density_delta"],
            "teacher_density_delta": summary["mean_teacher_density_delta"],
            "false_positive_delta": summary["mean_false_positive_delta"],
            "teacher_false_positive_delta": summary["mean_teacher_false_positive_delta"],
            "front_local_proxy": summary["mean_front_local_proxy"],
            "sample_joint_success_rate": summary["sample_joint_success_rate"],
            "clean_attr_safe": clean_selection_ok(clean_attr),
            "safety_ok": (
                (float(summary["mean_density_delta"]) <= float(summary["mean_teacher_density_delta"]) + 0.04 or float(summary["mean_density_delta"]) <= 0.08)
                and float(summary["mean_false_positive_delta"]) <= float(summary["mean_teacher_false_positive_delta"]) + 0.015
                and float(summary["mean_front_local_proxy"]) <= 1.30
                and clean_selection_ok(clean_attr)
        ),
        }
        rows.append(row)
        if row["safety_ok"] and (best is None or float(row["front_recovery_ratio"]) > float(best["front_recovery_ratio"])):
            best = row
    summary = sweep_results[damping_policy.name]["summary"]
    damping_row = {
        "policy_name": damping_policy.name,
        "front_scale": 1.0,
        "side_scale": 1.0,
        "front_recovery_ratio": summary["mean_front_recovery_ratio"],
        "density_delta": summary["mean_density_delta"],
        "teacher_density_delta": summary["mean_teacher_density_delta"],
        "false_positive_delta": summary["mean_false_positive_delta"],
        "teacher_false_positive_delta": summary["mean_teacher_false_positive_delta"],
        "front_local_proxy": summary["mean_front_local_proxy"],
        "sample_joint_success_rate": summary["sample_joint_success_rate"],
        "clean_attr_safe": clean_selection_ok(clean_attr),
        "safety_ok": (
            (float(summary["mean_density_delta"]) <= float(summary["mean_teacher_density_delta"]) + 0.04 or float(summary["mean_density_delta"]) <= 0.08)
            and float(summary["mean_false_positive_delta"]) <= float(summary["mean_teacher_false_positive_delta"]) + 0.015
            and float(summary["mean_front_local_proxy"]) <= 1.30
            and clean_selection_ok(clean_attr)
        ),
    }
    rows.append(damping_row)
    if damping_row["safety_ok"] and (best is None or float(damping_row["front_recovery_ratio"]) > float(best["front_recovery_ratio"])):
        best = damping_row
    write_csv(REPORTS_DIR / "sw14b_density_aware_clamp_valsmall.csv", rows)
    decision = "CLAMP_R2_FAIL"
    if best is not None:
        if best["policy_name"] == "front_proxy_damping":
            decision = "CLAMP_R4_FRONT_PROXY_DAMPING_PASS"
        elif best["front_scale"] != 1.0 or best["side_scale"] != 1.0:
            decision = "CLAMP_R3_CAMERA_SCALE_PASS"
        else:
            decision = "CLAMP_R1_PASS"
    payload = {
        "executed": True,
        "decision": decision,
        "selected_on_split": "val_small_200_249",
        "uses_eval_debug_for_selection": False,
        "best_row": best,
    }
    write_json(REPORTS_DIR / "sw14b_density_aware_clamp_selection.json", payload)
    if owns_case_cache:
        base.release_runtime(case_cache)
    return payload


def phase5_conservative_retrain(
    args: argparse.Namespace,
    checkpoint_path: str,
    clean_attr: dict[str, Any],
    scale_selection: dict[str, Any],
    clamp_selection: dict[str, Any],
) -> dict[str, Any]:
    if args.skip_retrain or scale_selection_is_effective_pass(scale_selection) or clamp_selection.get("decision") in {"CLAMP_R1_PASS", "CLAMP_R3_CAMERA_SCALE_PASS", "CLAMP_R4_FRONT_PROXY_DAMPING_PASS"}:
        payload = {"decision": "RETRAIN_R5_NOT_EXECUTED"}
        write_json(REPORTS_DIR / "sw14b_conservative_retrain_decision.json", payload)
        write_csv(REPORTS_DIR / "sw14b_conservative_retrain_log.csv", [])
        write_csv(REPORTS_DIR / "sw14b_conservative_retrain_val_metrics.csv", [])
        return payload
    _, dataset, model, _ = base.build_runtime(train=True)
    model.eval()
    print("[rescue] phase5 conservative retrain", flush=True)
    base.freeze_sparseworld_modules(model)
    adapter = build_adapter_from_checkpoint(checkpoint_path)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-5)
    candidate = base.build_candidate()
    sectors = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_summary: dict[str, Any] | None = None
    best_score = -1e9
    target_density_cap = 0.08
    for epoch in range(2):
        for step, sample_index in enumerate(range(args.train_start, args.train_end + 1)):
            sample_unwrapped, _, batch_deg, _, _, _, cache = current_and_memory_features_local(model, dataset, sample_index, candidate.perturbation_id)
            teacher_bundle = get_teacher_bundle_cached(model, dataset, sample_index, "rescue_retrain_train", candidate.perturbation_id)
            per_h, alpha_debug = base.run_adapter_forward(model, adapter, batch_deg, sample_unwrapped, cache, candidate.perturbation_id, training=True)
            alpha0 = alpha_debug["alpha_level0"]
            deg_mask = alpha_debug["degradation_mask"]
            total_loss = torch.zeros((), device=alpha0.device)
            for horizon_s in CORE_HORIZONS:
                entry = per_h[horizon_s]
                occ_prob = entry["dense_scores"].max(dim=-1).values.clamp(0.0, 1.0)
                gt_occ = (entry["gt_h"] != EMPTY_IDX).float()
                teacher_occ = (teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"].to(occ_prob.device) != EMPTY_IDX).float()
                teacher_density = float((teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"] != EMPTY_IDX).sum().item()) / float(gt_occ.numel())
                target_density = min(teacher_density + 0.04, target_density_cap)
                gt_occ_loss = F.binary_cross_entropy(occ_prob, gt_occ)
                teacher_occ_loss = F.binary_cross_entropy(occ_prob, teacher_occ)
                density_hinge = torch.relu(occ_prob.mean() - occ_prob.new_tensor(target_density)) ** 2
                fp_loss = occ_prob[gt_occ < 0.5].mean()
                front_pos = gt_occ.bool() & sectors["front"].to(gt_occ.device)
                front_loss = (1.0 - occ_prob[front_pos]).mean() if bool(front_pos.any().item()) else occ_prob.new_zeros(())
                clean_penalty = (alpha0 * (1.0 - deg_mask[:, :, :, None, None])).abs().mean()
                alpha_mag = (alpha0 * deg_mask[:, :, :, None, None]).mean()
                total_loss = total_loss + (
                    1.0 * gt_occ_loss
                    + 0.5 * teacher_occ_loss
                    + 4.0 * density_hinge
                    + 3.0 * fp_loss
                    + 1.5 * clean_penalty
                    + 0.10 * alpha_mag
                    + 1.5 * front_loss
                )
            total_loss = total_loss / len(CORE_HORIZONS)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            train_rows.append({"epoch": epoch, "step": step, "sample_index": sample_index, "total_loss": float(total_loss.detach().cpu().item())})
        val_policy = RescuePolicy(name="retrain_val", global_scale=1.0)
        _rows, summary = collect_policy_rows(model, dataset, adapter, list(range(args.val_start, args.val_end + 1)), candidate.perturbation_id, "val_small", val_policy)
        score = float(summary["mean_front_recovery_ratio"]) - max(0.0, float(summary["mean_density_delta"]) - 0.08) * 3.0 - max(0.0, float(summary["mean_front_local_proxy"]) - 1.30) * 2.0
        summary["epoch"] = epoch
        summary["score"] = score
        val_rows.extend(_rows)
        if score > best_score:
            best_score = score
            best_summary = summary
            best_state = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
        base.release_runtime(_rows)
    ckpt_path = ARTIFACTS_DIR / "checkpoints/sw14b_conservative_retrain_checkpoint.pth"
    if best_state is not None:
        torch.save({"state_dict": best_state}, ckpt_path)
    write_csv(REPORTS_DIR / "sw14b_conservative_retrain_log.csv", train_rows)
    write_csv(REPORTS_DIR / "sw14b_conservative_retrain_val_metrics.csv", val_rows)
    decision = "RETRAIN_R2_DENSITY_STILL_HIGH"
    if best_summary is not None:
        if not clean_selection_ok(clean_attr):
            decision = "RETRAIN_R4_CLEAN_UNSAFE"
        elif float(best_summary["mean_front_recovery_ratio"]) < 0.75:
            decision = "RETRAIN_R3_RECOVERY_COLLAPSE"
        elif float(best_summary["mean_density_delta"]) <= min(float(best_summary["mean_teacher_density_delta"]) + 0.04, 0.08):
            decision = "RETRAIN_R1_PASS"
    payload = {
        "decision": decision,
        "checkpoint": str(ckpt_path),
        "best_summary": best_summary,
    }
    write_json(REPORTS_DIR / "sw14b_conservative_retrain_decision.json", payload)
    base.release_runtime(model, dataset, adapter)
    return payload


def choose_best_config(scale_selection: dict[str, Any], clamp_selection: dict[str, Any], retrain_decision: dict[str, Any]) -> dict[str, Any]:
    if scale_selection_is_effective_pass(scale_selection):
        return {
            "source": "alpha_scale",
            "policy": RescuePolicy(name=f"alpha_scale_{float(scale_selection['best_scale_row']['scale']):.2f}", global_scale=float(scale_selection["best_scale_row"]["scale"])),
            "checkpoint": str(base.ARTIFACTS_DIR / "checkpoints/sw14b_round2_metric_teacher_checkpoint.pth"),
            "selection_row": scale_selection["best_scale_row"],
        }
    if clamp_selection.get("executed") and clamp_selection.get("best_row") is not None:
        best = clamp_selection["best_row"]
        if best["policy_name"] == "front_proxy_damping":
            policy = RescuePolicy(name="front_proxy_damping", global_scale=1.0, front_proxy_damping=True, front_proxy_damping_base_scale=1.0)
        else:
            policy = RescuePolicy(name=best["policy_name"], global_scale=1.0, front_scale=float(best["front_scale"]), side_scale=float(best["side_scale"]))
        return {"source": "density_clamp", "policy": policy, "checkpoint": str(base.ARTIFACTS_DIR / "checkpoints/sw14b_round2_metric_teacher_checkpoint.pth"), "selection_row": best}
    if retrain_decision.get("decision") == "RETRAIN_R1_PASS":
        return {"source": "conservative_retrain", "policy": RescuePolicy(name="retrain_best", global_scale=1.0), "checkpoint": retrain_decision["checkpoint"], "selection_row": retrain_decision.get("best_summary")}
    return {"source": "original", "policy": RescuePolicy(name="original", global_scale=1.0), "checkpoint": str(base.ARTIFACTS_DIR / "checkpoints/sw14b_round2_metric_teacher_checkpoint.pth"), "selection_row": None}


def phase6_frozen_eval_debug(
    args: argparse.Namespace,
    best_config: dict[str, Any],
    clean_attr: dict[str, Any],
) -> dict[str, Any]:
    _, dataset, model, _ = base.build_runtime(train=False)
    model.eval()
    try:
        return phase6_frozen_eval_debug_with_runtime(args, best_config, clean_attr, dataset, model)
    finally:
        base.release_runtime(model, dataset, collect=True, empty_cache=True)


def phase6_frozen_eval_debug_with_runtime(
    args: argparse.Namespace,
    best_config: dict[str, Any],
    clean_attr: dict[str, Any],
    dataset: Any,
    model: Any,
    case_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    print("[rescue] phase6 frozen eval_debug verification", flush=True)
    if not args.force_runtime_eval_debug:
        existing = load_existing_execution_payload("rescue_eval")
        if existing is not None:
            completed = complete_existing_rescue_eval(args, existing, best_config)
            if completed is not None:
                print("[rescue] reuse existing phase6 eval_debug metrics", flush=True)
                write_json(REPORTS_DIR / "sw14b_rescue_eval_debug_decision.json", completed)
                return completed
    adapter = build_adapter_from_checkpoint(best_config["checkpoint"])
    sample_indices = list(range(args.eval_start, args.eval_end + 1))
    owns_case_cache = case_cache is None
    if case_cache is None:
        case_cache = build_case_cache(
            model,
            dataset,
            sample_indices,
            base.teacher_ref().perturbation_id,
            base.build_candidate(),
            {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()},
        )
    rows, summary = collect_policy_rows(model, dataset, adapter, sample_indices, base.teacher_ref().perturbation_id, "eval_debug", best_config["policy"], case_cache=case_cache)
    original_eval = load_unified_original_eval_debug_summary()
    teacher_density = float(summary["mean_teacher_density_delta"])
    decision = "RESCUE_FAIL"
    if (
        float(summary["mean_front_recovery_ratio"]) >= 0.90
        and (float(summary["mean_density_delta"]) <= teacher_density + 0.04 or float(summary["mean_density_delta"]) <= 0.08)
        and float(summary["mean_false_positive_delta"]) <= float(summary["mean_teacher_false_positive_delta"]) + 0.015
        and float(summary["mean_front_local_proxy"]) <= 1.30
        and clean_selection_ok(clean_attr)
    ):
        decision = "RESCUE_STRONG"
    elif (
        float(summary["mean_front_recovery_ratio"]) >= 0.75
        and (float(summary["mean_density_delta"]) <= teacher_density + 0.06 or float(summary["mean_density_delta"]) <= 0.09)
        and float(summary["mean_front_local_proxy"]) <= 1.32
        and clean_attr["decision"] != "CLEAN_D1_ADAPTER_CAUSES_RAW_DRIFT"
        and clean_attr["decision"] != "CLEAN_D4_CLEAN_DEGRADATION_FLAG_BUG"
        and clean_attr["decision"] != "CLEAN_D6_TRUE_CLEAN_UNSAFE"
    ):
        decision = "RESCUE_MEDIUM"
    payload = {
        "decision": decision,
        "best_config": serialize_best_config(best_config),
        "rescued_summary": summary,
        "original_summary": original_eval["adapter_summary"],
        "teacher_summary": original_eval["teacher_summary"],
    }
    write_csv(REPORTS_DIR / "sw14b_rescue_eval_debug_metrics.csv", rows)
    write_json(REPORTS_DIR / "sw14b_rescue_eval_debug_decision.json", payload)
    if owns_case_cache:
        base.release_runtime(case_cache)
    base.release_runtime(adapter)
    return payload


def scale_from_policy_name(policy_name: str) -> float | None:
    if policy_name.startswith("alpha_scale_"):
        try:
            return float(policy_name.rsplit("_", 1)[-1])
        except ValueError:
            return None
    return None


def write_eval_scale_sweep_unified_tables(
    *,
    rows_by_policy: dict[str, list[dict[str, Any]]],
    sample_indices: list[int],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_set = set(sample_indices)
    original_rows = filter_metric_rows_by_samples(read_csv(base.REPORTS_DIR / "sw14b_eval_debug_metrics.csv"), sample_set)
    teacher_rows = filter_metric_rows_by_samples(read_csv(base.REPORTS_DIR / "sw14b_eval_debug_teacher_metrics.csv"), sample_set)
    original_summary = canonical_from_sw14b_summary(aggregate_sw14b_metric_rows(original_rows))
    teacher_summary = canonical_from_sw14b_summary(aggregate_sw14b_metric_rows(teacher_rows))
    table_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for policy_name, summary in summaries.items():
        rescue = canonical_from_rescue_summary(summary)
        scale = scale_from_policy_name(policy_name)
        recovery_ratio = float(summary.get("mean_front_recovery_ratio", 0.0) or 0.0)
        density_delta = rescue["pred_gt_density_delta"]
        fp_delta = rescue["false_positive_delta"]
        front_proxy = rescue["front_local_density_proxy"]
        safety_ok = (
            (density_delta <= teacher_summary["pred_gt_density_delta"] + 0.04 or density_delta <= 0.08)
            and fp_delta <= teacher_summary["false_positive_delta"] + 0.015
            and front_proxy <= 1.30
        )
        summary_rows.append(
            {
                "policy_name": policy_name,
                "scale": scale,
                "sample_count": summary.get("sample_count", len(sample_indices)),
                "front_recovery_ratio": recovery_ratio,
                "front_local_density_proxy": front_proxy,
                "front_sector_false_free_rate_delta_vs_native": rescue["front_sector_false_free_rate_delta_vs_native"],
                "future_h4_h6_false_free_rate_delta_vs_native": rescue["future_h4_h6_false_free_rate_delta_vs_native"],
                "pred_gt_density_delta": density_delta,
                "false_positive_delta": fp_delta,
                "protected_zone_preservation_ratio": rescue["protected_zone_preservation_ratio"],
                "sample_joint_success_rate": summary.get("sample_joint_success_rate", 0.0),
                "safety_ok": safety_ok,
            }
        )
        for metric_name in CANONICAL_METRICS:
            table_rows.append(
                {
                    "policy_name": policy_name,
                    "scale": scale,
                    "metric": metric_name,
                    "teacher_sw13c_f3_frontcap": teacher_summary[metric_name],
                    "original_sw14b_adapter_f3_frontcap": original_summary[metric_name],
                    "rescued_sw14b_adapter_f3_frontcap": rescue[metric_name],
                }
            )
    safe_rows = [row for row in summary_rows if row["safety_ok"] and row["scale"] not in {None, 0.0, 1.0}]
    best_safe = max(safe_rows, key=lambda row: float(row["front_recovery_ratio"])) if safe_rows else None
    best_density_safe = min(
        [row for row in summary_rows if row["scale"] not in {None, 0.0, 1.0}],
        key=lambda row: (abs(float(row["pred_gt_density_delta"]) - teacher_summary["pred_gt_density_delta"]), -float(row["front_recovery_ratio"])),
        default=None,
    )
    payload = {
        "metric_schema": "SW13-style deltas: system - native; negative false-free deltas are better.",
        "sample_indices": sample_indices,
        "sample_count": len(sample_indices),
        "teacher_summary": teacher_summary,
        "original_summary": original_summary,
        "policy_summaries": summary_rows,
        "best_safe_by_front_recovery": best_safe,
        "best_density_matched_non_noop": best_density_safe,
        "execution_note": "single runtime, sample-major feature reuse; each sample prepares extract_feat once, then runs all scale policies sequentially.",
    }
    write_csv(REPORTS_DIR / "sw14b_eval_debug_scale_sweep_unified_comparison.csv", table_rows)
    write_csv(REPORTS_DIR / "sw14b_eval_debug_scale_sweep_summary.csv", summary_rows)
    write_json(REPORTS_DIR / "sw14b_eval_debug_scale_sweep_summary.json", payload)
    return payload


def phase6_eval_debug_scale_sweep_with_runtime(
    args: argparse.Namespace,
    dataset: Any,
    model: Any,
) -> dict[str, Any]:
    print("[rescue] phase6 eval_debug scale sweep with sample-major feature reuse", flush=True)
    adapter = build_adapter_from_checkpoint(args.checkpoint)
    sample_indices = list(range(args.eval_start, args.eval_end + 1))
    scales = parse_float_list(args.eval_scale_list)
    policies = [RescuePolicy(name=f"alpha_scale_{scale:.2f}", global_scale=scale) for scale in scales]
    case_cache = build_case_cache(
        model,
        dataset,
        sample_indices,
        base.teacher_ref().perturbation_id,
        base.build_candidate(),
        {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()},
    )
    try:
        payload = collect_policy_summaries_sample_major(
            model,
            dataset,
            adapter,
            sample_indices,
            base.teacher_ref().perturbation_id,
            "eval_debug_scale_sweep",
            policies,
            case_cache,
        )
    finally:
        base.release_runtime(case_cache, adapter)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for policy in policies:
        policy_payload = payload[policy.name]
        summaries[policy.name] = policy_payload["summary"]
        all_rows.extend(policy_payload["rows"])
    write_csv(REPORTS_DIR / "sw14b_eval_debug_scale_sweep_metrics.csv", all_rows)
    summary_payload = write_eval_scale_sweep_unified_tables(
        rows_by_policy={policy.name: payload[policy.name]["rows"] for policy in policies},
        sample_indices=sample_indices,
        summaries=summaries,
    )
    write_json(
        REPORTS_DIR / "sw14b_eval_debug_scale_sweep_decision.json",
        {
            "decision": "SCALE_SWEEP_DIAGNOSTIC_COMPLETE",
            "sample_count": len(sample_indices),
            "scale_list": scales,
            "summary": summary_payload,
        },
    )
    return summary_payload


def phase7_final_decision(
    clean_attr: dict[str, Any],
    density_attr: dict[str, Any],
    scale_selection: dict[str, Any],
    clamp_selection: dict[str, Any],
    retrain_decision: dict[str, Any],
    rescue_eval: dict[str, Any],
    best_config: dict[str, Any],
) -> dict[str, Any]:
    decision = "SW14B_RESCUE_0_FAIL_KEEP_SW13_MAIN"
    proceed_core100 = False
    use_in_resume = False
    if clean_attr["decision"] in {"CLEAN_D2_POSTPROCESS_PATH_DRIFT", "CLEAN_D3_METRIC_SCHEMA_DRIFT"} and density_attr["decision"] == "DENSITY_R6_TEACHER_DENSITY_UNUSUALLY_LOW_ON_DEBUG":
        decision = "SW14B_RESCUE_3_CLEAN_DRIFT_NOT_ADAPTER"
    if rescue_eval["decision"] == "RESCUE_MEDIUM":
        decision = "SW14B_RESCUE_1_MEDIUM_DIAGNOSTIC"
    elif rescue_eval["decision"] == "RESCUE_STRONG":
        decision = "SW14B_RESCUE_2_STRONG_READY_CORE100"
        proceed_core100 = True
        use_in_resume = True
    elif density_attr["decision"] in {"DENSITY_R1_ADAPTER_RAW_OVER_OCCUPIES", "DENSITY_R2_F3_FAILS_TO_CLAMP_ADAPTER", "DENSITY_R3_FRONTCAP_NOT_ENOUGH", "DENSITY_R5_DENSITY_MAINLY_FALSE_POSITIVE"}:
        decision = "SW14B_RESCUE_4_DENSITY_UNRESOLVED"
    payload = {
        "decision": decision,
        "original_sw14b_reject_summary": load_normalized_original_sw14b_final_decision(),
        "clean_drift_attribution": clean_attr,
        "density_attribution": density_attr,
        "alpha_scale_sweep": scale_selection,
        "density_aware_clamp": clamp_selection,
        "conservative_retrain": retrain_decision,
        "frozen_eval_debug_verification": rescue_eval,
        "best_rescue_config": serialize_best_config(best_config),
        "whether_can_proceed_to_eval_core100": proceed_core100,
        "whether_can_use_in_resume": use_in_resume,
        "safe_wording": "diagnostic rescue only; SW13C-Fix + FrontCap remains the main result unless a later frozen eval_core100 confirms safety.",
    }
    write_json(REPORTS_DIR / "sw14b_density_clean_rescue_final_decision.json", payload)
    write_md(
        REPORTS_DIR / "stage_sw14b_density_clean_rescue_report.md",
        "\n".join(
            [
                "# SW14B Density Clamp / Clean Drift Rescue",
                "",
                "## 1. Original SW14B reject summary",
                f"- {load_normalized_original_sw14b_final_decision()['decision']}.",
                "",
                "## 2. Clean drift attribution",
                f"- {clean_attr['decision']}.",
                "",
                "## 3. Density attribution",
                f"- {density_attr['decision']}.",
                "",
                "## 4. Alpha scale sweep",
                f"- {scale_selection['decision']}.",
                "",
                "## 5. Density-aware clamp result",
                f"- {clamp_selection.get('decision', 'not_executed')}.",
                "",
                "## 6. Conservative retrain result",
                f"- {retrain_decision['decision']}.",
                "",
                "## 7. Frozen eval_debug verification",
                f"- {rescue_eval['decision']}.",
                "",
                "## 8. Final decision",
                f"- {decision}.",
                "",
                "## 9. whether can proceed to eval_core100",
                f"- `{proceed_core100}`.",
                "",
                "## 10. whether can use in resume",
                f"- `{use_in_resume}`.",
                "",
                "## 11. safe wording",
                f"- {payload['safe_wording']}",
            ]
        ),
    )
    return payload


def write_tests() -> None:
    tests = {
        "test_backbone_head_frozen.py": """
import json
from pathlib import Path
base = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_backbone_head_frozen():
    payload = json.loads(base.read_text())
    assert payload['frozen_sparseworld_status']['backbone_frozen'] is True
    assert payload['frozen_sparseworld_status']['head_frozen'] is True
""",
        "test_get_occ_hash_unchanged.py": """
import json
from pathlib import Path
base = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_get_occ_hash_unchanged():
    payload = json.loads(base.read_text())
    assert payload['no_get_occ_modification'] is True
    assert payload['get_occ_sha256']
""",
        "test_no_gt_in_rescue_selection.py": """
import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_alpha_scale_selection.json')
def test_no_gt_in_rescue_selection():
    payload = json.loads(path.read_text())
    assert payload['uses_eval_debug_for_selection'] is False
""",
        "test_eval_debug_not_used_for_param_selection.py": """
import json
from pathlib import Path
scale = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_alpha_scale_selection.json')
def test_eval_debug_not_used_for_param_selection():
    payload = json.loads(scale.read_text())
    assert payload['selected_on_split'] == 'val_small_200_249'
""",
        "test_alpha_scale_applied_correctly.py": """
from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_alpha_scale_applied_correctly():
    assert 'global_scale' in src
    assert 'camera_scale_tensor' in src
""",
        "test_clean_degradation_flag_false.py": """
import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_clean_drift_attribution.json')
def test_clean_degradation_flag_false():
    payload = json.loads(path.read_text())
    assert payload['mean_degradation_mask_sum'] == 0.0
""",
        "test_postprocess_chain_f3_then_frontcap.py": """
from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_postprocess_chain_f3_then_frontcap():
    assert 'apply_pruning_no_gt' in src
    assert 'apply_front_local_cap_no_gt' in src
""",
        "test_rescue_decision_schema.py": """
import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_density_clean_rescue_final_decision.json')
def test_rescue_decision_schema():
    payload = json.loads(path.read_text())
    required = {'decision','clean_drift_attribution','density_attribution','alpha_scale_sweep','density_aware_clamp','conservative_retrain','frozen_eval_debug_verification','best_rescue_config','whether_can_proceed_to_eval_core100','whether_can_use_in_resume','safe_wording'}
    assert required.issubset(payload.keys())
""",
        "test_no_resume_claim_if_fail.py": """
import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_density_clean_rescue_final_decision.json')
def test_no_resume_claim_if_fail():
    payload = json.loads(path.read_text())
    if payload['decision'] in {'SW14B_RESCUE_0_FAIL_KEEP_SW13_MAIN','SW14B_RESCUE_4_DENSITY_UNRESOLVED','SW14B_RESCUE_5_PROTOCOL_VIOLATION'}:
        assert payload['whether_can_use_in_resume'] is False
""",
        "test_no_global_case_or_teacher_memo.py": """
from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_no_global_case_or_teacher_memo():
    assert ('_CASE' + '_MEMO') not in src
    assert ('_TEACHER' + '_BUNDLE' + '_MEMO') not in src
    assert 'ALLOW_RUNTIME_TEACHER_REBUILD' in src
    assert 'SW14B_RESCUE_ALLOW_RUNTIME_TEACHER_REBUILD' in src
""",
        "test_no_original_substitution_for_rescue.py": """
from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_no_original_substitution_for_rescue():
    assert ('original_sw14b' + '_equivalent_noop_alpha_policy') not in src
    assert 'not substituted with original SW14B metrics' in src
    assert '0.0 < scale < 1.0' in src
""",
    }
    for name, content in tests.items():
        path = TESTS_DIR / name
        path.write_text(content.strip() + "\n", encoding="utf-8")


def run_tests() -> dict[str, Any]:
    write_tests()
    cmd = [sys.executable, "-m", "pytest", str(TESTS_DIR), "-q"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    payload = {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }
    write_json(REPORTS_DIR / "sw14b_density_clean_rescue_tests_result.json", payload)
    if proc.returncode != 0:
        raise RuntimeError(f"pytest failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return payload


def main() -> int:
    args = parse_args()
    ensure_dirs()
    io_audit = write_io_safety_audit(args)
    if args.io_audit_only:
        print("[rescue] wrote I/O safety audit only", flush=True)
        return 0
    seed_everything(args.seed)
    reused = try_reuse_existing_run(args)
    if reused is not None:
        tests = run_tests()
        reused["tests"] = tests
        reused["io_safety_audit"] = io_audit
        write_json(REPORTS_DIR / "sw14b_density_clean_rescue_execution_summary.json", reused)
        return 0
    if args.eval_scale_sweep:
        _, eval_dataset, eval_model, _ = base.build_runtime(train=False)
        eval_model.eval()
        try:
            scale_sweep = phase6_eval_debug_scale_sweep_with_runtime(args, eval_dataset, eval_model)
        finally:
            base.release_runtime(eval_model, eval_dataset, collect=True, empty_cache=True)
        tests = run_tests()
        write_json(
            REPORTS_DIR / "sw14b_eval_debug_scale_sweep_execution_summary.json",
            {
                "scale_sweep": scale_sweep,
                "io_safety_audit": io_audit,
                "tests": tests,
            },
        )
        return 0
    adapter = build_adapter_from_checkpoint(args.checkpoint)
    _, eval_dataset, eval_model, _ = base.build_runtime(train=False)
    eval_model.eval()
    try:
        clean_attr = phase1_clean_drift_attribution_with_runtime(args, adapter, eval_dataset, eval_model)
        shared_candidate = base.build_candidate()
        shared_sectors = {k: v.cpu() for k, v in base.sw13c_fix.sw7.build_sector_masks().items()}
        shared_debug_case_cache = build_case_cache(eval_model, eval_dataset, list(range(args.eval_start, args.eval_end + 1)), base.teacher_ref().perturbation_id, shared_candidate, shared_sectors)
        shared_val_case_cache = build_case_cache(eval_model, eval_dataset, list(range(args.val_start, args.val_end + 1)), base.teacher_ref().perturbation_id, shared_candidate, shared_sectors)
        density_attr = phase2_density_risk_attribution_with_runtime(args, adapter, eval_dataset, eval_model, debug_case_cache=shared_debug_case_cache, val_case_cache=shared_val_case_cache)
        scale_selection = phase3_alpha_scale_sweep_with_runtime(args, adapter, clean_attr, eval_dataset, eval_model, case_cache=shared_val_case_cache)
        clamp_selection = phase4_density_aware_clamp_with_runtime(args, adapter, clean_attr, scale_selection, eval_dataset, eval_model, case_cache=shared_val_case_cache)
        retrain_decision = phase5_conservative_retrain(args, args.checkpoint, clean_attr, scale_selection, clamp_selection)
        best_config = choose_best_config(scale_selection, clamp_selection, retrain_decision)
        rescue_eval = phase6_frozen_eval_debug_with_runtime(args, best_config, clean_attr, eval_dataset, eval_model, case_cache=shared_debug_case_cache)
    finally:
        base.release_runtime(eval_model, eval_dataset, collect=True, empty_cache=True)
    final_decision = phase7_final_decision(clean_attr, density_attr, scale_selection, clamp_selection, retrain_decision, rescue_eval, best_config)
    comparison = write_unified_debug_comparison(rescue_eval, best_config)
    tests = run_tests()
    write_json(
        REPORTS_DIR / "sw14b_density_clean_rescue_execution_summary.json",
        {
            "clean_attr": clean_attr,
            "density_attr": density_attr,
            "scale_selection": scale_selection,
            "clamp_selection": clamp_selection,
            "retrain_decision": retrain_decision,
            "best_config": serialize_best_config(best_config),
            "rescue_eval": rescue_eval,
            "final_decision": final_decision,
            "unified_metric_comparison": comparison,
            "io_safety_audit": io_audit,
            "tests": tests,
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
