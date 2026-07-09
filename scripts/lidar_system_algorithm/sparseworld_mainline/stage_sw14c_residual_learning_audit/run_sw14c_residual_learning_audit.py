from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SW14C_SCRIPT_DIR = SCRIPT_DIR.parent / "stage_sw14c_residual_teacher_adapter"
if str(SW14C_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SW14C_SCRIPT_DIR))

from sw14c_residual_audit_utils import (  # noqa: E402
    CORE_HORIZONS,
    EMPTY_IDX,
    binary_diff_counts,
    finite_mean,
    finite_std,
    metric_value,
    occ_mask,
    read_csv,
    read_json,
    run_postprocess_with_stages,
    safe_div,
    tensor_abs_stats,
    tensor_value_stats,
    write_csv,
    write_json,
    write_md,
)
from sw14c_residual_adapter_modules import (  # noqa: E402
    CAMERA_NAMES,
    FRONT_TRIPLET_INDICES,
    REAR_CAMERA_INDICES,
    SpatialResidualAdapter,
    build_sw13_r8_base_repair,
    count_parameters,
    front_triplet_degradation_mask,
)
import run_sw14c_residual_teacher_adapter as sw14c  # noqa: E402
from sw14c_losses import SW14CLossConfig  # noqa: E402


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_learning_audit"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14c_residual_learning_audit"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14c_residual_learning_audit"

SOURCE_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
SOURCE_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
ROUND2_CKPT = SOURCE_ARTIFACTS_DIR / "checkpoints/sw14c_round2_smoke_checkpoint.pth"
INIT_CKPT = SOURCE_ARTIFACTS_DIR / "checkpoints/sw14c_round1_init_equivalence_checkpoint.pth"
GAMMA_SELECTION = SOURCE_REPORTS_DIR / "sw14c_gamma_selection.json"
ROUND2_LOG = SOURCE_REPORTS_DIR / "sw14c_training_log.csv"
ROUND2_GAMMA_DETAIL = SOURCE_REPORTS_DIR / "sw14c_gamma_sweep_valsmall_metrics.csv"
GET_OCC_PATH = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/opus_head.py"


FINAL_ENUMS = {
    "SW14C_AUDIT_1_RESIDUAL_ZERO_COLLAPSE",
    "SW14C_AUDIT_2_GATE_OR_REG_TOO_STRONG",
    "SW14C_AUDIT_3_TEACHER_ERROR_TOO_SPARSE",
    "SW14C_AUDIT_4_RAW_CHANGED_BUT_POSTPROCESS_SWALLOWED",
    "SW14C_AUDIT_5_WRONG_REGION_BEHAVIOR",
    "SW14C_AUDIT_6_SIGNAL_ONLY_UNSAFE",
    "SW14C_AUDIT_7_READY_FOR_ROUND3_WITH_LOSS_ADJUST",
    "SW14C_AUDIT_8_STOP_SW14C_KEEP_SW13",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14C Round2 residual learning attribution audit")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=99)
    parser.add_argument("--val-start", type=int, default=100)
    parser.add_argument("--val-end", type=int, default=149)
    parser.add_argument("--selected-gamma", type=float, default=None)
    parser.add_argument("--gamma-candidates", default="0.0,0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="debug-only cap; default audits the full requested split")
    parser.add_argument("--skip-forward-if-existing", action="store_true", help="reuse existing new-stage audit files if all required outputs exist")
    parser.add_argument("--gc-interval", type=int, default=10, help="collect Python/CUDA garbage every N samples instead of every sample")
    parser.add_argument("--reuse-round2-gamma-metrics", action="store_true", default=True)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, ARTIFACTS_DIR, FIGURES_DIR, TESTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def selected_gamma_from_reports(args: argparse.Namespace) -> float:
    if args.selected_gamma is not None:
        return float(args.selected_gamma)
    payload = read_json(GAMMA_SELECTION)
    return float(payload["selected_gamma"])


def load_adapter(device: torch.device | str) -> tuple[SpatialResidualAdapter, dict[str, Any]]:
    payload = torch.load(ROUND2_CKPT, map_location=device, weights_only=False)
    adapter = sw14c.build_residual_adapter()
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    return adapter, payload


def release(*objs: Any, empty_cache: bool = False, collect: bool = False) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    if collect:
        gc.collect()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_mem_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": float(torch.cuda.memory_allocated() / (1024.0 * 1024.0)),
        "reserved_mb": float(torch.cuda.memory_reserved() / (1024.0 * 1024.0)),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
    }


def state_dict_param_diff_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trained_payload = torch.load(ROUND2_CKPT, map_location="cpu", weights_only=False)
    init_payload = torch.load(INIT_CKPT, map_location="cpu", weights_only=False)
    trained = trained_payload["state_dict"]
    init = init_payload["state_dict"]
    rows: list[dict[str, Any]] = []
    total_sq = 0.0
    total_n = 0
    max_abs = 0.0
    changed_count = 0
    for name, value in trained.items():
        init_value = init[name]
        diff = value.float() - init_value.float()
        sq = float((diff * diff).sum().item())
        n = int(diff.numel())
        l2 = float(sq**0.5)
        row_max = float(diff.abs().max().item()) if n else 0.0
        row_mean = float(diff.abs().mean().item()) if n else 0.0
        changed = row_max > 1e-12
        rows.append(
            {
                "parameter": name,
                "shape": list(value.shape),
                "numel": n,
                "l2_diff_vs_init": l2,
                "mean_abs_diff_vs_init": row_mean,
                "max_abs_diff_vs_init": row_max,
                "changed_vs_init": changed,
            }
        )
        total_sq += sq
        total_n += n
        max_abs = max(max_abs, row_max)
        changed_count += int(changed)
    summary = {
        "total_l2_diff_vs_init": float(total_sq**0.5),
        "rms_diff_vs_init": float((total_sq / max(1, total_n)) ** 0.5),
        "max_abs_diff_vs_init": max_abs,
        "changed_parameter_tensors": changed_count,
        "total_parameter_tensors": len(rows),
        "round2_checkpoint_meta": trained_payload.get("meta", {}),
        "init_checkpoint_meta": init_payload.get("meta", {}),
    }
    for label, predicate in {
        "delta_weight": lambda key: "delta.weight" in key,
        "delta_bias": lambda key: "delta.bias" in key,
        "gate_weight": lambda key: "gate.weight" in key,
        "gate_bias": lambda key: "gate.bias" in key,
        "camera_embedding": lambda key: "camera_embedding" in key,
    }.items():
        group = [row for row in rows if predicate(str(row["parameter"]))]
        summary[f"{label}_l2_diff_vs_init"] = float(sum(float(row["l2_diff_vs_init"]) ** 2 for row in group) ** 0.5) if group else 0.0
        summary[f"{label}_max_abs_diff_vs_init"] = max([float(row["max_abs_diff_vs_init"]) for row in group], default=0.0)
    return rows, summary


def phase1_checkpoint_sanity(selected_gamma: float) -> dict[str, Any]:
    rows, diff_summary = state_dict_param_diff_rows()
    write_csv(REPORTS_DIR / "sw14c_checkpoint_param_diff.csv", rows)
    adapter = SpatialResidualAdapter(channels_per_level=[256, 256, 256, 256], hidden_channels=32, camera_embed_dim=4)
    stats = count_parameters(adapter)
    load_success = True
    load_error = None
    try:
        trained_payload = torch.load(ROUND2_CKPT, map_location="cpu", weights_only=False)
        adapter.load_state_dict(trained_payload["state_dict"], strict=True)
    except Exception as exc:  # pragma: no cover - kept for audit payload
        load_success = False
        load_error = repr(exc)
        trained_payload = {}
    gamma_payload = read_json(GAMMA_SELECTION)
    gamma_file_ok = abs(float(gamma_payload.get("selected_gamma", -999.0)) - 0.1) < 1e-12
    gamma_eval_ok = abs(float(selected_gamma) - 0.1) < 1e-12
    current = [torch.randn(1, 6, 256, 2, 2) for _ in range(4)]
    memory = [level + 1.0 for level in current]
    base_levels = build_sw13_r8_base_repair(current, memory, "A10_drop_front_triplet")
    mask = front_triplet_degradation_mask("A10_drop_front_triplet", batch_size=1)
    clean_mask = front_triplet_degradation_mask("A0_clean", batch_size=1)
    with torch.inference_mode():
        final_levels, debug = adapter.apply_to_levels(current, memory, base_levels, mask, gamma=selected_gamma)
        clean_base = build_sw13_r8_base_repair(current, memory, "A0_clean")
        clean_final, clean_debug = adapter.apply_to_levels(current, memory, clean_base, clean_mask, gamma=selected_gamma)
    rear_max = max(float((final[:, REAR_CAMERA_INDICES] - base[:, REAR_CAMERA_INDICES]).abs().max().item()) for final, base in zip(final_levels, base_levels))
    clean_max = max(float((final - base).abs().max().item()) for final, base in zip(clean_final, clean_base))
    mask_bug = (
        mask.shape != (1, 6, 1)
        or float(mask[:, FRONT_TRIPLET_INDICES].sum().item()) != 3.0
        or float(mask[:, REAR_CAMERA_INDICES].sum().item()) != 0.0
        or float(clean_mask.sum().item()) != 0.0
        or rear_max > 1e-12
        or clean_max > 1e-12
    )
    trained_changed = float(diff_summary["total_l2_diff_vs_init"]) > 1e-8 and float(diff_summary["max_abs_diff_vs_init"]) > 1e-8
    if not load_success:
        decision = "CKPT_A5_LOAD_FAILURE"
    elif mask_bug:
        decision = "CKPT_A4_MASK_BUG"
    elif not gamma_file_ok or not gamma_eval_ok:
        decision = "CKPT_A3_GAMMA_NOT_APPLIED"
    elif not trained_changed:
        decision = "CKPT_A2_CHECKPOINT_EQUALS_INIT"
    else:
        decision = "CKPT_A1_TRAINED_PARAMS_CHANGED"
    payload = {
        "phase": "Phase 1 checkpoint / config sanity audit",
        "decision": decision,
        "checkpoint_exists": ROUND2_CKPT.exists(),
        "init_checkpoint_exists": INIT_CKPT.exists(),
        "checkpoint_path": str(ROUND2_CKPT),
        "init_checkpoint_path": str(INIT_CKPT),
        "checkpoint_load_success": load_success,
        "checkpoint_load_error": load_error,
        "adapter_trainable_params_count": int(stats.trainable_parameter_count),
        "loaded_adapter_params_different_from_init": trained_changed,
        "parameter_diff_summary": diff_summary,
        "residual_gate_bias_changed": diff_summary["gate_bias_max_abs_diff_vs_init"] > 1e-12,
        "residual_delta_conv_weight_changed": diff_summary["delta_weight_max_abs_diff_vs_init"] > 1e-12,
        "gamma_selection_file_selected_gamma": gamma_payload.get("selected_gamma"),
        "gamma_selection_is_0p1": gamma_file_ok,
        "eval_gamma_applied": selected_gamma,
        "eval_gamma_is_0p1": gamma_eval_ok,
        "degradation_mask_front_triplet_only": not mask_bug,
        "clean_degradation_mask_sum": float(clean_mask.sum().item()),
        "rear_cameras_residual_max_abs": rear_max,
        "clean_residual_max_abs": clean_max,
        "get_occ_sha256": sha256(GET_OCC_PATH),
        "training_performed_in_audit": False,
        "checkpoint_modified_in_audit": False,
    }
    write_json(REPORTS_DIR / "sw14c_checkpoint_sanity_audit.json", payload)
    return payload


def front_rear_stats(tensor: torch.Tensor) -> dict[str, float]:
    return {
        "front_triplet_mean_abs": float(tensor[:, FRONT_TRIPLET_INDICES].detach().float().abs().mean().item()),
        "rear_mean_abs": float(tensor[:, REAR_CAMERA_INDICES].detach().float().abs().mean().item()),
    }


def feature_delta_row(
    split_name: str,
    sample_index: int,
    level_idx: int,
    camera_idx: int | None,
    base_level: torch.Tensor,
    final_level: torch.Tensor,
    current_level: torch.Tensor,
    memory_level: torch.Tensor,
) -> dict[str, Any]:
    if camera_idx is None:
        base_sel = base_level
        final_sel = final_level
        current_sel = current_level
        memory_sel = memory_level
        camera_name = "ALL"
    else:
        base_sel = base_level[:, camera_idx]
        final_sel = final_level[:, camera_idx]
        current_sel = current_level[:, camera_idx]
        memory_sel = memory_level[:, camera_idx]
        camera_name = CAMERA_NAMES[camera_idx]
    delta = final_sel - base_sel
    base_norm = float(base_sel.detach().float().norm().item())
    current_memory_norm = float((memory_sel - current_sel).detach().float().norm().item())
    row = {
        "split": split_name,
        "sample_index": sample_index,
        "horizon_s": "feature",
        "level": level_idx,
        "camera_index": "all" if camera_idx is None else camera_idx,
        "camera_name": camera_name,
        **tensor_abs_stats(delta, ""),
        "relative_to_base_norm": safe_div(float(delta.detach().float().norm().item()), base_norm),
        "relative_to_current_memory_diff": safe_div(float(delta.detach().float().norm().item()), current_memory_norm),
        "front_triplet_feature_delta": float((final_level[:, FRONT_TRIPLET_INDICES] - base_level[:, FRONT_TRIPLET_INDICES]).detach().float().abs().mean().item()),
        "rear_feature_delta": float((final_level[:, REAR_CAMERA_INDICES] - base_level[:, REAR_CAMERA_INDICES]).detach().float().abs().mean().item()),
    }
    return row


def collect_residual_feature_stats(
    adapter: SpatialResidualAdapter,
    prepared: dict[str, Any],
    split_name: str,
    sample_index: int,
    selected_gamma: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor], list[torch.Tensor]]:
    with torch.inference_mode():
        final_levels, debug = adapter.apply_to_levels(
            prepared["frame_levels"][0],
            prepared["memory_levels"],
            prepared["base_levels"],
            prepared["degradation_mask"],
            gamma=selected_gamma,
        )
        clean_mask = front_triplet_degradation_mask(
            "A0_clean",
            batch_size=int(prepared["frame_levels"][0][0].shape[0]),
            device=prepared["frame_levels"][0][0].device,
            dtype=prepared["frame_levels"][0][0].dtype,
        )
        clean_base = build_sw13_r8_base_repair(prepared["frame_levels"][0], prepared["memory_levels"], "A0_clean")
        _, clean_debug = adapter.apply_to_levels(prepared["frame_levels"][0], prepared["memory_levels"], clean_base, clean_mask, gamma=selected_gamma)
    delta_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for level_idx, (current, memory, base_level, final_level) in enumerate(
        zip(prepared["frame_levels"][0], prepared["memory_levels"], prepared["base_levels"], final_levels)
    ):
        delta = debug[f"residual_delta_level{level_idx}"]
        gate = debug[f"residual_gate_level{level_idx}"]
        clean_delta = clean_debug[f"residual_delta_level{level_idx}"]
        for camera_idx, camera_name in enumerate(CAMERA_NAMES):
            delta_sel = delta[:, camera_idx]
            gate_sel = gate[:, camera_idx]
            delta_row = {
                "split": split_name,
                "sample_index": sample_index,
                "horizon_s": "feature",
                "level": level_idx,
                "camera_index": camera_idx,
                "camera_name": camera_name,
                **tensor_abs_stats(delta_sel, ""),
                **front_rear_stats(delta),
                "clean_mean_abs": float(clean_delta.abs().mean().item()),
            }
            gate_row = {
                "split": split_name,
                "sample_index": sample_index,
                "horizon_s": "feature",
                "level": level_idx,
                "camera_index": camera_idx,
                "camera_name": camera_name,
                **tensor_value_stats(gate_sel, ""),
                "front_triplet_gate_mean": float(gate[:, FRONT_TRIPLET_INDICES].mean().item()),
                "rear_gate_mean": float(gate[:, REAR_CAMERA_INDICES].mean().item()),
            }
            delta_rows.append(delta_row)
            gate_rows.append(gate_row)
            feature_rows.append(feature_delta_row(split_name, sample_index, level_idx, camera_idx, base_level, final_level, current, memory))
        feature_rows.append(feature_delta_row(split_name, sample_index, level_idx, None, base_level, final_level, current, memory))
    return delta_rows, gate_rows, feature_rows, debug, final_levels


def summarize_residual_magnitude(
    delta_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    front_delta = [float(row["mean_abs"]) for row in delta_rows if int(row["camera_index"]) in FRONT_TRIPLET_INDICES]
    rear_delta = [float(row["mean_abs"]) for row in delta_rows if int(row["camera_index"]) in REAR_CAMERA_INDICES]
    gate_front = [float(row["mean"]) for row in gate_rows if int(row["camera_index"]) in FRONT_TRIPLET_INDICES]
    gate_low = [float(row["saturated_low_ratio"]) for row in gate_rows if int(row["camera_index"]) in FRONT_TRIPLET_INDICES]
    feature_front = [float(row["front_triplet_feature_delta"]) for row in feature_rows if row["camera_index"] == "all"]
    feature_rear = [float(row["rear_feature_delta"]) for row in feature_rows if row["camera_index"] == "all"]
    clean = [float(row["clean_mean_abs"]) for row in delta_rows]
    max_rear_delta = max(rear_delta, default=0.0)
    max_rear_feature = max(feature_rear, default=0.0)
    max_clean = max(clean, default=0.0)
    mean_gate = finite_mean(gate_front) or 0.0
    mean_feature = finite_mean(feature_front) or 0.0
    mean_delta = finite_mean(front_delta) or 0.0
    if max_rear_delta > 1e-12 or max_rear_feature > 1e-12 or max_clean > 1e-12:
        decision = "RESMAG_R6_MASK_LEAKAGE"
    elif mean_delta < 1e-7:
        decision = "RESMAG_R1_RESIDUAL_ZERO_COLLAPSE"
    elif mean_gate < 0.01:
        decision = "RESMAG_R2_GATE_TOO_SMALL"
    elif mean_feature < 1e-7:
        decision = "RESMAG_R3_GAMMA_EFFECT_TOO_SMALL"
    elif mean_feature < 1e-5:
        decision = "RESMAG_R4_RESIDUAL_ACTIVE_BUT_NOT_EFFECTIVE"
    else:
        decision = "RESMAG_R5_RESIDUAL_MAGNITUDE_HEALTHY"
    return {
        "phase": "Phase 2 residual magnitude / gate audit",
        "decision": decision,
        "front_triplet_delta_mean_abs": mean_delta,
        "front_triplet_gate_mean": mean_gate,
        "front_triplet_gate_saturated_low_ratio_mean": finite_mean(gate_low),
        "gamma_applied_front_feature_delta_mean_abs": mean_feature,
        "rear_delta_max_abs_mean_proxy": max_rear_delta,
        "rear_feature_delta_max_abs_mean_proxy": max_rear_feature,
        "clean_delta_max_abs_mean_proxy": max_clean,
        "delta_rows": len(delta_rows),
        "gate_rows": len(gate_rows),
        "feature_rows": len(feature_rows),
    }


def error_region_row(
    split_name: str,
    sample_index: int,
    horizon_s: int,
    teacher_stage: dict[str, Any],
    sectors: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    final_semantic = teacher_stage["final_semantic"].long()
    gt_h = teacher_stage["gt_h"].long()
    teacher_occ = occ_mask(final_semantic)
    gt_occ = occ_mask(gt_h)
    gt_free = ~gt_occ
    teacher_fn = gt_occ & ~teacher_occ
    teacher_fp = gt_free & teacher_occ
    teacher_correct_occ = gt_occ & teacher_occ
    teacher_correct_free = gt_free & ~teacher_occ
    front_mask = sectors["front"].bool()
    protected = teacher_stage["protected_mask"].bool()
    raw_delta = teacher_stage["raw_delta_mask"].bool()
    near = sectors.get("near", torch.zeros_like(front_mask)).bool()
    mid = sectors.get("mid", torch.zeros_like(front_mask)).bool()
    far = sectors.get("far", torch.zeros_like(front_mask)).bool()
    fn_count = int(teacher_fn.sum().item())
    fp_count = int(teacher_fp.sum().item())
    gt_occ_count = int(gt_occ.sum().item())
    gt_free_count = int(gt_free.sum().item())
    row = {
        "split": split_name,
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "teacher_FN_count": fn_count,
        "teacher_FP_count": fp_count,
        "teacher_correct_occ_count": int(teacher_correct_occ.sum().item()),
        "teacher_correct_free_count": int(teacher_correct_free.sum().item()),
        "teacher_FN_rate": safe_div(fn_count, gt_occ_count),
        "teacher_FP_rate": safe_div(fp_count, gt_free_count),
        "front_teacher_FN_count": int((teacher_fn & front_mask).sum().item()),
        "future_h4h6_teacher_FN_count": int(fn_count if horizon_s in {4, 6} else 0),
        "teacher_FN_density_by_horizon": safe_div(fn_count, max(1, int(front_mask.numel()))),
        "protected_zone_teacher_FN_count": int((teacher_fn & protected).sum().item()),
        "raw_delta_teacher_FN_count": int((teacher_fn & raw_delta).sum().item()),
        "near_teacher_FN_count": int((teacher_fn & near).sum().item()),
        "mid_teacher_FN_count": int((teacher_fn & mid).sum().item()),
        "far_teacher_FN_count": int((teacher_fn & far).sum().item()),
        "dynamic_like_available": False,
        "dynamic_like_teacher_FN_count": None,
        "samples_with_zero_teacher_FN": fn_count == 0,
        "samples_with_high_teacher_FN": fn_count >= 1000,
        "teacher_error_sparsity_score": safe_div(fn_count + fp_count, int(gt_h.numel())),
    }
    masks = {
        "teacher_fn": teacher_fn,
        "teacher_fp": teacher_fp,
        "teacher_correct": teacher_correct_occ | teacher_correct_free,
        "teacher_correct_occ": teacher_correct_occ,
        "teacher_correct_free": teacher_correct_free,
    }
    return row, masks


def summarize_teacher_errors(rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    fn_counts = [int(row["teacher_FN_count"]) for row in rows]
    fp_counts = [int(row["teacher_FP_count"]) for row in rows]
    sample_totals: dict[int, int] = {}
    for row in rows:
        sample_totals.setdefault(int(row["sample_index"]), 0)
        sample_totals[int(row["sample_index"])] += int(row["teacher_FN_count"])
    zero_samples = sum(1 for total in sample_totals.values() if total == 0)
    high_samples = sum(1 for total in sample_totals.values() if total >= 1000)
    total_fn = sum(fn_counts)
    total_fp = sum(fp_counts)
    future_fn = sum(int(row["teacher_FN_count"]) for row in rows if int(row["horizon_s"]) in {4, 6})
    front_fn = sum(int(row["front_teacher_FN_count"]) for row in rows)
    if total_fn + total_fp == 0 or (sample_totals and zero_samples / len(sample_totals) > 0.50):
        decision = "ERR_R1_TEACHER_ERROR_TOO_SPARSE"
    elif sample_totals and high_samples <= max(1, int(0.10 * len(sample_totals))) and high_samples > 0:
        decision = "ERR_R2_TEACHER_FN_CONCENTRATED_FEW_SAMPLES"
    elif total_fp > total_fn * 1.5:
        decision = "ERR_R3_TEACHER_FP_DOMINANT"
    elif future_fn > total_fn * 0.60:
        decision = "ERR_R4_FUTURE_HORIZON_ERROR_DOMINANT"
    elif front_fn > 0:
        decision = "ERR_R5_FRONT_ERROR_HAS_LEARNABLE_SIGNAL"
    else:
        decision = "ERR_R6_ERROR_REGION_BALANCED"
    return {
        "split": split_name,
        "decision": decision,
        "row_count": len(rows),
        "sample_count": len(sample_totals),
        "teacher_FN_total": total_fn,
        "teacher_FP_total": total_fp,
        "teacher_FN_mean_per_row": finite_mean(fn_counts),
        "teacher_FP_mean_per_row": finite_mean(fp_counts),
        "front_teacher_FN_total": front_fn,
        "future_h4h6_teacher_FN_total": future_fn,
        "samples_with_zero_teacher_FN": zero_samples,
        "samples_with_high_teacher_FN": high_samples,
        "teacher_error_sparsity_score_mean": finite_mean([float(row["teacher_error_sparsity_score"]) for row in rows]),
    }


def behavior_row(
    sample_index: int,
    horizon_s: int,
    teacher_stage: dict[str, Any],
    student_stage: dict[str, Any],
    teacher_raw: dict[str, torch.Tensor],
    student_raw: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    sample_residual_proxy: dict[str, float],
) -> dict[str, Any]:
    teacher_fn = masks["teacher_fn"]
    teacher_fp = masks["teacher_fp"]
    teacher_correct = masks["teacher_correct"]
    student_final_occ = occ_mask(student_stage["final_semantic"])
    student_raw_occ = occ_mask(student_raw["raw_semantic"].detach().cpu())
    teacher_final_occ = occ_mask(teacher_stage["final_semantic"])
    raw_recovers_fn = teacher_fn & student_raw_occ
    final_recovers_fn = teacher_fn & student_final_occ
    suppresses_fp = teacher_fp & ~student_final_occ
    keeps_fp = teacher_fp & student_final_occ
    breaks_correct = teacher_correct & (student_final_occ != teacher_final_occ)
    def mean_on(mask: torch.Tensor, tensor: torch.Tensor) -> float | None:
        mask = mask.bool()
        if not bool(mask.any().item()):
            return None
        return float(tensor.detach().cpu().float()[mask].mean().item())
    teacher_conf = teacher_raw["teacher_confidence"].detach().cpu().float()
    teacher_margin = teacher_raw["teacher_margin"].detach().cpu().float()
    student_conf = student_raw["raw_confidence"].detach().cpu().float()
    student_margin = student_raw["raw_margin"].detach().cpu().float()
    teacher_fn_count = int(teacher_fn.sum().item())
    teacher_fp_count = int(teacher_fp.sum().item())
    teacher_correct_count = int(teacher_correct.sum().item())
    return {
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "student_recovers_teacher_FN_count": int(final_recovers_fn.sum().item()),
        "student_recovers_teacher_FN_rate": safe_div(int(final_recovers_fn.sum().item()), teacher_fn_count),
        "student_raw_recovers_teacher_FN_count": int(raw_recovers_fn.sum().item()),
        "student_still_FN_count": int((teacher_fn & ~student_final_occ).sum().item()),
        "student_conf_change_on_teacher_FN": None if teacher_fn_count == 0 else (mean_on(teacher_fn, student_conf) or 0.0) - (mean_on(teacher_fn, teacher_conf) or 0.0),
        "student_margin_change_on_teacher_FN": None if teacher_fn_count == 0 else (mean_on(teacher_fn, student_margin) or 0.0) - (mean_on(teacher_fn, teacher_margin) or 0.0),
        "residual_magnitude_on_teacher_FN": sample_residual_proxy["front_feature_delta_mean_abs"],
        "gate_mean_on_teacher_FN": sample_residual_proxy["front_gate_mean"],
        "student_suppresses_teacher_FP_count": int(suppresses_fp.sum().item()),
        "student_suppresses_teacher_FP_rate": safe_div(int(suppresses_fp.sum().item()), teacher_fp_count),
        "student_keeps_teacher_FP_count": int(keeps_fp.sum().item()),
        "student_additional_FP_count": int((~teacher_fp & ~occ_mask(teacher_stage["final_semantic"]) & student_final_occ & ~occ_mask(teacher_stage["gt_h"])).sum().item()),
        "residual_magnitude_on_teacher_FP": sample_residual_proxy["front_feature_delta_mean_abs"],
        "gate_mean_on_teacher_FP": sample_residual_proxy["front_gate_mean"],
        "student_breaks_teacher_correct_count": int(breaks_correct.sum().item()),
        "student_preserves_teacher_correct_rate": 1.0 - safe_div(int(breaks_correct.sum().item()), teacher_correct_count),
        "residual_magnitude_on_teacher_correct": sample_residual_proxy["front_feature_delta_mean_abs"],
        "raw_action_swallowed_by_postprocess_count": int((raw_recovers_fn & ~final_recovers_fn).sum().item()),
        "residual_region_alignment_method": "sample_front_triplet_feature_proxy_no_voxel_feature_alignment",
    }


def summarize_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recover_counts = [int(row["student_recovers_teacher_FN_count"]) for row in rows]
    raw_recover_counts = [int(row["student_raw_recovers_teacher_FN_count"]) for row in rows]
    suppress_counts = [int(row["student_suppresses_teacher_FP_count"]) for row in rows]
    break_counts = [int(row["student_breaks_teacher_correct_count"]) for row in rows]
    swallowed = [int(row["raw_action_swallowed_by_postprocess_count"]) for row in rows]
    total_recover = sum(recover_counts)
    total_raw_recover = sum(raw_recover_counts)
    total_suppress = sum(suppress_counts)
    total_break = sum(break_counts)
    if total_recover == 0 and total_suppress == 0 and total_raw_recover == 0:
        decision = "BEHAV_R1_NO_ACTION_ON_TEACHER_ERRORS"
    elif total_recover > 0 and total_break > total_recover:
        decision = "BEHAV_R2_RECOVERS_FN_BUT_ADDS_FP"
    elif total_suppress > 0 and total_recover == 0:
        decision = "BEHAV_R3_SUPPRESSES_FP_BUT_LOSES_RECALL"
    elif total_break > 0:
        decision = "BEHAV_R4_BREAKS_TEACHER_CORRECT"
    elif sum(swallowed) > 0 and total_raw_recover > total_recover:
        decision = "BEHAV_R6_ACTION_SWALLOWED_BY_POSTPROCESS"
    else:
        decision = "BEHAV_R5_PROMISING_ERROR_TARGETED_BEHAVIOR"
    return {
        "phase": "Phase 5 student behavior on teacher error regions",
        "decision": decision,
        "row_count": len(rows),
        "student_recovers_teacher_FN_total": total_recover,
        "student_raw_recovers_teacher_FN_total": total_raw_recover,
        "student_suppresses_teacher_FP_total": total_suppress,
        "student_breaks_teacher_correct_total": total_break,
        "raw_action_swallowed_by_postprocess_total": sum(swallowed),
        "mean_gate_on_teacher_FN_proxy": finite_mean([row["gate_mean_on_teacher_FN"] for row in rows]),
        "mean_residual_magnitude_on_teacher_FN_proxy": finite_mean([row["residual_magnitude_on_teacher_FN"] for row in rows]),
    }


def stagewise_row(
    sample_index: int,
    horizon_s: int,
    feature_proxy: dict[str, float],
    teacher_stage: dict[str, Any],
    student_stage: dict[str, Any],
    teacher_raw: dict[str, torch.Tensor],
    student_raw: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    teacher_raw_occ = occ_mask(teacher_raw["teacher_raw_semantic"])
    student_raw_occ = occ_mask(student_raw["raw_semantic"].detach().cpu())
    teacher_f3_occ = occ_mask(teacher_stage["f3_semantic"])
    student_f3_occ = occ_mask(student_stage["f3_semantic"])
    teacher_final_occ = occ_mask(teacher_stage["final_semantic"])
    student_final_occ = occ_mask(student_stage["final_semantic"])
    raw_counts = binary_diff_counts(student_raw_occ, teacher_raw_occ)
    f3_counts = binary_diff_counts(student_f3_occ, teacher_f3_occ)
    final_counts = binary_diff_counts(student_final_occ, teacher_final_occ)
    front_mask = masks.get("front_mask")
    gt_h = teacher_stage["gt_h"].long()
    teacher_fn_final = occ_mask(gt_h) & ~teacher_final_occ
    student_fn_final = occ_mask(gt_h) & ~student_final_occ
    teacher_fp_final = ~occ_mask(gt_h) & teacher_final_occ
    student_fp_final = ~occ_mask(gt_h) & student_final_occ
    row = {
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "feature_diff_base_vs_student": feature_proxy["front_feature_delta_mean_abs"],
        "front_camera_feature_diff": feature_proxy["front_feature_delta_mean_abs"],
        "rear_camera_feature_diff": feature_proxy["rear_feature_delta_mean_abs"],
        "raw_occ_diff_count": raw_counts["occ_diff_count"],
        "raw_occ_jaccard": raw_counts["occ_jaccard"],
        "raw_semantic_abs_diff_mean": float((student_raw["raw_semantic"].detach().cpu().long() - teacher_raw["teacher_raw_semantic"].long()).abs().float().mean().item()),
        "raw_confidence_diff_mean": float((student_raw["raw_confidence"].detach().cpu().float() - teacher_raw["teacher_confidence"].float()).abs().mean().item()),
        "raw_margin_diff_mean": float((student_raw["raw_margin"].detach().cpu().float() - teacher_raw["teacher_margin"].float()).abs().mean().item()),
        "raw_new_occupied_vs_teacher": raw_counts["new_occupied_vs_teacher"],
        "raw_removed_occupied_vs_teacher": raw_counts["removed_occupied_vs_teacher"],
        "f3_occ_diff_count": f3_counts["occ_diff_count"],
        "f3_occ_jaccard": f3_counts["occ_jaccard"],
        "f3_new_occupied_vs_teacher": f3_counts["new_occupied_vs_teacher"],
        "f3_removed_occupied_vs_teacher": f3_counts["removed_occupied_vs_teacher"],
        "f3_pruned_set_diff": int((student_stage["f3_pruned_mask"].bool() ^ teacher_stage["f3_pruned_mask"].bool()).sum().item()),
        "f3_density_delta_diff": float(student_stage["meta"]["f3_density_delta"] - teacher_stage["meta"]["f3_density_delta"]),
        "final_occ_diff_count": final_counts["occ_diff_count"],
        "final_occ_jaccard": final_counts["occ_jaccard"],
        "final_new_occupied_vs_teacher": final_counts["new_occupied_vs_teacher"],
        "final_removed_occupied_vs_teacher": final_counts["removed_occupied_vs_teacher"],
        "final_density_delta_diff": float(student_stage["meta"]["final_density_delta"] - teacher_stage["meta"]["final_density_delta"]),
        "final_front_fn_diff": int((student_fn_final & front_mask).sum().item() - (teacher_fn_final & front_mask).sum().item()) if front_mask is not None else None,
        "final_fp_diff": int(student_fp_final.sum().item() - teacher_fp_final.sum().item()),
    }
    return row


def summarize_stagewise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature = finite_mean([row["feature_diff_base_vs_student"] for row in rows]) or 0.0
    raw_diff = finite_mean([row["raw_occ_diff_count"] for row in rows]) or 0.0
    f3_diff = finite_mean([row["f3_occ_diff_count"] for row in rows]) or 0.0
    final_diff = finite_mean([row["final_occ_diff_count"] for row in rows]) or 0.0
    fp_diff = finite_mean([row["final_fp_diff"] for row in rows]) or 0.0
    if feature < 1e-7:
        decision = "STAGE_R1_NO_FEATURE_EFFECT"
    elif raw_diff == 0:
        decision = "STAGE_R2_RAW_UNCHANGED"
    elif raw_diff > 0 and f3_diff == 0:
        decision = "STAGE_R3_RAW_CHANGED_BUT_F3_SWALLOWED"
    elif f3_diff > 0 and final_diff == 0:
        decision = "STAGE_R4_F3_CHANGED_BUT_FRONTCAP_SWALLOWED"
    elif final_diff > 0 and fp_diff > 0:
        decision = "STAGE_R6_FINAL_CHANGED_UNSAFE"
    elif final_diff > 0:
        decision = "STAGE_R5_FINAL_CHANGED_BUT_METRIC_NEUTRAL"
    else:
        decision = "STAGE_R7_FINAL_CHANGED_PROMISING"
    return {
        "phase": "Phase 3 stage-wise output diff audit",
        "decision": decision,
        "row_count": len(rows),
        "feature_diff_base_vs_student_mean": feature,
        "raw_occ_diff_count_mean": raw_diff,
        "f3_occ_diff_count_mean": f3_diff,
        "final_occ_diff_count_mean": final_diff,
        "final_fp_diff_mean": fp_diff,
        "raw_occ_diff_count_total": int(sum(int(row["raw_occ_diff_count"]) for row in rows)),
        "f3_occ_diff_count_total": int(sum(int(row["f3_occ_diff_count"]) for row in rows)),
        "final_occ_diff_count_total": int(sum(int(row["final_occ_diff_count"]) for row in rows)),
    }


def gamma_metric_row(
    sample_index: int,
    horizon_s: int,
    gamma: float,
    teacher_stage: dict[str, Any],
    student_stage: dict[str, Any],
) -> dict[str, Any]:
    teacher_meta = teacher_stage["meta"]
    student_meta = student_stage["meta"]
    teacher_final = teacher_stage["final_semantic"]
    student_final = student_stage["final_semantic"]
    front_reduction = float(teacher_meta["final_front_false_free_delta"] - student_meta["final_front_false_free_delta"])
    future_reduction = float(teacher_meta["final_eval"]["false_free_rate"] - student_meta["final_eval"]["false_free_rate"]) if horizon_s in {4, 6} else 0.0
    density_delta_over_teacher = float(student_meta["final_density_delta"] - teacher_meta["final_density_delta"])
    fp_delta_over_teacher = float(student_meta["final_false_positive_delta"] - teacher_meta["final_false_positive_delta"])
    final_occ_diff = int((occ_mask(student_final) ^ occ_mask(teacher_final)).sum().item())
    front_local_proxy = float(student_meta["front_local_density_proxy_after"])
    safety_pass = density_delta_over_teacher <= 0.03 and fp_delta_over_teacher <= 0.01 and front_local_proxy <= 1.30
    joint_success = front_reduction > 0 and density_delta_over_teacher <= 0.03 and fp_delta_over_teacher <= 0.01 and front_local_proxy <= 1.30
    return {
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "gamma": gamma,
        "front_fn_reduction_over_teacher": front_reduction,
        "future_h4h6_fn_reduction_over_teacher": future_reduction,
        "density_delta_over_teacher": density_delta_over_teacher,
        "false_positive_delta_over_teacher": fp_delta_over_teacher,
        "front_local_proxy": front_local_proxy,
        "sample_joint_success": bool(joint_success),
        "final_occ_diff_vs_teacher": final_occ_diff,
        "safety_pass": bool(safety_pass),
        "gamma_zero_counts_as_improvement": False,
    }


def summarize_gamma(rows: list[dict[str, Any]], gamma_candidates: list[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for gamma in gamma_candidates:
        rows_g = [row for row in rows if abs(float(row["gamma"]) - float(gamma)) < 1e-12]
        summary_rows.append(
            {
                "gamma": gamma,
                "row_count": len(rows_g),
                "sample_count": len({int(row["sample_index"]) for row in rows_g}),
                "front_fn_reduction_over_teacher": finite_mean([row["front_fn_reduction_over_teacher"] for row in rows_g]),
                "future_h4h6_fn_reduction_over_teacher": finite_mean([row["future_h4h6_fn_reduction_over_teacher"] for row in rows_g if int(row["horizon_s"]) in {4, 6}]),
                "density_delta_over_teacher": finite_mean([row["density_delta_over_teacher"] for row in rows_g]),
                "false_positive_delta_over_teacher": finite_mean([row["false_positive_delta_over_teacher"] for row in rows_g]),
                "front_local_proxy": finite_mean([row["front_local_proxy"] for row in rows_g]),
                "sample_joint_success_rate": safe_div(sum(bool(row["sample_joint_success"]) for row in rows_g), len(rows_g)),
                "final_occ_diff_vs_teacher": finite_mean([row["final_occ_diff_vs_teacher"] for row in rows_g]),
                "safety_pass_rate": safe_div(sum(bool(row["safety_pass"]) for row in rows_g), len(rows_g)),
                "safety_pass_all": bool(rows_g) and all(bool(row["safety_pass"]) for row in rows_g),
                "allowed_as_improvement": abs(gamma) > 1e-12,
            }
        )
    nonzero = [row for row in summary_rows if bool(row["allowed_as_improvement"])]
    safe_signal = [
        row
        for row in nonzero
        if float(row["front_fn_reduction_over_teacher"] or 0.0) > 1e-7 and bool(row["safety_pass_all"])
    ]
    unsafe_signal = [
        row
        for row in nonzero
        if float(row["front_fn_reduction_over_teacher"] or 0.0) > 1e-7 and not bool(row["safety_pass_all"])
    ]
    max_abs_signal = max(abs(float(row["front_fn_reduction_over_teacher"] or 0.0)) for row in nonzero) if nonzero else 0.0
    if max_abs_signal <= 1e-7:
        decision = "GAMMA_A1_NO_SIGNAL_ANY_GAMMA"
    elif unsafe_signal and not safe_signal:
        decision = "GAMMA_A2_SIGNAL_ONLY_UNSAFE"
    elif safe_signal and max(float(row["gamma"]) for row in safe_signal) > 0.1:
        decision = "GAMMA_A3_SIGNAL_SAFE_AT_LARGER_GAMMA"
    elif safe_signal:
        decision = "GAMMA_A4_SELECTION_TOO_CONSERVATIVE"
    else:
        decision = "GAMMA_A5_FLAT_AROUND_ZERO"
    summary = {
        "phase": "Phase 7 gamma sensitivity audit",
        "decision": decision,
        "gamma_candidates": gamma_candidates,
        "max_abs_front_fn_signal_nonzero_gamma": max_abs_signal,
        "summary_rows": summary_rows,
        "gamma_zero_is_teacher_lower_bound": True,
        "gamma_zero_counts_as_improvement": False,
        "uses_eval_debug": False,
        "uses_eval_core100_or_core500": False,
    }
    return summary_rows, summary


def phase6_loss_dominance() -> dict[str, Any]:
    rows = read_csv(ROUND2_LOG)
    config = SW14CLossConfig()
    if not rows:
        write_csv(REPORTS_DIR / "sw14c_loss_dominance_audit.csv", [])
        payload = {"phase": "Phase 6 loss dominance audit", "decision": "LOSS_R6_LOSS_LOG_MISSING", "training_log": str(ROUND2_LOG)}
        write_json(REPORTS_DIR / "sw14c_loss_dominance_summary.json", payload)
        return payload
    terms = [
        ("teacher_correct_preserve", "gt_occ_loss", config.lambda_teacher_correct_preserve),
        ("teacher_fn_recover", "teacher_fn_recover_loss", config.lambda_teacher_fn_recover),
        ("future_recover", "future_recover_loss", config.lambda_teacher_fn_recover * 0.5),
        ("teacher_fp_suppress_proxy", "fp_loss", config.lambda_fp),
        ("density", "density_hinge_loss", config.lambda_density),
        ("residual_l1", "residual_l1", config.lambda_residual_l1),
        ("residual_smooth", "residual_smooth", config.lambda_residual_smooth),
        ("gate_sparse", "gate_sparse", config.lambda_gate_sparse),
    ]
    audit_rows: list[dict[str, Any]] = []
    weighted_means: dict[str, float] = {}
    for term_name, key, weight in terms:
        values = [float(row[key]) for row in rows if key in row and row[key] not in ("", None)]
        weighted = [float(v) * float(weight) for v in values]
        weighted_mean = finite_mean(weighted) or 0.0
        weighted_means[term_name] = weighted_mean
        audit_rows.append(
            {
                "loss_term": term_name,
                "log_column": key,
                "configured_weight": float(weight),
                "initial_value": values[0] if values else None,
                "final_value": values[-1] if values else None,
                "mean": finite_mean(values),
                "std": finite_std(values),
                "weighted_mean_contribution": weighted_mean,
                "gradient_proxy": None,
                "logged": bool(values),
            }
        )
    total_weighted = sum(weighted_means.values())
    for row in audit_rows:
        row["relative_contribution_to_total_loss"] = safe_div(float(row["weighted_mean_contribution"]), total_weighted)
    audit_rows.extend(
        [
            {
                "loss_term": "teacher_fp_suppress_config",
                "log_column": None,
                "configured_weight": float(config.lambda_teacher_fp_suppress),
                "initial_value": None,
                "final_value": None,
                "mean": None,
                "std": None,
                "weighted_mean_contribution": None,
                "relative_contribution_to_total_loss": None,
                "gradient_proxy": None,
                "logged": False,
                "note": "configured but not separately logged by Round2 compute_training_loss",
            },
            {
                "loss_term": "clean",
                "log_column": None,
                "configured_weight": float(config.lambda_clean),
                "logged": False,
                "note": "configured but not separately present in Round2 training log",
            },
            {
                "loss_term": "rear_consistency",
                "log_column": None,
                "configured_weight": float(config.lambda_rear_consistency),
                "logged": False,
                "note": "configured but not separately present in Round2 training log",
            },
        ]
    )
    write_csv(REPORTS_DIR / "sw14c_loss_dominance_audit.csv", audit_rows)
    reg_share = sum(
        float(row["relative_contribution_to_total_loss"] or 0.0)
        for row in audit_rows
        if row["loss_term"] in {"residual_l1", "residual_smooth", "gate_sparse"}
    )
    density_share = next((float(row["relative_contribution_to_total_loss"] or 0.0) for row in audit_rows if row["loss_term"] == "density"), 0.0)
    preserve_share = next((float(row["relative_contribution_to_total_loss"] or 0.0) for row in audit_rows if row["loss_term"] == "teacher_correct_preserve"), 0.0)
    fn_share = next((float(row["relative_contribution_to_total_loss"] or 0.0) for row in audit_rows if row["loss_term"] == "teacher_fn_recover"), 0.0)
    if reg_share > 0.30:
        decision = "LOSS_R1_REGULARIZATION_DOMINATES"
    elif fn_share < 0.05:
        decision = "LOSS_R2_TEACHER_FN_SIGNAL_TOO_WEAK"
    elif density_share > 0.30:
        decision = "LOSS_R3_DENSITY_PENALTY_DOMINATES"
    elif preserve_share > 0.50:
        decision = "LOSS_R4_PRESERVE_LOSS_DOMINATES"
    else:
        decision = "LOSS_R5_LOSS_BALANCED_BUT_NO_SIGNAL"
    payload = {
        "phase": "Phase 6 loss dominance audit",
        "decision": decision,
        "training_log": str(ROUND2_LOG),
        "row_count": len(rows),
        "regularization_relative_contribution": reg_share,
        "density_relative_contribution": density_share,
        "preserve_relative_contribution": preserve_share,
        "teacher_fn_relative_contribution": fn_share,
        "dominant_terms": sorted(
            [
                {
                    "loss_term": row["loss_term"],
                    "relative_contribution_to_total_loss": row.get("relative_contribution_to_total_loss"),
                }
                for row in audit_rows
                if row.get("relative_contribution_to_total_loss") is not None
            ],
            key=lambda item: float(item["relative_contribution_to_total_loss"]),
            reverse=True,
        )[:5],
        "forward_only_loss_recompute_performed": False,
        "training_performed_in_audit": False,
    }
    write_json(REPORTS_DIR / "sw14c_loss_dominance_summary.json", payload)
    return payload


def sample_residual_proxy_from_feature_rows(feature_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]], sample_index: int) -> dict[str, float]:
    sample_feature_rows = [row for row in feature_rows if int(row["sample_index"]) == sample_index and row["camera_index"] == "all"]
    sample_gate_rows = [row for row in gate_rows if int(row["sample_index"]) == sample_index and int(row["camera_index"]) in FRONT_TRIPLET_INDICES]
    return {
        "front_feature_delta_mean_abs": finite_mean([row["front_triplet_feature_delta"] for row in sample_feature_rows]) or 0.0,
        "rear_feature_delta_mean_abs": finite_mean([row["rear_feature_delta"] for row in sample_feature_rows]) or 0.0,
        "front_gate_mean": finite_mean([row["mean"] for row in sample_gate_rows]) or 0.0,
    }


def audit_teacher_cache_path(sample_index: int) -> Path:
    return ARTIFACTS_DIR / "runtime_teacher_cache/audit_A10_drop_front_triplet" / f"A10_drop_front_triplet__sample{sample_index:03d}.pt"


def load_teacher_bundle_if_available(sample_index: int) -> tuple[dict[str, Any] | None, str | None]:
    audit_path = audit_teacher_cache_path(sample_index)
    if audit_path.exists():
        return torch.load(audit_path, map_location="cpu", weights_only=False), "audit_stage_cache"
    return None, None


def get_teacher_bundle_cached(
    model: Any,
    prepared: dict[str, Any],
    sample_index: int,
    sectors: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], str]:
    cached, source = load_teacher_bundle_if_available(sample_index)
    if cached is not None and source is not None:
        return cached, source
    bundle = sw14c.build_self_teacher_bundle_for_prepared(model, prepared, sample_index, sectors)
    path = audit_teacher_cache_path(sample_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return bundle, "built_same_runtime_and_cached"


def existing_round2_gamma_rows(gamma_candidates: list[float], val_indices: list[int]) -> tuple[list[dict[str, Any]], set[float]]:
    if not ROUND2_GAMMA_DETAIL.exists():
        return [], set()
    wanted = {round(float(g), 12) for g in gamma_candidates}
    val_set = set(int(idx) for idx in val_indices)
    source_rows = read_csv(ROUND2_GAMMA_DETAIL)
    rows: list[dict[str, Any]] = []
    covered: set[float] = set()
    for row in source_rows:
        sample_index = int(row["sample_index"])
        if sample_index not in val_set:
            continue
        gamma = float(row["gamma"])
        if round(gamma, 12) not in wanted:
            continue
        horizon_s = int(row["horizon_s"])
        front_reduction = float(row["teacher_front_sector_false_free_rate_delta_vs_native"]) - float(row["front_sector_false_free_rate_delta_vs_native"])
        future_reduction = (
            float(row["teacher_future_h4_h6_false_free_rate_delta_vs_native"]) - float(row["future_h4_h6_false_free_rate_delta_vs_native"])
            if horizon_s in {4, 6}
            else 0.0
        )
        density_delta = float(row["pred_gt_density_delta"]) - float(row["teacher_pred_gt_density_delta"])
        fp_delta = float(row["false_positive_delta"]) - float(row["teacher_false_positive_delta"])
        front_proxy = float(row["front_local_density_proxy"])
        safety = density_delta <= 0.03 and fp_delta <= 0.01 and front_proxy <= 1.30
        rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "gamma": gamma,
                "front_fn_reduction_over_teacher": front_reduction,
                "future_h4h6_fn_reduction_over_teacher": future_reduction,
                "density_delta_over_teacher": density_delta,
                "false_positive_delta_over_teacher": fp_delta,
                "front_local_proxy": front_proxy,
                "sample_joint_success": bool(front_reduction > 0 and safety),
                "final_occ_diff_vs_teacher": int(float(row["teacher_gap_occ"])),
                "safety_pass": bool(safety),
                "gamma_zero_counts_as_improvement": False,
                "source": "round2_existing_gamma_sweep",
            }
        )
        covered.add(round(gamma, 12))
    return rows, covered


def process_splits(args: argparse.Namespace, adapter: SpatialResidualAdapter, selected_gamma: float, gamma_candidates: list[float]) -> dict[str, Any]:
    train_indices = list(range(args.train_start, args.train_end + 1))
    val_indices = list(range(args.val_start, args.val_end + 1))
    if args.max_samples_per_split is not None:
        train_indices = train_indices[: args.max_samples_per_split]
        val_indices = val_indices[: args.max_samples_per_split]
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    sw14c.base.freeze_sparseworld_modules(model)
    sectors = {key: value.cpu() for key, value in sw14c.base.sw13c_fix.sw7.build_sector_masks().items()}
    candidate = sw14c.base.build_candidate()
    all_delta_rows: list[dict[str, Any]] = []
    all_gate_rows: list[dict[str, Any]] = []
    all_feature_rows: list[dict[str, Any]] = []
    teacher_region_rows_train: list[dict[str, Any]] = []
    teacher_region_rows_val: list[dict[str, Any]] = []
    teacher_rank_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    gamma_detail_rows: list[dict[str, Any]] = []
    covered_gamma_values: set[float] = set()
    if args.reuse_round2_gamma_metrics:
        existing_gamma, covered_gamma_values = existing_round2_gamma_rows(gamma_candidates, val_indices)
        gamma_detail_rows.extend(existing_gamma)
        if existing_gamma:
            print(
                f"[sw14c-audit] reused Round2 gamma rows={len(existing_gamma)} "
                f"gammas={sorted(covered_gamma_values)}",
                flush=True,
            )
    start = time.time()
    for split_name, indices in [("train", train_indices), ("val", val_indices)]:
        for sample_pos, sample_index in enumerate(indices, start=1):
            sample_start = time.time()
            prepared = None
            teacher_bundle = None
            teacher_source = None
            try:
                prepared = sw14c.prepare_residual_feature_case(model, dataset, sample_index)
                delta_rows, gate_rows, feature_rows, _feature_debug, _final_levels = collect_residual_feature_stats(
                    adapter, prepared, split_name, sample_index, selected_gamma
                )
                all_delta_rows.extend(delta_rows)
                all_gate_rows.extend(gate_rows)
                all_feature_rows.extend(feature_rows)
                proxy = sample_residual_proxy_from_feature_rows(feature_rows, gate_rows, sample_index)
                teacher_bundle, teacher_source = get_teacher_bundle_cached(model, prepared, sample_index, sectors)
                sample_fn_total = 0
                sample_fp_total = 0
                teacher_stages: dict[int, dict[str, Any]] = {}
                masks_by_h: dict[int, dict[str, torch.Tensor]] = {}
                for horizon_s in CORE_HORIZONS:
                    teacher_h = teacher_bundle["by_horizon"][horizon_s]
                    teacher_stage = run_postprocess_with_stages(
                        sw14c.base,
                        raw_semantic=teacher_h["teacher_raw_semantic"],
                        raw_confidence=teacher_h["teacher_confidence"],
                        raw_margin=teacher_h["teacher_margin"],
                        native_semantic=teacher_h["native_semantic"],
                        gt_h=teacher_h["gt_h"],
                        gt0=teacher_h["gt0"],
                        candidate=candidate,
                        sample_index=sample_index,
                        horizon_s=horizon_s,
                        sectors=sectors,
                        agreement_map=teacher_h["agreement"],
                    )
                    teacher_stage["gt_h"] = teacher_h["gt_h"].long()
                    teacher_stage["gt0"] = teacher_h["gt0"].long()
                    teacher_stages[horizon_s] = teacher_stage
                    row, masks = error_region_row(split_name, sample_index, horizon_s, teacher_stage, sectors)
                    masks["front_mask"] = sectors["front"].bool()
                    masks_by_h[horizon_s] = masks
                    sample_fn_total += int(row["teacher_FN_count"])
                    sample_fp_total += int(row["teacher_FP_count"])
                    if split_name == "train":
                        teacher_region_rows_train.append(row)
                    else:
                        teacher_region_rows_val.append(row)
                teacher_rank_rows.append(
                    {
                        "split": split_name,
                        "sample_index": sample_index,
                        "teacher_FN_total": sample_fn_total,
                        "teacher_FP_total": sample_fp_total,
                        "teacher_error_total": sample_fn_total + sample_fp_total,
                    }
                )
                if split_name != "val":
                    print(
                        f"[sw14c-audit] {split_name} sample {sample_index} "
                        f"({sample_pos}/{len(indices)}) teacher={teacher_source} "
                        f"elapsed={time.time() - sample_start:.2f}s cuda={cuda_mem_mb()}",
                        flush=True,
                    )
                    continue
                student_by_gamma: dict[float, dict[int, dict[str, torch.Tensor]]] = {}
                forward_gammas = {
                    float(gamma)
                    for gamma in gamma_candidates
                    if round(float(gamma), 12) not in covered_gamma_values and abs(float(gamma)) >= 1e-12
                }
                if abs(float(selected_gamma)) >= 1e-12:
                    forward_gammas.add(float(selected_gamma))
                for gamma in sorted(forward_gammas):
                    per_h, debug = sw14c.run_residual_forward_prepared(model, adapter, prepared, gamma=float(gamma), training=False)
                    student_by_gamma[float(gamma)] = per_h
                    release(debug, empty_cache=False)
                for gamma in gamma_candidates:
                    if round(float(gamma), 12) in covered_gamma_values:
                        continue
                    for horizon_s in CORE_HORIZONS:
                        teacher_h = teacher_bundle["by_horizon"][horizon_s]
                        if abs(float(gamma)) < 1e-12:
                            student_stage = teacher_stages[horizon_s]
                        else:
                            student_raw = student_by_gamma[float(gamma)][horizon_s]
                            student_stage = run_postprocess_with_stages(
                                sw14c.base,
                                raw_semantic=student_raw["raw_semantic"],
                                raw_confidence=student_raw["raw_confidence"],
                                raw_margin=student_raw["raw_margin"],
                                native_semantic=teacher_h["native_semantic"],
                                gt_h=teacher_h["gt_h"],
                                gt0=teacher_h["gt0"],
                                candidate=candidate,
                                sample_index=sample_index,
                                horizon_s=horizon_s,
                                sectors=sectors,
                                agreement_map=teacher_h["agreement"],
                            )
                            student_stage["gt_h"] = teacher_h["gt_h"].long()
                        gamma_detail_rows.append(gamma_metric_row(sample_index, horizon_s, float(gamma), teacher_stages[horizon_s], student_stage))
                if selected_gamma in student_by_gamma:
                    selected_per_h = student_by_gamma[selected_gamma]
                elif abs(selected_gamma) < 1e-12:
                    selected_per_h = {}
                else:
                    selected_per_h, debug = sw14c.run_residual_forward_prepared(model, adapter, prepared, gamma=selected_gamma, training=False)
                    release(debug, empty_cache=False)
                for horizon_s in CORE_HORIZONS:
                    teacher_h = teacher_bundle["by_horizon"][horizon_s]
                    if abs(selected_gamma) < 1e-12:
                        selected_stage = teacher_stages[horizon_s]
                        selected_raw = {
                            "raw_semantic": teacher_h["teacher_raw_semantic"],
                            "raw_confidence": teacher_h["teacher_confidence"],
                            "raw_margin": teacher_h["teacher_margin"],
                        }
                    else:
                        selected_raw = selected_per_h[horizon_s]
                        selected_stage = run_postprocess_with_stages(
                            sw14c.base,
                            raw_semantic=selected_raw["raw_semantic"],
                            raw_confidence=selected_raw["raw_confidence"],
                            raw_margin=selected_raw["raw_margin"],
                            native_semantic=teacher_h["native_semantic"],
                            gt_h=teacher_h["gt_h"],
                            gt0=teacher_h["gt0"],
                            candidate=candidate,
                            sample_index=sample_index,
                            horizon_s=horizon_s,
                            sectors=sectors,
                            agreement_map=teacher_h["agreement"],
                        )
                        selected_stage["gt_h"] = teacher_h["gt_h"].long()
                    stage_rows.append(
                        stagewise_row(
                            sample_index,
                            horizon_s,
                            proxy,
                            teacher_stages[horizon_s],
                            selected_stage,
                            teacher_h,
                            selected_raw,
                            masks_by_h[horizon_s],
                        )
                    )
                    behavior_rows.append(
                        behavior_row(
                            sample_index,
                            horizon_s,
                            teacher_stages[horizon_s],
                            selected_stage,
                            teacher_h,
                            selected_raw,
                            masks_by_h[horizon_s],
                            proxy,
                        )
                    )
                release(student_by_gamma, empty_cache=False)
            finally:
                collect_now = args.gc_interval > 0 and sample_pos % int(args.gc_interval) == 0
                release(prepared, teacher_bundle, empty_cache=collect_now, collect=collect_now)
            print(
                f"[sw14c-audit] {split_name} sample {sample_index} "
                f"({sample_pos}/{len(indices)}) teacher={teacher_source} "
                f"elapsed={time.time() - sample_start:.2f}s cuda={cuda_mem_mb()}",
                flush=True,
            )
    write_csv(REPORTS_DIR / "sw14c_residual_magnitude_train.csv", [row for row in all_delta_rows if row["split"] == "train"])
    write_csv(REPORTS_DIR / "sw14c_residual_magnitude_val.csv", [row for row in all_delta_rows if row["split"] == "val"])
    write_csv(REPORTS_DIR / "sw14c_residual_gate_train.csv", [row for row in all_gate_rows if row["split"] == "train"])
    write_csv(REPORTS_DIR / "sw14c_residual_gate_val.csv", [row for row in all_gate_rows if row["split"] == "val"])
    write_csv(REPORTS_DIR / "sw14c_gamma_applied_feature_delta.csv", all_feature_rows)
    write_csv(REPORTS_DIR / "sw14c_stagewise_diff_val.csv", stage_rows)
    write_csv(REPORTS_DIR / "sw14c_teacher_error_regions_train.csv", teacher_region_rows_train)
    write_csv(REPORTS_DIR / "sw14c_teacher_error_regions_val.csv", teacher_region_rows_val)
    teacher_rank_rows = sorted(teacher_rank_rows, key=lambda row: int(row["teacher_error_total"]), reverse=True)
    write_csv(REPORTS_DIR / "sw14c_teacher_error_sample_rank.csv", teacher_rank_rows)
    write_csv(REPORTS_DIR / "sw14c_student_behavior_on_teacher_errors.csv", behavior_rows)
    write_csv(REPORTS_DIR / "sw14c_gamma_sensitivity_audit_detail.csv", gamma_detail_rows)
    resmag_summary = summarize_residual_magnitude(all_delta_rows, all_gate_rows, all_feature_rows)
    write_json(REPORTS_DIR / "sw14c_residual_magnitude_summary.json", resmag_summary)
    stage_summary = summarize_stagewise(stage_rows)
    write_json(REPORTS_DIR / "sw14c_stagewise_diff_summary.json", stage_summary)
    train_error_summary = summarize_teacher_errors(teacher_region_rows_train, "train")
    val_error_summary = summarize_teacher_errors(teacher_region_rows_val, "val")
    teacher_error_summary = {
        "phase": "Phase 4 teacher error region audit",
        "decision": val_error_summary["decision"],
        "train": train_error_summary,
        "val": val_error_summary,
        "uses_gt_eval_only": True,
        "uses_eval_debug": False,
    }
    write_json(REPORTS_DIR / "sw14c_teacher_error_region_summary.json", teacher_error_summary)
    behavior_summary = summarize_behavior(behavior_rows)
    write_json(REPORTS_DIR / "sw14c_student_error_region_behavior_summary.json", behavior_summary)
    gamma_summary_rows, gamma_summary = summarize_gamma(gamma_detail_rows, gamma_candidates)
    write_csv(REPORTS_DIR / "sw14c_gamma_sensitivity_audit.csv", gamma_summary_rows)
    write_json(REPORTS_DIR / "sw14c_gamma_sensitivity_summary.json", gamma_summary)
    plot_gamma(gamma_summary_rows)
    release(model, dataset, empty_cache=True)
    return {
        "elapsed_s": time.time() - start,
        "residual_magnitude": resmag_summary,
        "stagewise": stage_summary,
        "teacher_error": teacher_error_summary,
        "behavior": behavior_summary,
        "gamma": gamma_summary,
    }


def plot_gamma(summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    xs = [float(row["gamma"]) for row in summary_rows]
    front = [float(row["front_fn_reduction_over_teacher"] or 0.0) for row in summary_rows]
    fp = [float(row["false_positive_delta_over_teacher"] or 0.0) for row in summary_rows]
    density = [float(row["density_delta_over_teacher"] or 0.0) for row in summary_rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, front, marker="o", label="front FN reduction")
    plt.plot(xs, fp, marker="s", label="FP delta")
    plt.plot(xs, density, marker="^", label="density delta")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("gamma")
    plt.ylabel("delta vs SW13 teacher")
    plt.title("SW14C Round2 gamma sensitivity audit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14c_gamma_sensitivity_plot.png", dpi=180)
    plt.close()


def final_decision(
    checkpoint: dict[str, Any],
    resmag: dict[str, Any],
    stagewise: dict[str, Any],
    teacher_error: dict[str, Any],
    behavior: dict[str, Any],
    loss: dict[str, Any],
    gamma: dict[str, Any],
) -> dict[str, Any]:
    secondary: list[str] = []
    if checkpoint["decision"] in {"CKPT_A2_CHECKPOINT_EQUALS_INIT", "CKPT_A3_GAMMA_NOT_APPLIED", "CKPT_A4_MASK_BUG", "CKPT_A5_LOAD_FAILURE"}:
        primary = "SW14C_AUDIT_1_RESIDUAL_ZERO_COLLAPSE"
        secondary.append(checkpoint["decision"])
    elif resmag["decision"] == "RESMAG_R1_RESIDUAL_ZERO_COLLAPSE":
        primary = "SW14C_AUDIT_1_RESIDUAL_ZERO_COLLAPSE"
    elif resmag["decision"] in {"RESMAG_R2_GATE_TOO_SMALL", "RESMAG_R3_GAMMA_EFFECT_TOO_SMALL"}:
        primary = "SW14C_AUDIT_2_GATE_OR_REG_TOO_STRONG"
    elif teacher_error["decision"] == "ERR_R1_TEACHER_ERROR_TOO_SPARSE":
        primary = "SW14C_AUDIT_3_TEACHER_ERROR_TOO_SPARSE"
    elif stagewise["decision"] in {"STAGE_R3_RAW_CHANGED_BUT_F3_SWALLOWED", "STAGE_R4_F3_CHANGED_BUT_FRONTCAP_SWALLOWED"}:
        primary = "SW14C_AUDIT_4_RAW_CHANGED_BUT_POSTPROCESS_SWALLOWED"
    elif behavior["decision"] in {"BEHAV_R1_NO_ACTION_ON_TEACHER_ERRORS", "BEHAV_R4_BREAKS_TEACHER_CORRECT"}:
        primary = "SW14C_AUDIT_5_WRONG_REGION_BEHAVIOR"
    elif gamma["decision"] == "GAMMA_A2_SIGNAL_ONLY_UNSAFE":
        primary = "SW14C_AUDIT_6_SIGNAL_ONLY_UNSAFE"
    elif gamma["decision"] in {"GAMMA_A3_SIGNAL_SAFE_AT_LARGER_GAMMA", "GAMMA_A4_SELECTION_TOO_CONSERVATIVE"}:
        primary = "SW14C_AUDIT_7_READY_FOR_ROUND3_WITH_LOSS_ADJUST"
    else:
        primary = "SW14C_AUDIT_8_STOP_SW14C_KEEP_SW13"
    for cause in [
        resmag["decision"],
        stagewise["decision"],
        teacher_error["decision"],
        behavior["decision"],
        loss["decision"],
        gamma["decision"],
    ]:
        if cause not in secondary:
            secondary.append(cause)
    if primary in {"SW14C_AUDIT_1_RESIDUAL_ZERO_COLLAPSE", "SW14C_AUDIT_2_GATE_OR_REG_TOO_STRONG"}:
        recommended = (
            "Do not run eval_debug. Run Round2B only after changing the loss/gate setup: lower residual_l1, "
            "gate_sparse, and teacher_correct_preserve pressure; increase teacher_FN_recover focus; relax gate "
            "bias from -4 toward -3 or -2; expand gamma candidates under the same train-derived val protocol."
        )
        round3_allowed = False
        teacher_mining = False
        loss_adjust = True
    elif primary == "SW14C_AUDIT_3_TEACHER_ERROR_TOO_SPARSE":
        recommended = "Do teacher error mining and rebuild train samples around error-rich cases; do not blindly expand Round3."
        round3_allowed = False
        teacher_mining = True
        loss_adjust = False
    elif primary == "SW14C_AUDIT_4_RAW_CHANGED_BUT_POSTPROCESS_SWALLOWED":
        recommended = "Use a postprocess-aware surrogate and verify residual-created occupied voxels survive F3 and FrontCap before any eval_debug."
        round3_allowed = False
        teacher_mining = False
        loss_adjust = True
    elif primary == "SW14C_AUDIT_5_WRONG_REGION_BEHAVIOR":
        recommended = "Add teacher error region mask losses with stronger teacher-correct preserve and focused FN/FP objectives before any scale-up."
        round3_allowed = False
        teacher_mining = True
        loss_adjust = True
    elif primary == "SW14C_AUDIT_6_SIGNAL_ONLY_UNSAFE":
        recommended = "Add density-aware loss, FP suppression, and residual region gating; do not expand training yet."
        round3_allowed = False
        teacher_mining = False
        loss_adjust = True
    elif primary == "SW14C_AUDIT_7_READY_FOR_ROUND3_WITH_LOSS_ADJUST":
        recommended = "Round3 is allowed only with adjusted loss and another train-derived val gate before frozen eval_debug."
        round3_allowed = True
        teacher_mining = False
        loss_adjust = True
    else:
        recommended = "Stop SW14C as a main-result branch; keep SW13C-Fix + FrontCap as the main result and record SW14C as an internal exploration."
        round3_allowed = False
        teacher_mining = False
        loss_adjust = False
    evidence = {
        "checkpoint_decision": checkpoint["decision"],
        "parameter_l2_diff_vs_init": checkpoint["parameter_diff_summary"]["total_l2_diff_vs_init"],
        "gate_front_mean": resmag.get("front_triplet_gate_mean"),
        "gamma_applied_front_feature_delta_mean_abs": resmag.get("gamma_applied_front_feature_delta_mean_abs"),
        "raw_occ_diff_count_total": stagewise.get("raw_occ_diff_count_total"),
        "final_occ_diff_count_total": stagewise.get("final_occ_diff_count_total"),
        "val_teacher_FN_total": teacher_error["val"]["teacher_FN_total"],
        "student_recovers_teacher_FN_total": behavior.get("student_recovers_teacher_FN_total"),
        "loss_decision": loss["decision"],
        "gamma_decision": gamma["decision"],
    }
    payload = {
        "primary_cause": primary,
        "secondary_causes": secondary,
        "evidence": evidence,
        "recommended_next_action": recommended,
        "whether_run_eval_debug_now": False,
        "whether_round3_allowed": round3_allowed,
        "whether_teacher_error_mining_needed": teacher_mining,
        "whether_loss_adjustment_needed": loss_adjust,
        "whether_resume_allowed": False,
        "training_performed_in_audit": False,
        "checkpoint_modified_in_audit": False,
        "uses_eval_debug": False,
        "uses_eval_core100_or_core500": False,
        "final_claim_allowed": False,
        "subset_diagnostic_only": True,
    }
    assert payload["primary_cause"] in FINAL_ENUMS
    write_json(REPORTS_DIR / "sw14c_residual_learning_audit_decision.json", payload)
    return payload


def write_report(
    checkpoint: dict[str, Any],
    resmag: dict[str, Any],
    stagewise: dict[str, Any],
    teacher_error: dict[str, Any],
    behavior: dict[str, Any],
    loss: dict[str, Any],
    gamma: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Stage SW14C Residual Learning Audit Report",
        "",
        "This is a train-derived subset diagnostic for SW14C Round2 only. It does not train, tune, run eval_debug, run core100/core500, or alter SW13/SW14C main decisions.",
        "",
        "## Decisions",
        f"- checkpoint sanity: `{checkpoint['decision']}`",
        f"- residual magnitude / gate: `{resmag['decision']}`",
        f"- stage-wise diff: `{stagewise['decision']}`",
        f"- teacher error regions: `{teacher_error['decision']}`",
        f"- student behavior on teacher errors: `{behavior['decision']}`",
        f"- loss dominance: `{loss['decision']}`",
        f"- gamma sensitivity: `{gamma['decision']}`",
        f"- final diagnostic decision: `{decision['primary_cause']}`",
        "",
        "## Key Evidence",
        f"- adapter parameter L2 diff vs init: `{checkpoint['parameter_diff_summary']['total_l2_diff_vs_init']}`",
        f"- front gate mean: `{resmag.get('front_triplet_gate_mean')}`",
        f"- gamma-applied front feature delta mean abs: `{resmag.get('gamma_applied_front_feature_delta_mean_abs')}`",
        f"- raw occ diff total at selected gamma: `{stagewise.get('raw_occ_diff_count_total')}`",
        f"- final occ diff total at selected gamma: `{stagewise.get('final_occ_diff_count_total')}`",
        f"- val teacher FN total: `{teacher_error['val']['teacher_FN_total']}`",
        f"- student recovered teacher FN total: `{behavior.get('student_recovers_teacher_FN_total')}`",
        "",
        "## Protocol",
        "- no training performed in this audit",
        "- checkpoint read-only",
        "- SparseWorld backbone/head/get_occ unchanged",
        "- F3/FrontCap parameters unchanged",
        "- GT used only for evaluation-only diagnostic regions",
        "- gamma=0 is treated only as the SW13 teacher lower bound, not an improvement",
        "",
        "## Recommendation",
        decision["recommended_next_action"],
    ]
    write_md(REPORTS_DIR / "stage_sw14c_residual_learning_audit_report.md", "\n".join(lines) + "\n")


def all_required_outputs_exist() -> bool:
    names = [
        "sw14c_checkpoint_sanity_audit.json",
        "sw14c_checkpoint_param_diff.csv",
        "sw14c_residual_magnitude_train.csv",
        "sw14c_residual_magnitude_val.csv",
        "sw14c_residual_gate_train.csv",
        "sw14c_residual_gate_val.csv",
        "sw14c_gamma_applied_feature_delta.csv",
        "sw14c_stagewise_diff_val.csv",
        "sw14c_stagewise_diff_summary.json",
        "sw14c_teacher_error_regions_train.csv",
        "sw14c_teacher_error_regions_val.csv",
        "sw14c_teacher_error_region_summary.json",
        "sw14c_teacher_error_sample_rank.csv",
        "sw14c_student_behavior_on_teacher_errors.csv",
        "sw14c_student_error_region_behavior_summary.json",
        "sw14c_loss_dominance_audit.csv",
        "sw14c_loss_dominance_summary.json",
        "sw14c_gamma_sensitivity_audit.csv",
        "sw14c_gamma_sensitivity_summary.json",
        "sw14c_residual_learning_audit_decision.json",
        "stage_sw14c_residual_learning_audit_report.md",
    ]
    return all((REPORTS_DIR / name).exists() and (REPORTS_DIR / name).stat().st_size > 0 for name in names) and (
        FIGURES_DIR / "sw14c_gamma_sensitivity_plot.png"
    ).exists()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.skip_forward_if_existing and all_required_outputs_exist():
        print("[sw14c-audit] required outputs already exist; skip forward audit", flush=True)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sw14c.configure_cuda_for_throughput()
    selected_gamma = selected_gamma_from_reports(args)
    gamma_candidates = parse_float_list(args.gamma_candidates)
    checkpoint_summary = phase1_checkpoint_sanity(selected_gamma)
    loss_summary = phase6_loss_dominance()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter, _payload = load_adapter(device)
    forward_summary = process_splits(args, adapter, selected_gamma, gamma_candidates)
    decision = final_decision(
        checkpoint_summary,
        forward_summary["residual_magnitude"],
        forward_summary["stagewise"],
        forward_summary["teacher_error"],
        forward_summary["behavior"],
        loss_summary,
        forward_summary["gamma"],
    )
    write_report(
        checkpoint_summary,
        forward_summary["residual_magnitude"],
        forward_summary["stagewise"],
        forward_summary["teacher_error"],
        forward_summary["behavior"],
        loss_summary,
        forward_summary["gamma"],
        decision,
    )
    print(f"[sw14c-audit] final decision {decision['primary_cause']}", flush=True)


if __name__ == "__main__":
    main()
