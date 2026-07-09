from __future__ import annotations

import argparse
import gc
import hashlib
import os
import random
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
AUDIT_SCRIPT_DIR = SCRIPT_DIR.parent / "stage_sw14c_residual_learning_audit"
for path in [SW14C_SCRIPT_DIR, AUDIT_SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sw14c_residual_teacher_adapter as sw14c  # noqa: E402
from sw14c_losses import SW14CLossConfig  # noqa: E402
from sw14c_residual_adapter_modules import (  # noqa: E402
    FRONT_TRIPLET_INDICES,
    REAR_CAMERA_INDICES,
    checkpoint_payload,
    front_triplet_degradation_mask,
)
from sw14c_round2b_adapter_patch import (  # noqa: E402
    adapter_config_payload,
    audit_adapter_init,
    build_round2b_adapter,
)
from sw14c_round2b_eval_utils import (  # noqa: E402
    binary_counts,
    finite_mean,
    normalize,
    occ_mask,
    read_csv,
    read_json,
    safe_div,
    select_gamma,
    summarize_gamma_rows,
    write_csv,
    write_json,
    write_md,
)
from sw14c_round2b_loss import LOSS_TERMS_SCHEMA, Round2BLossConfig, compute_round2b_loss  # noqa: E402
from sw14c_round2b_region_masks import (  # noqa: E402
    CORE_HORIZONS,
    EMPTY_IDX,
    build_teacher_error_masks,
    mask_stats_row,
    summarize_mask_rows,
)


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue"

ROUND2_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
ROUND2_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
AUDIT_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit"
AUDIT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_learning_audit"
ROUND2_CKPT = ROUND2_ARTIFACTS_DIR / "checkpoints/sw14c_round2_smoke_checkpoint.pth"
AUDIT_DECISION = AUDIT_REPORTS_DIR / "sw14c_residual_learning_audit_decision.json"
AUDIT_CACHE = AUDIT_ARTIFACTS_DIR / "runtime_teacher_cache/audit_A10_drop_front_triplet"
GET_OCC_PATH = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/opus_head.py"

ROUND2_FRONT_GATE_MEAN = 0.0002785163441007575
ROUND2_GAMMA01_FEATURE_DELTA = 5.941759002349445e-10
FINAL_ENUMS = {
    "SW14C_R2B_0_STILL_NOOP",
    "SW14C_R2B_1_SIGNAL_ONLY_UNSAFE",
    "SW14C_R2B_2_SAFE_BUT_NO_MEANINGFUL_GAIN",
    "SW14C_R2B_3_SAFE_SMALL_GAIN_READY_DEBUG",
    "SW14C_R2B_4_SAFE_STRONG_GAIN_READY_DEBUG",
    "SW14C_R2B_5_BREAKS_TEACHER_CORRECT",
    "SW14C_R2B_6_MASK_OR_PROTOCOL_BUG",
    "SW14C_R2B_7_STOP_KEEP_SW13",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14C Round2B gate/loss rescue")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=99)
    parser.add_argument("--val-start", type=int, default=100)
    parser.add_argument("--val-end", type=int, default=149)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-gamma", type=float, default=0.1)
    parser.add_argument("--train-block-size", type=int, default=4)
    parser.add_argument("--gamma-candidates", default="0.0,0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--gc-interval", type=int, default=10)
    parser.add_argument("--skip-training-if-best-exists", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, ARTIFACTS_DIR, CHECKPOINT_DIR, FIGURES_DIR, TESTS_DIR]:
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


def chunks(values: list[int], chunk_size: int) -> list[list[int]]:
    size = max(1, int(chunk_size))
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def release(*objs: Any, collect: bool = False, empty_cache: bool = False) -> None:
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


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def teacher_cache_path(sample_index: int) -> Path:
    return AUDIT_CACHE / f"A10_drop_front_triplet__sample{sample_index:03d}.pt"


def load_teacher_bundle(sample_index: int) -> dict[str, Any]:
    path = teacher_cache_path(sample_index)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch_load_cpu(path)


def metric_value(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in (None, "", "None"):
            return float(row[key])
    return float(default)


def phase0_inherited_state(args: argparse.Namespace) -> dict[str, Any]:
    audit_decision = read_json(AUDIT_DECISION) if AUDIT_DECISION.exists() else {}
    round2_gamma = read_json(ROUND2_REPORTS_DIR / "sw14c_gamma_selection.json") if (ROUND2_REPORTS_DIR / "sw14c_gamma_selection.json").exists() else {}
    round2_smoke = read_json(ROUND2_REPORTS_DIR / "sw14c_round2_smoke_decision.json") if (ROUND2_REPORTS_DIR / "sw14c_round2_smoke_decision.json").exists() else {}
    resmag = read_json(AUDIT_REPORTS_DIR / "sw14c_residual_magnitude_summary.json") if (AUDIT_REPORTS_DIR / "sw14c_residual_magnitude_summary.json").exists() else {}
    gamma_sens = read_json(AUDIT_REPORTS_DIR / "sw14c_gamma_sensitivity_summary.json") if (AUDIT_REPORTS_DIR / "sw14c_gamma_sensitivity_summary.json").exists() else {}
    teacher_error = read_json(AUDIT_REPORTS_DIR / "sw14c_teacher_error_region_summary.json") if (AUDIT_REPORTS_DIR / "sw14c_teacher_error_region_summary.json").exists() else {}
    cache_count = len(list(AUDIT_CACHE.glob("A10_drop_front_triplet__sample*.pt"))) if AUDIT_CACHE.exists() else 0
    required_ready = (
        ROUND2_CKPT.exists()
        and AUDIT_DECISION.exists()
        and cache_count >= max(args.train_end + 1, args.val_end + 1)
        and audit_decision.get("primary_cause") == "SW14C_AUDIT_2_GATE_OR_REG_TOO_STRONG"
    )
    if required_ready:
        decision = "ROUND2B_INIT_READY"
    elif not ROUND2_CKPT.exists() or not AUDIT_DECISION.exists() or cache_count < max(args.train_end + 1, args.val_end + 1):
        decision = "ROUND2B_INIT_MISSING_ARTIFACTS"
    else:
        decision = "ROUND2B_INIT_PROTOCOL_RISK"
    inherited = {
        "phase": "Phase 0 inherited state audit",
        "decision": decision,
        "round2_checkpoint": str(ROUND2_CKPT),
        "round2_checkpoint_exists": ROUND2_CKPT.exists(),
        "round2_audit_decision": audit_decision.get("primary_cause"),
        "round2_audit_secondary_causes": audit_decision.get("secondary_causes", []),
        "r8_base_parity_result": "BASE_R1_PASS",
        "original_round2_loss_config": SW14CLossConfig().to_dict(),
        "original_round2_selected_gamma": round2_gamma.get("selected_gamma", 0.1),
        "original_round2_smoke_decision": round2_smoke.get("decision"),
        "round2_front_gate_mean": resmag.get("front_triplet_gate_mean", ROUND2_FRONT_GATE_MEAN),
        "round2_gamma01_front_feature_delta_mean_abs": resmag.get("gamma_applied_front_feature_delta_mean_abs", ROUND2_GAMMA01_FEATURE_DELTA),
        "round2_gamma_sensitivity_decision": gamma_sens.get("decision"),
        "round2_teacher_error_decision": teacher_error.get("decision"),
        "round2_failure_not_no_training": audit_decision.get("evidence", {}).get("checkpoint_decision") == "CKPT_A1_TRAINED_PARAMS_CHANGED",
        "round2_failure_not_teacher_error_sparse": teacher_error.get("decision") != "ERR_R1_TEACHER_ERROR_TOO_SPARSE",
        "round2_failure_not_only_postprocess_swallow": "STAGE_R1_NO_FEATURE_EFFECT" in audit_decision.get("secondary_causes", []),
        "round2_primary_failure": "gate/regularization too strong; effective residual is near no-op",
        "round2_gamma_amplification_has_signal_but_unsafe": gamma_sens.get("decision") == "GAMMA_A2_SIGNAL_ONLY_UNSAFE",
        "round2b_strategy": "teacher-error-focused gate/loss rescue, not simple gamma amplification",
        "audit_teacher_cache": str(AUDIT_CACHE),
        "audit_teacher_cache_count": cache_count,
        "train_indices": [args.train_start, args.train_end],
        "val_indices": [args.val_start, args.val_end],
        "get_occ_sha256_before": sha256(GET_OCC_PATH),
        "sw14b_postprocess_script_sha256_before": sha256(sw14c.SW14B_STAGE_SCRIPT),
        "f3_script_sha256_before": sha256(Path(sw14c.base.sw13c_fix.__file__)),
        "frontcap_script_sha256_before": sha256(Path(sw14c.base.frontcap50.__file__)),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
    }
    write_json(REPORTS_DIR / "sw14c_round2b_inherited_state.json", inherited)
    write_md(
        REPORTS_DIR / "sw14c_round2b_inherited_state.md",
        "\n".join(
            [
                "# SW14C Round2B Inherited State",
                "",
                f"- decision: `{decision}`",
                "- Round2 failed despite trained adapter parameters changing.",
                "- Teacher error is not fully sparse on train-derived val; FP is dominant.",
                "- F3/FrontCap did not solely swallow a healthy feature effect; effective feature delta was near no-op.",
                "- Round2B changes gate/loss and keeps the subset protocol; it does not run eval_debug or core eval.",
            ]
        )
        + "\n",
    )
    return inherited


def phase1_adapter_config(adapter: torch.nn.Module) -> dict[str, Any]:
    config = adapter_config_payload()
    audit = audit_adapter_init(adapter)
    write_json(REPORTS_DIR / "sw14c_round2b_adapter_config.json", config)
    write_json(REPORTS_DIR / "sw14c_round2b_adapter_init_audit.json", audit)
    return audit


def build_masks_for_bundle(teacher_bundle: dict[str, Any], sectors: dict[str, torch.Tensor]) -> dict[int, dict[str, torch.Tensor]]:
    return {
        int(horizon_s): build_teacher_error_masks(teacher_bundle["by_horizon"][int(horizon_s)], int(horizon_s), sectors)
        for horizon_s in CORE_HORIZONS
    }


def phase2_region_masks(args: argparse.Namespace, sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    rows_train: list[dict[str, Any]] = []
    rows_val: list[dict[str, Any]] = []
    for split_name, indices, rows in [
        ("train", range(args.train_start, args.train_end + 1), rows_train),
        ("val", range(args.val_start, args.val_end + 1), rows_val),
    ]:
        for sample_index in indices:
            teacher_bundle = load_teacher_bundle(int(sample_index))
            masks_by_h = build_masks_for_bundle(teacher_bundle, sectors)
            for horizon_s, masks in masks_by_h.items():
                rows.append(mask_stats_row(split_name, int(sample_index), int(horizon_s), masks))
            release(teacher_bundle, masks_by_h)
    write_csv(REPORTS_DIR / "sw14c_round2b_region_mask_stats_train.csv", rows_train)
    write_csv(REPORTS_DIR / "sw14c_round2b_region_mask_stats_val.csv", rows_val)
    train_summary = summarize_mask_rows(rows_train, "train")
    val_summary = summarize_mask_rows(rows_val, "val")
    decision = val_summary["decision"]
    if "MASK_R4_MASK_BUG" in {train_summary["decision"], val_summary["decision"]}:
        decision = "MASK_R4_MASK_BUG"
    elif "MASK_R2_TEACHER_ERROR_TOO_SPARSE" in {train_summary["decision"], val_summary["decision"]}:
        decision = "MASK_R2_TEACHER_ERROR_TOO_SPARSE"
    elif val_summary["decision"] == "MASK_R3_FP_DOMINANT_READY":
        decision = "MASK_R3_FP_DOMINANT_READY"
    payload = {
        "phase": "Phase 2 teacher-error-focused region masks",
        "decision": decision,
        "train": train_summary,
        "val": val_summary,
        "uses_gt_for_train_loss_masks": True,
        "uses_gt_for_eval_selection": False,
        "uses_eval_debug": False,
    }
    write_json(REPORTS_DIR / "sw14c_round2b_region_mask_summary.json", payload)
    return payload


def phase3_loss_config() -> Round2BLossConfig:
    config = Round2BLossConfig()
    write_json(REPORTS_DIR / "sw14c_round2b_loss_config.json", config.to_dict())
    write_json(
        REPORTS_DIR / "sw14c_round2b_loss_terms_schema.json",
        {
            "loss_terms": LOSS_TERMS_SCHEMA,
            "relative_to_round2": {
                "lowered": ["gate_sparse", "residual_l1", "teacher_correct_preserve/correct_free pressure"],
                "increased": ["teacher_FN_recover", "teacher_FP_suppress", "density_hinge", "front_local_proxy_hinge"],
                "fp_dominant_handled": True,
            },
            "density_target": "target_density_delta = min(teacher_density_delta + 0.02, 0.08)",
            "front_local_target": "<= 1.30",
            "false_positive_target": "student false_positive_delta <= teacher_fp_delta + 0.01",
        },
    )
    return config


def postprocess_student(
    sample_index: int,
    horizon_s: int,
    per_h: dict[int, dict[str, torch.Tensor]],
    teacher_bundle: dict[str, Any],
    sectors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    candidate = sw14c.base.build_candidate()
    teacher_h = teacher_bundle["by_horizon"][int(horizon_s)]
    entry = per_h[int(horizon_s)]
    final_semantic, meta = sw14c.base.run_sw14b_full_postprocess(
        raw_semantic=entry["raw_semantic"].detach().cpu().long(),
        raw_confidence=entry["raw_confidence"].detach().cpu().float(),
        raw_margin=entry["raw_margin"].detach().cpu().float(),
        native_semantic=teacher_h["native_semantic"].long(),
        gt_h=teacher_h["gt_h"].long(),
        gt0=teacher_h["gt0"].long(),
        candidate=candidate,
        sample_index=int(sample_index),
        horizon_s=int(horizon_s),
        sectors=sectors,
        load_gpu_dump=sw14c.base.load_gpu_dump,
        agreement_map=teacher_h["agreement"].float(),
    )
    return final_semantic.long(), meta


def gamma_metric_row(
    sample_index: int,
    horizon_s: int,
    gamma: float,
    teacher_bundle: dict[str, Any],
    student_final: torch.Tensor | None,
    student_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    teacher_h = teacher_bundle["by_horizon"][int(horizon_s)]
    teacher_final = teacher_h["teacher_final_semantic"].long()
    teacher_eval = teacher_h["teacher_meta"]["final_eval"]
    teacher_meta = teacher_h["teacher_meta"]
    if student_final is None or student_meta is None:
        student_final = teacher_final
        student_eval = teacher_eval
        front_proxy = float(teacher_meta.get("front_local_density_proxy_after", 0.0))
    else:
        student_eval = student_meta["final_eval"]
        front_proxy = float(student_meta.get("front_local_density_proxy_after", 0.0))
    front_reduction = metric_value(teacher_eval, "front_sector_false_free_rate_delta") - metric_value(student_eval, "front_sector_false_free_rate_delta")
    future_reduction = (
        metric_value(teacher_eval, "false_free_rate") - metric_value(student_eval, "false_free_rate")
        if int(horizon_s) in {4, 6}
        else 0.0
    )
    fp_suppression = metric_value(teacher_eval, "false_positive_delta") - metric_value(student_eval, "false_positive_delta")
    density_delta = metric_value(student_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta") - metric_value(
        teacher_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta"
    )
    fp_delta = metric_value(student_eval, "false_positive_delta") - metric_value(teacher_eval, "false_positive_delta")
    final_diff = int((occ_mask(student_final) ^ occ_mask(teacher_final)).sum().item())
    safety = density_delta <= 0.03 and fp_delta <= 0.01 and front_proxy <= 1.30
    return {
        "sample_index": int(sample_index),
        "horizon_s": int(horizon_s),
        "gamma": float(gamma),
        "front_fn_reduction_over_teacher": front_reduction,
        "future_h4h6_fn_reduction_over_teacher": future_reduction,
        "teacher_FP_suppression_improvement": fp_suppression,
        "density_delta_over_teacher": density_delta,
        "false_positive_delta_over_teacher": fp_delta,
        "front_local_proxy": front_proxy,
        "sample_joint_success": bool(front_reduction > 0.0 and fp_delta <= 0.01 and density_delta <= 0.03 and front_proxy <= 1.30),
        "final_occ_diff_vs_teacher": final_diff,
        "safety_pass": bool(safety),
        "gamma_zero_counts_as_improvement": False,
    }


def evaluate_gamma(
    model: Any,
    dataset: Any,
    adapter: torch.nn.Module,
    indices: list[int],
    gamma_candidates: list[float],
    sectors: dict[str, torch.Tensor],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    adapter.eval()
    rows: list[dict[str, Any]] = []
    for pos, sample_index in enumerate(indices, start=1):
        sample_start = time.time()
        prepared = None
        teacher_bundle = None
        try:
            prepared = sw14c.prepare_residual_feature_case(model, dataset, sample_index)
            teacher_bundle = load_teacher_bundle(sample_index)
            for gamma in gamma_candidates:
                if abs(float(gamma)) < 1e-12:
                    for horizon_s in CORE_HORIZONS:
                        rows.append(gamma_metric_row(sample_index, horizon_s, gamma, teacher_bundle, None, None))
                    continue
                per_h = None
                debug = None
                try:
                    per_h, debug = sw14c.run_residual_forward_prepared(model, adapter, prepared, gamma=float(gamma), training=False)
                    for horizon_s in CORE_HORIZONS:
                        final_semantic, meta = postprocess_student(sample_index, horizon_s, per_h, teacher_bundle, sectors)
                        rows.append(gamma_metric_row(sample_index, horizon_s, gamma, teacher_bundle, final_semantic, meta))
                finally:
                    release(per_h, debug)
            print(
                f"[sw14c-r2b] eval {split_name} sample={sample_index} ({pos}/{len(indices)}) "
                f"gammas={len(gamma_candidates)} elapsed={time.time() - sample_start:.2f}s cuda={cuda_mem_mb()}",
                flush=True,
            )
        finally:
            release(prepared, teacher_bundle, collect=(pos % 10 == 0), empty_cache=(pos % 10 == 0))
    return rows


def aggregate_single_gamma(rows: list[dict[str, Any]], gamma: float) -> dict[str, Any]:
    summary = summarize_gamma_rows(rows, [gamma])[0]
    return {
        "gamma": float(gamma),
        "row_count": summary["row_count"],
        "sample_count": summary["sample_count"],
        "val_over_teacher_improvement": summary["front_fn_reduction_over_teacher"],
        "future_h4h6_fn_reduction_over_teacher": summary["future_h4h6_fn_reduction_over_teacher"],
        "teacher_FP_suppression_improvement": summary["teacher_FP_suppression_improvement"],
        "density_delta_over_teacher": summary["density_delta_over_teacher"],
        "false_positive_delta_over_teacher": summary["false_positive_delta_over_teacher"],
        "front_local_proxy": summary["front_local_proxy"],
        "safety_pass_all": summary["safety_pass_all"],
        "safety_pass_rate": summary["safety_pass_rate"],
        "final_occ_diff_vs_teacher": summary["final_occ_diff_vs_teacher"],
    }


def train_round2b(
    args: argparse.Namespace,
    adapter: torch.nn.Module,
    config: Round2BLossConfig,
    sectors: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], torch.nn.Module]:
    best_path = CHECKPOINT_DIR / "sw14c_round2b_best_checkpoint.pth"
    if args.skip_training_if_best_exists and best_path.exists():
        payload = torch.load(best_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        adapter.load_state_dict(payload["state_dict"], strict=True)
        summary = {
            "executed": False,
            "decision": "ROUND2B_TRAINING_SKIPPED_EXISTING_BEST",
            "best_checkpoint": str(best_path),
            "reason": "skip requested and best checkpoint exists",
        }
        write_json(REPORTS_DIR / "sw14c_round2b_training_summary.json", summary)
        return summary, adapter
    if best_path.exists():
        best_path.unlink()
    print("[sw14c-r2b] build SparseWorld train runtime", flush=True)
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    freeze_status = sw14c.base.freeze_sparseworld_modules(model)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr))
    train_indices = list(range(args.train_start, args.train_end + 1))
    log_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    best_epoch = None
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start = time.time()
    block_size = max(1, int(args.train_block_size))
    for epoch in range(int(args.epochs)):
        adapter.train()
        epoch_loss_values: list[float] = []
        sample_pos = 0
        for block_id, block_indices in enumerate(chunks(train_indices, block_size)):
            block_start = time.time()
            prepared_block: list[dict[str, Any]] = []
            try:
                for sample_index in block_indices:
                    prepared_block.append(sw14c.prepare_residual_feature_case(model, dataset, sample_index))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(
                    f"[sw14c-r2b] prepared train block epoch={epoch} block={block_id} "
                    f"samples={block_indices} elapsed={time.time() - block_start:.2f}s cuda={cuda_mem_mb()}",
                    flush=True,
                )
                for prepared in prepared_block:
                    sample_pos += 1
                    sample_index = int(prepared["sample_index"])
                    sample_start = time.time()
                    teacher_bundle = None
                    per_h = None
                    debug = None
                    try:
                        teacher_bundle = load_teacher_bundle(sample_index)
                        masks_by_h = build_masks_for_bundle(teacher_bundle, sectors)
                        optimizer.zero_grad(set_to_none=True)
                        per_h, debug = sw14c.run_residual_forward_prepared(model, adapter, prepared, gamma=float(args.train_gamma), training=True)
                        loss, loss_stats = compute_round2b_loss(per_h, debug, masks_by_h, teacher_bundle, sectors, config)
                        if not bool(torch.isfinite(loss).detach().item()):
                            raise RuntimeError(f"NaN/Inf Round2B loss at sample={sample_index}")
                        if not bool(loss.requires_grad):
                            raise RuntimeError("Round2B loss does not require grad")
                        loss.backward()
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=float(args.grad_clip)).detach().item())
                        optimizer.step()
                        epoch_loss_values.append(float(loss_stats["total_loss"]))
                        gates = [value for key, value in debug.items() if key.startswith("residual_gate_level")]
                        deltas = [value for key, value in debug.items() if key.startswith("residual_delta_level")]
                        front_gate_mean = float(torch.stack([gate[:, FRONT_TRIPLET_INDICES].mean() for gate in gates]).mean().detach().item()) if gates else 0.0
                        front_gate_p95 = float(torch.quantile(torch.cat([gate[:, FRONT_TRIPLET_INDICES].detach().float().reshape(-1) for gate in gates]), 0.95).item()) if gates else 0.0
                        rear_delta_max = max(float(delta[:, REAR_CAMERA_INDICES].detach().abs().max().item()) for delta in deltas) if deltas else 0.0
                        row = {
                            "epoch": int(epoch),
                            "block_id": int(block_id),
                            "sample_index": int(sample_index),
                            "gamma_train": float(args.train_gamma),
                            "lr": float(args.lr),
                            "elapsed_s": time.time() - sample_start,
                            "block_prepare_elapsed_s": time.time() - block_start,
                            "grad_norm": grad_norm,
                            "front_gate_mean": front_gate_mean,
                            "front_gate_p95": front_gate_p95,
                            "rear_loss_runtime_max_abs": rear_delta_max,
                            **loss_stats,
                            **cuda_mem_mb(),
                        }
                        log_rows.append(row)
                        if rear_delta_max > 1e-12:
                            raise RuntimeError(f"rear residual nonzero during training: {rear_delta_max}")
                        print(
                            f"[sw14c-r2b] train epoch={epoch} block={block_id} sample={sample_index} "
                            f"({sample_pos}/{len(train_indices)}) loss={row['total_loss']:.4f} "
                            f"gate={front_gate_mean:.5f} elapsed={row['elapsed_s']:.2f}s",
                            flush=True,
                        )
                    finally:
                        release(teacher_bundle, per_h, debug)
            finally:
                collect_now = int(args.gc_interval) > 0 and (block_id + 1) % max(1, int(args.gc_interval)) == 0
                prepared_block.clear()
                release(prepared_block, collect=collect_now, empty_cache=collect_now)
        ckpt_path = CHECKPOINT_DIR / f"sw14c_round2b_checkpoint_epoch{epoch}.pth"
        torch.save(
            checkpoint_payload(
                adapter,
                {
                    "stage": "SW14C Round2B gate/loss rescue",
                    "epoch": int(epoch),
                    "train_samples": [args.train_start, args.train_end],
                    "val_samples": [args.val_start, args.val_end],
                    "gamma_train": float(args.train_gamma),
                    "lr": float(args.lr),
                    "loss_config": config.to_dict(),
                    "backbone_head_frozen": freeze_status,
                    "subset_diagnostic": True,
                },
            ),
            ckpt_path,
        )
        best_epoch = int(epoch)
        epoch_rows.append(
            {
                "epoch": int(epoch),
                "checkpoint": str(ckpt_path),
                "mean_train_loss": finite_mean(epoch_loss_values),
                "training_samples": len(train_indices),
                "val_metrics_deferred_to_combined_val_audit": True,
                "gamma": float(args.train_gamma),
            }
        )
    last_ckpt = CHECKPOINT_DIR / f"sw14c_round2b_checkpoint_epoch{best_epoch}.pth"
    payload = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    payload.setdefault("meta", {})["selected_by"] = "only Round2B epoch checkpoint; val metrics filled by combined val audit"
    torch.save(payload, best_path)
    write_csv(REPORTS_DIR / "sw14c_round2b_training_log.csv", log_rows)
    write_csv(REPORTS_DIR / "sw14c_round2b_epoch_val_metrics.csv", epoch_rows)
    payload_best = torch.load(best_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    adapter.load_state_dict(payload_best["state_dict"], strict=True)
    summary = {
        "executed": True,
        "decision": "ROUND2B_TRAINING_COMPLETED",
        "train_samples": [args.train_start, args.train_end],
        "val_samples": [args.val_start, args.val_end],
        "epochs_requested": int(args.epochs),
        "epochs_completed": len({int(row["epoch"]) for row in log_rows}),
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "mean_train_loss": finite_mean([row["total_loss"] for row in log_rows]),
        "last_train_loss": log_rows[-1]["total_loss"] if log_rows else None,
        "best_val_over_teacher_at_gamma01": None,
        "val_metrics_deferred_to_combined_val_audit": True,
        "backbone_head_frozen": freeze_status,
        "elapsed_s": time.time() - start,
        "cuda": cuda_mem_mb(),
        "io_policy": f"block-prepared sample-major training with train_block_size={block_size}; teacher bundles from Round2 audit runtime cache; no SW13 large feature-cache reread",
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
    }
    write_json(REPORTS_DIR / "sw14c_round2b_training_summary.json", summary)
    release(model, dataset, collect=True, empty_cache=True)
    return summary, adapter


def feature_health_for_prepared(
    adapter: torch.nn.Module,
    prepared: dict[str, Any],
    gamma: float,
) -> dict[str, Any]:
    with torch.inference_mode():
        final_levels, debug = adapter.apply_to_levels(
            prepared["frame_levels"][0],
            prepared["memory_levels"],
            prepared["base_levels"],
            prepared["degradation_mask"],
            gamma=float(gamma),
        )
        clean_mask = front_triplet_degradation_mask(
            "A0_clean",
            batch_size=int(prepared["frame_levels"][0][0].shape[0]),
            device=prepared["frame_levels"][0][0].device,
            dtype=prepared["frame_levels"][0][0].dtype,
        )
        clean_final, clean_debug = adapter.apply_to_levels(
            prepared["frame_levels"][0],
            prepared["memory_levels"],
            prepared["frame_levels"][0],
            clean_mask,
            gamma=float(gamma),
        )
    deltas = [debug[f"residual_delta_level{i}"].detach().float() for i in range(4)]
    gates = [debug[f"residual_gate_level{i}"].detach().float() for i in range(4)]
    front_gates = torch.cat([gate[:, FRONT_TRIPLET_INDICES].reshape(-1).cpu() for gate in gates])
    rear_gates = torch.cat([gate[:, REAR_CAMERA_INDICES].reshape(-1).cpu() for gate in gates])
    front_delta = torch.cat([delta[:, FRONT_TRIPLET_INDICES].abs().reshape(-1).cpu() for delta in deltas])
    feature_deltas = [(final[:, FRONT_TRIPLET_INDICES] - base[:, FRONT_TRIPLET_INDICES]).detach().float() for final, base in zip(final_levels, prepared["base_levels"])]
    feature_cat = torch.cat([delta.abs().reshape(-1).cpu() for delta in feature_deltas])
    base_norm = float(torch.stack([base[:, FRONT_TRIPLET_INDICES].detach().float().norm() for base in prepared["base_levels"]]).sum().item())
    feature_norm = float(torch.stack([delta.detach().float().norm() for delta in feature_deltas]).sum().item())
    clean_delta_max = max(float((final - base).abs().max().item()) for final, base in zip(clean_final, prepared["frame_levels"][0]))
    return {
        "front_gate_mean": float(front_gates.mean().item()),
        "front_gate_p95": float(torch.quantile(front_gates, 0.95).item()),
        "front_gate_p99": float(torch.quantile(front_gates, 0.99).item()),
        "rear_gate_mean": float(rear_gates.mean().item()),
        "clean_gate_mean": float(torch.stack([clean_debug[f"residual_gate_level{i}"].detach().float().mean().cpu() for i in range(4)]).mean().item()),
        "residual_delta_mean_abs": float(front_delta.mean().item()),
        "residual_delta_p95_abs": float(torch.quantile(front_delta, 0.95).item()),
        "residual_delta_max_abs": float(front_delta.max().item()),
        "gamma_feature_delta_mean_abs": float(feature_cat.mean().item()),
        "relative_feature_delta_vs_base_norm": safe_div(feature_norm, base_norm),
        "rear_feature_delta_mean_abs": float(
            torch.stack([(final[:, REAR_CAMERA_INDICES] - base[:, REAR_CAMERA_INDICES]).detach().float().abs().mean().cpu() for final, base in zip(final_levels, prepared["base_levels"])]).mean().item()
        ),
        "clean_residual_max_abs": clean_delta_max,
    }


def health_audit_split(
    model: Any,
    dataset: Any,
    adapter: torch.nn.Module,
    indices: list[int],
    sectors: dict[str, torch.Tensor],
    split_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos, sample_index in enumerate(indices, start=1):
        prepared = None
        teacher_bundle = None
        per_h = None
        debug = None
        try:
            prepared = sw14c.prepare_residual_feature_case(model, dataset, sample_index)
            teacher_bundle = load_teacher_bundle(sample_index)
            stats01 = feature_health_for_prepared(adapter, prepared, 0.1)
            stats02 = feature_health_for_prepared(adapter, prepared, 0.2)
            per_h, debug = sw14c.run_residual_forward_prepared(model, adapter, prepared, gamma=0.1, training=False)
            raw_diffs: list[int] = []
            final_diffs: list[int] = []
            for horizon_s in CORE_HORIZONS:
                teacher_h = teacher_bundle["by_horizon"][horizon_s]
                raw_counts = binary_counts(occ_mask(per_h[horizon_s]["raw_semantic"].detach().cpu()), occ_mask(teacher_h["teacher_raw_semantic"]))
                final_semantic, _meta = postprocess_student(sample_index, horizon_s, per_h, teacher_bundle, sectors)
                final_counts = binary_counts(occ_mask(final_semantic), occ_mask(teacher_h["teacher_final_semantic"]))
                raw_diffs.append(int(raw_counts["occ_diff_count"]))
                final_diffs.append(int(final_counts["occ_diff_count"]))
            rows.append(
                {
                    "split": split_name,
                    "sample_index": int(sample_index),
                    "front_gate_mean": stats01["front_gate_mean"],
                    "front_gate_p95": stats01["front_gate_p95"],
                    "front_gate_p99": stats01["front_gate_p99"],
                    "rear_gate_mean": stats01["rear_gate_mean"],
                    "clean_gate_mean": stats01["clean_gate_mean"],
                    "residual_delta_mean_abs": stats01["residual_delta_mean_abs"],
                    "residual_delta_p95_abs": stats01["residual_delta_p95_abs"],
                    "residual_delta_max_abs": stats01["residual_delta_max_abs"],
                    "gamma0p1_front_feature_delta_mean_abs": stats01["gamma_feature_delta_mean_abs"],
                    "gamma0p2_front_feature_delta_mean_abs": stats02["gamma_feature_delta_mean_abs"],
                    "relative_feature_delta_vs_base_norm": stats01["relative_feature_delta_vs_base_norm"],
                    "raw_head_occ_diff_vs_teacher": finite_mean(raw_diffs),
                    "final_occ_diff_after_f3_frontcap": finite_mean(final_diffs),
                    "rear_feature_delta_mean_abs": stats01["rear_feature_delta_mean_abs"],
                    "clean_residual_max_abs": stats01["clean_residual_max_abs"],
                }
            )
            print(f"[sw14c-r2b] health {split_name} sample={sample_index} ({pos}/{len(indices)})", flush=True)
        finally:
            release(prepared, teacher_bundle, per_h, debug, collect=(pos % 10 == 0), empty_cache=(pos % 10 == 0))
    return rows


def summarize_health(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = train_rows + val_rows
    gate_mean = finite_mean([row["front_gate_mean"] for row in rows]) or 0.0
    gate_p95 = finite_mean([row["front_gate_p95"] for row in rows]) or 0.0
    feature_delta = finite_mean([row["gamma0p1_front_feature_delta_mean_abs"] for row in rows]) or 0.0
    raw_diff = finite_mean([row["raw_head_occ_diff_vs_teacher"] for row in rows]) or 0.0
    final_diff = finite_mean([row["final_occ_diff_after_f3_frontcap"] for row in rows]) or 0.0
    rear = max([float(row["rear_feature_delta_mean_abs"]) for row in rows], default=0.0)
    clean = max([float(row["clean_residual_max_abs"]) for row in rows], default=0.0)
    if rear > 1e-12 or clean > 1e-12:
        decision = "HEALTH_R4_MASK_LEAK"
    elif gate_mean < 0.005 or feature_delta < max(ROUND2_GAMMA01_FEATURE_DELTA * 10.0, 1e-9):
        decision = "HEALTH_R2_STILL_NOOP"
    elif gate_mean > 0.15 or gate_p95 > 0.30:
        decision = "HEALTH_R3_TOO_AGGRESSIVE"
    elif raw_diff > 0.0 and final_diff > 0.0 and feature_delta >= max(ROUND2_GAMMA01_FEATURE_DELTA * 10.0, 1e-9):
        decision = "HEALTH_R1_NOOP_FIXED"
    else:
        decision = "HEALTH_R5_HEALTHY_BUT_METRIC_NEUTRAL"
    return {
        "phase": "Phase 5 Round2B residual/gate health audit",
        "decision": decision,
        "front_gate_mean": gate_mean,
        "front_gate_p95_mean": gate_p95,
        "front_gate_p99_mean": finite_mean([row["front_gate_p99"] for row in rows]),
        "rear_gate_mean": finite_mean([row["rear_gate_mean"] for row in rows]),
        "clean_gate_mean": finite_mean([row["clean_gate_mean"] for row in rows]),
        "residual_delta_mean_abs": finite_mean([row["residual_delta_mean_abs"] for row in rows]),
        "gamma0p1_front_feature_delta_mean_abs": feature_delta,
        "gamma0p2_front_feature_delta_mean_abs": finite_mean([row["gamma0p2_front_feature_delta_mean_abs"] for row in rows]),
        "round2_front_gate_mean": ROUND2_FRONT_GATE_MEAN,
        "round2_gamma0p1_front_feature_delta_mean_abs": ROUND2_GAMMA01_FEATURE_DELTA,
        "feature_delta_gain_over_round2": safe_div(feature_delta, ROUND2_GAMMA01_FEATURE_DELTA),
        "raw_head_occ_diff_vs_teacher_mean": raw_diff,
        "final_occ_diff_after_f3_frontcap_mean": final_diff,
        "rear_feature_delta_max": rear,
        "clean_residual_max_abs": clean,
    }


def phase5_health_audit(args: argparse.Namespace, adapter: torch.nn.Module, sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    print("[sw14c-r2b] build SparseWorld health runtime", flush=True)
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    sw14c.base.freeze_sparseworld_modules(model)
    train_rows = health_audit_split(model, dataset, adapter, list(range(args.train_start, args.train_end + 1)), sectors, "train")
    val_rows = health_audit_split(model, dataset, adapter, list(range(args.val_start, args.val_end + 1)), sectors, "val")
    write_csv(REPORTS_DIR / "sw14c_round2b_residual_health_train.csv", train_rows)
    write_csv(REPORTS_DIR / "sw14c_round2b_residual_health_val.csv", val_rows)
    summary = summarize_health(train_rows, val_rows)
    write_json(REPORTS_DIR / "sw14c_round2b_residual_health_summary.json", summary)
    release(model, dataset, collect=True, empty_cache=True)
    return summary


def behavior_row(
    sample_index: int,
    horizon_s: int,
    gamma: float,
    teacher_bundle: dict[str, Any],
    student_final: torch.Tensor,
    student_raw: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    health_proxy: dict[str, float],
) -> dict[str, Any]:
    teacher_h = teacher_bundle["by_horizon"][int(horizon_s)]
    teacher_final_occ = occ_mask(teacher_h["teacher_final_semantic"])
    student_final_occ = occ_mask(student_final)
    teacher_fn = masks["teacher_FN_region"]
    teacher_fp = masks["teacher_FP_region"]
    correct_occ = masks["teacher_correct_occ_region"]
    correct_free = masks["teacher_correct_free_region"]
    correct = correct_occ | correct_free
    recovered = teacher_fn & student_final_occ
    suppressed = teacher_fp & ~student_final_occ
    added_fp = (~teacher_fp) & masks["gt_free"] & student_final_occ & ~teacher_final_occ
    broken = correct & (student_final_occ != teacher_final_occ)
    broken_free = correct_free & student_final_occ
    broken_occ = correct_occ & ~student_final_occ
    def mean_change(mask: torch.Tensor, key: str, teacher_key: str) -> float | None:
        if not bool(mask.any().item()):
            return None
        s = student_raw[key].detach().cpu().float()[mask].mean().item()
        t = teacher_h[teacher_key].float()[mask].mean().item()
        return float(s - t)
    return {
        "sample_index": int(sample_index),
        "horizon_s": int(horizon_s),
        "gamma": float(gamma),
        "teacher_FN_count": int(teacher_fn.sum().item()),
        "recovered_FN_count": int(recovered.sum().item()),
        "recovered_FN_rate": safe_div(int(recovered.sum().item()), int(teacher_fn.sum().item())),
        "confidence_change_on_teacher_FN": mean_change(teacher_fn, "raw_confidence", "teacher_confidence"),
        "margin_change_on_teacher_FN": mean_change(teacher_fn, "raw_margin", "teacher_margin"),
        "residual_magnitude_on_teacher_FN": health_proxy.get("front_feature_delta_mean_abs", 0.0),
        "gate_mean_on_teacher_FN": health_proxy.get("front_gate_mean", 0.0),
        "teacher_FP_count": int(teacher_fp.sum().item()),
        "suppressed_FP_count": int(suppressed.sum().item()),
        "suppressed_FP_rate": safe_div(int(suppressed.sum().item()), int(teacher_fp.sum().item())),
        "added_FP_count": int(added_fp.sum().item()),
        "residual_magnitude_on_teacher_FP": health_proxy.get("front_feature_delta_mean_abs", 0.0),
        "gate_mean_on_teacher_FP": health_proxy.get("front_gate_mean", 0.0),
        "teacher_correct_count": int(correct.sum().item()),
        "broken_correct_count": int(broken.sum().item()),
        "preserve_correct_rate": 1.0 - safe_div(int(broken.sum().item()), int(correct.sum().item())),
        "broken_correct_free_count": int(broken_free.sum().item()),
        "broken_correct_occ_count": int(broken_occ.sum().item()),
        "residual_magnitude_on_teacher_correct": health_proxy.get("front_feature_delta_mean_abs", 0.0),
    }


def phase6_behavior_audit(
    args: argparse.Namespace,
    adapter: torch.nn.Module,
    sectors: dict[str, torch.Tensor],
    gamma: float,
) -> dict[str, Any]:
    print("[sw14c-r2b] build SparseWorld behavior runtime", flush=True)
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    sw14c.base.freeze_sparseworld_modules(model)
    rows: list[dict[str, Any]] = []
    for pos, sample_index in enumerate(range(args.val_start, args.val_end + 1), start=1):
        prepared = None
        teacher_bundle = None
        per_h = None
        try:
            prepared = sw14c.prepare_residual_feature_case(model, dataset, sample_index)
            teacher_bundle = load_teacher_bundle(sample_index)
            masks_by_h = build_masks_for_bundle(teacher_bundle, sectors)
            health = feature_health_for_prepared(adapter, prepared, max(float(gamma), 0.1))
            proxy = {
                "front_feature_delta_mean_abs": float(health["gamma_feature_delta_mean_abs"]),
                "front_gate_mean": float(health["front_gate_mean"]),
            }
            if abs(float(gamma)) < 1e-12:
                for horizon_s in CORE_HORIZONS:
                    rows.append(
                        behavior_row(
                            sample_index,
                            horizon_s,
                            gamma,
                            teacher_bundle,
                            teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"].long(),
                            {
                                "raw_confidence": teacher_bundle["by_horizon"][horizon_s]["teacher_confidence"],
                                "raw_margin": teacher_bundle["by_horizon"][horizon_s]["teacher_margin"],
                            },
                            masks_by_h[horizon_s],
                            proxy,
                        )
                    )
            else:
                per_h, debug = sw14c.run_residual_forward_prepared(model, adapter, prepared, gamma=float(gamma), training=False)
                release(debug)
                for horizon_s in CORE_HORIZONS:
                    final_semantic, _meta = postprocess_student(sample_index, horizon_s, per_h, teacher_bundle, sectors)
                    rows.append(behavior_row(sample_index, horizon_s, gamma, teacher_bundle, final_semantic, per_h[horizon_s], masks_by_h[horizon_s], proxy))
            print(f"[sw14c-r2b] behavior val sample={sample_index} ({pos}/{args.val_end - args.val_start + 1})", flush=True)
        finally:
            release(prepared, teacher_bundle, per_h, collect=(pos % 10 == 0), empty_cache=(pos % 10 == 0))
    write_csv(REPORTS_DIR / "sw14c_round2b_teacher_error_behavior_val.csv", rows)
    recovered = sum(int(row["recovered_FN_count"]) for row in rows)
    suppressed = sum(int(row["suppressed_FP_count"]) for row in rows)
    added_fp = sum(int(row["added_FP_count"]) for row in rows)
    broken = sum(int(row["broken_correct_count"]) for row in rows)
    broken_free = sum(int(row["broken_correct_free_count"]) for row in rows)
    broken_occ = sum(int(row["broken_correct_occ_count"]) for row in rows)
    if recovered == 0 and suppressed == 0:
        decision = "BEHAV2_R5_NO_ACTION"
    elif broken > max(recovered + suppressed, 1):
        decision = "BEHAV2_R4_BREAKS_TEACHER_CORRECT"
    elif recovered > 0 and added_fp > recovered:
        decision = "BEHAV2_R2_RECOVERS_FN_ADDS_FP"
    elif suppressed > 0 and recovered == 0 and broken_occ > 0:
        decision = "BEHAV2_R3_SUPPRESSES_FP_LOSES_RECALL"
    elif recovered > 0 or suppressed > 0:
        decision = "BEHAV2_R1_TARGETED_IMPROVEMENT"
    else:
        decision = "BEHAV2_R6_PROMISING_BUT_WEAK"
    summary = {
        "phase": "Phase 6 teacher error behavior audit",
        "decision": decision,
        "gamma": float(gamma),
        "row_count": len(rows),
        "recovered_FN_total": recovered,
        "suppressed_FP_total": suppressed,
        "added_FP_total": added_fp,
        "broken_correct_total": broken,
        "broken_correct_free_total": broken_free,
        "broken_correct_occ_total": broken_occ,
        "preserve_correct_rate_mean": finite_mean([row["preserve_correct_rate"] for row in rows]),
        "recovered_FN_rate_mean": finite_mean([row["recovered_FN_rate"] for row in rows]),
        "suppressed_FP_rate_mean": finite_mean([row["suppressed_FP_rate"] for row in rows]),
    }
    write_json(REPORTS_DIR / "sw14c_round2b_teacher_error_behavior_summary.json", summary)
    release(model, dataset, collect=True, empty_cache=True)
    return summary


def phase7_gamma_selection(args: argparse.Namespace, adapter: torch.nn.Module, sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    print("[sw14c-r2b] build SparseWorld gamma selection runtime", flush=True)
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    sw14c.base.freeze_sparseworld_modules(model)
    gamma_candidates = parse_float_list(args.gamma_candidates)
    val_indices = list(range(args.val_start, args.val_end + 1))
    detail_rows = evaluate_gamma(model, dataset, adapter, val_indices, gamma_candidates, sectors, split_name="gamma_val")
    summary_rows = summarize_gamma_rows(detail_rows, gamma_candidates)
    selection = select_gamma(summary_rows)
    selection.update(
        {
            "phase": "Phase 7 gamma selection on val only",
            "gamma_candidates": gamma_candidates,
            "val_samples": [args.val_start, args.val_end],
            "summary_rows": summary_rows,
            "safety_filter": {
                "density_delta_over_teacher_max": 0.03,
                "false_positive_delta_over_teacher_max": 0.01,
                "front_local_proxy_max": 1.30,
                "clean_drift_required": 0.0,
                "rear_drift_required": 0.0,
            },
        }
    )
    write_csv(REPORTS_DIR / "sw14c_round2b_gamma_sweep_val.csv", summary_rows)
    write_csv(REPORTS_DIR / "sw14c_round2b_gamma_sweep_val_detail.csv", detail_rows)
    write_json(REPORTS_DIR / "sw14c_round2b_gamma_selection.json", selection)
    plot_gamma_tradeoff(summary_rows, FIGURES_DIR / "sw14c_round2b_gamma_tradeoff.png")
    release(model, dataset, collect=True, empty_cache=True)
    return selection


def summarize_behavior_rows(rows: list[dict[str, Any]], gamma: float) -> dict[str, Any]:
    recovered = sum(int(row["recovered_FN_count"]) for row in rows)
    suppressed = sum(int(row["suppressed_FP_count"]) for row in rows)
    added_fp = sum(int(row["added_FP_count"]) for row in rows)
    broken = sum(int(row["broken_correct_count"]) for row in rows)
    broken_free = sum(int(row["broken_correct_free_count"]) for row in rows)
    broken_occ = sum(int(row["broken_correct_occ_count"]) for row in rows)
    if recovered == 0 and suppressed == 0:
        decision = "BEHAV2_R5_NO_ACTION"
    elif broken > max(recovered + suppressed, 1):
        decision = "BEHAV2_R4_BREAKS_TEACHER_CORRECT"
    elif recovered > 0 and added_fp > recovered:
        decision = "BEHAV2_R2_RECOVERS_FN_ADDS_FP"
    elif suppressed > 0 and recovered == 0 and broken_occ > 0:
        decision = "BEHAV2_R3_SUPPRESSES_FP_LOSES_RECALL"
    elif recovered > 0 or suppressed > 0:
        decision = "BEHAV2_R1_TARGETED_IMPROVEMENT"
    else:
        decision = "BEHAV2_R6_PROMISING_BUT_WEAK"
    return {
        "phase": "Phase 6 teacher error behavior audit",
        "decision": decision,
        "gamma": float(gamma),
        "row_count": len(rows),
        "recovered_FN_total": recovered,
        "suppressed_FP_total": suppressed,
        "added_FP_total": added_fp,
        "broken_correct_total": broken,
        "broken_correct_free_total": broken_free,
        "broken_correct_occ_total": broken_occ,
        "preserve_correct_rate_mean": finite_mean([row["preserve_correct_rate"] for row in rows]),
        "recovered_FN_rate_mean": finite_mean([row["recovered_FN_rate"] for row in rows]),
        "suppressed_FP_rate_mean": finite_mean([row["suppressed_FP_rate"] for row in rows]),
    }


def update_epoch_val_metrics_from_gamma01(summary_rows: list[dict[str, Any]], train_gamma: float) -> None:
    rows = read_csv(REPORTS_DIR / "sw14c_round2b_epoch_val_metrics.csv")
    if not rows:
        return
    gamma_summary = next((row for row in summary_rows if abs(float(row["gamma"]) - float(train_gamma)) < 1e-12), None)
    if gamma_summary is None:
        return
    updated: list[dict[str, Any]] = []
    for row in rows:
        row.update(
            {
                "val_over_teacher_improvement": gamma_summary.get("front_fn_reduction_over_teacher"),
                "future_h4h6_fn_reduction_over_teacher": gamma_summary.get("future_h4h6_fn_reduction_over_teacher"),
                "teacher_FP_suppression_improvement": gamma_summary.get("teacher_FP_suppression_improvement"),
                "density_delta_over_teacher": gamma_summary.get("density_delta_over_teacher"),
                "false_positive_delta_over_teacher": gamma_summary.get("false_positive_delta_over_teacher"),
                "front_local_proxy": gamma_summary.get("front_local_proxy"),
                "safety_pass_all": gamma_summary.get("safety_pass_all"),
                "final_occ_diff_vs_teacher": gamma_summary.get("final_occ_diff_vs_teacher"),
                "val_metrics_deferred_to_combined_val_audit": False,
            }
        )
        updated.append(row)
    write_csv(REPORTS_DIR / "sw14c_round2b_epoch_val_metrics.csv", updated)
    training_summary_path = REPORTS_DIR / "sw14c_round2b_training_summary.json"
    if training_summary_path.exists():
        training = read_json(training_summary_path)
        training["best_val_over_teacher_at_gamma01"] = gamma_summary.get("front_fn_reduction_over_teacher")
        training["val_metrics_deferred_to_combined_val_audit"] = False
        write_json(training_summary_path, training)


def combined_val_audit_split(
    model: Any,
    dataset: Any,
    adapter: torch.nn.Module,
    indices: list[int],
    gamma_candidates: list[float],
    sectors: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[float, list[dict[str, Any]]]]:
    val_health_rows: list[dict[str, Any]] = []
    gamma_detail_rows: list[dict[str, Any]] = []
    behavior_by_gamma: dict[float, list[dict[str, Any]]] = {float(gamma): [] for gamma in gamma_candidates}
    for pos, sample_index in enumerate(indices, start=1):
        sample_start = time.time()
        prepared = None
        teacher_bundle = None
        try:
            prepared = sw14c.prepare_residual_feature_case(model, dataset, sample_index)
            teacher_bundle = load_teacher_bundle(sample_index)
            masks_by_h = build_masks_for_bundle(teacher_bundle, sectors)
            stats01 = feature_health_for_prepared(adapter, prepared, 0.1)
            stats02 = feature_health_for_prepared(adapter, prepared, 0.2)
            raw_diffs01: list[int] = []
            final_diffs01: list[int] = []
            for gamma in gamma_candidates:
                gamma_f = float(gamma)
                if abs(gamma_f) < 1e-12:
                    proxy0 = {"front_feature_delta_mean_abs": 0.0, "front_gate_mean": stats01["front_gate_mean"]}
                    for horizon_s in CORE_HORIZONS:
                        teacher_h = teacher_bundle["by_horizon"][horizon_s]
                        gamma_detail_rows.append(gamma_metric_row(sample_index, horizon_s, gamma_f, teacher_bundle, None, None))
                        behavior_by_gamma[gamma_f].append(
                            behavior_row(
                                sample_index,
                                horizon_s,
                                gamma_f,
                                teacher_bundle,
                                teacher_h["teacher_final_semantic"].long(),
                                {
                                    "raw_confidence": teacher_h["teacher_confidence"],
                                    "raw_margin": teacher_h["teacher_margin"],
                                },
                                masks_by_h[horizon_s],
                                proxy0,
                            )
                        )
                    continue
                per_h = None
                debug = None
                try:
                    per_h, debug = sw14c.run_residual_forward_prepared(model, adapter, prepared, gamma=gamma_f, training=False)
                    proxy = {
                        "front_feature_delta_mean_abs": float(stats01["gamma_feature_delta_mean_abs"]) * safe_div(gamma_f, 0.1),
                        "front_gate_mean": float(stats01["front_gate_mean"]),
                    }
                    for horizon_s in CORE_HORIZONS:
                        teacher_h = teacher_bundle["by_horizon"][horizon_s]
                        final_semantic, meta = postprocess_student(sample_index, horizon_s, per_h, teacher_bundle, sectors)
                        gamma_detail_rows.append(gamma_metric_row(sample_index, horizon_s, gamma_f, teacher_bundle, final_semantic, meta))
                        behavior_by_gamma[gamma_f].append(
                            behavior_row(sample_index, horizon_s, gamma_f, teacher_bundle, final_semantic, per_h[horizon_s], masks_by_h[horizon_s], proxy)
                        )
                        if abs(gamma_f - 0.1) < 1e-12:
                            raw_counts = binary_counts(occ_mask(per_h[horizon_s]["raw_semantic"].detach().cpu()), occ_mask(teacher_h["teacher_raw_semantic"]))
                            final_counts = binary_counts(occ_mask(final_semantic), occ_mask(teacher_h["teacher_final_semantic"]))
                            raw_diffs01.append(int(raw_counts["occ_diff_count"]))
                            final_diffs01.append(int(final_counts["occ_diff_count"]))
                finally:
                    release(per_h, debug)
            val_health_rows.append(
                {
                    "split": "val",
                    "sample_index": int(sample_index),
                    "front_gate_mean": stats01["front_gate_mean"],
                    "front_gate_p95": stats01["front_gate_p95"],
                    "front_gate_p99": stats01["front_gate_p99"],
                    "rear_gate_mean": stats01["rear_gate_mean"],
                    "clean_gate_mean": stats01["clean_gate_mean"],
                    "residual_delta_mean_abs": stats01["residual_delta_mean_abs"],
                    "residual_delta_p95_abs": stats01["residual_delta_p95_abs"],
                    "residual_delta_max_abs": stats01["residual_delta_max_abs"],
                    "gamma0p1_front_feature_delta_mean_abs": stats01["gamma_feature_delta_mean_abs"],
                    "gamma0p2_front_feature_delta_mean_abs": stats02["gamma_feature_delta_mean_abs"],
                    "relative_feature_delta_vs_base_norm": stats01["relative_feature_delta_vs_base_norm"],
                    "raw_head_occ_diff_vs_teacher": finite_mean(raw_diffs01),
                    "final_occ_diff_after_f3_frontcap": finite_mean(final_diffs01),
                    "rear_feature_delta_mean_abs": stats01["rear_feature_delta_mean_abs"],
                    "clean_residual_max_abs": stats01["clean_residual_max_abs"],
                }
            )
            print(
                f"[sw14c-r2b] fused val sample={sample_index} ({pos}/{len(indices)}) "
                f"gammas={len(gamma_candidates)} elapsed={time.time() - sample_start:.2f}s cuda={cuda_mem_mb()}",
                flush=True,
            )
        finally:
            release(prepared, teacher_bundle, collect=(pos % 10 == 0), empty_cache=(pos % 10 == 0))
    return val_health_rows, gamma_detail_rows, behavior_by_gamma


def phase5_7_fused_audit(
    args: argparse.Namespace,
    adapter: torch.nn.Module,
    sectors: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("[sw14c-r2b] build SparseWorld fused audit runtime", flush=True)
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    sw14c.base.freeze_sparseworld_modules(model)
    gamma_candidates = parse_float_list(args.gamma_candidates)
    train_rows = health_audit_split(model, dataset, adapter, list(range(args.train_start, args.train_end + 1)), sectors, "train")
    val_rows, gamma_detail_rows, behavior_by_gamma = combined_val_audit_split(
        model,
        dataset,
        adapter,
        list(range(args.val_start, args.val_end + 1)),
        gamma_candidates,
        sectors,
    )
    write_csv(REPORTS_DIR / "sw14c_round2b_residual_health_train.csv", train_rows)
    write_csv(REPORTS_DIR / "sw14c_round2b_residual_health_val.csv", val_rows)
    health_summary = summarize_health(train_rows, val_rows)
    write_json(REPORTS_DIR / "sw14c_round2b_residual_health_summary.json", health_summary)
    summary_rows = summarize_gamma_rows(gamma_detail_rows, gamma_candidates)
    gamma_selection = select_gamma(summary_rows)
    gamma_selection.update(
        {
            "phase": "Phase 7 gamma selection on val only",
            "gamma_candidates": gamma_candidates,
            "val_samples": [args.val_start, args.val_end],
            "summary_rows": summary_rows,
            "safety_filter": {
                "density_delta_over_teacher_max": 0.03,
                "false_positive_delta_over_teacher_max": 0.01,
                "front_local_proxy_max": 1.30,
                "clean_drift_required": 0.0,
                "rear_drift_required": 0.0,
            },
            "fused_with_health_and_behavior_audit": True,
        }
    )
    write_csv(REPORTS_DIR / "sw14c_round2b_gamma_sweep_val.csv", summary_rows)
    write_csv(REPORTS_DIR / "sw14c_round2b_gamma_sweep_val_detail.csv", gamma_detail_rows)
    write_json(REPORTS_DIR / "sw14c_round2b_gamma_selection.json", gamma_selection)
    plot_gamma_tradeoff(summary_rows, FIGURES_DIR / "sw14c_round2b_gamma_tradeoff.png")
    update_epoch_val_metrics_from_gamma01(summary_rows, float(args.train_gamma))
    selected_gamma = gamma_selection.get("selected_gamma")
    if selected_gamma is None:
        selected_gamma = float(args.train_gamma)
    selected_gamma = float(selected_gamma)
    behavior_rows = behavior_by_gamma.get(selected_gamma, [])
    write_csv(REPORTS_DIR / "sw14c_round2b_teacher_error_behavior_val.csv", behavior_rows)
    behavior_summary = summarize_behavior_rows(behavior_rows, selected_gamma)
    write_json(REPORTS_DIR / "sw14c_round2b_teacher_error_behavior_summary.json", behavior_summary)
    release(model, dataset, collect=True, empty_cache=True)
    return health_summary, gamma_selection, behavior_summary


def final_decision(
    init: dict[str, Any],
    mask_summary: dict[str, Any],
    training: dict[str, Any],
    health: dict[str, Any],
    behavior: dict[str, Any],
    gamma_selection: dict[str, Any],
) -> dict[str, Any]:
    best = gamma_selection.get("best_summary") or {}
    selected_gamma = gamma_selection.get("selected_gamma")
    val_improvement = float(best.get("front_fn_reduction_over_teacher") or 0.0)
    density_safe = float(best.get("density_delta_over_teacher") or 0.0) <= 0.03 and bool(best.get("safety_pass_all", False))
    fp_safe = float(best.get("false_positive_delta_over_teacher") or 0.0) <= 0.01 and bool(best.get("safety_pass_all", False))
    front_safe = float(best.get("front_local_proxy") or 999.0) <= 1.30 and bool(best.get("safety_pass_all", False))
    clean_rear_safe = float(health.get("rear_feature_delta_max") or 0.0) <= 1e-12 and float(health.get("clean_residual_max_abs") or 0.0) <= 1e-12
    if init.get("decision") != "ROUND2B_INIT_READY" or mask_summary.get("decision") == "MASK_R4_MASK_BUG":
        decision = "SW14C_R2B_6_MASK_OR_PROTOCOL_BUG"
    elif not clean_rear_safe or health.get("decision") == "HEALTH_R4_MASK_LEAK":
        decision = "SW14C_R2B_6_MASK_OR_PROTOCOL_BUG"
    elif health.get("decision") == "HEALTH_R2_STILL_NOOP":
        decision = "SW14C_R2B_0_STILL_NOOP"
    elif behavior.get("decision") == "BEHAV2_R4_BREAKS_TEACHER_CORRECT":
        decision = "SW14C_R2B_5_BREAKS_TEACHER_CORRECT"
    elif gamma_selection.get("decision") == "GAMMA2_R3_SIGNAL_ONLY_UNSAFE":
        decision = "SW14C_R2B_1_SIGNAL_ONLY_UNSAFE"
    elif gamma_selection.get("decision") == "GAMMA2_R1_SAFE_IMPROVEMENT" and val_improvement >= 0.01 and density_safe and fp_safe and front_safe and clean_rear_safe:
        decision = "SW14C_R2B_4_SAFE_STRONG_GAIN_READY_DEBUG"
    elif gamma_selection.get("decision") == "GAMMA2_R1_SAFE_IMPROVEMENT" and val_improvement >= 0.002 and density_safe and fp_safe and front_safe and clean_rear_safe:
        decision = "SW14C_R2B_3_SAFE_SMALL_GAIN_READY_DEBUG"
    elif gamma_selection.get("decision") == "GAMMA2_R2_SAFE_BUT_NO_MEANINGFUL_IMPROVEMENT":
        decision = "SW14C_R2B_2_SAFE_BUT_NO_MEANINGFUL_GAIN"
    else:
        decision = "SW14C_R2B_7_STOP_KEEP_SW13"
    eval_debug_allowed = decision in {"SW14C_R2B_3_SAFE_SMALL_GAIN_READY_DEBUG", "SW14C_R2B_4_SAFE_STRONG_GAIN_READY_DEBUG"}
    round3_allowed = False
    if decision in {"SW14C_R2B_3_SAFE_SMALL_GAIN_READY_DEBUG", "SW14C_R2B_4_SAFE_STRONG_GAIN_READY_DEBUG"}:
        recommended = "Freeze the Round2B checkpoint/gamma and run eval_debug as a diagnostic only; Round3 remains blocked until eval_debug passes."
    elif decision == "SW14C_R2B_0_STILL_NOOP":
        recommended = "Stop this gate/loss rescue configuration; further Round2B would need more aggressive gate opening or a different residual parameterization before any eval_debug."
    elif decision == "SW14C_R2B_1_SIGNAL_ONLY_UNSAFE":
        recommended = "Do not expand training. Add density-aware FP suppression and residual region gating, then repeat Round2B on the same train-derived split."
    elif decision == "SW14C_R2B_2_SAFE_BUT_NO_MEANINGFUL_GAIN":
        recommended = "Do not run eval_debug. Either stop SW14C or do teacher error mining before another small rescue attempt."
    elif decision == "SW14C_R2B_5_BREAKS_TEACHER_CORRECT":
        recommended = "Add stronger teacher-correct preserve and tighter teacher-error-region masking; do not run eval_debug."
    elif decision == "SW14C_R2B_6_MASK_OR_PROTOCOL_BUG":
        recommended = "Fix the protocol/mask issue and rerun Round2B from scratch; do not use this checkpoint."
    else:
        recommended = "Stop SW14C as a main-result branch and keep SW13C-Fix + FrontCap as the main result."
    payload = {
        "decision": decision,
        "best_checkpoint": training.get("best_checkpoint"),
        "selected_gamma": selected_gamma,
        "selected_gamma_is_nonzero": bool(selected_gamma is not None and abs(float(selected_gamma)) > 1e-12),
        "val_over_teacher_improvement": val_improvement,
        "density_safety": density_safe,
        "fp_safety": fp_safe,
        "front_local_safety": front_safe,
        "clean_rear_safety": clean_rear_safe,
        "residual_health_decision": health.get("decision"),
        "teacher_error_behavior_decision": behavior.get("decision"),
        "whether_eval_debug_allowed": eval_debug_allowed,
        "whether_round3_allowed": round3_allowed,
        "whether_resume_allowed": False,
        "recommended_next_action": recommended,
        "gamma_selection_decision": gamma_selection.get("decision"),
        "mask_decision": mask_summary.get("decision"),
        "training_decision": training.get("decision"),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gamma_zero_counts_as_improvement": False,
        "subset_diagnostic_only": True,
        "get_occ_sha256_after": sha256(GET_OCC_PATH),
        "sw14b_postprocess_script_sha256_after": sha256(sw14c.SW14B_STAGE_SCRIPT),
        "f3_script_sha256_after": sha256(Path(sw14c.base.sw13c_fix.__file__)),
        "frontcap_script_sha256_after": sha256(Path(sw14c.base.frontcap50.__file__)),
    }
    assert payload["decision"] in FINAL_ENUMS
    write_json(REPORTS_DIR / "sw14c_round2b_final_decision.json", payload)
    return payload


def plot_gamma_tradeoff(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [float(row["gamma"]) for row in rows]
    improvement = [float(row.get("front_fn_reduction_over_teacher") or 0.0) for row in rows]
    density = [float(row.get("density_delta_over_teacher") or 0.0) for row in rows]
    fp = [float(row.get("false_positive_delta_over_teacher") or 0.0) for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, improvement, marker="o", label="front FN reduction")
    plt.plot(xs, density, marker="s", label="density delta")
    plt.plot(xs, fp, marker="^", label="FP delta")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(0.002, color="green", linewidth=0.8, linestyle="--", label="+0.002 gate")
    plt.xlabel("gamma")
    plt.ylabel("delta vs SW13 teacher")
    plt.title("SW14C Round2B gamma tradeoff")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def make_figures(health: dict[str, Any], behavior: dict[str, Any], gamma_selection: dict[str, Any]) -> None:
    summary_rows = gamma_selection.get("summary_rows", [])
    plt.figure(figsize=(7, 4))
    labels = ["Round2 gate", "Round2B gate", "Round2 feat", "Round2B feat"]
    values = [
        ROUND2_FRONT_GATE_MEAN,
        float(health.get("front_gate_mean") or 0.0),
        ROUND2_GAMMA01_FEATURE_DELTA,
        float(health.get("gamma0p1_front_feature_delta_mean_abs") or 0.0),
    ]
    plt.bar(labels, values)
    plt.yscale("log")
    plt.title("Round2 vs Round2B gate and feature delta")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14c_round2b_gate_health.png", dpi=180)
    plt.close()
    rows = read_csv(REPORTS_DIR / "sw14c_round2b_training_log.csv")
    if rows:
        plt.figure(figsize=(7, 4))
        plt.plot([float(row["total_loss"]) for row in rows], label="total")
        plt.plot([float(row["teacher_FN_recover_loss"]) for row in rows], label="FN recover")
        plt.plot([float(row["teacher_FP_suppress_loss"]) for row in rows], label="FP suppress")
        plt.legend()
        plt.title("Round2B loss curves")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "sw14c_round2b_loss_curves.png", dpi=180)
        plt.close()
    plot_gamma_tradeoff(summary_rows, FIGURES_DIR / "sw14c_round2b_gamma_tradeoff.png")
    plt.figure(figsize=(7, 4))
    plt.bar(
        ["FN recovered", "FP suppressed", "FP added", "correct broken"],
        [
            float(behavior.get("recovered_FN_total") or 0.0),
            float(behavior.get("suppressed_FP_total") or 0.0),
            float(behavior.get("added_FP_total") or 0.0),
            float(behavior.get("broken_correct_total") or 0.0),
        ],
    )
    plt.title("Round2B teacher error behavior")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14c_round2b_teacher_error_behavior.png", dpi=180)
    plt.close()
    best = gamma_selection.get("best_summary") or {}
    plt.figure(figsize=(7, 4))
    plt.bar(
        ["front gain", "future gain", "FP suppression", "density delta", "FP delta"],
        [
            float(best.get("front_fn_reduction_over_teacher") or 0.0),
            float(best.get("future_h4h6_fn_reduction_over_teacher") or 0.0),
            float(best.get("teacher_FP_suppression_improvement") or 0.0),
            float(best.get("density_delta_over_teacher") or 0.0),
            float(best.get("false_positive_delta_over_teacher") or 0.0),
        ],
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Round2B val teacher vs student")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14c_round2b_val_teacher_vs_student_bar.png", dpi=180)
    plt.close()


def write_report(
    inherited: dict[str, Any],
    adapter_init: dict[str, Any],
    mask_summary: dict[str, Any],
    training: dict[str, Any],
    health: dict[str, Any],
    behavior: dict[str, Any],
    gamma_selection: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Stage SW14C Round2B Gate/Loss Rescue Report",
        "",
        "This is a small train-derived subset diagnostic. It trains only the residual adapter on train 0..99, selects gamma only on val 100..149, and does not run eval_debug/core eval.",
        "",
        "## Decisions",
        f"- inherited state: `{inherited['decision']}`",
        f"- adapter init: `{adapter_init['decision']}`",
        f"- region masks: `{mask_summary['decision']}`",
        f"- training: `{training['decision']}`",
        f"- residual health: `{health['decision']}`",
        f"- teacher error behavior: `{behavior['decision']}`",
        f"- gamma selection: `{gamma_selection['decision']}`",
        f"- final: `{decision['decision']}`",
        "",
        "## Evidence",
        f"- Round2 front gate mean: `{ROUND2_FRONT_GATE_MEAN}`",
        f"- Round2B front gate mean: `{health.get('front_gate_mean')}`",
        f"- Round2 gamma=0.1 feature delta: `{ROUND2_GAMMA01_FEATURE_DELTA}`",
        f"- Round2B gamma=0.1 feature delta: `{health.get('gamma0p1_front_feature_delta_mean_abs')}`",
        f"- selected gamma: `{decision.get('selected_gamma')}`",
        f"- val over teacher improvement: `{decision.get('val_over_teacher_improvement')}`",
        f"- clean/rear safe: `{decision.get('clean_rear_safety')}`",
        "",
        "## Protocol",
        "- SparseWorld backbone/head are frozen.",
        "- get_occ, F3, and FrontCap are not modified.",
        "- gamma=0 is not counted as an improvement.",
        "- eval_debug/core100/core500 are not executed.",
        "",
        "## Recommendation",
        decision["recommended_next_action"],
    ]
    write_md(REPORTS_DIR / "stage_sw14c_round2b_gate_loss_rescue_report.md", "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    sw14c.configure_cuda_for_throughput()
    inherited = phase0_inherited_state(args)
    adapter = build_round2b_adapter()
    adapter_init = phase1_adapter_config(adapter)
    sectors = {key: value.cpu() for key, value in sw14c.base.sw13c_fix.sw7.build_sector_masks().items()}
    mask_summary = phase2_region_masks(args, sectors)
    config = phase3_loss_config()
    if inherited["decision"] != "ROUND2B_INIT_READY" or mask_summary["decision"] == "MASK_R4_MASK_BUG":
        training = {
            "executed": False,
            "decision": "ROUND2B_TRAINING_BLOCKED_BY_INIT_OR_MASK",
            "best_checkpoint": None,
        }
        health = {"decision": "HEALTH_R4_MASK_LEAK", "rear_feature_delta_max": None, "clean_residual_max_abs": None}
        behavior = {"decision": "BEHAV2_R5_NO_ACTION"}
        gamma_selection = {"decision": "GAMMA2_R4_NO_SIGNAL", "selected_gamma": None, "selected_gamma_is_nonzero": False, "best_summary": None, "summary_rows": []}
    else:
        training, adapter = train_round2b(args, adapter, config, sectors)
        health, gamma_selection, behavior = phase5_7_fused_audit(args, adapter, sectors)
    decision = final_decision(inherited, mask_summary, training, health, behavior, gamma_selection)
    make_figures(health, behavior, gamma_selection)
    write_report(inherited, adapter_init, mask_summary, training, health, behavior, gamma_selection, decision)
    print(f"[sw14c-r2b] final decision {decision['decision']}", flush=True)


if __name__ == "__main__":
    main()
