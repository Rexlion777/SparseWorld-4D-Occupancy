from __future__ import annotations

import argparse
import copy
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
from mmcv.runner.fp16_utils import cast_tensor_type


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
for rel in ["stage_sw14c_residual_teacher_adapter", "stage_sw14c_round2b_gate_loss_rescue"]:
    path = SCRIPT_DIR.parent / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sw14c_residual_teacher_adapter as sw14c  # noqa: E402
from sw14c_dual_branch_residual_adapter import (  # noqa: E402
    FRONT_TRIPLET_INDICES,
    REAR_CAMERA_INDICES,
    build_dual_branch_adapter,
    checkpoint_payload,
    count_parameters,
)
from sw14c_error_mask_utils import (  # noqa: E402
    CORE_HORIZONS,
    EMPTY_IDX,
    build_oracle_proxy_masks,
    mask_stats_row,
    occ_mask,
    summarize_mask_rows,
)
from sw14c_oracle_mask_eval_utils import (  # noqa: E402
    finite_mean,
    read_json,
    safe_div,
    write_csv,
    write_json,
    write_md,
)
from sw14c_oracle_mask_losses import LOSS_TERMS_SCHEMA, OracleMaskLossConfig, compute_oracle_mask_loss  # noqa: E402


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_oracle_error_mask_residual"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_oracle_error_mask_residual"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14c_oracle_error_mask_residual"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14c_oracle_error_mask_residual"

ROUND2B_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue"
AUDIT_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit"
ROUND2_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
AUDIT_CACHE = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_learning_audit/runtime_teacher_cache/audit_A10_drop_front_triplet"
GET_OCC_PATH = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/opus_head.py"

FINAL_ENUMS = {
    "SW14C_OEM_0_ORACLE_MASK_FAIL_KEEP_SW13",
    "SW14C_OEM_1_ORACLE_MASK_LOW_POTENTIAL",
    "SW14C_OEM_2_ORACLE_MASK_SIGNAL_UNSAFE",
    "SW14C_OEM_3_ORACLE_MASK_SAFE_GAIN",
    "SW14C_OEM_4_PROXY_MASK_SAFE_GAIN_READY_DEBUG",
    "SW14C_OEM_5_PROXY_FAIL_BUT_ORACLE_SUCCESS",
    "SW14C_OEM_6_DUAL_BRANCH_BREAKS_CORRECT",
    "SW14C_OEM_7_PROTOCOL_VIOLATION",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14C oracle teacher-error mask residual diagnostic")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=99)
    parser.add_argument("--val-start", type=int, default=100)
    parser.add_argument("--val-end", type=int, default=149)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-block-size", type=int, default=4)
    parser.add_argument("--gamma-execution-mode", choices=["online_fast", "direct_batch"], default="online_fast")
    parser.add_argument("--gamma-batch-size", type=int, default=4)
    parser.add_argument("--interleave-val-postprocess", action="store_true", help="legacy path; default defers CPU F3/FrontCap work until after sample GPU gamma forwards")
    parser.add_argument("--gamma-add-candidates", default="0.0,0.05,0.1,0.2,0.3")
    parser.add_argument("--gamma-sup-candidates", default="0.0,0.05,0.1,0.2,0.3")
    parser.add_argument("--train-gamma-add", type=float, default=0.1)
    parser.add_argument("--train-gamma-sup", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    audit = read_json(AUDIT_REPORTS / "sw14c_residual_learning_audit_decision.json") if (AUDIT_REPORTS / "sw14c_residual_learning_audit_decision.json").exists() else {}
    round2b = read_json(ROUND2B_REPORTS / "sw14c_round2b_final_decision.json") if (ROUND2B_REPORTS / "sw14c_round2b_final_decision.json").exists() else {}
    teacher_regions = read_json(AUDIT_REPORTS / "sw14c_teacher_error_region_summary.json") if (AUDIT_REPORTS / "sw14c_teacher_error_region_summary.json").exists() else {}
    cache_count = len(list(AUDIT_CACHE.glob("A10_drop_front_triplet__sample*.pt"))) if AUDIT_CACHE.exists() else 0
    ready = (
        audit.get("primary_cause") == "SW14C_AUDIT_2_GATE_OR_REG_TOO_STRONG"
        and round2b.get("decision") in {"SW14C_R2B_0_STILL_NOOP", "SW14C_R2B_1_SIGNAL_ONLY_UNSAFE"}
        and cache_count >= max(args.train_end + 1, args.val_end + 1)
    )
    decision = "ORACLE_INIT_READY" if ready else ("ORACLE_INIT_MISSING_ARTIFACTS" if cache_count < max(args.train_end + 1, args.val_end + 1) else "ORACLE_INIT_PROTOCOL_RISK")
    payload = {
        "phase": "Phase 0 inherited state and protocol",
        "decision": decision,
        "sw13c_fix_frontcap_remains_main_result": True,
        "diagnostic_upper_bound_only": True,
        "oracle_mask_uses_gt": True,
        "oracle_mask_is_not_deployable": True,
        "if_oracle_fails_recommend_candidate_reranker": True,
        "if_oracle_succeeds_next_is_no_gt_proxy_or_learned_error_mask": True,
        "round2_audit_primary_cause": audit.get("primary_cause"),
        "round2b_final_decision": round2b.get("decision"),
        "round2b_residual_health": round2b.get("residual_health_decision"),
        "teacher_error_region_decision": teacher_regions.get("decision"),
        "audit_teacher_cache": str(AUDIT_CACHE),
        "audit_teacher_cache_count": cache_count,
        "get_occ_sha256_before": sha256(GET_OCC_PATH),
        "f3_script_sha256_before": sha256(Path(sw14c.base.sw13c_fix.__file__)),
        "frontcap_script_sha256_before": sha256(Path(sw14c.base.frontcap50.__file__)),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gpu_policy": (
            "All GPU stages use block-prepared train or fused val audit; train health reuses training debug stats; "
            "val gamma selection batches multiple gamma_add/gamma_sup pairs per SparseWorld forward with OOM fallback; "
            "no repeated val health/gamma/behavior passes."
        ),
    }
    write_json(REPORTS_DIR / "sw14c_oracle_mask_inherited_state.json", payload)
    write_md(
        REPORTS_DIR / "sw14c_oracle_mask_inherited_state.md",
        "\n".join(
            [
                "# SW14C Oracle Error Mask Residual Inherited State",
                "",
                f"- decision: `{decision}`",
                "- Diagnostic upper bound only; oracle masks use GT and are not deployable.",
                "- SW13C-Fix + FrontCap remains the main result.",
                "- No eval_debug, core100, core500, F3/FrontCap/get_occ modification, or Round3 is allowed.",
            ]
        )
        + "\n",
    )
    return payload


def build_masks_for_bundle(teacher_bundle: dict[str, Any], sectors: dict[str, torch.Tensor]) -> dict[int, dict[str, torch.Tensor]]:
    return {h: build_oracle_proxy_masks(teacher_bundle["by_horizon"][h], h, sectors) for h in CORE_HORIZONS}


def phase1_masks(args: argparse.Namespace, sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for split, indices, rows in [
        ("train", range(args.train_start, args.train_end + 1), train_rows),
        ("val", range(args.val_start, args.val_end + 1), val_rows),
    ]:
        for sample_index in indices:
            bundle = load_teacher_bundle(sample_index)
            masks_by_h = build_masks_for_bundle(bundle, sectors)
            for h, masks in masks_by_h.items():
                rows.append(mask_stats_row(split, sample_index, h, masks))
            release(bundle, masks_by_h)
    write_csv(REPORTS_DIR / "sw14c_error_mask_stats_train.csv", train_rows)
    write_csv(REPORTS_DIR / "sw14c_error_mask_stats_val.csv", val_rows)
    train_summary = summarize_mask_rows(train_rows)
    val_summary = summarize_mask_rows(val_rows)
    if val_summary["decision"] == "MASK_O2_ERROR_TOO_SPARSE":
        decision = "MASK_O2_ERROR_TOO_SPARSE"
    elif val_summary["decision"] == "MASK_O3_PROXY_HAS_REASONABLE_OVERLAP":
        decision = "MASK_O3_PROXY_HAS_REASONABLE_OVERLAP"
    elif val_summary["decision"] == "MASK_O4_PROXY_POOR_BUT_ORACLE_READY":
        decision = "MASK_O4_PROXY_POOR_BUT_ORACLE_READY"
    elif val_summary["oracle_strict_total"] > 0:
        decision = "MASK_O1_READY_ERROR_SIGNAL_EXISTS"
    else:
        decision = "MASK_O5_MASK_BUG"
    payload = {
        "phase": "Phase 1 teacher error masks",
        "decision": decision,
        "train": train_summary,
        "val": val_summary,
        "oracle_mask_uses_gt": True,
        "proxy_mask_uses_gt": False,
        "dilation_radius": 1,
    }
    write_json(REPORTS_DIR / "sw14c_error_mask_summary.json", payload)
    return payload


def phase2_adapter_init(adapter: torch.nn.Module) -> dict[str, Any]:
    stats = count_parameters(adapter)
    device = next(adapter.parameters()).device
    current = [torch.randn(1, 6, 256, 2, 2, device=device) for _ in range(4)]
    memory = [level + 1.0 for level in current]
    base = sw14c.build_sw13_r8_base_repair(current, memory, "A10_drop_front_triplet")
    mask = sw14c.front_triplet_degradation_mask("A10_drop_front_triplet", batch_size=1, device=device)
    clean_mask = sw14c.front_triplet_degradation_mask("A0_clean", batch_size=1, device=device)
    with torch.inference_mode():
        gamma0, _ = adapter.apply_to_levels(current, memory, base, mask, gamma_add=0.0, gamma_sup=0.0)
        gamma1, debug = adapter.apply_to_levels(current, memory, base, mask, gamma_add=0.1, gamma_sup=0.1)
        clean_final, clean_debug = adapter.apply_to_levels(current, memory, current, clean_mask, gamma_add=0.1, gamma_sup=0.1)
    gamma0_max = max(float((f - b).abs().max().item()) for f, b in zip(gamma0, base))
    rear_max = max(float((f[:, REAR_CAMERA_INDICES] - b[:, REAR_CAMERA_INDICES]).abs().max().item()) for f, b in zip(gamma1, base))
    clean_max = max(float((f - b).abs().max().item()) for f, b in zip(clean_final, current))
    payload = {
        "phase": "Phase 2 dual branch adapter init",
        "decision": "ARCH_O1_READY" if gamma0_max <= 1e-12 and rear_max <= 1e-12 and clean_max <= 1e-12 else "ARCH_O4_REAR_CLEAN_LEAKAGE",
        "adapter_trainable_params": int(stats.trainable_parameter_count),
        "adapter_total_params": int(stats.parameter_count),
        "hidden_dim": 64,
        "gate_bias_init": -2.0,
        "residual_bound_value": 0.05,
        "gamma0_equals_base_max_abs": gamma0_max,
        "rear_residual_max_abs": rear_max,
        "clean_residual_max_abs": clean_max,
        "front_add_gate_init_mean": float(torch.stack([debug[f"add_gate_level{i}"][:, FRONT_TRIPLET_INDICES].mean() for i in range(4)]).mean().item()),
        "front_sup_gate_init_mean": float(torch.stack([debug[f"sup_gate_level{i}"][:, FRONT_TRIPLET_INDICES].mean() for i in range(4)]).mean().item()),
        "voxel_to_feature_mask_alignment_available": False,
        "mask_out_enforcement": "output-space preserve loss plus structural front-triplet feature mask",
    }
    write_json(REPORTS_DIR / "sw14c_dual_branch_adapter_config.json", {k: payload[k] for k in ["hidden_dim", "gate_bias_init", "residual_bound_value", "mask_out_enforcement"]})
    write_json(REPORTS_DIR / "sw14c_dual_branch_adapter_init_audit.json", payload)
    write_md(
        REPORTS_DIR / "sw14c_masked_residual_architecture.md",
        "# SW14C Oracle Mask Dual Branch Residual Adapter\n\n"
        "Feature residual is split into add and suppress branches. The final feature is `base + gamma_add * gate_add * delta_add - gamma_sup * gate_sup * delta_sup`.\n\n"
        "Oracle masks are voxel-space GT diagnostic masks. The current SparseWorld path does not expose a feature-to-voxel projection, so mask-out is enforced in output space and front-triplet/rear/clean are enforced structurally at feature level.\n",
    )
    return payload


def eval_row_for_semantic(sample_index: int, horizon_s: int, semantic: torch.Tensor, teacher_h: dict[str, Any], sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    native = teacher_h["native_semantic"].long()
    native_eval = sw14c.base.sw12b.build_eval_row(native, teacher_h["gt_h"].long(), teacher_h["gt0"].long(), "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=native)
    row = sw14c.base.sw12b.build_eval_row(semantic.long(), teacher_h["gt_h"].long(), teacher_h["gt0"].long(), "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=native)
    return sw14c.base.attach_sw13_style_deltas(row, native_eval, horizon_s)


def phase3_oracle_upper_bound(args: argparse.Namespace, sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    all_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split, indices in [("train", range(args.train_start, args.train_end + 1)), ("val", range(args.val_start, args.val_end + 1))]:
        for sample_index in indices:
            bundle = load_teacher_bundle(sample_index)
            masks_by_h = build_masks_for_bundle(bundle, sectors)
            for h in CORE_HORIZONS:
                teacher_h = bundle["by_horizon"][h]
                masks = masks_by_h[h]
                teacher_final = teacher_h["teacher_final_semantic"].long()
                oracle = teacher_final.clone()
                add = masks["teacher_FN_region"] & (masks["raw_evidence_region"] | masks["oracle_error_mask_dilated"])
                sup = masks["teacher_FP_region"]
                oracle[add] = teacher_h["gt_h"].long()[add]
                oracle[sup] = EMPTY_IDX
                teacher_eval = teacher_h["teacher_meta"]["final_eval"]
                oracle_eval = eval_row_for_semantic(sample_index, h, oracle, teacher_h, sectors)
                native_front = max(1.0, float(((teacher_h["native_semantic"].long() != EMPTY_IDX) & sectors["front"].bool()).sum().item()))
                front_proxy = float(((oracle != EMPTY_IDX) & sectors["front"].bool()).sum().item()) / native_front
                density_delta = metric_value(oracle_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta") - metric_value(teacher_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta")
                fp_delta = metric_value(oracle_eval, "false_positive_delta") - metric_value(teacher_eval, "false_positive_delta")
                safety = density_delta <= 0.03 and fp_delta <= 0.01 and front_proxy <= 1.30
                all_rows[split].append(
                    {
                        "split": split,
                        "sample_index": sample_index,
                        "horizon_s": h,
                        "max_front_fn_reduction_over_teacher": metric_value(teacher_eval, "front_sector_false_free_rate_delta") - metric_value(oracle_eval, "front_sector_false_free_rate_delta"),
                        "max_future_h4h6_reduction_over_teacher": (metric_value(teacher_eval, "false_free_rate") - metric_value(oracle_eval, "false_free_rate")) if h in {4, 6} else 0.0,
                        "max_fp_reduction_over_teacher": metric_value(teacher_eval, "false_positive_delta") - metric_value(oracle_eval, "false_positive_delta"),
                        "density_delta_over_teacher_after_oracle": density_delta,
                        "front_local_proxy_after_oracle": front_proxy,
                        "oracle_add_count": int(add.sum().item()),
                        "oracle_suppress_count": int(sup.sum().item()),
                        "oracle_safety_pass": bool(safety),
                    }
                )
            release(bundle, masks_by_h)
    write_csv(REPORTS_DIR / "sw14c_oracle_candidate_upper_bound_train.csv", all_rows["train"])
    write_csv(REPORTS_DIR / "sw14c_oracle_candidate_upper_bound_val.csv", all_rows["val"])
    val = all_rows["val"]
    safe_rate = safe_div(sum(bool(r["oracle_safety_pass"]) for r in val), len(val))
    front_gain = finite_mean([r["max_front_fn_reduction_over_teacher"] for r in val]) or 0.0
    fp_gain = finite_mean([r["max_fp_reduction_over_teacher"] for r in val]) or 0.0
    if front_gain + fp_gain < 0.002:
        decision = "ORACLE_UB_2_LOW_POTENTIAL"
    elif safe_rate < 1.0 and front_gain + fp_gain >= 0.002:
        decision = "ORACLE_UB_3_ONLY_UNSAFE_POTENTIAL"
    elif front_gain + fp_gain >= 0.01:
        decision = "ORACLE_UB_1_HIGH_POTENTIAL"
    else:
        decision = "ORACLE_UB_4_CANDIDATE_SET_TOO_SMALL"
    payload = {
        "phase": "Phase 3 oracle candidate upper bound",
        "decision": decision,
        "val_front_gain_mean": front_gain,
        "val_fp_gain_mean": fp_gain,
        "val_safety_pass_rate": safe_rate,
        "oracle_uses_gt": True,
    }
    write_json(REPORTS_DIR / "sw14c_oracle_candidate_upper_bound_summary.json", payload)
    return payload


def run_dual_forward_prepared(model: Any, adapter: torch.nn.Module, prepared: dict[str, Any], gamma_add: float, gamma_sup: float, training: bool) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    holder: dict[str, Any] = {}
    original_forward = sw14c.base.attach_query_capture_live(model, holder)
    original_extract_feat = model.extract_feat
    frame_cursor = {"idx": 0}
    debug_holder: dict[str, torch.Tensor] = {}

    def patched_extract_feat(img: torch.Tensor, img_metas_curr: list[dict[str, Any]]):
        idx = frame_cursor["idx"]
        frame_cursor["idx"] += 1
        if idx != 0:
            return prepared["frame_levels"][idx]
        final_levels, debug = adapter.apply_to_levels(
            prepared["frame_levels"][0],
            prepared["memory_levels"],
            prepared["base_levels"],
            prepared["degradation_mask"],
            gamma_add=gamma_add,
            gamma_sup=gamma_sup,
        )
        debug_holder.update(debug)
        return final_levels

    model.extract_feat = patched_extract_feat  # type: ignore[assignment]
    try:
        sw14c.base.sw13a.reset_model_cache(model)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            _ = model(return_loss=False, rescale=True, **prepared["moved_batch"])
            head = sw14c.base.sw4_inst.get_pts_bbox_head(model)
            per_h: dict[int, dict[str, torch.Tensor]] = {}
            for horizon_s in CORE_HORIZONS:
                pred_dict = sw14c.base.extract_pred_dict_live(holder, horizon_s)
                occ_pred, debug_list = sw14c.base.sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                dense_scores = debug_list[0]["dense_occ_after_padding"]
                conf, margin = sw14c.base.occupancy_confidence_and_margin(dense_scores)
                per_h[horizon_s] = {
                    "raw_semantic": occ_pred[0],
                    "raw_confidence": conf,
                    "raw_margin": margin,
                    "dense_scores": dense_scores,
                }
        return per_h, debug_holder
    finally:
        model.extract_feat = original_extract_feat  # type: ignore[assignment]
        model.forward_backbone = original_forward  # type: ignore[assignment]


def repeat_tensor_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.ndim >= 1 and int(tensor.shape[0]) == 1:
        reps = [int(batch_size)] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*reps)
    return tensor


def repeat_nested_batch(obj: Any, batch_size: int) -> Any:
    if isinstance(obj, torch.Tensor):
        return repeat_tensor_batch(obj, batch_size)
    if isinstance(obj, dict):
        return {key: repeat_nested_batch(value, batch_size) for key, value in obj.items()}
    if isinstance(obj, list):
        return [repeat_nested_batch(value, batch_size) for value in obj]
    if isinstance(obj, tuple):
        return tuple(repeat_nested_batch(value, batch_size) for value in obj)
    return copy.deepcopy(obj)


def gamma_batched_moved_batch(moved_batch: dict[str, Any], batch_size: int) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in moved_batch.items():
        if key == "img_metas":
            meta = value[0][0]
            batch[key] = [[copy.deepcopy(meta) for _ in range(int(batch_size))]]
        else:
            batch[key] = repeat_nested_batch(value, int(batch_size))
    return batch


def expand_feature_levels(levels: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for level in levels:
        if int(level.shape[0]) == int(batch_size):
            out.append(level)
        else:
            out.append(level.expand(int(batch_size), *level.shape[1:]).contiguous())
    return out


def merged_frame_metas(prepared: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    merged = copy.deepcopy(prepared["frame_metas"][0][0])
    for frame_idx in range(1, len(prepared["frame_metas"])):
        frame_meta = prepared["frame_metas"][frame_idx][0]
        for key, value in frame_meta.items():
            if isinstance(value, list):
                if key not in merged or not isinstance(merged[key], list):
                    merged[key] = []
                merged[key].extend(copy.deepcopy(value))
    return [copy.deepcopy(merged) for _ in range(int(batch_size))]


def forward_backbone_from_reorganized_features(
    model: Any,
    img_feats: list[torch.Tensor],
    img_metas: list[dict[str, Any]],
    prepared: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    head = sw14c.base.sw4_inst.get_pts_bbox_head(model)
    ego_state0 = prepared["moved_batch"]["temporal_ego_states"][0][0]
    ego_states = repeat_tensor_batch(ego_state0, batch_size)
    bs, _, dim_ = ego_states.shape
    ego_states = ego_states.view((bs, 1, dim_))
    ego_feat = model.plan_head(ego_states)
    points_scale = torch.tanh(model.points_scale_branch(ego_feat))
    head.points_scale = (points_scale + 1) / 2 * (1.5 - 0.8) + 0.8

    img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)
    outs = head(img_feats, img_metas)
    ind_stamps_all = head.ind_stamps_all
    query_feat = outs["query_feat"]
    query_pos = outs["all_refine_pts"][-1]
    query_cls = outs["all_cls_scores"][-1]
    curr_query_feat = query_feat[:, ind_stamps_all == 0]
    curr_query_pos = query_pos[:, ind_stamps_all == 0].detach()
    curr_query_timestamp = query_pos.new_zeros(batch_size, model.num_query, model.num_refines, 1)
    curr_query_cls = query_cls[:, ind_stamps_all == 0]
    outputs: dict[str, Any] = {
        "cls_score": curr_query_cls,
        "refine_pts": curr_query_pos,
        "outs": outs,
    }

    forecast_points_list: list[torch.Tensor] = []
    forecast_semantics_list: list[torch.Tensor] = []
    pred_trajs_list: list[torch.Tensor] = []
    forecast_points_mask_list: list[torch.Tensor] = []
    num_fu_frames = model.num_fu_frames
    for interval in range(num_fu_frames):
        fused_ego_feat, _ = model.ego_cross_attn(
            ego_feat.new_ones(batch_size, 1, 3) * 0.5,
            ego_feat,
            curr_query_pos.detach(),
            curr_query_feat.detach(),
        )
        pred_traj = model.traj_head(fused_ego_feat)
        pred_trajs_list.append(pred_traj)
        curr_query_feat = torch.cat([curr_query_feat, query_feat[:, ind_stamps_all == interval + 1]], dim=1)
        curr_query_pos = torch.cat([curr_query_pos, query_pos[:, ind_stamps_all == interval + 1]], dim=1).detach()
        if interval < 6:
            curr_query_timestamp = torch.cat(
                [
                    curr_query_timestamp,
                    curr_query_pos.new_ones(batch_size, model.num_fu_query[interval], model.num_refines, 1) * 0.5,
                ],
                dim=1,
            )
        pos_embedding = model.position_encoder(torch.cat([curr_query_pos, curr_query_timestamp], dim=-1).flatten(2, 3))
        curr_query_feat = curr_query_feat + fused_ego_feat + pos_embedding
        reg_offset = model.reg_branch(curr_query_feat).unflatten(-1, (-1, 3)) * 0.5
        cls_score = model.cls_branch(curr_query_feat).unflatten(-1, (-1, 17))
        vel_offset = model.vel_branch(curr_query_feat).unflatten(-1, (-1, 2))
        pred_labels = cls_score.argmax(-1)
        pred_moving_mask = torch.logical_and(pred_labels >= 2, pred_labels <= 10).unsqueeze(-1)
        reg_offset = torch.cat([reg_offset[..., :2] + vel_offset * pred_moving_mask, reg_offset[..., 2:]], dim=-1)
        reg_offset = reg_offset.flatten(2, 3)
        curr_query_pos = model.refine_points(curr_query_pos, reg_offset)
        forecast_semantics_list.append(cls_score)
        forecast_points_list.append(curr_query_pos)

    if not model.pretrain and len(pred_trajs_list) < model.num_fu_frames:
        fused_ego_feat, _ = model.ego_cross_attn(ego_feat.new_zeros(batch_size, 1, 3), ego_feat, curr_query_pos, curr_query_feat)
        pred_traj = model.traj_head(fused_ego_feat)
        pred_trajs_list.append(pred_traj)

    outputs.update(
        {
            "forecast_semantics_list": forecast_semantics_list,
            "forecast_points_list": forecast_points_list,
            "pred_trajs_list": pred_trajs_list,
            "forecast_points_mask_list": forecast_points_mask_list,
        }
    )
    return outputs


def run_dual_forward_prepared_gamma_batch(
    model: Any,
    adapter: torch.nn.Module,
    prepared: dict[str, Any],
    gamma_pairs: list[tuple[float, float]],
) -> tuple[dict[tuple[float, float], dict[int, dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    batch_size = len(gamma_pairs)
    if batch_size <= 0:
        return {}, {}
    gamma_add_t = torch.tensor([pair[0] for pair in gamma_pairs], device=prepared["base_levels"][0].device, dtype=prepared["base_levels"][0].dtype)
    gamma_sup_t = torch.tensor([pair[1] for pair in gamma_pairs], device=prepared["base_levels"][0].device, dtype=prepared["base_levels"][0].dtype)
    with torch.inference_mode():
        final_level0, debug_holder = adapter.apply_to_levels_gamma_batch(
            expand_feature_levels(prepared["frame_levels"][0], batch_size),
            expand_feature_levels(prepared["memory_levels"], batch_size),
            expand_feature_levels(prepared["base_levels"], batch_size),
            prepared["degradation_mask"].expand(batch_size, *prepared["degradation_mask"].shape[1:]).contiguous(),
            gamma_add=gamma_add_t,
            gamma_sup=gamma_sup_t,
        )
        frame_levels_batch: list[list[torch.Tensor]] = [final_level0]
        for frame_idx in range(1, len(prepared["frame_levels"])):
            frame_levels_batch.append(expand_feature_levels(prepared["frame_levels"][frame_idx], batch_size))
        img_feats: list[torch.Tensor] = []
        for level_idx in range(len(frame_levels_batch[0])):
            frames = torch.stack([frame_levels_batch[frame_idx][level_idx] for frame_idx in range(len(frame_levels_batch))], dim=1)
            bsz, num_frames, cam_count, channels, height, width = frames.shape
            img_feats.append(frames.reshape(bsz, num_frames * cam_count, channels, height, width).contiguous())
        img_metas = merged_frame_metas(prepared, batch_size)
        outputs = forward_backbone_from_reorganized_features(model, img_feats, img_metas, prepared, batch_size)
        head = sw14c.base.sw4_inst.get_pts_bbox_head(model)
        per_pair: dict[tuple[float, float], dict[int, dict[str, torch.Tensor]]] = {pair: {} for pair in gamma_pairs}
        for horizon_s in CORE_HORIZONS:
            if horizon_s == 0:
                pred_dict = {"cls_scores": outputs["cls_score"], "refine_pts": outputs["refine_pts"]}
            else:
                pred_dict = {
                    "cls_scores": outputs["forecast_semantics_list"][horizon_s - 1],
                    "refine_pts": outputs["forecast_points_list"][horizon_s - 1],
                }
            occ_pred, debug_list = sw14c.base.sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
            for batch_idx, pair in enumerate(gamma_pairs):
                dense_scores = debug_list[batch_idx]["dense_occ_after_padding"]
                conf, margin = sw14c.base.occupancy_confidence_and_margin(dense_scores)
                per_pair[pair][horizon_s] = {
                    "raw_semantic": occ_pred[batch_idx],
                    "raw_confidence": conf,
                    "raw_margin": margin,
                    "dense_scores": dense_scores,
                }
        return per_pair, debug_holder


def postprocess_student(sample_index: int, horizon_s: int, per_h: dict[int, dict[str, torch.Tensor]], teacher_bundle: dict[str, Any], sectors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    candidate = sw14c.base.build_candidate()
    teacher_h = teacher_bundle["by_horizon"][horizon_s]
    entry = per_h[horizon_s]
    final_semantic, meta = sw14c.base.run_sw14b_full_postprocess(
        raw_semantic=entry["raw_semantic"].detach().cpu().long(),
        raw_confidence=entry["raw_confidence"].detach().cpu().float(),
        raw_margin=entry["raw_margin"].detach().cpu().float(),
        native_semantic=teacher_h["native_semantic"].long(),
        gt_h=teacher_h["gt_h"].long(),
        gt0=teacher_h["gt0"].long(),
        candidate=candidate,
        sample_index=sample_index,
        horizon_s=horizon_s,
        sectors=sectors,
        load_gpu_dump=sw14c.base.load_gpu_dump,
        agreement_map=teacher_h["agreement"].float(),
    )
    return final_semantic.long(), meta


def cache_per_h_for_cpu_postprocess(per_h: dict[int, dict[str, torch.Tensor]]) -> dict[int, dict[str, torch.Tensor]]:
    cached: dict[int, dict[str, torch.Tensor]] = {}
    for horizon_s, entry in per_h.items():
        cached[int(horizon_s)] = {
            "raw_semantic": entry["raw_semantic"].detach().cpu().long(),
            "raw_confidence": entry["raw_confidence"].detach().cpu().float(),
            "raw_margin": entry["raw_margin"].detach().cpu().float(),
        }
    return cached


def debug_feature_stats(debug: dict[str, torch.Tensor], prepared: dict[str, Any], gamma_add: float, gamma_sup: float, batch_index: int | None = None) -> dict[str, float]:
    def select_batch(tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach().float()
        if batch_index is not None:
            return value[int(batch_index) : int(batch_index) + 1]
        return value

    add_gates = [select_batch(debug[f"add_gate_level{i}"]) for i in range(4)]
    sup_gates = [select_batch(debug[f"sup_gate_level{i}"]) for i in range(4)]
    add_deltas = [select_batch(debug[f"add_delta_level{i}"]) for i in range(4)]
    sup_deltas = [select_batch(debug[f"sup_delta_level{i}"]) for i in range(4)]
    front_add_gate = torch.cat([g[:, FRONT_TRIPLET_INDICES].reshape(-1).cpu() for g in add_gates])
    front_sup_gate = torch.cat([g[:, FRONT_TRIPLET_INDICES].reshape(-1).cpu() for g in sup_gates])
    front_add_delta = torch.cat([d[:, FRONT_TRIPLET_INDICES].abs().reshape(-1).cpu() for d in add_deltas])
    front_sup_delta = torch.cat([d[:, FRONT_TRIPLET_INDICES].abs().reshape(-1).cpu() for d in sup_deltas])
    final_deltas = [select_batch(debug[f"final_level{i}"]) - prepared["base_levels"][i].detach().float() for i in range(4)]
    feature_delta = torch.cat([d[:, FRONT_TRIPLET_INDICES].abs().reshape(-1).cpu() for d in final_deltas])
    rear_feature = max(float(d[:, REAR_CAMERA_INDICES].abs().max().item()) for d in final_deltas)
    return {
        "gate_add_mean_inside_mask_proxy": float(front_add_gate.mean().item()),
        "gate_add_p95_inside_mask_proxy": float(torch.quantile(front_add_gate, 0.95).item()),
        "gate_add_p99_inside_mask_proxy": float(torch.quantile(front_add_gate, 0.99).item()),
        "gate_sup_mean_inside_mask_proxy": float(front_sup_gate.mean().item()),
        "gate_sup_p95_inside_mask_proxy": float(torch.quantile(front_sup_gate, 0.95).item()),
        "gate_sup_p99_inside_mask_proxy": float(torch.quantile(front_sup_gate, 0.99).item()),
        "residual_add_magnitude_inside_mask_proxy": float(front_add_delta.mean().item()),
        "residual_sup_magnitude_inside_mask_proxy": float(front_sup_delta.mean().item()),
        "gamma_applied_feature_delta_inside_mask_proxy": float(feature_delta.mean().item()),
        "gamma_applied_feature_delta_outside_mask_proxy": None,
        "residual_outside_mask_proxy": None,
        "rear_residual": rear_feature,
        "clean_residual": 0.0,
        "final_feature_diff_vs_base": float(feature_delta.mean().item()),
        "gamma_add": float(gamma_add),
        "gamma_sup": float(gamma_sup),
        "feature_mask_alignment_available": False,
    }


def train_variant_b(args: argparse.Namespace, adapter: torch.nn.Module, config: OracleMaskLossConfig, sectors: dict[str, torch.Tensor]) -> tuple[dict[str, Any], list[dict[str, Any]], torch.nn.Module]:
    print("[sw14c-oem] build SparseWorld train runtime", flush=True)
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    freeze_status = sw14c.base.freeze_sparseworld_modules(model)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr))
    train_indices = list(range(args.train_start, args.train_end + 1))
    log_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(args.epochs)):
        sample_pos = 0
        adapter.train()
        for block_id, block_indices in enumerate(chunks(train_indices, int(args.train_block_size))):
            prepared_block: list[dict[str, Any]] = []
            block_start = time.time()
            try:
                for sample_index in block_indices:
                    prepared_block.append(sw14c.prepare_residual_feature_case(model, dataset, sample_index))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(
                    f"[sw14c-oem] prepared train block epoch={epoch} block={block_id} samples={block_indices} "
                    f"elapsed={time.time() - block_start:.2f}s cuda={cuda_mem_mb()}",
                    flush=True,
                )
                for prepared in prepared_block:
                    sample_pos += 1
                    sample_index = int(prepared["sample_index"])
                    sample_start = time.time()
                    bundle = None
                    per_h = None
                    debug = None
                    try:
                        bundle = load_teacher_bundle(sample_index)
                        masks_by_h = build_masks_for_bundle(bundle, sectors)
                        optimizer.zero_grad(set_to_none=True)
                        per_h, debug = run_dual_forward_prepared(model, adapter, prepared, args.train_gamma_add, args.train_gamma_sup, training=True)
                        loss, loss_stats = compute_oracle_mask_loss(per_h, debug, masks_by_h, bundle, sectors, config, mask_variant="dilated")
                        if not bool(torch.isfinite(loss).detach().item()):
                            raise RuntimeError(f"NaN oracle mask loss at sample {sample_index}")
                        loss.backward()
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip)).detach().item())
                        optimizer.step()
                        fstats = debug_feature_stats(debug, prepared, args.train_gamma_add, args.train_gamma_sup)
                        row = {
                            "variant": "oracle_dilated_mask",
                            "epoch": epoch,
                            "block_id": block_id,
                            "sample_index": sample_index,
                            "elapsed_s": time.time() - sample_start,
                            "grad_norm": grad_norm,
                            **loss_stats,
                            **fstats,
                            **cuda_mem_mb(),
                        }
                        log_rows.append(row)
                        health_rows.append({"split": "train", **{k: v for k, v in row.items() if k not in {"total_loss", "grad_norm", "elapsed_s"}}})
                        print(
                            f"[sw14c-oem] train B epoch={epoch} block={block_id} sample={sample_index} "
                            f"({sample_pos}/{len(train_indices)}) loss={row['total_loss']:.4f} "
                            f"add_gate={row['gate_add_mean_inside_mask_proxy']:.4f} sup_gate={row['gate_sup_mean_inside_mask_proxy']:.4f} "
                            f"elapsed={row['elapsed_s']:.2f}s",
                            flush=True,
                        )
                    finally:
                        release(bundle, per_h, debug)
            finally:
                prepared_block.clear()
                release(prepared_block, collect=False, empty_cache=False)
    ckpt = CHECKPOINT_DIR / "sw14c_oracle_mask_variantB_best.pth"
    torch.save(
        checkpoint_payload(
            adapter,
            {
                "stage": "SW14C OracleErrorMaskResidual",
                "variant": "oracle_dilated_mask",
                "train_samples": [args.train_start, args.train_end],
                "val_samples": [args.val_start, args.val_end],
                "epochs": int(args.epochs),
                "gamma_add_train": float(args.train_gamma_add),
                "gamma_sup_train": float(args.train_gamma_sup),
                "loss_config": config.to_dict(),
                "backbone_head_frozen": freeze_status,
                "oracle_gt_mask_diagnostic_only": True,
            },
        ),
        ckpt,
    )
    write_csv(REPORTS_DIR / "sw14c_oracle_mask_training_log.csv", log_rows)
    summary = {
        "phase": "Phase 5 training protocol",
        "decision": "TRAIN_O1_VARIANT_B_COMPLETED",
        "variantB_executed": True,
        "variantA_executed": False,
        "variantC_executed": False,
        "variantD_reused_round2b": True,
        "best_checkpoint": str(ckpt),
        "mean_train_loss": finite_mean([r["total_loss"] for r in log_rows]),
        "elapsed_s": time.time() - start,
        "backbone_head_frozen": freeze_status,
        "gpu_policy": f"block-prepared train with train_block_size={args.train_block_size}; train health reuses debug stats; CUDA cache is not emptied inside blocks",
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
    }
    write_json(REPORTS_DIR / "sw14c_oracle_mask_training_summary.json", summary)
    release(model, dataset, collect=True, empty_cache=True)
    return summary, health_rows, adapter


def gamma_behavior_row(sample_index: int, horizon_s: int, gamma_add: float, gamma_sup: float, teacher_bundle: dict[str, Any], final_semantic: torch.Tensor, meta: dict[str, Any], masks: dict[str, torch.Tensor], fstats: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    teacher_h = teacher_bundle["by_horizon"][horizon_s]
    teacher_final = teacher_h["teacher_final_semantic"].long()
    teacher_eval = teacher_h["teacher_meta"]["final_eval"]
    student_eval = meta["final_eval"]
    teacher_occ = occ_mask(teacher_final)
    student_occ = occ_mask(final_semantic)
    teacher_fn = masks["teacher_FN_region"]
    teacher_fp = masks["teacher_FP_region"]
    correct_occ = masks["teacher_correct_occ_region"]
    correct_free = masks["teacher_correct_free_region"]
    recovered = teacher_fn & student_occ
    suppressed = teacher_fp & ~student_occ
    broken_occ = correct_occ & ~student_occ
    broken_free = correct_free & student_occ
    density_delta = metric_value(student_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta") - metric_value(teacher_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta")
    fp_delta = metric_value(student_eval, "false_positive_delta") - metric_value(teacher_eval, "false_positive_delta")
    native_front = max(1.0, float(((teacher_h["native_semantic"].long() != EMPTY_IDX) & masks["front_region"]).sum().item()))
    front_proxy = float(((final_semantic != EMPTY_IDX) & masks["front_region"]).sum().item()) / native_front
    broken_correct = int(broken_occ.sum().item() + broken_free.sum().item())
    correct_total = int(correct_occ.sum().item() + correct_free.sum().item())
    broken_rate = safe_div(broken_correct, correct_total)
    gamma_row = {
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "gamma_add": float(gamma_add),
        "gamma_sup": float(gamma_sup),
        "front_fn_reduction_over_teacher": metric_value(teacher_eval, "front_sector_false_free_rate_delta") - metric_value(student_eval, "front_sector_false_free_rate_delta"),
        "future_h4h6_fn_reduction_over_teacher": (metric_value(teacher_eval, "false_free_rate") - metric_value(student_eval, "false_free_rate")) if horizon_s in {4, 6} else 0.0,
        "fp_reduction_over_teacher": metric_value(teacher_eval, "false_positive_delta") - metric_value(student_eval, "false_positive_delta"),
        "density_delta_over_teacher": density_delta,
        "false_positive_delta_over_teacher": fp_delta,
        "front_local_proxy": front_proxy,
        "broken_correct_count": broken_correct,
        "broken_correct_rate": broken_rate,
        "sample_joint_success": bool(density_delta <= 0.03 and fp_delta <= 0.01 and front_proxy <= 1.30 and broken_rate <= 0.02),
        "safety_pass": bool(density_delta <= 0.03 and fp_delta <= 0.01 and front_proxy <= 1.30 and broken_rate <= 0.02),
        "gamma_zero_pair_counts_as_improvement": False,
    }
    beh = {
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "gamma_add": float(gamma_add),
        "gamma_sup": float(gamma_sup),
        "teacher_FN_count": int(teacher_fn.sum().item()),
        "recovered_FN_count": int(recovered.sum().item()),
        "recovered_FN_rate": safe_div(int(recovered.sum().item()), int(teacher_fn.sum().item())),
        "added_FP_near_FN": int((teacher_fn & student_occ & masks["gt_free"]).sum().item()),
        "add_branch_gate_mean": fstats["gate_add_mean_inside_mask_proxy"],
        "add_residual_magnitude": fstats["residual_add_magnitude_inside_mask_proxy"],
        "teacher_FP_count": int(teacher_fp.sum().item()),
        "suppressed_FP_count": int(suppressed.sum().item()),
        "suppressed_FP_rate": safe_div(int(suppressed.sum().item()), int(teacher_fp.sum().item())),
        "lost_correct_occ_near_FP": int(broken_occ.sum().item()),
        "suppress_branch_gate_mean": fstats["gate_sup_mean_inside_mask_proxy"],
        "suppress_residual_magnitude": fstats["residual_sup_magnitude_inside_mask_proxy"],
        "broken_correct_occ_count": int(broken_occ.sum().item()),
        "preserve_correct_occ_rate": 1.0 - safe_div(int(broken_occ.sum().item()), int(correct_occ.sum().item())),
        "broken_correct_free_count": int(broken_free.sum().item()),
        "preserve_correct_free_rate": 1.0 - safe_div(int(broken_free.sum().item()), int(correct_free.sum().item())),
        "broken_correct_total": broken_correct,
        "net_FN_improvement": int(recovered.sum().item()),
        "net_FP_improvement": int(suppressed.sum().item()),
        "net_score": int(recovered.sum().item()) + int(suppressed.sum().item()) - broken_correct,
    }
    return gamma_row, beh


def summarize_gamma2d(rows: list[dict[str, Any]], gamma_adds: list[float], gamma_sups: list[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for ga in gamma_adds:
        for gs in gamma_sups:
            pair = [r for r in rows if abs(float(r["gamma_add"]) - ga) < 1e-12 and abs(float(r["gamma_sup"]) - gs) < 1e-12]
            allowed = not (abs(ga) < 1e-12 and abs(gs) < 1e-12)
            front = finite_mean([r["front_fn_reduction_over_teacher"] for r in pair]) or 0.0
            future = finite_mean([r["future_h4h6_fn_reduction_over_teacher"] for r in pair]) or 0.0
            fp_gain = finite_mean([r["fp_reduction_over_teacher"] for r in pair]) or 0.0
            broken = finite_mean([r["broken_correct_rate"] for r in pair]) or 0.0
            net = front + fp_gain - 0.2 * broken
            summary_rows.append(
                {
                    "gamma_add": ga,
                    "gamma_sup": gs,
                    "row_count": len(pair),
                    "sample_count": len({int(r["sample_index"]) for r in pair}),
                    "front_fn_reduction_over_teacher": front,
                    "future_h4h6_fn_reduction_over_teacher": future,
                    "fp_reduction_over_teacher": fp_gain,
                    "density_delta_over_teacher": finite_mean([r["density_delta_over_teacher"] for r in pair]),
                    "false_positive_delta_over_teacher": finite_mean([r["false_positive_delta_over_teacher"] for r in pair]),
                    "front_local_proxy": finite_mean([r["front_local_proxy"] for r in pair]),
                    "broken_correct_rate": broken,
                    "sample_joint_success_rate": safe_div(sum(bool(r["sample_joint_success"]) for r in pair), len(pair)),
                    "safety_pass_all": bool(pair) and all(bool(r["safety_pass"]) for r in pair),
                    "net_improvement": net,
                    "allowed_as_improvement": allowed,
                }
            )
    candidates = [r for r in summary_rows if bool(r["allowed_as_improvement"])]
    safe = [r for r in candidates if bool(r["safety_pass_all"])]
    unsafe_gain = [r for r in candidates if float(r["net_improvement"]) >= 0.002 and not bool(r["safety_pass_all"])]
    if safe and max(float(r["net_improvement"]) for r in safe) >= 0.002:
        decision = "GAMMA_O1_SAFE_IMPROVES_TEACHER"
        best = max(safe, key=lambda r: float(r["net_improvement"]))
    elif safe:
        decision = "GAMMA_O2_SAFE_NO_MEANINGFUL_GAIN"
        best = max(safe, key=lambda r: float(r["net_improvement"]))
    elif unsafe_gain:
        decision = "GAMMA_O3_ONLY_UNSAFE_GAIN"
        best = max(unsafe_gain, key=lambda r: float(r["net_improvement"]))
    elif candidates:
        decision = "GAMMA_O4_NO_SIGNAL"
        best = max(candidates, key=lambda r: float(r["net_improvement"]))
    else:
        decision = "GAMMA_O5_ADD_SUP_CONFLICT"
        best = None
    selection = {
        "phase": "Phase 8 gamma add/sup selection",
        "decision": decision,
        "selected_gamma_add": None if best is None else float(best["gamma_add"]),
        "selected_gamma_sup": None if best is None else float(best["gamma_sup"]),
        "selected_pair_is_nonzero": bool(best and (abs(float(best["gamma_add"])) > 1e-12 or abs(float(best["gamma_sup"])) > 1e-12)),
        "best_summary": best,
        "gamma_zero_pair_counts_as_improvement": False,
        "uses_val_only_for_selection": True,
        "uses_eval_debug_for_selection": False,
        "uses_core100_or_core500_for_selection": False,
        "summary_rows": summary_rows,
    }
    return summary_rows, selection


def fused_val_audit(args: argparse.Namespace, adapter: torch.nn.Module, train_health_rows: list[dict[str, Any]], sectors: dict[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("[sw14c-oem] build SparseWorld fused val audit runtime", flush=True)
    _, dataset, model, _ = sw14c.base.build_runtime(train=True)
    model.eval()
    sw14c.base.freeze_sparseworld_modules(model)
    gamma_adds = parse_float_list(args.gamma_add_candidates)
    gamma_sups = parse_float_list(args.gamma_sup_candidates)
    gamma_pairs = [(ga, gs) for ga in gamma_adds for gs in gamma_sups]
    gamma_batch_size = max(1, int(args.gamma_batch_size))
    val_health_rows: list[dict[str, Any]] = []
    gamma_detail_rows: list[dict[str, Any]] = []
    behavior_by_pair: dict[tuple[float, float], list[dict[str, Any]]] = {(ga, gs): [] for ga in gamma_adds for gs in gamma_sups}
    for pos, sample_index in enumerate(range(args.val_start, args.val_end + 1), start=1):
        start = time.time()
        prepared = None
        bundle = None
        try:
            prepared = sw14c.prepare_residual_feature_case(model, dataset, sample_index)
            bundle = load_teacher_bundle(sample_index)
            masks_by_h = build_masks_for_bundle(bundle, sectors)
            raw_records: list[tuple[float, float, dict[int, dict[str, torch.Tensor]], dict[str, float]]] = []
            if args.gamma_execution_mode == "online_fast":
                for ga, gs in gamma_pairs:
                    per_h = None
                    debug = None
                    try:
                        per_h, debug = run_dual_forward_prepared(model, adapter, prepared, ga, gs, training=False)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        fstats = debug_feature_stats(debug, prepared, ga, gs)
                        if abs(ga - args.train_gamma_add) < 1e-12 and abs(gs - args.train_gamma_sup) < 1e-12:
                            val_health_rows.append({"split": "val", "sample_index": sample_index, "variant": "oracle_dilated_mask", **fstats})
                        if args.interleave_val_postprocess:
                            for h in CORE_HORIZONS:
                                final_semantic, meta = postprocess_student(sample_index, h, per_h, bundle, sectors)
                                grow, brow = gamma_behavior_row(sample_index, h, ga, gs, bundle, final_semantic, meta, masks_by_h[h], fstats)
                                gamma_detail_rows.append(grow)
                                behavior_by_pair[(ga, gs)].append(brow)
                        else:
                            raw_records.append((ga, gs, cache_per_h_for_cpu_postprocess(per_h), fstats))
                    finally:
                        release(per_h, debug)
            else:
                pending_batches = list(chunks(gamma_pairs, gamma_batch_size))
                while pending_batches:
                    pair_batch = pending_batches.pop(0)
                    per_pair = None
                    debug = None
                    try:
                        per_pair, debug = run_dual_forward_prepared_gamma_batch(model, adapter, prepared, pair_batch)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        for batch_idx, (ga, gs) in enumerate(pair_batch):
                            per_h = per_pair[(ga, gs)]
                            fstats = debug_feature_stats(debug, prepared, ga, gs, batch_index=batch_idx)
                            if abs(ga - args.train_gamma_add) < 1e-12 and abs(gs - args.train_gamma_sup) < 1e-12:
                                val_health_rows.append({"split": "val", "sample_index": sample_index, "variant": "oracle_dilated_mask", **fstats})
                            if args.interleave_val_postprocess:
                                for h in CORE_HORIZONS:
                                    final_semantic, meta = postprocess_student(sample_index, h, per_h, bundle, sectors)
                                    grow, brow = gamma_behavior_row(sample_index, h, ga, gs, bundle, final_semantic, meta, masks_by_h[h], fstats)
                                    gamma_detail_rows.append(grow)
                                    behavior_by_pair[(ga, gs)].append(brow)
                            else:
                                raw_records.append((ga, gs, cache_per_h_for_cpu_postprocess(per_h), fstats))
                    except RuntimeError as exc:
                        is_oom = "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower()
                        if not is_oom or len(pair_batch) <= 1:
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        mid = max(1, len(pair_batch) // 2)
                        pending_batches.insert(0, pair_batch[mid:])
                        pending_batches.insert(0, pair_batch[:mid])
                        print(
                            f"[sw14c-oem] gamma batch OOM fallback sample={sample_index} "
                            f"old_batch={len(pair_batch)} new_batches={[len(pair_batch[:mid]), len(pair_batch[mid:])]}",
                            flush=True,
                        )
                    finally:
                        release(per_pair, debug)
            if raw_records:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                for ga, gs, per_h_cpu, fstats in raw_records:
                    for h in CORE_HORIZONS:
                        final_semantic, meta = postprocess_student(sample_index, h, per_h_cpu, bundle, sectors)
                        grow, brow = gamma_behavior_row(sample_index, h, ga, gs, bundle, final_semantic, meta, masks_by_h[h], fstats)
                        gamma_detail_rows.append(grow)
                        behavior_by_pair[(ga, gs)].append(brow)
                raw_records.clear()
            print(
                f"[sw14c-oem] fused val sample={sample_index} ({pos}/{args.val_end - args.val_start + 1}) "
                f"gamma_pairs={len(gamma_pairs)} gamma_mode={args.gamma_execution_mode} gamma_batch_size={gamma_batch_size} "
                f"elapsed={time.time() - start:.2f}s cuda={cuda_mem_mb()}",
                flush=True,
            )
        finally:
            release(prepared, bundle, collect=(pos % 10 == 0), empty_cache=(pos % 10 == 0))
    summary_rows, gamma_selection = summarize_gamma2d(gamma_detail_rows, gamma_adds, gamma_sups)
    gamma_selection["gamma_execution_mode"] = args.gamma_execution_mode
    gamma_selection["gamma_batch_size"] = gamma_batch_size
    gamma_selection["direct_batch_path_available"] = True
    gamma_selection["online_fast_path_used_for_time_to_result"] = args.gamma_execution_mode == "online_fast"
    gamma_selection["val_postprocess_deferred_until_after_sample_gpu_forwards"] = not bool(args.interleave_val_postprocess)
    gamma_selection["val_gpu_flattening_rework"] = "sample-level raw GPU outputs are cached on CPU before F3/FrontCap postprocess unless --interleave-val-postprocess is set"
    write_json(
        REPORTS_DIR / "sw14c_oracle_mask_gpu_val_rework.json",
        {
            "implemented_after_full_diagnostic_run": True,
            "affects_existing_selection_outputs": False,
            "default_future_val_path": "defer CPU F3/FrontCap postprocess until all gamma GPU forwards for the sample have completed",
            "legacy_interleaved_path_flag": "--interleave-val-postprocess",
            "direct_batch_path_available": True,
            "sparseworld_online_batch_limitation": "SparseWorld simple_test_online asserts batch_size=1; direct-head batching is implemented without modifying SparseWorld files but was not faster in smoke tests.",
            "next_gpu_utilization_target": "prepared feature cache + direct batched head/raw-output cache + CPU postprocess separation before future formal reruns",
        },
    )
    write_csv(REPORTS_DIR / "sw14c_oracle_mask_gamma2d_sweep_val.csv", summary_rows)
    write_csv(REPORTS_DIR / "sw14c_oracle_mask_gamma2d_sweep_val_detail.csv", gamma_detail_rows)
    write_csv(REPORTS_DIR / "sw14c_oracle_mask_epoch_val_metrics.csv", summary_rows)
    write_json(REPORTS_DIR / "sw14c_oracle_mask_gamma_selection.json", gamma_selection)
    selected_pair = (float(gamma_selection.get("selected_gamma_add") or args.train_gamma_add), float(gamma_selection.get("selected_gamma_sup") or args.train_gamma_sup))
    behavior_rows = behavior_by_pair.get(selected_pair, [])
    write_csv(REPORTS_DIR / "sw14c_oracle_mask_teacher_error_behavior_val.csv", behavior_rows)
    behavior_summary = summarize_behavior(behavior_rows, selected_pair)
    write_json(REPORTS_DIR / "sw14c_oracle_mask_teacher_error_behavior_summary.json", behavior_summary)
    write_csv(REPORTS_DIR / "sw14c_oracle_mask_residual_health_train.csv", train_health_rows)
    write_csv(REPORTS_DIR / "sw14c_oracle_mask_residual_health_val.csv", val_health_rows)
    health_summary = summarize_health(train_health_rows, val_health_rows)
    write_json(REPORTS_DIR / "sw14c_oracle_mask_residual_health_summary.json", health_summary)
    release(model, dataset, collect=True, empty_cache=True)
    return health_summary, behavior_summary, gamma_selection


def summarize_health(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = train_rows + val_rows
    add_gate = finite_mean([r.get("gate_add_mean_inside_mask_proxy") for r in rows]) or 0.0
    sup_gate = finite_mean([r.get("gate_sup_mean_inside_mask_proxy") for r in rows]) or 0.0
    feature_delta = finite_mean([r.get("gamma_applied_feature_delta_inside_mask_proxy") for r in rows]) or 0.0
    rear = max([float(r.get("rear_residual") or 0.0) for r in rows], default=0.0)
    clean = max([float(r.get("clean_residual") or 0.0) for r in rows], default=0.0)
    if rear > 1e-12 or clean > 1e-12:
        decision = "HEALTH_O3_MASK_OUT_LEAKAGE"
    elif feature_delta < 1e-8 or (add_gate < 0.005 and sup_gate < 0.005):
        decision = "HEALTH_O2_STILL_NOOP"
    elif max(add_gate, sup_gate) > 0.3:
        decision = "HEALTH_O4_TOO_AGGRESSIVE"
    elif safe_div(max(add_gate, sup_gate), max(1e-9, min(add_gate, sup_gate))) > 10.0:
        decision = "HEALTH_O5_BRANCH_IMBALANCE"
    else:
        decision = "HEALTH_O1_MASKED_RESIDUAL_ACTIVE"
    return {
        "phase": "Phase 6 residual health audit",
        "decision": decision,
        "gate_add_mean_inside_mask": add_gate,
        "gate_sup_mean_inside_mask": sup_gate,
        "residual_add_magnitude_inside_mask": finite_mean([r.get("residual_add_magnitude_inside_mask_proxy") for r in rows]),
        "residual_sup_magnitude_inside_mask": finite_mean([r.get("residual_sup_magnitude_inside_mask_proxy") for r in rows]),
        "gamma_applied_feature_delta_inside_mask": feature_delta,
        "gamma_applied_feature_delta_outside_mask": None,
        "rear_residual": rear,
        "clean_residual": clean,
        "feature_mask_alignment_available": False,
        "mask_out_residual_zero_method": "output-space preserve loss; exact voxel-to-feature mask unavailable",
    }


def summarize_behavior(rows: list[dict[str, Any]], pair: tuple[float, float]) -> dict[str, Any]:
    recovered = sum(int(r["recovered_FN_count"]) for r in rows)
    suppressed = sum(int(r["suppressed_FP_count"]) for r in rows)
    broken = sum(int(r["broken_correct_total"]) for r in rows)
    fn_total = sum(int(r["teacher_FN_count"]) for r in rows)
    fp_total = sum(int(r["teacher_FP_count"]) for r in rows)
    if recovered == 0 and suppressed == 0:
        decision = "BEHAV_O5_NO_MEANINGFUL_ACTION"
    elif broken > max(recovered + suppressed, 1):
        decision = "BEHAV_O4_BREAKS_CORRECT_TOO_MUCH"
    elif recovered > 0 and broken > recovered:
        decision = "BEHAV_O2_FN_RECOVERY_ADDS_TOO_MUCH_FP"
    elif suppressed > 0 and recovered == 0:
        decision = "BEHAV_O3_FP_SUPPRESSION_LOSES_RECALL"
    else:
        decision = "BEHAV_O1_TARGETED_FN_FP_IMPROVEMENT"
    return {
        "phase": "Phase 7 teacher error behavior audit",
        "decision": decision,
        "gamma_add": pair[0],
        "gamma_sup": pair[1],
        "row_count": len(rows),
        "recovered_FN_total": recovered,
        "recovered_FN_rate": safe_div(recovered, fn_total),
        "suppressed_FP_total": suppressed,
        "suppressed_FP_rate": safe_div(suppressed, fp_total),
        "broken_correct_total": broken,
        "net_score_total": sum(int(r["net_score"]) for r in rows),
        "add_branch_gate_mean": finite_mean([r["add_branch_gate_mean"] for r in rows]),
        "suppress_branch_gate_mean": finite_mean([r["suppress_branch_gate_mean"] for r in rows]),
    }


def proxy_feasibility(mask_summary: dict[str, Any], gamma_selection: dict[str, Any], final_oracle_success: bool) -> dict[str, Any]:
    val = mask_summary["val"]
    if not final_oracle_success:
        decision = "PROXY_O4_PROXY_NOT_EXECUTED_ORACLE_FAILED"
    elif val["proxy_precision_vs_oracle"] < 0.10:
        decision = "PROXY_O3_PROXY_TOO_NOISY"
    elif gamma_selection["decision"] == "GAMMA_O1_SAFE_IMPROVES_TEACHER":
        decision = "PROXY_O2_ORACLE_ONLY_NOT_DEPLOYABLE"
    else:
        decision = "PROXY_O4_PROXY_NOT_EXECUTED_ORACLE_FAILED"
    rows = [
        {
            "comparison": "proxy_vs_oracle_mask",
            "proxy_precision": val["proxy_precision_vs_oracle"],
            "proxy_recall": val["proxy_recall_vs_oracle"],
            "oracle_gamma_decision": gamma_selection["decision"],
            "proxy_trained": False,
            "reason": "proxy training is deferred unless oracle_dilated achieves safe gain",
        }
    ]
    write_csv(REPORTS_DIR / "sw14c_proxy_mask_feasibility.csv", rows)
    payload = {
        "phase": "Phase 9 proxy mask feasibility",
        "decision": decision,
        "proxy_vs_oracle_precision": val["proxy_precision_vs_oracle"],
        "proxy_vs_oracle_recall": val["proxy_recall_vs_oracle"],
        "proxy_mask_uses_gt": False,
        "proxy_training_executed": False,
    }
    write_json(REPORTS_DIR / "sw14c_proxy_mask_feasibility_summary.json", payload)
    return payload


def final_decision(init: dict[str, Any], ub: dict[str, Any], health: dict[str, Any], behavior: dict[str, Any], gamma: dict[str, Any], proxy: dict[str, Any]) -> dict[str, Any]:
    best = gamma.get("best_summary") or {}
    oracle_safe_gain = gamma["decision"] == "GAMMA_O1_SAFE_IMPROVES_TEACHER" and float(best.get("net_improvement") or 0.0) >= 0.002
    if init["decision"] != "ORACLE_INIT_READY":
        decision = "SW14C_OEM_7_PROTOCOL_VIOLATION"
    elif health["decision"] == "HEALTH_O2_STILL_NOOP" and not oracle_safe_gain:
        decision = "SW14C_OEM_0_ORACLE_MASK_FAIL_KEEP_SW13"
    elif ub["decision"] in {"ORACLE_UB_2_LOW_POTENTIAL", "ORACLE_UB_4_CANDIDATE_SET_TOO_SMALL"}:
        decision = "SW14C_OEM_1_ORACLE_MASK_LOW_POTENTIAL"
    elif behavior["decision"] == "BEHAV_O4_BREAKS_CORRECT_TOO_MUCH":
        decision = "SW14C_OEM_6_DUAL_BRANCH_BREAKS_CORRECT"
    elif gamma["decision"] == "GAMMA_O3_ONLY_UNSAFE_GAIN":
        decision = "SW14C_OEM_2_ORACLE_MASK_SIGNAL_UNSAFE"
    elif oracle_safe_gain and proxy["decision"] == "PROXY_O1_FEASIBLE_READY_NEXT_STAGE":
        decision = "SW14C_OEM_4_PROXY_MASK_SAFE_GAIN_READY_DEBUG"
    elif oracle_safe_gain:
        decision = "SW14C_OEM_5_PROXY_FAIL_BUT_ORACLE_SUCCESS"
    else:
        decision = "SW14C_OEM_0_ORACLE_MASK_FAIL_KEEP_SW13"
    eval_debug = decision == "SW14C_OEM_4_PROXY_MASK_SAFE_GAIN_READY_DEBUG"
    learned_mask = decision in {"SW14C_OEM_3_ORACLE_MASK_SAFE_GAIN", "SW14C_OEM_5_PROXY_FAIL_BUT_ORACLE_SUCCESS"}
    if decision in {"SW14C_OEM_0_ORACLE_MASK_FAIL_KEEP_SW13", "SW14C_OEM_1_ORACLE_MASK_LOW_POTENTIAL"}:
        next_action = "Stop SW14C feature residual as a main path; move to SW14D candidate-level reranker or close on SW13."
    elif decision == "SW14C_OEM_2_ORACLE_MASK_SIGNAL_UNSAFE":
        next_action = "Do not run eval_debug. Residual has oracle signal but unsafe behavior; candidate-level reranking is favored."
    elif decision in {"SW14C_OEM_3_ORACLE_MASK_SAFE_GAIN", "SW14C_OEM_5_PROXY_FAIL_BUT_ORACLE_SUCCESS"}:
        next_action = "Oracle upper-bound works but is not deployable; next stage is learned/no-GT error mask predictor, not eval_debug."
    elif decision == "SW14C_OEM_4_PROXY_MASK_SAFE_GAIN_READY_DEBUG":
        next_action = "Freeze proxy mask checkpoint/gamma and run eval_debug only as a diagnostic."
    elif decision == "SW14C_OEM_6_DUAL_BRANCH_BREAKS_CORRECT":
        next_action = "Stop or redesign dual branch mask constraints before any eval_debug."
    else:
        next_action = "Discard this run and fix the protocol violation."
    payload = {
        "decision": decision,
        "oracle_has_potential": ub["decision"] in {"ORACLE_UB_1_HIGH_POTENTIAL", "ORACLE_UB_3_ONLY_UNSAFE_POTENTIAL"},
        "masked_residual_noop_fixed": health["decision"] != "HEALTH_O2_STILL_NOOP",
        "fn_fp_behavior_targeted": behavior["decision"] == "BEHAV_O1_TARGETED_FN_FP_IMPROVEMENT",
        "proxy_mask_feasible": proxy["decision"] == "PROXY_O1_FEASIBLE_READY_NEXT_STAGE",
        "whether_eval_debug_allowed": eval_debug,
        "whether_learned_error_mask_next": learned_mask,
        "whether_use_in_resume": False,
        "sw13_remains_main_result": True,
        "best_checkpoint": str(CHECKPOINT_DIR / "sw14c_oracle_mask_variantB_best.pth"),
        "selected_gamma_add": gamma.get("selected_gamma_add"),
        "selected_gamma_sup": gamma.get("selected_gamma_sup"),
        "val_net_improvement": best.get("net_improvement"),
        "recommended_next_action": next_action,
        "gpu_flattening_policy": (
            f"train: block-prepared sample-major GPU reuse; val: {gamma.get('gamma_execution_mode', 'unknown')} "
            "with fused health/gamma/behavior audit, no eval_debug/core100, no large cache rebuild; "
            "direct_batch path is implemented but only used when it is faster or explicitly requested"
        ),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "get_occ_sha256_after": sha256(GET_OCC_PATH),
        "f3_script_sha256_after": sha256(Path(sw14c.base.sw13c_fix.__file__)),
        "frontcap_script_sha256_after": sha256(Path(sw14c.base.frontcap50.__file__)),
    }
    assert payload["decision"] in FINAL_ENUMS
    write_json(REPORTS_DIR / "sw14c_oracle_error_mask_residual_final_decision.json", payload)
    return payload


def plot_outputs(mask_summary: dict[str, Any], ub: dict[str, Any], health: dict[str, Any], behavior: dict[str, Any], gamma: dict[str, Any], proxy: dict[str, Any], final: dict[str, Any]) -> None:
    def bar(path: Path, title: str, labels: list[str], values: list[float]) -> None:
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values)
        plt.title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
    bar(FIGURES_DIR / "sw14c_oracle_mask_region_stats.png", "Oracle mask uses GT; diagnostic only", ["FN", "FP", "strict", "proxy"], [mask_summary["val"]["teacher_FN_total"], mask_summary["val"]["teacher_FP_total"], mask_summary["val"]["oracle_strict_total"], mask_summary["val"]["proxy_mask_total"]])
    bar(FIGURES_DIR / "sw14c_oracle_upper_bound_bar.png", "Oracle candidate upper bound vs SW13", ["front", "FP", "safe_rate"], [ub["val_front_gain_mean"], ub["val_fp_gain_mean"], ub["val_safety_pass_rate"]])
    bar(FIGURES_DIR / "sw14c_masked_residual_health.png", "Masked residual health", ["add_gate", "sup_gate", "feat_delta"], [health["gate_add_mean_inside_mask"], health["gate_sup_mean_inside_mask"], health["gamma_applied_feature_delta_inside_mask"]])
    bar(FIGURES_DIR / "sw14c_teacher_error_behavior_bar.png", "Teacher error behavior", ["FN recovered", "FP suppressed", "correct broken"], [behavior["recovered_FN_total"], behavior["suppressed_FP_total"], behavior["broken_correct_total"]])
    rows = gamma["summary_rows"]
    adds = sorted({float(r["gamma_add"]) for r in rows})
    sups = sorted({float(r["gamma_sup"]) for r in rows})
    heat = np.zeros((len(adds), len(sups)), dtype=float)
    for r in rows:
        heat[adds.index(float(r["gamma_add"])), sups.index(float(r["gamma_sup"]))] = float(r["net_improvement"])
    plt.figure(figsize=(6, 5))
    plt.imshow(heat, origin="lower", aspect="auto")
    plt.xticks(range(len(sups)), sups)
    plt.yticks(range(len(adds)), adds)
    plt.xlabel("gamma_sup")
    plt.ylabel("gamma_add")
    plt.colorbar(label="net improvement")
    plt.title("Oracle mask gamma2d heatmap; no eval_debug used")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14c_gamma2d_heatmap.png", dpi=180)
    plt.savefig(FIGURES_DIR / "sw14c_oracle_mask_gamma2d_heatmap.png", dpi=180)
    plt.close()
    bar(FIGURES_DIR / "sw14c_oracle_vs_proxy_comparison.png", "Oracle GT mask vs proxy no-GT mask", ["proxy precision", "proxy recall"], [proxy["proxy_vs_oracle_precision"], proxy["proxy_vs_oracle_recall"]])
    bar(FIGURES_DIR / "sw14c_final_decision_flow.png", "SW13 remains baseline; final decision", ["oracle UB", "health", "gamma", "proxy"], [1.0 if ub["decision"] else 0.0, 1.0 if health["decision"] else 0.0, 1.0 if gamma["decision"] else 0.0, 1.0 if proxy["decision"] else 0.0])


def write_report(mask_summary: dict[str, Any], ub: dict[str, Any], health: dict[str, Any], behavior: dict[str, Any], gamma: dict[str, Any], proxy: dict[str, Any], final: dict[str, Any]) -> None:
    write_md(
        REPORTS_DIR / "stage_sw14c_oracle_error_mask_residual_report.md",
        "\n".join(
            [
                "# SW14C Oracle Error Mask Residual Report",
                "",
                "Oracle masks use GT and are diagnostic upper bounds only. SW13C-Fix + FrontCap remains the main result.",
                "",
                f"- mask decision: `{mask_summary['decision']}`",
                f"- oracle candidate upper bound: `{ub['decision']}`",
                f"- residual health: `{health['decision']}`",
                f"- teacher error behavior: `{behavior['decision']}`",
                f"- gamma selection: `{gamma['decision']}`",
                f"- proxy feasibility: `{proxy['decision']}`",
                f"- final decision: `{final['decision']}`",
                "",
                "No eval_debug/core100/core500 was run. get_occ, F3, and FrontCap were not modified.",
                "",
                final["recommended_next_action"],
            ]
        )
        + "\n",
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    sw14c.configure_cuda_for_throughput()
    sectors = {key: value.cpu() for key, value in sw14c.base.sw13c_fix.sw7.build_sector_masks().items()}
    init = phase0_inherited_state(args)
    mask_summary = phase1_masks(args, sectors)
    adapter = build_dual_branch_adapter()
    arch = phase2_adapter_init(adapter)
    ub = phase3_oracle_upper_bound(args, sectors)
    config = OracleMaskLossConfig()
    write_json(REPORTS_DIR / "sw14c_oracle_mask_loss_config.json", config.to_dict())
    write_json(REPORTS_DIR / "sw14c_oracle_mask_loss_terms_schema.json", {"loss_terms": LOSS_TERMS_SCHEMA})
    if init["decision"] != "ORACLE_INIT_READY" or arch["decision"] != "ARCH_O1_READY":
        training = {"decision": "TRAIN_O0_BLOCKED", "best_checkpoint": None}
        write_json(REPORTS_DIR / "sw14c_oracle_mask_training_summary.json", training)
        health = {"decision": "HEALTH_O3_MASK_OUT_LEAKAGE", "gate_add_mean_inside_mask": 0.0, "gate_sup_mean_inside_mask": 0.0, "gamma_applied_feature_delta_inside_mask": 0.0}
        behavior = {"decision": "BEHAV_O5_NO_MEANINGFUL_ACTION", "recovered_FN_total": 0, "suppressed_FP_total": 0, "broken_correct_total": 0}
        gamma = {"decision": "GAMMA_O4_NO_SIGNAL", "selected_gamma_add": None, "selected_gamma_sup": None, "best_summary": {}, "summary_rows": []}
        proxy = proxy_feasibility(mask_summary, gamma, False)
    else:
        training, train_health_rows, adapter = train_variant_b(args, adapter, config, sectors)
        health, behavior, gamma = fused_val_audit(args, adapter, train_health_rows, sectors)
        proxy = proxy_feasibility(mask_summary, gamma, gamma["decision"] == "GAMMA_O1_SAFE_IMPROVES_TEACHER")
    final = final_decision(init, ub, health, behavior, gamma, proxy)
    plot_outputs(mask_summary, ub, health, behavior, gamma, proxy, final)
    write_report(mask_summary, ub, health, behavior, gamma, proxy, final)
    print(f"[sw14c-oem] final decision {final['decision']}", flush=True)


if __name__ == "__main__":
    main()
