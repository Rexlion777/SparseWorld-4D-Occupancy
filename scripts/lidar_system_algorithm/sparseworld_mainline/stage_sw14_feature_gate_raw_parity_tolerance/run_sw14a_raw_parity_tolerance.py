from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv.parallel import collate as collate_fn


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_raw_parity_tolerance"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_raw_parity_tolerance"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14_feature_gate_raw_parity_tolerance"

BASE_SW14 = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/run_sw14_main.py"
BASE_CAUSAL = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_causal_replay/run_sw14a_causal_replay.py"

EVAL500_ARTIFACTS = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
SW13A_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"

EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14 = load_module("sw14_raw_tol_base", BASE_SW14)
causal = load_module("sw14_raw_tol_causal", BASE_CAUSAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14A raw parity tolerance and F3/FrontCap parity audit")
    parser.add_argument("--sample-start", type=int, default=100)
    parser.add_argument("--sample-end", type=int, default=119)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, SCRIPTS_DIR, ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(obj: Any) -> Any:
    return sw14.normalize(obj)


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
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize(row))


def mean_of(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def max_of(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(max(float(row[key]) for row in rows))


def candidate():
    return sw14.teacher_candidates()[0]


def front_mask() -> torch.Tensor:
    return sw14.sw7.build_sector_masks()["front"].cpu().bool()


def final_output_path(sample_index: int, horizon_s: int) -> Path:
    candidate_name, frontcap_variant = sw14.candidate_key_and_frontcap(candidate())
    return EVAL500_ARTIFACTS / "shard_100_500" / "final_outputs" / f"{candidate_name}__{frontcap_variant}__sample{sample_index:03d}_h{horizon_s}.npz"


def raw_dump_path(sample_index: int, horizon_s: int, variant_label: str) -> Path:
    cand = candidate()
    return EVAL500_ARTIFACTS / "gpu_phase_dumps" / f"{cand.perturbation_id}__{variant_label}__sample{sample_index:03d}_h{horizon_s}.npz"


def load_npz_torch(path: Path) -> dict[str, torch.Tensor]:
    payload = np.load(path, allow_pickle=True)
    return {key: torch.from_numpy(payload[key]) for key in payload.files}


def check_required_artifacts(sample_ids: list[int]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    cand = candidate()
    for sample_index in sample_ids:
        cache_path = SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_index:03d}.pt"
        if not cache_path.exists():
            missing.append(str(cache_path))
        for horizon_s in CORE_HORIZONS:
            for variant_label in ["native_baseline", cand.base_repair_variant, "R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3"]:
                dump_path = raw_dump_path(sample_index, horizon_s, variant_label)
                if not dump_path.exists():
                    missing.append(str(dump_path))
            out_path = final_output_path(sample_index, horizon_s)
            if not out_path.exists():
                missing.append(str(out_path))
    return len(missing) == 0, missing


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic != EMPTY_IDX


def occ_count(semantic: torch.Tensor) -> int:
    return int(occ_mask(semantic).sum().item())


def occ_diff_count(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((occ_mask(a) ^ occ_mask(b)).sum().item())


def occ_diff_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((occ_mask(a) ^ occ_mask(b)).float().mean().item())


def semantic_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def front_local_proxy(pred: torch.Tensor, native: torch.Tensor) -> float:
    front = front_mask()
    return sw14.safe_div(float((occ_mask(pred) & front).sum().item()), max(1.0, float((occ_mask(native) & front).sum().item())))


def build_metric_row(pred: torch.Tensor, gt_h: torch.Tensor, gt0: torch.Tensor, native: torch.Tensor, horizon_s: int) -> dict[str, Any]:
    sectors = {name: tensor.cpu().bool() for name, tensor in sw14.sw7.build_sector_masks().items()}
    pred_row = sw14.sw13c_fix.build_eval_row(pred.long(), gt_h.long(), gt0.long(), candidate().perturbation_id, horizon_s, sectors, native.long())
    native_row = sw14.sw13c_fix.build_eval_row(native.long(), gt_h.long(), gt0.long(), candidate().perturbation_id, horizon_s, sectors, native.long())
    out = dict(pred_row)
    out["front_sector_false_free_rate_delta"] = float(pred_row["front_sector_false_free_rate"]) - float(native_row["front_sector_false_free_rate"])
    out["future_h4_h6_false_free_rate_delta"] = (
        float(pred_row["future_h4_h6_false_free_rate"]) - float(native_row["future_h4_h6_false_free_rate"])
        if horizon_s in {4, 6}
        else 0.0
    )
    out["false_positive_delta"] = float(pred_row["false_positive_rate"]) - float(native_row["false_positive_rate"])
    out["wrong_class_delta"] = float(pred_row["wrong_class_rate"]) - float(native_row["wrong_class_rate"])
    out["front_local_density_proxy"] = front_local_proxy(pred, native)
    return out


def candidate_fix_variant() -> Any:
    for base in sw14.sw13c_fix.base_specs():
        if base.perturbation_id == candidate().perturbation_id and base.raw_variant_label == candidate().base_repair_variant:
            for variant in sw14.sw13c_fix.fix_variants_for(base):
                if variant.label == candidate().base_variant:
                    return variant
    raise KeyError(candidate().base_variant)


def run_raw_replay(model: Any, dataset: Any, sample_index: int) -> dict[int, dict[str, Any]]:
    cand = candidate()
    variant = sw14.variant_spec_for_candidate(cand)
    cache = torch.load(SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_index:03d}.pt", map_location="cpu", weights_only=False)
    raw_sample, batch_clean = sw14.sw2.extract_sample_batch(dataset, sample_index, collate_fn)
    sample_unwrapped = sw14.sw2.unwrap(raw_sample)
    meta = sw14.sw13a.get_meta_dict(sample_unwrapped)
    img_metas_curr = sw14.sw13a.clone_meta_for_indices(meta, list(range(6)))
    batch_deg = sw14.perturbation_batch_compatible(batch_clean)
    batch_deg = sw14.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, sw14.sw81.sw5_engine.build_catalog()[cand.perturbation_id])[0]
    moved_deg = sw14.sw2.move_to_cuda(batch_deg)
    img_deg = sw14.sw13a.frame_img_tensor(moved_deg)
    with torch.no_grad():
        current_levels = [feat.detach().cpu().float() for feat in model.extract_feat(img_deg[:, :6], img_metas_curr)]
    memory_levels = sw14.build_memory_levels(cache, variant)
    hard_levels = sw14.build_hard_repair_levels([feat.clone() for feat in current_levels], memory_levels, cand)
    per_h = causal.run_levels_forward(model, sample_unwrapped, batch_deg, hard_levels)
    out: dict[int, dict[str, Any]] = {}
    for horizon_s in CORE_HORIZONS:
        semantic, dense_debug = sw14.sw13b.dense_debug_for_case(model, per_h[horizon_s]["pred_dict"])
        conf = sw14.sw13b.confidence_maps(dense_debug)
        out[horizon_s] = {
            "semantic": semantic.long(),
            "gt_h": per_h[horizon_s]["gt_h"].long(),
            "gt0": per_h[horizon_s]["gt0"].long(),
            "occ_conf": conf["occ_conf"].float(),
            "top1_margin": conf["top1_margin"].float(),
        }
    return out


def raw_parity_tolerance_audit(sample_ids: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    for sample_index in sample_ids:
        replay_by_h = run_raw_replay(model, dataset, sample_index)
        for horizon_s in CORE_HORIZONS:
            replay = replay_by_h[horizon_s]
            sw13_raw = load_npz_torch(raw_dump_path(sample_index, horizon_s, candidate().base_repair_variant))
            native = load_npz_torch(raw_dump_path(sample_index, horizon_s, "native_baseline"))
            gt_h = sw13_raw["gt_h"].long()
            gt0 = sw13_raw["gt0"].long()
            replay_eval = build_metric_row(replay["semantic"], gt_h, gt0, native["semantic"].long(), horizon_s)
            sw13_eval = build_metric_row(sw13_raw["semantic"].long(), gt_h, gt0, native["semantic"].long(), horizon_s)
            row = {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "raw_occ_count_replay": occ_count(replay["semantic"]),
                "raw_occ_count_sw13": occ_count(sw13_raw["semantic"]),
                "raw_occ_count_diff_ratio": sw14.safe_div(
                    abs(occ_count(replay["semantic"]) - occ_count(sw13_raw["semantic"])),
                    max(1.0, float(occ_count(sw13_raw["semantic"]))),
                ),
                "replay_vs_sw13_raw_occ_diff_ratio": occ_diff_ratio(replay["semantic"], sw13_raw["semantic"]),
                "semantic_abs_diff_mean": semantic_abs_diff(replay["semantic"], sw13_raw["semantic"]),
                "front_sector_false_free_rate_delta_replay": float(replay_eval["front_sector_false_free_rate_delta"]),
                "front_sector_false_free_rate_delta_sw13": float(sw13_eval["front_sector_false_free_rate_delta"]),
                "front_false_free_delta_abs_diff": abs(float(replay_eval["front_sector_false_free_rate_delta"]) - float(sw13_eval["front_sector_false_free_rate_delta"])),
                "pred_gt_density_delta_replay": float(replay_eval["pred_gt_density_delta"]),
                "pred_gt_density_delta_sw13": float(sw13_eval["pred_gt_density_delta"]),
                "pred_gt_density_delta_abs_diff": abs(float(replay_eval["pred_gt_density_delta"]) - float(sw13_eval["pred_gt_density_delta"])),
                "false_positive_delta_replay": float(replay_eval["false_positive_delta"]),
                "false_positive_delta_sw13": float(sw13_eval["false_positive_delta"]),
                "false_positive_delta_abs_diff": abs(float(replay_eval["false_positive_delta"]) - float(sw13_eval["false_positive_delta"])),
                "wrong_class_delta_replay": float(replay_eval["wrong_class_delta"]),
                "wrong_class_delta_sw13": float(sw13_eval["wrong_class_delta"]),
                "wrong_class_delta_abs_diff": abs(float(replay_eval["wrong_class_delta"]) - float(sw13_eval["wrong_class_delta"])),
            }
            rows.append(row)
    del model
    write_csv(REPORTS_DIR / "sw14a_raw_parity_tolerance_metrics.csv", rows)
    summary = {
        "sample_count": len(sample_ids),
        "row_count": len(rows),
        "mean_replay_vs_sw13_raw_occ_diff_ratio": mean_of(rows, "replay_vs_sw13_raw_occ_diff_ratio"),
        "mean_semantic_abs_diff_mean": mean_of(rows, "semantic_abs_diff_mean"),
        "mean_pred_gt_density_delta_abs_diff": mean_of(rows, "pred_gt_density_delta_abs_diff"),
        "mean_front_false_free_delta_abs_diff": mean_of(rows, "front_false_free_delta_abs_diff"),
        "mean_false_positive_delta_abs_diff": mean_of(rows, "false_positive_delta_abs_diff"),
        "max_replay_vs_sw13_raw_occ_diff_ratio": max_of(rows, "replay_vs_sw13_raw_occ_diff_ratio"),
        "top_mismatch_samples": [
            {"sample_index": row["sample_index"], "horizon_s": row["horizon_s"], "replay_vs_sw13_raw_occ_diff_ratio": row["replay_vs_sw13_raw_occ_diff_ratio"]}
            for row in sorted(rows, key=lambda item: float(item["replay_vs_sw13_raw_occ_diff_ratio"]), reverse=True)[:10]
        ],
        "strict_thresholds": {
            "mean_replay_vs_sw13_raw_occ_diff_ratio": 0.001,
            "mean_semantic_abs_diff_mean": 0.003,
            "mean_pred_gt_density_delta_abs_diff": 0.005,
            "mean_front_false_free_delta_abs_diff": 0.005,
            "mean_false_positive_delta_abs_diff": 0.0025,
        },
        "soft_thresholds": {
            "mean_replay_vs_sw13_raw_occ_diff_ratio": 0.002,
            "mean_semantic_abs_diff_mean": 0.005,
            "mean_pred_gt_density_delta_abs_diff": 0.01,
            "mean_front_false_free_delta_abs_diff": 0.01,
            "mean_false_positive_delta_abs_diff": 0.005,
        },
    }
    strict_pass = (
        summary["mean_replay_vs_sw13_raw_occ_diff_ratio"] <= summary["strict_thresholds"]["mean_replay_vs_sw13_raw_occ_diff_ratio"]
        and summary["mean_semantic_abs_diff_mean"] <= summary["strict_thresholds"]["mean_semantic_abs_diff_mean"]
        and summary["mean_pred_gt_density_delta_abs_diff"] <= summary["strict_thresholds"]["mean_pred_gt_density_delta_abs_diff"]
        and summary["mean_front_false_free_delta_abs_diff"] <= summary["strict_thresholds"]["mean_front_false_free_delta_abs_diff"]
        and summary["mean_false_positive_delta_abs_diff"] <= summary["strict_thresholds"]["mean_false_positive_delta_abs_diff"]
    )
    soft_pass = (
        summary["mean_replay_vs_sw13_raw_occ_diff_ratio"] <= summary["soft_thresholds"]["mean_replay_vs_sw13_raw_occ_diff_ratio"]
        and summary["mean_semantic_abs_diff_mean"] <= summary["soft_thresholds"]["mean_semantic_abs_diff_mean"]
        and summary["mean_pred_gt_density_delta_abs_diff"] <= summary["soft_thresholds"]["mean_pred_gt_density_delta_abs_diff"]
        and summary["mean_front_false_free_delta_abs_diff"] <= summary["soft_thresholds"]["mean_front_false_free_delta_abs_diff"]
        and summary["mean_false_positive_delta_abs_diff"] <= summary["soft_thresholds"]["mean_false_positive_delta_abs_diff"]
    )
    decision = "RPT1_RAW_METRIC_PARITY_PASS" if strict_pass else ("RPT2_RAW_METRIC_PARITY_SOFT_PASS" if soft_pass else "RPT3_RAW_METRIC_PARITY_FAIL")
    summary["decision"] = decision
    write_json(REPORTS_DIR / "sw14a_raw_parity_tolerance_summary.json", summary)
    return rows, summary, decision


def derive_f3_support(
    sample_index: int,
    horizon_s: int,
    replay_raw: dict[str, Any],
    native_semantic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    fix_variant = candidate_fix_variant()
    sectors = {name: tensor.cpu().bool() for name, tensor in sw14.sw7.build_sector_masks().items()}
    agreement = sw14.agreement_map_for_sample(sample_index, horizon_s, candidate()).float()
    raw_occ = occ_mask(replay_raw["semantic"])
    native_occ = occ_mask(native_semantic)
    raw_delta = raw_occ & ~native_occ
    protected = sw14.sw13c_fix.protected_zone_fix(
        candidate().protected_variant,
        raw_occ,
        raw_delta,
        replay_raw["occ_conf"],
        replay_raw["top1_margin"],
        agreement,
        sectors,
        horizon_s,
    )
    low_value = sw14.sw13c_fix.low_value_score(
        raw_occ,
        protected,
        replay_raw["occ_conf"],
        replay_raw["top1_margin"],
        agreement,
        sectors,
        horizon_s,
        fix_variant.wrong_class_aware,
        fix_variant.front_bias,
    )
    replay_f3, pruned, meta = sw14.sw13c_fix.apply_pruning_no_gt(
        raw_semantic=replay_raw["semantic"].long(),
        protected_mask=protected.bool(),
        low_value_score_map=low_value.float(),
        native_occ_count=occ_count(native_semantic),
        raw_occ_count=occ_count(replay_raw["semantic"]),
        raw_delta_count=int(raw_delta.sum().item()),
        budget_mode=fix_variant.budget_mode,
        expansion_ratio=fix_variant.expansion_ratio,
        keep_ratio=fix_variant.keep_ratio,
    )
    return replay_f3.long(), pruned.bool(), protected.bool(), agreement.float(), meta


def derive_f3_consistency(pred: torch.Tensor, raw: torch.Tensor, native: torch.Tensor, pruned: torch.Tensor, protected: torch.Tensor) -> dict[str, float]:
    raw_delta = occ_mask(raw) & ~occ_mask(native)
    pruned_mask = occ_mask(raw) & ~occ_mask(pred)
    return {
        "protected_pruning_ratio": sw14.safe_div(float((pruned_mask & protected).sum().item()), max(1.0, float(pruned_mask.sum().item()))),
        "final_native_expansion_ratio": sw14.safe_div(float(occ_count(pred) - occ_count(native)), max(1.0, float(occ_count(native)))),
        "raw_delta_keep_ratio": sw14.safe_div(float((occ_mask(pred) & raw_delta).sum().item()), max(1.0, float(raw_delta.sum().item()))),
        "front_pruned_count": float((pruned & front_mask()).sum().item()),
    }


def f3_parity_audit(sample_ids: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    artifact_available = True
    for sample_index in sample_ids:
        replay_by_h = run_raw_replay(model, dataset, sample_index)
        for horizon_s in CORE_HORIZONS:
            replay_raw = replay_by_h[horizon_s]
            native = load_npz_torch(raw_dump_path(sample_index, horizon_s, "native_baseline"))
            teacher = load_npz_torch(final_output_path(sample_index, horizon_s))
            if "final_before_cap" not in teacher:
                artifact_available = False
                continue
            replay_f3, pruned, protected, agreement, budget_meta = derive_f3_support(sample_index, horizon_s, replay_raw, native["semantic"].long())
            gt_h = replay_raw["gt_h"].long()
            gt0 = replay_raw["gt0"].long()
            replay_eval = build_metric_row(replay_f3, gt_h, gt0, native["semantic"].long(), horizon_s)
            teacher_eval = build_metric_row(teacher["final_before_cap"].long(), gt_h, gt0, native["semantic"].long(), horizon_s)
            replay_consistency = derive_f3_consistency(replay_f3, replay_raw["semantic"], native["semantic"].long(), pruned, protected)
            teacher_pruned = occ_mask(teacher["raw_semantic"].long()) & ~occ_mask(teacher["final_before_cap"].long())
            teacher_consistency = derive_f3_consistency(teacher["final_before_cap"].long(), teacher["raw_semantic"].long(), teacher["native_semantic"].long(), teacher_pruned.bool(), teacher["protected_zone"].bool())
            row = {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "pred_gt_density_delta_replay": float(replay_eval["pred_gt_density_delta"]),
                "pred_gt_density_delta_teacher": float(teacher_eval["pred_gt_density_delta"]),
                "pred_gt_density_delta_abs_diff": abs(float(replay_eval["pred_gt_density_delta"]) - float(teacher_eval["pred_gt_density_delta"])),
                "front_false_free_delta_replay": float(replay_eval["front_sector_false_free_rate_delta"]),
                "front_false_free_delta_teacher": float(teacher_eval["front_sector_false_free_rate_delta"]),
                "front_false_free_delta_abs_diff": abs(float(replay_eval["front_sector_false_free_rate_delta"]) - float(teacher_eval["front_sector_false_free_rate_delta"])),
                "false_positive_delta_replay": float(replay_eval["false_positive_delta"]),
                "false_positive_delta_teacher": float(teacher_eval["false_positive_delta"]),
                "false_positive_delta_abs_diff": abs(float(replay_eval["false_positive_delta"]) - float(teacher_eval["false_positive_delta"])),
                "protected_pruning_ratio_replay": replay_consistency["protected_pruning_ratio"],
                "protected_pruning_ratio_teacher": teacher_consistency["protected_pruning_ratio"],
                "protected_pruning_ratio_abs_diff": abs(replay_consistency["protected_pruning_ratio"] - teacher_consistency["protected_pruning_ratio"]),
                "final_native_expansion_ratio_replay": replay_consistency["final_native_expansion_ratio"],
                "final_native_expansion_ratio_teacher": teacher_consistency["final_native_expansion_ratio"],
                "final_native_expansion_ratio_abs_diff": abs(replay_consistency["final_native_expansion_ratio"] - teacher_consistency["final_native_expansion_ratio"]),
                "raw_delta_keep_ratio_replay": replay_consistency["raw_delta_keep_ratio"],
                "raw_delta_keep_ratio_teacher": teacher_consistency["raw_delta_keep_ratio"],
                "raw_delta_keep_ratio_abs_diff": abs(replay_consistency["raw_delta_keep_ratio"] - teacher_consistency["raw_delta_keep_ratio"]),
                "replay_vs_teacher_f3_occ_diff_ratio": occ_diff_ratio(replay_f3, teacher["final_before_cap"].long()),
                "budget_meta": budget_meta,
                "agreement_source_count": candidate().agreement_source_count,
                "uses_gt_budget": False,
                "uses_pred_gt_density_for_selection": False,
            }
            rows.append(row)
    del model
    write_csv(REPORTS_DIR / "sw14a_f3_parity_metrics.csv", rows)
    if not artifact_available or not rows:
        summary = {
            "artifact_available": False,
            "decision": "F3P4_F3_ARTIFACT_MISSING",
        }
        write_json(REPORTS_DIR / "sw14a_f3_parity_summary.json", summary)
        return rows, summary, "F3P4_F3_ARTIFACT_MISSING"
    summary = {
        "artifact_available": True,
        "mean_pred_gt_density_delta_abs_diff": mean_of(rows, "pred_gt_density_delta_abs_diff"),
        "mean_front_false_free_delta_abs_diff": mean_of(rows, "front_false_free_delta_abs_diff"),
        "mean_false_positive_delta_abs_diff": mean_of(rows, "false_positive_delta_abs_diff"),
        "mean_protected_pruning_ratio_abs_diff": mean_of(rows, "protected_pruning_ratio_abs_diff"),
        "mean_final_native_expansion_ratio_abs_diff": mean_of(rows, "final_native_expansion_ratio_abs_diff"),
        "mean_raw_delta_keep_ratio_abs_diff": mean_of(rows, "raw_delta_keep_ratio_abs_diff"),
        "mean_replay_vs_teacher_f3_occ_diff_ratio": mean_of(rows, "replay_vs_teacher_f3_occ_diff_ratio"),
        "top_mismatch_rows": [
            {"sample_index": row["sample_index"], "horizon_s": row["horizon_s"], "replay_vs_teacher_f3_occ_diff_ratio": row["replay_vs_teacher_f3_occ_diff_ratio"]}
            for row in sorted(rows, key=lambda item: float(item["replay_vs_teacher_f3_occ_diff_ratio"]), reverse=True)[:10]
        ],
        "strict_thresholds": {
            "pred_gt_density_delta_abs_diff": 0.01,
            "front_false_free_delta_abs_diff": 0.01,
            "false_positive_delta_abs_diff": 0.005,
            "protected_pruning_ratio_abs_diff": 0.02,
            "final_native_expansion_ratio_abs_diff": 0.02,
            "raw_delta_keep_ratio_abs_diff": 0.02,
        },
        "soft_thresholds": {
            "pred_gt_density_delta_abs_diff": 0.02,
            "front_false_free_delta_abs_diff": 0.02,
            "false_positive_delta_abs_diff": 0.01,
            "protected_pruning_ratio_abs_diff": 0.05,
            "final_native_expansion_ratio_abs_diff": 0.05,
            "raw_delta_keep_ratio_abs_diff": 0.05,
        },
    }
    strict_pass = (
        summary["mean_pred_gt_density_delta_abs_diff"] <= summary["strict_thresholds"]["pred_gt_density_delta_abs_diff"]
        and summary["mean_front_false_free_delta_abs_diff"] <= summary["strict_thresholds"]["front_false_free_delta_abs_diff"]
        and summary["mean_false_positive_delta_abs_diff"] <= summary["strict_thresholds"]["false_positive_delta_abs_diff"]
        and summary["mean_protected_pruning_ratio_abs_diff"] <= summary["strict_thresholds"]["protected_pruning_ratio_abs_diff"]
        and summary["mean_final_native_expansion_ratio_abs_diff"] <= summary["strict_thresholds"]["final_native_expansion_ratio_abs_diff"]
        and summary["mean_raw_delta_keep_ratio_abs_diff"] <= summary["strict_thresholds"]["raw_delta_keep_ratio_abs_diff"]
    )
    soft_pass = (
        summary["mean_pred_gt_density_delta_abs_diff"] <= summary["soft_thresholds"]["pred_gt_density_delta_abs_diff"]
        and summary["mean_front_false_free_delta_abs_diff"] <= summary["soft_thresholds"]["front_false_free_delta_abs_diff"]
        and summary["mean_false_positive_delta_abs_diff"] <= summary["soft_thresholds"]["false_positive_delta_abs_diff"]
        and summary["mean_protected_pruning_ratio_abs_diff"] <= summary["soft_thresholds"]["protected_pruning_ratio_abs_diff"]
        and summary["mean_final_native_expansion_ratio_abs_diff"] <= summary["soft_thresholds"]["final_native_expansion_ratio_abs_diff"]
        and summary["mean_raw_delta_keep_ratio_abs_diff"] <= summary["soft_thresholds"]["raw_delta_keep_ratio_abs_diff"]
    )
    decision = "F3P1_F3_PARITY_PASS" if strict_pass else ("F3P2_F3_PARITY_SOFT_PASS" if soft_pass else "F3P3_F3_PARITY_FAIL")
    summary["decision"] = decision
    write_json(REPORTS_DIR / "sw14a_f3_parity_summary.json", summary)
    return rows, summary, decision


def frontcap_parity_audit(sample_ids: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    artifact_available = True
    for sample_index in sample_ids:
        replay_by_h = run_raw_replay(model, dataset, sample_index)
        for horizon_s in CORE_HORIZONS:
            replay_raw = replay_by_h[horizon_s]
            native = load_npz_torch(raw_dump_path(sample_index, horizon_s, "native_baseline"))
            teacher = load_npz_torch(final_output_path(sample_index, horizon_s))
            if "final_after_cap" not in teacher or "final_before_cap" not in teacher:
                artifact_available = False
                continue
            replay_f3, _, protected, agreement, _ = derive_f3_support(sample_index, horizon_s, replay_raw, native["semantic"].long())
            sectors = {name: tensor.cpu().bool() for name, tensor in sw14.sw7.build_sector_masks().items()}
            low_value_frontcap = sw14.sw13c_fix.low_value_score(
                occ_mask(replay_f3),
                protected,
                replay_raw["occ_conf"],
                replay_raw["top1_margin"],
                agreement,
                sectors,
                horizon_s,
                candidate().name.startswith("A10"),
                0.0,
            )
            replay_final, front_pruned, cap_meta = sw14.frontcap50.apply_front_local_cap_no_gt(
                final_semantic_before_cap=replay_f3.long(),
                native_semantic=native["semantic"].long(),
                raw_semantic=replay_raw["semantic"].long(),
                protected_mask=protected.bool(),
                front_mask=front_mask(),
                confidence=replay_raw["occ_conf"],
                margin=replay_raw["top1_margin"],
                agreement=agreement,
                low_value_score_map=low_value_frontcap.float(),
                cap_ratio=candidate().cap_ratio,
                cap_mode="front_native_ratio_cap",
            )
            gt_h = replay_raw["gt_h"].long()
            gt0 = replay_raw["gt0"].long()
            replay_eval = build_metric_row(replay_final, gt_h, gt0, native["semantic"].long(), horizon_s)
            teacher_eval = build_metric_row(teacher["final_after_cap"].long(), gt_h, gt0, native["semantic"].long(), horizon_s)
            row = {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "final_occ_count_replay": occ_count(replay_final),
                "final_occ_count_teacher": occ_count(teacher["final_after_cap"].long()),
                "replay_vs_teacher_occ_diff_ratio": occ_diff_ratio(replay_final, teacher["final_after_cap"].long()),
                "pred_gt_density_delta_replay": float(replay_eval["pred_gt_density_delta"]),
                "pred_gt_density_delta_teacher": float(teacher_eval["pred_gt_density_delta"]),
                "pred_gt_density_delta_abs_diff": abs(float(replay_eval["pred_gt_density_delta"]) - float(teacher_eval["pred_gt_density_delta"])),
                "front_false_free_delta_replay": float(replay_eval["front_sector_false_free_rate_delta"]),
                "front_false_free_delta_teacher": float(teacher_eval["front_sector_false_free_rate_delta"]),
                "front_false_free_delta_abs_diff": abs(float(replay_eval["front_sector_false_free_rate_delta"]) - float(teacher_eval["front_sector_false_free_rate_delta"])),
                "future_h4_h6_false_free_rate_delta_replay": float(replay_eval["future_h4_h6_false_free_rate_delta"]),
                "future_h4_h6_false_free_rate_delta_teacher": float(teacher_eval["future_h4_h6_false_free_rate_delta"]),
                "future_h4_h6_false_free_rate_delta_abs_diff": abs(float(replay_eval["future_h4_h6_false_free_rate_delta"]) - float(teacher_eval["future_h4_h6_false_free_rate_delta"])),
                "false_positive_delta_replay": float(replay_eval["false_positive_delta"]),
                "false_positive_delta_teacher": float(teacher_eval["false_positive_delta"]),
                "false_positive_delta_abs_diff": abs(float(replay_eval["false_positive_delta"]) - float(teacher_eval["false_positive_delta"])),
                "front_local_density_proxy_replay": float(replay_eval["front_local_density_proxy"]),
                "front_local_density_proxy_teacher": float(teacher_eval["front_local_density_proxy"]),
                "front_local_density_proxy_abs_diff": abs(float(replay_eval["front_local_density_proxy"]) - float(teacher_eval["front_local_density_proxy"])),
                "frontcap_pruned_count": int(front_pruned.sum().item()),
                "frontcap_target_unmet": bool(cap_meta["front_cap_target_unmet"]),
            }
            rows.append(row)
    del model
    write_csv(REPORTS_DIR / "sw14a_frontcap_parity_metrics.csv", rows)
    if not artifact_available or not rows:
        summary = {
            "artifact_available": False,
            "decision": "FCP4_TEACHER_ARTIFACT_MISSING",
        }
        write_json(REPORTS_DIR / "sw14a_frontcap_parity_summary.json", summary)
        return rows, summary, "FCP4_TEACHER_ARTIFACT_MISSING"
    summary = {
        "artifact_available": True,
        "mean_replay_vs_teacher_occ_diff_ratio": mean_of(rows, "replay_vs_teacher_occ_diff_ratio"),
        "mean_pred_gt_density_delta_abs_diff": mean_of(rows, "pred_gt_density_delta_abs_diff"),
        "mean_front_false_free_delta_abs_diff": mean_of(rows, "front_false_free_delta_abs_diff"),
        "mean_false_positive_delta_abs_diff": mean_of(rows, "false_positive_delta_abs_diff"),
        "mean_front_local_density_proxy_abs_diff": mean_of(rows, "front_local_density_proxy_abs_diff"),
        "mean_future_h4_h6_false_free_rate_delta_abs_diff": mean_of(rows, "future_h4_h6_false_free_rate_delta_abs_diff"),
        "top_mismatch_rows": [
            {"sample_index": row["sample_index"], "horizon_s": row["horizon_s"], "replay_vs_teacher_occ_diff_ratio": row["replay_vs_teacher_occ_diff_ratio"]}
            for row in sorted(rows, key=lambda item: float(item["replay_vs_teacher_occ_diff_ratio"]), reverse=True)[:10]
        ],
        "strict_thresholds": {
            "mean_replay_vs_teacher_occ_diff_ratio": 0.0025,
            "mean_pred_gt_density_delta_abs_diff": 0.01,
            "mean_front_false_free_delta_abs_diff": 0.01,
            "mean_false_positive_delta_abs_diff": 0.005,
            "mean_front_local_density_proxy_abs_diff": 0.015,
        },
        "soft_thresholds": {
            "mean_replay_vs_teacher_occ_diff_ratio": 0.005,
            "mean_pred_gt_density_delta_abs_diff": 0.02,
            "mean_front_false_free_delta_abs_diff": 0.02,
            "mean_false_positive_delta_abs_diff": 0.01,
            "mean_front_local_density_proxy_abs_diff": 0.03,
        },
    }
    strict_pass = (
        summary["mean_replay_vs_teacher_occ_diff_ratio"] <= summary["strict_thresholds"]["mean_replay_vs_teacher_occ_diff_ratio"]
        and summary["mean_pred_gt_density_delta_abs_diff"] <= summary["strict_thresholds"]["mean_pred_gt_density_delta_abs_diff"]
        and summary["mean_front_false_free_delta_abs_diff"] <= summary["strict_thresholds"]["mean_front_false_free_delta_abs_diff"]
        and summary["mean_false_positive_delta_abs_diff"] <= summary["strict_thresholds"]["mean_false_positive_delta_abs_diff"]
        and summary["mean_front_local_density_proxy_abs_diff"] <= summary["strict_thresholds"]["mean_front_local_density_proxy_abs_diff"]
    )
    soft_pass = (
        summary["mean_replay_vs_teacher_occ_diff_ratio"] <= summary["soft_thresholds"]["mean_replay_vs_teacher_occ_diff_ratio"]
        and summary["mean_pred_gt_density_delta_abs_diff"] <= summary["soft_thresholds"]["mean_pred_gt_density_delta_abs_diff"]
        and summary["mean_front_false_free_delta_abs_diff"] <= summary["soft_thresholds"]["mean_front_false_free_delta_abs_diff"]
        and summary["mean_false_positive_delta_abs_diff"] <= summary["soft_thresholds"]["mean_false_positive_delta_abs_diff"]
        and summary["mean_front_local_density_proxy_abs_diff"] <= summary["soft_thresholds"]["mean_front_local_density_proxy_abs_diff"]
    )
    decision = "FCP1_FULL_PIPELINE_PARITY_PASS" if strict_pass else ("FCP2_FULL_PIPELINE_PARITY_SOFT_PASS" if soft_pass else "FCP3_FRONTCAP_PARITY_FAIL")
    summary["decision"] = decision
    write_json(REPORTS_DIR / "sw14a_frontcap_parity_summary.json", summary)
    return rows, summary, decision


def final_decision(
    raw_decision: str,
    f3_decision: str | None,
    frontcap_decision: str | None,
) -> str:
    if raw_decision == "RPT3_RAW_METRIC_PARITY_FAIL":
        return "Q1_RAW_PARITY_FAIL_RUNTIME_DEBUG_REQUIRED"
    if f3_decision == "F3P4_F3_ARTIFACT_MISSING" or frontcap_decision == "FCP4_TEACHER_ARTIFACT_MISSING":
        return "Q6_ARTIFACT_MISSING_CANNOT_VERIFY"
    if f3_decision == "F3P3_F3_PARITY_FAIL":
        return "Q2_RAW_SOFT_PASS_F3_FAILS"
    if frontcap_decision == "FCP3_FRONTCAP_PARITY_FAIL":
        return "Q3_RAW_AND_F3_PASS_FRONTCAP_FAILS"
    if raw_decision == "RPT1_RAW_METRIC_PARITY_PASS" and f3_decision == "F3P1_F3_PARITY_PASS" and frontcap_decision == "FCP1_FULL_PIPELINE_PARITY_PASS":
        return "Q5_FULL_PIPELINE_STRICT_PARITY_CONFIRMED"
    return "Q4_FULL_PIPELINE_SOFT_PARITY_CONFIRMED"


def build_report(
    raw_summary: dict[str, Any],
    f3_summary: dict[str, Any] | None,
    frontcap_summary: dict[str, Any] | None,
    final_decision_value: str,
) -> None:
    lines = [
        "# SW14A Raw Parity Tolerance And F3 Parity Report",
        "",
        "## Raw parity tolerance",
        f"- raw decision: `{raw_summary['decision']}`.",
        f"- mean replay_vs_sw13_raw_occ_diff_ratio: `{raw_summary['mean_replay_vs_sw13_raw_occ_diff_ratio']:.8f}`.",
        f"- mean semantic_abs_diff_mean: `{raw_summary['mean_semantic_abs_diff_mean']:.8f}`.",
        f"- mean pred_gt_density_delta_abs_diff: `{raw_summary['mean_pred_gt_density_delta_abs_diff']:.8f}`.",
        f"- mean front_false_free_delta_abs_diff: `{raw_summary['mean_front_false_free_delta_abs_diff']:.8f}`.",
        f"- mean false_positive_delta_abs_diff: `{raw_summary['mean_false_positive_delta_abs_diff']:.8f}`.",
    ]
    if f3_summary is not None:
        lines.extend(
            [
                "",
                "## F3 parity",
                f"- F3 decision: `{f3_summary['decision']}`.",
            ]
        )
        if f3_summary.get("artifact_available", True):
            lines.extend(
                [
                    f"- mean pred_gt_density_delta_abs_diff: `{f3_summary['mean_pred_gt_density_delta_abs_diff']:.8f}`.",
                    f"- mean front_false_free_delta_abs_diff: `{f3_summary['mean_front_false_free_delta_abs_diff']:.8f}`.",
                    f"- mean false_positive_delta_abs_diff: `{f3_summary['mean_false_positive_delta_abs_diff']:.8f}`.",
                    f"- mean final_native_expansion_ratio_abs_diff: `{f3_summary['mean_final_native_expansion_ratio_abs_diff']:.8f}`.",
                    f"- mean raw_delta_keep_ratio_abs_diff: `{f3_summary['mean_raw_delta_keep_ratio_abs_diff']:.8f}`.",
                ]
            )
    if frontcap_summary is not None:
        lines.extend(
            [
                "",
                "## FrontCap parity",
                f"- FrontCap decision: `{frontcap_summary['decision']}`.",
            ]
        )
        if frontcap_summary.get("artifact_available", True):
            lines.extend(
                [
                    f"- mean replay_vs_teacher_occ_diff_ratio: `{frontcap_summary['mean_replay_vs_teacher_occ_diff_ratio']:.8f}`.",
                    f"- mean pred_gt_density_delta_abs_diff: `{frontcap_summary['mean_pred_gt_density_delta_abs_diff']:.8f}`.",
                    f"- mean front_false_free_delta_abs_diff: `{frontcap_summary['mean_front_false_free_delta_abs_diff']:.8f}`.",
                    f"- mean false_positive_delta_abs_diff: `{frontcap_summary['mean_false_positive_delta_abs_diff']:.8f}`.",
                    f"- mean front_local_density_proxy_abs_diff: `{frontcap_summary['mean_front_local_density_proxy_abs_diff']:.8f}`.",
                ]
            )
    lines.extend(
        [
            "",
            "## Final decision",
            f"- final decision: `{final_decision_value}`.",
            "- no training.",
            "- no SW14B.",
            "- no backbone/head/get_occ modification.",
            "- GT remains evaluation-only.",
        ]
    )
    write_md(REPORTS_DIR / "stage_sw14a_raw_tolerance_f3_frontcap_report.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_dirs()
    sample_ids = list(range(args.sample_start, args.sample_end + 1))
    ok, missing = check_required_artifacts(sample_ids)
    if not ok:
        decision = {
            "decision_type": "Q6_ARTIFACT_MISSING_CANNOT_VERIFY",
            "missing_artifact_count": len(missing),
            "missing_artifacts": missing[:200],
            "no_training": True,
            "no_sw14b": True,
            "no_backbone_change": True,
            "no_head_change": True,
            "no_get_occ_change": True,
        }
        write_json(REPORTS_DIR / "sw14a_raw_tolerance_f3_frontcap_decision.json", decision)
        build_report({"decision": "RPT3_RAW_METRIC_PARITY_FAIL", "mean_replay_vs_sw13_raw_occ_diff_ratio": 0.0, "mean_semantic_abs_diff_mean": 0.0, "mean_pred_gt_density_delta_abs_diff": 0.0, "mean_front_false_free_delta_abs_diff": 0.0, "mean_false_positive_delta_abs_diff": 0.0}, None, None, decision["decision_type"])
        return

    raw_rows, raw_summary, raw_decision = raw_parity_tolerance_audit(sample_ids)
    f3_rows: list[dict[str, Any]] | None = None
    f3_summary: dict[str, Any] | None = None
    f3_decision: str | None = None
    frontcap_rows: list[dict[str, Any]] | None = None
    frontcap_summary: dict[str, Any] | None = None
    frontcap_decision: str | None = None

    if raw_decision in {"RPT1_RAW_METRIC_PARITY_PASS", "RPT2_RAW_METRIC_PARITY_SOFT_PASS"}:
        f3_rows, f3_summary, f3_decision = f3_parity_audit(sample_ids)
        if f3_decision in {"F3P1_F3_PARITY_PASS", "F3P2_F3_PARITY_SOFT_PASS"}:
            frontcap_rows, frontcap_summary, frontcap_decision = frontcap_parity_audit(sample_ids)

    final_decision_value = final_decision(raw_decision, f3_decision, frontcap_decision)
    decision = {
        "decision_type": final_decision_value,
        "raw_parity_decision": raw_decision,
        "raw_parity_summary": raw_summary,
        "f3_parity_decision": f3_decision,
        "f3_parity_summary": f3_summary,
        "frontcap_parity_decision": frontcap_decision,
        "frontcap_parity_summary": frontcap_summary,
        "key_mismatch_samples": {
            "raw": raw_summary["top_mismatch_samples"][:5],
            "f3": [] if f3_summary is None else f3_summary.get("top_mismatch_rows", [])[:5],
            "frontcap": [] if frontcap_summary is None else frontcap_summary.get("top_mismatch_rows", [])[:5],
        },
        "no_training": True,
        "no_sw14b": True,
        "no_backbone_change": True,
        "no_head_change": True,
        "no_get_occ_change": True,
        "gt_evaluation_only": True,
    }
    write_json(REPORTS_DIR / "sw14a_raw_tolerance_f3_frontcap_decision.json", decision)
    build_report(raw_summary, f3_summary, frontcap_summary, final_decision_value)


if __name__ == "__main__":
    main()
