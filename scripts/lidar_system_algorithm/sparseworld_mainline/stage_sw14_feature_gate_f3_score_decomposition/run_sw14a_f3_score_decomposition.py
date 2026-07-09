from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_f3_score_decomposition"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_f3_score_decomposition"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14_feature_gate_f3_score_decomposition"

F3_AUDIT_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_f3_pruning_audit/run_sw14a_f3_pruning_audit.py"

AUDIT_CASES = [
    (119, 6),
    (112, 2),
    (118, 2),
    (118, 6),
    (119, 2),
    (100, 6),
]
EMPTY_IDX = 17


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fa = load_module("sw14_f3_score_audit_base", F3_AUDIT_SCRIPT)


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, SCRIPTS_DIR, ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(obj: Any) -> Any:
    return fa.normalize(obj)


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


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(r[key]) for r in rows) / len(rows)) if rows else 0.0


def maxv(rows: list[dict[str, Any]], key: str) -> float:
    return float(max(float(r[key]) for r in rows)) if rows else 0.0


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic != EMPTY_IDX


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool()
    b = b.bool()
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    return inter / union if union > 0 else 1.0


def top_percent_mask(values: torch.Tensor, base_mask: torch.Tensor, pct: float = 0.01) -> torch.Tensor:
    if not base_mask.any():
        return torch.zeros_like(base_mask, dtype=torch.bool)
    flat = values[base_mask].float()
    k = max(1, int(round(flat.numel() * pct)))
    _, idx = torch.topk(flat, k=k, largest=True)
    coords = torch.nonzero(base_mask, as_tuple=False)
    out = torch.zeros_like(base_mask, dtype=torch.bool)
    pick = coords[idx]
    out[pick[:, 0], pick[:, 1], pick[:, 2]] = True
    return out


def component_wrapper(
    raw_occ: torch.Tensor,
    protected: torch.Tensor,
    occ_conf: torch.Tensor,
    margin: torch.Tensor,
    agreement: torch.Tensor,
    sectors: dict[str, torch.Tensor],
    horizon_s: int,
    wrong_class_aware: bool,
    front_bias: float,
) -> dict[str, torch.Tensor]:
    front = sectors["front"].bool()
    entropy = fa.rt.sw14.sw13c_fix.confidence_entropy(torch.clamp(occ_conf, 0, 1))
    conf_bad = 1.0 - torch.clamp(occ_conf, 0, 1)
    margin_norm = torch.zeros_like(margin)
    if raw_occ.any():
        margin_norm = margin.float() / (margin[raw_occ].float().max() + 1e-6)
    margin_bad = 1.0 - torch.clamp(margin_norm, 0, 1)
    isolated = (fa.rt.sw14.sw13c_fix.local_support(raw_occ) < 3).float()
    nonfront = (~front).float()
    future_front_discount = 0.35 if horizon_s in {4, 6} else 0.10

    conf_bad_component = 1.2 * conf_bad
    margin_bad_component = 0.9 * margin_bad
    entropy_component = 0.8 * entropy
    agreement_component = 1.4 * (1.0 - agreement)
    nonfront_component = front_bias * nonfront
    isolated_component = 0.8 * isolated
    wrong_class_component = 0.8 * (margin_bad + entropy) if wrong_class_aware else torch.zeros_like(margin_bad)
    protected_penalty_component = -100.0 * protected.float()
    front_discount_component = -future_front_discount * front.float()
    invalid_or_non_prunable_mask = ~raw_occ
    invalid_sentinel_component = torch.zeros_like(conf_bad_component)
    invalid_sentinel_component[invalid_or_non_prunable_mask] = -1e6
    raw_delta_candidate_mask = raw_occ.bool()
    final_score = (
        conf_bad_component
        + margin_bad_component
        + entropy_component
        + agreement_component
        + nonfront_component
        + isolated_component
        + wrong_class_component
        + protected_penalty_component
        + front_discount_component
        + invalid_sentinel_component
    )
    return {
        "conf_bad_component": conf_bad_component.float(),
        "margin_bad_component": margin_bad_component.float(),
        "entropy_component": entropy_component.float(),
        "agreement_component": agreement_component.float(),
        "nonfront_component": nonfront_component.float(),
        "isolated_component": isolated_component.float(),
        "protected_penalty_component": protected_penalty_component.float(),
        "front_discount_component": front_discount_component.float(),
        "wrong_class_component": wrong_class_component.float(),
        "raw_delta_candidate_mask": raw_delta_candidate_mask.bool(),
        "invalid_or_non_prunable_mask": invalid_or_non_prunable_mask.bool(),
        "invalid_sentinel_component": invalid_sentinel_component.float(),
        "front_mask": front.bool(),
        "final_low_value_score": final_score.float(),
    }


def save_component_npz(path: Path, per_case: dict[str, dict[str, torch.Tensor]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for case_key, comps in per_case.items():
        for name, tensor in comps.items():
            arrays[f"{case_key}__{name}"] = tensor.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def collect_context():
    runtime_replays = fa.get_runtime_replays()
    _, full_context = fa.input_consistency_audit(runtime_replays)
    ctx = {k: v for k, v in full_context.items() if k in AUDIT_CASES}
    return ctx


def phase1_components(context: dict[tuple[int, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    replay_npz: dict[str, dict[str, torch.Tensor]] = {}
    teacher_npz: dict[str, dict[str, torch.Tensor]] = {}
    sectors = {name: tensor.cpu().bool() for name, tensor in fa.rt.sw14.sw7.build_sector_masks().items()}
    fix_variant = fa.rt.candidate_fix_variant()
    for sample_index, horizon_s in AUDIT_CASES:
        c = context[(sample_index, horizon_s)]
        replay = component_wrapper(
            c["replay_ctx"]["raw_occ"],
            c["replay_ctx"]["protected"],
            c["replay_raw"]["occ_conf"].float(),
            c["replay_raw"]["top1_margin"].float(),
            c["replay_agreement"].float(),
            sectors,
            horizon_s,
            fix_variant.wrong_class_aware,
            fix_variant.front_bias,
        )
        teacher = component_wrapper(
            c["teacher_input_ctx"]["raw_occ"],
            c["teacher_input_ctx"]["protected"],
            c["stored_raw"]["occ_conf"].float(),
            c["stored_raw"]["top1_margin"].float(),
            c["stored_agreement"].float(),
            sectors,
            horizon_s,
            fix_variant.wrong_class_aware,
            fix_variant.front_bias,
        )
        case_key = f"sample{sample_index:03d}_h{horizon_s}"
        replay_npz[case_key] = replay
        teacher_npz[case_key] = teacher
        orig_replay = c["replay_ctx"]["low_value"].float()
        orig_teacher = c["teacher_input_ctx"]["low_value"].float()
        rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "replay_wrapper_allclose_original": bool(torch.allclose(replay["final_low_value_score"], orig_replay, atol=1e-4)),
                "teacher_wrapper_allclose_original": bool(torch.allclose(teacher["final_low_value_score"], orig_teacher, atol=1e-4)),
                "replay_wrapper_mean_abs_diff": float((replay["final_low_value_score"] - orig_replay).abs().mean().item()),
                "teacher_wrapper_mean_abs_diff": float((teacher["final_low_value_score"] - orig_teacher).abs().mean().item()),
            }
        )
    save_component_npz(ARTIFACTS_DIR / "sw14a_f3_score_components_replay.npz", replay_npz)
    save_component_npz(ARTIFACTS_DIR / "sw14a_f3_score_components_teacher_or_stored.npz", teacher_npz)
    write_csv(REPORTS_DIR / "sw14a_f3_score_component_summary.csv", rows)
    return rows, replay_npz, teacher_npz


def phase2_component_diff(context: dict[tuple[int, int], dict[str, Any]], replay_npz: dict[str, dict[str, torch.Tensor]], teacher_npz: dict[str, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    component_names = [
        "conf_bad_component",
        "margin_bad_component",
        "entropy_component",
        "agreement_component",
        "nonfront_component",
        "isolated_component",
        "protected_penalty_component",
        "front_discount_component",
        "invalid_sentinel_component",
    ]
    for sample_index, horizon_s in AUDIT_CASES:
        case_key = f"sample{sample_index:03d}_h{horizon_s}"
        c = context[(sample_index, horizon_s)]
        teacher = fa.rt.load_npz_torch(fa.rt.final_output_path(sample_index, horizon_s))
        teacher_pruned = occ_mask(teacher["raw_semantic"].long()) & ~occ_mask(teacher["final_before_cap"].long())
        boundary = teacher_pruned | c["replay_ctx"]["pruned"]
        raw_delta_replay = c["replay_ctx"]["raw_delta"]
        front = fa.rt.front_mask()
        for name in component_names:
            a = replay_npz[case_key][name].float()
            b = teacher_npz[case_key][name].float()
            top1 = top_percent_mask(a.abs() + b.abs(), raw_delta_replay.bool(), pct=0.01)
            flip_mask = (a.sign() != b.sign()) & ((a.abs() > 1e-8) | (b.abs() > 1e-8))
            rows.append(
                {
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "component_name": name,
                    "component_mean_abs_diff": float((a - b).abs().mean().item()),
                    "component_max_abs_diff": float((a - b).abs().max().item()),
                    "component_top1_percent_mean_abs_diff": float((a[top1] - b[top1]).abs().mean().item()) if top1.any() else 0.0,
                    "component_flip_count": int(flip_mask.sum().item()),
                    "component_front_region_diff": float((a[front] - b[front]).abs().mean().item()),
                    "component_raw_delta_region_diff": float((a[raw_delta_replay] - b[raw_delta_replay]).abs().mean().item()) if raw_delta_replay.any() else 0.0,
                    "component_pruned_boundary_region_diff": float((a[boundary] - b[boundary]).abs().mean().item()) if boundary.any() else 0.0,
                    "protected_flip_count": int((replay_npz[case_key]["protected_penalty_component"] != teacher_npz[case_key]["protected_penalty_component"]).sum().item()),
                    "raw_delta_flip_count": int((replay_npz[case_key]["raw_delta_candidate_mask"] != teacher_npz[case_key]["raw_delta_candidate_mask"]).sum().item()),
                    "isolated_flip_count": int((replay_npz[case_key]["isolated_component"] != teacher_npz[case_key]["isolated_component"]).sum().item()),
                    "invalid_sentinel_flip_count": int((replay_npz[case_key]["invalid_sentinel_component"] != teacher_npz[case_key]["invalid_sentinel_component"]).sum().item()),
                    "front_mask_flip_count": int((replay_npz[case_key]["front_mask"] != teacher_npz[case_key]["front_mask"]).sum().item()),
                    "nonfront_component_flip_count": int((replay_npz[case_key]["nonfront_component"] != teacher_npz[case_key]["nonfront_component"]).sum().item()),
                    "agreement_quarter_step_flip_count": int((((c["replay_agreement"] * 4).round() != (c["stored_agreement"] * 4).round())).sum().item()),
                }
            )
    write_csv(REPORTS_DIR / "sw14a_f3_component_diff_audit.csv", rows)
    return rows


def phase3_sentinel_amplification(context: dict[tuple[int, int], dict[str, Any]], replay_npz: dict[str, dict[str, torch.Tensor]], teacher_npz: dict[str, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    front = fa.rt.front_mask()
    for sample_index, horizon_s in AUDIT_CASES:
        case_key = f"sample{sample_index:03d}_h{horizon_s}"
        replay_score = replay_npz[case_key]["final_low_value_score"]
        teacher_score = teacher_npz[case_key]["final_low_value_score"]
        diff = (replay_score - teacher_score).abs()
        large = diff > 1e3
        raw_delta = replay_npz[case_key]["raw_delta_candidate_mask"]
        protected_flip = replay_npz[case_key]["protected_penalty_component"] != teacher_npz[case_key]["protected_penalty_component"]
        raw_delta_flip = replay_npz[case_key]["raw_delta_candidate_mask"] != teacher_npz[case_key]["raw_delta_candidate_mask"]
        isolated_flip = replay_npz[case_key]["isolated_component"] != teacher_npz[case_key]["isolated_component"]
        invalid_flip = replay_npz[case_key]["invalid_or_non_prunable_mask"] != teacher_npz[case_key]["invalid_or_non_prunable_mask"]
        agreement_flip = ((context[(sample_index, horizon_s)]["replay_agreement"] * 4).round() != (context[(sample_index, horizon_s)]["stored_agreement"] * 4).round())
        other = large & ~(protected_flip | raw_delta_flip | isolated_flip | invalid_flip | agreement_flip)
        rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "large_score_diff_count": int(large.sum().item()),
                "large_score_diff_ratio": float(large.float().mean().item()),
                "large_score_diff_in_front_count": int((large & front).sum().item()),
                "large_score_diff_in_raw_delta_count": int((large & raw_delta).sum().item()),
                "cause_breakdown_protected_flip": int((large & protected_flip).sum().item()),
                "cause_breakdown_raw_delta_flip": int((large & raw_delta_flip).sum().item()),
                "cause_breakdown_isolated_flip": int((large & isolated_flip).sum().item()),
                "cause_breakdown_invalid_mask_flip": int((large & invalid_flip).sum().item()),
                "cause_breakdown_agreement_flip": int((large & agreement_flip).sum().item()),
                "cause_breakdown_other": int((large & other).sum().item()),
            }
        )
    write_csv(REPORTS_DIR / "sw14a_f3_sentinel_amplification_audit.csv", rows)
    return rows


def phase4_boundary_attribution(context: dict[tuple[int, int], dict[str, Any]], replay_npz: dict[str, dict[str, torch.Tensor]], teacher_npz: dict[str, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    components = [
        "conf_bad_component",
        "margin_bad_component",
        "entropy_component",
        "agreement_component",
        "nonfront_component",
        "isolated_component",
        "protected_penalty_component",
        "invalid_sentinel_component",
    ]
    for sample_index, horizon_s in AUDIT_CASES:
        case_key = f"sample{sample_index:03d}_h{horizon_s}"
        teacher = fa.rt.load_npz_torch(fa.rt.final_output_path(sample_index, horizon_s))
        raw_occ_teacher = occ_mask(teacher["raw_semantic"].long())
        pruned_teacher = raw_occ_teacher & ~occ_mask(teacher["final_before_cap"].long())
        pruned_replay = context[(sample_index, horizon_s)]["replay_ctx"]["pruned"]
        replay_score = replay_npz[case_key]["final_low_value_score"]
        candidate = replay_npz[case_key]["raw_delta_candidate_mask"] | teacher_npz[case_key]["raw_delta_candidate_mask"]
        boundary = top_percent_mask(replay_score.abs(), candidate.bool(), pct=0.01)
        mism1 = pruned_replay & ~pruned_teacher
        mism2 = pruned_teacher & ~pruned_replay
        mism = mism1 | mism2
        comp_diffs = {name: float((replay_npz[case_key][name][boundary] - teacher_npz[case_key][name][boundary]).abs().mean().item()) if boundary.any() else 0.0 for name in components}
        dominant = max(comp_diffs, key=comp_diffs.get)
        rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "boundary_voxel_count": int(boundary.sum().item()),
                "boundary_overlap_with_teacher_pruned": int((boundary & pruned_teacher).sum().item()),
                "boundary_component_dominant_diff": dominant,
                "replay_pruned_teacher_kept_count": int(mism1.sum().item()),
                "teacher_pruned_replay_kept_count": int(mism2.sum().item()),
                "dominant_reason_for_mismatch": dominant if mism.any() else "none",
            }
        )
    write_csv(REPORTS_DIR / "sw14a_f3_pruning_boundary_attribution.csv", rows)
    return rows


def apply_counterfactual(case_key: str, context_case: dict[str, Any], replay_npz_case: dict[str, torch.Tensor], teacher_npz_case: dict[str, torch.Tensor], replace_mode: str) -> dict[str, Any]:
    def choose(name: str) -> torch.Tensor:
        sentinel_modes = {"replace_raw_delta_candidate_mask", "replace_protected_penalty_component", "replace_isolated_component", "replace_agreement_component", "replace_conf_margin_components", "replace_all_sentinel_mask_components", "replace_continuous_score_components"}
        if replace_mode not in sentinel_modes:
            return replay_npz_case[name]
        if replace_mode == "replace_raw_delta_candidate_mask" and name == "raw_delta_candidate_mask":
            return teacher_npz_case[name]
        if replace_mode == "replace_protected_penalty_component" and name == "protected_penalty_component":
            return teacher_npz_case[name]
        if replace_mode == "replace_isolated_component" and name == "isolated_component":
            return teacher_npz_case[name]
        if replace_mode == "replace_agreement_component" and name == "agreement_component":
            return teacher_npz_case[name]
        if replace_mode == "replace_conf_margin_components" and name in {"conf_bad_component", "margin_bad_component", "entropy_component"}:
            return teacher_npz_case[name]
        if replace_mode == "replace_all_sentinel_mask_components" and name in {"protected_penalty_component", "invalid_sentinel_component", "raw_delta_candidate_mask", "front_mask", "isolated_component"}:
            return teacher_npz_case[name]
        if replace_mode == "replace_continuous_score_components" and name in {"conf_bad_component", "margin_bad_component", "entropy_component", "agreement_component", "nonfront_component"}:
            return teacher_npz_case[name]
        return replay_npz_case[name]

    raw_semantic = context_case["replay_raw"]["semantic"].long()
    teacher = fa.rt.load_npz_torch(fa.rt.final_output_path(int(case_key[6:9]), int(case_key.split("_h")[1])))
    native = context_case["native"]["semantic"].long()
    raw_occ = occ_mask(raw_semantic)
    protected_mask = choose("protected_penalty_component") < 0
    score = (
        choose("conf_bad_component")
        + choose("margin_bad_component")
        + choose("entropy_component")
        + choose("agreement_component")
        + choose("nonfront_component")
        + choose("isolated_component")
        + replay_npz_case["front_discount_component"]
        + choose("protected_penalty_component")
        + choose("invalid_sentinel_component")
    ).float()
    final, pruned, _ = fa.rt.sw14.sw13c_fix.apply_pruning_no_gt(
        raw_semantic=raw_semantic,
        protected_mask=protected_mask.bool(),
        low_value_score_map=score,
        native_occ_count=fa.rt.occ_count(native),
        raw_occ_count=fa.rt.occ_count(raw_semantic),
        raw_delta_count=int((raw_occ & ~occ_mask(native)).sum().item()),
        budget_mode=fa.rt.candidate_fix_variant().budget_mode,
        expansion_ratio=fa.rt.candidate_fix_variant().expansion_ratio,
        keep_ratio=fa.rt.candidate_fix_variant().keep_ratio,
    )
    gt_h = context_case["replay_raw"]["gt_h"].long()
    gt0 = context_case["replay_raw"]["gt0"].long()
    pred_eval = fa.rt.build_metric_row(final.long(), gt_h, gt0, native, int(case_key.split("_h")[1]))
    teacher_eval = fa.rt.build_metric_row(teacher["final_before_cap"].long(), gt_h, gt0, native, int(case_key.split("_h")[1]))
    teacher_pruned = occ_mask(teacher["raw_semantic"].long()) & ~occ_mask(teacher["final_before_cap"].long())
    kept_delta_pred = occ_mask(final.long()) & ~occ_mask(native)
    kept_delta_teacher = occ_mask(teacher["final_before_cap"].long()) & ~occ_mask(native)
    return {
        "pruned_jaccard": jaccard(pruned.bool(), teacher_pruned.bool()),
        "kept_delta_jaccard": jaccard(kept_delta_pred.bool(), kept_delta_teacher.bool()),
        "final_occ_jaccard": jaccard(occ_mask(final.long()), occ_mask(teacher["final_before_cap"].long())),
        "front_false_free_delta_abs_diff": abs(float(pred_eval["front_sector_false_free_rate_delta"]) - float(teacher_eval["front_sector_false_free_rate_delta"])),
    }


def phase5_component_replacement(context: dict[tuple[int, int], dict[str, Any]], replay_npz: dict[str, dict[str, torch.Tensor]], teacher_npz: dict[str, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    modes = [
        "replace_raw_delta_candidate_mask",
        "replace_protected_penalty_component",
        "replace_isolated_component",
        "replace_agreement_component",
        "replace_conf_margin_components",
        "replace_all_sentinel_mask_components",
        "replace_continuous_score_components",
    ]
    rows: list[dict[str, Any]] = []
    for sample_index, horizon_s in AUDIT_CASES:
        case_key = f"sample{sample_index:03d}_h{horizon_s}"
        for mode in modes:
            metrics = apply_counterfactual(case_key, context[(sample_index, horizon_s)], replay_npz[case_key], teacher_npz[case_key], mode)
            rows.append({"sample_index": sample_index, "horizon_s": horizon_s, "replacement_mode": mode, **metrics})
    write_csv(REPORTS_DIR / "sw14a_f3_component_replacement_ablation.csv", rows)
    return rows


def decide(component_rows: list[dict[str, Any]], sentinel_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], replacement_rows: list[dict[str, Any]]) -> str:
    def rmean(mode: str, key: str) -> float:
        vals = [float(r[key]) for r in replacement_rows if r["replacement_mode"] == mode]
        return sum(vals) / len(vals) if vals else 0.0
    baseline = rmean("replace_conf_margin_components", "front_false_free_delta_abs_diff")
    mode_scores = {
        "F3D1_RAW_DELTA_MASK_AMPLIFICATION": rmean("replace_raw_delta_candidate_mask", "front_false_free_delta_abs_diff"),
        "F3D2_PROTECTED_SENTINEL_AMPLIFICATION": rmean("replace_all_sentinel_mask_components", "front_false_free_delta_abs_diff"),
        "F3D3_ISOLATED_COMPONENT_AMPLIFICATION": rmean("replace_isolated_component", "front_false_free_delta_abs_diff"),
        "F3D4_AGREEMENT_OR_SECTOR_MASK_MISMATCH": rmean("replace_agreement_component", "front_false_free_delta_abs_diff"),
        "F3D5_CONTINUOUS_SCORE_MISMATCH": rmean("replace_conf_margin_components", "front_false_free_delta_abs_diff"),
    }
    best = min(mode_scores, key=mode_scores.get)
    if mode_scores[best] <= 0.6 * baseline:
        return best
    protected_large = sum(int(r["cause_breakdown_protected_flip"]) + int(r["cause_breakdown_invalid_mask_flip"]) for r in sentinel_rows)
    raw_delta_large = sum(int(r["cause_breakdown_raw_delta_flip"]) for r in sentinel_rows)
    isolated_large = sum(int(r["cause_breakdown_isolated_flip"]) for r in sentinel_rows)
    agreement_large = sum(int(r["cause_breakdown_agreement_flip"]) for r in sentinel_rows)
    if raw_delta_large > max(protected_large, isolated_large, agreement_large) * 1.2:
        return "F3D1_RAW_DELTA_MASK_AMPLIFICATION"
    if protected_large > max(raw_delta_large, isolated_large, agreement_large) * 1.2:
        return "F3D2_PROTECTED_SENTINEL_AMPLIFICATION"
    if isolated_large > max(raw_delta_large, protected_large, agreement_large) * 1.2:
        return "F3D3_ISOLATED_COMPONENT_AMPLIFICATION"
    if agreement_large > max(raw_delta_large, protected_large, isolated_large) * 1.2:
        return "F3D4_AGREEMENT_OR_SECTOR_MASK_MISMATCH"
    mask_related = protected_large + raw_delta_large + isolated_large + agreement_large
    continuous_diff = mean(component_rows, "component_mean_abs_diff")
    if mask_related > 0 and rmean("replace_all_sentinel_mask_components", "front_false_free_delta_abs_diff") < rmean("replace_continuous_score_components", "front_false_free_delta_abs_diff"):
        return "F3D6_MULTIPLE_MASK_COMPONENTS_AMPLIFY_SMALL_RAW_DRIFT"
    if continuous_diff > 1.0:
        return "F3D5_CONTINUOUS_SCORE_MISMATCH"
    return "F3D7_NO_SINGLE_DOMINANT_CAUSE"


def build_report(component_rows: list[dict[str, Any]], sentinel_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], replacement_rows: list[dict[str, Any]], decision: str) -> None:
    def rmean(key: str) -> float:
        return mean(component_rows, key)
    def mmean(mode: str, key: str) -> float:
        vals = [float(r[key]) for r in replacement_rows if r["replacement_mode"] == mode]
        return sum(vals) / len(vals) if vals else 0.0
    lines = [
        "# SW14A F3 Score Decomposition Report",
        "",
        "## Audited cases",
        f"- case count: `{len(AUDIT_CASES)}`.",
        "",
        "## Component diff summary",
        f"- mean component_mean_abs_diff: `{rmean('component_mean_abs_diff'):.8f}`.",
        f"- max component_max_abs_diff: `{maxv(component_rows, 'component_max_abs_diff'):.8f}`.",
        "",
        "## Sentinel amplification summary",
        f"- mean large_score_diff_ratio: `{mean(sentinel_rows, 'large_score_diff_ratio'):.8f}`.",
        f"- mean protected_flip large count: `{mean(sentinel_rows, 'cause_breakdown_protected_flip'):.4f}`.",
        f"- mean raw_delta_flip large count: `{mean(sentinel_rows, 'cause_breakdown_raw_delta_flip'):.4f}`.",
        "",
        "## Pruning boundary attribution",
        f"- mean boundary_voxel_count: `{mean(boundary_rows, 'boundary_voxel_count'):.4f}`.",
        "",
        "## Component replacement ablation",
        f"- replace_raw_delta_candidate_mask mean front_ff diff: `{mmean('replace_raw_delta_candidate_mask', 'front_false_free_delta_abs_diff'):.8f}`.",
        f"- replace_all_sentinel_mask_components mean front_ff diff: `{mmean('replace_all_sentinel_mask_components', 'front_false_free_delta_abs_diff'):.8f}`.",
        f"- replace_continuous_score_components mean front_ff diff: `{mmean('replace_continuous_score_components', 'front_false_free_delta_abs_diff'):.8f}`.",
        "",
        "## Decision",
        f"- decision: `{decision}`.",
    ]
    write_md(REPORTS_DIR / "stage_sw14a_f3_score_decomposition_report.md", "\n".join(lines))


def main() -> None:
    ensure_dirs()
    context = collect_context()
    phase1_rows, replay_npz, teacher_npz = phase1_components(context)
    component_rows = phase2_component_diff(context, replay_npz, teacher_npz)
    sentinel_rows = phase3_sentinel_amplification(context, replay_npz, teacher_npz)
    boundary_rows = phase4_boundary_attribution(context, replay_npz, teacher_npz)
    replacement_rows = phase5_component_replacement(context, replay_npz, teacher_npz)
    decision = decide(component_rows, sentinel_rows, boundary_rows, replacement_rows)
    payload = {
        "decision_type": decision,
        "audited_cases": [{"sample_index": s, "horizon_s": h} for s, h in AUDIT_CASES],
        "component_diff_summary": {
            "mean_component_mean_abs_diff": mean(component_rows, "component_mean_abs_diff"),
            "max_component_max_abs_diff": maxv(component_rows, "component_max_abs_diff"),
        },
        "sentinel_amplification_summary": {
            "mean_large_score_diff_ratio": mean(sentinel_rows, "large_score_diff_ratio"),
            "mean_protected_flip_large_count": mean(sentinel_rows, "cause_breakdown_protected_flip"),
            "mean_raw_delta_flip_large_count": mean(sentinel_rows, "cause_breakdown_raw_delta_flip"),
            "mean_isolated_flip_large_count": mean(sentinel_rows, "cause_breakdown_isolated_flip"),
        },
        "pruning_boundary_attribution_summary": {
            "mean_boundary_voxel_count": mean(boundary_rows, "boundary_voxel_count"),
        },
        "component_replacement_ablation_summary": {
            mode: {
                "mean_pruned_jaccard": mean([r for r in replacement_rows if r["replacement_mode"] == mode], "pruned_jaccard"),
                "mean_front_false_free_delta_abs_diff": mean([r for r in replacement_rows if r["replacement_mode"] == mode], "front_false_free_delta_abs_diff"),
            }
            for mode in sorted({r["replacement_mode"] for r in replacement_rows})
        },
        "no_training": True,
        "no_sw14b": True,
        "no_backbone_change": True,
        "no_head_change": True,
        "no_get_occ_change": True,
        "gt_evaluation_only": True,
    }
    write_json(REPORTS_DIR / "sw14a_f3_score_decomposition_decision.json", payload)
    build_report(component_rows, sentinel_rows, boundary_rows, replacement_rows, decision)


if __name__ == "__main__":
    main()
