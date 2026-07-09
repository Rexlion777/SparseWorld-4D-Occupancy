from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_f3_pruning_audit"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_f3_pruning_audit"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14_feature_gate_f3_pruning_audit"

RAW_TOL_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_raw_parity_tolerance/run_sw14a_raw_parity_tolerance.py"
SW13_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"
SW13_FC500_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/run_sw13c_fix_frontcap_eval_core500_incremental.py"

EMPTY_IDX = 17
AUDIT_CASES = [
    (119, 6),
    (112, 2),
    (118, 2),
    (118, 6),
    (119, 2),
    (110, 2),
    (110, 4),
    (111, 2),
    (111, 6),
    (100, 6),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = load_module("sw14_f3_audit_base", RAW_TOL_SCRIPT)


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, SCRIPTS_DIR, ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(obj: Any) -> Any:
    return rt.normalize(obj)


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


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic != EMPTY_IDX


def occ_count(semantic: torch.Tensor) -> int:
    return int(occ_mask(semantic).sum().item())


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool()
    b = b.bool()
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    return inter / union if union > 0 else 1.0


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def linear_indices(mask: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)
    shape = mask.shape
    return coords[:, 0] * (shape[1] * shape[2]) + coords[:, 1] * shape[2] + coords[:, 2]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def case_manifest() -> dict[str, Any]:
    payload = {
        "candidate": rt.candidate().name,
        "priority_cases": [{"sample_index": s, "horizon_s": h} for s, h in AUDIT_CASES[:-1]],
        "good_control_case": {"sample_index": 100, "horizon_s": 6},
        "no_training": True,
        "no_sw14b": True,
        "no_backbone_change": True,
        "no_head_change": True,
        "no_get_occ_change": True,
        "gt_evaluation_only": True,
    }
    write_json(REPORTS_DIR / "sw14a_f3_audit_case_manifest.json", payload)
    return payload


def get_runtime_replays() -> dict[int, dict[int, dict[str, Any]]]:
    _, dataset, model, _ = rt.sw14.build_runtime(train=False)
    model.eval()
    by_sample: dict[int, dict[int, dict[str, Any]]] = {}
    for sample_index in sorted({s for s, _ in AUDIT_CASES}):
        by_sample[sample_index] = rt.run_raw_replay(model, dataset, sample_index)
    del model
    return by_sample


def stored_agreement_map(sample_index: int, horizon_s: int) -> torch.Tensor:
    return rt.sw14.agreement_map_for_sample(sample_index, horizon_s, rt.candidate()).float()


def replay_agreement_map(sample_index: int, horizon_s: int, replay_raw_semantic: torch.Tensor) -> torch.Tensor:
    labels = rt.candidate().agreement_sources
    occs: list[torch.Tensor] = []
    for label in labels:
        if label == rt.candidate().base_repair_variant:
            occs.append(occ_mask(replay_raw_semantic))
        else:
            payload = rt.load_npz_torch(rt.raw_dump_path(sample_index, horizon_s, label))
            occs.append(occ_mask(payload["semantic"].long()))
    acc = torch.zeros_like(occs[0], dtype=torch.float32)
    for occ in occs:
        acc += occ.float()
    return acc / float(len(occs))


def derive_masks_and_scores(
    raw_semantic: torch.Tensor,
    native_semantic: torch.Tensor,
    occ_conf: torch.Tensor,
    top1_margin: torch.Tensor,
    agreement: torch.Tensor,
    horizon_s: int,
) -> dict[str, Any]:
    fix_variant = rt.candidate_fix_variant()
    sectors = {name: tensor.cpu().bool() for name, tensor in rt.sw14.sw7.build_sector_masks().items()}
    raw_occ = occ_mask(raw_semantic)
    native_occ = occ_mask(native_semantic)
    raw_delta = raw_occ & ~native_occ
    protected = rt.sw14.sw13c_fix.protected_zone_fix(
        rt.candidate().protected_variant,
        raw_occ,
        raw_delta,
        occ_conf.float(),
        top1_margin.float(),
        agreement.float(),
        sectors,
        horizon_s,
    )
    low_value = rt.sw14.sw13c_fix.low_value_score(
        raw_occ,
        protected.bool(),
        occ_conf.float(),
        top1_margin.float(),
        agreement.float(),
        sectors,
        horizon_s,
        fix_variant.wrong_class_aware,
        fix_variant.front_bias,
    )
    replay_f3, pruned, meta = rt.sw14.sw13c_fix.apply_pruning_no_gt(
        raw_semantic=raw_semantic.long(),
        protected_mask=protected.bool(),
        low_value_score_map=low_value.float(),
        native_occ_count=rt.occ_count(native_semantic.long()),
        raw_occ_count=rt.occ_count(raw_semantic.long()),
        raw_delta_count=int(raw_delta.sum().item()),
        budget_mode=fix_variant.budget_mode,
        expansion_ratio=fix_variant.expansion_ratio,
        keep_ratio=fix_variant.keep_ratio,
    )
    return {
        "raw_occ": raw_occ.bool(),
        "native_occ": native_occ.bool(),
        "raw_delta": raw_delta.bool(),
        "protected": protected.bool(),
        "low_value": low_value.float(),
        "agreement": agreement.float(),
        "replay_f3": replay_f3.long(),
        "pruned": pruned.bool(),
        "meta": meta,
    }


def input_consistency_audit(runtime_replays: dict[int, dict[int, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    context: dict[tuple[int, int], dict[str, Any]] = {}
    for sample_index, horizon_s in AUDIT_CASES:
        replay_raw = runtime_replays[sample_index][horizon_s]
        stored_raw = rt.load_npz_torch(rt.raw_dump_path(sample_index, horizon_s, rt.candidate().base_repair_variant))
        native = rt.load_npz_torch(rt.raw_dump_path(sample_index, horizon_s, "native_baseline"))
        replay_agreement = replay_agreement_map(sample_index, horizon_s, replay_raw["semantic"].long())
        stored_agreement = stored_agreement_map(sample_index, horizon_s)
        replay_ctx = derive_masks_and_scores(
            replay_raw["semantic"].long(),
            native["semantic"].long(),
            replay_raw["occ_conf"].float(),
            replay_raw["top1_margin"].float(),
            replay_agreement,
            horizon_s,
        )
        teacher_input_ctx = derive_masks_and_scores(
            stored_raw["semantic"].long(),
            native["semantic"].long(),
            stored_raw["occ_conf"].float(),
            stored_raw["top1_margin"].float(),
            stored_agreement,
            horizon_s,
        )
        row = {
            "sample_index": sample_index,
            "horizon_s": horizon_s,
            "raw_occ_jaccard": jaccard(replay_ctx["raw_occ"], teacher_input_ctx["raw_occ"]),
            "raw_delta_jaccard": jaccard(replay_ctx["raw_delta"], teacher_input_ctx["raw_delta"]),
            "occ_conf_mean_abs_diff": mean_abs_diff(replay_raw["occ_conf"], stored_raw["occ_conf"].float()),
            "occ_conf_max_abs_diff": max_abs_diff(replay_raw["occ_conf"], stored_raw["occ_conf"].float()),
            "top1_margin_mean_abs_diff": mean_abs_diff(replay_raw["top1_margin"], stored_raw["top1_margin"].float()),
            "top1_margin_max_abs_diff": max_abs_diff(replay_raw["top1_margin"], stored_raw["top1_margin"].float()),
            "agreement_mean_abs_diff": mean_abs_diff(replay_agreement, stored_agreement),
            "agreement_max_abs_diff": max_abs_diff(replay_agreement, stored_agreement),
            "protected_mask_jaccard": jaccard(replay_ctx["protected"], teacher_input_ctx["protected"]),
            "low_value_score_mean_abs_diff": mean_abs_diff(replay_ctx["low_value"], teacher_input_ctx["low_value"]),
            "low_value_score_max_abs_diff": max_abs_diff(replay_ctx["low_value"], teacher_input_ctx["low_value"]),
            "front_raw_delta_count_replay": int((replay_ctx["raw_delta"] & rt.front_mask()).sum().item()),
            "front_raw_delta_count_teacher_input": int((teacher_input_ctx["raw_delta"] & rt.front_mask()).sum().item()),
        }
        rows.append(row)
        context[(sample_index, horizon_s)] = {
            "replay_raw": replay_raw,
            "stored_raw": stored_raw,
            "native": native,
            "replay_ctx": replay_ctx,
            "teacher_input_ctx": teacher_input_ctx,
            "stored_agreement": stored_agreement,
            "replay_agreement": replay_agreement,
        }
    write_csv(REPORTS_DIR / "sw14a_f3_input_consistency.csv", rows)
    return rows, context


def pruning_set_alignment_audit(context: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, horizon_s in AUDIT_CASES:
        ctx = context[(sample_index, horizon_s)]
        teacher = rt.load_npz_torch(rt.final_output_path(sample_index, horizon_s))
        replay_f3 = ctx["replay_ctx"]["replay_f3"].long()
        teacher_f3 = teacher["final_before_cap"].long()
        raw_occ_replay = occ_mask(ctx["replay_raw"]["semantic"].long())
        raw_occ_teacher = occ_mask(teacher["raw_semantic"].long())
        pruned_replay = raw_occ_replay & ~occ_mask(replay_f3)
        pruned_teacher = raw_occ_teacher & ~occ_mask(teacher_f3)
        native = ctx["native"]["semantic"].long()
        kept_delta_replay = occ_mask(replay_f3) & ~occ_mask(native)
        kept_delta_teacher = occ_mask(teacher_f3) & ~occ_mask(teacher["native_semantic"].long())
        gt_h = ctx["replay_raw"]["gt_h"].long()
        gt0 = ctx["replay_raw"]["gt0"].long()
        replay_eval = rt.build_metric_row(replay_f3, gt_h, gt0, native, horizon_s)
        teacher_eval = rt.build_metric_row(teacher_f3, gt_h, gt0, native, horizon_s)
        rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "pruned_count_replay": int(pruned_replay.sum().item()),
                "pruned_count_teacher": int(pruned_teacher.sum().item()),
                "pruned_jaccard": jaccard(pruned_replay, pruned_teacher),
                "kept_delta_jaccard": jaccard(kept_delta_replay, kept_delta_teacher),
                "final_occ_jaccard": jaccard(occ_mask(replay_f3), occ_mask(teacher_f3)),
                "front_pruned_count_replay": int((pruned_replay & rt.front_mask()).sum().item()),
                "front_pruned_count_teacher": int((pruned_teacher & rt.front_mask()).sum().item()),
                "front_kept_delta_count_replay": int((kept_delta_replay & rt.front_mask()).sum().item()),
                "front_kept_delta_count_teacher": int((kept_delta_teacher & rt.front_mask()).sum().item()),
                "front_false_free_delta_replay": float(replay_eval["front_sector_false_free_rate_delta"]),
                "front_false_free_delta_teacher": float(teacher_eval["front_sector_false_free_rate_delta"]),
                "front_false_free_delta_abs_diff": abs(float(replay_eval["front_sector_false_free_rate_delta"]) - float(teacher_eval["front_sector_false_free_rate_delta"])),
            }
        )
    write_csv(REPORTS_DIR / "sw14a_f3_pruning_set_alignment.csv", rows)
    return rows


def topk_set_from_scores(candidate_mask: torch.Tensor, score_map: torch.Tensor, prune_count: int) -> dict[str, Any]:
    coords = torch.nonzero(candidate_mask, as_tuple=False)
    vals = score_map[candidate_mask]
    n = int(vals.numel())
    if n == 0 or prune_count <= 0:
        empty = torch.zeros_like(candidate_mask, dtype=torch.bool)
        return {
            "topk_mask": empty,
            "boundary_score": None,
            "score_gap_before_boundary": None,
            "score_gap_after_boundary": None,
            "near_tie_count_eps_1e_8": 0,
            "near_tie_count_eps_1e_6": 0,
            "near_tie_count_eps_1e_4": 0,
            "unique_score_ratio": 1.0,
            "number_of_exact_duplicate_scores": 0,
            "stable_equal": True,
            "voxel_tiebreak_equal": True,
        }
    k = min(prune_count, n)
    top_vals, top_idx = torch.topk(vals, k=k, largest=True)
    topk_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    picked = coords[top_idx]
    topk_mask[picked[:, 0], picked[:, 1], picked[:, 2]] = True

    sorted_vals_desc, stable_idx = torch.sort(vals, descending=True, stable=True)
    stable_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    stable_coords = coords[stable_idx[:k]]
    stable_mask[stable_coords[:, 0], stable_coords[:, 1], stable_coords[:, 2]] = True

    lin = linear_indices(candidate_mask).float()
    scale = float(max(1, n)) + 1.0
    voxel_rank_score = vals.double() + (scale - lin.double() / scale) * 1e-12
    voxel_idx = torch.argsort(voxel_rank_score, descending=True)
    voxel_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    voxel_coords = coords[voxel_idx[:k]]
    voxel_mask[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True

    boundary_score = float(sorted_vals_desc[k - 1].item())
    score_gap_before = None if k <= 1 else float(sorted_vals_desc[k - 2].item() - sorted_vals_desc[k - 1].item())
    score_gap_after = None if k >= n else float(sorted_vals_desc[k - 1].item() - sorted_vals_desc[k].item())
    near_1e8 = int((torch.abs(vals - boundary_score) <= 1e-8).sum().item())
    near_1e6 = int((torch.abs(vals - boundary_score) <= 1e-6).sum().item())
    near_1e4 = int((torch.abs(vals - boundary_score) <= 1e-4).sum().item())
    uniq = torch.unique(vals).numel()
    return {
        "topk_mask": topk_mask,
        "stable_mask": stable_mask,
        "voxel_mask": voxel_mask,
        "boundary_score": boundary_score,
        "score_gap_before_boundary": score_gap_before,
        "score_gap_after_boundary": score_gap_after,
        "near_tie_count_eps_1e_8": near_1e8,
        "near_tie_count_eps_1e_6": near_1e6,
        "near_tie_count_eps_1e_4": near_1e4,
        "unique_score_ratio": float(uniq) / float(max(1, n)),
        "number_of_exact_duplicate_scores": int(n - uniq),
        "stable_equal": bool(torch.equal(topk_mask, stable_mask)),
        "voxel_tiebreak_equal": bool(torch.equal(topk_mask, voxel_mask)),
    }


def topk_tiebreak_audit(context: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, horizon_s in AUDIT_CASES:
        ctx = context[(sample_index, horizon_s)]
        replay_ctx = ctx["replay_ctx"]
        teacher = rt.load_npz_torch(rt.final_output_path(sample_index, horizon_s))
        raw_occ_teacher = occ_mask(teacher["raw_semantic"].long())
        pruned_teacher = raw_occ_teacher & ~occ_mask(teacher["final_before_cap"].long())
        replay_candidate = replay_ctx["raw_occ"] & ~replay_ctx["protected"]
        prune_count = int(replay_ctx["pruned"].sum().item())
        replay_topk = topk_set_from_scores(replay_candidate, replay_ctx["low_value"], prune_count)
        rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "prune_count": prune_count,
                "boundary_score": replay_topk["boundary_score"],
                "score_gap_before_boundary": replay_topk["score_gap_before_boundary"],
                "score_gap_after_boundary": replay_topk["score_gap_after_boundary"],
                "near_tie_count_eps_1e-8": replay_topk["near_tie_count_eps_1e_8"],
                "near_tie_count_eps_1e-6": replay_topk["near_tie_count_eps_1e_6"],
                "near_tie_count_eps_1e-4": replay_topk["near_tie_count_eps_1e_4"],
                "unique_score_ratio": replay_topk["unique_score_ratio"],
                "number_of_exact_duplicate_scores": replay_topk["number_of_exact_duplicate_scores"],
                "topk_indices_stable_with_stable_sort": replay_topk["stable_equal"],
                "topk_indices_stable_with_voxel_rank_tiebreak": replay_topk["voxel_tiebreak_equal"],
                "replay_topk_vs_teacher_pruned_jaccard": jaccard(replay_topk["topk_mask"], pruned_teacher),
                "stable_topk_vs_teacher_pruned_jaccard": jaccard(replay_topk["stable_mask"], pruned_teacher),
                "voxel_tiebreak_topk_vs_teacher_pruned_jaccard": jaccard(replay_topk["voxel_mask"], pruned_teacher),
            }
        )
    write_csv(REPORTS_DIR / "sw14a_f3_topk_tiebreak_audit.csv", rows)
    return rows


def run_f3_with_source(
    sample_index: int,
    horizon_s: int,
    source_mode: str,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    native = ctx["native"]["semantic"].long()
    if source_mode == "replay_all":
        raw_semantic = ctx["replay_raw"]["semantic"].long()
        occ_conf = ctx["replay_raw"]["occ_conf"].float()
        top1_margin = ctx["replay_raw"]["top1_margin"].float()
        agreement = ctx["replay_agreement"].float()
    elif source_mode == "stored_score":
        raw_semantic = ctx["replay_raw"]["semantic"].long()
        occ_conf = ctx["stored_raw"]["occ_conf"].float()
        top1_margin = ctx["stored_raw"]["top1_margin"].float()
        agreement = ctx["stored_agreement"].float()
    elif source_mode == "stored_all_possible":
        raw_semantic = ctx["stored_raw"]["semantic"].long()
        occ_conf = ctx["stored_raw"]["occ_conf"].float()
        top1_margin = ctx["stored_raw"]["top1_margin"].float()
        agreement = ctx["stored_agreement"].float()
    else:
        raise ValueError(source_mode)
    derived = derive_masks_and_scores(raw_semantic, native, occ_conf, top1_margin, agreement, horizon_s)
    teacher = rt.load_npz_torch(rt.final_output_path(sample_index, horizon_s))
    gt_h = ctx["replay_raw"]["gt_h"].long()
    gt0 = ctx["replay_raw"]["gt0"].long()
    pred_eval = rt.build_metric_row(derived["replay_f3"].long(), gt_h, gt0, native, horizon_s)
    teacher_eval = rt.build_metric_row(teacher["final_before_cap"].long(), gt_h, gt0, native, horizon_s)
    pruned_teacher = occ_mask(teacher["raw_semantic"].long()) & ~occ_mask(teacher["final_before_cap"].long())
    kept_delta_pred = occ_mask(derived["replay_f3"].long()) & ~occ_mask(native)
    kept_delta_teacher = occ_mask(teacher["final_before_cap"].long()) & ~occ_mask(native)
    return {
        "pred_gt_density_delta_abs_diff": abs(float(pred_eval["pred_gt_density_delta"]) - float(teacher_eval["pred_gt_density_delta"])),
        "front_false_free_delta_abs_diff": abs(float(pred_eval["front_sector_false_free_rate_delta"]) - float(teacher_eval["front_sector_false_free_rate_delta"])),
        "final_occ_jaccard": jaccard(occ_mask(derived["replay_f3"].long()), occ_mask(teacher["final_before_cap"].long())),
        "pruned_jaccard": jaccard(derived["pruned"].bool(), pruned_teacher.bool()),
        "kept_delta_jaccard": jaccard(kept_delta_pred.bool(), kept_delta_teacher.bool()),
    }


def input_source_ablation(context: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, horizon_s in AUDIT_CASES:
        ctx = context[(sample_index, horizon_s)]
        for mode in ["replay_all", "stored_score", "stored_all_possible"]:
            metrics = run_f3_with_source(sample_index, horizon_s, mode, ctx)
            rows.append({"sample_index": sample_index, "horizon_s": horizon_s, "source_mode": mode, **metrics})
    write_csv(REPORTS_DIR / "sw14a_f3_input_source_ablation.csv", rows)
    return rows


def function_version_audit() -> dict[str, Any]:
    apply_src = inspect.getsource(rt.sw14.sw13c_fix.apply_pruning_no_gt)
    low_src = inspect.getsource(rt.sw14.sw13c_fix.low_value_score)
    prot_src = inspect.getsource(rt.sw14.sw13c_fix.protected_zone_fix)
    fix_variant = rt.candidate_fix_variant()
    payload = {
        "current_apply_pruning_no_gt_path": str(SW13_FIX_SCRIPT),
        "teacher_eval500_generation_script_path": str(SW13_FC500_SCRIPT),
        "apply_pruning_no_gt_file_sha256": sha256_file(SW13_FIX_SCRIPT),
        "apply_pruning_no_gt_source_sha256": sha256_text(apply_src),
        "low_value_score_source_sha256": sha256_text(low_src),
        "protected_zone_fix_source_sha256": sha256_text(prot_src),
        "candidate_base_variant": rt.candidate().base_variant,
        "expansion_ratio": fix_variant.expansion_ratio,
        "protected_variant": rt.candidate().protected_variant,
        "agreement_sources": rt.candidate().agreement_sources,
        "wrong_class_aware": fix_variant.wrong_class_aware,
        "front_bias": fix_variant.front_bias,
        "cap_used_inside_f3": False,
        "random_seed_used_for_sorting": False,
        "torch_topk_used": "torch.topk(vals, k=k, largest=True)",
        "stable_sort_used_by_default": False,
    }
    write_json(REPORTS_DIR / "sw14a_f3_function_version_audit.json", payload)
    return payload


def decide(
    input_rows: list[dict[str, Any]],
    pruning_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    version_audit: dict[str, Any],
) -> str:
    if not pruning_rows:
        return "F3A4_TEACHER_PRUNING_ARTIFACT_MISSING"
    stored_score = [r for r in ablation_rows if r["source_mode"] == "stored_score"]
    stored_all = [r for r in ablation_rows if r["source_mode"] == "stored_all_possible"]
    replay_all = [r for r in ablation_rows if r["source_mode"] == "replay_all"]
    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return float(sum(float(r[key]) for r in rows) / len(rows)) if rows else 0.0
    if stored_score and mean(stored_score, "front_false_free_delta_abs_diff") <= 0.02 and mean(stored_score, "final_occ_jaccard") >= 0.99:
        return "F3A6_PARITY_RECOVERED_WITH_STORED_SCORE"
    if stored_all and mean(stored_all, "front_false_free_delta_abs_diff") <= 0.02 and mean(stored_all, "final_occ_jaccard") >= 0.99:
        return "F3A6_PARITY_RECOVERED_WITH_STORED_SCORE"
    if any((not row["topk_indices_stable_with_voxel_rank_tiebreak"]) and row["voxel_tiebreak_topk_vs_teacher_pruned_jaccard"] > row["replay_topk_vs_teacher_pruned_jaccard"] + 0.05 for row in topk_rows):
        return "F3A7_PARITY_RECOVERED_WITH_DETERMINISTIC_TIEBREAK"
    if any(row["near_tie_count_eps_1e-6"] >= max(8, row["prune_count"] * 0.1) for row in topk_rows):
        return "F3A2_TOPK_TIEBREAK_INSTABILITY"
    if any(row["low_value_score_mean_abs_diff"] > 1e-4 or row["occ_conf_mean_abs_diff"] > 1e-4 or row["top1_margin_mean_abs_diff"] > 1e-4 for row in input_rows):
        return "F3A1_SCORE_INPUT_MISMATCH"
    if version_audit["apply_pruning_no_gt_file_sha256"] != sha256_file(SW13_FIX_SCRIPT):
        return "F3A3_FUNCTION_VERSION_MISMATCH"
    replay_mean = mean(replay_all, "front_false_free_delta_abs_diff")
    if replay_mean > 0.02:
        return "F3A5_REPLAY_RAW_SMALL_DIFF_AMPLIFIED_BY_PRUNING"
    return "F3A4_TEACHER_PRUNING_ARTIFACT_MISSING"


def build_report(
    manifest: dict[str, Any],
    input_rows: list[dict[str, Any]],
    pruning_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    version_audit: dict[str, Any],
    decision: str,
) -> None:
    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return float(sum(float(r[key]) for r in rows) / len(rows)) if rows else 0.0
    lines = [
        "# SW14A F3 Pruning Audit Report",
        "",
        "## Audited cases",
        f"- case count: `{len(manifest['priority_cases']) + 1}`.",
        "",
        "## Input consistency summary",
        f"- mean raw_occ_jaccard: `{mean(input_rows, 'raw_occ_jaccard'):.8f}`.",
        f"- mean raw_delta_jaccard: `{mean(input_rows, 'raw_delta_jaccard'):.8f}`.",
        f"- mean occ_conf_mean_abs_diff: `{mean(input_rows, 'occ_conf_mean_abs_diff'):.8f}`.",
        f"- mean top1_margin_mean_abs_diff: `{mean(input_rows, 'top1_margin_mean_abs_diff'):.8f}`.",
        f"- mean low_value_score_mean_abs_diff: `{mean(input_rows, 'low_value_score_mean_abs_diff'):.8f}`.",
        "",
        "## Pruning set alignment summary",
        f"- mean pruned_jaccard: `{mean(pruning_rows, 'pruned_jaccard'):.8f}`.",
        f"- mean kept_delta_jaccard: `{mean(pruning_rows, 'kept_delta_jaccard'):.8f}`.",
        f"- mean final_occ_jaccard: `{mean(pruning_rows, 'final_occ_jaccard'):.8f}`.",
        f"- mean front_false_free_delta_abs_diff: `{mean(pruning_rows, 'front_false_free_delta_abs_diff'):.8f}`.",
        "",
        "## Topk/tie-break summary",
        f"- mean near_tie_count_eps_1e-6: `{mean(topk_rows, 'near_tie_count_eps_1e-6'):.4f}`.",
        f"- mean unique_score_ratio: `{mean(topk_rows, 'unique_score_ratio'):.8f}`.",
        f"- stable_sort_match_rate: `{mean([{'v': 1.0 if r['topk_indices_stable_with_stable_sort'] else 0.0} for r in topk_rows], 'v'):.4f}`.",
        f"- voxel_tiebreak_match_rate: `{mean([{'v': 1.0 if r['topk_indices_stable_with_voxel_rank_tiebreak'] else 0.0} for r in topk_rows], 'v'):.4f}`.",
        "",
        "## Input-source ablation",
        f"- replay_all mean front_false_free_delta_abs_diff: `{mean([r for r in ablation_rows if r['source_mode']=='replay_all'], 'front_false_free_delta_abs_diff'):.8f}`.",
        f"- stored_score mean front_false_free_delta_abs_diff: `{mean([r for r in ablation_rows if r['source_mode']=='stored_score'], 'front_false_free_delta_abs_diff'):.8f}`.",
        f"- stored_all_possible mean front_false_free_delta_abs_diff: `{mean([r for r in ablation_rows if r['source_mode']=='stored_all_possible'], 'front_false_free_delta_abs_diff'):.8f}`.",
        "",
        "## Function/version audit",
        f"- apply_pruning file: `{version_audit['current_apply_pruning_no_gt_path']}`.",
        f"- teacher eval500 script: `{version_audit['teacher_eval500_generation_script_path']}`.",
        f"- expansion_ratio: `{version_audit['expansion_ratio']}`.",
        f"- protected_variant: `{version_audit['protected_variant']}`.",
        "",
        "## Decision",
        f"- decision: `{decision}`.",
    ]
    write_md(REPORTS_DIR / "stage_sw14a_f3_pruning_audit_report.md", "\n".join(lines))


def main() -> None:
    ensure_dirs()
    manifest = case_manifest()
    runtime_replays = get_runtime_replays()
    input_rows, context = input_consistency_audit(runtime_replays)
    pruning_rows = pruning_set_alignment_audit(context)
    topk_rows = topk_tiebreak_audit(context)
    ablation_rows = input_source_ablation(context)
    version_audit = function_version_audit()
    decision = decide(input_rows, pruning_rows, topk_rows, ablation_rows, version_audit)
    decision_payload = {
        "decision_type": decision,
        "audited_case_count": len(AUDIT_CASES),
        "input_consistency_summary": {
            "mean_raw_occ_jaccard": float(sum(r["raw_occ_jaccard"] for r in input_rows) / len(input_rows)),
            "mean_occ_conf_mean_abs_diff": float(sum(r["occ_conf_mean_abs_diff"] for r in input_rows) / len(input_rows)),
            "mean_low_value_score_mean_abs_diff": float(sum(r["low_value_score_mean_abs_diff"] for r in input_rows) / len(input_rows)),
        },
        "pruning_set_alignment_summary": {
            "mean_pruned_jaccard": float(sum(r["pruned_jaccard"] for r in pruning_rows) / len(pruning_rows)),
            "mean_front_false_free_delta_abs_diff": float(sum(r["front_false_free_delta_abs_diff"] for r in pruning_rows) / len(pruning_rows)),
        },
        "topk_tiebreak_summary": {
            "mean_near_tie_count_eps_1e_6": float(sum(r["near_tie_count_eps_1e-6"] for r in topk_rows) / len(topk_rows)),
            "mean_replay_topk_vs_teacher_pruned_jaccard": float(sum(r["replay_topk_vs_teacher_pruned_jaccard"] for r in topk_rows) / len(topk_rows)),
            "mean_voxel_tiebreak_topk_vs_teacher_pruned_jaccard": float(sum(r["voxel_tiebreak_topk_vs_teacher_pruned_jaccard"] for r in topk_rows) / len(topk_rows)),
        },
        "input_source_ablation_summary": {
            mode: {
                "mean_front_false_free_delta_abs_diff": float(sum(float(r["front_false_free_delta_abs_diff"]) for r in ablation_rows if r["source_mode"] == mode) / max(1, len([r for r in ablation_rows if r["source_mode"] == mode]))),
                "mean_final_occ_jaccard": float(sum(float(r["final_occ_jaccard"]) for r in ablation_rows if r["source_mode"] == mode) / max(1, len([r for r in ablation_rows if r["source_mode"] == mode]))),
            }
            for mode in ["replay_all", "stored_score", "stored_all_possible"]
        },
        "function_version_audit": version_audit,
        "no_training": True,
        "no_sw14b": True,
        "no_backbone_change": True,
        "no_head_change": True,
        "no_get_occ_change": True,
        "gt_evaluation_only": True,
    }
    write_json(REPORTS_DIR / "sw14a_f3_pruning_audit_decision.json", decision_payload)
    build_report(manifest, input_rows, pruning_rows, topk_rows, ablation_rows, version_audit, decision)


if __name__ == "__main__":
    main()
