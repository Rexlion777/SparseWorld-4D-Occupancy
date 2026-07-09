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

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_postprocess_parity"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_postprocess_parity"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14_feature_gate_postprocess_parity"

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


sw14 = load_module("sw14_postprocess_base", BASE_SW14)
causal = load_module("sw14_postprocess_causal", BASE_CAUSAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14A postprocess parity audit for hard R8 dense replay")
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


def candidate():
    return sw14.teacher_candidates()[0]


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
                path = raw_dump_path(sample_index, horizon_s, variant_label)
                if not path.exists():
                    missing.append(str(path))
            out_path = final_output_path(sample_index, horizon_s)
            if not out_path.exists():
                missing.append(str(out_path))
    return len(missing) == 0, missing


def occ_count(semantic: torch.Tensor) -> int:
    return int((semantic != EMPTY_IDX).sum().item())


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic != EMPTY_IDX


def bool_mean(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())


def front_occ_count(semantic: torch.Tensor, front_mask: torch.Tensor) -> int:
    return int(((semantic != EMPTY_IDX) & front_mask).sum().item())


def semantic_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def occ_diff_count(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((occ_mask(a) ^ occ_mask(b)).sum().item())


def occ_diff_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return bool_mean(occ_mask(a) ^ occ_mask(b))


def build_metric_row(pred: torch.Tensor, gt_h: torch.Tensor, gt0: torch.Tensor, native: torch.Tensor, horizon_s: int) -> dict[str, Any]:
    sectors = {name: tensor.cpu().bool() for name, tensor in sw14.sw7.build_sector_masks().items()}
    pred_row = sw14.sw13c_fix.build_eval_row(pred.long(), gt_h.long(), gt0.long(), candidate().perturbation_id, horizon_s, sectors, native.long())
    native_row = sw14.sw13c_fix.build_eval_row(native.long(), gt_h.long(), gt0.long(), candidate().perturbation_id, horizon_s, sectors, native.long())
    out = dict(pred_row)
    out["front_sector_false_free_rate_delta_vs_native"] = float(pred_row["front_sector_false_free_rate"]) - float(native_row["front_sector_false_free_rate"])
    out["future_h4_h6_false_free_rate_delta_vs_native"] = (
        float(pred_row["future_h4_h6_false_free_rate"]) - float(native_row["future_h4_h6_false_free_rate"])
        if horizon_s in {4, 6}
        else 0.0
    )
    out["A10_front_h6_recovery_rate_delta_vs_native"] = float(pred_row["A10_front_h6_recovery_rate"]) - float(native_row["A10_front_h6_recovery_rate"])
    out["false_positive_delta"] = float(pred_row["false_positive_rate"]) - float(native_row["false_positive_rate"])
    out["wrong_class_delta"] = float(pred_row["wrong_class_rate"]) - float(native_row["wrong_class_rate"])
    return out


def candidate_fix_variant() -> Any:
    for base in sw14.sw13c_fix.base_specs():
        if base.perturbation_id == candidate().perturbation_id and base.raw_variant_label == candidate().base_repair_variant:
            for variant in sw14.sw13c_fix.fix_variants_for(base):
                if variant.label == candidate().base_variant:
                    return variant
    raise KeyError(candidate().base_variant)


def run_raw_replay(
    model: Any,
    dataset: Any,
    sample_index: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    cand = candidate()
    variant = sw14.variant_spec_for_candidate(cand)
    cache_path = SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_index:03d}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
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
    return out, {"sample_index": sample_index, "camera_order": list(sw14.CAMERA_NAMES), "variant": cand.base_repair_variant}


def phase1_manifest(sample_ids: list[int]) -> dict[str, Any]:
    cand = candidate()
    payload = {
        "teacher_candidate": cand.name,
        "perturbation_id": cand.perturbation_id,
        "base_repair_variant": cand.base_repair_variant,
        "base_variant": cand.base_variant,
        "expansion_ratio": cand.expansion_ratio,
        "front_cap_variant": cand.front_cap_variant,
        "cap_ratio": cand.cap_ratio,
        "protected_variant": cand.protected_variant,
        "agreement_sources": cand.agreement_sources,
        "agreement_source_count": cand.agreement_source_count,
        "teacher_final_pipeline": "hard R8 dense feature replay -> raw semantic/logits -> F3 no-GT density budget -> FC1_1p3 FrontCap",
        "teacher_final_is_not": "raw R8 replay -> FrontCap only",
        "sample_range": [sample_ids[0], sample_ids[-1]],
        "horizons": CORE_HORIZONS,
        "no_training": True,
        "no_sw14b": True,
        "no_backbone_change": True,
        "no_head_change": True,
        "no_get_occ_change": True,
        "gt_evaluation_only": True,
    }
    write_json(REPORTS_DIR / "sw14a_postprocess_teacher_pipeline_manifest.json", payload)
    return payload


def phase2_raw_parity(sample_ids: list[int]) -> tuple[list[dict[str, Any]], str]:
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    phase_decision = "R1_RAW_REPLAY_MATCHES_SW13_R8"
    for sample_index in sample_ids:
        replay_by_h, replay_meta = run_raw_replay(model, dataset, sample_index)
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
                "phase_decision": None,
                "replay_raw_occ_count": occ_count(replay["semantic"]),
                "sw13_raw_occ_count": occ_count(sw13_raw["semantic"]),
                "replay_vs_sw13_raw_occ_diff_count": occ_diff_count(replay["semantic"], sw13_raw["semantic"]),
                "replay_vs_sw13_raw_occ_diff": occ_diff_ratio(replay["semantic"], sw13_raw["semantic"]),
                "replay_vs_sw13_raw_semantic_abs_diff": semantic_abs_diff(replay["semantic"], sw13_raw["semantic"]),
                "replay_raw_density_delta": float(replay_eval["pred_gt_density_delta"]),
                "sw13_raw_density_delta": float(sw13_eval["pred_gt_density_delta"]),
                "replay_vs_sw13_occ_conf_abs_diff": float((replay["occ_conf"] - sw13_raw["occ_conf"].float()).abs().mean().item()),
                "replay_vs_sw13_margin_abs_diff": float((replay["top1_margin"] - sw13_raw["top1_margin"].float()).abs().mean().item()),
                "runtime_camera_order": replay_meta["camera_order"],
            }
            rows.append(row)
            if row["replay_vs_sw13_raw_occ_diff_count"] != 0:
                phase_decision = "R2_RAW_REPLAY_DIFFERS_FROM_SW13_R8"
    del model
    for row in rows:
        row["phase_decision"] = phase_decision
    write_csv(REPORTS_DIR / "sw14a_postprocess_raw_r8_parity.csv", rows)
    return rows, phase_decision


def phase2_runtime_context_audit(sample_ids: list[int]) -> dict[str, Any]:
    probe_ids = [sample_ids[0]]
    if len(sample_ids) > 1:
        probe_ids.append(sample_ids[1])
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    probes: list[dict[str, Any]] = []
    for sample_index in probe_ids:
        cand = candidate()
        variant = sw14.variant_spec_for_candidate(cand)
        raw_sample, batch_clean = sw14.sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw14.sw2.unwrap(raw_sample)
        batch_deg = sw14.perturbation_batch_compatible(batch_clean)
        batch_deg = sw14.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, sw14.sw81.sw5_engine.build_catalog()[cand.perturbation_id])[0]
        cache_artifact = torch.load(SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_index:03d}.pt", map_location="cpu", weights_only=False)
        moved_clean = sw14.sw2.move_to_cuda(batch_clean)
        sw14.sw13a.reset_model_cache(model)
        _, _, cache_live = sw14.sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_index)
        replay_artifact, meta_artifact = run_raw_replay(model, dataset, sample_index)
        # rebuild with live cache through sw13a.run_variant_forward
        _, per_h_live, debug_live = sw14.sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache_live, variant, cand.perturbation_id)
        _, per_h_art, debug_art = sw14.sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache_artifact, variant, cand.perturbation_id)
        horizon_rows: list[dict[str, Any]] = []
        for horizon_s in CORE_HORIZONS:
            sw13_raw = load_npz_torch(raw_dump_path(sample_index, horizon_s, cand.base_repair_variant))
            sem_live, _ = sw14.sw13b.dense_debug_for_case(model, per_h_live[horizon_s]["pred_dict"])
            sem_art, _ = sw14.sw13b.dense_debug_for_case(model, per_h_art[horizon_s]["pred_dict"])
            horizon_rows.append(
                {
                    "horizon_s": horizon_s,
                    "causal_replay_vs_sw13_occ_diff_count": occ_diff_count(replay_artifact[horizon_s]["semantic"], sw13_raw["semantic"].long()),
                    "sw13a_live_cache_vs_sw13_occ_diff_count": occ_diff_count(sem_live.long(), sw13_raw["semantic"].long()),
                    "sw13a_artifact_cache_vs_sw13_occ_diff_count": occ_diff_count(sem_art.long(), sw13_raw["semantic"].long()),
                    "live_vs_artifact_cache_occ_diff_count": occ_diff_count(sem_live.long(), sem_art.long()),
                }
            )
        probes.append(
            {
                "sample_index": sample_index,
                "camera_order_causal": meta_artifact["camera_order"],
                "camera_order_sw13a_frame0": debug_art["frame_debug"][0]["camera_names"],
                "frame0_repair_applied": debug_art["frame_debug"][0]["repair_applied"],
                "repair_camera_names": debug_art["frame_debug"][0]["repaired_camera_names"],
                "source_time_offsets": debug_art["frame_debug"][0]["source_time_offsets"],
                "horizons": horizon_rows,
            }
        )
    del model
    payload = {
        "probe_samples": probe_ids,
        "interpretation": [
            "sample100 is included as an expected near-exact reference",
            "sample101 is included as the first mismatching raw replay case when available",
            "if sw13a.run_variant_forward with live cache and artifact cache both mismatch SW13 raw dump by the same amount, cache serialization is not the main cause",
            "if camera order and repaired camera names are correct, the remaining mismatch is upstream runtime-context or eval-stage path divergence relative to stored SW13C eval500 raw dumps",
        ],
        "probes": probes,
    }
    write_json(REPORTS_DIR / "sw14a_postprocess_runtime_context_audit.json", payload)
    return payload


def phase3_f3_parity(raw_rows: list[dict[str, Any]], sample_ids: list[int]) -> tuple[list[dict[str, Any]], bool]:
    sectors = {name: tensor.cpu().bool() for name, tensor in sw14.sw7.build_sector_masks().items()}
    fix_variant = candidate_fix_variant()
    rows: list[dict[str, Any]] = []
    all_match = True
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    for sample_index in sample_ids:
        replay_by_h, _ = run_raw_replay(model, dataset, sample_index)
        for horizon_s in CORE_HORIZONS:
            replay = replay_by_h[horizon_s]
            native = load_npz_torch(raw_dump_path(sample_index, horizon_s, "native_baseline"))
            teacher = load_npz_torch(final_output_path(sample_index, horizon_s))
            agreement = sw14.agreement_map_for_sample(sample_index, horizon_s, candidate()).float()
            replay_raw_occ = occ_mask(replay["semantic"])
            native_occ = occ_mask(native["semantic"].long())
            replay_delta = replay_raw_occ & ~native_occ
            protected = sw14.sw13c_fix.protected_zone_fix(
                candidate().protected_variant,
                replay_raw_occ,
                replay_delta,
                replay["occ_conf"],
                replay["top1_margin"],
                agreement,
                sectors,
                horizon_s,
            )
            low_value = sw14.sw13c_fix.low_value_score(
                replay_raw_occ,
                protected,
                replay["occ_conf"],
                replay["top1_margin"],
                agreement,
                sectors,
                horizon_s,
                fix_variant.wrong_class_aware,
                fix_variant.front_bias,
            )
            replay_f3, pruned, meta = sw14.sw13c_fix.apply_pruning_no_gt(
                raw_semantic=replay["semantic"].long(),
                protected_mask=protected.bool(),
                low_value_score_map=low_value.float(),
                native_occ_count=occ_count(native["semantic"].long()),
                raw_occ_count=occ_count(replay["semantic"].long()),
                raw_delta_count=int(replay_delta.sum().item()),
                budget_mode=fix_variant.budget_mode,
                expansion_ratio=fix_variant.expansion_ratio,
                keep_ratio=fix_variant.keep_ratio,
            )
            gt_h = replay["gt_h"].long()
            gt0 = replay["gt0"].long()
            replay_eval = build_metric_row(replay_f3, gt_h, gt0, native["semantic"].long(), horizon_s)
            row = {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "replay_f3_occ_count": occ_count(replay_f3),
                "teacher_f3_occ_count": occ_count(teacher["final_before_cap"].long()),
                "replay_vs_teacher_f3_occ_diff_count": occ_diff_count(replay_f3, teacher["final_before_cap"].long()),
                "replay_vs_teacher_f3_occ_diff": occ_diff_ratio(replay_f3, teacher["final_before_cap"].long()),
                "replay_vs_teacher_f3_semantic_abs_diff": semantic_abs_diff(replay_f3, teacher["final_before_cap"].long()),
                "replay_f3_pred_gt_density_delta": float(replay_eval["pred_gt_density_delta"]),
                "replay_f3_front_false_free_delta": float(replay_eval["front_sector_false_free_rate_delta_vs_native"]),
                "replay_f3_raw_delta_keep_ratio": sw14.safe_div(
                    float((occ_mask(replay_f3) & replay_delta).sum().item()),
                    max(1.0, float(replay_delta.sum().item())),
                ),
                "replay_f3_final_native_expansion_ratio": sw14.safe_div(
                    float(occ_count(replay_f3) - occ_count(native["semantic"].long())),
                    max(1.0, float(occ_count(native["semantic"].long()))),
                ),
                "replay_f3_pruning_from_protected_ratio": sw14.safe_div(
                    float((pruned.bool() & protected.bool()).sum().item()),
                    max(1.0, float(pruned.bool().sum().item())),
                ),
                "budget_meta": meta,
            }
            rows.append(row)
            if row["replay_vs_teacher_f3_occ_diff_count"] != 0:
                all_match = False
    del model
    write_csv(REPORTS_DIR / "sw14a_postprocess_f3_parity.csv", rows)
    return rows, all_match


def phase4_frontcap_parity(sample_ids: list[int]) -> tuple[list[dict[str, Any]], bool]:
    sectors = {name: tensor.cpu().bool() for name, tensor in sw14.sw7.build_sector_masks().items()}
    front_mask = sectors["front"].bool()
    fix_variant = candidate_fix_variant()
    rows: list[dict[str, Any]] = []
    all_match = True
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    for sample_index in sample_ids:
        replay_by_h, _ = run_raw_replay(model, dataset, sample_index)
        for horizon_s in CORE_HORIZONS:
            replay = replay_by_h[horizon_s]
            native = load_npz_torch(raw_dump_path(sample_index, horizon_s, "native_baseline"))
            teacher = load_npz_torch(final_output_path(sample_index, horizon_s))
            agreement = sw14.agreement_map_for_sample(sample_index, horizon_s, candidate()).float()
            replay_raw_occ = occ_mask(replay["semantic"])
            native_occ = occ_mask(native["semantic"].long())
            replay_delta = replay_raw_occ & ~native_occ
            protected = sw14.sw13c_fix.protected_zone_fix(
                candidate().protected_variant,
                replay_raw_occ,
                replay_delta,
                replay["occ_conf"],
                replay["top1_margin"],
                agreement,
                sectors,
                horizon_s,
            )
            low_value_f3 = sw14.sw13c_fix.low_value_score(
                replay_raw_occ,
                protected,
                replay["occ_conf"],
                replay["top1_margin"],
                agreement,
                sectors,
                horizon_s,
                fix_variant.wrong_class_aware,
                fix_variant.front_bias,
            )
            replay_f3, _, _ = sw14.sw13c_fix.apply_pruning_no_gt(
                raw_semantic=replay["semantic"].long(),
                protected_mask=protected.bool(),
                low_value_score_map=low_value_f3.float(),
                native_occ_count=occ_count(native["semantic"].long()),
                raw_occ_count=occ_count(replay["semantic"].long()),
                raw_delta_count=int(replay_delta.sum().item()),
                budget_mode=fix_variant.budget_mode,
                expansion_ratio=fix_variant.expansion_ratio,
                keep_ratio=fix_variant.keep_ratio,
            )
            low_value_frontcap = sw14.sw13c_fix.low_value_score(
                occ_mask(replay_f3),
                protected,
                replay["occ_conf"],
                replay["top1_margin"],
                agreement,
                sectors,
                horizon_s,
                candidate().name.startswith("A10"),
                0.0,
            )
            replay_final, front_pruned, cap_meta = sw14.frontcap50.apply_front_local_cap_no_gt(
                final_semantic_before_cap=replay_f3.long(),
                native_semantic=native["semantic"].long(),
                raw_semantic=replay["semantic"].long(),
                protected_mask=protected.bool(),
                front_mask=front_mask,
                confidence=replay["occ_conf"],
                margin=replay["top1_margin"],
                agreement=agreement,
                low_value_score_map=low_value_frontcap.float(),
                cap_ratio=candidate().cap_ratio,
                cap_mode="front_native_ratio_cap",
            )
            gt_h = replay["gt_h"].long()
            gt0 = replay["gt0"].long()
            replay_eval = build_metric_row(replay_final, gt_h, gt0, native["semantic"].long(), horizon_s)
            teacher_eval = build_metric_row(teacher["final_after_cap"].long(), gt_h, gt0, native["semantic"].long(), horizon_s)
            row = {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "final_occ_count": occ_count(replay_final),
                "teacher_final_occ_count": occ_count(teacher["final_after_cap"].long()),
                "pred_gt_density_delta": float(replay_eval["pred_gt_density_delta"]),
                "teacher_pred_gt_density_delta": float(teacher_eval["pred_gt_density_delta"]),
                "front_sector_false_free_rate_delta": float(replay_eval["front_sector_false_free_rate_delta_vs_native"]),
                "teacher_front_sector_false_free_rate_delta": float(teacher_eval["front_sector_false_free_rate_delta_vs_native"]),
                "future_h4_h6_false_free_rate_delta": float(replay_eval["future_h4_h6_false_free_rate_delta_vs_native"]),
                "teacher_future_h4_h6_false_free_rate_delta": float(teacher_eval["future_h4_h6_false_free_rate_delta_vs_native"]),
                "false_positive_delta": float(replay_eval["false_positive_delta"]),
                "teacher_false_positive_delta": float(teacher_eval["false_positive_delta"]),
                "front_local_density_proxy": sw14.safe_div(
                    float(front_occ_count(replay_final, front_mask)),
                    max(1.0, float(front_occ_count(native["semantic"].long(), front_mask))),
                ),
                "teacher_front_local_density_proxy": sw14.safe_div(
                    float(front_occ_count(teacher["final_after_cap"].long(), front_mask)),
                    max(1.0, float(front_occ_count(native["semantic"].long(), front_mask))),
                ),
                "replay_vs_teacher_occ_diff": occ_diff_ratio(replay_final, teacher["final_after_cap"].long()),
                "replay_vs_teacher_occ_diff_count": occ_diff_count(replay_final, teacher["final_after_cap"].long()),
                "replay_vs_teacher_semantic_abs_diff": semantic_abs_diff(replay_final, teacher["final_after_cap"].long()),
                "frontcap_pruned_count": int(front_pruned.bool().sum().item()),
                "frontcap_meta": cap_meta,
            }
            metric_ok = (
                abs(row["pred_gt_density_delta"] - row["teacher_pred_gt_density_delta"]) <= 1e-6
                and abs(row["front_sector_false_free_rate_delta"] - row["teacher_front_sector_false_free_rate_delta"]) <= 1e-6
                and abs(row["future_h4_h6_false_free_rate_delta"] - row["teacher_future_h4_h6_false_free_rate_delta"]) <= 1e-6
                and abs(row["false_positive_delta"] - row["teacher_false_positive_delta"]) <= 1e-6
                and abs(row["front_local_density_proxy"] - row["teacher_front_local_density_proxy"]) <= 1e-6
            )
            row["reproduce_teacher_flag"] = row["replay_vs_teacher_occ_diff_count"] == 0 and metric_ok
            rows.append(row)
            if row["replay_vs_teacher_occ_diff_count"] != 0:
                all_match = False
    del model
    write_csv(REPORTS_DIR / "sw14a_postprocess_f3_frontcap_parity.csv", rows)
    return rows, all_match


def phase5_metric_schema_audit(sample_ids: list[int]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    gt_schema_ok = True
    for sample_index in sample_ids[:3]:
        for horizon_s in CORE_HORIZONS:
            raw = load_npz_torch(raw_dump_path(sample_index, horizon_s, candidate().base_repair_variant))
            same = bool(torch.equal(raw["gt_h"].long(), raw["gt0"].long()))
            checks.append(
                {
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "gt_h_equals_gt0": same,
                    "gt_h_shape": list(raw["gt_h"].shape),
                    "gt0_shape": list(raw["gt0"].shape),
                }
            )
            if horizon_s in {2, 4, 6} and same:
                gt_schema_ok = False
    payload = {
        "metric_schema_rule": {
            "gt_h": "current horizon GT",
            "gt0": "horizon-0 GT",
            "forbidden_bug": "reusing teacher['gt_h'] as both gt_h and gt0",
        },
        "checks": checks,
        "gt_schema_consistent": gt_schema_ok,
    }
    write_json(REPORTS_DIR / "sw14a_postprocess_metric_schema_audit.json", payload)
    return payload


def build_report(
    manifest: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    raw_phase_decision: str,
    f3_rows: list[dict[str, Any]] | None,
    frontcap_rows: list[dict[str, Any]] | None,
    metric_audit: dict[str, Any] | None,
    runtime_audit: dict[str, Any] | None,
    final_decision: str,
) -> None:
    def mean_of(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return float(sum(float(r[key]) for r in rows) / len(rows))

    lines = [
        "# SW14A Postprocess Parity Report",
        "",
        "## Executive summary",
        f"- final decision: `{final_decision}`.",
        "- no training.",
        "- no SW14B.",
        "- no backbone/head/get_occ change.",
        "- GT is evaluation-only.",
        "",
        "## Teacher pipeline manifest",
        f"- teacher final pipeline: `{manifest['teacher_final_pipeline']}`.",
        f"- teacher final is not: `{manifest['teacher_final_is_not']}`.",
        "",
        "## Raw R8 replay parity",
        f"- raw parity phase decision: `{raw_phase_decision}`.",
        f"- mean replay vs SW13 raw occupancy diff ratio: `{mean_of(raw_rows, 'replay_vs_sw13_raw_occ_diff'):.8f}`.",
        f"- mean replay vs SW13 raw semantic abs diff: `{mean_of(raw_rows, 'replay_vs_sw13_raw_semantic_abs_diff'):.8f}`.",
    ]
    if runtime_audit is not None:
        lines.extend(
            [
                "",
                "## Runtime context audit",
                "- sample100 remains the exact-reference probe when available.",
                "- sample101 is used as the first mismatching probe when available.",
                "- live cache and artifact cache are compared through `sw13a.run_variant_forward`.",
                "- if both still mismatch the stored SW13 raw dump while camera order is verified, the remaining gap is runtime-context divergence rather than cache serialization drift.",
            ]
        )
    if f3_rows is not None:
        lines.extend(
            [
                "",
                "## F3 parity",
                f"- mean replay vs teacher F3 occupancy diff ratio: `{mean_of(f3_rows, 'replay_vs_teacher_f3_occ_diff'):.8f}`.",
                f"- mean replay F3 pred_gt_density_delta: `{mean_of(f3_rows, 'replay_f3_pred_gt_density_delta'):.8f}`.",
            ]
        )
    if frontcap_rows is not None:
        lines.extend(
            [
                "",
                "## FrontCap parity",
                f"- mean replay vs teacher final occupancy diff ratio: `{mean_of(frontcap_rows, 'replay_vs_teacher_occ_diff'):.8f}`.",
                f"- reproduce_teacher_flag rate: `{mean_of(frontcap_rows, 'reproduce_teacher_flag'):.4f}`.",
                f"- mean front_sector_false_free_rate_delta: `{mean_of(frontcap_rows, 'front_sector_false_free_rate_delta'):.8f}`.",
                f"- mean pred_gt_density_delta: `{mean_of(frontcap_rows, 'pred_gt_density_delta'):.8f}`.",
            ]
        )
    if metric_audit is not None:
        lines.extend(
            [
                "",
                "## Metric schema audit",
                f"- gt_schema_consistent: `{metric_audit['gt_schema_consistent']}`.",
                "- eval must use `gt_h` for current horizon and `gt0` for horizon-0 reference.",
            ]
        )
    write_md(REPORTS_DIR / "stage_sw14a_postprocess_parity_report.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_dirs()
    sample_ids = list(range(args.sample_start, args.sample_end + 1))

    ok, missing = check_required_artifacts(sample_ids)
    if not ok:
        manifest = phase1_manifest(sample_ids)
        metric_audit = phase5_metric_schema_audit(sample_ids)
        decision = {
            "decision_type": "P6_TEACHER_ARTIFACT_MISSING",
            "missing_artifacts": missing[:200],
            "missing_artifact_count": len(missing),
            "no_training": True,
            "no_sw14b": True,
        }
        write_json(REPORTS_DIR / "sw14a_postprocess_parity_decision.json", decision)
        build_report(manifest, [], "R2_RAW_REPLAY_DIFFERS_FROM_SW13_R8", None, None, metric_audit, decision["decision_type"])
        return

    manifest = phase1_manifest(sample_ids)
    raw_rows, raw_phase_decision = phase2_raw_parity(sample_ids)
    if raw_phase_decision != "R1_RAW_REPLAY_MATCHES_SW13_R8":
        metric_audit = phase5_metric_schema_audit(sample_ids)
        runtime_audit = phase2_runtime_context_audit(sample_ids)
        decision = {
            "decision_type": "P2_RAW_REPLAY_NOT_MATCH_SW13_R8",
            "teacher_pipeline_manifest": manifest,
            "raw_phase_decision": raw_phase_decision,
            "raw_phase_mean_occ_diff": float(sum(r["replay_vs_sw13_raw_occ_diff"] for r in raw_rows) / max(1, len(raw_rows))),
            "metric_schema_consistent": metric_audit["gt_schema_consistent"],
            "runtime_context_audit": runtime_audit,
            "no_training": True,
            "no_sw14b": True,
            "no_backbone_change": True,
            "no_head_change": True,
            "no_get_occ_change": True,
        }
        write_json(REPORTS_DIR / "sw14a_postprocess_parity_decision.json", decision)
        build_report(manifest, raw_rows, raw_phase_decision, None, None, metric_audit, runtime_audit, decision["decision_type"])
        return

    f3_rows, f3_match = phase3_f3_parity(raw_rows, sample_ids)
    if not f3_match:
        metric_audit = phase5_metric_schema_audit(sample_ids)
        decision = {
            "decision_type": "P3_F3_PARITY_FAILS",
            "teacher_pipeline_manifest": manifest,
            "raw_phase_decision": raw_phase_decision,
            "f3_mean_occ_diff": float(sum(r["replay_vs_teacher_f3_occ_diff"] for r in f3_rows) / max(1, len(f3_rows))),
            "metric_schema_consistent": metric_audit["gt_schema_consistent"],
            "no_training": True,
            "no_sw14b": True,
        }
        write_json(REPORTS_DIR / "sw14a_postprocess_parity_decision.json", decision)
        build_report(manifest, raw_rows, raw_phase_decision, f3_rows, None, metric_audit, None, decision["decision_type"])
        return

    frontcap_rows, frontcap_match = phase4_frontcap_parity(sample_ids)
    metric_audit = phase5_metric_schema_audit(sample_ids)
    if not metric_audit["gt_schema_consistent"]:
        decision_type = "P5_METRIC_SCHEMA_MISMATCH"
    elif not frontcap_match:
        decision_type = "P4_FRONTCAP_PARITY_FAILS"
    else:
        decision_type = "P1_FULL_PIPELINE_REPRODUCES_TEACHER"
    decision = {
        "decision_type": decision_type,
        "teacher_pipeline_manifest": manifest,
        "raw_phase_decision": raw_phase_decision,
        "f3_parity_exact": f3_match,
        "frontcap_parity_exact": frontcap_match,
        "metric_schema_consistent": metric_audit["gt_schema_consistent"],
        "mean_raw_occ_diff": float(sum(r["replay_vs_sw13_raw_occ_diff"] for r in raw_rows) / max(1, len(raw_rows))),
        "mean_f3_occ_diff": float(sum(r["replay_vs_teacher_f3_occ_diff"] for r in f3_rows) / max(1, len(f3_rows))),
        "mean_frontcap_occ_diff": float(sum(r["replay_vs_teacher_occ_diff"] for r in frontcap_rows) / max(1, len(frontcap_rows))),
        "reproduce_teacher_rate": float(sum(float(r["reproduce_teacher_flag"]) for r in frontcap_rows) / max(1, len(frontcap_rows))),
        "no_training": True,
        "no_sw14b": True,
        "no_backbone_change": True,
        "no_head_change": True,
        "no_get_occ_change": True,
    }
    write_json(REPORTS_DIR / "sw14a_postprocess_parity_decision.json", decision)
    build_report(manifest, raw_rows, raw_phase_decision, f3_rows, frontcap_rows, metric_audit, None, decision_type)


if __name__ == "__main__":
    main()
