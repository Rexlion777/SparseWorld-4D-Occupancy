from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
EMPTY_IDX = 17


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SW13C_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"
FRONTCAP_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/run_sw13c_fix_frontcap_main.py"
SW12B_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py"
SW13A_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/run_sw13a_main.py"

sw13c_fix = load_module("sw14b_pp_sw13c_fix", SW13C_FIX_SCRIPT)
frontcap50 = load_module("sw14b_pp_frontcap", FRONTCAP_SCRIPT)
sw12b = load_module("sw14b_pp_sw12b", SW12B_SCRIPT)
sw13a = load_module("sw14b_pp_sw13a", SW13A_SCRIPT)


@dataclass(frozen=True)
class PostprocessCandidate:
    name: str
    perturbation_id: str
    base_repair_variant: str
    expansion_ratio: float | None
    cap_ratio: float | None
    protected_variant: str
    agreement_sources: list[str]
    wrong_class_aware: bool = False
    front_bias: float = 1.0


def normalize(obj: Any) -> Any:
    return sw13c_fix.normalize_export(obj)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def metric_value(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            return float(row[key])
    return float(default)


def attach_sw13_style_deltas(eval_row: dict[str, Any], native_eval: dict[str, Any], horizon_s: int) -> dict[str, Any]:
    out = dict(eval_row)
    out["front_sector_false_free_rate"] = float(eval_row["front_sector_false_free"])
    out["small_object_false_free_rate"] = float(eval_row["small_object_false_free"])
    out["dynamic_object_false_free_rate"] = float(eval_row["dynamic_false_free"])
    out["future_h4_h6_false_free_rate"] = float(eval_row["false_free_rate"]) if int(horizon_s) in {4, 6} else 0.0
    out["false_positive_rate"] = float(eval_row["false_occupied_rate"])
    out["wrong_class_rate"] = float(eval_row["wrong_class_activation"])
    out["front_sector_false_free_rate_delta"] = float(eval_row["front_sector_false_free"]) - float(native_eval["front_sector_false_free"])
    out["future_h4_h6_false_free_rate_delta"] = float(eval_row["false_free_rate"]) - float(native_eval["false_free_rate"]) if int(horizon_s) in {4, 6} else 0.0
    out["false_positive_delta"] = float(eval_row["false_occupied_rate"]) - float(native_eval["false_occupied_rate"])
    out["wrong_class_delta"] = float(eval_row["wrong_class_activation"]) - float(native_eval["wrong_class_activation"])
    return out


def occupancy_confidence_and_margin(dense_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    top2 = torch.topk(dense_scores.float(), k=min(2, dense_scores.shape[-1]), dim=-1).values
    confidence = top2[..., 0]
    if top2.shape[-1] == 1:
        margin = torch.zeros_like(confidence)
    else:
        margin = top2[..., 0] - top2[..., 1]
    return confidence.clamp(0.0, 1.0), margin.clamp(min=0.0)


def build_agreement_map(load_gpu_dump, sample_index: int, horizon_s: int, candidate: PostprocessCandidate) -> torch.Tensor:
    occs: list[torch.Tensor] = []
    for label in candidate.agreement_sources:
        payload = load_gpu_dump(sample_index, horizon_s, candidate.perturbation_id, label)
        occs.append(payload["semantic"].long() != EMPTY_IDX)
    acc = torch.zeros_like(occs[0], dtype=torch.float32)
    for occ in occs:
        acc += occ.float()
    return acc / float(len(occs))


def run_sw14b_full_postprocess(
    raw_semantic: torch.Tensor,
    raw_confidence: torch.Tensor,
    raw_margin: torch.Tensor,
    native_semantic: torch.Tensor,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    candidate: PostprocessCandidate,
    sample_index: int,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    load_gpu_dump,
    agreement_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    raw_semantic = raw_semantic.long().cpu()
    raw_confidence = raw_confidence.float().cpu()
    raw_margin = raw_margin.float().cpu()
    native_semantic = native_semantic.long().cpu()
    gt_h = gt_h.long().cpu()
    gt0 = gt0.long().cpu()
    raw_occ = raw_semantic != EMPTY_IDX
    native_occ = native_semantic != EMPTY_IDX
    raw_delta = raw_occ & ~native_occ
    agreement = agreement_map.float().cpu() if agreement_map is not None else build_agreement_map(load_gpu_dump, sample_index, horizon_s, candidate)
    protected = sw13c_fix.protected_zone_fix(
        candidate.protected_variant,
        raw_occ,
        raw_delta,
        raw_confidence,
        raw_margin,
        agreement,
        sectors,
        horizon_s,
    )
    low_value = sw13c_fix.low_value_score(
        raw_occ,
        protected,
        raw_confidence,
        raw_margin,
        agreement,
        sectors,
        horizon_s,
        candidate.wrong_class_aware,
        candidate.front_bias,
    )
    f3_semantic, f3_pruned, f3_meta = sw13c_fix.apply_pruning_no_gt(
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
    final_semantic, front_pruned, cap_meta = frontcap50.apply_front_local_cap_no_gt(
        final_semantic_before_cap=f3_semantic.long(),
        native_semantic=native_semantic.long(),
        raw_semantic=raw_semantic.long(),
        protected_mask=protected,
        front_mask=sectors["front"].bool(),
        confidence=raw_confidence,
        margin=raw_margin,
        agreement=agreement,
        low_value_score_map=low_value,
        cap_ratio=candidate.cap_ratio,
        cap_mode="disabled" if candidate.cap_ratio is None else "front_native_ratio_cap",
    )
    native_eval = sw12b.build_eval_row(native_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long())
    raw_eval = attach_sw13_style_deltas(
        sw12b.build_eval_row(raw_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    f3_eval = attach_sw13_style_deltas(
        sw12b.build_eval_row(f3_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    final_eval = attach_sw13_style_deltas(
        sw12b.build_eval_row(final_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    protected_zone_preservation_ratio = 1.0 - safe_div(float((protected & f3_pruned).sum().item()), max(1.0, float(protected.sum().item())))
    meta = {
        "raw_occ_count": int(raw_occ.sum().item()),
        "f3_occ_count": int((f3_semantic != EMPTY_IDX).sum().item()),
        "final_occ_count": int((final_semantic != EMPTY_IDX).sum().item()),
        "raw_density_delta": metric_value(raw_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "f3_density_delta": metric_value(f3_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "final_density_delta": metric_value(final_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "raw_front_false_free_delta": metric_value(raw_eval, "front_sector_false_free_rate_delta", "front_sector_false_free_delta", "front_sector_false_free_rate", "front_sector_false_free"),
        "f3_front_false_free_delta": metric_value(f3_eval, "front_sector_false_free_rate_delta", "front_sector_false_free_delta", "front_sector_false_free_rate", "front_sector_false_free"),
        "final_front_false_free_delta": metric_value(final_eval, "front_sector_false_free_rate_delta", "front_sector_false_free_delta", "front_sector_false_free_rate", "front_sector_false_free"),
        "raw_delta_keep_ratio": safe_div(float(((f3_semantic != EMPTY_IDX) & raw_delta).sum().item()), max(1.0, float(raw_delta.sum().item()))),
        "final_native_expansion_ratio": safe_div(float(((final_semantic != EMPTY_IDX).sum().item() - native_occ.sum().item())), max(1.0, float(native_occ.sum().item()))),
        "protected_zone_preservation_ratio": protected_zone_preservation_ratio,
        "protected_pruning_ratio": safe_div(float((f3_pruned & protected).sum().item()), max(1.0, float(f3_pruned.sum().item()))),
        "frontcap_pruned_count": int(front_pruned.sum().item()),
        "front_local_density_proxy_before": float(cap_meta.get("front_local_density_proxy_before", 0.0)),
        "front_local_density_proxy_after": float(cap_meta.get("front_local_density_proxy_after", 0.0)),
        "native_eval": normalize(native_eval),
        "f3_meta": normalize(f3_meta),
        "frontcap_meta": normalize(cap_meta),
        "raw_eval": normalize(raw_eval),
        "f3_eval": normalize(f3_eval),
        "final_eval": normalize(final_eval),
        "no_gt_budget": True,
        "no_gt_front_cap": True,
        "uses_gt_budget": False,
        "uses_gt_frontcap": False,
    }
    return final_semantic.long(), meta
