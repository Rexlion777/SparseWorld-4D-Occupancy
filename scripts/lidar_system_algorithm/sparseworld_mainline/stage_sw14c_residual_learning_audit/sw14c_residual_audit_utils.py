from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return normalize(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize(row))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_div(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def finite_mean(values: list[float | int | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def finite_std(values: list[float | int | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.std(xs)) if xs else None


def tensor_abs_stats(tensor: torch.Tensor, prefix: str = "") -> dict[str, float]:
    values_t = tensor.detach().float().abs().reshape(-1).cpu()
    if values_t.numel() == 0:
        return {
            f"{prefix}mean_abs": 0.0,
            f"{prefix}max_abs": 0.0,
            f"{prefix}p50_abs": 0.0,
            f"{prefix}p90_abs": 0.0,
            f"{prefix}p95_abs": 0.0,
            f"{prefix}p99_abs": 0.0,
            f"{prefix}std": 0.0,
            f"{prefix}nonzero_ratio": 0.0,
        }
    values = values_t.numpy()
    quantiles = np.percentile(values, [50.0, 90.0, 95.0, 99.0])
    return {
        f"{prefix}mean_abs": float(values.mean()),
        f"{prefix}max_abs": float(values.max()),
        f"{prefix}p50_abs": float(quantiles[0]),
        f"{prefix}p90_abs": float(quantiles[1]),
        f"{prefix}p95_abs": float(quantiles[2]),
        f"{prefix}p99_abs": float(quantiles[3]),
        f"{prefix}std": float(values.std()),
        f"{prefix}nonzero_ratio": float((values > 0).mean()),
    }


def tensor_value_stats(tensor: torch.Tensor, prefix: str = "") -> dict[str, float]:
    values_t = tensor.detach().float().reshape(-1).cpu()
    if values_t.numel() == 0:
        return {
            f"{prefix}mean": 0.0,
            f"{prefix}max": 0.0,
            f"{prefix}p50": 0.0,
            f"{prefix}p90": 0.0,
            f"{prefix}p95": 0.0,
            f"{prefix}p99": 0.0,
            f"{prefix}nonzero_ratio": 0.0,
            f"{prefix}saturated_low_ratio": 0.0,
            f"{prefix}saturated_high_ratio": 0.0,
        }
    values = values_t.numpy()
    quantiles = np.percentile(values, [50.0, 90.0, 95.0, 99.0])
    return {
        f"{prefix}mean": float(values.mean()),
        f"{prefix}max": float(values.max()),
        f"{prefix}p50": float(quantiles[0]),
        f"{prefix}p90": float(quantiles[1]),
        f"{prefix}p95": float(quantiles[2]),
        f"{prefix}p99": float(quantiles[3]),
        f"{prefix}nonzero_ratio": float((values > 0).mean()),
        f"{prefix}saturated_low_ratio": float((values < 0.01).mean()),
        f"{prefix}saturated_high_ratio": float((values > 0.90).mean()),
    }


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic.long() != EMPTY_IDX


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.bool()
    bb = b.bool()
    union = (aa | bb).sum().item()
    if union == 0:
        return 1.0
    return float((aa & bb).sum().item()) / float(union)


def binary_diff_counts(student_occ: torch.Tensor, teacher_occ: torch.Tensor) -> dict[str, int | float]:
    s = student_occ.bool()
    t = teacher_occ.bool()
    return {
        "occ_diff_count": int((s ^ t).sum().item()),
        "occ_jaccard": jaccard(s, t),
        "new_occupied_vs_teacher": int((s & ~t).sum().item()),
        "removed_occupied_vs_teacher": int((~s & t).sum().item()),
    }


def metric_value(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in (None, "", "None"):
            return float(row[key])
    return float(default)


def run_postprocess_with_stages(
    base_module: Any,
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
    raw_semantic = raw_semantic.long().detach().cpu()
    raw_confidence = raw_confidence.float().detach().cpu()
    raw_margin = raw_margin.float().detach().cpu()
    native_semantic = native_semantic.long().detach().cpu()
    gt_h = gt_h.long().detach().cpu()
    gt0 = gt0.long().detach().cpu()
    agreement = agreement_map.float().detach().cpu()
    raw_occ = raw_semantic != EMPTY_IDX
    native_occ = native_semantic != EMPTY_IDX
    raw_delta = raw_occ & ~native_occ
    protected = base_module.sw13c_fix.protected_zone_fix(
        candidate.protected_variant,
        raw_occ,
        raw_delta,
        raw_confidence,
        raw_margin,
        agreement,
        sectors,
        horizon_s,
    )
    low_value = base_module.sw13c_fix.low_value_score(
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
    f3_semantic, f3_pruned, f3_meta = base_module.sw13c_fix.apply_pruning_no_gt(
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
    final_semantic, front_pruned, cap_meta = base_module.frontcap50.apply_front_local_cap_no_gt(
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
    native_eval = base_module.sw12b.build_eval_row(
        native_semantic.long(),
        gt_h,
        gt0,
        candidate.perturbation_id,
        horizon_s,
        sectors,
        baseline_pred=native_semantic.long(),
    )
    raw_eval = base_module.attach_sw13_style_deltas(
        base_module.sw12b.build_eval_row(raw_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    f3_eval = base_module.attach_sw13_style_deltas(
        base_module.sw12b.build_eval_row(f3_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
    final_eval = base_module.attach_sw13_style_deltas(
        base_module.sw12b.build_eval_row(final_semantic.long(), gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long()),
        native_eval,
        horizon_s,
    )
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
        "raw_false_positive_delta": metric_value(raw_eval, "false_positive_delta"),
        "f3_false_positive_delta": metric_value(f3_eval, "false_positive_delta"),
        "final_false_positive_delta": metric_value(final_eval, "false_positive_delta"),
        "raw_delta_keep_ratio": safe_div(float(((f3_semantic != EMPTY_IDX) & raw_delta).sum().item()), max(1.0, float(raw_delta.sum().item()))),
        "final_native_expansion_ratio": safe_div(float(((final_semantic != EMPTY_IDX).sum().item() - native_occ.sum().item())), max(1.0, float(native_occ.sum().item()))),
        "protected_zone_preservation_ratio": 1.0 - safe_div(float((protected & f3_pruned).sum().item()), max(1.0, float(protected.sum().item()))),
        "protected_pruning_ratio": safe_div(float((f3_pruned & protected).sum().item()), max(1.0, float(f3_pruned.sum().item()))),
        "frontcap_pruned_count": int(front_pruned.sum().item()),
        "front_local_density_proxy_before": float(cap_meta.get("front_local_density_proxy_before", 0.0)),
        "front_local_density_proxy_after": float(cap_meta.get("front_local_density_proxy_after", 0.0)),
        "native_eval": normalize(native_eval),
        "raw_eval": normalize(raw_eval),
        "f3_eval": normalize(f3_eval),
        "final_eval": normalize(final_eval),
        "f3_meta": normalize(f3_meta),
        "frontcap_meta": normalize(cap_meta),
        "no_gt_budget": True,
        "no_gt_front_cap": True,
    }
    return {
        "raw_semantic": raw_semantic.long(),
        "f3_semantic": f3_semantic.long(),
        "final_semantic": final_semantic.long(),
        "protected_mask": protected.bool(),
        "raw_delta_mask": raw_delta.bool(),
        "f3_pruned_mask": f3_pruned.bool(),
        "front_pruned_mask": front_pruned.bool(),
        "low_value_score": low_value.float(),
        "meta": meta,
        "sample_index": int(sample_index),
        "horizon_s": int(horizon_s),
    }
