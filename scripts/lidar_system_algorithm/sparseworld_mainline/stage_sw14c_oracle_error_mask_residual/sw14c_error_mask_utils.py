from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]


def safe_div(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic.long() != EMPTY_IDX


def dilate_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.bool().float()[None, None]
    kernel = 2 * int(radius) + 1
    out = F.max_pool3d(x, kernel_size=kernel, stride=1, padding=int(radius))
    return out[0, 0].bool()


def build_oracle_proxy_masks(
    teacher_h: dict[str, Any],
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    *,
    dilation_radius: int = 1,
) -> dict[str, torch.Tensor]:
    gt_h = teacher_h["gt_h"].long()
    teacher_final = teacher_h["teacher_final_semantic"].long()
    teacher_raw = teacher_h["teacher_raw_semantic"].long()
    native = teacher_h["native_semantic"].long()
    confidence = teacher_h["teacher_confidence"].float()
    margin = teacher_h["teacher_margin"].float()
    agreement = teacher_h["agreement"].float()
    gt_occ = occ_mask(gt_h)
    gt_free = ~gt_occ
    final_occ = occ_mask(teacher_final)
    raw_occ = occ_mask(teacher_raw)
    native_occ = occ_mask(native)
    teacher_fn = gt_occ & ~final_occ
    teacher_fp = gt_free & final_occ
    correct_occ = gt_occ & final_occ
    correct_free = gt_free & ~final_occ
    front = sectors["front"].bool()
    future = torch.ones_like(front, dtype=torch.bool) if int(horizon_s) in {4, 6} else torch.zeros_like(front, dtype=torch.bool)
    raw_delta = raw_occ & ~native_occ
    final_removed_raw = raw_occ & ~final_occ
    final_added_raw = final_occ & ~raw_occ
    boundary = dilate_mask(raw_occ ^ final_occ, radius=1) & (front | future | raw_occ | final_occ)
    raw_evidence = (~final_occ) & (raw_occ | (confidence > 0.45) | (margin > 0.05)) & (front | future)
    high_risk_boundary = boundary | (final_occ & ((confidence < 0.40) | (agreement < 0.50) | final_added_raw))
    oracle_strict = teacher_fn | teacher_fp
    oracle_dilated = dilate_mask(oracle_strict, radius=dilation_radius)
    mask_add_strict = teacher_fn
    mask_sup_strict = teacher_fp
    mask_add_dilated = dilate_mask(teacher_fn, radius=dilation_radius) | (raw_evidence & (front | future))
    mask_sup_dilated = dilate_mask(teacher_fp, radius=dilation_radius) | high_risk_boundary
    proxy = (
        (confidence < 0.35)
        | (margin < 0.03)
        | (agreement < 0.45)
        | final_removed_raw
        | (final_occ & (confidence < 0.45))
        | boundary
        | ((front | future) & (confidence < 0.50))
    )
    proxy = proxy & (front | future | raw_occ | final_occ)
    return {
        "gt_occ": gt_occ,
        "gt_free": gt_free,
        "teacher_occ": final_occ,
        "teacher_FN_region": teacher_fn,
        "teacher_FP_region": teacher_fp,
        "teacher_correct_occ_region": correct_occ,
        "teacher_correct_free_region": correct_free,
        "front_region": front,
        "future_region": future,
        "raw_delta_region": raw_delta,
        "raw_evidence_region": raw_evidence,
        "high_risk_boundary_region": high_risk_boundary,
        "oracle_error_mask_strict": oracle_strict,
        "oracle_error_mask_dilated": oracle_dilated,
        "proxy_error_mask_no_gt": proxy,
        "mask_add_strict": mask_add_strict,
        "mask_sup_strict": mask_sup_strict,
        "mask_add_dilated": mask_add_dilated,
        "mask_sup_dilated": mask_sup_dilated,
        "mask_add_proxy": proxy & (~final_occ | final_removed_raw | raw_evidence),
        "mask_sup_proxy": proxy & final_occ,
    }


def mask_stats_row(split: str, sample_index: int, horizon_s: int, masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    strict = masks["oracle_error_mask_strict"]
    dilated = masks["oracle_error_mask_dilated"]
    proxy = masks["proxy_error_mask_no_gt"]
    front = masks["front_region"]
    future = masks["future_region"]
    overlap = proxy & strict
    return {
        "split": split,
        "sample_index": int(sample_index),
        "horizon_s": int(horizon_s),
        "teacher_FN_count": int(masks["teacher_FN_region"].sum().item()),
        "teacher_FP_count": int(masks["teacher_FP_region"].sum().item()),
        "correct_occ_count": int(masks["teacher_correct_occ_region"].sum().item()),
        "correct_free_count": int(masks["teacher_correct_free_region"].sum().item()),
        "oracle_strict_count": int(strict.sum().item()),
        "oracle_dilated_count": int(dilated.sum().item()),
        "proxy_mask_count": int(proxy.sum().item()),
        "overlap_proxy_oracle_strict": int(overlap.sum().item()),
        "proxy_precision_vs_oracle": safe_div(int(overlap.sum().item()), int(proxy.sum().item())),
        "proxy_recall_vs_oracle": safe_div(int(overlap.sum().item()), int(strict.sum().item())),
        "front_oracle_strict_count": int((strict & front).sum().item()),
        "h4h6_oracle_strict_count": int(strict.sum().item() if int(horizon_s) in {4, 6} else 0),
        "front_proxy_count": int((proxy & front).sum().item()),
        "h4h6_proxy_count": int(proxy.sum().item() if int(horizon_s) in {4, 6} else 0),
        "fp_fn_dominant_ratio": safe_div(int(masks["teacher_FP_region"].sum().item()), int(masks["teacher_FN_region"].sum().item())),
        "samples_with_zero_error_mask": bool(strict.sum().item() == 0),
        "samples_with_high_error_mask": bool(strict.sum().item() >= 1000),
    }


def summarize_mask_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_fn = sum(int(row["teacher_FN_count"]) for row in rows)
    total_fp = sum(int(row["teacher_FP_count"]) for row in rows)
    total_strict = sum(int(row["oracle_strict_count"]) for row in rows)
    total_proxy = sum(int(row["proxy_mask_count"]) for row in rows)
    total_overlap = sum(int(row["overlap_proxy_oracle_strict"]) for row in rows)
    samples = sorted({int(row["sample_index"]) for row in rows})
    zero_samples = len({int(row["sample_index"]) for row in rows if bool(row["samples_with_zero_error_mask"])})
    high_samples = len({int(row["sample_index"]) for row in rows if bool(row["samples_with_high_error_mask"])})
    precision = safe_div(total_overlap, total_proxy)
    recall = safe_div(total_overlap, total_strict)
    if total_strict == 0 or zero_samples > len(samples) * 0.5:
        decision = "MASK_O2_ERROR_TOO_SPARSE"
    elif precision > 0.20 and recall > 0.20:
        decision = "MASK_O3_PROXY_HAS_REASONABLE_OVERLAP"
    elif total_strict > 0:
        decision = "MASK_O4_PROXY_POOR_BUT_ORACLE_READY"
    else:
        decision = "MASK_O5_MASK_BUG"
    return {
        "decision": decision,
        "row_count": len(rows),
        "sample_count": len(samples),
        "teacher_FN_total": int(total_fn),
        "teacher_FP_total": int(total_fp),
        "oracle_strict_total": int(total_strict),
        "oracle_dilated_total": int(sum(int(row["oracle_dilated_count"]) for row in rows)),
        "proxy_mask_total": int(total_proxy),
        "proxy_oracle_overlap_total": int(total_overlap),
        "proxy_precision_vs_oracle": precision,
        "proxy_recall_vs_oracle": recall,
        "fp_fn_dominant_ratio": safe_div(total_fp, total_fn),
        "samples_with_zero_error_mask": int(zero_samples),
        "samples_with_high_error_mask": int(high_samples),
    }
