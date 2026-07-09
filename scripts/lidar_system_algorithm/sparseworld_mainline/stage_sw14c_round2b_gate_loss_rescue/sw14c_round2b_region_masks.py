from __future__ import annotations

from typing import Any

import torch


EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]


def safe_div(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic.long() != EMPTY_IDX


def build_teacher_error_masks(
    teacher_h: dict[str, Any],
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    gt_h = teacher_h["gt_h"].long()
    teacher_final = teacher_h["teacher_final_semantic"].long()
    native = teacher_h["native_semantic"].long()
    teacher_raw = teacher_h["teacher_raw_semantic"].long()
    gt_occ = occ_mask(gt_h)
    gt_free = ~gt_occ
    teacher_occ = occ_mask(teacher_final)
    native_occ = occ_mask(native)
    raw_occ = occ_mask(teacher_raw)
    teacher_fn = gt_occ & ~teacher_occ
    teacher_fp = gt_free & teacher_occ
    correct_occ = gt_occ & teacher_occ
    correct_free = gt_free & ~teacher_occ
    front = sectors["front"].bool()
    future = torch.ones_like(front, dtype=torch.bool) if int(horizon_s) in {4, 6} else torch.zeros_like(front, dtype=torch.bool)
    raw_delta = raw_occ & ~native_occ
    low_conf = teacher_h["teacher_confidence"].float() < 0.35
    high_risk_free = correct_free & (front | raw_delta | low_conf)
    recover = teacher_fn & (front | future)
    suppress = teacher_fp | high_risk_free
    preserve = correct_occ | correct_free
    residual_allowed = (teacher_fn | teacher_fp | raw_delta | high_risk_free) & front
    residual_forbidden = correct_free & ~residual_allowed
    return {
        "gt_occ": gt_occ,
        "gt_free": gt_free,
        "teacher_occ": teacher_occ,
        "teacher_FN_region": teacher_fn,
        "teacher_FP_region": teacher_fp,
        "teacher_correct_occ_region": correct_occ,
        "teacher_correct_free_region": correct_free,
        "front_region": front,
        "future_region": future,
        "raw_delta_region": raw_delta,
        "high_risk_free_region": high_risk_free,
        "M_recover": recover,
        "M_suppress": suppress,
        "M_preserve": preserve,
        "M_residual_allowed": residual_allowed,
        "M_residual_forbidden": residual_forbidden,
    }


def mask_stats_row(
    split_name: str,
    sample_index: int,
    horizon_s: int,
    masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    teacher_fn = masks["teacher_FN_region"]
    teacher_fp = masks["teacher_FP_region"]
    correct_occ = masks["teacher_correct_occ_region"]
    correct_free = masks["teacher_correct_free_region"]
    recover = masks["M_recover"]
    suppress = masks["M_suppress"]
    preserve = masks["M_preserve"]
    allowed = masks["M_residual_allowed"]
    front = masks["front_region"]
    future = masks["future_region"]
    return {
        "split": split_name,
        "sample_index": int(sample_index),
        "horizon_s": int(horizon_s),
        "teacher_FN_count": int(teacher_fn.sum().item()),
        "teacher_FP_count": int(teacher_fp.sum().item()),
        "teacher_correct_occ_count": int(correct_occ.sum().item()),
        "teacher_correct_free_count": int(correct_free.sum().item()),
        "M_recover_count": int(recover.sum().item()),
        "M_suppress_count": int(suppress.sum().item()),
        "M_preserve_count": int(preserve.sum().item()),
        "M_residual_allowed_count": int(allowed.sum().item()),
        "samples_with_zero_recover_mask": bool(recover.sum().item() == 0),
        "samples_with_high_FP": bool(teacher_fp.sum().item() >= 1000),
        "front_teacher_FN_count": int((teacher_fn & front).sum().item()),
        "future_teacher_FN_count": int((teacher_fn & future).sum().item()),
        "front_M_recover_count": int((recover & front).sum().item()),
        "future_M_recover_count": int((recover & future).sum().item()),
        "teacher_FN_rate": safe_div(int(teacher_fn.sum().item()), int(masks["gt_occ"].sum().item())),
        "teacher_FP_rate": safe_div(int(teacher_fp.sum().item()), int(masks["gt_free"].sum().item())),
    }


def summarize_mask_rows(rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    samples = sorted({int(row["sample_index"]) for row in rows})
    total_fn = sum(int(row["teacher_FN_count"]) for row in rows)
    total_fp = sum(int(row["teacher_FP_count"]) for row in rows)
    total_recover = sum(int(row["M_recover_count"]) for row in rows)
    total_suppress = sum(int(row["M_suppress_count"]) for row in rows)
    zero_recover_samples = len(
        {
            int(row["sample_index"])
            for row in rows
            if bool(row["samples_with_zero_recover_mask"])
        }
    )
    high_fp_samples = len(
        {
            int(row["sample_index"])
            for row in rows
            if bool(row["samples_with_high_FP"])
        }
    )
    if total_fn + total_fp == 0:
        decision = "MASK_R2_TEACHER_ERROR_TOO_SPARSE"
    elif total_fp > total_fn * 1.5:
        decision = "MASK_R3_FP_DOMINANT_READY"
    elif total_recover > 0 and total_suppress > 0:
        decision = "MASK_R1_READY"
    else:
        decision = "MASK_R4_MASK_BUG"
    return {
        "split": split_name,
        "decision": decision,
        "row_count": len(rows),
        "sample_count": len(samples),
        "teacher_FN_total": int(total_fn),
        "teacher_FP_total": int(total_fp),
        "M_recover_total": int(total_recover),
        "M_suppress_total": int(total_suppress),
        "M_preserve_total": int(sum(int(row["M_preserve_count"]) for row in rows)),
        "M_residual_allowed_total": int(sum(int(row["M_residual_allowed_count"]) for row in rows)),
        "samples_with_zero_recover_mask": int(zero_recover_samples),
        "samples_with_high_FP": int(high_fp_samples),
        "front_teacher_FN_total": int(sum(int(row["front_teacher_FN_count"]) for row in rows)),
        "future_teacher_FN_total": int(sum(int(row["future_teacher_FN_count"]) for row in rows)),
        "teacher_fp_dominant": bool(total_fp > total_fn * 1.5),
    }
