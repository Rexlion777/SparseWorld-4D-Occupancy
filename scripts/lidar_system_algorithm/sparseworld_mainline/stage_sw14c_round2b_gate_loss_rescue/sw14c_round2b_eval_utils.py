from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


EMPTY_IDX = 17


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


def safe_div(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def finite_mean(values: list[Any]) -> float | None:
    xs = [float(value) for value in values if value not in (None, "", "None") and np.isfinite(float(value))]
    return float(np.mean(xs)) if xs else None


def finite_sum(values: list[Any]) -> float:
    xs = [float(value) for value in values if value not in (None, "", "None") and np.isfinite(float(value))]
    return float(np.sum(xs)) if xs else 0.0


def occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic.long() != EMPTY_IDX


def binary_counts(student_occ: torch.Tensor, teacher_occ: torch.Tensor) -> dict[str, int | float]:
    s = student_occ.bool()
    t = teacher_occ.bool()
    union = int((s | t).sum().item())
    return {
        "occ_diff_count": int((s ^ t).sum().item()),
        "new_occupied_vs_teacher": int((s & ~t).sum().item()),
        "removed_occupied_vs_teacher": int((~s & t).sum().item()),
        "occ_jaccard": float((s & t).sum().item()) / float(union) if union else 1.0,
    }


def summarize_gamma_rows(rows: list[dict[str, Any]], gamma_candidates: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for gamma in gamma_candidates:
        rows_g = [row for row in rows if abs(float(row["gamma"]) - float(gamma)) < 1e-12]
        out.append(
            {
                "gamma": float(gamma),
                "row_count": len(rows_g),
                "sample_count": len({int(row["sample_index"]) for row in rows_g}),
                "front_fn_reduction_over_teacher": finite_mean([row.get("front_fn_reduction_over_teacher") for row in rows_g]),
                "future_h4h6_fn_reduction_over_teacher": finite_mean(
                    [row.get("future_h4h6_fn_reduction_over_teacher") for row in rows_g if int(row.get("horizon_s", -1)) in {4, 6}]
                ),
                "teacher_FP_suppression_improvement": finite_mean([row.get("teacher_FP_suppression_improvement") for row in rows_g]),
                "density_delta_over_teacher": finite_mean([row.get("density_delta_over_teacher") for row in rows_g]),
                "false_positive_delta_over_teacher": finite_mean([row.get("false_positive_delta_over_teacher") for row in rows_g]),
                "front_local_proxy": finite_mean([row.get("front_local_proxy") for row in rows_g]),
                "sample_joint_success_rate": safe_div(sum(bool(row.get("sample_joint_success")) for row in rows_g), len(rows_g)),
                "final_occ_diff_vs_teacher": finite_mean([row.get("final_occ_diff_vs_teacher") for row in rows_g]),
                "safety_pass_rate": safe_div(sum(bool(row.get("safety_pass")) for row in rows_g), len(rows_g)),
                "safety_pass_all": bool(rows_g) and all(bool(row.get("safety_pass")) for row in rows_g),
                "allowed_as_improvement": abs(float(gamma)) > 1e-12,
            }
        )
    return out


def select_gamma(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonzero = [row for row in summary_rows if bool(row["allowed_as_improvement"])]
    safe = [
        row
        for row in nonzero
        if bool(row["safety_pass_all"])
        and float(row.get("density_delta_over_teacher") or 0.0) <= 0.03
        and float(row.get("false_positive_delta_over_teacher") or 0.0) <= 0.01
        and float(row.get("front_local_proxy") or 0.0) <= 1.30
    ]
    safe_improve = [row for row in safe if float(row.get("front_fn_reduction_over_teacher") or 0.0) >= 0.002]
    safe_any = [row for row in safe if float(row.get("front_fn_reduction_over_teacher") or 0.0) > 0.0]
    unsafe_improve = [
        row
        for row in nonzero
        if float(row.get("front_fn_reduction_over_teacher") or 0.0) > 0.0 and row not in safe
    ]
    if safe_improve:
        best = max(safe_improve, key=lambda row: float(row.get("front_fn_reduction_over_teacher") or 0.0))
        decision = "GAMMA2_R1_SAFE_IMPROVEMENT"
    elif safe:
        best = max(safe, key=lambda row: float(row.get("front_fn_reduction_over_teacher") or 0.0))
        decision = "GAMMA2_R2_SAFE_BUT_NO_MEANINGFUL_IMPROVEMENT"
    elif unsafe_improve:
        best = max(unsafe_improve, key=lambda row: float(row.get("front_fn_reduction_over_teacher") or 0.0))
        decision = "GAMMA2_R3_SIGNAL_ONLY_UNSAFE"
    else:
        best = max(nonzero, key=lambda row: float(row.get("front_fn_reduction_over_teacher") or 0.0)) if nonzero else None
        decision = "GAMMA2_R4_NO_SIGNAL"
    return {
        "decision": decision,
        "selected_gamma": None if best is None else float(best["gamma"]),
        "selected_gamma_is_nonzero": bool(best is not None and abs(float(best["gamma"])) > 1e-12),
        "best_summary": best,
        "gamma_zero_counts_as_improvement": False,
        "uses_val_only_for_selection": True,
        "uses_eval_debug_for_selection": False,
        "uses_core100_or_core500_for_selection": False,
    }
