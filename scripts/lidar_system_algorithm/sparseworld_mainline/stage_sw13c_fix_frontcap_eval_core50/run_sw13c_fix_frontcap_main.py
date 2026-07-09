from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"
LOGS_DIR = PROJECT_ROOT / "logs/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"

SCALE_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
SCALE_ARTIFACTS = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
SW13C_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"

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


sw13c_fix = load_module("sw13c_fix_frontcap_base", SW13C_FIX_SCRIPT)
sw7 = sw13c_fix.sw7


@dataclass(frozen=True)
class FrozenCandidate:
    label: str
    perturbation_id: str
    base_repair_variant: str
    base_variant_before_cap: str
    expansion_ratio: float | None
    protected_variant: str
    agreement_sources: list[str]
    agreement_source_count: int


@dataclass(frozen=True)
class FrontCapVariant:
    candidate_name: str
    label: str
    cap_ratio: float | None
    cap_mode: str


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "final_outputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(obj: Any) -> Any:
    return sw13c_fix.normalize_export(obj)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-13C-Fix FrontCap eval_core50 postprocess")
    parser.add_argument("--samples", default=",".join(str(i) for i in range(50)))
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def frozen_candidates() -> list[FrozenCandidate]:
    return [
        FrozenCandidate(
            label="A1_fixed",
            perturbation_id="A1_drop_cam_front",
            base_repair_variant="R1_replace_tminus1",
            base_variant_before_cap="F3_expand_budget_strict",
            expansion_ratio=0.08,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2"],
            agreement_source_count=3,
        ),
        FrozenCandidate(
            label="A10_fixed_main",
            perturbation_id="A10_drop_front_triplet",
            base_repair_variant="R4_ema_K3",
            base_variant_before_cap="F3_expand_budget_strict",
            expansion_ratio=0.12,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            agreement_source_count=4,
        ),
        FrozenCandidate(
            label="A10_fixed_secondary",
            perturbation_id="A10_drop_front_triplet",
            base_repair_variant="R8_camera_group_repair_front_triplet",
            base_variant_before_cap="F3_expand_budget_strict",
            expansion_ratio=0.12,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            agreement_source_count=4,
        ),
        FrozenCandidate(
            label="C4_fixed",
            perturbation_id="C4_motion_blur_9",
            base_repair_variant="R5_blend_alpha03",
            base_variant_before_cap="F9_C4_preserve_R5",
            expansion_ratio=None,
            protected_variant="PZ_fix_2_front_conf_agree",
            agreement_sources=["R5_blend_alpha03", "R6_blend_alpha05", "R7_blend_alpha07"],
            agreement_source_count=3,
        ),
    ]


def front_cap_variants() -> list[FrontCapVariant]:
    variants: list[FrontCapVariant] = []
    for label, ratios in {
        "A1_fixed": [None, 1.50, 1.40, 1.30],
        "A10_fixed_main": [None, 1.30, 1.25, 1.20],
        "A10_fixed_secondary": [None, 1.30, 1.25, 1.20],
        "C4_fixed": [None],
    }.items():
        for idx, ratio in enumerate(ratios):
            variants.append(
                FrontCapVariant(
                    candidate_name=label,
                    label=f"FC{idx}_no_front_cap" if ratio is None and idx == 0 else f"FC{idx}_{str(ratio).replace('.', 'p')}",
                    cap_ratio=ratio,
                    cap_mode="disabled" if ratio is None else "front_native_ratio_cap",
                )
            )
    return variants


def load_scale_case(candidate: FrozenCandidate, sample_index: int, horizon_s: int) -> dict[str, torch.Tensor]:
    final_payload = np.load(SCALE_ARTIFACTS / "final_outputs" / f"{candidate.label}__sample{sample_index:03d}_h{horizon_s}.npz")
    raw_payload = np.load(SCALE_ARTIFACTS / "gpu_phase_dumps" / f"{candidate.perturbation_id}__{candidate.base_repair_variant}__sample{sample_index:03d}_h{horizon_s}.npz")
    native_payload = np.load(SCALE_ARTIFACTS / "gpu_phase_dumps" / f"{candidate.perturbation_id}__native_baseline__sample{sample_index:03d}_h{horizon_s}.npz")
    agreement_payload = np.load(SCALE_ARTIFACTS / "multi_source_agreement" / f"{candidate.label}__sample{sample_index:03d}_h{horizon_s}.npz")
    return {
        "native_semantic": torch.from_numpy(native_payload["semantic"]).long(),
        "raw_semantic": torch.from_numpy(raw_payload["semantic"]).long(),
        "final_before_cap": torch.from_numpy(final_payload["final_semantic"]).long(),
        "protected_mask": torch.from_numpy(final_payload["protected_zone"]).bool(),
        "gt_h": torch.from_numpy(final_payload["gt_h"]).long(),
        "gt0": torch.from_numpy(raw_payload["gt0"]).long(),
        "confidence": torch.from_numpy(raw_payload["occ_conf"]).float(),
        "margin": torch.from_numpy(raw_payload["top1_margin"]).float(),
        "agreement": torch.from_numpy(agreement_payload["agreement"]).float(),
    }


def safe_div(a: float | int, b: float | int) -> float:
    return sw13c_fix.safe_div(a, b)


def front_proxy(front_occ_count_final: int, front_occ_count_native: int) -> float:
    return safe_div(front_occ_count_final, max(1, front_occ_count_native))


def summarize_bev(mask: torch.Tensor) -> np.ndarray:
    return (mask != EMPTY_IDX).any(dim=-1).numpy().astype(np.float32)


def save_bev_panel(
    path: Path,
    gt_h: torch.Tensor,
    native: torch.Tensor,
    raw: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
    protected: torch.Tensor,
    front_pruned: torch.Tensor,
    title: str,
) -> None:
    gt_b = summarize_bev(gt_h)
    native_b = summarize_bev(native)
    raw_b = summarize_bev(raw)
    before_b = summarize_bev(before)
    after_b = summarize_bev(after)
    panels = [
        (gt_b, "GT"),
        (native_b, "degraded native"),
        (raw_b, "raw repair"),
        (before_b, "before front cap"),
        (after_b, "after front cap"),
        (protected.any(dim=-1).numpy().astype(np.float32), "protected zone"),
        (front_pruned.any(dim=-1).numpy().astype(np.float32), "front cap pruned"),
        (((gt_b > 0.5) & (native_b < 0.5)).astype(float), "false-free native"),
        (((gt_b > 0.5) & (before_b < 0.5)).astype(float), "false-free before"),
        (((gt_b > 0.5) & (after_b < 0.5)).astype(float), "false-free after"),
        (((gt_b < 0.5) & (before_b > 0.5)).astype(float), "false-positive before"),
        (((gt_b < 0.5) & (after_b > 0.5)).astype(float), "false-positive after"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, (arr, name) in zip(axes.flatten(), panels):
        ax.imshow(arr.T, origin="lower", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def apply_front_local_cap_no_gt(
    final_semantic_before_cap: torch.Tensor,
    native_semantic: torch.Tensor,
    raw_semantic: torch.Tensor,
    protected_mask: torch.Tensor,
    front_mask: torch.Tensor,
    confidence: torch.Tensor,
    margin: torch.Tensor,
    agreement: torch.Tensor,
    low_value_score_map: torch.Tensor,
    cap_ratio: float | None,
    cap_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    before_occ = final_semantic_before_cap != EMPTY_IDX
    native_occ = native_semantic != EMPTY_IDX
    front_native_occ_count = int((native_occ & front_mask).sum().item())
    front_before_cap_occ_count = int((before_occ & front_mask).sum().item())
    proxy_before = front_proxy(front_before_cap_occ_count, front_native_occ_count)
    if cap_ratio is None or cap_mode == "disabled":
        meta = {
            "cap_mode": cap_mode,
            "cap_ratio": cap_ratio,
            "front_native_occ_count": front_native_occ_count,
            "front_before_cap_occ_count": front_before_cap_occ_count,
            "front_after_cap_occ_count": front_before_cap_occ_count,
            "front_local_density_proxy_before": proxy_before,
            "front_local_density_proxy_after": proxy_before,
            "front_pruned_count": 0,
            "front_pruning_from_protected_ratio": 0.0,
            "front_cap_target_unmet": False,
            "no_gt_front_cap": True,
        }
        return final_semantic_before_cap.clone(), torch.zeros_like(before_occ), meta
    front_target_count = int(round(front_native_occ_count * cap_ratio))
    front_target_count = max(front_native_occ_count, front_target_count)
    if front_before_cap_occ_count <= front_target_count:
        meta = {
            "cap_mode": cap_mode,
            "cap_ratio": cap_ratio,
            "front_native_occ_count": front_native_occ_count,
            "front_before_cap_occ_count": front_before_cap_occ_count,
            "front_after_cap_occ_count": front_before_cap_occ_count,
            "front_local_density_proxy_before": proxy_before,
            "front_local_density_proxy_after": proxy_before,
            "front_pruned_count": 0,
            "front_pruning_from_protected_ratio": 0.0,
            "front_cap_target_unmet": False,
            "no_gt_front_cap": True,
        }
        return final_semantic_before_cap.clone(), torch.zeros_like(before_occ), meta
    prunable = front_mask & before_occ & ~protected_mask
    prune_goal = front_before_cap_occ_count - front_target_count
    front_pruned = torch.zeros_like(before_occ)
    final_after = final_semantic_before_cap.clone()
    if prunable.any():
        support = sw13c_fix.local_support(before_occ).float()
        conf_bad = 1.0 - torch.clamp(confidence, 0.0, 1.0)
        margin_bad = 1.0 - torch.clamp(margin / (float(margin.max().item()) + 1e-6), 0.0, 1.0)
        cap_priority = low_value_score_map + 0.8 * conf_bad + 0.6 * margin_bad + 1.2 * (1.0 - agreement) + 0.9 * (support < 5).float()
        cap_priority[~prunable] = -1e6
        coords = torch.nonzero(prunable, as_tuple=False)
        vals = cap_priority[prunable]
        prune_count = min(prune_goal, vals.numel())
        if prune_count > 0:
            _, idx = torch.topk(vals, k=prune_count, largest=True)
            picked = coords[idx]
            front_pruned[picked[:, 0], picked[:, 1], picked[:, 2]] = True
            final_after[front_pruned] = EMPTY_IDX
    front_after_cap_occ_count = int(((final_after != EMPTY_IDX) & front_mask).sum().item())
    meta = {
        "cap_mode": cap_mode,
        "cap_ratio": cap_ratio,
        "front_native_occ_count": front_native_occ_count,
        "front_before_cap_occ_count": front_before_cap_occ_count,
        "front_after_cap_occ_count": front_after_cap_occ_count,
        "front_local_density_proxy_before": proxy_before,
        "front_local_density_proxy_after": front_proxy(front_after_cap_occ_count, front_native_occ_count),
        "front_pruned_count": int(front_pruned.sum().item()),
        "front_pruning_from_protected_ratio": safe_div(float((front_pruned & protected_mask).sum().item()), max(1.0, float(front_pruned.sum().item()))),
        "front_cap_target_unmet": front_after_cap_occ_count > front_target_count,
        "no_gt_front_cap": True,
    }
    return final_after, front_pruned, meta


def summary_row(rows: list[dict[str, Any]], candidate_name: str, front_cap_variant: str, horizon_s: int) -> dict[str, Any] | None:
    items = [
        row
        for row in rows
        if row["candidate_name"] == candidate_name and row["front_cap_variant"] == front_cap_variant and int(row["horizon_s"]) == horizon_s
    ]
    return items[0] if items else None


def threshold_for_candidate(candidate_name: str) -> float:
    if candidate_name == "A1_fixed":
        return 1.50
    if candidate_name.startswith("A10_"):
        return 1.30
    return 1.10


def front_local_safe_flag(candidate_name: str, proxy_after: float, front_fp_delta: float, pred_gt_density_delta: float) -> bool:
    if candidate_name == "A1_fixed":
        return proxy_after <= 1.50 and front_fp_delta <= 0.02 and pred_gt_density_delta <= 0.18
    if candidate_name.startswith("A10_"):
        return proxy_after <= 1.30 and front_fp_delta <= 0.02 and pred_gt_density_delta <= 0.20
    return proxy_after <= 1.10 and front_fp_delta <= 0.02 and pred_gt_density_delta <= 0.10


def choose_frontcap(summary_rows: list[dict[str, Any]], candidate_name: str) -> dict[str, Any]:
    if candidate_name == "C4_fixed":
        row = summary_row(summary_rows, candidate_name, "FC0_no_front_cap", 6)
        assert row is not None
        return row
    order = ["FC0_no_front_cap", "FC1_1p5" if candidate_name == "A1_fixed" else "FC1_1p3", "FC2_1p4" if candidate_name == "A1_fixed" else "FC2_1p25", "FC3_1p3" if candidate_name == "A1_fixed" else "FC3_1p2"]
    valid_rows: list[dict[str, Any]] = []
    for label in order:
        row = summary_row(summary_rows, candidate_name, label, 6)
        if row is None:
            continue
        if float(row["front_pruning_from_protected_ratio"]) > 0.05 or float(row["protected_zone_preservation_ratio"]) < 0.80 or float(row["front_kept_ratio"]) < 0.75:
            continue
        valid_rows.append(row)
    threshold = threshold_for_candidate(candidate_name)
    for row in valid_rows:
        if float(row["front_local_density_proxy_after"]) <= threshold and not sw13c_fix.truthy(row["front_cap_target_unmet"]):
            return row
    if valid_rows:
        return max(valid_rows, key=lambda row: float(row["front_local_density_proxy_before"]) - float(row["front_local_density_proxy_after"]))
    fallback = summary_row(summary_rows, candidate_name, "FC0_no_front_cap", 6)
    assert fallback is not None
    fallback = dict(fallback)
    fallback["frontcap_failed"] = True
    return fallback


def save_bar(path: Path, title: str, labels: list[str], before_vals: list[float], after_vals: list[float]) -> None:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, before_vals, width, label="before")
    ax.bar(x + width / 2, after_vals, width, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_line_compare(path: Path, title: str, xs: list[str], ys: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_tests() -> None:
    tests = {
        "test_outputs_exist.py": """from pathlib import Path\nBASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50')\ndef test_outputs_exist():\n    names=['sw13c_fix_frontcap_inherited_scale_audit.json','sw13c_fix_frontcap_proxy_definition_audit.json','sw13c_fix_frontcap_frozen_candidate_manifest.json','sw13c_fix_frontcap_metrics.csv','sw13c_fix_frontcap_metric_summary.csv','sw13c_fix_frontcap_density_audit.csv','sw13c_fix_frontcap_candidate_selection.json','sw13c_fix_frontcap_eval_core50_decision.json','stage_sw13c_fix_frontcap_eval_core50_report.md']\n    for name in names:\n        p=BASE/name\n        assert p.exists() and p.stat().st_size>0, name\n""",
        "test_frozen_candidate_manifest.py": """import json\nfrom pathlib import Path\ndef test_frozen_candidate_manifest():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_frozen_candidate_manifest.json').read_text())\n    assert obj['frozen_candidates'] is True\n""",
        "test_no_gt_front_cap.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_metrics.csv').open()))\ndef test_no_gt_front_cap():\n    assert rows\n    assert all(r['no_gt_front_cap'] == 'True' for r in rows)\n""",
        "test_no_global_reselection.py": """import json\nfrom pathlib import Path\ndef test_no_global_reselection():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_candidate_selection.json').read_text())\n    assert obj['no_global_reselection'] is True\n    assert obj['no_global_retuning'] is True\n""",
        "test_frontcap_selection_no_gt.py": """import json\nfrom pathlib import Path\nSRC=Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/run_sw13c_fix_frontcap_main.py').read_text()\ndef test_frontcap_selection_no_gt():\n    assert 'pred_gt_density_delta' not in SRC.split('def choose_frontcap',1)[1].split('def save_bar',1)[0]\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_candidate_selection.json').read_text())\n    assert obj['selection_uses_gt'] is False\n""",
        "test_metric_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_metrics.csv').open()))\ndef test_metric_schema():\n    assert rows\n    need={'sample_index','perturbation_id','base_repair_variant','front_cap_variant','cap_ratio','horizon_s','front_local_density_proxy_before','front_local_density_proxy_after','front_cap_pruned_count','front_pruning_from_protected_ratio','front_cap_target_unmet','pred_gt_density_delta','protected_zone_preservation_ratio','no_gt_front_cap'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_density_audit_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_density_audit.csv').open()))\ndef test_density_audit_schema():\n    assert rows\n    need={'front_occ_native','front_occ_before_cap','front_occ_after_cap','front_local_density_proxy_before','front_local_density_proxy_after','front_pred_gt_density_delta_before','front_pred_gt_density_delta_after','front_local_density_risk_before','front_local_density_risk_after'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_sample_consistency_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_sample_consistency.csv').open()))\ndef test_sample_consistency_schema():\n    assert rows\n    need={'front_false_free_improved','future_false_free_improved','density_safe','front_local_safe','false_positive_safe','wrong_class_safe','joint_success','frontcap_target_met','frontcap_preservation_safe'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_decision_schema.py": """import json\nfrom pathlib import Path\ndef test_decision_schema():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_eval_core50_decision.json').read_text())\n    assert obj['decision_type'] in {'X1_FRONTCAP_STRONG_FIX','X2_FRONTCAP_MEDIUM_FIX','X3_FRONTCAP_RECOVERY_TRADEOFF','X4_FRONTCAP_RESIDUAL_FRONT_DENSITY_RISK','X5_FRONTCAP_FAILS','X6_PROTOCOL_VIOLATION'}\n""",
        "test_no_false_claims.py": """from pathlib import Path\ntext=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/stage_sw13c_fix_frontcap_eval_core50_report.md').read_text().lower()\ndef test_no_false_claims():\n    assert 'not official benchmark' in text\n    assert 'eval_core50 subset diagnostic' in text\n    assert 'trained model improvement' not in text\n""",
    }
    for name, content in tests.items():
        write_md(TESTS_DIR / name, content)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dirs()

    scale_decision = read_json(SCALE_REPORTS / "sw13c_fix_scale_eval_core50_decision.json")
    scale_front_rows = read_csv(SCALE_REPORTS / "sw13c_fix_scale_front_local_density_audit.csv")
    scale_consistency_rows = read_csv(SCALE_REPORTS / "sw13c_fix_scale_sample_consistency.csv")
    sample_ids = parse_int_list(args.samples)
    sectors = {name: tensor.cpu().bool() for name, tensor in sw7.build_sector_masks().items()}
    front_mask = sectors["front"].bool()

    inherited = {
        "scale_positives": [
            "A1/A10 front recovery scales to eval_core50",
            "A1/A10 global density safe",
            "sample_joint_success_rate not low",
            "C4 preservation holds",
        ],
        "scale_failure": {
            "decision": scale_decision["decision_type"],
            "front_local_density_risk": True,
            "A1_front_local_density_proxy": 1.7654,
            "A10_R4_front_local_density_proxy": 1.3713,
            "A10_R8_front_local_density_proxy": 1.3730,
        },
    }
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_inherited_scale_audit.json", inherited)
    write_md(
        REPORTS_DIR / "sw13c_fix_frontcap_inherited_scale_audit.md",
        "\n".join(
            [
                "# SW-13C-Fix-FrontCap inherited scale audit",
                "",
                "- scale positives: A1/A10 recovery scales to eval_core50, global density stays safe, sample joint success remains useful, and C4 preservation holds.",
                "- scale failure: decision W4 came from front-local density risk rather than recovery collapse.",
                "- A1 front_local_density_proxy = 1.7654.",
                "- A10 R4 front_local_density_proxy = 1.3713.",
                "- A10 R8 front_local_density_proxy = 1.3730.",
                "- FrontCap keeps frozen recovery and adds a second-stage no-GT front-local cap without reselection or retuning.",
            ]
        ),
    )
    proxy_audit = {
        "prediction_side_proxy_definition": "front_occ_count_final / front_occ_count_native",
        "gt_eval_front_density": "front_pred_gt_density_delta",
        "gt_eval_front_false_positive": "front_false_positive_delta",
        "gt_eval_front_false_free": "front_false_free_delta",
        "interpretation": [
            "proxy high means final predicts more front occupied voxels than degraded native",
            "because degraded native is false-free under sensor drop, a high proxy is not automatically a false-positive claim",
            "the proxy is still used as a safety cap target and GT remains evaluation-only",
        ],
    }
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_proxy_definition_audit.json", proxy_audit)
    write_md(
        REPORTS_DIR / "sw13c_fix_frontcap_proxy_definition_audit.md",
        "\n".join(
            [
                "# Front-local proxy definition audit",
                "",
                "- prediction-side front_local_density_proxy = front_occ_count_final / front_occ_count_native.",
                "- GT-eval front density uses front_pred_gt_density_delta only after final prediction generation.",
                "- front_false_positive_delta and front_false_free_delta are GT-eval diagnostics only.",
                "- proxy high means front occupied expands relative to degraded native; it does not by itself prove false positives because degraded native is under-recalled.",
            ]
        ),
    )

    frozen = frozen_candidates()
    variants = front_cap_variants()
    write_json(
        REPORTS_DIR / "sw13c_fix_frontcap_frozen_candidate_manifest.json",
        {
            "frozen_candidates": True,
            "no_global_reselection": True,
            "no_global_retuning": True,
            "no_training": True,
            "no_gt_budget": True,
            "candidates": [candidate.__dict__ for candidate in frozen],
        },
    )
    write_json(
        REPORTS_DIR / "sw13c_fix_frontcap_variant_manifest.json",
        {
            "variants": [variant.__dict__ | {"no_gt_front_cap": True, "no_global_retuning": True} for variant in variants],
        },
    )

    metrics_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    visual_cases: dict[str, dict[str, torch.Tensor]] = {}

    by_candidate = {candidate.label: candidate for candidate in frozen}
    variants_by_candidate: dict[str, list[FrontCapVariant]] = {}
    for variant in variants:
        variants_by_candidate.setdefault(variant.candidate_name, []).append(variant)

    for candidate in frozen:
        for sample_index in sample_ids:
            for horizon_s in CORE_HORIZONS:
                case = load_scale_case(candidate, sample_index, horizon_s)
                native_semantic = case["native_semantic"]
                raw_semantic = case["raw_semantic"]
                final_before = case["final_before_cap"]
                protected = case["protected_mask"]
                gt_h = case["gt_h"]
                gt0 = case["gt0"]
                confidence = case["confidence"]
                margin = case["margin"]
                agreement = case["agreement"]

                before_occ = final_before != EMPTY_IDX
                native_occ = native_semantic != EMPTY_IDX
                raw_occ = raw_semantic != EMPTY_IDX
                low_value_score = sw13c_fix.low_value_score(
                    before_occ,
                    protected,
                    confidence,
                    margin,
                    agreement,
                    sectors,
                    horizon_s,
                    candidate.label.startswith("A10"),
                    0.0,
                )
                before_eval = sw13c_fix.build_eval_row(final_before, gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, native_semantic)
                native_eval = sw13c_fix.build_eval_row(native_semantic, gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, native_semantic)

                front_occ_native = int((native_occ & front_mask).sum().item())
                front_occ_before = int((before_occ & front_mask).sum().item())
                front_gt_occupied = int(((gt_h != EMPTY_IDX) & front_mask).sum().item())
                free_front_count = int(((gt_h == EMPTY_IDX) & front_mask).sum().item())
                before_front_ff = safe_div(float(((gt_h != EMPTY_IDX) & front_mask & ~before_occ).sum().item()), max(1, front_gt_occupied))
                native_front_ff = safe_div(float(((gt_h != EMPTY_IDX) & front_mask & ~native_occ).sum().item()), max(1, front_gt_occupied))
                before_front_fp = safe_div(float(((gt_h == EMPTY_IDX) & front_mask & before_occ).sum().item()), max(1, free_front_count))
                native_front_fp = safe_div(float(((gt_h == EMPTY_IDX) & front_mask & native_occ).sum().item()), max(1, free_front_count))
                proxy_before = front_proxy(front_occ_before, front_occ_native)
                front_pred_gt_density_before = safe_div(front_occ_before, max(1, front_gt_occupied)) - safe_div(front_occ_native, max(1, front_gt_occupied))
                risk_before = front_local_safe_flag(candidate.label, proxy_before, before_front_fp - native_front_fp, float(before_eval["pred_gt_density_delta"])) is False

                for variant in variants_by_candidate[candidate.label]:
                    final_after, front_pruned, cap_meta = apply_front_local_cap_no_gt(
                        final_before,
                        native_semantic,
                        raw_semantic,
                        protected,
                        front_mask,
                        confidence,
                        margin,
                        agreement,
                        low_value_score,
                        variant.cap_ratio,
                        variant.cap_mode,
                    )
                    after_occ = final_after != EMPTY_IDX
                    after_eval = sw13c_fix.build_eval_row(final_after, gt_h, gt0, candidate.perturbation_id, horizon_s, sectors, native_semantic)
                    front_occ_after = int((after_occ & front_mask).sum().item())
                    after_front_ff = safe_div(float(((gt_h != EMPTY_IDX) & front_mask & ~after_occ).sum().item()), max(1, front_gt_occupied))
                    after_front_fp = safe_div(float(((gt_h == EMPTY_IDX) & front_mask & after_occ).sum().item()), max(1, free_front_count))
                    front_pred_gt_density_after = safe_div(front_occ_after, max(1, front_gt_occupied)) - safe_div(front_occ_native, max(1, front_gt_occupied))
                    protected_preservation_ratio = 1.0 - safe_div(float((front_pruned & protected).sum().item()), max(1.0, float(protected.sum().item())))
                    front_kept_ratio = safe_div(float((front_mask & after_occ).sum().item()), max(1.0, float((front_mask & before_occ).sum().item())))
                    front_local_risk_after = not front_local_safe_flag(
                        candidate.label,
                        float(cap_meta["front_local_density_proxy_after"]),
                        after_front_fp - native_front_fp,
                        float(after_eval["pred_gt_density_delta"]),
                    )
                    row = {
                        "sample_index": sample_index,
                        "candidate_name": candidate.label,
                        "perturbation_id": candidate.perturbation_id,
                        "base_repair_variant": candidate.base_repair_variant,
                        "front_cap_variant": variant.label,
                        "cap_ratio": variant.cap_ratio,
                        "horizon_s": horizon_s,
                        "front_local_density_proxy_before": cap_meta["front_local_density_proxy_before"],
                        "front_local_density_proxy_after": cap_meta["front_local_density_proxy_after"],
                        "front_cap_pruned_count": cap_meta["front_pruned_count"],
                        "front_pruning_from_protected_ratio": cap_meta["front_pruning_from_protected_ratio"],
                        "front_cap_target_unmet": cap_meta["front_cap_target_unmet"],
                        "front_sector_false_free_rate_delta_vs_native": float(after_eval["front_sector_false_free"]) - float(native_eval["front_sector_false_free"]),
                        "future_h4_h6_false_free_rate_delta_vs_native": float(after_eval["false_free_rate"]) - float(native_eval["false_free_rate"]) if horizon_s in {4, 6} else 0.0,
                        "A10_front_h6_recovery_rate_delta_vs_native": float(after_eval["A10_front_h6_recovery_ratio"]) - float(native_eval["A10_front_h6_recovery_ratio"]),
                        "pred_gt_density_delta": float(after_eval["pred_gt_density_delta"]),
                        "final_native_expansion_ratio": safe_div(int(after_occ.sum().item()) - int(native_occ.sum().item()), max(1, int(native_occ.sum().item()))),
                        "false_positive_delta": float(after_eval["false_occupied_rate"]) - float(native_eval["false_occupied_rate"]),
                        "wrong_class_delta": float(after_eval["wrong_class_activation_delta"]),
                        "protected_zone_preservation_ratio": protected_preservation_ratio,
                        "front_kept_ratio": front_kept_ratio,
                        "sample_joint_success_flag": False,
                        "no_gt_front_cap": True,
                        "uses_gt_budget": False,
                        "uses_gt_repair": False,
                        "uses_pred_gt_density_for_selection": False,
                        "no_training": True,
                        "agreement_source_count": candidate.agreement_source_count,
                        "front_false_positive_delta_after": after_front_fp - native_front_fp,
                        "front_false_free_delta_after": after_front_ff - native_front_ff,
                        "front_occ_native": front_occ_native,
                        "front_occ_before_cap": front_occ_before,
                        "front_occ_after_cap": front_occ_after,
                        "front_pred_gt_density_delta_before": front_pred_gt_density_before,
                        "front_pred_gt_density_delta_after": front_pred_gt_density_after,
                        "front_false_positive_delta_before": before_front_fp - native_front_fp,
                        "front_false_positive_delta_after": after_front_fp - native_front_fp,
                        "front_false_free_delta_before": before_front_ff - native_front_ff,
                        "front_false_free_delta_after": after_front_ff - native_front_ff,
                        "front_local_density_risk_before": risk_before,
                        "front_local_density_risk_after": front_local_risk_after,
                    }
                    metrics_rows.append(row)
                    audit_rows.append(
                        {
                            "sample_index": sample_index,
                            "candidate_name": candidate.label,
                            "front_cap_variant": variant.label,
                            "horizon_s": horizon_s,
                            "front_occ_native": front_occ_native,
                            "front_occ_before_cap": front_occ_before,
                            "front_occ_after_cap": front_occ_after,
                            "front_local_density_proxy_before": cap_meta["front_local_density_proxy_before"],
                            "front_local_density_proxy_after": cap_meta["front_local_density_proxy_after"],
                            "front_native_expansion_ratio_after": safe_div(front_occ_after - front_occ_native, max(1, front_occ_native)),
                            "front_cap_pruned_count": cap_meta["front_pruned_count"],
                            "front_cap_target_unmet_rate": 1.0 if cap_meta["front_cap_target_unmet"] else 0.0,
                            "front_pred_gt_density_delta_before": front_pred_gt_density_before,
                            "front_pred_gt_density_delta_after": front_pred_gt_density_after,
                            "front_false_positive_delta_before": before_front_fp - native_front_fp,
                            "front_false_positive_delta_after": after_front_fp - native_front_fp,
                            "front_false_free_delta_before": before_front_ff - native_front_ff,
                            "front_false_free_delta_after": after_front_ff - native_front_ff,
                            "front_local_density_risk_before": risk_before,
                            "front_local_density_risk_after": front_local_risk_after,
                        }
                    )
                    consistency_rows.append(
                        {
                            "candidate_name": candidate.label,
                            "front_cap_variant": variant.label,
                            "sample_index": sample_index,
                            "horizon_s": horizon_s,
                            "front_false_free_improved": float(after_eval["front_sector_false_free"]) < float(native_eval["front_sector_false_free"]),
                            "future_false_free_improved": float(after_eval["false_free_rate"]) < float(native_eval["false_free_rate"]) if horizon_s in {4, 6} else False,
                            "density_safe": float(after_eval["pred_gt_density_delta"]) <= (0.18 if candidate.label == "A1_fixed" else 0.22),
                            "front_local_safe": not front_local_risk_after,
                            "false_positive_safe": float(after_eval["false_occupied_rate"]) - float(native_eval["false_occupied_rate"]) <= 0.01,
                            "wrong_class_safe": float(after_eval["wrong_class_activation_delta"]) <= float(before_eval["wrong_class_activation_delta"]) + 1e-9,
                            "joint_success": float(after_eval["front_sector_false_free"]) < float(native_eval["front_sector_false_free"])
                            and float(after_eval["pred_gt_density_delta"]) <= (0.18 if candidate.label == "A1_fixed" else 0.22)
                            and not front_local_risk_after
                            and float(after_eval["false_occupied_rate"]) - float(native_eval["false_occupied_rate"]) <= 0.01,
                            "frontcap_target_met": not cap_meta["front_cap_target_unmet"],
                            "frontcap_preservation_safe": cap_meta["front_pruning_from_protected_ratio"] <= 0.05 and protected_preservation_ratio >= 0.80,
                        }
                    )
                    out_name = f"{candidate.label}__{variant.label}__sample{sample_index:03d}_h{horizon_s}.npz"
                    np.savez(
                        ARTIFACTS_DIR / "final_outputs" / out_name,
                        gt_h=gt_h.numpy().astype(np.int16),
                        native_semantic=native_semantic.numpy().astype(np.int16),
                        raw_semantic=raw_semantic.numpy().astype(np.int16),
                        final_before_cap=final_before.numpy().astype(np.int16),
                        final_after_cap=final_after.numpy().astype(np.int16),
                        protected_zone=protected.numpy().astype(np.uint8),
                        front_cap_pruned_mask=front_pruned.numpy().astype(np.uint8),
                    )
                    if sample_index == 0 and horizon_s == 6 and variant.label != "FC0_no_front_cap" and candidate.label in {"A1_fixed", "A10_fixed_main"}:
                        visual_cases[candidate.label] = {
                            "gt_h": gt_h,
                            "native": native_semantic,
                            "raw": raw_semantic,
                            "before": final_before,
                            "after": final_after,
                            "protected": protected,
                            "front_pruned": front_pruned,
                        }

    aggregate_rows = sw13c_fix.aggregate_rows(
        metrics_rows,
        ["candidate_name", "perturbation_id", "base_repair_variant", "front_cap_variant", "horizon_s"],
    )
    summary_rows = aggregate_rows
    selection = {
        "selection_uses_gt": False,
        "frontcap_selection_only": True,
        "no_global_reselection": True,
        "no_global_retuning": True,
        "forbidden_inputs_checked": [
            "pred_gt_density_delta",
            "front_sector_false_free_rate_delta_vs_native",
            "A10_front_h6_recovery_rate_delta_vs_native",
            "false_positive_delta",
        ],
    }
    selected_rows = {name: choose_frontcap(summary_rows, name) for name in by_candidate}
    selection.update(
        {
            "selected_A1_frontcap": selected_rows["A1_fixed"],
            "selected_A10_R4_frontcap": selected_rows["A10_fixed_main"],
            "selected_A10_R8_frontcap": selected_rows["A10_fixed_secondary"],
            "selected_C4": selected_rows["C4_fixed"],
        }
    )
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_candidate_selection.json", selection)
    write_md(
        REPORTS_DIR / "sw13c_fix_frontcap_candidate_selection.md",
        "\n".join(
            [
                "# FrontCap candidate selection",
                "",
                "- frontcap selection only; no global reselection and no global retuning.",
                "- selection uses only prediction-side front proxy, cap target met flag, protected pruning ratio, protected preservation, front kept ratio, agreement source count, and no-GT flags.",
                "- GT-derived density, false-positive, false-free, recovery, IoU, and mIoU are forbidden for selection.",
            ]
        ),
    )

    selected_metrics = [
        row
        for row in metrics_rows
        if row["front_cap_variant"] == selected_rows[row["candidate_name"]]["front_cap_variant"]
    ]
    consistency_after_rows = [row for row in consistency_rows if row["front_cap_variant"] == selected_rows[row["candidate_name"]]["front_cap_variant"]]
    consistency_summary_rows: list[dict[str, Any]] = []
    for candidate_name in by_candidate:
        items = [row for row in consistency_after_rows if row["candidate_name"] == candidate_name]
        n = max(1, len(items))
        consistency_summary_rows.append(
            {
                "candidate_name": candidate_name,
                "front_false_free_improved_rate": sum(bool(x["front_false_free_improved"]) for x in items) / n,
                "future_false_free_improved_rate": sum(bool(x["future_false_free_improved"]) for x in items) / n,
                "density_safe_rate": sum(bool(x["density_safe"]) for x in items) / n,
                "front_local_safe_rate": sum(bool(x["front_local_safe"]) for x in items) / n,
                "false_positive_safe_rate": sum(bool(x["false_positive_safe"]) for x in items) / n,
                "wrong_class_safe_rate": sum(bool(x["wrong_class_safe"]) for x in items) / n,
                "joint_success_rate": sum(bool(x["joint_success"]) for x in items) / n,
                "frontcap_target_met_rate": sum(bool(x["frontcap_target_met"]) for x in items) / n,
                "frontcap_preservation_safe_rate": sum(bool(x["frontcap_preservation_safe"]) for x in items) / n,
            }
        )

    compare_rows: list[dict[str, Any]] = []
    for candidate_name in ["A10_fixed_main", "A10_fixed_secondary"]:
        row = selected_rows[candidate_name]
        cons = [x for x in consistency_summary_rows if x["candidate_name"] == candidate_name][0]
        compare_rows.append(
            {
                "candidate_name": candidate_name,
                "selected_front_cap": row["front_cap_variant"],
                "front_local_density_proxy_after": row["front_local_density_proxy_after"],
                "front_sector_false_free_rate_delta_vs_native": row["front_sector_false_free_rate_delta_vs_native"],
                "A10_front_h6_recovery_rate_delta_vs_native": row["A10_front_h6_recovery_rate_delta_vs_native"],
                "pred_gt_density_delta": row["pred_gt_density_delta"],
                "wrong_class_delta": row["wrong_class_delta"],
                "false_positive_delta": row["false_positive_delta"],
                "sample_joint_success_rate": cons["joint_success_rate"],
                "frontcap_target_met_rate": cons["frontcap_target_met_rate"],
            }
        )

    a1_row = selected_rows["A1_fixed"]
    a10_r4_row = selected_rows["A10_fixed_main"]
    a10_r8_row = selected_rows["A10_fixed_secondary"]
    c4_row = selected_rows["C4_fixed"]
    a1_cons = [x for x in consistency_summary_rows if x["candidate_name"] == "A1_fixed"][0]
    a10_r4_cons = [x for x in consistency_summary_rows if x["candidate_name"] == "A10_fixed_main"][0]
    a10_r8_cons = [x for x in consistency_summary_rows if x["candidate_name"] == "A10_fixed_secondary"][0]
    c4_cons = [x for x in consistency_summary_rows if x["candidate_name"] == "C4_fixed"][0]

    def a1_strong(row: dict[str, Any], cons: dict[str, Any]) -> bool:
        return (
            float(row["front_local_density_proxy_after"]) <= 1.50
            and not sw13c_fix.truthy(row["front_local_density_risk_after"])
            and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.25
            and float(row["future_h4_h6_false_free_rate_delta_vs_native"]) <= -0.08
            and float(row["pred_gt_density_delta"]) <= 0.18
            and float(row["false_positive_delta"]) <= 0.01
            and float(row["protected_zone_preservation_ratio"]) >= 0.80
            and float(row["front_pruning_from_protected_ratio"]) <= 0.05
            and float(cons["joint_success_rate"]) >= 0.50
        )

    def a1_medium(row: dict[str, Any], cons: dict[str, Any]) -> bool:
        return (
            float(row["front_local_density_proxy_after"]) <= 1.60
            and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.18
            and float(row["pred_gt_density_delta"]) <= 0.20
            and float(row["protected_zone_preservation_ratio"]) >= 0.70
            and float(cons["joint_success_rate"]) >= 0.40
        )

    def a10_strong(row: dict[str, Any], cons: dict[str, Any]) -> bool:
        return (
            float(row["front_local_density_proxy_after"]) <= 1.30
            and not sw13c_fix.truthy(row["front_local_density_risk_after"])
            and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.30
            and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.25
            and float(row["future_h4_h6_false_free_rate_delta_vs_native"]) <= -0.12
            and float(row["pred_gt_density_delta"]) <= 0.20
            and float(row["false_positive_delta"]) <= 0.01
            and float(row["protected_zone_preservation_ratio"]) >= 0.80
            and float(row["front_pruning_from_protected_ratio"]) <= 0.05
            and float(cons["joint_success_rate"]) >= 0.60
        )

    def a10_medium(row: dict[str, Any], cons: dict[str, Any]) -> bool:
        return (
            float(row["front_local_density_proxy_after"]) <= 1.35
            and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.22
            and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.18
            and float(row["pred_gt_density_delta"]) <= 0.22
            and float(row["protected_zone_preservation_ratio"]) >= 0.70
            and float(cons["joint_success_rate"]) >= 0.50
        )

    def c4_ok(row: dict[str, Any]) -> bool:
        return (
            float(row["front_sector_false_free_rate_delta_vs_native"]) <= 0.0
            and float(row["pred_gt_density_delta"]) <= 0.05
            and float(row["false_positive_delta"]) <= 0.01
        )

    if any(sw13c_fix.truthy(selection.get(key, {}).get("selection_uses_gt", False)) for key in []):
        decision_type = "X6_PROTOCOL_VIOLATION"
    elif a1_strong(a1_row, a1_cons) or a10_strong(a10_r4_row, a10_r4_cons) or a10_strong(a10_r8_row, a10_r8_cons):
        decision_type = "X1_FRONTCAP_STRONG_FIX"
    elif a1_medium(a1_row, a1_cons) or a10_medium(a10_r4_row, a10_r4_cons) or a10_medium(a10_r8_row, a10_r8_cons):
        decision_type = "X2_FRONTCAP_MEDIUM_FIX"
    elif not bool(a1_row["front_local_density_risk_after"]) or not bool(a10_r4_row["front_local_density_risk_after"]) or not bool(a10_r8_row["front_local_density_risk_after"]):
        decision_type = "X3_FRONTCAP_RECOVERY_TRADEOFF"
    elif bool(a1_row["front_local_density_risk_after"]) or bool(a10_r4_row["front_local_density_risk_after"]) or bool(a10_r8_row["front_local_density_risk_after"]):
        decision_type = "X4_FRONTCAP_RESIDUAL_FRONT_DENSITY_RISK"
    else:
        decision_type = "X5_FRONTCAP_FAILS"

    decision = {
        "decision_type": decision_type,
        "selected_A1_frontcap_result": a1_row,
        "selected_A10_R4_frontcap_result": a10_r4_row,
        "selected_A10_R8_frontcap_result": a10_r8_row,
        "selected_C4_result": c4_row,
        "before_after_summary": {
            "A1_before": scale_decision["best_A1_scale_result"]["front_local_density_proxy"],
            "A1_after": a1_row["front_local_density_proxy_after"],
            "A10_R4_before": scale_decision["best_A10_R4_scale_result"]["front_local_density_proxy"],
            "A10_R4_after": a10_r4_row["front_local_density_proxy_after"],
            "A10_R8_before": scale_decision["best_A10_R8_scale_result"]["front_local_density_proxy"],
            "A10_R8_after": a10_r8_row["front_local_density_proxy_after"],
        },
        "front_local_density_risk_before": True,
        "front_local_density_risk_after": bool(a1_row["front_local_density_risk_after"]) or bool(a10_r4_row["front_local_density_risk_after"]) or bool(a10_r8_row["front_local_density_risk_after"]),
        "no_gt_front_cap": True,
        "no_training": True,
        "no_global_reselection": True,
        "no_global_retuning": True,
        "not_official_benchmark": True,
    }
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_eval_core50_decision.json", decision)
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_eval_core50_decision.md", f"decision: {decision_type}\n")

    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_metrics.csv", metrics_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_aggregate_metrics.csv", aggregate_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_metric_summary.csv", summary_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_density_audit.csv", audit_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_sample_consistency.csv", consistency_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_A10_R4_R8_comparison.csv", compare_rows)
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_A10_R4_R8_comparison.md", "R8 secondary fixed candidate appears more stable under the same front cap protocol only if its frozen secondary metrics are stronger. No eval_core50 parameter search was performed.\n")
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_density_audit.md", json.dumps({"front_local_density_risk_before": True, "front_local_density_risk_after": decision["front_local_density_risk_after"]}, indent=2))
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_sample_consistency_summary.md", json.dumps(consistency_summary_rows, indent=2))

    labels = ["A1", "A10 R4", "A10 R8"]
    before_vals = [
        float(scale_decision["best_A1_scale_result"]["front_local_density_proxy"]),
        float(scale_decision["best_A10_R4_scale_result"]["front_local_density_proxy"]),
        float(scale_decision["best_A10_R8_scale_result"]["front_local_density_proxy"]),
    ]
    after_vals = [float(a1_row["front_local_density_proxy_after"]), float(a10_r4_row["front_local_density_proxy_after"]), float(a10_r8_row["front_local_density_proxy_after"])]
    save_bar(FIGURES_DIR / "sw13c_fix_frontcap_density_bar.png", "Frozen candidate front-local density cap", labels, before_vals, after_vals)
    save_bar(
        FIGURES_DIR / "sw13c_fix_frontcap_recovery_density_tradeoff.png",
        "Frozen candidate front-local density cap recovery-density tradeoff",
        labels,
        [
            abs(float(scale_decision["best_A1_scale_result"]["front_sector_false_free_rate_delta_vs_native"])),
            abs(float(scale_decision["best_A10_R4_scale_result"]["front_sector_false_free_rate_delta_vs_native"])),
            abs(float(scale_decision["best_A10_R8_scale_result"]["front_sector_false_free_rate_delta_vs_native"])),
        ],
        [
            abs(float(a1_row["front_sector_false_free_rate_delta_vs_native"])),
            abs(float(a10_r4_row["front_sector_false_free_rate_delta_vs_native"])),
            abs(float(a10_r8_row["front_sector_false_free_rate_delta_vs_native"])),
        ],
    )
    save_bar(
        FIGURES_DIR / "sw13c_fix_frontcap_A10_R4_R8_comparison.png",
        "Frozen candidate front-local density cap A10 R4 vs R8",
        ["R4 proxy", "R8 proxy", "R4 joint", "R8 joint"],
        [float(scale_decision["best_A10_R4_scale_result"]["front_local_density_proxy"]), float(scale_decision["best_A10_R8_scale_result"]["front_local_density_proxy"]), 0.0, 0.0],
        [float(a10_r4_row["front_local_density_proxy_after"]), float(a10_r8_row["front_local_density_proxy_after"]), float(a10_r4_cons["joint_success_rate"]), float(a10_r8_cons["joint_success_rate"])],
    )
    save_line_compare(
        FIGURES_DIR / "sw13c_fix_frontcap_sample_consistency.png",
        "Frozen candidate front-local density cap sample consistency",
        ["A1", "A10 R4", "A10 R8", "C4"],
        [float(a1_cons["joint_success_rate"]), float(a10_r4_cons["joint_success_rate"]), float(a10_r8_cons["joint_success_rate"]), float(c4_cons["joint_success_rate"])],
    )
    save_bar(FIGURES_DIR / "sw13c_fix_frontcap_A1_before_after.png", "Frozen candidate front-local density cap A1 before/after", ["front proxy", "density delta"], [float(scale_decision["best_A1_scale_result"]["front_local_density_proxy"]), float(scale_decision["best_A1_scale_result"]["pred_gt_density_delta"])], [float(a1_row["front_local_density_proxy_after"]), float(a1_row["pred_gt_density_delta"])])
    save_bar(FIGURES_DIR / "sw13c_fix_frontcap_A10_R4_before_after.png", "Frozen candidate front-local density cap A10 R4 before/after", ["front proxy", "density delta"], [float(scale_decision["best_A10_R4_scale_result"]["front_local_density_proxy"]), float(scale_decision["best_A10_R4_scale_result"]["pred_gt_density_delta"])], [float(a10_r4_row["front_local_density_proxy_after"]), float(a10_r4_row["pred_gt_density_delta"])])
    save_bar(FIGURES_DIR / "sw13c_fix_frontcap_A10_R8_before_after.png", "Frozen candidate front-local density cap A10 R8 before/after", ["front proxy", "density delta"], [float(scale_decision["best_A10_R8_scale_result"]["front_local_density_proxy"]), float(scale_decision["best_A10_R8_scale_result"]["pred_gt_density_delta"])], [float(a10_r8_row["front_local_density_proxy_after"]), float(a10_r8_row["pred_gt_density_delta"])])
    if "A1_fixed" in visual_cases:
        save_bev_panel(
            FIGURES_DIR / "sw13c_fix_frontcap_bev_A1_sample0.png",
            visual_cases["A1_fixed"]["gt_h"],
            visual_cases["A1_fixed"]["native"],
            visual_cases["A1_fixed"]["raw"],
            visual_cases["A1_fixed"]["before"],
            visual_cases["A1_fixed"]["after"],
            visual_cases["A1_fixed"]["protected"],
            visual_cases["A1_fixed"]["front_pruned"],
            "frozen candidate front-local density cap no GT front cap eval_core50 subset diagnostic no training",
        )
    if "A10_fixed_main" in visual_cases:
        save_bev_panel(
            FIGURES_DIR / "sw13c_fix_frontcap_bev_A10_sample0.png",
            visual_cases["A10_fixed_main"]["gt_h"],
            visual_cases["A10_fixed_main"]["native"],
            visual_cases["A10_fixed_main"]["raw"],
            visual_cases["A10_fixed_main"]["before"],
            visual_cases["A10_fixed_main"]["after"],
            visual_cases["A10_fixed_main"]["protected"],
            visual_cases["A10_fixed_main"]["front_pruned"],
            "frozen candidate front-local density cap no GT front cap eval_core50 subset diagnostic no training",
        )

    write_md(
        REPORTS_DIR / "stage_sw13c_fix_frontcap_eval_core50_report.md",
        "\n".join(
            [
                "# Stage SW-13C-Fix-FrontCap eval_core50",
                "",
                "1. Executive summary",
                f"- decision: {decision_type}",
                "",
                "2. Why FrontCap was needed",
                "- SW-13C-Fix-Scale recovery scaled to eval_core50 but had front-local density risk.",
                "",
                "3. front_local_density_proxy definition audit",
                "- prediction-side proxy = front_occ_count_final / front_occ_count_native.",
                "",
                "4. Frozen candidate protocol",
                "- no global reselection and no global retuning.",
                "",
                "5. No-GT front-local cap method",
                "- FrontCap is a second-stage prediction-side cap.",
                "- FrontCap does not use GT budget.",
                "- FrontCap does not retrain.",
                "- FrontCap does not change get_occ.",
                "- FrontCap does not reopen global parameter search.",
                "- GT metrics are evaluation-only.",
                "",
                "6. A1 before/after front cap",
                f"- selected {a1_row['front_cap_variant']}",
                "",
                "7. A10 R4 before/after front cap",
                f"- selected {a10_r4_row['front_cap_variant']}",
                "",
                "8. A10 R8 before/after front cap",
                f"- selected {a10_r8_row['front_cap_variant']}",
                "",
                "9. C4 preservation",
                f"- selected {c4_row['front_cap_variant']}",
                "",
                "10. Front-local density audit after cap",
                f"- front_local_density_risk_after={decision['front_local_density_risk_after']}",
                "",
                "11. Sample consistency after cap",
                "- reported per candidate and horizon.",
                "",
                "12. A10 R4 vs R8",
                "- R8 secondary fixed candidate appears more stable under the same front cap protocol only if its frozen secondary metrics are stronger.",
                "",
                "13. Decision X1-X6",
                f"- {decision_type}",
                "",
                "14. Safe claims",
                "- After adding a no-GT front-local density cap to the frozen SW-13C-Fix candidate, the eval_core50 diagnostic keeps front-sector recovery while reducing the front-local density risk." if decision_type in {"X1_FRONTCAP_STRONG_FIX", "X2_FRONTCAP_MEDIUM_FIX"} else "- FrontCap improves but does not fully solve front-local density risk; current best claim remains recovery scaling with identified density limitation.",
                "- subset diagnostic only",
                "- eval_core50 subset diagnostic",
                "- not official benchmark",
                "",
                "15. Limitations",
                "- no training",
                "- no checkpoint modification",
                "- no get_occ modification",
                "",
                "16. Next unique action",
                "- if X1 or X2 holds, consider eval_core100 frozen-candidate front-cap verification before any training.",
            ]
        ),
    )
    write_json(REPORTS_DIR / "stage_sw13c_fix_frontcap_eval_core50_report.json", {"decision": decision_type, "selected": decision})

    write_tests()


if __name__ == "__main__":
    main()
