from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import collate as collate_fn
import gc


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget"

SW13A_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
SW13A_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
SW13B_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
SW13C_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory"
EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw2 = load_module(
    "sw13c_fix_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw13c_fix_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw13c_fix_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw13c_fix_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw12b = load_module(
    "sw13c_fix_sw12b",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py",
)
sw13a = load_module(
    "sw13c_fix_sw13a",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/run_sw13a_main.py",
)
sw13b = load_module(
    "sw13c_fix_sw13b",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/run_sw13b_main.py",
)


@dataclass(frozen=True)
class BaseRepairSpec:
    label: str
    perturbation_id: str
    raw_variant_label: str
    source_variant_labels: list[str]
    preserve_label: str | None


@dataclass(frozen=True)
class FixVariant:
    label: str
    budget_mode: str
    expansion_ratio: float | None
    keep_ratio: float | None
    protected_label: str
    wrong_class_aware: bool = False
    front_bias: float = 1.0


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "multi_source_agreement",
        ARTIFACTS_DIR / "protected_zones",
        ARTIFACTS_DIR / "pruning_scores",
        ARTIFACTS_DIR / "final_outputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_export(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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
            writer.writerow(normalize_export(row))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-13C-Fix no-GT density budget and true multi-source agreement")
    parser.add_argument("--samples", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--scale50", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_runtime():
    cfg, dataset, model, _ = sw81.build_sparseworld_runtime(train=False, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    model.eval()
    return cfg, dataset, model


def source_variants() -> dict[str, Any]:
    return {variant.label: variant for variant in sw13a.make_variants()}


def base_specs() -> list[BaseRepairSpec]:
    return [
        BaseRepairSpec("A1_base", "A1_drop_cam_front", "R1_replace_tminus1", ["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2"], None),
        BaseRepairSpec("A10_base_1", "A10_drop_front_triplet", "R4_ema_K3", ["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"], None),
        BaseRepairSpec("A10_base_2", "A10_drop_front_triplet", "R8_camera_group_repair_front_triplet", ["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"], None),
        BaseRepairSpec("C4_base", "C4_motion_blur_9", "R5_blend_alpha03", ["R5_blend_alpha03", "R6_blend_alpha05", "R7_blend_alpha07"], "F9_C4_preserve_R5"),
    ]


def fix_variants_for(base: BaseRepairSpec) -> list[FixVariant]:
    if base.perturbation_id == "C4_motion_blur_9":
        return [
            FixVariant("F0_raw_repair", "raw", None, None, "PZ_fix_2_front_conf_agree"),
            FixVariant("F9_C4_preserve_R5", "preserve", None, None, "PZ_fix_2_front_conf_agree"),
        ]
    if base.perturbation_id == "A1_drop_cam_front":
        return [
            FixVariant("F0_raw_repair", "raw", None, None, "PZ_fix_2_front_conf_agree"),
            FixVariant("F1_expand_budget_light", "native_expansion_ratio", 0.16, None, "PZ_fix_2_front_conf_agree"),
            FixVariant("F2_expand_budget_medium", "native_expansion_ratio", 0.12, None, "PZ_fix_2_front_conf_agree"),
            FixVariant("F3_expand_budget_strict", "native_expansion_ratio", 0.08, None, "PZ_fix_3_strong_core"),
            FixVariant("F4_keep_raw_delta_65", "raw_delta_keep_ratio", None, 0.65, "PZ_fix_2_front_conf_agree"),
            FixVariant("F5_keep_raw_delta_50", "raw_delta_keep_ratio", None, 0.50, "PZ_fix_2_front_conf_agree"),
            FixVariant("F6_keep_raw_delta_35", "raw_delta_keep_ratio", None, 0.35, "PZ_fix_3_strong_core"),
            FixVariant("F7_wrong_class_aware_medium", "native_expansion_ratio", 0.12, None, "PZ_fix_2_front_conf_agree", True),
            FixVariant("F8_front_budget_reallocation_no_gt", "native_expansion_ratio", 0.12, None, "PZ_fix_2_front_conf_agree", False, 1.75),
        ]
    return [
        FixVariant("F0_raw_repair", "raw", None, None, "PZ_fix_2_front_conf_agree"),
        FixVariant("F1_expand_budget_light", "native_expansion_ratio", 0.20, None, "PZ_fix_2_front_conf_agree"),
        FixVariant("F2_expand_budget_medium", "native_expansion_ratio", 0.16, None, "PZ_fix_2_front_conf_agree"),
        FixVariant("F3_expand_budget_strict", "native_expansion_ratio", 0.12, None, "PZ_fix_3_strong_core"),
        FixVariant("F4_keep_raw_delta_65", "raw_delta_keep_ratio", None, 0.65, "PZ_fix_2_front_conf_agree"),
        FixVariant("F5_keep_raw_delta_50", "raw_delta_keep_ratio", None, 0.50, "PZ_fix_2_front_conf_agree"),
        FixVariant("F6_keep_raw_delta_35", "raw_delta_keep_ratio", None, 0.35, "PZ_fix_3_strong_core"),
        FixVariant("F7_wrong_class_aware_medium", "native_expansion_ratio", 0.16, None, "PZ_fix_2_front_conf_agree", True),
        FixVariant("F8_front_budget_reallocation_no_gt", "native_expansion_ratio", 0.16, None, "PZ_fix_2_front_conf_agree", False, 1.75),
    ]


def load_or_build_cache_map(dataset: Any, model: Any, sample_ids: list[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for sample_id in sample_ids:
        cache_path = SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_id:03d}.pt"
        if cache_path.exists():
            out[sample_id] = torch.load(cache_path, map_location="cpu", weights_only=False)
            continue
        raw_sample, batch = sw2.extract_sample_batch(dataset, sample_id, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        moved = sw2.move_to_cuda(batch)
        sw13a.reset_model_cache(model)
        _, _, cache = sw13a.extract_clean_memory_for_sample(model, moved, sample_unwrapped, sample_id)
        out[sample_id] = cache
    return out


def confidence_entropy(conf: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(conf, 1e-4, 1 - 1e-4)
    return -(c * torch.log(c) + (1 - c) * torch.log(1 - c))


def local_support(mask: torch.Tensor) -> torch.Tensor:
    k = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=mask.device)
    return F.conv3d(mask.float()[None, None], k, padding=1)[0, 0]


def qmask(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    if not mask.any():
        return torch.zeros_like(mask)
    thr = torch.quantile(values[mask].float(), q)
    return mask & (values >= thr)


def source_source_counts() -> dict[str, int]:
    return {
        "A1_drop_cam_front": 3,
        "A10_drop_front_triplet": 4,
        "C4_motion_blur_9": 3,
    }


def protected_zone_fix(label: str, raw_occ: torch.Tensor, raw_delta: torch.Tensor, occ_conf: torch.Tensor, margin: torch.Tensor, agreement: torch.Tensor, sectors: dict[str, torch.Tensor], horizon_s: int) -> torch.Tensor:
    front = sectors["front"].bool()
    future = horizon_s in {4, 6}
    conf50 = qmask(occ_conf, raw_delta, 0.50)
    conf70 = qmask(occ_conf, raw_delta, 0.70)
    margin_good = margin >= torch.quantile(margin[raw_delta].float(), 0.50) if raw_delta.any() else torch.zeros_like(raw_delta)
    support5 = local_support(raw_occ) >= 5
    if label == "PZ_fix_1_front_conf":
        return raw_occ & front & conf50 & (torch.ones_like(raw_occ, dtype=torch.bool) if future else torch.ones_like(raw_occ, dtype=torch.bool))
    if label == "PZ_fix_2_front_conf_agree":
        return raw_occ & front & conf50 & (agreement >= 0.5)
    if label == "PZ_fix_3_strong_core":
        return raw_occ & front & conf70 & (agreement >= 0.67) & support5 & margin_good
    return torch.zeros_like(raw_occ)


def low_value_score(raw_occ: torch.Tensor, protected: torch.Tensor, occ_conf: torch.Tensor, margin: torch.Tensor, agreement: torch.Tensor, sectors: dict[str, torch.Tensor], horizon_s: int, wrong_class_aware: bool, front_bias: float) -> torch.Tensor:
    front = sectors["front"].bool()
    entropy = confidence_entropy(torch.clamp(occ_conf, 0, 1))
    conf_bad = 1.0 - torch.clamp(occ_conf, 0, 1)
    margin_norm = torch.zeros_like(margin)
    if raw_occ.any():
        margin_norm = margin.float() / (margin[raw_occ].float().max() + 1e-6)
    margin_bad = 1.0 - torch.clamp(margin_norm, 0, 1)
    isolated = (local_support(raw_occ) < 3).float()
    nonfront = (~front).float()
    future_front_discount = 0.35 if horizon_s in {4, 6} else 0.10
    score = 1.2 * conf_bad + 0.9 * margin_bad + 0.8 * entropy + 1.4 * (1.0 - agreement) + front_bias * nonfront + 0.8 * isolated
    if wrong_class_aware:
        score += 0.8 * (margin_bad + entropy)
    score = score - 100.0 * protected.float() - future_front_discount * front.float()
    score[~raw_occ] = -1e6
    return score


def apply_pruning_no_gt(
    raw_semantic: torch.Tensor,
    protected_mask: torch.Tensor,
    low_value_score_map: torch.Tensor,
    native_occ_count: int,
    raw_occ_count: int,
    raw_delta_count: int,
    budget_mode: str,
    expansion_ratio: float | None = None,
    keep_ratio: float | None = None,
):
    raw_occ = raw_semantic != EMPTY_IDX
    if budget_mode in {"raw", "preserve"}:
        budget_meta = {
            "budget_mode": budget_mode,
            "native_occ_count": int(native_occ_count),
            "raw_occ_count": int(raw_occ_count),
            "raw_delta_count": int(raw_delta_count),
            "target_final_occ_count": int(raw_occ_count),
            "target_expansion_ratio": None,
            "target_keep_ratio": None,
            "no_gt_budget": True,
        }
        return raw_semantic.clone(), torch.zeros_like(raw_occ), budget_meta
    if budget_mode == "native_expansion_ratio":
        assert expansion_ratio is not None
        target_final_occ_count = int(round(native_occ_count * (1.0 + expansion_ratio)))
    elif budget_mode == "raw_delta_keep_ratio":
        assert keep_ratio is not None
        target_final_occ_count = int(round(native_occ_count + keep_ratio * raw_delta_count))
    else:
        raise ValueError(f"unknown budget_mode={budget_mode}")
    target_final_occ_count = max(native_occ_count, min(raw_occ_count, target_final_occ_count))
    prune_count = raw_occ_count - target_final_occ_count
    candidates = raw_occ & ~protected_mask
    final = raw_semantic.clone()
    pruned = torch.zeros_like(raw_occ)
    if prune_count > 0 and candidates.any():
        coords = torch.nonzero(candidates, as_tuple=False)
        vals = low_value_score_map[candidates]
        k = min(prune_count, vals.numel())
        _, idx = torch.topk(vals, k=k, largest=True)
        picked = coords[idx]
        pruned[picked[:, 0], picked[:, 1], picked[:, 2]] = True
        final[pruned] = EMPTY_IDX
    final_occ_count = int((final != EMPTY_IDX).sum().item())
    budget_meta = {
        "budget_mode": budget_mode,
        "native_occ_count": int(native_occ_count),
        "raw_occ_count": int(raw_occ_count),
        "raw_delta_count": int(raw_delta_count),
        "target_final_occ_count": int(target_final_occ_count),
        "target_expansion_ratio": expansion_ratio,
        "target_keep_ratio": keep_ratio,
        "final_occ_count": int(final_occ_count),
        "no_gt_budget": True,
    }
    return final, pruned, budget_meta


def build_eval_row(pred: torch.Tensor, gt_h: torch.Tensor, gt0: torch.Tensor, perturbation_id: str, horizon_s: int, sectors: dict[str, torch.Tensor], baseline_pred: torch.Tensor) -> dict[str, Any]:
    row = sw12b.build_eval_row(pred, gt_h, gt0, perturbation_id, horizon_s, sectors, baseline_pred=baseline_pred)
    row["front_sector_false_free_rate"] = row["front_sector_false_free"]
    row["small_object_false_free_rate"] = row["small_object_false_free"]
    row["dynamic_object_false_free_rate"] = row["dynamic_false_free"]
    row["future_h4_h6_false_free_rate"] = row["false_free_rate"] if horizon_s in {4, 6} else 0.0
    row["A10_front_h6_recovery_rate"] = row["A10_front_h6_recovery_ratio"]
    row["false_positive_rate"] = row["false_occupied_rate"]
    row["wrong_class_rate"] = row["wrong_class_activation"]
    return row


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    return sw12b.aggregate_rows(rows, keys)


def front_counts(mask: torch.Tensor, sectors: dict[str, torch.Tensor]) -> int:
    return int((mask & sectors["front"].bool()).sum().item())


def run_source_cases_for_sample(
    model: Any,
    dataset: Any,
    cache: dict[str, Any],
    sample_index: int,
    perturbation_id: str,
    variant_labels: list[str],
    variant_map: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    raw_sample, batch_clean = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
    sample_unwrapped = sw2.unwrap(raw_sample)
    batch_deg = copy.deepcopy(batch_clean)
    if perturbation_id != "A0_clean":
        batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, sw81.sw5_engine.build_catalog()[perturbation_id])[0]
    native_result, native_per_h, native_debug = sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache, variant_map["R0_degraded_native"], perturbation_id)
    head = sw4_inst.get_pts_bbox_head(model)
    native_cases: dict[int, dict[str, Any]] = {}
    for horizon_s in CORE_HORIZONS:
        native_semantic, native_debug_occ = sw13b.dense_debug_for_case(model, native_per_h[horizon_s]["pred_dict"])
        conf = sw13b.confidence_maps(native_debug_occ)
        native_cases[horizon_s] = {
            "semantic": native_semantic,
            "occ_mask": native_semantic != EMPTY_IDX,
            "top1_conf": conf["top1_conf"],
            "top1_margin": conf["top1_margin"],
            "occ_conf": conf["occ_conf"],
            "free_conf": conf["free_conf"],
            "gt_h": native_per_h[horizon_s]["gt_h"],
            "gt0": native_per_h[horizon_s]["gt0"],
            "runtime_debug": native_debug,
        }
    source_cases: dict[str, dict[int, dict[str, Any]]] = {}
    for variant_label in variant_labels:
        result_cpu, per_h, runtime_debug = sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache, variant_map[variant_label], perturbation_id)
        del result_cpu
        cases_by_h: dict[int, dict[str, Any]] = {}
        for horizon_s in CORE_HORIZONS:
            semantic, dbg_occ = sw13b.dense_debug_for_case(model, per_h[horizon_s]["pred_dict"])
            conf = sw13b.confidence_maps(dbg_occ)
            cases_by_h[horizon_s] = {
                "semantic": semantic,
                "occ_mask": semantic != EMPTY_IDX,
                "top1_conf": conf["top1_conf"],
                "top1_margin": conf["top1_margin"],
                "occ_conf": conf["occ_conf"],
                "free_conf": conf["free_conf"],
                "delta_occ_mask": (semantic != EMPTY_IDX) & ~(native_cases[horizon_s]["occ_mask"]),
                "gt_h": per_h[horizon_s]["gt_h"],
                "gt0": per_h[horizon_s]["gt0"],
                "runtime_debug": runtime_debug,
            }
        source_cases[variant_label] = cases_by_h
        sw13a.reset_model_cache(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return native_cases, source_cases, sample_unwrapped


def agreement_for_horizon(source_cases: dict[str, dict[int, dict[str, Any]]], source_labels: list[str], horizon_s: int) -> tuple[torch.Tensor, list[str], list[str]]:
    occs: list[torch.Tensor] = []
    available: list[str] = []
    missing: list[str] = []
    for label in source_labels:
        if label in source_cases:
            occs.append(source_cases[label][horizon_s]["occ_mask"])
            available.append(label)
        else:
            missing.append(label)
    if not occs:
        return torch.zeros((200, 200, 16), dtype=torch.float32), available, missing
    acc = torch.zeros_like(occs[0], dtype=torch.float32)
    for occ in occs:
        acc += occ.float()
    return acc / float(len(occs)), available, missing


def save_bev(path: Path, gt_h: torch.Tensor, native: torch.Tensor, raw: torch.Tensor, final: torch.Tensor, protected: torch.Tensor, prune: torch.Tensor, title: str) -> None:
    def bev(x: torch.Tensor) -> np.ndarray:
        return (x != EMPTY_IDX).any(dim=-1).float().numpy()

    gt_b, native_b, raw_b, final_b = bev(gt_h), bev(native), bev(raw), bev(final)
    panels = [
        (gt_b, "GT"),
        (native_b, "degraded native"),
        (raw_b, "raw repair"),
        (final_b, "no-GT final"),
        (protected.any(dim=-1).float().numpy(), "protected zone"),
        (prune.any(dim=-1).float().numpy(), "pruned low-value"),
        (((gt_b > 0.5) & (native_b < 0.5)).astype(float), "false-free native"),
        (((gt_b > 0.5) & (raw_b < 0.5)).astype(float), "false-free raw"),
        (((gt_b > 0.5) & (final_b < 0.5)).astype(float), "false-free final"),
        (((gt_b < 0.5) & (raw_b > 0.5)).astype(float), "false-positive raw"),
        (((gt_b < 0.5) & (final_b > 0.5)).astype(float), "false-positive final"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, (arr, name) in zip(axes.flatten(), panels + [(np.zeros_like(gt_b), "")]):
        ax.imshow(arr.T.astype(float), origin="lower", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_with_native(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agg = aggregate_rows(rows, ["base_label", "perturbation_id", "base_repair_variant", "variant_label", "horizon_s"])
    native_map = {(r["perturbation_id"], int(r["horizon_s"])): r for r in agg if r["variant_label"] == "native_baseline"}
    raw_map = {(r["base_label"], int(r["horizon_s"])): r for r in agg if r["variant_label"] == "F0_raw_repair"}
    mode_map = {variant.label: variant.budget_mode for base in base_specs() for variant in fix_variants_for(base)}
    mode_map["native_baseline"] = "native"
    summary: list[dict[str, Any]] = []
    for row in agg:
        out = dict(row)
        out["budget_mode"] = mode_map.get(row["variant_label"], "")
        native = native_map.get((row["perturbation_id"], int(row["horizon_s"])))
        raw = raw_map.get((row["base_label"], int(row["horizon_s"])))
        if native is not None and row["variant_label"] != "native_baseline":
            out["front_sector_false_free_rate_delta_vs_native"] = float(row["front_sector_false_free"]) - float(native["front_sector_false_free"])
            out["future_h4_h6_false_free_rate_delta_vs_native"] = float(row["false_free_rate"]) - float(native["false_free_rate"]) if int(row["horizon_s"]) in {4, 6} else 0.0
            out["A10_front_h6_recovery_rate_delta_vs_native"] = float(row["A10_front_h6_recovery_ratio"]) - float(native["A10_front_h6_recovery_ratio"])
            out["false_positive_delta"] = float(row["false_occupied_rate"]) - float(native["false_occupied_rate"])
            out["pred_gt_density_delta"] = float(row["pred_gt_density_delta"])
        if raw is not None and row["variant_label"] not in {"native_baseline", "F0_raw_repair"}:
            out["wrong_class_reduction_ratio_vs_raw"] = safe_div(float(raw["wrong_class_activation_delta"]) - float(row["wrong_class_activation_delta"]), max(1e-6, float(raw["wrong_class_activation_delta"])))
        summary.append(out)
    return summary


def selection_score(row: dict[str, Any]) -> float:
    return (
        float(row["protected_zone_preservation_ratio"])
        + float(row["protected_confidence_mean"])
        + float(row["protected_agreement_mean"])
        + float(row["front_kept_ratio"])
        + 0.5 * float(row["raw_delta_keep_ratio"])
        + 0.05 * float(row["low_value_removed_mean"])
        - 5.0 * float(row["pruning_from_protected_ratio"])
    )


def select_candidates_no_gt(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for base in [b for b in base_specs() if b.perturbation_id != "C4_motion_blur_9"]:
        rows = [
            r
            for r in summary_rows
            if r["base_label"] == base.label and int(r["horizon_s"]) == 6 and r["variant_label"].startswith("F") and r["variant_label"] != "F0_raw_repair"
        ]
        valid: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            threshold = 0.16 if base.perturbation_id == "A1_drop_cam_front" else 0.20
            if not (
                truthy(row["no_gt_budget"])
                and not truthy(row["selection_uses_gt"])
                and float(row["protected_zone_preservation_ratio"]) >= 0.80
                and float(row["pruning_from_protected_ratio"]) <= 0.05
                and float(row["agreement_source_count"]) >= 2.0
                and float(row["final_native_expansion_ratio"]) <= threshold
                and float(row["raw_delta_keep_ratio"]) >= 0.35
                and float(row["front_kept_ratio"]) >= 0.80
                and float(row["protected_agreement_mean"]) >= 0.5
            ):
                continue
            valid.append((selection_score(row), row))
        if valid:
            selected[base.label] = max(valid, key=lambda item: item[0])[1]
    c4_rows = [r for r in summary_rows if r["base_label"] == "C4_base" and r["variant_label"] == "F9_C4_preserve_R5" and int(r["horizon_s"]) == 6]
    if c4_rows:
        selected["C4_base"] = c4_rows[0]
    return selected


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dirs()

    write_json(
        REPORTS_DIR / "sw13c_fix_inherited_sw13c_audit.json",
        {
            "sw13c_original_positives": {
                "A1_R1_C5_h6": {
                    "front_sector_false_free_rate_delta_vs_native": -0.3605,
                    "density_delta": 0.0491,
                    "protected_zone_preservation_ratio": 1.0,
                },
                "A10_R4_C5_h6": {
                    "front_sector_false_free_rate_delta_vs_native": -0.4910,
                    "A10_front_h6_recovery_rate_delta_vs_native": 0.4779,
                    "density_delta": 0.0797,
                    "protected_zone_preservation_ratio": 1.0,
                },
            },
            "sw13c_original_risks": [
                "pruning target used gt_occ_count",
                "candidate selection used pred_gt_density_delta",
                "temporal agreement may be incomplete or single-source",
            ],
        },
    )
    write_md(
        REPORTS_DIR / "sw13c_fix_original_risk_audit.md",
        "\n".join(
            [
                "# SW-13C original risk audit",
                "",
                "- original SW-13C showed strong diagnostic positives on A1/A10 with reduced density.",
                "- original SW-13C still had GT-density-budget risk because pruning target used `gt_occ_count`.",
                "- original SW-13C selection still had GT-derived risk because `pred_gt_density_delta` gated candidates.",
                "- original SW-13C temporal agreement was not guaranteed to be true multi-source for all selected bases.",
                "- therefore original SW-13C is retained as diagnostic upper-bound only.",
            ]
        ),
    )

    write_md(
        REPORTS_DIR / "sw13c_fix_code_change_summary.md",
        "\n".join(
            [
                "# SW-13C-Fix code changes",
                "",
                "- new stage file copied from SW-13C concepts but not in-place modified.",
                "- `apply_pruning_no_gt` removes any `gt_occ_count` dependency.",
                "- pruning targets now use prediction-side `native_expansion_ratio` or `raw_delta_keep_ratio` only.",
                "- candidate selection removes `pred_gt_density_delta` and all GT-derived ranking inputs.",
                "- multi-source agreement is rebuilt explicitly from A1 `{R1,R2,R3}`, A10 `{R1,R3,R4,R8}`, and C4 `{R5,R6,R7}`.",
            ]
        ),
    )

    cfg, dataset, model = build_runtime()
    sectors = {name: tensor.cpu().bool() for name, tensor in sw7.build_sector_masks().items()}
    sample_ids = parse_int_list(args.samples)
    cache_map = load_or_build_cache_map(dataset, model, sample_ids)
    variant_map = source_variants()
    write_json(REPORTS_DIR / "sw13c_fix_variant_manifest.json", {"variants": [normalize_export(v.__dict__) for b in base_specs() for v in fix_variants_for(b)]})
    write_json(REPORTS_DIR / "sw13c_fix_protected_zone_manifest.json", {"variants": ["PZ_fix_1_front_conf", "PZ_fix_2_front_conf_agree", "PZ_fix_3_strong_core"], "uses_gt": False})
    write_md(REPORTS_DIR / "sw13c_fix_pruning_pool_definition.md", "Prediction-side low-value pruning uses confidence, margin, entropy, agreement, component support, and non-front bias only. GT is not used for pruning or budget.")
    write_md(REPORTS_DIR / "sw13c_fix_no_gt_candidate_selection_rule.md", "Selection uses only no_gt_budget, protected preservation, pruning_from_protected_ratio, agreement_source_count, final_native_expansion_ratio, raw_delta_keep_ratio, front_kept_ratio, protected_confidence_mean, protected_agreement_mean, and low_value_removed_mean. GT-derived fields are forbidden.")

    metric_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    front_audit_rows: list[dict[str, Any]] = []
    sample_improve_rows: list[dict[str, Any]] = []
    visual_cases: dict[tuple[str, str], dict[str, Any]] = {}

    for sample_index in sample_ids:
        for base in base_specs():
            native_cases, source_cases, sample_unwrapped = run_source_cases_for_sample(model, dataset, cache_map[sample_index], sample_index, base.perturbation_id, base.source_variant_labels, variant_map)
            # native/control rows
            for horizon_s in CORE_HORIZONS:
                native_row = build_eval_row(native_cases[horizon_s]["semantic"], native_cases[horizon_s]["gt_h"], native_cases[horizon_s]["gt0"], base.perturbation_id, horizon_s, sectors, native_cases[horizon_s]["semantic"])
                native_row.update(
                    {
                        "sample_index": sample_index,
                        "base_label": base.label,
                        "perturbation_id": base.perturbation_id,
                        "base_repair_variant": "native",
                        "variant_label": "native_baseline",
                        "horizon_s": horizon_s,
                        "budget_mode": "native",
                        "expansion_ratio": 0.0,
                        "keep_ratio": 0.0,
                        "native_occ_count": int(native_cases[horizon_s]["occ_mask"].sum().item()),
                        "raw_occ_count": int(native_cases[horizon_s]["occ_mask"].sum().item()),
                        "final_occ_count": int(native_cases[horizon_s]["occ_mask"].sum().item()),
                        "raw_delta_count": 0,
                        "no_gt_budget": True,
                        "uses_gt_budget": False,
                        "selection_uses_gt": False,
                        "uses_pred_gt_density_for_selection": False,
                        "agreement_source_count": 0,
                        "protected_zone_preservation_ratio": 1.0,
                        "pruning_from_protected_ratio": 0.0,
                        "front_kept_ratio": 1.0,
                        "raw_delta_keep_ratio": 0.0,
                        "final_native_expansion_ratio": 0.0,
                    }
                )
                metric_rows.append(native_row)
            for horizon_s in CORE_HORIZONS:
                agreement_map, available, missing = agreement_for_horizon(source_cases, base.source_variant_labels, horizon_s)
                agreement_mean = float(agreement_map.mean().item()) if agreement_map.numel() else 0.0
                agreement_rows.append(
                    {
                        "sample_index": sample_index,
                        "perturbation_id": base.perturbation_id,
                        "horizon_s": horizon_s,
                        "required_sources": base.source_variant_labels,
                        "available_sources": available,
                        "agreement_source_count": len(available),
                        "agreement_mean": agreement_mean,
                        "agreement_in_protected_mean": 0.0,
                        "missing_sources": missing,
                        "uses_gt": False,
                        "uses_future_info": False,
                        "uses_current_clean_same_frame": False,
                    }
                )
                np.savez_compressed(
                    ARTIFACTS_DIR / "multi_source_agreement" / f"{base.perturbation_id}__{base.raw_variant_label}__sample{sample_index:03d}_h{horizon_s}.npz",
                    agreement=agreement_map.numpy().astype(np.float32),
                    agreement_source_count=np.int16(len(available)),
                    uses_gt=np.bool_(False),
                    uses_future_info=np.bool_(False),
                    uses_current_clean_same_frame=np.bool_(False),
                )
                raw_case = source_cases[base.raw_variant_label][horizon_s]
                raw_semantic = raw_case["semantic"]
                raw_occ = raw_semantic != EMPTY_IDX
                native_case = native_cases[horizon_s]
                native_occ = native_case["occ_mask"]
                raw_delta = raw_occ & ~native_occ
                if horizon_s == 6:
                    np.savez_compressed(
                        ARTIFACTS_DIR / "final_outputs" / f"{base.perturbation_id}__{base.raw_variant_label}__rawstate__sample{sample_index:03d}_h{horizon_s}.npz",
                        native_semantic=native_case["semantic"].numpy().astype(np.int16),
                        raw_semantic=raw_semantic.numpy().astype(np.int16),
                        raw_occ_conf=raw_case["occ_conf"].numpy().astype(np.float32),
                        raw_margin=raw_case["top1_margin"].numpy().astype(np.float32),
                        agreement=agreement_map.numpy().astype(np.float32),
                        gt_h=raw_case["gt_h"].numpy().astype(np.int16),
                        gt0=raw_case["gt0"].numpy().astype(np.int16),
                    )
                for variant in fix_variants_for(base):
                    protected = protected_zone_fix(variant.protected_label, raw_occ, raw_delta, raw_case["occ_conf"], raw_case["top1_margin"], agreement_map, sectors, horizon_s)
                    protected_agreement_mean = float(agreement_map[protected].mean().item()) if protected.any() else 0.0
                    protected_conf_mean = float(raw_case["occ_conf"][protected].mean().item()) if protected.any() else 0.0
                    protected_rows.append(
                        {
                            "sample_index": sample_index,
                            "perturbation_id": base.perturbation_id,
                            "base_repair_variant": base.raw_variant_label,
                            "variant_label": variant.label,
                            "horizon_s": horizon_s,
                            "protected_voxel_count": int(protected.sum().item()),
                            "protected_raw_delta_ratio": safe_div(float((protected & raw_delta).sum().item()), max(1.0, float(raw_delta.sum().item()))),
                            "protected_confidence_mean": protected_conf_mean,
                            "protected_agreement_mean": protected_agreement_mean,
                            "protected_front_sector_ratio": safe_div(float((protected & sectors["front"].bool()).sum().item()), max(1.0, float(protected.sum().item()))),
                            "agreement_source_count": len(available),
                            "uses_gt": False,
                        }
                    )
                    if variant.label in {"F0_raw_repair", "F9_C4_preserve_R5"}:
                        final_semantic = raw_semantic.clone()
                        pruned = torch.zeros_like(raw_occ)
                        score = torch.zeros_like(raw_case["occ_conf"])
                        budget_meta = {
                            "budget_mode": variant.budget_mode,
                            "native_occ_count": int(native_occ.sum().item()),
                            "raw_occ_count": int(raw_occ.sum().item()),
                            "raw_delta_count": int(raw_delta.sum().item()),
                            "target_final_occ_count": int(raw_occ.sum().item()),
                            "target_expansion_ratio": variant.expansion_ratio,
                            "target_keep_ratio": variant.keep_ratio,
                            "final_occ_count": int(raw_occ.sum().item()),
                            "no_gt_budget": True,
                        }
                    else:
                        score = low_value_score(raw_occ, protected, raw_case["occ_conf"], raw_case["top1_margin"], agreement_map, sectors, horizon_s, variant.wrong_class_aware, variant.front_bias)
                        final_semantic, pruned, budget_meta = apply_pruning_no_gt(
                            raw_semantic,
                            protected,
                            score,
                            int(native_occ.sum().item()),
                            int(raw_occ.sum().item()),
                            int(raw_delta.sum().item()),
                            variant.budget_mode,
                            expansion_ratio=variant.expansion_ratio,
                            keep_ratio=variant.keep_ratio,
                        )
                    final_occ = final_semantic != EMPTY_IDX
                    kept_delta = final_occ & ~native_occ
                    front_native = front_counts(native_occ, sectors)
                    front_raw = front_counts(raw_occ, sectors)
                    front_final = front_counts(final_occ, sectors)
                    front_raw_delta = front_counts(raw_delta, sectors)
                    front_kept_delta = front_counts(kept_delta, sectors)
                    front_pruned = front_counts(pruned, sectors)
                    pruned_from_protected_ratio = safe_div(float((pruned & protected).sum().item()), max(1.0, float(pruned.sum().item())))
                    pruned_from_nonprotected_ratio = safe_div(float((pruned & ~protected).sum().item()), max(1.0, float(pruned.sum().item())))
                    nonfront_pruning_ratio = safe_div(float((pruned & ~sectors["front"].bool()).sum().item()), max(1.0, float(pruned.sum().item())))
                    protected_preservation_ratio = 1.0 - safe_div(float((protected & pruned).sum().item()), max(1.0, float(protected.sum().item())))
                    front_kept_ratio = safe_div(float((sectors["front"].bool() & final_occ).sum().item()), max(1.0, float((sectors["front"].bool() & raw_occ).sum().item())))
                    eval_row = build_eval_row(final_semantic, raw_case["gt_h"], raw_case["gt0"], base.perturbation_id, horizon_s, sectors, native_case["semantic"])
                    eval_row.update(
                        {
                            "sample_index": sample_index,
                            "base_label": base.label,
                            "perturbation_id": base.perturbation_id,
                            "base_repair_variant": base.raw_variant_label,
                            "variant_label": variant.label,
                            "horizon_s": horizon_s,
                            "budget_mode": budget_meta["budget_mode"],
                            "expansion_ratio": budget_meta["target_expansion_ratio"],
                            "keep_ratio": budget_meta["target_keep_ratio"],
                            "native_occ_count": budget_meta["native_occ_count"],
                            "raw_occ_count": budget_meta["raw_occ_count"],
                            "final_occ_count": budget_meta["final_occ_count"],
                            "raw_delta_count": budget_meta["raw_delta_count"],
                            "final_native_expansion_ratio": safe_div(budget_meta["final_occ_count"] - budget_meta["native_occ_count"], max(1, budget_meta["native_occ_count"])),
                            "raw_delta_keep_ratio": safe_div(int(kept_delta.sum().item()), max(1, budget_meta["raw_delta_count"])),
                            "protected_zone_preservation_ratio": protected_preservation_ratio,
                            "pruning_from_protected_ratio": pruned_from_protected_ratio,
                            "pruning_from_nonprotected_ratio": pruned_from_nonprotected_ratio,
                            "nonfront_pruning_ratio": nonfront_pruning_ratio,
                            "front_kept_ratio": front_kept_ratio,
                            "front_local_density_proxy": safe_div(front_final, max(1, front_native)),
                            "agreement_source_count": len(available),
                            "protected_agreement_mean": protected_agreement_mean,
                            "protected_confidence_mean": protected_conf_mean,
                            "low_value_removed_mean": float(score[pruned].mean().item()) if pruned.any() else 0.0,
                            "no_gt_budget": True,
                            "uses_gt_budget": False,
                            "uses_gt_repair": False,
                            "uses_pred_gt_density_for_selection": False,
                            "selection_uses_gt": False,
                        }
                    )
                    metric_rows.append(eval_row)
                    replay_rows.append(
                        {
                            "sample_index": sample_index,
                            "base_label": base.label,
                            "perturbation_id": base.perturbation_id,
                            "base_repair_variant": base.raw_variant_label,
                            "variant_label": variant.label,
                            "horizon_s": horizon_s,
                            "no_gt_budget": True,
                            "budget_mode": budget_meta["budget_mode"],
                            "expansion_ratio": budget_meta["target_expansion_ratio"],
                            "keep_ratio": budget_meta["target_keep_ratio"],
                            "native_occ_count": budget_meta["native_occ_count"],
                            "raw_occ_count": budget_meta["raw_occ_count"],
                            "final_occ_count": budget_meta["final_occ_count"],
                            "raw_delta_count": budget_meta["raw_delta_count"],
                            "final_native_expansion_ratio": safe_div(budget_meta["final_occ_count"] - budget_meta["native_occ_count"], max(1, budget_meta["native_occ_count"])),
                            "raw_delta_keep_ratio": safe_div(int(kept_delta.sum().item()), max(1, budget_meta["raw_delta_count"])),
                            "protected_zone_preservation_ratio": protected_preservation_ratio,
                            "pruning_from_protected_ratio": pruned_from_protected_ratio,
                            "nonfront_pruning_ratio": nonfront_pruning_ratio,
                            "front_local_density_proxy": safe_div(front_final, max(1, front_native)),
                            "agreement_source_count": len(available),
                            "selection_uses_gt": False,
                            "uses_gt_repair": False,
                            "uses_gt_budget": False,
                            "uses_pred_gt_density_for_selection": False,
                        }
                    )
                    front_gt_occ = int(((raw_case["gt_h"] != EMPTY_IDX) & sectors["front"].bool()).sum().item())
                    native_front_ff = safe_div(float(((raw_case["gt_h"] != EMPTY_IDX) & sectors["front"].bool() & ~native_occ).sum().item()), max(1, front_gt_occ))
                    final_front_ff = safe_div(float(((raw_case["gt_h"] != EMPTY_IDX) & sectors["front"].bool() & ~final_occ).sum().item()), max(1, front_gt_occ))
                    native_front_fp = safe_div(float(((raw_case["gt_h"] == EMPTY_IDX) & sectors["front"].bool() & native_occ).sum().item()), max(1, int(((raw_case["gt_h"] == EMPTY_IDX) & sectors["front"].bool()).sum().item())))
                    final_front_fp = safe_div(float(((raw_case["gt_h"] == EMPTY_IDX) & sectors["front"].bool() & final_occ).sum().item()), max(1, int(((raw_case["gt_h"] == EMPTY_IDX) & sectors["front"].bool()).sum().item())))
                    front_audit_rows.append(
                        {
                            "sample_index": sample_index,
                            "base_label": base.label,
                            "variant_label": variant.label,
                            "horizon_s": horizon_s,
                            "front_occ_count_native": front_native,
                            "front_occ_count_raw": front_raw,
                            "front_occ_count_final": front_final,
                            "front_native_expansion_ratio": safe_div(front_final - front_native, max(1, front_native)),
                            "front_raw_delta_keep_ratio": safe_div(front_kept_delta, max(1, front_raw_delta)),
                            "front_protected_keep_ratio": 1.0 - safe_div(float((protected & pruned & sectors["front"].bool()).sum().item()), max(1.0, float((protected & sectors["front"].bool()).sum().item()))),
                            "front_nonprotected_prune_ratio": safe_div(float((pruned & ~protected & sectors["front"].bool()).sum().item()), max(1.0, float((pruned & sectors["front"].bool()).sum().item()))),
                            "front_gt_occupied_count": front_gt_occ,
                            "front_pred_gt_density_delta": safe_div(front_final, max(1, front_gt_occ)) - safe_div(front_native, max(1, front_gt_occ)),
                            "front_false_positive_delta": final_front_fp - native_front_fp,
                            "front_false_free_delta": final_front_ff - native_front_ff,
                        }
                    )
                    sample_improve_rows.append(
                        {
                            "sample_index": sample_index,
                            "base_label": base.label,
                            "variant_label": variant.label,
                            "horizon_s": horizon_s,
                            "front_false_free_improved": final_front_ff < native_front_ff,
                            "density_safe_gt_eval": float(eval_row["pred_gt_density_delta"]) <= (0.18 if "A10" in base.label else 0.18),
                            "prediction_expansion_safe": float(eval_row["final_native_expansion_ratio"]) <= (0.20 if "A10" in base.label else 0.16),
                            "joint_success": (final_front_ff < native_front_ff) and (float(eval_row["pred_gt_density_delta"]) <= (0.18 if "A10" in base.label else 0.18)),
                        }
                    )
                    if sample_index == 0 and horizon_s == 6 and variant.label in {"F0_raw_repair", "F2_expand_budget_medium", "F7_wrong_class_aware_medium", "F8_front_budget_reallocation_no_gt", "F9_C4_preserve_R5"}:
                        fp = ARTIFACTS_DIR / "final_outputs" / f"{base.perturbation_id}__{base.raw_variant_label}__{variant.label}__sample{sample_index:03d}_h{horizon_s}.npz"
                        np.savez_compressed(
                            fp,
                            gt_h=raw_case["gt_h"].numpy().astype(np.int16),
                            native_semantic=native_case["semantic"].numpy().astype(np.int16),
                            raw_semantic=raw_semantic.numpy().astype(np.int16),
                            final_semantic=final_semantic.numpy().astype(np.int16),
                            protected_zone=protected.numpy().astype(np.uint8),
                            pruned_mask=pruned.numpy().astype(np.uint8),
                        )
                        visual_cases[(base.label, variant.label)] = {
                            "gt_h": raw_case["gt_h"],
                            "native": native_case["semantic"],
                            "raw": raw_semantic,
                            "final": final_semantic,
                            "protected": protected,
                            "pruned": pruned,
                        }
                        np.savez_compressed(
                            ARTIFACTS_DIR / "pruning_scores" / f"{base.perturbation_id}__{base.raw_variant_label}__{variant.label}__sample{sample_index:03d}_h{horizon_s}.npz",
                            low_value_score=score.numpy().astype(np.float32),
                            no_gt=np.bool_(True),
                        )
                del raw_semantic, raw_occ, native_occ, raw_delta
            del native_cases, source_cases, sample_unwrapped
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_rows = summarize_with_native(metric_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_no_gt_density_metrics.csv", metric_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_no_gt_density_aggregate_metrics.csv", aggregate_rows(metric_rows, ["base_label", "perturbation_id", "base_repair_variant", "variant_label", "horizon_s"]))
    write_csv(REPORTS_DIR / "sw13c_fix_metric_summary.csv", summary_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_replay_manifest.csv", replay_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_protected_zone_metrics.csv", protected_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_multi_source_agreement_manifest.csv", agreement_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_front_local_density_audit.csv", front_audit_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_sample_consistency.csv", sample_improve_rows)
    write_md(
        REPORTS_DIR / "sw13c_fix_agreement_availability.md",
        "\n".join(
            [
                "# SW-13C-Fix agreement availability",
                "",
                "- A1 requires `{R1,R2,R3}`.",
                "- A10 requires `{R1,R3,R4,R8}`.",
                "- C4 requires `{R5,R6,R7}`.",
                "- source_count < 2 would invalidate true multi-source agreement claims.",
            ]
        ),
    )
    write_md(
        REPORTS_DIR / "sw13c_fix_metric_definition.md",
        "\n".join(
            [
                "# SW-13C-Fix metric definition",
                "",
                "- GT metrics are evaluation-only after final prediction generation.",
                "- prediction-side safety metrics are `final_native_expansion_ratio`, `raw_delta_keep_ratio`, `protected_zone_preservation_ratio`, `pruning_from_protected_ratio`, `nonfront_pruning_ratio`, `agreement_source_count`, `protected_agreement_mean`, `protected_confidence_mean`, `front_kept_ratio`, and `front_local_density_proxy`.",
            ]
        ),
    )
    write_json(
        REPORTS_DIR / "sw13c_fix_success_criteria.json",
        {
            "A1_strong": "front delta <= -0.20 and future delta <= -0.06 and pred_gt_density_delta <= +0.15 and final_native_expansion_ratio <= 0.16",
            "A10_strong": "front delta <= -0.25 and A10 recovery delta >= +0.15 and future delta <= -0.08 and pred_gt_density_delta <= +0.18 and final_native_expansion_ratio <= 0.20",
            "selection_uses_gt": False,
        },
    )

    selected = select_candidates_no_gt(summary_rows)
    selection_payload = {
        "selection_uses_gt": False,
        "selected": selected,
        "candidate_selection_inputs": [
            "no_gt_budget",
            "protected_zone_preservation_ratio",
            "pruning_from_protected_ratio",
            "agreement_source_count",
            "final_native_expansion_ratio",
            "raw_delta_keep_ratio",
            "front_kept_ratio",
            "protected_confidence_mean",
            "protected_agreement_mean",
            "low_value_removed_mean",
        ],
        "forbidden_inputs_checked": [
            "pred_gt_density_delta",
            "false_positive_delta",
            "front_sector_false_free_rate_delta_vs_native",
            "A10_front_h6_recovery_rate_delta_vs_native",
            "occupied_iou",
            "semantic_miou",
        ],
        "no_pred_gt_density_delta_used": True,
        "no_gt_occ_count_used": True,
    }
    write_json(REPORTS_DIR / "sw13c_fix_no_gt_candidate_selection.json", selection_payload)

    selected_h6 = {(row["base_label"], row["variant_label"]): row for row in summary_rows if int(row["horizon_s"]) == 6}
    a1 = selected_h6.get(("A1_base", selected.get("A1_base", {}).get("variant_label", "")))
    a10 = selected_h6.get(("A10_base_1", selected.get("A10_base_1", {}).get("variant_label", "")))
    if a10 is None:
        a10 = selected_h6.get(("A10_base_2", selected.get("A10_base_2", {}).get("variant_label", "")))
    c4 = selected_h6.get(("C4_base", "F9_C4_preserve_R5"))

    def a1_strong(row: dict[str, Any] | None) -> bool:
        return row is not None and float(row["agreement_source_count"]) >= 2 and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.20 and float(row["future_h4_h6_false_free_rate_delta_vs_native"]) <= -0.06 and float(row["pred_gt_density_delta"]) <= 0.15 and float(row["final_native_expansion_ratio"]) <= 0.16 and float(row["protected_zone_preservation_ratio"]) >= 0.80 and float(row["pruning_from_protected_ratio"]) <= 0.05

    def a1_medium(row: dict[str, Any] | None) -> bool:
        return row is not None and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.12 and float(row["pred_gt_density_delta"]) <= 0.18 and float(row["final_native_expansion_ratio"]) <= 0.20 and float(row["protected_zone_preservation_ratio"]) >= 0.70

    def a10_strong(row: dict[str, Any] | None) -> bool:
        return row is not None and float(row["agreement_source_count"]) >= 2 and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.25 and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.15 and float(row["future_h4_h6_false_free_rate_delta_vs_native"]) <= -0.08 and float(row["pred_gt_density_delta"]) <= 0.18 and float(row["final_native_expansion_ratio"]) <= 0.20 and float(row.get("wrong_class_reduction_ratio_vs_raw", 0.0)) >= 0.20 and float(row["protected_zone_preservation_ratio"]) >= 0.80 and float(row["pruning_from_protected_ratio"]) <= 0.05

    def a10_medium(row: dict[str, Any] | None) -> bool:
        return row is not None and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.15 and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.08 and float(row["pred_gt_density_delta"]) <= 0.22 and float(row["final_native_expansion_ratio"]) <= 0.24 and float(row["protected_zone_preservation_ratio"]) >= 0.70

    front_local_risk = False
    if a1 is not None and float(a1.get("front_local_density_proxy", 0.0)) > 2.2:
        front_local_risk = True
    if a10 is not None and float(a10.get("front_local_density_proxy", 0.0)) > 2.4:
        front_local_risk = True

    # Ablations on selected candidates at h6 only.
    ablation_rows: list[dict[str, Any]] = []
    for base_label, selected_row in [("A1_base", a1), ("A10_base_1" if a10 and a10["base_label"] == "A10_base_1" else "A10_base_2", a10)]:
        if selected_row is None:
            continue
        base_repair_variant = selected_row["base_repair_variant"]
        for sample_index in sample_ids:
            raw_state = np.load(ARTIFACTS_DIR / "final_outputs" / f"{selected_row['perturbation_id']}__{base_repair_variant}__rawstate__sample{sample_index:03d}_h6.npz")
            native_case = {
                "semantic": torch.from_numpy(raw_state["native_semantic"]).long(),
                "occ_mask": torch.from_numpy(raw_state["native_semantic"]).long() != EMPTY_IDX,
            }
            raw_case = {
                "semantic": torch.from_numpy(raw_state["raw_semantic"]).long(),
                "occ_mask": torch.from_numpy(raw_state["raw_semantic"]).long() != EMPTY_IDX,
                "delta_occ_mask": (torch.from_numpy(raw_state["raw_semantic"]).long() != EMPTY_IDX) & ~native_case["occ_mask"],
                "occ_conf": torch.from_numpy(raw_state["raw_occ_conf"]).float(),
                "top1_margin": torch.from_numpy(raw_state["raw_margin"]).float(),
                "gt_h": torch.from_numpy(raw_state["gt_h"]).long(),
                "gt0": torch.from_numpy(raw_state["gt0"]).long(),
            }
            agreement_map = torch.from_numpy(raw_state["agreement"]).float()
            selected_variant = next(v for b in base_specs() if b.label == base_label for v in fix_variants_for(b) if v.label == selected_row["variant_label"])
            protected = protected_zone_fix(selected_variant.protected_label, raw_case["occ_mask"], raw_case["delta_occ_mask"], raw_case["occ_conf"], raw_case["top1_margin"], agreement_map, sectors, 6)
            score = low_value_score(raw_case["occ_mask"], protected, raw_case["occ_conf"], raw_case["top1_margin"], agreement_map, sectors, 6, selected_variant.wrong_class_aware, selected_variant.front_bias)
            selected_final, selected_prune, meta = apply_pruning_no_gt(raw_case["semantic"], protected, score, int(native_case["occ_mask"].sum().item()), int(raw_case["occ_mask"].sum().item()), int(raw_case["delta_occ_mask"].sum().item()), selected_variant.budget_mode, selected_variant.expansion_ratio, selected_variant.keep_ratio)
            prune_count = int(selected_prune.sum().item())
            # random non-front
            random_metrics: list[dict[str, Any]] = []
            candidates = torch.nonzero(raw_case["occ_mask"] & ~protected & ~sectors["front"].bool(), as_tuple=False)
            for seed in [0, 1, 2]:
                g = torch.Generator().manual_seed(seed + sample_index)
                pick = min(prune_count, candidates.shape[0])
                idx = torch.randperm(candidates.shape[0], generator=g)[:pick] if pick > 0 else torch.zeros((0,), dtype=torch.long)
                prune = torch.zeros_like(raw_case["occ_mask"])
                if pick > 0:
                    sel = candidates[idx]
                    prune[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                final = raw_case["semantic"].clone()
                final[prune] = EMPTY_IDX
                row = build_eval_row(final, raw_case["gt_h"], raw_case["gt0"], selected_row["perturbation_id"], 6, sectors, native_case["semantic"])
                random_metrics.append(row)
            random_front_delta = np.mean([float(r["front_sector_false_free"]) for r in random_metrics])
            random_density = np.mean([float(r["pred_gt_density_delta"]) for r in random_metrics])
            # low confidence only
            conf_score = (1.0 - torch.clamp(raw_case["occ_conf"], 0, 1))
            conf_score[~raw_case["occ_mask"]] = -1e6
            conf_final, conf_prune, _ = apply_pruning_no_gt(raw_case["semantic"], torch.zeros_like(protected), conf_score, int(native_case["occ_mask"].sum().item()), int(raw_case["occ_mask"].sum().item()), int(raw_case["delta_occ_mask"].sum().item()), selected_variant.budget_mode, selected_variant.expansion_ratio, selected_variant.keep_ratio)
            conf_row = build_eval_row(conf_final, raw_case["gt_h"], raw_case["gt0"], selected_row["perturbation_id"], 6, sectors, native_case["semantic"])
            sel_row = build_eval_row(selected_final, raw_case["gt_h"], raw_case["gt0"], selected_row["perturbation_id"], 6, sectors, native_case["semantic"])
            ablation_rows.append(
                {
                    "base_label": base_label,
                    "sample_index": sample_index,
                    "selected_variant": selected_row["variant_label"],
                    "selected_front_sector_false_free": sel_row["front_sector_false_free"],
                    "selected_pred_gt_density_delta": sel_row["pred_gt_density_delta"],
                    "selected_wrong_class_delta": sel_row["wrong_class_activation_delta"],
                    "selected_protected_preservation": 1.0 - safe_div(float((protected & selected_prune).sum().item()), max(1.0, float(protected.sum().item()))),
                    "random_nonfront_front_sector_false_free_mean": random_front_delta,
                    "random_nonfront_pred_gt_density_delta_mean": random_density,
                    "low_confidence_only_front_sector_false_free": conf_row["front_sector_false_free"],
                    "low_confidence_only_pred_gt_density_delta": conf_row["pred_gt_density_delta"],
                    "low_confidence_only_wrong_class_delta": conf_row["wrong_class_activation_delta"],
                }
            )
    write_csv(REPORTS_DIR / "sw13c_fix_pruning_ablation_metrics.csv", ablation_rows)
    ablation_selected_better = True
    if ablation_rows:
        for base_label in {"A1_base", "A10_base_1", "A10_base_2"}:
            sub = [row for row in ablation_rows if row["base_label"] == base_label]
            if not sub:
                continue
            selected_front = float(np.mean([float(row["selected_front_sector_false_free"]) for row in sub]))
            random_front = float(np.mean([float(row["random_nonfront_front_sector_false_free_mean"]) for row in sub]))
            if selected_front >= random_front:
                ablation_selected_better = False

    # Figures.
    for pert, fname, ykey in [
        ("A1_drop_cam_front", "sw13c_fix_A1_no_gt_recovery_density_pareto.png", "front_sector_false_free_rate_delta_vs_native"),
        ("A10_drop_front_triplet", "sw13c_fix_A10_no_gt_recovery_density_pareto.png", "A10_front_h6_recovery_rate_delta_vs_native"),
    ]:
        rows = [r for r in summary_rows if r["perturbation_id"] == pert and int(r["horizon_s"]) == 6 and r["variant_label"].startswith("F")]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([float(r.get("pred_gt_density_delta", 0)) for r in rows], [abs(float(r.get(ykey, 0))) for r in rows], alpha=0.8)
        ax.set_xlabel("pred_gt_density_delta (evaluation only, not used for selection)")
        ax.set_ylabel(ykey)
        ax.set_title(f"SW-13C-Fix no-GT density budget subset diagnostic {pert}")
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / fname, dpi=180, bbox_inches="tight")
        plt.close(fig)
    rows_h6 = [r for r in summary_rows if int(r["horizon_s"]) == 6 and r["variant_label"].startswith("F")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([float(r["final_native_expansion_ratio"]) for r in rows_h6], [float(r["raw_delta_keep_ratio"]) for r in rows_h6], label="raw_delta_keep_ratio")
    ax.scatter([float(r["final_native_expansion_ratio"]) for r in rows_h6], [float(r["front_kept_ratio"]) for r in rows_h6], label="front_kept_ratio")
    ax.set_xlabel("final_native_expansion_ratio")
    ax.set_ylabel("prediction-side budget ratios")
    ax.set_title("SW-13C-Fix prediction-side budget tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "sw13c_fix_prediction_side_budget_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    audit_h6 = [r for r in front_audit_rows if r["horizon_s"] == 6 and r["variant_label"] in {selected.get("A1_base", {}).get("variant_label", ""), selected.get("A10_base_1", {}).get("variant_label", ""), "F9_C4_preserve_R5"}]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(audit_h6))
    ax.bar(xs - 0.2, [float(r["front_native_expansion_ratio"]) for r in audit_h6], width=0.2, label="front_native_expansion_ratio")
    ax.bar(xs, [float(r["front_raw_delta_keep_ratio"]) for r in audit_h6], width=0.2, label="front_raw_delta_keep_ratio")
    ax.bar(xs + 0.2, [float(r["front_pred_gt_density_delta"]) for r in audit_h6], width=0.2, label="front_pred_gt_density_delta")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['base_label']}:{r['sample_index']}" for r in audit_h6], rotation=75, fontsize=7)
    ax.set_title("SW-13C-Fix front-local density audit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sw13c_fix_front_local_density_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r['base_label']}:{r['sample_index']}" for r in ablation_rows]
    ax.bar(np.arange(len(ablation_rows)) - 0.2, [float(r["selected_pred_gt_density_delta"]) for r in ablation_rows], width=0.2, label="selected")
    ax.bar(np.arange(len(ablation_rows)), [float(r["random_nonfront_pred_gt_density_delta_mean"]) for r in ablation_rows], width=0.2, label="random_nonfront")
    ax.bar(np.arange(len(ablation_rows)) + 0.2, [float(r["low_confidence_only_pred_gt_density_delta"]) for r in ablation_rows], width=0.2, label="low_conf_only")
    ax.set_xticks(np.arange(len(ablation_rows)))
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_title("SW-13C-Fix pruning ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sw13c_fix_pruning_ablation_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if a1 is not None:
        payload = visual_cases.get(("A1_base", a1["variant_label"]))
        if payload:
            save_bev(FIGURES_DIR / "sw13c_fix_bev_A1_sample0.png", payload["gt_h"], payload["native"], payload["raw"], payload["final"], payload["protected"], payload["pruned"], "SW-13C-Fix no-GT density budget no training no GT repair subset diagnostic")
    if a10 is not None:
        payload = visual_cases.get((a10["base_label"], a10["variant_label"]))
        if payload:
            save_bev(FIGURES_DIR / "sw13c_fix_bev_A10_sample0.png", payload["gt_h"], payload["native"], payload["raw"], payload["final"], payload["protected"], payload["pruned"], "SW-13C-Fix no-GT density budget no training no GT repair subset diagnostic")

    if (a1_strong(a1) or a10_strong(a10)) and not front_local_risk and ablation_selected_better:
        decision_type = "V1_STRONG_NO_GT_DENSITY_NEUTRAL_RECOVERY"
    elif a1_medium(a1) or a10_medium(a10):
        decision_type = "V2_MEDIUM_NO_GT_DENSITY_NEUTRAL_RECOVERY"
    elif (a1 is not None and float(a1["agreement_source_count"]) < 2) or (a10 is not None and float(a10["agreement_source_count"]) < 2):
        decision_type = "V5_AGREEMENT_NOT_TRUE_MULTISOURCE"
    elif selection_payload["no_gt_occ_count_used"] is not True or selection_payload["no_pred_gt_density_delta_used"] is not True:
        decision_type = "V7_ORACLE_RISK_REMAINS"
    elif a1 is None and a10 is None:
        decision_type = "V6_ORIGINAL_SW13C_ONLY_DIAGNOSTIC"
    elif front_local_risk:
        decision_type = "V3_NO_GT_RECOVERY_RETAINED_BUT_DENSITY_HIGH"
    else:
        decision_type = "V4_NO_GT_DENSITY_CONTROLLED_BUT_RECOVERY_LOST"

    scaling_summary = "eval_core_50 not run in this pass; eval_core_20 selected candidates are frozen for the next scaling step."
    write_md(REPORTS_DIR / "sw13c_fix_eval_core50_scaling_summary.md", scaling_summary)

    front_local_md = "\n".join(
        [
            "# SW-13C-Fix front-local density audit",
            "",
            "- global density was not the only audit target.",
            "- front-local native/raw/final occupied counts were tracked per sample.",
            f"- front-local density risk remains: {front_local_risk}",
        ]
    )
    write_md(REPORTS_DIR / "sw13c_fix_front_local_density_audit.md", front_local_md)

    decision = {
        "decision_type": decision_type,
        "selected_A1_candidate": a1,
        "selected_A10_candidate": a10,
        "selected_C4_candidate": c4,
        "selection_uses_gt": False,
        "no_pred_gt_density_delta_used": True,
        "no_gt_occ_count_used": True,
        "front_local_density_risk": front_local_risk,
    }
    write_json(REPORTS_DIR / "sw13c_fix_no_gt_density_budget_decision.json", decision)
    write_md(REPORTS_DIR / "sw13c_fix_no_gt_density_budget_decision.md", json.dumps(normalize_export(decision), indent=2, ensure_ascii=False) + "\n")

    report = "\n".join(
        [
            "# Stage SW-13C-Fix No-GT Density Budget",
            "",
            "1. Executive summary",
            f"- decision: {decision_type}",
            "",
            "2. Why SW-13C original needed fixing",
            "- original SW-13C had GT-density-budget risk and GT-derived selection risk.",
            "",
            "3. Code-level fixes",
            "- SW-13C-Fix removes gt_occ_count from pruning target.",
            "- SW-13C-Fix removes pred_gt_density_delta from selection.",
            "",
            "4. No-GT density budget",
            "- prediction-side budgets use native expansion ratio or raw delta keep ratio only.",
            "",
            "5. True multi-source temporal agreement",
            "- A1 uses R1/R2/R3, A10 uses R1/R3/R4/R8, C4 uses R5/R6/R7.",
            "",
            "6. Protected zone and pruning pool",
            "- protected zones remain prediction-side only.",
            "",
            "7. Prediction-side candidate selection",
            "- GT metrics are evaluation-only and not used for selection.",
            "",
            "8. A1 eval_core20 results",
            f"- selected: {a1['variant_label'] if a1 else 'none'}",
            "",
            "9. A10 eval_core20 results",
            f"- selected: {a10['variant_label'] if a10 else 'none'}",
            "",
            "10. C4 preservation",
            "- preserve baseline retained.",
            "",
            "11. Front-local density audit",
            f"- front-local density risk remains: {front_local_risk}",
            "",
            "12. Random pruning ablation",
            "- selected candidate is compared against random non-front pruning and low-confidence-only pruning.",
            "",
            "13. Optional eval_core50 scaling",
            f"- {scaling_summary}",
            "",
            "14. Decision V1-V8",
            f"- {decision_type}",
            "",
            "15. Safe claims",
            "- original SW-13C had GT-density-budget risk",
            "- SW-13C-Fix removes gt_occ_count from pruning target",
            "- SW-13C-Fix removes pred_gt_density_delta from selection",
            "- GT metrics are evaluation-only",
            "- no training",
            "- no checkpoint modification",
            "- no get_occ modification",
            "- no GT repair",
            "- no future-frame feature",
            "- no current clean same-frame feature",
            "- subset diagnostic only",
            "- not official benchmark",
            "",
            "16. Limitations",
            "- eval_core50 scaling was not executed in this pass.",
            "",
            "17. Next unique action",
            "- run fixed eval_core50 scaling with the frozen eval_core20-selected no-GT candidate only if this decision is V1 or V2.",
            "",
        ]
    )
    write_md(REPORTS_DIR / "stage_sw13c_fix_no_gt_density_budget_report.md", report)
    write_json(REPORTS_DIR / "stage_sw13c_fix_no_gt_density_budget_report.json", {"decision_type": decision_type, "selection_uses_gt": False, "no_gt_budget": True})

    tests = {
        "test_outputs_exist.py": """from pathlib import Path\nBASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget')\ndef test_outputs_exist():\n    for n in ['sw13c_fix_inherited_sw13c_audit.json','sw13c_fix_no_gt_density_metrics.csv','sw13c_fix_metric_summary.csv','sw13c_fix_no_gt_candidate_selection.json','sw13c_fix_front_local_density_audit.csv','sw13c_fix_pruning_ablation_metrics.csv','sw13c_fix_no_gt_density_budget_decision.json','stage_sw13c_fix_no_gt_density_budget_report.md']:\n        p=BASE/n\n        assert p.exists() and p.stat().st_size>0, n\n""",
        "test_no_gt_count_in_apply_pruning.py": """from pathlib import Path\nSRC=Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py').read_text()\ndef test_no_gt_count_in_apply_pruning():\n    assert 'def apply_pruning_no_gt' in SRC\n    chunk=SRC.split('def apply_pruning_no_gt',1)[1].split('def build_eval_row',1)[0]\n    assert 'gt_occ_count' not in chunk\n""",
        "test_no_pred_gt_density_in_selection.py": """from pathlib import Path\nimport json\nSRC=Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py').read_text()\nSEL=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_no_gt_candidate_selection.json').read_text())\ndef test_no_pred_gt_density_in_selection():\n    chunk=SRC.split('def select_candidates_no_gt',1)[1].split('def main',1)[0]\n    for bad in ['pred_gt_density_delta','false_positive_delta','front_sector_false_free_rate_delta_vs_native','A10_front_h6_recovery_rate_delta_vs_native','occupied_iou','semantic_miou']:\n        assert bad not in chunk\n    assert SEL['no_pred_gt_density_delta_used'] is True\n""",
        "test_multi_source_agreement_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_multi_source_agreement_manifest.csv').open()))\ndef test_multi_source_agreement_schema():\n    assert rows\n    req={'sample_index','perturbation_id','horizon_s','required_sources','available_sources','agreement_source_count','agreement_mean','missing_sources','uses_gt','uses_future_info','uses_current_clean_same_frame'}\n    assert req.issubset(rows[0])\n""",
        "test_selected_agreement_source_count.py": """import json\nfrom pathlib import Path\nd=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_no_gt_density_budget_decision.json').read_text())\ndef test_selected_agreement_source_count():\n    if d['decision_type'] in {'V1_STRONG_NO_GT_DENSITY_NEUTRAL_RECOVERY','V2_MEDIUM_NO_GT_DENSITY_NEUTRAL_RECOVERY'}:\n        assert float(d['selected_A1_candidate']['agreement_source_count']) >= 2\n        assert float(d['selected_A10_candidate']['agreement_source_count']) >= 2\n""",
        "test_no_gt_budget_flags.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_no_gt_density_metrics.csv').open()))\ndef test_no_gt_budget_flags():\n    assert rows\n    assert all(r['uses_gt_budget'] == 'False' for r in rows)\n""",
        "test_front_local_density_audit_exists.py": """from pathlib import Path\ndef test_front_local_density_audit_exists():\n    p=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_front_local_density_audit.csv')\n    assert p.exists() and p.stat().st_size>0\n""",
        "test_pruning_ablation_exists.py": """from pathlib import Path\ndef test_pruning_ablation_exists():\n    p=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_pruning_ablation_metrics.csv')\n    assert p.exists() and p.stat().st_size>0\n""",
        "test_metric_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_metric_summary.csv').open()))\ndef test_metric_schema():\n    assert rows\n    req={'no_gt_budget','budget_mode','native_occ_count','raw_occ_count','final_occ_count','raw_delta_count','final_native_expansion_ratio','raw_delta_keep_ratio','protected_zone_preservation_ratio','pruning_from_protected_ratio','nonfront_pruning_ratio','agreement_source_count','selection_uses_gt'}\n    assert req.issubset(rows[0])\n""",
        "test_decision_schema.py": """import json\nfrom pathlib import Path\nd=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_no_gt_density_budget_decision.json').read_text())\ndef test_decision_schema():\n    assert d['decision_type'] in {'V1_STRONG_NO_GT_DENSITY_NEUTRAL_RECOVERY','V2_MEDIUM_NO_GT_DENSITY_NEUTRAL_RECOVERY','V3_NO_GT_RECOVERY_RETAINED_BUT_DENSITY_HIGH','V4_NO_GT_DENSITY_CONTROLLED_BUT_RECOVERY_LOST','V5_AGREEMENT_NOT_TRUE_MULTISOURCE','V6_ORIGINAL_SW13C_ONLY_DIAGNOSTIC','V7_ORACLE_RISK_REMAINS','V8_IMPLEMENTATION_BLOCKED'}\n""",
        "test_no_false_claims.py": """from pathlib import Path\ntext=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/stage_sw13c_fix_no_gt_density_budget_report.md').read_text().lower()\ndef test_no_false_claims():\n    assert 'not official benchmark' in text\n    assert 'trained model improvement' not in text\n    assert 'original sw-13c had gt-density-budget risk' in text\n""",
    }
    for name, content in tests.items():
        write_md(TESTS_DIR / name, content)


if __name__ == "__main__":
    main()
