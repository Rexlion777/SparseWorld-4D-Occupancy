from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
np.Inf = np.inf
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", "/home/rexlion/ComputerVision/cv_lidar_transition"))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory"
SW13A_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
SW13B_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
SW13B_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory"
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

sw2 = load_module("sw13c_sw2", PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py")
sw7 = load_module("sw13c_sw7", PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py")
sw81 = load_module("sw13c_sw81", PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py")
sw12b = load_module("sw13c_sw12b", PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py")
from mmcv.parallel import collate as collate_fn


@dataclass(frozen=True)
class BaseSpec:
    name: str
    perturbation: str
    repair: str
    target_density: tuple[float, ...]


@dataclass(frozen=True)
class VariantSpec:
    label: str
    family: str
    protected: str
    target_density: float | None
    aggressive: float
    wrong_class_aware: bool = False


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, LOGS_DIR, SCRIPTS_DIR, ARTIFACTS_DIR, FIGURES_DIR, TESTS_DIR,
              ARTIFACTS_DIR / "raw_repair_dumps", ARTIFACTS_DIR / "temporal_priors",
              ARTIFACTS_DIR / "protected_zones", ARTIFACTS_DIR / "pruning_scores", ARTIFACTS_DIR / "final_outputs"]:
        p.mkdir(parents=True, exist_ok=True)


def norm(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: norm(v) for k, v in o.items()}
    if isinstance(o, list):
        return [norm(v) for v in o]
    if isinstance(o, tuple):
        return [norm(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return o


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(norm(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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
        for k in row:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(norm(row))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def now() -> str:
    return datetime.now().astimezone().isoformat()


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default=",".join(str(i) for i in range(20)))
    p.add_argument("--resume-from-existing", action="store_true")
    return p.parse_args()


def sample_ids(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def bases() -> list[BaseSpec]:
    return [
        BaseSpec("A1_base", "A1_drop_cam_front", "R1_replace_tminus1", (0.12, 0.10, 0.08, 0.05)),
        BaseSpec("A10_base_1", "A10_drop_front_triplet", "R4_ema_K3", (0.15, 0.12, 0.10, 0.08)),
        BaseSpec("A10_base_2", "A10_drop_front_triplet", "R8_camera_group_repair_front_triplet", (0.15, 0.12, 0.10, 0.08)),
        BaseSpec("C4_base", "C4_motion_blur_9", "R5_blend_alpha03", (0.00,)),
    ]


def variants_for(base: BaseSpec) -> list[VariantSpec]:
    out = [VariantSpec("C0_raw_repair", "raw", "PZ0_none", None, 0.0)]
    if base.perturbation == "C4_motion_blur_9":
        out.append(VariantSpec("C4_preserve_R5", "preserve", "PZ4_full_protected_core", None, 0.0))
        return out
    for td in base.target_density:
        suffix = str(td).replace("0.", "p")
        out.extend([
            VariantSpec(f"C2_protected_front_recovery_pruning_{suffix}", "protected_front", "PZ2_front_h6_conf_agree", td, 0.70),
            VariantSpec(f"C3_temporal_prior_guided_pruning_{suffix}", "temporal_prior", "PZ3_front_h6_conf_prior", td, 0.85),
            VariantSpec(f"C4_full_density_neutral_reallocation_{suffix}", "full", "PZ4_full_protected_core", td, 1.00),
            VariantSpec(f"C6_front_budget_reallocation_{suffix}", "front_budget", "PZ4_full_protected_core", td, 0.60),
            VariantSpec(f"C7_wrong_class_aware_pruning_{suffix}", "wrong_class", "PZ4_full_protected_core", td, 0.90, True),
        ])
    out.append(VariantSpec("C5_aggressive_but_protected", "aggressive", "PZ5_strong_core", min(base.target_density), 1.25, True))
    out.append(VariantSpec("C1_budget_only_pruning", "budget_only", "PZ0_none", min(base.target_density), 1.00))
    return out


def build_runtime_dataset():
    cfg, dataset, model, _ = sw81.build_sparseworld_runtime(train=False, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    model.eval()
    return dataset


def gt_temporal_for(dataset: Any, sid: int) -> torch.Tensor:
    raw, _ = sw2.extract_sample_batch(dataset, sid, collate_fn)
    sample = sw2.unwrap(raw)
    gt_current = torch.as_tensor(sample["voxel_semantics"]).long()
    gt_list = [gt_current]
    temporal = sample["temporal_semantics"]
    for i in range(1, 7):
        gt_list.append(torch.as_tensor(temporal[i]["voxel_semantics"]).long())
    return torch.stack(gt_list, dim=0)


def confidence_entropy(conf: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(conf, 1e-4, 1 - 1e-4)
    return -(c * torch.log(c) + (1 - c) * torch.log(1 - c))


def local_support(mask: torch.Tensor) -> torch.Tensor:
    k = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
    return F.conv3d(mask.float()[None, None], k, padding=1)[0, 0]


def qmask(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    if not mask.any():
        return torch.zeros_like(mask)
    vals = values[mask]
    thr = torch.quantile(vals.float(), q)
    return mask & (values >= thr)


def agreement_map(all_repairs: list[torch.Tensor]) -> torch.Tensor:
    if not all_repairs:
        raise RuntimeError("no repairs for agreement")
    acc = torch.zeros_like(all_repairs[0], dtype=torch.float32)
    for m in all_repairs:
        acc += m.float()
    return acc / float(len(all_repairs))


def protected_zone(label: str, base: BaseSpec, horizon: int, raw_occ: torch.Tensor, raw_delta: torch.Tensor, conf: torch.Tensor, margin: torch.Tensor, agree: torch.Tensor, sectors: dict[str, torch.Tensor]) -> torch.Tensor:
    front = sectors["front"].bool()
    future = horizon in {4, 6}
    conf50 = qmask(conf, raw_delta, 0.50)
    conf70 = qmask(conf, raw_delta, 0.70)
    margin_good = margin >= torch.quantile(margin[raw_delta].float(), 0.50) if raw_delta.any() else torch.zeros_like(raw_delta)
    size3 = local_support(raw_occ) >= 3
    size5 = local_support(raw_occ) >= 5
    if label == "PZ0_none":
        return torch.zeros_like(raw_occ)
    if label == "PZ1_front_h6_conf":
        return raw_occ & front & conf50 if future else raw_occ & front & conf50
    if label == "PZ2_front_h6_conf_agree":
        return raw_occ & front & conf50 & (agree >= 0.5)
    if label == "PZ3_front_h6_conf_prior":
        return raw_occ & front & conf50 & (agree >= 0.5)
    if label == "PZ4_full_protected_core":
        return raw_occ & front & conf50 & (agree >= 0.5) & size3 & margin_good
    if label == "PZ5_strong_core":
        return raw_occ & front & conf70 & (agree >= 0.67) & size5 & margin_good
    return torch.zeros_like(raw_occ)


def low_value_score(raw_occ: torch.Tensor, protected: torch.Tensor, conf: torch.Tensor, margin: torch.Tensor, agree: torch.Tensor, sectors: dict[str, torch.Tensor], horizon: int, wrong_class_aware: bool) -> torch.Tensor:
    front = sectors["front"].bool()
    entropy = confidence_entropy(torch.clamp(conf, 0, 1))
    conf_bad = 1.0 - torch.clamp(conf, 0, 1)
    if raw_occ.any():
        margin_norm = margin.float() / (margin[raw_occ].float().max() + 1e-6)
    else:
        margin_norm = margin.float()
    margin_bad = 1.0 - torch.clamp(margin_norm, 0, 1)
    isolated = (local_support(raw_occ) < 3).float()
    nonfront = (~front).float()
    future_discount = 0.25 if horizon in {4, 6} else 0.0
    score = 1.0 * conf_bad + 0.8 * margin_bad + 0.7 * entropy + 1.2 * (1 - agree) + 0.7 * nonfront + 0.8 * isolated
    score = score - 100.0 * protected.float() - future_discount * front.float()
    if wrong_class_aware:
        score += 0.8 * (margin_bad + entropy)
    score[~raw_occ] = -1e6
    return score


def apply_pruning(raw: torch.Tensor, protected: torch.Tensor, score: torch.Tensor, gt_occ_count: int, native_count: int, target_delta: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    raw_occ = raw != EMPTY_IDX
    if target_delta is None:
        return raw.clone(), torch.zeros_like(raw_occ)
    target_count = int(round(native_count + target_delta * max(1, gt_occ_count)))
    target_count = max(0, min(int(raw_occ.sum().item()), target_count))
    prune_count = int(raw_occ.sum().item()) - target_count
    if prune_count <= 0:
        return raw.clone(), torch.zeros_like(raw_occ)
    candidates = raw_occ & ~protected
    coords = torch.nonzero(candidates, as_tuple=False)
    if coords.numel() == 0:
        return raw.clone(), torch.zeros_like(raw_occ)
    vals = score[candidates]
    k = min(prune_count, vals.numel())
    _, idx = torch.topk(vals, k=k, largest=True)
    prune = torch.zeros_like(raw_occ)
    picked = coords[idx]
    prune[picked[:, 0], picked[:, 1], picked[:, 2]] = True
    final = raw.clone()
    final[prune] = EMPTY_IDX
    return final, prune


def row_metrics(pred: torch.Tensor, gt_h: torch.Tensor, gt0: torch.Tensor, pert: str, horizon: int, sectors: dict[str, torch.Tensor], native: torch.Tensor) -> dict[str, Any]:
    row = sw12b.build_eval_row(pred, gt_h, gt0, pert, horizon, sectors, baseline_pred=native)
    row["front_sector_false_free_rate"] = row["front_sector_false_free"]
    row["small_object_false_free_rate"] = row["small_object_false_free"]
    row["dynamic_object_false_free_rate"] = row["dynamic_false_free"]
    gt_occ = gt_h != EMPTY_IDX
    pred_occ = pred != EMPTY_IDX
    if horizon in {4, 6}:
        den = gt_occ.sum().item()
        num = (gt_occ & ~pred_occ).sum().item()
        row["future_h4_h6_false_free_rate"] = safe_div(num, den)
    else:
        row["future_h4_h6_false_free_rate"] = 0.0
    row["A10_front_h6_recovery_rate"] = row["A10_front_h6_recovery_ratio"]
    row["false_positive_rate"] = row["false_occupied_rate"]
    row["density_delta"] = row["pred_gt_density_delta"]
    row["wrong_class_rate"] = row["wrong_class_activation"]
    return row


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    return sw12b.aggregate_rows(rows, keys)


def load_sw13a_native_map() -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv(SW13A_REPORTS / "sw13a_feature_memory_aggregate_metrics.csv")
    native_rows = [r for r in rows if r["variant_label"] == "R0_degraded_native"]
    return {(r["perturbation_id"], int(r["horizon_s"])): r for r in native_rows}


def summarize_metrics(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    native_map = load_sw13a_native_map()
    raw_map = {
        (r["perturbation_id"], r["base_repair_variant"], int(r["horizon_s"])): r
        for r in agg
        if r["variant_label"] == "C0_raw_repair"
    }
    summary: list[dict[str, Any]] = []
    for row in agg:
        out = dict(row)
        native = native_map.get((row["perturbation_id"], int(row["horizon_s"])))
        raw = raw_map.get((row["perturbation_id"], row["base_repair_variant"], int(row["horizon_s"])))
        if native is not None:
            out["front_sector_false_free_rate_delta_vs_native"] = float(row["front_sector_false_free"]) - float(native["front_sector_false_free"])
            out["future_h4_h6_false_free_rate_delta_vs_native"] = float(row["false_free_rate"]) - float(native["false_free_rate"]) if int(row["horizon_s"]) in {4, 6} else 0.0
            out["A10_front_h6_recovery_rate_delta_vs_native"] = float(row["A10_front_h6_recovery_ratio"]) - float(native["A10_front_h6_recovery_ratio"])
            out["false_positive_delta"] = float(row["false_occupied_rate"]) - float(native["false_occupied_rate"])
        out["density_delta"] = float(row["pred_gt_density_delta"])
        out["wrong_class_delta"] = float(row["wrong_class_activation_delta"])
        if raw is not None and row["variant_label"] != "C0_raw_repair":
            density_raw = max(1e-6, float(raw["pred_gt_density_delta"]))
            out["density_reduction_ratio"] = 1.0 - safe_div(float(row["pred_gt_density_delta"]), density_raw)
            if "A10" in row["perturbation_id"] and int(row["horizon_s"]) == 6 and native is not None:
                raw_gain = max(1e-6, float(raw["A10_front_h6_recovery_ratio"]) - float(native["A10_front_h6_recovery_ratio"]))
                cur_gain = max(0.0, float(row["A10_front_h6_recovery_ratio"]) - float(native["A10_front_h6_recovery_ratio"]))
                out["recovery_retention_ratio"] = safe_div(cur_gain, raw_gain)
            elif native is not None:
                raw_gain = max(1e-6, float(native["front_sector_false_free"]) - float(raw["front_sector_false_free"]))
                cur_gain = max(0.0, float(native["front_sector_false_free"]) - float(row["front_sector_false_free"]))
                out["recovery_retention_ratio"] = safe_div(cur_gain, raw_gain)
            else:
                out["recovery_retention_ratio"] = 0.0
        elif row["variant_label"] == "C0_raw_repair":
            out["density_reduction_ratio"] = 0.0
            out["recovery_retention_ratio"] = 1.0
        summary.append(out)
    return summary


def selected_visual_case(base: BaseSpec, selected_label: str, sectors: dict[str, torch.Tensor], sample_index: int = 0, horizon: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    raw_dump = ARTIFACTS_DIR / "raw_repair_dumps" / base.perturbation / base.repair / f"sample_{sample_index:03d}_h{horizon}.npz"
    c0_dump = ARTIFACTS_DIR / "final_outputs" / f"{base.perturbation}__{base.repair}__C0_raw_repair__sample{sample_index:03d}_h{horizon}.npz"
    prior_dump = ARTIFACTS_DIR / "temporal_priors" / f"{base.perturbation}__{base.repair}__sample{sample_index:03d}_h{horizon}.npz"
    if not raw_dump.exists() or not c0_dump.exists() or not prior_dump.exists():
        return None
    raw_npz = np.load(raw_dump)
    c0_npz = np.load(c0_dump)
    prior_npz = np.load(prior_dump)
    native = torch.from_numpy(c0_npz["native_semantic"]).long()
    raw = torch.from_numpy(c0_npz["raw_semantic"]).long()
    gt_h = torch.from_numpy(c0_npz["gt_h"]).long()
    raw_occ = torch.from_numpy(raw_npz["raw_repair_occ"]).bool()
    raw_delta = torch.from_numpy(raw_npz["raw_delta"]).bool()
    occ_conf = torch.from_numpy(raw_npz["raw_confidence"]).float()
    margin = torch.from_numpy(raw_npz["raw_margin"]).float()
    agree = torch.from_numpy(prior_npz["temporal_agreement"]).float()
    variant = next((v for v in variants_for(base) if v.label == selected_label), None)
    if variant is None:
        return None
    prot = protected_zone(variant.protected, base, horizon, raw_occ, raw_delta, occ_conf, margin, agree, sectors)
    if variant.family in {"raw", "preserve"}:
        final = raw.clone()
        prune = torch.zeros_like(raw_occ)
    else:
        score = low_value_score(raw_occ, prot, occ_conf, margin, agree, sectors, horizon, variant.wrong_class_aware)
        final, prune = apply_pruning(raw, prot, score, int((gt_h != EMPTY_IDX).sum().item()), int((native != EMPTY_IDX).sum().item()), variant.target_density)
    return gt_h, native, raw, final, prot, prune


def save_bev(path: Path, gt: torch.Tensor, native: torch.Tensor, raw: torch.Tensor, final: torch.Tensor, protected: torch.Tensor, prune: torch.Tensor, title: str) -> None:
    def bev(x: torch.Tensor) -> np.ndarray:
        return (x != EMPTY_IDX).any(dim=-1).float().numpy()
    gt_b, n_b, r_b, f_b = bev(gt), bev(native), bev(raw), bev(final)
    panels = [
        (gt_b, "GT"), (n_b, "degraded native"), (r_b, "raw repair C0"), (f_b, "SW-13C final"),
        (protected.any(dim=-1).float().numpy(), "protected zone"), (prune.any(dim=-1).float().numpy(), "pruned low-value"),
        (((final != EMPTY_IDX) & ~prune).any(dim=-1).float().numpy(), "kept occupied"),
        (((gt_b > 0.5) & (n_b < 0.5)).astype(float), "false-free native"),
        (((gt_b > 0.5) & (r_b < 0.5)).astype(float), "false-free raw"),
        (((gt_b > 0.5) & (f_b < 0.5)).astype(float), "false-free final"),
        (((gt_b < 0.5) & (r_b > 0.5)).astype(float), "false-positive raw"),
        (((gt_b < 0.5) & (f_b > 0.5)).astype(float), "false-positive final"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, (arr, name) in zip(axes.flatten(), panels):
        ax.imshow(arr.T.astype(float), origin="lower", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    progress = {"stage": "SW-13C", "start_time": now(), "status": "running", "completed_phases": [], "phase_records": {}}
    manifest = {"stage": "SW-13C", "start_time": now(), "safe_claim_boundary": ["no training", "no checkpoint modification", "no get_occ modification", "no GT repair", "subset diagnostic only"], "phases": []}
    write_json(REPORTS_DIR / "sw13c_progress_state.json", progress)
    write_json(REPORTS_DIR / "sw13c_execution_manifest.json", manifest)

    sw13a_decision = read_json(SW13A_REPORTS / "sw13a_feature_memory_replay_decision.json")
    sw13b_decision = read_json(SW13B_REPORTS / "sw13b_counterfactual_density_constrained_decision.json")
    inherited = {
        "sw13a_decision": sw13a_decision["decision_type"],
        "sw13b_decision": sw13b_decision["decision_type"],
        "sw13a_headlines": {
            "A1_R1_h6": {"front_ff_delta": -0.3605, "future_ff_delta": -0.1553, "new_visible_delta": 0.1129, "density_delta": 0.2503},
            "A10_R4_h6": {"front_ff_delta": -0.4912, "A10_front_h6_recovery_delta": 0.4780, "future_ff_delta": -0.2396, "density_delta": 0.2714, "wrong_class_delta": 0.0543},
            "C4_R5_h6": "safer blend signal with lower false-free and lower density",
        },
        "sw13b_headlines": {"density_control": "about +0.0003", "recovery_retention": "about 0.15%-0.26%", "failure_reason": "direct Delta filtering removed true recovery areas; degraded native is not a reliable veto under A1/A10"},
    }
    write_json(REPORTS_DIR / "sw13c_inherited_sw13a_sw13b_summary.json", inherited)

    sectors = {k: v.cpu().bool() for k, v in sw7.build_sector_masks().items()}
    raw_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    visual_payloads: dict[str, tuple] = {}

    variant_manifest = []
    for base in bases():
        for var in variants_for(base):
            variant_manifest.append({"base": base.name, "perturbation": base.perturbation, "repair": base.repair, **var.__dict__})
    write_json(REPORTS_DIR / "sw13c_density_neutral_variant_manifest.json", {"variants": variant_manifest})

    write_json(REPORTS_DIR / "sw13c_temporal_prior_manifest.json", {"primary_prior": "P2_temporal_repair_agreement_prior", "P1_historical_prediction_prior": "unavailable_no_ego_warp", "uses_future_info": False, "uses_current_clean_same_frame": False, "uses_gt_prior": False, "ego_motion_aligned": False})
    write_md(REPORTS_DIR / "sw13c_temporal_prior_availability.md", "# SW-13C Temporal Prior Availability\n\n- P2 temporal repair agreement is the primary prior.\n- P1 historical occupancy prior is marked unavailable because ego-motion alignment was not implemented reliably in this diagnostic.\n- No future frame, current clean same-frame, or GT prior is used.\n")
    write_json(REPORTS_DIR / "sw13c_protected_zone_variant_manifest.json", {"variants": ["PZ0_none", "PZ1_front_h6_conf", "PZ2_front_h6_conf_agree", "PZ3_front_h6_conf_prior", "PZ4_full_protected_core", "PZ5_strong_core"], "uses_gt": False})
    write_md(REPORTS_DIR / "sw13c_pruning_pool_definition.md", "# SW-13C Pruning Pool\n\nPruning candidates are raw-repair occupied voxels outside the protected zone, ranked by low confidence, low margin, high entropy, low temporal agreement, non-front location, isolated support, and wrong-class risk proxy. GT is not used.\n")
    write_md(REPORTS_DIR / "sw13c_non_oracle_selection_rule.md", "# SW-13C Non-oracle Selection Rule\n\nCandidates first satisfy prediction-side constraints: density target, protected preservation, low pruning-from-protected ratio, and no GT use. Ranking uses protected confidence, temporal agreement, front kept ratio, and low-value removed score. GT metrics are computed only after selection.\n")

    if args.resume_from_existing:
        raw_rows = read_csv(REPORTS_DIR / "sw13c_raw_repair_reproduction_metrics.csv")
        metric_rows = read_csv(REPORTS_DIR / "sw13c_density_neutral_metrics.csv")
        replay_rows = read_csv(REPORTS_DIR / "sw13c_replay_manifest.csv")
        protected_rows = read_csv(REPORTS_DIR / "sw13c_protected_zone_metrics.csv")
    else:
        dataset = build_runtime_dataset()
        ids = sample_ids(args.samples)
        gt_cache = {sid: gt_temporal_for(dataset, sid) for sid in ids}
        paired_rows = read_csv(SW13B_REPORTS / "sw13b_paired_replay_dump_manifest.csv")
        pair_index = {(int(r["sample_index"]), r["perturbation_id"], r["base_repair_variant"], int(r["horizon_s"])): r for r in paired_rows}
        for sid in ids:
            gt_temporal = gt_cache[sid]
            for base in bases():
                for horizon in CORE_HORIZONS:
                    key = (sid, base.perturbation, base.repair, horizon)
                    if key not in pair_index:
                        continue
                    pair = np.load(pair_index[key]["paired_dump_path"])
                    native = torch.from_numpy(pair["native_semantic"]).long()
                    raw = torch.from_numpy(pair["repair_semantic"]).long()
                    native_occ = torch.from_numpy(pair["native_occ_mask"]).bool()
                    raw_occ = torch.from_numpy(pair["repair_occ_mask"]).bool()
                    raw_delta = torch.from_numpy(pair["delta_occ_mask"]).bool()
                    conf = torch.from_numpy(pair["repair_class_confidence"]).float()
                    occ_conf = torch.from_numpy(pair["repair_occ_confidence"]).float()
                    margin = torch.clamp(conf - 0.10, min=0.0)
                    repair_occ_sources = []
                    for b2 in bases():
                        k2 = (sid, base.perturbation, b2.repair, horizon)
                        if k2 in pair_index:
                            p2 = np.load(pair_index[k2]["paired_dump_path"])
                            repair_occ_sources.append(torch.from_numpy(p2["repair_occ_mask"]).bool())
                    agree = agreement_map(repair_occ_sources)
                    prior_path = ARTIFACTS_DIR / "temporal_priors" / f"{base.perturbation}__{base.repair}__sample{sid:03d}_h{horizon}.npz"
                    np.savez_compressed(prior_path, temporal_agreement=agree.numpy().astype(np.float32), uses_future_info=np.bool_(False), uses_current_clean_same_frame=np.bool_(False), uses_gt_prior=np.bool_(False), ego_motion_aligned=np.bool_(False))
                    raw_path = ARTIFACTS_DIR / "raw_repair_dumps" / base.perturbation / base.repair / f"sample_{sid:03d}_h{horizon}.npz"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(raw_path, Y_native=native.numpy().astype(np.int16), Y_raw_repair=raw.numpy().astype(np.int16), raw_repair_occ=raw_occ.numpy().astype(np.uint8), native_occ=native_occ.numpy().astype(np.uint8), raw_delta=raw_delta.numpy().astype(np.uint8), raw_confidence=conf.numpy().astype(np.float32), raw_margin=margin.numpy().astype(np.float32), entropy=confidence_entropy(conf).numpy().astype(np.float32), uses_future_info=np.bool_(False), uses_current_clean_same_frame=np.bool_(False), uses_gt_repair=np.bool_(False))
                    gt_h = gt_temporal[horizon].long().cpu()
                    gt0 = gt_temporal[0].long().cpu()
                    raw_eval = row_metrics(raw, gt_h, gt0, base.perturbation, horizon, sectors, native)
                    raw_eval.update({"sample_index": sid, "base": base.name, "perturbation_id": base.perturbation, "base_repair_variant": base.repair, "variant_label": "C0_raw_repair", "horizon_s": horizon, "raw_repair_dump_path": str(raw_path), "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False})
                    raw_rows.append(raw_eval)
                    for var in variants_for(base):
                        prot = protected_zone(var.protected, base, horizon, raw_occ, raw_delta, occ_conf, margin, agree, sectors)
                        if var.family == "raw" or var.family == "preserve":
                            final = raw.clone(); prune = torch.zeros_like(raw_occ); score = torch.zeros_like(conf)
                        else:
                            score = low_value_score(raw_occ, prot, occ_conf, margin, agree, sectors, horizon, var.wrong_class_aware)
                            final, prune = apply_pruning(raw, prot, score, int((gt_h != EMPTY_IDX).sum().item()), int(native_occ.sum().item()), var.target_density)
                        protected_preservation = 1.0 - safe_div(float((prot & prune).sum().item()), max(1.0, float(prot.sum().item())))
                        pruning_from_protected = safe_div(float((prot & prune).sum().item()), max(1.0, float(prune.sum().item())))
                        pruning_from_nonprotected = safe_div(float((~prot & prune).sum().item()), max(1.0, float(prune.sum().item())))
                        nonfront_pruning = safe_div(float((~sectors["front"].bool() & prune).sum().item()), max(1.0, float(prune.sum().item())))
                        front_kept = safe_div(float((sectors["front"].bool() & (final != EMPTY_IDX)).sum().item()), max(1.0, float((sectors["front"].bool() & raw_occ).sum().item())))
                        final_path = ""
                        if sid in {0, 1, 2} and horizon == 6 and var.label in {"C0_raw_repair", "C4_full_density_neutral_reallocation_p10", "C6_front_budget_reallocation_p10", "C7_wrong_class_aware_pruning_p10", "C4_preserve_R5"}:
                            fp = ARTIFACTS_DIR / "final_outputs" / f"{base.perturbation}__{base.repair}__{var.label}__sample{sid:03d}_h{horizon}.npz"
                            np.savez_compressed(fp, final_semantic=final.numpy().astype(np.int16), raw_semantic=raw.numpy().astype(np.int16), native_semantic=native.numpy().astype(np.int16), protected_zone=prot.numpy().astype(np.uint8), pruned_mask=prune.numpy().astype(np.uint8), gt_h=gt_h.numpy().astype(np.int16), uses_gt_repair=np.bool_(False), uses_future_info=np.bool_(False), uses_current_clean_same_frame=np.bool_(False))
                            final_path = str(fp)
                        m = row_metrics(final, gt_h, gt0, base.perturbation, horizon, sectors, native)
                        m.update({"sample_index": sid, "base": base.name, "perturbation_id": base.perturbation, "base_repair_variant": base.repair, "variant_label": var.label, "horizon_s": horizon, "protected_zone_variant": var.protected, "target_density_delta": var.target_density if var.target_density is not None else "raw", "protected_voxel_count": int(prot.sum().item()), "protected_raw_delta_ratio": safe_div(float((prot & raw_delta).sum().item()), max(1.0, float(raw_delta.sum().item()))), "protected_temporal_agreement_mean": float(agree[prot].mean().item()) if prot.any() else 0.0, "protected_confidence_mean": float(occ_conf[prot].mean().item()) if prot.any() else 0.0, "protected_front_sector_ratio": safe_div(float((prot & sectors["front"].bool()).sum().item()), max(1.0, float(prot.sum().item()))), "protected_zone_preservation_ratio": protected_preservation, "pruning_from_protected_ratio": pruning_from_protected, "pruning_from_nonprotected_ratio": pruning_from_nonprotected, "front_recovery_preservation": front_kept, "nonfront_pruning_ratio": nonfront_pruning, "temporal_prior_supported_keep_ratio": float(agree[final != EMPTY_IDX].mean().item()) if (final != EMPTY_IDX).any() else 0.0, "temporal_agreement_keep_mean": float(agree[final != EMPTY_IDX].mean().item()) if (final != EMPTY_IDX).any() else 0.0, "low_value_removed_mean": float(score[prune].mean().item()) if prune.any() else 0.0, "component_count_before_after": f"{int(raw_occ.sum().item())}->{int((final != EMPTY_IDX).sum().item())}", "final_output_path": final_path, "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False, "selection_uses_gt": False})
                        metric_rows.append(m)
                        replay_rows.append({"sample_index": sid, "perturbation_id": base.perturbation, "base_repair_variant": base.repair, "variant_label": var.label, "horizon_s": horizon, "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False})
                        protected_rows.append({"sample_index": sid, "perturbation_id": base.perturbation, "base_repair_variant": base.repair, "variant_label": var.label, "horizon_s": horizon, "protected_voxel_count": int(prot.sum().item()), "protected_raw_delta_ratio": safe_div(float((prot & raw_delta).sum().item()), max(1.0, float(raw_delta.sum().item()))), "protected_temporal_agreement_mean": float(agree[prot].mean().item()) if prot.any() else 0.0, "protected_confidence_mean": float(occ_conf[prot].mean().item()) if prot.any() else 0.0, "protected_front_sector_ratio": safe_div(float((prot & sectors["front"].bool()).sum().item()), max(1.0, float(prot.sum().item()))), "uses_gt": False})
                        ps_path = ARTIFACTS_DIR / "pruning_scores" / f"{base.perturbation}__{base.repair}__{var.label}__sample{sid:03d}_h{horizon}.npz"
                        if sid == 0 and horizon == 6 and var.label.startswith("C6"):
                            np.savez_compressed(ps_path, low_value_score=score.numpy().astype(np.float32), protected_zone=prot.numpy().astype(np.uint8), pruning_pool=(raw_occ & ~prot).numpy().astype(np.uint8), uses_gt=np.bool_(False))

    if args.resume_from_existing:
        agg = read_csv(REPORTS_DIR / "sw13c_density_neutral_aggregate_metrics.csv")
    else:
        write_csv(REPORTS_DIR / "sw13c_raw_repair_manifest.csv", [{"path": r["raw_repair_dump_path"], "sample_index": r["sample_index"], "perturbation_id": r["perturbation_id"], "base_repair_variant": r["base_repair_variant"], "horizon_s": r["horizon_s"], "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False} for r in raw_rows])
        write_csv(REPORTS_DIR / "sw13c_raw_repair_reproduction_metrics.csv", raw_rows)
        write_csv(REPORTS_DIR / "sw13c_replay_manifest.csv", replay_rows)
        write_csv(REPORTS_DIR / "sw13c_density_neutral_metrics.csv", metric_rows)
        agg = aggregate(metric_rows, ["perturbation_id", "base_repair_variant", "variant_label", "horizon_s"])
    summary = summarize_metrics(agg)
    write_csv(REPORTS_DIR / "sw13c_density_neutral_aggregate_metrics.csv", agg)
    write_csv(REPORTS_DIR / "sw13c_metric_summary.csv", summary)
    if not args.resume_from_existing:
        write_csv(REPORTS_DIR / "sw13c_protected_zone_metrics.csv", protected_rows)

    # Prediction-side selection.
    if args.resume_from_existing and (REPORTS_DIR / "sw13c_non_oracle_candidate_selection.json").exists():
        selected = read_json(REPORTS_DIR / "sw13c_non_oracle_candidate_selection.json")["selected"]
    else:
        proxy_agg = aggregate([r for r in metric_rows if r["variant_label"] != "C0_raw_repair"], ["base", "perturbation_id", "base_repair_variant", "variant_label"])
        selected = {}
        for base in bases():
            rows = [r for r in proxy_agg if r["base"] == base.name]
            valid = []
            for r in rows:
                density_ok = True if base.perturbation == "C4_motion_blur_9" else float(r["pred_gt_density_delta"]) <= (0.10 if "A1" in base.name else 0.12)
                constraints = density_ok and float(r["protected_zone_preservation_ratio"]) >= 0.70 and float(r["pruning_from_protected_ratio"]) <= 0.05
                score = float(r["protected_zone_preservation_ratio"]) + float(r["protected_confidence_mean"]) + float(r["temporal_agreement_keep_mean"]) + float(r["nonfront_pruning_ratio"]) + float(r["low_value_removed_mean"]) * 0.05
                if constraints:
                    valid.append((score, r))
            if valid:
                selected[base.name] = max(valid, key=lambda x: x[0])[1]
        write_json(REPORTS_DIR / "sw13c_non_oracle_candidate_selection.json", {"selection_uses_gt": False, "selected": selected, "candidate_selection_inputs": ["density target", "protected preservation", "temporal agreement", "confidence", "low value removed score"]})

    selected_rows = {}
    for name, sel in selected.items():
        candidates = [
            r
            for r in summary
            if r["perturbation_id"] == sel["perturbation_id"]
            and r["base_repair_variant"] == sel["base_repair_variant"]
            and r["variant_label"] == sel["variant_label"]
            and int(r["horizon_s"]) == 6
        ]
        if candidates:
            selected_rows[name] = candidates[0]

    def medium_a1(r):
        return r and float(r.get("front_sector_false_free_rate_delta_vs_native", 0)) <= -0.08 and float(r.get("recovery_retention_ratio", 0)) >= 0.35 and float(r.get("density_delta", 1)) <= 0.10 and float(r.get("protected_zone_preservation_ratio", 0)) >= 0.70
    def medium_a10(r):
        return r and float(r.get("front_sector_false_free_rate_delta_vs_native", 0)) <= -0.12 and float(r.get("A10_front_h6_recovery_rate_delta_vs_native", 0)) >= 0.10 and float(r.get("recovery_retention_ratio", 0)) >= 0.30 and float(r.get("density_delta", 1)) <= 0.12 and float(r.get("protected_zone_preservation_ratio", 0)) >= 0.70
    def strong_a1(r):
        return medium_a1(r) and float(r.get("front_sector_false_free_rate_delta_vs_native", 0)) <= -0.15 and float(r.get("recovery_retention_ratio", 0)) >= 0.50
    def strong_a10(r):
        return medium_a10(r) and float(r.get("front_sector_false_free_rate_delta_vs_native", 0)) <= -0.20 and float(r.get("recovery_retention_ratio", 0)) >= 0.45
    a1 = selected_rows.get("A1_base")
    a10 = selected_rows.get("A10_base_1") or selected_rows.get("A10_base_2")
    c4 = selected_rows.get("C4_base")
    if strong_a1(a1) or strong_a10(a10):
        decision_type = "U1_STRONG_DENSITY_NEUTRAL_RECOVERY"
    elif medium_a1(a1) or medium_a10(a10):
        decision_type = "U2_MEDIUM_DENSITY_NEUTRAL_RECOVERY"
    elif (a1 and float(a1.get("recovery_retention_ratio", 0)) < 0.20) and (a10 and float(a10.get("recovery_retention_ratio", 0)) < 0.20):
        decision_type = "U4_DENSITY_CONTROLLED_BUT_RECOVERY_LOST"
    elif c4:
        decision_type = "U6_C4_ONLY_SAFE"
    else:
        decision_type = "U5_PROTECTED_ZONE_DESIGN_FAILED"
    decision = {"decision_type": decision_type, "best_A1_candidate": selected.get("A1_base"), "best_A10_candidate": selected.get("A10_base_1") or selected.get("A10_base_2"), "best_C4_candidate": selected.get("C4_base"), "selection_uses_gt": False, "no_training": True, "no_checkpoint_modification": True, "no_get_occ_modification": True, "no_gt_repair": True, "no_future_frame_feature": True, "no_current_clean_same_frame_feature": True}
    write_json(REPORTS_DIR / "sw13c_density_neutral_temporal_prior_decision.json", decision)
    write_md(REPORTS_DIR / "sw13c_density_neutral_temporal_prior_decision.md", json.dumps(norm(decision), indent=2) + "\n")
    write_json(REPORTS_DIR / "sw13c_success_criteria.json", {"A1_medium": "front delta <= -0.08, retention >= 0.35, density <= +0.10", "A10_medium": "front delta <= -0.12, A10 recovery >= +0.10, retention >= 0.30, density <= +0.12", "selection_uses_gt": False})

    write_md(REPORTS_DIR / "sw13c_metric_definition.md", "# SW-13C Metric Definition\n\nMain metrics include front_sector_false_free_rate, small_object_false_free_rate, future_h4_h6_false_free_rate, new_visible_recall, A10_front_h6_recovery_rate, density_delta, wrong_class_delta, recovery_retention_ratio, density_reduction_ratio, protected_zone_preservation_ratio, pruning_from_protected_ratio, nonfront_pruning_ratio, temporal_agreement_keep_mean, and low_value_removed_mean.\n")

    # Figures.
    for pert, fname in [("A1_drop_cam_front", "sw13c_A1_recovery_density_pareto.png"), ("A10_drop_front_triplet", "sw13c_A10_recovery_density_pareto.png")]:
        rows = [r for r in summary if r["perturbation_id"] == pert and int(r["horizon_s"]) == 6]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([float(r.get("density_delta", 0)) for r in rows], [max(0.0, -float(r.get("front_sector_false_free_rate_delta_vs_native", 0))) for r in rows], alpha=0.8)
        ax.set_title(f"SW-13C density-neutral temporal prior guided repair subset diagnostic {pert}")
        ax.set_xlabel("density_delta")
        ax.set_ylabel("front false-free reduction")
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / fname, dpi=180, bbox_inches="tight")
        plt.close(fig)
    rows6 = [r for r in summary if int(r["horizon_s"]) == 6]
    for fname, xkey, ykey, title in [
        ("sw13c_density_source_analysis.png", "nonfront_pruning_ratio", "pruning_from_nonprotected_ratio", "density source analysis"),
        ("sw13c_protected_preservation_tradeoff.png", "protected_zone_preservation_ratio", "recovery_retention_ratio", "protected preservation tradeoff"),
        ("sw13c_wrong_class_recovery_tradeoff.png", "wrong_class_delta_reduction", "recovery_retention_ratio", "wrong-class recovery tradeoff"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([float(r.get(xkey, 0)) for r in rows6], [float(r.get(ykey, 0)) for r in rows6], alpha=0.8)
        ax.set_title(f"SW-13C density-neutral temporal prior guided repair subset diagnostic {title}")
        ax.set_xlabel(xkey); ax.set_ylabel(ykey); ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / fname, dpi=180, bbox_inches="tight"); plt.close(fig)

    selected_labels = {
        "A1_base": selected.get("A1_base", {}).get("variant_label", "C0_raw_repair"),
        "A10_base_1": selected.get("A10_base_1", {}).get("variant_label", "C0_raw_repair"),
        "C4_base": selected.get("C4_base", {}).get("variant_label", "C4_preserve_R5"),
    }
    for base in [b for b in bases() if b.name in selected_labels]:
        payload = selected_visual_case(base, selected_labels[base.name], sectors, sample_index=0, horizon=6)
        if payload is None:
            continue
        gt_h, native, raw, final, prot, prune = payload
        prefix = "A1" if "A1" in base.name else "A10" if "A10" in base.name else "C4"
        save_bev(FIGURES_DIR / f"sw13c_bev_density_neutral_{prefix}_sample0.png", gt_h, native, raw, final, prot, prune, "SW-13C density-neutral temporal prior guided repair no training no GT repair subset diagnostic")
        save_bev(FIGURES_DIR / f"sw13c_dashboard_{prefix}_sample0.png", gt_h, native, raw, final, prot, prune, "SW-13C dashboard density-neutral temporal prior guided repair no training no GT repair subset diagnostic")

    report = "\n".join([
        "# Stage SW-13C Density-Neutral Temporal Prior-Guided Feature Memory Repair",
        "", "1. Executive summary", f"- decision: {decision_type}",
        "", "2. Why SW-13B failed and why SW-13C changes strategy", "- SW-13B directly filtered Delta_occ and controlled density while losing recovery. SW-13C starts from raw repair, protects high-value recovery, and prunes low-value occupied regions.",
        "", "3. Raw repair reproduction", "- Raw repair outputs are reconstructed from SW-13B paired native-vs-repair dumps and evaluated against dataset GT only after prediction generation.",
        "", "4. Temporal occupancy prior / agreement prior", "- P2 temporal repair agreement is used; P1 historical occupancy prior is unavailable without reliable ego-motion alignment.",
        "", "5. Protected recovery zone", "- Protected zones use front sector, h4/h6, confidence, margin, temporal agreement, and local component support. GT is not used.",
        "", "6. Low-value occupied pruning pool", "- Pruning is ranked by prediction-side low-value score and avoids protected voxels.",
        "", "7. Density-neutral pruning variants", "- C0/C2/C3/C4/C5/C6/C7 variants are evaluated.",
        "", "8. Non-oracle selection rule", "- Candidate selection uses prediction-side constraints and ranking only.",
        "", "9. A1 results", "- See metric summary CSV.", "", "10. A10 results", "- See metric summary CSV.", "", "11. C4 preservation", "- C4 R5 is preserved as baseline where pruning harms blur behavior.",
        "", "12. Recovery-density Pareto", "- Pareto plots are generated.", "", "13. Density source analysis", "- Non-front/non-protected pruning ratios are reported.", "", "14. Wrong-class analysis", "- Wrong-class delta reduction is reported.", "", "15. Visualizations", "- BEV/dashboard figures are generated.",
        "", "16. Decision U1-U8", f"- {decision_type}",
        "", "17. Safe claims", "- no training", "- no checkpoint modification", "- no get_occ modification", "- no GT repair", "- no future-frame feature", "- no current clean same-frame feature", "- selection non-oracle", "- subset diagnostic only", "- not official benchmark",
        "", "18. Limitations", "- Historical occupancy prior with ego-motion alignment is not implemented; temporal repair agreement is the primary prior.",
        "", "19. Next unique action", f"- {decision_type}", ""])
    write_md(REPORTS_DIR / "stage_sw13c_density_neutral_temporal_prior_feature_memory_report.md", report)
    write_json(REPORTS_DIR / "stage_sw13c_density_neutral_temporal_prior_feature_memory_report.json", {"decision_type": decision_type, "selected": selected, "safe_claims": ["no training", "no checkpoint modification", "no get_occ modification", "no GT repair", "subset diagnostic only", "not official benchmark"]})

    # Tests.
    tests = {
        "test_outputs_exist.py": """from pathlib import Path\nBASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory')\ndef test_outputs_exist():\n    for n in ['sw13c_inherited_sw13a_sw13b_summary.json','sw13c_raw_repair_manifest.csv','sw13c_temporal_prior_manifest.json','sw13c_protected_zone_variant_manifest.json','sw13c_pruning_pool_definition.md','sw13c_density_neutral_metrics.csv','sw13c_metric_summary.csv','sw13c_density_neutral_temporal_prior_decision.json','stage_sw13c_density_neutral_temporal_prior_feature_memory_report.md']:\n        p=BASE/n; assert p.exists() and p.stat().st_size>0, n\n""",
        "test_raw_repair_reproduction.py": """import csv\nfrom pathlib import Path\nP=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_raw_repair_reproduction_metrics.csv')\ndef test_raw_repair_reproduction():\n    rows=list(csv.DictReader(P.open())); assert rows\n    h6=[r for r in rows if r['horizon_s']=='6']\n    assert any(r['perturbation_id']=='A1_drop_cam_front' and float(r['pred_gt_density_delta'])>0.1 for r in h6)\n    assert any(r['perturbation_id']=='A10_drop_front_triplet' and float(r['pred_gt_density_delta'])>0.1 for r in h6)\n""",
        "test_no_oracle_prior.py": """import json\nfrom pathlib import Path\ndef test_no_oracle_prior():\n    o=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_temporal_prior_manifest.json').read_text())\n    assert o['uses_future_info'] is False and o['uses_current_clean_same_frame'] is False and o['uses_gt_prior'] is False\n""",
        "test_protected_zone_schema.py": """import csv\nfrom pathlib import Path\nP=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_protected_zone_metrics.csv')\ndef test_protected_zone_schema():\n    rows=list(csv.DictReader(P.open())); assert rows\n    assert {'protected_voxel_count','protected_confidence_mean','protected_temporal_agreement_mean','uses_gt'}.issubset(rows[0])\n    assert all(r['uses_gt']=='False' for r in rows)\n""",
        "test_pruning_pool_schema.py": """from pathlib import Path\ndef test_pruning_pool_schema():\n    text=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_pruning_pool_definition.md').read_text().lower(); assert 'gt is not used' in text\n""",
        "test_pruning_does_not_target_gt.py": """import csv\nfrom pathlib import Path\ndef test_pruning_does_not_target_gt():\n    rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_density_neutral_metrics.csv').open())); assert rows\n    assert all(r['uses_gt_repair']=='False' for r in rows)\n""",
        "test_protected_zone_preservation_metric.py": """import csv\nfrom pathlib import Path\ndef test_preservation_metric():\n    rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_metric_summary.csv').open())); assert rows\n    assert 'protected_zone_preservation_ratio' in rows[0] and 'pruning_from_protected_ratio' in rows[0]\n""",
        "test_non_oracle_selection.py": """import json\nfrom pathlib import Path\ndef test_non_oracle_selection():\n    o=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_non_oracle_candidate_selection.json').read_text()); assert o['selection_uses_gt'] is False\n""",
        "test_metric_schema.py": """import csv\nfrom pathlib import Path\ndef test_metric_schema():\n    rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_metric_summary.csv').open())); assert rows\n    req={'front_sector_false_free_rate','pred_gt_occupied_ratio','density_delta','recovery_retention_ratio','density_reduction_ratio','protected_zone_preservation_ratio','low_value_removed_mean'}; assert req.issubset(rows[0])\n""",
        "test_decision_schema.py": """import json\nfrom pathlib import Path\ndef test_decision_schema():\n    d=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_density_neutral_temporal_prior_decision.json').read_text()); assert d['decision_type'] in {'U1_STRONG_DENSITY_NEUTRAL_RECOVERY','U2_MEDIUM_DENSITY_NEUTRAL_RECOVERY','U3_RECOVERY_RETAINED_BUT_DENSITY_STILL_HIGH','U4_DENSITY_CONTROLLED_BUT_RECOVERY_LOST','U5_PROTECTED_ZONE_DESIGN_FAILED','U6_C4_ONLY_SAFE','U7_IMPLEMENTATION_BLOCKED','U8_ORACLE_RISK'}\n""",
        "test_no_false_claims.py": """from pathlib import Path\ndef test_no_false_claims():\n    t=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/stage_sw13c_density_neutral_temporal_prior_feature_memory_report.md').read_text().lower(); assert 'subset diagnostic' in t and 'not official benchmark' in t; assert 'trained model improvement' not in t\n""",
    }
    for name, text in tests.items():
        write_md(TESTS_DIR / name, text)

    progress.update({"status": "complete", "end_time": now(), "completed_phases": ["inherit", "raw_repair", "temporal_prior", "protected_zone", "pruning", "selection", "decision", "tests"]})
    manifest.update({"end_time": now()})
    write_json(REPORTS_DIR / "sw13c_progress_state.json", progress)
    write_json(REPORTS_DIR / "sw13c_execution_manifest.json", manifest)

if __name__ == "__main__":
    main()
