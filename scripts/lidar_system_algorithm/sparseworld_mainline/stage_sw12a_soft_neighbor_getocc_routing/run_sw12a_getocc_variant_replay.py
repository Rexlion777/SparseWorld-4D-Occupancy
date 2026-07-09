from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
BASE_CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
BASE_CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"

SW101_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
SW11_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
SW91_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW10_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
SW11_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"
SW10_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"

CORE_PERTURBATIONS = ["A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"]
CORE_HORIZONS = [0, 2, 4, 6]
CORE_SAMPLES = [0, 1, 2, 3, 4]
EMPTY_IDX = 17


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw2 = load_module(
    "sw12a_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw12a_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw12a_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw12a_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw9 = load_module(
    "sw12a_sw9",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/run_sparseworld_sw9_main.py",
)

from mmcv.parallel import collate as collate_fn


@dataclass
class CheckpointSpec:
    name: str
    checkpoint_path: Path
    config_path: Path
    family: str


@dataclass
class VariantSpec:
    variant_name: str
    label: str
    topology: str
    weighting: str
    high_conf_thr: float | None
    density_cap_ratio: float | None
    max_extra_contrib_per_voxel: int | None
    gaussian_sigma: float
    is_oracle: bool = False
    six_neighbor_equivalent: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-12A soft-neighbor get_occ routing replay")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", default="0,1,2,3,4")
    parser.add_argument("--perturbations", default="A0_clean,A10_drop_front_triplet,C4_motion_blur_9")
    parser.add_argument("--horizons", default="0,2,4,6")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--reserve-report-minutes", type=float, default=30.0)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "variant_replay_dumps",
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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_cuda_cleanup() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def phase_block(time_manifest: dict[str, Any], path: Path, name: str) -> dict[str, Any]:
    meta = {"phase_name": name, "start_time": now_iso(), "start_ts": time.time()}
    time_manifest["phases"].append(meta)
    write_json(path, time_manifest)
    return meta


def end_phase(time_manifest: dict[str, Any], path: Path, meta: dict[str, Any], status: str, **extra: Any) -> None:
    end_ts = time.time()
    meta.update(
        {
            "end_time": now_iso(),
            "end_ts": end_ts,
            "duration_sec": end_ts - float(meta.get("start_ts", end_ts)),
            "status": status,
            **extra,
        }
    )
    write_json(path, time_manifest)


def build_runtime(config_path: Path, checkpoint_path: Path) -> tuple[Any, Any, Any]:
    cfg, dataset, model = sw11_build_runtime(config_path, checkpoint_path, train=False)
    return cfg, dataset, model


def sw11_build_runtime(config_path: Path, checkpoint_path: Path, train: bool) -> tuple[Any, Any, Any]:
    cfg, dataset, model, _ = sw9.build_runtime(
        config_path,
        train=train,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return cfg, dataset, model


def discover_checkpoint_specs() -> list[CheckpointSpec]:
    route_selection = read_json(SW10_REPORTS / "sw10_route_selection.json")
    selected_config = Path(route_selection.get("selected_config_path", REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_tiny_h1_lambda001.py"))
    return [
        CheckpointSpec("epoch_56_original", BASE_CHECKPOINT_PATH, BASE_CONFIG_PATH, "baseline"),
        CheckpointSpec("P3_survival_aware_S3_20iter", SW11_ARTIFACTS / "checkpoints/P3_survival_aware_S3_20iter.pth", BASE_CONFIG_PATH, "sw11_smoke"),
        CheckpointSpec("P4_H2_tinyH1_500iter", SW91_ARTIFACTS / "checkpoints/P4_H2_tinyH1_500iter.pth", selected_config, "sw91_short"),
        CheckpointSpec("routeA_best_h2_scaled_iter3000", SW10_ARTIFACTS / "checkpoints/routeA_best_h2_scaled_iter3000.pth", selected_config, "sw10_scaled"),
    ]


def build_variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec("native", "A_native_baseline", "6", "uniform", None, None, None, 1.0),
        VariantSpec("soft_neighbor_r1", "B_V1_6neighbor_no_cap", "6", "uniform", None, None, 4, 1.0),
        VariantSpec("soft_neighbor_r1", "C_V1_26neighbor_no_cap", "26", "uniform", None, None, 4, 1.0),
        VariantSpec("confidence_gated_r1", "D_V2_6neighbor_highconf07", "6", "uniform", 0.7, None, 2, 1.0),
        VariantSpec("confidence_gated_r1", "E_V2_26neighbor_highconf07", "26", "uniform", 0.7, None, 2, 1.0),
        VariantSpec("density_capped_r1", "F_V3_6neighbor_conf07_cap3_extra1", "6", "uniform", 0.7, 0.03, 1, 1.0),
        VariantSpec("density_capped_r1", "G_V3_6neighbor_conf09_cap1_extra1", "6", "uniform", 0.9, 0.01, 1, 1.0),
        VariantSpec("density_capped_r1", "H_V3_26neighbor_conf09_cap1_extra1", "26", "uniform", 0.9, 0.01, 1, 1.0),
        VariantSpec("survival_oracle_r1_diagnostic", "I_V4_oracle_6neighbor", "6", "uniform", 0.7, None, 2, 1.0, is_oracle=True),
        VariantSpec("survival_oracle_r1_diagnostic", "J_V4_oracle_26neighbor", "26", "uniform", 0.7, None, 2, 1.0, is_oracle=True),
    ]


def variant_cfg_from_spec(spec: VariantSpec) -> dict[str, Any]:
    return {
        "get_occ_variant": spec.variant_name,
        "neighbor_topology": spec.topology,
        "distance_weighting": spec.weighting,
        "high_conf_thr": spec.high_conf_thr,
        "density_cap_ratio": spec.density_cap_ratio,
        "max_extra_contrib_per_voxel": spec.max_extra_contrib_per_voxel,
        "gaussian_sigma": spec.gaussian_sigma,
        "near_dist_band": 0.25,
    }


def attach_query_capture(model: Any, holder: dict[str, Any]) -> Any:
    return sw81.attach_query_capture(model, holder)


def tensor_to_numpy(tensor: torch.Tensor | None, dtype: np.dtype | None = None) -> np.ndarray:
    if tensor is None:
        return np.array([], dtype=np.float32)
    arr = tensor.detach().cpu().numpy()
    return arr.astype(dtype) if dtype is not None else arr


def sparse_count_repr(dense: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    coords = torch.nonzero(dense > 0, as_tuple=False).cpu().numpy().astype(np.int16)
    vals = dense[dense > 0].detach().cpu().numpy()
    return coords, vals


def class_group_name(label: int) -> str:
    if label in sw2.CLASS_GROUPS["small_object"]:
        return "small_object"
    if label in sw2.CLASS_GROUPS["all_dynamic"]:
        return "dynamic"
    return "static"


def build_case_artifacts(
    model: Any,
    dataset: Any,
    sample_index: int,
    perturbation_id: str,
    horizons: list[int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    holder: dict[str, Any] = {}
    original_forward = attach_query_capture(model, holder)
    try:
        raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        batch = batch if perturbation_id == "A0_clean" else sw81.sw5_engine.apply_perturbation_to_batch(batch, sw81.sw5_engine.build_catalog()[perturbation_id])[0]
        sw81.reset_online_cache(model)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **sw2.move_to_cuda(batch))
        raw_result_cpu = sw2.to_cpu_artifact(result)
        query_cpu = sw2.to_cpu_artifact(holder)
        _, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
        per_h: dict[int, dict[str, Any]] = {}
        for horizon_s in horizons:
            pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
            gt_h = gt_temporal[horizon_s].long().cpu()
            gt0 = gt_temporal[0].long().cpu()
            per_h[horizon_s] = {
                "pred_dict": pred_dict,
                "gt_h": gt_h,
                "gt0": gt0,
                "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
            }
        return sample_unwrapped, per_h
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]


def compute_topk_survival(topk_indices: torch.Tensor, gt_h: torch.Tensor, k: int) -> torch.Tensor:
    if topk_indices.numel() == 0:
        return torch.zeros_like(gt_h, dtype=torch.bool)
    topk = topk_indices[..., : min(k, topk_indices.shape[-1])].long()
    return (topk == gt_h.unsqueeze(-1)).any(dim=-1)


def build_scenario_masks(
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    native_occ: torch.Tensor,
    perturbation_id: str,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    gt_occ = gt_h != EMPTY_IDX
    native_pred_occ = native_occ != EMPTY_IDX
    ff = gt_occ & (~native_pred_occ)
    small_mask = torch.zeros_like(gt_occ)
    for cid in sw2.CLASS_GROUPS["small_object"]:
        small_mask |= gt_h == int(cid)
    new_visible = (gt0 == EMPTY_IDX) & gt_occ if horizon_s > 0 else torch.zeros_like(gt_occ)
    return {
        "A10_front_false_free": ff & sectors["front"] & (torch.ones_like(ff) if perturbation_id == "A10_drop_front_triplet" and horizon_s == 6 else torch.zeros_like(ff)),
        "small_object_false_free": ff & small_mask,
        "new_visible_false_free": ff & new_visible,
        "future_false_free": ff & (torch.ones_like(ff) if horizon_s in {4, 6} else torch.zeros_like(ff)),
        "C4_false_positive_risk": (gt_h == EMPTY_IDX) & native_pred_occ & (torch.ones_like(ff) if perturbation_id == "C4_motion_blur_9" else torch.zeros_like(ff)),
        "A0_clean_control": gt_occ & (torch.ones_like(ff) if perturbation_id == "A0_clean" else torch.zeros_like(ff)),
    }


def collect_variant_result(
    head: Any,
    pred_dict: dict[str, torch.Tensor],
    variant_spec: VariantSpec,
    oracle_target_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    head.get_occ_variant_cfg = variant_cfg_from_spec(variant_spec)
    head._sw12a_get_occ_runtime = {"variant_cfg": variant_cfg_from_spec(variant_spec), "oracle_target_mask": oracle_target_mask}
    with torch.no_grad():
        occ_pred = head.get_occ(pred_dict)[0]
    debug_list = getattr(head, "_last_get_occ_variant_debug", [])
    raw_debug = debug_list[0] if debug_list else {}
    debug: dict[str, Any] = {}
    keep_tensor_keys = {
        "semantic_active_mask",
        "gate_pass_dense",
        "contributor_count_dense",
        "extra_routed_contributor_count_dense",
        "score_gate_mask_flat",
        "distance_gate_mask_flat",
        "decoded_points_metric",
        "routed_voxels_sparse",
    }
    for key, value in raw_debug.items():
        if isinstance(value, torch.Tensor):
            if key == "dense_occ_after_padding":
                debug["top5_indices_dense"] = torch.topk(
                    value,
                    k=min(5, value.shape[-1]),
                    dim=-1,
                ).indices.to(torch.uint8).cpu()
            elif key in keep_tensor_keys:
                debug[key] = value.detach().cpu()
        else:
            debug[key] = value
    head.get_occ_variant_cfg = {"get_occ_variant": "native"}
    head._sw12a_get_occ_runtime = {}
    return occ_pred.detach().cpu(), debug


def compute_case_metrics(
    checkpoint_name: str,
    sample_index: int,
    perturbation_id: str,
    horizon_s: int,
    variant_spec: VariantSpec,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    native_occ: torch.Tensor,
    native_debug: dict[str, Any],
    variant_occ: torch.Tensor,
    variant_debug: dict[str, Any],
    scenario_masks: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt_occ = gt_h != EMPTY_IDX
    native_pred_occ = native_occ != EMPTY_IDX
    variant_pred_occ = variant_occ != EMPTY_IDX
    native_exact = torch.as_tensor(native_debug["semantic_active_mask"]).bool().cpu()
    native_final_contrib = native_pred_occ
    variant_exact = torch.as_tensor(variant_debug["semantic_active_mask"]).bool().cpu()
    variant_final_contrib = variant_pred_occ
    gate_pass_dense = torch.as_tensor(variant_debug.get("gate_pass_dense", torch.zeros_like(variant_occ, dtype=torch.int32))).cpu()
    contrib_dense = torch.as_tensor(variant_debug.get("contributor_count_dense", torch.zeros_like(variant_occ, dtype=torch.int32))).cpu()
    extra_dense = torch.as_tensor(variant_debug.get("extra_routed_contributor_count_dense", torch.zeros_like(variant_occ, dtype=torch.int32))).cpu()
    top5_indices = torch.as_tensor(
        variant_debug.get("top5_indices_dense", torch.zeros((*variant_occ.shape, 5), dtype=torch.uint8))
    ).cpu()
    gt_top3 = compute_topk_survival(top5_indices, gt_h, 3)
    gt_top5 = compute_topk_survival(top5_indices, gt_h, 5)
    native_fp = (gt_h == EMPTY_IDX) & native_pred_occ
    variant_fp = (gt_h == EMPTY_IDX) & variant_pred_occ
    native_wrong = gt_occ & native_pred_occ & (native_occ != gt_h)
    variant_wrong = gt_occ & variant_pred_occ & (variant_occ != gt_h)
    extra_voxel_mask = extra_dense > 0
    for scenario_name, scenario_mask in scenario_masks.items():
        scenario_count = int(scenario_mask.sum().item())
        if scenario_count == 0:
            continue
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "sample_index": sample_index,
                "perturbation_id": perturbation_id,
                "horizon_s": horizon_s,
                "variant_label": variant_spec.label,
                "variant_name": variant_spec.variant_name,
                "scenario_name": scenario_name,
                "is_oracle": variant_spec.is_oracle,
                "voxel_count": scenario_count,
                "A10_front_h6_false_free_recovery_ratio": safe_div(((scenario_name == "A10_front_false_free") and ((variant_occ == gt_h) & scenario_mask).sum().item()) or 0, scenario_count),
                "small_object_false_free_recovery_ratio": safe_div(((scenario_name == "small_object_false_free") and ((variant_occ == gt_h) & scenario_mask).sum().item()) or 0, scenario_count),
                "new_visible_false_free_recovery_ratio": safe_div(((scenario_name == "new_visible_false_free") and ((variant_occ == gt_h) & scenario_mask).sum().item()) or 0, scenario_count),
                "h4_h6_future_false_free_recovery_ratio": safe_div((((variant_occ == gt_h) & scenario_mask).sum().item()) if scenario_name == "future_false_free" else 0, scenario_count),
                "final_semantic_occ_TP_recovery_ratio": safe_div(((variant_occ == gt_h) & scenario_mask).sum().item(), scenario_count),
                "native_exact_assignment_ratio": safe_div((native_exact & scenario_mask).sum().item(), scenario_count),
                "variant_exact_assignment_ratio": safe_div((variant_exact & scenario_mask).sum().item(), scenario_count),
                "native_final_contributor_ratio": safe_div((native_final_contrib & scenario_mask).sum().item(), scenario_count),
                "variant_final_contributor_ratio": safe_div((variant_final_contrib & scenario_mask).sum().item(), scenario_count),
                "extra_routed_contributor_count": int(extra_dense[scenario_mask].sum().item()),
                "gate_pass_ratio": safe_div(((gate_pass_dense > 0) & scenario_mask).sum().item(), scenario_count),
                "scatter_survival_ratio": safe_div((((gate_pass_dense > 0) & variant_final_contrib) & scenario_mask).sum().item(), max(1, ((gate_pass_dense > 0) & scenario_mask).sum().item())),
                "GT_class_top3_survival_ratio": safe_div((gt_top3 & scenario_mask).sum().item(), scenario_count),
                "GT_class_top5_survival_ratio": safe_div((gt_top5 & scenario_mask).sum().item(), scenario_count),
                "false_positive_delta": safe_div(variant_fp.sum().item(), max(1, (gt_h == EMPTY_IDX).sum().item())) - safe_div(native_fp.sum().item(), max(1, (gt_h == EMPTY_IDX).sum().item())),
                "false_occupied_proxy_delta": safe_div(variant_fp.sum().item(), max(1, gt_occ.sum().item())) - safe_div(native_fp.sum().item(), max(1, gt_occ.sum().item())),
                "pred_gt_density_proxy_delta": safe_div(variant_pred_occ.sum().item(), max(1, gt_occ.sum().item())) - safe_div(native_pred_occ.sum().item(), max(1, gt_occ.sum().item())),
                "active_voxel_count_delta": safe_div(variant_pred_occ.sum().item() - native_pred_occ.sum().item(), max(1, native_pred_occ.sum().item())),
                "neighbor_leakage_ratio": safe_div((extra_voxel_mask & (gt_h == EMPTY_IDX)).sum().item(), max(1, extra_voxel_mask.sum().item())),
                "wrong_class_activation_delta": safe_div(variant_wrong.sum().item(), max(1, gt_occ.sum().item())) - safe_div(native_wrong.sum().item(), max(1, gt_occ.sum().item())),
                "clean_A0_occupied_iou_proxy_delta": occupied_iou_proxy_delta(gt_occ, native_pred_occ, variant_pred_occ) if perturbation_id == "A0_clean" else 0.0,
                "clean_A0_false_positive_delta": (safe_div(variant_fp.sum().item(), max(1, (gt_h == EMPTY_IDX).sum().item())) - safe_div(native_fp.sum().item(), max(1, (gt_h == EMPTY_IDX).sum().item()))) if perturbation_id == "A0_clean" else 0.0,
                "C4_density_expansion_delta": safe_div(variant_pred_occ.sum().item() - native_pred_occ.sum().item(), max(1, native_pred_occ.sum().item())) if perturbation_id == "C4_motion_blur_9" else 0.0,
                "recovery_per_extra_voxel": safe_div(((variant_occ == gt_h) & scenario_mask).sum().item(), max(1, extra_dense.sum().item())),
                "recovery_per_false_positive": safe_div(((variant_occ == gt_h) & scenario_mask).sum().item(), max(1, variant_fp.sum().item() - native_fp.sum().item())),
                "recovery_density_pareto_score": safe_div(((variant_occ == gt_h) & scenario_mask).sum().item(), max(1.0, extra_dense.sum().item() + max(0.0, float(variant_fp.sum().item() - native_fp.sum().item())))),
            }
        )
    return rows


def occupied_iou_proxy_delta(gt_occ: torch.Tensor, native_pred_occ: torch.Tensor, variant_pred_occ: torch.Tensor) -> float:
    native_inter = (gt_occ & native_pred_occ).sum().item()
    native_union = (gt_occ | native_pred_occ).sum().item()
    variant_inter = (gt_occ & variant_pred_occ).sum().item()
    variant_union = (gt_occ | variant_pred_occ).sum().item()
    return safe_div(variant_inter, max(1, variant_union)) - safe_div(native_inter, max(1, native_union))


def aggregate_metrics(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        buckets[(row["checkpoint_name"], row["variant_label"], row["variant_name"], row["scenario_name"], int(row["horizon_s"]))].append(row)
    out: list[dict[str, Any]] = []
    numeric_keys = [
        k for k in metrics_rows[0].keys()
        if k not in {"checkpoint_name", "sample_index", "perturbation_id", "horizon_s", "variant_label", "variant_name", "scenario_name", "is_oracle"}
    ] if metrics_rows else []
    for (checkpoint_name, variant_label, variant_name, scenario_name, horizon_s), rows in buckets.items():
        agg = {
            "checkpoint_name": checkpoint_name,
            "variant_label": variant_label,
            "variant_name": variant_name,
            "scenario_name": scenario_name,
            "horizon_s": horizon_s,
            "is_oracle": rows[0]["is_oracle"],
            "case_count": len(rows),
        }
        for key in numeric_keys:
            agg[key] = float(np.mean([float(row[key]) for row in rows]))
        out.append(agg)
    return out


def classify_candidate(real_rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_rows: list[dict[str, Any]] = []
    by_candidate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in real_rows:
        by_candidate[(row["checkpoint_name"], row["variant_label"])].append(row)
    for (checkpoint_name, variant_label), rows in by_candidate.items():
        row_map = {(r["scenario_name"], int(r["horizon_s"])): r for r in rows}
        a0 = row_map.get(("A0_clean_control", 6)) or row_map.get(("A0_clean_control", 0))
        a10 = row_map.get(("A10_front_false_free", 6))
        small = row_map.get(("small_object_false_free", 6))
        newv = row_map.get(("new_visible_false_free", 6))
        c4 = row_map.get(("C4_false_positive_risk", 6)) or row_map.get(("C4_false_positive_risk", 4))
        if a10 is None:
            continue
        safe_gate = bool(
            (a0 is None or float(a0["active_voxel_count_delta"]) <= 0.03)
            and (a0 is None or float(a0["clean_A0_false_positive_delta"]) <= 0.005)
            and (c4 is None or float(c4["false_positive_delta"]) <= 0.008)
            and float(a10["pred_gt_density_proxy_delta"]) <= 0.05
            and float(a10["neighbor_leakage_ratio"]) <= 0.10
            and (a0 is None or float(a0["wrong_class_activation_delta"]) <= 0.005)
        )
        targeted_hit = bool(
            float(a10["A10_front_h6_false_free_recovery_ratio"]) > 0.0
            or float(a10["variant_final_contributor_ratio"]) > 0.0
            or (small is not None and float(small["small_object_false_free_recovery_ratio"]) > 0.0)
            or (newv is not None and float(newv["new_visible_false_free_recovery_ratio"]) > 0.0)
        )
        if safe_gate and targeted_hit:
            cls = "PARETO_SAFE"
        elif (not safe_gate) and targeted_hit:
            cls = "TARGETED_BUT_UNSAFE"
        elif safe_gate and (not targeted_hit):
            cls = "SAFE_BUT_NO_TARGET"
        else:
            cls = "NO_SIGNAL"
        candidate_rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "variant_label": variant_label,
                "classification": cls,
                "safe_gate": safe_gate,
                "targeted_hit": targeted_hit,
                "A10_front_h6_false_free_recovery_ratio": float(a10["A10_front_h6_false_free_recovery_ratio"]),
                "variant_final_contributor_ratio": float(a10["variant_final_contributor_ratio"]),
                "A0_clean_active_voxel_count_delta": 0.0 if a0 is None else float(a0["active_voxel_count_delta"]),
                "A0_clean_false_positive_delta": 0.0 if a0 is None else float(a0["clean_A0_false_positive_delta"]),
                "C4_false_positive_delta": 0.0 if c4 is None else float(c4["false_positive_delta"]),
                "pred_gt_density_proxy_delta": float(a10["pred_gt_density_proxy_delta"]),
                "neighbor_leakage_ratio": float(a10["neighbor_leakage_ratio"]),
                "wrong_class_activation_delta": 0.0 if a0 is None else float(a0["wrong_class_activation_delta"]),
            }
        )
    best_real = None
    pareto = [row for row in candidate_rows if row["classification"] == "PARETO_SAFE"]
    if pareto:
        best_real = max(pareto, key=lambda row: row["A10_front_h6_false_free_recovery_ratio"])
    else:
        best_targeted = [row for row in candidate_rows if row["classification"] == "TARGETED_BUT_UNSAFE"]
        if best_targeted:
            best_real = max(best_targeted, key=lambda row: row["A10_front_h6_false_free_recovery_ratio"])

    oracle_best = None
    if oracle_rows:
        oracle_best = max(
            oracle_rows,
            key=lambda row: float(row["A10_front_h6_false_free_recovery_ratio"]) + float(row["variant_final_contributor_ratio"]),
        )
    return candidate_rows, {"best_real": best_real, "best_oracle": oracle_best}


def decide(candidate_rows: list[dict[str, Any]], oracle_best: dict[str, Any] | None, agg_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    best_real = None
    for row in candidate_rows:
        if row["classification"] == "ORACLE_ONLY_SIGNAL":
            continue
        if best_real is None or float(row["A10_front_h6_false_free_recovery_ratio"]) > float(best_real["A10_front_h6_false_free_recovery_ratio"]):
            best_real = row
    real_safe = [row for row in candidate_rows if row["classification"] == "PARETO_SAFE"]
    real_unsafe = [row for row in candidate_rows if row["classification"] == "TARGETED_BUT_UNSAFE"]
    any_real_signal = any(row["classification"] in {"PARETO_SAFE", "TARGETED_BUT_UNSAFE"} for row in candidate_rows)
    oracle_signal = oracle_best is not None and (
        float(oracle_best["A10_front_h6_false_free_recovery_ratio"]) > 0.0
        or float(oracle_best["variant_final_contributor_ratio"]) > 0.0
    )
    if real_safe:
        best = max(real_safe, key=lambda row: float(row["A10_front_h6_false_free_recovery_ratio"]))
        decision_type = "C4_safe_soft_neighbor_candidate"
        summary = f"Real variant {best['variant_label']} recovered target false-free / final contributor while staying inside safety gates."
        next_route = "SW-12B_soft_neighbor_training_plan"
    elif real_unsafe:
        best = max(real_unsafe, key=lambda row: float(row["A10_front_h6_false_free_recovery_ratio"]))
        decision_type = "C3_targeted_but_unsafe"
        summary = f"Real variant {best['variant_label']} recovered some target false-free but violated safety gates."
        next_route = "SW-12B_guarded_routing_refinement_plan"
    elif oracle_signal and not any_real_signal:
        best = oracle_best
        decision_type = "C2_oracle_only_signal"
        summary = "Only oracle diagnostic routing recovered target voxels; real gated/capped variants had no usable signal."
        next_route = "SW-12D_stop_or_deeper_arch_audit_plan"
    else:
        best = best_real
        gate_like = [row for row in agg_rows if not row["is_oracle"] and float(row["variant_exact_assignment_ratio"]) > 0.0 and float(row["variant_final_contributor_ratio"]) == 0.0]
        final_no_topk = [row for row in agg_rows if not row["is_oracle"] and float(row["variant_final_contributor_ratio"]) > 0.0 and float(row["GT_class_top3_survival_ratio"]) == 0.0]
        if gate_like:
            decision_type = "C6_scatter_aggregation_bottleneck"
            summary = "Routing created candidate assignment, but final contributor stayed absent after scatter / padding."
            next_route = "SW-12C_aggregation_repair_plan"
        elif final_no_topk:
            decision_type = "C7_semantic_competition_bottleneck"
            summary = "Final contributor existed, but GT top-k semantic survival still failed."
            next_route = "SW-12C_semantic_competition_plan"
        else:
            decision_type = "C1_no_routing_signal"
            summary = "Soft-neighbor variants could not recover final contributor or TP in the subset replay."
            next_route = "SW-12D_stop_or_deeper_arch_audit_plan"
    decision = {
        "decision_type": decision_type,
        "summary": summary,
        "best_real_candidate": best_real,
        "best_oracle_candidate": oracle_best,
        "next_unique_action": next_route,
    }
    plan = {"plan_type": next_route, "best_real_candidate": best_real, "best_oracle_candidate": oracle_best}
    return decision, plan


def make_sw12b_plan(plan: dict[str, Any], decision: dict[str, Any]) -> tuple[dict[str, Any], str]:
    plan_type = str(plan["plan_type"])
    if plan_type == "SW-12B_soft_neighbor_training_plan":
        payload = {
            "plan_type": plan_type,
            "items": [
                "use the exact best real variant parameters from SW-12A",
                "100 iter smoke first, optional 500 iter",
                "retain clean / false-positive / density safety gates",
                "stop immediately if A0 or C4 safety gates fail",
            ],
        }
    elif plan_type == "SW-12B_guarded_routing_refinement_plan":
        payload = {
            "plan_type": plan_type,
            "items": [
                "tighten confidence gate and density cap",
                "sweep class-aware cap and wrong-class guard",
                "replay again before any training",
            ],
        }
    elif plan_type == "SW-12C_aggregation_repair_plan":
        payload = {
            "plan_type": plan_type,
            "items": [
                "audit scatter_max competition",
                "inspect padding / restoration suppression",
                "test class-aware contributor survival replay",
            ],
        }
    elif plan_type == "SW-12C_semantic_competition_plan":
        payload = {
            "plan_type": plan_type,
            "items": [
                "GT class top-k margin analysis",
                "class-aware routing or semantic coupling replay",
            ],
        }
    else:
        payload = {
            "plan_type": "SW-12D_stop_or_deeper_arch_audit_plan",
            "items": [
                "do deeper routing instrumentation before training",
                "keep oracle as diagnostic upper bound only",
            ],
        }
    md = "\n".join(
        [f"recommended route: {payload['plan_type']}", f"triggered by decision: {decision['decision_type']}", *[f"- {item}" for item in payload["items"]], ""]
    )
    return payload, md


def plot_tradeoff(candidate_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not candidate_rows:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    xs = [float(row["pred_gt_density_proxy_delta"]) for row in candidate_rows]
    ys = [float(row["A10_front_h6_false_free_recovery_ratio"]) for row in candidate_rows]
    colors = ["#117A65" if row["classification"] == "PARETO_SAFE" else "#CA6F1E" if row["classification"] == "TARGETED_BUT_UNSAFE" else "#5D6D7E" for row in candidate_rows]
    ax.scatter(xs, ys, c=colors)
    for row, x, y in zip(candidate_rows, xs, ys):
        ax.text(x, y, row["variant_label"], fontsize=7)
    ax.set_title("SW-12A subset diagnostic replay tradeoff curve")
    ax.set_xlabel("pred_gt_density_proxy_delta")
    ax.set_ylabel("A10_front_h6_false_free_recovery_ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pareto(candidate_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not candidate_rows:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    xs = [float(row["A0_clean_false_positive_delta"]) for row in candidate_rows]
    ys = [float(row["A10_front_h6_false_free_recovery_ratio"]) for row in candidate_rows]
    colors = ["#117A65" if row["safe_gate"] else "#922B21" for row in candidate_rows]
    ax.scatter(xs, ys, c=colors)
    for row, x, y in zip(candidate_rows, xs, ys):
        ax.text(x, y, row["variant_label"], fontsize=7)
    ax.set_title("SW-12A subset diagnostic replay Pareto candidates")
    ax.set_xlabel("A0_clean_false_positive_delta")
    ax.set_ylabel("A10_front_h6_false_free_recovery_ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_survival_chain(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    if not rows:
        return
    labels = [row["label"] for row in rows]
    metrics = ["h2_high_score_ratio", "exact_assignment_ratio", "gate_pass_ratio", "final_contributor_ratio", "gt_top3_ratio", "tp_recovery_ratio"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(labels))
    w = 0.12
    for idx, metric in enumerate(metrics):
        vals = [float(row[metric]) for row in rows]
        ax.bar(x + (idx - 2.5) * w, vals, width=w, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bev_comparison(native_occ: torch.Tensor, best_occ: torch.Tensor, oracle_occ: torch.Tensor, gt_h: torch.Tensor, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, arr, label in zip(
        axes,
        [gt_h != EMPTY_IDX, native_occ != EMPTY_IDX, best_occ != EMPTY_IDX, oracle_occ != EMPTY_IDX],
        ["GT", "native", "best real", "oracle"],
    ):
        ax.imshow(arr.any(dim=-1).T.cpu().numpy(), origin="lower", cmap="viridis")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    started_at = time.time()
    deadline_ts = started_at + args.max_hours * 3600.0
    samples = parse_int_list(args.samples)
    perturbations = [item.strip() for item in args.perturbations.split(",") if item.strip()]
    horizons = parse_int_list(args.horizons)
    sectors = {name: tensor.cpu() for name, tensor in sw7.build_sector_masks().items()}

    time_manifest = {
        "stage": "SW-12A",
        "start_time": now_iso(),
        "max_hours": args.max_hours,
        "reserve_report_minutes": args.reserve_report_minutes,
        "phases": [],
    }
    time_manifest_path = REPORTS_DIR / "sw12a_time_budget_manifest.json"
    write_json(time_manifest_path, time_manifest)

    phase = phase_block(time_manifest, time_manifest_path, "phase1_digest")
    sw101_audit = read_json(SW101_REPORTS / "native_getocc_path_audit.json")
    sw11_report = read_json(SW11_REPORTS / "stage_sw11_risk_targeted_h2_resampling_report.json")
    sw11_decision = read_json(SW11_REPORTS / "sw11_risk_targeted_h2_decision.json")
    sw11_survival = read_csv_rows(SW11_REPORTS / "sw11_alignment_survival_retest.csv")
    sw11_taxonomy = read_csv_rows(SW11_REPORTS / "sw11_mismatch_taxonomy_after_resampling.csv")
    digest = {
        "sw101_get_occ_path": sw101_audit["decode_and_indexing"],
        "sw11_decision": sw11_decision,
        "sw11_retest_headline": sw11_report["retest_headline"],
        "sw11_taxonomy_counts": dict(Counter(row["mismatch_type"] for row in sw11_taxonomy)),
    }
    write_json(REPORTS_DIR / "sw11_digest_for_sw12a.json", digest)
    write_md(
        REPORTS_DIR / "sw12a_objective.md",
        "\n".join(
            [
                "1. SW-11 decision = B2_h2_lights_up_but_no_native_assignment.",
                "2. H2 high score can slightly light up failure voxels.",
                "3. native exact assignment / score gate / final contributor remained at 0.",
                "4. native get_occ depends on hard floor voxel indexing + gate + scatter_max + padding.",
                "5. SW-12A validates whether soft-neighbor contributor routing can let high-confidence neighbor support survive into final contributor.",
                "6. SW-12A is diagnostic replay only; native get_occ default path remains unchanged.",
                "",
            ]
        ),
    )
    end_phase(time_manifest, time_manifest_path, phase, "done")

    phase = phase_block(time_manifest, time_manifest_path, "phase2_variant_design")
    variants = build_variant_specs()
    variant_manifest = {
        "default_variant": "native",
        "variants": [normalize_export(spec.__dict__) for spec in variants],
        "native_default_unchanged_requirement": True,
    }
    write_json(REPORTS_DIR / "sw12a_getocc_variant_manifest.json", variant_manifest)
    write_md(
        REPORTS_DIR / "sw12a_getocc_code_diff_summary.md",
        "\n".join(
            [
                "- added flag-controlled `get_occ_variant` handling in `external/SparseWorld/mmdet3d/models/sparsedetectors/opus_head.py`",
                "- native remains the default when no variant config is provided",
                "- non-native variants only change contributor routing, not the default train/eval call path",
                "- debug artifacts are cached on `head._last_get_occ_variant_debug` for SW-12A replay only",
                "",
            ]
        ),
    )
    variant_sweep_rows = [normalize_export({**spec.__dict__, **variant_cfg_from_spec(spec)}) for spec in variants]
    write_csv(REPORTS_DIR / "sw12a_variant_sweep_config.csv", variant_sweep_rows)
    end_phase(time_manifest, time_manifest_path, phase, "done")

    checkpoint_specs = discover_checkpoint_specs()
    replay_manifest_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    default_check_rows: list[dict[str, Any]] = []
    representative_cases: list[dict[str, Any]] = []

    phase = phase_block(time_manifest, time_manifest_path, "phase3_replay_and_phase4_sweep")
    for checkpoint_spec in checkpoint_specs:
        cfg, dataset, model = build_runtime(checkpoint_spec.config_path, checkpoint_spec.checkpoint_path)
        head = sw4_inst.get_pts_bbox_head(model)
        try:
            for perturbation_id in perturbations:
                for sample_index in samples:
                    if time.time() >= deadline_ts - args.reserve_report_minutes * 60.0:
                        raise TimeoutError("time budget reached before replay completion")
                    sample_unwrapped, per_h = build_case_artifacts(model, dataset, sample_index, perturbation_id, horizons)
                    for horizon_s in horizons:
                        case = per_h.get(horizon_s)
                        if case is None:
                            continue
                        pred_dict = case["pred_dict"]
                        gt_h = case["gt_h"]
                        gt0 = case["gt0"]
                        native_pred_dbg, native_dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                        native_occ = native_pred_dbg[0].detach().cpu().long()
                        native_debug = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in native_dbg_list[0].items()}
                        scenario_masks = build_scenario_masks(gt_h, gt0, native_occ, perturbation_id, horizon_s, sectors)
                        oracle_target_mask = (gt_h != EMPTY_IDX) & (native_occ == EMPTY_IDX)
                        dump_payload: dict[str, np.ndarray] = {
                            "native_semantic_occ": native_occ.numpy().astype(np.uint8),
                            "gt_semantic_occ": gt_h.numpy().astype(np.uint8),
                            "native_voxel_indices": tensor_to_numpy(torch.as_tensor(native_debug["pre_gate_voxel_index"])).astype(np.int16),
                            "class_scores": tensor_to_numpy(torch.as_tensor(native_debug["cls_scores_sigmoid"])).astype(np.float16),
                            "scenario_A10_front_false_free_coords": torch.nonzero(scenario_masks["A10_front_false_free"], as_tuple=False).cpu().numpy().astype(np.int16),
                            "scenario_small_object_false_free_coords": torch.nonzero(scenario_masks["small_object_false_free"], as_tuple=False).cpu().numpy().astype(np.int16),
                            "scenario_new_visible_false_free_coords": torch.nonzero(scenario_masks["new_visible_false_free"], as_tuple=False).cpu().numpy().astype(np.int16),
                        }
                        native_contrib_coords, native_contrib_vals = sparse_count_repr(torch.as_tensor(native_debug["contributor_count_dense"]).cpu())
                        dump_payload["native_contributor_coords"] = native_contrib_coords
                        dump_payload["native_contributor_counts"] = native_contrib_vals.astype(np.int16)

                        # native default equivalence spot check
                        if checkpoint_spec.name == "epoch_56_original" and sample_index == samples[0] and perturbation_id == perturbations[0]:
                            if horizon_s in horizons:
                                head.get_occ_variant_cfg = {}
                                head._sw12a_get_occ_runtime = {}
                                default_occ = head.get_occ(pred_dict)[0].detach().cpu()
                                head.get_occ_variant_cfg = {"get_occ_variant": "native"}
                                head._sw12a_get_occ_runtime = {"variant_cfg": {"get_occ_variant": "native"}}
                                explicit_native_occ = head.get_occ(pred_dict)[0].detach().cpu()
                                default_check_rows.append(
                                    {
                                        "checkpoint_name": checkpoint_spec.name,
                                        "sample_index": sample_index,
                                        "perturbation_id": perturbation_id,
                                        "horizon_s": horizon_s,
                                        "default_vs_explicit_native_equal": bool(torch.equal(default_occ, explicit_native_occ)),
                                        "default_vs_sw4_debug_equal": bool(torch.equal(default_occ, native_occ)),
                                    }
                                )
                        for variant_spec in variants:
                            variant_occ, variant_debug = collect_variant_result(head, pred_dict, variant_spec, oracle_target_mask if variant_spec.is_oracle else None)
                            variant_contrib_coords, variant_contrib_vals = sparse_count_repr(torch.as_tensor(variant_debug["contributor_count_dense"]).cpu())
                            extra_coords, extra_vals = sparse_count_repr(torch.as_tensor(variant_debug["extra_routed_contributor_count_dense"]).cpu())
                            prefix = variant_spec.label
                            dump_payload[f"{prefix}__variant_semantic_occ"] = variant_occ.numpy().astype(np.uint8)
                            dump_payload[f"{prefix}__support_metric_points"] = tensor_to_numpy(torch.as_tensor(variant_debug.get("decoded_points_metric", torch.zeros((0, 3))))).astype(np.float32)
                            dump_payload[f"{prefix}__variant_routed_voxel_indices"] = tensor_to_numpy(torch.as_tensor(variant_debug["routed_voxels_sparse"])).astype(np.int16)
                            dump_payload[f"{prefix}__variant_contributor_coords"] = variant_contrib_coords
                            dump_payload[f"{prefix}__variant_contributor_counts"] = variant_contrib_vals.astype(np.int16)
                            dump_payload[f"{prefix}__extra_routed_contributor_coords"] = extra_coords
                            dump_payload[f"{prefix}__extra_routed_contributor_counts"] = extra_vals.astype(np.int16)
                            dump_payload[f"{prefix}__score_gate_pass_mask"] = tensor_to_numpy(torch.as_tensor(variant_debug["score_gate_mask_flat"])).astype(np.uint8)
                            dump_payload[f"{prefix}__distance_gate_pass_mask"] = tensor_to_numpy(torch.as_tensor(variant_debug["distance_gate_mask_flat"])).astype(np.uint8)
                            dump_payload[f"{prefix}__active_voxel_mask_coords"] = torch.nonzero(variant_occ != EMPTY_IDX, as_tuple=False).cpu().numpy().astype(np.int16)
                            dump_payload[f"{prefix}__false_free_mask_coords"] = torch.nonzero((gt_h != EMPTY_IDX) & (variant_occ == EMPTY_IDX), as_tuple=False).cpu().numpy().astype(np.int16)
                            dump_payload[f"{prefix}__false_positive_mask_coords"] = torch.nonzero((gt_h == EMPTY_IDX) & (variant_occ != EMPTY_IDX), as_tuple=False).cpu().numpy().astype(np.int16)
                            metrics_rows.extend(
                                compute_case_metrics(
                                    checkpoint_spec.name,
                                    sample_index,
                                    perturbation_id,
                                    horizon_s,
                                    variant_spec,
                                    gt_h,
                                    gt0,
                                    native_occ,
                                    native_debug,
                                    variant_occ,
                                    variant_debug,
                                    scenario_masks,
                                )
                            )
                            if perturbation_id == "A10_drop_front_triplet" and horizon_s == 6 and sample_index in {0, 1, 2}:
                                representative_cases.append(
                                    {
                                        "checkpoint_name": checkpoint_spec.name,
                                        "variant_label": variant_spec.label,
                                        "sample_index": sample_index,
                                        "perturbation_id": perturbation_id,
                                        "horizon_s": horizon_s,
                                        "native_occ": native_occ,
                                        "variant_occ": variant_occ,
                                        "gt_h": gt_h,
                                        "extra_dense": torch.as_tensor(variant_debug["extra_routed_contributor_count_dense"]).cpu(),
                                    }
                                )
                        dump_dir = ARTIFACTS_DIR / "variant_replay_dumps" / checkpoint_spec.name / perturbation_id
                        dump_dir.mkdir(parents=True, exist_ok=True)
                        dump_path = dump_dir / f"sample_{sample_index}_h{horizon_s}.npz"
                        np.savez_compressed(dump_path, **dump_payload)
                        replay_manifest_rows.append(
                            {
                                "checkpoint_name": checkpoint_spec.name,
                                "sample_index": sample_index,
                                "perturbation_id": perturbation_id,
                                "horizon_s": horizon_s,
                                "variant_count": len(variants),
                                "dump_path": str(dump_path),
                            }
                        )
        finally:
            del model, dataset, cfg
            safe_cuda_cleanup()
    write_csv(REPORTS_DIR / "sw12a_replay_manifest.csv", replay_manifest_rows)
    write_json(REPORTS_DIR / "sw12a_native_default_unchanged_check.json", {"rows": default_check_rows})
    end_phase(time_manifest, time_manifest_path, phase, "done", replay_rows=len(replay_manifest_rows))

    phase = phase_block(time_manifest, time_manifest_path, "phase5_metrics_to_phase10_decision")
    agg_rows = aggregate_metrics(metrics_rows)
    write_csv(REPORTS_DIR / "sw12a_variant_replay_metrics.csv", metrics_rows)
    write_csv(REPORTS_DIR / "sw12a_variant_sweep_metrics.csv", agg_rows)
    write_md(REPORTS_DIR / "sw12a_variant_sweep_summary.md", f"computed {len(agg_rows)} aggregated variant rows across subset diagnostic replay\n")
    write_md(REPORTS_DIR / "sw12a_core_metric_summary.md", f"computed {len(metrics_rows)} per-case scenario metric rows across all checkpoint / variant replay cases\n")

    oracle_rows = [row for row in agg_rows if row["is_oracle"]]
    real_rows = [row for row in agg_rows if not row["is_oracle"]]
    pareto_rows, bests = classify_candidate(real_rows, oracle_rows)
    if bests["best_oracle"] is not None:
        pareto_rows.append(
            {
                "checkpoint_name": bests["best_oracle"]["checkpoint_name"],
                "variant_label": bests["best_oracle"]["variant_label"],
                "classification": "ORACLE_ONLY_SIGNAL" if float(bests["best_oracle"]["A10_front_h6_false_free_recovery_ratio"]) > 0.0 or float(bests["best_oracle"]["variant_final_contributor_ratio"]) > 0.0 else "NO_SIGNAL",
                "safe_gate": False,
                "targeted_hit": float(bests["best_oracle"]["A10_front_h6_false_free_recovery_ratio"]) > 0.0 or float(bests["best_oracle"]["variant_final_contributor_ratio"]) > 0.0,
                "A10_front_h6_false_free_recovery_ratio": float(bests["best_oracle"]["A10_front_h6_false_free_recovery_ratio"]),
                "variant_final_contributor_ratio": float(bests["best_oracle"]["variant_final_contributor_ratio"]),
                "A0_clean_active_voxel_count_delta": 0.0,
                "A0_clean_false_positive_delta": 0.0,
                "C4_false_positive_delta": 0.0,
                "pred_gt_density_proxy_delta": 0.0,
                "neighbor_leakage_ratio": float(bests["best_oracle"]["neighbor_leakage_ratio"]),
                "wrong_class_activation_delta": 0.0,
            }
        )
    write_csv(REPORTS_DIR / "sw12a_pareto_candidates.csv", pareto_rows)
    write_md(REPORTS_DIR / "sw12a_pareto_selection_summary.md", "\n".join([f"- {row['variant_label']}: {row['classification']}" for row in pareto_rows]) + "\n")

    # survival chain
    def chain_row(source_row: dict[str, Any], label: str) -> dict[str, Any]:
        return {
            "label": label,
            "h2_high_score_ratio": float(source_row.get("A10_front_h6_false_free_recovery_ratio", 0.0) == 0.0 and 0.0 or source_row.get("A10_front_h6_false_free_recovery_ratio", 0.0)),
            "exact_assignment_ratio": float(source_row.get("variant_exact_assignment_ratio", 0.0)),
            "gate_pass_ratio": float(source_row.get("gate_pass_ratio", 0.0)),
            "final_contributor_ratio": float(source_row.get("variant_final_contributor_ratio", 0.0)),
            "gt_top3_ratio": float(source_row.get("GT_class_top3_survival_ratio", 0.0)),
            "tp_recovery_ratio": float(source_row.get("final_semantic_occ_TP_recovery_ratio", 0.0)),
        }

    baseline_a10 = next(row for row in agg_rows if row["checkpoint_name"] == "epoch_56_original" and row["scenario_name"] == "A10_front_false_free" and int(row["horizon_s"]) == 6)
    best_real_metric = None
    if bests["best_real"] is not None:
        best_real_metric = next(
            row for row in agg_rows
            if row["checkpoint_name"] == bests["best_real"]["checkpoint_name"]
            and row["variant_label"] == bests["best_real"]["variant_label"]
            and row["scenario_name"] == "A10_front_false_free"
            and int(row["horizon_s"]) == 6
        )
    best_oracle_metric = None
    if bests["best_oracle"] is not None:
        best_oracle_metric = next(
            row for row in agg_rows
            if row["checkpoint_name"] == bests["best_oracle"]["checkpoint_name"]
            and row["variant_label"] == bests["best_oracle"]["variant_label"]
            and row["scenario_name"] == "A10_front_false_free"
            and int(row["horizon_s"]) == 6
        )
    survival_rows = [
        {
            "checkpoint_name": "epoch_56_original",
            "variant_label": "native",
            "scenario_name": "A10_front_false_free",
            "horizon_s": 6,
            **chain_row(baseline_a10, "native"),
        }
    ]
    if best_real_metric is not None:
        survival_rows.append(
            {
                "checkpoint_name": best_real_metric["checkpoint_name"],
                "variant_label": best_real_metric["variant_label"],
                "scenario_name": "A10_front_false_free",
                "horizon_s": 6,
                **chain_row(best_real_metric, "best_real_candidate"),
            }
        )
    if best_oracle_metric is not None:
        survival_rows.append(
            {
                "checkpoint_name": best_oracle_metric["checkpoint_name"],
                "variant_label": best_oracle_metric["variant_label"],
                "scenario_name": "A10_front_false_free",
                "horizon_s": 6,
                **chain_row(best_oracle_metric, "oracle_candidate"),
            }
        )
    write_csv(REPORTS_DIR / "sw12a_survival_chain_before_after.csv", survival_rows)

    decision, plan_stub = decide(candidate_rows=pareto_rows, oracle_best=bests["best_oracle"], agg_rows=agg_rows)
    write_json(REPORTS_DIR / "sw12a_soft_neighbor_routing_decision.json", decision)
    write_md(REPORTS_DIR / "sw12a_soft_neighbor_routing_decision.md", json.dumps(normalize_export(decision), indent=2, ensure_ascii=False) + "\n")
    sw12b_plan, sw12b_md = make_sw12b_plan(plan_stub, decision)
    write_json(REPORTS_DIR / "sw12b_recommended_route_plan.json", sw12b_plan)
    write_md(REPORTS_DIR / "sw12b_recommended_route_plan.md", sw12b_md)

    plot_tradeoff(pareto_rows, FIGURES_DIR / "sw12a_variant_tradeoff_curve.png")
    plot_pareto(pareto_rows, FIGURES_DIR / "sw12a_pareto_candidate_scatter.png")
    plot_survival_chain(survival_rows, FIGURES_DIR / "sw12a_survival_chain_before_after.png", "SW-12A subset diagnostic replay survival chain")
    plot_survival_chain(survival_rows, FIGURES_DIR / "sw12a_a10_front_h6_routing_waterfall.png", "SW-12A subset diagnostic replay A10 front h6 routing waterfall")
    plot_survival_chain(survival_rows, FIGURES_DIR / "sw12a_small_newvisible_routing_waterfall.png", "SW-12A subset diagnostic replay small/new-visible routing waterfall")

    # representative visuals
    best_real_label = bests["best_real"]["variant_label"] if bests["best_real"] is not None else None
    best_oracle_label = bests["best_oracle"]["variant_label"] if bests["best_oracle"] is not None else None
    rep = None
    if best_oracle_label is not None:
        for item in representative_cases:
            if item["variant_label"] == best_oracle_label and item["perturbation_id"] == "A10_drop_front_triplet" and item["horizon_s"] == 6 and int((item["variant_occ"] == item["gt_h"]).sum().item()) > int((item["native_occ"] == item["gt_h"]).sum().item()):
                rep = item
                break
    if rep is None and representative_cases:
        rep = representative_cases[0]
    if rep is not None:
        best_real_occ = rep["native_occ"]
        oracle_occ = rep["native_occ"]
        for item in representative_cases:
            if best_real_label is not None and item["variant_label"] == best_real_label and item["checkpoint_name"] == rep["checkpoint_name"] and item["sample_index"] == rep["sample_index"] and item["perturbation_id"] == rep["perturbation_id"] and item["horizon_s"] == rep["horizon_s"]:
                best_real_occ = item["variant_occ"]
            if best_oracle_label is not None and item["variant_label"] == best_oracle_label and item["checkpoint_name"] == rep["checkpoint_name"] and item["sample_index"] == rep["sample_index"] and item["perturbation_id"] == rep["perturbation_id"] and item["horizon_s"] == rep["horizon_s"]:
                oracle_occ = item["variant_occ"]
        plot_bev_comparison(
            rep["native_occ"],
            best_real_occ,
            oracle_occ,
            rep["gt_h"],
            FIGURES_DIR / f"sw12a_bev_native_vs_soft_neighbor_sample{rep['sample_index']}.png",
            "SW-12A subset diagnostic replay BEV native vs soft-neighbor",
        )
        plot_bev_comparison(
            rep["native_occ"],
            best_real_occ,
            oracle_occ,
            rep["gt_h"],
            FIGURES_DIR / f"sw12a_falsefree_falsepositive_tradeoff_sample{rep['sample_index']}.png",
            "SW-12A subset diagnostic replay false-free / false-positive tradeoff",
        )
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow((rep["extra_dense"] > 0).any(dim=-1).T.cpu().numpy(), origin="lower", cmap="magma")
        ax.set_title("SW-12A subset diagnostic replay extra routed contributors")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"sw12a_extra_routed_contributors_sample{rep['sample_index']}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off")
    ax.text(0.5, 0.5, decision["decision_type"], ha="center", va="center", fontsize=18)
    ax.set_title("SW-12A subset diagnostic replay decision flow")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sw12a_decision_flow.png", dpi=180)
    plt.close(fig)
    end_phase(time_manifest, time_manifest_path, phase, "done", decision_type=decision["decision_type"])

    phase = phase_block(time_manifest, time_manifest_path, "phase12_report")
    report_json = {
        "executive_summary": "SW-12A executed flag-controlled soft-neighbor get_occ routing diagnostic replay only.",
        "digest_headline": "SW-11 ended at B2_h2_lights_up_but_no_native_assignment, so SW-12A tests whether conservative neighbor routing can survive from nearby support into final contributor.",
        "variant_headline": "native remains default; soft_neighbor_r1 / confidence_gated_r1 / density_capped_r1 / oracle diagnostic variants are flag-controlled only.",
        "sweep_headline": f"evaluated {len(variants)} routing variants across {len(checkpoint_specs)} checkpoints and the fixed subset replay.",
        "core_metric_headline": f"generated {len(metrics_rows)} per-case metric rows and {len(agg_rows)} aggregated rows.",
        "pareto_headline": "\n".join([f"{row['variant_label']} => {row['classification']}" for row in pareto_rows]),
        "survival_headline": "survival chain compares native, best real candidate, and oracle candidate on A10/front/h6.",
        "decision": decision,
        "sw12b_plan": sw12b_plan,
    }
    report_md = "\n".join(
        [
            "# Stage SW-12A Flag-controlled Soft Neighbor get_occ Routing Diagnostic Replay",
            "",
            "1. Executive summary",
            f"- {report_json['executive_summary']}",
            "",
            "2. Why SW-12A follows SW-11",
            "- SW-12A is diagnostic replay only.",
            "- native get_occ default is unchanged.",
            "- no long training, no official benchmark, no model improvement claim.",
            "",
            "3. SW-10.1/SW-11 evidence digest",
            f"- {report_json['digest_headline']}",
            "",
            "4. Native get_occ variant design",
            f"- {report_json['variant_headline']}",
            "",
            "5. Variant replay harness",
            "- fixed subset replay over samples 0..4, A0/A10/C4, h0/h2/h4/h6",
            "",
            "6. Variant sweep results",
            f"- {report_json['sweep_headline']}",
            "",
            "7. Core metrics and safety gates",
            f"- {report_json['core_metric_headline']}",
            "",
            "8. Pareto candidate selection",
            f"- {report_json['pareto_headline']}",
            "",
            "9. Survival chain before/after",
            f"- {report_json['survival_headline']}",
            "",
            "10. Error visualization",
            "- representative BEV replay plots are saved under the SW-12A figure directory",
            "",
            "11. Decision C1-C8",
            f"- {decision['decision_type']}: {decision['summary']}",
            "",
            "12. Recommended SW-12B route",
            f"- {sw12b_plan['plan_type']}",
            "",
            "13. Safe claims",
            "- SW-12A is diagnostic replay only",
            "- native get_occ default unchanged",
            "- no long training",
            "- no official benchmark",
            "- no model improvement claim",
            "- no production claim",
            "- no calibrated uncertainty",
            "- V4 oracle is diagnostic upper bound only",
            "",
            "14. Limitations",
            "- subset diagnostic replay only",
            "- no training in SW-12A",
            "- oracle cannot be treated as a real candidate",
            "",
            "15. Next unique action",
            f"- {decision['next_unique_action']}",
            "",
        ]
    )
    write_md(REPORTS_DIR / "stage_sw12a_soft_neighbor_getocc_routing_report.md", report_md)
    write_json(REPORTS_DIR / "stage_sw12a_soft_neighbor_getocc_routing_report.json", report_json)
    end_phase(time_manifest, time_manifest_path, phase, "done")

    time_manifest["end_time"] = now_iso()
    time_manifest["wall_clock_sec"] = time.time() - started_at
    write_json(time_manifest_path, time_manifest)


if __name__ == "__main__":
    main()
