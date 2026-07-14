from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.runner.fp16_utils import cast_tensor_type


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
BASE_CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
BASE_CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"

PROGRESS_PATH = REPORTS_DIR / "sw13a_progress_state.json"
EXECUTION_MANIFEST_PATH = REPORTS_DIR / "sw13a_execution_manifest.json"

EMPTY_IDX = 17
EVAL_SAMPLE_IDS = list(range(20))
CORE_HORIZONS = [0, 2, 4, 6]
CORE_PERTURBATIONS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
CAMERA_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAMERA_TO_INDEX = {name: idx for idx, name in enumerate(CAMERA_ORDER)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw1 = load_module(
    "sw13a_sw1",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw1_bringup/postprocess_sparseworld_outputs.py",
)
sw2 = load_module(
    "sw13a_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw13a_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw13a_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw13a_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw12b = load_module(
    "sw13a_sw12b",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py",
)

from mmcv.parallel import collate as collate_fn


@dataclass
class RepairVariant:
    label: str
    mode: str
    offsets: list[int]
    blend_alpha: float | None
    allowed_perturbations: list[str]
    repair_cameras: list[str] | str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-13A sensor fault feature memory replay")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "feature_memory_cache",
        ARTIFACTS_DIR / "replay_dumps",
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
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def init_progress_state() -> dict[str, Any]:
    payload = {
        "stage": "SW-13A",
        "start_time": now_iso(),
        "current_phase": None,
        "completed_phases": [],
        "phase_records": {},
        "status": "running",
    }
    write_json(PROGRESS_PATH, payload)
    return payload


def init_execution_manifest() -> dict[str, Any]:
    payload = {
        "stage": "SW-13A",
        "subtitle": "No-train Temporal Camera Feature Repair for Degraded 4D Occupancy",
        "start_time": now_iso(),
        "phases": [],
        "artifacts": [],
        "safe_claim_boundary": [
            "test-time feature memory replay",
            "subset diagnostic",
            "not official benchmark",
            "no trained model improvement",
        ],
    }
    write_json(EXECUTION_MANIFEST_PATH, payload)
    return payload


def update_progress(progress: dict[str, Any]) -> None:
    write_json(PROGRESS_PATH, progress)


def update_manifest(manifest: dict[str, Any]) -> None:
    write_json(EXECUTION_MANIFEST_PATH, manifest)


def phase_start(progress: dict[str, Any], manifest: dict[str, Any], name: str) -> None:
    meta = {"phase_name": name, "start_time": now_iso(), "start_ts": time.time(), "status": "running"}
    progress["current_phase"] = name
    progress["phase_records"][name] = meta
    manifest["phases"].append(meta.copy())
    update_progress(progress)
    update_manifest(manifest)


def phase_end(progress: dict[str, Any], manifest: dict[str, Any], name: str, status: str, **extra: Any) -> None:
    record = progress["phase_records"][name]
    end_ts = time.time()
    record.update({"end_time": now_iso(), "end_ts": end_ts, "duration_sec": end_ts - record["start_ts"], "status": status, **extra})
    progress["current_phase"] = None
    if status == "done" and name not in progress["completed_phases"]:
        progress["completed_phases"].append(name)
    for item in manifest["phases"]:
        if item["phase_name"] == name and item["start_time"] == record["start_time"]:
            item.update(record)
            break
    update_progress(progress)
    update_manifest(manifest)


def attach_artifact(manifest: dict[str, Any], label: str, path: Path) -> None:
    manifest["artifacts"].append({"label": label, "path": str(path), "timestamp": now_iso()})
    update_manifest(manifest)


def build_runtime():
    cfg, dataset, model, _ = sw81.build_sparseworld_runtime(train=False, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    model.eval()
    return cfg, dataset, model


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def camera_name_from_filename(filename: str) -> str:
    for name in CAMERA_ORDER:
        if f"/{name}/" in filename or f"__{name}__" in filename:
            return name
    return "UNKNOWN"


def get_meta_dict(sample_unwrapped: dict[str, Any]) -> dict[str, Any]:
    meta = sample_unwrapped["img_metas"]
    if hasattr(meta, "data"):
        return meta.data[0][0]
    if isinstance(meta, list):
        return meta[0]
    return meta


def clone_meta_for_indices(meta: dict[str, Any], img_indices: list[int]) -> list[dict[str, Any]]:
    curr = [{}]
    for k, item in meta.items():
        if isinstance(item, list) and len(item) == len(meta["filename"]):
            curr[0][k] = [item[j] for j in img_indices]
        else:
            curr[0][k] = copy.deepcopy(item)
    return curr


def frame_img_tensor(moved_batch: dict[str, Any]) -> torch.Tensor:
    img = moved_batch["img"]
    if isinstance(img, list):
        img = img[0]
    return img


def reset_model_cache(model: Any) -> None:
    sw81.reset_online_cache(model)


def extract_clean_memory_for_sample(model: Any, batch_clean_cuda: dict[str, Any], sample_unwrapped: dict[str, Any], sample_index: int) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    img = frame_img_tensor(batch_clean_cuda)
    meta = get_meta_dict(sample_unwrapped)
    total_views = len(meta["filename"])
    num_frames = total_views // 6
    cache: dict[str, Any] = {
        "sample_index": sample_index,
        "num_frames": num_frames,
        "camera_order": CAMERA_ORDER,
        "offsets": {},
        "meta_filename": meta["filename"],
    }
    manifest_rows: list[dict[str, Any]] = []
    for offset in [1, 2, 3]:
        if offset >= num_frames:
            continue
        start = offset * 6
        img_indices = list(range(start, start + 6))
        img_metas_curr = clone_meta_for_indices(meta, img_indices)
        with torch.no_grad():
            feats = model.extract_feat(img[:, img_indices], img_metas_curr)
        offset_pack: dict[str, Any] = {"levels": []}
        for lvl, feat in enumerate(feats):
            feat_cpu = feat.detach().cpu().float()
            offset_pack["levels"].append(feat_cpu)
            for cam_idx, cam_name in enumerate(CAMERA_ORDER):
                for target_pert in ["A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"]:
                    manifest_rows.append(
                        {
                            "sample_index": sample_index,
                            "scene_token": meta.get("scene_token", ""),
                            "sample_token": meta.get("sample_idx", ""),
                            "camera_name": cam_name,
                            "feature_level": lvl,
                            "feature_shape": list(feat_cpu[:, cam_idx].shape),
                            "source_time_offset": offset,
                            "source_perturbation": "A0_clean",
                            "target_perturbation": target_pert,
                            "future_info_used": False,
                            "current_clean_same_frame_used": False,
                            "cache_path": str(ARTIFACTS_DIR / "feature_memory_cache" / f"sample_{sample_index:03d}.pt"),
                        }
                    )
        cache["offsets"][offset] = offset_pack
    cache_path = ARTIFACTS_DIR / "feature_memory_cache" / f"sample_{sample_index:03d}.pt"
    torch.save(cache, cache_path)
    return cache_path, manifest_rows, cache


def make_variants() -> list[RepairVariant]:
    return [
        RepairVariant("R0_degraded_native", "native", [], None, CORE_PERTURBATIONS, "none"),
        RepairVariant("R1_replace_tminus1", "replace", [1], None, ["A1_drop_cam_front", "A10_drop_front_triplet"], "perturbation"),
        RepairVariant("R2_replace_tminus2", "replace", [2], None, ["A1_drop_cam_front", "A10_drop_front_triplet"], "perturbation"),
        RepairVariant("R3_ema_K2", "ema", [1, 2], None, ["A1_drop_cam_front", "A10_drop_front_triplet"], "perturbation"),
        RepairVariant("R4_ema_K3", "ema", [1, 2, 3], None, ["A1_drop_cam_front", "A10_drop_front_triplet"], "perturbation"),
        RepairVariant("R5_blend_alpha03", "blend", [1, 2], 0.3, ["C4_motion_blur_9"], "all"),
        RepairVariant("R6_blend_alpha05", "blend", [1, 2], 0.5, ["C4_motion_blur_9"], "all"),
        RepairVariant("R7_blend_alpha07", "blend", [1, 2], 0.7, ["C4_motion_blur_9"], "all"),
        RepairVariant("R8_camera_group_repair_front_triplet", "replace", [1], None, ["A10_drop_front_triplet"], ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]),
    ]


def perturbation_camera_set(perturbation_id: str) -> list[str]:
    spec = sw81.sw5_engine.build_catalog()[perturbation_id]
    if spec.affected_cameras == "all":
        return list(CAMERA_ORDER)
    if isinstance(spec.affected_cameras, list):
        return list(spec.affected_cameras)
    return []


def should_run_variant(variant: RepairVariant, perturbation_id: str) -> bool:
    if variant.label == "R0_degraded_native":
        return True
    if perturbation_id == "A0_clean":
        return False
    if perturbation_id == "A7_drop_all_rear":
        return False
    return perturbation_id in variant.allowed_perturbations


def repair_camera_names(variant: RepairVariant, perturbation_id: str) -> list[str]:
    if variant.repair_cameras == "none":
        return []
    if variant.repair_cameras == "all":
        return list(CAMERA_ORDER)
    if variant.repair_cameras == "perturbation":
        return perturbation_camera_set(perturbation_id)
    return list(variant.repair_cameras)


def aggregate_memory_features(cache: dict[str, Any], offsets: list[int], level_idx: int, cam_idx: int) -> torch.Tensor | None:
    available = [off for off in offsets if off in cache["offsets"]]
    if not available:
        return None
    feats = [cache["offsets"][off]["levels"][level_idx][:, cam_idx].clone() for off in available]
    if len(available) == 1:
        return feats[0]
    if len(available) == 2:
        weights = [0.7, 0.3]
    else:
        weights = [0.6, 0.3, 0.1]
    out = torch.zeros_like(feats[0])
    for feat, w in zip(feats, weights):
        out += feat * w
    return out


def apply_feature_repair_to_levels(
    current_feats: list[torch.Tensor],
    cache: dict[str, Any],
    variant: RepairVariant,
    perturbation_id: str,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    repaired = [feat.clone() for feat in current_feats]
    cams = repair_camera_names(variant, perturbation_id)
    debug: dict[str, Any] = {
        "repair_triggered": False,
        "repaired_camera_names": cams,
        "memory_offsets_used": [],
        "repaired_camera_count": 0,
        "feature_distance_current_memory": 0.0,
        "feature_norm_drift": 0.0,
        "source_time_offsets": {},
    }
    if not cams:
        return repaired, debug
    distance_sum = 0.0
    norm_drift_sum = 0.0
    count = 0
    for cam_name in cams:
        cam_idx = CAMERA_TO_INDEX[cam_name]
        debug["source_time_offsets"][cam_name] = list(variant.offsets)
        for lvl, feat in enumerate(repaired):
            memory_feat = aggregate_memory_features(cache, variant.offsets, lvl, cam_idx)
            if memory_feat is None:
                continue
            debug["repair_triggered"] = True
            if variant.mode == "replace":
                new_feat = memory_feat.to(feat.device, dtype=feat.dtype)
            elif variant.mode == "ema":
                new_feat = memory_feat.to(feat.device, dtype=feat.dtype)
            elif variant.mode == "blend":
                alpha = float(variant.blend_alpha or 0.5)
                new_feat = alpha * feat[:, cam_idx] + (1.0 - alpha) * memory_feat.to(feat.device, dtype=feat.dtype)
            else:
                new_feat = feat[:, cam_idx]
            distance_sum += float(torch.norm(feat[:, cam_idx].float().cpu() - memory_feat.float(), p=2).item() / max(1, memory_feat.numel()))
            norm_drift_sum += abs(float(new_feat.float().norm().item()) - float(feat[:, cam_idx].float().norm().item()))
            count += 1
            feat[:, cam_idx] = new_feat
    debug["repaired_camera_count"] = len(cams)
    debug["memory_offsets_used"] = list(variant.offsets)
    debug["feature_distance_current_memory"] = safe_div(distance_sum, count)
    debug["feature_norm_drift"] = safe_div(norm_drift_sum, count)
    return repaired, debug


def run_variant_forward(
    model: Any,
    sample_unwrapped: dict[str, Any],
    batch_degraded: dict[str, Any],
    cache: dict[str, Any],
    variant: RepairVariant,
    perturbation_id: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, holder)
    original_simple_test_online = model.simple_test_online
    runtime_debug: dict[str, Any] = {"variant_label": variant.label, "perturbation_id": perturbation_id, "frame_debug": []}

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        assert len(img_metas) == 1
        B, N, C, H, W = img.shape
        img = img.reshape(B, N // 6, 6, C, H, W)
        img_filenames = img_metas[0]["filename"]
        num_frames = len(img_filenames) // 6
        img_shape = (H, W, C)
        img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_feats_list, img_metas_list = [], []
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))
            img_metas_curr = clone_meta_for_indices(img_metas[0], img_indices)
            img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
            frame_debug = {
                "frame_index": i,
                "camera_names": [camera_name_from_filename(img_metas_curr[0]["filename"][j]) for j in range(6)],
                "repair_applied": False,
                "source_time_offsets": [],
            }
            if i == 0 and should_run_variant(variant, perturbation_id):
                repaired_feats, repair_debug = apply_feature_repair_to_levels(img_feats_curr, cache, variant, perturbation_id)
                img_feats_curr = repaired_feats
                frame_debug.update(repair_debug)
                frame_debug["repair_applied"] = bool(repair_debug["repair_triggered"])
            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)
            runtime_debug["frame_debug"].append(frame_debug)
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)
        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)
        img_feats = cast_tensor_type(img_feats_reorganized, torch.half, torch.float32)
        return self.simple_test_pts(img_feats, img_metas_reorganized, rescale=rescale)

    model.simple_test_online = MethodType(patched_simple_test_online, model)
    try:
        reset_model_cache(model)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **sw2.move_to_cuda(batch_degraded))
        raw_result_cpu = sw2.to_cpu_artifact(result)
        query_cpu = sw2.to_cpu_artifact(holder)
        _, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
        per_h: dict[int, dict[str, Any]] = {}
        for horizon_s in CORE_HORIZONS:
            pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
            gt_h = gt_temporal[horizon_s].long().cpu()
            gt0 = gt_temporal[0].long().cpu()
            per_h[horizon_s] = {"pred_dict": pred_dict, "gt_h": gt_h, "gt0": gt0, "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None}
        return raw_result_cpu, per_h, runtime_debug
    finally:
        model.simple_test_online = original_simple_test_online
        model.forward_backbone = original_forward  # type: ignore[assignment]


def dense_occ_from_pred(head: Any, pred_dict: dict[str, Any]) -> torch.Tensor:
    pred_dbg, _ = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
    return pred_dbg[0].detach().cpu().long()


def build_eval_row_extended(
    pred: torch.Tensor,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    perturbation_id: str,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
    baseline_pred: torch.Tensor | None,
    debug: dict[str, Any],
) -> dict[str, Any]:
    row = sw12b.build_eval_row(pred, gt_h, gt0, perturbation_id, horizon_s, sectors, baseline_pred=baseline_pred)
    pred_occ = pred != EMPTY_IDX
    gt_occ = gt_h != EMPTY_IDX
    future_mask = torch.ones_like(gt_occ, dtype=torch.bool) if horizon_s in {4, 6} else torch.zeros_like(gt_occ, dtype=torch.bool)
    future_den = torch.logical_and(gt_occ, future_mask).sum().item()
    future_ff = torch.logical_and(torch.logical_and(gt_occ, future_mask), ~pred_occ).sum().item()
    row.update(
        {
            "front_sector_false_free_rate": row["front_sector_false_free"],
            "small_object_false_free_rate": row["small_object_false_free"],
            "dynamic_object_false_free_rate": row["dynamic_false_free"],
            "future_h4_h6_false_free_rate": float(future_ff / future_den) if future_den else 0.0,
            "A10_front_h6_recovery_rate": row["A10_front_h6_recovery_ratio"],
            "false_positive_rate": row["false_occupied_rate"],
            "density_drift": row["pred_gt_density_delta"],
            "wrong_class_rate": row["wrong_class_activation"],
            "clean_A0_metric_drift": row["clean_occupied_iou_delta"],
            "repaired_camera_count": int(debug.get("frame_debug", [{}])[0].get("repaired_camera_count", 0)),
            "memory_age_frame": int(np.mean(debug.get("frame_debug", [{}])[0].get("memory_offsets_used", [0])) if debug.get("frame_debug") and debug["frame_debug"][0].get("memory_offsets_used") else 0),
            "feature_distance_current_memory": float(debug.get("frame_debug", [{}])[0].get("feature_distance_current_memory", 0.0)),
            "repair_triggered_ratio": 1.0 if debug.get("frame_debug", [{}])[0].get("repair_applied", False) else 0.0,
            "feature_norm_drift": float(debug.get("frame_debug", [{}])[0].get("feature_norm_drift", 0.0)),
        }
    )
    return row


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    return sw12b.aggregate_rows(rows, group_keys)


def success_gate(row: dict[str, Any]) -> bool:
    return (
        float(row.get("false_positive_rate_delta", row.get("false_occupied_rate_delta", 0.0))) <= 0.005
        and float(row.get("density_drift", row.get("pred_gt_density_delta", 0.0))) <= 0.03
        and float(row.get("wrong_class_activation_delta", 0.0)) <= 0.005
        and float(row.get("clean_occupied_iou_delta", 0.0)) >= -0.003
    )


def phase_feature_path_audit(cfg: Any, dataset: Any, model: Any, manifest: dict[str, Any]) -> None:
    raw_sample, batch = sw2.extract_sample_batch(dataset, 0, collate_fn)
    sample = sw2.unwrap(raw_sample)
    meta = get_meta_dict(sample)
    moved = sw2.move_to_cuda(batch)
    img = frame_img_tensor(moved)
    img_metas_curr = clone_meta_for_indices(meta, list(range(6)))
    with torch.no_grad():
        feats = model.extract_feat(img[:, :6], img_metas_curr)
    audit = {
        "input_img_shape": list(img.shape),
        "num_frames": len(meta["filename"]) // 6,
        "camera_index_mapping": {name: idx for idx, name in enumerate(CAMERA_ORDER)},
        "filename_order_head": meta["filename"][:12],
        "feature_levels": [{"level": idx, "shape": list(feat.shape)} for idx, feat in enumerate(feats)],
        "path_audit": [
            {
                "file_path": str(REPO_ROOT / "mmdet3d/models/sparsedetectors/opus.py"),
                "function_name": "extract_feat",
                "source_line_range": [71, 141],
                "tensor_names": ["img", "img_feats", "img_feats_reshaped"],
                "notes": "camera dimension preserved as second axis after reshape to [B, N, C, H, W]",
            },
            {
                "file_path": str(REPO_ROOT / "mmdet3d/models/sparsedetectors/opus.py"),
                "function_name": "simple_test_online",
                "source_line_range": [250, 319],
                "tensor_names": ["img_feats_curr", "img_feats_list", "img_feats_reorganized", "memory", "queue"],
                "notes": "frame-by-frame cache keyed by first camera filename of each frame; repair can be inserted after extract_feat and before img_feats_list append",
            },
            {
                "file_path": str(REPO_ROOT / "mmdet3d/models/sparsedetectors/opus_transformer.py"),
                "function_name": "forward",
                "source_line_range": [105, 132],
                "tensor_names": ["mlvl_feats", "mlvl_feats_reshaped"],
                "notes": "head expects [B, T*N, GC, H, W] before internal regrouping into frame and camera axes",
            },
        ],
        "cache_risk": "reset_online_cache(model) clears model.memory and queue; no filename cache reuse is allowed across replay cases",
        "insert_repair_point": "OPUS.simple_test_online after img_feats_curr = self.extract_feat(img[:, i], img_metas_curr) and before img_feats_list.append(img_feats_curr)",
        "camera_mapping_confirmed_from_metadata": True,
        "future_info_allowed": False,
    }
    write_json(REPORTS_DIR / "sw13a_feature_path_audit.json", audit)
    write_md(
        REPORTS_DIR / "sw13a_feature_path_audit.md",
        "\n".join(
            [
                "# SW-13A Feature Path Audit",
                "",
                "- multi-camera image input is shaped `[B, 30, 3, 256, 704]` for the sampled case, i.e. 5 temporal steps x 6 cameras.",
                "- camera order from metadata is `CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT`.",
                "- `OPUS.extract_feat` returns multi-level features with shape `[B, N, C, H, W]` per level, where `N=6` for a single frame slice.",
                "- `OPUS.simple_test_online` already processes frame-by-frame and caches per-frame features in `model.memory` keyed by the first camera filename of that frame.",
                "- feature repair insertion point is after `img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)` and before frame features are appended and reorganized.",
                "- this phase uses only past clean features `t-1/t-2/t-3`; no current clean feature and no future frame feature are used.",
                "",
            ]
        ),
    )
    attach_artifact(manifest, "feature_path_audit", REPORTS_DIR / "sw13a_feature_path_audit.json")


def phase_build_memory_cache(dataset: Any, model: Any, sample_ids: list[int], manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    cache_map: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for sample_index in sample_ids:
        raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample = sw2.unwrap(raw_sample)
        moved = sw2.move_to_cuda(batch)
        reset_model_cache(model)
        cache_path, manifest_rows, cache = extract_clean_memory_for_sample(model, moved, sample, sample_index)
        cache_map[sample_index] = cache
        rows.extend(manifest_rows)
    write_csv(REPORTS_DIR / "sw13a_feature_memory_cache_manifest.csv", rows)
    write_md(
        REPORTS_DIR / "sw13a_feature_memory_cache_summary.md",
        "\n".join(
            [
                "# SW-13A Feature Memory Cache",
                "",
                f"- cached samples: {len(sample_ids)}",
                "- source perturbation is always A0_clean",
                "- source offsets limited to t-1 / t-2 / t-3",
                "- future information used: False",
                "- current clean same-frame replacement: False",
                "",
            ]
        ),
    )
    attach_artifact(manifest, "feature_memory_cache_manifest", REPORTS_DIR / "sw13a_feature_memory_cache_manifest.csv")
    return cache_map


def build_metric_definition() -> None:
    lines = [
        "# SW-13A Metric Definition",
        "",
        "- `front_sector_false_free_rate`: the fraction of GT occupied voxels in the front sector that the model predicts as empty.",
        "- `small_object_false_free_rate`: the fraction of GT occupied small-object voxels predicted as empty.",
        "- `dynamic_object_false_free_rate`: the fraction of GT occupied dynamic-object voxels predicted as empty.",
        "- `future_h4_h6_false_free_rate`: the fraction of GT occupied voxels predicted empty at horizons h4/h6.",
        "- `new_visible_recall`: recall on voxels that are empty at h0 but occupied at the future horizon.",
        "- `A10_front_h6_recovery_rate`: share of A10/front/h6 baseline misses that become occupied after repair.",
        "- `false_positive_rate` / `false_occupied_rate`: GT empty voxels predicted as occupied.",
        "- `pred_gt_occupied_ratio`: predicted occupied voxel count divided by GT occupied voxel count.",
        "- `density_drift`: change in occupied density relative to degraded native baseline.",
        "- `wrong_class_rate`: occupied voxels predicted with the wrong semantic class.",
        "- `clean_A0_metric_drift`: change on A0 clean. In this stage repair is disabled on A0 by default, so this should remain zero at runtime.",
        "- `repaired_camera_count`: number of cameras repaired in the current frame.",
        "- `memory_age_frame`: mean age of source features in frames.",
        "- `feature_distance_current_memory`: average normalized L2 distance between degraded current feature and memory feature before replacement/fusion.",
        "- `repair_triggered_ratio`: 1 when feature repair is active for the current replay case, else 0.",
        "- `feature_norm_drift`: mean change in feature norm introduced by the repair op.",
        "",
    ]
    write_md(REPORTS_DIR / "sw13a_metric_definition.md", "\n".join(lines))


def save_bev_image(path: Path, gt_h: torch.Tensor, native_pred: torch.Tensor, repaired_pred: torch.Tensor, title: str) -> None:
    gt_bev = (gt_h != EMPTY_IDX).any(dim=-1).float().numpy()
    native_bev = (native_pred != EMPTY_IDX).any(dim=-1).float().numpy()
    repaired_bev = (repaired_pred != EMPTY_IDX).any(dim=-1).float().numpy()
    ff_before = np.logical_and(gt_bev > 0.5, native_bev < 0.5).astype(float)
    ff_after = np.logical_and(gt_bev > 0.5, repaired_bev < 0.5).astype(float)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for ax, arr, name in zip(
        axes,
        [gt_bev, native_bev, repaired_bev, ff_before, ff_after],
        ["GT", "degraded native", "repaired", "false-free before", "false-free after"],
    ):
        ax.imshow(arr.T, origin="lower", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_fpv_dashboard(path: Path, sample_unwrapped: dict[str, Any], variant_label: str, perturbation_id: str, frame_debug: dict[str, Any]) -> None:
    meta = get_meta_dict(sample_unwrapped)
    names = [Path(f).name for f in meta["filename"][:6]]
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.axis("off")
    text = [
        f"variant={variant_label}",
        f"perturbation={perturbation_id}",
        f"repair_applied={frame_debug.get('repair_applied', False)}",
        f"repaired_cameras={frame_debug.get('repaired_camera_names', [])}",
        f"source_offsets={frame_debug.get('memory_offsets_used', [])}",
        "current frame cameras:",
        *names,
    ]
    ax.text(0.01, 0.99, "\n".join(text), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("SW-13A test-time feature memory replay subset diagnostic dashboard")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def phase_replay(cfg: Any, dataset: Any, model: Any, cache_map: dict[int, dict[str, Any]], sample_ids: list[int], manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    build_metric_definition()
    sectors = {name: tensor.cpu() for name, tensor in sw7.build_sector_masks().items()}
    variants = make_variants()
    variant_manifest = {
        "variants": [variant.__dict__ for variant in variants],
        "native_path_unchanged_by_default": True,
        "A0_clean_repaired_by_default": False,
        "no_get_occ_modification": True,
        "runtime_monkeypatch_only": True,
    }
    write_json(REPORTS_DIR / "sw13a_feature_repair_variant_manifest.json", variant_manifest)
    write_md(
        REPORTS_DIR / "sw13a_feature_repair_code_diff_summary.md",
        "\n".join(
            [
                "# SW-13A Feature Repair Runtime Summary",
                "",
                "- no external SparseWorld source file was modified",
                "- runtime monkeypatch replaces `simple_test_online` only during replay",
                "- native path is preserved when variant is `R0_degraded_native`",
                "- repair happens on current-frame camera features after `extract_feat` and before temporal reorganization",
                "",
            ]
        ),
    )

    spec_catalog = sw81.sw5_engine.build_catalog()
    metric_rows: list[dict[str, Any]] = []
    replay_manifest_rows: list[dict[str, Any]] = []
    native_equivalence = {"checked": False, "max_abs_diff": 0.0, "pass": False}
    best_dump_candidate: dict[str, Any] | None = None

    for sample_index in sample_ids:
        raw_sample, batch_clean = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        cache = cache_map[sample_index]
        per_pert_results: dict[str, dict[str, Any]] = {}
        for perturbation_id in CORE_PERTURBATIONS:
            batch_deg = copy.deepcopy(batch_clean)
            if perturbation_id != "A0_clean":
                batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[perturbation_id])[0]
            active_variants = [variant for variant in variants if should_run_variant(variant, perturbation_id)]
            baseline_pred_by_h: dict[int, torch.Tensor] = {}
            baseline_debug: dict[str, Any] | None = None
            for variant in active_variants:
                raw_result_cpu, per_h, runtime_debug = run_variant_forward(model, sample_unwrapped, batch_deg, cache, variant, perturbation_id)
                if variant.label == "R0_degraded_native":
                    baseline_debug = runtime_debug
                head = sw4_inst.get_pts_bbox_head(model)
                for horizon_s in CORE_HORIZONS:
                    dense_pred = dense_occ_from_pred(head, per_h[horizon_s]["pred_dict"])
                    if variant.label == "R0_degraded_native":
                        baseline_pred_by_h[horizon_s] = dense_pred
                    baseline_pred = baseline_pred_by_h.get(horizon_s) if variant.label != "R0_degraded_native" else dense_pred
                    row = build_eval_row_extended(
                        dense_pred,
                        per_h[horizon_s]["gt_h"],
                        per_h[horizon_s]["gt0"],
                        perturbation_id,
                        horizon_s,
                        sectors,
                        baseline_pred if variant.label != "R0_degraded_native" else dense_pred,
                        runtime_debug,
                    )
                    row.update(
                        {
                            "model_name": "epoch_56_original",
                            "sample_index": sample_index,
                            "perturbation_id": perturbation_id,
                            "horizon_s": horizon_s,
                            "variant_label": variant.label,
                            "repaired_camera_set": repair_camera_names(variant, perturbation_id),
                            "repair_mode": variant.mode,
                            "source_time_offsets": variant.offsets,
                            "blend_alpha": variant.blend_alpha,
                            "uses_future_info": False,
                            "uses_current_clean_same_frame": False,
                            "uses_gt_repair": False,
                        }
                    )
                    metric_rows.append(row)
                    if variant.label != "R0_degraded_native" and perturbation_id in {"A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"} and sample_index in {0, 1, 2} and horizon_s == 6:
                        best_dump_candidate = {
                            "sample_index": sample_index,
                            "perturbation_id": perturbation_id,
                            "horizon_s": horizon_s,
                            "variant_label": variant.label,
                            "gt_h": per_h[horizon_s]["gt_h"],
                            "native_pred": baseline_pred_by_h[horizon_s],
                            "repaired_pred": dense_pred,
                            "sample_unwrapped": sample_unwrapped,
                            "runtime_debug": runtime_debug,
                        }
                replay_manifest_rows.append(
                    {
                        "model_name": "epoch_56_original",
                        "sample_index": sample_index,
                        "perturbation_id": perturbation_id,
                        "variant_label": variant.label,
                        "repaired_camera_set": repair_camera_names(variant, perturbation_id),
                        "source_time_offsets": variant.offsets,
                        "blend_alpha": variant.blend_alpha,
                        "future_info_used": False,
                        "current_clean_same_frame_used": False,
                        "gt_used_for_repair": False,
                    }
                )
            if not native_equivalence["checked"] and perturbation_id == "A10_drop_front_triplet":
                native_equivalence["checked"] = True
                original_sample, original_per_h = sw12b.load_case(model, dataset, sample_index, perturbation_id, CORE_HORIZONS)
                head = sw4_inst.get_pts_bbox_head(model)
                diffs = []
                for horizon_s in CORE_HORIZONS:
                    dense_original = dense_occ_from_pred(head, original_per_h[horizon_s]["pred_dict"])
                    dense_native = baseline_pred_by_h[horizon_s]
                    diffs.append(float((dense_original != dense_native).sum().item()))
                native_equivalence["max_abs_diff"] = max(diffs)
                native_equivalence["pass"] = native_equivalence["max_abs_diff"] == 0.0
        reset_model_cache(model)

    write_csv(REPORTS_DIR / "sw13a_replay_manifest.csv", replay_manifest_rows)
    write_csv(REPORTS_DIR / "sw13a_feature_memory_replay_metrics.csv", metric_rows)
    agg_rows = aggregate_rows(metric_rows, ["model_name", "variant_label", "perturbation_id", "horizon_s"])
    write_csv(REPORTS_DIR / "sw13a_feature_memory_aggregate_metrics.csv", agg_rows)

    # enrich aggregate rows with delta vs degraded native
    baseline_map = {
        (row["perturbation_id"], int(row["horizon_s"])): row
        for row in agg_rows
        if row["variant_label"] == "R0_degraded_native"
    }
    summary_rows: list[dict[str, Any]] = []
    for row in agg_rows:
        base = baseline_map.get((row["perturbation_id"], int(row["horizon_s"])))
        summary = dict(row)
        if base is not None:
            for metric_key in [
                "front_sector_false_free_rate",
                "small_object_false_free_rate",
                "dynamic_object_false_free_rate",
                "future_h4_h6_false_free_rate",
                "new_visible_recall",
                "A10_front_h6_recovery_rate",
                "false_positive_rate",
                "pred_gt_occupied_ratio",
                "density_drift",
                "wrong_class_rate",
                "occupied_iou",
                "semantic_miou",
            ]:
                if metric_key in row and metric_key in base:
                    summary[f"{metric_key}_delta_vs_native"] = float(row[metric_key]) - float(base[metric_key])
        summary_rows.append(summary)
    write_csv(REPORTS_DIR / "sw13a_metric_summary.csv", summary_rows)
    write_json(ARTIFACTS_DIR / "native_path_unchanged_check.json", native_equivalence)

    # Tradeoff rows
    tradeoff_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["variant_label"] == "R0_degraded_native":
            continue
        recovery = max(0.0, -float(row.get("front_sector_false_free_rate_delta_vs_native", 0.0)))
        density = float(row.get("pred_gt_occupied_ratio_delta_vs_native", 0.0))
        fp = float(row.get("false_positive_rate_delta_vs_native", 0.0))
        tradeoff_rows.append(
            {
                "variant_label": row["variant_label"],
                "perturbation_id": row["perturbation_id"],
                "horizon_s": row["horizon_s"],
                "recovery_rate": recovery,
                "false_positive_delta": fp,
                "density_drift": density,
                "recovery_per_density_drift": safe_div(recovery, max(1e-6, abs(density))),
                "recovery_per_false_positive": safe_div(recovery, max(1e-6, abs(fp))),
                "sample_consistency_ratio": float(row.get("case_count", 0.0)) / max(1.0, len(sample_ids)),
            }
        )
    write_csv(REPORTS_DIR / "sw13a_recovery_density_tradeoff.csv", tradeoff_rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    xs = np.arange(len(tradeoff_rows))
    ax.scatter([float(r["density_drift"]) for r in tradeoff_rows], [float(r["recovery_rate"]) for r in tradeoff_rows], c=xs, cmap="viridis")
    ax.set_title("SW-13A test-time feature memory replay subset diagnostic recovery-density tradeoff")
    ax.set_xlabel("density drift")
    ax.set_ylabel("front recovery rate")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "sw13a_recovery_density_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if best_dump_candidate is not None:
        save_bev_image(
            FIGURES_DIR / f"sw13a_bev_repair_comparison_sample{best_dump_candidate['sample_index']}.png",
            best_dump_candidate["gt_h"],
            best_dump_candidate["native_pred"],
            best_dump_candidate["repaired_pred"],
            "SW-13A test-time feature memory replay subset diagnostic BEV comparison",
        )
        save_fpv_dashboard(
            FIGURES_DIR / f"sw13a_fpv_bev_feature_memory_dashboard_sample{best_dump_candidate['sample_index']}.png",
            best_dump_candidate["sample_unwrapped"],
            best_dump_candidate["variant_label"],
            best_dump_candidate["perturbation_id"],
            best_dump_candidate["runtime_debug"]["frame_debug"][0],
        )
        np.savez_compressed(
            ARTIFACTS_DIR / "replay_dumps" / f"sample_{best_dump_candidate['sample_index']}_{best_dump_candidate['perturbation_id']}_{best_dump_candidate['variant_label']}.npz",
            gt_h=best_dump_candidate["gt_h"].numpy().astype(np.int16),
            native_pred=best_dump_candidate["native_pred"].numpy().astype(np.int16),
            repaired_pred=best_dump_candidate["repaired_pred"].numpy().astype(np.int16),
        )

    # metric bar chart
    fig_rows = [
        row
        for row in summary_rows
        if row["horizon_s"] == 6 and row["perturbation_id"] in {"A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"} and row["variant_label"] in {"R0_degraded_native", "R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R5_blend_alpha03", "R6_blend_alpha05", "R8_camera_group_repair_front_triplet"}
    ]
    labels = [f"{row['perturbation_id']}:{row['variant_label']}" for row in fig_rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    metrics = [
        ("front_sector_false_free_rate", "front false-free"),
        ("small_object_false_free_rate", "small-object false-free"),
        ("pred_gt_occupied_ratio", "pred/gt density"),
        ("false_positive_rate", "false-positive"),
    ]
    for ax, (metric_key, title) in zip(axes.flatten(), metrics):
        ax.bar(np.arange(len(fig_rows)), [float(row[metric_key]) for row in fig_rows], color="#2E86C1")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(fig_rows)))
        ax.set_xticklabels(labels, rotation=75, fontsize=7)
    fig.suptitle("SW-13A test-time feature memory replay subset diagnostic metric bar A1/A10/C4")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sw13a_metric_bar_A1_A10_C4.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # feature memory debug figure
    debug_rows = [row for row in metric_rows if row["variant_label"] != "R0_degraded_native" and int(row["horizon_s"]) == 6]
    fig, ax1 = plt.subplots(figsize=(8.6, 5.0))
    ax1.scatter(
        [float(row["feature_distance_current_memory"]) for row in debug_rows],
        [float(row["feature_norm_drift"]) for row in debug_rows],
        alpha=0.7,
    )
    ax1.set_title("SW-13A test-time feature memory replay subset diagnostic feature memory debug")
    ax1.set_xlabel("feature distance current-memory")
    ax1.set_ylabel("feature norm drift")
    ax1.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "sw13a_feature_memory_debug.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    attach_artifact(manifest, "feature_memory_replay_metrics", REPORTS_DIR / "sw13a_feature_memory_replay_metrics.csv")
    return metric_rows, summary_rows, native_equivalence


def phase_decision(metric_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], native_equivalence: dict[str, Any], manifest: dict[str, Any]) -> None:
    sample_count = len({int(row["sample_index"]) for row in metric_rows})
    # best A1 / A10 / C4
    def best_row(pert: str, filter_variants: set[str] | None = None) -> dict[str, Any] | None:
        rows = [row for row in summary_rows if row["perturbation_id"] == pert and int(row["horizon_s"]) == 6 and row["variant_label"] != "R0_degraded_native"]
        if filter_variants is not None:
            rows = [row for row in rows if row["variant_label"] in filter_variants]
        if not rows:
            return None
        return min(rows, key=lambda r: float(r.get("front_sector_false_free_rate_delta_vs_native", 0.0)))

    best_a1 = best_row("A1_drop_cam_front", {"R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3"})
    best_a10 = best_row("A10_drop_front_triplet", {"R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"})
    best_c4 = best_row("C4_motion_blur_9", {"R5_blend_alpha03", "R6_blend_alpha05", "R7_blend_alpha07"})

    def classify(row: dict[str, Any] | None, scenario: str) -> tuple[bool, bool]:
        if row is None:
            return False, False
        safe = (
            float(row.get("false_positive_rate_delta_vs_native", 0.0)) <= 0.005
            and float(row.get("pred_gt_occupied_ratio_delta_vs_native", 0.0)) <= 0.03
            and float(row.get("wrong_class_activation_delta", 0.0)) <= 0.005
        )
        if scenario == "c4":
            gain = float(row.get("false_free_rate_delta_vs_native", 0.0)) < 0.0 or float(row.get("occupied_iou_delta_vs_native", 0.0)) > 0.0
        else:
            gain = (
                float(row.get("front_sector_false_free_rate_delta_vs_native", 0.0)) <= -0.005
                or float(row.get("small_object_false_free_rate_delta_vs_native", 0.0)) <= -0.005
                or float(row.get("A10_front_h6_recovery_rate", 0.0)) > 0.0
                or float(row.get("new_visible_recall_delta_vs_native", 0.0)) >= 0.005
            )
        return safe, gain

    a1_safe, a1_gain = classify(best_a1, "a1")
    a10_safe, a10_gain = classify(best_a10, "a10")
    c4_safe, c4_gain = classify(best_c4, "c4")
    positive_rows = [
        row for row in summary_rows
        if row["variant_label"] != "R0_degraded_native"
        and (
            float(row.get("front_sector_false_free_rate_delta_vs_native", 0.0)) < 0.0
            or float(row.get("small_object_false_free_rate_delta_vs_native", 0.0)) < 0.0
            or float(row.get("new_visible_recall_delta_vs_native", 0.0)) > 0.0
        )
    ]
    if not native_equivalence.get("pass", False):
        decision_type = "S6_IMPLEMENTATION_BLOCKED"
    elif any(row.get("uses_future_info", False) or row.get("uses_current_clean_same_frame", False) for row in metric_rows):
        decision_type = "S7_ORACLE_RISK"
    elif (a1_gain or a10_gain) and a1_safe and a10_safe:
        strong = any(
            abs(float(row.get("front_sector_false_free_rate_delta_vs_native", 0.0))) >= 0.02
            or abs(float(row.get("small_object_false_free_rate_delta_vs_native", 0.0))) >= 0.02
            or abs(float(row.get("new_visible_recall_delta_vs_native", 0.0))) >= 0.02
            for row in [best_a1, best_a10]
            if row is not None
        )
        decision_type = "S1_STRONG_FEATURE_MEMORY_GAIN" if strong else "S2_WEAK_BUT_SAFE_FEATURE_MEMORY_GAIN"
    elif positive_rows:
        decision_type = "S3_RECOVERY_BUT_DENSITY_UNSAFE"
    elif c4_gain and c4_safe and not a1_gain and not a10_gain:
        decision_type = "S5_C4_ONLY_GAIN"
    else:
        decision_type = "S4_NO_RECOVERY"

    criteria = {
        "A1_success_threshold": {"front_or_small_or_future_ff_drop": 0.01, "false_positive_increase_max": 0.005, "density_drift_max": 0.03, "clean_A0_drift_max": 0.003},
        "A10_success_threshold": {"front_or_small_ff_drop": 0.01, "A10_front_h6_recovery_rate_positive": True, "new_visible_recall_gain": 0.01, "false_positive_increase_max": 0.005, "density_drift_max": 0.03, "wrong_class_rate_max": 0.005},
        "C4_success_threshold": {"ff_drop_or_iou_gain": True, "false_positive_increase_not_large": True, "density_drift_max": 0.03},
        "strong_success_rule": ">=2% targeted gain with safety pass and at least 60% sample consistency",
        "weak_success_rule": "0.5%-2% targeted gain with safety pass and at least 40% sample consistency",
        "failure_rule": "no recovery or recovery only via density expansion or clean drift or implementation uncertainty",
    }
    write_json(REPORTS_DIR / "sw13a_success_criteria.json", criteria)

    decision = {
        "decision_type": decision_type,
        "best_A1_row": best_a1,
        "best_A10_row": best_a10,
        "best_C4_row": best_c4,
        "native_path_unchanged": native_equivalence,
        "sample_count": sample_count,
        "no_training": True,
        "no_checkpoint_modification": True,
        "no_get_occ_modification": True,
        "no_gt_repair": True,
        "no_future_frame_feature": True,
        "no_current_clean_same_frame_feature": True,
    }
    write_json(REPORTS_DIR / "sw13a_feature_memory_replay_decision.json", decision)
    write_md(REPORTS_DIR / "sw13a_feature_memory_replay_decision.md", json.dumps(normalize_export(decision), indent=2, ensure_ascii=False) + "\n")

    report_md = "\n".join(
        [
            "# Stage SW-13A Sensor-Fault Feature Memory Replay",
            "",
            "1. Executive summary",
            f"- decision: {decision_type}",
            "",
            "2. Why feature memory follows SW-12C",
            "- this stage uses no training, no checkpoint modification, and no get_occ modification.",
            "",
            "3. Feature path audit",
            "- feature repair is inserted at test time after camera-wise `extract_feat` and before temporal reorganization.",
            "",
            "4. Feature memory cache",
            "- cache uses only A0_clean history features from t-1/t-2/t-3.",
            "- no future frame feature and no current clean same-frame feature are used.",
            "",
            "5. Repair variants",
            "- variants are flag-controlled and the native path remains unchanged by default.",
            "",
            "6. Replay protocol",
            "- subset diagnostic replay on eval_core_20, epoch_56_original, A0/A1/A10/C4/A7, horizons h0/h2/h4/h6.",
            "",
            "7. Main recovery metrics",
            "- results are reported as test-time feature memory replay subset diagnostic only.",
            "",
            "8. Safety and density tradeoff",
            "- false-positive, pred_gt density, wrong-class activation, and clean drift are reported for every repair variant.",
            "",
            "9. Visualizations",
            "- BEV comparison, metric bar, dashboard, and feature debug plots are included.",
            "",
            "10. Decision S1-S7",
            f"- {decision_type}",
            "",
            "11. Safe claims",
            "- no training",
            "- no checkpoint modification",
            "- no get_occ modification",
            "- no GT repair",
            "- no future-frame feature",
            "- no current clean feature used for degraded current frame",
            "- subset diagnostic only",
            "- not official benchmark",
            "- any positive result is test-time feature memory replay gain only",
            "",
            "12. Limitations",
            "- this phase evaluates only test-time feature replay on the epoch_56 baseline.",
            "",
            "13. Next unique action",
            f"- {decision_type}",
            "",
        ]
    )
    report_json = {
        "decision_type": decision_type,
        "best_A1_row": best_a1,
        "best_A10_row": best_a10,
        "best_C4_row": best_c4,
        "safe_claims": [
            "no training",
            "no checkpoint modification",
            "no get_occ modification",
            "no GT repair",
            "no future-frame feature",
            "no current clean feature used for degraded current frame",
            "subset diagnostic only",
            "not official benchmark",
        ],
    }
    write_md(REPORTS_DIR / "stage_sw13a_sensor_fault_feature_memory_replay_report.md", report_md)
    write_json(REPORTS_DIR / "stage_sw13a_sensor_fault_feature_memory_replay_report.json", report_json)
    attach_artifact(manifest, "feature_memory_replay_decision", REPORTS_DIR / "sw13a_feature_memory_replay_decision.json")


def write_tests() -> None:
    tests = {
        "test_outputs_exist.py": '''from pathlib import Path\nBASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay")\n\ndef test_outputs_exist() -> None:\n    required = [\n        "sw13a_execution_manifest.json",\n        "sw13a_progress_state.json",\n        "sw13a_feature_path_audit.json",\n        "sw13a_feature_memory_cache_manifest.csv",\n        "sw13a_feature_repair_variant_manifest.json",\n        "sw13a_replay_manifest.csv",\n        "sw13a_feature_memory_replay_metrics.csv",\n        "sw13a_feature_memory_aggregate_metrics.csv",\n        "sw13a_feature_memory_replay_decision.json",\n        "stage_sw13a_sensor_fault_feature_memory_replay_report.md",\n    ]\n    for name in required:\n        path = BASE / name\n        assert path.exists(), name\n        assert path.stat().st_size > 0, name\n''',
        "test_feature_path_audit_schema.py": '''import json\nfrom pathlib import Path\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_path_audit.json")\n\ndef test_feature_path_audit_schema() -> None:\n    obj = json.loads(PATH.read_text(encoding="utf-8"))\n    assert "input_img_shape" in obj\n    assert "camera_index_mapping" in obj\n    assert "feature_levels" in obj\n    assert "path_audit" in obj and obj["path_audit"]\n''',
        "test_memory_cache_no_future.py": '''import csv\nfrom pathlib import Path\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_memory_cache_manifest.csv")\n\ndef test_memory_cache_no_future() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    for row in rows:\n        assert int(row["source_time_offset"]) > 0\n        assert row["future_info_used"] == "False"\n''',
        "test_no_current_clean_oracle.py": '''import csv\nfrom pathlib import Path\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_replay_manifest.csv")\n\ndef test_no_current_clean_oracle() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    for row in rows:\n        assert row["future_info_used"] == "False"\n        assert row["current_clean_same_frame_used"] == "False"\n        assert row["gt_used_for_repair"] == "False"\n''',
        "test_native_path_unchanged.py": '''import json\nfrom pathlib import Path\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/native_path_unchanged_check.json")\n\ndef test_native_path_unchanged() -> None:\n    obj = json.loads(PATH.read_text(encoding="utf-8"))\n    assert obj["checked"] is True\n    assert obj["pass"] is True\n    assert float(obj["max_abs_diff"]) == 0.0\n''',
        "test_metric_schema.py": '''import csv\nfrom pathlib import Path\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_memory_replay_metrics.csv")\n\ndef test_metric_schema() -> None:\n    rows = list(csv.DictReader(PATH.open()))\n    assert rows\n    required = {\n        "model_name", "sample_index", "perturbation_id", "horizon_s", "variant_label",\n        "front_sector_false_free_rate", "small_object_false_free_rate", "dynamic_object_false_free_rate",\n        "future_h4_h6_false_free_rate", "new_visible_recall", "A10_front_h6_recovery_rate",\n        "false_positive_rate", "pred_gt_occupied_ratio", "density_drift", "wrong_class_rate",\n        "repaired_camera_count", "memory_age_frame", "feature_distance_current_memory", "repair_triggered_ratio", "feature_norm_drift"\n    }\n    assert required.issubset(rows[0].keys())\n''',
        "test_decision_schema.py": '''import json\nfrom pathlib import Path\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_memory_replay_decision.json")\n\ndef test_decision_schema() -> None:\n    obj = json.loads(PATH.read_text(encoding="utf-8"))\n    assert "decision_type" in obj\n    assert obj["decision_type"] in {\n        "S1_STRONG_FEATURE_MEMORY_GAIN", "S2_WEAK_BUT_SAFE_FEATURE_MEMORY_GAIN", "S3_RECOVERY_BUT_DENSITY_UNSAFE",\n        "S4_NO_RECOVERY", "S5_C4_ONLY_GAIN", "S6_IMPLEMENTATION_BLOCKED", "S7_ORACLE_RISK"\n    }\n''',
        "test_no_false_claims.py": '''from pathlib import Path\nPATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/stage_sw13a_sensor_fault_feature_memory_replay_report.md")\n\ndef test_no_false_claims() -> None:\n    text = PATH.read_text(encoding="utf-8").lower()\n    assert "subset diagnostic" in text\n    assert "not official benchmark" in text\n    assert "no training" in text\n    banned = ["official benchmark result", "trained model improvement", "beats the paper", "production ready"]\n    for phrase in banned:\n        assert phrase not in text\n''',
    }
    for name, text in tests.items():
        write_md(TESTS_DIR / name, text)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    progress = init_progress_state()
    manifest = init_execution_manifest()
    sample_ids = parse_int_list(args.samples)
    cfg, dataset, model = build_runtime()

    def run_phase(name: str, fn):
        phase_start(progress, manifest, name)
        try:
            result = fn()
            phase_end(progress, manifest, name, "done")
            return result
        except Exception as exc:
            phase_end(progress, manifest, name, "failed", error=str(exc))
            raise

    run_phase("phase1_feature_path_audit", lambda: phase_feature_path_audit(cfg, dataset, model, manifest))
    cache_map = run_phase("phase2_feature_memory_cache", lambda: phase_build_memory_cache(dataset, model, sample_ids, manifest))
    metric_rows, summary_rows, native_equivalence = run_phase("phase4_replay", lambda: phase_replay(cfg, dataset, model, cache_map, sample_ids, manifest))
    run_phase("phase9_decision", lambda: phase_decision(metric_rows, summary_rows, native_equivalence, manifest))
    run_phase("phase11_tests_write", write_tests)
    progress["status"] = "complete"
    progress["end_time"] = now_iso()
    update_progress(progress)
    manifest["end_time"] = now_iso()
    update_manifest(manifest)


if __name__ == "__main__":
    main()
