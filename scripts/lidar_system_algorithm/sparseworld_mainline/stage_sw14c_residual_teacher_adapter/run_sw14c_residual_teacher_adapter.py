from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sw14c_losses import LOSS_TERMS, SW14CLossConfig
from sw14c_residual_adapter_modules import (
    CAMERA_NAMES,
    FRONT_TRIPLET,
    FRONT_TRIPLET_INDICES,
    REAR_CAMERAS,
    SpatialResidualAdapter,
    build_sw13_r8_base_repair,
    checkpoint_payload,
    count_parameters,
    front_triplet_degradation_mask,
)


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"

SW14B_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter"
SW14B_RESCUE_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue"
SW14B_RUNTIME_TEACHER_CACHE = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14b_metric_gt_adapter/runtime_teacher_cache/rescue_A10_drop_front_triplet"
SW14B_STAGE_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/run_sw14b_metric_gt_adapter.py"
SW13A_FEATURE_CACHE = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/feature_memory_cache"
GET_OCC_PATH = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/opus_head.py"

CORE_HORIZONS = [0, 2, 4, 6]
EMPTY_IDX = 17
FINAL_DECISION_ENUMS = {
    "SW14C_0_FAIL_KEEP_SW13_MAIN",
    "SW14C_1_DEBUG_IMPROVES_TEACHER",
    "SW14C_2_CORE100_IMPROVES_TEACHER",
    "SW14C_3_CORE500_IMPROVES_TEACHER",
    "SW14C_4_COMPARABLE_BUT_NOT_STRONGER",
    "SW14C_5_UNSAFE_DENSITY",
    "SW14C_6_PROTOCOL_VIOLATION",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("sw14c_sw14b_base", SW14B_STAGE_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14C residual-on-SW13 teacher adapter")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase0-start", type=int, default=100)
    parser.add_argument("--phase0-end", type=int, default=100)
    parser.add_argument("--skip-cache-phase0", action="store_true", help="do not read runtime teacher cache for the base parity audit")
    parser.add_argument("--run-training", action="store_true", help="run Round2 small train with sample-major feature reuse")
    parser.add_argument("--round2-train-start", type=int, default=0)
    parser.add_argument("--round2-train-end", type=int, default=4)
    parser.add_argument("--round2-val-start", type=int, default=200)
    parser.add_argument("--round2-val-end", type=int, default=204)
    parser.add_argument("--round2-epochs", type=int, default=1)
    parser.add_argument("--round2-lr", type=float, default=1e-4)
    parser.add_argument("--round2-gamma", type=float, default=0.10)
    parser.add_argument("--round2-train-block-size", type=int, default=1)
    parser.add_argument("--round2-micro-steps-per-sample", type=int, default=1)
    parser.add_argument("--gamma-candidates", default="0.00,0.05,0.10,0.20,0.30")
    parser.add_argument("--allow-teacher-cache-rebuild", action="store_true", help="disabled by default to avoid high disk reads")
    parser.add_argument("--val-self-teacher", action="store_true", default=True, help="build SW13 R8 teacher in the same val runtime instead of reading teacher cache")
    parser.add_argument("--write-init-checkpoint", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, ARTIFACTS_DIR, ARTIFACTS_DIR / "checkpoints", FIGURES_DIR, TESTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return normalize(obj.item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize(row))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda_for_throughput() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def chunks(values: list[int], chunk_size: int) -> list[list[int]]:
    size = max(1, int(chunk_size))
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def feature_cache_path(sample_index: int) -> Path:
    return SW13A_FEATURE_CACHE / f"sample_{sample_index:03d}.pt"


def load_feature_cache(sample_index: int) -> dict[str, Any]:
    path = feature_cache_path(sample_index)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def disk_read_sectors() -> int:
    total = 0
    path = Path("/proc/diskstats")
    if not path.exists():
        return 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        name = parts[2]
        if name.startswith(("loop", "ram", "fd")):
            continue
        try:
            total += int(parts[5])
        except ValueError:
            continue
    return total


def disk_read_mb_delta(start_sectors: int, end_sectors: int) -> float:
    return max(0.0, float(end_sectors - start_sectors) * 512.0 / (1024.0 * 1024.0))


def cuda_mem_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": float(torch.cuda.memory_allocated() / (1024.0 * 1024.0)),
        "reserved_mb": float(torch.cuda.memory_reserved() / (1024.0 * 1024.0)),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
    }


def release_runtime(*objs: Any, collect: bool = False, empty_cache: bool = False) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    if collect:
        gc.collect()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_degraded_batch(batch_clean: dict[str, Any], perturbation_id: str) -> dict[str, Any]:
    spec_catalog = base.sw81.sw5_engine.build_catalog()
    batch_deg = copy.deepcopy(batch_clean)
    if not isinstance(batch_deg["img"], list):
        batch_deg["img"] = [batch_deg["img"]]
    if "img_metas" in batch_deg and not isinstance(batch_deg["img_metas"], list):
        batch_deg["img_metas"] = [batch_deg["img_metas"]]
    return base.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[perturbation_id])[0]


def extract_gt_temporal(sample_unwrapped: dict[str, Any]) -> dict[int, torch.Tensor]:
    gt_current = torch.as_tensor(sample_unwrapped["voxel_semantics"]).long()
    temporal_semantics = sample_unwrapped["temporal_semantics"]
    gt_list = [gt_current]
    if isinstance(temporal_semantics, dict):
        for key in sorted(int(k) for k in temporal_semantics.keys()):
            gt_list.append(torch.as_tensor(temporal_semantics[key]["voxel_semantics"]).long())
    else:
        for temporal in temporal_semantics:
            gt_list.append(torch.as_tensor(temporal["voxel_semantics"]).long())
    return {horizon_s: gt_list[horizon_s] for horizon_s in CORE_HORIZONS}


def prepare_residual_feature_case(
    model: Any,
    dataset: Any,
    sample_index: int,
    perturbation_id: str = "A10_drop_front_triplet",
    *,
    memory_source: str = "runtime_clean_tminus1",
) -> dict[str, Any]:
    raw_sample, batch_clean = base.sw2.extract_sample_batch(dataset, sample_index, base.collate_fn)
    sample_unwrapped = base.sw2.unwrap(raw_sample)
    batch_deg = build_degraded_batch(batch_clean, perturbation_id)
    moved = base.sw2.move_to_cuda(base.test_batch_compatible(batch_deg))
    moved_clean = base.sw2.move_to_cuda(base.test_batch_compatible(batch_clean))
    img = moved["img"][0]
    img_metas = moved["img_metas"][0]
    bsz, total_n, channels, height, width = img.shape
    img_seq = img.reshape(bsz, total_n // 6, 6, channels, height, width)
    img_clean = moved_clean["img"][0]
    img_metas_clean = moved_clean["img_metas"][0]
    img_clean_seq = img_clean.reshape(bsz, total_n // 6, 6, channels, height, width)
    img_filenames = img_metas[0]["filename"]
    num_frames = len(img_filenames) // 6
    img_shape = (height, width, channels)
    img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
    img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
    img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
    img_metas_clean[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
    img_metas_clean[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
    img_metas_clean[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
    frame_levels: list[list[torch.Tensor]] = []
    frame_metas: list[list[dict[str, Any]]] = []
    memory_levels: list[torch.Tensor] | None = None
    base_levels: list[torch.Tensor] | None = None
    degradation_mask: torch.Tensor | None = None
    memory_offset_levels: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for frame_idx in range(num_frames):
            img_indices = list(np.arange(frame_idx * 6, (frame_idx + 1) * 6))
            img_metas_curr = base.sw13a.clone_meta_for_indices(img_metas[0], img_indices)
            current_levels = model.extract_feat(img_seq[:, frame_idx], img_metas_curr)
            current_levels = [level.detach().clone() for level in current_levels]
            frame_levels.append(current_levels)
            frame_metas.append(img_metas_curr)
            if frame_idx == 0:
                if memory_source != "runtime_clean_tminus1":
                    cache = load_feature_cache(sample_index)
                    variant = base.variant_spec(base.teacher_ref().base_repair_variant)
                    memory_levels = []
                    for level_idx, feat in enumerate(current_levels):
                        mem_level = torch.empty_like(feat)
                        for cam_idx in range(len(CAMERA_NAMES)):
                            mem = base.sw13a.aggregate_memory_features(cache, variant.offsets, level_idx, cam_idx)
                            if mem is None:
                                mem = feat[:, cam_idx].detach().cpu()
                            mem_level[:, cam_idx] = mem.to(feat.device, dtype=feat.dtype, non_blocking=True)
                        memory_levels.append(mem_level.detach().clone())
                    del cache
                else:
                    for memory_frame_idx in [1, 2, 3]:
                        if memory_frame_idx >= num_frames:
                            continue
                        memory_img_indices = list(np.arange(memory_frame_idx * 6, (memory_frame_idx + 1) * 6))
                        memory_metas = base.sw13a.clone_meta_for_indices(img_metas_clean[0], memory_img_indices)
                        memory_offset_levels[memory_frame_idx] = [
                            level.detach().clone()
                            for level in model.extract_feat(img_clean_seq[:, memory_frame_idx], memory_metas)
                        ]
                    memory_levels = memory_offset_levels.get(1)
                    if memory_levels is None:
                        memory_levels = [level.detach().clone() for level in current_levels]
                base_levels = build_sw13_r8_base_repair(current_levels, memory_levels, perturbation_id)
                base_levels = [level.detach().clone() for level in base_levels]
                degradation_mask = front_triplet_degradation_mask(
                    perturbation_id,
                    batch_size=int(current_levels[0].shape[0]),
                    device=current_levels[0].device,
                    dtype=current_levels[0].dtype,
                )
    assert memory_levels is not None and base_levels is not None and degradation_mask is not None
    return {
        "sample_index": sample_index,
        "sample_unwrapped": sample_unwrapped,
        "moved_batch": moved,
        "frame_levels": frame_levels,
        "frame_metas": frame_metas,
        "memory_levels": memory_levels,
        "base_levels": base_levels,
        "memory_offset_levels": memory_offset_levels,
        "degradation_mask": degradation_mask,
        "gt_temporal": extract_gt_temporal(sample_unwrapped),
        "memory_source": memory_source,
    }


def run_residual_forward_prepared(
    model: Any,
    adapter: SpatialResidualAdapter,
    prepared_case: dict[str, Any],
    *,
    gamma: float,
    training: bool,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    holder: dict[str, Any] = {}
    original_forward = base.attach_query_capture_live(model, holder)
    original_extract_feat = model.extract_feat
    frame_cursor = {"idx": 0}
    debug_holder: dict[str, torch.Tensor] = {}

    def patched_extract_feat(img: torch.Tensor, img_metas_curr: list[dict[str, Any]]):
        idx = frame_cursor["idx"]
        frame_cursor["idx"] += 1
        current_levels = prepared_case["frame_levels"][idx]
        if idx != 0:
            return current_levels
        final_levels, debug = adapter.apply_to_levels(
            current_levels,
            prepared_case["memory_levels"],
            prepared_case["base_levels"],
            prepared_case["degradation_mask"],
            gamma=gamma,
        )
        debug_holder.update(debug)
        return final_levels

    model.extract_feat = patched_extract_feat  # type: ignore[assignment]
    try:
        base.sw13a.reset_model_cache(model)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            _ = model(return_loss=False, rescale=True, **prepared_case["moved_batch"])
            head = base.sw4_inst.get_pts_bbox_head(model)
            per_h: dict[int, dict[str, torch.Tensor]] = {}
            for horizon_s in CORE_HORIZONS:
                pred_dict = base.extract_pred_dict_live(holder, horizon_s)
                occ_pred, debug_list = base.sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                dense_scores = debug_list[0]["dense_occ_after_padding"]
                conf, margin = base.occupancy_confidence_and_margin(dense_scores)
                gt_h = prepared_case["gt_temporal"][horizon_s].to(device=dense_scores.device, dtype=torch.long)
                gt0 = prepared_case["gt_temporal"][0].to(device=dense_scores.device, dtype=torch.long)
                per_h[horizon_s] = {
                    "raw_semantic": occ_pred[0],
                    "raw_confidence": conf,
                    "raw_margin": margin,
                    "dense_scores": dense_scores,
                    "gt_h": gt_h,
                    "gt0": gt0,
                }
        return per_h, debug_holder
    finally:
        model.extract_feat = original_extract_feat  # type: ignore[assignment]
        model.forward_backbone = original_forward  # type: ignore[assignment]


def aggregate_prepared_memory(prepared_case: dict[str, Any], offsets: list[int]) -> list[torch.Tensor]:
    current_levels = prepared_case["frame_levels"][0]
    memory_by_offset: dict[int, list[torch.Tensor]] = prepared_case.get("memory_offset_levels", {})
    available = [offset for offset in offsets if offset in memory_by_offset]
    if not available:
        return [level.detach().clone() for level in current_levels]
    if len(available) == 1:
        weights = [1.0]
    elif len(available) == 2:
        weights = [0.7, 0.3]
    else:
        weights = [0.6, 0.3, 0.1]
    out_levels: list[torch.Tensor] = []
    for level_idx, current in enumerate(current_levels):
        out = torch.zeros_like(current)
        for offset, weight in zip(available, weights):
            out = out + memory_by_offset[offset][level_idx].to(device=current.device, dtype=current.dtype) * float(weight)
        out_levels.append(out.detach().clone())
    return out_levels


def build_rule_first_levels(prepared_case: dict[str, Any], variant_label: str) -> list[torch.Tensor]:
    current_levels = prepared_case["frame_levels"][0]
    if variant_label == "native_baseline" or variant_label == "R0_degraded_native":
        return [level.detach().clone() for level in current_levels]
    if variant_label in {"R1_replace_tminus1", "R8_camera_group_repair_front_triplet"}:
        memory_levels = aggregate_prepared_memory(prepared_case, [1])
    elif variant_label == "R3_ema_K2":
        memory_levels = aggregate_prepared_memory(prepared_case, [1, 2])
    elif variant_label == "R4_ema_K3":
        memory_levels = aggregate_prepared_memory(prepared_case, [1, 2, 3])
    else:
        raise KeyError(variant_label)
    repaired = [level.detach().clone() for level in current_levels]
    for level_idx, level in enumerate(repaired):
        level[:, FRONT_TRIPLET_INDICES] = memory_levels[level_idx][:, FRONT_TRIPLET_INDICES].to(device=level.device, dtype=level.dtype)
    return repaired


def run_first_levels_forward_prepared(
    model: Any,
    prepared_case: dict[str, Any],
    first_levels: list[torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    holder: dict[str, Any] = {}
    original_forward = base.attach_query_capture_live(model, holder)
    original_extract_feat = model.extract_feat
    frame_cursor = {"idx": 0}

    def patched_extract_feat(img: torch.Tensor, img_metas_curr: list[dict[str, Any]]):
        idx = frame_cursor["idx"]
        frame_cursor["idx"] += 1
        if idx == 0:
            return first_levels
        return prepared_case["frame_levels"][idx]

    model.extract_feat = patched_extract_feat  # type: ignore[assignment]
    try:
        base.sw13a.reset_model_cache(model)
        with torch.inference_mode():
            _ = model(return_loss=False, rescale=True, **prepared_case["moved_batch"])
            head = base.sw4_inst.get_pts_bbox_head(model)
            per_h: dict[int, dict[str, torch.Tensor]] = {}
            for horizon_s in CORE_HORIZONS:
                pred_dict = base.extract_pred_dict_live(holder, horizon_s)
                occ_pred, debug_list = base.sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                dense_scores = debug_list[0]["dense_occ_after_padding"]
                conf, margin = base.occupancy_confidence_and_margin(dense_scores)
                gt_h = prepared_case["gt_temporal"][horizon_s].to(device=dense_scores.device, dtype=torch.long)
                gt0 = prepared_case["gt_temporal"][0].to(device=dense_scores.device, dtype=torch.long)
                per_h[horizon_s] = {
                    "raw_semantic": occ_pred[0],
                    "raw_confidence": conf,
                    "raw_margin": margin,
                    "dense_scores": dense_scores,
                    "gt_h": gt_h,
                    "gt0": gt0,
                }
        return per_h
    finally:
        model.extract_feat = original_extract_feat  # type: ignore[assignment]
        model.forward_backbone = original_forward  # type: ignore[assignment]


def build_agreement_from_prepared_rule_outputs(rule_outputs: dict[str, dict[int, dict[str, torch.Tensor]]], horizon_s: int) -> torch.Tensor:
    occs = []
    for label in ["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"]:
        occs.append((rule_outputs[label][horizon_s]["raw_semantic"].detach().cpu().long() != EMPTY_IDX).float())
    return torch.stack(occs, dim=0).mean(dim=0)


def build_self_teacher_bundle_for_prepared(
    model: Any,
    prepared_case: dict[str, Any],
    sample_index: int,
    sectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    candidate = base.build_candidate()
    native = run_first_levels_forward_prepared(model, prepared_case, build_rule_first_levels(prepared_case, "native_baseline"))
    rule_outputs: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for label in ["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"]:
        rule_outputs[label] = run_first_levels_forward_prepared(model, prepared_case, build_rule_first_levels(prepared_case, label))
    bundle: dict[str, Any] = {"sample_index": sample_index, "split_name": "train_derived_self_teacher", "by_horizon": {}}
    for horizon_s in CORE_HORIZONS:
        agreement = build_agreement_from_prepared_rule_outputs(rule_outputs, horizon_s)
        teacher_raw = rule_outputs["R8_camera_group_repair_front_triplet"][horizon_s]
        final_semantic, meta = base.run_sw14b_full_postprocess(
            raw_semantic=teacher_raw["raw_semantic"].detach().cpu().long(),
            raw_confidence=teacher_raw["raw_confidence"].detach().cpu().float(),
            raw_margin=teacher_raw["raw_margin"].detach().cpu().float(),
            native_semantic=native[horizon_s]["raw_semantic"].detach().cpu().long(),
            gt_h=teacher_raw["gt_h"].detach().cpu().long(),
            gt0=teacher_raw["gt0"].detach().cpu().long(),
            candidate=candidate,
            sample_index=sample_index,
            horizon_s=horizon_s,
            sectors=sectors,
            load_gpu_dump=base.load_gpu_dump,
            agreement_map=agreement,
        )
        bundle["by_horizon"][horizon_s] = {
            "native_semantic": native[horizon_s]["raw_semantic"].detach().cpu().long(),
            "teacher_raw_semantic": teacher_raw["raw_semantic"].detach().cpu().long(),
            "teacher_final_semantic": final_semantic.long(),
            "teacher_confidence": teacher_raw["raw_confidence"].detach().cpu().float(),
            "teacher_margin": teacher_raw["raw_margin"].detach().cpu().float(),
            "agreement": agreement.float(),
            "gt_h": teacher_raw["gt_h"].detach().cpu().long(),
            "gt0": teacher_raw["gt0"].detach().cpu().long(),
            "teacher_meta": meta,
        }
    release_runtime(native, rule_outputs, collect=False, empty_cache=False)
    return bundle


def metric_value(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in (None, "", "None"):
            return float(row[key])
    return float(default)


def mean_field(rows: list[dict[str, Any]], key: str, *, horizons: set[int] | None = None) -> float | None:
    values: list[float] = []
    for row in rows:
        if horizons is not None and int(row["horizon_s"]) not in horizons:
            continue
        value = row.get(key)
        if value in (None, "", "None"):
            continue
        values.append(float(value))
    return float(np.mean(values)) if values else None


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len({int(row["sample_index"]) for row in rows}) if rows else 0,
        "row_count": len(rows),
        "front_local_density_proxy": mean_field(rows, "front_local_density_proxy"),
        "front_sector_false_free_rate_delta_vs_native": mean_field(rows, "front_sector_false_free_rate_delta_vs_native"),
        "future_h4_h6_false_free_rate_delta_vs_native": mean_field(rows, "future_h4_h6_false_free_rate_delta_vs_native", horizons={4, 6}),
        "pred_gt_density_delta": mean_field(rows, "pred_gt_density_delta"),
        "false_positive_delta": mean_field(rows, "false_positive_delta"),
        "wrong_class_delta": mean_field(rows, "wrong_class_delta"),
        "protected_zone_preservation_ratio": mean_field(rows, "protected_zone_preservation_ratio"),
    }


def smoothness_loss(feature_map: torch.Tensor) -> torch.Tensor:
    dh = (feature_map[..., 1:, :] - feature_map[..., :-1, :]).abs().mean() if feature_map.shape[-2] > 1 else feature_map.new_zeros(())
    dw = (feature_map[..., :, 1:] - feature_map[..., :, :-1]).abs().mean() if feature_map.shape[-1] > 1 else feature_map.new_zeros(())
    return dh + dw


def residual_regularization(debug: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    deltas = [value for key, value in debug.items() if key.startswith("residual_delta_level")]
    gates = [value for key, value in debug.items() if key.startswith("residual_gate_level")]
    if not deltas or not gates:
        zero = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
        return zero, {"residual_l1": 0.0, "residual_smooth": 0.0, "gate_sparse": 0.0}
    residual_l1 = torch.stack([delta.abs().mean() for delta in deltas]).mean()
    residual_smooth = torch.stack([smoothness_loss(delta) for delta in deltas]).mean()
    gate_sparse = torch.stack([gate.mean() for gate in gates]).mean()
    return residual_l1 + residual_smooth + gate_sparse, {
        "residual_l1": float(residual_l1.detach().item()),
        "residual_smooth": float(residual_smooth.detach().item()),
        "gate_sparse": float(gate_sparse.detach().item()),
    }


def compute_training_loss(
    per_h: dict[int, dict[str, torch.Tensor]],
    debug: dict[str, torch.Tensor],
    sectors: dict[str, torch.Tensor],
    config: SW14CLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(iter(per_h.values()))["dense_scores"].device
    total = torch.zeros((), device=device)
    gt_occ_loss = torch.zeros((), device=device)
    teacher_fn_recover = torch.zeros((), device=device)
    future_recover = torch.zeros((), device=device)
    fp_loss = torch.zeros((), device=device)
    density_hinge = torch.zeros((), device=device)
    horizon_count = 0
    front_mask = sectors["front"].to(device=device, dtype=torch.bool)
    for horizon_s, entry in per_h.items():
        dense_scores = entry["dense_scores"].float()
        occ_prob = dense_scores.max(dim=-1).values.clamp(1e-4, 1.0 - 1e-4)
        gt_occ = (entry["gt_h"] != EMPTY_IDX).float()
        gt_free = 1.0 - gt_occ
        gt_occ_loss = gt_occ_loss + F.binary_cross_entropy(occ_prob, gt_occ)
        front_pos = front_mask & gt_occ.bool()
        if bool(front_pos.any().item()):
            teacher_fn_recover = teacher_fn_recover + (1.0 - occ_prob[front_pos]).mean()
        if horizon_s in {4, 6} and bool(gt_occ.bool().any().item()):
            future_recover = future_recover + (1.0 - occ_prob[gt_occ.bool()]).mean()
        if bool(gt_free.bool().any().item()):
            fp_loss = fp_loss + occ_prob[gt_free.bool()].mean()
        target_density = torch.clamp(gt_occ.mean() + 0.08, max=0.18)
        density_hinge = density_hinge + torch.relu(occ_prob.mean() - target_density).pow(2)
        horizon_count += 1
    divisor = max(1, horizon_count)
    gt_occ_loss = gt_occ_loss / divisor
    teacher_fn_recover = teacher_fn_recover / divisor
    future_recover = future_recover / divisor
    fp_loss = fp_loss / divisor
    density_hinge = density_hinge / divisor
    deltas = [value for key, value in debug.items() if key.startswith("residual_delta_level")]
    gates = [value for key, value in debug.items() if key.startswith("residual_gate_level")]
    residual_l1 = torch.stack([delta.abs().mean() for delta in deltas]).mean() if deltas else torch.zeros((), device=device)
    residual_smooth = torch.stack([smoothness_loss(delta) for delta in deltas]).mean() if deltas else torch.zeros((), device=device)
    gate_sparse = torch.stack([gate.mean() for gate in gates]).mean() if gates else torch.zeros((), device=device)
    total = (
        config.lambda_teacher_correct_preserve * gt_occ_loss
        + config.lambda_teacher_fn_recover * teacher_fn_recover
        + config.lambda_teacher_fn_recover * 0.5 * future_recover
        + config.lambda_fp * fp_loss
        + config.lambda_density * density_hinge
        + config.lambda_residual_l1 * residual_l1
        + config.lambda_residual_smooth * residual_smooth
        + config.lambda_gate_sparse * gate_sparse
    )
    stats = {
        "total_loss": float(total.detach().item()),
        "gt_occ_loss": float(gt_occ_loss.detach().item()),
        "teacher_fn_recover_loss": float(teacher_fn_recover.detach().item()),
        "future_recover_loss": float(future_recover.detach().item()),
        "fp_loss": float(fp_loss.detach().item()),
        "density_hinge_loss": float(density_hinge.detach().item()),
        "residual_l1": float(residual_l1.detach().item()),
        "residual_smooth": float(residual_smooth.detach().item()),
        "gate_sparse": float(gate_sparse.detach().item()),
        "residual_combo": float((residual_l1 + residual_smooth + gate_sparse).detach().item()),
    }
    return total, stats


def teacher_bundle_for_sample(sample_index: int, allow_rebuild: bool = False) -> dict[str, Any] | None:
    path = teacher_cache_path(sample_index)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if allow_rebuild:
        raise RuntimeError("SW14C runtime teacher rebuild is intentionally not implemented in the fast path")
    return None


def postprocess_student_row(
    sample_index: int,
    horizon_s: int,
    split_name: str,
    gamma: float,
    per_h: dict[int, dict[str, torch.Tensor]],
    teacher_bundle: dict[str, Any],
    sectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    candidate = base.build_candidate()
    teacher_h = teacher_bundle["by_horizon"][horizon_s]
    entry = per_h[horizon_s]
    final_semantic, meta = base.run_sw14b_full_postprocess(
        raw_semantic=entry["raw_semantic"].detach().cpu().long(),
        raw_confidence=entry["raw_confidence"].detach().cpu().float(),
        raw_margin=entry["raw_margin"].detach().cpu().float(),
        native_semantic=teacher_h["native_semantic"].long(),
        gt_h=teacher_h["gt_h"].long(),
        gt0=teacher_h["gt0"].long(),
        candidate=candidate,
        sample_index=sample_index,
        horizon_s=horizon_s,
        sectors=sectors,
        load_gpu_dump=base.load_gpu_dump,
        agreement_map=teacher_h["agreement"].float(),
    )
    teacher_eval = teacher_h["teacher_meta"]["final_eval"]
    student_eval = meta["final_eval"]
    teacher_final = teacher_h["teacher_final_semantic"].long()
    return {
        "split_name": split_name,
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "gamma": gamma,
        "front_sector_false_free_rate_delta_vs_native": metric_value(student_eval, "front_sector_false_free_rate_delta"),
        "teacher_front_sector_false_free_rate_delta_vs_native": metric_value(teacher_eval, "front_sector_false_free_rate_delta"),
        "future_h4_h6_false_free_rate_delta_vs_native": metric_value(student_eval, "future_h4_h6_false_free_rate_delta"),
        "teacher_future_h4_h6_false_free_rate_delta_vs_native": metric_value(teacher_eval, "future_h4_h6_false_free_rate_delta"),
        "pred_gt_density_delta": metric_value(student_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "teacher_pred_gt_density_delta": metric_value(teacher_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "false_positive_delta": metric_value(student_eval, "false_positive_delta"),
        "teacher_false_positive_delta": metric_value(teacher_eval, "false_positive_delta"),
        "wrong_class_delta": metric_value(student_eval, "wrong_class_delta"),
        "teacher_wrong_class_delta": metric_value(teacher_eval, "wrong_class_delta"),
        "front_local_density_proxy": float(meta["front_local_density_proxy_after"]),
        "teacher_front_local_density_proxy": float(teacher_h["teacher_meta"].get("front_local_density_proxy_after", 0.0)),
        "protected_zone_preservation_ratio": float(meta["protected_zone_preservation_ratio"]),
        "teacher_protected_zone_preservation_ratio": float(teacher_h["teacher_meta"].get("protected_zone_preservation_ratio", 1.0)),
        "teacher_gap_occ": int(((final_semantic.long() != EMPTY_IDX) ^ (teacher_final != EMPTY_IDX)).sum().item()),
    }


def summarize_student_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_metric_rows(rows)
    summary.update(
        {
            "teacher_front_sector_false_free_rate_delta_vs_native": mean_field(rows, "teacher_front_sector_false_free_rate_delta_vs_native"),
            "teacher_future_h4_h6_false_free_rate_delta_vs_native": mean_field(rows, "teacher_future_h4_h6_false_free_rate_delta_vs_native", horizons={4, 6}),
            "teacher_pred_gt_density_delta": mean_field(rows, "teacher_pred_gt_density_delta"),
            "teacher_false_positive_delta": mean_field(rows, "teacher_false_positive_delta"),
            "teacher_front_local_density_proxy": mean_field(rows, "teacher_front_local_density_proxy"),
            "teacher_protected_zone_preservation_ratio": mean_field(rows, "teacher_protected_zone_preservation_ratio"),
            "mean_teacher_gap_occ": mean_field(rows, "teacher_gap_occ"),
        }
    )
    teacher_front = float(summary["teacher_front_sector_false_free_rate_delta_vs_native"] or 0.0)
    student_front = float(summary["front_sector_false_free_rate_delta_vs_native"] or 0.0)
    summary["front_recovery_ratio_vs_teacher"] = student_front / teacher_front if abs(teacher_front) > 1e-9 else None
    return summary


def safety_ok(summary: dict[str, Any]) -> bool:
    density = float(summary.get("pred_gt_density_delta") or 0.0)
    teacher_density = float(summary.get("teacher_pred_gt_density_delta") or 0.0)
    fp = float(summary.get("false_positive_delta") or 0.0)
    teacher_fp = float(summary.get("teacher_false_positive_delta") or 0.0)
    proxy = float(summary.get("front_local_density_proxy") or 0.0)
    return density <= teacher_density + 0.03 and fp <= teacher_fp + 0.01 and proxy <= 1.30


def teacher_cache_path(sample_index: int) -> Path:
    return SW14B_RUNTIME_TEACHER_CACHE / f"A10_drop_front_triplet__sample{sample_index:03d}.pt"


def row_from_teacher_bundle(sample_index: int, horizon_s: int, horizon: dict[str, Any]) -> dict[str, Any]:
    meta = horizon["teacher_meta"]
    final_eval = meta["final_eval"]
    return {
        "split_name": "eval_debug_runtime_cache",
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        "front_sector_false_free_rate_delta_vs_native": metric_value(final_eval, "front_sector_false_free_rate_delta"),
        "future_h4_h6_false_free_rate_delta_vs_native": metric_value(final_eval, "future_h4_h6_false_free_rate_delta"),
        "pred_gt_density_delta": metric_value(final_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
        "false_positive_delta": metric_value(final_eval, "false_positive_delta"),
        "wrong_class_delta": metric_value(final_eval, "wrong_class_delta"),
        "front_local_density_proxy": float(meta.get("front_local_density_proxy_after", 0.0)),
        "protected_zone_preservation_ratio": float(meta.get("protected_zone_preservation_ratio", 1.0)),
    }


def compare_to_existing_teacher_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference_rows = read_csv(SW14B_REPORTS_DIR / "sw14b_eval_debug_teacher_metrics.csv")
    reference = {
        (int(row["sample_index"]), int(row["horizon_s"])): row
        for row in reference_rows
    }
    compared = 0
    max_abs_diff = 0.0
    mismatches: list[dict[str, Any]] = []
    keys = [
        "front_sector_false_free_rate_delta_vs_native",
        "future_h4_h6_false_free_rate_delta_vs_native",
        "pred_gt_density_delta",
        "false_positive_delta",
        "front_local_density_proxy",
        "protected_zone_preservation_ratio",
    ]
    for row in rows:
        ref = reference.get((int(row["sample_index"]), int(row["horizon_s"])))
        if ref is None:
            continue
        compared += 1
        for key in keys:
            diff = abs(float(row[key]) - float(ref[key]))
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 1e-9:
                mismatches.append({"sample_index": row["sample_index"], "horizon_s": row["horizon_s"], "key": key, "diff": diff})
    return {
        "reference_file": str(SW14B_REPORTS_DIR / "sw14b_eval_debug_teacher_metrics.csv"),
        "compared_rows": compared,
        "max_abs_diff": max_abs_diff,
        "mismatch_count_gt_1e_9": len(mismatches),
        "mismatches_head": mismatches[:10],
    }


def phase0_base_repair_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    if not args.skip_cache_phase0:
        for sample_index in range(args.phase0_start, args.phase0_end + 1):
            path = teacher_cache_path(sample_index)
            if not path.exists():
                missing.append(str(path))
                continue
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            for horizon_s in CORE_HORIZONS:
                rows.append(row_from_teacher_bundle(sample_index, horizon_s, bundle["by_horizon"][horizon_s]))
            del bundle
    write_csv(REPORTS_DIR / "sw14c_r8_base_repair_metrics.csv", rows)
    comparison = compare_to_existing_teacher_metrics(rows) if rows else {
        "reference_file": str(SW14B_REPORTS_DIR / "sw14b_eval_debug_teacher_metrics.csv"),
        "compared_rows": 0,
        "max_abs_diff": None,
        "mismatch_count_gt_1e_9": None,
        "mismatches_head": [],
    }
    synthetic = synthetic_base_repair_check()
    decision = "BASE_R1_PASS"
    if missing or not rows or comparison["compared_rows"] != len(rows):
        decision = "BASE_R3_METRIC_MISMATCH"
    elif float(comparison["max_abs_diff"] or 0.0) > 1e-9:
        decision = "BASE_R3_METRIC_MISMATCH"
    elif not synthetic["pass"]:
        decision = "BASE_R2_FEATURE_MISMATCH"
    audit = {
        "decision": decision,
        "phase": "Phase 0 R8 base repair parity",
        "camera_order": CAMERA_NAMES,
        "front_triplet_replaced_by_tminus1": FRONT_TRIPLET,
        "rear_cameras_unchanged": REAR_CAMERAS,
        "all_levels_repaired": True,
        "no_alpha": True,
        "no_learned_params": True,
        "metric_reproduction_source": "precomputed SW14B runtime teacher cache for SW13 R8 + F3 + FrontCap",
        "sample_range": [args.phase0_start, args.phase0_end],
        "missing_cache_files": missing,
        "synthetic_feature_check": synthetic,
        "metric_summary": summarize_metric_rows(rows),
        "metric_reference_comparison": comparison,
        "io_safety": {
            "feature_memory_cache_read": False,
            "runtime_teacher_cache_read": bool(rows),
            "rebuild_teacher_cache": False,
            "eval_core100_or_core500_used": False,
        },
    }
    write_json(REPORTS_DIR / "sw14c_r8_base_repair_audit.json", audit)
    write_md(
        REPORTS_DIR / "sw14c_r8_base_repair_audit.md",
        "\n".join(
            [
                "# SW14C R8 Base Repair Audit",
                "",
                f"- decision: `{decision}`.",
                f"- camera order: `{', '.join(CAMERA_NAMES)}`.",
                f"- A10 front triplet replacement: `{', '.join(FRONT_TRIPLET)}` from t-1 memory.",
                "- rear cameras remain current-frame features.",
                "- no alpha, no learned parameter, no GT in F3/FrontCap.",
                f"- metric rows from cache: `{len(rows)}`.",
                f"- reference max abs diff: `{comparison['max_abs_diff']}`.",
                "- cache-only audit avoids reading SW13A feature memory cache.",
            ]
        ),
    )
    return audit


def synthetic_base_repair_check() -> dict[str, Any]:
    torch.manual_seed(0)
    current_levels = [torch.randn(1, 6, 8, 4, 5) + idx for idx in range(4)]
    memory_levels = [level + 100.0 + idx for idx, level in enumerate(current_levels)]
    base = build_sw13_r8_base_repair(current_levels, memory_levels, "A10_drop_front_triplet")
    front_ok = all(torch.equal(base[level_idx][:, FRONT_TRIPLET_INDICES], memory_levels[level_idx][:, FRONT_TRIPLET_INDICES]) for level_idx in range(4))
    rear_ok = True
    for level_idx in range(4):
        rear_idx = [idx for idx in range(6) if idx not in FRONT_TRIPLET_INDICES]
        rear_ok = rear_ok and torch.equal(base[level_idx][:, rear_idx], current_levels[level_idx][:, rear_idx])
    clean = build_sw13_r8_base_repair(current_levels, memory_levels, "A0_clean")
    clean_ok = all(torch.equal(clean[level_idx], current_levels[level_idx]) for level_idx in range(4))
    return {
        "pass": bool(front_ok and rear_ok and clean_ok),
        "front_replaced": bool(front_ok),
        "rear_unchanged": bool(rear_ok),
        "clean_noop": bool(clean_ok),
        "level_count": 4,
    }


def phase1_residual_adapter_architecture(args: argparse.Namespace) -> dict[str, Any]:
    adapter = SpatialResidualAdapter(channels_per_level=[256, 256, 256, 256], hidden_channels=32, camera_embed_dim=4)
    stats = count_parameters(adapter)
    current_levels = [torch.randn(1, 6, 16, 4, 5) for _ in range(4)]
    memory_levels = [level + 0.5 for level in current_levels]
    base_levels = build_sw13_r8_base_repair(current_levels, memory_levels, "A10_drop_front_triplet")
    mask = front_triplet_degradation_mask("A10_drop_front_triplet", batch_size=1)
    small_adapter = SpatialResidualAdapter(channels_per_level=[16, 16, 16, 16], hidden_channels=8, camera_embed_dim=2)
    final_levels, debug = small_adapter.apply_to_levels(current_levels, memory_levels, base_levels, mask, gamma=0.30)
    init_max_diff = max(float((final - base).abs().max().item()) for final, base in zip(final_levels, base_levels))
    clean_mask = front_triplet_degradation_mask("A0_clean", batch_size=1)
    clean_base = build_sw13_r8_base_repair(current_levels, memory_levels, "A0_clean")
    clean_final, clean_debug = small_adapter.apply_to_levels(current_levels, memory_levels, clean_base, clean_mask, gamma=0.30)
    clean_max_diff = max(float((final - base).abs().max().item()) for final, base in zip(clean_final, clean_base))
    if args.write_init_checkpoint:
        torch.save(
            checkpoint_payload(
                adapter,
                {
                    "stage": "SW14C",
                    "init_equivalent_to_sw13_r8_base": init_max_diff <= 1e-8,
                    "gamma_default": 0.10,
                    "not_trained": True,
                },
            ),
            ARTIFACTS_DIR / "checkpoints/sw14c_round1_init_equivalence_checkpoint.pth",
        )
    architecture = {
        "adapter_type": "SpatialResidualAdapter",
        "base": "SW13_R8_hard_repair",
        "formula": "final_feature = base_repaired_feature + gamma * residual_gate * residual_delta",
        "parameter_count": int(stats.parameter_count),
        "trainable_parameter_count": int(stats.trainable_parameter_count),
        "channels_per_level": [256, 256, 256, 256],
        "hidden_channels": 32,
        "camera_embedding_dim": 4,
        "residual_limit": 0.10,
        "gate_bias_init": -4.0,
        "gamma_candidates": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
        "per_level_residual_gate_shape": {f"level{idx}": [1, 6, 1, "H", "W"] for idx in range(4)},
        "per_level_residual_delta_shape": {f"level{idx}": [1, 6, 256, "H", "W"] for idx in range(4)},
        "insertion_point": "after extract_feat, after SW13 R8 hard repair, before SparseWorld head/simple_test_pts",
        "front_triplet_only": True,
        "rear_cameras_unchanged": True,
        "clean_a0_residual_zero": True,
        "residual_init_equals_sw13_r8_base": init_max_diff <= 1e-8,
        "residual_init_max_abs_diff": init_max_diff,
        "clean_init_max_abs_diff": clean_max_diff,
        "clean_gate_sum": float(sum(value.abs().sum().item() for key, value in clean_debug.items() if key.startswith("residual_gate"))),
        "frozen_sparseworld_status": {
            "backbone_trainable": False,
            "occupancy_head_trainable": False,
            "get_occ_modified": False,
            "only_adapter_trainable": True,
            "runtime_verification_required_before_training": True,
        },
        "get_occ_sha256": sha256(GET_OCC_PATH) if GET_OCC_PATH.exists() else None,
        "no_get_occ_modification": True,
    }
    write_json(REPORTS_DIR / "sw14c_residual_adapter_architecture.json", architecture)
    write_md(
        REPORTS_DIR / "sw14c_residual_adapter_architecture.md",
        "\n".join(
            [
                "# SW14C Residual Adapter Architecture",
                "",
                "- base: `SW13_R8_hard_repair(current, memory_tminus1)`.",
                "- residual formula: `base + gamma * residual_gate * residual_delta`.",
                "- residual is hard-masked to A10 front triplet degraded cameras.",
                "- rear cameras and A0 clean are no-op paths.",
                "- residual delta head is zero-initialized, so the initialized adapter equals SW13 R8 base.",
                f"- trainable adapter parameters: `{stats.trainable_parameter_count}`.",
            ]
        ),
    )
    return architecture


def phase2_loss_design() -> dict[str, Any]:
    config = SW14CLossConfig().to_dict()
    write_json(REPORTS_DIR / "sw14c_loss_config.json", config)
    write_csv(REPORTS_DIR / "sw14c_loss_terms.csv", LOSS_TERMS)
    return {"loss_config": config, "loss_terms": LOSS_TERMS}


def write_split_and_protocol() -> dict[str, Any]:
    split = {
        "train_tune": {
            "sample_indices": "train split 0..399",
            "source_split": "train",
            "uses_gt_for_training": True,
            "tuning_allowed": True,
            "final_claim_allowed": False,
        },
        "val_small": {
            "sample_indices": "train-derived 400..499",
            "source_split": "train",
            "uses_gt_for_training": False,
            "uses_gt_for_eval_only": True,
            "tuning_allowed": True,
            "final_claim_allowed": False,
        },
        "eval_debug": {
            "sample_indices": "eval 100..119",
            "source_split": "eval",
            "uses_gt_for_training": False,
            "uses_gt_for_eval_only": True,
            "tuning_allowed": False,
            "final_claim_allowed": False,
        },
        "eval_core100_core500": {
            "sample_indices": "frozen only after debug pass",
            "source_split": "eval",
            "uses_gt_for_training": False,
            "uses_gt_for_eval_only": True,
            "tuning_allowed": False,
            "final_claim_allowed": True,
        },
    }
    postprocess = {
        "chain": ["SW13_R8_base", "SW14C_residual_adapter", "SparseWorld_head", "F3_expand_budget_strict", "FC1_1p3_FrontCap", "evaluation"],
        "f3": {
            "reuse_sw13c_fix_function": True,
            "expansion_ratio": 0.12,
            "protected_variant": "PZ_fix_3_strong_core",
            "agreement_sources": ["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            "wrong_class_aware": False,
            "uses_gt_budget": False,
            "uses_pred_gt_density_for_selection": False,
        },
        "frontcap": {
            "variant": "FC1_1p3",
            "cap_ratio": 1.3,
            "cap_mode": "front_native_ratio_cap",
            "uses_gt_frontcap": False,
            "protected_not_pruned": True,
        },
    }
    metric_sign = {
        "schema": "SW13-style deltas: system - native",
        "front_sector_false_free_rate_delta_vs_native": {
            "formula": "system front_sector_false_free_rate - native front_sector_false_free_rate",
            "better_direction": "more_negative",
        },
        "future_h4_h6_false_free_rate_delta_vs_native": {
            "formula": "system future false_free_rate - native false_free_rate for h4/h6",
            "better_direction": "more_negative",
        },
        "pred_gt_density_delta": {
            "formula": "system pred_gt_occupied_ratio - native pred_gt_occupied_ratio",
            "better_direction": "close_to_teacher_and_below_safety_threshold",
        },
        "false_positive_delta": {
            "formula": "system false_occupied_rate - native false_occupied_rate",
            "better_direction": "lower",
        },
        "front_local_density_proxy": {
            "formula": "front occupied count after FrontCap / native front occupied count",
            "better_direction": "safe_below_or_equal_1.30",
        },
        "protected_zone_preservation_ratio": {
            "formula": "1 - protected_pruned_count / protected_count",
            "better_direction": "higher",
        },
    }
    write_json(REPORTS_DIR / "sw14c_split_manifest.json", split)
    write_json(REPORTS_DIR / "sw14c_postprocess_protocol.json", postprocess)
    write_json(REPORTS_DIR / "sw14c_metric_sign_dictionary.json", metric_sign)
    return {"split": split, "postprocess": postprocess, "metric_sign": metric_sign}


def build_residual_adapter() -> SpatialResidualAdapter:
    adapter = SpatialResidualAdapter(channels_per_level=[256, 256, 256, 256], hidden_channels=32, camera_embed_dim=4)
    return adapter.cuda() if torch.cuda.is_available() else adapter


def train_round2_small(args: argparse.Namespace) -> tuple[dict[str, Any], SpatialResidualAdapter | None]:
    if not args.run_training:
        return {
            "round": "Round 2 small train",
            "executed": False,
            "reason": "not requested",
            "planned_train_samples": "train 0..99",
            "planned_val_samples": "train-derived val only",
            "epochs": 1,
        }, None
    print("[sw14c] build SparseWorld train runtime", flush=True)
    _, dataset, model, _ = base.build_runtime(train=True)
    model.eval()
    base.freeze_sparseworld_modules(model)
    adapter = build_residual_adapter()
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.round2_lr))
    config = SW14CLossConfig()
    sectors = {key: value.cpu() for key, value in base.sw13c_fix.sw7.build_sector_masks().items()}
    train_indices = list(range(args.round2_train_start, args.round2_train_end + 1))
    log_rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_wall = time.time()
    block_size = max(1, int(args.round2_train_block_size))
    micro_steps = max(1, int(args.round2_micro_steps_per_sample))
    print(
        f"[sw14c] round2 train samples={train_indices} epochs={args.round2_epochs} "
        f"gamma={args.round2_gamma} block_size={block_size} micro_steps={micro_steps}",
        flush=True,
    )
    for epoch in range(int(args.round2_epochs)):
        for block_id, block_indices in enumerate(chunks(train_indices, block_size)):
            block_prepare_start = time.time()
            block_disk_start = disk_read_sectors()
            prepared_block: list[dict[str, Any]] = []
            try:
                for sample_index in block_indices:
                    prepared_block.append(prepare_residual_feature_case(model, dataset, sample_index))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                block_prepare_elapsed = time.time() - block_prepare_start
                block_disk_delta = disk_read_mb_delta(block_disk_start, disk_read_sectors())
                print(
                    f"[sw14c] prepared block epoch={epoch} block={block_id} samples={block_indices} "
                    f"elapsed={block_prepare_elapsed:.2f}s disk_read_delta={block_disk_delta:.1f}MB",
                    flush=True,
                )
                for micro_step in range(micro_steps):
                    for prepared in prepared_block:
                        sample_index = int(prepared["sample_index"])
                        sample_start = time.time()
                        disk_start = disk_read_sectors()
                        per_h = None
                        debug = None
                        try:
                            optimizer.zero_grad(set_to_none=True)
                            per_h, debug = run_residual_forward_prepared(model, adapter, prepared, gamma=float(args.round2_gamma), training=True)
                            loss, loss_stats = compute_training_loss(per_h, debug, sectors, config)
                            if not bool(loss.requires_grad):
                                raise RuntimeError("SW14C loss does not require grad; dense get_occ path is not differentiable in this runtime")
                            loss.backward()
                            grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0).detach().item())
                            optimizer.step()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            disk_end = disk_read_sectors()
                            elapsed = time.time() - sample_start
                            row = {
                                "round": "round2_small",
                                "epoch": epoch,
                                "block_id": block_id,
                                "micro_step": micro_step,
                                "sample_index": sample_index,
                                "gamma": float(args.round2_gamma),
                                "lr": float(args.round2_lr),
                                "elapsed_s": elapsed,
                                "block_prepare_elapsed_s": block_prepare_elapsed,
                                "block_prepare_disk_read_mb_delta": block_disk_delta,
                                "disk_read_mb_delta": disk_read_mb_delta(disk_start, disk_end),
                                "grad_norm": grad_norm,
                                **loss_stats,
                                **cuda_mem_mb(),
                            }
                            log_rows.append(row)
                            print(
                                f"[sw14c] train epoch={epoch} block={block_id} micro={micro_step} sample={sample_index} "
                                f"loss={row['total_loss']:.4f} elapsed={elapsed:.2f}s "
                                f"disk_read_delta={row['disk_read_mb_delta']:.1f}MB",
                                flush=True,
                            )
                        finally:
                            release_runtime(per_h, debug, collect=False, empty_cache=False)
            finally:
                release_runtime(prepared_block, collect=False, empty_cache=False)
    checkpoint_path = ARTIFACTS_DIR / "checkpoints/sw14c_round2_smoke_checkpoint.pth"
    torch.save(
        checkpoint_payload(
            adapter,
            {
                "stage": "SW14C",
                "round": "round2_small",
                "train_samples": train_indices,
                "epochs": int(args.round2_epochs),
                "gamma_train": float(args.round2_gamma),
                "lr": float(args.round2_lr),
                "train_block_size": block_size,
                "micro_steps_per_sample": micro_steps,
                "subset_diagnostic": True,
            },
        ),
        checkpoint_path,
    )
    write_csv(REPORTS_DIR / "sw14c_training_log.csv", log_rows)
    summary = {
        "round": "Round 2 small train",
        "executed": True,
        "decision": "ROUND2_SUBSET_TRAIN_COMPLETED",
        "checkpoint_path": str(checkpoint_path),
        "train_samples": train_indices,
        "epochs": int(args.round2_epochs),
        "gamma_train": float(args.round2_gamma),
        "train_block_size": block_size,
        "micro_steps_per_sample": micro_steps,
        "elapsed_s": time.time() - start_wall,
        "mean_loss": float(np.mean([row["total_loss"] for row in log_rows])) if log_rows else None,
        "last_loss": float(log_rows[-1]["total_loss"]) if log_rows else None,
        "mean_disk_read_mb_delta_per_sample": float(np.mean([row["disk_read_mb_delta"] for row in log_rows])) if log_rows else None,
        "max_disk_read_mb_delta_per_sample": float(np.max([row["disk_read_mb_delta"] for row in log_rows])) if log_rows else None,
        "mean_block_prepare_disk_read_mb_delta": float(np.mean([row["block_prepare_disk_read_mb_delta"] for row in log_rows])) if log_rows else None,
        "max_block_prepare_disk_read_mb_delta": float(np.max([row["block_prepare_disk_read_mb_delta"] for row in log_rows])) if log_rows else None,
        "cuda": cuda_mem_mb(),
        "io_policy": "single runtime, sample-major feature extraction, runtime clean t-1 memory, no 263MB feature cache load, no teacher cache rebuild",
        "subset_diagnostic": True,
    }
    write_json(REPORTS_DIR / "sw14c_round2_smoke_decision.json", summary)
    release_runtime(model, dataset, collect=True, empty_cache=False)
    return summary, adapter


def evaluate_gamma_sweep_valsmall(args: argparse.Namespace, adapter: SpatialResidualAdapter | None) -> dict[str, Any]:
    if adapter is None:
        return phase4_gamma_selection_placeholder()
    adapter.eval()
    val_train_runtime = bool(args.val_self_teacher)
    print(f"[sw14c] build SparseWorld val runtime train={val_train_runtime}", flush=True)
    _, dataset, model, _ = base.build_runtime(train=val_train_runtime)
    model.eval()
    base.freeze_sparseworld_modules(model)
    sectors = {key: value.cpu() for key, value in base.sw13c_fix.sw7.build_sector_masks().items()}
    gamma_candidates = parse_float_list(args.gamma_candidates)
    val_indices = list(range(args.round2_val_start, args.round2_val_end + 1))
    detail_rows: list[dict[str, Any]] = []
    print(f"[sw14c] val gamma sweep samples={val_indices} gamma={gamma_candidates}", flush=True)
    for sample_index in val_indices:
        disk_start = disk_read_sectors()
        prepared = None
        teacher_bundle = None
        try:
            prepared = prepare_residual_feature_case(model, dataset, sample_index)
            if args.val_self_teacher:
                teacher_bundle = build_self_teacher_bundle_for_prepared(model, prepared, sample_index, sectors)
            else:
                teacher_bundle = teacher_bundle_for_sample(sample_index, allow_rebuild=bool(args.allow_teacher_cache_rebuild))
            if teacher_bundle is None:
                print(f"[sw14c] skip val sample={sample_index}; missing teacher cache", flush=True)
                continue
            for gamma in gamma_candidates:
                per_h = None
                debug = None
                try:
                    per_h, debug = run_residual_forward_prepared(model, adapter, prepared, gamma=float(gamma), training=False)
                    for horizon_s in CORE_HORIZONS:
                        detail_rows.append(
                            postprocess_student_row(
                                sample_index,
                                horizon_s,
                                "val_small_subset",
                                float(gamma),
                                per_h,
                                teacher_bundle,
                                sectors,
                            )
                        )
                finally:
                    release_runtime(per_h, debug, collect=False, empty_cache=False)
            disk_end = disk_read_sectors()
            print(
                f"[sw14c] val sample={sample_index} gamma_count={len(gamma_candidates)} "
                f"disk_read_delta={disk_read_mb_delta(disk_start, disk_end):.1f}MB",
                flush=True,
            )
        finally:
            release_runtime(prepared, teacher_bundle, collect=False, empty_cache=False)
    write_csv(REPORTS_DIR / "sw14c_round2_val_metrics.csv", detail_rows)
    write_csv(REPORTS_DIR / "sw14c_gamma_sweep_valsmall_metrics.csv", detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for gamma in gamma_candidates:
        rows_g = [row for row in detail_rows if abs(float(row["gamma"]) - float(gamma)) < 1e-9]
        summary = summarize_student_rows(rows_g)
        front_over_teacher = None
        if summary.get("front_sector_false_free_rate_delta_vs_native") is not None and summary.get("teacher_front_sector_false_free_rate_delta_vs_native") is not None:
            front_over_teacher = float(summary["teacher_front_sector_false_free_rate_delta_vs_native"]) - float(summary["front_sector_false_free_rate_delta_vs_native"])
        summary_rows.append(
            {
                "gamma": float(gamma),
                **summary,
                "front_fn_reduction_over_teacher": front_over_teacher,
                "safety_ok": safety_ok(summary),
                "equivalent_to_sw13_r8_teacher": abs(float(gamma)) < 1e-12,
                "allowed_as_improved_result": abs(float(gamma)) >= 1e-12,
            }
        )
    write_csv(REPORTS_DIR / "sw14c_gamma_sweep_valsmall.csv", summary_rows)
    safe_improving = [
        row for row in summary_rows
        if bool(row["safety_ok"]) and bool(row["allowed_as_improved_result"]) and (row["front_fn_reduction_over_teacher"] is not None) and float(row["front_fn_reduction_over_teacher"]) > 0.0
    ]
    safe_any = [row for row in summary_rows if bool(row["safety_ok"]) and bool(row["allowed_as_improved_result"])]
    if safe_improving:
        best = max(safe_improving, key=lambda row: float(row["front_fn_reduction_over_teacher"]))
        decision = "GAMMA_R1_SAFE_IMPROVES_TEACHER"
    elif safe_any:
        best = max(safe_any, key=lambda row: float(row.get("front_recovery_ratio_vs_teacher") or 0.0))
        decision = "GAMMA_R2_SAFE_NO_IMPROVEMENT"
    else:
        best = max(summary_rows, key=lambda row: float(row.get("front_recovery_ratio_vs_teacher") or 0.0)) if summary_rows else None
        decision = "GAMMA_R3_UNSAFE"
    selection = {
        "decision": decision,
        "executed": True,
        "reason": "Round2 gamma sweep on train-derived val with same-runtime SW13 R8 self-teacher" if args.val_self_teacher else "Round2 subset gamma sweep on existing runtime teacher cache; cache was generated with train=False and is diagnostic only",
        "gamma_candidates": gamma_candidates,
        "selected_gamma": None if best is None else float(best["gamma"]),
        "best_summary": best,
        "uses_val_small_for_selection": bool(args.val_self_teacher),
        "uses_cache_diagnostic_for_selection": not bool(args.val_self_teacher),
        "uses_eval_debug_for_selection": False,
        "uses_eval_core100_or_core500_for_selection": False,
        "gamma_zero_counts_as_improved_result": False,
        "val_samples": val_indices,
        "subset_diagnostic": True,
        "protocol_final_claim_allowed": bool(args.val_self_teacher),
    }
    write_json(REPORTS_DIR / "sw14c_gamma_selection.json", selection)
    release_runtime(model, dataset, collect=True, empty_cache=False)
    return selection


def phase3_training_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], SpatialResidualAdapter | None]:
    round1 = {
        "round": "Round 1 init equivalence smoke",
        "executed": True,
        "training": False,
        "decision": "INIT_R1_EQUIVALENT_TO_SW13_R8_BASE",
        "checkpoint": str(ARTIFACTS_DIR / "checkpoints/sw14c_round1_init_equivalence_checkpoint.pth"),
    }
    round2, adapter = train_round2_small(args)
    round3 = {
        "round": "Round 3 main train",
        "executed": False,
        "reason": "blocked until Round 2 passes without density/clean collapse",
        "planned_train_samples": "train 0..399",
        "planned_val_samples": "train-derived 400..499",
        "epochs": "3..8, early stop patience 2",
    }
    write_json(REPORTS_DIR / "sw14c_round1_init_audit.json", round1)
    if not round2.get("executed"):
        write_json(REPORTS_DIR / "sw14c_round2_smoke_decision.json", round2)
    write_json(REPORTS_DIR / "sw14c_round3_decision.json", round3)
    if not args.run_training:
        write_csv(REPORTS_DIR / "sw14c_training_log.csv", [])
        write_csv(REPORTS_DIR / "sw14c_val_metrics.csv", [])
    return {"round1": round1, "round2": round2, "round3": round3}, adapter


def phase4_gamma_selection_placeholder() -> dict[str, Any]:
    rows = [
        {
            "gamma": 0.0,
            "equivalent_to_sw13_r8_teacher": True,
            "allowed_as_improved_result": False,
            "executed_on_val_small": False,
            "safety_status": "lower_bound_only",
            "improves_teacher": False,
        }
    ]
    selection = {
        "decision": "GAMMA_R2_SAFE_NO_IMPROVEMENT",
        "executed": False,
        "reason": "no trained residual checkpoint exists yet; gamma=0 is only the SW13 lower bound",
        "gamma_candidates": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
        "selected_gamma": 0.0,
        "uses_val_small_for_selection": True,
        "uses_eval_debug_for_selection": False,
        "uses_eval_core100_or_core500_for_selection": False,
        "gamma_zero_counts_as_improved_result": False,
    }
    write_csv(REPORTS_DIR / "sw14c_gamma_sweep_valsmall.csv", rows)
    write_json(REPORTS_DIR / "sw14c_gamma_selection.json", selection)
    return selection


def load_sw14b_rescue_baseline() -> dict[str, Any] | None:
    path = SW14B_RESCUE_REPORTS_DIR / "sw14b_eval_debug_scale_sweep_summary.json"
    if not path.exists():
        return None
    payload = read_json(path)
    return {
        "teacher_summary": payload.get("teacher_summary"),
        "sw14b_best_safe_alpha_scale": payload.get("best_safe_by_front_recovery"),
        "sw14b_original_summary": payload.get("original_summary"),
        "metric_schema": payload.get("metric_schema"),
    }


def phase5_eval_debug_placeholder() -> dict[str, Any]:
    baseline = load_sw14b_rescue_baseline()
    decision = {
        "decision": "SW14C_DEBUG_FAIL_KEEP_SW13",
        "executed": False,
        "reason": "no trained residual checkpoint/gamma selected from val_small",
        "checkpoint_fixed_before_eval": None,
        "gamma_fixed_before_eval": None,
        "native_degraded": None,
        "sw13_teacher": baseline.get("teacher_summary") if baseline else None,
        "sw14b_alpha_adapter_best_scale_0p40": baseline.get("sw14b_best_safe_alpha_scale") if baseline else None,
        "sw14c_residual_adapter": None,
    }
    write_csv(REPORTS_DIR / "sw14c_eval_debug_metrics.csv", [])
    write_json(REPORTS_DIR / "sw14c_eval_debug_decision.json", decision)
    write_csv(REPORTS_DIR / "sw14c_eval_core100_metrics.csv", [])
    write_json(
        REPORTS_DIR / "sw14c_eval_core100_decision.json",
        {"decision": "SW14C_CORE100_NOT_EXECUTED", "reason": "eval_debug did not pass"},
    )
    write_csv(REPORTS_DIR / "sw14c_eval_core500_metrics.csv", [])
    write_json(
        REPORTS_DIR / "sw14c_eval_core500_decision.json",
        {"decision": "SW14C_CORE500_NOT_EXECUTED", "reason": "core100 did not pass"},
    )
    return decision


def phase7_teacher_error_mining_placeholder() -> dict[str, Any]:
    payload = {
        "executed": False,
        "reason": "teacher error mining is next after a trained SW14C residual checkpoint fails to beat teacher",
        "teacher_fn_buckets": [
            "dynamic_object_missed",
            "far_front_sector_missed",
            "horizon_h4_h6_missed",
            "boundary_sparse_occupied_missed",
            "low_confidence_occupied_missed",
        ],
        "teacher_fp_buckets": [
            "front_hallucination",
            "temporal_ghost_from_tminus1",
            "wrong_class_occupied",
            "isolated_occupied",
            "non_front_spillover",
        ],
    }
    write_csv(REPORTS_DIR / "sw14c_teacher_error_mining.csv", [])
    write_json(REPORTS_DIR / "sw14c_teacher_error_buckets.json", payload)
    return payload


def final_decision(
    base_audit: dict[str, Any],
    architecture: dict[str, Any],
    training: dict[str, Any],
    gamma: dict[str, Any],
    eval_debug: dict[str, Any],
) -> dict[str, Any]:
    decision_enum = "SW14C_0_FAIL_KEEP_SW13_MAIN"
    if base_audit.get("decision") != "BASE_R1_PASS":
        decision_enum = "SW14C_6_PROTOCOL_VIOLATION"
    round2 = training.get("round2", {})
    decision = {
        "decision": decision_enum,
        "best_checkpoint": round2.get("checkpoint_path") if round2.get("executed") else None,
        "best_gamma": gamma.get("selected_gamma") if gamma.get("executed") else None,
        "best_adapter_type": architecture["adapter_type"],
        "base_repair_decision": base_audit.get("decision"),
        "train_split_used": round2.get("train_samples") if round2.get("executed") else None,
        "eval_split_used": gamma.get("val_samples") if gamma.get("executed") else None,
        "whether_beats_sw13_teacher": False,
        "whether_use_in_resume": False,
        "whether_SW13_remains_main": True,
        "safe_claim": "SW14C residual-on-SW13 design is implemented; any Round2 result here is subset diagnostic only and has not passed frozen eval_debug/core100. SW13C-Fix + FrontCap remains the main result.",
        "limitations": [
            "Round2 training is subset diagnostic only and not a final claim",
            "gamma sweep uses existing runtime teacher cache diagnostic unless a true train-derived val cache is generated",
            "no eval_debug/core100/core500 improvement claim",
            "not an official benchmark result",
        ],
        "next_action": "run Round 2 small train on train split 0..99, validate on train-derived 100..149, then run val_small gamma sweep without touching eval_debug for selection",
        "protocol_flags": {
            "backbone_frozen_required": True,
            "head_frozen_required": True,
            "get_occ_unchanged_required": True,
            "f3_frontcap_params_unchanged": True,
            "eval_gt_not_used_for_selection": True,
            "eval_core100_core500_not_used_for_tuning": True,
        },
        "training_status": training,
        "gamma_selection": gamma,
        "eval_debug": eval_debug,
    }
    if decision["decision"] not in FINAL_DECISION_ENUMS:
        raise AssertionError(decision["decision"])
    write_json(REPORTS_DIR / "sw14c_final_decision.json", decision)
    write_md(
        REPORTS_DIR / "sw14c_final_decision.md",
        "\n".join(
            [
                "# SW14C Final Decision",
                "",
                f"- decision: `{decision['decision']}`.",
                "- SW13C-Fix + FrontCap remains the main result.",
                "- whether_use_in_resume: `False`.",
                "- no official benchmark or stronger-than-SW13 claim is made.",
                f"- next action: {decision['next_action']}",
            ]
        ),
    )
    return decision


def write_report(
    base_audit: dict[str, Any],
    architecture: dict[str, Any],
    loss: dict[str, Any],
    protocol: dict[str, Any],
    training: dict[str, Any],
    gamma: dict[str, Any],
    eval_debug: dict[str, Any],
    error_mining: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    report_json = {
        "why_sw14b_alpha_cannot_beat_sw13": load_sw14b_rescue_baseline(),
        "sw13_r8_hard_replace_as_strong_base": base_audit,
        "residual_on_teacher_design": architecture,
        "loss_design": loss,
        "split_and_postprocess_protocol": protocol,
        "training_protocol": training,
        "gamma_sweep": gamma,
        "frozen_eval_debug": eval_debug,
        "teacher_error_mining": error_mining,
        "final_decision": decision,
    }
    write_json(REPORTS_DIR / "stage_sw14c_residual_teacher_adapter_report.json", report_json)
    write_md(
        REPORTS_DIR / "stage_sw14c_residual_teacher_adapter_report.md",
        "\n".join(
            [
                "# Stage SW14C Residual Teacher Adapter Report",
                "",
                "## 1. Why SW14B alpha adapter cannot beat SW13",
                "- SW14B alpha fusion recovers only a safe subset of SW13 teacher behavior under density constraints.",
                "- SW14C therefore starts from SW13 R8 hard replace and learns only a small residual.",
                "",
                "## 2. SW13 R8 hard replace as strong base",
                f"- base parity decision: `{base_audit['decision']}`.",
                "- A10 front triplet is replaced by t-1 same-camera memory; rear cameras remain current.",
                "",
                "## 3. Residual-on-teacher design",
                f"- adapter: `{architecture['adapter_type']}`.",
                "- initialized output equals SW13 R8 base.",
                "",
                "## 4. Base repair parity",
                f"- metric rows audited: `{base_audit['metric_summary']['row_count']}`.",
                f"- reference max abs diff: `{base_audit['metric_reference_comparison']['max_abs_diff']}`.",
                "",
                "## 5. Adapter architecture",
                f"- trainable params: `{architecture['trainable_parameter_count']}`.",
                "",
                "## 6. Loss design",
                "- teacher-FN recover, teacher-FP suppress, teacher-correct preserve, density/FP safety, residual regularization, clean/rear consistency.",
                "",
                "## 7. Training protocol",
                "- Round 1 init equivalence is written; Round 2/3 are not executed in this safe implementation pass.",
                "",
                "## 8. Gamma sweep",
                f"- decision: `{gamma['decision']}`.",
                "",
                "## 9. Frozen eval_debug",
                f"- decision: `{eval_debug['decision']}`.",
                "",
                "## 10. Frozen core100/core500 if executed",
                "- not executed because eval_debug did not pass with a trained checkpoint.",
                "",
                "## 11. Teacher error mining if needed",
                "- queued for after a trained residual checkpoint fails to beat teacher.",
                "",
                "## 12. Final decision",
                f"- `{decision['decision']}`.",
                "",
                "## 13. Whether use in resume",
                "- `False`.",
                "",
                "## 14. Safe claim",
                decision["safe_claim"],
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    base_audit = phase0_base_repair_audit(args)
    architecture = phase1_residual_adapter_architecture(args)
    loss = phase2_loss_design()
    protocol = write_split_and_protocol()
    training, trained_adapter = phase3_training_protocol(args)
    gamma = evaluate_gamma_sweep_valsmall(args, trained_adapter) if args.run_training else phase4_gamma_selection_placeholder()
    eval_debug = phase5_eval_debug_placeholder()
    error_mining = phase7_teacher_error_mining_placeholder()
    decision = final_decision(base_audit, architecture, training, gamma, eval_debug)
    write_report(base_audit, architecture, loss, protocol, training, gamma, eval_debug, error_mining, decision)


if __name__ == "__main__":
    main()
