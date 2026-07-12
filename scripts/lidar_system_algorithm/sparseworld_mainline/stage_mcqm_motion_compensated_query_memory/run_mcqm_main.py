from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from mmcv.parallel import collate as collate_fn


PROJECT_ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
WORKSPACE_ROOT = PROJECT_ROOT.parent
BASE_CONFIG = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_mcqm_motion_compensated_query_memory"
V2_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_mcqm_v2_interface_oracle"
GEOMETRY_ONLY_AUDIT_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_mcqm_v2_geometry_only_audit"
Q2F_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_mcqm_q2f_shallow_v3"
PROGRESS_PATH = V2_ARTIFACTS_DIR / "progress.json"
LOG_PATH = REPORTS_DIR / "mcqm_worklog.md"
CONTRACT_PATH = REPORTS_DIR / "mcqm_tensor_contract.md"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
ORACLE_E2E_PROTOCOL_VERSION = "mcqm_v2_oracle_e2e_v4_target_window_seeded_alignment_hash"
ORACLE_E2E_FORCED_SELECTION_POLICY = "positive_nonconflicting_topk_v1"
ORACLE_E2E_DECISION_K = 8
GEOMETRY_ONLY_AUDIT_VERSION = "mcqm_v2_geometry_only_audit_v1"
MCQM_Q2F_PROTOCOL_VERSION = (
    "mcqm_q2f_v3_shallow_stage_"
    "hard_replace_pre_neck"
)
SW15_DCLASS_CONF_THRESHOLD = 0.20
SW15_DCLASS_MARGIN_THRESHOLD = 0.01
S3_STRUCTURE = np.ones((3, 3, 3), dtype=bool)

HORIZON_TO_KEY = {0: "semantic_occ_0s", 1: "semantic_occ_1s", 2: "semantic_occ_2s", 3: "semantic_occ_3s", 4: "semantic_occ_4s", 5: "semantic_occ_5s", 6: "semantic_occ_6s"}
CORE_HORIZONS = [0, 2, 4, 6]
_SECTOR_MASK_CACHE: dict[str, torch.Tensor] | None = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw2 = load_module(
    "mcqm_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw5 = load_module(
    "mcqm_sw5",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation/sensor_perturbation_engine.py",
)
sw7 = load_module(
    "mcqm_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw12b = load_module(
    "mcqm_sw12b",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py",
)
sw13a = load_module(
    "mcqm_sw13a",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/run_sw13a_main.py",
)
sw81 = load_module(
    "mcqm_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)


def log(text: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_sector_masks_cached() -> dict[str, torch.Tensor]:
    global _SECTOR_MASK_CACHE
    if _SECTOR_MASK_CACHE is None:
        _SECTOR_MASK_CACHE = {
            name: tensor.cpu()
            for name, tensor in sw7.build_sector_masks().items()
        }
    return _SECTOR_MASK_CACHE


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iso_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_jsonl_text(path: Path, serialized_line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(serialized_line)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def cpu_rss_gb() -> float:
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_kb / (1024.0 * 1024.0)


def gpu_mem_stats_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        float(torch.cuda.memory_allocated()) / (1024.0 ** 3),
        float(torch.cuda.memory_reserved()) / (1024.0 ** 3),
    )


def write_progress(payload: dict[str, Any]) -> None:
    full_payload = {
        **payload,
        "updated_at": iso_now(),
        "pid": int(os.getpid()),
    }
    atomic_write_json(PROGRESS_PATH, full_payload)
    stage = str(full_payload.get("stage", ""))
    status = str(full_payload.get("status", ""))
    variant = str(full_payload.get("variant", ""))
    sample_index = full_payload.get("sample_index", None)
    requested_k = full_payload.get("requested_k", None)
    completed_jobs = int(full_payload.get("completed_jobs", 0))
    total_jobs = int(full_payload.get("total_jobs", 0))
    eta_seconds = float(full_payload.get("eta_seconds", 0.0) or 0.0)
    print(
        (
            f"[progress] status={status} stage={stage} variant={variant} "
            f"sample={sample_index} K={requested_k} jobs={completed_jobs}/{total_jobs} "
            f"eta_sec={eta_seconds:.1f}"
        ),
        flush=True,
    )


def write_test_report(status: str, **fields: Any) -> None:
    payload = {
        "status": str(status),
        "updated_at": iso_now(),
        "artifacts_dir": str(V2_ARTIFACTS_DIR),
    }
    payload.update(fields)
    write_json(V2_ARTIFACTS_DIR / "test_report.json", payload)


def file_exists_and_valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        read_json(path)
        return True
    except Exception:
        return False


def archive_invalidated_artifact(path: Path, reason: str) -> None:
    if not path.exists():
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.stem}.invalidated_by_{reason}.{timestamp}{path.suffix}")
    path.replace(archived)


@dataclass
class MCQMV2EvalRuntime:
    variant: str
    oracle_mode: str
    cfg: Any
    dataset: Any
    model: Any
    runtime_meta: dict[str, Any]
    checkpoint_sha256: str
    architecture_signature_sha256: str
    runtime_config_hash: str


@dataclass
class OracleE2ETimingTotals:
    wall_seconds: float = 0.0
    model_build_seconds: float = 0.0
    checkpoint_load_seconds: float = 0.0
    data_prepare_seconds: float = 0.0
    cuda_transfer_seconds: float = 0.0
    forward_seconds: float = 0.0
    metric_seconds: float = 0.0
    cpu_serialize_seconds: float = 0.0
    artifact_write_seconds: float = 0.0
    teacher_capture_seconds: float = 0.0
    seed_capture_seconds: float = 0.0
    alignment_seconds: float = 0.0
    baseline_seconds: float = 0.0
    model_build_count: int = 0
    checkpoint_load_count: int = 0


@dataclass
class OracleE2EContext:
    init_checkpoint_path: Path
    init_checkpoint_sha256: str
    preflight: dict[str, Any]
    forward_artifact_reused: bool
    alignment_artifact_reused: bool
    suite_started_at: float
    timing_totals: OracleE2ETimingTotals = field(default_factory=OracleE2ETimingTotals)
    cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]] = field(default_factory=dict)
    p0_seed_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    p0_baseline_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    seeded_alignment_cache: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    teacher_by_sample: dict[int, dict[str, Any]] = field(default_factory=dict)
    runtimes: dict[str, MCQMV2EvalRuntime] = field(default_factory=dict)
    partial_rows_by_job: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_jobs: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=[
            "contract",
            "pairs",
            "stage_a",
            "stage_a_pair",
            "stage_b_smoke",
            "stage_c_smoke",
            "train_stage_b",
            "train_stage_c",
            "train_stage_d",
            "eval",
            "eval_fix_suite",
            "eval_v2_interface_oracle",
            "eval_v2_oracle_e2e_only",
            "eval_v2_decision_only",
            "audit_v2_geometry_only",
            "train_q2f_preflight",
            "train_q2f",
            "eval_q2f",
            "full",
        ],
        default="stage_a",
    )
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--sample-end", type=int, default=19)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--stage-b-iters", type=int, default=1000)
    p.add_argument("--stage-c-iters", type=int, default=3000)
    p.add_argument("--stage-d-iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--variant", default="A2")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def mcqm_variant_cfg(variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {
        "enabled": True,
        "injection_after_layer_idx": 1,
        "replacement_budget": 64,
        "continuity_tol_sec": 1.0,
        "mcqm_motion_mode": "ego_only",
        "disable_source_embedding": False,
        "disable_novelty": False,
        "replacement_margin": 0.05,
        "memory_confidence_floor": 0.50,
        "memory_in_range_floor": 0.75,
        "motion_sigma": 2.0,
        "static_max_motion_residual": 1.5,
        "dynamic_max_motion_residual": 6.0,
        "legacy_forced_replacement": False,
        "detach_memory_logits": True,
        "mcqm_projector_mode": "identity_safe",
        "static_only_in_ego_mode": True,
        "allow_memory_repropagation": False,
        "matching_mode": "local_same_class",
        "local_match_radius_m": 2.0,
        "native_bottomk_ratio": 0.3,
        "mcqm_full_time_mode": "reg_only_delta_t",
        "mcqm_full_residual_mode": "full_all_reg",
        "strict_memory_lineage": True,
    }
    meta = {"label": variant, "uses_mcqm": True}
    if variant == "N0":
        base["enabled"] = False
        meta["uses_mcqm"] = False
    elif variant == "A1":
        base["mcqm_projector_mode"] = "raw_bypass"
    elif variant == "A2":
        base["mcqm_projector_mode"] = "identity_safe"
    elif variant == "A3":
        base["mcqm_projector_mode"] = "identity_safe"
        base["allow_memory_repropagation"] = True
    elif variant == "A4A":
        base["mcqm_projector_mode"] = "identity_safe"
        base["matching_mode"] = "global_same_class"
    elif variant == "A4B":
        base["mcqm_projector_mode"] = "identity_safe"
        base["matching_mode"] = "global"
    elif variant == "N1":
        base["mcqm_projector_mode"] = "identity_safe"
        base["replacement_budget"] = 0
    elif variant == "T0":
        base["mcqm_projector_mode"] = "identity_safe"
        base["mcqm_motion_mode"] = "full"
        base["static_only_in_ego_mode"] = False
        base["mcqm_full_time_mode"] = "fixed_horizon_mcqm"
    elif variant == "T1":
        base.update({
            "mcqm_projector_mode": "identity_safe",
            "mcqm_motion_mode": "full",
            "static_only_in_ego_mode": False,
            "mcqm_full_time_mode": "reg_only_delta_t",
            "mcqm_full_residual_mode": "full_all_reg",
        })
    elif variant == "T2":
        base.update({
            "mcqm_projector_mode": "identity_safe",
            "mcqm_motion_mode": "full",
            "static_only_in_ego_mode": False,
            "mcqm_full_time_mode": "reg_and_vel_delta_t",
            "mcqm_full_residual_mode": "full_all_reg",
        })
    elif variant == "T3":
        base.update({
            "mcqm_projector_mode": "identity_safe",
            "mcqm_motion_mode": "full",
            "static_only_in_ego_mode": False,
            "mcqm_full_time_mode": "reg_only_delta_t",
            "mcqm_full_residual_mode": "full_dynamic_only",
        })
    else:
        raise ValueError(f"unsupported MCQM variant: {variant}")
    return base, meta


def build_fault_catalog():
    return sw5.build_catalog()


def resolve_checkpoint_path(path_like: str | Path | None) -> Path:
    if path_like is None or str(path_like) == "":
        return CHECKPOINT_PATH
    path = Path(path_like)
    if path.is_absolute():
        return path
    candidate_roots = (
        Path.cwd(),
        WORKSPACE_ROOT,
        PROJECT_ROOT,
    )
    for root in candidate_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def object_sha256(obj: Any) -> str:
    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            normalized = {}
            for key, val in value.items():
                normalized_key = str(_normalize(key))
                if normalized_key in normalized:
                    raise ValueError(f"duplicate object key after normalization: {normalized_key}")
                normalized[normalized_key] = _normalize(val)
            return normalized
        if isinstance(value, list):
            return [_normalize(v) for v in value]
        if isinstance(value, tuple):
            return [_normalize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value

    payload = json.dumps(_normalize(sw2.to_cpu_artifact(obj)), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def torch_load_compat(path: str | Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def normalize_state_dict_key(key: str) -> str:
    prefixes = ("module.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


def is_mcqm_only_key(key: str) -> bool:
    parts = key.split(".")
    if any(part == "mcqm_memory_projector" or part.startswith("mcqm_") for part in parts):
        return True
    padded = f".{key}."
    return ".mcqm_memory_projector." in padded or ".mcqm_" in padded


def model_state_sha256(
    model: torch.nn.Module,
    excluded_prefixes: tuple[str, ...] = (),
    excluded_keys: tuple[str, ...] = (),
) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if any(key.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if key in excluded_keys:
            continue
        cpu = value.detach().contiguous().cpu()
        h.update(key.encode("utf-8"))
        h.update(str(cpu.dtype).encode("utf-8"))
        h.update(str(tuple(cpu.shape)).encode("utf-8"))
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def checkpoint_file_info(requested_path: str | Path, checkpoint_obj: Any | None = None) -> dict[str, Any]:
    resolved = resolve_checkpoint_path(requested_path)
    meta = checkpoint_obj.get("meta", {}) if isinstance(checkpoint_obj, dict) else {}
    return {
        "requested_checkpoint_path": str(requested_path),
        "resolved_checkpoint_path": str(resolved),
        "checkpoint_sha256": file_sha256(resolved),
        "checkpoint_size_bytes": int(resolved.stat().st_size),
        "checkpoint_meta_epoch": int(meta.get("epoch", -1)) if isinstance(meta, dict) and meta.get("epoch") is not None else -1,
        "checkpoint_meta_iter": int(meta.get("iter", -1)) if isinstance(meta, dict) and meta.get("iter") is not None else -1,
    }


def checkpoint_state_audit_full(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    ckpt = torch_load_compat(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint state_dict must be a dict")
    normalized_state_dict: dict[str, torch.Tensor] = {}
    normalization_collisions: dict[str, list[str]] = {}
    normalization_sources: dict[str, list[str]] = defaultdict(list)
    non_tensor_keys: list[str] = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            non_tensor_keys.append(key)
            continue
        norm_key = normalize_state_dict_key(key)
        normalization_sources[norm_key].append(key)
        if norm_key in normalized_state_dict:
            normalization_collisions[norm_key] = list(normalization_sources[norm_key])
            continue
        normalized_state_dict[norm_key] = value
    if non_tensor_keys:
        raise RuntimeError(
            "checkpoint state_dict contains non-tensor values: "
            f"{non_tensor_keys[:20]}"
        )
    model_state = model.state_dict()
    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    shape_mismatches = []
    for key, value in normalized_state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append({
                "key": key,
                "ckpt_shape": list(value.shape),
                "model_shape": list(model_state[key].shape),
            })
            continue
        matched_keys.append(key)
    for key in model_state.keys():
        if key not in normalized_state_dict:
            missing_keys.append(key)
    return {
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "normalization_collisions": normalization_collisions,
        "normalization_sources": dict(normalization_sources),
        "normalized_state_dict": normalized_state_dict,
        "checkpoint_meta": ckpt.get("meta", {}) if isinstance(ckpt, dict) else {},
    }


def matched_parameter_content_audit(
    model: torch.nn.Module,
    normalized_checkpoint_state: dict[str, torch.Tensor],
    excluded_key_predicate,
) -> dict[str, Any]:
    mismatch = []
    model_state = model.state_dict()
    matched_key_count = 0
    matched_numel = 0
    core_model_key_count = 0
    core_model_numel = 0
    for key, value in model_state.items():
        if excluded_key_predicate(key):
            continue
        core_model_key_count += 1
        core_model_numel += int(value.numel())
    for key, ckpt_value in normalized_checkpoint_state.items():
        if excluded_key_predicate(key):
            continue
        if key not in model_state:
            continue
        if tuple(model_state[key].shape) != tuple(ckpt_value.shape):
            continue
        matched_key_count += 1
        matched_numel += int(ckpt_value.numel())
        loaded = model_state[key].detach().cpu()
        expected = ckpt_value.detach().cpu()
        if not torch.equal(loaded, expected):
            mismatch.append(key)
    return {
        "content_mismatch_count": len(mismatch),
        "content_mismatch_preview": mismatch[:20],
        "matched_key_count": matched_key_count,
        "matched_parameter_numel": matched_numel,
        "core_model_key_count": core_model_key_count,
        "core_model_numel": core_model_numel,
        "matched_key_coverage": float(matched_key_count / core_model_key_count) if core_model_key_count else 1.0,
        "matched_numel_coverage": float(matched_numel / core_model_numel) if core_model_numel else 1.0,
    }


def container_is_empty(value: Any) -> bool | None:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    if torch.is_tensor(value):
        return value.numel() == 0
    if hasattr(value, "__len__"):
        try:
            return len(value) == 0
        except Exception:
            pass
    if hasattr(value, "empty"):
        try:
            return bool(value.empty())
        except Exception:
            pass
    return None


def is_valid_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value.lower())


def mcqm_memory_is_empty(model: Any) -> dict[str, Any]:
    manager = getattr(model, "mcqm_memory_manager", None)
    if manager is None:
        return {"verified": True, "empty": True, "state_container_type": "none"}
    state = getattr(manager, "_state_by_batch", None)
    if isinstance(state, dict):
        return {"verified": True, "empty": len(state) == 0, "state_container_type": "dict"}
    return {"verified": False, "empty": False, "state_container_type": type(state).__name__ if state is not None else "missing"}


def r8_cache_is_empty(model: Any) -> dict[str, Any]:
    memory = getattr(model, "memory", None)
    queue_obj = getattr(model, "queue", None)
    memory_empty = container_is_empty(memory)
    queue_empty = container_is_empty(queue_obj)
    verified = memory_empty is not None and queue_empty is not None
    return {
        "verified": verified,
        "empty": bool(memory_empty) and bool(queue_empty) if verified else False,
        "memory_container_type": type(memory).__name__ if memory is not None else "none",
        "queue_container_type": type(queue_obj).__name__ if queue_obj is not None else "none",
        "reason": "" if verified else "unsupported cache container",
    }


def batch_img_tensor(batch: dict[str, Any]) -> torch.Tensor:
    img = batch["img"]
    if isinstance(img, list):
        img = img[0]
    if hasattr(img, "data"):
        img = img.data[0]
    return img


def sample_gt_hashes(sample_unwrapped: dict[str, Any]) -> dict[str, str]:
    gt_current = torch.as_tensor(sample_unwrapped["voxel_semantics"]).long()
    temporal_semantics = sample_unwrapped.get("temporal_semantics", [])
    gt_temporal_list = [gt_current]
    if isinstance(temporal_semantics, list):
        for item in temporal_semantics:
            if isinstance(item, dict) and "voxel_semantics" in item:
                gt_temporal_list.append(torch.as_tensor(item["voxel_semantics"]).long())
    gt_temporal = torch.stack(gt_temporal_list, dim=0)
    return {
        "gt_occ_sha256": tensor_sha256(gt_current),
        "gt_temporal_sha256": tensor_sha256(gt_temporal),
    }


def camera_valid_mask_tensor(total_views: int, affected_indices: list[int]) -> torch.Tensor:
    mask = torch.ones(total_views, dtype=torch.uint8)
    if affected_indices:
        mask[torch.as_tensor(affected_indices, dtype=torch.long)] = 0
    return mask


def sample_protocol_record(
    dataset: Any,
    sample_index: int,
    sample_unwrapped: dict[str, Any],
    degraded_batch: dict[str, Any],
    perturb_manifest: dict[str, Any],
) -> dict[str, Any]:
    info = dataset.data_infos[int(sample_index)]
    meta = unwrap_meta(degraded_batch)
    valid_mask = camera_valid_mask_tensor(len(meta.get("filename", [])), list(perturb_manifest.get("affected_indices", [])))
    meta_subset = {
        "filename": meta.get("filename"),
        "img_timestamp": meta.get("img_timestamp"),
        "ego2global": meta.get("ego2global"),
        "ego2lidar": meta.get("ego2lidar"),
        "lidar2img": meta.get("lidar2img"),
        "scene_token": meta.get("scene_token"),
        "timestamp": meta.get("timestamp"),
        "sample_idx": meta.get("sample_idx"),
    }
    out = {
        "sample_index": int(sample_index),
        "sample_token": str(info.get("token", "")),
        "scene_token": str(info.get("scene_token", "")),
        "timestamp": float(info.get("timestamp", 0)) / 1e6,
        "degraded_input_sha256": tensor_sha256(batch_img_tensor(degraded_batch)),
        "camera_mask_sha256": tensor_sha256(valid_mask),
        "perturbation_config_hash": object_sha256(perturb_manifest),
        "img_metas_hash": object_sha256(meta_subset),
        **sample_gt_hashes(sample_unwrapped),
    }
    return out


def diff_meta_subset(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    diffs = []

    def _shape_of(value: Any) -> list[int] | None:
        if torch.is_tensor(value):
            return list(value.shape)
        if isinstance(value, np.ndarray):
            return list(value.shape)
        return None

    def _numeric_array(value: Any):
        if torch.is_tensor(value):
            return value.detach().cpu().float().numpy()
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        return None

    def _container_kind(value: Any) -> str:
        if torch.is_tensor(value):
            return "tensor"
        if isinstance(value, np.ndarray):
            return "ndarray"
        if isinstance(value, list):
            return "list"
        if isinstance(value, tuple):
            return "tuple"
        return type(value).__name__

    def _walk(path: str, lval: Any, rval: Any) -> None:
        lnum = _numeric_array(lval)
        rnum = _numeric_array(rval)
        if lnum is not None and rnum is not None and lnum.shape == rnum.shape:
            max_abs_diff = float(np.max(np.abs(lnum - rnum))) if lnum.size else 0.0
            equal = bool(max_abs_diff == 0.0)
            diffs.append({
                "field_path": path,
                "left_type": _container_kind(lval),
                "right_type": _container_kind(rval),
                "left_shape": _shape_of(lval),
                "right_shape": _shape_of(rval),
                "max_abs_diff": max_abs_diff,
                "value_equal": equal,
                "container_only_difference": equal and _container_kind(lval) != _container_kind(rval),
                "benign_representation_difference": equal and _container_kind(lval) != _container_kind(rval),
            })
            return
        if isinstance(lval, (list, tuple)) and isinstance(rval, (list, tuple)):
            max_len = max(len(lval), len(rval))
            for idx in range(max_len):
                _walk(f"{path}[{idx}]", lval[idx] if idx < len(lval) else None, rval[idx] if idx < len(rval) else None)
            return
        if isinstance(lval, dict) and isinstance(rval, dict):
            for key in sorted(set(lval.keys()) | set(rval.keys())):
                _walk(f"{path}.{key}" if path else str(key), lval.get(key), rval.get(key))
            return
        equal = object_sha256(lval) == object_sha256(rval)
        diffs.append({
            "field_path": path,
            "left_type": _container_kind(lval),
            "right_type": _container_kind(rval),
            "left_shape": _shape_of(lval),
            "right_shape": _shape_of(rval),
            "max_abs_diff": 0.0 if equal else float("nan"),
            "value_equal": bool(equal),
            "container_only_difference": bool(equal) and _container_kind(lval) != _container_kind(rval),
            "benign_representation_difference": bool(equal) and _container_kind(lval) != _container_kind(rval),
        })

    _walk("", left, right)
    protocol_mismatch = any(
        (not row["value_equal"]) and (
            row["field_path"].startswith("ego2global")
            or row["field_path"].startswith("ego2lidar")
            or row["field_path"].startswith("lidar2img")
            or row["field_path"] == "timestamp"
        )
        for row in diffs
    )
    return {"field_diffs": diffs, "protocol_mismatch": protocol_mismatch}


def build_runtime_strict(
    train: bool,
    mcqm_cfg: dict[str, Any] | None,
    checkpoint_path: str | Path,
    *,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    cfg, dataset, model = build_runtime_unloaded(train=train, mcqm_cfg=mcqm_cfg, cfg_overrides=cfg_overrides)
    checkpoint_audit = load_model_state_strict(
        model,
        checkpoint_path,
        allowed_missing_keys=set(),
    )
    checkpoint_meta = checkpoint_audit.get("checkpoint_meta", {})
    if isinstance(checkpoint_meta, dict) and "architecture_signature" in checkpoint_meta and mcqm_cfg is not None:
        expected = checkpoint_meta["architecture_signature"]
        current = build_mcqm_v2_architecture_signature(
            model,
            mcqm_cfg,
            checkpoint_meta.get("base_checkpoint_sha256", file_sha256(resolve_checkpoint_path(CHECKPOINT_PATH))),
        )
        expected_structural = expected["structural_signature"] if isinstance(expected, dict) and "structural_signature" in expected else expected
        current_structural = current["structural_signature"]
        if object_sha256(expected_structural) != object_sha256(current_structural):
            raise RuntimeError("MCQM V2 structural architecture signature mismatch")
    model = model.cuda()
    if hasattr(model, "set_epoch"):
        meta = checkpoint_audit.get("checkpoint_meta", {})
        if isinstance(meta, dict) and meta.get("epoch") is not None:
            model.set_epoch(int(meta["epoch"]))
    model.train() if train else model.eval()
    runtime_meta = {
        **checkpoint_file_info(checkpoint_path),
        "strict_state_dict_audit": {
            "expected_new_v2_keys": [],
            "unexpected_missing_core_keys": [],
            "unexpected_checkpoint_keys": [],
            "shape_mismatch_keys": [],
        },
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_parameter_key_hash": object_sha256(sorted(model.state_dict().keys())),
        "core_model_state_sha256": model_state_sha256(model),
    }
    return cfg, dataset, model, runtime_meta


def create_mcqm_v2_init_checkpoint() -> dict[str, Any]:
    base_checkpoint_path = resolve_checkpoint_path(CHECKPOINT_PATH)
    base_checkpoint_sha256 = file_sha256(base_checkpoint_path)
    base_cfg = mcqm_v2_variant_cfg("P0")
    cfg, dataset, model = build_runtime_unloaded(
        train=False,
        mcqm_cfg=base_cfg,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    expected_new_v2_keys = set(expected_new_v2_key_names(model))
    epoch56_audit = load_model_state_strict(
        model,
        base_checkpoint_path,
        allowed_missing_keys=expected_new_v2_keys,
    )
    parameter_contract = apply_mcqm_v2_parameter_freeze(model)
    signature = build_mcqm_v2_architecture_signature(model, base_cfg, base_checkpoint_sha256)
    V2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    init_checkpoint_path = V2_ARTIFACTS_DIR / "checkpoints/mcqm_v2_init_from_epoch56.pth"
    init_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "meta": {
            "base_checkpoint_path": str(base_checkpoint_path),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "architecture_signature": signature,
            "architecture_signature_sha256": signature["architecture_signature_sha256"],
            "runtime_config_signature_sha256": signature["runtime_config_signature_sha256"],
            "mcqm_v2_config": base_cfg,
            "projector_parameter_shapes": signature["structural_signature"]["projector_parameter_shapes"],
            "trainable_parameter_keys": parameter_contract["trainable_parameter_keys"],
            "frozen_parameter_keys": parameter_contract["frozen_parameter_keys"],
            "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_info": {},
        },
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint_payload, init_checkpoint_path)
    reload_cfg, reload_dataset, reload_model = build_runtime_unloaded(
        train=False,
        mcqm_cfg=base_cfg,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    _ = reload_cfg, reload_dataset
    reload_audit = load_model_state_strict(reload_model, init_checkpoint_path, allowed_missing_keys=set())
    content_audit = matched_parameter_content_audit(
        reload_model,
        reload_audit["normalized_state_dict"],
        excluded_key_predicate=lambda _key: False,
    )
    if content_audit["content_mismatch_count"] != 0:
        raise RuntimeError("V2 init checkpoint content mismatch after strict reload")
    manifest = {
        "base_checkpoint_path": str(base_checkpoint_path),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "epoch56_core_load_audit": {
            "expected_new_v2_keys": sorted(expected_new_v2_keys),
            "unexpected_missing_core_keys": epoch56_audit["unexpected_missing_core_keys"],
            "unexpected_checkpoint_keys": epoch56_audit["unexpected_checkpoint_keys"],
            "shape_mismatch_keys": epoch56_audit["shape_mismatch_keys"],
        },
        "parameter_contract": parameter_contract,
        "architecture_signature": signature,
        "init_checkpoint_path": str(init_checkpoint_path),
        "init_checkpoint_sha256": file_sha256(init_checkpoint_path),
        "strict_reload_audit": {
            "unexpected_missing_core_keys": reload_audit["unexpected_missing_core_keys"],
            "unexpected_checkpoint_keys": reload_audit["unexpected_checkpoint_keys"],
            "shape_mismatch_keys": reload_audit["shape_mismatch_keys"],
            "content_mismatch_count": content_audit["content_mismatch_count"],
        },
    }
    write_json(V2_ARTIFACTS_DIR / "mcqm_v2_architecture_contract.json", signature)
    write_json(V2_ARTIFACTS_DIR / "mcqm_v2_parameter_contract.json", parameter_contract)
    write_json(V2_ARTIFACTS_DIR / "mcqm_v2_checkpoint_audit.json", manifest)
    write_json(V2_ARTIFACTS_DIR / "mcqm_v2_init_checkpoint_manifest.json", manifest)
    write_json(V2_ARTIFACTS_DIR / "trainable_parameter_keys.json", parameter_contract["trainable_parameter_keys"])
    write_json(V2_ARTIFACTS_DIR / "frozen_parameter_keys.json", parameter_contract["frozen_parameter_keys"])
    write_json(V2_ARTIFACTS_DIR / "parameter_contract.json", parameter_contract)
    return manifest


def freeze_v1_reference() -> dict[str, Any]:
    source_path = ARTIFACTS_DIR / "eval_fix_suite_results.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    stage_d = payload["stage_d_same_checkpoint_fair_comparison"]
    original = payload["original_checkpoint_historical_reference"]
    frozen = {
        "original_checkpoint_sha256": payload["legacy_path_audit"]["B1_old_path_default_sha256"],
        "stage_d_checkpoint_sha256": stage_d["checkpoint_sha256_values"][0],
        "B0_B1_B2_original": {row["Variant"]: row for row in original["table"]},
        "B0S_B1S_B2S_stage_d": {
            row["Variant"]: row
            for row in stage_d["table"]
            if row["Variant"] in {"B0S_STAGE_D_CLEAN", "B1S_STAGE_D_DEGRADED", "B2S_STAGE_D_R8"}
        },
        "N1": next(row for row in stage_d["table"] if row["Variant"] == "N1_STAGE_D_CALLBACK_NO_INJECTION"),
        "variants": {
            row["Variant"]: row
            for row in stage_d["table"]
            if row["Variant"] in {
                "A2", "A3", "A4A_GLOBAL_SAME_CLASS", "A4B_GLOBAL_ANY_CLASS", "T0", "T1", "T2", "T3",
            }
        },
        "delta_vs_N1": stage_d["delta_vs_n1"],
        "fair_eval_status_label": "MCQM_FAIR_EVAL_ESTABLISHED_BUT_CURRENT_INJECTION_HAS_NO_FORWARD_GAIN",
    }
    write_json(V2_ARTIFACTS_DIR / "v1_frozen_reference.json", frozen)
    return frozen


def ensure_collect_meta_keys(cfg: Any) -> None:
    required = {"ego2global", "ego2lidar", "occ_gt_path", "img_timestamp", "sample_idx"}

    def patch_pipeline(pipeline: list[Any]) -> None:
        for step in pipeline:
            if isinstance(step, dict) and step.get("type") == "MultiScaleFlipAug3D":
                patch_pipeline(step.get("transforms", []))
            if isinstance(step, dict) and step.get("type") == "Collect4D":
                keys = list(step.get("meta_keys", []))
                for key in required:
                    if key not in keys:
                        keys.append(key)
                step["meta_keys"] = keys

    for split in ["train", "val", "test"]:
        split_cfg = getattr(cfg.data, split, None)
        if split_cfg is not None and hasattr(split_cfg, "pipeline"):
            patch_pipeline(split_cfg.pipeline)


def build_runtime_unloaded(
    train: bool,
    mcqm_cfg: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
):
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    from mmcv import Config
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    cfg = Config.fromfile(str(BASE_CONFIG))
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)
    setup_multi_processes(cfg)
    cfg.gpu_ids = [0]
    cfg.model.pretrained = None
    ensure_collect_meta_keys(cfg)
    if mcqm_cfg is not None:
        cfg.model.mcqm_cfg = dict(mcqm_cfg)
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            cfg.merge_from_dict({key: value})
    split_cfg = cfg.data.train if train else cfg.data.val
    split_cfg.test_mode = not train
    dataset = build_dataset(split_cfg)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    return cfg, dataset, model


def build_dataset_only(
    train: bool,
    *,
    cfg_overrides: dict[str, Any] | None = None,
):
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    from mmcv import Config
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import patch_config

    cfg = Config.fromfile(str(BASE_CONFIG))
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)
    setup_multi_processes(cfg)
    cfg.gpu_ids = [0]
    ensure_collect_meta_keys(cfg)
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            cfg.merge_from_dict({key: value})
    split_cfg = cfg.data.train if train else cfg.data.val
    split_cfg.test_mode = not train
    dataset = build_dataset(split_cfg)
    return cfg, dataset


def build_runtime(
    train: bool,
    mcqm_cfg: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
):
    from mmcv.runner import load_checkpoint

    cfg, dataset, model = build_runtime_unloaded(train=train, mcqm_cfg=mcqm_cfg, cfg_overrides=cfg_overrides)
    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    state_dict_audit = sw2.ckpt_state_audit(model, resolved_checkpoint_path)
    checkpoint = load_checkpoint(model, str(resolved_checkpoint_path), map_location="cpu")
    model = model.cuda()
    meta_epoch = 56
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
        meta_epoch = int(checkpoint["meta"].get("epoch", meta_epoch))
    if hasattr(model, "set_epoch"):
        model.set_epoch(meta_epoch)
    model.train() if train else model.eval()
    runtime_meta = {
        **checkpoint_file_info(checkpoint_path if checkpoint_path is not None else resolved_checkpoint_path, checkpoint),
        "state_dict_audit": state_dict_audit,
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_parameter_key_hash": object_sha256(sorted(model.state_dict().keys())),
        "core_model_state_sha256": model_state_sha256(model),
    }
    return cfg, dataset, model, runtime_meta


def build_pairs(dataset: Any, start: int, end: int) -> list[dict[str, Any]]:
    infos = dataset.data_infos
    rows = []
    prev = None
    for idx in range(start, min(end + 1, len(infos))):
        info = infos[idx]
        row = {
            "index": idx,
            "scene_token": info.get("scene_token", ""),
            "scene_name": info.get("scene_name", ""),
            "timestamp": int(info.get("timestamp", 0)),
        }
        if prev is not None and prev["scene_token"] == row["scene_token"]:
            rows.append(
                {
                    "prev_index": prev["index"],
                    "curr_index": row["index"],
                    "scene_token": row["scene_token"],
                    "scene_name": row["scene_name"],
                    "delta_us": row["timestamp"] - prev["timestamp"],
                }
            )
        prev = row
    return rows


def mcqm_v2_variant_cfg(variant: str, *, oracle_mode: str = "disabled") -> dict[str, Any]:
    base = {
        "enabled": True,
        "mcqm_version": "v2",
        "injection_after_layer_idx": 1,
        "replacement_budget": 64,
        "continuity_tol_sec": 1.0,
        "mcqm_motion_mode": "ego_only",
        "disable_source_embedding": False,
        "disable_novelty": False,
        "replacement_margin": 0.05,
        "memory_confidence_floor": 0.50,
        "memory_in_range_floor": 0.75,
        "motion_sigma": 2.0,
        "static_max_motion_residual": 1.5,
        "dynamic_max_motion_residual": 6.0,
        "detach_memory_logits": True,
        "static_only_in_ego_mode": True,
        "allow_memory_repropagation": False,
        "matching_mode": "local_same_class",
        "local_match_radius_m": 2.0,
        "native_bottomk_ratio": 0.3,
        "strict_memory_lineage": True,
        "mcqm_v2_oracle_mode": oracle_mode,
        "mcqm_v2_oracle_lambda_point": 1.0,
        "mcqm_v2_injection_mode": "passthrough",
        "mcqm_residual_alpha": 0.0,
    }
    if variant == "P0":
        return base
    if variant == "H0":
        base["mcqm_v2_injection_mode"] = "hard_feature_geometry"
        return base
    if variant == "H1":
        base["mcqm_v2_injection_mode"] = "feature_only_replace"
        return base
    if variant == "H3":
        base["mcqm_v2_injection_mode"] = "geometry_only"
        return base
    if variant.startswith("H2_A"):
        base["mcqm_v2_injection_mode"] = "feature_residual"
        alpha = float(variant.split("_A", 1)[1]) / 100.0
        base["mcqm_residual_alpha"] = alpha
        return base
    raise ValueError(f"unsupported MCQM V2 variant: {variant}")


def expected_new_v2_key_names(model: torch.nn.Module) -> list[str]:
    return sorted(
        key for key in model.state_dict().keys()
        if key.startswith("mcqm_v2_")
    )


def apply_mcqm_v2_parameter_freeze(model: torch.nn.Module) -> dict[str, Any]:
    trainable_prefixes = (
        "mcqm_v2_projector.",
        "mcqm_v2_residual_gate",
        "mcqm_v2_reliability_scorer.",
        "mcqm_v2_utility_scorer.",
    )
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        allow = any(name.startswith(prefix) for prefix in trainable_prefixes)
        param.requires_grad = allow
        if allow:
            trainable.append(name)
        else:
            frozen.append(name)
    non_v2_trainable = [name for name, param in model.named_parameters() if param.requires_grad and not name.startswith("mcqm_v2_")]
    if non_v2_trainable:
        raise RuntimeError(f"non-v2 parameters left trainable: {non_v2_trainable[:20]}")
    return {
        "trainable_parameter_keys": trainable,
        "frozen_parameter_keys": frozen,
        "trainable_parameter_count": int(sum(model.state_dict()[k].numel() for k in trainable if k in model.state_dict())),
        "frozen_parameter_count": int(sum(model.state_dict()[k].numel() for k in frozen if k in model.state_dict())),
    }


def mcqm_q2f_v1_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "mcqm_version": "q2f_v3_shallow",
        "mcqm_q2f_enabled": True,
        "mcqm_q2f_failed_camera_only": True,
        "mcqm_q2f_memory_mask_value": 1.0,
        "mcqm_motion_mode": "ego_only",
        "strict_memory_lineage": True,
        "detach_memory_state": True,
        "allow_memory_repropagation": False,
        "mcqm_q2f_scatter_mode": "bilinear_splat",
        "mcqm_q2f_aggregation": "confidence_weighted_mean",
        "mcqm_q2f_decoder_depth": 2,
        "mcqm_q2f_kernel_size": 3,
        "mcqm_q2f_injection_mode": "shallow_hard_replace",
        "mcqm_q2f_feature_distill_enabled": True,
        "mcqm_q2f_feature_loss_weight": 1.0,
        "mcqm_q2f_cosine_loss_weight": 0.1,
        "mcqm_q2f_occ_loss_weight": 1.0,
        "mcqm_q2f_freeze_base_model": True,
        "mcqm_q2f_failed_camera_names": ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"],
    }


def apply_mcqm_q2f_parameter_freeze(model: torch.nn.Module) -> dict[str, Any]:
    trainable_prefixes = (
        "mcqm_q2f_shallow_query_projector.",
        "mcqm_q2f_shallow_decoder.",
    )
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        allow = any(name.startswith(prefix) for prefix in trainable_prefixes)
        param.requires_grad = allow
        if allow:
            trainable.append(name)
        else:
            frozen.append(name)
    non_q2f_trainable = [name for name, param in model.named_parameters() if param.requires_grad and not name.startswith("mcqm_q2f_")]
    if non_q2f_trainable:
        raise RuntimeError(f"non-q2f parameters left trainable: {non_q2f_trainable[:20]}")
    trainable_param_count = int(sum(param.numel() for name, param in model.named_parameters() if name in trainable))
    frozen_param_count = int(sum(param.numel() for name, param in model.named_parameters() if name in frozen))
    total = trainable_param_count + frozen_param_count
    return {
        "trainable_parameter_keys": trainable,
        "frozen_parameter_keys": frozen,
        "trainable_parameter_count": trainable_param_count,
        "frozen_parameter_count": frozen_param_count,
        "trainable_parameter_ratio": float(trainable_param_count / total) if total else 0.0,
    }


def build_mcqm_q2f_architecture_signature(
    model: Any,
    cfg: dict[str, Any],
    base_checkpoint_sha256: str,
    *,
    camera_count_override: int | None = None,
) -> dict[str, Any]:
    transformer = model.pts_bbox_head.transformer
    if camera_count_override is not None:
        camera_count = int(camera_count_override)
    else:
        camera_count = 0
        try:
            camera_count = int(model._mcqm_q2f_camera_count())
        except Exception:
            camera_count = int(getattr(transformer, "num_views", 0) or 0)
    topology = model._mcqm_q2f_backbone_topology() if hasattr(model, "_mcqm_q2f_backbone_topology") else {}
    projector_shapes = {
        name: list(param.shape)
        for name, param in model.named_parameters()
        if name.startswith("mcqm_q2f_shallow_query_projector.")
    }
    decoder_shapes = {
        name: list(param.shape)
        for name, param in model.named_parameters()
        if name.startswith("mcqm_q2f_shallow_decoder.")
    }
    structural_signature = {
        "model_class_name": model.__class__.__name__,
        "mcqm_version": str(cfg.get("mcqm_version", "")),
        "query_feature_dim": int(getattr(model, "out_dim", 0)),
        "camera_count": int(camera_count),
        "decoder_num_layers": int(len(transformer.decoder.decoder_layers)),
        "selected_shallow_module_path": str(getattr(model, "mcqm_q2f_selected_module_path", "")),
        "selected_shallow_channels": int((getattr(model, "mcqm_q2f_selected_shallow_shape", [0]) or [0])[0]),
        "selected_shallow_spatial_shape": list((getattr(model, "mcqm_q2f_selected_shallow_shape", [0, 0, 0]) or [0, 0, 0])[1:]),
        "shallow_query_projector_parameter_shapes": projector_shapes,
        "shallow_decoder_parameter_shapes": decoder_shapes,
        "injection_mode": "shallow_hard_replace",
        "downstream_backbone_stage_paths": list(topology.get("downstream_backbone_stage_paths", [])),
        "neck_type": str(topology.get("neck_type", "")),
        "motion_mode": str(cfg.get("mcqm_motion_mode", "")),
        "base_checkpoint_sha256": str(base_checkpoint_sha256),
    }
    structural_signature["architecture_signature_sha256"] = object_sha256(structural_signature)
    runtime_signature = {
        "mcqm_q2f_scatter_mode": str(cfg.get("mcqm_q2f_scatter_mode", "")),
        "mcqm_q2f_aggregation": str(cfg.get("mcqm_q2f_aggregation", "")),
        "mcqm_q2f_decoder_depth": int(cfg.get("mcqm_q2f_decoder_depth", 0)),
        "mcqm_q2f_kernel_size": int(cfg.get("mcqm_q2f_kernel_size", 0)),
        "selected_shallow_module_path": str(getattr(model, "mcqm_q2f_selected_module_path", "")),
        "mcqm_q2f_failed_camera_only": bool(cfg.get("mcqm_q2f_failed_camera_only", False)),
        "mcqm_q2f_feature_distill_enabled": bool(cfg.get("mcqm_q2f_feature_distill_enabled", False)),
        "allow_memory_repropagation": bool(cfg.get("allow_memory_repropagation", False)),
    }
    runtime_signature["runtime_config_signature_sha256"] = object_sha256(runtime_signature)
    return {
        "structural_signature": structural_signature,
        "runtime_config_signature": runtime_signature,
        "architecture_signature_sha256": structural_signature["architecture_signature_sha256"],
        "runtime_config_signature_sha256": runtime_signature["runtime_config_signature_sha256"],
    }


def build_mcqm_v2_architecture_signature(model: Any, cfg: dict[str, Any], base_checkpoint_sha256: str) -> dict[str, Any]:
    head = model.pts_bbox_head
    current_mask = None
    if getattr(head, "ind_stamps_all", None) is not None:
        current_mask = head.ind_stamps_all.reshape(-1) == 0
    structural_signature = {
        "model_class_name": model.__class__.__name__,
        "decoder_num_layers": int(len(head.transformer.decoder.decoder_layers)),
        "query_total_count": int(head.query_embedding.weight.shape[0]) if hasattr(head, "query_embedding") else int(getattr(model, "num_query", 0)),
        "current_query_count": int(current_mask.sum().item()) if current_mask is not None else int(getattr(model, "num_query", 0)),
        "query_feature_dim": int(getattr(model, "out_dim", 0)),
        "num_refines": int(getattr(model, "num_refines", 0)),
        "num_classes": int(getattr(model, "mcqm_num_classes", 0)),
        "current_mask_hash": object_sha256(current_mask.detach().cpu().tolist() if current_mask is not None else []),
        "mcqm_injection_layer_idx": int(cfg.get("injection_after_layer_idx", 1)),
        "projector_parameter_shapes": {
            name: list(param.shape)
            for name, param in model.named_parameters()
            if name.startswith("mcqm_v2_projector.")
        },
        "gate_shapes": {
            name: list(param.shape)
            for name, param in model.named_parameters()
            if name == "mcqm_v2_residual_gate"
        },
        "mcqm_version": str(cfg.get("mcqm_version", "")),
        "allow_memory_repropagation": bool(cfg.get("allow_memory_repropagation", False)),
        "motion_mode": str(cfg.get("mcqm_motion_mode", "")),
        "base_checkpoint_sha256": base_checkpoint_sha256,
    }
    structural_signature["architecture_signature_sha256"] = object_sha256(structural_signature)
    runtime_signature = {
        "mcqm_v2_injection_mode": str(cfg.get("mcqm_v2_injection_mode", "")),
        "mcqm_residual_alpha": None if cfg.get("mcqm_residual_alpha", None) is None else float(cfg.get("mcqm_residual_alpha")),
        "matching_mode": str(cfg.get("matching_mode", "")),
        "replacement_budget": int(cfg.get("replacement_budget", 0)),
        "oracle_mode": str(cfg.get("mcqm_v2_oracle_mode", "disabled")),
        "motion_mode": str(cfg.get("mcqm_motion_mode", "")),
        "modifies_feature": str(cfg.get("mcqm_v2_injection_mode", "")) in {"hard_feature_geometry", "feature_only_replace", "feature_residual"},
        "modifies_geometry": str(cfg.get("mcqm_v2_injection_mode", "")) in {"hard_feature_geometry", "geometry_only"},
    }
    runtime_signature["runtime_config_signature_sha256"] = object_sha256(runtime_signature)
    return {
        "structural_signature": structural_signature,
        "runtime_config_signature": runtime_signature,
        "architecture_signature_sha256": structural_signature["architecture_signature_sha256"],
        "runtime_config_signature_sha256": runtime_signature["runtime_config_signature_sha256"],
    }


def load_model_state_strict(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    allowed_missing_keys: set[str] | None = None,
) -> dict[str, Any]:
    ckpt = torch_load_compat(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint state_dict must be a dict")
    normalized = {}
    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise RuntimeError(f"non-tensor checkpoint entry: {raw_key}")
        norm_key = normalize_state_dict_key(raw_key)
        if norm_key in normalized:
            raise RuntimeError(f"duplicate normalized key: {norm_key}")
        normalized[norm_key] = value
    model_state = model.state_dict()
    allowed_missing = allowed_missing_keys or set()
    missing = sorted(key for key in model_state.keys() if key not in normalized and key not in allowed_missing)
    expected_new = sorted(key for key in model_state.keys() if key not in normalized and key in allowed_missing)
    unexpected = sorted(key for key in normalized.keys() if key not in model_state)
    shape_mismatch = []
    filtered = {}
    for key, value in normalized.items():
        if key not in model_state:
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append({
                "key": key,
                "checkpoint_shape": list(value.shape),
                "model_shape": list(model_state[key].shape),
            })
            continue
        filtered[key] = value
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "strict checkpoint audit failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}"
        )
    load_result = model.load_state_dict(filtered, strict=False)
    if set(load_result.missing_keys) != set(expected_new) or load_result.unexpected_keys:
        raise RuntimeError(
            f"load_state_dict mismatch: missing={load_result.missing_keys} "
            f"unexpected={load_result.unexpected_keys}"
        )
    return {
        "checkpoint_meta": ckpt.get("meta", {}) if isinstance(ckpt, dict) else {},
        "expected_new_v2_keys": expected_new,
        "unexpected_missing_core_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
        "normalized_state_dict": normalized,
    }


def extract_inputs(dataset: Any, sample_index: int):
    raw, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
    return raw, batch


def unwrap_meta(batch: dict[str, Any]) -> dict[str, Any]:
    meta_container = batch["img_metas"]
    if isinstance(meta_container, list):
        container = meta_container[0]
    else:
        container = meta_container
    if hasattr(container, "data"):
        meta = container.data[0]
    else:
        meta = container
    if isinstance(meta, list):
        meta = meta[0]
    return meta


def enrich_batch_scene_meta(batch: dict[str, Any], dataset: Any, sample_index: int) -> dict[str, Any]:
    info = dataset.data_infos[int(sample_index)]
    meta = unwrap_meta(batch)
    meta["scene_token"] = info.get("scene_token", "")
    meta["scene_name"] = info.get("scene_name", "")
    meta["timestamp"] = float(info.get("timestamp", 0)) / 1e6
    return batch


def apply_perturbation_to_batch(batch: dict[str, Any], perturbation_id: str):
    return apply_perturbation_with_manifest_to_batch(batch, perturbation_id)[0]


def apply_perturbation_with_manifest_to_batch(batch: dict[str, Any], perturbation_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if perturbation_id == "A0_clean":
        spec = build_fault_catalog()[perturbation_id]
        compat_batch = {}
        for key, value in batch.items():
            compat_batch[key] = value if isinstance(value, list) else [value]
        perturbed, manifest = sw5.apply_perturbation_to_batch(compat_batch, spec)
        return perturbed, manifest
    catalog = build_fault_catalog()
    spec = catalog[perturbation_id]
    compat_batch = {}
    for key, value in batch.items():
        compat_batch[key] = value if isinstance(value, list) else [value]
    perturbed, manifest = sw5.apply_perturbation_to_batch(compat_batch, spec)
    return perturbed, manifest


def choose_pair_fault_pattern(rng: random.Random) -> tuple[str, str]:
    p = rng.random()
    if p < 0.20:
        return "A0_clean", "A0_clean"
    if p < 0.50:
        return "A0_clean", "A10_drop_front_triplet"
    if p < 0.80:
        return "A10_drop_front_triplet", "A10_drop_front_triplet"
    singles = ["A1_drop_cam_front", "A2_drop_cam_front_left", "A3_drop_cam_front_right"]
    return rng.choice(singles), rng.choice(singles)


def set_stage_trainability(model: torch.nn.Module, stage: str) -> list[str]:
    for _, param in model.named_parameters():
        param.requires_grad = False
    enabled = []
    for name, param in model.named_parameters():
        allow = False
        if stage == "stage_b":
            allow = "mcqm_memory_projector" in name
        elif stage in {"stage_c", "stage_d"}:
            allow = (
                "mcqm_memory_projector" in name
                or ".decoder_layers.2." in name
                or ".decoder_layers.3." in name
                or ".decoder_layers.4." in name
                or ".decoder_layers.5." in name
                or name.startswith("ego_cross_attn.")
                or name.startswith("position_encoder.")
                or name.startswith("reg_branch.")
                or name.startswith("vel_branch.")
                or name.startswith("cls_branch.")
            )
        if allow:
            param.requires_grad = True
            enabled.append(name)
    return enabled


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def compute_clean_preservation_loss(model_mcqm: Any, model_base: Any, curr_inputs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        model_base.train()
        _ = model_base(return_loss=True, **curr_inputs)
    curr_cache = getattr(model_mcqm, "latest_mcqm_train_cache", {})
    base_cache = getattr(model_base, "latest_mcqm_train_cache", {})
    curr_cls = curr_cache["curr_query_cls"]
    base_cls = base_cache["curr_query_cls"].detach()
    curr_occ = torch.sigmoid(curr_cls).amax(dim=-1)
    base_occ = torch.sigmoid(base_cls).amax(dim=-1)
    kl_logits = F.kl_div(
        F.log_softmax(curr_cls, dim=-1),
        F.softmax(base_cls, dim=-1),
        reduction="none",
    ).mean()
    kl_occ = F.kl_div(
        torch.log(torch.clamp(curr_occ, min=1e-6)),
        torch.clamp(base_occ, min=1e-6),
        reduction="none",
    ).mean()
    loss = kl_logits + kl_occ
    return loss, {"clean_preserve_kl_logits": float(kl_logits.detach().cpu().item()), "clean_preserve_kl_occ": float(kl_occ.detach().cpu().item())}


def train_mcqm_stage(
    stage: str,
    sample_start: int,
    sample_end: int,
    iters: int,
    lr: float,
    seed: int,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    mcqm_cfg, _ = mcqm_variant_cfg("A2")
    _, dataset, model, _ = build_runtime(
        train=True,
        mcqm_cfg=mcqm_cfg,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
        checkpoint_path=resolve_checkpoint_path(init_checkpoint),
    )
    model_base = None
    if stage == "stage_d":
        _, _, model_base, _ = build_runtime(
            train=False,
            mcqm_cfg=None,
            cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
            checkpoint_path=resolve_checkpoint_path(CHECKPOINT_PATH),
        )
        model_base.eval()
        for _, param in model_base.named_parameters():
            param.requires_grad = False

    pair_rows = build_pairs(dataset, sample_start, sample_end)
    if not pair_rows:
        raise RuntimeError("no sequential pairs available")
    enabled = set_stage_trainability(model, stage)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=lr)
    if init_checkpoint is not None:
        ckpt = torch_load_compat(resolve_checkpoint_path(init_checkpoint), map_location="cpu")
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
    log_rows = []
    out_ckpt = CHECKPOINT_DIR / f"{stage}_iter{iters:04d}.pth"
    model.train()
    for step in range(iters):
        pair = pair_rows[step % len(pair_rows)]
        prev_fault, curr_fault = choose_pair_fault_pattern(rng)
        if stage == "stage_d" and step % 4 == 0:
            prev_fault, curr_fault = "A0_clean", "A0_clean"
        model.reset_mcqm_state()
        raw_prev, prev_batch = extract_inputs(dataset, int(pair["prev_index"]))
        raw_curr, curr_batch = extract_inputs(dataset, int(pair["curr_index"]))
        prev_batch = enrich_batch_scene_meta(prev_batch, dataset, int(pair["prev_index"]))
        curr_batch = enrich_batch_scene_meta(curr_batch, dataset, int(pair["curr_index"]))
        prev_batch = apply_perturbation_to_batch(prev_batch, prev_fault)
        curr_batch = apply_perturbation_to_batch(curr_batch, curr_fault)
        prev_inputs = sw81.move_train_batch_to_cuda(prev_batch)
        curr_inputs = sw81.move_train_batch_to_cuda(curr_batch)
        model.train()
        with torch.no_grad():
            _ = model(return_loss=True, **prev_inputs)
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **curr_inputs)
        total_loss = None
        scalar_log = {}
        for key, value in losses.items():
            if torch.is_tensor(value):
                scalar_log[key] = float(value.detach().cpu().item())
                total_loss = value if total_loss is None else total_loss + value
        if total_loss is None:
            raise RuntimeError("no tensor loss returned")
        clean_preserve = total_loss.new_tensor(0.0)
        if stage == "stage_d" and prev_fault == "A0_clean" and curr_fault == "A0_clean" and model_base is not None:
            model_base.reset_mcqm_state() if hasattr(model_base, "reset_mcqm_state") else None
            clean_inputs = sw81.move_train_batch_to_cuda(curr_batch)
            clean_preserve, preserve_log = compute_clean_preservation_loss(model, model_base, clean_inputs)
            scalar_log.update(preserve_log)
        total_loss = total_loss + 0.2 * clean_preserve
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(optim_params, max_norm=5.0)
        optimizer.step()
        mcqm_debug = getattr(model, "latest_mcqm_debug", {})
        row = {
            "iter": step,
            "prev_index": int(pair["prev_index"]),
            "curr_index": int(pair["curr_index"]),
            "prev_fault": prev_fault,
            "curr_fault": curr_fault,
            "loss_total": float(total_loss.detach().cpu().item()),
            "actual_replacement_count": int(mcqm_debug.get("debug_rows", [{"actual_replacement_count": -1}])[0]["actual_replacement_count"]) if mcqm_debug else -1,
            "current_query_count": int(mcqm_debug.get("debug_rows", [{"current_query_count": -1}])[0]["current_query_count"]) if mcqm_debug else -1,
            **scalar_log,
        }
        log_rows.append(row)
        if (step + 1) % 50 == 0 or step == 0 or step + 1 == iters:
            log(f"{stage} iter={step+1}/{iters} loss={row['loss_total']:.4f} repl={row['actual_replacement_count']}")
    save_checkpoint(
        out_ckpt,
        model,
        optimizer,
        {
            "stage": stage,
            "iters": iters,
            "lr": lr,
            "seed": seed,
            "sample_start": sample_start,
            "sample_end": sample_end,
            "enabled_names": enabled,
            "mcqm_cfg": mcqm_cfg,
        },
    )
    result = {
        "stage": stage,
        "iters": iters,
        "checkpoint_path": str(out_ckpt),
        "trainable_param_count": int(sum(p.numel() for p in optim_params)),
        "trainable_name_count": len(enabled),
        "enabled_name_sample": enabled[:40],
        "rows": log_rows[-min(20, len(log_rows)):],
    }
    write_json(ARTIFACTS_DIR / f"{stage}.json", result)
    return result


def write_contract() -> None:
    lines = [
        "# MCQM Tensor Contract",
        "",
        "- Config: `num_query=720`, `num_fu_query=[60,60,60,60,40,40]`, `num_layers=6`, `num_refines=[1,4,16,24,32,48]`.",
        "- Decoder total query count: `1040 = 720 + sum(num_fu_query)`.",
        "- Layer0: `[B,1040,1,3] -> reg [B,1040,3] -> [B,1040,1,3]`.",
        "- Layer1: `[B,1040,1,3] -> reg [B,1040,12] -> [B,1040,4,3]`.",
        "- Layer2: `[B,1040,4,3] -> reg [B,1040,48] -> [B,1040,16,3]`.",
        "- Layer3: `[B,1040,16,3] -> reg [B,1040,72] -> [B,1040,24,3]`.",
        "- Layer4: `[B,1040,24,3] -> reg [B,1040,96] -> [B,1040,32,3]`.",
        "- Layer5: `[B,1040,32,3] -> reg [B,1040,144] -> [B,1040,48,3]`.",
        "- `mean(dim=2)` is over per-query refine points.",
        "- Injection point is fixed after decoder layer1 and before layer2.",
    ]
    CONTRACT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage_a(sample_index: int) -> dict[str, Any]:
    _, dataset_base, model_base, _ = build_runtime(train=False, mcqm_cfg=None, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    _, base_batch = extract_inputs(dataset_base, sample_index)
    base_batch = sw2.move_to_cuda(base_batch)
    with torch.no_grad():
        base_out = model_base(return_loss=False, rescale=True, **base_batch)
    base_occ = base_out["semantic_occ_0s"][0]

    mcqm_cfg, _ = mcqm_variant_cfg("A2")
    _, dataset_mcqm, model_mcqm, _ = build_runtime(train=False, mcqm_cfg=mcqm_cfg, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    model_mcqm.reset_mcqm_state()
    _, mcqm_batch = extract_inputs(dataset_mcqm, sample_index)
    mcqm_batch = sw2.move_to_cuda(mcqm_batch)
    with torch.no_grad():
        mcqm_out = model_mcqm(return_loss=False, rescale=True, **mcqm_batch)
    mcqm_occ = mcqm_out["semantic_occ_0s"][0]
    diff = int((base_occ != mcqm_occ).sum())
    result = {
        "sample_index": sample_index,
        "stage_a_first_frame_parity": diff == 0,
        "occupancy_diff_voxels": diff,
        "mcqm_debug": str(getattr(model_mcqm, "latest_mcqm_debug", {})),
    }
    write_json(ARTIFACTS_DIR / "stage_a_parity.json", result)
    return result


def sequential_pair_smoke(sample_index: int) -> dict[str, Any]:
    mcqm_cfg, _ = mcqm_variant_cfg("A2")
    _, dataset, model, _ = build_runtime(train=False, mcqm_cfg=mcqm_cfg, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    model.reset_mcqm_state()
    rows = []
    for idx in [sample_index, sample_index + 1]:
        _, batch = extract_inputs(dataset, idx)
        batch = sw2.move_to_cuda(batch)
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **batch)
        rows.append({"sample_index": idx, "debug_rows": getattr(model, "latest_mcqm_debug", {}).get("debug_rows", [])})
    result = {
        "pair": [sample_index, sample_index + 1],
        "first_frame_memory_used": int(rows[0]["debug_rows"][0]["actual_replacement_count"]),
        "second_frame_memory_used": int(rows[1]["debug_rows"][0]["actual_replacement_count"]),
        "second_frame_current_query_count": int(rows[1]["debug_rows"][0]["current_query_count"]),
        "rows": rows,
    }
    write_json(ARTIFACTS_DIR / "stage_a_pair_memory.json", result)
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value)):
                metrics[key].append(float(value))
    out = {}
    for key, vals in metrics.items():
        out[f"mean_{key}"] = float(np.mean(vals))
    return out


def quantile_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def reset_runtime_state(model: Any) -> None:
    try:
        sw81.reset_online_cache(model)
    except Exception:
        pass
    if hasattr(model, "reset_mcqm_state"):
        model.reset_mcqm_state()


def reset_mcqm_memory(model: Any) -> None:
    if hasattr(model, "reset_mcqm_state"):
        model.reset_mcqm_state()


def reset_r8_cache(model: Any) -> None:
    try:
        sw13a.reset_model_cache(model)
    except Exception:
        pass


def compare_tensors(a: torch.Tensor | None, b: torch.Tensor | None, *, atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    if a is None or b is None:
        return {
            "present_a": a is not None,
            "present_b": b is not None,
            "shape_equal": False,
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "allclose": False,
            "shape_a": list(a.shape) if a is not None else None,
            "shape_b": list(b.shape) if b is not None else None,
        }
    a_cpu = a.detach().float().cpu()
    b_cpu = b.detach().float().cpu()
    if a_cpu.shape != b_cpu.shape:
        return {
            "present_a": True,
            "present_b": True,
            "shape_equal": False,
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "allclose": False,
            "shape_a": list(a_cpu.shape),
            "shape_b": list(b_cpu.shape),
        }
    diff = (a_cpu - b_cpu).abs()
    return {
        "present_a": True,
        "present_b": True,
        "shape_equal": True,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "allclose": bool(torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)),
        "shape_a": list(a_cpu.shape),
        "shape_b": list(b_cpu.shape),
    }


def current_query_occ_from_cache(model: Any, flags_value: int | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    cache = getattr(model, "latest_mcqm_train_cache", {})
    cls = cache.get("curr_query_cls", None)
    pts = cache.get("curr_query_pos", None)
    flags = cache.get("query_source", None)
    if cls is None or pts is None:
        return None, None
    if flags_value is None or flags is None:
        return cls, pts
    query_count = int(cls.shape[1])
    if isinstance(flags, (list, tuple)):
        raw_flags = flags[0]
    elif torch.is_tensor(flags):
        if flags.ndim == 2:
            if flags.shape[0] != cls.shape[0]:
                raise RuntimeError(
                    "latest_mcqm_train_cache.query_source batch mismatch: "
                    f"expected B={cls.shape[0]}, got {flags.shape[0]}"
                )
            raw_flags = flags[0]
        elif flags.ndim in (0, 1):
            raw_flags = flags
        else:
            raise RuntimeError(
                "latest_mcqm_train_cache.query_source must have shape [B,Q], [Q], "
                f"or scalar default; got {tuple(flags.shape)}"
            )
    else:
        raise RuntimeError("latest_mcqm_train_cache.query_source must be tensor-like")
    flat_flags = raw_flags.detach().reshape(-1).to(dtype=torch.long, device=cls.device)
    if flat_flags.numel() == 1 and int(flat_flags.item()) == 0:
        flat_flags = torch.zeros((query_count,), device=cls.device, dtype=torch.long)
    if flat_flags.numel() != query_count:
        raise RuntimeError(
            "latest_mcqm_train_cache.query_source shape mismatch: "
            f"expected {query_count}, got {flat_flags.numel()}"
        )
    mask = flat_flags == int(flags_value)
    if not bool(mask.any().item()):
        return None, None
    return cls[:, mask], pts[:, mask]


def occ_from_subset(model: Any, cls_scores: torch.Tensor | None, refine_pts: torch.Tensor | None) -> torch.Tensor | None:
    if cls_scores is None or refine_pts is None:
        return None
    pred_dict = {"cls_scores": cls_scores, "refine_pts": refine_pts}
    out = model.pts_bbox_head.get_occ(pred_dict)[0]
    return out.detach().cpu()


def summarize_provenance(model: Any, gt_h: torch.Tensor, horizon_s: int) -> dict[str, Any]:
    all_cls, all_pts = current_query_occ_from_cache(model, None)
    mem_cls, mem_pts = current_query_occ_from_cache(model, 1)
    nat_cls, nat_pts = current_query_occ_from_cache(model, 0)
    all_occ = occ_from_subset(model, all_cls, all_pts)
    mem_occ = occ_from_subset(model, mem_cls, mem_pts)
    nat_occ = occ_from_subset(model, nat_cls, nat_pts)
    out = {"horizon_s": horizon_s}
    if all_occ is None:
        return out
    gt_occ = gt_h != sw2.EMPTY_IDX
    all_occ_mask = all_occ != sw2.EMPTY_IDX
    out["final_active_voxels"] = int(all_occ_mask.sum().item())
    if mem_occ is not None:
        mem_mask = mem_occ != sw2.EMPTY_IDX
        out["memory_active_voxels"] = int(mem_mask.sum().item())
        out["memory_correct_voxels"] = int((mem_mask & gt_occ).sum().item())
        out["memory_fp_voxels"] = int((mem_mask & (~gt_occ)).sum().item())
    if nat_occ is not None:
        nat_mask = nat_occ != sw2.EMPTY_IDX
        out["native_active_voxels"] = int(nat_mask.sum().item())
    cache = getattr(model, "latest_mcqm_train_cache", {})
    flags = cache.get("query_source", None)
    if flags is not None:
        out["memory_query_final_count"] = int((flags[0] == 1).sum().item())
        out["native_query_final_count"] = int((flags[0] == 0).sum().item())
    return out


def extract_final_pred_tensors(query_holder: dict[str, Any]) -> dict[str, torch.Tensor]:
    pred_dict = sw13a.sw4_inst.extract_pred_dict_for_horizon(query_holder, 0)
    return {
        "cls_scores": pred_dict["cls_scores"].detach().cpu(),
        "refine_pts": pred_dict["refine_pts"].detach().cpu(),
    }


def extract_mcqm_parity_capture(model: Any) -> dict[str, torch.Tensor] | None:
    debug = getattr(model, "latest_mcqm_debug", {})
    capture = debug.get("parity_capture", None) if isinstance(debug, dict) else None
    if not isinstance(capture, dict):
        return None
    out = {}
    for key, value in capture.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu()
    return out


def extract_meta_subset_from_batch(batch: dict[str, Any]) -> dict[str, Any]:
    meta = unwrap_meta(batch)
    return {
        "filename": meta.get("filename"),
        "img_timestamp": meta.get("img_timestamp"),
        "ego2global": meta.get("ego2global"),
        "ego2lidar": meta.get("ego2lidar"),
        "lidar2img": meta.get("lidar2img"),
        "scene_token": meta.get("scene_token"),
        "timestamp": meta.get("timestamp"),
        "sample_idx": meta.get("sample_idx"),
    }


def build_target_metric_map(result: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["sample_index"]), int(row["horizon_s"])): row
        for row in result["rows"]
    }


def preflight_oracle_e2e(
    dataset: Any,
    sample_start: int,
    sample_end: int,
) -> dict[str, Any]:
    candidate_targets = [40, 43, 47, 49]
    valid_targets = [
        idx
        for idx in candidate_targets
        if sample_start < idx <= sample_end
        and has_valid_previous_frame(dataset, idx)
    ]
    if len(valid_targets) < 3:
        raise RuntimeError(
            "Oracle E2E preflight failed: "
            f"window=[{sample_start},{sample_end}], "
            f"valid_targets={valid_targets}"
        )
    return {
        "status": "passed",
        "sample_start": int(sample_start),
        "sample_end": int(sample_end),
        "candidate_targets": candidate_targets,
        "valid_targets": valid_targets,
        "variant_count": 5,
        "variants": ["H0", "H1", "H2_A10", "H2_A25", "H3"],
        "runtime_variants": ["P0", "H0", "H1", "H2_A10", "H2_A25", "H3"],
        "k_values": [1, 4, 8, 16],
        "forced_job_count": int(len(valid_targets) * 5 * 4),
    }


def build_positive_nonconflicting_forced_pairs(
    oracle_rows: list[dict[str, Any]],
    requested_k: int,
) -> dict[str, Any]:
    valid_rows = [
        row
        for row in oracle_rows
        if math.isfinite(
            float(row.get("utility_query", float("nan")))
        )
    ]
    ranked = sorted(
        valid_rows,
        key=lambda item: float(item.get("utility_query", float("-inf"))),
        reverse=True,
    )
    forced_pairs: list[dict[str, int]] = []
    used_native: set[int] = set()
    used_memory: set[int] = set()
    positive_nonconflicting_pair_count = 0
    for item in ranked:
        utility_query = float(item.get("utility_query", float("-inf")))
        if not math.isfinite(utility_query):
            continue
        if utility_query <= 0.0:
            break
        native_idx = int(item["native_slot_id"])
        memory_query_id = int(item["memory_query_id"])
        if native_idx in used_native or memory_query_id in used_memory:
            continue
        positive_nonconflicting_pair_count += 1
        used_native.add(native_idx)
        used_memory.add(memory_query_id)
        if len(forced_pairs) < requested_k:
            forced_pairs.append(
                {
                    "native_slot_id": native_idx,
                    "memory_query_id": memory_query_id,
                }
            )
    return {
        "forced_pairs": forced_pairs,
        "positive_nonconflicting_pair_count": int(
            positive_nonconflicting_pair_count
        ),
        "requested_pair_count": int(
            min(requested_k, positive_nonconflicting_pair_count)
        ),
        "applied_target_k": int(requested_k),
    }


def has_valid_previous_frame(
    dataset: Any,
    sample_index: int,
) -> bool:
    if sample_index <= 0:
        return False
    current = dataset.data_infos[sample_index]
    previous = dataset.data_infos[sample_index - 1]
    if current.get("scene_token") != previous.get("scene_token"):
        return False
    return True


def find_first_valid_q2f_pair(dataset: Any, sample_start: int, sample_end: int) -> tuple[int, int]:
    for sample_index in range(max(sample_start + 1, 1), sample_end + 1):
        if has_valid_previous_frame(dataset, sample_index):
            return sample_index - 1, sample_index
    raise RuntimeError(
        f"no valid consecutive same-scene pair found inside [{sample_start}, {sample_end}]"
    )


def materialize_mcqm_q2f_modules(model: Any, batch_cuda: dict[str, Any]) -> dict[str, Any]:
    img = batch_img_tensor(batch_cuda)
    if torch.is_tensor(img) and img.ndim == 4:
        img = img.unsqueeze(0)
    img_metas = [unwrap_meta(batch_cuda)]
    topology = model._mcqm_q2f_backbone_topology()
    hook_capture: dict[str, Any] = {}
    module_path, module = model._mcqm_q2f_resolve_shallow_module()

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (list, tuple)) else output
        hook_capture["shape"] = list(tensor.shape)
        hook_capture["channels"] = int(tensor.shape[1])
        hook_capture["spatial_shape"] = [int(tensor.shape[2]), int(tensor.shape[3])]
        model._mcqm_q2f_ensure_modules(tensor)

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        img_feats = model.extract_feat(img, img_metas)
    handle.remove()
    camera_count = int(model._mcqm_q2f_camera_count(img_metas))
    camera_order = model._mcqm_q2f_camera_order(img_metas, camera_count)
    return {
        "backbone_type": topology["backbone_type"],
        "neck_type": topology["neck_type"],
        "backbone_stage_paths": topology["backbone_stage_paths"],
        "backbone_out_indices": topology["backbone_out_indices"],
        "selected_shallow_module_path": module_path,
        "selected_shallow_shape": list(hook_capture.get("shape", [])),
        "selected_shallow_channels": int(hook_capture.get("channels", 0)),
        "selected_shallow_spatial_shape": list(hook_capture.get("spatial_shape", [])),
        "downstream_backbone_stage_paths": topology["downstream_backbone_stage_paths"],
        "fpn_level_shapes": [list(level.shape) for level in img_feats],
        "fpn_level_channels": [int(level.shape[2]) for level in img_feats],
        "query_dim": int(getattr(model, "out_dim", 0)),
        "camera_count": int(camera_count),
        "camera_order": camera_order,
    }


def build_q2f_shallow_topology_audit(model: Any, batch_cuda: dict[str, Any]) -> dict[str, Any]:
    img = batch_img_tensor(batch_cuda)
    if torch.is_tensor(img) and img.ndim == 4:
        img = img.unsqueeze(0)
    img_metas = [unwrap_meta(batch_cuda)]
    backbone = model.img_backbone
    neck = model.img_neck
    stage_paths = [f"img_backbone.{name}" for name in list(getattr(backbone, "res_layers", []))]
    stage_shapes: dict[str, list[int]] = {}
    neck_input_shapes: list[list[int]] = []
    neck_output_shapes: list[list[int]] = []
    handles = []
    for stage_name in list(getattr(backbone, "res_layers", [])):
        module = getattr(backbone, stage_name)
        def _stage_hook(_m, _i, o, stage_name=stage_name):
            tensor = o[0] if isinstance(o, (list, tuple)) else o
            stage_shapes[f"img_backbone.{stage_name}"] = list(tensor.shape)
        handles.append(
            module.register_forward_hook(_stage_hook)
        )
    if neck is not None:
        def _neck_hook(_m, i, o):
            neck_input_shapes.extend([list(x.shape) for x in i[0]])
            neck_output_shapes.extend([list(x.shape) for x in o])
        handles.append(
            neck.register_forward_hook(_neck_hook)
        )
    with torch.no_grad():
        fpn_levels = model.extract_feat(img.cuda(), img_metas)
    for handle in handles:
        handle.remove()
    topology = model._mcqm_q2f_backbone_topology()
    selected_path = str(topology["selected_shallow_module_path"])
    selected_shape = list(stage_shapes.get(selected_path, []))
    total_views = selected_shape[0] if selected_shape else 0
    batch_size = int(img.shape[0])
    camera_count = int(model._mcqm_q2f_camera_count(img_metas))
    topology_valid = bool(
        topology.get("has_downstream_backbone_stages", False)
        and topology.get("camera_processing_independent_before_hook", False)
        and selected_shape
        and len(selected_shape) == 4
        and total_views == batch_size * (len(img_metas[0].get("filename", [])) or camera_count)
    )
    return {
        "backbone_type": type(backbone).__name__,
        "neck_type": type(neck).__name__ if neck is not None else "",
        "backbone_stage_paths": stage_paths,
        "backbone_stage_output_shapes": [stage_shapes.get(path, []) for path in stage_paths],
        "backbone_out_indices": list(getattr(backbone, "out_indices", [])),
        "neck_input_shapes": neck_input_shapes,
        "neck_output_shapes": [list(level.shape) for level in fpn_levels] if not neck_output_shapes else neck_output_shapes,
        "selected_shallow_module_path": selected_path,
        "selected_shallow_shape": selected_shape,
        "downstream_backbone_stage_paths": list(topology.get("downstream_backbone_stage_paths", [])),
        "has_downstream_backbone_stages": bool(topology.get("has_downstream_backbone_stages", False)),
        "camera_processing_independent_before_hook": bool(topology.get("camera_processing_independent_before_hook", False)),
        "topology_valid_for_strict_algorithm": bool(topology_valid),
    }


def direct_model_inputs_from_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    img = batch_img_tensor(batch)
    if torch.is_tensor(img) and img.ndim == 4:
        img = img.unsqueeze(0)
    return img, [unwrap_meta(batch)]


def normalized_model_batch_kwargs(batch: dict[str, Any]) -> dict[str, Any]:
    img, img_metas = direct_model_inputs_from_batch(batch)
    out = dict(batch)
    out["img"] = img
    out["img_metas"] = img_metas
    for key in ("temporal_semantics", "temporal_rays", "temporal_ego_states", "temporal2ego", "temporal_ego2global"):
        value = out.get(key, None)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            out[key] = value[0]
    for key in ("temporal_trajs", "rays"):
        value = out.get(key, None)
        if isinstance(value, list) and len(value) == 1 and torch.is_tensor(value[0]):
            out[key] = value[0]
    for key, value in list(out.items()):
        if key in {"img", "img_metas"}:
            continue
        if isinstance(value, list) and len(value) == 1 and torch.is_tensor(value[0]):
            out[key] = value[0]
    return out


def overwrite_failed_camera_images(
    batch: dict[str, Any],
    failed_camera_names: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    out = copy.deepcopy(batch)
    meta = unwrap_meta(out)
    camera_groups = sw5.resolve_camera_groups([str(name) for name in meta.get("filename", [])])
    affected_indices: list[int] = []
    for name in failed_camera_names:
        affected_indices.extend(int(idx) for idx in camera_groups.get(str(name), []))
    if not affected_indices:
        raise RuntimeError(f"failed to find image indices for cameras: {failed_camera_names}")
    img = batch_img_tensor(out)
    if img.ndim == 4:
        img = img.unsqueeze(0)
    generator = torch.Generator(device=img.device)
    generator.manual_seed(int(seed))
    replacement = torch.randint(
        low=0,
        high=256,
        size=img[:, affected_indices].shape,
        generator=generator,
        device=img.device,
        dtype=img.dtype,
    )
    img[:, affected_indices] = replacement
    return out


def clone_mcqm_memory_seed(model: Any) -> dict[int, Any]:
    manager = getattr(model, "mcqm_memory_manager", None)
    state_by_batch = getattr(manager, "_state_by_batch", None)
    if not isinstance(state_by_batch, dict):
        return {}
    return clone_mcqm_memory_seed_dict(
        {int(batch_index): state for batch_index, state in state_by_batch.items()}
    )


def clone_mcqm_memory_seed_dict(seed: dict[int, Any]) -> dict[int, Any]:
    if not isinstance(seed, dict):
        raise TypeError("memory seed must be a dict")
    cloned = {}
    for batch_index, state in seed.items():
        if not isinstance(batch_index, int):
            raise TypeError("memory seed batch index must be int")
        if not hasattr(state, "__dict__"):
            raise TypeError(f"memory seed entry {batch_index} is not a state object")
        state_copy = copy.copy(state)
        for key, value in state.__dict__.items():
            if torch.is_tensor(value):
                setattr(state_copy, key, value.detach().clone())
            else:
                setattr(state_copy, key, copy.deepcopy(value))
        cloned[int(batch_index)] = state_copy
    return cloned


def memory_seed_audit(seed: dict[int, Any]) -> dict[str, Any]:
    cloned = clone_mcqm_memory_seed_dict(seed)
    rows = []
    all_native = True
    query_count = 0
    for batch_index, state in sorted(cloned.items()):
        query_count += int(getattr(state, "query_feat").shape[0])
        query_source = getattr(state, "query_source", None)
        if torch.is_tensor(query_source):
            all_native = all_native and bool((query_source == 0).all().item())
        rows.append({
            "batch_index": int(batch_index),
            "scene_token": str(getattr(state, "scene_token", "")),
            "query_feat_sha256": tensor_sha256(state.query_feat),
            "refine_points_sha256": tensor_sha256(state.refine_points),
            "cls_logits_sha256": tensor_sha256(state.cls_logits),
            "dominant_class_sha256": tensor_sha256(state.dominant_class),
            "confidence_sha256": tensor_sha256(state.confidence),
            "query_ids_sha256": tensor_sha256(state.query_ids),
            "query_source_sha256": tensor_sha256(state.query_source),
            "memory_age_sha256": tensor_sha256(state.memory_age),
            "was_memory_injected_sha256": tensor_sha256(state.was_memory_injected),
            "parent_memory_query_id_sha256": tensor_sha256(state.parent_memory_query_id),
            "source_native_query_id_sha256": tensor_sha256(state.source_native_query_id),
            "timestamp_sha256": tensor_sha256(state.timestamp),
            "ego2global_sha256": tensor_sha256(state.ego2global),
        })
    return {
        "seed_memory_state_sha256": object_sha256(rows),
        "seed_query_count": int(query_count),
        "seed_source_all_native": bool(all_native),
        "seed_state_rows": rows,
    }


def restore_mcqm_memory_seed(model: Any, seed: dict[int, Any]) -> dict[str, Any]:
    manager = getattr(model, "mcqm_memory_manager", None)
    if manager is None:
        raise RuntimeError("MCQM memory manager unavailable for seed restore")
    device = next(model.parameters()).device
    cloned = clone_mcqm_memory_seed_dict(seed)
    for batch_index, state in cloned.items():
        if not isinstance(batch_index, int):
            raise TypeError("memory seed batch indices must be integers")
        for key, value in state.__dict__.items():
            if torch.is_tensor(value) and value.device != device:
                state.__dict__[key] = value.to(device=device)
    manager.reset()
    manager._state_by_batch = cloned
    return memory_seed_audit(cloned)


def capture_p0_memory_seed_for_target(
    checkpoint_path: str | Path,
    warmup_start: int,
    target_index: int,
) -> dict[str, Any]:
    if warmup_start >= target_index:
        raise ValueError("warmup capture requires at least one frame before target")
    mcqm_cfg = mcqm_v2_variant_cfg("P0", oracle_mode="disabled")
    _, dataset, model, _ = build_runtime_strict(
        train=False,
        mcqm_cfg=mcqm_cfg,
        checkpoint_path=checkpoint_path,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    reset_runtime_state(model)
    reset_mcqm_memory(model)
    prev_scene = None
    for idx in range(warmup_start, target_index):
        scene_token = dataset.data_infos[idx].get("scene_token", "")
        if prev_scene is not None and scene_token != prev_scene:
            reset_runtime_state(model)
            reset_mcqm_memory(model)
        prev_scene = scene_token
        _, batch = extract_inputs(dataset, idx)
        batch = enrich_batch_scene_meta(batch, dataset, idx)
        batch, _ = apply_perturbation_with_manifest_to_batch(batch, "A10_drop_front_triplet")
        inputs = sw2.move_to_cuda(batch)
        with torch.inference_mode():
            _ = model(return_loss=False, rescale=True, **inputs)
    seed_state = clone_mcqm_memory_seed(model)
    return {
        "state": seed_state,
        "audit": memory_seed_audit(seed_state),
    }


def collect_img_metas_protocol_diff(sample_start: int, sample_end: int) -> dict[str, Any]:
    _, dataset, _, _ = build_runtime(train=False, mcqm_cfg=None, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0}, checkpoint_path=CHECKPOINT_PATH)
    rows = []
    for idx in range(sample_start, sample_end + 1):
        sample_raw, batch = extract_inputs(dataset, idx)
        _ = sample_raw
        batch = enrich_batch_scene_meta(batch, dataset, idx)
        _, batch_deg_b1s = extract_inputs(dataset, idx)
        batch_deg_b1s = enrich_batch_scene_meta(batch_deg_b1s, dataset, idx)
        batch_deg_b1s, _ = apply_perturbation_with_manifest_to_batch(batch_deg_b1s, "A10_drop_front_triplet")
        batch_deg_b2s = sw2.extract_sample_batch(dataset, idx, collate_fn)[1]
        batch_deg_b2s = enrich_batch_scene_meta(batch_deg_b2s, dataset, idx)
        batch_deg_b2s, _ = apply_perturbation_with_manifest_to_batch(batch_deg_b2s, "A10_drop_front_triplet")
        batch_deg_n1 = sw2.extract_sample_batch(dataset, idx, collate_fn)[1]
        batch_deg_n1 = enrich_batch_scene_meta(batch_deg_n1, dataset, idx)
        batch_deg_n1, _ = apply_perturbation_with_manifest_to_batch(batch_deg_n1, "A10_drop_front_triplet")
        batch_deg_a2 = sw2.extract_sample_batch(dataset, idx, collate_fn)[1]
        batch_deg_a2 = enrich_batch_scene_meta(batch_deg_a2, dataset, idx)
        batch_deg_a2, _ = apply_perturbation_with_manifest_to_batch(batch_deg_a2, "A10_drop_front_triplet")
        b1s_meta = extract_meta_subset_from_batch(batch_deg_b1s)
        b2s_meta = extract_meta_subset_from_batch(batch_deg_b2s)
        n1_meta = extract_meta_subset_from_batch(batch_deg_n1)
        a2_meta = extract_meta_subset_from_batch(batch_deg_a2)
        rows.append({
            "sample_index": idx,
            "B1S_vs_B2S": diff_meta_subset(b1s_meta, b2s_meta),
            "B1S_vs_N1": diff_meta_subset(b1s_meta, n1_meta),
            "N1_vs_A2": diff_meta_subset(n1_meta, a2_meta),
        })
    benign_only = all(
        (not row["B1S_vs_B2S"]["protocol_mismatch"])
        and (not row["B1S_vs_N1"]["protocol_mismatch"])
        and (not row["N1_vs_A2"]["protocol_mismatch"])
        for row in rows
    )
    out = {"rows": rows, "all_benign": benign_only}
    write_json(V2_ARTIFACTS_DIR / "img_metas_protocol_diff.json", out)
    return out


def capture_v2_clean_teacher_states(
    checkpoint_path: str | Path,
    sample_start: int,
    sample_end: int,
) -> dict[int, dict[str, Any]]:
    cfg = mcqm_v2_variant_cfg("P0", oracle_mode="disabled")
    _, dataset, model, _ = build_runtime_strict(
        train=False,
        mcqm_cfg=cfg,
        checkpoint_path=checkpoint_path,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, query_holder)
    reset_runtime_state(model)
    reset_mcqm_memory(model)
    teacher_by_sample = {}
    try:
        prev_scene = None
        for idx in range(sample_start, sample_end + 1):
            scene_token = dataset.data_infos[idx].get("scene_token", "")
            if prev_scene is not None and scene_token != prev_scene:
                reset_runtime_state(model)
                reset_mcqm_memory(model)
            prev_scene = scene_token
            sample_raw, batch = extract_inputs(dataset, idx)
            _ = sample_raw
            batch = enrich_batch_scene_meta(batch, dataset, idx)
            batch, _ = apply_perturbation_with_manifest_to_batch(batch, "A0_clean")
            inputs = sw2.move_to_cuda(batch)
            with torch.inference_mode():
                _ = model(return_loss=False, rescale=True, **inputs)
            capture = extract_mcqm_parity_capture(model) or {}
            teacher_by_sample[idx] = {
                "teacher_current_query_feat": capture.get("layer1_query_feat"),
                "teacher_current_query_points": capture.get("layer1_query_points"),
                "teacher_current_cls_score": capture.get("callback_input_cls_score"),
            }
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]
    return teacher_by_sample


def teacher_state_sha256(teacher_state: dict[str, Any]) -> str:
    payload = {}
    for key, value in teacher_state.items():
        if torch.is_tensor(value):
            payload[key] = tensor_sha256(value)
        else:
            payload[key] = object_sha256(value)
    return object_sha256(payload)


def oracle_rows_sha256(rows: list[dict[str, Any]]) -> str:
    normalized_rows = []
    for row in rows:
        normalized = {}
        for key in sorted(row.keys()):
            value = row[key]
            if torch.is_tensor(value):
                normalized[key] = tensor_sha256(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    normalized_rows.sort(
        key=lambda item: (
            int(item.get("sample_index", -1)),
            int(item.get("native_slot_id", -1)),
            int(item.get("memory_query_id", -1)),
            str(item.get("injection_mode", "")),
        )
    )
    return object_sha256(normalized_rows)


def load_existing_forward_artifact() -> dict[str, Any]:
    path = V2_ARTIFACTS_DIR / "interface_forward_ablation.json"
    if not file_exists_and_valid_json(path):
        raise RuntimeError(f"missing valid forward artifact: {path}")
    return read_json(path)


def load_existing_oracle_alignment_artifact() -> dict[str, Any]:
    path = V2_ARTIFACTS_DIR / "oracle_query_alignment.json"
    if not file_exists_and_valid_json(path):
        raise RuntimeError(f"missing valid oracle alignment artifact: {path}")
    return read_json(path)


def load_existing_init_manifest() -> dict[str, Any]:
    path = V2_ARTIFACTS_DIR / "mcqm_v2_init_checkpoint_manifest.json"
    if not file_exists_and_valid_json(path):
        return create_mcqm_v2_init_checkpoint()
    return read_json(path)


def current_protocol_hash(
    *,
    checkpoint_sha256: str,
    architecture_signature_sha256: str,
    runtime_config_hash: str,
    sample_start: int,
    sample_end: int,
    target_samples: list[int],
    k_values: list[int],
    teacher_hash: str,
    p0_seed_hash: str,
    seeded_oracle_rows_sha256: str,
) -> str:
    return object_sha256(
        {
            "protocol_version": ORACLE_E2E_PROTOCOL_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "architecture_signature_sha256": architecture_signature_sha256,
            "runtime_config_hash": runtime_config_hash,
            "sample_start": int(sample_start),
            "sample_end": int(sample_end),
            "target_samples": [int(v) for v in target_samples],
            "k_values": [int(v) for v in k_values],
            "teacher_hash": str(teacher_hash),
            "p0_seed_hash": str(p0_seed_hash),
            "seeded_oracle_rows_sha256": str(seeded_oracle_rows_sha256),
            "forced_selection_policy": ORACLE_E2E_FORCED_SELECTION_POLICY,
            "decision_k": int(ORACLE_E2E_DECISION_K),
        }
    )


def load_partial_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    valid_lines: list[str] = []
    truncated_last_line_ignored = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
            valid_lines.append(line)
        except json.JSONDecodeError as exc:
            is_last_line = index == len(lines) - 1
            if is_last_line:
                truncated_last_line_ignored = True
                break
            raise RuntimeError(
                "corrupted Oracle E2E partial JSONL "
                f"at line {index + 1}"
            ) from exc
    if truncated_last_line_ignored:
        rewritten = "".join(f"{line}\n" for line in valid_lines)
        with path.open("w", encoding="utf-8") as f:
            f.write(rewritten)
            f.flush()
            os.fsync(f.fileno())
    return rows


def load_partial_e2e_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_partial_jsonl_rows(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = str(row["job_id"])
        protocol_hash = str(row["protocol_hash"])
        prev = out.get(job_id)
        if prev is not None and str(prev["protocol_hash"]) != protocol_hash:
            raise RuntimeError(
                "partial Oracle E2E protocol hash mismatch for job "
                f"{job_id}: {prev['protocol_hash']} vs {protocol_hash}"
            )
        out[job_id] = row
    return out


def aggregate_oracle_e2e_partial(
    partial_rows_by_job: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        partial_rows_by_job[job_id]
        for job_id in sorted(partial_rows_by_job.keys())
    ]


def build_runtime_strict_timed(
    *,
    train: bool,
    variant: str,
    oracle_mode: str,
    checkpoint_path: str | Path,
) -> tuple[MCQMV2EvalRuntime, float, float]:
    mcqm_cfg = mcqm_v2_variant_cfg(variant, oracle_mode=oracle_mode)
    build_start = time.perf_counter()
    cfg, dataset, model = build_runtime_unloaded(
        train=train,
        mcqm_cfg=mcqm_cfg,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    build_seconds = time.perf_counter() - build_start
    load_start = time.perf_counter()
    checkpoint_audit = load_model_state_strict(
        model,
        checkpoint_path,
        allowed_missing_keys=set(),
    )
    checkpoint_meta = checkpoint_audit.get("checkpoint_meta", {})
    if (
        isinstance(checkpoint_meta, dict)
        and "architecture_signature" in checkpoint_meta
        and mcqm_cfg is not None
    ):
        expected = checkpoint_meta["architecture_signature"]
        current = build_mcqm_v2_architecture_signature(
            model,
            mcqm_cfg,
            checkpoint_meta.get(
                "base_checkpoint_sha256",
                file_sha256(resolve_checkpoint_path(CHECKPOINT_PATH)),
            ),
        )
        expected_structural = (
            expected["structural_signature"]
            if isinstance(expected, dict) and "structural_signature" in expected
            else expected
        )
        current_structural = current["structural_signature"]
        if object_sha256(expected_structural) != object_sha256(current_structural):
            raise RuntimeError("MCQM V2 structural architecture signature mismatch")
    model = model.cuda()
    if hasattr(model, "set_epoch"):
        meta = checkpoint_audit.get("checkpoint_meta", {})
        if isinstance(meta, dict) and meta.get("epoch") is not None:
            model.set_epoch(int(meta["epoch"]))
    model.train() if train else model.eval()
    load_seconds = time.perf_counter() - load_start
    runtime_meta = {
        **checkpoint_file_info(checkpoint_path),
        "strict_state_dict_audit": {
            "expected_new_v2_keys": [],
            "unexpected_missing_core_keys": [],
            "unexpected_checkpoint_keys": [],
            "shape_mismatch_keys": [],
        },
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_parameter_key_hash": object_sha256(sorted(model.state_dict().keys())),
        "core_model_state_sha256": model_state_sha256(model),
    }
    architecture_signature_sha256 = ""
    if isinstance(checkpoint_meta, dict):
        architecture_signature_sha256 = str(
            checkpoint_meta.get("architecture_signature_sha256", "")
        )
    runtime = MCQMV2EvalRuntime(
        variant=variant,
        oracle_mode=oracle_mode,
        cfg=cfg,
        dataset=dataset,
        model=model,
        runtime_meta=runtime_meta,
        checkpoint_sha256=str(runtime_meta.get("checkpoint_sha256", "")),
        architecture_signature_sha256=architecture_signature_sha256,
        runtime_config_hash=object_sha256(mcqm_cfg),
    )
    return runtime, build_seconds, load_seconds


def clone_cached_sample(raw_sample: Any, batch: Any) -> tuple[Any, Any]:
    return copy.deepcopy(raw_sample), copy.deepcopy(batch)


def get_cpu_sample_batch(
    runtime: MCQMV2EvalRuntime,
    idx: int,
    perturbation: str,
    cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]],
) -> tuple[Any, Any, float]:
    key = (int(idx), str(perturbation))
    start = time.perf_counter()
    if key in cpu_batch_cache:
        raw_sample, batch = cpu_batch_cache[key]
        return (*clone_cached_sample(raw_sample, batch), time.perf_counter() - start)
    raw_sample, batch = extract_inputs(runtime.dataset, idx)
    batch = enrich_batch_scene_meta(batch, runtime.dataset, idx)
    batch, _ = apply_perturbation_with_manifest_to_batch(batch, perturbation)
    cpu_batch_cache[key] = (copy.deepcopy(raw_sample), copy.deepcopy(batch))
    return (*clone_cached_sample(raw_sample, batch), time.perf_counter() - start)


def timed_move_to_cuda(batch: Any) -> tuple[Any, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    inputs = sw2.move_to_cuda(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return inputs, time.perf_counter() - start


def target_eval_with_runtime(
    runtime: MCQMV2EvalRuntime,
    *,
    target_index: int,
    seed_memory_state: dict[int, Any],
    oracle_mode: str,
    teacher_state: dict[str, Any] | None,
    forced_pairs: list[dict[str, int]],
    artifact_label: str,
    cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]],
) -> dict[str, Any]:
    apply_mcqm_v2_parameter_freeze(runtime.model)
    reset_runtime_state(runtime.model)
    reset_mcqm_memory(runtime.model)
    seed_audit = restore_mcqm_memory_seed(runtime.model, seed_memory_state)
    sample_raw, batch, data_prepare_seconds = get_cpu_sample_batch(
        runtime,
        target_index,
        "A10_drop_front_triplet",
        cpu_batch_cache,
    )
    runtime.model.mcqm_v2_runtime_oracle = {
        "eval_only": True,
        "artifact_label": artifact_label,
        "forced_pairs": forced_pairs,
        **(teacher_state or {}),
    }
    inputs, cuda_transfer_seconds = timed_move_to_cuda(batch)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        out = runtime.model(return_loss=False, rescale=True, **inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_start
    metric_start = time.perf_counter()
    mcqm_debug = getattr(runtime.model, "latest_mcqm_debug", {})
    row0 = (mcqm_debug.get("debug_rows", [{}]) or [{}])[0]
    selection_signature_rows = [
        {"sample_index": target_index, **row}
        for row in mcqm_debug.get("selection_signature_rows", [])
    ]
    oracle_rows = [
        {"sample_index": target_index, **row}
        for row in mcqm_debug.get("oracle_query_rows", [])
    ]
    pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sw2.unwrap(sample_raw), out)
    sectors = get_sector_masks_cached()
    rows = []
    provenance_rows = []
    capture_rows = []
    for horizon_s in CORE_HORIZONS:
        pred_h = pred_temporal[horizon_s].cpu()
        gt_h = gt_temporal[horizon_s].cpu()
        gt0 = gt_temporal[0].cpu()
        row = sw12b.build_eval_row(
            pred_h,
            gt_h,
            gt0,
            "A10_drop_front_triplet",
            horizon_s,
            sectors,
            baseline_pred=None,
        )
        row.update(
            {
                "sample_index": int(target_index),
                "variant": runtime.variant,
                "horizon_s": int(horizon_s),
            }
        )
        rows.append(row)
        if horizon_s == 0:
            capture_rows.append(
                {
                    "sample_index": int(target_index),
                    "variant": runtime.variant,
                    "final_occ_dense": pred_h.detach().clone(),
                }
            )
        if horizon_s == 0:
            provenance_rows.append(
                {
                    "sample_index": int(target_index),
                    "variant": runtime.variant,
                    **summarize_provenance(runtime.model, gt_h, horizon_s),
                }
            )
    metric_seconds = time.perf_counter() - metric_start
    summary = aggregate_rows(rows)
    summary["variant"] = runtime.variant
    summary["debug"] = aggregate_rows(
        [{"sample_index": int(target_index), "variant": runtime.variant, **row0}]
    )
    summary["provenance"] = aggregate_rows(provenance_rows) if provenance_rows else {}
    summary["target_index"] = int(target_index)
    summary["seed_audit"] = seed_audit
    summary["timings"] = {
        "data_prepare_seconds": float(data_prepare_seconds),
        "cuda_transfer_seconds": float(cuda_transfer_seconds),
        "forward_seconds": float(forward_seconds),
        "metric_seconds": float(metric_seconds),
    }
    return {
        "summary": summary,
        "rows": rows,
        "capture_rows": capture_rows,
        "debug_rows": [{"sample_index": int(target_index), "variant": runtime.variant, **row0}],
        "selection_signature_rows": selection_signature_rows,
        "oracle_rows": oracle_rows,
        "provenance_rows": provenance_rows,
        "checkpoint": runtime.runtime_meta,
        "timings": summary["timings"],
        "raw_output": out,
        "sample_raw": sample_raw,
    }


def gt_occ_mask(semantic: torch.Tensor) -> torch.Tensor:
    return semantic.long() != int(sw2.EMPTY_IDX)


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if int(den) > 0 else 0.0


def dense_occ_top1_conf_margin(dense_occ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = dense_occ.detach().cpu().float()
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    top1_conf = top2.values[..., 0]
    top1_cls = top2.indices[..., 0].long()
    if top2.values.shape[-1] > 1:
        margin = top2.values[..., 0] - top2.values[..., 1]
    else:
        margin = top2.values[..., 0]
    return top1_conf, top1_cls, margin


def build_active_mask_from_occ_debug(debug: dict[str, Any]) -> torch.Tensor:
    active_voxels = torch.as_tensor(debug["active_voxels"]).detach().cpu().long()
    geometric_mask = torch.as_tensor(debug["geometric_mask"]).detach().cpu().bool()
    mask = torch.zeros_like(geometric_mask, dtype=torch.bool)
    if active_voxels.numel() > 0:
        mask[active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]] = True
    return mask


def build_s0_s1_s2_s3_masks(
    d3_mask: torch.Tensor,
    debug: dict[str, Any],
) -> dict[str, torch.Tensor]:
    d3 = d3_mask.detach().cpu().bool()
    geometric_mask = torch.as_tensor(debug["geometric_mask"]).detach().cpu().bool()
    semantic_active_mask = torch.as_tensor(debug["semantic_active_mask"]).detach().cpu().bool()
    active_mask = build_active_mask_from_occ_debug(debug)
    active_neighbor_mask = torch.from_numpy(
        ndi.binary_dilation(
            active_mask.numpy().astype(bool),
            structure=S3_STRUCTURE,
        )
    ).bool()
    s0 = d3 & (~geometric_mask)
    s1 = d3 & geometric_mask & (~semantic_active_mask)
    s2 = d3 & semantic_active_mask
    s3 = s0 & active_neighbor_mask
    if int(s0.sum().item() + s1.sum().item() + s2.sum().item()) != int(d3.sum().item()):
        raise RuntimeError("S0/S1/S2 partition mismatch in geometry-only audit")
    return {
        "D3": d3,
        "S0": s0,
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "geometric_mask": geometric_mask,
        "semantic_active_mask": semantic_active_mask,
        "active_mask": active_mask,
        "active_neighbor_mask": active_neighbor_mask,
    }


def build_d3_mask_from_occ(
    gt_h: torch.Tensor,
    raw_semantic: torch.Tensor,
    final_semantic: torch.Tensor,
    confidence: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    return (
        gt_occ_mask(gt_h.detach().cpu().long())
        & (~gt_occ_mask(final_semantic.detach().cpu().long()))
        & (~gt_occ_mask(raw_semantic.detach().cpu().long()))
        & (confidence.detach().cpu().float() < float(SW15_DCLASS_CONF_THRESHOLD))
        & (margin.detach().cpu().float() < float(SW15_DCLASS_MARGIN_THRESHOLD))
    )


def _count_mask(mask: torch.Tensor) -> int:
    return int(mask.detach().cpu().bool().sum().item())


def compare_tensor_pair(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    same_shape = tuple(left_cpu.shape) == tuple(right_cpu.shape)
    if not same_shape:
        return {
            "same_shape": False,
            "exact_equal": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    diff = (left_cpu.float() - right_cpu.float()).abs()
    return {
        "same_shape": True,
        "exact_equal": bool(torch.equal(left_cpu, right_cpu)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def compare_unselected_slots(
    before: torch.Tensor,
    after: torch.Tensor,
    selected_indices: list[int],
) -> dict[str, Any]:
    before_cpu = before.detach().cpu()
    after_cpu = after.detach().cpu()
    if tuple(before_cpu.shape) != tuple(after_cpu.shape):
        return {"same_shape": False, "exact_equal": False, "max_abs_diff": None}
    keep = torch.ones(before_cpu.shape[1], dtype=torch.bool)
    if selected_indices:
        keep[torch.as_tensor(selected_indices, dtype=torch.long)] = False
    before_sub = before_cpu[:, keep]
    after_sub = after_cpu[:, keep]
    diff = (before_sub.float() - after_sub.float()).abs()
    return {
        "same_shape": True,
        "exact_equal": bool(torch.equal(before_sub, after_sub)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "unselected_slot_count": int(before_sub.shape[1]),
    }


def capture_clean_teacher_artifacts_with_runtime(
    runtime: MCQMV2EvalRuntime,
    sample_start: int,
    sample_end: int,
    target_samples: list[int],
    cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, float]]:
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(runtime.model, query_holder)
    reset_runtime_state(runtime.model)
    reset_mcqm_memory(runtime.model)
    teacher_by_sample: dict[int, dict[str, Any]] = {}
    teacher_occ_by_sample: dict[int, dict[str, Any]] = {}
    start = time.perf_counter()
    data_prepare_seconds = 0.0
    cuda_transfer_seconds = 0.0
    forward_seconds = 0.0
    try:
        prev_scene = None
        target_set = set(int(v) for v in target_samples)
        for idx in range(sample_start, sample_end + 1):
            scene_token = runtime.dataset.data_infos[idx].get("scene_token", "")
            if prev_scene is not None and scene_token != prev_scene:
                reset_runtime_state(runtime.model)
                reset_mcqm_memory(runtime.model)
            prev_scene = scene_token
            sample_raw, batch, batch_prepare_seconds = get_cpu_sample_batch(
                runtime,
                idx,
                "A0_clean",
                cpu_batch_cache,
            )
            data_prepare_seconds += float(batch_prepare_seconds)
            query_holder.clear()
            inputs, batch_cuda_seconds = timed_move_to_cuda(batch)
            cuda_transfer_seconds += float(batch_cuda_seconds)
            forward_start = time.perf_counter()
            with torch.inference_mode():
                out = runtime.model(return_loss=False, rescale=True, **inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_seconds += float(time.perf_counter() - forward_start)
            if idx not in target_set:
                continue
            capture = extract_mcqm_parity_capture(runtime.model) or {}
            teacher_by_sample[idx] = {
                "teacher_current_query_feat": capture.get("layer1_query_feat"),
                "teacher_current_query_points": capture.get("layer1_query_points"),
                "teacher_current_cls_score": capture.get("callback_input_cls_score"),
            }
            pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sw2.unwrap(sample_raw), out)
            pred_dict = sw12b.sw4_inst.extract_pred_dict_for_horizon(query_holder, 0)
            _, dbg_list = sw12b.sw4_inst.get_occ_debug(
                runtime.model.pts_bbox_head,
                pred_dict,
                capture_dense=True,
            )
            occ_debug = dbg_list[0]
            raw_conf, raw_sem, raw_margin = dense_occ_top1_conf_margin(
                torch.as_tensor(occ_debug["dense_occ_before_padding"]).detach().cpu()
            )
            teacher_occ_by_sample[idx] = {
                "gt_h": gt_temporal[0].detach().cpu().long(),
                "gt0": gt_temporal[0].detach().cpu().long(),
                "teacher_raw_semantic": raw_sem.long(),
                "teacher_final_semantic": torch.as_tensor(pred_temporal[0]).detach().cpu().long(),
                "teacher_confidence": raw_conf.float(),
                "teacher_margin": raw_margin.float(),
                "debug": {
                    key: (
                        value.detach().cpu()
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in occ_debug.items()
                },
            }
    finally:
        runtime.model.forward_backbone = original_forward  # type: ignore[assignment]
    wall_seconds = float(time.perf_counter() - start)
    return teacher_by_sample, teacher_occ_by_sample, {
        "wall_seconds": wall_seconds,
        "data_prepare_seconds": float(data_prepare_seconds),
        "cuda_transfer_seconds": float(cuda_transfer_seconds),
        "forward_seconds": float(forward_seconds),
    }


def target_eval_with_occ_debug(
    runtime: MCQMV2EvalRuntime,
    *,
    target_index: int,
    perturbation_id: str,
    seed_memory_state: dict[int, Any],
    teacher_state: dict[str, Any] | None,
    forced_pairs: list[dict[str, int]],
    artifact_label: str,
    cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]],
) -> dict[str, Any]:
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(runtime.model, query_holder)
    try:
        result = target_eval_with_runtime(
            runtime,
            target_index=target_index,
            seed_memory_state=seed_memory_state,
            oracle_mode=runtime.oracle_mode,
            teacher_state=teacher_state,
            forced_pairs=forced_pairs,
            artifact_label=artifact_label,
            cpu_batch_cache=cpu_batch_cache,
        )
        mcqm_debug = getattr(runtime.model, "latest_mcqm_debug", {})
        parity_capture = extract_mcqm_parity_capture(runtime.model) or {}
        sample_raw = result["sample_raw"]
        pred_dict = sw12b.sw4_inst.extract_pred_dict_for_horizon(query_holder, 0)
        _, dbg_list = sw12b.sw4_inst.get_occ_debug(
            runtime.model.pts_bbox_head,
            pred_dict,
            capture_dense=True,
        )
        occ_debug = {
            key: (
                value.detach().cpu()
                if torch.is_tensor(value)
                else value
            )
            for key, value in dbg_list[0].items()
        }
        pred_temporal_out, gt_temporal, _ = sw2.extract_standard_tensors(
            sw2.unwrap(sample_raw),
            result["raw_output"],
        )
        result["parity_capture"] = parity_capture
        result["occ_debug_h0"] = occ_debug
        result["pred_h0"] = pred_temporal_out[0].detach().cpu().long()
        result["gt_h0"] = gt_temporal[0].detach().cpu().long()
        result["raw_mcqm_debug"] = mcqm_debug
        return result
    finally:
        runtime.model.forward_backbone = original_forward  # type: ignore[assignment]


def capture_p0_memory_seed_with_runtime(
    runtime: MCQMV2EvalRuntime,
    warmup_start: int,
    target_index: int,
    cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    if warmup_start >= target_index:
        raise ValueError("warmup capture requires at least one frame before target")
    reset_runtime_state(runtime.model)
    reset_mcqm_memory(runtime.model)
    prev_scene = None
    start = time.perf_counter()
    data_prepare_seconds = 0.0
    cuda_transfer_seconds = 0.0
    forward_seconds = 0.0
    for idx in range(warmup_start, target_index):
        scene_token = runtime.dataset.data_infos[idx].get("scene_token", "")
        if prev_scene is not None and scene_token != prev_scene:
            reset_runtime_state(runtime.model)
            reset_mcqm_memory(runtime.model)
        prev_scene = scene_token
        _, batch, batch_prepare_seconds = get_cpu_sample_batch(
            runtime,
            idx,
            "A10_drop_front_triplet",
            cpu_batch_cache,
        )
        data_prepare_seconds += float(batch_prepare_seconds)
        inputs, batch_cuda_seconds = timed_move_to_cuda(batch)
        cuda_transfer_seconds += float(batch_cuda_seconds)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            _ = runtime.model(return_loss=False, rescale=True, **inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_seconds += float(time.perf_counter() - forward_start)
    seed_state = clone_mcqm_memory_seed(runtime.model)
    wall_seconds = float(time.perf_counter() - start)
    return {
        "state": seed_state,
        "audit": memory_seed_audit(seed_state),
    }, {
        "wall_seconds": wall_seconds,
        "data_prepare_seconds": float(data_prepare_seconds),
        "cuda_transfer_seconds": float(cuda_transfer_seconds),
        "forward_seconds": float(forward_seconds),
    }


def capture_clean_teacher_states_with_runtime(
    runtime: MCQMV2EvalRuntime,
    sample_start: int,
    sample_end: int,
    target_samples: list[int],
    cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(runtime.model, query_holder)
    reset_runtime_state(runtime.model)
    reset_mcqm_memory(runtime.model)
    teacher_by_sample = {}
    start = time.perf_counter()
    data_prepare_seconds = 0.0
    cuda_transfer_seconds = 0.0
    forward_seconds = 0.0
    try:
        prev_scene = None
        target_set = set(int(v) for v in target_samples)
        for idx in range(sample_start, sample_end + 1):
            scene_token = runtime.dataset.data_infos[idx].get("scene_token", "")
            if prev_scene is not None and scene_token != prev_scene:
                reset_runtime_state(runtime.model)
                reset_mcqm_memory(runtime.model)
            prev_scene = scene_token
            _, batch, batch_prepare_seconds = get_cpu_sample_batch(
                runtime,
                idx,
                "A0_clean",
                cpu_batch_cache,
            )
            data_prepare_seconds += float(batch_prepare_seconds)
            inputs, batch_cuda_seconds = timed_move_to_cuda(batch)
            cuda_transfer_seconds += float(batch_cuda_seconds)
            forward_start = time.perf_counter()
            with torch.inference_mode():
                _ = runtime.model(return_loss=False, rescale=True, **inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_seconds += float(time.perf_counter() - forward_start)
            if idx not in target_set:
                continue
            capture = extract_mcqm_parity_capture(runtime.model) or {}
            teacher_by_sample[idx] = {
                "teacher_current_query_feat": capture.get("layer1_query_feat"),
                "teacher_current_query_points": capture.get("layer1_query_points"),
                "teacher_current_cls_score": capture.get("callback_input_cls_score"),
            }
    finally:
        runtime.model.forward_backbone = original_forward  # type: ignore[assignment]
    wall_seconds = float(time.perf_counter() - start)
    return teacher_by_sample, {
        "wall_seconds": wall_seconds,
        "data_prepare_seconds": float(data_prepare_seconds),
        "cuda_transfer_seconds": float(cuda_transfer_seconds),
        "forward_seconds": float(forward_seconds),
    }


def load_or_build_runtime(
    ctx: OracleE2EContext,
    variant: str,
    *,
    oracle_mode: str,
) -> MCQMV2EvalRuntime:
    cache_key = f"{variant}:{oracle_mode}"
    cached = ctx.runtimes.get(cache_key)
    if cached is not None:
        return cached
    runtime, build_seconds, load_seconds = build_runtime_strict_timed(
        train=False,
        variant=variant,
        oracle_mode=oracle_mode,
        checkpoint_path=ctx.init_checkpoint_path,
    )
    ctx.timing_totals.model_build_seconds += build_seconds
    ctx.timing_totals.checkpoint_load_seconds += load_seconds
    ctx.timing_totals.model_build_count += 1
    ctx.timing_totals.checkpoint_load_count += 1
    ctx.runtimes[cache_key] = runtime
    return runtime


def run_mcqm_v2_eval_variant(
    variant: str,
    checkpoint_path: str | Path,
    sample_start: int,
    sample_end: int,
    *,
    capture_enabled: bool = False,
    oracle_mode: str = "disabled",
    teacher_by_sample: dict[int, dict[str, Any]] | None = None,
    forced_pairs_by_sample: dict[int, list[dict[str, int]]] | None = None,
    artifact_label: str = "",
) -> dict[str, Any]:
    mcqm_cfg = mcqm_v2_variant_cfg(variant, oracle_mode=oracle_mode)
    _, dataset, model, runtime_meta = build_runtime_strict(
        train=False,
        mcqm_cfg=mcqm_cfg,
        checkpoint_path=checkpoint_path,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    apply_mcqm_v2_parameter_freeze(model)
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, query_holder) if capture_enabled else None
    sectors = get_sector_masks_cached()
    per_row = []
    debug_rows = []
    protocol_rows = []
    capture_rows = []
    selection_signature_rows = []
    oracle_rows = []
    provenance_rows = []
    reset_runtime_state(model)
    reset_mcqm_memory(model)
    try:
        prev_scene = None
        for idx in range(sample_start, sample_end + 1):
            scene_token = dataset.data_infos[idx].get("scene_token", "")
            if prev_scene is not None and scene_token != prev_scene:
                reset_runtime_state(model)
                reset_mcqm_memory(model)
            prev_scene = scene_token
            _, batch = extract_inputs(dataset, idx)
            batch = enrich_batch_scene_meta(batch, dataset, idx)
            batch, perturb_manifest = apply_perturbation_with_manifest_to_batch(batch, "A10_drop_front_triplet")
            protocol_rows.append(sample_protocol_record(dataset, idx, sw2.unwrap(sw2.extract_sample_batch(dataset, idx, collate_fn)[0]), batch, perturb_manifest) | {"variant": variant})
            model.mcqm_v2_runtime_oracle = {
                "eval_only": True,
                "artifact_label": artifact_label,
                "forced_pairs": (forced_pairs_by_sample or {}).get(idx, []),
                **((teacher_by_sample or {}).get(idx, {})),
            }
            inputs = sw2.move_to_cuda(batch)
            with torch.inference_mode():
                out = model(return_loss=False, rescale=True, **inputs)
            mcqm_debug = getattr(model, "latest_mcqm_debug", {})
            row0 = (mcqm_debug.get("debug_rows", [{}]) or [{}])[0]
            debug_rows.append({"sample_index": idx, "variant": variant, **row0})
            selection_signature_rows.extend(
                [{"sample_index": idx, **row} for row in mcqm_debug.get("selection_signature_rows", [])]
            )
            oracle_rows.extend(
                [{"sample_index": idx, **row} for row in mcqm_debug.get("oracle_query_rows", [])]
            )
            sample_unwrapped = sw2.unwrap(sw2.extract_sample_batch(dataset, idx, collate_fn)[0])
            pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sample_unwrapped, out)
            if capture_enabled:
                final_pred = extract_final_pred_tensors(query_holder)
                parity_capture = extract_mcqm_parity_capture(model)
                capture_rows.append({
                    "sample_index": idx,
                    "variant": variant,
                    "actual_replacement_count": int(row0.get("actual_replacement_count", 0)),
                    "final_cls_scores": final_pred["cls_scores"],
                    "final_refine_pts": final_pred["refine_pts"],
                    "final_occ_dense": pred_temporal[0].cpu(),
                    "parity_capture": parity_capture,
                })
            for horizon_s in CORE_HORIZONS:
                pred_h = pred_temporal[horizon_s].cpu()
                gt_h = gt_temporal[horizon_s].cpu()
                gt0 = gt_temporal[0].cpu()
                row = sw12b.build_eval_row(pred_h, gt_h, gt0, "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=None)
                row.update({"sample_index": idx, "variant": variant, "horizon_s": horizon_s})
                per_row.append(row)
                if horizon_s == 0:
                    provenance_rows.append({"sample_index": idx, "variant": variant, **summarize_provenance(model, gt_h, horizon_s)})
    finally:
        if original_forward is not None:
            model.forward_backbone = original_forward  # type: ignore[assignment]
    summary = aggregate_rows(per_row)
    summary["variant"] = variant
    summary["debug"] = aggregate_rows(debug_rows) if debug_rows else {}
    summary["provenance"] = aggregate_rows(provenance_rows) if provenance_rows else {}
    return {
        "summary": summary,
        "rows": per_row,
        "debug_rows": debug_rows,
        "protocol_rows": protocol_rows,
        "capture_rows": capture_rows,
        "selection_signature_rows": selection_signature_rows,
        "oracle_rows": oracle_rows,
        "provenance_rows": provenance_rows,
        "checkpoint": runtime_meta,
    }


def run_mcqm_v2_target_eval(
    variant: str,
    checkpoint_path: str | Path,
    warmup_start: int,
    target_index: int,
    *,
    oracle_mode: str,
    teacher_by_sample: dict[int, dict[str, Any]] | None,
    forced_pairs: list[dict[str, int]],
    artifact_label: str,
    seed_memory_state: dict[int, Any] | None = None,
) -> dict[str, Any]:
    if warmup_start > target_index:
        raise ValueError("warmup_start must be <= target_index")
    mcqm_cfg = mcqm_v2_variant_cfg(variant, oracle_mode=oracle_mode)
    _, dataset, model, runtime_meta = build_runtime_strict(
        train=False,
        mcqm_cfg=mcqm_cfg,
        checkpoint_path=checkpoint_path,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    apply_mcqm_v2_parameter_freeze(model)
    sectors = get_sector_masks_cached()
    reset_runtime_state(model)
    reset_mcqm_memory(model)
    rows = []
    debug_rows = []
    capture_rows = []
    selection_signature_rows = []
    oracle_rows = []
    provenance_rows = []
    prev_scene = None
    start_index = warmup_start
    seed_audit = None
    if seed_memory_state is not None:
        seed_audit = restore_mcqm_memory_seed(model, seed_memory_state)
        start_index = target_index
    for idx in range(start_index, target_index + 1):
        scene_token = dataset.data_infos[idx].get("scene_token", "")
        if prev_scene is not None and scene_token != prev_scene:
            reset_runtime_state(model)
            reset_mcqm_memory(model)
        prev_scene = scene_token
        sample_raw, batch = extract_inputs(dataset, idx)
        batch = enrich_batch_scene_meta(batch, dataset, idx)
        batch, _ = apply_perturbation_with_manifest_to_batch(batch, "A10_drop_front_triplet")
        forced_for_idx = forced_pairs if idx == target_index else []
        oracle_runtime = {
            "eval_only": True,
            "artifact_label": artifact_label if idx == target_index else f"{artifact_label}_WARMUP",
            "forced_pairs": forced_for_idx,
        }
        if idx == target_index:
            oracle_runtime.update((teacher_by_sample or {}).get(idx, {}))
        model.mcqm_v2_runtime_oracle = {
            **oracle_runtime,
        }
        inputs = sw2.move_to_cuda(batch)
        with torch.inference_mode():
            out = model(return_loss=False, rescale=True, **inputs)
        mcqm_debug = getattr(model, "latest_mcqm_debug", {})
        row0 = (mcqm_debug.get("debug_rows", [{}]) or [{}])[0]
        if idx != target_index:
            continue
        debug_rows.append({"sample_index": idx, "variant": variant, **row0})
        selection_signature_rows.extend([{"sample_index": idx, **row} for row in mcqm_debug.get("selection_signature_rows", [])])
        oracle_rows.extend([{"sample_index": idx, **row} for row in mcqm_debug.get("oracle_query_rows", [])])
        pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sw2.unwrap(sample_raw), out)
        capture_rows.append({
            "sample_index": idx,
            "variant": variant,
            "actual_replacement_count": int(row0.get("actual_replacement_count", 0)),
            "final_occ_dense": pred_temporal[0].cpu(),
        })
        for horizon_s in CORE_HORIZONS:
            pred_h = pred_temporal[horizon_s].cpu()
            gt_h = gt_temporal[horizon_s].cpu()
            gt0 = gt_temporal[0].cpu()
            row = sw12b.build_eval_row(pred_h, gt_h, gt0, "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=None)
            row.update({"sample_index": idx, "variant": variant, "horizon_s": horizon_s})
            rows.append(row)
            if horizon_s == 0:
                provenance_rows.append({"sample_index": idx, "variant": variant, **summarize_provenance(model, gt_h, horizon_s)})
    summary = aggregate_rows(rows)
    summary["variant"] = variant
    summary["debug"] = aggregate_rows(debug_rows) if debug_rows else {}
    summary["provenance"] = aggregate_rows(provenance_rows) if provenance_rows else {}
    summary["warmup_start"] = int(warmup_start)
    summary["target_index"] = int(target_index)
    if seed_audit is not None:
        summary["seed_audit"] = seed_audit
    return {
        "summary": summary,
        "rows": rows,
        "debug_rows": debug_rows,
        "capture_rows": capture_rows,
        "selection_signature_rows": selection_signature_rows,
        "oracle_rows": oracle_rows,
        "provenance_rows": provenance_rows,
        "checkpoint": runtime_meta,
    }


def build_oracle_job_id(variant: str, sample_idx: int, k: int) -> str:
    return f"{variant}:sample{int(sample_idx)}:K{int(k)}"


def compute_eta_seconds(
    completed_jobs: int,
    total_jobs: int,
    elapsed_seconds: float,
) -> float:
    if completed_jobs <= 0 or total_jobs <= completed_jobs:
        return 0.0
    average = elapsed_seconds / float(completed_jobs)
    return average * float(total_jobs - completed_jobs)


def write_oracle_progress(
    ctx: OracleE2EContext,
    *,
    status: str,
    stage: str,
    variant: str = "",
    sample_index: int | None = None,
    requested_k: int | None = None,
    last_job_seconds: float = 0.0,
    forward_seconds: float = 0.0,
    cpu_overhead_seconds: float = 0.0,
    decision_status: str = "",
    error_type: str = "",
    error: str = "",
) -> None:
    allocated_gb, reserved_gb = gpu_mem_stats_gb()
    elapsed_seconds = float(time.time() - ctx.suite_started_at)
    average_job_seconds = (
        elapsed_seconds / float(ctx.completed_jobs)
        if ctx.completed_jobs > 0
        else 0.0
    )
    payload = {
        "status": status,
        "stage": stage,
        "variant": variant,
        "sample_index": sample_index,
        "requested_k": requested_k,
        "completed_jobs": int(ctx.completed_jobs),
        "total_jobs": int(ctx.preflight["forced_job_count"]),
        "progress_percent": (
            100.0 * float(ctx.completed_jobs) / float(ctx.preflight["forced_job_count"])
            if ctx.preflight["forced_job_count"] > 0
            else 0.0
        ),
        "last_job_seconds": float(last_job_seconds),
        "average_job_seconds": float(average_job_seconds),
        "eta_seconds": float(
            compute_eta_seconds(
                ctx.completed_jobs,
                int(ctx.preflight["forced_job_count"]),
                elapsed_seconds,
            )
        ),
        "gpu_allocated_gb": float(allocated_gb),
        "gpu_reserved_gb": float(reserved_gb),
        "cpu_rss_gb": float(cpu_rss_gb()),
        "forward_seconds": float(forward_seconds),
        "cpu_overhead_seconds": float(cpu_overhead_seconds),
        "decision_status": decision_status,
        "error_type": error_type,
        "error": error,
    }
    write_progress(payload)


def summarize_oracle_timing(totals: OracleE2ETimingTotals) -> dict[str, Any]:
    cpu_overhead = (
        totals.wall_seconds
        - totals.forward_seconds
    )
    exclusive_components = {
        "model_build_seconds": float(totals.model_build_seconds),
        "checkpoint_load_seconds": float(totals.checkpoint_load_seconds),
        "data_prepare_seconds": float(totals.data_prepare_seconds),
        "cuda_transfer_seconds": float(totals.cuda_transfer_seconds),
        "metric_seconds": float(totals.metric_seconds),
        "cpu_serialize_seconds": float(totals.cpu_serialize_seconds),
        "artifact_write_seconds": float(totals.artifact_write_seconds),
    }
    phase_totals = {
        "teacher_capture_seconds": float(totals.teacher_capture_seconds),
        "seed_capture_seconds": float(totals.seed_capture_seconds),
        "alignment_seconds": float(totals.alignment_seconds),
        "baseline_seconds": float(totals.baseline_seconds),
    }
    biggest_wait_item = (
        max(exclusive_components.items(), key=lambda item: item[1])[0]
        if exclusive_components
        else ""
    )
    return {
        "total_wall_time": float(totals.wall_seconds),
        "total_forward_time": float(totals.forward_seconds),
        "total_model_build_time": float(totals.model_build_seconds),
        "total_checkpoint_load_time": float(totals.checkpoint_load_seconds),
        "total_data_prepare_time": float(totals.data_prepare_seconds),
        "total_serialization_write_time": float(
            totals.cpu_serialize_seconds + totals.artifact_write_seconds
        ),
        "gpu_active_ratio": (
            float(totals.forward_seconds) / float(totals.wall_seconds)
            if totals.wall_seconds > 0
            else 0.0
        ),
        "cpu_overhead_time": float(cpu_overhead),
        "biggest_wait_item": biggest_wait_item,
        "exclusive_components": exclusive_components,
        "phase_totals": phase_totals,
        "model_build_count": int(totals.model_build_count),
        "checkpoint_load_count": int(totals.checkpoint_load_count),
    }


def decision_from_oracle_rows(
    rows: list[dict[str, Any]],
    oracle_alignment_rows_local: list[dict[str, Any]],
    required_target_samples: list[int],
) -> dict[str, Any]:
    if not rows:
        return {"status": "MCQM_V2_ORACLE_E2E_NOT_VALIDATED"}

    protocol_skip_reasons = {
        "target_has_no_same_scene_previous_frame",
        "target_has_no_in_window_previous_frame",
        "invalid_or_empty_seed",
    }
    protocol_skips = [
        row for row in rows
        if str(row.get("skip_reason", "")) in protocol_skip_reasons
    ]
    if protocol_skips:
        return {
            "status": "MCQM_V2_ORACLE_E2E_NOT_VALIDATED",
            "protocol_skip_count": int(len(protocol_skips)),
            "protocol_skip_reasons": sorted(
                {str(row.get("skip_reason", "")) for row in protocol_skips}
            ),
        }

    effective_rows: list[dict[str, Any]] = []
    for row in rows:
        skip_reason = str(row.get("skip_reason", ""))
        if skip_reason == "no_positive_oracle_pairs":
            effective_rows.append({
                **row,
                "delta_IoU": 0.0,
                "delta_false_free": 0.0,
                "delta_front_false_free": 0.0,
                "applied_pair_count": 0,
                "requested_forced_pair_count": 0,
                "resolved_forced_pair_count": 0,
                "effective_k": 0,
            })
        else:
            effective_rows.append({
                **row,
                "effective_k": int(row.get("requested_K", 0)),
            })

    seed_valid = all(
        is_valid_sha256(row.get("baseline_seed_memory_state_sha256"))
        and is_valid_sha256(row.get("forced_seed_memory_state_sha256"))
        and int(row.get("seed_query_count", 0)) > 0
        and bool(row.get("seed_source_all_native", False))
        for row in effective_rows
        if "baseline_seed_memory_state_sha256" in row
    )
    seed_consistent = all(
        str(row.get("baseline_seed_memory_state_sha256", "")) ==
        str(row.get("forced_seed_memory_state_sha256", ""))
        for row in effective_rows
        if "baseline_seed_memory_state_sha256" in row
    )
    applied_counts_valid = all(
        int(row.get("applied_pair_count", 0)) ==
        int(row.get("requested_forced_pair_count", 0)) ==
        int(row.get("resolved_forced_pair_count", 0))
        for row in effective_rows
        if str(row.get("skip_reason", "")) != "no_positive_oracle_pairs"
    )
    if not seed_valid or not seed_consistent or not applied_counts_valid:
        return {
            "status": "MCQM_V2_ORACLE_E2E_NOT_VALIDATED",
            "seed_valid": bool(seed_valid),
            "seed_consistent": bool(seed_consistent),
            "applied_counts_valid": bool(applied_counts_valid),
        }

    for row in effective_rows:
        for metric in ("delta_IoU", "delta_false_free", "delta_front_false_free"):
            value = float(row.get(metric, 0.0))
            if not math.isfinite(value):
                return {
                    "status": "MCQM_V2_ORACLE_E2E_NOT_VALIDATED",
                    "reason": f"non_finite_{metric}",
                    "variant": str(row.get("variant", "")),
                    "sample_index": int(row.get("sample_index", -1)),
                    "requested_K": int(row.get("requested_K", -1)),
                }

    variant_k_sample_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in effective_rows:
        variant = str(row["variant"])
        requested_k = int(row.get("requested_K", 0))
        sample_idx = int(row["sample_index"])
        if sample_idx in required_target_samples:
            variant_k_sample_rows[(variant, requested_k)].append(row)

    summary_by_variant_k: dict[str, dict[int, Any]] = defaultdict(dict)
    for variant in ["H0", "H1", "H2_A10", "H2_A25", "H3"]:
        for requested_k in [1, 4, 8, 16]:
            rows_for_variant_k = sorted(
                variant_k_sample_rows.get((variant, requested_k), []),
                key=lambda item: int(item["sample_index"]),
            )
            if len(rows_for_variant_k) != len(required_target_samples):
                return {
                    "status": "MCQM_V2_ORACLE_E2E_NOT_VALIDATED",
                    "variant": variant,
                    "requested_K": requested_k,
                    "expected_target_samples": required_target_samples,
                    "observed_target_samples": [
                        int(row["sample_index"]) for row in rows_for_variant_k
                    ],
                }
            delta_iou_values = [float(row.get("delta_IoU", 0.0)) for row in rows_for_variant_k]
            delta_ff_values = [float(row.get("delta_false_free", 0.0)) for row in rows_for_variant_k]
            delta_front_ff_values = [float(row.get("delta_front_false_free", 0.0)) for row in rows_for_variant_k]
            positive_sample_count = sum(1 for value in delta_iou_values if value > 0.0)
            summary_by_variant_k[variant][requested_k] = {
                "sample_rows": rows_for_variant_k,
                "mean_delta_IoU": float(sum(delta_iou_values) / len(delta_iou_values)),
                "mean_delta_false_free": float(sum(delta_ff_values) / len(delta_ff_values)),
                "mean_delta_front_false_free": float(sum(delta_front_ff_values) / len(delta_front_ff_values)),
                "positive_sample_count": int(positive_sample_count),
                "sample_count": int(len(rows_for_variant_k)),
                "requested_K": int(requested_k),
            }

    def _supports_h2_training(stats: dict[str, Any]) -> bool:
        mean_delta_iou = float(stats["mean_delta_IoU"])
        mean_delta_ff = float(stats["mean_delta_false_free"])
        mean_delta_front_ff = float(stats["mean_delta_front_false_free"])
        positive_sample_count = int(stats["positive_sample_count"])
        strong_condition = mean_delta_iou > 0.0 and mean_delta_ff <= 0.0
        secondary_condition = (
            mean_delta_iou >= 0.003
            and mean_delta_front_ff < 0.0
            and mean_delta_ff <= 0.01
        )
        return (strong_condition or secondary_condition) and positive_sample_count >= 2

    decision_k = ORACLE_E2E_DECISION_K
    best_stats_by_variant = {
        variant: summary_by_variant_k[variant][decision_k]
        for variant in ["H0", "H1", "H2_A10", "H2_A25", "H3"]
    }
    h2_support_variants = [
        variant
        for variant in ["H2_A10", "H2_A25"]
        if _supports_h2_training(best_stats_by_variant[variant])
    ]
    max_proxy_utility_by_variant: dict[str, float] = defaultdict(lambda: float("-inf"))
    for row in oracle_alignment_rows_local:
        variant = str(row.get("variant", ""))
        utility_query = float(row.get("utility_query", float("-inf")))
        if math.isfinite(utility_query):
            max_proxy_utility_by_variant[variant] = max(
                max_proxy_utility_by_variant[variant],
                utility_query,
            )

    if h2_support_variants:
        best_variant = max(
            h2_support_variants,
            key=lambda variant: float(best_stats_by_variant[variant]["mean_delta_IoU"]),
        )
        return {
            "status": "MCQM_V2_ORACLE_SUPPORTS_GATED_FEATURE_RESIDUAL_TRAINING",
            "selected_variant": best_variant,
            "selected_K": int(decision_k),
            "decision_k": int(decision_k),
            "variant_summary_by_K": summary_by_variant_k,
            "best_stats_by_variant": best_stats_by_variant,
        }

    h1_stats = best_stats_by_variant["H1"]
    if float(h1_stats["mean_delta_IoU"]) > 0.0 and int(h1_stats["positive_sample_count"]) >= 2:
        return {
            "status": "MCQM_V2_ORACLE_SUPPORTS_FEATURE_REPLACEMENT_BUT_NOT_RESIDUAL",
            "selected_K": int(decision_k),
            "decision_k": int(decision_k),
            "variant_summary_by_K": summary_by_variant_k,
            "best_stats_by_variant": best_stats_by_variant,
        }

    h3_stats = best_stats_by_variant["H3"]
    if float(h3_stats["mean_delta_IoU"]) > 0.0 and int(h3_stats["positive_sample_count"]) >= 2:
        return {
            "status": "MCQM_V2_ORACLE_SUPPORTS_GEOMETRY_MEMORY_ONLY",
            "selected_K": int(decision_k),
            "decision_k": int(decision_k),
            "variant_summary_by_K": summary_by_variant_k,
            "best_stats_by_variant": best_stats_by_variant,
        }

    any_positive_e2e = any(float(stats["mean_delta_IoU"]) > 0.0 for stats in best_stats_by_variant.values())
    any_positive_proxy = any(value > 0.0 for value in max_proxy_utility_by_variant.values() if math.isfinite(value))
    all_e2e_nonpositive = all(float(stats["mean_delta_IoU"]) <= 0.0 for stats in best_stats_by_variant.values())
    if any_positive_e2e:
        return {
            "status": "MCQM_V2_HAS_POSITIVE_E2E_GAIN_BUT_FAILS_TRAINING_GATE",
            "decision_k": int(decision_k),
            "variant_summary_by_K": summary_by_variant_k,
            "best_stats_by_variant": best_stats_by_variant,
            "max_proxy_utility_by_variant": dict(max_proxy_utility_by_variant),
        }
    if any_positive_proxy and all_e2e_nonpositive:
        return {
            "status": "MCQM_V2_QUERY_ALIGNMENT_PROXY_DOES_NOT_PREDICT_E2E_GAIN",
            "decision_k": int(decision_k),
            "variant_summary_by_K": summary_by_variant_k,
            "best_stats_by_variant": best_stats_by_variant,
            "max_proxy_utility_by_variant": dict(max_proxy_utility_by_variant),
        }
    return {
        "status": "MCQM_V2_QUERY_MEMORY_INTERFACE_HAS_NO_ORACLE_UPPER_BOUND",
        "decision_k": int(decision_k),
        "variant_summary_by_K": summary_by_variant_k,
        "best_stats_by_variant": best_stats_by_variant,
        "max_proxy_utility_by_variant": dict(max_proxy_utility_by_variant),
    }


def dry_run_oracle_e2e(sample_start: int, sample_end: int) -> dict[str, Any]:
    _, dataset = build_dataset_only(
        train=False,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    preflight = preflight_oracle_e2e(dataset, sample_start, sample_end)
    init_manifest = load_existing_init_manifest()
    runtime_variants = preflight["runtime_variants"]
    expected_model_builds = int(len(runtime_variants))
    expected_checkpoint_loads = int(len(runtime_variants))
    if (
        expected_model_builds > len(runtime_variants)
        or expected_checkpoint_loads > len(runtime_variants)
    ):
        raise RuntimeError(
            "dry-run failed: expected model/checkpoint loads exceed runtime variants"
        )
    partial_path = V2_ARTIFACTS_DIR / "oracle_e2e_validation.partial.jsonl"
    partial_rows = load_partial_e2e_index(partial_path)
    payload = {
        "status": "dry_run",
        "sample_window": [int(sample_start), int(sample_end)],
        "valid_target_samples": preflight["valid_targets"],
        "variant_list": preflight["variants"],
        "runtime_variant_list": runtime_variants,
        "k_values": preflight["k_values"],
        "total_forced_job_count": int(preflight["forced_job_count"]),
        "reuse_existing_forward_artifact": file_exists_and_valid_json(
            V2_ARTIFACTS_DIR / "interface_forward_ablation.json"
        ),
        "reuse_existing_alignment_artifact": file_exists_and_valid_json(
            V2_ARTIFACTS_DIR / "oracle_query_alignment.json"
        ),
        "reuse_existing_init_checkpoint": file_exists_and_valid_json(
            V2_ARTIFACTS_DIR / "mcqm_v2_init_checkpoint_manifest.json"
        ),
        "expected_model_build_count": expected_model_builds,
        "expected_checkpoint_load_count": expected_checkpoint_loads,
        "partial_resume_job_count": int(len(partial_rows)),
        "partial_path": str(partial_path),
        "init_checkpoint_path": init_manifest["init_checkpoint_path"],
    }
    write_json(V2_ARTIFACTS_DIR / "preflight.json", payload)
    return payload


def run_mcqm_v2_decision_only(sample_start: int, sample_end: int) -> dict[str, Any]:
    write_test_report("running", stage="decision_only")
    try:
        _, dataset = build_dataset_only(
            train=False,
            cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
        )
        preflight = preflight_oracle_e2e(dataset, sample_start, sample_end)
        oracle_e2e = read_json(V2_ARTIFACTS_DIR / "oracle_e2e_validation.json")
        oracle_alignment = load_existing_oracle_alignment_artifact()
        decision = decision_from_oracle_rows(
            oracle_e2e["rows"],
            oracle_alignment["rows"],
            preflight["valid_targets"],
        )
        write_json(V2_ARTIFACTS_DIR / "decision.json", decision)
        write_progress(
            {
                "status": "completed",
                "stage": "decision_completed",
                "completed_jobs": int(preflight["forced_job_count"]),
                "total_jobs": int(preflight["forced_job_count"]),
                "progress_percent": 100.0,
                "decision_status": decision["status"],
                "gpu_allocated_gb": 0.0,
                "gpu_reserved_gb": 0.0,
                "cpu_rss_gb": float(cpu_rss_gb()),
                "forward_seconds": 0.0,
                "cpu_overhead_seconds": 0.0,
                "last_job_seconds": 0.0,
                "average_job_seconds": 0.0,
                "eta_seconds": 0.0,
                "variant": "",
                "sample_index": None,
                "requested_k": None,
                "error_type": "",
                "error": "",
            }
        )
        write_test_report(
            "completed",
            stage="decision_only",
            decision_status=decision["status"],
            completed_jobs=int(preflight["forced_job_count"]),
            total_jobs=int(preflight["forced_job_count"]),
        )
        return decision
    except Exception as exc:
        write_progress(
            {
                "status": "failed",
                "stage": "decision_only_failed",
                "completed_jobs": 0,
                "total_jobs": 0,
                "progress_percent": 0.0,
                "decision_status": "",
                "gpu_allocated_gb": 0.0,
                "gpu_reserved_gb": 0.0,
                "cpu_rss_gb": float(cpu_rss_gb()),
                "forward_seconds": 0.0,
                "cpu_overhead_seconds": 0.0,
                "last_job_seconds": 0.0,
                "average_job_seconds": 0.0,
                "eta_seconds": 0.0,
                "variant": "",
                "sample_index": None,
                "requested_k": None,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        write_test_report(
            "failed",
            stage="decision_only",
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
        raise


def load_h3_formal_oracle_context() -> dict[str, Any]:
    oracle_path = V2_ARTIFACTS_DIR / "oracle_e2e_validation.json"
    decision_path = V2_ARTIFACTS_DIR / "decision.json"
    if not file_exists_and_valid_json(oracle_path):
        raise RuntimeError(f"missing formal Oracle E2E artifact: {oracle_path}")
    if not file_exists_and_valid_json(decision_path):
        raise RuntimeError(f"missing formal decision artifact: {decision_path}")
    oracle_payload = read_json(oracle_path)
    decision_payload = read_json(decision_path)
    decision_k = int(decision_payload.get("decision_k", decision_payload.get("selected_K", ORACLE_E2E_DECISION_K)))
    h3_rows = sorted(
        [
            row
            for row in oracle_payload["rows"]
            if str(row.get("variant", "")) == "H3"
            and int(row.get("requested_K", -1)) == decision_k
        ],
        key=lambda item: int(item["sample_index"]),
    )
    if not h3_rows:
        raise RuntimeError("formal Oracle E2E artifact has no H3 rows for selected K")
    target_samples = [int(row["sample_index"]) for row in h3_rows]
    return {
        "oracle_payload": oracle_payload,
        "decision_payload": decision_payload,
        "decision_k": decision_k,
        "target_samples": target_samples,
        "formal_h3_rows": h3_rows,
        "formal_h3_row_by_sample": {
            int(row["sample_index"]): row for row in h3_rows
        },
    }


def _selection_payload_pair_lists(selection_signature_rows: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    if not selection_signature_rows:
        return [], [], []
    payload = selection_signature_rows[0].get("selection_signature_payload", {})
    native_indices = [int(v) for v in payload.get("native_replaced_indices", [])]
    memory_query_ids = [int(v) for v in payload.get("selected_memory_query_ids", [])]
    memory_classes = [int(v) for v in payload.get("selected_memory_classes", [])]
    return native_indices, memory_query_ids, memory_classes


def _geometry_only_interface_purity_row(
    *,
    sample_idx: int,
    p0_result: dict[str, Any],
    h3_result: dict[str, Any],
    forced_pairs: list[dict[str, int]],
    requested_k: int,
) -> dict[str, Any]:
    p0_debug = p0_result["debug_rows"][0]
    h3_debug = h3_result["debug_rows"][0]
    p0_parity = p0_result.get("parity_capture", {})
    h3_parity = h3_result.get("parity_capture", {})
    p0_native_indices, p0_memory_ids, _ = _selection_payload_pair_lists(
        p0_result.get("selection_signature_rows", [])
    )
    h3_native_indices, h3_memory_ids, h3_memory_classes = _selection_payload_pair_lists(
        h3_result.get("selection_signature_rows", [])
    )
    feature_compare = compare_tensor_pair(
        h3_parity["layer1_query_feat"],
        h3_parity["layer2_query_feat"],
    )
    cls_compare = compare_tensor_pair(
        h3_parity["callback_input_cls_score"],
        h3_parity["callback_output_cls_score"],
    )
    points_compare = compare_tensor_pair(
        h3_parity["layer1_query_points"],
        h3_parity["layer2_query_points"],
    )
    unselected_points_compare = compare_unselected_slots(
        h3_parity["layer1_query_points"],
        h3_parity["layer2_query_points"],
        h3_native_indices,
    )
    p0_feature_compare = compare_tensor_pair(
        p0_parity["layer1_query_feat"],
        p0_parity["layer2_query_feat"],
    )
    p0_cls_compare = compare_tensor_pair(
        p0_parity["callback_input_cls_score"],
        p0_parity["callback_output_cls_score"],
    )
    p0_points_compare = compare_tensor_pair(
        p0_parity["layer1_query_points"],
        p0_parity["layer2_query_points"],
    )
    forced_native = [int(item["native_slot_id"]) for item in forced_pairs]
    forced_memory = [int(item["memory_query_id"]) for item in forced_pairs]
    return {
        "sample_index": int(sample_idx),
        "requested_k": int(requested_k),
        "forced_pair_count": int(len(forced_pairs)),
        "h3_injection_mode": str(h3_debug.get("mcqm_v2_injection_mode", "")),
        "p0_injection_mode": str(p0_debug.get("mcqm_v2_injection_mode", "")),
        "h3_feature_exact_passthrough": bool(feature_compare["exact_equal"]),
        "h3_cls_exact_passthrough": bool(cls_compare["exact_equal"]),
        "h3_points_changed": bool(not points_compare["exact_equal"]),
        "h3_unselected_points_exact_equal": bool(unselected_points_compare["exact_equal"]),
        "p0_feature_exact_passthrough": bool(p0_feature_compare["exact_equal"]),
        "p0_cls_exact_passthrough": bool(p0_cls_compare["exact_equal"]),
        "p0_points_exact_passthrough": bool(p0_points_compare["exact_equal"]),
        "p0_actual_applied_pair_count": int(p0_debug.get("actual_applied_pair_count", 0)),
        "h3_actual_applied_pair_count": int(h3_debug.get("actual_applied_pair_count", 0)),
        "h3_feature_delta_norm_mean": float(h3_debug.get("feature_delta_norm_mean", 0.0)),
        "h3_point_delta_metric_norm_mean": float(h3_debug.get("point_delta_metric_norm_mean", 0.0)),
        "h3_selected_native_indices": h3_native_indices,
        "h3_selected_memory_query_ids": h3_memory_ids,
        "h3_selected_memory_classes": h3_memory_classes,
        "forced_native_indices": forced_native,
        "forced_memory_query_ids": forced_memory,
        "forced_matches_h3_selection": bool(
            forced_native == h3_native_indices and forced_memory == h3_memory_ids
        ),
        "p0_proposed_selection_count": int(p0_debug.get("proposed_selection_count", 0)),
        "h3_proposed_selection_count": int(h3_debug.get("proposed_selection_count", 0)),
        "p0_selection_memory_ids": p0_memory_ids,
        "p0_selection_native_indices": p0_native_indices,
    }


def _geometry_only_transition_rows(
    *,
    sample_idx: int,
    gt_h: torch.Tensor,
    front_mask: torch.Tensor,
    baseline_pred: torch.Tensor,
    h3_pred: torch.Tensor,
    baseline_masks: dict[str, torch.Tensor],
    h3_masks: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gt_occ = gt_occ_mask(gt_h)
    baseline_pred_occ = gt_occ_mask(baseline_pred)
    h3_pred_occ = gt_occ_mask(h3_pred)
    baseline_geo = baseline_masks["geometric_mask"]
    h3_geo = h3_masks["geometric_mask"]
    baseline_sem = baseline_masks["semantic_active_mask"]
    h3_sem = h3_masks["semantic_active_mask"]
    baseline_active = baseline_masks["active_mask"]
    h3_active = h3_masks["active_mask"]

    added_direct_coverage = gt_occ & h3_geo & (~baseline_geo)
    lost_direct_coverage = gt_occ & baseline_geo & (~h3_geo)
    corrected_gt_occ = gt_occ & (~baseline_pred_occ) & h3_pred_occ
    regressed_gt_occ = gt_occ & baseline_pred_occ & (~h3_pred_occ)
    added_fp = (~gt_occ) & (~baseline_pred_occ) & h3_pred_occ
    removed_fp = (~gt_occ) & baseline_pred_occ & (~h3_pred_occ)
    added_active = h3_active & (~baseline_active)
    removed_active = baseline_active & (~h3_active)
    baseline_s0 = baseline_masks["S0"]
    h3_s0 = h3_masks["S0"]
    baseline_s1 = baseline_masks["S1"]
    h3_s1 = h3_masks["S1"]
    baseline_s2 = baseline_masks["S2"]
    h3_s2 = h3_masks["S2"]
    baseline_s3 = baseline_masks["S3"]
    h3_s3 = h3_masks["S3"]

    coverage_row = {
        "sample_index": int(sample_idx),
        "gt_occupied_voxel_count": _count_mask(gt_occ),
        "baseline_direct_coverage_gt_occ_count": _count_mask(gt_occ & baseline_geo),
        "h3_direct_coverage_gt_occ_count": _count_mask(gt_occ & h3_geo),
        "added_direct_coverage_gt_occ_count": _count_mask(added_direct_coverage),
        "lost_direct_coverage_gt_occ_count": _count_mask(lost_direct_coverage),
        "added_direct_coverage_pred_occupied_correct_count": _count_mask(added_direct_coverage & h3_pred_occ),
        "added_direct_coverage_pred_still_free_count": _count_mask(added_direct_coverage & (~h3_pred_occ)),
        "baseline_correct_h3_regressed_gt_occ_count": _count_mask(regressed_gt_occ),
        "baseline_wrong_h3_corrected_gt_occ_count": _count_mask(corrected_gt_occ),
        "false_occupied_added_count": _count_mask(added_fp),
        "false_occupied_removed_count": _count_mask(removed_fp),
        "baseline_geometric_voxel_count": _count_mask(baseline_geo),
        "h3_geometric_voxel_count": _count_mask(h3_geo),
        "baseline_semantic_active_voxel_count": _count_mask(baseline_sem),
        "h3_semantic_active_voxel_count": _count_mask(h3_sem),
        "baseline_final_active_voxel_count": _count_mask(baseline_active),
        "h3_final_active_voxel_count": _count_mask(h3_active),
        "corrected_gt_occ_in_baseline_S0_count": _count_mask(corrected_gt_occ & baseline_s0),
        "corrected_gt_occ_in_baseline_S1_count": _count_mask(corrected_gt_occ & baseline_s1),
        "corrected_gt_occ_in_baseline_S2_count": _count_mask(corrected_gt_occ & baseline_s2),
        "added_direct_coverage_in_baseline_S0_count": _count_mask(added_direct_coverage & baseline_s0),
        "added_direct_coverage_in_baseline_S1_count": _count_mask(added_direct_coverage & baseline_s1),
        "added_direct_coverage_in_baseline_S2_count": _count_mask(added_direct_coverage & baseline_s2),
    }
    coverage_row.update(
        {
            "front_gt_occupied_voxel_count": _count_mask(gt_occ & front_mask),
            "front_baseline_direct_coverage_gt_occ_count": _count_mask(gt_occ & front_mask & baseline_geo),
            "front_h3_direct_coverage_gt_occ_count": _count_mask(gt_occ & front_mask & h3_geo),
            "front_added_direct_coverage_gt_occ_count": _count_mask(added_direct_coverage & front_mask),
            "front_lost_direct_coverage_gt_occ_count": _count_mask(lost_direct_coverage & front_mask),
            "front_corrected_gt_occ_count": _count_mask(corrected_gt_occ & front_mask),
            "front_regressed_gt_occ_count": _count_mask(regressed_gt_occ & front_mask),
            "front_false_occupied_added_count": _count_mask(added_fp & front_mask),
            "front_false_occupied_removed_count": _count_mask(removed_fp & front_mask),
            "front_corrected_gt_occ_in_baseline_S0_count": _count_mask(corrected_gt_occ & front_mask & baseline_s0),
        }
    )

    bucket_rows: list[dict[str, Any]] = []
    for region_name, region_mask in [
        ("overall", torch.ones_like(gt_occ, dtype=torch.bool)),
        ("front", front_mask.bool()),
    ]:
        for src_name in ["S0", "S1", "S2"]:
            src_mask = baseline_masks[src_name] & region_mask
            for dst_name in ["S0", "S1", "S2"]:
                dst_mask = h3_masks[dst_name] & region_mask
                bucket_rows.append(
                    {
                        "sample_index": int(sample_idx),
                        "region": region_name,
                        "source_bucket": src_name,
                        "target_bucket": dst_name,
                        "count": _count_mask(src_mask & dst_mask),
                    }
                )
        bucket_rows.append(
            {
                "sample_index": int(sample_idx),
                "region": region_name,
                "source_bucket": "S3",
                "target_bucket": "S3",
                "count": _count_mask(baseline_s3 & h3_s3 & region_mask),
            }
        )

    fp_rows: list[dict[str, Any]] = []
    for region_name, region_mask in [
        ("overall", torch.ones_like(gt_occ, dtype=torch.bool)),
        ("front", front_mask.bool()),
    ]:
        fp_rows.append(
            {
                "sample_index": int(sample_idx),
                "region": region_name,
                "false_occupied_added_count": _count_mask(added_fp & region_mask),
                "false_occupied_removed_count": _count_mask(removed_fp & region_mask),
                "active_voxels_added_count": _count_mask(added_active & region_mask),
                "active_voxels_removed_count": _count_mask(removed_active & region_mask),
            }
        )

    per_target_row = {
        "sample_index": int(sample_idx),
        "baseline_d3_count": _count_mask(baseline_masks["D3"]),
        "h3_d3_count": _count_mask(h3_masks["D3"]),
        "baseline_s0_count": _count_mask(baseline_s0),
        "baseline_s1_count": _count_mask(baseline_s1),
        "baseline_s2_count": _count_mask(baseline_s2),
        "baseline_s3_count": _count_mask(baseline_s3),
        "h3_s0_count": _count_mask(h3_s0),
        "h3_s1_count": _count_mask(h3_s1),
        "h3_s2_count": _count_mask(h3_s2),
        "h3_s3_count": _count_mask(h3_s3),
        "s0_to_s1_count": _count_mask(baseline_s0 & h3_s1),
        "s0_to_s2_count": _count_mask(baseline_s0 & h3_s2),
        "s1_to_s2_count": _count_mask(baseline_s1 & h3_s2),
        "s2_to_s0_count": _count_mask(baseline_s2 & h3_s0),
    }
    return coverage_row, bucket_rows, fp_rows, [per_target_row]


def _geometry_only_metric_row(
    *,
    sample_idx: int,
    formal_row: dict[str, Any],
    replay_baseline_row: dict[str, Any],
    replay_h3_row: dict[str, Any],
    coverage_row: dict[str, Any],
    purity_row: dict[str, Any],
) -> dict[str, Any]:
    replay_delta_iou = float(replay_h3_row.get("occupied_iou", 0.0)) - float(replay_baseline_row.get("occupied_iou", 0.0))
    replay_delta_ff = float(replay_h3_row.get("false_free_rate", 0.0)) - float(replay_baseline_row.get("false_free_rate", 0.0))
    replay_delta_front_ff = float(replay_h3_row.get("front_sector_false_free", 0.0)) - float(replay_baseline_row.get("front_sector_false_free", 0.0))
    return {
        "sample_index": int(sample_idx),
        "formal_delta_IoU": float(formal_row.get("delta_IoU", 0.0)),
        "formal_delta_false_free": float(formal_row.get("delta_false_free", 0.0)),
        "formal_delta_front_false_free": float(formal_row.get("delta_front_false_free", 0.0)),
        "replay_delta_IoU": float(replay_delta_iou),
        "replay_delta_false_free": float(replay_delta_ff),
        "replay_delta_front_false_free": float(replay_delta_front_ff),
        "replay_matches_formal": bool(
            abs(replay_delta_iou - float(formal_row.get("delta_IoU", 0.0))) < 1e-9
            and abs(replay_delta_ff - float(formal_row.get("delta_false_free", 0.0))) < 1e-9
            and abs(replay_delta_front_ff - float(formal_row.get("delta_front_false_free", 0.0))) < 1e-9
        ),
        "baseline_iou": float(replay_baseline_row.get("occupied_iou", 0.0)),
        "h3_iou": float(replay_h3_row.get("occupied_iou", 0.0)),
        "baseline_false_free": float(replay_baseline_row.get("false_free_rate", 0.0)),
        "h3_false_free": float(replay_h3_row.get("false_free_rate", 0.0)),
        "baseline_front_false_free": float(replay_baseline_row.get("front_sector_false_free", 0.0)),
        "h3_front_false_free": float(replay_h3_row.get("front_sector_false_free", 0.0)),
        "added_direct_coverage_gt_occ_count": int(coverage_row["added_direct_coverage_gt_occ_count"]),
        "lost_direct_coverage_gt_occ_count": int(coverage_row["lost_direct_coverage_gt_occ_count"]),
        "corrected_gt_occ_count": int(coverage_row["baseline_wrong_h3_corrected_gt_occ_count"]),
        "regressed_gt_occ_count": int(coverage_row["baseline_correct_h3_regressed_gt_occ_count"]),
        "false_occupied_added_count": int(coverage_row["false_occupied_added_count"]),
        "false_occupied_removed_count": int(coverage_row["false_occupied_removed_count"]),
        "front_corrected_gt_occ_count": int(coverage_row["front_corrected_gt_occ_count"]),
        "front_regressed_gt_occ_count": int(coverage_row["front_regressed_gt_occ_count"]),
        "front_false_occupied_added_count": int(coverage_row["front_false_occupied_added_count"]),
        "front_false_occupied_removed_count": int(coverage_row["front_false_occupied_removed_count"]),
        "corrected_gt_occ_in_baseline_S0_count": int(coverage_row["corrected_gt_occ_in_baseline_S0_count"]),
        "added_direct_coverage_in_baseline_S0_count": int(coverage_row["added_direct_coverage_in_baseline_S0_count"]),
        "feature_exact_passthrough": bool(purity_row["h3_feature_exact_passthrough"]),
        "cls_exact_passthrough": bool(purity_row["h3_cls_exact_passthrough"]),
        "unselected_points_exact_equal": bool(purity_row["h3_unselected_points_exact_equal"]),
    }


def geometry_only_decision(
    per_target_rows: list[dict[str, Any]],
    blocking_issues: list[dict[str, Any]],
    interface_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if blocking_issues:
        return {
            "status": "GEOMETRY_ONLY_RESULT_INVALID",
            "blocking_issue_count": int(len(blocking_issues)),
            "blocking_issues": blocking_issues,
        }
    positive_rows = [row for row in per_target_rows if float(row["replay_delta_IoU"]) > 0.0]
    corrected_from_s0 = [
        row for row in positive_rows
        if int(row["corrected_gt_occ_in_baseline_S0_count"]) > 0
        and int(row["added_direct_coverage_in_baseline_S0_count"]) > 0
    ]
    purity_ok = all(
        bool(row["h3_feature_exact_passthrough"])
        and bool(row["h3_cls_exact_passthrough"])
        and bool(row["h3_unselected_points_exact_equal"])
        and bool(row["forced_matches_h3_selection"])
        for row in interface_rows
    )
    if purity_ok and len(corrected_from_s0) >= 2:
        return {
            "status": "GEOMETRY_ONLY_MECHANISM_CONFIRMED",
            "positive_target_count": int(len(positive_rows)),
            "s0_supported_positive_target_count": int(len(corrected_from_s0)),
        }
    return {
        "status": "GEOMETRY_ONLY_WEAK_SIGNAL_NOT_CONFIRMED",
        "positive_target_count": int(len(positive_rows)),
        "s0_supported_positive_target_count": int(len(corrected_from_s0)),
        "purity_ok": bool(purity_ok),
    }


def run_mcqm_v2_geometry_only_audit(
    sample_start: int,
    sample_end: int,
) -> dict[str, Any]:
    GEOMETRY_ONLY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        GEOMETRY_ONLY_AUDIT_DIR / "test_report.json",
        {
            "status": "running",
            "stage": "geometry_only_audit",
            "updated_at": iso_now(),
        },
    )
    blocking_issues: list[dict[str, Any]] = []
    try:
        formal = load_h3_formal_oracle_context()
        target_samples = [int(v) for v in formal["target_samples"]]
        decision_k = int(formal["decision_k"])
        if target_samples != [43, 47, 49]:
            blocking_issues.append(
                {
                    "type": "unexpected_formal_targets",
                    "target_samples": target_samples,
                }
            )
        init_manifest = load_existing_init_manifest()
        oracle_decision = formal["decision_payload"]
        p0_runtime, p0_build_seconds, p0_load_seconds = build_runtime_strict_timed(
            train=False,
            variant="P0",
            oracle_mode="disabled",
            checkpoint_path=Path(init_manifest["init_checkpoint_path"]),
        )
        h3_runtime, h3_build_seconds, h3_load_seconds = build_runtime_strict_timed(
            train=False,
            variant="H3",
            oracle_mode="query_alignment_report_only",
            checkpoint_path=Path(init_manifest["init_checkpoint_path"]),
        )
        cpu_batch_cache: dict[tuple[int, str], tuple[Any, Any]] = {}
        teacher_by_sample, teacher_occ_by_sample, teacher_timings = capture_clean_teacher_artifacts_with_runtime(
            p0_runtime,
            sample_start,
            sample_end,
            target_samples,
            cpu_batch_cache,
        )
        front_mask = get_sector_masks_cached()["front"].cpu().bool()
        coverage_rows: list[dict[str, Any]] = []
        bucket_rows: list[dict[str, Any]] = []
        fp_rows: list[dict[str, Any]] = []
        per_target_rows: list[dict[str, Any]] = []
        interface_rows: list[dict[str, Any]] = []
        for sample_idx in target_samples:
            warmup_start = sample_idx - 1
            p0_seed, _ = capture_p0_memory_seed_with_runtime(
                p0_runtime,
                warmup_start,
                sample_idx,
                cpu_batch_cache,
            )
            teacher_state = teacher_by_sample[sample_idx]
            teacher_occ = teacher_occ_by_sample[sample_idx]
            baseline = target_eval_with_occ_debug(
                p0_runtime,
                target_index=sample_idx,
                perturbation_id="A10_drop_front_triplet",
                seed_memory_state=p0_seed["state"],
                teacher_state=None,
                forced_pairs=[],
                artifact_label=f"MCQM_V2_GEOMETRY_AUDIT_P0_{sample_idx}",
                cpu_batch_cache=cpu_batch_cache,
            )
            alignment = target_eval_with_occ_debug(
                h3_runtime,
                target_index=sample_idx,
                perturbation_id="A10_drop_front_triplet",
                seed_memory_state=p0_seed["state"],
                teacher_state=teacher_state,
                forced_pairs=[],
                artifact_label=f"MCQM_V2_ORACLE_GEOMETRY_AUDIT_H3_ALIGNMENT_{sample_idx}",
                cpu_batch_cache=cpu_batch_cache,
            )
            forced_pair_info = build_positive_nonconflicting_forced_pairs(
                alignment["oracle_rows"],
                decision_k,
            )
            forced_pairs = forced_pair_info["forced_pairs"]
            h3_forced = target_eval_with_occ_debug(
                h3_runtime,
                target_index=sample_idx,
                perturbation_id="A10_drop_front_triplet",
                seed_memory_state=p0_seed["state"],
                teacher_state=teacher_state,
                forced_pairs=forced_pairs,
                artifact_label=f"MCQM_V2_ORACLE_GEOMETRY_AUDIT_H3_FORCED_{sample_idx}",
                cpu_batch_cache=cpu_batch_cache,
            )
            formal_row = formal["formal_h3_row_by_sample"][sample_idx]
            baseline_row = build_target_metric_map(baseline)[(sample_idx, 0)]
            h3_row = build_target_metric_map(h3_forced)[(sample_idx, 0)]
            replay_delta_iou = float(h3_row["occupied_iou"]) - float(baseline_row["occupied_iou"])
            replay_delta_ff = float(h3_row["false_free_rate"]) - float(baseline_row["false_free_rate"])
            replay_delta_front_ff = float(h3_row["front_sector_false_free"]) - float(baseline_row["front_sector_false_free"])
            if (
                abs(replay_delta_iou - float(formal_row["delta_IoU"])) > 1e-9
                or abs(replay_delta_ff - float(formal_row["delta_false_free"])) > 1e-9
                or abs(replay_delta_front_ff - float(formal_row["delta_front_false_free"])) > 1e-9
            ):
                blocking_issues.append(
                    {
                        "type": "formal_replay_mismatch",
                        "sample_index": int(sample_idx),
                        "formal_delta_IoU": float(formal_row["delta_IoU"]),
                        "replay_delta_IoU": float(replay_delta_iou),
                        "formal_delta_false_free": float(formal_row["delta_false_free"]),
                        "replay_delta_false_free": float(replay_delta_ff),
                        "formal_delta_front_false_free": float(formal_row["delta_front_false_free"]),
                        "replay_delta_front_false_free": float(replay_delta_front_ff),
                    }
                )
            purity_row = _geometry_only_interface_purity_row(
                sample_idx=sample_idx,
                p0_result=baseline,
                h3_result=h3_forced,
                forced_pairs=forced_pairs,
                requested_k=decision_k,
            )
            interface_rows.append(purity_row)
            if not purity_row["forced_matches_h3_selection"]:
                blocking_issues.append(
                    {
                        "type": "forced_pair_selection_mismatch",
                        "sample_index": int(sample_idx),
                        "forced_native_indices": purity_row["forced_native_indices"],
                        "h3_selected_native_indices": purity_row["h3_selected_native_indices"],
                        "forced_memory_query_ids": purity_row["forced_memory_query_ids"],
                        "h3_selected_memory_query_ids": purity_row["h3_selected_memory_query_ids"],
                    }
                )
            if not (
                purity_row["h3_feature_exact_passthrough"]
                and purity_row["h3_cls_exact_passthrough"]
                and purity_row["h3_unselected_points_exact_equal"]
                and purity_row["p0_feature_exact_passthrough"]
                and purity_row["p0_cls_exact_passthrough"]
                and purity_row["p0_points_exact_passthrough"]
            ):
                blocking_issues.append(
                    {
                        "type": "geometry_only_interface_purity_violation",
                        "sample_index": int(sample_idx),
                        "purity_row": purity_row,
                    }
                )
            baseline_raw_conf, baseline_raw_sem, baseline_raw_margin = dense_occ_top1_conf_margin(
                torch.as_tensor(baseline["occ_debug_h0"]["dense_occ_before_padding"]).detach().cpu()
            )
            baseline_d3 = build_d3_mask_from_occ(
                baseline["gt_h0"],
                baseline_raw_sem,
                baseline["pred_h0"],
                baseline_raw_conf,
                baseline_raw_margin,
            )
            baseline_masks = build_s0_s1_s2_s3_masks(
                baseline_d3,
                baseline["occ_debug_h0"],
            )
            h3_masks = build_s0_s1_s2_s3_masks(
                baseline_d3,
                h3_forced["occ_debug_h0"],
            )
            if (
                baseline["occ_debug_h0"]["pc_range"] != h3_forced["occ_debug_h0"]["pc_range"]
                or baseline["occ_debug_h0"]["voxel_num"] != h3_forced["occ_debug_h0"]["voxel_num"]
                or baseline["occ_debug_h0"]["voxel_size"] != h3_forced["occ_debug_h0"]["voxel_size"]
            ):
                blocking_issues.append(
                    {
                        "type": "geometry_grid_mismatch",
                        "sample_index": int(sample_idx),
                    }
                )
            coverage_row, sample_bucket_rows, sample_fp_rows, sample_per_target = _geometry_only_transition_rows(
                sample_idx=sample_idx,
                gt_h=baseline["gt_h0"],
                front_mask=front_mask,
                baseline_pred=baseline["pred_h0"],
                h3_pred=h3_forced["pred_h0"],
                baseline_masks=baseline_masks,
                h3_masks=h3_masks,
            )
            coverage_rows.append(coverage_row)
            bucket_rows.extend(sample_bucket_rows)
            fp_rows.extend(sample_fp_rows)
            metric_row = _geometry_only_metric_row(
                sample_idx=sample_idx,
                formal_row=formal_row,
                replay_baseline_row=baseline_row,
                replay_h3_row=h3_row,
                coverage_row=coverage_row,
                purity_row=purity_row,
            )
            metric_row.update(sample_per_target[0])
            per_target_rows.append(metric_row)

        decision = geometry_only_decision(
            per_target_rows,
            blocking_issues,
            interface_rows,
        )
        summary = {
            "audit_version": GEOMETRY_ONLY_AUDIT_VERSION,
            "source_artifacts": {
                "oracle_e2e_validation": str(V2_ARTIFACTS_DIR / "oracle_e2e_validation.json"),
                "decision": str(V2_ARTIFACTS_DIR / "decision.json"),
            },
            "fixed_requested_k": int(decision_k),
            "target_samples": target_samples,
            "formal_oracle_decision_status": str(oracle_decision.get("status", "")),
            "teacher_capture_timings": teacher_timings,
            "runtime_build_seconds": {
                "P0": float(p0_build_seconds),
                "H3": float(h3_build_seconds),
            },
            "runtime_checkpoint_load_seconds": {
                "P0": float(p0_load_seconds),
                "H3": float(h3_load_seconds),
            },
            "interface_rows": interface_rows,
            "per_target_rows": per_target_rows,
            "metric_semantics": {
                "false_free_definition": "FN / GT_occupied; lower is better",
                "front_false_free_definition": "front-region FN / front-region GT_occupied; lower is better",
                "delta_false_free_sign": "positive means worse",
                "delta_front_false_free_sign": "negative means better",
            },
            "blocking_issue_count": int(len(blocking_issues)),
            "decision": decision,
        }
        write_json(
            GEOMETRY_ONLY_AUDIT_DIR / "geometry_only_audit_summary.json",
            summary,
        )
        write_csv(
            GEOMETRY_ONLY_AUDIT_DIR / "geometry_only_per_target.csv",
            per_target_rows,
        )
        write_csv(
            GEOMETRY_ONLY_AUDIT_DIR / "geometry_only_bucket_transition.csv",
            bucket_rows,
        )
        write_csv(
            GEOMETRY_ONLY_AUDIT_DIR / "geometry_only_coverage_transition.csv",
            coverage_rows,
        )
        write_csv(
            GEOMETRY_ONLY_AUDIT_DIR / "geometry_only_false_positive_transition.csv",
            fp_rows,
        )
        write_json(
            GEOMETRY_ONLY_AUDIT_DIR / "geometry_only_blocking_issues.json",
            {
                "blocking_issue_count": int(len(blocking_issues)),
                "blocking_issues": blocking_issues,
            },
        )
        write_json(GEOMETRY_ONLY_AUDIT_DIR / "decision.json", decision)
        write_json(
            GEOMETRY_ONLY_AUDIT_DIR / "test_report.json",
            {
                "status": "completed",
                "stage": "geometry_only_audit",
                "updated_at": iso_now(),
                "decision_status": decision["status"],
                "target_samples": target_samples,
                "requested_k": int(decision_k),
            },
        )
        return summary
    except Exception as exc:
        blocking_issues.append(
            {
                "type": "exception",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        write_json(
            GEOMETRY_ONLY_AUDIT_DIR / "geometry_only_blocking_issues.json",
            {
                "blocking_issue_count": int(len(blocking_issues)),
                "blocking_issues": blocking_issues,
            },
        )
        write_json(
            GEOMETRY_ONLY_AUDIT_DIR / "decision.json",
            {
                "status": "GEOMETRY_ONLY_RESULT_INVALID",
                "blocking_issue_count": int(len(blocking_issues)),
                "blocking_issues": blocking_issues,
            },
        )
        write_json(
            GEOMETRY_ONLY_AUDIT_DIR / "test_report.json",
            {
                "status": "failed",
                "stage": "geometry_only_audit",
                "updated_at": iso_now(),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        raise


def run_mcqm_v2_oracle_e2e_only(
    sample_start: int,
    sample_end: int,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return dry_run_oracle_e2e(sample_start, sample_end)

    V2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    write_test_report("running", stage="preflight")
    write_progress(
        {
            "status": "running",
            "stage": "preflight_started",
            "completed_jobs": 0,
            "total_jobs": 0,
            "progress_percent": 0.0,
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "cpu_rss_gb": float(cpu_rss_gb()),
            "forward_seconds": 0.0,
            "cpu_overhead_seconds": 0.0,
            "last_job_seconds": 0.0,
            "average_job_seconds": 0.0,
            "eta_seconds": 0.0,
            "variant": "",
            "sample_index": None,
            "requested_k": None,
            "decision_status": "",
            "error_type": "",
            "error": "",
        }
    )
    suite_started_at = time.time()
    try:
        _, dataset = build_dataset_only(
            train=False,
            cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
        )
        preflight = preflight_oracle_e2e(dataset, sample_start, sample_end)
        preflight_payload = {
            **preflight,
            "reuse_existing_forward_artifact": file_exists_and_valid_json(
                V2_ARTIFACTS_DIR / "interface_forward_ablation.json"
            ),
            "reuse_existing_alignment_artifact": file_exists_and_valid_json(
                V2_ARTIFACTS_DIR / "oracle_query_alignment.json"
            ),
        }
        write_json(V2_ARTIFACTS_DIR / "preflight.json", preflight_payload)
        write_progress(
            {
                "status": "running",
                "stage": "preflight_passed",
                "completed_jobs": 0,
                "total_jobs": int(preflight["forced_job_count"]),
                "progress_percent": 0.0,
                "gpu_allocated_gb": 0.0,
                "gpu_reserved_gb": 0.0,
                "cpu_rss_gb": float(cpu_rss_gb()),
                "forward_seconds": 0.0,
                "cpu_overhead_seconds": 0.0,
                "last_job_seconds": 0.0,
                "average_job_seconds": 0.0,
                "eta_seconds": 0.0,
                "variant": "",
                "sample_index": None,
                "requested_k": None,
                "decision_status": "",
                "error_type": "",
                "error": "",
            }
        )

        init_manifest = load_existing_init_manifest()
        load_existing_forward_artifact()
        oracle_alignment_artifact = load_existing_oracle_alignment_artifact()
        ctx = OracleE2EContext(
            init_checkpoint_path=Path(init_manifest["init_checkpoint_path"]),
            init_checkpoint_sha256=str(init_manifest["init_checkpoint_sha256"]),
            preflight=preflight,
            forward_artifact_reused=True,
            alignment_artifact_reused=True,
            suite_started_at=suite_started_at,
        )
        partial_path = V2_ARTIFACTS_DIR / "oracle_e2e_validation.partial.jsonl"
        expected_job_ids = {
            build_oracle_job_id(variant, sample_idx, k)
            for variant in preflight["variants"]
            for sample_idx in preflight["valid_targets"]
            for k in preflight["k_values"]
        }
        if partial_path.exists():
            loaded_partial = load_partial_e2e_index(partial_path)
            needs_archive = any(job_id not in expected_job_ids for job_id in loaded_partial)
            if not needs_archive:
                needs_archive = any(
                    str(row.get("protocol_version", "")) != ORACLE_E2E_PROTOCOL_VERSION
                    for row in loaded_partial.values()
                )
            if needs_archive:
                archive_invalidated_artifact(
                    partial_path,
                    "target_window_bug",
                )
                loaded_partial = {}
            ctx.partial_rows_by_job = {
                job_id: row
                for job_id, row in loaded_partial.items()
                if job_id in expected_job_ids
            }
        else:
            ctx.partial_rows_by_job = {}
        archive_invalidated_artifact(
            V2_ARTIFACTS_DIR / "oracle_e2e_validation.json",
            "target_window_bug",
        )
        archive_invalidated_artifact(
            V2_ARTIFACTS_DIR / "decision.json",
            "target_window_bug",
        )

        write_oracle_progress(ctx, status="running", stage="runtime_build_started")
        p0_runtime = load_or_build_runtime(ctx, "P0", oracle_mode="disabled")
        variant_runtimes = {
            variant: load_or_build_runtime(ctx, variant, oracle_mode="query_alignment_report_only")
            for variant in preflight["variants"]
        }
        if (
            ctx.timing_totals.model_build_count > len(preflight["runtime_variants"])
            or ctx.timing_totals.checkpoint_load_count > len(preflight["runtime_variants"])
        ):
            raise RuntimeError("runtime build/load count exceeds runtime variant count")
        write_oracle_progress(ctx, status="running", stage="runtime_build_completed")

        teacher_by_sample, teacher_timings = capture_clean_teacher_states_with_runtime(
            p0_runtime,
            sample_start,
            sample_end,
            preflight["valid_targets"],
            ctx.cpu_batch_cache,
        )
        ctx.teacher_by_sample = teacher_by_sample
        ctx.timing_totals.teacher_capture_seconds += float(teacher_timings["wall_seconds"])
        ctx.timing_totals.data_prepare_seconds += float(teacher_timings["data_prepare_seconds"])
        ctx.timing_totals.cuda_transfer_seconds += float(teacher_timings["cuda_transfer_seconds"])
        ctx.timing_totals.forward_seconds += float(teacher_timings["forward_seconds"])

        for sample_idx in preflight["valid_targets"]:
            write_oracle_progress(
                ctx,
                status="running",
                stage="seed_capture_started",
                sample_index=sample_idx,
            )
            if sample_idx not in ctx.p0_seed_cache:
                warmup_start = sample_idx - 1
                p0_seed, seed_timings = capture_p0_memory_seed_with_runtime(
                    p0_runtime,
                    warmup_start,
                    sample_idx,
                    ctx.cpu_batch_cache,
                )
                ctx.p0_seed_cache[sample_idx] = p0_seed
                ctx.timing_totals.seed_capture_seconds += float(seed_timings["wall_seconds"])
                ctx.timing_totals.data_prepare_seconds += float(seed_timings["data_prepare_seconds"])
                ctx.timing_totals.cuda_transfer_seconds += float(seed_timings["cuda_transfer_seconds"])
                ctx.timing_totals.forward_seconds += float(seed_timings["forward_seconds"])
            p0_seed = ctx.p0_seed_cache[sample_idx]
            write_oracle_progress(
                ctx,
                status="running",
                stage="seed_capture_completed",
                sample_index=sample_idx,
            )

            teacher_state = ctx.teacher_by_sample.get(sample_idx, {})
            teacher_hash = teacher_state_sha256(teacher_state)
            baseline_cache = ctx.p0_baseline_cache.get(sample_idx)
            if baseline_cache is None:
                write_oracle_progress(
                    ctx,
                    status="running",
                    stage="baseline_started",
                    sample_index=sample_idx,
                    variant="P0",
                )
                baseline_start = time.perf_counter()
                baseline_target = target_eval_with_runtime(
                    p0_runtime,
                    target_index=sample_idx,
                    seed_memory_state=p0_seed["state"],
                    oracle_mode="disabled",
                    teacher_state=None,
                    forced_pairs=[],
                    artifact_label=f"MCQM_V2_P0_TARGET_{sample_idx}",
                    cpu_batch_cache=ctx.cpu_batch_cache,
                )
                baseline_seconds = time.perf_counter() - baseline_start
                ctx.timing_totals.baseline_seconds += baseline_seconds
                ctx.timing_totals.data_prepare_seconds += float(
                    baseline_target["timings"]["data_prepare_seconds"]
                )
                ctx.timing_totals.cuda_transfer_seconds += float(
                    baseline_target["timings"]["cuda_transfer_seconds"]
                )
                ctx.timing_totals.forward_seconds += float(
                    baseline_target["timings"]["forward_seconds"]
                )
                ctx.timing_totals.metric_seconds += float(
                    baseline_target["timings"]["metric_seconds"]
                )
                baseline_cache = {
                    "result": baseline_target,
                    "checkpoint_sha256": p0_runtime.checkpoint_sha256,
                    "runtime_config_hash": p0_runtime.runtime_config_hash,
                    "seed_sha": p0_seed["audit"]["seed_memory_state_sha256"],
                }
                ctx.p0_baseline_cache[sample_idx] = baseline_cache
                write_oracle_progress(
                    ctx,
                    status="running",
                    stage="baseline_completed",
                    sample_index=sample_idx,
                    variant="P0",
                )
            baseline_target = baseline_cache["result"]
            if (
                baseline_cache["checkpoint_sha256"] != p0_runtime.checkpoint_sha256
                or baseline_cache["runtime_config_hash"] != p0_runtime.runtime_config_hash
                or baseline_cache["seed_sha"] != p0_seed["audit"]["seed_memory_state_sha256"]
            ):
                raise RuntimeError("cached P0 baseline validation failed")

            baseline_map = build_target_metric_map(baseline_target)
            baseline_row = baseline_map[(sample_idx, 0)]

            for variant in preflight["variants"]:
                seeded_key = (variant, sample_idx)
                if seeded_key not in ctx.seeded_alignment_cache:
                    write_oracle_progress(
                        ctx,
                        status="running",
                        stage="alignment_started",
                        variant=variant,
                        sample_index=sample_idx,
                    )
                    alignment_start = time.perf_counter()
                    seeded_alignment = target_eval_with_runtime(
                        variant_runtimes[variant],
                        target_index=sample_idx,
                        seed_memory_state=p0_seed["state"],
                        oracle_mode="query_alignment_report_only",
                        teacher_state=teacher_state,
                        forced_pairs=[],
                        artifact_label=f"MCQM_V2_ORACLE_ALIGNMENT_TARGET_{variant}_{sample_idx}",
                        cpu_batch_cache=ctx.cpu_batch_cache,
                    )
                    alignment_seconds = time.perf_counter() - alignment_start
                    ctx.timing_totals.alignment_seconds += alignment_seconds
                    ctx.timing_totals.data_prepare_seconds += float(
                        seeded_alignment["timings"]["data_prepare_seconds"]
                    )
                    ctx.timing_totals.cuda_transfer_seconds += float(
                        seeded_alignment["timings"]["cuda_transfer_seconds"]
                    )
                    ctx.timing_totals.forward_seconds += float(
                        seeded_alignment["timings"]["forward_seconds"]
                    )
                    ctx.timing_totals.metric_seconds += float(
                        seeded_alignment["timings"]["metric_seconds"]
                    )
                    expected_injection_mode = mcqm_v2_variant_cfg(
                        variant,
                        oracle_mode="query_alignment_report_only",
                    )["mcqm_v2_injection_mode"]
                    seeded_oracle_rows = [
                        row
                        for row in seeded_alignment.get("oracle_rows", [])
                        if int(row.get("sample_index", -1)) == int(sample_idx)
                        and row.get("injection_mode") == expected_injection_mode
                    ]
                    ctx.seeded_alignment_cache[seeded_key] = {
                        "result": seeded_alignment,
                        "oracle_rows": seeded_oracle_rows,
                    }
                    write_oracle_progress(
                        ctx,
                        status="running",
                        stage="alignment_completed",
                        variant=variant,
                        sample_index=sample_idx,
                    )
                seeded_alignment = ctx.seeded_alignment_cache[seeded_key]
                seeded_oracle_rows = seeded_alignment["oracle_rows"]
                seeded_oracle_rows_hash = oracle_rows_sha256(seeded_oracle_rows)

                for k in preflight["k_values"]:
                    job_id = build_oracle_job_id(variant, sample_idx, k)
                    protocol_hash = current_protocol_hash(
                        checkpoint_sha256=variant_runtimes[variant].checkpoint_sha256,
                        architecture_signature_sha256=variant_runtimes[variant].architecture_signature_sha256,
                        runtime_config_hash=variant_runtimes[variant].runtime_config_hash,
                        sample_start=sample_start,
                        sample_end=sample_end,
                        target_samples=preflight["valid_targets"],
                        k_values=preflight["k_values"],
                        teacher_hash=teacher_hash,
                        p0_seed_hash=p0_seed["audit"]["seed_memory_state_sha256"],
                        seeded_oracle_rows_sha256=seeded_oracle_rows_hash,
                    )
                    existing = ctx.partial_rows_by_job.get(job_id)
                    if existing is not None:
                        if str(existing["protocol_hash"]) != protocol_hash:
                            raise RuntimeError(
                                f"resume protocol hash mismatch for {job_id}"
                            )
                        ctx.completed_jobs += 1
                        continue

                    forced_pair_info = build_positive_nonconflicting_forced_pairs(
                        seeded_oracle_rows,
                        k,
                    )
                    forced_pairs = forced_pair_info["forced_pairs"]
                    row_start = time.perf_counter()
                    write_oracle_progress(
                        ctx,
                        status="running",
                        stage="forced_forward_started",
                        variant=variant,
                        sample_index=sample_idx,
                        requested_k=k,
                    )
                    if not forced_pairs:
                        row = {
                            "job_id": job_id,
                            "protocol_hash": protocol_hash,
                            "protocol_version": ORACLE_E2E_PROTOCOL_VERSION,
                            "variant": variant,
                            "sample_index": int(sample_idx),
                            "requested_K": int(k),
                            "positive_nonconflicting_pair_count": int(
                                forced_pair_info["positive_nonconflicting_pair_count"]
                            ),
                            "requested_pair_count": 0,
                            "skip_reason": "no_positive_oracle_pairs",
                            "seeded_oracle_rows_sha256": seeded_oracle_rows_hash,
                            "seeded_alignment_row_count": int(len(seeded_oracle_rows)),
                            "warmup_start": int(sample_idx - 1),
                            **p0_seed["audit"],
                            "model_build_seconds": 0.0,
                            "checkpoint_load_seconds": 0.0,
                            "data_prepare_seconds": 0.0,
                            "cuda_transfer_seconds": 0.0,
                            "forward_seconds": 0.0,
                            "metric_seconds": 0.0,
                            "cpu_serialize_seconds": 0.0,
                            "artifact_write_seconds": 0.0,
                            "total_seconds": 0.0,
                        }
                    else:
                        forced = target_eval_with_runtime(
                            variant_runtimes[variant],
                            target_index=sample_idx,
                            seed_memory_state=p0_seed["state"],
                            oracle_mode="query_alignment_report_only",
                            teacher_state=teacher_state,
                            forced_pairs=forced_pairs,
                            artifact_label=f"MCQM_V2_ORACLE_E2E_{variant}_K{k}",
                            cpu_batch_cache=ctx.cpu_batch_cache,
                        )
                        ctx.timing_totals.data_prepare_seconds += float(
                            forced["timings"]["data_prepare_seconds"]
                        )
                        ctx.timing_totals.cuda_transfer_seconds += float(
                            forced["timings"]["cuda_transfer_seconds"]
                        )
                        ctx.timing_totals.forward_seconds += float(
                            forced["timings"]["forward_seconds"]
                        )
                        ctx.timing_totals.metric_seconds += float(
                            forced["timings"]["metric_seconds"]
                        )
                        forced_map = build_target_metric_map(forced)
                        forced_row = forced_map[(sample_idx, 0)]
                        forced_debug = forced["debug_rows"][0] if forced["debug_rows"] else {}
                        row = {
                            "job_id": job_id,
                            "protocol_hash": protocol_hash,
                            "protocol_version": ORACLE_E2E_PROTOCOL_VERSION,
                            "variant": variant,
                            "sample_index": int(sample_idx),
                            "requested_K": int(k),
                            "positive_nonconflicting_pair_count": int(
                                forced_pair_info["positive_nonconflicting_pair_count"]
                            ),
                            "requested_pair_count": int(
                                forced_pair_info["requested_pair_count"]
                            ),
                            "applied_pair_count": int(
                                forced_debug.get(
                                    "actual_applied_pair_count",
                                    forced_debug.get("actual_replacement_count", 0),
                                )
                            ),
                            "requested_forced_pair_count": int(
                                forced_debug.get("requested_forced_pair_count", len(forced_pairs))
                            ),
                            "resolved_forced_pair_count": int(
                                forced_debug.get("resolved_forced_pair_count", 0)
                            ),
                            "seeded_oracle_rows_sha256": seeded_oracle_rows_hash,
                            "seeded_alignment_row_count": int(len(seeded_oracle_rows)),
                            "baseline_iou": float(baseline_row.get("occupied_iou", 0.0)),
                            "oracle_iou": float(forced_row.get("occupied_iou", 0.0)),
                            "delta_IoU": float(forced_row.get("occupied_iou", 0.0)) - float(baseline_row.get("occupied_iou", 0.0)),
                            "baseline_false_free": float(baseline_row.get("false_free_rate", 0.0)),
                            "oracle_false_free": float(forced_row.get("false_free_rate", 0.0)),
                            "delta_false_free": float(forced_row.get("false_free_rate", 0.0)) - float(baseline_row.get("false_free_rate", 0.0)),
                            "baseline_front_false_free": float(baseline_row.get("front_sector_false_free", 0.0)),
                            "oracle_front_false_free": float(forced_row.get("front_sector_false_free", 0.0)),
                            "delta_front_false_free": float(forced_row.get("front_sector_false_free", 0.0)) - float(baseline_row.get("front_sector_false_free", 0.0)),
                            "warmup_start": int(sample_idx - 1),
                            **p0_seed["audit"],
                            "baseline_seed_memory_state_sha256": str(
                                baseline_target["summary"]["seed_audit"]["seed_memory_state_sha256"]
                            ),
                            "forced_seed_memory_state_sha256": str(
                                forced["summary"]["seed_audit"]["seed_memory_state_sha256"]
                            ),
                            "model_build_seconds": 0.0,
                            "checkpoint_load_seconds": 0.0,
                            "data_prepare_seconds": float(forced["timings"]["data_prepare_seconds"]),
                            "cuda_transfer_seconds": float(forced["timings"]["cuda_transfer_seconds"]),
                            "forward_seconds": float(forced["timings"]["forward_seconds"]),
                            "metric_seconds": float(forced["timings"]["metric_seconds"]),
                        }
                        if (
                            int(row["requested_forced_pair_count"])
                            != int(row["resolved_forced_pair_count"])
                            or int(row["requested_forced_pair_count"])
                            != int(row["applied_pair_count"])
                        ):
                            raise RuntimeError("Oracle forced pair application incomplete")
                    row["artifact_write_seconds"] = 0.0
                    row["total_seconds"] = float(time.perf_counter() - row_start)
                    serialize_start = time.perf_counter()
                    serialized_row = json.dumps(row, ensure_ascii=False)
                    row["cpu_serialize_seconds"] = float(time.perf_counter() - serialize_start)
                    write_start = time.perf_counter()
                    append_jsonl_text(partial_path, serialized_row)
                    row["artifact_write_seconds"] = float(time.perf_counter() - write_start)
                    row["total_seconds"] = float(time.perf_counter() - row_start)
                    ctx.timing_totals.cpu_serialize_seconds += float(row["cpu_serialize_seconds"])
                    ctx.timing_totals.artifact_write_seconds += float(row["artifact_write_seconds"])
                    ctx.partial_rows_by_job[job_id] = row
                    ctx.completed_jobs += 1
                    write_oracle_progress(
                        ctx,
                        status="running",
                        stage="forced_forward_completed",
                        variant=variant,
                        sample_index=sample_idx,
                        requested_k=k,
                        last_job_seconds=float(row["total_seconds"]),
                        forward_seconds=float(row.get("forward_seconds", 0.0)),
                        cpu_overhead_seconds=float(row["total_seconds"]) - float(row.get("forward_seconds", 0.0)),
                    )

        final_rows = aggregate_oracle_e2e_partial(ctx.partial_rows_by_job)
        artifact_write_start = time.perf_counter()
        write_json(
            V2_ARTIFACTS_DIR / "oracle_e2e_validation.json",
            {"rows": final_rows},
        )
        write_csv(V2_ARTIFACTS_DIR / "oracle_e2e_validation.csv", final_rows)
        ctx.timing_totals.artifact_write_seconds += time.perf_counter() - artifact_write_start
        ctx.timing_totals.wall_seconds = float(time.time() - ctx.suite_started_at)
        timing_summary = summarize_oracle_timing(ctx.timing_totals)
        write_json(
            V2_ARTIFACTS_DIR / "oracle_e2e_validation.json",
            {"rows": final_rows, "timing_summary": timing_summary},
        )
        decision = decision_from_oracle_rows(
            final_rows,
            oracle_alignment_artifact["rows"],
            preflight["valid_targets"],
        )
        write_json(V2_ARTIFACTS_DIR / "decision.json", decision)
        write_json(V2_ARTIFACTS_DIR / "oracle_e2e_timing_summary.json", timing_summary)
        write_oracle_progress(
            ctx,
            status="completed",
            stage="completed",
            decision_status=decision["status"],
        )
        write_test_report(
            "completed",
            stage="oracle_e2e_only",
            decision_status=decision["status"],
            completed_jobs=int(ctx.completed_jobs),
            total_jobs=int(preflight["forced_job_count"]),
        )
        return {
            "preflight": preflight,
            "timing_summary": timing_summary,
            "completed_jobs": ctx.completed_jobs,
            "total_jobs": int(preflight["forced_job_count"]),
            "decision": decision,
            "progress_path": str(PROGRESS_PATH),
            "partial_path": str(partial_path),
        }
    except Exception as exc:
        write_progress(
            {
                "status": "failed",
                "stage": "failed",
                "completed_jobs": int(
                    ctx.completed_jobs if 'ctx' in locals() else 0
                ),
                "total_jobs": int(
                    ctx.preflight["forced_job_count"] if 'ctx' in locals() else 0
                ),
                "progress_percent": (
                    100.0 * float(ctx.completed_jobs) / float(ctx.preflight["forced_job_count"])
                    if 'ctx' in locals() and ctx.preflight["forced_job_count"] > 0
                    else 0.0
                ),
                "decision_status": "",
                "gpu_allocated_gb": gpu_mem_stats_gb()[0],
                "gpu_reserved_gb": gpu_mem_stats_gb()[1],
                "cpu_rss_gb": float(cpu_rss_gb()),
                "forward_seconds": 0.0,
                "cpu_overhead_seconds": 0.0,
                "last_job_seconds": 0.0,
                "average_job_seconds": 0.0,
                "eta_seconds": 0.0,
                "variant": "",
                "sample_index": None,
                "requested_k": None,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        write_test_report(
            "failed",
            stage="oracle_e2e_only",
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
        raise


def run_mcqm_eval_variant(
    variant: str,
    checkpoint_path: str | Path,
    sample_start: int,
    sample_end: int,
    *,
    capture_enabled: bool = False,
) -> dict[str, Any]:
    mcqm_cfg, meta = mcqm_variant_cfg(variant)
    _, dataset, model, runtime_meta = build_runtime(
        train=False,
        mcqm_cfg=mcqm_cfg,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
        checkpoint_path=resolve_checkpoint_path(checkpoint_path),
    )
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, query_holder) if capture_enabled else None
    sectors = get_sector_masks_cached()
    per_row = []
    provenance_rows = []
    debug_rows = []
    protocol_rows = []
    capture_rows = []
    native_quality_values = []
    memory_quality_values = []
    quality_gain_values = []
    reset_runtime_state(model)
    reset_mcqm_memory(model)
    initial_memory_empty = mcqm_memory_is_empty(model)
    reset_count = 1
    scene_reset_count = 0
    prev_scene = None
    try:
        for idx in range(sample_start, sample_end + 1):
            scene_token = dataset.data_infos[idx].get("scene_token", "")
            if prev_scene is not None and scene_token != prev_scene:
                reset_runtime_state(model)
                reset_mcqm_memory(model)
                reset_count += 1
                scene_reset_count += 1
            prev_scene = scene_token
            sample_raw, batch = extract_inputs(dataset, idx)
            batch = enrich_batch_scene_meta(batch, dataset, idx)
            batch, perturb_manifest = apply_perturbation_with_manifest_to_batch(batch, "A10_drop_front_triplet")
            sample_unwrapped = sw2.unwrap(sample_raw)
            protocol_rows.append(sample_protocol_record(dataset, idx, sample_unwrapped, batch, perturb_manifest) | {"variant": variant})
            inputs = sw2.move_to_cuda(batch)
            with torch.inference_mode():
                out = model(return_loss=False, rescale=True, **inputs)
            mcqm_debug = getattr(model, "latest_mcqm_debug", {})
            row0 = (mcqm_debug.get("debug_rows", [{}]) or [{}])[0]
            native_q = mcqm_debug.get("native_quality", None)
            if native_q is not None:
                native_quality_values.extend(native_q.detach().reshape(-1).float().cpu().tolist())
            memory_q = mcqm_debug.get("memory_quality_values", None)
            if memory_q is not None:
                memory_quality_values.extend(memory_q.detach().reshape(-1).float().cpu().tolist())
            quality_gain = mcqm_debug.get("quality_gain_values", None)
            if quality_gain is not None:
                quality_gain_values.extend(quality_gain.detach().reshape(-1).float().cpu().tolist())
            debug_rows.append({
                "sample_index": idx,
                "variant": variant,
                "timestamp": float(row0.get("timestamp", float("nan"))),
                "delta_t": float(row0.get("delta_t", float("nan"))),
                "callback_invoked": int(bool(row0.get("callback_invoked", False))),
                "callback_layer_idx": int(row0.get("callback_layer_idx", -1)),
                "memory_raw_count": int(row0.get("memory_raw_count", 0)),
                "memory_native_source_count": int(row0.get("memory_native_source_count", 0)),
                "memory_rejected_recursive_count": int(row0.get("memory_rejected_recursive_count", 0)),
                "raw_static_candidates": int(row0.get("raw_static_candidates", 0)),
                "raw_dynamic_candidates": int(row0.get("raw_dynamic_candidates", 0)),
                "valid_static_candidates": int(row0.get("valid_static_candidates", 0)),
                "valid_dynamic_candidates": int(row0.get("valid_dynamic_candidates", 0)),
                "selected_static_memory": int(row0.get("selected_static_memory", 0)),
                "selected_dynamic_memory": int(row0.get("selected_dynamic_memory", 0)),
                "actual_selected_static_memory": int(row0.get("actual_selected_static_memory", 0)),
                "actual_selected_dynamic_memory": int(row0.get("actual_selected_dynamic_memory", 0)),
                "actual_selected_uncertain_memory": int(row0.get("actual_selected_uncertain_memory", 0)),
                "actual_replacement_count": int(row0.get("actual_replacement_count", 0)),
                "memory_candidate_count": int(row0.get("memory_candidate_count", 0)),
                "memory_valid_count": int(row0.get("memory_valid_count", 0)),
                "replacement_margin": float(row0.get("replacement_margin", -1.0)),
                "quality_gain_mean": float(row0.get("quality_gain_mean", 0.0)),
                "current_query_count": int(row0.get("current_query_count", native_q.shape[1] if native_q is not None else -1)),
                "local_matched_count": int(row0.get("local_matched_count", 0)),
                "semantic_rejected_count": int(row0.get("semantic_rejected_count", 0)),
                "distance_rejected_count": int(row0.get("distance_rejected_count", 0)),
                "already_occupied_native_rejected_count": int(row0.get("already_occupied_native_rejected_count", 0)),
                "native_quality_rejected_count": int(row0.get("native_quality_rejected_count", 0)),
                "memory_native_match_distance_mean": float(row0.get("memory_native_match_distance_mean", 0.0)),
                "projector_gate_value": float(row0.get("projector_gate_value", 0.0)),
                "projector_residual_norm_mean": float(row0.get("projector_residual_norm_mean", 0.0)),
                "mcqm_full_time_mode": str(row0.get("mcqm_full_time_mode", mcqm_cfg.get("mcqm_full_time_mode", ""))),
                "mcqm_full_residual_mode": str(row0.get("mcqm_full_residual_mode", mcqm_cfg.get("mcqm_full_residual_mode", ""))),
                "memory_bank_repropagated_query_count": int(row0.get("memory_bank_repropagated_query_count", 0)),
            })
            pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sample_unwrapped, out)
            if capture_enabled:
                final_pred = extract_final_pred_tensors(query_holder)
                parity_capture = extract_mcqm_parity_capture(model)
                capture_rows.append({
                    "sample_index": idx,
                    "variant": variant,
                    "actual_replacement_count": int(row0.get("actual_replacement_count", 0)),
                    "final_cls_scores": final_pred["cls_scores"],
                    "final_refine_pts": final_pred["refine_pts"],
                    "final_occ_dense": pred_temporal[0].cpu(),
                    "parity_capture": parity_capture,
                })
            for horizon_s in CORE_HORIZONS:
                pred_h = pred_temporal[horizon_s].cpu()
                gt_h = gt_temporal[horizon_s].cpu()
                gt0 = gt_temporal[0].cpu()
                row = sw12b.build_eval_row(pred_h, gt_h, gt0, "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=None)
                row.update({"sample_index": idx, "variant": variant, "horizon_s": horizon_s})
                per_row.append(row)
                if horizon_s == 0:
                    provenance_rows.append({"sample_index": idx, "variant": variant, **summarize_provenance(model, gt_h, horizon_s)})
    finally:
        if original_forward is not None:
            model.forward_backbone = original_forward  # type: ignore[assignment]
    summary = aggregate_rows(per_row)
    summary["variant"] = variant
    summary["uses_mcqm"] = meta["uses_mcqm"]
    summary["provenance"] = aggregate_rows(provenance_rows) if provenance_rows else {}
    summary["debug"] = aggregate_rows(debug_rows) if debug_rows else {}
    summary["native_quality"] = quantile_stats(native_quality_values)
    summary["memory_quality"] = quantile_stats(memory_quality_values)
    summary["quality_gain"] = quantile_stats(quality_gain_values)
    summary["runtime_reset"] = {
        "reset_count": int(reset_count),
        "scene_reset_count": int(scene_reset_count),
        "initial_memory_state_audit": initial_memory_empty,
    }
    return {
        "summary": summary,
        "rows": per_row,
        "provenance_rows": provenance_rows,
        "debug_rows": debug_rows,
        "protocol_rows": protocol_rows,
        "capture_rows": capture_rows,
        "checkpoint": runtime_meta,
    }


def run_baseline_eval(
    sample_start: int,
    sample_end: int,
    checkpoint_path: str | Path,
    *,
    degraded: bool,
    capture_enabled: bool = False,
) -> dict[str, Any]:
    sectors = get_sector_masks_cached()
    label = "B1" if degraded else "B0"
    perturb = "A10_drop_front_triplet" if degraded else "A0_clean"
    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    _, dataset, model, runtime_meta = build_runtime(
        train=False,
        mcqm_cfg=None,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
        checkpoint_path=resolved_checkpoint_path,
    )
    query_holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, query_holder) if capture_enabled else None
    reset_runtime_state(model)
    initial_memory_empty = mcqm_memory_is_empty(model)
    reset_count = 1
    scene_reset_count = 0
    prev_scene = None
    per_row = []
    protocol_rows = []
    capture_rows = []
    try:
        for idx in range(sample_start, sample_end + 1):
            scene_token = dataset.data_infos[idx].get("scene_token", "")
            if prev_scene is not None and scene_token != prev_scene:
                reset_runtime_state(model)
                reset_count += 1
                scene_reset_count += 1
            prev_scene = scene_token
            sample_raw, batch = extract_inputs(dataset, idx)
            batch = enrich_batch_scene_meta(batch, dataset, idx)
            batch, perturb_manifest = apply_perturbation_with_manifest_to_batch(batch, perturb)
            sample_unwrapped = sw2.unwrap(sample_raw)
            protocol_rows.append(sample_protocol_record(dataset, idx, sample_unwrapped, batch, perturb_manifest) | {"variant": label})
            inputs = sw2.move_to_cuda(batch)
            with torch.inference_mode():
                out = model(return_loss=False, rescale=True, **inputs)
            pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sample_unwrapped, out)
            if capture_enabled:
                final_pred = extract_final_pred_tensors(query_holder)
                capture_rows.append({
                    "sample_index": idx,
                    "variant": label,
                    "final_cls_scores": final_pred["cls_scores"],
                    "final_refine_pts": final_pred["refine_pts"],
                    "final_occ_dense": pred_temporal[0].cpu(),
                })
            for horizon_s in CORE_HORIZONS:
                pred_h = pred_temporal[horizon_s].cpu()
                gt_h = gt_temporal[horizon_s].cpu()
                gt0 = gt_temporal[0].cpu()
                row = sw12b.build_eval_row(pred_h, gt_h, gt0, perturb, horizon_s, sectors, baseline_pred=None)
                row.update({"sample_index": idx, "variant": label, "horizon_s": horizon_s})
                per_row.append(row)
    finally:
        if original_forward is not None:
            model.forward_backbone = original_forward  # type: ignore[assignment]
    summary = {"variant": label, **aggregate_rows(per_row)}
    summary["runtime_reset"] = {
        "reset_count": int(reset_count),
        "scene_reset_count": int(scene_reset_count),
        "initial_memory_state_audit": initial_memory_empty,
    }
    return {
        "summary": summary,
        "rows": per_row,
        "protocol_rows": protocol_rows,
        "capture_rows": capture_rows,
        "checkpoint": runtime_meta,
    }


def run_r8_eval(
    sample_start: int,
    sample_end: int,
    checkpoint_path: str | Path,
    label: str,
    *,
    capture_enabled: bool = False,
) -> dict[str, Any]:
    from mmcv.runner import load_checkpoint

    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    _, dataset, model = sw13a.build_runtime()
    full_audit = checkpoint_state_audit_full(model, resolved_checkpoint_path)
    if full_audit["normalization_collisions"]:
        raise RuntimeError(
            f"duplicate checkpoint keys after normalization for {resolved_checkpoint_path}: "
            f"{list(full_audit['normalization_collisions'].items())[:5]}"
        )
    core_unexpected_keys = [
        key for key in full_audit["unexpected_keys"]
        if not is_mcqm_only_key(key)
    ]
    if full_audit["shape_mismatches"] or full_audit["missing_keys"] or core_unexpected_keys:
        raise RuntimeError(
            f"R8 checkpoint load audit failed for {resolved_checkpoint_path}: "
            f"missing={len(full_audit['missing_keys'])} "
            f"core_unexpected={len(core_unexpected_keys)} "
            f"shape_mismatch={len(full_audit['shape_mismatches'])}"
        )
    checkpoint_obj = load_checkpoint(model, str(resolved_checkpoint_path), map_location="cpu", strict=False)
    normalized_checkpoint_state = full_audit.pop("normalized_state_dict")
    content_audit = matched_parameter_content_audit(
        model,
        normalized_checkpoint_state,
        excluded_key_predicate=is_mcqm_only_key,
    )
    del normalized_checkpoint_state
    if content_audit["content_mismatch_count"] != 0:
        raise RuntimeError(
            f"R8 checkpoint content audit failed for {resolved_checkpoint_path}: "
            f"content_mismatch={content_audit['content_mismatch_count']}"
        )
    if (
        content_audit["matched_key_coverage"] != 1.0
        or content_audit["matched_numel_coverage"] != 1.0
    ):
        raise RuntimeError(
            f"R8 checkpoint coverage audit failed for {resolved_checkpoint_path}: "
            f"matched_key_coverage={content_audit['matched_key_coverage']:.6f} "
            f"matched_numel_coverage={content_audit['matched_numel_coverage']:.6f}"
        )
    variants = {v.label: v for v in sw13a.make_variants()}
    r8 = variants["R8_camera_group_repair_front_triplet"]
    sectors = get_sector_masks_cached()
    per_row = []
    protocol_rows = []
    capture_rows = []
    reset_count = 0
    scene_reset_count = 0
    reset_runtime_state(model)
    reset_r8_cache(model)
    initial_memory_empty = r8_cache_is_empty(model)
    reset_count += 1
    runtime_reset_count = 1
    r8_cache_reset_count = 1
    prev_scene = None
    for idx in range(sample_start, sample_end + 1):
        scene_token = dataset.data_infos[idx].get("scene_token", "")
        if prev_scene is not None and scene_token != prev_scene:
            reset_runtime_state(model)
            reset_r8_cache(model)
            reset_count += 1
            scene_reset_count += 1
            runtime_reset_count += 1
            r8_cache_reset_count += 1
        prev_scene = scene_token
        sample_raw, batch_clean = sw2.extract_sample_batch(dataset, idx, collate_fn)
        sample_unwrapped = sw13a.sw2.unwrap(sample_raw)
        batch_clean = sw13a.sw2.move_to_cuda(batch_clean)
        cache_path, _, cache = sw13a.extract_clean_memory_for_sample(model, batch_clean, sample_unwrapped, idx)
        _ = cache_path
        _, batch_deg = sw2.extract_sample_batch(dataset, idx, collate_fn)
        batch_deg = enrich_batch_scene_meta(batch_deg, dataset, idx)
        batch_deg, perturb_manifest = apply_perturbation_with_manifest_to_batch(batch_deg, "A10_drop_front_triplet")
        protocol_rows.append(sample_protocol_record(dataset, idx, sample_unwrapped, batch_deg, perturb_manifest) | {"variant": label})
        raw_result, per_h, _ = sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache, r8, "A10_drop_front_triplet")
        if capture_enabled:
            pred_temporal, _, _ = sw2.extract_standard_tensors(sample_unwrapped, raw_result)
            capture_rows.append({
                "sample_index": idx,
                "variant": label,
                "final_cls_scores": per_h[0]["pred_dict"]["cls_scores"].detach().cpu(),
                "final_refine_pts": per_h[0]["pred_dict"]["refine_pts"].detach().cpu(),
                "final_occ_dense": pred_temporal[0].cpu(),
            })
        for horizon_s in CORE_HORIZONS:
            pred_h = sw13a.dense_occ_from_pred(model.pts_bbox_head, per_h[horizon_s]["pred_dict"]).long()
            gt_h = per_h[horizon_s]["gt_h"].long()
            gt0 = per_h[horizon_s]["gt0"].long()
            row = sw12b.build_eval_row(pred_h, gt_h, gt0, "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=None)
            row.update({"sample_index": idx, "variant": label, "horizon_s": horizon_s})
            per_row.append(row)
    summary = {"variant": label, **aggregate_rows(per_row)}
    summary["runtime_reset"] = {
        "reset_count": int(reset_count),
        "scene_reset_count": int(scene_reset_count),
        "runtime_reset_count": int(runtime_reset_count),
        "r8_cache_reset_count": int(r8_cache_reset_count),
        "initial_memory_state_audit": initial_memory_empty,
    }
    return {
        "summary": summary,
        "rows": per_row,
        "protocol_rows": protocol_rows,
        "capture_rows": capture_rows,
        "checkpoint": {
            **checkpoint_file_info(checkpoint_path, checkpoint_obj),
            "state_dict_audit": {
                "missing_key_count": len(full_audit["missing_keys"]),
                "core_unexpected_key_count": len(core_unexpected_keys),
                "mcqm_only_unexpected_key_count": len(full_audit["unexpected_keys"]) - len(core_unexpected_keys),
                "shape_mismatch_count": len(full_audit["shape_mismatches"]),
                "normalization_collision_count": len(full_audit["normalization_collisions"]),
                "missing_keys_preview": full_audit["missing_keys"][:20],
                "core_unexpected_keys_preview": core_unexpected_keys[:20],
                "mcqm_only_unexpected_keys_preview": [
                    key for key in full_audit["unexpected_keys"]
                    if is_mcqm_only_key(key)
                ][:20],
                "shape_mismatch_preview": full_audit["shape_mismatches"][:20],
                "normalization_collision_preview": dict(list(full_audit["normalization_collisions"].items())[:5]),
            },
            "core_model_state_sha256": model_state_sha256(model, excluded_prefixes=("mcqm_", "mcqm_memory_projector.")),
            "content_audit": content_audit,
        },
    }


def relabel_eval_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    result = dict(result)
    summary = dict(result["summary"])
    summary["variant"] = label
    result["summary"] = summary
    for key in ["rows", "protocol_rows", "capture_rows", "provenance_rows", "debug_rows"]:
        rows = result.get(key, None)
        if rows is None:
            continue
        new_rows = []
        for row in rows:
            if isinstance(row, dict):
                upd = dict(row)
                upd["variant"] = label
                new_rows.append(upd)
            else:
                new_rows.append(row)
        result[key] = new_rows
    return result


def sample_protocol_consistency(*result_groups: dict[str, Any]) -> list[dict[str, Any]]:
    by_sample: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in result_groups:
        for row in group.get("protocol_rows", []):
            by_sample[int(row["sample_index"])][str(row["variant"])] = row
    out = []
    for sample_index, rows_by_variant in sorted(by_sample.items()):
        rows = list(rows_by_variant.values())
        base = rows[0]
        out.append({
            "sample_index": sample_index,
            "variant_count": len(rows),
            "gt_occ_hash_consistent": all(row["gt_occ_sha256"] == base["gt_occ_sha256"] for row in rows),
            "gt_temporal_hash_consistent": all(row["gt_temporal_sha256"] == base["gt_temporal_sha256"] for row in rows),
            "degraded_input_hash_consistent": all(row["degraded_input_sha256"] == base["degraded_input_sha256"] for row in rows),
            "camera_mask_hash_consistent": all(row["camera_mask_sha256"] == base["camera_mask_sha256"] for row in rows),
            "perturbation_hash_consistent": all(row["perturbation_config_hash"] == base["perturbation_config_hash"] for row in rows),
            "img_metas_hash_consistent": all(row["img_metas_hash"] == base["img_metas_hash"] for row in rows),
            "critical_input_consistent": (
                all(row["gt_occ_sha256"] == base["gt_occ_sha256"] for row in rows)
                and all(row["gt_temporal_sha256"] == base["gt_temporal_sha256"] for row in rows)
                and all(row["degraded_input_sha256"] == base["degraded_input_sha256"] for row in rows)
                and all(row["camera_mask_sha256"] == base["camera_mask_sha256"] for row in rows)
            ),
        })
    return out


def capture_row_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["sample_index"]): row for row in result.get("capture_rows", [])}


def final_output_parity_between_results(
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[dict[str, Any]]:
    left_map = capture_row_map(left)
    right_map = capture_row_map(right)
    out = []
    for sample_index in sorted(set(left_map.keys()) & set(right_map.keys())):
        lrow = left_map[sample_index]
        rrow = right_map[sample_index]
        cls_cmp = compare_tensors(lrow.get("final_cls_scores"), rrow.get("final_cls_scores"))
        pts_cmp = compare_tensors(lrow.get("final_refine_pts"), rrow.get("final_refine_pts"))
        dense_cmp = compare_tensors(lrow.get("final_occ_dense"), rrow.get("final_occ_dense"), atol=0.0, rtol=0.0)
        out.append({
            "sample_index": sample_index,
            "left_variant": str(lrow.get("variant")),
            "right_variant": str(rrow.get("variant")),
            "final_query_cls_scores_max_abs_diff": cls_cmp["max_abs_diff"],
            "final_query_cls_scores_mean_abs_diff": cls_cmp["mean_abs_diff"],
            "final_query_cls_scores_allclose": cls_cmp["allclose"],
            "final_refine_pts_max_abs_diff": pts_cmp["max_abs_diff"],
            "final_refine_pts_mean_abs_diff": pts_cmp["mean_abs_diff"],
            "final_refine_pts_allclose": pts_cmp["allclose"],
            "final_occ_dense_max_abs_diff": dense_cmp["max_abs_diff"],
            "final_occ_dense_mean_abs_diff": dense_cmp["mean_abs_diff"],
            "final_occ_dense_allclose": dense_cmp["allclose"],
        })
    return out


def mcqm_callback_parity_between_results(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    zero_replacement_only_right: bool = False,
) -> list[dict[str, Any]]:
    left_map = capture_row_map(left)
    right_map = capture_row_map(right)
    out = []
    for sample_index in sorted(set(left_map.keys()) & set(right_map.keys())):
        lrow = left_map[sample_index]
        rrow = right_map[sample_index]
        if zero_replacement_only_right and int(rrow.get("actual_replacement_count", -1)) != 0:
            continue
        lcap = lrow.get("parity_capture")
        rcap = rrow.get("parity_capture")
        if not isinstance(lcap, dict):
            out.append({
                "sample_index": sample_index,
                "left_variant": str(lrow.get("variant")),
                "right_variant": str(rrow.get("variant")),
                "comparison_available": False,
                "reason": "left parity_capture missing",
            })
            continue
        if not isinstance(rcap, dict):
            out.append({
                "sample_index": sample_index,
                "left_variant": str(lrow.get("variant")),
                "right_variant": str(rrow.get("variant")),
                "comparison_available": False,
                "reason": "right parity_capture missing",
            })
            continue
        l1_feat_cmp = compare_tensors(lcap.get("layer1_query_feat"), rcap.get("layer1_query_feat"))
        l1_pts_cmp = compare_tensors(lcap.get("layer1_query_points"), rcap.get("layer1_query_points"))
        l2_feat_cmp = compare_tensors(lcap.get("layer2_query_feat"), rcap.get("layer2_query_feat"))
        l2_pts_cmp = compare_tensors(lcap.get("layer2_query_points"), rcap.get("layer2_query_points"))
        in_cls_cmp = compare_tensors(lcap.get("callback_input_cls_score"), rcap.get("callback_input_cls_score"))
        out_cls_cmp = compare_tensors(lcap.get("callback_output_cls_score"), rcap.get("callback_output_cls_score"))
        out.append({
            "sample_index": sample_index,
            "left_variant": str(lrow.get("variant")),
            "right_variant": str(rrow.get("variant")),
            "right_actual_replacement_count": int(rrow.get("actual_replacement_count", -1)),
            "comparison_available": True,
            "layer1_query_feat_max_abs_diff": l1_feat_cmp["max_abs_diff"],
            "layer1_query_feat_mean_abs_diff": l1_feat_cmp["mean_abs_diff"],
            "layer1_query_feat_allclose": l1_feat_cmp["allclose"],
            "layer1_query_feat_shape_a": l1_feat_cmp["shape_a"],
            "layer1_query_feat_shape_b": l1_feat_cmp["shape_b"],
            "layer1_query_points_max_abs_diff": l1_pts_cmp["max_abs_diff"],
            "layer1_query_points_mean_abs_diff": l1_pts_cmp["mean_abs_diff"],
            "layer1_query_points_allclose": l1_pts_cmp["allclose"],
            "layer1_query_points_shape_a": l1_pts_cmp["shape_a"],
            "layer1_query_points_shape_b": l1_pts_cmp["shape_b"],
            "layer2_query_feat_max_abs_diff": l2_feat_cmp["max_abs_diff"],
            "layer2_query_feat_mean_abs_diff": l2_feat_cmp["mean_abs_diff"],
            "layer2_query_feat_allclose": l2_feat_cmp["allclose"],
            "layer2_query_feat_shape_a": l2_feat_cmp["shape_a"],
            "layer2_query_feat_shape_b": l2_feat_cmp["shape_b"],
            "layer2_query_points_max_abs_diff": l2_pts_cmp["max_abs_diff"],
            "layer2_query_points_mean_abs_diff": l2_pts_cmp["mean_abs_diff"],
            "layer2_query_points_allclose": l2_pts_cmp["allclose"],
            "layer2_query_points_shape_a": l2_pts_cmp["shape_a"],
            "layer2_query_points_shape_b": l2_pts_cmp["shape_b"],
            "callback_input_cls_score_max_abs_diff": in_cls_cmp["max_abs_diff"],
            "callback_input_cls_score_mean_abs_diff": in_cls_cmp["mean_abs_diff"],
            "callback_input_cls_score_allclose": in_cls_cmp["allclose"],
            "callback_output_cls_score_max_abs_diff": out_cls_cmp["max_abs_diff"],
            "callback_output_cls_score_mean_abs_diff": out_cls_cmp["mean_abs_diff"],
            "callback_output_cls_score_allclose": out_cls_cmp["allclose"],
        })
    return out


def fair_summary_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    debug = summary.get("debug", {})
    prov = summary.get("provenance", {})
    ckpt = result.get("checkpoint", {})
    memory_correct = float(prov.get("mean_memory_correct_voxels", 0.0))
    memory_fp = float(prov.get("mean_memory_fp_voxels", 0.0))
    return {
        "Variant": label,
        "Checkpoint SHA256": ckpt.get("checkpoint_sha256", ""),
        "IoU": float(summary.get("mean_occupied_iou", 0.0)),
        "False-free": float(summary.get("mean_false_free_rate", 0.0)),
        "Front false-free": float(summary.get("mean_front_sector_false_free", 0.0)),
        "Mean replacement": float(debug.get("mean_actual_replacement_count", 0.0)),
        "Memory correct": memory_correct,
        "Memory FP": memory_fp,
        "FP/Correct": (memory_fp / memory_correct) if memory_correct > 0 else float("inf"),
        "Actual static selected": float(debug.get("mean_actual_selected_static_memory", 0.0)),
        "Actual dynamic selected": float(debug.get("mean_actual_selected_dynamic_memory", 0.0)),
        "Memory-bank repropagated count": float(debug.get("mean_memory_bank_repropagated_query_count", 0.0)),
    }


def strip_raw_captures(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    if "capture_rows" in out:
        compact_rows = []
        for row in out["capture_rows"]:
            compact_rows.append({
                "sample_index": int(row["sample_index"]),
                "variant": str(row["variant"]),
                "actual_replacement_count": int(row.get("actual_replacement_count", -1)),
            })
        out["capture_rows"] = compact_rows
    return out


def validate_n1_callback_audit(
    debug_rows: list[dict[str, Any]],
    expected_frame_count: int,
    expected_layer_idx: int,
) -> dict[str, Any]:
    observed_layers = sorted({
        int(row.get("callback_layer_idx", -1))
        for row in debug_rows
    })
    errors = []
    if len(debug_rows) != expected_frame_count:
        errors.append(
            f"expected {expected_frame_count} rows, got {len(debug_rows)}"
        )
    if not all(int(row.get("callback_invoked", 0)) == 1 for row in debug_rows):
        errors.append("callback not invoked on every frame")
    if observed_layers != [expected_layer_idx]:
        errors.append(
            f"expected layer {expected_layer_idx}, observed {observed_layers}"
        )
    if not all(int(row.get("actual_replacement_count", -1)) == 0 for row in debug_rows):
        errors.append("N1 contains nonzero replacement")
    if errors:
        raise RuntimeError("N1 callback audit failed: " + "; ".join(errors))
    return {
        "expected_frame_count": expected_frame_count,
        "observed_frame_count": len(debug_rows),
        "expected_callback_layer_idx": expected_layer_idx,
        "observed_callback_layer_idx_values": observed_layers,
        "all_callback_invoked": True,
        "all_zero_replacement": True,
    }


def run_full_eval(sample_start: int, sample_end: int, checkpoint_path: str | Path) -> dict[str, Any]:
    baseline = {
        "B0": run_baseline_eval(sample_start, sample_end, CHECKPOINT_PATH, degraded=False),
        "B1": run_baseline_eval(sample_start, sample_end, CHECKPOINT_PATH, degraded=True),
    }
    r8 = run_r8_eval(sample_start, sample_end, CHECKPOINT_PATH, "B2")
    mcqm_results = {}
    for variant in ["A2"]:
        mcqm_results[variant] = run_mcqm_eval_variant(variant, checkpoint_path, sample_start, sample_end)
    table = []
    for label in ["B0", "B1"]:
        table.append(baseline[label]["summary"])
    table.append(r8["summary"])
    for label in ["A2"]:
        row = dict(mcqm_results[label]["summary"])
        b2 = r8["summary"]
        for metric in ["mean_occupied_iou", "mean_false_free_rate", "mean_false_occupied_rate", "mean_pred_gt_occupied_ratio", "mean_front_sector_false_free"]:
            if metric in row and metric in b2:
                row[f"{metric}_delta_vs_B2"] = float(row[metric]) - float(b2[metric])
        table.append(row)
    result = {
        "sample_range": [sample_start, sample_end],
        "checkpoint_path": str(checkpoint_path),
        "table": table,
        "baseline": baseline,
        "r8": r8,
        "mcqm": mcqm_results,
    }
    write_json(
        ARTIFACTS_DIR / "eval_results.json",
        {
            **result,
            "baseline": {k: strip_raw_captures(v) for k, v in baseline.items()},
            "r8": strip_raw_captures(r8),
            "mcqm": {k: strip_raw_captures(v) for k, v in mcqm_results.items()},
        },
    )
    return result


def run_fix_eval_suite(sample_start: int, sample_end: int, checkpoint_path: str | Path) -> dict[str, Any]:
    original_ckpt = CHECKPOINT_PATH
    stage_d_ckpt = resolve_checkpoint_path(checkpoint_path)
    legacy_path_audit = {
        "B1_old_path_default_requested": str(CHECKPOINT_PATH),
        "B1_old_path_default_resolved": str(resolve_checkpoint_path(CHECKPOINT_PATH)),
        "B1_old_path_default_sha256": file_sha256(resolve_checkpoint_path(CHECKPOINT_PATH)),
        "R8_old_path_default_requested": str(sw13a.BASE_CHECKPOINT_PATH),
        "R8_old_path_default_resolved": str(resolve_checkpoint_path(sw13a.BASE_CHECKPOINT_PATH)),
        "R8_old_path_default_sha256": file_sha256(resolve_checkpoint_path(sw13a.BASE_CHECKPOINT_PATH)),
    }

    original_results = {
        "B0_ORIGINAL_CLEAN": relabel_eval_result(run_baseline_eval(sample_start, sample_end, original_ckpt, degraded=False, capture_enabled=False), "B0_ORIGINAL_CLEAN"),
        "B1_ORIGINAL_DEGRADED": relabel_eval_result(run_baseline_eval(sample_start, sample_end, original_ckpt, degraded=True, capture_enabled=False), "B1_ORIGINAL_DEGRADED"),
        "B2_ORIGINAL_R8": run_r8_eval(sample_start, sample_end, original_ckpt, "B2_ORIGINAL_R8"),
    }

    stage_d_results = {
        "B0S_STAGE_D_CLEAN": relabel_eval_result(run_baseline_eval(sample_start, sample_end, stage_d_ckpt, degraded=False, capture_enabled=False), "B0S_STAGE_D_CLEAN"),
        "B1S_STAGE_D_DEGRADED": relabel_eval_result(run_baseline_eval(sample_start, sample_end, stage_d_ckpt, degraded=True, capture_enabled=True), "B1S_STAGE_D_DEGRADED"),
        "B2S_STAGE_D_R8": run_r8_eval(sample_start, sample_end, stage_d_ckpt, "B2S_STAGE_D_R8", capture_enabled=False),
        "N0_STAGE_D_BASELINE_REPEAT": relabel_eval_result(run_baseline_eval(sample_start, sample_end, stage_d_ckpt, degraded=True, capture_enabled=True), "N0_STAGE_D_BASELINE_REPEAT"),
        "N1_STAGE_D_CALLBACK_NO_INJECTION": relabel_eval_result(run_mcqm_eval_variant("N1", stage_d_ckpt, sample_start, sample_end, capture_enabled=True), "N1_STAGE_D_CALLBACK_NO_INJECTION"),
        "A2": run_mcqm_eval_variant("A2", stage_d_ckpt, sample_start, sample_end, capture_enabled=False),
        "A3": run_mcqm_eval_variant("A3", stage_d_ckpt, sample_start, sample_end, capture_enabled=False),
        "A4A_GLOBAL_SAME_CLASS": relabel_eval_result(run_mcqm_eval_variant("A4A", stage_d_ckpt, sample_start, sample_end, capture_enabled=False), "A4A_GLOBAL_SAME_CLASS"),
        "A4B_GLOBAL_ANY_CLASS": relabel_eval_result(run_mcqm_eval_variant("A4B", stage_d_ckpt, sample_start, sample_end, capture_enabled=False), "A4B_GLOBAL_ANY_CLASS"),
        "T0": run_mcqm_eval_variant("T0", stage_d_ckpt, sample_start, sample_end, capture_enabled=False),
        "T1": run_mcqm_eval_variant("T1", stage_d_ckpt, sample_start, sample_end, capture_enabled=False),
        "T2": run_mcqm_eval_variant("T2", stage_d_ckpt, sample_start, sample_end, capture_enabled=False),
        "T3": run_mcqm_eval_variant("T3", stage_d_ckpt, sample_start, sample_end, capture_enabled=True),
    }

    original_table = [fair_summary_row(label, result) for label, result in original_results.items()]
    stage_d_table = [fair_summary_row(label, result) for label, result in stage_d_results.items()]

    n1_summary = stage_d_results["N1_STAGE_D_CALLBACK_NO_INJECTION"]["summary"]
    n1_iou = float(n1_summary.get("mean_occupied_iou", 0.0))
    n1_ff = float(n1_summary.get("mean_false_free_rate", 0.0))
    n1_front_ff = float(n1_summary.get("mean_front_sector_false_free", 0.0))
    delta_vs_n1 = []
    for label in ["A2", "A3", "A4A_GLOBAL_SAME_CLASS", "A4B_GLOBAL_ANY_CLASS", "T0", "T1", "T2", "T3"]:
        summary = stage_d_results[label]["summary"]
        delta_vs_n1.append({
            "Variant": label,
            "delta_IoU": float(summary.get("mean_occupied_iou", 0.0)) - n1_iou,
            "delta_false_free": float(summary.get("mean_false_free_rate", 0.0)) - n1_ff,
            "delta_front_false_free": float(summary.get("mean_front_sector_false_free", 0.0)) - n1_front_ff,
        })

    b1s_summary = stage_d_results["B1S_STAGE_D_DEGRADED"]["summary"]
    b2s_summary = stage_d_results["B2S_STAGE_D_R8"]["summary"]
    b2s_delta_vs_b1s = {
        "delta_IoU": float(b2s_summary.get("mean_occupied_iou", 0.0)) - float(b1s_summary.get("mean_occupied_iou", 0.0)),
        "delta_false_free": float(b2s_summary.get("mean_false_free_rate", 0.0)) - float(b1s_summary.get("mean_false_free_rate", 0.0)),
        "delta_front_false_free": float(b2s_summary.get("mean_front_sector_false_free", 0.0)) - float(b1s_summary.get("mean_front_sector_false_free", 0.0)),
    }

    checkpoint_sha_set = {str(result["checkpoint"]["checkpoint_sha256"]) for result in stage_d_results.values()}
    expected_frame_count = sample_end - sample_start + 1
    n1_debug = stage_d_results["N1_STAGE_D_CALLBACK_NO_INJECTION"]["debug_rows"]
    expected_callback_layer_idx = int(mcqm_variant_cfg("N1")[0].get("injection_after_layer_idx", 1))
    n1_callback_audit = validate_n1_callback_audit(
        n1_debug,
        expected_frame_count=expected_frame_count,
        expected_layer_idx=expected_callback_layer_idx,
    )
    protocol_consistency = sample_protocol_consistency(
        stage_d_results["B1S_STAGE_D_DEGRADED"],
        stage_d_results["B2S_STAGE_D_R8"],
        stage_d_results["N0_STAGE_D_BASELINE_REPEAT"],
        stage_d_results["N1_STAGE_D_CALLBACK_NO_INJECTION"],
        stage_d_results["A2"],
        stage_d_results["T3"],
    )
    b1s_vs_n0 = final_output_parity_between_results(stage_d_results["B1S_STAGE_D_DEGRADED"], stage_d_results["N0_STAGE_D_BASELINE_REPEAT"])
    n0_vs_n1 = final_output_parity_between_results(stage_d_results["N0_STAGE_D_BASELINE_REPEAT"], stage_d_results["N1_STAGE_D_CALLBACK_NO_INJECTION"])
    t3_vs_n1_zero = mcqm_callback_parity_between_results(stage_d_results["N1_STAGE_D_CALLBACK_NO_INJECTION"], stage_d_results["T3"], zero_replacement_only_right=True)

    checkpoint_effect = {
        "B0_stage_d_minus_original": {
            "delta_IoU": float(stage_d_results["B0S_STAGE_D_CLEAN"]["summary"].get("mean_occupied_iou", 0.0)) - float(original_results["B0_ORIGINAL_CLEAN"]["summary"].get("mean_occupied_iou", 0.0)),
            "delta_false_free": float(stage_d_results["B0S_STAGE_D_CLEAN"]["summary"].get("mean_false_free_rate", 0.0)) - float(original_results["B0_ORIGINAL_CLEAN"]["summary"].get("mean_false_free_rate", 0.0)),
            "delta_front_false_free": float(stage_d_results["B0S_STAGE_D_CLEAN"]["summary"].get("mean_front_sector_false_free", 0.0)) - float(original_results["B0_ORIGINAL_CLEAN"]["summary"].get("mean_front_sector_false_free", 0.0)),
        },
        "B1_stage_d_minus_original": {
            "delta_IoU": float(stage_d_results["B1S_STAGE_D_DEGRADED"]["summary"].get("mean_occupied_iou", 0.0)) - float(original_results["B1_ORIGINAL_DEGRADED"]["summary"].get("mean_occupied_iou", 0.0)),
            "delta_false_free": float(stage_d_results["B1S_STAGE_D_DEGRADED"]["summary"].get("mean_false_free_rate", 0.0)) - float(original_results["B1_ORIGINAL_DEGRADED"]["summary"].get("mean_false_free_rate", 0.0)),
            "delta_front_false_free": float(stage_d_results["B1S_STAGE_D_DEGRADED"]["summary"].get("mean_front_sector_false_free", 0.0)) - float(original_results["B1_ORIGINAL_DEGRADED"]["summary"].get("mean_front_sector_false_free", 0.0)),
        },
        "B2_stage_d_minus_original": {
            "delta_IoU": float(stage_d_results["B2S_STAGE_D_R8"]["summary"].get("mean_occupied_iou", 0.0)) - float(original_results["B2_ORIGINAL_R8"]["summary"].get("mean_occupied_iou", 0.0)),
            "delta_false_free": float(stage_d_results["B2S_STAGE_D_R8"]["summary"].get("mean_false_free_rate", 0.0)) - float(original_results["B2_ORIGINAL_R8"]["summary"].get("mean_false_free_rate", 0.0)),
            "delta_front_false_free": float(stage_d_results["B2S_STAGE_D_R8"]["summary"].get("mean_front_sector_false_free", 0.0)) - float(original_results["B2_ORIGINAL_R8"]["summary"].get("mean_front_sector_false_free", 0.0)),
        },
    }

    result = {
        "sample_range": [sample_start, sample_end],
        "legacy_path_audit": legacy_path_audit,
        "original_checkpoint_historical_reference": {
            "checkpoint_path": str(original_ckpt),
            "table": original_table,
            "results": original_results,
        },
        "stage_d_same_checkpoint_fair_comparison": {
            "checkpoint_path": str(stage_d_ckpt),
            "table": stage_d_table,
            "results": stage_d_results,
            "all_checkpoint_sha256_equal": len(checkpoint_sha_set) == 1,
            "checkpoint_sha256_values": sorted(checkpoint_sha_set),
            "n1_callback_audit": n1_callback_audit,
            "protocol_consistency": protocol_consistency,
            "b1s_vs_n0_parity": b1s_vs_n0,
            "n0_vs_n1_parity": n0_vs_n1,
            "t3_zero_replacement_vs_n1_parity": t3_vs_n1_zero,
            "delta_vs_n1": delta_vs_n1,
            "b2s_delta_vs_b1s": b2s_delta_vs_b1s,
        },
        "checkpoint_effect_audit": checkpoint_effect,
    }
    export_result = {
        **result,
        "original_checkpoint_historical_reference": {
            **result["original_checkpoint_historical_reference"],
            "results": {k: strip_raw_captures(v) for k, v in original_results.items()},
        },
        "stage_d_same_checkpoint_fair_comparison": {
            **result["stage_d_same_checkpoint_fair_comparison"],
            "results": {k: strip_raw_captures(v) for k, v in stage_d_results.items()},
        },
    }
    write_json(ARTIFACTS_DIR / "eval_fix_suite_results.json", export_result)
    return export_result


def run_mcqm_v2_interface_oracle_suite(sample_start: int, sample_end: int) -> dict[str, Any]:
    V2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    v1_frozen = freeze_v1_reference()
    meta_diff = collect_img_metas_protocol_diff(sample_start, sample_end)
    init_manifest = create_mcqm_v2_init_checkpoint()
    init_checkpoint_path = init_manifest["init_checkpoint_path"]

    p0 = run_mcqm_v2_eval_variant(
        "P0",
        init_checkpoint_path,
        sample_start,
        sample_end,
        capture_enabled=True,
        oracle_mode="disabled",
        artifact_label="MCQM_V2_P0",
    )
    h0 = run_mcqm_v2_eval_variant("H0", init_checkpoint_path, sample_start, sample_end, capture_enabled=False, oracle_mode="disabled", artifact_label="MCQM_V2_H0")
    h1 = run_mcqm_v2_eval_variant("H1", init_checkpoint_path, sample_start, sample_end, capture_enabled=False, oracle_mode="disabled", artifact_label="MCQM_V2_H1")
    h2_a00 = run_mcqm_v2_eval_variant("H2_A00", init_checkpoint_path, sample_start, sample_end, capture_enabled=True, oracle_mode="disabled", artifact_label="MCQM_V2_H2_A00")
    h2_a05 = run_mcqm_v2_eval_variant("H2_A05", init_checkpoint_path, sample_start, sample_end, capture_enabled=False, oracle_mode="disabled", artifact_label="MCQM_V2_H2_A05")
    h2_a10 = run_mcqm_v2_eval_variant("H2_A10", init_checkpoint_path, sample_start, sample_end, capture_enabled=False, oracle_mode="disabled", artifact_label="MCQM_V2_H2_A10")
    h2_a25 = run_mcqm_v2_eval_variant("H2_A25", init_checkpoint_path, sample_start, sample_end, capture_enabled=False, oracle_mode="disabled", artifact_label="MCQM_V2_H2_A25")
    h2_a50 = run_mcqm_v2_eval_variant("H2_A50", init_checkpoint_path, sample_start, sample_end, capture_enabled=False, oracle_mode="disabled", artifact_label="MCQM_V2_H2_A50")
    h3 = run_mcqm_v2_eval_variant("H3", init_checkpoint_path, sample_start, sample_end, capture_enabled=False, oracle_mode="disabled", artifact_label="MCQM_V2_H3")

    teacher_by_sample = capture_v2_clean_teacher_states(init_checkpoint_path, sample_start, sample_end)
    oracle_alignment_variants = {}
    for variant in ["H0", "H1", "H2_A10", "H2_A25", "H3"]:
        oracle_alignment_variants[variant] = run_mcqm_v2_eval_variant(
            variant,
            init_checkpoint_path,
            sample_start,
            sample_end,
            capture_enabled=False,
            oracle_mode="query_alignment_report_only",
            teacher_by_sample=teacher_by_sample,
            artifact_label=f"MCQM_V2_ORACLE_{variant}",
        )

    oracle_forced_results = {}
    oracle_e2e_rows = []
    _, dataset_for_targets, _, _ = build_runtime(
        train=False,
        mcqm_cfg=None,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
        checkpoint_path=CHECKPOINT_PATH,
    )
    preflight = preflight_oracle_e2e(dataset_for_targets, sample_start, sample_end)
    write_json(V2_ARTIFACTS_DIR / "preflight.json", preflight)
    target_samples = preflight["valid_targets"]
    for variant in ["H0", "H1", "H2_A10", "H2_A25", "H3"]:
        expected_injection_mode = mcqm_v2_variant_cfg(
            variant,
            oracle_mode="query_alignment_report_only",
        )["mcqm_v2_injection_mode"]
        for sample_idx in target_samples:
            for k in [1, 4, 8, 16]:
                warmup_start = max(sample_start, sample_idx - 1)
                p0_seed = capture_p0_memory_seed_for_target(init_checkpoint_path, warmup_start, sample_idx)
                seed_valid = (
                    is_valid_sha256(p0_seed["audit"].get("seed_memory_state_sha256"))
                    and int(p0_seed["audit"].get("seed_query_count", 0)) > 0
                    and bool(p0_seed["audit"].get("seed_source_all_native", False))
                )
                if not seed_valid:
                    oracle_e2e_rows.append({
                        "variant": variant,
                        "sample_index": sample_idx,
                        "requested_K": k,
                        "skip_reason": "invalid_or_empty_seed",
                        **p0_seed["audit"],
                    })
                    continue
                seeded_alignment = run_mcqm_v2_target_eval(
                    variant,
                    init_checkpoint_path,
                    warmup_start,
                    sample_idx,
                    oracle_mode="query_alignment_report_only",
                    teacher_by_sample=teacher_by_sample,
                    forced_pairs=[],
                    artifact_label=f"MCQM_V2_ORACLE_ALIGNMENT_TARGET_{variant}_{sample_idx}",
                    seed_memory_state=p0_seed["state"],
                )
                seeded_oracle_rows = [
                    row
                    for row in seeded_alignment.get("oracle_rows", [])
                    if int(row.get("sample_index", -1)) == int(sample_idx)
                    and row.get("injection_mode") == expected_injection_mode
                ]
                forced_pair_info = build_positive_nonconflicting_forced_pairs(
                    seeded_oracle_rows,
                    k,
                )
                forced_pairs = forced_pair_info["forced_pairs"]
                if not forced_pairs:
                    oracle_e2e_rows.append({
                        "variant": variant,
                        "sample_index": sample_idx,
                        "requested_K": k,
                        "positive_nonconflicting_pair_count": int(
                            forced_pair_info[
                                "positive_nonconflicting_pair_count"
                            ]
                        ),
                        "requested_pair_count": 0,
                        "skip_reason": "no_positive_oracle_pairs",
                        "seeded_alignment_row_count": int(
                            len(seeded_oracle_rows)
                        ),
                        "warmup_start": warmup_start,
                        **p0_seed["audit"],
                    })
                    continue
                baseline_target = run_mcqm_v2_target_eval(
                    "P0",
                    init_checkpoint_path,
                    warmup_start,
                    sample_idx,
                    oracle_mode="disabled",
                    teacher_by_sample=None,
                    forced_pairs=[],
                    artifact_label=f"MCQM_V2_P0_TARGET_{sample_idx}",
                    seed_memory_state=p0_seed["state"],
                )
                forced = run_mcqm_v2_target_eval(
                    variant,
                    init_checkpoint_path,
                    warmup_start,
                    sample_idx,
                    oracle_mode="query_alignment_report_only",
                    teacher_by_sample=teacher_by_sample,
                    forced_pairs=forced_pairs,
                    artifact_label=f"MCQM_V2_ORACLE_E2E_{variant}_K{k}",
                    seed_memory_state=p0_seed["state"],
                )
                baseline_seed_hash = str(
                    baseline_target["summary"]
                    .get("seed_audit", {})
                    .get("seed_memory_state_sha256", "")
                )
                forced_seed_hash = str(
                    forced["summary"]
                    .get("seed_audit", {})
                    .get("seed_memory_state_sha256", "")
                )
                oracle_forced_results[f"{variant}_sample{sample_idx}_K{k}"] = forced
                oracle_forced_results[f"P0_sample{sample_idx}_K{k}"] = baseline_target
                oracle_forced_results[
                    f"{variant}_sample{sample_idx}_K{k}_seeded_alignment"
                ] = seeded_alignment
                baseline_map = build_target_metric_map(baseline_target)
                forced_map = build_target_metric_map(forced)
                baseline_row = baseline_map[(sample_idx, 0)]
                forced_row = forced_map[(sample_idx, 0)]
                forced_debug = forced["debug_rows"][0] if forced["debug_rows"] else {}
                oracle_e2e_rows.append({
                    "variant": variant,
                    "sample_index": sample_idx,
                    "requested_K": k,
                    "positive_nonconflicting_pair_count": int(
                        forced_pair_info["positive_nonconflicting_pair_count"]
                    ),
                    "requested_pair_count": int(
                        forced_pair_info["requested_pair_count"]
                    ),
                    "applied_pair_count": int(forced_debug.get("actual_applied_pair_count", forced_debug.get("actual_replacement_count", 0))),
                    "requested_forced_pair_count": int(forced_debug.get("requested_forced_pair_count", len(forced_pairs))),
                    "resolved_forced_pair_count": int(forced_debug.get("resolved_forced_pair_count", 0)),
                    "seeded_alignment_row_count": int(len(seeded_oracle_rows)),
                    "baseline_iou": float(baseline_row.get("occupied_iou", 0.0)),
                    "oracle_iou": float(forced_row.get("occupied_iou", 0.0)),
                    "delta_IoU": float(forced_row.get("occupied_iou", 0.0)) - float(baseline_row.get("occupied_iou", 0.0)),
                    "baseline_false_free": float(baseline_row.get("false_free_rate", 0.0)),
                    "oracle_false_free": float(forced_row.get("false_free_rate", 0.0)),
                    "delta_false_free": float(forced_row.get("false_free_rate", 0.0)) - float(baseline_row.get("false_free_rate", 0.0)),
                    "baseline_front_false_free": float(baseline_row.get("front_sector_false_free", 0.0)),
                    "oracle_front_false_free": float(forced_row.get("front_sector_false_free", 0.0)),
                    "delta_front_false_free": float(forced_row.get("front_sector_false_free", 0.0)) - float(baseline_row.get("front_sector_false_free", 0.0)),
                    "warmup_start": warmup_start,
                    **p0_seed["audit"],
                    "baseline_seed_memory_state_sha256": baseline_seed_hash,
                    "forced_seed_memory_state_sha256": forced_seed_hash,
                })

    forward_variants = {
        "P0": p0,
        "H0": h0,
        "H1": h1,
        "H2_A00": h2_a00,
        "H2_A05": h2_a05,
        "H2_A10": h2_a10,
        "H2_A25": h2_a25,
        "H2_A50": h2_a50,
        "H3": h3,
    }
    forward_table = []
    p0_summary = p0["summary"]
    selection_signature_audit_rows = []
    for label, result in forward_variants.items():
        summary = result["summary"]
        debug = summary.get("debug", {})
        prov = summary.get("provenance", {})
        memory_correct = float(prov.get("mean_memory_correct_voxels", 0.0))
        memory_fp = float(prov.get("mean_memory_fp_voxels", 0.0))
        selection_signature = object_sha256(
            [
                {
                    "sample_index": int(row["sample_index"]),
                    "selection_signature_sha256": str(row["selection_signature_sha256"]),
                }
                for row in result["selection_signature_rows"]
            ]
        ) if result["selection_signature_rows"] else ""
        forward_table.append({
            "Variant": label,
            "IoU": float(summary.get("mean_occupied_iou", 0.0)),
            "False-free": float(summary.get("mean_false_free_rate", 0.0)),
            "Front false-free": float(summary.get("mean_front_sector_false_free", 0.0)),
            "Mean replacement": float(debug.get("mean_actual_replacement_count", 0.0)),
            "Memory correct": memory_correct,
            "Memory FP": memory_fp,
            "FP/Correct": (memory_fp / memory_correct) if memory_correct > 0 else float("inf"),
            "Feature delta norm": float(debug.get("mean_feature_delta_norm_mean", 0.0)),
            "Point delta metric norm": float(debug.get("mean_point_delta_metric_norm_mean", 0.0)),
            "Selection signature": selection_signature,
            "delta_IoU": float(summary.get("mean_occupied_iou", 0.0)) - float(p0_summary.get("mean_occupied_iou", 0.0)),
            "delta_false_free": float(summary.get("mean_false_free_rate", 0.0)) - float(p0_summary.get("mean_false_free_rate", 0.0)),
            "delta_front_false_free": float(summary.get("mean_front_sector_false_free", 0.0)) - float(p0_summary.get("mean_front_sector_false_free", 0.0)),
        })
    for sample_idx in range(sample_start, sample_end + 1):
        warmup_start = max(sample_start, sample_idx - 1)
        if warmup_start == sample_idx:
            continue
        p0_seed = capture_p0_memory_seed_for_target(init_checkpoint_path, warmup_start, sample_idx)
        base_target = run_mcqm_v2_target_eval(
            "P0",
            init_checkpoint_path,
            warmup_start,
            sample_idx,
            oracle_mode="disabled",
            teacher_by_sample=None,
            forced_pairs=[],
            artifact_label=f"MCQM_V2_SELECTION_P0_{sample_idx}",
            seed_memory_state=p0_seed["state"],
        )
        base_sig = base_target["selection_signature_rows"][0]["selection_signature_sha256"] if base_target["selection_signature_rows"] else ""
        for label in ["H0", "H1", "H2_A00", "H2_A10", "H3"]:
            target_result = run_mcqm_v2_target_eval(
                label,
                init_checkpoint_path,
                warmup_start,
                sample_idx,
                oracle_mode="disabled",
                teacher_by_sample=None,
                forced_pairs=[],
                artifact_label=f"MCQM_V2_SELECTION_{label}_{sample_idx}",
                seed_memory_state=p0_seed["state"],
            )
            variant_sig = target_result["selection_signature_rows"][0]["selection_signature_sha256"] if target_result["selection_signature_rows"] else ""
            selection_signature_audit_rows.append({
                "variant": label,
                "sample_index": int(sample_idx),
                "selection_signature_sha256": variant_sig,
                "matches_P0_seeded_target": variant_sig == base_sig,
                "warmup_start": int(warmup_start),
                **p0_seed["audit"],
            })

    parity_audit = {
        "P0_vs_H2_A00": mcqm_callback_parity_between_results(p0, h2_a00, zero_replacement_only_right=False),
    }
    oracle_alignment_rows = []
    for label, result in oracle_alignment_variants.items():
        for row in result["oracle_rows"]:
            oracle_alignment_rows.append({"variant": label, **row})

    def _decision_from_oracle(
        rows: list[dict[str, Any]],
        oracle_alignment_rows_local: list[dict[str, Any]],
        required_target_samples: list[int],
    ) -> dict[str, Any]:
        return decision_from_oracle_rows(
            rows,
            oracle_alignment_rows_local,
            required_target_samples,
        )

    decision = _decision_from_oracle(
        oracle_e2e_rows,
        oracle_alignment_rows,
        target_samples,
    )
    write_json(V2_ARTIFACTS_DIR / "interface_forward_ablation.json", {"table": forward_table, "results": {k: strip_raw_captures(v) for k, v in forward_variants.items()}})
    write_csv(V2_ARTIFACTS_DIR / "interface_forward_ablation.csv", forward_table)
    write_json(V2_ARTIFACTS_DIR / "oracle_query_alignment.json", {"rows": oracle_alignment_rows, "results": {k: strip_raw_captures(v) for k, v in oracle_alignment_variants.items()}})
    write_csv(V2_ARTIFACTS_DIR / "oracle_query_alignment.csv", oracle_alignment_rows)
    write_json(V2_ARTIFACTS_DIR / "oracle_e2e_validation.json", {"rows": oracle_e2e_rows, "forced_results": {k: strip_raw_captures(v) for k, v in oracle_forced_results.items()}})
    write_csv(V2_ARTIFACTS_DIR / "oracle_e2e_validation.csv", oracle_e2e_rows)
    write_json(V2_ARTIFACTS_DIR / "selection_signature_audit.json", {"rows": selection_signature_audit_rows})
    write_json(V2_ARTIFACTS_DIR / "parity_audit.json", parity_audit)
    write_json(V2_ARTIFACTS_DIR / "decision.json", decision)
    result = {
        "v1_frozen_reference": v1_frozen,
        "img_metas_protocol_diff": meta_diff,
        "init_manifest": init_manifest,
        "forward_variants": {k: strip_raw_captures(v) for k, v in forward_variants.items()},
        "forward_table": forward_table,
        "oracle_alignment_results": {k: strip_raw_captures(v) for k, v in oracle_alignment_variants.items()},
        "oracle_alignment_rows": oracle_alignment_rows,
        "oracle_e2e_rows": oracle_e2e_rows,
        "selection_signature_audit_rows": selection_signature_audit_rows,
        "parity_audit": parity_audit,
        "decision": decision,
    }
    write_test_report(
        "completed",
        stage="eval_v2_interface_oracle",
        decision_status=decision["status"],
        completed_jobs=len(oracle_e2e_rows),
        total_jobs=len(oracle_e2e_rows),
    )
    return result


def _q2f_tensor_losses(losses: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu().item())
    return out


def _sum_tensor_losses(losses: dict[str, Any], predicate: Callable[[str], bool]) -> torch.Tensor | None:
    total = None
    for key, value in losses.items():
        if torch.is_tensor(value) and predicate(key):
            total = value if total is None else total + value
    return total


def _q2f_total_loss(model: Any, losses: dict[str, Any]) -> torch.Tensor:
    task_loss = _sum_tensor_losses(
        losses,
        lambda key: not str(key).startswith("loss_mcqm_q2f_"),
    )
    q2f_loss = _sum_tensor_losses(
        losses,
        lambda key: str(key).startswith("loss_mcqm_q2f_"),
    )
    if task_loss is None and q2f_loss is None:
        raise RuntimeError("no tensor losses produced by Q2F preflight")
    occ_weight = float(getattr(model, "mcqm_cfg", {}).get("mcqm_q2f_occ_loss_weight", 1.0))
    total = None
    if task_loss is not None:
        total = task_loss * occ_weight
    if q2f_loss is not None:
        total = q2f_loss if total is None else total + q2f_loss
    assert total is not None
    return total


def set_mcqm_q2f_training_mode(model: Any) -> None:
    model.train()
    for name, module in model.named_modules():
        if not name:
            continue
        if name.startswith("mcqm_q2f_"):
            module.train()
        else:
            module.eval()


def _q2f_extra_model_kwargs(batch_cuda: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in batch_cuda.items()
        if key not in {"img", "img_metas"}
    }


Q2F_BASE_RUNTIME_STATE_KEYS = (
    "pts_bbox_head.num_stamps_all",
)


def checkpoint_has_q2f_state(checkpoint_path: str | Path) -> bool:
    ckpt = torch_load_compat(resolve_checkpoint_path(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        return False
    for raw_key in state_dict.keys():
        if normalize_state_dict_key(str(raw_key)).startswith("mcqm_q2f_"):
            return True
    return False


def build_q2f_runtime(
    *,
    train: bool,
    checkpoint_path: str | Path | None,
    sample_start: int,
    sample_end: int,
) -> tuple[dict[str, Any], Any, Any, dict[str, Any], dict[str, Any]]:
    cfg = mcqm_q2f_v1_cfg()
    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path or CHECKPOINT_PATH)
    q2f_checkpoint = checkpoint_has_q2f_state(resolved_checkpoint_path)
    checkpoint_obj = torch_load_compat(resolved_checkpoint_path, map_location="cpu")
    checkpoint_meta = checkpoint_obj.get("meta", {}) if isinstance(checkpoint_obj, dict) else {}
    base_checkpoint_path = resolve_checkpoint_path(CHECKPOINT_PATH)
    if q2f_checkpoint:
        checkpoint_version = str(checkpoint_meta.get("q2f_protocol_version", ""))
        if checkpoint_version != MCQM_Q2F_PROTOCOL_VERSION:
            raise RuntimeError(
                "Q2F checkpoint protocol mismatch: "
                f"checkpoint={checkpoint_version}, "
                f"runtime={MCQM_Q2F_PROTOCOL_VERSION}"
            )
    runtime_cfg, dataset, model = build_runtime_unloaded(
        train=train,
        mcqm_cfg=cfg,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    _ = runtime_cfg
    load_model_state_strict(
        model,
        base_checkpoint_path if q2f_checkpoint else resolved_checkpoint_path,
        allowed_missing_keys=set(),
    )
    model = model.cuda()
    prev_index, _ = find_first_valid_q2f_pair(dataset, sample_start, sample_end)
    _, prev_batch = extract_inputs(dataset, prev_index)
    prev_batch = enrich_batch_scene_meta(prev_batch, dataset, prev_index)
    prev_cuda = sw2.move_to_cuda(prev_batch)
    materialized = materialize_mcqm_q2f_modules(model, prev_cuda)
    if q2f_checkpoint:
        load_model_state_strict(
            model,
            resolved_checkpoint_path,
            allowed_missing_keys=set(),
        )
    if q2f_checkpoint:
        runtime_architecture_signature = build_mcqm_q2f_architecture_signature(
            model,
            cfg,
            file_sha256(base_checkpoint_path),
            camera_count_override=materialized["camera_count"],
        )
        checkpoint_architecture_sha = str(checkpoint_meta.get("architecture_signature_sha256", ""))
        runtime_architecture_sha = str(runtime_architecture_signature["architecture_signature_sha256"])
        if checkpoint_architecture_sha != runtime_architecture_sha:
            raise RuntimeError(
                "Q2F checkpoint architecture signature mismatch: "
                f"checkpoint={checkpoint_architecture_sha}, runtime={runtime_architecture_sha}"
            )
    if hasattr(model, "set_epoch") and isinstance(checkpoint_meta, dict) and checkpoint_meta.get("epoch") is not None:
        model.set_epoch(int(checkpoint_meta["epoch"]))
    model.train() if train else model.eval()
    runtime_meta = {
        **checkpoint_file_info(resolved_checkpoint_path),
        "q2f_checkpoint_loaded": bool(q2f_checkpoint),
        "base_checkpoint_path": str(base_checkpoint_path),
        "base_checkpoint_sha256": file_sha256(base_checkpoint_path),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_parameter_key_hash": object_sha256(sorted(model.state_dict().keys())),
        "core_model_state_sha256": model_state_sha256(model),
    }
    return cfg, dataset, model, runtime_meta, materialized


def seed_q2f_memory_from_previous_frame(model: Any, prev_batch_cuda: dict[str, Any]) -> None:
    reset_runtime_state(model)
    prev_inputs = normalized_model_batch_kwargs(prev_batch_cuda)
    prev_img = prev_inputs.pop("img")
    prev_metas = prev_inputs.pop("img_metas")
    runtime_prev = getattr(model, "mcqm_v2_runtime_oracle", None)
    runtime_next = dict(runtime_prev) if isinstance(runtime_prev, dict) else {}
    runtime_next["q2f_shallow_bypass"] = True
    model.mcqm_v2_runtime_oracle = runtime_next
    with torch.no_grad():
        try:
            model.forward_backbone(
                prev_img,
                prev_metas,
                **prev_inputs,
            )
        finally:
            model.mcqm_v2_runtime_oracle = runtime_prev


def run_q2f_current_frame_forward(
    model: Any,
    current_batch_cuda: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"disabled", "bypass", "active"}:
        raise ValueError(f"unsupported q2f forward mode: {mode}")
    runtime_prev = getattr(model, "mcqm_v2_runtime_oracle", None)
    runtime_next = dict(runtime_prev) if isinstance(runtime_prev, dict) else {}
    runtime_next["q2f_shallow_disable"] = mode == "disabled"
    runtime_next["q2f_shallow_bypass"] = mode == "bypass"
    model.mcqm_v2_runtime_oracle = runtime_next
    try:
        with torch.no_grad():
            return model(return_loss=False, rescale=True, **current_batch_cuda)
    finally:
        model.mcqm_v2_runtime_oracle = runtime_prev


def build_q2f_eval_parity_record(
    sample_index: int,
    baseline_pred_temporal: dict[int, torch.Tensor],
    bypass_run1_pred_temporal: dict[int, torch.Tensor],
    bypass_run2_pred_temporal: dict[int, torch.Tensor],
) -> dict[str, Any]:
    horizon_rows = []
    all_equal = True
    for horizon_s in CORE_HORIZONS:
        baseline_pred = baseline_pred_temporal[horizon_s].cpu()
        bypass_run1_pred = bypass_run1_pred_temporal[horizon_s].cpu()
        bypass_run2_pred = bypass_run2_pred_temporal[horizon_s].cpu()
        baseline_vs_bypass_run1_equal = bool(torch.equal(baseline_pred, bypass_run1_pred))
        bypass_run1_vs_run2_equal = bool(torch.equal(bypass_run1_pred, bypass_run2_pred))
        baseline_sha = tensor_sha256(baseline_pred)
        bypass_run1_sha = tensor_sha256(bypass_run1_pred)
        bypass_run2_sha = tensor_sha256(bypass_run2_pred)
        baseline_vs_bypass_run1_sha_equal = baseline_sha == bypass_run1_sha
        bypass_run1_vs_run2_sha_equal = bypass_run1_sha == bypass_run2_sha
        all_equal = (
            all_equal
            and baseline_vs_bypass_run1_equal
            and bypass_run1_vs_run2_equal
            and baseline_vs_bypass_run1_sha_equal
            and bypass_run1_vs_run2_sha_equal
        )
        horizon_rows.append(
            {
                "horizon_s": int(horizon_s),
                "baseline_vs_bypass_run1_equal": baseline_vs_bypass_run1_equal,
                "bypass_run1_vs_run2_equal": bypass_run1_vs_run2_equal,
                "baseline_sha256": baseline_sha,
                "bypass_run1_sha256": bypass_run1_sha,
                "bypass_run2_sha256": bypass_run2_sha,
                "baseline_vs_bypass_run1_sha256_equal": baseline_vs_bypass_run1_sha_equal,
                "bypass_run1_vs_run2_sha256_equal": bypass_run1_vs_run2_sha_equal,
            }
        )
    return {
        "sample_index": int(sample_index),
        "all_horizons_equal": bool(all_equal),
        "rows": horizon_rows,
    }


def build_q2f_train_checkpoint_path(iters: int) -> Path:
    out_dir = Q2F_ARTIFACTS_DIR / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"mcqm_q2f_shallow_iter{int(iters):04d}.pth"


def run_q2f_training(
    sample_start: int,
    sample_end: int,
    iters: int,
    lr: float,
    seed: int,
    *,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    Q2F_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg, dataset, model, runtime_meta, materialized = build_q2f_runtime(
        train=True,
        checkpoint_path=init_checkpoint or CHECKPOINT_PATH,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    pair_rows = build_pairs(dataset, sample_start, sample_end)
    if not pair_rows:
        raise RuntimeError(f"no valid q2f training pairs inside [{sample_start}, {sample_end}]")
    freeze_manifest = apply_mcqm_q2f_parameter_freeze(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("Q2F training found no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    resolved_init_checkpoint = resolve_checkpoint_path(init_checkpoint or CHECKPOINT_PATH)
    if checkpoint_has_q2f_state(resolved_init_checkpoint):
        ckpt = torch_load_compat(resolved_init_checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
    if hasattr(model, "set_epoch"):
        model.set_epoch(max(int(getattr(model, "finetune_epoch", 0)), 1))
    train_rows = []
    set_mcqm_q2f_training_mode(model)
    for step in range(iters):
        pair = pair_rows[step % len(pair_rows)]
        prev_index = int(pair["prev_index"])
        curr_index = int(pair["curr_index"])
        _, prev_batch = extract_inputs(dataset, prev_index)
        _, curr_clean_batch = extract_inputs(dataset, curr_index)
        prev_batch = enrich_batch_scene_meta(prev_batch, dataset, prev_index)
        curr_clean_batch = enrich_batch_scene_meta(curr_clean_batch, dataset, curr_index)
        curr_deg_batch, perturb_manifest = apply_perturbation_with_manifest_to_batch(
            copy.deepcopy(curr_clean_batch),
            "A10_drop_front_triplet",
        )
        prev_cuda = sw2.move_to_cuda(prev_batch)
        curr_clean_cuda = sw2.move_to_cuda(curr_clean_batch)
        curr_deg_cuda = sw2.move_to_cuda(curr_deg_batch)
        seed_q2f_memory_from_previous_frame(model, prev_cuda)
        optimizer.zero_grad(set_to_none=True)
        iter_inputs = normalized_model_batch_kwargs(curr_deg_cuda)
        teacher_img, teacher_metas = direct_model_inputs_from_batch(curr_clean_cuda)
        iter_inputs["mcqm_q2f_teacher_img"] = teacher_img
        iter_inputs["mcqm_q2f_teacher_img_metas"] = teacher_metas
        losses = model(return_loss=True, **iter_inputs)
        total_loss = _q2f_total_loss(model, losses)
        scalar_losses = _q2f_tensor_losses(losses)
        total_loss.backward()
        q2f_grad_norms = {}
        for name, param in model.named_parameters():
            if not name.startswith("mcqm_q2f_"):
                continue
            if param.grad is None:
                q2f_grad_norms[name] = 0.0
                continue
            q2f_grad_norms[name] = float(param.grad.detach().norm().item())
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer.step()
        q2f_debug = copy.deepcopy(getattr(model, "latest_mcqm_q2f_debug", {}))
        level_rows = q2f_debug.get("level_rows", []) if isinstance(q2f_debug, dict) else []
        supported_cells = sum(
            int(row.get("supported_cell_count", 0))
            for row in level_rows
            if isinstance(row, dict)
        )
        valid_points = sum(
            int(row.get("valid_projected_point_count", 0))
            for row in level_rows
            if isinstance(row, dict)
        )
        train_rows.append(
            {
                "iter": int(step),
                "prev_index": prev_index,
                "curr_index": curr_index,
                "perturbation_id": "A10_drop_front_triplet",
                "perturbation_manifest_hash": object_sha256(perturb_manifest),
                "loss_total": float(total_loss.detach().cpu().item()),
                "q2f_grad_norm_max": float(max(q2f_grad_norms.values())) if q2f_grad_norms else 0.0,
                "q2f_grad_norm_mean": float(sum(q2f_grad_norms.values()) / len(q2f_grad_norms)) if q2f_grad_norms else 0.0,
                "valid_projected_point_count": int(valid_points),
                "supported_cell_count": int(supported_cells),
                **scalar_losses,
            }
        )
        if (step + 1) % 5 == 0 or step == 0 or step + 1 == iters:
            print(
                f"Q2F train iter={step + 1}/{iters} loss={train_rows[-1]['loss_total']:.4f} "
                f"support={supported_cells} valid_points={valid_points}",
                flush=True,
            )
    architecture_signature = build_mcqm_q2f_architecture_signature(
        model,
        cfg,
        runtime_meta["base_checkpoint_sha256"],
        camera_count_override=materialized["camera_count"],
    )
    out_ckpt = build_q2f_train_checkpoint_path(iters)
    save_checkpoint(
        out_ckpt,
        model,
        optimizer,
        {
            "stage": "q2f_train",
            "iters": int(iters),
            "lr": float(lr),
            "seed": int(seed),
            "sample_start": int(sample_start),
            "sample_end": int(sample_end),
            "q2f_protocol_version": MCQM_Q2F_PROTOCOL_VERSION,
            "mcqm_cfg": cfg,
            "selected_shallow_module_path": str(getattr(model, "mcqm_q2f_selected_module_path", "")),
            "selected_shallow_shape": list(getattr(model, "mcqm_q2f_selected_shallow_shape", [])),
            "backbone_type": type(model.img_backbone).__name__,
            "neck_type": type(model.img_neck).__name__ if getattr(model, "img_neck", None) is not None else "",
            "architecture_signature": architecture_signature,
            "architecture_signature_sha256": architecture_signature["architecture_signature_sha256"],
            "base_checkpoint_path": runtime_meta["base_checkpoint_path"],
            "base_checkpoint_sha256": runtime_meta["base_checkpoint_sha256"],
            "freeze_manifest": freeze_manifest,
        },
    )
    summary = aggregate_rows(train_rows)
    result = {
        "status": "completed",
        "iters": int(iters),
        "sample_start": int(sample_start),
        "sample_end": int(sample_end),
        "checkpoint_path": str(out_ckpt),
        "checkpoint_sha256": file_sha256(out_ckpt),
        "materialized": materialized,
        "runtime_meta": runtime_meta,
        "freeze_manifest": freeze_manifest,
        "summary": summary,
        "rows": train_rows[-min(len(train_rows), 50):],
        "gradient_norm_per_iter": [
            {
                "iter": int(row["iter"]),
                "q2f_grad_norm_max": float(row["q2f_grad_norm_max"]),
                "q2f_grad_norm_mean": float(row["q2f_grad_norm_mean"]),
            }
            for row in train_rows
        ],
    }
    write_json(Q2F_ARTIFACTS_DIR / "q2f_train.json", result)
    return result


def run_q2f_eval(
    sample_start: int,
    sample_end: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    sectors = get_sector_masks_cached()
    cfg, dataset_q2f, model_q2f, q2f_runtime_meta, materialized = build_q2f_runtime(
        train=False,
        checkpoint_path=checkpoint_path,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    _ = cfg
    pair_rows = build_pairs(dataset_q2f, sample_start, sample_end)
    eval_rows = []
    parity_rows = []
    for pair in pair_rows:
        prev_index = int(pair["prev_index"])
        curr_index = int(pair["curr_index"])
        sample_raw, curr_clean_batch = extract_inputs(dataset_q2f, curr_index)
        _, prev_batch = extract_inputs(dataset_q2f, prev_index)
        prev_batch = enrich_batch_scene_meta(prev_batch, dataset_q2f, prev_index)
        curr_clean_batch = enrich_batch_scene_meta(curr_clean_batch, dataset_q2f, curr_index)
        curr_deg_batch, perturb_manifest = apply_perturbation_with_manifest_to_batch(
            copy.deepcopy(curr_clean_batch),
            "A10_drop_front_triplet",
        )
        _ = perturb_manifest
        prev_cuda = sw2.move_to_cuda(prev_batch)
        curr_deg_cuda = sw2.move_to_cuda(curr_deg_batch)
        seed_q2f_memory_from_previous_frame(model_q2f, prev_cuda)
        base_out = run_q2f_current_frame_forward(
            model_q2f,
            curr_deg_cuda,
            mode="disabled",
        )
        seed_q2f_memory_from_previous_frame(model_q2f, prev_cuda)
        q2f_bypass_run1_out = run_q2f_current_frame_forward(
            model_q2f,
            curr_deg_cuda,
            mode="bypass",
        )
        seed_q2f_memory_from_previous_frame(model_q2f, prev_cuda)
        q2f_bypass_run2_out = run_q2f_current_frame_forward(
            model_q2f,
            curr_deg_cuda,
            mode="bypass",
        )
        parity_record = build_q2f_eval_parity_record(
            curr_index,
            sw2.extract_standard_tensors(sw2.unwrap(sample_raw), base_out)[0],
            sw2.extract_standard_tensors(sw2.unwrap(sample_raw), q2f_bypass_run1_out)[0],
            sw2.extract_standard_tensors(sw2.unwrap(sample_raw), q2f_bypass_run2_out)[0],
        )
        parity_rows.append(parity_record)
        if not bool(parity_record["all_horizons_equal"]):
            write_json(
                Q2F_ARTIFACTS_DIR / "q2f_eval_parity.json",
                {
                    "status": "failed",
                    "reason": "baseline_q2f_bypass_or_bypass_repeat_mismatch",
                    "sample_start": int(sample_start),
                    "sample_end": int(sample_end),
                    "checkpoint_path": str(resolve_checkpoint_path(checkpoint_path)),
                    "rows": parity_rows,
                },
            )
            raise RuntimeError(
                "Q2F eval parity failed: baseline degraded path != Q2F bypass path "
                "or repeated bypass runs differ "
                f"at sample {curr_index}"
            )
        seed_q2f_memory_from_previous_frame(model_q2f, prev_cuda)
        q2f_out = run_q2f_current_frame_forward(
            model_q2f,
            curr_deg_cuda,
            mode="active",
        )
        pred_temporal_base, gt_temporal, _ = sw2.extract_standard_tensors(sw2.unwrap(sample_raw), base_out)
        pred_temporal_q2f, _, _ = sw2.extract_standard_tensors(sw2.unwrap(sample_raw), q2f_out)
        for horizon_s in CORE_HORIZONS:
            pred_base = pred_temporal_base[horizon_s].cpu()
            pred_q2f = pred_temporal_q2f[horizon_s].cpu()
            gt_h = gt_temporal[horizon_s].cpu()
            gt0 = gt_temporal[0].cpu()
            base_row = sw12b.build_eval_row(pred_base, gt_h, gt0, "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=None)
            q2f_row = sw12b.build_eval_row(pred_q2f, gt_h, gt0, "A10_drop_front_triplet", horizon_s, sectors, baseline_pred=None)
            eval_rows.append(
                {
                    "sample_index": int(curr_index),
                    "horizon_s": int(horizon_s),
                    "baseline_vs_q2f_bypass_equal": True,
                    "q2f_bypass_repeat_equal": True,
                    "baseline_occupied_iou": float(base_row["occupied_iou"]),
                    "q2f_occupied_iou": float(q2f_row["occupied_iou"]),
                    "delta_iou": float(q2f_row["occupied_iou"] - base_row["occupied_iou"]),
                    "baseline_false_free_rate": float(base_row["false_free_rate"]),
                    "q2f_false_free_rate": float(q2f_row["false_free_rate"]),
                    "delta_false_free_rate": float(q2f_row["false_free_rate"] - base_row["false_free_rate"]),
                    "baseline_front_false_free": float(base_row["front_sector_false_free"]),
                    "q2f_front_false_free": float(q2f_row["front_sector_false_free"]),
                    "delta_front_false_free": float(q2f_row["front_sector_false_free"] - base_row["front_sector_false_free"]),
                }
            )
    summary = aggregate_rows(eval_rows)
    write_json(
        Q2F_ARTIFACTS_DIR / "q2f_eval_parity.json",
        {
            "status": "passed",
            "sample_start": int(sample_start),
            "sample_end": int(sample_end),
            "checkpoint_path": str(resolve_checkpoint_path(checkpoint_path)),
            "checkpoint_sha256": file_sha256(resolve_checkpoint_path(checkpoint_path)),
            "rows": parity_rows,
        },
    )
    result = {
        "status": "completed",
        "sample_start": int(sample_start),
        "sample_end": int(sample_end),
        "checkpoint_path": str(resolve_checkpoint_path(checkpoint_path)),
        "checkpoint_sha256": file_sha256(resolve_checkpoint_path(checkpoint_path)),
        "q2f_runtime_meta": q2f_runtime_meta,
        "materialized": materialized,
        "parity_rows": parity_rows,
        "summary": summary,
        "rows": eval_rows,
    }
    write_json(Q2F_ARTIFACTS_DIR / "q2f_eval.json", result)
    decision_status = "MCQM_Q2F_V1_READY_FOR_TRAINING" if float(summary.get("mean_delta_iou", 0.0)) > 0.0 else "MCQM_Q2F_V1_NOT_READY_FOR_TRAINING"
    write_json(
        Q2F_ARTIFACTS_DIR / "decision.json",
        {
            "status": decision_status,
            "reason": "performance_validation",
            "mean_delta_iou": float(summary.get("mean_delta_iou", 0.0)),
            "mean_delta_false_free_rate": float(summary.get("mean_delta_false_free_rate", 0.0)),
            "mean_delta_front_false_free": float(summary.get("mean_delta_front_false_free", 0.0)),
        },
    )
    write_json(
        Q2F_ARTIFACTS_DIR / "test_report.json",
        {
            "status": "completed",
            "decision_status": decision_status,
            "evaluated_pairs": int(len(pair_rows)),
        },
    )
    return result


def run_mcqm_q2f_preflight(sample_start: int, sample_end: int, *, dry_run: bool = False) -> dict[str, Any]:
    Q2F_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = mcqm_q2f_v1_cfg()
    checkpoint_path = resolve_checkpoint_path(CHECKPOINT_PATH)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    write_json(Q2F_ARTIFACTS_DIR / "q2f_config.json", cfg)
    write_json(Q2F_ARTIFACTS_DIR / "test_report.json", {"status": "running", "updated_at": iso_now()})
    blocking_issues: list[dict[str, Any]] = []

    runtime_cfg, dataset, model, runtime_meta = build_runtime_strict(
        train=True,
        mcqm_cfg=cfg,
        checkpoint_path=checkpoint_path,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    _ = runtime_cfg
    prev_index, curr_index = find_first_valid_q2f_pair(dataset, sample_start, sample_end)
    prev_raw, prev_batch = sw2.extract_sample_batch(dataset, prev_index, collate_fn)
    curr_raw, curr_clean_batch = sw2.extract_sample_batch(dataset, curr_index, collate_fn)
    _ = prev_raw
    prev_batch = enrich_batch_scene_meta(prev_batch, dataset, prev_index)
    curr_clean_batch = enrich_batch_scene_meta(curr_clean_batch, dataset, curr_index)
    curr_deg_batch, perturb_manifest = apply_perturbation_with_manifest_to_batch(
        copy.deepcopy(curr_clean_batch),
        "A10_drop_front_triplet",
    )
    prev_cuda = sw2.move_to_cuda(prev_batch)
    curr_clean_cuda = sw2.move_to_cuda(curr_clean_batch)
    curr_deg_cuda = sw2.move_to_cuda(curr_deg_batch)

    materialized = materialize_mcqm_q2f_modules(model, prev_cuda)
    topology_audit = build_q2f_shallow_topology_audit(model, prev_cuda)
    write_json(Q2F_ARTIFACTS_DIR / "q2f_shallow_topology_audit.json", topology_audit)
    freeze_manifest = apply_mcqm_q2f_parameter_freeze(model)
    signature = build_mcqm_q2f_architecture_signature(
        model,
        cfg,
        checkpoint_sha256,
        camera_count_override=materialized["camera_count"],
    )
    write_json(Q2F_ARTIFACTS_DIR / "implementation_manifest.json", {
        "protocol_version": MCQM_Q2F_PROTOCOL_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_meta": runtime_meta,
        "topology_audit": topology_audit,
        "sample_pair": {"prev_index": int(prev_index), "curr_index": int(curr_index)},
        "perturbation_id": "A10_drop_front_triplet",
        "perturbation_manifest_hash": object_sha256(perturb_manifest),
    })
    write_json(Q2F_ARTIFACTS_DIR / "module_contract.json", {
        "backbone_type": materialized["backbone_type"],
        "neck_type": materialized["neck_type"],
        "query_dim": materialized["query_dim"],
        "camera_count": materialized["camera_count"],
        "camera_order": materialized["camera_order"],
        "selected_shallow_module_path": materialized["selected_shallow_module_path"],
        "selected_shallow_shape": materialized["selected_shallow_shape"],
        "architecture_signature": signature,
    })
    write_json(Q2F_ARTIFACTS_DIR / "projection_contract.json", {
        "projection_source": [
            "mmdet3d.models.sparsedetectors.mcqm.build_occ2img_from_img_metas",
            "mmdet3d.models.sparsedetectors.mcqm.project_occ_points_to_image",
        ],
        "metadata_fields": ["lidar2img", "ego2lidar", "img_shape"],
        "image_augmentation_contract": "uses img_metas lidar2img after dataset-side ida composition",
        "motion_compensation_source": "mmdet3d.models.sparsedetectors.mcqm.transform_points_between_egos",
    })
    write_json(Q2F_ARTIFACTS_DIR / "fpn_contract.json", {
        "selected_shallow_module_path": materialized["selected_shallow_module_path"],
        "selected_shallow_shape": materialized["selected_shallow_shape"],
        "final_fpn_shapes": materialized["fpn_level_shapes"],
        "camera_order": materialized["camera_order"],
        "camera_order_source": "img_metas[0]['filename'] inferred current-frame order",
        "replacement_scope": "current frame failed cameras shallow hard replacement only",
        "healthy_camera_passthrough": True,
        "failed_camera_replacement": "raw reconstructed shallow feature",
    })
    write_json(Q2F_ARTIFACTS_DIR / "freeze_manifest.json", freeze_manifest)
    write_csv(
        Q2F_ARTIFACTS_DIR / "trainable_parameters.csv",
        [{"name": name} for name in freeze_manifest["trainable_parameter_keys"]],
    )

    if dry_run:
        report = {
            "status": "dry_run",
            "sample_pair": {"prev_index": int(prev_index), "curr_index": int(curr_index)},
            "topology_audit": topology_audit,
            "materialized": materialized,
            "freeze_manifest": freeze_manifest,
        }
        write_json(Q2F_ARTIFACTS_DIR / "preflight_report.json", report)
        write_json(Q2F_ARTIFACTS_DIR / "blocking_issues.json", {"blocking_issue_count": 0, "blocking_issues": []})
        write_json(Q2F_ARTIFACTS_DIR / "decision.json", {"status": "MCQM_Q2F_V1_NOT_READY_FOR_TRAINING", "reason": "dry_run_only"})
        write_json(Q2F_ARTIFACTS_DIR / "test_report.json", {"status": "completed", "decision_status": "MCQM_Q2F_V1_NOT_READY_FOR_TRAINING"})
        return report

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("Q2F preflight found no trainable parameters")
    if not bool(topology_audit.get("topology_valid_for_strict_algorithm", False)):
        blocking_issues.append({"type": "STRICT_SHALLOW_TO_FPN_TOPOLOGY_NOT_SUPPORTED"})
    expected_failed_names = set(cfg.get("mcqm_q2f_failed_camera_names", []))
    camera_order = [str(name) for name in materialized.get("camera_order", [])]
    if materialized["camera_count"] <= 0:
        blocking_issues.append({"type": "invalid_camera_count", "camera_count": int(materialized["camera_count"])})
    if len(camera_order) != materialized["camera_count"]:
        blocking_issues.append({
            "type": "camera_order_length_mismatch",
            "camera_count": int(materialized["camera_count"]),
            "camera_order_length": int(len(camera_order)),
        })
    if len(set(camera_order)) != len(camera_order):
        blocking_issues.append({"type": "duplicate_camera_order", "camera_order": camera_order})
    missing_failed_names = sorted(expected_failed_names - set(camera_order))
    if missing_failed_names:
        blocking_issues.append({
            "type": "missing_failed_camera_names",
            "missing_failed_camera_names": missing_failed_names,
            "camera_order": camera_order,
        })
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    def _run_q2f_forward_debug(current_batch_cuda: dict[str, Any], *, bypass: bool, perturb_eps: float = 0.0) -> dict[str, Any]:
        seed_q2f_memory_from_previous_frame(model, prev_cuda)
        current_inputs = normalized_model_batch_kwargs(current_batch_cuda)
        model_kwargs = {
            key: value
            for key, value in current_inputs.items()
            if key not in {"img", "img_metas"}
        }
        runtime_prev = getattr(model, "mcqm_v2_runtime_oracle", None)
        runtime_next = dict(runtime_prev) if isinstance(runtime_prev, dict) else {}
        runtime_next["q2f_shallow_bypass"] = bool(bypass)
        runtime_next["q2f_shallow_debug_perturb_eps"] = float(perturb_eps)
        model.mcqm_v2_runtime_oracle = runtime_next
        try:
            with torch.no_grad():
                outputs = model.forward_backbone(
                    current_inputs["img"],
                    current_inputs["img_metas"],
                    **model_kwargs,
                )
            return {
                "debug": copy.deepcopy(getattr(model, "latest_mcqm_q2f_debug", {})),
                "cache": copy.deepcopy(outputs.get("mcqm_q2f_cache", None)),
            }
        finally:
            model.mcqm_v2_runtime_oracle = runtime_prev

    model.eval()
    seed_q2f_memory_from_previous_frame(model, prev_cuda)
    base_out = run_q2f_current_frame_forward(model, curr_deg_cuda, mode="disabled")
    seed_q2f_memory_from_previous_frame(model, prev_cuda)
    bypass_run1_out = run_q2f_current_frame_forward(model, curr_deg_cuda, mode="bypass")
    seed_q2f_memory_from_previous_frame(model, prev_cuda)
    bypass_run2_out = run_q2f_current_frame_forward(model, curr_deg_cuda, mode="bypass")
    pred_temporal_base, _, _ = sw2.extract_standard_tensors(sw2.unwrap(curr_raw), base_out)
    pred_temporal_bypass1, _, _ = sw2.extract_standard_tensors(sw2.unwrap(curr_raw), bypass_run1_out)
    pred_temporal_bypass2, _, _ = sw2.extract_standard_tensors(sw2.unwrap(curr_raw), bypass_run2_out)
    parity_record = build_q2f_eval_parity_record(
        curr_index,
        pred_temporal_base,
        pred_temporal_bypass1,
        pred_temporal_bypass2,
    )
    write_json(Q2F_ARTIFACTS_DIR / "q2f_eval_parity.json", {"status": "preflight", "rows": [parity_record]})
    if not bool(parity_record["all_horizons_equal"]):
        blocking_issues.append({"type": "baseline_bypass_parity_failed", "parity_record": parity_record})

    debug_nominal = _run_q2f_forward_debug(curr_deg_cuda, bypass=False, perturb_eps=0.0)
    debug_perturb = _run_q2f_forward_debug(curr_deg_cuda, bypass=False, perturb_eps=1e-4)
    q2f_debug = debug_nominal["debug"] if isinstance(debug_nominal, dict) else {}
    q2f_cache = debug_nominal["cache"] if isinstance(debug_nominal, dict) else {}
    perturb_cache = debug_perturb["cache"] if isinstance(debug_perturb, dict) else {}
    if not bool(q2f_debug.get("healthy_current_passthrough_exact", False)):
        blocking_issues.append({"type": "healthy_current_passthrough_violation"})
    if not bool(q2f_debug.get("history_passthrough_exact", False)):
        blocking_issues.append({"type": "history_passthrough_violation"})
    if not bool(q2f_debug.get("failed_hard_replacement_exact", False)):
        blocking_issues.append({"type": "failed_hard_replacement_violation"})
    if not bool(q2f_debug.get("failed_native_removed", False)):
        blocking_issues.append({"type": "failed_native_not_removed"})
    propagation_rows = []
    final_hashes = dict(q2f_cache.get("final_fpn_hashes", {})) if isinstance(q2f_cache, dict) else {}
    perturb_hashes = dict(perturb_cache.get("final_fpn_hashes", {})) if isinstance(perturb_cache, dict) else {}
    for key, value in final_hashes.items():
        changed = value != perturb_hashes.get(key, "")
        propagation_rows.append({"fpn_key": key, "changed_under_shallow_perturbation": bool(changed)})
        if not changed:
            blocking_issues.append({
                "type": "selected_shallow_node_does_not_propagate_to_all_required_fpn_levels",
                "fpn_key": key,
            })
    if hasattr(model, "set_epoch"):
        model.set_epoch(max(int(getattr(model, "finetune_epoch", 0)), 1))
    set_mcqm_q2f_training_mode(model)

    forward_rows = []
    backward_rows = []
    grad_norm_per_iter = []
    base_core_sha_before = model_state_sha256(
        model,
        excluded_prefixes=("mcqm_q2f_shallow_",),
        excluded_keys=Q2F_BASE_RUNTIME_STATE_KEYS,
    )
    for iteration in range(1):
        seed_q2f_memory_from_previous_frame(model, prev_cuda)
        optimizer.zero_grad(set_to_none=True)
        iter_inputs = normalized_model_batch_kwargs(curr_deg_cuda)
        teacher_img, teacher_metas = direct_model_inputs_from_batch(curr_clean_cuda)
        iter_inputs["mcqm_q2f_teacher_img"] = teacher_img
        iter_inputs["mcqm_q2f_teacher_img_metas"] = teacher_metas
        losses = model(return_loss=True, **iter_inputs)
        feature_l1 = losses.get("loss_mcqm_q2f_feature_l1", None)
        feature_cos = losses.get("loss_mcqm_q2f_feature_cos", None)
        feature_loss = None
        if torch.is_tensor(feature_l1) and torch.is_tensor(feature_cos):
            feature_loss = feature_l1 + feature_cos
            if not bool(feature_loss.requires_grad):
                blocking_issues.append({
                    "type": "q2f_feature_distillation_loss_detached",
                })
        else:
            blocking_issues.append({
                "type": "q2f_feature_distillation_loss_missing",
                "has_feature_l1": torch.is_tensor(feature_l1),
                "has_feature_cos": torch.is_tensor(feature_cos),
            })
        total_loss = _q2f_total_loss(model, losses)
        scalar_losses = _q2f_tensor_losses(losses)
        feature_grad_norms = {}
        if torch.is_tensor(feature_loss) and bool(feature_loss.requires_grad):
            q2f_named_params = [
                (name, param)
                for name, param in model.named_parameters()
                if name.startswith("mcqm_q2f_shallow_")
                and param.requires_grad
            ]
            if not q2f_named_params:
                blocking_issues.append({
                    "type": "q2f_feature_gradient_parameters_missing",
                })
            else:
                feature_grads = torch.autograd.grad(
                    feature_loss,
                    [param for _, param in q2f_named_params],
                    retain_graph=True,
                    allow_unused=True,
                )
                projector_nonzero = False
                decoder_nonzero = False
                for (name, _param), grad in zip(q2f_named_params, feature_grads):
                    grad_norm = 0.0
                    if grad is not None:
                        grad_norm = float(grad.detach().norm().item())
                        if not math.isfinite(grad_norm):
                            blocking_issues.append({
                                "type": "non_finite_feature_gradient",
                                "name": name,
                            })
                    feature_grad_norms[name] = grad_norm
                    if name.startswith("mcqm_q2f_shallow_query_projector.") and grad_norm > 0.0:
                        projector_nonzero = True
                    if name.startswith("mcqm_q2f_shallow_decoder.") and grad_norm > 0.0:
                        decoder_nonzero = True
                if not projector_nonzero:
                    blocking_issues.append({
                        "type": "q2f_feature_gradient_missing_projector",
                    })
                if not decoder_nonzero:
                    blocking_issues.append({
                        "type": "q2f_feature_gradient_missing_decoder",
                    })
        total_loss.backward()
        q2f_grad_norms = {}
        nonzero_grad = False
        for name, param in model.named_parameters():
            if not name.startswith("mcqm_q2f_"):
                if param.grad is not None and torch.is_tensor(param.grad) and bool(torch.isfinite(param.grad).all().item()) and float(param.grad.abs().sum().item()) > 0.0:
                    blocking_issues.append({"type": "frozen_parameter_has_gradient", "name": name})
                continue
            if param.grad is None:
                q2f_grad_norms[name] = 0.0
                continue
            grad_norm = float(param.grad.detach().norm().item())
            q2f_grad_norms[name] = grad_norm
            if not math.isfinite(grad_norm):
                blocking_issues.append({"type": "non_finite_gradient", "name": name})
            if grad_norm > 0.0:
                nonzero_grad = True
        if not nonzero_grad:
            blocking_issues.append({"type": "all_q2f_gradients_zero", "iteration": int(iteration)})
        optimizer.step()
        forward_rows.append({"iteration": int(iteration), **scalar_losses, "loss_total": float(total_loss.detach().cpu().item())})
        backward_rows.append({
            "iteration": int(iteration),
            "gradient_norm_per_q2f_module": q2f_grad_norms,
            "feature_gradient_norm_per_q2f_module": feature_grad_norms,
        })
        grad_norm_per_iter.append(q2f_grad_norms)
    base_core_sha_after = model_state_sha256(
        model,
        excluded_prefixes=("mcqm_q2f_shallow_",),
        excluded_keys=Q2F_BASE_RUNTIME_STATE_KEYS,
    )
    if base_core_sha_before != base_core_sha_after:
        blocking_issues.append({
            "type": "base_core_model_state_changed",
            "before": base_core_sha_before,
            "after": base_core_sha_after,
        })

    level_rows = q2f_debug.get("level_rows", []) if isinstance(q2f_debug, dict) else []
    supported_cells = sum(int(row.get("supported_cell_count", 0)) for row in level_rows if isinstance(row, dict))
    valid_points = sum(int(row.get("valid_projected_point_count", 0)) for row in level_rows if isinstance(row, dict))
    q2f_camera_order = [str(name) for name in q2f_debug.get("camera_order", [])] if isinstance(q2f_debug, dict) else []
    if q2f_camera_order != camera_order:
        blocking_issues.append({
            "type": "runtime_camera_order_mismatch",
            "materialized_camera_order": camera_order,
            "runtime_camera_order": q2f_camera_order,
        })
    if supported_cells <= 0:
        blocking_issues.append({"type": "no_supported_cells", "sample_index": int(curr_index)})
    if valid_points <= 0:
        blocking_issues.append({"type": "no_valid_projected_points", "sample_index": int(curr_index)})

    write_json(Q2F_ARTIFACTS_DIR / "smoke_forward_report.json", {"rows": forward_rows})
    write_json(Q2F_ARTIFACTS_DIR / "smoke_backward_report.json", {"rows": backward_rows})
    preflight_report = {
        "status": "completed",
        "sample_pair": {"prev_index": int(prev_index), "curr_index": int(curr_index)},
        "topology_audit": topology_audit,
        "materialized": materialized,
        "freeze_manifest": freeze_manifest,
        "q2f_debug": q2f_debug,
        "parity_record": parity_record,
        "propagation_rows": propagation_rows,
        "valid_projected_point_count": int(valid_points),
        "supported_cell_count": int(supported_cells),
        "base_core_sha_before": base_core_sha_before,
        "base_core_sha_after": base_core_sha_after,
        "gradient_iterations": grad_norm_per_iter,
    }
    write_json(Q2F_ARTIFACTS_DIR / "preflight_report.json", preflight_report)
    write_json(Q2F_ARTIFACTS_DIR / "blocking_issues.json", {"blocking_issue_count": int(len(blocking_issues)), "blocking_issues": blocking_issues})
    decision_status = "MCQM_Q2F_V1_READY_FOR_TRAINING" if not blocking_issues else "MCQM_Q2F_V1_NOT_READY_FOR_TRAINING"
    decision = {
        "status": decision_status,
        "blocking_issue_count": int(len(blocking_issues)),
        "sample_pair": {"prev_index": int(prev_index), "curr_index": int(curr_index)},
        "valid_projected_point_count": int(valid_points),
        "supported_cell_count": int(supported_cells),
    }
    write_json(Q2F_ARTIFACTS_DIR / "decision.json", decision)
    write_json(Q2F_ARTIFACTS_DIR / "test_report.json", {
        "status": "completed" if not blocking_issues else "failed",
        "decision_status": decision_status,
        "blocking_issue_count": int(len(blocking_issues)),
    })
    return preflight_report


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"start mode={args.mode} seed={args.seed}")
    if args.mode == "contract":
        write_contract()
        return
    if args.mode == "pairs":
        _, dataset, _, _ = build_runtime(train=False, mcqm_cfg=None, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
        rows = build_pairs(dataset, args.sample_start, args.sample_end)
        write_json(ARTIFACTS_DIR / "pair_manifest.json", rows)
        return
    if args.mode == "stage_a":
        write_contract()
        print(json.dumps(run_stage_a(args.sample_index), indent=2, ensure_ascii=False))
        return
    if args.mode == "stage_a_pair":
        write_contract()
        print(json.dumps(sequential_pair_smoke(args.sample_index), indent=2, ensure_ascii=False))
        return
    if args.mode == "stage_b_smoke":
        print(json.dumps(train_mcqm_stage("stage_b", args.sample_start, args.sample_end, args.iters, args.lr, args.seed), indent=2, ensure_ascii=False))
        return
    if args.mode == "stage_c_smoke":
        init_ckpt = CHECKPOINT_DIR / "stage_b_iter0001.pth"
        if not init_ckpt.exists():
            init_ckpt = CHECKPOINT_PATH
        print(json.dumps(train_mcqm_stage("stage_c", args.sample_start, args.sample_end, args.iters, args.lr * 0.1, args.seed, init_checkpoint=init_ckpt), indent=2, ensure_ascii=False))
        return
    if args.mode == "train_stage_b":
        print(json.dumps(train_mcqm_stage("stage_b", args.sample_start, args.sample_end, args.stage_b_iters, args.lr, args.seed), indent=2, ensure_ascii=False))
        return
    if args.mode == "train_stage_c":
        init_ckpt = args.checkpoint or (CHECKPOINT_DIR / f"stage_b_iter{args.stage_b_iters:04d}.pth")
        print(json.dumps(train_mcqm_stage("stage_c", args.sample_start, args.sample_end, args.stage_c_iters, args.lr * 0.1, args.seed, init_checkpoint=init_ckpt), indent=2, ensure_ascii=False))
        return
    if args.mode == "train_stage_d":
        init_ckpt = args.checkpoint or (CHECKPOINT_DIR / f"stage_c_iter{args.stage_c_iters:04d}.pth")
        print(json.dumps(train_mcqm_stage("stage_d", args.sample_start, args.sample_end, args.stage_d_iters, args.lr * 0.05, args.seed, init_checkpoint=init_ckpt), indent=2, ensure_ascii=False))
        return
    if args.mode == "eval":
        ckpt = args.checkpoint or (CHECKPOINT_DIR / f"stage_d_iter{args.stage_d_iters:04d}.pth")
        print(json.dumps(run_full_eval(args.sample_start, args.sample_end, ckpt), indent=2, ensure_ascii=False))
        return
    if args.mode == "eval_fix_suite":
        ckpt = args.checkpoint or (CHECKPOINT_DIR / f"stage_d_iter{args.stage_d_iters:04d}.pth")
        print(json.dumps(run_fix_eval_suite(args.sample_start, args.sample_end, ckpt), indent=2, ensure_ascii=False))
        return
    if args.mode == "eval_v2_interface_oracle":
        print(json.dumps(run_mcqm_v2_interface_oracle_suite(args.sample_start, args.sample_end), indent=2, ensure_ascii=False))
        return
    if args.mode == "eval_v2_oracle_e2e_only":
        print(
            json.dumps(
                run_mcqm_v2_oracle_e2e_only(
                    args.sample_start,
                    args.sample_end,
                    dry_run=args.dry_run,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "eval_v2_decision_only":
        print(
            json.dumps(
                run_mcqm_v2_decision_only(args.sample_start, args.sample_end),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "audit_v2_geometry_only":
        print(
            json.dumps(
                run_mcqm_v2_geometry_only_audit(
                    args.sample_start,
                    args.sample_end,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "train_q2f_preflight":
        print(
            json.dumps(
                run_mcqm_q2f_preflight(
                    args.sample_start,
                    args.sample_end,
                    dry_run=args.dry_run,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "train_q2f":
        print(
            json.dumps(
                run_q2f_training(
                    args.sample_start,
                    args.sample_end,
                    args.iters,
                    args.lr,
                    args.seed,
                    init_checkpoint=args.checkpoint or CHECKPOINT_PATH,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "eval_q2f":
        print(
            json.dumps(
                run_q2f_eval(
                    args.sample_start,
                    args.sample_end,
                    args.checkpoint or build_q2f_train_checkpoint_path(args.iters),
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "full":
        write_contract()
        stage_b = train_mcqm_stage("stage_b", args.sample_start, args.sample_end, args.stage_b_iters, args.lr, args.seed)
        stage_c = train_mcqm_stage("stage_c", args.sample_start, args.sample_end, args.stage_c_iters, args.lr * 0.1, args.seed, init_checkpoint=stage_b["checkpoint_path"])
        stage_d = train_mcqm_stage("stage_d", args.sample_start, args.sample_end, args.stage_d_iters, args.lr * 0.05, args.seed, init_checkpoint=stage_c["checkpoint_path"])
        result = run_full_eval(args.sample_start, args.sample_end, stage_d["checkpoint_path"])
        result["stage_b"] = stage_b
        result["stage_c"] = stage_c
        result["stage_d"] = stage_d
        write_json(ARTIFACTS_DIR / "full_run.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
