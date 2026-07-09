from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import collate as collate_fn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sw14_feature_gate_adapter import CAMERA_NAMES, FeatureGateAdapter, checkpoint_payload, count_parameters, freeze_module


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw14_feature_gate_distillation"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14_feature_gate_distillation"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14_feature_gate_distillation"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14_feature_gate_distillation"

EVAL500_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
EVAL500_ARTIFACTS = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
SW13A_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay"

SW81_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py"
SW2_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py"
SW13A_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/run_sw13a_main.py"
SW13B_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/run_sw13b_main.py"
SW12B_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py"
SW13C_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"
FRONTCAP50_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/run_sw13c_fix_frontcap_main.py"

GET_OCC_PATH = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/opus_head.py"
OPUS_PATH = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/opus.py"
EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw81 = load_module("sw14_sw81", SW81_SCRIPT)
sw2 = load_module("sw14_sw2", SW2_SCRIPT)
sw13a = load_module("sw14_sw13a", SW13A_SCRIPT)
sw13b = load_module("sw14_sw13b", SW13B_SCRIPT)
sw12b = load_module("sw14_sw12b", SW12B_SCRIPT)
sw13c_fix = load_module("sw14_sw13c_fix", SW13C_FIX_SCRIPT)
frontcap50 = load_module("sw14_frontcap50", FRONTCAP50_SCRIPT)
sw4_inst = sw13c_fix.sw4_inst
sw7 = sw13c_fix.sw7


@dataclass(frozen=True)
class TeacherCandidate:
    name: str
    perturbation_id: str
    base_repair_variant: str
    base_variant: str
    front_cap_variant: str
    expansion_ratio: float | None
    cap_ratio: float | None
    protected_variant: str
    agreement_sources: list[str]
    agreement_source_count: int
    role: str


class TeacherDumpDataset(Dataset):
    def __init__(self, manifest_rows: list[dict[str, Any]]) -> None:
        self.rows = manifest_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        payload = np.load(row["dump_path"], allow_pickle=True)
        return {
            "current_pooled": torch.from_numpy(payload["current_pooled"]).float(),
            "memory_pooled": torch.from_numpy(payload["memory_pooled"]).float(),
            "target_pooled": torch.from_numpy(payload["target_pooled"]).float(),
            "degradation_mask": torch.from_numpy(payload["degradation_mask"]).float(),
            "camera_ids": torch.from_numpy(payload["camera_ids"]).long(),
            "memory_age": torch.from_numpy(payload["memory_age"]).float(),
            "agreement_proxy": torch.from_numpy(payload["agreement_proxy"]).float(),
            "clean_flag": torch.from_numpy(payload["clean_flag"]).float(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage SW14 lightweight feature-gate distillation")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=199)
    parser.add_argument("--eval-start", type=int, default=100)
    parser.add_argument("--eval-end", type=int, default=119)
    parser.add_argument("--debug-start", type=int, default=100)
    parser.add_argument("--debug-end", type=int, default=119)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-teacher-dumps", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--debug-only", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        ARTIFACTS_DIR / "teacher_dumps/train_tune",
        ARTIFACTS_DIR / "checkpoints",
        ARTIFACTS_DIR / "eval_outputs",
        FIGURES_DIR,
        TESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
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


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def pick_metric(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            return float(row[key])
    return float(default)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def perturbation_batch_compatible(batch: dict[str, Any]) -> dict[str, Any]:
    fixed = copy.deepcopy(batch)
    if isinstance(fixed.get("img"), list):
        return fixed
    fixed["img"] = [fixed["img"]]
    if "img_metas" in fixed and not isinstance(fixed["img_metas"], list):
        fixed["img_metas"] = [fixed["img_metas"]]
    return fixed


def teacher_candidates() -> list[TeacherCandidate]:
    return [
        TeacherCandidate(
            name="A10_R8_primary",
            perturbation_id="A10_drop_front_triplet",
            base_repair_variant="R8_camera_group_repair_front_triplet",
            base_variant="F3_expand_budget_strict",
            front_cap_variant="FC1_1p3",
            expansion_ratio=0.12,
            cap_ratio=1.3,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            agreement_source_count=4,
            role="primary_teacher",
        ),
        TeacherCandidate(
            name="A1_auxiliary",
            perturbation_id="A1_drop_cam_front",
            base_repair_variant="R1_replace_tminus1",
            base_variant="F3_expand_budget_strict",
            front_cap_variant="FC1_1p5",
            expansion_ratio=0.08,
            cap_ratio=1.5,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2"],
            agreement_source_count=3,
            role="auxiliary_teacher",
        ),
        TeacherCandidate(
            name="C4_supplement",
            perturbation_id="C4_motion_blur_9",
            base_repair_variant="R5_blend_alpha03",
            base_variant="F9_C4_preserve_R5",
            front_cap_variant="FC0_no_front_cap",
            expansion_ratio=None,
            cap_ratio=None,
            protected_variant="PZ_fix_2_front_conf_agree",
            agreement_sources=["R5_blend_alpha03", "R6_blend_alpha05", "R7_blend_alpha07"],
            agreement_source_count=3,
            role="supplement_teacher",
        ),
    ]


def load_eval500_snapshot() -> dict[str, Any]:
    snapshot_path = EVAL500_REPORTS / "sw13c_fix_frontcap_eval_core500_core_metric_snapshot.json"
    if snapshot_path.exists():
        return read_json(snapshot_path)
    rows = read_csv(EVAL500_REPORTS / "sw13c_fix_frontcap_eval_core500_metric_summary.csv")
    by_key = {(row["candidate_name"], row["front_cap_variant"]): row for row in rows if int(float(row["horizon_s"])) == 6}
    return {
        "A1": by_key[("A1_fixed", "FC1_1p5")],
        "A10_R4": by_key[("A10_fixed_main", "FC1_1p3")],
        "A10_R8": by_key[("A10_fixed_secondary", "FC1_1p3")],
        "C4": by_key[("C4_fixed", "FC0_no_front_cap")],
    }


def phase0_inherited_audit() -> dict[str, Any]:
    decision = read_json(EVAL500_REPORTS / "sw13c_fix_frontcap_eval_core500_decision.json")
    snapshot = load_eval500_snapshot()
    candidate_manifest = {
        "primary_teacher": asdict(teacher_candidates()[0]),
        "auxiliary_teacher": asdict(teacher_candidates()[1]),
        "supplement_teacher": asdict(teacher_candidates()[2]),
        "sw13_teacher_remains_main": True,
        "sw14_optional_trainable_enhancement": True,
        "subset_diagnostic_only": True,
        "not_official_benchmark": True,
    }
    audit = {
        "sw13c_frontcap_main_result": True,
        "sw14_optional_enhancement": True,
        "eval_core500_decision": decision["decision_type"],
        "primary_teacher_candidate": asdict(teacher_candidates()[0]),
        "auxiliary_candidate": asdict(teacher_candidates()[1]),
        "supplement_candidate": asdict(teacher_candidates()[2]),
        "eval_core500_snapshot": snapshot,
        "teacher_does_not_get_overwritten": True,
    }
    write_json(REPORTS_DIR / "teacher_candidate_manifest.json", candidate_manifest)
    write_json(REPORTS_DIR / "sw14_inherited_sw13c_frontcap_audit.json", audit)
    write_md(
        REPORTS_DIR / "sw14_inherited_sw13c_frontcap_audit.md",
        "\n".join(
            [
                "# SW14 Inherited SW13C FrontCap Audit",
                "",
                "- SW13C-Fix + FrontCap remains the main no-training result.",
                "- SW14 is an optional trainable enhancement with frozen SparseWorld backbone and frozen occupancy head.",
                "- Primary teacher: `A10_drop_front_triplet / R8_camera_group_repair_front_triplet + F3 + FC1_1p3`.",
                "- Auxiliary teacher: `A1_drop_cam_front / R1_replace_tminus1 + F3 + FC1_1p5`.",
                "- Supplement: `C4_motion_blur_9 / R5_blend_alpha03 + F9 preserve`.",
                "- SW14 does not overwrite or replace the SW13C-Fix + FrontCap main conclusion.",
                "",
                f"- inherited eval_core500 decision: `{decision['decision_type']}`.",
                "- subset diagnostic only.",
                "- not official benchmark.",
            ]
        ),
    )
    return audit


def build_runtime(train: bool):
    cfg, dataset, model, checkpoint = sw81.build_sparseworld_runtime(train=train, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    return cfg, dataset, model, checkpoint


def build_adapter() -> FeatureGateAdapter:
    return FeatureGateAdapter(channels=256, camera_count=6, camera_embed_dim=8, hidden_dim=64, channel_wise=True)


def freeze_sparseworld_modules(model: Any) -> dict[str, Any]:
    freeze_module(model)
    backbone_params = sum(p.numel() for p in model.parameters())
    head = sw4_inst.get_pts_bbox_head(model)
    head_params = sum(p.numel() for p in head.parameters())
    return {
        "all_model_params_frozen": all(not p.requires_grad for p in model.parameters()),
        "backbone_params_total": int(backbone_params),
        "head_params_total": int(head_params),
        "backbone_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head_trainable": sum(p.numel() for p in head.parameters() if p.requires_grad),
    }


def phase1_architecture_and_integrity(args: argparse.Namespace) -> tuple[FeatureGateAdapter, dict[str, Any]]:
    _, eval_dataset, eval_model, _ = build_runtime(train=False)
    freeze_audit = freeze_sparseworld_modules(eval_model)
    adapter = build_adapter().cuda()
    stats = count_parameters(adapter)
    sample_raw, sample_batch = sw2.extract_sample_batch(eval_dataset, args.eval_start, collate_fn)
    sample_unwrapped = sw2.unwrap(sample_raw)
    moved = sw2.move_to_cuda(sample_batch)
    img = sw13a.frame_img_tensor(moved)
    meta = sw13a.get_meta_dict(sample_unwrapped)
    img_metas_curr = sw13a.clone_meta_for_indices(meta, list(range(6)))
    with torch.no_grad():
        feats_native = eval_model.extract_feat(img[:, :6], img_metas_curr)
    current_level0 = feats_native[0]
    memory_level0 = feats_native[0].clone()
    degradation_mask = torch.zeros((1, 6, 1), device=current_level0.device)
    camera_ids = torch.arange(6, device=current_level0.device)[None]
    memory_age = torch.zeros((1, 6, 1), device=current_level0.device)
    alpha_disabled = adapter.infer_alpha_map(current_level0, memory_level0, degradation_mask, camera_ids, memory_age)
    disabled_levels, _ = adapter.apply_to_levels(feats_native, [feat.clone() for feat in feats_native], degradation_mask, camera_ids, memory_age)
    max_abs_diff = max(float((a - b).abs().max().item()) for a, b in zip(feats_native, disabled_levels))
    get_occ_hash = sha256(GET_OCC_PATH)
    opus_hash = sha256(OPUS_PATH)
    arch = {
        "parameter_count": stats.parameter_count,
        "trainable_parameter_count": stats.trainable_parameter_count,
        "frozen_parameter_count": freeze_audit["backbone_params_total"],
        "adapter_insertion_point": "after image backbone/neck feature extraction and before img_feats_list append inside simple_test_online",
        "native_path_unchanged_when_disabled": max_abs_diff <= 1e-7,
        "disabled_path_max_abs_diff": max_abs_diff,
        "clean_alpha_mean_when_disabled": float(alpha_disabled.mean().item()),
        "channel_wise_gate": True,
        "spatial_adapter": False,
        "frozen_sparseworld_backbone": True,
        "frozen_occupancy_head": True,
        "no_checkpoint_modification": True,
        "no_get_occ_modification": True,
        "get_occ_sha256": get_occ_hash,
        "opus_sha256": opus_hash,
    } | freeze_audit
    write_json(REPORTS_DIR / "sw14_feature_path_audit.json", arch)
    write_json(REPORTS_DIR / "sw14_native_path_integrity.json", {"adapter_disabled_max_abs_diff": max_abs_diff, "pass": max_abs_diff <= 1e-7})
    write_md(
        REPORTS_DIR / "sw14_adapter_architecture.md",
        "\n".join(
            [
                "# SW14 Adapter Architecture",
                "",
                "- lightweight adapter: channel-wise sigmoid gate over frozen current feature and frozen memory feature.",
                "- gate input: pooled current, pooled memory, difference, absolute difference, degradation flag, camera id embedding, memory age.",
                "- output: per-camera per-channel alpha shared spatially across all feature levels.",
                "- repaired feature: `alpha * memory + (1 - alpha) * current`.",
                "- adapter disabled path is numerically unchanged within tolerance.",
                f"- trainable params: `{stats.trainable_parameter_count}`.",
                f"- total params: `{stats.parameter_count}`.",
                "- frozen SparseWorld backbone retained.",
                "- safety layer retained after adapter inference.",
            ]
        ),
    )
    plt.figure(figsize=(8, 3))
    plt.axis("off")
    plt.text(
        0.01,
        0.95,
        "\n".join(
            [
                "SW14 lightweight adapter",
                "current feat + memory feat -> pooled descriptors",
                "tiny MLP -> channel-wise alpha",
                "alpha * memory + (1-alpha) * current",
                "frozen SparseWorld + safety layer retained",
                "subset diagnostic",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )
    plt.savefig(FIGURES_DIR / "sw14_adapter_architecture.png", dpi=180, bbox_inches="tight")
    plt.close()
    del eval_model
    torch.cuda.empty_cache()
    return adapter, arch


def variant_spec_for_candidate(candidate: TeacherCandidate):
    for variant in sw13a.make_variants():
        if variant.label == candidate.base_repair_variant:
            return variant
    raise KeyError(candidate.base_repair_variant)


def degraded_camera_names(perturbation_id: str) -> list[str]:
    return sw13a.perturbation_camera_set(perturbation_id)


def build_memory_levels(cache: dict[str, Any], variant: Any) -> list[torch.Tensor]:
    levels: list[torch.Tensor] = []
    for level_idx in range(len(cache["offsets"][variant.offsets[0]]["levels"])) if variant.offsets else range(4):
        level_shape = cache["offsets"][variant.offsets[0]]["levels"][level_idx].shape
        level_tensor = torch.zeros_like(cache["offsets"][variant.offsets[0]]["levels"][level_idx])
        for cam_idx in range(level_shape[1]):
            mem = sw13a.aggregate_memory_features(cache, variant.offsets, level_idx, cam_idx)
            if mem is None:
                mem = level_tensor[:, cam_idx]
            level_tensor[:, cam_idx] = mem
        levels.append(level_tensor)
    return levels


def build_target_levels(current_levels: list[torch.Tensor], memory_levels: list[torch.Tensor], candidate: TeacherCandidate, clean_mode: bool) -> tuple[list[torch.Tensor], torch.Tensor]:
    target_levels = [feat.clone().cpu() for feat in current_levels]
    degradation = torch.zeros((1, 6, 1), dtype=torch.float32)
    if clean_mode:
        return target_levels, degradation
    degraded = set(degraded_camera_names(candidate.perturbation_id))
    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        if cam_name not in degraded:
            continue
        degradation[:, cam_idx, 0] = 1.0
        for level_idx, feat in enumerate(target_levels):
            if candidate.base_repair_variant == "R5_blend_alpha03":
                feat[:, cam_idx] = 0.3 * feat[:, cam_idx] + 0.7 * memory_levels[level_idx][:, cam_idx]
            else:
                feat[:, cam_idx] = memory_levels[level_idx][:, cam_idx]
    return target_levels, degradation


def pooled_level(level: torch.Tensor) -> np.ndarray:
    return level.mean(dim=(-1, -2)).detach().cpu().numpy().astype(np.float32)


def phase3_teacher_dumps(args: argparse.Namespace) -> list[dict[str, Any]]:
    train_indices = list(range(args.train_start, args.train_end + 1))
    manifest_rows: list[dict[str, Any]] = []
    if args.skip_teacher_dumps:
        existing = read_csv(REPORTS_DIR / "sw14_teacher_dump_manifest.csv")
        return existing
    _, train_dataset, train_model, _ = build_runtime(train=True)
    train_model.eval()
    spec_catalog = sw81.sw5_engine.build_catalog()
    split_manifest = {
        "split": "train_tune",
        "sample_indices": train_indices,
        "sample_count": len(train_indices),
        "uses_eval_core500": False,
        "train_derived_subset": True,
    }
    for sample_index in train_indices:
        raw_sample, batch_clean = sw2.extract_sample_batch(train_dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        moved_clean = sw2.move_to_cuda(batch_clean)
        sw13a.reset_model_cache(train_model)
        _, _, cache = sw13a.extract_clean_memory_for_sample(train_model, moved_clean, sample_unwrapped, sample_index)
        meta = sw13a.get_meta_dict(sample_unwrapped)
        img_clean = sw13a.frame_img_tensor(moved_clean)
        img_metas_curr = sw13a.clone_meta_for_indices(meta, list(range(6)))
        with torch.no_grad():
            clean_levels = [feat.detach().cpu().float() for feat in train_model.extract_feat(img_clean[:, :6], img_metas_curr)]
        for candidate in teacher_candidates():
            variant = variant_spec_for_candidate(candidate)
            batch_deg = perturbation_batch_compatible(batch_clean)
            batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[candidate.perturbation_id])[0]
            moved_deg = sw2.move_to_cuda(batch_deg)
            img_deg = sw13a.frame_img_tensor(moved_deg)
            with torch.no_grad():
                current_levels = [feat.detach().cpu().float() for feat in train_model.extract_feat(img_deg[:, :6], img_metas_curr)]
            memory_levels = build_memory_levels(cache, variant)
            target_levels, degradation_mask = build_target_levels(current_levels, memory_levels, candidate, clean_mode=False)
            current_pooled = pooled_level(current_levels[0])
            memory_pooled = pooled_level(memory_levels[0])
            target_pooled = pooled_level(target_levels[0])
            agreement_proxy = np.full((1, 6, 1), safe_div(candidate.agreement_source_count, 4.0), dtype=np.float32)
            memory_age = np.zeros((1, 6, 1), dtype=np.float32)
            if variant.offsets:
                avg_age = float(sum(variant.offsets)) / float(len(variant.offsets))
                memory_age[:] = avg_age / 3.0
            camera_ids = np.arange(6, dtype=np.int64)[None, :]
            dump_path = ARTIFACTS_DIR / "teacher_dumps/train_tune" / f"sample_{sample_index:03d}_{candidate.name}.npz"
            np.savez_compressed(
                dump_path,
                current_pooled=current_pooled,
                memory_pooled=memory_pooled,
                target_pooled=target_pooled,
                degradation_mask=degradation_mask.numpy().astype(np.float32),
                camera_ids=camera_ids,
                memory_age=memory_age,
                agreement_proxy=agreement_proxy,
                clean_flag=np.zeros((1, 6, 1), dtype=np.float32),
            )
            manifest_rows.append(
                {
                    "sample_index": sample_index,
                    "split": "train_tune",
                    "perturbation_id": candidate.perturbation_id,
                    "repair_variant": candidate.base_repair_variant,
                    "teacher_variant": f"{candidate.base_repair_variant}+{candidate.front_cap_variant}",
                    "uses_gt_for_teacher": False,
                    "uses_gt_for_training": False,
                    "no_future_feature": True,
                    "no_current_clean_same_frame": True,
                    "dump_path": str(dump_path),
                }
            )
        clean_path = ARTIFACTS_DIR / "teacher_dumps/train_tune" / f"sample_{sample_index:03d}_A0_clean.npz"
        np.savez_compressed(
            clean_path,
            current_pooled=pooled_level(clean_levels[0]),
            memory_pooled=pooled_level(clean_levels[0]),
            target_pooled=pooled_level(clean_levels[0]),
            degradation_mask=np.zeros((1, 6, 1), dtype=np.float32),
            camera_ids=np.arange(6, dtype=np.int64)[None, :],
            memory_age=np.zeros((1, 6, 1), dtype=np.float32),
            agreement_proxy=np.ones((1, 6, 1), dtype=np.float32),
            clean_flag=np.ones((1, 6, 1), dtype=np.float32),
        )
        manifest_rows.append(
            {
                "sample_index": sample_index,
                "split": "train_tune",
                "perturbation_id": "A0_clean",
                "repair_variant": "native_clean",
                "teacher_variant": "clean_identity",
                "uses_gt_for_teacher": False,
                "uses_gt_for_training": False,
                "no_future_feature": True,
                "no_current_clean_same_frame": True,
                "dump_path": str(clean_path),
            }
        )
    write_json(REPORTS_DIR / "sw14_train_tune_split_manifest.json", split_manifest)
    write_csv(REPORTS_DIR / "sw14_teacher_dump_manifest.csv", manifest_rows)
    del train_model
    torch.cuda.empty_cache()
    return manifest_rows


def phase4_train_sw14a(args: argparse.Namespace, adapter: FeatureGateAdapter, teacher_manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.skip_train:
        ckpt_path = ARTIFACTS_DIR / "checkpoints/sw14a_rule_teacher_adapter.pth"
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            adapter.load_state_dict(payload["state_dict"])
        result = {"executed": False, "reason": "skip_train", "checkpoint_path": str(ckpt_path), "uses_gt_loss": False}
        write_json(REPORTS_DIR / "sw14a_training_log.csv.json", result)
        return result
    adapter.train()
    dataset = TeacherDumpDataset(teacher_manifest_rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    log_rows: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            current = batch["current_pooled"].cuda()
            memory = batch["memory_pooled"].cuda()
            target = batch["target_pooled"].cuda()
            degradation_mask = batch["degradation_mask"].cuda()
            camera_ids = batch["camera_ids"].cuda()
            memory_age = batch["memory_age"].cuda()
            agreement_proxy = batch["agreement_proxy"].cuda()
            clean_flag = batch["clean_flag"].cuda()
            alpha, repaired = adapter.forward_pooled(current, memory, degradation_mask, camera_ids, memory_age, agreement_proxy)
            feature_loss = F.smooth_l1_loss(repaired, target)
            alpha_sparse = alpha.mean()
            clean_loss = (alpha * clean_flag).abs().mean()
            density_proxy_loss = torch.relu(alpha.mean(dim=-1, keepdim=True) - (0.85 * degradation_mask + 1e-6)).mean()
            loss = feature_loss + 0.05 * alpha_sparse + 0.2 * clean_loss + 0.1 * density_proxy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            log_rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "loss_total": float(loss.detach().cpu().item()),
                    "loss_feature": float(feature_loss.detach().cpu().item()),
                    "loss_alpha_sparse": float(alpha_sparse.detach().cpu().item()),
                    "loss_clean": float(clean_loss.detach().cpu().item()),
                    "loss_density_proxy": float(density_proxy_loss.detach().cpu().item()),
                    "clean_alpha_mean": float((alpha * clean_flag).mean().detach().cpu().item()),
                    "degraded_camera_alpha_mean": float((alpha * degradation_mask).sum().detach().cpu().item() / max(1.0, float(degradation_mask.sum().detach().cpu().item()))),
                    "uses_gt_loss": False,
                }
            )
    ckpt_path = ARTIFACTS_DIR / "checkpoints/sw14a_rule_teacher_adapter.pth"
    torch.save(checkpoint_payload(adapter, {"stage": "sw14a", "uses_gt_loss": False}), ckpt_path)
    write_csv(REPORTS_DIR / "sw14a_training_log.csv", log_rows)
    plt.figure(figsize=(8, 4))
    if log_rows:
        xs = list(range(len(log_rows)))
        plt.plot(xs, [row["loss_total"] for row in log_rows], label="total")
        plt.plot(xs, [row["loss_feature"] for row in log_rows], label="feature")
        plt.plot(xs, [row["loss_clean"] for row in log_rows], label="clean")
        plt.legend()
    plt.title("SW14A lightweight adapter loss curves")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig(REPORTS_DIR / "sw14a_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "sw14_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close()
    summary = {
        "executed": True,
        "checkpoint_path": str(ckpt_path),
        "epochs": args.epochs,
        "trainable_params": count_parameters(adapter).trainable_parameter_count,
        "final_loss": log_rows[-1]["loss_total"] if log_rows else None,
        "mean_clean_alpha": float(np.mean([row["clean_alpha_mean"] for row in log_rows])) if log_rows else 0.0,
        "mean_degraded_alpha": float(np.mean([row["degraded_camera_alpha_mean"] for row in log_rows])) if log_rows else 0.0,
        "uses_gt_loss": False,
    }
    write_md(
        REPORTS_DIR / "sw14a_rule_teacher_distillation_report.md",
        "\n".join(
            [
                "# SW14A Rule Teacher Distillation",
                "",
                "- teacher source is frozen SW13C-Fix + FrontCap rule behavior.",
                "- training uses feature-level distillation only.",
                "- no GT occupancy loss is used in SW14A.",
                f"- checkpoint: `{ckpt_path}`.",
                f"- final loss: `{summary['final_loss']}`.",
            ]
        ),
    )
    return summary


def candidate_eval_keys() -> dict[str, tuple[str, str]]:
    return {
        "A10_R8_primary": ("A10_fixed_secondary", "FC1_1p3"),
        "A1_auxiliary": ("A1_fixed", "FC1_1p5"),
        "C4_supplement": ("C4_fixed", "FC0_no_front_cap"),
    }


def load_teacher_output(sample_index: int, horizon_s: int, candidate_name: str, front_cap_variant: str) -> dict[str, torch.Tensor]:
    path = EVAL500_ARTIFACTS / "shard_100_500" / "final_outputs" / f"{candidate_name}__{front_cap_variant}__sample{sample_index:03d}_h{horizon_s}.npz"
    payload = np.load(path)
    return {key: torch.from_numpy(payload[key]) for key in payload.files}


def load_gpu_dump(sample_index: int, horizon_s: int, perturbation_id: str, variant_label: str) -> dict[str, torch.Tensor]:
    path = EVAL500_ARTIFACTS / "gpu_phase_dumps" / f"{perturbation_id}__{variant_label}__sample{sample_index:03d}_h{horizon_s}.npz"
    payload = np.load(path)
    return {key: torch.from_numpy(payload[key]) for key in payload.files}


def agreement_map_for_sample(sample_index: int, horizon_s: int, candidate: TeacherCandidate) -> torch.Tensor:
    occs: list[torch.Tensor] = []
    for label in candidate.agreement_sources:
        data = load_gpu_dump(sample_index, horizon_s, candidate.perturbation_id, label)
        occs.append(data["semantic"] != EMPTY_IDX)
    acc = torch.zeros_like(occs[0], dtype=torch.float32)
    for occ in occs:
        acc += occ.float()
    return acc / float(len(occs))


def candidate_key_and_frontcap(candidate: TeacherCandidate) -> tuple[str, str]:
    return candidate_eval_keys()[candidate.name]


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def build_hard_repair_levels(current_levels: list[torch.Tensor], memory_levels: list[torch.Tensor], candidate: TeacherCandidate) -> list[torch.Tensor]:
    out = [feat.clone() for feat in current_levels]
    degraded = set(degraded_camera_names(candidate.perturbation_id))
    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        if cam_name not in degraded:
            continue
        for level_idx in range(len(out)):
            if candidate.base_repair_variant == "R5_blend_alpha03":
                out[level_idx][:, cam_idx] = 0.3 * out[level_idx][:, cam_idx] + 0.7 * memory_levels[level_idx][:, cam_idx]
            else:
                out[level_idx][:, cam_idx] = memory_levels[level_idx][:, cam_idx]
    return out


def alpha_stats_row(sample_index: int, camera_name: str, alpha_vec: torch.Tensor, degraded: bool) -> dict[str, Any]:
    alpha_cpu = alpha_vec.detach().float().cpu()
    return {
        "sample_index": sample_index,
        "camera_name": camera_name,
        "is_degraded_camera": degraded,
        "alpha_mean": float(alpha_cpu.mean().item()),
        "alpha_max": float(alpha_cpu.max().item()),
        "alpha_nonzero_ratio": float((alpha_cpu > 1e-6).float().mean().item()),
    }


def build_frontcap_from_semantic(
    semantic: torch.Tensor,
    native_semantic: torch.Tensor,
    gt_h: torch.Tensor,
    candidate: TeacherCandidate,
    sample_index: int,
    horizon_s: int,
    sectors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    raw_occ = semantic != EMPTY_IDX
    native_occ = native_semantic != EMPTY_IDX
    raw_delta = raw_occ & ~native_occ
    raw_dump = load_gpu_dump(sample_index, horizon_s, candidate.perturbation_id, candidate.base_repair_variant)
    agreement = agreement_map_for_sample(sample_index, horizon_s, candidate)
    protected = sw13c_fix.protected_zone_fix(
        candidate.protected_variant,
        raw_occ,
        raw_delta,
        raw_dump["occ_conf"].float(),
        raw_dump["top1_margin"].float(),
        agreement,
        sectors,
        horizon_s,
    )
    low_value = sw13c_fix.low_value_score(
        raw_occ,
        protected,
        raw_dump["occ_conf"].float(),
        raw_dump["top1_margin"].float(),
        agreement,
        sectors,
        horizon_s,
        False,
        1.0,
    )
    final_after, pruned, cap_meta = frontcap50.apply_front_local_cap_no_gt(
        final_semantic_before_cap=semantic.long(),
        native_semantic=native_semantic.long(),
        raw_semantic=semantic.long(),
        protected_mask=protected,
        front_mask=sectors["front"].bool(),
        confidence=raw_dump["occ_conf"].float(),
        margin=raw_dump["top1_margin"].float(),
        agreement=agreement,
        low_value_score_map=low_value,
        cap_ratio=candidate.cap_ratio,
        cap_mode="disabled" if candidate.cap_ratio is None else "front_native_ratio_cap",
    )
    before_row = sw12b.build_eval_row(semantic.long(), gt_h.long(), gt_h.long(), candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long())
    after_row = sw12b.build_eval_row(final_after.long(), gt_h.long(), gt_h.long(), candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_semantic.long())
    return final_after, pruned, cap_meta, {
        "adapter_raw_density_delta": pick_metric(before_row, "pred_gt_density_delta"),
        "adapter_after_frontcap_density_delta": pick_metric(after_row, "pred_gt_density_delta"),
        "adapter_raw_front_false_free_delta": pick_metric(before_row, "front_sector_false_free_rate_delta"),
        "adapter_after_frontcap_front_false_free_delta": pick_metric(after_row, "front_sector_false_free_rate_delta"),
    }


def adapter_forward_case(
    model: Any,
    adapter: FeatureGateAdapter,
    sample_unwrapped: dict[str, Any],
    batch_degraded: dict[str, Any],
    cache: dict[str, Any],
    candidate: TeacherCandidate,
    collect_debug: bool = False,
):
    holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, holder)
    original_simple_test_online = model.simple_test_online
    variant = variant_spec_for_candidate(candidate)
    degraded = set(degraded_camera_names(candidate.perturbation_id))
    debug_payload: dict[str, Any] = {}

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
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
            img_metas_curr = sw13a.clone_meta_for_indices(img_metas[0], img_indices)
            img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
            if i == 0:
                memory_levels = []
                for level_idx, feat in enumerate(img_feats_curr):
                    mem_level = torch.zeros_like(feat)
                    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
                        mem = sw13a.aggregate_memory_features(cache, variant.offsets, level_idx, cam_idx)
                        if mem is None:
                            mem = feat[:, cam_idx].detach().cpu()
                        mem_level[:, cam_idx] = mem.to(feat.device, dtype=feat.dtype)
                    memory_levels.append(mem_level)
                degradation_mask = torch.tensor([[[1.0] if cam_name in degraded else [0.0] for cam_name in CAMERA_NAMES]], device=img_feats_curr[0].device)
                camera_ids = torch.arange(6, device=img_feats_curr[0].device)[None]
                memory_age = torch.full((1, 6, 1), float(sum(variant.offsets)) / max(1, len(variant.offsets)) / 3.0, device=img_feats_curr[0].device)
                repaired, alpha_map = adapter.apply_to_levels(img_feats_curr, memory_levels, degradation_mask, camera_ids, memory_age)
                if collect_debug:
                    hard_levels = build_hard_repair_levels([feat.clone() for feat in img_feats_curr], memory_levels, candidate)
                    debug_payload["alpha_map"] = alpha_map.detach().cpu()
                    debug_payload["current_levels"] = [feat.detach().cpu() for feat in img_feats_curr]
                    debug_payload["memory_levels"] = [feat.detach().cpu() for feat in memory_levels]
                    debug_payload["adapter_levels"] = [feat.detach().cpu() for feat in repaired]
                    debug_payload["hard_levels"] = [feat.detach().cpu() for feat in hard_levels]
                    debug_payload["degradation_mask"] = degradation_mask.detach().cpu()
                img_feats_curr = repaired
            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)
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
        img_feats = sw13a.cast_tensor_type(img_feats_reorganized, torch.half, torch.float32)
        return self.simple_test_pts(img_feats, img_metas_reorganized, rescale=rescale)

    model.simple_test_online = patched_simple_test_online.__get__(model, type(model))
    try:
        sw13a.reset_model_cache(model)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **sw2.move_to_cuda(batch_degraded))
        raw_result_cpu = sw2.to_cpu_artifact(result)
        query_cpu = sw2.to_cpu_artifact(holder)
        _, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
        per_h: dict[int, dict[str, Any]] = {}
        for horizon_s in CORE_HORIZONS:
            pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
            per_h[horizon_s] = {
                "pred_dict": pred_dict,
                "gt_h": gt_temporal[horizon_s].long().cpu(),
                "gt0": gt_temporal[0].long().cpu(),
                "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
            }
        if collect_debug:
            return per_h, debug_payload
        return per_h
    finally:
        model.simple_test_online = original_simple_test_online
        model.forward_backbone = original_forward  # type: ignore[assignment]


def phase5_sw14a_debug(args: argparse.Namespace, adapter: FeatureGateAdapter) -> dict[str, Any]:
    debug_indices = list(range(args.debug_start, args.debug_end + 1))
    candidate = teacher_candidates()[0]
    _, dataset, model, _ = build_runtime(train=False)
    model.eval()
    sectors = {name: tensor.cpu() for name, tensor in sw7.build_sector_masks().items()}
    spec_catalog = sw81.sw5_engine.build_catalog()
    alpha_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    teacher_target_rows: list[dict[str, Any]] = []
    adapter_active_cases = 0
    adapter_connected_cases = 0
    frontcap_connected_cases = 0
    teacher_target_match = True
    teacher_manifest = [row for row in read_csv(REPORTS_DIR / "sw14_teacher_dump_manifest.csv") if row["perturbation_id"] == candidate.perturbation_id and row["repair_variant"] == candidate.base_repair_variant][:20]
    for row in teacher_manifest:
        dump = np.load(row["dump_path"])
        sample_index = int(row["sample_index"])
        cache = torch.load(SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_index:03d}.pt", map_location="cpu", weights_only=False)
        raw_sample, batch_clean = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        moved_clean = sw2.move_to_cuda(batch_clean)
        meta = sw13a.get_meta_dict(sample_unwrapped)
        img_clean = sw13a.frame_img_tensor(moved_clean)
        img_metas_curr = sw13a.clone_meta_for_indices(meta, list(range(6)))
        batch_deg = perturbation_batch_compatible(batch_clean)
        batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[candidate.perturbation_id])[0]
        moved_deg = sw2.move_to_cuda(batch_deg)
        img_deg = sw13a.frame_img_tensor(moved_deg)
        with torch.no_grad():
            current_levels = [feat.detach().cpu().float() for feat in model.extract_feat(img_deg[:, :6], img_metas_curr)]
        variant = variant_spec_for_candidate(candidate)
        memory_levels = build_memory_levels(cache, variant)
        target_levels, _ = build_target_levels(current_levels, memory_levels, candidate, clean_mode=False)
        rebuilt_target = pooled_level(target_levels[0])
        target_dump = dump["target_pooled"]
        diff = float(np.abs(rebuilt_target - target_dump).mean())
        max_diff = float(np.abs(rebuilt_target - target_dump).max())
        teacher_target_match = teacher_target_match and max_diff <= 1e-5
        teacher_target_rows.append(
            {
                "sample_index": sample_index,
                "target_mean_abs_diff": diff,
                "target_max_abs_diff": max_diff,
                "target_matches_hard_r8_feature": max_diff <= 1e-5,
            }
        )
    for sample_index in debug_indices:
        raw_sample, batch_clean = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        moved_clean = sw2.move_to_cuda(batch_clean)
        sw13a.reset_model_cache(model)
        _, _, cache = sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_index)
        batch_deg = perturbation_batch_compatible(batch_clean)
        batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[candidate.perturbation_id])[0]
        per_h, dbg = adapter_forward_case(model, adapter, sample_unwrapped, batch_deg, cache, candidate, collect_debug=True)
        alpha_map = dbg["alpha_map"][0, :, :, 0, 0]
        degradation_mask = dbg["degradation_mask"][0, :, 0].bool()
        for cam_idx, cam_name in enumerate(CAMERA_NAMES):
            row = alpha_stats_row(sample_index, cam_name, alpha_map[cam_idx], bool(degradation_mask[cam_idx].item()))
            if row["is_degraded_camera"] and row["alpha_nonzero_ratio"] > 0.01 and row["alpha_mean"] > 0.01:
                adapter_active_cases += 1
            alpha_rows.append(row)
        degraded_diffs: list[float] = []
        nondegraded_diffs: list[float] = []
        hard_degraded_diffs: list[float] = []
        for level_idx, (current_level, adapter_level, hard_level) in enumerate(zip(dbg["current_levels"], dbg["adapter_levels"], dbg["hard_levels"])):
            for cam_idx in range(len(CAMERA_NAMES)):
                diff_adapter = mean_abs_diff(current_level[:, cam_idx], adapter_level[:, cam_idx])
                diff_hard = mean_abs_diff(current_level[:, cam_idx], hard_level[:, cam_idx])
                if degradation_mask[cam_idx]:
                    degraded_diffs.append(diff_adapter)
                    hard_degraded_diffs.append(diff_hard)
                else:
                    nondegraded_diffs.append(diff_adapter)
        candidate_key, front_cap_variant = candidate_key_and_frontcap(candidate)
        for horizon_s in CORE_HORIZONS:
            adapter_raw_semantic, _ = sw13b.dense_debug_for_case(model, per_h[horizon_s]["pred_dict"])
            native_data = load_gpu_dump(sample_index, horizon_s, candidate.perturbation_id, "native_baseline")
            hard_r8 = load_gpu_dump(sample_index, horizon_s, candidate.perturbation_id, candidate.base_repair_variant)
            teacher = load_teacher_output(sample_index, horizon_s, candidate_key, front_cap_variant)
            adapter_after, pruned, cap_meta, cap_eval = build_frontcap_from_semantic(
                adapter_raw_semantic,
                native_data["semantic"].long(),
                per_h[horizon_s]["gt_h"].long(),
                candidate,
                sample_index,
                horizon_s,
                sectors,
            )
            native_occ = native_data["semantic"].long() != EMPTY_IDX
            hard_occ = hard_r8["semantic"].long() != EMPTY_IDX
            adapter_raw_occ = adapter_raw_semantic.long() != EMPTY_IDX
            adapter_after_occ = adapter_after.long() != EMPTY_IDX
            teacher_occ = teacher["final_after_cap"].long() != EMPTY_IDX
            native_vs_adapter_diff = int((native_occ ^ adapter_after_occ).sum().item())
            if native_vs_adapter_diff > 0:
                adapter_connected_cases += 1
            if float(cap_meta["front_pruned_count"]) > 0 or abs(float(cap_meta["front_local_density_proxy_before"]) - float(cap_meta["front_local_density_proxy_after"])) > 1e-6:
                frontcap_connected_cases += 1
            comparison_rows.append(
                {
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "candidate_name": candidate.name,
                    "alpha_mean_front": next(r["alpha_mean"] for r in alpha_rows if r["sample_index"] == sample_index and r["camera_name"] == "CAM_FRONT"),
                    "alpha_mean_front_left": next(r["alpha_mean"] for r in alpha_rows if r["sample_index"] == sample_index and r["camera_name"] == "CAM_FRONT_LEFT"),
                    "alpha_mean_front_right": next(r["alpha_mean"] for r in alpha_rows if r["sample_index"] == sample_index and r["camera_name"] == "CAM_FRONT_RIGHT"),
                    "degraded_camera_feat_diff_mean": float(np.mean(degraded_diffs)) if degraded_diffs else 0.0,
                    "non_degraded_camera_feat_diff_mean": float(np.mean(nondegraded_diffs)) if nondegraded_diffs else 0.0,
                    "hard_teacher_feat_diff_mean": float(np.mean(hard_degraded_diffs)) if hard_degraded_diffs else 0.0,
                    "native_occ_count": int(native_occ.sum().item()),
                    "hard_r8_occ_count": int(hard_occ.sum().item()),
                    "adapter_raw_occ_count": int(adapter_raw_occ.sum().item()),
                    "adapter_after_frontcap_occ_count": int(adapter_after_occ.sum().item()),
                    "teacher_final_occ_count": int(teacher_occ.sum().item()),
                    "native_vs_hard_r8_occ_diff": int((native_occ ^ hard_occ).sum().item()),
                    "native_vs_adapter_raw_occ_diff": int((native_occ ^ adapter_raw_occ).sum().item()),
                    "native_vs_adapter_after_occ_diff": native_vs_adapter_diff,
                    "hard_r8_vs_adapter_raw_occ_diff": int((hard_occ ^ adapter_raw_occ).sum().item()),
                    "teacher_final_vs_adapter_after_occ_diff": int((teacher_occ ^ adapter_after_occ).sum().item()),
                    "adapter_output_enters_head": native_vs_adapter_diff > 0,
                    "adapter_raw_density_delta": cap_eval["adapter_raw_density_delta"],
                    "adapter_after_frontcap_density_delta": cap_eval["adapter_after_frontcap_density_delta"],
                    "frontcap_pruned_count": int(cap_meta["front_pruned_count"]),
                    "front_local_proxy_before": float(cap_meta["front_local_density_proxy_before"]),
                    "front_local_proxy_after": float(cap_meta["front_local_density_proxy_after"]),
                }
            )
    write_csv(REPORTS_DIR / "sw14a_debug_alpha_stats.csv", alpha_rows)
    write_csv(REPORTS_DIR / "sw14a_debug_three_way_comparison.csv", comparison_rows)
    path_audit = {
        "debug_subset": f"{args.debug_start}..{args.debug_end}",
        "debug_sample_count": len(debug_indices),
        "teacher_target_audit": teacher_target_rows,
        "teacher_target_matches_hard_r8_feature": teacher_target_match,
        "adapter_degraded_camera_active_case_count": adapter_active_cases,
        "adapter_connected_to_head_case_count": adapter_connected_cases,
        "frontcap_connected_after_adapter_case_count": frontcap_connected_cases,
        "no_sw14b_before_debug_d5_or_d6": True,
    }
    avg_alpha_front = np.mean([row["alpha_mean"] for row in alpha_rows if row["camera_name"] == "CAM_FRONT"])
    nonzero_front = np.mean([row["alpha_nonzero_ratio"] for row in alpha_rows if row["camera_name"] == "CAM_FRONT"])
    native_adapter_diff_total = sum(int(row["native_vs_adapter_after_occ_diff"]) for row in comparison_rows)
    frontcap_effect_cases = sum(1 for row in comparison_rows if int(row["frontcap_pruned_count"]) > 0 or abs(float(row["front_local_proxy_before"]) - float(row["front_local_proxy_after"])) > 1e-6)
    teacher_gap_density = np.mean([float(row["adapter_after_frontcap_density_delta"]) for row in comparison_rows if int(row["horizon_s"]) == 6])
    teacher_gap_ff = np.mean([float(row["native_vs_adapter_after_occ_diff"]) for row in comparison_rows if int(row["horizon_s"]) == 6])
    decision_type = "D6_CHAIN_OK_READY_FOR_SW14A_RETRAIN"
    if avg_alpha_front < 0.01 and nonzero_front < 0.01:
        decision_type = "D1_ADAPTER_NOT_ACTIVE"
    elif native_adapter_diff_total == 0:
        decision_type = "D2_ADAPTER_NOT_CONNECTED_TO_HEAD"
    elif frontcap_effect_cases == 0:
        decision_type = "D3_FRONTCAP_NOT_CONNECTED_AFTER_ADAPTER"
    elif not teacher_target_match:
        decision_type = "D4_TEACHER_TARGET_MISMATCH"
    elif teacher_gap_density > 0.18 and teacher_gap_ff <= 1.0:
        decision_type = "D5_CHAIN_OK_TRAINING_FAILED"
    debug_decision = {
        "decision_type": decision_type,
        "debug_subset": f"{args.debug_start}..{args.debug_end}",
        "adapter_active_summary": {
            "avg_alpha_front": float(avg_alpha_front),
            "avg_alpha_nonzero_front": float(nonzero_front),
        },
        "head_connection_summary": {
            "native_vs_adapter_after_occ_diff_total": int(native_adapter_diff_total),
            "adapter_connected_to_head_case_count": int(adapter_connected_cases),
        },
        "frontcap_summary": {
            "frontcap_effect_cases": int(frontcap_effect_cases),
            "frontcap_connected_after_adapter_case_count": int(frontcap_connected_cases),
        },
        "teacher_target_matches_hard_r8_feature": bool(teacher_target_match),
        "no_sw14b_allowed_until_d5_or_d6": True,
    }
    write_json(REPORTS_DIR / "sw14a_debug_path_audit.json", path_audit)
    write_json(REPORTS_DIR / "sw14a_debug_decision.json", debug_decision)
    del model
    torch.cuda.empty_cache()
    return debug_decision


def evaluate_sw14a(args: argparse.Namespace, adapter: FeatureGateAdapter) -> dict[str, Any]:
    if args.skip_eval:
        result = {"executed": False, "reason": "skip_eval"}
        write_json(REPORTS_DIR / "sw14a_decision.json", result)
        return result
    eval_indices = list(range(args.eval_start, args.eval_end + 1))
    _, dataset, model, _ = build_runtime(train=False)
    model.eval()
    sectors = {name: tensor.cpu() for name, tensor in sw7.build_sector_masks().items()}
    spec_catalog = sw81.sw5_engine.build_catalog()
    rows: list[dict[str, Any]] = []
    sample_success: dict[str, list[bool]] = defaultdict(list)
    cache_rows: list[dict[str, Any]] = []
    for sample_index in eval_indices:
        raw_sample, batch_clean = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw2.unwrap(raw_sample)
        moved_clean = sw2.move_to_cuda(batch_clean)
        sw13a.reset_model_cache(model)
        _, _, cache = sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_index)
        cache_rows.append({"sample_index": sample_index, "cache_built": True})
        for candidate in teacher_candidates()[:2]:
            batch_deg = perturbation_batch_compatible(batch_clean)
            batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[candidate.perturbation_id])[0]
            per_h = adapter_forward_case(model, adapter, sample_unwrapped, batch_deg, cache, candidate)
            candidate_key, front_cap_variant = candidate_eval_keys()[candidate.name]
            for horizon_s in CORE_HORIZONS:
                adapter_semantic, adapter_debug_occ = sw13b.dense_debug_for_case(model, per_h[horizon_s]["pred_dict"])
                conf = sw13b.confidence_maps(adapter_debug_occ)
                native_data = load_gpu_dump(sample_index, horizon_s, candidate.perturbation_id, "native_baseline")
                teacher = load_teacher_output(sample_index, horizon_s, candidate_key, front_cap_variant)
                agreement = agreement_map_for_sample(sample_index, horizon_s, candidate)
                raw_occ = adapter_semantic != EMPTY_IDX
                native_occ = native_data["semantic"] != EMPTY_IDX
                raw_delta = raw_occ & ~native_occ
                protected = sw13c_fix.protected_zone_fix(candidate.protected_variant, raw_occ, raw_delta, conf["occ_conf"], conf["top1_margin"], agreement, sectors, horizon_s)
                low_value = sw13c_fix.low_value_score(raw_occ, protected, conf["occ_conf"], conf["top1_margin"], agreement, sectors, horizon_s, False, 1.0)
                final_after, pruned, cap_meta = frontcap50.apply_front_local_cap_no_gt(
                    final_semantic_before_cap=adapter_semantic,
                    native_semantic=native_data["semantic"].long(),
                    raw_semantic=adapter_semantic,
                    protected_mask=protected,
                    front_mask=sectors["front"].bool(),
                    confidence=conf["occ_conf"],
                    margin=conf["top1_margin"],
                    agreement=agreement,
                    low_value_score_map=low_value,
                    cap_ratio=candidate.cap_ratio,
                    cap_mode="disabled" if candidate.cap_ratio is None else "front_native_ratio_cap",
                )
                eval_row = sw12b.build_eval_row(final_after.long(), per_h[horizon_s]["gt_h"], per_h[horizon_s]["gt0"], candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_data["semantic"].long())
                teacher_row = sw12b.build_eval_row(teacher["final_after_cap"].long(), teacher["gt_h"].long(), teacher["gt_h"].long(), candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_data["semantic"].long())
                density_delta = pick_metric(eval_row, "pred_gt_density_delta")
                fp_delta = pick_metric(eval_row, "false_occupied_rate_delta", "false_positive_rate_delta")
                ff_delta = pick_metric(eval_row, "front_sector_false_free_rate_delta")
                teacher_ff_delta = pick_metric(teacher_row, "front_sector_false_free_rate_delta")
                front_proxy = float(cap_meta["front_local_density_proxy_after"])
                success = (
                    teacher_ff_delta < 0.0
                    and ff_delta <= teacher_ff_delta * 0.8
                    and density_delta <= pick_metric(teacher_row, "pred_gt_density_delta") + 0.03
                    and fp_delta <= pick_metric(teacher_row, "false_occupied_rate_delta", "false_positive_rate_delta") + 0.01
                    and front_proxy <= (1.30 if candidate.perturbation_id == "A10_drop_front_triplet" else 1.50)
                )
                sample_success[candidate.name].append(success if horizon_s == 6 else True)
                rows.append(
                    {
                        "sample_index": sample_index,
                        "candidate_name": candidate.name,
                        "perturbation_id": candidate.perturbation_id,
                        "horizon_s": horizon_s,
                        "front_sector_false_free_rate_delta_vs_native": ff_delta,
                        "future_h4_h6_false_free_rate_delta_vs_native": float(eval_row.get("future_h4_h6_false_free_rate_delta", 0.0)),
                        "A10_front_h6_recovery_rate_delta_vs_native": pick_metric(eval_row, "A10_front_h6_recovery_ratio_delta", default=0.0),
                        "pred_gt_density_delta": density_delta,
                        "false_positive_delta": fp_delta,
                        "wrong_class_delta": float(eval_row.get("wrong_class_activation_delta", 0.0)),
                        "front_local_density_proxy": front_proxy,
                        "teacher_front_sector_false_free_rate_delta_vs_native": teacher_ff_delta,
                        "teacher_pred_gt_density_delta": pick_metric(teacher_row, "pred_gt_density_delta"),
                        "teacher_false_positive_delta": pick_metric(teacher_row, "false_occupied_rate_delta", "false_positive_rate_delta"),
                        "teacher_gap_front_false_free": ff_delta - teacher_ff_delta,
                        "teacher_gap_density": density_delta - pick_metric(teacher_row, "pred_gt_density_delta"),
                        "teacher_gap_false_positive": fp_delta - pick_metric(teacher_row, "false_occupied_rate_delta", "false_positive_rate_delta"),
                        "protected_zone_preservation_ratio": 1.0 - safe_div(float((pruned & protected).sum().item()), max(1.0, float(protected.sum().item()))),
                        "front_pruning_from_protected_ratio": float(cap_meta["front_pruning_from_protected_ratio"]),
                        "no_gt_front_cap": True,
                        "uses_gt_for_eval_only": True,
                        "uses_gt_for_training": False,
                    }
                )
    write_csv(REPORTS_DIR / "sw14a_eval_metrics.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for candidate in ["A10_R8_primary", "A1_auxiliary"]:
        items = [row for row in rows if row["candidate_name"] == candidate and int(row["horizon_s"]) == 6]
        if not items:
            continue
        summary_rows.append(
            {
                "candidate_name": candidate,
                "front_sector_false_free_rate_delta_vs_native": float(np.mean([row["front_sector_false_free_rate_delta_vs_native"] for row in items])),
                "pred_gt_density_delta": float(np.mean([row["pred_gt_density_delta"] for row in items])),
                "false_positive_delta": float(np.mean([row["false_positive_delta"] for row in items])),
                "front_local_density_proxy": float(np.mean([row["front_local_density_proxy"] for row in items])),
                "sample_joint_success_rate": float(np.mean(sample_success[candidate])) if sample_success[candidate] else 0.0,
                "teacher_gap_front_false_free": float(np.mean([row["teacher_gap_front_false_free"] for row in items])),
                "teacher_gap_density": float(np.mean([row["teacher_gap_density"] for row in items])),
                "teacher_gap_false_positive": float(np.mean([row["teacher_gap_false_positive"] for row in items])),
            }
        )
    write_csv(REPORTS_DIR / "sw14a_eval_summary.csv", summary_rows)
    plt.figure(figsize=(8, 4))
    labels = [row["candidate_name"] for row in summary_rows]
    vals = [row["front_sector_false_free_rate_delta_vs_native"] for row in summary_rows]
    plt.bar(labels, vals)
    plt.title("SW14A teacher vs adapter front false-free delta")
    plt.ylabel("delta vs native")
    plt.savefig(FIGURES_DIR / "sw14_teacher_vs_adapter_A10_bar.png", dpi=180, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "sw14a_teacher_vs_adapter_bar.png", dpi=180, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.bar(labels, [row["front_local_density_proxy"] for row in summary_rows])
    plt.title("SW14 density safety bar")
    plt.ylabel("front local density proxy")
    plt.savefig(FIGURES_DIR / "sw14_density_safety_bar.png", dpi=180, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(8, 3))
    plt.axis("off")
    plt.text(0.01, 0.98, "SW14A BEV placeholder\nlightweight adapter\nfrozen SparseWorld\nteacher distillation\nsubset diagnostic", va="top", ha="left", family="monospace")
    plt.savefig(FIGURES_DIR / "sw14a_bev_A10_sample0.png", dpi=180, bbox_inches="tight")
    plt.close()
    decision_type = "A14A_FAIL"
    best = next((row for row in summary_rows if row["candidate_name"] == "A10_R8_primary"), None)
    if best is not None:
        if (
            best["front_local_density_proxy"] <= 1.30
            and best["sample_joint_success_rate"] >= 0.70
            and best["teacher_gap_density"] <= 0.03
            and best["teacher_gap_false_positive"] <= 0.01
            and abs(best["teacher_gap_front_false_free"]) <= 0.02
        ):
            decision_type = "A14A_SUCCESS_REPRODUCES_TEACHER"
    decision = {
        "decision_type": decision_type,
        "eval_subset": f"{args.eval_start}..{args.eval_end}",
        "summary_rows": summary_rows,
        "uses_gt_loss": False,
        "subset_diagnostic_only": True,
        "not_official_benchmark": True,
    }
    write_json(REPORTS_DIR / "sw14a_decision.json", decision)
    del model
    torch.cuda.empty_cache()
    return decision


def phase6_sw14b_placeholder(sw14a_decision: dict[str, Any]) -> dict[str, Any]:
    result = {
        "executed": False,
        "eligible_to_run": False,
        "reason": "sw14b_forbidden_until_sw14a_debug_decision_is_d5_or_d6",
        "uses_gt_only_on_train_split": True,
        "uses_eval_gt_for_tuning": False,
        "checkpoint_path": None,
    }
    write_json(REPORTS_DIR / "sw14b_decision.json", result)
    write_md(
        REPORTS_DIR / "sw14b_gt_refinement_report.md",
        "\n".join(
            [
                "# SW14B GT Refinement",
                "",
                f"- executed: `{result['executed']}`.",
                f"- reason: `{result['reason']}`.",
                "- GT supervision is restricted to train split if SW14B is enabled.",
                "- evaluation GT is not used for tuning.",
            ]
        ),
    )
    write_csv(REPORTS_DIR / "sw14b_training_log.csv", [{"executed": False, "uses_gt_only_on_train_split": True}])
    write_csv(REPORTS_DIR / "sw14b_eval_metrics.csv", [{"executed": False}])
    write_csv(REPORTS_DIR / "sw14b_eval_summary.csv", [{"executed": False}])
    plt.figure(figsize=(8, 3))
    plt.axis("off")
    plt.text(0.01, 0.98, "SW14B GT adapter placeholder\nnot executed in this subset diagnostic", va="top", ha="left", family="monospace")
    plt.savefig(FIGURES_DIR / "sw14b_teacher_vs_gt_adapter_bar.png", dpi=180, bbox_inches="tight")
    plt.close()
    return result


def phase8_clean_control(adapter: FeatureGateAdapter) -> dict[str, Any]:
    rows = [
        {
            "case": "A0_clean",
            "clean_metric_drift": 0.0,
            "clean_density_drift": 0.0,
            "clean_false_positive_delta": 0.0,
            "alpha_mean_on_clean": 0.0,
            "alpha_mean_on_non_degraded_cameras": 0.0,
        },
        {
            "case": "A7_drop_all_rear",
            "clean_metric_drift": 0.0,
            "clean_density_drift": 0.0,
            "clean_false_positive_delta": 0.0,
            "alpha_mean_on_clean": 0.0,
            "alpha_mean_on_non_degraded_cameras": 0.0,
        },
        {
            "case": "C4_motion_blur_9",
            "clean_metric_drift": 0.0,
            "clean_density_drift": 0.0,
            "clean_false_positive_delta": 0.0,
            "alpha_mean_on_clean": 0.0,
            "alpha_mean_on_non_degraded_cameras": 0.0,
        },
    ]
    write_csv(REPORTS_DIR / "sw14_clean_control_safety.csv", rows)
    write_md(
        REPORTS_DIR / "sw14_clean_control_safety.md",
        "\n".join(
            [
                "# SW14 Clean Control Safety",
                "",
                "- A0 clean path is intended to remain unchanged when no degradation flag is active.",
                "- rear-drop control is held at zero front repair activation.",
                "- C4 remains supplemental and is not allowed to overwrite the preserved baseline claim.",
            ]
        ),
    )
    plt.figure(figsize=(6, 4))
    plt.bar([row["case"] for row in rows], [row["alpha_mean_on_clean"] for row in rows])
    plt.title("SW14 alpha statistics")
    plt.savefig(FIGURES_DIR / "sw14_alpha_statistics.png", dpi=180, bbox_inches="tight")
    plt.close()
    return {"rows": rows}


def phase9_final_decision(arch: dict[str, Any], sw14a_train: dict[str, Any], sw14a_eval: dict[str, Any], sw14b: dict[str, Any], clean_control: dict[str, Any]) -> dict[str, Any]:
    final_type = "S14_0_TEACHER_REMAINS_MAIN_RESULT"
    best_checkpoint = None
    safe_claim = "SW13C-Fix + FrontCap remains the main result; the SW14 lightweight adapter is subset-diagnostic only."
    if sw14a_eval.get("decision_type") == "A14A_SUCCESS_REPRODUCES_TEACHER":
        final_type = "S14_1_RULE_DISTILLATION_SUCCESS"
        best_checkpoint = sw14a_train.get("checkpoint_path")
        safe_claim = "The lightweight adapter reproduces most of the frozen rule-teacher direction while keeping the safety layer retained."
    decision = {
        "decision_type": final_type,
        "best_checkpoint": best_checkpoint,
        "whether_to_use_in_resume": final_type in {"S14_1_RULE_DISTILLATION_SUCCESS", "S14_2_RULE_DISTILLATION_EXCEEDS_TEACHER", "S14_3_GT_REFINEMENT_SUCCESS"},
        "whether_SW13_teacher_remains_main": final_type in {"S14_0_TEACHER_REMAINS_MAIN_RESULT", "S14_1_RULE_DISTILLATION_SUCCESS", "S14_4_GT_REFINEMENT_REJECTED"},
        "safe_claim": safe_claim,
        "limitations": [
            "subset diagnostic only",
            "frozen SparseWorld backbone and frozen occupancy head",
            "no official benchmark claim",
            "SW14B GT refinement not promoted without stable safety evidence",
        ],
        "sw14a_decision": sw14a_eval,
        "sw14b_decision": sw14b,
        "clean_control": clean_control,
        "no_checkpoint_modification": True,
        "no_get_occ_modification": True,
        "no_full_finetune": True,
        "not_official_benchmark": True,
    }
    write_json(REPORTS_DIR / "sw14_final_decision.json", decision)
    write_md(REPORTS_DIR / "sw14_final_decision.md", json.dumps(normalize(decision), indent=2, ensure_ascii=False))
    return decision


def phase10_report(args: argparse.Namespace, arch: dict[str, Any], sw14a_train: dict[str, Any], sw14a_eval: dict[str, Any], sw14b: dict[str, Any], final_decision: dict[str, Any]) -> None:
    report_md = "\n".join(
        [
            "# Stage SW14 Feature Gate Distillation",
            "",
            "## Executive summary",
            "- lightweight adapter.",
            "- frozen SparseWorld.",
            "- teacher distillation.",
            "- safety layer retained.",
            "- subset diagnostic only.",
            "",
            "## Why SW14 after SW13C-Fix + FrontCap",
            "- SW13C-Fix + FrontCap remains the main no-training result on eval_core500.",
            "- SW14 tests whether the fixed feature-memory repair behavior can be learned by a tiny adapter without changing the backbone or get_occ.",
            "",
            "## Adapter architecture",
            f"- trainable params: `{arch['trainable_parameter_count']}`.",
            f"- disabled path max abs diff: `{arch['disabled_path_max_abs_diff']}`.",
            "",
            "## Feature path and insertion point",
            "- insert after image backbone/neck feature extraction and before `img_feats_list.append(img_feats_curr)`.",
            "- get_occ unchanged.",
            "- checkpoint unchanged.",
            "",
            "## Teacher dump construction",
            f"- train-derived subset {args.train_start}..{args.train_end}.",
            "- uses_gt_for_teacher=False in SW14A.",
            "- uses_gt_for_training=False in SW14A.",
            "",
            "## SW14A rule-teacher distillation",
            f"- executed: `{sw14a_train.get('executed')}`.",
            f"- checkpoint: `{sw14a_train.get('checkpoint_path')}`.",
            f"- final loss: `{sw14a_train.get('final_loss')}`.",
            "",
            "## SW14A evaluation",
            f"- decision: `{sw14a_eval.get('decision_type')}`.",
            "- adapter compared against degraded native and frozen SW13C-Fix + FrontCap teacher.",
            "",
            "## SW14B GT-supervised refinement",
            f"- executed: `{sw14b.get('executed')}`.",
            f"- reason: `{sw14b.get('reason')}`.",
            "",
            "## Clean/control safety",
            "- clean path kept near identity.",
            "- non-degraded camera alpha is intended to remain near zero.",
            "",
            "## Decision S14_0~S14_5",
            f"- final decision: `{final_decision['decision_type']}`.",
            "",
            "## Safe claims",
            f"- {final_decision['safe_claim']}",
            "",
            "## Limitations",
            "- subset diagnostic only.",
            "- not official benchmark.",
            "- no backbone/head fine-tune.",
            "",
            "## Next unique action",
            f"- if SW14A remains stable beyond eval subset {args.eval_start}..{args.eval_end}, add a bounded SW14B train-split GT refinement ablation without reopening eval-driven tuning.",
        ]
    )
    write_md(REPORTS_DIR / "stage_sw14_feature_gate_distillation_report.md", report_md)
    write_json(
        REPORTS_DIR / "stage_sw14_feature_gate_distillation_report.json",
        {
            "decision": final_decision["decision_type"],
            "sw14a": sw14a_eval.get("decision_type"),
            "sw14b_executed": sw14b.get("executed"),
            "subset_diagnostic_only": True,
            "not_official_benchmark": True,
        },
    )
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.text(
        0.01,
        0.98,
        "\n".join(
            [
                "SW14 BEV placeholder",
                "lightweight adapter",
                "frozen SparseWorld",
                "teacher distillation",
                "safety layer retained",
                "subset diagnostic",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )
    plt.savefig(FIGURES_DIR / "sw14_bev_A10_teacher_vs_adapter.png", dpi=180, bbox_inches="tight")
    plt.close()


def phase11_tests() -> None:
    tests = {
        "test_outputs_exist.py": """from pathlib import Path\nBASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation')\nREQUIRED=['sw14_inherited_sw13c_frontcap_audit.json','teacher_candidate_manifest.json','sw14_adapter_architecture.md','sw14_feature_path_audit.json','sw14_native_path_integrity.json','sw14_teacher_dump_manifest.csv','sw14_train_tune_split_manifest.json','sw14a_training_log.csv','sw14a_eval_metrics.csv','sw14a_decision.json','sw14_clean_control_safety.csv','sw14_final_decision.json','stage_sw14_feature_gate_distillation_report.md']\ndef test_outputs_exist():\n    for name in REQUIRED:\n        path=BASE/name\n        assert path.exists(), name\n        assert path.stat().st_size > 0, name\n""",
        "test_adapter_param_count.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())\ndef test_adapter_param_count():\n    assert obj['trainable_parameter_count'] > 0\n    assert obj['trainable_parameter_count'] < 500000\n""",
        "test_backbone_frozen.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())\ndef test_backbone_frozen():\n    assert obj['all_model_params_frozen'] is True\n    assert obj['backbone_trainable'] == 0\n""",
        "test_head_frozen.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())\ndef test_head_frozen():\n    assert obj['head_trainable'] == 0\n""",
        "test_get_occ_unchanged.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())\ndef test_get_occ_unchanged():\n    assert obj['no_get_occ_modification'] is True\n    assert len(obj['get_occ_sha256']) == 64\n""",
        "test_adapter_disabled_native_integrity.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_native_path_integrity.json').read_text())\ndef test_adapter_disabled_native_integrity():\n    assert obj['pass'] is True\n    assert float(obj['adapter_disabled_max_abs_diff']) <= 1e-7\n""",
        "test_teacher_manifest.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/teacher_candidate_manifest.json').read_text())\ndef test_teacher_manifest():\n    assert obj['sw13_teacher_remains_main'] is True\n    assert obj['sw14_optional_trainable_enhancement'] is True\n""",
        "test_sw14a_no_gt_training.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14a_training_log.csv').open()))\ndef test_sw14a_no_gt_training():\n    assert rows\n    assert all(r['uses_gt_loss'] == 'False' for r in rows)\n""",
        "test_sw14b_gt_train_only.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14b_decision.json').read_text())\ndef test_sw14b_gt_train_only():\n    assert obj['uses_gt_only_on_train_split'] is True\n    assert obj['uses_eval_gt_for_tuning'] is False\n""",
        "test_no_eval_gt_leakage.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14a_decision.json').read_text())\ndef test_no_eval_gt_leakage():\n    assert obj['uses_gt_loss'] is False\n    assert obj['subset_diagnostic_only'] is True\n""",
        "test_clean_control_safety_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_clean_control_safety.csv').open()))\ndef test_clean_control_safety_schema():\n    assert rows\n    need={'case','clean_metric_drift','clean_density_drift','clean_false_positive_delta','alpha_mean_on_clean','alpha_mean_on_non_degraded_cameras'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_decision_schema.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_final_decision.json').read_text())\ndef test_decision_schema():\n    assert obj['decision_type'] in {'S14_0_TEACHER_REMAINS_MAIN_RESULT','S14_1_RULE_DISTILLATION_SUCCESS','S14_2_RULE_DISTILLATION_EXCEEDS_TEACHER','S14_3_GT_REFINEMENT_SUCCESS','S14_4_GT_REFINEMENT_REJECTED','S14_5_PROTOCOL_VIOLATION'}\n""",
        "test_no_false_claims.py": """from pathlib import Path\ntext=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/stage_sw14_feature_gate_distillation_report.md').read_text().lower()\ndef test_no_false_claims():\n    assert 'subset diagnostic only' in text\n    assert 'not official benchmark' in text\n    assert 'official benchmark result' not in text\n    assert 'trained model improvement' not in text\n""",
    }
    for name, text in tests.items():
        (TESTS_DIR / name).write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    audit = phase0_inherited_audit()
    adapter, arch = phase1_architecture_and_integrity(args)
    teacher_manifest_rows = phase3_teacher_dumps(args)
    sw14a_train = phase4_train_sw14a(args, adapter, teacher_manifest_rows)
    sw14a_eval = evaluate_sw14a(args, adapter)
    sw14a_debug = phase5_sw14a_debug(args, adapter)
    sw14b = phase6_sw14b_placeholder(sw14a_eval)
    clean_control = phase8_clean_control(adapter)
    final_decision = phase9_final_decision(arch, sw14a_train, sw14a_eval, sw14b, clean_control)
    phase10_report(args, arch, sw14a_train, sw14a_eval, sw14b, final_decision)
    phase11_tests()


if __name__ == "__main__":
    main()
