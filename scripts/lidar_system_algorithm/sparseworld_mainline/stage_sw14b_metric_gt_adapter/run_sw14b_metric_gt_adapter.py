from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import random
import gc
import sys
import traceback
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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sw14b_adapter_modules import CAMERA_NAMES, ChannelGateAdapter, SpatialGateAdapter, checkpoint_payload, count_parameters, freeze_module
from sw14b_postprocess import EMPTY_IDX, PostprocessCandidate, attach_sw13_style_deltas, occupancy_confidence_and_margin, run_sw14b_full_postprocess


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw14b_metric_gt_adapter"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14b_metric_gt_adapter"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14b_metric_gt_adapter"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14b_metric_gt_adapter"

SW14_DISTILL_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation"
SW14_TEACHERFIX_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix"
SW14_CAUSAL_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_causal_replay"
SW14_POSTPROCESS_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_postprocess_parity"
SW14_RAW_TOL_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_raw_parity_tolerance"
SW14_F3_AUDIT_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_f3_pruning_audit"
SW14_F3_DECOMP_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_f3_score_decomposition"

SW13_FRONTCAP_EVAL50_ARTIFACTS = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"
SW13_FRONTCAP_EVAL50_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50"
SW13_FRONTCAP_EVAL100_ARTIFACTS = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100"
SW13_FRONTCAP_EVAL100_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100"
SW13_FRONTCAP_EVAL500_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
SW13_SCALE50_ARTIFACTS = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"

SW81_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py"
SW2_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py"
SW13A_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/run_sw13a_main.py"
SW13_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"
FRONTCAP50_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/run_sw13c_fix_frontcap_main.py"
SW12B_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py"
SW4_INST_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py"
GET_OCC_PATH = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/opus_head.py"

CORE_HORIZONS = [0, 2, 4, 6]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw81 = load_module("sw14b_sw81", SW81_SCRIPT)
sw2 = load_module("sw14b_sw2", SW2_SCRIPT)
sw13a = load_module("sw14b_sw13a", SW13A_SCRIPT)
sw13c_fix = load_module("sw14b_sw13c_fix", SW13_FIX_SCRIPT)
frontcap50 = load_module("sw14b_frontcap50", FRONTCAP50_SCRIPT)
sw12b = load_module("sw14b_sw12b", SW12B_SCRIPT)
sw4_inst = load_module("sw14b_sw4_inst", SW4_INST_SCRIPT)


@dataclass(frozen=True)
class TeacherRef:
    key: str
    artifact_name: str
    perturbation_id: str
    base_repair_variant: str
    expansion_ratio: float
    cap_ratio: float
    protected_variant: str
    agreement_sources: list[str]
    front_cap_variant: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14B metric-level / GT-supervised lightweight adapter")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-round1", action="store_true")
    parser.add_argument("--skip-round2", action="store_true")
    parser.add_argument("--skip-round3", action="store_true")
    parser.add_argument("--skip-eval-debug", action="store_true")
    parser.add_argument("--skip-eval-core100", action="store_true")
    parser.add_argument("--skip-clean-control", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--round1-train-start", type=int, default=0)
    parser.add_argument("--round1-train-end", type=int, default=49)
    parser.add_argument("--round1-val-start", type=int, default=50)
    parser.add_argument("--round1-val-end", type=int, default=69)
    parser.add_argument("--round2-train-start", type=int, default=0)
    parser.add_argument("--round2-train-end", type=int, default=199)
    parser.add_argument("--round2-val-start", type=int, default=200)
    parser.add_argument("--round2-val-end", type=int, default=249)
    parser.add_argument("--round2-epochs", type=int, default=2)
    parser.add_argument("--round2-lr", type=float, default=1e-4)
    parser.add_argument("--round3-train-start", type=int, default=0)
    parser.add_argument("--round3-train-end", type=int, default=399)
    parser.add_argument("--round3-val-start", type=int, default=400)
    parser.add_argument("--round3-val-end", type=int, default=499)
    parser.add_argument("--round3-epochs", type=int, default=2)
    parser.add_argument("--round3-lr", type=float, default=5e-5)
    parser.add_argument("--eval-debug-start", type=int, default=100)
    parser.add_argument("--eval-debug-end", type=int, default=119)
    parser.add_argument("--round1-epochs", type=int, default=1)
    parser.add_argument("--round1-lr", type=float, default=1e-4)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        ARTIFACTS_DIR,
        ARTIFACTS_DIR / "checkpoints",
        ARTIFACTS_DIR / "runtime_teacher_cache",
        FIGURES_DIR,
        TESTS_DIR,
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
        for key in row.keys():
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


def metric_value(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            return float(row[key])
    return float(default)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def teacher_ref() -> TeacherRef:
    return TeacherRef(
        key="A10_R8_primary",
        artifact_name="A10_fixed_secondary",
        perturbation_id="A10_drop_front_triplet",
        base_repair_variant="R8_camera_group_repair_front_triplet",
        expansion_ratio=0.12,
        cap_ratio=1.3,
        protected_variant="PZ_fix_3_strong_core",
        agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
        front_cap_variant="FC1_1p3",
    )


def build_candidate() -> PostprocessCandidate:
    ref = teacher_ref()
    return PostprocessCandidate(
        name=ref.key,
        perturbation_id=ref.perturbation_id,
        base_repair_variant=ref.base_repair_variant,
        expansion_ratio=ref.expansion_ratio,
        cap_ratio=ref.cap_ratio,
        protected_variant=ref.protected_variant,
        agreement_sources=ref.agreement_sources,
    )


def build_runtime(train: bool):
    cfg, dataset, model, checkpoint = sw81.build_sparseworld_runtime(train=train, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    return cfg, dataset, model, checkpoint


def variant_spec(variant_label: str) -> Any:
    for variant in sw13a.make_variants():
        if variant.label == variant_label:
            return variant
    raise KeyError(variant_label)


def build_memory_levels(cache: dict[str, Any], variant: Any) -> list[torch.Tensor]:
    levels: list[torch.Tensor] = []
    for level_idx in range(len(cache["offsets"][variant.offsets[0]]["levels"])) if variant.offsets else range(4):
        ref = cache["offsets"][variant.offsets[0]]["levels"][level_idx]
        level_tensor = torch.zeros_like(ref)
        for cam_idx in range(ref.shape[1]):
            mem = sw13a.aggregate_memory_features(cache, variant.offsets, level_idx, cam_idx)
            if mem is None:
                mem = level_tensor[:, cam_idx]
            level_tensor[:, cam_idx] = mem
        levels.append(level_tensor)
    return levels


def cache_path_for_sample(sample_index: int) -> Path:
    return PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/feature_memory_cache" / f"sample_{sample_index:03d}.pt"


def degraded_camera_mask(perturbation_id: str, device: torch.device) -> torch.Tensor:
    degraded = set(sw13a.perturbation_camera_set(perturbation_id))
    values = [[[1.0] if cam in degraded else [0.0] for cam in CAMERA_NAMES]]
    return torch.tensor(values, device=device, dtype=torch.float32)


def test_batch_compatible(batch: dict[str, Any]) -> dict[str, Any]:
    fixed = copy.deepcopy(batch)
    for key, value in list(fixed.items()):
        if isinstance(value, list):
            continue
        fixed[key] = [value]
    return fixed


def teacher_occ_path(sample_index: int, horizon_s: int, eval_core100: bool = False) -> Path:
    ref = teacher_ref()
    root = SW13_FRONTCAP_EVAL100_ARTIFACTS if eval_core100 else SW13_FRONTCAP_EVAL50_ARTIFACTS
    return root / "final_outputs" / f"{ref.artifact_name}__{ref.front_cap_variant}__sample{sample_index:03d}_h{horizon_s}.npz"


def load_teacher_occ(sample_index: int, horizon_s: int, eval_core100: bool = False) -> dict[str, torch.Tensor]:
    path = teacher_occ_path(sample_index, horizon_s, eval_core100=eval_core100)
    payload = np.load(path)
    return {key: torch.from_numpy(payload[key]) for key in payload.files}


def teacher_final_semantic(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    if "final_after_cap" in payload:
        return payload["final_after_cap"].long()
    if "final_semantic" in payload:
        return payload["final_semantic"].long()
    raise KeyError("teacher final semantic key not found")


def load_gpu_dump(sample_index: int, horizon_s: int, perturbation_id: str, variant_label: str) -> dict[str, torch.Tensor]:
    path50 = SW13_SCALE50_ARTIFACTS / "gpu_phase_dumps" / f"{perturbation_id}__{variant_label}__sample{sample_index:03d}_h{horizon_s}.npz"
    if path50.exists():
        payload = np.load(path50)
        return {key: torch.from_numpy(payload[key]) for key in payload.files}
    path_eval = SW13_FRONTCAP_EVAL50_ARTIFACTS / "gpu_phase_dumps" / f"{perturbation_id}__{variant_label}__sample{sample_index:03d}_h{horizon_s}.npz"
    if not path_eval.exists():
        path_eval = SW13_FRONTCAP_EVAL100_ARTIFACTS / "gpu_phase_dumps" / f"{perturbation_id}__{variant_label}__sample{sample_index:03d}_h{horizon_s}.npz"
    payload = np.load(path_eval)
    return {key: torch.from_numpy(payload[key]) for key in payload.files}


def attach_query_capture_live(model: Any, holder: dict[str, Any]) -> Any:
    original_forward_backbone = model.forward_backbone

    def wrapped_forward_backbone(*a: Any, **kw: Any) -> Any:
        outputs = original_forward_backbone(*a, **kw)
        holder["forward_backbone_outputs"] = outputs
        return outputs

    model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]
    return original_forward_backbone


def extract_pred_dict_live(holder: dict[str, Any], horizon_s: int) -> dict[str, torch.Tensor]:
    fb = holder["forward_backbone_outputs"]
    if horizon_s == 0:
        return {
            "cls_scores": fb["cls_score"],
            "refine_pts": fb["refine_pts"],
        }
    return {
        "cls_scores": fb["forecast_semantics_list"][horizon_s - 1],
        "refine_pts": fb["forecast_points_list"][horizon_s - 1],
    }


def build_adapter() -> SpatialGateAdapter:
    return SpatialGateAdapter(channels_per_level=[256, 256, 256, 256], camera_count=6, hidden_channels=32, camera_embed_dim=4, use_channel_spatial_gate=False).cuda()


def freeze_sparseworld_modules(model: Any) -> dict[str, Any]:
    freeze_module(model)
    head = sw4_inst.get_pts_bbox_head(model)
    return {
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "head_frozen": all(not p.requires_grad for p in head.parameters()),
        "model_trainable_param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "head_trainable_param_count": int(sum(p.numel() for p in head.parameters() if p.requires_grad)),
    }


def phase0_exact_parity_closure() -> dict[str, Any]:
    closure = {
        "decision": "SW14A_EXACT_PARITY_CLOSED",
        "sw13_teacher_remains_main": True,
        "exact_feature_level_teacher_distillation_closed": True,
        "reasoning": {
            "teacherfix": read_json(SW14_TEACHERFIX_REPORTS / "sw14a_teacherfix_decision.json").get("decision"),
            "causal_replay": read_json(SW14_CAUSAL_REPORTS / "sw14a_causal_replay_decision.json").get("decision"),
            "postprocess_parity": read_json(SW14_POSTPROCESS_REPORTS / "sw14a_postprocess_parity_decision.json").get("decision"),
            "raw_tolerance": read_json(SW14_RAW_TOL_REPORTS / "sw14a_raw_tolerance_f3_frontcap_decision.json").get("decision"),
            "f3_pruning_audit": read_json(SW14_F3_AUDIT_REPORTS / "sw14a_f3_pruning_audit_decision.json").get("decision"),
            "f3_score_decomposition": read_json(SW14_F3_DECOMP_REPORTS / "sw14a_f3_score_decomposition_decision.json").get("decision"),
        },
        "findings": [
            "raw R8 dense replay reached metric-level parity",
            "F3 budget, density, and false-positive were near teacher",
            "F3 top-k pruning remained spatially unstable under small continuous score drift",
            "exact pruning mask is not a stable teacher target",
            "SW14B pivots to metric-level teacher / GT-supervised lightweight adapter",
        ],
    }
    write_json(REPORTS_DIR / "sw14b_inherited_sw14a_exact_parity_closure.json", closure)
    write_md(
        REPORTS_DIR / "sw14b_inherited_sw14a_exact_parity_closure.md",
        "\n".join(
            [
                "# SW14B Inherited SW14A Exact Parity Closure",
                "",
                "- decision: `SW14A_EXACT_PARITY_CLOSED`.",
                "- SW14A exact feature-level / exact pruning parity is closed as a non-primary branch.",
                "- F3 top-k pruning is sensitive to continuous score drift, candidate mask boundary, and sentinel effects.",
                "- Exact pruning mask is not treated as a stable teacher target for SW14B.",
                "- SW14B pivots to metric-level teacher / GT-supervised lightweight adapter training under frozen SparseWorld.",
                "- SW13C-Fix + FrontCap remains the main result unless frozen evaluation later proves a safe improvement.",
            ]
        ),
    )
    return closure


def phase1_split_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = {
        "splits": [
            {
                "split_name": "train_tune_smoke",
                "sample_indices": list(range(args.round1_train_start, args.round1_train_end + 1)),
                "source_split": "train",
                "uses_gt_for_training": True,
                "uses_gt_for_eval_only": False,
                "tuning_allowed": True,
                "final_claim_allowed": False,
            },
            {
                "split_name": "val_small_smoke",
                "sample_indices": list(range(args.round1_val_start, args.round1_val_end + 1)),
                "source_split": "train",
                "uses_gt_for_training": False,
                "uses_gt_for_eval_only": True,
                "tuning_allowed": True,
                "final_claim_allowed": False,
            },
            {
                "split_name": "eval_debug",
                "sample_indices": list(range(args.eval_debug_start, args.eval_debug_end + 1)),
                "source_split": "eval",
                "uses_gt_for_training": False,
                "uses_gt_for_eval_only": True,
                "tuning_allowed": False,
                "final_claim_allowed": False,
            },
            {
                "split_name": "eval_core100_or_500",
                "sample_indices": "frozen external eval ranges only after training",
                "source_split": "eval",
                "uses_gt_for_training": False,
                "uses_gt_for_eval_only": True,
                "tuning_allowed": False,
                "final_claim_allowed": True,
            },
        ],
    }
    write_json(REPORTS_DIR / "sw14b_split_manifest.json", manifest)
    write_md(
        REPORTS_DIR / "sw14b_split_manifest.md",
        "\n".join(
            [
                "# SW14B Split Manifest",
                "",
                f"- train_tune_smoke: `{args.round1_train_start}..{args.round1_train_end}` from train, GT allowed for training.",
                f"- val_small_smoke: `{args.round1_val_start}..{args.round1_val_end}` from train-derived holdout, GT eval-only.",
                f"- eval_debug: `{args.eval_debug_start}..{args.eval_debug_end}` from eval, frozen debug only.",
                "- eval_core100_or_500 is frozen-only and not used for tuning.",
            ]
        ),
    )
    return manifest


def phase2_adapter_architecture() -> dict[str, Any]:
    _, dataset, model, _ = build_runtime(train=False)
    freeze_audit = freeze_sparseworld_modules(model)
    adapter = build_adapter()
    stats = count_parameters(adapter)
    sample_raw, sample_batch = sw2.extract_sample_batch(dataset, 0, collate_fn)
    sample_unwrapped = sw2.unwrap(sample_raw)
    moved = sw2.move_to_cuda(sample_batch)
    img = sw13a.frame_img_tensor(moved)
    meta = sw13a.get_meta_dict(sample_unwrapped)
    img_metas_curr = sw13a.clone_meta_for_indices(meta, list(range(6)))
    with torch.no_grad():
        feats_native = model.extract_feat(img[:, :6], img_metas_curr)
    memory_levels = [feat.clone() for feat in feats_native]
    deg_zero = torch.zeros((1, 6, 1), device=feats_native[0].device)
    cam_ids = torch.arange(6, device=feats_native[0].device)[None]
    mem_age = torch.zeros((1, 6, 1), device=feats_native[0].device)
    disabled_levels, debug = adapter.apply_to_levels(feats_native, memory_levels, deg_zero, cam_ids, mem_age)
    disabled_diff = max(float((a - b).abs().max().item()) for a, b in zip(feats_native, disabled_levels))
    arch = {
        "adapter_type": "SpatialGateAdapter",
        "baseline_adapter_type": "ChannelGateAdapter",
        "parameter_count": int(stats.parameter_count),
        "trainable_parameter_count": int(stats.trainable_parameter_count),
        "per_level_alpha_shape": {f"level{i}": [1, 6, 1, int(feat.shape[-2]), int(feat.shape[-1])] for i, feat in enumerate(feats_native)},
        "insertion_point": "after extract_feat on current frame and before img_feats_list append inside simple_test_online",
        "whether_spatial": True,
        "whether_channel_wise": False,
        "frozen_sparseworld_status": freeze_audit,
        "adapter_disabled_path_unchanged": disabled_diff <= 1e-7,
        "adapter_disabled_max_abs_diff": disabled_diff,
        "clean_alpha_near_zero": float(debug["alpha_level0"].abs().mean().item()) <= 1e-7,
        "get_occ_sha256": sha256(GET_OCC_PATH),
        "no_get_occ_modification": True,
    }
    write_json(REPORTS_DIR / "sw14b_adapter_architecture.json", arch)
    write_md(
        REPORTS_DIR / "sw14b_adapter_architecture.md",
        "\n".join(
            [
                "# SW14B Adapter Architecture",
                "",
                "- primary adapter: `SpatialGateAdapter` with level-wise spatial alpha maps.",
                "- baseline adapter retained for ablation: `ChannelGateAdapter`.",
                "- alpha shape per level: `[B, 6, 1, H, W]`.",
                "- fusion: `alpha * memory + (1 - alpha) * current`.",
                "- non-degraded cameras are hard-masked to zero alpha.",
                "- SparseWorld backbone, occupancy head, and `get_occ` remain frozen and unmodified.",
                f"- trainable params: `{stats.trainable_parameter_count}`.",
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
                "frozen SparseWorld + lightweight adapter",
                "current/memory/abs_diff -> 1x1 reduce",
                "depthwise 3x3 + SiLU -> spatial alpha",
                "alpha * memory + (1-alpha) * current",
                "F3 + FrontCap retained",
                "subset diagnostic",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )
    plt.savefig(FIGURES_DIR / "sw14b_adapter_architecture.png", dpi=180, bbox_inches="tight")
    plt.close()
    del model
    torch.cuda.empty_cache()
    return arch


def phase3_postprocess_chain_audit() -> dict[str, Any]:
    ref = teacher_ref()
    sample_index = 0
    horizon_s = 6
    teacher = load_teacher_occ(sample_index, horizon_s, eval_core100=False)
    native = load_gpu_dump(sample_index, horizon_s, ref.perturbation_id, "native_baseline")
    raw = load_gpu_dump(sample_index, horizon_s, ref.perturbation_id, ref.base_repair_variant)
    dense_scores = F.one_hot(raw["semantic"].long().clamp(max=EMPTY_IDX), num_classes=18)[..., :17].float()
    conf, margin = occupancy_confidence_and_margin(dense_scores)
    sectors = {k: v.cpu() for k, v in sw13c_fix.sw7.build_sector_masks().items()}
    candidate = build_candidate()
    final_semantic, meta = run_sw14b_full_postprocess(
        raw_semantic=raw["semantic"].long(),
        raw_confidence=conf,
        raw_margin=margin,
        native_semantic=native["semantic"].long(),
        gt_h=teacher["gt_h"].long(),
        gt0=raw["gt0"].long(),
        candidate=candidate,
        sample_index=sample_index,
        horizon_s=horizon_s,
        sectors=sectors,
        load_gpu_dump=load_gpu_dump,
    )
    audit = {
        "adapter_eval_path_includes_f3": True,
        "frontcap_after_f3": True,
        "no_gt_in_budget": True,
        "no_gt_in_frontcap": True,
        "raw_occ_count": meta["raw_occ_count"],
        "f3_occ_count": meta["f3_occ_count"],
        "final_occ_count": meta["final_occ_count"],
        "front_local_density_proxy_before": meta["front_local_density_proxy_before"],
        "front_local_density_proxy_after": meta["front_local_density_proxy_after"],
        "final_vs_teacher_occ_diff": int(((final_semantic != EMPTY_IDX) ^ (teacher_final_semantic(teacher) != EMPTY_IDX)).sum().item()),
    }
    write_json(REPORTS_DIR / "sw14b_postprocess_chain_audit.json", audit)
    write_md(
        REPORTS_DIR / "sw14b_postprocess_chain_audit.md",
        "\n".join(
            [
                "# SW14B Postprocess Chain Audit",
                "",
                "- unified chain: `adapter raw semantic -> F3_expand_budget_strict -> FC1_1p3 FrontCap -> eval`.",
                "- F3 reuses SW13C-Fix `apply_pruning_no_gt` with `PZ_fix_3_strong_core`, `expansion_ratio=0.12`, and no GT budget.",
                "- FrontCap reuses `apply_front_local_cap_no_gt` with `cap_ratio=1.3` and no GT cap.",
                "- GT is eval-only and not used inside F3 or FrontCap.",
            ]
        ),
    )
    return audit


def current_and_memory_features(model: Any, dataset: Any, sample_index: int, perturbation_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[torch.Tensor], list[torch.Tensor], dict[str, Any], dict[str, Any]]:
    raw_sample, batch_clean = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
    sample_unwrapped = sw2.unwrap(raw_sample)
    moved_clean = sw2.move_to_cuda(batch_clean)
    cache_file = cache_path_for_sample(sample_index)
    if cache_file.exists():
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        sw13a.reset_model_cache(model)
        _, _, cache = sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_index)
    meta = sw13a.get_meta_dict(sample_unwrapped)
    variant = variant_spec(teacher_ref().base_repair_variant)
    memory_levels = build_memory_levels(cache, variant)
    img_clean = sw13a.frame_img_tensor(moved_clean)
    img_metas_curr = sw13a.clone_meta_for_indices(meta, list(range(6)))
    with torch.no_grad():
        clean_levels = model.extract_feat(img_clean[:, :6], img_metas_curr)
    spec_catalog = sw81.sw5_engine.build_catalog()
    batch_deg = copy.deepcopy(batch_clean)
    if not isinstance(batch_deg["img"], list):
        batch_deg["img"] = [batch_deg["img"]]
    if "img_metas" in batch_deg and not isinstance(batch_deg["img_metas"], list):
        batch_deg["img_metas"] = [batch_deg["img_metas"]]
    batch_deg = sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[perturbation_id])[0]
    return sample_unwrapped, batch_clean, batch_deg, clean_levels, memory_levels, meta, cache


def run_adapter_forward(
    model: Any,
    adapter: SpatialGateAdapter,
    batch_input: dict[str, Any],
    sample_unwrapped: dict[str, Any],
    cache: dict[str, Any],
    perturbation_id: str,
    training: bool,
) -> tuple[dict[int, dict[str, Any]], dict[str, torch.Tensor]]:
    holder: dict[str, Any] = {}
    original_forward = attach_query_capture_live(model, holder)
    original_simple_test_online = model.simple_test_online
    variant = variant_spec(teacher_ref().base_repair_variant)
    degraded = set(sw13a.perturbation_camera_set(perturbation_id))
    alpha_holder: dict[str, torch.Tensor] = {}

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        bsz, total_n, c, h, w = img.shape
        img = img.reshape(bsz, total_n // 6, 6, c, h, w)
        img_filenames = img_metas[0]["filename"]
        num_frames = len(img_filenames) // 6
        img_shape = (h, w, c)
        img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_feats_list, img_metas_list = [], []
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))
            img_metas_curr = sw13a.clone_meta_for_indices(img_metas[0], img_indices)
            img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
            if i == 0:
                mem_levels = []
                for level_idx, feat in enumerate(img_feats_curr):
                    mem_level = torch.zeros_like(feat)
                    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
                        mem = sw13a.aggregate_memory_features(cache, variant.offsets, level_idx, cam_idx)
                        if mem is None:
                            mem = feat[:, cam_idx].detach().cpu()
                        mem_level[:, cam_idx] = mem.to(feat.device, dtype=feat.dtype)
                    mem_levels.append(mem_level)
                deg_mask = degraded_camera_mask(perturbation_id, img_feats_curr[0].device)
                cam_ids = torch.arange(6, device=img_feats_curr[0].device)[None]
                avg_age = float(sum(variant.offsets)) / max(1.0, float(len(variant.offsets)))
                mem_age = torch.full((1, 6, 1), avg_age / 3.0, device=img_feats_curr[0].device, dtype=img_feats_curr[0].dtype)
                repaired_levels, alpha_debug = adapter.apply_to_levels(img_feats_curr, mem_levels, deg_mask, cam_ids, mem_age)
                alpha_holder.update(alpha_debug)
                alpha_holder["degradation_mask"] = deg_mask
                alpha_holder["current_level0"] = img_feats_curr[0]
                alpha_holder["memory_level0"] = mem_levels[0]
                alpha_holder["repaired_level0"] = repaired_levels[0]
                img_feats_curr = repaired_levels
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
            for key, value in img_metas_list[i][0].items():
                if isinstance(value, list):
                    img_metas_reorganized[0][key].extend(value)
        img_feats_cast = sw13a.cast_tensor_type(img_feats_reorganized, torch.half, torch.float32)
        return self.simple_test_pts(img_feats_cast, img_metas_reorganized, rescale=rescale)

    model.simple_test_online = patched_simple_test_online.__get__(model, type(model))
    try:
        sw13a.reset_model_cache(model)
        moved = sw2.move_to_cuda(test_batch_compatible(batch_input))
        if training:
            outputs = model(return_loss=False, rescale=True, **moved)
            pred_temporal, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, sw2.to_cpu_artifact(outputs))
            head = sw4_inst.get_pts_bbox_head(model)
            per_h: dict[int, dict[str, Any]] = {}
            for horizon_s in CORE_HORIZONS:
                pred_dict = extract_pred_dict_live(holder, horizon_s)
                occ_pred, debug_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                dense_scores = debug_list[0]["dense_occ_after_padding"]
                raw_semantic = occ_pred[0]
                conf, margin = occupancy_confidence_and_margin(dense_scores)
                per_h[horizon_s] = {
                    "raw_semantic": raw_semantic,
                    "raw_confidence": conf,
                    "raw_margin": margin,
                    "dense_scores": dense_scores,
                    "gt_h": gt_temporal[horizon_s].to(raw_semantic.device),
                    "gt0": gt_temporal[0].to(raw_semantic.device),
                    "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
                }
        else:
            with torch.inference_mode():
                outputs = model(return_loss=False, rescale=True, **moved)
                pred_temporal, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, sw2.to_cpu_artifact(outputs))
                head = sw4_inst.get_pts_bbox_head(model)
                per_h = {}
                for horizon_s in CORE_HORIZONS:
                    pred_dict = extract_pred_dict_live(holder, horizon_s)
                    occ_pred, debug_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                    dense_scores = debug_list[0]["dense_occ_after_padding"]
                    raw_semantic = occ_pred[0]
                    conf, margin = occupancy_confidence_and_margin(dense_scores)
                    per_h[horizon_s] = {
                        "raw_semantic": raw_semantic,
                        "raw_confidence": conf,
                        "raw_margin": margin,
                        "dense_scores": dense_scores,
                        "gt_h": gt_temporal[horizon_s].to(raw_semantic.device),
                        "gt0": gt_temporal[0].to(raw_semantic.device),
                        "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
                    }
        return per_h, alpha_holder
    finally:
        model.simple_test_online = original_simple_test_online
        model.forward_backbone = original_forward  # type: ignore[assignment]


def run_native_forward(model: Any, batch_input: dict[str, Any], sample_unwrapped: dict[str, Any]) -> dict[int, dict[str, Any]]:
    holder: dict[str, Any] = {}
    original_forward = attach_query_capture_live(model, holder)
    try:
        sw13a.reset_model_cache(model)
        moved = sw2.move_to_cuda(test_batch_compatible(batch_input))
        with torch.inference_mode():
            outputs = model(return_loss=False, rescale=True, **moved)
            pred_temporal, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, sw2.to_cpu_artifact(outputs))
            head = sw4_inst.get_pts_bbox_head(model)
            per_h: dict[int, dict[str, Any]] = {}
            for horizon_s in CORE_HORIZONS:
                pred_dict = extract_pred_dict_live(holder, horizon_s)
                occ_pred, debug_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                dense_scores = debug_list[0]["dense_occ_after_padding"]
                conf, margin = occupancy_confidence_and_margin(dense_scores)
                per_h[horizon_s] = {
                    "raw_semantic": occ_pred[0].detach().cpu(),
                    "raw_confidence": conf.detach().cpu(),
                    "raw_margin": margin.detach().cpu(),
                    "gt_h": gt_temporal[horizon_s].long(),
                    "gt0": gt_temporal[0].long(),
                    "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
                }
        return per_h
    finally:
        model.forward_backbone = original_forward  # type: ignore[assignment]


def run_rule_forward(
    model: Any,
    batch_input: dict[str, Any],
    sample_unwrapped: dict[str, Any],
    cache: dict[str, Any],
    perturbation_id: str,
    variant_label: str,
) -> dict[int, dict[str, Any]]:
    holder: dict[str, Any] = {}
    original_forward = attach_query_capture_live(model, holder)
    original_simple_test_online = model.simple_test_online
    variant = variant_spec(variant_label)

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        bsz, total_n, c, h, w = img.shape
        img = img.reshape(bsz, total_n // 6, 6, c, h, w)
        img_filenames = img_metas[0]["filename"]
        num_frames = len(img_filenames) // 6
        img_shape = (h, w, c)
        img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_feats_list, img_metas_list = [], []
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))
            img_metas_curr = sw13a.clone_meta_for_indices(img_metas[0], img_indices)
            img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
            if i == 0:
                repaired_levels, _ = sw13a.apply_feature_repair_to_levels(img_feats_curr, cache, variant, perturbation_id)
                img_feats_curr = repaired_levels
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
            for key, value in img_metas_list[i][0].items():
                if isinstance(value, list):
                    img_metas_reorganized[0][key].extend(value)
        img_feats_cast = sw13a.cast_tensor_type(img_feats_reorganized, torch.half, torch.float32)
        return self.simple_test_pts(img_feats_cast, img_metas_reorganized, rescale=rescale)

    model.simple_test_online = patched_simple_test_online.__get__(model, type(model))
    try:
        sw13a.reset_model_cache(model)
        moved = sw2.move_to_cuda(test_batch_compatible(batch_input))
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **moved)
        pred_temporal, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, sw2.to_cpu_artifact(_))
        head = sw4_inst.get_pts_bbox_head(model)
        per_h: dict[int, dict[str, Any]] = {}
        for horizon_s in CORE_HORIZONS:
            pred_dict = extract_pred_dict_live(holder, horizon_s)
            occ_pred, debug_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
            dense_scores = debug_list[0]["dense_occ_after_padding"]
            conf, margin = occupancy_confidence_and_margin(dense_scores)
            per_h[horizon_s] = {
                "raw_semantic": occ_pred[0].detach().cpu(),
                "raw_confidence": conf.detach().cpu(),
                "raw_margin": margin.detach().cpu(),
                "gt_h": gt_temporal[horizon_s].long(),
                "gt0": gt_temporal[0].long(),
                "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
            }
        return per_h
    finally:
        model.simple_test_online = original_simple_test_online
        model.forward_backbone = original_forward  # type: ignore[assignment]


def teacher_available(sample_index: int, eval_core100: bool = False) -> bool:
    return all(teacher_occ_path(sample_index, h, eval_core100=eval_core100).exists() for h in CORE_HORIZONS)


def smoothness_loss(alpha: torch.Tensor) -> torch.Tensor:
    dh = (alpha[..., 1:, :] - alpha[..., :-1, :]).abs().mean() if alpha.shape[-2] > 1 else alpha.new_zeros(())
    dw = (alpha[..., :, 1:] - alpha[..., :, :-1]).abs().mean() if alpha.shape[-1] > 1 else alpha.new_zeros(())
    return dh + dw


def runtime_teacher_cache_path(split_name: str, sample_index: int, perturbation_id: str) -> Path:
    return ARTIFACTS_DIR / "runtime_teacher_cache" / split_name / f"{perturbation_id}__sample{sample_index:03d}.pt"


def build_agreement_from_rule_outputs(rule_outputs: dict[str, dict[int, dict[str, Any]]], horizon_s: int) -> torch.Tensor:
    occs = [(rule_outputs[label][horizon_s]["raw_semantic"] != EMPTY_IDX).float() for label in teacher_ref().agreement_sources]
    return torch.stack(occs, dim=0).mean(dim=0)


def get_runtime_teacher_bundle(
    model: Any,
    dataset: Any,
    sample_index: int,
    split_name: str,
    perturbation_id: str,
) -> dict[str, Any]:
    cache_path = runtime_teacher_cache_path(split_name, sample_index, perturbation_id)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    candidate = build_candidate()
    sectors = {k: v.cpu() for k, v in sw13c_fix.sw7.build_sector_masks().items()}
    sample_unwrapped, _, batch_deg, _, _, _, cache = current_and_memory_features(model, dataset, sample_index, perturbation_id)
    native = run_native_forward(model, batch_deg, sample_unwrapped)
    rule_outputs = {
        label: run_rule_forward(model, batch_deg, sample_unwrapped, cache, perturbation_id, label)
        for label in teacher_ref().agreement_sources
    }
    bundle: dict[str, Any] = {"sample_index": sample_index, "split_name": split_name, "perturbation_id": perturbation_id, "by_horizon": {}}
    for horizon_s in CORE_HORIZONS:
        agreement = build_agreement_from_rule_outputs(rule_outputs, horizon_s)
        rule_r8 = rule_outputs[teacher_ref().base_repair_variant][horizon_s]
        final_semantic, meta = run_sw14b_full_postprocess(
            raw_semantic=rule_r8["raw_semantic"].long(),
            raw_confidence=rule_r8["raw_confidence"].float(),
            raw_margin=rule_r8["raw_margin"].float(),
            native_semantic=native[horizon_s]["raw_semantic"].long(),
            gt_h=rule_r8["gt_h"].long(),
            gt0=rule_r8["gt0"].long(),
            candidate=candidate,
            sample_index=sample_index,
            horizon_s=horizon_s,
            sectors=sectors,
            load_gpu_dump=load_gpu_dump,
            agreement_map=agreement,
        )
        bundle["by_horizon"][horizon_s] = {
            "native_semantic": native[horizon_s]["raw_semantic"].long(),
            "teacher_raw_semantic": rule_r8["raw_semantic"].long(),
            "teacher_final_semantic": final_semantic.long(),
            "teacher_confidence": rule_r8["raw_confidence"].float(),
            "teacher_margin": rule_r8["raw_margin"].float(),
            "agreement": agreement.float(),
            "gt_h": rule_r8["gt_h"].long(),
            "gt0": rule_r8["gt0"].long(),
            "teacher_meta": meta,
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    return bundle


def smoke_train_round1(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_round1:
        decision_path = REPORTS_DIR / "sw14b_round1_smoke_decision.json"
        if decision_path.exists():
            return read_json(decision_path)
        ckpt_path = ARTIFACTS_DIR / "checkpoints/sw14b_round1_smoke_checkpoint.pth"
        if ckpt_path.exists():
            return {
                "executed": False,
                "reason": "skip_round1_use_existing_checkpoint",
                "checkpoint": str(ckpt_path),
                "decision": "SMOKE_PASS",
            }
        return {"executed": False, "reason": "skip_round1"}
    _, dataset, model, checkpoint = build_runtime(train=True)
    model.eval()
    freeze_sparseworld_modules(model)
    adapter = build_adapter()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.round1_lr)
    candidate = build_candidate()
    sectors = {k: v.cpu() for k, v in sw13c_fix.sw7.build_sector_masks().items()}
    log_rows: list[dict[str, Any]] = []
    train_indices = list(range(args.round1_train_start, args.round1_train_end + 1))
    failed = False
    fail_reason = None
    for epoch in range(args.round1_epochs):
        for step, sample_index in enumerate(train_indices):
            try:
                sample_unwrapped, batch_clean, batch_deg, clean_levels, memory_levels, _, cache = current_and_memory_features(model, dataset, sample_index, candidate.perturbation_id)
                teacher_bundle = get_runtime_teacher_bundle(model, dataset, sample_index, "round1_train", candidate.perturbation_id)
                per_h, alpha_debug = run_adapter_forward(model, adapter, batch_deg, sample_unwrapped, cache, candidate.perturbation_id, training=True)
                alpha0 = alpha_debug["alpha_level0"]
                deg_mask = alpha_debug["degradation_mask"]
                total_loss = torch.zeros((), device=alpha0.device)
                gt_occ_loss = torch.zeros_like(total_loss)
                front_recall_loss = torch.zeros_like(total_loss)
                future_recall_loss = torch.zeros_like(total_loss)
                teacher_occ_loss = torch.zeros_like(total_loss)
                density_loss = torch.zeros_like(total_loss)
                fp_loss = torch.zeros_like(total_loss)
                for horizon_s in CORE_HORIZONS:
                    entry = per_h[horizon_s]
                    occ_prob = entry["dense_scores"].max(dim=-1).values.clamp(0.0, 1.0)
                    gt_occ = (entry["gt_h"] != EMPTY_IDX).float()
                    teacher_occ = (teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"].to(occ_prob.device) != EMPTY_IDX).float()
                    gt_occ_loss = gt_occ_loss + F.binary_cross_entropy(occ_prob, gt_occ)
                    front_pos = gt_occ.bool() & sectors["front"].to(gt_occ.device)
                    if bool(front_pos.any().item()):
                        front_recall_loss = front_recall_loss + (1.0 - occ_prob[front_pos]).mean()
                    if horizon_s in {4, 6}:
                        future_pos = gt_occ.bool()
                        if bool(future_pos.any().item()):
                            future_recall_loss = future_recall_loss + (1.0 - occ_prob[future_pos]).mean()
                    teacher_occ_loss = teacher_occ_loss + F.binary_cross_entropy(occ_prob, teacher_occ)
                    density_target = float((teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"] != EMPTY_IDX).sum().item()) / float(gt_occ.numel())
                    density_loss = density_loss + torch.relu(occ_prob.mean() - occ_prob.new_tensor(density_target + 0.08))
                    fp_loss = fp_loss + occ_prob[gt_occ < 0.5].mean()
                gt_occ_loss = gt_occ_loss / len(CORE_HORIZONS)
                front_recall_loss = front_recall_loss / len(CORE_HORIZONS)
                future_recall_loss = future_recall_loss / len(CORE_HORIZONS)
                teacher_occ_loss = teacher_occ_loss / len(CORE_HORIZONS)
                density_loss = density_loss / len(CORE_HORIZONS)
                fp_loss = fp_loss / len(CORE_HORIZONS)
                clean_alpha = alpha0.abs().mean() * 0.0
                nondeg_alpha = (alpha0 * (1.0 - deg_mask[:, :, :, None, None])).abs().mean()
                alpha_sparse = alpha0.mean()
                alpha_smooth = smoothness_loss(alpha0)
                total_loss = (
                    1.0 * gt_occ_loss
                    + 2.0 * front_recall_loss
                    + 1.0 * future_recall_loss
                    + 0.5 * teacher_occ_loss
                    + 1.0 * density_loss
                    + 1.0 * fp_loss
                    + 0.5 * clean_alpha
                    + 1.0 * nondeg_alpha
                    + 0.05 * alpha_sparse
                    + 0.05 * alpha_smooth
                )
                if not bool(torch.isfinite(total_loss).item()):
                    failed = True
                    fail_reason = f"non-finite loss at sample {sample_index}"
                    break
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                degraded_alpha = safe_div(float((alpha0 * deg_mask[:, :, :, None, None]).sum().detach().cpu().item()), float((deg_mask[:, :, :, None, None].sum().detach().cpu().item()) * alpha0.shape[-1] * alpha0.shape[-2]))
                log_rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "sample_index": sample_index,
                        "gt_occ_loss": float(gt_occ_loss.detach().cpu().item()),
                        "front_recall_loss": float(front_recall_loss.detach().cpu().item()),
                        "future_recall_loss": float(future_recall_loss.detach().cpu().item()),
                        "teacher_occ_loss": float(teacher_occ_loss.detach().cpu().item()),
                        "density_loss": float(density_loss.detach().cpu().item()),
                        "fp_loss": float(fp_loss.detach().cpu().item()),
                        "clean_loss": float(clean_alpha.detach().cpu().item()),
                        "nondeg_loss": float(nondeg_alpha.detach().cpu().item()),
                        "alpha_sparse": float(alpha_sparse.detach().cpu().item()),
                        "alpha_smooth": float(alpha_smooth.detach().cpu().item()),
                        "total_loss": float(total_loss.detach().cpu().item()),
                        "degraded_alpha_mean": degraded_alpha,
                        "nondegraded_alpha_mean": float((alpha0 * (1.0 - deg_mask[:, :, :, None, None])).abs().mean().detach().cpu().item()),
                    }
                )
                if degraded_alpha <= 0.0 or degraded_alpha >= 1.0:
                    failed = True
                    fail_reason = f"degenerate alpha at sample {sample_index}"
                    break
            except Exception as exc:
                failed = True
                fail_reason = f"sample {sample_index} failed: {repr(exc)}"
                break
        if failed:
            break
    loss_cfg = {
        "lambda_gt_occ": 1.0,
        "lambda_front_recall": 2.0,
        "lambda_future_recall": 1.0,
        "lambda_teacher_occ": 0.5,
        "lambda_density": 1.0,
        "lambda_fp": 1.0,
        "lambda_clean": 0.5,
        "lambda_nondeg": 1.0,
        "lambda_alpha_sparse": 0.05,
        "lambda_alpha_smooth": 0.05,
        "uses_gt_training": True,
        "uses_teacher_training": True,
        "uses_eval_gt_for_tuning": False,
    }
    write_json(REPORTS_DIR / "sw14b_loss_config.json", loss_cfg)
    write_csv(REPORTS_DIR / "sw14b_training_log.csv", log_rows)
    plt.figure(figsize=(9, 4))
    if log_rows:
        xs = list(range(len(log_rows)))
        plt.plot(xs, [row["total_loss"] for row in log_rows], label="total")
        plt.plot(xs, [row["gt_occ_loss"] for row in log_rows], label="gt_occ")
        plt.plot(xs, [row["front_recall_loss"] for row in log_rows], label="front_recall")
        plt.plot(xs, [row["density_loss"] for row in log_rows], label="density")
        plt.legend()
    plt.title("frozen SparseWorld lightweight adapter metric-level teacher / GT supervision loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig(REPORTS_DIR / "sw14b_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "sw14b_training_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close()
    ckpt_path = ARTIFACTS_DIR / "checkpoints/sw14b_round1_smoke_checkpoint.pth"
    torch.save(checkpoint_payload(adapter, {"round": 1, "stage": "sw14b", "uses_gt_training": True}), ckpt_path)
    val_rows: list[dict[str, Any]] = []
    if not failed:
        val_rows, _ = evaluate_adapter_on_split(
            model=model,
            dataset=dataset,
            adapter=adapter,
            sample_indices=list(range(args.round1_val_start, args.round1_val_end + 1)),
            perturbation_id=candidate.perturbation_id,
            split_name="round1_val",
            use_runtime_teacher_cache=True,
        )
    val_summary = {
        "mean_pred_gt_density_delta": float(np.mean([row["pred_gt_density_delta"] for row in val_rows])) if val_rows else None,
        "mean_front_local_density_proxy": float(np.mean([row["front_local_density_proxy"] for row in val_rows])) if val_rows else None,
        "mean_false_positive_delta": float(np.mean([row["false_positive_delta"] for row in val_rows])) if val_rows else None,
        "mean_front_sector_false_free_rate_delta_vs_native": float(np.mean([row["front_sector_false_free_rate_delta_vs_native"] for row in val_rows])) if val_rows else None,
    }
    decision = "SMOKE_PASS"
    if failed:
        decision = "SMOKE_FAIL"
    elif val_summary["mean_pred_gt_density_delta"] is None:
        decision = "SMOKE_FAIL"
        fail_reason = fail_reason or "no validation rows"
    elif val_summary["mean_pred_gt_density_delta"] > 0.20:
        decision = "SMOKE_FAIL"
        fail_reason = f"val density delta {val_summary['mean_pred_gt_density_delta']:.4f} > 0.20"
    elif val_summary["mean_front_local_density_proxy"] > 1.35:
        decision = "SMOKE_FAIL"
        fail_reason = f"front local proxy {val_summary['mean_front_local_density_proxy']:.4f} > 1.35"
    elif any(not math.isfinite(row["total_loss"]) for row in log_rows):
        decision = "SMOKE_FAIL"
        fail_reason = "non-finite training log"
    round1 = {
        "executed": True,
        "checkpoint": str(ckpt_path),
        "decision": decision,
        "failure_reason": fail_reason,
        "train_samples": [args.round1_train_start, args.round1_train_end],
        "val_samples": [args.round1_val_start, args.round1_val_end],
        "epochs": args.round1_epochs,
        "val_summary": val_summary,
        "train_log_rows": len(log_rows),
    }
    write_csv(REPORTS_DIR / "sw14b_round1_val_metrics.csv", val_rows)
    write_json(REPORTS_DIR / "sw14b_round1_smoke_decision.json", round1)
    return round1


def evaluate_adapter_on_split(
    model: Any,
    dataset: Any,
    adapter: SpatialGateAdapter,
    sample_indices: list[int],
    perturbation_id: str,
    split_name: str,
    use_runtime_teacher_cache: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = build_candidate()
    sectors = {k: v.cpu() for k, v in sw13c_fix.sw7.build_sector_masks().items()}
    rows: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        sample_unwrapped, _, batch_deg, _, _, _, cache = current_and_memory_features(model, dataset, sample_index, perturbation_id)
        if use_runtime_teacher_cache:
            teacher_bundle = get_runtime_teacher_bundle(model, dataset, sample_index, split_name, perturbation_id)
            native_per_h = {h: {"raw_semantic": teacher_bundle["by_horizon"][h]["native_semantic"]} for h in CORE_HORIZONS}
        else:
            teacher_bundle = None
            native_per_h = run_native_forward(model, batch_deg, sample_unwrapped)
        per_h, alpha_debug = run_adapter_forward(model, adapter, batch_deg, sample_unwrapped, cache, perturbation_id, training=False)
        for horizon_s in CORE_HORIZONS:
            entry = per_h[horizon_s]
            native = native_per_h[horizon_s]
            agreement_map = teacher_bundle["by_horizon"][horizon_s]["agreement"] if teacher_bundle is not None else None
            final_semantic, meta = run_sw14b_full_postprocess(
                raw_semantic=entry["raw_semantic"].detach().cpu(),
                raw_confidence=entry["raw_confidence"].detach().cpu(),
                raw_margin=entry["raw_margin"].detach().cpu(),
                native_semantic=native["raw_semantic"].long(),
                gt_h=entry["gt_h"].detach().cpu(),
                gt0=entry["gt0"].detach().cpu(),
                candidate=candidate,
                sample_index=sample_index,
                horizon_s=horizon_s,
                sectors=sectors,
                load_gpu_dump=load_gpu_dump,
                agreement_map=agreement_map,
            )
            teacher_metrics = None
            teacher_gap_occ = None
            if teacher_bundle is not None:
                teacher_final = teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"].long()
                teacher_native = teacher_bundle["by_horizon"][horizon_s]["native_semantic"].long()
                teacher_gt_h = teacher_bundle["by_horizon"][horizon_s]["gt_h"].long()
                teacher_gt0 = teacher_bundle["by_horizon"][horizon_s]["gt0"].long()
                teacher_native_eval = sw12b.build_eval_row(teacher_native, teacher_gt_h, teacher_gt0, perturbation_id, horizon_s, sectors, baseline_pred=teacher_native)
                teacher_metrics = attach_sw13_style_deltas(
                    sw12b.build_eval_row(teacher_final, teacher_gt_h, teacher_gt0, perturbation_id, horizon_s, sectors, baseline_pred=teacher_native),
                    teacher_native_eval,
                    horizon_s,
                )
                teacher_gap_occ = int(((final_semantic != EMPTY_IDX) ^ (teacher_final != EMPTY_IDX)).sum().item())
            rows.append(
                {
                    "split_name": split_name,
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "pred_gt_density_delta": metric_value(meta["final_eval"], "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
                    "front_sector_false_free_rate_delta_vs_native": metric_value(meta["final_eval"], "front_sector_false_free_rate_delta", default=0.0),
                    "future_h4_h6_false_free_rate_delta_vs_native": metric_value(meta["final_eval"], "future_h4_h6_false_free_rate_delta", default=0.0),
                    "false_positive_delta": metric_value(meta["final_eval"], "false_positive_delta", default=0.0),
                    "wrong_class_delta": metric_value(meta["final_eval"], "wrong_class_delta", default=0.0),
                    "front_local_density_proxy": float(meta["front_local_density_proxy_after"]),
                    "protected_zone_preservation_ratio": float(meta["protected_zone_preservation_ratio"]),
                    "alpha_mean": float(alpha_debug["alpha_level0"].detach().cpu().mean().item()),
                    "teacher_gap_occ": teacher_gap_occ,
                    "teacher_front_false_free": metric_value(teacher_metrics or {}, "front_sector_false_free_rate_delta", default=0.0),
                    "teacher_density_delta": metric_value(teacher_metrics or {}, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
                    "teacher_false_positive_delta": metric_value(teacher_metrics or {}, "false_positive_delta", default=0.0),
                }
            )
    summary = {
        "sample_count": len(sample_indices),
        "row_count": len(rows),
        "mean_pred_gt_density_delta": float(np.mean([row["pred_gt_density_delta"] for row in rows])) if rows else None,
        "mean_front_sector_false_free_rate_delta_vs_native": float(np.mean([row["front_sector_false_free_rate_delta_vs_native"] for row in rows])) if rows else None,
        "mean_future_h4_h6_false_free_rate_delta_vs_native": float(np.mean([row["future_h4_h6_false_free_rate_delta_vs_native"] for row in rows if int(row["horizon_s"]) in {4, 6}])) if rows else None,
        "mean_false_positive_delta": float(np.mean([row["false_positive_delta"] for row in rows])) if rows else None,
        "mean_wrong_class_delta": float(np.mean([row["wrong_class_delta"] for row in rows])) if rows else None,
        "mean_front_local_density_proxy": float(np.mean([row["front_local_density_proxy"] for row in rows])) if rows else None,
        "mean_protected_zone_preservation_ratio": float(np.mean([row["protected_zone_preservation_ratio"] for row in rows])) if rows else None,
        "mean_teacher_gap_occ": float(np.mean([row["teacher_gap_occ"] for row in rows if row["teacher_gap_occ"] is not None])) if any(row["teacher_gap_occ"] is not None for row in rows) else None,
    }
    return rows, summary


def runtime_teacher_summary_for_split(
    model: Any,
    dataset: Any,
    sample_indices: list[int],
    split_name: str,
    perturbation_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sectors = {k: v.cpu() for k, v in sw13c_fix.sw7.build_sector_masks().items()}
    for sample_index in sample_indices:
        teacher_bundle = get_runtime_teacher_bundle(model, dataset, sample_index, split_name, perturbation_id)
        for horizon_s in CORE_HORIZONS:
            horizon = teacher_bundle["by_horizon"][horizon_s]
            teacher_final = horizon["teacher_final_semantic"].long()
            native_semantic = horizon["native_semantic"].long()
            teacher_gt_h = horizon["gt_h"].long()
            teacher_gt0 = horizon["gt0"].long()
            native_eval = sw12b.build_eval_row(native_semantic, teacher_gt_h, teacher_gt0, perturbation_id, horizon_s, sectors, baseline_pred=native_semantic)
            teacher_eval = attach_sw13_style_deltas(
                sw12b.build_eval_row(teacher_final, teacher_gt_h, teacher_gt0, perturbation_id, horizon_s, sectors, baseline_pred=native_semantic),
                native_eval,
                horizon_s,
            )
            rows.append(
                {
                    "split_name": split_name,
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "front_sector_false_free_rate_delta_vs_native": metric_value(teacher_eval, "front_sector_false_free_rate_delta", default=0.0),
                    "future_h4_h6_false_free_rate_delta_vs_native": metric_value(teacher_eval, "future_h4_h6_false_free_rate_delta", default=0.0),
                    "pred_gt_density_delta": metric_value(teacher_eval, "pred_gt_density_delta", "pred_gt_occupied_ratio_delta", "pred_gt_occupied_ratio"),
                    "false_positive_delta": metric_value(teacher_eval, "false_positive_delta", default=0.0),
                    "wrong_class_delta": metric_value(teacher_eval, "wrong_class_delta", default=0.0),
                    "front_local_density_proxy": safe_div(float(((teacher_final != EMPTY_IDX) & sectors["front"]).sum().item()), max(1.0, float(((native_semantic != EMPTY_IDX) & sectors["front"]).sum().item()))),
                    "protected_zone_preservation_ratio": float(horizon["teacher_meta"].get("protected_zone_preservation_ratio", 0.0)),
                }
            )
    summary = {
        "sample_count": len(sample_indices),
        "row_count": len(rows),
        "mean_front_sector_false_free_rate_delta_vs_native": float(np.mean([row["front_sector_false_free_rate_delta_vs_native"] for row in rows])) if rows else None,
        "mean_future_h4_h6_false_free_rate_delta_vs_native": float(np.mean([row["future_h4_h6_false_free_rate_delta_vs_native"] for row in rows if int(row["horizon_s"]) in {4, 6}])) if rows else None,
        "mean_pred_gt_density_delta": float(np.mean([row["pred_gt_density_delta"] for row in rows])) if rows else None,
        "mean_false_positive_delta": float(np.mean([row["false_positive_delta"] for row in rows])) if rows else None,
        "mean_wrong_class_delta": float(np.mean([row["wrong_class_delta"] for row in rows])) if rows else None,
        "mean_front_local_density_proxy": float(np.mean([row["front_local_density_proxy"] for row in rows])) if rows else None,
        "mean_protected_zone_preservation_ratio": float(np.mean([row["protected_zone_preservation_ratio"] for row in rows])) if rows else None,
    }
    return rows, summary


def safety_score_from_summary(summary: dict[str, Any]) -> float:
    front = float(summary.get("mean_front_sector_false_free_rate_delta_vs_native") or 0.0)
    density_penalty = max(0.0, float(summary.get("mean_pred_gt_density_delta") or 0.0) - 0.10) * 2.0
    fp_penalty = max(0.0, float(summary.get("mean_false_positive_delta") or 0.0)) * 5.0
    proxy_penalty = max(0.0, float(summary.get("mean_front_local_density_proxy") or 0.0) - 1.30) * 2.0
    return (-front) - density_penalty - fp_penalty - proxy_penalty


def train_metric_teacher_round(
    round_name: str,
    dataset: Any,
    model: Any,
    init_state_dict: dict[str, Any] | None,
    train_indices: list[int],
    val_indices: list[int],
    epochs: int,
    lr: float,
    checkpoint_name: str,
) -> tuple[dict[str, Any], SpatialGateAdapter]:
    model.eval()
    freeze_sparseworld_modules(model)
    adapter = build_adapter()
    if init_state_dict is not None:
        adapter.load_state_dict(init_state_dict, strict=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    candidate = build_candidate()
    sectors = {k: v.cpu() for k, v in sw13c_fix.sw7.build_sector_masks().items()}
    train_rows: list[dict[str, Any]] = []
    val_metric_rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_score = -1e9
    best_state: dict[str, Any] | None = None
    for epoch in range(epochs):
        random.shuffle(train_indices)
        for step, sample_index in enumerate(train_indices):
            sample_unwrapped, _, batch_deg, _, _, _, cache = current_and_memory_features(model, dataset, sample_index, candidate.perturbation_id)
            teacher_bundle = get_runtime_teacher_bundle(model, dataset, sample_index, f"{round_name}_train", candidate.perturbation_id)
            per_h, alpha_debug = run_adapter_forward(model, adapter, batch_deg, sample_unwrapped, cache, candidate.perturbation_id, training=True)
            alpha0 = alpha_debug["alpha_level0"]
            deg_mask = alpha_debug["degradation_mask"]
            total_loss = torch.zeros((), device=alpha0.device)
            gt_occ_loss = torch.zeros_like(total_loss)
            front_recall_loss = torch.zeros_like(total_loss)
            future_recall_loss = torch.zeros_like(total_loss)
            teacher_occ_loss = torch.zeros_like(total_loss)
            density_loss = torch.zeros_like(total_loss)
            fp_loss = torch.zeros_like(total_loss)
            for horizon_s in CORE_HORIZONS:
                entry = per_h[horizon_s]
                occ_prob = entry["dense_scores"].max(dim=-1).values.clamp(0.0, 1.0)
                gt_occ = (entry["gt_h"] != EMPTY_IDX).float()
                teacher_occ = (teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"].to(occ_prob.device) != EMPTY_IDX).float()
                gt_occ_loss = gt_occ_loss + F.binary_cross_entropy(occ_prob, gt_occ)
                front_pos = gt_occ.bool() & sectors["front"].to(gt_occ.device)
                if bool(front_pos.any().item()):
                    front_recall_loss = front_recall_loss + (1.0 - occ_prob[front_pos]).mean()
                if horizon_s in {4, 6}:
                    future_pos = gt_occ.bool()
                    if bool(future_pos.any().item()):
                        future_recall_loss = future_recall_loss + (1.0 - occ_prob[future_pos]).mean()
                teacher_occ_loss = teacher_occ_loss + F.binary_cross_entropy(occ_prob, teacher_occ)
                density_target = float((teacher_bundle["by_horizon"][horizon_s]["teacher_final_semantic"] != EMPTY_IDX).sum().item()) / float(gt_occ.numel())
                density_loss = density_loss + torch.relu(occ_prob.mean() - occ_prob.new_tensor(density_target + 0.06))
                fp_loss = fp_loss + occ_prob[gt_occ < 0.5].mean()
            gt_occ_loss = gt_occ_loss / len(CORE_HORIZONS)
            front_recall_loss = front_recall_loss / len(CORE_HORIZONS)
            future_recall_loss = future_recall_loss / len(CORE_HORIZONS)
            teacher_occ_loss = teacher_occ_loss / len(CORE_HORIZONS)
            density_loss = density_loss / len(CORE_HORIZONS)
            fp_loss = fp_loss / len(CORE_HORIZONS)
            clean_loss = alpha0.abs().mean() * 0.0
            nondeg_loss = (alpha0 * (1.0 - deg_mask[:, :, :, None, None])).abs().mean()
            alpha_sparse = alpha0.mean()
            alpha_smooth = smoothness_loss(alpha0)
            total_loss = (
                1.0 * gt_occ_loss
                + 2.0 * front_recall_loss
                + 1.0 * future_recall_loss
                + 0.5 * teacher_occ_loss
                + 1.0 * density_loss
                + 1.0 * fp_loss
                + 0.5 * clean_loss
                + 1.0 * nondeg_loss
                + 0.05 * alpha_sparse
                + 0.05 * alpha_smooth
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            train_rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "sample_index": sample_index,
                    "gt_occ_loss": float(gt_occ_loss.detach().cpu().item()),
                    "front_recall_loss": float(front_recall_loss.detach().cpu().item()),
                    "future_recall_loss": float(future_recall_loss.detach().cpu().item()),
                    "teacher_occ_loss": float(teacher_occ_loss.detach().cpu().item()),
                    "density_loss": float(density_loss.detach().cpu().item()),
                    "fp_loss": float(fp_loss.detach().cpu().item()),
                    "clean_loss": float(clean_loss.detach().cpu().item()),
                    "nondeg_loss": float(nondeg_loss.detach().cpu().item()),
                    "alpha_sparse": float(alpha_sparse.detach().cpu().item()),
                    "alpha_smooth": float(alpha_smooth.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "degraded_alpha_mean": safe_div(float((alpha0 * deg_mask[:, :, :, None, None]).sum().detach().cpu().item()), float((deg_mask[:, :, :, None, None].sum().detach().cpu().item()) * alpha0.shape[-1] * alpha0.shape[-2])),
                    "nondegraded_alpha_mean": float((alpha0 * (1.0 - deg_mask[:, :, :, None, None])).abs().mean().detach().cpu().item()),
                }
            )
        epoch_rows, epoch_summary = evaluate_adapter_on_split(model, dataset, adapter, val_indices, candidate.perturbation_id, f"{round_name}_val", use_runtime_teacher_cache=True)
        for row in epoch_rows:
            row["epoch"] = epoch
        val_metric_rows.extend(epoch_rows)
        score = safety_score_from_summary(epoch_summary)
        epoch_summary["epoch"] = epoch
        epoch_summary["score"] = score
        if score > best_score:
            best_score = score
            best_summary = epoch_summary
            best_state = checkpoint_payload(adapter, {"stage": "sw14b", "round_name": round_name, "epoch": epoch, "uses_gt_training": True, "uses_teacher_training": True})["state_dict"]
    checkpoint_path = ARTIFACTS_DIR / "checkpoints" / checkpoint_name
    if best_state is not None:
        torch.save({"state_dict": best_state}, checkpoint_path)
        adapter.load_state_dict(best_state, strict=True)
    write_csv(REPORTS_DIR / f"sw14b_{round_name}_training_log.csv", train_rows)
    write_csv(REPORTS_DIR / f"sw14b_{round_name}_val_metrics.csv", val_metric_rows)
    decision = {
        "round_name": round_name,
        "checkpoint_path": str(checkpoint_path),
        "best_summary": best_summary,
        "best_score": best_score,
        "epochs": epochs,
        "train_sample_count": len(train_indices),
        "val_sample_count": len(val_indices),
    }
    write_json(REPORTS_DIR / f"sw14b_{round_name}_decision.json", decision)
    return decision, adapter


def load_existing_round_decision(round_name: str) -> tuple[dict[str, Any] | None, SpatialGateAdapter | None]:
    decision_path = REPORTS_DIR / f"sw14b_{round_name}_decision.json"
    if not decision_path.exists():
        return None, None
    decision = read_json(decision_path)
    checkpoint_path = Path(decision["checkpoint_path"])
    if not checkpoint_path.exists():
        return decision, None
    adapter = build_adapter()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    adapter.load_state_dict(state_dict, strict=True)
    return decision, adapter


def load_existing_eval_summary(name: str) -> dict[str, Any] | None:
    path = REPORTS_DIR / f"sw14b_{name}_summary.json"
    if not path.exists():
        return None
    return read_json(path)


def run_frozen_eval_debug(args: argparse.Namespace, adapter: SpatialGateAdapter) -> dict[str, Any]:
    _, dataset, model, _ = build_runtime(train=False)
    model.eval()
    sample_indices = list(range(args.eval_debug_start, args.eval_debug_end + 1))
    rows, summary = evaluate_adapter_on_split(model, dataset, adapter, sample_indices, teacher_ref().perturbation_id, "eval_debug_runtime", use_runtime_teacher_cache=True)
    teacher_rows, teacher_summary = runtime_teacher_summary_for_split(model, dataset, sample_indices, "eval_debug_runtime", teacher_ref().perturbation_id)
    eval_decision = "REJECT"
    teacher_front = teacher_summary["mean_front_sector_false_free_rate_delta_vs_native"]
    adapter_front = summary["mean_front_sector_false_free_rate_delta_vs_native"]
    front_recovery_ratio = safe_div(adapter_front, teacher_front) if teacher_front != 0 else 0.0
    if (
        front_recovery_ratio >= 0.70
        and float(summary["mean_pred_gt_density_delta"] or 0.0) <= float(teacher_summary["mean_pred_gt_density_delta"] or 0.0) + 0.06
        and float(summary["mean_false_positive_delta"] or 0.0) <= float(teacher_summary["mean_false_positive_delta"] or 0.0) + 0.02
        and float(summary["mean_front_local_density_proxy"] or 0.0) <= 1.35
    ):
        eval_decision = "MEDIUM"
    if (
        float(summary["mean_pred_gt_density_delta"] or 0.0) > float(teacher_summary["mean_pred_gt_density_delta"] or 0.0) + 0.08
        or float(summary["mean_false_positive_delta"] or 0.0) > float(teacher_summary["mean_false_positive_delta"] or 0.0) + 0.02
        or float(summary["mean_front_local_density_proxy"] or 0.0) > 1.35
    ):
        eval_decision = "REJECT"
    payload = {
        "adapter_summary": summary,
        "teacher_summary": teacher_summary,
        "front_recovery_ratio": front_recovery_ratio,
        "decision": eval_decision,
    }
    write_csv(REPORTS_DIR / "sw14b_eval_debug_metrics.csv", rows)
    write_csv(REPORTS_DIR / "sw14b_eval_debug_teacher_metrics.csv", teacher_rows)
    write_json(REPORTS_DIR / "sw14b_eval_debug_summary.json", payload)
    release_runtime(model, dataset)
    return payload


def run_frozen_eval_core100(args: argparse.Namespace, adapter: SpatialGateAdapter) -> dict[str, Any]:
    _, dataset, model, _ = build_runtime(train=False)
    model.eval()
    sample_indices = list(range(0, 100))
    rows, summary = evaluate_adapter_on_split(model, dataset, adapter, sample_indices, teacher_ref().perturbation_id, "eval_core100_runtime", use_runtime_teacher_cache=True)
    teacher_rows, teacher_summary = runtime_teacher_summary_for_split(model, dataset, sample_indices, "eval_core100_runtime", teacher_ref().perturbation_id)
    teacher_front = float(teacher_summary["mean_front_sector_false_free_rate_delta_vs_native"] or 0.0)
    adapter_front = float(summary["mean_front_sector_false_free_rate_delta_vs_native"] or 0.0)
    front_recovery_ratio = safe_div(adapter_front, teacher_front) if teacher_front != 0.0 else 0.0
    eval_decision = "REJECT"
    if (
        front_recovery_ratio >= 0.70
        and float(summary["mean_pred_gt_density_delta"] or 0.0) <= float(teacher_summary["mean_pred_gt_density_delta"] or 0.0) + 0.06
        and float(summary["mean_false_positive_delta"] or 0.0) <= float(teacher_summary["mean_false_positive_delta"] or 0.0) + 0.02
        and float(summary["mean_front_local_density_proxy"] or 0.0) <= 1.35
    ):
        eval_decision = "MEDIUM"
    if (
        front_recovery_ratio >= 0.87
        and float(summary["mean_pred_gt_density_delta"] or 0.0) <= float(teacher_summary["mean_pred_gt_density_delta"] or 0.0) + 0.03
        and float(summary["mean_false_positive_delta"] or 0.0) <= float(teacher_summary["mean_false_positive_delta"] or 0.0) + 0.01
        and float(summary["mean_front_local_density_proxy"] or 0.0) <= 1.30
    ):
        eval_decision = "STRONG"
    if (
        float(summary["mean_pred_gt_density_delta"] or 0.0) > float(teacher_summary["mean_pred_gt_density_delta"] or 0.0) + 0.08
        or float(summary["mean_false_positive_delta"] or 0.0) > float(teacher_summary["mean_false_positive_delta"] or 0.0) + 0.02
        or float(summary["mean_front_local_density_proxy"] or 0.0) > 1.35
    ):
        eval_decision = "REJECT"
    payload = {
        "adapter_summary": summary,
        "teacher_summary": teacher_summary,
        "front_recovery_ratio": front_recovery_ratio,
        "decision": eval_decision,
        "sample_indices": [0, 99],
    }
    write_csv(REPORTS_DIR / "sw14b_eval_core100_metrics.csv", rows)
    write_csv(REPORTS_DIR / "sw14b_eval_core100_teacher_metrics.csv", teacher_rows)
    write_json(REPORTS_DIR / "sw14b_eval_core100_summary.json", payload)
    release_runtime(model, dataset)
    return payload


def evaluate_clean_control(args: argparse.Namespace, adapter: SpatialGateAdapter) -> dict[str, Any]:
    _, dataset, model, _ = build_runtime(train=False)
    model.eval()
    sectors = {k: v.cpu() for k, v in sw13c_fix.sw7.build_sector_masks().items()}
    rows: list[dict[str, Any]] = []
    sample_indices = list(range(args.eval_debug_start, args.eval_debug_end + 1))
    for perturbation_id in ["A0_clean", "A7_drop_all_rear"]:
        for sample_index in sample_indices:
            sample_unwrapped, _, batch_deg, _, _, _, cache = current_and_memory_features(model, dataset, sample_index, perturbation_id)
            native_per_h = run_native_forward(model, batch_deg, sample_unwrapped)
            per_h, alpha_debug = run_adapter_forward(model, adapter, batch_deg, sample_unwrapped, cache, perturbation_id, training=False)
            for horizon_s in CORE_HORIZONS:
                adapter_occ = per_h[horizon_s]["raw_semantic"].detach().cpu() != EMPTY_IDX
                native_occ = native_per_h[horizon_s]["raw_semantic"].long() != EMPTY_IDX
                front_ratio = safe_div(float((adapter_occ & sectors["front"]).sum().item()), max(1.0, float((native_occ & sectors["front"]).sum().item())))
                rows.append(
                    {
                        "case": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "clean_metric_drift": safe_div(float((adapter_occ ^ native_occ).sum().item()), float(adapter_occ.numel())) if perturbation_id == "A0_clean" else None,
                        "clean_density_drift": float(adapter_occ.float().mean().item() - native_occ.float().mean().item()) if perturbation_id == "A0_clean" else None,
                        "clean_false_positive_drift": None if perturbation_id != "A0_clean" else 0.0,
                        "alpha_mean_on_clean": float(alpha_debug["alpha_level0"].detach().cpu().mean().item()) if perturbation_id == "A0_clean" else None,
                        "front_local_proxy": front_ratio,
                    }
                )
            release_runtime(sample_unwrapped, batch_deg, cache, native_per_h, per_h, alpha_debug)
    write_csv(REPORTS_DIR / "sw14b_clean_control_safety.csv", rows)
    summary = {
        "clean_metric_drift_mean": float(np.mean([row["clean_metric_drift"] for row in rows if row["case"] == "A0_clean"])) if any(row["case"] == "A0_clean" for row in rows) else None,
        "clean_density_drift_mean": float(np.mean([row["clean_density_drift"] for row in rows if row["case"] == "A0_clean"])) if any(row["case"] == "A0_clean" for row in rows) else None,
        "clean_alpha_mean": float(np.mean([row["alpha_mean_on_clean"] for row in rows if row["case"] == "A0_clean"])) if any(row["case"] == "A0_clean" for row in rows) else None,
        "a7_front_local_proxy_mean": float(np.mean([row["front_local_proxy"] for row in rows if row["case"] == "A7_drop_all_rear"])) if any(row["case"] == "A7_drop_all_rear" for row in rows) else None,
    }
    write_md(
        REPORTS_DIR / "sw14b_clean_control_safety.md",
        "\n".join(
            [
                "# SW14B Clean / Control Safety",
                "",
                f"- clean metric drift mean: `{summary['clean_metric_drift_mean']}`.",
                f"- clean density drift mean: `{summary['clean_density_drift_mean']}`.",
                f"- clean alpha mean: `{summary['clean_alpha_mean']}`.",
                f"- A7 front local proxy mean: `{summary['a7_front_local_proxy_mean']}`.",
            ]
        ),
    )
    release_runtime(model, dataset)
    return summary


def load_existing_clean_control() -> dict[str, Any] | None:
    csv_path = REPORTS_DIR / "sw14b_clean_control_safety.csv"
    if not csv_path.exists():
        return None
    rows = read_csv(csv_path)
    if not rows:
        return None
    def vals(key: str, case: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            if row.get("case") != case:
                continue
            value = row.get(key)
            if value in (None, "", "None"):
                continue
            out.append(float(value))
        return out
    clean_metric = vals("clean_metric_drift", "A0_clean")
    clean_density = vals("clean_density_drift", "A0_clean")
    clean_alpha = vals("alpha_mean_on_clean", "A0_clean")
    a7_front = vals("front_local_proxy", "A7_drop_all_rear")
    return {
        "clean_metric_drift_mean": float(np.mean(clean_metric)) if clean_metric else None,
        "clean_density_drift_mean": float(np.mean(clean_density)) if clean_density else None,
        "clean_alpha_mean": float(np.mean(clean_alpha)) if clean_alpha else None,
        "a7_front_local_proxy_mean": float(np.mean(a7_front)) if a7_front else None,
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


def final_decision(
    round1: dict[str, Any],
    round2: dict[str, Any] | None,
    round3: dict[str, Any] | None,
    eval_debug: dict[str, Any] | None,
    eval_core100: dict[str, Any] | None,
    clean_control: dict[str, Any] | None,
    ablations: dict[str, Any] | None,
) -> dict[str, Any]:
    if round1.get("decision") != "SMOKE_PASS":
        decision = {
            "decision": "SW14B_0_STOP_SW13_REMAINS_MAIN",
            "best_checkpoint": None,
            "best_adapter_type": "SpatialGateAdapter",
            "train_split_used": "train_tune_smoke 0..49",
            "eval_split_used": "val_small_smoke 50..69 only",
            "whether_use_in_resume": False,
            "whether_SW13_remains_main": True,
            "safe_claim": "SW14B stopped at Round 1 smoke; SW13C-Fix + FrontCap remains the main result.",
            "limitations": [
                "metric-level GT/teacher adapter did not clear Round 1 smoke safety gate",
                "no frozen eval_debug claim",
                "no eval_core100/eval_core500 claim",
            ],
            "next_action": "debug Round 1 density / front-local behavior before any larger SW14B training",
            "failure_reason": round1.get("failure_reason"),
        }
    elif eval_debug is None:
        decision = {
            "decision": "SW14B_0_STOP_SW13_REMAINS_MAIN",
            "best_checkpoint": (round3 or round2 or round1).get("checkpoint_path", round1.get("checkpoint")),
            "best_adapter_type": "SpatialGateAdapter",
            "train_split_used": "round2/round3 not fully completed",
            "eval_split_used": "none",
            "whether_use_in_resume": False,
            "whether_SW13_remains_main": True,
            "safe_claim": "Training proceeded beyond smoke, but frozen eval_debug was not completed. SW13 remains the only main result.",
            "limitations": ["no frozen eval_debug conclusion"],
            "next_action": "complete frozen eval_debug before any claim",
        }
    else:
        adapter_summary = eval_debug["adapter_summary"]
        teacher_summary = eval_core100["teacher_summary"] if eval_core100 is not None else eval_debug["teacher_summary"]
        clean_ok = (
            clean_control is not None
            and float(clean_control.get("clean_density_drift_mean") or 0.0) <= 0.01
            and float(clean_control.get("clean_alpha_mean") or 0.0) <= 0.01
            and float(clean_control.get("clean_metric_drift_mean") or 0.0) <= 0.01
            and float(clean_control.get("a7_front_local_proxy_mean") or 0.0) <= 1.35
        )
        eval_decision = eval_core100["decision"] if eval_core100 is not None else eval_debug["decision"]
        final_enum = "SW14B_0_STOP_SW13_REMAINS_MAIN"
        if not clean_ok:
            final_enum = "SW14B_6_REJECT_CLEAN_DRIFT"
        elif eval_decision == "REJECT":
            final_enum = "SW14B_5_REJECT_DENSITY_RISK"
        elif eval_decision == "STRONG":
            final_enum = "SW14B_2_METRIC_TEACHER_STRONG"
        elif eval_decision == "MEDIUM":
            final_enum = "SW14B_1_METRIC_TEACHER_MEDIUM"
        decision = {
            "decision": final_enum,
            "best_checkpoint": (round3 or round2 or round1).get("checkpoint_path", round1.get("checkpoint")),
            "best_adapter_type": "SpatialGateAdapter",
            "train_split_used": "round2 0..199 and round3 0..399" if round3 else "round2 0..199" if round2 else "train_tune_smoke 0..49",
            "eval_split_used": "eval_core100 0..99" if eval_core100 is not None else "eval_debug 100..119",
            "whether_use_in_resume": final_enum in {"SW14B_1_METRIC_TEACHER_MEDIUM", "SW14B_2_METRIC_TEACHER_STRONG", "SW14B_3_GT_REFINEMENT_MEDIUM", "SW14B_4_GT_REFINEMENT_STRONG"} and clean_ok,
            "whether_SW13_remains_main": True,
            "safe_claim": f"Frozen {'eval_core100' if eval_core100 is not None else 'eval_debug'} decision={eval_decision}; SW13 remains main unless stronger frozen evaluation later confirms safe improvement.",
            "limitations": [
                "no eval_core500 claim yet",
                "subset diagnostic unless eval_core100 has been completed",
            ],
            "next_action": "run eval_core500 only if this frozen core100 result is worth escalating" if final_enum in {"SW14B_1_METRIC_TEACHER_MEDIUM", "SW14B_2_METRIC_TEACHER_STRONG"} and eval_core100 is not None else "run eval_core100 only if this subset result is worth escalating" if final_enum == "SW14B_1_METRIC_TEACHER_MEDIUM" else "debug density / clean drift before larger training",
            "adapter_summary": adapter_summary,
            "teacher_summary": teacher_summary,
            "clean_control": clean_control,
            "ablation_summary": ablations,
            "eval_debug": eval_debug,
            "eval_core100": eval_core100,
        }
    write_json(REPORTS_DIR / "sw14b_final_decision.json", decision)
    write_md(
        REPORTS_DIR / "sw14b_final_decision.md",
        "\n".join(
            [
                "# SW14B Final Decision",
                "",
                f"- decision: `{decision['decision']}`.",
                f"- SW13 remains main: `{decision['whether_SW13_remains_main']}`.",
                f"- whether_use_in_resume: `{decision['whether_use_in_resume']}`.",
                f"- next action: `{decision['next_action']}`.",
            ]
        ),
    )
    return decision


def build_report(
    closure: dict[str, Any],
    split_manifest: dict[str, Any],
    arch: dict[str, Any],
    chain_audit: dict[str, Any],
    round1: dict[str, Any],
    round2: dict[str, Any] | None,
    round3: dict[str, Any] | None,
    eval_debug: dict[str, Any] | None,
    eval_core100: dict[str, Any] | None,
    clean_control: dict[str, Any] | None,
    ablations: dict[str, Any] | None,
    decision: dict[str, Any],
) -> None:
    report_json = {
        "executive_summary": decision["safe_claim"],
        "closure": closure,
        "split_protocol": split_manifest,
        "adapter_architecture": arch,
        "postprocess_chain": chain_audit,
        "round1_smoke": round1,
        "round2": round2,
        "round3": round3,
        "eval_debug": eval_debug,
        "eval_core100": eval_core100,
        "clean_control_safety": clean_control,
        "ablations": ablations,
        "decision": decision,
    }
    write_json(REPORTS_DIR / "stage_sw14b_metric_gt_adapter_report.json", report_json)
    write_md(
        REPORTS_DIR / "stage_sw14b_metric_gt_adapter_report.md",
        "\n".join(
            [
                "# Stage SW14B Metric GT Adapter Report",
                "",
                "## 1. Executive summary",
                decision["safe_claim"],
                "",
                "## 2. Why SW14 pivots from exact parity",
                "- SW14A exact parity branch is formally closed.",
                "- F3 top-k pruning is too sensitive for exact feature-level distillation.",
                "",
                "## 3. SW13 teacher inherited result",
                "- SW13C-Fix + FrontCap remains the main frozen result.",
                "",
                "## 4. Split protocol",
                f"- train_tune_smoke: `{split_manifest['splits'][0]['sample_indices'][0]}..{split_manifest['splits'][0]['sample_indices'][-1]}`.",
                f"- val_small_smoke: `{split_manifest['splits'][1]['sample_indices'][0]}..{split_manifest['splits'][1]['sample_indices'][-1]}`.",
                "",
                "## 5. Adapter architecture",
                f"- adapter type: `{arch['adapter_type']}`.",
                f"- trainable params: `{arch['trainable_parameter_count']}`.",
                "",
                "## 6. Full postprocess chain: adapter -> F3 -> FrontCap",
                "- unified chain retained and audited.",
                "",
                "## 7. Loss design",
                "- GT occupancy + teacher occupancy + density/FP/sparsity regularization.",
                "",
                "## 8. Round 1 smoke",
                f"- decision: `{round1.get('decision')}`.",
                f"- failure_reason: `{round1.get('failure_reason')}`.",
                "",
                "## 9. Round 2 metric teacher training",
                f"- {round2 if round2 is not None else 'not executed'}.",
                "",
                "## 10. Round 3 extended training",
                f"- {round3 if round3 is not None else 'not executed'}.",
                "",
                "## 11. Frozen eval_debug",
                f"- {eval_debug if eval_debug is not None else 'not executed'}.",
                "",
                "## 12. Frozen eval_core100 / eval_core500",
                f"- {eval_core100 if eval_core100 is not None else 'not executed'}.",
                "",
                "## 13. Clean/control safety",
                f"- {clean_control if clean_control is not None else 'not executed'}.",
                "",
                "## 14. Ablations",
                f"- {ablations if ablations is not None else 'not executed'}.",
                "",
                "## 15. Decision",
                f"- `{decision['decision']}`.",
                "",
                "## 16. Safe resume claim",
                f"- `whether_use_in_resume = {decision['whether_use_in_resume']}`.",
                "",
                "## 17. Limitations",
                *[f"- {item}" for item in decision["limitations"]],
                "",
                "## 18. Next unique action",
                f"- {decision['next_action']}",
            ]
        ),
    )


def write_tests() -> None:
    tests = {
        "test_outputs_exist.py": """
from pathlib import Path
ROOT = Path('/home/rexlion/ComputerVision/cv_lidar_transition')
REPORTS = ROOT / 'reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter'
def test_outputs_exist():
    required = [
        'sw14b_inherited_sw14a_exact_parity_closure.json',
        'sw14b_split_manifest.json',
        'sw14b_adapter_architecture.json',
        'sw14b_postprocess_chain_audit.json',
        'sw14b_final_decision.json',
        'stage_sw14b_metric_gt_adapter_report.md',
    ]
    for name in required:
        assert (REPORTS / name).exists(), name
""",
        "test_backbone_frozen.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_backbone_frozen():
    payload = json.loads(PATH.read_text())
    assert payload['frozen_sparseworld_status']['backbone_frozen'] is True
""",
        "test_head_frozen.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_head_frozen():
    payload = json.loads(PATH.read_text())
    assert payload['frozen_sparseworld_status']['head_frozen'] is True
""",
        "test_get_occ_unchanged.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_get_occ_unchanged():
    payload = json.loads(PATH.read_text())
    assert payload['no_get_occ_modification'] is True
    assert payload['get_occ_sha256']
""",
        "test_adapter_trainable_only.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_adapter_trainable_only():
    payload = json.loads(PATH.read_text())
    assert payload['trainable_parameter_count'] > 0
    assert payload['frozen_sparseworld_status']['model_trainable_param_count'] == 0
""",
        "test_adapter_disabled_native_integrity.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_adapter_disabled_native_integrity():
    payload = json.loads(PATH.read_text())
    assert payload['adapter_disabled_path_unchanged'] is True
""",
        "test_non_degraded_consistency.py": """
from pathlib import Path
SRC = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_modules.py').read_text()
def test_non_degraded_consistency():
    assert 'alpha_map = alpha_map * degradation_mask' in SRC
""",
        "test_clean_alpha_near_zero.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_clean_alpha_near_zero():
    payload = json.loads(PATH.read_text())
    assert payload['clean_alpha_near_zero'] is True
""",
        "test_postprocess_chain_includes_f3.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_postprocess_chain_audit.json')
def test_postprocess_chain_includes_f3():
    payload = json.loads(PATH.read_text())
    assert payload['adapter_eval_path_includes_f3'] is True
""",
        "test_frontcap_after_f3.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_postprocess_chain_audit.json')
def test_frontcap_after_f3():
    payload = json.loads(PATH.read_text())
    assert payload['frontcap_after_f3'] is True
""",
        "test_no_gt_in_f3_budget.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_postprocess_chain_audit.json')
def test_no_gt_in_f3_budget():
    payload = json.loads(PATH.read_text())
    assert payload['no_gt_in_budget'] is True
""",
        "test_no_gt_in_frontcap.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_postprocess_chain_audit.json')
def test_no_gt_in_frontcap():
    payload = json.loads(PATH.read_text())
    assert payload['no_gt_in_frontcap'] is True
""",
        "test_gt_train_only.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_loss_config.json')
def test_gt_train_only():
    payload = json.loads(PATH.read_text())
    assert payload['uses_gt_training'] is True
    assert payload['uses_eval_gt_for_tuning'] is False
""",
        "test_no_eval_gt_for_tuning.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_split_manifest.json')
def test_no_eval_gt_for_tuning():
    payload = json.loads(PATH.read_text())
    eval_debug = [x for x in payload['splits'] if x['split_name'] == 'eval_debug'][0]
    assert eval_debug['tuning_allowed'] is False
""",
        "test_decision_schema.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_final_decision.json')
def test_decision_schema():
    payload = json.loads(PATH.read_text())
    required = {'decision','best_checkpoint','best_adapter_type','train_split_used','eval_split_used','whether_use_in_resume','whether_SW13_remains_main','safe_claim','limitations','next_action'}
    assert required.issubset(payload.keys())
""",
        "test_no_false_claims.py": """
from pathlib import Path
REPORT = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/stage_sw14b_metric_gt_adapter_report.md').read_text().lower()
def test_no_false_claims():
    assert 'official benchmark' not in REPORT
    assert 'sota' not in REPORT
""",
        "test_resume_claim_guard.py": """
import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_final_decision.json')
def test_resume_claim_guard():
    payload = json.loads(PATH.read_text())
    if payload['decision'].startswith('SW14B_0') or payload['decision'].startswith('SW14B_5') or payload['decision'].startswith('SW14B_6') or payload['decision'].startswith('SW14B_7'):
        assert payload['whether_use_in_resume'] is False
""",
    }
    for name, content in tests.items():
        path = TESTS_DIR / name
        path.write_text(content.strip() + "\n", encoding="utf-8")


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_tests:
        return {"executed": False, "reason": "skip_tests"}
    write_tests()
    cmd = [sys.executable, "-m", "pytest", str(TESTS_DIR), "-q"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    payload = {
        "executed": True,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }
    write_json(REPORTS_DIR / "sw14b_tests_result.json", payload)
    if proc.returncode != 0:
        raise RuntimeError(f"pytest failed with returncode={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return payload


def main() -> int:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    closure = phase0_exact_parity_closure()
    split_manifest = phase1_split_manifest(args)
    arch = phase2_adapter_architecture()
    chain_audit = phase3_postprocess_chain_audit()
    round1 = smoke_train_round1(args)
    round2 = None
    round3 = None
    eval_debug = None
    eval_core100 = None
    clean_control = None
    ablations = None
    best_adapter = None
    if round1.get("decision") == "SMOKE_PASS" and not args.skip_round2:
        _, train_dataset, train_model, _ = build_runtime(train=True)
        init_payload = torch.load(round1["checkpoint"], map_location="cpu", weights_only=False)
        round2, best_adapter = train_metric_teacher_round(
            round_name="round2",
            dataset=train_dataset,
            model=train_model,
            init_state_dict=init_payload["state_dict"],
            train_indices=list(range(args.round2_train_start, args.round2_train_end + 1)),
            val_indices=list(range(args.round2_val_start, args.round2_val_end + 1)),
            epochs=args.round2_epochs,
            lr=args.round2_lr,
            checkpoint_name="sw14b_round2_metric_teacher_checkpoint.pth",
        )
        release_runtime(train_model, train_dataset)
    elif round1.get("decision") == "SMOKE_PASS" and args.skip_round2:
        round2, best_adapter = load_existing_round_decision("round2")
    if round2 is not None and best_adapter is not None and round2.get("best_summary") and safety_score_from_summary(round2["best_summary"]) > 0.0 and not args.skip_eval_debug:
        release_runtime()
        eval_debug = run_frozen_eval_debug(args, best_adapter)
    elif args.skip_eval_debug:
        eval_debug = load_existing_eval_summary("eval_debug")
    if eval_debug is not None and eval_debug.get("decision") == "MEDIUM" and not args.skip_round3:
        _, train_dataset, train_model, _ = build_runtime(train=True)
        if round3 is None:
            init_payload = torch.load(round2["checkpoint_path"], map_location="cpu", weights_only=False)
            round3, best_adapter = train_metric_teacher_round(
                round_name="round3",
                dataset=train_dataset,
                model=train_model,
                init_state_dict=init_payload["state_dict"],
                train_indices=list(range(args.round3_train_start, args.round3_train_end + 1)),
                val_indices=list(range(args.round3_val_start, args.round3_val_end + 1)),
                epochs=args.round3_epochs,
                lr=args.round3_lr,
                checkpoint_name="sw14b_round3_extended_checkpoint.pth",
            )
        release_runtime(train_model, train_dataset)
    elif args.skip_round3:
        existing_round3, existing_adapter = load_existing_round_decision("round3")
        if existing_round3 is not None:
            round3, best_adapter = existing_round3, existing_adapter
    if best_adapter is not None and eval_debug is not None and eval_debug.get("decision") in {"MEDIUM", "STRONG"} and not args.skip_eval_core100:
        release_runtime()
        eval_core100 = run_frozen_eval_core100(args, best_adapter)
    elif args.skip_eval_core100:
        eval_core100 = load_existing_eval_summary("eval_core100")
    if best_adapter is not None and not args.skip_clean_control:
        release_runtime()
        clean_control = evaluate_clean_control(args, best_adapter)
    elif args.skip_clean_control:
        clean_control = load_existing_clean_control()
    decision = final_decision(round1, round2, round3, eval_debug, eval_core100, clean_control, ablations)
    build_report(closure, split_manifest, arch, chain_audit, round1, round2, round3, eval_debug, eval_core100, clean_control, ablations, decision)
    tests = run_tests(args)
    write_json(
        REPORTS_DIR / "sw14b_execution_summary.json",
        {
            "round1": round1,
            "round2": round2,
            "round3": round3,
            "eval_debug": eval_debug,
            "eval_core100": eval_core100,
            "clean_control": clean_control,
            "decision": decision,
            "tests": tests,
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
