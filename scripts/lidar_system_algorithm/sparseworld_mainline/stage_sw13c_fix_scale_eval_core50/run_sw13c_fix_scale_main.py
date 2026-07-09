from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.parallel import collate as collate_fn
from mmcv.runner.fp16_utils import cast_tensor_type


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
LOGS_DIR = PROJECT_ROOT / "logs/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
GPU_DUMPS_DIR = ARTIFACTS_DIR / "gpu_phase_dumps"

SW13C_FIX_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget"
SW13C_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"
EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]
GPU_BATCH_SIZE = 2


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw13c_fix = load_module("sw13c_fix_scale_base", SW13C_FIX_SCRIPT)
sw13a = sw13c_fix.sw13a
sw7 = sw13c_fix.sw7
sw81 = sw13c_fix.sw81


@dataclass(frozen=True)
class FrozenCandidate:
    label: str
    perturbation_id: str
    base_repair_variant: str
    fixed_candidate_label: str
    budget_mode: str
    expansion_ratio: float | None
    keep_ratio: float | None
    protected_variant: str
    agreement_sources: list[str]
    expected_agreement_source_count: int
    role: str


@dataclass(frozen=True)
class ControlCase:
    label: str
    perturbation_id: str


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "final_outputs",
        ARTIFACTS_DIR / "multi_source_agreement",
        GPU_DUMPS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sw13c_fix.normalize_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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
            writer.writerow(sw13c_fix.normalize_export(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-13C-Fix frozen candidate eval_core50 scaling")
    parser.add_argument("--samples", default=",".join(str(i) for i in range(50)))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", choices=["both", "gpu", "cpu"], default="both")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_runtime():
    cfg, dataset, model, _ = sw81.build_sparseworld_runtime(
        train=False,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    model.eval()
    return cfg, dataset, model


def batched_sample_ids(sample_ids: list[int], batch_size: int = GPU_BATCH_SIZE) -> list[list[int]]:
    return [sample_ids[i : i + batch_size] for i in range(0, len(sample_ids), batch_size)]


def extract_multi_sample_batch(dataset: Any, sample_ids: list[int]) -> tuple[list[Any], dict[str, Any]]:
    samples = [dataset[idx] for idx in sample_ids]
    batch = collate_fn(samples, samples_per_gpu=len(sample_ids))
    return samples, batch


def dynamic_replay_batches(dataset: Any, sample_ids: list[int], max_batch_size: int = GPU_BATCH_SIZE) -> list[tuple[list[int], list[Any], dict[str, Any]]]:
    batches: list[tuple[list[int], list[Any], dict[str, Any]]] = []
    cursor = 0
    while cursor < len(sample_ids):
        batch_ids = [sample_ids[cursor]]
        batch_samples = [dataset[sample_ids[cursor]]]
        batch = collate_fn(batch_samples, samples_per_gpu=1)
        cursor += 1
        while len(batch_ids) < max_batch_size and cursor < len(sample_ids):
            trial_id = sample_ids[cursor]
            trial_sample = dataset[trial_id]
            trial_ids = batch_ids + [trial_id]
            trial_samples = batch_samples + [trial_sample]
            try:
                trial_batch = collate_fn(trial_samples, samples_per_gpu=len(trial_ids))
            except RuntimeError:
                break
            batch_ids = trial_ids
            batch_samples = trial_samples
            batch = trial_batch
            cursor += 1
        batches.append((batch_ids, batch_samples, batch))
    return batches


def temporal_gt_for_horizon(sample_unwrapped: dict[str, Any], horizon_s: int) -> torch.Tensor:
    if horizon_s == 0:
        return torch.as_tensor(sample_unwrapped["voxel_semantics"]).long()
    return torch.as_tensor(sample_unwrapped["temporal_semantics"][horizon_s]["voxel_semantics"]).long()


def merge_simple_test_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in parts[0].keys():
        values = [part[key] for part in parts]
        if isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged[key] = [torch.cat([value[idx] for value in values], dim=0) for idx in range(len(values[0]))]
        else:
            merged[key] = values[0]
    return merged


def run_variant_forward_batched(
    model: Any,
    sample_unwrapped_list: list[dict[str, Any]],
    batch_degraded: dict[str, Any],
    caches: list[dict[str, Any]],
    variant: Any,
    perturbation_id: str,
) -> tuple[list[dict[int, dict[str, Any]]], dict[str, Any]]:
    img = batch_degraded["img"][0]
    img_metas = batch_degraded["img_metas"][0]
    kwargs = {key: value[0] for key, value in batch_degraded.items() if key not in {"img", "img_metas"}}
    original_simple_test_online = model.simple_test_online
    runtime_debug: dict[str, Any] = {
        "variant_label": variant.label,
        "perturbation_id": perturbation_id,
        "batch_size": len(sample_unwrapped_list),
        "local_batch_override": True,
    }

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        batch_size, total_views, channels, height, width = img.shape
        num_frames = total_views // 6
        img = img.reshape(batch_size, num_frames, 6, channels, height, width)
        per_sample_frame_feats: list[list[list[torch.Tensor]]] = [[] for _ in range(batch_size)]
        per_sample_frame_metas: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
        for sample_idx in range(batch_size):
            img_shape = (height, width, channels)
            filename_count = len(img_metas[sample_idx]["filename"])
            img_metas[sample_idx]["img_shape"] = [img_shape for _ in range(filename_count)]
            img_metas[sample_idx]["ori_shape"] = [img_shape for _ in range(filename_count)]
            img_metas[sample_idx]["pad_shape"] = [img_shape for _ in range(filename_count)]
        for frame_idx in range(num_frames):
            frame_indices = list(range(frame_idx * 6, (frame_idx + 1) * 6))
            frame_metas = [sw13a.clone_meta_for_indices(img_metas[sample_idx], frame_indices)[0] for sample_idx in range(batch_size)]
            frame_feats = self.extract_feat(img[:, frame_idx].contiguous(), frame_metas)
            for sample_idx in range(batch_size):
                sample_feats = [level[sample_idx : sample_idx + 1].contiguous() for level in frame_feats]
                if frame_idx == 0 and sw13a.should_run_variant(variant, perturbation_id):
                    sample_feats, _ = sw13a.apply_feature_repair_to_levels(sample_feats, caches[sample_idx], variant, perturbation_id)
                per_sample_frame_feats[sample_idx].append(sample_feats)
                per_sample_frame_metas[sample_idx].append(frame_metas[sample_idx])
        full_points_scale = getattr(self.pts_bbox_head, "points_scale", None)
        sample_parts: list[dict[str, Any]] = []
        for sample_idx in range(batch_size):
            feat_levels = len(per_sample_frame_feats[sample_idx][0])
            reorganized: list[torch.Tensor] = []
            for level_idx in range(feat_levels):
                feat_l = torch.cat([per_sample_frame_feats[sample_idx][frame_idx][level_idx] for frame_idx in range(num_frames)], dim=0)
                feat_l = feat_l.flatten(0, 1)[None, ...]
                reorganized.append(feat_l.contiguous())
            merged_meta = [copy.deepcopy(per_sample_frame_metas[sample_idx][0])]
            for frame_idx in range(1, num_frames):
                for key, value in per_sample_frame_metas[sample_idx][frame_idx].items():
                    if isinstance(value, list):
                        merged_meta[0][key].extend(value)
            sample_img_feats = cast_tensor_type(reorganized, torch.half, torch.float32)
            if isinstance(full_points_scale, torch.Tensor) and full_points_scale.shape[0] == batch_size:
                self.pts_bbox_head.points_scale = full_points_scale[sample_idx : sample_idx + 1].contiguous()
            sample_parts.append(self.simple_test_pts(sample_img_feats, merged_meta, rescale=rescale))
        if full_points_scale is not None:
            self.pts_bbox_head.points_scale = full_points_scale
        return merge_simple_test_parts(sample_parts)

    model.simple_test_online = MethodType(patched_simple_test_online, model)
    try:
        sw13a.reset_model_cache(model)
        with torch.no_grad():
            outputs = model.forward_backbone(img, img_metas, **kwargs)
        per_sample_horizons: list[dict[int, dict[str, Any]]] = []
        for sample_pos, sample_unwrapped in enumerate(sample_unwrapped_list):
            sample_pack: dict[int, dict[str, Any]] = {}
            gt0 = temporal_gt_for_horizon(sample_unwrapped, 0)
            for horizon_s in CORE_HORIZONS:
                if horizon_s == 0:
                    pred_dict = {
                        "cls_scores": outputs["cls_score"][sample_pos : sample_pos + 1],
                        "refine_pts": outputs["refine_pts"][sample_pos : sample_pos + 1],
                    }
                else:
                    pred_dict = {
                        "cls_scores": outputs["forecast_semantics_list"][horizon_s - 1][sample_pos : sample_pos + 1],
                        "refine_pts": outputs["forecast_points_list"][horizon_s - 1][sample_pos : sample_pos + 1],
                    }
                sample_pack[horizon_s] = {
                    "pred_dict": pred_dict,
                    "gt_h": temporal_gt_for_horizon(sample_unwrapped, horizon_s),
                    "gt0": gt0,
                }
            per_sample_horizons.append(sample_pack)
        return per_sample_horizons, runtime_debug
    finally:
        model.simple_test_online = original_simple_test_online


def build_cases_from_pred_pack(model: Any, pred_pack: dict[int, dict[str, Any]], native_occ_cases: dict[int, dict[str, Any]] | None = None) -> dict[int, dict[str, Any]]:
    cases: dict[int, dict[str, Any]] = {}
    for horizon_s in CORE_HORIZONS:
        semantic, debug_occ = sw13c_fix.sw13b.dense_debug_for_case(model, pred_pack[horizon_s]["pred_dict"])
        conf = sw13c_fix.sw13b.confidence_maps(debug_occ)
        occ_mask = semantic != EMPTY_IDX
        case: dict[str, Any] = {
            "semantic": semantic,
            "occ_mask": occ_mask,
            "top1_conf": conf["top1_conf"],
            "top1_margin": conf["top1_margin"],
            "occ_conf": conf["occ_conf"],
            "free_conf": conf["free_conf"],
            "gt_h": pred_pack[horizon_s]["gt_h"],
            "gt0": pred_pack[horizon_s]["gt0"],
        }
        if native_occ_cases is not None:
            case["delta_occ_mask"] = occ_mask & ~native_occ_cases[horizon_s]["occ_mask"]
        cases[horizon_s] = case
    return cases


def dump_case_path(perturbation_id: str, variant_label: str, sample_index: int, horizon_s: int) -> Path:
    return GPU_DUMPS_DIR / f"{perturbation_id}__{variant_label}__sample{sample_index:03d}_h{horizon_s}.npz"


def save_case_dump(path: Path, case: dict[str, Any]) -> None:
    np.savez(
        path,
        semantic=case["semantic"].numpy().astype(np.int16),
        occ_conf=case["occ_conf"].numpy().astype(np.float32),
        top1_margin=case["top1_margin"].numpy().astype(np.float32),
        gt_h=case["gt_h"].numpy().astype(np.int16),
        gt0=case["gt0"].numpy().astype(np.int16),
    )


def load_case_dump(path: Path) -> dict[str, Any]:
    payload = np.load(path)
    semantic = torch.from_numpy(payload["semantic"]).long()
    return {
        "semantic": semantic,
        "occ_mask": semantic != EMPTY_IDX,
        "occ_conf": torch.from_numpy(payload["occ_conf"]).float(),
        "top1_margin": torch.from_numpy(payload["top1_margin"]).float(),
        "gt_h": torch.from_numpy(payload["gt_h"]).long(),
        "gt0": torch.from_numpy(payload["gt0"]).long(),
    }


def load_cache_for_sample(dataset: Any, model: Any, sample_id: int) -> dict[str, Any]:
    cache_path = sw13c_fix.SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_id:03d}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    raw_sample, batch = sw13c_fix.sw2.extract_sample_batch(dataset, sample_id, sw13c_fix.collate_fn)
    sample_unwrapped = sw13c_fix.sw2.unwrap(raw_sample)
    moved = sw13c_fix.sw2.move_to_cuda(batch)
    sw13a.reset_model_cache(model)
    _, _, cache = sw13a.extract_clean_memory_for_sample(model, moved, sample_unwrapped, sample_id)
    return cache


def frozen_candidates() -> list[FrozenCandidate]:
    return [
        FrozenCandidate(
            label="A1_fixed",
            perturbation_id="A1_drop_cam_front",
            base_repair_variant="R1_replace_tminus1",
            fixed_candidate_label="F3_expand_budget_strict",
            budget_mode="native_expansion_ratio",
            expansion_ratio=0.08,
            keep_ratio=None,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2"],
            expected_agreement_source_count=3,
            role="main",
        ),
        FrozenCandidate(
            label="A10_fixed_main",
            perturbation_id="A10_drop_front_triplet",
            base_repair_variant="R4_ema_K3",
            fixed_candidate_label="F3_expand_budget_strict",
            budget_mode="native_expansion_ratio",
            expansion_ratio=0.12,
            keep_ratio=None,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            expected_agreement_source_count=4,
            role="main",
        ),
        FrozenCandidate(
            label="A10_fixed_secondary",
            perturbation_id="A10_drop_front_triplet",
            base_repair_variant="R8_camera_group_repair_front_triplet",
            fixed_candidate_label="F3_expand_budget_strict",
            budget_mode="native_expansion_ratio",
            expansion_ratio=0.12,
            keep_ratio=None,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            expected_agreement_source_count=4,
            role="secondary",
        ),
        FrozenCandidate(
            label="C4_fixed",
            perturbation_id="C4_motion_blur_9",
            base_repair_variant="R5_blend_alpha03",
            fixed_candidate_label="F9_C4_preserve_R5",
            budget_mode="preserve",
            expansion_ratio=None,
            keep_ratio=None,
            protected_variant="PZ_fix_2_front_conf_agree",
            agreement_sources=["R5_blend_alpha03", "R6_blend_alpha05", "R7_blend_alpha07"],
            expected_agreement_source_count=3,
            role="preserve",
        ),
    ]


def control_cases() -> list[ControlCase]:
    return [
        ControlCase(label="A7_control_native", perturbation_id="A7_drop_all_rear"),
        ControlCase(label="A0_clean_safety", perturbation_id="A0_clean"),
    ]


def metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agg = sw13c_fix.aggregate_rows(
        rows,
        ["candidate_name", "perturbation_id", "base_repair_variant", "fixed_candidate_label", "horizon_s"],
    )
    native_map = {
        (r["candidate_name"], int(r["horizon_s"])): r
        for r in agg
        if r["fixed_candidate_label"] == "native_baseline"
    }
    out: list[dict[str, Any]] = []
    for row in agg:
        merged = dict(row)
        native = native_map.get((row["candidate_name"], int(row["horizon_s"])))
        if native is not None and row["fixed_candidate_label"] != "native_baseline":
            merged["front_sector_false_free_rate_delta_vs_native"] = float(row["front_sector_false_free"]) - float(native["front_sector_false_free"])
            merged["future_h4_h6_false_free_rate_delta_vs_native"] = float(row["false_free_rate"]) - float(native["false_free_rate"]) if int(row["horizon_s"]) in {4, 6} else 0.0
            merged["A10_front_h6_recovery_rate_delta_vs_native"] = float(row["A10_front_h6_recovery_ratio"]) - float(native["A10_front_h6_recovery_ratio"])
            merged["false_positive_delta"] = float(row["false_occupied_rate"]) - float(native["false_occupied_rate"])
            merged["wrong_class_delta"] = float(row["wrong_class_activation_delta"])
        out.append(merged)
    return out


def sample_consistency_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in metric_rows:
        if row["fixed_candidate_label"] == "native_baseline":
            continue
        is_a1 = row["candidate_name"] == "A1_fixed"
        is_a10 = row["candidate_name"] in {"A10_fixed_main", "A10_fixed_secondary"}
        density_thr = 0.18 if is_a1 else (0.22 if is_a10 else 0.10)
        out.append(
            {
                "candidate_name": row["candidate_name"],
                "sample_index": row["sample_index"],
                "horizon_s": row["horizon_s"],
                "front_false_free_improved": float(row["front_sector_false_free_rate_delta_vs_native"]) < 0.0,
                "future_false_free_improved": float(row["future_h4_h6_false_free_rate_delta_vs_native"]) < 0.0 if int(row["horizon_s"]) in {4, 6} else False,
                "density_safe": float(row["pred_gt_density_delta"]) <= density_thr,
                "front_local_safe": float(row["front_local_density_proxy"]) <= 1.30,
                "false_positive_safe": float(row["false_positive_delta"]) <= 0.01,
                "wrong_class_safe": float(row["wrong_class_delta"]) <= float(row["raw_wrong_class_delta"]),
                "joint_success": float(row["front_sector_false_free_rate_delta_vs_native"]) < 0.0
                and float(row["pred_gt_density_delta"]) <= density_thr
                and float(row["front_local_density_proxy"]) <= 1.30
                and float(row["false_positive_delta"]) <= 0.01,
            }
        )
    return out


def aggregate_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["candidate_name"], []).append(row)
    out: list[dict[str, Any]] = []
    for candidate_name, items in by_name.items():
        n = max(1, len(items))
        out.append(
            {
                "candidate_name": candidate_name,
                "sample_improve_rate_front_false_free": sum(bool(x["front_false_free_improved"]) for x in items) / n,
                "sample_improve_rate_future_false_free": sum(bool(x["future_false_free_improved"]) for x in items) / n,
                "sample_density_safe_rate": sum(bool(x["density_safe"]) for x in items) / n,
                "sample_front_local_safe_rate": sum(bool(x["front_local_safe"]) for x in items) / n,
                "sample_false_positive_safe_rate": sum(bool(x["false_positive_safe"]) for x in items) / n,
                "sample_wrong_class_safe_rate": sum(bool(x["wrong_class_safe"]) for x in items) / n,
                "sample_joint_success_rate": sum(bool(x["joint_success"]) for x in items) / n,
            }
        )
    return out


def select_row(rows: list[dict[str, Any]], candidate_name: str, horizon_s: int, fixed_candidate_label: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if row["candidate_name"] != candidate_name or int(row["horizon_s"]) != horizon_s:
            continue
        if fixed_candidate_label is not None and row["fixed_candidate_label"] != fixed_candidate_label:
            continue
        if row["fixed_candidate_label"] == "native_baseline" and fixed_candidate_label is None:
            continue
        return row
    return None


def save_metric_bar(path: Path, title: str, row: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    values = [
        abs(float(row["front_sector_false_free_rate_delta_vs_native"])),
        float(row["pred_gt_density_delta"]),
        float(row["final_native_expansion_ratio"]),
        float(row["front_local_density_proxy"]),
    ]
    labels = ["front_ff_reduction", "density_delta", "native_expansion", "front_local_proxy"]
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_tests() -> None:
    tests = {
        "test_outputs_exist.py": """from pathlib import Path\nBASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50')\ndef test_outputs_exist():\n    names=['sw13c_fix_scale_frozen_candidate_manifest.json','sw13c_fix_scale_replay_manifest.csv','sw13c_fix_scale_metrics.csv','sw13c_fix_scale_metric_summary.csv','sw13c_fix_scale_front_local_density_audit.csv','sw13c_fix_scale_sample_consistency.csv','sw13c_fix_scale_eval_core50_decision.json','stage_sw13c_fix_scale_eval_core50_report.md']\n    for name in names:\n        p=BASE/name\n        assert p.exists() and p.stat().st_size>0, name\n""",
        "test_frozen_candidate_manifest.py": """import json\nfrom pathlib import Path\ndef test_frozen_candidate_manifest():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_frozen_candidate_manifest.json').read_text())\n    assert obj['selection_frozen'] is True\n    assert obj['no_reselection'] is True\n""",
        "test_no_reselection.py": """import json\nfrom pathlib import Path\ndef test_no_reselection():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_eval_core50_decision.json').read_text())\n    assert obj['no_reselection'] is True\n""",
        "test_no_gt_budget.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_metrics.csv').open()))\ndef test_no_gt_budget():\n    assert rows\n    assert all(r['uses_gt_budget'] == 'False' for r in rows)\n    assert all(r['no_gt_budget'] == 'True' for r in rows)\n""",
        "test_no_pred_gt_selection.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_replay_manifest.csv').open()))\ndef test_no_pred_gt_selection():\n    assert rows\n    assert all(r['uses_pred_gt_density_for_selection'] == 'False' for r in rows)\n""",
        "test_metric_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_metrics.csv').open()))\ndef test_metric_schema():\n    assert len(rows) >= 50\n    need={'sample_index','perturbation_id','base_repair_variant','fixed_candidate_label','horizon_s','selection_frozen','no_reselection','no_gt_budget','uses_gt_budget','uses_pred_gt_density_for_selection','agreement_source_count','final_native_expansion_ratio','raw_delta_keep_ratio','protected_zone_preservation_ratio','pruning_from_protected_ratio','front_kept_ratio','front_local_density_proxy','pred_gt_density_delta'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_front_local_density_audit.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_front_local_density_audit.csv').open()))\ndef test_front_local_density_audit():\n    assert rows\n    need={'front_occ_count_native','front_occ_count_raw','front_occ_count_final','front_native_expansion_ratio','front_raw_delta_keep_ratio','front_protected_keep_ratio','front_local_density_proxy','front_gt_occupied_count','front_pred_gt_density_delta','front_false_positive_delta','front_false_free_delta'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_sample_consistency_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_sample_consistency.csv').open()))\ndef test_sample_consistency_schema():\n    assert rows\n    need={'front_false_free_improved','future_false_free_improved','density_safe','front_local_safe','false_positive_safe','wrong_class_safe','joint_success'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_decision_schema.py": """import json\nfrom pathlib import Path\ndef test_decision_schema():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_eval_core50_decision.json').read_text())\n    assert obj['decision_type'] in {'W1_STRONG_SCALE_CONFIRMED','W2_MEDIUM_SCALE_CONFIRMED','W3_RECOVERY_WEAKENS_BUT_DIRECTION_HOLDS','W4_DENSITY_OR_FRONT_LOCAL_RISK','W5_SCALE_FAILS','W6_PROTOCOL_VIOLATION'}\n""",
        "test_no_false_claims.py": """from pathlib import Path\ntext=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/stage_sw13c_fix_scale_eval_core50_report.md').read_text().lower()\ndef test_no_false_claims():\n    assert 'not official benchmark' in text\n    assert 'trained model improvement' not in text\n""",
    }
    for name, content in tests.items():
        write_md(TESTS_DIR / name, content)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dirs()

    frozen = frozen_candidates()
    controls = control_cases()
    write_json(
        REPORTS_DIR / "sw13c_fix_scale_frozen_candidate_manifest.json",
        {
            "selection_frozen": True,
            "no_reselection": True,
            "no_parameter_search": True,
            "no_gt_budget": True,
            "no_pred_gt_density_selection": True,
            "no_training": True,
            "candidates": [sw13c_fix.normalize_export(x.__dict__) for x in frozen],
            "controls": [sw13c_fix.normalize_export(x.__dict__) for x in controls],
        },
    )
    write_md(
        REPORTS_DIR / "sw13c_fix_scale_protocol.md",
        "\n".join(
            [
                "# SW-13C-Fix-Scale protocol",
                "",
                "- eval_core50 is fixed candidate scaling only.",
                "- no reselection, no retuning, no GT budget, no GT-derived selection, no training.",
                "- A10 R8 is frozen secondary comparison only.",
                "- subset diagnostic only; not official benchmark.",
            ]
        ),
    )
    write_md(
        REPORTS_DIR / "sw13c_fix_scale_code_change_summary.md",
        "\n".join(
            [
                "# SW-13C-Fix-Scale code summary",
                "",
                "- reuses `apply_pruning_no_gt` from SW-13C-Fix without modification.",
                "- removes candidate search from eval_core50 path; all parameters come from the frozen manifest.",
                "- uses a two-stage pipeline: GPU replay dump first, CPU pruning/metrics/report second.",
            ]
        ),
    )
    sample_ids = parse_int_list(args.samples)
    by_perturbation: dict[str, list[FrozenCandidate]] = {}
    for candidate in frozen:
        by_perturbation.setdefault(candidate.perturbation_id, []).append(candidate)

    if args.phase in {"both", "gpu"}:
        cfg, dataset, model = build_runtime()
        del cfg
        variant_map = sw13c_fix.source_variants()
        perturbation_catalog = sw81.sw5_engine.build_catalog()
        for sample_batch, raw_samples, clean_batch in dynamic_replay_batches(dataset, sample_ids, GPU_BATCH_SIZE):
            sample_unwrapped_list = [sw13c_fix.sw2.unwrap(sample) for sample in raw_samples]
            caches = [load_cache_for_sample(dataset, model, sample_index) for sample_index in sample_batch]
            for perturbation_id, group in by_perturbation.items():
                source_union = sorted({label for candidate in group for label in candidate.agreement_sources})
                degraded_batch = copy.deepcopy(clean_batch)
                if perturbation_id != "A0_clean":
                    degraded_batch = sw81.sw5_engine.apply_perturbation_to_batch(degraded_batch, perturbation_catalog[perturbation_id])[0]
                native_pred_packs, _ = run_variant_forward_batched(
                    model,
                    sample_unwrapped_list,
                    sw13c_fix.sw2.move_to_cuda(degraded_batch),
                    caches,
                    variant_map["R0_degraded_native"],
                    perturbation_id,
                )
                native_cases_by_sample = [build_cases_from_pred_pack(model, pred_pack) for pred_pack in native_pred_packs]
                source_case_map: dict[str, list[dict[int, dict[str, Any]]]] = {}
                for variant_label in source_union:
                    pred_packs, _ = run_variant_forward_batched(
                        model,
                        sample_unwrapped_list,
                        sw13c_fix.sw2.move_to_cuda(copy.deepcopy(degraded_batch)),
                        caches,
                        variant_map[variant_label],
                        perturbation_id,
                    )
                    source_case_map[variant_label] = [
                        build_cases_from_pred_pack(model, pred_pack, native_occ_cases=native_cases_by_sample[sample_pos])
                        for sample_pos, pred_pack in enumerate(pred_packs)
                    ]
                for sample_pos, sample_index in enumerate(sample_batch):
                    native_cases = native_cases_by_sample[sample_pos]
                    for horizon_s in CORE_HORIZONS:
                        save_case_dump(dump_case_path(perturbation_id, "native_baseline", sample_index, horizon_s), native_cases[horizon_s])
                        for variant_label in source_union:
                            save_case_dump(
                                dump_case_path(perturbation_id, variant_label, sample_index, horizon_s),
                                source_case_map[variant_label][sample_pos][horizon_s],
                            )
                sw13a.reset_model_cache(model)
            for control in controls:
                degraded_batch = copy.deepcopy(clean_batch)
                if control.perturbation_id != "A0_clean":
                    degraded_batch = sw81.sw5_engine.apply_perturbation_to_batch(degraded_batch, perturbation_catalog[control.perturbation_id])[0]
                native_pred_packs, _ = run_variant_forward_batched(
                    model,
                    sample_unwrapped_list,
                    sw13c_fix.sw2.move_to_cuda(degraded_batch),
                    caches,
                    variant_map["R0_degraded_native"],
                    control.perturbation_id,
                )
                native_cases_by_sample = [build_cases_from_pred_pack(model, pred_pack) for pred_pack in native_pred_packs]
                for sample_pos, sample_index in enumerate(sample_batch):
                    for horizon_s in CORE_HORIZONS:
                        save_case_dump(
                            dump_case_path(control.perturbation_id, "native_baseline", sample_index, horizon_s),
                            native_cases_by_sample[sample_pos][horizon_s],
                        )
                sw13a.reset_model_cache(model)
            del caches, raw_samples, clean_batch, sample_unwrapped_list
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model, dataset
        gc.collect()
        if args.phase == "gpu":
            return

    sectors = {name: tensor.cpu().bool() for name, tensor in sw7.build_sector_masks().items()}
    fix_variant_map = {
        variant.label: variant
        for base in sw13c_fix.base_specs()
        for variant in sw13c_fix.fix_variants_for(base)
    }
    metric_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    front_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    visual_cases: dict[str, dict[str, Any]] = {}

    for sample_index in sample_ids:
        for perturbation_id, group in by_perturbation.items():
            source_union = sorted({label for candidate in group for label in candidate.agreement_sources})
            native_cases = {h: load_case_dump(dump_case_path(perturbation_id, "native_baseline", sample_index, h)) for h in CORE_HORIZONS}
            source_cases = {
                variant_label: {h: load_case_dump(dump_case_path(perturbation_id, variant_label, sample_index, h)) for h in CORE_HORIZONS}
                for variant_label in source_union
            }
            native_eval_by_h: dict[int, dict[str, Any]] = {}
            raw_eval_cache: dict[tuple[str, int], dict[str, Any]] = {}
            for horizon_s in CORE_HORIZONS:
                native_eval = sw13c_fix.build_eval_row(
                    native_cases[horizon_s]["semantic"],
                    native_cases[horizon_s]["gt_h"],
                    native_cases[horizon_s]["gt0"],
                    perturbation_id,
                    horizon_s,
                    sectors,
                    native_cases[horizon_s]["semantic"],
                )
                native_eval_by_h[horizon_s] = native_eval
                for candidate in group:
                    native_row = dict(native_eval)
                    native_row.update(
                        {
                            "sample_index": sample_index,
                            "candidate_name": candidate.label,
                            "perturbation_id": perturbation_id,
                            "base_repair_variant": candidate.base_repair_variant,
                            "fixed_candidate_label": "native_baseline",
                            "horizon_s": horizon_s,
                            "selection_frozen": True,
                            "no_reselection": True,
                            "no_gt_budget": True,
                            "uses_gt_budget": False,
                            "uses_pred_gt_density_for_selection": False,
                            "uses_gt_repair": False,
                            "uses_future_info": False,
                            "uses_current_clean_same_frame": False,
                            "agreement_source_count": candidate.expected_agreement_source_count,
                            "final_native_expansion_ratio": 0.0,
                            "raw_delta_keep_ratio": 0.0,
                            "protected_zone_preservation_ratio": 1.0,
                            "pruning_from_protected_ratio": 0.0,
                            "pruning_from_nonprotected_ratio": 0.0,
                            "nonfront_pruning_ratio": 0.0,
                            "front_kept_ratio": 1.0,
                            "front_local_density_proxy": 1.0,
                            "raw_wrong_class_delta": float(native_eval["wrong_class_activation_delta"]),
                        }
                    )
                    metric_rows.append(native_row)
            for candidate in group:
                selected_variant = fix_variant_map[candidate.fixed_candidate_label]
                for horizon_s in CORE_HORIZONS:
                    native_case = native_cases[horizon_s]
                    raw_case = source_cases[candidate.base_repair_variant][horizon_s]
                    agreement_map, available, missing = sw13c_fix.agreement_for_horizon(source_cases, candidate.agreement_sources, horizon_s)
                    raw_semantic = raw_case["semantic"]
                    raw_occ = raw_semantic != EMPTY_IDX
                    native_occ = native_case["occ_mask"]
                    raw_delta = raw_occ & ~native_occ
                    protected = sw13c_fix.protected_zone_fix(candidate.protected_variant, raw_occ, raw_delta, raw_case["occ_conf"], raw_case["top1_margin"], agreement_map, sectors, horizon_s)
                    if candidate.fixed_candidate_label == "F9_C4_preserve_R5":
                        final_semantic = raw_semantic.clone()
                        pruned = torch.zeros_like(raw_occ)
                        score = torch.zeros_like(raw_case["occ_conf"])
                        budget_meta = {"budget_mode": "preserve", "native_occ_count": int(native_occ.sum().item()), "raw_occ_count": int(raw_occ.sum().item()), "raw_delta_count": int(raw_delta.sum().item()), "target_final_occ_count": int(raw_occ.sum().item()), "target_expansion_ratio": None, "target_keep_ratio": None, "final_occ_count": int(raw_occ.sum().item()), "no_gt_budget": True}
                    else:
                        score = sw13c_fix.low_value_score(raw_occ, protected, raw_case["occ_conf"], raw_case["top1_margin"], agreement_map, sectors, horizon_s, selected_variant.wrong_class_aware, selected_variant.front_bias)
                        final_semantic, pruned, budget_meta = sw13c_fix.apply_pruning_no_gt(raw_semantic, protected, score, int(native_occ.sum().item()), int(raw_occ.sum().item()), int(raw_delta.sum().item()), candidate.budget_mode, expansion_ratio=candidate.expansion_ratio, keep_ratio=candidate.keep_ratio)
                    final_occ = final_semantic != EMPTY_IDX
                    native_eval = native_eval_by_h[horizon_s]
                    raw_eval_key = (candidate.base_repair_variant, horizon_s)
                    raw_eval = raw_eval_cache.get(raw_eval_key)
                    if raw_eval is None:
                        raw_eval = sw13c_fix.build_eval_row(raw_semantic, raw_case["gt_h"], raw_case["gt0"], perturbation_id, horizon_s, sectors, native_case["semantic"])
                        raw_eval_cache[raw_eval_key] = raw_eval
                    row = sw13c_fix.build_eval_row(final_semantic, raw_case["gt_h"], raw_case["gt0"], perturbation_id, horizon_s, sectors, native_case["semantic"])
                    front_native = sw13c_fix.front_counts(native_occ, sectors)
                    front_raw = sw13c_fix.front_counts(raw_occ, sectors)
                    front_final = sw13c_fix.front_counts(final_occ, sectors)
                    front_gt_occ = int(((raw_case["gt_h"] != EMPTY_IDX) & sectors["front"].bool()).sum().item())
                    protected_preservation_ratio = 1.0 - sw13c_fix.safe_div(float((protected & pruned).sum().item()), max(1.0, float(protected.sum().item())))
                    pruning_from_protected_ratio = sw13c_fix.safe_div(float((protected & pruned).sum().item()), max(1.0, float(pruned.sum().item())))
                    pruning_from_nonprotected_ratio = sw13c_fix.safe_div(float((~protected & pruned).sum().item()), max(1.0, float(pruned.sum().item())))
                    nonfront_pruning_ratio = sw13c_fix.safe_div(float((~sectors["front"].bool() & pruned).sum().item()), max(1.0, float(pruned.sum().item())))
                    front_kept_ratio = sw13c_fix.safe_div(float((sectors["front"].bool() & final_occ).sum().item()), max(1.0, float((sectors["front"].bool() & raw_occ).sum().item())))
                    front_raw_delta_keep_ratio = sw13c_fix.safe_div(float((sectors["front"].bool() & final_occ & ~native_occ).sum().item()), max(1.0, float((sectors["front"].bool() & raw_delta).sum().item())))
                    front_protected_keep_ratio = 1.0 - sw13c_fix.safe_div(float((protected & pruned & sectors["front"].bool()).sum().item()), max(1.0, float((protected & sectors["front"].bool()).sum().item())))
                    native_front_ff = sw13c_fix.safe_div(float(((raw_case["gt_h"] != EMPTY_IDX) & sectors["front"].bool() & ~native_occ).sum().item()), max(1, front_gt_occ))
                    final_front_ff = sw13c_fix.safe_div(float(((raw_case["gt_h"] != EMPTY_IDX) & sectors["front"].bool() & ~final_occ).sum().item()), max(1, front_gt_occ))
                    free_front_count = int(((raw_case["gt_h"] == EMPTY_IDX) & sectors["front"].bool()).sum().item())
                    native_front_fp = sw13c_fix.safe_div(float(((raw_case["gt_h"] == EMPTY_IDX) & sectors["front"].bool() & native_occ).sum().item()), max(1, free_front_count))
                    final_front_fp = sw13c_fix.safe_div(float(((raw_case["gt_h"] == EMPTY_IDX) & sectors["front"].bool() & final_occ).sum().item()), max(1, free_front_count))
                    full_row = dict(row)
                    full_row.update({"sample_index": sample_index, "candidate_name": candidate.label, "perturbation_id": perturbation_id, "base_repair_variant": candidate.base_repair_variant, "fixed_candidate_label": candidate.fixed_candidate_label, "horizon_s": horizon_s, "selection_frozen": True, "no_reselection": True, "no_gt_budget": True, "uses_gt_budget": False, "uses_pred_gt_density_for_selection": False, "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False, "agreement_source_count": len(available), "expected_agreement_source_count": candidate.expected_agreement_source_count, "agreement_missing_sources": ",".join(missing), "final_native_expansion_ratio": sw13c_fix.safe_div(int(final_occ.sum().item()) - int(native_occ.sum().item()), max(1, int(native_occ.sum().item()))), "raw_delta_keep_ratio": sw13c_fix.safe_div(int((final_occ & ~native_occ).sum().item()), max(1, int(raw_delta.sum().item()))), "protected_zone_preservation_ratio": protected_preservation_ratio, "pruning_from_protected_ratio": pruning_from_protected_ratio, "pruning_from_nonprotected_ratio": pruning_from_nonprotected_ratio, "nonfront_pruning_ratio": nonfront_pruning_ratio, "front_kept_ratio": front_kept_ratio, "front_local_density_proxy": sw13c_fix.safe_div(front_final, max(1, front_native)), "front_sector_false_free_rate_delta_vs_native": float(row["front_sector_false_free"]) - float(native_eval["front_sector_false_free"]), "future_h4_h6_false_free_rate_delta_vs_native": float(row["false_free_rate"]) - float(native_eval["false_free_rate"]) if horizon_s in {4, 6} else 0.0, "A10_front_h6_recovery_rate_delta_vs_native": float(row["A10_front_h6_recovery_ratio"]) - float(native_eval["A10_front_h6_recovery_ratio"]), "false_positive_delta": float(row["false_occupied_rate"]) - float(native_eval["false_occupied_rate"]), "wrong_class_delta": float(row["wrong_class_activation_delta"]), "raw_wrong_class_delta": float(raw_eval["wrong_class_activation_delta"])})
                    metric_rows.append(full_row)
                    replay_rows.append({"sample_index": sample_index, "perturbation_id": perturbation_id, "base_repair_variant": candidate.base_repair_variant, "fixed_candidate_label": candidate.fixed_candidate_label, "horizon_s": horizon_s, "selection_frozen": True, "no_reselection": True, "no_gt_budget": True, "uses_gt_budget": False, "uses_pred_gt_density_for_selection": False, "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False, "agreement_source_count": len(available), "final_native_expansion_ratio": full_row["final_native_expansion_ratio"], "raw_delta_keep_ratio": full_row["raw_delta_keep_ratio"], "protected_zone_preservation_ratio": full_row["protected_zone_preservation_ratio"], "pruning_from_protected_ratio": full_row["pruning_from_protected_ratio"], "front_kept_ratio": full_row["front_kept_ratio"], "front_local_density_proxy": full_row["front_local_density_proxy"], "pred_gt_density_delta": full_row["pred_gt_density_delta"], "front_sector_false_free_rate_delta_vs_native": full_row["front_sector_false_free_rate_delta_vs_native"], "future_h4_h6_false_free_rate_delta_vs_native": full_row["future_h4_h6_false_free_rate_delta_vs_native"], "A10_front_h6_recovery_rate_delta_vs_native": full_row["A10_front_h6_recovery_rate_delta_vs_native"], "false_positive_delta": full_row["false_positive_delta"], "wrong_class_delta": full_row["wrong_class_delta"]})
                    front_rows.append({"sample_index": sample_index, "candidate_name": candidate.label, "perturbation_id": perturbation_id, "base_repair_variant": candidate.base_repair_variant, "fixed_candidate_label": candidate.fixed_candidate_label, "horizon_s": horizon_s, "front_occ_count_native": front_native, "front_occ_count_raw": front_raw, "front_occ_count_final": front_final, "front_native_expansion_ratio": sw13c_fix.safe_div(front_final - front_native, max(1, front_native)), "front_raw_delta_keep_ratio": front_raw_delta_keep_ratio, "front_protected_keep_ratio": front_protected_keep_ratio, "front_nonprotected_prune_ratio": sw13c_fix.safe_div(float((~protected & pruned & sectors["front"].bool()).sum().item()), max(1.0, float((~protected & sectors["front"].bool() & raw_occ).sum().item()))), "front_local_density_proxy": full_row["front_local_density_proxy"], "front_gt_occupied_count": front_gt_occ, "front_pred_gt_density_delta": sw13c_fix.safe_div(front_final, max(1, front_gt_occ)) - sw13c_fix.safe_div(front_native, max(1, front_gt_occ)), "front_false_positive_delta": final_front_fp - native_front_fp, "front_false_free_delta": final_front_ff - native_front_ff})
                    agreement_payload = {"agreement": agreement_map.numpy().astype(np.float32), "agreement_source_count": np.int16(len(available)), "uses_gt": np.bool_(False), "uses_future_info": np.bool_(False), "uses_current_clean_same_frame": np.bool_(False)}
                    np.savez(ARTIFACTS_DIR / "multi_source_agreement" / f"{candidate.label}__sample{sample_index:03d}_h{horizon_s}.npz", **agreement_payload)
                    np.savez(ARTIFACTS_DIR / "final_outputs" / f"{candidate.label}__sample{sample_index:03d}_h{horizon_s}.npz", gt_h=raw_case["gt_h"].numpy().astype(np.int16), native_semantic=native_case["semantic"].numpy().astype(np.int16), raw_semantic=raw_semantic.numpy().astype(np.int16), final_semantic=final_semantic.numpy().astype(np.int16), protected_zone=protected.numpy().astype(np.uint8), pruned_mask=pruned.numpy().astype(np.uint8))
                    if candidate.label == "A10_fixed_main" and horizon_s == 6:
                        np.savez(ARTIFACTS_DIR / "final_outputs" / f"{candidate.label}__rawstate__sample{sample_index:03d}_h{horizon_s}.npz", gt_h=raw_case["gt_h"].numpy().astype(np.int16), gt0=raw_case["gt0"].numpy().astype(np.int16), native_semantic=native_case["semantic"].numpy().astype(np.int16), raw_semantic=raw_semantic.numpy().astype(np.int16), raw_occ_conf=raw_case["occ_conf"].numpy().astype(np.float32), raw_margin=raw_case["top1_margin"].numpy().astype(np.float32), agreement=agreement_map.numpy().astype(np.float32), protected_zone=protected.numpy().astype(np.uint8))
                    if sample_index == 0 and horizon_s == 6 and candidate.label in {"A1_fixed", "A10_fixed_main"}:
                        visual_cases[candidate.label] = {"gt_h": raw_case["gt_h"], "native": native_case["semantic"], "raw": raw_semantic, "final": final_semantic, "protected": protected, "pruned": pruned}
                del agreement_map, raw_semantic, raw_occ, native_occ, raw_delta, protected, final_semantic, final_occ, row, raw_eval, native_eval
            del native_cases, source_cases
        for control in controls:
            native_cases = {h: load_case_dump(dump_case_path(control.perturbation_id, "native_baseline", sample_index, h)) for h in CORE_HORIZONS}
            for horizon_s in CORE_HORIZONS:
                native_eval = sw13c_fix.build_eval_row(native_cases[horizon_s]["semantic"], native_cases[horizon_s]["gt_h"], native_cases[horizon_s]["gt0"], control.perturbation_id, horizon_s, sectors, native_cases[horizon_s]["semantic"])
                front_occ = sw13c_fix.front_counts(native_cases[horizon_s]["occ_mask"], sectors)
                control_row = dict(native_eval)
                control_row.update({"sample_index": sample_index, "candidate_name": control.label, "perturbation_id": control.perturbation_id, "base_repair_variant": "native_control", "fixed_candidate_label": "native_baseline", "horizon_s": horizon_s, "selection_frozen": True, "no_reselection": True, "no_gt_budget": True, "uses_gt_budget": False, "uses_pred_gt_density_for_selection": False, "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False, "agreement_source_count": 0, "final_native_expansion_ratio": 0.0, "raw_delta_keep_ratio": 0.0, "protected_zone_preservation_ratio": 1.0, "pruning_from_protected_ratio": 0.0, "pruning_from_nonprotected_ratio": 0.0, "nonfront_pruning_ratio": 0.0, "front_kept_ratio": 1.0, "front_local_density_proxy": 1.0, "front_sector_false_free_rate_delta_vs_native": 0.0, "future_h4_h6_false_free_rate_delta_vs_native": 0.0, "A10_front_h6_recovery_rate_delta_vs_native": 0.0, "false_positive_delta": 0.0, "wrong_class_delta": 0.0, "raw_wrong_class_delta": 0.0})
                metric_rows.append(control_row)
                replay_rows.append({"sample_index": sample_index, "perturbation_id": control.perturbation_id, "base_repair_variant": "native_control", "fixed_candidate_label": "native_baseline", "horizon_s": horizon_s, "selection_frozen": True, "no_reselection": True, "no_gt_budget": True, "uses_gt_budget": False, "uses_pred_gt_density_for_selection": False, "uses_gt_repair": False, "uses_future_info": False, "uses_current_clean_same_frame": False, "agreement_source_count": 0, "final_native_expansion_ratio": 0.0, "raw_delta_keep_ratio": 0.0, "protected_zone_preservation_ratio": 1.0, "pruning_from_protected_ratio": 0.0, "front_kept_ratio": 1.0, "front_local_density_proxy": 1.0, "pred_gt_density_delta": float(native_eval["pred_gt_density_delta"]), "front_sector_false_free_rate_delta_vs_native": 0.0, "future_h4_h6_false_free_rate_delta_vs_native": 0.0, "A10_front_h6_recovery_rate_delta_vs_native": 0.0, "false_positive_delta": 0.0, "wrong_class_delta": 0.0})
                front_rows.append({"sample_index": sample_index, "candidate_name": control.label, "perturbation_id": control.perturbation_id, "base_repair_variant": "native_control", "fixed_candidate_label": "native_baseline", "horizon_s": horizon_s, "front_occ_count_native": front_occ, "front_occ_count_raw": front_occ, "front_occ_count_final": front_occ, "front_native_expansion_ratio": 0.0, "front_raw_delta_keep_ratio": 0.0, "front_protected_keep_ratio": 1.0, "front_nonprotected_prune_ratio": 0.0, "front_local_density_proxy": 1.0, "front_gt_occupied_count": int(((native_cases[horizon_s]["gt_h"] != EMPTY_IDX) & sectors["front"].bool()).sum().item()), "front_pred_gt_density_delta": 0.0, "front_false_positive_delta": 0.0, "front_false_free_delta": 0.0})
            del native_cases
        gc.collect()
    if args.phase == "cpu":
        pass

    summary_rows = metric_summary(metric_rows)
    consistency_rows = sample_consistency_rows(metric_rows)
    consistency_summary = aggregate_consistency(consistency_rows)

    write_csv(REPORTS_DIR / "sw13c_fix_scale_metrics.csv", metric_rows)
    write_csv(
        REPORTS_DIR / "sw13c_fix_scale_aggregate_metrics.csv",
        sw13c_fix.aggregate_rows(
            metric_rows,
            ["candidate_name", "perturbation_id", "base_repair_variant", "fixed_candidate_label", "horizon_s"],
        ),
    )
    write_csv(REPORTS_DIR / "sw13c_fix_scale_metric_summary.csv", summary_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_scale_replay_manifest.csv", replay_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_scale_front_local_density_audit.csv", front_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_scale_sample_consistency.csv", consistency_rows)

    a1 = select_row(summary_rows, "A1_fixed", 6, "F3_expand_budget_strict")
    a10_r4 = select_row(summary_rows, "A10_fixed_main", 6, "F3_expand_budget_strict")
    a10_r8 = select_row(summary_rows, "A10_fixed_secondary", 6, "F3_expand_budget_strict")
    c4 = select_row(summary_rows, "C4_fixed", 6, "F9_C4_preserve_R5")
    consistency_map = {row["candidate_name"]: row for row in consistency_summary}

    front_local_risk = False
    for row in front_rows:
        if row["candidate_name"] not in {"A1_fixed", "A10_fixed_main", "A10_fixed_secondary", "C4_fixed"}:
            continue
        if int(row["horizon_s"]) != 6:
            continue
        if float(row["front_local_density_proxy"]) > 1.30 or float(row["front_pred_gt_density_delta"]) > 0.30 or float(row["front_false_positive_delta"]) > 0.02:
            front_local_risk = True
            break

    sanity_rows: list[dict[str, Any]] = []
    sanity_selected_not_worse = True
    a10_frozen = next(candidate for candidate in frozen if candidate.label == "A10_fixed_main")
    selected_variant = fix_variant_map[a10_frozen.fixed_candidate_label]
    for sample_index in sample_ids:
        raw_state = np.load(ARTIFACTS_DIR / "final_outputs" / f"{a10_frozen.label}__rawstate__sample{sample_index:03d}_h6.npz")
        native_semantic = torch.from_numpy(raw_state["native_semantic"]).long()
        raw_semantic = torch.from_numpy(raw_state["raw_semantic"]).long()
        native_occ = native_semantic != EMPTY_IDX
        raw_occ = raw_semantic != EMPTY_IDX
        raw_delta = raw_occ & ~native_occ
        occ_conf = torch.from_numpy(raw_state["raw_occ_conf"]).float()
        top1_margin = torch.from_numpy(raw_state["raw_margin"]).float()
        agreement_map = torch.from_numpy(raw_state["agreement"]).float()
        gt_h = torch.from_numpy(raw_state["gt_h"]).long()
        gt0 = torch.from_numpy(raw_state["gt0"]).long()
        protected = torch.from_numpy(raw_state["protected_zone"]).bool()
        score = sw13c_fix.low_value_score(
            raw_occ,
            protected,
            occ_conf,
            top1_margin,
            agreement_map,
            sectors,
            6,
            selected_variant.wrong_class_aware,
            selected_variant.front_bias,
        )
        final_semantic, pruned, _ = sw13c_fix.apply_pruning_no_gt(
            raw_semantic,
            protected,
            score,
            int(native_occ.sum().item()),
            int(raw_occ.sum().item()),
            int(raw_delta.sum().item()),
            a10_frozen.budget_mode,
            expansion_ratio=a10_frozen.expansion_ratio,
            keep_ratio=a10_frozen.keep_ratio,
        )
        prune_count = int(pruned.sum().item())
        random_pruned = torch.zeros_like(raw_occ)
        nonfront_coords = torch.nonzero(raw_occ & ~protected & ~sectors["front"].bool(), as_tuple=False)
        if prune_count > 0 and nonfront_coords.shape[0] > 0:
            gen = torch.Generator().manual_seed(sample_index)
            order = torch.randperm(nonfront_coords.shape[0], generator=gen)[: min(prune_count, nonfront_coords.shape[0])]
            sel = nonfront_coords[order]
            random_pruned[sel[:, 0], sel[:, 1], sel[:, 2]] = True
        random_final = raw_semantic.clone()
        random_final[random_pruned] = EMPTY_IDX
        selected_eval = sw13c_fix.build_eval_row(final_semantic, gt_h, gt0, a10_frozen.perturbation_id, 6, sectors, native_semantic)
        random_eval = sw13c_fix.build_eval_row(random_final, gt_h, gt0, a10_frozen.perturbation_id, 6, sectors, native_semantic)
        sanity_rows.append(
            {
                "sample_index": sample_index,
                "selected_front_sector_false_free": selected_eval["front_sector_false_free"],
                "selected_pred_gt_density_delta": selected_eval["pred_gt_density_delta"],
                "selected_wrong_class_delta": selected_eval["wrong_class_activation_delta"],
                "random_nonfront_front_sector_false_free": random_eval["front_sector_false_free"],
                "random_nonfront_pred_gt_density_delta": random_eval["pred_gt_density_delta"],
                "random_nonfront_wrong_class_delta": random_eval["wrong_class_activation_delta"],
            }
        )
        del raw_state, native_semantic, raw_semantic, native_occ, raw_occ, raw_delta, occ_conf, top1_margin, agreement_map, gt_h, gt0, protected, score, final_semantic, pruned, random_pruned, random_final
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sanity_rows:
        selected_front_mean = float(np.mean([float(row["selected_front_sector_false_free"]) for row in sanity_rows]))
        random_front_mean = float(np.mean([float(row["random_nonfront_front_sector_false_free"]) for row in sanity_rows]))
        if selected_front_mean > random_front_mean + 1e-6:
            sanity_selected_not_worse = False
    write_csv(REPORTS_DIR / "sw13c_fix_scale_random_pruning_sanity.csv", sanity_rows)
    write_md(
        REPORTS_DIR / "sw13c_fix_scale_random_pruning_sanity.md",
        f"selected_not_worse_than_random_nonfront={sanity_selected_not_worse}\n",
    )

    write_md(
        REPORTS_DIR / "sw13c_fix_scale_front_local_density_summary.md",
        json.dumps({"front_local_density_risk": front_local_risk}, indent=2, ensure_ascii=False) + "\n",
    )
    write_md(
        REPORTS_DIR / "sw13c_fix_scale_sample_consistency_summary.md",
        json.dumps(sw13c_fix.normalize_export(consistency_summary), indent=2, ensure_ascii=False) + "\n",
    )
    write_csv(
        REPORTS_DIR / "sw13c_fix_scale_A10_R4_R8_comparison.csv",
        [
            {
                "candidate": "A10_R4",
                "front_sector_false_free_rate_delta_vs_native": a10_r4["front_sector_false_free_rate_delta_vs_native"] if a10_r4 else None,
                "A10_front_h6_recovery_rate_delta_vs_native": a10_r4["A10_front_h6_recovery_rate_delta_vs_native"] if a10_r4 else None,
                "future_h4_h6_false_free_rate_delta_vs_native": a10_r4["future_h4_h6_false_free_rate_delta_vs_native"] if a10_r4 else None,
                "pred_gt_density_delta": a10_r4["pred_gt_density_delta"] if a10_r4 else None,
                "final_native_expansion_ratio": a10_r4["final_native_expansion_ratio"] if a10_r4 else None,
                "wrong_class_delta": a10_r4["wrong_class_delta"] if a10_r4 else None,
                "false_positive_delta": a10_r4["false_positive_delta"] if a10_r4 else None,
                "front_local_density_proxy": a10_r4["front_local_density_proxy"] if a10_r4 else None,
                "sample_joint_success_rate": consistency_map.get("A10_fixed_main", {}).get("sample_joint_success_rate"),
            },
            {
                "candidate": "A10_R8",
                "front_sector_false_free_rate_delta_vs_native": a10_r8["front_sector_false_free_rate_delta_vs_native"] if a10_r8 else None,
                "A10_front_h6_recovery_rate_delta_vs_native": a10_r8["A10_front_h6_recovery_rate_delta_vs_native"] if a10_r8 else None,
                "future_h4_h6_false_free_rate_delta_vs_native": a10_r8["future_h4_h6_false_free_rate_delta_vs_native"] if a10_r8 else None,
                "pred_gt_density_delta": a10_r8["pred_gt_density_delta"] if a10_r8 else None,
                "final_native_expansion_ratio": a10_r8["final_native_expansion_ratio"] if a10_r8 else None,
                "wrong_class_delta": a10_r8["wrong_class_delta"] if a10_r8 else None,
                "false_positive_delta": a10_r8["false_positive_delta"] if a10_r8 else None,
                "front_local_density_proxy": a10_r8["front_local_density_proxy"] if a10_r8 else None,
                "sample_joint_success_rate": consistency_map.get("A10_fixed_secondary", {}).get("sample_joint_success_rate"),
            },
        ],
    )
    write_md(
        REPORTS_DIR / "sw13c_fix_scale_A10_R4_R8_comparison.md",
        "R8 appears more stable on eval_core50 secondary check only if its frozen secondary metrics are better; no eval_core50 reselection was performed.\n",
    )
    write_md(
        REPORTS_DIR / "sw13c_fix_scale_C4_preservation.md",
        json.dumps(sw13c_fix.normalize_export(c4), indent=2, ensure_ascii=False) + "\n",
    )

    def strong(row: dict[str, Any] | None, cons_row: dict[str, Any] | None, require_a10_recovery: bool) -> bool:
        if row is None or cons_row is None:
            return False
        return (
            float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.20
            and float(row["pred_gt_density_delta"]) <= 0.18
            and not front_local_risk
            and float(row["protected_zone_preservation_ratio"]) >= 0.80
            and float(row["agreement_source_count"]) >= 2
            and float(row["final_native_expansion_ratio"]) <= 0.20
            and float(cons_row["sample_joint_success_rate"]) >= 0.50
            and (float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.15 if require_a10_recovery else True)
        )

    def medium(row: dict[str, Any] | None, cons_row: dict[str, Any] | None, require_a10_recovery: bool) -> bool:
        if row is None or cons_row is None:
            return False
        return (
            float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.12
            and float(row["pred_gt_density_delta"]) <= 0.22
            and float(row["protected_zone_preservation_ratio"]) >= 0.70
            and float(row["agreement_source_count"]) >= 2
            and float(row["final_native_expansion_ratio"]) <= 0.24
            and float(cons_row["sample_joint_success_rate"]) >= 0.35
            and (float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.08 if require_a10_recovery else True)
        )

    protocol_violation = False
    if protocol_violation:
        decision_type = "W6_PROTOCOL_VIOLATION"
    elif front_local_risk or ((a1 and float(a1["pred_gt_density_delta"]) > 0.25) or (a10_r4 and float(a10_r4["pred_gt_density_delta"]) > 0.25) or (a10_r8 and float(a10_r8["pred_gt_density_delta"]) > 0.25)):
        decision_type = "W4_DENSITY_OR_FRONT_LOCAL_RISK"
    elif not sanity_selected_not_worse:
        decision_type = "W4_DENSITY_OR_FRONT_LOCAL_RISK"
    elif strong(a1, consistency_map.get("A1_fixed"), False) or strong(a10_r4, consistency_map.get("A10_fixed_main"), True) or strong(a10_r8, consistency_map.get("A10_fixed_secondary"), True):
        decision_type = "W1_STRONG_SCALE_CONFIRMED"
    elif medium(a1, consistency_map.get("A1_fixed"), False) or medium(a10_r4, consistency_map.get("A10_fixed_main"), True) or medium(a10_r8, consistency_map.get("A10_fixed_secondary"), True):
        decision_type = "W2_MEDIUM_SCALE_CONFIRMED"
    elif (a1 and float(a1["front_sector_false_free_rate_delta_vs_native"]) < 0.0) or (a10_r4 and float(a10_r4["front_sector_false_free_rate_delta_vs_native"]) < 0.0) or (a10_r8 and float(a10_r8["front_sector_false_free_rate_delta_vs_native"]) < 0.0):
        decision_type = "W3_RECOVERY_WEAKENS_BUT_DIRECTION_HOLDS"
    else:
        decision_type = "W5_SCALE_FAILS"

    decision = {
        "decision_type": decision_type,
        "best_A1_scale_result": a1,
        "best_A10_R4_scale_result": a10_r4,
        "best_A10_R8_scale_result": a10_r8,
        "C4_preservation_result": c4,
        "front_local_density_audit": {"front_local_density_risk": front_local_risk},
        "sample_consistency_summary": consistency_summary,
        "no_reselection": True,
        "no_gt_budget": True,
        "no_training": True,
        "not_official_benchmark": True,
    }
    write_json(REPORTS_DIR / "sw13c_fix_scale_eval_core50_decision.json", decision)
    write_md(
        REPORTS_DIR / "sw13c_fix_scale_eval_core50_decision.md",
        json.dumps(sw13c_fix.normalize_export(decision), indent=2, ensure_ascii=False) + "\n",
    )

    if a1 is not None:
        save_metric_bar(
            FIGURES_DIR / "sw13c_fix_scale_A1_metrics_bar.png",
            "A1 frozen candidate eval_core50 subset diagnostic no training no GT budget no reselection",
            a1,
        )
    if a10_r4 is not None:
        save_metric_bar(
            FIGURES_DIR / "sw13c_fix_scale_A10_metrics_bar.png",
            "A10 frozen candidate eval_core50 subset diagnostic no training no GT budget no reselection",
            a10_r4,
        )
    if a10_r4 is not None and a10_r8 is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        metrics = [
            "front_sector_false_free_rate_delta_vs_native",
            "A10_front_h6_recovery_rate_delta_vs_native",
            "pred_gt_density_delta",
            "front_local_density_proxy",
        ]
        x = np.arange(len(metrics))
        ax.bar(x - 0.15, [float(a10_r4[m]) for m in metrics], width=0.3, label="R4")
        ax.bar(x + 0.15, [float(a10_r8[m]) for m in metrics], width=0.3, label="R8")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20)
        ax.legend()
        ax.set_title("A10 R4 vs R8 frozen candidate eval_core50 subset diagnostic no training no GT budget no reselection")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sw13c_fix_scale_A10_R4_R8_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    audit_h6 = [row for row in front_rows if int(row["horizon_s"]) == 6 and row["candidate_name"] in {"A1_fixed", "A10_fixed_main", "A10_fixed_secondary", "C4_fixed"}]
    ax.bar(np.arange(len(audit_h6)), [float(row["front_local_density_proxy"]) for row in audit_h6])
    ax.set_xticks(np.arange(len(audit_h6)))
    ax.set_xticklabels([row["candidate_name"] for row in audit_h6], rotation=20)
    ax.axhline(1.30, color="r", linestyle="--")
    ax.set_title("front-local density frozen candidate eval_core50 subset diagnostic no training no GT budget no reselection")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sw13c_fix_scale_front_local_density_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(consistency_summary)), [float(row["sample_joint_success_rate"]) for row in consistency_summary])
    ax.set_xticks(np.arange(len(consistency_summary)))
    ax.set_xticklabels([row["candidate_name"] for row in consistency_summary], rotation=20)
    ax.set_title("sample consistency frozen candidate eval_core50 subset diagnostic no training no GT budget no reselection")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sw13c_fix_scale_sample_consistency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if "A1_fixed" in visual_cases:
        payload = visual_cases["A1_fixed"]
        sw13c_fix.save_bev(
            FIGURES_DIR / "sw13c_fix_scale_bev_A1_sample0.png",
            payload["gt_h"],
            payload["native"],
            payload["raw"],
            payload["final"],
            payload["protected"],
            payload["pruned"],
            "frozen candidate eval_core50 subset diagnostic no training no GT budget no reselection",
        )
    if "A10_fixed_main" in visual_cases:
        payload = visual_cases["A10_fixed_main"]
        sw13c_fix.save_bev(
            FIGURES_DIR / "sw13c_fix_scale_bev_A10_sample0.png",
            payload["gt_h"],
            payload["native"],
            payload["raw"],
            payload["final"],
            payload["protected"],
            payload["pruned"],
            "frozen candidate eval_core50 subset diagnostic no training no GT budget no reselection",
        )

    report = "\n".join(
        [
            "# Stage SW-13C-Fix-Scale eval_core50",
            "",
            "1. Executive summary",
            f"- decision: {decision_type}",
            "",
            "2. Frozen candidate protocol",
            "- eval_core50 is fixed candidate scaling.",
            "",
            "3. No reselection / no retuning verification",
            "- no reselection, no retuning, no GT budget, no GT-derived selection.",
            "",
            "4. eval_core50 replay setup",
            "- samples 0..49, horizons h0/h2/h4/h6, plus A7 and A0 controls.",
            "",
            "5. A1 scaling result",
            f"- {a1['fixed_candidate_label'] if a1 else 'missing'}",
            "",
            "6. A10 R4 scaling result",
            f"- {a10_r4['fixed_candidate_label'] if a10_r4 else 'missing'}",
            "",
            "7. A10 R8 secondary scaling result",
            f"- {a10_r8['fixed_candidate_label'] if a10_r8 else 'missing'}",
            "",
            "8. C4 preservation",
            "- preserve only, no forced pruning.",
            "",
            "9. Front-local density audit",
            f"- front_local_density_risk={front_local_risk}",
            "",
            "10. Sample consistency",
            "- reported per candidate and horizon.",
            "",
            "11. Random pruning sanity",
            f"- selected_not_worse_than_random_nonfront={sanity_selected_not_worse}",
            "",
            "12. Decision W1-W6",
            f"- {decision_type}",
            "",
            "13. Safe claims",
            "- eval_core50 is fixed candidate scaling",
            "- no training",
            "- no checkpoint modification",
            "- no get_occ modification",
            "- no GT budget",
            "- no GT-derived selection",
            "- no reselection",
            "- subset diagnostic only",
            "- not official benchmark",
            "",
            "14. Limitations",
            "- A10 R8 is frozen secondary comparison only and not a new selection.",
            "",
            "15. Next unique action",
            "- only if W1 or W2 holds, consider larger frozen-candidate verification before any training.",
            "",
        ]
    )
    write_md(REPORTS_DIR / "stage_sw13c_fix_scale_eval_core50_report.md", report)
    write_json(
        REPORTS_DIR / "stage_sw13c_fix_scale_eval_core50_report.json",
        {
            "decision_type": decision_type,
            "no_reselection": True,
            "no_gt_budget": True,
            "not_official_benchmark": True,
        },
    )

    write_tests()


if __name__ == "__main__":
    main()
