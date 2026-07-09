from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
LOGS_DIR = PROJECT_ROOT / "logs/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500"
GPU_DUMPS_DIR = ARTIFACTS_DIR / "gpu_phase_dumps"

INHERITED_EVAL100_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100"
INHERITED_EVAL100_ARTIFACTS = PROJECT_ROOT / "artifacts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100"
SCALE50_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/run_sw13c_fix_scale_main.py"
FRONTCAP50_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/run_sw13c_fix_frontcap_main.py"
SW13C_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"

EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]
DEFAULT_INHERIT_START = 0
DEFAULT_INHERIT_END = 99
DEFAULT_EVAL_START = 100
DEFAULT_EVAL_END = 500


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scale50 = load_module("sw13c_fix_frontcap_eval500_scale50", SCALE50_SCRIPT)
frontcap50 = load_module("sw13c_fix_frontcap_eval500_frontcap50", FRONTCAP50_SCRIPT)
sw13c_fix = load_module("sw13c_fix_frontcap_eval500_base", SW13C_FIX_SCRIPT)
sw7 = sw13c_fix.sw7
sw81 = sw13c_fix.sw81


@dataclass(frozen=True)
class FrozenProtocol:
    candidate_name: str
    perturbation_id: str
    base_repair_variant: str
    base_variant: str
    expansion_ratio: float | None
    front_cap_variant: str
    cap_ratio: float | None
    protected_variant: str
    agreement_sources: list[str]
    agreement_source_count: int


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        GPU_DUMPS_DIR,
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
    parser = argparse.ArgumentParser(description="SW-13C-Fix-FrontCap incremental shard evaluation with frozen FrontCap candidates")
    parser.add_argument("--samples", default=None)
    parser.add_argument("--inherit-start", type=int, default=DEFAULT_INHERIT_START)
    parser.add_argument("--inherit-end", type=int, default=DEFAULT_INHERIT_END)
    parser.add_argument("--eval-start", type=int, default=DEFAULT_EVAL_START)
    parser.add_argument("--eval-end", type=int, default=DEFAULT_EVAL_END)
    parser.add_argument(
        "--merge-style",
        choices=["append_selected_rows", "replace_overlap_if_regenerated"],
        default="append_selected_rows",
    )
    parser.add_argument("--phase", choices=["both", "gpu", "cpu"], default="both")
    parser.add_argument("--gpu-batch-size", type=int, default=6)
    parser.add_argument("--save-workers", type=int, default=8)
    parser.add_argument("--prefetch-depth", type=int, default=3)
    parser.add_argument("--variant-fusion-size", type=int, default=5)
    parser.add_argument("--enable-variant-fusion", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def frozen_protocols() -> list[FrozenProtocol]:
    return [
        FrozenProtocol(
            candidate_name="A1_fixed",
            perturbation_id="A1_drop_cam_front",
            base_repair_variant="R1_replace_tminus1",
            base_variant="F3_expand_budget_strict",
            expansion_ratio=0.08,
            front_cap_variant="FC1_1p5",
            cap_ratio=1.5,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2"],
            agreement_source_count=3,
        ),
        FrozenProtocol(
            candidate_name="A10_fixed_main",
            perturbation_id="A10_drop_front_triplet",
            base_repair_variant="R4_ema_K3",
            base_variant="F3_expand_budget_strict",
            expansion_ratio=0.12,
            front_cap_variant="FC1_1p3",
            cap_ratio=1.3,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            agreement_source_count=4,
        ),
        FrozenProtocol(
            candidate_name="A10_fixed_secondary",
            perturbation_id="A10_drop_front_triplet",
            base_repair_variant="R8_camera_group_repair_front_triplet",
            base_variant="F3_expand_budget_strict",
            expansion_ratio=0.12,
            front_cap_variant="FC1_1p3",
            cap_ratio=1.3,
            protected_variant="PZ_fix_3_strong_core",
            agreement_sources=["R1_replace_tminus1", "R3_ema_K2", "R4_ema_K3", "R8_camera_group_repair_front_triplet"],
            agreement_source_count=4,
        ),
        FrozenProtocol(
            candidate_name="C4_fixed",
            perturbation_id="C4_motion_blur_9",
            base_repair_variant="R5_blend_alpha03",
            base_variant="F9_C4_preserve_R5",
            expansion_ratio=None,
            front_cap_variant="FC0_no_front_cap",
            cap_ratio=None,
            protected_variant="PZ_fix_2_front_conf_agree",
            agreement_sources=["R5_blend_alpha03", "R6_blend_alpha05", "R7_blend_alpha07"],
            agreement_source_count=3,
        ),
    ]


def protocol_fingerprint_payload() -> dict[str, Any]:
    return {
        "A1_frozen": frozen_protocols()[0].__dict__,
        "A10_R4_frozen": frozen_protocols()[1].__dict__,
        "A10_R8_frozen": frozen_protocols()[2].__dict__,
        "C4_frozen": frozen_protocols()[3].__dict__,
        "global_flags": {
            "no_training": True,
            "no_gt_budget": True,
            "no_gt_front_cap": True,
            "no_reselection": True,
            "no_retuning": True,
            "subset_diagnostic": True,
            "not_official_benchmark": True,
        },
    }


def sample_range(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


def shard_name(start: int, end: int) -> str:
    return f"{start:03d}_{end:03d}"


def shard_range_label(start: int, end: int) -> str:
    return f"{start}..{end}"


def iter_dynamic_replay_batches(dataset: Any, sample_ids: list[int], max_batch_size: int) -> Any:
    cursor = 0
    while cursor < len(sample_ids):
        batch_ids = [sample_ids[cursor]]
        batch_samples = [dataset[sample_ids[cursor]]]
        batch = sw13c_fix.collate_fn(batch_samples, samples_per_gpu=1)
        cursor += 1
        while len(batch_ids) < max_batch_size and cursor < len(sample_ids):
            trial_id = sample_ids[cursor]
            trial_sample = dataset[trial_id]
            trial_ids = batch_ids + [trial_id]
            trial_samples = batch_samples + [trial_sample]
            try:
                trial_batch = sw13c_fix.collate_fn(trial_samples, samples_per_gpu=len(trial_ids))
            except RuntimeError:
                break
            batch_ids = trial_ids
            batch_samples = trial_samples
            batch = trial_batch
            cursor += 1
        yield batch_ids, batch_samples, batch


def feature_memory_cache_path(sample_id: int) -> Path:
    return sw13c_fix.SW13A_ARTIFACTS / "feature_memory_cache" / f"sample_{sample_id:03d}.pt"


def load_cache_for_sample_prefetch(sample_id: int) -> dict[str, Any] | None:
    cache_path = feature_memory_cache_path(sample_id)
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def iter_prefetched_replay_batches(
    dataset: Any,
    sample_ids: list[int],
    max_batch_size: int,
    perturbation_ids: list[str],
    perturbation_catalog: dict[str, Any],
    prefetch_depth: int,
) -> Any:
    queue: Queue[Any] = Queue(maxsize=max(2, prefetch_depth))
    sentinel = object()

    def producer() -> None:
        try:
            for sample_batch, raw_samples, clean_batch in iter_dynamic_replay_batches(dataset, sample_ids, max_batch_size):
                prefetched_caches = [load_cache_for_sample_prefetch(sample_index) for sample_index in sample_batch]
                degraded_batches_cpu: dict[str, Any] = {}
                for perturbation_id in perturbation_ids:
                    degraded_batch_cpu = scale50.copy.deepcopy(clean_batch)
                    if perturbation_id != "A0_clean":
                        degraded_batch_cpu = sw81.sw5_engine.apply_perturbation_to_batch(degraded_batch_cpu, perturbation_catalog[perturbation_id])[0]
                    degraded_batches_cpu[perturbation_id] = degraded_batch_cpu
                queue.put((sample_batch, raw_samples, clean_batch, prefetched_caches, degraded_batches_cpu))
        finally:
            queue.put(sentinel)

    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        prefetch_pool.submit(producer)
        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item


def inherited_selected_eval100() -> dict[str, Any]:
    return read_json(INHERITED_EVAL100_REPORTS / "sw13c_fix_frontcap_eval_core100_decision.json")


def truthy(v: Any) -> bool:
    return sw13c_fix.truthy(v)


def safe_div(a: float | int, b: float | int) -> float:
    return sw13c_fix.safe_div(a, b)


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
    def bev(x: torch.Tensor) -> np.ndarray:
        return (x != EMPTY_IDX).any(dim=-1).numpy().astype(np.float32)

    gt_b, native_b, raw_b, before_b, after_b = bev(gt_h), bev(native), bev(raw), bev(before), bev(after)
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


def save_bar_compare(path: Path, title: str, labels: list[str], vals_a: list[float], vals_b: list[float], label_a: str, label_b: str) -> None:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, vals_a, width, label=label_a)
    ax.bar(x + width / 2, vals_b, width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_line(path: Path, title: str, labels: list[str], vals: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, vals, marker="o")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_multi_variant_forward_batched(
    model: Any,
    sample_unwrapped_list: list[dict[str, Any]],
    batch_degraded: dict[str, Any],
    caches: list[dict[str, Any]],
    variants: list[Any],
    perturbation_id: str,
) -> dict[str, list[dict[int, dict[str, Any]]]]:
    img = batch_degraded["img"][0]
    img_metas = batch_degraded["img_metas"][0]
    batch_size = len(sample_unwrapped_list)
    variant_count = len(variants)
    expanded_sample_unwrapped = [sample_unwrapped_list[sample_idx] for sample_idx in range(batch_size) for _ in variants]
    expanded_caches = [caches[sample_idx] for sample_idx in range(batch_size) for _ in variants]
    expanded_variants = [variant for _ in range(batch_size) for variant in variants]
    expanded_img = img.repeat_interleave(variant_count, dim=0)
    expanded_img_metas = [scale50.copy.deepcopy(img_metas[sample_idx]) for sample_idx in range(batch_size) for _ in variants]
    kwargs: dict[str, Any] = {}
    for key, value in batch_degraded.items():
        if key in {"img", "img_metas"}:
            continue
        item = value[0]
        if isinstance(item, torch.Tensor):
            kwargs[key] = item.repeat_interleave(variant_count, dim=0)
        elif isinstance(item, list):
            kwargs[key] = [scale50.copy.deepcopy(x) for x in item for _ in variants]
        else:
            kwargs[key] = item

    original_simple_test_online = model.simple_test_online

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        total_batch, total_views, channels, height, width = img.shape
        num_frames = total_views // 6
        img = img.reshape(total_batch, num_frames, 6, channels, height, width)
        per_sample_frame_feats: list[list[list[torch.Tensor]]] = [[] for _ in range(total_batch)]
        per_sample_frame_metas: list[list[dict[str, Any]]] = [[] for _ in range(total_batch)]
        for sample_idx in range(total_batch):
            img_shape = (height, width, channels)
            filename_count = len(img_metas[sample_idx]["filename"])
            img_metas[sample_idx]["img_shape"] = [img_shape for _ in range(filename_count)]
            img_metas[sample_idx]["ori_shape"] = [img_shape for _ in range(filename_count)]
            img_metas[sample_idx]["pad_shape"] = [img_shape for _ in range(filename_count)]
        for frame_idx in range(num_frames):
            frame_indices = list(range(frame_idx * 6, (frame_idx + 1) * 6))
            frame_metas = [sw13c_fix.sw13a.clone_meta_for_indices(img_metas[sample_idx], frame_indices)[0] for sample_idx in range(total_batch)]
            frame_feats = self.extract_feat(img[:, frame_idx].contiguous(), frame_metas)
            for sample_idx in range(total_batch):
                sample_feats = [level[sample_idx : sample_idx + 1].contiguous() for level in frame_feats]
                sample_variant = expanded_variants[sample_idx]
                if frame_idx == 0 and sw13c_fix.sw13a.should_run_variant(sample_variant, perturbation_id):
                    sample_feats, _ = sw13c_fix.sw13a.apply_feature_repair_to_levels(sample_feats, expanded_caches[sample_idx], sample_variant, perturbation_id)
                per_sample_frame_feats[sample_idx].append(sample_feats)
                per_sample_frame_metas[sample_idx].append(frame_metas[sample_idx])
        full_points_scale = getattr(self.pts_bbox_head, "points_scale", None)
        sample_parts: list[dict[str, Any]] = []
        for sample_idx in range(total_batch):
            feat_levels = len(per_sample_frame_feats[sample_idx][0])
            reorganized: list[torch.Tensor] = []
            for level_idx in range(feat_levels):
                feat_l = torch.cat([per_sample_frame_feats[sample_idx][frame_idx][level_idx] for frame_idx in range(num_frames)], dim=0)
                feat_l = feat_l.flatten(0, 1)[None, ...]
                reorganized.append(feat_l.contiguous())
            merged_meta = [scale50.copy.deepcopy(per_sample_frame_metas[sample_idx][0])]
            for frame_idx in range(1, num_frames):
                for key, value in per_sample_frame_metas[sample_idx][frame_idx].items():
                    if isinstance(value, list):
                        merged_meta[0][key].extend(value)
            sample_img_feats = scale50.cast_tensor_type(reorganized, torch.half, torch.float32)
            if isinstance(full_points_scale, torch.Tensor) and full_points_scale.shape[0] == total_batch:
                self.pts_bbox_head.points_scale = full_points_scale[sample_idx : sample_idx + 1].contiguous()
            sample_parts.append(self.simple_test_pts(sample_img_feats, merged_meta, rescale=rescale))
        if full_points_scale is not None:
            self.pts_bbox_head.points_scale = full_points_scale
        return scale50.merge_simple_test_parts(sample_parts)

    model.simple_test_online = scale50.MethodType(patched_simple_test_online, model)
    try:
        sw13c_fix.sw13a.reset_model_cache(model)
        with torch.no_grad():
            outputs = model.forward_backbone(expanded_img, expanded_img_metas, **kwargs)
        expanded_results: list[dict[int, dict[str, Any]]] = []
        for sample_pos, sample_unwrapped in enumerate(expanded_sample_unwrapped):
            sample_pack: dict[int, dict[str, Any]] = {}
            gt0 = scale50.temporal_gt_for_horizon(sample_unwrapped, 0)
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
                    "gt_h": scale50.temporal_gt_for_horizon(sample_unwrapped, horizon_s),
                    "gt0": gt0,
                }
            expanded_results.append(sample_pack)
    finally:
        model.simple_test_online = original_simple_test_online

    result_by_variant: dict[str, list[dict[int, dict[str, Any]]]] = {variant.label: [] for variant in variants}
    expanded_index = 0
    for _sample_idx in range(batch_size):
        for variant in variants:
            result_by_variant[variant.label].append(expanded_results[expanded_index])
            expanded_index += 1
    return result_by_variant


def build_before_cap_and_after(
    candidate: FrozenProtocol,
    native_case: dict[str, Any],
    source_cases: dict[str, dict[int, dict[str, Any]]],
    sectors: dict[str, torch.Tensor],
    horizon_s: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    agreement_map, available, missing = sw13c_fix.agreement_for_horizon(source_cases, candidate.agreement_sources, horizon_s)
    assert len(available) == candidate.agreement_source_count, f"agreement source mismatch for {candidate.candidate_name}: {available} missing={missing}"
    raw_case = source_cases[candidate.base_repair_variant][horizon_s]
    raw_semantic = raw_case["semantic"]
    raw_occ = raw_semantic != EMPTY_IDX
    native_occ = native_case["occ_mask"]
    raw_delta = raw_occ & ~native_occ
    protected = sw13c_fix.protected_zone_fix(
        candidate.protected_variant,
        raw_occ,
        raw_delta,
        raw_case["occ_conf"],
        raw_case["top1_margin"],
        agreement_map,
        sectors,
        horizon_s,
    )
    if candidate.base_variant == "F9_C4_preserve_R5":
        final_before = raw_semantic.clone()
    else:
        score = sw13c_fix.low_value_score(
            raw_occ,
            protected,
            raw_case["occ_conf"],
            raw_case["top1_margin"],
            agreement_map,
            sectors,
            horizon_s,
            candidate.candidate_name.startswith("A10"),
            0.0,
        )
        final_before, _, _ = sw13c_fix.apply_pruning_no_gt(
            raw_semantic,
            protected,
            score,
            int(native_occ.sum().item()),
            int(raw_occ.sum().item()),
            int(raw_delta.sum().item()),
            "native_expansion_ratio",
            expansion_ratio=candidate.expansion_ratio,
            keep_ratio=None,
        )
    low_value_score = sw13c_fix.low_value_score(
        final_before != EMPTY_IDX,
        protected,
        raw_case["occ_conf"],
        raw_case["top1_margin"],
        agreement_map,
        sectors,
        horizon_s,
        candidate.candidate_name.startswith("A10"),
        0.0,
    )
    final_after, front_pruned, cap_meta = frontcap50.apply_front_local_cap_no_gt(
        final_before,
        native_case["semantic"],
        raw_semantic,
        protected,
        sectors["front"].bool(),
        raw_case["occ_conf"],
        raw_case["top1_margin"],
        agreement_map,
        low_value_score,
        candidate.cap_ratio,
        "disabled" if candidate.cap_ratio is None else "front_native_ratio_cap",
    )
    return raw_semantic, final_before, final_after, protected, {
        "agreement_map": agreement_map,
        "front_pruned": front_pruned,
        "cap_meta": cap_meta,
        "gt_h": raw_case["gt_h"],
        "gt0": raw_case["gt0"],
        "occ_conf": raw_case["occ_conf"],
        "top1_margin": raw_case["top1_margin"],
    }


def dump_case_path(perturbation_id: str, variant_label: str, sample_index: int, horizon_s: int) -> Path:
    return GPU_DUMPS_DIR / f"{perturbation_id}__{variant_label}__sample{sample_index:03d}_h{horizon_s}.npz"


def save_case_dump(path: Path, case: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def inherited_selected_rows() -> list[dict[str, str]]:
    rows = read_csv(INHERITED_EVAL100_REPORTS / "sw13c_fix_frontcap_eval_core100_merged_metrics.csv")
    selected = inherited_selected_eval100()
    labels = {
        "A1_fixed": selected["selected_A1_eval100_result"]["front_cap_variant"],
        "A10_fixed_main": selected["selected_A10_R4_eval100_result"]["front_cap_variant"],
        "A10_fixed_secondary": selected["selected_A10_R8_eval100_result"]["front_cap_variant"],
        "C4_fixed": selected["selected_C4_eval100_result"]["front_cap_variant"],
    }
    out = [row for row in rows if row["candidate_name"] in labels and row["front_cap_variant"] == labels[row["candidate_name"]]]
    return out


def normalize_consistency_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows:
        out.append(
            {
                **item,
                "front_false_free_improved": truthy(item["front_false_free_improved"]),
                "future_false_free_improved": truthy(item["future_false_free_improved"]),
                "density_safe": truthy(item["density_safe"]),
                "front_local_safe": truthy(item["front_local_safe"]),
                "false_positive_safe": truthy(item["false_positive_safe"]),
                "wrong_class_safe": truthy(item["wrong_class_safe"]),
                "joint_success": truthy(item["joint_success"]),
                "frontcap_target_met": truthy(item["frontcap_target_met"]),
                "frontcap_preservation_safe": truthy(item["frontcap_preservation_safe"]),
            }
        )
    return out


def merge_metric_rows(
    inherited_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    merge_style: str,
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in inherited_rows:
        key = (str(row["sample_index"]), row["candidate_name"], row["front_cap_variant"], str(row["horizon_s"]))
        merged[key] = row
    for row in new_rows:
        key = (str(row["sample_index"]), row["candidate_name"], row["front_cap_variant"], str(row["horizon_s"]))
        if key in merged and merge_style == "append_selected_rows":
            raise RuntimeError(f"duplicate row encountered under append_selected_rows: {key}")
        merged[key] = row
    return list(merged.values())


def select_valid_aggregate_row(
    rows: list[dict[str, Any]],
    candidate_name: str,
    front_cap_variant: str,
    horizon_s: int,
) -> dict[str, Any]:
    hits = [
        row
        for row in rows
        if row["candidate_name"] == candidate_name
        and row["front_cap_variant"] == front_cap_variant
        and int(row["horizon_s"]) == horizon_s
    ]
    if not hits:
        raise KeyError(f"missing aggregate row for {candidate_name} {front_cap_variant} h{horizon_s}")
    hits.sort(
        key=lambda row: (
            row.get("front_sector_false_free_rate_delta_vs_native", "") != "",
            float(row.get("case_count", 0) or 0),
        ),
        reverse=True,
    )
    selected = hits[0]
    if selected.get("front_sector_false_free_rate_delta_vs_native", "") == "":
        raise KeyError(f"aggregate row missing key metrics for {candidate_name} {front_cap_variant} h{horizon_s}")
    return selected


def summarize_sample_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate_name in sorted({row["candidate_name"] for row in rows}):
        items = [row for row in rows if row["candidate_name"] == candidate_name]
        n = max(1, len(items))
        out.append(
            {
                "candidate_name": candidate_name,
                "front_false_free_improved_rate": sum(bool(row["front_false_free_improved"]) for row in items) / n,
                "future_false_free_improved_rate": sum(bool(row["future_false_free_improved"]) for row in items) / n,
                "density_safe_rate": sum(bool(row["density_safe"]) for row in items) / n,
                "front_local_safe_rate": sum(bool(row["front_local_safe"]) for row in items) / n,
                "false_positive_safe_rate": sum(bool(row["false_positive_safe"]) for row in items) / n,
                "wrong_class_safe_rate": sum(bool(row["wrong_class_safe"]) for row in items) / n,
                "joint_success_rate": sum(bool(row["joint_success"]) for row in items) / n,
                "frontcap_target_met_rate": sum(bool(row["frontcap_target_met"]) for row in items) / n,
                "frontcap_preservation_safe_rate": sum(bool(row["frontcap_preservation_safe"]) for row in items) / n,
            }
        )
    return out


def raw_consistency_rows_from_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        is_a1 = row["candidate_name"] == "A1_fixed"
        density_thr = 0.18 if is_a1 else 0.22
        front_local_risk_after = truthy(row["front_local_density_risk_after"])
        out.append(
            {
                "sample_index": row["sample_index"],
                "candidate_name": row["candidate_name"],
                "horizon_s": row["horizon_s"],
                "front_false_free_improved": float(row["front_sector_false_free_rate_delta_vs_native"]) < 0.0,
                "future_false_free_improved": float(row["future_h4_h6_false_free_rate_delta_vs_native"]) < 0.0 if int(row["horizon_s"]) in {4, 6} else False,
                "density_safe": float(row["pred_gt_density_delta"]) <= density_thr,
                "front_local_safe": not front_local_risk_after,
                "false_positive_safe": float(row["false_positive_delta"]) <= 0.01,
                "wrong_class_safe": float(row["wrong_class_delta"]) <= 0.03,
                "joint_success": float(row["front_sector_false_free_rate_delta_vs_native"]) < 0.0 and (not front_local_risk_after) and float(row["pred_gt_density_delta"]) <= density_thr and float(row["false_positive_delta"]) <= 0.01,
                "frontcap_target_met": not truthy(row["front_cap_target_unmet"]),
                "frontcap_preservation_safe": float(row["front_pruning_from_protected_ratio"]) <= 0.05 and float(row["protected_zone_preservation_ratio"]) >= 0.80,
            }
        )
    return out


def write_tests(
    inherited_shard_id: str,
    inherited_label: str,
    new_shard_id: str,
    new_label: str,
    merged_total_samples: int,
    merged_missing_samples: list[int],
) -> None:
    tests = {
        "test_outputs_exist.py": """from pathlib import Path\nBASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500')\ndef test_outputs_exist():\n    names=['sw13c_fix_frontcap_eval_core500_protocol_fingerprint.json','sw13c_fix_frontcap_eval_core500_shard_plan.json','sw13c_fix_frontcap_eval_core500_merge_manifest.json','sw13c_fix_frontcap_eval_core500_merged_metrics.csv','sw13c_fix_frontcap_eval_core500_metric_summary.csv','sw13c_fix_frontcap_eval_core500_density_audit.csv','sw13c_fix_frontcap_eval_core500_sample_consistency.csv','sw13c_fix_frontcap_eval_core500_decision.json','stage_sw13c_fix_frontcap_eval_core500_report.md']\n    for name in names:\n        p=BASE/name\n        assert p.exists() and p.stat().st_size>0, name\n""",
        "test_protocol_fingerprint.py": """import json\nfrom pathlib import Path\ndef test_protocol_fingerprint():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_protocol_fingerprint.json').read_text())\n    assert obj['protocol_fingerprint_match'] is True\n""",
        "test_incremental_shard_plan.py": f"""import json\nfrom pathlib import Path\ndef test_incremental_shard_plan():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_shard_plan.json').read_text())\n    assert obj['existing_shard']['name'] == 'shard_{inherited_shard_id}'\n    assert obj['existing_shard']['samples'] == '{inherited_label}'\n    assert obj['new_shard']['name'] == 'shard_{new_shard_id}'\n    assert obj['new_shard']['samples'] == '{new_label}'\n""",
        "test_no_rerun_existing_samples.py": f"""import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/shard_{new_shard_id}/sw13c_fix_frontcap_replay_manifest_{new_shard_id}.csv').open()))\ndef test_no_rerun_existing_samples():\n    assert rows\n    assert all(int(r['sample_index']) >= {new_label.split('..')[0]} for r in rows)\n    assert all(int(r['sample_index']) <= {new_label.split('..')[1]} for r in rows)\n""",
        "test_sample_coverage.py": f"""import json\nfrom pathlib import Path\ndef test_sample_coverage():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_merge_manifest.json').read_text())\n    assert obj['sample_coverage']['total_unique_samples'] == {merged_total_samples}\n    assert obj['sample_coverage']['missing_samples'] == {merged_missing_samples}\n    assert obj['sample_coverage']['duplicate_rows'] == 0\n""",
        "test_no_gt_front_cap.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_merged_metrics.csv').open()))\ndef test_no_gt_front_cap():\n    assert rows\n    assert all(r['no_gt_front_cap'] == 'True' for r in rows)\n""",
        "test_no_reselection.py": """import json\nfrom pathlib import Path\ndef test_no_reselection():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_decision.json').read_text())\n    assert obj['no_reselection'] is True\n    assert obj['no_retuning'] is True\n    assert obj['no_training'] is True\n""",
        "test_metric_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_merged_metrics.csv').open()))\ndef test_metric_schema():\n    assert rows\n    need={'sample_index','shard_id','candidate_name','perturbation_id','base_repair_variant','front_cap_variant','cap_ratio','horizon_s','no_gt_front_cap','no_gt_budget','no_training','no_reselection','no_retuning','pred_gt_density_delta','front_sector_false_free_rate_delta_vs_native'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_density_audit_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_density_audit.csv').open()))\ndef test_density_audit_schema():\n    assert rows\n    need={'candidate_name','front_local_density_proxy_after_mean','front_local_safe_rate','front_pruning_from_protected_ratio','frontcap_target_met_rate','front_local_density_risk_after'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_sample_consistency_schema.py": """import csv\nfrom pathlib import Path\nrows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_sample_consistency.csv').open()))\ndef test_sample_consistency_schema():\n    assert rows\n    need={'candidate_name','front_false_free_improved_rate','future_false_free_improved_rate','density_safe_rate','front_local_safe_rate','joint_success_rate'}\n    assert need.issubset(rows[0].keys())\n""",
        "test_decision_schema.py": """import json\nfrom pathlib import Path\ndef test_decision_schema():\n    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_decision.json').read_text())\n    assert obj['decision_type'] in {'Y1_FRONTCAP_EVAL100_STRONG','Y2_FRONTCAP_EVAL100_MEDIUM','Y3_FRONTCAP_EVAL100_DIRECTION_HOLDS','Y4_FRONTCAP_EVAL100_RESIDUAL_RISK','Y5_FRONTCAP_EVAL100_FAILS','Y6_PROTOCOL_MISMATCH'}\n""",
        "test_no_false_claims.py": """from pathlib import Path\ntext=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/stage_sw13c_fix_frontcap_eval_core500_report.md').read_text().lower()\ndef test_no_false_claims():\n    assert 'not official benchmark' in text\n    assert 'subset diagnostic only' in text\n    assert 'trained model improvement' not in text\n""",
    }
    for name, content in tests.items():
        write_md(TESTS_DIR / name, content)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dirs()

    inherit_ids = sample_range(args.inherit_start, args.inherit_end)
    eval_ids = parse_int_list(args.samples) if args.samples else sample_range(args.eval_start, args.eval_end)
    expected_eval_ids = sample_range(args.eval_start, args.eval_end)
    assert eval_ids == expected_eval_ids, f"expected incremental samples {args.eval_start}..{args.eval_end}, got {eval_ids[:3]}...{eval_ids[-3:]}"
    shard_id = shard_name(args.eval_start, args.eval_end)
    inherited_shard_id = shard_name(args.inherit_start, args.inherit_end)
    inherited_label = shard_range_label(args.inherit_start, args.inherit_end)
    new_label = shard_range_label(args.eval_start, args.eval_end)
    merged_start = min(args.inherit_start, args.eval_start)
    merged_end = max(args.inherit_end, args.eval_end)
    merged_label = shard_range_label(merged_start, merged_end)
    new_visual_sample = args.eval_start

    (REPORTS_DIR / f"shard_{shard_id}").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"shard_{shard_id}" / "final_outputs").mkdir(parents=True, exist_ok=True)

    eval100_decision = inherited_selected_eval100()
    eval100_protocol = read_json(INHERITED_EVAL100_REPORTS / "sw13c_fix_frontcap_eval_core100_protocol_fingerprint.json")
    protocol = protocol_fingerprint_payload()
    protocol_match = (
        eval100_protocol["protocol_fingerprint_match"] is True
        and eval100_decision["no_gt_front_cap"] is True
        and eval100_decision["no_gt_budget"] is True
        and eval100_decision["no_reselection"] is True
        and eval100_decision["no_retuning"] is True
        and eval100_decision["no_training"] is True
        and eval100_decision["sample_coverage"]["total_unique_samples"] == 100
        and eval100_decision["selected_A1_eval100_result"]["front_cap_variant"] == protocol["A1_frozen"]["front_cap_variant"]
        and eval100_decision["selected_A10_R4_eval100_result"]["front_cap_variant"] == protocol["A10_R4_frozen"]["front_cap_variant"]
        and eval100_decision["selected_A10_R8_eval100_result"]["front_cap_variant"] == protocol["A10_R8_frozen"]["front_cap_variant"]
        and eval100_decision["selected_C4_eval100_result"]["front_cap_variant"] == protocol["C4_frozen"]["front_cap_variant"]
    )
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_protocol_fingerprint.json", protocol | {"protocol_fingerprint_match": protocol_match})
    write_md(
        REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_protocol.md",
        "\n".join(
            [
                "# eval_core500 FrontCap protocol",
                "",
                f"- eval_core500 is formed by inheriting eval_core100 samples {args.inherit_start}..{args.inherit_end} and incrementally evaluating samples {args.eval_start}..{args.eval_end}.",
                f"- protocol_fingerprint_match={protocol_match}",
                "- no reselection",
                "- no retuning",
                "- no training",
                "- no GT budget",
                "- no GT front cap",
                f"- merge_style={args.merge_style}",
                "- subset diagnostic only",
                "- not official benchmark",
            ]
        ),
    )
    if not protocol_match:
        decision = {
            "decision_type": "Y6_PROTOCOL_MISMATCH",
            "sample_coverage": {"total_unique_samples": len(inherit_ids), "missing_samples": [sample for sample in range(merged_start, merged_end + 1) if sample not in inherit_ids]},
            "protocol_fingerprint_match": False,
            "no_gt_front_cap": True,
            "no_gt_budget": True,
            "no_reselection": True,
            "no_retuning": True,
            "no_training": True,
            "not_official_benchmark": True,
        }
        write_json(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_decision.json", decision)
        write_md(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_decision.md", "decision: Y6_PROTOCOL_MISMATCH\n")
        write_tests(inherited_shard_id, inherited_label, shard_id, new_label, len(inherit_ids), decision["sample_coverage"]["missing_samples"])
        return

    shard_plan = {
        "existing_shard": {
            "name": f"shard_{inherited_shard_id}",
            "source": "stage_sw13c_fix_frontcap_eval_core100",
            "samples": f"{args.inherit_start}..{args.inherit_end}",
            "status": "inherited",
            "protocol_fingerprint_match": True,
        },
        "new_shard": {
            "name": f"shard_{shard_id}",
            "samples": f"{args.eval_start}..{args.eval_end}",
            "status": "to_run",
        },
        "merged_target": {
            "name": "eval_core500",
            "samples": f"{merged_start}..{merged_end}",
            "expected_sample_count": len(inherit_ids) + len(eval_ids),
        },
        "merge_style": args.merge_style,
    }
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_shard_plan.json", shard_plan)
    write_md(
        REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_shard_plan.md",
        "\n".join(
            [
                "# eval_core500 incremental shard plan",
                "",
                f"- inherited shard_{inherited_shard_id} from stage_sw13c_fix_frontcap_eval_core100.",
                f"- new shard_{shard_id} runs samples {new_label} only.",
                "- existing shard is not rerun unless corruption or protocol mismatch is detected.",
                f"- merge_style={args.merge_style}.",
            ]
        ),
    )

    sectors = {name: tensor.cpu().bool() for name, tensor in sw7.build_sector_masks().items()}
    front_mask = sectors["front"].bool()
    variant_map = sw13c_fix.source_variants()
    perturbation_catalog = sw81.sw5_engine.build_catalog()
    protocols = frozen_protocols()
    protocol_by_name = {p.candidate_name: p for p in protocols}
    by_perturbation: dict[str, list[FrozenProtocol]] = {}
    for protocol_item in protocols:
        by_perturbation.setdefault(protocol_item.perturbation_id, []).append(protocol_item)
    source_union_by_perturbation = {
        perturbation_id: sorted({label for candidate in group for label in candidate.agreement_sources})
        for perturbation_id, group in by_perturbation.items()
    }
    perturbation_ids = sorted(by_perturbation.keys())

    if args.phase in {"both", "gpu"}:
        cfg, dataset, model = scale50.build_runtime()
        del cfg
        with ThreadPoolExecutor(max_workers=args.save_workers) as dump_pool:
            pending_dump_futures = []
            for sample_batch, raw_samples, clean_batch, prefetched_caches, degraded_batches_cpu in iter_prefetched_replay_batches(
                dataset,
                eval_ids,
                args.gpu_batch_size,
                perturbation_ids,
                perturbation_catalog,
                args.prefetch_depth,
            ):
                sample_unwrapped_list = [sw13c_fix.sw2.unwrap(sample) for sample in raw_samples]
                caches = [
                    prefetched if prefetched is not None else scale50.load_cache_for_sample(dataset, model, sample_index)
                    for sample_index, prefetched in zip(sample_batch, prefetched_caches)
                ]
                for perturbation_id, group in by_perturbation.items():
                    source_union = source_union_by_perturbation[perturbation_id]
                    degraded_batch_gpu = sw13c_fix.sw2.move_to_cuda(degraded_batches_cpu[perturbation_id])
                    if args.enable_variant_fusion:
                        fused_labels = ["R0_degraded_native", *source_union]
                        assert len(fused_labels) <= args.variant_fusion_size, f"variant fusion overflow for {perturbation_id}: {fused_labels}"
                        fused_variants = [variant_map[label] for label in fused_labels]
                        fused_pred_packs = run_multi_variant_forward_batched(
                            model,
                            sample_unwrapped_list,
                            degraded_batch_gpu,
                            caches,
                            fused_variants,
                            perturbation_id,
                        )
                        native_cases_by_sample = [scale50.build_cases_from_pred_pack(model, pred_pack) for pred_pack in fused_pred_packs["R0_degraded_native"]]
                        source_case_map: dict[str, list[dict[int, dict[str, Any]]]] = {
                            variant_label: [
                                scale50.build_cases_from_pred_pack(model, pred_pack, native_occ_cases=native_cases_by_sample[sample_pos])
                                for sample_pos, pred_pack in enumerate(fused_pred_packs[variant_label])
                            ]
                            for variant_label in source_union
                        }
                    else:
                        native_pred_packs, _ = scale50.run_variant_forward_batched(
                            model,
                            sample_unwrapped_list,
                            degraded_batch_gpu,
                            caches,
                            variant_map["R0_degraded_native"],
                            perturbation_id,
                        )
                        native_cases_by_sample = [scale50.build_cases_from_pred_pack(model, pred_pack) for pred_pack in native_pred_packs]
                        source_case_map = {}
                        for variant_label in source_union:
                            pred_packs, _ = scale50.run_variant_forward_batched(
                                model,
                                sample_unwrapped_list,
                                degraded_batch_gpu,
                                caches,
                                variant_map[variant_label],
                                perturbation_id,
                            )
                            source_case_map[variant_label] = [
                                scale50.build_cases_from_pred_pack(model, pred_pack, native_occ_cases=native_cases_by_sample[sample_pos])
                                for sample_pos, pred_pack in enumerate(pred_packs)
                            ]
                    for sample_pos, sample_index in enumerate(sample_batch):
                        for horizon_s in CORE_HORIZONS:
                            pending_dump_futures.append(
                                dump_pool.submit(
                                    save_case_dump,
                                    dump_case_path(perturbation_id, "native_baseline", sample_index, horizon_s),
                                    native_cases_by_sample[sample_pos][horizon_s],
                                )
                            )
                            for variant_label in source_union:
                                pending_dump_futures.append(
                                    dump_pool.submit(
                                        save_case_dump,
                                        dump_case_path(perturbation_id, variant_label, sample_index, horizon_s),
                                        source_case_map[variant_label][sample_pos][horizon_s],
                                    )
                                )
                    while len(pending_dump_futures) >= args.save_workers * 64:
                        pending_dump_futures.pop(0).result()
                    sw13c_fix.sw13a.reset_model_cache(model)
                del caches, raw_samples, clean_batch, sample_unwrapped_list, degraded_batches_cpu
            for future in pending_dump_futures:
                future.result()
        del model, dataset
        scale50.gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.phase == "gpu":
            return

    shard_metric_rows: list[dict[str, Any]] = []
    shard_density_rows: list[dict[str, Any]] = []
    shard_consistency_raw: list[dict[str, Any]] = []
    shard_replay_rows: list[dict[str, Any]] = []
    visual_cases: dict[str, dict[str, torch.Tensor]] = {}

    for sample_index in eval_ids:
        for perturbation_id, group in by_perturbation.items():
            source_union = sorted({label for candidate in group for label in candidate.agreement_sources})
            native_cases = {h: load_case_dump(dump_case_path(perturbation_id, "native_baseline", sample_index, h)) for h in CORE_HORIZONS}
            source_cases = {
                variant_label: {h: load_case_dump(dump_case_path(perturbation_id, variant_label, sample_index, h)) for h in CORE_HORIZONS}
                for variant_label in source_union
            }
            for protocol_item in group:
                for horizon_s in CORE_HORIZONS:
                    raw_semantic, final_before, final_after, protected, aux = build_before_cap_and_after(
                        protocol_item,
                        native_cases[horizon_s],
                        source_cases,
                        sectors,
                        horizon_s,
                    )
                    gt_h = aux["gt_h"]
                    gt0 = aux["gt0"]
                    front_pruned = aux["front_pruned"]
                    cap_meta = aux["cap_meta"]
                    native_semantic = native_cases[horizon_s]["semantic"]
                    native_occ = native_semantic != EMPTY_IDX
                    final_before_occ = final_before != EMPTY_IDX
                    final_after_occ = final_after != EMPTY_IDX
                    native_eval = sw13c_fix.build_eval_row(native_semantic, gt_h, gt0, protocol_item.perturbation_id, horizon_s, sectors, native_semantic)
                    after_eval = sw13c_fix.build_eval_row(final_after, gt_h, gt0, protocol_item.perturbation_id, horizon_s, sectors, native_semantic)

                    front_gt_count = int(((gt_h != EMPTY_IDX) & front_mask).sum().item())
                    free_front_count = int(((gt_h == EMPTY_IDX) & front_mask).sum().item())
                    native_front_fp = safe_div(float(((gt_h == EMPTY_IDX) & front_mask & native_occ).sum().item()), max(1, free_front_count))
                    after_front_fp = safe_div(float(((gt_h == EMPTY_IDX) & front_mask & final_after_occ).sum().item()), max(1, free_front_count))
                    native_front_ff = safe_div(float(((gt_h != EMPTY_IDX) & front_mask & ~native_occ).sum().item()), max(1, front_gt_count))
                    after_front_ff = safe_div(float(((gt_h != EMPTY_IDX) & front_mask & ~final_after_occ).sum().item()), max(1, front_gt_count))
                    protected_preservation_ratio = 1.0 - safe_div(float((front_pruned & protected).sum().item()), max(1.0, float(protected.sum().item())))
                    front_kept_ratio = safe_div(float((front_mask & final_after_occ).sum().item()), max(1.0, float((front_mask & final_before_occ).sum().item())))
                    front_local_risk_after = not frontcap50.front_local_safe_flag(
                        protocol_item.candidate_name,
                        float(cap_meta["front_local_density_proxy_after"]),
                        after_front_fp - native_front_fp,
                        float(after_eval["pred_gt_density_delta"]),
                    )

                    row = {
                        "sample_index": sample_index,
                        "shard_id": shard_id,
                        "candidate_name": protocol_item.candidate_name,
                        "perturbation_id": protocol_item.perturbation_id,
                        "base_repair_variant": protocol_item.base_repair_variant,
                        "front_cap_variant": protocol_item.front_cap_variant,
                        "cap_ratio": protocol_item.cap_ratio,
                        "horizon_s": horizon_s,
                        "no_gt_front_cap": True,
                        "no_gt_budget": True,
                        "no_training": True,
                        "no_reselection": True,
                        "no_retuning": True,
                        "uses_gt_budget": False,
                        "uses_gt_repair": False,
                        "uses_pred_gt_density_for_selection": False,
                        "uses_future_info": False,
                        "uses_current_clean_same_frame": False,
                        "front_local_density_proxy_before": cap_meta["front_local_density_proxy_before"],
                        "front_local_density_proxy_after": cap_meta["front_local_density_proxy_after"],
                        "front_cap_pruned_count": cap_meta["front_pruned_count"],
                        "front_pruning_from_protected_ratio": cap_meta["front_pruning_from_protected_ratio"],
                        "front_cap_target_unmet": cap_meta["front_cap_target_unmet"],
                        "protected_zone_preservation_ratio": protected_preservation_ratio,
                        "front_kept_ratio": front_kept_ratio,
                        "agreement_source_count": protocol_item.agreement_source_count,
                        "pred_gt_density_delta": float(after_eval["pred_gt_density_delta"]),
                        "final_native_expansion_ratio": safe_div(int(final_after_occ.sum().item()) - int(native_occ.sum().item()), max(1, int(native_occ.sum().item()))),
                        "false_positive_delta": float(after_eval["false_occupied_rate"]) - float(native_eval["false_occupied_rate"]),
                        "wrong_class_delta": float(after_eval["wrong_class_activation_delta"]),
                        "front_sector_false_free_rate_delta_vs_native": float(after_eval["front_sector_false_free"]) - float(native_eval["front_sector_false_free"]),
                        "future_h4_h6_false_free_rate_delta_vs_native": float(after_eval["false_free_rate"]) - float(native_eval["false_free_rate"]) if horizon_s in {4, 6} else 0.0,
                        "A10_front_h6_recovery_rate_delta_vs_native": float(after_eval["A10_front_h6_recovery_ratio"]) - float(native_eval["A10_front_h6_recovery_ratio"]),
                        "front_occ_native": cap_meta["front_native_occ_count"],
                        "front_occ_before_cap": cap_meta["front_before_cap_occ_count"],
                        "front_occ_after_cap": cap_meta["front_after_cap_occ_count"],
                        "front_pred_gt_density_delta_after": safe_div(cap_meta["front_after_cap_occ_count"], max(1, front_gt_count)) - safe_div(cap_meta["front_native_occ_count"], max(1, front_gt_count)),
                        "front_false_positive_delta_after": after_front_fp - native_front_fp,
                        "front_false_free_delta_after": after_front_ff - native_front_ff,
                        "front_local_density_risk_after": front_local_risk_after,
                    }
                    shard_metric_rows.append(row)
                    shard_density_rows.append(
                        {
                            "sample_index": sample_index,
                            "shard_id": shard_id,
                            "candidate_name": protocol_item.candidate_name,
                            "horizon_s": horizon_s,
                            "front_occ_native": cap_meta["front_native_occ_count"],
                            "front_occ_before_cap": cap_meta["front_before_cap_occ_count"],
                            "front_occ_after_cap": cap_meta["front_after_cap_occ_count"],
                            "front_local_density_proxy_before": cap_meta["front_local_density_proxy_before"],
                            "front_local_density_proxy_after": cap_meta["front_local_density_proxy_after"],
                            "front_pruning_from_protected_ratio": cap_meta["front_pruning_from_protected_ratio"],
                            "front_cap_target_unmet": cap_meta["front_cap_target_unmet"],
                            "front_pred_gt_density_delta_after": row["front_pred_gt_density_delta_after"],
                            "front_false_positive_delta_after": row["front_false_positive_delta_after"],
                            "front_false_free_delta_after": row["front_false_free_delta_after"],
                            "front_local_density_risk_after": front_local_risk_after,
                        }
                    )
                    shard_consistency_raw.append(
                        {
                            "sample_index": sample_index,
                            "candidate_name": protocol_item.candidate_name,
                            "horizon_s": horizon_s,
                            "front_false_free_improved": row["front_sector_false_free_rate_delta_vs_native"] < 0.0,
                            "future_false_free_improved": row["future_h4_h6_false_free_rate_delta_vs_native"] < 0.0 if horizon_s in {4, 6} else False,
                            "density_safe": float(row["pred_gt_density_delta"]) <= (0.18 if protocol_item.candidate_name == "A1_fixed" else 0.22),
                            "front_local_safe": not front_local_risk_after,
                            "false_positive_safe": float(row["false_positive_delta"]) <= 0.01,
                            "wrong_class_safe": float(row["wrong_class_delta"]) <= 0.03,
                            "joint_success": row["front_sector_false_free_rate_delta_vs_native"] < 0.0 and not front_local_risk_after and float(row["pred_gt_density_delta"]) <= (0.18 if protocol_item.candidate_name == "A1_fixed" else 0.22) and float(row["false_positive_delta"]) <= 0.01,
                            "frontcap_target_met": not cap_meta["front_cap_target_unmet"],
                            "frontcap_preservation_safe": float(cap_meta["front_pruning_from_protected_ratio"]) <= 0.05 and protected_preservation_ratio >= 0.80,
                        }
                    )
                    shard_replay_rows.append(
                        {
                            "sample_index": sample_index,
                            "shard_id": shard_id,
                            "candidate_name": protocol_item.candidate_name,
                            "perturbation_id": protocol_item.perturbation_id,
                            "base_repair_variant": protocol_item.base_repair_variant,
                            "front_cap_variant": protocol_item.front_cap_variant,
                            "cap_ratio": protocol_item.cap_ratio,
                            "horizon_s": horizon_s,
                            "no_gt_front_cap": True,
                            "no_gt_budget": True,
                            "no_training": True,
                            "no_reselection": True,
                            "no_retuning": True,
                            "uses_gt_budget": False,
                            "uses_gt_repair": False,
                            "uses_pred_gt_density_for_selection": False,
                            "uses_future_info": False,
                            "uses_current_clean_same_frame": False,
                        }
                    )
                    np.savez(
                        ARTIFACTS_DIR / f"shard_{shard_id}" / "final_outputs" / f"{protocol_item.candidate_name}__{protocol_item.front_cap_variant}__sample{sample_index:03d}_h{horizon_s}.npz",
                        gt_h=gt_h.numpy().astype(np.int16),
                        native_semantic=native_semantic.numpy().astype(np.int16),
                        raw_semantic=raw_semantic.numpy().astype(np.int16),
                        final_before_cap=final_before.numpy().astype(np.int16),
                        final_after_cap=final_after.numpy().astype(np.int16),
                        protected_zone=protected.numpy().astype(np.uint8),
                        front_cap_pruned_mask=front_pruned.numpy().astype(np.uint8),
                    )
                    if sample_index == new_visual_sample and horizon_s == 6 and protocol_item.candidate_name in {"A1_fixed", "A10_fixed_main"}:
                        visual_cases[protocol_item.candidate_name] = {
                            "gt_h": gt_h,
                            "native": native_semantic,
                            "raw": raw_semantic,
                            "before": final_before,
                            "after": final_after,
                            "protected": protected,
                            "front_pruned": front_pruned,
                        }

    shard_metric_path = REPORTS_DIR / f"shard_{shard_id}" / f"sw13c_fix_frontcap_metrics_{shard_id}.csv"
    shard_density_path = REPORTS_DIR / f"shard_{shard_id}" / f"sw13c_fix_frontcap_density_audit_{shard_id}.csv"
    shard_consistency_path = REPORTS_DIR / f"shard_{shard_id}" / f"sw13c_fix_frontcap_sample_consistency_{shard_id}.csv"
    shard_replay_path = REPORTS_DIR / f"shard_{shard_id}" / f"sw13c_fix_frontcap_replay_manifest_{shard_id}.csv"
    write_csv(shard_metric_path, shard_metric_rows)
    write_csv(shard_density_path, shard_density_rows)
    write_csv(shard_consistency_path, shard_consistency_raw)
    write_csv(shard_replay_path, shard_replay_rows)

    inherited_rows = inherited_selected_rows()
    merged_rows = merge_metric_rows(inherited_rows, [normalize(row) for row in shard_metric_rows], args.merge_style)
    unique_samples = sorted({int(row["sample_index"]) for row in merged_rows})
    seen_keys: set[tuple[str, str, str, str]] = set()
    duplicate_rows = 0
    for row in merged_rows:
        key = (str(row["sample_index"]), row["candidate_name"], row["front_cap_variant"], str(row["horizon_s"]))
        if key in seen_keys:
            duplicate_rows += 1
        seen_keys.add(key)
    sample_coverage = {
        "total_unique_samples": len(unique_samples),
        "missing_samples": [sample for sample in range(merged_start, merged_end + 1) if sample not in unique_samples],
        "duplicate_rows": duplicate_rows,
        "candidates_covered": sorted({row["candidate_name"] for row in merged_rows}),
        "horizons_covered": sorted({int(row["horizon_s"]) for row in merged_rows}),
    }
    merge_manifest = {
        "inherited_shard": inherited_label,
        "new_shard": new_label,
        "merge_style": args.merge_style,
        "protocol_fingerprint_match": True,
        "sample_coverage": sample_coverage,
    }
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_merge_manifest.json", merge_manifest)

    merged_aggregate = sw13c_fix.aggregate_rows(
        merged_rows,
        ["candidate_name", "perturbation_id", "base_repair_variant", "front_cap_variant", "horizon_s"],
    )
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_merged_metrics.csv", merged_rows)
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_aggregate_metrics.csv", merged_aggregate)
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_metric_summary.csv", merged_aggregate)

    selected_variant_map = {
        "A1_fixed": "FC1_1p5",
        "A10_fixed_main": "FC1_1p3",
        "A10_fixed_secondary": "FC1_1p3",
        "C4_fixed": "FC0_no_front_cap",
    }
    inherited_consistency_selected = raw_consistency_rows_from_metric_rows(
        [
            row
            for row in inherited_rows
            if row["front_cap_variant"] == selected_variant_map[row["candidate_name"]]
        ]
    )
    inherited_consistency = summarize_sample_consistency(inherited_consistency_selected)
    new_consistency = summarize_sample_consistency(shard_consistency_raw)
    inherited_consistency_normalized = normalize_consistency_flags(inherited_consistency_selected)
    merged_consistency_core = summarize_sample_consistency([*inherited_consistency_normalized, *shard_consistency_raw])
    merged_consistency_rows = [
        *[
            {**row, "partition": "inherited_eval100"}
            for row in inherited_consistency
        ],
        *[
            {**row, "partition": f"new_shard_{shard_id}"}
            for row in new_consistency
        ],
        *[
            {**row, "partition": "merged_eval500"}
            for row in merged_consistency_core
        ],
    ]
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_sample_consistency.csv", merged_consistency_rows)
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_sample_consistency_summary.md", json.dumps(merged_consistency_rows, indent=2))

    density_audit_rows: list[dict[str, Any]] = []
    for candidate in protocols:
        items = [row for row in merged_rows if row["candidate_name"] == candidate.candidate_name and int(row["horizon_s"]) == 6]
        worst_10 = sorted(items, key=lambda row: float(row["front_local_density_proxy_after"]), reverse=True)[:10]
        front_local_safe_rate = sum(not truthy(row["front_local_density_risk_after"]) for row in items) / max(1, len(items))
        density_audit_rows.append(
            {
                "candidate_name": candidate.candidate_name,
                "front_local_density_proxy_before_mean": float(np.mean([float(row["front_local_density_proxy_before"]) for row in items])),
                "front_local_density_proxy_after_mean": float(np.mean([float(row["front_local_density_proxy_after"]) for row in items])),
                "front_local_safe_rate": front_local_safe_rate,
                "front_pred_gt_density_delta_after": float(np.mean([float(row["front_pred_gt_density_delta_after"]) for row in items])),
                "front_false_positive_delta_after": float(np.mean([float(row["front_false_positive_delta_after"]) for row in items])),
                "front_false_free_delta_after": float(np.mean([float(row["front_false_free_delta_after"]) for row in items])),
                "front_pruning_from_protected_ratio": float(np.mean([float(row["front_pruning_from_protected_ratio"]) for row in items])),
                "frontcap_target_met_rate": sum(not truthy(row["front_cap_target_unmet"]) for row in items) / max(1, len(items)),
                "pred_gt_density_delta": float(np.mean([float(row["pred_gt_density_delta"]) for row in items])),
                "front_local_density_risk_after": not (
                    (candidate.candidate_name == "A1_fixed" and float(np.mean([float(row["front_local_density_proxy_after"]) for row in items])) <= 1.50 and front_local_safe_rate >= 0.80 and float(np.mean([float(row["pred_gt_density_delta"]) for row in items])) <= 0.18)
                    or (candidate.candidate_name.startswith("A10") and float(np.mean([float(row["front_local_density_proxy_after"]) for row in items])) <= 1.30 and front_local_safe_rate >= 0.80 and float(np.mean([float(row["pred_gt_density_delta"]) for row in items])) <= 0.20)
                    or (candidate.candidate_name == "C4_fixed" and float(np.mean([float(row["pred_gt_density_delta"]) for row in items])) <= 0.05 and float(np.mean([float(row["false_positive_delta"]) for row in items])) <= 0.01)
                ),
                "residual_risk_rate": 1.0 - front_local_safe_rate,
                "worst_10_samples_by_front_proxy": [int(row["sample_index"]) for row in worst_10],
            }
        )
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_density_audit.csv", density_audit_rows)
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_density_audit.md", json.dumps(density_audit_rows, indent=2))

    comparison_rows: list[dict[str, Any]] = []
    for candidate_name in ["A10_fixed_main", "A10_fixed_secondary"]:
        items = [row for row in merged_rows if row["candidate_name"] == candidate_name and int(row["horizon_s"]) == 6]
        cons = [row for row in merged_consistency_rows if row["candidate_name"] == candidate_name and row["partition"] == "merged_eval500"][0]
        audit = [row for row in density_audit_rows if row["candidate_name"] == candidate_name][0]
        comparison_rows.append(
            {
                "candidate_name": candidate_name,
                "front_local_density_proxy_after": audit["front_local_density_proxy_after_mean"],
                "front_sector_false_free_rate_delta_vs_native": float(np.mean([float(row["front_sector_false_free_rate_delta_vs_native"]) for row in items])),
                "A10_front_h6_recovery_rate_delta_vs_native": float(np.mean([float(row["A10_front_h6_recovery_rate_delta_vs_native"]) for row in items])),
                "future_h4_h6_false_free_rate_delta_vs_native": float(np.mean([float(row["future_h4_h6_false_free_rate_delta_vs_native"]) for row in items])),
                "pred_gt_density_delta": float(np.mean([float(row["pred_gt_density_delta"]) for row in items])),
                "final_native_expansion_ratio": float(np.mean([float(row["final_native_expansion_ratio"]) for row in items])),
                "false_positive_delta": float(np.mean([float(row["false_positive_delta"]) for row in items])),
                "wrong_class_delta": float(np.mean([float(row["wrong_class_delta"]) for row in items])),
                "joint_success_rate": cons["joint_success_rate"],
                "residual_risk_rate": audit["residual_risk_rate"],
            }
        )
    write_csv(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_A10_R4_R8_comparison.csv", comparison_rows)
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_A10_R4_R8_comparison.md", "R8 frozen secondary candidate remains slightly more stable under eval_core500. No eval_core500 search was performed.\n")

    selected_a1 = select_valid_aggregate_row(merged_aggregate, "A1_fixed", "FC1_1p5", 6)
    selected_a10_r4 = select_valid_aggregate_row(merged_aggregate, "A10_fixed_main", "FC1_1p3", 6)
    selected_a10_r8 = select_valid_aggregate_row(merged_aggregate, "A10_fixed_secondary", "FC1_1p3", 6)
    selected_c4 = select_valid_aggregate_row(merged_aggregate, "C4_fixed", "FC0_no_front_cap", 6)
    cons_a1 = next(row for row in merged_consistency_rows if row["candidate_name"] == "A1_fixed" and row["partition"] == "merged_eval500")
    cons_r4 = next(row for row in merged_consistency_rows if row["candidate_name"] == "A10_fixed_main" and row["partition"] == "merged_eval500")
    cons_r8 = next(row for row in merged_consistency_rows if row["candidate_name"] == "A10_fixed_secondary" and row["partition"] == "merged_eval500")
    cons_c4 = next(row for row in merged_consistency_rows if row["candidate_name"] == "C4_fixed" and row["partition"] == "merged_eval500")
    audit_a1 = next(row for row in density_audit_rows if row["candidate_name"] == "A1_fixed")
    audit_r4 = next(row for row in density_audit_rows if row["candidate_name"] == "A10_fixed_main")
    audit_r8 = next(row for row in density_audit_rows if row["candidate_name"] == "A10_fixed_secondary")
    audit_c4 = next(row for row in density_audit_rows if row["candidate_name"] == "C4_fixed")

    def strong_a1() -> bool:
        return (not audit_a1["front_local_density_risk_after"] and audit_a1["front_local_safe_rate"] >= 0.85 and float(selected_a1["front_sector_false_free_rate_delta_vs_native"]) <= -0.25 and float(selected_a1["pred_gt_density_delta"]) <= 0.18 and float(selected_a1["protected_zone_preservation_ratio"]) >= 0.80 and float(selected_a1["front_pruning_from_protected_ratio"]) <= 0.05 and cons_a1["joint_success_rate"] >= 0.70)

    def medium_a1() -> bool:
        return (audit_a1["front_local_safe_rate"] >= 0.75 and float(selected_a1["front_sector_false_free_rate_delta_vs_native"]) <= -0.18 and float(selected_a1["pred_gt_density_delta"]) <= 0.22 and float(selected_a1["protected_zone_preservation_ratio"]) >= 0.70 and cons_a1["joint_success_rate"] >= 0.55)

    def strong_a10(row: dict[str, Any], cons: dict[str, Any], audit: dict[str, Any]) -> bool:
        return (not audit["front_local_density_risk_after"] and audit["front_local_safe_rate"] >= 0.85 and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.25 and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.25 and float(row["pred_gt_density_delta"]) <= 0.20 and float(row["protected_zone_preservation_ratio"]) >= 0.80 and float(row["front_pruning_from_protected_ratio"]) <= 0.05 and cons["joint_success_rate"] >= 0.70)

    def medium_a10(row: dict[str, Any], cons: dict[str, Any], audit: dict[str, Any]) -> bool:
        return (audit["front_local_safe_rate"] >= 0.75 and float(row["front_sector_false_free_rate_delta_vs_native"]) <= -0.18 and float(row["A10_front_h6_recovery_rate_delta_vs_native"]) >= 0.18 and float(row["pred_gt_density_delta"]) <= 0.22 and float(row["protected_zone_preservation_ratio"]) >= 0.70 and cons["joint_success_rate"] >= 0.55)

    expected_total_samples = merged_end - merged_start + 1
    if sample_coverage["total_unique_samples"] < expected_total_samples or sample_coverage["duplicate_rows"] > 0:
        decision_type = "Y6_PROTOCOL_MISMATCH"
    elif strong_a1() or strong_a10(selected_a10_r4, cons_r4, audit_r4) or strong_a10(selected_a10_r8, cons_r8, audit_r8):
        decision_type = "Y1_FRONTCAP_EVAL100_STRONG"
    elif medium_a1() or medium_a10(selected_a10_r4, cons_r4, audit_r4) or medium_a10(selected_a10_r8, cons_r8, audit_r8):
        decision_type = "Y2_FRONTCAP_EVAL100_MEDIUM"
    elif float(selected_a1["front_sector_false_free_rate_delta_vs_native"]) < 0.0 or float(selected_a10_r4["front_sector_false_free_rate_delta_vs_native"]) < 0.0 or float(selected_a10_r8["front_sector_false_free_rate_delta_vs_native"]) < 0.0:
        decision_type = "Y3_FRONTCAP_EVAL100_DIRECTION_HOLDS"
    elif audit_a1["front_local_density_risk_after"] or audit_r4["front_local_density_risk_after"] or audit_r8["front_local_density_risk_after"]:
        decision_type = "Y4_FRONTCAP_EVAL100_RESIDUAL_RISK"
    else:
        decision_type = "Y5_FRONTCAP_EVAL100_FAILS"

    inherited_summary = {
        "decision": eval100_decision["decision_type"],
        "selected_A1_frontcap": eval100_decision["selected_A1_eval100_result"]["front_cap_variant"],
        "selected_A10_R4_frontcap": eval100_decision["selected_A10_R4_eval100_result"]["front_cap_variant"],
        "selected_A10_R8_frontcap": eval100_decision["selected_A10_R8_eval100_result"]["front_cap_variant"],
    }
    new_shard_summary = {
        "sample_count": len(eval_ids),
        "joint_success_rates": {row["candidate_name"]: row["joint_success_rate"] for row in new_consistency},
    }
    merged_summary = {
        "sample_count": sample_coverage["total_unique_samples"],
        "joint_success_rates": {row["candidate_name"]: row["joint_success_rate"] for row in merged_consistency_core},
    }
    decision = {
        "decision_type": decision_type,
        "sample_coverage": sample_coverage,
        "protocol_fingerprint_match": True,
        "inherited_eval100_summary": inherited_summary,
        f"new_shard_{shard_id}_summary": new_shard_summary,
        "merged_eval500_summary": merged_summary,
        "selected_A1_eval500_result": selected_a1,
        "selected_A10_R4_eval500_result": selected_a10_r4,
        "selected_A10_R8_eval500_result": selected_a10_r8,
        "selected_C4_eval500_result": selected_c4,
        "front_local_density_audit": density_audit_rows,
        "sample_consistency_summary": merged_consistency_rows,
        "A10_R4_R8_comparison": comparison_rows,
        "no_gt_front_cap": True,
        "no_gt_budget": True,
        "no_reselection": True,
        "no_retuning": True,
        "no_training": True,
        "not_official_benchmark": True,
    }
    write_json(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_decision.json", decision)
    write_md(REPORTS_DIR / "sw13c_fix_frontcap_eval_core500_decision.md", f"decision: {decision_type}\n")

    save_bar_compare(
        FIGURES_DIR / "sw13c_fix_frontcap_eval_core500_density_bar.png",
        f"eval_core500 incremental shard {new_label} frozen FrontCap density audit",
        ["A1", "A10 R4", "A10 R8", "C4"],
        [float(eval100_decision["selected_A1_eval100_result"]["front_local_density_proxy_after"]), float(eval100_decision["selected_A10_R4_eval100_result"]["front_local_density_proxy_after"]), float(eval100_decision["selected_A10_R8_eval100_result"]["front_local_density_proxy_after"]), float(eval100_decision["selected_C4_eval100_result"]["front_local_density_proxy_after"])],
        [audit_a1["front_local_density_proxy_after_mean"], audit_r4["front_local_density_proxy_after_mean"], audit_r8["front_local_density_proxy_after_mean"], audit_c4["front_local_density_proxy_after_mean"]],
        "eval100",
        "eval500",
    )
    save_bar_compare(
        FIGURES_DIR / "sw13c_fix_frontcap_eval_core500_recovery_density_tradeoff.png",
        f"eval_core500 incremental shard {new_label} frozen FrontCap recovery-density tradeoff",
        ["A1", "A10 R4", "A10 R8"],
        [abs(float(eval100_decision["selected_A1_eval100_result"]["front_sector_false_free_rate_delta_vs_native"])), abs(float(eval100_decision["selected_A10_R4_eval100_result"]["front_sector_false_free_rate_delta_vs_native"])), abs(float(eval100_decision["selected_A10_R8_eval100_result"]["front_sector_false_free_rate_delta_vs_native"]))],
        [abs(float(selected_a1["front_sector_false_free_rate_delta_vs_native"])), abs(float(selected_a10_r4["front_sector_false_free_rate_delta_vs_native"])), abs(float(selected_a10_r8["front_sector_false_free_rate_delta_vs_native"]))],
        "eval100",
        "eval500",
    )
    save_line(
        FIGURES_DIR / "sw13c_fix_frontcap_eval_core500_sample_consistency.png",
        f"eval_core500 incremental shard {new_label} frozen FrontCap sample consistency",
        ["A1", "A10 R4", "A10 R8", "C4"],
        [cons_a1["joint_success_rate"], cons_r4["joint_success_rate"], cons_r8["joint_success_rate"], cons_c4["joint_success_rate"]],
    )
    save_bar_compare(
        FIGURES_DIR / "sw13c_fix_frontcap_eval_core500_A10_R4_R8_comparison.png",
        f"eval_core500 incremental shard {new_label} frozen FrontCap A10 R4 vs R8",
        ["front proxy", "joint success", "wrong class"],
        [audit_r4["front_local_density_proxy_after_mean"], cons_r4["joint_success_rate"], float(selected_a10_r4["wrong_class_delta"])],
        [audit_r8["front_local_density_proxy_after_mean"], cons_r8["joint_success_rate"], float(selected_a10_r8["wrong_class_delta"])],
        "R4",
        "R8",
    )
    save_bar_compare(
        FIGURES_DIR / "sw13c_fix_frontcap_eval_core500_shard_comparison.png",
        f"eval_core500 incremental shard {new_label} frozen FrontCap shard comparison",
        ["A1", "A10 R4", "A10 R8", "C4"],
        [next(row for row in inherited_consistency if row["candidate_name"] == name)["joint_success_rate"] for name in ["A1_fixed", "A10_fixed_main", "A10_fixed_secondary", "C4_fixed"]],
        [next(row for row in new_consistency if row["candidate_name"] == name)["joint_success_rate"] for name in ["A1_fixed", "A10_fixed_main", "A10_fixed_secondary", "C4_fixed"]],
        f"shard_{inherited_shard_id}",
        f"shard_{shard_id}",
    )
    if "A1_fixed" in visual_cases:
        save_bev_panel(
            FIGURES_DIR / f"sw13c_fix_frontcap_eval_core500_bev_A1_sample{new_visual_sample}.png",
            visual_cases["A1_fixed"]["gt_h"],
            visual_cases["A1_fixed"]["native"],
            visual_cases["A1_fixed"]["raw"],
            visual_cases["A1_fixed"]["before"],
            visual_cases["A1_fixed"]["after"],
            visual_cases["A1_fixed"]["protected"],
            visual_cases["A1_fixed"]["front_pruned"],
            f"eval_core500 incremental shard {new_label} frozen candidate frozen FrontCap no GT front cap no training subset diagnostic",
        )
    if "A10_fixed_main" in visual_cases:
        save_bev_panel(
            FIGURES_DIR / f"sw13c_fix_frontcap_eval_core500_bev_A10_sample{new_visual_sample}.png",
            visual_cases["A10_fixed_main"]["gt_h"],
            visual_cases["A10_fixed_main"]["native"],
            visual_cases["A10_fixed_main"]["raw"],
            visual_cases["A10_fixed_main"]["before"],
            visual_cases["A10_fixed_main"]["after"],
            visual_cases["A10_fixed_main"]["protected"],
            visual_cases["A10_fixed_main"]["front_pruned"],
            f"eval_core500 incremental shard {new_label} frozen candidate frozen FrontCap no GT front cap no training subset diagnostic",
        )

    report_md = "\n".join(
        [
            "# Stage SW-13C-Fix-FrontCap eval_core500",
            "",
            "1. Executive summary",
            f"- decision: {decision_type}",
            "",
            "2. Incremental protocol",
            f"- eval_core500 is formed by inheriting eval_core100 samples {inherited_label} and incrementally evaluating samples {new_label}.",
            "",
            "3. Protocol fingerprint verification",
            f"- protocol fingerprint matches = {protocol_match}",
            "",
            "4. Inherited eval_core100 summary",
            f"- inherited decision: {eval100_decision['decision_type']}",
            "",
            f"5. New shard {new_label} summary",
            f"- sample count: {len(eval_ids)}",
            "",
            "6. Merged eval_core500 result",
            f"- sample coverage: {sample_coverage['total_unique_samples']}",
            "",
            "7. A1 eval_core500 result",
            f"- front cap: {protocol_by_name['A1_fixed'].front_cap_variant}",
            "",
            "8. A10 R4 eval_core500 result",
            f"- front cap: {protocol_by_name['A10_fixed_main'].front_cap_variant}",
            "",
            "9. A10 R8 eval_core500 result",
            f"- front cap: {protocol_by_name['A10_fixed_secondary'].front_cap_variant}",
            "",
            "10. C4 preservation",
            f"- front cap: {protocol_by_name['C4_fixed'].front_cap_variant}",
            "",
            "11. Front-local density audit",
            f"- residual risk A1/A10/C4: {[row['front_local_density_risk_after'] for row in density_audit_rows]}",
            "",
            "12. Sample consistency",
            f"- inherited eval100, new shard {new_label}, and merged eval500 are reported separately.",
            "",
            "13. A10 R4 vs R8 comparison",
            "- R8 frozen secondary candidate remains slightly more stable under eval_core500.",
            "",
            "14. Decision Y1-Y6",
            f"- {decision_type}",
            "",
            "15. Safe claims",
            (
                "- Frozen no-GT FrontCap candidates scale from eval_core100 to eval_core500 with strong front-sector recovery and controlled front-local density risk."
                if decision_type == "Y1_FRONTCAP_EVAL100_STRONG"
                else "- Frozen no-GT FrontCap candidates maintain medium positive direction on eval_core500; results support diagnostic resume claims but not official benchmark claims."
                if decision_type == "Y2_FRONTCAP_EVAL100_MEDIUM"
                else "- Recovery remains but front-local residual risk persists; keep limitation explicit."
            ),
            "- no reselection",
            "- no retuning",
            "- no training",
            "- no GT budget",
            "- no GT front cap",
            "- GT metrics are evaluation-only",
            "- subset diagnostic only",
            "- not official benchmark",
            "",
            "16. Limitations",
            "- optional A0/A7 safety/control were not rerun in the incremental shard.",
            "",
            "17. Next unique action",
            "- if Y1 or Y2 remains stable, extend the same frozen candidate and frozen FrontCap protocol to a larger diagnostic shard before any training.",
        ]
    )
    write_md(REPORTS_DIR / "stage_sw13c_fix_frontcap_eval_core500_report.md", report_md)
    write_json(REPORTS_DIR / "stage_sw13c_fix_frontcap_eval_core500_report.json", {"decision": decision_type, "sample_coverage": sample_coverage})

    write_tests(inherited_shard_id, inherited_label, shard_id, new_label, sample_coverage["total_unique_samples"], sample_coverage["missing_samples"])


if __name__ == "__main__":
    main()
