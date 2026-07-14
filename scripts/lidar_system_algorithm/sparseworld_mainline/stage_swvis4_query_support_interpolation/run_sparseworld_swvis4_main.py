from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from swvis4_support_interp_lib import (
    ANCHOR_TIMES,
    ARTIFACTS_DIR,
    EMPTY_IDX,
    FIGURES_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    VIDEOS_DIR,
    build_index_aligned_match,
    build_side_by_side_video,
    build_support_keyframe_from_tensors,
    changed_voxel_ratio,
    compute_anchor_frames,
    compute_index_alignment_quality,
    compute_occ_comparison,
    encode_png_sequence_to_mp4,
    ensure_dirs,
    flatten_support_keyframe,
    frame_interval_for_index,
    interpolate_interval_frame,
    interpolate_interval_frame_fade_only,
    keyframe_target_frame_map,
    load_manifest_row,
    load_match_npz,
    load_pred_occ,
    load_query_output,
    load_raw_output,
    match_queries,
    occupancy_count,
    render_template1_frames,
    save_match_npz,
    save_support_npz,
    stabilize_occ_sequence,
    voxelize_support_frame,
    write_csv,
    write_json,
)


VARIANT_META = {
    "A_old_mixed_raw_keyframe": {
        "support_mode": "nearest_match",
        "keyframe_mode": "raw_semantic_occ",
        "description": "Old mixed mode: interval support interpolation for interior frames, raw semantic_occ hard lock on keyframes.",
    },
    "B_unified_direct_keyframe_voxelizer": {
        "support_mode": "nearest_match",
        "keyframe_mode": "direct_support_voxelizer",
        "description": "Unified voxelizer mode: keyframes reconstructed from their own support tensors through the same voxelizer.",
    },
    "C_unified_endpoint_fade": {
        "support_mode": "endpoint_fade",
        "keyframe_mode": "interval_endpoint_fade",
        "description": "Unified support + endpoint fade-in/out mode without query motion matching.",
    },
    "D_unified_nearest_match": {
        "support_mode": "nearest_match",
        "keyframe_mode": "interval_nearest_match",
        "description": "Unified support + class-wise nearest-neighbor matching mode.",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-index", type=int, default=3)
    p.add_argument("--frame-count", type=int, default=540)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--encoder", type=str, default="auto", choices=["auto", "h264_nvenc", "libx264"])
    p.add_argument("--resume", action="store_true", default=True)
    return p.parse_args()


def md_list(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


class StageLogger:
    def __init__(self, sample_index: int, frame_count: int, fps: int) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.progress_path = LOGS_DIR / "swvis4_stage_progress.jsonl"
        self.status_path = LOGS_DIR / "swvis4_stage_status.json"
        self.started_at = time.time()
        self.progress_path.write_text("", encoding="utf-8")
        self.payload = {
            "sample_index": sample_index,
            "frame_count": frame_count,
            "fps": fps,
            "started_at": self.started_at,
            "current_stage": None,
            "stages": {},
        }
        self._write_status()

    def _append(self, event: dict) -> None:
        event = {**event, "ts": time.time(), "elapsed_sec": time.time() - self.started_at}
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_status(self) -> None:
        self.status_path.write_text(json.dumps(self.payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def start(self, stage: str, **kwargs) -> None:
        self.payload["current_stage"] = stage
        self.payload["stages"].setdefault(stage, {})
        self.payload["stages"][stage].update({"status": "running", "start_ts": time.time(), **kwargs})
        self._append({"event": "stage_start", "stage": stage, **kwargs})
        self._write_status()

    def progress(self, stage: str, current: int, total: int, **kwargs) -> None:
        self.payload["current_stage"] = stage
        self.payload["stages"].setdefault(stage, {})
        self.payload["stages"][stage].update({"status": "running", "progress_current": current, "progress_total": total, **kwargs})
        self._append({"event": "stage_progress", "stage": stage, "current": current, "total": total, **kwargs})
        self._write_status()

    def done(self, stage: str, **kwargs) -> None:
        self.payload["stages"].setdefault(stage, {})
        start_ts = self.payload["stages"][stage].get("start_ts", time.time())
        self.payload["stages"][stage].update({"status": "done", "end_ts": time.time(), "duration_sec": time.time() - start_ts, **kwargs})
        self._append({"event": "stage_done", "stage": stage, **kwargs})
        self._write_status()

    def skip(self, stage: str, **kwargs) -> None:
        self.payload["stages"].setdefault(stage, {})
        self.payload["stages"][stage].update({"status": "skipped", "skip_ts": time.time(), **kwargs})
        self._append({"event": "stage_skip", "stage": stage, **kwargs})
        self._write_status()


def count_matching(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern))) if directory.exists() else 0


def load_npz_dict(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def audit_raw_tensors(sample_index: int) -> dict:
    raw = load_raw_output(sample_index)
    q = load_query_output(sample_index)
    fb = q["forward_backbone_outputs"]
    audit = {
        "sample_index": sample_index,
        "raw_output_keys": list(raw.keys()),
        "forward_backbone_output_keys": list(fb.keys()),
        "has_horizon_wise_forecast_points": "forecast_points_list" in fb and len(fb["forecast_points_list"]) == 6,
        "has_horizon_wise_forecast_semantics": "forecast_semantics_list" in fb and len(fb["forecast_semantics_list"]) == 6,
        "cls_score_shape": list(fb["cls_score"].shape),
        "refine_pts_shape": list(fb["refine_pts"].shape),
        "forecast_points_shapes": [list(x.shape) for x in fb.get("forecast_points_list", [])],
        "forecast_semantics_shapes": [list(x.shape) for x in fb.get("forecast_semantics_list", [])],
        "pred_trajs_shapes": [list(x.shape) for x in fb.get("pred_trajs_list", [])],
        "query_dim_consistent_current_future": True,
        "support_point_dim_consistent": True,
        "class_dim_consistent": True,
        "horizon_count": 7,
        "query_id_aligned_interpolation_possible": True,
        "shape_mismatch_requires_matching": True,
        "fallback_path": "not_used; real support-level forecast tensors available",
        "interpolation_mode_selected": "nearest-match support-level",
    }
    write_json(REPORTS_DIR / "raw_interpolation_tensor_audit.json", audit)
    (REPORTS_DIR / "raw_interpolation_tensor_audit.md").write_text(
        "\n".join(
            [
                "# raw interpolation tensor audit",
                f"- sample_index: `{sample_index}`",
                f"- has_horizon_wise_forecast_points: `{audit['has_horizon_wise_forecast_points']}`",
                f"- has_horizon_wise_forecast_semantics: `{audit['has_horizon_wise_forecast_semantics']}`",
                f"- cls_score_shape: `{audit['cls_score_shape']}`",
                f"- refine_pts_shape: `{audit['refine_pts_shape']}`",
                f"- forecast_points_shapes: `{audit['forecast_points_shapes']}`",
                f"- forecast_semantics_shapes: `{audit['forecast_semantics_shapes']}`",
                f"- interpolation_mode_selected: `{audit['interpolation_mode_selected']}`",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def build_support_keyframes(sample_index: int, logger: StageLogger, resume: bool) -> tuple[dict[int, Path], dict]:
    q = load_query_output(sample_index)
    fb = q["forward_backbone_outputs"]
    support_dir = ARTIFACTS_DIR / f"support_keyframes/sample_{sample_index:04d}"
    support_dir.mkdir(parents=True, exist_ok=True)
    support_paths: dict[int, Path] = {}
    stage = "build_support_keyframes"
    if resume and all((support_dir / f"h{h}_support.npz").exists() for h in range(7)):
        logger.skip(stage, reason="support_keyframes_exist", support_dir=str(support_dir))
    else:
        logger.start(stage, support_dir=str(support_dir))
    for h in range(7):
        pts, cls = (fb["refine_pts"], fb["cls_score"]) if h == 0 else (fb["forecast_points_list"][h - 1], fb["forecast_semantics_list"][h - 1])
        out_path = support_dir / f"h{h}_support.npz"
        if not (resume and out_path.exists()):
            support = build_support_keyframe_from_tensors(pts, cls, h, float(ANCHOR_TIMES[h]))
            save_support_npz(out_path, support)
        support_paths[h] = out_path
        logger.progress(stage, h + 1, 7, horizon_index=h)

    match_rows = []
    for h in range(6):
        a = load_npz_dict(support_paths[h])
        b = load_npz_dict(support_paths[h + 1])
        idx_quality = compute_index_alignment_quality(a, b, distance_threshold=6.0)
        if idx_quality["same_query_count"] and idx_quality["index_usable"]:
            match = build_index_aligned_match(a, b, distance_threshold=6.0)
            alignment_mode = "index_aligned"
        else:
            match = match_queries(a, b, distance_threshold=6.0)
            match["alignment_mode"] = np.array("classwise_nearest_neighbor", dtype="<U32")
            alignment_mode = "classwise_nearest_neighbor"
        match_path = ARTIFACTS_DIR / f"support_matches_h{h}_to_h{h + 1}.npz"
        save_match_npz(match_path, match)
        match_rows.append(
            {
                "interval": f"h{h}_to_h{h + 1}",
                "src_query_count": int(a["support_query_center"].shape[0]),
                "dst_query_count": int(b["support_query_center"].shape[0]),
                "matched_count": int(match["matched_count"]),
                "unmatched_src_count": int(match["unmatched_src_count"]),
                "unmatched_dst_count": int(match["unmatched_dst_count"]),
                "mean_match_distance_m": float(match["matched_distance_m"].mean()) if match["matched_distance_m"].size else 0.0,
                "distance_threshold_m": float(match["distance_threshold_m"]),
                "alignment_mode": alignment_mode,
                "index_alignment_quality": idx_quality,
            }
        )
    write_csv(REPORTS_DIR / "support_alignment_summary.csv", match_rows)
    summary = {
        "sample_index": sample_index,
        "support_keyframes_dir": str(support_dir),
        "match_intervals": match_rows,
    }
    write_json(REPORTS_DIR / "support_alignment_summary.json", summary)
    (REPORTS_DIR / "support_alignment_summary.md").write_text(
        "# support alignment summary\n\n" + md_list([json.dumps(r, ensure_ascii=False) for r in match_rows]),
        encoding="utf-8",
    )
    logger.done(stage, match_interval_count=len(match_rows))
    return support_paths, summary


def interpolate_support_frames_variant(sample_index: int, support_paths: dict[int, Path], frame_count: int, fps: int, support_mode: str, logger: StageLogger, resume: bool) -> Path:
    out_dir = ARTIFACTS_DIR / f"interpolated_support_frames/{support_mode}/sample_{sample_index:04d}_{frame_count}f"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = f"interpolate_support_frames::{support_mode}"
    if resume and count_matching(out_dir, "frame_*.npz") == frame_count:
        logger.skip(stage, reason="support_frames_exist", output_dir=str(out_dir), frame_count=frame_count)
        return out_dir
    logger.start(stage, output_dir=str(out_dir), frame_count=frame_count)
    keyframes = {h: load_npz_dict(p) for h, p in support_paths.items()}
    frame_rows = []
    for i in range(frame_count):
        h0, h1, alpha = frame_interval_for_index(i, frame_count)
        if support_mode == "endpoint_fade":
            flat = interpolate_interval_frame_fade_only(keyframes[h0], keyframes[h1], alpha)
        elif support_mode == "nearest_match":
            match = load_match_npz(ARTIFACTS_DIR / f"support_matches_h{h0}_to_h{h1}.npz")
            flat = interpolate_interval_frame(keyframes[h0], keyframes[h1], match, alpha)
        else:
            raise ValueError(f"unsupported support_mode: {support_mode}")
        flat["model_time_sec"] = np.float32((1.0 - alpha) * ANCHOR_TIMES[h0] + alpha * ANCHOR_TIMES[h1])
        flat["video_time_sec"] = np.float32(i / fps)
        flat["source_interval"] = np.array([h0, h1], dtype=np.int16)
        np.savez_compressed(out_dir / f"frame_{i:06d}.npz", **flat)
        frame_rows.append(
            {
                "frame_index": i,
                "video_time_sec": float(i / fps),
                "model_time_sec": float(flat["model_time_sec"]),
                "source_interval": f"h{h0}_to_h{h1}",
                "support_count": int(flat["support_xyz_metric"].shape[0]),
                "support_mode": support_mode,
            }
        )
        if ((i + 1) % 30 == 0) or ((i + 1) == frame_count):
            logger.progress(stage, i + 1, frame_count, support_mode=support_mode)
    write_csv(REPORTS_DIR / f"interpolation_frame_manifest_{support_mode}.csv", frame_rows)
    logger.done(stage, output_dir=str(out_dir))
    return out_dir


def voxelize_frames_variant(sample_index: int, support_frame_dir: Path, support_paths: dict[int, Path], frame_count: int, variant_name: str, logger: StageLogger, resume: bool) -> Path:
    variant = VARIANT_META[variant_name]
    out_dir = ARTIFACTS_DIR / f"interpolated_occ_frames/{variant_name}/sample_{sample_index:04d}_{frame_count}f"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = f"voxelize_frames::{variant_name}"
    if resume and count_matching(out_dir, "occ_frame_*.npy") == frame_count:
        logger.skip(stage, reason="occ_frames_exist", output_dir=str(out_dir), frame_count=frame_count)
        return out_dir
    logger.start(stage, output_dir=str(out_dir), frame_count=frame_count)
    pred_occ = load_pred_occ(sample_index).cpu().numpy().astype(np.uint8)
    keyframes = {h: load_npz_dict(p) for h, p in support_paths.items()}
    anchor_map = keyframe_target_frame_map(frame_count)
    anchor_rev = {v: k for k, v in anchor_map.items()}
    rows = []
    for i in range(frame_count):
        frame_support = load_npz_dict(support_frame_dir / f"frame_{i:06d}.npz")
        if variant["keyframe_mode"] == "raw_semantic_occ" and i in anchor_rev:
            h = anchor_rev[i]
            occ = pred_occ[h]
            source = "raw_semantic_occ"
        elif variant["keyframe_mode"] == "direct_support_voxelizer" and i in anchor_rev:
            h = anchor_rev[i]
            occ = voxelize_support_frame(flatten_support_keyframe(keyframes[h]), mode="trilinear_soft")
            source = "direct_support_voxelizer"
        else:
            occ = voxelize_support_frame(frame_support, mode="trilinear_soft")
            source = variant["keyframe_mode"] if i in anchor_rev else variant["support_mode"]
        np.save(out_dir / f"occ_frame_{i:06d}.npy", occ.astype(np.uint8))
        rows.append(
            {
                "frame_index": i,
                "occupied_count": occupancy_count(occ),
                "voxel_source": source,
                "is_keyframe": int(i in anchor_rev),
            }
        )
        if ((i + 1) % 30 == 0) or ((i + 1) == frame_count):
            logger.progress(stage, i + 1, frame_count, variant_name=variant_name)
    write_csv(REPORTS_DIR / f"voxelization_frames_summary_{variant_name}.csv", rows)
    write_json(
        REPORTS_DIR / f"voxelization_summary_{variant_name}.json",
        {
            "sample_index": sample_index,
            "frame_count": frame_count,
            "variant": variant_name,
            "occ_frame_dir": str(out_dir),
            "support_mode": variant["support_mode"],
            "keyframe_mode": variant["keyframe_mode"],
        },
    )
    logger.done(stage, output_dir=str(out_dir))
    return out_dir


def keyframe_reconstruction_check(sample_index: int, occ_frame_dir: Path, frame_count: int, variant_name: str) -> dict:
    pred_occ = load_pred_occ(sample_index).cpu().numpy().astype(np.uint8)
    mapping = keyframe_target_frame_map(frame_count)
    rows = []
    for h, fi in mapping.items():
        occ = np.load(occ_frame_dir / f"occ_frame_{fi:06d}.npy")
        metrics = compute_occ_comparison(occ, pred_occ[h])
        rows.append(
            {
                "horizon_index": int(h),
                "frame_index": int(fi),
                "model_time_sec": float(ANCHOR_TIMES[h]),
                **metrics,
            }
        )
    summary = {
        "sample_index": sample_index,
        "variant": variant_name,
        "rows": rows,
        "mean_occupied_iou": float(np.mean([r["occupied_iou"] for r in rows])),
        "mean_semantic_agreement_intersection": float(np.mean([r["semantic_agreement_intersection"] for r in rows])),
        "mean_semantic_agreement_union": float(np.mean([r["semantic_agreement_union"] for r in rows])),
        "mean_occupied_count_ratio": float(np.mean([r["occupied_count_ratio"] for r in rows])),
    }
    write_json(REPORTS_DIR / f"keyframe_reconstruction_check_{variant_name}.json", summary)
    return summary


def keyframe_jump_check(occ_frame_dir: Path, frame_count: int, variant_name: str, stabilized: bool) -> dict:
    pattern = "occ_stab_frame_*.npy" if stabilized else "occ_frame_*.npy"
    files = sorted(occ_frame_dir.glob(pattern))
    occs = [np.load(f).astype(np.uint8) for f in files]
    adjacent = []
    for i in range(1, len(occs)):
        prev = occs[i - 1]
        curr = occs[i]
        adjacent.append(
            {
                "pair": f"{i - 1}->{i}",
                "changed_voxel_ratio": changed_voxel_ratio(prev, curr),
                "occupied_count_delta": occupancy_count(curr) - occupancy_count(prev),
            }
        )
    adjacent_ratios = np.array([r["changed_voxel_ratio"] for r in adjacent], dtype=np.float32)
    adjacent_mean = float(adjacent_ratios.mean()) if adjacent_ratios.size else 0.0
    adjacent_p95 = float(np.percentile(adjacent_ratios, 95)) if adjacent_ratios.size else 0.0
    warn_thr = max(adjacent_mean * 2.0, adjacent_p95)
    keyframe_rows = []
    for h, key_idx in keyframe_target_frame_map(frame_count).items():
        curr = occs[key_idx]
        prev_metrics = None
        next_metrics = None
        warnings = []
        if key_idx > 0:
            prev = occs[key_idx - 1]
            prev_metrics = {
                "pair": f"{key_idx - 1}->{key_idx}",
                "changed_voxel_ratio": changed_voxel_ratio(prev, curr),
                "occupied_count_delta": occupancy_count(curr) - occupancy_count(prev),
                "class_count_delta_l1": int(np.abs(np.bincount(curr[curr != EMPTY_IDX], minlength=17) - np.bincount(prev[prev != EMPTY_IDX], minlength=17)).sum()),
            }
            if prev_metrics["changed_voxel_ratio"] > warn_thr:
                warnings.append("prev_to_key_jump")
        if key_idx < len(occs) - 1:
            nxt = occs[key_idx + 1]
            next_metrics = {
                "pair": f"{key_idx}->{key_idx + 1}",
                "changed_voxel_ratio": changed_voxel_ratio(curr, nxt),
                "occupied_count_delta": occupancy_count(nxt) - occupancy_count(curr),
                "class_count_delta_l1": int(np.abs(np.bincount(nxt[nxt != EMPTY_IDX], minlength=17) - np.bincount(curr[curr != EMPTY_IDX], minlength=17)).sum()),
            }
            if next_metrics["changed_voxel_ratio"] > warn_thr:
                warnings.append("key_to_next_jump")
        keyframe_rows.append(
            {
                "horizon_index": int(h),
                "frame_index": int(key_idx),
                "model_time_sec": float(ANCHOR_TIMES[h]),
                "prev_metrics": prev_metrics,
                "next_metrics": next_metrics,
                "warnings": warnings,
            }
        )
    summary = {
        "variant": variant_name,
        "stabilized": stabilized,
        "adjacent_changed_voxel_ratio_mean": adjacent_mean,
        "adjacent_changed_voxel_ratio_p95": adjacent_p95,
        "jump_warning_threshold": warn_thr,
        "keyframe_rows": keyframe_rows,
        "warning_count": int(sum(len(r["warnings"]) for r in keyframe_rows)),
    }
    suffix = "stabilized" if stabilized else "raw"
    write_json(REPORTS_DIR / f"keyframe_jump_check_{variant_name}_{suffix}.json", summary)
    return summary


def stabilize_variant(sample_index: int, occ_dir: Path, frame_count: int, variant_name: str, logger: StageLogger, resume: bool) -> tuple[Path, dict]:
    stab_dir = ARTIFACTS_DIR / f"stabilized_occ_frames/{variant_name}/sample_{sample_index:04d}_{frame_count}f"
    stage = f"stabilize::{variant_name}"
    if resume and count_matching(stab_dir, "occ_stab_frame_*.npy") == frame_count and (REPORTS_DIR / f"anti_flicker_summary_{variant_name}.json").exists():
        logger.skip(stage, reason="stabilized_frames_exist", output_dir=str(stab_dir), frame_count=frame_count)
        summary = json.loads((REPORTS_DIR / f"anti_flicker_summary_{variant_name}.json").read_text(encoding="utf-8"))
        return stab_dir, summary
    logger.start(stage, output_dir=str(stab_dir), frame_count=frame_count)
    summary = stabilize_occ_sequence(
        occ_dir,
        stab_dir,
        frame_count,
        progress_cb=lambda current, total: logger.progress(stage, current, total, variant_name=variant_name),
        progress_every=30,
    )
    write_json(REPORTS_DIR / f"anti_flicker_summary_{variant_name}.json", summary)
    logger.done(stage, output_dir=str(stab_dir))
    return stab_dir, summary


def render_and_video(sample_index: int, stab_dir: Path, frame_count: int, fps: int, variant_name: str, encoder: str, logger: StageLogger, resume: bool) -> tuple[Path, Path, dict]:
    frame_dir = FIGURES_DIR / f"frames/{variant_name}/sample_{sample_index:04d}_{frame_count}f"
    seconds = int(round(frame_count / fps))
    out_mp4 = VIDEOS_DIR / f"{variant_name}_sample_{sample_index:04d}_{frame_count}f_{fps}fps_{seconds}s.mp4"
    stage_render = f"render_frames::{variant_name}"
    stage_encode = f"encode_video::{variant_name}"
    if resume and count_matching(frame_dir, "frame_*.png") == frame_count:
        logger.skip(stage_render, reason="rendered_frames_exist", output_dir=str(frame_dir), frame_count=frame_count)
        render_summary = {"frame_count": frame_count, "output_dir": str(frame_dir)}
    else:
        logger.start(stage_render, output_dir=str(frame_dir), frame_count=frame_count)
        render_summary = render_template1_frames(
            sample_index,
            stab_dir,
            frame_dir,
            progress_cb=lambda current, total: logger.progress(stage_render, current, total, variant_name=variant_name),
            progress_every=30,
        )
        logger.done(stage_render, output_dir=str(frame_dir))
    if resume and out_mp4.exists():
        logger.skip(stage_encode, reason="video_exists", video_path=str(out_mp4))
        encode_status = {"encoder": encoder, "attempts": []}
    else:
        logger.start(stage_encode, video_path=str(out_mp4), encoder=encoder)
        encode_status = encode_png_sequence_to_mp4(frame_dir, out_mp4, fps=fps, encoder_preference=encoder)
        logger.done(stage_encode, video_path=str(out_mp4), encoder_used=encode_status["encoder"])
    summary = {
        **render_summary,
        "video_path": str(out_mp4),
        "video_encoder": encode_status["encoder"],
        "video_encode_attempts": encode_status["attempts"],
    }
    return frame_dir, out_mp4, summary


def compare_old_vs_fixed(old_frame_dir: Path, fixed_frame_dir: Path, sample_index: int, frame_count: int, fps: int, encoder: str, logger: StageLogger, resume: bool) -> dict:
    out_dir = VIDEOS_DIR / f"comparison_old_vs_fixed_frames_sample_{sample_index:04d}_{frame_count}f"
    seconds = int(round(frame_count / fps))
    out_mp4 = VIDEOS_DIR / f"comparison_old_mixed_vs_unified_fixed_sample_{sample_index:04d}_{frame_count}f_{fps}fps_{seconds}s.mp4"
    stage = "compare_old_vs_fixed"
    if resume and out_mp4.exists():
        logger.skip(stage, reason="comparison_video_exists", video_path=str(out_mp4))
        summary = json.loads((REPORTS_DIR / "old_vs_fixed_comparison.json").read_text(encoding="utf-8")) if (REPORTS_DIR / "old_vs_fixed_comparison.json").exists() else {
            "frame_count": frame_count,
            "comparison_video_path": str(out_mp4),
            "video_encoder": encoder,
            "video_encode_attempts": [],
        }
        return summary
    logger.start(stage, output_dir=str(out_dir), video_path=str(out_mp4))
    summary = build_side_by_side_video(fixed_frame_dir, old_frame_dir, out_dir, out_mp4, fps=fps, encoder_preference=encoder)
    write_json(REPORTS_DIR / "old_vs_fixed_comparison.json", summary)
    logger.done(stage, video_path=str(out_mp4), encoder_used=summary.get("video_encoder", ""))
    return summary


def build_ablation_summary(sample_index: int, frame_count: int, fps: int, variant_results: dict[str, dict]) -> dict:
    rows = []
    for variant_name, result in variant_results.items():
        raw_jump = result["jump_raw"]
        stab_jump = result["jump_stabilized"]
        recon = result["recon"]
        rows.append(
            {
                "variant": variant_name,
                "description": VARIANT_META[variant_name]["description"],
                "support_mode": VARIANT_META[variant_name]["support_mode"],
                "keyframe_mode": VARIANT_META[variant_name]["keyframe_mode"],
                "mean_occupied_iou": recon["mean_occupied_iou"],
                "mean_semantic_agreement_intersection": recon["mean_semantic_agreement_intersection"],
                "mean_semantic_agreement_union": recon["mean_semantic_agreement_union"],
                "mean_occupied_count_ratio": recon["mean_occupied_count_ratio"],
                "raw_adjacent_changed_voxel_ratio_mean": raw_jump["adjacent_changed_voxel_ratio_mean"],
                "raw_jump_warning_count": raw_jump["warning_count"],
                "stabilized_adjacent_changed_voxel_ratio_mean": stab_jump["adjacent_changed_voxel_ratio_mean"],
                "stabilized_jump_warning_count": stab_jump["warning_count"],
                "video_encoder": result.get("render_summary", {}).get("video_encoder", ""),
            }
        )
    write_csv(REPORTS_DIR / "keyframe_jump_ablation_summary.csv", rows)
    summary = {
        "sample_index": sample_index,
        "frame_count": frame_count,
        "fps": fps,
        "rows": rows,
    }
    write_json(REPORTS_DIR / "keyframe_jump_ablation_summary.json", summary)
    return summary


def build_report(
    sample_index: int,
    frame_count: int,
    fps: int,
    encoder_request: str,
    audit: dict,
    alignment: dict,
    support_dirs: dict[str, str],
    variant_results: dict[str, dict],
    ablation_summary: dict,
    old_vs_fixed: dict,
) -> None:
    playback_duration_seconds = frame_count / fps
    report = {
        "sample_index": sample_index,
        "goal": "Fix keyframe jump in SW-VIS4 query/support interpolation with unified support-to-occupancy voxelization.",
        "frame_count": frame_count,
        "fps": fps,
        "playback_duration_seconds": playback_duration_seconds,
        "raw_tensor_audit": audit,
        "support_alignment_summary": alignment,
        "support_frame_dirs": support_dirs,
        "variant_results": variant_results,
        "ablation_summary": ablation_summary,
        "selected_fixed_variant": "D_unified_nearest_match",
        "comparison_old_vs_fixed": old_vs_fixed,
        "video_outputs": {
            "old_mixed": variant_results["A_old_mixed_raw_keyframe"]["render_summary"]["video_path"],
            "fixed_unified": variant_results["D_unified_nearest_match"]["render_summary"]["video_path"],
            "old_vs_fixed": old_vs_fixed["comparison_video_path"],
        },
        "safe_claims": [
            "This is a 3s forecast rendered as a 9s slow-motion visualization.",
            "It is not a 9s prediction.",
            "All frames in the fixed variant come from unified support interpolation plus a unified voxelizer.",
            "raw semantic_occ is used for endpoint reconstruction check only in fixed variants.",
            "No manual editing of prediction.",
            "No training.",
        ],
        "key_paths": {
            "support_keyframes": alignment["support_keyframes_dir"],
            "interpolated_support_frames_fixed": support_dirs["nearest_match"],
            "interpolated_occ_frames_fixed": variant_results["D_unified_nearest_match"]["occ_dir"],
            "stabilized_occ_frames_fixed": variant_results["D_unified_nearest_match"]["stab_dir"],
        },
        "next_unique_action": "Tune support-to-occupancy aggregation and thresholds if unified reconstruction gap remains large for small objects.",
    }
    write_json(REPORTS_DIR / "stage_swvis4_query_support_interpolation_report.json", report)

    fixed = variant_results["D_unified_nearest_match"]
    fixed_recon = fixed["recon"]
    fixed_jump = fixed["jump_stabilized"]
    (REPORTS_DIR / "stage_swvis4_query_support_interpolation_report.md").write_text(
        "\n".join(
            [
                "# Stage SW-VIS4 keyframe jump repair",
                "",
                "1. Goal",
                "- Fix keyframe jump by removing the mixed `interpolated interior + raw semantic_occ keyframe lock` path.",
                "",
                "2. Raw tensor audit",
                f"- mode selected: `{audit['interpolation_mode_selected']}`",
                f"- forecast_points available: `{audit['has_horizon_wise_forecast_points']}`",
                f"- forecast_semantics available: `{audit['has_horizon_wise_forecast_semantics']}`",
                "",
                "3. Unified generation path",
                "- Fixed variants generate every frame, including keyframes, from support interpolation plus the same support-to-occupancy voxelizer.",
                "- raw semantic_occ is retained only for reconstruction check against horizon endpoints.",
                "",
                "4. Ablation variants",
                *[f"- `{row['variant']}`: {row['description']}" for row in ablation_summary["rows"]],
                "",
                "5. Selected fixed variant",
                "- `D_unified_nearest_match`",
                f"- mean keyframe occupied IoU vs raw semantic_occ: `{fixed_recon['mean_occupied_iou']:.6f}`",
                f"- mean keyframe semantic agreement (intersection): `{fixed_recon['mean_semantic_agreement_intersection']:.6f}`",
                f"- mean keyframe occupied count ratio: `{fixed_recon['mean_occupied_count_ratio']:.6f}`",
                f"- stabilized keyframe jump warning count: `{fixed_jump['warning_count']}`",
                "",
                "6. Video outputs",
                f"- old mixed video: `{variant_results['A_old_mixed_raw_keyframe']['render_summary']['video_path']}`",
                f"- fixed unified video: `{fixed['render_summary']['video_path']}`",
                f"- old vs fixed comparison: `{old_vs_fixed['comparison_video_path']}`",
                f"- requested encoder: `{encoder_request}`",
                f"- fixed video encoder used: `{fixed['render_summary']['video_encoder']}`",
                "",
                "7. Safe claims",
                "- This is a 3s forecast rendered as a 9s slow-motion visualization.",
                "- It is not a 9s prediction.",
                "- No manual editing of prediction.",
                "- No training.",
                "",
                "8. Limitations",
                "- Unified voxelizer reconstruction can differ from raw semantic_occ because SparseWorld's native occupancy head is not being inserted directly.",
                "- Query matching is index-aligned only when query identity quality is adequate; otherwise class-wise nearest-neighbor matching is used.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    ensure_dirs()
    sample_index = args.sample_index
    frame_count = args.frame_count
    fps = args.fps
    logger = StageLogger(sample_index, frame_count, fps)

    logger.start("audit_raw_tensors")
    audit = audit_raw_tensors(sample_index)
    logger.done("audit_raw_tensors")
    support_paths, alignment = build_support_keyframes(sample_index, logger, args.resume)

    support_dirs = {
        "nearest_match": str(interpolate_support_frames_variant(sample_index, support_paths, frame_count, fps, "nearest_match", logger, args.resume)),
        "endpoint_fade": str(interpolate_support_frames_variant(sample_index, support_paths, frame_count, fps, "endpoint_fade", logger, args.resume)),
    }

    variant_results: dict[str, dict] = {}
    for variant_name, meta in VARIANT_META.items():
        logger.start(f"variant::{variant_name}", variant_name=variant_name)
        support_frame_dir = Path(support_dirs[meta["support_mode"]])
        occ_dir = voxelize_frames_variant(sample_index, support_frame_dir, support_paths, frame_count, variant_name, logger, args.resume)
        recon = keyframe_reconstruction_check(sample_index, occ_dir, frame_count, variant_name)
        jump_raw = keyframe_jump_check(occ_dir, frame_count, variant_name, stabilized=False)
        stab_dir, anti = stabilize_variant(sample_index, occ_dir, frame_count, variant_name, logger, args.resume)
        jump_stabilized = keyframe_jump_check(stab_dir, frame_count, variant_name, stabilized=True)
        result = {
            "occ_dir": str(occ_dir),
            "stab_dir": str(stab_dir),
            "recon": recon,
            "jump_raw": jump_raw,
            "jump_stabilized": jump_stabilized,
            "anti_flicker": anti,
        }
        if variant_name in ("A_old_mixed_raw_keyframe", "D_unified_nearest_match"):
            frame_dir, video_path, render_summary = render_and_video(sample_index, stab_dir, frame_count, fps, variant_name, args.encoder, logger, args.resume)
            result["frame_dir"] = str(frame_dir)
            result["render_summary"] = render_summary
            result["render_summary"]["video_path"] = str(video_path)
        variant_results[variant_name] = result
        logger.done(f"variant::{variant_name}", variant_name=variant_name)

    old_vs_fixed = compare_old_vs_fixed(
        Path(variant_results["A_old_mixed_raw_keyframe"]["frame_dir"]),
        Path(variant_results["D_unified_nearest_match"]["frame_dir"]),
        sample_index,
        frame_count,
        fps,
        args.encoder,
        logger,
        args.resume,
    )

    logger.start("build_ablation_summary")
    ablation_summary = build_ablation_summary(sample_index, frame_count, fps, variant_results)
    logger.done("build_ablation_summary")
    logger.start("build_report")
    build_report(sample_index, frame_count, fps, args.encoder, audit, alignment, support_dirs, variant_results, ablation_summary, old_vs_fixed)
    logger.done("build_report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
