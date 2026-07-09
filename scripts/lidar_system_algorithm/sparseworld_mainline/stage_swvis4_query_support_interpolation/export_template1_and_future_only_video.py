from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import swvis4_support_interp_lib as swlib


PROJECT_ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
SWVIS1_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"
SWVIS1_VIDEOS = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis1_paper_style_visualization"
SWVIS4_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation"
SWVIS4_LOGS = PROJECT_ROOT / "logs/sparseworld_mainline/stage_swvis4_query_support_interpolation"
SWVIS4_FIGURES = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis4_query_support_interpolation"
SWVIS4_VIDEOS = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis4_query_support_interpolation"

STATUS_PATH = SWVIS4_LOGS / "swvis4_future_only_export_status.json"
PROGRESS_PATH = SWVIS4_LOGS / "swvis4_future_only_export_progress.jsonl"


class StageLogger:
    def __init__(self, sample_index: int, frame_count: int, fps: int) -> None:
        self.status: dict[str, Any] = {
            "sample_index": int(sample_index),
            "frame_count": int(frame_count),
            "fps": int(fps),
            "started_at": time.time(),
            "current_stage": None,
            "stages": {},
        }
        SWVIS4_LOGS.mkdir(parents=True, exist_ok=True)
        self._write()

    def _write(self) -> None:
        STATUS_PATH.write_text(json.dumps(self.status, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append(self, stage: str, payload: dict[str, Any]) -> None:
        with PROGRESS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "stage": stage, **payload}, ensure_ascii=False) + "\n")

    def start(self, stage: str, **kwargs: Any) -> None:
        self.status["current_stage"] = stage
        self.status["stages"][stage] = {"status": "running", "start_ts": time.time(), **kwargs}
        self._append(stage, {"event": "start", **kwargs})
        self._write()

    def progress(self, stage: str, current: int, total: int, **kwargs: Any) -> None:
        entry = self.status["stages"].setdefault(stage, {"status": "running", "start_ts": time.time()})
        entry.update({"status": "running", "progress_current": int(current), "progress_total": int(total), **kwargs})
        self._append(stage, {"event": "progress", "progress_current": int(current), "progress_total": int(total), **kwargs})
        self._write()

    def done(self, stage: str, **kwargs: Any) -> None:
        entry = self.status["stages"].setdefault(stage, {"start_ts": time.time()})
        end_ts = time.time()
        entry.update({"status": "done", "end_ts": end_ts, "duration_sec": float(end_ts - entry.get("start_ts", end_ts)), **kwargs})
        self._append(stage, {"event": "done", **kwargs})
        self._write()

    def skip(self, stage: str, reason: str, **kwargs: Any) -> None:
        self.status["stages"][stage] = {"status": "skipped", "skip_ts": time.time(), "reason": reason, **kwargs}
        self._append(stage, {"event": "skip", "reason": reason, **kwargs})
        self._write()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-index", type=int, default=3)
    p.add_argument("--frame-count", type=int, default=240)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--encoder", type=str, default="h264_nvenc")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def copy_template1_video(sample_index: int, frame_count: int, fps: int, logger: StageLogger) -> Path:
    stage = "copy_template1_video"
    src = SWVIS4_VIDEOS / f"sparseworld_support_interp_endpoint_exact_layered_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    dst = SWVIS1_VIDEOS / f"template1_sparseworld_endpoint_exact_layered_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    SWVIS1_VIDEOS.mkdir(parents=True, exist_ok=True)
    logger.start(stage, src=str(src), dst=str(dst))
    shutil.copy2(src, dst)
    manifest_path = SWVIS1_REPORTS / "paper_style_template1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"template_name": "template1"}
    variants = manifest.setdefault("video_variants", [])
    entry = {
        "name": "endpoint_exact_layered_240f_60fps" if frame_count == 240 and fps == 60 else f"endpoint_exact_layered_{frame_count}f_{fps}fps",
        "sample_index": sample_index,
        "frame_count": frame_count,
        "fps": fps,
        "path": str(dst),
        "source": "SW-VIS4 endpoint-exact layered output",
    }
    variants = [v for v in variants if not (v.get("sample_index") == sample_index and v.get("frame_count") == frame_count and v.get("fps") == fps)]
    variants.append(entry)
    manifest["video_variants"] = variants
    write_json(manifest_path, manifest)
    logger.done(stage, dst=str(dst))
    return dst


def render_future_only_frames(sample_index: int, frame_count: int, logger: StageLogger, resume: bool) -> Path:
    stage = "render_future_only_frames"
    occ_dir = PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_swvis4_query_support_interpolation/layered_occ_frames_stabilized/sample_{sample_index:04d}_{frame_count}f"
    out_dir = SWVIS4_FIGURES / f"frames_future_only/sample_{sample_index:04d}_{frame_count}f"
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume and len(list(out_dir.glob("frame_*.png"))) == frame_count:
        logger.skip(stage, reason="future_only_frames_exist", output_dir=str(out_dir), frame_count=frame_count)
        return out_dir
    logger.start(stage, output_dir=str(out_dir), frame_count=frame_count)

    renderer = swlib.load_renderer_module()
    style = renderer.RendererStyle(
        dpi=180,
        backend="open3d",
        mode="voxel_mesh",
        viewpoint="top",
        render_width=1400,
        render_height=900,
        open3d_point_size=6.8,
    )
    for i in range(frame_count):
        occ = torch.from_numpy(np.load(occ_dir / f"occ_stab_frame_{i:06d}.npy").astype(np.int64))
        img = renderer.render_semantic_occ(occ, "", style)
        img.save(out_dir / f"frame_{i:06d}.png")
        if ((i + 1) % 30 == 0) or ((i + 1) == frame_count):
            logger.progress(stage, i + 1, frame_count)
    logger.done(stage, output_dir=str(out_dir))
    return out_dir


def encode_future_only_video(sample_index: int, frame_count: int, fps: int, encoder: str, frame_dir: Path, logger: StageLogger, resume: bool) -> Path:
    stage = "encode_future_only_video"
    out_mp4 = SWVIS4_VIDEOS / f"sparseworld_predicted_future_only_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    if resume and out_mp4.exists():
        logger.skip(stage, reason="future_only_video_exists", video_path=str(out_mp4))
        return out_mp4
    logger.start(stage, video_path=str(out_mp4), encoder=encoder)
    status = swlib.encode_png_sequence_to_mp4(frame_dir, out_mp4, fps=fps, encoder_preference=encoder)
    logger.done(stage, video_path=str(out_mp4), encoder_used=status["encoder"])
    return out_mp4


def main() -> None:
    args = parse_args()
    for p in [SWVIS1_REPORTS, SWVIS1_VIDEOS, SWVIS4_REPORTS, SWVIS4_FIGURES, SWVIS4_VIDEOS, SWVIS4_LOGS]:
        p.mkdir(parents=True, exist_ok=True)

    logger = StageLogger(args.sample_index, args.frame_count, args.fps)
    template1_video = copy_template1_video(args.sample_index, args.frame_count, args.fps, logger)
    frame_dir = render_future_only_frames(args.sample_index, args.frame_count, logger, args.resume)
    future_only_video = encode_future_only_video(args.sample_index, args.frame_count, args.fps, args.encoder, frame_dir, logger, args.resume)

    manifest = {
        "sample_index": args.sample_index,
        "frame_count": args.frame_count,
        "fps": args.fps,
        "template1_video_path": str(template1_video),
        "future_only_frames_dir": str(frame_dir),
        "future_only_video_path": str(future_only_video),
        "future_only_note": "No camera panel, no text, no legend; pure get_occ-derived predicted future voxel frames only.",
    }
    write_json(SWVIS4_REPORTS / "template1_and_future_only_export_manifest.json", manifest)


if __name__ == "__main__":
    main()
