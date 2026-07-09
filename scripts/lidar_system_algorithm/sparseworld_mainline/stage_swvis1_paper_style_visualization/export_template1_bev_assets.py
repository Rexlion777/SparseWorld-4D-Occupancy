from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis1_paper_style_visualization"
VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis1_paper_style_visualization"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_swvis1_paper_style_visualization"
SWVIS4_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_swvis4_query_support_interpolation"
SWVIS4_VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis4_query_support_interpolation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vismod = load_module("swvis1_main_template1_bev", SCRIPT_DIR / "run_sparseworld_paper_style_visualization.py")
renderer = load_module("swvis1_renderer_template1_bev", SCRIPT_DIR / "semantic_occupancy_renderer.py")
mp4mod = load_module("swvis1_mp4_template1_bev", SCRIPT_DIR / "generate_interpolated_future_mp4.py")
swlib = load_module(
    "swvis4_swlib_template1_bev",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/swvis4_support_interp_lib.py",
)


class StageLogger:
    def __init__(self, status_path: Path, progress_path: Path) -> None:
        self.status_path = status_path
        self.progress_path = progress_path
        self.state: dict[str, Any] = {"started_at": time.time(), "current_stage": None, "stages": {}}
        self._flush()

    def _flush(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def start(self, name: str, **payload: Any) -> None:
        self.state["current_stage"] = name
        self.state["stages"].setdefault(name, {})
        self.state["stages"][name].update({"status": "running", "start_ts": time.time(), **payload})
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "event": "start", "stage": name, **payload}, ensure_ascii=False) + "\n")
        self._flush()

    def progress(self, name: str, current: int, total: int, **payload: Any) -> None:
        self.state["stages"].setdefault(name, {})
        self.state["stages"][name].update({"status": "running", "progress_current": current, "progress_total": total, **payload})
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "event": "progress", "stage": name, "current": current, "total": total, **payload}, ensure_ascii=False) + "\n")
        self._flush()

    def done(self, name: str, **payload: Any) -> None:
        stage = self.state["stages"].setdefault(name, {})
        end_ts = time.time()
        start_ts = float(stage.get("start_ts", end_ts))
        stage.update({"status": "done", "end_ts": end_ts, "duration_sec": end_ts - start_ts, **payload})
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": end_ts, "event": "done", "stage": name, **payload}, ensure_ascii=False) + "\n")
        self._flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-index", type=int, default=3)
    p.add_argument("--frame-count", type=int, default=240)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--encoder", type=str, default="h264_nvenc")
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_template1_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "template_name": "template1",
        "description": "SparseWorld voxel-paper composite with fixed 6-camera present panel and SW-only current/future voxel rollout below.",
        "camera_panel": "top 2x3 nuScenes camera grid",
        "lower_panel": {
            "left": "Observations = SparseWorld current voxel h0",
            "right": "Predicted Futures = SparseWorld future voxels",
        },
        "text_style": {"present": "black", "observations": "black", "predicted_futures": "black"},
        "image_variants": [],
        "video_variants": [],
    }


def upsert_variant(items: list[dict[str, Any]], variant: dict[str, Any], key: str = "name") -> list[dict[str, Any]]:
    out = []
    seen = False
    for item in items:
        if item.get(key) == variant.get(key):
            out.append(variant)
            seen = True
        else:
            out.append(item)
    if not seen:
        out.append(variant)
    return out


def build_style() -> Any:
    return renderer.RendererStyle(
        dpi=300,
        backend="open3d" if getattr(renderer, "o3d", None) is not None else "matplotlib",
        mode="voxel_mesh" if getattr(renderer, "o3d", None) is not None else "point_cloud",
        viewpoint="top",
        render_width=2200,
        render_height=1400,
        open3d_point_size=6.8,
    )


def main() -> int:
    args = parse_args()
    sample_index = int(args.sample_index)
    frame_count = int(args.frame_count)
    fps = int(args.fps)
    repo_root = Path(args.repo_root).resolve()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    status_path = LOGS_DIR / "template1_bev_export_status.json"
    progress_path = LOGS_DIR / "template1_bev_export_progress.jsonl"
    logger = StageLogger(status_path, progress_path)

    style = build_style()
    pred_temporal, _ = vismod.load_occ_pair(sample_index)
    sample_manifest_full = {
        int(r["sample_index"]): r
        for r in vismod.read_csv(
            PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv"
        )
    }
    all_infos = vismod.load_sparseworld_info_list(repo_root)
    mapped_info_index = int(sample_manifest_full[sample_index]["mapped_info_index"])
    cam_paths = vismod.current_camera_paths_from_info(all_infos[mapped_info_index])
    cam_imgs = vismod.load_camera_strip(cam_paths, repo_root)
    cam_labels = [Path(p).parent.name for p in cam_paths]
    cam_panel = vismod.camera_grid(cam_imgs, cam_labels)

    legend_classes = ["car", "truck", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    legend_strip = renderer.build_legend_strip(legend_classes, width=cam_panel.width, height=58)
    legend_strip = ImageOps.contain(legend_strip, (cam_panel.width, 52))
    observation_panel = renderer.render_semantic_occ(pred_temporal[0], "", style)

    static_png = FIGURES_DIR / f"template1_bev_observation_future_sample{sample_index}.png"
    bev_video_frames_dir = FIGURES_DIR / f"template1_bev_video_frames_sample_{sample_index:04d}_{frame_count}f"
    bev_video_path = VIDEOS_DIR / f"template1_bev_observation_future_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    src_future_only_video = SWVIS4_VIDEOS_DIR / f"sparseworld_predicted_future_only_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    dst_future_only_video = VIDEOS_DIR / f"template1_predicted_future_only_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    future_frames_dir = PROJECT_ROOT / f"projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis4_query_support_interpolation/frames_future_only/sample_{sample_index:04d}_{frame_count}f"

    logger.start("render_static_template1_bev", output_path=str(static_png))
    vismod.compose_observation_future(
        cam_panel,
        observation_panel,
        [renderer.render_semantic_occ(pred_temporal[h], "", style) for h in [2, 4, 6]],
        f"Sample {sample_index} | template1 BEV",
        [renderer.HORIZON_LABELS[h] for h in [0, 2, 4, 6]],
        static_png,
    )
    logger.done("render_static_template1_bev", output_path=str(static_png))

    logger.start("copy_future_only_video_into_template1", src=str(src_future_only_video), dst=str(dst_future_only_video))
    if not src_future_only_video.exists():
        raise FileNotFoundError(f"missing future-only video: {src_future_only_video}")
    shutil.copy2(src_future_only_video, dst_future_only_video)
    logger.done("copy_future_only_video_into_template1", dst=str(dst_future_only_video))

    frame_files = sorted(future_frames_dir.glob("frame_*.png"))
    if len(frame_files) != frame_count:
        raise RuntimeError(f"expected {frame_count} future-only frames in {future_frames_dir}, found {len(frame_files)}")

    logger.start("compose_template1_bev_video_frames", output_dir=str(bev_video_frames_dir), frame_count=frame_count)
    bev_video_frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_path in enumerate(frame_files):
        out_frame = bev_video_frames_dir / f"frame_{i:06d}.png"
        if args.resume and out_frame.exists():
            if i % 20 == 0 or i == frame_count - 1:
                logger.progress("compose_template1_bev_video_frames", i + 1, frame_count, resumed=True)
            continue
        future_panel = Image.open(frame_path).convert("RGB")
        frame = mp4mod.compose_template1_video_frame(cam_panel, observation_panel, future_panel, legend_strip)
        frame.save(out_frame)
        if i % 20 == 0 or i == frame_count - 1:
            logger.progress("compose_template1_bev_video_frames", i + 1, frame_count)
    logger.done("compose_template1_bev_video_frames", output_dir=str(bev_video_frames_dir))

    logger.start("encode_template1_bev_video", output_path=str(bev_video_path), encoder=args.encoder)
    encode_status = swlib.encode_png_sequence_to_mp4(bev_video_frames_dir, bev_video_path, fps, encoder_preference=args.encoder)
    logger.done("encode_template1_bev_video", output_path=str(bev_video_path), encoder_used=encode_status.get("encoder"))

    manifest_path = REPORTS_DIR / "paper_style_template1_manifest.json"
    manifest = load_template1_manifest(manifest_path)
    manifest["description"] = (
        "SparseWorld voxel-paper composite with fixed 6-camera present panel and SW-only current/future voxel rollout below. "
        "Includes BEV/top-view template1 static image and video variants plus a pure predicted-future video variant."
    )
    manifest["image_variants"] = upsert_variant(
        list(manifest.get("image_variants", [])),
        {
            "name": "template1_bev_observation_future_static",
            "sample_index": sample_index,
            "horizons": [0, 2, 4, 6],
            "path": str(static_png),
            "viewpoint": "top",
            "source": "SparseWorld prediction h0/h2/h4/h6 rendered with template1 BEV voxel style",
        },
    )
    video_variants = list(manifest.get("video_variants", []))
    video_variants = upsert_variant(
        video_variants,
        {
            "name": "template1_bev_observation_future_240f_60fps",
            "sample_index": sample_index,
            "frame_count": frame_count,
            "fps": fps,
            "path": str(bev_video_path),
            "viewpoint": "top",
            "source": "Template1 composition video using BEV get_occ-derived future frames",
        },
    )
    video_variants = upsert_variant(
        video_variants,
        {
            "name": "template1_predicted_future_only_240f_60fps",
            "sample_index": sample_index,
            "frame_count": frame_count,
            "fps": fps,
            "path": str(dst_future_only_video),
            "viewpoint": "top",
            "source": "Pure predicted future voxel video with no camera panel, legend, or text",
        },
    )
    manifest["video_variants"] = video_variants
    write_json(manifest_path, manifest)

    export_manifest = {
        "sample_index": sample_index,
        "frame_count": frame_count,
        "fps": fps,
        "static_png": str(static_png),
        "template1_bev_video": str(bev_video_path),
        "template1_predicted_future_only_video": str(dst_future_only_video),
        "template1_bev_frames_dir": str(bev_video_frames_dir),
        "future_only_frames_source_dir": str(future_frames_dir),
        "encoder_used": encode_status.get("encoder"),
    }
    write_json(REPORTS_DIR / "template1_bev_export_manifest.json", export_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
