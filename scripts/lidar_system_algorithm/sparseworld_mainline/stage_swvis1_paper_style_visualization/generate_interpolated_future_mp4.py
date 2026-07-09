from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis1_paper_style_visualization"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_swvis1_paper_style_visualization"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vismod = load_module("swvis1_main_for_mp4", SCRIPT_DIR / "run_sparseworld_paper_style_visualization.py")
renderer = load_module("swvis1_renderer_for_mp4", SCRIPT_DIR / "semantic_occupancy_renderer.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-index", type=int, default=3)
    p.add_argument("--frame-count", type=int, default=540)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--actual-hz", type=int, default=180)
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    p.add_argument(
        "--python-open3d",
        default=r"C:\Users\Administrator\miniconda3\envs\cv_lidar\python.exe",
    )
    return p.parse_args()


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_pred_temporal(sample_index: int) -> torch.Tensor:
    path = PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/standard_occ/sample_{sample_index:04d}_standard_pred_occ_temporal.pt"
    return torch.load(path, map_location="cpu", weights_only=False).long()


def load_sw2_manifest_row(sample_index: int) -> dict[str, str]:
    path = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv"
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return next(r for r in rows if int(r["sample_index"]) == sample_index)


def load_sparseworld_infos(repo_root: Path) -> list[dict[str, Any]]:
    with (repo_root / "data/nuscenes/bevdetv2-nuscenes_infos_val.pkl").open("rb") as f:
        payload = pickle.load(f)
    return list(sorted(payload["infos"], key=lambda e: e["timestamp"]))


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def blend_between(anchor_images: list[Image.Image], anchor_times: list[float], t: float) -> Image.Image:
    if t <= anchor_times[0]:
        return anchor_images[0]
    if t >= anchor_times[-1]:
        return anchor_images[-1]
    for i in range(len(anchor_times) - 1):
        t0, t1 = anchor_times[i], anchor_times[i + 1]
        if t0 <= t <= t1:
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            alpha = smoothstep(u)
            return Image.blend(anchor_images[i], anchor_images[i + 1], alpha)
    return anchor_images[-1]


def compose_template1_video_frame(
    camera_panel: Image.Image,
    observation_panel: Image.Image,
    future_panel: Image.Image,
    legend_strip: Image.Image,
) -> Image.Image:
    margin = 22
    top_label_band = 52
    mid_label_band = 126
    bottom_margin = 28
    voxel_pad_top = 8

    target_bottom_width = camera_panel.width
    panel_gap = 18
    obs_w = int(target_bottom_width * 0.26)
    fut_w = target_bottom_width - panel_gap - obs_w
    aspect = future_panel.height / max(1, future_panel.width)
    bottom_h = max(300, int(fut_w * aspect))

    obs_panel = renderer.fit_panel(observation_panel, (obs_w, bottom_h), inner_scale=0.995)
    fut_panel = renderer.fit_panel(future_panel, (fut_w, bottom_h), inner_scale=0.995)

    total_w = camera_panel.width + 2 * margin
    total_h = top_label_band + camera_panel.height + mid_label_band + voxel_pad_top + bottom_h + bottom_margin
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_t = ImageFont.truetype("times.ttf", 34)
        font_s = ImageFont.truetype("times.ttf", 30)
    except Exception:
        font_t = font_s = ImageFont.load_default()

    obs_bbox = draw.textbbox((0, 0), "Observations", font=font_s)
    obs_x = margin + max(0, obs_panel.width // 2 - (obs_bbox[2] - obs_bbox[0]) // 2)
    draw.text((obs_x, 10), "Present", fill="black", font=font_t)

    canvas.paste(camera_panel, (margin, top_label_band))
    obs_label_y = top_label_band + camera_panel.height + 46
    draw.text((obs_x, obs_label_y), "Observations", fill="black", font=font_s)

    pf_text = "Predicted Futures"
    pf_bbox = draw.textbbox((0, 0), pf_text, font=font_s)
    pf_x = margin + obs_w + panel_gap + max(0, fut_w // 2 - (pf_bbox[2] - pf_bbox[0]) // 2)
    draw.text((pf_x, obs_label_y), pf_text, fill="black", font=font_s)

    legend_y = obs_label_y + 38
    canvas.paste(legend_strip, (margin, legend_y))

    voxel_y = top_label_band + camera_panel.height + mid_label_band + voxel_pad_top
    canvas.paste(obs_panel, (margin, voxel_y))
    canvas.paste(fut_panel, (margin + obs_w + panel_gap, voxel_y))
    return canvas


def build_camera_panel(repo_root: Path, info: dict[str, Any]) -> Image.Image:
    cam_paths = vismod.current_camera_paths_from_info(info)
    cam_imgs = vismod.load_camera_strip(cam_paths, repo_root)
    cam_labels = [Path(p).parent.name for p in cam_paths]
    return vismod.camera_grid(cam_imgs, cam_labels)


def render_anchor_panels(pred_temporal: torch.Tensor) -> list[Image.Image]:
    style = renderer.RendererStyle(
        dpi=300,
        backend="open3d",
        mode="voxel_mesh",
        viewpoint="top",
        render_width=2200,
        render_height=1400,
        open3d_point_size=6.8,
    )
    panels: list[Image.Image] = []
    for h in range(pred_temporal.shape[0]):
        panels.append(renderer.render_semantic_occ(pred_temporal[h], "", style))
    target_w = max(p.width for p in panels)
    target_h = max(p.height for p in panels)
    panels = [renderer.fit_panel(p, (target_w, target_h), inner_scale=0.995) for p in panels]
    return panels


def encode_mp4(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "16",
        "-preset",
        "slow",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    ensure_dirs()

    sample_index = args.sample_index
    repo_root = Path(args.repo_root).resolve()
    pred_temporal = load_pred_temporal(sample_index)
    sw2_row = load_sw2_manifest_row(sample_index)
    mapped_info_index = int(sw2_row["mapped_info_index"])
    infos = load_sparseworld_infos(repo_root)
    anchor_infos = infos[mapped_info_index : mapped_info_index + pred_temporal.shape[0]]
    base_ts = anchor_infos[0]["timestamp"]
    anchor_times = [float((info["timestamp"] - base_ts) / 1e6) for info in anchor_infos]

    camera_panel = build_camera_panel(repo_root, anchor_infos[0])
    anchor_panels = render_anchor_panels(pred_temporal)
    observation_panel = anchor_panels[0]
    legend_classes = ["car", "truck", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    legend_strip = renderer.build_legend_strip(legend_classes, width=camera_panel.width, height=58)
    legend_strip = ImageOps.contain(legend_strip, (camera_panel.width, 52))

    run_root = ARTIFACTS_DIR / f"sample_{sample_index:04d}_interpolated_future_mp4"
    frames_dir = run_root / "frames_540"
    frames_dir.mkdir(parents=True, exist_ok=True)

    total_semantic_seconds = float(anchor_times[-1] - anchor_times[0])
    frame_count = int(args.frame_count)
    for frame_idx in range(frame_count):
        if frame_count == 1:
            t = anchor_times[0]
        else:
            t = anchor_times[0] + total_semantic_seconds * (frame_idx / (frame_count - 1))
        future_panel = blend_between(anchor_panels, anchor_times, t)
        frame = compose_template1_video_frame(camera_panel, observation_panel, future_panel, legend_strip)
        frame.save(frames_dir / f"frame_{frame_idx:04d}.png")

    start_frame = FIGURES_DIR / f"paper_style_future_rollout_sample{sample_index}_start_frame.png"
    compose_template1_video_frame(camera_panel, observation_panel, anchor_panels[0], legend_strip).save(start_frame)

    mp4_path = FIGURES_DIR / f"paper_style_future_rollout_sample{sample_index}_interpolated_60fps_9s.mp4"
    encode_mp4(frames_dir, mp4_path, args.fps)

    manifest = {
        "sample_index": sample_index,
        "sample_token": sw2_row["sample_token"],
        "scene_token": sw2_row["scene_token"],
        "mapped_info_index": mapped_info_index,
        "anchor_times_seconds": anchor_times,
        "frame_count": frame_count,
        "semantic_span_seconds": total_semantic_seconds,
        "playback_fps": args.fps,
        "playback_duration_seconds": frame_count / args.fps,
        "actual_hz_visualization_target": args.actual_hz,
        "camera_panel_source": "sample current observations at t~-1s start",
        "future_source": "real SparseWorld sample_0003 future occupancy anchors with smooth image interpolation",
        "start_frame_path": str(start_frame),
        "frames_dir": str(frames_dir),
        "mp4_path": str(mp4_path),
    }
    write_json(REPORTS_DIR / f"paper_style_future_rollout_sample{sample_index}_interpolated_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
