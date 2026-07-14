from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis1_paper_style_visualization"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"
VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis1_paper_style_visualization"
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-indices", default="0,1,5,12,15")
    p.add_argument("--horizons", default="0,2,4,6")
    p.add_argument("--video-sample-index", type=int, default=3)
    p.add_argument("--video-frame-count", type=int, default=240)
    p.add_argument("--video-fps", type=int, default=60)
    p.add_argument("--encoder", default="h264_nvenc")
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_style() -> Any:
    return renderer.RendererStyle(
        dpi=300,
        backend="open3d" if getattr(renderer, "o3d", None) is not None else "matplotlib",
        mode="voxel_mesh" if getattr(renderer, "o3d", None) is not None else "point_cloud",
        viewpoint="bev_strict",
        render_width=2200,
        render_height=1400,
        open3d_point_size=6.8,
        bev_height_shading=True,
    )


def encode_mp4_from_pngs(frames_dir: Path, out_mp4: Path, fps: int, encoder: str) -> dict[str, Any]:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if encoder == "h264_nvenc":
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p", str(out_mp4),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", str(out_mp4),
        ]
    subprocess.run(cmd, check=True)
    return {"encoder_used": encoder, "output_path": str(out_mp4)}


def generate_rollout_and_composite(sample_indices: list[int], horizons: list[int], style: Any, repo_root: Path) -> dict[str, list[str]]:
    sample_manifest_full = {
        int(r["sample_index"]): r
        for r in vismod.read_csv(
            PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv"
        )
    }
    infos = vismod.load_sparseworld_info_list(repo_root)
    rollout_paths: list[str] = []
    composite_paths: list[str] = []

    for sample_index in sample_indices:
        pred_temporal, gt_temporal = vismod.load_occ_pair(sample_index)
        sample_label = f"Sample {sample_index} | template1 strict BEV"

        gt_panels = [renderer.render_semantic_occ(gt_temporal[h], "", style) for h in horizons]
        pred_panels = [renderer.render_semantic_occ(pred_temporal[h], "", style) for h in horizons]

        rollout_png = FIGURES_DIR / f"paper_style_gt_sparseworld_rollout_bev_strict_sample{sample_index}.png"
        rollout_pdf = FIGURES_DIR / f"paper_style_gt_sparseworld_rollout_bev_strict_sample{sample_index}.pdf"
        renderer.make_rollout_grid(
            gt_panels,
            pred_panels,
            [renderer.HORIZON_LABELS[h] for h in horizons],
            sample_label,
            rollout_png,
            rollout_pdf,
        )
        rollout_paths.append(str(rollout_png))

        mapped_info_index = int(sample_manifest_full[sample_index]["mapped_info_index"])
        cam_paths = vismod.current_camera_paths_from_info(infos[mapped_info_index])
        cam_imgs = vismod.load_camera_strip(cam_paths, repo_root)
        cam_labels = [Path(p).parent.name for p in cam_paths]
        cam_panel = vismod.camera_grid(cam_imgs, cam_labels)
        composite_path = FIGURES_DIR / f"paper_style_observation_future_composite_bev_strict_sample{sample_index}.png"
        vismod.compose_observation_future(
            cam_panel,
            pred_panels[0],
            pred_panels[1:],
            sample_label,
            [renderer.HORIZON_LABELS[h] for h in horizons],
            composite_path,
        )
        composite_paths.append(str(composite_path))

    return {"rollout_paths": rollout_paths, "composite_paths": composite_paths}


def compose_template1_video_frame(camera_panel: Image.Image, observation_panel: Image.Image, future_panel: Image.Image) -> Image.Image:
    margin = 22
    top_label_band = 52
    mid_label_band = 126
    bottom_margin = 28
    voxel_pad_top = 8
    panel_gap = 18
    target_bottom_width = camera_panel.width
    obs_w = int(target_bottom_width * 0.26)
    fut_w = target_bottom_width - panel_gap - obs_w
    aspect = future_panel.height / max(1, future_panel.width)
    bottom_h = max(300, int(fut_w * aspect))

    obs_panel = renderer.fit_panel(observation_panel, (obs_w, bottom_h), inner_scale=0.995)
    fut_panel = renderer.fit_panel(future_panel, (fut_w, bottom_h), inner_scale=0.995)

    total_w = camera_panel.width + 2 * margin
    total_h = top_label_band + camera_panel.height + mid_label_band + voxel_pad_top + bottom_h + bottom_margin
    canvas = Image.new("RGB", (total_w, total_h), "white")
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(canvas)
    try:
        font_t = ImageFont.truetype("times.ttf", 34)
        font_s = ImageFont.truetype("times.ttf", 30)
    except Exception:
        font_t = ImageFont.load_default()
        font_s = ImageFont.load_default()

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
    legend_classes = ["car", "truck", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    legend_strip = renderer.build_legend_strip(legend_classes, width=camera_panel.width, height=58)
    legend_strip = ImageOps.contain(legend_strip, (camera_panel.width, 52))
    canvas.paste(legend_strip, (margin, obs_label_y + 38))
    voxel_y = top_label_band + camera_panel.height + mid_label_band + voxel_pad_top
    canvas.paste(obs_panel, (margin, voxel_y))
    canvas.paste(fut_panel, (margin + obs_w + panel_gap, voxel_y))
    return canvas


def generate_template1_video_strict_bev(sample_index: int, frame_count: int, fps: int, encoder: str, style: Any, repo_root: Path) -> str:
    occ_dir = SWVIS4_ARTIFACTS_DIR / f"layered_occ_frames_stabilized/sample_{sample_index:04d}_{frame_count}f"
    if not occ_dir.exists():
        raise FileNotFoundError(f"missing occupancy frames: {occ_dir}")

    sample_manifest_full = {
        int(r["sample_index"]): r
        for r in vismod.read_csv(
            PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv"
        )
    }
    infos = vismod.load_sparseworld_info_list(repo_root)
    mapped_info_index = int(sample_manifest_full[sample_index]["mapped_info_index"])
    cam_paths = vismod.current_camera_paths_from_info(infos[mapped_info_index])
    cam_imgs = vismod.load_camera_strip(cam_paths, repo_root)
    cam_labels = [Path(p).parent.name for p in cam_paths]
    cam_panel = vismod.camera_grid(cam_imgs, cam_labels)

    pred_temporal, _ = vismod.load_occ_pair(sample_index)
    observation_panel = renderer.render_semantic_occ(pred_temporal[0], "", style)

    out_frames_dir = FIGURES_DIR / f"frames_template1_bev_strict/sample_{sample_index:04d}_{frame_count}f"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    occ_files = sorted(occ_dir.glob("occ_stab_frame_*.npy"))
    for i, occ_file in enumerate(occ_files):
        occ = torch.from_numpy(np.load(occ_file).astype(np.int64))
        future_panel = renderer.render_semantic_occ(occ, "", style)
        compose_template1_video_frame(cam_panel, observation_panel, future_panel).save(out_frames_dir / f"frame_{i:06d}.png")
    out_mp4 = VIDEOS_DIR / f"template1_sparseworld_endpoint_exact_layered_bev_strict_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    encode_mp4_from_pngs(out_frames_dir, out_mp4, fps, encoder)
    return str(out_mp4)


def generate_future_only_video_strict_bev(sample_index: int, frame_count: int, fps: int, encoder: str, style: Any) -> str:
    occ_dir = SWVIS4_ARTIFACTS_DIR / f"layered_occ_frames_stabilized/sample_{sample_index:04d}_{frame_count}f"
    if not occ_dir.exists():
        raise FileNotFoundError(f"missing occupancy frames: {occ_dir}")
    out_frames_dir = FIGURES_DIR / f"frames_future_only_bev_strict/sample_{sample_index:04d}_{frame_count}f"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    occ_files = sorted(occ_dir.glob("occ_stab_frame_*.npy"))
    for i, occ_file in enumerate(occ_files):
        occ = torch.from_numpy(np.load(occ_file).astype(np.int64))
        panel = renderer.render_semantic_occ(occ, "", style)
        renderer.fit_panel(panel, (style.render_width, style.render_height), inner_scale=1.0).save(out_frames_dir / f"frame_{i:06d}.png")
    out_mp4 = VIDEOS_DIR / f"template1_sparseworld_predicted_future_only_bev_strict_sample_{sample_index:04d}_{frame_count}f_{fps}fps.mp4"
    encode_mp4_from_pngs(out_frames_dir, out_mp4, fps, encoder)
    return str(out_mp4)


def main() -> int:
    args = parse_args()
    sample_indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    repo_root = Path(args.repo_root).resolve()
    style = build_style()

    figure_summary = generate_rollout_and_composite(sample_indices, horizons, style, repo_root)
    template1_video = generate_template1_video_strict_bev(args.video_sample_index, args.video_frame_count, args.video_fps, args.encoder, style, repo_root)
    future_only_video = generate_future_only_video_strict_bev(args.video_sample_index, args.video_frame_count, args.video_fps, args.encoder, style)

    manifest = {
        "template_name": "template1",
        "active_variant": "strict_bev",
        "strict_definition": "Template1 refers only to GT vs SparseWorld rollout grid figures, observation-plus-future composite figures, and exactly two videos: template1 endpoint-exact layered video and pure predicted-future video.",
        "render_style": {
            "viewpoint": "bev_strict",
            "color_style": "original semantic palette",
            "camera_panel": "unchanged 2x3 nuScenes camera grid",
        },
        "figure_assets": {
            "gt_vs_sparseworld_rollout": figure_summary["rollout_paths"],
            "observation_future_composite": figure_summary["composite_paths"],
        },
        "video_assets": [
            {
                "name": "endpoint_exact_layered_bev_strict",
                "sample_index": args.video_sample_index,
                "frame_count": args.video_frame_count,
                "fps": args.video_fps,
                "path": template1_video,
            },
            {
                "name": "predicted_future_only_bev_strict",
                "sample_index": args.video_sample_index,
                "frame_count": args.video_frame_count,
                "fps": args.video_fps,
                "path": future_only_video,
            },
        ],
    }
    write_json(REPORTS_DIR / "paper_style_template1_manifest.json", manifest)
    write_json(REPORTS_DIR / "paper_style_template1_strict_bev_generation_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
