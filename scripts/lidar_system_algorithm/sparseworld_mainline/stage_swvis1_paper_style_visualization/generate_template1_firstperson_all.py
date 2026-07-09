from __future__ import annotations

import csv
import importlib.util
import json
import math
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vismod = load_module("swvis1_front_main", SCRIPT_DIR / "run_sparseworld_paper_style_visualization.py")
renderer = load_module("swvis1_front_renderer", SCRIPT_DIR / "semantic_occupancy_renderer.py")
video_comp = load_module("swvis1_front_video", SCRIPT_DIR / "generate_interpolated_future_mp4.py")


SAMPLES = [0, 1, 5, 12, 15]
ROLLOUT_HORIZONS = [0, 2, 4, 6]
VIDEO_SAMPLE = 3


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ffmpeg_encoder() -> str:
    try:
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        if "h264_nvenc" in probe.stdout:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def encode_mp4(frames_dir: Path, out_mp4: Path, fps: int = 60) -> str:
    enc = ffmpeg_encoder()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png")]
    if enc == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"]
    cmd.append(str(out_mp4))
    subprocess.run(cmd, check=True)
    return enc


def load_info_rows() -> dict[int, dict[str, str]]:
    rows = read_csv_rows(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv")
    return {int(r["sample_index"]): r for r in rows}


def load_pred_gt(sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    pred, gt = vismod.load_occ_pair(sample_index)
    return pred.long(), gt.long()


def front_style() -> Any:
    return renderer.RendererStyle(
        dpi=300,
        backend="open3d",
        mode="voxel_mesh",
        viewpoint="front",
        render_width=2000,
        render_height=1200,
        open3d_point_size=6.8,
        shadow_alpha=54,
        shadow_blur_radius=5.2,
    )


def render_panel(grid: torch.Tensor) -> Image.Image:
    return renderer.render_semantic_occ(grid, "", front_style())


def current_camera_panel(sample_index: int, repo_root: Path) -> Image.Image:
    manifest = load_info_rows()
    infos = vismod.load_sparseworld_info_list(repo_root)
    mapped_info_index = int(manifest[sample_index]["mapped_info_index"])
    cam_paths = vismod.current_camera_paths_from_info(infos[mapped_info_index])
    cam_imgs = vismod.load_camera_strip(cam_paths, repo_root)
    cam_labels = [Path(p).parent.name for p in cam_paths]
    return vismod.camera_grid(cam_imgs, cam_labels)


def generate_static_assets(repo_root: Path) -> dict[str, list[str]]:
    figure_assets: dict[str, list[str]] = {"gt_vs_sparseworld_rollout_front": [], "observation_future_composite_front": []}
    manifest_rows = load_info_rows()
    for sample_index in SAMPLES:
        pred, gt = load_pred_gt(sample_index)
        gt_panels = [render_panel(gt[h]) for h in ROLLOUT_HORIZONS]
        pred_panels = [render_panel(pred[h]) for h in ROLLOUT_HORIZONS]
        sample_label = f"sample {sample_index} | scene {manifest_rows[sample_index]['scene_token']}"
        rollout_png = FIGURES_DIR / f"paper_style_gt_sparseworld_rollout_front_sample{sample_index}.png"
        rollout_pdf = FIGURES_DIR / f"paper_style_gt_sparseworld_rollout_front_sample{sample_index}.pdf"
        renderer.make_rollout_grid(
            gt_panels,
            pred_panels,
            [renderer.HORIZON_LABELS[h] for h in ROLLOUT_HORIZONS],
            sample_label,
            rollout_png,
            rollout_pdf,
        )
        figure_assets["gt_vs_sparseworld_rollout_front"].append(str(rollout_png))

        cam_panel = current_camera_panel(sample_index, repo_root)
        composite_path = FIGURES_DIR / f"paper_style_observation_future_composite_front_sample{sample_index}.png"
        vismod.compose_observation_future(
            cam_panel,
            pred_panels[0],
            pred_panels[1:],
            sample_label,
            [renderer.HORIZON_LABELS[h] for h in ROLLOUT_HORIZONS],
            composite_path,
        )
        figure_assets["observation_future_composite_front"].append(str(composite_path))
    return figure_assets


def load_video_occ_frames() -> list[Path]:
    layered_summary = read_json(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/layered_interpolation_summary.json")
    frame_dir = Path(layered_summary["stabilized_occ_dir"])
    return sorted(frame_dir.glob("occ_stab_frame_*.npy"))


def generate_video_assets(repo_root: Path) -> list[dict[str, Any]]:
    frames = load_video_occ_frames()
    if not frames:
        raise RuntimeError("no stabilized occ frames found for template1 front-view video export")
    cam_panel = current_camera_panel(VIDEO_SAMPLE, repo_root)
    legend_classes = ["car", "truck", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    legend_strip = renderer.build_legend_strip(legend_classes, width=cam_panel.width, height=58)
    legend_strip = ImageOps.contain(legend_strip, (cam_panel.width, 52))
    observation_occ = torch.from_numpy(np.load(frames[0])).long()
    observation_panel = render_panel(observation_occ)

    comp_frames_dir = FIGURES_DIR / f"template1_front_video_frames_sample_{VIDEO_SAMPLE:04d}_240f"
    pure_frames_dir = FIGURES_DIR / f"template1_front_future_only_frames_sample_{VIDEO_SAMPLE:04d}_240f"
    comp_frames_dir.mkdir(parents=True, exist_ok=True)
    pure_frames_dir.mkdir(parents=True, exist_ok=True)

    target_panel_size = (1600, 900)
    for idx, occ_path in enumerate(frames):
        occ = torch.from_numpy(np.load(occ_path)).long()
        fut_panel = renderer.fit_panel(render_panel(occ), target_panel_size, inner_scale=0.995)
        frame = video_comp.compose_template1_video_frame(cam_panel, observation_panel, fut_panel, legend_strip)
        frame.save(comp_frames_dir / f"frame_{idx:04d}.png")
        fut_panel.save(pure_frames_dir / f"frame_{idx:04d}.png")

    comp_mp4 = VIDEOS_DIR / f"template1_sparseworld_endpoint_exact_layered_front_sample_{VIDEO_SAMPLE:04d}_240f_60fps.mp4"
    pure_mp4 = VIDEOS_DIR / f"template1_sparseworld_predicted_future_only_front_sample_{VIDEO_SAMPLE:04d}_240f_60fps.mp4"
    enc_a = encode_mp4(comp_frames_dir, comp_mp4, fps=60)
    enc_b = encode_mp4(pure_frames_dir, pure_mp4, fps=60)
    return [
        {
            "name": "endpoint_exact_layered_front",
            "sample_index": VIDEO_SAMPLE,
            "frame_count": len(frames),
            "fps": 60,
            "encoder": enc_a,
            "path": str(comp_mp4),
        },
        {
            "name": "predicted_future_only_front",
            "sample_index": VIDEO_SAMPLE,
            "frame_count": len(frames),
            "fps": 60,
            "encoder": enc_b,
            "path": str(pure_mp4),
        },
    ]


def main() -> int:
    repo_root = PROJECT_ROOT / "external/SparseWorld"
    figure_assets = generate_static_assets(repo_root)
    video_assets = generate_video_assets(repo_root)
    payload = {
        "template_name": "template1_firstperson",
        "based_on": str(REPORTS_DIR / "paper_style_template1_manifest.json"),
        "viewpoint": "front",
        "static_samples": SAMPLES,
        "video_sample": VIDEO_SAMPLE,
        "figure_assets": figure_assets,
        "video_assets": video_assets,
        "note": "Front-view regeneration of the existing template1 asset family using the same semantic voxel palette and first-person camera pose.",
    }
    write_json(REPORTS_DIR / "paper_style_template1_firstperson_manifest.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
