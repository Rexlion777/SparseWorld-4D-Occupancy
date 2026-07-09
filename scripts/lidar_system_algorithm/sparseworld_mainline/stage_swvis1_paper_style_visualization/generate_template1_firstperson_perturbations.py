from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis1_paper_style_visualization"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"
VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis1_paper_style_visualization"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_swvis1_paper_style_visualization"
SW6_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
SW7_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vismod = load_module("swvis1_pert_main", SCRIPT_DIR / "run_sparseworld_paper_style_visualization.py")
renderer = load_module("swvis1_pert_renderer", SCRIPT_DIR / "semantic_occupancy_renderer.py")
video_comp = load_module("swvis1_pert_video", SCRIPT_DIR / "generate_interpolated_future_mp4.py")
swvis4_lib = load_module(
    "swvis4_interp_lib_for_pert",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/swvis4_support_interp_lib.py",
)
swvis4_fix = load_module(
    "swvis4_endpoint_fix_for_pert",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/run_sparseworld_swvis4_endpoint_exact_fix.py",
)


PERT_ORDER = [
    ("A0_clean", 0),
    ("A1_drop_cam_front", 0),
    ("A10_drop_front_triplet", 0),
    ("C4_motion_blur_9", 0),
    ("A7_drop_all_rear", 0),
]
ROLLOUT_HORIZONS = [0, 2, 4, 6]
VIDEO_FRAME_COUNT = 240
VIDEO_FPS = 60


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_manifest() -> dict[int, dict[str, str]]:
    rows = read_csv_rows(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv")
    return {int(r["sample_index"]): r for r in rows}


def load_gt(sample_index: int) -> torch.Tensor:
    return torch.load(
        PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/standard_occ/sample_{sample_index:04d}_standard_gt_occ_temporal.pt",
        map_location="cpu",
        weights_only=False,
    ).long()


def load_pred(perturbation_id: str, sample_index: int) -> torch.Tensor:
    obj = torch.load(
        SW6_ARTIFACTS / f"occ_artifacts/{perturbation_id}/sample_{sample_index:04d}_occ_temporal.pt",
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(obj, dict):
        if "pred_temporal" in obj:
            return obj["pred_temporal"].long()
        if "semantic_occ" in obj:
            return obj["semantic_occ"].long()
        for v in obj.values():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                return v.long()
    if isinstance(obj, torch.Tensor):
        return obj.long()
    raise RuntimeError(f"unable to load perturbed pred temporal for {perturbation_id} sample {sample_index}")


def load_query_artifact(perturbation_id: str, sample_index: int) -> dict[str, Any]:
    return torch.load(
        SW6_ARTIFACTS / f"query_artifacts/{perturbation_id}/sample_{sample_index:04d}_query_artifact.pt",
        map_location="cpu",
        weights_only=False,
    )


def ffmpeg_encoder() -> str:
    try:
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        if "h264_nvenc" in probe.stdout:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def encode_mp4(frames_dir: Path, out_mp4: Path, fps: int) -> str:
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


def swvis4_style() -> Any:
    return renderer.RendererStyle(
        dpi=300,
        backend="open3d",
        mode="voxel_mesh",
        viewpoint="top",
        render_width=1400,
        render_height=900,
        open3d_point_size=6.8,
        shadow_alpha=54,
        shadow_blur_radius=5.2,
    )


def render_panel(grid: torch.Tensor) -> Image.Image:
    return renderer.render_semantic_occ(grid, "", swvis4_style())


def selected_base_cfg() -> dict[str, Any]:
    classwise_static_thr = np.array([0.12] * int(swvis4_fix.NUM_CLASSES), dtype=np.float32)
    classwise_static_thr[11] = 0.06
    classwise_static_thr[13] = 0.06
    classwise_static_thr[14] = 0.06
    classwise_static_thr[15] = 0.05
    classwise_static_thr[16] = 0.05
    return {
        "global_occ_threshold": 0.12,
        "classwise_occ_threshold": classwise_static_thr,
        "aggregation": "sum_prob",
        "splat": "trilinear",
    }


def build_support_keyframes_from_fb(fb: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for h in range(7):
        pts, cls = (fb["refine_pts"], fb["cls_score"]) if h == 0 else (fb["forecast_points_list"][h - 1], fb["forecast_semantics_list"][h - 1])
        out[h] = swvis4_lib.build_support_keyframe_from_tensors(pts, cls, h, float(swvis4_lib.ANCHOR_TIMES[h]))
    return out


def build_interval_matches(keyframes: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    matches: dict[int, dict[str, Any]] = {}
    for h in range(6):
        a = keyframes[h]
        b = keyframes[h + 1]
        idx_quality = swvis4_lib.compute_index_alignment_quality(a, b, distance_threshold=6.0)
        if idx_quality["same_query_count"] and idx_quality["index_usable"]:
            match = swvis4_lib.build_index_aligned_match(a, b, distance_threshold=6.0)
        else:
            match = swvis4_lib.match_queries(a, b, distance_threshold=6.0)
        matches[h] = match
    return matches


def build_endpoint_residuals_local(native_occ: dict[int, np.ndarray], base_recon_cache: dict[int, np.ndarray]) -> dict[int, dict[str, Any]]:
    residuals: dict[int, dict[str, Any]] = {}
    for h in range(7):
        raw_occ = native_occ[h].astype(np.uint8)
        recon_occ = base_recon_cache[h].astype(np.uint8)
        raw_mask = raw_occ != swvis4_fix.EMPTY_IDX
        recon_mask = recon_occ != swvis4_fix.EMPTY_IDX
        missing_mask = raw_mask & ~recon_mask
        class_fix_mask = raw_mask & recon_mask & (raw_occ != recon_occ)
        add_mask = missing_mask | class_fix_mask
        extra_mask = recon_mask & ~raw_mask
        add_label = np.full_like(raw_occ, swvis4_fix.EMPTY_IDX, dtype=np.uint8)
        add_label[add_mask] = raw_occ[add_mask]
        extra_source_class = np.full_like(raw_occ, swvis4_fix.EMPTY_IDX, dtype=np.uint8)
        extra_source_class[extra_mask] = recon_occ[extra_mask]
        residuals[h] = {
            "add_label": add_label,
            "extra_mask": extra_mask,
            "extra_source_class": extra_source_class,
        }
    return residuals


def render_swvis4_chain_frames(
    perturbation_id: str,
    sample_index: int,
    cam_panel: Image.Image,
    observation_panel: Image.Image,
) -> tuple[Path, Path, str, str]:
    pred = load_pred(perturbation_id, sample_index).cpu().numpy().astype(np.uint8)
    fb = load_query_artifact(perturbation_id, sample_index)["forward_backbone_outputs"]
    keyframes = build_support_keyframes_from_fb(fb)
    matches = build_interval_matches(keyframes)

    native_occ = {h: pred[h] for h in range(7)}
    base_cfg = selected_base_cfg()
    base_recon_cache: dict[int, np.ndarray] = {}
    for h in range(7):
        flat = swvis4_lib.flatten_support_keyframe(keyframes[h])
        base_recon_cache[h] = swvis4_fix.reconstruct_surrogate_from_flat_support(
            flat,
            global_occ_threshold=float(base_cfg["global_occ_threshold"]),
            classwise_occ_threshold=base_cfg["classwise_occ_threshold"],
            aggregation=str(base_cfg["aggregation"]),
            splat=str(base_cfg["splat"]),
        )
    residuals = build_endpoint_residuals_local(native_occ, base_recon_cache)

    occ_frames_dir = ARTIFACTS_DIR / f"template1_firstperson_{perturbation_id}_layered_occ_sample_{sample_index:04d}_240f"
    occ_frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(VIDEO_FRAME_COUNT):
        h0, h1, alpha = swvis4_lib.frame_interval_for_index(frame_idx, VIDEO_FRAME_COUNT)
        support = swvis4_lib.interpolate_interval_frame(keyframes[h0], keyframes[h1], matches[h0], alpha)
        base_occ = swvis4_fix.reconstruct_surrogate_from_flat_support(
            support,
            global_occ_threshold=float(base_cfg["global_occ_threshold"]),
            classwise_occ_threshold=base_cfg["classwise_occ_threshold"],
            aggregation=str(base_cfg["aggregation"]),
            splat=str(base_cfg["splat"]),
        )
        layered = swvis4_fix.build_layered_frame(base_occ, residuals[h0], residuals[h1], alpha)
        np.save(occ_frames_dir / f"occ_frame_{frame_idx:06d}.npy", layered.astype(np.uint8))

    stab_dir = ARTIFACTS_DIR / f"template1_firstperson_{perturbation_id}_layered_occ_stabilized_sample_{sample_index:04d}_240f"
    swvis4_lib.stabilize_occ_sequence(occ_frames_dir, stab_dir, VIDEO_FRAME_COUNT, progress_cb=None, progress_every=30)

    legend_classes = ["car", "truck", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    legend_strip = renderer.build_legend_strip(legend_classes, width=cam_panel.width, height=58)
    legend_strip = ImageOps.contain(legend_strip, (cam_panel.width, 52))

    comp_frames_dir = FIGURES_DIR / f"template1_firstperson_{perturbation_id}_video_frames_sample_{sample_index:04d}_240f"
    pure_frames_dir = FIGURES_DIR / f"template1_firstperson_{perturbation_id}_future_only_frames_sample_{sample_index:04d}_240f"
    comp_frames_dir.mkdir(parents=True, exist_ok=True)
    pure_frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(VIDEO_FRAME_COUNT):
        occ = torch.from_numpy(np.load(stab_dir / f"occ_stab_frame_{frame_idx:06d}.npy").astype(np.int64))
        future_panel = renderer.fit_panel(render_panel(occ), (1400, 900), inner_scale=0.995)
        frame = video_comp.compose_template1_video_frame(cam_panel, observation_panel, future_panel, legend_strip)
        frame.save(comp_frames_dir / f"frame_{frame_idx:04d}.png")
        future_panel.save(pure_frames_dir / f"frame_{frame_idx:04d}.png")

    composite_mp4 = VIDEOS_DIR / f"template1_firstperson_{perturbation_id}_observation_future_sample_{sample_index:04d}_240f_60fps.mp4"
    future_only_mp4 = VIDEOS_DIR / f"template1_firstperson_{perturbation_id}_predicted_future_only_sample_{sample_index:04d}_240f_60fps.mp4"
    enc_a = encode_mp4(comp_frames_dir, composite_mp4, VIDEO_FPS)
    enc_b = encode_mp4(pure_frames_dir, future_only_mp4, VIDEO_FPS)
    return composite_mp4, future_only_mp4, enc_a, enc_b


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


def perturb_camera_panel(cam_panel: Image.Image, cam_labels: list[str], perturbation_id: str) -> Image.Image:
    w = max(im.width for im in []) if False else None
    tile_w = (cam_panel.width - 4 * 10) // 3
    tile_h = (cam_panel.height - 3 * 10) // 2
    margin = 10
    affected = set()
    if perturbation_id == "A1_drop_cam_front":
        affected = {"CAM_FRONT"}
    elif perturbation_id == "A10_drop_front_triplet":
        affected = {"CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"}
    elif perturbation_id == "A7_drop_all_rear":
        affected = {"CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"}

    out = Image.new("RGB", cam_panel.size, "white")
    for i, label in enumerate(cam_labels):
        r, c = divmod(i, 3)
        x = margin + c * (tile_w + margin)
        y = margin + r * (tile_h + margin)
        tile = cam_panel.crop((x, y, x + tile_w, y + tile_h))
        if perturbation_id in {"A1_drop_cam_front", "A10_drop_front_triplet", "A7_drop_all_rear"} and label in affected:
            tile = Image.new("RGB", tile.size, (24, 24, 24))
        elif perturbation_id == "C4_motion_blur_9":
            tile = tile.filter(ImageFilter.GaussianBlur(radius=2.2))
        out.paste(tile, (x, y))
    return out


def camera_panel_for_sample(sample_index: int, perturbation_id: str, repo_root: Path) -> tuple[Image.Image, list[str]]:
    manifest = sample_manifest()
    infos = vismod.load_sparseworld_info_list(repo_root)
    mapped_info_index = int(manifest[sample_index]["mapped_info_index"])
    info = infos[mapped_info_index]
    cam_paths = vismod.current_camera_paths_from_info(info)
    cam_imgs = vismod.load_camera_strip(cam_paths, repo_root)
    cam_labels = [Path(p).parent.name for p in cam_paths]
    cam_panel = vismod.camera_grid(cam_imgs, cam_labels)
    return perturb_camera_panel(cam_panel, cam_labels, perturbation_id), cam_labels


def sample_label(sample_index: int, perturbation_id: str) -> str:
    meta = sample_manifest()[sample_index]
    return f"{perturbation_id} | sample {sample_index} | scene {meta['scene_token']}"


def get_anchor_times(sample_index: int, repo_root: Path) -> list[float]:
    meta = sample_manifest()[sample_index]
    infos = vismod.load_sparseworld_info_list(repo_root)
    mapped_info_index = int(meta["mapped_info_index"])
    anchor_infos = infos[mapped_info_index : mapped_info_index + 7]
    base_ts = anchor_infos[0]["timestamp"]
    return [float((info["timestamp"] - base_ts) / 1e6) for info in anchor_infos]


def generate_one(repo_root: Path, perturbation_id: str, sample_index: int) -> dict[str, Any]:
    pred = load_pred(perturbation_id, sample_index)
    gt = load_gt(sample_index)
    gt_panels = [render_panel(gt[h]) for h in ROLLOUT_HORIZONS]
    pred_panels = [render_panel(pred[h]) for h in ROLLOUT_HORIZONS]
    label = sample_label(sample_index, perturbation_id)

    rollout_png = FIGURES_DIR / f"template1_firstperson_{perturbation_id}_gt_vs_sparseworld_rollout_sample{sample_index}.png"
    rollout_pdf = FIGURES_DIR / f"template1_firstperson_{perturbation_id}_gt_vs_sparseworld_rollout_sample{sample_index}.pdf"
    renderer.make_rollout_grid(
        gt_panels,
        pred_panels,
        [renderer.HORIZON_LABELS[h] for h in ROLLOUT_HORIZONS],
        label,
        rollout_png,
        rollout_pdf,
    )

    cam_panel, _ = camera_panel_for_sample(sample_index, perturbation_id, repo_root)
    composite_png = FIGURES_DIR / f"template1_firstperson_{perturbation_id}_observation_future_composite_sample{sample_index}.png"
    vismod.compose_observation_future(
        cam_panel,
        pred_panels[0],
        pred_panels[1:],
        label,
        [renderer.HORIZON_LABELS[h] for h in ROLLOUT_HORIZONS],
        composite_png,
    )

    observation_panel = pred_panels[0]
    composite_mp4, future_only_mp4, enc_a, enc_b = render_swvis4_chain_frames(
        perturbation_id,
        sample_index,
        cam_panel,
        observation_panel,
    )

    return {
        "perturbation_id": perturbation_id,
        "sample_index": sample_index,
        "figure_assets": {
            "gt_vs_sparseworld_rollout": str(rollout_png),
            "observation_future_composite": str(composite_png),
        },
        "video_assets": {
            "template1_composite_video": str(composite_mp4),
            "predicted_future_only_video": str(future_only_mp4),
        },
        "encoders": {
            "template1_composite_video": enc_a,
            "predicted_future_only_video": enc_b,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--perturbation-id", action="append", default=[], help="Generate only the specified perturbation id. Can be passed multiple times.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = PROJECT_ROOT / "external/SparseWorld"
    selected = set(args.perturbation_id or [])
    pert_rows = [(pid, sidx) for pid, sidx in PERT_ORDER if not selected or pid in selected]
    rows = [generate_one(repo_root, perturbation_id, sample_index) for perturbation_id, sample_index in pert_rows]
    payload = {
        "template_name": "template1_firstperson_perturbation_bundle",
        "strict_definition": "For each perturbation: 2 figures (GT vs SparseWorld rollout, observation/future composite) + 2 videos (template1 composite, pure predicted future).",
        "requested_labels": [pid.split("_")[0] for pid, _ in pert_rows],
        "note": "Current repo assets contain A7_drop_all_rear control and do not contain a C7 perturbation id.",
        "rows": rows,
    }
    write_json(REPORTS_DIR / "paper_style_template1_firstperson_perturbation_manifest.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
