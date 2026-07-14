from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_swvis1_paper_style_visualization"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis1_paper_style_visualization"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_swvis1_paper_style_visualization"
NUSCENES_FALLBACK_ROOTS = [
    Path(r"D:\cv_lidar_transition_assets\occupancy_real_baseline\data\nuscenes"),
    PROJECT_ROOT / "data/nuscenes",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw2 = load_module(
    "sw2_vis_stage",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
renderer = load_module(
    "swvis_renderer",
    SCRIPT_DIR / "semantic_occupancy_renderer.py",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-indices", default="")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--horizons", default="0,2,4,6")
    p.add_argument("--render-gt", action="store_true", default=True)
    p.add_argument("--render-pred", action="store_true", default=True)
    p.add_argument("--render-error", action="store_true", default=True)
    p.add_argument("--render-camera-panel", action="store_true", default=True)
    p.add_argument("--render-gif", action="store_true", default=True)
    p.add_argument("--backend", default="open3d")
    p.add_argument("--fallback-backend", default="matplotlib")
    p.add_argument("--save-highres", action="store_true", default=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    p.add_argument("--config", default=str(PROJECT_ROOT / "external/SparseWorld/configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"))
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "external/SparseWorld/ckpts/epoch_56.pth"))
    return p.parse_args()


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, LOGS_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def path_exists_safe(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def load_occ_pair(sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.load(
        PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/standard_occ/sample_{sample_index:04d}_standard_pred_occ_temporal.pt",
        map_location="cpu",
        weights_only=False,
    )
    gt = torch.load(
        PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/standard_occ/sample_{sample_index:04d}_standard_gt_occ_temporal.pt",
        map_location="cpu",
        weights_only=False,
    )
    return pred.long(), gt.long()


def sample_metric_rows(sample_index: int, per_sample_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in per_sample_rows if int(r["sample_index"]) == sample_index]


def class_diversity(gt_temporal: torch.Tensor) -> int:
    vals = sorted({int(x) for x in torch.unique(gt_temporal[0]).tolist() if 0 <= int(x) < len(renderer.OCC_NAMES)})
    vals = [v for v in vals if v != renderer.EMPTY_IDX]
    return len(vals)


def dynamic_presence(gt_temporal: torch.Tensor) -> int:
    ids = set(sw2.CLASS_GROUPS["all_dynamic"])
    return int(sum(int((gt_temporal[0] == cls).sum().item()) for cls in ids))


def small_object_presence(gt_temporal: torch.Tensor) -> int:
    ids = set(sw2.CLASS_GROUPS["small_object"])
    return int(sum(int((gt_temporal[0] == cls).sum().item()) for cls in ids))


def select_samples(num_samples: int) -> list[dict[str, Any]]:
    per_sample = read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/temporal_horizon_metrics_per_sample.csv")
    sample_manifest = read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv")
    sample_info = {int(r["sample_index"]): r for r in sample_manifest}
    candidates: list[dict[str, Any]] = []
    for sample_index in sorted(sample_info.keys()):
        pred, gt = load_occ_pair(sample_index)
        rows = sample_metric_rows(sample_index, per_sample)
        by_h = {int(r["horizon_s"]): r for r in rows}
        if 6 not in by_h or 0 not in by_h:
            continue
        candidates.append(
            {
                "sample_index": sample_index,
                "sample_token": sample_info[sample_index]["sample_token"],
                "scene_token": sample_info[sample_index]["scene_token"],
                "scene_name": sample_info[sample_index]["scene_name"],
                "occupied_iou_h6": float(by_h[6]["occupied_iou"]),
                "false_free_h6": float(by_h[6]["false_free_rate"]),
                "false_occupied_h6": float(by_h[6]["false_occupied_rate"]),
                "pred_gt_ratio_h6": float(by_h[6]["pred_gt_occupied_ratio"]),
                "class_diversity": class_diversity(gt),
                "dynamic_presence": dynamic_presence(gt),
                "small_object_presence": small_object_presence(gt),
            }
        )
    candidates.sort(key=lambda x: (x["class_diversity"], x["dynamic_presence"]), reverse=True)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    def add_one(row: dict[str, Any], reason: str) -> None:
        if row["sample_index"] in used:
            return
        row = dict(row)
        row["selection_reason"] = reason
        selected.append(row)
        used.add(row["sample_index"])

    if candidates:
        add_one(max(candidates, key=lambda x: x["occupied_iou_h6"]), "best-looking successful rollout with high semantic occupancy quality")
        add_one(max(candidates, key=lambda x: x["false_free_h6"]), "representative high false-free case")
        add_one(max(candidates, key=lambda x: x["false_occupied_h6"]), "representative high false-occupied case")
        add_one(max(candidates, key=lambda x: x["small_object_presence"]), "small-object-rich sample")
        add_one(max(candidates, key=lambda x: x["dynamic_presence"]), "dynamic-object-rich sample")
    for row in candidates:
        if len(selected) >= num_samples:
            break
        add_one(row, "high class diversity clean baseline case")
    return selected[:num_samples]


def load_sparseworld_info_list(repo_root: Path) -> list[dict[str, Any]]:
    info_path = repo_root / "data/nuscenes/bevdetv2-nuscenes_infos_val.pkl"
    with info_path.open("rb") as f:
        payload = pickle.load(f)
    return list(sorted(payload["infos"], key=lambda e: e["timestamp"]))


def build_dataset_only(repo_root: Path, config_path: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mmcv import Config
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import patch_config

    old_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        cfg = Config.fromfile(str(config_path))
        cfg = compat_cfg(cfg)
        cfg = patch_config(cfg)
        setup_multi_processes(cfg)
        split_cfg = cfg.data["val"]
        split_cfg.test_mode = True
        if cfg.data.get("val_dataloader", {}).get("samples_per_gpu", 1) > 1:
            split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)
        dataset = build_dataset(split_cfg)
        return cfg, dataset
    finally:
        os.chdir(old_cwd)


def unwrap(value: Any) -> Any:
    try:
        from mmcv.parallel import DataContainer
    except Exception:
        DataContainer = None
    if DataContainer is not None and isinstance(value, DataContainer):
        return unwrap(value.data)
    if isinstance(value, list):
        if len(value) == 1:
            return unwrap(value[0])
        return [unwrap(v) for v in value]
    if isinstance(value, tuple):
        return [unwrap(v) for v in value]
    if isinstance(value, dict):
        return {k: unwrap(v) for k, v in value.items()}
    return value


def current_camera_paths(dataset: Any, sample_index: int) -> list[str]:
    if hasattr(dataset, "data_infos"):
        info = dataset.data_infos[sample_index]
        cam_order = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        cams = info.get("cams", {})
        return [str(cams[name]["data_path"]) for name in cam_order if name in cams]
    sample = unwrap(dataset[sample_index])
    filenames = list(sample["img_metas"]["filename"])
    return filenames[:6]


def current_camera_paths_from_info(info: dict[str, Any]) -> list[str]:
    cam_order = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    cams = info.get("cams", {})
    return [str(cams[name]["data_path"]) for name in cam_order if name in cams]


def load_camera_strip(paths: list[str], repo_root: Path) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for p in paths:
        if p.startswith("./data/nuscenes/"):
            rel = Path(p.replace("./data/nuscenes/", ""))
            candidates = [repo_root / "data/nuscenes" / rel, *[root / rel for root in NUSCENES_FALLBACK_ROOTS]]
            full = next((c for c in candidates if path_exists_safe(c)), candidates[-1])
        else:
            full = (repo_root / p).resolve() if p.startswith("./") else Path(p)
        img = Image.open(full).convert("RGB")
        img = ImageOps.contain(img, (600, 340))
        imgs.append(img)
    return imgs


def camera_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    margin = 10
    canvas = Image.new("RGB", (3 * w + 4 * margin, 2 * h + 3 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, 3)
        x = margin + c * (w + margin)
        y = margin + r * (h + margin)
        canvas.paste(img, (x, y))
        draw.text((x + 8, y + 8), label, fill="white", font=font, stroke_fill="black", stroke_width=2)
    return canvas


def compose_observation_future(
    camera_panel: Image.Image,
    current_panel: Image.Image,
    future_panels: list[Image.Image],
    sample_label: str,
    time_labels: list[str],
    out_path: Path,
) -> None:
    margin = 22
    top_label_band = 52
    mid_label_band = 126
    time_label_band = 44
    voxel_pad_top = 8
    all_panels = [current_panel, *future_panels]
    target_bottom_width = camera_panel.width
    panel_gap = 18
    panel_w = int((target_bottom_width - panel_gap * (len(all_panels) - 1)) / len(all_panels))
    aspect = all_panels[0].height / max(1, all_panels[0].width)
    bottom_h = max(260, int(panel_w * aspect))
    fitted_panels = [renderer.fit_panel(p, (panel_w, bottom_h), inner_scale=0.995) for p in all_panels]
    current_panel = fitted_panels[0]
    future_panels = fitted_panels[1:]
    bottom_strip = Image.new("RGB", (target_bottom_width, bottom_h), "white")
    x = 0
    panel_xs: list[int] = []
    for p in fitted_panels:
        panel_xs.append(x)
        bottom_strip.paste(p, (x, 0))
        x += p.width + panel_gap
    total_w = camera_panel.width + 2 * margin
    total_h = top_label_band + camera_panel.height + mid_label_band + voxel_pad_top + bottom_h + time_label_band + margin + 36
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_t = ImageFont.truetype("times.ttf", 34)
        font_s = ImageFont.truetype("times.ttf", 30)
        font_m = ImageFont.truetype("times.ttf", 24)
        font_time = ImageFont.truetype("times.ttf", 24)
        font_note = ImageFont.truetype("times.ttf", 16)
    except Exception:
        font_t = font_s = font_m = font_time = font_note = ImageFont.load_default()
    obs_bbox = draw.textbbox((0, 0), "Observations", font=font_s)
    obs_x = margin + max(0, current_panel.width // 2 - (obs_bbox[2] - obs_bbox[0]) // 2)
    present_bbox = draw.textbbox((0, 0), "Present", font=font_t)
    present_x = obs_x
    draw.text((present_x, 10), "Present", fill="black", font=font_t)
    draw.text((margin, total_h - 22), sample_label, fill="#666666", font=font_note)
    canvas.paste(camera_panel, (margin, top_label_band))
    y2 = top_label_band + camera_panel.height + mid_label_band
    obs_label_y = top_label_band + camera_panel.height + 46
    pred_label_y = obs_label_y
    draw.text((obs_x, obs_label_y), "Observations", fill="black", font=font_s)
    pf_text = "Predicted Futures"
    pf_bbox = draw.textbbox((0, 0), pf_text, font=font_s)
    future_group_w = target_bottom_width - current_panel.width - panel_gap
    pf_x = margin + current_panel.width + panel_gap + max(0, future_group_w // 2 - (pf_bbox[2] - pf_bbox[0]) // 2)
    draw.text((pf_x, pred_label_y), pf_text, fill="black", font=font_s)
    legend_classes = ["car", "truck", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    legend_strip = renderer.build_legend_strip(legend_classes, width=target_bottom_width, height=58)
    legend_strip = ImageOps.contain(legend_strip, (target_bottom_width, 52))
    legend_x = margin
    legend_y = obs_label_y + 38
    canvas.paste(legend_strip, (legend_x, legend_y))
    canvas.paste(bottom_strip, (margin, y2 + voxel_pad_top))
    time_y = y2 + voxel_pad_top + bottom_h + 4
    for px, label in zip(panel_xs, time_labels):
        tb = draw.textbbox((0, 0), label, font=font_time)
        tx = margin + px + max(0, panel_w // 2 - (tb[2] - tb[0]) // 2)
        draw.text((tx, time_y), label, fill="black", font=font_time)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def representative_gallery_name(reason: str) -> str:
    if "false-free" in reason:
        return "gallery_false_free_case"
    if "false-occupied" in reason:
        return "gallery_false_occupied_case"
    if "small-object" in reason:
        return "gallery_small_object_case"
    if "dynamic" in reason:
        return "gallery_dynamic_case"
    return "gallery_success_case"


def quality_check(paths: list[Path], sample_rows: list[dict[str, Any]], style_path: Path) -> dict[str, Any]:
    checks = {
        "figure_count": len(paths),
        "style_config_exists": style_path.exists(),
        "highres_pass": True,
        "white_background_pass": True,
        "at_least_three_rollouts": sum(1 for p in paths if "paper_style_gt_sparseworld_rollout" in p.name) >= 3,
        "at_least_one_composite": any("observation_future_composite" in p.name for p in paths),
        "at_least_one_gif": any(p.suffix.lower() == ".gif" for p in paths),
        "horizon_labels_correct": True,
        "overall_pass": True,
    }
    for p in paths:
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = Image.open(p).convert("RGB")
        require_highres = "zoom_inset" not in p.name.lower()
        if require_highres and img.width < 2000:
            checks["highres_pass"] = False
        corner = np.asarray(img.crop((0, 0, 60, 60))).mean()
        if corner < 180:
            checks["white_background_pass"] = False
        name = p.name.lower()
        if "6s" in name:
            checks["horizon_labels_correct"] = False
    checks["overall_pass"] = all(
        bool(checks[k])
        for k in ["style_config_exists", "highres_pass", "white_background_pass", "at_least_three_rollouts", "at_least_one_composite", "at_least_one_gif", "horizon_labels_correct"]
    )
    return checks


def main() -> int:
    args = parse_args()
    ensure_dirs()
    requested_horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    actual_backend = "open3d_windows_native" if getattr(renderer, "o3d", None) is not None else "matplotlib"
    style = renderer.RendererStyle(
        dpi=args.dpi,
        backend="open3d" if actual_backend != "matplotlib" else "matplotlib",
        mode="voxel_mesh" if actual_backend != "matplotlib" else "point_cloud",
        viewpoint="top",
        render_width=2200,
        render_height=1400,
        open3d_point_size=6.8,
    )
    style_path = REPORTS_DIR / "paper_style_visualization_config.json"
    renderer.save_style_config(style_path, style)
    write_json(
        REPORTS_DIR / "paper_style_template1_manifest.json",
        {
            "template_name": "template1",
            "description": "SparseWorld voxel-paper composite with fixed 6-camera present panel and SW-only current/future voxel rollout below.",
            "camera_panel": "top 2x3 nuScenes camera grid",
            "lower_panel": {
                "left": "Observations = SparseWorld current voxel h0",
                "right": "Predicted Futures = SparseWorld future voxels h2/h4/h6",
            },
            "text_style": {"present": "black", "observations": "black", "predicted_futures": "black"},
        },
    )

    if args.sample_indices.strip():
        selected = []
        for idx in [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]:
            selected.append({"sample_index": idx, "selection_reason": "user-specified"})
    else:
        selected = select_samples(args.num_samples)
    write_csv(REPORTS_DIR / "selected_visualization_samples.csv", selected)
    write_json(REPORTS_DIR / "swvis1_run_manifest.json", {"backend_requested": args.backend, "backend_used": actual_backend, "selected_samples": selected, "horizons": requested_horizons})

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    sample_manifest_full = {
        int(r["sample_index"]): r
        for r in read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv")
    }
    all_infos = load_sparseworld_info_list(repo_root)

    generated_paths: list[Path] = []
    rollout_manifest: list[dict[str, Any]] = []
    composite_manifest: list[dict[str, Any]] = []
    gallery_manifest: list[dict[str, Any]] = []
    new_visible_rows = read_csv(
        PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/new_visible_proxy_metrics.csv"
    )
    new_visible_by_sample: dict[int, float] = {}
    for item in new_visible_rows:
        sample_index = int(item["sample_index"])
        new_visible_by_sample.setdefault(sample_index, 0.0)
        new_visible_by_sample[sample_index] = max(new_visible_by_sample[sample_index], float(item["new_visible_false_free"]))
    selected_new_visible_sample = None
    if selected:
        selected_new_visible_sample = max(
            [int(row["sample_index"]) for row in selected],
            key=lambda idx: new_visible_by_sample.get(idx, -1.0),
        )

    for row in selected:
        sample_index = int(row["sample_index"])
        pred_temporal, gt_temporal = load_occ_pair(sample_index)
        sample_label = f"Sample {sample_index} | {row.get('selection_reason', '')}"
        gt_panels = []
        pred_panels = []
        error_panels = []
        legend_classes = None
        for h in requested_horizons:
            gt_panel = renderer.render_semantic_occ(gt_temporal[h], renderer.HORIZON_LABELS[h], style)
            pred_panel = renderer.render_semantic_occ(pred_temporal[h], renderer.HORIZON_LABELS[h], style)
            error_panel = renderer.error_overlay_bev(pred_temporal[h], gt_temporal[h])
            gt_panels.append(gt_panel)
            pred_panels.append(pred_panel)
            error_panels.append(error_panel)
            if legend_classes is None:
                _, labels = renderer.occupancy_points(gt_temporal[h])
                legend_classes = renderer.present_classes(labels)
        rollout_png = FIGURES_DIR / f"paper_style_gt_sparseworld_rollout_sample{sample_index}.png"
        rollout_pdf = FIGURES_DIR / f"paper_style_gt_sparseworld_rollout_sample{sample_index}.pdf"
        renderer.make_rollout_grid(gt_panels, pred_panels, [renderer.HORIZON_LABELS[h] for h in requested_horizons], sample_label, rollout_png, rollout_pdf)
        generated_paths.extend([rollout_png, rollout_pdf])
        rollout_manifest.append({"sample_index": sample_index, "rollout_png": str(rollout_png), "rollout_pdf": str(rollout_pdf)})

        error_grid = Image.new("RGB", (error_panels[0].width * len(error_panels), error_panels[0].height), "white")
        x = 0
        for p in error_panels:
            error_grid.paste(p, (x, 0))
            x += p.width
        error_full = FIGURES_DIR / f"paper_style_error_overlay_sample{sample_index}.png"
        error_grid.save(error_full)
        generated_paths.append(error_full)

        inset = error_panels[-1].crop((220, 220, 680, 680)).resize((900, 900), Image.Resampling.BILINEAR)
        inset_path = FIGURES_DIR / f"paper_style_error_zoom_inset_sample{sample_index}.png"
        inset.save(inset_path)
        generated_paths.append(inset_path)

        gallery_name = representative_gallery_name(str(row.get("selection_reason", "")))
        gallery_path = FIGURES_DIR / f"{gallery_name}_{sample_index}.png"
        gallery_canvas = Image.new("RGB", (max(rollout_png.stat().st_size, 1), max(1, 1)), "white")
        # rebuild gallery with GT / Pred / Error stacked
        gw, gh = gt_panels[0].size
        gallery = Image.new("RGB", (len(requested_horizons) * gw, 3 * gh), "white")
        for i, p in enumerate(gt_panels):
            gallery.paste(p, (i * gw, 0))
        for i, p in enumerate(pred_panels):
            gallery.paste(p, (i * gw, gh))
        for i, p in enumerate(error_panels):
            gallery.paste(renderer.fit_panel(p, (gw, gh)), (i * gw, 2 * gh))
        gallery.save(gallery_path)
        generated_paths.append(gallery_path)
        gallery_manifest.append({"sample_index": sample_index, "gallery_path": str(gallery_path), "reason": row.get("selection_reason", "")})
        if selected_new_visible_sample == sample_index:
            new_visible_gallery = FIGURES_DIR / f"gallery_new_visible_case_{sample_index}.png"
            gallery.save(new_visible_gallery)
            generated_paths.append(new_visible_gallery)
            gallery_manifest.append({"sample_index": sample_index, "gallery_path": str(new_visible_gallery), "reason": "new-visible failure case"})

        if args.render_camera_panel:
            mapped_info_index = int(sample_manifest_full[sample_index]["mapped_info_index"])
            cam_paths = current_camera_paths_from_info(all_infos[mapped_info_index])
            cam_imgs = load_camera_strip(cam_paths, repo_root)
            cam_labels = [Path(p).parent.name for p in cam_paths]
            cam_panel = camera_grid(cam_imgs, cam_labels)
            current_panel = pred_panels[0]
            futures = pred_panels[1:]
            composite_path = FIGURES_DIR / f"paper_style_observation_future_composite_sample{sample_index}.png"
            compose_observation_future(
                cam_panel,
                current_panel,
                futures,
                sample_label,
                [renderer.HORIZON_LABELS[h] for h in requested_horizons],
                composite_path,
            )
            generated_paths.append(composite_path)
            composite_manifest.append({"sample_index": sample_index, "composite_path": str(composite_path)})

            if args.render_gif:
                gif_frames = [
                    renderer.render_semantic_occ(pred_temporal[h], renderer.HORIZON_LABELS.get(h, f"h{h}"), style)
                    for h in range(pred_temporal.shape[0])
                ]
                gif_path = FIGURES_DIR / f"paper_style_future_rollout_sample{sample_index}.gif"
                renderer.save_gif(gif_frames, gif_path)
                generated_paths.append(gif_path)
                mp4_path = FIGURES_DIR / f"paper_style_future_rollout_sample{sample_index}.mp4"
                if renderer.save_mp4_stub(gif_frames, mp4_path):
                    generated_paths.append(mp4_path)

    write_json(REPORTS_DIR / "paper_style_rollout_manifest.json", rollout_manifest)
    write_json(REPORTS_DIR / "camera_future_composite_manifest.json", composite_manifest)
    write_json(REPORTS_DIR / "visual_case_gallery_manifest.json", gallery_manifest)

    checks = quality_check(generated_paths, selected, style_path)
    write_json(REPORTS_DIR / "visualization_quality_check.json", checks)
    (REPORTS_DIR / "visualization_quality_check.md").write_text(
        "\n".join([f"- {k}: `{v}`" for k, v in checks.items()]),
        encoding="utf-8",
    )

    readme = [
        "# SparseWorld paper-style visualization",
        "",
        "1. Purpose: selected qualitative visualization assets for resume / PPT / interview.",
        "2. Inputs: real SparseWorld clean baseline outputs and nuScenes camera observations.",
        "3. Rendering: paper-style white background shaded voxel render with fixed pose.",
        "4. Figures: rollout grid, observation/future composite, case gallery, error overlay, GIF.",
        "5. Safe claims:",
        "- selected qualitative visualization",
        "- not official benchmark",
        "- real model output",
        "- no manual editing of prediction",
    ]
    (REPORTS_DIR / "paper_style_visualization_readme.md").write_text("\n".join(readme), encoding="utf-8")
    report = {
        "stage": "SW-VIS1",
        "backend_used": actual_backend,
        "selected_samples": selected,
        "generated_paths": [str(p) for p in generated_paths],
        "quality_check": checks,
        "next_unique_action": "Stage SW-VIS2 animation polish or sensor-aware failure gallery overlay for presentation-ready storytelling.",
    }
    write_json(REPORTS_DIR / "stage_swvis1_paper_style_visualization_report.json", report)
    (REPORTS_DIR / "stage_swvis1_paper_style_visualization_report.md").write_text(
        "\n".join(
            [
                "# Stage SW-VIS1 paper-style visualization",
                "",
                f"- Backend used: `{actual_backend}`",
                f"- Selected samples: `{[x['sample_index'] for x in selected]}`",
                f"- Quality check overall: `{checks['overall_pass']}`",
                "- This is selected qualitative visualization, not an official benchmark.",
                "- Predictions were not manually edited.",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
