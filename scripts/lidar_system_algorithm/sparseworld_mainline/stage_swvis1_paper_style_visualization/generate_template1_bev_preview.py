from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis1_paper_style_visualization"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vismod = load_module("swvis1_main_preview", SCRIPT_DIR / "run_sparseworld_paper_style_visualization.py")
renderer = load_module("swvis1_renderer_preview", SCRIPT_DIR / "semantic_occupancy_renderer.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-index", type=int, default=3)
    p.add_argument("--horizons", default="0,2,4,6")
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    return p.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    sample_index = int(args.sample_index)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    repo_root = Path(args.repo_root).resolve()

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

    style = renderer.RendererStyle(
        dpi=300,
        backend="open3d" if getattr(renderer, "o3d", None) is not None else "matplotlib",
        mode="voxel_mesh" if getattr(renderer, "o3d", None) is not None else "point_cloud",
        viewpoint="bev_strict",
        render_width=2200,
        render_height=1400,
        open3d_point_size=6.8,
    )

    current_panel = renderer.render_semantic_occ(pred_temporal[horizons[0]], "", style)
    future_panels = [renderer.render_semantic_occ(pred_temporal[h], "", style) for h in horizons[1:]]
    sample_label = f"Sample {sample_index} | template1 BEV preview"

    out_path = FIGURES_DIR / f"template1_bev_preview_sample{sample_index}.png"
    vismod.compose_observation_future(
        cam_panel,
        current_panel,
        future_panels,
        sample_label,
        [renderer.HORIZON_LABELS[h] for h in horizons],
        out_path,
    )

    manifest = {
        "sample_index": sample_index,
        "horizons": horizons,
        "viewpoint": "bev_strict",
        "style_backend": style.backend,
        "style_mode": style.mode,
        "output_path": str(out_path),
        "note": "Preview only. Uses current template1 layout with strict BEV semantic rendering and original semantic palette.",
    }
    write_json(REPORTS_DIR / "template1_bev_preview_manifest.json", manifest)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
