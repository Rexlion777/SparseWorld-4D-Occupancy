#!/usr/bin/env python3
"""Export frozen R0/R8 semantic anchors with the existing Template 1 renderer.

This bridge is intentionally read-only: it consumes a frozen SW14C teacher
cache whose ``native_semantic`` and ``teacher_raw_semantic`` fields are the
degraded-native and raw SW13A/R8 predictions respectively.  It never runs a
model, reads ground truth for repair, or writes into the research workspace.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


HORIZONS = (0, 2, 4, 6)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    source = project_root / (
        "scripts/lidar_system_algorithm/sparseworld_mainline/"
        "stage_swvis1_paper_style_visualization/"
        "generate_template1_firstperson_perturbations.py"
    )
    template1 = load_module("template1_r8_anchor_export", source)
    bundle = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    camera_panel, _ = template1.camera_panel_for_sample(
        args.sample_index,
        "A10_drop_front_triplet",
        project_root / "external/SparseWorld",
    )
    camera_panel.save(output / "a10_front_triplet_camera_panel.png")

    for horizon in HORIZONS:
        row = bundle["by_horizon"][horizon]
        native = row["native_semantic"].long()
        repaired = row["teacher_raw_semantic"].long()
        # The portfolio export uses the renderer's deterministic matplotlib
        # path so it also works in headless SSH sessions without OpenGL.
        style = template1.renderer.RendererStyle(
            backend="matplotlib",
            mode="voxel_mesh",
            viewpoint="front",
            figsize=(7.2, 5.2),
            dpi=180,
            elev=23.0,
            azim=-68.0,
            point_size=2.2,
        )
        template1.renderer.render_semantic_occ(native, "", style).save(output / f"native_h{horizon}.png")
        template1.renderer.render_semantic_occ(repaired, "", style).save(output / f"r8_h{horizon}.png")

    print(f"exported Template 1 anchors to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
