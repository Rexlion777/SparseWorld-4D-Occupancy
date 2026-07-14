from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from swvis4_support_interp_lib import PROJECT_ROOT, load_renderer_module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--occ-path", type=str, required=True)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--sleep-sec", type=float, default=0.15)
    p.add_argument("--save-every", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    renderer = load_renderer_module()
    style = renderer.RendererStyle(
        dpi=180,
        backend="open3d",
        mode="voxel_mesh",
        viewpoint="top",
        render_width=1400,
        render_height=900,
        open3d_point_size=6.8,
    )
    occ = torch.from_numpy(np.load(args.occ_path).astype(np.int64))
    out_dir = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_swvis4_query_support_interpolation/render_gpu_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    for i in range(args.iterations):
        t0 = time.perf_counter()
        img = renderer.render_semantic_occ(occ, "", style)
        t1 = time.perf_counter()
        timings.append({"iter": i, "render_sec": t1 - t0})
        if args.save_every > 0 and (i % args.save_every == 0):
            img.save(out_dir / f"diagnostic_frame_{i:04d}.png")
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
    payload = {
        "occ_path": args.occ_path,
        "iterations": args.iterations,
        "sleep_sec": args.sleep_sec,
        "mean_render_sec": float(np.mean([x["render_sec"] for x in timings])) if timings else 0.0,
        "max_render_sec": float(np.max([x["render_sec"] for x in timings])) if timings else 0.0,
        "timings": timings,
    }
    (out_dir / "render_gpu_diagnostic_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
