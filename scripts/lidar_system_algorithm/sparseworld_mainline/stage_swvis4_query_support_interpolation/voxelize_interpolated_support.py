from __future__ import annotations

from pathlib import Path

from run_sparseworld_swvis4_main import build_support_keyframes, interpolate_support_frames, voxelize_frames
from swvis4_support_interp_lib import ensure_dirs


if __name__ == "__main__":
    ensure_dirs()
    support_paths, _ = build_support_keyframes(3)
    interp_dir = interpolate_support_frames(3, support_paths)
    voxelize_frames(3, Path(interp_dir))
