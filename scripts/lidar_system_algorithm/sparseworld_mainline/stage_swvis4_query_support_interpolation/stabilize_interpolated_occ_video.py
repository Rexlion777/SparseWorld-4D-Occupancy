from __future__ import annotations

from pathlib import Path

from run_sparseworld_swvis4_main import build_support_keyframes, interpolate_support_frames, voxelize_frames
from swvis4_support_interp_lib import ARTIFACTS_DIR, ensure_dirs, stabilize_occ_sequence


if __name__ == "__main__":
    ensure_dirs()
    support_paths, _ = build_support_keyframes(3)
    interp_dir = interpolate_support_frames(3, support_paths)
    occ_dir = voxelize_frames(3, Path(interp_dir))
    stabilize_occ_sequence(occ_dir, ARTIFACTS_DIR / "stabilized_occ_frames/sample_0003")
