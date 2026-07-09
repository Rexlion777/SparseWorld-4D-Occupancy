from __future__ import annotations

from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
DIR = ROOT / "artifacts/sparseworld_mainline/stage_swvis4_query_support_interpolation/support_keyframes/sample_0003"


def test_support_keyframes_exist() -> None:
    assert DIR.exists()
    for h in range(7):
        assert (DIR / f"h{h}_support.npz").exists()
