from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/stage_swvis4_query_support_interpolation_report.json"


def test_interpolated_support_frame_count() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    support_dir = Path(data["key_paths"]["interpolated_support_frames_fixed"])
    assert data["frame_count"] == 540
    assert len(list(support_dir.glob("frame_*.npz"))) == data["frame_count"]


def test_interpolated_occ_frame_count() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    occ_dir = Path(data["key_paths"]["interpolated_occ_frames_fixed"])
    assert len(list(occ_dir.glob("occ_frame_*.npy"))) == data["frame_count"]
