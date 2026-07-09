from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT_DIR = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation"
VIDEO_DIR = ROOT / "videos/sparseworld_mainline/stage_swvis4_query_support_interpolation"


def test_endpoint_exact_fix_outputs_exist() -> None:
    required = [
        REPORT_DIR / "native_get_occ_replay_audit.json",
        REPORT_DIR / "surrogate_voxelizer_ablation.csv",
        REPORT_DIR / "endpoint_residual_summary.json",
        REPORT_DIR / "layered_interpolation_summary.json",
        REPORT_DIR / "keyframe_exactness_check.csv",
        REPORT_DIR / "keyframe_jump_check.csv",
        REPORT_DIR / "swvis4_endpoint_exact_fix_report.json",
    ]
    for path in required:
        assert path.exists(), str(path)


def test_endpoint_exact_fix_video_exists() -> None:
    report = json.loads((REPORT_DIR / "swvis4_endpoint_exact_fix_report.json").read_text(encoding="utf-8"))
    assert Path(report["video_path"]).exists()
    assert Path(report["comparison_video_path"]).exists()
