from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/dashboard_manifest.json"


def resolve_any(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if len(path_str) > 2 and path_str[1:3] == ":\\":
        drive = path_str[0].lower()
        rest = path_str[3:].replace("\\", "/")
        alt = Path(f"/mnt/{drive}/{rest}")
        if alt.exists():
            return alt
    return p


def test_dashboard_manifest() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert len(payload) >= 4
    for row in payload:
        assert resolve_any(row["frame_dir"]).exists()
        assert resolve_any(row["video_path"]).exists()
        assert row["diagnostic_only"] is True
