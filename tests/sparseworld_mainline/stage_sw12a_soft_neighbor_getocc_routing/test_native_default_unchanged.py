from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing/sw12a_native_default_unchanged_check.json"


def test_native_default_unchanged() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert rows, "expected at least one native-default equivalence row"
    for row in rows:
        assert row["default_vs_explicit_native_equal"] is True
        assert row["default_vs_sw4_debug_equal"] is True
