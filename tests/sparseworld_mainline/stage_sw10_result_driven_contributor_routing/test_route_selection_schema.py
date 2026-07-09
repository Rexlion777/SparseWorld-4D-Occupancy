from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing/sw10_route_selection.json"


def test_route_selection_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["selected_route"] in {"A", "B", "C", "D", "E"}
    assert isinstance(payload["summary"], str) and payload["summary"]
    assert isinstance(payload["evidence"], dict)
    assert isinstance(payload["selected_candidate"], str) and payload["selected_candidate"]
