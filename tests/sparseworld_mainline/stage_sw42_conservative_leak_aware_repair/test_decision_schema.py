import json
from pathlib import Path

BASE = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair")


def test_decision_schema():
    payload = json.loads((BASE / "sw42_conservative_repair_decision.json").read_text(encoding="utf-8"))
    assert payload["case"] in {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert "reason" in payload
