import json
from pathlib import Path

BASE = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair")


def test_pareto_criteria_schema():
    payload = json.loads((BASE / "sw42_pareto_acceptance_criteria.json").read_text(encoding="utf-8"))
    for key in ["strict_safe", "moderate_safe", "reject", "bonus"]:
        assert key in payload
