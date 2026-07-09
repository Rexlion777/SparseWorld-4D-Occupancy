import json
from pathlib import Path

PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/sw12c_decision.json")


def test_decision_schema() -> None:
    obj = json.loads(PATH.read_text(encoding="utf-8"))
    assert "decision_labels" in obj
    assert "branchA_decision" in obj
    assert "branchB_decision" in obj
    assert obj["decision_labels"]
