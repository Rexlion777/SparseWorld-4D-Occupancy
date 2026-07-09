import json
from pathlib import Path


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"


def test_sw41_candidate_decision_schema():
    path = REPORTS_DIR / "sw41_repair_candidate_decision.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["case"] in {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}
    for key in [
        "best_soft_variant",
        "best_gate_variant",
        "best_class_variant",
        "best_overall_candidate",
        "best_safe_candidate",
        "best_small_object_candidate",
        "best_new_visible_candidate",
    ]:
        assert key in data
