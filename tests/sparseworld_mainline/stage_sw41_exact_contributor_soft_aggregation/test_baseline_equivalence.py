import json
from pathlib import Path


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"


def test_sw41_baseline_equivalence_passed():
    path = REPORTS_DIR / "baseline_replay_equivalence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["original_get_occ_checks"]
    assert data["saved_pred_checks"]
    assert all(item["exact_equal"] for item in data["original_get_occ_checks"])
    assert all(item["exact_equal"] for item in data["saved_pred_checks"])
