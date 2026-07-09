import json
from pathlib import Path


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"


def test_sw41_no_false_claims():
    report_md = (REPORTS_DIR / "stage_sw41_exact_contributor_soft_aggregation_report.md").read_text(encoding="utf-8").lower()
    forbidden = [
        "official benchmark reproduced",
        "training completed",
        "production-ready",
        "deployable repair",
    ]
    assert not any(token in report_md for token in forbidden)

    report_json = json.loads((REPORTS_DIR / "stage_sw41_exact_contributor_soft_aggregation_report.json").read_text(encoding="utf-8"))
    safe_claims = " ".join(report_json.get("safe_claims", [])).lower()
    assert "no training" in safe_claims
    assert "not official sparseworld benchmark" in safe_claims
