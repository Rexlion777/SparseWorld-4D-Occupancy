import json
from pathlib import Path


def repo_root() -> Path:
    candidates = [Path("D:/ComputerVision/cv_lidar_transition"), Path("/mnt/d/ComputerVision/cv_lidar_transition")]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


ROOT = repo_root()
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw71_reliability_cleanup"


def test_no_false_claims():
    json_payload = json.loads((REPORTS / "stage_sw71_reliability_cleanup_report.json").read_text(encoding="utf-8"))
    md_text = (REPORTS / "stage_sw71_reliability_cleanup_report.md").read_text(encoding="utf-8").lower()
    card_text = (REPORTS / "sw7_one_page_project_card.md").read_text(encoding="utf-8").lower()

    not_claims = set(json_payload["not_claims"])
    assert "not calibrated uncertainty" in not_claims
    assert "not official benchmark" in not_claims
    assert "not production planning policy" in not_claims
    assert "no training" in not_claims
    assert "no model improvement claim" in not_claims

    assert "official benchmark" in md_text
    assert "not official benchmark" in md_text
    assert "not calibrated uncertainty" in md_text
    assert "not production planning policy" in md_text
    assert "no model improvement claim" in md_text
    assert "training completed" not in md_text
    assert "training completed" not in card_text
