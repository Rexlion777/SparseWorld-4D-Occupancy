import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("CV_LIDAR_PROJECT_ROOT")
    if env:
        path = Path(env)
        if path.exists():
            return path
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/stage_sw91_h2_contributor_assignment_training_report.md"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    banned = [
        "official benchmark winner",
        "we beat the paper",
        "beating the paper",
        "final model claim",
    ]
    for phrase in banned:
        assert phrase not in text, f"forbidden phrase present: {phrase}"
    assert "not an official benchmark" in text
    assert "not full training" in text
    assert "not a final model claim" in text or "not full training" in text
