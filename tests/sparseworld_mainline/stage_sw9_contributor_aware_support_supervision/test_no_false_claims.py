import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("CV_LIDAR_PROJECT_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/stage_sw9_contributor_aware_support_supervision_report.md"


def test_no_false_claims():
    text = REPORT.read_text(encoding="utf-8").lower()
    banned = [
        "official benchmark winner",
        "production-ready",
        "we beat the paper",
        "we are beating paper",
        "calibrated uncertainty claim",
    ]
    for phrase in banned:
        assert phrase not in text, f"forbidden phrase present: {phrase}"
    assert "not an official benchmark" in text
    assert "not full training" in text or "short training for feasibility + signal check" in text
    assert "no claim of beating the paper" in text
