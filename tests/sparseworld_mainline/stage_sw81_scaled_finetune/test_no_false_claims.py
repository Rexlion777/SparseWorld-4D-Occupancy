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
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/stage_sw81_scaled_finetune_report.md"


def test_no_false_claims():
    text = REPORT.read_text(encoding="utf-8").lower()
    banned = [
        "beat the paper",
        "beating paper",
        "production-ready",
    ]
    for phrase in banned:
        assert phrase not in text, f"forbidden phrase present: {phrase}"
    assert "not an official benchmark" in text or "not full official benchmark" in text
    assert "1-iter/1-sample" not in text, "report should not rely on 1-iter/1-sample claims"
