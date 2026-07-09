import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT_MD = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/stage_sw6_frontview_blur_fragility_diagnosis_report.md"
REPORT_JSON = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/stage_sw6_frontview_blur_fragility_diagnosis_report.json"


def test_no_false_claims():
    md = REPORT_MD.read_text(encoding="utf-8").lower()
    js = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert "official benchmark" in md
    assert "not official benchmark" in md
    assert "no training" in md
    assert "oracle restore probes are diagnostic only" in md
    forbidden = [
        "official sparseworld benchmark achieved",
        "production-ready",
        "real-world sensor robustness proven",
        "training completed",
        "deployable repair",
    ]
    for phrase in forbidden:
        assert phrase not in md
    safe_claims = " ".join(js.get("safe_claims", [])).lower()
    assert "subset diagnostic" in safe_claims
    assert "not official benchmark" in safe_claims
