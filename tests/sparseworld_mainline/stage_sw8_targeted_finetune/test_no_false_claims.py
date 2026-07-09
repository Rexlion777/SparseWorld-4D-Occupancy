from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune/stage_sw8_targeted_finetune_report.md"


def test_no_false_claims():
    text = REPORT.read_text(encoding="utf-8").lower()
    banned = [
        "official benchmark",
        "production-ready",
        "training completed with full validation improvement",
        "beat the paper",
        "beating paper",
    ]
    for phrase in banned:
        assert phrase not in text, f"forbidden phrase present: {phrase}"
