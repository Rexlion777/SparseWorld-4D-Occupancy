from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune"


def test_sw8_outputs_exist():
    required = [
        REPORTS / "training_entry_audit.json",
        REPORTS / "e0_baseline_resume_control_eval.csv",
        REPORTS / "sw8_finetune_decision.json",
        REPORTS / "stage_sw8_targeted_finetune_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing required output: {path}"
