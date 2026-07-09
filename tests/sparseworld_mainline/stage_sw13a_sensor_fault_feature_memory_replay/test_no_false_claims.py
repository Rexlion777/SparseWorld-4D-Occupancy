from pathlib import Path
PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/stage_sw13a_sensor_fault_feature_memory_replay_report.md")

def test_no_false_claims() -> None:
    text = PATH.read_text(encoding="utf-8").lower()
    assert "subset diagnostic" in text
    assert "not official benchmark" in text
    assert "no training" in text
    banned = ["official benchmark result", "trained model improvement", "beats the paper", "production ready"]
    for phrase in banned:
        assert phrase not in text
