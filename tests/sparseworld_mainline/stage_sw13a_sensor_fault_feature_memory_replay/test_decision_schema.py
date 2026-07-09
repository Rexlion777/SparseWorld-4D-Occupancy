import json
from pathlib import Path
PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_memory_replay_decision.json")

def test_decision_schema() -> None:
    obj = json.loads(PATH.read_text(encoding="utf-8"))
    assert "decision_type" in obj
    assert obj["decision_type"] in {
        "S1_STRONG_FEATURE_MEMORY_GAIN", "S2_WEAK_BUT_SAFE_FEATURE_MEMORY_GAIN", "S3_RECOVERY_BUT_DENSITY_UNSAFE",
        "S4_NO_RECOVERY", "S5_C4_ONLY_GAIN", "S6_IMPLEMENTATION_BLOCKED", "S7_ORACLE_RISK"
    }
