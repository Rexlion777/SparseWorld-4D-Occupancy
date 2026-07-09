import json
from pathlib import Path
PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/artifacts/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/native_path_unchanged_check.json")

def test_native_path_unchanged() -> None:
    obj = json.loads(PATH.read_text(encoding="utf-8"))
    assert obj["checked"] is True
    assert obj["pass"] is True
    assert float(obj["max_abs_diff"]) == 0.0
