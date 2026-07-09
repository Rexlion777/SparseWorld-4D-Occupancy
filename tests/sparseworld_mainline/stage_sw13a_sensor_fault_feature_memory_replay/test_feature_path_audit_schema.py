import json
from pathlib import Path
PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_path_audit.json")

def test_feature_path_audit_schema() -> None:
    obj = json.loads(PATH.read_text(encoding="utf-8"))
    assert "input_img_shape" in obj
    assert "camera_index_mapping" in obj
    assert "feature_levels" in obj
    assert "path_audit" in obj and obj["path_audit"]
