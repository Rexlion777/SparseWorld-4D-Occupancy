import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_native_path_integrity.json').read_text())
def test_adapter_disabled_path_integrity_teacherfix():
    assert obj['pass'] is True
