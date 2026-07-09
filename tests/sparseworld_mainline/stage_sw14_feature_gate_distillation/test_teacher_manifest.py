import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/teacher_candidate_manifest.json').read_text())
def test_teacher_manifest():
    assert obj['sw13_teacher_remains_main'] is True
    assert obj['sw14_optional_trainable_enhancement'] is True
