import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_decision.json').read_text())
def test_no_gt_in_sw14a_teacherfix():
    assert obj['uses_gt_for_teacher'] is False
    assert obj['uses_gt_for_training'] is False
