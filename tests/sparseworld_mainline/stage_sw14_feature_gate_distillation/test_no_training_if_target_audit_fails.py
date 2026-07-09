import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_decision.json').read_text())
train=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_checkpoint_manifest.json').read_text())
def test_no_training_if_target_audit_fails():
    if obj['decision_type'] != 'T1_TEACHER_TARGET_FIXED':
        assert train['executed'] is False
