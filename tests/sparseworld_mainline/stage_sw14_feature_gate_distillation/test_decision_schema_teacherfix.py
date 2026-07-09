import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_decision.json').read_text())
def test_decision_schema_teacherfix():
    assert obj['decision_type'] in {'T1_TEACHER_TARGET_FIXED','T2_CAMERA_ORDER_MISMATCH','T3_FPN_LEVEL_MISMATCH','T4_SAMPLE_INDEX_MISMATCH','T5_TARGET_STILL_NOT_HARD_R8','T6_UNABLE_TO_RECONSTRUCT_R8_FEATURE'}
