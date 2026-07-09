import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_final_decision.json').read_text())
def test_decision_schema():
    assert obj['decision_type'] in {'S14_0_TEACHER_REMAINS_MAIN_RESULT','S14_1_RULE_DISTILLATION_SUCCESS','S14_2_RULE_DISTILLATION_EXCEEDS_TEACHER','S14_3_GT_REFINEMENT_SUCCESS','S14_4_GT_REFINEMENT_REJECTED','S14_5_PROTOCOL_VIOLATION'}
