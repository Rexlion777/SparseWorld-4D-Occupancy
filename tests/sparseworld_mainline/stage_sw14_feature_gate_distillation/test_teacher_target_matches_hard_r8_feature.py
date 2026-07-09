import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_target_audit.json').read_text())
def test_teacher_target_matches_hard_r8_feature():
    assert all(r['target_matches_hard_r8_feature'] for r in obj['rows'])
