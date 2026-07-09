import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_target_audit.json').read_text())
def test_fpn_level_verified():
    assert all(r['fpn_level_verified'] for r in obj['rows'])
