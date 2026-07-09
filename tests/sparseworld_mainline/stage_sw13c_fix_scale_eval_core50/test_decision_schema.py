import json
from pathlib import Path
def test_decision_schema():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_eval_core50_decision.json').read_text())
    assert obj['decision_type'] in {'W1_STRONG_SCALE_CONFIRMED','W2_MEDIUM_SCALE_CONFIRMED','W3_RECOVERY_WEAKENS_BUT_DIRECTION_HOLDS','W4_DENSITY_OR_FRONT_LOCAL_RISK','W5_SCALE_FAILS','W6_PROTOCOL_VIOLATION'}
