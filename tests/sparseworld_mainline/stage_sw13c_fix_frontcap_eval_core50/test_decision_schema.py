import json
from pathlib import Path
def test_decision_schema():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_eval_core50_decision.json').read_text())
    assert obj['decision_type'] in {'X1_FRONTCAP_STRONG_FIX','X2_FRONTCAP_MEDIUM_FIX','X3_FRONTCAP_RECOVERY_TRADEOFF','X4_FRONTCAP_RESIDUAL_FRONT_DENSITY_RISK','X5_FRONTCAP_FAILS','X6_PROTOCOL_VIOLATION'}
