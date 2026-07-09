import json
from pathlib import Path
def test_decision_schema():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/sw13c_fix_frontcap_eval_core100_decision.json').read_text())
    assert obj['decision_type'] in {'Y1_FRONTCAP_EVAL100_STRONG','Y2_FRONTCAP_EVAL100_MEDIUM','Y3_FRONTCAP_EVAL100_DIRECTION_HOLDS','Y4_FRONTCAP_EVAL100_RESIDUAL_RISK','Y5_FRONTCAP_EVAL100_FAILS','Y6_PROTOCOL_MISMATCH'}
