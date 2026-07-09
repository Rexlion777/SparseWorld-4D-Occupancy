import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_density_clean_rescue_final_decision.json')
def test_no_resume_claim_if_fail():
    payload = json.loads(path.read_text())
    if payload['decision'] in {'SW14B_RESCUE_0_FAIL_KEEP_SW13_MAIN','SW14B_RESCUE_4_DENSITY_UNRESOLVED','SW14B_RESCUE_5_PROTOCOL_VIOLATION'}:
        assert payload['whether_can_use_in_resume'] is False
