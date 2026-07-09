import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_final_decision.json')
def test_resume_claim_guard():
    payload = json.loads(PATH.read_text())
    if payload['decision'].startswith('SW14B_0') or payload['decision'].startswith('SW14B_5') or payload['decision'].startswith('SW14B_6') or payload['decision'].startswith('SW14B_7'):
        assert payload['whether_use_in_resume'] is False
