import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_postprocess_chain_audit.json')
def test_no_gt_in_frontcap():
    payload = json.loads(PATH.read_text())
    assert payload['no_gt_in_frontcap'] is True
