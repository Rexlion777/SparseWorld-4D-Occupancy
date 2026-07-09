import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_split_manifest.json')
def test_no_eval_gt_for_tuning():
    payload = json.loads(PATH.read_text())
    eval_debug = [x for x in payload['splits'] if x['split_name'] == 'eval_debug'][0]
    assert eval_debug['tuning_allowed'] is False
