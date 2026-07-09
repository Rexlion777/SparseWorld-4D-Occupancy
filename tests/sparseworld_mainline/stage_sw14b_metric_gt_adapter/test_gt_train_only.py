import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_loss_config.json')
def test_gt_train_only():
    payload = json.loads(PATH.read_text())
    assert payload['uses_gt_training'] is True
    assert payload['uses_eval_gt_for_tuning'] is False
