import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_final_decision.json')
def test_decision_schema():
    payload = json.loads(PATH.read_text())
    required = {'decision','best_checkpoint','best_adapter_type','train_split_used','eval_split_used','whether_use_in_resume','whether_SW13_remains_main','safe_claim','limitations','next_action'}
    assert required.issubset(payload.keys())
