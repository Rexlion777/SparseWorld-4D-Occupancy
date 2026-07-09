import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14b_decision.json').read_text())
def test_sw14b_gt_train_only():
    assert obj['uses_gt_only_on_train_split'] is True
    assert obj['uses_eval_gt_for_tuning'] is False
