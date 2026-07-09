import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14a_decision.json').read_text())
def test_no_eval_gt_leakage():
    assert obj['uses_gt_loss'] is False
    assert obj['subset_diagnostic_only'] is True
