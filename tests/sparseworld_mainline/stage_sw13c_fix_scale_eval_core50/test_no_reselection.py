import json
from pathlib import Path
def test_no_reselection():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_eval_core50_decision.json').read_text())
    assert obj['no_reselection'] is True
