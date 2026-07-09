import json
from pathlib import Path
def test_no_reselection():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/sw13c_fix_frontcap_eval_core100_decision.json').read_text())
    assert obj['no_reselection'] is True
    assert obj['no_retuning'] is True
    assert obj['no_training'] is True
