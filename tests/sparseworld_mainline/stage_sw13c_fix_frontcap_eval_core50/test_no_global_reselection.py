import json
from pathlib import Path
def test_no_global_reselection():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_candidate_selection.json').read_text())
    assert obj['no_global_reselection'] is True
    assert obj['no_global_retuning'] is True
