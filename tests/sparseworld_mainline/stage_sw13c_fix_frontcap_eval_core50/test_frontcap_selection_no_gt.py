import json
from pathlib import Path
SRC=Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/run_sw13c_fix_frontcap_main.py').read_text()
def test_frontcap_selection_no_gt():
    assert 'pred_gt_density_delta' not in SRC.split('def choose_frontcap',1)[1].split('def save_bar',1)[0]
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_candidate_selection.json').read_text())
    assert obj['selection_uses_gt'] is False
