import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_alpha_scale_selection.json')
def test_no_gt_in_rescue_selection():
    payload = json.loads(path.read_text())
    assert payload['uses_eval_debug_for_selection'] is False
