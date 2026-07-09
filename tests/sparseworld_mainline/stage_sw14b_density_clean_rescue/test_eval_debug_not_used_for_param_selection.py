import json
from pathlib import Path
scale = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_alpha_scale_selection.json')
def test_eval_debug_not_used_for_param_selection():
    payload = json.loads(scale.read_text())
    assert payload['selected_on_split'] == 'val_small_200_249'
