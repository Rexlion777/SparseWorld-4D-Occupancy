import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_clean_drift_attribution.json')
def test_clean_degradation_flag_false():
    payload = json.loads(path.read_text())
    assert payload['mean_degradation_mask_sum'] == 0.0
