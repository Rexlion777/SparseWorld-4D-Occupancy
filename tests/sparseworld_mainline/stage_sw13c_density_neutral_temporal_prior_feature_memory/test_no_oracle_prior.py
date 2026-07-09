import json
from pathlib import Path
def test_no_oracle_prior():
    o=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_temporal_prior_manifest.json').read_text())
    assert o['uses_future_info'] is False and o['uses_current_clean_same_frame'] is False and o['uses_gt_prior'] is False
