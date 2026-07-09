import json
from pathlib import Path
def test_non_oracle_selection():
    o=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_non_oracle_candidate_selection.json').read_text()); assert o['selection_uses_gt'] is False
