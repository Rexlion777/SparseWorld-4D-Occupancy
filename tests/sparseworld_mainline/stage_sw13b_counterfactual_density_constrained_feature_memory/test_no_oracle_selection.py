import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_non_oracle_candidate_selection.json')

def test_no_oracle_selection() -> None:
    obj = json.loads(PATH.read_text())
    assert obj['selection_uses_gt'] is False
    assert obj['sw13a_decision_type'] == 'S3_RECOVERY_BUT_DENSITY_UNSAFE'
