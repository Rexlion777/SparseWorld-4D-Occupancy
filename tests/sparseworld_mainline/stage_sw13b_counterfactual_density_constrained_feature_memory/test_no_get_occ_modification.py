import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_counterfactual_density_constrained_decision.json')

def test_no_get_occ_modification() -> None:
    obj = json.loads(PATH.read_text())
    assert obj['no_get_occ_modification'] is True
