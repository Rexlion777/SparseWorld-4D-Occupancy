import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_counterfactual_density_constrained_decision.json')

def test_decision_schema() -> None:
    obj = json.loads(PATH.read_text())
    assert obj['decision_type'] in {
        'T1_STRONG_SAFE_DENSITY_CONSTRAINED_RECOVERY','T2_MEDIUM_SAFE_DENSITY_CONSTRAINED_RECOVERY','T3_RECOVERY_RETAINED_BUT_DENSITY_STILL_HIGH',
        'T4_DENSITY_CONTROLLED_BUT_RECOVERY_LOST','T5_C4_ONLY_SAFE','T6_IMPLEMENTATION_BLOCKED','T7_ORACLE_RISK'
    }
