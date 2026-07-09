import json
from pathlib import Path
d=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_no_gt_density_budget_decision.json').read_text())
def test_decision_schema():
    assert d['decision_type'] in {'V1_STRONG_NO_GT_DENSITY_NEUTRAL_RECOVERY','V2_MEDIUM_NO_GT_DENSITY_NEUTRAL_RECOVERY','V3_NO_GT_RECOVERY_RETAINED_BUT_DENSITY_HIGH','V4_NO_GT_DENSITY_CONTROLLED_BUT_RECOVERY_LOST','V5_AGREEMENT_NOT_TRUE_MULTISOURCE','V6_ORIGINAL_SW13C_ONLY_DIAGNOSTIC','V7_ORACLE_RISK_REMAINS','V8_IMPLEMENTATION_BLOCKED'}
