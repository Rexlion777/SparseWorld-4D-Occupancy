import json
from pathlib import Path
d=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_no_gt_density_budget_decision.json').read_text())
def test_selected_agreement_source_count():
    if d['decision_type'] in {'V1_STRONG_NO_GT_DENSITY_NEUTRAL_RECOVERY','V2_MEDIUM_NO_GT_DENSITY_NEUTRAL_RECOVERY'}:
        assert float(d['selected_A1_candidate']['agreement_source_count']) >= 2
        assert float(d['selected_A10_candidate']['agreement_source_count']) >= 2
