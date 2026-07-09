from pathlib import Path
text=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/stage_sw13c_fix_no_gt_density_budget_report.md').read_text().lower()
def test_no_false_claims():
    assert 'not official benchmark' in text
    assert 'trained model improvement' not in text
    assert 'original sw-13c had gt-density-budget risk' in text
