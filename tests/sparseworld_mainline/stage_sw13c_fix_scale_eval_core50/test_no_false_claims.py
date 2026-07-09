from pathlib import Path
text=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/stage_sw13c_fix_scale_eval_core50_report.md').read_text().lower()
def test_no_false_claims():
    assert 'not official benchmark' in text
    assert 'trained model improvement' not in text
