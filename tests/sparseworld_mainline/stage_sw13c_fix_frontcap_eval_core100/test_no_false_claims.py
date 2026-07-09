from pathlib import Path
text=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/stage_sw13c_fix_frontcap_eval_core100_report.md').read_text().lower()
def test_no_false_claims():
    assert 'not official benchmark' in text
    assert 'subset diagnostic only' in text
    assert 'trained model improvement' not in text
