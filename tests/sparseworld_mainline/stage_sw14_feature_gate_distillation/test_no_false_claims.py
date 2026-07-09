from pathlib import Path
text=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/stage_sw14_feature_gate_distillation_report.md').read_text().lower()
def test_no_false_claims():
    assert 'subset diagnostic only' in text
    assert 'not official benchmark' in text
    assert 'official benchmark result' not in text
    assert 'trained model improvement' not in text
