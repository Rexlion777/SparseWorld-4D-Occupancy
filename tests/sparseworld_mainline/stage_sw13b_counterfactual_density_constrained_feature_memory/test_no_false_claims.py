from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/stage_sw13b_counterfactual_density_constrained_feature_memory_report.md')

def test_no_false_claims() -> None:
    text = PATH.read_text(encoding='utf-8').lower()
    assert 'subset diagnostic' in text
    assert 'not official benchmark' in text
    banned = ['official benchmark result','trained model improvement','beats the paper']
    for phrase in banned:
        assert phrase not in text
