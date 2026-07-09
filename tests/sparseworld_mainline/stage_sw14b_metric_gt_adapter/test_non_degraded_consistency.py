from pathlib import Path
SRC = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_modules.py').read_text()
def test_non_degraded_consistency():
    assert 'alpha_map = alpha_map * degradation_mask' in SRC
