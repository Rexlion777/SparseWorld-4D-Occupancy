from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_alpha_scale_applied_correctly():
    assert 'global_scale' in src
    assert 'camera_scale_tensor' in src
