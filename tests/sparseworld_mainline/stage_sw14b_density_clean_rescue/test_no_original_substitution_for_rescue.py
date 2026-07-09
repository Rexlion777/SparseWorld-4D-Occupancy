from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_no_original_substitution_for_rescue():
    assert ('original_sw14b' + '_equivalent_noop_alpha_policy') not in src
    assert 'not substituted with original SW14B metrics' in src
    assert '0.0 < scale < 1.0' in src
