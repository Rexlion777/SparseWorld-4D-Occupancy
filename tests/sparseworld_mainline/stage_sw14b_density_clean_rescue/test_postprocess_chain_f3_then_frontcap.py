from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_postprocess_chain_f3_then_frontcap():
    assert 'apply_pruning_no_gt' in src
    assert 'apply_front_local_cap_no_gt' in src
