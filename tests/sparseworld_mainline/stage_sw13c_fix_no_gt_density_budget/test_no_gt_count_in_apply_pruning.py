from pathlib import Path
SRC=Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py').read_text()
def test_no_gt_count_in_apply_pruning():
    assert 'def apply_pruning_no_gt' in SRC
    chunk=SRC.split('def apply_pruning_no_gt',1)[1].split('def build_eval_row',1)[0]
    assert 'gt_occ_count' not in chunk
