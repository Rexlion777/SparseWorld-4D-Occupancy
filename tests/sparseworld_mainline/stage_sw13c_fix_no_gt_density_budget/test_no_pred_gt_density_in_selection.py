from pathlib import Path
import json
SRC=Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py').read_text()
SEL=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_no_gt_candidate_selection.json').read_text())
def test_no_pred_gt_density_in_selection():
    chunk=SRC.split('def select_candidates_no_gt',1)[1].split('def main',1)[0]
    for bad in ['pred_gt_density_delta','false_positive_delta','front_sector_false_free_rate_delta_vs_native','A10_front_h6_recovery_rate_delta_vs_native','occupied_iou','semantic_miou']:
        assert bad not in chunk
    assert SEL['no_pred_gt_density_delta_used'] is True
