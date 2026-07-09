import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_front_local_density_audit.csv').open()))
def test_front_local_density_audit():
    assert rows
    need={'front_occ_count_native','front_occ_count_raw','front_occ_count_final','front_native_expansion_ratio','front_raw_delta_keep_ratio','front_protected_keep_ratio','front_local_density_proxy','front_gt_occupied_count','front_pred_gt_density_delta','front_false_positive_delta','front_false_free_delta'}
    assert need.issubset(rows[0].keys())
