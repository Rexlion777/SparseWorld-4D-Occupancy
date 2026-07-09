import csv
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_delta_filter_metrics.csv')

def test_native_occ_unchanged() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    for row in rows:
        if row['perturbation_id'] in {'A1_drop_cam_front','A10_drop_front_triplet','C4_motion_blur_9'} and row['filter_label'] not in {'NATIVE_REFERENCE', 'D0_none'}:
            assert row['native_occ_unchanged'] == 'True'
