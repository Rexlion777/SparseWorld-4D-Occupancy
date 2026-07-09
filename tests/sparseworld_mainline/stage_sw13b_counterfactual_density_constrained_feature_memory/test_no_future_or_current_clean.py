import csv
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_delta_filter_metrics.csv')

def test_no_future_or_current_clean() -> None:
    rows = list(csv.DictReader(PATH.open()))
    for row in rows:
        assert row['uses_future_info'] == 'False'
        assert row['uses_current_clean_same_frame'] == 'False'
        assert row['uses_gt_repair'] == 'False'
