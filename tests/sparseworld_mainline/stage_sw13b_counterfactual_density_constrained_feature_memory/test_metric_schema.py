import csv
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_metric_summary.csv')

def test_metric_schema() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    required = {
        'perturbation_id','base_repair_variant','filter_label','horizon_s','front_sector_false_free_rate','small_object_false_free_rate',
        'future_h4_h6_false_free_rate','new_visible_recall','A10_front_h6_recovery_rate','false_positive_rate','pred_gt_occupied_ratio',
        'recovery_retention_ratio','density_reduction_ratio','delta_keep_ratio','temporal_agreement_score','boundary_distance_mean'
    }
    assert required.issubset(rows[0].keys())
