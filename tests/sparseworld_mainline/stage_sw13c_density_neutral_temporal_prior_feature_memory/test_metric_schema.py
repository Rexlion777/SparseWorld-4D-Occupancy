import csv
from pathlib import Path
def test_metric_schema():
    rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_metric_summary.csv').open())); assert rows
    req={'front_sector_false_free_rate','pred_gt_occupied_ratio','density_delta','recovery_retention_ratio','density_reduction_ratio','protected_zone_preservation_ratio','low_value_removed_mean'}; assert req.issubset(rows[0])
