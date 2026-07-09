import csv
from pathlib import Path
def test_pruning_does_not_target_gt():
    rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_density_neutral_metrics.csv').open())); assert rows
    assert all(r['uses_gt_repair']=='False' for r in rows)
