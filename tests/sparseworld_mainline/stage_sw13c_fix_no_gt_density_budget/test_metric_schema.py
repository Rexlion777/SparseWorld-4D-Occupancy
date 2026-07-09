import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_metric_summary.csv').open()))
def test_metric_schema():
    assert rows
    req={'no_gt_budget','budget_mode','native_occ_count','raw_occ_count','final_occ_count','raw_delta_count','final_native_expansion_ratio','raw_delta_keep_ratio','protected_zone_preservation_ratio','pruning_from_protected_ratio','nonfront_pruning_ratio','agreement_source_count','selection_uses_gt'}
    assert any(req.issubset(row) for row in rows)
