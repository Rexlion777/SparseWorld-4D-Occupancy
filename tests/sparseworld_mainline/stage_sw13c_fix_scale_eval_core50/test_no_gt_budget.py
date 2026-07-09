import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_metrics.csv').open()))
def test_no_gt_budget():
    assert rows
    assert all(r['uses_gt_budget'] == 'False' for r in rows)
    assert all(r['no_gt_budget'] == 'True' for r in rows)
