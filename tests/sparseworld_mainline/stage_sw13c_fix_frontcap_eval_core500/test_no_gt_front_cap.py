import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_merged_metrics.csv').open()))
def test_no_gt_front_cap():
    assert rows
    assert all(r['no_gt_front_cap'] == 'True' for r in rows)
