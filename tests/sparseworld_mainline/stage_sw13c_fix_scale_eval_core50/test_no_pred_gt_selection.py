import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_replay_manifest.csv').open()))
def test_no_pred_gt_selection():
    assert rows
    assert all(r['uses_pred_gt_density_for_selection'] == 'False' for r in rows)
