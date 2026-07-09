import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_multi_source_agreement_manifest.csv').open()))
def test_multi_source_agreement_schema():
    assert rows
    req={'sample_index','perturbation_id','horizon_s','required_sources','available_sources','agreement_source_count','agreement_mean','missing_sources','uses_gt','uses_future_info','uses_current_clean_same_frame'}
    assert req.issubset(rows[0])
