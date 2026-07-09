import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/sw13c_fix_frontcap_eval_core100_merged_metrics.csv').open()))
def test_metric_schema():
    assert rows
    need={'sample_index','shard_id','candidate_name','perturbation_id','base_repair_variant','front_cap_variant','cap_ratio','horizon_s','no_gt_front_cap','no_gt_budget','no_training','no_reselection','no_retuning','pred_gt_density_delta','front_sector_false_free_rate_delta_vs_native'}
    assert need.issubset(rows[0].keys())
