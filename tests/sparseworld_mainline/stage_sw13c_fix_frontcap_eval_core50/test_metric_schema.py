import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_metrics.csv').open()))
def test_metric_schema():
    assert rows
    need={'sample_index','perturbation_id','base_repair_variant','front_cap_variant','cap_ratio','horizon_s','front_local_density_proxy_before','front_local_density_proxy_after','front_cap_pruned_count','front_pruning_from_protected_ratio','front_cap_target_unmet','pred_gt_density_delta','protected_zone_preservation_ratio','no_gt_front_cap'}
    assert need.issubset(rows[0].keys())
