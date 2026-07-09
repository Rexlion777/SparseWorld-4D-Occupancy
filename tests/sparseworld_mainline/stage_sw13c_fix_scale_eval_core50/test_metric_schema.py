import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_metrics.csv').open()))
def test_metric_schema():
    assert len(rows) >= 50
    need={'sample_index','perturbation_id','base_repair_variant','fixed_candidate_label','horizon_s','selection_frozen','no_reselection','no_gt_budget','uses_gt_budget','uses_pred_gt_density_for_selection','agreement_source_count','final_native_expansion_ratio','raw_delta_keep_ratio','protected_zone_preservation_ratio','pruning_from_protected_ratio','front_kept_ratio','front_local_density_proxy','pred_gt_density_delta'}
    assert need.issubset(rows[0].keys())
