from pathlib import Path
BASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget')
def test_outputs_exist():
    for n in ['sw13c_fix_inherited_sw13c_audit.json','sw13c_fix_no_gt_density_metrics.csv','sw13c_fix_metric_summary.csv','sw13c_fix_no_gt_candidate_selection.json','sw13c_fix_front_local_density_audit.csv','sw13c_fix_pruning_ablation_metrics.csv','sw13c_fix_no_gt_density_budget_decision.json','stage_sw13c_fix_no_gt_density_budget_report.md']:
        p=BASE/n
        assert p.exists() and p.stat().st_size>0, n
