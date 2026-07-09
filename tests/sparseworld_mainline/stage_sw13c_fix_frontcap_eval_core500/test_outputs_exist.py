from pathlib import Path
BASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500')
def test_outputs_exist():
    names=['sw13c_fix_frontcap_eval_core500_protocol_fingerprint.json','sw13c_fix_frontcap_eval_core500_shard_plan.json','sw13c_fix_frontcap_eval_core500_merge_manifest.json','sw13c_fix_frontcap_eval_core500_merged_metrics.csv','sw13c_fix_frontcap_eval_core500_metric_summary.csv','sw13c_fix_frontcap_eval_core500_density_audit.csv','sw13c_fix_frontcap_eval_core500_sample_consistency.csv','sw13c_fix_frontcap_eval_core500_decision.json','stage_sw13c_fix_frontcap_eval_core500_report.md']
    for name in names:
        p=BASE/name
        assert p.exists() and p.stat().st_size>0, name
