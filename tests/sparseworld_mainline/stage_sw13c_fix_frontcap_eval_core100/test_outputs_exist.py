from pathlib import Path
BASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100')
def test_outputs_exist():
    names=['sw13c_fix_frontcap_eval_core100_protocol_fingerprint.json','sw13c_fix_frontcap_eval_core100_shard_plan.json','sw13c_fix_frontcap_eval_core100_merge_manifest.json','sw13c_fix_frontcap_eval_core100_merged_metrics.csv','sw13c_fix_frontcap_eval_core100_metric_summary.csv','sw13c_fix_frontcap_eval_core100_density_audit.csv','sw13c_fix_frontcap_eval_core100_sample_consistency.csv','sw13c_fix_frontcap_eval_core100_decision.json','stage_sw13c_fix_frontcap_eval_core100_report.md']
    for name in names:
        p=BASE/name
        assert p.exists() and p.stat().st_size>0, name
