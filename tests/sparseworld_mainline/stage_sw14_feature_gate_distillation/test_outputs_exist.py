from pathlib import Path
BASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation')
REQUIRED=['sw14_inherited_sw13c_frontcap_audit.json','teacher_candidate_manifest.json','sw14_adapter_architecture.md','sw14_feature_path_audit.json','sw14_native_path_integrity.json','sw14_teacher_dump_manifest.csv','sw14_train_tune_split_manifest.json','sw14a_training_log.csv','sw14a_eval_metrics.csv','sw14a_decision.json','sw14_clean_control_safety.csv','sw14_final_decision.json','stage_sw14_feature_gate_distillation_report.md']
def test_outputs_exist():
    for name in REQUIRED:
        path=BASE/name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
