from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")
ART = Path("/home/rexlion/ComputerVision/cv_lidar_transition/artifacts/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")
FIG = Path("/home/rexlion/ComputerVision/cv_lidar_transition/projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_outputs_exist():
    names = [
        "sw14c_round2b_inherited_state.json",
        "sw14c_round2b_inherited_state.md",
        "sw14c_round2b_adapter_config.json",
        "sw14c_round2b_adapter_init_audit.json",
        "sw14c_round2b_region_mask_stats_train.csv",
        "sw14c_round2b_region_mask_stats_val.csv",
        "sw14c_round2b_region_mask_summary.json",
        "sw14c_round2b_loss_config.json",
        "sw14c_round2b_loss_terms_schema.json",
        "sw14c_round2b_training_log.csv",
        "sw14c_round2b_epoch_val_metrics.csv",
        "sw14c_round2b_training_summary.json",
        "sw14c_round2b_residual_health_train.csv",
        "sw14c_round2b_residual_health_val.csv",
        "sw14c_round2b_residual_health_summary.json",
        "sw14c_round2b_teacher_error_behavior_val.csv",
        "sw14c_round2b_teacher_error_behavior_summary.json",
        "sw14c_round2b_gamma_sweep_val.csv",
        "sw14c_round2b_gamma_selection.json",
        "sw14c_round2b_final_decision.json",
        "stage_sw14c_round2b_gate_loss_rescue_report.md",
    ]
    for name in names:
        path = BASE / name
        assert path.exists() and path.stat().st_size > 0, name
    for name in ["sw14c_round2b_checkpoint_epoch0.pth", "sw14c_round2b_best_checkpoint.pth"]:
        path = ART / "checkpoints" / name
        assert path.exists() and path.stat().st_size > 0, name
    for name in [
        "sw14c_round2b_gate_health.png",
        "sw14c_round2b_loss_curves.png",
        "sw14c_round2b_gamma_tradeoff.png",
        "sw14c_round2b_teacher_error_behavior.png",
        "sw14c_round2b_val_teacher_vs_student_bar.png",
    ]:
        path = FIG / name
        assert path.exists() and path.stat().st_size > 0, name
