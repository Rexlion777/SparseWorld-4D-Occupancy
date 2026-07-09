from pathlib import Path


REPORTS = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")
FIGURES = Path("/home/rexlion/ComputerVision/cv_lidar_transition/projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_outputs_exist() -> None:
    names = [
        "sw14c_checkpoint_sanity_audit.json",
        "sw14c_checkpoint_param_diff.csv",
        "sw14c_residual_magnitude_train.csv",
        "sw14c_residual_magnitude_val.csv",
        "sw14c_residual_gate_train.csv",
        "sw14c_residual_gate_val.csv",
        "sw14c_gamma_applied_feature_delta.csv",
        "sw14c_stagewise_diff_val.csv",
        "sw14c_stagewise_diff_summary.json",
        "sw14c_teacher_error_regions_train.csv",
        "sw14c_teacher_error_regions_val.csv",
        "sw14c_teacher_error_region_summary.json",
        "sw14c_teacher_error_sample_rank.csv",
        "sw14c_student_behavior_on_teacher_errors.csv",
        "sw14c_student_error_region_behavior_summary.json",
        "sw14c_loss_dominance_audit.csv",
        "sw14c_loss_dominance_summary.json",
        "sw14c_gamma_sensitivity_audit.csv",
        "sw14c_residual_learning_audit_decision.json",
        "stage_sw14c_residual_learning_audit_report.md",
    ]
    for name in names:
        path = REPORTS / name
        assert path.exists() and path.stat().st_size > 0, name
    plot = FIGURES / "sw14c_gamma_sensitivity_plot.png"
    assert plot.exists() and plot.stat().st_size > 0
