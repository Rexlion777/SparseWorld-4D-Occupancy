from __future__ import annotations

from conftest import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR


def test_outputs_exist() -> None:
    required_reports = [
        "sw14c_oracle_mask_inherited_state.json",
        "sw14c_error_mask_summary.json",
        "sw14c_dual_branch_adapter_config.json",
        "sw14c_dual_branch_adapter_init_audit.json",
        "sw14c_oracle_candidate_upper_bound_summary.json",
        "sw14c_oracle_mask_loss_config.json",
        "sw14c_oracle_mask_training_log.csv",
        "sw14c_oracle_mask_epoch_val_metrics.csv",
        "sw14c_oracle_mask_residual_health_summary.json",
        "sw14c_oracle_mask_teacher_error_behavior_summary.json",
        "sw14c_oracle_mask_gamma_selection.json",
        "sw14c_proxy_mask_feasibility_summary.json",
        "sw14c_oracle_error_mask_residual_final_decision.json",
        "stage_sw14c_oracle_error_mask_residual_report.md",
    ]
    required_figures = [
        "sw14c_oracle_mask_region_stats.png",
        "sw14c_oracle_upper_bound_bar.png",
        "sw14c_masked_residual_health.png",
        "sw14c_teacher_error_behavior_bar.png",
        "sw14c_gamma2d_heatmap.png",
        "sw14c_oracle_mask_gamma2d_heatmap.png",
        "sw14c_oracle_vs_proxy_comparison.png",
        "sw14c_final_decision_flow.png",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in required_figures:
        path = FIGURES_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    ckpt = ARTIFACTS_DIR / "checkpoints/sw14c_oracle_mask_variantB_best.pth"
    assert ckpt.exists()
    assert ckpt.stat().st_size > 0
