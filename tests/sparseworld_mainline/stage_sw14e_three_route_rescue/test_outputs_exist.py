from __future__ import annotations

from conftest import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR


def test_outputs_exist() -> None:
    reports = [
        "sw14e_inherited_state.json",
        "sw14e_baseline_freeze_manifest.json",
        "sw14e_protocol.md",
        "sw14d_b_candidate_stats.json",
        "sw14d_b_candidate_stats.csv",
        "sw14d_b_oracle_action_upper_bound.json",
        "sw14d_b_budget_sweep_val.csv",
        "sw14d_b_selection_config.json",
        "sw14d_b_selection_summary.json",
        "sw14d_b_final_decision.json",
        "stage_sw14d_b_report.md",
        "sw14f_raw_logit_causal_probe.csv",
        "sw14f_raw_logit_causal_summary.json",
        "sw14f_final_decision.json",
        "stage_sw14f_report.md",
        "sw14g_insertion_point_audit.json",
        "sw14g_insertion_point_audit.md",
        "sw14g_final_decision.json",
        "stage_sw14g_report.md",
        "sw14e_three_route_final_decision.json",
        "stage_sw14e_three_route_rescue_report.md",
    ]
    figures = [
        "sw14e_route_overview.png",
        "sw14d_b_candidate_stats.png",
        "sw14d_b_oracle_upper_bound.png",
        "sw14d_b_budget_tradeoff.png",
        "sw14f_raw_logit_probe.png",
        "sw14f_survival_loss_effect.png",
        "sw14g_insertion_point_map.png",
        "sw14g_router_weights.png",
        "sw14e_final_decision_flow.png",
    ]
    artifacts = [
        "checkpoints/sw14d_b_best.pth",
        "candidate_tables/sw14d_b_candidate_table_train.parquet",
        "candidate_tables/sw14d_b_candidate_table_val.parquet",
        "sw14d_b_batched_scores.pt",
    ]
    for name in reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in figures:
        path = FIGURES_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in artifacts:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
