from __future__ import annotations

from conftest import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR


def test_outputs_exist() -> None:
    required_reports = [
        "sw14hi_inherited_state.json",
        "sw14hi_problem_statement.md",
        "sw14hi_baseline_freeze_manifest.json",
        "sw14hi_add_failure_audit_summary.json",
        "sw14hi_add_failure_audit_train.csv",
        "sw14hi_add_failure_audit_val.csv",
        "sw14hi_add_true_vs_false_feature_gap.csv",
        "sw14h_evidence_token_config.json",
        "sw14h_training_log.csv",
        "sw14h_val_topk_precision_by_epoch.csv",
        "sw14h_selection_summary.json",
        "sw14h_final_decision.json",
        "sw14i_generation_target_stats.json",
        "sw14i_roi_stats.csv",
        "sw14i_topk_precision.csv",
        "sw14i_selection_summary.json",
        "sw14i_final_decision.json",
        "sw14hi_nuclear_add_rescue_final_decision.json",
        "stage_sw14hi_nuclear_add_rescue_report.md",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "checkpoints/sw14h_candidate_transformer_best.pth",
        "checkpoints/sw14i_generative_completion_best.pth",
        "sw14h_candidate_transformer_scores.pt",
        "sw14i_generative_completion_scores.pt",
    ]:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14hi_add_failure_true_vs_false.png",
        "sw14h_evidence_token_distribution.png",
        "sw14h_attention_examples.png",
        "sw14h_topk_precision_curve.png",
        "sw14h_selection_tradeoff.png",
        "sw14i_missing_occ_target_examples.png",
        "sw14i_generation_heatmap_examples.png",
        "sw14i_topk_precision_curve.png",
        "sw14hi_final_decision_flow.png",
    ]:
        path = FIGURES_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
