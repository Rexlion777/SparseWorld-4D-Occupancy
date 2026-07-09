from __future__ import annotations

from conftest import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR


def test_outputs_exist() -> None:
    reports = [
        "sw14d_candidate_reranker_inherited_state.json",
        "sw14d_candidate_reranker_candidate_stats_train.csv",
        "sw14d_candidate_reranker_training_log.csv",
        "sw14d_candidate_reranker_threshold_selection.json",
        "sw14d_candidate_reranker_eval_train.csv",
        "sw14d_candidate_reranker_eval_val.csv",
        "sw14d_candidate_reranker_eval_summary.json",
        "sw14d_candidate_reranker_final_decision.json",
        "stage_sw14d_candidate_reranker_report.md",
    ]
    figures = ["sw14d_candidate_reranker_behavior.png", "sw14d_candidate_reranker_train_val.png"]
    for name in reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in figures:
        path = FIGURES_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    ckpt = ARTIFACTS_DIR / "checkpoints/sw14d_candidate_reranker_best.pth"
    assert ckpt.exists()
    assert ckpt.stat().st_size > 0
