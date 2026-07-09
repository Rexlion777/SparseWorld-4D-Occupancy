from __future__ import annotations

from conftest import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, load_json


def test_h2_h5_outputs_exist() -> None:
    required_reports = [
        "sw14h2_true_local_attention_config.json",
        "sw14h2_true_local_attention_topk_precision.csv",
        "sw14h2_true_local_attention_final_decision.json",
        "sw14h3_dense_neighborhood_config.json",
        "sw14h3_dense_neighborhood_topk_precision.csv",
        "sw14h3_dense_neighborhood_final_decision.json",
        "sw14h4_component_morphology_config.json",
        "sw14h4_component_morphology_topk_precision.csv",
        "sw14h4_component_morphology_final_decision.json",
        "sw14h5_score_ensemble_search.csv",
        "sw14h5_score_ensemble_final_decision.json",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14h2_true_local_attention_scores.pt",
        "sw14h3_dense_neighborhood_scores.pt",
        "sw14h4_component_morphology_scores.pt",
        "checkpoints/sw14h2_true_local_attention_best.pth",
        "checkpoints/sw14h3_dense_neighborhood_ranker_best.pth",
        "checkpoints/sw14h4_component_morphology_ranker_best.pth",
    ]:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14h2_true_local_attention_precision.png",
        "sw14h3_dense_neighborhood_precision.png",
        "sw14h4_component_morphology_precision.png",
    ]:
        path = FIGURES_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def test_h2_h5_decision_schema_and_target_honesty() -> None:
    names = [
        "sw14h2_true_local_attention_final_decision.json",
        "sw14h3_dense_neighborhood_final_decision.json",
        "sw14h4_component_morphology_final_decision.json",
        "sw14h5_score_ensemble_final_decision.json",
    ]
    for name in names:
        payload = load_json(name)
        assert payload["target_precision_at_100k"] == 0.9
        assert isinstance(payload["target_reached"], bool)
        assert payload["uses_eval_debug"] is False
        assert payload["uses_core100_or_core500"] is False
        assert payload["gt_used_as_inference_feature"] is False
        if payload["target_reached"]:
            best = payload.get("best_val_top100k", payload.get("best", {}))
            assert best.get("precision", best.get("p100000")) >= 0.9


def test_h6_outputs_exist_and_are_honest() -> None:
    required_reports = [
        "sw14h6_postprocess_survival_config.json",
        "sw14h6_postprocess_survival_topk_precision.csv",
        "sw14h6_postprocess_survival_final_decision.json",
        "sw14h6b_survival_score_blend_search.csv",
        "sw14h6b_survival_score_blend_final_decision.json",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14h6_postprocess_survival_features_train.pt",
        "sw14h6_postprocess_survival_features_val.pt",
        "sw14h6_postprocess_survival_scores.pt",
        "checkpoints/sw14h6_postprocess_survival_ranker_best.pth",
    ]:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    path = FIGURES_DIR / "sw14h6_postprocess_survival_precision.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_h6_target_and_protocol_claims() -> None:
    payload = load_json("sw14h6_postprocess_survival_final_decision.json")
    assert payload["target_precision_at_100k"] == 0.9
    assert payload["target_reached"] is False
    assert payload["uses_eval_debug"] is False
    assert payload["uses_core100_or_core500"] is False
    assert payload["gt_used_as_inference_feature"] is False
    assert payload["whether_resume_claim_allowed"] is False
    assert payload["sw13_remains_main_result"] is True
    assert payload["best_val_top100k"]["precision"] < 0.9
    blend = load_json("sw14h6b_survival_score_blend_final_decision.json")
    assert blend["target_precision_at_100k"] == 0.9
    assert blend["target_reached"] is False
    assert blend["gt_used_as_inference_feature"] is False
    assert blend["resume_claim"] is False
    assert blend["best"]["p100000"] < 0.9


def test_h7_outputs_exist_and_are_honest() -> None:
    required_reports = [
        "sw14h7_groupwise_config.json",
        "sw14h7_groupwise_topk_precision.csv",
        "sw14h7_groupwise_final_decision.json",
        "sw14h7b_groupwise_blend_search.csv",
        "sw14h7b_groupwise_blend_final_decision.json",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14h7_groupwise_features_train.pt",
        "sw14h7_groupwise_features_val.pt",
        "sw14h7_groupwise_scores.pt",
        "checkpoints/sw14h7_groupwise_ranker_best.pth",
    ]:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    path = FIGURES_DIR / "sw14h7_groupwise_precision.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_h7_target_and_protocol_claims() -> None:
    payload = load_json("sw14h7_groupwise_final_decision.json")
    assert payload["target_precision_at_100k"] == 0.9
    assert payload["target_reached"] is False
    assert payload["uses_eval_debug"] is False
    assert payload["uses_core100_or_core500"] is False
    assert payload["gt_used_as_inference_feature"] is False
    assert payload["whether_resume_claim_allowed"] is False
    assert payload["best_val_top100k"]["precision"] < 0.9
    blend = load_json("sw14h7b_groupwise_blend_final_decision.json")
    assert blend["target_precision_at_100k"] == 0.9
    assert blend["target_reached"] is False
    assert blend["gt_used_as_inference_feature"] is False
    assert blend["resume_claim"] is False
    assert blend["best"]["p100000"] < 0.9


def test_h8_query_ranker_outputs_exist_and_are_honest() -> None:
    required_reports = [
        "sw14h8_query_ranker_config.json",
        "sw14h8_query_ranker_training_log.csv",
        "sw14h8_query_ranker_topk_precision.csv",
        "sw14h8_query_ranker_selection_sweep_val.csv",
        "sw14h8_query_ranker_final_decision.json",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14h8_online_query_score_dump_train_0_99.pt",
        "sw14h8_online_query_score_dump_val_100_149.pt",
        "sw14h8_query_ranker_scores.pt",
        "checkpoints/sw14h8_query_ranker_best.pth",
    ]:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    path = FIGURES_DIR / "sw14h8_query_ranker_precision.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_h8_target_and_protocol_claims() -> None:
    payload = load_json("sw14h8_query_ranker_final_decision.json")
    assert payload["target_precision_at_100k"] == 0.9
    assert payload["target_reached"] is False
    assert payload["uses_eval_debug"] is False
    assert payload["uses_core100_or_core500"] is False
    assert payload["gt_used_as_inference_feature"] is False
    assert payload["whether_resume_claim_allowed"] is False
    assert payload["sw13_remains_main_result"] is True
    assert payload["best_val_top100k"]["precision"] < 0.9
    assert payload["h7b_val_top100k"]["precision"] < 0.9


def test_h9_h10_compact_sparse_outputs_exist_and_are_honest() -> None:
    required_reports = [
        "sw14h9_sparse_contributor_dump_config.json",
        "sw14h9_sparse_contributor_dump_summary_train.json",
        "sw14h9_sparse_contributor_dump_summary_val.json",
        "sw14h9_sparse_contributor_val_feature_precision.csv",
        "sw14h10_group_budget_audit.csv",
        "sw14h10_compact_sparse_config.json",
        "sw14h10_compact_sparse_training_log.csv",
        "sw14h10_compact_sparse_topk_precision.csv",
        "sw14h10_compact_sparse_selection_sweep_val.csv",
        "sw14h10_compact_sparse_final_decision.json",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14h9_sparse_contributor_dump_train_0_99.pt",
        "sw14h9_sparse_contributor_dump_val_100_149.pt",
        "sw14h10_compact_sparse_scores.pt",
        "checkpoints/sw14h10_compact_sparse_ranker_best.pth",
    ]:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    path = FIGURES_DIR / "sw14h10_compact_sparse_precision.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_h10_target_and_protocol_claims() -> None:
    payload = load_json("sw14h10_compact_sparse_final_decision.json")
    assert payload["target_precision_at_100k"] == 0.9
    assert payload["target_reached"] is False
    assert payload["uses_eval_debug"] is False
    assert payload["uses_core100_or_core500"] is False
    assert payload["gt_used_as_inference_feature"] is False
    assert payload["whether_resume_claim_allowed"] is False
    assert payload["sw13_remains_main_result"] is True
    assert payload["best_val_top100k"]["precision"] < 0.9
    assert payload["h7b_val_top100k"]["precision"] < 0.9


def test_h11_h12_h13_outputs_exist_and_are_honest() -> None:
    required_reports = [
        "sw14h11_bev_completion_config.json",
        "sw14h11_bev_completion_training_log.csv",
        "sw14h11_bev_completion_topk_precision.csv",
        "sw14h11_bev_completion_final_decision.json",
        "sw14h12_sklearn_hgb_topk_precision.csv",
        "sw14h12_sklearn_hgb_final_decision.json",
        "sw14h13_bev_topk_earlystop_training_log.csv",
        "sw14h13_bev_topk_earlystop_val_precision_by_epoch.csv",
        "sw14h13_bev_topk_earlystop_final_decision.json",
    ]
    for name in required_reports:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    for name in [
        "sw14h11_bev_completion_scores.pt",
        "sw14h13_bev_topk_earlystop_scores.pt",
        "checkpoints/sw14h11_bev_completion_best.pth",
        "checkpoints/sw14h13_bev_topk_earlystop_best.pth",
    ]:
        path = ARTIFACTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def test_h11_h12_h13_target_and_protocol_claims() -> None:
    for name in [
        "sw14h11_bev_completion_final_decision.json",
        "sw14h12_sklearn_hgb_final_decision.json",
        "sw14h13_bev_topk_earlystop_final_decision.json",
    ]:
        payload = load_json(name)
        assert payload["target_precision_at_100k"] == 0.9
        assert payload["target_reached"] is False
        assert payload["uses_eval_debug"] is False
        assert payload["uses_core100_or_core500"] is False
        assert payload["gt_used_as_inference_feature"] is False
        best = payload["best_val_top100k"]
        assert best["precision"] < 0.9
    h11_payload = load_json("sw14h11_bev_completion_final_decision.json")
    assert h11_payload["whether_resume_claim_allowed"] is False
    assert h11_payload["sw13_remains_main_result"] is True
    h12_payload = load_json("sw14h12_sklearn_hgb_final_decision.json")
    assert h12_payload["resume_claim"] is False
    h13_payload = load_json("sw14h13_bev_topk_earlystop_final_decision.json")
    assert h13_payload["whether_resume_claim_allowed"] is False
    assert h13_payload["sw13_remains_main_result"] is True
