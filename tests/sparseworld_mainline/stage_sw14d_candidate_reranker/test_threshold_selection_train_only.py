from __future__ import annotations

from conftest import load_json


def test_threshold_selection_train_only() -> None:
    thresholds = load_json("sw14d_candidate_reranker_threshold_selection.json")
    assert thresholds["gt_used_for_threshold_selection"] == "train_split_only"
    assert thresholds["val_gt_used_for_threshold_selection"] is False
    assert thresholds["config"]["post_rerank_front_cap"] is True
    assert float(thresholds["config"]["front_cap_ratio"]) <= 1.30
