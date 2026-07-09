from __future__ import annotations

from conftest import load_csv


def test_eval_schema() -> None:
    rows = load_csv("sw14d_candidate_reranker_eval_val.csv")
    assert len(rows) == 200
    required = {
        "selected_add_count",
        "selected_suppress_count",
        "selected_frontcap_suppress_count",
        "add_recovers_front_gt_occ_count",
        "suppress_breaks_front_gt_occ_count",
        "frontcap_breaks_gt_occ_count",
        "frontcap_breaks_added_gt_occ_count",
        "fn_reduction_rate",
        "front_fn_reduction_rate",
        "fp_reduction_rate",
        "density_delta_over_teacher",
        "false_positive_delta_over_teacher",
        "teacher_front_local_proxy",
        "front_local_proxy",
        "front_local_delta_over_teacher",
        "front_local_safety_pass",
        "broken_correct_rate",
        "safety_pass",
        "net_score",
    }
    assert required.issubset(rows[0])
