from __future__ import annotations

from conftest import load_json


def test_sw14d_b_selection_constraints() -> None:
    route1 = load_json("sw14d_b_final_decision.json")
    selected = route1["selected_budget"]
    assert selected["safety_pass_all"] is True
    assert selected["recall_nonregression_pass"] is True
    assert float(selected["mean_front_fn_reduction_rate"]) >= -0.001
    assert float(selected["mean_future_h4h6_fn_reduction_rate"]) >= -0.001
    assert float(selected["mean_fp_reduction_rate"]) > 0.0
    assert float(selected["mean_density_delta_over_teacher"]) <= 0.003
    assert float(selected["mean_false_positive_delta_over_teacher"]) <= 0.001
    assert float(selected["front_local_no_worse_pass_rate"]) == 1.0
    assert "Absolute pass rate is reported separately" in route1["front_local_safety_rule"]
