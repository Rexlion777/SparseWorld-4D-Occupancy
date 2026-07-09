from __future__ import annotations

from conftest import load_json


def _check_selection(name: str) -> None:
    selection = load_json(name)
    assert "safety_pass_all" in selection
    assert "recall_nonregression_pass" in selection
    assert "mean_density_delta_over_teacher" in selection
    assert "mean_false_positive_delta_over_teacher" in selection
    assert "mean_front_local_proxy" in selection
    assert selection["add_strength"] in {"strong", "strong_medium"}


def test_selection_constraints() -> None:
    _check_selection("sw14h_selection_summary.json")
    _check_selection("sw14i_selection_summary.json")
