from __future__ import annotations

from conftest import load_json


def test_gamma_zero_not_improvement() -> None:
    gamma = load_json("sw14c_oracle_mask_gamma_selection.json")
    assert gamma["gamma_zero_pair_counts_as_improvement"] is False
    zero_rows = [
        row
        for row in gamma["summary_rows"]
        if abs(float(row["gamma_add"])) < 1e-12 and abs(float(row["gamma_sup"])) < 1e-12
    ]
    assert len(zero_rows) == 1
    assert zero_rows[0]["allowed_as_improvement"] is False
    assert gamma["selected_pair_is_nonzero"] is True
