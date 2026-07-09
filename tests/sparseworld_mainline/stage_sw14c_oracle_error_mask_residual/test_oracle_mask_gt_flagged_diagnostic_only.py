from __future__ import annotations

from conftest import load_json


def test_oracle_mask_gt_flagged_diagnostic_only() -> None:
    inherited = load_json("sw14c_oracle_mask_inherited_state.json")
    masks = load_json("sw14c_error_mask_summary.json")
    upper_bound = load_json("sw14c_oracle_candidate_upper_bound_summary.json")
    assert inherited["diagnostic_upper_bound_only"] is True
    assert inherited["oracle_mask_uses_gt"] is True
    assert inherited["oracle_mask_is_not_deployable"] is True
    assert masks["oracle_mask_uses_gt"] is True
    assert upper_bound["oracle_uses_gt"] is True
