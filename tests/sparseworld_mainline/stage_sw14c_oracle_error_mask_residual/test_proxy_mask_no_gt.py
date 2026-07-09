from __future__ import annotations

from conftest import load_json


def test_proxy_mask_no_gt() -> None:
    masks = load_json("sw14c_error_mask_summary.json")
    proxy = load_json("sw14c_proxy_mask_feasibility_summary.json")
    assert masks["proxy_mask_uses_gt"] is False
    assert proxy["proxy_mask_uses_gt"] is False
    assert proxy["proxy_training_executed"] is False
