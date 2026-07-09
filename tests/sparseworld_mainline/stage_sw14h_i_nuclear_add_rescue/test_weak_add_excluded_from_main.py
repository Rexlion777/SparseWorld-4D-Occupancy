from __future__ import annotations

from conftest import load_json


def test_weak_add_excluded_from_main() -> None:
    manifest = load_json("sw14hi_baseline_freeze_manifest.json")["inference_feature_manifest"]
    final = load_json("sw14hi_nuclear_add_rescue_final_decision.json")
    assert manifest["weak_add_excluded_from_main"] is True
    assert final["weak_add_excluded_from_main"] is True
