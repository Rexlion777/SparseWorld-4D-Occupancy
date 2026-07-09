from __future__ import annotations

from conftest import load_json


def test_baseline_frozen() -> None:
    freeze = load_json("sw14hi_baseline_freeze_manifest.json")
    assert freeze["decision"] == "BASELINE_FROZEN"
    assert freeze["sparseworld_backbone_unchanged"] is True
    assert freeze["occupancy_head_unchanged"] is True
    assert freeze["sw13_main_result_unchanged"] is True


def test_get_occ_hash_unchanged() -> None:
    freeze = load_json("sw14hi_baseline_freeze_manifest.json")
    assert freeze["get_occ_unchanged"] is True


def test_f3_frontcap_unchanged() -> None:
    freeze = load_json("sw14hi_baseline_freeze_manifest.json")
    assert freeze["f3_config_unchanged"] is True
    assert freeze["frontcap_config_unchanged"] is True
