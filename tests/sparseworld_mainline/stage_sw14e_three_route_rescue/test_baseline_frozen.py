from __future__ import annotations

from conftest import load_json


def test_baseline_frozen() -> None:
    manifest = load_json("sw14e_baseline_freeze_manifest.json")
    assert manifest["decision"] == "SW14E_INIT_READY"
    assert manifest["sparseworld_backbone_unchanged"] is True
    assert manifest["occupancy_head_unchanged"] is True
    assert manifest["get_occ_unchanged"] is True
    assert manifest["f3_config_unchanged"] is True
    assert manifest["frontcap_config_unchanged"] is True
    assert manifest["sw13_teacher_output_source_fixed"]
    assert manifest["sw14d_a_checkpoint_source_fixed"]
