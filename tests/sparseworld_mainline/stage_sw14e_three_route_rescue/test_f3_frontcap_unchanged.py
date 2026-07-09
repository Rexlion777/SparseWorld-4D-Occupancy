from __future__ import annotations

from conftest import load_json


def test_f3_frontcap_unchanged() -> None:
    manifest = load_json("sw14e_baseline_freeze_manifest.json")
    assert manifest["f3_config_unchanged"] is True
    assert manifest["frontcap_config_unchanged"] is True
    assert manifest["hashes"]["sparseworld_config"]
