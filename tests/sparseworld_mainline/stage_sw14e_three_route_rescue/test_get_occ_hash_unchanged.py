from __future__ import annotations

from conftest import load_json


def test_get_occ_hash_unchanged() -> None:
    manifest = load_json("sw14e_baseline_freeze_manifest.json")
    assert manifest["get_occ_unchanged"] is True
    assert manifest["hashes"]["get_occ_source"]
    assert len(manifest["hashes"]["get_occ_source"]) == 64
