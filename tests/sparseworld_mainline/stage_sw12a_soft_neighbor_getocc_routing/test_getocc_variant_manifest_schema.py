from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing/sw12a_getocc_variant_manifest.json"


def test_getocc_variant_manifest_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["default_variant"] == "native"
    assert payload["native_default_unchanged_requirement"] is True
    variants = payload["variants"]
    assert isinstance(variants, list) and len(variants) >= 10
    names = {item["variant_name"] for item in variants}
    assert {"native", "soft_neighbor_r1", "confidence_gated_r1", "density_capped_r1", "survival_oracle_r1_diagnostic"} <= names
