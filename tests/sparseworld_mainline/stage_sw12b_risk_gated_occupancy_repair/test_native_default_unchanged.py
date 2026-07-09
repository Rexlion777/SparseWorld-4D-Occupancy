from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
MANIFEST = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/branchA_routing_variant_manifest.json"


def test_native_default_unchanged() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["native_default_unchanged"] is True
