from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"


def test_branchA_manifest_and_candidates() -> None:
    manifest = json.loads((REPORTS / "branchA_routing_variant_manifest.json").read_text(encoding="utf-8"))
    assert manifest["native_default_unchanged"] is True
    assert len(manifest["variants"]) >= 6
    with (REPORTS / "branchA_routing_candidates.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    allowed = {"A_SAFE_TARGETED", "A_TARGETED_UNSAFE", "A_SAFE_NO_TARGET", "A_ORACLE_ONLY", "A_NO_SIGNAL"}
    for row in rows:
        assert row["classification"] in allowed
