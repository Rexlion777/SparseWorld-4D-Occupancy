from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage"


def test_corrected_support_adapter_manifest_exists_and_matches_definition() -> None:
    payload = json.loads((REPORTS / "corrected_support_adapter_manifest.json").read_text(encoding="utf-8"))
    support_def = payload["support_definition"]
    assert support_def["tensor_name"] == "refine_pts_current"
    assert support_def["coord_mode"] == "decoded_metric"
    assert support_def["order_name"] == "xyz"
    assert support_def["flip_variant"] == "noflip"
    assert support_def["point_mode"] == "all48"


def test_sw35_run_manifest_effective_samples() -> None:
    payload = json.loads((REPORTS / "sw35_run_manifest.json").read_text(encoding="utf-8"))
    assert payload["effective_sample_count"] == 20
    assert payload["corrected_support_definition"]["point_mode"] == "all48"
