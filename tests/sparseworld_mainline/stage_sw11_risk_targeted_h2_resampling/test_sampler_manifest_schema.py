from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling/sw11_sampler_config_manifest.json"


def test_sampler_manifest_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["seed"] == 17
    assert "modes" in payload and isinstance(payload["modes"], dict)
    for key in ["S0_original_h2_sampling", "S1_risk_targeted_h2_sampling", "S2_native_missing_h2_sampling", "S3_survival_aware_h2_sampling"]:
        assert key in payload["modes"], f"missing sampler mode {key}"
