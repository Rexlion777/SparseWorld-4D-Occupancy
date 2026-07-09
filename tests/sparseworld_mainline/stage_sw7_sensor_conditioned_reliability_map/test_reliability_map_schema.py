from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/reliability_maps"


def test_reliability_map_schema() -> None:
    npz_files = sorted(ARTIFACTS.rglob("h*_reliability.npz"))
    assert npz_files, "no reliability npz files found"
    data = np.load(npz_files[0], allow_pickle=True)
    required = {
        "semantic_occ",
        "reliability_map",
        "risk_map",
        "semantic_reliability",
        "support_reliability",
        "contributor_reliability",
        "sensor_condition_prior",
        "temporal_reliability",
        "risk_type",
        "metadata",
    }
    assert required.issubset(set(data.files))
    meta = json.loads(data["metadata"][0])
    assert {"perturbation_id", "sample_index", "horizon_s", "component_names", "risk_type_names"}.issubset(meta)
