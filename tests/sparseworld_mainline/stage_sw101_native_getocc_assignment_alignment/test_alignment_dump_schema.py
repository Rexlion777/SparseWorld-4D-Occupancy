from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"


def test_alignment_dump_schema() -> None:
    schema = json.loads((REPORTS / "alignment_dump_schema.json").read_text(encoding="utf-8"))
    with (REPORTS / "alignment_dump_manifest.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty alignment dump manifest"
    row = rows[0]
    for key in [
        "checkpoint_name",
        "perturbation_id",
        "sample_index",
        "horizon_s",
        "dump_path",
        "same_decoded_support_between_h2_and_native",
    ]:
        assert key in row and row[key] != "", f"missing {key}"
    dump_path = Path(row["dump_path"])
    assert dump_path.exists(), f"missing dump {dump_path}"
    with np.load(dump_path) as data:
        for key in schema["npz_keys"]:
            assert key in data.files, f"missing npz key {key}"
