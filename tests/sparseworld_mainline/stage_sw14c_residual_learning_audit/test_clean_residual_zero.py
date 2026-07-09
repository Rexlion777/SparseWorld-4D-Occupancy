import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_clean_residual_zero() -> None:
    sanity = json.loads((BASE / "sw14c_checkpoint_sanity_audit.json").read_text(encoding="utf-8"))
    assert sanity["clean_degradation_mask_sum"] == 0.0
    assert sanity["clean_residual_max_abs"] == 0.0
    rows = list(csv.DictReader((BASE / "sw14c_residual_magnitude_train.csv").open(encoding="utf-8")))
    assert rows
    assert max(float(row["clean_mean_abs"]) for row in rows) == 0.0
