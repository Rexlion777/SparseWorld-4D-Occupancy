import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_rear_residual_zero() -> None:
    sanity = json.loads((BASE / "sw14c_checkpoint_sanity_audit.json").read_text(encoding="utf-8"))
    assert sanity["rear_cameras_residual_max_abs"] == 0.0
    rows = list(csv.DictReader((BASE / "sw14c_residual_magnitude_val.csv").open(encoding="utf-8")))
    rear_rows = [row for row in rows if row["camera_name"].startswith("CAM_BACK")]
    assert rear_rows
    assert max(float(row["mean_abs"]) for row in rear_rows) == 0.0
