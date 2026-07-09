import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_gamma_zero_not_improvement():
    selection = json.loads((BASE / "sw14c_round2b_gamma_selection.json").read_text())
    assert selection["gamma_zero_counts_as_improvement"] is False
    rows = list(csv.DictReader((BASE / "sw14c_round2b_gamma_sweep_val.csv").open()))
    zero = [row for row in rows if abs(float(row["gamma"])) < 1e-12]
    assert zero and all(row["allowed_as_improvement"] == "False" for row in zero)
