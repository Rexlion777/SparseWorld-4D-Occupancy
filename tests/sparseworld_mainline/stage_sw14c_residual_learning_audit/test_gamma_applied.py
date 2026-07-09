import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_gamma_applied() -> None:
    sanity = json.loads((BASE / "sw14c_checkpoint_sanity_audit.json").read_text(encoding="utf-8"))
    decision = json.loads((BASE / "sw14c_residual_learning_audit_decision.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((BASE / "sw14c_gamma_sensitivity_audit.csv").open(encoding="utf-8")))
    gammas = {float(row["gamma"]) for row in rows}
    assert sanity["gamma_selection_is_0p1"] is True
    assert sanity["eval_gamma_is_0p1"] is True
    assert sanity["eval_gamma_applied"] == 0.1
    assert 0.0 in gammas and 0.1 in gammas and 1.0 in gammas
    assert decision["evidence"]["gamma_decision"]
