import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_residual_health_schema():
    summary = json.loads((BASE / "sw14c_round2b_residual_health_summary.json").read_text())
    assert summary["decision"] in {"HEALTH_R1_NOOP_FIXED", "HEALTH_R2_STILL_NOOP", "HEALTH_R3_TOO_AGGRESSIVE", "HEALTH_R4_MASK_LEAK", "HEALTH_R5_HEALTHY_BUT_METRIC_NEUTRAL"}
    rows = list(csv.DictReader((BASE / "sw14c_round2b_residual_health_val.csv").open()))
    assert rows
    required = {"front_gate_mean", "front_gate_p95", "gamma0p1_front_feature_delta_mean_abs", "raw_head_occ_diff_vs_teacher", "final_occ_diff_after_f3_frontcap"}
    assert required.issubset(rows[0])
