import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_teacher_error_masks_schema():
    summary = json.loads((BASE / "sw14c_round2b_region_mask_summary.json").read_text())
    assert summary["decision"] in {"MASK_R1_READY", "MASK_R2_TEACHER_ERROR_TOO_SPARSE", "MASK_R3_FP_DOMINANT_READY", "MASK_R4_MASK_BUG"}
    rows = list(csv.DictReader((BASE / "sw14c_round2b_region_mask_stats_val.csv").open()))
    assert rows
    required = {"teacher_FN_count", "teacher_FP_count", "M_recover_count", "M_suppress_count", "M_preserve_count", "M_residual_allowed_count"}
    assert required.issubset(rows[0])
