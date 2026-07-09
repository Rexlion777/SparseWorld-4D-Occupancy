import csv
from pathlib import Path

PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_mask_audit.csv")
LIMITS = {
    "R1_residual_conf09_K128_neg2": (128, 2),
    "R2_residual_conf09_K256_neg2": (256, 2),
    "R3_residual_conf095_K128_neg2": (128, 2),
    "R4_residual_conf095_K256_neg4": (256, 4),
    "R5_residual_front_h46_only_conf095_K128_neg4": (128, 4),
}


def test_strict_masks() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    for row in rows:
        k, ratio = LIMITS[row["config_name"]]
        pos = int(row["selected_positive_count"])
        neg = int(row["selected_negative_count"])
        assert int(row["positive_already_student_occupied_count"]) == 0
        assert int(row["positive_negative_overlap_count"]) == 0
        assert pos <= k
        assert neg <= ratio * max(1, pos) if pos > 0 else neg == 0
        if int(row["horizon_s"]) in {4, 6}:
            assert int(row["positive_not_high_risk_count"]) == 0
