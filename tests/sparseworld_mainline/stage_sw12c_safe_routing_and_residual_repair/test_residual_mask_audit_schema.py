import csv
from pathlib import Path

PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_mask_audit.csv")


def test_residual_mask_audit_schema() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    required = {
        "config_name", "split", "sample_index", "perturbation_id", "horizon_s",
        "teacher_occupied_count_raw", "teacher_reliable_occupied_count_raw", "student_native_occupied_count_raw",
        "residual_positive_count_raw", "high_risk_count_raw", "positive_repair_count_raw",
        "selected_positive_count", "selected_negative_count", "ignored_count",
        "teacher_reliable_free_count_raw", "negative_guard_count_raw",
        "positive_negative_overlap_count", "positive_already_student_occupied_count", "positive_not_high_risk_count"
    }
    assert required.issubset(rows[0].keys())
