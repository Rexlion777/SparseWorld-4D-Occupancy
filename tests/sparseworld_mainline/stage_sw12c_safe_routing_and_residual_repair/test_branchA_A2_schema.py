import csv
from pathlib import Path

PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchA_A2_100iter_eval.csv")


def test_branchA_schema() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    required = {"phase", "checkpoint_name", "variant_label", "perturbation_id", "horizon_s", "target_recovery", "A10_front_h6_recovery_ratio", "clean_false_positive_delta", "pred_gt_density_delta"}
    assert required.issubset(rows[0].keys())
