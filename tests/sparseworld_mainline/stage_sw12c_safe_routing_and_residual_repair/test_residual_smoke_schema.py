import csv
from pathlib import Path

PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_20iter_smoke_metrics.csv")


def test_residual_smoke_schema() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    required = {"config_name", "iter", "total_loss", "L_residual_occ", "L_residual_sem", "L_negative_empty", "L_density_budget", "pred_density_proxy_on_sampled", "positive_occ_prob_mean", "negative_occ_prob_mean", "new_occ_budget_proxy"}
    if "skipped_reason" not in rows[0]:
        assert required.issubset(rows[0].keys())
