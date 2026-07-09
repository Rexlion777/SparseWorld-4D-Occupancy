from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"


def test_branchB_train_metrics_schema() -> None:
    with (REPORTS / "branchB_gradient_check.csv").open("r", newline="", encoding="utf-8") as handle:
        grad_rows = list(csv.DictReader(handle))
    assert grad_rows
    required = {"config_name", "total_loss", "consistency_loss", "grad_norm", "grad_finite", "pred_density_proxy"}
    assert required <= set(grad_rows[0].keys())

    smoke_path = REPORTS / "branchB_20iter_smoke_metrics.csv"
    with smoke_path.open("r", newline="", encoding="utf-8") as handle:
        smoke_rows = list(csv.DictReader(handle))
    assert smoke_rows
    smoke_required = {"config_name", "iter", "total_loss", "fp_guard_loss", "density_loss", "pred_density_proxy"}
    assert smoke_required <= set(smoke_rows[0].keys())
