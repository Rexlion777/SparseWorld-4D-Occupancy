from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing/sw12a_variant_sweep_metrics.csv"


def test_variant_sweep_metrics_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected non-empty variant sweep metrics"
    required = {
        "checkpoint_name",
        "variant_label",
        "variant_name",
        "scenario_name",
        "horizon_s",
        "A10_front_h6_false_free_recovery_ratio",
        "variant_final_contributor_ratio",
        "false_positive_delta",
        "pred_gt_density_proxy_delta",
        "neighbor_leakage_ratio",
        "is_oracle",
    }
    assert required <= set(rows[0].keys())
