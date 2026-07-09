from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/branchB_fixed_subset_eval.csv"


def test_branchB_eval_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    required = {
        "model_name",
        "sample_index",
        "perturbation_id",
        "horizon_s",
        "occupied_iou",
        "semantic_miou",
        "false_free_rate",
        "false_occupied_rate",
        "pred_gt_occupied_ratio",
        "front_sector_false_free",
        "active_voxel_count_delta",
        "wrong_class_activation_delta",
    }
    assert required <= set(rows[0].keys())
