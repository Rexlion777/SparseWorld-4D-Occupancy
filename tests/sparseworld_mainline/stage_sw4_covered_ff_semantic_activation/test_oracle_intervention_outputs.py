from __future__ import annotations

import csv
from pathlib import Path


REPORTS = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation")


def test_oracle_intervention_outputs_exist() -> None:
    path = REPORTS / "semantic_activation_oracle_interventions.csv"
    assert path.exists()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) > 0
    names = {row["intervention_name"] for row in rows}
    expected = {
        "A_gt_class_boost_covered_ff",
        "B_foreground_boost_covered_ff",
        "C_gate_relax_covered_ff",
        "D_duplicate_support_covered_ff",
        "E_small_object_class_oracle",
        "F_new_visible_future_semantics_boost",
    }
    assert expected.issubset(names)
