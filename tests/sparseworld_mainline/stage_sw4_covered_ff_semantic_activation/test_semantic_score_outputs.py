from __future__ import annotations

import csv
from pathlib import Path


REPORTS = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation")


def test_semantic_score_outputs_exist() -> None:
    required = [
        "support_level_semantic_scores.csv",
        "get_occ_gate_filter_analysis.csv",
        "voxel_aggregation_logit_decomposition.csv",
        "small_object_semantic_activation_diagnosis.csv",
        "new_visible_semantic_activation_diagnosis.csv",
    ]
    for name in required:
        path = REPORTS / name
        assert path.exists(), name
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert len(rows) > 0, name
