from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    candidates = [Path("D:/ComputerVision/cv_lidar_transition"), Path("/mnt/d/ComputerVision/cv_lidar_transition")]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


ROOT = repo_root()
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw71_reliability_cleanup"


def test_topk_validation_schema():
    df = pd.read_csv(REPORTS / "high_risk_topk_validation.csv")
    required_cols = {
        "perturbation_id",
        "scope_type",
        "scope_name",
        "topk_ratio",
        "error_type",
        "precision",
        "recall",
        "selected_voxels",
        "error_voxels_in_scope",
        "hit_voxels",
    }
    assert required_cols.issubset(df.columns)
    assert set(["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]).issubset(set(df["perturbation_id"]))
    assert set([0.01, 0.05, 0.1]).issubset(set(round(float(x), 2) for x in df["topk_ratio"].unique()))
    assert set(["false_free", "false_positive"]).issubset(set(df["error_type"]))
