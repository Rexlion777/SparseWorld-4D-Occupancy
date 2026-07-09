import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_stagewise_diff_schema() -> None:
    rows = list(csv.DictReader((BASE / "sw14c_stagewise_diff_val.csv").open(encoding="utf-8")))
    assert rows
    required = {
        "sample_index",
        "horizon_s",
        "feature_diff_base_vs_student",
        "raw_occ_diff_count",
        "raw_occ_jaccard",
        "f3_occ_diff_count",
        "f3_pruned_set_diff",
        "final_occ_diff_count",
        "final_front_fn_diff",
        "final_fp_diff",
    }
    assert required.issubset(rows[0].keys())
    summary = json.loads((BASE / "sw14c_stagewise_diff_summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] in {
        "STAGE_R1_NO_FEATURE_EFFECT",
        "STAGE_R2_RAW_UNCHANGED",
        "STAGE_R3_RAW_CHANGED_BUT_F3_SWALLOWED",
        "STAGE_R4_F3_CHANGED_BUT_FRONTCAP_SWALLOWED",
        "STAGE_R5_FINAL_CHANGED_BUT_METRIC_NEUTRAL",
        "STAGE_R6_FINAL_CHANGED_UNSAFE",
        "STAGE_R7_FINAL_CHANGED_PROMISING",
    }
