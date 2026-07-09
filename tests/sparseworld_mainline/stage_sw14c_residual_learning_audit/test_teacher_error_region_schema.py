import csv
import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_teacher_error_region_schema() -> None:
    rows = list(csv.DictReader((BASE / "sw14c_teacher_error_regions_val.csv").open(encoding="utf-8")))
    assert rows
    required = {
        "split",
        "sample_index",
        "horizon_s",
        "teacher_FN_count",
        "teacher_FP_count",
        "teacher_correct_occ_count",
        "teacher_correct_free_count",
        "teacher_FN_rate",
        "teacher_FP_rate",
        "front_teacher_FN_count",
        "future_h4h6_teacher_FN_count",
        "teacher_error_sparsity_score",
    }
    assert required.issubset(rows[0].keys())
    summary = json.loads((BASE / "sw14c_teacher_error_region_summary.json").read_text(encoding="utf-8"))
    assert summary["uses_gt_eval_only"] is True
    assert summary["decision"] in {
        "ERR_R1_TEACHER_ERROR_TOO_SPARSE",
        "ERR_R2_TEACHER_FN_CONCENTRATED_FEW_SAMPLES",
        "ERR_R3_TEACHER_FP_DOMINANT",
        "ERR_R4_FUTURE_HORIZON_ERROR_DOMINANT",
        "ERR_R5_FRONT_ERROR_HAS_LEARNABLE_SIGNAL",
        "ERR_R6_ERROR_REGION_BALANCED",
    }
