import csv
from pathlib import Path
PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_memory_replay_metrics.csv")

def test_metric_schema() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    required = {
        "model_name", "sample_index", "perturbation_id", "horizon_s", "variant_label",
        "front_sector_false_free_rate", "small_object_false_free_rate", "dynamic_object_false_free_rate",
        "future_h4_h6_false_free_rate", "new_visible_recall", "A10_front_h6_recovery_rate",
        "false_positive_rate", "pred_gt_occupied_ratio", "density_drift", "wrong_class_rate",
        "repaired_camera_count", "memory_age_frame", "feature_distance_current_memory", "repair_triggered_ratio", "feature_norm_drift"
    }
    assert required.issubset(rows[0].keys())
