import json
from pathlib import Path


PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter/sw14c_metric_sign_dictionary.json")


def test_metric_sign_dictionary() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "SW13-style deltas: system - native"
    assert payload["front_sector_false_free_rate_delta_vs_native"]["better_direction"] == "more_negative"
    assert payload["future_h4_h6_false_free_rate_delta_vs_native"]["better_direction"] == "more_negative"
    assert payload["false_positive_delta"]["better_direction"] == "lower"
    assert payload["protected_zone_preservation_ratio"]["better_direction"] == "higher"
