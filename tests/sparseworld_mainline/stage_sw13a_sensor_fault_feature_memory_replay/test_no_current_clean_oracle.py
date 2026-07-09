import csv
from pathlib import Path
PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_replay_manifest.csv")

def test_no_current_clean_oracle() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    for row in rows:
        assert row["future_info_used"] == "False"
        assert row["current_clean_same_frame_used"] == "False"
        assert row["gt_used_for_repair"] == "False"
