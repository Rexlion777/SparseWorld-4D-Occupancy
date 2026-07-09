import csv
from pathlib import Path
PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/sw13a_feature_memory_cache_manifest.csv")

def test_memory_cache_no_future() -> None:
    rows = list(csv.DictReader(PATH.open()))
    assert rows
    for row in rows:
        assert int(row["source_time_offset"]) > 0
        assert row["future_info_used"] == "False"
