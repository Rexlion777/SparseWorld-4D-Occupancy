import json
from pathlib import Path

BASE = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair")


def test_baseline_k7_replay():
    payload = json.loads((BASE / "baseline_k7_replay_equivalence.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert "k7_mean_false_free_rate_delta" in payload
