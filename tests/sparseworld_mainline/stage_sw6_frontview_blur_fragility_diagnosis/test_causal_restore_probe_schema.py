import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/causal_restore_probe_metrics.csv"
JSON_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/causal_restore_probe_metrics.json"


def test_causal_restore_probe_schema():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    required = {
        "probe_name",
        "probe_mode",
        "perturbation_id",
        "sample_index",
        "horizon_s",
        "occupied_iou",
        "semantic_miou",
        "false_free_rate",
        "false_occupied_rate",
        "pred_gt_occupied_ratio",
        "dynamic_false_free",
        "small_object_false_free",
        "new_visible_recall",
        "gate_pass_ratio",
        "contributor_ratio",
    }
    assert required.issubset(row.keys())
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert {"rows", "aggregate", "unavailable"}.issubset(payload.keys())
    assert "P4_clean_gate_restore" in payload["unavailable"]
