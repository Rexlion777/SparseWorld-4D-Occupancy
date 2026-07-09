import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/sw6_refined_failure_taxonomy.csv"
JSON_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/sw6_refined_failure_taxonomy.json"


def test_failure_taxonomy_schema():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    required = {
        "perturbation_id",
        "primary_type",
        "secondary_type",
        "evidence_probe_best",
        "confidence",
        "safe_wording",
    }
    assert required.issubset(row.keys())
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert isinstance(payload.get("rows"), list)
    assert payload["rows"]
