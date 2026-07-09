import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/model_level_repair_proposal_ranking.csv"


def test_repair_proposal_ranking_schema():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    required = {
        "proposal_id",
        "ranking_score",
        "expected_benefit",
        "false_positive_risk",
        "implementation_cost",
        "evidence_strength",
        "relation_to_sensor_findings",
        "relation_to_sw4x_bottleneck",
        "resume_interview_value",
    }
    assert required.issubset(row.keys())
