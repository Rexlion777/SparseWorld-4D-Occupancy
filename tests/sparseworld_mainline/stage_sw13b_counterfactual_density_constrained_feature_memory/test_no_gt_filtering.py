import csv, json
from pathlib import Path
CSV_PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_delta_filter_metrics.csv')
JSON_PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_non_oracle_candidate_selection.json')

def test_no_gt_filtering() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    assert rows
    for row in rows:
        assert row['selection_uses_gt'] == 'False'
    obj = json.loads(JSON_PATH.read_text())
    assert obj['selection_uses_gt'] is False
