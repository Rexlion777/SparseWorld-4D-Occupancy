import csv
import numpy as np
from pathlib import Path
MANIFEST = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_paired_replay_dump_manifest.csv')

def test_delta_definition() -> None:
    rows = list(csv.DictReader(MANIFEST.open()))
    sample = np.load(rows[0]['paired_dump_path'])
    expected = np.logical_and(sample['repair_occ_mask'] > 0, np.logical_not(sample['native_occ_mask'] > 0))
    assert np.array_equal(sample['delta_occ_mask'] > 0, expected)
