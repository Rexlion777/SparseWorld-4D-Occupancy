import csv
import numpy as np
from pathlib import Path
MANIFEST = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory/sw13b_paired_replay_dump_manifest.csv')

def test_paired_dump_schema() -> None:
    rows = list(csv.DictReader(MANIFEST.open()))
    assert rows
    sample = np.load(rows[0]['paired_dump_path'])
    for key in ['native_occ_mask','repair_occ_mask','delta_occ_mask','native_semantic','repair_semantic']:
        assert key in sample, key
