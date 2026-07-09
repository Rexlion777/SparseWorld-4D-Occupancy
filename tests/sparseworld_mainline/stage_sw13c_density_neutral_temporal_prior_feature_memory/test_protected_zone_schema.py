import csv
from pathlib import Path
P=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_protected_zone_metrics.csv')
def test_protected_zone_schema():
    rows=list(csv.DictReader(P.open())); assert rows
    assert {'protected_voxel_count','protected_confidence_mean','protected_temporal_agreement_mean','uses_gt'}.issubset(rows[0])
    assert all(r['uses_gt']=='False' for r in rows)
