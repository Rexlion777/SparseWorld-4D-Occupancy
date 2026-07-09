import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/shard_050_099/sw13c_fix_frontcap_replay_manifest_050_099.csv').open()))
def test_no_rerun_existing_samples():
    assert rows
    assert all(int(r['sample_index']) >= 50 for r in rows)
    assert all(int(r['sample_index']) <= 99 for r in rows)
