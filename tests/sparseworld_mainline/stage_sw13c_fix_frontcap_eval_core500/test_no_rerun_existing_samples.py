import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/shard_100_500/sw13c_fix_frontcap_replay_manifest_100_500.csv').open()))
def test_no_rerun_existing_samples():
    assert rows
    assert all(int(r['sample_index']) >= 100 for r in rows)
    assert all(int(r['sample_index']) <= 500 for r in rows)
