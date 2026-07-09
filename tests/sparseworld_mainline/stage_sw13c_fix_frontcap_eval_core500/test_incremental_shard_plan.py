import json
from pathlib import Path
def test_incremental_shard_plan():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_shard_plan.json').read_text())
    assert obj['existing_shard']['name'] == 'shard_000_099'
    assert obj['existing_shard']['samples'] == '0..99'
    assert obj['new_shard']['name'] == 'shard_100_500'
    assert obj['new_shard']['samples'] == '100..500'
