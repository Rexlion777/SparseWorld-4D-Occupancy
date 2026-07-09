import json
from pathlib import Path
def test_incremental_shard_plan():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/sw13c_fix_frontcap_eval_core100_shard_plan.json').read_text())
    assert obj['existing_shard']['name'] == 'shard_000_049'
    assert obj['existing_shard']['samples'] == '0..49'
    assert obj['new_shard']['name'] == 'shard_050_099'
    assert obj['new_shard']['samples'] == '50..99'
