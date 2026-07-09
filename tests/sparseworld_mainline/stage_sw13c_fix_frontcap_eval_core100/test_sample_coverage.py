import json
from pathlib import Path
def test_sample_coverage():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/sw13c_fix_frontcap_eval_core100_merge_manifest.json').read_text())
    assert obj['sample_coverage']['total_unique_samples'] == 100
    assert obj['sample_coverage']['missing_samples'] == []
    assert obj['sample_coverage']['duplicate_rows'] == 0
