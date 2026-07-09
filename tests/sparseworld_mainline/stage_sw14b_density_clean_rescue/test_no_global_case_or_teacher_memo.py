from pathlib import Path
src = Path('/home/rexlion/ComputerVision/cv_lidar_transition/scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/run_sw14b_density_clean_rescue.py').read_text()
def test_no_global_case_or_teacher_memo():
    assert ('_CASE' + '_MEMO') not in src
    assert ('_TEACHER' + '_BUNDLE' + '_MEMO') not in src
    assert 'ALLOW_RUNTIME_TEACHER_REBUILD' in src
    assert 'SW14B_RESCUE_ALLOW_RUNTIME_TEACHER_REBUILD' in src
