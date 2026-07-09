import json
from pathlib import Path
def test_decision_schema():
    d=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_density_neutral_temporal_prior_decision.json').read_text()); assert d['decision_type'] in {'U1_STRONG_DENSITY_NEUTRAL_RECOVERY','U2_MEDIUM_DENSITY_NEUTRAL_RECOVERY','U3_RECOVERY_RETAINED_BUT_DENSITY_STILL_HIGH','U4_DENSITY_CONTROLLED_BUT_RECOVERY_LOST','U5_PROTECTED_ZONE_DESIGN_FAILED','U6_C4_ONLY_SAFE','U7_IMPLEMENTATION_BLOCKED','U8_ORACLE_RISK'}
