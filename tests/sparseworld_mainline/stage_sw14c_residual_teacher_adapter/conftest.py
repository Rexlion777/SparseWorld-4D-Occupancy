from pathlib import Path
import sys

ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
SCRIPT_DIR = ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
