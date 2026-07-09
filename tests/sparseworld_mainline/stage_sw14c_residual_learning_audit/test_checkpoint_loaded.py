import json
from pathlib import Path


PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit/sw14c_checkpoint_sanity_audit.json")


def test_checkpoint_loaded() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["checkpoint_exists"] is True
    assert payload["init_checkpoint_exists"] is True
    assert payload["checkpoint_load_success"] is True
    assert payload["adapter_trainable_params_count"] == 199836
    assert payload["decision"] in {
        "CKPT_A1_TRAINED_PARAMS_CHANGED",
        "CKPT_A2_CHECKPOINT_EQUALS_INIT",
        "CKPT_A3_GAMMA_NOT_APPLIED",
        "CKPT_A4_MASK_BUG",
        "CKPT_A5_LOAD_FAILURE",
    }
