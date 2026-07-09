import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune/training_entry_audit.json"


def test_training_audit_schema():
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    for key in ["smoke_checkpoint_path", "iter_count", "loss_finite", "checkpoint_reload_eval_ok"]:
        assert key in payload, f"missing key: {key}"
