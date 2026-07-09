import json
import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("CV_LIDAR_PROJECT_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/sw81_eval_protocol.json"


def test_eval_protocol_schema():
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    for key in ["quick_eval_10", "eval_core_20", "eval_stress_20"]:
        assert key in payload, f"missing key: {key}"
        assert isinstance(payload[key], list) and payload[key], f"invalid subset: {key}"
