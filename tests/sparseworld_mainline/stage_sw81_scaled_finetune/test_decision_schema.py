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
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/sw81_scaled_finetune_decision.json"


def test_decision_schema():
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "S1_safe_targeted_improvement",
        "S2_partial_improvement_tradeoff",
        "S3_no_clear_gain_after_scaled_training",
        "S4_architecture_limited_evidence_strengthened",
        "S5_training_unstable",
    }
    for key in ["safe_claim", "next_action", "false_positive_headline"]:
        assert key in payload, f"missing key: {key}"
