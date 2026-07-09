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
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/sw9_contributor_supervision_decision.json"


def test_decision_schema():
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "C1_hook_not_trainable",
        "C2_hook_trainable_no_signal",
        "C3_safe_targeted_improvement",
        "C4_tradeoff_improvement",
        "C5_promising_but_undertrained",
        "C6_architecture_modification_needed",
    }
    for key in ["gradient_trainable", "next_action", "false_positive_headline"]:
        assert key in payload, f"missing key: {key}"

