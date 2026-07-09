import json
import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("CV_LIDAR_PROJECT_ROOT")
    if env:
        path = Path(env)
        if path.exists():
            return path
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/sw91_h2_contributor_assignment_decision.json"


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "D1_h2_not_trainable",
        "D2_h2_trainable_no_proxy_change",
        "D3_h2_proxy_improves_no_metric_gain",
        "D4_h2_safe_targeted_improvement",
        "D5_h2_tradeoff",
        "D6_h2_h3_best_promising",
        "D7_move_to_SW10_getocc_routing",
    }
    for key in ["summary", "gradient_trainable", "proxy_improved", "next_unique_action"]:
        assert key in payload, f"missing decision key: {key}"
