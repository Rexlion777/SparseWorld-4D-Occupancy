import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune/sw8_finetune_decision.json"


def test_decision_schema():
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "T1_success_safe_improvement",
        "T2_partial_improvement_with_tradeoff",
        "T3_no_meaningful_improvement",
        "T4_architecture_limited",
        "T5_training_pipeline_unstable",
    }
    for key in ["safe_claim", "next_action"]:
        assert key in payload, f"missing key: {key}"
