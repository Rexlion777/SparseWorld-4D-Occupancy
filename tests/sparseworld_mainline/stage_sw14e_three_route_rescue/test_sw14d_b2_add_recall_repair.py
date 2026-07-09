from __future__ import annotations

from pathlib import Path

from conftest import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, load_csv, load_json


def test_sw14d_b2_add_recall_repair_outputs() -> None:
    required = [
        "sw14d_b2_add_training_log.csv",
        "sw14d_b2_add_topk_precision.csv",
        "sw14d_b2_add_recall_repair_search.csv",
        "sw14d_b2_add_recall_repair_final_decision.json",
        "stage_sw14d_b2_add_recall_repair_report.md",
    ]
    for name in required:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    ckpt = ARTIFACTS_DIR / "checkpoints/sw14d_b2_add_recall_repair.pth"
    scores = ARTIFACTS_DIR / "sw14d_b2_add_recall_scores.pt"
    fig = FIGURES_DIR / "sw14d_b2_add_recall_repair.png"
    assert ckpt.exists() and ckpt.stat().st_size > 0
    assert scores.exists() and scores.stat().st_size > 0
    assert fig.exists() and fig.stat().st_size > 0


def test_sw14d_b2_decision_schema() -> None:
    final = load_json("sw14d_b2_add_recall_repair_final_decision.json")
    assert final["decision"] in {
        "SW14D_B2_ADD_0_NO_SAFE_RECALL_IMPROVEMENT",
        "SW14D_B2_ADD_1_FRONT_GAIN_ONLY",
        "SW14D_B2_ADD_2_RECALL_IMPROVES_NET_LOWER",
        "SW14D_B2_ADD_3_RECALL_AND_NET_IMPROVE",
    }
    assert final["uses_eval_debug"] is False
    assert final["uses_core100_or_core500"] is False
    assert final["whether_resume_claim_allowed"] is False
    assert final["sw13_remains_main_result"] is True
    assert "current_sw14d_b" in final


def test_sw14d_b2_add_precision_schema() -> None:
    rows = load_csv("sw14d_b2_add_topk_precision.csv")
    assert rows
    required = {"split", "score_name", "topk", "precision", "front_precision", "future_precision"}
    assert required.issubset(rows[0])
    b1 = [r for r in rows if r["score_name"] == "b1_model" and r["topk"] == "10000"]
    b2 = [r for r in rows if r["score_name"] == "b2_recall" and r["topk"] == "10000"]
    assert b1 and b2
    assert float(b2[0]["precision"]) >= float(b1[0]["precision"])
