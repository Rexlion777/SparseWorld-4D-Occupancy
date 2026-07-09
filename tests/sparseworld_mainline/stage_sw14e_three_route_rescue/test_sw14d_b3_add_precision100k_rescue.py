from __future__ import annotations

from conftest import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, load_csv, load_json


def test_sw14d_b3_precision100k_outputs() -> None:
    required = [
        "sw14d_b3_add_precision100k_training_log.csv",
        "sw14d_b3_add_precision100k_topk.csv",
        "sw14d_b3_add_precision100k_rule_feasibility.csv",
        "sw14d_b3_add_precision100k_feasibility.json",
        "sw14d_b3_add_precision100k_final_decision.json",
        "stage_sw14d_b3_add_precision100k_rescue_report.md",
    ]
    for name in required:
        path = REPORTS_DIR / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    ckpt = ARTIFACTS_DIR / "checkpoints/sw14d_b3_add_precision100k_rescue.pth"
    scores = ARTIFACTS_DIR / "sw14d_b3_add_precision100k_scores.pt"
    fig = FIGURES_DIR / "sw14d_b3_add_precision100k_rescue.png"
    assert ckpt.exists() and ckpt.stat().st_size > 0
    assert scores.exists() and scores.stat().st_size > 0
    assert fig.exists() and fig.stat().st_size > 0


def test_sw14d_b3_precision100k_decision_schema() -> None:
    final = load_json("sw14d_b3_add_precision100k_final_decision.json")
    assert final["decision"] in {
        "ADD100K_R1_TARGET_REACHED",
        "ADD100K_R2_IMPROVED_BUT_BELOW_TARGET",
        "ADD100K_R3_NOT_REACHABLE_CURRENT_FEATURES",
    }
    assert final["target_precision"] == 0.9
    assert final["target_topk"] == 100000
    assert isinstance(final["target_reached"], bool)
    assert final["uses_eval_debug"] is False
    assert final["uses_core100_or_core500"] is False
    assert final["whether_resume_claim_allowed"] is False
    assert final["sw13_remains_main_result"] is True


def test_sw14d_b3_precision100k_honest_target_claim() -> None:
    final = load_json("sw14d_b3_add_precision100k_final_decision.json")
    best = final["best_val_precision_at_100k"]
    assert best["topk"] == 100000
    assert best["precision"] >= final["b2_val_precision_at_100k"]["precision"]
    if final["target_reached"]:
        assert final["decision"] == "ADD100K_R1_TARGET_REACHED"
        assert best["precision"] >= 0.9
    else:
        assert final["decision"] != "ADD100K_R1_TARGET_REACHED"
        assert best["precision"] < 0.9


def test_sw14d_b3_topk_schema() -> None:
    rows = load_csv("sw14d_b3_add_precision100k_topk.csv")
    assert rows
    required = {"split", "score_name", "topk", "precision", "positive_count", "front_precision", "future_precision"}
    assert required.issubset(rows[0])
    val_100k = [r for r in rows if r["split"] == "val" and r["topk"] == "100000"]
    assert {r["score_name"] for r in val_100k} >= {"b1_model", "b2_recall", "b3_hard_negative"}
