from __future__ import annotations

from conftest import load_csv, load_json


def test_sw14f_probe_schema() -> None:
    rows = load_csv("sw14f_raw_logit_causal_probe.csv")
    assert rows
    required = {
        "split",
        "region",
        "count",
        "raw_occ_rate",
        "final_occ_rate",
        "raw_to_final_pruned_proxy_rate",
        "survival_through_F3_probability_proxy",
        "survival_through_FrontCap_probability_proxy",
    }
    assert required.issubset(rows[0])
    final = load_json("sw14f_final_decision.json")
    assert final["decision"] in {
        "SW14F_0_STOP_NO_RAW_SIGNAL",
        "SW14F_1_RAW_SIGNAL_BUT_TRAINING_FAIL",
        "SW14F_2_SURVIVAL_IMPROVES_BUT_FINAL_UNSAFE",
        "SW14F_3_SURVIVAL_IMPROVES_SAFE_READY_COMPARE",
        "SW14F_4_FEATURE_ROUTE_PROTOCOL_BUG",
    }
    assert final["uses_eval_debug"] is False
