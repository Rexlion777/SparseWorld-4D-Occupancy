from __future__ import annotations

from conftest import load_csv, load_json


def test_evidence_token_schema() -> None:
    config = load_json("sw14h_evidence_token_config.json")
    assert config["decision"] in {
        "H_TOK_1_READY",
        "H_TOK_2_TOO_MANY_ZERO_EVIDENCE",
        "H_TOK_3_MEMORY_TOO_HIGH",
        "H_TOK_4_TOKEN_SCHEMA_BUG",
    }
    assert config["token_schema"]["max_tokens"] == 8
    assert config["token_schema"]["token_dim"] == 12
    assert config["token_schema"]["type_count"] == 8
    rows = load_csv("sw14h_evidence_token_stats_val.csv")
    assert rows
    assert float(rows[0]["avg_tokens_per_candidate"]) > 0
