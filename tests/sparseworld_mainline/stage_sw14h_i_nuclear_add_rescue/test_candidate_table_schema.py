from __future__ import annotations

from pathlib import Path

from conftest import load_json


def test_candidate_table_schema() -> None:
    state = load_json("sw14hi_inherited_state.json")
    assert state["decision"] == "HI_INIT_READY"
    for key in ["candidate_table_train", "candidate_table_val"]:
        path = Path(state[key])
        assert path.exists()
        assert path.stat().st_size > 0
    assert state["candidate_table_train_sha256"]
    assert state["candidate_table_val_sha256"]
