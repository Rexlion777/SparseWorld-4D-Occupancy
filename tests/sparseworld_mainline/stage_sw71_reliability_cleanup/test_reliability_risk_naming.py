import json
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    candidates = [Path("D:/ComputerVision/cv_lidar_transition"), Path("/mnt/d/ComputerVision/cv_lidar_transition")]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


ROOT = repo_root()
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw71_reliability_cleanup"


def test_reliability_risk_naming():
    payload = json.loads((REPORTS / "stage_sw71_reliability_cleanup_report.json").read_text(encoding="utf-8"))
    assert payload["reliability_definition"] == "internal diagnostic reliability indicator"
    assert payload["risk_definition"] == "risk = 1 - reliability"

    df = pd.read_csv(REPORTS / "component_breakdown_values.csv")
    assert {"component_reliability", "component_risk"}.issubset(df.columns)
    assert (df["component_reliability"].between(0, 1)).all()
    assert (df["component_risk"].between(0, 1)).all()
