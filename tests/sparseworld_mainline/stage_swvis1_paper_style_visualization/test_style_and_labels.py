from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"


def test_style_config_and_horizon_labels() -> None:
    cfg = json.loads((REPORTS / "paper_style_visualization_config.json").read_text(encoding="utf-8"))
    assert cfg["horizon_label_mapping"]["h6"] == "3s"
    assert "palette" in cfg and len(cfg["palette"]) >= 10


def test_quality_check_pass() -> None:
    qc = json.loads((REPORTS / "visualization_quality_check.json").read_text(encoding="utf-8"))
    assert qc["overall_pass"] is True

