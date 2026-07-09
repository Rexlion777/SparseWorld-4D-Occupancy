from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization/stage_swvis1_paper_style_visualization_report.md"


def test_no_fake_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "official benchmark" in text
    assert "not an official benchmark" in text
    assert "manually edited" in text
