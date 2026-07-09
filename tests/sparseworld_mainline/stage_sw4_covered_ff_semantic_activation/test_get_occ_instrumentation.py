from __future__ import annotations

import json
from pathlib import Path


REPORTS = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation")


def test_get_occ_instrumentation_outputs_exist() -> None:
    manifest_path = REPORTS / "get_occ_instrumentation_manifest.json"
    eq_path = REPORTS / "get_occ_output_equivalence_check.json"
    assert manifest_path.exists()
    assert eq_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eq = json.loads(eq_path.read_text(encoding="utf-8"))
    assert manifest["head_class"] == "OPUSHead"
    assert eq["all_exact_equal"] is True
