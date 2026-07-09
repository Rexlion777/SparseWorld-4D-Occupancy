from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14e_three_route_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14e_three_route_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14e_three_route_rescue"


def load_json(name: str):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def load_csv(name: str):
    with (REPORTS_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_table(split: str):
    path = ARTIFACTS_DIR / "candidate_tables" / f"sw14d_b_candidate_table_{split}.parquet"
    return torch.load(path, map_location="cpu", weights_only=False)
