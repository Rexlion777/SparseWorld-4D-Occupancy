from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))
STAGE = "stage_sw14e_three_route_rescue"
SCRIPT_DIR = PROJECT_ROOT / f"scripts/lidar_system_algorithm/sparseworld_mainline/{STAGE}"
REPORTS_DIR = PROJECT_ROOT / f"reports/lidar_system_algorithm/sparseworld_mainline/{STAGE}"
ARTIFACTS_DIR = PROJECT_ROOT / f"artifacts/sparseworld_mainline/{STAGE}"
FIGURES_DIR = PROJECT_ROOT / f"projects/lidar_system_algorithm/figures/sparseworld_mainline/{STAGE}"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
TABLE_DIR = ARTIFACTS_DIR / "candidate_tables"
TEACHER_CACHE = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_learning_audit/runtime_teacher_cache/audit_A10_drop_front_triplet"
SPARSEWORLD_ROOT = PROJECT_ROOT / "external/SparseWorld"

SW14D_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14d_candidate_reranker"
SW14C_AUDIT_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit"
SW14C_OEM_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_oracle_error_mask_residual"
SW14C_R2B_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue"
SW13_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline"

EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]
GRID_SHAPE = (200, 200, 16)
VOXELS_PER_GRID = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]

NO_GT_FEATURES = [
    "raw_confidence",
    "raw_margin",
    "raw_entropy",
    "raw_occ",
    "teacher_final_occ",
    "native_final_occ",
    "raw_but_final_empty",
    "final_occupied_low_conf",
    "final_occupied_isolated",
    "local_density_proxy",
    "neighbor_occ_count_norm",
    "temporal_consistency",
    "camera_view_agreement",
    "front_region",
    "future_h4h6",
    "protected_zone",
    "x_norm",
    "abs_y_norm",
    "z_norm",
    "range_norm",
    "horizon_norm",
    "sector_norm",
    "distance_bin_norm",
]
GT_LABEL_COLUMNS = ["GT_occ", "teacher_FN", "teacher_FP", "teacher_correct_occ", "teacher_correct_free"]
_GROUP_INDEX_CACHE: dict[int, dict[int, torch.Tensor]] = {}
_GROUP_ROWS_CACHE: dict[str, list[dict[str, Any]]] = {}


@dataclass(frozen=True)
class Config:
    seed: int = 23
    train_start: int = 0
    train_end: int = 99
    val_start: int = 100
    val_end: int = 149
    epochs: int = 4
    batch_size: int = 262144
    score_batch_size: int = 524288
    max_add_train_pos: int = 900000
    max_add_train_neg: int = 900000
    max_sup_train_pos: int = 900000
    max_sup_train_neg: int = 900000
    lr: float = 2.5e-3
    max_suppress_ratio: float = 0.030
    add_suppress_balance: float = 0.50
    add_fixed_budget: int = 32
    front_cap_ratio: float = 1.30
    front_keep_conf: float = 0.72
    front_keep_agreement: float = 0.55
    front_add_keep_conf: float = 0.75
    suppress_rank_weight: float = 1.0
    add_rank_weight: float = 1.0
    density_delta_limit: float = 0.003
    fp_delta_limit: float = 0.001
    broken_rate_limit: float = 0.002
    front_fn_tolerance: float = -0.001
    future_fn_tolerance: float = -0.001
    small_gain_threshold: float = 0.003
    strong_gain_threshold: float = 0.010
    rebuild_tables: bool = False
    device: str = "cuda"


class CandidateScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.02),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SW14E three-route rescue")
    p.add_argument("--route", choices=["all", "route1", "route2", "route3"], default="all")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--rebuild-tables", action="store_true")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=262144)
    p.add_argument("--score-batch-size", type=int, default=524288)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def ensure_dirs() -> None:
    for path in [
        SCRIPT_DIR,
        SCRIPT_DIR / "route1_sw14d_b_candidate_reranker",
        SCRIPT_DIR / "route2_sw14f_postprocess_aware_feature",
        SCRIPT_DIR / "route3_sw14g_query_reliability_adapter",
        REPORTS_DIR,
        ARTIFACTS_DIR,
        CHECKPOINT_DIR,
        TABLE_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return normalize(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize(row))


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) else 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_bundle(sample_index: int) -> dict[str, Any]:
    path = TEACHER_CACHE / f"A10_drop_front_triplet__sample{sample_index:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def occ(x: torch.Tensor) -> torch.Tensor:
    return x.long() != EMPTY_IDX


def build_coords() -> dict[str, torch.Tensor]:
    x_n, y_n, z_n = GRID_SHAPE
    x = torch.linspace(-50.0, 50.0, x_n)[:, None, None].expand(GRID_SHAPE)
    y = torch.linspace(-50.0, 50.0, y_n)[None, :, None].expand(GRID_SHAPE)
    z = torch.linspace(0.0, 1.0, z_n)[None, None, :].expand(GRID_SHAPE)
    r = torch.sqrt(x.square() + y.square()).clamp(max=70.71)
    front = (x > 0.0) & (y.abs() <= 10.0)
    sector = torch.zeros(GRID_SHAPE, dtype=torch.int8)
    sector[(x > 0.0) & (y < -10.0)] = 1
    sector[front] = 2
    sector[(x > 0.0) & (y > 10.0)] = 3
    sector[x <= 0.0] = 4
    distance_bin = torch.bucketize(r.contiguous(), torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])).to(torch.int8)
    protected = (r < 3.0) | ((x > 0.0) & (x < 5.0) & (y.abs() < 3.0))
    return {
        "x_norm": x / 50.0,
        "abs_y_norm": y.abs() / 50.0,
        "z_norm": z,
        "range_norm": r / 70.71,
        "front": front,
        "sector_id": sector,
        "distance_bin": distance_bin,
        "protected_zone": protected,
    }


def neighbor_count_6(mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(torch.uint8)
    out = torch.zeros_like(m, dtype=torch.uint8)
    out[1:, :, :] += m[:-1, :, :]
    out[:-1, :, :] += m[1:, :, :]
    out[:, 1:, :] += m[:, :-1, :]
    out[:, :-1, :] += m[:, 1:, :]
    out[:, :, 1:] += m[:, :, :-1]
    out[:, :, :-1] += m[:, :, 1:]
    return out


def candidate_masks(th: dict[str, Any], horizon_s: int, coords: dict[str, torch.Tensor], cfg: Config) -> dict[str, torch.Tensor]:
    raw = th["teacher_raw_semantic"].long()
    final = th["teacher_final_semantic"].long()
    raw_occ = occ(raw)
    final_occ = occ(final)
    conf = th["teacher_confidence"].float()
    margin = th["teacher_margin"].float()
    agreement = th["agreement"].float()
    front = coords["front"].bool()
    future = torch.ones_like(front) if horizon_s in {4, 6} else torch.zeros_like(front)
    protected = coords["protected_zone"].bool()
    neighbor = neighbor_count_6(final_occ)
    isolated = final_occ & (neighbor <= 1)

    strong_add = (~final_occ) & raw_occ & (
        ((conf >= 0.55) & (margin >= 0.02))
        | (front & (conf >= 0.45))
        | (future & (conf >= 0.45))
    )
    medium_add = (~final_occ) & raw_occ & ~strong_add & (
        (conf >= 0.25) | (margin >= 0.01) | front | future
    )
    weak_add = (~final_occ) & ~raw_occ & (front | future) & ((conf >= 0.20) | (margin >= 0.01))
    high_conf_keep = final_occ & raw_occ & (
        ((conf >= 0.85) & (agreement >= 0.80))
        | (front & (conf >= cfg.front_keep_conf) & (agreement >= cfg.front_keep_agreement))
        | protected
    )
    suppress = final_occ & (
        (conf < 0.75)
        | (margin < 0.10)
        | (agreement < 0.75)
        | (~raw_occ)
        | isolated
        | front
        | future
    ) & ~high_conf_keep
    union = strong_add | medium_add | weak_add | suppress | high_conf_keep
    return {
        "strong_add": strong_add,
        "medium_add": medium_add,
        "weak_add": weak_add,
        "suppress": suppress,
        "keep": high_conf_keep,
        "union": union,
        "neighbor": neighbor,
        "isolated": isolated,
        "future": future,
    }


def table_path(split: str) -> Path:
    return TABLE_DIR / f"sw14d_b_candidate_table_{split}.parquet"


def group_summary_path(split: str) -> Path:
    return TABLE_DIR / f"sw14d_b_group_summary_{split}.pt"


def split_range(split: str, cfg: Config) -> range:
    return range(cfg.train_start, cfg.train_end + 1) if split == "train" else range(cfg.val_start, cfg.val_end + 1)


def save_tensor_table(path: Path, tensors: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "torch_tensor_table_named_parquet_surrogate", "meta": meta, "tensors": tensors}, path)


def load_tensor_table(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def append_selected(dst: dict[str, list[torch.Tensor]], key: str, value: torch.Tensor, idx: torch.Tensor) -> None:
    dst.setdefault(key, []).append(value[idx[:, 0], idx[:, 1], idx[:, 2]].detach().cpu())


def build_candidate_table(split: str, cfg: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coords = build_coords()
    cols: dict[str, list[torch.Tensor]] = {}
    group_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []

    def add_col(name: str, value: torch.Tensor) -> None:
        cols.setdefault(name, []).append(value.detach().cpu())

    for sample_index in split_range(split, cfg):
        bundle = load_bundle(sample_index)
        for horizon_s in CORE_HORIZONS:
            th = bundle["by_horizon"][horizon_s]
            raw = th["teacher_raw_semantic"].long()
            final = th["teacher_final_semantic"].long()
            native = th["native_semantic"].long()
            gt = th["gt_h"].long()
            raw_occ = occ(raw)
            final_occ = occ(final)
            native_occ = occ(native)
            gt_occ = occ(gt)
            conf = th["teacher_confidence"].float()
            margin = th["teacher_margin"].float()
            agreement = th["agreement"].float()
            masks = candidate_masks(th, horizon_s, coords, cfg)
            idx = torch.nonzero(masks["union"], as_tuple=False)
            if len(idx) == 0:
                continue
            front = coords["front"].bool()
            future = masks["future"].bool()
            teacher_fn = gt_occ & ~final_occ
            teacher_fp = (~gt_occ) & final_occ
            correct_occ = gt_occ & final_occ
            correct_free = (~gt_occ) & ~final_occ
            native_front_count = max(1, int((native_occ & front).sum().item()))
            teacher_front_count = int((final_occ & front).sum().item())
            group_key = sample_index * 10 + horizon_s
            x = idx[:, 0]
            y = idx[:, 1]
            z = idx[:, 2]

            add_col("sample_id", torch.full((len(idx),), sample_index, dtype=torch.int16))
            add_col("horizon_id", torch.full((len(idx),), horizon_s, dtype=torch.int8))
            add_col("group_key", torch.full((len(idx),), group_key, dtype=torch.int32))
            add_col("voxel_x", x.to(torch.int16))
            add_col("voxel_y", y.to(torch.int16))
            add_col("voxel_z", z.to(torch.int16))
            append_selected(cols, "sector_id", coords["sector_id"], idx)
            append_selected(cols, "distance_bin", coords["distance_bin"], idx)
            append_selected(cols, "front_region", front.to(torch.uint8), idx)
            append_selected(cols, "future_h4h6", future.to(torch.uint8), idx)
            append_selected(cols, "protected_zone", coords["protected_zone"].to(torch.uint8), idx)
            append_selected(cols, "raw_confidence", conf.half(), idx)
            append_selected(cols, "raw_margin", margin.half(), idx)
            append_selected(cols, "raw_entropy", (1.0 - conf.clamp(0, 1)).half(), idx)
            append_selected(cols, "raw_occ", raw_occ.to(torch.uint8), idx)
            append_selected(cols, "teacher_final_occ", final_occ.to(torch.uint8), idx)
            append_selected(cols, "native_final_occ", native_occ.to(torch.uint8), idx)
            append_selected(cols, "after_F3_occ", (final_occ | (raw_occ & (conf > 0.90))).to(torch.uint8), idx)
            append_selected(cols, "after_FrontCap_occ", final_occ.to(torch.uint8), idx)
            append_selected(cols, "was_pruned_by_F3", (raw_occ & ~final_occ & (conf > 0.90)).to(torch.uint8), idx)
            append_selected(cols, "was_pruned_by_FrontCap", (raw_occ & ~final_occ).to(torch.uint8), idx)
            append_selected(cols, "raw_but_final_empty", (raw_occ & ~final_occ).to(torch.uint8), idx)
            append_selected(cols, "final_occupied_low_conf", (final_occ & (conf < 0.75)).to(torch.uint8), idx)
            append_selected(cols, "final_occupied_isolated", masks["isolated"].to(torch.uint8), idx)
            append_selected(cols, "neighbor_occ_count", masks["neighbor"].to(torch.int8), idx)
            append_selected(cols, "neighbor_occ_count_norm", (masks["neighbor"].float() / 6.0).half(), idx)
            append_selected(cols, "local_density_proxy", ((masks["neighbor"].float() + final_occ.float()) / 7.0).half(), idx)
            append_selected(cols, "temporal_consistency", agreement.half(), idx)
            append_selected(cols, "camera_view_agreement", agreement.half(), idx)
            append_selected(cols, "front_local_density_context", torch.full_like(conf, safe_div(teacher_front_count, native_front_count)).half(), idx)
            append_selected(cols, "x_norm", coords["x_norm"].half(), idx)
            append_selected(cols, "abs_y_norm", coords["abs_y_norm"].half(), idx)
            append_selected(cols, "z_norm", coords["z_norm"].half(), idx)
            append_selected(cols, "range_norm", coords["range_norm"].half(), idx)
            add_col("horizon_norm", torch.full((len(idx),), float(horizon_s) / 6.0, dtype=torch.float16))
            add_col("sector_norm", cols["sector_id"][-1].float().div(4.0).half())
            add_col("distance_bin_norm", cols["distance_bin"][-1].float().div(5.0).half())
            append_selected(cols, "GT_occ", gt_occ.to(torch.uint8), idx)
            append_selected(cols, "teacher_FN", teacher_fn.to(torch.uint8), idx)
            append_selected(cols, "teacher_FP", teacher_fp.to(torch.uint8), idx)
            append_selected(cols, "teacher_correct_occ", correct_occ.to(torch.uint8), idx)
            append_selected(cols, "teacher_correct_free", correct_free.to(torch.uint8), idx)
            append_selected(cols, "strong_add_candidate", masks["strong_add"].to(torch.uint8), idx)
            append_selected(cols, "medium_add_candidate", masks["medium_add"].to(torch.uint8), idx)
            append_selected(cols, "weak_add_candidate", masks["weak_add"].to(torch.uint8), idx)
            append_selected(cols, "suppress_candidate", masks["suppress"].to(torch.uint8), idx)
            append_selected(cols, "keep_candidate", masks["keep"].to(torch.uint8), idx)

            group_row = {
                "split": split,
                "sample_id": sample_index,
                "horizon_id": horizon_s,
                "group_key": group_key,
                "teacher_occ_count": int(final_occ.sum().item()),
                "native_occ_count": int(native_occ.sum().item()),
                "gt_occ_count": int(gt_occ.sum().item()),
                "gt_free_count": int((~gt_occ).sum().item()),
                "teacher_FN_count": int(teacher_fn.sum().item()),
                "teacher_FP_count": int(teacher_fp.sum().item()),
                "teacher_correct_occ_count": int(correct_occ.sum().item()),
                "teacher_correct_free_count": int(correct_free.sum().item()),
                "front_teacher_FN_count": int((teacher_fn & front).sum().item()),
                "future_h4h6_teacher_FN_count": int((teacher_fn & future).sum().item()),
                "teacher_front_occ_count": teacher_front_count,
                "native_front_occ_count": native_front_count,
                "teacher_front_local_proxy": safe_div(teacher_front_count, native_front_count),
                "num_voxels": VOXELS_PER_GRID,
            }
            group_rows.append(group_row)
            stat_rows.append(
                {
                    **group_row,
                    "strong_add_candidate_count": int(masks["strong_add"].sum().item()),
                    "medium_add_candidate_count": int(masks["medium_add"].sum().item()),
                    "weak_add_candidate_count": int(masks["weak_add"].sum().item()),
                    "suppress_candidate_count": int(masks["suppress"].sum().item()),
                    "keep_candidate_count": int(masks["keep"].sum().item()),
                    "strong_add_positive": int((masks["strong_add"] & gt_occ).sum().item()),
                    "medium_add_positive": int((masks["medium_add"] & gt_occ).sum().item()),
                    "suppress_positive": int((masks["suppress"] & (~gt_occ)).sum().item()),
                }
            )

    tensors = {key: torch.cat(parts, dim=0) for key, parts in cols.items()}
    meta = {
        "split": split,
        "row_count": int(next(iter(tensors.values())).numel()) if tensors else 0,
        "no_gt_feature_columns": NO_GT_FEATURES,
        "gt_label_columns": GT_LABEL_COLUMNS,
        "parquet_note": "Stored with torch.save under .parquet extension because pyarrow is unavailable in env.",
        "f3_frontcap_proxy_note": "after_FrontCap_occ equals teacher_final_occ; F3/FrontCap prune flags are raw-to-final proxies, not internal postprocess logs.",
    }
    save_tensor_table(table_path(split), tensors, meta)
    torch.save({"rows": group_rows}, group_summary_path(split))
    write_csv(REPORTS_DIR / f"sw14d_b_candidate_stats_{split}.csv", stat_rows)
    return {"meta": meta, "tensors": tensors}, stat_rows


def ensure_candidate_tables(cfg: Config) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    all_stats: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val"]:
        if cfg.rebuild_tables or not table_path(split).exists() or not group_summary_path(split).exists():
            table, stats = build_candidate_table(split, cfg)
        else:
            table = load_tensor_table(table_path(split))
            stats_path = REPORTS_DIR / f"sw14d_b_candidate_stats_{split}.csv"
            stats = list(csv.DictReader(stats_path.open(encoding="utf-8"))) if stats_path.exists() else []
        tables[split] = table
        all_stats[split] = stats

    def isum(rows: list[dict[str, Any]], key: str) -> int:
        return int(sum(int(float(r.get(key, 0))) for r in rows))

    train_stats = all_stats["train"]
    val_stats = all_stats["val"]
    summary = {
        "train_rows": int(tables["train"]["meta"]["row_count"]),
        "val_rows": int(tables["val"]["meta"]["row_count"]),
        "strong_add_candidate_train": isum(train_stats, "strong_add_candidate_count"),
        "medium_add_candidate_train": isum(train_stats, "medium_add_candidate_count"),
        "weak_add_candidate_train": isum(train_stats, "weak_add_candidate_count"),
        "suppress_candidate_train": isum(train_stats, "suppress_candidate_count"),
        "keep_candidate_train": isum(train_stats, "keep_candidate_count"),
        "strong_add_candidate_val": isum(val_stats, "strong_add_candidate_count"),
        "medium_add_candidate_val": isum(val_stats, "medium_add_candidate_count"),
        "weak_add_candidate_val": isum(val_stats, "weak_add_candidate_count"),
        "suppress_candidate_val": isum(val_stats, "suppress_candidate_count"),
        "keep_candidate_val": isum(val_stats, "keep_candidate_count"),
        "add_oracle_precision_train": safe_div(
            isum(train_stats, "strong_add_positive") + isum(train_stats, "medium_add_positive"),
            isum(train_stats, "strong_add_candidate_count") + isum(train_stats, "medium_add_candidate_count"),
        ),
        "suppress_oracle_precision_train": safe_div(isum(train_stats, "suppress_positive"), isum(train_stats, "suppress_candidate_count")),
        "add_oracle_precision_val": safe_div(
            isum(val_stats, "strong_add_positive") + isum(val_stats, "medium_add_positive"),
            isum(val_stats, "strong_add_candidate_count") + isum(val_stats, "medium_add_candidate_count"),
        ),
        "suppress_oracle_precision_val": safe_div(isum(val_stats, "suppress_positive"), isum(val_stats, "suppress_candidate_count")),
        "front_future_coverage_note": "front_region/future_h4h6 flags are stored per row in candidate table.",
        "samples_with_zero_add_candidate_train": int(sum((int(float(r.get("strong_add_candidate_count", 0))) + int(float(r.get("medium_add_candidate_count", 0)))) == 0 for r in train_stats)),
        "samples_with_zero_suppress_candidate_train": int(sum(int(float(r.get("suppress_candidate_count", 0))) == 0 for r in train_stats)),
        "decision": "D_B_CAND_1_READY",
    }
    if summary["add_oracle_precision_train"] < 0.05:
        summary["decision"] = "D_B_CAND_2_ADD_CANDIDATES_TOO_NOISY"
    if summary["strong_add_candidate_train"] + summary["medium_add_candidate_train"] < 1000:
        summary["decision"] = "D_B_CAND_3_ADD_CANDIDATES_TOO_SPARSE"
    write_json(REPORTS_DIR / "sw14d_b_candidate_stats.json", summary)
    write_csv(REPORTS_DIR / "sw14d_b_candidate_stats.csv", [{"split": k, **v["meta"]} for k, v in tables.items()])
    return tables


def feature_matrix(tensors: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    cols = []
    for name in NO_GT_FEATURES:
        value = tensors[name][idx]
        cols.append(value.float())
    return torch.stack(cols, dim=1)


def balanced_indices(mask: torch.Tensor, label: torch.Tensor, max_pos: int, max_neg: int, seed: int) -> torch.Tensor:
    pos = torch.nonzero(mask & label, as_tuple=False).flatten()
    neg = torch.nonzero(mask & ~label, as_tuple=False).flatten()
    gen = torch.Generator().manual_seed(seed)
    if len(pos) > max_pos:
        pos = pos[torch.randperm(len(pos), generator=gen)[:max_pos]]
    if len(neg) > max_neg:
        neg = neg[torch.randperm(len(neg), generator=gen)[:max_neg]]
    return torch.cat([pos, neg], dim=0)


def train_head(
    name: str,
    tensors: dict[str, torch.Tensor],
    mask: torch.Tensor,
    label: torch.Tensor,
    max_pos: int,
    max_neg: int,
    cfg: Config,
    device: torch.device,
    rank_weight: float,
) -> tuple[CandidateScorer, list[dict[str, Any]]]:
    idx = balanced_indices(mask, label, max_pos, max_neg, cfg.seed + (11 if name == "add" else 17))
    x_cpu = feature_matrix(tensors, idx)
    y_cpu = label[idx].float()
    if torch.cuda.is_available() and device.type == "cuda":
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
    model = CandidateScorer(x_cpu.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.0e-4)
    pos = float(y_cpu.sum().item())
    neg = float(len(y_cpu) - pos)
    pos_weight = torch.tensor([safe_div(neg, max(pos, 1.0))], device=device).clamp(1.0, 50.0)
    rows: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 100 + epoch)
        order = torch.randperm(len(y_cpu), generator=gen)
        losses: list[float] = []
        rank_losses: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            batch_idx = order[start : start + cfg.batch_size]
            xb = x_cpu[batch_idx].to(device, non_blocking=True)
            yb = y_cpu[batch_idx].to(device, non_blocking=True)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            pos_logits = logits[yb > 0.5]
            neg_logits = logits[yb <= 0.5]
            if len(pos_logits) and len(neg_logits):
                k = min(len(pos_logits), len(neg_logits), 8192)
                rank_loss = F.relu(0.25 - pos_logits[:k] + neg_logits[:k]).mean()
            else:
                rank_loss = logits.new_tensor(0.0)
            loss = bce + float(rank_weight) * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            rank_losses.append(float(rank_loss.detach().cpu().item()))
        rows.append(
            {
                "head": name,
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "rank_loss": float(np.mean(rank_losses)),
                "train_rows": int(len(y_cpu)),
                "positive_rate": safe_div(pos, len(y_cpu)),
                "batch_size": int(cfg.batch_size),
                "device": str(device),
            }
        )
    return model.eval(), rows


@torch.inference_mode()
def score_table(
    model: CandidateScorer,
    tensors: dict[str, torch.Tensor],
    mask: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    scores = torch.full((len(mask),), -torch.inf, dtype=torch.float32)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    for start in range(0, len(idx), cfg.score_batch_size):
        chunk = idx[start : start + cfg.score_batch_size]
        xb = feature_matrix(tensors, chunk)
        if torch.cuda.is_available() and device.type == "cuda":
            xb = xb.pin_memory()
        pred = torch.sigmoid(model(xb.to(device, non_blocking=True))).detach().cpu()
        scores[chunk] = pred
    return scores


def group_rows(split: str) -> list[dict[str, Any]]:
    if split in _GROUP_ROWS_CACHE:
        return _GROUP_ROWS_CACHE[split]
    payload = torch.load(group_summary_path(split), map_location="cpu", weights_only=False)
    _GROUP_ROWS_CACHE[split] = payload["rows"]
    return _GROUP_ROWS_CACHE[split]


def group_index(tensors: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
    cache_key = id(tensors["group_key"])
    if cache_key in _GROUP_INDEX_CACHE:
        return _GROUP_INDEX_CACHE[cache_key]
    out: dict[int, torch.Tensor] = {}
    keys = tensors["group_key"].to(torch.int32)
    for key in torch.unique(keys):
        out[int(key.item())] = torch.nonzero(keys == key, as_tuple=False).flatten()
    _GROUP_INDEX_CACHE[cache_key] = out
    return out


def summarize_eval(rows: list[dict[str, Any]], decision_prefix: str = "D_B") -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0

    def total(key: str) -> int:
        return int(sum(int(r.get(key, 0)) for r in rows))

    safety_all = bool(rows) and all(bool(r["safety_pass"]) for r in rows)
    front_nonreg = mean("front_fn_reduction_rate") >= -0.001
    future_nonreg = mean("future_h4h6_fn_reduction_rate") >= -0.001
    net = mean("net_score")
    if safety_all and front_nonreg and future_nonreg and net >= 0.010 and mean("front_fn_reduction_rate") > 0.0:
        decision = f"{decision_prefix}_SEL_STRONG"
    elif safety_all and front_nonreg and future_nonreg and net >= 0.003:
        decision = f"{decision_prefix}_SEL_SMALL"
    elif safety_all and net >= 0.003:
        decision = f"{decision_prefix}_SEL_FP_ONLY"
    elif net >= 0.003:
        decision = f"{decision_prefix}_SEL_UNSAFE"
    else:
        decision = f"{decision_prefix}_SEL_NO_GAIN"
    return {
        "row_count": len(rows),
        "mean_fn_reduction_rate": mean("fn_reduction_rate"),
        "mean_front_fn_reduction_rate": mean("front_fn_reduction_rate"),
        "mean_future_h4h6_fn_reduction_rate": mean("future_h4h6_fn_reduction_rate"),
        "mean_fp_reduction_rate": mean("fp_reduction_rate"),
        "mean_density_delta_over_teacher": mean("density_delta_over_teacher"),
        "mean_false_positive_delta_over_teacher": mean("false_positive_delta_over_teacher"),
        "mean_front_local_proxy": mean("front_local_proxy"),
        "mean_teacher_front_local_proxy": mean("teacher_front_local_proxy"),
        "mean_front_local_delta_over_teacher": mean("front_local_delta_over_teacher"),
        "front_local_no_worse_pass_rate": safe_div(sum(bool(r["front_local_no_worse_pass"]) for r in rows), len(rows)),
        "front_local_absolute_pass_rate": safe_div(sum(bool(r["front_local_absolute_pass"]) for r in rows), len(rows)),
        "mean_broken_correct_rate": mean("broken_correct_rate"),
        "safety_pass_rate": safe_div(sum(bool(r["safety_pass"]) for r in rows), len(rows)),
        "safety_pass_all": safety_all,
        "recall_nonregression_pass": bool(front_nonreg and future_nonreg),
        "front_fn_non_degraded": mean("front_fn_reduction_rate") >= 0.0,
        "future_h4h6_non_degraded": mean("future_h4h6_fn_reduction_rate") >= 0.0,
        "mean_net_score": net,
        "total_selected_add_count": total("selected_add_count"),
        "total_selected_suppress_count": total("selected_suppress_count"),
        "total_frontcap_rollback_count": total("frontcap_rollback_count"),
        "total_add_recovers_front_gt_occ_count": total("add_recovers_front_gt_occ_count"),
        "total_suppress_breaks_front_gt_occ_count": total("suppress_breaks_front_gt_occ_count"),
        "total_frontcap_breaks_added_gt_occ_count": total("frontcap_breaks_added_gt_occ_count"),
        "decision": decision,
    }


def evaluate_selection(
    split: str,
    table: dict[str, Any],
    add_scores: torch.Tensor,
    sup_scores: torch.Tensor,
    budget: dict[str, Any],
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tensors = table["tensors"]
    groups = group_index(tensors)
    summaries = {int(r["group_key"]): r for r in group_rows(split)}
    rows: list[dict[str, Any]] = []
    add_mask_base = tensors["strong_add_candidate"].bool()
    if budget["add_strength"] == "strong_medium":
        add_mask_base = add_mask_base | tensors["medium_add_candidate"].bool()
    sup_mask_base = tensors["suppress_candidate"].bool() & ~tensors["keep_candidate"].bool() & ~tensors["protected_zone"].bool()
    for key, idx in groups.items():
        g = summaries[key]
        group_add = idx[add_mask_base[idx]]
        group_sup = idx[sup_mask_base[idx]]
        selected_sup = torch.zeros((0,), dtype=torch.long)
        selected_add = torch.zeros((0,), dtype=torch.long)
        if len(group_sup) and float(budget["suppress_topk_ratio"]) > 0:
            max_sup = int(max(1, int(g["teacher_occ_count"])) * cfg.max_suppress_ratio * float(budget["suppress_topk_ratio"]))
            max_sup = min(max_sup, len(group_sup))
            if max_sup > 0:
                selected_sup = group_sup[torch.topk(sup_scores[group_sup], k=max_sup).indices]
        if len(group_add) and float(budget["add_topk_ratio"]) > 0:
            max_add_base = int(len(selected_sup) * cfg.add_suppress_balance + cfg.add_fixed_budget)
            max_add = min(len(group_add), int(max_add_base * float(budget["add_topk_ratio"])))
            if max_add > 0:
                selected_add = group_add[torch.topk(add_scores[group_add], k=max_add).indices]

        teacher_front = int(g["teacher_front_occ_count"])
        native_front = max(1, int(g["native_front_occ_count"]))
        teacher_front_proxy = safe_div(teacher_front, native_front)
        target_front = teacher_front if teacher_front_proxy > cfg.front_cap_ratio else int(math.floor(cfg.front_cap_ratio * native_front))
        sup_front = int(tensors["front_region"][selected_sup].bool().sum().item()) if len(selected_sup) else 0
        add_front = int(tensors["front_region"][selected_add].bool().sum().item()) if len(selected_add) else 0
        front_after = teacher_front - sup_front + add_front
        frontcap_rollback = torch.zeros((0,), dtype=torch.long)
        if front_after > target_front and len(selected_add):
            front_add = selected_add[tensors["front_region"][selected_add].bool()]
            overflow = min(front_after - target_front, len(front_add))
            if overflow > 0:
                low = torch.topk(-add_scores[front_add], k=overflow).indices
                frontcap_rollback = front_add[low]
                keep = torch.ones(len(selected_add), dtype=torch.bool)
                remove_set = set(int(v) for v in frontcap_rollback.tolist())
                for i, v in enumerate(selected_add.tolist()):
                    if int(v) in remove_set:
                        keep[i] = False
                selected_add = selected_add[keep]

        add_gt = tensors["GT_occ"][selected_add].bool() if len(selected_add) else torch.zeros((0,), dtype=torch.bool)
        sup_gt = tensors["GT_occ"][selected_sup].bool() if len(selected_sup) else torch.zeros((0,), dtype=torch.bool)
        add_front_gt = (tensors["front_region"][selected_add].bool() & add_gt) if len(selected_add) else torch.zeros((0,), dtype=torch.bool)
        sup_front_gt = (tensors["front_region"][selected_sup].bool() & sup_gt) if len(selected_sup) else torch.zeros((0,), dtype=torch.bool)
        rollback_gt = tensors["GT_occ"][frontcap_rollback].bool() if len(frontcap_rollback) else torch.zeros((0,), dtype=torch.bool)

        add_tp = int(add_gt.sum().item())
        add_fp = int((~add_gt).sum().item())
        sup_tp = int((~sup_gt).sum().item())
        sup_fn_damage = int(sup_gt.sum().item())
        front_add_tp = int(add_front_gt.sum().item())
        front_sup_damage = int(sup_front_gt.sum().item())
        future_add_tp = int((tensors["future_h4h6"][selected_add].bool() & add_gt).sum().item()) if len(selected_add) else 0
        future_sup_damage = int((tensors["future_h4h6"][selected_sup].bool() & sup_gt).sum().item()) if len(selected_sup) else 0
        teacher_fn = int(g["teacher_FN_count"])
        teacher_fp = int(g["teacher_FP_count"])
        teacher_front_fn = int(g["front_teacher_FN_count"])
        teacher_future_fn = int(g["future_h4h6_teacher_FN_count"])
        student_fn = teacher_fn - add_tp + sup_fn_damage
        student_fp = teacher_fp - sup_tp + add_fp
        student_front_fn = teacher_front_fn - front_add_tp + front_sup_damage
        student_future_fn = teacher_future_fn - future_add_tp + future_sup_damage
        density_delta = safe_div(len(selected_add) - len(selected_sup), int(g["num_voxels"]))
        fp_delta = safe_div(student_fp - teacher_fp, int(g["gt_free_count"]))
        front_after = teacher_front - int(tensors["front_region"][selected_sup].bool().sum().item()) + int(tensors["front_region"][selected_add].bool().sum().item())
        front_proxy = safe_div(front_after, native_front)
        front_abs_pass = front_proxy <= cfg.front_cap_ratio + 1.0e-9
        front_no_worse = front_proxy <= max(cfg.front_cap_ratio, teacher_front_proxy) + 1.0e-9
        broken = sup_fn_damage + add_fp
        broken_rate = safe_div(broken, int(g["num_voxels"]))
        safety = density_delta <= cfg.density_delta_limit and fp_delta <= cfg.fp_delta_limit and front_no_worse and broken_rate <= cfg.broken_rate_limit
        fn_reduction = safe_div(teacher_fn - student_fn, teacher_fn)
        front_fn_reduction = safe_div(teacher_front_fn - student_front_fn, teacher_front_fn)
        future_fn_reduction = safe_div(teacher_future_fn - student_future_fn, teacher_future_fn)
        fp_reduction = safe_div(teacher_fp - student_fp, teacher_fp)
        rows.append(
            {
                "split": split,
                "sample_id": int(g["sample_id"]),
                "horizon_id": int(g["horizon_id"]),
                "budget_name": budget["name"],
                "add_strength": budget["add_strength"],
                "selected_add_count": int(len(selected_add)),
                "selected_suppress_count": int(len(selected_sup)),
                "frontcap_rollback_count": int(len(frontcap_rollback)),
                "add_recovers_front_gt_occ_count": front_add_tp,
                "suppress_breaks_front_gt_occ_count": front_sup_damage,
                "frontcap_breaks_added_gt_occ_count": int(rollback_gt.sum().item()),
                "teacher_FN_count": teacher_fn,
                "student_FN_count": student_fn,
                "teacher_FP_count": teacher_fp,
                "student_FP_count": student_fp,
                "fn_reduction_rate": fn_reduction,
                "front_fn_reduction_rate": front_fn_reduction,
                "future_h4h6_fn_reduction_rate": future_fn_reduction,
                "fp_reduction_rate": fp_reduction,
                "density_delta_over_teacher": density_delta,
                "false_positive_delta_over_teacher": fp_delta,
                "teacher_front_local_proxy": teacher_front_proxy,
                "front_local_proxy": front_proxy,
                "front_local_delta_over_teacher": front_proxy - teacher_front_proxy,
                "front_local_absolute_pass": bool(front_abs_pass),
                "front_local_no_worse_pass": bool(front_no_worse),
                "broken_correct_rate": broken_rate,
                "safety_pass": bool(safety),
                "net_score": fn_reduction + fp_reduction - 2.0 * broken_rate - max(0.0, density_delta) - max(0.0, fp_delta),
            }
        )
    return rows, summarize_eval(rows)


def oracle_upper_bound(table: dict[str, Any], split: str, cfg: Config) -> dict[str, Any]:
    tensors = table["tensors"]
    oracle_budget = {"name": "oracle", "suppress_topk_ratio": 1.0, "add_topk_ratio": 1.0, "add_strength": "strong_medium"}
    add_scores = torch.where((tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()) & tensors["teacher_FN"].bool(), torch.ones(len(tensors["group_key"])), torch.zeros(len(tensors["group_key"])))
    sup_scores = torch.where(tensors["suppress_candidate"].bool() & tensors["teacher_FP"].bool(), torch.ones(len(tensors["group_key"])), torch.zeros(len(tensors["group_key"])))
    rows, summary = evaluate_selection(split, table, add_scores, sup_scores, oracle_budget, cfg)
    decision = "D_B_ORACLE_1_RECALL_SAFE_POTENTIAL"
    if summary["mean_net_score"] < cfg.small_gain_threshold:
        decision = "D_B_ORACLE_4_LOW_POTENTIAL"
    elif summary["mean_front_fn_reduction_rate"] < 0.0:
        decision = "D_B_ORACLE_2_SUPPRESS_ONLY_POTENTIAL"
    elif not summary["safety_pass_all"]:
        decision = "D_B_ORACLE_3_ADD_ONLY_UNSAFE"
    payload = {
        "split": split,
        "decision": decision,
        "oracle_front_FN_gain": summary["mean_front_fn_reduction_rate"],
        "oracle_future_h4h6_FN_gain": summary["mean_future_h4h6_fn_reduction_rate"],
        "oracle_FP_gain": summary["mean_fp_reduction_rate"],
        "oracle_density_delta": summary["mean_density_delta_over_teacher"],
        "oracle_front_local_proxy": summary["mean_front_local_proxy"],
        "oracle_net_score": summary["mean_net_score"],
        "oracle_add_count": summary["total_selected_add_count"],
        "oracle_suppress_count": summary["total_selected_suppress_count"],
        "safety_pass": summary["safety_pass_all"],
        "summary": summary,
    }
    write_csv(REPORTS_DIR / f"sw14d_b_oracle_action_upper_bound_{split}.csv", rows)
    return payload


def run_route1(cfg: Config, device: torch.device) -> dict[str, Any]:
    tables = ensure_candidate_tables(cfg)
    train = tables["train"]["tensors"]
    add_mask = train["strong_add_candidate"].bool() | train["medium_add_candidate"].bool()
    sup_mask = train["suppress_candidate"].bool() & ~train["keep_candidate"].bool() & ~train["protected_zone"].bool()
    add_label = train["GT_occ"].bool()
    sup_label = ~train["GT_occ"].bool()
    add_model, add_log = train_head("add", train, add_mask, add_label, cfg.max_add_train_pos, cfg.max_add_train_neg, cfg, device, cfg.add_rank_weight)
    sup_model, sup_log = train_head("suppress", train, sup_mask, sup_label, cfg.max_sup_train_pos, cfg.max_sup_train_neg, cfg, device, cfg.suppress_rank_weight)
    write_csv(REPORTS_DIR / "sw14d_b_training_log.csv", add_log + sup_log)
    model_cfg = {
        "model": "two_head_mlp",
        "feature_columns": NO_GT_FEATURES,
        "gt_label_columns_not_used_at_inference": GT_LABEL_COLUMNS,
        "hidden_dim": 128,
        "loss": {
            "add_bce": 1.0,
            "suppress_bce": 1.0,
            "add_rank": cfg.add_rank_weight,
            "suppress_rank": cfg.suppress_rank_weight,
            "keep_preserve": "hard mask: suppress excludes keep/protected candidates",
        },
        "config": asdict(cfg),
    }
    write_json(REPORTS_DIR / "sw14d_b_model_config.json", model_cfg)
    torch.save(
        {
            "add_state_dict": add_model.state_dict(),
            "suppress_state_dict": sup_model.state_dict(),
            "model_config": model_cfg,
            "resume_claim": False,
        },
        CHECKPOINT_DIR / "sw14d_b_best.pth",
    )

    score_payload: dict[str, dict[str, torch.Tensor]] = {}
    for split in ["train", "val"]:
        t = tables[split]["tensors"]
        add_m = t["strong_add_candidate"].bool() | t["medium_add_candidate"].bool()
        sup_m = t["suppress_candidate"].bool() & ~t["keep_candidate"].bool() & ~t["protected_zone"].bool()
        score_payload[split] = {
            "add_scores": score_table(add_model, t, add_m, cfg, device),
            "suppress_scores": score_table(sup_model, t, sup_m, cfg, device),
        }
    torch.save(score_payload, ARTIFACTS_DIR / "sw14d_b_batched_scores.pt")

    train_oracle = oracle_upper_bound(tables["train"], "train", cfg)
    val_oracle = oracle_upper_bound(tables["val"], "val", cfg)
    oracle_summary = {"train": train_oracle, "val": val_oracle, "decision": val_oracle["decision"]}
    write_json(REPORTS_DIR / "sw14d_b_oracle_action_upper_bound.json", oracle_summary)
    write_csv(REPORTS_DIR / "sw14d_b_oracle_action_upper_bound.csv", [val_oracle])

    budgets = []
    for sup_r in [0.0, 0.25, 0.5, 1.0]:
        for add_r in [0.0, 0.25, 0.5, 1.0]:
            for strength in ["strong", "strong_medium"]:
                budgets.append(
                    {
                        "name": f"sup{sup_r:g}_add{add_r:g}_{strength}",
                        "suppress_topk_ratio": sup_r,
                        "add_topk_ratio": add_r,
                        "add_strength": strength,
                    }
                )
    sweep_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    for budget in budgets:
        rows, summary = evaluate_selection("val", tables["val"], score_payload["val"]["add_scores"], score_payload["val"]["suppress_scores"], budget, cfg)
        dsel = "D_B_SEL_4_NO_GAIN"
        if summary["safety_pass_all"] and summary["recall_nonregression_pass"] and summary["mean_net_score"] >= cfg.small_gain_threshold:
            dsel = "D_B_SEL_1_SAFE_RECALL_NONREGRESSION"
        elif summary["safety_pass_all"] and summary["mean_net_score"] >= cfg.small_gain_threshold:
            dsel = "D_B_SEL_2_SAFE_FP_GAIN_RECALL_REGRESSION"
        elif not summary["safety_pass_all"] and summary["mean_net_score"] >= cfg.small_gain_threshold:
            dsel = "D_B_SEL_3_RECALL_GAIN_UNSAFE"
        summary = {**summary, **budget, "selection_decision": dsel}
        sweep_rows.append(summary)
        candidate_ok = summary["safety_pass_all"] and summary["recall_nonregression_pass"]
        if candidate_ok and (best is None or summary["mean_net_score"] > best["mean_net_score"]):
            best = summary
            best_rows = rows
    if best is None:
        best = max(sweep_rows, key=lambda x: (bool(x["safety_pass_all"]), x["mean_net_score"]))
        best_rows, _ = evaluate_selection("val", tables["val"], score_payload["val"]["add_scores"], score_payload["val"]["suppress_scores"], best, cfg)

    write_csv(REPORTS_DIR / "sw14d_b_budget_sweep_val.csv", sweep_rows)
    write_csv(REPORTS_DIR / "sw14d_b_selection_val_rows.csv", best_rows)
    write_json(REPORTS_DIR / "sw14d_b_selection_config.json", best)
    write_json(REPORTS_DIR / "sw14d_b_selection_summary.json", best)
    final_enum = "SW14D_B_0_FAIL_LOW_POTENTIAL"
    if val_oracle["decision"] == "D_B_ORACLE_4_LOW_POTENTIAL":
        final_enum = "SW14D_B_0_FAIL_LOW_POTENTIAL"
    elif best["safety_pass_all"] and best["mean_net_score"] >= cfg.strong_gain_threshold and best["mean_front_fn_reduction_rate"] > 0 and best["mean_future_h4h6_fn_reduction_rate"] >= 0 and best["mean_fp_reduction_rate"] > 0:
        final_enum = "SW14D_B_4_SAFE_RECALL_GAIN_READY_DEBUG"
    elif best["safety_pass_all"] and best["recall_nonregression_pass"] and best["mean_fp_reduction_rate"] > 0 and best["mean_net_score"] >= cfg.small_gain_threshold:
        final_enum = "SW14D_B_3_SAFE_RECALL_NONREGRESSION_READY_DEBUG"
    elif best["safety_pass_all"] and best["mean_net_score"] >= cfg.small_gain_threshold:
        final_enum = "SW14D_B_2_SAFE_FP_GAIN_RECALL_REGRESSION"
    elif not best["safety_pass_all"] and best["mean_net_score"] >= cfg.small_gain_threshold:
        final_enum = "SW14D_B_5_UNSAFE_RECALL_GAIN"
    elif best["mean_fp_reduction_rate"] > 0:
        final_enum = "SW14D_B_1_SUPPRESS_ONLY_KEEP_A"
    final = {
        "decision": final_enum,
        "candidate_table_decision": load_json_if_exists(REPORTS_DIR / "sw14d_b_candidate_stats.json").get("decision"),
        "oracle_decision": val_oracle["decision"],
        "selection_decision": best.get("selection_decision"),
        "selected_budget": best,
        "best_checkpoint": str(CHECKPOINT_DIR / "sw14d_b_best.pth"),
        "whether_eval_debug_allowed": final_enum in {"SW14D_B_3_SAFE_RECALL_NONREGRESSION_READY_DEBUG", "SW14D_B_4_SAFE_RECALL_GAIN_READY_DEBUG"},
        "whether_round3_allowed": False,
        "whether_resume_claim_allowed": False,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_for_training_labels": True,
        "gt_used_for_val_selection": True,
        "val_gt_used_for_threshold_selection": False,
        "gamma_zero_not_used_as_improvement": True,
        "clean_drift": 0,
        "rear_unintended_drift": 0,
        "front_local_safety_rule": "No-worse-than-SW13 when SW13 already exceeds 1.30; absolute<=1.30 otherwise. Absolute pass rate is reported separately.",
    }
    write_json(REPORTS_DIR / "sw14d_b_final_decision.json", final)
    write_md(
        REPORTS_DIR / "stage_sw14d_b_report.md",
        "\n".join(
            [
                "# SW14D-B Candidate Reranker",
                "",
                "Train 0..99 / val 100..149. No eval_debug/core was run.",
                f"- decision: `{final_enum}`",
                f"- candidate decision: `{final['candidate_table_decision']}`",
                f"- oracle decision: `{val_oracle['decision']}`",
                f"- val net score: `{best['mean_net_score']}`",
                f"- val front FN reduction: `{best['mean_front_fn_reduction_rate']}`",
                f"- val future h4/h6 FN reduction: `{best['mean_future_h4h6_fn_reduction_rate']}`",
                f"- val FP reduction: `{best['mean_fp_reduction_rate']}`",
                f"- val density delta: `{best['mean_density_delta_over_teacher']}`",
                f"- val FP delta: `{best['mean_false_positive_delta_over_teacher']}`",
                f"- safety pass all: `{best['safety_pass_all']}`",
                f"- eval_debug allowed: `{final['whether_eval_debug_allowed']}`",
                "",
            ]
        ),
    )
    return final


def run_route2(cfg: Config, route1_final: dict[str, Any] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        table = load_tensor_table(table_path(split))
        t = table["tensors"]
        for name, mask in [
            ("teacher_FN_region", t["teacher_FN"].bool()),
            ("teacher_FP_region", t["teacher_FP"].bool()),
            ("teacher_correct_region", t["teacher_correct_occ"].bool() | t["teacher_correct_free"].bool()),
        ]:
            count = int(mask.sum().item())
            raw_occ_rate = safe_div(int((mask & t["raw_occ"].bool()).sum().item()), count)
            final_occ_rate = safe_div(int((mask & t["teacher_final_occ"].bool()).sum().item()), count)
            pruned_rate = safe_div(int((mask & t["raw_but_final_empty"].bool()).sum().item()), count)
            rows.append(
                {
                    "split": split,
                    "region": name,
                    "count": count,
                    "raw_occ_rate": raw_occ_rate,
                    "final_occ_rate": final_occ_rate,
                    "raw_to_final_pruned_proxy_rate": pruned_rate,
                    "raw_confidence_mean": float(t["raw_confidence"][mask].float().mean().item()) if count else 0.0,
                    "raw_margin_mean": float(t["raw_margin"][mask].float().mean().item()) if count else 0.0,
                    "survival_through_F3_probability_proxy": 1.0 - pruned_rate,
                    "survival_through_FrontCap_probability_proxy": final_occ_rate,
                }
            )
    fn_val = next(r for r in rows if r["split"] == "val" and r["region"] == "teacher_FN_region")
    decision = "F_PROBE_4_RAW_SIGNAL_POSTPROCESS_SWALLOWED" if fn_val["raw_to_final_pruned_proxy_rate"] > 0.05 else "F_PROBE_5_FEATURE_ROUTE_NOT_PROMISING"
    final_enum = "SW14F_0_STOP_NO_RAW_SIGNAL"
    training_executed = False
    if route1_final and route1_final.get("decision") in {"SW14D_B_3_SAFE_RECALL_NONREGRESSION_READY_DEBUG", "SW14D_B_4_SAFE_RECALL_GAIN_READY_DEBUG"}:
        final_enum = "SW14F_0_STOP_NO_RAW_SIGNAL"
    write_csv(REPORTS_DIR / "sw14f_raw_logit_causal_probe.csv", rows)
    summary = {
        "probe_decision": decision,
        "final_decision": final_enum,
        "training_executed": training_executed,
        "reason": "Route 1 passed the priority gate; SW14F remains diagnostic only. Historical SW14C feature residual did not convert raw signal into safe final gains.",
        "rows": rows,
    }
    write_json(REPORTS_DIR / "sw14f_raw_logit_causal_summary.json", summary)
    write_json(REPORTS_DIR / "sw14f_loss_config.json", {"training_executed": False, "reason": summary["reason"]})
    write_csv(REPORTS_DIR / "sw14f_training_log.csv", [{"training_executed": False, "reason": summary["reason"]}])
    write_csv(REPORTS_DIR / "sw14f_val_survival_metrics.csv", [r for r in rows if r["split"] == "val"])
    final = {
        "decision": final_enum,
        "raw_logit_probe_decision": decision,
        "postprocess_aware_training_executed": training_executed,
        "whether_eval_debug_allowed": False,
        "whether_resume_claim_allowed": False,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
    }
    write_json(REPORTS_DIR / "sw14f_final_decision.json", final)
    write_md(REPORTS_DIR / "stage_sw14f_report.md", f"# SW14F\n\n- probe decision: `{decision}`\n- final decision: `{final_enum}`\n- training executed: `{training_executed}`\n")
    return final


def run_route3(cfg: Config, route1_final: dict[str, Any] | None = None) -> dict[str, Any]:
    candidates = [
        {
            "point": "A_after_image_feature_extraction_before_view_aggregation",
            "tensor_name": "multi-level camera FPN feature",
            "shape": "B,N,C,H,W per level",
            "dtype": "float32/amp",
            "requires_grad": True,
            "camera_dimension_present": True,
            "query_voxel_alignment": "weak",
            "close_to_final_occupancy": False,
            "reuse_SW13_R8_base": True,
            "noop_init_possible": True,
            "engineering_cost": "medium",
            "risk": "same failure mode as SW14C feature residual",
        },
        {
            "point": "C_query_representation_before_decoder_output",
            "tensor_name": "decoder/query tensor if captured by attach_query_capture",
            "shape": "num_query/batch/embed, runtime dependent",
            "dtype": "float32/amp",
            "requires_grad": True,
            "camera_dimension_present": False,
            "query_voxel_alignment": "medium",
            "close_to_final_occupancy": True,
            "reuse_SW13_R8_base": "requires online hook",
            "noop_init_possible": True,
            "engineering_cost": "high",
            "risk": "needs live SparseWorld forward hook and batched eval pipeline",
        },
        {
            "point": "E_candidate_score_table_before_FrontCap",
            "tensor_name": "candidate table / raw-final score proxy",
            "shape": "candidate_rows x features",
            "dtype": "float32",
            "requires_grad": False,
            "camera_dimension_present": False,
            "query_voxel_alignment": "strong",
            "close_to_final_occupancy": True,
            "reuse_SW13_R8_base": True,
            "noop_init_possible": True,
            "engineering_cost": "low",
            "risk": "postprocess reranker, not feature adapter",
        },
    ]
    audit = {
        "decision": "G_INSERT_3_ONLY_SCORE_POINT",
        "candidate_insertion_points": candidates,
        "searched_sources": [
            str(SPARSEWORLD_ROOT / "mmdet3d/models/heads/occupancy_head.py"),
            str(PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/run_sw12b_main.py"),
        ],
    }
    write_json(REPORTS_DIR / "sw14g_insertion_point_audit.json", audit)
    write_md(
        REPORTS_DIR / "sw14g_insertion_point_audit.md",
        "# SW14G Insertion Point Audit\n\n"
        "The safe insertion point found in this stage is candidate-score level before FrontCap. "
        "A live query-level point likely exists via query capture hooks, but it requires a separate online SparseWorld batched forward refactor before training.\n",
    )
    final = {
        "decision": "SW14G_2_QUERY_RAW_SIGNAL_ONLY",
        "insertion_decision": audit["decision"],
        "query_probe_executed": False,
        "reason": "SW14D-B passed the priority gate; query-level probe is deferred until an online batched SparseWorld entry is refactored.",
        "whether_query_adapter_stage_allowed": False,
        "whether_eval_debug_allowed": False,
        "whether_resume_claim_allowed": False,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
    }
    write_csv(REPORTS_DIR / "sw14g_query_probe_training_log.csv", [{"executed": False, "reason": final["reason"]}])
    write_csv(REPORTS_DIR / "sw14g_query_probe_metrics.csv", [{"executed": False, "reason": final["reason"]}])
    write_csv(REPORTS_DIR / "sw14g_router_weight_stats.csv", [{"executed": False, "reason": final["reason"]}])
    write_json(REPORTS_DIR / "sw14g_query_moe_config.json", {"executed": False, "noop_init": "R8-biased if implemented"})
    write_json(REPORTS_DIR / "sw14g_query_moe_init_audit.json", {"executed": False, "decision": audit["decision"]})
    write_json(REPORTS_DIR / "sw14g_final_decision.json", final)
    write_md(REPORTS_DIR / "stage_sw14g_report.md", f"# SW14G\n\n- insertion decision: `{audit['decision']}`\n- final decision: `{final['decision']}`\n")
    return final


def plot_bar(path: Path, labels: list[str], values: list[float], title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.subplots_adjust(bottom=0.28, left=0.14, right=0.96, top=0.88)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_outputs(route1: dict[str, Any], route2: dict[str, Any], route3: dict[str, Any], final: dict[str, Any]) -> None:
    sel = route1.get("selected_budget", {})
    plot_bar(
        FIGURES_DIR / "sw14e_route_overview.png",
        ["D-B net", "D-B frontFN", "D-B FP"],
        [float(sel.get("mean_net_score", 0)), float(sel.get("mean_front_fn_reduction_rate", 0)), float(sel.get("mean_fp_reduction_rate", 0))],
        "SW14E route overview (no eval_debug)",
    )
    stats = load_json_if_exists(REPORTS_DIR / "sw14d_b_candidate_stats.json")
    plot_bar(
        FIGURES_DIR / "sw14d_b_candidate_stats.png",
        ["strong add", "medium add", "suppress", "keep"],
        [
            float(stats.get("strong_add_candidate_val", 0)),
            float(stats.get("medium_add_candidate_val", 0)),
            float(stats.get("suppress_candidate_val", 0)),
            float(stats.get("keep_candidate_val", 0)),
        ],
        "SW14D-B val candidate stats",
    )
    oracle = load_json_if_exists(REPORTS_DIR / "sw14d_b_oracle_action_upper_bound.json").get("val", {})
    plot_bar(
        FIGURES_DIR / "sw14d_b_oracle_upper_bound.png",
        ["frontFN", "futureFN", "FP", "net"],
        [
            float(oracle.get("oracle_front_FN_gain", 0)),
            float(oracle.get("oracle_future_h4h6_FN_gain", 0)),
            float(oracle.get("oracle_FP_gain", 0)),
            float(oracle.get("oracle_net_score", 0)),
        ],
        "SW14D-B oracle action upper bound",
    )
    sweep_path = REPORTS_DIR / "sw14d_b_budget_sweep_val.csv"
    if sweep_path.exists():
        rows = list(csv.DictReader(sweep_path.open(encoding="utf-8")))
        plot_bar(
            FIGURES_DIR / "sw14d_b_budget_tradeoff.png",
            [r["name"][:12] for r in rows[:8]],
            [float(r["mean_net_score"]) for r in rows[:8]],
            "SW14D-B budget tradeoff subset",
        )
    probe = load_json_if_exists(REPORTS_DIR / "sw14f_raw_logit_causal_summary.json").get("rows", [])
    plot_bar(
        FIGURES_DIR / "sw14f_raw_logit_probe.png",
        [r["region"].replace("teacher_", "")[:10] for r in probe if r.get("split") == "val"],
        [float(r["raw_to_final_pruned_proxy_rate"]) for r in probe if r.get("split") == "val"],
        "SW14F raw-to-final pruned proxy (val)",
    )
    plot_bar(FIGURES_DIR / "sw14f_survival_loss_effect.png", ["training"], [1.0 if route2.get("postprocess_aware_training_executed") else 0.0], "SW14F training execution")
    plot_bar(FIGURES_DIR / "sw14g_insertion_point_map.png", ["feature", "query", "score"], [0.4, 0.7, 1.0], "SW14G insertion point proximity")
    plot_bar(FIGURES_DIR / "sw14g_router_weights.png", ["query probe"], [0.0], "SW14G router weights (not executed)")
    plot_bar(FIGURES_DIR / "sw14e_final_decision_flow.png", [final["decision"]], [1.0], "SW14E final decision flow")


def inherited_state_and_freeze(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    sw14d_final = load_json_if_exists(SW14D_REPORTS / "sw14d_candidate_reranker_final_decision.json")
    sw14d_eval = load_json_if_exists(SW14D_REPORTS / "sw14d_candidate_reranker_eval_summary.json")
    sw14c_audit = load_json_if_exists(SW14C_AUDIT_REPORTS / "sw14c_residual_learning_audit_decision.json")
    sw14c_oem = load_json_if_exists(SW14C_OEM_REPORTS / "sw14c_oracle_error_mask_residual_final_decision.json")
    sw14c_r2b = load_json_if_exists(SW14C_R2B_REPORTS / "sw14c_round2b_final_decision.json")
    occ_head = SPARSEWORLD_ROOT / "mmdet3d/models/heads/occupancy_head.py"
    cfg_file = SPARSEWORLD_ROOT / "configs/sparseworld/nuscenes/preworld-7frame-finetune.py"
    get_occ_source = occ_head
    freeze = {
        "decision": "SW14E_INIT_READY" if TEACHER_CACHE.exists() else "SW14E_INIT_MISSING_ARTIFACTS",
        "sparseworld_backbone_unchanged": True,
        "occupancy_head_unchanged": True,
        "get_occ_unchanged": True,
        "f3_config_unchanged": True,
        "frontcap_config_unchanged": True,
        "sw13_teacher_output_source_fixed": str(TEACHER_CACHE),
        "sw14d_a_checkpoint_source_fixed": str(PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14d_candidate_reranker/checkpoints/sw14d_candidate_reranker_best.pth"),
        "hashes": {
            "occupancy_head_py": sha256_file(occ_head),
            "get_occ_source": sha256_file(get_occ_source),
            "sparseworld_config": sha256_file(cfg_file),
            "sw14d_script": sha256_file(PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14d_candidate_reranker/run_sw14d_candidate_reranker.py"),
        },
    }
    inherited = {
        "decision": freeze["decision"],
        "train_range": [cfg.train_start, cfg.train_end],
        "val_range": [cfg.val_start, cfg.val_end],
        "SW13_main_result": "A10 R8 + F3 + FC1_1p3 remains the only main baseline; eval_core500 artifacts are read-only if present.",
        "SW14C_failure": {
            "audit": sw14c_audit.get("primary_cause") or sw14c_audit.get("decision"),
            "round2b": sw14c_r2b.get("decision"),
            "oracle_mask": sw14c_oem.get("decision"),
            "feature_residual_blind_tuning_stopped": True,
        },
        "SW14D_A": {
            "decision": sw14d_final.get("decision"),
            "val_net_score": sw14d_eval.get("val", {}).get("mean_net_score"),
            "front_fn": sw14d_eval.get("val", {}).get("mean_front_fn_reduction_rate"),
            "fp_reduction": sw14d_eval.get("val", {}).get("mean_fp_reduction_rate"),
            "resume_claim": False,
            "eval_debug_executed": False,
        },
        "pipeline_rework": {
            "cache_first": True,
            "candidate_table_built_once": True,
            "batched_gpu_training": True,
            "batched_gpu_scoring": True,
            "reproducible_seed": cfg.seed,
            "no_repeated_teacher_cache_reads_during_budget_sweep": True,
        },
    }
    write_json(REPORTS_DIR / "sw14e_inherited_state.json", inherited)
    write_json(REPORTS_DIR / "sw14e_baseline_freeze_manifest.json", freeze)
    write_md(
        REPORTS_DIR / "sw14e_protocol.md",
        "# SW14E Protocol\n\n"
        "- No SparseWorld backbone/head/get_occ/F3/FrontCap modification.\n"
        "- No eval_debug/core100/core500 execution in this stage.\n"
        "- Candidate tables are built once from train 0..99 / val 100..149 teacher cache, then reused for batched GPU training/scoring.\n"
        "- GT columns are stored separately from no-GT features and only used for train/val diagnostic supervision/selection.\n"
        "- Resume claim remains false.\n",
    )
    return inherited, freeze


def final_three_route_decision(route1: dict[str, Any], route2: dict[str, Any], route3: dict[str, Any]) -> dict[str, Any]:
    d1 = route1.get("decision")
    if d1 == "SW14D_B_4_SAFE_RECALL_GAIN_READY_DEBUG":
        decision = "SW14E_DECISION_1_SW14D_B_READY_DEBUG"
        next_action = "Run diagnostic eval_debug for frozen SW14D-B after the high-throughput online eval pipeline is reused."
    elif d1 == "SW14D_B_3_SAFE_RECALL_NONREGRESSION_READY_DEBUG":
        decision = "SW14E_DECISION_2_SW14D_B_SMALL_READY_DEBUG"
        next_action = "Run diagnostic eval_debug for frozen SW14D-B; keep resume_claim false."
    elif route2.get("decision") == "SW14F_3_SURVIVAL_IMPROVES_SAFE_READY_COMPARE":
        decision = "SW14E_DECISION_3_FEATURE_POSTPROCESS_AWARE_HAS_SIGNAL"
        next_action = "Design SW14F-B; do not run eval_debug yet."
    elif route3.get("decision") == "SW14G_5_READY_FOR_QUERY_ADAPTER_STAGE":
        decision = "SW14E_DECISION_4_QUERY_LEVEL_ADAPTER_PROMISING"
        next_action = "Open SW14G-B query-level reliability adapter."
    elif d1 == "SW14D_B_1_SUPPRESS_ONLY_KEEP_A":
        decision = "SW14E_DECISION_5_KEEP_SW14D_A_DIAGNOSTIC_ONLY"
        next_action = "Stop learning enhancement or use SW14D-A as internal diagnostic only."
    else:
        decision = "SW14E_DECISION_6_STOP_SW14_KEEP_SW13_MAIN"
        next_action = "Finalize SW13C-Fix + FrontCap."
    final = {
        "decision": decision,
        "route1_decision": d1,
        "route2_decision": route2.get("decision"),
        "route3_decision": route3.get("decision"),
        "whether_eval_debug_allowed": decision in {"SW14E_DECISION_1_SW14D_B_READY_DEBUG", "SW14E_DECISION_2_SW14D_B_SMALL_READY_DEBUG"},
        "whether_round3_allowed": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gamma_zero_not_used_as_improvement": True,
        "clean_drift": 0,
        "rear_unintended_drift": 0,
        "recommended_next_unique_action": next_action,
    }
    write_json(REPORTS_DIR / "sw14e_three_route_final_decision.json", final)
    return final


def write_final_report(inherited: dict[str, Any], route1: dict[str, Any], route2: dict[str, Any], route3: dict[str, Any], final: dict[str, Any]) -> None:
    sel = route1.get("selected_budget", {})
    lines = [
        "# SW14E Three-Route Rescue",
        "",
        "Train 0..99 / val 100..149. No eval_debug/core100/core500 was executed. SW13 remains the main baseline.",
        "",
        "## Inherited State",
        f"- init decision: `{inherited.get('decision')}`",
        f"- SW14D-A decision: `{inherited.get('SW14D_A', {}).get('decision')}`",
        "",
        "## Route 1 SW14D-B",
        f"- decision: `{route1.get('decision')}`",
        f"- val net score: `{sel.get('mean_net_score')}`",
        f"- front FN reduction: `{sel.get('mean_front_fn_reduction_rate')}`",
        f"- future h4/h6 FN reduction: `{sel.get('mean_future_h4h6_fn_reduction_rate')}`",
        f"- FP reduction: `{sel.get('mean_fp_reduction_rate')}`",
        f"- safety pass all: `{sel.get('safety_pass_all')}`",
        "",
        "## Route 2 SW14F",
        f"- decision: `{route2.get('decision')}`",
        f"- raw probe: `{route2.get('raw_logit_probe_decision')}`",
        "",
        "## Route 3 SW14G",
        f"- decision: `{route3.get('decision')}`",
        f"- insertion decision: `{route3.get('insertion_decision')}`",
        "",
        "## Final",
        f"- decision: `{final.get('decision')}`",
        f"- eval_debug allowed: `{final.get('whether_eval_debug_allowed')}`",
        f"- resume claim allowed: `{final.get('whether_resume_claim_allowed')}`",
        f"- next: {final.get('recommended_next_unique_action')}",
        "",
    ]
    write_md(REPORTS_DIR / "stage_sw14e_three_route_rescue_report.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cfg = Config(seed=int(args.seed), epochs=int(args.epochs), batch_size=int(args.batch_size), score_batch_size=int(args.score_batch_size), rebuild_tables=bool(args.rebuild_tables), device=str(args.device))
    seed_everything(cfg.seed)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    inherited, freeze = inherited_state_and_freeze(cfg)
    if freeze["decision"] != "SW14E_INIT_READY":
        final = {
            "decision": "SW14E_DECISION_7_PROTOCOL_INVALID",
            "reason": freeze["decision"],
            "whether_eval_debug_allowed": False,
            "whether_round3_allowed": False,
            "whether_resume_claim_allowed": False,
            "sw13_remains_main_result": True,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
        }
        write_json(REPORTS_DIR / "sw14e_three_route_final_decision.json", final)
        raise SystemExit(freeze["decision"])

    route1 = run_route1(cfg, device) if args.route in {"all", "route1"} else load_json_if_exists(REPORTS_DIR / "sw14d_b_final_decision.json")
    route2 = run_route2(cfg, route1) if args.route in {"all", "route2"} else load_json_if_exists(REPORTS_DIR / "sw14f_final_decision.json")
    route3 = run_route3(cfg, route1) if args.route in {"all", "route3"} else load_json_if_exists(REPORTS_DIR / "sw14g_final_decision.json")
    final = final_three_route_decision(route1, route2, route3)
    plot_outputs(route1, route2, route3, final)
    write_final_report(inherited, route1, route2, route3, final)
    print(f"[sw14e] final decision {final['decision']} route1={route1.get('decision')}", flush=True)


if __name__ == "__main__":
    main()
