from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import random
import sys
import time
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


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[5])))
SW14H8_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h8_query_ranker.py"
SW14H9_DUMP_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h9_sparse_contributor_dump.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h8 = load_module("sw14h8_for_h9_ranker", SW14H8_SCRIPT)
h9dump = load_module("sw14h9_dump_for_ranker", SW14H9_DUMP_SCRIPT)
sw14hi = h8.sw14hi
sw14e = h8.sw14e

H9_RAW_FEATURES = list(h9dump.H9_FEATURE_NAMES)
H9_NORM_FEATURES = [f"{name}_norm" for name in H9_RAW_FEATURES]
H9_GROUP_FEATURES: list[str] = []
for _name in [
    "h9_geo_count_norm",
    "h9_geo_query_count_norm",
    "h9_geo_score_pass_count_norm",
    "h9_geo_top1_max_norm",
    "h9_geo_top1_mean_norm",
    "h9_geo_score_over_thr_max_norm",
    "h9_geo_dist_margin_max_norm",
    "h9_gate_top1_max_norm",
    "h9_gate_fraction_norm",
    "h9_non_gate_geo_count_norm",
]:
    H9_GROUP_FEATURES.extend([f"{_name}_group_rank", f"{_name}_group_z"])
H9_INTERACTION_FEATURES = [
    "h9_geo_count_h7b",
    "h9_score_pass_h7b",
    "h9_geo_top1_h7b",
    "h9_gate_fraction_h7b",
    "h9_geo_count_h8_any",
    "h9_score_pass_h8_any",
    "h9_gate_fraction_h8_any",
    "h9_geo_count_front",
    "h9_geo_count_future",
    "h9_score_pass_front",
    "h9_near_gate_candidate",
]

LEAN_EXTRA_FEATURES = [
    "h2_score_norm",
    "h6_score_norm",
    "h7_score_norm",
    "h7b_prior_score",
]


@dataclass(frozen=True)
class H9RankerConfig:
    seed: int = 307
    epochs: int = 12
    batch_size: int = 262_144
    score_batch_size: int = 1_048_576
    hidden_dim: int = 640
    lr: float = 5.5e-4
    weight_decay: float = 1.0e-4
    target_precision_at_100k: float = 0.90
    device: str = "cuda"


class H9SparseRanker(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        residual = self.net(x).squeeze(-1)
        return torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4)) + residual


def normalize(obj: Any) -> Any:
    return sw14hi.normalize(obj)


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
        writer.writerows([normalize(row) for row in rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14H9 sparse contributor fusion ranker")
    parser.add_argument("--epochs", type=int, default=H9RankerConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=H9RankerConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=H9RankerConfig.hidden_dim)
    parser.add_argument("--device", default=H9RankerConfig.device)
    return parser.parse_args()


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


def add_mask(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()


def build_lean_tables_scores_cache() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    tables = sw14hi.load_tables()
    scores = sw14hi.load_score_payload(tables)
    train_h2, val_h2, train_h6, val_h6 = h8.h7.load_h2_h6_scores(
        len(tables["train"]["tensors"]["GT_occ"]),
        len(tables["val"]["tensors"]["GT_occ"]),
    )
    h7_scores = torch.load(ARTIFACTS_DIR / "sw14h7_groupwise_scores.pt", map_location="cpu", weights_only=False)
    split_scores = {
        "train": (train_h2, train_h6, h7_scores["train_sw14h7_add_scores"].float()),
        "val": (val_h2, val_h6, h7_scores["val_sw14h7_add_scores"].float()),
    }
    caches: dict[str, dict[str, torch.Tensor]] = {}
    for split in ["train", "val"]:
        tensors = tables[split]["tensors"]
        roi = add_mask(tensors)
        cache = sw14hi.derived_cache(tensors, scores[split])
        h2_score, h6_score, h7_score = split_scores[split]
        h2_norm = sw14hi.normalize_score(h2_score.float(), roi).nan_to_num(0.0, neginf=0.0)
        h6_norm = sw14hi.normalize_score(h6_score.float(), roi).nan_to_num(0.0, neginf=0.0)
        h7_norm = sw14hi.normalize_score(h7_score.float(), roi).nan_to_num(0.0, neginf=0.0)
        b3 = cache["b3_add_score_norm"].float().nan_to_num(0.0, neginf=0.0)
        # Lean H7B prior: omits H6 low-value replay feature to avoid loading the 638MB survival table.
        h6b_lean = (0.65 * (0.25 * h2_norm + 0.75 * b3) + 0.20 * h6_norm).clamp(0.0, 1.0)
        h7b = (0.80 * h6b_lean + 0.12 * h7_norm + 0.08 * h2_norm).clamp(0.0, 1.0)
        cache.update(
            {
                "h2_score_norm": h2_norm,
                "h6_score_norm": h6_norm,
                "h7_score_norm": h7_norm,
                "h7b_prior_score": h7b,
            }
        )
        print(f"[h9-ranker] {split} lean_h7b p@100k={h8.precision_at_k(tensors, h7b, 100000):.6f}", flush=True)
        caches[split] = cache
    return tables, scores, caches


def h9_dump_path(split: str) -> Path:
    return ARTIFACTS_DIR / ("sw14h9_sparse_contributor_dump_train_0_99.pt" if split == "train" else "sw14h9_sparse_contributor_dump_val_100_149.pt")


def minmax_norm_full(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((int(values.shape[0]),), dtype=torch.float32)
    vals = values[rows].float().nan_to_num(0.0)
    if len(vals) == 0:
        return out
    vmin = vals.min()
    vmax = vals.max()
    out[rows] = ((vals - vmin) / (vmax - vmin + 1.0e-6)).clamp(0.0, 1.0)
    return out


def add_h9_features(split: str, tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor]) -> dict[str, Any]:
    path = h9_dump_path(split)
    if not path.exists():
        raise FileNotFoundError(f"Missing H9 sparse contributor dump: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload["features"]
    rows = features["row_index"].long()
    expected = torch.nonzero(add_mask(tensors), as_tuple=False).flatten()
    coverage = float(len(rows) / max(1, len(expected)))
    if coverage < 0.98:
        raise RuntimeError(f"{split} H9 coverage too low: {coverage:.4f}")
    n = len(tensors["GT_occ"])
    for name in H9_RAW_FEATURES:
        raw = torch.zeros((n,), dtype=torch.float32)
        raw[rows] = features[name].float().nan_to_num(0.0)
        cache[name] = raw
        cache[f"{name}_norm"] = minmax_norm_full(raw, rows)

    group_keys = tensors["group_key"][rows].long()
    unique_keys = torch.unique(group_keys)
    for name in [
        "h9_geo_count_norm",
        "h9_geo_query_count_norm",
        "h9_geo_score_pass_count_norm",
        "h9_geo_top1_max_norm",
        "h9_geo_top1_mean_norm",
        "h9_geo_score_over_thr_max_norm",
        "h9_geo_dist_margin_max_norm",
        "h9_gate_top1_max_norm",
        "h9_gate_fraction_norm",
        "h9_non_gate_geo_count_norm",
    ]:
        rank_out = torch.zeros((n,), dtype=torch.float32)
        z_out = torch.zeros((n,), dtype=torch.float32)
        vals_full = cache[name].float()
        for key in unique_keys.tolist():
            g_rows = rows[group_keys == int(key)]
            if len(g_rows) <= 1:
                continue
            vals = vals_full[g_rows].nan_to_num(0.0)
            order = torch.argsort(vals, descending=False)
            ranks = torch.empty((len(g_rows),), dtype=torch.float32)
            ranks[order] = torch.arange(len(g_rows), dtype=torch.float32) / float(max(1, len(g_rows) - 1))
            rank_out[g_rows] = ranks
            z = (vals - vals.mean()) / vals.std(unbiased=False).clamp_min(1.0e-4)
            z_out[g_rows] = z.clamp(-5.0, 5.0)
        cache[f"{name}_group_rank"] = rank_out
        cache[f"{name}_group_z"] = z_out

    h7b = cache["h7b_prior_score"].float().nan_to_num(0.0)
    h8_any = cache.get("h8_any_active", torch.zeros_like(h7b)).float().nan_to_num(0.0)
    geo_count = cache["h9_geo_count_norm"]
    score_pass = cache["h9_geo_score_pass_count_norm"]
    gate_fraction = cache["h9_gate_fraction_norm"]
    cache["h9_geo_count_h7b"] = geo_count * h7b
    cache["h9_score_pass_h7b"] = score_pass * h7b
    cache["h9_geo_top1_h7b"] = cache["h9_geo_top1_max_norm"] * h7b
    cache["h9_gate_fraction_h7b"] = gate_fraction * h7b
    cache["h9_geo_count_h8_any"] = geo_count * h8_any
    cache["h9_score_pass_h8_any"] = score_pass * h8_any
    cache["h9_gate_fraction_h8_any"] = gate_fraction * h8_any
    cache["h9_geo_count_front"] = geo_count * tensors["front_region"].float()
    cache["h9_geo_count_future"] = geo_count * tensors["future_h4h6"].float()
    cache["h9_score_pass_front"] = score_pass * tensors["front_region"].float()
    cache["h9_near_gate_candidate"] = (
        (cache["h9_geo_score_over_thr_max"].float() > -0.02).float()
        * (cache["h9_geo_dist_margin_max"].float() > -0.25).float()
        * h7b
    )
    label = tensors["GT_occ"][rows].bool()
    return {
        "split": split,
        "artifact": str(path),
        "rows": int(len(rows)),
        "expected_add_roi_rows": int(len(expected)),
        "coverage": coverage,
        "positive_rows": int(label.sum().item()),
        "positive_rate": float(label.float().mean().item()) if len(label) else 0.0,
        "gt_used_as_inference_feature": False,
    }


def feature_names() -> list[str]:
    return list(sw14hi.H_QUERY_FEATURES) + LEAN_EXTRA_FEATURES + h8.H8_NORM_FEATURES + h8.H8_GROUP_FEATURES + h8.H8_INTERACTION_FEATURES + H9_NORM_FEATURES + H9_GROUP_FEATURES + H9_INTERACTION_FEATURES


def feature_matrix(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    return sw14hi.query_matrix(tensors, cache, idx, feature_names()).float().nan_to_num(0.0)


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, max_pairs: int = 65_536) -> torch.Tensor:
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    neg_prior = prior[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    pos_sel = pos[torch.randperm(len(pos), device=logits.device)[:k]]
    neg_sel = neg[torch.topk(neg_prior, k=k).indices]
    return F.softplus(0.85 - pos_sel + neg_sel).mean()


def topk_surrogate_loss(logits: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, frac: float = 0.06) -> torch.Tensor:
    if len(logits) == 0:
        return logits.new_tensor(0.0)
    k = min(len(logits), max(8192, int(len(logits) * frac)))
    top = torch.unique(torch.cat([torch.topk(logits, k=k).indices, torch.topk(prior, k=k).indices]))
    weight = 1.0 + 5.0 * prior[top] + 4.0 * (1.0 - labels[top])
    loss = F.binary_cross_entropy_with_logits(logits[top], labels[top], reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def train_ranker(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: H9RankerConfig, device: torch.device) -> tuple[H9SparseRanker, list[dict[str, Any]]]:
    rows = torch.load(h9_dump_path("train"), map_location="cpu", weights_only=False)["features"]["row_index"].long()
    idx = rows[add_mask(tensors)[rows]]
    y_cpu = tensors["GT_occ"][idx].float()
    prior_cpu = cache["h7b_prior_score"][idx].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    front_cpu = tensors["front_region"][idx].float()
    future_cpu = tensors["future_h4h6"][idx].float()
    density_cpu = tensors["local_density_proxy"][idx].float()
    h9_active_cpu = cache["h9_geo_count_norm"][idx].float()
    if device.type == "cuda":
        for name, tensor in [
            ("y", y_cpu),
            ("prior", prior_cpu),
            ("front", front_cpu),
            ("future", future_cpu),
            ("density", density_cpu),
            ("h9", h9_active_cpu),
        ]:
            try:
                locals()[f"{name}_cpu"] = tensor.pin_memory()
            except RuntimeError:
                pass
    model = H9SparseRanker(len(feature_names()), cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = torch.tensor([(len(y_cpu) - y_cpu.sum().item()) / max(1.0, y_cpu.sum().item())], device=device).clamp(1.0, 8.0)
    rows_log: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        order = torch.randperm(len(idx), generator=torch.Generator().manual_seed(cfg.seed * 1000 + epoch))
        losses: list[float] = []
        ranks: list[float] = []
        topks: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            row_idx = idx[b]
            xb_cpu = feature_matrix(tensors, cache, row_idx)
            if device.type == "cuda":
                try:
                    xb_cpu = xb_cpu.pin_memory()
                except RuntimeError:
                    pass
            xb = xb_cpu.to(device, non_blocking=True)
            yb = y_cpu[b].to(device, non_blocking=True)
            prior = prior_cpu[b].to(device, non_blocking=True)
            front = front_cpu[b].to(device, non_blocking=True)
            future = future_cpu[b].to(device, non_blocking=True)
            density = density_cpu[b].to(device, non_blocking=True)
            h9_active = h9_active_cpu[b].to(device, non_blocking=True)
            logits = model(xb, prior)
            sample_weight = (
                1.0
                + yb * (1.8 + 1.2 * front + 1.2 * future + 0.8 * h9_active)
                + (1.0 - yb) * (3.5 * prior + 1.8 * h9_active + 1.2 * density + 0.8 * front)
            )
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            bce = (bce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            rank = pairwise_rank_loss(logits, yb, prior)
            topk = topk_surrogate_loss(logits, yb, prior)
            prob = torch.sigmoid(logits)
            fp_penalty = (prob * (1.0 - yb) * (1.0 + 3.0 * prior) * (1.0 + 2.0 * h9_active + density)).mean()
            residual = logits - torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
            loss = bce + 3.2 * rank + 4.5 * topk + 1.2 * fp_penalty + 0.008 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 6.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
            topks.append(float(topk.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "pairwise_rank": float(np.mean(ranks)),
            "topk_surrogate": float(np.mean(topks)),
            "train_rows": int(len(idx)),
            "positive_rate": float(y_cpu.mean().item()),
            "batch_size": cfg.batch_size,
            "device": str(device),
            "seconds": float(time.time() - t0),
        }
        rows_log.append(row)
        print(f"[h9-ranker] epoch={epoch} loss={row['loss']:.5f} sec={row['seconds']:.1f}", flush=True)
    return model.eval(), rows_log


@torch.inference_mode()
def score_ranker(model: H9SparseRanker, split: str, tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: H9RankerConfig, device: torch.device) -> torch.Tensor:
    rows = torch.load(h9_dump_path(split), map_location="cpu", weights_only=False)["features"]["row_index"].long()
    idx = rows[add_mask(tensors)[rows]]
    out = torch.full((len(tensors["GT_occ"]),), -torch.inf, dtype=torch.float32)
    for start in range(0, len(idx), cfg.score_batch_size):
        chunk = idx[start : start + cfg.score_batch_size]
        x = feature_matrix(tensors, cache, chunk)
        prior = cache["h7b_prior_score"][chunk].float().clamp(1.0e-4, 1.0 - 1.0e-4)
        if device.type == "cuda":
            try:
                x = x.pin_memory()
                prior = prior.pin_memory()
            except RuntimeError:
                pass
        logits = model(x.to(device, non_blocking=True), prior.to(device, non_blocking=True))
        out[chunk] = torch.sigmoid(logits).detach().cpu()
    return out


def score_maps(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], h9_score: torch.Tensor) -> dict[str, torch.Tensor]:
    h9_norm = sw14hi.normalize_score(h9_score, add_mask(tensors)).nan_to_num(0.0, neginf=0.0)
    h7b = cache["h7b_prior_score"].float()
    return {
        "h7b_prior": h7b,
        "h9_sparse_ranker": h9_score,
        "h9_blend_h7b_10": 0.10 * h9_norm + 0.90 * h7b,
        "h9_blend_h7b_20": 0.20 * h9_norm + 0.80 * h7b,
        "h9_blend_h7b_35": 0.35 * h9_norm + 0.65 * h7b,
        "h9_blend_h7b_50": 0.50 * h9_norm + 0.50 * h7b,
        "h9_geo_count_h7b": cache["h9_geo_count_h7b"],
        "h9_score_pass_h7b": cache["h9_score_pass_h7b"],
        "h9_near_gate_candidate": cache["h9_near_gate_candidate"],
    }


def plot_precision(rows: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for name in ["h7b_prior", "h9_sparse_ranker", "h9_blend_h7b_10", "h9_blend_h7b_20", "h9_blend_h7b_35", "h9_best_val"]:
        sub = sorted([r for r in rows if r["split"] == "val" and r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H9 sparse contributor add precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h9_sparse_contributor_ranker_precision.png", dpi=180)
    plt.close()


def run_selection(tables: dict[str, dict[str, Any]], add_score: torch.Tensor, suppress_score: torch.Tensor) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for sup in [0.02, 0.03, 0.04, 0.05]:
        for bal in [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
            for fixed in [0, 16, 32, 64, 96, 128, 192]:
                cfg = sw14e.Config(max_suppress_ratio=sup, add_suppress_balance=bal, add_fixed_budget=fixed)
                budget = {"name": f"h9_sup{sup}_bal{bal}_fixed{fixed}", "suppress_topk_ratio": 1.0, "add_topk_ratio": 1.0, "add_strength": "strong_medium"}
                _, summary = sw14e.evaluate_selection("val", tables["val"], add_score, suppress_score, budget, cfg)
                row = {**summary, "max_suppress_ratio": sup, "add_suppress_balance": bal, "add_fixed_budget": fixed}
                rows.append(row)
                if summary["safety_pass_all"] and summary["recall_nonregression_pass"]:
                    if best is None or float(summary["mean_net_score"]) > float(best["mean_net_score"]):
                        best = row
    write_csv(REPORTS_DIR / "sw14h9_sparse_contributor_selection_sweep_val.csv", rows)
    return best or (max(rows, key=lambda r: float(r["mean_net_score"])) if rows else {})


def main() -> None:
    args = parse_args()
    cfg = H9RankerConfig(epochs=args.epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim, device=args.device)
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("[h9-ranker] loading lean base/H8/H9 caches", flush=True)
    tables, scores, caches = build_lean_tables_scores_cache()
    h8_stats = [
        h8.add_h8_features("train", tables["train"]["tensors"], caches["train"]),
        h8.add_h8_features("val", tables["val"]["tensors"], caches["val"]),
    ]
    h9_stats = [
        add_h9_features("train", tables["train"]["tensors"], caches["train"]),
        add_h9_features("val", tables["val"]["tensors"], caches["val"]),
    ]
    write_json(
        REPORTS_DIR / "sw14h9_sparse_contributor_ranker_config.json",
        {
            "model": "H9 sparse get_occ contributor fusion residual ranker",
            "target_precision_at_100k": cfg.target_precision_at_100k,
            "feature_count": len(feature_names()),
            "h8_stats": h8_stats,
            "h9_stats": h9_stats,
            "config": asdict(cfg),
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
            "resume_claim": False,
        },
    )
    print("[h9-ranker] training", flush=True)
    model, train_log = train_ranker(tables["train"]["tensors"], caches["train"], cfg, device)
    write_csv(REPORTS_DIR / "sw14h9_sparse_contributor_training_log.csv", train_log)
    torch.save(
        {"state_dict": model.state_dict(), "feature_names": feature_names(), "config": asdict(cfg), "resume_claim": False, "gt_used_as_inference_feature": False},
        CHECKPOINT_DIR / "sw14h9_sparse_contributor_ranker_best.pth",
    )
    print("[h9-ranker] releasing train cache before val scoring", flush=True)
    del tables["train"], scores["train"], caches["train"]
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[h9-ranker] scoring val", flush=True)
    val_h9 = score_ranker(model, "val", tables["val"]["tensors"], caches["val"], cfg, device)
    torch.save({"val_sw14h9_sparse_scores": val_h9}, ARTIFACTS_DIR / "sw14h9_sparse_contributor_scores.pt")
    val_scores = score_maps(tables["val"]["tensors"], caches["val"], val_h9)
    topk = h8.topk_rows(tables["val"]["tensors"], val_scores, "val")
    best_val_100k = max((r for r in topk if int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    h7b_val_100k = next(r for r in topk if r["score_name"] == "h7b_prior" and int(r["topk"]) == 100000)
    val_scores["h9_best_val"] = val_scores[best_val_100k["score_name"]]
    topk_plot = topk + h8.topk_rows(tables["val"]["tensors"], {"h9_best_val": val_scores["h9_best_val"]}, "val")
    write_csv(REPORTS_DIR / "sw14h9_sparse_contributor_topk_precision.csv", topk)
    plot_precision(topk_plot)
    selection = run_selection(tables, val_scores[best_val_100k["score_name"]], scores["val"]["suppress"])
    gain = float(best_val_100k["precision"]) - float(h7b_val_100k["precision"])
    if float(best_val_100k["precision"]) >= cfg.target_precision_at_100k:
        decision = "SW14H9_1_TARGET_090_REACHED"
    elif gain >= 0.10:
        decision = "SW14H9_2_MATERIAL_SPARSE_CONTRIBUTOR_GAIN"
    elif gain > 0.005:
        decision = "SW14H9_3_SMALL_SPARSE_CONTRIBUTOR_GAIN"
    else:
        decision = "SW14H9_4_SPARSE_CONTRIBUTOR_NOT_ENOUGH"
    final = {
        "decision": decision,
        "target_precision_at_100k": cfg.target_precision_at_100k,
        "target_reached": bool(float(best_val_100k["precision"]) >= cfg.target_precision_at_100k),
        "best_val_top100k": best_val_100k,
        "h7b_val_top100k": h7b_val_100k,
        "precision_gain_over_h7b": gain,
        "selection": selection,
        "h9_stats": h9_stats,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "recommended_next_action": "If H9 remains far below 0.9@100K, the next required change is a new proposal source or explicit decoder-query tensor cache, not more scoring on the same add ROI.",
    }
    write_json(REPORTS_DIR / "sw14h9_sparse_contributor_final_decision.json", final)
    print(f"[h9-ranker] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
