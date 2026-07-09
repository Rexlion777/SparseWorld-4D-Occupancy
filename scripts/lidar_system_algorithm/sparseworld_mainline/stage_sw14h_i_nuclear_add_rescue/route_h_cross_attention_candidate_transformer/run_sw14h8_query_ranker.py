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
SW14HI_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/run_sw14h_i_nuclear_add_rescue.py"
SW14H7_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h7_groupwise_ranker.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

H8_RAW_FEATURES = [
    "h8_dense_top1_score",
    "h8_dense_top1_margin",
    "h8_geometric_active",
    "h8_semantic_active",
    "h8_contributor_count",
    "h8_occ_pred_nonempty",
    "h8_gate_pass_count",
    "h8_extra_routed_count",
]
H8_NORM_FEATURES = [f"{name}_norm" for name in H8_RAW_FEATURES]
H8_GROUP_FEATURES = []
for _name in [
    "h8_dense_top1_score_norm",
    "h8_dense_top1_margin_norm",
    "h8_geometric_active_norm",
    "h8_semantic_active_norm",
    "h8_contributor_count_norm",
    "h8_occ_pred_nonempty_norm",
]:
    H8_GROUP_FEATURES.extend([f"{_name}_group_rank", f"{_name}_group_z"])
H8_INTERACTION_FEATURES = [
    "h8_any_active",
    "h8_any_active_h7b",
    "h8_contributor_h7b",
    "h8_geometric_h7b",
    "h8_semantic_h7b",
    "h8_occ_pred_h7b",
    "h8_dense_score_h7b",
    "h8_dense_margin_h7b",
    "h8_contributor_raw_confidence",
    "h8_contributor_neighbor",
    "h8_boundary_active",
    "h8_front_active",
    "h8_future_active",
]
EXTRA_PRIOR_FEATURES = [
    "h7b_prior_score",
    "h7_norm_score",
    "h2_rank_score",
]


@dataclass(frozen=True)
class H8RankerConfig:
    seed: int = 211
    epochs: int = 14
    batch_size: int = 393_216
    score_batch_size: int = 1_048_576
    hidden_dim: int = 768
    lr: float = 6.5e-4
    weight_decay: float = 1.0e-4
    target_precision_at_100k: float = 0.90
    early_stop_patience: int = 4
    device: str = "cuda"


class H8ResidualRanker(nn.Module):
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
        prior_logit = torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
        return prior_logit + residual


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_module("sw14hi_h8_ranker", SW14HI_SCRIPT)
h7 = load_module("sw14h7_for_h8_ranker", SW14H7_SCRIPT)
sw14e = sw14hi.sw14e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14H8 full online query/debug add ranker")
    parser.add_argument("--epochs", type=int, default=H8RankerConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=H8RankerConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=H8RankerConfig.hidden_dim)
    parser.add_argument("--device", default=H8RankerConfig.device)
    parser.add_argument("--score-only", action="store_true", help="Load checkpoint and score val without retraining")
    parser.add_argument("--score-train", action="store_true", help="Also score train; disabled by default to keep memory bounded")
    return parser.parse_args()


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


def rank_norm(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(score, dtype=torch.float32)
    idx = torch.nonzero(mask & torch.isfinite(score.float()), as_tuple=False).flatten()
    if len(idx) <= 1:
        return out
    vals = score[idx].float().nan_to_num(0.0)
    order = torch.argsort(vals, descending=False)
    ranks = torch.empty((len(idx),), dtype=torch.float32)
    ranks[order] = torch.arange(len(idx), dtype=torch.float32) / float(max(1, len(idx) - 1))
    out[idx] = ranks
    return out


def minmax_norm_full(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((int(values.shape[0]),), dtype=torch.float32)
    vals = values[rows].float().nan_to_num(0.0)
    if len(vals) == 0:
        return out
    vmin = vals.min()
    vmax = vals.max()
    out[rows] = ((vals - vmin) / (vmax - vmin + 1.0e-6)).clamp(0.0, 1.0)
    return out


def load_tables_and_base_cache() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    tables = sw14hi.load_tables()
    scores = sw14hi.load_score_payload(tables)
    train_survival = torch.load(ARTIFACTS_DIR / "sw14h6_postprocess_survival_features_train.pt", map_location="cpu", weights_only=False)
    val_survival = torch.load(ARTIFACTS_DIR / "sw14h6_postprocess_survival_features_val.pt", map_location="cpu", weights_only=False)
    train_h2, val_h2, train_h6, val_h6 = h7.load_h2_h6_scores(
        len(tables["train"]["tensors"]["GT_occ"]),
        len(tables["val"]["tensors"]["GT_occ"]),
    )
    train_cache = h7.base_cache(tables["train"]["tensors"], scores["train"], train_survival, train_h2, train_h6)
    val_cache = h7.base_cache(tables["val"]["tensors"], scores["val"], val_survival, val_h2, val_h6)
    train_group = h7.group_rank_cache("train", tables["train"]["tensors"], train_cache)
    val_group = h7.group_rank_cache("val", tables["val"]["tensors"], val_cache)
    train_cache.update({k: v.float().nan_to_num(0.0) for k, v in train_group.items()})
    val_cache.update({k: v.float().nan_to_num(0.0) for k, v in val_group.items()})
    h7_scores = torch.load(ARTIFACTS_DIR / "sw14h7_groupwise_scores.pt", map_location="cpu", weights_only=False)
    train_h7 = h7_scores["train_sw14h7_add_scores"].float()
    val_h7 = h7_scores["val_sw14h7_add_scores"].float()
    for split, cache, tensors, h2_score, h7_score in [
        ("train", train_cache, tables["train"]["tensors"], train_h2, train_h7),
        ("val", val_cache, tables["val"]["tensors"], val_h2, val_h7),
    ]:
        roi = add_mask(tensors)
        h7_norm = sw14hi.normalize_score(h7_score, roi).nan_to_num(0.0, neginf=0.0)
        h2_rank = rank_norm(h2_score.float(), roi)
        h7b = (0.80 * cache["h6b_prior_score"].float() + 0.12 * h7_norm + 0.08 * h2_rank).clamp(0.0, 1.0)
        cache["h7_norm_score"] = h7_norm
        cache["h2_rank_score"] = h2_rank
        cache["h7b_prior_score"] = h7b
        print(f"[h8-ranker] {split} h7b p@100k={precision_at_k(tensors, h7b, 100000):.6f}", flush=True)
    return tables, scores, {"train": train_cache, "val": val_cache}


def h8_dump_path(split: str) -> Path:
    if split == "train":
        return ARTIFACTS_DIR / "sw14h8_online_query_score_dump_train_0_99.pt"
    return ARTIFACTS_DIR / "sw14h8_online_query_score_dump_val_100_149.pt"


def load_h8_dump(split: str) -> dict[str, Any]:
    path = h8_dump_path(split)
    if not path.exists():
        raise FileNotFoundError(f"Missing H8 dump: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def add_h8_features(split: str, tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor]) -> dict[str, Any]:
    payload = load_h8_dump(split)
    features: dict[str, torch.Tensor] = payload["features"]
    rows = features["row_index"].long()
    expected = torch.nonzero(add_mask(tensors), as_tuple=False).flatten()
    coverage = float(len(rows) / max(1, len(expected)))
    if len(torch.unique(rows)) != len(rows):
        raise RuntimeError(f"{split} H8 dump has duplicate row_index")
    if coverage < 0.98:
        raise RuntimeError(f"{split} H8 dump coverage too low: {coverage:.4f}")

    n = len(tensors["GT_occ"])
    for name in H8_RAW_FEATURES:
        raw = torch.zeros((n,), dtype=torch.float32)
        if name in features:
            raw[rows] = features[name].float().nan_to_num(0.0)
        cache[name] = raw
        cache[f"{name}_norm"] = minmax_norm_full(raw, rows)

    group_keys = tensors["group_key"][rows].long()
    unique_keys = torch.unique(group_keys)
    for name in [
        "h8_dense_top1_score_norm",
        "h8_dense_top1_margin_norm",
        "h8_geometric_active_norm",
        "h8_semantic_active_norm",
        "h8_contributor_count_norm",
        "h8_occ_pred_nonempty_norm",
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

    any_active = torch.maximum(
        torch.maximum(cache["h8_geometric_active_norm"], cache["h8_semantic_active_norm"]),
        cache["h8_occ_pred_nonempty_norm"],
    )
    h7b = cache["h7b_prior_score"].float().nan_to_num(0.0)
    cache["h8_any_active"] = any_active
    cache["h8_any_active_h7b"] = any_active * h7b
    cache["h8_contributor_h7b"] = cache["h8_contributor_count_norm"] * h7b
    cache["h8_geometric_h7b"] = cache["h8_geometric_active_norm"] * h7b
    cache["h8_semantic_h7b"] = cache["h8_semantic_active_norm"] * h7b
    cache["h8_occ_pred_h7b"] = cache["h8_occ_pred_nonempty_norm"] * h7b
    cache["h8_dense_score_h7b"] = cache["h8_dense_top1_score_norm"] * h7b
    cache["h8_dense_margin_h7b"] = cache["h8_dense_top1_margin_norm"] * h7b
    cache["h8_contributor_raw_confidence"] = cache["h8_contributor_count_norm"] * tensors["raw_confidence"].float()
    cache["h8_contributor_neighbor"] = cache["h8_contributor_count_norm"] * tensors["neighbor_occ_count_norm"].float()
    boundary = (tensors["was_pruned_by_F3"].float() + tensors["was_pruned_by_FrontCap"].float()).clamp(0.0, 1.0)
    cache["h8_boundary_active"] = any_active * boundary
    cache["h8_front_active"] = any_active * tensors["front_region"].float()
    cache["h8_future_active"] = any_active * tensors["future_h4h6"].float()

    label = tensors["GT_occ"][rows].bool()
    return {
        "split": split,
        "artifact": str(h8_dump_path(split)),
        "rows": int(len(rows)),
        "expected_add_roi_rows": int(len(expected)),
        "coverage": coverage,
        "positive_rows": int(label.sum().item()),
        "positive_rate": float(label.float().mean().item()) if len(label) else 0.0,
        "feature_columns_used": H8_RAW_FEATURES + H8_NORM_FEATURES + H8_GROUP_FEATURES + H8_INTERACTION_FEATURES,
        "gt_used_as_inference_feature": False,
    }


def feature_names() -> list[str]:
    return (
        list(h7.feature_names())
        + EXTRA_PRIOR_FEATURES
        + H8_NORM_FEATURES
        + H8_GROUP_FEATURES
        + H8_INTERACTION_FEATURES
        + [
            "horizon_id",
            "sector_id",
            "distance_bin",
        ]
    )


def feature_matrix(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    return sw14hi.query_matrix(tensors, cache, idx, feature_names()).float().nan_to_num(0.0)


def precision_at_k(tensors: dict[str, torch.Tensor], score: torch.Tensor, k: int) -> float:
    roi = add_mask(tensors)
    idx = torch.nonzero(roi & torch.isfinite(score.float()), as_tuple=False).flatten()
    if len(idx) < k:
        return 0.0
    top = idx[torch.topk(score[idx].float(), k=k).indices]
    return float(tensors["GT_occ"][top].float().mean().item())


def topk_rows(tensors: dict[str, torch.Tensor], scores: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    return h7.topk_rows(tensors, scores, split)


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, max_pairs: int = 65536) -> torch.Tensor:
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    pos_sel = pos[torch.randperm(len(pos), device=logits.device)[:k]]
    neg_hard = neg[torch.topk(prior[labels <= 0.5], k=k).indices]
    return F.softplus(0.75 - pos_sel + neg_hard).mean()


def topk_surrogate_loss(logits: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, frac: float = 0.08) -> torch.Tensor:
    if len(logits) == 0:
        return logits.new_tensor(0.0)
    k = min(len(logits), max(8192, int(len(logits) * frac)))
    top = torch.unique(torch.cat([torch.topk(logits, k=k).indices, torch.topk(prior, k=k).indices]))
    weight = 1.0 + 4.0 * prior[top] + 3.0 * (1.0 - labels[top])
    loss = F.binary_cross_entropy_with_logits(logits[top], labels[top], reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    pt = prob * labels + (1.0 - prob) * (1.0 - labels)
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = loss * (1.0 - pt).pow(2.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def train_ranker(
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    cfg: H8RankerConfig,
    device: torch.device,
) -> tuple[H8ResidualRanker, list[dict[str, Any]]]:
    h8_rows = load_h8_dump("train")["features"]["row_index"].long()
    idx = h8_rows[add_mask(tensors)[h8_rows]]
    y_cpu = tensors["GT_occ"][idx].float()
    prior_cpu = cache["h7b_prior_score"][idx].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    front_cpu = tensors["front_region"][idx].float()
    future_cpu = tensors["future_h4h6"][idx].float()
    density_cpu = tensors["local_density_proxy"][idx].float()
    active_cpu = cache["h8_any_active"][idx].float()
    if device.type == "cuda":
        try:
            y_cpu = y_cpu.pin_memory()
            prior_cpu = prior_cpu.pin_memory()
            front_cpu = front_cpu.pin_memory()
            future_cpu = future_cpu.pin_memory()
            density_cpu = density_cpu.pin_memory()
            active_cpu = active_cpu.pin_memory()
        except RuntimeError:
            print("[h8-ranker] pin_memory skipped due to host memory pressure", flush=True)

    model = H8ResidualRanker(len(feature_names()), cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = torch.tensor([(len(y_cpu) - y_cpu.sum().item()) / max(1.0, y_cpu.sum().item())], device=device).clamp(1.0, 8.0)
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    stale = 0
    for epoch in range(cfg.epochs):
        start_time = time.time()
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + epoch)
        order = torch.randperm(len(idx), generator=gen)
        losses: list[float] = []
        bces: list[float] = []
        focals: list[float] = []
        ranks: list[float] = []
        topks: list[float] = []
        fp_pens: list[float] = []
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
            active = active_cpu[b].to(device, non_blocking=True)
            logits = model(xb, prior)
            sample_weight = (
                1.0
                + yb * (1.8 + 1.4 * front + 1.2 * future + 1.0 * active)
                + (1.0 - yb) * (3.0 * prior + 2.0 * active + 1.2 * density + 0.8 * front)
            )
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            bce = (bce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            foc = focal_loss(logits, yb, sample_weight)
            rank = pairwise_rank_loss(logits, yb, prior)
            topk = topk_surrogate_loss(logits, yb, prior)
            prob = torch.sigmoid(logits)
            fp_penalty = (prob * (1.0 - yb) * (1.0 + 3.0 * prior) * (1.0 + 2.0 * active + density)).mean()
            residual = logits - torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
            loss = bce + 1.5 * foc + 3.0 * rank + 4.0 * topk + 1.0 * fp_penalty + 0.006 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 6.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            bces.append(float(bce.detach().cpu().item()))
            focals.append(float(foc.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
            topks.append(float(topk.detach().cpu().item()))
            fp_pens.append(float(fp_penalty.detach().cpu().item()))
        epoch_loss = float(np.mean(losses))
        row = {
            "epoch": epoch,
            "loss": epoch_loss,
            "bce": float(np.mean(bces)),
            "focal": float(np.mean(focals)),
            "pairwise_rank": float(np.mean(ranks)),
            "topk_surrogate": float(np.mean(topks)),
            "fp_top_penalty": float(np.mean(fp_pens)),
            "train_rows": int(len(idx)),
            "positive_rate": float(y_cpu.mean().item()),
            "batch_size": cfg.batch_size,
            "device": str(device),
            "seconds": float(time.time() - start_time),
        }
        rows.append(row)
        print(f"[h8-ranker] epoch={epoch} loss={epoch_loss:.5f} sec={row['seconds']:.1f}", flush=True)
        if epoch_loss + 1.0e-4 < best_loss:
            best_loss = epoch_loss
            stale = 0
            torch.save({"state_dict": model.state_dict(), "feature_names": feature_names(), "config": asdict(cfg)}, CHECKPOINT_DIR / "sw14h8_query_ranker_best_loss.pth")
        else:
            stale += 1
        if stale >= cfg.early_stop_patience:
            print(f"[h8-ranker] early stop at epoch={epoch}", flush=True)
            break
    return model.eval(), rows


@torch.inference_mode()
def score_ranker(
    model: H8ResidualRanker,
    split: str,
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    cfg: H8RankerConfig,
    device: torch.device,
) -> torch.Tensor:
    h8_rows = load_h8_dump(split)["features"]["row_index"].long()
    idx = h8_rows[add_mask(tensors)[h8_rows]]
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


def run_selection(tables: dict[str, dict[str, Any]], add_score: torch.Tensor, suppress_score: torch.Tensor) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for sup in [0.02, 0.03, 0.04, 0.05]:
        for bal in [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
            for fixed in [0, 16, 32, 64, 96, 128, 192]:
                cfg = sw14e.Config(max_suppress_ratio=sup, add_suppress_balance=bal, add_fixed_budget=fixed)
                budget = {
                    "name": f"h8_sup{sup}_bal{bal}_fixed{fixed}",
                    "suppress_topk_ratio": 1.0,
                    "add_topk_ratio": 1.0,
                    "add_strength": "strong_medium",
                }
                _, summary = sw14e.evaluate_selection("val", tables["val"], add_score, suppress_score, budget, cfg)
                row = {**summary, "max_suppress_ratio": sup, "add_suppress_balance": bal, "add_fixed_budget": fixed}
                rows.append(row)
                if summary["safety_pass_all"] and summary["recall_nonregression_pass"]:
                    if best is None or float(summary["mean_net_score"]) > float(best["mean_net_score"]):
                        best = row
    write_csv(REPORTS_DIR / "sw14h8_query_ranker_selection_sweep_val.csv", rows)
    return best or (max(rows, key=lambda r: float(r["mean_net_score"])) if rows else {})


def plot_precision(rows: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    keep = {
        "h7b_prior",
        "h8_query_ranker",
        "h8_blend_h7b_20",
        "h8_blend_h7b_35",
        "h8_blend_h7b_50",
        "h8_best_val",
    }
    for name in sorted(keep):
        sub = sorted([r for r in rows if r["split"] == "val" and r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H8 online query ranker add precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h8_query_ranker_precision.png", dpi=180)
    plt.close()


def score_maps_for_split(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], h8_score: torch.Tensor) -> dict[str, torch.Tensor]:
    h8_norm = sw14hi.normalize_score(h8_score, add_mask(tensors)).nan_to_num(0.0, neginf=0.0)
    h7b = cache["h7b_prior_score"].float()
    return {
        "h7b_prior": h7b,
        "h8_query_ranker": h8_score,
        "h8_blend_h7b_20": 0.20 * h8_norm + 0.80 * h7b,
        "h8_blend_h7b_35": 0.35 * h8_norm + 0.65 * h7b,
        "h8_blend_h7b_50": 0.50 * h8_norm + 0.50 * h7b,
        "h8_blend_h7b_65": 0.65 * h8_norm + 0.35 * h7b,
        "h8_contributor_h7b": cache["h8_contributor_h7b"],
        "h8_any_active_h7b": cache["h8_any_active_h7b"],
    }


def main() -> None:
    args = parse_args()
    cfg = H8RankerConfig(epochs=args.epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim, device=args.device)
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("[h8-ranker] loading tables/base caches", flush=True)
    tables, scores, caches = load_tables_and_base_cache()
    h8_stats: list[dict[str, Any]] = []
    if not args.score_only:
        h8_stats.append(add_h8_features("train", tables["train"]["tensors"], caches["train"]))
    h8_stats.append(add_h8_features("val", tables["val"]["tensors"], caches["val"]))
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    write_json(
        REPORTS_DIR / "sw14h8_query_ranker_config.json",
        {
            "model": "full-train H8 online query/debug residual ranker",
            "target_precision_at_100k": cfg.target_precision_at_100k,
            "feature_names": feature_names(),
            "h8_stats": h8_stats,
            "device": str(device),
            "gpu": gpu,
            "config": asdict(cfg),
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
            "resume_claim": False,
        },
    )
    if args.score_only:
        ckpt = torch.load(CHECKPOINT_DIR / "sw14h8_query_ranker_best.pth", map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt.get("config", {})
        model_hidden = int(ckpt_cfg.get("hidden_dim", cfg.hidden_dim))
        model = H8ResidualRanker(len(feature_names()), model_hidden).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print("[h8-ranker] loaded checkpoint for score-only", flush=True)
    else:
        print("[h8-ranker] training", flush=True)
        model, train_log = train_ranker(tables["train"]["tensors"], caches["train"], cfg, device)
        write_csv(REPORTS_DIR / "sw14h8_query_ranker_training_log.csv", train_log)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_names": feature_names(),
                "config": asdict(cfg),
                "resume_claim": False,
                "gt_used_as_inference_feature": False,
                "uses_eval_debug": False,
            },
            CHECKPOINT_DIR / "sw14h8_query_ranker_best.pth",
        )
    print("[h8-ranker] scoring val", flush=True)
    best_train_100k: dict[str, Any] | None = None
    train_h8: torch.Tensor | None = None
    train_scores: dict[str, torch.Tensor] = {}
    if args.score_train:
        if "train" not in caches:
            raise RuntimeError("--score-train requires train cache")
        train_h8 = score_ranker(model, "train", tables["train"]["tensors"], caches["train"], cfg, device)
        train_scores = score_maps_for_split(tables["train"]["tensors"], caches["train"], train_h8)
        train_topk_rows = topk_rows(tables["train"]["tensors"], train_scores, "train")
        best_train_100k = max((r for r in train_topk_rows if int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    else:
        train_topk_rows = []
    if "train" in tables:
        del tables["train"]
    if "train" in scores:
        del scores["train"]
    if "train" in caches:
        del caches["train"]
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    val_h8 = score_ranker(model, "val", tables["val"]["tensors"], caches["val"], cfg, device)
    score_payload = {"val_sw14h8_query_scores": val_h8}
    if train_h8 is not None:
        score_payload["train_sw14h8_query_scores"] = train_h8
    torch.save(score_payload, ARTIFACTS_DIR / "sw14h8_query_ranker_scores.pt")
    val_scores = score_maps_for_split(tables["val"]["tensors"], caches["val"], val_h8)
    topk = list(train_topk_rows)
    topk.extend(topk_rows(tables["val"]["tensors"], val_scores, "val"))
    write_csv(REPORTS_DIR / "sw14h8_query_ranker_topk_precision.csv", topk)
    best_val_100k = max((r for r in topk if r["split"] == "val" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    h7b_val_100k = next(r for r in topk if r["split"] == "val" and r["score_name"] == "h7b_prior" and int(r["topk"]) == 100000)
    val_scores["h8_best_val"] = val_scores[best_val_100k["score_name"]]
    plot_precision(topk + topk_rows(tables["val"]["tensors"], {"h8_best_val": val_scores["h8_best_val"]}, "val"))
    selection = run_selection(tables, val_scores[best_val_100k["score_name"]], scores["val"]["suppress"])

    precision_gain = float(best_val_100k["precision"]) - float(h7b_val_100k["precision"])
    if float(best_val_100k["precision"]) >= cfg.target_precision_at_100k:
        decision = "SW14H8_1_TARGET_090_REACHED"
    elif precision_gain >= 0.10:
        decision = "SW14H8_2_MATERIAL_QUERY_RANKER_GAIN"
    elif precision_gain > 0.005:
        decision = "SW14H8_3_SMALL_QUERY_RANKER_GAIN"
    else:
        decision = "SW14H8_4_QUERY_RANKER_NOT_ENOUGH"
    final = {
        "decision": decision,
        "target_precision_at_100k": cfg.target_precision_at_100k,
        "target_reached": bool(float(best_val_100k["precision"]) >= cfg.target_precision_at_100k),
        "best_train_top100k": best_train_100k,
        "best_val_top100k": best_val_100k,
        "h7b_val_top100k": h7b_val_100k,
        "precision_gain_over_h7b": precision_gain,
        "selection": selection,
        "h8_stats": h8_stats,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "recommended_next_action": (
            "If H8 remains far below 0.9@100K after full train online-query training, "
            "the current no-GT add evidence is not separable enough; next escalation should change proposal space "
            "or cache richer decoder/query tensors, not just retrain the same table ranker."
        ),
    }
    write_json(REPORTS_DIR / "sw14h8_query_ranker_final_decision.json", final)
    print(f"[h8-ranker] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
