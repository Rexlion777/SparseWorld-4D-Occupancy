from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
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
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def load_sw14hi_module():
    spec = importlib.util.spec_from_file_location("sw14hi_h2", SW14HI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SW14HI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sw14hi_h2"] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_sw14hi_module()
sw14e = sw14hi.sw14e


TOKEN_DIM = 20
MAX_TOKENS = 27
GRID_Y = 200
GRID_Z = 16
QUERY_FEATURES = sw14hi.H_QUERY_FEATURES


@dataclass(frozen=True)
class H2Config:
    seed: int = 83
    epochs: int = 5
    batch_size: int = 32_768
    score_batch_size: int = 65_536
    train_max_pos: int = 520_000
    train_max_neg: int = 760_000
    hard_neg_topk: int = 520_000
    d_model: int = 128
    lr: float = 8.0e-4
    residual_scale: float = 1.0
    device: str = "cuda"


class LocalResidualAttentionRanker(nn.Module):
    def __init__(self, query_dim: int, token_dim: int, d_model: int = 128, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.query_encoder = nn.Sequential(nn.Linear(query_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.token_encoder = nn.Sequential(nn.Linear(token_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.type_embedding = nn.Embedding(8, d_model)
        self.ffn1 = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Dropout(0.05), nn.Linear(d_model * 2, d_model))
        self.ffn2 = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Dropout(0.05), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.residual_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
        last = self.residual_head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _attend(self, q: torch.Tensor, tok: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = (tok * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(tok.shape[-1]))
        logits = logits.masked_fill(~valid.bool(), -1.0e4)
        weights = torch.softmax(logits, dim=1)
        ctx = (weights.unsqueeze(-1) * tok).sum(dim=1)
        return ctx, weights

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        token_valid: torch.Tensor,
        prior_score: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.query_encoder(query)
        tok = self.token_encoder(tokens) + self.type_embedding(token_types.clamp(0, 7))
        ctx, weights = self._attend(q, tok, token_valid)
        q = self.norm1(q + ctx)
        q = self.norm2(q + self.ffn1(q))
        ctx, weights = self._attend(q, tok, token_valid)
        q = self.norm3(q + ctx)
        q = self.norm4(q + self.ffn2(q))
        residual = self.residual_head(q).squeeze(-1) * self.residual_scale
        prior = torch.logit(prior_score.float().clamp(1.0e-4, 1.0 - 1.0e-4))
        return prior + residual, weights if return_attention else None


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
        writer.writerows([normalize(r) for r in rows])


def linear_coord(tensors: dict[str, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
    return (
        tensors["voxel_x"][rows].long() * GRID_Y * GRID_Z
        + tensors["voxel_y"][rows].long() * GRID_Z
        + tensors["voxel_z"][rows].long()
    )


def offset_table() -> tuple[torch.Tensor, torch.Tensor]:
    offsets: list[tuple[int, int, int]] = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                offsets.append((dx, dy, dz))
    offsets = sorted(offsets, key=lambda p: (abs(p[0]) + abs(p[1]) + abs(p[2]), abs(p[0]), abs(p[1]), abs(p[2])))
    off = torch.tensor(offsets, dtype=torch.long)
    lin = off[:, 0] * GRID_Y * GRID_Z + off[:, 1] * GRID_Z + off[:, 2]
    return off, lin


def build_group_lookup(tensors: dict[str, torch.Tensor]) -> dict[int, dict[str, torch.Tensor]]:
    keys = tensors["group_key"].to(torch.int32)
    out: dict[int, dict[str, torch.Tensor]] = {}
    for key in torch.unique(keys):
        group = torch.nonzero(keys == key, as_tuple=False).flatten()
        lin = linear_coord(tensors, group).to(torch.int64)
        order = torch.argsort(lin)
        out[int(key.item())] = {
            "lins": lin[order].contiguous(),
            "rows": group[order].contiguous(),
        }
    return out


def token_type(tensors: dict[str, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
    t = torch.zeros((len(rows),), dtype=torch.long)
    t[tensors["raw_occ"][rows].bool()] = 1
    t[tensors["teacher_final_occ"][rows].bool()] = 2
    t[tensors["was_pruned_by_F3"][rows].bool()] = 3
    t[tensors["was_pruned_by_FrontCap"][rows].bool()] = 4
    t[tensors["native_final_occ"][rows].bool()] = 5
    return t


def local_tokens_for_indices(
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    lookup: dict[int, dict[str, torch.Tensor]],
    idx: torch.Tensor,
    offsets: torch.Tensor,
    offset_lins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(idx)
    tokens = torch.zeros((n, MAX_TOKENS, TOKEN_DIM), dtype=torch.float32)
    token_types = torch.zeros((n, MAX_TOKENS), dtype=torch.long)
    valid = torch.zeros((n, MAX_TOKENS), dtype=torch.bool)
    group_keys = tensors["group_key"][idx].to(torch.int32)
    for key in torch.unique(group_keys):
        sel = torch.nonzero(group_keys == key, as_tuple=False).flatten()
        cand = idx[sel]
        group = lookup[int(key.item())]
        cand_lin = linear_coord(tensors, cand).to(torch.int64)
        targets = cand_lin[:, None] + offset_lins[None, :]
        flat_targets = targets.reshape(-1)
        pos = torch.searchsorted(group["lins"], flat_targets)
        in_range = pos < len(group["lins"])
        safe_pos = pos.clamp(max=max(0, len(group["lins"]) - 1))
        matched_lin = group["lins"][safe_pos]
        matched_rows = group["rows"][safe_pos]
        valid_flat = in_range & (matched_lin == flat_targets)
        matched_rows = matched_rows.reshape(len(cand), MAX_TOKENS)
        valid_g = valid_flat.reshape(len(cand), MAX_TOKENS)
        # Prevent linear-coordinate wraparound by checking the actual xyz coordinate after lookup.
        tx = tensors["voxel_x"][cand].long()[:, None] + offsets[:, 0][None, :]
        ty = tensors["voxel_y"][cand].long()[:, None] + offsets[:, 1][None, :]
        tz = tensors["voxel_z"][cand].long()[:, None] + offsets[:, 2][None, :]
        valid_g &= tensors["voxel_x"][matched_rows].long() == tx
        valid_g &= tensors["voxel_y"][matched_rows].long() == ty
        valid_g &= tensors["voxel_z"][matched_rows].long() == tz
        rows = matched_rows
        rows_safe = rows.clamp(min=0)
        rel = offsets.float()
        token = torch.stack(
            [
                rel[:, 0].expand(len(cand), -1) / 1.0,
                rel[:, 1].expand(len(cand), -1) / 1.0,
                rel[:, 2].expand(len(cand), -1) / 1.0,
                (rel.square().sum(dim=1).sqrt().expand(len(cand), -1) / 1.732),
                tensors["raw_confidence"][rows_safe].float(),
                tensors["raw_margin"][rows_safe].float(),
                tensors["raw_occ"][rows_safe].float(),
                tensors["teacher_final_occ"][rows_safe].float(),
                tensors["was_pruned_by_F3"][rows_safe].float(),
                tensors["was_pruned_by_FrontCap"][rows_safe].float(),
                tensors["raw_but_final_empty"][rows_safe].float(),
                tensors["local_density_proxy"][rows_safe].float(),
                tensors["neighbor_occ_count_norm"][rows_safe].float(),
                tensors["temporal_consistency"][rows_safe].float(),
                tensors["camera_view_agreement"][rows_safe].float(),
                tensors["front_region"][rows_safe].float(),
                tensors["future_h4h6"][rows_safe].float(),
                cache["b1_add_score_norm"][rows_safe].float(),
                cache["b2_add_score_norm"][rows_safe].float(),
                cache["b3_add_score_norm"][rows_safe].float(),
            ],
            dim=2,
        )
        token = token.masked_fill(~valid_g.unsqueeze(-1), 0.0)
        tokens[sel] = token
        token_types[sel] = token_type(tensors, rows_safe.reshape(-1)).reshape(len(cand), MAX_TOKENS)
        valid[sel] = valid_g
    return tokens, token_types, valid


def query_matrix(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    return sw14hi.query_matrix(tensors, cache, idx, QUERY_FEATURES)


def add_mask(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()


def sample_train_indices(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: H2Config) -> torch.Tensor:
    mask = add_mask(tensors)
    y = tensors["GT_occ"].bool()
    pos = torch.nonzero(mask & y, as_tuple=False).flatten()
    neg = torch.nonzero(mask & ~y, as_tuple=False).flatten()
    hard_score = cache["b3_add_score_norm"]
    hard_neg = neg[torch.argsort(hard_score[neg], descending=True)[: min(cfg.hard_neg_topk, len(neg))]]
    gen = torch.Generator().manual_seed(cfg.seed)
    if len(pos) > cfg.train_max_pos:
        pos = pos[torch.randperm(len(pos), generator=gen)[: cfg.train_max_pos]]
    random_neg_budget = max(0, cfg.train_max_neg - len(hard_neg))
    random_neg = neg[torch.randperm(len(neg), generator=gen)[: min(random_neg_budget, len(neg))]]
    neg_sel = torch.unique(torch.cat([hard_neg, random_neg]))
    if len(neg_sel) > cfg.train_max_neg:
        neg_sel = neg_sel[torch.randperm(len(neg_sel), generator=gen)[: cfg.train_max_neg]]
    idx = torch.cat([pos, neg_sel])
    return idx[torch.randperm(len(idx), generator=gen)]


def pairwise_rank_loss(logits: torch.Tensor, y: torch.Tensor, max_pairs: int = 16384) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    return F.softplus(0.45 - pos[:k] + neg[:k]).mean()


def topk_surrogate_loss(logits: torch.Tensor, y: torch.Tensor, frac: float = 0.35) -> torch.Tensor:
    if len(logits) == 0:
        return logits.new_tensor(0.0)
    k = min(len(logits), max(256, int(len(logits) * frac)))
    top = torch.topk(logits, k=k).indices
    return F.binary_cross_entropy_with_logits(logits[top], y[top])


def train_h2(
    train_tensors: dict[str, torch.Tensor],
    train_cache: dict[str, torch.Tensor],
    train_lookup: dict[int, dict[str, torch.Tensor]],
    cfg: H2Config,
    device: torch.device,
) -> tuple[LocalResidualAttentionRanker, list[dict[str, Any]]]:
    idx = sample_train_indices(train_tensors, train_cache, cfg)
    offsets, offset_lins = offset_table()
    x_cpu = query_matrix(train_tensors, train_cache, idx).float()
    y_cpu = train_tensors["GT_occ"][idx].float()
    prior_cpu = train_cache["b3_add_score_norm"][idx].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    density_cpu = train_tensors["local_density_proxy"][idx].float()
    front_cpu = train_tensors["front_region"][idx].float()
    future_cpu = train_tensors["future_h4h6"][idx].float()
    if device.type == "cuda":
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
        prior_cpu = prior_cpu.pin_memory()
        density_cpu = density_cpu.pin_memory()
        front_cpu = front_cpu.pin_memory()
        future_cpu = future_cpu.pin_memory()
    model = LocalResidualAttentionRanker(x_cpu.shape[1], TOKEN_DIM, cfg.d_model, cfg.residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=2.0e-4)
    rows: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 100 + epoch)
        order = torch.randperm(len(idx), generator=gen)
        losses: list[float] = []
        bces: list[float] = []
        ranks: list[float] = []
        topks: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            global_idx = idx[b]
            tok, typ, valid = local_tokens_for_indices(train_tensors, train_cache, train_lookup, global_idx, offsets, offset_lins)
            if device.type == "cuda":
                tok = tok.pin_memory()
                typ = typ.pin_memory()
                valid = valid.pin_memory()
            xb = x_cpu[b].to(device, non_blocking=True)
            tb = tok.to(device, non_blocking=True)
            tyb = typ.to(device, non_blocking=True)
            vb = valid.to(device, non_blocking=True)
            yb = y_cpu[b].to(device, non_blocking=True)
            prior = prior_cpu[b].to(device, non_blocking=True)
            density = density_cpu[b].to(device, non_blocking=True)
            weight = 1.0 + yb * (0.8 * front_cpu[b].to(device, non_blocking=True) + 1.0 * future_cpu[b].to(device, non_blocking=True))
            weight = weight + (1.0 - yb) * (1.0 + 1.5 * prior + 0.6 * density)
            logits, _ = model(xb, tb, tyb, vb, prior)
            bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
            rank = pairwise_rank_loss(logits, yb)
            topk = topk_surrogate_loss(logits, yb)
            fp_top_penalty = (torch.sigmoid(logits) * (1.0 - yb) * prior).mean()
            # Keep the residual model close to B3 unless local evidence improves the ranking.
            prior_logits = torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
            residual_l2 = (logits - prior_logits).square().mean()
            loss = bce + 2.0 * rank + 2.5 * topk + 0.4 * fp_top_penalty + 0.015 * residual_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            bces.append(float(bce.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
            topks.append(float(topk.detach().cpu().item()))
        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "bce": float(np.mean(bces)),
                "pairwise_rank": float(np.mean(ranks)),
                "topk_surrogate": float(np.mean(topks)),
                "train_rows": int(len(idx)),
                "positive_rate": float(y_cpu.mean().item()),
            }
        )
    return model.eval(), rows


@torch.inference_mode()
def score_h2(
    split: str,
    model: LocalResidualAttentionRanker,
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    lookup: dict[int, dict[str, torch.Tensor]],
    cfg: H2Config,
    device: torch.device,
) -> torch.Tensor:
    mask = add_mask(tensors)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    out = torch.full((len(mask),), -torch.inf, dtype=torch.float32)
    offsets, offset_lins = offset_table()
    for start in range(0, len(idx), cfg.score_batch_size):
        chunk = idx[start : start + cfg.score_batch_size]
        x = query_matrix(tensors, cache, chunk).float()
        tok, typ, valid = local_tokens_for_indices(tensors, cache, lookup, chunk, offsets, offset_lins)
        prior = cache["b3_add_score_norm"][chunk].float().clamp(1.0e-4, 1.0 - 1.0e-4)
        if device.type == "cuda":
            x = x.pin_memory()
            tok = tok.pin_memory()
            typ = typ.pin_memory()
            valid = valid.pin_memory()
            prior = prior.pin_memory()
        logits, _ = model(
            x.to(device, non_blocking=True),
            tok.to(device, non_blocking=True),
            typ.to(device, non_blocking=True),
            valid.to(device, non_blocking=True),
            prior.to(device, non_blocking=True),
        )
        out[chunk] = torch.sigmoid(logits).detach().cpu()
    return out


def topk_rows(tensors: dict[str, torch.Tensor], scores: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    mask = add_mask(tensors)
    label = tensors["GT_occ"].bool()
    front = tensors["front_region"].bool()
    future = tensors["future_h4h6"].bool()
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    rows: list[dict[str, Any]] = []
    for name, score in scores.items():
        order = idx[torch.argsort(score[idx].float(), descending=True)]
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000]:
            if len(order) < k:
                continue
            top = order[:k]
            rows.append(
                {
                    "split": split,
                    "score_name": name,
                    "topk": k,
                    "precision": float(label[top].float().mean().item()),
                    "positive_count": int(label[top].sum().item()),
                    "front_precision": float(label[top][front[top]].float().mean().item()) if bool(front[top].any().item()) else 0.0,
                    "future_precision": float(label[top][future[top]].float().mean().item()) if bool(future[top].any().item()) else 0.0,
                    "front_count": int(front[top].sum().item()),
                    "future_count": int(future[top].sum().item()),
                }
            )
    return rows


def local_token_stats(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], lookup: dict[int, dict[str, torch.Tensor]], split: str) -> dict[str, Any]:
    idx_all = torch.nonzero(add_mask(tensors), as_tuple=False).flatten()
    # Use a deterministic prefix and high-score slice so stats cover normal and top-ranked candidates.
    high = idx_all[torch.argsort(cache["b3_add_score_norm"][idx_all], descending=True)[: min(150_000, len(idx_all))]]
    idx = torch.unique(torch.cat([idx_all[: min(100_000, len(idx_all))], high]))
    offsets, offset_lins = offset_table()
    counts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    type_sums = torch.zeros((8,), dtype=torch.float64)
    for start in range(0, len(idx), 65_536):
        chunk = idx[start : start + 65_536]
        _, typ, valid = local_tokens_for_indices(tensors, cache, lookup, chunk, offsets, offset_lins)
        counts.append(valid.float().sum(dim=1))
        labels.append(tensors["GT_occ"][chunk].bool())
        for t in range(8):
            type_sums[t] += ((typ == t) & valid).sum().item()
    count = torch.cat(counts)
    label = torch.cat(labels)
    return {
        "split": split,
        "sampled_candidates": int(len(idx)),
        "avg_real_local_tokens": float(count.mean().item()),
        "p05_real_local_tokens": float(torch.quantile(count, 0.05).item()),
        "p95_real_local_tokens": float(torch.quantile(count, 0.95).item()),
        "zero_real_local_tokens": int((count == 0).sum().item()),
        "true_avg_real_local_tokens": float(count[label].mean().item()) if bool(label.any().item()) else 0.0,
        "false_avg_real_local_tokens": float(count[~label].mean().item()) if bool((~label).any().item()) else 0.0,
        **{f"token_type_{i}_count": int(type_sums[i].item()) for i in range(8)},
    }


def plot_precision(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 4.5))
    for name in ["b3_baseline", "h2_local_attention", "h2_blend_b3_25", "h2_blend_b3_50"]:
        sub = sorted([r for r in rows if r["split"] == "val" and r["score_name"] == name], key=lambda r: int(r["topk"]))
        if not sub:
            continue
        plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H2 true local attention add precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h2_true_local_attention_precision.png", dpi=180)
    plt.close()


def run_selection(tables: dict[str, dict[str, Any]], add_score: torch.Tensor, suppress_score: torch.Tensor) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for sup in [0.02, 0.03]:
        for bal in [0.25, 0.5, 0.75]:
            for fixed in [16, 32, 64]:
                cfg = sw14e.Config(max_suppress_ratio=sup, add_suppress_balance=bal, add_fixed_budget=fixed)
                budget = {
                    "name": f"h2_sup{sup}_bal{bal}_fixed{fixed}",
                    "suppress_topk_ratio": 1.0,
                    "add_topk_ratio": 1.0,
                    "add_strength": "strong_medium",
                }
                _, summary = sw14e.evaluate_selection("val", tables["val"], add_score, suppress_score, budget, cfg)
                row = {**summary, "max_suppress_ratio": sup, "add_suppress_balance": bal, "add_fixed_budget": fixed}
                rows.append(row)
                if summary["safety_pass_all"] and summary["recall_nonregression_pass"]:
                    if best is None or summary["mean_net_score"] > best["mean_net_score"]:
                        best = row
    write_csv(REPORTS_DIR / "sw14h2_true_local_attention_selection_sweep_val.csv", rows)
    return best or (max(rows, key=lambda r: r["mean_net_score"]) if rows else {})


def main() -> None:
    cfg = H2Config()
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")

    tables = sw14hi.load_tables()
    score_payload = sw14hi.load_score_payload(tables)
    train_cache = sw14hi.derived_cache(tables["train"]["tensors"], score_payload["train"])
    val_cache = sw14hi.derived_cache(tables["val"]["tensors"], score_payload["val"])
    print("[h2] building train group lookup", flush=True)
    train_lookup = build_group_lookup(tables["train"]["tensors"])
    print("[h2] building val group lookup", flush=True)
    val_lookup = build_group_lookup(tables["val"]["tensors"])

    stats = [
        local_token_stats(tables["train"]["tensors"], train_cache, train_lookup, "train"),
        local_token_stats(tables["val"]["tensors"], val_cache, val_lookup, "val"),
    ]
    write_csv(REPORTS_DIR / "sw14h2_true_local_token_stats.csv", stats)
    write_json(
        REPORTS_DIR / "sw14h2_true_local_attention_config.json",
        {
            "model": "baseline-preserving local cross-attention residual ranker",
            "query_features": QUERY_FEATURES,
            "token_dim": TOKEN_DIM,
            "max_tokens": MAX_TOKENS,
            "offset_window": "3x3x3 real candidate-table neighbor lookup by sample/horizon/voxel xyz",
            "gt_used_as_inference_feature": False,
            "target": "val add precision@100K >= 0.9",
            "config": cfg.__dict__,
        },
    )

    print("[h2] training", flush=True)
    model, train_rows = train_h2(tables["train"]["tensors"], train_cache, train_lookup, cfg, device)
    write_csv(REPORTS_DIR / "sw14h2_true_local_attention_training_log.csv", train_rows)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "query_features": QUERY_FEATURES,
            "token_dim": TOKEN_DIM,
            "max_tokens": MAX_TOKENS,
            "seed": cfg.seed,
            "resume_claim": False,
        },
        CHECKPOINT_DIR / "sw14h2_true_local_attention_best.pth",
    )

    print("[h2] scoring train", flush=True)
    train_h2_scores = score_h2("train", model, tables["train"]["tensors"], train_cache, train_lookup, cfg, device)
    print("[h2] scoring val", flush=True)
    val_h2_scores = score_h2("val", model, tables["val"]["tensors"], val_cache, val_lookup, cfg, device)
    torch.save({"train_sw14h2_add_scores": train_h2_scores, "val_sw14h2_add_scores": val_h2_scores}, ARTIFACTS_DIR / "sw14h2_true_local_attention_scores.pt")

    train_b3 = score_payload["train"]["b3_add"]
    val_b3 = score_payload["val"]["b3_add"]
    train_h2_norm = sw14hi.normalize_score(train_h2_scores, add_mask(tables["train"]["tensors"]))
    val_h2_norm = sw14hi.normalize_score(val_h2_scores, add_mask(tables["val"]["tensors"]))
    train_b3_norm = train_cache["b3_add_score_norm"]
    val_b3_norm = val_cache["b3_add_score_norm"]
    train_scores = {
        "b3_baseline": train_b3,
        "h2_local_attention": train_h2_scores,
        "h2_blend_b3_25": 0.25 * train_h2_norm + 0.75 * train_b3_norm,
        "h2_blend_b3_50": 0.50 * train_h2_norm + 0.50 * train_b3_norm,
    }
    val_scores = {
        "b3_baseline": val_b3,
        "h2_local_attention": val_h2_scores,
        "h2_blend_b3_25": 0.25 * val_h2_norm + 0.75 * val_b3_norm,
        "h2_blend_b3_50": 0.50 * val_h2_norm + 0.50 * val_b3_norm,
    }
    topk = topk_rows(tables["train"]["tensors"], train_scores, "train") + topk_rows(tables["val"]["tensors"], val_scores, "val")
    write_csv(REPORTS_DIR / "sw14h2_true_local_attention_topk_precision.csv", topk)
    plot_precision(topk)
    best_val_100k = max((r for r in topk if r["split"] == "val" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    best_train_100k = max((r for r in topk if r["split"] == "train" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    b3_val_100k = next(r for r in topk if r["split"] == "val" and r["score_name"] == "b3_baseline" and int(r["topk"]) == 100000)
    selection = run_selection(tables, val_scores[best_val_100k["score_name"]], score_payload["val"]["suppress"])
    write_json(REPORTS_DIR / "sw14h2_true_local_attention_selection_summary.json", selection)
    target = 0.9
    if float(best_val_100k["precision"]) >= target:
        decision = "SW14H2_1_TARGET_090_REACHED"
    elif float(best_train_100k["precision"]) < float(next(r for r in topk if r["split"] == "train" and r["score_name"] == "b3_baseline" and int(r["topk"]) == 100000)["precision"]) + 0.05:
        decision = "SW14H2_2_TRAIN_GATE_FAILED"
    elif float(best_val_100k["precision"]) > float(b3_val_100k["precision"]) + 0.02:
        decision = "SW14H2_3_VAL_IMPROVED_BELOW_090"
    else:
        decision = "SW14H2_4_NO_VAL_PRECISION_GAIN"
    final = {
        "decision": decision,
        "target_precision_at_100k": target,
        "target_reached": bool(float(best_val_100k["precision"]) >= target),
        "best_train_top100k": best_train_100k,
        "best_val_top100k": best_val_100k,
        "b3_val_top100k": b3_val_100k,
        "selection": selection,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "gt_used_as_inference_feature": False,
        "recommended_next_action": "If target is not reached, expand evidence beyond candidate table proxies: true raw logits/F3/FrontCap survival cache and temporal multi-frame neighborhoods.",
    }
    write_json(REPORTS_DIR / "sw14h2_true_local_attention_final_decision.json", final)
    print(f"[h2] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
