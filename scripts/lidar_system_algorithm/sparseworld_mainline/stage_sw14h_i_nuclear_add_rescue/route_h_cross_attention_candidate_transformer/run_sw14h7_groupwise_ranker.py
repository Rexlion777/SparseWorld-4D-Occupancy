from __future__ import annotations

import csv
import importlib.util
import json
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

SURVIVAL_FEATURES = [
    "h6_replay_raw_delta",
    "h6_replay_protected",
    "h6_replay_f3_occ",
    "h6_replay_final_occ",
    "h6_replay_f3_pruned",
    "h6_replay_frontcap_pruned",
    "h6_low_value_norm",
    "h6_f3_prune_margin",
    "h6_f3_budget_pressure",
    "h6_frontcap_priority_norm",
    "h6_frontcap_prune_margin",
    "h6_frontcap_budget_pressure",
    "h6_raw_to_f3_survival",
    "h6_f3_to_final_survival",
    "h6_replay_diff_teacher_final",
]

RANK_SIGNAL_NAMES = [
    "raw_confidence",
    "raw_margin",
    "camera_view_agreement",
    "neighbor_occ_count_norm",
    "local_density_proxy",
    "b1_add_score_norm",
    "b2_add_score_norm",
    "b3_add_score_norm",
    "h2_score_norm",
    "h6_score_norm",
    "h6b_prior_score",
    "h6_low_value_inv",
    "h6_f3_keep_margin",
    "h6_frontcap_keep_margin",
    "h6_frontcap_priority_inv",
]

GROUP_RANK_FEATURES = [f"h7_group_rank_{name}" for name in RANK_SIGNAL_NAMES]
GROUP_Z_FEATURES = [
    "h7_group_z_h6b_prior_score",
    "h7_group_z_raw_confidence",
    "h7_group_z_h6_low_value_inv",
]
GROUP_SCALAR_FEATURES = [
    "h7_group_add_count_norm",
    "h7_group_strong_ratio",
    "h7_group_front_ratio",
    "h7_group_future_ratio",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_module("sw14hi_h7", SW14HI_SCRIPT)
sw14e = sw14hi.sw14e


@dataclass(frozen=True)
class H7Config:
    seed: int = 127
    epochs: int = 10
    batch_size: int = 393_216
    score_batch_size: int = 1_048_576
    train_max_pos: int = 1_400_000
    train_max_neg: int = 2_200_000
    hard_neg_topk: int = 1_700_000
    hidden_dim: int = 512
    lr: float = 9.0e-4
    target_precision_at_100k: float = 0.90
    device: str = "cuda"


class GroupwiseResidualRanker(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
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


def contiguous_group_slices(keys: torch.Tensor) -> list[tuple[int, int, int]]:
    keys_i = keys.to(torch.int64)
    change = torch.nonzero(keys_i[1:] != keys_i[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([torch.zeros((1,), dtype=torch.long), change])
    ends = torch.cat([change, torch.tensor([len(keys_i)], dtype=torch.long)])
    return [(int(keys_i[s].item()), int(s.item()), int(e.item())) for s, e in zip(starts, ends)]


def normalize_score(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return sw14hi.normalize_score(score.float(), mask).nan_to_num(0.0, neginf=0.0)


def load_h2_h6_scores(n_train: int, n_val: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h2_path = ARTIFACTS_DIR / "sw14h2_true_local_attention_scores.pt"
    h6_path = ARTIFACTS_DIR / "sw14h6_postprocess_survival_scores.pt"
    h2_payload = torch.load(h2_path, map_location="cpu", weights_only=False)
    h6_payload = torch.load(h6_path, map_location="cpu", weights_only=False)
    return (
        h2_payload.get("train_sw14h2_add_scores", torch.full((n_train,), -torch.inf)).float(),
        h2_payload.get("val_sw14h2_add_scores", torch.full((n_val,), -torch.inf)).float(),
        h6_payload.get("train_sw14h6_add_scores", torch.full((n_train,), -torch.inf)).float(),
        h6_payload.get("val_sw14h6_add_scores", torch.full((n_val,), -torch.inf)).float(),
    )


def base_cache(
    tensors: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    survival: dict[str, torch.Tensor],
    h2_score: torch.Tensor,
    h6_score: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cache = sw14hi.derived_cache(tensors, scores)
    mask = add_mask(tensors)
    h2_norm = normalize_score(h2_score, mask)
    h6_norm = normalize_score(h6_score, mask)
    b3 = cache["b3_add_score_norm"].float().nan_to_num(0.0, neginf=0.0)
    low_inv = 1.0 - survival["h6_low_value_norm"].float().nan_to_num(0.0)
    h6b = (0.65 * (0.25 * h2_norm + 0.75 * b3) + 0.20 * h6_norm + 0.15 * normalize_score(low_inv, mask)).float()
    cache.update(
        {
            "h2_score_norm": h2_norm,
            "h6_score_norm": h6_norm,
            "h6b_prior_score": h6b,
            "h6_low_value_inv": low_inv.clamp(0.0, 1.0),
            "h6_f3_keep_margin": (-survival["h6_f3_prune_margin"].float()).nan_to_num(0.0).clamp(-5.0, 5.0),
            "h6_frontcap_keep_margin": (-survival["h6_frontcap_prune_margin"].float()).nan_to_num(0.0).clamp(-5.0, 5.0),
            "h6_frontcap_priority_inv": (1.0 - survival["h6_frontcap_priority_norm"].float().nan_to_num(0.0)).clamp(0.0, 1.0),
        }
    )
    for name in SURVIVAL_FEATURES:
        cache[name] = survival[name].float().nan_to_num(0.0)
    return cache


def group_rank_cache(split: str, tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], force: bool = False) -> dict[str, torch.Tensor]:
    path = ARTIFACTS_DIR / f"sw14h7_groupwise_features_{split}.pt"
    if path.exists() and not force:
        return torch.load(path, map_location="cpu", weights_only=False)
    n = len(tensors["group_key"])
    roi = add_mask(tensors)
    out: dict[str, torch.Tensor] = {name: torch.zeros((n,), dtype=torch.float16) for name in GROUP_RANK_FEATURES + GROUP_Z_FEATURES + GROUP_SCALAR_FEATURES}
    signal_map: dict[str, torch.Tensor] = {}
    for name in RANK_SIGNAL_NAMES:
        signal_map[name] = cache[name].float() if name in cache else tensors[name].float()
    z_sources = {
        "h7_group_z_h6b_prior_score": signal_map["h6b_prior_score"],
        "h7_group_z_raw_confidence": signal_map["raw_confidence"],
        "h7_group_z_h6_low_value_inv": signal_map["h6_low_value_inv"],
    }
    rows: list[dict[str, Any]] = []
    for group_key, start, end in contiguous_group_slices(tensors["group_key"]):
        idx_all = torch.arange(start, end)
        idx = idx_all[roi[start:end]]
        if len(idx) == 0:
            continue
        denom = max(1, len(idx) - 1)
        for sig_name, feature_name in zip(RANK_SIGNAL_NAMES, GROUP_RANK_FEATURES):
            vals = signal_map[sig_name][idx].float().nan_to_num(0.0)
            order = torch.argsort(vals, descending=False)
            ranks = torch.empty_like(vals)
            ranks[order] = torch.arange(len(vals), dtype=torch.float32) / float(denom)
            out[feature_name][idx] = ranks.half()
        for feature_name, vals_all in z_sources.items():
            vals = vals_all[idx].float().nan_to_num(0.0)
            z = (vals - vals.mean()) / vals.std(unbiased=False).clamp_min(1.0e-4)
            out[feature_name][idx] = z.clamp(-5.0, 5.0).half()
        strong_ratio = float(tensors["strong_add_candidate"][idx].float().mean().item())
        front_ratio = float(tensors["front_region"][idx].float().mean().item())
        future_ratio = float(tensors["future_h4h6"][idx].float().mean().item())
        add_count_norm = min(1.0, len(idx) / 10000.0)
        out["h7_group_add_count_norm"][idx] = torch.full((len(idx),), add_count_norm, dtype=torch.float16)
        out["h7_group_strong_ratio"][idx] = torch.full((len(idx),), strong_ratio, dtype=torch.float16)
        out["h7_group_front_ratio"][idx] = torch.full((len(idx),), front_ratio, dtype=torch.float16)
        out["h7_group_future_ratio"][idx] = torch.full((len(idx),), future_ratio, dtype=torch.float16)
        rows.append(
            {
                "split": split,
                "group_key": group_key,
                "add_roi_count": int(len(idx)),
                "strong_ratio": strong_ratio,
                "front_ratio": front_ratio,
                "future_ratio": future_ratio,
            }
        )
        if len(rows) % 80 == 0:
            print(f"[h7] {split} group rank features {len(rows)} groups", flush=True)
    torch.save(out, path)
    write_csv(REPORTS_DIR / f"sw14h7_groupwise_feature_stats_{split}.csv", rows)
    write_json(
        REPORTS_DIR / f"sw14h7_groupwise_feature_summary_{split}.json",
        {
            "split": split,
            "groups": len(rows),
            "feature_names": list(out),
            "uses_gt_as_inference_feature": False,
            "uses_eval_debug": False,
        },
    )
    return out


def feature_names() -> list[str]:
    base = list(sw14hi.H_QUERY_FEATURES)
    extra = [
        "h2_score_norm",
        "h6_score_norm",
        "h6b_prior_score",
        "h6_low_value_inv",
        "h6_f3_keep_margin",
        "h6_frontcap_keep_margin",
        "h6_frontcap_priority_inv",
    ]
    return base + SURVIVAL_FEATURES + extra + GROUP_RANK_FEATURES + GROUP_Z_FEATURES + GROUP_SCALAR_FEATURES


def feature_matrix(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    return sw14hi.query_matrix(tensors, cache, idx, feature_names()).float().nan_to_num(0.0)


def sample_train_indices(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: H7Config) -> torch.Tensor:
    roi = add_mask(tensors)
    label = tensors["GT_occ"].bool()
    pos = torch.nonzero(roi & label, as_tuple=False).flatten()
    neg = torch.nonzero(roi & ~label, as_tuple=False).flatten()
    hard_score = cache["h6b_prior_score"].float()
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


def pairwise_rank_loss(logits: torch.Tensor, y: torch.Tensor, max_pairs: int = 49152) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    return F.softplus(0.60 - pos[:k] + neg[:k]).mean()


def topk_surrogate_loss(logits: torch.Tensor, y: torch.Tensor, prior: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    if len(logits) == 0:
        return logits.new_tensor(0.0)
    k = min(len(logits), max(4096, int(len(logits) * frac)))
    top = torch.unique(torch.cat([torch.topk(logits, k=k).indices, torch.topk(prior, k=k).indices]))
    return F.binary_cross_entropy_with_logits(logits[top], y[top])


def train_ranker(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: H7Config, device: torch.device) -> tuple[GroupwiseResidualRanker, list[dict[str, Any]]]:
    idx = sample_train_indices(tensors, cache, cfg)
    x_cpu = feature_matrix(tensors, cache, idx)
    y_cpu = tensors["GT_occ"][idx].float()
    prior_cpu = cache["h6b_prior_score"][idx].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    front_cpu = tensors["front_region"][idx].float()
    future_cpu = tensors["future_h4h6"][idx].float()
    density_cpu = tensors["local_density_proxy"][idx].float()
    if device.type == "cuda":
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
        prior_cpu = prior_cpu.pin_memory()
        front_cpu = front_cpu.pin_memory()
        future_cpu = future_cpu.pin_memory()
        density_cpu = density_cpu.pin_memory()
    model = GroupwiseResidualRanker(x_cpu.shape[1], cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.2e-4)
    pos_weight = torch.tensor([(len(y_cpu) - y_cpu.sum().item()) / max(1.0, y_cpu.sum().item())], device=device).clamp(1.0, 10.0)
    rows: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + epoch)
        order = torch.randperm(len(idx), generator=gen)
        losses: list[float] = []
        bces: list[float] = []
        ranks: list[float] = []
        topks: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            xb = x_cpu[b].to(device, non_blocking=True)
            yb = y_cpu[b].to(device, non_blocking=True)
            prior = prior_cpu[b].to(device, non_blocking=True)
            front = front_cpu[b].to(device, non_blocking=True)
            future = future_cpu[b].to(device, non_blocking=True)
            density = density_cpu[b].to(device, non_blocking=True)
            logits = model(xb, prior)
            weight = 1.0 + yb * (1.0 * front + 1.15 * future) + (1.0 - yb) * (2.2 * prior + 0.7 * density)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
            rank = pairwise_rank_loss(logits, yb)
            topk = topk_surrogate_loss(logits, yb, prior)
            fp_top_penalty = (torch.sigmoid(logits) * (1.0 - yb) * (1.0 + prior) * (1.0 + density)).mean()
            residual = logits - torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
            loss = bce + 3.0 * rank + 3.5 * topk + 0.75 * fp_top_penalty + 0.008 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            bces.append(float(bce.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
            topks.append(float(topk.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "bce": float(np.mean(bces)),
            "pairwise_rank": float(np.mean(ranks)),
            "topk_surrogate": float(np.mean(topks)),
            "train_rows": int(len(idx)),
            "positive_rate": float(y_cpu.mean().item()),
            "batch_size": cfg.batch_size,
            "device": str(device),
        }
        rows.append(row)
        print(f"[h7] epoch={epoch} loss={row['loss']:.5f}", flush=True)
    return model.eval(), rows


@torch.inference_mode()
def score_ranker(model: GroupwiseResidualRanker, tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: H7Config, device: torch.device) -> torch.Tensor:
    roi = add_mask(tensors)
    idx = torch.nonzero(roi, as_tuple=False).flatten()
    out = torch.full((len(roi),), -torch.inf, dtype=torch.float32)
    for start in range(0, len(idx), cfg.score_batch_size):
        chunk = idx[start : start + cfg.score_batch_size]
        x = feature_matrix(tensors, cache, chunk)
        prior = cache["h6b_prior_score"][chunk].float().clamp(1.0e-4, 1.0 - 1.0e-4)
        if device.type == "cuda":
            x = x.pin_memory()
            prior = prior.pin_memory()
        logits = model(x.to(device, non_blocking=True), prior.to(device, non_blocking=True))
        out[chunk] = torch.sigmoid(logits).detach().cpu()
    return out


def topk_rows(tensors: dict[str, torch.Tensor], scores: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    roi = add_mask(tensors)
    label = tensors["GT_occ"].bool()
    front = tensors["front_region"].bool()
    future = tensors["future_h4h6"].bool()
    idx = torch.nonzero(roi, as_tuple=False).flatten()
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


def plot_precision(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 4.5))
    for name in ["b3_baseline", "h6b_prior", "h7_groupwise_ranker", "h7_blend_prior_25", "h7_blend_prior_50"]:
        sub = sorted([r for r in rows if r["split"] == "val" and r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H7 groupwise add precision")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "sw14h7_groupwise_precision.png", dpi=180)
    plt.close()


def run_selection(tables: dict[str, dict[str, Any]], add_score: torch.Tensor, suppress_score: torch.Tensor) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for sup in [0.02, 0.03, 0.04, 0.05]:
        for bal in [0.25, 0.5, 0.75, 1.0, 1.25]:
            for fixed in [32, 64, 96, 128, 192]:
                cfg = sw14e.Config(max_suppress_ratio=sup, add_suppress_balance=bal, add_fixed_budget=fixed)
                budget = {
                    "name": f"h7_sup{sup}_bal{bal}_fixed{fixed}",
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
    write_csv(REPORTS_DIR / "sw14h7_groupwise_selection_sweep_val.csv", rows)
    return best or (max(rows, key=lambda r: float(r["mean_net_score"])) if rows else {})


def main() -> None:
    cfg = H7Config()
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("[h7] loading tables/scores/features", flush=True)
    tables = sw14hi.load_tables()
    score_payload = sw14hi.load_score_payload(tables)
    train_survival = torch.load(ARTIFACTS_DIR / "sw14h6_postprocess_survival_features_train.pt", map_location="cpu", weights_only=False)
    val_survival = torch.load(ARTIFACTS_DIR / "sw14h6_postprocess_survival_features_val.pt", map_location="cpu", weights_only=False)
    train_h2, val_h2, train_h6, val_h6 = load_h2_h6_scores(len(tables["train"]["tensors"]["GT_occ"]), len(tables["val"]["tensors"]["GT_occ"]))
    train_cache = base_cache(tables["train"]["tensors"], score_payload["train"], train_survival, train_h2, train_h6)
    val_cache = base_cache(tables["val"]["tensors"], score_payload["val"], val_survival, val_h2, val_h6)
    print("[h7] building/reusing groupwise features", flush=True)
    train_group = group_rank_cache("train", tables["train"]["tensors"], train_cache)
    val_group = group_rank_cache("val", tables["val"]["tensors"], val_cache)
    train_cache.update({k: v.float().nan_to_num(0.0) for k, v in train_group.items()})
    val_cache.update({k: v.float().nan_to_num(0.0) for k, v in val_group.items()})
    write_json(
        REPORTS_DIR / "sw14h7_groupwise_config.json",
        {
            "model": "scene/horizon group-normalized baseline-preserving residual ranker",
            "features": feature_names(),
            "rank_signal_names": RANK_SIGNAL_NAMES,
            "target_precision_at_100k": cfg.target_precision_at_100k,
            "gt_used_as_inference_feature": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "config": cfg.__dict__,
        },
    )
    print("[h7] training", flush=True)
    model, train_log = train_ranker(tables["train"]["tensors"], train_cache, cfg, device)
    write_csv(REPORTS_DIR / "sw14h7_groupwise_training_log.csv", train_log)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": feature_names(),
            "rank_signal_names": RANK_SIGNAL_NAMES,
            "resume_claim": False,
            "gt_used_as_inference_feature": False,
            "seed": cfg.seed,
        },
        CHECKPOINT_DIR / "sw14h7_groupwise_ranker_best.pth",
    )
    print("[h7] scoring train/val", flush=True)
    train_h7 = score_ranker(model, tables["train"]["tensors"], train_cache, cfg, device)
    val_h7 = score_ranker(model, tables["val"]["tensors"], val_cache, cfg, device)
    torch.save({"train_sw14h7_add_scores": train_h7, "val_sw14h7_add_scores": val_h7}, ARTIFACTS_DIR / "sw14h7_groupwise_scores.pt")
    train_h7_norm = normalize_score(train_h7, add_mask(tables["train"]["tensors"]))
    val_h7_norm = normalize_score(val_h7, add_mask(tables["val"]["tensors"]))
    train_prior = train_cache["h6b_prior_score"]
    val_prior = val_cache["h6b_prior_score"]
    train_scores = {
        "b3_baseline": score_payload["train"]["b3_add"],
        "h6b_prior": train_prior,
        "h7_groupwise_ranker": train_h7,
        "h7_blend_prior_25": 0.25 * train_h7_norm + 0.75 * train_prior,
        "h7_blend_prior_50": 0.50 * train_h7_norm + 0.50 * train_prior,
    }
    val_scores = {
        "b3_baseline": score_payload["val"]["b3_add"],
        "h6b_prior": val_prior,
        "h7_groupwise_ranker": val_h7,
        "h7_blend_prior_25": 0.25 * val_h7_norm + 0.75 * val_prior,
        "h7_blend_prior_50": 0.50 * val_h7_norm + 0.50 * val_prior,
    }
    topk = topk_rows(tables["train"]["tensors"], train_scores, "train") + topk_rows(tables["val"]["tensors"], val_scores, "val")
    write_csv(REPORTS_DIR / "sw14h7_groupwise_topk_precision.csv", topk)
    plot_precision(topk)
    best_val_100k = max((r for r in topk if r["split"] == "val" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    best_train_100k = max((r for r in topk if r["split"] == "train" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    b3_val_100k = next(r for r in topk if r["split"] == "val" and r["score_name"] == "b3_baseline" and int(r["topk"]) == 100000)
    prior_val_100k = next(r for r in topk if r["split"] == "val" and r["score_name"] == "h6b_prior" and int(r["topk"]) == 100000)
    selection = run_selection(tables, val_scores[best_val_100k["score_name"]], score_payload["val"]["suppress"])
    if float(best_val_100k["precision"]) >= cfg.target_precision_at_100k:
        decision = "SW14H7_1_TARGET_090_REACHED"
    elif float(best_val_100k["precision"]) >= float(prior_val_100k["precision"]) + 0.05:
        decision = "SW14H7_2_GROUPWISE_SIGNAL_MATERIAL"
    elif float(best_val_100k["precision"]) > float(prior_val_100k["precision"]) + 0.005:
        decision = "SW14H7_3_GROUPWISE_SIGNAL_SMALL"
    else:
        decision = "SW14H7_4_NO_GROUPWISE_GAIN"
    final = {
        "decision": decision,
        "target_precision_at_100k": cfg.target_precision_at_100k,
        "target_reached": bool(float(best_val_100k["precision"]) >= cfg.target_precision_at_100k),
        "best_train_top100k": best_train_100k,
        "best_val_top100k": best_val_100k,
        "b3_val_top100k": b3_val_100k,
        "prior_val_top100k": prior_val_100k,
        "selection": selection,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "recommended_next_action": "If H7 remains far below 0.9@100K, build a new online raw-query/logit dump for train 0..99 and val 100..149; groupwise semantic/cache features have been exhausted.",
    }
    write_json(REPORTS_DIR / "sw14h7_groupwise_final_decision.json", final)
    print(f"[h7] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
