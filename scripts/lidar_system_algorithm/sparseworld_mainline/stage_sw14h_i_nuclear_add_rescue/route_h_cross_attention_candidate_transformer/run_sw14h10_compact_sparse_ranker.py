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
SW14H9_DUMP_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h9_sparse_contributor_dump.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
SW14E_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14e_three_route_rescue"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_module("sw14hi_h10_compact", SW14HI_SCRIPT)
h9dump = load_module("sw14h9_dump_for_h10", SW14H9_DUMP_SCRIPT)
sw14e = sw14hi.sw14e

TABLE_FEATURES = [
    "raw_confidence",
    "raw_margin",
    "raw_entropy",
    "raw_occ",
    "native_final_occ",
    "after_F3_occ",
    "was_pruned_by_F3",
    "was_pruned_by_FrontCap",
    "raw_but_final_empty",
    "neighbor_occ_count_norm",
    "local_density_proxy",
    "temporal_consistency",
    "camera_view_agreement",
    "front_local_density_context",
    "x_norm",
    "abs_y_norm",
    "z_norm",
    "range_norm",
    "horizon_norm",
    "sector_norm",
    "distance_bin_norm",
    "front_region",
    "future_h4h6",
    "strong_add_candidate",
    "medium_add_candidate",
]
H8_FEATURES = [
    "h8_dense_top1_score",
    "h8_dense_top1_margin",
    "h8_geometric_active",
    "h8_semantic_active",
    "h8_contributor_count",
    "h8_occ_pred_nonempty",
]
H9_FEATURES = list(h9dump.H9_FEATURE_NAMES)
SCORE_FEATURES = [
    "b1_add_norm",
    "b2_add_norm",
    "b3_add_norm",
    "h2_norm",
    "h6_norm",
    "h7_norm",
    "h7b_lean",
    "b2_b3_agreement",
    "b1_b3_agreement",
]
GROUP_RANK_SOURCES = [
    "h7b_lean",
    "b3_add_norm",
    "h8_contributor_count_norm",
    "h8_geometric_active_norm",
    "h8_occ_pred_nonempty_norm",
    "h9_geo_count_norm",
    "h9_geo_score_pass_count_norm",
    "h9_geo_query_count_norm",
    "h9_geo_top1_mean_norm",
    "h9_gate_fraction_norm",
]
INTERACTION_FEATURES = [
    "h8_any_active",
    "h8_any_h7b",
    "h9_geo_count_h7b",
    "h9_score_pass_h7b",
    "h9_gate_fraction_h7b",
    "h9_geo_count_h8_any",
    "h9_score_pass_h8_any",
    "front_h7b",
    "future_h7b",
    "front_h9_geo",
    "future_h9_geo",
    "near_gate_h7b",
]


@dataclass(frozen=True)
class H10Config:
    seed: int = 401
    epochs: int = 14
    batch_size: int = 262_144
    score_batch_size: int = 1_048_576
    hidden_dim: int = 768
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    target_precision_at_100k: float = 0.90
    device: str = "cuda"


class CompactRanker(nn.Module):
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
        return torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4)) + self.net(x).squeeze(-1)


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
    parser = argparse.ArgumentParser(description="SW14H10 compact H8/H9 sparse add ranker")
    parser.add_argument("--epochs", type=int, default=H10Config.epochs)
    parser.add_argument("--batch-size", type=int, default=H10Config.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=H10Config.hidden_dim)
    parser.add_argument("--device", default=H10Config.device)
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


def dump_path(kind: str, split: str) -> Path:
    if kind == "h8":
        return ARTIFACTS_DIR / ("sw14h8_online_query_score_dump_train_0_99.pt" if split == "train" else "sw14h8_online_query_score_dump_val_100_149.pt")
    return ARTIFACTS_DIR / ("sw14h9_sparse_contributor_dump_train_0_99.pt" if split == "train" else "sw14h9_sparse_contributor_dump_val_100_149.pt")


def norm(values: torch.Tensor) -> torch.Tensor:
    vals = values.float().nan_to_num(0.0, neginf=0.0, posinf=0.0)
    return ((vals - vals.min()) / (vals.max() - vals.min() + 1.0e-6)).clamp(0.0, 1.0) if vals.numel() and vals.max() > vals.min() else torch.zeros_like(vals)


def score_norm_full(score: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return norm(score[rows].float())


def load_score_vector(path: Path, key: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)[key].float()


def add_group_ranks(features: dict[str, torch.Tensor], group_key: torch.Tensor) -> list[str]:
    added: list[str] = []
    for name in GROUP_RANK_SOURCES:
        vals_all = features[name].float()
        rank = torch.zeros_like(vals_all, dtype=torch.float16)
        z_out = torch.zeros_like(vals_all, dtype=torch.float16)
        for key in torch.unique(group_key).tolist():
            idx = torch.nonzero(group_key == int(key), as_tuple=False).flatten()
            if len(idx) <= 1:
                continue
            vals = vals_all[idx].float().nan_to_num(0.0)
            order = torch.argsort(vals, descending=False)
            ranks = torch.empty((len(idx),), dtype=torch.float32)
            ranks[order] = torch.arange(len(idx), dtype=torch.float32) / float(max(1, len(idx) - 1))
            rank[idx] = ranks.half()
            z = (vals - vals.mean()) / vals.std(unbiased=False).clamp_min(1.0e-4)
            z_out[idx] = z.clamp(-5.0, 5.0).half()
        features[f"{name}_group_rank"] = rank
        features[f"{name}_group_z"] = z_out
        added.extend([f"{name}_group_rank", f"{name}_group_z"])
    return added


def build_compact_split(split: str) -> dict[str, Any]:
    print(f"[h10] build compact split={split}", flush=True)
    table = sw14e.load_tensor_table(sw14e.table_path(split))
    tensors = table["tensors"]
    h8_payload = torch.load(dump_path("h8", split), map_location="cpu", weights_only=False)
    h9_payload = torch.load(dump_path("h9", split), map_location="cpu", weights_only=False)
    h8f = h8_payload["features"]
    h9f = h9_payload["features"]
    rows = h9f["row_index"].long()
    if not torch.equal(rows, h8f["row_index"].long()):
        order = torch.argsort(h8f["row_index"].long())
        h8_rows_sorted = h8f["row_index"].long()[order]
        if not torch.equal(rows, h8_rows_sorted):
            raise RuntimeError(f"{split} H8/H9 row_index mismatch")
        h8f = {k: (v[order] if isinstance(v, torch.Tensor) and len(v) == len(order) else v) for k, v in h8f.items()}
    y = tensors["GT_occ"][rows].float()
    group_key = tensors["group_key"][rows].long()
    features: dict[str, torch.Tensor] = {}
    for name in TABLE_FEATURES:
        features[name] = tensors[name][rows].float().nan_to_num(0.0).half()
    for name in H8_FEATURES:
        raw = h8f[name].float().nan_to_num(0.0)
        features[name] = raw.half()
        features[f"{name}_norm"] = norm(raw).half()
    for name in H9_FEATURES:
        raw = h9f[name].float().nan_to_num(0.0)
        features[name] = raw.half()
        features[f"{name}_norm"] = norm(raw).half()

    base = torch.load(SW14E_ARTIFACTS / "sw14d_b_batched_scores.pt", map_location="cpu", weights_only=False)[split]
    b2_path = SW14E_ARTIFACTS / "sw14d_b2_add_recall_scores.pt"
    b3_path = SW14E_ARTIFACTS / "sw14d_b3_add_precision100k_scores.pt"
    b2 = torch.load(b2_path, map_location="cpu", weights_only=False).get(f"{split}_add_b2_scores", base["add_scores"]).float()
    b3 = torch.load(b3_path, map_location="cpu", weights_only=False).get(f"{split}_b3_add_scores", base["add_scores"]).float()
    h2 = load_score_vector(ARTIFACTS_DIR / "sw14h2_true_local_attention_scores.pt", f"{split}_sw14h2_add_scores")
    h6 = load_score_vector(ARTIFACTS_DIR / "sw14h6_postprocess_survival_scores.pt", f"{split}_sw14h6_add_scores")
    h7 = load_score_vector(ARTIFACTS_DIR / "sw14h7_groupwise_scores.pt", f"{split}_sw14h7_add_scores")
    features["b1_add_norm"] = score_norm_full(base["add_scores"].float(), rows).half()
    features["b2_add_norm"] = score_norm_full(b2, rows).half()
    features["b3_add_norm"] = score_norm_full(b3, rows).half()
    features["h2_norm"] = score_norm_full(h2, rows).half()
    features["h6_norm"] = score_norm_full(h6, rows).half()
    features["h7_norm"] = score_norm_full(h7, rows).half()
    h6b_lean = (0.65 * (0.25 * features["h2_norm"].float() + 0.75 * features["b3_add_norm"].float()) + 0.20 * features["h6_norm"].float()).clamp(0.0, 1.0)
    h7b = (0.80 * h6b_lean + 0.12 * features["h7_norm"].float() + 0.08 * features["h2_norm"].float()).clamp(0.0, 1.0)
    features["h7b_lean"] = h7b.half()
    features["b2_b3_agreement"] = (features["b2_add_norm"].float() * features["b3_add_norm"].float()).half()
    features["b1_b3_agreement"] = (features["b1_add_norm"].float() * features["b3_add_norm"].float()).half()
    features["h8_any_active"] = torch.maximum(
        torch.maximum(features["h8_geometric_active_norm"].float(), features["h8_semantic_active_norm"].float()),
        features["h8_occ_pred_nonempty_norm"].float(),
    ).half()
    features["h8_any_h7b"] = (features["h8_any_active"].float() * h7b).half()
    features["h9_geo_count_h7b"] = (features["h9_geo_count_norm"].float() * h7b).half()
    features["h9_score_pass_h7b"] = (features["h9_geo_score_pass_count_norm"].float() * h7b).half()
    features["h9_gate_fraction_h7b"] = (features["h9_gate_fraction_norm"].float() * h7b).half()
    features["h9_geo_count_h8_any"] = (features["h9_geo_count_norm"].float() * features["h8_any_active"].float()).half()
    features["h9_score_pass_h8_any"] = (features["h9_geo_score_pass_count_norm"].float() * features["h8_any_active"].float()).half()
    features["front_h7b"] = (features["front_region"].float() * h7b).half()
    features["future_h7b"] = (features["future_h4h6"].float() * h7b).half()
    features["front_h9_geo"] = (features["front_region"].float() * features["h9_geo_count_norm"].float()).half()
    features["future_h9_geo"] = (features["future_h4h6"].float() * features["h9_geo_count_norm"].float()).half()
    features["near_gate_h7b"] = (
        (features["h9_geo_score_over_thr_max"].float() > -0.02).float()
        * (features["h9_geo_dist_margin_max"].float() > -0.25).float()
        * h7b
    ).half()
    group_features = add_group_ranks(features, group_key)
    feature_names = TABLE_FEATURES + [f"{name}_norm" for name in H8_FEATURES] + [f"{name}_norm" for name in H9_FEATURES] + SCORE_FEATURES + INTERACTION_FEATURES + group_features
    return {
        "split": split,
        "rows": rows,
        "group_key": group_key,
        "features": features,
        "feature_names": feature_names,
        "label": y,
        "prior": features["h7b_lean"].float(),
        "table": table,
        "suppress_scores": base["suppress_scores"].float(),
        "stats": {
            "split": split,
            "rows": int(len(rows)),
            "positive_rows": int(y.sum().item()),
            "positive_rate": float(y.mean().item()),
            "lean_h7b_p100k": precision_at_k_compact(y, features["h7b_lean"].float(), 100000),
            "feature_count": len(feature_names),
            "gt_used_as_inference_feature": False,
        },
    }


def compact_matrix(split_data: dict[str, Any], idx: torch.Tensor) -> torch.Tensor:
    f = split_data["features"]
    return torch.stack([f[name][idx].float() for name in split_data["feature_names"]], dim=1).nan_to_num(0.0)


def precision_at_k_compact(label: torch.Tensor, score: torch.Tensor, k: int) -> float:
    top = torch.topk(score.float(), k=min(k, len(score))).indices
    return float(label[top].float().mean().item()) if len(top) else 0.0


def topk_rows(split_data: dict[str, Any], scores: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    label = split_data["label"].bool()
    rows = split_data["rows"]
    tensors = split_data["table"]["tensors"]
    front = tensors["front_region"][rows].bool()
    future = tensors["future_h4h6"][rows].bool()
    out: list[dict[str, Any]] = []
    for name, score in scores.items():
        order = torch.argsort(score.float(), descending=True)
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000]:
            if len(order) < k:
                continue
            top = order[:k]
            out.append(
                {
                    "split": split_data["split"],
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
    return out


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
    k = min(len(logits), max(8192, int(len(logits) * frac)))
    top = torch.unique(torch.cat([torch.topk(logits, k=k).indices, torch.topk(prior, k=k).indices]))
    weight = 1.0 + 5.0 * prior[top] + 4.0 * (1.0 - labels[top])
    loss = F.binary_cross_entropy_with_logits(logits[top], labels[top], reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def train_model(train: dict[str, Any], cfg: H10Config, device: torch.device) -> tuple[CompactRanker, list[dict[str, Any]]]:
    model = CompactRanker(len(train["feature_names"]), cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_all = train["label"].float()
    prior_all = train["prior"].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    front_all = train["features"]["front_region"].float()
    future_all = train["features"]["future_h4h6"].float()
    h9_all = train["features"]["h9_geo_count_norm"].float()
    density_all = train["features"]["local_density_proxy"].float()
    pos_weight = torch.tensor([(len(y_all) - y_all.sum().item()) / max(1.0, y_all.sum().item())], device=device).clamp(1.0, 8.0)
    log: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        order = torch.randperm(len(y_all), generator=torch.Generator().manual_seed(cfg.seed * 1000 + epoch))
        losses: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            xb = compact_matrix(train, b)
            if device.type == "cuda":
                try:
                    xb = xb.pin_memory()
                except RuntimeError:
                    pass
            xb = xb.to(device, non_blocking=True)
            yb = y_all[b].to(device, non_blocking=True)
            prior = prior_all[b].to(device, non_blocking=True)
            front = front_all[b].to(device, non_blocking=True)
            future = future_all[b].to(device, non_blocking=True)
            h9v = h9_all[b].to(device, non_blocking=True)
            density = density_all[b].to(device, non_blocking=True)
            logits = model(xb, prior)
            sample_weight = 1.0 + yb * (1.8 + 1.2 * front + 1.2 * future + 0.8 * h9v) + (1.0 - yb) * (3.5 * prior + 2.0 * h9v + 1.2 * density + 0.8 * front)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            bce = (bce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            rank = pairwise_rank_loss(logits, yb, prior)
            topk = topk_surrogate_loss(logits, yb, prior)
            prob = torch.sigmoid(logits)
            fp_penalty = (prob * (1.0 - yb) * (1.0 + 3.0 * prior) * (1.0 + 2.0 * h9v + density)).mean()
            residual = logits - torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
            loss = bce + 3.2 * rank + 4.5 * topk + 1.2 * fp_penalty + 0.008 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 6.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_rows": int(len(y_all)),
            "positive_rate": float(y_all.mean().item()),
            "batch_size": cfg.batch_size,
            "device": str(device),
            "seconds": float(time.time() - t0),
        }
        log.append(row)
        print(f"[h10] epoch={epoch} loss={row['loss']:.5f} sec={row['seconds']:.1f}", flush=True)
    return model.eval(), log


@torch.inference_mode()
def score_model(model: CompactRanker, data: dict[str, Any], cfg: H10Config, device: torch.device) -> torch.Tensor:
    out = torch.empty((len(data["label"]),), dtype=torch.float32)
    prior_all = data["prior"].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    idx_all = torch.arange(len(out))
    for start in range(0, len(idx_all), cfg.score_batch_size):
        idx = idx_all[start : start + cfg.score_batch_size]
        x = compact_matrix(data, idx)
        prior = prior_all[idx]
        if device.type == "cuda":
            try:
                x = x.pin_memory()
                prior = prior.pin_memory()
            except RuntimeError:
                pass
        logits = model(x.to(device, non_blocking=True), prior.to(device, non_blocking=True))
        out[idx] = torch.sigmoid(logits).detach().cpu()
    return out


def full_score_for_selection(data: dict[str, Any], compact_score: torch.Tensor) -> torch.Tensor:
    full = torch.full((len(data["table"]["tensors"]["GT_occ"]),), -torch.inf, dtype=torch.float32)
    full[data["rows"]] = compact_score.float()
    return full


def run_selection(val: dict[str, Any], add_score: torch.Tensor, suppress_score: torch.Tensor) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    tables = {"val": val["table"]}
    for sup in [0.02, 0.03, 0.04, 0.05]:
        for bal in [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
            for fixed in [0, 16, 32, 64, 96, 128, 192]:
                cfg = sw14e.Config(max_suppress_ratio=sup, add_suppress_balance=bal, add_fixed_budget=fixed)
                budget = {"name": f"h10_sup{sup}_bal{bal}_fixed{fixed}", "suppress_topk_ratio": 1.0, "add_topk_ratio": 1.0, "add_strength": "strong_medium"}
                _, summary = sw14e.evaluate_selection("val", tables["val"], add_score, suppress_score, budget, cfg)
                row = {**summary, "max_suppress_ratio": sup, "add_suppress_balance": bal, "add_fixed_budget": fixed}
                rows.append(row)
                if summary["safety_pass_all"] and summary["recall_nonregression_pass"]:
                    if best is None or float(summary["mean_net_score"]) > float(best["mean_net_score"]):
                        best = row
    write_csv(REPORTS_DIR / "sw14h10_compact_sparse_selection_sweep_val.csv", rows)
    return best or (max(rows, key=lambda r: float(r["mean_net_score"])) if rows else {})


def plot_precision(rows: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for name in ["h7b_lean", "h10_compact_ranker", "h10_blend_h7b_10", "h10_blend_h7b_20", "h10_blend_h7b_35", "h10_best_val"]:
        sub = sorted([r for r in rows if r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H10 compact sparse add precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h10_compact_sparse_precision.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = H10Config(epochs=args.epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim, device=args.device)
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    train = build_compact_split("train")
    write_json(REPORTS_DIR / "sw14h10_compact_sparse_config.json", {"config": asdict(cfg), "train_stats": train["stats"], "feature_names": train["feature_names"], "uses_eval_debug": False, "uses_core100_or_core500": False, "gt_used_as_inference_feature": False})
    model, log = train_model(train, cfg, device)
    write_csv(REPORTS_DIR / "sw14h10_compact_sparse_training_log.csv", log)
    torch.save({"state_dict": model.state_dict(), "feature_names": train["feature_names"], "config": asdict(cfg), "resume_claim": False}, CHECKPOINT_DIR / "sw14h10_compact_sparse_ranker_best.pth")
    del train
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    val = build_compact_split("val")
    val_score = score_model(model, val, cfg, device)
    h10_norm = norm(val_score)
    h7b = val["prior"].float()
    score_map = {
        "h7b_lean": h7b,
        "h10_compact_ranker": val_score,
        "h10_blend_h7b_10": 0.10 * h10_norm + 0.90 * h7b,
        "h10_blend_h7b_20": 0.20 * h10_norm + 0.80 * h7b,
        "h10_blend_h7b_35": 0.35 * h10_norm + 0.65 * h7b,
        "h10_blend_h7b_50": 0.50 * h10_norm + 0.50 * h7b,
        "h10_geo_count_h7b": val["features"]["h9_geo_count_h7b"].float(),
        "h10_score_pass_h7b": val["features"]["h9_score_pass_h7b"].float(),
    }
    topk = topk_rows(val, score_map)
    best_val_100k = max((r for r in topk if int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    h7b_val_100k = next(r for r in topk if r["score_name"] == "h7b_lean" and int(r["topk"]) == 100000)
    score_map["h10_best_val"] = score_map[best_val_100k["score_name"]]
    plot_precision(topk + topk_rows(val, {"h10_best_val": score_map["h10_best_val"]}))
    write_csv(REPORTS_DIR / "sw14h10_compact_sparse_topk_precision.csv", topk)
    torch.save({"val_sw14h10_compact_scores": val_score, "val_row_index": val["rows"], "feature_names": val["feature_names"]}, ARTIFACTS_DIR / "sw14h10_compact_sparse_scores.pt")
    full_add = full_score_for_selection(val, score_map[best_val_100k["score_name"]])
    selection = run_selection(val, full_add, val["suppress_scores"])
    gain = float(best_val_100k["precision"]) - float(h7b_val_100k["precision"])
    if float(best_val_100k["precision"]) >= cfg.target_precision_at_100k:
        decision = "SW14H10_1_TARGET_090_REACHED"
    elif gain >= 0.10:
        decision = "SW14H10_2_MATERIAL_COMPACT_SPARSE_GAIN"
    elif gain > 0.005:
        decision = "SW14H10_3_SMALL_COMPACT_SPARSE_GAIN"
    else:
        decision = "SW14H10_4_COMPACT_SPARSE_NOT_ENOUGH"
    final = {
        "decision": decision,
        "target_precision_at_100k": cfg.target_precision_at_100k,
        "target_reached": bool(float(best_val_100k["precision"]) >= cfg.target_precision_at_100k),
        "best_val_top100k": best_val_100k,
        "h7b_val_top100k": h7b_val_100k,
        "precision_gain_over_h7b": gain,
        "selection": selection,
        "val_stats": val["stats"],
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "recommended_next_action": "If H10 remains far below 0.9@100K, stop re-ranking this ROI and build a new add proposal source from decoder-query tensors or generative ROI proposals.",
    }
    write_json(REPORTS_DIR / "sw14h10_compact_sparse_final_decision.json", final)
    print(f"[h10] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
