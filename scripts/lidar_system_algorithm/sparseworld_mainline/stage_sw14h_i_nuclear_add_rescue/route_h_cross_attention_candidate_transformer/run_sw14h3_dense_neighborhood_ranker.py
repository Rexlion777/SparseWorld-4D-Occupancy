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
    spec = importlib.util.spec_from_file_location("sw14hi_h3", SW14HI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SW14HI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sw14hi_h3"] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_sw14hi_module()
sw14e = sw14hi.sw14e
EMPTY_IDX = sw14e.EMPTY_IDX
TEACHER_CACHE = sw14e.TEACHER_CACHE


DENSE_FEATURE_NAMES = [
    "r3_raw_count",
    "r3_final_count",
    "r3_native_count",
    "r3_pruned_count",
    "r3_raw_final_empty_count",
    "r3_conf_mean",
    "r3_margin_mean",
    "r3_agree_mean",
    "r3_conf_max",
    "r3_margin_max",
    "r5_raw_count",
    "r5_final_count",
    "r5_native_count",
    "r5_pruned_count",
    "r5_raw_final_empty_count",
    "r5_conf_mean",
    "r5_margin_mean",
    "r5_agree_mean",
    "r5_conf_max",
    "r5_margin_max",
    "r3_pruned_ratio",
    "r5_pruned_ratio",
    "r3_support_ratio",
    "r5_support_ratio",
    "r3_final_gap",
    "r5_final_gap",
    "prior_b1",
    "prior_b2",
    "prior_b3",
    "prior_h2",
    "prior_h2b3",
    "front_region",
    "future_h4h6",
    "raw_confidence",
    "raw_margin",
    "camera_view_agreement",
    "local_density_proxy",
    "neighbor_occ_count_norm",
    "front_local_density_context",
    "x_norm",
    "abs_y_norm",
    "z_norm",
    "range_norm",
    "horizon_norm",
    "sector_norm",
    "distance_bin_norm",
]


@dataclass(frozen=True)
class H3Config:
    seed: int = 97
    epochs: int = 6
    train_max_pos: int = 700_000
    train_max_neg: int = 950_000
    hard_neg_topk: int = 700_000
    batch_size: int = 262_144
    score_batch_size: int = 524_288
    hidden_dim: int = 256
    lr: float = 1.5e-3
    device: str = "cuda"


class DenseResidualRanker(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.03),
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


def load_bundle(sample_id: int) -> dict[str, Any]:
    path = TEACHER_CACHE / f"A10_drop_front_triplet__sample{sample_id:03d}.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def occ(x: torch.Tensor) -> torch.Tensor:
    return x.long() != EMPTY_IDX


def conv_sum(channels: torch.Tensor, kernel: tuple[int, int, int]) -> torch.Tensor:
    c = channels.shape[0]
    weight = torch.ones((c, 1, *kernel), dtype=torch.float32)
    pad = (kernel[0] // 2, kernel[1] // 2, kernel[2] // 2)
    return F.conv3d(channels.unsqueeze(0).float(), weight, padding=pad, groups=c).squeeze(0)


def max_pool(channels: torch.Tensor, kernel: tuple[int, int, int]) -> torch.Tensor:
    pad = (kernel[0] // 2, kernel[1] // 2, kernel[2] // 2)
    return F.max_pool3d(channels.unsqueeze(0).float(), kernel_size=kernel, stride=1, padding=pad).squeeze(0)


def dense_group_features(th: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    raw = occ(th["teacher_raw_semantic"])
    final = occ(th["teacher_final_semantic"])
    native = occ(th["native_semantic"])
    conf = th["teacher_confidence"].float().clamp(0, 1)
    margin = th["teacher_margin"].float().clamp(0, 1)
    agree = th["agreement"].float().clamp(0, 1)
    pruned = raw & ~final
    raw_final_empty = raw & ~final
    base = torch.stack(
        [
            raw.float(),
            final.float(),
            native.float(),
            pruned.float(),
            raw_final_empty.float(),
            conf,
            margin,
            agree,
        ],
        dim=0,
    )
    r3_sum = conv_sum(base, (3, 3, 3))
    r5_sum = conv_sum(base, (5, 5, 3))
    r3_max = max_pool(torch.stack([conf, margin], dim=0), (3, 3, 3))
    r5_max = max_pool(torch.stack([conf, margin], dim=0), (5, 5, 3))
    vol3 = 27.0
    vol5 = 75.0
    g3 = r3_sum[:, x, y, z]
    g5 = r5_sum[:, x, y, z]
    m3 = r3_max[:, x, y, z]
    m5 = r5_max[:, x, y, z]
    r3_raw = g3[0].clamp_min(0)
    r5_raw = g5[0].clamp_min(0)
    features = torch.stack(
        [
            g3[0] / vol3,
            g3[1] / vol3,
            g3[2] / vol3,
            g3[3] / vol3,
            g3[4] / vol3,
            g3[5] / vol3,
            g3[6] / vol3,
            g3[7] / vol3,
            m3[0],
            m3[1],
            g5[0] / vol5,
            g5[1] / vol5,
            g5[2] / vol5,
            g5[3] / vol5,
            g5[4] / vol5,
            g5[5] / vol5,
            g5[6] / vol5,
            g5[7] / vol5,
            m5[0],
            m5[1],
            g3[3] / r3_raw.clamp_min(1.0),
            g5[3] / r5_raw.clamp_min(1.0),
            (g3[0] + g3[2]) / vol3,
            (g5[0] + g5[2]) / vol5,
            (g3[0] - g3[1]).clamp_min(0) / vol3,
            (g5[0] - g5[1]).clamp_min(0) / vol5,
        ],
        dim=1,
    )
    return features.float()


def add_mask(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()


def build_feature_table(
    split: str,
    table: dict[str, Any],
    scores: dict[str, torch.Tensor],
    h2_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    tensors = table["tensors"]
    cache = sw14hi.derived_cache(tensors, scores)
    mask = add_mask(tensors)
    group_keys = tensors["group_key"].to(torch.int32)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    priors: list[torch.Tensor] = []
    rows_out: list[torch.Tensor] = []
    h2_norm = sw14hi.normalize_score(h2_scores, mask).nan_to_num(0.0, neginf=0.0)
    h2b3 = 0.25 * h2_norm + 0.75 * cache["b3_add_score_norm"]
    for key in torch.unique(group_keys):
        group = torch.nonzero((group_keys == key) & mask, as_tuple=False).flatten()
        if len(group) == 0:
            continue
        sample_id = int(key.item()) // 10
        horizon_id = int(key.item()) % 10
        th = load_bundle(sample_id)["by_horizon"][horizon_id]
        x = tensors["voxel_x"][group].long()
        y = tensors["voxel_y"][group].long()
        z = tensors["voxel_z"][group].long()
        dense = dense_group_features(th, x, y, z)
        center = torch.stack(
            [
                cache["b1_add_score_norm"][group].float(),
                cache["b2_add_score_norm"][group].float(),
                cache["b3_add_score_norm"][group].float(),
                h2_norm[group].float(),
                h2b3[group].float(),
                tensors["front_region"][group].float(),
                tensors["future_h4h6"][group].float(),
                tensors["raw_confidence"][group].float(),
                tensors["raw_margin"][group].float(),
                tensors["camera_view_agreement"][group].float(),
                tensors["local_density_proxy"][group].float(),
                tensors["neighbor_occ_count_norm"][group].float(),
                tensors["front_local_density_context"][group].float(),
                tensors["x_norm"][group].float(),
                tensors["abs_y_norm"][group].float(),
                tensors["z_norm"][group].float(),
                tensors["range_norm"][group].float(),
                tensors["horizon_norm"][group].float(),
                tensors["sector_norm"][group].float(),
                tensors["distance_bin_norm"][group].float(),
            ],
            dim=1,
        )
        xs.append(torch.cat([dense, center], dim=1).half())
        ys.append(tensors["GT_occ"][group].float())
        priors.append(h2b3[group].float().clamp(1.0e-4, 1.0 - 1.0e-4))
        rows_out.append(group.to(torch.long))
        if len(xs) % 50 == 0:
            print(f"[h3] {split} groups {len(xs)} rows {sum(len(v) for v in rows_out)}", flush=True)
    return {
        "x": torch.cat(xs, dim=0),
        "y": torch.cat(ys, dim=0),
        "prior": torch.cat(priors, dim=0),
        "rows": torch.cat(rows_out, dim=0),
        "feature_names": torch.arange(len(DENSE_FEATURE_NAMES)),
    }


def sample_indices(y: torch.Tensor, prior: torch.Tensor, cfg: H3Config) -> torch.Tensor:
    pos = torch.nonzero(y > 0.5, as_tuple=False).flatten()
    neg = torch.nonzero(y <= 0.5, as_tuple=False).flatten()
    hard_neg = neg[torch.argsort(prior[neg], descending=True)[: min(cfg.hard_neg_topk, len(neg))]]
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


def pairwise_loss(logits: torch.Tensor, y: torch.Tensor, max_pairs: int = 32768) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    return F.softplus(0.50 - pos[:k] + neg[:k]).mean()


def topk_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    k = min(len(logits), max(512, int(len(logits) * 0.30)))
    top = torch.topk(logits, k=k).indices
    return F.binary_cross_entropy_with_logits(logits[top], y[top])


def train_model(train: dict[str, torch.Tensor], cfg: H3Config, device: torch.device) -> tuple[DenseResidualRanker, list[dict[str, Any]]]:
    idx = sample_indices(train["y"], train["prior"], cfg)
    x = train["x"][idx].float()
    y = train["y"][idx].float()
    prior = train["prior"][idx].float()
    if device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()
        prior = prior.pin_memory()
    model = DenseResidualRanker(x.shape[1], cfg.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=2.0e-4)
    rows: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 100 + epoch)
        order = torch.randperm(len(y), generator=gen)
        losses: list[float] = []
        bces: list[float] = []
        ranks: list[float] = []
        topks: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            xb = x[b].to(device, non_blocking=True)
            yb = y[b].to(device, non_blocking=True)
            pb = prior[b].to(device, non_blocking=True)
            logits = model(xb, pb)
            w = 1.0 + yb * 0.8 + (1.0 - yb) * (1.0 + 1.5 * pb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce = (bce * w).sum() / w.sum().clamp_min(1.0)
            rank = pairwise_loss(logits, yb)
            tk = topk_loss(logits, yb)
            prior_logits = torch.logit(pb.clamp(1.0e-4, 1.0 - 1.0e-4))
            residual = (logits - prior_logits).square().mean()
            loss = bce + 2.0 * rank + 3.0 * tk + 0.010 * residual
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            bces.append(float(bce.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
            topks.append(float(tk.detach().cpu().item()))
        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "bce": float(np.mean(bces)),
                "pairwise": float(np.mean(ranks)),
                "topk": float(np.mean(topks)),
                "train_rows": int(len(y)),
                "positive_rate": float(y.mean().item()),
            }
        )
    return model.eval(), rows


@torch.inference_mode()
def score_model(model: DenseResidualRanker, data: dict[str, torch.Tensor], cfg: H3Config, device: torch.device) -> torch.Tensor:
    scores = torch.empty((len(data["y"]),), dtype=torch.float32)
    for start in range(0, len(scores), cfg.score_batch_size):
        end = min(start + cfg.score_batch_size, len(scores))
        x = data["x"][start:end].float()
        p = data["prior"][start:end].float()
        if device.type == "cuda":
            x = x.pin_memory()
            p = p.pin_memory()
        logits = model(x.to(device, non_blocking=True), p.to(device, non_blocking=True))
        scores[start:end] = torch.sigmoid(logits).detach().cpu()
    return scores


def topk_rows(data: dict[str, torch.Tensor], full_scores: dict[str, torch.Tensor], table: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows_global = data["rows"]
    y_full = table["tensors"]["GT_occ"].bool()
    front = table["tensors"]["front_region"].bool()
    future = table["tensors"]["future_h4h6"].bool()
    out: list[dict[str, Any]] = []
    for name, score in full_scores.items():
        order_local = torch.argsort(score, descending=True)
        ordered_rows = rows_global[order_local]
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000]:
            if len(ordered_rows) < k:
                continue
            top = ordered_rows[:k]
            out.append(
                {
                    "split": split,
                    "score_name": name,
                    "topk": k,
                    "precision": float(y_full[top].float().mean().item()),
                    "positive_count": int(y_full[top].sum().item()),
                    "front_precision": float(y_full[top][front[top]].float().mean().item()) if bool(front[top].any().item()) else 0.0,
                    "future_precision": float(y_full[top][future[top]].float().mean().item()) if bool(future[top].any().item()) else 0.0,
                    "front_count": int(front[top].sum().item()),
                    "future_count": int(future[top].sum().item()),
                }
            )
    return out


def assign_full_score(table: dict[str, Any], data: dict[str, torch.Tensor], score: torch.Tensor) -> torch.Tensor:
    full = torch.full((len(table["tensors"]["GT_occ"]),), -torch.inf)
    full[data["rows"]] = score.float()
    return full


def plot_precision(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 4.5))
    for name in ["h2b3_prior", "h3_dense", "h3_blend_prior_25", "h3_blend_prior_50"]:
        sub = sorted([r for r in rows if r["split"] == "val" and r["score_name"] == name], key=lambda r: int(r["topk"]))
        if not sub:
            continue
        plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H3 dense neighborhood precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h3_dense_neighborhood_precision.png", dpi=180)
    plt.close()


def main() -> None:
    cfg = H3Config()
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    tables = sw14hi.load_tables()
    score_payload = sw14hi.load_score_payload(tables)
    h2_path = ARTIFACTS_DIR / "sw14h2_true_local_attention_scores.pt"
    h2 = torch.load(h2_path, map_location="cpu", weights_only=False)
    print("[h3] building train dense feature table", flush=True)
    train = build_feature_table("train", tables["train"], score_payload["train"], h2["train_sw14h2_add_scores"])
    print("[h3] building val dense feature table", flush=True)
    val = build_feature_table("val", tables["val"], score_payload["val"], h2["val_sw14h2_add_scores"])
    torch.save({"train": train, "val": val, "feature_names": DENSE_FEATURE_NAMES}, ARTIFACTS_DIR / "sw14h3_dense_neighborhood_feature_table.pt")
    write_json(
        REPORTS_DIR / "sw14h3_dense_neighborhood_config.json",
        {
            "model": "dense teacher-cache neighborhood residual ranker",
            "feature_names": DENSE_FEATURE_NAMES,
            "target": "val add precision@100K >= 0.9",
            "gt_used_as_inference_feature": False,
            "cache_source": str(TEACHER_CACHE),
            "config": cfg.__dict__,
        },
    )
    print("[h3] training", flush=True)
    model, train_log = train_model(train, cfg, device)
    write_csv(REPORTS_DIR / "sw14h3_dense_neighborhood_training_log.csv", train_log)
    torch.save({"state_dict": model.state_dict(), "feature_names": DENSE_FEATURE_NAMES, "seed": cfg.seed}, CHECKPOINT_DIR / "sw14h3_dense_neighborhood_ranker_best.pth")
    print("[h3] scoring", flush=True)
    train_h3 = score_model(model, train, cfg, device)
    val_h3 = score_model(model, val, cfg, device)
    torch.save({"train_sw14h3_add_scores": train_h3, "val_sw14h3_add_scores": val_h3}, ARTIFACTS_DIR / "sw14h3_dense_neighborhood_scores.pt")
    train_prior = train["prior"].float()
    val_prior = val["prior"].float()
    train_scores = {
        "h2b3_prior": train_prior,
        "h3_dense": train_h3,
        "h3_blend_prior_25": 0.25 * train_h3 + 0.75 * train_prior,
        "h3_blend_prior_50": 0.50 * train_h3 + 0.50 * train_prior,
    }
    val_scores = {
        "h2b3_prior": val_prior,
        "h3_dense": val_h3,
        "h3_blend_prior_25": 0.25 * val_h3 + 0.75 * val_prior,
        "h3_blend_prior_50": 0.50 * val_h3 + 0.50 * val_prior,
    }
    rows = topk_rows(train, train_scores, tables["train"], "train") + topk_rows(val, val_scores, tables["val"], "val")
    write_csv(REPORTS_DIR / "sw14h3_dense_neighborhood_topk_precision.csv", rows)
    plot_precision(rows)
    best_train_100k = max((r for r in rows if r["split"] == "train" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    best_val_100k = max((r for r in rows if r["split"] == "val" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    prior_val_100k = next(r for r in rows if r["split"] == "val" and r["score_name"] == "h2b3_prior" and int(r["topk"]) == 100000)
    best_score_name = best_val_100k["score_name"]
    full_val_score = assign_full_score(tables["val"], val, val_scores[best_score_name])
    selection = {}
    try:
        rows_sel, selection = sw14hi.run_selection_sweep("SW14H3", tables, full_val_score, score_payload["val"]["suppress"])
        write_csv(REPORTS_DIR / "sw14h3_dense_neighborhood_selection_sweep_val.csv", rows_sel)
    except Exception as exc:
        selection = {"error": str(exc)}
    write_json(REPORTS_DIR / "sw14h3_dense_neighborhood_selection_summary.json", selection)
    target = 0.9
    if float(best_val_100k["precision"]) >= target:
        decision = "SW14H3_1_TARGET_090_REACHED"
    elif float(best_val_100k["precision"]) > float(prior_val_100k["precision"]) + 0.02:
        decision = "SW14H3_2_DENSE_VAL_IMPROVED_BELOW_090"
    elif float(best_train_100k["precision"]) > float(next(r for r in rows if r["split"] == "train" and r["score_name"] == "h2b3_prior" and int(r["topk"]) == 100000)["precision"]) + 0.05:
        decision = "SW14H3_3_TRAIN_OVERFIT_ONLY"
    else:
        decision = "SW14H3_4_NO_DENSE_GAIN"
    final = {
        "decision": decision,
        "target_precision_at_100k": target,
        "target_reached": bool(float(best_val_100k["precision"]) >= target),
        "best_train_top100k": best_train_100k,
        "best_val_top100k": best_val_100k,
        "prior_val_top100k": prior_val_100k,
        "selection": selection,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "gt_used_as_inference_feature": False,
        "recommended_next_action": "If not reached, construct real postprocess survival labels/features or learned proposal from raw occupancy logits; dense local counts alone are insufficient.",
    }
    write_json(REPORTS_DIR / "sw14h3_dense_neighborhood_final_decision.json", final)
    print(f"[h3] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
