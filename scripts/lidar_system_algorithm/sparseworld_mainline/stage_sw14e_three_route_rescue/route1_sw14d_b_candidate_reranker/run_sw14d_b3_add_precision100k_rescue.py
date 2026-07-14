from __future__ import annotations

import csv
import importlib.util
import json
import os
import random
import sys
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
SW14E_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14e_three_route_rescue/run_sw14e_three_route_rescue.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14e_three_route_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14e_three_route_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14e_three_route_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def load_sw14e_module():
    spec = importlib.util.spec_from_file_location("sw14e_mod_b3", SW14E_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SW14E_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sw14e_mod_b3"] = module
    spec.loader.exec_module(module)
    return module


sw14e = load_sw14e_module()


B3_EXTRA_FEATURES = [
    "strong_add_candidate",
    "medium_add_candidate",
    "front_confidence",
    "future_confidence",
    "front_margin",
    "future_margin",
    "neighbor_confidence",
    "high_conf_raw_add_rule",
    "survival_gap",
    "frontcap_boundary_proxy",
    "f3_boundary_proxy",
    "raw_final_reliability",
    "native_raw_reliability",
    "density_supported_add_prior",
    "rule_score_norm",
    "b1_add_score_norm",
    "b2_add_score_norm",
    "b1_b2_agreement",
    "confidence_agreement",
    "margin_agreement",
    "front_future_confidence",
]


class Precision100KScorer(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Dropout(p=0.04),
            nn.Linear(256, 160),
            nn.SiLU(),
            nn.Dropout(p=0.03),
            nn.Linear(160, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def normalize(payload: Any) -> Any:
    return sw14e.normalize(payload)


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


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def base_rule_score(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    conf = tensors["raw_confidence"].float()
    margin = tensors["raw_margin"].float()
    agreement = tensors["camera_view_agreement"].float()
    neighbor = tensors["neighbor_occ_count_norm"].float()
    front = tensors["front_region"].float()
    future = tensors["future_h4h6"].float()
    strong = tensors["strong_add_candidate"].float()
    raw_but_final_empty = tensors["raw_but_final_empty"].float()
    native = tensors["native_final_occ"].float()
    return (
        conf
        + 0.45 * margin
        + 0.22 * agreement
        + 0.22 * neighbor
        + 0.14 * front
        + 0.16 * future
        + 0.12 * strong
        + 0.10 * raw_but_final_empty
        + 0.08 * native
    )


def normalize_score(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(score.float(), -torch.inf)
    vals = score[mask].float()
    if len(vals) == 0:
        return out
    out[mask] = (vals - vals.min()) / (vals.max() - vals.min() + 1.0e-9)
    return out


def get_b2_scores(split: str, n_rows: int) -> torch.Tensor:
    path = ARTIFACTS_DIR / "sw14d_b2_add_recall_scores.pt"
    if not path.exists():
        return torch.full((n_rows,), -torch.inf)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = "train_add_b2_scores" if split == "train" else "val_add_b2_scores"
    return payload.get(key, torch.full((n_rows,), -torch.inf)).float()


def prepare_feature_cache(
    tensors: dict[str, torch.Tensor],
    base_add_scores: torch.Tensor,
    b2_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = add_mask(tensors)
    rule = normalize_score(base_rule_score(tensors), mask)
    b1 = normalize_score(base_add_scores.float(), mask)
    b2 = normalize_score(b2_scores.float(), mask)
    conf = tensors["raw_confidence"].float()
    margin = tensors["raw_margin"].float()
    agreement = tensors["camera_view_agreement"].float()
    neighbor = tensors["neighbor_occ_count_norm"].float()
    front = tensors["front_region"].float()
    future = tensors["future_h4h6"].float()
    raw_occ = tensors["raw_occ"].float()
    final_occ = tensors["teacher_final_occ"].float()
    native_occ = tensors["native_final_occ"].float()
    raw_but_final_empty = tensors["raw_but_final_empty"].float()
    was_pruned_f3 = tensors["was_pruned_by_F3"].float()
    was_pruned_fc = tensors["was_pruned_by_FrontCap"].float()
    density = tensors["local_density_proxy"].float()
    strong = tensors["strong_add_candidate"].float()
    medium = tensors["medium_add_candidate"].float()
    return {
        "rule": rule,
        "b1": b1,
        "b2": b2,
        "strong": strong,
        "medium": medium,
        "front_confidence": front * conf,
        "future_confidence": future * conf,
        "front_margin": front * margin,
        "future_margin": future * margin,
        "neighbor_confidence": neighbor * conf,
        "high_conf_raw_add_rule": raw_but_final_empty * (conf > 0.55).float() * (agreement > 0.6).float(),
        "survival_gap": raw_but_final_empty * (conf + margin) * 0.5,
        "frontcap_boundary_proxy": was_pruned_fc * front * (0.5 + 0.5 * density),
        "f3_boundary_proxy": was_pruned_f3 * (0.5 + 0.5 * agreement),
        "raw_final_reliability": raw_occ * (1.0 - final_occ) * agreement * conf,
        "native_raw_reliability": raw_occ * native_occ * (0.5 + 0.5 * agreement),
        "density_supported_add_prior": raw_but_final_empty * density * (0.5 + 0.5 * agreement),
        "b1_b2_agreement": b1 * b2,
        "confidence_agreement": conf * agreement,
        "margin_agreement": margin * agreement,
        "front_future_confidence": torch.maximum(front, future) * conf,
        "rule_score_norm": rule,
        "b1_add_score_norm": b1,
        "b2_add_score_norm": b2,
    }


def feature_matrix(
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    idx: torch.Tensor,
) -> torch.Tensor:
    base = [tensors[name][idx].float() for name in sw14e.NO_GT_FEATURES]
    extra = [
        cache["strong"][idx].float(),
        cache["medium"][idx].float(),
        cache["front_confidence"][idx].float(),
        cache["future_confidence"][idx].float(),
        cache["front_margin"][idx].float(),
        cache["future_margin"][idx].float(),
        cache["neighbor_confidence"][idx].float(),
        cache["high_conf_raw_add_rule"][idx].float(),
        cache["survival_gap"][idx].float(),
        cache["frontcap_boundary_proxy"][idx].float(),
        cache["f3_boundary_proxy"][idx].float(),
        cache["raw_final_reliability"][idx].float(),
        cache["native_raw_reliability"][idx].float(),
        cache["density_supported_add_prior"][idx].float(),
        cache["rule_score_norm"][idx].float(),
        cache["b1_add_score_norm"][idx].float(),
        cache["b2_add_score_norm"][idx].float(),
        cache["b1_b2_agreement"][idx].float(),
        cache["confidence_agreement"][idx].float(),
        cache["margin_agreement"][idx].float(),
        cache["front_future_confidence"][idx].float(),
    ]
    return torch.stack(base + extra, dim=1)


def sample_indices(
    tensors: dict[str, torch.Tensor],
    seed: int,
    hard_neg: torch.Tensor | None = None,
    hard_pos: torch.Tensor | None = None,
    max_pos: int = 900_000,
    max_neg: int = 1_000_000,
) -> torch.Tensor:
    mask = add_mask(tensors)
    label = tensors["GT_occ"].bool()
    pos = torch.nonzero(mask & label, as_tuple=False).flatten()
    neg = torch.nonzero(mask & ~label, as_tuple=False).flatten()
    gen = torch.Generator().manual_seed(seed)
    if hard_pos is not None and len(hard_pos):
        pos = torch.unique(torch.cat([pos, hard_pos.cpu()]))
    if hard_neg is not None and len(hard_neg):
        neg = torch.unique(torch.cat([neg, hard_neg.cpu()]))
    if len(pos) > max_pos:
        pos = pos[torch.randperm(len(pos), generator=gen)[:max_pos]]
    if len(neg) > max_neg:
        neg = neg[torch.randperm(len(neg), generator=gen)[:max_neg]]
    idx = torch.cat([pos, neg])
    return idx[torch.randperm(len(idx), generator=gen)]


def pairwise_rank_loss(logits: torch.Tensor, y: torch.Tensor, max_pairs: int = 16384) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    return F.softplus(0.35 - pos[:k] + neg[:k]).mean()


def train_one_stage(
    model: Precision100KScorer,
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    idx: torch.Tensor,
    device: torch.device,
    seed: int,
    stage: str,
    epochs: int,
    lr: float,
) -> list[dict[str, Any]]:
    x_cpu = feature_matrix(tensors, cache, idx).float()
    y_cpu = tensors["GT_occ"][idx].float()
    front = tensors["front_region"][idx].float()
    future = tensors["future_h4h6"][idx].float()
    hard_prior = torch.maximum(cache["b2"][idx].nan_to_num(-1.0), cache["b1"][idx].nan_to_num(-1.0)).clamp_min(0.0)
    sample_weight = 1.0 + (1.0 - y_cpu) * (1.5 + 2.0 * hard_prior) + y_cpu * (0.6 * front + 0.8 * future)
    if device.type == "cuda":
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
        sample_weight = sample_weight.pin_memory()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2.0e-4)
    rows: list[dict[str, Any]] = []
    batch_size = 262_144
    for epoch in range(epochs):
        gen = torch.Generator().manual_seed(seed * 1000 + epoch + (0 if stage == "warm" else 100))
        order = torch.randperm(len(y_cpu), generator=gen)
        losses: list[float] = []
        bces: list[float] = []
        ranks: list[float] = []
        for start in range(0, len(order), batch_size):
            bidx = order[start : start + batch_size]
            xb = x_cpu[bidx].to(device, non_blocking=True)
            yb = y_cpu[bidx].to(device, non_blocking=True)
            wb = sample_weight[bidx].to(device, non_blocking=True)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce = (bce * wb).sum() / wb.sum().clamp_min(1.0)
            rank = pairwise_rank_loss(logits, yb)
            neg_top = logits[yb <= 0.5]
            top_neg_penalty = F.softplus(neg_top).mean() if len(neg_top) else logits.new_tensor(0.0)
            loss = bce + 1.75 * rank + 0.20 * top_neg_penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            bces.append(float(bce.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
        rows.append(
            {
                "stage": stage,
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "bce": float(np.mean(bces)),
                "rank_loss": float(np.mean(ranks)),
                "train_rows": int(len(y_cpu)),
                "positive_rate": float(y_cpu.mean().item()),
            }
        )
    return rows


@torch.inference_mode()
def score_model(
    model: Precision100KScorer,
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    mask = add_mask(tensors)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    scores = torch.full((len(mask),), -torch.inf, dtype=torch.float32)
    batch = 524_288
    for start in range(0, len(idx), batch):
        chunk = idx[start : start + batch]
        xb = feature_matrix(tensors, cache, chunk).float()
        if device.type == "cuda":
            xb = xb.pin_memory()
        scores[chunk] = torch.sigmoid(model(xb.to(device, non_blocking=True))).detach().cpu()
    return scores


def topk_rows(tensors: dict[str, torch.Tensor], scores: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    mask = add_mask(tensors)
    label = tensors["GT_occ"].bool()
    front = tensors["front_region"].bool()
    future = tensors["future_h4h6"].bool()
    rows: list[dict[str, Any]] = []
    for name, score in scores.items():
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        order = idx[torch.argsort(score[idx], descending=True)]
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


def threshold_feasibility(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor]) -> dict[str, Any]:
    mask0 = add_mask(tensors)
    label = tensors["GT_occ"].bool()
    rows: list[dict[str, Any]] = []
    filter_defs = {
        "global": torch.ones_like(mask0, dtype=torch.bool),
        "front": tensors["front_region"].bool(),
        "future": tensors["future_h4h6"].bool(),
        "front_future": tensors["front_region"].bool() & tensors["future_h4h6"].bool(),
        "front_strong": tensors["front_region"].bool() & tensors["strong_add_candidate"].bool(),
        "future_strong": tensors["future_h4h6"].bool() & tensors["strong_add_candidate"].bool(),
        "native_front": tensors["native_final_occ"].bool() & tensors["front_region"].bool(),
        "pruned_boundary": tensors["was_pruned_by_F3"].bool() | tensors["was_pruned_by_FrontCap"].bool(),
    }
    confs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    margins = [0.0, 0.02, 0.05, 0.10, 0.20]
    agrees = [None, 0.5, 0.65, 0.8]
    neighs = [0.0, 0.17, 0.34, 0.50]
    for name, fmask in filter_defs.items():
        for c in confs:
            for m in margins:
                for a in agrees:
                    for nb in neighs:
                        mask = mask0 & fmask
                        mask &= tensors["raw_confidence"].float() >= c
                        mask &= tensors["raw_margin"].float() >= m
                        mask &= tensors["neighbor_occ_count_norm"].float() >= nb
                        if a is not None:
                            mask &= tensors["camera_view_agreement"].float() >= a
                        n = int(mask.sum().item())
                        if n < 1000:
                            continue
                        pos = int(label[mask].sum().item())
                        rows.append(
                            {
                                "filter": name,
                                "n": n,
                                "pos": pos,
                                "precision": sw14e.safe_div(pos, n),
                                "conf": c,
                                "margin": m,
                                "agreement": a,
                                "neighbor": nb,
                            }
                        )
    rows = sorted(rows, key=lambda r: (r["n"] >= 100000, r["precision"], r["n"]), reverse=True)
    best_100k = [r for r in rows if r["n"] >= 100000][:20]
    best_any = rows[:20]
    write_csv(REPORTS_DIR / "sw14d_b3_add_precision100k_rule_feasibility.csv", rows)
    return {"best_n_ge_100k": best_100k, "best_any_n_ge_1k": best_any}


def plot_topk(rows: list[dict[str, Any]]) -> None:
    wanted = ["b1_model", "b2_recall", "b3_hard_negative", "b3_blend_b2_rule"]
    plt.figure(figsize=(8, 4.5))
    for name in wanted:
        sub = [r for r in rows if r["score_name"] == name and r["split"] == "val"]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: int(r["topk"]))
        plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k add candidates")
    plt.ylabel("GT precision")
    plt.title("SW14D-B3 add precision@K")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "sw14d_b3_add_precision100k_rescue.png", dpi=180)
    plt.close()


def main() -> None:
    seed = 53
    seed_everything(seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_table = sw14e.load_tensor_table(sw14e.table_path("train"))
    val_table = sw14e.load_tensor_table(sw14e.table_path("val"))
    base_scores = torch.load(ARTIFACTS_DIR / "sw14d_b_batched_scores.pt", map_location="cpu", weights_only=False)
    train_tensors = train_table["tensors"]
    val_tensors = val_table["tensors"]

    train_b2 = get_b2_scores("train", len(train_tensors["GT_occ"]))
    val_b2 = get_b2_scores("val", len(val_tensors["GT_occ"]))
    train_cache = prepare_feature_cache(train_tensors, base_scores["train"]["add_scores"], train_b2)
    val_cache = prepare_feature_cache(val_tensors, base_scores["val"]["add_scores"], val_b2)

    model = Precision100KScorer(len(sw14e.NO_GT_FEATURES) + len(B3_EXTRA_FEATURES)).to(device)
    warm_idx = sample_indices(train_tensors, seed=seed, max_pos=900_000, max_neg=1_000_000)
    train_rows = train_one_stage(model, train_tensors, train_cache, warm_idx, device, seed, "warm", epochs=3, lr=2.0e-3)

    warm_scores = score_model(model, train_tensors, train_cache, device)
    train_mask = add_mask(train_tensors)
    train_label = train_tensors["GT_occ"].bool()
    add_idx = torch.nonzero(train_mask, as_tuple=False).flatten()
    order = add_idx[torch.argsort(warm_scores[add_idx], descending=True)]
    top = order[: min(300_000, len(order))]
    hard_neg = top[~train_label[top]][:180_000]
    hard_pos = top[train_label[top]][:180_000]
    hard_idx = sample_indices(train_tensors, seed=seed + 1, hard_neg=hard_neg, hard_pos=hard_pos, max_pos=900_000, max_neg=1_000_000)
    train_rows.extend(train_one_stage(model, train_tensors, train_cache, hard_idx, device, seed + 1, "hard_negative", epochs=4, lr=1.0e-3))

    train_scores = score_model(model, train_tensors, train_cache, device)
    val_scores = score_model(model, val_tensors, val_cache, device)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": sw14e.NO_GT_FEATURES + B3_EXTRA_FEATURES,
            "seed": seed,
            "target": "add precision@100K >= 0.9",
        },
        CHECKPOINT_DIR / "sw14d_b3_add_precision100k_rescue.pth",
    )
    torch.save(
        {"train_b3_add_scores": train_scores, "val_b3_add_scores": val_scores},
        ARTIFACTS_DIR / "sw14d_b3_add_precision100k_scores.pt",
    )

    train_rule = normalize_score(base_rule_score(train_tensors), add_mask(train_tensors))
    val_rule = normalize_score(base_rule_score(val_tensors), add_mask(val_tensors))
    val_b1 = normalize_score(base_scores["val"]["add_scores"], add_mask(val_tensors))
    val_b2n = normalize_score(val_b2, add_mask(val_tensors))
    val_b3n = normalize_score(val_scores, add_mask(val_tensors))
    train_b1 = normalize_score(base_scores["train"]["add_scores"], add_mask(train_tensors))
    train_b2n = normalize_score(train_b2, add_mask(train_tensors))
    train_b3n = normalize_score(train_scores, add_mask(train_tensors))
    score_sets_train = {
        "b1_model": base_scores["train"]["add_scores"],
        "b2_recall": train_b2,
        "b3_hard_negative": train_scores,
        "b3_blend_b2_rule": 0.70 * train_b3n + 0.20 * train_b2n + 0.10 * train_rule,
        "b3_blend_b1_b2": 0.55 * train_b3n + 0.25 * train_b2n + 0.20 * train_b1,
    }
    score_sets_val = {
        "b1_model": base_scores["val"]["add_scores"],
        "b2_recall": val_b2,
        "b3_hard_negative": val_scores,
        "b3_blend_b2_rule": 0.70 * val_b3n + 0.20 * val_b2n + 0.10 * val_rule,
        "b3_blend_b1_b2": 0.55 * val_b3n + 0.25 * val_b2n + 0.20 * val_b1,
    }
    topk = topk_rows(train_tensors, score_sets_train, "train") + topk_rows(val_tensors, score_sets_val, "val")
    write_csv(REPORTS_DIR / "sw14d_b3_add_precision100k_topk.csv", topk)
    write_csv(REPORTS_DIR / "sw14d_b3_add_precision100k_training_log.csv", train_rows)
    rule_feas = threshold_feasibility(val_tensors, val_cache)
    plot_topk(topk)

    val_100k = [r for r in topk if r["split"] == "val" and int(r["topk"]) == 100000]
    best_100k = max(val_100k, key=lambda r: float(r["precision"]))
    b2_100k = next(r for r in val_100k if r["score_name"] == "b2_recall")
    best_rule_100k = rule_feas["best_n_ge_100k"][0] if rule_feas["best_n_ge_100k"] else {}
    target_precision = 0.90
    target_count = 100_000
    if float(best_100k["precision"]) >= target_precision:
        decision = "ADD100K_R1_TARGET_REACHED"
    elif float(best_100k["precision"]) > float(b2_100k["precision"]) + 0.02:
        decision = "ADD100K_R2_IMPROVED_BUT_BELOW_TARGET"
    else:
        decision = "ADD100K_R3_NOT_REACHABLE_CURRENT_FEATURES"
    final = {
        "decision": decision,
        "target_precision": target_precision,
        "target_topk": target_count,
        "target_reached": bool(float(best_100k["precision"]) >= target_precision),
        "best_val_precision_at_100k": best_100k,
        "b2_val_precision_at_100k": b2_100k,
        "best_rule_n_ge_100k": best_rule_100k,
        "base_add_candidate_count_val": int(add_mask(val_tensors).sum().item()),
        "base_add_candidate_precision_val": float(val_tensors["GT_occ"][add_mask(val_tensors)].float().mean().item()),
        "val_positive_add_candidates": int(val_tensors["GT_occ"][add_mask(val_tensors)].sum().item()),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "interpretation": (
            "The current no-GT candidate features do not separate enough true add candidates to support "
            "precision@100K=0.9 unless target_reached is true."
        ),
        "recommended_next_action": (
            "Add stronger evidence features before retraining: true raw logits/margins, real F3/FrontCap survival "
            "logs, multi-frame temporal support, connected-component support, and query-level reliability features."
        ),
    }
    write_json(REPORTS_DIR / "sw14d_b3_add_precision100k_feasibility.json", {**rule_feas, "final": final})
    write_json(REPORTS_DIR / "sw14d_b3_add_precision100k_final_decision.json", final)
    write_md(
        REPORTS_DIR / "stage_sw14d_b3_add_precision100k_rescue_report.md",
        "\n".join(
            [
                "# SW14D-B3 Add Precision@100K Rescue",
                "",
                "This diagnostic trains a hard-negative add scorer on the existing candidate table and checks the requested precision@100K target. No eval_debug/core was run.",
                f"- decision: `{decision}`",
                f"- target: `{target_precision}` at top `{target_count}`",
                f"- best val precision@100K: `{best_100k}`",
                f"- B2 val precision@100K: `{b2_100k}`",
                f"- best rule with n>=100K: `{best_rule_100k}`",
                "- result is diagnostic only; SW13 remains the main result.",
                "",
            ]
        ),
    )
    print(f"[sw14d-b3] decision {decision} best100k={best_100k}", flush=True)


if __name__ == "__main__":
    main()
