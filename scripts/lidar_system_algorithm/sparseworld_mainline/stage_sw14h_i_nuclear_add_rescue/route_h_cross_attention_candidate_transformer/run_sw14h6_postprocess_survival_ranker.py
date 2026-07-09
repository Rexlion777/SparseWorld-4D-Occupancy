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
from types import SimpleNamespace
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
SW13C_FIX_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/run_sw13c_fix_main.py"
FRONTCAP50_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/run_sw13c_fix_frontcap_main.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

GRID_SHAPE = (200, 200, 16)
EMPTY_IDX = 17


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_module("sw14hi_h6", SW14HI_SCRIPT)
sw14e = sw14hi.sw14e
sw13c_fix = load_module("sw14h6_sw13c_fix", SW13C_FIX_SCRIPT)
frontcap50 = load_module("sw14h6_frontcap50", FRONTCAP50_SCRIPT)


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

BASE_FEATURES = sw14hi.H_QUERY_FEATURES + SURVIVAL_FEATURES


@dataclass(frozen=True)
class H6Config:
    seed: int = 109
    epochs: int = 8
    batch_size: int = 262_144
    score_batch_size: int = 786_432
    train_max_pos: int = 1_200_000
    train_max_neg: int = 1_600_000
    hard_neg_topk: int = 1_100_000
    hidden_dim: int = 512
    lr: float = 1.2e-3
    target_precision_at_100k: float = 0.90
    device: str = "cuda"


class SurvivalResidualRanker(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.04),
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
        last = self.backbone[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        residual = self.backbone(x).squeeze(-1)
        prior_logit = torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
        return prior_logit + residual


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


def candidate() -> SimpleNamespace:
    return SimpleNamespace(
        protected_variant="PZ_fix_3_strong_core",
        wrong_class_aware=False,
        front_bias=1.0,
        expansion_ratio=0.12,
        cap_ratio=1.3,
        perturbation_id="A10_drop_front_triplet",
    )


def sector_maps() -> dict[str, torch.Tensor]:
    coords = sw14e.build_coords()
    return {"front": coords["front"].bool()}


def contiguous_group_slices(keys: torch.Tensor) -> list[tuple[int, int, int]]:
    keys_i = keys.to(torch.int64)
    if len(keys_i) == 0:
        return []
    change = torch.nonzero(keys_i[1:] != keys_i[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([torch.zeros((1,), dtype=torch.long), change])
    ends = torch.cat([change, torch.tensor([len(keys_i)], dtype=torch.long)])
    return [(int(keys_i[s].item()), int(s.item()), int(e.item())) for s, e in zip(starts, ends)]


def safe_minmax_norm(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(values, dtype=torch.float32)
    if not bool(valid.any().item()):
        return out
    vals = values[valid].float()
    vmin = vals.min()
    vmax = vals.max()
    out[valid] = (values[valid].float() - vmin) / (vmax - vmin + 1.0e-6)
    return out.clamp(0.0, 1.0)


def replay_postprocess(th: dict[str, torch.Tensor], horizon_s: int, sectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    cand = candidate()
    raw = th["teacher_raw_semantic"].long()
    native = th["native_semantic"].long()
    teacher_final = th["teacher_final_semantic"].long()
    conf = th["teacher_confidence"].float()
    margin = th["teacher_margin"].float()
    agreement = th["agreement"].float()
    raw_occ = raw != EMPTY_IDX
    native_occ = native != EMPTY_IDX
    raw_delta = raw_occ & ~native_occ
    protected = sw13c_fix.protected_zone_fix(
        cand.protected_variant,
        raw_occ,
        raw_delta,
        conf,
        margin,
        agreement,
        sectors,
        horizon_s,
    )
    low_value = sw13c_fix.low_value_score(
        raw_occ,
        protected,
        conf,
        margin,
        agreement,
        sectors,
        horizon_s,
        cand.wrong_class_aware,
        cand.front_bias,
    )
    f3_semantic, f3_pruned, f3_meta = sw13c_fix.apply_pruning_no_gt(
        raw_semantic=raw,
        protected_mask=protected,
        low_value_score_map=low_value,
        native_occ_count=int(native_occ.sum().item()),
        raw_occ_count=int(raw_occ.sum().item()),
        raw_delta_count=int(raw_delta.sum().item()),
        budget_mode="native_expansion_ratio",
        expansion_ratio=cand.expansion_ratio,
        keep_ratio=None,
    )
    final_semantic, front_pruned, cap_meta = frontcap50.apply_front_local_cap_no_gt(
        final_semantic_before_cap=f3_semantic.long(),
        native_semantic=native,
        raw_semantic=raw,
        protected_mask=protected,
        front_mask=sectors["front"].bool(),
        confidence=conf,
        margin=margin,
        agreement=agreement,
        low_value_score_map=low_value,
        cap_ratio=cand.cap_ratio,
        cap_mode="front_native_ratio_cap",
    )
    before_occ = f3_semantic.long() != EMPTY_IDX
    support = sw13c_fix.local_support(before_occ).float()
    conf_bad = 1.0 - conf.clamp(0.0, 1.0)
    margin_bad = 1.0 - torch.clamp(margin / (float(margin.max().item()) + 1.0e-6), 0.0, 1.0)
    cap_priority = low_value + 0.8 * conf_bad + 0.6 * margin_bad + 1.2 * (1.0 - agreement) + 0.9 * (support < 5).float()
    prunable_f3 = raw_occ & ~protected
    prunable_fc = sectors["front"].bool() & before_occ & ~protected
    f3_threshold = torch.tensor(float("inf"))
    if int(f3_pruned.sum().item()) > 0:
        f3_threshold = low_value[f3_pruned].float().min()
    fc_threshold = torch.tensor(float("inf"))
    if int(front_pruned.sum().item()) > 0:
        fc_threshold = cap_priority[front_pruned].float().min()
    replay_diff = (final_semantic.long() != teacher_final).bool()
    return {
        "raw_delta": raw_delta.bool(),
        "protected": protected.bool(),
        "f3_occ": (f3_semantic.long() != EMPTY_IDX),
        "final_occ": (final_semantic.long() != EMPTY_IDX),
        "f3_pruned": f3_pruned.bool(),
        "front_pruned": front_pruned.bool(),
        "low_value": low_value.float(),
        "low_value_norm": safe_minmax_norm(low_value.float(), prunable_f3),
        "f3_prune_margin": torch.where(
            torch.isfinite(f3_threshold),
            (low_value.float() - f3_threshold) / (f3_threshold.abs() + 1.0e-3),
            torch.full_like(low_value.float(), -1.0),
        ).clamp(-5.0, 5.0),
        "f3_budget_pressure": float(f3_meta.get("raw_occ_count", 0) - f3_meta.get("target_final_occ_count", f3_meta.get("raw_occ_count", 0))) / max(1.0, float(f3_meta.get("raw_occ_count", 1))),
        "frontcap_priority_norm": safe_minmax_norm(cap_priority.float(), prunable_fc),
        "frontcap_prune_margin": torch.where(
            torch.isfinite(fc_threshold),
            (cap_priority.float() - fc_threshold) / (fc_threshold.abs() + 1.0e-3),
            torch.full_like(cap_priority.float(), -1.0),
        ).clamp(-5.0, 5.0),
        "frontcap_budget_pressure": max(
            0.0,
            float(cap_meta.get("front_before_cap_occ_count", 0) - round(float(cap_meta.get("front_native_occ_count", 0)) * float(cap_meta.get("cap_ratio", 1.3)))),
        )
        / max(1.0, float(cap_meta.get("front_before_cap_occ_count", 1))),
        "replay_diff": replay_diff,
        "f3_meta": f3_meta,
        "cap_meta": cap_meta,
    }


def build_survival_cache_for_split(split: str, table: dict[str, Any], force: bool = False) -> dict[str, torch.Tensor]:
    path = ARTIFACTS_DIR / f"sw14h6_postprocess_survival_features_{split}.pt"
    if path.exists() and not force:
        return torch.load(path, map_location="cpu", weights_only=False)
    tensors = table["tensors"]
    n = len(tensors["group_key"])
    feat: dict[str, torch.Tensor] = {
        "h6_replay_raw_delta": torch.zeros((n,), dtype=torch.float16),
        "h6_replay_protected": torch.zeros((n,), dtype=torch.float16),
        "h6_replay_f3_occ": torch.zeros((n,), dtype=torch.float16),
        "h6_replay_final_occ": torch.zeros((n,), dtype=torch.float16),
        "h6_replay_f3_pruned": torch.zeros((n,), dtype=torch.float16),
        "h6_replay_frontcap_pruned": torch.zeros((n,), dtype=torch.float16),
        "h6_low_value_norm": torch.zeros((n,), dtype=torch.float16),
        "h6_f3_prune_margin": torch.zeros((n,), dtype=torch.float16),
        "h6_f3_budget_pressure": torch.zeros((n,), dtype=torch.float16),
        "h6_frontcap_priority_norm": torch.zeros((n,), dtype=torch.float16),
        "h6_frontcap_prune_margin": torch.zeros((n,), dtype=torch.float16),
        "h6_frontcap_budget_pressure": torch.zeros((n,), dtype=torch.float16),
        "h6_raw_to_f3_survival": torch.zeros((n,), dtype=torch.float16),
        "h6_f3_to_final_survival": torch.zeros((n,), dtype=torch.float16),
        "h6_replay_diff_teacher_final": torch.zeros((n,), dtype=torch.float16),
    }
    sectors = sector_maps()
    rows: list[dict[str, Any]] = []
    for group_key, start, end in contiguous_group_slices(tensors["group_key"]):
        sample_id = int(tensors["sample_id"][start].item())
        horizon_s = int(tensors["horizon_id"][start].item())
        bundle = sw14e.load_bundle(sample_id)
        replay = replay_postprocess(bundle["by_horizon"][horizon_s], horizon_s, sectors)
        xs = tensors["voxel_x"][start:end].long()
        ys = tensors["voxel_y"][start:end].long()
        zs = tensors["voxel_z"][start:end].long()
        sl = slice(start, end)
        feat["h6_replay_raw_delta"][sl] = replay["raw_delta"][xs, ys, zs].half()
        feat["h6_replay_protected"][sl] = replay["protected"][xs, ys, zs].half()
        feat["h6_replay_f3_occ"][sl] = replay["f3_occ"][xs, ys, zs].half()
        feat["h6_replay_final_occ"][sl] = replay["final_occ"][xs, ys, zs].half()
        feat["h6_replay_f3_pruned"][sl] = replay["f3_pruned"][xs, ys, zs].half()
        feat["h6_replay_frontcap_pruned"][sl] = replay["front_pruned"][xs, ys, zs].half()
        feat["h6_low_value_norm"][sl] = replay["low_value_norm"][xs, ys, zs].half()
        feat["h6_f3_prune_margin"][sl] = replay["f3_prune_margin"][xs, ys, zs].half()
        feat["h6_f3_budget_pressure"][sl] = torch.full((end - start,), float(replay["f3_budget_pressure"]), dtype=torch.float16)
        feat["h6_frontcap_priority_norm"][sl] = replay["frontcap_priority_norm"][xs, ys, zs].half()
        feat["h6_frontcap_prune_margin"][sl] = replay["frontcap_prune_margin"][xs, ys, zs].half()
        feat["h6_frontcap_budget_pressure"][sl] = torch.full((end - start,), float(replay["frontcap_budget_pressure"]), dtype=torch.float16)
        feat["h6_raw_to_f3_survival"][sl] = (feat["h6_replay_raw_delta"][sl].float() * feat["h6_replay_f3_occ"][sl].float()).half()
        feat["h6_f3_to_final_survival"][sl] = (feat["h6_replay_f3_occ"][sl].float() * feat["h6_replay_final_occ"][sl].float()).half()
        feat["h6_replay_diff_teacher_final"][sl] = replay["replay_diff"][xs, ys, zs].half()
        rows.append(
            {
                "split": split,
                "group_key": group_key,
                "sample_id": sample_id,
                "horizon_id": horizon_s,
                "rows": end - start,
                "f3_pruned_count": int(replay["f3_pruned"].sum().item()),
                "frontcap_pruned_count": int(replay["front_pruned"].sum().item()),
                "replay_diff_count": int(replay["replay_diff"].sum().item()),
                "frontcap_pressure": float(replay["frontcap_budget_pressure"]),
                "f3_pressure": float(replay["f3_budget_pressure"]),
            }
        )
        if len(rows) % 40 == 0:
            print(f"[h6] {split} replayed {len(rows)} groups", flush=True)
    torch.save(feat, path)
    write_csv(REPORTS_DIR / f"sw14h6_postprocess_survival_replay_{split}.csv", rows)
    summary = {
        "split": split,
        "groups": len(rows),
        "rows": n,
        "mean_replay_diff_count": float(np.mean([r["replay_diff_count"] for r in rows])) if rows else 0.0,
        "total_candidate_replay_diff_rows": int(feat["h6_replay_diff_teacher_final"].float().sum().item()),
        "feature_names": SURVIVAL_FEATURES,
        "uses_gt_as_inference_feature": False,
        "uses_eval_debug": False,
    }
    write_json(REPORTS_DIR / f"sw14h6_postprocess_survival_feature_summary_{split}.json", summary)
    return feat


def combined_cache(tensors: dict[str, torch.Tensor], scores: dict[str, torch.Tensor], survival: dict[str, torch.Tensor], h2_score: torch.Tensor) -> dict[str, torch.Tensor]:
    cache = sw14hi.derived_cache(tensors, scores)
    mask = add_mask(tensors)
    h2_norm = sw14hi.normalize_score(h2_score.float(), mask).nan_to_num(0.0, neginf=0.0)
    b3 = cache["b3_add_score_norm"].float().nan_to_num(0.0, neginf=0.0)
    cache["h6_prior_h2b3"] = (0.25 * h2_norm + 0.75 * b3).float()
    for name in SURVIVAL_FEATURES:
        cache[name] = survival[name].float().nan_to_num(0.0)
    return cache


def load_h2_scores(n_train: int, n_val: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = ARTIFACTS_DIR / "sw14h2_true_local_attention_scores.pt"
    if not path.exists():
        return torch.full((n_train,), -torch.inf), torch.full((n_val,), -torch.inf)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["train_sw14h2_add_scores"].float(), payload["val_sw14h2_add_scores"].float()


def feature_matrix(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    return sw14hi.query_matrix(tensors, cache, idx, BASE_FEATURES).float().nan_to_num(0.0)


def sample_train_indices(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: H6Config) -> torch.Tensor:
    roi = add_mask(tensors)
    label = tensors["GT_occ"].bool()
    pos = torch.nonzero(roi & label, as_tuple=False).flatten()
    neg = torch.nonzero(roi & ~label, as_tuple=False).flatten()
    hard_score = cache["h6_prior_h2b3"].float()
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


def pairwise_rank_loss(logits: torch.Tensor, y: torch.Tensor, max_pairs: int = 32768) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    return F.softplus(0.55 - pos[:k] + neg[:k]).mean()


def topk_surrogate_loss(logits: torch.Tensor, y: torch.Tensor, prior: torch.Tensor, frac: float = 0.30) -> torch.Tensor:
    if len(logits) == 0:
        return logits.new_tensor(0.0)
    k = min(len(logits), max(2048, int(len(logits) * frac)))
    union = torch.unique(torch.cat([torch.topk(logits, k=k).indices, torch.topk(prior, k=k).indices]))
    return F.binary_cross_entropy_with_logits(logits[union], y[union])


def train_ranker(
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    cfg: H6Config,
    device: torch.device,
) -> tuple[SurvivalResidualRanker, list[dict[str, Any]]]:
    idx = sample_train_indices(tensors, cache, cfg)
    x_cpu = feature_matrix(tensors, cache, idx)
    y_cpu = tensors["GT_occ"][idx].float()
    prior_cpu = cache["h6_prior_h2b3"][idx].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    front_cpu = tensors["front_region"][idx].float()
    future_cpu = tensors["future_h4h6"][idx].float()
    density_cpu = tensors["local_density_proxy"][idx].float()
    protected_cpu = tensors["protected_zone"][idx].float()
    if device.type == "cuda":
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
        prior_cpu = prior_cpu.pin_memory()
        front_cpu = front_cpu.pin_memory()
        future_cpu = future_cpu.pin_memory()
        density_cpu = density_cpu.pin_memory()
        protected_cpu = protected_cpu.pin_memory()
    model = SurvivalResidualRanker(x_cpu.shape[1], cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.5e-4)
    pos_weight = ((len(y_cpu) - float(y_cpu.sum().item())) / max(1.0, float(y_cpu.sum().item())))
    pos_weight_t = torch.tensor([pos_weight], device=device).clamp(1.0, 12.0)
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
            protected = protected_cpu[b].to(device, non_blocking=True)
            logits = model(xb, prior)
            weight = 1.0 + yb * (0.9 * front + 1.1 * future) + (1.0 - yb) * (2.0 * prior + 0.8 * density + 0.5 * protected)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight_t, reduction="none")
            bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
            rank = pairwise_rank_loss(logits, yb)
            topk = topk_surrogate_loss(logits, yb, prior)
            fp_top_penalty = (torch.sigmoid(logits) * (1.0 - yb) * (1.0 + prior) * (1.0 + density)).mean()
            residual = logits - torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
            residual_l2 = residual.square().mean()
            loss = bce + 2.5 * rank + 3.0 * topk + 0.6 * fp_top_penalty + 0.01 * residual_l2
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
                "device": str(device),
                "batch_size": cfg.batch_size,
            }
        )
        print(f"[h6] epoch={epoch} loss={rows[-1]['loss']:.5f}", flush=True)
    return model.eval(), rows


@torch.inference_mode()
def score_ranker(
    model: SurvivalResidualRanker,
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    cfg: H6Config,
    device: torch.device,
) -> torch.Tensor:
    roi = add_mask(tensors)
    idx = torch.nonzero(roi, as_tuple=False).flatten()
    out = torch.full((len(roi),), -torch.inf, dtype=torch.float32)
    for start in range(0, len(idx), cfg.score_batch_size):
        chunk = idx[start : start + cfg.score_batch_size]
        x = feature_matrix(tensors, cache, chunk)
        prior = cache["h6_prior_h2b3"][chunk].float().clamp(1.0e-4, 1.0 - 1.0e-4)
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
    for name in ["b3_baseline", "h2_prior", "h6_survival_ranker", "h6_blend_prior_25", "h6_blend_prior_50"]:
        sub = sorted([r for r in rows if r["split"] == "val" and r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H6 postprocess survival add precision")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "sw14h6_postprocess_survival_precision.png", dpi=180)
    plt.close()


def run_selection(tables: dict[str, dict[str, Any]], add_score: torch.Tensor, suppress_score: torch.Tensor) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for sup in [0.02, 0.03, 0.04]:
        for bal in [0.25, 0.50, 0.75, 1.0]:
            for fixed in [16, 32, 64, 96]:
                cfg = sw14e.Config(max_suppress_ratio=sup, add_suppress_balance=bal, add_fixed_budget=fixed)
                budget = {
                    "name": f"h6_sup{sup}_bal{bal}_fixed{fixed}",
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
    write_csv(REPORTS_DIR / "sw14h6_postprocess_survival_selection_sweep_val.csv", rows)
    return best or (max(rows, key=lambda r: float(r["mean_net_score"])) if rows else {})


def main() -> None:
    cfg = H6Config()
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")

    print("[h6] loading tables/scores", flush=True)
    tables = sw14hi.load_tables()
    score_payload = sw14hi.load_score_payload(tables)
    train_h2, val_h2 = load_h2_scores(len(tables["train"]["tensors"]["GT_occ"]), len(tables["val"]["tensors"]["GT_occ"]))

    print("[h6] building/reusing postprocess survival cache", flush=True)
    train_survival = build_survival_cache_for_split("train", tables["train"])
    val_survival = build_survival_cache_for_split("val", tables["val"])
    train_cache = combined_cache(tables["train"]["tensors"], score_payload["train"], train_survival, train_h2)
    val_cache = combined_cache(tables["val"]["tensors"], score_payload["val"], val_survival, val_h2)

    write_json(
        REPORTS_DIR / "sw14h6_postprocess_survival_config.json",
        {
            "model": "baseline-preserving postprocess-survival residual ranker",
            "base_features": BASE_FEATURES,
            "survival_features": SURVIVAL_FEATURES,
            "candidate": candidate().__dict__,
            "target_precision_at_100k": cfg.target_precision_at_100k,
            "gt_used_as_inference_feature": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "config": cfg.__dict__,
        },
    )

    print("[h6] training survival ranker", flush=True)
    model, train_log = train_ranker(tables["train"]["tensors"], train_cache, cfg, device)
    write_csv(REPORTS_DIR / "sw14h6_postprocess_survival_training_log.csv", train_log)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": BASE_FEATURES,
            "survival_features": SURVIVAL_FEATURES,
            "seed": cfg.seed,
            "resume_claim": False,
            "gt_used_as_inference_feature": False,
        },
        CHECKPOINT_DIR / "sw14h6_postprocess_survival_ranker_best.pth",
    )

    print("[h6] scoring train/val", flush=True)
    train_h6 = score_ranker(model, tables["train"]["tensors"], train_cache, cfg, device)
    val_h6 = score_ranker(model, tables["val"]["tensors"], val_cache, cfg, device)
    torch.save(
        {"train_sw14h6_add_scores": train_h6, "val_sw14h6_add_scores": val_h6},
        ARTIFACTS_DIR / "sw14h6_postprocess_survival_scores.pt",
    )

    train_prior = train_cache["h6_prior_h2b3"]
    val_prior = val_cache["h6_prior_h2b3"]
    train_h6_norm = sw14hi.normalize_score(train_h6, add_mask(tables["train"]["tensors"])).nan_to_num(0.0, neginf=0.0)
    val_h6_norm = sw14hi.normalize_score(val_h6, add_mask(tables["val"]["tensors"])).nan_to_num(0.0, neginf=0.0)
    train_scores = {
        "b3_baseline": score_payload["train"]["b3_add"],
        "h2_prior": train_prior,
        "h6_survival_ranker": train_h6,
        "h6_blend_prior_25": 0.25 * train_h6_norm + 0.75 * train_prior,
        "h6_blend_prior_50": 0.50 * train_h6_norm + 0.50 * train_prior,
    }
    val_scores = {
        "b3_baseline": score_payload["val"]["b3_add"],
        "h2_prior": val_prior,
        "h6_survival_ranker": val_h6,
        "h6_blend_prior_25": 0.25 * val_h6_norm + 0.75 * val_prior,
        "h6_blend_prior_50": 0.50 * val_h6_norm + 0.50 * val_prior,
    }
    topk = topk_rows(tables["train"]["tensors"], train_scores, "train") + topk_rows(tables["val"]["tensors"], val_scores, "val")
    write_csv(REPORTS_DIR / "sw14h6_postprocess_survival_topk_precision.csv", topk)
    plot_precision(topk)

    best_val_100k = max((r for r in topk if r["split"] == "val" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    best_train_100k = max((r for r in topk if r["split"] == "train" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    b3_val_100k = next(r for r in topk if r["split"] == "val" and r["score_name"] == "b3_baseline" and int(r["topk"]) == 100000)
    prior_val_100k = next(r for r in topk if r["split"] == "val" and r["score_name"] == "h2_prior" and int(r["topk"]) == 100000)
    selection = run_selection(tables, val_scores[best_val_100k["score_name"]], score_payload["val"]["suppress"])

    if float(best_val_100k["precision"]) >= cfg.target_precision_at_100k:
        decision = "SW14H6_1_TARGET_090_REACHED"
    elif float(best_val_100k["precision"]) >= float(prior_val_100k["precision"]) + 0.05:
        decision = "SW14H6_2_SURVIVAL_SIGNAL_MATERIAL"
    elif float(best_val_100k["precision"]) > float(prior_val_100k["precision"]) + 0.005:
        decision = "SW14H6_3_SURVIVAL_SIGNAL_SMALL"
    else:
        decision = "SW14H6_4_NO_SURVIVAL_GAIN"

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
        "recommended_next_action": "If H6 remains far below 0.9@100K, build richer raw-logit/query-level proposal data; the current semantic/confidence survival cache is insufficient for high-precision recall repair.",
    }
    write_json(REPORTS_DIR / "sw14h6_postprocess_survival_final_decision.json", final)
    print(f"[h6] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
