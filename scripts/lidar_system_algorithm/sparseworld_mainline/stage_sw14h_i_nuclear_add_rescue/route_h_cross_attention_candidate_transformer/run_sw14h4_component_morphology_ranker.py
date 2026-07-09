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
from scipy import ndimage
from torch import nn
import torch.nn.functional as F

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[5])))
SW14HI_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/run_sw14h_i_nuclear_add_rescue.py"
H3_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h3_dense_neighborhood_ranker.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_module(SW14HI_SCRIPT, "sw14hi_h4")
h3mod = load_module(H3_SCRIPT, "sw14h3_mod")
sw14e = sw14hi.sw14e
EMPTY_IDX = sw14e.EMPTY_IDX
TEACHER_CACHE = sw14e.TEACHER_CACHE


MORPH_FEATURE_NAMES = [
    "component_exists",
    "component_size_log",
    "component_size_norm",
    "component_conf_mean",
    "component_margin_mean",
    "component_agree_mean",
    "component_final_touch_ratio",
    "component_native_touch_ratio",
    "component_raw_density",
    "component_front_prior",
]


@dataclass(frozen=True)
class H4Config:
    seed: int = 109
    epochs: int = 6
    train_max_pos: int = 800_000
    train_max_neg: int = 1_000_000
    hard_neg_topk: int = 800_000
    batch_size: int = 262_144
    score_batch_size: int = 524_288
    hidden_dim: int = 320
    lr: float = 1.2e-3
    device: str = "cuda"


class MorphResidualRanker(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
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


def occ(x: torch.Tensor) -> np.ndarray:
    return (x.long() != EMPTY_IDX).cpu().numpy().astype(bool)


def morph_features_for_group(th: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, front_flag: torch.Tensor) -> torch.Tensor:
    raw = occ(th["teacher_raw_semantic"])
    final = occ(th["teacher_final_semantic"])
    native = occ(th["native_semantic"])
    raw_empty = raw & ~final
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, num = ndimage.label(raw_empty, structure=structure)
    lab = labels[x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()].astype(np.int64)
    if num == 0:
        return torch.zeros((len(x), len(MORPH_FEATURE_NAMES)), dtype=torch.float16)
    flat_lab = labels.reshape(-1)
    valid = flat_lab > 0
    size = np.bincount(flat_lab[valid], minlength=num + 1).astype(np.float64)
    conf = th["teacher_confidence"].float().cpu().numpy().reshape(-1)
    margin = th["teacher_margin"].float().cpu().numpy().reshape(-1)
    agree = th["agreement"].float().cpu().numpy().reshape(-1)
    raw_flat = raw.reshape(-1)
    final_touch = ndimage.binary_dilation(final, structure=structure) & raw_empty
    native_touch = ndimage.binary_dilation(native, structure=structure) & raw_empty
    conf_sum = np.bincount(flat_lab[valid], weights=conf[valid], minlength=num + 1)
    margin_sum = np.bincount(flat_lab[valid], weights=margin[valid], minlength=num + 1)
    agree_sum = np.bincount(flat_lab[valid], weights=agree[valid], minlength=num + 1)
    final_touch_sum = np.bincount(flat_lab[valid], weights=final_touch.reshape(-1)[valid].astype(np.float64), minlength=num + 1)
    native_touch_sum = np.bincount(flat_lab[valid], weights=native_touch.reshape(-1)[valid].astype(np.float64), minlength=num + 1)
    denom = np.maximum(size, 1.0)
    comp_exists = (lab > 0).astype(np.float32)
    comp_size = size[lab]
    feat = np.stack(
        [
            comp_exists,
            np.log1p(comp_size) / np.log(640000.0),
            np.clip(comp_size / 512.0, 0.0, 1.0),
            conf_sum[lab] / np.maximum(size[lab], 1.0),
            margin_sum[lab] / np.maximum(size[lab], 1.0),
            agree_sum[lab] / np.maximum(size[lab], 1.0),
            final_touch_sum[lab] / np.maximum(size[lab], 1.0),
            native_touch_sum[lab] / np.maximum(size[lab], 1.0),
            np.clip(comp_size / 75.0, 0.0, 1.0),
            front_flag.float().cpu().numpy().astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    feat[lab == 0] = 0.0
    return torch.from_numpy(feat).half()


def build_morph_table(split: str, h3_data: dict[str, torch.Tensor], candidate_table: dict[str, Any]) -> torch.Tensor:
    tensors = candidate_table["tensors"]
    rows = h3_data["rows"].long()
    group_keys = tensors["group_key"][rows].to(torch.int32)
    morph_parts: list[torch.Tensor] = []
    for i, key in enumerate(torch.unique(group_keys)):
        local = torch.nonzero(group_keys == key, as_tuple=False).flatten()
        global_rows = rows[local]
        sample_id = int(key.item()) // 10
        horizon_id = int(key.item()) % 10
        th = load_bundle(sample_id)["by_horizon"][horizon_id]
        feat = morph_features_for_group(
            th,
            tensors["voxel_x"][global_rows].long(),
            tensors["voxel_y"][global_rows].long(),
            tensors["voxel_z"][global_rows].long(),
            tensors["front_region"][global_rows].float(),
        )
        morph_parts.append(feat)
        if (i + 1) % 50 == 0:
            print(f"[h4] {split} morph groups {i + 1} rows {sum(len(v) for v in morph_parts)}", flush=True)
    return torch.cat(morph_parts, dim=0)


def sample_indices(y: torch.Tensor, prior: torch.Tensor, cfg: H4Config) -> torch.Tensor:
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


def pairwise_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), 32768)
    return F.softplus(0.55 - pos[:k] + neg[:k]).mean()


def topk_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    k = min(len(logits), max(512, int(len(logits) * 0.30)))
    top = torch.topk(logits, k=k).indices
    return F.binary_cross_entropy_with_logits(logits[top], y[top])


def train_model(x_all: torch.Tensor, y_all: torch.Tensor, prior_all: torch.Tensor, cfg: H4Config, device: torch.device) -> tuple[MorphResidualRanker, list[dict[str, Any]]]:
    idx = sample_indices(y_all, prior_all, cfg)
    x = x_all[idx].float()
    y = y_all[idx].float()
    prior = prior_all[idx].float()
    if device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()
        prior = prior.pin_memory()
    model = MorphResidualRanker(x.shape[1], cfg.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=2.0e-4)
    rows: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 100 + epoch)
        order = torch.randperm(len(y), generator=gen)
        losses: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            xb = x[b].to(device, non_blocking=True)
            yb = y[b].to(device, non_blocking=True)
            pb = prior[b].to(device, non_blocking=True)
            logits = model(xb, pb)
            w = 1.0 + yb * 0.9 + (1.0 - yb) * (1.0 + 1.7 * pb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce = (bce * w).sum() / w.sum().clamp_min(1.0)
            rank = pairwise_loss(logits, yb)
            topk = topk_loss(logits, yb)
            residual = (logits - torch.logit(pb.clamp(1.0e-4, 1 - 1.0e-4))).square().mean()
            loss = bce + 2.0 * rank + 3.0 * topk + 0.008 * residual
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        rows.append({"epoch": epoch, "loss": float(np.mean(losses)), "train_rows": int(len(y)), "positive_rate": float(y.mean().item())})
    return model.eval(), rows


@torch.inference_mode()
def score_model(model: MorphResidualRanker, x: torch.Tensor, prior: torch.Tensor, cfg: H4Config, device: torch.device) -> torch.Tensor:
    out = torch.empty((len(prior),), dtype=torch.float32)
    for start in range(0, len(out), cfg.score_batch_size):
        end = min(start + cfg.score_batch_size, len(out))
        xb = x[start:end].float()
        pb = prior[start:end].float()
        if device.type == "cuda":
            xb = xb.pin_memory()
            pb = pb.pin_memory()
        out[start:end] = torch.sigmoid(model(xb.to(device, non_blocking=True), pb.to(device, non_blocking=True))).detach().cpu()
    return out


def topk_rows(data: dict[str, torch.Tensor], table: dict[str, Any], scores: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    rows_global = data["rows"].long()
    y = table["tensors"]["GT_occ"].bool()
    front = table["tensors"]["front_region"].bool()
    future = table["tensors"]["future_h4h6"].bool()
    out: list[dict[str, Any]] = []
    for name, score in scores.items():
        ordered = rows_global[torch.argsort(score, descending=True)]
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000]:
            if len(ordered) < k:
                continue
            top = ordered[:k]
            out.append(
                {
                    "split": split,
                    "score_name": name,
                    "topk": k,
                    "precision": float(y[top].float().mean().item()),
                    "positive_count": int(y[top].sum().item()),
                    "front_precision": float(y[top][front[top]].float().mean().item()) if bool(front[top].any().item()) else 0.0,
                    "future_precision": float(y[top][future[top]].float().mean().item()) if bool(future[top].any().item()) else 0.0,
                    "front_count": int(front[top].sum().item()),
                    "future_count": int(future[top].sum().item()),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    h3mod.write_csv(path, rows)


def write_json(path: Path, payload: Any) -> None:
    h3mod.write_json(path, payload)


def plot(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 4.5))
    for name in ["h2b3_prior", "h4_morph", "h4_blend_prior_25", "h4_blend_prior_50"]:
        sub = sorted([r for r in rows if r["split"] == "val" and r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H4 morphology precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h4_component_morphology_precision.png", dpi=180)
    plt.close()


def main() -> None:
    cfg = H4Config()
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    tables = sw14hi.load_tables()
    h3 = torch.load(ARTIFACTS_DIR / "sw14h3_dense_neighborhood_feature_table.pt", map_location="cpu", weights_only=False)
    print("[h4] building train morphology", flush=True)
    train_morph = build_morph_table("train", h3["train"], tables["train"])
    print("[h4] building val morphology", flush=True)
    val_morph = build_morph_table("val", h3["val"], tables["val"])
    train_x = torch.cat([h3["train"]["x"], train_morph], dim=1)
    val_x = torch.cat([h3["val"]["x"], val_morph], dim=1)
    train_prior = h3["train"]["prior"].float()
    val_prior = h3["val"]["prior"].float()
    torch.save({"train_morph": train_morph, "val_morph": val_morph, "feature_names": MORPH_FEATURE_NAMES}, ARTIFACTS_DIR / "sw14h4_component_morphology_features.pt")
    write_json(REPORTS_DIR / "sw14h4_component_morphology_config.json", {"feature_names": MORPH_FEATURE_NAMES, "gt_used_as_inference_feature": False, "target": "val add precision@100K >= 0.9", "config": cfg.__dict__})
    print("[h4] training", flush=True)
    model, train_log = train_model(train_x, h3["train"]["y"].float(), train_prior, cfg, device)
    write_csv(REPORTS_DIR / "sw14h4_component_morphology_training_log.csv", train_log)
    torch.save({"state_dict": model.state_dict(), "feature_names": h3["feature_names"] + MORPH_FEATURE_NAMES, "seed": cfg.seed}, CHECKPOINT_DIR / "sw14h4_component_morphology_ranker_best.pth")
    print("[h4] scoring", flush=True)
    train_h4 = score_model(model, train_x, train_prior, cfg, device)
    val_h4 = score_model(model, val_x, val_prior, cfg, device)
    torch.save({"train_sw14h4_add_scores": train_h4, "val_sw14h4_add_scores": val_h4}, ARTIFACTS_DIR / "sw14h4_component_morphology_scores.pt")
    train_scores = {
        "h2b3_prior": train_prior,
        "h4_morph": train_h4,
        "h4_blend_prior_25": 0.25 * train_h4 + 0.75 * train_prior,
        "h4_blend_prior_50": 0.50 * train_h4 + 0.50 * train_prior,
    }
    val_scores = {
        "h2b3_prior": val_prior,
        "h4_morph": val_h4,
        "h4_blend_prior_25": 0.25 * val_h4 + 0.75 * val_prior,
        "h4_blend_prior_50": 0.50 * val_h4 + 0.50 * val_prior,
    }
    rows = topk_rows(h3["train"], tables["train"], train_scores, "train") + topk_rows(h3["val"], tables["val"], val_scores, "val")
    write_csv(REPORTS_DIR / "sw14h4_component_morphology_topk_precision.csv", rows)
    plot(rows)
    best_train = max((r for r in rows if r["split"] == "train" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    best_val = max((r for r in rows if r["split"] == "val" and int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    prior_val = next(r for r in rows if r["split"] == "val" and r["score_name"] == "h2b3_prior" and int(r["topk"]) == 100000)
    target = 0.9
    decision = "SW14H4_1_TARGET_090_REACHED" if float(best_val["precision"]) >= target else (
        "SW14H4_2_MORPH_VAL_IMPROVED_BELOW_090" if float(best_val["precision"]) > float(prior_val["precision"]) + 0.02 else "SW14H4_3_NO_MORPH_GAIN"
    )
    final = {
        "decision": decision,
        "target_precision_at_100k": target,
        "target_reached": bool(float(best_val["precision"]) >= target),
        "best_train_top100k": best_train,
        "best_val_top100k": best_val,
        "prior_val_top100k": prior_val,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "gt_used_as_inference_feature": False,
        "recommended_next_action": "If target is not reached, the remaining path needs richer raw logits / postprocess survival reason features, not more table-only morphology.",
    }
    write_json(REPORTS_DIR / "sw14h4_component_morphology_final_decision.json", final)
    print(f"[h4] decision={decision} best_val_100k={best_val}", flush=True)


if __name__ == "__main__":
    main()
