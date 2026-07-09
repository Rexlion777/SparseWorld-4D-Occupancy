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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[5])))
H10_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h10_compact_sparse_ranker.py"
H11_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_i_generative_occupancy_completion/run_sw14h11_bev_completion_ranker.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

GRID_X = 200
GRID_Y = 200
GRID_Z = 16

TABLE_LOCAL_CHANNELS = [
    "raw_occ",
    "raw_confidence",
    "raw_margin",
    "raw_entropy",
    "native_final_occ",
    "after_F3_occ",
    "after_FrontCap_occ",
    "teacher_final_occ",
    "raw_but_final_empty",
    "was_pruned_by_F3",
    "was_pruned_by_FrontCap",
    "neighbor_occ_count_norm",
    "local_density_proxy",
    "temporal_consistency",
    "camera_view_agreement",
    "front_region",
    "future_h4h6",
]
COMPACT_LOCAL_CHANNELS = [
    "h7b_lean",
    "b3_add_norm",
    "h8_contributor_count_norm",
    "h8_dense_top1_score_norm",
    "h8_dense_top1_margin_norm",
    "h8_geometric_active_norm",
    "h8_occ_pred_nonempty_norm",
    "h9_geo_count_norm",
    "h9_geo_score_pass_count_norm",
    "h9_geo_top1_mean_norm",
    "h9_gate_fraction_norm",
    "h9_non_gate_geo_count_norm",
]
LOCAL_RADII = [1, 2, 3, 5]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h10 = load_module("sw14h10_for_h14", H10_SCRIPT)
h11 = load_module("sw14h11_for_h14", H11_SCRIPT)
sw14e = h10.sw14e


@dataclass(frozen=True)
class H14Config:
    seed: int = 714
    epochs: int = 12
    batch_size: int = 262_144
    score_batch_size: int = 1_048_576
    hidden_dim: int = 768
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    target_precision_at_100k: float = 0.90
    device: str = "cuda"
    feature_device: str = "cuda"


class LocalPoolRanker(nn.Module):
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
    return h10.normalize(obj)


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
    parser = argparse.ArgumentParser(description="SW14H14 multiscale local 3D pooling add ranker")
    parser.add_argument("--epochs", type=int, default=H14Config.epochs)
    parser.add_argument("--batch-size", type=int, default=H14Config.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=H14Config.hidden_dim)
    parser.add_argument("--device", default=H14Config.device)
    parser.add_argument("--feature-device", default=H14Config.feature_device)
    parser.add_argument("--reuse-features", action="store_true")
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


def feature_path(split: str) -> Path:
    return ARTIFACTS_DIR / f"sw14h14_multiscale_local_pool_features_{split}.pt"


def assign_grid(grid: torch.Tensor, channel: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, values: torch.Tensor) -> None:
    grid[channel, z.long(), x.long(), y.long()] = values.float()


def build_local_features(split: str, device: torch.device, reuse: bool) -> dict[str, Any]:
    path = feature_path(split)
    if reuse and path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    print(f"[h14] build local features split={split}", flush=True)
    data = h11.build_split(split)
    table_t = data["table"]["tensors"]
    compact = data["compact"]
    channels = TABLE_LOCAL_CHANNELS + COMPACT_LOCAL_CHANNELS
    names: list[str] = []
    for source in channels:
        names.append(f"{source}_center")
        for radius in LOCAL_RADII:
            names.append(f"{source}_r{radius}_max")
            names.append(f"{source}_r{radius}_avg")
            names.append(f"{source}_r{radius}_delta_max")
    out_parts: list[torch.Tensor] = []
    rows_parts: list[torch.Tensor] = []
    t0 = time.time()
    for gi, key in enumerate(data["keys"]):
        full_start, full_end = data["full_group_slices"][key]
        comp_start, comp_end = data["compact_group_slices"][key]
        if comp_end <= comp_start:
            continue
        full_idx = torch.arange(full_start, full_end)
        comp_idx = torch.arange(comp_start, comp_end)
        full_rows = full_idx
        comp_rows = compact["rows"][comp_idx]
        grid = torch.zeros((len(channels), GRID_Z, GRID_X, GRID_Y), dtype=torch.float32, device=device)
        fx = table_t["voxel_x"][full_rows].long().to(device)
        fy = table_t["voxel_y"][full_rows].long().to(device)
        fz = table_t["voxel_z"][full_rows].long().to(device)
        channel_idx = 0
        for name in TABLE_LOCAL_CHANNELS:
            vals = table_t[name][full_rows].float().nan_to_num(0.0).to(device)
            assign_grid(grid, channel_idx, fx, fy, fz, vals)
            channel_idx += 1
        cx = table_t["voxel_x"][comp_rows].long().to(device)
        cy = table_t["voxel_y"][comp_rows].long().to(device)
        cz = table_t["voxel_z"][comp_rows].long().to(device)
        for name in COMPACT_LOCAL_CHANNELS:
            vals = compact["features"][name][comp_idx].float().nan_to_num(0.0).to(device)
            assign_grid(grid, channel_idx, cx, cy, cz, vals)
            channel_idx += 1
        gathered = [grid[:, cz, cx, cy].transpose(0, 1)]
        vol = grid.unsqueeze(0)
        for radius in LOCAL_RADII:
            kernel = 2 * radius + 1
            maxed = F.max_pool3d(vol, kernel_size=kernel, stride=1, padding=radius).squeeze(0)
            avged = F.avg_pool3d(vol, kernel_size=kernel, stride=1, padding=radius).squeeze(0)
            center = gathered[0]
            max_g = maxed[:, cz, cx, cy].transpose(0, 1)
            avg_g = avged[:, cz, cx, cy].transpose(0, 1)
            gathered.extend([max_g, avg_g, (max_g - center).clamp_min(0.0)])
        feat = torch.cat(gathered, dim=1).detach().cpu().half()
        out_parts.append(feat)
        rows_parts.append(comp_rows.detach().cpu().long())
        if gi % 50 == 0:
            print(f"[h14] split={split} group={gi + 1}/{len(data['keys'])} rows={sum(len(x) for x in rows_parts)}", flush=True)
    features = torch.cat(out_parts, dim=0) if out_parts else torch.zeros((0, len(names)), dtype=torch.float16)
    rows = torch.cat(rows_parts, dim=0) if rows_parts else torch.zeros((0,), dtype=torch.long)
    if not torch.equal(rows, data["compact"]["rows"].long()):
        raise RuntimeError(f"{split} H14 row order mismatch")
    payload = {
        "split": split,
        "rows": rows,
        "features": features,
        "feature_names": names,
        "channels": channels,
        "radii": LOCAL_RADII,
        "seconds": float(time.time() - t0),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
    }
    torch.save(payload, path)
    return payload


def compact_matrix(data: dict[str, Any], local: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    base = h10.compact_matrix(data, idx)
    return torch.cat([base, local[idx].float().nan_to_num(0.0)], dim=1)


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, max_pairs: int = 65_536) -> torch.Tensor:
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    neg_prior = prior[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    pos_sel = pos[torch.randperm(len(pos), device=logits.device)[:k]]
    neg_sel = neg[torch.topk(neg_prior, k=k).indices]
    return F.softplus(0.8 - pos_sel + neg_sel).mean()


def topk_surrogate_loss(logits: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, frac: float = 0.06) -> torch.Tensor:
    k = min(len(logits), max(8192, int(len(logits) * frac)))
    top = torch.unique(torch.cat([torch.topk(logits, k=k).indices, torch.topk(prior, k=k).indices]))
    weight = 1.0 + 5.0 * prior[top] + 5.0 * (1.0 - labels[top])
    loss = F.binary_cross_entropy_with_logits(logits[top], labels[top], reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def train_model(train: dict[str, Any], train_local: torch.Tensor, cfg: H14Config, device: torch.device) -> tuple[LocalPoolRanker, list[dict[str, Any]]]:
    in_dim = len(train["feature_names"]) + train_local.shape[1]
    model = LocalPoolRanker(in_dim, cfg.hidden_dim).to(device)
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
            xb = compact_matrix(train, train_local, b)
            prior = prior_all[b]
            if device.type == "cuda":
                try:
                    xb = xb.pin_memory()
                    prior = prior.pin_memory()
                except RuntimeError:
                    pass
            xb = xb.to(device, non_blocking=True)
            yb = y_all[b].to(device, non_blocking=True)
            prior_gpu = prior.to(device, non_blocking=True)
            front = front_all[b].to(device, non_blocking=True)
            future = future_all[b].to(device, non_blocking=True)
            h9v = h9_all[b].to(device, non_blocking=True)
            density = density_all[b].to(device, non_blocking=True)
            logits = model(xb, prior_gpu)
            sample_weight = 1.0 + yb * (2.0 + 1.3 * front + 1.2 * future + 0.8 * h9v) + (1.0 - yb) * (4.0 * prior_gpu + 1.8 * h9v + 1.5 * density)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            bce = (bce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            rank = pairwise_rank_loss(logits, yb, prior_gpu)
            topk = topk_surrogate_loss(logits, yb, prior_gpu)
            prob = torch.sigmoid(logits)
            fp_penalty = (prob * (1.0 - yb) * (1.0 + 3.0 * prior_gpu) * (1.0 + h9v + density)).mean()
            residual = logits - torch.logit(prior_gpu.clamp(1.0e-4, 1.0 - 1.0e-4))
            loss = bce + 3.0 * rank + 4.2 * topk + 1.0 * fp_penalty + 0.006 * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 6.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_rows": int(len(y_all)),
            "batch_size": cfg.batch_size,
            "device": str(device),
            "seconds": float(time.time() - t0),
        }
        log.append(row)
        print(f"[h14] epoch={epoch} loss={row['loss']:.5f} sec={row['seconds']:.1f}", flush=True)
    return model.eval(), log


@torch.inference_mode()
def score_model(model: LocalPoolRanker, data: dict[str, Any], local: torch.Tensor, cfg: H14Config, device: torch.device) -> torch.Tensor:
    out = torch.empty((len(data["label"]),), dtype=torch.float32)
    prior_all = data["prior"].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    idx_all = torch.arange(len(out))
    for start in range(0, len(idx_all), cfg.score_batch_size):
        idx = idx_all[start : start + cfg.score_batch_size]
        x = compact_matrix(data, local, idx)
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


def topk_rows(split_data: dict[str, Any], scores: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    return h10.topk_rows(split_data, scores)


def plot_precision(rows: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    for name in ["h7b_lean", "h14_local_pool_ranker", "h14_blend_h7b_10", "h14_blend_h7b_20", "h14_blend_h7b_35", "h14_best_val"]:
        sub = sorted([r for r in rows if r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H14 multiscale local pool add precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h14_multiscale_local_pool_precision.png", dpi=180)
    plt.close()


def run_selection(val: dict[str, Any], add_score: torch.Tensor) -> dict[str, Any]:
    return h10.run_selection(val, h10.full_score_for_selection(val, add_score), val["suppress_scores"])


def main() -> None:
    args = parse_args()
    cfg = H14Config(epochs=args.epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim, device=args.device, feature_device=args.feature_device)
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    feature_device = torch.device("cuda" if cfg.feature_device == "cuda" and torch.cuda.is_available() else "cpu")
    train_local_payload = build_local_features("train", feature_device, args.reuse_features)
    val_local_payload = build_local_features("val", feature_device, args.reuse_features)
    train = h10.build_compact_split("train")
    val = h10.build_compact_split("val")
    train_local = train_local_payload["features"]
    val_local = val_local_payload["features"]
    write_json(
        REPORTS_DIR / "sw14h14_multiscale_local_pool_config.json",
        {
            "config": asdict(cfg),
            "table_local_channels": TABLE_LOCAL_CHANNELS,
            "compact_local_channels": COMPACT_LOCAL_CHANNELS,
            "local_radii": LOCAL_RADII,
            "local_feature_count": int(train_local.shape[1]),
            "base_feature_count": len(train["feature_names"]),
            "train_stats": train["stats"],
            "val_stats": val["stats"],
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
        },
    )
    model, log = train_model(train, train_local, cfg, device)
    write_csv(REPORTS_DIR / "sw14h14_multiscale_local_pool_training_log.csv", log)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "resume_claim": False}, CHECKPOINT_DIR / "sw14h14_multiscale_local_pool_ranker_best.pth")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    val_score = score_model(model, val, val_local, cfg, device)
    h14_norm = h10.norm(val_score)
    h7b = val["prior"].float()
    score_map = {
        "h7b_lean": h7b,
        "h14_local_pool_ranker": val_score,
        "h14_blend_h7b_10": 0.10 * h14_norm + 0.90 * h7b,
        "h14_blend_h7b_20": 0.20 * h14_norm + 0.80 * h7b,
        "h14_blend_h7b_35": 0.35 * h14_norm + 0.65 * h7b,
        "h14_blend_h7b_50": 0.50 * h14_norm + 0.50 * h7b,
        "h14_blend_h7b_65": 0.65 * h14_norm + 0.35 * h7b,
        "h14_blend_h7b_80": 0.80 * h14_norm + 0.20 * h7b,
    }
    topk = topk_rows(val, score_map)
    best_val_100k = max((r for r in topk if int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    h7b_val_100k = next(r for r in topk if r["score_name"] == "h7b_lean" and int(r["topk"]) == 100000)
    score_map["h14_best_val"] = score_map[best_val_100k["score_name"]]
    plot_precision(topk + topk_rows(val, {"h14_best_val": score_map["h14_best_val"]}))
    write_csv(REPORTS_DIR / "sw14h14_multiscale_local_pool_topk_precision.csv", topk)
    torch.save({"val_sw14h14_local_pool_scores": val_score, "val_row_index": val["rows"], "feature_names": train_local_payload["feature_names"]}, ARTIFACTS_DIR / "sw14h14_multiscale_local_pool_scores.pt")
    selection = run_selection(val, score_map[best_val_100k["score_name"]])
    gain = float(best_val_100k["precision"]) - float(h7b_val_100k["precision"])
    if float(best_val_100k["precision"]) >= cfg.target_precision_at_100k:
        decision = "SW14H14_1_TARGET_090_REACHED"
    elif gain >= 0.10:
        decision = "SW14H14_2_MATERIAL_LOCAL_POOL_GAIN"
    elif gain > 0.005:
        decision = "SW14H14_3_SMALL_LOCAL_POOL_GAIN"
    else:
        decision = "SW14H14_4_LOCAL_POOL_NOT_ENOUGH"
    final = {
        "decision": decision,
        "target_precision_at_100k": cfg.target_precision_at_100k,
        "target_reached": bool(float(best_val_100k["precision"]) >= cfg.target_precision_at_100k),
        "best_val_top100k": best_val_100k,
        "h7b_val_top100k": h7b_val_100k,
        "precision_gain_over_h7b": gain,
        "selection": selection,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
    }
    write_json(REPORTS_DIR / "sw14h14_multiscale_local_pool_final_decision.json", final)
    print(f"[h14] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
