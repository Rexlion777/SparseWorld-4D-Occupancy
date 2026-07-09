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
SW14H10_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h10_compact_sparse_ranker.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

GRID_X = 200
GRID_Y = 200
GRID_Z = 16

TABLE_DENSE_CHANNELS = [
    "raw_occ",
    "teacher_final_occ",
    "raw_confidence",
    "raw_margin",
    "camera_view_agreement",
    "neighbor_occ_count_norm",
    "local_density_proxy",
    "front_region",
    "future_h4h6",
    "strong_add_candidate",
    "medium_add_candidate",
]
COMPACT_CHANNELS = [
    "h7b_lean",
    "b3_add_norm",
    "h8_contributor_count_norm",
    "h8_geometric_active_norm",
    "h8_occ_pred_nonempty_norm",
    "h9_geo_count_norm",
    "h9_geo_score_pass_count_norm",
    "h9_geo_top1_mean_norm",
    "h9_gate_fraction_norm",
    "h9_non_gate_geo_count_norm",
    "h8_any_h7b",
    "h9_geo_count_h7b",
    "h9_score_pass_h7b",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14hi = load_module("sw14hi_h11_bev", SW14HI_SCRIPT)
h10 = load_module("sw14h10_for_h11_bev", SW14H10_SCRIPT)
sw14e = sw14hi.sw14e


@dataclass(frozen=True)
class H11Config:
    seed: int = 503
    epochs: int = 5
    batch_groups: int = 4
    base_channels: int = 96
    lr: float = 8.0e-4
    weight_decay: float = 1.0e-4
    target_precision_at_100k: float = 0.90
    device: str = "cuda"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class BEVCompletionNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 96) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, base_channels), nn.SiLU(), ResidualBlock(base_channels))
        self.down2 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, base_channels * 2), nn.SiLU(), ResidualBlock(base_channels * 2))
        self.mid = nn.Sequential(ResidualBlock(base_channels * 2), ResidualBlock(base_channels * 2))
        self.up1 = nn.Sequential(nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1), nn.GroupNorm(8, base_channels), nn.SiLU(), ResidualBlock(base_channels))
        self.up0 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1), nn.GroupNorm(8, base_channels), nn.SiLU(), ResidualBlock(base_channels))
        self.head = nn.Conv2d(base_channels, GRID_Z, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        m = self.mid(s2)
        u1 = F.interpolate(m, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, s1], dim=1))
        u0 = F.interpolate(u1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, s0], dim=1))
        return self.head(u0)


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
    parser = argparse.ArgumentParser(description="SW14H11 BEV occupancy completion add proposal ranker")
    parser.add_argument("--epochs", type=int, default=H11Config.epochs)
    parser.add_argument("--batch-groups", type=int, default=H11Config.batch_groups)
    parser.add_argument("--base-channels", type=int, default=H11Config.base_channels)
    parser.add_argument("--device", default=H11Config.device)
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


def group_slices(group_key: torch.Tensor) -> list[tuple[int, int, int]]:
    key = group_key.long()
    change = torch.nonzero(key[1:] != key[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([torch.zeros((1,), dtype=torch.long), change])
    ends = torch.cat([change, torch.tensor([len(key)], dtype=torch.long)])
    return [(int(key[s].item()), int(s.item()), int(e.item())) for s, e in zip(starts, ends)]


def build_split(split: str) -> dict[str, Any]:
    compact = h10.build_compact_split(split)
    table = compact["table"]
    tensors = table["tensors"]
    full_group_slices = {key: (start, end) for key, start, end in group_slices(tensors["group_key"])}
    compact_group_slices = {key: (start, end) for key, start, end in group_slices(compact["group_key"])}
    keys = [key for key in compact_group_slices if key in full_group_slices]
    return {
        "split": split,
        "compact": compact,
        "table": table,
        "keys": keys,
        "full_group_slices": full_group_slices,
        "compact_group_slices": compact_group_slices,
        "stats": compact["stats"],
    }


def assign_grid(grid: torch.Tensor, channel: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, values: torch.Tensor) -> None:
    grid[channel * GRID_Z + z.long(), x.long(), y.long()] = values.float()


def make_group_tensor(data: dict[str, Any], key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    table_t = data["table"]["tensors"]
    compact = data["compact"]
    full_start, full_end = data["full_group_slices"][key]
    comp_start, comp_end = data["compact_group_slices"][key]
    full_idx = torch.arange(full_start, full_end)
    comp_idx = torch.arange(comp_start, comp_end)
    in_channels = len(TABLE_DENSE_CHANNELS) + len(COMPACT_CHANNELS)
    x = torch.zeros((in_channels * GRID_Z, GRID_X, GRID_Y), dtype=torch.float32)
    dense_ch = 0
    fx = table_t["voxel_x"][full_idx].long()
    fy = table_t["voxel_y"][full_idx].long()
    fz = table_t["voxel_z"][full_idx].long()
    for name in TABLE_DENSE_CHANNELS:
        assign_grid(x, dense_ch, fx, fy, fz, table_t[name][full_idx].float().nan_to_num(0.0))
        dense_ch += 1
    cx = table_t["voxel_x"][compact["rows"][comp_idx]].long()
    cy = table_t["voxel_y"][compact["rows"][comp_idx]].long()
    cz = table_t["voxel_z"][compact["rows"][comp_idx]].long()
    for name in COMPACT_CHANNELS:
        assign_grid(x, dense_ch, cx, cy, cz, compact["features"][name][comp_idx].float().nan_to_num(0.0))
        dense_ch += 1
    target = torch.zeros((GRID_Z, GRID_X, GRID_Y), dtype=torch.float32)
    mask = torch.zeros_like(target)
    target[cz, cx, cy] = compact["label"][comp_idx].float()
    mask[cz, cx, cy] = 1.0
    return x, target, mask, comp_idx


def batch_groups(data: dict[str, Any], keys: list[int], start: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    ms: list[torch.Tensor] = []
    for key in keys[start : start + batch_size]:
        x, y, m, _ = make_group_tensor(data, key)
        xs.append(x)
        ys.append(y)
        ms.append(m)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), torch.stack(ms, dim=0)


def masked_focal_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not bool(valid.any().item()):
        return logits.sum() * 0.0
    logits_v = logits[valid]
    target_v = target[valid]
    pos = target_v.sum()
    neg = len(target_v) - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 8.0)
    bce = F.binary_cross_entropy_with_logits(logits_v, target_v, pos_weight=pos_weight, reduction="none")
    prob = torch.sigmoid(logits_v)
    pt = prob * target_v + (1.0 - prob) * (1.0 - target_v)
    focal = bce * (1.0 - pt).pow(2.0)
    top_k = min(len(logits_v), max(2048, int(len(logits_v) * 0.15)))
    top_idx = torch.topk(logits_v, k=top_k).indices
    top_logits = logits_v[top_idx]
    top_target = target_v[top_idx]
    top_weight = 1.0 + 4.0 * (1.0 - top_target)
    top_bce = F.binary_cross_entropy_with_logits(top_logits, top_target, reduction="none")
    top_bce = (top_bce * top_weight).sum() / top_weight.sum().clamp_min(1.0)
    fp_prob = prob[target_v < 0.5]
    fp_penalty = fp_prob.square().mean() if len(fp_prob) else logits_v.sum() * 0.0
    return 0.35 * bce.mean() + 1.2 * focal.mean() + 2.0 * top_bce + 0.5 * fp_penalty


def train_model(train: dict[str, Any], cfg: H11Config, device: torch.device) -> tuple[BEVCompletionNet, list[dict[str, Any]]]:
    in_channels = (len(TABLE_DENSE_CHANNELS) + len(COMPACT_CHANNELS)) * GRID_Z
    model = BEVCompletionNet(in_channels, cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    keys = list(train["keys"])
    log: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        rng = random.Random(cfg.seed * 1000 + epoch)
        rng.shuffle(keys)
        losses: list[float] = []
        for start in range(0, len(keys), cfg.batch_groups):
            xb, yb, mb = batch_groups(train, keys, start, cfg.batch_groups)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            logits = model(xb)
            loss = masked_focal_bce(logits, yb, mb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "group_count": len(keys),
            "batch_groups": cfg.batch_groups,
            "device": str(device),
            "seconds": float(time.time() - t0),
        }
        log.append(row)
        print(f"[h11] epoch={epoch} loss={row['loss']:.5f} sec={row['seconds']:.1f}", flush=True)
    return model.eval(), log


@torch.inference_mode()
def score_split(model: BEVCompletionNet, data: dict[str, Any], cfg: H11Config, device: torch.device) -> torch.Tensor:
    out = torch.empty((len(data["compact"]["label"]),), dtype=torch.float32)
    keys = list(data["keys"])
    for start in range(0, len(keys), cfg.batch_groups):
        xs: list[torch.Tensor] = []
        idxs: list[torch.Tensor] = []
        for key in keys[start : start + cfg.batch_groups]:
            x, _target, _mask, comp_idx = make_group_tensor(data, key)
            xs.append(x)
            idxs.append(comp_idx)
        xb = torch.stack(xs, dim=0).to(device, non_blocking=True)
        prob = torch.sigmoid(model(xb)).detach().cpu()
        for b, key in enumerate(keys[start : start + cfg.batch_groups]):
            comp_idx = idxs[b]
            rows = data["compact"]["rows"][comp_idx]
            t = data["table"]["tensors"]
            cx = t["voxel_x"][rows].long()
            cy = t["voxel_y"][rows].long()
            cz = t["voxel_z"][rows].long()
            out[comp_idx] = prob[b, cz, cx, cy]
    return out


def norm(values: torch.Tensor) -> torch.Tensor:
    vals = values.float().nan_to_num(0.0)
    return ((vals - vals.min()) / (vals.max() - vals.min() + 1.0e-6)).clamp(0.0, 1.0) if vals.max() > vals.min() else torch.zeros_like(vals)


def topk_rows(data: dict[str, Any], scores: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    label = data["compact"]["label"].bool()
    rows = data["compact"]["rows"]
    t = data["table"]["tensors"]
    front = t["front_region"][rows].bool()
    future = t["future_h4h6"][rows].bool()
    out: list[dict[str, Any]] = []
    for name, score in scores.items():
        order = torch.argsort(score.float(), descending=True)
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000]:
            if len(order) < k:
                continue
            top = order[:k]
            out.append(
                {
                    "split": data["split"],
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


def full_score_for_selection(data: dict[str, Any], compact_score: torch.Tensor) -> torch.Tensor:
    full = torch.full((len(data["table"]["tensors"]["GT_occ"]),), -torch.inf, dtype=torch.float32)
    full[data["compact"]["rows"]] = compact_score.float()
    return full


def run_selection(val: dict[str, Any], add_score: torch.Tensor) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    suppress = val["compact"]["suppress_scores"].float()
    for sup in [0.02, 0.03, 0.04, 0.05]:
        for bal in [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
            for fixed in [0, 16, 32, 64, 96, 128, 192]:
                cfg = sw14e.Config(max_suppress_ratio=sup, add_suppress_balance=bal, add_fixed_budget=fixed)
                budget = {"name": f"h11_sup{sup}_bal{bal}_fixed{fixed}", "suppress_topk_ratio": 1.0, "add_topk_ratio": 1.0, "add_strength": "strong_medium"}
                _, summary = sw14e.evaluate_selection("val", val["table"], add_score, suppress, budget, cfg)
                row = {**summary, "max_suppress_ratio": sup, "add_suppress_balance": bal, "add_fixed_budget": fixed}
                rows.append(row)
                if summary["safety_pass_all"] and summary["recall_nonregression_pass"]:
                    if best is None or float(summary["mean_net_score"]) > float(best["mean_net_score"]):
                        best = row
    write_csv(REPORTS_DIR / "sw14h11_bev_completion_selection_sweep_val.csv", rows)
    return best or (max(rows, key=lambda r: float(r["mean_net_score"])) if rows else {})


def plot_precision(rows: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for name in ["h7b_lean", "h11_bev_completion", "h11_blend_h7b_10", "h11_blend_h7b_20", "h11_best_val"]:
        sub = sorted([r for r in rows if r["score_name"] == name], key=lambda r: int(r["topk"]))
        if sub:
            plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1.0, label="target 0.9")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title("SW14H11 BEV completion add precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h11_bev_completion_precision.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = H11Config(epochs=args.epochs, batch_groups=args.batch_groups, base_channels=args.base_channels, device=args.device)
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    train = build_split("train")
    write_json(
        REPORTS_DIR / "sw14h11_bev_completion_config.json",
        {
            "config": asdict(cfg),
            "table_dense_channels": TABLE_DENSE_CHANNELS,
            "compact_channels": COMPACT_CHANNELS,
            "train_stats": train["stats"],
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
        },
    )
    model, log = train_model(train, cfg, device)
    write_csv(REPORTS_DIR / "sw14h11_bev_completion_training_log.csv", log)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "resume_claim": False}, CHECKPOINT_DIR / "sw14h11_bev_completion_best.pth")
    del train
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    val = build_split("val")
    h11 = score_split(model, val, cfg, device)
    h7b = val["compact"]["features"]["h7b_lean"].float()
    h11n = norm(h11)
    score_map = {
        "h7b_lean": h7b,
        "h11_bev_completion": h11,
        "h11_blend_h7b_10": 0.10 * h11n + 0.90 * h7b,
        "h11_blend_h7b_20": 0.20 * h11n + 0.80 * h7b,
        "h11_blend_h7b_35": 0.35 * h11n + 0.65 * h7b,
        "h11_blend_h7b_50": 0.50 * h11n + 0.50 * h7b,
        "h11_blend_h7b_65": 0.65 * h11n + 0.35 * h7b,
        "h11_blend_h7b_80": 0.80 * h11n + 0.20 * h7b,
    }
    topk = topk_rows(val, score_map)
    best_val_100k = max((r for r in topk if int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    h7b_val_100k = next(r for r in topk if r["score_name"] == "h7b_lean" and int(r["topk"]) == 100000)
    score_map["h11_best_val"] = score_map[best_val_100k["score_name"]]
    plot_precision(topk + topk_rows(val, {"h11_best_val": score_map["h11_best_val"]}))
    write_csv(REPORTS_DIR / "sw14h11_bev_completion_topk_precision.csv", topk)
    torch.save({"val_sw14h11_bev_scores": h11, "val_row_index": val["compact"]["rows"]}, ARTIFACTS_DIR / "sw14h11_bev_completion_scores.pt")
    selection = run_selection(val, full_score_for_selection(val, score_map[best_val_100k["score_name"]]))
    gain = float(best_val_100k["precision"]) - float(h7b_val_100k["precision"])
    if float(best_val_100k["precision"]) >= cfg.target_precision_at_100k:
        decision = "SW14H11_1_TARGET_090_REACHED"
    elif gain >= 0.10:
        decision = "SW14H11_2_MATERIAL_BEV_COMPLETION_GAIN"
    elif gain > 0.005:
        decision = "SW14H11_3_SMALL_BEV_COMPLETION_GAIN"
    else:
        decision = "SW14H11_4_BEV_COMPLETION_NOT_ENOUGH"
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
        "recommended_next_action": "If H11 remains far below 0.9@100K, the deployable no-GT add route needs deeper decoder-query tensors or a different candidate-generation mechanism.",
    }
    write_json(REPORTS_DIR / "sw14h11_bev_completion_final_decision.json", final)
    print(f"[h11] decision={decision} best_val_100k={best_val_100k}", flush=True)


if __name__ == "__main__":
    main()
