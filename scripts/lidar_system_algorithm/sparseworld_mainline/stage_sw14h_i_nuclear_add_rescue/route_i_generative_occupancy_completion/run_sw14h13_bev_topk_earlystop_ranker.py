from __future__ import annotations

import argparse
import csv
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
H11_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_i_generative_occupancy_completion/run_sw14h11_bev_completion_ranker.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h11 = load_module("sw14h11_for_h13", H11_SCRIPT)


@dataclass(frozen=True)
class H13Config:
    seed: int = 613
    epochs: int = 10
    batch_groups: int = 4
    base_channels: int = 96
    lr: float = 7.0e-4
    weight_decay: float = 1.0e-4
    target_precision_at_100k: float = 0.90
    device: str = "cuda"


def normalize(obj: Any) -> Any:
    return h11.normalize(obj)


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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([normalize(row) for row in rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14H13 BEV topK early-stop add ranker")
    parser.add_argument("--epochs", type=int, default=H13Config.epochs)
    parser.add_argument("--batch-groups", type=int, default=H13Config.batch_groups)
    parser.add_argument("--base-channels", type=int, default=H13Config.base_channels)
    parser.add_argument("--device", default=H13Config.device)
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


def topk_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, epoch_frac: float) -> torch.Tensor:
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
    base = 0.45 * bce.mean() + 1.55 * focal.mean()

    k = min(len(logits_v), max(4096, int(len(logits_v) * 0.10)))
    top = torch.topk(logits_v, k=k).indices
    top_target = target_v[top]
    top_bce = F.binary_cross_entropy_with_logits(logits_v[top], top_target, reduction="none")
    top_weight = 1.0 + 7.0 * (1.0 - top_target)
    top_term = (top_bce * top_weight).sum() / top_weight.sum().clamp_min(1.0)

    pos_logits = logits_v[target_v > 0.5]
    neg_top_logits = logits_v[top][top_target < 0.5]
    if len(pos_logits) and len(neg_top_logits):
        pair_k = min(len(pos_logits), len(neg_top_logits), 8192)
        pos_sel = pos_logits[torch.randint(0, len(pos_logits), (pair_k,), device=logits.device)]
        neg_sel = neg_top_logits[torch.randint(0, len(neg_top_logits), (pair_k,), device=logits.device)]
        rank_term = F.softplus(0.75 - pos_sel + neg_sel).mean()
    else:
        rank_term = logits_v.sum() * 0.0

    fp_prob = prob[target_v < 0.5]
    fp_term = fp_prob.square().mean() if len(fp_prob) else logits_v.sum() * 0.0
    hard_weight = min(1.0, max(0.0, epoch_frac))
    return base + hard_weight * (1.5 * top_term + 1.2 * rank_term + 0.35 * fp_term)


def precision_at_100k(label: torch.Tensor, score: torch.Tensor) -> dict[str, Any]:
    top = torch.topk(score.float(), k=min(100000, len(score))).indices
    return {
        "topk": int(len(top)),
        "precision": float(label[top].float().mean().item()) if len(top) else 0.0,
        "positive_count": int(label[top].sum().item()) if len(top) else 0,
    }


@torch.inference_mode()
def evaluate_val(model: nn.Module, val: dict[str, Any], cfg: H13Config, device: torch.device, epoch: int) -> tuple[dict[str, Any], torch.Tensor]:
    h13 = h11.score_split(model, val, cfg, device)
    h13n = h11.norm(h13)
    h7b = val["compact"]["features"]["h7b_lean"].float()
    label = val["compact"]["label"].bool()
    best: dict[str, Any] | None = None
    best_score = h13
    rows: list[dict[str, Any]] = []
    for weight in [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0]:
        score = weight * h13n + (1.0 - weight) * h7b
        row = precision_at_100k(label, score)
        row.update({"epoch": epoch, "score_name": f"h13_blend_h7b_{weight:.2f}", "h13_weight": weight})
        rows.append(row)
        if best is None or float(row["precision"]) > float(best["precision"]):
            best = row
            best_score = score
    assert best is not None
    best["epoch_rows"] = rows
    return best, best_score


def main() -> None:
    args = parse_args()
    cfg = H13Config(epochs=args.epochs, batch_groups=args.batch_groups, base_channels=args.base_channels, device=args.device)
    seed_everything(cfg.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    train = h11.build_split("train")
    val = h11.build_split("val")
    model = h11.BEVCompletionNet((len(h11.TABLE_DENSE_CHANNELS) + len(h11.COMPACT_CHANNELS)) * h11.GRID_Z, cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    keys = list(train["keys"])
    training_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score: torch.Tensor | None = None

    for epoch in range(cfg.epochs):
        t0 = time.time()
        rng = random.Random(cfg.seed * 1000 + epoch)
        rng.shuffle(keys)
        losses: list[float] = []
        model.train()
        for start in range(0, len(keys), cfg.batch_groups):
            xb, yb, mb = h11.batch_groups(train, keys, start, cfg.batch_groups)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            logits = model(xb)
            loss = topk_loss(logits, yb, mb, epoch / max(1, cfg.epochs - 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        epoch_best, epoch_score = evaluate_val(model, val, cfg, device, epoch)
        for row in epoch_best.pop("epoch_rows"):
            val_rows.append(row)
        train_row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "val_best_precision_at_100k": float(epoch_best["precision"]),
            "val_best_positive_count_at_100k": int(epoch_best["positive_count"]),
            "val_best_score_name": epoch_best["score_name"],
            "seconds": float(time.time() - t0),
            "device": str(device),
        }
        training_rows.append(train_row)
        print(
            f"[h13] epoch={epoch} loss={train_row['loss']:.5f} "
            f"val_p100k={train_row['val_best_precision_at_100k']:.5f} "
            f"score={train_row['val_best_score_name']} sec={train_row['seconds']:.1f}",
            flush=True,
        )
        if best is None or float(epoch_best["precision"]) > float(best["precision"]):
            best = dict(epoch_best)
            best_score = epoch_score.detach().cpu()
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "resume_claim": False, "best": best}, CHECKPOINT_DIR / "sw14h13_bev_topk_earlystop_best.pth")

    assert best is not None and best_score is not None
    write_csv(REPORTS_DIR / "sw14h13_bev_topk_earlystop_training_log.csv", training_rows)
    write_csv(REPORTS_DIR / "sw14h13_bev_topk_earlystop_val_precision_by_epoch.csv", val_rows)
    torch.save({"val_sw14h13_best_scores": best_score, "val_row_index": val["compact"]["rows"]}, ARTIFACTS_DIR / "sw14h13_bev_topk_earlystop_scores.pt")
    selection = h11.run_selection(val, h11.full_score_for_selection(val, best_score))
    decision = "SW14H13_1_TARGET_090_REACHED" if float(best["precision"]) >= cfg.target_precision_at_100k else "SW14H13_2_TOPK_EARLYSTOP_BELOW_TARGET"
    write_json(
        REPORTS_DIR / "sw14h13_bev_topk_earlystop_final_decision.json",
        {
            "decision": decision,
            "target_precision_at_100k": cfg.target_precision_at_100k,
            "target_reached": bool(float(best["precision"]) >= cfg.target_precision_at_100k),
            "best_val_top100k": best,
            "selection": selection,
            "config": asdict(cfg),
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
            "whether_resume_claim_allowed": False,
            "sw13_remains_main_result": True,
        },
    )
    print(f"[h13] decision={decision} best={best}", flush=True)


if __name__ == "__main__":
    main()
