from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

SW7_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
SW7_FIGURES = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
SW7_VIDEOS = PROJECT_ROOT / "videos/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
SW7_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
SW6_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw71_reliability_cleanup"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw71_reliability_cleanup"
VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_sw71_reliability_cleanup"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw71_reliability_cleanup"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw2 = load_module(
    "sw71_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
renderer = load_module(
    "sw71_renderer",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization/semantic_occupancy_renderer.py",
)


CORE_PERTS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
SECTOR_NAMES = ["front", "front_left", "front_right", "rear"]
CLASS_GROUP_NAMES = ["small_object", "new_visible", "static", "dynamic"]
TOPKS = [0.01, 0.05, 0.10]
EMPTY_IDX = sw2.EMPTY_IDX


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, FIGURES_DIR, VIDEOS_DIR, TESTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def build_sector_masks() -> dict[str, torch.Tensor]:
    x = torch.arange(sw2.GRID_SIZE[0], dtype=torch.float32) * sw2.VOXEL_SIZE[0] + sw2.PC_RANGE[0] + sw2.VOXEL_SIZE[0] / 2
    y = torch.arange(sw2.GRID_SIZE[1], dtype=torch.float32) * sw2.VOXEL_SIZE[1] + sw2.PC_RANGE[1] + sw2.VOXEL_SIZE[1] / 2
    z = torch.arange(sw2.GRID_SIZE[2], dtype=torch.float32) * sw2.VOXEL_SIZE[2] + sw2.PC_RANGE[2] + sw2.VOXEL_SIZE[2] / 2
    xx = x[:, None, None].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    yy = y[None, :, None].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    zz = z[None, None, :].expand(sw2.GRID_SIZE[0], sw2.GRID_SIZE[1], sw2.GRID_SIZE[2])
    _ = zz
    return {
        "front": (xx > 0) & (yy.abs() <= 10.0),
        "front_left": (xx > 0) & (yy > 10.0),
        "front_right": (xx > 0) & (yy < -10.0),
        "rear": xx < 0,
    }


def load_pred_occ(perturbation_id: str, sample_index: int) -> torch.Tensor:
    obj = torch.load(
        SW6_ARTIFACTS / f"occ_artifacts/{perturbation_id}/sample_{sample_index:04d}_occ_temporal.pt",
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(obj, dict):
        if "pred_temporal" in obj:
            return obj["pred_temporal"].long()
        if "semantic_occ" in obj:
            return obj["semantic_occ"].long()
        for v in obj.values():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                return v.long()
        raise KeyError(f"unable to locate occupancy tensor in {perturbation_id} sample {sample_index}")
    return obj.long()


def load_gt_occ(sample_index: int) -> torch.Tensor:
    return torch.load(
        PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/standard_occ/sample_{sample_index:04d}_standard_gt_occ_temporal.pt",
        map_location="cpu",
        weights_only=False,
    ).long()


def load_risk_npz(perturbation_id: str, sample_index: int, horizon: int) -> dict[str, Any]:
    path = SW7_ARTIFACTS / f"reliability_maps/{perturbation_id}/sample_{sample_index:04d}/h{horizon}_reliability.npz"
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def class_group_mask(name: str, pred: torch.Tensor, gt: torch.Tensor, gt0: torch.Tensor) -> torch.Tensor:
    if name == "small_object":
        ids = set(sw2.CLASS_GROUPS["small_object"])
        return torch.isin(gt, torch.tensor(sorted(ids), dtype=gt.dtype))
    if name == "dynamic":
        ids = set(sw2.CLASS_GROUPS["all_dynamic"])
        return torch.isin(gt, torch.tensor(sorted(ids), dtype=gt.dtype))
    if name == "static":
        ids = set(sw2.CLASS_GROUPS["all_static"])
        return torch.isin(gt, torch.tensor(sorted(ids), dtype=gt.dtype))
    if name == "new_visible":
        return (gt0 == EMPTY_IDX) & (gt != EMPTY_IDX)
    raise KeyError(name)


def topk_mask_from_scores(scores: torch.Tensor, scope_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    valid = scope_mask.bool()
    count = int(valid.sum().item())
    out = torch.zeros_like(valid)
    if count <= 0:
        return out
    k = max(1, int(math.ceil(count * ratio)))
    flat_scores = scores[valid].flatten()
    if k >= flat_scores.numel():
        out[valid] = True
        return out
    thresh = torch.topk(flat_scores, k).values[-1]
    chosen = valid & (scores >= thresh)
    if int(chosen.sum().item()) > k:
        idx = torch.nonzero(chosen, as_tuple=False)
        vals = scores[chosen]
        order = torch.argsort(vals, descending=True)[:k]
        out[idx[order, 0], idx[order, 1], idx[order, 2]] = True
        return out
    return chosen


def build_bev_risk_map(semantic_occ: np.ndarray, risk_map: np.ndarray) -> Image.Image:
    occ = np.asarray(semantic_occ)
    risk = np.asarray(risk_map, dtype=np.float32)
    active = occ != EMPTY_IDX
    canvas = np.ones((occ.shape[1], occ.shape[0], 3), dtype=np.float32)
    if active.any():
        has_any = active.any(axis=2)
        top_idx = np.where(has_any, active.shape[2] - 1 - np.argmax(active[:, :, ::-1], axis=2), -1)
        xs, ys = np.where(has_any)
        zs = top_idx[xs, ys]
        labels = occ[xs, ys, zs]
        rr = risk[xs, ys, zs]
        for x, y, cls_id, rv in zip(xs.tolist(), ys.tolist(), labels.tolist(), rr.tolist()):
            cls_name = renderer.OCC_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(renderer.OCC_NAMES) else "others"
            base = hex_to_rgb01(renderer.PALETTE.get(cls_name, "#808080"))
            heat = np.array([1.0, 0.25, 0.05], dtype=np.float32)
            mix = min(max(float(rv), 0.0), 1.0) * 0.75
            color = np.clip((1.0 - mix) * base + mix * heat, 0.0, 1.0)
            canvas[occ.shape[1] - 1 - y, x] = color
    return Image.fromarray((canvas * 255).astype(np.uint8), mode="RGB").resize((960, 540), Image.Resampling.NEAREST)


def build_component_breakdown() -> tuple[list[dict[str, Any]], Path, Path]:
    df = read_csv_df(SW7_REPORTS / "sector_reliability_summary.csv")
    df = df[(df["horizon_s"] == 6) & (df["sector_name"] == "front") & (df["perturbation_id"].isin(CORE_PERTS))].copy()
    mapping = {
        "semantic": "mean_semantic_component",
        "support": "mean_support_component",
        "contributor": "mean_contributor_component",
        "sensor_prior": "mean_sensor_component",
        "temporal": "mean_temporal_component",
    }
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for name, col in mapping.items():
            reliability = float(row[col])
            rows.append(
                {
                    "perturbation_id": row["perturbation_id"],
                    "component": name,
                    "component_reliability": reliability,
                    "component_risk": 1.0 - reliability,
                }
            )
    csv_path = REPORTS_DIR / "component_breakdown_values.csv"
    write_csv(csv_path, rows)

    plot_df = pd.DataFrame(rows)
    order = CORE_PERTS
    comps = list(mapping.keys())
    x = np.arange(len(order))
    w = 0.15

    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    for i, comp in enumerate(comps):
        vals = [float(plot_df[(plot_df["perturbation_id"] == p) & (plot_df["component"] == comp)]["component_reliability"].iloc[0]) for p in order]
        ax.bar(x + (i - 2) * w, vals, width=w, label=comp.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("component reliability")
    ax.set_title("Reliability component breakdown")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    rel_png = FIGURES_DIR / "reliability_component_breakdown.png"
    fig.tight_layout()
    fig.savefig(rel_png, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    for i, comp in enumerate(comps):
        vals = [float(plot_df[(plot_df["perturbation_id"] == p) & (plot_df["component"] == comp)]["component_risk"].iloc[0]) for p in order]
        ax.bar(x + (i - 2) * w, vals, width=w, label=comp.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("component risk = 1 - component reliability")
    ax.set_title("Risk component breakdown")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    risk_png = FIGURES_DIR / "risk_component_breakdown.png"
    fig.tight_layout()
    fig.savefig(risk_png, bbox_inches="tight")
    plt.close(fig)
    return rows, rel_png, risk_png


def compute_topk_validation() -> tuple[list[dict[str, Any]], str]:
    sector_masks = build_sector_masks()
    rows: list[dict[str, Any]] = []
    sample_indices = list(range(10))
    for perturbation_id in CORE_PERTS:
        for sample_index in sample_indices:
            pred_t = load_pred_occ(perturbation_id, sample_index)
            gt_t = load_gt_occ(sample_index)
            gt0 = gt_t[0]
            for horizon in range(pred_t.shape[0]):
                pred = pred_t[horizon]
                gt = gt_t[horizon]
                ff = (pred == EMPTY_IDX) & (gt != EMPTY_IDX)
                fp = (pred != EMPTY_IDX) & (gt == EMPTY_IDX)
                union_occ = (pred != EMPTY_IDX) | (gt != EMPTY_IDX)
                payload = load_risk_npz(perturbation_id, sample_index, horizon)
                risk = torch.from_numpy(payload["risk_map"].astype(np.float32))

                scope_defs: list[tuple[str, str, torch.Tensor]] = [("global", "global", union_occ)]
                for sector_name, sector_mask in sector_masks.items():
                    scope_defs.append(("sector", sector_name, union_occ & sector_mask))
                for group_name in CLASS_GROUP_NAMES:
                    scope_defs.append(("class_group", group_name, union_occ & class_group_mask(group_name, pred, gt, gt0)))

                for scope_type, scope_name, scope_mask in scope_defs:
                    if int(scope_mask.sum().item()) == 0:
                        continue
                    for topk in TOPKS:
                        selected = topk_mask_from_scores(risk, scope_mask, topk)
                        selected_count = int(selected.sum().item())
                        for error_name, error_mask in [("false_free", ff), ("false_positive", fp)]:
                            hits = int((selected & error_mask).sum().item())
                            total_error = int((scope_mask & error_mask).sum().item())
                            rows.append(
                                {
                                    "perturbation_id": perturbation_id,
                                    "sample_index": sample_index,
                                    "horizon_s": horizon,
                                    "scope_type": scope_type,
                                    "scope_name": scope_name,
                                    "topk_ratio": topk,
                                    "error_type": error_name,
                                    "selected_voxels": selected_count,
                                    "error_voxels_in_scope": total_error,
                                    "hit_voxels": hits,
                                    "precision": safe_div(hits, selected_count),
                                    "recall": safe_div(hits, total_error),
                                }
                            )

    summary_df = pd.DataFrame(rows)
    agg = (
        summary_df.groupby(["perturbation_id", "scope_type", "scope_name", "topk_ratio", "error_type"], dropna=False)[["precision", "recall", "selected_voxels", "error_voxels_in_scope", "hit_voxels"]]
        .mean()
        .reset_index()
    )
    agg_rows = agg.to_dict(orient="records")
    write_csv(REPORTS_DIR / "high_risk_topk_validation.csv", agg_rows)

    plot_df = agg[(agg["scope_type"] == "global")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220, sharex=True)
    for error_name, ax in [("false_free", axes[0]), ("false_positive", axes[1])]:
        sub = plot_df[plot_df["error_type"] == error_name]
        for pert in CORE_PERTS:
            ss = sub[sub["perturbation_id"] == pert].sort_values("topk_ratio")
            ax.plot(ss["topk_ratio"] * 100.0, ss["recall"], marker="o", label=f"{pert} recall")
            ax.plot(ss["topk_ratio"] * 100.0, ss["precision"], marker="x", linestyle="--", label=f"{pert} precision")
        ax.set_title(error_name.replace("_", " "))
        ax.set_xlabel("top-K high-risk voxels (%)")
        ax.set_ylabel("precision / recall")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "high_risk_topk_precision_recall.png", bbox_inches="tight")
    plt.close(fig)

    a10_ff = agg[(agg["perturbation_id"] == "A10_drop_front_triplet") & (agg["scope_name"] == "front") & (agg["topk_ratio"] == 0.05) & (agg["error_type"] == "false_free")]
    c4_fp = agg[(agg["perturbation_id"] == "C4_motion_blur_9") & (agg["scope_name"] == "front") & (agg["topk_ratio"] == 0.05) & (agg["error_type"] == "false_positive")]
    headline = (
        f"A10 front top-5% high-risk voxels capture false-free with recall={float(a10_ff['recall'].mean()):.3f}, precision={float(a10_ff['precision'].mean()):.3f}; "
        f"C4 front top-5% high-risk voxels capture false-positive with recall={float(c4_fp['recall'].mean()):.3f}, precision={float(c4_fp['precision'].mean()):.3f}."
    )
    return agg_rows, headline


def build_ppt_ready_summary() -> tuple[Path, Path]:
    report = read_json(SW7_REPORTS / "stage_sw7_sensor_conditioned_reliability_map_report.json")
    sample_index = 0
    h = 6
    pert_labels = {
        "A0_clean": "A0 clean",
        "A1_drop_cam_front": "A1 drop front",
        "A10_drop_front_triplet": "A10 drop front triplet",
        "C4_motion_blur_9": "C4 motion blur",
        "A7_drop_all_rear": "A7 rear dropout control",
    }
    panels: list[Image.Image] = []
    for pert in CORE_PERTS:
        payload = load_risk_npz(pert, sample_index, h)
        img = build_bev_risk_map(payload["semantic_occ"], payload["risk_map"])
        canvas = Image.new("RGB", (980, 620), "white")
        canvas.paste(img, (10, 60))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("times.ttf", 34)
        except Exception:
            font = ImageFont.load_default()
        draw.text((18, 14), pert_labels[pert], fill="black", font=font)
        panels.append(canvas)

    sector_df = read_csv_df(SW7_REPORTS / "sector_reliability_summary.csv")
    class_df = read_csv_df(SW7_REPORTS / "class_group_reliability_summary.csv")
    metric_rows = []
    for pert in CORE_PERTS:
        front = sector_df[(sector_df["perturbation_id"] == pert) & (sector_df["horizon_s"] == 6) & (sector_df["sector_name"] == "front")].iloc[0]
        small = class_df[(class_df["perturbation_id"] == pert) & (class_df["horizon_s"] == 6) & (class_df["class_group"] == "small_object")].iloc[0]
        newv = class_df[(class_df["perturbation_id"] == pert) & (class_df["horizon_s"] == 6) & (class_df["class_group"] == "new_visible")].iloc[0]
        metric_rows.append(
            {
                "perturbation_id": pert_labels[pert],
                "front_reliability": float(front["mean_mean_reliability"]),
                "small_object_reliability": float(small["mean_mean_reliability"]),
                "new_visible_reliability": float(newv["mean_mean_reliability"]),
            }
        )
    metrics_df = pd.DataFrame(metric_rows)

    fig = plt.figure(figsize=(19, 8), dpi=220)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 2.8, 1.15], height_ratios=[0.22, 0.78])
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.55, "Sensor-Conditioned Occupancy Reliability Map", ha="center", va="center", fontsize=24, fontweight="bold")
    title_ax.text(0.5, 0.1, "A10 front-sector risk highest · small-object / new-visible risk high · full reliability > single components · A7 is weaker control", ha="center", va="center", fontsize=12)

    left_ax = fig.add_subplot(gs[1, 0])
    left_ax.axis("off")
    left_text = [
        "Degradation inputs",
        "",
        "A0 clean",
        "A1 drop front camera",
        "A10 drop front triplet",
        "C4 motion blur",
        "A7 rear dropout control",
        "",
        "This stage outputs an internal diagnostic",
        "reliability indicator and risk map, not",
        "calibrated uncertainty or an official benchmark.",
    ]
    left_ax.text(0.02, 0.98, "\n".join(left_text), va="top", fontsize=13)

    mid_canvas = Image.new("RGB", (2940, 1240), "white")
    x = 0
    for p in panels:
        mid_canvas.paste(p, (x, 0))
        x += p.width
    mid_ax = fig.add_subplot(gs[1, 1])
    mid_ax.imshow(mid_canvas)
    mid_ax.axis("off")
    mid_ax.set_title("Reliability / risk map results (sample0, h6≈+3s)", fontsize=13)

    right_ax = fig.add_subplot(gs[1, 2])
    y = np.arange(len(metrics_df))
    right_ax.barh(y - 0.22, metrics_df["front_reliability"], height=0.22, label="front")
    right_ax.barh(y, metrics_df["small_object_reliability"], height=0.22, label="small object")
    right_ax.barh(y + 0.22, metrics_df["new_visible_reliability"], height=0.22, label="new visible")
    right_ax.set_yticks(y)
    right_ax.set_yticklabels(metrics_df["perturbation_id"])
    right_ax.set_xlim(0, 1.0)
    right_ax.set_xlabel("mean reliability")
    right_ax.set_title("Key indicators @ h6")
    right_ax.grid(axis="x", alpha=0.2)
    right_ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    png_path = FIGURES_DIR / "sw7_reliability_summary_ppt_ready.png"
    pdf_path = FIGURES_DIR / "sw7_reliability_summary_ppt_ready.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    _ = report
    return png_path, pdf_path


def build_one_page_project_card(topk_headline: str) -> Path:
    report = read_json(SW7_REPORTS / "stage_sw7_sensor_conditioned_reliability_map_report.json")
    ablation_df = read_csv_df(SW7_REPORTS / "reliability_component_ablation.csv")
    corr_map = {
        str(r["component_name"]): float(r["mean_risk_error_correlation"])
        for _, r in ablation_df.iterrows()
        if pd.notna(r["mean_risk_error_correlation"])
    }
    text = "\n".join(
        [
            "# SW-7 one-page project card",
            "",
            "## What we built",
            "",
            "A sensor-conditioned occupancy reliability / risk map on top of SparseWorld future semantic occupancy, combining semantic confidence, support density, contributor strength, perturbation prior, and horizon-aware decay into a voxel-level internal diagnostic reliability indicator.",
            "",
            "## Why it matters",
            "",
            "SparseWorld under front-view loss and motion blur does not fail uniformly. The reliability map localizes where future occupancy is less trustworthy, which is more useful for analysis, presentation, and future model-side intervention planning than a single global metric.",
            "",
            "## Main finding",
            "",
            f"A10 drop-front-triplet is the lowest-reliability degradation overall, with front-sector, small-object, and new-visible regions degrading the most. Full reliability explains error better than any single component. {topk_headline}",
            "",
            "## Evidence",
            "",
            f"- Lowest reliability degradation: `{report['lowest_reliability_degradation']}`",
            f"- Lowest reliability sector: `{report['lowest_reliability_sector']}`",
            f"- Lowest reliability class group: `{report['lowest_reliability_class_group']}`",
            f"- Full reliability correlation vs actual error: `{corr_map['full_reliability']:.4f}`",
            f"- Semantic-only correlation: `{corr_map['semantic']:.4f}`",
            f"- Support-only correlation: `{corr_map['support']:.4f}`",
            f"- Contributor-only correlation: `{corr_map['contributor']:.4f}`",
            "",
            "## Safe claim",
            "",
            "This is an internal diagnostic reliability indicator / risk map built on a controlled synthetic-proxy perturbation subset. It is not calibrated uncertainty, not an official benchmark, not a production planning policy, and does not claim model performance improvement.",
            "",
            "## Next step SW-8",
            "",
            f"Prioritize `{report['sw8_recommendation']}` as the smallest training-facing follow-up, then re-check front-sector small-object and new-visible reliability under A1/A10.",
        ]
    )
    path = REPORTS_DIR / "sw7_one_page_project_card.md"
    path.write_text(text, encoding="utf-8")
    return path


def build_selected_assets() -> tuple[Path, Path]:
    resume_shot_src = SW7_FIGURES / "dashboard_frames/A10_drop_front_triplet/frame_000.png"
    resume_shot_dst = FIGURES_DIR / "selected_resume_dashboard_a10.png"
    shutil.copy2(resume_shot_src, resume_shot_dst)
    payload = {
        "ppt_figures": [
            {
                "path": str(FIGURES_DIR / "sw7_reliability_summary_ppt_ready.png"),
                "shows": "single-slide summary of degradation, risk map result, and key reliability indicators",
                "conclusion": "A10 front-sector risk is highest and full reliability outperforms single components",
                "how_to_tell": "Use this as the entry slide for the whole reliability-map stage",
            },
            {
                "path": str(SW7_FIGURES / "risk_map_clean_vs_a1_a10_c4_h6.png"),
                "shows": "clean vs degraded risk overlays at h6≈+3s",
                "conclusion": "front dropout and motion blur move the risk map into different spatial / semantic regimes",
                "how_to_tell": "Contrast A10 front collapse against C4 semantic confusion / false-positive risk",
            },
            {
                "path": str(FIGURES_DIR / "high_risk_topk_precision_recall.png"),
                "shows": "top-K high-risk precision / recall against false-free and false-positive",
                "conclusion": "the risk map is not decorative; it correlates with actual error",
                "how_to_tell": "Explain that top-risk voxels recover more front false-free under A10 and more false-positive under C4",
            },
        ],
        "interview_videos": [
            {
                "path": str(SW7_VIDEOS / "a10_front_triplet_reliability_dashboard.mp4"),
                "shows": "front-triplet dropout dashboard with BEV occupancy, reliability, and risk",
                "conclusion": "support/contributor collapse propagates to front future occupancy failure",
                "how_to_tell": "Narrate the sensor degradation -> support/contributor -> risk chain",
            },
            {
                "path": str(SW7_VIDEOS / "c4_motion_blur_reliability_dashboard.mp4"),
                "shows": "motion blur dashboard with risk and confidence fading",
                "conclusion": "C4 is more semantic-confusion / false-positive heavy than A1/A10",
                "how_to_tell": "Use this to contrast geometry-support collapse vs semantic confusion modes",
            },
        ],
        "resume_dashboard": {
            "path": str(resume_shot_dst),
            "shows": "single dashboard screenshot for a resume / project card thumbnail",
            "conclusion": "A10 front dropout is the clearest compact visual story",
            "how_to_tell": "One screenshot that already exposes input degradation, occupancy change, and risk output together",
        },
    }
    manifest = REPORTS_DIR / "selected_assets_manifest.json"
    write_json(manifest, payload)
    readme = REPORTS_DIR / "selected_assets_readme.md"
    lines = ["# Selected SW-7 assets", ""]
    for section in ["ppt_figures", "interview_videos"]:
        lines.append(f"## {section}")
        lines.append("")
        for item in payload[section]:
            lines.append(f"- `{item['path']}`")
            lines.append(f"  - shows: {item['shows']}")
            lines.append(f"  - conclusion: {item['conclusion']}")
            lines.append(f"  - how to tell it: {item['how_to_tell']}")
        lines.append("")
    lines.append("## resume_dashboard")
    lines.append("")
    lines.append(f"- `{payload['resume_dashboard']['path']}`")
    lines.append(f"  - shows: {payload['resume_dashboard']['shows']}")
    lines.append(f"  - conclusion: {payload['resume_dashboard']['conclusion']}")
    lines.append(f"  - how to tell it: {payload['resume_dashboard']['how_to_tell']}")
    readme.write_text("\n".join(lines), encoding="utf-8")
    return manifest, readme


def build_clean_report(topk_headline: str) -> tuple[Path, Path]:
    sw7_report = read_json(SW7_REPORTS / "stage_sw7_sensor_conditioned_reliability_map_report.json")
    payload = {
        "stage": "SW-7.1",
        "purpose": "Reliability map figure / report cleanup on top of existing SW-7 outputs without rerunning large forward passes.",
        "reliability_definition": "internal diagnostic reliability indicator",
        "risk_definition": "risk = 1 - reliability",
        "not_claims": [
            "not calibrated uncertainty",
            "not official benchmark",
            "not production planning policy",
            "no training",
            "no model improvement claim",
        ],
        "lowest_reliability_degradation": sw7_report["lowest_reliability_degradation"],
        "lowest_reliability_sector": sw7_report["lowest_reliability_sector"],
        "lowest_reliability_class_group": sw7_report["lowest_reliability_class_group"],
        "topk_validation_headline": topk_headline,
        "key_outputs": {
            "reliability_component_breakdown": str(FIGURES_DIR / "reliability_component_breakdown.png"),
            "risk_component_breakdown": str(FIGURES_DIR / "risk_component_breakdown.png"),
            "topk_validation_csv": str(REPORTS_DIR / "high_risk_topk_validation.csv"),
            "ppt_ready_figure": str(FIGURES_DIR / "sw7_reliability_summary_ppt_ready.png"),
            "one_page_card": str(REPORTS_DIR / "sw7_one_page_project_card.md"),
            "selected_assets_manifest": str(REPORTS_DIR / "selected_assets_manifest.json"),
        },
        "next_unique_action": "Stage SW-8 small-object activation / front-sector robustness fine-tune feasibility prototype guided by A10 and C4 risk concentration.",
    }
    json_path = REPORTS_DIR / "stage_sw71_reliability_cleanup_report.json"
    md_path = REPORTS_DIR / "stage_sw71_reliability_cleanup_report.md"
    write_json(json_path, payload)
    md_lines = [
        "# Stage SW-7.1 reliability cleanup",
        "",
        "- reliability = internal diagnostic reliability indicator",
        "- risk = 1 - reliability",
        "- not calibrated uncertainty",
        "- not official benchmark",
        "- not production planning policy",
        "- no training",
        "- no model improvement claim",
        "",
        f"- lowest reliability degradation: `{payload['lowest_reliability_degradation']}`",
        f"- lowest reliability sector: `{payload['lowest_reliability_sector']}`",
        f"- lowest reliability class group: `{payload['lowest_reliability_class_group']}`",
        f"- top-K validation headline: {topk_headline}",
        "",
        "This cleanup stage fixes naming ambiguity between reliability and risk component breakdown figures, adds top-K high-risk validation against actual false-free / false-positive voxels, and packages the stage into PPT / resume / interview-ready assets.",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    ensure_dirs()
    _, _, _ = build_component_breakdown()
    _, topk_headline = compute_topk_validation()
    build_ppt_ready_summary()
    build_one_page_project_card(topk_headline)
    build_selected_assets()
    build_clean_report(topk_headline)
    summary_md = REPORTS_DIR / "high_risk_topk_validation_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# High-risk top-K validation",
                "",
                topk_headline,
                "",
                "Selection is computed inside the occupied union scope `(pred occupied) ∪ (gt occupied)` for each perturbation / sector / class-group slice, so the validation measures whether the risk map concentrates onto occupied failure regions rather than trivial empty space.",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
