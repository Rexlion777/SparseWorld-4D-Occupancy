#!/usr/bin/env python3
"""Generate the public SparseWorld supported-result gallery.

Only frozen SW13A/R8 evidence and the already published headline values are
used.  Unsupported learned-repair experiments are intentionally excluded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence"
RAW = EVIDENCE / "raw"
OUT = ROOT / "assets" / "portfolio"

INK = "#152238"
MUTED = "#66758B"
GRID = "#DCE4EE"
BLUE = "#2563EB"
BLUE_LIGHT = "#DBEAFE"
TEAL = "#0F766E"
TEAL_LIGHT = "#CCFBF1"
GOLD = "#D97706"
GOLD_LIGHT = "#FEF3C7"
PINK = "#BE185D"
PURPLE = "#7C3AED"
ORANGE = "#EA580C"
GREY = "#94A3B8"

R0 = "R0_degraded_native"
R8 = "R8_camera_group_repair_front_triplet"
A10 = "A10_drop_front_triplet"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#C8D2E0",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean(ax: plt.Axes, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.955, title, color=INK, fontsize=22, fontweight="bold", va="top")
    fig.text(0.055, 0.915, subtitle, color=MUTED, fontsize=11, va="top")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    headline = pd.read_csv(EVIDENCE / "headline_supported_results.csv")
    aggregate = pd.read_csv(RAW / "sw13a_feature_memory_aggregate_metrics.csv")
    rows = pd.read_csv(RAW / "sw13a_feature_memory_replay_metrics.csv")
    tradeoff = pd.read_csv(RAW / "sw13a_recovery_density_tradeoff.csv")
    return headline, aggregate, rows, tradeoff


def a10_pair(aggregate: pd.DataFrame) -> pd.DataFrame:
    x = aggregate[(aggregate.perturbation_id == A10) & aggregate.variant_label.isin([R0, R8])].copy()
    return x.sort_values(["horizon_s", "variant_label"])


def a10_wide(aggregate: pd.DataFrame, metric: str) -> pd.DataFrame:
    return a10_pair(aggregate).pivot(index="horizon_s", columns="variant_label", values=metric).sort_index()


def paired_rows(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    x = rows[(rows.perturbation_id == A10) & rows.variant_label.isin([R0, R8])]
    wide = x.pivot_table(index=["sample_index", "horizon_s"], columns="variant_label", values=metric, aggfunc="first").dropna()
    return wide.reset_index()


def grouped_metric_chart(aggregate: pd.DataFrame, metric: str, title: str, subtitle: str, ylabel: str, filename: str, scale: float = 1.0) -> None:
    wide = a10_wide(aggregate, metric) * scale
    x = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, title, subtitle)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.80)
    width = 0.34
    b0 = ax.bar(x - width / 2, wide[R0], width, color=GREY, label="Degraded native")
    b8 = ax.bar(x + width / 2, wide[R8], width, color=BLUE, label="R8 repair")
    ax.bar_label(b0, fmt="%.3f", padding=3, fontsize=9, color=MUTED)
    ax.bar_label(b8, fmt="%.3f", padding=3, fontsize=9, color=BLUE, fontweight="bold")
    ax.set_xticks(x, [f"{int(v)} s" for v in wide.index])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(ylabel)
    ax.set_title("Frozen A10 front-triplet comparison", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.015), ncol=2)
    clean(ax)
    save(fig, filename)


def fig01_headline(headline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "Supported fault-recovery results", "Relative false-negative count reduction · frozen degraded-input comparisons")
    fig.subplots_adjust(left=0.23, right=0.95, bottom=0.11, top=0.80)
    labels = ["Front triplet missing", "Front camera missing", "A10 at 4 s", "Motion blur"]
    order = [0, 2, 1, 3]
    values = headline.iloc[order].relative_improvement_pct.to_numpy()
    y = np.arange(len(values))
    colors = [BLUE, TEAL, PURPLE, GOLD]
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=5, color=INK, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative FN reduction")
    ax.set_xlim(0, 26)
    ax.set_title("Positive results retained for the public portfolio", loc="left", color=INK, pad=13)
    clean(ax, "x")
    save(fig, "01_supported_recovery_summary.png")


def fig02_false_free(aggregate: pd.DataFrame) -> None:
    wide = a10_wide(aggregate, "false_free_rate")
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "False-free error across future horizons", "A10 front-triplet camera loss · lower is better · 20 paired samples per horizon")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.80)
    ax.plot(wide.index, wide[R0], color=GREY, marker="o", markersize=8, linewidth=2.8, label="Degraded native")
    ax.plot(wide.index, wide[R8], color=BLUE, marker="s", markersize=8, linewidth=2.8, label="R8 repair")
    for h in wide.index:
        reduction = (wide.loc[h, R0] - wide.loc[h, R8]) / wide.loc[h, R0] * 100
        ax.annotate(f"−{reduction:.1f}%", (h, wide.loc[h, R8]), xytext=(0, -24), textcoords="offset points", ha="center", color=BLUE, fontweight="bold")
    ax.set_xticks(wide.index, [f"{int(v)} s" for v in wide.index])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("False-free rate")
    ax.set_title("R8 maintains recovery through the 4D rollout", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, "02_a10_false_free_horizons.png")


def fig03_iou(aggregate: pd.DataFrame) -> None:
    grouped_metric_chart(aggregate, "occupied_iou", "Occupied IoU under front-triplet loss", "A10 · 20 paired samples per horizon", "Occupied IoU", "03_a10_occupied_iou.png")


def fig04_miou(aggregate: pd.DataFrame) -> None:
    grouped_metric_chart(aggregate, "semantic_miou", "Semantic mIoU under front-triplet loss", "A10 · 20 paired samples per horizon", "Semantic mIoU", "04_a10_semantic_miou.png")


def fig05_front_sector(aggregate: pd.DataFrame) -> None:
    grouped_metric_chart(aggregate, "front_sector_false_free", "Front-sector error after targeted repair", "A10 · failed front camera group only · lower is better", "Front-sector false-free rate", "05_front_sector_recovery.png")


def fig06_dynamic(aggregate: pd.DataFrame) -> None:
    grouped_metric_chart(aggregate, "dynamic_false_free", "Dynamic-object recovery", "A10 · 20 paired samples per horizon · lower is better", "Dynamic-object false-free rate", "06_dynamic_object_recovery.png")


def fig07_small_object(aggregate: pd.DataFrame) -> None:
    grouped_metric_chart(aggregate, "small_object_false_free", "Small-object recovery", "A10 · 20 paired samples per horizon · lower is better", "Small-object false-free rate", "07_small_object_recovery.png")


def distribution_chart(rows: pd.DataFrame, metric: str, direction: str, title: str, ylabel: str, filename: str) -> None:
    wide = paired_rows(rows, metric)
    if direction == "higher":
        wide["delta"] = wide[R8] - wide[R0]
    else:
        wide["delta"] = wide[R0] - wide[R8]
    horizons = sorted(wide.horizon_s.unique())
    groups = [wide.loc[wide.horizon_s == h, "delta"].to_numpy() for h in horizons]
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, title, "A10 · 20 paired samples at each horizon · positive values favor R8")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.80)
    box = ax.boxplot(groups, patch_artist=True, widths=0.54, showmeans=True, meanprops={"marker": "D", "markerfacecolor": GOLD, "markeredgecolor": "white", "markersize": 7})
    for patch in box["boxes"]:
        patch.set_facecolor(BLUE_LIGHT)
        patch.set_edgecolor(BLUE)
    for key in ["whiskers", "caps", "medians"]:
        for line in box[key]:
            line.set_color(INK)
    rng = np.random.default_rng(7)
    for i, values in enumerate(groups, start=1):
        ax.scatter(rng.normal(i, 0.035, size=len(values)), values, s=24, color=BLUE, alpha=0.52, edgecolors="none")
    ax.axhline(0, color=INK, linewidth=1.2, linestyle="--")
    ax.set_xticks(range(1, len(horizons) + 1), [f"{int(h)} s" for h in horizons])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(ylabel)
    ax.set_title("Paired sample-level improvement distribution", loc="left", color=INK, pad=13)
    clean(ax)
    save(fig, filename)


def fig08_sample_iou(rows: pd.DataFrame) -> None:
    distribution_chart(rows, "occupied_iou", "higher", "Sample-level occupied-IoU improvement", "R8 − degraded occupied IoU", "08_sample_iou_improvement.png")


def fig09_sample_false_free(rows: pd.DataFrame) -> None:
    distribution_chart(rows, "false_free_rate", "lower", "Sample-level false-free reduction", "Degraded − R8 false-free rate", "09_sample_false_free_reduction.png")


def fig10_sample_miou(rows: pd.DataFrame) -> None:
    distribution_chart(rows, "semantic_miou", "higher", "Sample-level semantic-mIoU improvement", "R8 − degraded semantic mIoU", "10_sample_semantic_improvement.png")


def fig11_pareto(tradeoff: pd.DataFrame) -> None:
    x = tradeoff[tradeoff.perturbation_id == A10].copy()
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "Recovery–density trade-off", "A10 supported temporal-memory variants · each point is one variant × horizon")
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.80)
    labels = {"R1_replace_tminus1": "t−1 replace", "R2_replace_tminus2": "t−2 replace", "R3_ema_K2": "EMA K2", "R4_ema_K3": "EMA K3", R8: "R8 front triplet"}
    colors = {"t−1 replace": TEAL, "t−2 replace": GOLD, "EMA K2": PURPLE, "EMA K3": PINK, "R8 front triplet": BLUE}
    for key, group in x.groupby("variant_label"):
        label = labels.get(key, key)
        ax.plot(group.density_drift * 100, group.recovery_rate * 100, marker="o", linewidth=2, color=colors[label], label=label, alpha=0.86)
    ax.set_xlabel("Occupancy density drift (%)")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_title("Causal memory variants occupy a measurable recovery frontier", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="lower right", ncols=2)
    clean(ax)
    save(fig, "11_recovery_density_pareto.png")


def fig12_memory_age(tradeoff: pd.DataFrame) -> None:
    x = tradeoff[(tradeoff.perturbation_id == "A1_drop_cam_front") & tradeoff.variant_label.isin(["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2", "R4_ema_K3"])].copy()
    labels = ["t−1", "t−2", "EMA K2", "EMA K3"]
    variants = ["R1_replace_tminus1", "R2_replace_tminus2", "R3_ema_K2", "R4_ema_K3"]
    matrix = x.pivot(index="variant_label", columns="horizon_s", values="recovery_rate").reindex(variants) * 100
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "Temporal-memory age ablation", "A1 single front-camera loss · recovery rate (%) · 20 paired samples per cell")
    fig.subplots_adjust(left=0.15, right=0.93, bottom=0.15, top=0.80)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=15, vmax=38, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="white" if value > 27 else INK, fontweight="bold")
    ax.set_yticks(np.arange(4), labels)
    ax.set_xticks(np.arange(4), [f"{int(v)} s" for v in matrix.columns])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Memory strategy")
    ax.set_title("Recent and EMA memory remain effective across horizons", loc="left", color=INK, pad=13)
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
    cbar.set_label("Recovery rate (%)")
    save(fig, "12_memory_age_ablation.png")


def fig13_blur_blend(tradeoff: pd.DataFrame) -> None:
    x = tradeoff[tradeoff.perturbation_id == "C4_motion_blur_9"].copy()
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "Temporal blending under motion blur", "C4 motion blur · supported recovery rates by blend coefficient and future horizon")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.80)
    mapping = {"R5_blend_alpha03": ("α=0.3", TEAL), "R6_blend_alpha05": ("α=0.5", GOLD), "R7_blend_alpha07": ("α=0.7", BLUE)}
    for variant, (label, color) in mapping.items():
        group = x[x.variant_label == variant].sort_values("horizon_s")
        ax.plot(group.horizon_s, group.recovery_rate * 100, marker="o", markersize=7, linewidth=2.7, color=color, label=label)
    ax.set_xticks([0, 2, 4, 6], ["0 s", "2 s", "4 s", "6 s"])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_title("Causal feature blending recovers future occupancy evidence", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, "13_motion_blur_blend_recovery.png")


def fig14_scenario_matrix(headline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "Supported recovery by fault scope", "Published relative FN count reduction · exact values are retained in docs/RESULTS.md")
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.14, top=0.80)
    x = np.arange(len(headline))
    bars = ax.bar(x, headline.relative_improvement_pct, color=[BLUE, PURPLE, TEAL, GOLD], width=0.62)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in headline.relative_improvement_pct], padding=4, color=INK, fontweight="bold")
    ax.set_xticks(x, ["A10\nfront triplet", "A10\n4 s horizon", "A1\nfront camera", "C4\nmotion blur"])
    ax.set_ylabel("Relative FN reduction (%)")
    ax.set_title("Positive recovery spans camera loss and visual degradation", loc="left", color=INK, pad=13)
    clean(ax)
    save(fig, "14_fault_scope_summary.png")


def fig15_density(aggregate: pd.DataFrame) -> None:
    grouped_metric_chart(aggregate, "pred_gt_occupied_ratio", "Predicted-to-GT occupancy density", "A10 · density is reported alongside recovery to expose over-occupation risk", "Predicted / GT occupied ratio", "15_occupancy_density_guardrail.png")


def fig16_risk_balance(aggregate: pd.DataFrame) -> None:
    ff = a10_wide(aggregate, "false_free_rate")
    fo = a10_wide(aggregate, "false_occupied_rate")
    benefit = (ff[R0] - ff[R8]) * 100
    risk = (fo[R8] - fo[R0]) * 100
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "Recovery benefit versus false-occupied cost", "A10 · percentage-point change from degraded native · one point per prediction horizon")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.80)
    scatter = ax.scatter(risk, benefit, c=ff.index, cmap="viridis", s=170, edgecolor="white", linewidth=1.5)
    for h, xx, yy in zip(ff.index, risk, benefit):
        ax.annotate(f"{int(h)} s", (xx, yy), xytext=(8, 5), textcoords="offset points", color=INK, fontweight="bold")
    ax.axvline(0, color=INK, linewidth=1.1, linestyle="--")
    ax.set_xlabel("False-occupied rate increase (pp)")
    ax.set_ylabel("False-free rate reduction (pp)")
    ax.set_title("Large recovery is achieved with a small measured risk increment", loc="left", color=INK, pad=13)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Prediction horizon (s)")
    clean(ax)
    save(fig, "16_recovery_risk_balance.png")


def card(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, value: str, color: str = BLUE_LIGHT) -> None:
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=color, edgecolor="white", linewidth=1.2, transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(xy[0] + 0.04, xy[1] + height * 0.63, value, transform=ax.transAxes, color=INK, fontsize=24, fontweight="bold", va="center")
    ax.text(xy[0] + 0.04, xy[1] + height * 0.27, title, transform=ax.transAxes, color=MUTED, fontsize=10.5, va="center")


def fig17_tensor_contract() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    header(fig, "SparseWorld 4D tensor contract", "Configuration-level dimensions used by the reliability experiments")
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.82)
    ax.axis("off")
    card(ax, (0.03, 0.56), 0.28, 0.27, "temporal camera images", "5 × 6 = 30", BLUE_LIGHT)
    card(ax, (0.36, 0.56), 0.28, 0.27, "current / future queries", "720 + 320", TEAL_LIGHT)
    card(ax, (0.69, 0.56), 0.28, 0.27, "total sparse queries", "1,040", GOLD_LIGHT)
    card(ax, (0.03, 0.17), 0.28, 0.27, "occupancy grid", "200 × 200 × 16", "#F3E8FF")
    card(ax, (0.36, 0.17), 0.28, 0.27, "dense voxels", "640,000", "#FCE7F3")
    card(ax, (0.69, 0.17), 0.28, 0.27, "future outputs", "0 / 2 / 4 / 6 s", "#FFEDD5")
    ax.text(0.03, 0.04, "The reliability layer preserves the upstream tensor contract and intervenes only on failed-camera FPN features.", transform=ax.transAxes, color=MUTED, fontsize=11)
    save(fig, "17_sparseworld_tensor_contract.png")


def draw_flow(ax: plt.Axes, labels: list[str], colors: list[str]) -> None:
    ax.axis("off")
    n = len(labels)
    # Keep a positive inter-node gap. Seven 0.145-wide boxes exceeded the
    # available canvas and visually reversed the arrows.
    width = 0.122
    gap = (0.94 - n * width) / (n - 1)
    y = 0.39
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.03 + i * (width + gap)
        patch = FancyBboxPatch((x, y), width, 0.24, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=color, edgecolor="white", linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.12, label, ha="center", va="center", transform=ax.transAxes, color=INK, fontsize=9.7, fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(x + width + gap * 0.82, y + 0.12), xytext=(x + width + gap * 0.18, y + 0.12), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.8})


def fig18_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    header(fig, "R8 causal feature-memory pipeline", "Historical features provide visual evidence; the frozen SparseWorld decoder and occupancy head remain unchanged")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.82)
    labels = ["5-frame\n6-camera input", "Backbone +\n4-level FPN", "Fault camera\ndetection", "Same-camera\nt−1 cache", "R8 feature\nreplacement", "Frozen OPUS\ndecoder", "4D semantic\nOccupancy"]
    colors = [BLUE_LIGHT, BLUE_LIGHT, GOLD_LIGHT, TEAL_LIGHT, "#F3E8FF", "#E2E8F0", "#FCE7F3"]
    draw_flow(ax, labels, colors)
    ax.text(0.5, 0.18, "No future frames · no current-clean oracle · no GT-guided repair · healthy cameras pass through unchanged", ha="center", transform=ax.transAxes, color=TEAL, fontsize=12, fontweight="bold")
    save(fig, "18_r8_causal_pipeline.png")


def fig19_atlas() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    header(fig, "Experiment atlas from bring-up to causal memory", "31 staged experiments · public emphasis remains on supported infrastructure and R8")
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.82)
    stages = [
        ("SW1–4", "Bring-up, geometry,\nquery support", 4, BLUE),
        ("SW5–7", "Fault injection and\nreliability maps", 3, TEAL),
        ("SW8–12", "Training, routing,\nand safety search", 5, GOLD),
        ("SW13", "Causal temporal\nfeature memory", 3, PURPLE),
        ("R8", "Front-triplet\nfeature repair", 1, PINK),
        ("SW14+", "Guardrails and\nmechanism audits", 15, GREY),
    ]
    left = 0
    for label, desc, count, color in stages:
        ax.barh([0], [count], left=left, height=0.48, color=color, edgecolor="white")
        center = left + count / 2
        if count >= 3:
            ax.text(center, 0, f"{label}\n{count} stages", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        else:
            ax.annotate(label, (center, 0.25), xytext=(center, 0.55), ha="center", color=color, fontweight="bold", arrowprops={"arrowstyle": "-", "color": color})
        left += count
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.8, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Cumulative staged experiments")
    ax.set_title("Engineering progression", loc="left", color=INK, pad=13)
    clean(ax, "x")
    y0 = -0.50
    for i, (label, desc, _, color) in enumerate(stages):
        x = 0.02 + (i % 3) * 0.33
        y = y0 - (i // 3) * 0.22
        ax.text(x, y, label, transform=ax.transAxes, color=color, fontweight="bold", fontsize=10)
        ax.text(x + 0.065, y, desc.replace("\n", " "), transform=ax.transAxes, color=MUTED, fontsize=9.5)
    save(fig, "19_experiment_atlas.png")


def fig20_evidence(headline: pd.DataFrame, aggregate: pd.DataFrame, rows: pd.DataFrame, tradeoff: pd.DataFrame) -> None:
    labels = ["Per-sample replay", "Aggregate metrics", "Trade-off matrix", "Headline results"]
    values = [len(rows), len(aggregate), len(tradeoff), len(headline)]
    colors = [BLUE, TEAL, GOLD, PURPLE]
    fig = plt.figure(figsize=(14.5, 7.5))
    header(fig, "Public evidence coverage", f"{sum(values):,} machine-readable records · supported SW13A/R8 evidence only")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.96, bottom=0.12, top=0.80, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(4)
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, color=INK, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, max(values) * 2)
    ax.set_xlabel("Record count (log scale)")
    ax.set_title("Evidence tables", loc="left", color=INK, pad=13)
    clean(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    card(ax, (0.06, 0.56), 0.40, 0.28, "paired samples", "20 / horizon", BLUE_LIGHT)
    card(ax, (0.54, 0.56), 0.40, 0.28, "future horizons", "4", TEAL_LIGHT)
    card(ax, (0.06, 0.17), 0.40, 0.28, "fault families", "2", GOLD_LIGHT)
    card(ax, (0.54, 0.17), 0.40, 0.28, "no-oracle checks", "passed", "#F3E8FF")
    ax.set_title("Frozen evaluation contract", loc="left", color=INK, pad=13)
    save(fig, "20_public_evidence_coverage.png")


def main() -> None:
    setup()
    headline, aggregate, rows, tradeoff = load()
    fig01_headline(headline)
    fig02_false_free(aggregate)
    fig03_iou(aggregate)
    fig04_miou(aggregate)
    fig05_front_sector(aggregate)
    fig06_dynamic(aggregate)
    fig07_small_object(aggregate)
    fig08_sample_iou(rows)
    fig09_sample_false_free(rows)
    fig10_sample_miou(rows)
    fig11_pareto(tradeoff)
    fig12_memory_age(tradeoff)
    fig13_blur_blend(tradeoff)
    fig14_scenario_matrix(headline)
    fig15_density(aggregate)
    fig16_risk_balance(aggregate)
    fig17_tensor_contract()
    fig18_pipeline()
    fig19_atlas()
    fig20_evidence(headline, aggregate, rows, tradeoff)


if __name__ == "__main__":
    main()
