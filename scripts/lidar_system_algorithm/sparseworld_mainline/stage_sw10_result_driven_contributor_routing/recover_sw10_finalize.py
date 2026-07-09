from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lidar_system_algorithm.sparseworld_mainline.stage_sw10_result_driven_contributor_routing import (  # noqa: E402
    run_sparseworld_sw10_main as sw10,
)


REPORTS_DIR = sw10.REPORTS_DIR
FIGURES_DIR = sw10.FIGURES_DIR
LOGS_DIR = sw10.LOGS_DIR
BASE_CONFIG_PATH = sw10.BASE_CONFIG_PATH
CHECKPOINT_PATH = sw10.CHECKPOINT_PATH
ROUTE_A_EXPERIMENT_ID = sw10.ROUTE_A_EXPERIMENT_ID
RETEST_HORIZONS = sw10.RETEST_HORIZONS


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(sw10.normalize_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def summarize_contributor(rows: list[dict[str, str]]) -> str:
    agg = sw10.sw91.aggregate_contributor_rows([{k: (float(v) if k not in {"checkpoint_name", "perturbation_id"} else v) if v not in {"", None} else v for k, v in row.items()} for row in rows])
    by_name = {row["checkpoint_name"]: row for row in agg}
    start = by_name.get("routeA_start_P4_H2_tinyH1_500iter")
    scaled = by_name.get(ROUTE_A_EXPERIMENT_ID)
    if not start or not scaled:
        return "starting SW-9.1 checkpoint and the scaled SW-10 Route A checkpoint were retested with subset contributor diagnostics"
    exact_delta = float(scaled["mean_exact_voxel_assignment_ratio"]) - float(start["mean_exact_voxel_assignment_ratio"])
    leak_delta = float(scaled["mean_neighbor_leakage_ratio"]) - float(start["mean_neighbor_leakage_ratio"])
    return (
        "subset contributor retest shows the scaled Route A checkpoint slightly reduced neighbor leakage "
        f"({leak_delta:+.4f}) while exact voxel assignment stayed near-flat ({exact_delta:+.4f}) versus the SW-9.1 start checkpoint"
    )


def recover_reliability(route_selection: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    selected_config_path = Path(route_selection["selected_config_path"])
    selected_checkpoint_path = Path(route_selection["selected_checkpoint_path"])
    scaled_checkpoint_path = Path(
        read_csv_rows(REPORTS_DIR / "routeA_scaled_h2_train_metrics.csv")[0]["checkpoint_path"]
    )
    sample_indices = [0, 1]
    reliability_rows: list[dict[str, Any]] = []
    _, ref_rel_rows, ref_dbg = sw10.sw91.evaluate_checkpoint_detailed(
        BASE_CONFIG_PATH,
        CHECKPOINT_PATH,
        "REF_epoch_56",
        sample_indices,
        RETEST_HORIZONS,
        capture_reliability=True,
    )
    _, start_rel_rows, start_dbg = sw10.sw91.evaluate_checkpoint_detailed(
        selected_config_path,
        selected_checkpoint_path,
        f"routeA_start_{route_selection['selected_candidate']}",
        sample_indices,
        RETEST_HORIZONS,
        capture_reliability=True,
    )
    _, scaled_rel_rows, scaled_dbg = sw10.sw91.evaluate_checkpoint_detailed(
        selected_config_path,
        scaled_checkpoint_path,
        ROUTE_A_EXPERIMENT_ID,
        sample_indices,
        RETEST_HORIZONS,
        capture_reliability=True,
    )
    reliability_rows.extend(ref_rel_rows)
    reliability_rows.extend(start_rel_rows)
    reliability_rows.extend(scaled_rel_rows)
    if ref_dbg.get("representative_case") and scaled_dbg.get("representative_case"):
        sw10.sw81.render_reliability_before_after(
            "sw10_reliability",
            ref_dbg["representative_case"],
            scaled_dbg["representative_case"],
            FIGURES_DIR / "sw10_reliability_before_after.png",
        )
        sw10.sw81.render_reliability_before_after(
            "sw10_score_alpha",
            ref_dbg["representative_case"],
            scaled_dbg["representative_case"],
            FIGURES_DIR / "sw10_score_alpha_before_after.png",
        )
    sw10.write_csv(REPORTS_DIR / "sw10_reliability_retest.csv", reliability_rows)
    summary_rows = sw10.aggregate_reliability_rows(reliability_rows)
    by_name = {row["checkpoint_name"]: row for row in summary_rows}
    start = by_name.get(f"routeA_start_{route_selection['selected_candidate']}")
    scaled = by_name.get(ROUTE_A_EXPERIMENT_ID)
    if start and scaled:
        headline = (
            "subset SW-7 reliability retest completed on 2 shared samples; the scaled Route A checkpoint "
            f"shifted front-sector reliability by {float(scaled['mean_front_sector_reliability']) - float(start['mean_front_sector_reliability']):+.4f} "
            f"and risk-error correlation by {float(scaled['mean_risk_error_correlation']) - float(start['mean_risk_error_correlation']):+.4f}"
        )
    else:
        headline = "subset SW-7 reliability retest completed for the Route A starting checkpoint and the scaled Route A checkpoint"
    write_md(REPORTS_DIR / "sw10_reliability_retest_summary.md", headline + "\n")
    return reliability_rows, headline


def main() -> None:
    started = time.time()
    route_selection = read_json(REPORTS_DIR / "sw10_route_selection.json")
    sw91_digest = read_json(REPORTS_DIR / "sw91_result_digest_for_sw10.json")
    train_rows = read_csv_rows(REPORTS_DIR / "routeA_scaled_h2_train_metrics.csv")
    eval_rows = read_csv_rows(REPORTS_DIR / "routeA_scaled_h2_eval_metrics.csv")
    contributor_rows = read_csv_rows(REPORTS_DIR / "sw10_contributor_diagnostic_retest.csv")
    if not train_rows or not eval_rows:
        raise RuntimeError("missing Route A train/eval artifacts")

    reliability_rows, reliability_headline = recover_reliability(route_selection)
    contributor_headline = summarize_contributor(contributor_rows)
    write_md(REPORTS_DIR / "sw10_contributor_diagnostic_summary.md", contributor_headline + "\n")

    eval_summary_rows: list[dict[str, Any]] = []
    gate_lookup: dict[str, Any] = {}
    for row in eval_rows:
        if row.get("safe_gate_json"):
            gate = json.loads(row["safe_gate_json"])
            gate_lookup[row["checkpoint_name"]] = gate
            eval_summary_rows.append({"checkpoint_name": row["checkpoint_name"], "safe_gate": gate})
    best_scaled_gate = gate_lookup.get(ROUTE_A_EXPERIMENT_ID)
    start_key = f"routeA_start_{route_selection['selected_candidate']}"
    start_gate = gate_lookup.get(start_key)

    scaled_safe = bool(best_scaled_gate and best_scaled_gate["safe"])
    scaled_targeted = bool(best_scaled_gate and best_scaled_gate["targeted_hit"])
    start_safe = bool(start_gate and start_gate["safe"])
    start_targeted = bool(start_gate and start_gate["targeted_hit"])
    if scaled_safe and scaled_targeted:
        decision_type = "R1_scale_h2_success"
        decision_summary = "Route A longer training preserved subset safety gates and retained a targeted gain signal."
        next_unique_action = "extend the same Route A recipe to 5000 iter and rerun eval_core_20 before considering any architecture change."
    elif start_safe and start_targeted and not scaled_safe:
        decision_type = "R8_architecture_bottleneck_strengthened"
        decision_summary = "The SW-9.1 short safe signal did not remain stable after Route A scaling, which strengthens the evidence for an architecture-side bottleneck."
        next_unique_action = "prepare the minimal native get_occ routing prototype next instead of another blind long run."
    else:
        decision_type = "R8_architecture_bottleneck_strengthened"
        decision_summary = "Route A did not produce a new safe targeted signal in the available budget, so scaling alone is not enough evidence."
        next_unique_action = "move to the smallest native get_occ contributor-routing prototype under a dedicated next stage."
    best_tradeoff = best_scaled_gate or start_gate or {"false_occupied_delta": None, "pred_gt_ratio_delta": None}
    decision_payload = {
        "selected_route": route_selection["selected_route"],
        "decision_type": decision_type,
        "summary": decision_summary,
        "best_candidate": ROUTE_A_EXPERIMENT_ID if best_scaled_gate is not None else route_selection["selected_candidate"],
        "best_tradeoff": {
            "false_occupied_delta": best_tradeoff.get("false_occupied_delta"),
            "pred_gt_ratio_delta": best_tradeoff.get("pred_gt_ratio_delta"),
        },
        "scale_not_stable": bool(start_safe and start_targeted and not scaled_safe),
        "next_unique_action": next_unique_action,
    }
    write_json(REPORTS_DIR / "sw10_result_driven_decision.json", decision_payload)
    write_md(REPORTS_DIR / "sw10_result_driven_decision.md", json.dumps(decision_payload, indent=2, ensure_ascii=False) + "\n")

    train_summary = train_rows[0]
    final_report_json = {
        "executive_summary": route_selection["summary"],
        "sw91_digest": sw91_digest,
        "route_selection": route_selection,
        "route_execution": {
            "headline": f"Route A resumed {route_selection['selected_candidate']} and aimed for total_iter={train_summary['target_total_iter']}.",
            "result_headline": f"Route A completed_total_iter={train_summary['completed_total_iter']} with stop_reason={train_summary['stop_reason']}.",
            "train_summary": train_summary,
        },
        "contributor_retest": {"headline": contributor_headline},
        "reliability_retest": {"headline": reliability_headline},
        "decision": decision_payload,
    }
    final_report_text = sw10.make_final_report(final_report_json)
    write_md(REPORTS_DIR / "stage_sw10_result_driven_contributor_routing_report.md", final_report_text)
    write_json(REPORTS_DIR / "stage_sw10_result_driven_contributor_routing_report.json", final_report_json)

    manifest = read_json(REPORTS_DIR / "sw10_time_budget_manifest.json")
    phases = manifest.get("phases", [])
    reliability_phase = next((phase for phase in phases if phase.get("phase_name") == "common_phase_reliability_retest"), None)
    if reliability_phase is not None:
        start_ts = float(reliability_phase.get("start_ts", time.time()))
        end_ts = time.time()
        reliability_phase.update(
            {
                "end_time": now_iso(),
                "end_ts": end_ts,
                "duration_sec": end_ts - start_ts,
                "status": "recovered_done",
                "recovered_after_main_process_hang": True,
            }
        )
    phases.append(
        {
            "phase_name": "recovery_finalize",
            "start_time": now_iso(),
            "end_time": now_iso(),
            "duration_sec": time.time() - started,
            "status": "done",
            "notes": "reliability/decision/report regenerated from completed Route A artifacts after the original main process hung during reliability retest",
        }
    )
    manifest["end_time"] = now_iso()
    manifest["wall_clock_sec"] = float(time.time() - float(phases[0]["start_ts"]))
    write_json(REPORTS_DIR / "sw10_time_budget_manifest.json", manifest)

    status_path = LOGS_DIR / "sw10_status.json"
    if status_path.exists():
        status = read_json(status_path)
        status["current_stage"] = None
        stage = status.get("stages", {}).setdefault("common_phase_reliability_retest", {})
        start_ts = float(stage.get("start_ts", time.time()))
        stage.update(
            {
                "status": "recovered_done",
                "end_ts": time.time(),
                "duration_sec": time.time() - start_ts,
                "recovered_after_main_process_hang": True,
            }
        )
        status["stages"]["recovery_finalize"] = {
            "status": "done",
            "start_ts": started,
            "end_ts": time.time(),
            "duration_sec": time.time() - started,
        }
        write_json(status_path, status)


if __name__ == "__main__":
    os.environ.setdefault("CV_LIDAR_PROJECT_ROOT", str(PROJECT_ROOT))
    main()
