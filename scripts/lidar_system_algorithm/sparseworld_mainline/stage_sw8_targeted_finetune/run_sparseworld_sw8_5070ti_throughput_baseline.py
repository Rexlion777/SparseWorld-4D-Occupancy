from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", "/mnt/d/ComputerVision/cv_lidar_transition"))
SCRIPT_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune/throughput_baseline"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw8_targeted_finetune/throughput_baseline"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw8_targeted_finetune/throughput_baseline"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw8_targeted_finetune/throughput_baseline"
WINDOWS_PROJECT_ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw8 = load_module("sw8_mainline_runtime", SCRIPT_DIR / "run_sparseworld_sw8_main.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-grid", default="2x4,2x6,3x4,3x6")
    p.add_argument("--iters-per-candidate", type=int, default=8)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--enable-tf32", action="store_true", default=True)
    p.add_argument("--enable-cudnn-benchmark", action="store_true", default=True)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--sample-util-interval-sec", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=1)
    return p.parse_args()


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, LOGS_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def to_windows_path(path: str | Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def normalize_export(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, Path):
        return to_windows_path(obj)
    if isinstance(obj, str):
        return to_windows_path(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class StatusLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "5070ti_baseline_status.json"
        self.progress_path = LOGS_DIR / "5070ti_baseline_progress.jsonl"
        self.state: dict[str, Any] = {"started_at": time.time(), "current_candidate": None, "candidates": {}}
        self.progress_path.write_text("", encoding="utf-8")
        self.flush()

    def flush(self) -> None:
        self.status_path.write_text(json.dumps(normalize_export(self.state), indent=2, ensure_ascii=False), encoding="utf-8")

    def event(self, candidate: str, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalize_export({"ts": time.time(), "candidate": candidate, "event": event, **payload}), ensure_ascii=False) + "\n")
        self.flush()

    def start(self, candidate: str, **payload: Any) -> None:
        self.state["current_candidate"] = candidate
        self.state["candidates"][candidate] = {"status": "running", "start_ts": time.time(), **payload}
        self.event(candidate, "start", payload)

    def progress(self, candidate: str, current: int, total: int, **payload: Any) -> None:
        item = self.state["candidates"].setdefault(candidate, {})
        item.update({"status": "running", "current": current, "total": total, **payload})
        self.event(candidate, "progress", {"current": current, "total": total, **payload})

    def done(self, candidate: str, **payload: Any) -> None:
        now = time.time()
        item = self.state["candidates"].setdefault(candidate, {})
        item.update({"status": "done", "end_ts": now, "duration_sec": now - float(item.get("start_ts", now)), **payload})
        if self.state.get("current_candidate") == candidate:
            self.state["current_candidate"] = None
        self.event(candidate, "done", payload)

    def fail(self, candidate: str, error: str, **payload: Any) -> None:
        now = time.time()
        item = self.state["candidates"].setdefault(candidate, {})
        item.update({"status": "failed", "end_ts": now, "duration_sec": now - float(item.get("start_ts", now)), "error": error, **payload})
        if self.state.get("current_candidate") == candidate:
            self.state["current_candidate"] = None
        self.event(candidate, "failed", {"error": error, **payload})


class GPUSampler:
    def __init__(self, interval_sec: float) -> None:
        self.interval_sec = interval_sec
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                line = proc.stdout.strip().splitlines()[0]
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 6:
                    self.rows.append(
                        {
                            "ts_text": parts[0],
                            "gpu_util": float(parts[1]),
                            "mem_util": float(parts[2]),
                            "mem_used_mb": float(parts[3]),
                            "mem_total_mb": float(parts[4]),
                            "power_w": float(parts[5]) if parts[5] not in {"[N/A]", "N/A"} else None,
                        }
                    )
            except Exception:
                pass
            self._stop.wait(self.interval_sec)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_candidate_grid(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        batch_s, workers_s = chunk.split("x", 1)
        out.append((int(batch_s), int(workers_s)))
    return out


def build_dataloader_with_extras(dataset: Any, samples_per_gpu: int, workers_per_gpu: int, seed: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int):
    try:
        from mmdet3d.datasets import build_dataloader
    except Exception:
        from mmdet.datasets import build_dataloader
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "samples_per_gpu": samples_per_gpu,
        "workers_per_gpu": workers_per_gpu,
        "num_gpus": 1,
        "dist": False,
        "shuffle": True,
        "seed": seed,
    }
    sig = inspect.signature(build_dataloader)
    if "pin_memory" in sig.parameters:
        kwargs["pin_memory"] = pin_memory
    if "persistent_workers" in sig.parameters and workers_per_gpu > 0:
        kwargs["persistent_workers"] = persistent_workers
    if "prefetch_factor" in sig.parameters and workers_per_gpu > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return build_dataloader(**kwargs)


def set_fast_runtime(enable_tf32: bool, enable_cudnn_benchmark: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
        torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
    torch.backends.cudnn.benchmark = bool(enable_cudnn_benchmark)


def benchmark_candidate(
    batch_size: int,
    workers: int,
    args: argparse.Namespace,
    status: StatusLogger,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_name = f"bs{batch_size}_wk{workers}"
    status.start(candidate_name, batch_size=batch_size, workers=workers, iters=args.iters_per_candidate)
    iter_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    try:
        seed_everything(args.seed)
        set_fast_runtime(args.enable_tf32, args.enable_cudnn_benchmark)
        cfg_overrides = {
            "data.samples_per_gpu": batch_size,
            "data.workers_per_gpu": workers,
        }
        cfg, dataset, model, _ = sw8.build_sparseworld_runtime(train=True, cfg_overrides=cfg_overrides)
        dataloader = build_dataloader_with_extras(
            dataset,
            samples_per_gpu=batch_size,
            workers_per_gpu=workers,
            seed=args.seed,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        from mmcv.runner import build_optimizer

        optimizer = build_optimizer(model, cfg.optimizer)
        data_iter = iter(dataloader)
        sampler = GPUSampler(args.sample_util_interval_sec)
        sampler.start()
        overall_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        valid_iters = 0
        for iter_idx in range(1, args.iters_per_candidate + 1):
            fetch_started = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            fetch_sec = time.perf_counter() - fetch_started

            optimizer.zero_grad(set_to_none=True)
            transfer_started = time.perf_counter()
            model_inputs = sw8.move_train_batch_to_cuda(batch)
            transfer_sec = time.perf_counter() - transfer_started

            step_started = time.perf_counter()
            losses = model(return_loss=True, **model_inputs)
            total_loss, log_vars = sw8.parse_losses(losses)
            if not bool(torch.isfinite(total_loss).item()):
                raise RuntimeError(f"non-finite loss: {log_vars}")
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).detach().cpu().item())
            optimizer.step()
            torch.cuda.synchronize()
            step_sec = time.perf_counter() - step_started

            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            row = {
                "candidate": candidate_name,
                "iter": iter_idx,
                "batch_size": batch_size,
                "workers": workers,
                "fetch_sec": fetch_sec,
                "transfer_sec": transfer_sec,
                "step_sec": step_sec,
                "total_iter_sec": fetch_sec + transfer_sec + step_sec,
                "loss_total": float(log_vars["loss_total"]),
                "grad_norm": grad_norm,
                "peak_mem_mb": peak_mem_mb,
                "warmup": iter_idx <= args.warmup_iters,
            }
            iter_rows.append(row)
            if iter_idx > args.warmup_iters:
                valid_iters += 1
            if (iter_idx % args.log_interval == 0) or (iter_idx == args.iters_per_candidate):
                status.progress(candidate_name, iter_idx, args.iters_per_candidate, last_loss=row["loss_total"], peak_mem_mb=peak_mem_mb)

        torch.cuda.synchronize()
        total_wall_sec = time.perf_counter() - overall_start
        sampler.stop()
        gpu_rows = [{"candidate": candidate_name, **r} for r in sampler.rows]
        effective_rows = [r for r in iter_rows if not r["warmup"]] or iter_rows
        mem_total_mb = max([r["mem_total_mb"] for r in sampler.rows], default=0.0)
        summary = {
            "candidate": candidate_name,
            "status": "ok",
            "batch_size": batch_size,
            "workers": workers,
            "iters": args.iters_per_candidate,
            "warmup_iters": args.warmup_iters,
            "effective_iters": len(effective_rows),
            "throughput_samples_per_sec": batch_size * len(effective_rows) / max(sum(float(r["total_iter_sec"]) for r in effective_rows), 1e-6),
            "mean_fetch_sec": statistics.mean(float(r["fetch_sec"]) for r in effective_rows),
            "mean_transfer_sec": statistics.mean(float(r["transfer_sec"]) for r in effective_rows),
            "mean_step_sec": statistics.mean(float(r["step_sec"]) for r in effective_rows),
            "mean_total_iter_sec": statistics.mean(float(r["total_iter_sec"]) for r in effective_rows),
            "data_wait_ratio": sum(float(r["fetch_sec"]) for r in effective_rows) / max(sum(float(r["total_iter_sec"]) for r in effective_rows), 1e-6),
            "peak_mem_mb": max(float(r["peak_mem_mb"]) for r in iter_rows),
            "gpu_util_mean": statistics.mean(float(r["gpu_util"]) for r in sampler.rows) if sampler.rows else None,
            "gpu_util_p90": float(np.percentile([float(r["gpu_util"]) for r in sampler.rows], 90)) if sampler.rows else None,
            "mem_util_mean": statistics.mean(float(r["mem_util"]) for r in sampler.rows) if sampler.rows else None,
            "power_w_mean": statistics.mean(float(r["power_w"]) for r in sampler.rows if r["power_w"] is not None) if any(r["power_w"] is not None for r in sampler.rows) else None,
            "loss_last": float(iter_rows[-1]["loss_total"]),
            "loss_first_effective": float(effective_rows[0]["loss_total"]),
            "loss_delta_effective": float(effective_rows[-1]["loss_total"]) - float(effective_rows[0]["loss_total"]),
            "total_wall_sec": total_wall_sec,
            "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers and workers > 0,
            "prefetch_factor": args.prefetch_factor if workers > 0 else None,
            "memory_headroom_mb": float(mem_total_mb - max(float(r["peak_mem_mb"]) for r in iter_rows)) if mem_total_mb else None,
        }
        status.done(candidate_name, throughput_samples_per_sec=summary["throughput_samples_per_sec"], peak_mem_mb=summary["peak_mem_mb"], gpu_util_mean=summary["gpu_util_mean"])
        return summary, iter_rows, gpu_rows
    except RuntimeError as exc:
        msg = repr(exc)
        torch.cuda.empty_cache()
        status.fail(candidate_name, msg)
        return {
            "candidate": candidate_name,
            "status": "oom" if "out of memory" in msg.lower() else "failed",
            "batch_size": batch_size,
            "workers": workers,
            "iters": args.iters_per_candidate,
            "error": msg,
        }, iter_rows, gpu_rows
    except Exception as exc:
        msg = repr(exc)
        status.fail(candidate_name, msg)
        return {
            "candidate": candidate_name,
            "status": "failed",
            "batch_size": batch_size,
            "workers": workers,
            "iters": args.iters_per_candidate,
            "error": msg,
        }, iter_rows, gpu_rows


def choose_recommendations(summary_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    if not ok_rows:
        return None, None

    def throughput_score(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row.get("throughput_samples_per_sec", 0.0) or 0.0),
            float(row.get("gpu_util_mean", 0.0) or 0.0),
            float(row.get("memory_headroom_mb", 0.0) or 0.0),
        )

    max_row = max(ok_rows, key=throughput_score)
    stable_pool = [
        r for r in ok_rows
        if float(r.get("memory_headroom_mb", 0.0) or 0.0) >= 1200.0
        and float(r.get("data_wait_ratio", 1.0) or 1.0) <= 0.25
    ] or ok_rows
    stable_row = max(
        stable_pool,
        key=lambda r: (
            float(r.get("gpu_util_mean", 0.0) or 0.0),
            float(r.get("throughput_samples_per_sec", 0.0) or 0.0),
            -float(r.get("mean_total_iter_sec", 0.0) or 0.0),
        ),
    )
    return max_row, stable_row


def config_payload_from_row(row: dict[str, Any], mode: str) -> dict[str, Any]:
    batch_size = int(row["batch_size"])
    workers = int(row["workers"])
    return {
        "profile_name": mode,
        "device": "RTX 5070 Ti single-GPU",
        "runtime": {
            "python_env": "/mnt/d/conda_envs/sparseworld_cu128/bin/python",
            "recommend_wsl_ext4_repo_copy": True,
            "tf32": True,
            "cudnn_benchmark": True,
        },
        "train": {
            "samples_per_gpu": batch_size,
            "workers_per_gpu": workers,
            "pin_memory": True,
            "persistent_workers": workers > 0,
            "prefetch_factor": 4 if workers > 0 else None,
            "amp_fp16": True,
            "eval_during_train": False,
            "checkpoint_interval_iters": 1000,
            "log_interval_iters": 20,
            "disable_per_iter_cuda_synchronize": True,
            "disable_per_iter_peak_memory_reset": True,
        },
        "expected_benchmark_summary": {
            "throughput_samples_per_sec": row.get("throughput_samples_per_sec"),
            "gpu_util_mean": row.get("gpu_util_mean"),
            "peak_mem_mb": row.get("peak_mem_mb"),
            "data_wait_ratio": row.get("data_wait_ratio"),
        },
    }


def plot_summary(summary_rows: list[dict[str, Any]]) -> None:
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    if not ok_rows:
        return
    labels = [r["candidate"] for r in ok_rows]
    throughput = [float(r["throughput_samples_per_sec"]) for r in ok_rows]
    util = [float(r.get("gpu_util_mean", 0.0) or 0.0) for r in ok_rows]
    wait = [float(r.get("data_wait_ratio", 0.0) or 0.0) * 100.0 for r in ok_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(labels))
    axes[0].bar(x, throughput, color="#2b6cb0")
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("samples / sec")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")

    axes[1].bar(x, util, color="#2f855a")
    axes[1].set_title("Mean GPU util")
    axes[1].set_ylabel("%")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    axes[2].bar(x, wait, color="#c05621")
    axes[2].set_title("Data wait ratio")
    axes[2].set_ylabel("%")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "5070ti_throughput_sweep_summary.png", dpi=180)
    plt.close(fig)


def make_report(summary_rows: list[dict[str, Any]], best_max: dict[str, Any] | None, best_stable: dict[str, Any] | None) -> str:
    lines = [
        "# 5070 Ti Throughput Baseline",
        "",
        "- Goal: establish a high-throughput single-GPU training baseline for SparseWorld SW-8+.",
        "- Scope: train-only throughput sweep, not model-quality comparison.",
        "- Runtime assumption: WSL2 + `sparseworld_cu128` + RTX 5070 Ti.",
        "",
        "## Key findings",
        "",
        f"- TF32 enabled during benchmark: True",
        f"- cudnn benchmark enabled during benchmark: True",
    ]
    if best_max:
        lines += [
            f"- Max-throughput candidate: `{best_max['candidate']}`",
            f"  - throughput={float(best_max['throughput_samples_per_sec']):.3f} samples/s",
            f"  - gpu_util_mean={float(best_max.get('gpu_util_mean', 0.0) or 0.0):.2f}%",
            f"  - peak_mem_mb={float(best_max['peak_mem_mb']):.1f}",
            f"  - data_wait_ratio={float(best_max['data_wait_ratio']):.3f}",
        ]
    if best_stable:
        lines += [
            f"- Stable experiment candidate: `{best_stable['candidate']}`",
            f"  - throughput={float(best_stable['throughput_samples_per_sec']):.3f} samples/s",
            f"  - gpu_util_mean={float(best_stable.get('gpu_util_mean', 0.0) or 0.0):.2f}%",
            f"  - peak_mem_mb={float(best_stable['peak_mem_mb']):.1f}",
            f"  - data_wait_ratio={float(best_stable['data_wait_ratio']):.3f}",
        ]
    lines += [
        "",
        "## Candidate table",
        "",
    ]
    for row in summary_rows:
        if row.get("status") == "ok":
            lines.append(
                f"- {row['candidate']}: throughput={float(row['throughput_samples_per_sec']):.3f} samples/s, "
                f"gpu_util_mean={float(row.get('gpu_util_mean', 0.0) or 0.0):.2f}%, "
                f"peak_mem_mb={float(row['peak_mem_mb']):.1f}, "
                f"data_wait_ratio={float(row['data_wait_ratio']):.3f}"
            )
        else:
            lines.append(f"- {row['candidate']}: status={row['status']}, error={row.get('error')}")
    lines += [
        "",
        "## Standard baseline rules",
        "",
        "- Put active repo/cache/data on WSL ext4 when possible; avoid `/mnt/d` for heavy train IO.",
        "- Keep evaluation outside the hot training loop.",
        "- Do not keep per-iter `torch.cuda.synchronize()` or per-iter peak-memory reset in real training.",
        "- Prefer continuous training windows (50-200 iter blocks) before evaluation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_dirs()
    candidates = parse_candidate_grid(args.candidate_grid)
    status = StatusLogger()

    manifest = {
        "candidate_grid": [{"batch_size": b, "workers": w} for b, w in candidates],
        "iters_per_candidate": args.iters_per_candidate,
        "warmup_iters": args.warmup_iters,
        "tf32": args.enable_tf32,
        "cudnn_benchmark": args.enable_cudnn_benchmark,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "project_root": str(PROJECT_ROOT),
    }
    write_json(REPORTS_DIR / "5070ti_throughput_manifest.json", manifest)

    summary_rows: list[dict[str, Any]] = []
    iter_rows_all: list[dict[str, Any]] = []
    gpu_rows_all: list[dict[str, Any]] = []
    for batch_size, workers in candidates:
        summary, iter_rows, gpu_rows = benchmark_candidate(batch_size, workers, args, status)
        summary_rows.append(summary)
        iter_rows_all.extend(iter_rows)
        gpu_rows_all.extend(gpu_rows)
        write_csv(REPORTS_DIR / "5070ti_throughput_sweep.csv", summary_rows)
        write_csv(ARTIFACTS_DIR / "5070ti_throughput_iter_metrics.csv", iter_rows_all)
        write_csv(ARTIFACTS_DIR / "5070ti_throughput_gpu_samples.csv", gpu_rows_all)

    best_max, best_stable = choose_recommendations(summary_rows)
    if best_max:
        write_json(REPORTS_DIR / "5070ti_max_throughput_baseline_config.json", config_payload_from_row(best_max, "5070ti_max_throughput_baseline"))
    if best_stable:
        write_json(REPORTS_DIR / "5070ti_stable_experiment_baseline_config.json", config_payload_from_row(best_stable, "5070ti_stable_experiment_baseline"))
    write_md(REPORTS_DIR / "5070ti_throughput_recommendation.md", make_report(summary_rows, best_max, best_stable))
    plot_summary(summary_rows)


if __name__ == "__main__":
    main()
