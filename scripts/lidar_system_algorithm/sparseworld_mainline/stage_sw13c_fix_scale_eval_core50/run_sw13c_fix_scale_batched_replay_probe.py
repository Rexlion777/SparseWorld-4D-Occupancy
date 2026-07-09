from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from mmcv.parallel import collate as collate_fn


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50"
SW13C_FIX_SCALE_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/run_sw13c_fix_scale_main.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scale = load_module("sw13c_fix_scale_probe_base", SW13C_FIX_SCALE_SCRIPT)
sw13a = scale.sw13c_fix.sw13a
sw2 = scale.sw13c_fix.sw2
sw81 = scale.sw81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether SparseWorld replay can be made batched inside the scale stage folder only.")
    parser.add_argument("--samples", default="0,1")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    sample_ids = parse_int_list(args.samples)
    cfg, dataset, model, _ = sw81.build_sparseworld_runtime(
        train=False,
        cfg_overrides={"data.samples_per_gpu": len(sample_ids), "data.workers_per_gpu": 0},
    )
    del cfg
    samples = [dataset[idx] for idx in sample_ids]
    batch = collate_fn(samples, samples_per_gpu=len(samples))
    raw_unwrapped = [sw2.unwrap(sample) for sample in samples]
    moved = sw2.move_to_cuda(batch)
    holder: dict[str, Any] = {}
    original_forward = sw81.attach_query_capture(model, holder)
    original_simple_test_online = model.simple_test_online
    result: dict[str, Any] = {
        "sample_ids": sample_ids,
        "samples_per_gpu": len(sample_ids),
        "batch_img_metas_type": str(type(moved["img_metas"])),
        "batch_img_type": str(type(moved["img"])),
        "batch_probe_passed": False,
        "failure": None,
    }
    try:
        sw13a.reset_model_cache(model)
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **moved)
        result["batch_probe_passed"] = True
    except Exception as exc:
        result["failure"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        model.simple_test_online = original_simple_test_online
        model.forward_backbone = original_forward  # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = REPORTS_DIR / "sw13c_fix_scale_batched_replay_probe.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
