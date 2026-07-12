from __future__ import annotations

import math
import sys
from pathlib import Path
import types
import importlib.util
import ast
import textwrap

import pytest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT / "external/SparseWorld"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "IPython" not in sys.modules:
    stub = types.ModuleType("IPython")
    stub.embed = lambda *args, **kwargs: None
    stub.get_ipython = lambda: None
    sys.modules["IPython"] = stub
if "torch_scatter" not in sys.modules:
    scatter_stub = types.ModuleType("torch_scatter")
    scatter_stub.scatter_max = lambda src, index, dim=0: (src, index)
    sys.modules["torch_scatter"] = scatter_stub
if "mmdet3d.models.sparsedetectors.bbox.utils" not in sys.modules:
    mmdet3d_stub = types.ModuleType("mmdet3d")
    models_stub = types.ModuleType("mmdet3d.models")
    sparse_stub = types.ModuleType("mmdet3d.models.sparsedetectors")
    bbox_stub = types.ModuleType("mmdet3d.models.sparsedetectors.bbox")
    utils_stub = types.ModuleType("mmdet3d.models.sparsedetectors.bbox.utils")

    def _pc_range_tensor(pc_range, device, dtype):
        return pc_range if isinstance(pc_range, torch.Tensor) else torch.tensor(pc_range, device=device, dtype=dtype)

    def _decode_points(points, pc_range):
        pc = _pc_range_tensor(pc_range, points.device, points.dtype)
        low, high = pc[:3], pc[3:]
        return low + points * (high - low)

    def _encode_points(points, pc_range):
        pc = _pc_range_tensor(pc_range, points.device, points.dtype)
        low, high = pc[:3], pc[3:]
        return (points - low) / (high - low)

    utils_stub.decode_points = _decode_points
    utils_stub.encode_points = _encode_points
    sys.modules["mmdet3d"] = mmdet3d_stub
    sys.modules["mmdet3d.models"] = models_stub
    sys.modules["mmdet3d.models.sparsedetectors"] = sparse_stub
    sys.modules["mmdet3d.models.sparsedetectors.bbox"] = bbox_stub
    sys.modules["mmdet3d.models.sparsedetectors.bbox.utils"] = utils_stub
spec = importlib.util.spec_from_file_location(
    "mcqm_local",
    ROOT / "mmdet3d/models/sparsedetectors/mcqm.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("failed to load mcqm.py")
mcqm_local = importlib.util.module_from_spec(spec)
sys.modules["mcqm_local"] = mcqm_local
spec.loader.exec_module(mcqm_local)

runner_spec = importlib.util.spec_from_file_location(
    "mcqm_runner_local",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_mcqm_motion_compensated_query_memory/run_mcqm_main.py",
)
if runner_spec is None or runner_spec.loader is None:
    raise RuntimeError("failed to load run_mcqm_main.py")
mcqm_runner_local = importlib.util.module_from_spec(runner_spec)
sys.modules["mcqm_runner_local"] = mcqm_runner_local
runner_spec.loader.exec_module(mcqm_runner_local)

from mcqm_local import (  # type: ignore[attr-defined]
    MCQMMemoryProjector,
    MCQMQ2FLevelDecoder,
    MCQMQ2FQueryProjector,
    MCQMV2Projector,
    aggregate_query_semantics,
    bilinear_splat_features,
    blend_failed_camera_fpn_levels,
    build_occ2img_from_img_metas,
    build_memory_cache_keep_mask,
    build_layer2_passthrough_state,
    compose_mcqm_v2_feature_update,
    compose_mcqm_v2_point_update,
    compose_mcqm_object_motion_offset,
    compute_mcqm_v2_query_utility,
    SceneQueryMemoryManager,
    compose_t_dst_from_src,
    compress_48_to_4,
    deterministic_fps_indices,
    encode_points_normalized,
    infer_camera_name_from_filename,
    normalized_feature_distance,
    infer_temporal_camera_layout,
    project_occ_points_to_image,
    replace_failed_camera_fpn_levels,
    resolve_mcqm_full_time_contract,
    infer_scene_token,
    infer_timestamp,
    resolve_mcqm_v2_alpha,
    stack_selected_memory_candidates,
    select_valid_memory_candidates_for_nms,
    select_dynamic_query_replacements,
    transform_points_between_egos,
)

apply_perturbation_with_manifest_to_batch = mcqm_runner_local.apply_perturbation_with_manifest_to_batch
batch_img_tensor = mcqm_runner_local.batch_img_tensor
diff_meta_subset = mcqm_runner_local.diff_meta_subset
expected_new_v2_key_names = mcqm_runner_local.expected_new_v2_key_names
mcqm_v2_variant_cfg = mcqm_runner_local.mcqm_v2_variant_cfg
build_target_metric_map = mcqm_runner_local.build_target_metric_map
apply_mcqm_q2f_parameter_freeze = mcqm_runner_local.apply_mcqm_q2f_parameter_freeze
build_mcqm_v2_architecture_signature = mcqm_runner_local.build_mcqm_v2_architecture_signature
mcqm_q2f_v1_cfg = mcqm_runner_local.mcqm_q2f_v1_cfg
set_mcqm_q2f_training_mode = mcqm_runner_local.set_mcqm_q2f_training_mode
_q2f_total_loss = mcqm_runner_local._q2f_total_loss
checkpoint_state_audit_full = mcqm_runner_local.checkpoint_state_audit_full
clone_mcqm_memory_seed_dict = mcqm_runner_local.clone_mcqm_memory_seed_dict
has_valid_previous_frame = mcqm_runner_local.has_valid_previous_frame
is_valid_sha256 = mcqm_runner_local.is_valid_sha256
memory_seed_audit = mcqm_runner_local.memory_seed_audit
object_sha256 = mcqm_runner_local.object_sha256
dense_occ_top1_conf_margin = mcqm_runner_local.dense_occ_top1_conf_margin
build_active_mask_from_occ_debug = mcqm_runner_local.build_active_mask_from_occ_debug
build_s0_s1_s2_s3_masks = mcqm_runner_local.build_s0_s1_s2_s3_masks
compare_unselected_slots = mcqm_runner_local.compare_unselected_slots
build_d3_mask_from_occ = mcqm_runner_local.build_d3_mask_from_occ
build_q2f_eval_parity_record = mcqm_runner_local.build_q2f_eval_parity_record


def make_transform(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0, yaw_deg: float = 0.0) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    out = torch.eye(4, dtype=torch.float32)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    out[0, 3] = tx
    out[1, 3] = ty
    out[2, 3] = tz
    return out


def test_deterministic_fps_indices_is_stable():
    points = torch.tensor([[float(i), 0.0, 0.0] for i in range(48)], dtype=torch.float32)
    idx0 = deterministic_fps_indices(points, 4)
    idx1 = deterministic_fps_indices(points, 4)
    assert torch.equal(idx0, idx1)
    assert idx0.shape[0] == 4


def test_compress_48_to_4_keeps_real_points():
    points = torch.randn(48, 3)
    anchors, geom = compress_48_to_4(points)
    assert anchors.shape == (4, 3)
    assert geom["four_anchor_points"].shape == (4, 3)
    for anchor in anchors:
        assert bool((points == anchor).all(dim=-1).any().item())


def test_pose_transform_identity_and_inverse():
    points = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=torch.float32)
    ident = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    same = transform_points_between_egos(points, ident, ident)
    assert torch.allclose(points, same)
    src = make_transform(tx=1.0, ty=2.0, yaw_deg=15.0).unsqueeze(0)
    dst = make_transform(tx=-0.5, ty=0.25, yaw_deg=-10.0).unsqueeze(0)
    moved = transform_points_between_egos(points, src, dst)
    recovered = transform_points_between_egos(moved, dst, src)
    assert torch.allclose(points, recovered, atol=1e-5)


def test_a0_clean_perturbation_is_identity():
    img = torch.arange(1 * 6 * 3 * 4 * 5, dtype=torch.uint8).reshape(1, 6, 3, 4, 5)
    meta = {
        "filename": [
            "/tmp/CAM_FRONT/x.jpg",
            "/tmp/CAM_FRONT_RIGHT/x.jpg",
            "/tmp/CAM_FRONT_LEFT/x.jpg",
            "/tmp/CAM_BACK/x.jpg",
            "/tmp/CAM_BACK_LEFT/x.jpg",
            "/tmp/CAM_BACK_RIGHT/x.jpg",
        ],
        "lidar2img": [torch.eye(4, dtype=torch.float32).numpy() for _ in range(6)],
    }
    batch = {
        "img": [types.SimpleNamespace(data=[img.clone()])],
        "img_metas": [types.SimpleNamespace(data=[[meta]])],
    }
    original = batch_img_tensor(batch).clone()
    original_meta_hash = object_sha256(batch["img_metas"][0].data[0])
    clean_batch, _ = apply_perturbation_with_manifest_to_batch(batch, "A0_clean")
    assert torch.equal(original, batch_img_tensor(clean_batch))
    assert object_sha256(clean_batch["img_metas"][0].data[0]) == original_meta_hash


def test_normalized_roundtrip():
    pc_range = [-80.0, -60.0, -3.0, 80.0, 60.0, 6.6]
    points = torch.tensor([[[[0.0, 0.0, 0.0], [10.0, -5.0, 2.0]]]], dtype=torch.float32)
    norm = encode_points_normalized(points, pc_range)
    back = transform_points_between_egos(points, torch.eye(4).unsqueeze(0), torch.eye(4).unsqueeze(0))
    assert torch.allclose(points, back)
    assert torch.isfinite(norm).all()


def test_scene_memory_manager_resets_on_scene_or_time_gap():
    mgr = SceneQueryMemoryManager(continuity_tol_sec=0.6)
    state = type("State", (), {"scene_token": "sceneA", "timestamp": torch.tensor(10.0)})()
    mgr.update(0, state)
    assert mgr.get(0, "sceneA", 10.4) is state
    assert mgr.get(0, "sceneB", 10.4) is None
    mgr.update(0, state)
    assert mgr.get(0, "sceneA", 11.0) is None


def test_scene_and_timestamp_inference():
    meta = {"occ_gt_path": "data/nuscenes/gts/scene-0001/token/labels.npz", "img_timestamp": [1.0, 0.9]}
    assert infer_scene_token(meta) == "scene-0001"
    assert infer_timestamp(meta) == 1.0


def _make_selector_tensors():
    native_feat = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    native_pts = torch.tensor([
        [[0.0, 0.0, 0.0]] * 4,
        [[0.5, 0.0, 0.0]] * 4,
        [[1.0, 0.0, 0.0]] * 4,
        [[1.5, 0.0, 0.0]] * 4,
    ], dtype=torch.float32)
    native_logits = torch.randn(4, 4, 17)
    native_quality = torch.tensor([0.9, 0.7, 0.4, 0.2], dtype=torch.float32)
    mem_feat = torch.arange(3 * 3, dtype=torch.float32).reshape(3, 3) + 100.0
    mem_pts = torch.tensor([
        [[1.4, 0.0, 0.0]] * 4,
        [[0.6, 0.0, 0.0]] * 4,
        [[10.0, 0.0, 0.0]] * 4,
    ], dtype=torch.float32)
    mem_logits = torch.randn(3, 4, 17)
    mem_quality = torch.tensor([0.95, 0.5, 0.1], dtype=torch.float32)
    mem_valid = torch.tensor([True, True, True])
    mem_ids = torch.tensor([11, 12, 13], dtype=torch.long)
    return native_feat, native_pts, native_logits, native_quality, mem_feat, mem_pts, mem_logits, mem_quality, mem_valid, mem_ids


def _load_sparseworld_hook_method():
    source_path = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method_src = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SparseWorld4DTraj":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_mcqm_inject_after_layer1":
                    method_src = ast.get_source_segment(source_path.read_text(encoding="utf-8"), item)
                    break
    if method_src is None:
        raise RuntimeError("failed to extract _mcqm_inject_after_layer1")
    ns = {
        "torch": torch,
        "MCQM_V2_EXPECTED_CALLBACK_LAYER_IDX": 1,
        "decode_points_metric": mcqm_local.decode_points_metric,
        "build_layer2_passthrough_state": build_layer2_passthrough_state,
        "select_dynamic_query_replacements": select_dynamic_query_replacements,
        "aggregate_query_semantics": aggregate_query_semantics,
    }
    exec(textwrap.dedent(method_src), ns)
    return ns["_mcqm_inject_after_layer1"]


def _load_sparseworld_extract_memory_state_method():
    source_path = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    method_src = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SparseWorld4DTraj":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_mcqm_extract_memory_state":
                    method_src = ast.get_source_segment(source_text, item)
                    break
    if method_src is None:
        raise RuntimeError("failed to extract _mcqm_extract_memory_state")
    ns = {
        "torch": torch,
        "aggregate_query_semantics": aggregate_query_semantics,
        "build_memory_cache_keep_mask": build_memory_cache_keep_mask,
        "MemoryFrameState": mcqm_local.MemoryFrameState,
        "np": __import__("numpy"),
    }
    exec(textwrap.dedent(method_src), ns)
    return ns["_mcqm_extract_memory_state"]


def _load_sparseworld_method(method_name: str):
    source_path = PROJECT_ROOT / "external/SparseWorld/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    method_src = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SparseWorld4DTraj":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    method_src = ast.get_source_segment(source_text, item)
                    break
    if method_src is None:
        raise RuntimeError(f"failed to extract {method_name}")
    ns = {
        "torch": torch,
        "json": __import__("json"),
        "hashlib": __import__("hashlib"),
        "SELECTION_VECTOR_FIELDS": (
            "native_replaced_indices",
            "memory_selected_indices",
            "selected_memory_query_ids",
            "selected_memory_classes",
            "selected_memory_quality_values",
            "replaced_native_quality_values",
            "quality_gain_values",
        ),
        "compose_mcqm_v2_feature_update": compose_mcqm_v2_feature_update,
        "compose_mcqm_v2_point_update": compose_mcqm_v2_point_update,
        "resolve_mcqm_v2_alpha": resolve_mcqm_v2_alpha,
        "decode_points_metric": mcqm_local.decode_points_metric,
        "compute_mcqm_v2_query_utility": compute_mcqm_v2_query_utility,
    }
    exec(textwrap.dedent(method_src), ns)
    return ns[method_name]


def test_dynamic_replacement_no_memory_keeps_native():
    native_feat, native_pts, native_logits, native_quality, *_ = _make_selector_tensors()
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        None, None, None, None, None,
        max_replacement_budget=180,
        replacement_margin=0.05,
    )
    assert out["actual_replacement_count"] == 0
    assert torch.equal(out["fused_query_feat"], native_feat)
    assert torch.equal(out["fused_query_points"], native_pts)
    assert torch.equal(out["fused_cls_logits"], native_logits)


def test_dynamic_replacement_low_memory_keeps_native():
    native_feat, native_pts, native_logits, native_quality, mem_feat, mem_pts, mem_logits, mem_quality, mem_valid, mem_ids = _make_selector_tensors()
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, torch.zeros_like(mem_quality), mem_valid,
        max_replacement_budget=180,
        replacement_margin=0.05,
        memory_query_ids=mem_ids,
        native_query_classes=torch.tensor([1, 1, 1, 1]),
        memory_query_classes=torch.tensor([7, 7, 7]),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
    )
    assert out["actual_replacement_count"] == 0
    assert torch.equal(out["fused_cls_logits"], native_logits)
    assert int(out["query_source"].sum().item()) == 0


def test_dynamic_replacement_single_winner():
    native_feat, native_pts, native_logits, native_quality, mem_feat, mem_pts, mem_logits, mem_quality, mem_valid, mem_ids = _make_selector_tensors()
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=1,
        replacement_margin=0.2,
        memory_query_ids=mem_ids,
        native_query_classes=torch.tensor([1, 1, 1, 1]),
        memory_query_classes=torch.tensor([1, 1, 1]),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
        native_bottomk_ratio=1.0,
    )
    assert out["actual_replacement_count"] == 1
    replaced_idx = int(out["native_replaced_indices"][0].item())
    assert replaced_idx == 3
    assert torch.equal(out["fused_cls_logits"][replaced_idx], mem_logits[0])
    assert int(out["query_source"][replaced_idx].item()) == 1
    assert int(out["source_memory_query_id"][replaced_idx].item()) == 11


def test_dynamic_replacement_budget_cap_and_query_count():
    native_feat = torch.randn(300, 3)
    native_pts = torch.randn(300, 4, 3)
    native_logits = torch.randn(300, 4, 17)
    native_quality = torch.linspace(0.0, 0.5, 300)
    mem_feat = torch.randn(300, 3)
    mem_pts = torch.randn(300, 4, 3)
    mem_logits = torch.randn(300, 4, 17)
    mem_quality = torch.linspace(1.0, 0.6, 300)
    mem_valid = torch.ones(300, dtype=torch.bool)
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=180,
        replacement_margin=0.0,
        native_query_classes=torch.zeros(300, dtype=torch.long),
        memory_query_classes=torch.zeros(300, dtype=torch.long),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
        local_match_radius_m=1e6,
        native_bottomk_ratio=1.0,
    )
    assert out["actual_replacement_count"] == 180
    assert out["fused_query_feat"].shape[0] == 300


def test_dynamic_replacement_preserves_full_logits_not_one_hot():
    native_feat, native_pts, native_logits, native_quality, mem_feat, mem_pts, _, mem_quality, mem_valid, mem_ids = _make_selector_tensors()
    mem_logits = torch.randn(3, 4, 17)
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=180,
        replacement_margin=0.2,
        memory_query_ids=mem_ids,
        native_query_classes=torch.tensor([1, 1, 1, 1]),
        memory_query_classes=torch.tensor([1, 1, 1]),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
        native_bottomk_ratio=1.0,
    )
    replaced_idx = int(out["native_replaced_indices"][0].item())
    assert torch.equal(out["fused_cls_logits"][replaced_idx], mem_logits[0])


def test_dynamic_replacement_provenance_and_margin_hold():
    native_feat, native_pts, native_logits, native_quality, mem_feat, mem_pts, mem_logits, mem_quality, mem_valid, mem_ids = _make_selector_tensors()
    margin = 0.05
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=180,
        replacement_margin=margin,
        memory_query_ids=mem_ids,
        native_query_classes=torch.tensor([1, 1, 1, 1]),
        memory_query_classes=torch.tensor([1, 1, 1]),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
        native_bottomk_ratio=1.0,
    )
    assert out["actual_replacement_count"] >= 1
    assert torch.all(out["quality_gain_values"] > margin)
    replaced_idx = out["native_replaced_indices"]
    assert torch.all(out["query_source"][replaced_idx] == 1)


def test_memory_projector_is_identity_at_initialization():
    projector = MCQMMemoryProjector(embed_dims=8, num_classes=17, mode="identity_safe")
    forecast_query_feat = torch.randn(5, 8)
    geom = {
        "point_mean": torch.randn(5, 3),
        "point_std": torch.randn(5, 3),
        "point_min": torch.randn(5, 3),
        "point_max": torch.randn(5, 3),
        "four_anchor_points": torch.randn(5, 4, 3),
    }
    out = projector(
        forecast_query_feat,
        geom,
        dominant_class=torch.zeros(5, dtype=torch.long),
        confidence=torch.ones(5, 1),
        ego_motion_vec=torch.zeros(5, 6),
        residual_stats=torch.zeros(5, 6),
        source_index=1,
        age_index=1,
    )
    assert torch.allclose(out, forecast_query_feat, atol=1e-7, rtol=0)


def test_mcqm_v2_projector_preserves_memory_identity_at_initialization():
    projector = MCQMV2Projector(embed_dims=8)
    memory_feat = torch.randn(5, 8)
    native_feat = torch.randn(5, 8)
    out = projector(memory_feat, native_feat)
    assert torch.allclose(out, memory_feat, atol=1e-7, rtol=0)


def test_mcqm_v2_residual_alpha_zero_resolves_to_zero_without_gate():
    reference = torch.randn(2, 3)
    alpha = resolve_mcqm_v2_alpha(None, 0.0, reference)
    assert torch.equal(alpha, reference.new_zeros(()))


def test_mcqm_v2_feature_residual_gate_zero_is_passthrough():
    native_feat = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    adapted_memory_feat = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    out = compose_mcqm_v2_feature_update(
        native_feat,
        adapted_memory_feat,
        injection_mode="feature_residual",
        alpha=native_feat.new_zeros(()),
        reliability=1.0,
    )
    assert torch.equal(out, native_feat)


def test_residual_gate_receives_gradient():
    gate = torch.nn.Parameter(torch.tensor(0.0))
    native_feat = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    adapted_memory_feat = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    alpha = resolve_mcqm_v2_alpha(gate, None, native_feat)
    out = compose_mcqm_v2_feature_update(
        native_feat,
        adapted_memory_feat,
        injection_mode="feature_residual",
        alpha=alpha,
        reliability=1.0,
    )
    out.sum().backward()
    assert gate.grad is not None


def test_mcqm_v2_feature_only_replace_changes_only_feature():
    native_feat = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    adapted_memory_feat = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    native_points = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)
    memory_points = torch.tensor([[[0.9, 0.8, 0.7]]], dtype=torch.float32)
    out_feat = compose_mcqm_v2_feature_update(
        native_feat,
        adapted_memory_feat,
        injection_mode="feature_only_replace",
        alpha=0.0,
        reliability=1.0,
    )
    out_points = compose_mcqm_v2_point_update(
        native_points,
        memory_points,
        injection_mode="feature_only_replace",
    )
    assert torch.equal(out_feat, adapted_memory_feat)
    assert torch.equal(out_points, native_points)


def test_mcqm_v2_geometry_only_changes_only_points():
    native_feat = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    adapted_memory_feat = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    native_points = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)
    memory_points = torch.tensor([[[0.9, 0.8, 0.7]]], dtype=torch.float32)
    out_feat = compose_mcqm_v2_feature_update(
        native_feat,
        adapted_memory_feat,
        injection_mode="geometry_only",
        alpha=0.25,
        reliability=1.0,
    )
    out_points = compose_mcqm_v2_point_update(
        native_points,
        memory_points,
        injection_mode="geometry_only",
    )
    assert torch.equal(out_feat, native_feat)
    assert torch.equal(out_points, memory_points)


def test_mcqm_v2_utility_formula_sign():
    native_feat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    teacher_feat = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    better_feat = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    worse_feat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    native_points_metric = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    teacher_points_metric = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    better_points_metric = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    worse_points_metric = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    pos = compute_mcqm_v2_query_utility(
        native_feat,
        native_points_metric,
        better_feat,
        better_points_metric,
        teacher_feat,
        teacher_points_metric,
        lambda_point=1.0,
    )
    neg = compute_mcqm_v2_query_utility(
        native_feat,
        native_points_metric,
        worse_feat,
        worse_points_metric,
        teacher_feat,
        teacher_points_metric,
        lambda_point=1.0,
    )
    assert float(pos["utility_query"].item()) > 0.0
    assert float(neg["utility_query"].item()) <= 0.0


def test_diff_meta_subset_marks_container_only_difference_as_benign():
    left = {"ego2global": [np.eye(4, dtype=np.float32)], "timestamp": 1.0}
    right = {"ego2global": (torch.eye(4, dtype=torch.float32),), "timestamp": 1.0}
    diff = diff_meta_subset(left, right)
    assert diff["protocol_mismatch"] is False
    assert any(row["benign_representation_difference"] for row in diff["field_diffs"])


def test_mcqm_v2_variant_cfg_h2_alpha():
    cfg = mcqm_v2_variant_cfg("H2_A25")
    assert cfg["mcqm_v2_injection_mode"] == "feature_residual"
    assert abs(float(cfg["mcqm_residual_alpha"]) - 0.25) < 1e-8


def test_non_oracle_rejects_teacher_state():
    method = _load_sparseworld_method("_mcqm_v2_oracle_flags")

    class Dummy:
        mcqm_cfg = {"mcqm_v2_oracle_mode": "disabled"}
        mcqm_v2_runtime_oracle = {"teacher_current_query_feat": torch.randn(1, 2)}
        training = False

    with pytest.raises(RuntimeError):
        method(Dummy())


def test_non_oracle_rejects_forced_pairs():
    method = _load_sparseworld_method("_mcqm_v2_oracle_flags")

    class Dummy:
        mcqm_cfg = {"mcqm_v2_oracle_mode": "disabled"}
        mcqm_v2_runtime_oracle = {"forced_pairs": [{"native_slot_id": 1, "memory_query_id": 7}]}
        training = False

    with pytest.raises(RuntimeError):
        method(Dummy())


def test_build_target_metric_map_uses_sample_horizon_keys():
    result = {
        "rows": [
            {"sample_index": 40, "horizon_s": 0, "occupied_iou": 0.1},
            {"sample_index": 40, "horizon_s": 2, "occupied_iou": 0.2},
        ]
    }
    out = build_target_metric_map(result)
    assert out[(40, 0)]["occupied_iou"] == 0.1
    assert out[(40, 2)]["occupied_iou"] == 0.2


def test_checkpoint_state_audit_full_uses_normalized_keys_for_missing(tmp_path):
    model = torch.nn.Linear(2, 2)
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"state_dict": {f"module.{k}": v.detach().clone() for k, v in model.state_dict().items()}}, ckpt_path)
    audit = checkpoint_state_audit_full(model, ckpt_path)
    assert audit["missing_keys"] == []
    assert audit["unexpected_keys"] == []


def test_clone_memory_seed_dict_deep_clones_tensors():
    state = mcqm_local.MemoryFrameState(
        query_feat=torch.randn(1, 4),
        refine_points=torch.randn(1, 4, 3),
        cls_logits=torch.randn(1, 4, 17),
        dominant_class=torch.tensor([4], dtype=torch.long),
        confidence=torch.tensor([0.8], dtype=torch.float32),
        query_ids=torch.tensor([25], dtype=torch.long),
        query_source=torch.tensor([0], dtype=torch.long),
        memory_age=torch.tensor([0], dtype=torch.long),
        was_memory_injected=torch.tensor([False]),
        parent_memory_query_id=torch.tensor([-1], dtype=torch.long),
        source_native_query_id=torch.tensor([25], dtype=torch.long),
        timestamp=torch.tensor(1.0, dtype=torch.float64),
        scene_token="scene",
        ego2global=torch.eye(4),
    )
    seed = {0: state}
    cloned = clone_mcqm_memory_seed_dict(seed)
    assert cloned[0].query_feat.data_ptr() != seed[0].query_feat.data_ptr()
    audit = memory_seed_audit(cloned)
    assert audit["seed_query_count"] == 1


def test_memory_seed_hash_changes_when_tensor_content_changes():
    state = mcqm_local.MemoryFrameState(
        query_feat=torch.zeros(1, 4),
        refine_points=torch.zeros(1, 4, 3),
        cls_logits=torch.zeros(1, 4, 17),
        dominant_class=torch.tensor([4], dtype=torch.long),
        confidence=torch.tensor([0.8], dtype=torch.float32),
        query_ids=torch.tensor([25], dtype=torch.long),
        query_source=torch.tensor([0], dtype=torch.long),
        memory_age=torch.tensor([0], dtype=torch.long),
        was_memory_injected=torch.tensor([False]),
        parent_memory_query_id=torch.tensor([-1], dtype=torch.long),
        source_native_query_id=torch.tensor([25], dtype=torch.long),
        timestamp=torch.tensor(1.0, dtype=torch.float64),
        scene_token="scene",
        ego2global=torch.eye(4),
    )
    seed_a = {0: state}
    seed_b = clone_mcqm_memory_seed_dict(seed_a)
    seed_b[0].query_feat[0, 0] = 1.0
    hash_a = memory_seed_audit(seed_a)["seed_memory_state_sha256"]
    hash_b = memory_seed_audit(seed_b)["seed_memory_state_sha256"]
    assert hash_a != hash_b


def test_has_valid_previous_frame_requires_same_scene():
    dataset = types.SimpleNamespace(
        data_infos=[
            {"scene_token": "scene-a"},
            {"scene_token": "scene-a"},
            {"scene_token": "scene-b"},
        ]
    )
    assert has_valid_previous_frame(dataset, 0) is False
    assert has_valid_previous_frame(dataset, 1) is True
    assert has_valid_previous_frame(dataset, 2) is False


def test_is_valid_sha256_rejects_empty_string():
    assert is_valid_sha256("") is False
    assert is_valid_sha256("0" * 64) is True


def test_architecture_signature_splits_structural_and_runtime():
    class QueryEmbedding:
        weight = torch.zeros(1040, 32)

    class Decoder:
        decoder_layers = [object(), object(), object(), object(), object(), object()]

    class Transformer:
        decoder = Decoder()

    class Head:
        transformer = Transformer()
        ind_stamps_all = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        query_embedding = QueryEmbedding()

    class Dummy:
        pts_bbox_head = Head()
        out_dim = 32
        num_refines = 4
        mcqm_num_classes = 17

        def named_parameters(self):
            yield "mcqm_v2_projector.out.weight", torch.nn.Parameter(torch.zeros(32, 32))
            yield "mcqm_v2_residual_gate", torch.nn.Parameter(torch.zeros(()))

    cfg = mcqm_v2_variant_cfg("H2_A10")
    sig = build_mcqm_v2_architecture_signature(Dummy(), cfg, "abc")
    assert "structural_signature" in sig
    assert "runtime_config_signature" in sig
    assert "mcqm_v2_injection_mode" not in sig["structural_signature"]
    assert sig["structural_signature"]["current_query_count"] == 2
    assert sig["runtime_config_signature"]["mcqm_v2_injection_mode"] == "feature_residual"


def test_mcqm_v2_apply_pairs_reports_zero_applied_for_passthrough_and_alpha_zero():
    method = _load_sparseworld_method("_mcqm_v2_apply_pairs")
    selection_sig_method = _load_sparseworld_method("_mcqm_v2_selection_signature")

    class Dummy:
        def __init__(self):
            self.mcqm_v2_projector = MCQMV2Projector(embed_dims=4)
            self.mcqm_v2_residual_gate = torch.nn.Parameter(torch.tensor(0.0))
            self.mcqm_cfg = {"mcqm_residual_alpha": 0.0}
            self.pts_bbox_head = type("Head", (), {"pc_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]})()

        def _mcqm_v2_selection_signature(self, result):
            return selection_sig_method(self, result)

    model = Dummy()
    native_feat = torch.zeros(1, 2, 4)
    native_points = torch.zeros(1, 2, 4, 3)
    native_cls = torch.zeros(1, 2, 4, 17)
    native_quality = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    memory_pack = {
        "feats": [torch.ones(1, 4)],
        "points": [torch.ones(1, 4, 3)],
        "query_ids": [torch.tensor([25], dtype=torch.long)],
        "dominant_class": [torch.tensor([4], dtype=torch.long)],
        "quality": [torch.tensor([0.9], dtype=torch.float32)],
    }
    result = {
        "native_replaced_indices": torch.tensor([1], dtype=torch.long),
        "memory_selected_indices": torch.tensor([0], dtype=torch.long),
        "selected_memory_query_ids": torch.tensor([25], dtype=torch.long),
        "selected_memory_classes": torch.tensor([4], dtype=torch.long),
        "quality_gain_values": torch.tensor([0.8], dtype=torch.float32),
        "selected_memory_quality_values": torch.tensor([0.9], dtype=torch.float32),
        "replaced_native_quality_values": torch.tensor([0.1], dtype=torch.float32),
        "match_distance_m": torch.tensor([0.2], dtype=torch.float32),
    }
    oracle_flags = {"uses_clean_current_teacher": False, "uses_gt_for_selection": False, "deployable": True}
    out_p0 = method(model, native_feat, native_points, native_cls, native_quality, memory_pack, 0, result, "passthrough", oracle_flags)
    assert out_p0["actual_applied_count"] == 0
    out_h2 = method(model, native_feat, native_points, native_cls, native_quality, memory_pack, 0, result, "feature_residual", oracle_flags)
    assert out_h2["actual_applied_count"] == 0


def test_forced_selection_result_rebuilds_schema_without_legacy_aliases():
    method = _load_sparseworld_method("_mcqm_v2_build_forced_selection_result")

    class Dummy:
        pass

    result = {
        "native_replaced_indices": torch.tensor([0], dtype=torch.long),
        "memory_selected_indices": torch.tensor([0], dtype=torch.long),
        "selected_memory_query_ids": torch.tensor([7], dtype=torch.long),
        "selected_memory_classes": torch.tensor([2], dtype=torch.long),
        "selected_memory_quality_values": torch.tensor([0.5], dtype=torch.float32),
        "replaced_native_quality_values": torch.tensor([0.2], dtype=torch.float32),
        "quality_gain_values": torch.tensor([0.3], dtype=torch.float32),
        "selected_memory_quality": torch.tensor([0.5], dtype=torch.float32),
        "replaced_native_quality": torch.tensor([0.2], dtype=torch.float32),
        "quality_gain": torch.tensor([0.3], dtype=torch.float32),
    }
    memory_pack = {
        "query_ids": [torch.tensor([11, 12], dtype=torch.long)],
        "dominant_class": [torch.tensor([4, 5], dtype=torch.long)],
        "quality": [torch.tensor([0.9, 0.7], dtype=torch.float32)],
    }
    native_quality = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    out = method(
        Dummy(),
        result,
        memory_pack,
        native_quality,
        0,
        [{"native_slot_id": 1, "memory_query_id": 12}],
        2,
    )
    assert torch.equal(out["native_replaced_indices"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(out["memory_selected_indices"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(out["selected_memory_query_ids"], torch.tensor([12], dtype=torch.long))
    assert torch.equal(out["selected_memory_classes"], torch.tensor([5], dtype=torch.long))
    assert torch.allclose(out["selected_memory_quality_values"], torch.tensor([0.7], dtype=torch.float32))
    assert torch.allclose(out["replaced_native_quality_values"], torch.tensor([0.2], dtype=torch.float32))
    assert torch.allclose(out["quality_gain_values"], torch.tensor([0.5], dtype=torch.float32))
    assert "selected_memory_quality" not in out
    assert "replaced_native_quality" not in out
    assert "quality_gain" not in out


def test_oracle_pair_rows_returns_empty_when_memory_pack_is_empty():
    method = _load_sparseworld_method("_mcqm_v2_oracle_pair_rows")

    class Dummy:
        mcqm_cfg = {"mcqm_v2_injection_mode": "feature_residual", "mcqm_residual_alpha": 0.1}
        pts_bbox_head = type("Head", (), {"pc_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]})()
        mcqm_v2_projector = MCQMV2Projector(embed_dims=4)
        mcqm_v2_residual_gate = torch.nn.Parameter(torch.tensor(0.0))

    native_feat = torch.zeros(1, 2, 4)
    native_points = torch.zeros(1, 2, 4, 3)
    native_quality = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    native_sem = {"dominant_class": torch.tensor([[4, 4]], dtype=torch.long)}
    memory_pack = {
        "feats": [None],
        "points": [None],
        "dominant_class": [None],
    }
    oracle_flags = {
        "uses_clean_current_teacher": True,
        "runtime": {
            "teacher_current_query_feat": torch.zeros(1, 2, 4),
            "teacher_current_query_points": torch.zeros(1, 2, 4, 3),
        },
    }
    rows, selected = method(
        Dummy(),
        native_feat,
        native_points,
        native_quality,
        native_sem,
        memory_pack,
        0,
        oracle_flags,
    )
    assert rows == []
    assert selected == []


def test_injected_memory_cannot_be_repropagated():
    query_source = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    keep_mask = build_memory_cache_keep_mask(query_source, allow_memory_repropagation=False)
    assert torch.equal(keep_mask, torch.tensor([True, False, True, False]))


def test_local_matching_rejects_distance_mismatch():
    native_feat = torch.randn(2, 3)
    native_pts = torch.tensor([
        [[0.0, 0.0, 0.0]] * 4,
        [[10.0, 0.0, 0.0]] * 4,
    ], dtype=torch.float32)
    native_logits = torch.randn(2, 4, 17)
    native_quality = torch.tensor([0.1, 0.2], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.tensor([[[20.0, 0.0, 0.0]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.9], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        native_query_classes=torch.tensor([4, 4]),
        memory_query_classes=torch.tensor([4]),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
        local_match_radius_m=2.0,
    )
    assert out["actual_replacement_count"] == 0


def test_local_matching_rejects_semantic_mismatch():
    native_feat = torch.randn(1, 3)
    native_pts = torch.tensor([[[0.0, 0.0, 0.0]] * 4], dtype=torch.float32)
    native_logits = torch.randn(1, 4, 17)
    native_quality = torch.tensor([0.1], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.tensor([[[0.5, 0.0, 0.0]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.9], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        native_query_classes=torch.tensor([4]),
        memory_query_classes=torch.tensor([7]),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
        local_match_radius_m=2.0,
    )
    assert out["actual_replacement_count"] == 0


def test_local_matching_chooses_lowest_quality_native_within_radius():
    native_feat = torch.randn(3, 3)
    native_pts = torch.tensor([
        [[0.0, 0.0, 0.0]] * 4,
        [[0.5, 0.0, 0.0]] * 4,
        [[5.0, 0.0, 0.0]] * 4,
    ], dtype=torch.float32)
    native_logits = torch.randn(3, 4, 17)
    native_quality = torch.tensor([0.9, 0.1, 0.05], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.tensor([[[0.4, 0.0, 0.0]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.8], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        native_query_classes=torch.tensor([4, 4, 4]),
        memory_query_classes=torch.tensor([4]),
        native_query_centers_metric=native_pts.mean(dim=-2),
        memory_query_centers_metric=mem_pts.mean(dim=-2),
        local_match_radius_m=2.0,
        native_bottomk_ratio=1.0,
    )
    assert out["actual_replacement_count"] == 1
    assert int(out["native_replaced_indices"][0].item()) == 1


def test_aggregate_query_semantics_uses_query_level_mean():
    logits = torch.tensor([[[0.0, 1.0], [0.0, 3.0]]], dtype=torch.float32)
    sem = aggregate_query_semantics(logits)
    assert sem["query_logits"].shape == (1, 2)
    assert int(sem["dominant_class"][0].item()) == 1


def test_build_layer2_passthrough_state_keeps_inputs_unchanged():
    query_feat = torch.randn(1, 1040, 8)
    query_points = torch.randn(1, 1040, 4, 3)
    cls_score = torch.randn(1, 1040, 4, 17)
    feat_before = query_feat.clone()
    points_before = query_points.clone()
    cls_before = cls_score.clone()
    out = build_layer2_passthrough_state(query_feat, query_points, cls_score)
    assert torch.equal(out["query_feat"], feat_before)
    assert torch.equal(out["query_points"], points_before)
    assert torch.equal(out["cls_score"], cls_before)
    assert torch.equal(query_feat, feat_before)
    assert torch.equal(query_points, points_before)
    assert torch.equal(cls_score, cls_before)


def test_real_hook_no_memory_parity_and_no_input_mutation():
    hook = _load_sparseworld_hook_method()

    class Dummy:
        mcqm_enabled = True
        mcqm_version = "v1"
        mcqm_cfg = {"replacement_budget": 180, "replacement_margin": 0.05}
        num_query = 720
        pts_bbox_head = type("Head", (), {"pc_range": [-80.0, -60.0, -3.0, 80.0, 60.0, 6.6]})()

        def _mcqm_inject_after_layer1_legacy(self, *args, **kwargs):
            raise AssertionError("legacy path should not be used")

        def _mcqm_valid_sampling_ratio(self, sampling_stats, current_mask):
            return torch.ones((1, int(current_mask.sum().item())), dtype=torch.float32)

        def _mcqm_native_quality(self, cls_score, valid_ratio):
            return torch.ones((cls_score.shape[0], cls_score.shape[1]), dtype=cls_score.dtype)

        def _mcqm_build_memory_candidates(self, img_metas, ego_feat, current_native_centers):
            return None

    model = Dummy()
    current_mask = torch.zeros((1, 1040), dtype=torch.bool)
    current_mask[0, :720] = True
    query_feat = torch.randn(1, 1040, 8)
    query_points = torch.randn(1, 1040, 4, 3)
    cls_score = torch.randn(1, 1040, 4, 17)
    feat_before = query_feat.clone()
    points_before = query_points.clone()
    cls_before = cls_score.clone()
    out = hook(
        model,
        decoder_runtime={"current_mask": current_mask, "ego_feat": torch.randn(1, 1, 32)},
        layer_idx=1,
        query_points=query_points,
        query_feat=query_feat,
        cls_score=cls_score,
        sampling_stats={},
        img_metas=[{}],
    )
    assert torch.equal(out["query_feat"], feat_before)
    assert torch.equal(out["query_points"], points_before)
    assert torch.equal(out["cls_score"], cls_before)
    assert out["query_feat"].data_ptr() != query_feat.data_ptr()
    assert out["query_points"].data_ptr() != query_points.data_ptr()
    assert out["cls_score"].data_ptr() != cls_score.data_ptr()
    assert torch.equal(query_feat, feat_before)
    assert torch.equal(query_points, points_before)
    assert torch.equal(cls_score, cls_before)


def test_real_hook_no_memory_parity_with_singleton_group_dim():
    hook = _load_sparseworld_hook_method()

    class Dummy:
        mcqm_enabled = True
        mcqm_version = "v1"
        mcqm_cfg = {"replacement_budget": 180, "replacement_margin": 0.05}
        num_query = 720
        pts_bbox_head = type("Head", (), {"pc_range": [-80.0, -60.0, -3.0, 80.0, 60.0, 6.6]})()

        def _mcqm_inject_after_layer1_legacy(self, *args, **kwargs):
            raise AssertionError("legacy path should not be used")

        def _mcqm_valid_sampling_ratio(self, sampling_stats, current_mask):
            return torch.ones((1, int(current_mask.sum().item())), dtype=torch.float32)

        def _mcqm_native_quality(self, cls_score, valid_ratio):
            return torch.ones((cls_score.shape[0], cls_score.shape[1]), dtype=cls_score.dtype)

        def _mcqm_build_memory_candidates(self, img_metas, ego_feat, current_native_centers):
            return None

    model = Dummy()
    current_mask = torch.zeros((1, 1040), dtype=torch.bool)
    current_mask[0, :720] = True
    query_feat = torch.randn(1, 1, 1040, 8)
    query_points = torch.randn(1, 1, 1040, 4, 3)
    cls_score = torch.randn(1, 1, 1040, 4, 17)
    feat_before = query_feat.clone()
    points_before = query_points.clone()
    cls_before = cls_score.clone()
    out = hook(
        model,
        decoder_runtime={"current_mask": current_mask, "ego_feat": torch.randn(1, 1, 32)},
        layer_idx=1,
        query_points=query_points,
        query_feat=query_feat,
        cls_score=cls_score,
        sampling_stats={},
        img_metas=[{}],
    )
    assert torch.equal(out["query_feat"], feat_before)
    assert torch.equal(out["query_points"], points_before)
    assert torch.equal(out["cls_score"], cls_before)


def test_real_hook_derives_current_mask_when_decoder_runtime_missing_it():
    hook = _load_sparseworld_hook_method()

    class Head:
        pc_range = [-80.0, -60.0, -3.0, 80.0, 60.0, 6.6]
        ind_stamps_all = torch.tensor([[0] * 720 + [1] * 320], dtype=torch.long)

    class Dummy:
        mcqm_enabled = True
        mcqm_version = "v1"
        mcqm_cfg = {"replacement_budget": 180, "replacement_margin": 0.05}
        num_query = 720
        pts_bbox_head = Head()

        def _mcqm_inject_after_layer1_legacy(self, *args, **kwargs):
            raise AssertionError("legacy path should not be used")

        def _mcqm_valid_sampling_ratio(self, sampling_stats, current_mask):
            return torch.ones((1, int(current_mask.sum().item())), dtype=torch.float32)

        def _mcqm_native_quality(self, cls_score, valid_ratio):
            return torch.ones((cls_score.shape[0], cls_score.shape[1]), dtype=cls_score.dtype)

        def _mcqm_build_memory_candidates(self, img_metas, ego_feat, current_native_centers):
            return None

    model = Dummy()
    query_feat = torch.randn(1, 1040, 8)
    query_points = torch.randn(1, 1040, 4, 3)
    cls_score = torch.randn(1, 1040, 4, 17)
    out = hook(
        model,
        decoder_runtime={"current_mask": None, "ego_feat": torch.randn(1, 1, 32)},
        layer_idx=1,
        query_points=query_points,
        query_feat=query_feat,
        cls_score=cls_score,
        sampling_stats={},
        img_metas=[{}],
    )
    assert torch.equal(out["query_feat"], query_feat)


def test_invalid_high_quality_candidate_cannot_suppress_valid_nms_candidate():
    cand_invalid = {
        "quality": torch.tensor(0.9),
        "valid": False,
        "dominant_class": 3,
        "anchor_center": torch.tensor([0.0, 0.0, 0.0]),
    }
    cand_valid = {
        "quality": torch.tensor(0.8),
        "valid": True,
        "dominant_class": 3,
        "anchor_center": torch.tensor([0.3, 0.0, 0.0]),
    }
    selected, stats = select_valid_memory_candidates_for_nms([cand_invalid, cand_valid], radius=0.8)
    assert stats["memory_candidates"] == 2
    assert stats["memory_valid_before_nms"] == 1
    assert stats["memory_candidates_after_nms"] == 1
    assert len(selected) == 1
    assert selected[0]["quality"].item() == cand_valid["quality"].item()


def test_local_matching_uses_metric_distance_not_normalized_distance():
    native_feat = torch.randn(1, 3)
    native_pts_norm = torch.tensor([[[0.10, 0.50, 0.50]] * 4], dtype=torch.float32)
    native_logits = torch.randn(1, 4, 17)
    native_quality = torch.tensor([0.1], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts_norm = torch.tensor([[[0.20, 0.50, 0.50]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.9], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    native_centers_metric = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    memory_centers_metric = torch.tensor([[20.0, 0.0, 0.0]], dtype=torch.float32)
    out = select_dynamic_query_replacements(
        native_feat, native_pts_norm, native_logits, native_quality,
        mem_feat, mem_pts_norm, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        native_query_classes=torch.tensor([4]),
        memory_query_classes=torch.tensor([4]),
        native_query_centers_metric=native_centers_metric,
        memory_query_centers_metric=memory_centers_metric,
        local_match_radius_m=2.0,
        native_bottomk_ratio=1.0,
        matching_mode="local_same_class",
    )
    assert out["actual_replacement_count"] == 0


def test_local_matching_requires_metric_centers():
    native_feat = torch.randn(1, 3)
    native_pts = torch.tensor([[[0.1, 0.5, 0.5]] * 4], dtype=torch.float32)
    native_logits = torch.randn(1, 4, 17)
    native_quality = torch.tensor([0.1], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.tensor([[[0.2, 0.5, 0.5]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.9], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    try:
        select_dynamic_query_replacements(
            native_feat, native_pts, native_logits, native_quality,
            mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
            max_replacement_budget=64,
            replacement_margin=0.0,
            native_query_classes=torch.tensor([4]),
            memory_query_classes=torch.tensor([4]),
            matching_mode="local_same_class",
        )
    except ValueError as exc:
        assert "metric" in str(exc)
    else:
        raise AssertionError("expected local matcher to require metric centers")


def test_global_matching_requires_metric_centers_too():
    native_feat = torch.randn(1, 3)
    native_pts = torch.randn(1, 4, 3)
    native_logits = torch.randn(1, 4, 17)
    native_quality = torch.tensor([0.1], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.randn(1, 4, 3)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.9], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    try:
        select_dynamic_query_replacements(
            native_feat, native_pts, native_logits, native_quality,
            mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
            max_replacement_budget=64,
            replacement_margin=0.0,
            native_query_classes=torch.tensor([4]),
            memory_query_classes=torch.tensor([7]),
            matching_mode="global",
        )
    except ValueError as exc:
        assert "metric" in str(exc)
    else:
        raise AssertionError("expected global matcher to require metric centers")


def test_full_time_contract_modes():
    t0 = resolve_mcqm_full_time_contract(0.6, 0.5, "fixed_horizon_mcqm")
    assert t0["forecast_query_time"] == 0.5
    assert t0["reg_time_scale"] == 0.5
    assert t0["vel_time_scale"] == 1.0
    t1 = resolve_mcqm_full_time_contract(0.6, 0.5, "reg_only_delta_t")
    assert t1["forecast_query_time"] == 0.6
    assert t1["reg_time_scale"] == 0.6
    assert t1["vel_time_scale"] == 1.0
    t2 = resolve_mcqm_full_time_contract(0.6, 0.5, "reg_and_vel_delta_t")
    assert t2["forecast_query_time"] == 0.6
    assert t2["reg_time_scale"] == 0.6
    assert abs(float(t2["vel_time_scale"]) - 1.2) < 1e-6


def test_full_time_contract_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        resolve_mcqm_full_time_contract(0.5, 0.0, "reg_and_vel_delta_t")


def test_full_time_contract_rejects_invalid_delta_t():
    for value in [0.0, -0.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            resolve_mcqm_full_time_contract(value, 0.5, "reg_only_delta_t")


def test_full_time_contract_rejects_invalid_mode():
    with pytest.raises(ValueError):
        resolve_mcqm_full_time_contract(0.5, 0.5, "bad_mode")


def test_full_residual_modes_static_behavior():
    scaled_reg = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float32)
    scaled_vel = torch.tensor([[[[4.0, 5.0]]]], dtype=torch.float32)
    static_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
    all_reg = compose_mcqm_object_motion_offset(
        scaled_reg, scaled_vel, static_mask, residual_mode="full_all_reg"
    )
    assert torch.equal(all_reg, scaled_reg)
    dynamic_only = compose_mcqm_object_motion_offset(
        scaled_reg, scaled_vel, static_mask, residual_mode="full_dynamic_only"
    )
    assert torch.equal(dynamic_only, torch.zeros_like(scaled_reg))


def test_full_residual_modes_dynamic_behavior():
    scaled_reg = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float32)
    scaled_vel = torch.tensor([[[[4.0, 5.0]]]], dtype=torch.float32)
    dynamic_mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    dynamic_out = compose_mcqm_object_motion_offset(
        scaled_reg, scaled_vel, dynamic_mask, residual_mode="full_dynamic_only"
    )
    expected = torch.tensor([[[[5.0, 7.0, 3.0]]]], dtype=torch.float32)
    assert torch.equal(dynamic_out, expected)


def test_full_residual_modes_query_level_broadcast():
    scaled_reg = torch.tensor(
        [[
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
        ]],
        dtype=torch.float32,
    )
    scaled_vel = torch.tensor(
        [[
            [[4.0, 5.0], [4.0, 5.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]],
        dtype=torch.float32,
    )
    query_level_mask = torch.tensor([[[[True]], [[False]]]], dtype=torch.bool)
    out = compose_mcqm_object_motion_offset(
        scaled_reg, scaled_vel, query_level_mask, residual_mode="full_dynamic_only"
    )
    assert torch.equal(out[0, 0], torch.tensor([[5.0, 7.0, 3.0], [5.0, 7.0, 3.0]]))
    assert torch.equal(out[0, 1], torch.zeros_like(out[0, 1]))


def test_stack_selected_memory_candidates_preserves_query_ids():
    selected_candidates = [
        {
            "feat": torch.randn(3),
            "points": torch.randn(4, 3),
            "cls_logits": torch.randn(4, 17),
            "quality": torch.tensor(0.9),
            "confidence": torch.tensor(0.8),
            "motion_residual_norm": torch.tensor(0.2),
            "in_range_ratio": torch.tensor(1.0),
            "dominant_class": 4,
            "query_id": 25,
            "projector_residual_norm": torch.tensor(0.0),
        }
    ]
    packed = stack_selected_memory_candidates(selected_candidates)
    assert packed is not None
    assert torch.equal(packed["query_ids"], torch.tensor([25], dtype=torch.long))


def test_global_matching_mode_is_not_distance_gated():
    native_feat = torch.randn(1, 3)
    native_pts = torch.tensor([[[0.0, 0.0, 0.0]] * 4], dtype=torch.float32)
    native_logits = torch.randn(1, 4, 17)
    native_quality = torch.tensor([0.1], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.tensor([[[100.0, 0.0, 0.0]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.9], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        native_query_classes=torch.tensor([4]),
        memory_query_classes=torch.tensor([7]),
        native_query_centers_metric=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        memory_query_centers_metric=torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float32),
        local_match_radius_m=2.0,
        native_bottomk_ratio=1.0,
        matching_mode="global",
    )
    assert out["actual_replacement_count"] == 1


def test_native_quality_rejection_is_tracked_separately():
    native_feat = torch.randn(2, 3)
    native_pts = torch.tensor([
        [[0.0, 0.0, 0.0]] * 4,
        [[10.0, 0.0, 0.0]] * 4,
    ], dtype=torch.float32)
    native_logits = torch.randn(2, 4, 17)
    native_quality = torch.tensor([0.95, 0.10], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.tensor([[[0.1, 0.0, 0.0]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.99], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        native_query_classes=torch.tensor([4, 4]),
        memory_query_classes=torch.tensor([4]),
        native_query_centers_metric=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32),
        memory_query_centers_metric=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32),
        local_match_radius_m=2.0,
        native_bottomk_ratio=0.5,
        matching_mode="local_same_class",
    )
    assert out["actual_replacement_count"] == 0
    assert int(out["native_quality_rejected_count"]) == 1
    assert int(out["already_occupied_native_rejected_count"]) == 0


def test_nearby_high_quality_native_does_not_fall_back_to_far_low_quality_native():
    native_feat = torch.randn(3, 3)
    native_pts = torch.tensor([
        [[0.0, 0.0, 0.0]] * 4,
        [[0.8, 0.0, 0.0]] * 4,
        [[20.0, 0.0, 0.0]] * 4,
    ], dtype=torch.float32)
    native_logits = torch.randn(3, 4, 17)
    native_quality = torch.tensor([0.92, 0.88, 0.05], dtype=torch.float32)
    mem_feat = torch.randn(1, 3)
    mem_pts = torch.tensor([[[0.2, 0.0, 0.0]] * 4], dtype=torch.float32)
    mem_logits = torch.randn(1, 4, 17)
    mem_quality = torch.tensor([0.99], dtype=torch.float32)
    mem_valid = torch.tensor([True])
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        native_query_classes=torch.tensor([4, 4, 4]),
        memory_query_classes=torch.tensor([4]),
        native_query_centers_metric=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=torch.float32),
        memory_query_centers_metric=torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float32),
        local_match_radius_m=2.0,
        native_bottomk_ratio=0.34,
        matching_mode="local_same_class",
    )
    assert out["actual_replacement_count"] == 0
    assert int(out["native_quality_rejected_count"]) == 1
    assert int(out["distance_rejected_count"]) == 0


def test_extract_memory_state_uses_current_frame_provenance_mask():
    extract_method = _load_sparseworld_extract_memory_state_method()

    class Dummy:
        mcqm_cfg = {"allow_memory_repropagation": False}

        def _mcqm_scene_and_timestamp(self, meta):
            return "scene", float(meta["timestamp"])

    model = Dummy()
    query_feat = torch.randn(1, 4, 8)
    query_pos = torch.randn(1, 4, 4, 3)
    query_cls = torch.randn(1, 4, 4, 17)
    prev_debug = {
        "query_source": torch.tensor([[0, 1, 0, 0]], dtype=torch.long),
        "source_memory_query_id": torch.tensor([[-1, 9, -1, -1]], dtype=torch.long),
    }
    curr_debug = {
        "query_source": torch.tensor([[0, 0, 1, 0]], dtype=torch.long),
        "source_memory_query_id": torch.tensor([[-1, -1, 25, -1]], dtype=torch.long),
    }
    states = extract_method(
        model,
        img_metas=[{"timestamp": 1.0}],
        query_feat=query_feat,
        query_pos=query_pos,
        query_cls=query_cls,
        mcqm_debug=curr_debug,
    )
    state = states[0]
    assert state.query_feat.shape[0] == 3
    assert torch.equal(state.query_ids, torch.tensor([0, 1, 3], dtype=torch.long))
    assert 2 not in state.query_ids.tolist()
    assert 1 in state.query_ids.tolist()
    assert not state.query_feat.requires_grad
    assert not state.refine_points.requires_grad
    assert not state.cls_logits.requires_grad


def test_extract_memory_state_accepts_batched_provenance_tensors():
    extract_method = _load_sparseworld_extract_memory_state_method()

    class Dummy:
        mcqm_cfg = {"allow_memory_repropagation": False}

        def _mcqm_scene_and_timestamp(self, meta):
            return str(meta["scene_token"]), float(meta["timestamp"])

    model = Dummy()
    query_feat = torch.randn(2, 3, 8)
    query_pos = torch.randn(2, 3, 4, 3)
    query_cls = torch.randn(2, 3, 4, 17)
    curr_debug = {
        "query_source": torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.long),
        "source_memory_query_id": torch.tensor([[-1, 25, -1], [9, -1, -1]], dtype=torch.long),
    }
    states = extract_method(
        model,
        img_metas=[
            {"timestamp": 1.0, "scene_token": "scene-a"},
            {"timestamp": 2.0, "scene_token": "scene-b"},
        ],
        query_feat=query_feat,
        query_pos=query_pos,
        query_cls=query_cls,
        mcqm_debug=curr_debug,
    )
    assert len(states) == 2
    assert torch.equal(states[0].query_ids, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(states[1].query_ids, torch.tensor([1, 2], dtype=torch.long))


def test_extract_memory_state_accumulates_memory_age():
    extract_method = _load_sparseworld_extract_memory_state_method()

    class MemoryManager:
        def __init__(self, state):
            self.state = state

        def get(self, batch_index, scene_token, timestamp):
            return self.state

    class Dummy:
        mcqm_cfg = {"allow_memory_repropagation": True}

        def __init__(self, state):
            self.mcqm_memory_manager = MemoryManager(state)

        def _mcqm_scene_and_timestamp(self, meta):
            return "scene", float(meta["timestamp"])

    prev_state = mcqm_local.MemoryFrameState(
        query_feat=torch.randn(2, 8),
        refine_points=torch.randn(2, 4, 3),
        cls_logits=torch.randn(2, 4, 17),
        dominant_class=torch.tensor([4, 4], dtype=torch.long),
        confidence=torch.tensor([0.8, 0.7], dtype=torch.float32),
        query_ids=torch.tensor([10, 11], dtype=torch.long),
        query_source=torch.tensor([0, 1], dtype=torch.long),
        memory_age=torch.tensor([0, 2], dtype=torch.long),
        was_memory_injected=torch.tensor([False, True]),
        parent_memory_query_id=torch.tensor([-1, 5], dtype=torch.long),
        source_native_query_id=torch.tensor([10, 11], dtype=torch.long),
        timestamp=torch.tensor(1.0, dtype=torch.float64),
        scene_token="scene",
        ego2global=torch.eye(4),
    )
    model = Dummy(prev_state)
    query_feat = torch.randn(1, 3, 8)
    query_pos = torch.randn(1, 3, 4, 3)
    query_cls = torch.randn(1, 3, 4, 17)
    curr_debug = {
        "query_source": torch.tensor([[0, 1, 1]], dtype=torch.long),
        "source_memory_query_id": torch.tensor([[-1, 10, 11]], dtype=torch.long),
    }
    states = extract_method(
        model,
        img_metas=[{"timestamp": 1.5}],
        query_feat=query_feat,
        query_pos=query_pos,
        query_cls=query_cls,
        mcqm_debug=curr_debug,
    )
    state = states[0]
    assert torch.equal(state.memory_age, torch.tensor([0, 1, 3], dtype=torch.long))


def test_extract_memory_state_raises_on_missing_parent_in_strict_mode():
    extract_method = _load_sparseworld_extract_memory_state_method()

    class MemoryManager:
        def __init__(self, state):
            self.state = state

        def get(self, batch_index, scene_token, timestamp):
            return self.state

    class Dummy:
        mcqm_cfg = {"allow_memory_repropagation": True, "strict_memory_lineage": True}

        def __init__(self, state):
            self.mcqm_memory_manager = MemoryManager(state)

        def _mcqm_scene_and_timestamp(self, meta):
            return "scene", float(meta["timestamp"])

    prev_state = mcqm_local.MemoryFrameState(
        query_feat=torch.randn(1, 8),
        refine_points=torch.randn(1, 4, 3),
        cls_logits=torch.randn(1, 4, 17),
        dominant_class=torch.tensor([4], dtype=torch.long),
        confidence=torch.tensor([0.8], dtype=torch.float32),
        query_ids=torch.tensor([25], dtype=torch.long),
        query_source=torch.tensor([1], dtype=torch.long),
        memory_age=torch.tensor([2], dtype=torch.long),
        was_memory_injected=torch.tensor([True]),
        parent_memory_query_id=torch.tensor([10], dtype=torch.long),
        source_native_query_id=torch.tensor([25], dtype=torch.long),
        timestamp=torch.tensor(1.0, dtype=torch.float64),
        scene_token="scene",
        ego2global=torch.eye(4),
    )
    model = Dummy(prev_state)
    query_feat = torch.randn(1, 1, 8)
    query_pos = torch.randn(1, 1, 4, 3)
    query_cls = torch.randn(1, 1, 4, 17)
    curr_debug = {
        "query_source": torch.tensor([[1]], dtype=torch.long),
        "source_memory_query_id": torch.tensor([[1]], dtype=torch.long),
    }
    try:
        extract_method(
            model,
            img_metas=[{"timestamp": 1.5}],
            query_feat=query_feat,
            query_pos=query_pos,
            query_cls=query_cls,
            mcqm_debug=curr_debug,
        )
    except RuntimeError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected strict lineage failure")


def test_selector_preserves_persistent_memory_query_id_not_candidate_index():
    native_feat = torch.randn(2, 3)
    native_pts = torch.randn(2, 4, 3)
    native_logits = torch.randn(2, 4, 17)
    native_quality = torch.tensor([0.9, 0.1], dtype=torch.float32)
    mem_feat = torch.randn(2, 3)
    mem_pts = torch.randn(2, 4, 3)
    mem_logits = torch.randn(2, 4, 17)
    mem_quality = torch.tensor([0.2, 0.95], dtype=torch.float32)
    mem_valid = torch.tensor([True, True])
    mem_query_ids = torch.tensor([10, 25], dtype=torch.long)
    out = select_dynamic_query_replacements(
        native_feat, native_pts, native_logits, native_quality,
        mem_feat, mem_pts, mem_logits, mem_quality, mem_valid,
        max_replacement_budget=64,
        replacement_margin=0.0,
        memory_query_ids=mem_query_ids,
        native_query_classes=torch.tensor([4, 4]),
        memory_query_classes=torch.tensor([4, 4]),
        native_query_centers_metric=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        memory_query_centers_metric=torch.tensor([[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        local_match_radius_m=2.0,
        native_bottomk_ratio=1.0,
        matching_mode="local_same_class",
    )
    replaced_idx = int(out["native_replaced_indices"][0].item())
    assert int(out["source_memory_query_id"][replaced_idx].item()) == 25


def test_candidate_nms_preserves_persistent_query_id_after_filter_and_sort():
    raw_candidates = [
        {
            "quality": torch.tensor(0.99),
            "valid": False,
            "dominant_class": 4,
            "anchor_center": torch.tensor([0.0, 0.0, 0.0]),
            "query_id": 10,
        },
        {
            "quality": torch.tensor(0.90),
            "valid": True,
            "dominant_class": 4,
            "anchor_center": torch.tensor([1.0, 0.0, 0.0]),
            "query_id": 25,
        },
        {
            "quality": torch.tensor(0.80),
            "valid": True,
            "dominant_class": 4,
            "anchor_center": torch.tensor([1.2, 0.0, 0.0]),
            "query_id": 73,
        },
    ]
    selected, stats = select_valid_memory_candidates_for_nms(raw_candidates, radius=0.8)
    assert stats["memory_valid_before_nms"] == 2
    assert stats["memory_candidates_after_nms"] == 1
    assert len(selected) == 1
    assert int(selected[0]["query_id"]) == 25


def test_dense_occ_top1_conf_margin_returns_expected_semantic_and_margin():
    dense = torch.zeros((2, 2, 2, 4), dtype=torch.float32)
    dense[0, 0, 0] = torch.tensor([0.1, 0.7, 0.2, 0.0])
    dense[1, 1, 1] = torch.tensor([0.4, 0.3, 0.1, 0.2])
    conf, sem, margin = dense_occ_top1_conf_margin(dense)
    assert sem[0, 0, 0].item() == 1
    assert conf[0, 0, 0].item() == pytest.approx(0.7)
    assert margin[0, 0, 0].item() == pytest.approx(0.5)
    assert sem[1, 1, 1].item() == 0
    assert margin[1, 1, 1].item() == pytest.approx(0.1)


def test_build_s0_s1_s2_s3_masks_partitions_d3():
    shape = (3, 3, 3)
    gt_h = torch.full(shape, 17, dtype=torch.long)
    gt_h[0, 0, 0] = 4
    gt_h[0, 0, 1] = 4
    gt_h[0, 0, 2] = 4
    teacher_raw = torch.full(shape, 17, dtype=torch.long)
    teacher_final = torch.full(shape, 17, dtype=torch.long)
    teacher_conf = torch.zeros(shape, dtype=torch.float32)
    teacher_margin = torch.zeros(shape, dtype=torch.float32)
    geometric_mask = torch.zeros(shape, dtype=torch.bool)
    geometric_mask[0, 0, 1] = True
    geometric_mask[0, 0, 2] = True
    semantic_active_mask = torch.zeros(shape, dtype=torch.bool)
    semantic_active_mask[0, 0, 2] = True
    active_voxels = torch.tensor([[0, 0, 0]], dtype=torch.long)
    debug = {
        "geometric_mask": geometric_mask,
        "semantic_active_mask": semantic_active_mask,
        "active_voxels": active_voxels,
    }
    d3_mask = build_d3_mask_from_occ(
        gt_h,
        teacher_raw,
        teacher_final,
        teacher_conf,
        teacher_margin,
    )
    masks = build_s0_s1_s2_s3_masks(d3_mask, debug)
    assert int(masks["D3"].sum().item()) == 3
    assert int(masks["S0"].sum().item()) == 1
    assert int(masks["S1"].sum().item()) == 1
    assert int(masks["S2"].sum().item()) == 1
    assert int(masks["S3"].sum().item()) == 1


def test_compare_unselected_slots_ignores_selected_columns():
    before = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
    after = before.clone()
    after[:, 1] += 5.0
    result = compare_unselected_slots(before, after, [1])
    assert result["exact_equal"] is True
    result_bad = compare_unselected_slots(before, after, [0])
    assert result_bad["exact_equal"] is False


def test_mcqm_q2f_v1_cfg_contract():
    cfg = mcqm_q2f_v1_cfg()
    assert cfg["enabled"] is True
    assert cfg["mcqm_version"] == "q2f_v3_shallow"
    assert cfg["mcqm_q2f_enabled"] is True
    assert cfg["mcqm_motion_mode"] == "ego_only"
    assert cfg["mcqm_q2f_injection_mode"] == "shallow_hard_replace"


def test_mcqm_q2f_parameter_freeze_only_q2f_trainable():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.mcqm_q2f_shallow_query_projector = torch.nn.Linear(4, 4)
            self.mcqm_q2f_shallow_decoder = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3, padding=1))

    model = Dummy()
    manifest = apply_mcqm_q2f_parameter_freeze(model)
    assert manifest["trainable_parameter_count"] > 0
    assert all(name.startswith("mcqm_q2f_") for name in manifest["trainable_parameter_keys"])
    assert all((not param.requires_grad) for name, param in model.named_parameters() if not name.startswith("mcqm_q2f_"))


def test_set_mcqm_q2f_training_mode_keeps_base_eval_and_q2f_train():
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            self.mcqm_q2f_shallow_query_projector = torch.nn.Linear(4, 4)
            self.mcqm_q2f_shallow_decoder = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3, padding=1))

    model = Dummy()
    set_mcqm_q2f_training_mode(model)
    assert model.training is True
    assert model.backbone.training is False
    assert model.backbone[1].training is False
    assert model.mcqm_q2f_shallow_query_projector.training is True
    assert model.mcqm_q2f_shallow_decoder.training is True


def test_q2f_total_loss_separates_occ_and_q2f_terms():
    model = types.SimpleNamespace(mcqm_cfg={"mcqm_q2f_occ_loss_weight": 2.0})
    losses = {
        "loss_occ": torch.tensor(3.0),
        "loss_aux": torch.tensor(1.0),
        "loss_mcqm_q2f_feature_l1": torch.tensor(5.0),
        "loss_mcqm_q2f_feature_cos": torch.tensor(0.5),
    }
    total = _q2f_total_loss(model, losses)
    assert torch.isclose(total, torch.tensor(13.5))


def test_q2f_protocol_version_upgraded():
    assert mcqm_runner_local.MCQM_Q2F_PROTOCOL_VERSION == (
        "mcqm_q2f_v3_shallow_stage_hard_replace_pre_neck"
    )


def test_q2f_query_projector_auto_dims():
    projector = MCQMQ2FQueryProjector(query_dim=6, level_channels=[4, 8], hidden_dim=10)
    memory = torch.randn(2, 5, 6)
    levels = projector(memory)
    assert len(levels) == 2
    assert levels[0].shape == (2, 5, 4)
    assert levels[1].shape == (2, 5, 8)


def test_infer_camera_name_from_filename_avoids_prefix_collision():
    assert infer_camera_name_from_filename("/tmp/CAM_FRONT_RIGHT/x.jpg") == "CAM_FRONT_RIGHT"
    assert infer_camera_name_from_filename("/tmp/CAM_FRONT_LEFT/x.jpg") == "CAM_FRONT_LEFT"
    assert infer_camera_name_from_filename("/tmp/CAM_FRONT/x.jpg") == "CAM_FRONT"


def test_infer_temporal_camera_layout_uses_current_frame_order():
    filenames = [
        "/tmp/CAM_FRONT/x0.jpg",
        "/tmp/CAM_FRONT_RIGHT/x0.jpg",
        "/tmp/CAM_FRONT_LEFT/x0.jpg",
        "/tmp/CAM_BACK/x0.jpg",
        "/tmp/CAM_BACK_LEFT/x0.jpg",
        "/tmp/CAM_BACK_RIGHT/x0.jpg",
        "/tmp/CAM_FRONT/x1.jpg",
        "/tmp/CAM_FRONT_RIGHT/x1.jpg",
        "/tmp/CAM_FRONT_LEFT/x1.jpg",
        "/tmp/CAM_BACK/x1.jpg",
        "/tmp/CAM_BACK_LEFT/x1.jpg",
        "/tmp/CAM_BACK_RIGHT/x1.jpg",
    ]
    layout = infer_temporal_camera_layout(filenames)
    assert layout["camera_count"] == 6
    assert layout["num_frames"] == 2
    assert layout["camera_order"] == [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]


def test_bilinear_splat_integer_coordinate_hits_single_cell():
    features = torch.tensor([[[[2.0, 4.0]]]], dtype=torch.float32)
    coords = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    weights = torch.tensor([[[1.0]]], dtype=torch.float32)
    valid = torch.tensor([[[True]]])
    out = bilinear_splat_features(features, coords, weights, valid, (4, 4))
    assert out["support_mask"][0, 0, 2, 1].item() is True
    assert out["weight_sum"][0, 0, 2, 1].item() == pytest.approx(1.0)
    assert out["sparse_feature"][0, :, 2, 1].tolist() == pytest.approx([2.0, 4.0])


def test_bilinear_splat_half_pixel_splits_weights():
    features = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    coords = torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float32)
    weights = torch.tensor([[[1.0]]], dtype=torch.float32)
    valid = torch.tensor([[[True]]])
    out = bilinear_splat_features(features, coords, weights, valid, (3, 3))
    expected = 0.25
    assert out["weight_sum"][0, 0, 0, 0].item() == pytest.approx(expected)
    assert out["weight_sum"][0, 0, 0, 1].item() == pytest.approx(expected)
    assert out["weight_sum"][0, 0, 1, 0].item() == pytest.approx(expected)
    assert out["weight_sum"][0, 0, 1, 1].item() == pytest.approx(expected)


def test_bilinear_splat_preserves_gradient():
    features = torch.randn(1, 2, 3, 4, requires_grad=True)
    coords = torch.tensor(
        [[[[0.2, 0.1], [1.4, 1.2], [2.1, 0.7]], [[0.5, 2.0], [1.1, 0.9], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    weights = torch.ones(1, 2, 3, dtype=torch.float32)
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    out = bilinear_splat_features(features, coords, weights, valid, (4, 4))
    out["sparse_feature"].sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert float(features.grad.abs().sum().item()) > 0.0


def test_bilinear_splat_invalid_point_writes_nothing():
    features = torch.tensor([[[[3.0]]]], dtype=torch.float32)
    coords = torch.tensor([[[[-1.0, -1.0]]]], dtype=torch.float32)
    weights = torch.tensor([[[1.0]]], dtype=torch.float32)
    valid = torch.tensor([[[False]]])
    out = bilinear_splat_features(features, coords, weights, valid, (2, 2))
    assert float(out["weight_sum"].sum().item()) == 0.0
    assert float(out["support_mask"].float().sum().item()) == 0.0


def test_replace_failed_camera_fpn_levels_preserves_healthy_cameras():
    original = [torch.randn(1, 3, 2, 4, 5), torch.randn(1, 3, 2, 2, 3)]
    reconstructed = [level.clone() + 10.0 for level in original]
    mask = torch.tensor([[False, True, False]])
    out = replace_failed_camera_fpn_levels(original, reconstructed, mask)
    for level_index, repaired in enumerate(out):
        assert torch.equal(repaired[:, 0], original[level_index][:, 0])
        assert torch.equal(repaired[:, 2], original[level_index][:, 2])
        assert torch.equal(repaired[:, 1], reconstructed[level_index][:, 1])


def test_blend_failed_camera_fpn_levels_only_updates_supported_failed_region():
    original = [torch.zeros(1, 2, 1, 2, 2)]
    reconstructed = [torch.ones(1, 2, 1, 2, 2) * 10.0]
    support = [torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]]], [[[1.0, 1.0], [0.0, 0.0]]]]], dtype=torch.float32)]
    mask = torch.tensor([[False, True]])
    alpha = torch.tensor(0.5)
    out = blend_failed_camera_fpn_levels(original, reconstructed, mask, support, alpha)[0]
    assert torch.equal(out[:, 0], original[0][:, 0])
    assert out[0, 1, 0, 0, 0].item() == pytest.approx(5.0)
    assert out[0, 1, 0, 0, 1].item() == pytest.approx(5.0)
    assert out[0, 1, 0, 1, 0].item() == pytest.approx(0.0)
    assert out[0, 1, 0, 1, 1].item() == pytest.approx(0.0)


def test_q2f_projection_path_uses_motion_compensation_call():
    source = (
        PROJECT_ROOT
        / "external/SparseWorld/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    ).read_text(encoding="utf-8")
    assert "transform_points_between_egos(" in source


def test_q2f_eval_branch_routes_through_simple_test_online_source_contract():
    source = (
        PROJECT_ROOT
        / "external/SparseWorld/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    ).read_text(encoding="utf-8")
    assert "if self.training:" in source
    assert "if self.training or self.mcqm_q2f_enabled:" not in source
    assert "outs = self.simple_test_online(img_metas, img, **kwargs)" in source
    assert "def simple_test_online(self, img_metas, img=None, rescale=False, **kwargs):" in source


def test_q2f_bypass_runtime_flag_source_contract():
    source = (
        PROJECT_ROOT
        / "external/SparseWorld/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    ).read_text(encoding="utf-8")
    assert '"q2f_shallow_bypass"' in source


def test_q2f_cache_source_contract_clones_cached_feats_before_q2f_repair():
    source = (
        PROJECT_ROOT
        / "external/SparseWorld/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    ).read_text(encoding="utf-8")
    assert 'self.memory[cache_key] = [level.detach().clone() for level in cached_feats]' in source
    assert 'img_feats_curr = [level.clone() for level in cached_feats]' in source
    assert 'view = shallow.view(' in source
    assert 'working = view.clone()' in source


def test_build_q2f_eval_parity_record_requires_exact_tensor_and_sha_match():
    pred_a = {
        0: torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        2: torch.tensor([[5, 6]], dtype=torch.int64),
        4: torch.tensor([[7]], dtype=torch.int64),
        6: torch.tensor([[8]], dtype=torch.int64),
    }
    pred_b = {k: v.clone() for k, v in pred_a.items()}
    pred_c = {k: v.clone() for k, v in pred_a.items()}
    record = build_q2f_eval_parity_record(47, pred_a, pred_b, pred_c)
    assert record["sample_index"] == 47
    assert record["all_horizons_equal"] is True
    assert all(bool(row["baseline_vs_bypass_run1_equal"]) for row in record["rows"])
    assert all(bool(row["bypass_run1_vs_run2_equal"]) for row in record["rows"])
    pred_b[4] = torch.tensor([[9]], dtype=torch.int64)
    mismatch = build_q2f_eval_parity_record(47, pred_a, pred_b, pred_c)
    assert mismatch["all_horizons_equal"] is False
    row4 = next(row for row in mismatch["rows"] if int(row["horizon_s"]) == 4)
    assert row4["baseline_vs_bypass_run1_equal"] is False
    assert row4["baseline_vs_bypass_run1_sha256_equal"] is False
    pred_b = {k: v.clone() for k, v in pred_a.items()}
    pred_c[6] = torch.tensor([[99]], dtype=torch.int64)
    repeat_mismatch = build_q2f_eval_parity_record(47, pred_a, pred_b, pred_c)
    assert repeat_mismatch["all_horizons_equal"] is False
    row6 = next(row for row in repeat_mismatch["rows"] if int(row["horizon_s"]) == 6)
    assert row6["bypass_run1_vs_run2_equal"] is False
    assert row6["bypass_run1_vs_run2_sha256_equal"] is False


def test_project_occ_points_to_image_uses_all_refine_points():
    img_metas = [{
        "lidar2img": [np.eye(4, dtype=np.float32)],
        "ego2lidar": np.eye(4, dtype=np.float32),
        "img_shape": [(10, 20, 3)],
    }]
    occ2img = build_occ2img_from_img_metas(img_metas, device=torch.device("cpu"), dtype=torch.float32)
    points = torch.tensor([[[[2.0, 4.0, 2.0], [8.0, 6.0, 2.0]]]], dtype=torch.float32)
    projected = project_occ_points_to_image(points, occ2img, image_h=10, image_w=20)
    assert projected["uv"].shape == (1, 1, 1, 2, 2)
    assert projected["valid_mask"].shape == (1, 1, 1, 2)
    assert int(projected["valid_mask"].sum().item()) == 2
    first_uv = projected["uv"][0, 0, 0, 0]
    second_uv = projected["uv"][0, 0, 0, 1]
    assert not torch.equal(first_uv, second_uv)


def test_q2f_level_decoder_preserves_shape():
    decoder = MCQMQ2FLevelDecoder(channels=4, hidden_channels=8)
    sparse = torch.randn(2, 4, 5, 6)
    support = torch.ones(2, 1, 5, 6, dtype=torch.bool)
    conf = torch.ones(2, 1, 5, 6)
    out = decoder(sparse, support, conf)
    assert out.shape == sparse.shape


def test_q2f_smoke_gradient_chain():
    projector = MCQMQ2FQueryProjector(query_dim=3, level_channels=[2], hidden_dim=4)
    decoder = MCQMQ2FLevelDecoder(channels=2, hidden_channels=4)
    memory = torch.randn(1, 2, 3, requires_grad=True)
    projected = projector(memory)[0]
    point_features = projected.unsqueeze(2).expand(-1, -1, 2, -1)
    coords = torch.tensor([[[[0.25, 0.25], [1.25, 1.25]], [[0.5, 1.5], [1.5, 0.5]]]], dtype=torch.float32)
    weights = torch.ones(1, 2, 2, dtype=torch.float32)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    splat = bilinear_splat_features(point_features, coords, weights, valid, (3, 3))
    reconstructed = decoder(splat["sparse_feature"], splat["support_mask"], splat["weight_sum"])
    loss = reconstructed.square().mean()
    loss.backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()
    assert float(memory.grad.abs().sum().item()) > 0.0
