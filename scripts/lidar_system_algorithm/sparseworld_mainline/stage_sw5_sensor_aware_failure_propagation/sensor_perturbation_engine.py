from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


FIRST_ROUND_PERTURBATIONS = [
    "A0_clean",
    "A1_drop_cam_front",
    "A2_drop_cam_front_left",
    "A3_drop_cam_front_right",
    "A7_drop_all_rear",
    "A10_drop_front_triplet",
    "B1_low_light_05",
    "B2_low_light_03",
    "C2_gaussian_blur_5",
    "C4_motion_blur_9",
    "D2_yaw_p1_all",
    "D6_tx_p5cm_all",
    "D10_front_only_yaw_p1",
    "G1_random_rectangle_occlusion_mild",
    "G4_road_center_occlusion",
]


@dataclass
class PerturbationSpec:
    perturbation_id: str
    family: str
    severity: str
    description: str
    affected_cameras: list[str] | str
    params: dict[str, Any]
    deterministic_seed: int
    geometry_changed: bool
    image_changed: bool
    is_proxy: bool = False
    physically_meaningful: bool = True


def build_catalog() -> dict[str, PerturbationSpec]:
    catalog = {
        "A0_clean": PerturbationSpec("A0_clean", "A", "clean", "No perturbation baseline", "all", {}, 0, False, False, False, True),
        "A1_drop_cam_front": PerturbationSpec("A1_drop_cam_front", "A", "single", "Drop CAM_FRONT", ["CAM_FRONT"], {}, 11, False, True),
        "A2_drop_cam_front_left": PerturbationSpec("A2_drop_cam_front_left", "A", "single", "Drop CAM_FRONT_LEFT", ["CAM_FRONT_LEFT"], {}, 12, False, True),
        "A3_drop_cam_front_right": PerturbationSpec("A3_drop_cam_front_right", "A", "single", "Drop CAM_FRONT_RIGHT", ["CAM_FRONT_RIGHT"], {}, 13, False, True),
        "A4_drop_cam_back": PerturbationSpec("A4_drop_cam_back", "A", "single", "Drop CAM_BACK", ["CAM_BACK"], {}, 14, False, True),
        "A5_drop_cam_back_left": PerturbationSpec("A5_drop_cam_back_left", "A", "single", "Drop CAM_BACK_LEFT", ["CAM_BACK_LEFT"], {}, 15, False, True),
        "A6_drop_cam_back_right": PerturbationSpec("A6_drop_cam_back_right", "A", "single", "Drop CAM_BACK_RIGHT", ["CAM_BACK_RIGHT"], {}, 16, False, True),
        "A7_drop_all_rear": PerturbationSpec("A7_drop_all_rear", "A", "triple", "Drop rear triplet", ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], {}, 17, False, True),
        "A8_drop_random_one": PerturbationSpec("A8_drop_random_one", "A", "random", "Drop one random camera", "random_one", {}, 18, False, True),
        "A9_drop_two_adjacent": PerturbationSpec("A9_drop_two_adjacent", "A", "double", "Drop two adjacent cameras", ["CAM_FRONT", "CAM_FRONT_RIGHT"], {}, 19, False, True),
        "A10_drop_front_triplet": PerturbationSpec("A10_drop_front_triplet", "A", "triple", "Drop front triplet", ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"], {}, 20, False, True),
        "B1_low_light_05": PerturbationSpec("B1_low_light_05", "B", "mild", "Multiply image intensity by 0.5", "all", {"scale": 0.5}, 21, False, True),
        "B2_low_light_03": PerturbationSpec("B2_low_light_03", "B", "strong", "Multiply image intensity by 0.3", "all", {"scale": 0.3}, 22, False, True),
        "B3_over_exposure_15": PerturbationSpec("B3_over_exposure_15", "B", "mild", "Multiply image intensity by 1.5", "all", {"scale": 1.5}, 23, False, True),
        "B4_contrast_down_05": PerturbationSpec("B4_contrast_down_05", "B", "mild", "Contrast down 0.5", "all", {"contrast": 0.5}, 24, False, True),
        "B5_contrast_up_15": PerturbationSpec("B5_contrast_up_15", "B", "mild", "Contrast up 1.5", "all", {"contrast": 1.5}, 25, False, True),
        "B6_gaussian_noise_mild": PerturbationSpec("B6_gaussian_noise_mild", "B", "mild", "Gaussian noise sigma 8", "all", {"sigma": 8.0}, 26, False, True),
        "B7_gaussian_noise_strong": PerturbationSpec("B7_gaussian_noise_strong", "B", "strong", "Gaussian noise sigma 20", "all", {"sigma": 20.0}, 27, False, True),
        "C1_gaussian_blur_3": PerturbationSpec("C1_gaussian_blur_3", "C", "mild", "Gaussian blur k=3", "all", {"kernel": 3, "sigma": 1.0}, 31, False, True),
        "C2_gaussian_blur_5": PerturbationSpec("C2_gaussian_blur_5", "C", "medium", "Gaussian blur k=5", "all", {"kernel": 5, "sigma": 1.2}, 32, False, True),
        "C3_motion_blur_5": PerturbationSpec("C3_motion_blur_5", "C", "mild", "Motion blur k=5", "all", {"kernel": 5}, 33, False, True),
        "C4_motion_blur_9": PerturbationSpec("C4_motion_blur_9", "C", "strong", "Motion blur k=9", "all", {"kernel": 9}, 34, False, True),
        "D1_yaw_p05_all": PerturbationSpec("D1_yaw_p05_all", "D", "mild", "Yaw +0.5 degree all cameras", "all", {"yaw_deg": 0.5}, 41, True, False),
        "D2_yaw_p1_all": PerturbationSpec("D2_yaw_p1_all", "D", "medium", "Yaw +1.0 degree all cameras", "all", {"yaw_deg": 1.0}, 42, True, False),
        "D3_yaw_m1_all": PerturbationSpec("D3_yaw_m1_all", "D", "medium", "Yaw -1.0 degree all cameras", "all", {"yaw_deg": -1.0}, 43, True, False),
        "D4_pitch_p05_all": PerturbationSpec("D4_pitch_p05_all", "D", "mild", "Pitch +0.5 degree all cameras", "all", {"pitch_deg": 0.5}, 44, True, False),
        "D5_pitch_p1_all": PerturbationSpec("D5_pitch_p1_all", "D", "medium", "Pitch +1.0 degree all cameras", "all", {"pitch_deg": 1.0}, 45, True, False),
        "D6_tx_p5cm_all": PerturbationSpec("D6_tx_p5cm_all", "D", "mild", "Translation x +5 cm all cameras", "all", {"tx_m": 0.05}, 46, True, False),
        "D7_tx_p10cm_all": PerturbationSpec("D7_tx_p10cm_all", "D", "medium", "Translation x +10 cm all cameras", "all", {"tx_m": 0.10}, 47, True, False),
        "D8_ty_p5cm_all": PerturbationSpec("D8_ty_p5cm_all", "D", "mild", "Translation y +5 cm all cameras", "all", {"ty_m": 0.05}, 48, True, False),
        "D9_tz_p5cm_all": PerturbationSpec("D9_tz_p5cm_all", "D", "mild", "Translation z +5 cm all cameras", "all", {"tz_m": 0.05}, 49, True, False),
        "D10_front_only_yaw_p1": PerturbationSpec("D10_front_only_yaw_p1", "D", "medium", "Yaw +1.0 degree front camera only", ["CAM_FRONT"], {"yaw_deg": 1.0}, 50, True, False),
        "D11_side_camera_yaw_mismatch": PerturbationSpec("D11_side_camera_yaw_mismatch", "D", "medium", "Yaw mismatch on side cameras", ["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], {"yaw_deg": 1.0}, 51, True, False),
        "E1_focal_scale_098": PerturbationSpec("E1_focal_scale_098", "E", "mild", "Focal scale 0.98", "all", {"focal_scale": 0.98}, 61, True, False),
        "E2_focal_scale_102": PerturbationSpec("E2_focal_scale_102", "E", "mild", "Focal scale 1.02", "all", {"focal_scale": 1.02}, 62, True, False),
        "E3_pp_shift_p5": PerturbationSpec("E3_pp_shift_p5", "E", "mild", "Principal point shift +5 px", "all", {"cx_shift": 5.0, "cy_shift": 5.0}, 63, True, False),
        "E4_pp_shift_p10": PerturbationSpec("E4_pp_shift_p10", "E", "medium", "Principal point shift +10 px", "all", {"cx_shift": 10.0, "cy_shift": 10.0}, 64, True, False),
        "F1_previous_frame_substitute_proxy": PerturbationSpec("F1_previous_frame_substitute_proxy", "F", "proxy", "Per-camera one-step lag proxy", "all", {}, 71, False, True, True, False),
        "F2_one_camera_timestamp_lag_proxy": PerturbationSpec("F2_one_camera_timestamp_lag_proxy", "F", "proxy", "Front camera one-step lag proxy", ["CAM_FRONT"], {}, 72, False, True, True, False),
        "F3_all_camera_one_frame_lag_proxy": PerturbationSpec("F3_all_camera_one_frame_lag_proxy", "F", "proxy", "All cameras one-frame lag proxy", "all", {}, 73, False, True, True, False),
        "F4_random_view_temporal_mismatch_proxy": PerturbationSpec("F4_random_view_temporal_mismatch_proxy", "F", "proxy", "Random-view temporal mismatch proxy", "random_one", {}, 74, False, True, True, False),
        "G1_random_rectangle_occlusion_mild": PerturbationSpec("G1_random_rectangle_occlusion_mild", "G", "mild", "Random rectangle occlusion mild", "all", {"ratio": 0.18}, 81, False, True),
        "G2_random_rectangle_occlusion_strong": PerturbationSpec("G2_random_rectangle_occlusion_strong", "G", "strong", "Random rectangle occlusion strong", "all", {"ratio": 0.32}, 82, False, True),
        "G3_center_vertical_occlusion": PerturbationSpec("G3_center_vertical_occlusion", "G", "medium", "Center vertical occlusion", "all", {"width_ratio": 0.18}, 83, False, True),
        "G4_road_center_occlusion": PerturbationSpec("G4_road_center_occlusion", "G", "medium", "Road-center occlusion", "all", {"width_ratio": 0.28, "height_ratio": 0.22}, 84, False, True),
        "G5_side_occlusion": PerturbationSpec("G5_side_occlusion", "G", "medium", "Side occlusion", "all", {"width_ratio": 0.25}, 85, False, True),
    }
    return catalog


def get_first_round_specs() -> list[PerturbationSpec]:
    catalog = build_catalog()
    return [catalog[key] for key in FIRST_ROUND_PERTURBATIONS]


def clone_collated_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(batch)


def camera_name_from_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    for name in CAMERA_NAMES:
        if f"/{name}/" in normalized or f"__{name}__" in normalized:
            return name
    match = re.search(r"CAM_[A-Z_]+", filename)
    return match.group(0) if match else "UNKNOWN"


def resolve_camera_groups(filenames: list[str]) -> dict[str, list[int]]:
    out = {name: [] for name in CAMERA_NAMES}
    for idx, name in enumerate(filenames):
        out.setdefault(camera_name_from_filename(name), []).append(idx)
    return out


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _to_float_img(img: torch.Tensor) -> torch.Tensor:
    return img.float()


def _back_to_uint8(img: torch.Tensor) -> torch.Tensor:
    return img.clamp(0, 255).round().to(torch.uint8)


def _apply_scale(img: torch.Tensor, scale: float) -> torch.Tensor:
    return _back_to_uint8(_to_float_img(img) * scale)


def _apply_contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
    mean = _to_float_img(img).mean(dim=(-2, -1), keepdim=True)
    return _back_to_uint8(((_to_float_img(img) - mean) * factor) + mean)


def _apply_noise(img: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=img.device)
    g.manual_seed(seed)
    noise = torch.randn(img.shape, generator=g, device=img.device) * sigma
    return _back_to_uint8(_to_float_img(img) + noise)


def _gaussian_kernel(kernel: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(kernel) - (kernel - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def _motion_kernel(kernel: int) -> torch.Tensor:
    k = torch.zeros((kernel, kernel), dtype=torch.float32)
    k[kernel // 2, :] = 1.0 / kernel
    return k


def _blur(img: torch.Tensor, kernel2d: torch.Tensor) -> torch.Tensor:
    x = _to_float_img(img)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise ValueError(f"unsupported blur input shape: {tuple(x.shape)}")
    kernel = kernel2d.to(x.device).view(1, 1, *kernel2d.shape).repeat(3, 1, 1, 1)
    out = F.conv2d(x, kernel, padding=kernel2d.shape[0] // 2, groups=3)
    return _back_to_uint8(out.squeeze(0) if img.dim() == 3 else out)


def _rotation_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    return rz @ ry


def _right_compose_lidar2img(mat: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return (mat @ delta).astype(np.float32)


def _intrinsic_left_compose(mat: np.ndarray, focal_scale: float = 1.0, cx_shift: float = 0.0, cy_shift: float = 0.0) -> np.ndarray:
    delta = np.eye(4, dtype=np.float32)
    delta[0, 0] = focal_scale
    delta[1, 1] = focal_scale
    delta[0, 2] = cx_shift
    delta[1, 2] = cy_shift
    return (delta @ mat).astype(np.float32)


def _target_cameras(spec: PerturbationSpec, filenames: list[str]) -> list[str]:
    if spec.affected_cameras == "all":
        return list(CAMERA_NAMES)
    if spec.affected_cameras == "random_one":
        g = _rng(spec.deterministic_seed)
        return [CAMERA_NAMES[int(g.integers(0, len(CAMERA_NAMES)))]]
    return list(spec.affected_cameras)


def _camera_indices_for_spec(spec: PerturbationSpec, filenames: list[str]) -> list[int]:
    groups = resolve_camera_groups(filenames)
    target_cams = _target_cameras(spec, filenames)
    idxs: list[int] = []
    for cam in target_cams:
        idxs.extend(groups.get(cam, []))
    return sorted(idxs)


def _apply_image_mask(img_tensor: torch.Tensor, indices: list[int], fill_value: int = 0) -> None:
    if not indices:
        return
    img_tensor[:, indices] = fill_value


def _apply_occlusion(img_tensor: torch.Tensor, idxs: list[int], x0: int, y0: int, x1: int, y1: int, fill_value: int = 0) -> None:
    if not idxs:
        return
    img_tensor[:, idxs, :, y0:y1, x0:x1] = fill_value


def _shift_temporal_proxy(img_tensor: torch.Tensor, filenames: list[str], idxs: list[int], reverse: bool = False) -> None:
    if not idxs:
        return
    for cam in CAMERA_NAMES:
        cam_idxs = [i for i in idxs if camera_name_from_filename(filenames[i]) == cam]
        if len(cam_idxs) <= 1:
            continue
        src = cam_idxs[1:] + cam_idxs[-1:] if not reverse else cam_idxs[:1] + cam_idxs[:-1]
        img_tensor[:, cam_idxs] = img_tensor[:, src]


def apply_perturbation_to_batch(batch: dict[str, Any], spec: PerturbationSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    batch = clone_collated_batch(batch)
    img_tensor = batch["img"][0].data[0].clone()
    batch["img"][0].data[0] = img_tensor
    meta_list = copy.deepcopy(batch["img_metas"][0].data[0])
    batch["img_metas"][0].data[0] = meta_list
    meta = meta_list[0]
    filenames = list(meta["filename"])
    target_indices = _camera_indices_for_spec(spec, filenames)
    manifest = {
        "perturbation_id": spec.perturbation_id,
        "family": spec.family,
        "severity": spec.severity,
        "description": spec.description,
        "affected_cameras": _target_cameras(spec, filenames),
        "affected_indices": target_indices,
        "params": dict(spec.params),
        "deterministic_seed": spec.deterministic_seed,
        "geometry_changed": spec.geometry_changed,
        "image_changed": spec.image_changed,
        "is_proxy": spec.is_proxy,
        "physically_meaningful": spec.physically_meaningful,
    }

    if spec.perturbation_id == "A0_clean":
        return batch, manifest

    if spec.family == "A":
        _apply_image_mask(img_tensor, target_indices, fill_value=0)
        manifest["image_tensor_changed"] = True
        return batch, manifest

    if spec.family == "B":
        if "scale" in spec.params:
            img_tensor[:, target_indices if target_indices else slice(None)] = _apply_scale(
                img_tensor[:, target_indices if target_indices else slice(None)], float(spec.params["scale"])
            )
        elif "contrast" in spec.params:
            img_tensor[:, target_indices if target_indices else slice(None)] = _apply_contrast(
                img_tensor[:, target_indices if target_indices else slice(None)], float(spec.params["contrast"])
            )
        elif "sigma" in spec.params:
            idxs = target_indices if target_indices else list(range(img_tensor.shape[1]))
            for offset, idx in enumerate(idxs):
                img_tensor[:, idx] = _apply_noise(img_tensor[:, idx], float(spec.params["sigma"]), spec.deterministic_seed + offset)
        manifest["image_tensor_changed"] = True
        return batch, manifest

    if spec.family == "C":
        if "gaussian" in spec.perturbation_id or "C1" in spec.perturbation_id or "C2" in spec.perturbation_id:
            kernel = _gaussian_kernel(int(spec.params["kernel"]), float(spec.params["sigma"]))
        else:
            kernel = _motion_kernel(int(spec.params["kernel"]))
        idxs = target_indices if target_indices else list(range(img_tensor.shape[1]))
        for idx in idxs:
            img_tensor[:, idx] = _blur(img_tensor[:, idx], kernel)
        manifest["image_tensor_changed"] = True
        return batch, manifest

    if spec.family == "D":
        delta = np.eye(4, dtype=np.float32)
        if "yaw_deg" in spec.params or "pitch_deg" in spec.params:
            delta[:3, :3] = _rotation_matrix(float(spec.params.get("yaw_deg", 0.0)), float(spec.params.get("pitch_deg", 0.0)))
        delta[:3, 3] = [
            float(spec.params.get("tx_m", 0.0)),
            float(spec.params.get("ty_m", 0.0)),
            float(spec.params.get("tz_m", 0.0)),
        ]
        for idx in target_indices:
            meta["lidar2img"][idx] = _right_compose_lidar2img(np.asarray(meta["lidar2img"][idx], dtype=np.float32), delta)
        manifest["geometry_matrix_changed_count"] = len(target_indices)
        return batch, manifest

    if spec.family == "E":
        for idx in target_indices:
            meta["lidar2img"][idx] = _intrinsic_left_compose(
                np.asarray(meta["lidar2img"][idx], dtype=np.float32),
                float(spec.params.get("focal_scale", 1.0)),
                float(spec.params.get("cx_shift", 0.0)),
                float(spec.params.get("cy_shift", 0.0)),
            )
        manifest["geometry_matrix_changed_count"] = len(target_indices)
        return batch, manifest

    if spec.family == "F":
        idxs = target_indices if target_indices else list(range(img_tensor.shape[1]))
        _shift_temporal_proxy(img_tensor, filenames, idxs)
        manifest["image_tensor_changed"] = True
        return batch, manifest

    if spec.family == "G":
        idxs = target_indices if target_indices else list(range(img_tensor.shape[1]))
        _, _, _, h, w = img_tensor.shape
        if spec.perturbation_id in {"G1_random_rectangle_occlusion_mild", "G2_random_rectangle_occlusion_strong"}:
            ratio = float(spec.params["ratio"])
            rng = _rng(spec.deterministic_seed)
            rw = max(8, int(w * ratio))
            rh = max(8, int(h * ratio))
            x0 = int(rng.integers(0, max(1, w - rw)))
            y0 = int(rng.integers(0, max(1, h - rh)))
            _apply_occlusion(img_tensor, idxs, x0, y0, x0 + rw, y0 + rh, fill_value=0)
            manifest["occlusion_box"] = [x0, y0, x0 + rw, y0 + rh]
        elif spec.perturbation_id == "G3_center_vertical_occlusion":
            ww = int(w * float(spec.params["width_ratio"]))
            x0 = (w - ww) // 2
            _apply_occlusion(img_tensor, idxs, x0, 0, x0 + ww, h, fill_value=0)
            manifest["occlusion_box"] = [x0, 0, x0 + ww, h]
        elif spec.perturbation_id == "G4_road_center_occlusion":
            ww = int(w * float(spec.params["width_ratio"]))
            hh = int(h * float(spec.params["height_ratio"]))
            x0 = (w - ww) // 2
            y0 = h - hh
            _apply_occlusion(img_tensor, idxs, x0, y0, x0 + ww, h, fill_value=0)
            manifest["occlusion_box"] = [x0, y0, x0 + ww, h]
        elif spec.perturbation_id == "G5_side_occlusion":
            ww = int(w * float(spec.params["width_ratio"]))
            _apply_occlusion(img_tensor, idxs, 0, 0, ww, h, fill_value=0)
            manifest["occlusion_box"] = [0, 0, ww, h]
        manifest["image_tensor_changed"] = True
        return batch, manifest

    return batch, manifest


def catalog_manifest(catalog: dict[str, PerturbationSpec]) -> dict[str, Any]:
    return {
        "perturbation_count": len(catalog),
        "first_round_ids": list(FIRST_ROUND_PERTURBATIONS),
        "catalog": {k: asdict(v) for k, v in catalog.items()},
    }
