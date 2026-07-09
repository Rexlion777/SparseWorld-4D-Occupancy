from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None

if not hasattr(np, "Inf"):
    np.Inf = np.inf


EMPTY_IDX = 17
IGNORE_IDX = 255
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
VOXEL_SIZE = [0.4, 0.4, 0.4]
GRID_SIZE = [200, 200, 16]
HORIZON_LABELS = {0: "t=0s", 2: "t=1s", 4: "t=2s", 6: "t=3s"}

OCC_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

PALETTE = {
    "others": "#A9A9A9",
    "barrier": "#5A1672",
    "bicycle": "#4F6BED",
    "bus": "#C62828",
    "car": "#F7A10A",
    "construction_vehicle": "#E15759",
    "motorcycle": "#304FFE",
    "pedestrian": "#1717D8",
    "traffic_cone": "#F1C40F",
    "trailer": "#C97A00",
    "truck": "#FF6245",
    "driveable_surface": "#1BC8C1",
    "other_flat": "#7BD0CB",
    "sidewalk": "#76C442",
    "terrain": "#5A1672",
    "manmade": "#DEBA84",
    "vegetation": "#10C000",
}

DEFAULT_VISIBLE_CLASSES = [
    "car",
    "truck",
    "bus",
    "pedestrian",
    "bicycle",
    "motorcycle",
    "traffic_cone",
    "driveable_surface",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "barrier",
]

_O3D_VIS_CACHE: dict[tuple[int, int, str, str], Any] = {}
_O3D_CAMERA_CACHE: dict[str, Any] = {}


@dataclass
class RendererStyle:
    background: str = "white"
    figsize: tuple[float, float] = (5.6, 4.8)
    dpi: int = 300
    elev: float = 26.0
    azim: float = -62.0
    point_size: float = 4.5
    zoom_point_size: float = 10.0
    xlim: tuple[float, float] = (-35.0, 35.0)
    ylim: tuple[float, float] = (-30.0, 30.0)
    zlim: tuple[float, float] = (-1.0, 4.8)
    title_fontsize: int = 13
    label_fontsize: int = 14
    legend_fontsize: int = 9
    camera_facecolor: str = "white"
    mode: str = "voxel_mesh"
    backend: str = "matplotlib"
    viewpoint: str = "top"
    render_width: int = 1600
    render_height: int = 900
    open3d_point_size: float = 3.0
    mesh_voxel_scale: float = 1.03
    shadow_offset_xy: tuple[int, int] = (10, 10)
    shadow_blur_radius: float = 7.0
    shadow_alpha: int = 68
    mask_sky_layers: int = 3
    top_mask_sky_layers: int = 6
    mask_ego: bool = True
    visible_classes: tuple[str, ...] = tuple(DEFAULT_VISIBLE_CLASSES)
    bev_height_shading: bool = True


def save_style_config(path: Path, style: RendererStyle) -> None:
    payload = {
        "pc_range": PC_RANGE,
        "voxel_size": VOXEL_SIZE,
        "grid_size": GRID_SIZE,
        "empty_idx": EMPTY_IDX,
        "ignore_idx": IGNORE_IDX,
        "occ_names": OCC_NAMES,
        "palette": PALETTE,
        "horizon_label_mapping": {"h0": "0s", "h2": "1s", "h4": "2s", "h6": "3s"},
        "style": asdict(style),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def voxel_to_metric(indices_xyz: np.ndarray) -> np.ndarray:
    coords = np.zeros_like(indices_xyz, dtype=np.float32)
    coords[:, 0] = PC_RANGE[0] + (indices_xyz[:, 0] + 0.5) * VOXEL_SIZE[0]
    coords[:, 1] = PC_RANGE[1] + (indices_xyz[:, 1] + 0.5) * VOXEL_SIZE[1]
    coords[:, 2] = PC_RANGE[2] + (indices_xyz[:, 2] + 0.5) * VOXEL_SIZE[2]
    return coords


def occupancy_points(label_grid: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    grid = torch.as_tensor(label_grid).cpu().long()
    active = (grid != EMPTY_IDX) & (grid != IGNORE_IDX)
    idx = active.nonzero(as_tuple=False).numpy()
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    coords = voxel_to_metric(idx.astype(np.float32))
    labels = grid[active].numpy()
    return coords, labels


def surface_voxel_indices_and_labels(label_grid: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.array(torch.as_tensor(label_grid).cpu().numpy(), copy=False)
    active = (grid != EMPTY_IDX) & (grid != IGNORE_IDX)
    if not active.any():
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int64)
    pad = np.pad(active, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False)
    neighbor_count = (
        pad[2:, 1:-1, 1:-1]
        + pad[:-2, 1:-1, 1:-1]
        + pad[1:-1, 2:, 1:-1]
        + pad[1:-1, :-2, 1:-1]
        + pad[1:-1, 1:-1, 2:]
        + pad[1:-1, 1:-1, :-2]
    )
    surface = active & (neighbor_count < 6)
    idx = np.argwhere(surface).astype(np.int32)
    labels = grid[surface].astype(np.int64)
    return idx, labels


def paper_mask_occ(label_grid: torch.Tensor | np.ndarray, top_view: bool = False, visual_ego: bool = False) -> np.ndarray:
    occ = np.array(torch.as_tensor(label_grid).cpu().numpy(), copy=True)
    occ[:, :, -3:] = EMPTY_IDX
    if top_view:
        occ[:, :, -6:] = EMPTY_IDX
    occ[93:107, 95:105, 4:8] = EMPTY_IDX
    if visual_ego:
        occ[96:103, 98:102, 4:7] = 4
    return occ


def class_colors(labels: np.ndarray) -> list[str]:
    colors: list[str] = []
    for cls_id in labels.tolist():
        if 0 <= cls_id < len(OCC_NAMES):
            colors.append(PALETTE[OCC_NAMES[int(cls_id)]])
        else:
            colors.append("#808080")
    return colors


def present_classes(labels: np.ndarray) -> list[str]:
    ids = sorted({int(x) for x in labels.tolist() if 0 <= int(x) < len(OCC_NAMES)})
    return [OCC_NAMES[i] for i in ids]


def autocrop_white(img: Image.Image, pad: int = 8) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < 252, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, img.width)
    y2 = min(int(ys.max()) + pad + 1, img.height)
    return img.crop((x1, y1, x2, y2))


def fit_panel(img: Image.Image, target_size: tuple[int, int], inner_scale: float = 0.98) -> Image.Image:
    bg = Image.new("RGB", target_size, "white")
    fitted = ImageOps.contain(img, (int(target_size[0] * inner_scale), int(target_size[1] * inner_scale)))
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    bg.paste(fitted, (x, y))
    return bg


def apply_soft_drop_shadow(img: Image.Image, style: RendererStyle) -> Image.Image:
    base = img.convert("RGBA")
    rgb = np.asarray(base.convert("RGB"))
    mask = np.any(rgb < 248, axis=2).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    shadow = Image.new("RGBA", base.size, (255, 255, 255, 0))
    shadow_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=float(style.shadow_blur_radius)))
    shadow_layer = Image.new("RGBA", base.size, (185, 185, 185, int(style.shadow_alpha)))
    shadow_alpha = Image.new("L", base.size, 0)
    shadow_alpha.paste(shadow_mask, tuple(int(x) for x in style.shadow_offset_xy))
    shadow_layer.putalpha(shadow_alpha)
    shadow = Image.alpha_composite(shadow, shadow_layer)
    merged = Image.alpha_composite(shadow, base)
    return merged.convert("RGB")


def hex_to_rgb01(color: str) -> list[float]:
    color = color.lstrip("#")
    return [int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def hex_to_rgb255(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def build_semantic_voxel_mesh(label_grid: torch.Tensor | np.ndarray) -> Any:
    if o3d is None:
        raise RuntimeError("open3d is not available in this Python environment")
    idx, labels = surface_voxel_indices_and_labels(label_grid)
    mesh = o3d.geometry.TriangleMesh()
    if idx.shape[0] == 0:
        return mesh
    centers = voxel_to_metric(idx.astype(np.float32))
    half = np.array(VOXEL_SIZE, dtype=np.float64) / 2.0
    for center, cls_id in zip(centers, labels.tolist()):
        cls_name = OCC_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(OCC_NAMES) else "others"
        color = hex_to_rgb01(PALETTE.get(cls_name, "#808080"))
        cube = o3d.geometry.TriangleMesh.create_box(
            width=float(VOXEL_SIZE[0]),
            height=float(VOXEL_SIZE[1]),
            depth=float(VOXEL_SIZE[2]),
        )
        cube.translate((center.astype(np.float64) - half).tolist())
        cube.paint_uniform_color(color)
        mesh += cube
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def make_shaded_cube_mesh(origin_xyz: np.ndarray, box_size_xyz: np.ndarray, base_color: list[float]) -> Any:
    if o3d is None:
        raise RuntimeError("open3d is not available in this Python environment")
    x0, y0, z0 = origin_xyz.tolist()
    sx, sy, sz = box_size_xyz.tolist()
    x1, y1, z1 = x0 + sx, y0 + sy, z0 + sz
    verts = np.array(
        [
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top
            [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],  # front
            [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1],  # right
            [x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1],  # left
            [x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1],  # back
            [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0],  # bottom
        ],
        dtype=np.float64,
    )
    tris = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [8, 9, 10], [8, 10, 11],
            [12, 13, 14], [12, 14, 15],
            [16, 17, 18], [16, 18, 19],
            [20, 21, 22], [20, 22, 23],
        ],
        dtype=np.int32,
    )
    base = np.array(base_color, dtype=np.float64)
    face_scales = np.array(
        [
            1.00,  # top
            0.98,  # front
            0.92,  # right
            0.95,  # left
            0.94,  # back
            0.90,  # bottom
        ],
        dtype=np.float64,
    )
    face_colors = np.clip(face_scales[:, None] * base[None, :], 0.0, 1.0)
    colors = np.repeat(face_colors, 4, axis=0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def build_semantic_voxel_mesh_with_style(label_grid: torch.Tensor | np.ndarray, style: RendererStyle) -> Any:
    if o3d is None:
        raise RuntimeError("open3d is not available in this Python environment")
    idx, labels = surface_voxel_indices_and_labels(label_grid)
    mesh = o3d.geometry.TriangleMesh()
    if idx.shape[0] == 0:
        return mesh
    scale = float(style.mesh_voxel_scale)
    box_size = np.array(VOXEL_SIZE, dtype=np.float64) * scale
    half = box_size / 2.0
    centers = voxel_to_metric(idx.astype(np.float32)).astype(np.float64)
    origins = centers - half[None, :]

    unit_verts = np.array(
        [
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
            [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1],
            [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1],
            [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1],
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
        ],
        dtype=np.float64,
    )
    unit_tris = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [8, 9, 10], [8, 10, 11],
            [12, 13, 14], [12, 14, 15],
            [16, 17, 18], [16, 18, 19],
            [20, 21, 22], [20, 22, 23],
        ],
        dtype=np.int32,
    )
    face_scales = np.array([1.00, 0.98, 0.92, 0.95, 0.94, 0.90], dtype=np.float64)
    face_scales_verts = np.repeat(face_scales, 4)[:, None]
    base_colors = np.array(
        [
            hex_to_rgb01(PALETTE.get(OCC_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(OCC_NAMES) else "others", "#808080"))
            for cls_id in labels.tolist()
        ],
        dtype=np.float64,
    )
    verts = unit_verts[None, :, :] * box_size[None, None, :] + origins[:, None, :]
    colors = np.clip(base_colors[:, None, :] * face_scales_verts[None, :, :], 0.0, 1.0)
    tri_offsets = (np.arange(origins.shape[0], dtype=np.int32) * unit_verts.shape[0])[:, None, None]
    tris = unit_tris[None, :, :] + tri_offsets

    mesh.vertices = o3d.utility.Vector3dVector(verts.reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(tris.reshape(-1, 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def build_colored_voxel_mesh_with_style(
    label_grid: torch.Tensor | np.ndarray,
    voxel_rgb_grid: torch.Tensor | np.ndarray,
    style: RendererStyle,
) -> Any:
    if o3d is None:
        raise RuntimeError("open3d is not available in this Python environment")
    idx, _ = surface_voxel_indices_and_labels(label_grid)
    mesh = o3d.geometry.TriangleMesh()
    if idx.shape[0] == 0:
        return mesh

    rgb = np.asarray(torch.as_tensor(voxel_rgb_grid).cpu().numpy(), dtype=np.float64)
    grid_shape = tuple(np.asarray(label_grid).shape[:3])
    if rgb.ndim != 4 or tuple(rgb.shape[:3]) != grid_shape or rgb.shape[3] != 3:
        raise ValueError("voxel_rgb_grid must have shape [X, Y, Z, 3] matching label_grid")
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    scale = float(style.mesh_voxel_scale)
    box_size = np.array(VOXEL_SIZE, dtype=np.float64) * scale
    half = box_size / 2.0
    centers = voxel_to_metric(idx.astype(np.float32)).astype(np.float64)
    origins = centers - half[None, :]

    unit_verts = np.array(
        [
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
            [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1],
            [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1],
            [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1],
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
        ],
        dtype=np.float64,
    )
    unit_tris = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [8, 9, 10], [8, 10, 11],
            [12, 13, 14], [12, 14, 15],
            [16, 17, 18], [16, 18, 19],
            [20, 21, 22], [20, 22, 23],
        ],
        dtype=np.int32,
    )
    face_scales = np.array([1.00, 0.98, 0.92, 0.95, 0.94, 0.90], dtype=np.float64)
    face_scales_verts = np.repeat(face_scales, 4)[:, None]
    base_colors = rgb[idx[:, 0], idx[:, 1], idx[:, 2]]
    verts = unit_verts[None, :, :] * box_size[None, None, :] + origins[:, None, :]
    colors = np.clip(base_colors[:, None, :] * face_scales_verts[None, :, :], 0.0, 1.0)
    tri_offsets = (np.arange(origins.shape[0], dtype=np.int32) * unit_verts.shape[0])[:, None, None]
    tris = unit_tris[None, :, :] + tri_offsets

    mesh.vertices = o3d.utility.Vector3dVector(verts.reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(tris.reshape(-1, 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def _get_camera_params(viewpoint: str) -> Any:
    if viewpoint not in _O3D_CAMERA_CACHE:
        view_path = (
            Path(__file__).resolve().parents[4]
            / "external/SparseWorld/tools/visualization/viewpoint_params"
            / f"cam_{viewpoint}.json"
        )
        _O3D_CAMERA_CACHE[viewpoint] = o3d.io.read_pinhole_camera_parameters(str(view_path))
    return _O3D_CAMERA_CACHE[viewpoint]


def _get_persistent_visualizer(style: RendererStyle) -> Any:
    key = (style.render_width, style.render_height, style.viewpoint, style.mode)
    vis = _O3D_VIS_CACHE.get(key)
    if vis is not None:
        return vis
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"swvis_renderer_{style.viewpoint}_{style.mode}",
        width=style.render_width,
        height=style.render_height,
        visible=False,
    )
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1], dtype=np.float64)
    opt.light_on = False if style.mode in {"voxel_mesh", "voxel_shaded"} else False
    opt.mesh_show_back_face = True
    _O3D_VIS_CACHE[key] = vis
    return vis


def render_semantic_occ_open3d_legacy(
    label_grid: torch.Tensor,
    title: str,
    style: RendererStyle,
    legend_classes: list[str] | None = None,
    point_scale: float | None = None,
) -> Image.Image:
    if o3d is None:
        raise RuntimeError("open3d is not available in this Python environment")
    occ = paper_mask_occ(label_grid, top_view=style.viewpoint == "top", visual_ego=False)
    coords, labels = occupancy_points(torch.from_numpy(occ))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    if labels.size > 0:
        colors = np.array([hex_to_rgb01(PALETTE.get(OCC_NAMES[int(x)], "#808080")) for x in labels], dtype=np.float64)
    else:
        colors = np.zeros((0, 3), dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_mesh = build_semantic_voxel_mesh_with_style(occ, style) if style.mode in {"voxel_mesh", "voxel_shaded"} else None

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=style.render_width,
        height=style.render_height,
        visible=False,
    )
    try:
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1], dtype=np.float64)
        opt.light_on = False if style.mode in {"voxel_mesh", "voxel_shaded"} else False
        opt.mesh_show_back_face = True
        if style.mode in {"voxel_mesh", "voxel_shaded"}:
            vis.add_geometry(voxel_mesh)
        elif style.mode in {"voxel_noline", "voxel_cube"}:
            vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE[0])
            vis.add_geometry(vox)
        else:
            opt.point_size = float(point_scale if point_scale is not None else style.open3d_point_size)
            vis.add_geometry(pcd)
        params = _get_camera_params(style.viewpoint)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        out = (img * 255).astype(np.uint8)
    finally:
        vis.destroy_window()
    out_img = Image.fromarray(out)
    if style.mode in {"voxel_mesh", "voxel_shaded"}:
        out_img = apply_soft_drop_shadow(out_img, style)
    return autocrop_white(out_img, pad=2)


def render_semantic_occ_open3d(
    label_grid: torch.Tensor,
    title: str,
    style: RendererStyle,
    legend_classes: list[str] | None = None,
    point_scale: float | None = None,
) -> Image.Image:
    if o3d is None:
        raise RuntimeError("open3d is not available in this Python environment")
    occ = paper_mask_occ(label_grid, top_view=style.viewpoint == "top", visual_ego=False)
    coords, labels = occupancy_points(torch.from_numpy(occ))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    if labels.size > 0:
        colors = np.array([hex_to_rgb01(PALETTE.get(OCC_NAMES[int(x)], "#808080")) for x in labels], dtype=np.float64)
    else:
        colors = np.zeros((0, 3), dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_mesh = build_semantic_voxel_mesh_with_style(occ, style) if style.mode in {"voxel_mesh", "voxel_shaded"} else None

    vis = _get_persistent_visualizer(style)
    vis.clear_geometries()
    if style.mode in {"voxel_mesh", "voxel_shaded"}:
        vis.add_geometry(voxel_mesh, reset_bounding_box=True)
    elif style.mode in {"voxel_noline", "voxel_cube"}:
        vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE[0])
        vis.add_geometry(vox, reset_bounding_box=True)
    else:
        opt = vis.get_render_option()
        opt.point_size = float(point_scale if point_scale is not None else style.open3d_point_size)
        vis.add_geometry(pcd, reset_bounding_box=True)
    params = _get_camera_params(style.viewpoint)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    out_img = Image.fromarray((img * 255).astype(np.uint8))
    if style.mode in {"voxel_mesh", "voxel_shaded"}:
        out_img = apply_soft_drop_shadow(out_img, style)
    return autocrop_white(out_img, pad=2)


def render_semantic_occ(
    label_grid: torch.Tensor,
    title: str,
    style: RendererStyle,
    legend_classes: list[str] | None = None,
    point_scale: float | None = None,
) -> Image.Image:
    if style.viewpoint == "bev_strict":
        occ = paper_mask_occ(label_grid, top_view=True, visual_ego=False)
        occ = np.array(occ, copy=False)
        active = (occ != EMPTY_IDX) & (occ != IGNORE_IDX)
        canvas = np.ones((occ.shape[1], occ.shape[0], 3), dtype=np.uint8) * 255
        if active.any():
            has_any = active.any(axis=2)
            top_idx = np.where(has_any, active.shape[2] - 1 - np.argmax(active[:, :, ::-1], axis=2), -1)
            xs, ys = np.where(has_any)
            zs = top_idx[xs, ys]
            labels = occ[xs, ys, zs]
            max_z = max(1, occ.shape[2] - 1)
            for x, y, cls_id, zz in zip(xs.tolist(), ys.tolist(), labels.tolist(), zs.tolist()):
                cls_name = OCC_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(OCC_NAMES) else "others"
                base = np.array(hex_to_rgb255(PALETTE.get(cls_name, "#808080")), dtype=np.float32)
                shade = 0.90 + 0.12 * (float(zz) / float(max_z)) if style.bev_height_shading else 1.0
                color = np.clip(base * shade, 0, 255).astype(np.uint8)
                canvas[occ.shape[1] - 1 - y, x] = color
        img = Image.fromarray(canvas, mode="RGB")
        return img.resize((style.render_width, style.render_height), Image.Resampling.NEAREST)
    if style.backend == "open3d":
        return render_semantic_occ_open3d(label_grid, title, style, legend_classes=legend_classes, point_scale=point_scale)
    coords, labels = occupancy_points(label_grid)
    fig = plt.figure(figsize=style.figsize, dpi=style.dpi, facecolor=style.background)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(style.camera_facecolor)
    ax.view_init(elev=style.elev, azim=style.azim)
    ax.set_xlim(*style.xlim)
    ax.set_ylim(*style.ylim)
    ax.set_zlim(*style.zlim)
    ax.grid(False)
    ax.set_axis_off()
    if coords.shape[0] > 0:
        size = point_scale if point_scale is not None else style.point_size
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=size,
            c=class_colors(labels),
            marker="o",
            depthshade=False,
            edgecolors="none",
            alpha=0.95,
        )
    ax.set_title(title, fontsize=style.title_fontsize, pad=10)
    if legend_classes:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=PALETTE[name], markersize=6, label=name.replace("_", " "))
            for name in legend_classes
            if name in PALETTE
        ]
        if handles:
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(-0.02, 1.02),
                frameon=False,
                fontsize=style.legend_fontsize,
                ncol=min(4, max(1, len(handles))),
            )
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=style.dpi, facecolor=style.background, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_colored_voxel_grid_open3d(
    label_grid: torch.Tensor | np.ndarray,
    voxel_rgb_grid: torch.Tensor | np.ndarray,
    style: RendererStyle,
) -> Image.Image:
    if o3d is None:
        raise RuntimeError("open3d is not available in this Python environment")
    occ = paper_mask_occ(label_grid, top_view=style.viewpoint == "top", visual_ego=False)
    rgb = np.asarray(torch.as_tensor(voxel_rgb_grid).cpu().numpy())
    if tuple(rgb.shape[:3]) != tuple(occ.shape[:3]):
        raise ValueError("voxel_rgb_grid spatial shape must match label_grid")
    voxel_mesh = build_colored_voxel_mesh_with_style(occ, rgb, style)
    vis = _get_persistent_visualizer(style)
    vis.clear_geometries()
    vis.add_geometry(voxel_mesh, reset_bounding_box=True)
    params = _get_camera_params(style.viewpoint)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    out_img = Image.fromarray((img * 255).astype(np.uint8))
    out_img = apply_soft_drop_shadow(out_img, style)
    return autocrop_white(out_img, pad=2)


def build_legend_strip(legend_classes: list[str], width: int, height: int = 84) -> Image.Image:
    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("times.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    x = 18
    y = 18
    sw = 34
    gap = 24
    for name in legend_classes:
        if name not in PALETTE:
            continue
        draw.rectangle((x, y, x + sw, y + sw), fill=PALETTE[name], outline=None)
        text = name.replace("_", " ")
        draw.text((x + sw + 12, y - 2), text, fill="black", font=font)
        text_bbox = draw.textbbox((x + sw + 12, y - 2), text, font=font)
        x = text_bbox[2] + gap
    return canvas


def make_rollout_grid(
    gt_panels: list[Image.Image],
    pred_panels: list[Image.Image],
    horizon_titles: list[str],
    sample_label: str,
    out_png: Path,
    out_pdf: Path | None = None,
) -> None:
    assert len(gt_panels) == len(pred_panels) == len(horizon_titles)
    target_panel = (860, 360)
    gt_panels = [fit_panel(p, target_panel) for p in gt_panels]
    pred_panels = [fit_panel(p, target_panel) for p in pred_panels]
    w, h = gt_panels[0].size
    margin = 24
    top = 128
    left = 210
    legend_classes = ["car", "truck", "construction_vehicle", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    canvas = Image.new("RGB", (left + len(horizon_titles) * (w + margin), top + 2 * (h + 18) + 26), "white")
    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont

    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("times.ttf", 42)
        font_label = ImageFont.truetype("times.ttf", 30)
        font_small = ImageFont.truetype("times.ttf", 30)
        font_sample = ImageFont.truetype("times.ttf", 16)
    except Exception:
        font_title = font_label = font_small = font_sample = ImageFont.load_default()
    legend = build_legend_strip(legend_classes, width=canvas.width - left, height=72)
    canvas.paste(legend, (left, 6))
    draw.text((16, 8), sample_label, fill="#555555", font=font_sample)
    row_y = [top, top + h + 18]
    for j, title in enumerate(horizon_titles):
        x = left + j * (w + margin)
        draw.text((x + w // 2 - 46, 76), title, fill="black", font=font_title)
        canvas.paste(gt_panels[j], (x, row_y[0]))
        canvas.paste(pred_panels[j], (x, row_y[1]))
    draw.text((16, row_y[0] + h // 2 - 18), "Ground Truth", fill="black", font=font_small)
    draw.text((16, row_y[1] + h // 2 - 18), "SparseWorld", fill="black", font=font_small)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    if out_pdf is not None:
        canvas.save(out_pdf, "PDF", resolution=300.0)


def error_overlay_bev(pred: torch.Tensor, gt: torch.Tensor) -> Image.Image:
    pred_occ = (torch.as_tensor(pred).cpu() != EMPTY_IDX).any(dim=-1).numpy()
    gt_occ = (torch.as_tensor(gt).cpu() != EMPTY_IDX).any(dim=-1).numpy()
    ff = gt_occ & (~pred_occ)
    fo = (~gt_occ) & pred_occ
    tp = gt_occ & pred_occ
    h, w = gt_occ.shape
    canvas = np.ones((h, w, 3), dtype=np.float32)
    canvas[tp] = np.array([0.82, 0.86, 0.82], dtype=np.float32)
    canvas[ff] = np.array([0.95, 0.45, 0.45], dtype=np.float32)
    canvas[fo] = np.array([0.45, 0.60, 0.95], dtype=np.float32)
    img = Image.fromarray((canvas * 255).astype(np.uint8)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img.resize((900, 900), Image.Resampling.BILINEAR)


def save_gif(frames: list[Image.Image], out_gif: Path, duration_ms: int = 850) -> None:
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_mp4_stub(frames: list[Image.Image], out_mp4: Path) -> bool:
    try:
        import imageio.v2 as imageio

        arrs = [np.asarray(f.convert("RGB")) for f in frames]
        imageio.mimsave(out_mp4, arrs, fps=1)
        if out_mp4.exists() and out_mp4.stat().st_size > 1024:
            return True
        if out_mp4.exists():
            out_mp4.unlink()
        return False
    except Exception:
        if out_mp4.exists():
            out_mp4.unlink()
        return False
