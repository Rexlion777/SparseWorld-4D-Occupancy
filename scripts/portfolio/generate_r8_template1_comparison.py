#!/usr/bin/env python3
"""Build aligned Template 1 R0-before / R8-after videos from frozen anchors."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


HORIZONS = (0, 2, 4, 6)
CANVAS = (1280, 1180)
BG = "#F8FAFC"
INK = "#13213A"
MUTED = "#64748B"
RED = "#DC4C4C"
BLUE = "#2563EB"
TEAL = "#0F8278"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=6.0)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def interpolate(anchors: list[Image.Image], t: float) -> Image.Image:
    if t <= HORIZONS[0]:
        return anchors[0].copy()
    if t >= HORIZONS[-1]:
        return anchors[-1].copy()
    for index in range(len(HORIZONS) - 1):
        left, right = HORIZONS[index], HORIZONS[index + 1]
        if left <= t <= right:
            alpha = smoothstep((t - left) / (right - left))
            return Image.blend(anchors[index], anchors[index + 1], alpha)
    return anchors[-1].copy()


def fit(image: Image.Image, size: tuple[int, int], background: str = "white") -> Image.Image:
    panel = Image.new("RGB", size, background)
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def draw_timeline(draw: ImageDraw.ImageDraw, t: float, accent: str) -> None:
    left, right, y = 70, 1210, 1134
    draw.line((left, y, right, y), fill="#CBD5E1", width=6)
    progress = left + int((right - left) * t / 6.0)
    draw.line((left, y, progress, y), fill=accent, width=7)
    for horizon in HORIZONS:
        x = left + int((right - left) * horizon / 6.0)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=accent if horizon <= t else "#CBD5E1")
        draw.text((x, y + 14), f"{horizon} s", anchor="ma", fill=MUTED, font=font(20, True))


def compose(
    camera: Image.Image,
    observation: Image.Image,
    future: Image.Image,
    t: float,
    repaired: bool,
) -> Image.Image:
    accent = TEAL if repaired else RED
    status = "R8 · CAUSAL t−1 FEATURE REPAIR" if repaired else "R0 · DEGRADED NATIVE"
    headline = "AFTER R8 REPAIR" if repaired else "BEFORE R8 REPAIR"
    canvas = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "A10 FRONT-TRIPLET CAMERA FAILURE", fill=MUTED, font=font(23, True))
    draw.text((40, 62), headline, fill=INK, font=font(44, True))
    badge_box = (875, 46, 1240, 104)
    draw.rounded_rectangle(badge_box, radius=18, fill=accent)
    draw.text((1057, 75), status, anchor="mm", fill="white", font=font(18, True))

    camera_panel = fit(camera, (1200, 435))
    canvas.paste(camera_panel, (40, 128))
    draw.text((40, 580), "Current occupancy", fill=INK, font=font(25, True))
    draw.text((430, 580), f"Predicted future · t = {t:0.1f} s", fill=INK, font=font(25, True))

    observation_panel = fit(observation, (350, 475))
    future_panel = fit(future, (810, 475))
    canvas.paste(observation_panel, (40, 620))
    canvas.paste(future_panel, (430, 620))
    draw.rounded_rectangle((40, 620, 390, 1095), radius=10, outline="#CBD5E1", width=2)
    draw.rounded_rectangle((430, 620, 1240, 1095), radius=10, outline=accent, width=4)
    draw_timeline(draw, t, accent)
    return canvas


def encode(frames: Path, output: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the comparison videos")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264",
            "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        check=True,
    )


def encode_gif(video: Path, output: Path) -> None:
    filter_graph = (
        "fps=10,scale=900:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=160:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", filter_graph, "-loop", "0", str(output)],
        check=True,
    )


def build_variant(anchors_dir: Path, output_dir: Path, repaired: bool, fps: int, duration: float) -> tuple[Path, Path]:
    prefix = "r8" if repaired else "native"
    camera = Image.open(anchors_dir / "a10_front_triplet_camera_panel.png").convert("RGB")
    anchors = [Image.open(anchors_dir / f"{prefix}_h{horizon}.png").convert("RGB") for horizon in HORIZONS]
    observation = anchors[0]
    frame_count = int(round(fps * duration))
    with tempfile.TemporaryDirectory(prefix=f"template1_{prefix}_") as tmp:
        frames = Path(tmp)
        for index in range(frame_count):
            t = 6.0 * index / max(1, frame_count - 1)
            compose(camera, observation, interpolate(anchors, t), t, repaired).save(frames / f"frame_{index:04d}.png")
        mp4 = output_dir / f"template1_a10_{prefix}_{'after' if repaired else 'before'}_r8.mp4"
        encode(frames, mp4, fps)
    gif = output_dir / f"template1_a10_{prefix}_{'after' if repaired else 'before'}_r8.gif"
    encode_gif(mp4, gif)
    return mp4, gif


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for repaired in (False, True):
        mp4, gif = build_variant(args.anchors, args.output_dir, repaired, args.fps, args.duration)
        print(mp4)
        print(gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
