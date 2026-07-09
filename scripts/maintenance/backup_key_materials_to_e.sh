#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/home/rexlion/ComputerVision}"
PROJECT_ROOT="${PROJECT_ROOT:-$SRC_ROOT/cv_lidar_transition}"
DEST_ROOT="${DEST_ROOT:-/mnt/e/ComputerVision_KeyBackup}"

TEXT_MAX_BYTES="${TEXT_MAX_BYTES:-10485760}"     # 10 MiB
IMAGE_MAX_BYTES="${IMAGE_MAX_BYTES:-1048576}"    # 1 MiB
VIDEO_MAX_BYTES="${VIDEO_MAX_BYTES:-209715200}"  # 200 MiB safety cap for finished videos

RUN_DAILY=1
RUN_ENV=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  backup_key_materials_to_e.sh [--daily-only|--env-only|--all] [--dry-run]

Defaults:
  --daily-only behavior: source + report text <=10MiB + figures <=1MiB + finished videos.
  The cu128 environment is copied only with --env-only or --all.

Environment overrides:
  SRC_ROOT=/home/rexlion/ComputerVision
  PROJECT_ROOT=$SRC_ROOT/cv_lidar_transition
  DEST_ROOT=/mnt/e/ComputerVision_KeyBackup
  TEXT_MAX_BYTES=10485760
  IMAGE_MAX_BYTES=1048576
  VIDEO_MAX_BYTES=209715200
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --daily-only)
      RUN_DAILY=1
      RUN_ENV=0
      ;;
    --env-only)
      RUN_DAILY=0
      RUN_ENV=1
      ;;
    --all)
      RUN_DAILY=1
      RUN_ENV=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing directory: $1" >&2
    exit 1
  fi
}

require_dir "$SRC_ROOT"
require_dir "$PROJECT_ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"
manifest_dir="$DEST_ROOT/manifests"
daily_dest="$DEST_ROOT/daily/cv_lidar_transition"
env_dest="$DEST_ROOT/envs/sparseworld_cu128"
mkdir -p "$manifest_dir"

rsync_flags=(-a --human-readable --info=stats2)
if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_flags=(-a -n --human-readable --info=stats2)
fi

append_rel_files() {
  local list_file="$1"
  shift
  local roots=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    roots+=("$1")
    shift
  done
  shift
  local find_args=("$@")

  local root
  for root in "${roots[@]}"; do
    [[ -e "$root" ]] || continue
    while IFS= read -r -d '' path; do
      path="${path#"$PROJECT_ROOT"/}"
      printf '%s\0' "$path" >> "$list_file"
    done < <(find "$root" "${find_args[@]}" -print0)
  done
}

make_daily_file_list() {
  local raw_list="$1"
  local deduped_list="$2"
  : > "$raw_list"

  local source_roots=()
  local name
  for name in configs notes perception_pipeline runtime scripts tests utils day01_yolo_demo day02_opencv_basics day03_open3d_lidar day04_lidar_clustering day05_camera_lidar_projection day06_2d_3d_fusion day08_mini_pointpillars day08_pointpillars day09_openpcdet day10_openpcdet_analysis day11_real_kitti_dataset day12_real_kitti_openpcdet day13_lidar_degradation_robustness day14_centerpoint_comparison day15_openpcdet_kitti_model_comparison; do
    source_roots+=("$PROJECT_ROOT/$name")
  done

  append_rel_files "$raw_list" "${source_roots[@]}" -- \
    -type f \
    ! -path '*/__pycache__/*' \
    ! -path '*/.pytest_cache/*' \
    ! -path '*/.ipynb_checkpoints/*' \
    \( -iname '*.py' -o -iname '*.sh' -o -iname '*.yaml' -o -iname '*.yml' -o -iname '*.toml' -o -iname '*.cfg' -o -iname '*.ini' -o -iname '*.txt' -o -iname '*.md' -o -iname '*.json' -o -iname '*.ipynb' \) \
    -size -"${TEXT_MAX_BYTES}c"

  append_rel_files "$raw_list" "$PROJECT_ROOT" -- \
    -maxdepth 1 -type f \
    \( -iname 'README*' -o -iname 'requirements*.txt' -o -iname '*.py' -o -iname '*.md' -o -iname '*.txt' -o -iname '*.json' -o -iname '*.yaml' -o -iname '*.yml' \) \
    -size -"${TEXT_MAX_BYTES}c"

  append_rel_files "$raw_list" "$PROJECT_ROOT/reports" "$PROJECT_ROOT/projects" -- \
    -type f \
    \( -iname '*.md' -o -iname '*.json' -o -iname '*.csv' -o -iname '*.txt' -o -iname '*.html' \) \
    -size -"${TEXT_MAX_BYTES}c"

  append_rel_files "$raw_list" "$PROJECT_ROOT/reports" "$PROJECT_ROOT/projects" -- \
    -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.svg' \) \
    -size -"${IMAGE_MAX_BYTES}c"

  append_rel_files "$raw_list" "$PROJECT_ROOT/videos" -- \
    -type f \
    \( -iname '*.mp4' -o -iname '*.webm' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' \) \
    -size -"${VIDEO_MAX_BYTES}c"

  sort -zu "$raw_list" > "$deduped_list"
}

summarize_list() {
  local list_file="$1"
  python3 - "$PROJECT_ROOT" "$list_file" <<'PY'
import os
import sys

root, list_path = sys.argv[1], sys.argv[2]
total = 0
count = 0
by_top = {}
with open(list_path, "rb") as f:
    for item in f.read().split(b"\0"):
        if not item:
            continue
        rel = item.decode("utf-8", "surrogateescape")
        path = os.path.join(root, rel)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        total += size
        count += 1
        top = rel.split("/", 1)[0]
        by_top[top] = by_top.get(top, 0) + size

def fmt(n):
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

print(f"selected_files={count}")
print(f"selected_size={fmt(total)}")
for key, value in sorted(by_top.items(), key=lambda kv: kv[1], reverse=True)[:20]:
    print(f"topdir\t{key}\t{fmt(value)}")
PY
}

if [[ "$RUN_DAILY" -eq 1 ]]; then
  require_dir "$PROJECT_ROOT"
  mkdir -p "$daily_dest" "$manifest_dir"
  raw_list="$(mktemp)"
  daily_list="$(mktemp)"
  trap 'rm -f "$raw_list" "$daily_list"' EXIT

  make_daily_file_list "$raw_list" "$daily_list"
  manifest="$manifest_dir/daily_key_files_${timestamp}.txt"
  tr '\0' '\n' < "$daily_list" > "$manifest"

  echo "Daily key-material selection:"
  summarize_list "$daily_list"
  echo "Manifest: $manifest"
  echo "Destination: $daily_dest"
  rsync "${rsync_flags[@]}" --from0 --files-from="$daily_list" "$PROJECT_ROOT/" "$daily_dest/"
fi

if [[ "$RUN_ENV" -eq 1 ]]; then
  require_dir "$SRC_ROOT/envs/sparseworld_cu128"
  mkdir -p "$env_dest" "$manifest_dir"
  env_manifest="$manifest_dir/cu128_env_backup_${timestamp}.txt"
  {
    echo "source=$SRC_ROOT/envs/sparseworld_cu128"
    echo "destination=$env_dest"
    echo "timestamp=$timestamp"
    du -sh "$SRC_ROOT/envs/sparseworld_cu128" 2>/dev/null || true
  } > "$env_manifest"

  echo "cu128 raw environment backup:"
  cat "$env_manifest"
  rsync "${rsync_flags[@]}" "$SRC_ROOT/envs/sparseworld_cu128/" "$env_dest/"
fi

