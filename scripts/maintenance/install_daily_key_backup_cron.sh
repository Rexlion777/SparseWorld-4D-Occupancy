#!/usr/bin/env bash
set -euo pipefail

BACKUP_SCRIPT="${BACKUP_SCRIPT:-/home/rexlion/ComputerVision/cv_lidar_transition/scripts/maintenance/backup_key_materials_to_e.sh}"
LOG_FILE="${LOG_FILE:-/home/rexlion/ComputerVision/cv_lidar_transition/logs/backup_key_materials_cron.log}"
CRON_TIME="${CRON_TIME:-30 2 * * *}"

if [[ ! -x "$BACKUP_SCRIPT" ]]; then
  echo "Backup script is not executable: $BACKUP_SCRIPT" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
line="$CRON_TIME $BACKUP_SCRIPT --daily-only >> $LOG_FILE 2>&1"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

crontab -l 2>/dev/null | grep -vF "$BACKUP_SCRIPT --daily-only" > "$tmp" || true
printf '%s\n' "$line" >> "$tmp"
crontab "$tmp"

echo "Installed daily key backup cron:"
echo "$line"

