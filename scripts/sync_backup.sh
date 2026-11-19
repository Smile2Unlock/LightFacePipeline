#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-/workspace/LightFacePipeline/repo}"
BACKUP_ROOT="${2:-/public/home/ation_ciger/LightFacePipeline}"

mkdir -p "${BACKUP_ROOT}/checkpoints" "${BACKUP_ROOT}/artifacts" "${BACKUP_ROOT}/reports" "${BACKUP_ROOT}/splits"

for sub in artifacts reports splits; do
  if [ -d "${SRC_ROOT}/${sub}" ]; then
    rsync -av "${SRC_ROOT}/${sub}/" "${BACKUP_ROOT}/${sub}/"
  fi
done

if [ -d "/workspace/LightFacePipeline/runs" ]; then
  find /workspace/LightFacePipeline/runs -maxdepth 3 -name "*best*.pt" -o -name "*epoch*.pt" | while read -r ckpt; do
    rel="${ckpt#/workspace/LightFacePipeline/runs/}"
    mkdir -p "${BACKUP_ROOT}/checkpoints/$(dirname "${rel}")"
    cp -u "${ckpt}" "${BACKUP_ROOT}/checkpoints/${rel}"
  done
fi

echo "=== Persistent Backup Usage (/public/home/ation_ciger) ==="
du -sh "${BACKUP_ROOT}" "${BACKUP_ROOT}"/* 2>/dev/null || true
du -sh /public/home/ation_ciger 2>/dev/null || true
