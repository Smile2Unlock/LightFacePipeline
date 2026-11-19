#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_TARGET="${1:-scnet-hcu:/workspace/LightFacePipeline/repo/}"

rsync -avz --delete \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='datasets/' \
  --exclude='checkpoints/' \
  --exclude='runs/' \
  --exclude='cache/' \
  --exclude='artifacts/models/' \
  --exclude='*.onnx' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.zip' \
  "${REPO_ROOT}/" "${REMOTE_TARGET}"
