#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly IMAGE_TAG="${1:-puma-prometheus-track2:latest}"

cd "${PROJECT_DIR}"
PYTHONPATH=src python scripts/check_checkpoint.py
docker build --platform linux/amd64 --tag "${IMAGE_TAG}" .
