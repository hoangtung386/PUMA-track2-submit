#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly IMAGE_TAG="${1:-puma-prometheus-track2:latest}"
readonly SAFE_TAG="${IMAGE_TAG//[\/:]/-}"
readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly ARCHIVE="${PROJECT_DIR}/${SAFE_TAG}-${TIMESTAMP}.tar.gz"

docker image inspect "${IMAGE_TAG}" >/dev/null
docker save "${IMAGE_TAG}" | gzip -c >"${ARCHIVE}"
sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"
echo "Saved ${ARCHIVE}"
