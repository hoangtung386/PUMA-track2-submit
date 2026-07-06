#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly IMAGE_TAG="${1:-puma-prometheus-track2:latest}"
readonly INPUT_DIR="${2:-${PROJECT_DIR}/test}"
readonly OUTPUT_DIR="${3:-${PROJECT_DIR}/output}"

mapfile -t INPUTS < <(find "${INPUT_DIR}" -maxdepth 1 -type f -iname '*.tif' ! -iname '*_context.tif' | sort)
if [[ "${#INPUTS[@]}" -ne 1 ]]; then
    echo "Expected exactly one primary TIFF in ${INPUT_DIR}; found ${#INPUTS[@]}" >&2
    exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

docker run --rm \
    --platform linux/amd64 \
    --network none \
    --gpus all \
    --shm-size 1g \
    --mount "type=bind,src=${INPUT_DIR},dst=/input/images/melanoma-wsi,readonly" \
    --mount "type=bind,src=${OUTPUT_DIR},dst=/output" \
    "${IMAGE_TAG}"

PYTHONPATH="${PROJECT_DIR}/src" python -m puma_submission.validate_cli \
    --input "${INPUTS[0]}" \
    --output "${OUTPUT_DIR}"
