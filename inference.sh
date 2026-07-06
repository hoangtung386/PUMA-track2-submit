#!/usr/bin/env bash
#
# PUMA Track 2 — Prometheus container inference entry point.
#
# Supersedes the legacy hovernext + nnU-Net baseline pipeline. A single
# Prometheus run reads the one primary ROI TIFF from the challenge input
# directory, decodes the 10-class nuclei detections from best_primary.ckpt
# (fold 2) and the 6-class tissue mask from best_tissue.ckpt (fold 1), then
# writes both challenge outputs and validates them.
#
# Input/output locations and the two checkpoints are resolved from the
# environment (see src/puma_submission/contract.py); the Dockerfile sets the
# defaults used inside the container.
set -euo pipefail

exec python -m puma_submission.run "$@"
