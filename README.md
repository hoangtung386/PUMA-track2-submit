# PUMA Track 2 — Prometheus submission

Dockerized inference for the PUMA melanoma challenge using the multitask
`PrometheusNet`. Two task-specific k-fold checkpoints are evaluated
sequentially: the ten-class nuclei detections come from the nuclei-selected fold
(`best_primary.ckpt`) and the six-class tissue mask from the tissue-selected
fold (`best_tissue.ckpt`).

The model architecture and checkpoint contract are vendored from Prometheus
commit `a35cf22b472fb473c8a127394491f7b5a409414d`. See `THIRD_PARTY.md` for source
provenance.

The maintained runtime architecture and ownership boundaries are documented in
[`docs/architecture.md`](docs/architecture.md). Development rules and required
quality gates are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Current handoff status and the remaining release-only gates are recorded in
[`docs/handoff.md`](docs/handoff.md).

## Project status

The source runtime is ready for engineering handoff. A production submission
image is not release-ready until both selected checkpoints pass compatibility
checks and the offline container smoke test.

## Repository layout

```text
configs/submission.toml   Frozen model, input, and post-processing settings
models/                   Private checkpoint location and artifact instructions
scripts/                  Checkpoint, build, smoke-test, and export commands
src/prometheus/           Vendored deployment-only Prometheus runtime
src/puma_submission/      PUMA filesystem adapter and end-to-end entry point
tests/                    Runtime contract and regression tests
```

## Prerequisites

Local source checks require Python 3.10 or newer. Building and exercising the
submission container additionally requires:

- Docker with `linux/amd64` build support;
- an NVIDIA GPU and compatible host driver;
- NVIDIA Container Toolkit configured for `docker run --gpus all`.

The image is based on PyTorch 2.4.1, CUDA 11.8, and cuDNN 9. Container runtime
inference can fall back to CPU, but the provided smoke-test script deliberately
requires a GPU to match the intended evaluation environment.

## Required checkpoint

Copy the two selected schema-v2 checkpoints to:

```text
models/best_primary.ckpt  # fold 2, nuclei selection
models/best_tissue.ckpt   # fold 1, tissue selection
```

The weights are intentionally not committed. Each file contains a complete
multitask model, but runtime evaluation is task-specific: nuclei are decoded
from `best_primary.ckpt`, then that model is released before tissue is generated
from `best_tissue.ckpt`. Before building, verify that `configs/submission.toml`
exactly matches the model config stored in both checkpoints:

```bash
PYTHONPATH=src python scripts/check_checkpoint.py
```

## Development checks

```bash
python -m pip install -e '.[dev]'
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q
```

`uv.lock` records the resolved development dependency graph. Docker currently
installs the exact direct runtime versions declared by `pyproject.toml`; it does
not invoke `uv sync` during image construction.

## Build and test

```bash
scripts/build.sh puma-prometheus-track2:latest
scripts/test_container.sh puma-prometheus-track2:latest ./test ./output
```

`scripts/build.sh` runs the strict checkpoint gate before invoking Docker. The
smoke test expects exactly one primary `.tif` file in the supplied input
directory, clears the selected output directory, mounts the input read-only,
runs without network access, and validates the generated files again on the
host.

## Container contract

Primary input path:

```text
/input/images/melanoma-wsi/<uuid>.tif
```

The legacy directory `/input/images/melanoma-whole-slide-image` is accepted as a
fallback. Exactly one primary TIFF must exist. Files ending in `_context.tif`
are ignored.

Outputs:

```text
/output/melanoma-10-class-nuclei-segmentation.json
/output/images/melanoma-tissue-mask-segmentation/<uuid>.tif
```

Every run validates tissue dimensions, labels and TIFF tags plus Track 2 nuclei
classes, scores and coordinate bounds before returning success.

### Runtime path overrides

The container uses challenge-compatible defaults, which can be overridden for
local diagnostics:

| Variable | Default |
|---|---|
| `PUMA_INPUT_DIR` | auto-detect the two supported challenge directories |
| `PUMA_OUTPUT_DIR` | `/output` |
| `PUMA_CONFIG` | `/opt/app/configs/submission.toml` |
| `PUMA_NUCLEI_CHECKPOINT` | `/opt/app/models/best_primary.ckpt` |
| `PUMA_TISSUE_CHECKPOINT` | `/opt/app/models/best_tissue.ckpt` |

When `PUMA_INPUT_DIR` is set, only that directory is searched. Without an
override, finding primary TIFFs in both supported challenge directories is an
error rather than an ambiguous selection.

## Export

```bash
scripts/save.sh puma-prometheus-track2:latest
```

This creates a timestamped `.tar.gz` and matching SHA-256 file. Always test
`docker load` and one offline forward pass from the exported artifact before
submission.

## Recommended release workflow

```bash
# 1. Validate source quality.
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q

# 2. Validate the private checkpoint and record its SHA-256.
PYTHONPATH=src python scripts/check_checkpoint.py

# 3. Build and run an offline GPU smoke test.
scripts/build.sh puma-prometheus-track2:latest
scripts/test_container.sh puma-prometheus-track2:latest ./test ./output

# 4. Export the tested image and checksum.
scripts/save.sh puma-prometheus-track2:latest
```
