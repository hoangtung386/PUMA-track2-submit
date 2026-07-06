# PUMA Track 2 — Prometheus submission

Dockerized inference for the PUMA melanoma challenge using the multitask
`PrometheusNet`. A single model produces the six-class tissue mask and ten-class
nuclei detections.

The model architecture and checkpoint contract are vendored from Prometheus
commit `a35cf22b472fb473c8a127394491f7b5a409414d`. See `THIRD_PARTY.md` for source
provenance.

The maintained runtime architecture and ownership boundaries are documented in
[`docs/architecture.md`](docs/architecture.md). Development rules and required
quality gates are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Current handoff status and the remaining release-only gates are recorded in
[`docs/handoff.md`](docs/handoff.md).

## Required checkpoint

Copy the selected schema-v2 checkpoint to:

```text
models/prometheus.ckpt
```

The weight is intentionally not committed. Before building, verify that
`configs/submission.toml` exactly matches the model config stored in the
checkpoint:

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

## Build and test

```bash
scripts/build.sh puma-prometheus-track2:latest
scripts/test_container.sh puma-prometheus-track2:latest ./test ./output
```

The smoke test runs with networking disabled and mounts the input read-only.
CUDA 11.8 is used by the pinned PyTorch 2.4.1 runtime image.

Python dependencies are locked in `uv.lock`; direct runtime requirements are
also pinned in `pyproject.toml` and `requirements.txt` for reviewability.

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

## Export

```bash
scripts/save.sh puma-prometheus-track2:latest
```

This creates a timestamped `.tar.gz` and matching SHA-256 file. Always test
`docker load` and one offline forward pass from the exported artifact before
submission.
