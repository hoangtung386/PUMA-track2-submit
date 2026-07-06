# Engineering handoff status

## Source status

The repository now contains one deterministic PrometheusNet inference path and
no legacy HoVer-NeXt or nnU-Net runtime.

Verified gates at handoff:

- 28 focused tests pass;
- Ruff lint and format checks pass;
- shell scripts pass `bash -n`;
- `git diff --check` passes;
- the Python project builds as a wheel;
- deployment config loads through a strict schema;
- missing checkpoints fail before Docker build or inference.

## Release blockers

Source code is ready for team ownership. A production submission artifact is
not ready until both private model artifacts are supplied:

```text
models/best_primary.ckpt
models/best_tissue.ckpt
```

The receiving team must then complete these release gates:

1. Run `scripts/check_checkpoint.py` and record both checkpoint SHA-256 values
   plus EMA/raw selection in `models/README.md`.
2. Confirm that both checkpoints' embedded model configs exactly match
   `configs/submission.toml`.
3. Confirm the official evaluator input directory and JSON schema against the
   challenge version being submitted.
4. Run one golden sample outside Docker and one inside Docker with networking
   disabled; compare output summaries.
5. Measure worst-case time, RAM and VRAM, and verify that `max_detections = 1000`
   does not truncate valid nuclei.
6. Export the Docker image, load the exported archive on a clean host and repeat
   the smoke test.

These are artifact/evaluator validations, not unresolved source refactors.
