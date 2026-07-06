# Contributing

## Local setup

```bash
python -m pip install -e '.[dev]'
```

## Required checks

Run all gates before handing a change to another developer:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q
git diff --check
bash -n scripts/*.sh
```

When both selected checkpoints are available, also run:

```bash
PYTHONPATH=src python scripts/check_checkpoint.py
scripts/build.sh
scripts/test_container.sh
```

## Change rules

1. Do not add training code to this repository.
2. Do not change model field names or module layout without a real checkpoint
   compatibility test.
3. Keep preprocessing and coordinate restoration covered by focused tests.
4. Any output schema change needs serializer and validator tests.
5. Runtime must remain network-independent.
6. Update `THIRD_PARTY.md` and the recorded source commit when vendoring a new
   Prometheus revision.
7. Never commit checkpoints, generated outputs, Docker archives or secrets.

Use small commits organized by contract: model vendor update, submission
adapter, packaging, or documentation. Avoid mixing architecture experiments
with deployment fixes.
