# Prometheus checkpoint

Place the final trained checkpoint at `models/prometheus.ckpt` before building.
The file is intentionally excluded from Git.

The checkpoint must use Prometheus checkpoint schema 2 and architecture
`prometheus_multitask_v1`. The build script validates it against
`configs/submission.toml` with strict state-dict loading.

Record the following before release:

- source run/fold and selected epoch;
- validation metrics;
- Prometheus source commit;
- whether inference selects EMA or raw weights;
- byte size and SHA-256 (`sha256sum models/prometheus.ckpt`).

Vendored Prometheus source commit: `a35cf22b472fb473c8a127394491f7b5a409414d`.

The repository is ready for source handoff without this private artifact, but
it is not release-ready until the real checkpoint passes the compatibility
script and an offline container smoke test.
