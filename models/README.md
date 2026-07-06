# Prometheus checkpoint

Place the two selected k-fold checkpoints at:

```text
models/best_primary.ckpt  # fold 2, selected for nuclei F1
models/best_tissue.ckpt   # fold 1, selected for tissue Dice
```

Both files are intentionally excluded from Git. Each is a complete multitask
PrometheusNet checkpoint; the runtime executes them sequentially and selects
nuclei predictions from `best_primary.ckpt` and the tissue mask from
`best_tissue.ckpt`.

## Selected artifacts

| Runtime task | Training source | Epoch | Selection metric | SHA-256 |
|---|---|---:|---:|---|
| Nuclei | `fold_2/best_primary.ckpt` | 99 | nuclei F1 `0.359474` | `04db493c0ed343d8b3ecab50ecf1e4792ae9de8884b49170fcef1fd34be7af62` |
| Tissue | `fold_1/best_tissue.ckpt` | 99 | tissue Dice `0.655757` | `1ef5b8e5c72070459c10d18d1502851b3b9ac5bffdbfa21ef3edb2b2b00fcead` |

Both artifacts contain EMA state and use the same architecture configuration,
including `drop_path_rate = 0.2`.

Both checkpoints must use Prometheus checkpoint schema 2 and architecture
`prometheus_multitask_v1`. The build script validates both against
`configs/submission.toml` with strict state-dict loading.

Record the following before release:

- source run/fold and selected epoch;
- validation metrics;
- Prometheus source commit;
- whether inference selects EMA or raw weights;
- byte size and SHA-256 for both files
  (`sha256sum models/best_primary.ckpt models/best_tissue.ckpt`).

Vendored Prometheus source commit: `a35cf22b472fb473c8a127394491f7b5a409414d`.

The repository is ready for source handoff without this private artifact, but
it is not release-ready until the real checkpoint passes the compatibility
script and an offline container smoke test.
