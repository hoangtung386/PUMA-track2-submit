# Third-party software

The inference-relevant modules under `src/prometheus` are derived from the
Prometheus research repository at commit
`a35cf22b472fb473c8a127394491f7b5a409414d`. Training-only modules were removed
after checkpoint, model, predictor, spatial and serialization contracts were
ported and tested.

Its MIT license is preserved in `THIRD_PARTY_PROMETHEUS_LICENSE`.

Runtime Python dependencies and their licenses are distributed through the
container's Python environment. See `pyproject.toml` for the dependency list.
