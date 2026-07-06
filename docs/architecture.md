# Deployment architecture

This repository is a deployment boundary, not a second training framework.
Training remains owned by the Prometheus repository. Only inference contracts
required to reconstruct and execute the trained model are vendored here.

```text
challenge filesystem
        |
        v
puma_submission.contract  -- deterministic input discovery and output names
        |
        v
puma_submission.prepare   -- TIFF read, letterbox, valid-pixel normalization
        |
        v
prometheus.api             -- strict config/checkpoint/model composition
        |
        v
PrometheusNet              -- tissue and nuclei heads in one forward pass
        |
        +--> tissue mask restoration --> TIFF serializer
        |
        +--> nuclei decoder -----------> Track 2 JSON serializer
                                              |
                                              v
                                 structural and bounds validation
```

## Ownership boundaries

| Package | Responsibility |
|---|---|
| `puma_submission` | Challenge paths, orchestration and final validation |
| `prometheus.config` | Strict deployment-only TOML schema |
| `prometheus.checkpoint` | Read-only schema-v2 loading and compatibility |
| `prometheus.models` | Checkpoint-compatible PrometheusNet architecture |
| `prometheus.inference` | Preprocessing, spatial restoration and decoding |
| `prometheus.io` | Native input and challenge output serialization |
| `prometheus.domain` | Labels, metadata and detection contracts |

## Invariants

- The checkpoint model configuration must equal `configs/submission.toml`
  exactly. State dictionaries load with `strict=True`.
- EMA weights are preferred when present, matching Prometheus inference.
- Input is converted to RGB float32, letterboxed without distortion and
  normalized only over non-padding pixels.
- Tissue predictions are restored with nearest-neighbor interpolation and
  explicitly remapped from model order to submission values.
- Nuclei coordinates and boxes are restored to source `(x, y)` pixel space.
- The container returns success only after both outputs pass validation.

## Deliberately excluded

Training datasets, augmentation, losses, metrics, trainers, visualization,
notebooks, pretrained-weight download and experimental YOLO integration do not
belong to this runtime. Changes to those components must be made and validated
in Prometheus, followed by an intentional vendor update here.
