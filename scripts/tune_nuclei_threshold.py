"""Tune the nuclei class-confidence threshold against a validation fold.

The Grand Challenge evaluator forbids a per-detection ``score`` (it treats every
emitted detection as equally confident), so the only precision lever left is the
decoder's class-confidence threshold. This script reproduces the official PUMA
Track 2 nuclei metric exactly and sweeps that threshold to find the value(s) that
maximise macro-F1 on a held-out fold — WITHOUT retraining.

Metric (verified against a real submission's metrics.json):
- per-image, class-aware greedy matching (nearest first) within a 15 px radius;
- per-image per-class F1 = 2*TP / (2*TP + FP + FN), zero when the denominator is 0;
- per-class aggregate = mean of per-image F1 over ALL evaluated images (a class
  absent from an image contributes 0);
- macro-F1 = mean of the per-class aggregates over the 10 canonical classes.

The model runs once per image; thresholds are swept on the cached output, so a
global sweep and per-class coordinate ascent are both cheap.

Example
-------
    PYTHONPATH=src python scripts/tune_nuclei_threshold.py \
        --image-dir /path/to/fold2_val/images \
        --nuclei-geojson-dir /path/to/fold2_val/nuclei \
        --nuclei-suffix _nuclei.geojson \
        --checkpoint models/best_primary.ckpt \
        --config configs/submission.toml \
        --per-class

Data layout: pass a directory of validation ``*.tif`` images and a directory of
matching nuclei GeoJSON files. Files are paired by stem: ``<stem>.tif`` matches
``<stem><--nuclei-suffix>``. To restrict to a k-fold split, pass ``--val-ids``
(a text/JSON file listing the validation sample stems) or ``--split-json`` +
``--fold``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prometheus.api import build_model, load_config
from prometheus.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    select_inference_state,
)
from prometheus.domain import NucleusClass
from prometheus.inference import decode_nuclei
from prometheus.models import MultitaskOutput
from puma_submission.prepare import prepare_image

CANONICAL_CLASSES = [nucleus_class.value for nucleus_class in NucleusClass]
MATCH_RADIUS_PX = 15.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def parse_gt_nuclei(path: Path) -> list[tuple[str, tuple[float, float]]]:
    """Return (class_name, (x, y)) per GT nucleus using the polygon vertex mean."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    instances: list[tuple[str, tuple[float, float]]] = []
    for feature in data.get("features", []):
        properties = feature.get("properties") or {}
        label = properties.get("label")
        if label is None:
            classification = properties.get("classification") or {}
            label = classification.get("name") if isinstance(classification, dict) else None
        if label is None:
            continue
        name = str(label).removeprefix("nuclei_")
        if name not in CANONICAL_CLASSES:
            continue
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if geometry_type == "Polygon":
            rings = [coordinates[0]] if coordinates else []
        elif geometry_type == "MultiPolygon":
            rings = [polygon[0] for polygon in coordinates if polygon]
        else:
            continue
        for ring in rings:
            points = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
            if len(points) > 1 and np.array_equal(points[0], points[-1]):
                points = points[:-1]
            if len(points) < 3:
                continue
            instances.append((name, (float(points[:, 0].mean()), float(points[:, 1].mean()))))
    return instances


def match_image(
    predictions: list[tuple[str, tuple[float, float]]],
    targets: list[tuple[str, tuple[float, float]]],
) -> dict[str, dict[str, int]]:
    """Class-aware greedy nearest matching within the radius, per image."""
    preds_by_class: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for index, (name, (x, y)) in enumerate(predictions):
        preds_by_class[name].append((x, y, index))
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CANONICAL_CLASSES}
    for name in CANONICAL_CLASSES:
        counts[name]["fp"] = sum(1 for candidate_name, _ in predictions if candidate_name == name)
    used: set[int] = set()
    for name, (gx, gy) in targets:
        counts[name]["fn"] += 1  # provisional; decremented on a hit
        best_index = -1
        best_distance = MATCH_RADIUS_PX
        for x, y, index in preds_by_class.get(name, []):
            if index in used:
                continue
            distance = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index >= 0:
            used.add(best_index)
            counts[name]["tp"] += 1
            counts[name]["fp"] -= 1
            counts[name]["fn"] -= 1
    return counts


def macro_f1(per_image_counts: list[dict[str, dict[str, int]]]) -> tuple[float, dict[str, float]]:
    """Challenge aggregation: per-class mean of per-image F1, then mean over classes."""
    per_class_scores: dict[str, list[float]] = {name: [] for name in CANONICAL_CLASSES}
    for counts in per_image_counts:
        for name in CANONICAL_CLASSES:
            values = counts[name]
            per_class_scores[name].append(_f1(values["tp"], values["fp"], values["fn"]))
    num_images = max(len(per_image_counts), 1)
    per_class_aggregate = {
        name: (sum(scores) / num_images if scores else 0.0) for name, scores in per_class_scores.items()
    }
    macro = sum(per_class_aggregate.values()) / len(CANONICAL_CLASSES)
    return macro, per_class_aggregate


def _to_cpu(output: MultitaskOutput) -> MultitaskOutput:
    return MultitaskOutput(
        tissue_logits=output.tissue_logits.cpu(),
        nuclei_center_logits=output.nuclei_center_logits.cpu(),
        nuclei_class_logits=output.nuclei_class_logits.cpu(),
        nuclei_offsets=output.nuclei_offsets.cpu(),
        nuclei_sizes=output.nuclei_sizes.cpu(),
        auxiliary={},
    )


def discover_samples(args) -> list[tuple[Path, Path]]:
    image_dir = Path(args.image_dir)
    nuclei_dir = Path(args.nuclei_geojson_dir)
    allowed: set[str] | None = None
    if args.val_ids:
        text = Path(args.val_ids).read_text(encoding="utf-8")
        try:
            allowed = set(json.loads(text))
        except json.JSONDecodeError:
            allowed = {line.strip() for line in text.splitlines() if line.strip()}
    elif args.split_json:
        manifest = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        fold = manifest[args.fold] if isinstance(manifest, list) else manifest[str(args.fold)]
        allowed = set(fold[args.split_key] if isinstance(fold, dict) else fold)
    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.tiff")):
        stem = image_path.stem
        if allowed is not None and stem not in allowed:
            continue
        geojson_path = nuclei_dir / f"{stem}{args.nuclei_suffix}"
        if not geojson_path.is_file():
            print(f"WARNING: no GT geojson for {image_path.name} (expected {geojson_path.name}); skipping")
            continue
        samples.append((image_path, geojson_path))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--nuclei-geojson-dir", type=Path, required=True)
    parser.add_argument("--nuclei-suffix", default="_nuclei.geojson")
    parser.add_argument("--config", type=Path, default=Path("configs/submission.toml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_primary.ckpt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-ids", type=Path, help="Text (one stem/line) or JSON list of validation stems")
    parser.add_argument("--split-json", type=Path, help="k-fold manifest to select validation stems from")
    parser.add_argument("--fold", type=int, default=2, help="Fold index into --split-json")
    parser.add_argument("--split-key", default="val", help="Key holding validation stems inside a fold entry")
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{v:.2f}" for v in np.arange(0.0, 0.95, 0.05)),
        help="Comma-separated class-confidence thresholds to sweep",
    )
    parser.add_argument("--per-class", action="store_true", help="Also search per-class thresholds (coordinate ascent)")
    args = parser.parse_args()

    grid = [float(value) for value in args.thresholds.split(",")]
    device = torch.device(args.device)
    config = load_config(args.config)

    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    assert_checkpoint_compatible(checkpoint, config)
    model = build_model(config)
    model.load_state_dict(select_inference_state(checkpoint), strict=True)
    model = model.to(device).eval()
    del checkpoint

    samples = discover_samples(args)
    if not samples:
        raise SystemExit("No (image, geojson) pairs found. Check --image-dir/--nuclei-geojson-dir/--nuclei-suffix.")
    print(f"Evaluating {len(samples)} validation images on {device}")

    cached: list[tuple[MultitaskOutput, object, list[tuple[str, tuple[float, float]]]]] = []
    with torch.no_grad():
        for image_path, geojson_path in samples:
            image, meta = prepare_image(image_path, config.input.image_size)
            output = model(image.to(device))
            cached.append((_to_cpu(output), meta, parse_gt_nuclei(geojson_path)))

    decode_kwargs = dict(
        stride=config.model.nuclei_feature_stride,
        threshold=config.postprocess.confidence_threshold,
        max_detections=config.postprocess.max_detections,
        local_max_kernel=config.postprocess.local_max_kernel,
    )

    def evaluate(class_threshold) -> tuple[float, dict[str, float]]:
        per_image = []
        for output, meta, gt in cached:
            detections = decode_nuclei(output, [meta], class_confidence_threshold=class_threshold, **decode_kwargs)[0]
            preds = [(det.label.value, det.centroid) for det in detections]
            per_image.append(match_image(preds, gt))
        return macro_f1(per_image)

    print("\n== Global class-confidence threshold sweep ==")
    print(f"{'threshold':>10}  {'macro_f1':>9}")
    best_threshold, best_macro = 0.0, -1.0
    for threshold in grid:
        macro, _ = evaluate(threshold)
        marker = ""
        if macro > best_macro:
            best_macro, best_threshold, marker = macro, threshold, "  <-- best"
        print(f"{threshold:10.2f}  {macro:9.4f}{marker}")
    baseline_macro, baseline_per_class = evaluate(0.0)
    print(f"\nBaseline (threshold 0.0) macro_f1 = {baseline_macro:.4f}")
    print(f"Best global threshold = {best_threshold:.2f}  ->  macro_f1 = {best_macro:.4f}")
    print(f"Set in configs/submission.toml:  class_confidence_threshold = {best_threshold}")

    if args.per_class:
        print("\n== Per-class coordinate ascent (start from best global) ==")
        vector = [best_threshold] * len(CANONICAL_CLASSES)
        current_macro, _ = evaluate(vector)
        for _ in range(3):  # a few passes; converges quickly on a small grid
            improved = False
            for class_index, class_name in enumerate(CANONICAL_CLASSES):
                best_local = vector[class_index]
                best_local_macro = current_macro
                for threshold in grid:
                    trial = list(vector)
                    trial[class_index] = threshold
                    macro, _ = evaluate(trial)
                    if macro > best_local_macro + 1e-9:
                        best_local_macro, best_local = macro, threshold
                if best_local != vector[class_index]:
                    vector[class_index] = best_local
                    current_macro = best_local_macro
                    improved = True
            if not improved:
                break
        final_macro, final_per_class = evaluate(vector)
        print(f"Per-class macro_f1 = {final_macro:.4f}  (global best was {best_macro:.4f})")
        print("Per-class thresholds and resulting aggregate F1:")
        print(f"{'class':16}{'threshold':>10}{'aggF1':>9}{'baseF1':>9}")
        for class_index, class_name in enumerate(CANONICAL_CLASSES):
            print(
                f"{class_name:16}{vector[class_index]:10.2f}"
                f"{final_per_class[class_name]:9.3f}{baseline_per_class[class_name]:9.3f}"
            )
        formatted = ", ".join(f"{value:.2f}" for value in vector)
        print(f"\nSet in configs/submission.toml:\n  class_confidence_thresholds = [{formatted}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
