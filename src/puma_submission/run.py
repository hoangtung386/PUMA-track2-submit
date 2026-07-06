"""Single-process inference entry point for a PUMA algorithm container."""

from __future__ import annotations

import logging
import sys

import torch

from prometheus.api import load_config, load_predictor
from prometheus.domain import Track
from prometheus.io import write_nuclei_json, write_tissue_tiff

from .contract import SubmissionPaths, discover_input
from .prepare import prepare_image
from .validation import validate_outputs

LOGGER = logging.getLogger(__name__)


def run(paths: SubmissionPaths, device: torch.device | str | None = None) -> None:
    """Execute one end-to-end prediction and validate its two outputs."""
    input_path = discover_input(paths.input_directories)
    if not paths.config_path.is_file():
        raise FileNotFoundError(f"Missing submission config: {paths.config_path}")
    if not paths.checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Prometheus checkpoint: {paths.checkpoint_path}")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved_device.type != "cuda":
        LOGGER.warning("CUDA is unavailable; inference will run on %s", resolved_device)

    config = load_config(paths.config_path)
    predictor = load_predictor(config, paths.checkpoint_path, resolved_device)
    image, metadata = prepare_image(input_path, config.input.image_size)
    prediction = predictor.predict(image, [metadata])

    tissue_path = paths.tissue_output_path(input_path)
    nuclei_path = paths.nuclei_output_path
    write_tissue_tiff(prediction.tissue_masks[0], tissue_path)
    write_nuclei_json(prediction.nuclei[0], nuclei_path, Track.TRACK_2)
    validate_outputs(input_path, tissue_path, nuclei_path)
    LOGGER.info("Wrote tissue segmentation to %s", tissue_path)
    LOGGER.info("Wrote %d nuclei to %s", len(prediction.nuclei[0]), nuclei_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run(SubmissionPaths.from_environment())
    except Exception:
        LOGGER.exception("PUMA inference failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
