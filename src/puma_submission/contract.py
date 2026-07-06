"""Filesystem contract for a PUMA Track 2 algorithm container."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PRIMARY_INPUT_DIR = Path("/input/images/melanoma-wsi")
LEGACY_INPUT_DIR = Path("/input/images/melanoma-whole-slide-image")
NUCLEI_OUTPUT_NAME = "melanoma-10-class-nuclei-segmentation.json"
TISSUE_OUTPUT_DIRECTORY = Path("images/melanoma-tissue-mask-segmentation")


@dataclass(frozen=True)
class SubmissionPaths:
    """Resolved immutable paths used for one container invocation."""

    input_directories: tuple[Path, ...]
    output_directory: Path
    config_path: Path
    checkpoint_path: Path

    @classmethod
    def from_environment(cls) -> SubmissionPaths:
        input_override = os.environ.get("PUMA_INPUT_DIR")
        input_directories = (Path(input_override),) if input_override else (PRIMARY_INPUT_DIR, LEGACY_INPUT_DIR)
        return cls(
            input_directories=input_directories,
            output_directory=Path(os.environ.get("PUMA_OUTPUT_DIR", "/output")),
            config_path=Path(os.environ.get("PUMA_CONFIG", "/opt/app/configs/submission.toml")),
            checkpoint_path=Path(os.environ.get("PUMA_CHECKPOINT", "/opt/app/models/prometheus.ckpt")),
        )

    @property
    def nuclei_output_path(self) -> Path:
        return self.output_directory / NUCLEI_OUTPUT_NAME

    def tissue_output_path(self, input_path: Path) -> Path:
        output_name = input_path.with_suffix(".tif").name
        return self.output_directory / TISSUE_OUTPUT_DIRECTORY / output_name


def discover_input(input_directories: tuple[Path, ...]) -> Path:
    """Return the single primary TIFF from the first populated input directory."""
    populated: list[tuple[Path, list[Path]]] = []
    for directory in input_directories:
        if not directory.is_dir():
            continue
        images = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and not path.stem.endswith("_context")
        )
        if images:
            populated.append((directory, images))

    if not populated:
        searched = ", ".join(str(path) for path in input_directories)
        raise FileNotFoundError(f"No primary TIFF found in: {searched}")
    if len(populated) > 1:
        locations = ", ".join(str(directory) for directory, _ in populated)
        raise ValueError(f"Input TIFFs found in multiple contract directories: {locations}")

    directory, images = populated[0]
    if len(images) != 1:
        names = ", ".join(path.name for path in images)
        raise ValueError(f"Expected exactly one primary TIFF in {directory}, found {len(images)}: {names}")
    return images[0]
