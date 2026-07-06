"""Host-side CLI for validating files emitted by a container."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contract import NUCLEI_OUTPUT_NAME, TISSUE_OUTPUT_DIRECTORY
from .validation import validate_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate_outputs(
        args.input,
        args.output / TISSUE_OUTPUT_DIRECTORY / args.input.with_suffix(".tif").name,
        args.output / NUCLEI_OUTPUT_NAME,
    )
    print("Submission outputs are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
