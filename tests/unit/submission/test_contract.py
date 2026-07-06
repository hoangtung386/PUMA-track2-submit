from pathlib import Path

import pytest

from puma_submission.contract import SubmissionPaths, discover_input


def test_discovers_one_tiff_and_ignores_context(tmp_path: Path) -> None:
    primary = tmp_path / "case.tif"
    primary.touch()
    (tmp_path / "case_context.tif").touch()

    assert discover_input((tmp_path,)) == primary


def test_discovery_rejects_multiple_primary_images(tmp_path: Path) -> None:
    (tmp_path / "a.tif").touch()
    (tmp_path / "b.tiff").touch()

    with pytest.raises(ValueError, match="exactly one"):
        discover_input((tmp_path,))


def test_discovery_rejects_images_in_two_contract_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.tif").touch()
    (second / "b.tif").touch()

    with pytest.raises(ValueError, match="multiple contract directories"):
        discover_input((first, second))


def test_output_paths_preserve_input_filename(tmp_path: Path) -> None:
    paths = SubmissionPaths(
        (tmp_path,),
        tmp_path,
        Path("config"),
        Path("nuclei.ckpt"),
        Path("tissue.ckpt"),
    )

    assert paths.nuclei_output_path.name == "melanoma-10-class-nuclei-segmentation.json"
    assert paths.tissue_output_path(Path("uuid.tif")).relative_to(tmp_path) == Path(
        "images/melanoma-tissue-mask-segmentation/uuid.tif"
    )
    assert paths.tissue_output_path(Path("uuid.tiff")).suffix == ".tif"
