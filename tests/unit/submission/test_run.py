from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puma_submission.contract import SubmissionPaths
from puma_submission.run import run


def test_run_fails_before_model_loading_when_checkpoint_is_missing(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input" / "sample.tif"
    input_path.parent.mkdir()
    input_path.touch()
    config = tmp_path / "submission.toml"
    config.touch()
    paths = SubmissionPaths(
        input_directories=(input_path.parent,),
        output_directory=tmp_path / "output",
        config_path=config,
        nuclei_checkpoint_path=tmp_path / "missing-nuclei.ckpt",
        tissue_checkpoint_path=tmp_path / "missing-tissue.ckpt",
    )

    with pytest.raises(FileNotFoundError, match="nuclei checkpoint"):
        run(paths, "cpu")


def test_run_selects_nuclei_and_tissue_from_different_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input" / "sample.tif"
    input_path.parent.mkdir()
    input_path.touch()
    config_path = tmp_path / "submission.toml"
    nuclei_path = tmp_path / "best_primary.ckpt"
    tissue_path = tmp_path / "best_tissue.ckpt"
    for path in (config_path, nuclei_path, tissue_path):
        path.touch()
    paths = SubmissionPaths(
        input_directories=(input_path.parent,),
        output_directory=tmp_path / "output",
        config_path=config_path,
        nuclei_checkpoint_path=nuclei_path,
        tissue_checkpoint_path=tissue_path,
    )
    calls = []
    written = {}

    class FakePredictor:
        def __init__(self, task: str) -> None:
            self.task = task
            self.model = torch.nn.Identity()

        def predict_nuclei(self, image, metadata):
            calls.append((self.task, "nuclei"))
            return [["fold-2-nuclei"]]

        def predict_tissue(self, image, metadata):
            calls.append((self.task, "tissue"))
            return [np.zeros((2, 2), dtype=np.uint8)]

    monkeypatch.setattr(
        "puma_submission.run.load_config",
        lambda _: SimpleNamespace(input=SimpleNamespace(image_size=2)),
    )
    monkeypatch.setattr("puma_submission.run.prepare_image", lambda *_: (torch.zeros(1, 3, 2, 2), object()))
    monkeypatch.setattr(
        "puma_submission.run.load_predictor",
        lambda _, path, __: FakePredictor("primary" if path == nuclei_path else "tissue"),
    )
    monkeypatch.setattr("puma_submission.run.write_nuclei_json", lambda value, *_: written.update(nuclei=value))
    monkeypatch.setattr("puma_submission.run.write_tissue_tiff", lambda value, *_: written.update(tissue=value))
    monkeypatch.setattr("puma_submission.run.validate_outputs", lambda *_: None)

    run(paths, "cpu")

    assert calls == [("primary", "nuclei"), ("tissue", "tissue")]
    assert written["nuclei"] == ["fold-2-nuclei"]
    assert written["tissue"].shape == (2, 2)
