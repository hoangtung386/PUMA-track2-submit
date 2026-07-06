from pathlib import Path

import pytest

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
        checkpoint_path=tmp_path / "missing.ckpt",
    )

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run(paths, "cpu")
