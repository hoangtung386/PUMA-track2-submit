from pathlib import Path

import pytest

from prometheus.config import load_config


def test_submission_config_loads() -> None:
    config = load_config("configs/submission.toml")
    assert config.model.name == "prometheus_multitask_v1"
    assert config.model.num_nucleus_types == 10
    assert config.model.drop_path_rate == 0.2
    assert config.input.image_size == 1024
    assert config.postprocess.confidence_threshold == 0.25


def test_config_rejects_training_sections(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[trainer]\nepochs = 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown deployment sections"):
        load_config(path)


def test_config_rejects_unknown_model_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[model]\nunknown_magic = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown PrometheusModelConfig fields"):
        load_config(path)
