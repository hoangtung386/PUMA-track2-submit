from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from prometheus.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    select_inference_state,
)
from prometheus.config import DeploymentConfig, PrometheusModelConfig
from prometheus.models import PrometheusNet


def _config() -> DeploymentConfig:
    return DeploymentConfig(
        model=PrometheusModelConfig(
            encoder_dims=[8, 16, 32, 64],
            encoder_depths=[1, 1, 1, 1],
            tissue_decoder_depths=[1, 1, 1],
        )
    )


def _write_checkpoint(path: Path, config: DeploymentConfig, ema=False) -> None:
    state = PrometheusNet(config.model).state_dict()
    torch.save(
        {
            "schema_version": 2,
            "architecture": "prometheus_multitask_v1",
            "model_state": state,
            "ema_state": state if ema else None,
            "config": {"model": asdict(config.model)},
        },
        path,
    )


def test_checkpoint_load_and_compatibility(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "model.ckpt"
    _write_checkpoint(path, config)

    payload = load_checkpoint(path)

    assert_checkpoint_compatible(payload, config)
    assert select_inference_state(payload) is payload["model_state"]


def test_checkpoint_prefers_ema_and_rejects_model_mismatch(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "model.ckpt"
    _write_checkpoint(path, config, ema=True)
    payload = load_checkpoint(path)
    assert select_inference_state(payload) is payload["ema_state"]

    config.model.context_enabled = False
    with pytest.raises(ValueError, match="does not match"):
        assert_checkpoint_compatible(payload, config)


def test_checkpoint_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.ckpt"
    torch.save({"schema_version": 1, "architecture": "prometheus_multitask_v1"}, path)
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path)
