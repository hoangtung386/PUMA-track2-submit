"""Stable composition API for Prometheus deployment."""

from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    select_inference_state,
)
from .config import DeploymentConfig, load_config
from .inference import PrometheusPredictor
from .models import PrometheusNet


def build_model(config: DeploymentConfig) -> PrometheusNet:
    """Construct the exact architecture described by deployment config."""
    return PrometheusNet(config.model)


def load_predictor(
    config: DeploymentConfig,
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> PrometheusPredictor:
    """Load a compatible checkpoint and return an evaluation-mode predictor."""
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Checkpoints also contain optimizer and RNG state. Keep the payload on CPU,
    # load only inference weights, then move the model to the target device.
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    assert_checkpoint_compatible(checkpoint, config)
    model = build_model(config)
    model.load_state_dict(select_inference_state(checkpoint), strict=True)
    del checkpoint
    return PrometheusPredictor(
        model=model,
        device=resolved_device,
        nuclei_stride=config.model.nuclei_feature_stride,
        confidence_threshold=config.postprocess.confidence_threshold,
        max_detections=config.postprocess.max_detections,
        local_max_kernel=config.postprocess.local_max_kernel,
        class_confidence_threshold=config.postprocess.resolved_class_thresholds(
            config.model.num_nucleus_types
        ),
    )


__all__ = ["build_model", "load_config", "load_predictor"]
