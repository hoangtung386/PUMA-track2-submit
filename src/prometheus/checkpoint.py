"""Read-only checkpoint contract for Prometheus inference."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import DeploymentConfig

CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ARCHITECTURE = "prometheus_multitask_v1"


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate the immutable outer checkpoint schema."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Prometheus checkpoint: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Prometheus checkpoint payload must be a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Expected checkpoint schema {CHECKPOINT_SCHEMA_VERSION}, got {payload.get('schema_version')}")
    if payload.get("architecture") != CHECKPOINT_ARCHITECTURE:
        raise ValueError(f"Unsupported architecture: {payload.get('architecture')}")
    if "model_state" not in payload:
        raise ValueError("Prometheus checkpoint has no model_state")
    return payload


def select_inference_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Prefer EMA weights when training persisted them."""
    ema_state = payload.get("ema_state")
    return ema_state if ema_state is not None else payload["model_state"]


def assert_checkpoint_compatible(
    payload: dict[str, Any],
    config: DeploymentConfig,
) -> None:
    """Require exact model configuration parity with the training checkpoint."""
    checkpoint_model = payload.get("config", {}).get("model")
    current_model = asdict(config.model)
    if checkpoint_model != current_model:
        raise ValueError(
            "Checkpoint model configuration does not match deployment config. "
            f"checkpoint={checkpoint_model}, requested={current_model}"
        )
