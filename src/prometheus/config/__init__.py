"""Strict deployment configuration for PrometheusNet."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


@dataclass
class PrometheusModelConfig:
    name: str = "prometheus_multitask_v1"
    in_channels: int = 3
    num_tissue_classes: int = 6
    num_nucleus_types: int = 10
    encoder_dims: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_depths: list[int] = field(default_factory=lambda: [3, 3, 9, 3])
    tissue_decoder_depths: list[int] = field(default_factory=lambda: [1, 2, 2])
    drop_path_rate: float = 0.1
    context_enabled: bool = True
    nuclei_feature_stride: int = 4

    def validate(self) -> None:
        if self.name != "prometheus_multitask_v1":
            raise ValueError(f"Unsupported model name: {self.name}")
        if self.in_channels != 3:
            raise ValueError("Deployment expects exactly three input channels")
        if len(self.encoder_dims) != 4 or len(self.encoder_depths) != 4:
            raise ValueError("Encoder dimensions and depths must contain four stages")
        if len(self.tissue_decoder_depths) != 3:
            raise ValueError("Tissue decoder depths must contain three levels")
        dimensions = self.encoder_dims + self.encoder_depths + self.tissue_decoder_depths
        if any(value <= 0 for value in dimensions):
            raise ValueError("Model dimensions and depths must be positive")
        if self.num_tissue_classes != 6 or self.num_nucleus_types != 10:
            raise ValueError("PUMA Track 2 requires six tissue and ten nucleus classes")
        if self.nuclei_feature_stride not in {4, 8}:
            raise ValueError("Nuclei feature stride must be 4 or 8")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("Drop-path rate must be in [0, 1)")


@dataclass
class InputConfig:
    image_size: int = 1024

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("Input image size must be positive")


@dataclass
class PostprocessConfig:
    confidence_threshold: float = 0.25
    max_detections: int = 1000
    local_max_kernel: int = 3

    def validate(self) -> None:
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("Confidence threshold must be between zero and one")
        if self.max_detections <= 0:
            raise ValueError("Maximum detections must be positive")
        if self.local_max_kernel <= 0 or self.local_max_kernel % 2 == 0:
            raise ValueError("Local-max kernel must be a positive odd integer")


@dataclass
class DeploymentConfig:
    model: PrometheusModelConfig = field(default_factory=PrometheusModelConfig)
    input: InputConfig = field(default_factory=InputConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    def validate(self) -> None:
        self.model.validate()
        self.input.validate()
        self.postprocess.validate()


def _strict_dataclass(cls, values: dict):
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> DeploymentConfig:
    """Load a deployment-only TOML file and reject unknown configuration."""
    with Path(path).open("rb") as file_obj:
        raw = tomllib.load(file_obj)
    allowed_sections = {"model", "input", "postprocess"}
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        raise ValueError(f"Unknown deployment sections: {sorted(unknown_sections)}")

    model_values = dict(raw.get("model", {}))
    context = model_values.pop("context", {})
    unknown_context = set(context) - {"enabled"}
    if unknown_context:
        raise ValueError(f"Unknown model.context fields: {sorted(unknown_context)}")
    if "enabled" in context:
        model_values["context_enabled"] = context["enabled"]

    config = DeploymentConfig(
        model=_strict_dataclass(PrometheusModelConfig, model_values),
        input=_strict_dataclass(InputConfig, raw.get("input", {})),
        postprocess=_strict_dataclass(PostprocessConfig, raw.get("postprocess", {})),
    )
    config.validate()
    return config


__all__ = [
    "DeploymentConfig",
    "InputConfig",
    "PostprocessConfig",
    "PrometheusModelConfig",
    "load_config",
]
