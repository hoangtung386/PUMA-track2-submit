"""Typed end-to-end predictor for PrometheusNet."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..domain import Detection, ImageMeta
from ..models import MultitaskOutput
from .nuclei_decoder import decode_nuclei
from .spatial import restore_mask


@dataclass
class MultitaskPrediction:
    tissue_masks: list[np.ndarray]
    nuclei: list[list[Detection]]


class PrometheusPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        nuclei_stride: int = 4,
        confidence_threshold: float = 0.25,
        max_detections: int = 1000,
        local_max_kernel: int = 3,
        class_confidence_threshold: float | Sequence[float] = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.nuclei_stride = nuclei_stride
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.local_max_kernel = local_max_kernel
        self.class_confidence_threshold = class_confidence_threshold

    @torch.no_grad()
    def predict(self, images: torch.Tensor, metadata: list[ImageMeta]) -> MultitaskPrediction:
        """Predict both tasks from one multitask checkpoint."""
        output = self.model(images.to(self.device))
        if not isinstance(output, MultitaskOutput):
            raise TypeError("PrometheusPredictor expects MultitaskOutput")
        masks = output.tissue_logits.argmax(dim=1)
        restored = [restore_mask(mask, meta) for mask, meta in zip(masks, metadata, strict=True)]
        nuclei = decode_nuclei(
            output,
            metadata,
            stride=self.nuclei_stride,
            threshold=self.confidence_threshold,
            max_detections=self.max_detections,
            local_max_kernel=self.local_max_kernel,
            class_confidence_threshold=self.class_confidence_threshold,
        )
        return MultitaskPrediction(restored, nuclei)

    @torch.no_grad()
    def predict_tissue(self, images: torch.Tensor, metadata: list[ImageMeta]) -> list[np.ndarray]:
        """Return only tissue masks while preserving the checkpoint's backbone."""
        output = self._forward(images)
        masks = output.tissue_logits.argmax(dim=1)
        return [restore_mask(mask, meta) for mask, meta in zip(masks, metadata, strict=True)]

    @torch.no_grad()
    def predict_nuclei(self, images: torch.Tensor, metadata: list[ImageMeta]) -> list[list[Detection]]:
        """Return only nuclei decoded from the selected checkpoint."""
        output = self._forward(images)
        return decode_nuclei(
            output,
            metadata,
            stride=self.nuclei_stride,
            threshold=self.confidence_threshold,
            max_detections=self.max_detections,
            local_max_kernel=self.local_max_kernel,
            class_confidence_threshold=self.class_confidence_threshold,
        )

    def _forward(self, images: torch.Tensor) -> MultitaskOutput:
        output = self.model(images.to(self.device))
        if not isinstance(output, MultitaskOutput):
            raise TypeError("PrometheusPredictor expects MultitaskOutput")
        return output
