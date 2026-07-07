import numpy as np
import torch

from prometheus.domain import ImageMeta, NucleusClass
from prometheus.inference import PrometheusPredictor
from prometheus.models import MultitaskOutput


class _FakePrometheus(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> MultitaskOutput:
        tissue = torch.zeros(images.shape[0], 6, *images.shape[-2:], device=images.device)
        tissue[:, 2] = 5
        centers = torch.full((images.shape[0], 10, 8, 8), -10.0, device=images.device)
        centers[:, 3, 2, 4] = 10.0
        classes = torch.full_like(centers, -5.0)
        classes[:, 3, 2, 4] = 5.0
        offsets = torch.zeros(images.shape[0], 2, 8, 8, device=images.device)
        sizes = torch.ones(images.shape[0], 2, 8, 8, device=images.device) * 2
        return MultitaskOutput(tissue, centers, classes, offsets, sizes)


def test_predictor_restores_source_space() -> None:
    meta = ImageMeta("x", (16, 32), (32, 32), (16, 32), (1.0, 1.0), (0, 8))
    result = PrometheusPredictor(_FakePrometheus(), nuclei_stride=4).predict(torch.zeros(1, 3, 32, 32), [meta])
    assert result.tissue_masks[0].shape == (16, 32)
    assert np.all(result.tissue_masks[0] == 2)
    assert len(result.nuclei[0]) == 1
    assert result.nuclei[0][0].label is NucleusClass.HISTIOCYTE
    assert result.nuclei[0][0].centroid == (16.0, 0.0)


def test_task_specific_prediction_methods() -> None:
    meta = ImageMeta("x", (16, 32), (32, 32), (16, 32), (1.0, 1.0), (0, 8))
    predictor = PrometheusPredictor(_FakePrometheus(), nuclei_stride=4)
    image = torch.zeros(1, 3, 32, 32)

    tissue = predictor.predict_tissue(image, [meta])
    nuclei = predictor.predict_nuclei(image, [meta])

    assert tissue[0].shape == (16, 32)
    assert np.all(tissue[0] == 2)
    assert len(nuclei[0]) == 1
    assert nuclei[0][0].label is NucleusClass.HISTIOCYTE


def test_global_class_confidence_threshold_filters_detections() -> None:
    meta = ImageMeta("x", (16, 32), (32, 32), (16, 32), (1.0, 1.0), (0, 8))
    image = torch.zeros(1, 3, 32, 32)

    # The fake predicts histiocyte with ~1.0 class probability; a near-1 threshold drops it.
    kept = PrometheusPredictor(_FakePrometheus(), nuclei_stride=4, class_confidence_threshold=0.5)
    dropped = PrometheusPredictor(_FakePrometheus(), nuclei_stride=4, class_confidence_threshold=0.999999)

    assert len(kept.predict_nuclei(image, [meta])[0]) == 1
    assert len(dropped.predict_nuclei(image, [meta])[0]) == 0


def test_per_class_threshold_targets_one_class() -> None:
    meta = ImageMeta("x", (16, 32), (32, 32), (16, 32), (1.0, 1.0), (0, 8))
    image = torch.zeros(1, 3, 32, 32)
    # The single detection is histiocyte (index 3). A high threshold only on that
    # class prunes it, while high thresholds on other classes leave it untouched.
    prune_histiocyte = [0.0] * 10
    prune_histiocyte[3] = 0.999999
    prune_others = [0.999999] * 10
    prune_others[3] = 0.0

    pruned = PrometheusPredictor(_FakePrometheus(), nuclei_stride=4, class_confidence_threshold=prune_histiocyte)
    spared = PrometheusPredictor(_FakePrometheus(), nuclei_stride=4, class_confidence_threshold=prune_others)

    assert len(pruned.predict_nuclei(image, [meta])[0]) == 0
    assert len(spared.predict_nuclei(image, [meta])[0]) == 1
