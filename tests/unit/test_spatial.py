import numpy as np

from prometheus.inference import letterbox_image, normalize_image
from prometheus.inference.spatial import (
    boxes_to_model,
    boxes_to_source,
    points_to_model,
    points_to_source,
)


def test_letterbox_coordinate_round_trip() -> None:
    _, metadata = letterbox_image(
        np.zeros((40, 80, 3), dtype=np.float32),
        (64, 64),
        "sample",
    )
    points = np.array([[0.0, 0.0], [79.0, 39.0], [20.5, 10.25]], dtype=np.float32)
    restored = points_to_source(points_to_model(points, metadata), metadata)
    np.testing.assert_allclose(restored, points, atol=1e-5)
    assert metadata.resized_size == (32, 64)
    assert metadata.pad_xy == (0, 16)


def test_box_coordinate_round_trip() -> None:
    _, metadata = letterbox_image(
        np.zeros((30, 60, 3), dtype=np.float32),
        (64, 64),
        "sample",
    )
    boxes = np.array([[2.0, 3.0, 20.0, 25.0]], dtype=np.float32)
    restored = boxes_to_source(boxes_to_model(boxes, metadata), metadata)
    np.testing.assert_allclose(restored, boxes, atol=1e-5)


def test_normalization_excludes_letterbox_padding() -> None:
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[:, 2:6] = np.linspace(0, 1, 32, dtype=np.float32).reshape(1, 4, 8)
    valid_mask = np.zeros((8, 8), dtype=bool)
    valid_mask[2:6] = True

    normalized = normalize_image(image, valid_mask)

    assert np.all(normalized[:, ~valid_mask] == 0)
    np.testing.assert_allclose(normalized[:, valid_mask].mean(axis=1), 0, atol=1e-5)
