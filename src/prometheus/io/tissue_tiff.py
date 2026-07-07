"""Write class-index tissue masks using explicit challenge value remapping.

Grand Challenge imports the tissue output with ``panimg`` before evaluation. Its
TIFF builder rejects a file whose pixel spacing cannot be derived ("Voxel width
could not be determined"), so the image MUST carry an X/Y resolution with a
determinable unit (inches). This matches the official PUMA baseline, which
writes the mask at 300 dpi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain import TISSUE_SUBMISSION_VALUE, TissueClass

TISSUE_MODEL_ORDER = (
    TissueClass.BACKGROUND,
    TissueClass.TUMOR,
    TissueClass.STROMA,
    TissueClass.EPIDERMIS,
    TissueClass.NECROSIS,
    TissueClass.BLOOD_VESSEL,
)

# Dots-per-inch matching the official baseline. tifffile records this as
# X/YResolution with ResolutionUnit=INCH, which panimg uses to derive spacing.
_RESOLUTION_DPI = 300


def write_tissue_tiff(mask: np.ndarray, path: str | Path) -> None:
    import tifffile

    source = np.asarray(mask)
    output = np.zeros(source.shape, dtype=np.uint8)
    for train_index, tissue_class in enumerate(TISSUE_MODEL_ORDER):
        output[source == train_index] = TISSUE_SUBMISSION_VALUE[tissue_class]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    foreground = output[output > 0]
    minimum = int(foreground.min()) if foreground.size else 0
    maximum = int(foreground.max()) if foreground.size else 0
    with tifffile.TiffWriter(destination) as tif:
        tif.write(
            output,
            photometric="minisblack",
            resolution=(_RESOLUTION_DPI, _RESOLUTION_DPI),
            resolutionunit="INCH",
            extratags=[
                (340, "H", 1, minimum, False),  # SMinSampleValue
                (341, "H", 1, maximum, False),  # SMaxSampleValue
            ],
        )
