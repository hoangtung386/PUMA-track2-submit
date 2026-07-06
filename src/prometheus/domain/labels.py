"""Canonical PUMA submission labels and explicit value mappings."""

from __future__ import annotations

from enum import Enum


class TissueClass(str, Enum):
    BACKGROUND = "background"
    STROMA = "stroma"
    BLOOD_VESSEL = "blood_vessel"
    TUMOR = "tumor"
    EPIDERMIS = "epidermis"
    NECROSIS = "necrosis"


class NucleusClass(str, Enum):
    TUMOR = "tumor"
    STROMA = "stroma"
    ENDOTHELIUM = "endothelium"
    HISTIOCYTE = "histiocyte"
    MELANOPHAGE = "melanophage"
    LYMPHOCYTE = "lymphocyte"
    PLASMA_CELL = "plasma_cell"
    NEUTROPHIL = "neutrophil"
    APOPTOSIS = "apoptosis"
    EPITHELIUM = "epithelium"


class Track(str, Enum):
    TRACK_1 = "track1"
    TRACK_2 = "track2"


TISSUE_SUBMISSION_VALUE = {
    TissueClass.BACKGROUND: 0,
    TissueClass.STROMA: 1,
    TissueClass.BLOOD_VESSEL: 2,
    TissueClass.TUMOR: 3,
    TissueClass.EPIDERMIS: 4,
    TissueClass.NECROSIS: 5,
}


def nucleus_class_for_track(nucleus_class: NucleusClass, track: Track) -> str:
    """Map a canonical nucleus class to its challenge-track class name."""
    if track is Track.TRACK_2:
        return nucleus_class.value
    if nucleus_class is NucleusClass.TUMOR:
        return "tumor"
    if nucleus_class in {NucleusClass.LYMPHOCYTE, NucleusClass.PLASMA_CELL}:
        return "lymphocyte"
    return "other"
