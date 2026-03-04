"""Submodule in which gas render maps are defined"""

MINIMAL_FIELDS = (
    "PartType0/Coordinates",
    "PartType0/SmoothingLength",
)

DEFAULT_MAPS = (
    "SurfaceDensity",
    "VelocityDispersion",
    "MassWeightedTemperature",
    "AlfvenSpeed",
    #    "XCoordinate",
    #  "ZCoordinate",
)
from .rendermap import RenderMap
