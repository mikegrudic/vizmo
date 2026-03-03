"""Submodule in which gas render maps are defined"""

from .rendermap import RenderMap

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
