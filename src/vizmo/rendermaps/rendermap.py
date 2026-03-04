"""Rendermap class definition: provides a method for rendering the map"""

from dataclasses import dataclass
from meshoid import Meshoid
import numpy as np
from . import MINIMAL_FIELDS


# @dataclass
class RenderMap:
    def __init__(self):
        self.required_datafields = MINIMAL_FIELDS

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        """
        Method that performs the rendering task provided the particle data, an optional meshoid instantiated with that
        particle data, and a set of mapargs specifying the map size, center, resolution, and orientation.

        By default we return a trivial value here.
        """
        return self._render_function(pdata, meshoid, mapargs)

    @property
    def cmap_default_limits(self):
        """Returns a tuple providing the default upper and lower colormap limits for rendering the map"""
        return None, None

    def add_required_datafields(self, datafields):
        self.required_datafields += datafields

    def set_render_method(self, function):
        self.render = function

    @staticmethod
    def custom_render(meshoid_maptype: str = "SurfaceDensity", datafield: str = "PartType0/Masses"):
        """Returns a function with signature (pdata: dict, meshoid: Meshoid, mapargs: dict)
        that generates a map by plugging the specified snapshot datafield into that method.

        Parameters
        ----------
        meshoid_maptype: string, optional
            Choices are SurfaceDensity, ProjectedAverage, Slice (default: SurfaceDensity)
        datafield: string, optional
            The snapshot datafield that will be plugged into the specified render method
        """

        rendermethod = getattr(Meshoid, meshoid_maptype)

        def renderfunc(pdata: dict, meshoid: Meshoid, mapargs: dict):
            return

    # def set_custom_rendermethod(self, meshoid_maptype="SurfaceDensity", datafield):
    # def self.render(pdata: dict, meshoid: Meshoid, mapargs: dict):
    # return 0
