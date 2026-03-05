"""Routines for interpolation of snapshot data"""

import numpy as np
from numba import vectorize


@vectorize
def NearestImage(dx, boxsize):
    """
    Given a coordinate difference dx, return the coordinate difference of the nearest periodic image
    of the point offset by dx, for a given triple-periodic box size.
    """
    if abs(dx) > 0.5 * boxsize:
        return -np.copysign(boxsize - abs(dx), dx)
    else:
        return dx


class Interpolator:
    """Class that implements interpolation with various methods"""

    def __init__(self, field=None, method=None, header=None):
        """
        Parameters:
        field: string, optional
            Data field to interpolate on: can automatically decide which method to use based upon this
        method: string, optional
            Override for method of interpolation. Options: None, linear, logarithmic, min
        header: dict, optional
            Snapshot header data, required for e.g. box bounds or other
        """
        self.field = field
        self.header = header
        if field:
            self.method = field_interpolator_method(field)
        else:
            self.method = method

    def interpolate(self, t, t1, t2, f1, f2):
        """Method that returns the interpolant of a data field defined at times t1 and t2, with values f1 and f2"""
        if f1.shape != f2.shape:
            raise ValueError(f"Cannot interpolate data fields with different shapes: f1={f1} f2={f2}")
        self.time_order(t1, t2, f1, f2)
        if "Coordinates" in self.field:  # special box-wrapping behaviour
            return self.interpolate_coordinates(t, t1, t2, f1, f2)

        if self.method is None:
            return f1
        else:
            func = getattr(self, "interpolate_" + self.method)  # e.g. self.interpolate_linear()

        return func(t, t1, t2, f1, f2)

    @staticmethod
    def interpolate_coordinates(t, t1, t2, f1, f2, boxsize=None):
        """Special method for handling box-wrapping"""
        if boxsize is None:
            return Interpolator.interpolate_linear(t, t1, t2, f1, f2)
        dx = f2 - f1
        dx = NearestImage(dx, boxsize)

    @staticmethod
    def interpolate_min(t, t1, t2, f1, f2):
        """Return the interpolated data of lesser magnitude"""
        return np.where(np.abs(f2) > np.abs(f1), f1, f2)

    @staticmethod
    def time_order(t1, t2, f1, f2):
        """
        Make sure times and data are time-ordered.
        """
        if t1 > t2:
            t1, t2 = t2, t1
            f1, f2 = f2, f1

    @staticmethod
    def interpolate_linear(t, t1, t2, f1, f2, extrapolate=True):
        """
        Linear time interpolant

        Parameters
        ----------
        t: scalar
            Time to interpolate to
        t1, t2: scalar
            Times at which the data are known
        f1, f2: array_like
            Data to interpolate
        extrapolate: Boolean, optional
            Whether to extrapolate outside the bounds of [t1,t2]
        """
        if not extrapolate:
            if t <= t1:
                return f1
            elif t >= t2:
                return f2

        dt = t2 - t1
        weight2 = (t - t1) / dt
        weight1 = 1 - weight2
        return weight1 * f2 + weight2 * f2

    @staticmethod
    def interpolate_log(t, t1, t2, f1, f2, extrapolate=True, interpolate_log_time=False):
        """
        Logarithmic time interpolant

        Parameters
        ----------
        t: scalar
            Time to interpolate to
        t1, t2: scalar
            Times at which the data are known
        f1, f2: array_like
            Data to interpolate
        extrapolate: Boolean, optional
            Whether to extrapolate outside the bounds of [t1,t2]
        """
        if not extrapolate:
            if t <= t1:
                return f1
            elif t >= t2:
                return f2

        if np.any(f1 <= 0) or np.any(f2 <= 0):
            raise ValueError("Cannot log-interpolate quantities with non-positive values.")

        if interpolate_log_time and t1 > 0:
            logt1, logt2 = np.log(t1), np.log(t2)
            dlogt = logt1 - logt2
            weight2 = np.log(t / t1) / dlogt
            weight1 = 1 - weight2
        else:
            dt = t2 - t1
            weight2 = (t - t1) / dt
            weight1 = 1 - weight2
        return f1**weight1 * f2**weight2


def field_interpolator_method(field):
    """Returns the interpolation method that should be used for a given data field"""

    fields_to_skip = (
        "Header",
        "PartType0/ParticleIDs",
        "PartType5/ParticleIDs",
        "PartType0/ParticleIDGenerationNumber",
        "PartType0/ParticleChildIDsNumber",
    )  # fields for which interpolation is not defined

    fields_to_interp_log = (
        "PartType0/SmoothingLength",
        "PartType0/InternalEnergy",
        "PartType0/Pressure",
        "PartType0/SoundSpeed",
        "PartType0/Density",
        "PartType5/Masses",
        "PartType5/BH_Mass",
        "PartType0/Masses",
        "PartType0/ElectronAbundance",
        "PartType0/HII",
        "PartType0/Temperature",
        "PartType0/PhotonEnergy",
        "PartType0/Metallicity",
    )
    fields_to_keep_smallest = ()

    if "Coordinates" in field:
        return "coordinates"
    elif "Particle" in field and "ID" in field or field in fields_to_skip:
        return None
    elif field in fields_to_interp_log:
        return "log"
    elif field in fields_to_keep_smallest:
        return "min"
    else:
        return "linear"

    # def SnapInterpolate(t, t1, t2, snapdata_buffer=None, timeline=None):
    #     fields_to_skip = (
    #         "Header",
    #         "PartType0/ParticleIDs",
    #         "PartType5/ParticleIDs",
    #         "PartType0/ParticleIDGenerationNumber",
    #         "PartType0/ParticleChildIDsNumber",
    #     )  # fields we throw away
    #     fields_to_interp_log = (
    #         "PartType0/SmoothingLength",
    #         "PartType0/InternalEnergy",
    #         "PartType0/Pressure",
    #         "PartType0/SoundSpeed",
    #         "PartType0/Density",
    #         "PartType5/Masses",
    #         "PartType5/BH_Mass",
    #         "PartType0/Masses",
    #         "PartType0/ElectronAbundance",
    #         "PartType0/HII",
    #         "PartType0/Temperature",
    #         "PartType0/PhotonEnergy",
    #         "PartType0/Metallicity",
    #     )  # interpolate logarithmically
    #     fields_to_keep_lowest = ()  # keep the lowest value between snapshots, can be used to remove spikes that exist for only one snapshot that are hard to interpolate

    # interpolated_data = snapdata_buffer[t1].copy()
    # idx1, idx2 = {}, {}
    # for ptype in "PartType0", "PartType5":
    #     if ptype + "/ParticleIDs" in snapdata_buffer[t1].keys():
    #         id1 = np.array(snapdata_buffer[t1][ptype + "/ParticleIDs"])
    #     else:
    #         id1 = np.array([])
    #     if ptype + "/ParticleIDs" in snapdata_buffer[t2].keys():
    #         id2 = np.array(snapdata_buffer[t2][ptype + "/ParticleIDs"])
    #     else:
    #         id2 = np.array([])
    #     common_ids = np.intersect1d(id1, id2)
    #     idx1[ptype] = np.in1d(np.sort(id1), common_ids)
    #     idx2[ptype] = np.in1d(np.sort(id2), common_ids)

    # for field in snapdata_buffer[t1].keys():
    #     if not (field in fields_to_skip):
    #         ptype = field.split("/")[0]
    #         f1, f2 = snapdata_buffer[t1][field][idx1[ptype]], snapdata_buffer[t2][field][idx2[ptype]]
    #         wt1 = (
    #             (t2 - t) / (t2 - t1) * np.ones_like(f1)
    #         )  # relative weights, can be set individually for each cell, for now we just use the same time linear weight for all cells
    #         wt2 = 1.0 - wt1
    #         if field in fields_to_leave_as_is:
    #             interpolated_data[field] = f1.copy()
    #         elif field in fields_to_keep_lowest:
    #             interpolated_data[field] = f1.copy()
    #             ind2 = np.abs(f1) > np.abs(f2)
    #             interpolated_data[field][ind2] = f2[ind2]
    #         else:
    #             if field in fields_to_interp_log:
    #                 positive = (f1 > 0) & (f2 > 0)

    #                 # we interpolate linearily for cells where the value in either snapashots is non-positive, log for the rest
    #                 if np.any(~positive):
    #                     interpolated_data[field] = f1 * wt1 + f2 * wt2
    #                     interpolated_data[field][positive] = np.exp(
    #                         np.log(f1[positive]) * wt1[positive] + np.log(f2[positive]) * wt2[positive]
    #                     )
    #                 else:
    #                     interpolated_data[field] = np.exp(np.log(f1) * wt1 + np.log(f2) * wt2)
    #             else:  # interpolate everything else linearily
    #                 if "Coordinates" in field:  # special behaviour to handle periodic BCs
    #                     dx = f2 - f1
    #                     dx = NearestImage(dx, snapdata_buffer[t1]["Header"]["BoxSize"])
    #                     interpolated_data[field] = (f1 + wt2 * dx) % (snapdata_buffer[t1]["Header"]["BoxSize"])
    #                 else:
    #                     interpolated_data[field] = f1 * wt1 + f2 * wt2

    # return interpolated_data
