"""IO routines for multipanel plots"""

from . import rendermaps
from .rendermaps import MINIMAL_FIELDS, DEFAULT_MAPS
from glob import glob
from os.path import isfile
from natsort import natsorted
import h5py
import numpy as np
from astropy import units as u
from meshoid import Meshoid
import astropy
from functools import cache
from warnings import warn

DEFAULT_UNITS = {
    "Length": u.pc,
    "Speed": u.km / u.s,
    "Time": u.Myr,
    "MagneticField": u.gauss,
    "Temperature": u.K,
    "Mass": u.Msun,
}


def get_snapshot_for_maps(snapshot_path: str, maps=DEFAULT_MAPS) -> dict:
    """Does the I/O to get the data required for the specified maps"""
    required_data = set.union(*[getattr(rendermaps, s).required_datafields for s in maps])
    snapdata = get_snapshot_data(snapshot_path, required_data)
    # # transformation so we are projecting to the x-z plane
    # data = snapdata[s2]
    # if len(data.shape) == 2 and data.shape[-1] == 3:
    #     snapdata[s2] = np.c_[data[:, 0], data[:, 2], data[:, 1]]
    # if "Coordinates" in s2:
    #     snapdata[s2] -= snapdata["Header"]["BoxSize"] * 0.5

    return snapdata


@cache
def get_snapshot_data(snapshot_path: str, required_data=MINIMAL_FIELDS, units=True) -> dict:
    """Given a tuple of required datafields, open the snapshot at snapshot_path and put those data into a dict"""
    snapdata = {}
    with h5py.File(snapshot_path, "r") as F:
        unitdict = get_snapshot_units(F)
        snapdata["Header"] = dict(F["Header"].attrs)
        for i in range(6):
            s = f"PartType{i}"
            if s not in F.keys():
                continue
            for k in F[s].keys():
                s2 = s + "/" + k
                if s2 in required_data or i == 5:  # always get star data because it doesn't take much space
                    snapdata[s2] = F[s2][:]

    if units:
        assign_units_to_snapdata(snapdata, unitdict)
    return snapdata


def get_snapdata_at_time(snapshot_directory: str, time, interpolation_order=1) -> dict:
    """Gets snapshot data at a given time, performing interpolation if required."""
    raise NotImplementedError("interpolation not yet implemented")
    timeline = get_snapshot_timeline(snapshot_directory)
    unit_time = list(timeline)[0].unit
    if not isinstance(time, astropy.units.Quantity):
        # if a simple numeric value is supplied, assume it's in the same units as the timeline
        time *= unit_time

    if time in timeline.keys():
        return get_snapshot_data(timeline[time])
    else:  # must identify the nearest times
        t_values = np.array([s.value for s in timeline])
        dt = t_values - time.value
        # dt = snaptimes - time
        t1, t2 = sorted(t_values[np.argsort(np.abs(dt))][:2]) * unit_time
        return {}  # interpolated_snapdata(time, t1, t2, timeline)


def assign_units_to_snapdata(snapdata: dict, unitdict: dict, default_units=DEFAULT_UNITS):
    """Assigns astropy units to snapshot data given a dictionary of code units"""

    unitdict = unitdict.copy()
    for k, unit in unitdict.items():
        if isinstance(default_units[k], u.Quantity):
            unitdict[k] = unit.to(default_units[k])
        # else:
        # unitdict[k]

    for field, data in snapdata.items():
        if field == "Header":
            snapdata[field]["BoxSize"] *= unitdict["Length"]
            snapdata[field]["Time"] *= unitdict["Time"]
            continue

        unit = 1.0
        if "Coordinates" in field or "SmoothingLength" in field:
            unit = unitdict["Length"]
        elif "Temperature" in field:
            unit = u.K
        elif "Magnetic" in field:
            unit = unitdict["MagneticField"]
        elif "Density" in field:
            unit = unitdict["Mass"] / unitdict["Length"] ** 3
        elif "Speed" in field or "Velo" in field:
            unit = unitdict["Speed"]
        snapdata[field] = data * unit


def get_snapshot_units(F: h5py.File, default_starforge_units=False):
    """Given an h5py file instance for a snapshot, returns a dictionary
    whose entries are astropy quantities giving the unit length, speed, mass, and magnetic field for the snapshot.

    Parameters
    ----------
    F: h5py.File
        h5py file instance for the snapshot
    Returns:
        Dictionary with keys "Length", "Speed", "Mass", and "MagneticField" giving the unit quantities for the simulation.
    """
    if "UnitLength_In_CGS" in F["Header"].attrs.keys():
        unit_length = F["Header"].attrs["UnitLength_In_CGS"] * u.cm
        unit_speed = F["Header"].attrs["UnitVelocity_In_CGS"] * u.cm / u.s
        unit_mass = F["Header"].attrs["UnitMass_In_CGS"] * u.g
    else:
        warn("vizmo warning: units not found in snapshot.")
        unit_length = unit_speed = unit_mass = 1
    unit_time = unit_length / unit_speed
    if default_starforge_units:
        unit_magnetic_field = 1e4 * u.gauss  # hardcoded right now, can we actually get this from the header???
        warn("Warning: unit magnetic field not specified, assuming unit magnetic field is in T")
    else:
        unit_magnetic_field = u.gauss  # hardcoded right now, can we actually get this from the header???
        warn("Warning: unit magnetic field not specified, assuming unit magnetic field is in gauss")
    return {
        "Length": unit_length,
        "Speed": unit_speed,
        "Mass": unit_mass,
        "MagneticField": unit_magnetic_field,
        "Time": unit_time,
    }


@cache
def get_snapshot_timeline(output_dir, verbose=False, cache_timeline=True, unit=u.Myr) -> dict:
    """
    Given a simulation directory, does a pass through the present HDF5 snapshots
    and compiles a list of snapshot paths and their associated

    Parameters
    ----------
    output_dir: string
        Path of the directory containing the snapshots
    verbose: boolean, optional
        Whether to print verbose status updates
    unit: astropy.units.core.PrefixUnit, optional
        Time unit to convert snapshot times to (default: Myr)
    cache_timeline: boolean, optional
        Whether to cache the timeline for future lookup in a file output_dir + "/.timeline" (default: True)

    Returns
    -------
    dict whose keys are list of snapshot times and values are the corresponding snapshot paths
    """
    times = []
    snappaths = []
    if verbose:
        print("Getting snapnum timeline...")
    snappaths = natsorted(glob(output_dir + "/snapshot*.hdf5"))
    if not snappaths:
        raise FileNotFoundError(f"No snapshots found in {output_dir}")

    if cache_timeline:
        timelinepath = output_dir + "/.timeline"
        if isfile(timelinepath):  # check if we have a cached timeline file
            times = np.loadtxt(timelinepath)

    with h5py.File(snappaths[0], "r") as F:
        units = get_snapshot_units(F)

    if len(times) < len(snappaths):
        for f in snappaths:
            with h5py.File(f, "r") as F:
                times.append(F["Header"].attrs["Time"])
    if verbose:
        print("Done!")

    if cache_timeline:
        np.savetxt(timelinepath, np.array(times))
    times = np.array(times) * (units["Length"] / units["Speed"]).to(unit)
    return dict(zip(times, snappaths))


def snapdata_to_meshoid(pdata: dict, type=0) -> Meshoid:
    """
    Given snapshot data in the dict format established here, instantiate a Meshoid
    from the particle data of a given type (default: 0)
    """
    ptype = f"PartType{type}"
    return Meshoid(
        pos=pdata[ptype + "/Coordinates"],
        m=(pdata[ptype + "/Masses"] if ptype + "/Masses" in pdata else None),
        kernel_radius=pdata[ptype + "/SmoothingLength"],
        boxsize=pdata["Header"]["BoxSize"],
    )


def snapshot_to_meshoid(snapshot: str, type=0) -> Meshoid:
    """
    Given snapshot data in the dict format established here, instantiate a Meshoid
    from the particle data of a given type (default: 0)
    """
    ptype = f"PartType{type}"
    pdata = get_snapshot_data(
        snapshot, required_data=(ptype + "/Coordinates", ptype + "/Masses", ptype + "/SmoothingLength")
    )
    return snapdata_to_meshoid(pdata)
