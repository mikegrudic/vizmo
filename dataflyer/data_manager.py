"""Load SPH simulation snapshots via h5py with lazy field loading."""

import os
import re
import numpy as np
import h5py
from natsort import natsorted

# Field name fallbacks for different GIZMO versions
_FIELD_FALLBACKS = {
    "KernelMaxRadius": ["SmoothingLength", "Hsml"],
    "SmoothingLength": ["KernelMaxRadius", "Hsml"],
    "Sink_Mass": ["BH_Mass"],
    "BH_Mass": ["Sink_Mass"],
    "StarLuminosity_Solar": ["StarLuminosity"],
}


class SnapshotData:
    """Manages particle data from an HDF5 snapshot with lazy field loading."""

    def __init__(self, path):
        self.path = path
        self._file = h5py.File(path, "r")
        self.header = dict(self._file["Header"].attrs)
        self.time = float(self.header.get("Time", 0))

        # Determine available particle types
        self._gas = "PartType0" in self._file
        self._stars = "PartType5" in self._file

        # Cache for loaded fields
        self._cache = {}

        # Eagerly load core gas fields
        if self._gas:
            self.positions = self._read_field("PartType0", "Coordinates").astype(np.float32)
            self.masses = self._read_field("PartType0", "Masses").astype(np.float32)
            self.hsml = self._read_field("PartType0", "KernelMaxRadius").astype(np.float32)
            self.n_particles = len(self.masses)
        else:
            self.positions = np.zeros((0, 3), dtype=np.float32)
            self.masses = np.zeros(0, dtype=np.float32)
            self.hsml = np.zeros(0, dtype=np.float32)
            self.n_particles = 0

        # Star data (eagerly loaded, usually small)
        if self._stars:
            self.star_positions = self._read_field("PartType5", "Coordinates").astype(np.float32)
            self.star_masses = self._read_field("PartType5", "Masses").astype(np.float32)
            self.n_stars = len(self.star_masses)
        else:
            self.star_positions = np.zeros((0, 3), dtype=np.float32)
            self.star_masses = np.zeros(0, dtype=np.float32)
            self.n_stars = 0

    def _read_field(self, ptype, field):
        """Read a field with fallback name resolution."""
        key = f"{ptype}/{field}"
        if key in self._cache:
            return self._cache[key]

        grp = self._file[ptype]
        if field in grp:
            data = grp[field][:]
        else:
            # Try fallbacks
            found = False
            for fallback in _FIELD_FALLBACKS.get(field, []):
                if fallback in grp:
                    data = grp[fallback][:]
                    found = True
                    break
            if not found:
                raise KeyError(f"Field {field} (and fallbacks) not found in {ptype}")

        # Apply cosmological corrections if needed
        data = self._cosmo_correct(data, ptype, field)

        self._cache[key] = data
        return data

    def _cosmo_correct(self, data, ptype, field):
        """Apply cosmological scale factor corrections if applicable."""
        h = self.header
        if "HubbleParam" not in h or h.get("Time", 1) == h.get("Redshift", 0):
            return data  # Not cosmological

        a = float(h.get("Time", 1))
        hubble = float(h["HubbleParam"])

        if field == "Coordinates":
            data = data * a / hubble
        elif field == "Masses":
            data = data / hubble
        elif field in ("KernelMaxRadius", "SmoothingLength", "Hsml"):
            data = data * a / hubble

        return data

    def get_quantity(self, name):
        """Load or compute a named physical quantity. Returns (N,) float32 array."""
        if name == "surface_density":
            return self.masses  # Surface density is just mass splatted

        if name == "temperature":
            grp = self._file.get("PartType0", {})
            if "Temperature" in grp:
                return self._read_field("PartType0", "Temperature").astype(np.float32)
            # Estimate from InternalEnergy: T ~ (gamma-1) * mu * u / k_B
            # For atomic hydrogen: mu ~ 1.23 m_p, gamma = 5/3
            u = self._read_field("PartType0", "InternalEnergy").astype(np.float64)
            # u is in code units (velocity^2). Convert to K assuming molecular weight ~0.6
            # T = (gamma-1) * mu_mol * m_p * u / k_B
            # With mu=0.6, gamma=5/3: T ~ 0.6 * 1.67e-24 * u * (UnitVelocity)^2 / 1.38e-16 * (2/3)
            unit_v = float(self.header.get("UnitVelocity_In_CGS", 1e5))
            mu = 0.6
            mp = 1.6726e-24  # g
            kB = 1.3807e-16  # erg/K
            gamma = 5.0 / 3.0
            T = (gamma - 1) * mu * mp * u * unit_v**2 / kB
            return T.astype(np.float32)

        if name == "density":
            return self._read_field("PartType0", "Density").astype(np.float32)

        if name == "velocity_magnitude":
            vel = self._read_field("PartType0", "Velocities").astype(np.float32)
            return np.linalg.norm(vel, axis=1).astype(np.float32)

        if name == "velocity_z":
            vel = self._read_field("PartType0", "Velocities").astype(np.float32)
            return vel[:, 2].copy()

        if name == "internal_energy":
            return self._read_field("PartType0", "InternalEnergy").astype(np.float32)

        raise KeyError(f"Unknown quantity: {name}")

    def available_quantities(self):
        """List quantities that can be loaded from this snapshot."""
        result = ["surface_density"]
        grp = self._file.get("PartType0", {})
        if "Temperature" in grp or "InternalEnergy" in grp:
            result.append("temperature")
        if "Density" in grp:
            result.append("density")
        if "Velocities" in grp:
            result.extend(["velocity_magnitude", "velocity_z"])
        if "InternalEnergy" in grp:
            result.append("internal_energy")
        return result

    def close(self):
        self._file.close()
        self._cache.clear()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass


def find_snapshots(path):
    """Given a snapshot path, find all sibling snapshots in the same directory.
    Returns a sorted list of snapshot paths."""
    dirpath = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path)

    # Extract the naming pattern: snapshot_NNN.hdf5, snap_NNN.hdf5, etc.
    match = re.match(r"^(.*?)(\d+)(\.hdf5)$", basename, re.IGNORECASE)
    if not match:
        return [path]

    prefix, _, suffix = match.groups()
    pattern = re.compile(re.escape(prefix) + r"\d+" + re.escape(suffix), re.IGNORECASE)

    snaps = [
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if pattern.match(f)
    ]
    return natsorted(snaps)
