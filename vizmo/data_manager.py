"""Load mesh-free simulation snapshots via h5py with lazy field loading."""

import numpy as np
import h5py

# Field name fallbacks for different GIZMO versions
_FIELD_FALLBACKS = {
    "KernelMaxRadius": ["SmoothingLength", "Hsml"],
    "SmoothingLength": ["KernelMaxRadius", "Hsml"],
    "Sink_Mass": ["BH_Mass"],
    "BH_Mass": ["Sink_Mass"],
    "StarLuminosity_Solar": ["StarLuminosity"],
}


def _zams_luminosity(mass_msun):
    """Piecewise ZAMS L(M) in Lsun for main-sequence stars.

    Coefficients from a standard mass-luminosity relation; accurate to
    within a factor of ~2 across 0.1 - 100 Msun and good enough to drive
    the realistic-stars renderer when the snapshot lacks a luminosity
    field.
    """
    m = np.asarray(mass_msun, dtype=np.float32)
    L = np.empty_like(m)
    a = m < 0.43
    b = (m >= 0.43) & (m < 2.0)
    c = (m >= 2.0) & (m < 55.0)
    d = m >= 55.0
    L[a] = 0.23 * m[a] ** 2.3
    L[b] = m[b] ** 4.0
    L[c] = 1.4 * m[c] ** 3.5
    L[d] = 32000.0 * m[d]
    return L


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
            # Prefer Sink_Mass (true accreted mass) for realistic-stars luminosity.
            grp = self._file["PartType5"]
            if "Sink_Mass" in grp:
                self.star_masses = grp["Sink_Mass"][:].astype(np.float32)
            else:
                self.star_masses = self._read_field("PartType5", "Masses").astype(np.float32)
            self.n_stars = len(self.star_masses)
            # Luminosity in Lsun. Use snapshot field if present, else ZAMS L(M).
            if "StarLuminosity_Solar" in grp:
                self.star_luminosity = grp["StarLuminosity_Solar"][:].astype(np.float32)
            elif "StarLuminosity" in grp:
                self.star_luminosity = grp["StarLuminosity"][:].astype(np.float32)
            else:
                self.star_luminosity = _zams_luminosity(self.star_masses)
        else:
            self.star_positions = np.zeros((0, 3), dtype=np.float32)
            self.star_masses = np.zeros(0, dtype=np.float32)
            self.star_luminosity = np.zeros(0, dtype=np.float32)
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

        # Cache as float32 so subsequent get_field/get_vector_field calls
        # don't have to re-cast (Velocities is ~1.5 GB at float32 — doing
        # it twice doubles the cost of every reweight).
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self._cache[key] = data
        return data

    def _cosmo_correct(self, data, ptype, field):
        """Apply cosmological scale factor corrections if applicable."""
        h = self.header
        if not h.get("ComovingIntegrationOn", 0):
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

    # Fields to exclude from the surface density weight dropdown
    _SKIP_FIELDS = {
        "Coordinates",
        "ParticleIDs",
        "ParticleChildIDsNumber",
        "ParticleIDGenerationNumber",
        "PhotonFluxDensity",
    }

    def available_fields(self):
        """List PartType0 scalar fields suitable as surface density weights.

        Scalar fields appear as-is. Multi-column fields (e.g. shape (N,5))
        are expanded to Name[0], Name[1], etc. 3-column vector fields also
        appear as individual components.
        """
        fields = ["Masses"]
        grp = self._file.get("PartType0", {})
        for name in grp:
            if name == "Masses" or name in self._SKIP_FIELDS:
                continue
            ds = grp[name]
            if not hasattr(ds, 'shape') or ds.shape[0] != self.n_particles:
                continue
            if ds.ndim == 1:
                fields.append(name)
            elif ds.ndim == 2:
                for i in range(ds.shape[1]):
                    fields.append(f"{name}[{i}]")
        return fields

    def available_vector_fields(self):
        """List PartType0 fields with shape (N, 3), suitable for LOS projection."""
        fields = []
        grp = self._file.get("PartType0", {})
        for name in grp:
            if name in self._SKIP_FIELDS:
                continue
            ds = grp[name]
            if hasattr(ds, 'shape') and ds.ndim == 2 and ds.shape == (self.n_particles, 3):
                fields.append(name)
        return fields

    def get_field(self, name):
        """Load a raw PartType0 field by name. Returns (N,) float32 array.

        Supports indexed names like 'PhotonEnergy[3]' for multi-column fields.
        """
        if "[" in name and name.endswith("]"):
            base, idx_str = name[:-1].split("[", 1)
            col = int(idx_str)
            data = self._read_field("PartType0", base)
            return np.ascontiguousarray(data[:, col])
        return self._read_field("PartType0", name)

    def get_vector_field(self, name):
        """Load a raw PartType0 vector field. Returns (N, 3) float32 array."""
        return self._read_field("PartType0", name)

    def close(self):
        self._file.close()
        self._cache.clear()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass
