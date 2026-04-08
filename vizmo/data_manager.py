"""Load mesh-free simulation snapshots via h5py with lazy field loading."""

import numpy as np
import h5py

# Field name fallbacks for different GIZMO versions
_FIELD_FALLBACKS = {
    "KernelMaxRadius": ["SmoothingLength", "Hsml"],
    "SmoothingLength": ["KernelMaxRadius", "Hsml"],
    "Hsml": ["KernelMaxRadius", "SmoothingLength"],
    "Sink_Mass": ["BH_Mass"],
    "BH_Mass": ["Sink_Mass"],
    "StarLuminosity_Solar": ["StarLuminosity"],
}


def _zams_luminosity(mass_msun):
    """Piecewise ZAMS L(M) in Lsun for main-sequence stars."""
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


def _hsml_field_names():
    return ("SmoothingLength", "KernelMaxRadius", "Hsml")


def _compute_hsml_kdtree(positions, boxsize=None, n_neighbors=50,
                         des_ngb=32, chunk_size=10_000_000):
    """Compute SPH-style smoothing lengths via KDTree + meshoid.HsmlIter.

    Builds a single periodic-aware cKDTree over `positions`, then queries
    `n_neighbors` nearest neighbors in chunks of at most `chunk_size`
    particles in parallel (`cKDTree.query(workers=-1)`) and feeds each
    chunk's neighbor distances to `meshoid.HsmlIter` to get the iterated
    smoothing length per particle.
    """
    import time
    from scipy.spatial import cKDTree
    from meshoid import HsmlIter

    x = np.ascontiguousarray(positions, dtype=np.float64)
    n = len(x)

    # cKDTree's `boxsize` arg requires positions in [0, L); fall back to
    # the open-domain tree if anything is outside, since scipy raises.
    bs_arg = None
    if boxsize is not None:
        bs = float(np.asarray(boxsize).ravel()[0])
        if bs > 0 and x.min() >= 0 and x.max() < bs:
            bs_arg = bs

    t0 = time.perf_counter()
    tree = cKDTree(x, boxsize=bs_arg)
    print(f"  KDTree built in {time.perf_counter()-t0:.1f}s")

    h = np.empty(n, dtype=np.float32)
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        a = ci * chunk_size
        b = min(a + chunk_size, n)
        t0 = time.perf_counter()
        dists, _ = tree.query(x[a:b], k=n_neighbors, workers=-1)
        h[a:b] = np.asarray(
            HsmlIter(dists, des_ngb=des_ngb, dim=3, error_norm=1e-2),
            dtype=np.float32,
        )
        print(
            f"  HsmlIter chunk {ci+1}/{n_chunks} ({b-a:,} parts) "
            f"in {time.perf_counter()-t0:.1f}s"
        )
    return h


class SnapshotData:
    """Manages particle data from an HDF5 snapshot with lazy field loading.

    Particles from the selected `particle_types` (default: PartType0 only)
    are concatenated into a single pool exposed as `positions`, `masses`,
    and `hsml`. Per-type slices are kept in `_type_slices`. Star data
    (PartType5 as a separate point-source pool) is loaded independently
    of the particle-type selection.
    """

    # Fields to exclude from the surface density weight dropdown
    _SKIP_FIELDS = {
        "Coordinates",
        "ParticleIDs",
        "ParticleChildIDsNumber",
        "ParticleIDGenerationNumber",
        "PhotonFluxDensity",
    }

    def __init__(self, path, particle_types=None, hsml_progress=None):
        self.path = path
        self._file = h5py.File(path, "r")
        self.header = dict(self._file["Header"].attrs)
        self.time = float(self.header.get("Time", 0))

        # Discover available particle types in this snapshot
        avail = []
        for k in self._file.keys():
            if not k.startswith("PartType"):
                continue
            grp = self._file[k]
            if isinstance(grp, h5py.Group) and "Coordinates" in grp:
                avail.append(int(k.replace("PartType", "")))
        self.available_types = sorted(avail)

        # Default selection: gas only if present, else first available type
        if particle_types is None:
            if 0 in self.available_types:
                particle_types = [0]
            elif self.available_types:
                particle_types = [self.available_types[0]]
            else:
                particle_types = []

        # Stars (PartType5) loaded independently as point-source pool
        self._stars = "PartType5" in self._file
        self._cache = {}
        # Per-ptype computed hsml cache (only fills for types whose hsml
        # had to be computed via the KDTree fallback — cheap to keep).
        self._hsml_cache = {}
        # LRU recency order of cached ptypes. Ptypes appearing later are
        # more recently used; eviction pops from the front.
        from collections import OrderedDict
        self._ptype_lru = OrderedDict()
        # Memory budget for the field/hsml cache, in bytes. Defaults to
        # half of system free memory if psutil is available, else 16 GB.
        self._cache_budget = self._default_cache_budget()

        self._load_stars()
        self._hsml_progress = hsml_progress
        self.set_particle_types(particle_types)

    @staticmethod
    def _default_cache_budget():
        try:
            import psutil
            return int(psutil.virtual_memory().available * 0.5)
        except Exception:
            return 16 * 1024**3

    def _cache_bytes(self):
        total = 0
        for v in self._cache.values():
            total += getattr(v, "nbytes", 0)
        for v in self._hsml_cache.values():
            total += getattr(v, "nbytes", 0)
        return total

    def _estimate_ptype_bytes(self, p):
        """Rough lower bound on the bytes a freshly-loaded ptype will add."""
        ptype = f"PartType{p}"
        grp = self._file.get(ptype)
        if grp is None or "Coordinates" not in grp:
            return 0
        n = grp["Coordinates"].shape[0]
        # Coordinates (f64), Masses (f32), Hsml (f32) ~ 32 bytes/particle
        return n * 32

    def _evict_for(self, p_new):
        """Evict LRU non-selected ptypes until we expect to fit p_new."""
        need = self._estimate_ptype_bytes(p_new)
        budget = self._cache_budget
        protected = set(getattr(self, "particle_types", []))
        protected.add(p_new)
        # Walk LRU front-to-back, evicting evictable entries.
        for victim in list(self._ptype_lru.keys()):
            if self._cache_bytes() + need <= budget:
                return
            if victim in protected:
                continue
            self._evict_ptype(victim)

    def _evict_ptype(self, p):
        prefix = f"PartType{p}/"
        for key in [k for k in self._cache if k.startswith(prefix)]:
            del self._cache[key]
        self._hsml_cache.pop(p, None)
        self._ptype_lru.pop(p, None)

    def _touch_ptype(self, p):
        self._ptype_lru.pop(p, None)
        self._ptype_lru[p] = None

    def _load_stars(self):
        if self._stars:
            self.star_positions = self._read_field("PartType5", "Coordinates").astype(np.float32)
            grp = self._file["PartType5"]
            if "Sink_Mass" in grp:
                self.star_masses = grp["Sink_Mass"][:].astype(np.float32)
            else:
                self.star_masses = self._read_field("PartType5", "Masses").astype(np.float32)
            self.n_stars = len(self.star_masses)
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

    def set_particle_types(self, particle_types):
        """(Re)load the particle pool from the given list of PartType ints."""
        self.particle_types = [int(p) for p in particle_types if int(p) in self.available_types]

        # The per-ptype field cache (`_cache`) and computed-hsml cache
        # (`_hsml_cache`) are keyed by ptype, not by the active selection,
        # so they survive a particle-type swap. Toggling a tickbox off
        # then back on doesn't re-read the file or recompute hsml — the
        # entries are just reused. The LOS projection cache is keyed by
        # the concatenated array length and must be dropped.
        if hasattr(self, "_projected_cache"):
            self._projected_cache = {}

        if not self.particle_types:
            self.positions = np.zeros((0, 3), dtype=np.float64)
            self.masses = np.zeros(0, dtype=np.float32)
            self.hsml = np.zeros(0, dtype=np.float32)
            self.n_particles = 0
            self._type_slices = {}
            return

        pos_chunks = []
        mass_chunks = []
        hsml_chunks = []
        slices = {}
        offset = 0
        for p in self.particle_types:
            # Make room before pulling this ptype off disk if it's not
            # already cached. Selected ptypes (including this one) are
            # protected from eviction.
            self._evict_for(p)
            ptype = f"PartType{p}"
            pos = self._read_field(ptype, "Coordinates")
            mass = self._read_field(ptype, "Masses").astype(np.float32)
            hsml = self._resolve_hsml(ptype, pos)
            self._touch_ptype(p)
            n = len(mass)
            pos_chunks.append(pos)
            mass_chunks.append(mass)
            hsml_chunks.append(hsml)
            slices[p] = slice(offset, offset + n)
            offset += n

        self.positions = (
            np.concatenate(pos_chunks, axis=0)
            if len(pos_chunks) > 1
            else pos_chunks[0]
        )
        self.masses = (
            np.concatenate(mass_chunks) if len(mass_chunks) > 1 else mass_chunks[0]
        )
        self.hsml = (
            np.concatenate(hsml_chunks) if len(hsml_chunks) > 1 else hsml_chunks[0]
        )
        self.n_particles = offset
        self._type_slices = slices

    def _resolve_hsml(self, ptype, pos):
        """Get smoothing lengths for `ptype`, with header softening + meshoid fallbacks."""
        p = int(ptype.replace("PartType", ""))
        cached = self._hsml_cache.get(p)
        if cached is not None and len(cached) == len(pos):
            return cached
        grp = self._file[ptype]
        for name in _hsml_field_names():
            if name in grp:
                h = self._read_field(ptype, name).astype(np.float32)
                self._hsml_cache[p] = h
                return h

        # Try header per-type softening (Gadget-style: SofteningTypeN /
        # SofteningComovingTypeN / SofteningTable)
        for key in (
            f"Softening_Type{p}",
            f"SofteningType{p}",
            f"SofteningComovingType{p}",
            f"Softening_Type_{p}",
        ):
            if key in self.header:
                soft = float(self.header[key])
                if soft > 0:
                    h = np.full(len(pos), soft, dtype=np.float32)
                    self._hsml_cache[p] = h
                    return h
        if "SofteningTable" in self.header:
            tbl = np.asarray(self.header["SofteningTable"]).ravel()
            if p < len(tbl) and float(tbl[p]) > 0:
                h = np.full(len(pos), float(tbl[p]), dtype=np.float32)
                self._hsml_cache[p] = h
                return h

        # Last resort: KDTree + meshoid.HsmlIter, computed in parallel chunks.
        msg = f"Computing kernel radii for PartType{p}..."
        print(f"  {msg}  ({len(pos):,} particles)")
        if self._hsml_progress is not None:
            try:
                self._hsml_progress(msg)
            except Exception:
                pass
        boxsize = self.header.get("BoxSize", None)
        h = _compute_hsml_kdtree(pos, boxsize)
        self._hsml_cache[p] = h
        return h

    def _read_field(self, ptype, field):
        """Read a field with fallback name resolution."""
        key = f"{ptype}/{field}"
        if key in self._cache:
            return self._cache[key]

        grp = self._file[ptype]
        if field in grp:
            data = grp[field][:]
        else:
            found = False
            for fallback in _FIELD_FALLBACKS.get(field, []):
                if fallback in grp:
                    data = grp[fallback][:]
                    found = True
                    break
            if not found:
                raise KeyError(f"Field {field} (and fallbacks) not found in {ptype}")

        data = self._cosmo_correct(data, ptype, field)

        if field != "Coordinates" and data.dtype != np.float32:
            data = data.astype(np.float32)

        self._cache[key] = data
        return data

    def _cosmo_correct(self, data, ptype, field):
        h = self.header
        if not h.get("ComovingIntegrationOn", 0):
            return data
        a = float(h.get("Time", 1))
        hubble = float(h["HubbleParam"])
        if field == "Coordinates":
            data = data * a / hubble
        elif field == "Masses":
            data = data / hubble
        elif field in ("KernelMaxRadius", "SmoothingLength", "Hsml"):
            data = data * a / hubble
        return data

    def _ptype_scalar_fields(self, p):
        """Set of scalar/per-component field names available for PartType<p>.

        Multi-column fields are expanded to Name[i]. Includes synthetic
        "Masses" since it's always exposed.
        """
        ptype = f"PartType{p}"
        grp = self._file.get(ptype)
        out = {"Masses"}
        if grp is None:
            return out
        n = grp["Masses"].shape[0] if "Masses" in grp else None
        for name in grp:
            if name in self._SKIP_FIELDS:
                continue
            ds = grp[name]
            if not hasattr(ds, "shape"):
                continue
            if n is not None and ds.shape[0] != n:
                continue
            if ds.ndim == 1:
                out.add(name)
            elif ds.ndim == 2:
                for i in range(ds.shape[1]):
                    out.add(f"{name}[{i}]")
        return out

    def _ptype_vector_fields(self, p):
        ptype = f"PartType{p}"
        grp = self._file.get(ptype)
        out = set()
        if grp is None:
            return out
        n = grp["Masses"].shape[0] if "Masses" in grp else None
        for name in grp:
            if name in self._SKIP_FIELDS:
                continue
            ds = grp[name]
            if not hasattr(ds, "shape") or ds.ndim != 2:
                continue
            if n is not None and ds.shape == (n, 3):
                out.add(name)
        return out

    def available_fields(self):
        """Scalar fields common to ALL currently-selected particle types."""
        if not self.particle_types:
            return ["Masses"]
        sets = [self._ptype_scalar_fields(p) for p in self.particle_types]
        common = set.intersection(*sets) if sets else set()
        common.add("Masses")
        # Stable order: put Masses first, then sorted remainder
        rest = sorted(common - {"Masses"})
        return ["Masses"] + rest

    def available_vector_fields(self):
        """Vector fields common to ALL currently-selected particle types."""
        if not self.particle_types:
            return []
        sets = [self._ptype_vector_fields(p) for p in self.particle_types]
        common = set.intersection(*sets) if sets else set()
        return sorted(common)

    def _get_field_concat(self, field_name):
        """Read `field_name` from every selected ptype and concatenate."""
        if not self.particle_types:
            return np.zeros(0, dtype=np.float32)
        if "[" in field_name and field_name.endswith("]"):
            base, idx_str = field_name[:-1].split("[", 1)
            col = int(idx_str)
            chunks = []
            for p in self.particle_types:
                d = self._read_field(f"PartType{p}", base)
                chunks.append(np.ascontiguousarray(d[:, col]))
            return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        chunks = [self._read_field(f"PartType{p}", field_name) for p in self.particle_types]
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    def get_field(self, name):
        """Load a scalar field across all selected particle types. (N,) float32."""
        return self._get_field_concat(name)

    def get_vector_field(self, name):
        """Load a vector (N,3) field across all selected particle types."""
        chunks = [self._read_field(f"PartType{p}", name) for p in self.particle_types]
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def close(self):
        self._file.close()
        self._cache.clear()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass
