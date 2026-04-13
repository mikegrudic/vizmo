"""Load mesh-free simulation snapshots via h5py with lazy field loading."""

import os
import re
import glob
import numpy as np
import h5py


def _part_index(p):
    m = re.match(r".+\.(\d+)\.hdf5$", os.path.basename(p))
    return int(m.group(1)) if m else 0


def _resolve_snapshot_parts(path):
    """Return sorted list of HDF5 files for a snapshot.

    Accepts a single .hdf5 file, one part of a multipart snapshot
    (e.g. snapshot_XXX.0.hdf5), or a directory containing parts
    (e.g. snapdir_XXX/).
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.hdf5")), key=_part_index)
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.athdf")))
        if not files:
            raise FileNotFoundError(f"No .hdf5/.athdf files in directory {path}")
        return files
    if os.path.isfile(path):
        base = os.path.basename(path)
        m = re.match(r"(.+)\.(\d+)\.hdf5$", base)
        if m:
            stem = m.group(1)
            d = os.path.dirname(path) or "."
            files = sorted(
                glob.glob(os.path.join(d, f"{stem}.*.hdf5")), key=_part_index
            )
            if files:
                return files
        return [path]
    raise FileNotFoundError(path)


class _MultiDataset:
    """Concatenated view of one field across multiple HDF5 part files."""

    def __init__(self, files, ptype, name):
        self._files = files
        self._ptype = ptype
        self._name = name
        first_shape = None
        first_dtype = None
        total_n = 0
        for f in files:
            grp = f.get(ptype)
            if grp is None or name not in grp:
                continue
            ds = grp[name]
            if first_shape is None:
                first_shape = ds.shape
                first_dtype = ds.dtype
            total_n += ds.shape[0]
        if first_shape is None:
            raise KeyError(name)
        self.shape = (total_n,) + tuple(first_shape[1:])
        self.ndim = len(self.shape)
        self.dtype = first_dtype

    def _concat(self):
        chunks = []
        for f in self._files:
            grp = f.get(self._ptype)
            if grp is None or self._name not in grp:
                continue
            chunks.append(grp[self._name][:])
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

    def __getitem__(self, key):
        return self._concat()[key]

    def __array__(self, dtype=None):
        a = self._concat()
        return a if dtype is None else a.astype(dtype)


class _MultiGroup:
    """Union view of one PartType group across multiple HDF5 part files."""

    def __init__(self, files, ptype):
        self._files = files
        self._ptype = ptype
        names = set()
        for f in files:
            g = f.get(ptype)
            if g is not None:
                names.update(g.keys())
        self._names = names

    def __contains__(self, name):
        return name in self._names

    def __iter__(self):
        return iter(sorted(self._names))

    def keys(self):
        return sorted(self._names)

    def __getitem__(self, name):
        if name not in self._names:
            raise KeyError(name)
        return _MultiDataset(self._files, self._ptype, name)

    def get(self, name, default=None):
        return self[name] if name in self._names else default


class _MemDataset:
    """Wrap an in-memory ndarray with the bits of the h5py.Dataset API used here."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.ndim = arr.ndim
        self.dtype = arr.dtype

    def __getitem__(self, key):
        return self._arr[key]

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)


class _MemGroup:
    def __init__(self, fields):
        self._fields = {k: _MemDataset(v) for k, v in fields.items()}

    def __contains__(self, name):
        return name in self._fields

    def __iter__(self):
        return iter(self._fields)

    def keys(self):
        return list(self._fields.keys())

    def __getitem__(self, name):
        return self._fields[name]

    def get(self, name, default=None):
        return self._fields.get(name, default)


class _HeaderGroup:
    def __init__(self, attrs):
        self.attrs = attrs


def _decode_str_list(arr):
    out = []
    for v in np.array(arr).ravel():
        if isinstance(v, bytes):
            out.append(v.decode())
        else:
            out.append(str(v))
    return out


def _read_athdf(path):
    """Flatten an Athena++ .athdf file into (header_attrs, PartType0_fields).

    Each mesh-block cell becomes one "particle" with position = cell center
    (converted to Cartesian xyz when needed), Masses = density * cell volume,
    SmoothingLength = volume^(1/3), and every variable in VariableNames is
    exposed as an additional scalar field. vel1/vel2/vel3 are bundled into a
    Cartesian Velocities (N,3) field.
    """
    f = h5py.File(path, "r")
    try:
        a = f.attrs
        nblocks = int(np.asarray(a["NumMeshBlocks"]).ravel()[0])
        mbsize = np.asarray(a["MeshBlockSize"]).astype(int).ravel()
        nx1, nx2, nx3 = int(mbsize[0]), int(mbsize[1]), int(mbsize[2])
        coord_sys = a["Coordinates"]
        if isinstance(coord_sys, bytes):
            coord_sys = coord_sys.decode()
        coord_sys = str(coord_sys)
        time = float(np.asarray(a.get("Time", 0.0)).ravel()[0])
        var_names = _decode_str_list(a["VariableNames"])
        ds_names = _decode_str_list(a["DatasetNames"])
        num_vars = np.asarray(a["NumVariables"]).astype(int).ravel()

        # var_name -> (dataset_name, index_within_dataset)
        var_map = {}
        offset = 0
        for ds, nv in zip(ds_names, num_vars):
            for i in range(int(nv)):
                var_map[var_names[offset + i]] = (ds, i)
            offset += int(nv)

        x1f = f["x1f"][:]  # (nblocks, nx1+1)
        x2f = f["x2f"][:]
        x3f = f["x3f"][:]

        x1c = 0.5 * (x1f[:, :-1] + x1f[:, 1:])
        x2c = 0.5 * (x2f[:, :-1] + x2f[:, 1:])
        x3c = 0.5 * (x3f[:, :-1] + x3f[:, 1:])
        dx1 = x1f[:, 1:] - x1f[:, :-1]
        dx2 = x2f[:, 1:] - x2f[:, :-1]
        dx3 = x3f[:, 1:] - x3f[:, :-1]

        shape4 = (nblocks, nx3, nx2, nx1)
        c1 = np.broadcast_to(x1c[:, None, None, :], shape4).reshape(-1)
        c2 = np.broadcast_to(x2c[:, None, :, None], shape4).reshape(-1)
        c3 = np.broadcast_to(x3c[:, :, None, None], shape4).reshape(-1)
        d1 = np.broadcast_to(dx1[:, None, None, :], shape4).reshape(-1)
        d2 = np.broadcast_to(dx2[:, None, :, None], shape4).reshape(-1)
        d3 = np.broadcast_to(dx3[:, :, None, None], shape4).reshape(-1)
        c1 = np.ascontiguousarray(c1)
        c2 = np.ascontiguousarray(c2)
        c3 = np.ascontiguousarray(c3)
        d1 = np.ascontiguousarray(d1)
        d2 = np.ascontiguousarray(d2)
        d3 = np.ascontiguousarray(d3)

        if coord_sys == "cartesian":
            x, y, z = c1, c2, c3
            vol = d1 * d2 * d3
        elif coord_sys == "cylindrical":
            R, phi, zc = c1, c2, c3
            r1 = R - 0.5 * d1
            r2 = R + 0.5 * d1
            vol = 0.5 * (r2 * r2 - r1 * r1) * d2 * d3
            x = R * np.cos(phi)
            y = R * np.sin(phi)
            z = zc
        elif coord_sys.startswith("spherical"):
            r, th, ph = c1, c2, c3
            r1 = r - 0.5 * d1
            r2 = r + 0.5 * d1
            t1 = th - 0.5 * d2
            t2 = th + 0.5 * d2
            vol = (1.0 / 3.0) * (r2**3 - r1**3) * (np.cos(t1) - np.cos(t2)) * d3
            sth = np.sin(th)
            x = r * sth * np.cos(ph)
            y = r * sth * np.sin(ph)
            z = r * np.cos(th)
        else:
            print(f"  Warning: unknown Athena coordinate system {coord_sys!r}, "
                  "treating as Cartesian")
            x, y, z = c1, c2, c3
            vol = d1 * d2 * d3

        positions = np.stack(
            [x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)],
            axis=1,
        )
        hsml = np.cbrt(vol).astype(np.float32)

        fields = {}
        for ds in ds_names:
            arr = f[ds][:]  # (nv_in_ds, nblocks, nx3, nx2, nx1)
            for vname, (vds, vi) in var_map.items():
                if vds != ds:
                    continue
                fields[vname] = np.ascontiguousarray(
                    arr[vi].reshape(-1)
                ).astype(np.float32)

        out = {"Coordinates": positions, "SmoothingLength": hsml}

        rho_name = next(
            (n for n in ("rho", "dens", "density") if n in fields), None
        )
        if rho_name is not None:
            density = fields[rho_name]
            out["Density"] = density
            out["Masses"] = (density * vol).astype(np.float32)
        else:
            out["Masses"] = vol.astype(np.float32)

        if all(f"vel{i}" in fields for i in (1, 2, 3)):
            v1 = fields["vel1"]
            v2 = fields["vel2"]
            v3 = fields["vel3"]
            if coord_sys == "cylindrical":
                cphi = np.cos(c2)
                sphi = np.sin(c2)
                vx = v1 * cphi - v2 * sphi
                vy = v1 * sphi + v2 * cphi
                vz = v3
            elif coord_sys.startswith("spherical"):
                sth = np.sin(c2)
                cth = np.cos(c2)
                cph = np.cos(c3)
                sph = np.sin(c3)
                vx = v1 * sth * cph + v2 * cth * cph - v3 * sph
                vy = v1 * sth * sph + v2 * cth * sph + v3 * cph
                vz = v1 * cth - v2 * sth
            else:
                vx, vy, vz = v1, v2, v3
            out["Velocities"] = np.stack(
                [vx.astype(np.float32), vy.astype(np.float32), vz.astype(np.float32)],
                axis=1,
            )

        skip = {"rho", "dens", "density", "vel1", "vel2", "vel3"}
        rename = {"press": "Pressure"}
        for vname, varr in fields.items():
            if vname in skip:
                continue
            out[rename.get(vname, vname)] = varr

        # Header attrs: Time is the only field SnapshotData truly needs;
        # include a BoxSize derived from RootGridX1/2/3 for downstream sizing.
        try:
            extents = []
            for k in ("RootGridX1", "RootGridX2", "RootGridX3"):
                rg = np.asarray(a[k]).ravel()
                extents.append(float(rg[1] - rg[0]))
            box = float(max(extents))
        except Exception:
            box = 0.0
        header_attrs = {"Time": time, "BoxSize": box}
        return header_attrs, out
    finally:
        f.close()


class _AthdfFile:
    """h5py.File-like wrapper that exposes a flattened .athdf as PartType0."""

    def __init__(self, path):
        header_attrs, fields = _read_athdf(path)
        self._groups = {
            "Header": _HeaderGroup(header_attrs),
            "PartType0": _MemGroup(fields),
        }

    def keys(self):
        return list(self._groups.keys())

    def __contains__(self, key):
        return key in self._groups

    def __getitem__(self, key):
        return self._groups[key]

    def get(self, key, default=None):
        return self._groups.get(key, default)

    def close(self):
        pass


def _is_flash_file(path):
    """Return True if *path* is a FLASH HDF5 file (plotfile or checkpoint)."""
    try:
        with h5py.File(path, "r") as f:
            return "coordinates" in f and "block size" in f and "node type" in f
    except Exception:
        return False


def _read_flash(path):
    """Flatten a FLASH AMR HDF5 file into (header_attrs, PartType0_fields).

    Only leaf blocks (node type == 1) are included. Each cell becomes one
    "particle" with position = cell centre, Masses = density * cell volume,
    SmoothingLength = volume^(1/3), and every variable stored in the file is
    exposed as a scalar field.
    """
    f = h5py.File(path, "r")
    try:
        # -- block metadata --------------------------------------------------
        coords = f["coordinates"][:]        # (nblocks, ndim)
        blk_size = f["block size"][:]       # (nblocks, ndim)
        node_type = f["node type"][:].ravel()
        leaf = node_type == 1
        nblocks_all = len(node_type)

        # Determine grid dimensions per block from the first field dataset.
        # FLASH stores fields as (nblocks, nzb, nyb, nxb).
        # Identify field datasets: same leading dim as nblocks_all and 4-D.
        # "unknown names" lists the plotfile variables; fall back to scanning.
        if "unknown names" in f:
            raw = f["unknown names"][:]
            var_names = []
            for row in raw:
                if isinstance(row, (bytes, np.bytes_)):
                    var_names.append(row.decode().strip())
                else:
                    var_names.append(
                        b"".join(row).decode().strip()
                        if hasattr(row, "__iter__")
                        else str(row).strip()
                    )
        else:
            # Guess field datasets: 4-D arrays with first axis == nblocks_all
            var_names = []
            for name in f:
                ds = f[name]
                if hasattr(ds, "ndim") and ds.ndim == 4 and ds.shape[0] == nblocks_all:
                    var_names.append(name)

        if not var_names:
            raise ValueError(f"No field variables found in FLASH file {path}")

        # Read the shape from the first recognised field
        first_ds = f[var_names[0]]
        _, nzb, nyb, nxb = first_ds.shape

        ndim = coords.shape[1]

        # -- cell positions ---------------------------------------------------
        # Build cell-centre positions for leaf blocks only.
        leaf_idx = np.where(leaf)[0]
        n_leaf = len(leaf_idx)
        lcoords = coords[leaf_idx].astype(np.float64)    # (n_leaf, ndim)
        lsize = blk_size[leaf_idx].astype(np.float64)  # (n_leaf, ndim)

        # fractional cell offsets inside a block, in [-0.5, 0.5)
        fx = (np.arange(nxb) + 0.5) / nxb - 0.5
        fy = (np.arange(nyb) + 0.5) / nyb - 0.5
        fz = (np.arange(nzb) + 0.5) / nzb - 0.5

        # shape: (n_leaf, nzb, nyb, nxb)
        shape4 = (n_leaf, nzb, nyb, nxb)

        if ndim >= 1:
            cx = lcoords[:, 0, None, None, None] + lsize[:, 0, None, None, None] * fx[None, None, None, :]
        if ndim >= 2:
            cy = lcoords[:, 1, None, None, None] + lsize[:, 1, None, None, None] * fy[None, None, :, None]
        else:
            cy = np.zeros(shape4)
        if ndim >= 3:
            cz = lcoords[:, 2, None, None, None] + lsize[:, 2, None, None, None] * fz[None, :, None, None]
        else:
            cz = np.zeros(shape4)

        cx = np.ascontiguousarray(np.broadcast_to(cx, shape4).reshape(-1))
        cy = np.ascontiguousarray(np.broadcast_to(cy, shape4).reshape(-1))
        cz = np.ascontiguousarray(np.broadcast_to(cz, shape4).reshape(-1))
        positions = np.stack(
            [cx.astype(np.float64), cy.astype(np.float64), cz.astype(np.float64)],
            axis=1,
        )

        # Cell volume
        dvol = (lsize[:, 0] / nxb) * (lsize[:, 1] / nyb if ndim >= 2 else 1.0) * (lsize[:, 2] / nzb if ndim >= 3 else 1.0)
        vol = np.broadcast_to(dvol[:, None, None, None], shape4).reshape(-1)
        hsml = np.cbrt(vol)

        # -- fields -----------------------------------------------------------
        fields = {}
        for vname in var_names:
            ds = f.get(vname)
            if ds is None or ds.ndim != 4 or ds.shape[0] != nblocks_all:
                continue
            arr = ds[:][leaf_idx]  # (n_leaf, nzb, nyb, nxb)
            fields[vname.strip()] = np.ascontiguousarray(arr.reshape(-1)).astype(np.float64)

        # -- assemble output --------------------------------------------------
        out = {"Coordinates": positions, "SmoothingLength": hsml}

        # density -> Masses
        dens_name = None
        for candidate in ("dens", "density", "Density", "rho"):
            if candidate in fields:
                dens_name = candidate
                break
        if dens_name is not None:
            density = fields[dens_name]
            out["Density"] = density
            out["Masses"] = density * vol
        else:
            out["Masses"] = vol.copy()

        # Velocities
        vx_name = vy_name = vz_name = None
        for vx_c, vy_c, vz_c in [("velx", "vely", "velz"),
                                   ("vel_x", "vel_y", "vel_z")]:
            if vx_c in fields and vy_c in fields and vz_c in fields:
                vx_name, vy_name, vz_name = vx_c, vy_c, vz_c
                break
        if vx_name is not None:
            out["Velocities"] = np.stack(
                [fields[vx_name], fields[vy_name], fields[vz_name]], axis=1
            )

        # Expose remaining fields, skipping ones already consumed
        consumed = {dens_name, vx_name, vy_name, vz_name} - {None}
        rename = {"pres": "Pressure", "temp": "Temperature", "magx": "MagneticField[0]",
                  "magy": "MagneticField[1]", "magz": "MagneticField[2]"}
        for vname, varr in fields.items():
            if vname in consumed:
                continue
            out[rename.get(vname, vname)] = varr

        # -- header -----------------------------------------------------------
        time = 0.0
        if "real scalars" in f:
            for row in f["real scalars"][:]:
                name_bytes = row[0]
                if isinstance(name_bytes, bytes):
                    nm = name_bytes.decode().strip()
                else:
                    nm = str(name_bytes).strip()
                if nm == "time":
                    time = float(row[1])
                    break
        elif "sim info" in f:
            si = f["sim info"]
            if "time" in si.attrs:
                time = float(si.attrs["time"])

        # BoxSize from bounding box
        box = 0.0
        if "bounding box" in f:
            bb = f["bounding box"][:]  # (nblocks, ndim, 2)
            box = float(np.max(bb[:, :, 1]) - np.min(bb[:, :, 0]))

        header_attrs = {"Time": time, "BoxSize": box}
        return header_attrs, out
    finally:
        f.close()


class _FlashFile:
    """h5py.File-like wrapper that exposes a flattened FLASH file as PartType0."""

    def __init__(self, path):
        header_attrs, fields = _read_flash(path)
        self._groups = {
            "Header": _HeaderGroup(header_attrs),
            "PartType0": _MemGroup(fields),
        }

    def keys(self):
        return list(self._groups.keys())

    def __contains__(self, key):
        return key in self._groups

    def __getitem__(self, key):
        return self._groups[key]

    def get(self, key, default=None):
        return self._groups.get(key, default)

    def close(self):
        pass


class _MultiFile:
    """Minimal h5py.File-like wrapper over a multipart snapshot."""

    def __init__(self, paths):
        self._files = [h5py.File(p, "r") for p in paths]
        ptypes = set()
        for f in self._files:
            for k in f.keys():
                if not k.startswith("PartType"):
                    continue
                g = f[k]
                if isinstance(g, h5py.Group) and "Coordinates" in g:
                    ptypes.add(k)
        self._ptypes = sorted(ptypes)

    def keys(self):
        return ["Header"] + self._ptypes

    def __contains__(self, key):
        return key == "Header" or key in self._ptypes

    def __getitem__(self, key):
        if key == "Header":
            return self._files[0]["Header"]
        if key in self._ptypes:
            return _MultiGroup(self._files, key)
        raise KeyError(key)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def close(self):
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass

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
        if os.path.isfile(path) and path.lower().endswith(".athdf"):
            self._file = _AthdfFile(path)
        elif os.path.isfile(path) and _is_flash_file(path):
            self._file = _FlashFile(path)
        else:
            parts = _resolve_snapshot_parts(path)
            if len(parts) == 1 and parts[0].lower().endswith(".athdf"):
                self._file = _AthdfFile(parts[0])
            elif len(parts) == 1 and _is_flash_file(parts[0]):
                self._file = _FlashFile(parts[0])
            elif len(parts) == 1:
                self._file = h5py.File(parts[0], "r")
            else:
                self._file = _MultiFile(parts)
        self.header = dict(self._file["Header"].attrs)
        self.time = float(self.header.get("Time", 0))

        # Discover available particle types in this snapshot
        avail = []
        for k in self._file.keys():
            if not k.startswith("PartType"):
                continue
            grp = self._file[k]
            if "Coordinates" in grp:
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
            mass = self._read_field(ptype, "Masses")
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
                h = self._read_field(ptype, name)
                self._hsml_cache[p] = h
                return h

        # Recover hsml from cell volume if available (Arepo-style "Volume",
        # or Masses/Density). Use 2*cbrt(V) as an approximate kernel radius.
        if "Volume" in grp:
            vol = np.asarray(self._read_field(ptype, "Volume"), dtype=np.float64)
            h = 2.0 * np.cbrt(vol)
            self._hsml_cache[p] = h
            return h
        if "Density" in grp and "Masses" in grp:
            mass = np.asarray(self._read_field(ptype, "Masses"), dtype=np.float64)
            dens = np.asarray(self._read_field(ptype, "Density"), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                vol = np.where(dens > 0, mass / dens, 0.0)
            h = 2.0 * np.cbrt(vol)
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
        if field == "Coordinates" and "CenterOfMass" in grp:
            data = grp["CenterOfMass"][:]
        elif field in grp:
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

        if field != "Coordinates" and data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)

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
