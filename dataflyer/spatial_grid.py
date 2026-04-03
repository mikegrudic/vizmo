"""Multi-level spatial grid for frustum culling and LOD rendering."""

import numpy as np
from numba import njit, prange


class CellMoments:
    """Extensive mass-weighted moments that can be summed hierarchically.

    Stores 5 extensive fields:
        mass:  (N,)   Sigma m
        mp:    (N, 3) Sigma m*x
        mq:    (N,)   Sigma m*q
        mh2:   (N,)   Sigma m*h^2
        mxx:   (N, 6) Sigma m*x*x^T (upper triangle: xx, xy, xz, yy, yz, zz)

    Provides methods to:
        - Accumulate from sorted particles (single-pass numba kernel)
        - Coarsen 2x2x2 blocks (reshape+sum)
        - Derive intensive quantities (com, qty, cov, hsml)
    """

    __slots__ = ('mass', 'mp', 'mq', 'mh2', 'mxx')

    def __init__(self, n):
        self.mass = np.zeros(n, dtype=np.float32)
        self.mp = np.zeros((n, 3), dtype=np.float32)
        self.mq = np.zeros(n, dtype=np.float32)
        self.mh2 = np.zeros(n, dtype=np.float32)
        self.mxx = np.zeros((n, 6), dtype=np.float32)

    def accumulate_from_particles(self, cell_start, sorted_pos, sorted_mass, sorted_hsml, sorted_qty):
        """Fill moments from sorted particle arrays via single-pass numba kernel."""
        _accumulate_cell_moments(
            cell_start, sorted_pos, sorted_mass, sorted_hsml, sorted_qty,
            self.mass, self.mp, self.mq, self.mh2, self.mxx,
        )

    def coarsen_2x2x2(self, child_moments, child_com, child_qty, cnc):
        """Sum 2x2x2 groups from child level. cnc = child grid cells per side."""
        nc = cnc // 2

        def _sum_222(flat):
            return flat.reshape(cnc, cnc, cnc).reshape(nc, 2, nc, 2, nc, 2).sum(axis=(1, 3, 5)).ravel()

        def _sum_222_cols(flat_2d, ncols):
            return flat_2d.reshape(cnc, cnc, cnc, ncols).reshape(
                nc, 2, nc, 2, nc, 2, ncols).sum(axis=(1, 3, 5)).reshape(-1, ncols)

        # Reconstruct extensive quantities from child's intensive + extensive
        child_mp = child_moments.mass[:, None] * child_com
        child_mq = child_moments.mass * child_qty

        self.mass = _sum_222(child_moments.mass)
        self.mp = _sum_222_cols(child_mp, 3)
        self.mq = _sum_222(child_mq)
        self.mxx = _sum_222_cols(child_moments.mxx, 6)
        self.mh2 = _sum_222(child_moments.mh2)

    def derive(self):
        """Compute intensive quantities from extensive moments.

        Returns: (com, qty, cov, hsml) arrays.
        """
        safe = np.maximum(self.mass, 1e-30)
        com = self.mp / safe[:, None]
        qty = self.mq / safe

        # Covariance = <xx> - <x><x>
        cov = np.empty_like(self.mxx)
        cov[:, 0] = self.mxx[:, 0] / safe - com[:, 0] * com[:, 0]
        cov[:, 1] = self.mxx[:, 1] / safe - com[:, 0] * com[:, 1]
        cov[:, 2] = self.mxx[:, 2] / safe - com[:, 0] * com[:, 2]
        cov[:, 3] = self.mxx[:, 3] / safe - com[:, 1] * com[:, 1]
        cov[:, 4] = self.mxx[:, 4] / safe - com[:, 1] * com[:, 2]
        cov[:, 5] = self.mxx[:, 5] / safe - com[:, 2] * com[:, 2]

        var = cov[:, 0] + cov[:, 3] + cov[:, 5]
        hsml = np.sqrt(np.maximum((1.0 / 0.225) * var + self.mh2 / safe, 1e-30))

        return com, qty, cov, hsml


@njit(parallel=True, cache=True)
def _gather_subsampled_direct(cell_start, vis_cells, budget, s_pos, s_hsml, s_mass, s_qty):
    """Gather particle data with uniform stride subsampling."""
    n_cells = len(vis_cells)

    total = np.int64(0)
    for i in range(n_cells):
        c = vis_cells[i]
        total += cell_start[c + 1] - cell_start[c]

    stride = np.int64(1)
    if total > budget:
        stride = max(np.int64(1), total // budget)

    out_counts = np.empty(n_cells, dtype=np.int64)
    out_total = np.int64(0)
    for i in range(n_cells):
        c = vis_cells[i]
        n = cell_start[c + 1] - cell_start[c]
        kept = (n + stride - 1) // stride
        out_counts[i] = kept
        out_total += kept

    out_offsets = np.empty(n_cells, dtype=np.int64)
    out_offsets[0] = 0
    for i in range(1, n_cells):
        out_offsets[i] = out_offsets[i - 1] + out_counts[i - 1]

    o_pos = np.empty((out_total, 3), dtype=np.float32)
    o_hsml = np.empty(out_total, dtype=np.float32)
    o_mass = np.empty(out_total, dtype=np.float32)
    o_qty = np.empty(out_total, dtype=np.float32)

    for i in prange(n_cells):
        c = vis_cells[i]
        start = cell_start[c]
        end = cell_start[c + 1]
        out_start = out_offsets[i]
        j = 0
        for k in range(start, end, stride):
            o_pos[out_start + j, 0] = s_pos[k, 0]
            o_pos[out_start + j, 1] = s_pos[k, 1]
            o_pos[out_start + j, 2] = s_pos[k, 2]
            o_hsml[out_start + j] = s_hsml[k]
            o_mass[out_start + j] = s_mass[k]
            o_qty[out_start + j] = s_qty[k]
            j += 1

    return o_pos, o_hsml, o_mass, o_qty, total


@njit(parallel=True, cache=True)
def _gather_importance_sampled(
    cell_start, vis_cells, budget, s_pos, s_hsml, s_mass, s_qty, cell_depths, cell_hsml_summary
):
    """Gather particles with per-cell importance-weighted strides.

    Each cell gets a stride based on its angular area weight:
      w_cell = (h_cell / depth_cell)^2
    Cells covering more pixels get stride=1, distant small cells get larger strides.
    Per-cell mass/h rescaling by the cell's stride ratio.

    Uses pre-computed cell depths and summary h (from the tree traversal),
    so no per-particle dot products needed -- same speed as uniform stride.
    """
    n_cells = len(vis_cells)

    # First pass: compute per-cell weights and total weighted budget allocation
    cell_weights = np.empty(n_cells, dtype=np.float64)
    cell_counts = np.empty(n_cells, dtype=np.int64)
    sum_w = np.float64(0.0)
    total = np.int64(0)
    for i in range(n_cells):
        c = vis_cells[i]
        n = cell_start[c + 1] - cell_start[c]
        cell_counts[i] = n
        total += n
        d = max(cell_depths[i], np.float64(0.01))
        h = np.float64(cell_hsml_summary[i])
        w = (h * h) / (d * d)
        cell_weights[i] = w * n  # total weight for this cell = per-particle weight * count
        sum_w += w * n

    if total == 0 or sum_w == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.int64(0),
        )

    # Per-cell: allocate budget proportional to weight
    # stride_i = max(1, n_i / allocated_i) where allocated_i = budget * w_i / sum_w
    cell_strides = np.empty(n_cells, dtype=np.int64)
    out_counts = np.empty(n_cells, dtype=np.int64)
    out_total = np.int64(0)
    for i in range(n_cells):
        n = cell_counts[i]
        allocated = cell_weights[i] / sum_w * budget
        if allocated >= n:
            cell_strides[i] = 1
        else:
            cell_strides[i] = max(np.int64(1), np.int64(n / max(allocated, 1.0)))
        kept = (n + cell_strides[i] - 1) // cell_strides[i]
        out_counts[i] = kept
        out_total += kept

    # Prefix sum for output offsets
    out_offsets = np.empty(n_cells, dtype=np.int64)
    out_offsets[0] = 0
    for i in range(1, n_cells):
        out_offsets[i] = out_offsets[i - 1] + out_counts[i - 1]

    # Allocate output
    o_pos = np.empty((out_total, 3), dtype=np.float32)
    o_hsml = np.empty(out_total, dtype=np.float32)
    o_mass = np.empty(out_total, dtype=np.float32)
    o_qty = np.empty(out_total, dtype=np.float32)

    # Parallel gather with per-cell stride and rescaling
    for i in prange(n_cells):
        c = vis_cells[i]
        start = cell_start[c]
        end = cell_start[c + 1]
        stride = cell_strides[i]
        out_start = out_offsets[i]
        n = end - start
        # Rescale factor for this cell
        ratio = np.float32(n) / np.float32(out_counts[i])
        h_scale = ratio ** np.float32(0.33333333)
        j = 0
        for k in range(start, end, stride):
            o_pos[out_start + j, 0] = s_pos[k, 0]
            o_pos[out_start + j, 1] = s_pos[k, 1]
            o_pos[out_start + j, 2] = s_pos[k, 2]
            o_hsml[out_start + j] = s_hsml[k] * h_scale
            o_mass[out_start + j] = s_mass[k] * ratio
            o_qty[out_start + j] = s_qty[k]
            j += 1

    return o_pos, o_hsml, o_mass, o_qty, total


@njit(parallel=True, cache=True)
def _accumulate_cell_moments(cell_start, s_pos, s_mass, s_hsml, s_qty,
                             cell_mass, cell_mp, cell_mq, cell_mh2, cell_mxx):
    """Single-pass accumulation of all moments from sorted particles into cells.

    Replaces 12 separate np.add.reduceat calls with one loop touching each
    particle exactly once. Much more cache-friendly for large datasets.
    """
    nc3 = len(cell_mass)
    for c in prange(nc3):
        start = cell_start[c]
        end = cell_start[c + 1]
        if start == end:
            continue

        m_sum = np.float64(0.0)
        mpx = np.float64(0.0)
        mpy = np.float64(0.0)
        mpz = np.float64(0.0)
        mq_sum = np.float64(0.0)
        mh2_sum = np.float64(0.0)
        mxx = np.float64(0.0)
        mxy = np.float64(0.0)
        mxz = np.float64(0.0)
        myy = np.float64(0.0)
        myz = np.float64(0.0)
        mzz = np.float64(0.0)

        for k in range(start, end):
            m = np.float64(s_mass[k])
            px = np.float64(s_pos[k, 0])
            py = np.float64(s_pos[k, 1])
            pz = np.float64(s_pos[k, 2])
            h = np.float64(s_hsml[k])
            q = np.float64(s_qty[k])

            m_sum += m
            mpx += m * px
            mpy += m * py
            mpz += m * pz
            mq_sum += m * q
            mh2_sum += m * h * h
            mxx += m * px * px
            mxy += m * px * py
            mxz += m * px * pz
            myy += m * py * py
            myz += m * py * pz
            mzz += m * pz * pz

        cell_mass[c] = np.float32(m_sum)
        cell_mp[c, 0] = np.float32(mpx)
        cell_mp[c, 1] = np.float32(mpy)
        cell_mp[c, 2] = np.float32(mpz)
        cell_mq[c] = np.float32(mq_sum)
        cell_mh2[c] = np.float32(mh2_sum)
        cell_mxx[c, 0] = np.float32(mxx)
        cell_mxx[c, 1] = np.float32(mxy)
        cell_mxx[c, 2] = np.float32(mxz)
        cell_mxx[c, 3] = np.float32(myy)
        cell_mxx[c, 4] = np.float32(myz)
        cell_mxx[c, 5] = np.float32(mzz)


class SpatialGrid:
    """Multi-level uniform grid for frustum culling and LOD rendering.

    Builds summaries at multiple levels (64, 32, 16, 8, 4, 2 cells per side).
    At query time, each region is rendered at the coarsest level where cells
    subtend >= lod_pixels, or at full particle resolution if close enough.
    """

    def __init__(self, positions, masses, hsml, quantity, n_cells=64):
        self.n_cells = n_cells
        self.pmin = positions.min(axis=0).astype(np.float32)
        self.pmax = positions.max(axis=0).astype(np.float32)
        box = self.pmax - self.pmin
        box[box == 0] = 1.0

        # Finest-level cell assignment and sort
        cs = box / n_cells
        cell_idx = np.clip(((positions - self.pmin) / cs).astype(np.int32), 0, n_cells - 1)
        cell_id = cell_idx[:, 0] * n_cells * n_cells + cell_idx[:, 1] * n_cells + cell_idx[:, 2]
        self.sort_order = np.argsort(cell_id)
        sorted_cell_id = cell_id[self.sort_order]

        nc3 = n_cells**3
        self.cell_start = np.zeros(nc3 + 1, dtype=np.int64)
        unique_cells, counts = np.unique(sorted_cell_id, return_counts=True)
        self.cell_start[unique_cells + 1] = counts
        np.cumsum(self.cell_start, out=self.cell_start)

        # Store particle data in cell-sorted order for cache-friendly access.
        so = self.sort_order
        self.sorted_pos = positions[so].astype(np.float32)
        self.sorted_hsml = hsml[so].astype(np.float32)
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)

        # Build finest level from particles
        finest = self._build_finest(n_cells, box)
        self.levels = [finest]

        # Build coarser levels by merging 2x2x2 groups of children
        prev = finest
        while prev["nc"] > 2:
            coarser = self._coarsen(prev)
            self.levels.append(coarser)
            prev = coarser

        # Precompute static child expansion offsets (used in query_frustum_lod)
        self._child_offsets = np.array(
            [[dx, dy, dz] for dx in range(2) for dy in range(2) for dz in range(2)],
            dtype=np.int64,
        )

        # Store box for reuse in update_weights
        self._box = box

    def update_weights(self, masses, quantity=None):
        """Re-weight the grid with new mass/quantity arrays without rebuilding structure.

        Skips argsort, cell_start computation, and center generation (~50% of build time).
        Only recomputes sorted arrays, moments, and coarsened levels.
        """
        if quantity is None:
            quantity = masses
        so = self.sort_order
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)

        # Rebuild moments and coarsened levels (reuses existing cell_start, centers)
        nc = self.n_cells
        finest = self._build_finest(nc, self._box)
        self.levels = [finest]
        prev = finest
        while prev["nc"] > 2:
            coarser = self._coarsen(prev)
            self.levels.append(coarser)
            prev = coarser

    def _make_centers(self, nc, cs):
        """Build cell center coordinates for a grid level."""
        cx = np.arange(nc, dtype=np.float32) * cs[0] + self.pmin[0] + cs[0] * 0.5
        cy = np.arange(nc, dtype=np.float32) * cs[1] + self.pmin[1] + cs[1] * 0.5
        cz = np.arange(nc, dtype=np.float32) * cs[2] + self.pmin[2] + cs[2] * 0.5
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def _build_finest(self, nc, box):
        """Build the finest level from sorted particle arrays."""
        cs = box / nc
        moments = CellMoments(nc**3)
        moments.accumulate_from_particles(
            self.cell_start,
            self.sorted_pos, self.sorted_mass, self.sorted_hsml, self.sorted_qty,
        )
        com, qty, cov, cell_hsml = moments.derive()

        return {
            "nc": nc, "cs": cs, "moments": moments,
            "mass": moments.mass, "com": com, "hsml": cell_hsml, "qty": qty,
            "mxx": moments.mxx, "mh2": moments.mh2, "cov": cov,
            "cell_start": self.cell_start, "sort_order": self.sort_order,
            "centers": self._make_centers(nc, cs),
            "half_diag": float(np.linalg.norm(cs) * 0.5),
        }

    def _coarsen(self, child):
        """Build a coarser level by merging 2x2x2 groups of child cells."""
        cnc = child["nc"]
        nc = cnc // 2
        cs = child["cs"] * 2

        moments = CellMoments(nc**3)
        moments.coarsen_2x2x2(child["moments"], child["com"], child["qty"], cnc)
        com, qty, cov, cell_hsml = moments.derive()

        return {
            "nc": nc, "cs": cs, "moments": moments,
            "mass": moments.mass, "com": com, "hsml": cell_hsml, "qty": qty,
            "mxx": moments.mxx, "mh2": moments.mh2, "cov": cov,
            "centers": self._make_centers(nc, cs),
            "half_diag": float(np.linalg.norm(cs) * 0.5),
        }

    def _frustum_cull_finest(self, camera, max_particles, importance_sampling=False):
        """Fast path: frustum cull directly on finest-level cells, skip tree."""
        finest = self.levels[0]
        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        half_tan = np.tan(fov_rad / 2)
        hd = finest["half_diag"]

        centers = finest["centers"]
        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        # Extend frustum by cell size to include particles whose kernels
        # overlap the view edge. Cap extension at 2*hd to avoid over-including
        # cells with large variance-based hsml.
        cell_extent = hd + np.minimum(finest["hsml"], 2 * hd)
        front_depth = np.maximum(depths + cell_extent, 0)
        lim_h = front_depth * half_tan * camera.aspect + cell_extent
        lim_v = front_depth * half_tan + cell_extent
        in_front = depths > -cell_extent
        has_mass = finest["mass"] > 0
        visible = has_mass & in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

        vis_cells = np.where(visible)[0]
        if len(vis_cells) == 0:
            z3 = np.zeros((0, 3), dtype=np.float32)
            z1 = np.zeros(0, dtype=np.float32)
            return z3, z1, z1, z1

        if importance_sampling:
            dist = np.sqrt(depths[visible] ** 2 + rights[visible] ** 2 + ups[visible] ** 2)
            safe_dist = np.maximum(dist, 0.01).astype(np.float64)
            vis_cell_h = finest["hsml"][vis_cells].astype(np.float64)
            pos, hsml, mass, qty, n_vis = _gather_importance_sampled(
                finest["cell_start"], vis_cells.astype(np.int64), max_particles,
                self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
                safe_dist, vis_cell_h,
            )
        else:
            pos, hsml, mass, qty, n_vis = _gather_subsampled_direct(
                finest["cell_start"], vis_cells.astype(np.int64), max_particles,
                self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
            )

        n_vis = int(n_vis)
        n_sampled = len(pos)
        if not importance_sampling and n_sampled > 0 and n_vis > n_sampled:
            ratio = n_vis / n_sampled
            mass = mass * ratio
            hsml = hsml * (ratio ** (1.0 / 3.0))

        return (
            pos.astype(np.float32), hsml.astype(np.float32),
            mass.astype(np.float32), qty.astype(np.float32),
        )

    def query_frustum_lod(self, camera, max_particles, lod_pixels=4,
                          importance_sampling=False, viewport_width=2048,
                          summary_overlap=0.0):
        """Top-down multi-level LOD query.

        Traverses coarsest-to-finest. Cells subtending <= lod_pixels emit a
        summary splat. Cells subtending more are refined to children. At the
        finest level, real particles are emitted via numba parallel gather.
        """
        if lod_pixels <= 2:
            return self._frustum_cull_finest(camera, max_particles, importance_sampling)

        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        pix_per_rad = viewport_width / (2.0 * np.tan(fov_rad / 2))
        finest = self.levels[0]
        summary_parts = []
        real_pos = None
        real_hsml = None
        real_mass = None
        real_qty = None
        n_visible_real = 0

        # Start from coarsest level
        lv = self.levels[-1]
        centers = lv["centers"]

        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        hd = lv["half_diag"]
        half_tan = np.tan(fov_rad / 2)
        has_mass = lv["mass"] > 0
        cell_extent = hd + np.minimum(lv["hsml"], 2 * hd)
        front_depth = np.maximum(depths + cell_extent, 0)
        lim_h = front_depth * half_tan * camera.aspect + cell_extent
        lim_v = front_depth * half_tan + cell_extent
        in_front = depths > -cell_extent
        visible = has_mass & in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

        dist = np.sqrt(depths**2 + rights**2 + ups**2)
        safe_dist = np.maximum(dist, 0.01)
        h_pix = lv["hsml"] / safe_dist * pix_per_rad

        # Always refine the coarsest levels (nc <= 8) — they're too coarse
        # to produce meaningful summary splats
        if lv["nc"] <= 8:
            refine_mask = visible & has_mass
        else:
            summary_mask = visible & has_mass & (h_pix <= lod_pixels)
            refine_mask = visible & has_mass & (h_pix > lod_pixels)

            s_idx = np.where(summary_mask)[0]
            if len(s_idx) > 0:
                summary_parts.append((
                    lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx],
                    lv["qty"][s_idx], lv["cov"][s_idx],
                    lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"],
                ))

        refine_cells = np.where(refine_mask)[0]

        # Traverse finer levels
        for li in range(len(self.levels) - 2, -1, -1):
            if len(refine_cells) == 0:
                break

            lv = self.levels[li]
            parent_nc = self.levels[li + 1]["nc"]
            nc = lv["nc"]
            hd = lv["half_diag"]

            # Map parent cells to 8 children
            pix = refine_cells // (parent_nc * parent_nc)
            piy = (refine_cells // parent_nc) % parent_nc
            piz = refine_cells % parent_nc
            offsets = self._child_offsets
            child_ix = (pix[:, None] * 2 + offsets[None, :, 0]).ravel()
            child_iy = (piy[:, None] * 2 + offsets[None, :, 1]).ravel()
            child_iz = (piz[:, None] * 2 + offsets[None, :, 2]).ravel()
            child_flat = child_ix * nc * nc + child_iy * nc + child_iz

            valid = lv["mass"][child_flat] > 0
            child_flat = child_flat[valid]
            if len(child_flat) == 0:
                break

            # Parent was already frustum-tested; only need distance for opening criterion
            centers = lv["centers"][child_flat]
            diff = centers - cam_pos
            dist = np.sqrt((diff * diff).sum(axis=1))
            safe_dist = np.maximum(dist, 0.01)
            h_pix = lv["hsml"][child_flat] / safe_dist * pix_per_rad

            small = h_pix <= lod_pixels
            large = ~small

            s_idx = child_flat[small]
            if len(s_idx) > 0:
                summary_parts.append((
                    lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx],
                    lv["qty"][s_idx], lv["cov"][s_idx],
                    lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"],
                ))

            if li == 0:
                vis_cells = child_flat[large]
                if len(vis_cells) > 0:
                    n_summaries = sum(p[0].shape[0] for p in summary_parts) if summary_parts else 0
                    budget = max(max_particles - n_summaries, max_particles // 2)
                    if importance_sampling:
                        vis_depths = safe_dist[large].astype(np.float64)
                        vis_cell_h = lv["hsml"][child_flat[large]].astype(np.float64)
                        real_pos, real_hsml, real_mass, real_qty, n_visible_real = _gather_importance_sampled(
                            finest["cell_start"], vis_cells.astype(np.int64), budget,
                            self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
                            vis_depths, vis_cell_h,
                        )
                    else:
                        real_pos, real_hsml, real_mass, real_qty, n_visible_real = _gather_subsampled_direct(
                            finest["cell_start"], vis_cells.astype(np.int64), budget,
                            self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
                        )
            else:
                refine_cells = child_flat[large]

        # Uniform rescaling for stride-based subsampling
        n_visible_real = int(n_visible_real)
        n_sampled = len(real_pos) if real_pos is not None else 0

        if not importance_sampling and n_sampled > 0 and n_visible_real > n_sampled:
            ratio = n_visible_real / n_sampled
            real_mass = real_mass * ratio
            real_hsml = real_hsml * (ratio ** (1.0 / 3.0))

        real_parts = []
        if n_sampled > 0:
            real_parts.append((real_pos, real_hsml, real_mass, real_qty))

        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros(0, dtype=np.float32)

        if real_parts:
            r_pos = np.concatenate([p[0] for p in real_parts]).astype(np.float32)
            r_hsml = np.concatenate([p[1] for p in real_parts]).astype(np.float32)
            r_mass = np.concatenate([p[2] for p in real_parts]).astype(np.float32)
            r_qty = np.concatenate([p[3] for p in real_parts]).astype(np.float32)
        else:
            r_pos, r_hsml, r_mass, r_qty = z3, z1, z1, z1

        # Assemble summary output with anisotropic covariance
        z6 = np.zeros((0, 6), dtype=np.float32)
        if summary_parts:
            s_pos = np.concatenate([p[0] for p in summary_parts]).astype(np.float32)
            s_hsml = np.concatenate([p[1] for p in summary_parts]).astype(np.float32)
            s_mass = np.concatenate([p[2] for p in summary_parts]).astype(np.float32)
            s_qty = np.concatenate([p[3] for p in summary_parts]).astype(np.float32)
            s_cov = np.concatenate([p[4] for p in summary_parts]).astype(np.float32)
            s_mean_h2 = np.concatenate([p[5] for p in summary_parts]).astype(np.float32)
            s_cs2 = np.concatenate([
                np.broadcast_to((p[6]**2)[None, :], (len(p[0]), 3))
                for p in summary_parts
            ]).astype(np.float32)
            s_cov[:, 0] += 0.225 * s_mean_h2
            s_cov[:, 3] += 0.225 * s_mean_h2
            s_cov[:, 5] += 0.225 * s_mean_h2
            alpha = summary_overlap
            s_cov[:, 0] += alpha * s_cs2[:, 0]
            s_cov[:, 3] += alpha * s_cs2[:, 1]
            s_cov[:, 5] += alpha * s_cs2[:, 2]
        else:
            s_pos, s_hsml, s_mass, s_qty, s_cov = z3, z1, z1, z1, z6

        return (
            r_pos, r_hsml, r_mass, r_qty,
            s_pos, s_hsml, s_mass, s_qty, s_cov,
        )
