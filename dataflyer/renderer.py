"""Core splat renderer: additive accumulation + resolve pass."""

import numpy as np
import moderngl
from pathlib import Path
from dataclasses import dataclass
from numba import njit, prange

SHADER_DIR = Path(__file__).parent / "shaders"


@dataclass
class RenderMode:
    """Defines how the two accumulation textures are combined in the resolve pass.

    All fragment shaders write:
        out_numerator   = sigma * quantity
        out_denominator = sigma           (where sigma = mass * W(r) / h²)

    The resolve shader then displays either the denominator directly (surface
    density) or the ratio numerator/denominator (mass-weighted average).

    To render a given mode, the caller sets:
        - masses (weight field) → the "mass" slot in set_particles
        - quantity             → the "quantity" slot in set_particles
        - resolve_mode         → 0 for surface density, 1 for ratio

    Examples:
        SurfaceDensity("Masses"):
            weight=Masses, qty=Masses (unused), resolve_mode=0
            displays: Σ Masses * W / h²

        SurfaceDensity("Density"):
            weight=Density, qty=Density (unused), resolve_mode=0
            displays: Σ Density * W / h²

        MassWeightedAverage("Temperature"):
            weight=Masses, qty=Temperature, resolve_mode=1
            displays: Σ(Masses * Temperature * W / h²) / Σ(Masses * W / h²)
    """
    name: str           # display name
    weight_field: str   # field loaded into the mass/weight slot
    qty_field: str      # field loaded into the quantity slot
    resolve_mode: int   # 0: display denominator, 1: display num/denom

    @staticmethod
    def surface_density(weight_field="Masses"):
        """Create a surface density render mode for the given weight field."""
        return RenderMode(
            name=f"Σ {weight_field}",
            weight_field=weight_field,
            qty_field=weight_field,  # unused in resolve_mode=0, but must be valid
            resolve_mode=0,
        )

    @staticmethod
    def mass_weighted_average(qty_field, weight_field="Masses"):
        """Create a mass-weighted average render mode (for future use)."""
        return RenderMode(
            name=f"<{qty_field}>",
            weight_field=weight_field,
            qty_field=qty_field,
            resolve_mode=1,
        )


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


# Maximum particles to render per frame for interactive performance
MAX_RENDER_PARTICLES = 4_000_000


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


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
        # After this, cell_start indices point directly into these arrays
        # with no indirection through sort_order.
        so = self.sort_order
        self.sorted_pos = positions[so].astype(np.float32)
        self.sorted_hsml = hsml[so].astype(np.float32)
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)

        # Build finest level from particles
        finest = self._build_finest(positions, masses, hsml, quantity, n_cells, box)
        self.levels = [finest]

        # Build coarser levels by merging 2x2x2 groups of children
        prev = finest
        while prev["nc"] > 2:
            coarser = self._coarsen(prev)
            self.levels.append(coarser)
            prev = coarser

    def _build_finest(self, positions, masses, hsml, quantity, nc, box):
        """Build the finest level directly from particles."""
        cs = box / nc
        so = self.sort_order

        s_m = masses[so]
        s_p = positions[so]
        s_h = hsml[so]
        s_q = quantity[so]

        nc3 = nc**3
        starts = self.cell_start[:-1].astype(np.intp)
        nonempty = starts < self.cell_start[1:]
        reduce_at = starts[nonempty]
        ne_idx = np.where(nonempty)[0]

        cell_mass = np.zeros(nc3, dtype=np.float32)
        cell_com = np.zeros((nc3, 3), dtype=np.float32)
        cell_hsml = np.zeros(nc3, dtype=np.float32)
        cell_qty = np.zeros(nc3, dtype=np.float32)
        # Mass-weighted second moments for variance/covariance at coarser levels
        # mxx[i] stores: (xx, xy, xz, yy, yz, zz) = upper triangle of m*x*x^T
        cell_mxx = np.zeros((nc3, 6), dtype=np.float32)
        cell_mh2 = np.zeros(nc3, dtype=np.float32)
        # 3D covariance upper triangle: (cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz)
        cell_cov = np.zeros((nc3, 6), dtype=np.float32)

        if len(reduce_at) > 0:
            mass_ne = np.add.reduceat(s_m, reduce_at)
            safe = np.maximum(mass_ne, 1e-30)
            mp_ne = np.add.reduceat(s_p * s_m[:, None], reduce_at)
            mq_ne = np.add.reduceat(s_m * s_q, reduce_at)
            mh2_ne = np.add.reduceat(s_m * s_h**2, reduce_at)

            # Mass-weighted outer products: xx, xy, xz, yy, yz, zz
            mxx_ne = np.column_stack([
                np.add.reduceat(s_m * s_p[:, 0] * s_p[:, 0], reduce_at),
                np.add.reduceat(s_m * s_p[:, 0] * s_p[:, 1], reduce_at),
                np.add.reduceat(s_m * s_p[:, 0] * s_p[:, 2], reduce_at),
                np.add.reduceat(s_m * s_p[:, 1] * s_p[:, 1], reduce_at),
                np.add.reduceat(s_m * s_p[:, 1] * s_p[:, 2], reduce_at),
                np.add.reduceat(s_m * s_p[:, 2] * s_p[:, 2], reduce_at),
            ])

            com_ne = mp_ne / safe[:, None]

            # Covariance = <xx> - <x><x>, etc.
            cov_ne = np.column_stack([
                mxx_ne[:, 0] / safe - com_ne[:, 0] * com_ne[:, 0],  # xx
                mxx_ne[:, 1] / safe - com_ne[:, 0] * com_ne[:, 1],  # xy
                mxx_ne[:, 2] / safe - com_ne[:, 0] * com_ne[:, 2],  # xz
                mxx_ne[:, 3] / safe - com_ne[:, 1] * com_ne[:, 1],  # yy
                mxx_ne[:, 4] / safe - com_ne[:, 1] * com_ne[:, 2],  # yz
                mxx_ne[:, 5] / safe - com_ne[:, 2] * com_ne[:, 2],  # zz
            ])

            var_ne = cov_ne[:, 0] + cov_ne[:, 3] + cov_ne[:, 5]  # trace = total variance

            cell_mass[ne_idx] = mass_ne
            cell_com[ne_idx] = com_ne
            cell_qty[ne_idx] = mq_ne / safe
            cell_mxx[ne_idx] = mxx_ne
            cell_mh2[ne_idx] = mh2_ne
            cell_cov[ne_idx] = cov_ne
            # Isotropic h from 3D variance matching (still used for LOD opening criterion)
            h_var = np.sqrt(np.maximum((1.0 / 0.225) * var_ne + mh2_ne / safe, 1e-30))
            cell_hsml[ne_idx] = h_var

        cx = np.arange(nc, dtype=np.float32) * cs[0] + self.pmin[0] + cs[0] * 0.5
        cy = np.arange(nc, dtype=np.float32) * cs[1] + self.pmin[1] + cs[1] * 0.5
        cz = np.arange(nc, dtype=np.float32) * cs[2] + self.pmin[2] + cs[2] * 0.5
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        return {
            "nc": nc,
            "cs": cs,
            "mass": cell_mass,
            "com": cell_com,
            "hsml": cell_hsml,
            "qty": cell_qty,
            "mxx": cell_mxx,
            "mh2": cell_mh2,  # for coarsening
            "cov": cell_cov,
            "cell_start": self.cell_start,
            "sort_order": self.sort_order,
            "centers": centers,
            "half_diag": float(np.linalg.norm(cs) * 0.5),
        }

    def _coarsen(self, child):
        """Build a coarser level by merging 2x2x2 groups of child cells."""
        cnc = child["nc"]
        nc = cnc // 2
        cs = child["cs"] * 2
        nc3 = nc**3

        # Map child cell (ix, iy, iz) -> parent cell (ix//2, iy//2, iz//2)
        cix = np.arange(cnc)
        ciy = np.arange(cnc)
        ciz = np.arange(cnc)
        gx, gy, gz = np.meshgrid(cix, ciy, ciz, indexing="ij")
        parent_id = (gx.ravel() // 2) * nc * nc + (gy.ravel() // 2) * nc + (gz.ravel() // 2)

        # Sum child properties into parent cells
        cell_mass = np.zeros(nc3, dtype=np.float32)
        cell_mp = np.zeros((nc3, 3), dtype=np.float32)
        cell_mq = np.zeros(nc3, dtype=np.float32)
        cell_mxx = np.zeros((nc3, 6), dtype=np.float32)
        cell_mh2 = np.zeros(nc3, dtype=np.float32)

        np.add.at(cell_mass, parent_id, child["mass"])
        for d in range(3):
            np.add.at(cell_mp[:, d], parent_id, child["mass"] * child["com"][:, d])
        for d in range(6):
            np.add.at(cell_mxx[:, d], parent_id, child["mxx"][:, d])
        np.add.at(cell_mq, parent_id, child["mass"] * child["qty"])
        np.add.at(cell_mh2, parent_id, child["mh2"])

        safe = np.maximum(cell_mass, 1e-30)
        cell_com = cell_mp / safe[:, None]
        cell_qty = cell_mq / safe

        # Full covariance: cov_ij = <x_i x_j> - <x_i><x_j>
        cell_cov = np.column_stack([
            cell_mxx[:, 0] / safe - cell_com[:, 0] * cell_com[:, 0],  # xx
            cell_mxx[:, 1] / safe - cell_com[:, 0] * cell_com[:, 1],  # xy
            cell_mxx[:, 2] / safe - cell_com[:, 0] * cell_com[:, 2],  # xz
            cell_mxx[:, 3] / safe - cell_com[:, 1] * cell_com[:, 1],  # yy
            cell_mxx[:, 4] / safe - cell_com[:, 1] * cell_com[:, 2],  # yz
            cell_mxx[:, 5] / safe - cell_com[:, 2] * cell_com[:, 2],  # zz
        ])

        var = cell_cov[:, 0] + cell_cov[:, 3] + cell_cov[:, 5]  # trace
        h_var = np.sqrt(np.maximum((1.0 / 0.225) * var + cell_mh2 / safe, 1e-30))
        cell_hsml = h_var

        cx = np.arange(nc, dtype=np.float32) * cs[0] + self.pmin[0] + cs[0] * 0.5
        cy = np.arange(nc, dtype=np.float32) * cs[1] + self.pmin[1] + cs[1] * 0.5
        cz = np.arange(nc, dtype=np.float32) * cs[2] + self.pmin[2] + cs[2] * 0.5
        gx2, gy2, gz2 = np.meshgrid(cx, cy, cz, indexing="ij")
        centers = np.stack([gx2.ravel(), gy2.ravel(), gz2.ravel()], axis=1)

        return {
            "nc": nc,
            "cs": cs,
            "mass": cell_mass,
            "com": cell_com,
            "hsml": cell_hsml,
            "qty": cell_qty,
            "mxx": cell_mxx,
            "mh2": cell_mh2,
            "cov": cell_cov,
            "centers": centers,
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

        front_depth = np.maximum(depths + hd, 0)
        lim_h = front_depth * half_tan * camera.aspect + hd
        lim_v = front_depth * half_tan + hd
        in_front = depths > -(hd + finest["hsml"])
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

    def query_frustum_lod(self, camera, max_particles, lod_pixels=4, importance_sampling=False, viewport_width=2048, summary_overlap=0.0):
        """Top-down multi-level LOD query. Returns (pos, hsml, mass, qty) arrays.

        All data comes from pre-sorted arrays built at grid construction time.
        No per-query access to the full particle arrays.

        Traverses coarsest-to-finest. Cells subtending <= lod_pixels emit a
        summary splat (at CoM with variance-based h). Cells subtending more
        are refined to children. At the finest level, real particles are emitted
        via numba parallel gather from the pre-sorted arrays.

        When the particle budget is exceeded, real particles are subsampled
        with mass and h rescaled to conserve the mass distribution.
        If importance_sampling=True, particles are weighted by angular area.
        """
        # Fast path: if LOD threshold is very low, skip tree traversal entirely
        # and frustum-cull directly on the finest level cells.
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
        coarsest = self.levels[-1]
        lv = coarsest
        centers = lv["centers"]

        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        hd = lv["half_diag"]
        half_tan = np.tan(fov_rad / 2)
        has_mass = lv["mass"] > 0
        # Frustum test: a cell is visible if any part of it *could* be on screen.
        # The cell extends ±hd from its center. The visible half-width at depth d
        # is d*tan(fov/2). Account for cell extent on both sides.
        front_depth = np.maximum(depths + hd, 0)  # nearest front face of cell
        lim_h = front_depth * half_tan * camera.aspect + hd
        lim_v = front_depth * half_tan + hd
        # Cell must have at least some part in front of camera
        in_front = depths > -(hd + lv["hsml"])  # include kernel radius
        visible = has_mass & in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

        dist = np.sqrt(depths**2 + rights**2 + ups**2)
        safe_dist = np.maximum(dist, 0.01)
        # Opening criterion: does the summary h subtend > lod_pixels?
        h_pix = lv["hsml"] / safe_dist * pix_per_rad

        summary_mask = visible & has_mass & (h_pix <= lod_pixels)
        refine_mask = visible & has_mass & (h_pix > lod_pixels)

        s_idx = np.where(summary_mask)[0]
        if len(s_idx) > 0:
            summary_parts.append((lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx], lv["qty"][s_idx], lv["cov"][s_idx], lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"]))

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
            offsets = np.array([[dx, dy, dz] for dx in range(2) for dy in range(2) for dz in range(2)])
            child_ix = (pix[:, None] * 2 + offsets[None, :, 0]).ravel()
            child_iy = (piy[:, None] * 2 + offsets[None, :, 1]).ravel()
            child_iz = (piz[:, None] * 2 + offsets[None, :, 2]).ravel()
            child_flat = child_ix * nc * nc + child_iy * nc + child_iz

            valid = lv["mass"][child_flat] > 0
            child_flat = child_flat[valid]
            if len(child_flat) == 0:
                break

            centers = lv["centers"][child_flat]
            depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
            rights = centers @ cam_right - np.dot(cam_pos, cam_right)
            ups = centers @ cam_up - np.dot(cam_pos, cam_up)

            # No frustum culling at child levels -- parent was already visible.
            # Only use opening criterion (h_pix) for LOD decisions.
            vis = np.ones(len(depths), dtype=bool)

            dist = np.sqrt(depths**2 + rights**2 + ups**2)
            safe_dist = np.maximum(dist, 0.01)
            h_pix = lv["hsml"][child_flat] / safe_dist * pix_per_rad

            small = vis & (h_pix <= lod_pixels)
            large = vis & (h_pix > lod_pixels)

            # Summary splats for small cells
            s_idx = child_flat[small]
            if len(s_idx) > 0:
                summary_parts.append((lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx], lv["qty"][s_idx], lv["cov"][s_idx], lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"]))

            if li == 0:
                # Finest level: real particles for large cells
                vis_cells = child_flat[large]
                if len(vis_cells) > 0:
                    n_summaries = sum(p[0].shape[0] for p in summary_parts) if summary_parts else 0
                    budget = max(max_particles - n_summaries, max_particles // 2)
                    if importance_sampling:
                        # Use cell depths and summary h for per-cell weighting
                        vis_depths = safe_dist[large].astype(np.float64)
                        vis_cell_h = lv["hsml"][child_flat[large]].astype(np.float64)
                        real_pos, real_hsml, real_mass, real_qty, n_visible_real = _gather_importance_sampled(
                            finest["cell_start"],
                            vis_cells.astype(np.int64),
                            budget,
                            self.sorted_pos,
                            self.sorted_hsml,
                            self.sorted_mass,
                            self.sorted_qty,
                            vis_depths,
                            vis_cell_h,
                        )
                    else:
                        real_pos, real_hsml, real_mass, real_qty, n_visible_real = _gather_subsampled_direct(
                            finest["cell_start"],
                            vis_cells.astype(np.int64),
                            budget,
                            self.sorted_pos,
                            self.sorted_hsml,
                            self.sorted_mass,
                            self.sorted_qty,
                        )
            else:
                refine_cells = child_flat[large]

        # Uniform rescaling for stride-based subsampling (importance does it per-particle)
        n_visible_real = int(n_visible_real)
        n_sampled = len(real_pos) if real_pos is not None else 0

        if not importance_sampling and n_sampled > 0 and n_visible_real > n_sampled:
            ratio = n_visible_real / n_sampled
            real_mass = real_mass * ratio
            real_hsml = real_hsml * (ratio ** (1.0 / 3.0))

        # Assemble real particle output
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
            # Per-splat cell size² (each level may differ)
            s_cs2 = np.concatenate([
                np.broadcast_to((p[6]**2)[None, :], (len(p[0]), 3))
                for p in summary_parts
            ]).astype(np.float32)
            # Add kernel smoothing to covariance: Σ_eff = Σ_spatial + 0.225*<h²>*I
            s_cov[:, 0] += 0.225 * s_mean_h2  # xx
            s_cov[:, 3] += 0.225 * s_mean_h2  # yy
            s_cov[:, 5] += 0.225 * s_mean_h2  # zz
            # Add cell-size padding to bridge voids at cell boundaries
            # α*cs² ensures neighboring Gaussians overlap sufficiently
            alpha = summary_overlap
            s_cov[:, 0] += alpha * s_cs2[:, 0]  # xx
            s_cov[:, 3] += alpha * s_cs2[:, 1]  # yy
            s_cov[:, 5] += alpha * s_cs2[:, 2]  # zz
        else:
            s_pos, s_hsml, s_mass, s_qty, s_cov = z3, z1, z1, z1, z6

        return (
            r_pos, r_hsml, r_mass, r_qty,
            s_pos, s_hsml, s_mass, s_qty, s_cov,
        )


class SplatRenderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.n_particles = 0
        self.n_total = 0  # total particles in dataset

        # Compile shader programs
        self.prog_additive = ctx.program(  # point sprites
            vertex_shader=_load_shader("splat.vert"),
            fragment_shader=_load_shader("splat_additive.frag"),
        )
        self.prog_quad = ctx.program(  # instanced quads for large particles
            vertex_shader=_load_shader("splat_quad.vert"),
            fragment_shader=_load_shader("splat_quad.frag"),
        )
        self.prog_resolve = ctx.program(
            vertex_shader=_load_shader("resolve.vert"),
            fragment_shader=_load_shader("resolve.frag"),
        )
        self.prog_star = ctx.program(
            vertex_shader=_load_shader("star.vert"),
            fragment_shader=_load_shader("star.frag"),
        )
        self.prog_aniso = ctx.program(  # anisotropic Gaussian summary splats
            vertex_shader=_load_shader("splat_aniso.vert"),
            fragment_shader=_load_shader("splat_aniso.frag"),
        )

        # Billboard quad for instanced rendering of large particles
        quad_corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.quad_vbo = ctx.buffer(quad_corners.tobytes())
        self.quad_ibo = ctx.buffer(np.array([0, 1, 2, 2, 1, 3], dtype=np.int32).tobytes())

        # Fullscreen quad for resolve pass
        fs_quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.fs_quad_vbo = ctx.buffer(fs_quad.tobytes())

        # Particle buffers -- separate sets for points (small) and quads (large)
        self.pos_vbo = None
        self.hsml_vbo = None
        self.mass_vbo = None
        self.qty_vbo = None
        self.vao_additive = None  # point sprites

        self.big_pos_vbo = None
        self.big_hsml_vbo = None
        self.big_mass_vbo = None
        self.big_qty_vbo = None
        self.vao_quad = None  # instanced quads
        self.n_big = 0

        # Anisotropic summary splat buffers
        self.aniso_pos_vbo = None
        self.aniso_mass_vbo = None
        self.aniso_qty_vbo = None
        self.aniso_cov_vbo = None  # 6 floats per instance: upper triangle of 3D covariance
        self.vao_aniso = None
        self.n_aniso = 0

        # Full dataset stored on CPU for culling
        self._all_pos = None
        self._all_hsml = None
        self._all_mass = None
        self._all_qty = None

        # FBO for accumulation (2 float textures: numerator + denominator)
        self._accum_fbo = None
        self._accum_tex_num = None
        self._accum_tex_den = None
        self._fbo_size = (0, 0)
        self._viewport_width = 1024  # updated each frame in render()

        # Resolve VAO
        self.vao_resolve = ctx.vertex_array(
            self.prog_resolve,
            [(self.fs_quad_vbo, "2f", "in_position")],
        )

        # Star particle buffers
        self.star_pos_vbo = None
        self.star_mass_vbo = None
        self.vao_star = None
        self.n_stars = 0
        self.star_point_size = 50.0

        # Colormap texture (set externally)
        self.colormap_tex = None

        # Render state
        self.alpha_scale = 1.0
        self.qty_min = -1.0
        self.qty_max = 3.0
        self.resolve_mode = 0  # 0: surface density, 1: weighted quantity (set by RenderMode)
        self.lod_pixels = 4  # cells subtending fewer pixels than this get summarized
        self.log_scale = 1  # 1: log10, 0: linear
        self.max_render_particles = MAX_RENDER_PARTICLES
        self.use_tree = True
        self.tree_min_particles = 0  # only build tree if N > this threshold
        self.use_importance_sampling = False
        self.KERNELS = ["cubic_spline", "wendland_c2", "gaussian", "quartic", "sphere"]
        self.kernel = "cubic_spline"
        self.use_hybrid_rendering = True  # use quads for >64px particles
        self.use_quad_rendering = False  # True = all quads (pre-optimization path)
        self.summary_scale = 1.0  # scaling factor applied to summary splats
        self.summary_overlap = 0.1  # cell-size padding to bridge voids at tree boundaries
        self.use_aniso_summaries = True  # False = isotropic spherical summaries
        self.cull_interval = 0.5  # seconds between culls while moving
        self._needs_grid_rebuild = False

    def set_particles(self, positions, hsml, masses, quantity=None):
        """Store particle data on CPU. Call update_visible() to upload a subset."""
        self._all_pos = positions.astype(np.float32)
        self._all_hsml = hsml.astype(np.float32)
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)
        self.n_total = len(masses)

        # Build spatial grid for frustum culling and LOD
        if self.use_tree and self.n_total > self.tree_min_particles:
            import time

            t0 = time.perf_counter()
            self._grid = SpatialGrid(self._all_pos, self._all_mass, self._all_hsml, self._all_qty)
            print(f"  Spatial grid built in {time.perf_counter()-t0:.1f}s")
        else:
            self._grid = None

    def update_visible(self, camera):
        """Cull and upload only visible particles for this frame."""
        if self._all_pos is None:
            return

        if self._needs_grid_rebuild:
            self._needs_grid_rebuild = False
            if self.use_tree and self.n_total > self.tree_min_particles:
                import time
                t0 = time.perf_counter()
                self._grid = SpatialGrid(self._all_pos, self._all_mass, self._all_hsml, self._all_qty)
                print(f"  Spatial grid rebuilt in {time.perf_counter()-t0:.1f}s")
            else:
                self._grid = None

        if self._grid is not None:
            result = self._grid.query_frustum_lod(
                camera,
                self.max_render_particles,
                lod_pixels=self.lod_pixels,
                importance_sampling=self.use_importance_sampling,
                viewport_width=self._viewport_width,
                summary_overlap=self.summary_overlap,
            )
            if len(result) == 9:
                # LOD path: real particles + anisotropic summaries
                r_pos, r_hsml, r_mass, r_qty, s_pos, s_hsml, s_mass, s_qty, s_cov = result
                if self.use_aniso_summaries:
                    # Real particles to isotropic, summaries to anisotropic
                    self._upload_arrays(r_pos, r_hsml, r_mass, r_qty, camera)
                    self._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
                else:
                    # Combine into isotropic path (old behavior)
                    pos = np.concatenate([r_pos, s_pos]) if len(r_pos) + len(s_pos) > 0 else r_pos
                    hsml = np.concatenate([r_hsml, s_hsml]) if len(r_hsml) + len(s_hsml) > 0 else r_hsml
                    mass = np.concatenate([r_mass, s_mass]) if len(r_mass) + len(s_mass) > 0 else r_mass
                    qty = np.concatenate([r_qty, s_qty]) if len(r_qty) + len(s_qty) > 0 else r_qty
                    self._upload_arrays(pos, hsml, mass, qty, camera)
                    self._upload_aniso_summaries(
                        np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                        np.zeros(0, np.float32), np.zeros((0, 6), np.float32),
                    )
            else:
                pos, hsml, mass, qty = result
                self._upload_arrays(pos, hsml, mass, qty, camera)
                self._upload_aniso_summaries(
                    np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                    np.zeros(0, np.float32), np.zeros((0, 6), np.float32),
                )
        else:
            # No tree: simple numpy frustum cull + subsample
            cam_fwd = camera.forward
            depths = self._all_pos @ cam_fwd - np.dot(camera.position, cam_fwd)
            in_front = depths > 0
            idx = np.where(in_front)[0]
            if len(idx) > self.max_render_particles:
                step = max(1, len(idx) // self.max_render_particles)
                n_vis = len(idx)
                idx = idx[::step]
                ratio = n_vis / len(idx)
                pos = self._all_pos[idx]
                hsml = self._all_hsml[idx] * ratio ** (1.0 / 3.0)
                mass = self._all_mass[idx] * ratio
                qty = self._all_qty[idx]
                self._upload_arrays(pos, hsml.astype(np.float32), mass.astype(np.float32), qty, camera)
            else:
                self._upload_arrays(
                    self._all_pos[idx], self._all_hsml[idx], self._all_mass[idx], self._all_qty[idx], camera
                )

    def _upload_arrays(self, pos, hsml, mass, qty, camera=None):
        """Upload pre-built arrays to GPU, splitting into points and quads."""
        self.n_particles = len(mass)
        self.n_big = 0

        if self.n_particles == 0:
            return

        # Release old buffers
        for attr in (
            "pos_vbo",
            "hsml_vbo",
            "mass_vbo",
            "qty_vbo",
            "big_pos_vbo",
            "big_hsml_vbo",
            "big_mass_vbo",
            "big_qty_vbo",
        ):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        # Quad-only mode: all particles as instanced quads (pre-optimization path)
        if self.use_quad_rendering:
            self.pos_vbo = None
            self.hsml_vbo = None
            self.mass_vbo = None
            self.qty_vbo = None
            self.big_pos_vbo = self.ctx.buffer(pos.tobytes())
            self.big_hsml_vbo = self.ctx.buffer(hsml.tobytes())
            self.big_mass_vbo = self.ctx.buffer(mass.tobytes())
            self.big_qty_vbo = self.ctx.buffer(qty.tobytes())
            self.n_particles = 0
            self.n_big = len(mass)
            self._build_vao()
            return

        # Split into small (point sprites) and big (instanced quads).
        # Upload all particles into shared buffers sorted smalls-first,
        # then build VAOs with byte offsets — avoids copying data twice.
        MAX_POINT_PX = 64.0
        if self.use_hybrid_rendering and camera is not None and len(pos) > 0:
            proj = camera.projection_matrix()
            scale = proj[0, 0] * self._viewport_width
            depths = (pos - camera.position) @ camera.forward
            safe_depths = np.maximum(np.abs(depths), 0.01)
            point_px = hsml / safe_depths * scale
            big_mask = point_px > MAX_POINT_PX
            n_big = int(big_mask.sum())
        else:
            n_big = 0

        self.n_big = n_big
        n_small = len(mass) - n_big

        if n_big > 0 and n_small > 0:
            # Partition so smalls come first — O(n) unlike full sort
            order = np.argpartition(big_mask, n_small)
            pos = pos[order]
            hsml = hsml[order]
            mass = mass[order]
            qty = qty[order]

        # Upload into separate buffers for small and big particles.
        # The partition reorders all 4 arrays with a single index gather.
        if n_small > 0:
            self.pos_vbo = self.ctx.buffer(pos[:n_small].tobytes())
            self.hsml_vbo = self.ctx.buffer(hsml[:n_small].tobytes())
            self.mass_vbo = self.ctx.buffer(mass[:n_small].tobytes())
            self.qty_vbo = self.ctx.buffer(qty[:n_small].tobytes())
        else:
            self.pos_vbo = None
            self.hsml_vbo = None
            self.mass_vbo = None
            self.qty_vbo = None

        if n_big > 0:
            self.big_pos_vbo = self.ctx.buffer(pos[n_small:].tobytes())
            self.big_hsml_vbo = self.ctx.buffer(hsml[n_small:].tobytes())
            self.big_mass_vbo = self.ctx.buffer(mass[n_small:].tobytes())
            self.big_qty_vbo = self.ctx.buffer(qty[n_small:].tobytes())
        else:
            self.big_pos_vbo = None
            self.big_hsml_vbo = None
            self.big_mass_vbo = None
            self.big_qty_vbo = None

        self.n_particles = n_small  # points count
        self._build_vao()

    def _upload_aniso_summaries(self, pos, mass, qty, cov):
        """Upload anisotropic summary splats (separate from regular particles)."""
        for attr in ("aniso_pos_vbo", "aniso_mass_vbo", "aniso_qty_vbo", "aniso_cov_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()
        if self.vao_aniso is not None:
            self.vao_aniso.release()
            self.vao_aniso = None

        self.n_aniso = len(mass)
        if self.n_aniso == 0:
            self.aniso_pos_vbo = None
            self.aniso_mass_vbo = None
            self.aniso_qty_vbo = None
            self.aniso_cov_vbo = None
            return

        self.aniso_pos_vbo = self.ctx.buffer(pos.tobytes())
        self.aniso_mass_vbo = self.ctx.buffer(mass.tobytes())
        self.aniso_qty_vbo = self.ctx.buffer(qty.tobytes())
        self.aniso_cov_vbo = self.ctx.buffer(cov.tobytes())

        self.vao_aniso = self.ctx.vertex_array(
            self.prog_aniso,
            [
                (self.quad_vbo, "2f", "in_corner"),
                (self.aniso_pos_vbo, "3f/i", "in_position"),
                (self.aniso_mass_vbo, "f/i", "in_mass"),
                (self.aniso_qty_vbo, "f/i", "in_quantity"),
                (self.aniso_cov_vbo, "3f 3f/i", "in_cov_a", "in_cov_b"),
            ],
            index_buffer=self.quad_ibo,
        )

    def _upload_subset(self, idx):
        """Upload a subset of particles by index."""
        self._upload_arrays(
            self._all_pos[idx],
            self._all_hsml[idx],
            self._all_mass[idx],
            self._all_qty[idx],
        )

    def upload_particles(self, positions, hsml, masses, quantity=None):
        """Upload all particles directly (for small datasets)."""
        self.set_particles(positions, hsml, masses, quantity)
        self.n_particles = self.n_total

        for attr in ("pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self.pos_vbo = self.ctx.buffer(self._all_pos.tobytes())
        self.hsml_vbo = self.ctx.buffer(self._all_hsml.tobytes())
        self.mass_vbo = self.ctx.buffer(self._all_mass.tobytes())
        self.qty_vbo = self.ctx.buffer(self._all_qty.tobytes())

        self._build_vao()

    def update_quantity(self, quantity):
        """Update just the quantity data."""
        self._all_qty = quantity.astype(np.float32)
        # If we have a subset uploaded, re-upload is deferred to next update_visible()
        # For now, re-upload all
        if self.qty_vbo is not None:
            self.qty_vbo.release()
        self.qty_vbo = self.ctx.buffer(self._all_qty.tobytes())
        self._build_vao()

    def upload_stars(self, positions, masses):
        """Upload star particle data for point sprite rendering."""
        self.n_stars = len(masses)
        if self.n_stars == 0:
            return

        for attr in ("star_pos_vbo", "star_mass_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self.star_pos_vbo = self.ctx.buffer(positions.astype(np.float32).tobytes())
        self.star_mass_vbo = self.ctx.buffer(masses.astype(np.float32).tobytes())

        if self.vao_star is not None:
            self.vao_star.release()
        self.vao_star = self.ctx.vertex_array(
            self.prog_star,
            [
                (self.star_pos_vbo, "3f", "in_position"),
                (self.star_mass_vbo, "f", "in_mass"),
            ],
        )

    def _build_vao(self):
        """(Re)build vertex array objects for points and quads."""
        if self.vao_additive is not None:
            self.vao_additive.release()
            self.vao_additive = None
        if self.vao_quad is not None:
            self.vao_quad.release()
            self.vao_quad = None

        # Point sprites VAO
        if self.pos_vbo is not None:
            self.vao_additive = self.ctx.vertex_array(
                self.prog_additive,
                [
                    (self.pos_vbo, "3f", "in_position"),
                    (self.hsml_vbo, "f", "in_hsml"),
                    (self.mass_vbo, "f", "in_mass"),
                    (self.qty_vbo, "f", "in_quantity"),
                ],
            )

        # Instanced quads VAO for large particles
        if self.big_pos_vbo is not None:
            self.vao_quad = self.ctx.vertex_array(
                self.prog_quad,
                [
                    (self.quad_vbo, "2f", "in_corner"),
                    (self.big_pos_vbo, "3f/i", "in_position"),
                    (self.big_hsml_vbo, "f/i", "in_hsml"),
                    (self.big_mass_vbo, "f/i", "in_mass"),
                    (self.big_qty_vbo, "f/i", "in_quantity"),
                ],
                index_buffer=self.quad_ibo,
            )

    def _ensure_accum_fbo(self, width, height):
        """Create or resize the accumulation FBO."""
        if self._fbo_size == (width, height) and self._accum_fbo is not None:
            return

        for attr in ("_accum_fbo", "_accum_tex_num", "_accum_tex_den"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self._accum_tex_num = self.ctx.texture((width, height), 1, dtype="f4")
        self._accum_tex_den = self.ctx.texture((width, height), 1, dtype="f4")
        self._accum_fbo = self.ctx.framebuffer(
            color_attachments=[self._accum_tex_num, self._accum_tex_den],
        )
        self._fbo_size = (width, height)

    def render(self, camera, width, height):
        """Render particle splats via additive accumulation + resolve."""
        self._viewport_width = width
        if (self.n_particles == 0 and self.n_big == 0) or self.colormap_tex is None:
            return

        # GLSL mat4 is column-major; numpy is row-major
        view = np.ascontiguousarray(camera.view_matrix().T)
        proj = np.ascontiguousarray(camera.projection_matrix().T)

        self._ensure_accum_fbo(width, height)

        # Pass 1: additive accumulation into float FBO
        self._accum_fbo.use()
        self._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)

        kernel_id = self.KERNELS.index(self.kernel)

        self.prog_additive["u_view"].write(view.tobytes())
        self.prog_additive["u_proj"].write(proj.tobytes())
        self.prog_additive["u_viewport_size"].value = (float(width), float(height))
        self.prog_additive["u_kernel"].value = kernel_id

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # pure additive
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Draw small particles as point sprites
        if self.vao_additive is not None and self.n_particles > 0:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.vao_additive.render(moderngl.POINTS)

        # Draw large particles as instanced quads (no point size limit)
        if self.vao_quad is not None and self.n_big > 0:
            self.prog_quad["u_view"].write(view.tobytes())
            self.prog_quad["u_proj"].write(proj.tobytes())
            self.prog_quad["u_kernel"].value = kernel_id
            self.vao_quad.render(moderngl.TRIANGLES, instances=self.n_big)

        # Draw anisotropic summary splats
        if self.vao_aniso is not None and self.n_aniso > 0:
            self.prog_aniso["u_view"].write(view.tobytes())
            self.prog_aniso["u_proj"].write(proj.tobytes())
            self.prog_aniso["u_cov_scale"].value = self.summary_scale
            self.vao_aniso.render(moderngl.TRIANGLES, instances=self.n_aniso)

        # Pass 2: resolve to screen
        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)

        self._accum_tex_num.use(location=0)
        self._accum_tex_den.use(location=1)
        self.colormap_tex.use(location=2)

        self.prog_resolve["u_numerator"].value = 0
        self.prog_resolve["u_denominator"].value = 1
        self.prog_resolve["u_colormap"].value = 2
        self.prog_resolve["u_qty_min"].value = self.qty_min
        self.prog_resolve["u_qty_max"].value = self.qty_max
        self.prog_resolve["u_mode"].value = self.resolve_mode
        self.prog_resolve["u_log_scale"].value = self.log_scale

        self.vao_resolve.render(moderngl.TRIANGLE_STRIP, vertices=4)

        # Pass 3: render star particles on top
        if self.n_stars > 0 and self.vao_star is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

            self.prog_star["u_view"].write(view.tobytes())
            self.prog_star["u_proj"].write(proj.tobytes())
            self.prog_star["u_point_size"].value = self.star_point_size

            self.vao_star.render(moderngl.POINTS)

            self.ctx.disable(moderngl.BLEND)

    def read_accum_range(self):
        """Read back the accumulation textures and compute percentile range.
        Returns (lo, hi) in log10 or linear depending on self.log_scale.
        Must be called after render()."""
        if self._accum_tex_den is None:
            return self.qty_min, self.qty_max

        w, h = self._fbo_size
        # Read denominator (surface density) for resolve_mode 0, or compute ratio for mode 1
        den_data = np.frombuffer(self._accum_tex_den.read(), dtype=np.float32)
        if self.resolve_mode == 1:
            num_data = np.frombuffer(self._accum_tex_num.read(), dtype=np.float32)
            mask = den_data > 1e-30
            vals = np.where(mask, num_data / den_data, 0)
            vals = vals[mask]
        else:
            vals = den_data[den_data > 1e-30]

        if len(vals) == 0:
            return self.qty_min, self.qty_max

        # Mass-weighted CDF limits (matches SinkVis):
        # find values that enclose 1% and 99% of total integrated signal
        sorted_vals = np.sort(vals)
        cdf = sorted_vals.cumsum() / sorted_vals.sum()
        lim_lo = float(np.interp(0.01, cdf, sorted_vals))
        lim_hi = float(np.interp(0.99, cdf, sorted_vals))

        if self.log_scale:
            if lim_lo <= 0:
                lim_lo = float(sorted_vals[sorted_vals > 0].min()) if (sorted_vals > 0).any() else 1e-10
            lo = float(np.log10(max(lim_lo, 1e-30)))
            hi = float(np.log10(max(lim_hi, 1e-30)))
            if hi - lo < 0.1:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1
        else:
            lo, hi = lim_lo, lim_hi
            if hi - lo < 1e-30:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1

        return lo, hi

    def release(self):
        """Clean up GPU resources."""
        for attr in (
            "pos_vbo",
            "hsml_vbo",
            "mass_vbo",
            "qty_vbo",
            "big_pos_vbo",
            "big_hsml_vbo",
            "big_mass_vbo",
            "big_qty_vbo",
            "quad_vbo",
            "quad_ibo",
            "fs_quad_vbo",
            "star_pos_vbo",
            "star_mass_vbo",
            "aniso_pos_vbo",
            "aniso_mass_vbo",
            "aniso_qty_vbo",
            "aniso_cov_vbo",
            "vao_additive",
            "vao_quad",
            "vao_aniso",
            "vao_resolve",
            "vao_star",
            "prog_additive",
            "prog_quad",
            "prog_aniso",
            "prog_resolve",
            "prog_star",
            "_accum_fbo",
            "_accum_tex_num",
            "_accum_tex_den",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
