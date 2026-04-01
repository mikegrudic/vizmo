"""Core splat renderer: additive accumulation + resolve pass."""

import numpy as np
import moderngl
from pathlib import Path
from numba import njit, prange

SHADER_DIR = Path(__file__).parent / "shaders"


@njit(parallel=True, cache=True)
def _gather_subsampled_direct(cell_start, vis_cells, budget,
                               s_pos, s_hsml, s_mass, s_qty):
    """Gather particle data from visible cells directly from pre-sorted arrays.

    Subsamples within each cell to stay within budget. Reads are sequential
    within each cell (cache-friendly). No visit to the full particle array.
    Returns (pos, hsml, mass, qty, n_visible_total).
    """
    n_cells = len(vis_cells)

    # First pass: count total visible particles
    total = np.int64(0)
    for i in range(n_cells):
        c = vis_cells[i]
        total += cell_start[c + 1] - cell_start[c]

    # Compute global stride
    stride = np.int64(1)
    if total > budget:
        stride = max(np.int64(1), total // budget)

    # Second pass: count output per cell and total
    out_counts = np.empty(n_cells, dtype=np.int64)
    out_total = np.int64(0)
    for i in range(n_cells):
        c = vis_cells[i]
        n = cell_start[c + 1] - cell_start[c]
        kept = (n + stride - 1) // stride
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

    # Parallel gather: each cell copies its strided particles sequentially
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
        cell_idx = np.clip(
            ((positions - self.pmin) / cs).astype(np.int32), 0, n_cells - 1
        )
        cell_id = cell_idx[:, 0] * n_cells * n_cells + cell_idx[:, 1] * n_cells + cell_idx[:, 2]
        self.sort_order = np.argsort(cell_id)
        sorted_cell_id = cell_id[self.sort_order]

        nc3 = n_cells ** 3
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

        nc3 = nc ** 3
        starts = self.cell_start[:-1].astype(np.intp)
        nonempty = starts < self.cell_start[1:]
        reduce_at = starts[nonempty]
        ne_idx = np.where(nonempty)[0]

        cell_mass = np.zeros(nc3, dtype=np.float32)
        cell_com = np.zeros((nc3, 3), dtype=np.float32)
        cell_hsml = np.zeros(nc3, dtype=np.float32)
        cell_qty = np.zeros(nc3, dtype=np.float32)
        # Also store mass-weighted x^2 sums for variance computation at coarser levels
        cell_mx2 = np.zeros((nc3, 3), dtype=np.float32)
        cell_mh2 = np.zeros(nc3, dtype=np.float32)

        if len(reduce_at) > 0:
            mass_ne = np.add.reduceat(s_m, reduce_at)
            safe = np.maximum(mass_ne, 1e-30)
            mp_ne = np.add.reduceat(s_p * s_m[:, None], reduce_at)
            mq_ne = np.add.reduceat(s_m * s_q, reduce_at)
            mh2_ne = np.add.reduceat(s_m * s_h ** 2, reduce_at)
            mp2_ne = np.add.reduceat(s_p ** 2 * s_m[:, None], reduce_at)

            com_ne = mp_ne / safe[:, None]
            var_ne = (mp2_ne / safe[:, None] - com_ne ** 2).sum(axis=1)

            cell_mass[ne_idx] = mass_ne
            cell_com[ne_idx] = com_ne
            cell_qty[ne_idx] = mq_ne / safe
            cell_mx2[ne_idx] = mp2_ne
            cell_mh2[ne_idx] = mh2_ne
            # h from variance matching (cubic spline 2D variance = h^2/12.65 per dim):
            # h = sqrt(4.22 * var_3d + mean(h^2))
            # Floor at 1.5 * cell_spacing so adjacent splats overlap smoothly
            # (cubic spline has compact support at r=h, need h > spacing for continuity)
            h_var = np.sqrt(np.maximum(4.22 * var_ne + mh2_ne / safe, 1e-30))
            h_floor = 1.5 * np.linalg.norm(cs)
            cell_hsml[ne_idx] = np.maximum(h_var, h_floor)

        cx = np.arange(nc, dtype=np.float32) * cs[0] + self.pmin[0] + cs[0] * 0.5
        cy = np.arange(nc, dtype=np.float32) * cs[1] + self.pmin[1] + cs[1] * 0.5
        cz = np.arange(nc, dtype=np.float32) * cs[2] + self.pmin[2] + cs[2] * 0.5
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing='ij')
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        return {
            "nc": nc, "cs": cs,
            "mass": cell_mass, "com": cell_com, "hsml": cell_hsml, "qty": cell_qty,
            "mx2": cell_mx2, "mh2": cell_mh2,  # for coarsening
            "cell_start": self.cell_start, "sort_order": self.sort_order,
            "centers": centers,
            "half_diag": float(np.linalg.norm(cs) * 0.5),
        }

    def _coarsen(self, child):
        """Build a coarser level by merging 2x2x2 groups of child cells."""
        cnc = child["nc"]
        nc = cnc // 2
        cs = child["cs"] * 2
        nc3 = nc ** 3

        # Map child cell (ix, iy, iz) -> parent cell (ix//2, iy//2, iz//2)
        cix = np.arange(cnc)
        ciy = np.arange(cnc)
        ciz = np.arange(cnc)
        gx, gy, gz = np.meshgrid(cix, ciy, ciz, indexing='ij')
        parent_id = (gx.ravel() // 2) * nc * nc + (gy.ravel() // 2) * nc + (gz.ravel() // 2)

        # Sum child properties into parent cells
        cell_mass = np.zeros(nc3, dtype=np.float32)
        cell_mp = np.zeros((nc3, 3), dtype=np.float32)
        cell_mq = np.zeros(nc3, dtype=np.float32)
        cell_mx2 = np.zeros((nc3, 3), dtype=np.float32)
        cell_mh2 = np.zeros(nc3, dtype=np.float32)

        np.add.at(cell_mass, parent_id, child["mass"])
        for d in range(3):
            np.add.at(cell_mp[:, d], parent_id, child["mass"] * child["com"][:, d])
            np.add.at(cell_mx2[:, d], parent_id, child["mx2"][:, d])
        np.add.at(cell_mq, parent_id, child["mass"] * child["qty"])
        np.add.at(cell_mh2, parent_id, child["mh2"])

        safe = np.maximum(cell_mass, 1e-30)
        cell_com = cell_mp / safe[:, None]
        cell_qty = cell_mq / safe
        var = (cell_mx2 / safe[:, None] - cell_com ** 2).sum(axis=1)
        h_var = np.sqrt(np.maximum(4.22 * var + cell_mh2 / safe, 1e-30))
        h_floor = 1.5 * np.linalg.norm(cs)
        cell_hsml = np.maximum(h_var, h_floor)

        cx = np.arange(nc, dtype=np.float32) * cs[0] + self.pmin[0] + cs[0] * 0.5
        cy = np.arange(nc, dtype=np.float32) * cs[1] + self.pmin[1] + cs[1] * 0.5
        cz = np.arange(nc, dtype=np.float32) * cs[2] + self.pmin[2] + cs[2] * 0.5
        gx2, gy2, gz2 = np.meshgrid(cx, cy, cz, indexing='ij')
        centers = np.stack([gx2.ravel(), gy2.ravel(), gz2.ravel()], axis=1)

        return {
            "nc": nc, "cs": cs,
            "mass": cell_mass, "com": cell_com, "hsml": cell_hsml, "qty": cell_qty,
            "mx2": cell_mx2, "mh2": cell_mh2,
            "centers": centers,
            "half_diag": float(np.linalg.norm(cs) * 0.5),
        }

    def query_frustum_lod(self, camera, max_particles, lod_pixels=4):
        """Top-down multi-level LOD query. Returns (pos, hsml, mass, qty) arrays.

        All data comes from pre-sorted arrays built at grid construction time.
        No per-query access to the full particle arrays.

        Traverses coarsest-to-finest. Cells subtending <= lod_pixels emit a
        summary splat (at CoM with variance-based h). Cells subtending more
        are refined to children. At the finest level, real particles are emitted
        via numba parallel gather from the pre-sorted arrays.

        When the particle budget is exceeded, real particles are subsampled
        with mass and h rescaled to conserve the mass distribution.
        """
        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        pix_per_rad = 1024.0 / (2.0 * np.tan(fov_rad / 2))
        half_tan = np.tan(fov_rad / 2) * 2.0  # wide frustum

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
        hd = lv["half_diag"]
        centers = lv["centers"]

        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        in_front = depths > -hd
        lim_h = (depths + hd) * half_tan * camera.aspect + hd
        lim_v = (depths + hd) * half_tan + hd
        visible = in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)
        has_mass = lv["mass"] > 0

        safe_depths = np.maximum(depths, 0.01)
        cell_pix = (hd * 2) / safe_depths * pix_per_rad

        summary_mask = visible & has_mass & (cell_pix <= lod_pixels)
        refine_mask = visible & has_mass & (cell_pix > lod_pixels)

        s_idx = np.where(summary_mask)[0]
        if len(s_idx) > 0:
            summary_parts.append((lv["com"][s_idx], lv["hsml"][s_idx],
                                  lv["mass"][s_idx], lv["qty"][s_idx]))

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

            in_front = depths > -hd
            lim_h = (depths + hd) * half_tan * camera.aspect + hd
            lim_v = (depths + hd) * half_tan + hd
            vis = in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

            safe_depths = np.maximum(depths, 0.01)
            cell_pix = (hd * 2) / safe_depths * pix_per_rad

            small = vis & (cell_pix <= lod_pixels)
            large = vis & (cell_pix > lod_pixels)

            # Summary splats for small cells
            s_idx = child_flat[small]
            if len(s_idx) > 0:
                summary_parts.append((lv["com"][s_idx], lv["hsml"][s_idx],
                                      lv["mass"][s_idx], lv["qty"][s_idx]))

            if li == 0:
                # Finest level: real particles for large cells
                vis_cells = child_flat[large]
                if len(vis_cells) > 0:
                    n_summaries = sum(p[0].shape[0] for p in summary_parts) if summary_parts else 0
                    budget = max(max_particles - n_summaries, max_particles // 2)
                    real_pos, real_hsml, real_mass, real_qty, n_visible_real = \
                        _gather_subsampled_direct(
                            finest["cell_start"], vis_cells.astype(np.int64), budget,
                            self.sorted_pos, self.sorted_hsml,
                            self.sorted_mass, self.sorted_qty,
                        )
            else:
                refine_cells = child_flat[large]

        # Compute mass/h rescaling from subsampling ratio
        n_visible_real = int(n_visible_real)
        n_sampled = len(real_pos) if real_pos is not None else 0

        if n_sampled > 0 and n_visible_real > n_sampled:
            ratio = n_visible_real / n_sampled
            real_mass = real_mass * ratio
            real_hsml = real_hsml * (ratio ** (1.0 / 3.0))

        # Assemble output
        parts_pos = []
        parts_hsml = []
        parts_mass = []
        parts_qty = []

        if n_sampled > 0:
            parts_pos.append(real_pos)
            parts_hsml.append(real_hsml)
            parts_mass.append(real_mass)
            parts_qty.append(real_qty)

        for sp_com, sp_h, sp_m, sp_q in summary_parts:
            parts_pos.append(sp_com)
            parts_hsml.append(sp_h)
            parts_mass.append(sp_m)
            parts_qty.append(sp_q)

        if not parts_pos:
            z3 = np.zeros((0, 3), dtype=np.float32)
            z1 = np.zeros(0, dtype=np.float32)
            return z3, z1, z1, z1

        return (
            np.concatenate(parts_pos).astype(np.float32),
            np.concatenate(parts_hsml).astype(np.float32),
            np.concatenate(parts_mass).astype(np.float32),
            np.concatenate(parts_qty).astype(np.float32),
        )


class SplatRenderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.n_particles = 0
        self.n_total = 0  # total particles in dataset

        # Compile shader programs
        self.prog_additive = ctx.program(
            vertex_shader=_load_shader("splat.vert"),
            fragment_shader=_load_shader("splat_additive.frag"),
        )
        self.prog_resolve = ctx.program(
            vertex_shader=_load_shader("resolve.vert"),
            fragment_shader=_load_shader("resolve.frag"),
        )
        self.prog_star = ctx.program(
            vertex_shader=_load_shader("star.vert"),
            fragment_shader=_load_shader("star.frag"),
        )

        # Quad vertices for billboard (two triangles)
        quad = np.array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ], dtype=np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())

        quad_indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.int32)
        self.quad_ibo = ctx.buffer(quad_indices.tobytes())

        # Fullscreen quad for resolve pass
        fs_quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.fs_quad_vbo = ctx.buffer(fs_quad.tobytes())

        # Particle buffers (set by upload_visible)
        self.pos_vbo = None
        self.hsml_vbo = None
        self.mass_vbo = None
        self.qty_vbo = None
        self.vao_additive = None

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
        self.mode = 0       # 0: surface density, 1: weighted quantity
        self.lod_pixels = 4  # cells subtending fewer pixels than this get summarized
        self.log_scale = 1  # 1: log10, 0: linear
        self.max_render_particles = MAX_RENDER_PARTICLES

    def set_particles(self, positions, hsml, masses, quantity=None):
        """Store particle data on CPU. Call update_visible() to upload a subset."""
        self._all_pos = positions.astype(np.float32)
        self._all_hsml = hsml.astype(np.float32)
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)
        self.n_total = len(masses)

        # Build spatial grid for fast frustum queries on large datasets
        if self.n_total > self.max_render_particles:
            import time
            t0 = time.perf_counter()
            self._grid = SpatialGrid(
                self._all_pos, self._all_mass, self._all_hsml, self._all_qty
            )
            print(f"  Spatial grid built in {time.perf_counter()-t0:.1f}s")
        else:
            self._grid = None

    def update_visible(self, camera):
        """Cull and upload only visible particles for this frame."""
        if self._all_pos is None:
            return

        if self._grid is not None:
            pos, hsml, mass, qty = self._grid.query_frustum_lod(
                camera, self.max_render_particles, lod_pixels=self.lod_pixels,
            )
            self._upload_arrays(pos, hsml, mass, qty)
        else:
            self._upload_subset(np.arange(self.n_total))

    def _upload_arrays(self, pos, hsml, mass, qty):
        """Upload pre-built arrays to GPU."""
        self.n_particles = len(mass)

        if self.n_particles == 0:
            return

        for attr in ("pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self.pos_vbo = self.ctx.buffer(pos.tobytes())
        self.hsml_vbo = self.ctx.buffer(hsml.tobytes())
        self.mass_vbo = self.ctx.buffer(mass.tobytes())
        self.qty_vbo = self.ctx.buffer(qty.tobytes())

        self._build_vao()

    def _upload_subset(self, idx):
        """Upload a subset of particles by index."""
        self._upload_arrays(
            self._all_pos[idx], self._all_hsml[idx],
            self._all_mass[idx], self._all_qty[idx],
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
        """(Re)build vertex array object after buffer changes."""
        if self.vao_additive is not None:
            self.vao_additive.release()

        if self.pos_vbo is None:
            return

        content = [
            (self.quad_vbo, "2f", "in_corner"),
            (self.pos_vbo, "3f/i", "in_position"),
            (self.hsml_vbo, "f/i", "in_hsml"),
            (self.mass_vbo, "f/i", "in_mass"),
            (self.qty_vbo, "f/i", "in_quantity"),
        ]

        self.vao_additive = self.ctx.vertex_array(
            self.prog_additive, content, index_buffer=self.quad_ibo,
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
        if self.n_particles == 0 or self.colormap_tex is None:
            return

        # GLSL mat4 is column-major; numpy is row-major
        view = np.ascontiguousarray(camera.view_matrix().T)
        proj = np.ascontiguousarray(camera.projection_matrix().T)

        self._ensure_accum_fbo(width, height)

        # Pass 1: additive accumulation into float FBO
        self._accum_fbo.use()
        self._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.prog_additive["u_view"].write(view.tobytes())
        self.prog_additive["u_proj"].write(proj.tobytes())
        self.prog_additive["u_viewport_size"].value = (float(width), float(height))

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # pure additive
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.vao_additive.render(
            moderngl.TRIANGLES,
            instances=self.n_particles,
        )

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
        self.prog_resolve["u_mode"].value = self.mode
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
        # Read denominator (surface density) for mode 0, or compute ratio for mode 1
        den_data = np.frombuffer(self._accum_tex_den.read(), dtype=np.float32)
        if self.mode == 1:
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
        for attr in ("pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo",
                     "quad_vbo", "quad_ibo", "fs_quad_vbo",
                     "star_pos_vbo", "star_mass_vbo",
                     "vao_additive", "vao_resolve", "vao_star",
                     "prog_additive", "prog_resolve", "prog_star",
                     "_accum_fbo", "_accum_tex_num", "_accum_tex_den"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
