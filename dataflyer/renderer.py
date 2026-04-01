"""Core splat renderer: additive accumulation + resolve pass."""

import numpy as np
import moderngl
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shaders"

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

        # Build multi-level summaries: level n_cells, n_cells/2, ..., 2
        # Each level stores (mass, com, hsml, qty) per cell
        self.levels = []  # list of dicts, finest first
        level_nc = n_cells
        while level_nc >= 2:
            level = self._build_level(positions, masses, hsml, quantity, level_nc, box)
            self.levels.append(level)
            level_nc //= 2

    def _build_level(self, positions, masses, hsml, quantity, nc, box):
        """Build summary arrays for a single grid level."""
        cs = box / nc
        cell_idx = np.clip(
            ((positions - self.pmin) / cs).astype(np.int32), 0, nc - 1
        )
        cell_id = cell_idx[:, 0] * nc * nc + cell_idx[:, 1] * nc + cell_idx[:, 2]
        order = np.argsort(cell_id)
        sorted_id = cell_id[order]

        nc3 = nc ** 3
        cell_start = np.zeros(nc3 + 1, dtype=np.int64)
        u, c = np.unique(sorted_id, return_counts=True)
        cell_start[u + 1] = c
        np.cumsum(cell_start, out=cell_start)

        # Reduceat for vectorized sums
        starts = cell_start[:-1].astype(np.intp)
        nonempty = starts < cell_start[1:]
        reduce_at = starts[nonempty]
        ne_idx = np.where(nonempty)[0]

        s_m = masses[order]
        s_p = positions[order]
        s_h = hsml[order]
        s_q = quantity[order]

        cell_mass = np.zeros(nc3, dtype=np.float32)
        cell_com = np.zeros((nc3, 3), dtype=np.float32)
        cell_hsml = np.zeros(nc3, dtype=np.float32)
        cell_qty = np.zeros(nc3, dtype=np.float32)

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
            # h = sqrt(spatial_variance + mean_h^2), floored to cell half-diagonal
            h_ne = np.sqrt(np.maximum(var_ne + mh2_ne / safe, 0))
            cell_hsml[ne_idx] = np.maximum(h_ne, np.linalg.norm(cs) * 0.5)

        # Cell centers
        cx = np.arange(nc, dtype=np.float32) * cs[0] + self.pmin[0] + cs[0] * 0.5
        cy = np.arange(nc, dtype=np.float32) * cs[1] + self.pmin[1] + cs[1] * 0.5
        cz = np.arange(nc, dtype=np.float32) * cs[2] + self.pmin[2] + cs[2] * 0.5
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing='ij')
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        return {
            "nc": nc, "cs": cs,
            "cell_start": cell_start, "sort_order": order,
            "mass": cell_mass, "com": cell_com, "hsml": cell_hsml, "qty": cell_qty,
            "centers": centers,
            "half_diag": float(np.linalg.norm(cs) * 0.5),
        }

    def query_frustum_lod(self, camera, positions, hsml, masses, quantity,
                          max_particles, lod_pixels=4):
        """Multi-level LOD query. Returns (pos, hsml, mass, qty) arrays."""
        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        pix_per_rad = 1024.0 / (2.0 * np.tan(fov_rad / 2))
        half_tan = np.tan(fov_rad / 2) * 2.0  # wide frustum

        # Find the coarsest level where cells subtend >= lod_pixels at any depth,
        # then for each visible cell, use the coarsest level that still subtends >= lod_pixels.
        # Strategy: iterate levels coarse->fine. Mark cells as "resolved" at each level.

        # Use finest level for frustum test
        finest = self.levels[0]
        hd = finest["half_diag"]
        centers = finest["centers"]
        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        in_front = depths > -hd
        limit_h = (depths + hd) * half_tan * camera.aspect + hd
        limit_v = (depths + hd) * half_tan + hd
        visible = in_front & (np.abs(rights) < limit_h) & (np.abs(ups) < limit_v)

        safe_depths = np.maximum(depths, 0.01)
        fine_pixels = (hd * 2) / safe_depths * pix_per_rad

        # Cells that are close enough to render at full particle resolution
        near_mask = visible & (fine_pixels > lod_pixels) & (finest["mass"] > 0)
        near_cells = np.where(near_mask)[0]

        # Gather real particles from near cells
        near_starts = finest["cell_start"][near_cells]
        near_ends = finest["cell_start"][near_cells + 1]
        near_counts = (near_ends - near_starts).astype(np.intp)
        real_count = int(near_counts.sum())

        if real_count > 0:
            offsets = np.repeat(near_starts, near_counts)
            within = np.arange(real_count, dtype=np.int64) - np.repeat(
                np.concatenate([[0], np.cumsum(near_counts[:-1])]), near_counts
            )
            all_real = finest["sort_order"][(offsets + within).astype(np.intp)]
        else:
            all_real = np.array([], dtype=np.intp)

        # For far cells, find the best coarse level
        far_mask = visible & (~near_mask) & (finest["mass"] > 0)
        summary_parts = []

        if far_mask.any():
            # Try each coarser level (skip finest = levels[0])
            for level in self.levels[1:]:
                lv_hd = level["half_diag"]
                lv_centers = level["centers"]
                lv_depths = lv_centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
                lv_safe = np.maximum(lv_depths, 0.01)
                lv_pixels = (lv_hd * 2) / lv_safe * pix_per_rad

                # Frustum test at this level
                lv_rights = lv_centers @ cam_right - np.dot(cam_pos, cam_right)
                lv_ups = lv_centers @ cam_up - np.dot(cam_pos, cam_up)
                lv_in_front = lv_depths > -lv_hd
                lv_lim_h = (lv_depths + lv_hd) * half_tan * camera.aspect + lv_hd
                lv_lim_v = (lv_depths + lv_hd) * half_tan + lv_hd
                lv_vis = lv_in_front & (np.abs(lv_rights) < lv_lim_h) & (np.abs(lv_ups) < lv_lim_v)

                # Use this level for cells that subtend [lod_pixels, 2*lod_pixels)
                use = lv_vis & (lv_pixels >= lod_pixels) & (lv_pixels < lod_pixels * 4) & (level["mass"] > 0)
                use_idx = np.where(use)[0]
                if len(use_idx) > 0:
                    summary_parts.append((
                        level["com"][use_idx],
                        level["hsml"][use_idx],
                        level["mass"][use_idx],
                        level["qty"][use_idx],
                    ))

            # Catch-all: coarsest level for anything still unresolved
            coarsest = self.levels[-1]
            lv_hd = coarsest["half_diag"]
            lv_centers = coarsest["centers"]
            lv_depths = lv_centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
            lv_rights = lv_centers @ cam_right - np.dot(cam_pos, cam_right)
            lv_ups = lv_centers @ cam_up - np.dot(cam_pos, cam_up)
            lv_in_front = lv_depths > -lv_hd
            lv_lim_h = (lv_depths + lv_hd) * half_tan * camera.aspect + lv_hd
            lv_lim_v = (lv_depths + lv_hd) * half_tan + lv_hd
            lv_vis = lv_in_front & (np.abs(lv_rights) < lv_lim_h) & (np.abs(lv_ups) < lv_lim_v)
            use = lv_vis & (coarsest["mass"] > 0)
            use_idx = np.where(use)[0]
            if len(use_idx) > 0:
                summary_parts.append((
                    coarsest["com"][use_idx],
                    coarsest["hsml"][use_idx],
                    coarsest["mass"][use_idx],
                    coarsest["qty"][use_idx],
                ))

        # Subsample real particles if over budget
        n_summaries = sum(p[0].shape[0] for p in summary_parts) if summary_parts else 0
        budget = max(max_particles - n_summaries, max_particles // 2)
        if len(all_real) > budget:
            step = max(1, len(all_real) // budget)
            all_real = all_real[::step]

        # Assemble output
        parts_pos = [positions[all_real]] if len(all_real) > 0 else []
        parts_hsml = [hsml[all_real]] if len(all_real) > 0 else []
        parts_mass = [masses[all_real]] if len(all_real) > 0 else []
        parts_qty = [quantity[all_real]] if len(all_real) > 0 else []

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
                camera, self._all_pos, self._all_hsml, self._all_mass, self._all_qty,
                self.max_render_particles, lod_pixels=self.lod_pixels,
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
