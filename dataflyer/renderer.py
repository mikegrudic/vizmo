"""Core splat renderer: additive accumulation + resolve pass."""

import numpy as np
import moderngl
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shaders"

# Maximum particles to render per frame for interactive performance
MAX_RENDER_PARTICLES = 4_000_000


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


def _frustum_cull_and_lod(positions, hsml, camera, max_particles):
    """Select visible particles with LOD. Returns index array.

    Strategy:
    1. Frustum cull: discard particles behind camera or outside FOV
    2. If still too many, subsample distant particles (LOD)
    """
    N = len(positions)
    cam_pos = camera.position
    cam_fwd = camera.forward
    cam_right = camera.right
    cam_up = camera.up

    # Camera-space coordinates
    rel = positions - cam_pos
    depths = rel @ cam_fwd    # distance along viewing direction
    rights = rel @ cam_right  # horizontal offset
    ups = rel @ cam_up        # vertical offset

    # Cull behind camera (with smoothing length margin)
    in_front = depths > -hsml

    # Frustum cull: for FOV, tan(fov/2) gives the half-width at depth=1
    half_tan = np.tan(np.radians(camera.fov) / 2)
    # At each particle's depth, the visible half-width is depth * half_tan
    # Include margin for particle size (hsml)
    vis_depth = np.abs(depths) * half_tan + hsml
    in_frustum = in_front & (np.abs(rights) < vis_depth * camera.aspect) & (np.abs(ups) < vis_depth)

    visible_idx = np.where(in_frustum)[0]

    if len(visible_idx) <= max_particles:
        return visible_idx

    # LOD: subsample. Keep all close particles, subsample distant ones.
    vis_depths = depths[visible_idx]
    median_depth = np.median(vis_depths[vis_depths > 0]) if (vis_depths > 0).any() else 1.0

    # Split into near (within 1 median depth) and far
    near_mask = vis_depths < median_depth
    near_idx = visible_idx[near_mask]
    far_idx = visible_idx[~near_mask]

    # How many far particles can we afford?
    budget = max_particles - len(near_idx)
    if budget <= 0:
        # Even near particles exceed budget -- uniform subsample
        step = max(1, len(visible_idx) // max_particles)
        return visible_idx[::step]

    if len(far_idx) > budget:
        # Random subsample of far particles
        rng = np.random.default_rng(42)
        far_idx = rng.choice(far_idx, budget, replace=False)

    return np.concatenate([near_idx, far_idx])


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

    def update_visible(self, camera):
        """Cull and upload only visible particles for this frame."""
        if self._all_pos is None:
            return

        idx = _frustum_cull_and_lod(
            self._all_pos, self._all_hsml, camera, self.max_render_particles
        )

        self._upload_subset(idx)

    def _upload_subset(self, idx):
        """Upload a subset of particles to GPU."""
        self.n_particles = len(idx)

        if self.n_particles == 0:
            return

        # Release old buffers
        for attr in ("pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self.pos_vbo = self.ctx.buffer(self._all_pos[idx].tobytes())
        self.hsml_vbo = self.ctx.buffer(self._all_hsml[idx].tobytes())
        self.mass_vbo = self.ctx.buffer(self._all_mass[idx].tobytes())
        self.qty_vbo = self.ctx.buffer(self._all_qty[idx].tobytes())

        self._build_vao()

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
