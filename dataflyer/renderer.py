"""Core splat renderer: additive accumulation + resolve pass."""

import numpy as np
import moderngl
from pathlib import Path
from dataclasses import dataclass

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
        - masses (weight field) -> the "mass" slot in set_particles
        - quantity              -> the "quantity" slot in set_particles
        - resolve_mode          -> 0 for surface density, 1 for ratio

    Examples:
        SurfaceDensity("Masses"):
            weight=Masses, qty=Masses (unused), resolve_mode=0
            displays: Sigma Masses * W / h^2

        MassWeightedAverage("Temperature"):
            weight=Masses, qty=Temperature, resolve_mode=1
            displays: Sigma(Masses * Temperature * W / h^2) / Sigma(Masses * W / h^2)
    """
    name: str           # display name
    weight_field: str   # field loaded into the mass/weight slot
    qty_field: str      # field loaded into the quantity slot
    resolve_mode: int   # 0: display denominator, 1: display num/denom

    @staticmethod
    def surface_density(weight_field="Masses"):
        """Create a surface density render mode for the given weight field."""
        return RenderMode(
            name=f"Sigma {weight_field}",
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

    @staticmethod
    def weighted_variance(qty_field, weight_field="Masses"):
        """Create a weighted variance render mode: sqrt(<f^2> - <f>^2)."""
        return RenderMode(
            name=f"sigma({qty_field})",
            weight_field=weight_field,
            qty_field=qty_field,
            resolve_mode=2,
        )


# Maximum particles to render per frame for interactive performance
MAX_RENDER_PARTICLES = 4_000_000


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


from .spatial_grid import SpatialGrid  # noqa: E402


class ParticleLayer:
    """A renderable particle type with its own shader, buffers, and render method.

    Decouples particle types from the main renderer. New types (dark matter,
    sinks, dust) can be added by creating a ParticleLayer without modifying
    SplatRenderer.
    """

    def __init__(self, ctx, program, uniforms=None):
        """
        Args:
            ctx: ModernGL context
            program: Compiled shader program
            uniforms: Dict of uniform name -> value to set before rendering
        """
        self.ctx = ctx
        self.program = program
        self.uniforms = uniforms or {}
        self.bufs = BufferSet(ctx)
        self.vao = None
        self.n_particles = 0
        self.visible = True
        self.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        self.render_mode = moderngl.POINTS
        self.point_size = True  # enable PROGRAM_POINT_SIZE

    def upload(self, vao_spec, n_particles):
        """Upload buffers and build VAO.

        Args:
            vao_spec: List of (buffer_name, format, *attrib_names) for vertex_array
            n_particles: Number of particles
        """
        if self.vao is not None:
            self.vao.release()
        self.n_particles = n_particles
        if n_particles == 0:
            self.vao = None
            return
        content = []
        for spec in vao_spec:
            buf_name = spec[0]
            fmt = spec[1]
            attribs = spec[2:]
            content.append((self.bufs.get(buf_name), fmt, *attribs))
        self.vao = self.ctx.vertex_array(self.program, content)

    def render(self, view_bytes, proj_bytes):
        """Render this layer. view_bytes and proj_bytes are column-major mat4."""
        if not self.visible or self.vao is None or self.n_particles == 0:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = self.blend_func
        if self.point_size:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        self.program["u_view"].write(view_bytes)
        self.program["u_proj"].write(proj_bytes)
        for name, value in self.uniforms.items():
            self.program[name].value = value

        self.vao.render(self.render_mode)
        self.ctx.disable(moderngl.BLEND)

    def release(self):
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.bufs.release_all()


class BufferSet:
    """Manages a named group of GPU buffers with automatic lifecycle."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._buffers = {}

    def upload(self, name, data):
        """Upload data to a named buffer, releasing any previous buffer."""
        if name in self._buffers:
            self._buffers[name].release()
        self._buffers[name] = self._ctx.buffer(data.tobytes())

    def get(self, name):
        """Get a buffer by name, or None."""
        return self._buffers.get(name)

    def release_all(self):
        """Release all buffers."""
        for buf in self._buffers.values():
            try:
                buf.release()
            except Exception:
                pass
        self._buffers.clear()

    def __contains__(self, name):
        return name in self._buffers

    def __bool__(self):
        return bool(self._buffers)


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
        self.prog_composite = ctx.program(
            vertex_shader=_load_shader("resolve.vert"),
            fragment_shader=_load_shader("composite.frag"),
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

        # FBO for accumulation (3 float textures: numerator, denominator, squared)
        self._accum_fbo = None
        self._accum_tex_num = None
        self._accum_tex_den = None
        self._accum_tex_sq = None
        self._fbo_size = (0, 0)
        self._viewport_width = 1024  # updated each frame in render()

        # Second FBO for composite mode (field 2)
        self._accum_fbo2 = None
        self._accum_tex_num2 = None
        self._accum_tex_den2 = None
        self._accum_tex_sq2 = None
        self._fbo_size2 = (0, 0)

        # Resolve VAOs
        self.vao_resolve = ctx.vertex_array(
            self.prog_resolve,
            [(self.fs_quad_vbo, "2f", "in_position")],
        )
        self.vao_composite = ctx.vertex_array(
            self.prog_composite,
            [(self.fs_quad_vbo, "2f", "in_position")],
        )

        # Star particle layer
        self.star_layer = ParticleLayer(ctx, self.prog_star, {"u_point_size": 50.0})
        self.n_stars = 0

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
        self.hsml_scale = 1.0  # global scaling factor for smoothing lengths
        self.summary_scale = 1.0  # scaling factor applied to summary splats
        self.summary_overlap = 0.1  # cell-size padding to bridge voids at tree boundaries
        self.use_aniso_summaries = True  # False = isotropic spherical summaries
        self.bypass_cull = False  # render all particles without frustum culling
        self.auto_lod = True  # auto-tune LOD to maintain target FPS while moving
        self.target_fps = 15.0  # target FPS for auto-LOD
        self.auto_lod_smooth = 1.0  # EMA smoothing timescale in seconds
        self.pid_Kp = 4.0  # PID proportional gain (log2-units/sec per unit error)
        self.pid_Ki = 0.0  # PID integral gain
        self.pid_Kd = 0.0  # PID derivative gain (0 recommended — D amplifies noise)
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

    def update_weights(self, masses, quantity=None):
        """Update weight/quantity arrays without rebuilding grid structure.

        Much faster than set_particles() when only the weight field changes
        (skips argsort + cell assignment). Positions and hsml are unchanged.
        """
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)

        if self._grid is not None:
            import time
            t0 = time.perf_counter()
            self._grid.update_weights(masses, quantity)
            print(f"  Grid re-weighted in {time.perf_counter()-t0:.1f}s")
        # No need to rebuild _grid from scratch — structure is position-only

    def update_visible(self, camera):
        """Cull and upload only visible particles for this frame."""
        import time as _time
        self._last_cull_ms = 0.0
        self._last_upload_ms = 0.0

        if self._all_pos is None:
            return

        if self.bypass_cull:
            t0 = _time.perf_counter()
            self._upload_arrays(self._all_pos, self._all_hsml, self._all_mass, self._all_qty, camera)
            self._last_upload_ms = (_time.perf_counter() - t0) * 1000
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
            t0 = _time.perf_counter()
            result = self._grid.query_frustum_lod(
                camera,
                self.max_render_particles,
                lod_pixels=self.lod_pixels,
                importance_sampling=self.use_importance_sampling,
                viewport_width=self._viewport_width,
                summary_overlap=self.summary_overlap,
            )
            self._last_cull_ms = (_time.perf_counter() - t0) * 1000
            t0 = _time.perf_counter()
            if len(result) == 9:
                r_pos, r_hsml, r_mass, r_qty, s_pos, s_hsml, s_mass, s_qty, s_cov = result
                if self.use_aniso_summaries:
                    self._upload_arrays(r_pos, r_hsml, r_mass, r_qty, camera)
                    self._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
                else:
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
            t0 = _time.perf_counter()
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
        self._last_upload_ms = (_time.perf_counter() - t0) * 1000

    def _upload_arrays(self, pos, hsml, mass, qty, camera=None):
        """Upload pre-built arrays to GPU, splitting into points and quads."""
        self.n_particles = len(mass)
        self.n_big = 0

        if self.n_particles == 0:
            return

        # Apply global smoothing length scaling
        if self.hsml_scale != 1.0:
            hsml = hsml * self.hsml_scale

        # Release old buffers
        for attr in (
            "pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo",
            "big_pos_vbo", "big_hsml_vbo", "big_mass_vbo", "big_qty_vbo",
        ):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        # Quad-only mode
        if self.use_quad_rendering:
            self.pos_vbo = self.hsml_vbo = self.mass_vbo = self.qty_vbo = None
            self.big_pos_vbo = self.ctx.buffer(pos.tobytes())
            self.big_hsml_vbo = self.ctx.buffer(hsml.tobytes())
            self.big_mass_vbo = self.ctx.buffer(mass.tobytes())
            self.big_qty_vbo = self.ctx.buffer(qty.tobytes())
            self.n_particles = 0
            self.n_big = len(mass)
            self._build_vao()
            return

        # Split into small (point sprites) and big (instanced quads)
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
            order = np.argpartition(big_mask, n_small)
            pos = pos[order]
            hsml = hsml[order]
            mass = mass[order]
            qty = qty[order]

        if n_small > 0:
            self.pos_vbo = self.ctx.buffer(pos[:n_small].tobytes())
            self.hsml_vbo = self.ctx.buffer(hsml[:n_small].tobytes())
            self.mass_vbo = self.ctx.buffer(mass[:n_small].tobytes())
            self.qty_vbo = self.ctx.buffer(qty[:n_small].tobytes())
        else:
            self.pos_vbo = self.hsml_vbo = self.mass_vbo = self.qty_vbo = None

        if n_big > 0:
            self.big_pos_vbo = self.ctx.buffer(pos[n_small:].tobytes())
            self.big_hsml_vbo = self.ctx.buffer(hsml[n_small:].tobytes())
            self.big_mass_vbo = self.ctx.buffer(mass[n_small:].tobytes())
            self.big_qty_vbo = self.ctx.buffer(qty[n_small:].tobytes())
        else:
            self.big_pos_vbo = self.big_hsml_vbo = self.big_mass_vbo = self.big_qty_vbo = None

        self.n_particles = n_small
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
            self.aniso_pos_vbo = self.aniso_mass_vbo = self.aniso_qty_vbo = self.aniso_cov_vbo = None
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

    def update_quantity(self, quantity):
        """Update just the quantity data."""
        self._all_qty = quantity.astype(np.float32)
        if self.qty_vbo is not None:
            self.qty_vbo.release()
        self.qty_vbo = self.ctx.buffer(self._all_qty.tobytes())
        self._build_vao()

    def upload_stars(self, positions, masses):
        """Upload star particle data for point sprite rendering."""
        self.n_stars = len(masses)
        if self.n_stars == 0:
            return
        self.star_layer.bufs.upload("pos", positions.astype(np.float32))
        self.star_layer.bufs.upload("mass", masses.astype(np.float32))
        self.star_layer.upload(
            [("pos", "3f", "in_position"), ("mass", "f", "in_mass")],
            self.n_stars,
        )

    def _build_vao(self):
        """(Re)build vertex array objects for points and quads."""
        if self.vao_additive is not None:
            self.vao_additive.release()
            self.vao_additive = None
        if self.vao_quad is not None:
            self.vao_quad.release()
            self.vao_quad = None

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

    def _ensure_fbo(self, width, height, which=1):
        """Create or resize an accumulation FBO (1=primary, 2=composite second)."""
        if which == 1:
            if self._fbo_size == (width, height) and self._accum_fbo is not None:
                return
            for attr in ("_accum_fbo", "_accum_tex_num", "_accum_tex_den", "_accum_tex_sq"):
                old = getattr(self, attr, None)
                if old is not None:
                    old.release()
            self._accum_tex_num = self.ctx.texture((width, height), 1, dtype="f4")
            self._accum_tex_den = self.ctx.texture((width, height), 1, dtype="f4")
            self._accum_tex_sq = self.ctx.texture((width, height), 1, dtype="f4")
            self._accum_fbo = self.ctx.framebuffer(
                color_attachments=[self._accum_tex_num, self._accum_tex_den, self._accum_tex_sq],
            )
            self._fbo_size = (width, height)
        else:
            if self._fbo_size2 == (width, height) and self._accum_fbo2 is not None:
                return
            for attr in ("_accum_fbo2", "_accum_tex_num2", "_accum_tex_den2", "_accum_tex_sq2"):
                old = getattr(self, attr, None)
                if old is not None:
                    old.release()
            self._accum_tex_num2 = self.ctx.texture((width, height), 1, dtype="f4")
            self._accum_tex_den2 = self.ctx.texture((width, height), 1, dtype="f4")
            self._accum_tex_sq2 = self.ctx.texture((width, height), 1, dtype="f4")
            self._accum_fbo2 = self.ctx.framebuffer(
                color_attachments=[self._accum_tex_num2, self._accum_tex_den2, self._accum_tex_sq2],
            )
            self._fbo_size2 = (width, height)

    def _render_accum(self, camera, width, height, fbo):
        """Render the additive accumulation pass into the given FBO."""
        view = np.ascontiguousarray(camera.view_matrix().T)
        proj = np.ascontiguousarray(camera.projection_matrix().T)

        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)

        kernel_id = self.KERNELS.index(self.kernel)

        self.prog_additive["u_view"].write(view.tobytes())
        self.prog_additive["u_proj"].write(proj.tobytes())
        self.prog_additive["u_viewport_size"].value = (float(width), float(height))
        self.prog_additive["u_kernel"].value = kernel_id

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self.ctx.disable(moderngl.DEPTH_TEST)

        if self.vao_additive is not None and self.n_particles > 0:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.vao_additive.render(moderngl.POINTS)

        if self.vao_quad is not None and self.n_big > 0:
            self.prog_quad["u_view"].write(view.tobytes())
            self.prog_quad["u_proj"].write(proj.tobytes())
            self.prog_quad["u_kernel"].value = kernel_id
            self.vao_quad.render(moderngl.TRIANGLES, instances=self.n_big)

        if self.vao_aniso is not None and self.n_aniso > 0:
            self.prog_aniso["u_view"].write(view.tobytes())
            self.prog_aniso["u_proj"].write(proj.tobytes())
            self.prog_aniso["u_cov_scale"].value = self.summary_scale
            self.vao_aniso.render(moderngl.TRIANGLES, instances=self.n_aniso)

    def render(self, camera, width, height):
        """Render particle splats via additive accumulation + resolve."""
        self._viewport_width = width
        if (self.n_particles == 0 and self.n_big == 0) or self.colormap_tex is None:
            return

        self._ensure_fbo(width, height, which=1)
        self._render_accum(camera, width, height, self._accum_fbo)

        # Resolve to screen
        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)

        self._accum_tex_num.use(location=0)
        self._accum_tex_den.use(location=1)
        self._accum_tex_sq.use(location=2)
        self.colormap_tex.use(location=3)

        self.prog_resolve["u_numerator"].value = 0
        self.prog_resolve["u_denominator"].value = 1
        self.prog_resolve["u_sq"].value = 2
        self.prog_resolve["u_colormap"].value = 3
        self.prog_resolve["u_qty_min"].value = self.qty_min
        self.prog_resolve["u_qty_max"].value = self.qty_max
        self.prog_resolve["u_mode"].value = self.resolve_mode
        self.prog_resolve["u_log_scale"].value = self.log_scale

        self.vao_resolve.render(moderngl.TRIANGLE_STRIP, vertices=4)

        # Star particles on top
        view = np.ascontiguousarray(camera.view_matrix().T)
        proj = np.ascontiguousarray(camera.projection_matrix().T)
        self.star_layer.render(view.tobytes(), proj.tobytes())

    def render_composite(self, camera, width, height,
                         mode1, min1, max1, log1,
                         mode2, min2, max2, log2):
        """Render two fields and composite: field1=lightness, field2=color.

        Call update_weights() with field1 data, then render_composite() which:
        1. Renders field1 into FBO1
        2. Swaps in field2 data via update_weights()
        3. Renders field2 into FBO2
        4. Composites with HSV blend

        Instead, the caller should:
        1. Set up field1 weights/qty → call update_visible → render_composite stores FBO1
        2. field2 weights/qty and FBO2 are passed as parameters

        Actually, the simplest approach: the caller renders twice externally.
        This method just does the composite resolve from two pre-filled FBOs.
        """
        self._viewport_width = width
        self._ensure_fbo(width, height, which=1)
        self._ensure_fbo(width, height, which=2)

        # Composite resolve
        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)

        self._accum_tex_num.use(location=0)
        self._accum_tex_den.use(location=1)
        self._accum_tex_sq.use(location=2)
        self._accum_tex_num2.use(location=3)
        self._accum_tex_den2.use(location=4)
        self._accum_tex_sq2.use(location=5)
        self.colormap_tex.use(location=6)

        p = self.prog_composite
        p["u_num1"].value = 0
        p["u_den1"].value = 1
        p["u_sq1"].value = 2
        p["u_num2"].value = 3
        p["u_den2"].value = 4
        p["u_sq2"].value = 5
        p["u_colormap"].value = 6
        p["u_mode1"].value = mode1
        p["u_min1"].value = min1
        p["u_max1"].value = max1
        p["u_log1"].value = log1
        p["u_mode2"].value = mode2
        p["u_min2"].value = min2
        p["u_max2"].value = max2
        p["u_log2"].value = log2

        self.vao_composite.render(moderngl.TRIANGLE_STRIP, vertices=4)

        # Stars on top
        view = np.ascontiguousarray(camera.view_matrix().T)
        proj = np.ascontiguousarray(camera.projection_matrix().T)
        self.star_layer.render(view.tobytes(), proj.tobytes())

    def read_accum_range(self):
        """Read back the accumulation textures and compute percentile range."""
        if self._accum_tex_den is None:
            return self.qty_min, self.qty_max

        den_data = np.frombuffer(self._accum_tex_den.read(), dtype=np.float32)
        if self.resolve_mode == 2:
            # Variance: sqrt(<f²> - <f>²)
            num_data = np.frombuffer(self._accum_tex_num.read(), dtype=np.float32)
            sq_data = np.frombuffer(self._accum_tex_sq.read(), dtype=np.float32)
            mask = den_data > 1e-30
            with np.errstate(invalid="ignore"):
                mean = np.where(mask, num_data / den_data, 0)
                mean_sq = np.where(mask, sq_data / den_data, 0)
            vals = np.sqrt(np.maximum(mean_sq - mean * mean, 0))[mask]
        elif self.resolve_mode == 1:
            num_data = np.frombuffer(self._accum_tex_num.read(), dtype=np.float32)
            mask = den_data > 1e-30
            with np.errstate(invalid="ignore"):
                vals = np.where(mask, num_data / den_data, 0)
            vals = vals[mask]
        else:
            vals = den_data[den_data > 1e-30]

        if len(vals) == 0:
            return self.qty_min, self.qty_max

        has_negative = (vals < 0).any()

        if has_negative:
            # Negative values: O(N) partition for 1st/99th percentiles
            n = len(vals)
            k_lo = max(n // 100, 0)
            k_hi = min(99 * n // 100, n - 1)
            partitioned = np.partition(vals, (k_lo, k_hi))
            lim_lo = float(partitioned[k_lo])
            lim_hi = float(partitioned[k_hi])
        else:
            # All positive: mass-weighted CDF for signal-aware limits.
            # Subsample to cap sort cost at ~100k elements.
            if len(vals) > 100_000:
                step = len(vals) // 100_000
                sub = vals[::step]
            else:
                sub = vals
            sorted_vals = np.sort(sub)
            cdf = sorted_vals.cumsum() / sorted_vals.sum()
            lim_lo = float(np.interp(0.01, cdf, sorted_vals))
            lim_hi = float(np.interp(0.99, cdf, sorted_vals))

        if self.log_scale and not has_negative:
            if lim_lo <= 0:
                lim_lo = float(vals[vals > 0].min()) if (vals > 0).any() else 1e-10
            lo = float(np.log10(max(lim_lo, 1e-30)))
            hi = float(np.log10(max(lim_hi, 1e-30)))
            if hi - lo < 0.1:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1
        else:
            # Linear scale (or forced linear when negative)
            if self.log_scale and has_negative:
                self.log_scale = 0  # auto-switch to linear for negative data
            lo, hi = lim_lo, lim_hi
            if hi - lo < 1e-30:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1

        return lo, hi

    def release(self):
        """Clean up GPU resources."""
        for attr in (
            "pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo",
            "big_pos_vbo", "big_hsml_vbo", "big_mass_vbo", "big_qty_vbo",
            "quad_vbo", "quad_ibo", "fs_quad_vbo",
            "aniso_pos_vbo", "aniso_mass_vbo", "aniso_qty_vbo", "aniso_cov_vbo",
            "vao_additive", "vao_quad", "vao_aniso", "vao_resolve", "vao_composite",
            "prog_additive", "prog_quad", "prog_aniso", "prog_resolve", "prog_composite", "prog_star",
            "_accum_fbo", "_accum_tex_num", "_accum_tex_den", "_accum_tex_sq",
            "_accum_fbo2", "_accum_tex_num2", "_accum_tex_den2", "_accum_tex_sq2",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self.star_layer.release()
