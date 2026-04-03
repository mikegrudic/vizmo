"""wgpu-based renderer: additive accumulation + resolve, matching SplatRenderer's interface."""

import time
import numpy as np
from pathlib import Path

import wgpu

SHADER_DIR = Path(__file__).parent / "shaders"


def _load_wgsl(name):
    return (SHADER_DIR / name).read_text()


def _additive_blend():
    """Additive blending for accumulation passes."""
    return {
        "color": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
        "alpha": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
    }


def _alpha_blend():
    """Premultiplied alpha blending for star overlay."""
    return {
        "color": {"operation": "add", "src_factor": "one", "dst_factor": "one-minus-src-alpha"},
        "alpha": {"operation": "add", "src_factor": "one", "dst_factor": "one-minus-src-alpha"},
    }


# Accumulation texture format.
# r32float with additive blending requires "float32-blendable" feature (not on macOS Metal).
# rgba16float is blendable by default everywhere. Shaders output vec4 so both formats work.
ACCUM_FORMAT = "r32float"
ACCUM_FORMAT_FALLBACK = "rgba16float"


class WGPURenderer:
    """wgpu renderer matching SplatRenderer's public interface."""

    # Kernel names (must match SplatRenderer.KERNELS)
    KERNELS = ["cubic_spline", "wendland_c2", "gaussian", "quartic", "sphere"]

    def __init__(self, device, canvas_context=None, present_format=None):
        self.device = device
        self.canvas_context = canvas_context
        self.present_format = present_format or "bgra8unorm"

        # Check if float32-blendable is available
        features = device.features
        if "float32-blendable" in features:
            self._accum_format = ACCUM_FORMAT
        else:
            self._accum_format = ACCUM_FORMAT_FALLBACK
            print(f"  wgpu: float32-blendable not available, using {ACCUM_FORMAT_FALLBACK}")

        # Render state (matches SplatRenderer attributes)
        self.n_particles = 0
        self.n_total = 0
        self.n_big = 0
        self.n_aniso = 0
        self.n_stars = 0
        self.alpha_scale = 1.0
        self.qty_min = -1.0
        self.qty_max = 3.0
        self.resolve_mode = 0
        self.lod_pixels = 4
        self.log_scale = 1
        self.max_render_particles = 4_000_000
        self.use_tree = True
        self.tree_min_particles = 0
        self.use_importance_sampling = False
        self.kernel = "cubic_spline"
        self.use_hybrid_rendering = True
        self.use_quad_rendering = False
        self.hsml_scale = 1.0
        self.summary_scale = 1.0
        self.summary_overlap = 0.1
        self.use_aniso_summaries = True
        self.bypass_cull = False
        self.auto_lod = True
        self.target_fps = 15.0
        self.auto_lod_smooth = 1.0
        self.cull_interval = 0.5
        self._needs_grid_rebuild = False

        # CPU-side particle data for culling
        self._all_pos = None
        self._all_hsml = None
        self._all_mass = None
        self._all_qty = None

        # Spatial grid (CPU culling, same as SplatRenderer)
        self._grid = None

        # Timing
        self._last_cull_ms = 0.0
        self._last_upload_ms = 0.0
        self._viewport_width = 1024

        # GPU resources
        self._accum_textures = None  # (num, den, sq) texture triple
        self._accum_textures2 = None  # second set for composite
        self._accum_size = (0, 0)
        self._accum_size2 = (0, 0)

        # Particle storage buffers (SoA)
        self._particle_bufs = {}  # "pos", "hsml", "mass", "qty"
        self._aniso_bufs = {}  # "pos", "mass", "qty", "cov"
        self._star_bufs = {}  # "pos", "mass"

        # Colormap
        self._colormap_tex = None
        self._colormap_sampler = None
        self.colormap_tex = None  # placeholder for compatibility

        # Compile shaders and create pipeline layouts
        self._init_pipelines()

    def _init_pipelines(self):
        """Compile all shader modules and create pipeline layouts."""
        dev = self.device

        # Shader modules
        self._splat_shader = dev.create_shader_module(code=_load_wgsl("splat_quad.wgsl"))
        self._aniso_shader = dev.create_shader_module(code=_load_wgsl("splat_aniso.wgsl"))
        self._resolve_shader = dev.create_shader_module(code=_load_wgsl("resolve.wgsl"))
        self._composite_shader = dev.create_shader_module(code=_load_wgsl("composite.wgsl"))
        self._star_shader = dev.create_shader_module(code=_load_wgsl("star.wgsl"))

        # Camera uniform buffer (shared by splat and star shaders)
        # struct Camera { view: mat4x4, proj: mat4x4, viewport_size: vec2, kernel_id: u32, _pad: u32 }
        # = 64 + 64 + 8 + 4 + 4 = 144 bytes
        self._camera_buf = dev.create_buffer(
            size=144, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # Aniso params uniform buffer
        # struct AnisoParams { cov_scale: f32, _pad: vec3<f32> } = 16 bytes
        self._aniso_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # Star params uniform buffer
        # struct StarParams { point_size: f32, _pad: vec3<f32> } = 16 bytes
        self._star_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # Resolve params uniform buffer
        # struct ResolveParams { qty_min: f32, qty_max: f32, mode: u32, log_scale: u32 } = 16 bytes
        self._resolve_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # Composite params uniform buffer = 32 bytes
        self._composite_params_buf = dev.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # Bind group layouts

        # Group 0 for splat: camera uniform
        self._splat_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "uniform"}},
        ])

        # Group 1 for splat: particle storage buffers (pos, hsml, mass, qty)
        self._splat_bgl1 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
        ])

        # Group 0 for aniso: camera + aniso_params
        self._aniso_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "uniform"}},
        ])

        # Group 1 for aniso: pos, mass, qty, cov storage buffers
        self._aniso_bgl1 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
        ])

        # Group 0 for star: camera + star_params
        self._star_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "uniform"}},
        ])

        # Group 1 for star: pos, mass
        self._star_bgl1 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
        ])

        # Group 2 for splat: sort index buffer
        self._splat_bgl2 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}},
        ])

        # Pipeline layouts
        self._splat_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._splat_bgl0, self._splat_bgl1, self._splat_bgl2]
        )
        self._aniso_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._aniso_bgl0, self._aniso_bgl1]
        )
        self._star_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._star_bgl0, self._star_bgl1]
        )

        # Resolve and composite use "auto" layout (simpler, no storage bufs)
        # We'll create these when we know the texture format.

        # Create splat pipeline (accumulation into 3 float targets with additive blend)
        accum_targets = [
            {"format": self._accum_format, "blend": _additive_blend()},
            {"format": self._accum_format, "blend": _additive_blend()},
            {"format": self._accum_format, "blend": _additive_blend()},
        ]

        self._splat_pipeline = dev.create_render_pipeline(
            layout=self._splat_layout,
            vertex={
                "module": self._splat_shader,
                "entry_point": "vs_main",
                "buffers": [],  # all data via storage buffers
            },
            primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
            fragment={
                "module": self._splat_shader,
                "entry_point": "fs_main",
                "targets": accum_targets,
            },
        )

        self._aniso_pipeline = dev.create_render_pipeline(
            layout=self._aniso_layout,
            vertex={
                "module": self._aniso_shader,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
            fragment={
                "module": self._aniso_shader,
                "entry_point": "fs_main",
                "targets": accum_targets,
            },
        )

        # Star pipeline renders to screen format with alpha blending
        self._star_pipeline = dev.create_render_pipeline(
            layout=self._star_layout,
            vertex={
                "module": self._star_shader,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
            fragment={
                "module": self._star_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format, "blend": _alpha_blend()}],
            },
        )

        # Bind group for camera uniform (shared)
        self._camera_bg = dev.create_bind_group(
            layout=self._splat_bgl0,
            entries=[{"binding": 0, "resource": {"buffer": self._camera_buf}}],
        )

        # Bind group for aniso group 0
        self._aniso_bg0 = dev.create_bind_group(
            layout=self._aniso_bgl0,
            entries=[
                {"binding": 0, "resource": {"buffer": self._camera_buf}},
                {"binding": 1, "resource": {"buffer": self._aniso_params_buf}},
            ],
        )

        # Bind group for star group 0
        self._star_bg0 = dev.create_bind_group(
            layout=self._star_bgl0,
            entries=[
                {"binding": 0, "resource": {"buffer": self._camera_buf}},
                {"binding": 1, "resource": {"buffer": self._star_params_buf}},
            ],
        )

    def set_colormap(self, rgba_data):
        """Set colormap from RGBA uint8 array of shape (N, 4).

        Args:
            rgba_data: numpy array of shape (N, 4) dtype uint8
        """
        dev = self.device
        n = len(rgba_data)

        self._colormap_tex = dev.create_texture(
            size=(n, 1, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        dev.queue.write_texture(
            {"texture": self._colormap_tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba_data.tobytes(),
            {"bytes_per_row": n * 4, "rows_per_image": 1},
            (n, 1, 1),
        )

        self._colormap_sampler = dev.create_sampler(
            mag_filter="linear", min_filter="linear",
            address_mode_u="clamp-to-edge",
        )

        # Rebuild resolve/composite pipelines and bind groups since they reference the colormap
        self._build_resolve_pipeline()
        self._build_composite_pipeline()
        self.colormap_tex = True  # compatibility flag

    def _build_resolve_pipeline(self):
        """Build resolve pipeline and bind group layout."""
        if self._colormap_tex is None:
            return
        dev = self.device

        self._resolve_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
            {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
        ])

        resolve_layout = dev.create_pipeline_layout(bind_group_layouts=[self._resolve_bgl])

        self._resolve_pipeline = dev.create_render_pipeline(
            layout=resolve_layout,
            vertex={"module": self._resolve_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list"},
            fragment={
                "module": self._resolve_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format}],
            },
        )

    def _build_composite_pipeline(self):
        """Build composite pipeline and bind group layout."""
        if self._colormap_tex is None:
            return
        dev = self.device

        self._composite_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 6, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "unfilterable-float"}},
            {"binding": 7, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
            {"binding": 8, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
        ])

        composite_layout = dev.create_pipeline_layout(bind_group_layouts=[self._composite_bgl])

        self._composite_pipeline = dev.create_render_pipeline(
            layout=composite_layout,
            vertex={"module": self._composite_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list"},
            fragment={
                "module": self._composite_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format}],
            },
        )

    # ---- Data management (matches SplatRenderer interface) ----

    def set_particles(self, positions, hsml, masses, quantity=None):
        """Store particle data on CPU. Call update_visible() to upload a subset."""
        from .spatial_grid import SpatialGrid

        self._all_pos = positions.astype(np.float32)
        self._all_hsml = hsml.astype(np.float32)
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)
        self.n_total = len(masses)

        if self.use_tree and self.n_total > self.tree_min_particles:
            t0 = time.perf_counter()
            self._grid = SpatialGrid(self._all_pos, self._all_mass, self._all_hsml, self._all_qty)
            print(f"  Spatial grid built in {time.perf_counter()-t0:.1f}s")
        else:
            self._grid = None

    def update_weights(self, masses, quantity=None):
        """Update weight/quantity arrays without rebuilding grid structure."""
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)

        if self._grid is not None:
            t0 = time.perf_counter()
            self._grid.update_weights(masses, quantity)
            print(f"  Grid re-weighted in {time.perf_counter()-t0:.1f}s")

    def update_visible(self, camera):
        """Cull and upload only visible particles for this frame."""
        self._last_cull_ms = 0.0
        self._last_upload_ms = 0.0

        if self._all_pos is None:
            return

        if self.bypass_cull:
            t0 = time.perf_counter()
            self._upload_arrays(self._all_pos, self._all_hsml, self._all_mass, self._all_qty, camera)
            self._last_upload_ms = (time.perf_counter() - t0) * 1000
            return

        if self._needs_grid_rebuild:
            self._needs_grid_rebuild = False
            from .spatial_grid import SpatialGrid
            if self.use_tree and self.n_total > self.tree_min_particles:
                t0 = time.perf_counter()
                self._grid = SpatialGrid(self._all_pos, self._all_mass, self._all_hsml, self._all_qty)
                print(f"  Spatial grid rebuilt in {time.perf_counter()-t0:.1f}s")
            else:
                self._grid = None

        if self._grid is not None:
            t0 = time.perf_counter()
            result = self._grid.query_frustum_lod(
                camera, self.max_render_particles,
                lod_pixels=self.lod_pixels,
                importance_sampling=self.use_importance_sampling,
                viewport_width=self._viewport_width,
                summary_overlap=self.summary_overlap,
            )
            self._last_cull_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
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
            # No tree: simple frustum cull
            t0 = time.perf_counter()
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
        self._last_upload_ms = (time.perf_counter() - t0) * 1000

    def _upload_arrays(self, pos, hsml, mass, qty, camera=None):
        """Upload particle arrays to GPU storage buffers."""
        n = len(mass)
        self.n_particles = n
        self.n_big = 0

        if n == 0:
            self._particle_bufs = {}
            return

        if self.hsml_scale != 1.0:
            hsml = hsml * self.hsml_scale

        dev = self.device

        # Pack positions as vec4 (xyz + padding) for alignment
        pos4 = np.zeros((n, 4), dtype=np.float32)
        pos4[:, :3] = pos

        self._particle_bufs = {
            "pos": dev.create_buffer_with_data(data=pos4, usage=wgpu.BufferUsage.STORAGE),
            "hsml": dev.create_buffer_with_data(data=hsml.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
            "mass": dev.create_buffer_with_data(data=mass.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
            "qty": dev.create_buffer_with_data(data=qty.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
        }

        # Create bind group for particle data
        self._particle_bg = dev.create_bind_group(
            layout=self._splat_bgl1,
            entries=[
                {"binding": 0, "resource": {"buffer": self._particle_bufs["pos"]}},
                {"binding": 1, "resource": {"buffer": self._particle_bufs["hsml"]}},
                {"binding": 2, "resource": {"buffer": self._particle_bufs["mass"]}},
                {"binding": 3, "resource": {"buffer": self._particle_bufs["qty"]}},
            ],
        )

        # Identity sort index (no sorting)
        self._set_identity_sort_index(n)

    def _set_identity_sort_index(self, n):
        """Create identity sort index buffer (no sorting)."""
        if n == 0:
            n = 1  # wgpu requires non-zero buffer size
        identity = np.arange(n, dtype=np.uint32)
        self._sort_index_buf = self.device.create_buffer_with_data(
            data=identity, usage=wgpu.BufferUsage.STORAGE)
        self._sort_bg = self.device.create_bind_group(
            layout=self._splat_bgl2,
            entries=[{"binding": 0, "resource": {"buffer": self._sort_index_buf}}],
        )

    def set_sort_index_buffer(self, sort_index_buf):
        """Set an external sort index buffer (from GPUCompute.dispatch_sort)."""
        self._sort_index_buf = sort_index_buf
        self._sort_bg = self.device.create_bind_group(
            layout=self._splat_bgl2,
            entries=[{"binding": 0, "resource": {"buffer": sort_index_buf}}],
        )

    def set_particle_buffers_from_gpu(self, gpu_bufs, n_particles):
        """Use pre-existing GPU storage buffers from GPUCompute (zero-copy)."""
        self.n_particles = n_particles
        self.n_big = 0
        self._particle_bufs = gpu_bufs

        self._particle_bg = self.device.create_bind_group(
            layout=self._splat_bgl1,
            entries=[
                {"binding": 0, "resource": {"buffer": gpu_bufs["pos"]}},
                {"binding": 1, "resource": {"buffer": gpu_bufs["hsml"]}},
                {"binding": 2, "resource": {"buffer": gpu_bufs["mass"]}},
                {"binding": 3, "resource": {"buffer": gpu_bufs["qty"]}},
            ],
        )

        self._set_identity_sort_index(n_particles)

    def _upload_aniso_summaries(self, pos, mass, qty, cov):
        """Upload anisotropic summary splat data."""
        n = len(mass)
        self.n_aniso = n

        if n == 0:
            self._aniso_bufs = {}
            return

        dev = self.device

        # Pack pos as vec4
        pos4 = np.zeros((n, 4), dtype=np.float32)
        pos4[:, :3] = pos

        # Pack cov (n, 6) as 2n vec4s: [xx,xy,xz,0], [yy,yz,zz,0]
        cov4 = np.zeros((n * 2, 4), dtype=np.float32)
        cov4[0::2, :3] = cov[:, :3]  # xx, xy, xz
        cov4[1::2, :3] = cov[:, 3:]  # yy, yz, zz

        self._aniso_bufs = {
            "pos": dev.create_buffer_with_data(data=pos4, usage=wgpu.BufferUsage.STORAGE),
            "mass": dev.create_buffer_with_data(data=mass.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
            "qty": dev.create_buffer_with_data(data=qty.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
            "cov": dev.create_buffer_with_data(data=cov4, usage=wgpu.BufferUsage.STORAGE),
        }

        self._aniso_bg1 = dev.create_bind_group(
            layout=self._aniso_bgl1,
            entries=[
                {"binding": 0, "resource": {"buffer": self._aniso_bufs["pos"]}},
                {"binding": 1, "resource": {"buffer": self._aniso_bufs["mass"]}},
                {"binding": 2, "resource": {"buffer": self._aniso_bufs["qty"]}},
                {"binding": 3, "resource": {"buffer": self._aniso_bufs["cov"]}},
            ],
        )

    def upload_stars(self, positions, masses):
        """Upload star particle data."""
        self.n_stars = len(masses)
        if self.n_stars == 0:
            return
        dev = self.device
        pos4 = np.zeros((self.n_stars, 4), dtype=np.float32)
        pos4[:, :3] = positions.astype(np.float32)
        self._star_bufs = {
            "pos": dev.create_buffer_with_data(data=pos4, usage=wgpu.BufferUsage.STORAGE),
            "mass": dev.create_buffer_with_data(
                data=masses.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
        }
        self._star_bg1 = dev.create_bind_group(
            layout=self._star_bgl1,
            entries=[
                {"binding": 0, "resource": {"buffer": self._star_bufs["pos"]}},
                {"binding": 1, "resource": {"buffer": self._star_bufs["mass"]}},
            ],
        )

    # ---- Accumulation textures ----

    def _ensure_fbo(self, width, height, which=1):
        """Create or resize accumulation textures."""
        if which == 1:
            if self._accum_size == (width, height) and self._accum_textures is not None:
                return
            self._accum_textures = self._create_accum_textures(width, height)
            self._accum_size = (width, height)
        else:
            if self._accum_size2 == (width, height) and self._accum_textures2 is not None:
                return
            self._accum_textures2 = self._create_accum_textures(width, height)
            self._accum_size2 = (width, height)

    def _create_accum_textures(self, width, height):
        """Create a triple of accumulation textures (num, den, sq)."""
        dev = self.device
        textures = []
        views = []
        for _ in range(3):
            tex = dev.create_texture(
                size=(width, height, 1),
                format=self._accum_format,
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT |
                       wgpu.TextureUsage.TEXTURE_BINDING |
                       wgpu.TextureUsage.COPY_SRC),
            )
            textures.append(tex)
            views.append(tex.create_view())
        return {"textures": textures, "views": views}

    # ---- Camera uniforms ----

    def _write_camera_uniforms(self, camera, width, height):
        """Write camera data to the uniform buffer."""
        # view and proj are column-major in WGSL (same as OpenGL)
        view = np.ascontiguousarray(camera.view_matrix().T, dtype=np.float32)
        proj = np.ascontiguousarray(camera.projection_matrix().T, dtype=np.float32)

        kernel_id = self.KERNELS.index(self.kernel) if self.kernel in self.KERNELS else 0

        data = np.zeros(36, dtype=np.float32)  # 144 bytes / 4
        data[0:16] = view.ravel()
        data[16:32] = proj.ravel()
        data[32] = float(width)
        data[33] = float(height)
        # kernel_id and pad as uint32
        data_bytes = bytearray(data.tobytes())
        # Write kernel_id at offset 136 (byte 34*4=136)
        import struct
        struct.pack_into("I", data_bytes, 136, kernel_id)
        struct.pack_into("I", data_bytes, 140, 0)  # padding

        self.device.queue.write_buffer(self._camera_buf, 0, data_bytes)

    # ---- Render passes ----

    def _render_accum(self, camera, width, height, accum_textures):
        """Render additive accumulation pass into given textures."""
        self._write_camera_uniforms(camera, width, height)

        dev = self.device
        encoder = dev.create_command_encoder()

        views = accum_textures["views"]
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {"view": views[0], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                {"view": views[1], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                {"view": views[2], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
            ],
        )

        # Draw regular particles (all as instanced quads)
        if self.n_particles > 0 and self._particle_bufs and hasattr(self, '_sort_bg'):
            render_pass.set_pipeline(self._splat_pipeline)
            render_pass.set_bind_group(0, self._camera_bg)
            render_pass.set_bind_group(1, self._particle_bg)
            render_pass.set_bind_group(2, self._sort_bg)
            render_pass.draw(4, self.n_particles, 0, 0)

        # Draw anisotropic summary splats
        if self.n_aniso > 0 and self._aniso_bufs:
            # Update aniso params
            aniso_data = np.array([self.summary_scale, 0, 0, 0], dtype=np.float32)
            dev.queue.write_buffer(self._aniso_params_buf, 0, aniso_data.tobytes())

            render_pass.set_pipeline(self._aniso_pipeline)
            render_pass.set_bind_group(0, self._aniso_bg0)
            render_pass.set_bind_group(1, self._aniso_bg1)
            render_pass.draw(4, self.n_aniso, 0, 0)

        render_pass.end()
        dev.queue.submit([encoder.finish()])

    def render(self, camera, width, height):
        """Render particle splats via additive accumulation + resolve."""
        self._viewport_width = width
        if (self.n_particles == 0 and self.n_big == 0 and self.n_aniso == 0) or self._colormap_tex is None:
            return

        self._ensure_fbo(width, height, which=1)
        self._render_accum(camera, width, height, self._accum_textures)

        # Resolve to screen
        if self.canvas_context is None:
            return

        current_tex = self.canvas_context.get_current_texture()
        screen_view = current_tex.create_view()

        # Write resolve params
        import struct
        resolve_data = struct.pack("ffII",
                                   self.qty_min, self.qty_max,
                                   self.resolve_mode, self.log_scale)
        self.device.queue.write_buffer(self._resolve_params_buf, 0, resolve_data)

        # Create resolve bind group (references accum textures)
        resolve_bg = self.device.create_bind_group(
            layout=self._resolve_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self._resolve_params_buf}},
                {"binding": 1, "resource": self._accum_textures["views"][0]},
                {"binding": 2, "resource": self._accum_textures["views"][1]},
                {"binding": 3, "resource": self._accum_textures["views"][2]},
                {"binding": 4, "resource": self._colormap_tex.create_view()},
                {"binding": 5, "resource": self._colormap_sampler},
            ],
        )

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[{
                "view": screen_view,
                "clear_value": (0, 0, 0, 1),
                "load_op": "clear",
                "store_op": "store",
            }],
        )
        render_pass.set_pipeline(self._resolve_pipeline)
        render_pass.set_bind_group(0, resolve_bg)
        render_pass.draw(3, 1, 0, 0)  # fullscreen triangle
        render_pass.end()

        # Star overlay
        if self.n_stars > 0 and self._star_bufs:
            star_data = np.array([50.0, 0, 0, 0], dtype=np.float32)
            self.device.queue.write_buffer(self._star_params_buf, 0, star_data.tobytes())

            render_pass2 = encoder.begin_render_pass(
                color_attachments=[{
                    "view": screen_view,
                    "load_op": "load",
                    "store_op": "store",
                }],
            )
            render_pass2.set_pipeline(self._star_pipeline)
            render_pass2.set_bind_group(0, self._star_bg0)
            render_pass2.set_bind_group(1, self._star_bg1)
            render_pass2.draw(4, self.n_stars, 0, 0)
            render_pass2.end()

        self.device.queue.submit([encoder.finish()])

    def render_composite(self, camera, width, height,
                         mode1, min1, max1, log1,
                         mode2, min2, max2, log2):
        """Composite two pre-filled FBOs."""
        self._viewport_width = width
        self._ensure_fbo(width, height, which=1)
        self._ensure_fbo(width, height, which=2)

        if self.canvas_context is None:
            return

        current_tex = self.canvas_context.get_current_texture()
        screen_view = current_tex.create_view()

        import struct
        comp_data = struct.pack("ffIIffII",
                                min1, max1, mode1, log1,
                                min2, max2, mode2, log2)
        self.device.queue.write_buffer(self._composite_params_buf, 0, comp_data)

        composite_bg = self.device.create_bind_group(
            layout=self._composite_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self._composite_params_buf}},
                {"binding": 1, "resource": self._accum_textures["views"][0]},
                {"binding": 2, "resource": self._accum_textures["views"][1]},
                {"binding": 3, "resource": self._accum_textures["views"][2]},
                {"binding": 4, "resource": self._accum_textures2["views"][0]},
                {"binding": 5, "resource": self._accum_textures2["views"][1]},
                {"binding": 6, "resource": self._accum_textures2["views"][2]},
                {"binding": 7, "resource": self._colormap_tex.create_view()},
                {"binding": 8, "resource": self._colormap_sampler},
            ],
        )

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[{
                "view": screen_view,
                "clear_value": (0, 0, 0, 1),
                "load_op": "clear",
                "store_op": "store",
            }],
        )
        render_pass.set_pipeline(self._composite_pipeline)
        render_pass.set_bind_group(0, composite_bg)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()

        # Star overlay
        if self.n_stars > 0 and self._star_bufs:
            star_data = np.array([50.0, 0, 0, 0], dtype=np.float32)
            self.device.queue.write_buffer(self._star_params_buf, 0, star_data.tobytes())

            render_pass2 = encoder.begin_render_pass(
                color_attachments=[{
                    "view": screen_view,
                    "load_op": "load",
                    "store_op": "store",
                }],
            )
            render_pass2.set_pipeline(self._star_pipeline)
            render_pass2.set_bind_group(0, self._star_bg0)
            render_pass2.set_bind_group(1, self._star_bg1)
            render_pass2.draw(4, self.n_stars, 0, 0)
            render_pass2.end()

        self.device.queue.submit([encoder.finish()])

    def _read_accum_texture_r(self, texture):
        """Read back an accumulation texture and return the red channel as float32 array."""
        w, h = self._accum_size
        fmt = self._accum_format

        if fmt == "r32float":
            bpp = 4
        elif fmt == "rgba16float":
            bpp = 8  # 4 channels * 2 bytes
        else:
            bpp = 4

        data = self.device.queue.read_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": w * bpp},
            (w, h, 1),
        )

        if fmt == "r32float":
            return np.frombuffer(data, dtype=np.float32)
        elif fmt == "rgba16float":
            # Interpret as float16, take every 4th element (R channel)
            all_channels = np.frombuffer(data, dtype=np.float16)
            return all_channels[0::4].astype(np.float32)
        else:
            return np.frombuffer(data, dtype=np.float32)

    def read_accum_range(self):
        """Read back accumulation textures and compute percentile range."""
        if self._accum_textures is None:
            return self.qty_min, self.qty_max

        w, h = self._accum_size
        if w == 0 or h == 0:
            return self.qty_min, self.qty_max

        den = self._read_accum_texture_r(self._accum_textures["textures"][1])

        if self.resolve_mode == 2:
            num = self._read_accum_texture_r(self._accum_textures["textures"][0])
            sq = self._read_accum_texture_r(self._accum_textures["textures"][2])
            mask = den > 1e-30
            with np.errstate(invalid="ignore"):
                mean = np.where(mask, num / den, 0)
                mean_sq = np.where(mask, sq / den, 0)
            vals = np.sqrt(np.maximum(mean_sq - mean * mean, 0))[mask]
        elif self.resolve_mode == 1:
            num = self._read_accum_texture_r(self._accum_textures["textures"][0])
            mask = den > 1e-30
            with np.errstate(invalid="ignore"):
                vals = np.where(mask, num / den, 0)
            vals = vals[mask]
        else:
            vals = den[den > 1e-30]

        if len(vals) == 0:
            return self.qty_min, self.qty_max

        has_negative = (vals < 0).any()

        if has_negative:
            n = len(vals)
            k_lo = max(n // 100, 0)
            k_hi = min(99 * n // 100, n - 1)
            partitioned = np.partition(vals, (k_lo, k_hi))
            lim_lo = float(partitioned[k_lo])
            lim_hi = float(partitioned[k_hi])
        else:
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
            if self.log_scale and has_negative:
                self.log_scale = 0
            lo, hi = lim_lo, lim_hi
            if hi - lo < 1e-30:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1

        return lo, hi

    def release(self):
        """Clean up GPU resources."""
        # wgpu resources are garbage-collected, but we clear references
        self._particle_bufs = {}
        self._aniso_bufs = {}
        self._star_bufs = {}
        self._accum_textures = None
        self._accum_textures2 = None
