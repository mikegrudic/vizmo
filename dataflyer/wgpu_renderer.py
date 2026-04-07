"""wgpu-based renderer: additive accumulation + resolve, matching SplatRenderer's interface."""

import time
import numpy as np
from pathlib import Path

import wgpu

SHADER_DIR = Path(__file__).parent / "shaders"


_COMMON_WGSL = (SHADER_DIR / "common.wgsl").read_text()


def _load_wgsl(name, include_common=False):
    src = (SHADER_DIR / name).read_text()
    if include_common:
        src = _COMMON_WGSL + "\n" + src
    return src


def _make_bind_group(dev, layout, buffers):
    """Create bind group from layout + ordered list of buffers."""
    return dev.create_bind_group(
        layout=layout,
        entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(buffers)],
    )


def _storage_bgl(dev, n_buffers, visibility):
    """Create bind group layout with N read-only-storage buffer entries."""
    return dev.create_bind_group_layout(entries=[
        {"binding": i, "visibility": visibility, "buffer": {"type": "read-only-storage"}}
        for i in range(n_buffers)
    ])


def _make_render_pipeline(dev, layout, shader, targets):
    """Create render pipeline for instanced quads (triangle-strip, no vertex buffers)."""
    return dev.create_render_pipeline(
        layout=layout,
        vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
        primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
        fragment={"module": shader, "entry_point": "fs_main", "targets": targets},
    )


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

        # GPU timestamp queries (for measuring render pass duration).
        # Disabled by default: synchronous read_buffer after submit stalls
        # the CPU on the GPU queue, making the measurement reflect queue
        # back-pressure rather than actual GPU work.
        self._timestamp_supported = False  # set True manually to enable profiling
        self._last_render_ms = 0.0
        if self._timestamp_supported and "timestamp-query" in features:
            self._ts_query_set = device.create_query_set(type="timestamp", count=2)
            self._ts_resolve_buf = device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
        else:
            self._timestamp_supported = False

        # Render state (matches SplatRenderer attributes)
        self.n_particles = 0
        self.n_total = 0
        self.n_big = 0
        self.n_aniso = 0
        self.n_summaries = 0  # tracks summaries regardless of iso/aniso path
        self.n_stars = 0
        self.alpha_scale = 1.0
        self.qty_min = -1.0
        self.qty_max = 3.0
        self.resolve_mode = 0
        self.lod_pixels = 16
        self.log_scale = 1
        self.max_render_particles = 64_000_000
        self.use_tree = True
        self.use_adaptive_tree = True
        self.tree_min_particles = 0
        self.tree_n_cells = 64  # legacy, unused with adaptive octree
        self.tree_leaf_size = 64
        # LOD strategy: how the LOD knob translates into rendering decisions.
        #   "geometric"      — summarize cells with h_pix <= lod_pixels (default)
        #   "particle_count" — summarize cells with npart <= lod_pixels
        #   "subsample"      — bypass tree LOD, use stride sampling on leaves
        self.lod_strategy = "subsample"
        self.LOD_STRATEGIES = ["geometric", "particle_count", "subsample"]
        self.use_importance_sampling = False
        self.kernel = "cubic_spline"
        self.use_hybrid_rendering = True
        self.use_quad_rendering = False
        self.hsml_scale = 1.0
        self.summary_scale = 1.0
        self.summary_overlap = 0.1
        self.use_aniso_summaries = False
        self.bypass_cull = False
        self.auto_lod = True
        self.target_fps = 15.0
        self.auto_lod_smooth = 0.3
        self.pid_Kp = 0.5
        self.pid_Kd = 0.0
        self.pid_Ki = 0.0
        self.skip_vsync = False  # skip alternate presents to reduce vsync overhead
        self.cull_interval = 0.033  # cull every frame at 30fps (GPU compute is fast enough)
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
        self._accum_textures_lo = None  # half-res for summary splats
        self._accum_size = (0, 0)
        self._accum_size2 = (0, 0)
        self._accum_size_lo = (0, 0)

        # Particle storage buffers (SoA)
        self._particle_bufs = {}  # "pos", "hsml", "mass", "qty"
        self._summary_bufs = {}  # parallel set for summary splats (half-res target)
        self._aniso_bufs = {}  # "pos", "mass", "qty", "cov"
        self._star_bufs = {}  # "pos", "mass"
        self.n_summary_splats = 0  # count of summary splats in _summary_bufs

        # Colormap
        self._colormap_tex = None
        self._colormap_sampler = None
        self.colormap_tex = None  # placeholder for compatibility

        # Compile shaders and create pipeline layouts
        self._init_pipelines()

    def _init_pipelines(self):
        """Compile all shader modules and create pipeline layouts."""
        dev = self.device

        # Shader modules (render shaders include common.wgsl for Camera, quad_corner, eval_kernel)
        self._splat_shader = dev.create_shader_module(code=_load_wgsl("splat_quad.wgsl", include_common=True))
        self._splat_subsample_shader = dev.create_shader_module(
            code=_load_wgsl("splat_subsample.wgsl", include_common=True))
        self._aniso_shader = dev.create_shader_module(code=_load_wgsl("splat_aniso.wgsl", include_common=True))
        self._resolve_shader = dev.create_shader_module(code=_load_wgsl("resolve.wgsl"))
        self._composite_shader = dev.create_shader_module(code=_load_wgsl("composite.wgsl"))
        self._star_shader = dev.create_shader_module(code=_load_wgsl("star.wgsl", include_common=True))

        # Camera uniform buffer (shared by splat and star shaders)
        # struct Camera { view: mat4x4, proj: mat4x4, viewport_size: vec2, kernel_id: u32, _pad: u32 }
        # = 64 + 64 + 8 + 4 + 4 = 144 bytes
        self._camera_buf = dev.create_buffer(size=144, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # Aniso params uniform buffer
        # struct AnisoParams { cov_scale: f32, _pad: vec3<f32> } = 16 bytes
        self._aniso_params_buf = dev.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # Star params uniform buffer
        # struct StarParams { point_size: f32, _pad: vec3<f32> } = 16 bytes
        self._star_params_buf = dev.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

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
        VF = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        V = wgpu.ShaderStage.VERTEX

        # Uniform-only layouts
        self._splat_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}}])
        # Subsample splat: camera + cull params (both vertex+fragment so the
        # fragment side can read camera.kernel_id).
        self._splat_subsample_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": V, "buffer": {"type": "uniform"}}])
        self._aniso_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": V, "buffer": {"type": "uniform"}}])
        self._star_bgl0 = self._aniso_bgl0  # same layout: camera + params

        # Storage layouts
        self._splat_bgl1 = _storage_bgl(dev, 4, V)   # pos, hsml, mass, qty
        self._aniso_bgl1 = _storage_bgl(dev, 4, V)   # pos, mass, qty, cov
        self._star_bgl1 = _storage_bgl(dev, 2, V)    # pos, mass
        self._splat_bgl2 = _storage_bgl(dev, 1, V)   # sort index

        # Pipeline layouts
        self._splat_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._splat_bgl0, self._splat_bgl1, self._splat_bgl2])
        # Subsample splat shares splat_bgl1 (pos/hsml/mass/qty) but uses
        # splat_subsample_bgl0 (camera + cull params) and no sort group.
        self._splat_subsample_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._splat_subsample_bgl0, self._splat_bgl1])
        self._aniso_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._aniso_bgl0, self._aniso_bgl1])
        self._star_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._star_bgl0, self._star_bgl1])

        # Create render pipelines
        accum_targets = [{"format": self._accum_format, "blend": _additive_blend()}] * 3
        self._splat_pipeline = _make_render_pipeline(
            dev, self._splat_layout, self._splat_shader, accum_targets)
        self._splat_subsample_pipeline = _make_render_pipeline(
            dev, self._splat_subsample_layout, self._splat_subsample_shader,
            accum_targets)
        self._aniso_pipeline = _make_render_pipeline(
            dev, self._aniso_layout, self._aniso_shader, accum_targets)
        self._star_pipeline = _make_render_pipeline(
            dev, self._star_layout, self._star_shader,
            [{"format": self.present_format, "blend": _alpha_blend()}])

        # Static bind groups
        self._camera_bg = _make_bind_group(dev, self._splat_bgl0, [self._camera_buf])
        # Subsample chunks: list of dicts {bg0, bg1, params_buf, n, start}
        # for the splat_subsample pipeline. Set via set_subsample_chunks()
        # once after upload; None means "use the legacy direct path".
        self._subsample_chunks = None
        self._subsample_stride = 1
        # Per-frame instance cap. wgpu_app raises it gradually after the
        # camera stops (refinement) and resets it on motion.
        self._subsample_max_per_frame = 4_000_000
        self._aniso_bg0 = _make_bind_group(dev, self._aniso_bgl0,
                                           [self._camera_buf, self._aniso_params_buf])
        self._star_bg0 = _make_bind_group(dev, self._star_bgl0,
                                          [self._camera_buf, self._star_params_buf])

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
            mag_filter="linear",
            min_filter="linear",
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

        self._resolve_bgl = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
                {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
                # Half-res summary accumulation textures
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 7,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 8,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
            ]
        )

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

        self._composite_bgl = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {"binding": 7, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
                {"binding": 8, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
            ]
        )

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
        from .adaptive_octree import AdaptiveOctree

        self._all_pos = positions.astype(np.float32)
        self._all_hsml = hsml.astype(np.float32)
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)
        self.n_total = len(masses)

        # Set default LOD from particle count: n^(1/3) / 8
        self.lod_pixels = max(1.0, self.n_total ** (1.0 / 3.0) / 16.0)

        if self.use_tree and self.n_total > self.tree_min_particles:
            self._grid = self._build_grid()
        else:
            self._grid = None

    def _build_grid(self):
        """Build spatial tree for frustum culling. Respects use_adaptive_tree flag."""
        t0 = time.perf_counter()
        if self.use_adaptive_tree:
            from .adaptive_octree import AdaptiveOctree
            # Subsample mode only needs the sorted particle arrays, not the
            # octree itself. Defer the expensive subdivision + moments build
            # until the user switches to a tree-using strategy.
            defer = (self.lod_strategy == "subsample")
            grid = AdaptiveOctree(
                self._all_pos, self._all_mass, self._all_hsml, self._all_qty,
                leaf_size=self.tree_leaf_size, defer_tree_build=defer)
            label = "Adaptive octree" + (" (subsample-only init)" if defer else "")
        else:
            from .spatial_grid import SpatialGrid
            grid = SpatialGrid(
                self._all_pos, self._all_mass, self._all_hsml, self._all_qty,
                n_cells=self.tree_n_cells)
            label = "Uniform grid"
        print(f"  {label} built in {time.perf_counter()-t0:.1f}s")
        return grid

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
            self._upload_summary_arrays(
                np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.float32))
            self.n_summaries = 0
            self._last_upload_ms = (time.perf_counter() - t0) * 1000
            return

        if self._needs_grid_rebuild:
            self._needs_grid_rebuild = False
            if self.use_tree and self.n_total > self.tree_min_particles:
                self._grid = self._build_grid()
            else:
                self._grid = None

        if self._grid is not None:
            t0 = time.perf_counter()
            result = self._grid.query_frustum_lod(
                camera,
                self.max_render_particles,
                lod_pixels=self.lod_pixels,
                importance_sampling=self.use_importance_sampling,
                viewport_width=self._viewport_width,
                summary_overlap=self.summary_overlap,
                anisotropic=self.use_aniso_summaries,
                summary_scale=self.summary_scale,
                lod_strategy=self.lod_strategy,
            )
            self._last_cull_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            if len(result) == 9:
                r_pos, r_hsml, r_mass, r_qty, s_pos, s_hsml, s_mass, s_qty, s_cov = result
                self.n_summaries = len(s_pos)
                if self.use_aniso_summaries:
                    self._upload_arrays(r_pos, r_hsml, r_mass, r_qty, camera)
                    self._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
                    self._upload_summary_arrays(
                        np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                        np.zeros(0, np.float32), np.zeros(0, np.float32))
                else:
                    # Isotropic mode: keep summaries in a separate buffer
                    # so they can be rendered to the half-res FBO.
                    self._upload_arrays(r_pos, r_hsml, r_mass, r_qty, camera)
                    self._upload_summary_arrays(s_pos, s_hsml, s_mass, s_qty)
                    self._upload_aniso_summaries(
                        np.zeros((0, 3), np.float32),
                        np.zeros(0, np.float32),
                        np.zeros(0, np.float32),
                        np.zeros((0, 6), np.float32),
                    )
            else:
                pos, hsml, mass, qty = result
                self.n_summaries = 0
                self._upload_arrays(pos, hsml, mass, qty, camera)
                self._upload_summary_arrays(
                    np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                    np.zeros(0, np.float32), np.zeros(0, np.float32))
                self._upload_aniso_summaries(
                    np.zeros((0, 3), np.float32),
                    np.zeros(0, np.float32),
                    np.zeros(0, np.float32),
                    np.zeros((0, 6), np.float32),
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
            self._upload_summary_arrays(
                np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.float32))
            self.n_summaries = 0
        self._last_upload_ms = (time.perf_counter() - t0) * 1000

    def _upload_summary_arrays(self, pos, hsml, mass, qty):
        """Upload summary splat arrays to a separate set of buffers
        (rendered to the half-resolution accumulation textures)."""
        n = len(mass)
        self.n_summary_splats = n
        if n == 0:
            self._summary_bufs = {}
            return

        if self.hsml_scale != 1.0:
            hsml = hsml * self.hsml_scale

        dev = self.device
        pos4 = np.zeros((n, 4), dtype=np.float32)
        pos4[:, :3] = pos
        self._summary_bufs = {
            "pos": dev.create_buffer_with_data(data=pos4, usage=wgpu.BufferUsage.STORAGE),
            "hsml": dev.create_buffer_with_data(data=hsml.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
            "mass": dev.create_buffer_with_data(data=mass.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
            "qty": dev.create_buffer_with_data(data=qty.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
        }
        sb = self._summary_bufs
        self._summary_bg = _make_bind_group(
            dev, self._splat_bgl1,
            [sb["pos"], sb["hsml"], sb["mass"], sb["qty"]])
        # Reuse identity sort buffer (grow if needed for summary count)
        self._set_identity_sort_index(n)

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
        pb = self._particle_bufs
        self._particle_bg = _make_bind_group(dev, self._splat_bgl1,
                                             [pb["pos"], pb["hsml"], pb["mass"], pb["qty"]])

        # Identity sort index (no sorting)
        self._set_identity_sort_index(n)

    def _set_identity_sort_index(self, n):
        """Create identity sort index buffer (no sorting). Cached by size."""
        if n == 0:
            n = 1
        if hasattr(self, "_identity_sort_n") and self._identity_sort_n >= n:
            return  # reuse existing buffer (large enough)
        identity = np.arange(n, dtype=np.uint32)
        self._sort_index_buf = self.device.create_buffer_with_data(data=identity, usage=wgpu.BufferUsage.STORAGE)
        self._sort_bg = _make_bind_group(self.device, self._splat_bgl2, [self._sort_index_buf])
        self._identity_sort_n = n

    def set_sort_index_buffer(self, sort_index_buf):
        """Set an external sort index buffer (from GPUCompute.dispatch_sort)."""
        self._sort_index_buf = sort_index_buf
        self._sort_bg = _make_bind_group(self.device, self._splat_bgl2, [sort_index_buf])

    def set_particle_buffers_from_gpu(self, gpu_bufs, n_particles):
        """Use pre-existing GPU storage buffers from GPUCompute (zero-copy)."""
        self.n_particles = n_particles
        self.n_big = 0
        self._particle_bufs = gpu_bufs

        self._particle_bg = _make_bind_group(self.device, self._splat_bgl1,
                                             [gpu_bufs["pos"], gpu_bufs["hsml"],
                                              gpu_bufs["mass"], gpu_bufs["qty"]])

        self._set_identity_sort_index(n_particles)

    def set_subsample_chunks(self, chunks, world_offset=None):
        """Configure compute-driven splat: render directly from per-chunk
        source particle buffers, with hash-stride + frustum cull happening
        in the vertex shader.

        Args:
            chunks: list of dicts {"pos","hsml","mass","qty","n","start"}
                from GPUCompute, or None to disable.
            world_offset: (3,) float32. Subtracted from camera position
                before building view matrix and cull params, to match the
                world-origin shift applied to particle positions on
                upload (avoids float32 precision loss for cosmological
                snapshots).
        """
        if chunks is None:
            self._subsample_chunks = None
            self._world_offset = None
            self._slot_subsample_bgs = [None, None]
            self._active_subsample_slot = None
            return
        self._world_offset = (
            np.asarray(world_offset, dtype=np.float32)
            if world_offset is not None else None)
        # Reset slot bindings whenever the base chunks change.
        self._slot_subsample_bgs = [None, None]
        self._active_subsample_slot = None
        dev = self.device
        out = []
        for cb in chunks:
            params_buf = dev.create_buffer(
                size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            bg0 = _make_bind_group(dev, self._splat_subsample_bgl0,
                                   [self._camera_buf, params_buf])
            bg1 = _make_bind_group(dev, self._splat_bgl1,
                                   [cb["pos"], cb["hsml"], cb["mass"], cb["qty"]])
            out.append({
                "bg0": bg0, "bg1": bg1, "params_buf": params_buf,
                "pos": cb["pos"], "hsml": cb["hsml"],
                "n": int(cb["n"]), "start": int(cb["start"]),
            })
        self._subsample_chunks = out

    def set_subsample_stride(self, stride):
        """Set the stride used by the compute-driven splat path."""
        self._subsample_stride = max(int(stride), 1)

    def set_subsample_slot_chunks(self, slot_idx, slot_chunks):
        """Bind a composite slot's mass/qty per-chunk buffers. Pos+hsml
        come from the shared self._subsample_chunks; only mass+qty are
        per-slot. Builds a parallel set of bg1 bind groups; the active
        slot is selected via set_active_subsample_slot().
        """
        if not hasattr(self, "_slot_subsample_bgs"):
            self._slot_subsample_bgs = [None, None]
        bgs = []
        for ck, sc in zip(self._subsample_chunks, slot_chunks):
            bg = _make_bind_group(
                self.device, self._splat_bgl1,
                [ck["pos"], ck["hsml"], sc["mass"], sc["qty"]])
            bgs.append(bg)
        self._slot_subsample_bgs[slot_idx] = bgs

    def set_active_subsample_slot(self, slot_idx):
        """Pick which composite slot's mass/qty bind groups the next
        _render_accum will use. Pass None for the default (chunks' own
        mass/qty, used outside composite mode).
        """
        self._active_subsample_slot = slot_idx

    def set_subsample_max_per_frame(self, max_inst):
        """Set the per-frame instance cap for the compute-driven splat
        path. wgpu_app raises this gradually as refinement progresses
        and resets it on camera motion.
        """
        self._subsample_max_per_frame = max(int(max_inst), 1)

    def _write_subsample_params(self, camera, stride):
        """Write each chunk's params uniform. Must be called BEFORE
        beginning the render pass (write_buffer cannot run mid-pass).

        `stride` is the (possibly fractional) coarsening factor:
            stride = n_total / budget
        where budget is the total number of instances dispatched across
        all chunks. Each rendered particle stands in for `stride`
        particles' worth of mass.
        """
        if self._subsample_chunks is None:
            return
        import struct as _struct
        ratio = float(stride)
        h_scale = ratio ** (1.0 / 3.0)
        mass_scale = ratio  # each sampled particle stands in for `stride`
        fov_rad = float(np.radians(camera.fov))
        # Apply the same world-origin shift as the view matrix.
        offset = getattr(self, "_world_offset", None)
        if offset is not None:
            cam_pos = camera.position - offset
        else:
            cam_pos = camera.position
        cam_bytes = _struct.pack(
            "fff f fff f fff f fff f",
            float(cam_pos[0]), float(cam_pos[1]),
            float(cam_pos[2]), 0.0,
            float(camera.forward[0]), float(camera.forward[1]),
            float(camera.forward[2]), 0.0,
            float(camera.right[0]), float(camera.right[1]),
            float(camera.right[2]), 0.0,
            float(camera.up[0]), float(camera.up[1]),
            float(camera.up[2]), 0.0,
        )
        for ck in self._subsample_chunks:
            tail = _struct.pack(
                "ffII ffII",
                fov_rad, float(camera.aspect), int(stride), int(ck["n"]),
                h_scale, mass_scale, 0, 0,
            )
            self.device.queue.write_buffer(
                ck["params_buf"], 0, cam_bytes + tail)

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

        ab = self._aniso_bufs
        self._aniso_bg1 = _make_bind_group(dev, self._aniso_bgl1,
                                           [ab["pos"], ab["mass"], ab["qty"], ab["cov"]])

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
            "mass": dev.create_buffer_with_data(data=masses.astype(np.float32), usage=wgpu.BufferUsage.STORAGE),
        }
        self._star_bg1 = _make_bind_group(dev, self._star_bgl1,
                                          [self._star_bufs["pos"], self._star_bufs["mass"]])

    # ---- Accumulation textures ----

    def _ensure_fbo(self, width, height, which=1):
        """Create or resize accumulation textures."""
        if which == 1:
            if self._accum_size != (width, height) or self._accum_textures is None:
                self._accum_textures = self._create_accum_textures(width, height)
                self._accum_size = (width, height)
            # Half-resolution textures for summary splats
            lo_w = max(1, width // 2)
            lo_h = max(1, height // 2)
            if self._accum_size_lo != (lo_w, lo_h) or self._accum_textures_lo is None:
                self._accum_textures_lo = self._create_accum_textures(lo_w, lo_h)
                self._accum_size_lo = (lo_w, lo_h)
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
                usage=(
                    wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
                ),
            )
            textures.append(tex)
            views.append(tex.create_view())
        return {"textures": textures, "views": views}

    # ---- Camera uniforms ----

    def _write_camera_uniforms(self, camera, width, height):
        """Write camera data to the uniform buffer."""
        # view and proj are column-major in WGSL (same as OpenGL).
        # Apply the world-origin shift (subsample mode pre-translates
        # particles by self._world_offset to avoid float32 precision loss
        # in the view-matrix multiply on cosmological-scale snapshots).
        offset = getattr(self, "_world_offset", None)
        if offset is not None:
            saved = camera.position
            camera.position = saved - offset
            try:
                view = np.ascontiguousarray(camera.view_matrix().T, dtype=np.float32)
            finally:
                camera.position = saved
        else:
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

    def _encode_star_overlay(self, encoder, screen_view):
        """Append a star overlay render pass to the given encoder."""
        if self.n_stars == 0 or not self._star_bufs:
            return
        star_data = np.array([50.0, 0, 0, 0], dtype=np.float32)
        self.device.queue.write_buffer(self._star_params_buf, 0, star_data.tobytes())
        rp = encoder.begin_render_pass(
            color_attachments=[{"view": screen_view, "load_op": "load", "store_op": "store"}])
        rp.set_pipeline(self._star_pipeline)
        rp.set_bind_group(0, self._star_bg0)
        rp.set_bind_group(1, self._star_bg1)
        rp.draw(4, self.n_stars, 0, 0)
        rp.end()

    def _render_accum(self, camera, width, height, accum_textures):
        """Render additive accumulation pass into given textures.

        Renders:
          - real particles + anisotropic summaries → full-res `accum_textures`
          - isotropic summary splats → half-res `_accum_textures_lo`
        """
        self._write_camera_uniforms(camera, width, height)

        # Subsample params must be written before the render pass begins.
        if self._subsample_chunks is not None:
            self._write_subsample_params(camera, self._subsample_stride)

        dev = self.device
        encoder = dev.create_command_encoder()

        # ---- Full-resolution pass: particles + aniso summaries ----
        views = accum_textures["views"]
        rp_args = dict(
            color_attachments=[
                {"view": views[0], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                {"view": views[1], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                {"view": views[2], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
            ],
        )
        if self._timestamp_supported:
            rp_args["timestamp_writes"] = {
                "query_set": self._ts_query_set,
                "beginning_of_pass_write_index": 0,
                "end_of_pass_write_index": 1,
            }
        render_pass = encoder.begin_render_pass(**rp_args)

        # Draw regular particles (all as instanced quads)
        if (self._subsample_chunks is None and
                getattr(self, "lod_strategy", "geometric") == "subsample"):
            # In subsample mode the legacy direct draw would dispatch
            # n_particles instances (often 30M+) before the chunk
            # bind groups are set up. Skip drawing entirely until the
            # GPU init has wired up the per-chunk source buffers.
            pass
        elif self._subsample_chunks is not None:
            # Compute-driven splat: one draw per chunk, dispatching only
            # ceil(n / stride) instances. The vertex shader picks one
            # particle per instance via hash and frustum-tests it.
            slot = getattr(self, "_active_subsample_slot", None)
            slot_bgs = (self._slot_subsample_bgs[slot]
                        if slot is not None
                        and self._slot_subsample_bgs[slot] is not None
                        else None)
            #
            # Hard cap on per-frame instances: 32M point splats overwhelm
            # Apple Metal's vertex stage. Honor the LOD-controlled stride
            # but raise it further if the budget would still exceed the
            # cap. mass_scale must be raised to match so the visualization
            # stays unbiased.
            # Compute the global budget from the (possibly fractional)
            # stride and clamp it to the per-frame instance cap. The
            # effective stride is then n_total / budget — fractional in
            # general — and per-chunk dispatch counts are proportional
            # to chunk size so the per-chunk sample is representative.
            cap = self._subsample_max_per_frame
            stride = max(float(self._subsample_stride), 1.0)
            n_total = sum(ck["n"] for ck in self._subsample_chunks)
            budget = max(1, int(round(n_total / stride)))
            if budget > cap:
                budget = cap
            # Effective fractional stride after capping.
            eff_stride = max(n_total / max(budget, 1), 1.0)
            self._write_subsample_params(camera, eff_stride)
            render_pass.set_pipeline(self._splat_subsample_pipeline)
            for i, ck in enumerate(self._subsample_chunks):
                # Proportional share of the budget for this chunk.
                instances = max(1, int(round(budget * ck["n"] / n_total)))
                instances = min(instances, ck["n"])
                render_pass.set_bind_group(0, ck["bg0"])
                bg1 = slot_bgs[i] if slot_bgs is not None else ck["bg1"]
                render_pass.set_bind_group(1, bg1)
                render_pass.draw(4, instances, 0, 0)
        elif self.n_particles > 0 and self._particle_bufs and hasattr(self, "_sort_bg"):
            render_pass.set_pipeline(self._splat_pipeline)
            render_pass.set_bind_group(0, self._camera_bg)
            render_pass.set_bind_group(1, self._particle_bg)
            render_pass.set_bind_group(2, self._sort_bg)
            render_pass.draw(4, self.n_particles, 0, 0)

        # Draw anisotropic summary splats
        if self.n_aniso > 0 and self._aniso_bufs:
            aniso_data = np.array([self.summary_scale, 0, 0, 0], dtype=np.float32)
            dev.queue.write_buffer(self._aniso_params_buf, 0, aniso_data.tobytes())

            render_pass.set_pipeline(self._aniso_pipeline)
            render_pass.set_bind_group(0, self._aniso_bg0)
            render_pass.set_bind_group(1, self._aniso_bg1)
            render_pass.draw(4, self.n_aniso, 0, 0)

        render_pass.end()

        # ---- Half-resolution pass: isotropic summary splats ----
        # Write camera uniforms with the half-res viewport so the splat
        # quad sizes are correct in pixel space.
        if self._accum_textures_lo is not None:
            lo_views = self._accum_textures_lo["views"]
            lo_rp = encoder.begin_render_pass(color_attachments=[
                {"view": lo_views[0], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                {"view": lo_views[1], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                {"view": lo_views[2], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
            ])
            if self.n_summary_splats > 0 and self._summary_bufs and hasattr(self, "_summary_bg"):
                lo_rp.set_pipeline(self._splat_pipeline)
                lo_rp.set_bind_group(0, self._camera_bg)
                lo_rp.set_bind_group(1, self._summary_bg)
                lo_rp.set_bind_group(2, self._sort_bg)
                lo_rp.draw(4, self.n_summary_splats, 0, 0)
            lo_rp.end()

        if self._timestamp_supported:
            encoder.resolve_query_set(
                self._ts_query_set, 0, 2, self._ts_resolve_buf, 0)
        dev.queue.submit([encoder.finish()])

        if self._timestamp_supported:
            try:
                ts_data = dev.queue.read_buffer(self._ts_resolve_buf, size=16)
                t0, t1 = np.frombuffer(ts_data, dtype=np.uint64)
                if t1 > t0:
                    self._last_render_ms = float(t1 - t0) * 1e-6
            except Exception:
                pass

    def render(self, camera, width, height):
        """Render particle splats via additive accumulation + resolve."""
        self._viewport_width = width
        if (self.n_particles == 0 and self.n_big == 0 and self.n_aniso == 0
                and self.n_summary_splats == 0) or self._colormap_tex is None:
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

        resolve_data = struct.pack("ffII", self.qty_min, self.qty_max, self.resolve_mode, self.log_scale)
        self.device.queue.write_buffer(self._resolve_params_buf, 0, resolve_data)

        # Create resolve bind group (references full-res + half-res accum textures)
        lo_views = self._accum_textures_lo["views"]
        resolve_bg = self.device.create_bind_group(
            layout=self._resolve_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self._resolve_params_buf}},
                {"binding": 1, "resource": self._accum_textures["views"][0]},
                {"binding": 2, "resource": self._accum_textures["views"][1]},
                {"binding": 3, "resource": self._accum_textures["views"][2]},
                {"binding": 4, "resource": self._colormap_tex.create_view()},
                {"binding": 5, "resource": self._colormap_sampler},
                {"binding": 6, "resource": lo_views[0]},
                {"binding": 7, "resource": lo_views[1]},
                {"binding": 8, "resource": lo_views[2]},
            ],
        )

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": screen_view,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
        )
        render_pass.set_pipeline(self._resolve_pipeline)
        render_pass.set_bind_group(0, resolve_bg)
        render_pass.draw(3, 1, 0, 0)  # fullscreen triangle
        render_pass.end()
        self._encode_star_overlay(encoder, screen_view)
        self.device.queue.submit([encoder.finish()])

    def render_composite(self, camera, width, height, mode1, min1, max1, log1, mode2, min2, max2, log2):
        """Composite two pre-filled FBOs."""
        self._viewport_width = width
        self._ensure_fbo(width, height, which=1)
        self._ensure_fbo(width, height, which=2)

        if self.canvas_context is None:
            return

        current_tex = self.canvas_context.get_current_texture()
        screen_view = current_tex.create_view()

        import struct

        comp_data = struct.pack("ffIIffII", min1, max1, mode1, log1, min2, max2, mode2, log2)
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
            color_attachments=[
                {
                    "view": screen_view,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
        )
        render_pass.set_pipeline(self._composite_pipeline)
        render_pass.set_bind_group(0, composite_bg)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        self._encode_star_overlay(encoder, screen_view)
        self.device.queue.submit([encoder.finish()])

    def _read_accum_texture_r(self, texture, size=None):
        """Read back an accumulation texture's red channel as float32 array.

        Args:
            texture: GPU texture to read.
            size: optional (w, h) tuple. Defaults to `self._accum_size`.
        """
        w, h = size if size is not None else self._accum_size
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

    def read_accum_range(self, mass_weighted=True):
        """Read back accumulation textures and compute percentile range.

        Reads BOTH the full-resolution (particles) and half-resolution
        (summary splats) accumulation textures so the auto-range reflects
        the union of all rendered content.

        Args:
            mass_weighted: when True the entropy maximization weights bins
                by accumulated mass (used for surface-density / lightness
                slots so very dense regions dominate the level choice).
                When False the entropy is computed on raw value counts
                (better for color slots where we want the dynamic range
                of the field itself, not where the mass piles up).
        """
        if self._accum_textures is None:
            return self.qty_min, self.qty_max

        w, h = self._accum_size
        if w == 0 or h == 0:
            return self.qty_min, self.qty_max

        den_hi = self._read_accum_texture_r(self._accum_textures["textures"][1])
        den_lo = None
        if self._accum_textures_lo is not None:
            lo_w, lo_h = self._accum_size_lo
            if lo_w > 0 and lo_h > 0:
                den_lo = self._read_accum_texture_r(
                    self._accum_textures_lo["textures"][1], size=(lo_w, lo_h))

        if self.resolve_mode == 2:
            num_hi = self._read_accum_texture_r(self._accum_textures["textures"][0])
            sq_hi = self._read_accum_texture_r(self._accum_textures["textures"][2])
            mask_hi = den_hi > 1e-30
            with np.errstate(invalid="ignore"):
                mean = np.where(mask_hi, num_hi / den_hi, 0)
                mean_sq = np.where(mask_hi, sq_hi / den_hi, 0)
            vals_hi = np.sqrt(np.maximum(mean_sq - mean * mean, 0))[mask_hi]
            mass_hi = den_hi[mask_hi]
            vals_parts = [vals_hi]
            mass_parts = [mass_hi]
            if den_lo is not None:
                num_lo = self._read_accum_texture_r(
                    self._accum_textures_lo["textures"][0], size=self._accum_size_lo)
                sq_lo = self._read_accum_texture_r(
                    self._accum_textures_lo["textures"][2], size=self._accum_size_lo)
                mask_lo = den_lo > 1e-30
                with np.errstate(invalid="ignore"):
                    mean_l = np.where(mask_lo, num_lo / den_lo, 0)
                    mean_sq_l = np.where(mask_lo, sq_lo / den_lo, 0)
                vals_lo = np.sqrt(np.maximum(mean_sq_l - mean_l * mean_l, 0))[mask_lo]
                vals_parts.append(vals_lo)
                mass_parts.append(den_lo[mask_lo])
            vals = np.concatenate(vals_parts) if vals_parts else np.zeros(0, np.float32)
            mass = np.concatenate(mass_parts) if mass_parts else np.zeros(0, np.float32)
        elif self.resolve_mode == 1:
            num_hi = self._read_accum_texture_r(self._accum_textures["textures"][0])
            mask_hi = den_hi > 1e-30
            with np.errstate(invalid="ignore"):
                v_hi = np.where(mask_hi, num_hi / den_hi, 0)[mask_hi]
            vals_parts = [v_hi]
            mass_parts = [den_hi[mask_hi]]
            if den_lo is not None:
                num_lo = self._read_accum_texture_r(
                    self._accum_textures_lo["textures"][0], size=self._accum_size_lo)
                mask_lo = den_lo > 1e-30
                with np.errstate(invalid="ignore"):
                    v_lo = np.where(mask_lo, num_lo / den_lo, 0)[mask_lo]
                vals_parts.append(v_lo)
                mass_parts.append(den_lo[mask_lo])
            vals = np.concatenate(vals_parts)
            mass = np.concatenate(mass_parts)
        else:
            # Surface density: combine the denominator values from both buffers
            mask_hi = den_hi > 1e-30
            vals_parts = [den_hi[mask_hi]]
            if den_lo is not None:
                mask_lo = den_lo > 1e-30
                vals_parts.append(den_lo[mask_lo])
            vals = np.concatenate(vals_parts)
            mass = vals

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
            from dataflyer.field_ops import max_entropy_limits
            entropy_weights = mass if mass_weighted else np.ones_like(vals)
            lim_lo, lim_hi = max_entropy_limits(vals, entropy_weights)

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
