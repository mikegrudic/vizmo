"""wgpu-based renderer: additive accumulation + resolve."""

import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import wgpu

SHADER_DIR = Path(__file__).parent / "shaders"


@dataclass
class RenderMode:
    """Defines how the two accumulation textures are combined in the
    resolve pass. Fragment shaders write
        out_numerator   = sigma * quantity
        out_denominator = sigma           (where sigma = mass * W(r) / h^2)
    The resolve shader displays the denominator (resolve_mode=0), the
    ratio numerator/denominator (resolve_mode=1), or the variance
    sqrt(<f^2> - <f>^2) (resolve_mode=2).
    """

    name: str
    weight_field: str
    qty_field: str
    resolve_mode: int

    @staticmethod
    def surface_density(weight_field="Masses"):
        return RenderMode(
            name=f"Sigma {weight_field}",
            weight_field=weight_field,
            qty_field=weight_field,
            resolve_mode=0,
        )

    @staticmethod
    def mass_weighted_average(qty_field, weight_field="Masses"):
        return RenderMode(
            name=f"<{qty_field}>",
            weight_field=weight_field,
            qty_field=qty_field,
            resolve_mode=1,
        )

    @staticmethod
    def weighted_variance(qty_field, weight_field="Masses"):
        return RenderMode(
            name=f"sigma({qty_field})",
            weight_field=weight_field,
            qty_field=qty_field,
            resolve_mode=2,
        )


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


# Accumulation texture format.
# r32float with additive blending requires "float32-blendable" feature (not on macOS Metal).
# rgba16float is blendable by default everywhere. Shaders output vec4 so both formats work.
ACCUM_FORMAT = "r32float"
ACCUM_FORMAT_FALLBACK = "rgba16float"


class WGPURenderer:
    """Splat renderer backed by wgpu."""

    KERNELS = ["cubic_spline", "wendland_c2", "gaussian", "quartic", "sphere"]

    def __init__(self, device, canvas_context=None, present_format=None):
        self.device = device
        self.canvas_context = canvas_context
        self.present_format = present_format or "bgra8unorm"

        features = device.features
        if "float32-blendable" in features:
            self._accum_format = ACCUM_FORMAT
        else:
            self._accum_format = ACCUM_FORMAT_FALLBACK
            print(f"  wgpu: float32-blendable not available, using {ACCUM_FORMAT_FALLBACK}")
        self._timestamp_supported = False
        self._last_render_ms = 0.0

        # HUD / UI state
        self.n_particles = 0
        self.n_total = 0
        self.n_stars = 0
        self.qty_min = -1.0
        self.qty_max = 3.0
        self.resolve_mode = 0
        self.log_scale = 1
        self.kernel = "cubic_spline"
        self.auto_lod = True
        self.target_fps = 15.0
        self.auto_lod_smooth = 0.3
        # PID gains for the auto-LOD controller (tunable from the dev panel).
        self.pid_Kp = 1.0
        self.pid_Ki = 0.0
        self.pid_Kd = 0.0
        self.skip_vsync = False
        self.hsml_scale = 1.0
        # Multigrid LOD: 1 = disabled (single-level, original behavior).
        # When > 1, particles are routed to a pyramid of accum textures
        # (level 0 = full res, level k = full_res / 2^k) by their kernel
        # screen radius, and a cascade upsample folds coarser levels back
        # into the finest before resolve.
        self.multigrid_levels = 6
        self.multigrid_n_grid_kernel = 4.0

        # Timing
        self._last_cull_ms = 0.0
        self._last_upload_ms = 0.0
        self._viewport_width = 1024

        # CPU-side particle data (for grid construction only)
        self._all_pos = None
        self._all_hsml = None
        self._all_mass = None
        self._all_qty = None

        # GPU resources
        self._accum_textures = None
        self._accum_textures2 = None  # second set for composite mode
        self._accum_size = (0, 0)
        self._accum_size2 = (0, 0)
        # Multigrid pyramid: list of accum-texture dicts, level 0 unused
        # (it aliases self._accum_textures); level k has res/2^k.
        # Rebuilt by _ensure_fbo when multigrid_levels or size changes.
        self._accum_pyramid = []
        self._accum_pyramid_levels = 0
        self._cascade_bgs = []  # bind groups, one per non-zero level
        # CPU-side star particle data, populated by upload_stars().
        self.n_stars = 0
        self._star_positions = None
        self._star_masses = None

        # Colormap
        self._colormap_tex = None
        self._colormap_tex_view = None
        self._colormap_sampler = None
        self.colormap_tex = None

        # Cached bind groups for resolve / composite passes. Invalidated
        # whenever the FBO triple is recreated or the colormap is reloaded.
        self._resolve_bg = None
        self._composite_bg = None

        self._init_pipelines()

    def _init_pipelines(self):
        """Compile all shader modules and create pipeline layouts."""
        dev = self.device

        # Shader modules (render shaders include common.wgsl for Camera, quad_corner, eval_kernel)
        self._splat_subsample_shader = dev.create_shader_module(
            code=_load_wgsl("splat_subsample.wgsl", include_common=True))
        self._resolve_shader = dev.create_shader_module(code=_load_wgsl("resolve.wgsl"))
        self._composite_shader = dev.create_shader_module(code=_load_wgsl("composite.wgsl"))
        self._cascade_shader = dev.create_shader_module(code=_load_wgsl("cascade.wgsl"))
        self._mg_bin_shader = dev.create_shader_module(
            code=_load_wgsl("multigrid_bin.wgsl"))

        # Camera uniform: view + proj + viewport + kernel_id + pad = 144 B
        self._camera_buf = dev.create_buffer(
            size=144, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        # Resolve params (qty_min, qty_max, mode, log_scale)
        self._resolve_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        # Composite params
        self._composite_params_buf = dev.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        VF = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        V = wgpu.ShaderStage.VERTEX

        # bg0: camera + subsample-cull params (both visible to vertex+fragment)
        self._splat_subsample_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": V, "buffer": {"type": "uniform"}}])
        # bg1: pos / hsml / mass / qty / index / bases per chunk
        self._splat_bgl1 = _storage_bgl(dev, 6, V)

        self._splat_subsample_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._splat_subsample_bgl0, self._splat_bgl1])

        accum_targets = [{"format": self._accum_format, "blend": _additive_blend()}] * 3
        self._splat_subsample_pipeline = _make_render_pipeline(
            dev, self._splat_subsample_layout, self._splat_subsample_shader,
            accum_targets)

        # Cascade pipeline: reads 3 coarse accum textures, additively
        # blends bilinear samples into the next-finer level.
        F = wgpu.ShaderStage.FRAGMENT
        self._cascade_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": F,
             "texture": {"sample_type": "unfilterable-float"
                         if self._accum_format == "r32float"
                         else "float"}},
            {"binding": 1, "visibility": F,
             "texture": {"sample_type": "unfilterable-float"
                         if self._accum_format == "r32float"
                         else "float"}},
            {"binding": 2, "visibility": F,
             "texture": {"sample_type": "unfilterable-float"
                         if self._accum_format == "r32float"
                         else "float"}},
        ])
        cascade_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._cascade_bgl])
        # Multigrid binning compute pipelines.
        C = wgpu.ShaderStage.COMPUTE
        self._mg_bin_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": C, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": C, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": C, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": C, "buffer": {"type": "storage"}},
            {"binding": 4, "visibility": C, "buffer": {"type": "storage"}},
            {"binding": 5, "visibility": C, "buffer": {"type": "storage"}},
            {"binding": 6, "visibility": C, "buffer": {"type": "storage"}},
        ])
        mg_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._mg_bin_bgl])
        self._mg_count_pipeline = dev.create_compute_pipeline(
            layout=mg_layout,
            compute={"module": self._mg_bin_shader, "entry_point": "cs_count"})
        self._mg_build_pipeline = dev.create_compute_pipeline(
            layout=mg_layout,
            compute={"module": self._mg_bin_shader, "entry_point": "cs_build"})
        self._mg_scatter_pipeline = dev.create_compute_pipeline(
            layout=mg_layout,
            compute={"module": self._mg_bin_shader, "entry_point": "cs_scatter"})

        self._cascade_pipeline = dev.create_render_pipeline(
            layout=cascade_layout,
            vertex={"module": self._cascade_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list"},
            fragment={"module": self._cascade_shader, "entry_point": "fs_main",
                      "targets": accum_targets},
        )

        # Subsample state set later via set_subsample_chunks. The
        # cap-based auto-LOD is the sole LOD knob — there is no
        # separate stride. Each draw renders min(n_total, cap)
        # particles, with each splat enlarged by (n_total/cap)^(1/3)
        # to compensate for the missing neighbors.
        self._subsample_chunks = None
        self._subsample_max_per_frame = 4_000_000
        self._slot_subsample_bgs = [None, None]
        self._active_subsample_slot = None
        self._world_offset = None

        # ---- Realistic stars overlay ----
        self._star_shader = dev.create_shader_module(
            code=_load_wgsl("star_realistic.wgsl"))
        self._star_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._star_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": VF, "buffer": {"type": "uniform"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float", "view_dimension": "2d-array"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": "filtering"}},
        ])
        # Load baked UBVRI PSFs (5 layers × 162×162) and upload as a
        # texture_2d_array. Falls back to a single-layer Gaussian if the
        # asset is missing.
        self._star_psf_tex, self._star_psf_view, self._star_psf_sampler = \
            self._load_star_psf_texture()
        self._star_bgl1 = _storage_bgl(dev, 1, V)
        star_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._star_bgl0, self._star_bgl1])
        # Additive blend over the resolved (sRGB / unorm) screen target.
        star_blend = {
            "color": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
            "alpha": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
        }
        self._star_pipeline = dev.create_render_pipeline(
            layout=star_layout,
            vertex={"module": self._star_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
            fragment={"module": self._star_shader, "entry_point": "fs_main",
                      "targets": [{"format": self.present_format, "blend": star_blend}]},
        )
        self._star_buf = None
        self._star_bg0 = None
        self._star_bg1 = None
        self._star_buf_dirty = True

        # Tunables — wired into the dev panel later.
        # Physical billboard radius, in world (snapshot) units. The
        # apparent screen size grows as 1/d as the camera approaches,
        # like a real spherical object. Tune from the dev panel.
        self.star_world_radius = 0.2
        self.star_intensity = 10.0
        # UBVRI dust opacities (cm^2/g, MRN-style averages). Σ in
        # Msun/pc^2 → multiply by 2.0834e-4 to get code-unit κ such
        # that τ = κ_code * Σ.
        _S2C = 2.0834e-4
        self.STAR_BANDS = [
            ("U", 700.0 * _S2C),
            ("B", 440.0 * _S2C),
            ("V", 200.0 * _S2C),
            ("R", 140.0 * _S2C),
            ("I", 100.0 * _S2C),
        ]
        # Index 5 is reserved for the RGB composite mode (I→R, V→G, B→B
        # mapping — standard astronomy false-color recipe).
        self.STAR_RGB_BANDS = (4, 2, 1)  # PSF/κ layer indices for (R, G, B)
        self.star_band_idx = 2  # default V
        self.star_extinction_kappa = self.STAR_BANDS[self.star_band_idx][1]
        self.star_extinction_enabled = True

    N_STAR_MODES = 6  # 5 single-band + 1 RGB composite

    @property
    def star_band(self):
        if self.star_band_idx >= len(self.STAR_BANDS):
            return "RGB"
        return self.STAR_BANDS[self.star_band_idx][0]

    def cycle_star_band(self, step=1):
        self.star_band_idx = (self.star_band_idx + step) % self.N_STAR_MODES
        if self.star_band_idx < len(self.STAR_BANDS):
            self.star_extinction_kappa = self.STAR_BANDS[self.star_band_idx][1]
        self._star_buf_dirty = True
        print(f"  [stars] band={self.star_band}")

    def toggle_star_extinction(self):
        self.star_extinction_enabled = not self.star_extinction_enabled
        self._star_buf_dirty = True
        print(f"  [stars] extinction {'ON' if self.star_extinction_enabled else 'OFF'}")

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
        self._colormap_tex_view = self._colormap_tex.create_view()
        # Invalidate any cached resolve/composite bind groups that
        # referenced the previous colormap.
        self._resolve_bg = None
        self._composite_bg = None
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

    # ---- Data management ----

    def set_particles(self, positions, hsml, masses, quantity=None):
        """Store CPU-side particle data. The actual GPU upload is
        performed by GPUCompute.upload_subsample_only on the first
        canvas tick.
        """
        self._all_pos = positions.astype(np.float32)
        self._all_hsml = hsml.astype(np.float32)
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)
        self.n_total = len(masses)

    def update_weights(self, masses, quantity=None):
        """Update the renderer's CPU mass/qty arrays after a field swap.
        The GPU side must be re-uploaded by GPUCompute.upload_weights.
        """
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)

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
        n_levels = max(1, int(self.multigrid_levels))
        for cb in chunks:
            # One params buffer (and bg0) per multigrid level. With
            # n_levels=1 this collapses to the original single-level
            # binding.
            params_bufs = []
            bg0s = []
            for _ in range(n_levels):
                pb = dev.create_buffer(
                    size=112, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
                params_bufs.append(pb)
                bg0s.append(_make_bind_group(
                    dev, self._splat_subsample_bgl0,
                    [self._camera_buf, pb]))
            # Per-chunk multigrid binning state. Sized for MAX_LEVELS=16.
            MAX_LV = 16
            bin_params_buf = dev.create_buffer(
                size=64,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            counts_buf = dev.create_buffer(
                size=MAX_LV * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            bases_buf = dev.create_buffer(
                size=MAX_LV * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            # Identity bases (all zero) for the n_levels==1 path so the
            # default splat shader binding works without the bin pass.
            dev.queue.write_buffer(bases_buf, 0, b"\x00" * (MAX_LV * 4))
            indirect_buf = dev.create_buffer(
                size=MAX_LV * 16,
                usage=(wgpu.BufferUsage.STORAGE
                       | wgpu.BufferUsage.INDIRECT
                       | wgpu.BufferUsage.COPY_DST))
            bg1 = _make_bind_group(
                dev, self._splat_bgl1,
                [cb["pos"], cb["hsml"], cb["mass"], cb["qty"],
                 cb["index"], bases_buf])
            mg_bg = dev.create_bind_group(
                layout=self._mg_bin_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": bin_params_buf}},
                    {"binding": 1, "resource": {"buffer": cb["pos"]}},
                    {"binding": 2, "resource": {"buffer": cb["hsml"]}},
                    {"binding": 3, "resource": {"buffer": counts_buf}},
                    {"binding": 4, "resource": {"buffer": bases_buf}},
                    {"binding": 5, "resource": {"buffer": cb["index"]}},
                    {"binding": 6, "resource": {"buffer": indirect_buf}},
                ],
            )
            out.append({
                "bg0": bg0s[0], "bg0s": bg0s, "bg1": bg1,
                "params_buf": params_bufs[0], "params_bufs": params_bufs,
                "pos": cb["pos"], "hsml": cb["hsml"], "index": cb["index"],
                "n": int(cb["n"]), "start": int(cb["start"]),
                "bin_params_buf": bin_params_buf,
                "counts_buf": counts_buf,
                "bases_buf": bases_buf,
                "indirect_buf": indirect_buf,
                "mg_bg": mg_bg,
            })
        self._subsample_chunks = out
        self._subsample_n_levels = n_levels
        # Stash the raw source chunks so set_multigrid_levels() can
        # rebuild the per-level bind groups without the caller having
        # to re-upload.
        self._subsample_source_chunks = list(chunks)
        self._subsample_source_offset = world_offset

    def set_multigrid_levels(self, n_levels):
        """Change the multigrid level count and rebuild the per-chunk
        bin state. Cheap (only allocates a few small uniform/storage
        buffers per chunk) and safe to call between frames.

        Preserves slot bind groups across the rebuild — the underlying
        source pos/hsml/mass/qty buffers are unchanged, so the app's
        composite-slot mass/qty bind groups remain valid; we just need
        to rebuild the parallel bgs that reference the (new) chunk
        objects' pos/hsml/index/bases buffers.
        """
        n_levels = max(1, int(n_levels))
        if n_levels == int(getattr(self, "_subsample_n_levels", 1)):
            self.multigrid_levels = n_levels
            return
        self.multigrid_levels = n_levels
        if getattr(self, "_subsample_source_chunks", None) is None:
            return

        # Capture the active slot index before set_subsample_chunks wipes it.
        prev_active = self._active_subsample_slot
        # Drop the previous chunk dicts and rebuild fresh — this gives
        # us new bases_buf storage initialized to all zeros, which is
        # critical when going from multigrid back to n_levels=1 (the
        # splat shader reads s_bases[0] and would otherwise see stale
        # base offsets left over from the prior bin pass).
        self.set_subsample_chunks(self._subsample_source_chunks,
                                  world_offset=self._subsample_source_offset)
        if self._accum_size[0] > 0:
            self._ensure_pyramid(*self._accum_size)
        # Restore active slot index so the next set_subsample_slot_chunks
        # call (or render) sees the prior selection.
        self._active_subsample_slot = prev_active
        # NOTE: prev_slot_bgs intentionally dropped — the bg1 layout is
        # unchanged but the bgs reference the OLD chunk dicts' pos/hsml/
        # index/bases buffers. If we kept them, vertex shader reads of
        # s_index/s_bases would see stale (freed) buffers. The caller
        # must re-issue set_subsample_slot_chunks() after this returns;
        # data_manager.py does so on its next render-state refresh.

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
                [ck["pos"], ck["hsml"], sc["mass"], sc["qty"],
                 ck["index"], ck["bases_buf"]])
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
        self._subsample_max_per_frame = max(int(max_inst), 1000)

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
        # User-controlled hsml multiplier (overlay slider) composes with
        # the stride-derived scaling so each splat both fills its share
        # of the subsample volume and honors the manual size knob.
        h_scale = (ratio ** (1.0 / 3.0)) * float(self.hsml_scale)
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
        n_levels = int(getattr(self, "_subsample_n_levels", 1))
        full_res_w = float(self._accum_size[0]) if self._accum_size[0] > 0 else 1.0
        n_grid_kernel = float(self.multigrid_n_grid_kernel)
        for ck in self._subsample_chunks:
            for lvl in range(n_levels):
                tail = _struct.pack(
                    "ffII ffII fff f",
                    fov_rad, float(camera.aspect), int(stride), int(ck["n"]),
                    h_scale, mass_scale,
                    int(lvl), int(n_levels),
                    full_res_w, n_grid_kernel,
                    0.0, 0.0,
                )
                pbs = ck.get("params_bufs", [ck["params_buf"]])
                self.device.queue.write_buffer(pbs[lvl], 0, cam_bytes + tail)

    def upload_stars(self, positions, masses, luminosity=None):
        """Upload star particle data.

        Keeps the CPU-side count and the position/mass arrays around so
        a future star-rendering path can pick them up without changing
        the call site. The GPU bind groups for the old dedicated star
        shader pipeline have been removed.
        """
        self.n_stars = len(masses)
        # Invalidate any cached extinction columns — star set has changed.
        self._star_columns = None
        self._star_columns_cam_pos = None
        if self.n_stars == 0:
            return
        self._star_positions = positions.astype(np.float64)
        self._star_masses = masses.astype(np.float32)
        if luminosity is None:
            # Fallback to mass as a stand-in luminosity proxy.
            self._star_luminosity = self._star_masses.copy()
        else:
            self._star_luminosity = np.asarray(luminosity, dtype=np.float32)
        self._star_buf_dirty = True

    def set_extinction_gas(self, positions, masses, hsml):
        """Provide gas particle data used to compute per-star extinction
        columns via a direct port of starforge_tools.star_gas_columns.

        Stored as float64 because the KDTree query in star_gas_columns
        runs on these arrays directly.
        """
        self._ext_xgas = np.ascontiguousarray(positions, dtype=np.float64)
        self._ext_mgas = np.ascontiguousarray(masses, dtype=np.float64)
        self._ext_hgas = np.ascontiguousarray(hsml, dtype=np.float64)
        self._star_columns = None
        self._star_columns_cam_pos = None

    def _update_star_columns(self, camera):
        """Recompute per-star gas column densities when the camera has
        translated. Pure rotation preserves every star→camera sightline
        (parallel-projection approximation along the camera forward axis,
        which only depends on the camera position relative to the stars
        when forward = normalize(mean_star_pos - cam_pos)), so we key the
        cache on camera.position alone.

        Direct port of starforge_tools.star_gas_columns; result lives in
        self._star_columns and is consumed by the future star draw path.
        """
        if getattr(self, "n_stars", 0) == 0:
            return
        if getattr(self, "_ext_xgas", None) is None:
            return

        cam_pos = np.asarray(camera.position, dtype=np.float64)
        prev = self._star_columns_cam_pos
        if (
            self._star_columns is not None
            and prev is not None
            and np.array_equal(prev, cam_pos)
        ):
            return  # rotation-only frame — cached columns are still valid

        xstar = self._star_positions
        xgas = self._ext_xgas
        mgas = self._ext_mgas
        hgas2 = self._ext_hgas * self._ext_hgas
        inv_pi = 1.0 / np.pi
        n_stars = xstar.shape[0]
        columns = np.zeros(n_stars, dtype=np.float64)

        # Per-star line-of-sight integration. Each star uses its own
        # ray (star → camera); a parallel projection along a single
        # direction would rotate all gas positions in lockstep as the
        # camera moves and produce correlated brightness flicker.
        #
        # For each gas particle g, compute its impact parameter b
        # against the ray and the along-ray coordinate t:
        #     r       = g − star
        #     t       = r · ray_dir         (positive = toward camera)
        #     b²      = |r|² − t²            (perpendicular to ray)
        # The particle contributes when:
        #     0 < t < |star − camera|       (lies between star and obs)
        #     b < h_g                       (impact parameter inside h)
        # using the smooth W(q) = (3/(π h²)) (1-q²)² deposit.
        for s in range(n_stars):
            ray = cam_pos - xstar[s]
            d_obs = float(np.linalg.norm(ray))
            if d_obs == 0.0:
                continue
            ray_dir = ray / d_obs
            r = xgas - xstar[s]                       # (n_gas, 3)
            t = r @ ray_dir                           # (n_gas,)
            r2 = np.einsum("ij,ij->i", r, r)          # |r|²
            b2 = np.maximum(r2 - t * t, 0.0)
            mask = (t > 0.0) & (t < d_obs) & (b2 < hgas2)
            if not np.any(mask):
                continue
            b2m = b2[mask]
            h2m = hgas2[mask]
            q2 = b2m / h2m
            w = 1.0 - q2
            w = w * w  # (1-q²)²
            columns[s] = np.sum(mgas[mask] * (3.0 * inv_pi / h2m) * w)

        self._star_columns = columns.astype(np.float32)
        self._star_columns_cam_pos = cam_pos.copy()


    # ---- Accumulation textures ----

    def _ensure_fbo(self, width, height, which=1):
        """Create or resize the accumulation texture triple."""
        if which == 1:
            if self._accum_size != (width, height) or self._accum_textures is None:
                self._accum_textures = self._create_accum_textures(width, height)
                self._accum_size = (width, height)
                # Resolve bind group references slot-1 accum views.
                self._resolve_bg = None
                self._composite_bg = None
                self._accum_pyramid = []
                self._accum_pyramid_levels = 0
                self._cascade_bgs = []
            self._ensure_pyramid(width, height)
        else:
            if self._accum_size2 == (width, height) and self._accum_textures2 is not None:
                return
            self._accum_textures2 = self._create_accum_textures(width, height)
            self._accum_size2 = (width, height)
            # Composite bind group references slot-2 accum views.
            self._composite_bg = None

    def _dispatch_multigrid_bin(self, camera, encoder, budget, n_total_chunks):
        """Run count → build → scatter for every chunk so each level's
        draw_indirect args and per-level index ranges are ready before
        the splat passes begin.
        """
        import struct as _struct
        dev = self.device
        n_levels = int(self._subsample_n_levels)
        full_res_w = float(self._accum_size[0]) if self._accum_size[0] > 0 else 1.0
        n_grid_kernel = float(self.multigrid_n_grid_kernel)
        offset = getattr(self, "_world_offset", None)
        cam_pos = camera.position - offset if offset is not None else camera.position
        proj = camera.projection_matrix()
        proj11 = float(proj[1, 1])
        # Reuse the same h_scale derivation as the splat path so the
        # binner sees the kernel size the splat will actually draw.
        ratio = max(n_total_chunks / max(budget, 1), 1.0)
        h_scale = (ratio ** (1.0 / 3.0)) * float(self.hsml_scale)

        zero_counts = b"\x00" * (16 * 4)
        for ck in self._subsample_chunks:
            n_to_consider = max(1, int(round(budget * ck["n"] / n_total_chunks)))
            n_to_consider = min(n_to_consider, int(ck["n"]))
            params = _struct.pack(
                "fff f fff f I I f f f f f f",
                float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), 0.0,
                float(camera.forward[0]), float(camera.forward[1]),
                float(camera.forward[2]), 0.0,
                int(n_to_consider), int(n_levels),
                full_res_w, n_grid_kernel,
                proj11, h_scale, 0.0, 0.0,
            )
            dev.queue.write_buffer(ck["bin_params_buf"], 0, params)
            dev.queue.write_buffer(ck["counts_buf"], 0, zero_counts)

        # Workgroup count cap is 65535 per dimension; with 256-thread
        # groups that's ~16.7 M particles per dispatch. Use a 2D grid
        # so chunks of up to ~4G particles fit in one dispatch.
        for ck in self._subsample_chunks:
            n_to_consider = max(1, int(round(budget * ck["n"] / n_total_chunks)))
            n_to_consider = min(n_to_consider, int(ck["n"]))
            n_groups = (n_to_consider + 255) // 256
            gx = min(n_groups, 65535)
            gy = (n_groups + 65535 - 1) // 65535
            cp = encoder.begin_compute_pass()
            cp.set_bind_group(0, ck["mg_bg"])
            cp.set_pipeline(self._mg_count_pipeline)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.set_pipeline(self._mg_build_pipeline)
            cp.dispatch_workgroups(1, 1, 1)
            cp.set_pipeline(self._mg_scatter_pipeline)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()

    def _ensure_pyramid(self, width, height):
        """Build (or rebuild) the multigrid pyramid for slot-1 accum.

        Levels [1..n_levels-1] are coarser than the full-res slot-1
        accum textures (which serve as level 0). Bind groups for the
        cascade pass are built here too — each binds level k's textures
        as inputs (the cascade fragment shader writes additively into
        level k-1, which is the bound render target).
        """
        n_levels = max(1, int(self.multigrid_levels))
        if (n_levels == self._accum_pyramid_levels
                and self._accum_pyramid is not None
                and len(self._accum_pyramid) == max(0, n_levels - 1)
                and self._accum_size == (width, height)):
            return
        self._accum_pyramid = []
        self._cascade_bgs = []
        for k in range(1, n_levels):
            w = max(1, width >> k)
            h = max(1, height >> k)
            self._accum_pyramid.append(self._create_accum_textures(w, h))
        # Build per-level cascade bind groups: input = level k textures.
        for k in range(1, n_levels):
            tex = self._accum_pyramid[k - 1]["views"]
            bg = self.device.create_bind_group(
                layout=self._cascade_bgl,
                entries=[
                    {"binding": 0, "resource": tex[0]},
                    {"binding": 1, "resource": tex[1]},
                    {"binding": 2, "resource": tex[2]},
                ],
            )
            self._cascade_bgs.append(bg)
        self._accum_pyramid_levels = n_levels

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

    def _load_star_psf_texture(self):
        """Load the baked UBVRI PSF stack into a 2D texture array.

        Asset: dataflyer/assets/star_psf_ubvri.npy, shape (5, H, W) float32.
        Falls back to a single-layer 2D Gaussian if the asset is missing.
        """
        dev = self.device
        asset = Path(__file__).parent / "assets" / "star_psf_ubvri.npy"
        if asset.exists():
            psf = np.load(asset).astype(np.float32)  # (5, H, W)
        else:
            print(f"  [stars] PSF asset not found at {asset}; using Gaussian fallback")
            H = 64
            yy, xx = np.mgrid[-1:1:H*1j, -1:1:H*1j]
            g = np.exp(-8.0 * (xx*xx + yy*yy)).astype(np.float32)
            g /= g.sum()
            psf = np.stack([g] * 5)
        n_layers, H, W = psf.shape
        # Per-layer normalization to peak=1 so star brightness is set
        # by the shader's intensity*flux factor, not by the PSF energy.
        peaks = psf.reshape(n_layers, -1).max(axis=1).reshape(n_layers, 1, 1)
        psf = psf / np.where(peaks > 0, peaks, 1.0)

        # r16float is universally filterable; r32float requires the
        # float32-filterable feature (not present on this backend).
        psf16 = psf.astype(np.float16)
        tex = dev.create_texture(
            size=(W, H, n_layers),
            format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d",
        )
        dev.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            psf16.tobytes(),
            {"offset": 0, "bytes_per_row": W * 2, "rows_per_image": H},
            (W, H, n_layers),
        )
        view = tex.create_view(dimension="2d-array")
        sampler = dev.create_sampler(
            mag_filter="linear", min_filter="linear",
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
        )
        return tex, view, sampler

    def _ensure_star_buffer(self):
        """(Re)build the per-star storage buffer holding (xyz, attenuated L)."""
        n = getattr(self, "n_stars", 0)
        if n == 0:
            return False
        if self._star_positions is None or self._star_luminosity is None:
            return False

        # World-origin shift mirrors the splat path so float32 precision
        # holds at cosmological scales.
        offset = getattr(self, "_world_offset", None)
        pos = self._star_positions
        if offset is not None:
            pos = pos - np.asarray(offset, dtype=np.float64)
        # Track the offset so we rebuild if it changes after upload.
        self._star_buf_offset_used = (
            None if offset is None else np.asarray(offset, dtype=np.float64).copy()
        )

        L0 = self._star_luminosity.astype(np.float32, copy=False)
        cols = getattr(self, "_star_columns", None)
        have_ext = (
            self.star_extinction_enabled
            and cols is not None
            and len(cols) == n
        )

        # Single-band channel: uses the *currently selected* band's κ
        # (or no extinction if disabled or in RGB mode where the single
        # value is just the bolometric L for fallback display).
        lum_single = L0.copy()
        if have_ext and self.star_band_idx < len(self.STAR_BANDS):
            tau = self.star_extinction_kappa * cols
            lum_single = lum_single * np.exp(-tau).astype(np.float32)

        # RGB triple: always computed using the fixed (I, V, B) → (R, G, B)
        # mapping so cycling into composite mode is instantaneous.
        rgb = np.empty((n, 3), dtype=np.float32)
        for ch, layer in enumerate(self.STAR_RGB_BANDS):
            kappa_ch = self.STAR_BANDS[layer][1]
            l = L0.copy()
            if have_ext:
                l = l * np.exp(-kappa_ch * cols).astype(np.float32)
            rgb[:, ch] = l

        # Layout: two vec4 per star.
        # vec4[0] = (x, y, z, L_raw)            ← drives billboard size
        # vec4[1] = (L_R_att, L_G_att, L_B_att, L_single_att)
        # The shader divides each L_*_att by L_raw to get a per-channel
        # attenuation factor in [0,1], so per-pixel surface brightness
        # is independent of intrinsic luminosity (which goes entirely
        # into the billboard area via radius ∝ sqrt(L)).
        data = np.zeros((n, 8), dtype=np.float32)
        data[:, 0:3] = pos.astype(np.float32)
        data[:, 3] = L0  # raw bolometric, for sizing
        data[:, 4:7] = rgb
        data[:, 7] = lum_single

        nbytes = data.nbytes
        if self._star_buf is None or self._star_buf.size < nbytes:
            self._star_buf = self.device.create_buffer(
                size=nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self._star_bg1 = self.device.create_bind_group(
                layout=self._star_bgl1,
                entries=[{"binding": 0,
                          "resource": {"buffer": self._star_buf}}],
            )
        self.device.queue.write_buffer(self._star_buf, 0, data.tobytes())

        if self._star_bg0 is None:
            self._star_bg0 = self.device.create_bind_group(
                layout=self._star_bgl0,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._camera_buf}},
                    {"binding": 1, "resource": {"buffer": self._star_params_buf}},
                    {"binding": 2, "resource": self._star_psf_view},
                    {"binding": 3, "resource": self._star_psf_sampler},
                ],
            )
        self._star_buf_dirty = False
        return True

    def _encode_star_overlay(self, encoder, screen_view):
        """Draw realistic-stars billboards over the resolved screen.

        Updates the per-star extinction column cache (cheap on rotation-
        only frames), refreshes the storage buffer when stars or columns
        changed, then issues an instanced quad draw additively blended
        into `screen_view`.
        """
        if getattr(self, "n_stars", 0) == 0:
            return
        cam = getattr(self, "_last_camera", None)
        if cam is None or screen_view is None:
            return

        prev_cam_pos = self._star_columns_cam_pos
        self._update_star_columns(cam)
        # Force a rebuild if either columns refreshed or the world-origin
        # offset changed since the last buffer build.
        cur_off = getattr(self, "_world_offset", None)
        used_off = getattr(self, "_star_buf_offset_used", None)
        offset_changed = (
            (cur_off is None) != (used_off is None)
            or (cur_off is not None and used_off is not None
                and not np.array_equal(np.asarray(cur_off, dtype=np.float64), used_off))
        )
        if (
            self._star_buf_dirty
            or self._star_columns_cam_pos is not prev_cam_pos
            or offset_changed
        ):
            self._star_buf_dirty = True
        if self._star_buf_dirty:
            if not self._ensure_star_buffer():
                return

        import struct
        params = struct.pack(
            "ffff",
            float(self.star_world_radius),
            float(self.star_intensity),
            float(self.star_band_idx),
            0.0,
        )
        self.device.queue.write_buffer(self._star_params_buf, 0, params)

        rp = encoder.begin_render_pass(
            color_attachments=[{
                "view": screen_view,
                "clear_value": (0, 0, 0, 1),
                "load_op": "load",
                "store_op": "store",
            }],
        )
        rp.set_pipeline(self._star_pipeline)
        rp.set_bind_group(0, self._star_bg0)
        rp.set_bind_group(1, self._star_bg1)
        rp.draw(4, int(self.n_stars), 0, 0)
        rp.end()

    def _render_accum(self, camera, width, height, accum_textures, encoder=None):
        """Render additive accumulation pass into given textures.

        Renders:
          - real particles + anisotropic summaries → full-res `accum_textures`
          - isotropic summary splats → half-res `_accum_textures_lo`

        If `encoder` is None, a fresh command encoder is created and
        submitted at the end of the call (legacy path used by screenshot).
        Otherwise the accum pass is appended to the supplied encoder and
        the caller is responsible for submission.
        """
        self._write_camera_uniforms(camera, width, height)

        # Compute the per-chunk dispatch budget BEFORE beginning the
        # render pass — wgpu forbids write_buffer mid-pass, so all
        # uniform writes must happen now.
        budget = 0
        n_total_chunks = 0
        eff_stride = 1.0
        if self._subsample_chunks is not None:
            n_total_chunks = sum(ck["n"] for ck in self._subsample_chunks)
            cap = self._subsample_max_per_frame
            budget = min(n_total_chunks, cap)
            budget = max(1, budget)
            # Each rendered splat stands in for `eff_stride` particles.
            eff_stride = max(n_total_chunks / max(budget, 1), 1.0)
            self._write_subsample_params(camera, eff_stride)

        dev = self.device
        owns_encoder = encoder is None
        if owns_encoder:
            encoder = dev.create_command_encoder()

        n_levels = max(1, int(getattr(self, "_subsample_n_levels", 1)))

        # Multigrid: bin pass writes per-chunk index buffer + indirect
        # args. Must run before begin_render_pass.
        if (n_levels > 1
                and self._subsample_chunks is not None
                and accum_textures is self._accum_textures):
            self._dispatch_multigrid_bin(camera, encoder, budget, n_total_chunks)

        # Pre-resolve the slot bg list once.
        slot = self._active_subsample_slot
        slot_bgs = (self._slot_subsample_bgs[slot]
                    if slot is not None
                    and self._slot_subsample_bgs[slot] is not None
                    else None)

        # Build a list of per-level (views, bg0_index) pairs. Level 0
        # is the supplied accum_textures (full res); higher levels come
        # from the multigrid pyramid attached to the slot-1 set.
        level_views = [accum_textures["views"]]
        if n_levels > 1 and accum_textures is self._accum_textures:
            for lvl in range(1, n_levels):
                level_views.append(self._accum_pyramid[lvl - 1]["views"])
        elif n_levels > 1:
            # Composite slot or other accum target: fall back to single
            # level for now (multigrid only wired into slot-1 path).
            n_levels = 1

        for lvl in range(n_levels):
            views = level_views[lvl]
            render_pass = encoder.begin_render_pass(color_attachments=[
                {"view": views[0], "clear_value": (0, 0, 0, 0),
                 "load_op": "clear", "store_op": "store"},
                {"view": views[1], "clear_value": (0, 0, 0, 0),
                 "load_op": "clear", "store_op": "store"},
                {"view": views[2], "clear_value": (0, 0, 0, 0),
                 "load_op": "clear", "store_op": "store"},
            ])
            if self._subsample_chunks is not None:
                render_pass.set_pipeline(self._splat_subsample_pipeline)
                for i, ck in enumerate(self._subsample_chunks):
                    instances = max(1, int(round(budget * ck["n"] / n_total_chunks)))
                    instances = min(instances, ck["n"])
                    bg0 = ck.get("bg0s", [ck["bg0"]])[lvl]
                    render_pass.set_bind_group(0, bg0)
                    bg1 = slot_bgs[i] if slot_bgs is not None else ck["bg1"]
                    render_pass.set_bind_group(1, bg1)
                    if n_levels > 1:
                        render_pass.draw_indirect(
                            ck["indirect_buf"], lvl * 16)
                    else:
                        render_pass.draw(4, instances, 0, 0)
            render_pass.end()

        # Cascade: fold coarser levels into finer ones, ending at level 0
        # which the resolve pass consumes.
        if n_levels > 1:
            for lvl in range(n_levels - 1, 0, -1):
                target_views = level_views[lvl - 1]
                rp = encoder.begin_render_pass(color_attachments=[
                    {"view": target_views[0], "load_op": "load", "store_op": "store"},
                    {"view": target_views[1], "load_op": "load", "store_op": "store"},
                    {"view": target_views[2], "load_op": "load", "store_op": "store"},
                ])
                rp.set_pipeline(self._cascade_pipeline)
                rp.set_bind_group(0, self._cascade_bgs[lvl - 1])
                rp.draw(3, 1, 0, 0)
                rp.end()

        if owns_encoder:
            dev.queue.submit([encoder.finish()])

    def render(self, camera, width, height, encoder=None, screen_view=None,
               skip_accum=False):
        """Render particle splats via additive accumulation + resolve.

        If `encoder` is provided, the accum + resolve passes are appended
        to it and the caller owns submission. `screen_view` must then also
        be supplied (the swapchain texture view to render into). When both
        are None the legacy path is used: this method creates its own
        encoder, acquires the current swapchain texture, and submits.

        `skip_accum=True` reuses the prior frame's accumulation textures
        (still resident on the GPU) and only re-runs the resolve pass.
        Used by the main loop on UI-only dirty frames so typing into a
        text field doesn't trigger an N-particle re-accum.
        """
        self._viewport_width = width
        self._last_camera = camera
        if self._colormap_tex is None:
            return
        if self.n_particles == 0 and self._subsample_chunks is None:
            return

        self._ensure_fbo(width, height, which=1)

        owns_encoder = encoder is None
        if owns_encoder:
            if self.canvas_context is None:
                return
            encoder = self.device.create_command_encoder()
            current_tex = self.canvas_context.get_current_texture()
            screen_view = current_tex.create_view()

        # Accum pass — appended to the shared encoder. Skipped on
        # UI-only dirty frames where the prior accum textures are still
        # valid.
        if not skip_accum:
            self._render_accum(camera, width, height, self._accum_textures,
                               encoder=encoder)

        import struct
        resolve_data = struct.pack("ffII", self.qty_min, self.qty_max,
                                   self.resolve_mode, self.log_scale)
        self.device.queue.write_buffer(self._resolve_params_buf, 0, resolve_data)

        if self._resolve_bg is None:
            self._resolve_bg = self.device.create_bind_group(
                layout=self._resolve_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._resolve_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._colormap_tex_view},
                    {"binding": 5, "resource": self._colormap_sampler},
                ],
            )

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
        render_pass.set_bind_group(0, self._resolve_bg)
        render_pass.draw(3, 1, 0, 0)  # fullscreen triangle
        render_pass.end()
        self._encode_star_overlay(encoder, screen_view)
        if owns_encoder:
            self.device.queue.submit([encoder.finish()])

    def render_composite(self, camera, width, height, mode1, min1, max1, log1,
                         mode2, min2, max2, log2,
                         encoder=None, screen_view=None):
        """Composite two pre-filled FBOs.

        Like `render()`, accepts an optional external encoder + swapchain
        view so the caller can bundle this pass with overlay/UI passes
        into a single submit.
        """
        self._viewport_width = width
        self._ensure_fbo(width, height, which=1)
        self._ensure_fbo(width, height, which=2)

        owns_encoder = encoder is None
        if owns_encoder:
            if self.canvas_context is None:
                return
            current_tex = self.canvas_context.get_current_texture()
            screen_view = current_tex.create_view()
            encoder = self.device.create_command_encoder()

        import struct

        comp_data = struct.pack("ffIIffII", min1, max1, mode1, log1, min2, max2, mode2, log2)
        self.device.queue.write_buffer(self._composite_params_buf, 0, comp_data)

        if self._composite_bg is None:
            self._composite_bg = self.device.create_bind_group(
                layout=self._composite_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._composite_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._accum_textures2["views"][0]},
                    {"binding": 5, "resource": self._accum_textures2["views"][1]},
                    {"binding": 6, "resource": self._accum_textures2["views"][2]},
                    {"binding": 7, "resource": self._colormap_tex_view},
                    {"binding": 8, "resource": self._colormap_sampler},
                ],
            )

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
        render_pass.set_bind_group(0, self._composite_bg)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        self._encode_star_overlay(encoder, screen_view)
        if owns_encoder:
            self.device.queue.submit([encoder.finish()])

    def screenshot(self, path, width, height, camera, composite_args=None):
        """Render one frame at (width, height) into an offscreen RGBA8
        texture and save it to `path`.

        Args:
            path: output file path. Format inferred from extension by PIL.
            width, height: framebuffer size in pixels.
            camera: Camera instance.
            composite_args: optional tuple
                (m1, lo1, hi1, log1, m2, lo2, hi2, log2)
                to render in composite mode. None → single-slot resolve.
        """
        if self._colormap_tex is None:
            raise RuntimeError("screenshot: colormap not set")

        dev = self.device

        # 1) Accumulate particles into the FBO triple. In composite mode
        # the caller is responsible for having uploaded both slots; we
        # render slot 0 into _accum_textures and slot 1 into
        # _accum_textures2 (matching the live composite path).
        if composite_args is not None:
            self._ensure_fbo(width, height, which=1)
            self._ensure_fbo(width, height, which=2)
            self.set_active_subsample_slot(0)
            self._write_camera_uniforms(camera, width, height)
            self._render_accum(camera, width, height, self._accum_textures)
            self.set_active_subsample_slot(1)
            self._write_camera_uniforms(camera, width, height)
            self._render_accum(camera, width, height, self._accum_textures2)
            self.set_active_subsample_slot(None)
        else:
            self._ensure_fbo(width, height, which=1)
            self._write_camera_uniforms(camera, width, height)
            self._render_accum(camera, width, height, self._accum_textures)

        # 2) Allocate an offscreen target in the same format the resolve
        # / composite pipeline was built against (the swapchain
        # present_format). We'll byte-swap on the CPU side if needed.
        out_tex = dev.create_texture(
            size=(width, height, 1),
            format=self.present_format,
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.COPY_SRC),
        )
        out_view = out_tex.create_view()

        # 3) Run the resolve (or composite) pass into out_tex. The two
        # paths build their own bind groups against the existing
        # _resolve_pipeline / _composite_pipeline.
        import struct
        if composite_args is not None:
            m1, lo1, hi1, log1, m2, lo2, hi2, log2 = composite_args
            comp_data = struct.pack(
                "ffIIffII", lo1, hi1, m1, log1, lo2, hi2, m2, log2)
            dev.queue.write_buffer(self._composite_params_buf, 0, comp_data)
            bind_group = dev.create_bind_group(
                layout=self._composite_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._composite_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._accum_textures2["views"][0]},
                    {"binding": 5, "resource": self._accum_textures2["views"][1]},
                    {"binding": 6, "resource": self._accum_textures2["views"][2]},
                    {"binding": 7, "resource": self._colormap_tex_view},
                    {"binding": 8, "resource": self._colormap_sampler},
                ],
            )
            pipeline = self._composite_pipeline
        else:
            resolve_data = struct.pack(
                "ffII", self.qty_min, self.qty_max,
                self.resolve_mode, self.log_scale)
            dev.queue.write_buffer(self._resolve_params_buf, 0, resolve_data)
            bind_group = dev.create_bind_group(
                layout=self._resolve_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._resolve_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._colormap_tex_view},
                    {"binding": 5, "resource": self._colormap_sampler},
                ],
            )
            pipeline = self._resolve_pipeline

        encoder = dev.create_command_encoder()
        rp = encoder.begin_render_pass(color_attachments=[{
            "view": out_view, "clear_value": (0, 0, 0, 1),
            "load_op": "clear", "store_op": "store",
        }])
        rp.set_pipeline(pipeline)
        rp.set_bind_group(0, bind_group)
        rp.draw(3, 1, 0, 0)
        rp.end()
        dev.queue.submit([encoder.finish()])

        # 4) Read back. read_texture requires bytes_per_row to be a
        # multiple of 256, so we round up and crop the padding.
        bytes_per_row = ((width * 4 + 255) // 256) * 256
        data = dev.queue.read_texture(
            {"texture": out_tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": bytes_per_row},
            (width, height, 1),
        )
        arr = np.frombuffer(data, dtype=np.uint8).reshape(
            height, bytes_per_row // 4, 4)
        arr = arr[:, :width, :]
        # Convert BGRA → RGBA if needed.
        if "bgra" in self.present_format:
            arr = arr[:, :, [2, 1, 0, 3]]
        from PIL import Image
        Image.fromarray(arr, mode="RGBA").save(path)
        print(f"  Saved screenshot: {path}")

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
        """Read back accumulation textures and compute the qty range.

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

        den = self._read_accum_texture_r(self._accum_textures["textures"][1])
        mask = den > 1e-30

        if self.resolve_mode == 2:
            num = self._read_accum_texture_r(self._accum_textures["textures"][0])
            sq = self._read_accum_texture_r(self._accum_textures["textures"][2])
            with np.errstate(invalid="ignore"):
                mean = np.where(mask, num / den, 0)
                mean_sq = np.where(mask, sq / den, 0)
            vals = np.sqrt(np.maximum(mean_sq - mean * mean, 0))[mask]
            mass = den[mask]
        elif self.resolve_mode == 1:
            num = self._read_accum_texture_r(self._accum_textures["textures"][0])
            with np.errstate(invalid="ignore"):
                vals = np.where(mask, num / den, 0)[mask]
            mass = den[mask]
        else:
            vals = den[mask]
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
            lim_lo, lim_hi = max_entropy_limits(
                vals, entropy_weights, log_scale=bool(self.log_scale))

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
        """Drop GPU resource references; wgpu garbage-collects them."""
        self._subsample_chunks = None
        self._slot_subsample_bgs = [None, None]
        self._star_positions = None
        self._star_masses = None
        self._accum_textures = None
        self._accum_textures2 = None
