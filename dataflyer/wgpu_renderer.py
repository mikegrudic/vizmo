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
        self.lod_pixels = 16
        self.log_scale = 1
        self.max_render_particles = 64_000_000
        # LOD: only "subsample" is supported. The dropdown / overlay
        # carry it for backwards compat with the existing UI strings.
        self.lod_strategy = "subsample"
        self.kernel = "cubic_spline"
        self.auto_lod = True
        self.target_fps = 15.0
        self.auto_lod_smooth = 0.3
        # PID gains kept on the renderer for the auto-LOD overlay panel
        self.pid_Kp = 0.5
        self.pid_Kd = 0.0
        self.pid_Ki = 0.0
        self.skip_vsync = False
        self.hsml_scale = 1.0

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
        self._star_bufs = {}

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
        self._star_shader = dev.create_shader_module(
            code=_load_wgsl("star.wgsl", include_common=True))

        # Camera uniform: view + proj + viewport + kernel_id + pad = 144 B
        self._camera_buf = dev.create_buffer(
            size=144, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        # Star params (point_size, ...)
        self._star_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
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
        # bg1: pos / hsml / mass / qty per chunk
        self._splat_bgl1 = _storage_bgl(dev, 4, V)
        # Star pipeline (always-on overlay for star particles, separate feature)
        self._star_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": V, "buffer": {"type": "uniform"}}])
        self._star_bgl1 = _storage_bgl(dev, 2, V)

        self._splat_subsample_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._splat_subsample_bgl0, self._splat_bgl1])
        self._star_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._star_bgl0, self._star_bgl1])

        accum_targets = [{"format": self._accum_format, "blend": _additive_blend()}] * 3
        self._splat_subsample_pipeline = _make_render_pipeline(
            dev, self._splat_subsample_layout, self._splat_subsample_shader,
            accum_targets)
        self._star_pipeline = _make_render_pipeline(
            dev, self._star_layout, self._star_shader,
            [{"format": self.present_format, "blend": _alpha_blend()}])

        self._star_bg0 = _make_bind_group(
            dev, self._star_bgl0, [self._camera_buf, self._star_params_buf])

        # Subsample state set later via set_subsample_chunks/set_subsample_stride.
        self._subsample_chunks = None
        self._subsample_stride = 1
        self._subsample_max_per_frame = 4_000_000
        self._slot_subsample_bgs = [None, None]
        self._active_subsample_slot = None
        self._world_offset = None

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

    # ---- Data management (matches SplatRenderer interface) ----

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
        # Default LOD: stride ~ n^(1/3)/16. Auto-LOD adapts from here.
        self.lod_pixels = max(1.0, self.n_total ** (1.0 / 3.0) / 16.0)

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
        for ck in self._subsample_chunks:
            tail = _struct.pack(
                "ffII ffII",
                fov_rad, float(camera.aspect), int(stride), int(ck["n"]),
                h_scale, mass_scale, 0, 0,
            )
            self.device.queue.write_buffer(
                ck["params_buf"], 0, cam_bytes + tail)

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
        """Create or resize the accumulation texture triple."""
        if which == 1:
            if self._accum_size != (width, height) or self._accum_textures is None:
                self._accum_textures = self._create_accum_textures(width, height)
                self._accum_size = (width, height)
                # Resolve bind group references slot-1 accum views.
                self._resolve_bg = None
                self._composite_bg = None
        else:
            if self._accum_size2 == (width, height) and self._accum_textures2 is not None:
                return
            self._accum_textures2 = self._create_accum_textures(width, height)
            self._accum_size2 = (width, height)
            # Composite bind group references slot-2 accum views.
            self._composite_bg = None

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

        # Subsample params must be written before the render pass begins.
        if self._subsample_chunks is not None:
            self._write_subsample_params(camera, self._subsample_stride)

        dev = self.device
        owns_encoder = encoder is None
        if owns_encoder:
            encoder = dev.create_command_encoder()

        views = accum_textures["views"]
        render_pass = encoder.begin_render_pass(color_attachments=[
            {"view": views[0], "clear_value": (0, 0, 0, 0),
             "load_op": "clear", "store_op": "store"},
            {"view": views[1], "clear_value": (0, 0, 0, 0),
             "load_op": "clear", "store_op": "store"},
            {"view": views[2], "clear_value": (0, 0, 0, 0),
             "load_op": "clear", "store_op": "store"},
        ])

        if self._subsample_chunks is not None:
            # Compute-driven splat: one draw per chunk. Per-chunk dispatch
            # count is proportional to the global instance budget.
            slot = self._active_subsample_slot
            slot_bgs = (self._slot_subsample_bgs[slot]
                        if slot is not None
                        and self._slot_subsample_bgs[slot] is not None
                        else None)
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
        render_pass.end()
        if owns_encoder:
            dev.queue.submit([encoder.finish()])

    def render(self, camera, width, height, encoder=None, screen_view=None):
        """Render particle splats via additive accumulation + resolve.

        If `encoder` is provided, the accum + resolve passes are appended
        to it and the caller owns submission. `screen_view` must then also
        be supplied (the swapchain texture view to render into). When both
        are None the legacy path is used: this method creates its own
        encoder, acquires the current swapchain texture, and submits.
        """
        self._viewport_width = width
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

        # Accum pass — appended to the shared encoder.
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
        """Drop GPU resource references; wgpu garbage-collects them."""
        self._subsample_chunks = None
        self._slot_subsample_bgs = [None, None]
        self._star_bufs = {}
        self._accum_textures = None
        self._accum_textures2 = None
