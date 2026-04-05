"""wgpu backend application: GLFW window + WGPURenderer.

Streamlined main loop that uses wgpu for rendering instead of moderngl.
Overlays are not yet ported — this provides core rendering only.
"""

import time
import atexit
import numpy as np
import glfw

import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info

from .camera import Camera
from .data_manager import SnapshotData, find_snapshots
from .wgpu_renderer import WGPURenderer
from .colormaps import colormap_to_texture_data, AVAILABLE_COLORMAPS
from .renderer import RenderMode
from .field_ops import (resolve_field, compute_weights, compute_slot_fields,
                        project_vector, combine_fields, make_default_app_state,
                        is_los_stale, SD_OPS, RENDER_MODES, VECTOR_PROJECTIONS)


def run_wgpu_app(snapshot_path, width=1920, height=1080, fov=90.0,
                 screenshot=None, benchmark=None, fullscreen=False):
    """Run the DataFlyer application with the wgpu backend."""
    import os
    snapshot_path = os.path.abspath(snapshot_path)

    # Initialize GLFW without OpenGL (wgpu uses Vulkan/Metal)
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    atexit.register(glfw.terminate)

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # No OpenGL context
    glfw.window_hint(glfw.RESIZABLE, True)

    monitor = glfw.get_primary_monitor() if fullscreen else None
    window = glfw.create_window(width, height, "DataFlyer [wgpu]", monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create wgpu device and canvas context
    present_info = get_glfw_present_info(window)
    canvas_context = wgpu.gpu.get_canvas_context(present_info)

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    print(f"  wgpu adapter: {info.get('description', 'unknown')}")
    print(f"    vendor:  {info.get('vendor', '?')}")
    print(f"    device:  {info.get('device', '?')}")
    print(f"    backend: {info.get('backend_type', info.get('adapter_type', '?'))}")
    print(f"    driver:  {info.get('driver', info.get('driver_description', '?'))}")

    # Request float32-blendable if available
    req_features = set()
    if "float32-blendable" in adapter.features:
        req_features.add("float32-blendable")

    # Request max storage buffer size the adapter supports (need ~1.6 GB for 134M particles)
    max_ssbo = min(adapter.limits["max-storage-buffer-binding-size"], 2**32 - 1)
    max_buf = min(adapter.limits["max-buffer-size"], 2**38)
    device = adapter.request_device_sync(
        required_features=req_features,
        required_limits={
            "max_storage_buffer_binding_size": max_ssbo,
            "max_buffer_size": max_buf,
        },
    )

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    canvas_context.set_physical_size(fb_w, fb_h)
    # Force linear format — colormap data from matplotlib is already sRGB-encoded,
    # so we must NOT apply sRGB gamma again (bgra8unorm-srgb would double-gamma).
    present_format = "bgra8unorm"
    canvas_context.configure(device=device, format=present_format)

    # Load data
    print(f"Loading {snapshot_path}...")
    data = SnapshotData(snapshot_path)
    print(f"  {data.n_particles:,} gas particles loaded")

    # Camera
    camera = Camera(fov=fov, aspect=width / height)
    boxsize = data.header.get("BoxSize", None)
    camera.auto_scale(data.positions, boxsize=boxsize)

    # Renderer
    renderer = WGPURenderer(device, canvas_context, present_format)
    renderer._viewport_width = width  # use window size for LOD, not retina framebuffer size

    # Colormap
    rgba = colormap_to_texture_data("magma")
    renderer.set_colormap(rgba)

    # Load particles — render first frame without grid, then build grid
    weights = data.get_field("Masses")
    renderer.use_tree = False  # skip grid build for instant first frame
    renderer.set_particles(data.positions, data.hsml, weights)
    renderer.update_visible(camera)

    # Render + present first frame immediately so window is responsive
    try:
        renderer.render(camera, fb_w, fb_h)
        canvas_context.present()
        glfw.poll_events()
    except Exception:
        pass

    # Build grid in background thread so the window stays responsive
    from .gpu_compute import GPUCompute
    import threading

    gpu_compute = None
    _bg = {"grid": None, "done": False}  # mutable container for thread result

    def _build_grid_bg():
        try:
            # Use renderer's _build_grid to respect use_adaptive_tree flag
            renderer._all_pos = data.positions.astype(np.float32)
            renderer._all_mass = weights.astype(np.float32)
            renderer._all_hsml = data.hsml.astype(np.float32)
            renderer._all_qty = weights.astype(np.float32)
            _bg["grid"] = renderer._build_grid()
        except Exception as e:
            print(f"  Grid build failed: {e}")
        _bg["done"] = True

    glfw.set_window_title(window, "DataFlyer [wgpu] | Building spatial grid...")
    grid_thread = threading.Thread(target=_build_grid_bg, daemon=True)
    grid_thread.start()

    # UI overlays
    from .wgpu_overlay import WGPUDevOverlay, WGPUUserMenu
    overlay = WGPUDevOverlay(device, present_format)
    user_menu = WGPUUserMenu(device, present_format)
    _timings = {"cull": 0, "upload": 0, "render": 0}
    _last_message = ""
    _render_mode = RenderMode.surface_density("Masses")
    _cmap_idx = 0
    _s = make_default_app_state(data)
    _sd_fields = _s["sd_fields"]
    _vector_fields = _s["vector_fields"]
    _sd_field = _s["sd_field"]
    _sd_field2 = _s["sd_field2"]
    _sd_op = _s["sd_op"]
    _render_mode_name = _s["render_mode_name"]
    _wa_data_field = _s["wa_data_field"]
    _RENDER_MODES = RENDER_MODES
    _SD_OPS = SD_OPS
    _VECTOR_PROJECTIONS = VECTOR_PROJECTIONS
    _vector_projection = _s["vector_projection"]
    _los_camera_fwd = _s["los_camera_fwd"]
    _composite = _s["composite"]
    _slot = _s["slot"]

    # Stars
    if data.n_stars > 0:
        renderer.upload_stars(data.star_positions, data.star_masses)
        print(f"  {data.n_stars} star particles loaded")

    # Input callbacks
    def _auto_range_composite_slot(slot_idx, label):
        s = _state["_slot"][slot_idx]
        renderer.resolve_mode = s["resolve"]
        renderer.log_scale = s["log"]

        gpu_ready_now = gpu_compute is not None and getattr(gpu_compute, '_upload_ready', False)
        fb_w_, fb_h_ = glfw.get_framebuffer_size(window)

        if gpu_ready_now and renderer._grid is not None:
            # GPU path: write sorted arrays directly, dispatch, render, read back
            sorted_mass, sorted_qty = _ensure_slot_sorted(slot_idx)
            device.queue.write_buffer(gpu_compute._particle_bufs["mass"], 0, sorted_mass.tobytes())

            sl = _state["_slot"][slot_idx]
            is_los_vec = (sl.get("proj") == "LOS" and
                          sl["mode"] in ("WeightedAverage", "WeightedVariance") and
                          sl["data"] in _state["_vector_fields"])
            if is_los_vec:
                if not gpu_compute.has_los_field() or getattr(gpu_compute, '_los_field_name', '') != sl["data"]:
                    gpu_compute.upload_vector_field(renderer._grid, sl["data"], data)
                gpu_compute.dispatch_los_project(camera)
            else:
                device.queue.write_buffer(gpu_compute._particle_bufs["qty"], 0, sorted_qty.tobytes())

            n_out, _, summary_data = gpu_compute.dispatch_cull(
                camera, renderer.max_render_particles,
                lod_pixels=renderer.lod_pixels,
                viewport_width=renderer._viewport_width,
                summary_overlap=renderer.summary_overlap,
            )
            renderer.set_particle_buffers_from_gpu(gpu_compute.get_output_buffers(), n_out)
        else:
            # CPU fallback
            w, q = app_proxy._compute_slot(s)
            renderer.update_weights(w, q)
            renderer.update_visible(camera)

        renderer._ensure_fbo(fb_w_, fb_h_, which=1)
        renderer._write_camera_uniforms(camera, fb_w_, fb_h_)
        renderer._render_accum(camera, fb_w_, fb_h_, renderer._accum_textures)
        lo, hi = renderer.read_accum_range()
        s["min"] = lo
        s["max"] = hi
        print(f"Auto-range {label}: {lo:.3g} .. {hi:.3g}")

    def _adjust_range(rend, factor):
        mid = (rend.qty_min + rend.qty_max) / 2
        half = (rend.qty_max - rend.qty_min) / 2 * factor
        rend.qty_min = mid - half
        rend.qty_max = mid + half

    def key_callback(win, key, scancode, action, mods):
        nonlocal user_lod, user_budget, _cmap_idx, needs_auto_range, ui_hidden
        if user_menu.on_key(key, action):
            return
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_R:
                if _state["_composite"]:
                    _auto_range_composite_slot(1, "Color")
                else:
                    lo, hi = renderer.read_accum_range()
                    renderer.qty_min = lo
                    renderer.qty_max = hi
                    print(f"Auto-range: {lo:.3g} .. {hi:.3g}")
            elif key == glfw.KEY_T:
                if _state["_composite"]:
                    _auto_range_composite_slot(0, "Lightness")
            elif key == glfw.KEY_L:
                    renderer.log_scale = 1 - renderer.log_scale
                    needs_auto_range = True
            elif key == glfw.KEY_RIGHT_BRACKET:
                renderer.lod_pixels = max(1, renderer.lod_pixels // 2)
                user_lod = renderer.lod_pixels
                print(f"LOD: {renderer.lod_pixels}px (more detail)")
            elif key == glfw.KEY_LEFT_BRACKET:
                renderer.lod_pixels = min(256, renderer.lod_pixels * 2)
                user_lod = renderer.lod_pixels
                print(f"LOD: {renderer.lod_pixels}px (faster)")
            elif key == glfw.KEY_PERIOD:
                renderer.max_render_particles = min(
                    renderer.n_total, int(renderer.max_render_particles * 2))
                user_budget = renderer.max_render_particles
                print(f"Max particles: {renderer.max_render_particles/1e6:.1f}M")
            elif key == glfw.KEY_COMMA:
                renderer.max_render_particles = max(
                    1_000, renderer.max_render_particles // 2)
                user_budget = renderer.max_render_particles
                print(f"Max particles: {renderer.max_render_particles/1e6:.1f}M")
            elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
                _adjust_range(renderer, 0.8)
            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
                _adjust_range(renderer, 1.25)
            elif key == glfw.KEY_C:
                nonlocal _cmap_idx
                _cmap_idx = (_cmap_idx + 1) % len(AVAILABLE_COLORMAPS)
                from .colormaps import colormap_to_texture_data
                renderer.set_colormap(colormap_to_texture_data(AVAILABLE_COLORMAPS[_cmap_idx]))
                _state["_cmap_idx"] = _cmap_idx
            elif key == glfw.KEY_P:
                renderer.screenshot(f"dataflyer_{int(time.time())}.png")
            elif key == glfw.KEY_F1 or key == glfw.KEY_BACKSLASH:
                overlay.enabled = not overlay.enabled
            elif key == glfw.KEY_TAB:
                ui_hidden = not ui_hidden
        camera.on_key(key, action)

    glfw.set_key_callback(window, key_callback)

    # Shared mutable state dict — the app proxy reads/writes this,
    # and the main loop syncs closure locals from it each frame.
    _state = {
        "_sd_field": _sd_field,
        "_sd_field2": _sd_field2,
        "_sd_op": _sd_op,
        "_render_mode_name": _render_mode_name,
        "_wa_data_field": _wa_data_field,
        "_vector_fields": _vector_fields,
        "_vector_projection": _vector_projection,
        "_los_camera_fwd": _los_camera_fwd,
        "_cmap_idx": _cmap_idx,
        "_slot": _slot,
        "_needs_auto_range": False,
        "_composite": _composite,
    }

    class _AppProxy:
        """Live proxy that reads/writes shared state dict."""

        def __getattr__(self, name):
            if name == 'renderer':
                return renderer
            if name in _state:
                return _state[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            _state[name] = value

        def _project_field(self, field_name):
            if field_name in _state["_vector_fields"] and _state["_vector_projection"] == "LOS":
                _state["_los_camera_fwd"] = camera.forward.copy()
            return resolve_field(field_name, _state["_vector_fields"], data,
                                 _state["_vector_projection"], camera.forward)

        def _compute_weights(self):
            if _state["_sd_field"] in _state["_vector_fields"] and _state["_vector_projection"] == "LOS":
                _state["_los_camera_fwd"] = camera.forward.copy()
            return compute_weights(
                _state["_sd_field"], _state.get("_sd_field2", "None"),
                _state.get("_sd_op", "*"), _state["_vector_fields"], data,
                _state["_vector_projection"], camera.forward)

        def _compute_slot(self, slot):
            """Compute weights and qty for a composite slot dict."""
            return compute_slot_fields(slot, _state["_vector_fields"], data, camera.forward)

        def _apply_render_mode(self, auto_range=True):
            nonlocal _render_mode, needs_auto_range, refine_budget, refine_saved_lod, refine_saved_budget
            _state["_los_camera_fwd"] = None
            refine_budget = 0
            refine_saved_lod = None
            refine_saved_budget = None

            mode = _state["_render_mode_name"]
            _state["_composite"] = (mode == "Composite")

            if mode == "Composite":
                _render_mode = RenderMode(name="Composite", weight_field="", qty_field="", resolve_mode=-1)
                # Pre-populate slot caches then auto-range
                if renderer._grid is not None:
                    for si in range(2):
                        _ensure_slot_sorted(si)
                    _auto_range_composite_slot(0, "Lightness")
                    _auto_range_composite_slot(1, "Color")
                return

            if mode in ("WeightedAverage", "WeightedVariance"):
                weights = self._project_field(_state["_sd_field"])
                qty = self._project_field(_state["_wa_data_field"])
                if mode == "WeightedVariance":
                    _render_mode = RenderMode.weighted_variance(_state["_wa_data_field"], _state["_sd_field"])
                    renderer.resolve_mode = 2
                else:
                    _render_mode = RenderMode.mass_weighted_average(_state["_wa_data_field"], _state["_sd_field"])
                    renderer.resolve_mode = 1
            else:
                weights = self._compute_weights()
                qty = None
                _render_mode = RenderMode.surface_density(_state["_sd_field"])
                renderer.resolve_mode = 0

            renderer.update_weights(weights, qty)
            if gpu_compute is not None and renderer._grid is not None:
                gpu_compute.upload_weights(renderer._grid)
                _uses_los = (_state["_vector_projection"] == "LOS" and
                             _state.get("_wa_data_field", "") in _state["_vector_fields"])
                if _uses_los:
                    gpu_compute.upload_vector_field(renderer._grid,
                                                   _state["_wa_data_field"], data)
            if auto_range:
                needs_auto_range = True

        def _set_sd_field(self, value):
            _state["_sd_field"] = value
            self._apply_render_mode()

        def _set_colormap(self, name):
            from .colormaps import colormap_to_texture_data as _cmap_data
            renderer.set_colormap(_cmap_data(name))
            _state["_cmap_idx"] = AVAILABLE_COLORMAPS.index(name) if name in AVAILABLE_COLORMAPS else 0

    app_proxy = _AppProxy()

    def _cursor_to_fb(win):
        """Convert GLFW cursor position (window coords) to framebuffer pixel coords."""
        cx, cy = glfw.get_cursor_pos(win)
        ww, wh = glfw.get_window_size(win)
        fw, fh = glfw.get_framebuffer_size(win)
        return cx * fw / max(ww, 1), cy * fh / max(wh, 1)

    def mouse_button_callback(win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            x, y = _cursor_to_fb(win)
            if user_menu.on_click(x, y, app_proxy):
                return
            if overlay.enabled and overlay.on_click(x, y, renderer):
                return
        camera.on_mouse_button(button, action)

    def cursor_callback(win, x, y):
        camera.on_cursor(x, y)

    def scroll_callback(win, xoffset, yoffset):
        if user_menu.on_scroll(yoffset):
            return
        camera.on_scroll(yoffset)

    def char_callback(win, codepoint):
        user_menu.on_char(codepoint, app_proxy)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_char_callback(window, char_callback)

    needs_auto_range = True
    ui_hidden = False
    # Main loop state
    last_time = time.perf_counter()
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0

    # Progressive refinement / auto-LOD state
    was_moving = False
    last_cull_time = 0.0
    user_lod = renderer.lod_pixels
    user_budget = renderer.max_render_particles
    smooth_frame_ms = 0.0
    smooth_fps_ema = 0.0
    last_lod_adjust = 0.0
    pid_prev_error = 0.0
    pid_integral = 0.0
    refine_budget = 0
    refine_saved_lod = None
    refine_saved_budget = None

    # Pre-sort and cache slot weight arrays (done once per slot config change)
    _slot_sorted = [None, None]  # (slot_id, sorted_mass, sorted_qty) per slot

    def _ensure_slot_sorted(slot_idx):
        """Ensure sorted mass/qty arrays are cached for this slot. Returns (mass, qty).

        For LOS vector qty fields, qty is a placeholder — GPU projection handles it.
        Only the weight (mass) array needs to be sorted on CPU.
        """
        sl = _state["_slot"][slot_idx]
        slot_id = (sl["weight"], sl.get("weight2", "None"), sl.get("op", "*"),
                   sl["mode"], sl["data"], sl.get("proj", "LOS"))
        cached = _slot_sorted[slot_idx]
        if cached is not None and cached[0] == slot_id:
            return cached[1], cached[2]

        # Only compute the weight field — skip qty for LOS (GPU handles it)
        is_los_qty = (sl.get("proj") == "LOS" and
                      sl["mode"] in ("WeightedAverage", "WeightedVariance") and
                      sl["data"] in _state["_vector_fields"])

        # Compute weight using shared helpers
        proj = sl.get("proj", "LOS")
        vf = _state["_vector_fields"]
        w = resolve_field(sl["weight"], vf, data, proj, camera.forward)
        w2_name = sl.get("weight2", "None")
        if w2_name != "None":
            w2 = resolve_field(w2_name, vf, data, proj, camera.forward)
            w = combine_fields(w, w2, sl.get("op", "*"))

        so = renderer._grid.sort_order
        sm = w[so].astype(np.float32)

        if is_los_qty:
            sq = sm  # placeholder — GPU LOS projection will fill qty buffer
        elif sl["mode"] in ("WeightedAverage", "WeightedVariance"):
            q = resolve_field(sl["data"], vf, data, proj, camera.forward)
            sq = q[so].astype(np.float32)
        else:
            sq = sm

        _slot_sorted[slot_idx] = (slot_id, sm, sq)
        return sm, sq

    def _render_composite_frame(fb_w, fb_h):
        """Render two fields into separate FBOs and composite them."""
        renderer._ensure_fbo(fb_w, fb_h, which=1)
        renderer._ensure_fbo(fb_w, fb_h, which=2)

        gpu_ready_now = gpu_compute is not None and getattr(gpu_compute, '_upload_ready', False)
        accum_sets = [renderer._accum_textures, renderer._accum_textures2]

        for i in range(2):
            sl = _state["_slot"][i]

            cache_ready = _slot_sorted[i] is not None
            if gpu_ready_now and renderer._grid is not None and cache_ready:
                sorted_mass, sorted_qty = _slot_sorted[i][1], _slot_sorted[i][2]
                slot_id = _slot_sorted[i][0]

                # Upload to persistent per-slot GPU buffers (no-op if unchanged)
                gpu_compute.upload_slot_data(i, slot_id, sorted_mass, sorted_qty)

                # GPU-to-GPU copy from slot buffers to active particle buffers (~1ms)
                gpu_compute.activate_slot(i)

                # For LOS vector fields: overwrite qty with GPU projection
                is_los_vec = (sl.get("proj") == "LOS" and
                              sl["mode"] in ("WeightedAverage", "WeightedVariance") and
                              sl["data"] in _state["_vector_fields"])
                if is_los_vec:
                    if not gpu_compute.has_los_field() or getattr(gpu_compute, '_los_field_name', '') != sl["data"]:
                        gpu_compute.upload_vector_field(renderer._grid, sl["data"], data)
                    gpu_compute.dispatch_los_project(camera)

                n_out, n_vis, summary_data = gpu_compute.dispatch_cull(
                    camera, renderer.max_render_particles,
                    lod_pixels=renderer.lod_pixels,
                    viewport_width=renderer._viewport_width,
                    summary_overlap=renderer.summary_overlap,
                )
                renderer.set_particle_buffers_from_gpu(gpu_compute.get_output_buffers(), n_out)
                s_pos, s_hsml, s_mass, s_qty, s_cov = summary_data
                if renderer.use_aniso_summaries and len(s_pos) > 0:
                    renderer._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
                else:
                    renderer._upload_aniso_summaries(
                        np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                        np.zeros(0, np.float32), np.zeros((0, 6), np.float32))
            else:
                w, q = app_proxy._compute_slot(sl)
                renderer.update_weights(w, q)
                renderer.update_visible(camera)

            renderer._write_camera_uniforms(camera, fb_w, fb_h)
            renderer._render_accum(camera, fb_w, fb_h, accum_sets[i])

        s0, s1 = _state["_slot"][0], _state["_slot"][1]
        renderer.render_composite(
            camera, fb_w, fb_h,
            s0["resolve"], s0["min"], s0["max"], s0["log"],
            s1["resolve"], s1["min"], s1["max"], s1["log"],
        )

    def do_cull():
        """Run cull via GPU compute (zero-copy) or CPU fallback."""
        if gpu_compute is not None and getattr(gpu_compute, '_upload_ready', False):
            n_out, n_vis, summary_data = gpu_compute.dispatch_cull(
                camera, renderer.max_render_particles,
                lod_pixels=renderer.lod_pixels,
                viewport_width=renderer._viewport_width,
                summary_overlap=renderer.summary_overlap,
            )
            # Zero-copy: point renderer at compute output buffers
            renderer.set_particle_buffers_from_gpu(gpu_compute.get_output_buffers(), n_out)
            # Upload summaries (few, still CPU)
            s_pos, s_hsml, s_mass, s_qty, s_cov = summary_data
            if renderer.use_aniso_summaries and len(s_pos) > 0:
                renderer._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
            else:
                renderer._upload_aniso_summaries(
                    np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                    np.zeros(0, np.float32), np.zeros((0, 6), np.float32))
        else:
            renderer.update_visible(camera)

    # --- Benchmark mode ---
    if benchmark is not None:
        _run_wgpu_benchmark(
            benchmark, window, canvas_context, device, renderer, camera, data,
            gpu_compute, _bg, grid_thread, do_cull, fb_w, fb_h,
            user_lod, user_budget, app_proxy,
        )
        renderer.release()
        data.close()
        glfw.destroy_window(window)
        return

    print("DataFlyer [wgpu] running. WASD=move, mouse=look, ESC=quit, R=auto-range.")

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        # FPS
        frame_count += 1
        if now - fps_time > 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now
            n_vis = renderer.n_particles + renderer.n_aniso
            n_tot = renderer.n_total
            glfw.set_window_title(
                window,
                f"DataFlyer [wgpu] | {fps:.0f} fps | {n_vis/1e6:.1f}M/{n_tot/1e6:.1f}M"
            )

        glfw.poll_events()

        # Pick up background grid build when complete
        if _bg["done"] and _bg["grid"] is not None and renderer._grid is None:
            renderer._grid = _bg["grid"]
            renderer.use_tree = True
            # Do one CPU cull to get initial visible data
            renderer.update_visible(camera)
            needs_auto_range = True
            _bg["grid"] = None  # don't repeat

        # GPU compute init (blocking upload once grid is available)
        gpu_ready = getattr(gpu_compute, '_upload_ready', False)
        # Only use GPU compute for datasets large enough to benefit (>10M particles)
        GPU_COMPUTE_THRESHOLD = 10_000_000
        if (not gpu_ready and gpu_compute is None and renderer._grid is not None
                and renderer.n_total >= GPU_COMPUTE_THRESHOLD):
            try:
                gpu_compute = GPUCompute(device)
                gpu_compute.upload_snapshot(renderer._grid)
                gpu_ready = True
                print("  GPU compute pipeline initialized (zero-copy)")
            except Exception as e:
                print(f"  GPU compute init failed: {e}")
                gpu_compute = None

        # Camera movement
        moved = camera.update(dt)

        # GPU-side LOS projection: update qty buffer when camera rotates
        if (gpu_compute is not None and gpu_ready and gpu_compute.has_los_field()
                and _state["_vector_projection"] == "LOS"):
            los_fwd = _state.get("_los_camera_fwd")
            if los_fwd is None or float(np.dot(los_fwd, camera.forward)) < 0.9998:
                gpu_compute.dispatch_los_project(camera)
                _state["_los_camera_fwd"] = camera.forward.copy()
        elif not moved:
            # CPU fallback for non-GPU LOS (when gpu_compute not ready)
            if is_los_stale(_state["_render_mode_name"], _state["_wa_data_field"],
                            _state["_sd_field"], _state.get("_sd_field2", "None"),
                            _vector_fields, _state["_vector_projection"],
                            _state.get("_los_camera_fwd"), camera.forward):
                app_proxy._apply_render_mode(auto_range=True)

        # --- Cull / progressive refinement / auto-LOD ---
        # Composite mode: progressive refinement via budget, cull happens in _render_composite_frame
        if _state["_composite"]:
            if moved:
                if refine_saved_lod is not None:
                    renderer.lod_pixels = user_lod
                    renderer.max_render_particles = user_budget
                    refine_saved_lod = None
                    refine_saved_budget = None
                refine_budget = 0
                if not was_moving and renderer.auto_lod:
                    renderer.lod_pixels = max(user_lod, 8)
                    renderer.max_render_particles = min(user_budget, 2_000_000)

                if not was_moving:
                    pid_integral = 0.0

                # Smoothed frame time (unbiased EMA of frame rate)
                import math
                if dt > 0:
                    tau = max(renderer.auto_lod_smooth, 0.01)
                    a = min(1.0, dt / tau)
                    fps_inst = 1.0 / dt
                    smooth_fps_ema = (1 - a) * smooth_fps_ema + a * fps_inst
                    smooth_frame_ms = 1000.0 / max(smooth_fps_ema, 0.01)
                # PID auto-LOD on log2(budget)
                if renderer.auto_lod and smooth_frame_ms > 0 and dt > 0:
                    target_ms = 1000.0 / max(renderer.target_fps, 1.0)
                    error = math.log2(max(smooth_frame_ms / target_ms, 0.01))
                    pid_integral += error * dt
                    pid_integral = max(-4.0, min(4.0, pid_integral))
                    derivative = (error - pid_prev_error) / dt
                    pid_prev_error = error
                    rate = renderer.pid_Kp * error + renderer.pid_Ki * pid_integral + renderer.pid_Kd * derivative
                    output = rate * dt
                    log2_budget = math.log2(max(renderer.max_render_particles, 1_000))
                    log2_budget -= output
                    log2_n = math.log2(max(renderer.n_total, 1))
                    log2_budget = max(math.log2(1_000), min(log2_n, log2_budget))
                    renderer.max_render_particles = max(1_000, min(
                        renderer.n_total, int(2 ** log2_budget)))
                    frac = renderer.max_render_particles / max(renderer.n_total, 1)
                    renderer.lod_pixels = max(1.0, min(256.0, 1.0 / max(frac, 0.004)))

            elif refine_budget < renderer.n_total:
                if was_moving:
                    smooth_frame_ms = 0.0
                    refine_saved_lod = renderer.lod_pixels
                    refine_saved_budget = renderer.max_render_particles
                    refine_budget = max(user_budget, 4_000_000)
                if refine_budget == 0:
                    refine_budget = max(user_budget, 4_000_000)
                    refine_saved_lod = renderer.lod_pixels
                    refine_saved_budget = renderer.max_render_particles
                refine_budget = min(refine_budget * 2, renderer.n_total)
                renderer.lod_pixels = 1
                renderer.max_render_particles = refine_budget
        elif moved:
            # Restore user base if coming out of refinement
            if refine_saved_lod is not None:
                renderer.lod_pixels = user_lod
                renderer.max_render_particles = user_budget
                refine_saved_lod = None
                refine_saved_budget = None
            refine_budget = 0

            if not was_moving and renderer.auto_lod:
                # Just started moving — low-quality cull for responsive first frame
                renderer.lod_pixels = max(user_lod, 4)
                renderer.max_render_particles = min(user_budget, 4_000_000)
                do_cull()
                last_cull_time = now

            if not was_moving:
                pid_integral = 0.0

            # Smoothed frame time (unbiased EMA of frame rate)
            import math
            if dt > 0:
                tau = max(renderer.auto_lod_smooth, 0.01)
                a = min(1.0, dt / tau)
                fps_inst = 1.0 / dt
                smooth_fps_ema = (1 - a) * smooth_fps_ema + a * fps_inst
                smooth_frame_ms = 1000.0 / max(smooth_fps_ema, 0.01)
            # PID auto-LOD on log2(budget)
            if renderer.auto_lod and smooth_frame_ms > 0 and dt > 0:
                target_ms = 1000.0 / max(renderer.target_fps, 1.0)
                error = math.log2(max(smooth_frame_ms / target_ms, 0.01))
                pid_integral += error * dt
                pid_integral = max(-4.0, min(4.0, pid_integral))
                derivative = (error - pid_prev_error) / dt
                pid_prev_error = error
                output = renderer.pid_Kp * error + renderer.pid_Ki * pid_integral + renderer.pid_Kd * derivative
                log2_budget = math.log2(max(renderer.max_render_particles, 1_000))
                log2_budget -= output
                log2_n = math.log2(max(renderer.n_total, 1))
                log2_budget = max(math.log2(1_000), min(log2_n, log2_budget))
                renderer.max_render_particles = max(1_000, min(
                    renderer.n_total, int(2 ** log2_budget)))
                frac = renderer.max_render_particles / max(renderer.n_total, 1)
                renderer.lod_pixels = max(1.0, min(256.0, 1.0 / max(frac, 0.004)))

            # Cull every frame while moving (PID controls budget to keep it fast)
            do_cull()
            last_cull_time = now

        elif refine_budget < renderer.n_total:
            # --- STOPPED: progressive refinement ---
            if was_moving:
                smooth_frame_ms = 0.0
                refine_saved_lod = renderer.lod_pixels
                refine_saved_budget = renderer.max_render_particles
                refine_budget = max(user_budget, 4_000_000)

            if refine_budget == 0:
                refine_budget = max(user_budget, 4_000_000)
                refine_saved_lod = renderer.lod_pixels
                refine_saved_budget = renderer.max_render_particles

            refine_budget = min(refine_budget * 2, renderer.n_total)
            renderer.lod_pixels = 1
            renderer.max_render_particles = refine_budget

            do_cull()

            # Restore settings (render uses data already uploaded)
            if refine_saved_lod is not None:
                renderer.lod_pixels = refine_saved_lod
                renderer.max_render_particles = refine_saved_budget

        was_moving = moved

        # Resize handling
        new_fb_w, new_fb_h = glfw.get_framebuffer_size(window)
        if (new_fb_w, new_fb_h) != (fb_w, fb_h):
            fb_w, fb_h = new_fb_w, new_fb_h
            canvas_context.set_physical_size(fb_w, fb_h)
            camera.aspect = fb_w / max(fb_h, 1)
            win_w, win_h = glfw.get_window_size(window)
            renderer._viewport_width = win_w  # LOD uses window size, not retina

        # Render
        try:
            if _state["_composite"]:
                _render_composite_frame(fb_w, fb_h)
            else:
                renderer.render(camera, fb_w, fb_h)
        except Exception as e:
            print(f"Render error: {e}")
            import traceback; traceback.print_exc()
            # Ensure we still present something (avoid black flash)
            try:
                renderer.render(camera, fb_w, fb_h)
            except Exception:
                pass

        # Auto-range on first frame
        if needs_auto_range and not _state["_composite"]:
            lo, hi = renderer.read_accum_range()
            renderer.qty_min = lo
            renderer.qty_max = hi
            print(f"Auto-range: {lo:.3g} .. {hi:.3g}")
            try:
                renderer.render(camera, fb_w, fb_h)
            except Exception:
                pass
            needs_auto_range = False

        # Update timing stats
        cull_s = getattr(renderer, '_last_cull_ms', 0) / 1000
        upload_s = getattr(renderer, '_last_upload_ms', 0) / 1000
        alpha = 0.2
        if cull_s > 0 or upload_s > 0:
            _timings["cull"] = _timings["cull"] * (1 - alpha) + cull_s * alpha
            _timings["upload"] = _timings["upload"] * (1 - alpha) + upload_s * alpha

        # Sync auto-range request from proxy
        if _state["_needs_auto_range"]:
            needs_auto_range = True
            _state["_needs_auto_range"] = False

        # Overlay rendering pass (on top of the resolved image)
        # Skip overlay + present on odd frames when skip_vsync is on
        _do_present = not renderer.skip_vsync or frame_count % 2 == 0
        if not ui_hidden and _do_present:
          try:
            current_tex = canvas_context.get_current_texture()
            screen_view = current_tex.create_view()

            overlay.set_framebuffer_size(fb_w, fb_h)
            user_menu.set_framebuffer_size(fb_w, fb_h)

            smooth_fps_val = smooth_fps_ema if smooth_fps_ema > 0 else fps
            # Only rebuild overlay texture at ~4Hz to avoid PIL cost every frame
            _overlay_age = getattr(overlay, '_last_update_time', 0)
            if now - _overlay_age > 0.25:
                overlay._last_update_time = now
                was_enabled = overlay.enabled
                overlay.enabled = True
                overlay.update(
                    renderer, camera, fps, _render_mode.name,
                    AVAILABLE_COLORMAPS[_state["_cmap_idx"]], _timings, _last_message,
                    smooth_fps=smooth_fps_val,
                )
                overlay.enabled = was_enabled
            user_menu.update(
                renderer,
                AVAILABLE_COLORMAPS[_state["_cmap_idx"]], AVAILABLE_COLORMAPS,
                sd_fields=_sd_fields, sd_field=_state["_sd_field"],
                sd_field2=_state["_sd_field2"], sd_op=_state["_sd_op"],
                sd_ops=_SD_OPS,
                render_modes=_RENDER_MODES, render_mode_name=_state["_render_mode_name"],
                wa_data_field=_state["_wa_data_field"],
                vector_fields=_vector_fields,
                vector_projection=_state["_vector_projection"],
                vector_projections=_VECTOR_PROJECTIONS,
                composite_slots=_state["_slot"] if _state["_composite"] else None,
            )

            encoder = device.create_command_encoder()
            rpass = encoder.begin_render_pass(color_attachments=[{
                "view": screen_view,
                "load_op": "load",
                "store_op": "store",
            }])
            user_menu.render_to_pass(rpass)
            if overlay.enabled:
                overlay.render_to_pass(rpass)
            rpass.end()
            device.queue.submit([encoder.finish()])
          except Exception:
            import traceback
            traceback.print_exc()

        if _do_present:
            canvas_context.present()

    # Cleanup
    renderer.release()
    data.close()
    glfw.destroy_window(window)


def _run_wgpu_benchmark(n_frames, window, canvas_context, device, renderer, camera,
                        data, gpu_compute, _bg, grid_thread, do_cull, fb_w, fb_h,
                        user_lod, user_budget, app_proxy):
    """Run the scripted benchmark using the wgpu backend."""
    import math

    # Wait for grid build
    print("  Waiting for spatial grid...")
    grid_thread.join()
    if _bg["grid"] is not None and renderer._grid is None:
        renderer._grid = _bg["grid"]
        renderer.use_tree = True
        renderer.update_visible(camera)
        _bg["grid"] = None

    # Init GPU compute if available
    from .gpu_compute import GPUCompute
    GPU_COMPUTE_THRESHOLD = 10_000_000
    if (gpu_compute is None and renderer._grid is not None
            and renderer.n_total >= GPU_COMPUTE_THRESHOLD):
        try:
            gpu_compute = GPUCompute(device)
            gpu_compute.upload_snapshot(renderer._grid)
            print("  GPU compute pipeline initialized")
        except Exception as e:
            print(f"  GPU compute init failed: {e}")

    boxsize = data.header.get("BoxSize", None)
    extent = float(boxsize) if boxsize else float(np.linalg.norm(
        data.positions.max(axis=0) - data.positions.min(axis=0)))
    center = np.mean([data.positions.min(axis=0), data.positions.max(axis=0)], axis=0)
    start_pos = camera.position.copy()
    fwd0 = camera.forward.copy()
    up0 = camera.up.copy()
    right0 = camera.right.copy()

    from .benchmark import build_benchmark_keyframes, slerp_vec, print_benchmark_results

    keyframes = build_benchmark_keyframes(camera, center, extent, n_frames)

    # Progressive refinement to full detail
    budget = max(4_000_000, renderer.max_render_particles)
    while budget < renderer.n_total:
        budget = min(budget * 2, renderer.n_total)
        renderer.lod_pixels = 1
        renderer.max_render_particles = budget
        if gpu_compute is not None and getattr(gpu_compute, '_upload_ready', False):
            gpu_compute.dispatch_cull(camera, budget, lod_pixels=1,
                                      viewport_width=renderer._viewport_width,
                                      summary_overlap=renderer.summary_overlap)
        else:
            renderer.update_visible(camera)
        try:
            renderer.render(camera, fb_w, fb_h)
            canvas_context.present()
        except Exception:
            pass
        glfw.poll_events()
        n_vis = renderer.n_particles + renderer.n_aniso
        print(f"  Refining: {n_vis:,} / {renderer.n_total:,}")
        if glfw.window_should_close(window):
            return

    # Auto-range at full quality
    try:
        renderer.render(camera, fb_w, fb_h)
        canvas_context.present()
    except Exception:
        pass
    lo, hi = renderer.read_accum_range()
    renderer.qty_min = lo
    renderer.qty_max = hi
    print(f"  Auto-range: {lo:.3g} .. {hi:.3g}")

    renderer.lod_pixels = user_lod
    renderer.max_render_particles = user_budget

    # PID state
    smooth_frame_ms = 0.0
    pid_integral = 0.0
    pid_prev_error = 0.0
    was_moving = False
    last_cull_time = 0.0
    refine_budget = 0

    TRANSITION_S = 0.5
    HOLD_S = 1.5

    frame_times = []
    cull_times = []
    n_vis_list = []
    phase_list = []
    kf_idx_list = []
    lod_list = []
    budget_list = []

    def render_frame():
        try:
            renderer.render(camera, fb_w, fb_h)
            canvas_context.present()
        except Exception:
            pass
        glfw.poll_events()

    def record(t_frame_ms, phase, ki):
        cull_times.append(getattr(renderer, '_last_cull_ms', 0) + getattr(renderer, '_last_upload_ms', 0))
        frame_times.append(t_frame_ms)
        n_vis_list.append(renderer.n_particles + renderer.n_aniso)
        phase_list.append(phase)
        kf_idx_list.append(ki)
        lod_list.append(renderer.lod_pixels)
        budget_list.append(renderer.max_render_particles)

    def pid_update(dt_s):
        nonlocal pid_integral, pid_prev_error
        if not renderer.auto_lod or dt_s <= 0:
            return
        target_ms = 1000.0 / max(renderer.target_fps, 1.0)
        error = math.log2(max(dt_s * 1000 / target_ms, 0.01))
        pid_integral += error * dt_s
        pid_integral = max(-4.0, min(4.0, pid_integral))
        derivative = (error - pid_prev_error) / dt_s
        pid_prev_error = error
        rate = renderer.pid_Kp * error + renderer.pid_Ki * pid_integral + renderer.pid_Kd * derivative
        output = rate * dt_s
        log2_budget = math.log2(max(renderer.max_render_particles, 1_000))
        log2_budget -= output
        log2_n = math.log2(max(renderer.n_total, 1))
        log2_budget = max(math.log2(1_000), min(log2_n, log2_budget))
        renderer.max_render_particles = max(1_000, min(renderer.n_total, int(2 ** log2_budget)))
        frac = renderer.max_render_particles / max(renderer.n_total, 1)
        renderer.lod_pixels = max(1.0, min(256.0, 1.0 / max(frac, 0.004)))

    prev_kf = keyframes[0]

    for ki, (pos, fwd, up, label) in enumerate(keyframes):
        if glfw.window_should_close(window):
            break
        p0, f0, u0, _ = prev_kf

        # Transition
        t_start = time.perf_counter()
        t_prev = t_start
        while not glfw.window_should_close(window):
            elapsed = time.perf_counter() - t_start
            if elapsed >= TRANSITION_S:
                break
            t = elapsed / TRANSITION_S
            camera.position = (p0 * (1 - t) + pos * t).astype(np.float32)
            camera._forward = slerp_vec(f0, fwd, t).astype(np.float32)
            camera._up = slerp_vec(u0, up, t).astype(np.float32)
            camera._dirty = True

            now = time.perf_counter()
            dt_s = now - t_prev
            if dt_s > 0:
                if not was_moving:
                    smooth_frame_ms = 0.0
                    pid_integral = 0.0
                elif smooth_frame_ms == 0.0:
                    smooth_frame_ms = dt_s * 1000
                else:
                    tau = max(renderer.auto_lod_smooth, 0.01)
                    a = min(dt_s / tau, 1.0)
                    smooth_frame_ms = (1 - a) * smooth_frame_ms + a * dt_s * 1000
                pid_update(dt_s)

            do_cull()
            render_frame()
            was_moving = True
            refine_budget = 0

            now2 = time.perf_counter()
            record((now2 - t_prev) * 1000, "transition", ki)
            t_prev = now2

        # Hold
        camera.position = pos.copy()
        camera._forward = fwd.copy()
        camera._up = up.copy()
        camera._dirty = True

        t_start = time.perf_counter()
        t_prev = t_start
        hold_refine_budget = max(user_budget, 4_000_000)
        while not glfw.window_should_close(window):
            if time.perf_counter() - t_start >= HOLD_S:
                break
            hold_refine_budget = min(hold_refine_budget * 2, renderer.n_total)
            renderer.lod_pixels = 1
            renderer.max_render_particles = hold_refine_budget
            do_cull()
            render_frame()
            renderer.lod_pixels = user_lod
            renderer.max_render_particles = user_budget
            was_moving = False

            now2 = time.perf_counter()
            record((now2 - t_prev) * 1000, "hold", ki)
            t_prev = now2

        prev_kf = (pos, fwd, up, label)

    # Print results
    frame_times = np.array(frame_times)
    cull_times = np.array(cull_times)

    print_benchmark_results(frame_times, cull_times, phase_list,
                            renderer.n_total, backend="wgpu")

    outfile = f"benchmark_wgpu_{int(time.time())}.npz"
    kf_labels = [kf[3] for kf in keyframes]
    np.savez(outfile,
             frame_time_ms=frame_times, cull_time_ms=cull_times,
             n_visible=np.array(n_vis_list), phase=np.array(phase_list),
             keyframe_idx=np.array(kf_idx_list),
             keyframe_labels=np.array(kf_labels),
             lod_pixels=np.array(lod_list),
             max_render_particles=np.array(budget_list),
             n_total=renderer.n_total, backend="wgpu")
    print(f"  Saved: {outfile} ({len(frame_times)} frames)")
    print("-----------------------------------\n")
