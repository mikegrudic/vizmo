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


def run_wgpu_app(snapshot_path, width=1920, height=1080, fov=90.0,
                 screenshot=None, benchmark=None):
    """Run the DataFlyer application with the wgpu backend."""
    import os
    snapshot_path = os.path.abspath(snapshot_path)

    # Initialize GLFW without OpenGL (wgpu uses Vulkan/Metal)
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    atexit.register(glfw.terminate)

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # No OpenGL context
    glfw.window_hint(glfw.RESIZABLE, True)

    window = glfw.create_window(width, height, "DataFlyer [wgpu]", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create wgpu device and canvas context
    present_info = get_glfw_present_info(window)
    canvas_context = wgpu.gpu.get_canvas_context(present_info)

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    print(f"  wgpu adapter: {adapter.info.get('description', 'unknown')}")

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
    renderer._viewport_width = fb_w

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
    from .spatial_grid import SpatialGrid
    import threading

    gpu_compute = None
    _bg = {"grid": None, "done": False}  # mutable container for thread result

    def _build_grid_bg():
        try:
            _bg["grid"] = SpatialGrid(data.positions, weights, data.hsml, weights)
            print("  Spatial grid built (background)")
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
    _sd_fields = data.available_fields()
    _vector_fields = data.available_vector_fields()
    _sd_field = "Masses"
    _sd_field2 = "None"
    _sd_op = "*"
    _render_mode_name = "SurfaceDensity"
    _wa_data_field = "Masses"
    _RENDER_MODES = ["SurfaceDensity", "WeightedAverage", "WeightedVariance"]
    _SD_OPS = ["*", "+", "-", "/", "min", "max"]
    _VECTOR_PROJECTIONS = ["LOS", "|v|", "|v|^2"]
    _vector_projection = "LOS"
    _los_camera_fwd = None

    # Stars
    if data.n_stars > 0:
        renderer.upload_stars(data.star_positions, data.star_masses)
        print(f"  {data.n_stars} star particles loaded")

    # Input callbacks
    def key_callback(win, key, scancode, action, mods):
        if user_menu.on_key(key, action):
            return
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_R:
                lo, hi = renderer.read_accum_range()
                renderer.qty_min = lo
                renderer.qty_max = hi
                print(f"Auto-range: {lo:.3g} .. {hi:.3g}")
            elif key == glfw.KEY_F1 or key == glfw.KEY_BACKSLASH:
                overlay.enabled = not overlay.enabled
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
        "_slot": [],
        "_needs_auto_range": False,
        "_composite": False,
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
            import numpy as np
            if field_name in _state["_vector_fields"]:
                vec = data.get_vector_field(field_name)
                proj = _state["_vector_projection"]
                if proj == "LOS":
                    fwd = camera.forward.copy()
                    _state["_los_camera_fwd"] = fwd.copy()
                    return (vec @ fwd).astype(np.float32)
                elif proj == "|v|":
                    return np.linalg.norm(vec, axis=1).astype(np.float32)
                else:
                    return (vec * vec).sum(axis=1).astype(np.float32)
            return data.get_field(field_name)

        def _compute_weights(self):
            import numpy as np
            w = self._project_field(_state["_sd_field"])
            f2 = _state.get("_sd_field2", "None")
            if f2 != "None":
                w2 = self._project_field(f2)
                op = _state.get("_sd_op", "*")
                if op == "*": w = w * w2
                elif op == "+": w = w + w2
                elif op == "-": w = w - w2
                elif op == "/": w = w / np.maximum(np.abs(w2), 1e-30) * np.sign(w2)
                elif op == "min": w = np.minimum(w, w2)
                elif op == "max": w = np.maximum(w, w2)
            return w

        def _apply_render_mode(self, auto_range=True):
            nonlocal _render_mode, needs_auto_range, refine_budget, refine_saved_lod, refine_saved_budget
            _state["_los_camera_fwd"] = None
            refine_budget = 0
            refine_saved_lod = None
            refine_saved_budget = None

            mode = _state["_render_mode_name"]
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
                # Upload vector field for GPU-side LOS projection
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

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    needs_auto_range = True
    _gpu_upload_iter = None  # chunked upload iterator

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
    last_lod_adjust = 0.0
    refine_budget = 0
    refine_saved_lod = None
    refine_saved_budget = None

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
        if not gpu_ready and gpu_compute is None and renderer._grid is not None:
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
            def _is_los_stale():
                if _state["_render_mode_name"] not in ("WeightedAverage", "WeightedVariance"):
                    sd_f = _state["_sd_field"]
                    sd_f2 = _state.get("_sd_field2", "None")
                    if sd_f not in _vector_fields and (sd_f2 == "None" or sd_f2 not in _vector_fields):
                        return False
                if _state["_vector_projection"] != "LOS":
                    return False
                if _state["_los_camera_fwd"] is None:
                    return True
                dot = float(np.dot(_state["_los_camera_fwd"], camera.forward))
                return dot < 0.9998
            if _is_los_stale():
                app_proxy._apply_render_mode(auto_range=True)

        # --- Cull / progressive refinement / auto-LOD ---
        # Skip all culling while GPU compute is still uploading — just render the existing frame
        if not gpu_ready and moved:
            pass  # keep rendering last frame — no cull, no lag
        elif moved:
            # Restore user base if coming out of refinement
            if refine_saved_lod is not None:
                renderer.lod_pixels = user_lod
                renderer.max_render_particles = user_budget
                refine_saved_lod = None
                refine_saved_budget = None
            refine_budget = 0

            if not was_moving:
                # Just started moving — low-quality cull for responsive first frame
                renderer.lod_pixels = max(user_lod, 4)
                renderer.max_render_particles = min(user_budget, 4_000_000)
                do_cull()
                last_cull_time = now

            # Smoothed frame time (EMA)
            if dt > 0:
                if not was_moving:
                    smooth_frame_ms = 0.0
                elif smooth_frame_ms == 0.0:
                    smooth_frame_ms = dt * 1000
                else:
                    tau = max(renderer.auto_lod_smooth, 0.01)
                    a = min(dt / tau, 1.0)
                    smooth_frame_ms = (1 - a) * smooth_frame_ms + a * dt * 1000

            # Auto-LOD: 2x adjust, at most once per tau
            tau = max(renderer.auto_lod_smooth, 0.01)
            if renderer.auto_lod and smooth_frame_ms > 0 and (now - last_lod_adjust) >= tau:
                target_ms = 1000.0 / max(renderer.target_fps, 1.0)
                if smooth_frame_ms > target_ms:
                    renderer.lod_pixels = min(256, renderer.lod_pixels * 2)
                    renderer.max_render_particles = max(100_000, renderer.max_render_particles // 2)
                    last_lod_adjust = now
                elif smooth_frame_ms < target_ms * 0.5:
                    renderer.lod_pixels = max(1, renderer.lod_pixels // 2)
                    renderer.max_render_particles = min(renderer.n_total, renderer.max_render_particles * 2)
                    last_lod_adjust = now

            # Throttled cull
            cull_dt = renderer.cull_interval
            if now - last_cull_time >= cull_dt:
                do_cull()
                last_cull_time = now

        elif refine_budget < renderer.n_total and getattr(gpu_compute, '_upload_ready', False):
            # --- STOPPED: progressive refinement (only when GPU compute is ready) ---
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

        # Render
        try:
            renderer.render(camera, fb_w, fb_h)
        except Exception as e:
            print(f"Render error: {e}")
            continue

        # Auto-range on first frame
        if needs_auto_range:
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

        # Overlay rendering pass (on top of the resolved image)
        try:
            current_tex = canvas_context.get_current_texture()
            screen_view = current_tex.create_view()

            overlay.set_framebuffer_size(fb_w, fb_h)
            user_menu.set_framebuffer_size(fb_w, fb_h)

            # Sync auto-range request from proxy
            if _state["_needs_auto_range"]:
                needs_auto_range = True
                _state["_needs_auto_range"] = False

            smooth_fps_val = 1000.0 / max(smooth_frame_ms, 1.0) if smooth_frame_ms > 0 else 0.0
            # Always build the overlay texture so toggle works instantly
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
            traceback.print_exc()  # print once, then continue

        canvas_context.present()

    # Cleanup
    renderer.release()
    data.close()
    glfw.destroy_window(window)
