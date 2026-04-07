"""wgpu backend application: GLFW window + WGPURenderer + UI overlays."""

import time
import atexit
import numpy as np
import glfw

import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info

from .camera import Camera
from .data_manager import SnapshotData
from .wgpu_renderer import WGPURenderer, RenderMode
from .colormaps import colormap_to_texture_data, AVAILABLE_COLORMAPS
from .field_ops import (resolve_field, compute_weights, compute_slot_fields,
                        combine_fields, make_default_app_state,
                        is_los_stale, SD_OPS, RENDER_MODES, VECTOR_PROJECTIONS)


def run_wgpu_app(snapshot_path, width=1920, height=1080, fov=90.0,
                 fullscreen=False, screenshot=None):
    """Run the DataFlyer application with the wgpu backend.

    If `screenshot` is set to a path, the canvas loop runs just long
    enough for GPU init + auto-range to complete, takes a screenshot,
    and exits without entering the interactive loop.
    """
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
    if "timestamp-query" in adapter.features:
        req_features.add("timestamp-query")

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
    camera.auto_scale(data.positions, masses=data.get_field("Masses"), boxsize=boxsize)

    # Renderer
    renderer = WGPURenderer(device, canvas_context, present_format)
    renderer._viewport_width = width  # use window size for LOD, not retina framebuffer size

    # Colormap
    rgba = colormap_to_texture_data("magma")
    renderer.set_colormap(rgba)

    # Load particles. set_particles only stores arrays + n_total; the
    # actual GPU upload happens in the background-built grid + GPUCompute
    # init below.
    weights = data.get_field("Masses")
    renderer.set_particles(data.positions, data.hsml, weights)

    # Render + present an empty first frame immediately so the window
    # comes up responsive. The GPU upload follows in the canvas tick.
    try:
        renderer.render(camera, fb_w, fb_h)
        canvas_context.present()
        glfw.poll_events()
    except Exception:
        pass

    from .gpu_compute import GPUCompute
    gpu_compute = None
    glfw.set_window_title(window, "DataFlyer [wgpu] | Initializing...")

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
    _composite = _s["composite"]
    _slot = _s["slot"]

    # Stars
    if data.n_stars > 0:
        renderer.upload_stars(data.star_positions, data.star_masses,
                              luminosity=getattr(data, "star_luminosity", None))
        if data.n_particles > 0:
            renderer.set_extinction_gas(data.positions, data.masses, data.hsml)
        print(f"  {data.n_stars} star particles loaded")

    # Input callbacks
    def _auto_range_composite_slot(slot_idx, label):
        s = _state["_slot"][slot_idx]
        renderer.resolve_mode = s["resolve"]
        renderer.log_scale = s["log"]

        gpu_ready_now = gpu_compute is not None and getattr(gpu_compute, '_upload_ready', False)
        fb_w_, fb_h_ = glfw.get_framebuffer_size(window)

        if gpu_ready_now:
            # GPU subsample path: upload this slot's mass/qty into the
            # composite slot's per-chunk buffers, bind them as the active
            # slot, and let _render_accum dispatch the splat draws.
            sorted_mass, sorted_qty = _ensure_slot_sorted(slot_idx)
            slot_id = _slot_sorted[slot_idx][0]
            slot_chunks = gpu_compute.upload_subsample_slot(
                slot_idx, slot_id, sorted_mass, sorted_qty)
            renderer.set_subsample_slot_chunks(slot_idx, slot_chunks)
            renderer.set_active_subsample_slot(slot_idx)
            renderer.n_particles = min(renderer.n_total,
                                       renderer._subsample_max_per_frame)
        else:
            # CPU fallback (subsample mode keeps the renderer empty here;
            # the GPU path is required to actually see anything)
            w, q = app_proxy._compute_slot(s)
            renderer.update_weights(w, q)

        renderer._ensure_fbo(fb_w_, fb_h_, which=1)
        renderer._write_camera_uniforms(camera, fb_w_, fb_h_)
        renderer._render_accum(camera, fb_w_, fb_h_, renderer._accum_textures)
        # Lightness (slot 0): mass-weighted entropy, so the surface density
        # range is set by where the actual mass is. Color (slot 1): raw
        # entropy on field values, so the dynamic range is the full spread
        # of the field itself rather than wherever it happens to be dense.
        lo, hi = renderer.read_accum_range(mass_weighted=(slot_idx == 0))
        s["min"] = lo
        s["max"] = hi
        print(f"Auto-range {label}: {lo:.3g} .. {hi:.3g}")

    def _adjust_range(rend, factor):
        mid = (rend.qty_min + rend.qty_max) / 2
        half = (rend.qty_max - rend.qty_min) / 2 * factor
        rend.qty_min = mid - half
        rend.qty_max = mid + half

    def key_callback(win, key, scancode, action, mods):
        nonlocal _cmap_idx, needs_auto_range, ui_hidden
        nonlocal subsample_cap_ceiling, last_subsample_cap
        nonlocal dirty, ui_dirty, idle_streak
        if action in (glfw.PRESS, glfw.REPEAT):
            idle_streak = 0
        # Keystrokes routed into the user_menu (text-field editing) only
        # mutate UI state — no scene re-render needed. The main loop's
        # ui-only dirty path will redraw the panel without re-running
        # particle accumulation.
        if user_menu.on_key(key, action):
            ui_dirty = True
            return
        # Any other PRESS may mutate scene state (R, L, ',', '.', mode
        # toggles, etc.), so force a full re-render.
        if action == glfw.PRESS:
            dirty = True
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
            elif key == glfw.KEY_PERIOD:
                # Raise the auto-LOD subsample-cap ceiling. The cap
                # itself adapts within [floor, ceiling]; this knob is
                # the ceiling. SUBSAMPLE_CAP_HARD_LIMIT is the hard
                # upper bound for safe GPU buffer / dispatch limits.
                subsample_cap_ceiling = min(
                    SUBSAMPLE_CAP_HARD_LIMIT,
                    int(subsample_cap_ceiling * 2))
                print(f"Subsample cap ceiling: "
                      f"{subsample_cap_ceiling/1e6:.1f}M")
            elif key == glfw.KEY_COMMA:
                subsample_cap_ceiling = max(
                    500_000, subsample_cap_ceiling // 2)
                # Clamp the running cap so the change is felt
                # immediately, not just at the next motion frame.
                last_subsample_cap = min(last_subsample_cap,
                                         subsample_cap_ceiling)
                renderer.set_subsample_max_per_frame(
                    min(renderer._subsample_max_per_frame,
                        subsample_cap_ceiling))
                print(f"Subsample cap ceiling: "
                      f"{subsample_cap_ceiling/1e6:.1f}M")
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
                _take_screenshot()
            elif key == glfw.KEY_F1 or key == glfw.KEY_BACKSLASH:
                overlay.enabled = not overlay.enabled
            elif key == glfw.KEY_TAB:
                ui_hidden = not ui_hidden
            elif key == glfw.KEY_B:
                renderer.cycle_star_band(-1 if (mods & glfw.MOD_SHIFT) else 1)
            elif key == glfw.KEY_O:
                renderer.toggle_star_extinction()
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
        "_los_camera_pos": None,
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
                _state["_los_camera_pos"] = camera.position.copy()
            return resolve_field(field_name, _state["_vector_fields"], data,
                                 _state["_vector_projection"], camera.forward,
                                 camera_position=camera.position)

        def _compute_weights(self):
            if _state["_sd_field"] in _state["_vector_fields"] and _state["_vector_projection"] == "LOS":
                _state["_los_camera_pos"] = camera.position.copy()
            return compute_weights(
                _state["_sd_field"], _state.get("_sd_field2", "None"),
                _state.get("_sd_op", "*"), _state["_vector_fields"], data,
                _state["_vector_projection"], camera.forward,
                camera_position=camera.position)

        def _compute_slot(self, slot):
            """Compute weights and qty for a composite slot dict."""
            return compute_slot_fields(slot, _state["_vector_fields"], data,
                                       camera.forward,
                                       camera_position=camera.position)

        def _apply_render_mode(self, auto_range=True):
            nonlocal _render_mode, needs_auto_range
            _state["_los_camera_pos"] = None

            mode = _state["_render_mode_name"]
            _state["_composite"] = (mode == "Composite")

            if mode == "Composite":
                _render_mode = RenderMode(name="Composite", weight_field="", qty_field="", resolve_mode=-1)
                # Pre-populate slot caches then auto-range
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
            print(f"  [diag] mode={mode} n_total={renderer.n_total} "
                  f"n_particles={renderer.n_particles} "
                  f"resolve_mode={renderer.resolve_mode} "
                  f"chunks={renderer._subsample_chunks is not None} "
                  f"max_per_frame={renderer._subsample_max_per_frame}", flush=True)
            if gpu_compute is not None:
                _t = time.perf_counter()
                gpu_compute.upload_weights(
                    renderer._all_mass, renderer._all_qty)
                print(f"  [diag] upload_weights: {(time.perf_counter()-_t)*1000:.0f}ms", flush=True)
                # GPU LOS projection isn't wired into the subsample path
                # (would need to update qty across every per-chunk buffer
                # per camera rotation). The CPU _project_field above
                # already produced the right LOS-projected qty for the
                # current camera orientation; subsequent rotations will
                # render with stale qty until the user re-applies the
                # render mode (the canvas tick re-fires _apply_render_mode
                # when LOS staleness is detected).
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
        nonlocal dirty, idle_streak
        # Any mouse click can mutate UI-internal state (open a dropdown,
        # focus a text field, drag a slider) that isn't reflected in
        # `state_sig`. Force at least the next frame to render so the
        # idle short-circuit doesn't swallow the visible response.
        dirty = True
        idle_streak = 0
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
        nonlocal dirty, ui_dirty, idle_streak
        idle_streak = 0
        if user_menu.on_scroll(yoffset):
            ui_dirty = True
            return
        # Camera zoom: full re-render
        dirty = True
        camera.on_scroll(yoffset)

    def char_callback(win, codepoint):
        nonlocal ui_dirty, idle_streak
        # Text-field editing only mutates UI state — no scene re-render.
        ui_dirty = True
        idle_streak = 0
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
    # Wall-clock duration of the previous renderer.render() call. This is
    # the only signal that reflects real GPU latency (encode/submit are
    # async). Used by the subsample cap-growth gate so we don't overshoot
    # into a backlogged swapchain.
    last_render_ms = 0.0
    smooth_render_ms = 0.0  # EMA of last_render_ms
    fps = 0.0

    # Idle-frame short-circuit. We re-render only when something visible
    # could have changed. Tracked: camera movement, framebuffer resize,
    # _state mutations (qty range, log, render mode, colormap, composite
    # slot params), the per-frame subsample cap (progressive refinement),
    # current LOD, and pending auto-range. When none changed we skip
    # render+overlay+submit entirely and sleep briefly.
    # `dirty` forces a full scene re-render (accum + resolve). `ui_dirty`
    # forces a UI-only redraw (resolve over the existing accum textures
    # + overlay/menu repaint), which is cheap enough to be responsive
    # even on 100M+ particle snapshots while text is being typed.
    dirty = True
    ui_dirty = False
    prev_state_sig = None
    prev_cap = None
    prev_n_particles = None
    prev_fb_size = (0, 0)
    # Number of consecutive idle frames seen. We only start sleeping
    # after a few in a row so a transient `moved=False` frame in the
    # middle of a slow rotation doesn't introduce a visible stutter.
    idle_streak = 0
    IDLE_STREAK_THRESHOLD = 6
    # Tracks the camera position from the previous frame so we can
    # detect translations independently of rotations. Per-particle LOS
    # is invariant under rotation, so only translations should freeze
    # the slot LOS cache.
    prev_camera_pos = camera.position.copy()

    # Cap-based auto-LOD state
    was_moving = False
    # Per-frame instance cap for the subsample splat path. Initialized
    # to 4M; afterwards adapted to the highest cap that sustained target FPS.
    last_subsample_cap = 4_000_000
    # Auto-LOD ceiling for the per-frame subsample cap. The cap adapts
    # within [floor, ceiling]; the , / . keys halve / double this.
    # SUBSAMPLE_CAP_HARD_LIMIT is the upper bound (safe GPU dispatch /
    # buffer headroom on commodity hardware).
    SUBSAMPLE_CAP_HARD_LIMIT = 200_000_000
    subsample_cap_ceiling = 16_000_000
    # Don't grow the cap before the user has moved the camera at least
    # once — startup has no FPS history to validate growth against.
    has_moved_ever = False
    # Frames to wait before firing the post-init auto-range. Set when
    # GPU subsample chunks are first wired up.
    pending_auto_range_frames = 0
    smooth_fps_ema = 0.0

    # Pre-sort and cache slot weight arrays (done once per slot config change)
    _slot_sorted = [None, None]  # (slot_id, sorted_mass, sorted_qty) per slot

    def _ensure_slot_sorted(slot_idx, skip_los_recompute=False):
        """Ensure sorted mass/qty arrays are cached for this slot. Returns (mass, qty).

        For LOS vector qty fields, qty was historically a placeholder — the
        GPU LOS projection path filled it in. The subsample-mode pipeline
        doesn't run GPU LOS projection, so we always compute qty on the CPU
        here. Per-particle LOS is invariant under camera rotation but goes
        stale on translation; the canvas loop calls _apply_render_mode again
        when translation moves the camera past the LOS staleness threshold.

        `skip_los_recompute=True` makes the cache key ignore the current
        camera position for LOS vector slots (any cached entry, regardless
        of where it was computed, will be reused). The per-frame composite
        render path uses this while the camera is translating to avoid a
        ~1.5 s/frame CPU reprojection on every micro-translation — the
        stale projection is acceptable while the user is moving; once they
        stop, the gate opens and the next frame refreshes once.
        """
        sl = _state["_slot"][slot_idx]
        # When the slot uses an LOS-projected vector field, the cached
        # projection depends on the camera position — include a quantized
        # position in the cache key so translation invalidates the entry.
        is_los_vec = (sl.get("proj") == "LOS"
                      and (sl["weight"] in _state["_vector_fields"]
                           or sl.get("weight2", "None") in _state["_vector_fields"]
                           or (sl["mode"] in ("WeightedAverage", "WeightedVariance")
                               and sl["data"] in _state["_vector_fields"])))
        if is_los_vec and not skip_los_recompute:
            # Per-particle LOS depends only on camera *position*
            # (rotation leaves the camera→particle direction unchanged).
            pos_key = tuple(int(round(float(c) * 1000)) for c in camera.position)
        else:
            pos_key = None
        slot_id = (sl["weight"], sl.get("weight2", "None"), sl.get("op", "*"),
                   sl["mode"], sl["data"], sl.get("proj", "LOS"), pos_key)
        cached = _slot_sorted[slot_idx]
        if cached is not None and cached[0] == slot_id:
            return cached[1], cached[2]

        # Compute weight (and optional second weight field) on the CPU.
        # The GPU upload path is in raw particle order, so no sort is
        # applied here.
        proj = sl.get("proj", "LOS")
        vf = _state["_vector_fields"]
        w = resolve_field(sl["weight"], vf, data, proj, camera.forward,
                          camera_position=camera.position)
        w2_name = sl.get("weight2", "None")
        if w2_name != "None":
            w2 = resolve_field(w2_name, vf, data, proj, camera.forward,
                               camera_position=camera.position)
            w = combine_fields(w, w2, sl.get("op", "*"))
        sm = w.astype(np.float32)

        if sl["mode"] in ("WeightedAverage", "WeightedVariance"):
            q = resolve_field(sl["data"], vf, data, proj, camera.forward,
                              camera_position=camera.position)
            sq = q.astype(np.float32)
        else:
            sq = sm

        _slot_sorted[slot_idx] = (slot_id, sm, sq)
        return sm, sq

    def _render_composite_frame(fb_w, fb_h, encoder=None, screen_view=None,
                                skip_los_recompute=False, skip_accum=False):
        """Render two fields into separate FBOs and composite them.

        If `encoder` and `screen_view` are provided, all sub-passes (two
        accum passes + one composite resolve) are appended to the shared
        encoder and the caller submits. Otherwise the legacy path runs
        each sub-call with its own encoder+submit.

        `skip_los_recompute=True` (passed from the main loop while the
        camera is translating) makes the LOS vector slot cache ignore the
        current camera position, so we don't pay a ~1.5 s/frame CPU
        reprojection on every micro-translation. The cached LOS field is
        slightly stale during motion; on stop the gate opens and the next
        frame recomputes once with the fresh position.
        """
        renderer._ensure_fbo(fb_w, fb_h, which=1)
        renderer._ensure_fbo(fb_w, fb_h, which=2)

        gpu_ready_now = gpu_compute is not None and getattr(gpu_compute, '_upload_ready', False)
        accum_sets = [renderer._accum_textures, renderer._accum_textures2]

        if not skip_accum:
            for i in range(2):
                sl = _state["_slot"][i]
                if gpu_ready_now:
                    # Subsample path: ensure the slot's sorted mass/qty is
                    # cached for the *current* camera direction (the cache
                    # key includes a quantized fwd for LOS slots), then
                    # upload to the slot's per-chunk buffers and render.
                    sorted_mass, sorted_qty = _ensure_slot_sorted(
                        i, skip_los_recompute=skip_los_recompute)
                    slot_id = _slot_sorted[i][0]
                    slot_chunks = gpu_compute.upload_subsample_slot(
                        i, slot_id, sorted_mass, sorted_qty)
                    renderer.set_subsample_slot_chunks(i, slot_chunks)
                    renderer.set_active_subsample_slot(i)
                    renderer.n_particles = min(renderer.n_total,
                                               renderer._subsample_max_per_frame)
                else:
                    w, q = app_proxy._compute_slot(sl)
                    renderer.update_weights(w, q)

                renderer._write_camera_uniforms(camera, fb_w, fb_h)
                renderer._render_accum(camera, fb_w, fb_h, accum_sets[i],
                                       encoder=encoder)
            # Reset active slot so non-composite passes use the default path.
            renderer.set_active_subsample_slot(None)

        s0, s1 = _state["_slot"][0], _state["_slot"][1]
        renderer.render_composite(
            camera, fb_w, fb_h,
            s0["resolve"], s0["min"], s0["max"], s0["log"],
            s1["resolve"], s1["min"], s1["max"], s1["log"],
            encoder=encoder, screen_view=screen_view,
        )

    def _take_screenshot(path=None):
        """Render the current view at full framebuffer resolution into
        an offscreen texture and save it. Filename defaults to
        dataflyer_<timestamp>.png in the cwd.
        """
        import os
        if path is None:
            path = f"dataflyer_{int(time.time())}.png"
        fb_w_, fb_h_ = glfw.get_framebuffer_size(window)
        if _state["_composite"]:
            s0 = _state["_slot"][0]
            s1 = _state["_slot"][1]
            comp = (s0["resolve"], s0["min"], s0["max"], s0["log"],
                    s1["resolve"], s1["min"], s1["max"], s1["log"])
            renderer.screenshot(path, fb_w_, fb_h_, camera, composite_args=comp)
        else:
            renderer.screenshot(path, fb_w_, fb_h_, camera)
        return os.path.abspath(path)

    print("DataFlyer [wgpu] running. WASD=move, mouse=look, ESC=quit, R=auto-range, P=screenshot.")

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt_raw = now - last_time
        last_time = now
        # Clamp dt so a single slow frame can't catapult the camera by
        # 10x the normal step. The auto-LOD PID can briefly oscillate
        # frame time on cost-curve discontinuities (e.g. multigrid level
        # boundaries), and an unclamped dt turns that into visible
        # position lurches. 1/15 s matches the default target FPS.
        dt = min(dt_raw, 1.0 / 15.0)

        # FPS
        frame_count += 1
        if now - fps_time > 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now
            n_vis = renderer.n_particles
            n_tot = renderer.n_total
            init_msg = " | Initializing GPU..." if gpu_compute is None else ""
            glfw.set_window_title(
                window,
                f"DataFlyer [wgpu] | {fps:.0f} fps | {n_vis/1e6:.1f}M/{n_tot/1e6:.1f}M{init_msg}"
            )

        glfw.poll_events()

        # GPU subsample pipeline init (one-time upload, first frame).
        gpu_ready = getattr(gpu_compute, '_upload_ready', False)
        if not gpu_ready and gpu_compute is None:
            try:
                gpu_compute = GPUCompute(device)
                gpu_compute.upload_subsample_only(
                    renderer._all_pos, renderer._all_hsml,
                    renderer._all_mass, renderer._all_qty)
                renderer.set_subsample_chunks(
                    gpu_compute.get_chunk_bufs(),
                    world_offset=gpu_compute.get_pos_offset())
                # Prime renderer.n_particles so the first render() call
                # doesn't early-out before the user has moved.
                renderer.n_particles = min(renderer.n_total,
                                           renderer._subsample_max_per_frame)
                print("  GPU subsample pipeline initialized")
                # Defer the post-init auto-range a few frames so the
                # GPU has actually rendered something into the FBO
                # before we read it back. The pre-init auto-range that
                # fires from set_particles ran against an empty FBO
                # so its qty range is garbage.
                pending_auto_range_frames = 3
                needs_auto_range = False
                gpu_ready = True
            except Exception as e:
                print(f"  GPU compute init failed: {e}")
                gpu_compute = None

        # Camera movement
        moved = camera.update(dt)
        # Per-particle LOS depends on camera position only, not on
        # orientation, so detect translation separately from rotation.
        translated = bool(np.any(camera.position != prev_camera_pos))
        prev_camera_pos = camera.position.copy()

        # Subsample per-frame instance cap. Init to 4M (used only at the
        # very first frame); after that the cap learns the highest value
        # that sustained the target FPS during motion ("last sustainable")
        # and resets to that on motion start. Stopped frames grow the cap
        # for progressive refinement, but only after the user has moved
        # at least once and only when the previous frame met its target
        # (so we never over-extend with no FPS history to validate).
        target_ms = 1000.0 / max(renderer.target_fps, 1.0)
        # Smoothed render-time EMA, fed by last_render_ms (not dt). dt
        # measures how often Python loops; render-time measures how
        # long the GPU actually takes to retire each frame.
        if last_render_ms > 0:
            smooth_render_ms = (0.85 * smooth_render_ms + 0.15 * last_render_ms
                                if smooth_render_ms > 0 else last_render_ms)
        if moved:
            has_moved_ever = True
            if not was_moving:
                # Resume from the last cap proven sustainable for
                # interactive motion. The on_submitted_work_done_sync
                # call after each stationary frame guarantees no
                # refinement frame is in flight, so this cap takes
                # effect on the very next render with zero drain wait.
                renderer.set_subsample_max_per_frame(last_subsample_cap)
                # Reset the smoothed render-time EMA so the just-elapsed
                # multi-second stationary frame doesn't poison the
                # cap-grow gate for the first few motion frames.
                smooth_render_ms = 0.0
            if smooth_render_ms > 0:
                cur_cap = renderer._subsample_max_per_frame
                if smooth_render_ms < target_ms * 0.9:
                    last_subsample_cap = min(subsample_cap_ceiling, int(cur_cap * 1.1) + 1)
                    renderer.set_subsample_max_per_frame(last_subsample_cap)
                elif smooth_render_ms > target_ms * 1.2:
                    last_subsample_cap = max(1, int(cur_cap * 0.85))
                    renderer.set_subsample_max_per_frame(last_subsample_cap)
                else:
                    last_subsample_cap = cur_cap
        elif has_moved_ever:
            # Stopped after at least one motion: refine progressively.
            # The growth gate uses last_render_ms (wall-clock duration of
            # the previous renderer.render() call), which captures real
            # GPU latency via get_current_texture's swapchain block.
            # Python's `dt` is a useless signal here — Metal queues
            # encode/submit asynchronously and Python keeps racing
            # ahead, so dt stays small even when the GPU is buried.
            # Stationary refinement is allowed to take long-ish frames —
            # the user is sitting still waiting for full detail. The cap
            # only stops growing if a frame takes more than ~750 ms,
            # which is the threshold past which input feels "stuck".
            REFINE_FRAME_MS = 750.0
            if 0 < last_render_ms <= REFINE_FRAME_MS:
                # Stationary refinement target: actual full detail. The
                # user-controlled `subsample_cap_ceiling` only governs
                # the cap during motion (where high FPS matters); when
                # sitting still we're allowed to push up to the hard
                # GPU limit and the snapshot's particle count.
                stationary_cap = min(SUBSAMPLE_CAP_HARD_LIMIT, renderer.n_total)
                cur = renderer._subsample_max_per_frame
                if cur < stationary_cap:
                    renderer.set_subsample_max_per_frame(
                        min(int(cur * 1.4) + 1, stationary_cap))
                # Promote the refined cap to last_subsample_cap only if
                # the frame we just rendered was above target FPS. That
                # way `last_subsample_cap` always reflects "the largest
                # cap proven sustainable for interactive motion", and
                # resuming motion restores that quality without ever
                # putting the user above their target frame budget.
                if 0 < last_render_ms <= target_ms and cur > last_subsample_cap:
                    last_subsample_cap = min(cur, subsample_cap_ceiling)

        # Per-particle LOS depends only on camera position, so the
        # cached projection is invalid only when the camera has
        # *translated* (rotation is free). Skip the recompute while
        # translation is in progress — it's too expensive (full CPU LOS
        # pass over every particle) to run mid-flight on large
        # snapshots; once the user stops translating, the gate opens
        # and the next frame refreshes once.
        if not translated:
            if is_los_stale(_state["_render_mode_name"], _state["_wa_data_field"],
                            _state["_sd_field"], _state.get("_sd_field2", "None"),
                            _vector_fields, _state["_vector_projection"],
                            _state.get("_los_camera_pos"), camera.position):
                app_proxy._apply_render_mode(auto_range=True)
                dirty = True

        # Keep n_particles in sync with the live cap so the overlay /
        # dirty signature reflect what's actually being drawn.
        if renderer._subsample_chunks is not None:
            renderer.n_particles = min(renderer.n_total,
                                       renderer._subsample_max_per_frame)

        was_moving = moved

        # Resize handling
        new_fb_w, new_fb_h = glfw.get_framebuffer_size(window)
        if (new_fb_w, new_fb_h) != (fb_w, fb_h):
            fb_w, fb_h = new_fb_w, new_fb_h
            canvas_context.set_physical_size(fb_w, fb_h)
            camera.aspect = fb_w / max(fb_h, 1)
            win_w, _ = glfw.get_window_size(window)
            renderer._viewport_width = win_w  # LOD uses window size, not retina

        # --- Idle-frame short-circuit ---
        # Build a signature of all state that affects the rendered image.
        # If nothing has changed since the last presented frame and we
        # aren't carrying any pending work, skip the entire render path
        # (no swapchain acquire, no encode, no submit) and sleep briefly
        # so the loop doesn't burn CPU.
        try:
            _slot0 = _state["_slot"][0]
            _slot1 = _state["_slot"][1]
            state_sig = (
                _state.get("_composite"),
                _slot0.get("min"), _slot0.get("max"),
                _slot0.get("log"), _slot0.get("resolve"),
                _slot1.get("min"), _slot1.get("max"),
                _slot1.get("log"), _slot1.get("resolve"),
                _state.get("_render_mode_name"),
                _state.get("_cmap_idx"),
                _state.get("_wa_data_field"),
                _state.get("_sd_field"),
                _state.get("_sd_field2"),
                _state.get("_sd_op"),
                _state.get("_vector_projection"),
                renderer.qty_min, renderer.qty_max, renderer.log_scale,
                renderer.hsml_scale,
                ui_hidden,
            )
        except Exception:
            state_sig = None  # be conservative: render
        cap_now = renderer._subsample_max_per_frame
        n_particles_now = renderer.n_particles
        fb_size_now = (fb_w, fb_h)

        scene_dirty = (
            dirty
            or moved
            or needs_auto_range
            or pending_auto_range_frames > 0
            or state_sig is None
            or state_sig != prev_state_sig
            or cap_now != prev_cap
            or n_particles_now != prev_n_particles
            or fb_size_now != prev_fb_size
        )
        # ui_dirty alone is enough to require a frame, but lets us skip
        # the (very expensive on big snapshots) accum pass.
        frame_dirty = scene_dirty or ui_dirty
        skip_accum_this_frame = ui_dirty and not scene_dirty

        if not frame_dirty:
            # Require a few consecutive idle frames before we actually
            # start sleeping. A single `moved=False` tick in the middle
            # of a slow rotation (e.g. between mouse-delta events) would
            # otherwise insert a 5ms sleep and cause visible stutter,
            # most noticeably in LOS variance / composite modes whose
            # CPU-cached projection already updates coarsely.
            idle_streak += 1
            if idle_streak >= IDLE_STREAK_THRESHOLD:
                # Don't increment frame_count for slept frames so the
                # FPS counter isn't inflated by no-op iterations.
                frame_count = max(frame_count - 1, 0)
                time.sleep(0.005)
                continue
            # Otherwise fall through and render anyway (cheap, since
            # nothing changed the GPU work is essentially the same as
            # the last frame).
        else:
            idle_streak = 0

        # Render. Time the call wall-clock — most of this is
        # get_current_texture blocking on the swapchain present, which
        # is the only signal that reflects real GPU latency (Python's
        # encode + submit are async w.r.t. the GPU).
        _t_render = time.perf_counter()

        # Single encoder + single submit per frame: acquire the swapchain
        # texture once, append accum/resolve and the overlay/UI passes to
        # the same command encoder, submit at the end. This halves the
        # per-frame wgpu FFI roundtrips vs. the old "render submits, then
        # overlay submits" path.
        _frame_encoder = None
        _frame_screen_view = None
        try:
            current_tex = canvas_context.get_current_texture()
            _frame_screen_view = current_tex.create_view()
            _frame_encoder = device.create_command_encoder()
        except Exception:
            import traceback; traceback.print_exc()

        try:
            if _frame_encoder is not None:
                if _state["_composite"]:
                    _render_composite_frame(
                        fb_w, fb_h,
                        encoder=_frame_encoder, screen_view=_frame_screen_view,
                        skip_los_recompute=translated,
                        skip_accum=skip_accum_this_frame)
                else:
                    renderer.render(
                        camera, fb_w, fb_h,
                        encoder=_frame_encoder, screen_view=_frame_screen_view,
                        skip_accum=skip_accum_this_frame)
        except Exception as e:
            print(f"Render error: {e}")
            import traceback; traceback.print_exc()
            # Ensure we still present something (avoid black flash). The
            # legacy fallback path uses its own encoder/submit.
            try:
                renderer.render(camera, fb_w, fb_h)
            except Exception:
                pass
        last_render_ms = (time.perf_counter() - _t_render) * 1000.0
        # Auto-range on first frame
        if pending_auto_range_frames > 0:
            pending_auto_range_frames -= 1
            if pending_auto_range_frames == 0:
                needs_auto_range = True
        if needs_auto_range and not _state["_composite"]:
            lo, hi = renderer.read_accum_range()
            renderer.qty_min = lo
            renderer.qty_max = hi
            print(f"Auto-range: {lo:.3g} .. {hi:.3g}")
            # The pre-built frame encoder already encoded a resolve with
            # the *stale* qty range. Drop it so we don't submit stale
            # pixels on top of the freshly-auto-ranged image. The legacy
            # render() call below uses its own encoder and submits
            # immediately; the overlay block will fall back to its own
            # encoder since _frame_encoder is now None.
            _frame_encoder = None
            _frame_screen_view = None
            try:
                renderer.render(camera, fb_w, fb_h)
            except Exception:
                pass
            needs_auto_range = False
            # Headless screenshot mode: take the shot now that init,
            # auto-range, and one full render have completed, then exit.
            if screenshot is not None:
                _take_screenshot(screenshot)
                glfw.set_window_should_close(window, True)

        # Update timing stats
        cull_s = getattr(renderer, '_last_cull_ms', 0) / 1000
        upload_s = getattr(renderer, '_last_upload_ms', 0) / 1000
        render_s = getattr(renderer, '_last_render_ms', 0) / 1000
        alpha = 0.2
        if cull_s > 0 or upload_s > 0 or render_s > 0:
            _timings["cull"] = _timings["cull"] * (1 - alpha) + cull_s * alpha
            _timings["upload"] = _timings["upload"] * (1 - alpha) + upload_s * alpha
            _timings["render"] = _timings["render"] * (1 - alpha) + render_s * alpha

        # Sync auto-range request from proxy
        if _state["_needs_auto_range"]:
            needs_auto_range = True
            _state["_needs_auto_range"] = False

        # Overlay rendering pass (on top of the resolved image)
        # Skip overlay + present on odd frames when skip_vsync is on
        _do_present = not renderer.skip_vsync or frame_count % 2 == 0
        # If the auto-range branch dropped the prebuilt frame encoder,
        # we need a fresh one (with its own swapchain view) just for the
        # overlay pass.
        if not ui_hidden and _do_present and _frame_encoder is None:
            try:
                current_tex = canvas_context.get_current_texture()
                _frame_screen_view = current_tex.create_view()
                _frame_encoder = device.create_command_encoder()
            except Exception:
                import traceback; traceback.print_exc()
        if not ui_hidden and _do_present and _frame_encoder is not None:
          try:
            screen_view = _frame_screen_view

            overlay.set_framebuffer_size(fb_w, fb_h)
            user_menu.set_framebuffer_size(fb_w, fb_h)

            smooth_fps_val = smooth_fps_ema if smooth_fps_ema > 0 else fps
            # Only rebuild overlay texture at ~4Hz to avoid PIL cost every frame
            _overlay_age = getattr(overlay, '_last_update_time', 0)
            if now - _overlay_age > 0.25:
                overlay._last_update_time = now
                was_enabled = overlay.enabled
                overlay.enabled = True
                init_status = ("Initializing: uploading to GPU..."
                               if gpu_compute is None else "")
                overlay_message = init_status if init_status else _last_message
                overlay.update(
                    renderer, camera, fps, _render_mode.name,
                    AVAILABLE_COLORMAPS[_state["_cmap_idx"]], _timings, overlay_message,
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

            rpass = _frame_encoder.begin_render_pass(color_attachments=[{
                "view": screen_view,
                "load_op": "load",
                "store_op": "store",
            }])
            user_menu.render_to_pass(rpass)
            if overlay.enabled:
                overlay.render_to_pass(rpass)
            rpass.end()
          except Exception:
            import traceback
            traceback.print_exc()

        # Submit the single per-frame encoder (covers accum + resolve +
        # optional overlay/UI). Then present.
        if _frame_encoder is not None:
            try:
                device.queue.submit([_frame_encoder.finish()])
            except Exception:
                import traceback
                traceback.print_exc()

        if _do_present:
            canvas_context.present()

        # On stationary refinement frames, block here until the GPU has
        # actually retired the work we just submitted. The point: when
        # the user resumes motion, there must be NO in-flight refinement
        # frame queued behind us. If we let refinement keep racing
        # ahead, the next motion-start get_current_texture() blocks for
        # hundreds of ms waiting for a stale 100M-particle accum to
        # drain. Synchronizing here pushes that wait into the stationary
        # period (where it doesn't matter — the user is sitting still)
        # and makes resume instantaneous.
        if (not moved) and has_moved_ever and _frame_encoder is not None:
            try:
                device.queue.on_submitted_work_done_sync()
            except Exception:
                pass

        # Frame committed: snapshot state for the next idle-check.
        prev_state_sig = state_sig
        prev_cap = cap_now
        prev_n_particles = n_particles_now
        prev_fb_size = fb_size_now
        dirty = False
        ui_dirty = False

    # Cleanup
    renderer.release()
    data.close()
    glfw.destroy_window(window)


