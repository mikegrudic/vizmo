"""Main application: glfw window, event loop, orchestration."""

import sys
import time
import glfw
import moderngl

from .camera import Camera
from .data_manager import SnapshotData, find_snapshots
from .renderer import SplatRenderer, RenderMode
from .colormaps import create_colormap_texture_safe, AVAILABLE_COLORMAPS
from .overlay import DevOverlay, UserMenu
from .field_ops import resolve_field, compute_weights, compute_slot_fields


class DataFlyerApp:
    def __init__(self, snapshot_path, width=1920, height=1080, fov=90.0, fullscreen=False):
        self.width = width
        self.height = height

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 0)

        monitor = glfw.get_primary_monitor() if fullscreen else None
        self.window = glfw.create_window(width, height, "DataFlyer", monitor, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        # Create moderngl context
        self.ctx = moderngl.create_context()

        # Discover all snapshots in the same directory
        import os
        snapshot_path = os.path.abspath(snapshot_path)
        self._snap_list = find_snapshots(snapshot_path)
        self._snap_idx = self._snap_list.index(snapshot_path) if snapshot_path in self._snap_list else 0
        if len(self._snap_list) > 1:
            print(f"Found {len(self._snap_list)} snapshots in directory")

        # Load data
        print(f"Loading {snapshot_path}...")
        self.data = SnapshotData(snapshot_path)
        print(f"  {self.data.n_particles:,} gas particles loaded")

        # Camera
        self.camera = Camera(fov=fov, aspect=width / height)
        boxsize = self.data.header.get("BoxSize", None)
        self.camera.auto_scale(self.data.positions, boxsize=boxsize)

        # Renderer
        self.renderer = SplatRenderer(self.ctx)
        fb_w, _ = glfw.get_framebuffer_size(self.window)
        self.renderer._viewport_width = fb_w

        # Dev overlay
        self.overlay = DevOverlay(self.ctx)
        self.user_menu = UserMenu(self.ctx)
        self._last_message = ""
        self._timings = {"cull": 0, "upload": 0, "render": 0}

        # Colormaps
        self._colormap_textures = {}
        self._cmap_idx = 0
        self._set_colormap(AVAILABLE_COLORMAPS[0])

        # Weight fields and render mode
        self._sd_fields = self.data.available_fields()
        self._vector_fields = self.data.available_vector_fields()
        self._sd_field = "Masses"
        self._sd_field2 = "None"  # optional second field
        self._sd_op = "*"  # operation between field1 and field2
        self._SD_OPS = ["*", "+", "-", "/", "min", "max"]
        self._RENDER_MODES = ["SurfaceDensity", "WeightedAverage", "WeightedVariance", "Composite"]
        self._render_mode_name = "SurfaceDensity"
        self._wa_data_field = "Masses"  # data field for WeightedAverage
        self._VECTOR_PROJECTIONS = ["LOS", "|v|", "|v|^2"]
        self._vector_projection = "LOS"  # active projection for vector data fields
        self._los_camera_fwd = None  # camera forward at last LOS recompute
        self._render_mode = RenderMode.surface_density("Masses")

        # Composite mode: slot 0 = lightness, slot 1 = color
        # Defaults match CrunchSnaps CoolMap: lightness=SurfaceDensity(Masses),
        # color=WeightedVariance(Velocities LOS) i.e. 1D velocity dispersion
        self._composite = False
        _has_vel = "Velocities" in self._vector_fields
        self._slot = [
            {  # slot 0: lightness — surface density of Masses (log)
                "mode": "SurfaceDensity", "weight": "Masses", "data": "Masses",
                "weight2": "None", "op": "*", "proj": "LOS",
                "min": -1.0, "max": 3.0, "log": 1, "resolve": 0,
            },
            {  # slot 1: color — 1D velocity dispersion (log)
                "mode": "WeightedVariance" if _has_vel else "SurfaceDensity",
                "weight": "Masses",
                "data": "Velocities" if _has_vel else "Masses",
                "weight2": "None", "op": "*",
                "proj": "LOS",
                "min": -1.0, "max": 3.0, "log": 1,
                "resolve": 2 if _has_vel else 0,
            },
        ]

        # Store particles and upload (with culling for large datasets)
        weights = self._compute_weights()
        self.renderer.resolve_mode = self._render_mode.resolve_mode
        self.renderer.set_particles(
            self.data.positions, self.data.hsml, weights,
        )
        self.renderer.update_visible(self.camera)
        print(f"  Rendering {self.renderer.n_particles:,} / {self.renderer.n_total:,} visible particles")

        # Upload star particles
        if self.data.n_stars > 0:
            self.renderer.upload_stars(self.data.star_positions, self.data.star_masses)
            print(f"  {self.data.n_stars} star particles loaded")

        # Auto-range will be set after the first render pass (needs framebuffer data)
        self._needs_auto_range = True

        # Input callbacks
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_callback)
        glfw.set_char_callback(self.window, self._char_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)

        # User's base LOD settings (set by [/] and </> keys)
        self._user_lod = self.renderer.lod_pixels
        self._user_budget = self.renderer.max_render_particles

        # Timing
        self._last_time = time.perf_counter()
        self._frame_count = 0
        self._fps_time = time.perf_counter()
        self._fps = 0.0
        # Async GPU timer query (read previous frame's result to avoid stalling)
        self._gpu_query = self.ctx.query(time=True)
        self._gpu_render_ns = 0  # last completed query result in nanoseconds

        # Key action table
        self._key_actions = self._build_key_actions()

    def _load_snapshot(self, path):
        """Load a new snapshot, keeping the current camera position."""
        import os
        self.data.close()
        print(f"Loading {os.path.basename(path)}...")
        self.data = SnapshotData(path)
        print(f"  {self.data.n_particles:,} gas, {self.data.n_stars} stars, t={self.data.time:.4g}")

        self._sd_fields = self.data.available_fields()
        self._vector_fields = self.data.available_vector_fields()
        if self._sd_field not in self._sd_fields:
            self._sd_field = "Masses"
        if self._sd_field2 != "None" and self._sd_field2 not in self._sd_fields:
            self._sd_field2 = "None"
        if self._wa_data_field not in self._sd_fields and self._wa_data_field not in self._vector_fields:
            self._wa_data_field = "Masses"
        self._los_camera_fwd = None

        weights = self.data.get_field(self._sd_field) if self._render_mode_name == "WeightedAverage" else self._compute_weights()
        qty = self._compute_data_field() if self._render_mode_name == "WeightedAverage" else None
        self.renderer.set_particles(
            self.data.positions, self.data.hsml, weights, qty,
        )
        self.renderer.resolve_mode = self._render_mode.resolve_mode
        self.renderer.update_visible(self.camera)

        if self.data.n_stars > 0:
            self.renderer.upload_stars(self.data.star_positions, self.data.star_masses)

        self._needs_auto_range = True

    def _msg(self, text):
        """Print and capture for dev overlay."""
        print(text)
        self._last_message = text

    def _range_str(self):
        lo, hi = self.renderer.qty_min, self.renderer.qty_max
        if self.renderer.log_scale:
            return f"10^{lo:.2f} .. 10^{hi:.2f}"
        return f"{lo:.3g} .. {hi:.3g}"

    def _auto_range_from_framebuffer(self):
        """Auto-range from actual rendered pixel values (reads back GPU texture)."""
        lo, hi = self.renderer.read_accum_range()
        self.renderer.qty_min = lo
        self.renderer.qty_max = hi
        self._msg(f"Auto-range: {self._range_str()}")

    def _set_colormap(self, name):
        if name not in self._colormap_textures:
            self._colormap_textures[name] = create_colormap_texture_safe(self.ctx, name)
        self.renderer.colormap_tex = self._colormap_textures[name]

    def _project_field(self, field_name):
        """Load a field and project to scalar if it's a vector field."""
        if field_name in self._vector_fields and self._vector_projection == "LOS":
            self._los_camera_fwd = self.camera.forward.copy()
        return resolve_field(field_name, self._vector_fields, self.data,
                             self._vector_projection, self.camera.forward)

    def _compute_weights(self):
        """Compute the final weight array from field1, op, and field2."""
        if self._sd_field in self._vector_fields and self._vector_projection == "LOS":
            self._los_camera_fwd = self.camera.forward.copy()
        return compute_weights(self._sd_field, self._sd_field2, self._sd_op,
                               self._vector_fields, self.data,
                               self._vector_projection, self.camera.forward)

    def _uses_vector_field(self):
        """Check if any active field is a vector field."""
        if self._render_mode_name in ("WeightedAverage", "WeightedVariance"):
            if self._wa_data_field in self._vector_fields:
                return True
        if self._sd_field in self._vector_fields:
            return True
        if self._sd_field2 != "None" and self._sd_field2 in self._vector_fields:
            return True
        return False

    def _is_los_stale(self):
        """Check if the LOS projection needs recomputing due to camera rotation."""
        if not self._uses_vector_field():
            return False
        if self._vector_projection != "LOS":
            return False
        if self._los_camera_fwd is None:
            return True
        import numpy as np
        # Recompute if camera direction changed by more than ~1 degree
        dot = float(np.dot(self._los_camera_fwd, self.camera.forward))
        return dot < 0.9998  # cos(1 degree) ~ 0.99985

    def _apply_render_mode(self, auto_range=True):
        """Rebuild render mode from current settings and re-weight the grid."""
        # Reset progressive refinement so it restarts from the new data
        self._refine_budget = 0
        self._refine_saved_lod = None
        self._refine_saved_budget = None
        # Force fresh LOS projection from current camera direction
        self._los_camera_fwd = None
        self._composite = (self._render_mode_name == "Composite")
        if self._composite:
            self._render_mode = RenderMode(
                name="Composite", weight_field="", qty_field="", resolve_mode=-1)
            # Only auto-range if slot limits are at defaults (never been set)
            needs_range = auto_range and any(
                s["min"] == -1.0 and s["max"] == 3.0 for s in self._slot
            )
            if needs_range:
                for i, s in enumerate(self._slot):
                    w, q = self._compute_slot(s)
                    self.renderer.resolve_mode = s["resolve"]
                    self.renderer.log_scale = s["log"]
                    self.renderer.update_weights(w, q)
                    self.renderer.update_visible(self.camera)
                    fb_w, fb_h = self.width, self.height
                    self.renderer._ensure_fbo(fb_w, fb_h, which=1)
                    self.renderer._render_accum(self.camera, fb_w, fb_h, self.renderer._accum_fbo)
                    lo, hi = self.renderer.read_accum_range()
                    s["min"] = lo
                    s["max"] = hi
                    self._msg(f"Slot {i} range: {lo:.3g} .. {hi:.3g}")
            return

        if self._render_mode_name in ("WeightedAverage", "WeightedVariance"):
            weights = self._project_field(self._sd_field)
            qty = self._project_field(self._wa_data_field)
            label = self._wa_data_field
            if self._wa_data_field in self._vector_fields:
                label = f"{self._wa_data_field} ({self._vector_projection})"
            if self._render_mode_name == "WeightedVariance":
                self._render_mode = RenderMode.weighted_variance(label, self._sd_field)
            else:
                self._render_mode = RenderMode.mass_weighted_average(label, self._sd_field)
        else:
            weights = self._compute_weights()
            qty = None
            self._render_mode = RenderMode.surface_density(self._sd_field)
        self.renderer.resolve_mode = self._render_mode.resolve_mode
        self.renderer.update_weights(weights, qty)
        self.renderer.update_visible(self.camera)
        if auto_range:
            self._needs_auto_range = True

    def _compute_slot(self, slot):
        """Compute weights and qty arrays for a composite slot dict."""
        return compute_slot_fields(slot, self._vector_fields, self.data, self.camera.forward)

    def _render_composite_frame(self, fb_width, fb_height):
        """Render two fields into separate FBOs and composite.

        Calls update_weights + update_visible per slot per frame.
        Cost: ~2x update_weights + 2x cull. Correct because tree moments
        are fully rebuilt for each slot.
        """
        r = self.renderer
        r._ensure_fbo(fb_width, fb_height, which=1)
        r._ensure_fbo(fb_width, fb_height, which=2)

        fbos = [r._accum_fbo, r._accum_fbo2]
        for i in range(2):
            w, q = self._compute_slot(self._slot[i])
            r.update_weights(w, q)
            r.update_visible(self.camera)
            r._render_accum(self.camera, fb_width, fb_height, fbos[i])

        s0, s1 = self._slot[0], self._slot[1]
        r.render_composite(
            self.camera, fb_width, fb_height,
            s0["resolve"], s0["min"], s0["max"], s0["log"],
            s1["resolve"], s1["min"], s1["max"], s1["log"],
        )

    def _set_sd_field(self, field_name):
        """Change the primary surface density weight field."""
        self._sd_field = field_name
        self._apply_render_mode()

    def _cycle_colormap(self, direction=1):
        self._cmap_idx = (self._cmap_idx + direction) % len(AVAILABLE_COLORMAPS)
        name = AVAILABLE_COLORMAPS[self._cmap_idx]
        self._msg(f"Colormap: {name}")
        self._set_colormap(name)

    def _contract_range(self):
        mid = (self.renderer.qty_min + self.renderer.qty_max) / 2
        half = (self.renderer.qty_max - self.renderer.qty_min) / 2 * 0.8
        self.renderer.qty_min = mid - half
        self.renderer.qty_max = mid + half
        self._msg(f"Range: {self._range_str()}")

    def _expand_range(self):
        mid = (self.renderer.qty_min + self.renderer.qty_max) / 2
        half = (self.renderer.qty_max - self.renderer.qty_min) / 2 * 1.25
        self.renderer.qty_min = mid - half
        self.renderer.qty_max = mid + half
        self._msg(f"Range: {self._range_str()}")

    def _increase_lod(self):
        self.renderer.lod_pixels = max(1, self.renderer.lod_pixels // 2)
        self._user_lod = self.renderer.lod_pixels
        self._msg(f"LOD: {self.renderer.lod_pixels}px (more detail)")
        self._refinement_level = 0
        self.renderer.update_visible(self.camera)

    def _decrease_lod(self):
        self.renderer.lod_pixels = min(256, self.renderer.lod_pixels * 2)
        self._user_lod = self.renderer.lod_pixels
        self._msg(f"LOD: {self.renderer.lod_pixels}px (faster)")
        self._refinement_level = 0
        self.renderer.update_visible(self.camera)

    def _increase_budget(self):
        self.renderer.max_render_particles = min(
            self.renderer.n_total, int(self.renderer.max_render_particles * 2))
        self._user_budget = self.renderer.max_render_particles
        self._msg(f"Max particles: {self.renderer.max_render_particles/1e6:.1f}M")
        self._refinement_level = 0
        self.renderer.update_visible(self.camera)

    def _decrease_budget(self):
        self.renderer.max_render_particles = max(
            1_000, self.renderer.max_render_particles // 2)
        self._user_budget = self.renderer.max_render_particles
        self._msg(f"Max particles: {self.renderer.max_render_particles/1e6:.1f}M")
        self._refinement_level = 0
        self.renderer.update_visible(self.camera)

    def _toggle_log_scale(self):
        self.renderer.log_scale = 1 - self.renderer.log_scale
        self._msg(f"Scale: {'log' if self.renderer.log_scale else 'linear'}")
        self._needs_auto_range = True

    def _toggle_overlay(self):
        self.overlay.enabled = not self.overlay.enabled
        self._msg(f"Dev overlay: {'on' if self.overlay.enabled else 'off'}")

    def _toggle_ui(self):
        self._ui_hidden = not getattr(self, '_ui_hidden', False)

    def _toggle_importance_sampling(self):
        self.renderer.use_importance_sampling = not self.renderer.use_importance_sampling
        self._msg(f"Importance sampling: {'on' if self.renderer.use_importance_sampling else 'off'}")
        self.renderer.update_visible(self.camera)

    def _cycle_kernel(self):
        kernels = self.renderer.KERNELS
        idx = (kernels.index(self.renderer.kernel) + 1) % len(kernels)
        self.renderer.kernel = kernels[idx]
        self._msg(f"Kernel: {self.renderer.kernel}")

    def _toggle_tree(self):
        self.renderer.use_tree = not self.renderer.use_tree
        self._msg(f"Tree: {'on' if self.renderer.use_tree else 'off'}")
        if self.renderer.use_tree and self.renderer._grid is None:
            weights = self._compute_weights()
            self.renderer.set_particles(self.data.positions, self.data.hsml, weights)
        elif not self.renderer.use_tree:
            self.renderer._grid = None
        self.renderer.update_visible(self.camera)

    def _next_snapshot(self):
        if len(self._snap_list) > 1:
            self._snap_idx = min(self._snap_idx + 1, len(self._snap_list) - 1)
            self._load_snapshot(self._snap_list[self._snap_idx])

    def _prev_snapshot(self):
        if len(self._snap_list) > 1:
            self._snap_idx = max(self._snap_idx - 1, 0)
            self._load_snapshot(self._snap_list[self._snap_idx])

    def _build_key_actions(self):
        """Build the key -> (handler, description) action table."""
        return {
            glfw.KEY_C:             (self._cycle_colormap,          "Next colormap"),
            glfw.KEY_RIGHT:         (self._next_snapshot,           "Next snapshot"),
            glfw.KEY_LEFT:          (self._prev_snapshot,           "Previous snapshot"),
            glfw.KEY_R:             (self._auto_range_from_framebuffer, "Auto-range"),
            glfw.KEY_EQUAL:         (self._contract_range,          "Contract range"),
            glfw.KEY_KP_ADD:        (self._contract_range,          None),
            glfw.KEY_MINUS:         (self._expand_range,            "Expand range"),
            glfw.KEY_KP_SUBTRACT:   (self._expand_range,            None),
            glfw.KEY_RIGHT_BRACKET: (self._increase_lod,            "More LOD detail"),
            glfw.KEY_LEFT_BRACKET:  (self._decrease_lod,            "Less LOD detail"),
            glfw.KEY_PERIOD:        (self._increase_budget,         "More particles"),
            glfw.KEY_COMMA:         (self._decrease_budget,         "Fewer particles"),
            glfw.KEY_L:             (self._toggle_log_scale,        "Toggle log/linear"),
            glfw.KEY_P:             (self._screenshot,              "Screenshot"),
            glfw.KEY_BACKSLASH:     (self._toggle_overlay,          "Toggle dev overlay"),
            glfw.KEY_TAB:           (self._toggle_ui,               "Hide/show all UI"),
            glfw.KEY_I:             (self._toggle_importance_sampling, "Toggle importance sampling"),
            glfw.KEY_K:             (self._cycle_kernel,            "Cycle kernel"),
            glfw.KEY_T:             (self._toggle_tree,             "Toggle tree"),
            glfw.KEY_H:             (self._print_help,              "Print help"),
        }

    def _key_callback(self, window, key, scancode, action, mods):
        if self.user_menu.on_key(key, action):
            return
        if action != glfw.PRESS:
            self.camera.on_key(key, action)
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return
        handler = self._key_actions.get(key)
        if handler is not None:
            handler[0]()
        else:
            self.camera.on_key(key, action)

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            win_w, win_h = glfw.get_window_size(window)
            fx = xpos * fb_w / win_w
            fy = ypos * fb_h / win_h
            # User menu (always active)
            if self.user_menu.on_click(fx, fy, self):
                return
            # Dev overlay
            if self.overlay.enabled and self.overlay.on_click(fx, fy, self.renderer):
                self._refine_budget = 0
                self._refine_saved_lod = None
                self._refine_saved_budget = None
                # When auto-LOD is turned off, snap to user base values
                if not self.renderer.auto_lod:
                    self.renderer.lod_pixels = self._user_lod
                    self.renderer.max_render_particles = self._user_budget
                self.renderer.update_visible(self.camera)
                return
        self.camera.on_mouse_button(button, action)

    def _cursor_callback(self, window, xpos, ypos):
        self.camera.on_cursor(xpos, ypos)

    def _char_callback(self, window, codepoint):
        if self.user_menu.on_char(codepoint, self):
            return

    def _scroll_callback(self, window, xoffset, yoffset):
        if self.user_menu.on_scroll(yoffset):
            return
        self.camera.on_scroll(yoffset)

    def _resize_callback(self, window, width, height):
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.camera.aspect = width / height
            self.ctx.viewport = (0, 0, width, height)

    def _screenshot(self, path=None):
        from PIL import Image
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        data = self.ctx.screen.read(components=4, alignment=1)
        img = Image.frombytes("RGBA", (fb_w, fb_h), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
        if path is None:
            path = f"dataflyer_{int(time.time())}.png"
        img.save(path)
        print(f"Screenshot saved: {path}")

    def _print_help(self):
        # Key name lookup for readable output
        _key_names = {
            glfw.KEY_RIGHT: "Right", glfw.KEY_LEFT: "Left",
            glfw.KEY_EQUAL: "+", glfw.KEY_MINUS: "-",
            glfw.KEY_RIGHT_BRACKET: "]", glfw.KEY_LEFT_BRACKET: "[",
            glfw.KEY_PERIOD: ">", glfw.KEY_COMMA: "<",
            glfw.KEY_BACKSLASH: "\\",
        }
        print("\n--- DataFlyer Controls ---")
        print("  LMB+drag : Look around")
        print("  WASD     : Move")
        print("  Z/X      : Up / Down")
        print("  Q/E      : Roll")
        print("  Scroll   : Adjust fly speed")
        print("  ESC      : Quit")
        seen = set()
        for key, (_, desc) in self._key_actions.items():
            if desc is None or desc in seen:
                continue
            seen.add(desc)
            name = _key_names.get(key, chr(key) if 32 < key < 127 else f"key{key}")
            print(f"  {name:<10s} : {desc}")
        scale = "log" if self.renderer.log_scale else "linear"
        print(f"\n  Render   : {self._render_mode.name}")
        print(f"  Colormap : {AVAILABLE_COLORMAPS[self._cmap_idx]}")
        print(f"  Scale    : {scale}")
        print(f"  Range    : {self._range_str()}")
        if self.renderer.n_total > self.renderer.n_particles:
            print(f"  Gas      : {self.renderer.n_particles:,} / {self.renderer.n_total:,} visible")
        else:
            print(f"  Gas      : {self.renderer.n_particles:,}")
        if self.renderer.n_stars > 0:
            print(f"  Stars    : {self.renderer.n_stars}")
        if len(self._snap_list) > 1:
            print(f"  Snapshot : {self._snap_idx + 1} / {len(self._snap_list)}")
        print(f"  Time     : {self.data.time:.4g}")
        print(f"  Speed    : {self.camera.speed:.3g}")
        print("--------------------------\n")

    def _run_cull_frame(self, now, dt, moved):
        """Run one frame of the cull / auto-LOD / progressive refinement logic.

        Called from both run() and run_benchmark() so the benchmark exercises
        the same PID and refinement code paths as interactive use.
        """
        import math

        # Check LOS staleness every frame
        if self._is_los_stale():
            self._apply_render_mode(auto_range=True)

        if moved:
            # --- MOVING: auto-LOD + throttled cull ---
            if self._refine_saved_lod is not None:
                self.renderer.lod_pixels = self._user_lod
                self.renderer.max_render_particles = self._user_budget
                self._refine_saved_lod = None
                self._refine_saved_budget = None
            self._refine_budget = 0

            if not self._was_moving:
                self._pid_integral = 0.0
                self._pid_prev_error = 0.0
                if self.renderer.auto_lod:
                    # Start moving with conservative settings; PID will adjust
                    self.renderer.lod_pixels = max(self._user_lod, 4)
                    self.renderer.max_render_particles = min(self._user_budget, 4_000_000)
                if not self._composite:
                    self.renderer.update_visible(self.camera)
                self._last_cull_time = now

            # Smoothed frame time (unbiased EMA — constant per-frame weight)
            if dt > 0:
                tau = max(self.renderer.auto_lod_smooth, 0.01)
                a = min(1.0, dt / tau)  # ~N-frame window where N=tau/dt
                # Use harmonic-mean-safe approach: smooth 1/dt (frame rate) then invert
                fps_inst = 1000.0 / (dt * 1000)
                self._smooth_fps = (1 - a) * self._smooth_fps + a * fps_inst
                self._smooth_frame_ms = 1000.0 / max(self._smooth_fps, 0.01)

            # PID auto-LOD (operates on smoothed frame time, not raw dt)
            if self.renderer.auto_lod and self._smooth_frame_ms > 0 and dt > 0:
                target_ms = 1000.0 / max(self.renderer.target_fps, 1.0)
                error = math.log2(max(self._smooth_frame_ms / target_ms, 0.01))

                self._pid_integral += error * dt
                self._pid_integral = max(-4.0, min(4.0, self._pid_integral))
                derivative = (error - self._pid_prev_error) / dt if dt > 0 else 0.0
                self._pid_prev_error = error

                rate = (self.renderer.pid_Kp * error
                        + self.renderer.pid_Ki * self._pid_integral
                        + self.renderer.pid_Kd * derivative)
                output = rate * dt

                log2_budget = math.log2(max(self.renderer.max_render_particles, 1_000))
                log2_budget = log2_budget - output
                log2_n = math.log2(max(self.renderer.n_total, 1))
                log2_budget = max(math.log2(1_000), min(log2_n, log2_budget))
                self.renderer.max_render_particles = max(1_000, min(
                    self.renderer.n_total, int(2 ** log2_budget)))

                # Scale lod_pixels proportionally: fewer particles → coarser LOD
                frac = self.renderer.max_render_particles / max(self.renderer.n_total, 1)
                # At frac=1 → lod=1, at frac=0.01 → lod~128
                self.renderer.lod_pixels = max(1.0, min(256.0, 1.0 / max(frac, 0.004)))

            # Throttled cull
            if now - self._last_cull_time >= self.renderer.cull_interval:
                if not self._composite:
                    self.renderer.update_visible(self.camera)
                self._last_cull_time = now

        elif self._refine_budget < self.renderer.n_total:
            # --- STOPPED: progressive refinement ---
            if self._was_moving:
                self._smooth_frame_ms = 0.0
                self._refine_saved_lod = self.renderer.lod_pixels
                self._refine_saved_budget = self.renderer.max_render_particles
                self._refine_budget = max(self._user_budget, 4_000_000)

            if self._refine_budget == 0:
                self._refine_budget = max(self._user_budget, 4_000_000)
                self._refine_saved_lod = self.renderer.lod_pixels
                self._refine_saved_budget = self.renderer.max_render_particles

            self._refine_budget = min(self._refine_budget * 2, self.renderer.n_total)
            self.renderer.lod_pixels = 1
            self.renderer.max_render_particles = self._refine_budget

            if not self._composite:
                self.renderer.update_visible(self.camera)
                if self._refine_saved_lod is not None:
                    self.renderer.lod_pixels = self._refine_saved_lod
                    self.renderer.max_render_particles = self._refine_saved_budget

        self._was_moving = moved

    def run(self):
        """Main event loop."""
        print("DataFlyer running. Press H for help, ESC to quit.")
        self._print_help()

        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt = now - self._last_time
            self._last_time = now

            # FPS counter
            self._frame_count += 1
            if now - self._fps_time > 1.0:
                self._fps = self._frame_count / (now - self._fps_time)
                self._frame_count = 0
                self._fps_time = now
                snap_info = ""
                if len(self._snap_list) > 1:
                    snap_info = f" | snap {self._snap_idx + 1}/{len(self._snap_list)}"
                n_vis = self.renderer.n_particles
                n_tot = self.renderer.n_total
                count = f"{n_vis/1e6:.1f}M/{n_tot/1e6:.1f}M" if n_tot > n_vis else f"{n_tot/1e6:.1f}M"
                glfw.set_window_title(
                    self.window,
                    f"DataFlyer | {self._fps:.0f} fps | {count} | {self._render_mode.name} | "
                    f"{AVAILABLE_COLORMAPS[self._cmap_idx]} | t={self.data.time:.4g}{snap_info}"
                )

            glfw.poll_events()

            # Update camera
            moved = self.camera.update(dt)

            # --- Cull / progressive refinement / auto-LOD ---
            t_cull = 0.0
            if not hasattr(self, '_was_moving'):
                self._was_moving = False
                self._last_cull_time = 0.0
                self._smooth_frame_ms = 0.0
                self._smooth_fps = 0.0
                self._last_lod_adjust = 0.0
                self._pid_integral = 0.0
                self._pid_prev_error = 0.0
                self._refine_budget = 0
                self._refine_saved_lod = None
                self._refine_saved_budget = None

            self._run_cull_frame(now, dt, moved)

            # Get framebuffer size (may differ from window size on retina)
            fb_width, fb_height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, fb_width, fb_height)

            # Read previous frame's GPU timer (1-frame lag, no stall)
            t_render_gpu = self._gpu_query.elapsed * 1e-9  # ns -> seconds

            # Clear and render
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.screen.use()
            if self._composite:
                with self._gpu_query:
                    self._render_composite_frame(fb_width, fb_height)
            else:
                with self._gpu_query:
                    self.renderer.render(self.camera, fb_width, fb_height)
                if self._needs_auto_range:
                    self._auto_range_from_framebuffer()
                    self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                    self.ctx.screen.use()
                    self.renderer.render(self.camera, fb_width, fb_height)
                    self._needs_auto_range = False

            # Update timing stats (exponential moving average)
            alpha = 0.2
            r = self.renderer
            cull_s = getattr(r, '_last_cull_ms', 0) / 1000
            upload_s = getattr(r, '_last_upload_ms', 0) / 1000
            if cull_s > 0 or upload_s > 0:
                self._timings["cull"] = self._timings["cull"] * (1 - alpha) + cull_s * alpha
                self._timings["upload"] = self._timings["upload"] * (1 - alpha) + upload_s * alpha
            self._timings["render"] = self._timings["render"] * (1 - alpha) + t_render_gpu * alpha

            # UI overlays
            if not getattr(self, '_ui_hidden', False):
                self.user_menu.set_framebuffer_size(fb_width, fb_height)
                self.user_menu.update(
                    self.renderer,
                    AVAILABLE_COLORMAPS[self._cmap_idx], AVAILABLE_COLORMAPS,
                    sd_fields=self._sd_fields, sd_field=self._sd_field,
                    sd_field2=self._sd_field2, sd_op=self._sd_op, sd_ops=self._SD_OPS,
                    render_modes=self._RENDER_MODES, render_mode_name=self._render_mode_name,
                    wa_data_field=self._wa_data_field,
                    vector_fields=self._vector_fields,
                    vector_projection=self._vector_projection,
                    vector_projections=self._VECTOR_PROJECTIONS,
                    composite_slots=self._slot if self._composite else None,
                )
                self.user_menu.render()

                if self.overlay.enabled:
                    self.overlay.set_framebuffer_size(fb_width, fb_height)
                    smooth_fps = self._smooth_fps if self._smooth_fps > 0 else self._fps
                    self.overlay.update(
                        self.renderer, self.camera, self._fps,
                        self._render_mode.name, AVAILABLE_COLORMAPS[self._cmap_idx],
                        self._timings, self._last_message,
                        smooth_fps=smooth_fps,
                    )
                    self.overlay.render()

            glfw.swap_buffers(self.window)

        self._cleanup()

    def render_one_frame(self, output_path):
        """Render a single frame and save to file."""
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, fb_width, fb_height)
        # First render to get accumulation data
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.screen.use()
        self.renderer.render(self.camera, fb_width, fb_height)
        # Auto-range from actual pixel values, then re-render
        self._auto_range_from_framebuffer()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.screen.use()
        self.renderer.render(self.camera, fb_width, fb_height)
        self._screenshot(output_path)
        glfw.swap_buffers(self.window)
        self._cleanup()

    def run_benchmark(self, n_frames=100):
        """Scripted benchmark: realistic camera maneuvers, measure frame times."""
        import numpy as np

        boxsize = self.data.header.get("BoxSize", None)
        if boxsize is None:
            extent = np.linalg.norm(
                self.data.positions.max(axis=0) - self.data.positions.min(axis=0)
            )
        else:
            extent = float(boxsize)

        center = np.mean([self.data.positions.min(axis=0),
                          self.data.positions.max(axis=0)], axis=0)
        start_pos = self.camera.position.copy()

        def rodrigues(v, axis, angle):
            c, s = np.cos(angle), np.sin(angle)
            return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)

        # Build waypoints: look around, fly across, turn, fly back
        fwd0 = self.camera.forward.copy()
        up0 = self.camera.up.copy()
        right0 = self.camera.right.copy()

        keyframes = []  # (position, forward, up, label)
        def kf(pos, fwd, up, label):
            keyframes.append((pos.astype(np.float32), fwd.astype(np.float32),
                              up.astype(np.float32), label))

        kf(start_pos, fwd0, up0, "start")
        kf(start_pos, rodrigues(fwd0, right0, np.radians(30)), up0, "look up")
        kf(start_pos, rodrigues(fwd0, right0, np.radians(-30)), up0, "look down")
        kf(start_pos, rodrigues(fwd0, up0, np.radians(45)), up0, "look left")
        kf(start_pos, rodrigues(fwd0, up0, np.radians(-45)), up0, "look right")
        kf(start_pos, fwd0, rodrigues(up0, fwd0, np.radians(30)), "roll left")
        kf(start_pos, fwd0, rodrigues(up0, fwd0, np.radians(-30)), "roll right")
        kf(start_pos, fwd0, up0, "reset")

        # Fly to opposite side
        far_pos = start_pos + fwd0 * extent * 1.5
        n_fly = max(n_frames // 4, 10)
        for i in range(n_fly):
            t = (i + 1) / n_fly
            pos = start_pos * (1 - t) + far_pos * t
            kf(pos, fwd0, up0, f"fly out {int(t*100)}%")

        # Turn around
        fwd_back = -fwd0
        kf(far_pos, fwd_back, up0, "turn around")

        # Fly back
        for i in range(n_fly):
            t = (i + 1) / n_fly
            pos = far_pos * (1 - t) + start_pos * t
            kf(pos, fwd_back, up0, f"fly back {int(t*100)}%")

        # Orbit around the data center for remaining frames
        n_orbit = max(n_frames - len(keyframes), 0)
        if n_orbit > 0:
            radius = extent * 0.6
            angles = np.linspace(0, 2 * np.pi, n_orbit, endpoint=False)
            for theta in angles:
                pos = np.array([
                    center[0] + radius * np.sin(theta),
                    center[1],
                    center[2] + radius * np.cos(theta),
                ], dtype=np.float32)
                fwd = (center - pos)
                fwd = fwd / np.linalg.norm(fwd)
                kf(pos, fwd, up0, "orbit")

        keyframes = keyframes[:n_frames]

        # Progressive refinement to full detail before starting benchmark
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, fb_width, fb_height)
        budget = max(4_000_000, self.renderer.max_render_particles)
        while budget < self.renderer.n_total:
            budget = min(budget * 2, self.renderer.n_total)
            self.renderer.lod_pixels = 1
            self.renderer.max_render_particles = budget
            self.renderer.update_visible(self.camera)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.screen.use()
            self.renderer.render(self.camera, fb_width, fb_height)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            n_vis = self.renderer.n_particles + self.renderer.n_big
            print(f"  Refining: {n_vis:,} / {self.renderer.n_total:,} particles")
            if glfw.window_should_close(self.window):
                self._cleanup()
                return

        # Auto-range at full quality
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.screen.use()
        self.renderer.render(self.camera, fb_width, fb_height)
        self._auto_range_from_framebuffer()

        # Restore user LOD for benchmark run
        self.renderer.lod_pixels = self._user_lod
        self.renderer.max_render_particles = self._user_budget

        # Warmup
        for pos, fwd, up, _ in keyframes[:10]:
            self.camera.position = pos.copy()
            self.camera._forward = fwd.copy()
            self.camera._up = up.copy()
            self.camera._dirty = True
            self.renderer.update_visible(self.camera)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.screen.use()
            self.renderer.render(self.camera, fb_width, fb_height)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        # Timed run: smoothly interpolate between keyframes
        # Each keyframe transition takes 0.5s, then hold for 1.5s
        TRANSITION_S = 0.5
        HOLD_S = 1.5
        frame_times = []
        cull_times = []
        n_vis_list = []
        cam_pos_list = []
        cam_fwd_list = []
        phase_list = []
        kf_idx_list = []

        def slerp_vec(a, b, t):
            """Interpolate unit vectors via slerp-like blend."""
            blend = a * (1 - t) + b * t
            n = np.linalg.norm(blend)
            return blend / n if n > 1e-8 else a

        def render_frame():
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, fb_w, fb_h)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.screen.use()
            self.renderer.render(self.camera, fb_w, fb_h)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        def record(t_frame_ms, phase, keyframe_idx):
            r = self.renderer
            cull_times.append(getattr(r, '_last_cull_ms', 0) + getattr(r, '_last_upload_ms', 0))
            frame_times.append(t_frame_ms)
            n_vis_list.append(r.n_particles + r.n_big)
            cam_pos_list.append(self.camera.position.copy())
            cam_fwd_list.append(self.camera.forward.copy())
            phase_list.append(phase)
            kf_idx_list.append(keyframe_idx)
            lod_list.append(r.lod_pixels)
            budget_list.append(r.max_render_particles)

        lod_list = []
        budget_list = []
        prev_kf = keyframes[0]

        # Initialize main-loop state so PID and refinement work
        self._was_moving = False
        self._last_cull_time = 0.0
        self._smooth_frame_ms = 0.0
        self._last_lod_adjust = 0.0
        self._pid_integral = 0.0
        self._refine_budget = 0
        self._refine_saved_lod = None
        self._refine_saved_budget = None

        for ki, (pos, fwd, up, label) in enumerate(keyframes):
            if glfw.window_should_close(self.window):
                break

            p0, f0, u0, _ = prev_kf

            # --- Transition: interpolate, using real PID auto-LOD ---
            t_start = time.perf_counter()
            t_prev = t_start
            while True:
                if glfw.window_should_close(self.window):
                    break
                elapsed = time.perf_counter() - t_start
                if elapsed >= TRANSITION_S:
                    break
                t = elapsed / TRANSITION_S
                self.camera.position = (p0 * (1 - t) + pos * t).astype(np.float32)
                self.camera._forward = slerp_vec(f0, fwd, t).astype(np.float32)
                self.camera._up = slerp_vec(u0, up, t).astype(np.float32)
                self.camera._dirty = True

                # Simulate main loop: moved=True
                now = time.perf_counter()
                dt = now - t_prev
                self._run_cull_frame(now, dt, moved=True)
                render_frame()

                now2 = time.perf_counter()
                record((now2 - t_prev) * 1000, "transition", ki)
                t_prev = now2

            # --- Hold: camera stationary, using real progressive refinement ---
            self.camera.position = pos.copy()
            self.camera._forward = fwd.copy()
            self.camera._up = up.copy()
            self.camera._dirty = True

            t_start = time.perf_counter()
            t_prev = t_start
            while True:
                if glfw.window_should_close(self.window):
                    break
                if time.perf_counter() - t_start >= HOLD_S:
                    break

                now = time.perf_counter()
                dt = now - t_prev
                self._run_cull_frame(now, dt, moved=False)
                render_frame()

                now2 = time.perf_counter()
                record((now2 - t_prev) * 1000, "hold", ki)
                t_prev = now2

            prev_kf = (pos, fwd, up, label)

        frame_times = np.array(frame_times)
        cull_times = np.array(cull_times)
        phases = np.array(phase_list)

        print(f"\n--- Benchmark Results ({len(frame_times)} frames) ---")
        for phase_name in ("transition", "hold"):
            mask = phases == phase_name
            if not mask.any():
                continue
            ft = frame_times[mask]
            ct = cull_times[mask]
            fps = 1000.0 / ft
            print(f"  {phase_name.capitalize()} ({mask.sum()} frames):")
            print(f"    Frame time:  median={np.median(ft):.1f}ms  "
                  f"p5={np.percentile(ft, 5):.1f}ms  "
                  f"p95={np.percentile(ft, 95):.1f}ms")
            print(f"    FPS:         median={np.median(fps):.0f}  "
                  f"p5={np.percentile(fps, 5):.0f}  "
                  f"p95={np.percentile(fps, 95):.0f}")
            print(f"    Cull time:   median={np.median(ct):.1f}ms  "
                  f"p95={np.percentile(ct, 95):.1f}ms")
        print(f"  Particles:   {self.renderer.n_total:,} total")

        # Save detailed per-frame data
        outfile = f"benchmark_{int(time.time())}.npz"
        kf_labels = [kf[3] for kf in keyframes]
        np.savez(outfile,
                 frame_time_ms=frame_times,
                 cull_time_ms=cull_times,
                 n_visible=np.array(n_vis_list),
                 cam_pos=np.array(cam_pos_list),
                 cam_fwd=np.array(cam_fwd_list),
                 phase=np.array(phase_list),
                 keyframe_idx=np.array(kf_idx_list),
                 keyframe_labels=np.array(kf_labels),
                 lod_pixels=np.array(lod_list),
                 max_render_particles=np.array(budget_list),
                 n_total=self.renderer.n_total)
        print(f"  Saved: {outfile} ({len(frame_times)} frames)")
        print("-----------------------------------\n")

        self._cleanup()

    def _cleanup(self):
        self.overlay.release()
        self.user_menu.release()
        self.renderer.release()
        self.data.close()
        glfw.terminate()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DataFlyer - Real-time mesh-free data explorer")
    parser.add_argument("snapshot", help="Path to HDF5 snapshot file")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of view in degrees")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Render one frame to this file and exit")
    parser.add_argument("--benchmark", type=int, default=None, metavar="N",
                        help="Run N-frame scripted benchmark and exit")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Run in fullscreen mode at specified resolution")
    parser.add_argument("--backend", type=str, default="wgpu",
                        choices=["moderngl", "wgpu"],
                        help="Rendering backend (default: wgpu)")
    args = parser.parse_args()

    if args.backend == "wgpu":
        from .wgpu_app import run_wgpu_app
        run_wgpu_app(args.snapshot, width=args.width, height=args.height, fov=args.fov,
                     screenshot=args.screenshot, benchmark=args.benchmark,
                     fullscreen=args.fullscreen)
        return

    app = DataFlyerApp(args.snapshot, width=args.width, height=args.height, fov=args.fov,
                       fullscreen=args.fullscreen)
    if args.screenshot:
        app.render_one_frame(args.screenshot)
    elif args.benchmark is not None:
        app.run_benchmark(n_frames=args.benchmark)
    else:
        app.run()


if __name__ == "__main__":
    main()
