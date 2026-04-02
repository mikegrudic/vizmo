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


class DataFlyerApp:
    def __init__(self, snapshot_path, width=1920, height=1080, fov=90.0):
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

        self.window = glfw.create_window(width, height, "DataFlyer", None, None)
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
        import numpy as np
        if field_name in self._vector_fields:
            vec = self.data.get_vector_field(field_name)
            proj = self._vector_projection
            if proj == "LOS":
                fwd = self.camera.forward
                self._los_camera_fwd = fwd.copy()
                return (vec @ fwd).astype(np.float32)
            elif proj == "|v|":
                return np.linalg.norm(vec, axis=1).astype(np.float32)
            else:  # |v|^2
                return (vec * vec).sum(axis=1).astype(np.float32)
        return self.data.get_field(field_name)

    def _compute_weights(self):
        """Compute the final weight array from field1, op, and field2."""
        import numpy as np
        w = self._project_field(self._sd_field)
        if self._sd_field2 != "None":
            w2 = self._project_field(self._sd_field2)
            op = self._sd_op
            if op == "*":
                w = w * w2
            elif op == "+":
                w = w + w2
            elif op == "-":
                w = w - w2
            elif op == "/":
                w = w / np.maximum(np.abs(w2), 1e-30) * np.sign(w2)
            elif op == "min":
                w = np.minimum(w, w2)
            elif op == "max":
                w = np.maximum(w, w2)
        return w

    def _uses_vector_field(self):
        """Check if any active field is a vector field."""
        if self._render_mode_name == "WeightedAverage":
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
        self._composite = (self._render_mode_name == "Composite")
        if self._composite:
            self._render_mode = RenderMode(
                name="Composite", weight_field="", qty_field="", resolve_mode=-1)
            # Auto-range each slot by doing a quick render + readback
            if auto_range:
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
        import numpy as np
        s = slot
        w_name = s["weight"]
        if w_name in self._vector_fields:
            vec = self.data.get_vector_field(w_name)
            proj = s["proj"]
            if proj == "LOS":
                weights = (vec @ self.camera.forward).astype(np.float32)
            elif proj == "|v|":
                weights = np.linalg.norm(vec, axis=1).astype(np.float32)
            else:
                weights = (vec * vec).sum(axis=1).astype(np.float32)
        else:
            weights = self.data.get_field(w_name)
            if s["weight2"] != "None":
                w2_name = s["weight2"]
                if w2_name in self._vector_fields:
                    vec = self.data.get_vector_field(w2_name)
                    proj = s["proj"]
                    if proj == "LOS":
                        w2 = (vec @ self.camera.forward).astype(np.float32)
                    elif proj == "|v|":
                        w2 = np.linalg.norm(vec, axis=1).astype(np.float32)
                    else:
                        w2 = (vec * vec).sum(axis=1).astype(np.float32)
                else:
                    w2 = self.data.get_field(w2_name)
                op = s["op"]
                if op == "*": weights = weights * w2
                elif op == "+": weights = weights + w2
                elif op == "-": weights = weights - w2
                elif op == "/": weights = weights / np.maximum(np.abs(w2), 1e-30) * np.sign(w2)
                elif op == "min": weights = np.minimum(weights, w2)
                elif op == "max": weights = np.maximum(weights, w2)

        if s["mode"] in ("WeightedAverage", "WeightedVariance"):
            d_name = s["data"]
            if d_name in self._vector_fields:
                vec = self.data.get_vector_field(d_name)
                proj = s["proj"]
                if proj == "LOS":
                    qty = (vec @ self.camera.forward).astype(np.float32)
                elif proj == "|v|":
                    qty = np.linalg.norm(vec, axis=1).astype(np.float32)
                else:
                    qty = (vec * vec).sum(axis=1).astype(np.float32)
            else:
                qty = self.data.get_field(d_name)
        else:
            qty = None

        resolve = {"SurfaceDensity": 0, "WeightedAverage": 1, "WeightedVariance": 2}[s["mode"]]
        s["resolve"] = resolve
        return weights, qty

    def _render_composite_frame(self, fb_width, fb_height):
        """Render two fields into separate FBOs and composite."""
        r = self.renderer
        r._ensure_fbo(fb_width, fb_height, which=1)
        r._ensure_fbo(fb_width, fb_height, which=2)

        # Slot 0 (lightness): load weights/qty, cull, render into FBO1
        s0 = self._slot[0]
        w0, q0 = self._compute_slot(s0)
        r.update_weights(w0, q0)
        r.update_visible(self.camera)
        r._render_accum(self.camera, fb_width, fb_height, r._accum_fbo)

        # Slot 1 (color): load weights/qty, cull, render into FBO2
        s1 = self._slot[1]
        w1, q1 = self._compute_slot(s1)
        r.update_weights(w1, q1)
        r.update_visible(self.camera)
        r._render_accum(self.camera, fb_width, fb_height, r._accum_fbo2)

        # Composite resolve
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
        self._msg(f"LOD: {self.renderer.lod_pixels}px (more detail)")
        self._refinement_level = 0
        self.renderer.update_visible(self.camera)

    def _decrease_lod(self):
        self.renderer.lod_pixels = min(256, self.renderer.lod_pixels * 2)
        self._msg(f"LOD: {self.renderer.lod_pixels}px (faster)")
        self._refinement_level = 0
        self.renderer.update_visible(self.camera)

    def _increase_budget(self):
        self.renderer.max_render_particles = min(
            self.renderer.n_total, int(self.renderer.max_render_particles * 2))
        self._msg(f"Max particles: {self.renderer.max_render_particles/1e6:.1f}M")
        self._refinement_level = 0
        self.renderer.update_visible(self.camera)

    def _decrease_budget(self):
        self.renderer.max_render_particles = max(
            100_000, self.renderer.max_render_particles // 2)
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

            # Re-cull visible particles when camera moves or progressively refine when still
            t_cull = 0.0
            if not hasattr(self, '_was_moving'):
                self._was_moving = False
                self._still_frames = 0
                self._refinement_level = 0  # 0=moving, 1=stopped, 2=refined, 3=full
                self._last_cull_time = 0.0
            if moved:
                self._still_frames = 0
                if self._refinement_level != 0:
                    self._refinement_level = 0
                # Throttle culls: skip if less than cull_interval since last cull
                if now - self._last_cull_time >= self.renderer.cull_interval:
                    t0 = time.perf_counter()
                    self.renderer.update_visible(self.camera)
                    t_cull = time.perf_counter() - t0
                    self._last_cull_time = now
            elif self._was_moving or self._refinement_level < 3:
                self._still_frames += 1
                # Progressive refinement: relax LOD and budget when stationary
                if self._still_frames >= 2 and self._refinement_level < 3:
                    # Recompute LOS projection if camera rotated since last compute
                    if self._refinement_level == 0 and self._is_los_stale():
                        self._apply_render_mode(auto_range=False)
                    saved_lod = self.renderer.lod_pixels
                    saved_budget = self.renderer.max_render_particles
                    if self._refinement_level == 0:
                        # First refinement: halve LOD threshold
                        self.renderer.lod_pixels = max(1, saved_lod // 2)
                        self.renderer.max_render_particles = min(saved_budget * 2, self.renderer.n_total)
                        self._refinement_level = 1
                    elif self._refinement_level == 1:
                        # Second: minimize LOD, double budget again
                        self.renderer.lod_pixels = 1
                        self.renderer.max_render_particles = min(saved_budget * 2, self.renderer.n_total)
                        self._refinement_level = 2
                    elif self._refinement_level == 2:
                        # Final: full quality (LOD off, max budget)
                        self.renderer.lod_pixels = 1
                        self.renderer.max_render_particles = self.renderer.n_total
                        self._refinement_level = 3
                    t0 = time.perf_counter()
                    self.renderer.update_visible(self.camera)
                    t_cull = time.perf_counter() - t0
                    # Restore user settings (renderer uses them next time camera moves)
                    self.renderer.lod_pixels = saved_lod
                    self.renderer.max_render_particles = saved_budget
            self._was_moving = moved

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
            if t_cull > 0:
                self._timings["cull"] = self._timings["cull"] * (1 - alpha) + t_cull * alpha
            self._timings["render"] = self._timings["render"] * (1 - alpha) + t_render_gpu * alpha

            # User menu (always visible)
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

            # Dev overlay
            if self.overlay.enabled:
                self.overlay.set_framebuffer_size(fb_width, fb_height)
                self.overlay.update(
                    self.renderer, self.camera, self._fps,
                    self._render_mode.name, AVAILABLE_COLORMAPS[self._cmap_idx],
                    self._timings, self._last_message, self.renderer.cull_interval,
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
        """Scripted benchmark: fly a camera path, measure real frame times, print stats."""
        import numpy as np

        boxsize = self.data.header.get("BoxSize", None)
        if boxsize is None:
            extent = np.linalg.norm(
                self.data.positions.max(axis=0) - self.data.positions.min(axis=0)
            )
        else:
            extent = float(boxsize)
        center = self.camera.position.copy()

        # Build a camera path: orbit around the data center
        angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        radius = extent * 0.6
        orbit_center = center.copy()
        orbit_center[2] -= radius  # start looking at center from current position

        # Auto-range first
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, fb_width, fb_height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.screen.use()
        self.renderer.render(self.camera, fb_width, fb_height)
        self._auto_range_from_framebuffer()

        # Warmup: 10 frames
        for i in range(10):
            theta = angles[i % len(angles)]
            self.camera.position = np.array([
                orbit_center[0] + radius * np.sin(theta),
                orbit_center[1],
                orbit_center[2] + radius * np.cos(theta),
            ], dtype=np.float32)
            self.camera._forward = (orbit_center - self.camera.position)
            self.camera._forward /= np.linalg.norm(self.camera._forward)
            self.camera._dirty = True

            self.renderer.update_visible(self.camera)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.screen.use()
            self.renderer.render(self.camera, fb_width, fb_height)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        # Timed run
        frame_times = []
        cull_times = []
        t_prev = time.perf_counter()

        for i in range(n_frames):
            if glfw.window_should_close(self.window):
                break

            theta = angles[i]
            self.camera.position = np.array([
                orbit_center[0] + radius * np.sin(theta),
                orbit_center[1],
                orbit_center[2] + radius * np.cos(theta),
            ], dtype=np.float32)
            self.camera._forward = (orbit_center - self.camera.position)
            self.camera._forward /= np.linalg.norm(self.camera._forward)
            self.camera._dirty = True

            t0 = time.perf_counter()
            self.renderer.update_visible(self.camera)
            cull_times.append((time.perf_counter() - t0) * 1000)

            fb_width, fb_height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, fb_width, fb_height)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.screen.use()
            self.renderer.render(self.camera, fb_width, fb_height)

            # Overlays
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

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            now = time.perf_counter()
            frame_times.append((now - t_prev) * 1000)
            t_prev = now

        frame_times = np.array(frame_times)
        cull_times = np.array(cull_times)
        fps = 1000.0 / frame_times

        print(f"\n--- Benchmark Results ({len(frame_times)} frames) ---")
        print(f"  Frame time:  median={np.median(frame_times):.1f}ms  "
              f"p5={np.percentile(frame_times, 5):.1f}ms  "
              f"p95={np.percentile(frame_times, 95):.1f}ms")
        print(f"  FPS:         median={np.median(fps):.0f}  "
              f"p5={np.percentile(fps, 5):.0f}  "
              f"p95={np.percentile(fps, 95):.0f}")
        print(f"  Cull time:   median={np.median(cull_times):.1f}ms  "
              f"p95={np.percentile(cull_times, 95):.1f}ms")
        print(f"  Particles:   {self.renderer.n_total:,} total")
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

    parser = argparse.ArgumentParser(description="DataFlyer - Real-time SPH data explorer")
    parser.add_argument("snapshot", help="Path to HDF5 snapshot file")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of view in degrees")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Render one frame to this file and exit")
    parser.add_argument("--benchmark", type=int, default=None, metavar="N",
                        help="Run N-frame scripted benchmark and exit")
    args = parser.parse_args()

    app = DataFlyerApp(args.snapshot, width=args.width, height=args.height, fov=args.fov)
    if args.screenshot:
        app.render_one_frame(args.screenshot)
    elif args.benchmark is not None:
        app.run_benchmark(n_frames=args.benchmark)
    else:
        app.run()


if __name__ == "__main__":
    main()
