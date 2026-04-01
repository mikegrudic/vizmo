"""Main application: glfw window, event loop, orchestration."""

import sys
import time
import glfw
import moderngl

from .camera import Camera
from .data_manager import SnapshotData, find_snapshots
from .renderer import SplatRenderer
from .colormaps import create_colormap_texture_safe, AVAILABLE_COLORMAPS


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

        # Colormaps
        self._colormap_textures = {}
        self._cmap_idx = 0
        self._set_colormap(AVAILABLE_COLORMAPS[0])

        # Quantities
        self._quantities = self.data.available_quantities()
        self._qty_idx = 0
        self._current_qty = self._quantities[0]

        # Store particles and upload (with culling for large datasets)
        quantity = self.data.get_quantity(self._current_qty)
        self.renderer.set_particles(
            self.data.positions, self.data.hsml, self.data.masses, quantity,
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
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)

        # Timing
        self._last_time = time.perf_counter()
        self._frame_count = 0
        self._fps_time = time.perf_counter()
        self._fps = 0.0

    def _load_snapshot(self, path):
        """Load a new snapshot, keeping the current camera position."""
        import os
        self.data.close()
        print(f"Loading {os.path.basename(path)}...")
        self.data = SnapshotData(path)
        print(f"  {self.data.n_particles:,} gas, {self.data.n_stars} stars, t={self.data.time:.4g}")

        self._quantities = self.data.available_quantities()
        self._qty_idx = min(self._qty_idx, len(self._quantities) - 1)
        self._current_qty = self._quantities[self._qty_idx]

        q = self.data.get_quantity(self._current_qty)
        self.renderer.set_particles(
            self.data.positions, self.data.hsml, self.data.masses, q,
        )
        self.renderer.update_visible(self.camera)
        self.renderer.mode = 0 if self._current_qty == "surface_density" else 1

        if self.data.n_stars > 0:
            self.renderer.upload_stars(self.data.star_positions, self.data.star_masses)

        self._needs_auto_range = True

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
        print(f"Auto-range: {self._range_str()}")

    def _set_colormap(self, name):
        if name not in self._colormap_textures:
            self._colormap_textures[name] = create_colormap_texture_safe(self.ctx, name)
        self.renderer.colormap_tex = self._colormap_textures[name]

    def _cycle_quantity(self, direction=1):
        self._qty_idx = (self._qty_idx + direction) % len(self._quantities)
        self._current_qty = self._quantities[self._qty_idx]
        print(f"Quantity: {self._current_qty}")
        q = self.data.get_quantity(self._current_qty)
        self.renderer.update_quantity(q)
        self._needs_auto_range = True  # auto-range after next render
        if self._current_qty == "surface_density":
            self.renderer.mode = 0
        else:
            self.renderer.mode = 1

    def _cycle_colormap(self, direction=1):
        self._cmap_idx = (self._cmap_idx + direction) % len(AVAILABLE_COLORMAPS)
        name = AVAILABLE_COLORMAPS[self._cmap_idx]
        print(f"Colormap: {name}")
        self._set_colormap(name)

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            # Still forward to camera for key release
            self.camera.on_key(key, action)
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return

        # Tab: cycle quantity
        if key == glfw.KEY_TAB:
            self._cycle_quantity(1 if not (mods & glfw.MOD_SHIFT) else -1)
            return

        # 1-9: direct quantity selection
        num_keys = [glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4,
                    glfw.KEY_5, glfw.KEY_6, glfw.KEY_7, glfw.KEY_8, glfw.KEY_9]
        for i, nk in enumerate(num_keys):
            if key == nk and i < len(self._quantities):
                self._qty_idx = i
                self._current_qty = self._quantities[i]
                print(f"Quantity: {self._current_qty}")
                q = self.data.get_quantity(self._current_qty)
                self.renderer.update_quantity(q)
                self.renderer.mode = 0 if self._current_qty == "surface_density" else 1
                self._needs_auto_range = True
                return

        # C: cycle colormap
        if key == glfw.KEY_C:
            self._cycle_colormap(1 if not (mods & glfw.MOD_SHIFT) else -1)
            return

        # Left/Right: navigate snapshots
        if key == glfw.KEY_RIGHT and len(self._snap_list) > 1:
            step = 10 if (mods & glfw.MOD_SHIFT) else 1
            self._snap_idx = min(self._snap_idx + step, len(self._snap_list) - 1)
            self._load_snapshot(self._snap_list[self._snap_idx])
            return
        if key == glfw.KEY_LEFT and len(self._snap_list) > 1:
            step = 10 if (mods & glfw.MOD_SHIFT) else 1
            self._snap_idx = max(self._snap_idx - step, 0)
            self._load_snapshot(self._snap_list[self._snap_idx])
            return

        # R: auto-range from rendered image
        if key == glfw.KEY_R:
            self._auto_range_from_framebuffer()
            return

        # +/-: expand/contract dynamic range
        if key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
            mid = (self.renderer.qty_min + self.renderer.qty_max) / 2
            half = (self.renderer.qty_max - self.renderer.qty_min) / 2
            half *= 0.8  # contract = more contrast
            self.renderer.qty_min = mid - half
            self.renderer.qty_max = mid + half
            print(f"Range: {self._range_str()}")
            return
        if key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            mid = (self.renderer.qty_min + self.renderer.qty_max) / 2
            half = (self.renderer.qty_max - self.renderer.qty_min) / 2
            half *= 1.25  # expand = less contrast
            self.renderer.qty_min = mid - half
            self.renderer.qty_max = mid + half
            print(f"Range: {self._range_str()}")
            return

        # L: toggle log/linear scale
        if key == glfw.KEY_L:
            self.renderer.log_scale = 1 - self.renderer.log_scale
            scale_name = "log" if self.renderer.log_scale else "linear"
            print(f"Scale: {scale_name}")
            self._needs_auto_range = True
            return

        # P: screenshot
        if key == glfw.KEY_P:
            self._screenshot()
            return

        # H: print help
        if key == glfw.KEY_H:
            self._print_help()
            return

        self.camera.on_key(key, action)

    def _mouse_button_callback(self, window, button, action, mods):
        self.camera.on_mouse_button(button, action)

    def _cursor_callback(self, window, xpos, ypos):
        self.camera.on_cursor(xpos, ypos)

    def _scroll_callback(self, window, xoffset, yoffset):
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
        data = self.ctx.screen.read(components=3)
        img = Image.frombytes("RGB", (fb_w, fb_h), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if path is None:
            path = f"dataflyer_{int(time.time())}.png"
        img.save(path)
        print(f"Screenshot saved: {path}")

    def _print_help(self):
        print("\n--- DataFlyer Controls ---")
        print("  LMB+drag : Look around")
        print("  WASD     : Move")
        print("  Space    : Up  |  Shift : Down")
        print("  Q/E      : Roll")
        print("  Scroll   : Adjust fly speed")
        print("  Left/Right: Previous/next snapshot (Shift: skip 10)")
        print("  1-9      : Select quantity directly")
        print("  Tab      : Next quantity  |  Shift+Tab : Previous")
        print("  C        : Next colormap  |  Shift+C   : Previous")
        print("  +/-      : Contract/expand dynamic range")
        print("  R        : Auto-range dynamic range")
        print("  L        : Toggle log/linear scale")
        print("  P        : Screenshot")
        print("  H        : Print this help")
        print("  ESC      : Quit")
        print(f"\n  Quantities:")
        for i, q in enumerate(self._quantities):
            marker = " *" if q == self._current_qty else ""
            print(f"    {i+1}: {q}{marker}")
        scale = "log" if self.renderer.log_scale else "linear"
        print(f"\n  Colormap : {AVAILABLE_COLORMAPS[self._cmap_idx]}")
        print(f"  Opacity  : {self.renderer.alpha_scale:.4f}")
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
                    f"DataFlyer | {self._fps:.0f} fps | {count} | {self._current_qty} | "
                    f"{AVAILABLE_COLORMAPS[self._cmap_idx]} | t={self.data.time:.4g}{snap_info}"
                )

            glfw.poll_events()

            # Update camera
            moved = self.camera.update(dt)

            # Re-cull visible particles when camera moves
            if moved and self.renderer.n_total > self.renderer.max_render_particles:
                self.renderer.update_visible(self.camera)

            # Get framebuffer size (may differ from window size on retina)
            fb_width, fb_height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, fb_width, fb_height)

            # Clear and render
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.screen.use()
            self.renderer.render(self.camera, fb_width, fb_height)

            # Deferred auto-range: read back actual pixel values after first render
            if self._needs_auto_range:
                self._auto_range_from_framebuffer()
                # Re-render with the correct range
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                self.ctx.screen.use()
                self.renderer.render(self.camera, fb_width, fb_height)
                self._needs_auto_range = False

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

    def _cleanup(self):
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
    args = parser.parse_args()

    app = DataFlyerApp(args.snapshot, width=args.width, height=args.height, fov=args.fov)
    if args.screenshot:
        app.render_one_frame(args.screenshot)
    else:
        app.run()


if __name__ == "__main__":
    main()
