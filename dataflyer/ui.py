"""Dear ImGui UI panels for the explorer."""

import imgui


class ExplorerUI:
    def __init__(self, renderer, data, camera):
        self.renderer = renderer
        self.data = data
        self.camera = camera

        self.quantities = data.available_quantities()
        self.current_qty_idx = 0
        self.current_qty = self.quantities[0] if self.quantities else "surface_density"

        from .colormaps import AVAILABLE_COLORMAPS
        self.colormaps = AVAILABLE_COLORMAPS
        self.current_cmap_idx = 0
        self.current_cmap = self.colormaps[0]

        self._fps_history = []
        self._qty_changed = False
        self._cmap_changed = False

    def render(self, dt):
        """Draw all UI panels. Returns (qty_changed, cmap_changed)."""
        self._qty_changed = False
        self._cmap_changed = False

        self._draw_controls()
        self._draw_info(dt)

        return self._qty_changed, self._cmap_changed

    def _draw_controls(self):
        imgui.begin("Controls", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        # Quantity selector
        changed, self.current_qty_idx = imgui.combo(
            "Quantity", self.current_qty_idx, self.quantities
        )
        if changed:
            self.current_qty = self.quantities[self.current_qty_idx]
            self._qty_changed = True

        # Colormap selector
        changed, self.current_cmap_idx = imgui.combo(
            "Colormap", self.current_cmap_idx, self.colormaps
        )
        if changed:
            self.current_cmap = self.colormaps[self.current_cmap_idx]
            self._cmap_changed = True

        # Render mode
        changed, self.renderer.mode = imgui.combo(
            "Mode", self.renderer.mode, ["Surface Density", "Weighted Quantity"]
        )

        imgui.separator()

        # Dynamic range
        changed, (self.renderer.qty_min, self.renderer.qty_max) = imgui.slider_float2(
            "Log Range", self.renderer.qty_min, self.renderer.qty_max,
            min_value=-6.0, max_value=10.0, format="%.1f"
        )

        # Alpha scale (log slider)
        changed, self.renderer.alpha_scale = imgui.slider_float(
            "Opacity", self.renderer.alpha_scale,
            min_value=0.001, max_value=1.0, format="%.3f",
            flags=imgui.SLIDER_FLAGS_LOGARITHMIC,
        )

        imgui.separator()

        # Camera speed
        changed, self.camera.speed = imgui.slider_float(
            "Fly Speed", self.camera.speed,
            min_value=1e-4, max_value=1e6, format="%.2e",
            flags=imgui.SLIDER_FLAGS_LOGARITHMIC,
        )

        imgui.end()

    def _draw_info(self, dt):
        imgui.begin("Info", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        fps = 1.0 / dt if dt > 0 else 0
        self._fps_history.append(fps)
        if len(self._fps_history) > 60:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        imgui.text(f"FPS: {avg_fps:.0f}")
        imgui.text(f"Particles: {self.renderer.n_particles:,}")
        imgui.text(f"Position: ({self.camera.position[0]:.3g}, "
                   f"{self.camera.position[1]:.3g}, "
                   f"{self.camera.position[2]:.3g})")
        if self.data.time:
            imgui.text(f"Sim Time: {self.data.time:.4g}")

        imgui.separator()
        imgui.text_wrapped(
            "LMB+drag: look | WASD: move | Q/E: roll\n"
            "Space/Shift: up/down | Scroll: speed"
        )

        imgui.end()
