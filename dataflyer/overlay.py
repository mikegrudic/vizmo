"""UI overlay panels rendered as PIL images. The actual GPU upload +
draw step lives in wgpu_overlay.py; this file contains only the
backend-agnostic widget layout, hit-testing, and PIL rendering.
"""

from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass


def _get_font(size):
    try:
        import matplotlib.font_manager as fm
        path = fm.findfont(fm.FontProperties(family="monospace"))
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@dataclass
class PanelStyle:
    font_size: int
    line_height: int
    margin: int
    min_width: int
    bg_color: tuple
    text_color: tuple
    accent_color: tuple
    toggle_on_color: tuple
    toggle_off_color: tuple
    dropdown_bg: tuple
    dropdown_hover: tuple
    slider_btn: tuple
    field_bg: tuple = (30, 30, 30, 255)
    field_active: tuple = (50, 50, 80, 255)
    position: str = "top-right"  # "top-right" or "bottom-left"


DEV_STYLE = PanelStyle(
    font_size=14, line_height=20, margin=8, min_width=200,
    bg_color=(0, 0, 0, 200),
    text_color=(0, 255, 0, 255),
    accent_color=(0, 255, 0, 255),
    toggle_on_color=(0, 200, 0, 255),
    toggle_off_color=(150, 50, 50, 255),
    dropdown_bg=(40, 40, 40, 255),
    dropdown_hover=(80, 80, 120, 255),
    slider_btn=(80, 80, 80, 255),
    position="top-right",
)

USER_STYLE = PanelStyle(
    font_size=28, line_height=37, margin=13, min_width=267,
    bg_color=(15, 15, 25, 100),
    text_color=(220, 220, 220, 255),
    accent_color=(100, 180, 255, 255),
    toggle_on_color=(100, 180, 255, 255),
    toggle_off_color=(220, 220, 220, 255),
    dropdown_bg=(30, 30, 30, 255),
    dropdown_hover=(80, 100, 140, 255),
    slider_btn=(80, 80, 80, 255),
    position="bottom-left",
)


class Panel:
    """Base class for PIL-rendered UI panels uploaded as GPU textures.

    Subclasses override build_items() to define widget content and
    on_widget_click() to handle interactions.
    """

    # Reference height for DPI scaling (styles are designed for 1080p)
    _REF_HEIGHT = 1080

    def __init__(self, style):
        self.style = style
        self._base_style = style  # unscaled original
        self._tex = None
        self._font = _get_font(style.font_size)
        self._dpi_scale = 1.0
        self._widgets = []
        self._dropdown_open = None
        self._dropdown_scroll = {}
        self._fb_width = 1
        self._fb_height = 1
        self._panel_x = 0
        self._panel_y = 0
        self._panel_w = 0
        self._panel_h = 0
        self._last_items_key = None  # for dirty-flag caching

    def set_framebuffer_size(self, w, h):
        self._fb_width = w
        self._fb_height = h
        # DPI scaling: scale UI relative to 1080p reference
        new_scale = max(h, 1) / self._REF_HEIGHT
        if abs(new_scale - self._dpi_scale) > 0.01:
            self._dpi_scale = new_scale
            bs = self._base_style
            self.style = PanelStyle(
                font_size=max(8, int(bs.font_size * new_scale)),
                line_height=max(12, int(bs.line_height * new_scale)),
                margin=max(4, int(bs.margin * new_scale)),
                min_width=max(100, int(bs.min_width * new_scale)),
                bg_color=bs.bg_color, text_color=bs.text_color,
                accent_color=bs.accent_color,
                toggle_on_color=bs.toggle_on_color,
                toggle_off_color=bs.toggle_off_color,
                dropdown_bg=bs.dropdown_bg,
                dropdown_hover=bs.dropdown_hover,
                slider_btn=bs.slider_btn,
                field_bg=bs.field_bg,
                field_active=bs.field_active,
                position=bs.position,
            )
            self._font = _get_font(self.style.font_size)
            self._last_items_key = None  # force re-render

    def render_panel(self, items):
        """Measure, draw, and upload a list of widget items. Skips if unchanged."""
        # Dirty check: hash items + dropdown state + framebuffer size to skip re-render
        items_key = (
            tuple(tuple(item) if not isinstance(item, tuple) else item for item in items),
            self._dropdown_open,
            tuple(sorted(self._dropdown_scroll.items())),
            self._fb_width, self._fb_height,
        )
        if items_key == self._last_items_key and self._tex is not None:
            return
        self._last_items_key = items_key

        s = self.style
        M, LH = s.margin, s.line_height

        # Measure width
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        max_w = s.min_width
        for item in items:
            t = item[0]
            if t in ("text", "field"):
                label = f"{item[1]}: {item[2]}" if len(item) > 2 else item[1]
                bbox = draw.textbbox((0, 0), label, font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)
            elif t == "toggle":
                bbox = draw.textbbox((0, 0), f"[ ] {item[1]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)
            elif t == "dropdown":
                bbox = draw.textbbox((0, 0), f"> {item[1]}: {item[2]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)
            elif t == "slider":
                bbox = draw.textbbox((0, 0), f"[-] {item[1]}: {item[2]:.2f} [+]", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)

        # Count lines including dropdown expansion
        n_lines = len(items)
        dropdown_extra = 0
        max_dd_visible = max(4, (self._fb_height // LH) - n_lines - 4)
        if self._dropdown_open:
            for item in items:
                if item[0] == "dropdown" and item[4] == self._dropdown_open:
                    n_opts = len(item[3])
                    dropdown_extra = min(n_opts, max_dd_visible)
                    if n_opts > max_dd_visible:
                        dropdown_extra += 2

        tw = max_w + M * 2
        th = (n_lines + dropdown_extra) * LH + M * 2

        img = Image.new("RGBA", (tw, th), s.bg_color)
        draw = ImageDraw.Draw(img)
        y = M
        self._widgets = []
        toggle_indent = int(self._font.getlength("[x] ")) + 5 if hasattr(self._font, 'getlength') else M + 35

        for item in items:
            t = item[0]

            if t == "text":
                draw.text((M, y), item[1], fill=s.text_color, font=self._font)
                y += LH

            elif t == "toggle":
                _, label, state, key = item
                indicator = "[x]" if state else "[ ]"
                color = s.toggle_on_color if state else s.toggle_off_color
                draw.text((M, y), indicator, fill=color, font=self._font)
                draw.text((M + toggle_indent, y), label, fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "toggle", key))
                y += LH

            elif t == "dropdown":
                _, label, current, options, key = item
                is_open = self._dropdown_open == key
                arrow = "v" if is_open else ">"
                text = f"{arrow} {label}: {current}"
                draw.text((M, y), text, fill=s.accent_color, font=self._font)
                self._widgets.append((y, y + LH, "dropdown_header", key))
                y += LH
                if is_open:
                    n_opts = len(options)
                    scrollable = n_opts > max_dd_visible
                    scroll = self._dropdown_scroll.get(key, 0)
                    if scrollable:
                        scroll = max(0, min(scroll, n_opts - max_dd_visible))
                        self._dropdown_scroll[key] = scroll
                        arrow_color = s.accent_color if scroll > 0 else (80, 80, 80, 255)
                        draw.text((M + 30, y), f"^ ({scroll} more)", fill=arrow_color, font=self._font)
                        self._widgets.append((y, y + LH, "dropdown_scroll", key, -3))
                        y += LH
                    vis_start = scroll if scrollable else 0
                    vis_end = min(vis_start + max_dd_visible, n_opts)
                    for opt in options[vis_start:vis_end]:
                        bg = s.dropdown_hover if opt == current else s.dropdown_bg
                        draw.rectangle([(M + 10, y), (tw - M, y + LH - 1)], fill=bg)
                        draw.text((M + 15, y), opt, fill=s.text_color, font=self._font)
                        self._widgets.append((y, y + LH, "dropdown_item", key, opt))
                        y += LH
                    if scrollable:
                        remaining = n_opts - vis_end
                        arrow_color = s.accent_color if remaining > 0 else (80, 80, 80, 255)
                        draw.text((M + 30, y), f"v ({remaining} more)", fill=arrow_color, font=self._font)
                        self._widgets.append((y, y + LH, "dropdown_scroll", key, 3))
                        y += LH

            elif t == "slider":
                _, label, value, vmin, vmax, key = item
                btn_w = 25
                text = f"{label}: {value:.2f}"
                draw.rectangle([(M, y), (M + btn_w, y + LH - 1)], fill=s.slider_btn)
                draw.text((M + 5, y), "-", fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "slider_dec", key, vmin, vmax))
                draw.text((M + btn_w + 5, y), text, fill=s.text_color, font=self._font)
                rx = tw - M - btn_w
                draw.rectangle([(rx, y), (tw - M, y + LH - 1)], fill=s.slider_btn)
                draw.text((rx + 5, y), "+", fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "slider_inc", key, vmin, vmax))
                y += LH

            elif t == "field":
                _, label, value, key = item
                active = getattr(self, '_editing', None) == key
                label_text = f"{label}:"
                bbox = draw.textbbox((0, 0), label_text, font=self._font)
                label_w = bbox[2] - bbox[0] + 10
                draw.text((M, y), label_text, fill=s.text_color, font=self._font)
                field_x = M + label_w
                field_bg = s.field_active if active else s.field_bg
                draw.rectangle([(field_x, y), (tw - M, y + LH - 2)], fill=field_bg)
                draw.text((field_x + 8, y), value,
                          fill=s.accent_color if active else s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "field", key))
                y += LH

        # Finalize: compute panel bounds, upload texture, build VAO
        self._panel_w = tw
        self._panel_h = th
        data = img.tobytes()

        fb_w, fb_h = self._fb_width, self._fb_height
        if s.position == "top-right":
            self._panel_x = fb_w - tw - 10
            self._panel_y = 10
        else:
            self._panel_x = 10
            self._panel_y = fb_h - th - 10

        self._upload_panel(tw, th, data)

    def _upload_panel(self, tw, th, data):
        """Upload PIL image to GPU and build vertex data.
        Concrete subclass (WGPU panel mixin) overrides this with the
        wgpu upload path; the base method is a no-op so headless tests
        and any code that constructs a Panel without a backend works.
        """
        pass

    def _hit_test(self, x, y):
        """Convert screen coords to panel-local and find widget. Returns widget tuple or None."""
        lx = x - self._panel_x
        ly = y - self._panel_y
        if lx < 0 or lx > self._panel_w or ly < 0 or ly > self._panel_h:
            if self._dropdown_open:
                self._dropdown_open = None
                return "outside_close"
            return None
        for widget in self._widgets:
            if widget[0] <= ly < widget[1]:
                return widget
        return "inside_miss"

    def _handle_base_click(self, widget, lx):
        """Handle common widget click types. Returns True if handled."""
        wtype = widget[2]
        if wtype == "dropdown_header":
            key = widget[3]
            self._dropdown_open = None if self._dropdown_open == key else key
            return True
        if wtype == "dropdown_scroll":
            key, delta = widget[3], widget[4]
            self._dropdown_scroll[key] = self._dropdown_scroll.get(key, 0) + delta
            return True
        if wtype in ("slider_dec", "slider_inc"):
            key, vmin, vmax = widget[3], widget[4], widget[5]
            if lx < self._panel_w // 3:
                return ("slider_dec", key, vmin, vmax)
            elif lx > self._panel_w * 2 // 3:
                return ("slider_inc", key, vmin, vmax)
            return True  # clicked in middle of slider, consume but do nothing
        return None

    def render(self):
        """Render hook overridden by the wgpu panel mixin."""
        pass

    def release(self):
        self._tex = None

    def on_scroll(self, yoffset):
        """Handle scroll for open dropdowns. Returns True if consumed."""
        if self._dropdown_open:
            key = self._dropdown_open
            delta = -3 if yoffset > 0 else 3
            self._dropdown_scroll[key] = self._dropdown_scroll.get(key, 0) + delta
            return True
        return False


class DevOverlay(Panel):
    """Developer HUD with performance info and interactive toggles."""

    def __init__(self):
        super().__init__(DEV_STYLE)
        self.enabled = False
        self._last_message = ""

    def update(self, renderer, camera, fps, render_mode_name, cmap_name, timings, message, **kwargs):
        if not self.enabled:
            return
        self._camera = camera
        self._last_message = message

        items = []
        n_vis = renderer.n_particles
        n_tot = renderer.n_total
        scale = "log" if renderer.log_scale else "linear"

        smooth_fps = kwargs.get('smooth_fps', 0)
        pid_str = f"  (PID: {smooth_fps:.0f})" if smooth_fps > 0 else ""
        items.append(("text", f"FPS: {fps:.0f}{pid_str}"))
        items.append(("text", f"Particles: {n_vis:,} / {n_tot:,}"))
        items.append(("text", f"LOD: {renderer.lod_pixels}px  Budget: {renderer._subsample_max_per_frame/1e6:.1f}M"))
        items.append(("text", f"Cull: {timings.get('cull',0)*1000:.0f}ms  Upload: {timings.get('upload',0)*1000:.0f}ms  Render: {timings.get('render',0)*1000:.0f}ms"))
        if renderer.log_scale:
            range_str = f"10^{renderer.qty_min:.2f} .. 10^{renderer.qty_max:.2f}"
        else:
            range_str = f"{renderer.qty_min:.3g} .. {renderer.qty_max:.3g}"
        items.append(("text", f"Render: {render_mode_name}  Scale: {scale}"))
        items.append(("text", f"Range: {range_str}"))
        items.append(("text", f"Colormap: {cmap_name}"))
        items.append(("text", f"Pos: ({camera.position[0]:.2f}, {camera.position[1]:.2f}, {camera.position[2]:.2f})"))
        items.append(("text", f"Speed: {camera.speed:.3g}"))
        items.append(("text", ""))

        items.append(("toggle", "Invert Mouse", camera.invert_mouse, "cam:invert_mouse"))
        items.append(("toggle", "Skip Vsync", renderer.skip_vsync, "skip_vsync"))
        items.append(("toggle", "Auto LOD", renderer.auto_lod, "auto_lod"))
        if renderer.auto_lod:
            items.append(("slider", "Target FPS", renderer.target_fps, 1.0, 60.0, "target_fps"))
            items.append(("slider", "LOD Smooth (s)", renderer.auto_lod_smooth, 0.05, 2.0, "auto_lod_smooth"))
            items.append(("slider", "PID Kp", renderer.pid_Kp, 0.0, 10.0, "pid_Kp"))
            items.append(("slider", "PID Ki", renderer.pid_Ki, 0.0, 2.0, "pid_Ki"))
            items.append(("slider", "PID Kd", renderer.pid_Kd, 0.0, 2.0, "pid_Kd"))
        items.append(("slider", "Hsml Scale", renderer.hsml_scale, 0.1, 5.0, "hsml_scale"))

        if message:
            items.append(("text", message))

        self.render_panel(items)

    def render(self):
        if not self.enabled:
            return
        super().render()

    def on_click(self, x, y, renderer):
        if not self.enabled:
            return False
        hit = self._hit_test(x, y)
        if hit is None:
            return False
        if hit == "outside_close" or hit == "inside_miss":
            return True

        widget = hit
        wtype = widget[2]
        lx = x - self._panel_x

        # Common widget handling
        base = self._handle_base_click(widget, lx)
        if base is True:
            return True
        if isinstance(base, tuple):
            # Slider
            action, key, vmin, vmax = base
            cur = getattr(renderer, key, 1.0)
            step = max((vmax - vmin) / 20, 0.01)
            if action == "slider_dec":
                setattr(renderer, key, max(vmin, cur - step))
            else:
                setattr(renderer, key, min(vmax, cur + step))
            return True

        if wtype == "toggle":
            key = widget[3]
            if key.startswith("cam:"):
                attr = key[4:]
                setattr(self._camera, attr, not getattr(self._camera, attr))
            else:
                setattr(renderer, key, not getattr(renderer, key))
            return True

        return True


class UserMenu(Panel):
    """Always-visible user menu with weight field, limits, scale, colorbar."""

    def __init__(self):
        super().__init__(USER_STYLE)
        self.show_colorbar = False
        self._editing = None
        self._edit_buffer = ""
        self._app_ref = None
        self._cbar_tex = None

    def on_key(self, key, action):
        import glfw
        if self._editing is None:
            return False
        if action not in (glfw.PRESS, glfw.REPEAT):
            return True
        if key == glfw.KEY_ESCAPE:
            self._editing = None
            self._edit_buffer = ""
            return True
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            if self._app_ref is not None:
                self._commit_edit(self._app_ref)
            return True
        if key == glfw.KEY_BACKSPACE:
            self._edit_buffer = self._edit_buffer[:-1]
            return True
        return True

    def on_char(self, codepoint, app):
        if self._editing is None:
            return False
        ch = chr(codepoint)
        if ch in "0123456789.eE+-":
            self._edit_buffer += ch
            return True
        if ch in ("\r", "\n"):
            self._commit_edit(app)
            return True
        return False

    def _slot_index(self, key):
        """Return 0 for L: prefix, 1 for C: prefix, None otherwise."""
        if key.startswith("L:"):
            return 0
        if key.startswith("C:"):
            return 1
        return None

    def _handle_slot_dropdown(self, app, slot_idx, field_key, value):
        """Handle a dropdown selection for a composite slot."""
        s = app._slot[slot_idx]
        if field_key == "mode":
            s["mode"] = value
            s["resolve"] = {"SurfaceDensity": 0, "WeightedAverage": 1, "WeightedVariance": 2}[value]
        elif field_key == "weight":
            s["weight"] = value
        elif field_key == "weight2":
            s["weight2"] = value
        elif field_key == "op":
            s["op"] = value
        elif field_key == "data":
            s["data"] = value
        elif field_key == "proj":
            s["proj"] = value
        # Reset limits to trigger auto-range on next _apply_render_mode
        s["min"] = -1.0
        s["max"] = 3.0

    def _commit_edit(self, app):
        if self._editing and self._edit_buffer:
            try:
                val = float(self._edit_buffer)
                slot_idx = self._slot_index(self._editing)
                if slot_idx is not None:
                    field_key = self._editing[2:]
                    if field_key.startswith("log "):
                        field_key = field_key[4:]
                    if "Min" in field_key or "min" in field_key:
                        app._slot[slot_idx]["min"] = val
                    elif "Max" in field_key or "max" in field_key:
                        app._slot[slot_idx]["max"] = val
                elif self._editing == "min":
                    app.renderer.qty_min = val
                elif self._editing == "max":
                    app.renderer.qty_max = val
            except ValueError:
                pass
        self._editing = None
        self._edit_buffer = ""

    def _build_slot_items(self, slot, prefix, all_fields, vf_set, slot_modes, vprojs):
        """Build widget items for a composite slot."""
        items = []
        s = slot
        items.append(("dropdown", f"{prefix}Mode", s["mode"], slot_modes, f"{prefix}mode"))
        items.append(("dropdown", f"{prefix}Weight", s["weight"], all_fields, f"{prefix}weight"))
        if s["mode"] == "SurfaceDensity":
            items.append(("dropdown", f"{prefix}Op", s["op"], self._SD_OPS, f"{prefix}op"))
            items.append(("dropdown", f"{prefix}Field2", s["weight2"], ["None"] + all_fields, f"{prefix}weight2"))
        else:
            items.append(("dropdown", f"{prefix}Data", s["data"], all_fields, f"{prefix}data"))
        # Vector projection
        uses_vec = (s["weight"] in vf_set
                    or (s["mode"] == "SurfaceDensity" and s["weight2"] in vf_set)
                    or (s["mode"] != "SurfaceDensity" and s["data"] in vf_set))
        if uses_vec:
            items.append(("dropdown", f"{prefix}Proj", s["proj"], vprojs, f"{prefix}proj"))
        # Limits
        lo = f"{s['min']:.2f}" if s["log"] else f"{s['min']:.3g}"
        hi = f"{s['max']:.2f}" if s["log"] else f"{s['max']:.3g}"
        if self._editing == f"{prefix}min":
            lo = self._edit_buffer + "_"
        if self._editing == f"{prefix}max":
            hi = self._edit_buffer + "_"
        lbl = "log " if s["log"] else ""
        items.append(("field", f"{prefix}{lbl}Min", lo, f"{prefix}min"))
        items.append(("field", f"{prefix}{lbl}Max", hi, f"{prefix}max"))
        items.append(("toggle", f"{prefix}Log", s["log"], f"{prefix}log"))
        return items

    def update(self, renderer, cmap_name, colormaps,
               sd_fields=None, sd_field="Masses",
               sd_field2="None", sd_op="*", sd_ops=None,
               render_modes=None, render_mode_name="SurfaceDensity",
               wa_data_field="Masses",
               vector_fields=None, vector_projection="LOS", vector_projections=None,
               composite_slots=None):
        self._SD_OPS = sd_ops or ["*"]
        items = []

        # All field dropdowns include vector field names
        vf_set = set(vector_fields or [])
        all_fields = list(sd_fields or [])
        for vf in (vector_fields or []):
            if vf not in all_fields:
                all_fields.append(vf)
        vprojs = vector_projections or ["LOS", "|v|", "|v|^2"]

        if render_modes and len(render_modes) > 1:
            items.append(("dropdown", "Mode", render_mode_name, render_modes, "render_mode"))

        if render_mode_name == "Composite" and composite_slots:
            # Two stacked slot panels
            slot_modes = ["SurfaceDensity", "WeightedAverage", "WeightedVariance"]
            items.append(("text", "--- Lightness ---"))
            items.extend(self._build_slot_items(composite_slots[0], "L:", all_fields, vf_set, slot_modes, vprojs))
            items.append(("text", "--- Color ---"))
            items.extend(self._build_slot_items(composite_slots[1], "C:", all_fields, vf_set, slot_modes, vprojs))
            items.append(("dropdown", "Cmap", cmap_name, colormaps, "colormap"))
            items.append(("toggle", "Colorbar", self.show_colorbar, "colorbar"))
        else:
            # Single-field mode
            if all_fields and len(all_fields) > 1:
                items.append(("dropdown", "Weight", sd_field, all_fields, "sd_field"))
                if render_mode_name == "SurfaceDensity":
                    items.append(("dropdown", "Op", sd_op, sd_ops or ["*"], "sd_op"))
                    items.append(("dropdown", "Field 2", sd_field2, ["None"] + all_fields, "sd_field2"))
                else:
                    items.append(("dropdown", "Data", wa_data_field, all_fields, "wa_data_field"))

            uses_vector = (sd_field in vf_set
                           or (render_mode_name == "SurfaceDensity" and sd_field2 in vf_set)
                           or (render_mode_name != "SurfaceDensity" and wa_data_field in vf_set))
            if uses_vector:
                items.append(("dropdown", "Proj", vector_projection, vprojs, "vector_projection"))
            items.append(("dropdown", "Cmap", cmap_name, colormaps, "colormap"))

            if self._editing == "min":
                lo_display = self._edit_buffer + "_"
            elif renderer.log_scale:
                lo_display = f"{renderer.qty_min:.2f}"
            else:
                lo_display = f"{renderer.qty_min:.3g}"

            if self._editing == "max":
                hi_display = self._edit_buffer + "_"
            elif renderer.log_scale:
                hi_display = f"{renderer.qty_max:.2f}"
            else:
                hi_display = f"{renderer.qty_max:.3g}"

            prefix = "log " if renderer.log_scale else ""
            items.append(("field", f"{prefix}Min", lo_display, "min"))
            items.append(("field", f"{prefix}Max", hi_display, "max"))
            items.append(("toggle", "Log scale", renderer.log_scale, "log_scale"))
            items.append(("toggle", "Colorbar", self.show_colorbar, "colorbar"))

        self._cmap_name = cmap_name
        if render_mode_name == "Composite" and composite_slots:
            # Colorbar uses slot 1 (color channel) limits
            s1 = composite_slots[1]
            self._lo_str = f"{s1['min']:.2f}" if s1["log"] else f"{s1['min']:.3g}"
            self._hi_str = f"{s1['max']:.2f}" if s1["log"] else f"{s1['max']:.3g}"
        else:
            self._lo_str = lo_display.rstrip("_")
            self._hi_str = hi_display.rstrip("_")
        self._renderer = renderer
        self.render_panel(items)

        if self.show_colorbar:
            self._build_colorbar()

    def _build_colorbar(self):
        fb_w, fb_h = self._fb_width, self._fb_height
        LH = self.style.line_height
        cbar_h = max(fb_h // 4, 100)
        cbar_w = 30
        label_pad = 10
        label_w = 200
        total_w = cbar_w + label_pad + label_w
        total_h = cbar_h + LH

        img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cbar_top = LH // 2
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(self._cmap_name)
            for j in range(cbar_h):
                t = 1.0 - j / max(cbar_h - 1, 1)
                rgba = cmap(t)
                c = tuple(int(v * 255) for v in rgba[:3]) + (255,)
                draw.rectangle([(0, cbar_top + j), (cbar_w, cbar_top + j)], fill=c)
        except Exception:
            draw.rectangle([(0, cbar_top), (cbar_w, cbar_top + cbar_h)],
                           fill=(128, 128, 128, 255))
        draw.rectangle([(0, cbar_top), (cbar_w, cbar_top + cbar_h)], outline=self.style.text_color)

        label_x = cbar_w + label_pad
        draw.text((label_x, cbar_top - 4), self._hi_str, fill=self.style.text_color, font=self._font)
        draw.text((label_x, cbar_top + cbar_h - LH + 4), self._lo_str, fill=self.style.text_color, font=self._font)

        # Hand the rendered colorbar PIL image off to the wgpu mixin
        # which actually uploads it to the GPU.
        self._cbar_data = (total_w, total_h, img.tobytes())

    def on_click(self, x, y, app):
        self._app_ref = app
        if self._editing:
            self._commit_edit(app)

        hit = self._hit_test(x, y)
        if hit is None:
            return False
        if hit == "outside_close":
            return True
        if hit == "inside_miss":
            return True

        widget = hit
        wtype = widget[2]

        # Common widget handling
        base = self._handle_base_click(widget, x - self._panel_x)
        if base is True:
            return True

        if wtype == "field":
            key = widget[3]
            self._editing = key
            # Pre-fill edit buffer from the right source
            slot_idx = self._slot_index(key)
            if slot_idx is not None:
                s = app._slot[slot_idx]
                field_key = key[2:]  # strip "L:" or "C:" prefix
                # strip "log " prefix from field key if present
                if field_key.startswith("log "):
                    field_key = field_key[4:]
                if field_key.endswith("Min") or field_key.endswith("min"):
                    self._edit_buffer = f"{s['min']:.4g}"
                elif field_key.endswith("Max") or field_key.endswith("max"):
                    self._edit_buffer = f"{s['max']:.4g}"
            else:
                r = app.renderer
                if key == "min":
                    self._edit_buffer = f"{r.qty_min:.4g}"
                elif key == "max":
                    self._edit_buffer = f"{r.qty_max:.4g}"
            return True

        if wtype == "toggle":
            key = widget[3]
            slot_idx = self._slot_index(key)
            if slot_idx is not None:
                app._slot[slot_idx]["log"] = 1 - app._slot[slot_idx]["log"]
            elif key == "log_scale":
                app.renderer.log_scale = 1 - app.renderer.log_scale
                app._needs_auto_range = True
            elif key == "colorbar":
                self.show_colorbar = not self.show_colorbar
            return True

        if wtype == "dropdown_item":
            key, value = widget[3], widget[4]
            # Composite slot dropdowns
            slot_idx = self._slot_index(key)
            if slot_idx is not None:
                self._handle_slot_dropdown(app, slot_idx, key[2:], value)
            elif key == "render_mode":
                app._render_mode_name = value
                # Set sensible defaults for WeightedVariance
                if value == "WeightedVariance" and app._vector_fields:
                    app._sd_field = "Masses"
                    app._wa_data_field = app._vector_fields[0]  # e.g. "Velocities"
                    app._vector_projection = "LOS"
                app._apply_render_mode()
            elif key == "sd_field":
                app._set_sd_field(value)
            elif key == "sd_field2":
                app._sd_field2 = value
                app._apply_render_mode()
            elif key == "sd_op":
                app._sd_op = value
                if app._sd_field2 != "None":
                    app._apply_render_mode()
            elif key == "wa_data_field":
                app._wa_data_field = value
                app._los_camera_fwd = None
                app._apply_render_mode()
            elif key == "vector_projection":
                app._vector_projection = value
                app._los_camera_fwd = None
                app._apply_render_mode()
            elif key == "colormap":
                from .colormaps import AVAILABLE_COLORMAPS
                idx = AVAILABLE_COLORMAPS.index(value) if value in AVAILABLE_COLORMAPS else 0
                app._cmap_idx = idx
                app._set_colormap(value)
            self._dropdown_open = None
            return True

        return True

    # render() / release() come from Panel; the wgpu mixin overrides
    # render to draw the optional colorbar quad alongside the panel.
