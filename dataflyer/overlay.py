"""UI overlay panels rendered as PIL images to GPU textures."""

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from dataclasses import dataclass

SHADER_DIR = Path(__file__).parent / "shaders"


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


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
    font_size=42, line_height=56, margin=20, min_width=400,
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

    def __init__(self, ctx, style):
        self.ctx = ctx
        self.style = style
        self._tex = None
        self._prog = ctx.program(
            vertex_shader=_load_shader("text.vert"),
            fragment_shader=_load_shader("text.frag"),
        )
        self._vbo = ctx.buffer(reserve=6 * 4 * 4)
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 2f", "in_position", "in_uv")],
        )
        self._font = _get_font(style.font_size)
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

        # Upload texture
        self._panel_w = tw
        self._panel_h = th
        data = img.tobytes()
        if self._tex is not None:
            self._tex.release()
        self._tex = self.ctx.texture((tw, th), 4, data=data)
        self._tex.filter = (0x2601, 0x2601)

        # Position
        fb_w, fb_h = self._fb_width, self._fb_height
        if s.position == "top-right":
            self._panel_x = fb_w - tw - 10
            self._panel_y = 10
            px_w = tw / fb_w * 2
            px_h = th / fb_h * 2
            x1 = 1.0 - px_w - 0.01
            x2 = 1.0 - 0.01
            y1 = 1.0 - px_h - 0.01
            y2 = 1.0 - 0.01
        else:  # bottom-left
            self._panel_x = 10
            self._panel_y = fb_h - th - 10
            px_w = tw / fb_w * 2
            px_h = th / fb_h * 2
            x1 = -1.0 + 0.01
            x2 = -1.0 + px_w + 0.01
            y1 = -1.0 + 0.01
            y2 = -1.0 + px_h + 0.01

        verts = np.array([
            x1, y1, 0, 1,  x2, y1, 1, 1,  x1, y2, 0, 0,
            x2, y1, 1, 1,  x2, y2, 1, 0,  x1, y2, 0, 0,
        ], dtype=np.float32)
        self._vbo.write(verts.tobytes())

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
        if wtype == "slider_dec":
            if lx < self._panel_w // 3:
                key, vmin, vmax = widget[3], widget[4], widget[5]
                return ("slider_dec", key, vmin, vmax)
        if wtype == "slider_inc":
            if lx > self._panel_w * 2 // 3:
                key, vmin, vmax = widget[3], widget[4], widget[5]
                return ("slider_inc", key, vmin, vmax)
        return None

    def render(self):
        if self._tex is None:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._tex.use(location=0)
        self._prog["u_texture"].value = 0
        self._vao.render(vertices=6)
        self.ctx.disable(moderngl.BLEND)

    def release(self):
        for attr in ("_tex", "_vbo", "_vao", "_prog"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass

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

    def __init__(self, ctx):
        super().__init__(ctx, DEV_STYLE)
        self.enabled = False
        self._last_message = ""

    def update(self, renderer, camera, fps, render_mode_name, cmap_name, timings, message, cull_interval=0.5):
        if not self.enabled:
            return
        self._last_message = message

        items = []
        n_vis = renderer.n_particles + renderer.n_big
        n_tot = renderer.n_total
        scale = "log" if renderer.log_scale else "linear"

        items.append(("text", f"FPS: {fps:.0f}"))
        items.append(("text", f"Particles: {n_vis:,} / {n_tot:,}"))
        items.append(("text", f"LOD: {renderer.lod_pixels}px  Budget: {renderer.max_render_particles/1e6:.1f}M"))
        items.append(("text", f"Cull: {timings.get('cull',0)*1000:.0f}ms  Render: {timings.get('render',0)*1000:.0f}ms  Throttle: {cull_interval*1000:.0f}ms"))
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

        items.append(("toggle", "Tree", renderer.use_tree, "use_tree"))
        items.append(("toggle", "Importance Sampling", renderer.use_importance_sampling, "use_importance_sampling"))
        items.append(("toggle", "Hybrid Rendering", renderer.use_hybrid_rendering, "use_hybrid_rendering"))
        items.append(("toggle", "Quad Rendering", renderer.use_quad_rendering, "use_quad_rendering"))
        items.append(("toggle", "Aniso Summaries", renderer.use_aniso_summaries, "use_aniso_summaries"))
        items.append(("text", ""))

        items.append(("dropdown", "Kernel", renderer.kernel, renderer.KERNELS, "kernel"))
        items.append(("text", ""))

        items.append(("slider", "Summary Scale", renderer.summary_scale, 0.1, 10.0, "summary_scale"))
        items.append(("slider", "Summary Overlap", renderer.summary_overlap, 0.0, 1.0, "summary_overlap"))
        items.append(("slider", "Tree Min N", renderer.tree_min_particles, 0, 1e7, "tree_min_particles"))
        items.append(("slider", "Cull Interval", renderer.cull_interval, 0.0, 5.0, "cull_interval"))
        items.append(("text", f"Aniso splats: {renderer.n_aniso:,}"))

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
            if key == "tree_min_particles":
                renderer._needs_grid_rebuild = True
            return True

        if wtype == "toggle":
            key = widget[3]
            setattr(renderer, key, not getattr(renderer, key))
            return True

        if wtype == "dropdown_item":
            key, value = widget[3], widget[4]
            if key == "kernel":
                renderer.kernel = value
            self._dropdown_open = None
            return True

        return True


class UserMenu(Panel):
    """Always-visible user menu with weight field, limits, scale, colorbar."""

    def __init__(self, ctx):
        super().__init__(ctx, USER_STYLE)
        self.show_colorbar = False
        self._editing = None
        self._edit_buffer = ""
        self._app_ref = None

        # Separate colorbar overlay
        self._cbar_tex = None
        self._cbar_vbo = ctx.buffer(reserve=6 * 4 * 4)
        self._cbar_vao = ctx.vertex_array(
            self._prog,
            [(self._cbar_vbo, "2f 2f", "in_position", "in_uv")],
        )

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

    def _commit_edit(self, app):
        if self._editing and self._edit_buffer:
            try:
                val = float(self._edit_buffer)
                if self._editing == "min":
                    app.renderer.qty_min = val
                elif self._editing == "max":
                    app.renderer.qty_max = val
            except ValueError:
                pass
        self._editing = None
        self._edit_buffer = ""

    def update(self, renderer, cmap_name, colormaps,
               sd_fields=None, sd_field="Masses",
               sd_field2="None", sd_op="*", sd_ops=None):
        items = []

        if sd_fields and len(sd_fields) > 1:
            items.append(("dropdown", "Field", sd_field, sd_fields, "sd_field"))
            items.append(("dropdown", "Op", sd_op, sd_ops or ["*"], "sd_op"))
            items.append(("dropdown", "Field 2", sd_field2, ["None"] + sd_fields, "sd_field2"))
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

        data = img.tobytes()
        if self._cbar_tex is not None:
            self._cbar_tex.release()
        self._cbar_tex = self.ctx.texture((total_w, total_h), 4, data=data)
        self._cbar_tex.filter = (0x2601, 0x2601)

        px_w = total_w / fb_w * 2
        px_h = total_h / fb_h * 2
        x1 = -1.0 + 0.01
        x2 = x1 + px_w
        y1 = -px_h / 2
        y2 = px_h / 2

        verts = np.array([
            x1, y1, 0, 1,  x2, y1, 1, 1,  x1, y2, 0, 0,
            x2, y1, 1, 1,  x2, y2, 1, 0,  x1, y2, 0, 0,
        ], dtype=np.float32)
        self._cbar_vbo.write(verts.tobytes())

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
            r = app.renderer
            if key == "min":
                self._edit_buffer = f"{r.qty_min:.4g}"
            elif key == "max":
                self._edit_buffer = f"{r.qty_max:.4g}"
            return True

        if wtype == "toggle":
            key = widget[3]
            if key == "log_scale":
                app.renderer.log_scale = 1 - app.renderer.log_scale
                app._needs_auto_range = True
            elif key == "colorbar":
                self.show_colorbar = not self.show_colorbar
            return True

        if wtype == "dropdown_item":
            key, value = widget[3], widget[4]
            if key == "sd_field":
                app._set_sd_field(value)
            elif key == "sd_field2":
                app._sd_field2 = value
                app._rebuild_sd_weights()
            elif key == "sd_op":
                app._sd_op = value
                if app._sd_field2 != "None":
                    app._rebuild_sd_weights()
            elif key == "colormap":
                from .colormaps import AVAILABLE_COLORMAPS
                idx = AVAILABLE_COLORMAPS.index(value) if value in AVAILABLE_COLORMAPS else 0
                app._cmap_idx = idx
                app._set_colormap(value)
            self._dropdown_open = None
            return True

        return True

    def render(self):
        if self._tex is None:
            return
        super().render()
        if self.show_colorbar and self._cbar_tex is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._cbar_tex.use(location=0)
            self._prog["u_texture"].value = 0
            self._cbar_vao.render(vertices=6)
            self.ctx.disable(moderngl.BLEND)

    def release(self):
        super().release()
        for attr in ("_cbar_tex", "_cbar_vbo", "_cbar_vao"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
