"""Dev overlay: renders text HUD with interactive toggles and dropdowns."""

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shaders"

# Layout constants
MARGIN = 8
LINE_H = 20
FONT_SIZE = 14
BG_COLOR = (0, 0, 0, 200)
TEXT_COLOR = (0, 255, 0, 255)
HIGHLIGHT_COLOR = (80, 80, 80, 255)
TOGGLE_ON_COLOR = (0, 200, 0, 255)
TOGGLE_OFF_COLOR = (150, 50, 50, 255)
DROPDOWN_BG = (40, 40, 40, 255)
DROPDOWN_HOVER = (80, 80, 120, 255)
SLIDER_BG = (50, 50, 50, 255)
SLIDER_FG = (0, 150, 200, 255)
SLIDER_BTN = (80, 80, 80, 255)


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


class DevOverlay:
    def __init__(self, ctx):
        self.ctx = ctx
        self.enabled = False
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
        self._font = self._get_font(FONT_SIZE)

        # Interactive state
        self._widgets = []  # list of (y_start, y_end, type, callback, state_key)
        self._dropdown_open = None  # key of currently open dropdown
        self._dropdown_items = {}  # key -> list of options
        self._fb_width = 1
        self._fb_height = 1
        self._panel_x = 0  # pixel x of panel left edge
        self._panel_y = 0  # pixel y of panel top edge
        self._panel_w = 0
        self._panel_h = 0
        self._last_message = ""

    def _get_font(self, size):
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

    def update(self, renderer, camera, fps, current_qty, cmap_name, timings, message, cull_interval=0.5):
        """Build the overlay image with interactive widgets."""
        if not self.enabled:
            return

        self._last_message = message

        # Collect display items
        items = []  # list of (type, label, ...) tuples
        self._widgets = []

        n_vis = renderer.n_particles + renderer.n_big
        n_tot = renderer.n_total
        scale = "log" if renderer.log_scale else "linear"

        # Info lines (non-interactive)
        items.append(("text", f"FPS: {fps:.0f}"))
        items.append(("text", f"Particles: {n_vis:,} / {n_tot:,}"))
        items.append(("text", f"LOD: {renderer.lod_pixels}px  Budget: {renderer.max_render_particles/1e6:.1f}M"))
        items.append(("text", f"Cull: {timings.get('cull',0)*1000:.0f}ms  Render: {timings.get('render',0)*1000:.0f}ms  Throttle: {cull_interval*1000:.0f}ms"))
        if renderer.log_scale:
            range_str = f"10^{renderer.qty_min:.2f} .. 10^{renderer.qty_max:.2f}"
        else:
            range_str = f"{renderer.qty_min:.3g} .. {renderer.qty_max:.3g}"
        items.append(("text", f"Quantity: {current_qty}  Scale: {scale}"))
        items.append(("text", f"Range: {range_str}"))
        items.append(("text", f"Colormap: {cmap_name}"))
        items.append(("text", f"Pos: ({camera.position[0]:.2f}, {camera.position[1]:.2f}, {camera.position[2]:.2f})"))
        items.append(("text", f"Speed: {camera.speed:.3g}"))
        items.append(("text", ""))

        # Toggles
        items.append(("toggle", "Tree", renderer.use_tree, "tree"))
        items.append(("toggle", "Importance Sampling", renderer.use_importance_sampling, "importance"))
        items.append(("toggle", "Hybrid Rendering", renderer.use_hybrid_rendering, "hybrid"))
        items.append(("toggle", "Quad Rendering", renderer.use_quad_rendering, "quad"))
        items.append(("toggle", "Aniso Summaries", renderer.use_aniso_summaries, "aniso"))
        items.append(("text", ""))

        # Dropdown
        items.append(("dropdown", "Kernel", renderer.kernel, renderer.KERNELS, "kernel"))
        items.append(("text", ""))

        # Sliders
        items.append(("slider", "Summary Scale", renderer.summary_scale, 0.1, 10.0, "summary_scale"))
        items.append(("slider", "Summary Overlap", renderer.summary_overlap, 0.0, 1.0, "summary_overlap"))
        items.append(("slider", "Cull Interval", renderer.cull_interval, 0.0, 5.0, "cull_interval"))
        items.append(("text", f"Aniso splats: {renderer.n_aniso:,}"))

        if message:
            items.append(("text", message))

        # Render to image
        self._render_items(items)

    def _render_items(self, items):
        # Measure width
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        max_w = 200
        for item in items:
            if item[0] == "text":
                bbox = draw.textbbox((0, 0), item[1], font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + MARGIN * 4)
            elif item[0] == "toggle":
                bbox = draw.textbbox((0, 0), f"[ ] {item[1]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + MARGIN * 4)
            elif item[0] == "dropdown":
                bbox = draw.textbbox((0, 0), f"{item[1]}: {item[2]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + MARGIN * 4 + 20)
            elif item[0] == "slider":
                bbox = draw.textbbox((0, 0), f"[-] {item[1]}: {item[2]:.2f} [+]", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + MARGIN * 4)

        # Count lines including dropdown expansion
        n_lines = len(items)
        dropdown_extra = 0
        if self._dropdown_open:
            for item in items:
                if item[0] == "dropdown" and item[4] == self._dropdown_open:
                    dropdown_extra = len(item[3])

        tw = max_w + MARGIN * 2
        th = (n_lines + dropdown_extra) * LINE_H + MARGIN * 2

        img = Image.new("RGBA", (tw, th), BG_COLOR)
        draw = ImageDraw.Draw(img)

        y = MARGIN
        self._widgets = []

        for item in items:
            if item[0] == "text":
                draw.text((MARGIN, y), item[1], fill=TEXT_COLOR, font=self._font)
                y += LINE_H

            elif item[0] == "toggle":
                _, label, state, key = item
                indicator = "[x]" if state else "[ ]"
                color = TOGGLE_ON_COLOR if state else TOGGLE_OFF_COLOR
                draw.text((MARGIN, y), indicator, fill=color, font=self._font)
                draw.text((MARGIN + 35, y), label, fill=TEXT_COLOR, font=self._font)
                self._widgets.append((y, y + LINE_H, "toggle", key))
                y += LINE_H

            elif item[0] == "dropdown":
                _, label, current, options, key = item
                is_open = self._dropdown_open == key
                arrow = "v" if is_open else ">"
                text = f"{arrow} {label}: {current}"
                draw.text((MARGIN, y), text, fill=TEXT_COLOR, font=self._font)
                self._widgets.append((y, y + LINE_H, "dropdown_header", key))
                y += LINE_H

                if is_open:
                    for opt in options:
                        bg = DROPDOWN_HOVER if opt == current else DROPDOWN_BG
                        draw.rectangle([(MARGIN + 10, y), (tw - MARGIN, y + LINE_H - 1)], fill=bg)
                        draw.text((MARGIN + 15, y), opt, fill=TEXT_COLOR, font=self._font)
                        self._widgets.append((y, y + LINE_H, "dropdown_item", key, opt))
                        y += LINE_H

            elif item[0] == "slider":
                _, label, value, vmin, vmax, key = item
                # Draw: [-] Label: value [+]
                btn_w = 25
                text = f"{label}: {value:.2f}"
                # Left button
                draw.rectangle([(MARGIN, y), (MARGIN + btn_w, y + LINE_H - 1)], fill=SLIDER_BTN)
                draw.text((MARGIN + 5, y), "-", fill=TEXT_COLOR, font=self._font)
                self._widgets.append((y, y + LINE_H, "slider_dec", key, vmin, vmax))
                # Label + value
                draw.text((MARGIN + btn_w + 5, y), text, fill=TEXT_COLOR, font=self._font)
                # Right button
                rx = tw - MARGIN - btn_w
                draw.rectangle([(rx, y), (tw - MARGIN, y + LINE_H - 1)], fill=SLIDER_BTN)
                draw.text((rx + 5, y), "+", fill=TEXT_COLOR, font=self._font)
                self._widgets.append((y, y + LINE_H, "slider_inc", key, vmin, vmax))
                y += LINE_H

        # Upload texture
        self._panel_w = tw
        self._panel_h = th
        data = img.tobytes()
        if self._tex is not None:
            self._tex.release()
        self._tex = self.ctx.texture((tw, th), 4, data=data)
        self._tex.filter = (0x2601, 0x2601)

        # Position in top-right
        fb_w, fb_h = self._fb_width, self._fb_height
        self._panel_x = fb_w - tw - 10
        self._panel_y = 10

        px_w = tw / fb_w * 2
        px_h = th / fb_h * 2
        x1 = 1.0 - px_w - 0.01
        x2 = 1.0 - 0.01
        y1 = 1.0 - px_h - 0.01
        y2 = 1.0 - 0.01

        verts = np.array([
            x1, y1, 0, 1,
            x2, y1, 1, 1,
            x1, y2, 0, 0,
            x2, y1, 1, 1,
            x2, y2, 1, 0,
            x1, y2, 0, 0,
        ], dtype=np.float32)
        self._vbo.write(verts.tobytes())

    def set_framebuffer_size(self, w, h):
        self._fb_width = w
        self._fb_height = h

    def on_click(self, x, y, renderer):
        """Handle mouse click. x, y in framebuffer pixels from top-left.
        Returns True if the click was consumed by the overlay."""
        if not self.enabled:
            return False

        # Convert to panel-local coordinates
        lx = x - self._panel_x
        ly = y - self._panel_y

        if lx < 0 or lx > self._panel_w or ly < 0 or ly > self._panel_h:
            # Click outside panel -- close any open dropdown
            if self._dropdown_open:
                self._dropdown_open = None
                return True
            return False

        for widget in self._widgets:
            wy_start, wy_end = widget[0], widget[1]
            if ly < wy_start or ly >= wy_end:
                continue

            wtype = widget[2]

            if wtype == "toggle":
                key = widget[3]
                if key == "tree":
                    renderer.use_tree = not renderer.use_tree
                elif key == "importance":
                    renderer.use_importance_sampling = not renderer.use_importance_sampling
                elif key == "hybrid":
                    renderer.use_hybrid_rendering = not renderer.use_hybrid_rendering
                elif key == "quad":
                    renderer.use_quad_rendering = not renderer.use_quad_rendering
                elif key == "aniso":
                    renderer.use_aniso_summaries = not renderer.use_aniso_summaries
                return True

            elif wtype == "dropdown_header":
                key = widget[3]
                if self._dropdown_open == key:
                    self._dropdown_open = None
                else:
                    self._dropdown_open = key
                return True

            elif wtype == "dropdown_item":
                key = widget[3]
                value = widget[4]
                if key == "kernel":
                    renderer.kernel = value
                self._dropdown_open = None
                return True

            elif wtype == "slider_dec":
                if lx < self._panel_w // 3:
                    key = widget[3]
                    vmin, vmax = widget[4], widget[5]
                    cur = getattr(renderer, key, 1.0)
                    step = max((vmax - vmin) / 20, 0.01)
                    setattr(renderer, key, max(vmin, cur - step))
                    return True

            elif wtype == "slider_inc":
                if lx > self._panel_w * 2 // 3:
                    key = widget[3]
                    vmin, vmax = widget[4], widget[5]
                    cur = getattr(renderer, key, 1.0)
                    step = max((vmax - vmin) / 20, 0.01)
                    setattr(renderer, key, min(vmax, cur + step))
                    return True

        return True  # clicked inside panel but not on a widget

    def render(self):
        if not self.enabled or self._tex is None:
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


# User menu constants (3x larger than dev overlay)
UM_FONT_SIZE = 42
UM_LINE_H = 56
UM_MARGIN = 20
UM_BG = (15, 15, 25, 100)
UM_TEXT = (220, 220, 220, 255)
UM_ACCENT = (100, 180, 255, 255)
UM_FIELD_BG = (30, 30, 30, 255)
UM_FIELD_ACTIVE = (50, 50, 80, 255)
UM_BTN_HOVER = (80, 100, 140, 255)


class UserMenu:
    """Always-visible user menu with quantity, editable limits, scale, colorbar."""

    def __init__(self, ctx):
        self.ctx = ctx
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
        self._font = DevOverlay._get_font(None, UM_FONT_SIZE)

        # Separate colorbar overlay
        self._cbar_tex = None
        self._cbar_vbo = ctx.buffer(reserve=6 * 4 * 4)
        self._cbar_vao = ctx.vertex_array(
            self._prog,
            [(self._cbar_vbo, "2f 2f", "in_position", "in_uv")],
        )

        self._widgets = []
        self._dropdown_open = None
        self._fb_width = 1
        self._fb_height = 1
        self._panel_x = 0
        self._panel_y = 0
        self._panel_w = 0
        self._panel_h = 0
        self.show_colorbar = False

        # Editable field state
        self._editing = None  # "min" or "max" or None
        self._edit_buffer = ""
        self._app_ref = None  # set during on_click for Enter key commit

    def set_framebuffer_size(self, w, h):
        self._fb_width = w
        self._fb_height = h

    def on_key(self, key, action):
        """Handle keyboard input for editable fields. Returns True if consumed."""
        import glfw
        if self._editing is None:
            return False
        # When editing, consume ALL keys to prevent app shortcuts
        if action not in (glfw.PRESS, glfw.REPEAT):
            return True
        if key == glfw.KEY_ESCAPE:
            self._editing = None
            self._edit_buffer = ""
            return True
        if key == glfw.KEY_ENTER or key == glfw.KEY_KP_ENTER:
            if self._app_ref is not None:
                self._commit_edit(self._app_ref)
            return True
        if key == glfw.KEY_BACKSPACE:
            self._edit_buffer = self._edit_buffer[:-1]
            return True
        return True  # consume all other keys while editing

    def on_char(self, codepoint, app):
        """Handle character input for editable fields. Returns True if consumed."""
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
        """Commit the current edit buffer to the renderer."""
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

    def update(self, renderer, quantities, current_qty, cmap_name, colormaps):
        items = []
        self._widgets = []

        items.append(("dropdown", "Map", current_qty, quantities, "quantity"))
        items.append(("dropdown", "Cmap", cmap_name, colormaps, "colormap"))

        # Editable range fields
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
        self._render_menu(items)

    def _render_menu(self, items):
        M = UM_MARGIN
        LH = UM_LINE_H
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        max_w = 400
        for item in items:
            if item[0] in ("text", "field"):
                bbox = draw.textbbox((0, 0), f"{item[1]}: {item[2]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)
            elif item[0] == "toggle":
                bbox = draw.textbbox((0, 0), f"[ ] {item[1]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)
            elif item[0] == "dropdown":
                bbox = draw.textbbox((0, 0), f"> {item[1]}: {item[2]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)

        n_lines = len(items)
        dropdown_extra = 0
        if self._dropdown_open:
            for item in items:
                if item[0] == "dropdown" and item[4] == self._dropdown_open:
                    dropdown_extra = len(item[3])

        tw = max_w + M * 2
        th = (n_lines + dropdown_extra) * LH + M * 2

        img = Image.new("RGBA", (tw, th), UM_BG)
        draw = ImageDraw.Draw(img)
        y = M
        self._widgets = []

        for item in items:
            if item[0] == "text":
                draw.text((M, y), item[1], fill=UM_TEXT, font=self._font)
                y += LH

            elif item[0] == "field":
                _, label, value, key = item
                active = self._editing == key
                # Label
                label_text = f"{label}:"
                bbox = draw.textbbox((0, 0), label_text, font=self._font)
                label_w = bbox[2] - bbox[0] + 10
                draw.text((M, y), label_text, fill=UM_TEXT, font=self._font)
                # Input field
                field_x = M + label_w
                field_bg = UM_FIELD_ACTIVE if active else UM_FIELD_BG
                draw.rectangle([(field_x, y), (tw - M, y + LH - 2)], fill=field_bg)
                draw.text((field_x + 8, y), value, fill=UM_ACCENT if active else UM_TEXT, font=self._font)
                self._widgets.append((y, y + LH, "field", key))
                y += LH

            elif item[0] == "toggle":
                _, label, state, key = item
                indicator = "[x]" if state else "[ ]"
                color = UM_ACCENT if state else UM_TEXT
                draw.text((M, y), indicator, fill=color, font=self._font)
                draw.text((M + 90, y), label, fill=UM_TEXT, font=self._font)
                self._widgets.append((y, y + LH, "toggle", key))
                y += LH

            elif item[0] == "dropdown":
                _, label, current, options, key = item
                is_open = self._dropdown_open == key
                arrow = "v" if is_open else ">"
                text = f"{arrow} {label}: {current}"
                draw.text((M, y), text, fill=UM_ACCENT, font=self._font)
                self._widgets.append((y, y + LH, "dropdown_header", key))
                y += LH
                if is_open:
                    for opt in options:
                        bg = UM_BTN_HOVER if opt == current else UM_FIELD_BG
                        draw.rectangle([(M + 20, y), (tw - M, y + LH - 2)], fill=bg)
                        draw.text((M + 30, y), opt, fill=UM_TEXT, font=self._font)
                        self._widgets.append((y, y + LH, "dropdown_item", key, opt))
                        y += LH

        # Build separate colorbar overlay if enabled
        if self.show_colorbar:
            self._build_colorbar()

        self._panel_w = tw
        self._panel_h = th
        data = img.tobytes()
        if self._tex is not None:
            self._tex.release()
        self._tex = self.ctx.texture((tw, th), 4, data=data)
        self._tex.filter = (0x2601, 0x2601)

        # Position in bottom-left
        fb_w, fb_h = self._fb_width, self._fb_height
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

    def _build_colorbar(self):
        """Build colorbar as a separate transparent overlay, centered on left side, 1/4 window height."""
        fb_w, fb_h = self._fb_width, self._fb_height
        LH = UM_LINE_H
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
        draw.rectangle([(0, cbar_top), (cbar_w, cbar_top + cbar_h)], outline=UM_TEXT)

        label_x = cbar_w + label_pad
        draw.text((label_x, cbar_top - 4), self._hi_str, fill=UM_TEXT, font=self._font)
        draw.text((label_x, cbar_top + cbar_h - LH + 4), self._lo_str, fill=UM_TEXT, font=self._font)

        data = img.tobytes()
        if self._cbar_tex is not None:
            self._cbar_tex.release()
        self._cbar_tex = self.ctx.texture((total_w, total_h), 4, data=data)
        self._cbar_tex.filter = (0x2601, 0x2601)

        # Position: centered vertically on left edge
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
        """Handle click. Returns True if consumed."""
        self._app_ref = app
        # Commit any pending edit on click elsewhere
        if self._editing:
            self._commit_edit(app)

        lx = x - self._panel_x
        ly = y - self._panel_y

        if lx < 0 or lx > self._panel_w or ly < 0 or ly > self._panel_h:
            if self._dropdown_open:
                self._dropdown_open = None
                return True
            return False

        for widget in self._widgets:
            wy_start, wy_end = widget[0], widget[1]
            if ly < wy_start or ly >= wy_end:
                continue
            wtype = widget[2]

            if wtype == "field":
                key = widget[3]
                self._editing = key
                # Pre-fill with current value
                r = app.renderer
                if key == "min":
                    self._edit_buffer = f"{r.qty_min:.4g}"
                elif key == "max":
                    self._edit_buffer = f"{r.qty_max:.4g}"
                return True

            elif wtype == "toggle":
                key = widget[3]
                if key == "log_scale":
                    app.renderer.log_scale = 1 - app.renderer.log_scale
                    app._needs_auto_range = True
                elif key == "colorbar":
                    self.show_colorbar = not self.show_colorbar
                return True

            elif wtype == "dropdown_header":
                key = widget[3]
                if self._dropdown_open == key:
                    self._dropdown_open = None
                else:
                    self._dropdown_open = key
                return True

            elif wtype == "dropdown_item":
                key = widget[3]
                value = widget[4]
                if key == "quantity":
                    idx = app._quantities.index(value) if value in app._quantities else 0
                    app._qty_idx = idx
                    app._current_qty = value
                    q = app.data.get_quantity(value)
                    app.renderer.update_quantity(q)
                    app.renderer.mode = 0 if value == "surface_density" else 1
                    app._needs_auto_range = True
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
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._tex.use(location=0)
        self._prog["u_texture"].value = 0
        self._vao.render(vertices=6)
        if self.show_colorbar and self._cbar_tex is not None:
            self._cbar_tex.use(location=0)
            self._prog["u_texture"].value = 0
            self._cbar_vao.render(vertices=6)
        self.ctx.disable(moderngl.BLEND)

    def release(self):
        for attr in ("_tex", "_cbar_tex", "_vbo", "_cbar_vbo", "_vao", "_cbar_vao", "_prog"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
