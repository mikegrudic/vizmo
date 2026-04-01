"""Convert matplotlib colormaps to OpenGL 1D textures."""

import numpy as np
import matplotlib.pyplot as plt


AVAILABLE_COLORMAPS = [
    "magma",
    "inferno",
    "viridis",
    "plasma",
    "coolwarm",
    "hot",
    "bone",
    "cividis",
]


def colormap_to_texture_data(name, n=256):
    """Convert a matplotlib colormap to an RGBA uint8 array of shape (n, 4)."""
    cmap = plt.get_cmap(name)
    x = np.linspace(0, 1, n)
    rgba = (cmap(x) * 255).astype(np.uint8)
    return rgba


def create_colormap_texture(ctx, name, n=256):
    """Create a moderngl 1D texture from a matplotlib colormap."""
    data = colormap_to_texture_data(name, n)
    tex = ctx.texture((n, 1), 4, data=data.tobytes())
    tex.filter = (ctx.ffi.LINEAR if hasattr(ctx, "ffi") else 0x2601, 0x2601)  # GL_LINEAR
    return tex


def create_colormap_texture_safe(ctx, name, n=256):
    """Create a moderngl texture (using 2D texture with height=1 as 1D substitute)."""
    data = colormap_to_texture_data(name, n)
    tex = ctx.texture((n, 1), 4, data=data.tobytes())
    tex.filter = (0x2601, 0x2601)  # GL_LINEAR, GL_LINEAR
    tex.repeat_x = False
    tex.repeat_y = False
    return tex
