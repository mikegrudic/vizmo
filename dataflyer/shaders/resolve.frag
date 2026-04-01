#version 330

// Fullscreen resolve pass.
// mode 0 (surface density): display sigma from denominator texture with colormap
// mode 1 (weighted quantity): divide numerator by denominator, apply colormap

in vec2 v_uv;

uniform sampler2D u_numerator;
uniform sampler2D u_denominator;
uniform sampler2D u_colormap;
uniform float u_qty_min;      // min of display range (log10 or linear)
uniform float u_qty_max;      // max of display range
uniform float u_alpha_scale;
uniform int u_mode;            // 0: surface density, 1: weighted quantity
uniform int u_log_scale;       // 1: log10 mapping, 0: linear mapping

out vec4 frag_color;

void main() {
    float denom = texture(u_denominator, v_uv).r;

    if (denom < 1e-30) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float val;
    if (u_mode == 0) {
        val = denom;  // surface density
    } else {
        float num = texture(u_numerator, v_uv).r;
        val = num / denom;  // mass-weighted average
    }

    // Map to colormap
    float t;
    if (u_log_scale == 1) {
        float log_val = log(max(val, 1e-30)) / log(10.0);
        t = clamp((log_val - u_qty_min) / (u_qty_max - u_qty_min), 0.0, 1.0);
    } else {
        t = clamp((val - u_qty_min) / (u_qty_max - u_qty_min), 0.0, 1.0);
    }
    vec3 color = texture(u_colormap, vec2(t, 0.5)).rgb;

    frag_color = vec4(color, 1.0);
}
