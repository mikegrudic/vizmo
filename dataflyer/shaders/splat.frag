#version 330

in vec2 v_offset;
in float v_mass;
in float v_hsml;
in float v_quantity;

uniform sampler2D u_colormap;  // 2D texture with height=1 (1D not supported on macOS Metal)
uniform float u_qty_min;      // log10(min)
uniform float u_qty_max;      // log10(max)
uniform float u_alpha_scale;  // overall opacity multiplier
uniform int u_mode;           // 0: surface density, 1: quantity-colored

out vec4 frag_color;

void main() {
    float r = length(v_offset);
    if (r > 1.0) discard;

    // Gaussian-like kernel: exp(-4r^2), compact support at r=1
    float w = exp(-4.0 * r * r);

    // Surface density contribution of this particle at this pixel
    float sigma = v_mass * w / (v_hsml * v_hsml);

    // Choose what value to map to color
    float val;
    if (u_mode == 0) {
        // Surface density mode: color by accumulated sigma
        val = sigma;
    } else {
        // Quantity mode: color by the particle's quantity value
        val = v_quantity;
    }

    // Map to colormap via log scale
    float log_val = log(max(val, 1e-30)) / log(10.0);
    float t = clamp((log_val - u_qty_min) / (u_qty_max - u_qty_min), 0.0, 1.0);
    vec3 color = texture(u_colormap, vec2(t, 0.5)).rgb;

    // Alpha from surface density contribution
    float alpha = clamp(sigma * u_alpha_scale, 0.0, 0.95);

    // Premultiplied alpha output
    frag_color = vec4(color * alpha, alpha);
}
