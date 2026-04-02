#version 330

// Per-vertex: one vertex per particle (point sprite)
in vec3 in_position;
in float in_hsml;
in float in_mass;
in float in_quantity;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport_size;  // (width, height) in pixels

out float v_mass;
out float v_hsml;
out float v_quantity;
out float v_coord_scale;  // ratio of desired/actual point size for PointCoord rescaling

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    vec4 clip_pos = u_proj * view_pos;

    // Point size in pixels: h in view space -> pixels
    float h_clip = in_hsml * u_proj[0][0] / max(abs(clip_pos.w), 1e-10);
    float desired_pixels = h_clip * u_viewport_size.x * 0.5;

    // Hardware clamps to [1, 64] on macOS Metal
    float actual_pixels = clamp(desired_pixels, 2.0, 64.0);
    gl_PointSize = actual_pixels;
    gl_Position = clip_pos;

    // If clamped, the fragment shader needs to rescale gl_PointCoord
    // so the kernel maps correctly over the full desired size.
    // coord_scale > 1 means the point is smaller than desired.
    v_coord_scale = desired_pixels / actual_pixels;

    v_mass = in_mass;
    v_hsml = in_hsml;
    v_quantity = in_quantity;
}
