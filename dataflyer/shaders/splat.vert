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

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    vec4 clip_pos = u_proj * view_pos;

    // Point size in pixels: h in view space -> pixels
    float h_clip = in_hsml * u_proj[0][0] / max(abs(clip_pos.w), 1e-10);
    float point_pixels = h_clip * u_viewport_size.x * 0.5;

    // Clamp to [2, 64] pixels (macOS Metal limit is 64)
    float clamped = clamp(point_pixels, 2.0, 64.0);
    gl_PointSize = clamped;
    gl_Position = clip_pos;

    // If point was clamped smaller, the kernel covers fewer pixels but
    // the same mass should be deposited. Pass the effective h that matches
    // the clamped point size so kernel normalization stays correct.
    float effective_h = in_hsml * (clamped / max(point_pixels, 1e-10));

    v_mass = in_mass;
    v_hsml = effective_h;
    v_quantity = in_quantity;
}
