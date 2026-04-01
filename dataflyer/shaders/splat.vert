#version 330

// Per-vertex: quad corners
in vec2 in_corner;  // (-1,-1), (1,-1), (-1,1), (1,1)

// Per-instance: particle data
in vec3 in_position;
in float in_hsml;
in float in_mass;
in float in_quantity;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport_size;  // (width, height) in pixels

out vec2 v_offset;
out float v_mass;
out float v_hsml;
out float v_quantity;

void main() {
    // Transform particle center to view space
    vec4 view_center = u_view * vec4(in_position, 1.0);

    // Clamp billboard to at least 2 pixels in screen space
    vec4 clip_center = u_proj * view_center;
    float min_h = 2.0 * abs(clip_center.w) / (u_proj[0][0] * u_viewport_size.x * 0.5);
    float h = max(in_hsml, min_h);

    // Billboard: offset in camera-aligned x/y (view space)
    vec4 view_pos = view_center;
    view_pos.xy += in_corner * h;

    gl_Position = u_proj * view_pos;

    v_offset = in_corner;
    v_mass = in_mass;
    v_hsml = h;
    v_quantity = in_quantity;
}
