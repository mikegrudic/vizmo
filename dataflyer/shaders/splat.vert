#version 330

// Per-vertex: quad corners
in vec2 in_corner;  // (-1,-1), (1,-1), (-1,1), (1,1)

// Per-instance: particle data (from SSBOs or VBOs via instancing)
in vec3 in_position;
in float in_hsml;
in float in_mass;
in float in_quantity;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec2 v_offset;
out float v_mass;
out float v_hsml;
out float v_quantity;

void main() {
    // Transform particle to view space
    vec4 view_pos = u_view * vec4(in_position, 1.0);

    // Billboard: offset in camera-aligned x/y by smoothing length
    view_pos.xy += in_corner * in_hsml;

    gl_Position = u_proj * view_pos;

    v_offset = in_corner;
    v_mass = in_mass;
    v_hsml = in_hsml;
    v_quantity = in_quantity;
}
