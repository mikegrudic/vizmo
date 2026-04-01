#version 330

in vec3 in_position;
in float in_mass;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_point_size;

out float v_mass;

void main() {
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
    gl_PointSize = u_point_size / gl_Position.w;
    v_mass = in_mass;
}
