#version 330

in vec2 in_position;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_position * 0.5 + 0.5;
}
