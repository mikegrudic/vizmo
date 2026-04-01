#version 330

in float v_mass;

out vec4 frag_color;

void main() {
    // Distance from center of point sprite
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r = length(coord);
    if (r > 1.0) discard;

    // Bright glowing core with falloff
    float glow = exp(-3.0 * r * r);

    // Warm white-yellow star color
    vec3 color = vec3(1.0, 0.95, 0.8) * glow;
    float alpha = glow * 0.9;

    frag_color = vec4(color * alpha, alpha);
}
