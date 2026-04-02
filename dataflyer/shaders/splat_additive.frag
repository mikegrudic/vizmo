#version 330

// Additive blending pass using point sprites.
// gl_PointCoord gives [0,1] within the point square.

in float v_mass;
in float v_hsml;
in float v_quantity;

layout(location = 0) out float out_numerator;
layout(location = 1) out float out_denominator;

float cubic_spline_2d(float q) {
    float k;
    if (q <= 0.5) {
        k = 1.0 - 6.0 * q * q * (1.0 - q);
    } else if (q <= 1.0) {
        float a = 1.0 - q;
        k = 2.0 * a * a * a;
    } else {
        return 0.0;
    }
    return k * 1.8189136353359467;
}

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r = length(coord);
    if (r > 1.0) discard;

    float w = cubic_spline_2d(r);
    float sigma = v_mass * w / (v_hsml * v_hsml);

    out_numerator = sigma * v_quantity;
    out_denominator = sigma;
}
