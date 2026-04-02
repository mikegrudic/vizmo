#version 330

in float v_mass;
in float v_hsml;
in float v_quantity;
in float v_coord_scale;

layout(location = 0) out float out_numerator;
layout(location = 1) out float out_denominator;

uniform int u_kernel;  // 0: cubic spline, 1: Wendland C2, 2: Gaussian, 3: quartic

float eval_kernel(float r) {
    if (r > 1.0) return 0.0;
    if (u_kernel == 1) {
        float a = 1.0 - r; float a2 = a * a;
        return a2 * a2 * (4.0 * r + 1.0) * 2.2281692033;
    }
    if (u_kernel == 2) {
        return exp(-4.0 * r * r) * 1.2969948338;
    }
    if (u_kernel == 3) {
        float a = 1.0 - r * r;
        return a * a * 0.9549296586;
    }
    // 0: cubic spline (default)
    float k;
    if (r <= 0.5) {
        k = 1.0 - 6.0 * r * r * (1.0 - r);
    } else {
        float a = 1.0 - r;
        k = 2.0 * a * a * a;
    }
    return k * 1.8189136353;
}

void main() {
    vec2 coord = (gl_PointCoord - 0.5) * 2.0 * v_coord_scale;
    float r = length(coord);
    if (r > 1.0) discard;

    float w = eval_kernel(r);
    float sigma = v_mass * w / (v_hsml * v_hsml);

    out_numerator = sigma * v_quantity;
    out_denominator = sigma;
}
