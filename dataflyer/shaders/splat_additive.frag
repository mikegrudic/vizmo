#version 330

// Additive blending pass for weighted-quantity rendering.
// Outputs to a 2-attachment FBO:
//   attachment 0: sum(mass * quantity * W / h^2)  (numerator)
//   attachment 1: sum(mass * W / h^2)             (denominator = surface density)

in vec2 v_offset;
in float v_mass;
in float v_hsml;
in float v_quantity;

layout(location = 0) out float out_numerator;
layout(location = 1) out float out_denominator;

void main() {
    float r = length(v_offset);
    if (r > 1.0) discard;

    float w = exp(-4.0 * r * r);
    float sigma = v_mass * w / (v_hsml * v_hsml);

    out_numerator = sigma * v_quantity;
    out_denominator = sigma;
}
