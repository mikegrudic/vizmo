// Shared kernel functions for splat rendering.
// Select via uniform int u_kernel.

uniform int u_kernel;  // 0: cubic spline, 1: Wendland C2, 2: Gaussian, 3: quartic

float eval_kernel(float r) {
    // 0: Cubic spline (meshoid default) - normalized 2D
    if (u_kernel == 0) {
        if (r > 1.0) return 0.0;
        float k;
        if (r <= 0.5) {
            k = 1.0 - 6.0 * r * r * (1.0 - r);
        } else {
            float a = 1.0 - r;
            k = 2.0 * a * a * a;
        }
        return k * 1.8189136353359467;
    }

    // 1: Wendland C2 - (1-r)^4 * (4r+1), compact support, branchless-ish
    if (u_kernel == 1) {
        if (r > 1.0) return 0.0;
        float a = 1.0 - r;
        float a2 = a * a;
        return a2 * a2 * (4.0 * r + 1.0) * 2.228169203;  // 2D normalization: 7/(pi)
    }

    // 2: Gaussian exp(-4r^2), truncated at r=1 - branchless except discard
    if (u_kernel == 2) {
        if (r > 1.0) return 0.0;
        return exp(-4.0 * r * r) * 1.295;  // approximate 2D normalization
    }

    // 3: Quartic (1-r^2)^2 - very cheap, branchless
    if (u_kernel == 3) {
        if (r > 1.0) return 0.0;
        float a = 1.0 - r * r;
        return a * a * 1.909859;  // 2D normalization: 3/pi * 2 (approx)
    }

    return 0.0;
}
