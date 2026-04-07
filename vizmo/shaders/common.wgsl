// Shared WGSL definitions: Camera struct, quad_corner, kernel evaluation.
// Included via Python-side concatenation in _load_wgsl_with_includes().

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    kernel_id: u32,
    _pad: u32,
};

// Quad corners: vertex_index 0..3 maps to (-1,-1), (1,-1), (-1,1), (1,1)
fn quad_corner(vertex_index: u32) -> vec2<f32> {
    let x = f32(vertex_index & 1u) * 2.0 - 1.0;
    let y = f32((vertex_index >> 1u) & 1u) * 2.0 - 1.0;
    return vec2<f32>(x, y);
}

// Kernel evaluation (matches kernels.glsl)
fn eval_kernel(r: f32, kernel_id: u32) -> f32 {
    if (r > 1.0) { return 0.0; }

    // Wendland C2
    if (kernel_id == 1u) {
        let a = 1.0 - r;
        let a2 = a * a;
        return a2 * a2 * (4.0 * r + 1.0) * 2.2281692033;
    }
    // Gaussian
    if (kernel_id == 2u) {
        return exp(-4.0 * r * r) * 1.2969948338;
    }
    // Quartic
    if (kernel_id == 3u) {
        let a = 1.0 - r * r;
        return a * a * 0.9549296586;
    }
    // Sphere
    if (kernel_id == 4u) {
        return sqrt(1.0 - r * r) * 0.4774648293;
    }
    // Cubic spline (default, kernel_id == 0)
    var k: f32;
    if (r <= 0.5) {
        k = 1.0 - 6.0 * r * r * (1.0 - r);
    } else {
        let a = 1.0 - r;
        k = 2.0 * a * a * a;
    }
    return k * 1.8189136353;
}
