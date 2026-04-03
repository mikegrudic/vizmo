// Instanced quad splat: vertex + fragment for additive accumulation.
// Replaces splat_quad.vert + splat_quad.frag + splat_additive.frag (point sprites).
// All particles rendered as instanced quads (no gl_PointSize in WebGPU).

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    kernel_id: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;

// Per-instance particle data via storage buffers (SoA layout)
@group(1) @binding(0) var<storage, read> s_pos: array<vec4<f32>>;   // xyz + pad
@group(1) @binding(1) var<storage, read> s_hsml: array<f32>;
@group(1) @binding(2) var<storage, read> s_mass: array<f32>;
@group(1) @binding(3) var<storage, read> s_qty: array<f32>;

// Optional sort index for depth-sorted rendering (group 2)
@group(2) @binding(0) var<storage, read> sort_index: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) mass: f32,
    @location(2) hsml: f32,
    @location(3) quantity: f32,
};

// Quad corners: vertex_index 0..3 maps to (-1,-1), (1,-1), (-1,1), (1,1)
fn quad_corner(vertex_index: u32) -> vec2<f32> {
    let x = f32(vertex_index & 1u) * 2.0 - 1.0;
    let y = f32((vertex_index >> 1u) & 1u) * 2.0 - 1.0;
    return vec2<f32>(x, y);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOutput {
    let corner = quad_corner(vi);
    // Use sort_index for depth-sorted rendering; identity when unsorted.
    // camera._pad (last u32 of the uniform) encodes use_sort flag via kernel_id field.
    let idx = sort_index[ii];
    let pos = s_pos[idx].xyz;
    let hsml = s_hsml[idx];
    let mass = s_mass[idx];
    let qty = s_qty[idx];

    let view_center = camera.view * vec4<f32>(pos, 1.0);
    let clip_center = camera.proj * view_center;

    var out: VertexOutput;
    out.position = clip_center;
    out.position.x += corner.x * hsml * camera.proj[0][0];
    out.position.y += corner.y * hsml * camera.proj[1][1];
    out.offset = corner;
    out.mass = mass;
    out.hsml = hsml;
    out.quantity = qty;
    return out;
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

struct FragmentOutput {
    @location(0) numerator: vec4<f32>,
    @location(1) denominator: vec4<f32>,
    @location(2) sq: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let r = length(in.offset);
    if (r > 1.0) { discard; }

    let w = eval_kernel(r, camera.kernel_id);
    let sigma = in.mass * w / (in.hsml * in.hsml);

    var out: FragmentOutput;
    out.numerator = vec4<f32>(sigma * in.quantity, 0.0, 0.0, 0.0);
    out.denominator = vec4<f32>(sigma, 0.0, 0.0, 0.0);
    out.sq = vec4<f32>(sigma * in.quantity * in.quantity, 0.0, 0.0, 0.0);
    return out;
}
