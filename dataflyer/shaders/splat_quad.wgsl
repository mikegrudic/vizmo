// Instanced quad splat: vertex + fragment for additive accumulation.
// Camera, quad_corner, eval_kernel provided by common.wgsl (prepended at load time).

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
