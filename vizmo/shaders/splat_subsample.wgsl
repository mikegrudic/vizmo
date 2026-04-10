// Compute-driven splat: vertex shader reads source particle arrays directly
// and performs hash-stride subsample + frustum cull inline. Particles that
// fail either test emit a degenerate quad (clipped behind the near plane).
// Surviving particles emit a normal additive splat with mass scaled by
// `stride` and hsml scaled by `stride^(1/3)`.
//
// One draw call per chunk; source bindings are the same per-chunk SoA
// arrays the legacy compute cull pipeline used. No output buffer, no
// atomic counter, no readback.

struct SubsampleParams {
    cam_pos: vec3<f32>,
    _p0: f32,
    cam_fwd: vec3<f32>,
    _p1: f32,
    cam_right: vec3<f32>,
    _p2: f32,
    cam_up: vec3<f32>,
    _p3: f32,
    // Low-order half of the camera position in the offset-shifted
    // frame, in the DSFUN90-style hi/lo split. The shader reconstructs
    // (pos - cam) as ((pos_hi - cam_pos) + (pos_lo - cam_pos_lo)),
    // recovering ~14 digits of precision so sub-f32-ULP camera moves
    // and particle positions are not quantized.
    cam_pos_lo: vec3<f32>,
    _p4: f32,
    fov_rad: f32,
    aspect: f32,
    stride: u32,
    n_in_chunk: u32,
    h_scale: f32,
    mass_scale: f32,
    level_index: u32,
    n_levels: u32,
    full_res_w: f32,
    n_grid_kernel: f32,
    perspective_correction: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> sub: SubsampleParams;

@group(1) @binding(0) var<storage, read> s_pos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> s_hsml: array<f32>;
@group(1) @binding(2) var<storage, read> s_mass: array<f32>;
@group(1) @binding(3) var<storage, read> s_qty: array<f32>;
@group(1) @binding(4) var<storage, read> s_index: array<u32>;
@group(1) @binding(5) var<storage, read> s_bases: array<u32>;
// Low-order half of the particle position (DSFUN90 hi/lo split).
@group(1) @binding(6) var<storage, read> s_pos_lo: array<vec4<f32>>;
// Per-particle spatial brick assignment (survives the global shuffle).
@group(1) @binding(7) var<storage, read> s_brick_id: array<u32>;
// Per-brick visibility flag written by the brick_cull compute pass.
@group(1) @binding(8) var<storage, read> s_brick_vis: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) mass: f32,
    @location(2) hsml: f32,
    @location(3) quantity: f32,
    // View-space position of the particle, for perspective-correct
    // impact-parameter computation in the fragment shader.
    @location(4) view_pos: vec3<f32>,
};

fn degenerate(corner: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    out.offset = corner;
    out.mass = 0.0;
    out.hsml = 0.0;
    out.quantity = 0.0;
    out.view_pos = vec3<f32>(0.0, 0.0, -1.0);
    return out;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOutput {
    let corner = quad_corner(vi);

    // Source arrays are pre-shuffled on upload (gpu_compute. The caller
    // dispatches K_chunk instances per chunk, where K_chunk is a fraction
    // of n_in_chunk equal to the global budget fraction. The first
    // K_chunk shuffled particles ARE the random K-subset — no hash, no
    // collisions, no source-order imprinting. Out-of-range indices (when
    // K_chunk doesn't divide n_in_chunk) emit degenerate quads.
    // ii indexes the per-chunk index buffer. For non-multigrid mode the
    // index buffer is identity, so this collapses to local_idx = ii.
    // For multigrid mode the binning compute pass scatters particle
    // indices into [base[level], base[level]+count[level]) ranges and
    // each per-level draw_indirect dispatches with first_instance=base
    // and instance_count=count.
    // Fast path for n_levels==1: skip s_index/s_bases entirely so the
    // splat shader has the same memory traffic as the pre-multigrid
    // version (the auto-LOD PID is tuned against that cost curve and
    // oscillates if the vertex cost shifts even a little).
    var local_idx: u32;
    let base = s_bases[sub.level_index];
    let abs_ii = base + ii;
    if (abs_ii >= sub.n_in_chunk) {
        return degenerate(corner);
    }
    local_idx = s_index[abs_ii];

    let pos_hi = s_pos[local_idx].xyz;
    let pos_lo = s_pos_lo[local_idx].xyz;
    // Camera-relative offset reconstructed via DSFUN90 hi/lo
    // cancellation: (pos_hi - cam_hi) cancels the large box-extent
    // magnitude in f32, leaving a small residual to which the lo
    // correction is added.
    let rel = (pos_hi - sub.cam_pos) + (pos_lo - sub.cam_pos_lo);
    let hsml = s_hsml[local_idx] * sub.h_scale;

    // Frustum test (matches compute cull). Use scaled hsml so enlarged
    // kernels at the edge of the frustum aren't dropped.
    let depth = dot(rel, sub.cam_fwd);
    let right_d = dot(rel, sub.cam_right);
    let up_d = dot(rel, sub.cam_up);
    let cell_extent = hsml;
    let half_tan = tan(sub.fov_rad * 0.5);
    let front_depth = max(depth + cell_extent, 0.0);
    let lim_h = front_depth * half_tan * sub.aspect + cell_extent;
    let lim_v = front_depth * half_tan + cell_extent;
    let in_front = depth > -cell_extent;
    if (!in_front || abs(right_d) >= lim_h || abs(up_d) >= lim_v) {
        return degenerate(corner);
    }

    let mass = s_mass[local_idx] * sub.mass_scale;
    let qty = s_qty[local_idx];

    // view = R * T(-cam) in the offset-shifted frame, so
    // view * pos == R * (pos - cam) == R * rel. We pass R as
    // camera.view_rot (the same view matrix with the translation
    // column zeroed), and feed it the precision-preserving rel.
    let view_center = camera.view_rot * vec4<f32>(rel, 1.0);
    let clip_center = camera.proj * view_center;

    var out: VertexOutput;
    out.position = clip_center;
    out.position.x += corner.x * hsml * camera.proj[0][0];
    out.position.y += corner.y * hsml * camera.proj[1][1];
    out.offset = corner;
    out.mass = mass;
    out.hsml = hsml;
    out.quantity = qty;
    out.view_pos = view_center.xyz;
    return out;
}

struct FragmentOutput {
    @location(0) numerator: vec4<f32>,
    @location(1) denominator: vec4<f32>,
    @location(2) sq: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var r: f32;
    if (sub.perspective_correction != 0u) {
        // Perspective-correct impact parameter.
        // Reconstruct the fragment's view-space ray direction from the
        // particle's view position and the quad offset, NOT from
        // gl_FragCoord (which is resolution-dependent and breaks
        // multigrid where each level has a different texture size).
        //
        // The quad offset (ox, oy) ∈ [-1,1] maps to a view-space
        // displacement of (ox*h, oy*h, 0) at the particle's depth,
        // so the fragment's view position is approximately:
        //   F = P + (ox*h, oy*h, 0)
        // The ray through F has direction d̂ = normalize(F), and the
        // impact parameter is b = |P × d̂|.
        let frag_view = in.view_pos + vec3<f32>(
            in.offset.x * in.hsml,
            in.offset.y * in.hsml,
            0.0
        );
        let d_hat = normalize(frag_view);
        let cross_v = cross(in.view_pos, d_hat);
        let b = length(cross_v);
        r = b / in.hsml;
    } else {
        r = length(in.offset);
    }
    if (r > 1.0) { discard; }

    let w = eval_kernel(r, camera.kernel_id);
    let sigma = in.mass * w / (in.hsml * in.hsml);

    var out: FragmentOutput;
    out.numerator = vec4<f32>(sigma * in.quantity, 0.0, 0.0, 0.0);
    out.denominator = vec4<f32>(sigma, 0.0, 0.0, 0.0);
    out.sq = vec4<f32>(sigma * in.quantity * in.quantity, 0.0, 0.0, 0.0);
    return out;
}
