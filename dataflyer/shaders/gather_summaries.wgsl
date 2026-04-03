// Gather summary splats from one grid level where decision == SUMMARY (1).
// Uses atomic counter for output offset — no prefix sum needed.

struct SummaryParams {
    nc3: u32,
    summary_overlap: f32,
    cs_x2: f32,
    cs_y2: f32,
    cs_z2: f32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<uniform> params: SummaryParams;
@group(0) @binding(1) var<storage, read> decision: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
// counters[0] = global output offset (atomicAdd to claim a slot)

// Level source data
@group(1) @binding(0) var<storage, read> src_com: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> src_hsml: array<f32>;
@group(1) @binding(2) var<storage, read> src_mass: array<f32>;
@group(1) @binding(3) var<storage, read> src_qty: array<f32>;
@group(1) @binding(4) var<storage, read> src_cov: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read> src_mh2: array<f32>;

// Output summary arrays
@group(2) @binding(0) var<storage, read_write> out_pos: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> out_mass: array<f32>;
@group(2) @binding(2) var<storage, read_write> out_qty: array<f32>;
@group(2) @binding(3) var<storage, read_write> out_cov: array<vec4<f32>>;

@compute @workgroup_size(256)
fn gather_summaries(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.nc3) { return; }
    if (decision[cell] != 1u) { return; }

    let mass = src_mass[cell];
    if (mass <= 0.0) { return; }

    // Claim output slot via atomic counter
    let out_idx = atomicAdd(&counters[0], 1u);

    out_pos[out_idx] = src_com[cell];
    out_mass[out_idx] = mass;
    out_qty[out_idx] = src_qty[cell];

    let safe_mass = max(mass, 1e-30);
    let mean_h2 = src_mh2[cell] / safe_mass;
    let kernel_pad = 0.225 * mean_h2;
    let alpha = params.summary_overlap;

    var cov_a = src_cov[cell * 2u];
    var cov_b = src_cov[cell * 2u + 1u];

    cov_a.x += kernel_pad + alpha * params.cs_x2;
    cov_b.x += kernel_pad + alpha * params.cs_y2;
    cov_b.z += kernel_pad + alpha * params.cs_z2;

    out_cov[out_idx * 2u] = cov_a;
    out_cov[out_idx * 2u + 1u] = cov_b;
}
