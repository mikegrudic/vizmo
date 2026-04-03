// Gather visible particles from finest-level cells into compacted SoA output buffers.
// Each workgroup handles one visible cell (decision == 3 / EMIT_PARTICLES).
//
// Two stages, each a separate entry point:
//   1. count_particles: count output particles per EMIT cell, compute stride
//   2. gather_particles: copy particles to compacted output using prefix-summed offsets

struct GatherParams {
    nc3: u32,              // total cells at finest level (64^3 = 262144)
    budget: u32,           // max output particles
    total_visible: u32,    // filled by count pass (atomic)
    stride: u32,           // global stride (filled after count pass)
};

@group(0) @binding(0) var<uniform> gather_params: GatherParams;

// Finest-level cell decisions
@group(0) @binding(1) var<storage, read> decision: array<u32>;

// CSR cell_start array
@group(0) @binding(2) var<storage, read> cell_start: array<u32>;

// Per-cell output counts (written by count, prefix-summed, read by gather)
@group(0) @binding(3) var<storage, read_write> cell_out_counts: array<u32>;

// Atomic counter for total visible particles
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
// counters[0] = total visible particles
// counters[1] = total output particles (after stride)

// Source sorted particle data (GPU-resident, uploaded once per snapshot)
@group(1) @binding(0) var<storage, read> src_pos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> src_hsml: array<f32>;
@group(1) @binding(2) var<storage, read> src_mass: array<f32>;
@group(1) @binding(3) var<storage, read> src_qty: array<f32>;

// Output compacted particle data (SoA, same buffers rendered by splat shader)
@group(2) @binding(0) var<storage, read_write> out_pos: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> out_hsml: array<f32>;
@group(2) @binding(2) var<storage, read_write> out_mass: array<f32>;
@group(2) @binding(3) var<storage, read_write> out_qty: array<f32>;

// Pass 1: Count particles per EMIT cell, compute total
@compute @workgroup_size(256)
fn count_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= gather_params.nc3) { return; }

    if (decision[cell] != 3u) {
        cell_out_counts[cell] = 0u;
        return;
    }

    let n = cell_start[cell + 1u] - cell_start[cell];
    cell_out_counts[cell] = n;
    atomicAdd(&counters[0], n);
}

// Pass 2: After CPU reads counters[0], computes stride, writes gather_params.stride,
//          and runs prefix sum on cell_out_counts (divided by stride):
//          This shader then does the actual gather.
@compute @workgroup_size(256)
fn apply_stride(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= gather_params.nc3) { return; }

    let n = cell_out_counts[cell];
    if (n == 0u) { return; }

    let stride = gather_params.stride;
    let kept = (n + stride - 1u) / stride;
    cell_out_counts[cell] = kept;
    atomicAdd(&counters[1], kept);  // exact output total in counters[1]
}

// Pass 3: After prefix sum on cell_out_counts, gather particles
@compute @workgroup_size(256)
fn gather_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= gather_params.nc3) { return; }

    if (decision[cell] != 3u) { return; }

    let start = cell_start[cell];
    let end = cell_start[cell + 1u];
    let n = end - start;
    if (n == 0u) { return; }

    let stride = gather_params.stride;
    let out_start = cell_out_counts[cell];  // now contains prefix-summed offset

    // Rescale factor
    let kept = (n + stride - 1u) / stride;
    let ratio = f32(n) / f32(max(kept, 1u));
    let h_scale = pow(ratio, 0.33333333);

    var j = 0u;
    var k = start;
    while (k < end) {
        let out_idx = out_start + j;
        out_pos[out_idx] = src_pos[k];
        out_hsml[out_idx] = src_hsml[k] * h_scale;
        out_mass[out_idx] = src_mass[k] * ratio;
        out_qty[out_idx] = src_qty[k];
        j += 1u;
        k += stride;
    }
}
