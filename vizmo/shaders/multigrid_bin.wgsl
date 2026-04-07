// Multigrid binning compute shader.
//
// Three entry points operating on a single chunk's particle arrays:
//   cs_count    — for each particle in [0, n_to_consider), compute its
//                 multigrid level and atomicAdd(counts[level]).
//   cs_build    — single workgroup, n_levels threads. Computes
//                 bases[level] = sum(counts[<level]), writes the
//                 DrawIndirect args (vertex_count=4, instance_count,
//                 first_vertex=0, first_instance=base) per level, and
//                 zeroes the counts so cs_scatter can re-use them as
//                 per-level write heads.
//   cs_scatter  — for each particle, recompute level, slot =
//                 atomicAdd(counts[level], 1), index_buf[base+slot]=i.
//
// All three share one bind group layout. Counts/bases buffers are
// sized n_levels * 4 bytes (max ~16 levels). Indirect buffer is
// n_levels * 16 bytes.

struct BinParams {
    cam_pos: vec3<f32>,
    _p0: f32,
    cam_fwd: vec3<f32>,
    _p1: f32,
    n_to_consider: u32,
    n_levels: u32,
    full_res_w: f32,
    n_grid_kernel: f32,
    proj11: f32,
    h_scale: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> bp: BinParams;
@group(0) @binding(1) var<storage, read> s_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_hsml: array<f32>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> bases: array<u32>;
@group(0) @binding(5) var<storage, read_write> index_buf: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect: array<u32>;

const MAX_LEVELS: u32 = 16u;

fn level_for(idx: u32) -> u32 {
    let pos = s_pos[idx].xyz;
    let hsml = s_hsml[idx] * bp.h_scale;
    let depth = max(dot(pos - bp.cam_pos, bp.cam_fwd), 1e-20);
    let r_clip = hsml * bp.proj11 / depth;
    let r_px = r_clip * 0.5 * bp.full_res_w;
    let ratio = max(r_px / bp.n_grid_kernel, 1.0);
    let lvl_f = floor(log2(ratio));
    return u32(clamp(lvl_f, 0.0, f32(bp.n_levels - 1u)));
}

var<workgroup> wg_counts: array<atomic<u32>, MAX_LEVELS>;

@compute @workgroup_size(256)
fn cs_count(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x < MAX_LEVELS) {
        atomicStore(&wg_counts[lid.x], 0u);
    }
    workgroupBarrier();
    let i = (wid.y * 65535u + wid.x) * 256u + lid.x;
    if (i < bp.n_to_consider) {
        let lvl = level_for(i);
        atomicAdd(&wg_counts[lvl], 1u);
    }
    workgroupBarrier();
    if (lid.x < bp.n_levels) {
        let c = atomicLoad(&wg_counts[lid.x]);
        if (c > 0u) {
            atomicAdd(&counts[lid.x], c);
        }
    }
}

var<workgroup> wg_offsets: array<atomic<u32>, MAX_LEVELS>;
var<workgroup> wg_local: array<atomic<u32>, MAX_LEVELS>;

@compute @workgroup_size(MAX_LEVELS)
fn cs_build(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Run with dispatch (1,1,1); single workgroup of MAX_LEVELS threads.
    // Thread 0 does the prefix sum + indirect args (n_levels is small,
    // serial is simpler than parallel scan and avoids barriers).
    if (lid.x != 0u) { return; }
    var base: u32 = 0u;
    for (var l: u32 = 0u; l < bp.n_levels; l = l + 1u) {
        let c = atomicLoad(&counts[l]);
        bases[l] = base;
        // DrawIndirect layout: vertex_count, instance_count,
        // first_vertex, first_instance.
        let off = l * 4u;
        indirect[off + 0u] = 4u;       // vertex_count (quad strip)
        indirect[off + 1u] = c;        // instance_count
        indirect[off + 2u] = 0u;       // first_vertex
        indirect[off + 3u] = 0u;       // first_instance (always 0; shader looks up base via s_bases)
        base = base + c;
        atomicStore(&counts[l], 0u);
    }
}

@compute @workgroup_size(256)
fn cs_scatter(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    // Two-pass workgroup-local reservation: each thread atomicAdds into
    // a small workgroup-private histogram, then thread 0 reserves a
    // contiguous slab in the global counts buffer per level. Each thread
    // then writes its index to bases[lvl] + slab_base + local_slot.
    if (lid.x < MAX_LEVELS) {
        atomicStore(&wg_local[lid.x], 0u);
        atomicStore(&wg_offsets[lid.x], 0u);
    }
    workgroupBarrier();

    let i = (wid.y * 65535u + wid.x) * 256u + lid.x;
    var lvl: u32 = 0u;
    var local_slot: u32 = 0u;
    let in_range = i < bp.n_to_consider;
    if (in_range) {
        lvl = level_for(i);
        local_slot = atomicAdd(&wg_local[lvl], 1u);
    }
    workgroupBarrier();

    // One global atomicAdd per (workgroup, level) instead of per particle.
    if (lid.x < bp.n_levels) {
        let c = atomicLoad(&wg_local[lid.x]);
        if (c > 0u) {
            let base = atomicAdd(&counts[lid.x], c);
            atomicStore(&wg_offsets[lid.x], base);
        }
    }
    workgroupBarrier();

    if (in_range) {
        let slab_base = atomicLoad(&wg_offsets[lvl]);
        index_buf[bases[lvl] + slab_base + local_slot] = i;
    }
}
