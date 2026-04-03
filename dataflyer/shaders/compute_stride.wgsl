// Compute stride from total_visible and budget, entirely on GPU.
// Eliminates the CPU readback of total_visible.
// Single thread — dispatched as (1,1,1).

struct StrideParams {
    nc3: u32,
    budget: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> params: StrideParams;
@group(0) @binding(1) var<storage, read_write> counters: array<u32>;
// counters[0] = total_visible (written by count_particles)
// counters[1] = n_output (written by apply_stride)
// counters[2] = stride (written here, read by apply_stride and gather)

@compute @workgroup_size(1)
fn main() {
    let total = counters[0];
    var stride = 1u;
    if (total > params.budget) {
        stride = max(1u, total / params.budget);
    }
    counters[2] = stride;
}
