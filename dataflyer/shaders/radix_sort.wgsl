// GPU radix sort on (key, index) pairs.
// 32-bit keys, 4-bit radix (16 bins), 8 passes.
//
// Each pass sorts by one 4-bit digit:
//   1. build_histogram: count keys per (digit, workgroup) pair
//   2. prefix_sum on histogram (done externally via prefix_sum.wgsl)
//   3. scatter: stable reorder using local prefix sums per digit
//
// Histogram layout: digit-major, histogram[digit * n_workgroups + wg_id]

const WG_SIZE: u32 = 256u;
const RADIX: u32 = 16u;

struct SortParams {
    n: u32,
    bit_offset: u32,
    n_workgroups: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> indices_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> indices_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> histogram: array<atomic<u32>>;

var<workgroup> local_hist: array<atomic<u32>, 16>;

// Stage 1: Build per-workgroup histogram
@compute @workgroup_size(256)
fn build_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    if (lid.x < RADIX) {
        atomicStore(&local_hist[lid.x], 0u);
    }
    workgroupBarrier();

    if (gid.x < params.n) {
        let digit = (keys_in[gid.x] >> params.bit_offset) & 0xFu;
        atomicAdd(&local_hist[digit], 1u);
    }
    workgroupBarrier();

    if (lid.x < RADIX) {
        let count = atomicLoad(&local_hist[lid.x]);
        atomicStore(&histogram[lid.x * params.n_workgroups + wgid.x], count);
    }
}

// Stage 3: Stable scatter using local prefix sums.
// Each thread computes its rank within its digit bucket in the workgroup
// by counting how many preceding threads in the workgroup have the same digit.

// Shared memory for digits and local offsets
var<workgroup> wg_digits: array<u32, 256>;
var<workgroup> wg_digit_counts: array<atomic<u32>, 16>;  // running count per digit

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    // Clear per-digit counters
    if (lid.x < RADIX) {
        atomicStore(&wg_digit_counts[lid.x], 0u);
    }
    workgroupBarrier();

    // Each thread determines its digit
    var my_digit: u32 = RADIX;  // sentinel for out-of-bounds threads
    var my_key: u32 = 0u;
    var my_idx: u32 = 0u;
    if (gid.x < params.n) {
        my_key = keys_in[gid.x];
        my_idx = indices_in[gid.x];
        my_digit = (my_key >> params.bit_offset) & 0xFu;
    }

    // Compute stable local rank: process threads in order by iterating
    // through local IDs sequentially. Each thread with a valid digit
    // atomicAdds its digit counter to get a unique local rank.
    // We process in chunks to maintain ordering.
    wg_digits[lid.x] = my_digit;
    workgroupBarrier();

    // Serial prefix within workgroup to get stable rank
    // Process all threads with lid < current in order
    var local_rank: u32 = 0u;
    if (my_digit < RADIX) {
        // Count how many earlier threads in this workgroup have the same digit
        for (var i = 0u; i < lid.x; i++) {
            if (wg_digits[i] == my_digit) {
                local_rank += 1u;
            }
        }
    }
    workgroupBarrier();

    // Write to output using global offset + local rank
    if (my_digit < RADIX) {
        let global_offset = atomicLoad(&histogram[my_digit * params.n_workgroups + wgid.x]);
        let dest = global_offset + local_rank;
        keys_out[dest] = my_key;
        indices_out[dest] = my_idx;
    }
}
