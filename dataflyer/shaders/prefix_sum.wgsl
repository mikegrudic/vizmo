// Parallel exclusive prefix sum (Blelloch scan).
//
// Two-pass approach for arrays up to 256*256 = 65536 elements:
//   Pass 1 (scan_local): Each workgroup scans 256 elements, writes block total to block_sums.
//   Pass 2 (propagate): Add block prefix to all elements in each block.
//
// For the grid (max 64^3 = 262144 cells), we need a 3-pass approach:
//   Pass 1: scan_local on data → block_sums_1 (1024 blocks of 256)
//   Pass 2: scan_local on block_sums_1 → block_sums_2 (4 blocks of 256)
//   Pass 3: propagate block_sums_2 into block_sums_1
//   Pass 4: propagate block_sums_1 into data

const WG_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;

struct ScanParams {
    n: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
};
@group(0) @binding(2) var<uniform> scan_params: ScanParams;

var<workgroup> wg_data: array<u32, 256>;

// Pass 1: Workgroup-local exclusive prefix sum + write block total
@compute @workgroup_size(256)
fn scan_local(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wgid: vec3<u32>) {
    let i = gid.x;
    let local_id = lid.x;

    // Load
    var val: u32 = 0u;
    if (i < scan_params.n) {
        val = data[i];
    }
    wg_data[local_id] = val;
    workgroupBarrier();

    // Up-sweep (reduce)
    for (var stride = 1u; stride < WG_SIZE; stride *= 2u) {
        let idx = (local_id + 1u) * stride * 2u - 1u;
        if (idx < WG_SIZE) {
            wg_data[idx] += wg_data[idx - stride];
        }
        workgroupBarrier();
    }

    // Save block total and clear last element
    if (local_id == WG_SIZE - 1u) {
        block_sums[wgid.x] = wg_data[WG_SIZE - 1u];
        wg_data[WG_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep (distribute)
    for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
        let idx = (local_id + 1u) * stride * 2u - 1u;
        if (idx < WG_SIZE) {
            let temp = wg_data[idx - stride];
            wg_data[idx - stride] = wg_data[idx];
            wg_data[idx] += temp;
        }
        workgroupBarrier();
    }

    // Write back exclusive prefix sum
    if (i < scan_params.n) {
        data[i] = wg_data[local_id];
    }
}

// Pass 2: Add block prefix to all elements
@compute @workgroup_size(256)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(workgroup_id) wgid: vec3<u32>) {
    let i = gid.x;
    if (i < scan_params.n && wgid.x > 0u) {
        data[i] += block_sums[wgid.x];
    }
}
