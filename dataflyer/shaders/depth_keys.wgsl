// Compute depth sort keys for back-to-front or front-to-back rendering.
//
// For each gathered particle, computes depth = dot(pos - cam_pos, cam_fwd)
// and converts to a sortable u32 key (bit trick for float-to-uint sorting).
// Also initializes sort_index[i] = i (identity permutation).

struct DepthParams {
    cam_pos: vec3<f32>,
    _pad0: f32,
    cam_fwd: vec3<f32>,
    n_particles: u32,
};

@group(0) @binding(0) var<uniform> params: DepthParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> depth_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> sort_index: array<u32>;

// Convert f32 depth to a u32 key that sorts in the same order.
// Float bit layout: sign(1) | exponent(8) | mantissa(23)
// For positive floats, the bit pattern is already ordered.
// For negative floats, flip all bits.
// This gives a total order on f32 values as u32 keys.
fn float_to_sort_key(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    // If sign bit is set (negative), flip all bits; else flip just sign bit
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_particles) { return; }

    let pos = positions[i].xyz;
    let depth = dot(pos - params.cam_pos, params.cam_fwd);

    // Back-to-front: negate depth so farther particles sort first
    depth_keys[i] = float_to_sort_key(-depth);
    sort_index[i] = i;
}
