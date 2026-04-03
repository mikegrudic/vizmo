// Project a 3D vector field along the line of sight.
// qty[i] = dot(vec[i], cam_fwd)
// Also computes mass-weighted qty for moment updates:
//   sorted_qty[i] = dot(vec[i], cam_fwd)  (written to particle qty buffer)

struct LosParams {
    cam_fwd: vec3<f32>,
    n: u32,
    dispatch_width: u32,  // workgroups in x dimension
    _p1: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<uniform> params: LosParams;
@group(0) @binding(1) var<storage, read> vec_field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> qty_out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 2D dispatch to handle >65535 workgroups
    let i = gid.x + gid.y * params.dispatch_width * 256u;
    if (i >= params.n) { return; }

    let v = vec_field[i].xyz;
    qty_out[i] = dot(v, params.cam_fwd);
}
