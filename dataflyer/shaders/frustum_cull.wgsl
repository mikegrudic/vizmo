// Frustum cull + LOD decision for one grid level.
// Dispatched once per level, coarsest to finest.
//
// For the coarsest level: tests frustum planes.
// For child levels: only processes cells whose parent was marked REFINE.
// At the finest level: REFINE cells become EMIT (gather particles).
//
// Per-cell decision: 0=HIDDEN, 1=SUMMARY, 2=REFINE, 3=EMIT_PARTICLES

struct CullParams {
    cam_pos: vec3<f32>,
    _pad0: f32,
    cam_fwd: vec3<f32>,
    _pad1: f32,
    cam_right: vec3<f32>,
    _pad2: f32,
    cam_up: vec3<f32>,
    _pad3: f32,
    fov_rad: f32,
    aspect: f32,
    pix_per_rad: f32,
    lod_pixels: f32,
    is_coarsest: u32,     // 1 if this is the coarsest level
    is_finest: u32,       // 1 if this is the finest level
    parent_nc: u32,       // cells per side at parent level (0 if coarsest)
    nc: u32,              // cells per side at this level
};

@group(0) @binding(0) var<uniform> params: CullParams;

// Grid level data for this level
@group(0) @binding(1) var<storage, read> cell_mass: array<f32>;
@group(0) @binding(2) var<storage, read> cell_hsml: array<f32>;
@group(0) @binding(3) var<storage, read> cell_centers: array<vec4<f32>>;  // xyz + half_diag in w

// Parent decision buffer (read) — only used when is_coarsest == 0
@group(0) @binding(4) var<storage, read> parent_decision: array<u32>;

// This level's decision buffer (write)
@group(0) @binding(5) var<storage, read_write> decision: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nc = params.nc;
    let nc3 = nc * nc * nc;
    if (idx >= nc3) { return; }

    // Default: hidden
    decision[idx] = 0u;

    let mass = cell_mass[idx];
    if (mass <= 0.0) { return; }

    // If not coarsest: check if parent was REFINE
    if (params.is_coarsest == 0u) {
        let pnc = params.parent_nc;
        let ix = idx / (nc * nc);
        let iy = (idx / nc) % nc;
        let iz = idx % nc;
        let px = ix / 2u;
        let py = iy / 2u;
        let pz = iz / 2u;
        let parent_idx = px * pnc * pnc + py * pnc + pz;
        if (parent_decision[parent_idx] != 2u) { return; }
    }

    let center = cell_centers[idx].xyz;
    let half_diag = cell_centers[idx].w;
    let hsml = cell_hsml[idx];

    // Camera-relative coordinates
    let depth = dot(center - params.cam_pos, params.cam_fwd);
    let right_dist = dot(center - params.cam_pos, params.cam_right);
    let up_dist = dot(center - params.cam_pos, params.cam_up);

    // Frustum test (coarsest level does full test; children inherit parent's frustum pass)
    if (params.is_coarsest == 1u) {
        let cell_extent = half_diag + min(hsml, 2.0 * half_diag);
        let half_tan = tan(params.fov_rad * 0.5);
        let front_depth = max(depth + cell_extent, 0.0);
        let lim_h = front_depth * half_tan * params.aspect + cell_extent;
        let lim_v = front_depth * half_tan + cell_extent;
        let in_front = depth > -cell_extent;
        let in_frustum = in_front && (abs(right_dist) < lim_h) && (abs(up_dist) < lim_v);
        if (!in_frustum) { return; }
    }

    // LOD criterion: screen-space size
    let dist = sqrt(depth * depth + right_dist * right_dist + up_dist * up_dist);
    let safe_dist = max(dist, 0.01);
    let h_pix = hsml / safe_dist * params.pix_per_rad;

    if (params.is_finest == 1u) {
        // Finest level: always emit particles (summary gather skips level 0,
        // so SUMMARY cells here would be silently dropped)
        decision[idx] = 3u;  // EMIT_PARTICLES
    } else if (h_pix <= params.lod_pixels) {
        decision[idx] = 1u;  // SUMMARY
    } else {
        decision[idx] = 2u;  // REFINE (go to children)
    }
}
