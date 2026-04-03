// Anisotropic Gaussian summary splat: vertex + fragment.
// Replaces splat_aniso.vert + splat_aniso.frag.

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    kernel_id: u32,
    _pad: u32,
};

struct AnisoParams {
    cov_scale: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> aniso_params: AnisoParams;

// Per-instance data via storage buffers
@group(1) @binding(0) var<storage, read> s_pos: array<vec4<f32>>;   // xyz + pad
@group(1) @binding(1) var<storage, read> s_mass: array<f32>;
@group(1) @binding(2) var<storage, read> s_qty: array<f32>;
@group(1) @binding(3) var<storage, read> s_cov: array<vec4<f32>>;   // 6 floats packed as 2 x vec4 (only first 3 of each used)
// s_cov layout: even indices = (xx, xy, xz, _), odd indices = (yy, yz, zz, _)

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,    // position in sigma units
    @location(1) mass: f32,
    @location(2) quantity: f32,
    @location(3) gauss_norm: f32,      // 1/(2pi * sqrt(det(Sigma_2D)))
};

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    let x = f32(vertex_index & 1u) * 2.0 - 1.0;
    let y = f32((vertex_index >> 1u) & 1u) * 2.0 - 1.0;
    return vec2<f32>(x, y);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOutput {
    let corner = quad_corner(vi);
    let pos = s_pos[ii].xyz;
    let mass = s_mass[ii];
    let qty = s_qty[ii];

    // Reconstruct 3D covariance from packed storage
    let cov_a = s_cov[ii * 2u];      // xx, xy, xz
    let cov_b = s_cov[ii * 2u + 1u]; // yy, yz, zz
    let c_xx = cov_a.x; let c_xy = cov_a.y; let c_xz = cov_a.z;
    let c_yy = cov_b.x; let c_yz = cov_b.y; let c_zz = cov_b.z;

    let view_center = camera.view * vec4<f32>(pos, 1.0);

    // View rotation (upper-left 3x3 of view matrix)
    let R = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );

    // Rotate covariance to view space: Sigma_view = R * S * R^T
    let S = mat3x3<f32>(
        vec3<f32>(c_xx, c_xy, c_xz),
        vec3<f32>(c_xy, c_yy, c_yz),
        vec3<f32>(c_xz, c_yz, c_zz),
    );
    let cov_view = R * S * transpose(R);

    // Marginalize over z: take upper-left 2x2
    var s_xx = max(cov_view[0][0], 1e-8);
    var s_xy = cov_view[0][1];
    var s_yy = max(cov_view[1][1], 1e-8);

    // Eigendecomposition of 2x2 symmetric matrix
    let trace = s_xx + s_yy;
    let diff = s_xx - s_yy;
    let disc = sqrt(max(diff * diff + 4.0 * s_xy * s_xy, 0.0));
    let lambda1 = max(0.5 * (trace + disc), 1e-8) * aniso_params.cov_scale;
    let lambda2 = max(0.5 * (trace - disc), 1e-8) * aniso_params.cov_scale;

    // Eigenvector for lambda1
    var ev1: vec2<f32>;
    if (abs(s_xy) > 1e-10) {
        ev1 = normalize(vec2<f32>(lambda1 / aniso_params.cov_scale - s_yy, s_xy));
    } else {
        if (s_xx >= s_yy) {
            ev1 = vec2<f32>(1.0, 0.0);
        } else {
            ev1 = vec2<f32>(0.0, 1.0);
        }
    }
    let ev2 = vec2<f32>(-ev1.y, ev1.x);

    // 3-sigma ellipse radii in view space
    let r1 = 3.0 * sqrt(lambda1);
    let r2 = 3.0 * sqrt(lambda2);

    // Offset in clip space
    let clip_center = camera.proj * view_center;
    let view_offset = corner.x * ev1 * r1 + corner.y * ev2 * r2;

    var out: VertexOutput;
    out.position = clip_center;
    out.position.x += view_offset.x * camera.proj[0][0];
    out.position.y += view_offset.y * camera.proj[1][1];
    out.offset = corner * 3.0;
    out.mass = mass;
    out.quantity = qty;
    out.gauss_norm = 1.0 / (6.2831853 * sqrt(lambda1 * lambda2));
    return out;
}

struct FragmentOutput {
    @location(0) numerator: vec4<f32>,
    @location(1) denominator: vec4<f32>,
    @location(2) sq: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let r2 = dot(in.offset, in.offset);
    if (r2 > 9.0) { discard; }

    let sigma = in.mass * in.gauss_norm * exp(-0.5 * r2);

    var out: FragmentOutput;
    out.numerator = vec4<f32>(sigma * in.quantity, 0.0, 0.0, 0.0);
    out.denominator = vec4<f32>(sigma, 0.0, 0.0, 0.0);
    out.sq = vec4<f32>(sigma * in.quantity * in.quantity, 0.0, 0.0, 0.0);
    return out;
}
