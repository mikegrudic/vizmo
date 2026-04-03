// Star particle rendering as instanced quads.
// Camera, quad_corner provided by common.wgsl (prepended at load time).

struct StarParams {
    point_size: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> star_params: StarParams;

@group(1) @binding(0) var<storage, read> s_pos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> s_mass: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coord: vec2<f32>,
    @location(1) mass: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOutput {
    let corner = quad_corner(vi);
    let pos = s_pos[ii].xyz;
    let mass = s_mass[ii];

    let clip_pos = camera.proj * camera.view * vec4<f32>(pos, 1.0);
    let size = star_params.point_size / clip_pos.w;

    var out: VertexOutput;
    out.position = clip_pos;
    // Offset in NDC proportional to size
    out.position.x += corner.x * size * clip_pos.w / camera.viewport_size.x * 2.0;
    out.position.y += corner.y * size * clip_pos.w / camera.viewport_size.y * 2.0;
    out.coord = corner;
    out.mass = mass;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r = length(in.coord);
    if (r > 1.0) { discard; }

    let glow = exp(-3.0 * r * r);
    let color = vec3<f32>(1.0, 0.95, 0.8) * glow;
    let alpha = glow * 0.9;

    return vec4<f32>(color * alpha, alpha);
}
