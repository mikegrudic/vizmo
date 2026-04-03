// Star particle rendering as instanced quads (replaces star.vert + star.frag).

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    kernel_id: u32,
    _pad: u32,
};

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
