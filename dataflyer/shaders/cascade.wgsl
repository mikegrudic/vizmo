// Multigrid cascade: upsample a coarser accumulation triple into the
// next finer level and additively blend. Run from coarsest to finest.
//
// Values stored in the accum textures are per-pixel *density* samples
// (mass * W / h^2), not per-pixel mass. Bilinear sampling therefore
// reconstructs the density field at fine-pixel centers without any
// rescaling; the additive blend adds the coarse contribution into the
// fine level's own splats.

@group(0) @binding(0) var t_num: texture_2d<f32>;
@group(0) @binding(1) var t_den: texture_2d<f32>;
@group(0) @binding(2) var t_sq:  texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = pos[vi];
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

struct FragmentOutput {
    @location(0) num: vec4<f32>,
    @location(1) den: vec4<f32>,
    @location(2) sq:  vec4<f32>,
};

fn sample_bilinear(t: texture_2d<f32>, uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(t, 0));
    let p = uv * dims - vec2<f32>(0.5, 0.5);
    let ip = floor(p);
    let f = p - ip;
    let x0 = i32(clamp(ip.x, 0.0, dims.x - 1.0));
    let y0 = i32(clamp(ip.y, 0.0, dims.y - 1.0));
    let x1 = i32(clamp(ip.x + 1.0, 0.0, dims.x - 1.0));
    let y1 = i32(clamp(ip.y + 1.0, 0.0, dims.y - 1.0));
    let v00 = textureLoad(t, vec2<i32>(x0, y0), 0).r;
    let v10 = textureLoad(t, vec2<i32>(x1, y0), 0).r;
    let v01 = textureLoad(t, vec2<i32>(x0, y1), 0).r;
    let v11 = textureLoad(t, vec2<i32>(x1, y1), 0).r;
    let v0 = mix(v00, v10, f.x);
    let v1 = mix(v01, v11, f.x);
    return mix(v0, v1, f.y);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.num = vec4<f32>(sample_bilinear(t_num, in.uv), 0.0, 0.0, 0.0);
    out.den = vec4<f32>(sample_bilinear(t_den, in.uv), 0.0, 0.0, 0.0);
    out.sq  = vec4<f32>(sample_bilinear(t_sq,  in.uv), 0.0, 0.0, 0.0);
    return out;
}
