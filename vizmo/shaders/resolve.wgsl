// Fullscreen resolve pass: reads accumulation textures, applies colormap.

struct ResolveParams {
    qty_min: f32,
    qty_max: f32,
    mode: u32,        // 0: surface density, 1: weighted avg, 2: weighted variance
    log_scale: u32,   // 1: log10 mapping, 0: linear
};

@group(0) @binding(0) var<uniform> params: ResolveParams;
@group(0) @binding(1) var t_numerator: texture_2d<f32>;
@group(0) @binding(2) var t_denominator: texture_2d<f32>;
@group(0) @binding(3) var t_sq: texture_2d<f32>;
@group(0) @binding(4) var t_colormap: texture_2d<f32>;
@group(0) @binding(5) var s_colormap: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle.
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
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.position.xy);
    let denom = textureLoad(t_denominator, coords, 0).r;

    if (denom < 1e-30) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    var val: f32;
    if (params.mode == 0u) {
        val = denom;
    } else if (params.mode == 1u) {
        let num = textureLoad(t_numerator, coords, 0).r;
        val = num / denom;
    } else {
        let num = textureLoad(t_numerator, coords, 0).r;
        let sq = textureLoad(t_sq, coords, 0).r;
        let mean = num / denom;
        let mean_sq = sq / denom;
        val = sqrt(max(mean_sq - mean * mean, 0.0));
    }

    var t: f32;
    if (params.log_scale == 1u) {
        let log_val = log(max(val, 1e-30)) / log(10.0);
        t = clamp((log_val - params.qty_min) / (params.qty_max - params.qty_min), 0.0, 1.0);
    } else {
        t = clamp((val - params.qty_min) / (params.qty_max - params.qty_min), 0.0, 1.0);
    }

    let color = textureSampleLevel(t_colormap, s_colormap, vec2<f32>(t, 0.5), 0.0).rgb;
    return vec4<f32>(color, 1.0);
}
