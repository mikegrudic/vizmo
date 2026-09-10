// Fullscreen pass: read num + den from accum textures, write per-pixel
// mean = num/den to a single-channel target. Used by the variance-mode
// two-pass renderer so the second splat pass can accumulate
// sigma * (qty - mean_pixel)^2 directly, avoiding the catastrophic
// cancellation in (mean_sq - mean^2) when bulk qty dominates the
// dispersion. Pixels with no coverage (den < tiny) emit 0; the splat
// path skips them via its frustum test, so writing 0 here is safe.

@group(0) @binding(0) var t_numerator: texture_2d<f32>;
@group(0) @binding(1) var t_denominator: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.position.xy);
    let den = textureLoad(t_denominator, coords, 0).r;
    let num = textureLoad(t_numerator, coords, 0).r;
    var mean: f32 = 0.0;
    if (den > 1e-30) {
        mean = num / den;
    }
    return vec4<f32>(mean, 0.0, 0.0, 0.0);
}
