// Composite resolve: two fields blended via HSV.
// Field 1 controls brightness (lightness), field 2 controls hue via colormap.
// Replaces composite.frag.

struct CompositeParams {
    min1: f32,
    max1: f32,
    mode1: u32,      // 0: surface density, 1: weighted avg, 2: variance
    log1: u32,
    min2: f32,
    max2: f32,
    mode2: u32,
    log2: u32,
};

@group(0) @binding(0) var<uniform> params: CompositeParams;

// Field 1 textures
@group(0) @binding(1) var t_num1: texture_2d<f32>;
@group(0) @binding(2) var t_den1: texture_2d<f32>;
@group(0) @binding(3) var t_sq1: texture_2d<f32>;

// Field 2 textures
@group(0) @binding(4) var t_num2: texture_2d<f32>;
@group(0) @binding(5) var t_den2: texture_2d<f32>;
@group(0) @binding(6) var t_sq2: texture_2d<f32>;

// Colormap
@group(0) @binding(7) var t_colormap: texture_2d<f32>;
@group(0) @binding(8) var s_colormap: sampler;

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
    return out;
}

fn resolve_value(num: f32, den: f32, sq: f32, mode: u32) -> f32 {
    if (den < 1e-30) { return -1e30; }
    if (mode == 0u) { return den; }
    if (mode == 1u) { return num / den; }
    // mode 2: variance
    let mean = num / den;
    let mean_sq = sq / den;
    return sqrt(max(mean_sq - mean * mean, 0.0));
}

fn normalize_value(val: f32, vmin: f32, vmax: f32, log_scale: u32) -> f32 {
    if (log_scale == 1u) {
        let lv = log(max(val, 1e-30)) / log(10.0);
        return clamp((lv - vmin) / (vmax - vmin), 0.0, 1.0);
    }
    return clamp((val - vmin) / (vmax - vmin), 0.0, 1.0);
}

// RGB <-> HSV
fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let cmax = max(c.r, max(c.g, c.b));
    let cmin = min(c.r, min(c.g, c.b));
    let delta = cmax - cmin;
    var h: f32 = 0.0;
    if (delta > 0.0) {
        if (cmax == c.r) {
            h = ((c.g - c.b) / delta) % 6.0;
        } else if (cmax == c.g) {
            h = (c.b - c.r) / delta + 2.0;
        } else {
            h = (c.r - c.g) / delta + 4.0;
        }
        h /= 6.0;
        if (h < 0.0) { h += 1.0; }
    }
    let s = select(0.0, delta / cmax, cmax > 0.0);
    return vec3<f32>(h, s, cmax);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let h = c.x * 6.0;
    let s = c.y;
    let v = c.z;
    let i = floor(h);
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let hi = u32(i) % 6u;
    if (hi == 0u) { return vec3<f32>(v, t, p); }
    if (hi == 1u) { return vec3<f32>(q, v, p); }
    if (hi == 2u) { return vec3<f32>(p, v, t); }
    if (hi == 3u) { return vec3<f32>(p, q, v); }
    if (hi == 4u) { return vec3<f32>(t, p, v); }
    return vec3<f32>(v, p, q);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.position.xy);

    let num1 = textureLoad(t_num1, coords, 0).r;
    let den1 = textureLoad(t_den1, coords, 0).r;
    let sq1 = textureLoad(t_sq1, coords, 0).r;
    let num2 = textureLoad(t_num2, coords, 0).r;
    let den2 = textureLoad(t_den2, coords, 0).r;
    let sq2 = textureLoad(t_sq2, coords, 0).r;

    let val1 = resolve_value(num1, den1, sq1, params.mode1);
    let val2 = resolve_value(num2, den2, sq2, params.mode2);

    if (val1 < -1e29 && val2 < -1e29) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    var lightness: f32 = 0.0;
    if (val1 > -1e29) {
        lightness = normalize_value(val1, params.min1, params.max1, params.log1);
    }
    var color_t: f32 = 0.5;
    if (val2 > -1e29) {
        color_t = normalize_value(val2, params.min2, params.max2, params.log2);
    }

    let rgb = textureSampleLevel(t_colormap, s_colormap, vec2<f32>(color_t, 0.5), 0.0).rgb;
    var hsv = rgb2hsv(rgb);
    hsv.z *= lightness;
    let blended = hsv2rgb(hsv);

    return vec4<f32>(blended, 1.0);
}
