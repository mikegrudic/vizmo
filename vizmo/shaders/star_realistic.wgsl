// Minimal "realistic stars" billboard pass.
// One instanced quad per star. Vertex shader projects the star center
// to clip space and offsets the four corners by a screen-space radius
// proportional to log(luminosity). Fragment shader produces a Gaussian
// core with a soft halo, scaled by the (extinction-attenuated) flux.
// Output is additively blended into the resolved screen target.

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    kernel_id: u32,
    _pad: u32,
};

struct StarParams {
    world_radius: f32,   // physical half-width of the billboard in world units
    intensity: f32,      // global brightness multiplier (PSF mode only)
    band_idx: f32,       // which PSF layer to sample
    marker_mode: f32,    // 0 = realistic PSF, ≥0.5 = simple filled-disk marker
    // border_rgb sits at a 16-byte boundary so std140 alignment works.
    border_rgb: vec3<f32>,
    border_frac: f32,    // ring thickness as a fraction of radius (0..1)
    sink_qty_min: f32,   // marker-mode colormap normalization range
    sink_qty_max: f32,
    sink_log_scale: f32, // 0 = linear, 1 = log10 the per-sink data value
    sink_color_active: f32, // 0 = colour field is "None" → use fill_rgb
    fill_rgb: vec3<f32>, // solid-fill colour when sink_color_active == 0
    _pad0: f32,
    // Sizing: r_world = world_radius * pow(field, size_exponent).
    // opacity multiplies the marker-mode fragment alpha; PSF mode
    // ignores it (additive blending).
    size_exponent: f32,
    opacity: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: StarParams;
@group(0) @binding(2) var psf_tex: texture_2d_array<f32>;
@group(0) @binding(3) var psf_samp: sampler;
@group(0) @binding(4) var sink_cmap: texture_1d<f32>;
@group(0) @binding(5) var sink_cmap_samp: sampler;

// Per-star data: pos.xyz + single-band L, then per-channel RGB lums,
// then per-particle scalar driving the marker colormap (.x).
struct Star {
    pos_lum: vec4<f32>,
    lum_rgb: vec4<f32>,
    data_val: vec4<f32>,
};
@group(1) @binding(0) var<storage, read> stars: array<Star>;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) atten_single: f32,   // L_single_att / L_raw, in [0,1]
    @location(2) view_dist: f32,
    @location(3) atten_rgb: vec3<f32>, // (L_R, L_G, L_B) / L_raw, each in [0,1]
    @location(4) marker_t: f32,        // colormap-normalized value, marker mode
};

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    let x = f32(vertex_index & 1u) * 2.0 - 1.0;
    let y = f32((vertex_index >> 1u) & 1u) * 2.0 - 1.0;
    return vec2<f32>(x, y);
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32,
           @builtin(instance_index) iid: u32) -> VOut {
    let s = stars[iid];
    let world_pos = vec4<f32>(s.pos_lum.xyz, 1.0);
    let view_pos = camera.view * world_pos;

    let lum = max(s.pos_lum.w, 1.0e-6);
    // Physical-size billboard. With size_exponent = 0.5 the radius
    // grows as √(field), so the marker AREA scales linearly with the
    // chosen size field — useful for fluxes/masses where total light
    // (or weight) is the perceptually meaningful quantity. Set to 1
    // for radius ∝ field (linear), 0 for constant size.
    let r_world_phys = params.world_radius * pow(lum, params.size_exponent);
    // Floor the marker so it never vanishes sub-pixel. proj[1][1] =
    // 1/tan(fovy/2), so the world-space size that subtends one pixel
    // at view depth |z| is w_per_px = 2|z| / (proj[1][1] * viewport_y).
    // The floor is two pixels because a single-pixel-radius quad can
    // land entirely outside the disk (r > 1) under subpixel jitter,
    // tripping the discard in marker mode.
    let depth = max(abs(view_pos.z), 1.0e-6);
    let w_per_px = 2.0 * depth / max(camera.proj[1][1] * camera.viewport_size.y, 1.0e-6);
    let r_world = max(r_world_phys, 2.0 * w_per_px);

    let corner = quad_corner(vid);
    // Offset the corner in *view space* (camera-aligned billboard),
    // then project — the perspective divide handles the inverse-d
    // scaling automatically.
    let view_corner = vec4<f32>(
        view_pos.x + corner.x * r_world,
        view_pos.y + corner.y * r_world,
        view_pos.z,
        1.0,
    );

    var out: VOut;
    out.clip_pos = camera.proj * view_corner;
    out.local = corner;
    out.atten_single = clamp(s.lum_rgb.w / lum, 0.0, 1.0);
    out.atten_rgb = clamp(s.lum_rgb.rgb / lum, vec3<f32>(0.0), vec3<f32>(1.0));
    out.view_dist = length(view_pos.xyz);
    // Marker colormap coordinate: per-sink scalar normalized to [0,1]
    // via (data - min)/(max - min), optionally after log10. Computed
    // here so the fragment shader stays branch-free on uniform values.
    var d = s.data_val.x;
    if (params.sink_log_scale > 0.5) {
        d = log(max(d, 1.0e-30)) / log(10.0);
    }
    let rng = max(params.sink_qty_max - params.sink_qty_min, 1.0e-12);
    out.marker_t = clamp((d - params.sink_qty_min) / rng, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Marker mode: filled disk with a contrasting border ring, no PSF.
    // Fill colour comes from sampling the sink colormap at the
    // normalized per-sink data value when a colour field is selected;
    // when the user picks "None" the fill is solid black. Border uses
    // the RGB sliders directly in both cases.
    if (params.marker_mode > 0.5) {
        let r = length(in.local);
        if (r > 1.0) {
            discard;
        }
        let inner = max(0.0, 1.0 - params.border_frac);
        if (r <= inner) {
            var fill = params.fill_rgb;
            if (params.sink_color_active > 0.5) {
                fill = textureSample(sink_cmap, sink_cmap_samp, in.marker_t).rgb;
            }
            return vec4<f32>(fill, params.opacity);
        }
        return vec4<f32>(params.border_rgb, params.opacity);
    }

    // local is in [-1,1]; map to PSF texture coords in [0,1].
    let uv = in.local * 0.5 + vec2<f32>(0.5, 0.5);
    // Surface brightness of a physical disk is independent of distance
    // (the billboard's screen area shrinks ∝ 1/d², canceling the
    // 1/d² flux falloff), so the per-pixel intensity is just a
    // function of L and the PSF profile.
    let scale = params.intensity;

    let band = i32(params.band_idx);
    if (band >= 5) {
        // RGB composite: I→R, V→G, B→B (texture array layers 4, 2, 1).
        let psf_r = sqrt(max(textureSample(psf_tex, psf_samp, uv, 4).r, 0.0));
        let psf_g = sqrt(max(textureSample(psf_tex, psf_samp, uv, 2).r, 0.0));
        let psf_b = sqrt(max(textureSample(psf_tex, psf_samp, uv, 1).r, 0.0));
        let rgb = vec3<f32>(
            psf_r * in.atten_rgb.r,
            psf_g * in.atten_rgb.g,
            psf_b * in.atten_rgb.b,
        ) * scale;
        return vec4<f32>(rgb, 1.0);
    } else {
        let psf = sqrt(max(textureSample(psf_tex, psf_samp, uv, band).r, 0.0));
        let amp = psf * scale * in.atten_single;
        return vec4<f32>(amp, amp, amp, 1.0);
    }
}
