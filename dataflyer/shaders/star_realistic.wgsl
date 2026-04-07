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
    intensity: f32,      // global brightness multiplier
    band_idx: f32,       // which PSF layer to sample
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: StarParams;
@group(0) @binding(2) var psf_tex: texture_2d_array<f32>;
@group(0) @binding(3) var psf_samp: sampler;

// Per-star data: pos.xyz + single-band L, then per-channel RGB lums.
struct Star {
    pos_lum: vec4<f32>,
    lum_rgb: vec4<f32>,
};
@group(1) @binding(0) var<storage, read> stars: array<Star>;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) atten_single: f32,   // L_single_att / L_raw, in [0,1]
    @location(2) view_dist: f32,
    @location(3) atten_rgb: vec3<f32>, // (L_R, L_G, L_B) / L_raw, each in [0,1]
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
    // Physical-size billboard: radius is in world units, so the
    // apparent screen size grows as 1/d like a real spherical object.
    // Radius scales as sqrt(L) (so area ∝ L), which keeps the
    // per-pixel surface brightness independent of intrinsic
    // luminosity — brighter stars look bigger, not whiter.
    let r_world = params.world_radius * sqrt(lum);

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
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
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
