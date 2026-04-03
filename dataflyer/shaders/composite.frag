#version 330

// Composite resolve: two fields blended via HSV.
// Field 1 (lightness) controls brightness.
// Field 2 (color) controls hue via colormap.

in vec2 v_uv;

// Field 1 (lightness) accumulation textures
uniform sampler2D u_num1;
uniform sampler2D u_den1;
uniform sampler2D u_sq1;
uniform float u_min1;
uniform float u_max1;
uniform int u_mode1;       // 0: surface density, 1: weighted avg, 2: variance
uniform int u_log1;

// Field 2 (color) accumulation textures
uniform sampler2D u_num2;
uniform sampler2D u_den2;
uniform sampler2D u_sq2;
uniform float u_min2;
uniform float u_max2;
uniform int u_mode2;
uniform int u_log2;

uniform sampler2D u_colormap;

out vec4 frag_color;

float resolve_value(float num, float den, float sq, int mode) {
    if (den < 1e-30) return -1e30;
    if (mode == 0) return den;
    if (mode == 1) return num / den;
    // mode 2: variance
    float mean = num / den;
    float mean_sq = sq / den;
    return sqrt(max(mean_sq - mean * mean, 0.0));
}

float normalize(float val, float vmin, float vmax, int log_scale) {
    if (log_scale == 1) {
        float lv = log(max(val, 1e-30)) / log(10.0);
        return clamp((lv - vmin) / (vmax - vmin), 0.0, 1.0);
    }
    return clamp((val - vmin) / (vmax - vmin), 0.0, 1.0);
}

// RGB <-> HSV conversion
vec3 rgb2hsv(vec3 c) {
    float cmax = max(c.r, max(c.g, c.b));
    float cmin = min(c.r, min(c.g, c.b));
    float delta = cmax - cmin;
    float h = 0.0;
    if (delta > 0.0) {
        if (cmax == c.r) h = mod((c.g - c.b) / delta, 6.0);
        else if (cmax == c.g) h = (c.b - c.r) / delta + 2.0;
        else h = (c.r - c.g) / delta + 4.0;
        h /= 6.0;
    }
    float s = (cmax > 0.0) ? delta / cmax : 0.0;
    return vec3(h, s, cmax);
}

vec3 hsv2rgb(vec3 c) {
    float h = c.x * 6.0;
    float s = c.y;
    float v = c.z;
    float i = floor(h);
    float f = h - i;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    int hi = int(mod(i, 6.0));
    if (hi == 0) return vec3(v, t, p);
    if (hi == 1) return vec3(q, v, p);
    if (hi == 2) return vec3(p, v, t);
    if (hi == 3) return vec3(p, q, v);
    if (hi == 4) return vec3(t, p, v);
    return vec3(v, p, q);
}

void main() {
    float num1 = texture(u_num1, v_uv).r;
    float den1 = texture(u_den1, v_uv).r;
    float sq1 = texture(u_sq1, v_uv).r;
    float num2 = texture(u_num2, v_uv).r;
    float den2 = texture(u_den2, v_uv).r;
    float sq2 = texture(u_sq2, v_uv).r;

    float val1 = resolve_value(num1, den1, sq1, u_mode1);
    float val2 = resolve_value(num2, den2, sq2, u_mode2);

    if (val1 < -1e29 && val2 < -1e29) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float lightness = (val1 > -1e29) ? normalize(val1, u_min1, u_max1, u_log1) : 0.0;
    float color_t = (val2 > -1e29) ? normalize(val2, u_min2, u_max2, u_log2) : 0.5;

    // Get color from colormap, blend lightness in HSV
    vec3 rgb = texture(u_colormap, vec2(color_t, 0.5)).rgb;
    vec3 hsv = rgb2hsv(rgb);
    hsv.z *= lightness;  // modulate value (brightness) by lightness
    vec3 blended = hsv2rgb(hsv);

    frag_color = vec4(blended, 1.0);
}
