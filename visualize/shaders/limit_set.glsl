#version 300 es
precision highp float;

uniform vec2 resolution;
uniform vec2 center;
uniform float zoom;
uniform int max_iter;

uniform mat3 gen_a_re;
uniform mat3 gen_a_im;
uniform mat3 gen_b_re;
uniform mat3 gen_b_im;

out vec4 fragColor;

const float PI = 3.14159265;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

vec2 cdiv(vec2 a, vec2 b) {
    float denom = dot(b, b);
    return vec2(dot(a, b), a.y*b.x - a.x*b.y) / denom;
}

vec2 mobius_apply(mat3 m_re, mat3 m_im, vec2 z) {
    vec2 a = vec2(m_re[0][0], m_im[0][0]);
    vec2 b = vec2(m_re[1][0], m_im[1][0]);
    vec2 c = vec2(m_re[0][1], m_im[0][1]);
    vec2 d = vec2(m_re[1][1], m_im[1][1]);
    vec2 num = cmul(a, z) + b;
    vec2 den = cmul(c, z) + d;
    return cdiv(num, den);
}

vec2 mobius_inv_apply(mat3 m_re, mat3 m_im, vec2 z) {
    vec2 a = vec2(m_re[0][0], m_im[0][0]);
    vec2 b = vec2(m_re[1][0], m_im[1][0]);
    vec2 c = vec2(m_re[0][1], m_im[0][1]);
    vec2 d = vec2(m_re[1][1], m_im[1][1]);
    vec2 num = cmul(d, z) - b;
    vec2 den = -cmul(c, z) + a;
    return cdiv(num, den);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / min(resolution.x, resolution.y);
    vec2 z = uv / zoom + center;

    float R = 100.0;
    float escape_iter = float(max_iter);
    float arg_accum = 0.0;
    float dist_est = 1e10;

    vec2 orbit = z;

    for (int i = 0; i < 256; i++) {
        if (i >= max_iter) break;

        int which = i % 4;
        vec2 next;
        if (which == 0) {
            next = mobius_apply(gen_a_re, gen_a_im, orbit);
        } else if (which == 1) {
            next = mobius_inv_apply(gen_a_re, gen_a_im, orbit);
        } else if (which == 2) {
            next = mobius_apply(gen_b_re, gen_b_im, orbit);
        } else {
            next = mobius_inv_apply(gen_b_re, gen_b_im, orbit);
        }

        float len = length(next);
        if (len > R) {
            escape_iter = float(i) + 1.0 - log(log(len)) / log(2.0);
            break;
        }

        dist_est = min(dist_est, len);
        arg_accum += atan(next.y, next.x);
        orbit = next;
    }

    float t = escape_iter / float(max_iter);
    float hue = fract(arg_accum / (2.0 * PI) * 0.5 + t * 0.7);
    float sat = 0.8;
    float val = (escape_iter < float(max_iter)) ? 1.0 : (dist_est < 0.1 ? 0.0 : 0.05);

    fragColor = vec4(hsv2rgb(vec3(hue, sat, val)), 1.0);
}
