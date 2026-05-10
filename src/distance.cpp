#include "vectordb/distance.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define VECTORDB_HAS_NEON 1
#else
#define VECTORDB_HAS_NEON 0
#endif

namespace vectordb {

#if VECTORDB_HAS_NEON

// 4-way unrolled NEON L2sq. Each NEON reg holds 4 floats; we process 16
// elements per iteration to (a) saturate the 4 FMA pipes on Apple Silicon
// and (b) keep enough ILP that we hide the load latency.
float l2sq(const float* a, const float* b, std::size_t d) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        float32x4_t d0 = vsubq_f32(a0, b0);
        float32x4_t d1 = vsubq_f32(a1, b1);
        float32x4_t d2 = vsubq_f32(a2, b2);
        float32x4_t d3 = vsubq_f32(a3, b3);
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t df = vsubq_f32(va, vb);
        acc0 = vfmaq_f32(acc0, df, df);
    }

    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc);

    for (; i < d; ++i) {
        float df = a[i] - b[i];
        sum += df * df;
    }
    return sum;
}

float dot(const float* a, const float* b, std::size_t d) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va, vb);
    }

    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc);

    for (; i < d; ++i) sum += a[i] * b[i];
    return sum;
}

#else  // scalar fallback

float l2sq(const float* a, const float* b, std::size_t d) noexcept {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float d0 = a[i + 0] - b[i + 0];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < d; ++i) { float df = a[i] - b[i]; s += df * df; }
    return s;
}

float dot(const float* a, const float* b, std::size_t d) noexcept {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}

#endif

}  // namespace vectordb
