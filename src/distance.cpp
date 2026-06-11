#include "vectordb/distance.hpp"

#include "vectordb/simd.hpp"
#define VECTORDB_HAS_NEON VECTORDB_USE_NEON

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

// ---- batch kernels --------------------------------------------------------
//
// 4 candidates x 4 accumulator chains = 16 live accumulators, plus 4 query
// regs and a rotating candidate reg: ~21 of the 32 AArch64 NEON registers.
// The query block is loaded ONCE per 16 elements and reused by all four
// candidates (the single-pair kernel re-loads it per call).
//
// Per-candidate accumulation order mirrors l2sq()/dot() exactly:
// 16-wide main loop into chains 0..3, 4-wide tail into chain 0, pairwise
// horizontal sum, scalar tail. This keeps batched results bit-identical to
// the single-pair kernels.

void l2sq_x4(const float* q,
             const float* p0, const float* p1,
             const float* p2, const float* p3,
             std::size_t d, float* out) noexcept {
    const float* p[4] = {p0, p1, p2, p3};
    float32x4_t acc[4][4];
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j) acc[k][j] = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t q0 = vld1q_f32(q + i);
        float32x4_t q1 = vld1q_f32(q + i + 4);
        float32x4_t q2 = vld1q_f32(q + i + 8);
        float32x4_t q3 = vld1q_f32(q + i + 12);
        for (int k = 0; k < 4; ++k) {
            float32x4_t d0 = vsubq_f32(q0, vld1q_f32(p[k] + i));
            float32x4_t d1 = vsubq_f32(q1, vld1q_f32(p[k] + i + 4));
            float32x4_t d2 = vsubq_f32(q2, vld1q_f32(p[k] + i + 8));
            float32x4_t d3 = vsubq_f32(q3, vld1q_f32(p[k] + i + 12));
            acc[k][0] = vfmaq_f32(acc[k][0], d0, d0);
            acc[k][1] = vfmaq_f32(acc[k][1], d1, d1);
            acc[k][2] = vfmaq_f32(acc[k][2], d2, d2);
            acc[k][3] = vfmaq_f32(acc[k][3], d3, d3);
        }
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qv = vld1q_f32(q + i);
        for (int k = 0; k < 4; ++k) {
            float32x4_t df = vsubq_f32(qv, vld1q_f32(p[k] + i));
            acc[k][0] = vfmaq_f32(acc[k][0], df, df);
        }
    }
    for (int k = 0; k < 4; ++k) {
        float32x4_t a = vaddq_f32(vaddq_f32(acc[k][0], acc[k][1]),
                                  vaddq_f32(acc[k][2], acc[k][3]));
        float sum = vaddvq_f32(a);
        for (std::size_t t = i; t < d; ++t) {
            float df = q[t] - p[k][t];
            sum += df * df;
        }
        out[k] = sum;
    }
}

void dot_x4(const float* q,
            const float* p0, const float* p1,
            const float* p2, const float* p3,
            std::size_t d, float* out) noexcept {
    const float* p[4] = {p0, p1, p2, p3};
    float32x4_t acc[4][4];
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j) acc[k][j] = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t q0 = vld1q_f32(q + i);
        float32x4_t q1 = vld1q_f32(q + i + 4);
        float32x4_t q2 = vld1q_f32(q + i + 8);
        float32x4_t q3 = vld1q_f32(q + i + 12);
        for (int k = 0; k < 4; ++k) {
            acc[k][0] = vfmaq_f32(acc[k][0], q0, vld1q_f32(p[k] + i));
            acc[k][1] = vfmaq_f32(acc[k][1], q1, vld1q_f32(p[k] + i + 4));
            acc[k][2] = vfmaq_f32(acc[k][2], q2, vld1q_f32(p[k] + i + 8));
            acc[k][3] = vfmaq_f32(acc[k][3], q3, vld1q_f32(p[k] + i + 12));
        }
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qv = vld1q_f32(q + i);
        for (int k = 0; k < 4; ++k) {
            acc[k][0] = vfmaq_f32(acc[k][0], qv, vld1q_f32(p[k] + i));
        }
    }
    for (int k = 0; k < 4; ++k) {
        float32x4_t a = vaddq_f32(vaddq_f32(acc[k][0], acc[k][1]),
                                  vaddq_f32(acc[k][2], acc[k][3]));
        float sum = vaddvq_f32(a);
        for (std::size_t t = i; t < d; ++t) sum += q[t] * p[k][t];
        out[k] = sum;
    }
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

// Scalar fallback: batched == four single calls (trivially bit-identical).
void l2sq_x4(const float* q,
             const float* p0, const float* p1,
             const float* p2, const float* p3,
             std::size_t d, float* out) noexcept {
    out[0] = l2sq(q, p0, d);
    out[1] = l2sq(q, p1, d);
    out[2] = l2sq(q, p2, d);
    out[3] = l2sq(q, p3, d);
}

void dot_x4(const float* q,
            const float* p0, const float* p1,
            const float* p2, const float* p3,
            std::size_t d, float* out) noexcept {
    out[0] = dot(q, p0, d);
    out[1] = dot(q, p1, d);
    out[2] = dot(q, p2, d);
    out[3] = dot(q, p3, d);
}

#endif

#if VECTORDB_HAS_NEON

void l2sq_4x4(const float* q, std::size_t q_stride,
              const float* c, std::size_t c_stride,
              std::size_t d, float* out) noexcept {
    float32x4_t acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = vdupq_n_f32(0.0f);

    std::size_t t = 0;
    for (; t + 4 <= d; t += 4) {
        float32x4_t qv[4], cv[4];
        for (int i = 0; i < 4; ++i) qv[i] = vld1q_f32(q + i * q_stride + t);
        for (int j = 0; j < 4; ++j) cv[j] = vld1q_f32(c + j * c_stride + t);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float32x4_t df = vsubq_f32(qv[i], cv[j]);
                acc[i][j] = vfmaq_f32(acc[i][j], df, df);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = vaddvq_f32(acc[i][j]);
            for (std::size_t r = t; r < d; ++r) {
                float df = q[i * q_stride + r] - c[j * c_stride + r];
                sum += df * df;
            }
            out[i * 4 + j] = sum;
        }
    }
}

#else

void l2sq_4x4(const float* q, std::size_t q_stride,
              const float* c, std::size_t c_stride,
              std::size_t d, float* out) noexcept {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out[i * 4 + j] = l2sq(q + i * q_stride, c + j * c_stride, d);
}

#endif

void l2sq_ny(float* out, const float* q, const float* base,
             std::size_t ny, std::size_t d) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        const float* row = base + i * d;
        l2sq_x4(q, row, row + d, row + 2 * d, row + 3 * d, d, out + i);
    }
    for (; i < ny; ++i) out[i] = l2sq(q, base + i * d, d);
}

void dot_ny(float* out, const float* q, const float* base,
            std::size_t ny, std::size_t d) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        const float* row = base + i * d;
        dot_x4(q, row, row + d, row + 2 * d, row + 3 * d, d, out + i);
    }
    for (; i < ny; ++i) out[i] = dot(q, base + i * d, d);
}

}  // namespace vectordb
