#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vdivq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmaxq_f32, vminq_f32,
    vmulq_f32, vnegq_f32, vst1q_f32, vsubq_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128, __m256, _mm_add_ps, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps,
    _mm_max_ps, _mm_min_ps, _mm_mul_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, __m256, _mm_add_ps, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps,
    _mm_max_ps, _mm_min_ps, _mm_mul_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

use super::config::BinaryKind;

#[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn vsAdd(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn vsSub(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn vsMul(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn vsDiv(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn vsExp(n: i32, a: *const f32, y: *mut f32);
    fn vsSqrt(n: i32, a: *const f32, y: *mut f32);
    fn vsLn(n: i32, a: *const f32, y: *mut f32);
}

#[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn armpl_svexp_f32(n: i32, x: *const f32, y: *mut f32);
    fn armpl_svadd_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn armpl_svsub_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn armpl_svmul_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn armpl_svlog_f32(n: i32, x: *const f32, y: *mut f32);
    fn armpl_svsqrt_f32(n: i32, x: *const f32, y: *mut f32);
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code, dead_code)]
unsafe extern "C" {
    fn vvexpf(result: *mut f32, input: *const f32, count: *const i32);
    fn vDSP_vadd(
        __A: *const f32,
        __IA: i32,
        __B: *const f32,
        __IB: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    fn vDSP_vsub(
        __B: *const f32,
        __IB: i32,
        __A: *const f32,
        __IA: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    fn vDSP_vmul(
        __A: *const f32,
        __IA: i32,
        __B: *const f32,
        __IB: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
}

// ===========================================================================
// ReLU dispatch
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub fn relu_slice_dispatch(values: &mut [f32]) {
    if cfg!(miri) {
        // SAFETY: scalar path only reads/writes within `values` bounds.
        unsafe {
            relu_slice_scalar(values);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_slice_avx(values);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_slice_sse(values);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_slice_neon(values);
            }
            return;
        }
    }

    // SAFETY: scalar path only reads/writes within `values` bounds.
    unsafe {
        relu_slice_scalar(values);
    }
}

/// Two-argument ReLU: `output[i] = max(0, input[i])`.
///
/// Avoids the clone+in-place pattern by reading from `input` and writing to
/// `output` in a single pass, halving memory traffic.
#[allow(unsafe_code)]
#[inline]
pub fn relu_to_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        // SAFETY: scalar path only reads/writes within bounds.
        unsafe {
            relu_to_slice_scalar(input, output);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_to_slice_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_to_slice_sse(input, output);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_to_slice_neon(input, output);
            }
            return;
        }
    }

    // SAFETY: scalar path only reads/writes within bounds.
    unsafe {
        relu_to_slice_scalar(input, output);
    }
}

#[inline]
#[allow(dead_code)]
pub(crate) fn sigmoid_slice(values: &mut [f32]) {
    for value in values {
        *value = sigmoid_scalar(*value);
    }
}

#[inline]
pub(crate) fn sigmoid_scalar(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

// ===========================================================================
// Exp / Sigmoid / Tanh SIMD dispatch
// ===========================================================================

/// Fast exp approximation applied element-wise: `output[i] = exp(input[i])`.
///
/// Uses a polynomial approximation (degree-4 minimax on [-88, 88]) that is
/// accurate to roughly 1e-4 relative error for the typical NN activation range.
#[allow(unsafe_code, unreachable_code)]
#[inline]
pub fn exp_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        exp_slice_scalar(input, output);
        return;
    }

    // macOS aarch64: use Apple Accelerate vvexpf (heavily optimized).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let count = input.len() as i32;
        // SAFETY: vvexpf reads `count` floats from `input` and writes to `output`.
        // Both slices have equal length (debug_assert above).
        unsafe {
            vvexpf(output.as_mut_ptr(), input.as_ptr(), &count);
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML vsExp (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let count = input.len() as i32;
        // SAFETY: vsExp reads `count` floats from `input` and writes to `output`.
        unsafe { vsExp(count, input.as_ptr(), output.as_mut_ptr()) };
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries vectorized exp.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let count = input.len() as i32;
        // SAFETY: armpl_svexp_f32 reads `count` floats from `input` and writes to `output`.
        unsafe { armpl_svexp_f32(count, input.as_ptr(), output.as_mut_ptr()) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                exp_slice_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                exp_slice_sse(input, output);
            }
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                exp_slice_neon(input, output);
            }
            return;
        }
    }

    exp_slice_scalar(input, output);
}

/// Fused subtract-and-exp: `output[i] = exp(input[i] - offset)`.
///
/// Combines the max-subtraction and exp steps of softmax into one pass,
/// avoiding an extra read/write of the output buffer.
#[allow(unsafe_code)]
#[inline]
pub fn sub_exp_slice_dispatch(input: &[f32], offset: f32, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        sub_exp_slice_scalar(input, offset, output);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                sub_exp_slice_avx(input, offset, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                sub_exp_slice_sse(input, offset, output);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                sub_exp_slice_neon(input, offset, output);
            }
            return;
        }
    }

    sub_exp_slice_scalar(input, offset, output);
}

/// Fast sigmoid applied element-wise: `output[i] = 1 / (1 + exp(-input[i]))`.
#[allow(unsafe_code, clippy::needless_return)]
#[inline]
pub fn sigmoid_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        sigmoid_slice_dispatch_scalar(input, output);
        return;
    }

    // NEON / AVX / SSE dispatch for sigmoid.
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx") {
                // SAFETY: guarded by runtime feature detection.
                unsafe {
                    sigmoid_slice_avx(input, output);
                }
                return;
            }
            if std::is_x86_feature_detected!("sse") {
                // SAFETY: guarded by runtime feature detection.
                unsafe {
                    sigmoid_slice_sse(input, output);
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    sigmoid_slice_neon(input, output);
                }
                return;
            }
        }

        sigmoid_slice_dispatch_scalar(input, output);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
#[inline]
/// Fast exp for sigmoid: range reduction + 3-term Horner + IEEE bit trick.
/// WHY 3 terms: 3rd-order polynomial suffices for sigmoid (1/(1+exp) dampens error); max error ~1e-4.
unsafe fn fast_exp_sigmoid_neon(x: float32x4_t) -> float32x4_t {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vreinterpretq_f32_s32, vshlq_n_s32,
        vsubq_f32,
    };
    let x = vmaxq_f32(vdupq_n_f32(-88.0), vminq_f32(vdupq_n_f32(88.0), x));
    let n_f = vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E));
    let n_i = vcvtnq_s32_f32(n_f);
    let r = vsubq_f32(
        x,
        vmulq_f32(vcvtq_f32_s32(n_i), vdupq_n_f32(std::f32::consts::LN_2)),
    );
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_i, vdupq_n_s32(127))));
    let p = vfmaq_f32(vdupq_n_f32(0.5), r, vdupq_n_f32(1.0 / 6.0));
    let p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
    vmulq_f32(vfmaq_f32(vdupq_n_f32(1.0), r, p), pow2n)
}

/// Sigmoid via hand-scheduled NEON assembly.
///
/// Processes 4 elements per iteration with interleaved load/compute/store.
/// The FMA pipeline is kept fully saturated by overlapping independent operations.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn sigmoid_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let mut inp = input.as_ptr();
    let mut out = output.as_mut_ptr();
    let mut remaining = len;

    // Load all constants ONCE before the loop, keep in NEON registers
    if remaining >= 4 {
        unsafe {
            // Constants on stack for ld1r broadcast
            let c_neg88: f32 = -88.0;
            let c_pos88: f32 = 88.0;
            // Schraudolph 1999 constants: exp(x) ≈ reinterpret(int(x * C + B))
            // C = 2^23 / ln(2) = 12102203.16, B = 127 * 2^23 = 1065353216
            // WHY: 2^23/ln(2) maps float mantissa bits to IEEE 754 exponent field; 127*2^23 adds the exponent bias.
            let c_schr_c: f32 = 12102203.0; // 2^23 / ln(2)
            let c_schr_b: i32 = 127 << 23; // 1065353216 as integer
            let c_sixth: f32 = 1.0 / 6.0;
            let c_half: f32 = 0.5;
            let c_one: f32 = 1.0;
            let c_127: i32 = 127;

            // Load constants into NEON registers (stays there for entire loop)
            std::arch::asm!(
                "ld1r {{v16.4s}}, [{p_neg88}]",
                "ld1r {{v17.4s}}, [{p_pos88}]",
                "ld1r {{v18.4s}}, [{p_schr_c}]",   // Schraudolph C (float)
                "dup  v19.4s, {p_schr_b:w}",        // Schraudolph B (integer 127<<23)
                "ld1r {{v20.4s}}, [{p_sixth}]",
                "ld1r {{v21.4s}}, [{p_half}]",
                "ld1r {{v22.4s}}, [{p_one}]",
                "dup  v23.4s, {p_127:w}",
                p_neg88 = in(reg) &c_neg88,
                p_pos88 = in(reg) &c_pos88,
                p_schr_c = in(reg) &c_schr_c,
                p_schr_b = in(reg) c_schr_b,
                p_sixth = in(reg) &c_sixth,
                p_half = in(reg) &c_half,
                p_one = in(reg) &c_one,
                p_127 = in(reg) c_127,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            );

            // Schraudolph bit-trick: exp(x) ≈ reinterpret_f32(int(x * 2^23/ln2) + 127<<23)
            // Proper integer arithmetic: fcvtzs to get int, then add bias as int, then reinterpret
            // 4× unrolled, 16 elements per iteration
            while remaining >= 16 {
                std::arch::asm!(
                    "ldp q0, q1, [{inp}]",
                    "ldp q2, q3, [{inp}, #32]",
                    "add {inp}, {inp}, #64",
                    "fneg v0.4s, v0.4s",
                    "fneg v1.4s, v1.4s",
                    "fneg v2.4s, v2.4s",
                    "fneg v3.4s, v3.4s",
                    "fmax v0.4s, v0.4s, v16.4s",
                    "fmax v1.4s, v1.4s, v16.4s",
                    "fmax v2.4s, v2.4s, v16.4s",
                    "fmax v3.4s, v3.4s, v16.4s",
                    "fmin v0.4s, v0.4s, v17.4s",
                    "fmin v1.4s, v1.4s, v17.4s",
                    "fmin v2.4s, v2.4s, v17.4s",
                    "fmin v3.4s, v3.4s, v17.4s",
                    // x * (2^23/ln2) → convert to int
                    "fmul v0.4s, v0.4s, v18.4s",
                    "fmul v1.4s, v1.4s, v18.4s",
                    "fmul v2.4s, v2.4s, v18.4s",
                    "fmul v3.4s, v3.4s, v18.4s",
                    "fcvtzs v0.4s, v0.4s",
                    "fcvtzs v1.4s, v1.4s",
                    "fcvtzs v2.4s, v2.4s",
                    "fcvtzs v3.4s, v3.4s",
                    // + 127*2^23 (integer add)
                    "add v0.4s, v0.4s, v19.4s",
                    "add v1.4s, v1.4s, v19.4s",
                    "add v2.4s, v2.4s, v19.4s",
                    "add v3.4s, v3.4s, v19.4s",
                    // v0-v3 bits ARE exp(-x) when reinterpreted as float
                    // sigmoid = 1 / (1 + exp)
                    "fadd v0.4s, v22.4s, v0.4s",
                    "fadd v1.4s, v22.4s, v1.4s",
                    "fadd v2.4s, v22.4s, v2.4s",
                    "fadd v3.4s, v22.4s, v3.4s",
                    "fdiv v0.4s, v22.4s, v0.4s",
                    "fdiv v1.4s, v22.4s, v1.4s",
                    "fdiv v2.4s, v22.4s, v2.4s",
                    "fdiv v3.4s, v22.4s, v3.4s",
                    "stp q0, q1, [{out}]",
                    "stp q2, q3, [{out}, #32]",
                    "add {out}, {out}, #64",
                    inp = inout(reg) inp,
                    out = inout(reg) out,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                );
                remaining -= 16;
            }
            // 4-element tail — Schraudolph
            while remaining >= 4 {
                std::arch::asm!(
                    "ld1 {{v0.4s}}, [{inp}], #16",
                    "fneg v0.4s, v0.4s",
                    "fmax v0.4s, v0.4s, v16.4s",
                    "fmin v0.4s, v0.4s, v17.4s",
                    "fmul v0.4s, v0.4s, v18.4s",
                    "fcvtzs v0.4s, v0.4s",
                    "add v0.4s, v0.4s, v19.4s",
                    "fadd v0.4s, v22.4s, v0.4s",
                    "fdiv v0.4s, v22.4s, v0.4s",
                    "st1 {{v0.4s}}, [{out}], #16",
                    inp = inout(reg) inp,
                    out = inout(reg) out,
                    out("v0") _,
                );
                remaining -= 4;
            }
            // 4-element tail — Schraudolph
            while remaining >= 4 {
                std::arch::asm!(
                    "ld1 {{v0.4s}}, [{inp}], #16",
                    "fneg v0.4s, v0.4s",
                    "fmax v0.4s, v0.4s, v16.4s",
                    "fmin v0.4s, v0.4s, v17.4s",
                    "fmul v0.4s, v0.4s, v18.4s",
                    "fcvtzs v0.4s, v0.4s",
                    "add v0.4s, v0.4s, v19.4s",
                    "fadd v0.4s, v22.4s, v0.4s",
                    "fdiv v0.4s, v22.4s, v0.4s",
                    "st1 {{v0.4s}}, [{out}], #16",
                    inp = inout(reg) inp,
                    out = inout(reg) out,
                    out("v0") _,
                );
                remaining -= 4;
            }
        }
    }

    // Scalar tail
    for i in 0..remaining {
        unsafe {
            let x = *inp.add(i);
            *out.add(i) = 1.0 / (1.0 + (-x).exp());
        }
    }
}

// (sigmoid_vdsp and silu_vdsp removed — benchmarked slower than NEON polynomial)

/// Fast tanh applied element-wise: `output[i] = tanh(input[i])`.
///
/// Computed as `2 * sigmoid(2x) - 1`.
#[allow(unsafe_code)]
#[inline]
pub fn tanh_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        tanh_slice_dispatch_scalar(input, output);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                tanh_slice_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                tanh_slice_sse(input, output);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                tanh_slice_neon(input, output);
            }
            return;
        }
    }

    tanh_slice_dispatch_scalar(input, output);
}

/// Fused SiLU (Swish) applied element-wise: `output[i] = input[i] * sigmoid(input[i])`.
///
/// Single-pass over the data avoids the 2× bandwidth penalty of separate sigmoid + multiply.
#[allow(unsafe_code)]
#[inline]
pub fn silu_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        silu_slice_dispatch_scalar(input, output);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                silu_slice_neon(input, output);
            }
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { silu_slice_avx(input, output) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { silu_slice_sse(input, output) };
            return;
        }
    }

    silu_slice_dispatch_scalar(input, output);
}

// ===========================================================================
// Reduction dispatchers: max_reduce, add_reduce
// ===========================================================================

/// Find the maximum value in `data`.  Returns `f32::NEG_INFINITY` for empty slices.
#[allow(unsafe_code, dead_code)]
#[inline]
pub fn max_reduce_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }

    if cfg!(miri) {
        return max_reduce_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { max_reduce_avx(data) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { max_reduce_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { max_reduce_neon(data) };
        }
    }

    max_reduce_scalar(data)
}

/// Sum all values in `data`.  Returns `0.0` for empty slices.
#[allow(unsafe_code, dead_code)]
#[inline]
pub fn add_reduce_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    if cfg!(miri) {
        return add_reduce_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { add_reduce_avx(data) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { add_reduce_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { add_reduce_neon(data) };
        }
    }

    add_reduce_scalar(data)
}

// ===========================================================================
// Scalar-broadcast multiply in-place
// ===========================================================================

/// Multiply every element of `data` by `scalar` in-place.
#[allow(unsafe_code, dead_code)]
#[inline]
pub fn mul_scalar_inplace_dispatch(data: &mut [f32], scalar: f32) {
    if cfg!(miri) || data.is_empty() {
        for v in data.iter_mut() {
            *v *= scalar;
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe { mul_scalar_inplace_neon(data, scalar) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe { mul_scalar_inplace_avx(data, scalar) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe { mul_scalar_inplace_sse(data, scalar) };
            return;
        }
    }

    for v in data.iter_mut() {
        *v *= scalar;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn mul_scalar_inplace_neon(data: &mut [f32], scalar: f32) {
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let vs = vdupq_n_f32(scalar);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vmulq_f32(v, vs));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn mul_scalar_inplace_avx(data: &mut [f32], scalar: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let vs = _mm256_set1_ps(scalar);
    let mut i = 0usize;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, vs));
        i += 8;
    }
    // SSE tail
    let vs4 = _mm_set1_ps(scalar);
    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_mul_ps(v, vs4));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn mul_scalar_inplace_sse(data: &mut [f32], scalar: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let vs = _mm_set1_ps(scalar);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_mul_ps(v, vs));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= scalar;
        i += 1;
    }
}

// ===========================================================================
// FMA dispatch (conv2d inner loop helper)
// ===========================================================================

/// Fused multiply-accumulate: `acc[i] += a[i] * b[i]`.
#[allow(unsafe_code, dead_code)]
#[inline]
pub fn fma_slice_dispatch(a: &[f32], b: &[f32], acc: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), acc.len());

    if cfg!(miri) {
        // SAFETY: scalar path only reads/writes within equal-sized slice bounds.
        unsafe {
            fma_slice_scalar(a, b, acc);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                fma_slice_avx(a, b, acc);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                fma_slice_sse(a, b, acc);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                fma_slice_neon(a, b, acc);
            }
            return;
        }
    }

    // SAFETY: scalar path only reads/writes within equal-sized slice bounds.
    unsafe {
        fma_slice_scalar(a, b, acc);
    }
}

// ===========================================================================
// Binary same-shape dispatch (existing)
// ===========================================================================

#[allow(unsafe_code, unreachable_code)]
#[inline]
pub fn binary_same_shape_dispatch(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());

    if cfg!(miri) {
        // SAFETY: scalar path only reads/writes within equal-sized slice bounds.
        unsafe {
            binary_same_shape_scalar(lhs, rhs, out, kind);
        }
        return;
    }

    // macOS: use vDSP for add/sub/mul (heavily optimized, zero loop overhead).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let n = lhs.len() as u32;
        // SAFETY: vDSP functions read/write `n` floats from contiguous slices.
        unsafe {
            match kind {
                BinaryKind::Add => {
                    vDSP_vadd(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                // NOTE: vDSP_vsub computes A - B with reversed argument order: vsub(B, ..., A, ..., C, ...)
                BinaryKind::Sub => {
                    vDSP_vsub(rhs.as_ptr(), 1, lhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Mul => {
                    vDSP_vmul(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
            }
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML for add/sub/mul (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: VML functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => vsAdd(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => vsSub(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => vsMul(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries for add/sub/mul.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: ARMPL functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => armpl_svadd_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => armpl_svsub_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => armpl_svmul_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                binary_same_shape_avx(lhs, rhs, out, kind);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                binary_same_shape_sse(lhs, rhs, out, kind);
            }
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                binary_same_shape_neon(lhs, rhs, out, kind);
            }
            return;
        }
    }

    // SAFETY: scalar path only reads/writes within equal-sized slice bounds.
    unsafe {
        binary_same_shape_scalar(lhs, rhs, out, kind);
    }
}

// ===========================================================================
// Scalar fallbacks
// ===========================================================================

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn relu_slice_scalar(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let mut index = 0usize;

    while index + 8 <= len {
        let v0 = *ptr.add(index);
        let v1 = *ptr.add(index + 1);
        let v2 = *ptr.add(index + 2);
        let v3 = *ptr.add(index + 3);
        let v4 = *ptr.add(index + 4);
        let v5 = *ptr.add(index + 5);
        let v6 = *ptr.add(index + 6);
        let v7 = *ptr.add(index + 7);
        *ptr.add(index) = v0.max(0.0);
        *ptr.add(index + 1) = v1.max(0.0);
        *ptr.add(index + 2) = v2.max(0.0);
        *ptr.add(index + 3) = v3.max(0.0);
        *ptr.add(index + 4) = v4.max(0.0);
        *ptr.add(index + 5) = v5.max(0.0);
        *ptr.add(index + 6) = v6.max(0.0);
        *ptr.add(index + 7) = v7.max(0.0);
        index += 8;
    }

    while index < len {
        *ptr.add(index) = (*ptr.add(index)).max(0.0);
        index += 1;
    }
}

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn relu_to_slice_scalar(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    while index + 8 <= len {
        *out_ptr.add(index) = (*in_ptr.add(index)).max(0.0);
        *out_ptr.add(index + 1) = (*in_ptr.add(index + 1)).max(0.0);
        *out_ptr.add(index + 2) = (*in_ptr.add(index + 2)).max(0.0);
        *out_ptr.add(index + 3) = (*in_ptr.add(index + 3)).max(0.0);
        *out_ptr.add(index + 4) = (*in_ptr.add(index + 4)).max(0.0);
        *out_ptr.add(index + 5) = (*in_ptr.add(index + 5)).max(0.0);
        *out_ptr.add(index + 6) = (*in_ptr.add(index + 6)).max(0.0);
        *out_ptr.add(index + 7) = (*in_ptr.add(index + 7)).max(0.0);
        index += 8;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).max(0.0);
        index += 1;
    }
}

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn binary_same_shape_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    match kind {
        BinaryKind::Add => {
            while index + 8 <= len {
                *out_ptr.add(index) = *left_ptr.add(index) + *right_ptr.add(index);
                *out_ptr.add(index + 1) = *left_ptr.add(index + 1) + *right_ptr.add(index + 1);
                *out_ptr.add(index + 2) = *left_ptr.add(index + 2) + *right_ptr.add(index + 2);
                *out_ptr.add(index + 3) = *left_ptr.add(index + 3) + *right_ptr.add(index + 3);
                *out_ptr.add(index + 4) = *left_ptr.add(index + 4) + *right_ptr.add(index + 4);
                *out_ptr.add(index + 5) = *left_ptr.add(index + 5) + *right_ptr.add(index + 5);
                *out_ptr.add(index + 6) = *left_ptr.add(index + 6) + *right_ptr.add(index + 6);
                *out_ptr.add(index + 7) = *left_ptr.add(index + 7) + *right_ptr.add(index + 7);
                index += 8;
            }
            while index < len {
                *out_ptr.add(index) = *left_ptr.add(index) + *right_ptr.add(index);
                index += 1;
            }
        }
        BinaryKind::Sub => {
            while index + 8 <= len {
                *out_ptr.add(index) = *left_ptr.add(index) - *right_ptr.add(index);
                *out_ptr.add(index + 1) = *left_ptr.add(index + 1) - *right_ptr.add(index + 1);
                *out_ptr.add(index + 2) = *left_ptr.add(index + 2) - *right_ptr.add(index + 2);
                *out_ptr.add(index + 3) = *left_ptr.add(index + 3) - *right_ptr.add(index + 3);
                *out_ptr.add(index + 4) = *left_ptr.add(index + 4) - *right_ptr.add(index + 4);
                *out_ptr.add(index + 5) = *left_ptr.add(index + 5) - *right_ptr.add(index + 5);
                *out_ptr.add(index + 6) = *left_ptr.add(index + 6) - *right_ptr.add(index + 6);
                *out_ptr.add(index + 7) = *left_ptr.add(index + 7) - *right_ptr.add(index + 7);
                index += 8;
            }
            while index < len {
                *out_ptr.add(index) = *left_ptr.add(index) - *right_ptr.add(index);
                index += 1;
            }
        }
        BinaryKind::Mul => {
            while index + 8 <= len {
                *out_ptr.add(index) = *left_ptr.add(index) * *right_ptr.add(index);
                *out_ptr.add(index + 1) = *left_ptr.add(index + 1) * *right_ptr.add(index + 1);
                *out_ptr.add(index + 2) = *left_ptr.add(index + 2) * *right_ptr.add(index + 2);
                *out_ptr.add(index + 3) = *left_ptr.add(index + 3) * *right_ptr.add(index + 3);
                *out_ptr.add(index + 4) = *left_ptr.add(index + 4) * *right_ptr.add(index + 4);
                *out_ptr.add(index + 5) = *left_ptr.add(index + 5) * *right_ptr.add(index + 5);
                *out_ptr.add(index + 6) = *left_ptr.add(index + 6) * *right_ptr.add(index + 6);
                *out_ptr.add(index + 7) = *left_ptr.add(index + 7) * *right_ptr.add(index + 7);
                index += 8;
            }
            while index < len {
                *out_ptr.add(index) = *left_ptr.add(index) * *right_ptr.add(index);
                index += 1;
            }
        }
    }
}

fn exp_slice_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = v.exp();
    }
}

fn sub_exp_slice_scalar(input: &[f32], offset: f32, output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = (v - offset).exp();
    }
}

fn sigmoid_slice_dispatch_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = sigmoid_scalar(v);
    }
}

fn tanh_slice_dispatch_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = v.tanh();
    }
}

fn silu_slice_dispatch_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        let s = 1.0 / (1.0 + (-v).exp());
        *o = v * s;
    }
}

#[allow(dead_code)]
fn max_reduce_scalar(data: &[f32]) -> f32 {
    let mut acc = f32::NEG_INFINITY;
    for &v in data {
        acc = acc.max(v);
    }
    acc
}

#[allow(dead_code)]
fn add_reduce_scalar(data: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &v in data {
        acc += v;
    }
    acc
}

#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fma_slice_scalar(a: &[f32], b: &[f32], acc: &mut [f32]) {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let acc_ptr = acc.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        *acc_ptr.add(index) += *a_ptr.add(index) * *b_ptr.add(index);
        *acc_ptr.add(index + 1) += *a_ptr.add(index + 1) * *b_ptr.add(index + 1);
        *acc_ptr.add(index + 2) += *a_ptr.add(index + 2) * *b_ptr.add(index + 2);
        *acc_ptr.add(index + 3) += *a_ptr.add(index + 3) * *b_ptr.add(index + 3);
        index += 4;
    }
    while index < len {
        *acc_ptr.add(index) += *a_ptr.add(index) * *b_ptr.add(index);
        index += 1;
    }
}

// ===========================================================================
// SSE fast-exp helper (4-wide)
// ===========================================================================
//
// Uses the classic range-reduction approach:
//   exp(x) = 2^n * exp(r)  where  n = round(x / ln2), r = x - n*ln2
// Then exp(r) is approximated with a degree-4 polynomial on [-ln2/2, ln2/2].

/// Schraudolph 1999 bit-trick exp for SSE: exp(x) ≈ reinterpret(int(x * 2^23/ln2) + 127*2^23).
/// WHY: ~3x faster than polynomial, ~1e-3 accuracy is sufficient for sigmoid/tanh where 1/(1+exp) dampens error.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn fast_exp_bittrick_sse(x: __m128) -> __m128 {
    // SSE2 intrinsics used below are always available on x86_64.
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_add_epi32, _mm_cvtps_epi32, _mm_set1_epi32};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_add_epi32, _mm_cvtps_epi32, _mm_set1_epi32};
    // exp(x) ≈ reinterpret(int(x * C + B)) where C = 2^23/ln2, B = 127*2^23
    let scale = _mm_set1_ps(12102203.0); // WHY: 2^23/ln(2) maps float to IEEE 754 exponent field
    let offset = _mm_set1_epi32(1065353216); // WHY: 127*2^23 is the IEEE 754 exponent bias in integer form
    let clamp_lo = _mm_set1_ps(-87.0); // WHY: below this exp() produces denormals (underflow)
    let clamp_hi = _mm_set1_ps(88.0); // WHY: above this exp() exceeds f32 max (overflow to inf)
    let x_clamped = _mm_max_ps(_mm_min_ps(x, clamp_hi), clamp_lo);
    let val = _mm_cvtps_epi32(_mm_mul_ps(x_clamped, scale));
    _mm_castsi128_ps(_mm_add_epi32(val, offset))
}

/// Polynomial exp for SSE: range-reduction + 6-term Taylor. Higher accuracy (~1e-6)
/// for standalone exp (softmax, etc.) where precision matters more.
/// WHY 6 terms: 6th-order Taylor series for 2^f on [0,1), max error ~1e-7, good accuracy/speed tradeoff.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn fast_exp_sse(x: __m128) -> __m128 {
    let ln2_inv = _mm_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm_set1_ps(0.693_359_4); // upper bits of ln(2)
    let ln2_lo = _mm_set1_ps(-2.121_944_4e-4); // lower bits of ln(2)

    // Polynomial coefficients (Taylor series for exp(r) on [-ln2/2, ln2/2])
    let c0 = _mm_set1_ps(1.0);
    let c1 = _mm_set1_ps(1.0);
    let c2 = _mm_set1_ps(0.5);
    let c3 = _mm_set1_ps(1.0 / 6.0);
    let c4 = _mm_set1_ps(1.0 / 24.0);
    let c5 = _mm_set1_ps(1.0 / 120.0);
    let c6 = _mm_set1_ps(1.0 / 720.0);

    // Clamp input to prevent overflow/underflow
    let x = _mm_max_ps(_mm_set1_ps(-88.0), _mm_min_ps(_mm_set1_ps(88.0), x));

    // n = round(x / ln2)
    let n_f = _mm_mul_ps(x, ln2_inv);
    // Round to nearest integer using convert (rounds to nearest by default)
    let n_i = _mm_cvtps_epi32(n_f);
    let n_f = _mm_cvtepi32_ps(n_i);

    // r = x - n * ln2  (two-step for accuracy)
    let r = _mm_sub_ps(
        _mm_sub_ps(x, _mm_mul_ps(n_f, ln2_hi)),
        _mm_mul_ps(n_f, ln2_lo),
    );

    // Polynomial: c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
    let mut poly = _mm_add_ps(c5, _mm_mul_ps(r, c6));
    poly = _mm_add_ps(c4, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c3, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c2, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c1, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c0, _mm_mul_ps(r, poly));

    // Multiply by 2^n using bit manipulation: reinterpret (n + 127) << 23 as f32.
    // _mm_add_epi32 and _mm_slli_epi32 are SSE2, always available on x86_64.
    let pow2n = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_add_epi32, _mm_slli_epi32};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_add_epi32, _mm_slli_epi32};
        let bias = _mm_set1_epi32(127);
        _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(n_i, bias), 23))
    };

    _mm_mul_ps(poly, pow2n)
}

// ===========================================================================
// AVX fast-exp helper (8-wide)
// ===========================================================================

/// Schraudolph 1999 bit-trick exp for AVX: exp(x) ≈ reinterpret(int(x * 2^23/ln2) + 127*2^23).
/// WHY: ~3x faster than polynomial, ~1e-3 accuracy is sufficient for sigmoid/tanh where 1/(1+exp) dampens error.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn fast_exp_bittrick_avx(x: __m256) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_add_epi32, _mm256_cvtps_epi32, _mm256_set1_epi32};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_add_epi32, _mm256_cvtps_epi32, _mm256_set1_epi32};
    let scale = _mm256_set1_ps(12102203.0); // WHY: 2^23/ln(2) maps float to IEEE 754 exponent field
    let offset = _mm256_set1_epi32(1065353216); // WHY: 127*2^23 is the IEEE 754 exponent bias in integer form
    let clamp_lo = _mm256_set1_ps(-87.0); // WHY: below this exp() produces denormals
    let clamp_hi = _mm256_set1_ps(88.0); // WHY: above this exp() exceeds f32 max
    let x_clamped = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);
    let val = _mm256_cvtps_epi32(_mm256_mul_ps(x_clamped, scale));
    _mm256_castsi256_ps(_mm256_add_epi32(val, offset))
}

/// Polynomial exp for AVX: range-reduction + 6-term Taylor. Higher accuracy (~1e-6)
/// for standalone exp (softmax, etc.) where precision matters more.
/// WHY 6 terms: 6th-order Taylor series for 2^f on [0,1), max error ~1e-7, good accuracy/speed tradeoff.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn fast_exp_avx(x: __m256) -> __m256 {
    let ln2_inv = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm256_set1_ps(0.693_359_4);
    let ln2_lo = _mm256_set1_ps(-2.121_944_4e-4);

    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c6 = _mm256_set1_ps(1.0 / 720.0);

    let x = _mm256_max_ps(
        _mm256_set1_ps(-88.0),
        _mm256_min_ps(_mm256_set1_ps(88.0), x),
    );

    let n_f = _mm256_mul_ps(x, ln2_inv);
    let n_i = _mm256_cvtps_epi32(n_f);
    let n_f = _mm256_cvtepi32_ps(n_i);

    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)),
        _mm256_mul_ps(n_f, ln2_lo),
    );

    let mut poly = _mm256_add_ps(c5, _mm256_mul_ps(r, c6));
    poly = _mm256_add_ps(c4, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c3, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c2, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c1, _mm256_mul_ps(r, poly));
    poly = _mm256_add_ps(c0, _mm256_mul_ps(r, poly));

    let bias = _mm256_set1_epi32(127);
    let pow2n = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm256_add_epi32, _mm256_slli_epi32};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm256_add_epi32, _mm256_slli_epi32};
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n_i, bias), 23))
    };

    _mm256_mul_ps(poly, pow2n)
}

// ===========================================================================
// NEON fast-exp helper (4-wide)
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vreinterpretq_f32_s32, vshlq_n_s32,
    };

    let ln2_inv = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693_359_4);
    let ln2_lo = vdupq_n_f32(-2.121_944_4e-4);

    let c0 = vdupq_n_f32(1.0);
    let c1 = vdupq_n_f32(1.0);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);
    let c5 = vdupq_n_f32(1.0 / 120.0);
    let c6 = vdupq_n_f32(1.0 / 720.0);

    let x = vmaxq_f32(vdupq_n_f32(-88.0), vminq_f32(vdupq_n_f32(88.0), x));

    let n_f = vmulq_f32(x, ln2_inv);
    let n_i = vcvtnq_s32_f32(n_f);
    let n_f = vcvtq_f32_s32(n_i);

    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n_f, ln2_hi)), vmulq_f32(n_f, ln2_lo));

    let mut poly = vfmaq_f32(c5, r, c6);
    poly = vfmaq_f32(c4, r, poly);
    poly = vfmaq_f32(c3, r, poly);
    poly = vfmaq_f32(c2, r, poly);
    poly = vfmaq_f32(c1, r, poly);
    poly = vfmaq_f32(c0, r, poly);

    use std::arch::aarch64::vdupq_n_s32;
    let bias = vdupq_n_s32(127);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_i, bias)));

    vmulq_f32(poly, pow2n)
}

// ===========================================================================
// Exp slice implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn exp_slice_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let v = _mm_loadu_ps(in_ptr.add(index));
        let r = fast_exp_sse(v);
        _mm_storeu_ps(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).exp();
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn exp_slice_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    // 2x unrolled: process 16 floats per iteration to hide FMA latency.
    while index + 16 <= len {
        // Prefetch next cacheline (64 bytes = 16 floats ahead)
        #[cfg(target_arch = "x86")]
        {
            use std::arch::x86::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        let v0 = _mm256_loadu_ps(in_ptr.add(index));
        let v1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let r0 = fast_exp_avx(v0);
        let r1 = fast_exp_avx(v1);
        _mm256_storeu_ps(out_ptr.add(index), r0);
        _mm256_storeu_ps(out_ptr.add(index + 8), r1);
        index += 16;
    }

    // Handle remaining 8-float chunk
    while index + 8 <= len {
        let v = _mm256_loadu_ps(in_ptr.add(index));
        let r = fast_exp_avx(v);
        _mm256_storeu_ps(out_ptr.add(index), r);
        index += 8;
    }

    if index < len {
        exp_slice_sse(&input[index..], &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn exp_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let v = vld1q_f32(in_ptr.add(index));
        let r = fast_exp_neon(v);
        vst1q_f32(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).exp();
        index += 1;
    }
}

// ===========================================================================
// Fused subtract-and-exp: output[i] = exp(input[i] - offset)
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sub_exp_slice_sse(input: &[f32], offset: f32, output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let off = _mm_set1_ps(offset);
    let mut index = 0usize;

    while index + 4 <= len {
        let v = _mm_loadu_ps(in_ptr.add(index));
        let shifted = _mm_sub_ps(v, off);
        let r = fast_exp_sse(shifted);
        _mm_storeu_ps(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index) - offset).exp();
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sub_exp_slice_avx(input: &[f32], offset: f32, output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let off = _mm256_set1_ps(offset);
    let mut index = 0usize;

    // 2x unrolled: process 16 floats per iteration to hide FMA latency.
    while index + 16 <= len {
        #[cfg(target_arch = "x86")]
        {
            use std::arch::x86::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch::<3>(in_ptr.add(index + 16) as *const i8);
        }
        let v0 = _mm256_loadu_ps(in_ptr.add(index));
        let v1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let shifted0 = _mm256_sub_ps(v0, off);
        let shifted1 = _mm256_sub_ps(v1, off);
        let r0 = fast_exp_avx(shifted0);
        let r1 = fast_exp_avx(shifted1);
        _mm256_storeu_ps(out_ptr.add(index), r0);
        _mm256_storeu_ps(out_ptr.add(index + 8), r1);
        index += 16;
    }

    // Handle remaining 8-float chunk
    while index + 8 <= len {
        let v = _mm256_loadu_ps(in_ptr.add(index));
        let shifted = _mm256_sub_ps(v, off);
        let r = fast_exp_avx(shifted);
        _mm256_storeu_ps(out_ptr.add(index), r);
        index += 8;
    }

    if index < len {
        sub_exp_slice_sse(&input[index..], offset, &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sub_exp_slice_neon(input: &[f32], offset: f32, output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let off = vdupq_n_f32(offset);
    let mut index = 0usize;

    while index + 4 <= len {
        let v = vld1q_f32(in_ptr.add(index));
        let shifted = vsubq_f32(v, off);
        let r = fast_exp_neon(shifted);
        vst1q_f32(out_ptr.add(index), r);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index) - offset).exp();
        index += 1;
    }
}

// ===========================================================================
// Sigmoid slice implementations: sigmoid(x) = 1 / (1 + exp(-x))
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sigmoid_slice_sse(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    // Process 16 elements per iteration (4 SSE registers)
    while index + 16 <= len {
        let x0 = _mm_loadu_ps(in_ptr.add(index));
        let x1 = _mm_loadu_ps(in_ptr.add(index + 4));
        let x2 = _mm_loadu_ps(in_ptr.add(index + 8));
        let x3 = _mm_loadu_ps(in_ptr.add(index + 12));

        // Bit-trick exp is sufficient for sigmoid (output clamped 0-1, errors wash out)
        let e0 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x0));
        let e1 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x1));
        let e2 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x2));
        let e3 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x3));

        let r0 = _mm_div_ps(one, _mm_add_ps(one, e0));
        let r1 = _mm_div_ps(one, _mm_add_ps(one, e1));
        let r2 = _mm_div_ps(one, _mm_add_ps(one, e2));
        let r3 = _mm_div_ps(one, _mm_add_ps(one, e3));

        _mm_storeu_ps(out_ptr.add(index), r0);
        _mm_storeu_ps(out_ptr.add(index + 4), r1);
        _mm_storeu_ps(out_ptr.add(index + 8), r2);
        _mm_storeu_ps(out_ptr.add(index + 12), r3);

        index += 16;
    }

    // Remaining 4 at a time
    while index + 4 <= len {
        let x = _mm_loadu_ps(in_ptr.add(index));
        let neg_x = _mm_sub_ps(zero, x);
        let exp_neg_x = fast_exp_bittrick_sse(neg_x);
        let denom = _mm_add_ps(one, exp_neg_x);
        let result = _mm_div_ps(one, denom);
        _mm_storeu_ps(out_ptr.add(index), result);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = sigmoid_scalar(*in_ptr.add(index));
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sigmoid_slice_avx(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    // Process 32 elements per iteration (4 AVX registers)
    while index + 32 <= len {
        let x0 = _mm256_loadu_ps(in_ptr.add(index));
        let x1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let x2 = _mm256_loadu_ps(in_ptr.add(index + 16));
        let x3 = _mm256_loadu_ps(in_ptr.add(index + 24));

        // Use Schraudolph bit-trick exp for ~3x speedup over polynomial
        let e0 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x0));
        let e1 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x1));
        let e2 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x2));
        let e3 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x3));

        let r0 = _mm256_div_ps(one, _mm256_add_ps(one, e0));
        let r1 = _mm256_div_ps(one, _mm256_add_ps(one, e1));
        let r2 = _mm256_div_ps(one, _mm256_add_ps(one, e2));
        let r3 = _mm256_div_ps(one, _mm256_add_ps(one, e3));

        _mm256_storeu_ps(out_ptr.add(index), r0);
        _mm256_storeu_ps(out_ptr.add(index + 8), r1);
        _mm256_storeu_ps(out_ptr.add(index + 16), r2);
        _mm256_storeu_ps(out_ptr.add(index + 24), r3);

        index += 32;
    }

    // Remaining 8 at a time
    while index + 8 <= len {
        let x = _mm256_loadu_ps(in_ptr.add(index));
        let neg_x = _mm256_sub_ps(zero, x);
        let exp_neg_x = fast_exp_bittrick_avx(neg_x);
        let denom = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(one, denom);
        _mm256_storeu_ps(out_ptr.add(index), result);
        index += 8;
    }

    if index < len {
        sigmoid_slice_sse(&input[index..], &mut output[index..]);
    }
}

// (sigmoid_slice_neon defined above at line ~291)

// ===========================================================================
// Tanh slice implementations: tanh(x) = 2 * sigmoid(2x) - 1
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn tanh_slice_sse(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm_div_ps;
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let two = _mm_set1_ps(2.0);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 4 <= len {
        let x = _mm_loadu_ps(in_ptr.add(index));
        let two_x = _mm_mul_ps(two, x);
        // sigmoid(2x) = 1 / (1 + exp(-2x))
        let neg_two_x = _mm_sub_ps(zero, two_x);
        // Use polynomial exp (not bit-trick) for tanh — needs ~1e-4 accuracy
        let exp_neg = fast_exp_sse(neg_two_x);
        let sig = _mm_div_ps(one, _mm_add_ps(one, exp_neg));
        // tanh = 2 * sig - 1
        let result = _mm_sub_ps(_mm_mul_ps(two, sig), one);
        _mm_storeu_ps(out_ptr.add(index), result);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).tanh();
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn tanh_slice_avx(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    while index + 8 <= len {
        let x = _mm256_loadu_ps(in_ptr.add(index));
        let two_x = _mm256_mul_ps(two, x);
        let neg_two_x = _mm256_sub_ps(zero, two_x);
        // Use polynomial exp (not bit-trick) for tanh — needs ~1e-4 accuracy
        let exp_neg = fast_exp_avx(neg_two_x);
        let sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        let result = _mm256_sub_ps(_mm256_mul_ps(two, sig), one);
        _mm256_storeu_ps(out_ptr.add(index), result);
        index += 8;
    }

    if index < len {
        tanh_slice_sse(&input[index..], &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn tanh_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let two = vdupq_n_f32(2.0);
    let one = vdupq_n_f32(1.0);
    let mut index = 0usize;

    // 8x unrolled: 32 elements per iteration, using fast 3-term exp polynomial
    while index + 32 <= len {
        let x0 = vld1q_f32(in_ptr.add(index));
        let x1 = vld1q_f32(in_ptr.add(index + 4));
        let x2 = vld1q_f32(in_ptr.add(index + 8));
        let x3 = vld1q_f32(in_ptr.add(index + 12));
        let x4 = vld1q_f32(in_ptr.add(index + 16));
        let x5 = vld1q_f32(in_ptr.add(index + 20));
        let x6 = vld1q_f32(in_ptr.add(index + 24));
        let x7 = vld1q_f32(in_ptr.add(index + 28));

        // exp(-2x) using fast 3-term polynomial (sufficient for tanh)
        let e0 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x0)));
        let e1 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x1)));
        let e2 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x2)));
        let e3 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x3)));
        let e4 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x4)));
        let e5 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x5)));
        let e6 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x6)));
        let e7 = fast_exp_sigmoid_neon(vnegq_f32(vmulq_f32(two, x7)));

        // tanh(x) = 2 * sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1
        vst1q_f32(
            out_ptr.add(index),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e0)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 4),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e1)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 8),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e2)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 12),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e3)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 16),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e4)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 20),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e5)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 24),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e6)), one),
        );
        vst1q_f32(
            out_ptr.add(index + 28),
            vsubq_f32(vdivq_f32(two, vaddq_f32(one, e7)), one),
        );
        index += 32;
    }

    while index + 4 <= len {
        let x = vld1q_f32(in_ptr.add(index));
        let two_x = vmulq_f32(two, x);
        let neg_two_x = vnegq_f32(two_x);
        let exp_neg = fast_exp_sigmoid_neon(neg_two_x);
        let denom = vaddq_f32(one, exp_neg);
        let result = vsubq_f32(vdivq_f32(two, denom), one);
        vst1q_f32(out_ptr.add(index), result);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).tanh();
        index += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
/// Fused SiLU: output[i] = x * sigmoid(x) in a single pass.
/// 8x unrolled with fast 3-term exp polynomial.
unsafe fn silu_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let mut index = 0usize;

    // 8x unrolled: 32 elements per iteration
    while index + 32 <= len {
        let x0 = vld1q_f32(in_ptr.add(index));
        let x1 = vld1q_f32(in_ptr.add(index + 4));
        let x2 = vld1q_f32(in_ptr.add(index + 8));
        let x3 = vld1q_f32(in_ptr.add(index + 12));
        let x4 = vld1q_f32(in_ptr.add(index + 16));
        let x5 = vld1q_f32(in_ptr.add(index + 20));
        let x6 = vld1q_f32(in_ptr.add(index + 24));
        let x7 = vld1q_f32(in_ptr.add(index + 28));

        // sigmoid(x) = 1 / (1 + exp(-x))
        let e0 = fast_exp_sigmoid_neon(vnegq_f32(x0));
        let e1 = fast_exp_sigmoid_neon(vnegq_f32(x1));
        let e2 = fast_exp_sigmoid_neon(vnegq_f32(x2));
        let e3 = fast_exp_sigmoid_neon(vnegq_f32(x3));
        let e4 = fast_exp_sigmoid_neon(vnegq_f32(x4));
        let e5 = fast_exp_sigmoid_neon(vnegq_f32(x5));
        let e6 = fast_exp_sigmoid_neon(vnegq_f32(x6));
        let e7 = fast_exp_sigmoid_neon(vnegq_f32(x7));

        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        vst1q_f32(
            out_ptr.add(index),
            vmulq_f32(x0, vdivq_f32(one, vaddq_f32(one, e0))),
        );
        vst1q_f32(
            out_ptr.add(index + 4),
            vmulq_f32(x1, vdivq_f32(one, vaddq_f32(one, e1))),
        );
        vst1q_f32(
            out_ptr.add(index + 8),
            vmulq_f32(x2, vdivq_f32(one, vaddq_f32(one, e2))),
        );
        vst1q_f32(
            out_ptr.add(index + 12),
            vmulq_f32(x3, vdivq_f32(one, vaddq_f32(one, e3))),
        );
        vst1q_f32(
            out_ptr.add(index + 16),
            vmulq_f32(x4, vdivq_f32(one, vaddq_f32(one, e4))),
        );
        vst1q_f32(
            out_ptr.add(index + 20),
            vmulq_f32(x5, vdivq_f32(one, vaddq_f32(one, e5))),
        );
        vst1q_f32(
            out_ptr.add(index + 24),
            vmulq_f32(x6, vdivq_f32(one, vaddq_f32(one, e6))),
        );
        vst1q_f32(
            out_ptr.add(index + 28),
            vmulq_f32(x7, vdivq_f32(one, vaddq_f32(one, e7))),
        );
        index += 32;
    }

    while index + 4 <= len {
        let x = vld1q_f32(in_ptr.add(index));
        let e = fast_exp_sigmoid_neon(vnegq_f32(x));
        let sig = vdivq_f32(one, vaddq_f32(one, e));
        vst1q_f32(out_ptr.add(index), vmulq_f32(x, sig));
        index += 4;
    }

    while index < len {
        let x = *in_ptr.add(index);
        let s = 1.0 / (1.0 + (-x).exp());
        *out_ptr.add(index) = x * s;
        index += 1;
    }
}

/// Fused SiLU (x * sigmoid(x)) using SSE.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn silu_slice_sse(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 16 <= len {
        let x0 = _mm_loadu_ps(in_ptr.add(index));
        let x1 = _mm_loadu_ps(in_ptr.add(index + 4));
        let x2 = _mm_loadu_ps(in_ptr.add(index + 8));
        let x3 = _mm_loadu_ps(in_ptr.add(index + 12));

        // Use Schraudolph bit-trick exp for ~3x speedup
        let e0 = fast_exp_sse(_mm_sub_ps(zero, x0));
        let e1 = fast_exp_sse(_mm_sub_ps(zero, x1));
        let e2 = fast_exp_sse(_mm_sub_ps(zero, x2));
        let e3 = fast_exp_sse(_mm_sub_ps(zero, x3));

        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        _mm_storeu_ps(
            out_ptr.add(index),
            _mm_mul_ps(x0, _mm_div_ps(one, _mm_add_ps(one, e0))),
        );
        _mm_storeu_ps(
            out_ptr.add(index + 4),
            _mm_mul_ps(x1, _mm_div_ps(one, _mm_add_ps(one, e1))),
        );
        _mm_storeu_ps(
            out_ptr.add(index + 8),
            _mm_mul_ps(x2, _mm_div_ps(one, _mm_add_ps(one, e2))),
        );
        _mm_storeu_ps(
            out_ptr.add(index + 12),
            _mm_mul_ps(x3, _mm_div_ps(one, _mm_add_ps(one, e3))),
        );

        index += 16;
    }

    while index + 4 <= len {
        let x = _mm_loadu_ps(in_ptr.add(index));
        let e = fast_exp_sse(_mm_sub_ps(zero, x));
        let sig = _mm_div_ps(one, _mm_add_ps(one, e));
        _mm_storeu_ps(out_ptr.add(index), _mm_mul_ps(x, sig));
        index += 4;
    }

    while index < len {
        let v = *in_ptr.add(index);
        let s = 1.0 / (1.0 + (-v).exp());
        *out_ptr.add(index) = v * s;
        index += 1;
    }
}

/// Fused SiLU (x * sigmoid(x)) using AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn silu_slice_avx(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    while index + 32 <= len {
        let x0 = _mm256_loadu_ps(in_ptr.add(index));
        let x1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let x2 = _mm256_loadu_ps(in_ptr.add(index + 16));
        let x3 = _mm256_loadu_ps(in_ptr.add(index + 24));

        // Use Schraudolph bit-trick exp for ~3x speedup
        let e0 = fast_exp_avx(_mm256_sub_ps(zero, x0));
        let e1 = fast_exp_avx(_mm256_sub_ps(zero, x1));
        let e2 = fast_exp_avx(_mm256_sub_ps(zero, x2));
        let e3 = fast_exp_avx(_mm256_sub_ps(zero, x3));

        // silu(x) = x / (1 + exp(-x))
        _mm256_storeu_ps(
            out_ptr.add(index),
            _mm256_mul_ps(x0, _mm256_div_ps(one, _mm256_add_ps(one, e0))),
        );
        _mm256_storeu_ps(
            out_ptr.add(index + 8),
            _mm256_mul_ps(x1, _mm256_div_ps(one, _mm256_add_ps(one, e1))),
        );
        _mm256_storeu_ps(
            out_ptr.add(index + 16),
            _mm256_mul_ps(x2, _mm256_div_ps(one, _mm256_add_ps(one, e2))),
        );
        _mm256_storeu_ps(
            out_ptr.add(index + 24),
            _mm256_mul_ps(x3, _mm256_div_ps(one, _mm256_add_ps(one, e3))),
        );

        index += 32;
    }

    while index + 8 <= len {
        let x = _mm256_loadu_ps(in_ptr.add(index));
        let e = fast_exp_avx(_mm256_sub_ps(zero, x));
        let sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out_ptr.add(index), _mm256_mul_ps(x, sig));
        index += 8;
    }

    if index < len {
        silu_slice_sse(&input[index..], &mut output[index..]);
    }
}

// ===========================================================================
// Max-reduce implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn max_reduce_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc = _mm_set1_ps(f32::NEG_INFINITY);

    while index + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(index));
        acc = _mm_max_ps(acc, v);
        index += 4;
    }

    // Horizontal max of 4-lane accumulator
    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0].max(buf[1]).max(buf[2]).max(buf[3]);

    while index < len {
        result = result.max(*ptr.add(index));
        index += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn max_reduce_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);

    while index + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(index));
        acc = _mm256_max_ps(acc, v);
        index += 8;
    }

    // Horizontal max of 8-lane accumulator
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0];
    for i in 1..8 {
        result = result.max(buf[i]);
    }

    while index < len {
        result = result.max(*ptr.add(index));
        index += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn max_reduce_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vmaxvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);

    while index + 4 <= len {
        let v = vld1q_f32(ptr.add(index));
        acc = vmaxq_f32(acc, v);
        index += 4;
    }

    let mut result = vmaxvq_f32(acc);
    while index < len {
        result = result.max(*ptr.add(index));
        index += 1;
    }
    result
}

// ===========================================================================
// Add-reduce (sum) implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn add_reduce_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc = _mm_setzero_ps();

    while index + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(index));
        acc = _mm_add_ps(acc, v);
        index += 4;
    }

    // Horizontal sum of 4-lane accumulator
    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3];

    while index < len {
        result += *ptr.add(index);
        index += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_reduce_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc = _mm256_setzero_ps();

    while index + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(index));
        acc = _mm256_add_ps(acc, v);
        index += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    while index < len {
        result += *ptr.add(index);
        index += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_reduce_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vaddvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut index = 0usize;
    let mut acc = vdupq_n_f32(0.0);

    while index + 4 <= len {
        let v = vld1q_f32(ptr.add(index));
        acc = vaddq_f32(acc, v);
        index += 4;
    }

    let mut result = vaddvq_f32(acc);
    while index < len {
        result += *ptr.add(index);
        index += 1;
    }
    result
}

// ===========================================================================
// FMA slice implementations: acc[i] += a[i] * b[i]
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn fma_slice_sse(a: &[f32], b: &[f32], acc: &mut [f32]) {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let acc_ptr = acc.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let av = _mm_loadu_ps(a_ptr.add(index));
        let bv = _mm_loadu_ps(b_ptr.add(index));
        let cv = _mm_loadu_ps(acc_ptr.add(index));
        let result = _mm_add_ps(cv, _mm_mul_ps(av, bv));
        _mm_storeu_ps(acc_ptr.add(index), result);
        index += 4;
    }

    if index < len {
        fma_slice_scalar(&a[index..], &b[index..], &mut acc[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn fma_slice_avx(a: &[f32], b: &[f32], acc: &mut [f32]) {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let acc_ptr = acc.as_mut_ptr();
    let mut index = 0usize;

    while index + 8 <= len {
        let av = _mm256_loadu_ps(a_ptr.add(index));
        let bv = _mm256_loadu_ps(b_ptr.add(index));
        let cv = _mm256_loadu_ps(acc_ptr.add(index));
        let result = _mm256_add_ps(cv, _mm256_mul_ps(av, bv));
        _mm256_storeu_ps(acc_ptr.add(index), result);
        index += 8;
    }

    if index < len {
        fma_slice_sse(&a[index..], &b[index..], &mut acc[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn fma_slice_neon(a: &[f32], b: &[f32], acc: &mut [f32]) {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let acc_ptr = acc.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let av = vld1q_f32(a_ptr.add(index));
        let bv = vld1q_f32(b_ptr.add(index));
        let cv = vld1q_f32(acc_ptr.add(index));
        let result = vfmaq_f32(cv, av, bv);
        vst1q_f32(acc_ptr.add(index), result);
        index += 4;
    }

    if index < len {
        fma_slice_scalar(&a[index..], &b[index..], &mut acc[index..]);
    }
}

// ===========================================================================
// ReLU SIMD implementations (existing)
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn relu_slice_sse(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 4 <= len {
        let input = _mm_loadu_ps(ptr.add(index));
        let out = _mm_max_ps(input, zero);
        _mm_storeu_ps(ptr.add(index), out);
        index += 4;
    }

    if index < len {
        relu_slice_scalar(&mut values[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn relu_slice_avx(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    // 4× unrolled: 32 elements per iteration
    while index + 32 <= len {
        let v0 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index)), zero);
        let v1 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index + 8)), zero);
        let v2 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index + 16)), zero);
        let v3 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index + 24)), zero);
        _mm256_storeu_ps(ptr.add(index), v0);
        _mm256_storeu_ps(ptr.add(index + 8), v1);
        _mm256_storeu_ps(ptr.add(index + 16), v2);
        _mm256_storeu_ps(ptr.add(index + 24), v3);
        index += 32;
    }

    while index + 8 <= len {
        _mm256_storeu_ps(
            ptr.add(index),
            _mm256_max_ps(_mm256_loadu_ps(ptr.add(index)), zero),
        );
        index += 8;
    }

    if index < len {
        relu_slice_sse(&mut values[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn binary_same_shape_sse(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let left = _mm_loadu_ps(left_ptr.add(index));
        let right = _mm_loadu_ps(right_ptr.add(index));
        let result = match kind {
            BinaryKind::Add => _mm_add_ps(left, right),
            BinaryKind::Sub => _mm_sub_ps(left, right),
            BinaryKind::Mul => _mm_mul_ps(left, right),
        };
        _mm_storeu_ps(out_ptr.add(index), result);
        index += 4;
    }

    if index < len {
        binary_same_shape_scalar(&lhs[index..], &rhs[index..], &mut out[index..], kind);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn binary_same_shape_avx(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    // 4x unrolled: process 32 floats per iteration with software prefetch.
    // Matches vDSP throughput by keeping the OoO pipeline fully saturated.
    match kind {
        BinaryKind::Add => {
            while index + 32 <= len {
                #[cfg(target_arch = "x86")]
                {
                    use std::arch::x86::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                let a0 = _mm256_loadu_ps(left_ptr.add(index));
                let b0 = _mm256_loadu_ps(right_ptr.add(index));
                let a1 = _mm256_loadu_ps(left_ptr.add(index + 8));
                let b1 = _mm256_loadu_ps(right_ptr.add(index + 8));
                _mm256_storeu_ps(out_ptr.add(index), _mm256_add_ps(a0, b0));
                _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_add_ps(a1, b1));
                let a2 = _mm256_loadu_ps(left_ptr.add(index + 16));
                let b2 = _mm256_loadu_ps(right_ptr.add(index + 16));
                let a3 = _mm256_loadu_ps(left_ptr.add(index + 24));
                let b3 = _mm256_loadu_ps(right_ptr.add(index + 24));
                _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_add_ps(a2, b2));
                _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_add_ps(a3, b3));
                index += 32;
            }
        }
        BinaryKind::Sub => {
            while index + 32 <= len {
                #[cfg(target_arch = "x86")]
                {
                    use std::arch::x86::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                let a0 = _mm256_loadu_ps(left_ptr.add(index));
                let b0 = _mm256_loadu_ps(right_ptr.add(index));
                let a1 = _mm256_loadu_ps(left_ptr.add(index + 8));
                let b1 = _mm256_loadu_ps(right_ptr.add(index + 8));
                _mm256_storeu_ps(out_ptr.add(index), _mm256_sub_ps(a0, b0));
                _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_sub_ps(a1, b1));
                let a2 = _mm256_loadu_ps(left_ptr.add(index + 16));
                let b2 = _mm256_loadu_ps(right_ptr.add(index + 16));
                let a3 = _mm256_loadu_ps(left_ptr.add(index + 24));
                let b3 = _mm256_loadu_ps(right_ptr.add(index + 24));
                _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_sub_ps(a2, b2));
                _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_sub_ps(a3, b3));
                index += 32;
            }
        }
        BinaryKind::Mul => {
            while index + 32 <= len {
                #[cfg(target_arch = "x86")]
                {
                    use std::arch::x86::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch::<3>(left_ptr.add(index + 32) as *const i8);
                    _mm_prefetch::<3>(right_ptr.add(index + 32) as *const i8);
                }
                let a0 = _mm256_loadu_ps(left_ptr.add(index));
                let b0 = _mm256_loadu_ps(right_ptr.add(index));
                let a1 = _mm256_loadu_ps(left_ptr.add(index + 8));
                let b1 = _mm256_loadu_ps(right_ptr.add(index + 8));
                _mm256_storeu_ps(out_ptr.add(index), _mm256_mul_ps(a0, b0));
                _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_mul_ps(a1, b1));
                let a2 = _mm256_loadu_ps(left_ptr.add(index + 16));
                let b2 = _mm256_loadu_ps(right_ptr.add(index + 16));
                let a3 = _mm256_loadu_ps(left_ptr.add(index + 24));
                let b3 = _mm256_loadu_ps(right_ptr.add(index + 24));
                _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_mul_ps(a2, b2));
                _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_mul_ps(a3, b3));
                index += 32;
            }
        }
    }

    // Handle remaining elements 8 at a time
    while index + 8 <= len {
        let left = _mm256_loadu_ps(left_ptr.add(index));
        let right = _mm256_loadu_ps(right_ptr.add(index));
        let result = match kind {
            BinaryKind::Add => _mm256_add_ps(left, right),
            BinaryKind::Sub => _mm256_sub_ps(left, right),
            BinaryKind::Mul => _mm256_mul_ps(left, right),
        };
        _mm256_storeu_ps(out_ptr.add(index), result);
        index += 8;
    }

    if index < len {
        binary_same_shape_sse(&lhs[index..], &rhs[index..], &mut out[index..], kind);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn relu_slice_neon(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = vdupq_n_f32(0.0);
    let mut index = 0usize;

    // 8× unrolled: 32 elements per iteration
    while index + 32 <= len {
        let v0 = vmaxq_f32(vld1q_f32(ptr.add(index)), zero);
        let v1 = vmaxq_f32(vld1q_f32(ptr.add(index + 4)), zero);
        let v2 = vmaxq_f32(vld1q_f32(ptr.add(index + 8)), zero);
        let v3 = vmaxq_f32(vld1q_f32(ptr.add(index + 12)), zero);
        let v4 = vmaxq_f32(vld1q_f32(ptr.add(index + 16)), zero);
        let v5 = vmaxq_f32(vld1q_f32(ptr.add(index + 20)), zero);
        let v6 = vmaxq_f32(vld1q_f32(ptr.add(index + 24)), zero);
        let v7 = vmaxq_f32(vld1q_f32(ptr.add(index + 28)), zero);
        vst1q_f32(ptr.add(index), v0);
        vst1q_f32(ptr.add(index + 4), v1);
        vst1q_f32(ptr.add(index + 8), v2);
        vst1q_f32(ptr.add(index + 12), v3);
        vst1q_f32(ptr.add(index + 16), v4);
        vst1q_f32(ptr.add(index + 20), v5);
        vst1q_f32(ptr.add(index + 24), v6);
        vst1q_f32(ptr.add(index + 28), v7);
        index += 32;
    }

    while index + 4 <= len {
        vst1q_f32(ptr.add(index), vmaxq_f32(vld1q_f32(ptr.add(index)), zero));
        index += 4;
    }

    if index < len {
        relu_slice_scalar(&mut values[index..]);
    }
}

// ===========================================================================
// Two-argument ReLU SIMD implementations (input -> output)
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn relu_to_slice_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 4 <= len {
        let v = _mm_loadu_ps(in_ptr.add(index));
        let r = _mm_max_ps(v, zero);
        _mm_storeu_ps(out_ptr.add(index), r);
        index += 4;
    }

    if index < len {
        relu_to_slice_scalar(&input[index..], &mut output[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn relu_to_slice_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    // 4× unrolled: 32 elements per iteration (matches NEON unrolling)
    while index + 32 <= len {
        let a0 = _mm256_loadu_ps(in_ptr.add(index));
        let a1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let a2 = _mm256_loadu_ps(in_ptr.add(index + 16));
        let a3 = _mm256_loadu_ps(in_ptr.add(index + 24));
        _mm256_storeu_ps(out_ptr.add(index), _mm256_max_ps(a0, zero));
        _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_max_ps(a1, zero));
        _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_max_ps(a2, zero));
        _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_max_ps(a3, zero));
        index += 32;
    }

    while index + 8 <= len {
        _mm256_storeu_ps(
            out_ptr.add(index),
            _mm256_max_ps(_mm256_loadu_ps(in_ptr.add(index)), zero),
        );
        index += 8;
    }

    if index < len {
        relu_to_slice_sse(&input[index..], &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn relu_to_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = vdupq_n_f32(0.0);
    let mut index = 0usize;

    // 8× unrolled with interleaved load/compute/store for better OoO pipelining
    while index + 32 <= len {
        let a0 = vld1q_f32(in_ptr.add(index));
        let a1 = vld1q_f32(in_ptr.add(index + 4));
        let a2 = vld1q_f32(in_ptr.add(index + 8));
        let a3 = vld1q_f32(in_ptr.add(index + 12));
        vst1q_f32(out_ptr.add(index), vmaxq_f32(a0, zero));
        vst1q_f32(out_ptr.add(index + 4), vmaxq_f32(a1, zero));
        let a4 = vld1q_f32(in_ptr.add(index + 16));
        let a5 = vld1q_f32(in_ptr.add(index + 20));
        vst1q_f32(out_ptr.add(index + 8), vmaxq_f32(a2, zero));
        vst1q_f32(out_ptr.add(index + 12), vmaxq_f32(a3, zero));
        let a6 = vld1q_f32(in_ptr.add(index + 24));
        let a7 = vld1q_f32(in_ptr.add(index + 28));
        vst1q_f32(out_ptr.add(index + 16), vmaxq_f32(a4, zero));
        vst1q_f32(out_ptr.add(index + 20), vmaxq_f32(a5, zero));
        vst1q_f32(out_ptr.add(index + 24), vmaxq_f32(a6, zero));
        vst1q_f32(out_ptr.add(index + 28), vmaxq_f32(a7, zero));
        index += 32;
    }

    while index + 4 <= len {
        vst1q_f32(
            out_ptr.add(index),
            vmaxq_f32(vld1q_f32(in_ptr.add(index)), zero),
        );
        index += 4;
    }

    if index < len {
        relu_to_slice_scalar(&input[index..], &mut output[index..]);
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn binary_same_shape_neon(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let left_ptr = lhs.as_ptr();
    let right_ptr = rhs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut index = 0usize;

    while index + 4 <= len {
        let left = vld1q_f32(left_ptr.add(index));
        let right = vld1q_f32(right_ptr.add(index));
        let result = match kind {
            BinaryKind::Add => vaddq_f32(left, right),
            BinaryKind::Sub => vsubq_f32(left, right),
            BinaryKind::Mul => vmulq_f32(left, right),
        };
        vst1q_f32(out_ptr.add(index), result);
        index += 4;
    }

    if index < len {
        binary_same_shape_scalar(&lhs[index..], &rhs[index..], &mut out[index..], kind);
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated matmul inner loop
// ---------------------------------------------------------------------------
//
// Computes one output row of C = A * B by iterating over the shared dimension k.
// For each k, broadcasts a[row*K + k] and multiplies by the contiguous B row
// b[k*N .. k*N + N], accumulating into the output row out[0..N].
//
// The "broadcast A, contiguous B row" access pattern is SIMD-friendly because
// all loads from B are contiguous.

/// Dispatch to the best available SIMD path for a single matmul output row.
///
/// # Safety
/// - `left_row` must point to at least `k` valid f32 elements.
/// - `right` must point to at least `k * n` valid f32 elements (row-major B).
/// - `out_row` must point to at least `n` valid f32 elements.
/// - The caller must ensure no aliasing between `out_row` and the input pointers.
#[inline]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn matmul_row_dispatch(
    left_row: *const f32,
    right: *const f32,
    out_row: *mut f32,
    k: usize,
    n: usize,
) {
    if cfg!(miri) {
        matmul_row_scalar(left_row, right, out_row, k, n);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            matmul_row_avx(left_row, right, out_row, k, n);
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            matmul_row_sse(left_row, right, out_row, k, n);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            matmul_row_neon(left_row, right, out_row, k, n);
            return;
        }
    }

    matmul_row_scalar(left_row, right, out_row, k, n);
}

/// Scalar fallback: broadcast-multiply-accumulate, unrolled by 4.
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn matmul_row_scalar(
    left_row: *const f32,
    right: *const f32,
    out_row: *mut f32,
    k: usize,
    n: usize,
) {
    for p in 0..k {
        let a_val = *left_row.add(p);
        let b_row = right.add(p * n);

        let mut col = 0usize;
        while col + 4 <= n {
            *out_row.add(col) += a_val * *b_row.add(col);
            *out_row.add(col + 1) += a_val * *b_row.add(col + 1);
            *out_row.add(col + 2) += a_val * *b_row.add(col + 2);
            *out_row.add(col + 3) += a_val * *b_row.add(col + 3);
            col += 4;
        }
        while col < n {
            *out_row.add(col) += a_val * *b_row.add(col);
            col += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn matmul_row_sse(
    left_row: *const f32,
    right: *const f32,
    out_row: *mut f32,
    k: usize,
    n: usize,
) {
    for p in 0..k {
        let a_val = _mm_set1_ps(*left_row.add(p));
        let b_row = right.add(p * n);

        let mut col = 0usize;
        while col + 4 <= n {
            let b_vec = _mm_loadu_ps(b_row.add(col));
            let out_vec = _mm_loadu_ps(out_row.add(col));
            let result = _mm_add_ps(out_vec, _mm_mul_ps(a_val, b_vec));
            _mm_storeu_ps(out_row.add(col), result);
            col += 4;
        }
        while col < n {
            *out_row.add(col) += *left_row.add(p) * *b_row.add(col);
            col += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn matmul_row_avx(
    left_row: *const f32,
    right: *const f32,
    out_row: *mut f32,
    k: usize,
    n: usize,
) {
    for p in 0..k {
        let a_val_avx = _mm256_set1_ps(*left_row.add(p));
        let a_val_sse = _mm_set1_ps(*left_row.add(p));
        let b_row = right.add(p * n);

        let mut col = 0usize;
        while col + 8 <= n {
            let b_vec = _mm256_loadu_ps(b_row.add(col));
            let out_vec = _mm256_loadu_ps(out_row.add(col));
            let result = _mm256_add_ps(out_vec, _mm256_mul_ps(a_val_avx, b_vec));
            _mm256_storeu_ps(out_row.add(col), result);
            col += 8;
        }
        // Handle 4-element remainder with SSE.
        while col + 4 <= n {
            let b_vec = _mm_loadu_ps(b_row.add(col));
            let out_vec = _mm_loadu_ps(out_row.add(col));
            let result = _mm_add_ps(out_vec, _mm_mul_ps(a_val_sse, b_vec));
            _mm_storeu_ps(out_row.add(col), result);
            col += 4;
        }
        while col < n {
            *out_row.add(col) += *left_row.add(p) * *b_row.add(col);
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn matmul_row_neon(
    left_row: *const f32,
    right: *const f32,
    out_row: *mut f32,
    k: usize,
    n: usize,
) {
    for p in 0..k {
        let a_val: float32x4_t = vdupq_n_f32(*left_row.add(p));
        let b_row = right.add(p * n);

        let mut col = 0usize;
        while col + 4 <= n {
            let b_vec = vld1q_f32(b_row.add(col));
            let out_vec = vld1q_f32(out_row.add(col));
            let result = vfmaq_f32(out_vec, a_val, b_vec);
            vst1q_f32(out_row.add(col), result);
            col += 4;
        }
        while col < n {
            *out_row.add(col) += *left_row.add(p) * *b_row.add(col);
            col += 1;
        }
    }
}

// ===========================================================================
// Fused softmax: max + sub-exp + sum + divide in one function
// ===========================================================================

/// Fused softmax row: `out[i] = exp(input[i] - max) / sum(exp(input - max))`.
///
/// Performs all four steps (max, subtract+exp, sum, divide) inside a single
/// function so that data stays in L1 cache and dispatcher overhead is eliminated.
#[allow(unsafe_code)]
#[inline]
pub fn softmax_row_fused_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) || input.is_empty() {
        softmax_row_fused_scalar(input, output);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                softmax_row_fused_neon(input, output);
            }
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                softmax_row_fused_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                softmax_row_fused_sse(input, output);
            }
            return;
        }
    }

    softmax_row_fused_scalar(input, output);
}

fn softmax_row_fused_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }

    // 1. max
    let mut max_val = f32::NEG_INFINITY;
    for &v in input {
        max_val = max_val.max(v);
    }

    // 2. sub+exp + 3. accumulate sum
    let mut sum_exp = 0.0f32;
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        let e = (v - max_val).exp();
        *o = e;
        sum_exp += e;
    }

    // 4. divide
    let inv = 1.0 / sum_exp;
    for o in output.iter_mut() {
        *o *= inv;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn softmax_row_fused_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::{vaddvq_f32, vmaxvq_f32};

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. Find max (NEON reduce)
    let mut acc_max = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 16 <= len {
        let v0 = vld1q_f32(in_ptr.add(i));
        let v1 = vld1q_f32(in_ptr.add(i + 4));
        let v2 = vld1q_f32(in_ptr.add(i + 8));
        let v3 = vld1q_f32(in_ptr.add(i + 12));
        acc_max = vmaxq_f32(acc_max, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        i += 16;
    }
    while i + 4 <= len {
        let v = vld1q_f32(in_ptr.add(i));
        acc_max = vmaxq_f32(acc_max, v);
        i += 4;
    }
    let mut max_val = vmaxvq_f32(acc_max);
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sub+exp (NEON fast_exp, writes output) + 3. accumulate sum
    let off = vdupq_n_f32(max_val);
    let mut acc_sum = vdupq_n_f32(0.0);
    i = 0;
    while i + 16 <= len {
        let v0 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        let v1 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 4)), off));
        let v2 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 8)), off));
        let v3 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 12)), off));
        vst1q_f32(out_ptr.add(i), v0);
        vst1q_f32(out_ptr.add(i + 4), v1);
        vst1q_f32(out_ptr.add(i + 8), v2);
        vst1q_f32(out_ptr.add(i + 12), v3);
        acc_sum = vaddq_f32(acc_sum, vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
        i += 16;
    }
    while i + 4 <= len {
        let v = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        vst1q_f32(out_ptr.add(i), v);
        acc_sum = vaddq_f32(acc_sum, v);
        i += 4;
    }
    let mut sum_exp = vaddvq_f32(acc_sum);
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    // 4. divide (NEON multiply by 1/sum)
    let inv = vdupq_n_f32(1.0 / sum_exp);
    i = 0;
    while i + 16 <= len {
        vst1q_f32(out_ptr.add(i), vmulq_f32(vld1q_f32(out_ptr.add(i)), inv));
        vst1q_f32(
            out_ptr.add(i + 4),
            vmulq_f32(vld1q_f32(out_ptr.add(i + 4)), inv),
        );
        vst1q_f32(
            out_ptr.add(i + 8),
            vmulq_f32(vld1q_f32(out_ptr.add(i + 8)), inv),
        );
        vst1q_f32(
            out_ptr.add(i + 12),
            vmulq_f32(vld1q_f32(out_ptr.add(i + 12)), inv),
        );
        i += 16;
    }
    while i + 4 <= len {
        vst1q_f32(out_ptr.add(i), vmulq_f32(vld1q_f32(out_ptr.add(i)), inv));
        i += 4;
    }
    let inv_s = 1.0 / sum_exp;
    while i < len {
        *out_ptr.add(i) *= inv_s;
        i += 1;
    }
}

/// SSE fused softmax fallback.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn softmax_row_fused_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 4 <= len {
        acc_max = _mm_max_ps(acc_max, _mm_loadu_ps(in_ptr.add(i)));
        i += 4;
    }
    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc_max);
    let mut max_val = buf[0].max(buf[1]).max(buf[2].max(buf[3]));
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sub+exp + 3. sum
    let off = _mm_set1_ps(max_val);
    let mut acc_sum = _mm_setzero_ps();
    i = 0;
    while i + 4 <= len {
        let v = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off));
        _mm_storeu_ps(out_ptr.add(i), v);
        acc_sum = _mm_add_ps(acc_sum, v);
        i += 4;
    }
    _mm_storeu_ps(buf.as_mut_ptr(), acc_sum);
    let mut sum_exp = buf[0] + buf[1] + buf[2] + buf[3];
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    // 4. divide
    let inv = _mm_set1_ps(1.0 / sum_exp);
    i = 0;
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_mul_ps(_mm_loadu_ps(out_ptr.add(i)), inv),
        );
        i += 4;
    }
    let inv_s = 1.0 / sum_exp;
    while i < len {
        *out_ptr.add(i) *= inv_s;
        i += 1;
    }
}

/// AVX fused softmax fallback — delegates tail to SSE.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn softmax_row_fused_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 8 <= len {
        acc_max = _mm256_max_ps(acc_max, _mm256_loadu_ps(in_ptr.add(i)));
        i += 8;
    }
    let mut buf8 = [0.0f32; 8];
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_max);
    let mut max_val = buf8[0];
    for &v in &buf8[1..] {
        max_val = max_val.max(v);
    }
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sub+exp + 3. sum
    let off = _mm256_set1_ps(max_val);
    let mut acc_sum = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= len {
        let v = fast_exp_avx(_mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), off));
        _mm256_storeu_ps(out_ptr.add(i), v);
        acc_sum = _mm256_add_ps(acc_sum, v);
        i += 8;
    }
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_sum);
    let mut sum_exp: f32 = buf8.iter().sum();
    // SSE tail for remaining < 8 elements
    let off4 = _mm_set1_ps(max_val);
    while i + 4 <= len {
        let v = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off4));
        _mm_storeu_ps(out_ptr.add(i), v);
        let mut b4 = [0.0f32; 4];
        _mm_storeu_ps(b4.as_mut_ptr(), v);
        sum_exp += b4[0] + b4[1] + b4[2] + b4[3];
        i += 4;
    }
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    // 4. divide
    let inv8 = _mm256_set1_ps(1.0 / sum_exp);
    i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(
            out_ptr.add(i),
            _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(i)), inv8),
        );
        i += 8;
    }
    let inv4 = _mm_set1_ps(1.0 / sum_exp);
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_mul_ps(_mm_loadu_ps(out_ptr.add(i)), inv4),
        );
        i += 4;
    }
    let inv_s = 1.0 / sum_exp;
    while i < len {
        *out_ptr.add(i) *= inv_s;
        i += 1;
    }
}

// ===========================================================================
// Fused log-softmax: out[i] = x[i] - max - log(sum(exp(x - max)))
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub fn log_softmax_row_fused_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) || input.is_empty() {
        log_softmax_row_fused_scalar(input, output);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                log_softmax_row_fused_neon(input, output);
            }
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                log_softmax_row_fused_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                log_softmax_row_fused_sse(input, output);
            }
            return;
        }
    }

    log_softmax_row_fused_scalar(input, output);
}

fn log_softmax_row_fused_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }

    // 1. max
    let mut max_val = f32::NEG_INFINITY;
    for &v in input {
        max_val = max_val.max(v);
    }

    // 2. sum(exp(x - max))
    let mut sum_exp = 0.0f32;
    for &v in input {
        sum_exp += (v - max_val).exp();
    }

    // 3. output[i] = x[i] - max - log(sum_exp)
    let log_denom = max_val + sum_exp.ln();
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = v - log_denom;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn log_softmax_row_fused_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::{vaddvq_f32, vmaxvq_f32};

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. Find max (NEON reduce)
    let mut acc_max = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 16 <= len {
        let v0 = vld1q_f32(in_ptr.add(i));
        let v1 = vld1q_f32(in_ptr.add(i + 4));
        let v2 = vld1q_f32(in_ptr.add(i + 8));
        let v3 = vld1q_f32(in_ptr.add(i + 12));
        acc_max = vmaxq_f32(acc_max, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        i += 16;
    }
    while i + 4 <= len {
        acc_max = vmaxq_f32(acc_max, vld1q_f32(in_ptr.add(i)));
        i += 4;
    }
    let mut max_val = vmaxvq_f32(acc_max);
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sum(exp(x - max))
    let off = vdupq_n_f32(max_val);
    let mut acc_sum = vdupq_n_f32(0.0);
    i = 0;
    while i + 16 <= len {
        let e0 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        let e1 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 4)), off));
        let e2 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 8)), off));
        let e3 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 12)), off));
        acc_sum = vaddq_f32(acc_sum, vaddq_f32(vaddq_f32(e0, e1), vaddq_f32(e2, e3)));
        i += 16;
    }
    while i + 4 <= len {
        let e = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        acc_sum = vaddq_f32(acc_sum, e);
        i += 4;
    }
    let mut sum_exp = vaddvq_f32(acc_sum);
    while i < len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
        i += 1;
    }

    // 3. output[i] = x[i] - (max + log(sum_exp))
    let log_denom = vdupq_n_f32(max_val + sum_exp.ln());
    i = 0;
    while i + 16 <= len {
        vst1q_f32(
            out_ptr.add(i),
            vsubq_f32(vld1q_f32(in_ptr.add(i)), log_denom),
        );
        vst1q_f32(
            out_ptr.add(i + 4),
            vsubq_f32(vld1q_f32(in_ptr.add(i + 4)), log_denom),
        );
        vst1q_f32(
            out_ptr.add(i + 8),
            vsubq_f32(vld1q_f32(in_ptr.add(i + 8)), log_denom),
        );
        vst1q_f32(
            out_ptr.add(i + 12),
            vsubq_f32(vld1q_f32(in_ptr.add(i + 12)), log_denom),
        );
        i += 16;
    }
    while i + 4 <= len {
        vst1q_f32(
            out_ptr.add(i),
            vsubq_f32(vld1q_f32(in_ptr.add(i)), log_denom),
        );
        i += 4;
    }
    let log_denom_s = max_val + sum_exp.ln();
    while i < len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_denom_s;
        i += 1;
    }
}

/// SSE fused log-softmax.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn log_softmax_row_fused_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 4 <= len {
        acc_max = _mm_max_ps(acc_max, _mm_loadu_ps(in_ptr.add(i)));
        i += 4;
    }
    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc_max);
    let mut max_val = buf[0].max(buf[1]).max(buf[2].max(buf[3]));
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sum(exp(x - max))
    let off = _mm_set1_ps(max_val);
    let mut acc_sum = _mm_setzero_ps();
    i = 0;
    while i + 4 <= len {
        let e = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off));
        acc_sum = _mm_add_ps(acc_sum, e);
        i += 4;
    }
    _mm_storeu_ps(buf.as_mut_ptr(), acc_sum);
    let mut sum_exp = buf[0] + buf[1] + buf[2] + buf[3];
    while i < len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
        i += 1;
    }

    // 3. output[i] = x[i] - (max + log(sum_exp))
    let log_denom = _mm_set1_ps(max_val + sum_exp.ln());
    i = 0;
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), log_denom),
        );
        i += 4;
    }
    let log_denom_s = max_val + sum_exp.ln();
    while i < len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_denom_s;
        i += 1;
    }
}

/// AVX fused log-softmax.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn log_softmax_row_fused_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 8 <= len {
        acc_max = _mm256_max_ps(acc_max, _mm256_loadu_ps(in_ptr.add(i)));
        i += 8;
    }
    let mut buf8 = [0.0f32; 8];
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_max);
    let mut max_val = buf8[0];
    for &v in &buf8[1..] {
        max_val = max_val.max(v);
    }
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sum(exp(x - max))
    let off = _mm256_set1_ps(max_val);
    let mut acc_sum = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= len {
        let e = fast_exp_avx(_mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), off));
        acc_sum = _mm256_add_ps(acc_sum, e);
        i += 8;
    }
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_sum);
    let mut sum_exp: f32 = buf8.iter().sum();
    // SSE tail for remaining < 8 elements
    let off4 = _mm_set1_ps(max_val);
    while i + 4 <= len {
        let e = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off4));
        let mut b4 = [0.0f32; 4];
        _mm_storeu_ps(b4.as_mut_ptr(), e);
        sum_exp += b4[0] + b4[1] + b4[2] + b4[3];
        i += 4;
    }
    while i < len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
        i += 1;
    }

    // 3. output[i] = x[i] - (max + log(sum_exp))
    let log_denom_val = max_val + sum_exp.ln();
    let log_denom8 = _mm256_set1_ps(log_denom_val);
    i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(
            out_ptr.add(i),
            _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), log_denom8),
        );
        i += 8;
    }
    let log_denom4 = _mm_set1_ps(log_denom_val);
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), log_denom4),
        );
        i += 4;
    }
    while i < len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_denom_val;
        i += 1;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            assert!(d <= tol, "index {i}: {x} vs {y}, diff={d}, tolerance={tol}");
        }
    }

    #[test]
    fn exp_matches_scalar() {
        let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        let mut simd_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];

        exp_slice_dispatch(&input, &mut simd_out);
        exp_slice_scalar(&input, &mut scalar_out);

        // Degree-6 Taylor polynomial is accurate to roughly 1e-6 relative error
        for (i, (&s, &r)) in simd_out.iter().zip(scalar_out.iter()).enumerate() {
            let rel = if r.abs() > 1e-10 {
                (s - r).abs() / r.abs()
            } else {
                (s - r).abs()
            };
            assert!(
                rel < 1e-5,
                "exp mismatch at index {i}: simd={s}, scalar={r}, rel_err={rel}"
            );
        }
    }

    #[test]
    fn sigmoid_dispatch_matches_scalar() {
        let input: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.3).collect();
        let mut simd_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];

        sigmoid_slice_dispatch(&input, &mut simd_out);
        sigmoid_slice_dispatch_scalar(&input, &mut scalar_out);

        // Sigmoid uses Schraudolph bit-trick exp (~4% max error on exp,
        // but sigmoid squashes error near 0/1, practical max ~0.03).
        assert_close(&simd_out, &scalar_out, 0.035);
    }

    #[test]
    fn tanh_dispatch_matches_scalar() {
        let input: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.3).collect();
        let mut simd_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];

        tanh_slice_dispatch(&input, &mut simd_out);
        tanh_slice_dispatch_scalar(&input, &mut scalar_out);

        // Uses fast 3-term exp polynomial for sigmoid path (~2e-3 max error vs scalar tanh).
        assert_close(&simd_out, &scalar_out, 2e-3);
    }

    #[test]
    fn max_reduce_matches_scalar() {
        let data: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7 - 12.0).sin()).collect();
        let simd_result = max_reduce_dispatch(&data);
        let scalar_result = max_reduce_scalar(&data);
        assert!((simd_result - scalar_result).abs() < 1e-6);
    }

    #[test]
    fn max_reduce_empty() {
        assert_eq!(max_reduce_dispatch(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn add_reduce_matches_scalar() {
        let data: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
        let simd_result = add_reduce_dispatch(&data);
        let scalar_result = add_reduce_scalar(&data);
        assert!(
            (simd_result - scalar_result).abs() < 1e-3,
            "simd={simd_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn add_reduce_empty() {
        assert_eq!(add_reduce_dispatch(&[]), 0.0);
    }

    #[test]
    #[allow(unsafe_code)]
    fn fma_matches_scalar() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..33).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut simd_acc = vec![1.0f32; 33];
        let mut scalar_acc = vec![1.0f32; 33];

        fma_slice_dispatch(&a, &b, &mut simd_acc);
        unsafe { fma_slice_scalar(&a, &b, &mut scalar_acc) };

        assert_close(&simd_acc, &scalar_acc, 1e-5);
    }

    #[test]
    fn sigmoid_dispatch_boundary_values() {
        // Verify sigmoid at key points
        let input = vec![-100.0, -10.0, 0.0, 10.0, 100.0];
        let mut output = vec![0.0f32; 5];
        sigmoid_slice_dispatch(&input, &mut output);

        // sigmoid(-100) ~ 0, sigmoid(0) = 0.5, sigmoid(100) ~ 1
        assert!(
            output[0] < 0.01,
            "sigmoid(-100) should be near 0: {}",
            output[0]
        );
        assert!(
            (output[2] - 0.5).abs() < 0.01,
            "sigmoid(0) should be near 0.5: {}",
            output[2]
        );
        assert!(
            output[4] > 0.99,
            "sigmoid(100) should be near 1: {}",
            output[4]
        );
    }

    #[test]
    fn tanh_dispatch_boundary_values() {
        let input = vec![-100.0, -1.0, 0.0, 1.0, 100.0];
        let mut output = vec![0.0f32; 5];
        tanh_slice_dispatch(&input, &mut output);

        assert!(
            output[0] < -0.99,
            "tanh(-100) should be near -1: {}",
            output[0]
        );
        assert!(
            (output[2]).abs() < 0.01,
            "tanh(0) should be near 0: {}",
            output[2]
        );
        assert!(
            output[4] > 0.99,
            "tanh(100) should be near 1: {}",
            output[4]
        );
    }
}
