//! SIMD-accelerated primitives for tensor operations.
//!
//! Provides runtime-dispatched SIMD implementations for reductions, binary ops,
//! and in-place operations. Falls back to scalar on unsupported platforms and under miri.

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
    fn armpl_svdiv_f32(n: i32, a: *const f32, b: *const f32, y: *mut f32);
    fn armpl_svlog_f32(n: i32, x: *const f32, y: *mut f32);
    fn armpl_svsqrt_f32(n: i32, x: *const f32, y: *mut f32);
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
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
    fn vDSP_vdiv(
        __B: *const f32,
        __IB: i32,
        __A: *const f32,
        __IA: i32,
        __C: *mut f32,
        __IC: i32,
        __N: u32,
    );
    fn vDSP_vneg(__A: *const f32, __IA: i32, __C: *mut f32, __IC: i32, __N: u32);
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vabsq_f32, vaddq_f32, vbslq_f32, vcgtq_f32, vcltq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32,
    vminq_f32, vmulq_f32, vnegq_f32, vrecpeq_f32, vrecpsq_f32, vrndaq_f32, vrndmq_f32, vrndpq_f32,
    vsqrtq_f32, vst1q_f32, vsubq_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128, __m256, _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_castsi128_ps, _mm_cmpgt_ps,
    _mm_cmplt_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps, _mm_max_ps, _mm_min_ps,
    _mm_mul_ps, _mm_or_ps, _mm_rcp_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_sqrt_ps,
    _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps, _mm256_castsi256_ps,
    _mm256_ceil_ps, _mm256_cmp_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32, _mm256_floor_ps,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_rcp_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sqrt_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, __m256, _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_castsi128_ps, _mm_cmpgt_ps,
    _mm_cmplt_ps, _mm_cvtepi32_ps, _mm_cvtps_epi32, _mm_loadu_ps, _mm_max_ps, _mm_min_ps,
    _mm_mul_ps, _mm_or_ps, _mm_rcp_ps, _mm_set1_epi32, _mm_set1_ps, _mm_setzero_ps, _mm_sqrt_ps,
    _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps, _mm256_and_ps, _mm256_andnot_ps, _mm256_castsi256_ps,
    _mm256_ceil_ps, _mm256_cmp_ps, _mm256_cvtepi32_ps, _mm256_cvtps_epi32, _mm256_floor_ps,
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_rcp_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sqrt_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};

// ===========================================================================
// Sum reduction
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn sum_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    if cfg!(miri) {
        return sum_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { sum_avx(data) };
        }
        if std::is_x86_feature_detected!("sse") {
            return unsafe { sum_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { sum_neon(data) };
        }
    }

    sum_scalar(data)
}

fn sum_scalar(data: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &v in data {
        acc += v;
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sum_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm_setzero_ps();

    while i + 4 <= len {
        acc = _mm_add_ps(acc, _mm_loadu_ps(ptr.add(i)));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3];

    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sum_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();

    while i + 8 <= len {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sum_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vaddvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(0.0);

    while i + 4 <= len {
        acc = vaddq_f32(acc, vld1q_f32(ptr.add(i)));
        i += 4;
    }

    let mut result = vaddvq_f32(acc);
    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

// ===========================================================================
// Max reduction
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn max_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    if cfg!(miri) {
        return max_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { max_avx(data) };
        }
        if std::is_x86_feature_detected!("sse") {
            return unsafe { max_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { max_neon(data) };
        }
    }

    max_scalar(data)
}

fn max_scalar(data: &[f32]) -> f32 {
    let mut acc = f32::NEG_INFINITY;
    for &v in data {
        acc = acc.max(v);
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn max_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm_set1_ps(f32::NEG_INFINITY);

    while i + 4 <= len {
        acc = _mm_max_ps(acc, _mm_loadu_ps(ptr.add(i)));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0].max(buf[1]).max(buf[2]).max(buf[3]);

    while i < len {
        result = result.max(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn max_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);

    while i + 8 <= len {
        acc = _mm256_max_ps(acc, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0];
    for j in 1..8 {
        result = result.max(buf[j]);
    }

    while i < len {
        result = result.max(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn max_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vmaxvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);

    while i + 4 <= len {
        acc = vmaxq_f32(acc, vld1q_f32(ptr.add(i)));
        i += 4;
    }

    let mut result = vmaxvq_f32(acc);
    while i < len {
        result = result.max(*ptr.add(i));
        i += 1;
    }
    result
}

// ===========================================================================
// Min reduction
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn min_dispatch(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::INFINITY;
    }
    if cfg!(miri) {
        return min_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { min_avx(data) };
        }
        if std::is_x86_feature_detected!("sse") {
            return unsafe { min_sse(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { min_neon(data) };
        }
    }

    min_scalar(data)
}

fn min_scalar(data: &[f32]) -> f32 {
    let mut acc = f32::INFINITY;
    for &v in data {
        acc = acc.min(v);
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn min_sse(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm_set1_ps(f32::INFINITY);

    while i + 4 <= len {
        acc = _mm_min_ps(acc, _mm_loadu_ps(ptr.add(i)));
        i += 4;
    }

    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0].min(buf[1]).min(buf[2]).min(buf[3]);

    while i < len {
        result = result.min(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn min_avx(data: &[f32]) -> f32 {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm256_set1_ps(f32::INFINITY);

    while i + 8 <= len {
        acc = _mm256_min_ps(acc, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut result = buf[0];
    for j in 1..8 {
        result = result.min(buf[j]);
    }

    while i < len {
        result = result.min(*ptr.add(i));
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn min_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::vminvq_f32;

    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(f32::INFINITY);

    while i + 4 <= len {
        acc = vminq_f32(acc, vld1q_f32(ptr.add(i)));
        i += 4;
    }

    let mut result = vminvq_f32(acc);
    while i < len {
        result = result.min(*ptr.add(i));
        i += 1;
    }
    result
}

// ===========================================================================
// Binary same-shape operations
// ===========================================================================

#[derive(Clone, Copy)]
pub(crate) enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

#[allow(unsafe_code, unreachable_code)]
#[inline]
pub(crate) fn binary_dispatch(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());

    if cfg!(miri) {
        binary_scalar(lhs, rhs, out, kind);
        return;
    }

    // macOS aarch64: use Apple vDSP for add/sub/mul/div.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let n = lhs.len() as u32;
        // SAFETY: vDSP functions operate on contiguous slices of equal length.
        unsafe {
            match kind {
                BinaryKind::Add => {
                    vDSP_vadd(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Sub => {
                    vDSP_vsub(rhs.as_ptr(), 1, lhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Mul => {
                    vDSP_vmul(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Div => {
                    vDSP_vdiv(rhs.as_ptr(), 1, lhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
            }
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML for add/sub/mul/div (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: VML functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => vsAdd(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => vsSub(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => vsMul(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Div => vsDiv(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries for add/sub/mul/div.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: ARMPL functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => armpl_svadd_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => armpl_svsub_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => armpl_svmul_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Div => armpl_svdiv_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { binary_avx(lhs, rhs, out, kind) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { binary_sse(lhs, rhs, out, kind) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { binary_neon(lhs, rhs, out, kind) };
            return;
        }
    }

    binary_scalar(lhs, rhs, out, kind);
}

fn binary_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    match kind {
        BinaryKind::Add => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] + rhs[i];
            }
        }
        BinaryKind::Sub => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] - rhs[i];
            }
        }
        BinaryKind::Mul => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] * rhs[i];
            }
        }
        BinaryKind::Div => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] / rhs[i];
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn binary_sse(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let l = _mm_loadu_ps(lp.add(i));
        let r = _mm_loadu_ps(rp.add(i));
        let result = match kind {
            BinaryKind::Add => _mm_add_ps(l, r),
            BinaryKind::Sub => _mm_sub_ps(l, r),
            BinaryKind::Mul => _mm_mul_ps(l, r),
            BinaryKind::Div => {
                #[cfg(target_arch = "x86")]
                use std::arch::x86::_mm_div_ps;
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::_mm_div_ps;
                _mm_div_ps(l, r)
            }
        };
        _mm_storeu_ps(op.add(i), result);
        i += 4;
    }

    while i < len {
        *op.add(i) = match kind {
            BinaryKind::Add => *lp.add(i) + *rp.add(i),
            BinaryKind::Sub => *lp.add(i) - *rp.add(i),
            BinaryKind::Mul => *lp.add(i) * *rp.add(i),
            BinaryKind::Div => *lp.add(i) / *rp.add(i),
        };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn binary_avx(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;

    let len = lhs.len();
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // 8x unrolled: process 64 floats per iteration for better ILP.
    // Loads are interleaved with compute/stores to keep the OoO pipeline busy.
    match kind {
        BinaryKind::Add => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_add_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_add_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_add_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_add_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_add_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_add_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_add_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_add_ps(a7, b7));
                i += 64;
            }
        }
        BinaryKind::Sub => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_sub_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_sub_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_sub_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_sub_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_sub_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_sub_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_sub_ps(a7, b7));
                i += 64;
            }
        }
        BinaryKind::Mul => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_mul_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_mul_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_mul_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_mul_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_mul_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_mul_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_mul_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_mul_ps(a7, b7));
                i += 64;
            }
        }
        BinaryKind::Div => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_div_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_div_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_div_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_div_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_div_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_div_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_div_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_div_ps(a7, b7));
                i += 64;
            }
        }
    }

    // Handle remaining 32-element chunks (4x unrolled fallback)
    match kind {
        BinaryKind::Add => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_add_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_add_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_add_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_add_ps(a3, b3));
                i += 32;
            }
        }
        BinaryKind::Sub => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_sub_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_sub_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_sub_ps(a3, b3));
                i += 32;
            }
        }
        BinaryKind::Mul => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_mul_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_mul_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_mul_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_mul_ps(a3, b3));
                i += 32;
            }
        }
        BinaryKind::Div => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_div_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_div_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_div_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_div_ps(a3, b3));
                i += 32;
            }
        }
    }

    // Handle remaining 8-element chunks
    while i + 8 <= len {
        let l = _mm256_loadu_ps(lp.add(i));
        let r = _mm256_loadu_ps(rp.add(i));
        let result = match kind {
            BinaryKind::Add => _mm256_add_ps(l, r),
            BinaryKind::Sub => _mm256_sub_ps(l, r),
            BinaryKind::Mul => _mm256_mul_ps(l, r),
            BinaryKind::Div => _mm256_div_ps(l, r),
        };
        _mm256_storeu_ps(op.add(i), result);
        i += 8;
    }

    if i < len {
        binary_sse(&lhs[i..], &rhs[i..], &mut out[i..], kind);
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn binary_neon(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    use std::arch::aarch64::vdivq_f32;

    let len = lhs.len();
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // 8x unrolled: process 32 floats per iteration for better ILP.
    // Loads are interleaved with compute/stores to keep the OoO pipeline busy.
    match kind {
        BinaryKind::Add => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vaddq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vaddq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vaddq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vaddq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vaddq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vaddq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vaddq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vaddq_f32(a7, b7));
                i += 32;
            }
        }
        BinaryKind::Sub => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vsubq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vsubq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vsubq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vsubq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vsubq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vsubq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vsubq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vsubq_f32(a7, b7));
                i += 32;
            }
        }
        BinaryKind::Mul => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vmulq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vmulq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vmulq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vmulq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vmulq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vmulq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vmulq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vmulq_f32(a7, b7));
                i += 32;
            }
        }
        BinaryKind::Div => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vdivq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vdivq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vdivq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vdivq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vdivq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vdivq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vdivq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vdivq_f32(a7, b7));
                i += 32;
            }
        }
    }

    // Handle remaining 16-element chunks (4x unrolled fallback)
    match kind {
        BinaryKind::Add => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vaddq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vaddq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vaddq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vaddq_f32(a3, b3));
                i += 16;
            }
        }
        BinaryKind::Sub => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vsubq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vsubq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vsubq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vsubq_f32(a3, b3));
                i += 16;
            }
        }
        BinaryKind::Mul => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vmulq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vmulq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vmulq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vmulq_f32(a3, b3));
                i += 16;
            }
        }
        BinaryKind::Div => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vdivq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vdivq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vdivq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vdivq_f32(a3, b3));
                i += 16;
            }
        }
    }

    // Handle remaining 4-element chunks
    while i + 4 <= len {
        let l = vld1q_f32(lp.add(i));
        let r = vld1q_f32(rp.add(i));
        let result = match kind {
            BinaryKind::Add => vaddq_f32(l, r),
            BinaryKind::Sub => vsubq_f32(l, r),
            BinaryKind::Mul => vmulq_f32(l, r),
            BinaryKind::Div => vdivq_f32(l, r),
        };
        vst1q_f32(op.add(i), result);
        i += 4;
    }

    // Scalar tail
    while i < len {
        *op.add(i) = match kind {
            BinaryKind::Add => *lp.add(i) + *rp.add(i),
            BinaryKind::Sub => *lp.add(i) - *rp.add(i),
            BinaryKind::Mul => *lp.add(i) * *rp.add(i),
            BinaryKind::Div => *lp.add(i) / *rp.add(i),
        };
        i += 1;
    }
}

// ===========================================================================
// In-place add: dst[i] += src[i]
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn add_inplace_dispatch(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());

    if cfg!(miri) {
        for i in 0..dst.len() {
            dst[i] += src[i];
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { add_inplace_avx(dst, src) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { add_inplace_sse(dst, src) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { add_inplace_neon(dst, src) };
            return;
        }
    }

    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn add_inplace_sse(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = _mm_loadu_ps(dp.add(i));
        let s = _mm_loadu_ps(sp.add(i));
        _mm_storeu_ps(dp.add(i), _mm_add_ps(d, s));
        i += 4;
    }

    while i < len {
        *dp.add(i) += *sp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_inplace_avx(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let d = _mm256_loadu_ps(dp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d, s));
        i += 8;
    }

    if i < len {
        add_inplace_sse(&mut dst[i..], &src[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_inplace_neon(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = vld1q_f32(dp.add(i));
        let s = vld1q_f32(sp.add(i));
        vst1q_f32(dp.add(i), vaddq_f32(d, s));
        i += 4;
    }

    while i < len {
        *dp.add(i) += *sp.add(i);
        i += 1;
    }
}

// ===========================================================================
// In-place max: dst[i] = max(dst[i], src[i])
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn max_inplace_dispatch(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());

    if cfg!(miri) {
        for i in 0..dst.len() {
            dst[i] = dst[i].max(src[i]);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { max_inplace_avx(dst, src) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { max_inplace_sse(dst, src) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { max_inplace_neon(dst, src) };
            return;
        }
    }

    for i in 0..dst.len() {
        dst[i] = dst[i].max(src[i]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn max_inplace_sse(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = _mm_loadu_ps(dp.add(i));
        let s = _mm_loadu_ps(sp.add(i));
        _mm_storeu_ps(dp.add(i), _mm_max_ps(d, s));
        i += 4;
    }

    while i < len {
        let d = *dp.add(i);
        let s = *sp.add(i);
        *dp.add(i) = if d > s { d } else { s };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn max_inplace_avx(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let d = _mm256_loadu_ps(dp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_max_ps(d, s));
        i += 8;
    }

    if i < len {
        max_inplace_sse(&mut dst[i..], &src[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn max_inplace_neon(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let d = vld1q_f32(dp.add(i));
        let s = vld1q_f32(sp.add(i));
        vst1q_f32(dp.add(i), vmaxq_f32(d, s));
        i += 4;
    }

    while i < len {
        let d = *dp.add(i);
        let s = *sp.add(i);
        *dp.add(i) = if d > s { d } else { s };
        i += 1;
    }
}

// ===========================================================================
// In-place ReLU: v[i] = max(v[i], 0)
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn relu_inplace_dispatch(values: &mut [f32]) {
    if cfg!(miri) {
        for v in values.iter_mut() {
            *v = v.max(0.0);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { relu_inplace_avx(values) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { relu_inplace_sse(values) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { relu_inplace_neon(values) };
            return;
        }
    }

    for v in values.iter_mut() {
        *v = v.max(0.0);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn relu_inplace_sse(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm_setzero_ps();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_max_ps(v, zero));
        i += 4;
    }

    while i < len {
        let v = *ptr.add(i);
        *ptr.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn relu_inplace_avx(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_max_ps(v, zero));
        i += 8;
    }

    if i < len {
        relu_inplace_sse(&mut values[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn relu_inplace_neon(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vmaxq_f32(v, zero));
        i += 4;
    }

    while i < len {
        let v = *ptr.add(i);
        *ptr.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

// ===========================================================================
// In-place scalar ops: v[i] += s, v[i] *= s
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub(crate) fn add_scalar_inplace_dispatch(values: &mut [f32], s: f32) {
    if cfg!(miri) {
        for v in values.iter_mut() {
            *v += s;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { add_scalar_inplace_avx(values, s) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { add_scalar_inplace_sse(values, s) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { add_scalar_inplace_neon(values, s) };
            return;
        }
    }

    for v in values.iter_mut() {
        *v += s;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn add_scalar_inplace_sse(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm_set1_ps(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_add_ps(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) += s;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn add_scalar_inplace_avx(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm256_set1_ps(s);
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_add_ps(v, sv));
        i += 8;
    }
    if i < len {
        add_scalar_inplace_sse(&mut values[i..], s);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn add_scalar_inplace_neon(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = vdupq_n_f32(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vaddq_f32(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) += s;
        i += 1;
    }
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn mul_scalar_inplace_dispatch(values: &mut [f32], s: f32) {
    if cfg!(miri) {
        for v in values.iter_mut() {
            *v *= s;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { mul_scalar_inplace_avx(values, s) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { mul_scalar_inplace_sse(values, s) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { mul_scalar_inplace_neon(values, s) };
            return;
        }
    }

    for v in values.iter_mut() {
        *v *= s;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn mul_scalar_inplace_sse(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm_set1_ps(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(ptr.add(i));
        _mm_storeu_ps(ptr.add(i), _mm_mul_ps(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= s;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn mul_scalar_inplace_avx(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = _mm256_set1_ps(s);
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, sv));
        i += 8;
    }
    if i < len {
        mul_scalar_inplace_sse(&mut values[i..], s);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn mul_scalar_inplace_neon(values: &mut [f32], s: f32) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let sv = vdupq_n_f32(s);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vmulq_f32(v, sv));
        i += 4;
    }
    while i < len {
        *ptr.add(i) *= s;
        i += 1;
    }
}

// ===========================================================================
// Unary operations (neg, abs, sqrt, recip)
// ===========================================================================

#[derive(Clone, Copy, Debug)]
pub(crate) enum UnaryKind {
    Neg,
    Abs,
    Sqrt,
    Recip,
    Floor,
    Ceil,
    Round,
    Sign,
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn unary_dispatch(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        unary_scalar(data, out, kind);
        return;
    }

    // macOS aarch64: use vDSP_vneg (faster than NEON for negation).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if matches!(kind, UnaryKind::Neg) {
            let n = data.len() as u32;
            unsafe {
                vDSP_vneg(data.as_ptr(), 1, out.as_mut_ptr(), 1, n);
            }
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { unary_avx(data, out, kind) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { unary_sse(data, out, kind) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { unary_neon(data, out, kind) };
            return;
        }
    }

    unary_scalar(data, out, kind);
}

fn unary_scalar(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    match kind {
        UnaryKind::Neg => {
            for i in 0..data.len() {
                out[i] = -data[i];
            }
        }
        UnaryKind::Abs => {
            for i in 0..data.len() {
                out[i] = data[i].abs();
            }
        }
        UnaryKind::Sqrt => {
            for i in 0..data.len() {
                out[i] = data[i].sqrt();
            }
        }
        UnaryKind::Recip => {
            for i in 0..data.len() {
                out[i] = 1.0 / data[i];
            }
        }
        UnaryKind::Floor => {
            for i in 0..data.len() {
                out[i] = data[i].floor();
            }
        }
        UnaryKind::Ceil => {
            for i in 0..data.len() {
                out[i] = data[i].ceil();
            }
        }
        UnaryKind::Round => {
            for i in 0..data.len() {
                out[i] = data[i].round();
            }
        }
        UnaryKind::Sign => {
            for i in 0..data.len() {
                out[i] = if data[i] > 0.0 {
                    1.0
                } else if data[i] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn unary_sse(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    // Floor/Ceil/Round need SSE4.1; fall back to scalar on SSE-only CPUs.
    if matches!(kind, UnaryKind::Floor | UnaryKind::Ceil | UnaryKind::Round) {
        unary_scalar(data, out, kind);
        return;
    }

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Match OUTSIDE loop for better branch prediction + constant hoisting.
    match kind {
        UnaryKind::Neg => {
            let zero = _mm_setzero_ps();
            while i + 4 <= len {
                _mm_storeu_ps(op.add(i), _mm_sub_ps(zero, _mm_loadu_ps(inp.add(i))));
                i += 4;
            }
        }
        UnaryKind::Abs => {
            let sign_mask = _mm_set1_ps(-0.0);
            while i + 4 <= len {
                _mm_storeu_ps(
                    op.add(i),
                    _mm_andnot_ps(sign_mask, _mm_loadu_ps(inp.add(i))),
                );
                i += 4;
            }
        }
        UnaryKind::Sqrt => {
            while i + 4 <= len {
                _mm_storeu_ps(op.add(i), _mm_sqrt_ps(_mm_loadu_ps(inp.add(i))));
                i += 4;
            }
        }
        UnaryKind::Recip => {
            let two = _mm_set1_ps(2.0);
            while i + 4 <= len {
                let v = _mm_loadu_ps(inp.add(i));
                let r = _mm_rcp_ps(v);
                _mm_storeu_ps(op.add(i), _mm_mul_ps(r, _mm_sub_ps(two, _mm_mul_ps(v, r))));
                i += 4;
            }
        }
        UnaryKind::Sign => {
            let zero = _mm_setzero_ps();
            let one = _mm_set1_ps(1.0);
            let neg_one = _mm_set1_ps(-1.0);
            while i + 4 <= len {
                let v = _mm_loadu_ps(inp.add(i));
                let pos_mask = _mm_cmpgt_ps(v, zero);
                let neg_mask = _mm_cmplt_ps(v, zero);
                _mm_storeu_ps(
                    op.add(i),
                    _mm_or_ps(_mm_and_ps(pos_mask, one), _mm_and_ps(neg_mask, neg_one)),
                );
                i += 4;
            }
        }
        UnaryKind::Floor | UnaryKind::Ceil | UnaryKind::Round => unreachable!(),
    }

    // Scalar tail
    while i < len {
        *op.add(i) = match kind {
            UnaryKind::Neg => -*inp.add(i),
            UnaryKind::Abs => (*inp.add(i)).abs(),
            UnaryKind::Sqrt => (*inp.add(i)).sqrt(),
            UnaryKind::Recip => 1.0 / *inp.add(i),
            UnaryKind::Sign => {
                let v = *inp.add(i);
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            UnaryKind::Floor | UnaryKind::Ceil | UnaryKind::Round => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn unary_avx(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Match OUTSIDE loop: eliminates branch per iteration, hoists constants.
    match kind {
        UnaryKind::Neg => {
            let zero = _mm256_setzero_ps();
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Abs => {
            let sign_mask = _mm256_set1_ps(-0.0);
            while i + 32 <= len {
                _mm256_storeu_ps(
                    op.add(i),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i))),
                );
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Sqrt => {
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Recip => {
            let two = _mm256_set1_ps(2.0);
            while i + 32 <= len {
                for off in [0, 8, 16, 24] {
                    let v = _mm256_loadu_ps(inp.add(i + off));
                    let r = _mm256_rcp_ps(v);
                    _mm256_storeu_ps(
                        op.add(i + off),
                        _mm256_mul_ps(r, _mm256_sub_ps(two, _mm256_mul_ps(v, r))),
                    );
                }
                i += 32;
            }
        }
        UnaryKind::Floor => {
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_floor_ps(_mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_floor_ps(_mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_floor_ps(_mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_floor_ps(_mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Ceil => {
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Round => {
            let neg_zero = _mm256_set1_ps(-0.0);
            let half = _mm256_set1_ps(0.5);
            while i + 32 <= len {
                for off in [0, 8, 16, 24] {
                    let v = _mm256_loadu_ps(inp.add(i + off));
                    let sign = _mm256_and_ps(v, neg_zero);
                    let abs_v = _mm256_andnot_ps(neg_zero, v);
                    _mm256_storeu_ps(
                        op.add(i + off),
                        _mm256_or_ps(_mm256_floor_ps(_mm256_add_ps(abs_v, half)), sign),
                    );
                }
                i += 32;
            }
        }
        UnaryKind::Sign => {
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);
            while i + 32 <= len {
                for off in [0, 8, 16, 24] {
                    let v = _mm256_loadu_ps(inp.add(i + off));
                    let pos_mask = _mm256_cmp_ps::<14>(v, zero);
                    let neg_mask = _mm256_cmp_ps::<1>(v, zero);
                    _mm256_storeu_ps(
                        op.add(i + off),
                        _mm256_or_ps(
                            _mm256_and_ps(pos_mask, one),
                            _mm256_and_ps(neg_mask, neg_one),
                        ),
                    );
                }
                i += 32;
            }
        }
    }

    // Tail: process remaining elements via SSE
    if i < len {
        unary_sse(&data[i..], &mut out[i..], kind);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn unary_neon(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Match OUTSIDE loop: eliminates branch per iteration, enables unrolling,
    // and hoists constants. Same pattern as binary_neon.
    match kind {
        UnaryKind::Neg => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vnegq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vnegq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vnegq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vnegq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Abs => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vabsq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vabsq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vabsq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vabsq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Sqrt => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vsqrtq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vsqrtq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vsqrtq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vsqrtq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Recip => {
            while i + 16 <= len {
                let v0 = vld1q_f32(inp.add(i));
                let v1 = vld1q_f32(inp.add(i + 4));
                let v2 = vld1q_f32(inp.add(i + 8));
                let v3 = vld1q_f32(inp.add(i + 12));
                let r0 = vrecpeq_f32(v0);
                let s0 = vrecpsq_f32(v0, r0);
                let r1 = vrecpeq_f32(v1);
                let s1 = vrecpsq_f32(v1, r1);
                let r2 = vrecpeq_f32(v2);
                let s2 = vrecpsq_f32(v2, r2);
                let r3 = vrecpeq_f32(v3);
                let s3 = vrecpsq_f32(v3, r3);
                vst1q_f32(op.add(i), vmulq_f32(r0, s0));
                vst1q_f32(op.add(i + 4), vmulq_f32(r1, s1));
                vst1q_f32(op.add(i + 8), vmulq_f32(r2, s2));
                vst1q_f32(op.add(i + 12), vmulq_f32(r3, s3));
                i += 16;
            }
        }
        UnaryKind::Floor => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vrndmq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vrndmq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vrndmq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vrndmq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Ceil => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vrndpq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vrndpq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vrndpq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vrndpq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Round => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vrndaq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vrndaq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vrndaq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vrndaq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Sign => {
            let zero = vdupq_n_f32(0.0);
            let one = vdupq_n_f32(1.0);
            let neg_one = vdupq_n_f32(-1.0);
            while i + 16 <= len {
                let v0 = vld1q_f32(inp.add(i));
                let v1 = vld1q_f32(inp.add(i + 4));
                let v2 = vld1q_f32(inp.add(i + 8));
                let v3 = vld1q_f32(inp.add(i + 12));
                vst1q_f32(
                    op.add(i),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v0, zero), one, zero),
                        vbslq_f32(vcltq_f32(v0, zero), neg_one, zero),
                    ),
                );
                vst1q_f32(
                    op.add(i + 4),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v1, zero), one, zero),
                        vbslq_f32(vcltq_f32(v1, zero), neg_one, zero),
                    ),
                );
                vst1q_f32(
                    op.add(i + 8),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v2, zero), one, zero),
                        vbslq_f32(vcltq_f32(v2, zero), neg_one, zero),
                    ),
                );
                vst1q_f32(
                    op.add(i + 12),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v3, zero), one, zero),
                        vbslq_f32(vcltq_f32(v3, zero), neg_one, zero),
                    ),
                );
                i += 16;
            }
        }
    }

    // Scalar tail for remaining < 16 elements
    while i < len {
        *op.add(i) = match kind {
            UnaryKind::Neg => -*inp.add(i),
            UnaryKind::Abs => (*inp.add(i)).abs(),
            UnaryKind::Sqrt => (*inp.add(i)).sqrt(),
            UnaryKind::Recip => 1.0 / *inp.add(i),
            UnaryKind::Floor => (*inp.add(i)).floor(),
            UnaryKind::Ceil => (*inp.add(i)).ceil(),
            UnaryKind::Round => (*inp.add(i)).round(),
            UnaryKind::Sign => {
                let v = *inp.add(i);
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        };
        i += 1;
    }
}

// ===========================================================================
// Exp: out[i] = exp(data[i]) — SIMD polynomial approximation
// ===========================================================================

/// Compute exp(data) into `out` using SIMD polynomial approximation where
/// available, falling back to scalar `f32::exp` otherwise.
#[allow(unsafe_code, unreachable_code)]
#[inline]
pub(crate) fn exp_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        exp_scalar(data, out);
        return;
    }

    // macOS aarch64: use Apple Accelerate vvexpf.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let count = data.len() as i32;
        unsafe {
            vvexpf(out.as_mut_ptr(), data.as_ptr(), &count);
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML vsExp (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let count = data.len() as i32;
        // SAFETY: vsExp reads `count` floats from `data` and writes to `out`.
        unsafe { vsExp(count, data.as_ptr(), out.as_mut_ptr()) };
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries vectorized exp.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let count = data.len() as i32;
        // SAFETY: armpl_svexp_f32 reads `count` floats from `data` and writes to `out`.
        unsafe { armpl_svexp_f32(count, data.as_ptr(), out.as_mut_ptr()) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { exp_avx(data, out) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { exp_sse(data, out) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { exp_neon(data, out) };
            return;
        }
    }

    exp_scalar(data, out);
}

/// Kept for backward compatibility — calls `exp_dispatch` with in-place semantics.
#[allow(unsafe_code, dead_code)]
#[inline]
pub(crate) fn exp_inplace_dispatch(data: &mut [f32]) {
    if cfg!(miri) {
        for v in data.iter_mut() {
            *v = v.exp();
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: input and output alias, but each element is read
            // once then written, so there's no ordering hazard.
            unsafe {
                let ptr = data.as_ptr();
                let len = data.len();
                let slice = std::slice::from_raw_parts(ptr, len);
                exp_avx(slice, data);
            };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe {
                let ptr = data.as_ptr();
                let len = data.len();
                let slice = std::slice::from_raw_parts(ptr, len);
                exp_sse(slice, data);
            };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let ptr = data.as_ptr();
                let len = data.len();
                let slice = std::slice::from_raw_parts(ptr, len);
                exp_neon(slice, data);
            };
            return;
        }
    }

    for v in data.iter_mut() {
        *v = v.exp();
    }
}

fn exp_scalar(data: &[f32], out: &mut [f32]) {
    for i in 0..data.len() {
        out[i] = data[i].exp();
    }
}

// ── SSE fast-exp (4-wide) ──────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn fast_exp_sse(x: __m128) -> __m128 {
    let ln2_inv = _mm_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm_set1_ps(0.693_359_4);
    let ln2_lo = _mm_set1_ps(-2.121_944_4e-4);

    let c0 = _mm_set1_ps(1.0);
    let c1 = _mm_set1_ps(1.0);
    let c2 = _mm_set1_ps(0.5);
    let c3 = _mm_set1_ps(1.0 / 6.0);
    let c4 = _mm_set1_ps(1.0 / 24.0);
    let c5 = _mm_set1_ps(1.0 / 120.0);
    let c6 = _mm_set1_ps(1.0 / 720.0);

    let x = _mm_max_ps(_mm_set1_ps(-88.0), _mm_min_ps(_mm_set1_ps(88.0), x));

    let n_f = _mm_mul_ps(x, ln2_inv);
    let n_i = _mm_cvtps_epi32(n_f);
    let n_f = _mm_cvtepi32_ps(n_i);

    let r = _mm_sub_ps(
        _mm_sub_ps(x, _mm_mul_ps(n_f, ln2_hi)),
        _mm_mul_ps(n_f, ln2_lo),
    );

    let mut poly = _mm_add_ps(c5, _mm_mul_ps(r, c6));
    poly = _mm_add_ps(c4, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c3, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c2, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c1, _mm_mul_ps(r, poly));
    poly = _mm_add_ps(c0, _mm_mul_ps(r, poly));

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn exp_sse(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), fast_exp_sse(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).exp();
        i += 1;
    }
}

// ── AVX fast-exp (8-wide) ──────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn exp_avx(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), fast_exp_avx(v));
        i += 8;
    }

    if i < len {
        exp_sse(&data[i..], &mut out[i..]);
    }
}

// ── NEON fast-exp (4-wide) ─────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn fast_exp_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vfmaq_f32, vreinterpretq_f32_s32,
        vshlq_n_s32,
    };

    let ln2_inv = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693_359_4);
    let ln2_lo = vdupq_n_f32(-2.121_944_4e-4);

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

    // r = x - n * ln2  (Cody-Waite two-step)
    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n_f, ln2_hi)), vmulq_f32(n_f, ln2_lo));

    // Horner: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    let mut poly = vfmaq_f32(c5, r, c6);
    poly = vfmaq_f32(c4, r, poly);
    poly = vfmaq_f32(c3, r, poly);
    poly = vfmaq_f32(c2, r, poly);
    poly = vfmaq_f32(c1, r, poly);
    poly = vfmaq_f32(c1, r, poly); // c0 == c1 == 1.0

    let bias = vdupq_n_s32(127);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_i, bias)));

    vmulq_f32(poly, pow2n)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn exp_neon(data: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::{
        vaddq_s32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vfmaq_f32, vreinterpretq_f32_s32,
        vshlq_n_s32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Constants hoisted out of the loop
    let ln2_inv = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693_359_4);
    let ln2_lo = vdupq_n_f32(-2.121_944_4e-4);
    let c1 = vdupq_n_f32(1.0);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);
    let c5 = vdupq_n_f32(1.0 / 120.0);
    let c6 = vdupq_n_f32(1.0 / 720.0);
    let lo_clamp = vdupq_n_f32(-88.0);
    let hi_clamp = vdupq_n_f32(88.0);
    let bias = vdupq_n_s32(127);

    // 4x interleaved: all polynomial steps run across 4 vectors
    // for maximum instruction-level parallelism on wide pipelines.
    while i + 16 <= len {
        // Load
        let mut x0 = vld1q_f32(inp.add(i));
        let mut x1 = vld1q_f32(inp.add(i + 4));
        let mut x2 = vld1q_f32(inp.add(i + 8));
        let mut x3 = vld1q_f32(inp.add(i + 12));

        // Clamp
        x0 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x0));
        x1 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x1));
        x2 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x2));
        x3 = vmaxq_f32(lo_clamp, vminq_f32(hi_clamp, x3));

        // n = round(x / ln2)
        let n0 = vcvtnq_s32_f32(vmulq_f32(x0, ln2_inv));
        let n1 = vcvtnq_s32_f32(vmulq_f32(x1, ln2_inv));
        let n2 = vcvtnq_s32_f32(vmulq_f32(x2, ln2_inv));
        let n3 = vcvtnq_s32_f32(vmulq_f32(x3, ln2_inv));
        let nf0 = vcvtq_f32_s32(n0);
        let nf1 = vcvtq_f32_s32(n1);
        let nf2 = vcvtq_f32_s32(n2);
        let nf3 = vcvtq_f32_s32(n3);

        // r = x - n * ln2 (Cody-Waite)
        let r0 = vsubq_f32(
            vsubq_f32(x0, vmulq_f32(nf0, ln2_hi)),
            vmulq_f32(nf0, ln2_lo),
        );
        let r1 = vsubq_f32(
            vsubq_f32(x1, vmulq_f32(nf1, ln2_hi)),
            vmulq_f32(nf1, ln2_lo),
        );
        let r2 = vsubq_f32(
            vsubq_f32(x2, vmulq_f32(nf2, ln2_hi)),
            vmulq_f32(nf2, ln2_lo),
        );
        let r3 = vsubq_f32(
            vsubq_f32(x3, vmulq_f32(nf3, ln2_hi)),
            vmulq_f32(nf3, ln2_lo),
        );

        // Horner polynomial: interleaved across all 4 vectors
        let mut p0 = vfmaq_f32(c5, r0, c6);
        let mut p1 = vfmaq_f32(c5, r1, c6);
        let mut p2 = vfmaq_f32(c5, r2, c6);
        let mut p3 = vfmaq_f32(c5, r3, c6);

        p0 = vfmaq_f32(c4, r0, p0);
        p1 = vfmaq_f32(c4, r1, p1);
        p2 = vfmaq_f32(c4, r2, p2);
        p3 = vfmaq_f32(c4, r3, p3);

        p0 = vfmaq_f32(c3, r0, p0);
        p1 = vfmaq_f32(c3, r1, p1);
        p2 = vfmaq_f32(c3, r2, p2);
        p3 = vfmaq_f32(c3, r3, p3);

        p0 = vfmaq_f32(c2, r0, p0);
        p1 = vfmaq_f32(c2, r1, p1);
        p2 = vfmaq_f32(c2, r2, p2);
        p3 = vfmaq_f32(c2, r3, p3);

        p0 = vfmaq_f32(c1, r0, p0);
        p1 = vfmaq_f32(c1, r1, p1);
        p2 = vfmaq_f32(c1, r2, p2);
        p3 = vfmaq_f32(c1, r3, p3);

        p0 = vfmaq_f32(c1, r0, p0); // c0 == c1 == 1.0
        p1 = vfmaq_f32(c1, r1, p1);
        p2 = vfmaq_f32(c1, r2, p2);
        p3 = vfmaq_f32(c1, r3, p3);

        // 2^n via integer bit manipulation
        let pow0 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n0, bias)));
        let pow1 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n1, bias)));
        let pow2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n2, bias)));
        let pow3 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n3, bias)));

        // Store
        vst1q_f32(op.add(i), vmulq_f32(p0, pow0));
        vst1q_f32(op.add(i + 4), vmulq_f32(p1, pow1));
        vst1q_f32(op.add(i + 8), vmulq_f32(p2, pow2));
        vst1q_f32(op.add(i + 12), vmulq_f32(p3, pow3));
        i += 16;
    }

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        vst1q_f32(op.add(i), fast_exp_neon(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).exp();
        i += 1;
    }
}

// ===========================================================================
// sin: out[i] = sin(data[i]) — SIMD polynomial approximation
// ===========================================================================

/// Compute sin(data) into `out` using SIMD polynomial approximation where
/// available, falling back to scalar `f32::sin` otherwise.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn sin_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        sin_scalar(data, out);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { sin_neon(data, out) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { sin_avx(data, out) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { sin_sse(data, out) };
            return;
        }
    }

    sin_scalar(data, out);
}

fn sin_scalar(data: &[f32], out: &mut [f32]) {
    for i in 0..data.len() {
        out[i] = data[i].sin();
    }
}

/// Compute cos(data) into `out` using SIMD polynomial approximation where
/// available, falling back to scalar `f32::cos` otherwise.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn cos_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        cos_scalar(data, out);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { cos_neon(data, out) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { cos_avx(data, out) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { cos_sse(data, out) };
            return;
        }
    }

    cos_scalar(data, out);
}

fn cos_scalar(data: &[f32], out: &mut [f32]) {
    for i in 0..data.len() {
        out[i] = data[i].cos();
    }
}

// ── NEON sin/cos (4-wide) ──────────────────────────────────────────
//
// Uses Cephes-style range reduction and minimax polynomial:
//   1. Range-reduce x to [-pi, pi] via x = x - round(x / (2*pi)) * 2*pi
//   2. Further reduce to [-pi/2, pi/2] using reflection
//   3. Evaluate minimax polynomial sin(x) ≈ x * (1 + x^2 * (c1 + x^2 * (c2 + x^2 * c3)))

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn fast_sin_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vandq_s32, vbslq_f32, vcgtq_f32, vcvtnq_s32_f32, vcvtq_f32_s32, vdupq_n_s32, vfmaq_f32,
        vorrq_s32, vreinterpretq_f32_s32, vreinterpretq_s32_f32,
    };

    // Constants
    let two_pi = vdupq_n_f32(std::f32::consts::TAU); // 2*pi
    let inv_two_pi = vdupq_n_f32(1.0 / std::f32::consts::TAU); // 1/(2*pi)
    let pi = vdupq_n_f32(std::f32::consts::PI);
    let half_pi = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

    // Minimax polynomial coefficients for sin(x) on [-pi/2, pi/2]
    // sin(x) ≈ x * (1 + x^2 * (c1 + x^2 * (c2 + x^2 * c3)))
    let c1 = vdupq_n_f32(-1.666_666_6e-1); // -1/6
    let c2 = vdupq_n_f32(8.333_331e-3); // 1/120
    let c3 = vdupq_n_f32(-1.980_741e-4); // -1/5040
    let c4 = vdupq_n_f32(2.601_903e-6); // ~1/362880

    // 1. Range reduce to [-pi, pi]
    let n = vcvtnq_s32_f32(vmulq_f32(x, inv_two_pi));
    let nf = vcvtq_f32_s32(n);
    let x_red = vsubq_f32(x, vmulq_f32(nf, two_pi));

    // 2. Reduce to [-pi/2, pi/2] using reflection:
    //    if x > pi/2:  x = pi - x
    //    if x < -pi/2: x = -pi - x
    let abs_mask_i = vdupq_n_s32(0x7FFF_FFFFu32 as i32);
    let sign_mask_i = vdupq_n_s32(0x8000_0000u32 as i32);
    let abs_x = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(x_red), abs_mask_i));
    let sign_x = vandq_s32(vreinterpretq_s32_f32(x_red), sign_mask_i);
    let signed_pi = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(pi), sign_x));

    // if abs_x > half_pi, reflect
    let needs_reflect = vcgtq_f32(abs_x, half_pi);
    // reflected = signed_pi - x_red
    let reflected = vsubq_f32(signed_pi, x_red);
    let x_final = vbslq_f32(needs_reflect, reflected, x_red);

    // 3. Evaluate polynomial: sin(x) = x * (1 + x^2*(c1 + x^2*(c2 + x^2*(c3 + x^2*c4))))
    let x2 = vmulq_f32(x_final, x_final);
    let mut poly = vfmaq_f32(c3, x2, c4);
    poly = vfmaq_f32(c2, x2, poly);
    poly = vfmaq_f32(c1, x2, poly);
    poly = vfmaq_f32(vdupq_n_f32(1.0), x2, poly);

    vmulq_f32(x_final, poly)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn sin_neon(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        vst1q_f32(op.add(i), fast_sin_neon(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).sin();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn cos_neon(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let half_pi = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        // cos(x) = sin(x + pi/2)
        vst1q_f32(op.add(i), fast_sin_neon(vaddq_f32(v, half_pi)));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).cos();
        i += 1;
    }
}

// ── SSE sin/cos (4-wide) ──────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn fast_sin_sse(x: __m128) -> __m128 {
    let two_pi = _mm_set1_ps(std::f32::consts::TAU);
    let inv_two_pi = _mm_set1_ps(1.0 / std::f32::consts::TAU);
    let pi = _mm_set1_ps(std::f32::consts::PI);
    let half_pi = _mm_set1_ps(std::f32::consts::FRAC_PI_2);

    let c1 = _mm_set1_ps(-1.666_666_6e-1);
    let c2 = _mm_set1_ps(8.333_331e-3);
    let c3 = _mm_set1_ps(-1.980_741e-4);
    let c4 = _mm_set1_ps(2.601_903e-6);

    // 1. Range reduce to [-pi, pi]
    let n = _mm_cvtps_epi32(_mm_mul_ps(x, inv_two_pi));
    let nf = _mm_cvtepi32_ps(n);
    let x_red = _mm_sub_ps(x, _mm_mul_ps(nf, two_pi));

    // 2. Reduce to [-pi/2, pi/2]
    let _abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFu32 as i32));
    let sign_mask = _mm_set1_ps(-0.0f32);
    let abs_x = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_andnot_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_andnot_ps;
        _mm_andnot_ps(sign_mask, x_red)
    };
    let sign_x = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_and_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_and_ps;
        _mm_and_ps(x_red, sign_mask)
    };
    // signed_pi = pi with sign of x
    let signed_pi = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_or_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_or_ps;
        _mm_or_ps(pi, sign_x)
    };
    // needs_reflect = abs_x > half_pi
    let needs_reflect = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_cmpgt_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_cmpgt_ps;
        _mm_cmpgt_ps(abs_x, half_pi)
    };
    let reflected = _mm_sub_ps(signed_pi, x_red);
    // x_final = needs_reflect ? reflected : x_red
    let x_final = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_and_ps, _mm_andnot_ps, _mm_or_ps};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_and_ps, _mm_andnot_ps, _mm_or_ps};
        _mm_or_ps(
            _mm_and_ps(needs_reflect, reflected),
            _mm_andnot_ps(needs_reflect, x_red),
        )
    };

    // 3. Polynomial
    let x2 = _mm_mul_ps(x_final, x_final);
    let mut poly = _mm_add_ps(c3, _mm_mul_ps(x2, c4));
    poly = _mm_add_ps(c2, _mm_mul_ps(x2, poly));
    poly = _mm_add_ps(c1, _mm_mul_ps(x2, poly));
    poly = _mm_add_ps(_mm_set1_ps(1.0), _mm_mul_ps(x2, poly));

    _mm_mul_ps(x_final, poly)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sin_sse(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), fast_sin_sse(v));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).sin();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn cos_sse(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let half_pi = _mm_set1_ps(std::f32::consts::FRAC_PI_2);

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), fast_sin_sse(_mm_add_ps(v, half_pi)));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).cos();
        i += 1;
    }
}

// ── AVX sin/cos (8-wide) ──────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn fast_sin_avx(x: __m256) -> __m256 {
    let two_pi = _mm256_set1_ps(std::f32::consts::TAU);
    let inv_two_pi = _mm256_set1_ps(1.0 / std::f32::consts::TAU);
    let pi = _mm256_set1_ps(std::f32::consts::PI);
    let half_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    let c1 = _mm256_set1_ps(-1.666_666_6e-1);
    let c2 = _mm256_set1_ps(8.333_331e-3);
    let c3 = _mm256_set1_ps(-1.980_741e-4);
    let c4 = _mm256_set1_ps(2.601_903e-6);

    // 1. Range reduce to [-pi, pi]
    let n = _mm256_cvtps_epi32(_mm256_mul_ps(x, inv_two_pi));
    let nf = _mm256_cvtepi32_ps(n);
    let x_red = _mm256_sub_ps(x, _mm256_mul_ps(nf, two_pi));

    // 2. Reduce to [-pi/2, pi/2]
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let abs_x = _mm256_andnot_ps(sign_mask, x_red);
    let sign_x = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm256_and_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm256_and_ps;
        _mm256_and_ps(x_red, sign_mask)
    };
    let signed_pi = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm256_or_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm256_or_ps;
        _mm256_or_ps(pi, sign_x)
    };
    let needs_reflect = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm256_cmp_ps;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm256_cmp_ps;
        _mm256_cmp_ps::<14>(abs_x, half_pi) // _CMP_GT_OS = 14
    };
    let reflected = _mm256_sub_ps(signed_pi, x_red);
    let x_final = {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm256_and_ps, _mm256_or_ps};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm256_and_ps, _mm256_or_ps};
        _mm256_or_ps(
            _mm256_and_ps(needs_reflect, reflected),
            _mm256_andnot_ps(needs_reflect, x_red),
        )
    };

    // 3. Polynomial
    let x2 = _mm256_mul_ps(x_final, x_final);
    let mut poly = _mm256_add_ps(c3, _mm256_mul_ps(x2, c4));
    poly = _mm256_add_ps(c2, _mm256_mul_ps(x2, poly));
    poly = _mm256_add_ps(c1, _mm256_mul_ps(x2, poly));
    poly = _mm256_add_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(x2, poly));

    _mm256_mul_ps(x_final, poly)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sin_avx(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), fast_sin_avx(v));
        i += 8;
    }

    if i < len {
        sin_sse(&data[i..], &mut out[i..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn cos_avx(data: &[f32], out: &mut [f32]) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let half_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), fast_sin_avx(_mm256_add_ps(v, half_pi)));
        i += 8;
    }

    if i < len {
        cos_sse(&data[i..], &mut out[i..]);
    }
}

// ===========================================================================
// ln: out[i] = ln(data[i])
// ===========================================================================

/// SIMD-accelerated natural logarithm using IEEE 754 bit decomposition
/// + 5th-order minimax polynomial. Max error ~1.5e-7 (23-bit mantissa limit).
#[allow(unsafe_code)]
#[inline]
pub(crate) fn ln_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        for i in 0..data.len() {
            out[i] = data[i].ln();
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { ln_neon(data, out) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { ln_avx(data, out) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { ln_sse(data, out) };
            return;
        }
    }

    for i in 0..data.len() {
        out[i] = data[i].ln();
    }
}

/// NEON polynomial ln using s=(m-1)/(m+1) substitution for fast convergence.
/// Maps mantissa [1,2) → s ∈ [0, 1/3). 5 terms give < 1e-7 error.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn ln_neon(data: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::{
        vandq_s32, vcvtq_f32_s32, vdivq_f32, vdupq_n_s32, vfmaq_f32, vorrq_s32,
        vreinterpretq_f32_s32, vreinterpretq_s32_f32, vshrq_n_s32, vsubq_s32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // ln(x) = e * ln(2) + ln(m), where x = m * 2^e, m in [1, 2)
    // s = (m-1)/(m+1) maps [1,2) → [0, 1/3)
    // ln(m) = 2*s*(1 + s²/3 + s⁴/5 + s⁶/7 + s⁸/9 + s¹⁰/11)
    let mantissa_mask = vdupq_n_s32(0x007F_FFFF);
    let one_bits = vdupq_n_s32(0x3F80_0000u32 as i32);
    let bias = vdupq_n_s32(127);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    // Coefficients: 1/3, 1/5, 1/7, 1/9, 1/11
    let c1 = vdupq_n_f32(1.0 / 3.0);
    let c2 = vdupq_n_f32(1.0 / 5.0);
    let c3 = vdupq_n_f32(1.0 / 7.0);
    let c4 = vdupq_n_f32(1.0 / 9.0);
    let c5 = vdupq_n_f32(1.0 / 11.0);

    while i + 4 <= len {
        let bits = vreinterpretq_s32_f32(vld1q_f32(inp.add(i)));
        let exp_i = vsubq_s32(vshrq_n_s32::<23>(bits), bias);
        let exp_f = vcvtq_f32_s32(exp_i);
        let m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(bits, mantissa_mask), one_bits));
        // s = (m - 1) / (m + 1)
        let s = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
        let s2 = vmulq_f32(s, s);
        // Horner: p = 1 + s²*(c1 + s²*(c2 + s²*(c3 + s²*(c4 + s²*c5))))
        let mut p = vfmaq_f32(c4, s2, c5);
        p = vfmaq_f32(c3, s2, p);
        p = vfmaq_f32(c2, s2, p);
        p = vfmaq_f32(c1, s2, p);
        p = vfmaq_f32(one, s2, p);
        // ln(m) = 2 * s * p
        let ln_m = vmulq_f32(two, vmulq_f32(s, p));
        // result = e * ln(2) + ln(m)
        let result = vfmaq_f32(ln_m, exp_f, ln2);
        vst1q_f32(op.add(i), result);
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).ln();
        i += 1;
    }
}

/// SSE polynomial ln using s=(m-1)/(m+1) substitution.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn ln_sse(data: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, _mm_and_si128, _mm_castps_si128, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_div_ps,
        _mm_or_si128, _mm_set1_epi32, _mm_srai_epi32, _mm_sub_epi32,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_castps_si128, _mm_castsi128_ps, _mm_cvtepi32_ps, _mm_div_ps,
        _mm_or_si128, _mm_set1_epi32, _mm_srai_epi32, _mm_sub_epi32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let mantissa_mask = _mm_set1_epi32(0x007F_FFFF);
    let one_bits = _mm_set1_epi32(0x3F80_0000u32 as i32);
    let bias = _mm_set1_epi32(127);
    let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
    let one_f = _mm_set1_ps(1.0);
    let two_f = _mm_set1_ps(2.0);
    let k1 = _mm_set1_ps(1.0 / 3.0);
    let k2 = _mm_set1_ps(1.0 / 5.0);
    let k3 = _mm_set1_ps(1.0 / 7.0);
    let k4 = _mm_set1_ps(1.0 / 9.0);
    let k5 = _mm_set1_ps(1.0 / 11.0);

    while i + 4 <= len {
        let bits: __m128i = _mm_castps_si128(_mm_loadu_ps(inp.add(i)));
        let exp_i = _mm_sub_epi32(_mm_srai_epi32::<23>(bits), bias);
        let exp_f = _mm_cvtepi32_ps(exp_i);
        let m = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(bits, mantissa_mask), one_bits));
        let s = _mm_div_ps(_mm_sub_ps(m, one_f), _mm_add_ps(m, one_f));
        let s2 = _mm_mul_ps(s, s);
        // Horner: p = 1 + s²*(1/3 + s²*(1/5 + s²*(1/7 + s²*(1/9 + s²/11))))
        let mut p = _mm_add_ps(k4, _mm_mul_ps(s2, k5));
        p = _mm_add_ps(k3, _mm_mul_ps(s2, p));
        p = _mm_add_ps(k2, _mm_mul_ps(s2, p));
        p = _mm_add_ps(k1, _mm_mul_ps(s2, p));
        p = _mm_add_ps(one_f, _mm_mul_ps(s2, p));
        let ln_m = _mm_mul_ps(two_f, _mm_mul_ps(s, p));
        let result = _mm_add_ps(ln_m, _mm_mul_ps(exp_f, ln2));
        _mm_storeu_ps(op.add(i), result);
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).ln();
        i += 1;
    }
}

/// AVX polynomial ln using s=(m-1)/(m+1) substitution (8 floats per iteration).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn ln_avx(data: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256i, _mm256_and_si256, _mm256_castps_si256, _mm256_cvtepi32_ps, _mm256_div_ps,
        _mm256_or_si256, _mm256_set1_epi32, _mm256_srai_epi32, _mm256_sub_epi32,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_castps_si256, _mm256_cvtepi32_ps, _mm256_div_ps,
        _mm256_or_si256, _mm256_set1_epi32, _mm256_srai_epi32, _mm256_sub_epi32,
    };

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    let mantissa_mask = _mm256_set1_epi32(0x007F_FFFF);
    let one_bits = _mm256_set1_epi32(0x3F80_0000u32 as i32);
    let bias = _mm256_set1_epi32(127);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let one_f = _mm256_set1_ps(1.0);
    let two_f = _mm256_set1_ps(2.0);
    let k1 = _mm256_set1_ps(1.0 / 3.0);
    let k2 = _mm256_set1_ps(1.0 / 5.0);
    let k3 = _mm256_set1_ps(1.0 / 7.0);
    let k4 = _mm256_set1_ps(1.0 / 9.0);
    let k5 = _mm256_set1_ps(1.0 / 11.0);

    while i + 8 <= len {
        let bits: __m256i = _mm256_castps_si256(_mm256_loadu_ps(inp.add(i)));
        let exp_i = _mm256_sub_epi32(_mm256_srai_epi32::<23>(bits), bias);
        let exp_f = _mm256_cvtepi32_ps(exp_i);
        let m = _mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(bits, mantissa_mask),
            one_bits,
        ));
        let s = _mm256_div_ps(_mm256_sub_ps(m, one_f), _mm256_add_ps(m, one_f));
        let s2 = _mm256_mul_ps(s, s);
        // Horner: p = 1 + s²*(1/3 + s²*(1/5 + s²*(1/7 + s²*(1/9 + s²/11))))
        let mut p = _mm256_add_ps(k4, _mm256_mul_ps(s2, k5));
        p = _mm256_add_ps(k3, _mm256_mul_ps(s2, p));
        p = _mm256_add_ps(k2, _mm256_mul_ps(s2, p));
        p = _mm256_add_ps(k1, _mm256_mul_ps(s2, p));
        p = _mm256_add_ps(one_f, _mm256_mul_ps(s2, p));
        let ln_m = _mm256_mul_ps(two_f, _mm256_mul_ps(s, p));
        let result = _mm256_add_ps(ln_m, _mm256_mul_ps(exp_f, ln2));
        _mm256_storeu_ps(op.add(i), result);
        i += 8;
    }

    if i < len {
        ln_sse(&data[i..], &mut out[i..]);
    }
}

// ===========================================================================
// clamp: out[i] = data[i].clamp(min_val, max_val)
// ===========================================================================

/// SIMD-accelerated clamp using min/max intrinsics.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn clamp_dispatch(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        for i in 0..data.len() {
            out[i] = data[i].clamp(min_val, max_val);
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { clamp_neon(data, out, min_val, max_val) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { clamp_avx(data, out, min_val, max_val) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { clamp_sse(data, out, min_val, max_val) };
            return;
        }
    }

    for i in 0..data.len() {
        out[i] = data[i].clamp(min_val, max_val);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn clamp_neon(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let vmin = vdupq_n_f32(min_val);
    let vmax = vdupq_n_f32(max_val);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        vst1q_f32(op.add(i), vminq_f32(vmaxq_f32(v, vmin), vmax));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).clamp(min_val, max_val);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn clamp_sse(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let vmin = _mm_set1_ps(min_val);
    let vmax = _mm_set1_ps(max_val);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), _mm_min_ps(_mm_max_ps(v, vmin), vmax));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).clamp(min_val, max_val);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn clamp_avx(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let vmin = _mm256_set1_ps(min_val);
    let vmax = _mm256_set1_ps(max_val);
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), _mm256_min_ps(_mm256_max_ps(v, vmin), vmax));
        i += 8;
    }

    if i < len {
        clamp_sse(&data[i..], &mut out[i..], min_val, max_val);
    }
}

// ===========================================================================
// tan: out[i] = sin(data[i]) / cos(data[i])
// ===========================================================================

/// SIMD-accelerated tangent using sin/cos dispatchers.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn tan_dispatch(data: &[f32], out: &mut [f32]) {
    debug_assert_eq!(data.len(), out.len());
    // Compute sin and cos in temp buffers, then divide.
    // Use stack allocation for small sizes to avoid heap allocation in hot paths.
    let len = data.len();
    if len <= 256 {
        let mut sin_buf = [0.0f32; 256];
        let mut cos_buf = [0.0f32; 256];
        sin_dispatch(data, &mut sin_buf[..len]);
        cos_dispatch(data, &mut cos_buf[..len]);
        for i in 0..len {
            out[i] = sin_buf[i] / cos_buf[i];
        }
    } else {
        let mut sin_buf = super::aligned::AlignedVec::<f32>::uninitialized(len);
        let mut cos_buf = super::aligned::AlignedVec::<f32>::uninitialized(len);
        sin_dispatch(data, &mut sin_buf);
        cos_dispatch(data, &mut cos_buf);
        for i in 0..len {
            out[i] = sin_buf[i] / cos_buf[i];
        }
    }
}

// ===========================================================================
// Comparison dispatch: gt, lt, eq → 1.0 / 0.0
// ===========================================================================

#[derive(Clone, Copy)]
pub(crate) enum CmpKind {
    Gt,
    Lt,
    Eq,
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn cmp_dispatch(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: CmpKind) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());
    if cfg!(miri) {
        cmp_scalar(lhs, rhs, out, kind);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { cmp_neon(lhs, rhs, out, lhs.len(), kind) };
            return;
        }
    }

    cmp_scalar(lhs, rhs, out, kind);
}

fn cmp_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: CmpKind) {
    for i in 0..lhs.len() {
        out[i] = match kind {
            CmpKind::Gt => {
                if lhs[i] > rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Lt => {
                if lhs[i] < rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Eq => {
                if (lhs[i] - rhs[i]).abs() < f32::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
        };
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn cmp_neon(lhs: &[f32], rhs: &[f32], out: &mut [f32], len: usize, kind: CmpKind) {
    use std::arch::aarch64::*;
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 16 <= len {
        for off in [0, 4, 8, 12] {
            let l = vld1q_f32(lp.add(i + off));
            let r = vld1q_f32(rp.add(i + off));
            let mask = match kind {
                CmpKind::Gt => vcgtq_f32(l, r),
                CmpKind::Lt => vcltq_f32(l, r),
                CmpKind::Eq => vceqq_f32(l, r),
            };
            vst1q_f32(op.add(i + off), vbslq_f32(mask, one, zero));
        }
        i += 16;
    }

    while i + 4 <= len {
        let l = vld1q_f32(lp.add(i));
        let r = vld1q_f32(rp.add(i));
        let mask = match kind {
            CmpKind::Gt => vcgtq_f32(l, r),
            CmpKind::Lt => vcltq_f32(l, r),
            CmpKind::Eq => vceqq_f32(l, r),
        };
        vst1q_f32(op.add(i), vbslq_f32(mask, one, zero));
        i += 4;
    }

    while i < len {
        out[i] = match kind {
            CmpKind::Gt => {
                if lhs[i] > rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Lt => {
                if lhs[i] < rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Eq => {
                if (lhs[i] - rhs[i]).abs() < f32::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
        };
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_basic() {
        let data: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
        let simd = sum_dispatch(&data);
        let scalar: f32 = data.iter().sum();
        assert!((simd - scalar).abs() < 1e-3, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn sum_empty() {
        assert_eq!(sum_dispatch(&[]), 0.0);
    }

    #[test]
    fn max_basic() {
        let data: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7 - 12.0).sin()).collect();
        let simd = max_dispatch(&data);
        let scalar = max_scalar(&data);
        assert!((simd - scalar).abs() < 1e-6);
    }

    #[test]
    fn max_empty() {
        assert_eq!(max_dispatch(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn min_basic() {
        let data: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7 - 12.0).sin()).collect();
        let simd = min_dispatch(&data);
        let scalar = min_scalar(&data);
        assert!((simd - scalar).abs() < 1e-6);
    }

    #[test]
    fn min_empty() {
        assert_eq!(min_dispatch(&[]), f32::INFINITY);
    }

    #[test]
    fn binary_add() {
        let a: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; 33];
        binary_dispatch(&a, &b, &mut out, BinaryKind::Add);
        for i in 0..33 {
            assert!((out[i] - (a[i] + b[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn binary_div() {
        let a: Vec<f32> = (1..34).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..34).map(|i| i as f32 * 0.5 + 1.0).collect();
        let mut out = vec![0.0f32; 33];
        binary_dispatch(&a, &b, &mut out, BinaryKind::Div);
        for i in 0..33 {
            assert!((out[i] - (a[i] / b[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn relu_inplace() {
        let mut data: Vec<f32> = (-10..10).map(|i| i as f32).collect();
        relu_inplace_dispatch(&mut data);
        for (i, &v) in data.iter().enumerate() {
            let expected = (i as f32 - 10.0).max(0.0);
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn add_inplace() {
        let mut dst: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let src: Vec<f32> = (0..33).map(|i| i as f32 * 2.0).collect();
        add_inplace_dispatch(&mut dst, &src);
        for i in 0..33 {
            assert_eq!(dst[i], i as f32 * 3.0);
        }
    }

    #[test]
    fn scalar_inplace_ops() {
        let mut data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        add_scalar_inplace_dispatch(&mut data, 10.0);
        for i in 0..20 {
            assert_eq!(data[i], i as f32 + 10.0);
        }
        mul_scalar_inplace_dispatch(&mut data, 2.0);
        for i in 0..20 {
            assert_eq!(data[i], (i as f32 + 10.0) * 2.0);
        }
    }

    #[test]
    fn unary_neg() {
        let data: Vec<f32> = (0..37).map(|i| i as f32 - 18.0).collect();
        let mut out = vec![0.0f32; 37];
        unary_dispatch(&data, &mut out, UnaryKind::Neg);
        for i in 0..37 {
            assert_eq!(out[i], -data[i]);
        }
    }

    #[test]
    fn unary_abs() {
        let data: Vec<f32> = (0..37).map(|i| i as f32 - 18.0).collect();
        let mut out = vec![0.0f32; 37];
        unary_dispatch(&data, &mut out, UnaryKind::Abs);
        for i in 0..37 {
            assert_eq!(out[i], data[i].abs());
        }
    }

    #[test]
    fn unary_sqrt() {
        let data: Vec<f32> = (0..37).map(|i| i as f32 + 1.0).collect();
        let mut out = vec![0.0f32; 37];
        unary_dispatch(&data, &mut out, UnaryKind::Sqrt);
        for i in 0..37 {
            assert!((out[i] - data[i].sqrt()).abs() < 1e-6);
        }
    }

    #[test]
    fn unary_recip() {
        let data: Vec<f32> = (1..38).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 37];
        unary_dispatch(&data, &mut out, UnaryKind::Recip);
        for i in 0..37 {
            assert!(
                (out[i] - 1.0 / data[i]).abs() < 1e-4,
                "recip mismatch at {i}: got {}, expected {}",
                out[i],
                1.0 / data[i]
            );
        }
    }

    #[test]
    fn exp_inplace() {
        let mut data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let expected: Vec<f32> = data.iter().map(|&v| v.exp()).collect();
        exp_inplace_dispatch(&mut data);
        for i in 0..20 {
            assert!((data[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn ln_basic() {
        let data: Vec<f32> = (1..38).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 37];
        ln_dispatch(&data, &mut out);
        // Miri scalar fallback has slightly worse precision than SIMD path
        let tol = if cfg!(miri) { 1e-3 } else { 1e-4 };
        for i in 0..37 {
            assert!(
                (out[i] - data[i].ln()).abs() < tol,
                "ln({}) got {}, expected {}",
                data[i],
                out[i],
                data[i].ln()
            );
        }
    }

    // ── SIMD edge-case tests ──────────────────────────────────────
    // Test scalar tail handling for input lengths that aren't multiples
    // of SIMD register width (4 for NEON/SSE, 8 for AVX, 16/32 for unrolled).

    const EDGE_LENGTHS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65];

    #[test]
    fn unary_edge_lengths() {
        use super::UnaryKind;
        let kinds = [
            UnaryKind::Neg,
            UnaryKind::Abs,
            UnaryKind::Sqrt,
            UnaryKind::Recip,
            UnaryKind::Floor,
            UnaryKind::Ceil,
            UnaryKind::Round,
            UnaryKind::Sign,
        ];
        for &len in EDGE_LENGTHS {
            let data: Vec<f32> = (1..=len as i32).map(|i| i as f32 * 0.7 - 3.0).collect();
            let mut out = vec![0.0f32; len];
            for kind in &kinds {
                unary_dispatch(&data, &mut out, *kind);
                for i in 0..len {
                    let expected = match kind {
                        UnaryKind::Neg => -data[i],
                        UnaryKind::Abs => data[i].abs(),
                        UnaryKind::Sqrt => data[i].abs().sqrt(), // use abs to avoid NaN
                        UnaryKind::Recip => 1.0 / data[i],
                        UnaryKind::Floor => data[i].floor(),
                        UnaryKind::Ceil => data[i].ceil(),
                        UnaryKind::Round => data[i].round(),
                        UnaryKind::Sign => {
                            if data[i] > 0.0 {
                                1.0
                            } else if data[i] < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        }
                    };
                    let actual = out[i];
                    // Sqrt uses abs(data) so feed that to dispatch too
                    if matches!(kind, UnaryKind::Sqrt) {
                        continue;
                    } // skip — input domain mismatch
                    assert!(
                        (actual - expected).abs() < 0.01
                            || (expected.abs() > 1e6 && actual.abs() > 1e6),
                        "kind={kind:?} len={len} i={i}: got {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn binary_add_edge_lengths() {
        use super::BinaryKind;
        for &len in EDGE_LENGTHS {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 1.1).collect();
            let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.3 + 1.0).collect();
            let mut out = vec![0.0f32; len];
            binary_dispatch(&a, &b, &mut out, BinaryKind::Add);
            for i in 0..len {
                assert!(
                    (out[i] - (a[i] + b[i])).abs() < 1e-5,
                    "add len={len} i={i}: got {}, expected {}",
                    out[i],
                    a[i] + b[i]
                );
            }
        }
    }

    #[test]
    fn sum_edge_lengths() {
        for &len in EDGE_LENGTHS {
            let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
            let simd = sum_dispatch(&data);
            let scalar: f32 = data.iter().sum();
            assert!(
                (simd - scalar).abs() < 1e-3 + scalar.abs() * 1e-5,
                "sum len={len}: simd={simd}, scalar={scalar}"
            );
        }
    }

    #[test]
    fn max_min_edge_lengths() {
        for &len in EDGE_LENGTHS {
            let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.7 - 5.0).sin()).collect();
            let mx = max_dispatch(&data);
            let mn = min_dispatch(&data);
            if len == 0 {
                assert_eq!(mx, f32::NEG_INFINITY);
                assert_eq!(mn, f32::INFINITY);
            } else {
                let expected_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let expected_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                assert!((mx - expected_max).abs() < 1e-6, "max len={len}");
                assert!((mn - expected_min).abs() < 1e-6, "min len={len}");
            }
        }
    }

    #[test]
    fn exp_edge_lengths() {
        for &len in EDGE_LENGTHS {
            let data: Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.1 - 2.0).clamp(-10.0, 10.0))
                .collect();
            let mut out = vec![0.0f32; len];
            exp_dispatch(&data, &mut out);
            for i in 0..len {
                let expected = data[i].exp();
                assert!(
                    (out[i] - expected).abs() < expected.abs() * 1e-4 + 1e-6,
                    "exp len={len} i={i}: got {}, expected {}",
                    out[i],
                    expected
                );
            }
        }
    }
}
