// ===========================================================================
// fma_slice_dispatch + impls, matmul_row_dispatch + impls
// ===========================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

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
// Scalar fallback
// ===========================================================================

#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn fma_slice_scalar(a: &[f32], b: &[f32], acc: &mut [f32]) {
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
// FMA SIMD implementations
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
