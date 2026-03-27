// ===========================================================================
// max_reduce, add_reduce dispatchers + implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

// ===========================================================================
// Dispatchers
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
// Scalar fallbacks
// ===========================================================================

#[allow(dead_code)]
pub(super) fn max_reduce_scalar(data: &[f32]) -> f32 {
    let mut acc = f32::NEG_INFINITY;
    for &v in data {
        acc = acc.max(v);
    }
    acc
}

#[allow(dead_code)]
pub(super) fn add_reduce_scalar(data: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &v in data {
        acc += v;
    }
    acc
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
