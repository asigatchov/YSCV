use rayon::{ThreadPool, prelude::*};
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{
    ParallelElementwiseConfig, Pool2dKind, Pool2dPlan, Pool2dSpec, should_parallelize_len,
};

pub fn max_pool2d_nhwc_with_config_and_pool(
    input: &Tensor,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    pool2d_nhwc_with_config_and_pool(input, spec, config, thread_pool, Pool2dKind::Max)
}

pub fn avg_pool2d_nhwc_with_config_and_pool(
    input: &Tensor,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    pool2d_nhwc_with_config_and_pool(input, spec, config, thread_pool, Pool2dKind::Avg)
}

fn pool2d_nhwc_with_config_and_pool(
    input: &Tensor,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: Pool2dKind,
) -> Result<Tensor, KernelError> {
    let plan = build_pool2d_plan(input, spec)?;
    let data = input.data();
    let out_row_len = plan.out_w * plan.channels;
    if plan.output_len == 0 || out_row_len == 0 {
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.channels],
            AlignedVec::<f32>::calloc(plan.output_len),
        )
        .map_err(Into::into);
    }

    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        let mut work = || {
            output
                .par_chunks_mut(out_row_len)
                .enumerate()
                .for_each(|(row_idx, out_row)| {
                    pool2d_nhwc_row(data, plan, row_idx, out_row, kind);
                });
        };
        if let Some(pool) = thread_pool {
            pool.install(work);
        } else {
            work();
        }
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            pool2d_nhwc_row(data, plan, row_idx, out_row, kind);
        }
    }

    Tensor::from_aligned(
        vec![plan.batch, plan.out_h, plan.out_w, plan.channels],
        output,
    )
    .map_err(Into::into)
}

fn build_pool2d_plan(input: &Tensor, spec: Pool2dSpec) -> Result<Pool2dPlan, KernelError> {
    let kernel_h = spec.kernel_h;
    let kernel_w = spec.kernel_w;
    let stride_h = spec.stride_h;
    let stride_w = spec.stride_w;
    if input.rank() != 4 {
        return Err(KernelError::InvalidPoolRank {
            got_rank: input.rank(),
        });
    }
    if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidPoolParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }

    let batch = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let channels = input.shape()[3];
    if kernel_h > in_h || kernel_w > in_w {
        return Err(KernelError::PoolKernelLargerThanInput {
            input_h: in_h,
            input_w: in_w,
            kernel_h,
            kernel_w,
        });
    }

    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;

    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, channels],
            })
        })?;

    Ok(Pool2dPlan {
        batch,
        in_h,
        in_w,
        channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_len,
    })
}

fn pool2d_nhwc_row(
    input: &[f32],
    plan: Pool2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    kind: Pool2dKind,
) {
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let window_area = (plan.kernel_h * plan.kernel_w) as f32;
    let inv_area = 1.0 / window_area;

    // Fast path: 2×2 max pool with stride 2 and channels == 1.
    // Process 4 output pixels at a time by loading 8 consecutive input pixels
    // and taking pairwise max.
    if plan.kernel_h == 2
        && plan.kernel_w == 2
        && plan.stride_h == 2
        && plan.stride_w == 2
        && plan.channels == 1
        && matches!(kind, Pool2dKind::Max)
    {
        pool2d_2x2s2_max_row(input, plan, batch_input_base, in_y0, out_row);
        return;
    }

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_cell_base = out_x * plan.channels;
        let out_slice = &mut out_row[out_cell_base..out_cell_base + plan.channels];

        // Initialize
        match kind {
            Pool2dKind::Max => out_slice.fill(f32::NEG_INFINITY),
            Pool2dKind::Avg => out_slice.fill(0.0),
        }

        // Accumulate over kernel window
        for ky in 0..plan.kernel_h {
            let in_y = in_y0 + ky;
            let row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.channels;
            for kx in 0..plan.kernel_w {
                let pixel_base = row_base + kx * plan.channels;
                let in_slice = &input[pixel_base..pixel_base + plan.channels];
                pool_accumulate(out_slice, in_slice, kind);
            }
        }

        // Finalize avg
        if matches!(kind, Pool2dKind::Avg) {
            for v in out_slice.iter_mut() {
                *v *= inv_area;
            }
        }
    }
}

/// Optimized 2x2 max-pool with stride 2, channels==1.
/// Processes 4 output pixels at a time using SIMD pairwise max.
#[allow(unsafe_code)]
fn pool2d_2x2s2_max_row(
    input: &[f32],
    plan: Pool2dPlan,
    batch_input_base: usize,
    in_y0: usize,
    out_row: &mut [f32],
) {
    let in_w = plan.in_w;
    let row0_base = batch_input_base + in_y0 * in_w;
    let row1_base = batch_input_base + (in_y0 + 1) * in_w;

    let mut out_x = 0usize;

    // SIMD batch: process 4 output pixels (= 8 input pixels per row) at a time.
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        unsafe {
            pool2d_2x2s2_max_row_neon(input, row0_base, row1_base, out_row, plan.out_w, &mut out_x);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && std::is_x86_feature_detected!("sse") {
        unsafe {
            pool2d_2x2s2_max_row_sse(input, row0_base, row1_base, out_row, plan.out_w, &mut out_x);
        }
    }

    // Scalar tail
    while out_x < plan.out_w {
        let in_x0 = out_x * 2;
        let a = input[row0_base + in_x0];
        let b = input[row0_base + in_x0 + 1];
        let c = input[row1_base + in_x0];
        let d = input[row1_base + in_x0 + 1];
        out_row[out_x] = a.max(b).max(c.max(d));
        out_x += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn pool2d_2x2s2_max_row_neon(
    input: &[f32],
    row0_base: usize,
    row1_base: usize,
    out_row: &mut [f32],
    out_w: usize,
    out_x: &mut usize,
) {
    use std::arch::aarch64::*;
    let ip = input.as_ptr();
    let op = out_row.as_mut_ptr();
    while *out_x + 4 <= out_w {
        let in_x0 = *out_x * 2;
        // Load 8 consecutive floats from each row (covers 4 output pixels × stride 2)
        let r0a = vld1q_f32(ip.add(row0_base + in_x0));
        let r0b = vld1q_f32(ip.add(row0_base + in_x0 + 4));
        let r1a = vld1q_f32(ip.add(row1_base + in_x0));
        let r1b = vld1q_f32(ip.add(row1_base + in_x0 + 4));
        // Max across rows
        let max0 = vmaxq_f32(r0a, r1a);
        let max1 = vmaxq_f32(r0b, r1b);
        // Pairwise max: take max of even/odd elements → 4 results
        let result = vpmaxq_f32(max0, max1);
        vst1q_f32(op.add(*out_x), result);
        *out_x += 4;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse3")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn pool2d_2x2s2_max_row_sse(
    input: &[f32],
    row0_base: usize,
    row1_base: usize,
    out_row: &mut [f32],
    out_w: usize,
    out_x: &mut usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let ip = input.as_ptr();
    let op = out_row.as_mut_ptr();
    while *out_x + 4 <= out_w {
        let in_x0 = *out_x * 2;
        // Load 8 consecutive floats from each row
        let r0a = _mm_loadu_ps(ip.add(row0_base + in_x0));
        let r0b = _mm_loadu_ps(ip.add(row0_base + in_x0 + 4));
        let r1a = _mm_loadu_ps(ip.add(row1_base + in_x0));
        let r1b = _mm_loadu_ps(ip.add(row1_base + in_x0 + 4));
        // Max across rows
        let max0 = _mm_max_ps(r0a, r1a); // [m0, m1, m2, m3]
        let max1 = _mm_max_ps(r0b, r1b); // [m4, m5, m6, m7]
        // Pairwise max: shuffle to get even/odd pairs then max
        // max0 = [m0, m1, m2, m3], max1 = [m4, m5, m6, m7]
        // We want: max(m0,m1), max(m2,m3), max(m4,m5), max(m6,m7)
        let evens = _mm_shuffle_ps(max0, max1, 0b10_00_10_00); // [m0, m2, m4, m6]
        let odds = _mm_shuffle_ps(max0, max1, 0b11_01_11_01); // [m1, m3, m5, m7]
        let result = _mm_max_ps(evens, odds);
        _mm_storeu_ps(op.add(*out_x), result);
        *out_x += 4;
    }
}

/// SIMD-accelerated pool accumulation across channels
#[allow(unsafe_code)]
fn pool_accumulate(out: &mut [f32], input: &[f32], kind: Pool2dKind) {
    let len = out.len();
    debug_assert_eq!(len, input.len());

    if cfg!(miri) || len < 4 {
        match kind {
            Pool2dKind::Max => {
                for i in 0..len {
                    out[i] = out[i].max(input[i]);
                }
            }
            Pool2dKind::Avg => {
                for i in 0..len {
                    out[i] += input[i];
                }
            }
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { pool_accumulate_neon(out, input, kind) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse") {
            unsafe { pool_accumulate_sse(out, input, kind) };
            return;
        }
    }

    match kind {
        Pool2dKind::Max => {
            for i in 0..len {
                out[i] = out[i].max(input[i]);
            }
        }
        Pool2dKind::Avg => {
            for i in 0..len {
                out[i] += input[i];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn pool_accumulate_neon(out: &mut [f32], input: &[f32], kind: Pool2dKind) {
    use std::arch::aarch64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let ip = input.as_ptr();
    let mut i = 0usize;
    match kind {
        Pool2dKind::Max => {
            while i + 4 <= len {
                let o = vld1q_f32(op.add(i));
                let v = vld1q_f32(ip.add(i));
                vst1q_f32(op.add(i), vmaxq_f32(o, v));
                i += 4;
            }
            while i < len {
                let o = *op.add(i);
                let v = *ip.add(i);
                *op.add(i) = if o > v { o } else { v };
                i += 1;
            }
        }
        Pool2dKind::Avg => {
            while i + 4 <= len {
                let o = vld1q_f32(op.add(i));
                let v = vld1q_f32(ip.add(i));
                vst1q_f32(op.add(i), vaddq_f32(o, v));
                i += 4;
            }
            while i < len {
                *op.add(i) += *ip.add(i);
                i += 1;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn pool_accumulate_sse(out: &mut [f32], input: &[f32], kind: Pool2dKind) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let ip = input.as_ptr();
    let mut i = 0usize;
    match kind {
        Pool2dKind::Max => {
            while i + 4 <= len {
                let o = _mm_loadu_ps(op.add(i));
                let v = _mm_loadu_ps(ip.add(i));
                _mm_storeu_ps(op.add(i), _mm_max_ps(o, v));
                i += 4;
            }
            while i < len {
                let o = *op.add(i);
                let v = *ip.add(i);
                *op.add(i) = if o > v { o } else { v };
                i += 1;
            }
        }
        Pool2dKind::Avg => {
            while i + 4 <= len {
                let o = _mm_loadu_ps(op.add(i));
                let v = _mm_loadu_ps(ip.add(i));
                _mm_storeu_ps(op.add(i), _mm_add_ps(o, v));
                i += 4;
            }
            while i < len {
                *op.add(i) += *ip.add(i);
                i += 1;
            }
        }
    }
}
