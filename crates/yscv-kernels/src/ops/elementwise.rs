use rayon::{ThreadPool, prelude::*};
use yscv_tensor::{AlignedVec, Tensor};

use super::super::error::KernelError;
use super::config::{
    BinaryKind, PARALLEL_SLICE_CHUNK_ELEMENTS, ParallelElementwiseConfig, should_parallelize_len,
};
use super::simd::{
    binary_same_shape_dispatch, exp_slice_dispatch, relu_slice_dispatch, relu_to_slice_dispatch,
    sigmoid_slice_dispatch, silu_slice_dispatch, tanh_slice_dispatch,
};

// GCD low-overhead parallelism (macOS: ~0.3µs dispatch, Linux: scoped threads).
#[allow(unsafe_code)]
mod par {
    #[cfg(target_os = "macos")]
    use std::ffi::c_void;

    #[cfg(target_os = "macos")]
    #[allow(unsafe_code)]
    unsafe extern "C" {
        fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *const c_void;
        fn dispatch_apply_f(
            iterations: usize,
            queue: *const c_void,
            context: *mut c_void,
            work: unsafe extern "C" fn(*mut c_void, usize),
        );
    }

    #[cfg(target_os = "macos")]
    #[inline]
    #[allow(unsafe_code)]
    pub fn parallel_for<F: Fn(usize) + Sync>(n: usize, f: F) {
        #[allow(unsafe_code)]
        unsafe extern "C" fn call<F: Fn(usize) + Sync>(ctx: *mut c_void, i: usize) {
            unsafe {
                (*(ctx as *const F))(i);
            }
        }
        let queue = unsafe { dispatch_get_global_queue(0, 0) };
        unsafe {
            dispatch_apply_f(n, queue, &f as *const F as *mut c_void, call::<F>);
        }
    }

    #[cfg(not(target_os = "macos"))]
    #[inline]
    pub fn parallel_for<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
        if n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        // Use rayon global thread pool — threads are pre-spawned, ~0.5µs dispatch.
        use rayon::prelude::*;
        (0..n).into_par_iter().for_each(f);
    }
}

/// Elementwise ReLU activation. GCD-parallelized for large tensors.
#[inline]
#[allow(unsafe_code)]
pub fn relu(input: &Tensor) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);

    const PAR_THRESH: usize = 100_000;
    if len >= PAR_THRESH {
        let n_chunks = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let chunk = len.div_ceil(n_chunks);
        let in_ptr = input_data.as_ptr() as usize;
        let out_ptr = output.as_mut_ptr() as usize;
        par::parallel_for(n_chunks, |t| {
            let start = t * chunk;
            let end = (start + chunk).min(len);
            let inp = unsafe {
                std::slice::from_raw_parts((in_ptr as *const f32).add(start), end - start)
            };
            let out = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
            };
            relu_to_slice_dispatch(inp, out);
        });
    } else {
        relu_to_slice_dispatch(input_data, &mut output);
    }

    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// In-place ReLU activation: clamps negative values to zero.
#[inline]
pub fn relu_inplace(tensor: &mut Tensor) {
    relu_slice_dispatch(tensor.data_mut());
}

/// ReLU writing into pre-allocated output tensor. Zero allocation overhead.
#[inline]
pub fn relu_out(input: &Tensor, output: &mut Tensor) {
    debug_assert_eq!(input.shape(), output.shape());
    relu_to_slice_dispatch(input.data(), output.data_mut());
}

/// Elementwise sigmoid activation.
pub fn sigmoid(input: &Tensor) -> Tensor {
    sigmoid_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise ReLU activation with explicit parallelization heuristics.
pub fn relu_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    relu_with_config_and_pool(input, config, None)
}

/// Elementwise sigmoid activation with explicit parallelization heuristics.
pub fn sigmoid_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    sigmoid_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `relu_to_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn relu_with_config_and_pool(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);
    if should_parallelize_len(len, config.min_parallel_elements, thread_pool) {
        let mut work = || {
            output
                .par_chunks_mut(PARALLEL_SLICE_CHUNK_ELEMENTS)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start = chunk_idx * PARALLEL_SLICE_CHUNK_ELEMENTS;
                    let end = start + out_chunk.len();
                    relu_to_slice_dispatch(&input_data[start..end], out_chunk);
                });
        };
        if let Some(pool) = thread_pool {
            pool.install(work);
        } else {
            work();
        }
    } else {
        relu_to_slice_dispatch(input_data, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `sigmoid_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn sigmoid_with_config_and_pool(
    input: &Tensor,
    _config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    // Sigmoid uses a heavy polynomial exp + divide per element.
    // Single-threaded SIMD is faster than rayon chunking for ≤4M elements
    // due to dispatch overhead (62 tasks × 50µs > compute savings).
    let mut output = AlignedVec::<f32>::uninitialized(len);
    sigmoid_slice_dispatch(input_data, &mut output);
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// Elementwise exp activation.
pub fn exp(input: &Tensor) -> Tensor {
    exp_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise exp activation with explicit parallelization heuristics.
pub fn exp_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    exp_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `exp_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn exp_with_config_and_pool(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);
    if should_parallelize_len(len, config.min_parallel_elements, thread_pool) {
        let mut work = || {
            output
                .par_chunks_mut(PARALLEL_SLICE_CHUNK_ELEMENTS)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start = chunk_idx * PARALLEL_SLICE_CHUNK_ELEMENTS;
                    let end = start + out_chunk.len();
                    exp_slice_dispatch(&input_data[start..end], out_chunk);
                });
        };
        if let Some(pool) = thread_pool {
            pool.install(work);
        } else {
            work();
        }
    } else {
        exp_slice_dispatch(input_data, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// Elementwise tanh activation.
pub fn tanh_act(input: &Tensor) -> Tensor {
    tanh_act_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise tanh activation with explicit parallelization heuristics.
pub fn tanh_act_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    tanh_act_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `tanh_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn tanh_act_with_config_and_pool(
    input: &Tensor,
    _config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    // Tanh uses a heavy polynomial exp + divide per element.
    // Single-threaded SIMD is faster than rayon chunking for typical sizes.
    let mut output = AlignedVec::<f32>::uninitialized(len);
    tanh_slice_dispatch(input_data, &mut output);
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

const ACTIVATION_PARALLEL_THRESHOLD: usize = 65536;
const ACTIVATION_CHUNK_SIZE: usize = 8192;

/// Elementwise GELU activation (fast approximation): `x * sigmoid(1.702 * x)`.
///
/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `gelu_slice_out`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn gelu(input: &Tensor) -> Tensor {
    let src = input.data();
    let len = src.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);
    if len >= ACTIVATION_PARALLEL_THRESHOLD {
        output
            .par_chunks_mut(ACTIVATION_CHUNK_SIZE)
            .enumerate()
            .for_each(|(ci, out_chunk)| {
                let start = ci * ACTIVATION_CHUNK_SIZE;
                gelu_slice_out(&src[start..start + out_chunk.len()], out_chunk);
            });
    } else {
        gelu_slice_out(src, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// Elementwise SiLU (Swish) activation: `x * sigmoid(x)`.
///
/// Uses fused SIMD kernel (sigmoid + multiply in one pass) to halve memory bandwidth.
pub fn silu(input: &Tensor) -> Tensor {
    silu_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise SiLU activation with explicit parallelization heuristics.
pub fn silu_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    silu_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `silu_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn silu_with_config_and_pool(
    input: &Tensor,
    _config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    // SiLU uses a heavy polynomial exp + divide + multiply per element.
    // Single-threaded SIMD is faster than rayon chunking for typical sizes.
    let mut output = AlignedVec::<f32>::uninitialized(len);
    silu_slice_dispatch(input_data, &mut output);
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// Elementwise Mish activation: `x * tanh(softplus(x))` = `x * tanh(ln(1 + exp(x)))`.
pub fn mish(input: &Tensor) -> Tensor {
    let mut output = input.clone();
    let data = output.data_mut();
    if data.len() >= ACTIVATION_PARALLEL_THRESHOLD {
        data.par_chunks_mut(ACTIVATION_CHUNK_SIZE)
            .for_each(mish_slice);
    } else {
        mish_slice(data);
    }
    output
}

fn gelu_slice_out(src: &[f32], dst: &mut [f32]) {
    for i in 0..src.len() {
        let x = src[i];
        let a = 1.702 * x;
        let ea = (-a).exp();
        let s = 1.0 / (1.0 + ea);
        dst[i] = x * s;
    }
}

fn mish_slice(data: &mut [f32]) {
    for i in 0..data.len() {
        let x = data[i];
        let sp = (1.0 + x.exp()).ln();
        data[i] = x * sp.tanh();
    }
}

/// Elementwise add with optional parallel same-shape execution.
pub fn add_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    add_with_config_and_pool(lhs, rhs, config, None)
}

pub fn add_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Add)
}

/// Elementwise subtract with optional parallel same-shape execution.
pub fn sub_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    sub_with_config_and_pool(lhs, rhs, config, None)
}

pub fn sub_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Sub)
}

/// Elementwise multiply with optional parallel same-shape execution.
pub fn mul_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    mul_with_config_and_pool(lhs, rhs, config, None)
}

pub fn mul_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Mul)
}

fn binary_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: BinaryKind,
) -> Result<Tensor, KernelError> {
    if lhs.shape() != rhs.shape() {
        return binary_fallback(lhs, rhs, kind);
    }

    let left = lhs.data();
    let right = rhs.data();
    let shape = lhs.shape().to_vec();
    let mut output = AlignedVec::<f32>::uninitialized(left.len());

    if should_parallelize_len(left.len(), config.min_parallel_elements, thread_pool) {
        let mut work = || {
            output
                .par_chunks_mut(PARALLEL_SLICE_CHUNK_ELEMENTS)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start = chunk_idx * PARALLEL_SLICE_CHUNK_ELEMENTS;
                    let end = start + out_chunk.len();
                    binary_same_shape_dispatch(
                        &left[start..end],
                        &right[start..end],
                        out_chunk,
                        kind,
                    );
                });
        };

        if let Some(pool) = thread_pool {
            pool.install(work);
        } else {
            work();
        }
    } else {
        binary_same_shape_dispatch(left, right, &mut output, kind);
    }

    Tensor::from_aligned(shape, output).map_err(Into::into)
}

fn binary_fallback(lhs: &Tensor, rhs: &Tensor, kind: BinaryKind) -> Result<Tensor, KernelError> {
    match kind {
        BinaryKind::Add => lhs.add(rhs),
        BinaryKind::Sub => lhs.sub(rhs),
        BinaryKind::Mul => lhs.mul(rhs),
    }
    .map_err(Into::into)
}
