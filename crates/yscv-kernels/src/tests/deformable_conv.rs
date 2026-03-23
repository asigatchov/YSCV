use yscv_tensor::Tensor;

use crate::{conv2d_nhwc, deformable_conv2d_nhwc};

use super::assert_slice_close;

#[test]
fn deformable_conv2d_zero_offsets_matches_standard_conv2d() {
    // With zero offsets and no padding, deformable conv should match standard conv2d.
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let standard = conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap();

    // out_h = (3-2)/1+1 = 2, out_w = 2, offsets shape: [1, 2, 2, 2*2*2] = [1, 2, 2, 8]
    let offsets = Tensor::zeros(vec![1, 2, 2, 8]).unwrap();
    let deformable = deformable_conv2d_nhwc(&input, &kernel, &offsets, None, 1, 0).unwrap();

    assert_eq!(standard.shape(), deformable.shape());
    assert_slice_close(standard.data(), deformable.data(), 1e-6);
}

#[test]
fn deformable_conv2d_shape() {
    let input = Tensor::from_vec(vec![1, 5, 5, 3], vec![0.0; 75]).unwrap();
    let kernel = Tensor::from_vec(vec![3, 3, 3, 4], vec![0.0; 108]).unwrap();
    // padding=1, stride=1 => out_h = (5+2-3)/1+1 = 5, out_w = 5
    let offsets = Tensor::zeros(vec![1, 5, 5, 3 * 3 * 2]).unwrap();
    let out = deformable_conv2d_nhwc(&input, &kernel, &offsets, None, 1, 1).unwrap();
    assert_eq!(out.shape(), &[1, 5, 5, 4]);
}

#[test]
fn deformable_conv2d_with_offsets_differs_from_standard() {
    let input = Tensor::from_vec(
        vec![1, 4, 4, 1],
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();
    let kernel = Tensor::from_vec(vec![2, 2, 1, 1], vec![1.0, 1.0, 1.0, 1.0]).unwrap();

    let standard = conv2d_nhwc(&input, &kernel, None, 1, 1).unwrap();

    // out_h = (4-2)/1+1 = 3, out_w = 3, offsets: [1, 3, 3, 8]
    // Use non-zero offsets
    let mut offset_data = vec![0.0f32; 3 * 3 * 8];
    // Shift all kernel taps by 0.5 in y
    for i in 0..9 {
        for tap in 0..4 {
            offset_data[i * 8 + tap * 2] = 0.5; // dy
        }
    }
    let offsets = Tensor::from_vec(vec![1, 3, 3, 8], offset_data).unwrap();
    let deformable = deformable_conv2d_nhwc(&input, &kernel, &offsets, None, 1, 0).unwrap();

    assert_eq!(standard.shape(), deformable.shape());
    // Results should differ because offsets shift the sampling positions
    assert_ne!(standard.data(), deformable.data());
}

#[test]
fn deformable_conv2d_with_bias() {
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let kernel = Tensor::from_vec(
        vec![2, 2, 1, 2],
        vec![1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0],
    )
    .unwrap();
    let bias = Tensor::from_vec(vec![2], vec![0.5, 1.0]).unwrap();

    // out: [1, 2, 2, 2], offsets: [1, 2, 2, 8]
    let offsets = Tensor::zeros(vec![1, 2, 2, 8]).unwrap();

    let with_bias = deformable_conv2d_nhwc(&input, &kernel, &offsets, Some(&bias), 1, 0).unwrap();
    let without_bias = deformable_conv2d_nhwc(&input, &kernel, &offsets, None, 1, 0).unwrap();

    assert_eq!(with_bias.shape(), &[1, 2, 2, 2]);
    // Each output should differ by exactly the bias value
    for (_wb, _nb) in with_bias.data().iter().zip(without_bias.data().iter()) {
        // even indices get +0.5, odd get +1.0 -- but indexing is interleaved
    }
    for i in 0..4 {
        let diff0 = with_bias.data()[i * 2] - without_bias.data()[i * 2];
        let diff1 = with_bias.data()[i * 2 + 1] - without_bias.data()[i * 2 + 1];
        assert!((diff0 - 0.5).abs() < 1e-6, "bias channel 0 mismatch");
        assert!((diff1 - 1.0).abs() < 1e-6, "bias channel 1 mismatch");
    }
}

#[test]
fn deformable_conv2d_with_padding() {
    // 3x3 input with 3x3 kernel, padding=1 => same spatial size
    let input = Tensor::from_vec(
        vec![1, 3, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let kernel = Tensor::from_vec(
        vec![3, 3, 1, 1],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    // Identity kernel (center tap = 1) with zero offsets => output = input
    let offsets = Tensor::zeros(vec![1, 3, 3, 18]).unwrap();
    let out = deformable_conv2d_nhwc(&input, &kernel, &offsets, None, 1, 1).unwrap();
    assert_eq!(out.shape(), &[1, 3, 3, 1]);
    assert_slice_close(out.data(), input.data(), 1e-6);
}
