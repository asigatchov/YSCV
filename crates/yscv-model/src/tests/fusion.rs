use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{
    BatchNorm2dLayer, Conv2dLayer, ModelLayer, SequentialModel, fuse_conv_bn, optimize_sequential,
};

/// Helper: create a Conv2dLayer with given weight data and optional bias.
fn make_conv(
    in_ch: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    weight_data: Vec<f32>,
    bias_data: Option<Vec<f32>>,
) -> Conv2dLayer {
    let weight = Tensor::from_vec(vec![kh, kw, in_ch, out_ch], weight_data).unwrap();
    let bias = bias_data.map(|b| Tensor::from_vec(vec![out_ch], b).unwrap());
    Conv2dLayer::new(in_ch, out_ch, kh, kw, 1, 1, weight, bias).unwrap()
}

/// Helper: create a BatchNorm2dLayer from per-channel vecs.
fn make_bn(
    num_features: usize,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
    eps: f32,
) -> BatchNorm2dLayer {
    BatchNorm2dLayer::new(
        num_features,
        eps,
        Tensor::from_vec(vec![num_features], gamma).unwrap(),
        Tensor::from_vec(vec![num_features], beta).unwrap(),
        Tensor::from_vec(vec![num_features], mean).unwrap(),
        Tensor::from_vec(vec![num_features], var).unwrap(),
    )
    .unwrap()
}

#[test]
fn test_fuse_conv_bn_identity_bn() {
    // BN with gamma=1, beta=0, mean=0, var=1 should leave weights unchanged.
    let out_ch = 2;
    let in_ch = 1;
    let kh = 3;
    let kw = 3;
    let w_data: Vec<f32> = (0..kh * kw * in_ch * out_ch)
        .map(|i| i as f32 * 0.1)
        .collect();
    let b_data = vec![0.5, -0.3];

    let conv = make_conv(in_ch, out_ch, kh, kw, w_data.clone(), Some(b_data.clone()));
    let bn = make_bn(
        out_ch,
        vec![1.0; out_ch],
        vec![0.0; out_ch],
        vec![0.0; out_ch],
        vec![1.0; out_ch],
        1e-5,
    );

    let fused = fuse_conv_bn(&conv, &bn);

    // Weights should be approximately unchanged (tiny epsilon perturbation).
    let fused_w = fused.weight().data();
    for (i, (&orig, &fused_val)) in w_data.iter().zip(fused_w.iter()).enumerate() {
        assert!(
            (orig - fused_val).abs() < 1e-4,
            "weight index {i}: orig={orig} fused={fused_val}"
        );
    }

    // Bias should be approximately unchanged.
    let fused_b = fused.bias().expect("fused should have bias").data();
    for (i, (&orig, &fused_val)) in b_data.iter().zip(fused_b.iter()).enumerate() {
        assert!(
            (orig - fused_val).abs() < 1e-4,
            "bias index {i}: orig={orig} fused={fused_val}"
        );
    }
}

#[test]
fn test_fuse_conv_bn_output_match() {
    // Create Conv2d + BN with known parameters, run input through both separately vs fused.
    let in_ch = 2;
    let out_ch = 3;
    let kh = 3;
    let kw = 3;
    let w_len = kh * kw * in_ch * out_ch;
    let w_data: Vec<f32> = (0..w_len).map(|i| ((i as f32) * 0.05) - 1.0).collect();
    let b_data = vec![0.1, -0.2, 0.3];

    let conv = make_conv(in_ch, out_ch, kh, kw, w_data, Some(b_data));
    let bn = make_bn(
        out_ch,
        vec![1.5, 0.8, 2.0],  // gamma
        vec![0.1, -0.5, 0.3], // beta
        vec![0.2, 0.0, -0.1], // running_mean
        vec![2.0, 0.5, 1.0],  // running_var
        1e-5,
    );

    // Create input: NHWC [1, 4, 4, 2]
    let h = 4;
    let w = 4;
    let input_data: Vec<f32> = (0..h * w * in_ch).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let input = Tensor::from_vec(vec![1, h, w, in_ch], input_data).unwrap();

    // Run Conv then BN separately.
    let conv_out = conv.forward_inference(&input).unwrap();
    let separate_out = bn.forward_inference(&conv_out).unwrap();

    // Run fused Conv.
    let fused = fuse_conv_bn(&conv, &bn);
    let fused_out = fused.forward_inference(&input).unwrap();

    // Compare outputs.
    assert_eq!(separate_out.shape(), fused_out.shape());
    let sep_data = separate_out.data();
    let fus_data = fused_out.data();
    for (i, (&s, &f)) in sep_data.iter().zip(fus_data.iter()).enumerate() {
        assert!(
            (s - f).abs() < 1e-3,
            "output index {i}: separate={s} fused={f} diff={}",
            (s - f).abs()
        );
    }
}

#[test]
fn test_fuse_conv_bn_nonunit_params() {
    // BN with gamma=2, beta=0.5, mean=1, var=4 → verify fused weights are correctly scaled.
    let in_ch = 1;
    let out_ch = 1;
    let kh = 1;
    let kw = 1;
    let w_data = vec![3.0]; // single weight
    let b_data = vec![1.0]; // single bias

    let conv = make_conv(in_ch, out_ch, kh, kw, w_data, Some(b_data));
    let bn = make_bn(
        out_ch,
        vec![2.0], // gamma
        vec![0.5], // beta
        vec![1.0], // mean
        vec![4.0], // var
        0.0001,    // eps (small enough to be negligible for manual check)
    );

    let fused = fuse_conv_bn(&conv, &bn);

    // scale = gamma / sqrt(var + eps) = 2.0 / sqrt(4.0001) ≈ 2.0 / 2.00002 ≈ 0.999990
    let expected_scale = 2.0_f32 / (4.0001_f32).sqrt();
    let expected_w = 3.0 * expected_scale;
    let expected_b = expected_scale * (1.0 - 1.0) + 0.5; // scale * (bias - mean) + beta = 0 + 0.5

    let fused_w = fused.weight().data();
    let fused_b = fused.bias().expect("fused should have bias").data();

    assert!(
        (fused_w[0] - expected_w).abs() < 1e-4,
        "weight: expected={expected_w} got={}",
        fused_w[0]
    );
    assert!(
        (fused_b[0] - expected_b).abs() < 1e-4,
        "bias: expected={expected_b} got={}",
        fused_b[0]
    );
}

#[test]
fn test_fuse_conv_bn_preserves_shape() {
    let in_ch = 3;
    let out_ch = 8;
    let kh = 5;
    let kw = 5;
    let w_len = kh * kw * in_ch * out_ch;
    let w_data: Vec<f32> = (0..w_len).map(|i| (i as f32) * 0.01).collect();
    let b_data = vec![0.0; out_ch];

    let conv = make_conv(in_ch, out_ch, kh, kw, w_data, Some(b_data));
    let bn = BatchNorm2dLayer::identity_init(out_ch, 1e-5).unwrap();

    let fused = fuse_conv_bn(&conv, &bn);

    assert_eq!(fused.weight().shape(), &[kh, kw, in_ch, out_ch]);
    assert_eq!(fused.bias().unwrap().shape(), &[out_ch]);
    assert_eq!(fused.in_channels(), in_ch);
    assert_eq!(fused.out_channels(), out_ch);
    assert_eq!(fused.kernel_h(), kh);
    assert_eq!(fused.kernel_w(), kw);
    assert_eq!(fused.stride_h(), 1);
    assert_eq!(fused.stride_w(), 1);
}

#[test]
fn test_fuse_conv_bn_no_bias() {
    // Conv2d without bias: the fusion should still work, treating missing bias as zeros.
    let in_ch = 1;
    let out_ch = 2;
    let kh = 1;
    let kw = 1;
    let w_data = vec![1.0, 2.0];

    let conv = make_conv(in_ch, out_ch, kh, kw, w_data, None);
    let bn = make_bn(
        out_ch,
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![0.5, -0.5],
        vec![1.0, 1.0],
        1e-5,
    );

    let fused = fuse_conv_bn(&conv, &bn);
    // Fused layer always has a bias (even if conv didn't).
    assert!(fused.bias().is_some());

    // Verify via inference.
    let input = Tensor::from_vec(vec![1, 1, 1, in_ch], vec![1.0]).unwrap();
    let conv_out = conv.forward_inference(&input).unwrap();
    let separate_out = bn.forward_inference(&conv_out).unwrap();
    let fused_out = fused.forward_inference(&input).unwrap();

    let sep = separate_out.data();
    let fus = fused_out.data();
    for (i, (&s, &f)) in sep.iter().zip(fus.iter()).enumerate() {
        assert!((s - f).abs() < 1e-4, "index {i}: separate={s} fused={f}");
    }
}

#[test]
fn test_optimize_sequential_fuses_conv_bn() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    let in_ch = 2;
    let out_ch = 4;
    let kh = 3;
    let kw = 3;
    let w_len = kh * kw * in_ch * out_ch;
    let w_data: Vec<f32> = (0..w_len).map(|i| (i as f32) * 0.01).collect();
    let b_data = vec![0.1; out_ch];

    model
        .add_conv2d(
            in_ch,
            out_ch,
            kh,
            kw,
            1,
            1,
            Tensor::from_vec(vec![kh, kw, in_ch, out_ch], w_data).unwrap(),
            Some(Tensor::from_vec(vec![out_ch], b_data).unwrap()),
        )
        .unwrap();
    model
        .add_batch_norm2d(
            out_ch,
            1e-5,
            Tensor::filled(vec![out_ch], 1.0).unwrap(),
            Tensor::zeros(vec![out_ch]).unwrap(),
            Tensor::zeros(vec![out_ch]).unwrap(),
            Tensor::filled(vec![out_ch], 1.0).unwrap(),
        )
        .unwrap();
    model.add_relu();

    // Original: 3 layers (Conv, BN, ReLU)
    assert_eq!(model.layers().len(), 3);

    let optimized = optimize_sequential(&model, &mut graph);

    // Optimized: 2 layers (fused Conv, ReLU)
    assert_eq!(optimized.layers().len(), 2);
    assert!(matches!(optimized.layers()[0], ModelLayer::Conv2d(_)));
    assert!(matches!(optimized.layers()[1], ModelLayer::ReLU(_)));
}

#[test]
fn test_optimize_sequential_output_match() {
    let in_ch = 2;
    let out_ch = 3;
    let kh = 3;
    let kw = 3;
    let w_len = kh * kw * in_ch * out_ch;
    let w_data: Vec<f32> = (0..w_len).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let b_data = vec![0.1, -0.2, 0.3];

    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_conv2d(
            in_ch,
            out_ch,
            kh,
            kw,
            1,
            1,
            Tensor::from_vec(vec![kh, kw, in_ch, out_ch], w_data).unwrap(),
            Some(Tensor::from_vec(vec![out_ch], b_data).unwrap()),
        )
        .unwrap();
    model
        .add_batch_norm2d(
            out_ch,
            1e-5,
            Tensor::from_vec(vec![out_ch], vec![1.5, 0.8, 2.0]).unwrap(),
            Tensor::from_vec(vec![out_ch], vec![0.1, -0.5, 0.3]).unwrap(),
            Tensor::from_vec(vec![out_ch], vec![0.2, 0.0, -0.1]).unwrap(),
            Tensor::from_vec(vec![out_ch], vec![2.0, 0.5, 1.0]).unwrap(),
        )
        .unwrap();
    model.add_relu();

    let optimized = optimize_sequential(&model, &mut graph);

    let h = 5;
    let w = 5;
    let input_data: Vec<f32> = (0..h * w * in_ch).map(|i| (i as f32) * 0.1 - 2.0).collect();
    let input = Tensor::from_vec(vec![1, h, w, in_ch], input_data).unwrap();

    let original_out = model.forward_inference(&input).unwrap();
    let optimized_out = optimized.forward_inference(&input).unwrap();

    assert_eq!(original_out.shape(), optimized_out.shape());
    let orig = original_out.data();
    let opt = optimized_out.data();
    for (i, (&o, &p)) in orig.iter().zip(opt.iter()).enumerate() {
        assert!(
            (o - p).abs() < 1e-3,
            "index {i}: original={o} optimized={p} diff={}",
            (o - p).abs()
        );
    }
}

#[test]
fn test_sequential_optimize_fuses_conv_bn() {
    // Build model with Conv+BN+ReLU, call optimize(), verify layer count decreased.
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);

    let in_ch = 2;
    let out_ch = 4;
    let kh = 3;
    let kw = 3;
    let w_len = kh * kw * in_ch * out_ch;
    let w_data: Vec<f32> = (0..w_len).map(|i| (i as f32) * 0.01).collect();
    let b_data = vec![0.1; out_ch];

    model
        .add_conv2d(
            in_ch,
            out_ch,
            kh,
            kw,
            1,
            1,
            Tensor::from_vec(vec![kh, kw, in_ch, out_ch], w_data).unwrap(),
            Some(Tensor::from_vec(vec![out_ch], b_data).unwrap()),
        )
        .unwrap();
    model
        .add_batch_norm2d(
            out_ch,
            1e-5,
            Tensor::filled(vec![out_ch], 1.0).unwrap(),
            Tensor::zeros(vec![out_ch]).unwrap(),
            Tensor::zeros(vec![out_ch]).unwrap(),
            Tensor::filled(vec![out_ch], 1.0).unwrap(),
        )
        .unwrap();
    model.add_relu();

    // Original: 3 layers (Conv, BN, ReLU)
    assert_eq!(model.layers().len(), 3);

    // Run inference before optimization for comparison.
    let h = 5;
    let w = 5;
    let input_data: Vec<f32> = (0..h * w * in_ch).map(|i| (i as f32) * 0.1 - 2.0).collect();
    let input = Tensor::from_vec(vec![1, h, w, in_ch], input_data).unwrap();
    let before_out = model.forward_inference(&input).unwrap();

    // Call optimize() in-place.
    let fusions = model.optimize(&mut graph);

    // Should have fused one Conv+BN pair.
    assert_eq!(fusions, 1);
    // Optimized: 2 layers (fused Conv, ReLU).
    assert_eq!(model.layers().len(), 2);
    assert!(matches!(model.layers()[0], ModelLayer::Conv2d(_)));
    assert!(matches!(model.layers()[1], ModelLayer::ReLU(_)));

    // Outputs should match.
    let after_out = model.forward_inference(&input).unwrap();
    assert_eq!(before_out.shape(), after_out.shape());
    let bef = before_out.data();
    let aft = after_out.data();
    for (i, (&b, &a)) in bef.iter().zip(aft.iter()).enumerate() {
        assert!(
            (b - a).abs() < 1e-3,
            "index {i}: before={b} after={a} diff={}",
            (b - a).abs()
        );
    }
}
