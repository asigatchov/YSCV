use super::*;

#[test]
fn exec_synthetic_cnn_conv_relu_pool_flatten_gemm() {
    // Build a mini ONNX CNN: Conv(NCHW) -> Relu -> GlobalAvgPool -> Flatten -> Gemm
    // Input: [1, 1, 4, 4] (NCHW)
    // Conv: 1->2 channels, 3x3 kernel, stride 1, no pad -> [1, 2, 2, 2]
    // Relu -> [1, 2, 2, 2]
    // GlobalAvgPool -> [1, 2, 1, 1]
    // Flatten -> [1, 2]
    // Gemm (transB=1): [1, 2] @ [1, 2]^T + [1] -> [1, 1]

    // Weight [O=2, I=1, KH=3, KW=3]
    let conv_w_data = vec![0.1f32; 18]; // 2*1*3*3 = 18
    let conv_w = onnx::TensorProto {
        name: Some("conv_w".into()),
        dims: vec![2, 1, 3, 3],
        data_type: Some(1),
        float_data: conv_w_data,
        ..Default::default()
    };

    // Gemm weight [1, 2]
    let fc_w = onnx::TensorProto {
        name: Some("fc_w".into()),
        dims: vec![1, 2],
        data_type: Some(1),
        float_data: vec![1.0, -1.0],
        ..Default::default()
    };
    // Gemm bias [1]
    let fc_b = onnx::TensorProto {
        name: Some("fc_b".into()),
        dims: vec![1],
        data_type: Some(1),
        float_data: vec![0.5],
        ..Default::default()
    };

    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["input".into(), "conv_w".into()],
            output: vec!["conv_out".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![3, 3]),
                make_ints_attr("strides", vec![1, 1]),
            ],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu0".into()),
            input: vec!["conv_out".into()],
            output: vec!["relu_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("GlobalAveragePool".into()),
            name: Some("gap".into()),
            input: vec!["relu_out".into()],
            output: vec!["gap_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Flatten".into()),
            name: Some("flat".into()),
            input: vec!["gap_out".into()],
            output: vec!["flat_out".into()],
            attribute: vec![make_int_attr("axis", 1)],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Gemm".into()),
            name: Some("fc".into()),
            input: vec!["flat_out".into(), "fc_w".into(), "fc_b".into()],
            output: vec!["output".into()],
            attribute: vec![make_int_attr("transB", 1)],
            ..Default::default()
        },
    ];

    let bytes = build_minimal_onnx_model(
        nodes,
        vec![conv_w, fc_w, fc_b],
        vec!["input", "conv_w", "fc_w", "fc_b"],
        vec!["output"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    let input_data: Vec<f32> = (0..16).map(|v| v as f32 / 16.0).collect();
    let input = Tensor::from_vec(vec![1, 1, 4, 4], input_data).unwrap();
    let mut feed = HashMap::new();
    feed.insert("input".to_string(), input);

    let result = run_onnx_model(&model, feed).unwrap();
    let output = &result["output"];
    assert_eq!(output.shape(), &[1, 1]);
    assert!(output.data()[0].is_finite());
}

#[test]
fn exec_nms_basic() {
    // 1 batch, 1 class, 3 boxes
    // Box 0 and 1 overlap heavily, box 2 is separate
    let boxes = Tensor::from_vec(
        vec![1, 3, 4],
        vec![
            0.0, 0.0, 10.0, 10.0, // box 0
            1.0, 1.0, 11.0, 11.0, // box 1 (overlaps box 0)
            50.0, 50.0, 60.0, 60.0, // box 2 (separate)
        ],
    )
    .unwrap();
    let scores = Tensor::from_vec(vec![1, 1, 3], vec![0.9, 0.8, 0.7]).unwrap();
    let max_boxes = Tensor::from_vec(vec![1], vec![10.0]).unwrap();
    let iou_thr = Tensor::from_vec(vec![1], vec![0.5]).unwrap();
    let score_thr = Tensor::from_vec(vec![1], vec![0.0]).unwrap();

    let result = run_single_op(
        "NonMaxSuppression",
        vec![("boxes", boxes), ("scores", scores)],
        vec![
            ("max_boxes", max_boxes),
            ("iou_thr", iou_thr),
            ("score_thr", score_thr),
        ],
        vec![],
        vec!["boxes", "scores", "max_boxes", "iou_thr", "score_thr"],
        "selected",
    );

    // Should keep box 0 (highest score) and box 2 (no overlap).
    // Box 1 is suppressed by box 0.
    assert_eq!(result.shape(), &[2, 3]);
    let data = result.data();
    // First selected: batch=0, class=0, box=0
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    // Second selected: batch=0, class=0, box=2
    assert_eq!(data[3], 0.0);
    assert_eq!(data[4], 0.0);
    assert_eq!(data[5], 2.0);
}
