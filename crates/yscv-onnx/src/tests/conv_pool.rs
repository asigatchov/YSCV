use super::*;

#[test]
fn exec_conv_with_padding() {
    // Conv with pads=[1,1,1,1] to produce same-size output
    // Input: [1, 1, 3, 3], kernel 3x3 stride 1 pad 1 -> [1, 1, 3, 3]
    let conv_w_data = vec![0.0f32; 9];
    let mut cw = conv_w_data;
    cw[4] = 1.0; // center element = 1 -> identity conv
    let conv_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![1, 1, 3, 3],
        data_type: Some(1),
        float_data: cw,
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("conv".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![conv_w], vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();

    let input = Tensor::from_vec(
        vec![1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input.clone());
    let result = run_onnx_model(&model, feed).unwrap();
    let output = &result["y"];
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
    // center pixel should be preserved by identity conv
    assert!((output.data()[4] - 5.0).abs() < 1e-5);
}
