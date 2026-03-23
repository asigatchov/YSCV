use super::*;

#[test]
fn exec_flatten_default_axis() {
    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
    let out = run_single_op(
        "Flatten",
        vec![("x", input)],
        vec![],
        vec![make_int_attr("axis", 1)],
        vec!["x"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 12]);
}

#[test]
fn exec_reshape_with_neg1() {
    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
    let shape = Tensor::from_vec(vec![2], vec![6.0, -1.0]).unwrap();
    let out = run_single_op(
        "Reshape",
        vec![("x", input)],
        vec![("shape", shape)],
        vec![],
        vec!["x", "shape"],
        "y",
    );
    assert_eq!(out.shape(), &[6, 4]);
}

#[test]
fn exec_transpose_perm() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = run_single_op(
        "Transpose",
        vec![("x", input)],
        vec![],
        vec![make_ints_attr("perm", vec![1, 0])],
        vec!["x"],
        "y",
    );
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn exec_concat_axis0() {
    let a = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![2, 3], vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let out = run_single_op(
        "Concat",
        vec![("a", a), ("b", b)],
        vec![],
        vec![make_int_attr("axis", 0)],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.shape(), &[3, 3]);
}

#[test]
fn onnx_depth_to_space_basic() {
    let input = Tensor::from_vec(vec![1, 4, 1, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let bs_attr = onnx::AttributeProto {
        name: Some("blocksize".into()),
        r#type: Some(2),
        i: Some(2),
        ..Default::default()
    };
    let out = run_single_op(
        "DepthToSpace",
        vec![("data", input)],
        vec![],
        vec![bs_attr],
        vec!["data"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_eq!(out.data().len(), 4);
}

#[test]
fn onnx_space_to_depth_basic() {
    let input = Tensor::from_vec(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let bs_attr = onnx::AttributeProto {
        name: Some("blocksize".into()),
        r#type: Some(2),
        i: Some(2),
        ..Default::default()
    };
    let out = run_single_op(
        "SpaceToDepth",
        vec![("data", input)],
        vec![],
        vec![bs_attr],
        vec!["data"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 4, 1, 1]);
    assert_eq!(out.data().len(), 4);
}
