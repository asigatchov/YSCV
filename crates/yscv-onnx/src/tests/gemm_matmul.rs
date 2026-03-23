use super::*;

#[test]
fn exec_matmul_2d() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let out = run_single_op(
        "MatMul",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn exec_gemm_with_transpose_and_bias() {
    // A[1,3], B[2,3] transB=1 -> A @ B^T = [1,2], C=[2]
    let a = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let c = Tensor::from_vec(vec![2], vec![10.0, 20.0]).unwrap();
    let out = run_single_op(
        "Gemm",
        vec![("a", a)],
        vec![("b", b), ("c", c)],
        vec![make_int_attr("transB", 1)],
        vec!["a", "b", "c"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(out.data(), &[11.0, 22.0]);
}

#[test]
fn exec_softmax_single_op() {
    let input = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = run_single_op(
        "Softmax",
        vec![("x", input)],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    let sum: f32 = out.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn onnx_einsum_matmul() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let eq_attr = onnx::AttributeProto {
        name: Some("equation".into()),
        r#type: Some(3), // STRING
        s: Some("ij,jk->ik".as_bytes().to_vec()),
        ..Default::default()
    };
    let out = run_single_op(
        "Einsum",
        vec![("a", a), ("b", b)],
        vec![],
        vec![eq_attr],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 2]);
}

#[test]
fn onnx_matmul_integer() {
    let a = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let out = run_single_op(
        "MatMulInteger",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 2]);
    assert!((out.data()[0] - 19.0).abs() < 1.0);
}
