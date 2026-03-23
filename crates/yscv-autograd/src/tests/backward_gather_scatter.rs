use yscv_tensor::Tensor;

use crate::Graph;

/// Verify gather through graph produces correct values.
#[test]
fn gather_forward_autograd() {
    let mut g = Graph::new();
    // input: 3x4 matrix
    let input = g.variable(
        Tensor::from_vec(
            vec![3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ],
        )
        .unwrap(),
    );
    // index: gather along axis 0, selecting rows [2, 0, 1, 0] for each column
    let index = g.constant(
        Tensor::from_vec(
            vec![4, 4],
            vec![
                2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0,
            ],
        )
        .unwrap(),
    );

    let result = g.gather(input, 0, index).unwrap();
    let val = g.value(result).unwrap();
    assert_eq!(val.shape(), &[4, 4]);

    // Verify a few values manually.
    // gather(axis=0): output[i,j] = input[index[i,j], j]
    let data = val.data();
    // output[0,0] = input[2, 0] = 9.0
    assert!((data[0] - 9.0).abs() < 1e-6);
    // output[0,1] = input[0, 1] = 2.0
    assert!((data[1] - 2.0).abs() < 1e-6);
    // output[0,2] = input[1, 2] = 7.0
    assert!((data[2] - 7.0).abs() < 1e-6);
    // output[1,0] = input[1, 0] = 5.0
    assert!((data[4] - 5.0).abs() < 1e-6);
}

/// Verify gradient flows correctly through gather.
/// gather backward: grad_input = zeros_like(input).scatter_add(axis, index, grad_output)
#[test]
fn gather_backward() {
    let mut g = Graph::new();
    // input: 3x2 matrix
    let input =
        g.variable(Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    // index along axis=0: 2x2, selecting rows [0, 2] and [1, 0]
    let index = g.constant(Tensor::from_vec(vec![2, 2], vec![0.0, 2.0, 1.0, 0.0]).unwrap());

    let gathered = g.gather(input, 0, index).unwrap();
    // gathered shape: [2, 2]
    // gathered = [[input[0,0], input[2,1]], [input[1,0], input[0,1]]]
    //          = [[1.0, 6.0], [3.0, 2.0]]
    let gval = g.value(gathered).unwrap().data().to_vec();
    assert!((gval[0] - 1.0).abs() < 1e-6);
    assert!((gval[1] - 6.0).abs() < 1e-6);
    assert!((gval[2] - 3.0).abs() < 1e-6);
    assert!((gval[3] - 2.0).abs() < 1e-6);

    let loss = g.sum(gathered).unwrap();
    g.backward(loss).unwrap();

    let grad = g.grad(input).unwrap().unwrap();
    assert_eq!(grad.shape(), &[3, 2]);
    let gd = grad.data();
    // grad_input = zeros(3,2).scatter_add(0, index, ones(2,2))
    // index = [[0,2],[1,0]]
    // For col 0: index vals are 0 and 1 => grad[0,0] += 1, grad[1,0] += 1
    // For col 1: index vals are 2 and 0 => grad[2,1] += 1, grad[0,1] += 1
    assert!((gd[0] - 1.0).abs() < 1e-6); // [0,0]
    assert!((gd[1] - 1.0).abs() < 1e-6); // [0,1]
    assert!((gd[2] - 1.0).abs() < 1e-6); // [1,0]
    assert!((gd[3] - 0.0).abs() < 1e-6); // [1,1] -- not gathered
    assert!((gd[4] - 0.0).abs() < 1e-6); // [2,0] -- not gathered
    assert!((gd[5] - 1.0).abs() < 1e-6); // [2,1]
}

/// Verify scatter_add gradient:
/// - grad_input = grad_output (identity)
/// - grad_src = grad_output.gather(axis, index)
#[test]
fn scatter_add_backward() {
    let mut g = Graph::new();
    // input: 3x2 zeros
    let input =
        g.variable(Tensor::from_vec(vec![3, 2], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap());
    // index: 2x2, scatter into rows [0, 2] and [1, 0]
    let index = g.constant(Tensor::from_vec(vec![2, 2], vec![0.0, 2.0, 1.0, 0.0]).unwrap());
    // src: 2x2
    let src = g.variable(Tensor::from_vec(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap());

    let result = g.scatter_add(input, index, src, 0).unwrap();
    // Verify forward: zeros.scatter_add(0, [[0,2],[1,0]], [[10,20],[30,40]])
    let val = g.value(result).unwrap();
    assert_eq!(val.shape(), &[3, 2]);
    let vd = val.data();
    // row 0, col 0: src[0,0]=10 scattered to row 0 => 10.0
    // row 0, col 1: src[1,1]=40 scattered to row 0 => 40.0
    // row 1, col 0: src[1,0]=30 scattered to row 1 => 30.0
    // row 1, col 1: nothing scattered => 0.0
    // row 2, col 0: nothing scattered => 0.0
    // row 2, col 1: src[0,1]=20 scattered to row 2 => 20.0
    assert!((vd[0] - 10.0).abs() < 1e-6);
    assert!((vd[1] - 40.0).abs() < 1e-6);
    assert!((vd[2] - 30.0).abs() < 1e-6);
    assert!((vd[3] - 0.0).abs() < 1e-6);
    assert!((vd[4] - 0.0).abs() < 1e-6);
    assert!((vd[5] - 20.0).abs() < 1e-6);

    let loss = g.sum(result).unwrap();
    g.backward(loss).unwrap();

    // grad_input = grad_output = all 1s (identity)
    let grad_input = g.grad(input).unwrap().unwrap();
    assert_eq!(grad_input.shape(), &[3, 2]);
    for &v in grad_input.data() {
        assert!((v - 1.0).abs() < 1e-6);
    }

    // grad_src = grad_output.gather(axis, index) = ones(3,2).gather(0, [[0,2],[1,0]])
    // All gathered values are 1.0 since upstream is all ones
    let grad_src = g.grad(src).unwrap().unwrap();
    assert_eq!(grad_src.shape(), &[2, 2]);
    for &v in grad_src.data() {
        assert!((v - 1.0).abs() < 1e-6);
    }
}
