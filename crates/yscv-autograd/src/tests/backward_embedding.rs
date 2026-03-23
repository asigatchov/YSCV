use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn test_scatter_forward() {
    let mut g = Graph::new();
    // input: 4x3 matrix
    let input = g.variable(
        Tensor::from_vec(
            vec![4, 3],
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
                7.0, 8.0, 9.0, // row 2
                10.0, 11.0, 12.0, // row 3
            ],
        )
        .unwrap(),
    );
    // indices: scatter into rows 1 and 3
    let indices = g.constant(Tensor::from_vec(vec![2], vec![1.0, 3.0]).unwrap());
    // src: 2x3 replacement rows
    let src = g.variable(
        Tensor::from_vec(vec![2, 3], vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]).unwrap(),
    );

    let result = g.scatter(input, indices, src).unwrap();
    let val = g.value(result).unwrap();
    assert_eq!(val.shape(), &[4, 3]);
    assert_eq!(
        val.data(),
        &[
            1.0, 2.0, 3.0, // row 0 unchanged
            100.0, 200.0, 300.0, // row 1 replaced
            7.0, 8.0, 9.0, // row 2 unchanged
            400.0, 500.0, 600.0, // row 3 replaced
        ]
    );
}

#[test]
fn test_embedding_lookup_forward() {
    let mut g = Graph::new();
    // weight: 4x3 embedding matrix (vocab=4, dim=3)
    let weight = g.variable(
        Tensor::from_vec(
            vec![4, 3],
            vec![
                0.1, 0.2, 0.3, // row 0
                0.4, 0.5, 0.6, // row 1
                0.7, 0.8, 0.9, // row 2
                1.0, 1.1, 1.2, // row 3
            ],
        )
        .unwrap(),
    );
    let indices = g.constant(Tensor::from_vec(vec![3], vec![2.0, 0.0, 3.0]).unwrap());

    let result = g.embedding_lookup(weight, indices).unwrap();
    let val = g.value(result).unwrap();
    assert_eq!(val.shape(), &[3, 3]);
    let data = val.data();
    // row 0 of output = weight[2]
    assert!((data[0] - 0.7).abs() < 1e-6);
    assert!((data[1] - 0.8).abs() < 1e-6);
    assert!((data[2] - 0.9).abs() < 1e-6);
    // row 1 of output = weight[0]
    assert!((data[3] - 0.1).abs() < 1e-6);
    assert!((data[4] - 0.2).abs() < 1e-6);
    assert!((data[5] - 0.3).abs() < 1e-6);
    // row 2 of output = weight[3]
    assert!((data[6] - 1.0).abs() < 1e-6);
    assert!((data[7] - 1.1).abs() < 1e-6);
    assert!((data[8] - 1.2).abs() < 1e-6);
}

#[test]
fn test_embedding_lookup_backward() {
    let mut g = Graph::new();
    // weight: 4x3 embedding matrix
    let weight = g.variable(
        Tensor::from_vec(
            vec![4, 3],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        )
        .unwrap(),
    );
    let indices = g.constant(Tensor::from_vec(vec![2], vec![1.0, 3.0]).unwrap());

    let lookup = g.embedding_lookup(weight, indices).unwrap();
    let loss = g.sum(lookup).unwrap();
    g.backward(loss).unwrap();

    let grad = g.grad(weight).unwrap().unwrap();
    assert_eq!(grad.shape(), &[4, 3]);
    let gd = grad.data();
    // row 0: not looked up => grad = 0
    assert!((gd[0]).abs() < 1e-6);
    assert!((gd[1]).abs() < 1e-6);
    assert!((gd[2]).abs() < 1e-6);
    // row 1: looked up once => grad = 1 (from sum backward)
    assert!((gd[3] - 1.0).abs() < 1e-6);
    assert!((gd[4] - 1.0).abs() < 1e-6);
    assert!((gd[5] - 1.0).abs() < 1e-6);
    // row 2: not looked up => grad = 0
    assert!((gd[6]).abs() < 1e-6);
    assert!((gd[7]).abs() < 1e-6);
    assert!((gd[8]).abs() < 1e-6);
    // row 3: looked up once => grad = 1
    assert!((gd[9] - 1.0).abs() < 1e-6);
    assert!((gd[10] - 1.0).abs() < 1e-6);
    assert!((gd[11] - 1.0).abs() < 1e-6);
}

#[test]
fn test_embedding_training() {
    let mut g = Graph::new();
    // vocab=5, dim=3
    let weight = g.variable(
        Tensor::from_vec(
            vec![5, 3],
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
                7.0, 8.0, 9.0, // row 2
                10.0, 11.0, 12.0, // row 3
                13.0, 14.0, 15.0, // row 4
            ],
        )
        .unwrap(),
    );
    // Look up indices [0, 2]
    let indices = g.constant(Tensor::from_vec(vec![2], vec![0.0, 2.0]).unwrap());

    let lookup = g.embedding_lookup(weight, indices).unwrap();
    // loss = sum of all looked-up values
    let loss = g.sum(lookup).unwrap();
    g.backward(loss).unwrap();

    let grad = g.grad(weight).unwrap().unwrap();
    assert_eq!(grad.shape(), &[5, 3]);
    let gd = grad.data();

    // row 0: looked up => grad = 1
    assert!((gd[0] - 1.0).abs() < 1e-6);
    assert!((gd[1] - 1.0).abs() < 1e-6);
    assert!((gd[2] - 1.0).abs() < 1e-6);
    // row 1: not looked up => grad = 0
    assert!((gd[3]).abs() < 1e-6);
    assert!((gd[4]).abs() < 1e-6);
    assert!((gd[5]).abs() < 1e-6);
    // row 2: looked up => grad = 1
    assert!((gd[6] - 1.0).abs() < 1e-6);
    assert!((gd[7] - 1.0).abs() < 1e-6);
    assert!((gd[8] - 1.0).abs() < 1e-6);
    // row 3: not looked up => grad = 0
    assert!((gd[9]).abs() < 1e-6);
    assert!((gd[10]).abs() < 1e-6);
    assert!((gd[11]).abs() < 1e-6);
    // row 4: not looked up => grad = 0
    assert!((gd[12]).abs() < 1e-6);
    assert!((gd[13]).abs() < 1e-6);
    assert!((gd[14]).abs() < 1e-6);
}
