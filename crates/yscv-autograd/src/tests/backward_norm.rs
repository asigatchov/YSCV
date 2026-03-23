use yscv_tensor::Tensor;

use crate::Graph;

#[test]
fn layer_norm_forward_and_backward() {
    let mut g = Graph::new();
    // Input: [2, 3] (two rows, each normalized over 3 features)
    let x = g.variable(Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let gamma = g.variable(Tensor::from_vec(vec![3], vec![1.0, 1.0, 1.0]).unwrap());
    let beta = g.variable(Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).unwrap());

    let out = g.layer_norm(x, gamma, beta, 1e-5).unwrap();
    assert_eq!(g.value(out).unwrap().shape(), &[2, 3]);

    // Each row should be normalized: mean ~ 0, var ~ 1
    let out_data = g.value(out).unwrap().data().to_vec();
    for row in 0..2 {
        let base = row * 3;
        let row_mean: f32 = out_data[base..base + 3].iter().sum::<f32>() / 3.0;
        assert!(row_mean.abs() < 1e-5, "row {row} mean = {row_mean}");
    }

    let loss = g.sum(out).unwrap();
    g.backward(loss).unwrap();

    // grad_beta = sum of upstream (all 1s) over 2 rows => [2, 2, 2]
    let beta_grad = g.grad(beta).unwrap().unwrap();
    assert_eq!(beta_grad.shape(), &[3]);
    for &v in beta_grad.data() {
        assert!((v - 2.0).abs() < 1e-4, "beta grad {v} not ~ 2.0");
    }

    let gamma_grad = g.grad(gamma).unwrap().unwrap();
    assert_eq!(gamma_grad.shape(), &[3]);

    let input_grad = g.grad(x).unwrap().unwrap();
    assert_eq!(input_grad.shape(), &[2, 3]);
    // Each row's input gradient should sum to ~0 (property of layer norm backward)
    for row in 0..2 {
        let base = row * 3;
        let row_sum: f32 = input_grad.data()[base..base + 3].iter().sum();
        assert!(
            row_sum.abs() < 1e-4,
            "row {row} input grad sum = {row_sum}, expected ~0"
        );
    }
}

#[test]
fn layer_norm_backward_finite_differences() {
    let eps = 1e-4_f32;
    let ln_eps = 1e-5_f32;
    let input_data = vec![1.5, -0.5, 2.0, 0.5];

    // Compute analytical gradients
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![2, 2], input_data.clone()).unwrap());
    let gamma = g.variable(Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap());
    let beta = g.variable(Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap());
    let out = g.layer_norm(x, gamma, beta, ln_eps).unwrap();
    let loss = g.sum(out).unwrap();
    g.backward(loss).unwrap();
    let analytical = g.grad(x).unwrap().unwrap().data().to_vec();

    // Compute numerical gradients via finite differences
    for idx in 0..input_data.len() {
        let mut data_plus = input_data.clone();
        data_plus[idx] += eps;
        let mut g2 = Graph::new();
        let xp = g2.variable(Tensor::from_vec(vec![2, 2], data_plus).unwrap());
        let gp = g2.variable(Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap());
        let bp = g2.variable(Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap());
        let op = g2.layer_norm(xp, gp, bp, ln_eps).unwrap();
        let lp = g2.sum(op).unwrap();
        let loss_plus = g2.value(lp).unwrap().data()[0];

        let mut data_minus = input_data.clone();
        data_minus[idx] -= eps;
        let mut g3 = Graph::new();
        let xm = g3.variable(Tensor::from_vec(vec![2, 2], data_minus).unwrap());
        let gm = g3.variable(Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap());
        let bm = g3.variable(Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap());
        let om = g3.layer_norm(xm, gm, bm, ln_eps).unwrap();
        let lm = g3.sum(om).unwrap();
        let loss_minus = g3.value(lm).unwrap().data()[0];

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        assert!(
            (analytical[idx] - numerical).abs() < 1e-3,
            "layer_norm grad mismatch at {idx}: analytical={}, numerical={}",
            analytical[idx],
            numerical
        );
    }
}

#[test]
fn group_norm_forward_and_backward() {
    let mut g = Graph::new();
    // [1, 2, 2, 4] input, 2 groups of 2 channels
    let x = g.variable(
        Tensor::from_vec(
            vec![1, 2, 2, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );
    let gamma = g.variable(Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap());
    let beta = g.variable(Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap());

    let out = g.group_norm(x, gamma, beta, 2, 1e-5).unwrap();
    assert_eq!(g.value(out).unwrap().shape(), &[1, 2, 2, 4]);

    let loss = g.sum(out).unwrap();
    g.backward(loss).unwrap();

    let beta_grad = g.grad(beta).unwrap().unwrap();
    assert_eq!(beta_grad.shape(), &[4]);
    // grad_beta[c] = number of spatial positions = 4 (2x2)
    for &v in beta_grad.data() {
        assert!((v - 4.0).abs() < 1e-4, "beta grad {v} not ~ 4.0");
    }

    let gamma_grad = g.grad(gamma).unwrap().unwrap();
    assert_eq!(gamma_grad.shape(), &[4]);

    let input_grad = g.grad(x).unwrap().unwrap();
    assert_eq!(input_grad.shape(), &[1, 2, 2, 4]);
}

#[test]
fn group_norm_backward_finite_differences() {
    let eps = 1e-4_f32;
    let gn_eps = 1e-5_f32;
    let input_data: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();

    // Compute analytical gradients
    let mut g = Graph::new();
    let x = g.variable(Tensor::from_vec(vec![1, 2, 2, 4], input_data.clone()).unwrap());
    let gamma = g.variable(Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap());
    let beta = g.variable(Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap());
    let out = g.group_norm(x, gamma, beta, 2, gn_eps).unwrap();
    let loss = g.sum(out).unwrap();
    g.backward(loss).unwrap();
    let analytical = g.grad(x).unwrap().unwrap().data().to_vec();

    // Compute numerical gradients via finite differences
    for idx in 0..input_data.len() {
        let mut data_plus = input_data.clone();
        data_plus[idx] += eps;
        let mut g2 = Graph::new();
        let xp = g2.variable(Tensor::from_vec(vec![1, 2, 2, 4], data_plus).unwrap());
        let gp = g2.variable(Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap());
        let bp = g2.variable(Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap());
        let op = g2.group_norm(xp, gp, bp, 2, gn_eps).unwrap();
        let lp = g2.sum(op).unwrap();
        let loss_plus = g2.value(lp).unwrap().data()[0];

        let mut data_minus = input_data.clone();
        data_minus[idx] -= eps;
        let mut g3 = Graph::new();
        let xm = g3.variable(Tensor::from_vec(vec![1, 2, 2, 4], data_minus).unwrap());
        let gm = g3.variable(Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap());
        let bm = g3.variable(Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap());
        let om = g3.group_norm(xm, gm, bm, 2, gn_eps).unwrap();
        let lm = g3.sum(om).unwrap();
        let loss_minus = g3.value(lm).unwrap().data()[0];

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        assert!(
            (analytical[idx] - numerical).abs() < 1e-2,
            "group_norm grad mismatch at {idx}: analytical={}, numerical={}",
            analytical[idx],
            numerical
        );
    }
}

#[test]
fn backward_batch_norm2d_nhwc_computes_gamma_beta_input_grads() {
    let mut graph = Graph::new();
    // [1, 2, 2, 2] input, 2 channels
    let input = graph.variable(
        Tensor::from_vec(
            vec![1, 2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap(),
    );
    let gamma = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap());
    let beta = graph.variable(Tensor::from_vec(vec![2], vec![0.0, 0.0]).unwrap());
    let running_mean = graph.constant(Tensor::from_vec(vec![2], vec![4.0, 5.0]).unwrap());
    let running_var = graph.constant(Tensor::from_vec(vec![2], vec![4.0, 4.0]).unwrap());

    let out = graph
        .batch_norm2d_nhwc(input, gamma, beta, running_mean, running_var, 1e-5)
        .unwrap();
    assert_eq!(graph.value(out).unwrap().shape(), &[1, 2, 2, 2]);

    let loss = graph.sum(out).unwrap();
    graph.backward(loss).unwrap();

    let beta_grad = graph.grad(beta).unwrap().unwrap();
    assert_eq!(beta_grad.shape(), &[2]);
    // grad_beta[c] = number_of_spatial_elements = 4
    assert!((beta_grad.data()[0] - 4.0).abs() < 1e-4);
    assert!((beta_grad.data()[1] - 4.0).abs() < 1e-4);

    let gamma_grad = graph.grad(gamma).unwrap().unwrap();
    assert_eq!(gamma_grad.shape(), &[2]);

    let input_grad = graph.grad(input).unwrap().unwrap();
    assert_eq!(input_grad.shape(), &[1, 2, 2, 2]);
    // All input grads should be gamma / sqrt(var + eps) = 1 / sqrt(4 + 1e-5) ~ 0.5
    for &g in input_grad.data() {
        assert!((g - 0.5).abs() < 0.01, "input grad {g} not ~ 0.5");
    }
}
