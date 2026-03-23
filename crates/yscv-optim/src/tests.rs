use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use super::{
    Adagrad, Adam, AdamW, CosineAnnealingLr, CosineAnnealingWarmRestarts, CyclicLr, ExponentialLr,
    Lamb, LambdaLr, Lars, LearningRate, LinearWarmupLr, Lookahead, LrScheduler, MultiStepLr,
    OneCycleLr, OptimError, PolynomialDecayLr, RAdam, ReduceLrOnPlateau, RmsProp, Sgd, StepLr,
    clip_grad_norm_, clip_grad_value_,
};

fn assert_slice_approx_eq(left: &[f32], right: &[f32], eps: f32) {
    assert_eq!(left.len(), right.len());
    for (index, (lhs, rhs)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() <= eps,
            "index={index} left={lhs} right={rhs} eps={eps}"
        );
    }
}

#[test]
fn sgd_step_updates_tensor_without_momentum() {
    let mut optimizer = Sgd::new(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();

    optimizer.step(0, &mut weights, &grad).unwrap();
    assert_eq!(weights.data(), &[0.95, 2.1]);
}

#[test]
fn sgd_step_applies_weight_decay() {
    let mut optimizer = Sgd::new(0.1).unwrap().with_weight_decay(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::zeros(vec![2]).unwrap();

    optimizer.step(1, &mut weights, &grad).unwrap();
    assert_eq!(weights.data(), &[0.99, 1.98]);
}

#[test]
fn sgd_step_with_momentum_accumulates_velocity() {
    let mut optimizer = Sgd::new(1.0).unwrap().with_momentum(0.9).unwrap();
    let mut weights = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let grad = Tensor::from_vec(vec![1], vec![1.0]).unwrap();

    optimizer.step(2, &mut weights, &grad).unwrap();
    assert_eq!(weights.data(), &[-1.0]);

    optimizer.step(2, &mut weights, &grad).unwrap();
    assert_eq!(weights.data(), &[-2.9]);
}

#[test]
fn sgd_step_with_nesterov_works() {
    let mut optimizer = Sgd::new(1.0)
        .unwrap()
        .with_momentum(0.9)
        .unwrap()
        .with_nesterov(true)
        .unwrap();
    let mut weights = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let grad = Tensor::from_vec(vec![1], vec![1.0]).unwrap();

    optimizer.step(3, &mut weights, &grad).unwrap();
    assert_eq!(weights.data(), &[-1.9]);
}

#[test]
fn sgd_rejects_shape_mismatch() {
    let mut optimizer = Sgd::new(0.1).unwrap();
    let mut weights = Tensor::zeros(vec![2]).unwrap();
    let grad = Tensor::zeros(vec![3]).unwrap();

    let err = optimizer.step(0, &mut weights, &grad).unwrap_err();
    assert_eq!(
        err,
        OptimError::ShapeMismatch {
            weights: vec![2],
            grad: vec![3]
        }
    );
}

#[test]
fn step_graph_node_updates_variable_from_backward_grad() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let c = graph.constant(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());
    let y = graph.add(x, c).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let mut optimizer = Sgd::new(0.5).unwrap();
    optimizer.step_graph_node(&mut graph, x).unwrap();

    let updated = graph.value(x).unwrap();
    assert_eq!(updated.data(), &[0.5, 1.5]);
}

#[test]
fn step_graph_node_requires_gradient() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let mut optimizer = Sgd::new(0.1).unwrap();

    let err = optimizer.step_graph_node(&mut graph, x).unwrap_err();
    assert_eq!(err, OptimError::MissingGradient { node: x.0 });
}

#[test]
fn adam_step_updates_tensor() {
    let mut optimizer = Adam::new(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();

    optimizer.step(11, &mut weights, &grad).unwrap();
    assert_slice_approx_eq(weights.data(), &[0.9, 2.1], 1e-6);

    optimizer.step(11, &mut weights, &grad).unwrap();
    assert_slice_approx_eq(weights.data(), &[0.8, 2.2], 1e-5);
}

#[test]
fn adam_step_applies_weight_decay() {
    let mut optimizer = Adam::new(0.1).unwrap().with_weight_decay(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::zeros(vec![2]).unwrap();

    optimizer.step(12, &mut weights, &grad).unwrap();
    assert_eq!(weights.data(), &[0.9, 1.9]);
}

#[test]
fn adam_rejects_invalid_configuration() {
    let beta1_err = Adam::new(0.1).unwrap().with_beta1(1.0).unwrap_err();
    assert_eq!(beta1_err, OptimError::InvalidBeta1 { beta1: 1.0 });

    let beta2_err = Adam::new(0.1).unwrap().with_beta2(-0.1).unwrap_err();
    assert_eq!(beta2_err, OptimError::InvalidBeta2 { beta2: -0.1 });

    let epsilon_err = Adam::new(0.1).unwrap().with_epsilon(0.0).unwrap_err();
    assert_eq!(epsilon_err, OptimError::InvalidEpsilon { epsilon: 0.0 });
}

#[test]
fn adam_step_graph_node_updates_variable_from_backward_grad() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let c = graph.constant(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());
    let y = graph.add(x, c).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let mut optimizer = Adam::new(0.1).unwrap();
    optimizer.step_graph_node(&mut graph, x).unwrap();

    let updated = graph.value(x).unwrap();
    assert_eq!(updated.data(), &[0.9, 1.9]);
}

#[test]
fn adam_step_graph_node_requires_gradient() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let mut optimizer = Adam::new(0.1).unwrap();

    let err = optimizer.step_graph_node(&mut graph, x).unwrap_err();
    assert_eq!(err, OptimError::MissingGradient { node: x.0 });
}

#[test]
fn adamw_step_updates_tensor() {
    let mut optimizer = AdamW::new(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();

    optimizer.step(21, &mut weights, &grad).unwrap();
    assert_slice_approx_eq(weights.data(), &[0.9, 2.1], 1e-6);
}

#[test]
fn adamw_applies_decoupled_weight_decay() {
    let mut optimizer = AdamW::new(0.1).unwrap().with_weight_decay(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::zeros(vec![2]).unwrap();

    optimizer.step(22, &mut weights, &grad).unwrap();
    assert_eq!(weights.data(), &[0.99, 1.98]);
}

#[test]
fn adamw_step_graph_node_updates_variable_from_backward_grad() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let c = graph.constant(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());
    let y = graph.add(x, c).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let mut optimizer = AdamW::new(0.1).unwrap();
    optimizer.step_graph_node(&mut graph, x).unwrap();

    let updated = graph.value(x).unwrap();
    assert_eq!(updated.data(), &[0.9, 1.9]);
}

#[test]
fn adamw_rejects_invalid_configuration() {
    let beta1_err = AdamW::new(0.1).unwrap().with_beta1(1.0).unwrap_err();
    assert_eq!(beta1_err, OptimError::InvalidBeta1 { beta1: 1.0 });

    let beta2_err = AdamW::new(0.1).unwrap().with_beta2(-0.1).unwrap_err();
    assert_eq!(beta2_err, OptimError::InvalidBeta2 { beta2: -0.1 });

    let epsilon_err = AdamW::new(0.1).unwrap().with_epsilon(0.0).unwrap_err();
    assert_eq!(epsilon_err, OptimError::InvalidEpsilon { epsilon: 0.0 });
}

#[test]
fn rmsprop_step_updates_tensor() {
    let mut optimizer = RmsProp::new(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let grad = Tensor::from_vec(vec![1], vec![2.0]).unwrap();

    optimizer.step(31, &mut weights, &grad).unwrap();
    assert_slice_approx_eq(weights.data(), &[0.0], 1e-5);
}

#[test]
fn rmsprop_step_with_momentum_accumulates_buffer() {
    let mut optimizer = RmsProp::new(0.1).unwrap().with_momentum(0.9).unwrap();
    let mut weights = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let grad = Tensor::from_vec(vec![1], vec![2.0]).unwrap();

    optimizer.step(32, &mut weights, &grad).unwrap();
    optimizer.step(32, &mut weights, &grad).unwrap();

    assert_slice_approx_eq(weights.data(), &[-1.6088824], 1e-5);
}

#[test]
fn rmsprop_centered_path_differs_from_uncentered() {
    let mut centered = RmsProp::new(0.1).unwrap().with_centered(true);
    let mut uncentered = RmsProp::new(0.1).unwrap();
    let mut centered_weights = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let mut uncentered_weights = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let grad = Tensor::from_vec(vec![1], vec![2.0]).unwrap();

    centered.step(33, &mut centered_weights, &grad).unwrap();
    uncentered.step(34, &mut uncentered_weights, &grad).unwrap();

    assert!(centered_weights.data()[0] < uncentered_weights.data()[0]);
}

#[test]
fn rmsprop_step_graph_node_updates_variable_from_backward_grad() {
    let mut graph = Graph::new();
    let x = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let c = graph.constant(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());
    let y = graph.add(x, c).unwrap();
    let loss = graph.sum(y).unwrap();
    graph.backward(loss).unwrap();

    let mut optimizer = RmsProp::new(0.1).unwrap();
    optimizer.step_graph_node(&mut graph, x).unwrap();

    let updated = graph.value(x).unwrap();
    assert_slice_approx_eq(updated.data(), &[0.0, 1.0], 1e-5);
}

#[test]
fn rmsprop_rejects_invalid_configuration() {
    let alpha_err = RmsProp::new(0.1).unwrap().with_alpha(1.0).unwrap_err();
    assert_eq!(alpha_err, OptimError::InvalidRmsPropAlpha { alpha: 1.0 });

    let momentum_err = RmsProp::new(0.1).unwrap().with_momentum(1.0).unwrap_err();
    assert_eq!(momentum_err, OptimError::InvalidMomentum { momentum: 1.0 });

    let epsilon_err = RmsProp::new(0.1).unwrap().with_epsilon(0.0).unwrap_err();
    assert_eq!(epsilon_err, OptimError::InvalidEpsilon { epsilon: 0.0 });
}

#[test]
fn learning_rate_trait_allows_runtime_lr_update() {
    let mut sgd = Sgd::new(0.01).unwrap();
    assert_eq!(sgd.learning_rate(), 0.01);
    <Sgd as LearningRate>::set_learning_rate(&mut sgd, 0.02).unwrap();
    assert_eq!(<Sgd as LearningRate>::learning_rate(&sgd), 0.02);
}

#[test]
fn step_lr_scheduler_updates_optimizer_lr_on_interval() {
    let mut optimizer = AdamW::new(0.1).unwrap();
    let mut scheduler = StepLr::new(2, 0.5).unwrap();

    let lr_epoch1 = scheduler.step(&mut optimizer).unwrap();
    assert!((lr_epoch1 - 0.1).abs() < 1e-8);
    let lr_epoch2 = scheduler.step(&mut optimizer).unwrap();
    assert!((lr_epoch2 - 0.05).abs() < 1e-8);
    let lr_epoch3 = scheduler.step(&mut optimizer).unwrap();
    assert!((lr_epoch3 - 0.05).abs() < 1e-8);
    let lr_epoch4 = scheduler.step(&mut optimizer).unwrap();
    assert!((lr_epoch4 - 0.025).abs() < 1e-8);
}

#[test]
fn step_lr_scheduler_rejects_invalid_configuration() {
    let err = StepLr::new(0, 0.5).unwrap_err();
    assert_eq!(err, OptimError::InvalidStepSize { step_size: 0 });

    let err = StepLr::new(1, 0.0).unwrap_err();
    assert_eq!(err, OptimError::InvalidStepGamma { gamma: 0.0 });
}

#[test]
fn cosine_scheduler_updates_optimizer_lr_until_floor() {
    let mut optimizer = AdamW::new(0.1).unwrap();
    let mut scheduler = CosineAnnealingLr::new(4, 0.01).unwrap();

    let lr1 = scheduler.step(&mut optimizer).unwrap();
    let lr2 = scheduler.step(&mut optimizer).unwrap();
    let lr3 = scheduler.step(&mut optimizer).unwrap();
    let lr4 = scheduler.step(&mut optimizer).unwrap();
    let lr5 = scheduler.step(&mut optimizer).unwrap();

    assert!(lr1 < 0.1 && lr1 > lr2);
    assert!(lr2 > lr3);
    assert!(lr3 > lr4);
    assert!((lr4 - 0.01).abs() < 1e-6);
    assert!((lr5 - 0.01).abs() < 1e-6);
}

#[test]
fn cosine_scheduler_rejects_invalid_configuration() {
    let err = CosineAnnealingLr::new(0, 0.0).unwrap_err();
    assert_eq!(err, OptimError::InvalidCosineTMax { t_max: 0 });

    let err = CosineAnnealingLr::new(4, 0.2)
        .unwrap()
        .with_base_lr(0.1)
        .unwrap_err();
    assert_eq!(
        err,
        OptimError::SchedulerMinLrExceedsBase {
            min_lr: 0.2,
            base_lr: 0.1
        }
    );
}

#[test]
fn cosine_scheduler_reset_restarts_epoch_counter() {
    let mut optimizer = AdamW::new(0.1).unwrap();
    let mut scheduler = CosineAnnealingLr::new(4, 0.0).unwrap();
    let _ = scheduler.step(&mut optimizer).unwrap();
    let _ = scheduler.step(&mut optimizer).unwrap();
    assert_eq!(scheduler.epoch(), 2);

    scheduler.reset();
    assert_eq!(scheduler.epoch(), 0);
}

#[test]
fn linear_warmup_scheduler_increases_lr_to_base() {
    let mut optimizer = AdamW::new(0.1).unwrap();
    let mut scheduler = LinearWarmupLr::new(4).unwrap().with_start_lr(0.0).unwrap();

    let lr1 = scheduler.step(&mut optimizer).unwrap();
    let lr2 = scheduler.step(&mut optimizer).unwrap();
    let lr3 = scheduler.step(&mut optimizer).unwrap();
    let lr4 = scheduler.step(&mut optimizer).unwrap();
    let lr5 = scheduler.step(&mut optimizer).unwrap();

    assert!((lr1 - 0.025).abs() < 1e-8);
    assert!((lr2 - 0.05).abs() < 1e-8);
    assert!((lr3 - 0.075).abs() < 1e-8);
    assert!((lr4 - 0.1).abs() < 1e-8);
    assert!((lr5 - 0.1).abs() < 1e-8);
}

#[test]
fn linear_warmup_scheduler_rejects_invalid_configuration() {
    let err = LinearWarmupLr::new(0).unwrap_err();
    assert_eq!(err, OptimError::InvalidWarmupSteps { warmup_steps: 0 });

    let err = LinearWarmupLr::new(2)
        .unwrap()
        .with_base_lr(0.01)
        .unwrap()
        .with_start_lr(0.02)
        .unwrap_err();
    assert_eq!(
        err,
        OptimError::SchedulerStartLrExceedsBase {
            start_lr: 0.02,
            base_lr: 0.01
        }
    );
}

#[test]
fn linear_warmup_scheduler_reset_restarts_epoch_counter() {
    let mut optimizer = AdamW::new(0.1).unwrap();
    let mut scheduler = LinearWarmupLr::new(3).unwrap();
    let _ = scheduler.step(&mut optimizer).unwrap();
    let _ = scheduler.step(&mut optimizer).unwrap();
    assert_eq!(scheduler.epoch(), 2);

    scheduler.reset();
    assert_eq!(scheduler.epoch(), 0);
}

#[test]
fn one_cycle_scheduler_updates_optimizer_lr_profile() {
    let mut optimizer = AdamW::new(0.01).unwrap();
    let mut scheduler = OneCycleLr::new(4, 0.1)
        .unwrap()
        .with_pct_start(0.5)
        .unwrap()
        .with_final_div_factor(10.0)
        .unwrap();

    let lr1 = scheduler.step(&mut optimizer).unwrap();
    let lr2 = scheduler.step(&mut optimizer).unwrap();
    let lr3 = scheduler.step(&mut optimizer).unwrap();
    let lr4 = scheduler.step(&mut optimizer).unwrap();
    let lr5 = scheduler.step(&mut optimizer).unwrap();

    assert!(lr1 > 0.01 && lr1 < 0.1);
    assert!((lr2 - 0.1).abs() < 1e-8);
    assert!(lr3 < lr2);
    assert!((lr4 - 0.001).abs() < 1e-8);
    assert!((lr5 - 0.001).abs() < 1e-8);
}

#[test]
fn one_cycle_scheduler_rejects_invalid_configuration() {
    let err = OneCycleLr::new(0, 0.1).unwrap_err();
    assert_eq!(
        err,
        OptimError::InvalidOneCycleTotalSteps { total_steps: 0 }
    );

    let err = OneCycleLr::new(10, 0.1)
        .unwrap()
        .with_pct_start(0.0)
        .unwrap_err();
    assert_eq!(err, OptimError::InvalidOneCyclePctStart { pct_start: 0.0 });

    let err = OneCycleLr::new(10, 0.1)
        .unwrap()
        .with_final_div_factor(1.0)
        .unwrap_err();
    assert_eq!(
        err,
        OptimError::InvalidOneCycleFinalDivFactor {
            final_div_factor: 1.0
        }
    );

    let err = OneCycleLr::new(10, 0.05)
        .unwrap()
        .with_initial_lr(0.1)
        .unwrap_err();
    assert_eq!(
        err,
        OptimError::SchedulerMaxLrBelowInitial {
            max_lr: 0.05,
            initial_lr: 0.1
        }
    );
}

#[test]
fn one_cycle_scheduler_reset_restarts_epoch_counter() {
    let mut optimizer = AdamW::new(0.01).unwrap();
    let mut scheduler = OneCycleLr::new(5, 0.1).unwrap();
    let _ = scheduler.step(&mut optimizer).unwrap();
    let _ = scheduler.step(&mut optimizer).unwrap();
    assert_eq!(scheduler.epoch(), 2);

    scheduler.reset();
    assert_eq!(scheduler.epoch(), 0);
}

// ---------------------------------------------------------------------------
// Gradient clipping tests
// ---------------------------------------------------------------------------

#[test]
fn clip_grad_norm_scales_gradients_when_total_norm_exceeds_max() {
    let mut graph = Graph::new();
    // Two variables: grad = [3, 4] => L2 norm = 5.
    let a = graph.variable(Tensor::from_vec(vec![1], vec![0.0]).unwrap());
    let b = graph.variable(Tensor::from_vec(vec![1], vec![0.0]).unwrap());
    let c = graph.constant(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());
    // Build a graph that yields grad(a)=3, grad(b)=4.
    // a * [3] + b * [4] => loss = 3a + 4b => grad(a)=3, grad(b)=4
    let sa = graph.select(c, 0, 0).unwrap();
    let sb = graph.select(c, 0, 1).unwrap();
    let t1 = graph.mul(a, sa).unwrap();
    let t2 = graph.mul(b, sb).unwrap();
    let s = graph.add(t1, t2).unwrap();
    let loss = graph.sum(s).unwrap();
    graph.backward(loss).unwrap();

    let ids = vec![a, b];
    let total_norm = clip_grad_norm_(&mut graph, &ids, 2.5, 2.0);
    assert_slice_approx_eq(&[total_norm], &[5.0], 1e-5);

    // After clipping: scale = 2.5 / 5.0 = 0.5
    let ga = graph.grad(a).unwrap().unwrap().data().to_vec();
    let gb = graph.grad(b).unwrap().unwrap().data().to_vec();
    assert_slice_approx_eq(&ga, &[1.5], 1e-5);
    assert_slice_approx_eq(&gb, &[2.0], 1e-5);
}

#[test]
fn clip_grad_norm_does_not_scale_when_below_max() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let loss = graph.sum(a).unwrap();
    graph.backward(loss).unwrap();

    // grad(a) = [1, 1], L2 norm = sqrt(2) ~ 1.414
    let ids = vec![a];
    let total_norm = clip_grad_norm_(&mut graph, &ids, 10.0, 2.0);
    assert!(total_norm < 10.0);

    // Gradients should be unchanged.
    let ga = graph.grad(a).unwrap().unwrap().data().to_vec();
    assert_slice_approx_eq(&ga, &[1.0, 1.0], 1e-6);
}

#[test]
fn clip_grad_norm_returns_zero_for_empty_ids() {
    let mut graph = Graph::new();
    let total = clip_grad_norm_(&mut graph, &[], 1.0, 2.0);
    assert_eq!(total, 0.0);
}

#[test]
fn clip_grad_norm_skips_nodes_without_gradients() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    // Do NOT call backward, so a has no gradient.
    let ids = vec![a];
    let total = clip_grad_norm_(&mut graph, &ids, 1.0, 2.0);
    assert_eq!(total, 0.0);
}

#[test]
fn clip_grad_norm_handles_non_positive_max_norm() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let loss = graph.sum(a).unwrap();
    graph.backward(loss).unwrap();

    let ids = vec![a];
    let total = clip_grad_norm_(&mut graph, &ids, 0.0, 2.0);
    assert_eq!(total, 0.0);

    let total = clip_grad_norm_(&mut graph, &ids, -1.0, 2.0);
    assert_eq!(total, 0.0);
}

#[test]
fn clip_grad_value_clamps_gradient_elements() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let c = graph.constant(Tensor::from_vec(vec![2], vec![10.0, -5.0]).unwrap());
    let prod = graph.mul(a, c).unwrap();
    let loss = graph.sum(prod).unwrap();
    graph.backward(loss).unwrap();

    // grad(a) = [10, -5]
    let ids = vec![a];
    clip_grad_value_(&mut graph, &ids, 3.0);

    let ga = graph.grad(a).unwrap().unwrap().data().to_vec();
    assert_slice_approx_eq(&ga, &[3.0, -3.0], 1e-6);
}

#[test]
fn clip_grad_value_does_nothing_for_empty_ids() {
    let mut graph = Graph::new();
    clip_grad_value_(&mut graph, &[], 1.0);
}

#[test]
fn clip_grad_value_skips_nodes_without_gradients() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let ids = vec![a];
    // No backward called, no gradient exists - should not panic.
    clip_grad_value_(&mut graph, &ids, 1.0);
}

#[test]
fn clip_grad_value_handles_non_positive_max_val() {
    let mut graph = Graph::new();
    let a = graph.variable(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
    let loss = graph.sum(a).unwrap();
    graph.backward(loss).unwrap();

    let ids = vec![a];
    clip_grad_value_(&mut graph, &ids, 0.0);
    // Gradients should be unchanged since max_val <= 0 is a no-op.
    let ga = graph.grad(a).unwrap().unwrap().data().to_vec();
    assert_slice_approx_eq(&ga, &[1.0, 1.0], 1e-6);
}

// ---- ExponentialLr tests ----

#[test]
fn exponential_lr_decays_every_step() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = ExponentialLr::new(0.5).unwrap();
    let lr1 = sched.step(&mut opt).unwrap();
    assert!((lr1 - 0.5).abs() < 1e-6);
    let lr2 = sched.step(&mut opt).unwrap();
    assert!((lr2 - 0.25).abs() < 1e-6);
    let lr3 = sched.step(&mut opt).unwrap();
    assert!((lr3 - 0.125).abs() < 1e-6);
    assert_eq!(LrScheduler::epoch(&sched), 3);
}

#[test]
fn exponential_lr_reset_restarts_epoch() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = ExponentialLr::new(0.9).unwrap();
    sched.step(&mut opt).unwrap();
    LrScheduler::reset(&mut sched);
    assert_eq!(LrScheduler::epoch(&sched), 0);
}

// ---- PolynomialDecayLr tests ----

#[test]
fn polynomial_decay_lr_reaches_end_lr() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = PolynomialDecayLr::new(10, 1.0, 0.0).unwrap();
    let mut lr = 0.0;
    for _ in 0..10 {
        lr = sched.step(&mut opt).unwrap();
    }
    assert!(lr.abs() < 1e-6, "expected ~0.0 at end, got {lr}");
}

#[test]
fn polynomial_decay_lr_power_2() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = PolynomialDecayLr::new(4, 2.0, 0.0).unwrap();
    // After 2 steps: (1-2/4)^2 = 0.25, lr = 1.0 * 0.25 = 0.25
    sched.step(&mut opt).unwrap();
    let lr2 = sched.step(&mut opt).unwrap();
    assert!((lr2 - 0.25).abs() < 1e-5, "got {lr2}");
}

#[test]
fn polynomial_decay_lr_clamps_after_total_steps() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = PolynomialDecayLr::new(5, 1.0, 0.1).unwrap();
    for _ in 0..10 {
        sched.step(&mut opt).unwrap();
    }
    let lr = opt.learning_rate();
    assert!((lr - 0.1).abs() < 1e-6, "expected 0.1, got {lr}");
}

// ---- ReduceLrOnPlateau tests ----

#[test]
fn reduce_lr_on_plateau_reduces_after_patience() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = ReduceLrOnPlateau::new(0.5, 2, 0.0).unwrap();
    // First call: metric=1.0 -> best, wait=0
    sched.step_with_metric(1.0, &mut opt).unwrap();
    assert!((opt.learning_rate() - 1.0).abs() < 1e-6);
    // No improvement
    sched.step_with_metric(2.0, &mut opt).unwrap();
    assert!((opt.learning_rate() - 1.0).abs() < 1e-6); // wait=1
    sched.step_with_metric(2.0, &mut opt).unwrap();
    // wait=2 >= patience=2, reduce
    assert!((opt.learning_rate() - 0.5).abs() < 1e-6);
}

#[test]
fn reduce_lr_on_plateau_respects_min_lr() {
    let mut opt = Sgd::new(0.01).unwrap();
    let mut sched = ReduceLrOnPlateau::new(0.1, 1, 0.005).unwrap();
    sched.step_with_metric(1.0, &mut opt).unwrap();
    // No improvement -> reduce after patience=1
    sched.step_with_metric(2.0, &mut opt).unwrap();
    let lr = opt.learning_rate();
    assert!((lr - 0.005).abs() < 1e-6, "expected min_lr 0.005, got {lr}");
}

#[test]
fn reduce_lr_on_plateau_reset_clears_state() {
    let mut sched = ReduceLrOnPlateau::new(0.5, 2, 0.0).unwrap();
    let mut opt = Sgd::new(1.0).unwrap();
    sched.step_with_metric(0.5, &mut opt).unwrap();
    LrScheduler::reset(&mut sched);
    assert_eq!(LrScheduler::epoch(&sched), 0);
    assert_eq!(sched.wait(), 0);
    assert_eq!(sched.best_metric(), f32::INFINITY);
}

// ---- CyclicLr tests ----

#[test]
fn cyclic_lr_triangular_cycle() {
    let mut opt = Sgd::new(0.0).unwrap();
    let mut sched = CyclicLr::new(0.0, 1.0, 4, 4).unwrap();
    let mut lrs = Vec::new();
    for _ in 0..8 {
        lrs.push(sched.step(&mut opt).unwrap());
    }
    // Ascending: 0/4=0.0, 1/4=0.25, 2/4=0.5, 3/4=0.75
    // Descending: 1.0-0/4=1.0, 1.0-0.25=0.75, 1.0-0.5=0.5, 1.0-0.75=0.25
    assert!((lrs[0] - 0.0).abs() < 1e-6);
    assert!((lrs[1] - 0.25).abs() < 1e-6);
    assert!((lrs[2] - 0.5).abs() < 1e-6);
    assert!((lrs[3] - 0.75).abs() < 1e-6);
    assert!((lrs[4] - 1.0).abs() < 1e-6);
    assert!((lrs[5] - 0.75).abs() < 1e-6);
    assert!((lrs[6] - 0.5).abs() < 1e-6);
    assert!((lrs[7] - 0.25).abs() < 1e-6);
}

#[test]
fn cyclic_lr_repeats_cycle() {
    let mut opt = Sgd::new(0.0).unwrap();
    let mut sched = CyclicLr::new(0.0, 1.0, 2, 2).unwrap();
    // First cycle
    let a = sched.step(&mut opt).unwrap();
    let _b = sched.step(&mut opt).unwrap();
    let _c = sched.step(&mut opt).unwrap();
    let _d = sched.step(&mut opt).unwrap();
    // Second cycle should repeat
    let a2 = sched.step(&mut opt).unwrap();
    assert!((a - a2).abs() < 1e-6);
}

// ---- MultiStepLr tests ----

#[test]
fn multi_step_lr_drops_at_milestones() {
    let mut opt = Sgd::new(0.1).unwrap();
    let mut sched = MultiStepLr::new(vec![5, 10], 0.1).unwrap();

    // Epochs 1-4: LR should remain 0.1 (no milestone reached yet)
    for _ in 0..4 {
        let lr = sched.step(&mut opt).unwrap();
        assert!((lr - 0.1).abs() < 1e-6, "expected 0.1, got {lr}");
    }
    // Epoch 5: first milestone, LR = 0.1 * 0.1 = 0.01
    let lr5 = sched.step(&mut opt).unwrap();
    assert!((lr5 - 0.01).abs() < 1e-6, "expected 0.01, got {lr5}");
    // Epochs 6-9: LR stays at 0.01
    for _ in 6..10 {
        let lr = sched.step(&mut opt).unwrap();
        assert!((lr - 0.01).abs() < 1e-6, "expected 0.01, got {lr}");
    }
    // Epoch 10: second milestone, LR = 0.1 * 0.1^2 = 0.001
    let lr10 = sched.step(&mut opt).unwrap();
    assert!((lr10 - 0.001).abs() < 1e-6, "expected 0.001, got {lr10}");
    // Epoch 11+: LR stays at 0.001
    let lr11 = sched.step(&mut opt).unwrap();
    assert!((lr11 - 0.001).abs() < 1e-6, "expected 0.001, got {lr11}");
}

#[test]
fn multi_step_lr_reset_restarts_epoch_and_base_lr() {
    let mut opt = Sgd::new(0.1).unwrap();
    let mut sched = MultiStepLr::new(vec![2], 0.5).unwrap();
    // Step past the milestone
    sched.step(&mut opt).unwrap();
    sched.step(&mut opt).unwrap();
    assert_eq!(sched.epoch(), 2);

    sched.reset();
    assert_eq!(sched.epoch(), 0);

    // After reset with a new optimizer LR, the schedule should use the new base LR
    opt.set_learning_rate(0.2).unwrap();
    let lr1 = sched.step(&mut opt).unwrap();
    assert!((lr1 - 0.2).abs() < 1e-6, "expected 0.2, got {lr1}");
    let lr2 = sched.step(&mut opt).unwrap();
    assert!((lr2 - 0.1).abs() < 1e-6, "expected 0.1, got {lr2}");
}

#[test]
fn multi_step_lr_single_milestone() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = MultiStepLr::new(vec![3], 0.5).unwrap();

    let lr1 = sched.step(&mut opt).unwrap();
    assert!((lr1 - 1.0).abs() < 1e-6);
    let lr2 = sched.step(&mut opt).unwrap();
    assert!((lr2 - 1.0).abs() < 1e-6);
    // Epoch 3: milestone hit, LR = 1.0 * 0.5 = 0.5
    let lr3 = sched.step(&mut opt).unwrap();
    assert!((lr3 - 0.5).abs() < 1e-6, "expected 0.5, got {lr3}");
    // Stays at 0.5 after
    let lr4 = sched.step(&mut opt).unwrap();
    assert!((lr4 - 0.5).abs() < 1e-6, "expected 0.5, got {lr4}");
    let lr5 = sched.step(&mut opt).unwrap();
    assert!((lr5 - 0.5).abs() < 1e-6, "expected 0.5, got {lr5}");
}

// ---- LambdaLr tests ----

#[test]
fn lambda_lr_identity() {
    let mut opt = Sgd::new(0.1).unwrap();
    let mut sched = LambdaLr::new(0.1, Box::new(|_epoch| 1.0));
    for _ in 0..5 {
        let lr = sched.step(&mut opt).unwrap();
        assert!((lr - 0.1).abs() < 1e-6, "expected 0.1, got {lr}");
    }
}

#[test]
fn lambda_lr_decay() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = LambdaLr::new(1.0, Box::new(|epoch| 0.5_f32.powi(epoch as i32)));
    let lr1 = sched.step(&mut opt).unwrap();
    assert!((lr1 - 0.5).abs() < 1e-6, "expected 0.5, got {lr1}");
    let lr2 = sched.step(&mut opt).unwrap();
    assert!((lr2 - 0.25).abs() < 1e-6, "expected 0.25, got {lr2}");
    let lr3 = sched.step(&mut opt).unwrap();
    assert!((lr3 - 0.125).abs() < 1e-6, "expected 0.125, got {lr3}");
}

#[test]
fn lambda_lr_reset() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = LambdaLr::new(1.0, Box::new(|epoch| 0.5_f32.powi(epoch as i32)));
    sched.step(&mut opt).unwrap();
    sched.step(&mut opt).unwrap();
    assert_eq!(sched.epoch(), 2);

    sched.reset();
    assert_eq!(sched.epoch(), 0);
    assert!((sched.current_lr() - 1.0).abs() < 1e-6);

    // After reset, stepping should produce the same sequence again
    let lr1 = sched.step(&mut opt).unwrap();
    assert!(
        (lr1 - 0.5).abs() < 1e-6,
        "expected 0.5 after reset, got {lr1}"
    );
}

// ---------------------------------------------------------------------------
// Adagrad tests
// ---------------------------------------------------------------------------

#[test]
fn adagrad_basic_step() {
    let mut optimizer = Adagrad::new(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
    let original = weights.data().to_vec();

    optimizer.step(40, &mut weights, &grad).unwrap();
    assert_ne!(weights.data(), original.as_slice());
}

#[test]
fn adagrad_accumulates_squared_grads() {
    let mut optimizer = Adagrad::new(0.5).unwrap();
    let mut weights = Tensor::from_vec(vec![1], vec![10.0]).unwrap();
    let grad = Tensor::from_vec(vec![1], vec![1.0]).unwrap();

    // First step: sum_sq = 1, update = 0.5 * 1 / (1 + eps) ~ 0.5
    optimizer.step(41, &mut weights, &grad).unwrap();
    let after_first = weights.data()[0];

    // Second step: sum_sq = 2, update = 0.5 * 1 / (sqrt(2) + eps) ~ 0.354
    optimizer.step(41, &mut weights, &grad).unwrap();
    let after_second = weights.data()[0];

    let delta_first = (10.0 - after_first).abs();
    let delta_second = (after_first - after_second).abs();

    // The effective step size should decrease over time.
    assert!(
        delta_second < delta_first,
        "effective lr should decrease: delta_first={delta_first}, delta_second={delta_second}"
    );
}

#[test]
fn adagrad_with_weight_decay() {
    let mut optimizer = Adagrad::new(0.1).unwrap().with_weight_decay(0.5).unwrap();
    let mut weights_wd = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let mut weights_no = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.1, 0.1]).unwrap();

    let grad2 = Tensor::from_vec(vec![2], vec![0.1, 0.1]).unwrap();
    optimizer.step(42, &mut weights_wd, &grad).unwrap();
    optimizer.step(42, &mut weights_wd, &grad2).unwrap();

    let mut optimizer_no = Adagrad::new(0.1).unwrap();
    optimizer_no.step(43, &mut weights_no, &grad).unwrap();
    optimizer_no.step(43, &mut weights_no, &grad2).unwrap();

    // After two steps the accumulated sum_sq differs, so results diverge.
    assert_ne!(weights_wd.data(), weights_no.data());
}

#[test]
fn adagrad_rejects_shape_mismatch() {
    let mut optimizer = Adagrad::new(0.1).unwrap();
    let mut weights = Tensor::zeros(vec![2]).unwrap();
    let grad = Tensor::zeros(vec![3]).unwrap();

    let err = optimizer.step(0, &mut weights, &grad).unwrap_err();
    assert_eq!(
        err,
        OptimError::ShapeMismatch {
            weights: vec![2],
            grad: vec![3]
        }
    );
}

// ---------------------------------------------------------------------------
// RAdam tests
// ---------------------------------------------------------------------------

#[test]
fn radam_basic_step() {
    let mut optimizer = RAdam::new(0.1).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
    let original = weights.data().to_vec();

    optimizer.step(50, &mut weights, &grad).unwrap();
    assert_ne!(weights.data(), original.as_slice());
}

#[test]
fn radam_early_steps_use_momentum() {
    // With default beta2=0.999, rho_inf ~ 999, and at step 1:
    // rho_t = 999 - 2 * 1 * 0.999 / (1 - 0.999) = 999 - 1998 = -999
    // which is <= 5, so SGD-like update is used (no adaptive denominator).
    let mut opt_radam = RAdam::new(0.01).unwrap();
    let mut opt_adam = Adam::new(0.01).unwrap();

    let mut w_radam = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let mut w_adam = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();

    // After a single step, RAdam uses m_hat only (no sqrt(v_hat)),
    // so it should differ from Adam which always uses adaptive update.
    opt_radam.step(51, &mut w_radam, &grad).unwrap();
    opt_adam.step(52, &mut w_adam, &grad).unwrap();

    assert_ne!(
        w_radam.data(),
        w_adam.data(),
        "early RAdam step should differ from Adam"
    );
}

#[test]
fn radam_converges_to_adam() {
    // After enough steps rho_t > 5, and r_t -> 1, so RAdam behaves like Adam.
    let mut opt_radam = RAdam::new(0.001).unwrap();
    let mut opt_adam = Adam::new(0.001).unwrap();

    let mut w_radam = Tensor::from_vec(vec![1], vec![5.0]).unwrap();
    let mut w_adam = Tensor::from_vec(vec![1], vec![5.0]).unwrap();

    // Run many steps with a constant gradient so both converge.
    for _i in 0..500 {
        let grad = Tensor::from_vec(vec![1], vec![0.1]).unwrap();
        opt_radam.step(53, &mut w_radam, &grad).unwrap();
        opt_adam.step(54, &mut w_adam, &grad).unwrap();
    }

    // After many steps, the rectification term r_t approaches 1,
    // so RAdam and Adam should produce similar (though not identical) results.
    let diff = (w_radam.data()[0] - w_adam.data()[0]).abs();
    assert!(
        diff < 0.5,
        "after many steps RAdam should be close to Adam, diff={diff}"
    );
}

#[test]
fn radam_rejects_invalid_lr() {
    let err = RAdam::new(-0.1).unwrap_err();
    assert_eq!(err, OptimError::InvalidLearningRate { lr: -0.1 });
}

// ---------------------------------------------------------------------------
// CosineAnnealingWarmRestarts tests
// ---------------------------------------------------------------------------

#[test]
fn cosine_warm_restarts_basic() {
    // t_0=5, t_mult=1, eta_min=0.0, base_lr=1.0
    // LR should decay over 5 epochs, then restart
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = CosineAnnealingWarmRestarts::new(5, 1, 0.0).unwrap();

    let mut lrs = Vec::new();
    for _ in 0..10 {
        lrs.push(sched.step(&mut opt).unwrap());
    }

    // At epoch 5 (end of first period), LR should be eta_min = 0.0
    assert!(
        lrs[4].abs() < 1e-6,
        "expected ~0.0 at end of period, got {}",
        lrs[4]
    );

    // At epoch 6 (start of second period), LR should restart high
    // epoch 6: t_cur=1 in period of 5 => cos(pi*1/5) ~ 0.809
    // lr = 0 + 0.5 * 1.0 * (1 + 0.809) ~ 0.905
    assert!(
        lrs[5] > 0.8,
        "expected LR to restart high at epoch 6, got {}",
        lrs[5]
    );

    // At epoch 10 (end of second period), LR should be eta_min again
    assert!(
        lrs[9].abs() < 1e-6,
        "expected ~0.0 at end of second period, got {}",
        lrs[9]
    );
}

#[test]
fn cosine_warm_restarts_t_mult() {
    // t_0=3, t_mult=2, eta_min=0.0, base_lr=1.0
    // First period: 3 epochs (1-3)
    // Second period: 6 epochs (4-9)
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = CosineAnnealingWarmRestarts::new(3, 2, 0.0).unwrap();

    let mut lrs = Vec::new();
    for _ in 0..9 {
        lrs.push(sched.step(&mut opt).unwrap());
    }

    // End of first period (epoch 3): LR should be ~0
    assert!(
        lrs[2].abs() < 1e-6,
        "expected ~0.0 at end of first period, got {}",
        lrs[2]
    );

    // Start of second period (epoch 4): LR restarts high
    assert!(
        lrs[3] > 0.5,
        "expected LR to restart high at epoch 4, got {}",
        lrs[3]
    );

    // End of second period (epoch 9): LR should be ~0
    assert!(
        lrs[8].abs() < 1e-6,
        "expected ~0.0 at end of second period, got {}",
        lrs[8]
    );

    // Mid second period (epoch 6 = t_cur=3 of 6): cos(pi*3/6) = cos(pi/2) = 0
    // lr = 0.5 * 1.0 * (1 + 0) = 0.5
    assert!(
        (lrs[5] - 0.5).abs() < 1e-6,
        "expected 0.5 at mid second period, got {}",
        lrs[5]
    );
}

#[test]
fn cosine_warm_restarts_reset() {
    let mut opt = Sgd::new(1.0).unwrap();
    let mut sched = CosineAnnealingWarmRestarts::new(5, 1, 0.0).unwrap();

    let lr1 = sched.step(&mut opt).unwrap();
    sched.step(&mut opt).unwrap();
    sched.step(&mut opt).unwrap();
    assert_eq!(sched.epoch(), 3);

    sched.reset();
    assert_eq!(sched.epoch(), 0);

    // After reset, stepping should produce the same LR as first step
    // Need to reset optimizer LR too
    opt.set_learning_rate(1.0).unwrap();
    let lr1_again = sched.step(&mut opt).unwrap();
    assert!(
        (lr1 - lr1_again).abs() < 1e-6,
        "expected same LR after reset, got {lr1} vs {lr1_again}"
    );
}

// ---------------------------------------------------------------------------
// Lookahead tests
// ---------------------------------------------------------------------------

#[test]
fn lookahead_sgd_basic() {
    let sgd = Sgd::new(0.1).unwrap();
    let mut optimizer = Lookahead::new(sgd, 0.5, 5);
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let original = weights.data().to_vec();

    for _ in 0..10 {
        let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
        optimizer.step(0, &mut weights, &grad).unwrap();
    }

    assert_ne!(
        weights.data(),
        original.as_slice(),
        "weights should have changed after 10 steps"
    );
}

#[test]
fn lookahead_sync_period() {
    let sgd = Sgd::new(0.1).unwrap();
    let k = 5;
    let mut optimizer = Lookahead::new(sgd, 0.5, k);
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();

    // Run k-1 steps
    for _ in 0..k - 1 {
        let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
        optimizer.step(0, &mut weights, &grad).unwrap();
    }
    let before_sync = weights.data().to_vec();

    // The k-th step triggers slow weight sync
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
    optimizer.step(0, &mut weights, &grad).unwrap();
    let after_sync = weights.data().to_vec();

    // After sync, weights are pulled back toward slow weights,
    // so they should differ from what a pure SGD continuation would give.
    assert_ne!(
        before_sync, after_sync,
        "weights should change at the sync step"
    );

    // After sync, weights should be between original and fast weights,
    // i.e. pulled back toward initial slow weights.
    // With alpha=0.5, slow = init + 0.5*(fast - init) = midpoint,
    // and fast is set to slow, so weights should be closer to initial.
    // Specifically, the magnitude of change should be less than k pure SGD steps.
    let pure_sgd_delta_0 = 0.1 * 0.5 * (k as f32); // 0.25
    let actual_delta_0 = (1.0 - after_sync[0]).abs();
    assert!(
        actual_delta_0 < pure_sgd_delta_0,
        "sync should pull weights back: actual_delta={actual_delta_0}, pure_sgd_delta={pure_sgd_delta_0}"
    );
}

#[test]
fn lookahead_adam_basic() {
    let adam = Adam::new(0.1).unwrap();
    let mut optimizer = Lookahead::new(adam, 0.5, 5);
    let mut weights = Tensor::from_vec(vec![1], vec![5.0]).unwrap();

    // Minimise f(w) = w^2 / 2  =>  grad = w
    for _ in 0..500 {
        let w = weights.data()[0];
        let grad = Tensor::from_vec(vec![1], vec![w]).unwrap();
        optimizer.step(0, &mut weights, &grad).unwrap();
    }

    let final_w = weights.data()[0].abs();
    assert!(
        final_w < 1.0,
        "expected convergence toward 0, got {final_w}"
    );
}

// ---------------------------------------------------------------------------
// LARS tests
// ---------------------------------------------------------------------------

#[test]
fn lars_basic_step() {
    let mut optimizer = Lars::new(0.01).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
    let original = weights.data().to_vec();

    optimizer.step(60, &mut weights, &grad).unwrap();
    assert_ne!(weights.data(), original.as_slice());
}

#[test]
fn lars_with_weight_decay() {
    let mut opt_wd = Lars::new(0.01).unwrap().with_weight_decay(0.1).unwrap();
    let mut opt_no = Lars::new(0.01).unwrap();

    let mut weights_wd = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let mut weights_no = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
    let grad2 = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();

    opt_wd.step(61, &mut weights_wd, &grad).unwrap();
    opt_no.step(62, &mut weights_no, &grad2).unwrap();

    assert_ne!(
        weights_wd.data(),
        weights_no.data(),
        "weight decay should produce different updates"
    );
}

// ---------------------------------------------------------------------------
// LAMB tests
// ---------------------------------------------------------------------------

#[test]
fn lamb_basic_step() {
    let mut optimizer = Lamb::new(0.01).unwrap();
    let mut weights = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let grad = Tensor::from_vec(vec![2], vec![0.5, -1.0]).unwrap();
    let original = weights.data().to_vec();

    optimizer.step(70, &mut weights, &grad).unwrap();
    assert_ne!(weights.data(), original.as_slice());
}

#[test]
fn lamb_trust_ratio() {
    // With weight decay, the trust ratio scales the update by ||w|| / ||adam_step||.
    // Compare the update magnitude with large vs small weight norms.
    let mut opt_large = Lamb::new(0.01).unwrap().with_weight_decay(0.1).unwrap();
    let mut opt_small = Lamb::new(0.01).unwrap().with_weight_decay(0.1).unwrap();

    let mut weights_large = Tensor::from_vec(vec![2], vec![10.0, 20.0]).unwrap();
    let mut weights_small = Tensor::from_vec(vec![2], vec![0.1, 0.2]).unwrap();
    let grad_large = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();
    let grad_small = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();

    let large_before = weights_large.data().to_vec();
    let small_before = weights_small.data().to_vec();

    opt_large.step(71, &mut weights_large, &grad_large).unwrap();
    opt_small.step(72, &mut weights_small, &grad_small).unwrap();

    // Compute the magnitude of weight change for each.
    let delta_large: f32 = weights_large
        .data()
        .iter()
        .zip(large_before.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let delta_small: f32 = weights_small
        .data()
        .iter()
        .zip(small_before.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    // Larger weight norm should produce a larger absolute update due to trust ratio scaling.
    assert!(
        delta_large > delta_small,
        "trust ratio should scale update with weight norm: delta_large={delta_large}, delta_small={delta_small}"
    );
}
