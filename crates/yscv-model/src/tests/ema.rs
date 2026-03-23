use yscv_tensor::Tensor;

use crate::ExponentialMovingAverage;

#[test]
fn ema_shadow_moves_toward_params() {
    let initial = vec![Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).unwrap()];
    let target = vec![Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()];

    let mut ema = ExponentialMovingAverage::new(0.9);
    ema.register(&initial);

    ema.update(&target);

    // After one update: shadow = 0.9 * 0 + 0.1 * target = 0.1 * target
    let shadow = ema.shadow_params();
    let s = shadow[0].data();

    // Shadow should now be closer to target than the initial zeros were.
    for (i, &val) in s.iter().enumerate() {
        let t = target[0].data()[i];
        assert!(
            val.abs() < t.abs(),
            "shadow[{i}]={val} should be between 0 and target={t}",
        );
        assert!(val > 0.0, "shadow[{i}] should be positive after update");
    }

    // Exact check
    super::assert_slice_approx_eq(s, &[0.1, 0.2, 0.3], 1e-6);
}

#[test]
fn ema_apply_shadow_replaces() {
    let params_data = vec![1.0, 2.0];
    let shadow_data = vec![10.0, 20.0];

    let mut ema = ExponentialMovingAverage::new(0.99);
    ema.register(&[Tensor::from_vec(vec![2], shadow_data.clone()).unwrap()]);

    let mut params = vec![Tensor::from_vec(vec![2], params_data).unwrap()];
    ema.apply_shadow(&mut params);

    super::assert_slice_approx_eq(params[0].data(), &shadow_data, 1e-6);
}

#[test]
fn ema_register_matches_count() {
    let mut ema = ExponentialMovingAverage::new(0.999);
    assert_eq!(ema.shadow_params().len(), 0);

    let params = vec![
        Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![3], vec![3.0, 4.0, 5.0]).unwrap(),
        Tensor::from_vec(vec![1], vec![6.0]).unwrap(),
    ];
    ema.register(&params);

    assert_eq!(ema.shadow_params().len(), 3);
}

#[test]
fn ema_num_updates_increments() {
    let params = vec![Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap()];

    let mut ema = ExponentialMovingAverage::new(0.9);
    ema.register(&params);

    assert_eq!(ema.num_updates(), 0);
    ema.update(&params);
    assert_eq!(ema.num_updates(), 1);
    ema.update(&params);
    assert_eq!(ema.num_updates(), 2);
    ema.update(&params);
    assert_eq!(ema.num_updates(), 3);
}
