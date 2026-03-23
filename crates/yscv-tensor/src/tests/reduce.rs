use crate::Tensor;

#[test]
fn sum_reduces_all_elements() {
    let tensor = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(tensor.sum(), 10.0);
}

#[test]
fn sum_axis_reduces_requested_axis() {
    let tensor = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let axis0 = tensor.sum_axis(0).unwrap();
    assert_eq!(axis0.shape(), &[3]);
    assert_eq!(axis0.data(), &[5.0, 7.0, 9.0]);

    let axis1 = tensor.sum_axis(1).unwrap();
    assert_eq!(axis1.shape(), &[2]);
    assert_eq!(axis1.data(), &[6.0, 15.0]);
}

#[test]
fn sum_axis_rejects_invalid_axis() {
    use crate::TensorError;
    let tensor = Tensor::zeros(vec![2, 3]).unwrap();
    let err = tensor.sum_axis(2).unwrap_err();
    assert_eq!(err, TensorError::InvalidAxis { axis: 2, rank: 2 });
}

#[test]
fn mean_reduces_all_elements() {
    let t = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(t.mean(), 2.5);
}

#[test]
fn mean_empty_returns_nan() {
    let t = Tensor::from_vec(vec![0], vec![]).unwrap();
    assert!(t.mean().is_nan());
}

#[test]
fn max_value_and_min_value() {
    let t = Tensor::from_vec(vec![4], vec![3.0, 1.0, 4.0, 2.0]).unwrap();
    assert_eq!(t.max_value(), 4.0);
    assert_eq!(t.min_value(), 1.0);
}

#[test]
fn argmax_and_argmin() {
    let t = Tensor::from_vec(vec![4], vec![3.0, 1.0, 4.0, 2.0]).unwrap();
    assert_eq!(t.argmax(), Some(2));
    assert_eq!(t.argmin(), Some(1));
}

#[test]
fn argmax_empty_returns_none() {
    let t = Tensor::from_vec(vec![0], vec![]).unwrap();
    assert_eq!(t.argmax(), None);
    assert_eq!(t.argmin(), None);
}

#[test]
fn var_and_std_dev() {
    let t = Tensor::from_vec(vec![4], vec![2.0, 4.0, 4.0, 4.0]).unwrap();
    assert!((t.mean() - 3.5).abs() < 1e-5);
    assert!((t.var() - 0.75).abs() < 1e-5);
    assert!((t.std_dev() - 0.75_f32.sqrt()).abs() < 1e-5);
}

#[test]
fn mean_axis_reduces_requested_axis() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let m0 = t.mean_axis(0).unwrap();
    assert_eq!(m0.shape(), &[3]);
    assert_eq!(m0.data(), &[2.5, 3.5, 4.5]);

    let m1 = t.mean_axis(1).unwrap();
    assert_eq!(m1.shape(), &[2]);
    assert_eq!(m1.data(), &[2.0, 5.0]);
}

#[test]
fn max_axis_and_min_axis() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]).unwrap();
    let mx0 = t.max_axis(0).unwrap();
    assert_eq!(mx0.data(), &[4.0, 5.0, 6.0]);
    let mn1 = t.min_axis(1).unwrap();
    assert_eq!(mn1.data(), &[1.0, 2.0]);
}

#[test]
fn var_axis_computes_population_variance() {
    let t = Tensor::from_vec(vec![2, 2], vec![1.0, 3.0, 5.0, 7.0]).unwrap();
    let v = t.var_axis(0).unwrap();
    assert_eq!(v.shape(), &[2]);
    assert!((v.data()[0] - 4.0).abs() < 1e-5);
    assert!((v.data()[1] - 4.0).abs() < 1e-5);
}

#[test]
fn median_odd_count() {
    let t = Tensor::from_vec(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
    assert_eq!(t.median(), 3.0);
}

#[test]
fn median_even_count() {
    let t = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(t.median(), 2.5);
}

#[test]
fn median_axis_removes_dim() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let m = t.median_axis(1).unwrap();
    assert_eq!(m.shape(), &[2]);
    assert_eq!(m.data()[0], 2.0); // median of [1,2,3]
    assert_eq!(m.data()[1], 5.0); // median of [4,5,6]
}

#[test]
fn any_with_zeros_and_nonzeros() {
    let t1 = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
    assert!(!t1.any());
    let t2 = Tensor::from_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap();
    assert!(t2.any());
}

#[test]
fn all_with_zeros_and_nonzeros() {
    let t1 = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert!(t1.all());
    let t2 = Tensor::from_vec(vec![3], vec![1.0, 0.0, 3.0]).unwrap();
    assert!(!t2.all());
}

#[test]
fn quantile_matches_median() {
    let t = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert!((t.quantile(0.5) - t.median()).abs() < 1e-6);
}

#[test]
fn quantile_extremes() {
    let t = Tensor::from_vec(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    assert!((t.quantile(0.0) - 10.0).abs() < 1e-6);
    assert!((t.quantile(1.0) - 50.0).abs() < 1e-6);
}

#[test]
fn quantile_interpolation() {
    let t = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let q25 = t.quantile(0.25);
    assert!((q25 - 1.75).abs() < 1e-5);
}
