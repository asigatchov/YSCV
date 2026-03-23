use crate::{Tensor, TensorError};

// ── binary arithmetic ───────────────────────────────────────────────

#[test]
fn add_supports_standard_broadcasting() {
    let left = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let out = left.add(&right).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn add_supports_scalar_broadcasting() {
    let left = Tensor::scalar(2.0);
    let right = Tensor::from_vec(vec![2], vec![1.0, 3.0]).unwrap();
    let out = left.add(&right).unwrap();

    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.data(), &[3.0, 5.0]);
}

#[test]
fn add_rejects_incompatible_broadcasting() {
    let left = Tensor::zeros(vec![2, 3]).unwrap();
    let right = Tensor::zeros(vec![2, 2]).unwrap();
    let err = left.add(&right).unwrap_err();
    assert_eq!(
        err,
        TensorError::BroadcastIncompatible {
            left: vec![2, 3],
            right: vec![2, 2]
        }
    );
}

#[test]
fn add_handles_zero_sized_dimensions() {
    let left = Tensor::zeros(vec![2, 0, 3]).unwrap();
    let right = Tensor::zeros(vec![1, 0, 1]).unwrap();
    let out = left.add(&right).unwrap();
    assert_eq!(out.shape(), &[2, 0, 3]);
    assert_eq!(out.len(), 0);
    assert!(out.is_empty());
}

#[test]
fn sub_supports_standard_broadcasting() {
    let left = Tensor::from_vec(vec![2, 3], vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]).unwrap();
    let right = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let out = left.sub(&right).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn mul_supports_scalar_broadcasting() {
    let left = Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap();
    let right = Tensor::scalar(4.0);
    let out = left.mul(&right).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.data(), &[8.0, 12.0]);
}

#[test]
fn div_elementwise_same_shape() {
    let a = Tensor::from_vec(vec![3], vec![6.0, 10.0, 15.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![2.0, 5.0, 3.0]).unwrap();
    let out = a.div(&b).unwrap();
    assert_eq!(out.data(), &[3.0, 2.0, 5.0]);
}

#[test]
fn div_broadcasts_scalar_divisor() {
    let a = Tensor::from_vec(vec![3], vec![4.0, 8.0, 12.0]).unwrap();
    let b = Tensor::scalar(4.0);
    let out = a.div(&b).unwrap();
    assert_eq!(out.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn pow_elementwise() {
    let base = Tensor::from_vec(vec![3], vec![2.0, 3.0, 4.0]).unwrap();
    let exp = Tensor::from_vec(vec![3], vec![3.0, 2.0, 0.5]).unwrap();
    let out = base.pow(&exp).unwrap();
    assert!(
        (out.data()[0] - 8.0).abs() < 1e-2,
        "2^3 = {}",
        out.data()[0]
    );
    assert!(
        (out.data()[1] - 9.0).abs() < 1e-2,
        "3^2 = {}",
        out.data()[1]
    );
    assert!(
        (out.data()[2] - 2.0).abs() < 1e-2,
        "4^0.5 = {}",
        out.data()[2]
    );
}

#[test]
fn minimum_elementwise() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 5.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![2.0, 4.0, 6.0]).unwrap();
    assert_eq!(a.minimum(&b).unwrap().data(), &[1.0, 4.0, 3.0]);
}

#[test]
fn maximum_elementwise() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 5.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![2.0, 4.0, 6.0]).unwrap();
    assert_eq!(a.maximum(&b).unwrap().data(), &[2.0, 5.0, 6.0]);
}

// ── unary ops ───────────────────────────────────────────────────────

#[test]
fn neg_flips_sign() {
    let t = Tensor::from_vec(vec![3], vec![1.0, -2.0, 0.0]).unwrap();
    assert_eq!(t.neg().data(), &[-1.0, 2.0, 0.0]);
}

#[test]
fn abs_removes_sign() {
    let t = Tensor::from_vec(vec![3], vec![-3.0, 0.0, 4.0]).unwrap();
    assert_eq!(t.abs().data(), &[3.0, 0.0, 4.0]);
}

#[test]
fn exp_computes_e_to_the_x() {
    let t = Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap();
    let out = t.exp();
    assert!((out.data()[0] - 1.0).abs() < 1e-5);
    assert!((out.data()[1] - std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn ln_computes_natural_log() {
    let t = Tensor::from_vec(vec![2], vec![1.0, std::f32::consts::E]).unwrap();
    let out = t.ln();
    assert!((out.data()[0]).abs() < 1e-5);
    assert!((out.data()[1] - 1.0).abs() < 1e-5);
}

#[test]
fn sqrt_computes_square_root() {
    let t = Tensor::from_vec(vec![3], vec![4.0, 9.0, 16.0]).unwrap();
    assert_eq!(t.sqrt().data(), &[2.0, 3.0, 4.0]);
}

#[test]
fn reciprocal_computes_one_over_x() {
    let t = Tensor::from_vec(vec![3], vec![2.0, 4.0, 5.0]).unwrap();
    assert_eq!(t.reciprocal().data(), &[0.5, 0.25, 0.2]);
}

#[test]
fn sign_returns_sign_values() {
    let t = Tensor::from_vec(vec![4], vec![-5.0, 0.0, 3.0, -0.1]).unwrap();
    assert_eq!(t.sign().data(), &[-1.0, 0.0, 1.0, -1.0]);
}

#[test]
fn floor_ceil_round() {
    let t = Tensor::from_vec(vec![3], vec![1.3, 2.7, -0.5]).unwrap();
    assert_eq!(t.floor().data(), &[1.0, 2.0, -1.0]);
    assert_eq!(t.ceil().data(), &[2.0, 3.0, 0.0]);
    assert_eq!(t.round().data(), &[1.0, 3.0, -1.0]);
}

#[test]
fn clamp_restricts_range() {
    let t = Tensor::from_vec(vec![5], vec![-2.0, 0.0, 0.5, 1.0, 3.0]).unwrap();
    assert_eq!(t.clamp(0.0, 1.0).data(), &[0.0, 0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn scale_multiplies_by_constant() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.scale(2.0).data(), &[2.0, 4.0, 6.0]);
}

#[test]
fn add_scalar_adds_constant() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(t.add_scalar(10.0).data(), &[11.0, 12.0, 13.0]);
}

// ── trig / math ops ────────────────────────────────────────────────

#[test]
fn sin_known_values() {
    let t = Tensor::from_vec(
        vec![3],
        vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
    )
    .unwrap();
    let s = t.sin();
    assert!((s.data()[0] - 0.0).abs() < 1e-6);
    assert!((s.data()[1] - 1.0).abs() < 1e-6);
    assert!(s.data()[2].abs() < 1e-5);
}

#[test]
fn cos_known_values() {
    let t = Tensor::from_vec(vec![2], vec![0.0, std::f32::consts::PI]).unwrap();
    let c = t.cos();
    assert!((c.data()[0] - 1.0).abs() < 1e-6);
    assert!((c.data()[1] + 1.0).abs() < 1e-5);
}

#[test]
fn tan_known_values() {
    let t = Tensor::from_vec(vec![2], vec![0.0, std::f32::consts::FRAC_PI_4]).unwrap();
    let r = t.tan();
    assert!(r.data()[0].abs() < 1e-6);
    assert!((r.data()[1] - 1.0).abs() < 1e-5);
}

#[test]
fn asin_acos_inverse() {
    let t = Tensor::from_vec(vec![3], vec![0.0, 0.5, 1.0]).unwrap();
    let s = t.asin().sin();
    for (a, b) in s.data().iter().zip(t.data()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn sinh_cosh_identity() {
    // cosh²(x) - sinh²(x) = 1
    let t = Tensor::from_vec(vec![3], vec![0.0, 1.0, -1.0]).unwrap();
    let sh = t.sinh();
    let ch = t.cosh();
    for i in 0..3 {
        let val = ch.data()[i] * ch.data()[i] - sh.data()[i] * sh.data()[i];
        assert!((val - 1.0).abs() < 1e-4, "cosh²-sinh²={val} at i={i}");
    }
}

#[test]
fn log2_log10_known() {
    let t = Tensor::from_vec(vec![2], vec![1.0, 100.0]).unwrap();
    assert!((t.log2().data()[0]).abs() < 1e-6); // log2(1) = 0
    assert!((t.log10().data()[1] - 2.0).abs() < 1e-5); // log10(100) = 2
}

#[test]
fn degrees_radians_roundtrip() {
    let t = Tensor::from_vec(vec![3], vec![0.0, 1.0, std::f32::consts::PI]).unwrap();
    let rt = t.degrees().radians();
    for (a, b) in rt.data().iter().zip(t.data()) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[test]
fn atan2_quadrants() {
    let y = Tensor::from_vec(vec![4], vec![1.0, 1.0, -1.0, -1.0]).unwrap();
    let x = Tensor::from_vec(vec![4], vec![1.0, -1.0, -1.0, 1.0]).unwrap();
    let a = y.atan2(&x).unwrap();
    let pi = std::f32::consts::PI;
    assert!((a.data()[0] - pi / 4.0).abs() < 1e-5); // Q1
    assert!((a.data()[1] - 3.0 * pi / 4.0).abs() < 1e-5); // Q2
    assert!((a.data()[2] + 3.0 * pi / 4.0).abs() < 1e-5); // Q3
    assert!((a.data()[3] + pi / 4.0).abs() < 1e-5); // Q4
}

// ── comparison ──────────────────────────────────────────────────────

#[test]
fn eq_tensor_works() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![1.0, 5.0, 3.0]).unwrap();
    assert_eq!(a.eq_tensor(&b).unwrap().data(), &[1.0, 0.0, 1.0]);
}

#[test]
fn gt_lt_tensor_works() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 5.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![2.0, 4.0, 3.0]).unwrap();
    assert_eq!(a.gt_tensor(&b).unwrap().data(), &[0.0, 1.0, 0.0]);
    assert_eq!(a.lt_tensor(&b).unwrap().data(), &[1.0, 0.0, 0.0]);
}

#[test]
fn all_finite_checks_correctness() {
    let ok = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert!(ok.all_finite());
    let bad = Tensor::from_vec(vec![2], vec![1.0, f32::NAN]).unwrap();
    assert!(!bad.all_finite());
}

// ── cumsum / cumprod ────────────────────────────────────────────────

#[test]
fn cumsum_axis0() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let cs = t.cumsum(0).unwrap();
    assert_eq!(cs.shape(), &[2, 3]);
    assert_eq!(cs.data(), &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn cumsum_axis1() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let cs = t.cumsum(1).unwrap();
    assert_eq!(cs.data(), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn cumprod_axis1() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0]).unwrap();
    let cp = t.cumprod(1).unwrap();
    assert_eq!(cp.data(), &[1.0, 2.0, 6.0, 2.0, 6.0, 24.0]);
}

// ── diag / pad ─────────────────────────────────────────────────────

#[test]
fn diag_creates_diagonal_matrix() {
    let v = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let d = Tensor::diag(&v).unwrap();
    assert_eq!(d.shape(), &[3, 3]);
    assert_eq!(d.data(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
}

#[test]
fn diag_extract_gets_diagonal() {
    let m = Tensor::from_vec(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let d = m.diag_extract().unwrap();
    assert_eq!(d.shape(), &[3]);
    assert_eq!(d.data(), &[1.0, 5.0, 9.0]);
}

#[test]
fn diag_roundtrip() {
    let v = Tensor::from_vec(vec![4], vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    let d = Tensor::diag(&v).unwrap().diag_extract().unwrap();
    assert_eq!(d.data(), v.data());
}

#[test]
fn pad_symmetric_2d() {
    let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let p = t.pad(&[(1, 1), (1, 1)], 0.0).unwrap();
    assert_eq!(p.shape(), &[4, 4]);
    assert_eq!(p.data()[5], 1.0);
    assert_eq!(p.data()[6], 2.0);
    assert_eq!(p.data()[9], 3.0);
    assert_eq!(p.data()[10], 4.0);
}

#[test]
fn pad_asymmetric_1d() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let p = t.pad(&[(2, 1)], -1.0).unwrap();
    assert_eq!(p.shape(), &[6]);
    assert_eq!(p.data(), &[-1.0, -1.0, 1.0, 2.0, 3.0, -1.0]);
}

#[test]
fn pad_zero_padding_is_identity() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let p = t.pad(&[(0, 0), (0, 0)], 0.0).unwrap();
    assert_eq!(p.data(), t.data());
}

#[test]
fn cumsum_invalid_axis() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    assert!(t.cumsum(2).is_err());
}

#[test]
fn cumprod_axis0() {
    let t = Tensor::from_vec(vec![3], vec![2.0, 3.0, 4.0]).unwrap();
    let cp = t.cumprod(0).unwrap();
    assert_eq!(cp.data(), &[2.0, 6.0, 24.0]);
}

// ── Random tensor creation ──────────────────────────────────────
#[test]
fn rand_produces_correct_shape() {
    let t = Tensor::rand(vec![3, 4], 42).unwrap();
    assert_eq!(t.shape(), &[3, 4]);
    assert_eq!(t.data().len(), 12);
}

#[test]
fn rand_values_in_unit_interval() {
    let t = Tensor::rand(vec![100], 42).unwrap();
    assert!(t.data().iter().all(|&v| (0.0..1.0).contains(&v)));
}

#[test]
fn rand_deterministic_seed() {
    let a = Tensor::rand(vec![10], 123).unwrap();
    let b = Tensor::rand(vec![10], 123).unwrap();
    assert_eq!(a.data(), b.data());
}

#[test]
fn randn_produces_correct_shape() {
    let t = Tensor::randn(vec![2, 5], 42).unwrap();
    assert_eq!(t.shape(), &[2, 5]);
}

#[test]
fn randn_has_reasonable_statistics() {
    let t = Tensor::randn(vec![10000], 42).unwrap();
    let mean = t.mean();
    let std = t.std_dev();
    assert!((mean).abs() < 0.1, "mean {mean} should be near 0");
    assert!((std - 1.0).abs() < 0.1, "std {std} should be near 1");
}

#[test]
fn randint_values_in_range() {
    let t = Tensor::randint(vec![100], 5, 10, 42).unwrap();
    assert!(t.data().iter().all(|&v| (5.0..10.0).contains(&v)));
}

#[test]
fn randperm_is_permutation() {
    let t = Tensor::randperm(10, 42).unwrap();
    assert_eq!(t.shape(), &[10]);
    let mut sorted: Vec<f32> = t.data().to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected: Vec<f32> = (0..10).map(|i| i as f32).collect();
    assert_eq!(sorted, expected);
}

#[test]
fn randperm_different_seeds_differ() {
    let a = Tensor::randperm(10, 42).unwrap();
    let b = Tensor::randperm(10, 99).unwrap();
    assert_ne!(a.data(), b.data());
}

// ── ne / le / ge comparison ─────────────────────────────────────────

#[test]
fn ne_tensor_basic() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![1.0, 0.0, 3.0]).unwrap();
    assert_eq!(a.ne_tensor(&b).unwrap().data(), &[0.0, 1.0, 0.0]);
}

#[test]
fn le_tensor_basic() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![2.0, 2.0, 2.0]).unwrap();
    assert_eq!(a.le_tensor(&b).unwrap().data(), &[1.0, 1.0, 0.0]);
}

#[test]
fn ge_tensor_basic() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![2.0, 2.0, 2.0]).unwrap();
    assert_eq!(a.ge_tensor(&b).unwrap().data(), &[0.0, 1.0, 1.0]);
}

// ── chunk ───────────────────────────────────────────────────────────

#[test]
fn chunk_even() {
    let t = Tensor::from_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].data(), &[1.0, 2.0]);
    assert_eq!(chunks[1].data(), &[3.0, 4.0]);
    assert_eq!(chunks[2].data(), &[5.0, 6.0]);
}

#[test]
fn chunk_uneven() {
    let t = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].data(), &[1.0, 2.0]);
    assert_eq!(chunks[1].data(), &[3.0, 4.0]);
    assert_eq!(chunks[2].data(), &[5.0]);
}

// ── histogram ───────────────────────────────────────────────────────

#[test]
fn histogram_basic() {
    let t = Tensor::from_vec(vec![6], vec![0.5, 1.5, 2.5, 0.1, 1.9, 2.9]).unwrap();
    let h = t.histogram(3, 0.0, 3.0).unwrap();
    assert_eq!(h.shape(), &[3]);
    // bin 0: [0,1) -> 0.5, 0.1 = 2
    // bin 1: [1,2) -> 1.5, 1.9 = 2
    // bin 2: [2,3] -> 2.5, 2.9 = 2
    assert_eq!(h.data(), &[2.0, 2.0, 2.0]);
}

// ── bincount ────────────────────────────────────────────────────────

#[test]
fn bincount_basic() {
    let t = Tensor::from_vec(vec![6], vec![0.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let bc = t.bincount(3).unwrap();
    assert_eq!(bc.shape(), &[3]);
    assert_eq!(bc.data(), &[1.0, 2.0, 3.0]);
}

// ── scalar convenience ─────────────────────────────────────────────

#[test]
fn item_scalar_tensor() {
    let t = Tensor::scalar(42.0);
    assert_eq!(t.item().unwrap(), 42.0);
}

#[test]
fn item_error_multi_element() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    assert!(t.item().is_err());
}

#[test]
fn is_scalar_true() {
    let t = Tensor::scalar(7.0);
    assert!(t.is_scalar());
}

#[test]
fn is_scalar_false() {
    let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    assert!(!t.is_scalar());
}

// ── gather / scatter / index_select / scatter_add ───────────────────

#[test]
fn gather_1d() {
    let src = Tensor::from_vec(vec![4], vec![0.0, 10.0, 20.0, 30.0]).unwrap();
    let idx = Tensor::from_vec(vec![2], vec![3.0, 1.0]).unwrap();
    let out = src.gather(0, &idx).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.data(), &[30.0, 10.0]);
}

#[test]
fn gather_2d() {
    // [[1, 2], [3, 4]]
    let src = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    // gather along dim=1 with index [[1, 0], [0, 1]]
    let idx = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let out = src.gather(1, &idx).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    // out[0][0] = src[0][1] = 2, out[0][1] = src[0][0] = 1
    // out[1][0] = src[1][0] = 3, out[1][1] = src[1][1] = 4
    assert_eq!(out.data(), &[2.0, 1.0, 3.0, 4.0]);
}

#[test]
fn scatter_2d() {
    let dst = Tensor::zeros(vec![2, 3]).unwrap();
    let idx = Tensor::from_vec(vec![2, 1], vec![1.0, 0.0]).unwrap();
    let src = Tensor::from_vec(vec![2, 1], vec![5.0, 7.0]).unwrap();
    let out = dst.scatter(1, &idx, &src).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    // row 0: [0, 5, 0]  (index 1 -> col 1)
    // row 1: [7, 0, 0]  (index 0 -> col 0)
    assert_eq!(out.data(), &[0.0, 5.0, 0.0, 7.0, 0.0, 0.0]);
}

#[test]
fn index_select_basic() {
    // [[1, 2, 3], [4, 5, 6]]
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let idx = Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap();
    let out = t.index_select(0, &idx).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    // row 1 then row 0
    assert_eq!(out.data(), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn scatter_add_accumulates() {
    let dst = Tensor::zeros(vec![1, 4]).unwrap();
    // scatter_add at indices [1, 1, 2] with values [10, 20, 30]
    let idx = Tensor::from_vec(vec![1, 3], vec![1.0, 1.0, 2.0]).unwrap();
    let src = Tensor::from_vec(vec![1, 3], vec![10.0, 20.0, 30.0]).unwrap();
    let out = dst.scatter_add(1, &idx, &src).unwrap();
    assert_eq!(out.shape(), &[1, 4]);
    // index 1 gets 10+20=30, index 2 gets 30
    assert_eq!(out.data(), &[0.0, 30.0, 30.0, 0.0]);
}

#[test]
fn gather_out_of_bounds_returns_zero() {
    // Existing gather silently returns 0.0 for out-of-bounds indices.
    let src = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let idx = Tensor::from_vec(vec![1], vec![5.0]).unwrap();
    let result = src.gather(0, &idx).unwrap();
    assert_eq!(result.data()[0], 0.0);
}
