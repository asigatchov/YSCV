use crate::Tensor;

#[test]
fn trace_3x3() {
    let m = Tensor::from_vec(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    assert_eq!(m.trace().unwrap(), 15.0);
}

#[test]
fn dot_product() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
    assert_eq!(a.dot(&b).unwrap(), 32.0);
}

#[test]
fn cross_product() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap();
    let c = a.cross(&b).unwrap();
    assert_eq!(c.data(), &[0.0, 0.0, 1.0]); // i x j = k
}

#[test]
fn norm_l1_and_l2() {
    let t = Tensor::from_vec(vec![3], vec![3.0, -4.0, 0.0]).unwrap();
    assert!((t.norm(1.0) - 7.0).abs() < 1e-6);
    assert!((t.norm(2.0) - 5.0).abs() < 1e-6);
}

#[test]
fn det_2x2() {
    let m = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!((m.det().unwrap() - (-2.0)).abs() < 1e-5);
}

#[test]
fn det_3x3() {
    let m = Tensor::from_vec(
        vec![3, 3],
        vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0],
    )
    .unwrap();
    assert!((m.det().unwrap() - (-306.0)).abs() < 1e-3);
}

#[test]
fn inv_2x2_roundtrip() {
    let m = Tensor::from_vec(vec![2, 2], vec![4.0, 7.0, 2.0, 6.0]).unwrap();
    let mi = m.inv().unwrap();
    // det = 24-14 = 10, inv = [0.6, -0.7, -0.2, 0.4]
    assert!((mi.data()[0] - 0.6).abs() < 1e-5);
    assert!((mi.data()[1] - (-0.7)).abs() < 1e-5);
}

#[test]
fn solve_2x2() {
    // [2, 1; 5, 3] * x = [4; 7]  =>  x = [5; -6]
    let a = Tensor::from_vec(vec![2, 2], vec![2.0, 1.0, 5.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![2], vec![4.0, 7.0]).unwrap();
    let x = a.solve(&b).unwrap();
    assert!((x.data()[0] - 5.0).abs() < 1e-4);
    assert!((x.data()[1] - (-6.0)).abs() < 1e-4);
}

#[test]
fn qr_orthogonal() {
    let m = Tensor::from_vec(
        vec![3, 3],
        vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();
    let (q, r) = m.qr().unwrap();
    assert_eq!(q.shape(), &[3, 3]);
    assert_eq!(r.shape(), &[3, 3]);
    // Q should be orthogonal: columns should be unit vectors
    let qd = q.data();
    let n = 3;
    for i in 0..n {
        let mut dot = 0.0f32;
        for k in 0..n {
            dot += qd[k * n + i] * qd[k * n + i];
        }
        assert!((dot - 1.0).abs() < 1e-4, "column {i} not unit: {dot}");
    }
}

#[test]
fn cholesky_positive_definite() {
    // A = [[4, 2], [2, 3]] is PD. L = [[2, 0], [1, sqrt(2)]]
    let a = Tensor::from_vec(vec![2, 2], vec![4.0, 2.0, 2.0, 3.0]).unwrap();
    let l = a.cholesky().unwrap();
    assert!((l.data()[0] - 2.0).abs() < 1e-5);
    assert!(l.data()[1].abs() < 1e-5); // upper triangle is 0
    assert!((l.data()[2] - 1.0).abs() < 1e-5);
    assert!((l.data()[3] - (2.0f32).sqrt()).abs() < 1e-5);
}

#[test]
fn inv_non_square_errors() {
    let m = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    assert!(m.inv().is_err());
}
