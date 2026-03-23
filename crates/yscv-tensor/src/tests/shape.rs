use crate::Tensor;

#[test]
fn transpose_2d_swaps_rows_and_cols() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tr = t.transpose_2d().unwrap();
    assert_eq!(tr.shape(), &[3, 2]);
    assert_eq!(tr.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_2d_rejects_non_rank2() {
    let t = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
    assert!(t.transpose_2d().is_err());
}

#[test]
fn permute_reorders_axes() {
    let t = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
    let p = t.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), &[4, 2, 3]);
    assert_eq!(p.len(), 24);
}

#[test]
fn permute_identity_preserves_data() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let p = t.permute(&[0, 1]).unwrap();
    assert_eq!(p.data(), t.data());
}

#[test]
fn permute_rejects_wrong_length() {
    let t = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
    assert!(t.permute(&[0]).is_err());
}

#[test]
fn unsqueeze_inserts_axis() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.shape(), &[1, 3]);
    let u2 = t.unsqueeze(1).unwrap();
    assert_eq!(u2.shape(), &[3, 1]);
}

#[test]
fn squeeze_removes_unit_axis() {
    let t = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let s = t.squeeze(0).unwrap();
    assert_eq!(s.shape(), &[3]);
}

#[test]
fn squeeze_rejects_non_unit_axis() {
    let t = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
    assert!(t.squeeze(0).is_err());
}

#[test]
fn cat_along_axis_0() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec(vec![1, 3], vec![7.0, 8.0, 9.0]).unwrap();
    let out = Tensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn cat_along_axis_1() {
    let a = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![2, 1], vec![5.0, 6.0]).unwrap();
    let out = Tensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
}

#[test]
fn cat_rejects_shape_mismatch() {
    let a = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
    let b = Tensor::from_vec(vec![2, 4], vec![0.0; 8]).unwrap();
    assert!(Tensor::cat(&[&a, &b], 0).is_err());
}

#[test]
fn stack_creates_new_axis() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
    let out = Tensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn select_picks_slice_along_axis() {
    let t = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let s = t.select(0, 1).unwrap();
    assert_eq!(s.shape(), &[2]);
    assert_eq!(s.data(), &[3.0, 4.0]);
}

#[test]
fn narrow_slices_along_axis() {
    let t = Tensor::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    let n = t.narrow(0, 1, 2).unwrap();
    assert_eq!(n.shape(), &[2]);
    assert_eq!(n.data(), &[20.0, 30.0]);
}

#[test]
fn narrow_2d_along_axis_1() {
    let t = Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let n = t.narrow(1, 1, 2).unwrap();
    assert_eq!(n.shape(), &[2, 2]);
    assert_eq!(n.data(), &[2.0, 3.0, 6.0, 7.0]);
}
