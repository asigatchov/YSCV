use crate::Tensor;

#[test]
fn where_cond_selects_correctly() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let f = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let cond = Tensor::from_vec(vec![3], vec![1.0, 0.0, 1.0]).unwrap();
    let out = t.where_cond(&cond, &f).unwrap();
    assert_eq!(out.data(), &[1.0, 20.0, 3.0]);
}

#[test]
fn masked_fill_replaces_correctly() {
    let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let mask = Tensor::from_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap();
    let out = t.masked_fill(&mask, -999.0).unwrap();
    assert_eq!(out.data(), &[1.0, -999.0, 3.0]);
}

#[test]
fn topk_returns_largest() {
    let t = Tensor::from_vec(vec![1, 5], vec![3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
    let (vals, idxs) = t.topk(3).unwrap();
    assert_eq!(vals.shape(), &[1, 3]);
    assert_eq!(vals.data()[0], 5.0);
    assert_eq!(vals.data()[1], 4.0);
    assert_eq!(vals.data()[2], 3.0);
    assert_eq!(idxs.data()[0], 4.0);
}

#[test]
fn triu_zeros_below_diagonal() {
    let t = Tensor::from_vec(vec![3, 3], vec![1.0; 9]).unwrap();
    let out = t.triu(0).unwrap();
    assert_eq!(out.data()[3], 0.0); // row=1,col=0
    assert_eq!(out.data()[4], 1.0); // row=1,col=1 (diagonal)
}

#[test]
fn tril_zeros_above_diagonal() {
    let t = Tensor::from_vec(vec![3, 3], vec![1.0; 9]).unwrap();
    let out = t.tril(0).unwrap();
    assert_eq!(out.data()[1], 0.0); // row=0,col=1
    assert_eq!(out.data()[4], 1.0); // row=1,col=1 (diagonal)
}

#[test]
fn eye_creates_identity() {
    let t = Tensor::eye(3).unwrap();
    assert_eq!(t.data()[0], 1.0);
    assert_eq!(t.data()[1], 0.0);
    assert_eq!(t.data()[4], 1.0);
}

#[test]
fn repeat_duplicates_along_axes() {
    let t = Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).unwrap();
    let out = t.repeat(&[2, 3]).unwrap();
    assert_eq!(out.shape(), &[2, 6]);
}

#[test]
fn gather_along_axis() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let idx = Tensor::from_vec(vec![2, 1], vec![1.0, 2.0]).unwrap();
    let out = t.gather(1, &idx).unwrap();
    assert_eq!(out.data(), &[2.0, 6.0]);
}

// ── sort / argsort ──────────────────────────────────────────────────

#[test]
fn sort_ascending_1d() {
    let t = Tensor::from_vec(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
    let (vals, idxs) = t.sort(0, false).unwrap();
    assert_eq!(vals.data(), &[1.0, 1.0, 3.0, 4.0, 5.0]);
    // original indices of the 1.0s are 1 and 3
    assert!(idxs.data()[0] == 1.0 || idxs.data()[0] == 3.0);
}

#[test]
fn sort_descending_2d() {
    let t = Tensor::from_vec(vec![2, 3], vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]).unwrap();
    let (vals, _) = t.sort(1, true).unwrap();
    assert_eq!(vals.data(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

#[test]
fn argsort_matches_sort_indices() {
    let t = Tensor::from_vec(vec![4], vec![10.0, 30.0, 20.0, 40.0]).unwrap();
    let idxs = t.argsort(0, false).unwrap();
    assert_eq!(idxs.data(), &[0.0, 2.0, 1.0, 3.0]);
}

// ── unique ──────────────────────────────────────────────────────────

#[test]
fn unique_basic() {
    let t = Tensor::from_vec(vec![6], vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0]).unwrap();
    let (vals, inv, counts) = t.unique();
    assert_eq!(vals.data(), &[1.0, 2.0, 3.0]);
    assert_eq!(counts.data(), &[2.0, 2.0, 2.0]);
    // inverse should map back: vals[inv[i]] == original[i]
    for (i, &idx) in inv.data().iter().enumerate() {
        assert_eq!(vals.data()[idx as usize], t.data()[i]);
    }
}

// ── nonzero ─────────────────────────────────────────────────────────

#[test]
fn nonzero_2d() {
    let t = Tensor::from_vec(vec![2, 3], vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]).unwrap();
    let nz = t.nonzero();
    assert_eq!(nz.shape(), &[3, 2]); // 3 nonzero elements, rank=2
    // coordinates: (0,1), (1,0), (1,2)
    assert_eq!(nz.data(), &[0.0, 1.0, 1.0, 0.0, 1.0, 2.0]);
}

// ── flip ────────────────────────────────────────────────────────────

#[test]
fn flip_1d() {
    let t = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = t.flip(&[0]).unwrap();
    assert_eq!(out.data(), &[4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn flip_2d_both() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = t.flip(&[0, 1]).unwrap();
    assert_eq!(out.data(), &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
}

// ── roll ────────────────────────────────────────────────────────────

#[test]
fn roll_positive() {
    let t = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = t.roll(1, 0).unwrap();
    assert_eq!(out.data(), &[4.0, 1.0, 2.0, 3.0]);
}

#[test]
fn roll_negative() {
    let t = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = t.roll(-1, 0).unwrap();
    assert_eq!(out.data(), &[2.0, 3.0, 4.0, 1.0]);
}

// ── linspace / arange ───────────────────────────────────────────────

#[test]
fn linspace_basic() {
    let t = Tensor::linspace(0.0, 1.0, 5).unwrap();
    assert_eq!(t.shape(), &[5]);
    assert!((t.data()[0] - 0.0).abs() < 1e-6);
    assert!((t.data()[4] - 1.0).abs() < 1e-6);
    assert!((t.data()[2] - 0.5).abs() < 1e-6);
}

#[test]
fn arange_basic() {
    let t = Tensor::arange(0.0, 5.0, 1.0).unwrap();
    assert_eq!(t.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn arange_fractional_step() {
    let t = Tensor::arange(0.0, 1.0, 0.5).unwrap();
    assert_eq!(t.data(), &[0.0, 0.5]);
}

// ── meshgrid ────────────────────────────────────────────────────────

#[test]
fn meshgrid_2d() {
    let x = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let y = Tensor::from_vec(vec![2], vec![10.0, 20.0]).unwrap();
    let grids = Tensor::meshgrid(&[x, y]).unwrap();
    assert_eq!(grids.len(), 2);
    assert_eq!(grids[0].shape(), &[3, 2]);
    assert_eq!(grids[1].shape(), &[3, 2]);
    // grids[0] should repeat x values across columns
    assert_eq!(grids[0].data(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    // grids[1] should repeat y values across rows
    assert_eq!(grids[1].data(), &[10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
}

// ── boolean_mask ────────────────────────────────────────────────────

#[test]
fn boolean_mask_basic() {
    let t = Tensor::from_vec(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let m = Tensor::from_vec(vec![5], vec![1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
    let out = t.boolean_mask(&m).unwrap();
    assert_eq!(out.data(), &[10.0, 30.0, 50.0]);
}

// ── index_select ────────────────────────────────────────────────────

#[test]
fn index_select_dim0() {
    let t = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let idx = Tensor::from_vec(vec![2], vec![0.0, 2.0]).unwrap();
    let out = t.index_select(0, &idx).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn index_select_dim1() {
    let t = Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let idx = Tensor::from_vec(vec![2], vec![1.0, 3.0]).unwrap();
    let out = t.index_select(1, &idx).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[2.0, 4.0, 6.0, 8.0]);
}

// ── step_slice ───────────────────────────────────────────────────────

#[test]
fn step_slice_every_other_1d() {
    let t = Tensor::from_vec(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let out = t.step_slice(0, 0, 6, 2).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.data(), &[0.0, 2.0, 4.0]);
}

#[test]
fn step_slice_with_offset() {
    let t = Tensor::from_vec(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let out = t.step_slice(0, 1, 5, 2).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.data(), &[1.0, 3.0]);
}

#[test]
fn step_slice_2d_along_rows() {
    let t = Tensor::from_vec(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let out = t.step_slice(0, 0, 4, 2).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[1.0, 2.0, 5.0, 6.0]);
}

// ── einsum ───────────────────────────────────────────────────────────

#[test]
fn einsum_matmul() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    assert_eq!(out.data(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn einsum_transpose() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = Tensor::einsum("ij->ji", &[&a]).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn einsum_diagonal() {
    let a = Tensor::from_vec(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    )
    .unwrap();
    let out = Tensor::einsum("ii->i", &[&a]).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn einsum_row_sum() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = Tensor::einsum("ij->i", &[&a]).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.data(), &[6.0, 15.0]);
}

#[test]
fn einsum_col_sum() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = Tensor::einsum("ij->j", &[&a]).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn einsum_total_sum() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = Tensor::einsum("ij->", &[&a]).unwrap();
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.data(), &[21.0]);
}

#[test]
fn einsum_dot_product() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
    let out = Tensor::einsum("i,i->", &[&a, &b]).unwrap();
    assert_eq!(out.data(), &[32.0]); // 1*4 + 2*5 + 3*6
}

#[test]
fn einsum_frobenius() {
    let a = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let out = Tensor::einsum("ij,ij->", &[&a, &b]).unwrap();
    assert_eq!(out.data(), &[70.0]); // 1*5 + 2*6 + 3*7 + 4*8 = 70
}

#[test]
fn einsum_unsupported_returns_error() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    let result = Tensor::einsum("ijk->ij", &[&a]);
    assert!(result.is_err());
}
