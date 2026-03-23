use yscv_tensor::Tensor;

use super::super::{crop, flip_horizontal, flip_vertical, pad_constant, rotate90_cw};

#[test]
fn flip_horizontal_mirrors_width_axis() {
    let input = Tensor::from_vec(
        vec![2, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();
    let out = flip_horizontal(&input).unwrap();
    assert_eq!(out.shape(), &[2, 3, 1]);
    assert_eq!(
        out.data(),
        &[
            3.0, 2.0, 1.0, //
            6.0, 5.0, 4.0,
        ]
    );
}

#[test]
fn flip_vertical_mirrors_height_axis() {
    let input = Tensor::from_vec(
        vec![2, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();
    let out = flip_vertical(&input).unwrap();
    assert_eq!(out.shape(), &[2, 3, 1]);
    assert_eq!(
        out.data(),
        &[
            4.0, 5.0, 6.0, //
            1.0, 2.0, 3.0,
        ]
    );
}

#[test]
fn rotate90_cw_swaps_height_width() {
    let input = Tensor::from_vec(
        vec![2, 3, 1],
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();
    let out = rotate90_cw(&input).unwrap();
    assert_eq!(out.shape(), &[3, 2, 1]);
    assert_eq!(
        out.data(),
        &[
            4.0, 1.0, //
            5.0, 2.0, //
            6.0, 3.0,
        ]
    );
}

#[test]
fn pad_constant_adds_border() {
    let img = Tensor::from_vec(vec![2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let padded = pad_constant(&img, 1, 1, 1, 1, 0.0).unwrap();
    assert_eq!(padded.shape(), &[4, 4, 1]);
    assert_eq!(padded.data()[0], 0.0);
    assert_eq!(padded.data()[5], 1.0);
}

#[test]
fn crop_extracts_subregion() {
    let img = Tensor::from_vec(
        vec![3, 3, 1],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let cropped = crop(&img, 1, 1, 2, 2).unwrap();
    assert_eq!(cropped.shape(), &[2, 2, 1]);
    assert_eq!(cropped.data(), &[5.0, 6.0, 8.0, 9.0]);
}
