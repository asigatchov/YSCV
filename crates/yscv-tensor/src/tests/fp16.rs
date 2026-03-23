use crate::{DType, Tensor};

#[test]
fn fp16_roundtrip_preserves_values() {
    let t = Tensor::from_vec(vec![4], vec![1.0, -0.5, 0.0, 65504.0]).unwrap();
    let fp16 = t.to_fp16();
    let back = Tensor::from_fp16(vec![4], &fp16).unwrap();
    assert!((back.data()[0] - 1.0).abs() < 0.01);
    assert!((back.data()[1] - (-0.5)).abs() < 0.01);
    assert_eq!(back.data()[2], 0.0);
    assert!((back.data()[3] - 65504.0).abs() < 100.0);
}

#[test]
fn native_f16_storage_roundtrip() {
    let t = Tensor::from_vec(vec![3], vec![1.0, -2.0, 0.5]).unwrap();
    let f16 = t.to_dtype(DType::F16);
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.shape(), &[3]);
    assert!(f16.data_f16().is_ok());
    assert!(f16.try_data_f32().is_err());
    let back = f16.to_dtype(DType::F32);
    assert_eq!(back.dtype(), DType::F32);
    assert!((back.data()[0] - 1.0).abs() < 0.01);
    assert!((back.data()[1] - (-2.0)).abs() < 0.01);
    assert!((back.data()[2] - 0.5).abs() < 0.01);
}

#[test]
fn native_bf16_storage_roundtrip() {
    let t = Tensor::from_vec(vec![3], vec![1.0, -2.0, 0.5]).unwrap();
    let bf = t.to_dtype(DType::BF16);
    assert_eq!(bf.dtype(), DType::BF16);
    assert!(bf.data_bf16().is_ok());
    let back = bf.to_dtype(DType::F32);
    assert!((back.data()[0] - 1.0).abs() < 0.01);
    assert!((back.data()[1] - (-2.0)).abs() < 0.01);
    assert!((back.data()[2] - 0.5).abs() < 0.01);
}

#[test]
fn from_f16_constructor() {
    let bits = vec![0x3C00u16, 0x4000, 0x3800]; // 1.0, 2.0, 0.5
    let t = Tensor::from_f16(vec![3], bits).unwrap();
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.len(), 3);
    assert!((t.get(&[0]).unwrap() - 1.0).abs() < 0.01);
    assert!((t.get(&[1]).unwrap() - 2.0).abs() < 0.01);
}

#[test]
fn from_bf16_constructor() {
    let bits = vec![0x3F80u16, 0x4000, 0x3F00]; // 1.0, 2.0, 0.5
    let t = Tensor::from_bf16(vec![3], bits).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert!((t.get(&[0]).unwrap() - 1.0).abs() < 0.01);
    assert!((t.get(&[1]).unwrap() - 2.0).abs() < 0.01);
}

#[test]
fn dtype_set_preserves_native_format() {
    let mut t = Tensor::from_vec(vec![2], vec![0.0, 0.0])
        .unwrap()
        .to_dtype(DType::F16);
    t.set(&[0], 3.15).unwrap();
    let val = t.get(&[0]).unwrap();
    assert!((val - 3.15).abs() < 0.02);
}

#[test]
fn f16_reshape() {
    let t = Tensor::from_vec(vec![6], vec![1.0; 6])
        .unwrap()
        .to_dtype(DType::F16);
    let reshaped = t.reshape(vec![2, 3]).unwrap();
    assert_eq!(reshaped.shape(), &[2, 3]);
    assert_eq!(reshaped.dtype(), DType::F16);
}
