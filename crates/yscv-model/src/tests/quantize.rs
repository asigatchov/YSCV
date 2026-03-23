use yscv_tensor::Tensor;

#[test]
fn quantize_symmetric_roundtrip() {
    let t = Tensor::from_vec(vec![2, 3], vec![1.0, -0.5, 0.3, 0.0, 0.7, -1.0]).unwrap();
    let qt = crate::QuantizedTensor::from_tensor(&t, crate::QuantMode::Symmetric);
    assert_eq!(qt.shape, vec![2, 3]);
    assert_eq!(qt.zero_point, 0);
    let restored = qt.to_tensor().unwrap();
    for (a, b) in t.data().iter().zip(restored.data()) {
        assert!((a - b).abs() < 0.02, "symmetric roundtrip: {a} vs {b}");
    }
}

#[test]
fn quantize_asymmetric_roundtrip() {
    let t = Tensor::from_vec(vec![4], vec![0.0, 0.25, 0.5, 1.0]).unwrap();
    let qt = crate::QuantizedTensor::from_tensor(&t, crate::QuantMode::Asymmetric);
    let restored = qt.to_tensor().unwrap();
    for (a, b) in t.data().iter().zip(restored.data()) {
        assert!((a - b).abs() < 0.02, "asymmetric roundtrip: {a} vs {b}");
    }
}

#[test]
fn quantize_weights_batch_dequantize() {
    let w1 = Tensor::from_vec(vec![2, 2], vec![1.0, -1.0, 0.5, -0.5]).unwrap();
    let w2 = Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap();
    let qw = crate::quantize_weights(&[w1.clone(), w2.clone()], crate::QuantMode::Symmetric);
    let dw = crate::dequantize_weights(&qw).unwrap();
    assert_eq!(dw.len(), 2);
    assert_eq!(dw[0].shape(), &[2, 2]);
    assert_eq!(dw[1].shape(), &[3]);
}
