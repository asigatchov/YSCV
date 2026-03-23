use yscv_tensor::Tensor;

#[test]
fn scaled_dot_product_attention_correct_shape() {
    let q = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let k = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let v = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let out = crate::scaled_dot_product_attention(&q, &k, &v).unwrap();
    assert_eq!(out.shape(), &[4, 8]);
}

#[test]
fn multi_head_attention_correct_shape() {
    let config = crate::MultiHeadAttentionConfig {
        d_model: 8,
        num_heads: 2,
    };
    let mha = crate::MultiHeadAttention::new(&config).unwrap();
    let input = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let out = mha.forward(&input).unwrap();
    assert_eq!(out.shape(), &[4, 8]);
}

#[test]
fn transformer_encoder_block_correct_shape() {
    let block = crate::TransformerEncoderBlock::new(8, 2, 16).unwrap();
    let input = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let out = block.forward(&input).unwrap();
    assert_eq!(out.shape(), &[4, 8]);
}

#[test]
fn causal_mask_shape_and_values() {
    let mask = crate::generate_causal_mask(4).unwrap();
    assert_eq!(mask.shape(), &[4, 4]);
    let data = mask.data();
    // Below and on diagonal should be 0.0
    // Row 0: [0, -inf, -inf, -inf]
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], f32::NEG_INFINITY);
    assert_eq!(data[2], f32::NEG_INFINITY);
    assert_eq!(data[3], f32::NEG_INFINITY);
    // Row 1: [0, 0, -inf, -inf]
    assert_eq!(data[4], 0.0);
    assert_eq!(data[5], 0.0);
    assert_eq!(data[6], f32::NEG_INFINITY);
    assert_eq!(data[7], f32::NEG_INFINITY);
    // Row 3: [0, 0, 0, 0]
    assert_eq!(data[12], 0.0);
    assert_eq!(data[13], 0.0);
    assert_eq!(data[14], 0.0);
    assert_eq!(data[15], 0.0);
}

#[test]
fn causal_mask_diagonal_zero() {
    let mask = crate::generate_causal_mask(4).unwrap();
    let data = mask.data();
    for i in 0..4 {
        assert_eq!(
            data[i * 4 + i],
            0.0,
            "diagonal element [{i},{i}] should be 0.0"
        );
    }
}

#[test]
fn padding_mask_basic() {
    let mask = crate::generate_padding_mask(&[3, 2], 4).unwrap();
    assert_eq!(mask.shape(), &[2, 4]);
    let data = mask.data();
    // Batch 0: length 3 -> [0, 0, 0, -inf]
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert_eq!(data[3], f32::NEG_INFINITY);
    // Batch 1: length 2 -> [0, 0, -inf, -inf]
    assert_eq!(data[4], 0.0);
    assert_eq!(data[5], 0.0);
    assert_eq!(data[6], f32::NEG_INFINITY);
    assert_eq!(data[7], f32::NEG_INFINITY);
}
