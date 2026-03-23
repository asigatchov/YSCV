use yscv_tensor::Tensor;

#[test]
fn cross_attention_forward_shape() {
    let ca = crate::CrossAttention::new(8, 2).unwrap();
    let query = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let kv = Tensor::from_vec(vec![6, 8], vec![0.1; 48]).unwrap();
    let out = ca.forward(&query, &kv).unwrap();
    assert_eq!(out.shape(), &[4, 8]);
}

#[test]
fn decoder_block_forward_shape() {
    let block = crate::TransformerDecoderBlock::new(8, 2, 16).unwrap();
    let target = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let memory = Tensor::from_vec(vec![6, 8], vec![0.1; 48]).unwrap();
    let out = block.forward(&target, &memory).unwrap();
    assert_eq!(out.shape(), &[4, 8]);
}

#[test]
fn decoder_stack_forward() {
    let decoder = crate::TransformerDecoder::new(8, 2, 16, 2).unwrap();
    let target = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).unwrap();
    let memory = Tensor::from_vec(vec![6, 8], vec![0.1; 48]).unwrap();
    let out = decoder.forward(&target, &memory).unwrap();
    assert_eq!(out.shape(), &[4, 8]);
}
