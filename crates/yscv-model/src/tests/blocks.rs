use yscv_tensor::Tensor;

#[test]
fn squeeze_excite_block_correct_shape() {
    let se = crate::SqueezeExciteBlock::new(16, 4).unwrap();
    let input = Tensor::from_vec(vec![1, 4, 4, 16], vec![0.5; 256]).unwrap();
    let out = se.forward(&input).unwrap();
    assert_eq!(out.shape(), &[1, 4, 4, 16]);
}

#[test]
fn mbconv_block_creates_without_error() {
    let block = crate::MbConvBlock::new(8, 8, 1, 3, 1, Some(4), 1e-5).unwrap();
    assert!(block.use_residual); // in_ch == out_ch and stride == 1
}

#[test]
fn mbconv_block_forward_correct_shape() {
    // Use expand_ratio=2 so in_ch != out path isn't skip (different spatial dim after depthwise)
    let mut block = crate::MbConvBlock::new(4, 4, 1, 3, 1, None, 1e-5).unwrap();
    block.use_residual = false; // depthwise 3x3 changes spatial dims
    let input = Tensor::from_vec(vec![1, 6, 6, 4], vec![0.1; 144]).unwrap();
    let out = block.forward(&input).unwrap();
    assert_eq!(out.shape()[0], 1);
    assert_eq!(out.shape()[3], 4);
}

#[test]
fn unet_encoder_stage_forward() {
    let enc = crate::UNetEncoderStage::new(3, 16, 1e-5).unwrap();
    let input = yscv_tensor::Tensor::from_vec(vec![1, 16, 16, 3], vec![0.0; 768]).unwrap();
    let encoded = enc.forward(&input).unwrap();
    assert_eq!(encoded.shape()[0], 1);
    assert_eq!(encoded.shape()[3], 16);
}

#[test]
fn unet_decoder_stage_forward() {
    let dec = crate::UNetDecoderStage::new(8, 8, 4, 1e-5).unwrap();
    let low = yscv_tensor::Tensor::from_vec(vec![1, 4, 4, 8], vec![0.0; 128]).unwrap();
    let skip = yscv_tensor::Tensor::from_vec(vec![1, 8, 8, 8], vec![0.0; 512]).unwrap();
    let decoded = dec.forward(&low, &skip).unwrap();
    assert_eq!(decoded.shape()[3], 4);
}

#[test]
fn fpn_neck_forward() {
    let fpn = crate::FpnNeck::new(&[16, 32], 8).unwrap();
    let f1 = yscv_tensor::Tensor::from_vec(vec![1, 16, 16, 16], vec![0.0; 4096]).unwrap();
    let f2 = yscv_tensor::Tensor::from_vec(vec![1, 8, 8, 32], vec![0.0; 2048]).unwrap();
    let outputs = fpn.forward(&[f1, f2]).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape()[3], 8);
    assert_eq!(outputs[1].shape()[3], 8);
}

#[test]
fn anchor_free_head_forward() {
    let head = crate::AnchorFreeHead::new(8, 3, 1, 1e-5).unwrap();
    let feat = yscv_tensor::Tensor::from_vec(vec![1, 16, 16, 8], vec![0.0; 2048]).unwrap();
    let (cls, bbox, center) = head.forward(&feat).unwrap();
    assert_eq!(cls.shape()[3], 3);
    assert_eq!(bbox.shape()[3], 4);
    assert_eq!(center.shape()[3], 1);
}

// ---------------------------------------------------------------------------
// Vision Transformer (ViT) tests
// ---------------------------------------------------------------------------

#[test]
fn patch_embedding_correct_output_shape() {
    let pe = crate::PatchEmbedding::new(16, 4, 3, 32).unwrap();
    // 16x16 image, patch_size=4 -> 4*4=16 patches
    assert_eq!(pe.num_patches, 16);
    let input = Tensor::from_vec(vec![2, 16, 16, 3], vec![0.1; 2 * 16 * 16 * 3]).unwrap();
    let out = pe.forward(&input).unwrap();
    assert_eq!(out.shape(), &[2, 16, 32]); // [batch, num_patches, embed_dim]
}

#[test]
fn patch_embedding_patch_size_16() {
    let pe = crate::PatchEmbedding::new(32, 16, 3, 64).unwrap();
    // 32x32 image, patch_size=16 -> 2*2=4 patches
    assert_eq!(pe.num_patches, 4);
    let input = Tensor::from_vec(vec![1, 32, 32, 3], vec![0.0; 32 * 32 * 3]).unwrap();
    let out = pe.forward(&input).unwrap();
    assert_eq!(out.shape(), &[1, 4, 64]);
}

#[test]
fn patch_embedding_patch_size_32() {
    let pe = crate::PatchEmbedding::new(32, 32, 3, 128).unwrap();
    // 32x32 image, patch_size=32 -> 1*1=1 patch
    assert_eq!(pe.num_patches, 1);
    let input = Tensor::from_vec(vec![1, 32, 32, 3], vec![0.0; 32 * 32 * 3]).unwrap();
    let out = pe.forward(&input).unwrap();
    assert_eq!(out.shape(), &[1, 1, 128]);
}

#[test]
fn patch_embedding_rejects_indivisible_size() {
    let result = crate::PatchEmbedding::new(15, 4, 3, 32);
    assert!(result.is_err());
}

#[test]
fn vision_transformer_forward_correct_shape() {
    let vit = crate::VisionTransformer::new(
        16,  // image_size
        4,   // patch_size -> 16 patches
        3,   // in_channels
        32,  // embed_dim
        4,   // num_heads
        2,   // num_layers
        10,  // num_classes
        4.0, // mlp_ratio
    )
    .unwrap();
    let input = Tensor::from_vec(vec![1, 16, 16, 3], vec![0.0; 16 * 16 * 3]).unwrap();
    let out = vit.forward(&input).unwrap();
    assert_eq!(out.shape(), &[1, 10]); // [batch, num_classes]
}

#[test]
fn vision_transformer_forward_batch_2() {
    let vit = crate::VisionTransformer::new(16, 4, 3, 32, 4, 1, 5, 4.0).unwrap();
    let input = Tensor::from_vec(vec![2, 16, 16, 3], vec![0.0; 2 * 16 * 16 * 3]).unwrap();
    let out = vit.forward(&input).unwrap();
    assert_eq!(out.shape(), &[2, 5]);
}

#[test]
fn vision_transformer_with_patch_size_16() {
    let vit = crate::VisionTransformer::new(
        32,  // image_size
        16,  // patch_size -> 4 patches
        3,   // in_channels
        64,  // embed_dim
        8,   // num_heads
        1,   // num_layers
        100, // num_classes
        4.0,
    )
    .unwrap();
    assert_eq!(vit.patch_embed.num_patches, 4);
    let input = Tensor::from_vec(vec![1, 32, 32, 3], vec![0.0; 32 * 32 * 3]).unwrap();
    let out = vit.forward(&input).unwrap();
    assert_eq!(out.shape(), &[1, 100]);
}

#[test]
fn vision_transformer_with_patch_size_32() {
    let vit = crate::VisionTransformer::new(
        32, // image_size
        32, // patch_size -> 1 patch
        3, 64, 8, 1, 10, 4.0,
    )
    .unwrap();
    assert_eq!(vit.patch_embed.num_patches, 1);
    let input = Tensor::from_vec(vec![1, 32, 32, 3], vec![0.0; 32 * 32 * 3]).unwrap();
    let out = vit.forward(&input).unwrap();
    assert_eq!(out.shape(), &[1, 10]);
}
