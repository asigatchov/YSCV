use yscv_kernels::{LayerNormLastDimParams, layer_norm_last_dim, matmul_2d};
use yscv_tensor::Tensor;

use crate::{ModelError, SequentialModel, TransformerEncoderBlock};

/// Adds a ResNet-style residual block to a SequentialModel (inference-mode).
///
/// Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN, then adds skip connection.
/// Input/output channels must match (`channels`). Operates in NHWC.
pub fn add_residual_block(
    model: &mut SequentialModel,
    channels: usize,
    epsilon: f32,
) -> Result<(), ModelError> {
    let kh = 3;
    let kw = 3;

    model.add_conv2d_zero(channels, channels, kh, kw, 1, 1, true)?;
    model.add_batch_norm2d_identity(channels, epsilon)?;
    model.add_relu();
    model.add_conv2d_zero(channels, channels, kh, kw, 1, 1, true)?;
    model.add_batch_norm2d_identity(channels, epsilon)?;

    Ok(())
}

/// Adds a MobileNetV2-style inverted bottleneck block to a SequentialModel.
///
/// Structure: Conv1x1 (expand) -> BN -> ReLU -> DepthwiseConv3x3 -> BN -> ReLU -> Conv1x1 (project) -> BN.
/// Since we don't have depthwise as a layer yet, we approximate with a grouped conv emulation.
/// For now this builds a standard bottleneck: 1x1 expand -> 3x3 conv -> 1x1 project.
pub fn add_bottleneck_block(
    model: &mut SequentialModel,
    in_channels: usize,
    expand_channels: usize,
    out_channels: usize,
    stride: usize,
    epsilon: f32,
) -> Result<(), ModelError> {
    // 1x1 pointwise expansion
    model.add_conv2d_zero(in_channels, expand_channels, 1, 1, 1, 1, false)?;
    model.add_batch_norm2d_identity(expand_channels, epsilon)?;
    model.add_relu();

    // 3x3 spatial convolution
    model.add_conv2d_zero(
        expand_channels,
        expand_channels,
        3,
        3,
        stride,
        stride,
        false,
    )?;
    model.add_batch_norm2d_identity(expand_channels, epsilon)?;
    model.add_relu();

    // 1x1 pointwise projection
    model.add_conv2d_zero(expand_channels, out_channels, 1, 1, 1, 1, false)?;
    model.add_batch_norm2d_identity(out_channels, epsilon)?;

    Ok(())
}

/// Builds a simple CNN classifier architecture for NHWC input.
///
/// Architecture: [Conv->BN->ReLU->MaxPool] x stages -> GlobalAvgPool -> Flatten -> Linear.
/// This is a convenient builder for common CV classification tasks.
pub fn build_simple_cnn_classifier(
    model: &mut SequentialModel,
    graph: &mut yscv_autograd::Graph,
    input_channels: usize,
    num_classes: usize,
    stage_channels: &[usize],
    epsilon: f32,
) -> Result<(), ModelError> {
    let mut ch = input_channels;
    for &out_ch in stage_channels {
        model.add_conv2d_zero(ch, out_ch, 3, 3, 1, 1, true)?;
        model.add_batch_norm2d_identity(out_ch, epsilon)?;
        model.add_relu();
        model.add_max_pool2d(2, 2, 2, 2)?;
        ch = out_ch;
    }
    model.add_global_avg_pool2d();
    model.add_flatten();

    let weight = Tensor::from_vec(vec![ch, num_classes], vec![0.0; ch * num_classes])?;
    let bias = Tensor::from_vec(vec![num_classes], vec![0.0; num_classes])?;
    model.add_linear(graph, ch, num_classes, weight, bias)?;

    Ok(())
}

/// Squeeze-and-Excitation block (inference-mode).
///
/// Channel attention: GlobalAvgPool -> FC(reduce) -> ReLU -> FC(expand) -> Sigmoid -> scale.
/// `reduction_ratio` controls bottleneck (typically 4 or 16).
pub struct SqueezeExciteBlock {
    pub fc_reduce_w: Tensor, // [channels, channels / reduction]
    pub fc_reduce_b: Tensor, // [channels / reduction]
    pub fc_expand_w: Tensor, // [channels / reduction, channels]
    pub fc_expand_b: Tensor, // [channels]
    pub channels: usize,
    pub reduced: usize,
}

impl SqueezeExciteBlock {
    pub fn new(channels: usize, reduction_ratio: usize) -> Result<Self, ModelError> {
        let reduced = (channels / reduction_ratio).max(1);
        Ok(Self {
            fc_reduce_w: Tensor::from_vec(vec![channels, reduced], vec![0.0; channels * reduced])?,
            fc_reduce_b: Tensor::from_vec(vec![reduced], vec![0.0; reduced])?,
            fc_expand_w: Tensor::from_vec(vec![reduced, channels], vec![0.0; reduced * channels])?,
            fc_expand_b: Tensor::from_vec(vec![channels], vec![0.0; channels])?,
            channels,
            reduced,
        })
    }

    /// Forward: input `[N,H,W,C]` -> scaled `[N,H,W,C]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let data = input.data();

        // Global average pool -> [N, C]
        let hw = (h * w) as f32;
        let mut pooled = vec![0.0f32; n * c];
        for b in 0..n {
            for ch in 0..c {
                let mut sum = 0.0f32;
                for y in 0..h {
                    for x in 0..w {
                        sum += data[((b * h + y) * w + x) * c + ch];
                    }
                }
                pooled[b * c + ch] = sum / hw;
            }
        }
        let pooled_t = Tensor::from_vec(vec![n, c], pooled)?;

        // FC reduce + ReLU
        let reduced = yscv_kernels::matmul_2d(&pooled_t, &self.fc_reduce_w)?;
        let reduced = reduced.add(&self.fc_reduce_b.unsqueeze(0)?)?;
        let reduced_data: Vec<f32> = reduced.data().iter().map(|&v| v.max(0.0)).collect();
        let reduced = Tensor::from_vec(vec![n, self.reduced], reduced_data)?;

        // FC expand + Sigmoid
        let expanded = yscv_kernels::matmul_2d(&reduced, &self.fc_expand_w)?;
        let expanded = expanded.add(&self.fc_expand_b.unsqueeze(0)?)?;
        let scale_data: Vec<f32> = expanded
            .data()
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect();

        // Scale input channels
        let mut out = Vec::with_capacity(n * h * w * c);
        for b in 0..n {
            for y in 0..h {
                for x in 0..w {
                    for ch in 0..c {
                        out.push(data[((b * h + y) * w + x) * c + ch] * scale_data[b * c + ch]);
                    }
                }
            }
        }
        Tensor::from_vec(shape.to_vec(), out).map_err(Into::into)
    }
}

/// MBConv block (EfficientNet / MobileNetV2 inverted residual, inference-mode).
///
/// Structure: expand 1x1 -> BN -> activation -> depthwise 3x3 -> BN -> activation -> SE -> project 1x1 -> BN.
/// Supports optional skip connection when stride=1 and in_channels=out_channels.
pub struct MbConvBlock {
    pub expand_conv: Option<crate::Conv2dLayer>,
    pub expand_bn: Option<crate::BatchNorm2dLayer>,
    pub depthwise_w: Tensor, // [kh, kw, expanded_ch, 1]
    pub depthwise_bn: crate::BatchNorm2dLayer,
    pub se: Option<SqueezeExciteBlock>,
    pub project_conv: crate::Conv2dLayer,
    pub project_bn: crate::BatchNorm2dLayer,
    pub use_residual: bool,
    pub expanded_ch: usize,
}

impl MbConvBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        expand_ratio: usize,
        kernel_size: usize,
        stride: usize,
        se_ratio: Option<usize>,
        epsilon: f32,
    ) -> Result<Self, ModelError> {
        let expanded_ch = in_channels * expand_ratio;
        let use_residual = stride == 1 && in_channels == out_channels;

        let (expand_conv, expand_bn) = if expand_ratio != 1 {
            let w = Tensor::from_vec(
                vec![1, 1, in_channels, expanded_ch],
                vec![0.0; in_channels * expanded_ch],
            )?;
            let b = Tensor::from_vec(vec![expanded_ch], vec![0.0; expanded_ch])?;
            (
                Some(crate::Conv2dLayer::new(
                    in_channels,
                    expanded_ch,
                    1,
                    1,
                    1,
                    1,
                    w,
                    Some(b),
                )?),
                Some(crate::BatchNorm2dLayer::identity_init(
                    expanded_ch,
                    epsilon,
                )?),
            )
        } else {
            (None, None)
        };

        let depthwise_w = Tensor::from_vec(
            vec![kernel_size, kernel_size, expanded_ch, 1],
            vec![0.0; kernel_size * kernel_size * expanded_ch],
        )?;
        let depthwise_bn = crate::BatchNorm2dLayer::identity_init(expanded_ch, epsilon)?;

        let se = se_ratio
            .map(|r| SqueezeExciteBlock::new(expanded_ch, r))
            .transpose()?;

        let proj_w = Tensor::from_vec(
            vec![1, 1, expanded_ch, out_channels],
            vec![0.0; expanded_ch * out_channels],
        )?;
        let proj_b = Tensor::from_vec(vec![out_channels], vec![0.0; out_channels])?;
        let project_conv =
            crate::Conv2dLayer::new(expanded_ch, out_channels, 1, 1, 1, 1, proj_w, Some(proj_b))?;
        let project_bn = crate::BatchNorm2dLayer::identity_init(out_channels, epsilon)?;

        Ok(Self {
            expand_conv,
            expand_bn,
            depthwise_w,
            depthwise_bn,
            se,
            project_conv,
            project_bn,
            use_residual,
            expanded_ch,
        })
    }

    /// Forward inference: input `[N,H,W,C_in]` -> `[N,H',W',C_out]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let mut x = input.clone();

        // Expand phase
        if let (Some(conv), Some(bn)) = (&self.expand_conv, &self.expand_bn) {
            x = conv.forward_inference(&x)?;
            x = bn.forward_inference(&x)?;
            let data: Vec<f32> = x.data().iter().map(|&v| v.clamp(0.0, 6.0)).collect();
            x = Tensor::from_vec(x.shape().to_vec(), data)?;
        }

        // Depthwise conv (using kernel directly)
        x = yscv_kernels::depthwise_conv2d_nhwc(&x, &self.depthwise_w, None, 1, 1)?;
        x = self.depthwise_bn.forward_inference(&x)?;
        let data: Vec<f32> = x.data().iter().map(|&v| v.clamp(0.0, 6.0)).collect();
        x = Tensor::from_vec(x.shape().to_vec(), data)?;

        // SE
        if let Some(se) = &self.se {
            x = se.forward(&x)?;
        }

        // Project
        x = self.project_conv.forward_inference(&x)?;
        x = self.project_bn.forward_inference(&x)?;

        // Residual skip
        if self.use_residual {
            x = x.add(input)?;
        }

        Ok(x)
    }
}

/// Builds a ResNet-like feature extractor (no final classifier).
///
/// Architecture: initial Conv7x7->BN->ReLU->MaxPool, then residual stages.
pub fn build_resnet_feature_extractor(
    model: &mut SequentialModel,
    input_channels: usize,
    stage_channels: &[usize],
    blocks_per_stage: usize,
    epsilon: f32,
) -> Result<(), ModelError> {
    let initial_ch = stage_channels.first().copied().unwrap_or(64);

    // Stem: Conv7x7 stride 2 -> BN -> ReLU -> MaxPool
    model.add_conv2d_zero(input_channels, initial_ch, 7, 7, 2, 2, true)?;
    model.add_batch_norm2d_identity(initial_ch, epsilon)?;
    model.add_relu();
    model.add_max_pool2d(3, 3, 2, 2)?;

    let mut ch = initial_ch;
    for &stage_ch in stage_channels {
        if stage_ch != ch {
            // Channel transition: 1x1 conv to match dimensions
            model.add_conv2d_zero(ch, stage_ch, 1, 1, 1, 1, false)?;
            model.add_batch_norm2d_identity(stage_ch, epsilon)?;
            model.add_relu();
        }
        for _ in 0..blocks_per_stage {
            add_residual_block(model, stage_ch, epsilon)?;
        }
        ch = stage_ch;
    }

    model.add_global_avg_pool2d();
    model.add_flatten();

    Ok(())
}

/// UNet encoder stage (inference-mode, NHWC).
///
/// Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU.
/// Returns the number of output channels for skip-connection bookkeeping.
pub struct UNetEncoderStage {
    conv1: crate::Conv2dLayer,
    bn1: crate::BatchNorm2dLayer,
    conv2: crate::Conv2dLayer,
    bn2: crate::BatchNorm2dLayer,
}

impl UNetEncoderStage {
    pub fn new(in_ch: usize, out_ch: usize, epsilon: f32) -> Result<Self, ModelError> {
        let w1 = Tensor::from_vec(vec![3, 3, in_ch, out_ch], vec![0.0; 9 * in_ch * out_ch])?;
        let b1 = Tensor::from_vec(vec![out_ch], vec![0.0; out_ch])?;
        let w2 = Tensor::from_vec(vec![3, 3, out_ch, out_ch], vec![0.0; 9 * out_ch * out_ch])?;
        let b2 = Tensor::from_vec(vec![out_ch], vec![0.0; out_ch])?;
        Ok(Self {
            conv1: crate::Conv2dLayer::new(in_ch, out_ch, 3, 3, 1, 1, w1, Some(b1))?,
            bn1: crate::BatchNorm2dLayer::identity_init(out_ch, epsilon)?,
            conv2: crate::Conv2dLayer::new(out_ch, out_ch, 3, 3, 1, 1, w2, Some(b2))?,
            bn2: crate::BatchNorm2dLayer::identity_init(out_ch, epsilon)?,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let x = self.conv1.forward_inference(input)?;
        let x = self.bn1.forward_inference(&x)?;
        let x = relu_nhwc(&x)?;
        let x = self.conv2.forward_inference(&x)?;
        let x = self.bn2.forward_inference(&x)?;
        relu_nhwc(&x)
    }
}

/// UNet decoder stage (inference-mode, NHWC).
///
/// Structure: nearest-neighbor 2x upsample -> cat(skip) -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU.
pub struct UNetDecoderStage {
    conv1: crate::Conv2dLayer,
    bn1: crate::BatchNorm2dLayer,
    conv2: crate::Conv2dLayer,
    bn2: crate::BatchNorm2dLayer,
}

impl UNetDecoderStage {
    pub fn new(
        in_ch: usize,
        skip_ch: usize,
        out_ch: usize,
        epsilon: f32,
    ) -> Result<Self, ModelError> {
        let cat_ch = in_ch + skip_ch;
        let w1 = Tensor::from_vec(vec![3, 3, cat_ch, out_ch], vec![0.0; 9 * cat_ch * out_ch])?;
        let b1 = Tensor::from_vec(vec![out_ch], vec![0.0; out_ch])?;
        let w2 = Tensor::from_vec(vec![3, 3, out_ch, out_ch], vec![0.0; 9 * out_ch * out_ch])?;
        let b2 = Tensor::from_vec(vec![out_ch], vec![0.0; out_ch])?;
        Ok(Self {
            conv1: crate::Conv2dLayer::new(cat_ch, out_ch, 3, 3, 1, 1, w1, Some(b1))?,
            bn1: crate::BatchNorm2dLayer::identity_init(out_ch, epsilon)?,
            conv2: crate::Conv2dLayer::new(out_ch, out_ch, 3, 3, 1, 1, w2, Some(b2))?,
            bn2: crate::BatchNorm2dLayer::identity_init(out_ch, epsilon)?,
        })
    }

    /// Forward: `upsampled` is the feature from the lower level, `skip` from the encoder.
    pub fn forward(&self, upsampled: &Tensor, skip: &Tensor) -> Result<Tensor, ModelError> {
        let up = upsample_nearest_2x_nhwc(upsampled)?;
        let cat = cat_nhwc_channel(&up, skip)?;
        let x = self.conv1.forward_inference(&cat)?;
        let x = self.bn1.forward_inference(&x)?;
        let x = relu_nhwc(&x)?;
        let x = self.conv2.forward_inference(&x)?;
        let x = self.bn2.forward_inference(&x)?;
        relu_nhwc(&x)
    }
}

/// Feature Pyramid Network lateral + top-down pathway (inference-mode, NHWC).
///
/// Reduces each backbone level to `out_channels` via 1x1 conv, then top-down
/// merges with 2x nearest-neighbor upsample + element-wise add + 3x3 smoothing.
pub struct FpnNeck {
    lateral_convs: Vec<crate::Conv2dLayer>,
    smooth_convs: Vec<crate::Conv2dLayer>,
    num_levels: usize,
}

impl FpnNeck {
    pub fn new(in_channels: &[usize], out_channels: usize) -> Result<Self, ModelError> {
        let mut lateral_convs = Vec::with_capacity(in_channels.len());
        let mut smooth_convs = Vec::with_capacity(in_channels.len());
        for &ch in in_channels {
            let w = Tensor::from_vec(vec![1, 1, ch, out_channels], vec![0.0; ch * out_channels])?;
            let b = Tensor::from_vec(vec![out_channels], vec![0.0; out_channels])?;
            lateral_convs.push(crate::Conv2dLayer::new(
                ch,
                out_channels,
                1,
                1,
                1,
                1,
                w,
                Some(b),
            )?);

            let w3 = Tensor::from_vec(
                vec![3, 3, out_channels, out_channels],
                vec![0.0; 9 * out_channels * out_channels],
            )?;
            let b3 = Tensor::from_vec(vec![out_channels], vec![0.0; out_channels])?;
            smooth_convs.push(crate::Conv2dLayer::new(
                out_channels,
                out_channels,
                3,
                3,
                1,
                1,
                w3,
                Some(b3),
            )?);
        }
        Ok(Self {
            lateral_convs,
            smooth_convs,
            num_levels: in_channels.len(),
        })
    }

    /// Forward: `features` is a list of backbone feature maps from finest to coarsest.
    /// Returns FPN outputs at the same spatial resolutions.
    pub fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>, ModelError> {
        if features.len() != self.num_levels {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.num_levels,
                got: vec![features.len()],
            });
        }
        let mut laterals: Vec<Tensor> = Vec::with_capacity(self.num_levels);
        for (i, feat) in features.iter().enumerate() {
            laterals.push(self.lateral_convs[i].forward_inference(feat)?);
        }

        for i in (0..self.num_levels - 1).rev() {
            let up = upsample_nearest_2x_nhwc(&laterals[i + 1])?;
            let shape_i = laterals[i].shape();
            let shape_up = up.shape();
            let min_h = shape_i[1].min(shape_up[1]);
            let min_w = shape_i[2].min(shape_up[2]);
            let cropped_lat = crop_nhwc(&laterals[i], min_h, min_w)?;
            let cropped_up = crop_nhwc(&up, min_h, min_w)?;
            laterals[i] = cropped_lat.add(&cropped_up)?;
        }

        let mut outputs = Vec::with_capacity(self.num_levels);
        for (i, lat) in laterals.iter().enumerate() {
            outputs.push(self.smooth_convs[i].forward_inference(lat)?);
        }
        Ok(outputs)
    }
}

/// Anchor-free detection head (FCOS-style, inference-mode, NHWC).
///
/// Per-pixel classification + centerness + bbox regression.
/// Operates on a single FPN level feature map.
pub struct AnchorFreeHead {
    cls_convs: Vec<(crate::Conv2dLayer, crate::BatchNorm2dLayer)>,
    reg_convs: Vec<(crate::Conv2dLayer, crate::BatchNorm2dLayer)>,
    cls_out: crate::Conv2dLayer,
    reg_out: crate::Conv2dLayer,
    centerness_out: crate::Conv2dLayer,
}

impl AnchorFreeHead {
    pub fn new(
        in_channels: usize,
        num_classes: usize,
        num_convs: usize,
        epsilon: f32,
    ) -> Result<Self, ModelError> {
        let mut cls_convs = Vec::with_capacity(num_convs);
        let mut reg_convs = Vec::with_capacity(num_convs);
        let mut ch = in_channels;
        for _ in 0..num_convs {
            let wc =
                Tensor::from_vec(vec![3, 3, ch, in_channels], vec![0.0; 9 * ch * in_channels])?;
            let bc = Tensor::from_vec(vec![in_channels], vec![0.0; in_channels])?;
            let bnc = crate::BatchNorm2dLayer::identity_init(in_channels, epsilon)?;
            cls_convs.push((
                crate::Conv2dLayer::new(ch, in_channels, 3, 3, 1, 1, wc, Some(bc))?,
                bnc,
            ));

            let wr =
                Tensor::from_vec(vec![3, 3, ch, in_channels], vec![0.0; 9 * ch * in_channels])?;
            let br = Tensor::from_vec(vec![in_channels], vec![0.0; in_channels])?;
            let bnr = crate::BatchNorm2dLayer::identity_init(in_channels, epsilon)?;
            reg_convs.push((
                crate::Conv2dLayer::new(ch, in_channels, 3, 3, 1, 1, wr, Some(br))?,
                bnr,
            ));
            ch = in_channels;
        }

        let wco = Tensor::from_vec(
            vec![3, 3, in_channels, num_classes],
            vec![0.0; 9 * in_channels * num_classes],
        )?;
        let bco = Tensor::from_vec(vec![num_classes], vec![0.0; num_classes])?;
        let cls_out =
            crate::Conv2dLayer::new(in_channels, num_classes, 3, 3, 1, 1, wco, Some(bco))?;

        let wro = Tensor::from_vec(vec![3, 3, in_channels, 4], vec![0.0; 9 * in_channels * 4])?;
        let bro = Tensor::from_vec(vec![4], vec![0.0; 4])?;
        let reg_out = crate::Conv2dLayer::new(in_channels, 4, 3, 3, 1, 1, wro, Some(bro))?;

        let wcn = Tensor::from_vec(vec![3, 3, in_channels, 1], vec![0.0; 9 * in_channels])?;
        let bcn = Tensor::from_vec(vec![1], vec![0.0; 1])?;
        let centerness_out = crate::Conv2dLayer::new(in_channels, 1, 3, 3, 1, 1, wcn, Some(bcn))?;

        Ok(Self {
            cls_convs,
            reg_convs,
            cls_out,
            reg_out,
            centerness_out,
        })
    }

    /// Forward on single feature map `[N, H, W, C]`.
    /// Returns `(cls_logits [N,H,W,num_classes], bbox_pred [N,H,W,4], centerness [N,H,W,1])`.
    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor, Tensor), ModelError> {
        let mut cls_feat = input.clone();
        for (conv, bn) in &self.cls_convs {
            cls_feat = conv.forward_inference(&cls_feat)?;
            cls_feat = bn.forward_inference(&cls_feat)?;
            cls_feat = relu_nhwc(&cls_feat)?;
        }

        let mut reg_feat = input.clone();
        for (conv, bn) in &self.reg_convs {
            reg_feat = conv.forward_inference(&reg_feat)?;
            reg_feat = bn.forward_inference(&reg_feat)?;
            reg_feat = relu_nhwc(&reg_feat)?;
        }

        let cls_logits = self.cls_out.forward_inference(&cls_feat)?;
        let bbox_pred = self.reg_out.forward_inference(&reg_feat)?;
        let centerness = self.centerness_out.forward_inference(&cls_feat)?;

        Ok((cls_logits, bbox_pred, centerness))
    }
}

fn relu_nhwc(t: &Tensor) -> Result<Tensor, ModelError> {
    let data: Vec<f32> = t.data().iter().map(|&v| v.max(0.0)).collect();
    Tensor::from_vec(t.shape().to_vec(), data).map_err(Into::into)
}

fn upsample_nearest_2x_nhwc(t: &Tensor) -> Result<Tensor, ModelError> {
    let shape = t.shape();
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let new_h = h * 2;
    let new_w = w * 2;
    let data = t.data();
    let mut out = vec![0.0f32; n * new_h * new_w * c];
    for b in 0..n {
        for y in 0..new_h {
            for x in 0..new_w {
                let sy = y / 2;
                let sx = x / 2;
                let src_off = ((b * h + sy) * w + sx) * c;
                let dst_off = ((b * new_h + y) * new_w + x) * c;
                out[dst_off..dst_off + c].copy_from_slice(&data[src_off..src_off + c]);
            }
        }
    }
    Tensor::from_vec(vec![n, new_h, new_w, c], out).map_err(Into::into)
}

fn cat_nhwc_channel(a: &Tensor, b: &Tensor) -> Result<Tensor, ModelError> {
    let sa = a.shape();
    let sb = b.shape();
    let (n, h, w) = (sa[0], sa[1], sa[2]);
    let ca = sa[3];
    let cb = sb[3];
    let da = a.data();
    let db = b.data();
    let mut out = vec![0.0f32; n * h * w * (ca + cb)];
    for b_idx in 0..n {
        for y in 0..h {
            for x in 0..w {
                let src_a = ((b_idx * h + y) * w + x) * ca;
                let src_b = ((b_idx * h + y) * w + x) * cb;
                let dst = ((b_idx * h + y) * w + x) * (ca + cb);
                out[dst..dst + ca].copy_from_slice(&da[src_a..src_a + ca]);
                out[dst + ca..dst + ca + cb].copy_from_slice(&db[src_b..src_b + cb]);
            }
        }
    }
    Tensor::from_vec(vec![n, h, w, ca + cb], out).map_err(Into::into)
}

fn crop_nhwc(t: &Tensor, target_h: usize, target_w: usize) -> Result<Tensor, ModelError> {
    let shape = t.shape();
    let (n, _h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let data = t.data();
    let mut out = vec![0.0f32; n * target_h * target_w * c];
    for b in 0..n {
        for y in 0..target_h {
            let src_off = ((b * _h + y) * w) * c;
            let dst_off = ((b * target_h + y) * target_w) * c;
            for x in 0..target_w {
                let so = src_off + x * c;
                let do_ = dst_off + x * c;
                out[do_..do_ + c].copy_from_slice(&data[so..so + c]);
            }
        }
    }
    Tensor::from_vec(vec![n, target_h, target_w, c], out).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Vision Transformer (ViT) components
// ---------------------------------------------------------------------------

/// Patch embedding layer for Vision Transformer.
///
/// Splits an NHWC image into non-overlapping patches and projects each patch
/// to an embedding vector via a linear projection (equivalent to Conv2d with
/// kernel_size=patch_size, stride=patch_size).
///
/// Input:  `[batch, H, W, C]` (NHWC)
/// Output: `[batch, num_patches, embed_dim]`
pub struct PatchEmbedding {
    /// Linear projection weight: `[patch_size * patch_size * in_channels, embed_dim]`
    pub projection_w: Tensor,
    /// Linear projection bias: `[embed_dim]`
    pub projection_b: Tensor,
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub num_patches: usize,
}

impl PatchEmbedding {
    /// Creates a zero-initialized patch embedding.
    pub fn new(
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
    ) -> Result<Self, ModelError> {
        if !image_size.is_multiple_of(patch_size) {
            return Err(ModelError::InvalidParameterShape {
                parameter: "image_size must be divisible by patch_size",
                expected: vec![image_size, patch_size],
                got: vec![image_size % patch_size],
            });
        }
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let patch_dim = patch_size * patch_size * in_channels;
        Ok(Self {
            projection_w: Tensor::from_vec(
                vec![patch_dim, embed_dim],
                vec![0.0; patch_dim * embed_dim],
            )?,
            projection_b: Tensor::from_vec(vec![embed_dim], vec![0.0; embed_dim])?,
            image_size,
            patch_size,
            in_channels,
            embed_dim,
            num_patches,
        })
    }

    /// Forward: `[batch, H, W, C]` -> `[batch, num_patches, embed_dim]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let ps = self.patch_size;
        let grid_h = h / ps;
        let grid_w = w / ps;
        let num_patches = grid_h * grid_w;
        let patch_dim = ps * ps * c;
        let data = input.data();

        // Extract patches -> [batch, num_patches, patch_dim]
        let mut patches = vec![0.0f32; batch * num_patches * patch_dim];
        for b in 0..batch {
            for gh in 0..grid_h {
                for gw in 0..grid_w {
                    let patch_idx = gh * grid_w + gw;
                    let dst_base = (b * num_patches + patch_idx) * patch_dim;
                    let mut offset = 0;
                    for ph in 0..ps {
                        for pw in 0..ps {
                            let iy = gh * ps + ph;
                            let ix = gw * ps + pw;
                            let src = ((b * h + iy) * w + ix) * c;
                            patches[dst_base + offset..dst_base + offset + c]
                                .copy_from_slice(&data[src..src + c]);
                            offset += c;
                        }
                    }
                }
            }
        }

        // Project: [batch * num_patches, patch_dim] @ [patch_dim, embed_dim] -> [batch * num_patches, embed_dim]
        let patches_t = Tensor::from_vec(vec![batch * num_patches, patch_dim], patches)?;
        let projected = matmul_2d(&patches_t, &self.projection_w)?;
        let projected = projected.add(&self.projection_b.unsqueeze(0)?)?;

        // Reshape to [batch, num_patches, embed_dim]
        projected
            .reshape(vec![batch, num_patches, self.embed_dim])
            .map_err(Into::into)
    }
}

/// Vision Transformer (ViT) for image classification (inference-mode).
///
/// Architecture:
/// 1. Patch embedding
/// 2. Prepend learnable class token
/// 3. Add learnable position embeddings
/// 4. N transformer encoder blocks
/// 5. Layer norm on output
/// 6. Linear classification head on class token
///
/// Input:  `[batch, H, W, C]` (NHWC)
/// Output: `[batch, num_classes]`
pub struct VisionTransformer {
    pub patch_embed: PatchEmbedding,
    /// Class token: `[1, embed_dim]`
    pub cls_token: Tensor,
    /// Position embeddings: `[1, num_patches + 1, embed_dim]`
    pub pos_embed: Tensor,
    /// Transformer encoder blocks
    pub encoder_blocks: Vec<TransformerEncoderBlock>,
    /// Final layer norm gamma: `[embed_dim]`
    pub ln_gamma: Tensor,
    /// Final layer norm beta: `[embed_dim]`
    pub ln_beta: Tensor,
    /// Classification head weight: `[embed_dim, num_classes]`
    pub head_w: Tensor,
    /// Classification head bias: `[num_classes]`
    pub head_b: Tensor,
    pub embed_dim: usize,
    pub num_classes: usize,
}

impl VisionTransformer {
    /// Creates a zero-initialized Vision Transformer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        num_heads: usize,
        num_layers: usize,
        num_classes: usize,
        mlp_ratio: f32,
    ) -> Result<Self, ModelError> {
        let patch_embed = PatchEmbedding::new(image_size, patch_size, in_channels, embed_dim)?;
        let num_patches = patch_embed.num_patches;
        let seq_len = num_patches + 1; // +1 for class token

        let cls_token = Tensor::from_vec(vec![1, embed_dim], vec![0.0; embed_dim])?;
        let pos_embed =
            Tensor::from_vec(vec![1, seq_len, embed_dim], vec![0.0; seq_len * embed_dim])?;

        let d_ff = (embed_dim as f32 * mlp_ratio) as usize;
        let mut encoder_blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            encoder_blocks.push(TransformerEncoderBlock::new(embed_dim, num_heads, d_ff)?);
        }

        let ln_gamma = Tensor::from_vec(vec![embed_dim], vec![1.0; embed_dim])?;
        let ln_beta = Tensor::from_vec(vec![embed_dim], vec![0.0; embed_dim])?;

        let head_w = Tensor::from_vec(
            vec![embed_dim, num_classes],
            vec![0.0; embed_dim * num_classes],
        )?;
        let head_b = Tensor::from_vec(vec![num_classes], vec![0.0; num_classes])?;

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            encoder_blocks,
            ln_gamma,
            ln_beta,
            head_w,
            head_b,
            embed_dim,
            num_classes,
        })
    }

    /// Forward inference: `[batch, H, W, C]` -> `[batch, num_classes]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let batch = input.shape()[0];

        // 1. Patch embedding -> [batch, num_patches, embed_dim]
        let patch_tokens = self.patch_embed.forward(input)?;
        let num_patches = patch_tokens.shape()[1];

        // 2. Prepend class token -> [batch, num_patches+1, embed_dim]
        // Expand cls_token [1, embed_dim] -> [batch, 1, embed_dim]
        let cls_expanded = self.cls_token.repeat(&[batch, 1])?; // [batch, embed_dim]
        let cls_expanded = cls_expanded.reshape(vec![batch, 1, self.embed_dim])?;

        // Concatenate along sequence dimension
        let seq_len = num_patches + 1;
        let patch_data = patch_tokens.data();
        let cls_data = cls_expanded.data();
        let mut combined = vec![0.0f32; batch * seq_len * self.embed_dim];
        for b in 0..batch {
            // Copy class token
            let cls_src = b * self.embed_dim;
            let dst_base = b * seq_len * self.embed_dim;
            combined[dst_base..dst_base + self.embed_dim]
                .copy_from_slice(&cls_data[cls_src..cls_src + self.embed_dim]);
            // Copy patch tokens
            let patch_src = b * num_patches * self.embed_dim;
            let patch_dst = dst_base + self.embed_dim;
            let patch_len = num_patches * self.embed_dim;
            combined[patch_dst..patch_dst + patch_len]
                .copy_from_slice(&patch_data[patch_src..patch_src + patch_len]);
        }
        let mut x = Tensor::from_vec(vec![batch, seq_len, self.embed_dim], combined)?;

        // 3. Add position embeddings (broadcast over batch)
        let pos = self.pos_embed.repeat(&[batch, 1, 1])?;
        x = x.add(&pos)?;

        // 4. Run through transformer encoder blocks
        // TransformerEncoderBlock expects [seq_len, d_model], so process each batch item
        let mut out_data = vec![0.0f32; batch * seq_len * self.embed_dim];
        for b in 0..batch {
            // Extract [seq_len, embed_dim] for this batch
            let start = b * seq_len * self.embed_dim;
            let end = start + seq_len * self.embed_dim;
            let slice = &x.data()[start..end];
            let mut seq = Tensor::from_vec(vec![seq_len, self.embed_dim], slice.to_vec())?;

            for block in &self.encoder_blocks {
                seq = block.forward(&seq)?;
            }

            let seq_data = seq.data();
            out_data[start..end].copy_from_slice(seq_data);
        }
        let x = Tensor::from_vec(vec![batch, seq_len, self.embed_dim], out_data)?;

        // 5. Layer norm on full output
        let x_2d = x.reshape(vec![batch * seq_len, self.embed_dim])?;
        let params = LayerNormLastDimParams {
            gamma: &self.ln_gamma,
            beta: &self.ln_beta,
            epsilon: 1e-5,
        };
        let normed = layer_norm_last_dim(&x_2d, params)?;
        let normed = normed.reshape(vec![batch, seq_len, self.embed_dim])?;

        // 6. Extract class token (index 0 along seq dimension) -> [batch, embed_dim]
        let normed_data = normed.data();
        let mut cls_out = vec![0.0f32; batch * self.embed_dim];
        for b in 0..batch {
            let src = b * seq_len * self.embed_dim;
            cls_out[b * self.embed_dim..(b + 1) * self.embed_dim]
                .copy_from_slice(&normed_data[src..src + self.embed_dim]);
        }
        let cls_features = Tensor::from_vec(vec![batch, self.embed_dim], cls_out)?;

        // 7. Classification head: linear projection
        let logits = matmul_2d(&cls_features, &self.head_w)?;
        let logits = logits.add(&self.head_b.unsqueeze(0)?)?;

        Ok(logits)
    }
}
