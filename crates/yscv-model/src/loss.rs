use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use crate::ModelError;

fn validate_loss_inputs(
    graph: &Graph,
    prediction: NodeId,
    target: NodeId,
) -> Result<usize, ModelError> {
    let prediction_shape = graph.value(prediction)?.shape().to_vec();
    let target_shape = graph.value(target)?.shape().to_vec();
    if prediction_shape != target_shape {
        return Err(ModelError::PredictionTargetShapeMismatch {
            prediction: prediction_shape,
            target: target_shape,
        });
    }

    let element_count = graph.value(prediction)?.len();
    if element_count == 0 {
        return Err(ModelError::EmptyLossTensor);
    }
    Ok(element_count)
}

fn abs_node(graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
    let zero = graph.constant(Tensor::scalar(0.0));
    let neg_input = graph.sub(zero, input)?;
    let positive = graph.relu(input)?;
    let negative = graph.relu(neg_input)?;
    graph.add(positive, negative).map_err(Into::into)
}

/// Mean squared error loss: `mean((prediction - target)^2)`.
pub fn mse_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
) -> Result<NodeId, ModelError> {
    let element_count = validate_loss_inputs(graph, prediction, target)?;

    let diff = graph.sub(prediction, target)?;
    let sq = graph.mul(diff, diff)?;
    let sum = graph.sum(sq)?;
    let inv_count = graph.constant(Tensor::scalar(1.0 / element_count as f32));
    graph.mul(sum, inv_count).map_err(Into::into)
}

/// Mean absolute error loss: `mean(abs(prediction - target))`.
pub fn mae_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
) -> Result<NodeId, ModelError> {
    let element_count = validate_loss_inputs(graph, prediction, target)?;

    let diff = graph.sub(prediction, target)?;
    let abs = abs_node(graph, diff)?;
    let sum = graph.sum(abs)?;
    let inv_count = graph.constant(Tensor::scalar(1.0 / element_count as f32));
    graph.mul(sum, inv_count).map_err(Into::into)
}

/// Mean Huber loss:
/// `mean(0.5 * min(|e|, delta)^2 + delta * max(|e| - delta, 0))`, where `e = prediction - target`.
pub fn huber_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    delta: f32,
) -> Result<NodeId, ModelError> {
    if !delta.is_finite() || delta <= 0.0 {
        return Err(ModelError::InvalidHuberDelta { delta });
    }
    let element_count = validate_loss_inputs(graph, prediction, target)?;

    let diff = graph.sub(prediction, target)?;
    let abs = abs_node(graph, diff)?;
    let delta_node = graph.constant(Tensor::scalar(delta));
    let abs_minus_delta = graph.sub(abs, delta_node)?;
    let excess = graph.relu(abs_minus_delta)?;
    let clipped = graph.sub(abs, excess)?;

    let clipped_sq = graph.mul(clipped, clipped)?;
    let half = graph.constant(Tensor::scalar(0.5));
    let quadratic = graph.mul(clipped_sq, half)?;
    let linear = graph.mul(excess, delta_node)?;
    let per_element = graph.add(quadratic, linear)?;
    let sum = graph.sum(per_element)?;
    let inv_count = graph.constant(Tensor::scalar(1.0 / element_count as f32));
    graph.mul(sum, inv_count).map_err(Into::into)
}

/// Mean hinge loss:
/// `mean(max(0, margin - prediction * target))`.
pub fn hinge_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    margin: f32,
) -> Result<NodeId, ModelError> {
    if !margin.is_finite() || margin <= 0.0 {
        return Err(ModelError::InvalidHingeMargin { margin });
    }
    let element_count = validate_loss_inputs(graph, prediction, target)?;

    let product = graph.mul(prediction, target)?;
    let margin_node = graph.constant(Tensor::scalar(margin));
    let raw = graph.sub(margin_node, product)?;
    let positive = graph.relu(raw)?;
    let sum = graph.sum(positive)?;
    let inv_count = graph.constant(Tensor::scalar(1.0 / element_count as f32));
    graph.mul(sum, inv_count).map_err(Into::into)
}

/// Binary cross-entropy loss for predictions already passed through sigmoid.
/// `bce = -mean(target * log(pred) + (1 - target) * log(1 - pred))`.
///
/// `prediction` values are clamped to `[eps, 1-eps]` for numerical stability.
pub fn bce_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
) -> Result<NodeId, ModelError> {
    let element_count = validate_loss_inputs(graph, prediction, target)?;

    let eps = 1e-7_f32;
    let eps_node = graph.constant(Tensor::scalar(eps));
    let one_node = graph.constant(Tensor::scalar(1.0));

    // pred_safe = clamp(pred, eps, 1-eps) via relu chain
    let shifted_low = graph.sub(prediction, eps_node)?;
    let positive_part = graph.relu(shifted_low)?;
    let pred_above_eps = graph.add(positive_part, eps_node)?;

    let one_minus_eps_node = graph.constant(Tensor::scalar(1.0 - eps));
    let over = graph.sub(pred_above_eps, one_minus_eps_node)?;
    let excess = graph.relu(over)?;
    let pred_safe = graph.sub(pred_above_eps, excess)?;

    // log(pred_safe)
    let log_pred = graph.log(pred_safe)?;

    // log(1 - pred_safe + eps) for stability
    let one_minus_pred = graph.sub(one_node, pred_safe)?;
    let one_minus_pred_safe = graph.add(one_minus_pred, eps_node)?;
    let log_one_minus_pred = graph.log(one_minus_pred_safe)?;

    // -mean(t*log(p) + (1-t)*log(1-p))
    let term1 = graph.mul(target, log_pred)?;
    let one_minus_t = graph.sub(one_node, target)?;
    let term2 = graph.mul(one_minus_t, log_one_minus_pred)?;
    let combined = graph.add(term1, term2)?;
    let sum = graph.sum(combined)?;
    let neg_sum = graph.neg(sum)?;
    let inv_count = graph.constant(Tensor::scalar(1.0 / element_count as f32));
    graph.mul(neg_sum, inv_count).map_err(Into::into)
}

/// Negative log-likelihood loss from log-probabilities.
/// Expects `log_probs` shape `[batch, classes]` and `targets` shape `[batch, 1]`
/// where targets contain class indices as f32.
///
/// `nll = -mean(log_probs[i, target[i]])` across the batch.
pub fn nll_loss(
    graph: &mut Graph,
    log_probs: NodeId,
    targets: NodeId,
) -> Result<NodeId, ModelError> {
    let lp_shape = graph.value(log_probs)?.shape().to_vec();
    let t_shape = graph.value(targets)?.shape().to_vec();

    if lp_shape.len() != 2 {
        return Err(ModelError::InvalidInputShape {
            expected_features: 0,
            got: lp_shape,
        });
    }
    if t_shape.len() != 2 || t_shape[1] != 1 {
        return Err(ModelError::PredictionTargetShapeMismatch {
            prediction: lp_shape.clone(),
            target: t_shape,
        });
    }
    let batch_size = lp_shape[0];
    let num_classes = lp_shape[1];
    if batch_size == 0 {
        return Err(ModelError::EmptyLossTensor);
    }

    let lp_data = graph.value(log_probs)?.data().to_vec();
    let t_data = graph.value(targets)?.data().to_vec();

    let mut selected = vec![0.0f32; batch_size];
    for i in 0..batch_size {
        let class_idx = t_data[i] as usize;
        if class_idx >= num_classes {
            return Err(ModelError::InvalidDatasetRecordValue {
                line: i,
                field: "nll_target",
                index: 0,
                reason: "class index out of range",
            });
        }
        selected[i] = lp_data[i * num_classes + class_idx];
    }

    let selected_node = graph.constant(Tensor::from_vec(vec![batch_size], selected)?);
    let sum = graph.sum(selected_node)?;
    let neg_sum = graph.neg(sum)?;
    let inv_batch = graph.constant(Tensor::scalar(1.0 / batch_size as f32));
    graph.mul(neg_sum, inv_batch).map_err(Into::into)
}

/// Cross-entropy loss from raw logits.
/// Computes `nll_loss(log_softmax(logits), targets)`.
///
/// Expects `logits` shape `[batch, classes]` and `targets` shape `[batch, 1]`
/// with class indices as f32.
pub fn cross_entropy_loss(
    graph: &mut Graph,
    logits: NodeId,
    targets: NodeId,
) -> Result<NodeId, ModelError> {
    let shape = graph.value(logits)?.shape().to_vec();
    if shape.len() != 2 {
        return Err(ModelError::InvalidInputShape {
            expected_features: 0,
            got: shape,
        });
    }
    let batch_size = shape[0];
    let num_classes = shape[1];

    if batch_size == 0 {
        return Err(ModelError::EmptyLossTensor);
    }

    let logits_data = graph.value(logits)?.data().to_vec();
    let t_data = graph.value(targets)?.data().to_vec();
    let t_shape = graph.value(targets)?.shape().to_vec();

    if t_shape.len() != 2 || t_shape[1] != 1 || t_shape[0] != batch_size {
        return Err(ModelError::PredictionTargetShapeMismatch {
            prediction: shape.clone(),
            target: t_shape,
        });
    }

    // Compute log-softmax manually for numerical stability:
    // log_softmax(x_i) = x_i - log(sum(exp(x_j - max_x)))  - max_x
    let mut log_probs = vec![0.0f32; batch_size * num_classes];
    for b in 0..batch_size {
        let row = &logits_data[b * num_classes..(b + 1) * num_classes];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
        let log_sum_exp = max_val + sum_exp.ln();
        for c in 0..num_classes {
            log_probs[b * num_classes + c] = row[c] - log_sum_exp;
        }
    }

    // Gather: pick log_probs[i, target[i]]
    let mut neg_sum = 0.0f32;
    for i in 0..batch_size {
        let class_idx = t_data[i] as usize;
        if class_idx >= num_classes {
            return Err(ModelError::InvalidDatasetRecordValue {
                line: i,
                field: "cross_entropy_target",
                index: 0,
                reason: "class index out of range",
            });
        }
        neg_sum -= log_probs[i * num_classes + class_idx];
    }

    let loss_val = neg_sum / batch_size as f32;
    let loss_node = graph.constant(Tensor::scalar(loss_val));
    Ok(loss_node)
}

/// Focal loss for imbalanced classification.
///
/// `FL = -alpha * (1 - p_t)^gamma * log(p_t)` averaged over elements.
/// `prediction` should be sigmoid probabilities, `target` binary labels.
pub fn focal_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    alpha: f32,
    gamma: f32,
) -> Result<NodeId, ModelError> {
    let element_count = validate_loss_inputs(graph, prediction, target)?;
    let eps = 1e-7_f32;

    let pred_data = graph.value(prediction)?.data().to_vec();
    let target_data = graph.value(target)?.data().to_vec();

    let mut loss_sum = 0.0f32;
    for i in 0..element_count {
        let p = pred_data[i].clamp(eps, 1.0 - eps);
        let t = target_data[i];
        let pt = if t > 0.5 { p } else { 1.0 - p };
        loss_sum += -alpha * (1.0 - pt).powf(gamma) * pt.ln();
    }

    let loss_val = loss_sum / element_count as f32;
    Ok(graph.constant(Tensor::scalar(loss_val)))
}

/// Dice loss for segmentation.
///
/// `DiceLoss = 1 - (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)`
/// where `prediction` and `target` are probability maps.
pub fn dice_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    smooth: f32,
) -> Result<NodeId, ModelError> {
    let element_count = validate_loss_inputs(graph, prediction, target)?;

    let pred_data = graph.value(prediction)?.data().to_vec();
    let target_data = graph.value(target)?.data().to_vec();

    let mut intersection = 0.0f32;
    let mut pred_sum = 0.0f32;
    let mut target_sum = 0.0f32;
    for i in 0..element_count {
        intersection += pred_data[i] * target_data[i];
        pred_sum += pred_data[i];
        target_sum += target_data[i];
    }

    let dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth);
    Ok(graph.constant(Tensor::scalar(1.0 - dice)))
}

/// Triplet loss for metric learning.
///
/// `L = mean(max(0, d(a,p) - d(a,n) + margin))` where d is L2 distance.
/// All inputs must have the same shape `[batch, embedding_dim]`.
pub fn triplet_loss(
    graph: &mut Graph,
    anchor: NodeId,
    positive: NodeId,
    negative: NodeId,
    margin: f32,
) -> Result<NodeId, ModelError> {
    let a_shape = graph.value(anchor)?.shape().to_vec();
    let p_shape = graph.value(positive)?.shape().to_vec();
    let n_shape = graph.value(negative)?.shape().to_vec();
    if a_shape != p_shape || a_shape != n_shape {
        return Err(ModelError::PredictionTargetShapeMismatch {
            prediction: a_shape,
            target: p_shape,
        });
    }
    if a_shape.len() != 2 || a_shape[0] == 0 {
        return Err(ModelError::EmptyLossTensor);
    }
    let batch = a_shape[0];
    let dim = a_shape[1];

    let a_data = graph.value(anchor)?.data().to_vec();
    let p_data = graph.value(positive)?.data().to_vec();
    let n_data = graph.value(negative)?.data().to_vec();

    let mut loss_sum = 0.0f32;
    for b in 0..batch {
        let mut dp = 0.0f32;
        let mut dn = 0.0f32;
        for d in 0..dim {
            let idx = b * dim + d;
            dp += (a_data[idx] - p_data[idx]).powi(2);
            dn += (a_data[idx] - n_data[idx]).powi(2);
        }
        loss_sum += (dp.sqrt() - dn.sqrt() + margin).max(0.0);
    }

    Ok(graph.constant(Tensor::scalar(loss_sum / batch as f32)))
}

/// Contrastive loss for siamese networks.
///
/// `L = mean( y * d^2 + (1-y) * max(0, margin - d)^2 )` where d = L2 distance.
/// `label`: 1.0 for same pair, 0.0 for different.
pub fn contrastive_loss(
    graph: &mut Graph,
    x1: NodeId,
    x2: NodeId,
    label: NodeId,
    margin: f32,
) -> Result<NodeId, ModelError> {
    let s1 = graph.value(x1)?.shape().to_vec();
    let s2 = graph.value(x2)?.shape().to_vec();
    if s1 != s2 || s1.len() != 2 || s1[0] == 0 {
        return Err(ModelError::PredictionTargetShapeMismatch {
            prediction: s1,
            target: s2,
        });
    }
    let batch = s1[0];
    let dim = s1[1];

    let x1d = graph.value(x1)?.data().to_vec();
    let x2d = graph.value(x2)?.data().to_vec();
    let ld = graph.value(label)?.data().to_vec();

    let mut loss_sum = 0.0f32;
    for b in 0..batch {
        let mut dist_sq = 0.0f32;
        for d in 0..dim {
            let idx = b * dim + d;
            dist_sq += (x1d[idx] - x2d[idx]).powi(2);
        }
        let dist = dist_sq.sqrt();
        let y = ld[b];
        loss_sum += y * dist_sq + (1.0 - y) * (margin - dist).max(0.0).powi(2);
    }

    Ok(graph.constant(Tensor::scalar(loss_sum / batch as f32)))
}

/// Cosine embedding loss.
///
/// `L = mean( y==1: 1-cos(x1,x2), y==-1: max(0, cos(x1,x2)-margin) )`
pub fn cosine_embedding_loss(
    graph: &mut Graph,
    x1: NodeId,
    x2: NodeId,
    label: NodeId,
    margin: f32,
) -> Result<NodeId, ModelError> {
    let s1 = graph.value(x1)?.shape().to_vec();
    let s2 = graph.value(x2)?.shape().to_vec();
    if s1 != s2 || s1.len() != 2 || s1[0] == 0 {
        return Err(ModelError::PredictionTargetShapeMismatch {
            prediction: s1,
            target: s2,
        });
    }
    let batch = s1[0];
    let dim = s1[1];

    let x1d = graph.value(x1)?.data().to_vec();
    let x2d = graph.value(x2)?.data().to_vec();
    let ld = graph.value(label)?.data().to_vec();

    let mut loss_sum = 0.0f32;
    for b in 0..batch {
        let mut dot = 0.0f32;
        let mut n1 = 0.0f32;
        let mut n2 = 0.0f32;
        for d in 0..dim {
            let idx = b * dim + d;
            dot += x1d[idx] * x2d[idx];
            n1 += x1d[idx] * x1d[idx];
            n2 += x2d[idx] * x2d[idx];
        }
        let cos = dot / (n1.sqrt() * n2.sqrt()).max(1e-8);
        let y = ld[b];
        if y > 0.0 {
            loss_sum += 1.0 - cos;
        } else {
            loss_sum += (cos - margin).max(0.0);
        }
    }

    Ok(graph.constant(Tensor::scalar(loss_sum / batch as f32)))
}

/// Cross-entropy with label smoothing.
///
/// Smoothed target: `(1 - smoothing) * one_hot + smoothing / num_classes`.
/// Expects `logits` shape `[batch, classes]` and `targets` shape `[batch, 1]`.
pub fn label_smoothing_cross_entropy(
    graph: &mut Graph,
    logits: NodeId,
    targets: NodeId,
    smoothing: f32,
) -> Result<NodeId, ModelError> {
    let shape = graph.value(logits)?.shape().to_vec();
    if shape.len() != 2 || shape[0] == 0 {
        return Err(ModelError::EmptyLossTensor);
    }
    let batch_size = shape[0];
    let num_classes = shape[1];

    let logits_data = graph.value(logits)?.data().to_vec();
    let t_data = graph.value(targets)?.data().to_vec();

    let smooth_val = smoothing / num_classes as f32;
    let confidence = 1.0 - smoothing;

    let mut total_loss = 0.0f32;
    for b in 0..batch_size {
        let row = &logits_data[b * num_classes..(b + 1) * num_classes];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&v| (v - max_val).exp()).sum();
        let log_sum_exp = max_val + sum_exp.ln();

        let class_idx = t_data[b] as usize;
        for c in 0..num_classes {
            let log_prob = row[c] - log_sum_exp;
            let target_prob = if c == class_idx {
                confidence + smooth_val
            } else {
                smooth_val
            };
            total_loss -= target_prob * log_prob;
        }
    }

    Ok(graph.constant(Tensor::scalar(total_loss / batch_size as f32)))
}

/// CTC (Connectionist Temporal Classification) loss.
///
/// Simplified implementation for sequence-to-sequence tasks (OCR/ASR).
/// `log_probs`: `[T, batch, classes]`, `targets`: `[batch, S]`, lengths as 1-D tensors.
pub fn ctc_loss(
    graph: &mut Graph,
    log_probs: NodeId,
    targets: NodeId,
    input_lengths: NodeId,
    target_lengths: NodeId,
    blank: usize,
) -> Result<NodeId, ModelError> {
    let lp_shape = graph.value(log_probs)?.shape().to_vec();
    if lp_shape.len() != 3 {
        return Err(ModelError::InvalidInputShape {
            expected_features: 0,
            got: lp_shape,
        });
    }
    let _t_max = lp_shape[0];
    let batch = lp_shape[1];
    let num_classes = lp_shape[2];

    let lp_data = graph.value(log_probs)?.data().to_vec();
    let tgt_data = graph.value(targets)?.data().to_vec();
    let il_data = graph.value(input_lengths)?.data().to_vec();
    let tl_data = graph.value(target_lengths)?.data().to_vec();

    let tgt_shape = graph.value(targets)?.shape().to_vec();
    let s_max = if tgt_shape.len() >= 2 {
        tgt_shape[1]
    } else {
        tgt_shape[0] / batch
    };

    let mut total_loss = 0.0f32;

    for b in 0..batch {
        let input_len = il_data[b] as usize;
        let target_len = tl_data[b] as usize;

        // Build label sequence with blanks: [blank, l1, blank, l2, blank, ...]
        let label_len = 2 * target_len + 1;
        let mut labels = vec![blank; label_len];
        for s in 0..target_len {
            labels[2 * s + 1] = tgt_data[b * s_max + s] as usize;
        }

        // Forward pass (alpha)
        let mut alpha = vec![f32::NEG_INFINITY; label_len * input_len];
        // Init t=0
        alpha[0] = lp_data[b * num_classes + labels[0]];
        if label_len > 1 {
            alpha[1] = lp_data[b * num_classes + labels[1]];
        }

        for t in 1..input_len {
            for s in 0..label_len {
                let lp_idx = t * batch * num_classes + b * num_classes + labels[s];
                let log_p = lp_data[lp_idx];
                let mut sum = alpha[(t - 1) * label_len + s];
                if s > 0 {
                    sum = log_sum_exp_pair(sum, alpha[(t - 1) * label_len + s - 1]);
                }
                if s > 1 && labels[s] != blank && labels[s] != labels[s - 2] {
                    sum = log_sum_exp_pair(sum, alpha[(t - 1) * label_len + s - 2]);
                }
                alpha[t * label_len + s] = sum + log_p;
            }
        }

        let last_t = input_len - 1;
        let log_likelihood = log_sum_exp_pair(
            alpha[last_t * label_len + label_len - 1],
            if label_len >= 2 {
                alpha[last_t * label_len + label_len - 2]
            } else {
                f32::NEG_INFINITY
            },
        );
        total_loss -= log_likelihood;
    }

    Ok(graph.constant(Tensor::scalar(total_loss / batch as f32)))
}

/// Smooth L1 loss (detection-style parameterization of Huber loss):
///
/// ```text
/// smooth_l1(x, beta) = 0.5 * x^2 / beta   if |x| < beta
///                     = |x| - 0.5 * beta    otherwise
/// ```
///
/// Equivalent to `huber_loss(pred, target, delta=beta) / beta`.
/// `beta` must be positive and finite.
pub fn smooth_l1_loss(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    beta: f32,
) -> Result<NodeId, ModelError> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(ModelError::InvalidHuberDelta { delta: beta });
    }
    let element_count = validate_loss_inputs(graph, prediction, target)?;

    let pred_data = graph.value(prediction)?.data().to_vec();
    let target_data = graph.value(target)?.data().to_vec();

    let mut loss_sum = 0.0f32;
    for i in 0..element_count {
        let x = (pred_data[i] - target_data[i]).abs();
        if x < beta {
            loss_sum += 0.5 * x * x / beta;
        } else {
            loss_sum += x - 0.5 * beta;
        }
    }

    Ok(graph.constant(Tensor::scalar(loss_sum / element_count as f32)))
}

/// KL divergence loss:
///
/// ```text
/// kl_div(log_pred, target) = sum(target * (log(target) - log_pred)) / n
/// ```
///
/// `log_pred` is the log of the predicted distribution (already log-transformed).
/// `target` is the true probability distribution. Target values <= 0 are skipped
/// (their contribution is treated as zero, matching the convention that `0 * log(0) = 0`).
pub fn kl_div_loss(
    graph: &mut Graph,
    log_prediction: NodeId,
    target: NodeId,
) -> Result<NodeId, ModelError> {
    let element_count = validate_loss_inputs(graph, log_prediction, target)?;

    let log_pred_data = graph.value(log_prediction)?.data().to_vec();
    let target_data = graph.value(target)?.data().to_vec();

    let mut loss_sum = 0.0f32;
    for i in 0..element_count {
        let t = target_data[i];
        if t > 0.0 {
            loss_sum += t * (t.ln() - log_pred_data[i]);
        }
    }

    Ok(graph.constant(Tensor::scalar(loss_sum / element_count as f32)))
}

fn log_sum_exp_pair(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

// ---------------------------------------------------------------------------
// Knowledge Distillation
// ---------------------------------------------------------------------------

/// Knowledge distillation loss (Hinton et al., 2015).
///
/// Combines soft target KL divergence with hard target cross-entropy:
///
/// ```text
/// L = alpha * T² * KL(softmax(student/T) || softmax(teacher/T))
///   + (1 - alpha) * CrossEntropy(student, labels)
/// ```
///
/// * `student`:  student logits `[batch, num_classes]`
/// * `teacher`:  teacher logits `[batch, num_classes]` (detached / no grad)
/// * `labels`:   hard labels node `[batch]` (class indices as f32)
/// * `temperature`: softening temperature (typically 3-20)
/// * `alpha`: weight for soft loss (0.0 = pure hard, 1.0 = pure soft)
pub fn distillation_loss(
    graph: &mut Graph,
    student: NodeId,
    teacher: NodeId,
    labels: NodeId,
    temperature: f32,
    alpha: f32,
) -> Result<NodeId, ModelError> {
    // Soft targets: KL(softmax(s/T) || softmax(t/T)) * T²
    let t_scalar = graph.constant(Tensor::scalar(temperature));
    let t2_scalar = graph.constant(Tensor::scalar(temperature * temperature));

    let s_scaled = graph.div(student, t_scalar)?;
    let t_scaled = graph.div(teacher, t_scalar)?;

    let s_log_softmax = graph.log_softmax(s_scaled)?;
    let t_softmax = graph.softmax(t_scaled)?;

    // KL divergence = sum(t_soft * (log(t_soft) - log_s_soft))
    let t_log = graph.log(t_softmax)?;
    let kl_pointwise = graph.sub(t_log, s_log_softmax)?;
    let kl_weighted = graph.mul(t_softmax, kl_pointwise)?;
    let kl_sum = graph.mean(kl_weighted)?;
    let soft_loss = graph.mul(kl_sum, t2_scalar)?;

    // Hard targets: cross-entropy(student, labels)
    let hard_loss = cross_entropy_loss(graph, student, labels)?;

    // Combined: alpha * soft + (1 - alpha) * hard
    let alpha_node = graph.constant(Tensor::scalar(alpha));
    let one_minus_alpha = graph.constant(Tensor::scalar(1.0 - alpha));

    let weighted_soft = graph.mul(soft_loss, alpha_node)?;
    let weighted_hard = graph.mul(hard_loss, one_minus_alpha)?;

    graph.add(weighted_soft, weighted_hard).map_err(Into::into)
}
