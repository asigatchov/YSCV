use yscv_autograd::{Graph, NodeId};

/// Clips the total norm of gradients for the given nodes in-place.
///
/// Computes the combined norm (controlled by `norm_type`, typically 2.0 for L2)
/// across all gradient tensors for `node_ids`. If the total norm exceeds
/// `max_norm`, every gradient is scaled by `max_norm / total_norm`.
///
/// Returns the computed total norm before clipping (useful for monitoring).
///
/// Nodes without gradients are silently skipped.
/// If `max_norm` is not positive or `node_ids` is empty, no clipping is
/// performed and the function returns 0.0.
pub fn clip_grad_norm_(
    graph: &mut Graph,
    node_ids: &[NodeId],
    max_norm: f32,
    norm_type: f32,
) -> f32 {
    if node_ids.is_empty() || !max_norm.is_finite() || max_norm <= 0.0 {
        return 0.0;
    }

    // Accumulate the total norm across all gradient tensors (read-only pass).
    let mut total_norm: f32 = if norm_type == f32::INFINITY {
        // Inf-norm: max absolute value across all gradients.
        let mut max_val: f32 = 0.0;
        for &id in node_ids {
            if let Ok(Some(grad)) = graph.grad(id) {
                for &v in grad.data() {
                    let abs = v.abs();
                    if abs > max_val {
                        max_val = abs;
                    }
                }
            }
        }
        max_val
    } else {
        // General p-norm: (sum |g_i|^p)^(1/p).
        let mut acc: f32 = 0.0;
        for &id in node_ids {
            if let Ok(Some(grad)) = graph.grad(id) {
                for &v in grad.data() {
                    acc += v.abs().powf(norm_type);
                }
            }
        }
        acc.powf(1.0 / norm_type)
    };

    if !total_norm.is_finite() {
        total_norm = 0.0;
    }

    // Scale gradients in-place if total norm exceeds max_norm.
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for &id in node_ids {
            if let Ok(Some(grad)) = graph.grad_mut(id) {
                for v in grad.data_mut() {
                    *v *= scale;
                }
            }
        }
    }

    total_norm
}

/// Clamps every gradient element to the range `[-max_val, max_val]` in-place.
///
/// Nodes without gradients are silently skipped.
/// If `max_val` is not positive or `node_ids` is empty, no clamping is performed.
pub fn clip_grad_value_(graph: &mut Graph, node_ids: &[NodeId], max_val: f32) {
    if node_ids.is_empty() || !max_val.is_finite() || max_val <= 0.0 {
        return;
    }

    for &id in node_ids {
        if let Ok(Some(grad)) = graph.grad_mut(id) {
            for v in grad.data_mut() {
                if *v > max_val {
                    *v = max_val;
                } else if *v < -max_val {
                    *v = -max_val;
                }
            }
        }
    }
}
