/// Result of an LR range test.
#[derive(Debug, Clone)]
pub struct LrFinderResult {
    /// LR values used at each step.
    pub lr_values: Vec<f32>,
    /// Loss values recorded at each step (smoothed).
    pub loss_values: Vec<f32>,
    /// Suggested LR (steepest loss decrease).
    pub suggested_lr: f32,
}

/// Configuration for LR range test.
#[derive(Debug, Clone)]
pub struct LrFinderConfig {
    /// Starting learning rate.
    pub start_lr: f32,
    /// Ending learning rate.
    pub end_lr: f32,
    /// Number of steps.
    pub num_steps: usize,
    /// Whether to use exponential (`true`) or linear (`false`) schedule.
    pub log_scale: bool,
    /// Smoothing factor for loss (exponential moving average).
    pub smoothing: f32,
}

impl Default for LrFinderConfig {
    fn default() -> Self {
        Self {
            start_lr: 1e-7,
            end_lr: 10.0,
            num_steps: 100,
            log_scale: true,
            smoothing: 0.05,
        }
    }
}

/// Run an LR range test.
///
/// This generates a schedule of learning rates and calls `compute_loss` for each
/// one. The user-supplied closure should set the optimiser LR and run one training
/// step, returning the resulting loss.
///
/// The function applies exponential-moving-average smoothing, then finds the LR
/// at which the smoothed loss decreased most steeply.
pub fn lr_range_test<F>(config: &LrFinderConfig, mut compute_loss: F) -> LrFinderResult
where
    F: FnMut(f32) -> f32,
{
    assert!(config.num_steps >= 2, "num_steps must be at least 2");

    // Generate LR values.
    let lr_values: Vec<f32> = (0..config.num_steps)
        .map(|i| {
            let t = i as f32 / (config.num_steps - 1) as f32;
            if config.log_scale {
                let log_start = config.start_lr.ln();
                let log_end = config.end_lr.ln();
                (log_start + t * (log_end - log_start)).exp()
            } else {
                config.start_lr + t * (config.end_lr - config.start_lr)
            }
        })
        .collect();

    // Collect raw losses.
    let raw_losses: Vec<f32> = lr_values.iter().map(|&lr| compute_loss(lr)).collect();

    // Apply exponential moving average smoothing.
    let beta = config.smoothing;
    let mut smoothed = Vec::with_capacity(config.num_steps);
    smoothed.push(raw_losses[0]);
    for i in 1..config.num_steps {
        let s = beta * raw_losses[i] + (1.0 - beta) * smoothed[i - 1];
        smoothed.push(s);
    }

    // Find steepest loss decrease (max negative derivative).
    let mut best_idx = 0usize;
    let mut best_drop = f32::INFINITY; // most negative = best
    for i in 1..config.num_steps {
        let drop = smoothed[i] - smoothed[i - 1];
        if drop < best_drop {
            best_drop = drop;
            best_idx = i;
        }
    }

    LrFinderResult {
        suggested_lr: lr_values[best_idx],
        lr_values,
        loss_values: smoothed,
    }
}
