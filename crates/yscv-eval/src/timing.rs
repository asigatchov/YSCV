use std::time::Duration;

use crate::EvalError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingStats {
    pub runs: usize,
    pub total_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub throughput_fps: f64,
}

pub fn summarize_durations(durations: &[Duration]) -> Result<TimingStats, EvalError> {
    if durations.is_empty() {
        return Err(EvalError::EmptyDurationSeries);
    }

    let mut ms_values = durations
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    ms_values.sort_by(|left, right| left.total_cmp(right));

    let runs = ms_values.len();
    let total_ms = ms_values.iter().copied().sum::<f64>();
    let min_ms = ms_values[0];
    let max_ms = ms_values[runs - 1];
    let mean_ms = total_ms / runs as f64;
    let p50_ms = percentile(&ms_values, 50.0);
    let p95_ms = percentile(&ms_values, 95.0);
    let throughput_fps = if total_ms <= 0.0 {
        0.0
    } else {
        runs as f64 / (total_ms / 1_000.0)
    };

    Ok(TimingStats {
        runs,
        total_ms,
        mean_ms,
        min_ms,
        max_ms,
        p50_ms,
        p95_ms,
        throughput_fps,
    })
}

fn percentile(sorted_ms: &[f64], percentile: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    if sorted_ms.len() == 1 {
        return sorted_ms[0];
    }

    let clamped = percentile.clamp(0.0, 100.0) / 100.0;
    let max_index = (sorted_ms.len() - 1) as f64;
    let rank = clamped * max_index;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted_ms[lower]
    } else {
        let weight = rank - lower as f64;
        sorted_ms[lower] * (1.0 - weight) + sorted_ms[upper] * weight
    }
}
