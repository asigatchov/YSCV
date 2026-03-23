use std::time::Duration;

use yscv_eval::{BenchmarkViolation, TimingStats};

#[derive(Debug, Default)]
pub struct BenchmarkCollector {
    pub detect: Vec<Duration>,
    pub track: Vec<Duration>,
    pub recognize: Vec<Duration>,
    pub end_to_end: Vec<Duration>,
}

pub fn format_benchmark_report(report: &yscv_eval::PipelineBenchmarkReport) -> String {
    let mut output = String::new();
    output.push_str("benchmark_report_v1\n");
    output.push_str(&format!("frames={}\n", report.frames));
    push_stage_stats(&mut output, "detect", &report.detect);
    push_stage_stats(&mut output, "track", &report.track);
    push_stage_stats(&mut output, "recognize", &report.recognize);
    push_stage_stats(&mut output, "end_to_end", &report.end_to_end);
    output
}

fn push_stage_stats(output: &mut String, name: &str, stats: &TimingStats) {
    output.push_str(&format!(
        "stage={name} runs={} total_ms={:.3} mean_ms={:.3} min_ms={:.3} max_ms={:.3} p50_ms={:.3} p95_ms={:.3} fps={:.3}\n",
        stats.runs,
        stats.total_ms,
        stats.mean_ms,
        stats.min_ms,
        stats.max_ms,
        stats.p50_ms,
        stats.p95_ms,
        stats.throughput_fps,
    ));
}

pub fn format_benchmark_violations(violations: &[BenchmarkViolation]) -> String {
    let mut output = String::new();
    for violation in violations {
        output.push_str(&format!(
            "  - {}.{} expected {:.3}, observed {:.3}\n",
            violation.stage, violation.metric, violation.expected, violation.observed
        ));
    }
    output
}
