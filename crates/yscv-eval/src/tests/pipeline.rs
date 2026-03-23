use super::*;

#[test]
fn parse_pipeline_thresholds_parses_supported_keys() {
    let text = r#"
            # comment
            detect.max_mean_ms=10
            detect.max_p95_ms=15
            end_to_end.min_fps=25
        "#;
    let thresholds = parse_pipeline_benchmark_thresholds(text).unwrap();
    assert_eq!(
        thresholds,
        PipelineBenchmarkThresholds {
            detect: StageThresholds {
                min_fps: None,
                max_mean_ms: Some(10.0),
                max_p95_ms: Some(15.0),
            },
            track: StageThresholds::default(),
            recognize: StageThresholds::default(),
            end_to_end: StageThresholds {
                min_fps: Some(25.0),
                max_mean_ms: None,
                max_p95_ms: None,
            },
        }
    );
}

#[test]
fn parse_pipeline_thresholds_rejects_bad_entries() {
    let err = parse_pipeline_benchmark_thresholds("bad").unwrap_err();
    assert_eq!(
        err,
        EvalError::InvalidThresholdEntry {
            line: 1,
            message: "expected `stage.metric=value` entry".to_string(),
        }
    );

    let err = parse_pipeline_benchmark_thresholds("unknown.max_mean_ms=1").unwrap_err();
    assert_eq!(
        err,
        EvalError::InvalidThresholdEntry {
            line: 1,
            message: "unsupported stage `unknown`".to_string(),
        }
    );
}

#[test]
fn validate_pipeline_thresholds_reports_violations() {
    let report = PipelineBenchmarkReport {
        frames: 4,
        detect: TimingStats {
            runs: 4,
            total_ms: 40.0,
            mean_ms: 10.0,
            min_ms: 8.0,
            max_ms: 12.0,
            p50_ms: 10.0,
            p95_ms: 11.8,
            throughput_fps: 100.0,
        },
        track: TimingStats {
            runs: 4,
            total_ms: 4.0,
            mean_ms: 1.0,
            min_ms: 1.0,
            max_ms: 1.0,
            p50_ms: 1.0,
            p95_ms: 1.0,
            throughput_fps: 1000.0,
        },
        recognize: TimingStats {
            runs: 4,
            total_ms: 8.0,
            mean_ms: 2.0,
            min_ms: 2.0,
            max_ms: 2.0,
            p50_ms: 2.0,
            p95_ms: 2.0,
            throughput_fps: 500.0,
        },
        end_to_end: TimingStats {
            runs: 4,
            total_ms: 80.0,
            mean_ms: 20.0,
            min_ms: 18.0,
            max_ms: 22.0,
            p50_ms: 20.0,
            p95_ms: 21.8,
            throughput_fps: 50.0,
        },
    };
    let thresholds = PipelineBenchmarkThresholds {
        detect: StageThresholds {
            min_fps: Some(110.0),
            max_mean_ms: Some(9.0),
            max_p95_ms: None,
        },
        track: StageThresholds::default(),
        recognize: StageThresholds::default(),
        end_to_end: StageThresholds {
            min_fps: Some(40.0),
            max_mean_ms: None,
            max_p95_ms: Some(21.0),
        },
    };
    let violations = validate_pipeline_benchmark_thresholds(&report, &thresholds);
    assert_eq!(violations.len(), 3);
    assert_eq!(violations[0].stage, "detect");
    assert_eq!(violations[1].stage, "detect");
    assert_eq!(violations[2].stage, "end_to_end");
}
