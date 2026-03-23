use super::*;

#[test]
fn summarize_durations_computes_expected_stats() {
    let durations = [
        Duration::from_millis(10),
        Duration::from_millis(20),
        Duration::from_millis(30),
        Duration::from_millis(40),
        Duration::from_millis(50),
    ];
    let stats = summarize_durations(&durations).unwrap();
    assert_eq!(stats.runs, 5);
    approx_eq64(stats.total_ms, 150.0);
    approx_eq64(stats.mean_ms, 30.0);
    approx_eq64(stats.min_ms, 10.0);
    approx_eq64(stats.max_ms, 50.0);
    approx_eq64(stats.p50_ms, 30.0);
    approx_eq64(stats.p95_ms, 48.0);
    approx_eq64(stats.throughput_fps, 33.333333333333336);
}

#[test]
fn summarize_durations_rejects_empty_input() {
    let err = summarize_durations(&[]).unwrap_err();
    assert_eq!(err, EvalError::EmptyDurationSeries);
}

#[test]
fn summarize_pipeline_rejects_length_mismatch() {
    let detect = [Duration::from_millis(1), Duration::from_millis(2)];
    let track = [Duration::from_millis(1)];
    let recognize = [Duration::from_millis(1), Duration::from_millis(2)];
    let end_to_end = [Duration::from_millis(3), Duration::from_millis(4)];

    let err = summarize_pipeline_durations(PipelineDurations {
        detect: &detect,
        track: &track,
        recognize: &recognize,
        end_to_end: &end_to_end,
    })
    .unwrap_err();
    assert_eq!(
        err,
        EvalError::DurationSeriesLengthMismatch {
            expected: 2,
            got: 1,
            series: "track",
        }
    );
}

#[test]
fn summarize_pipeline_computes_report() {
    let detect = [Duration::from_millis(2), Duration::from_millis(4)];
    let track = [Duration::from_millis(1), Duration::from_millis(1)];
    let recognize = [Duration::from_millis(3), Duration::from_millis(5)];
    let end_to_end = [Duration::from_millis(7), Duration::from_millis(10)];

    let report = summarize_pipeline_durations(PipelineDurations {
        detect: &detect,
        track: &track,
        recognize: &recognize,
        end_to_end: &end_to_end,
    })
    .unwrap();

    assert_eq!(report.frames, 2);
    assert_eq!(report.detect.runs, 2);
    assert_eq!(report.track.runs, 2);
    assert_eq!(report.recognize.runs, 2);
    assert_eq!(report.end_to_end.runs, 2);
}
