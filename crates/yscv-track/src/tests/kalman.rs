use crate::KalmanFilter;
use yscv_detect::BoundingBox;

#[test]
fn kalman_initial_bbox_round_trip() {
    let bbox = BoundingBox {
        x1: 10.0,
        y1: 20.0,
        x2: 30.0,
        y2: 50.0,
    };
    let kf = KalmanFilter::new(bbox);
    let out = kf.bbox();
    assert!((out.x1 - 10.0).abs() < 1e-3);
    assert!((out.y1 - 20.0).abs() < 1e-3);
    assert!((out.x2 - 30.0).abs() < 1e-3);
    assert!((out.y2 - 50.0).abs() < 1e-3);
}

#[test]
fn kalman_predict_moves_state() {
    let bbox = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let mut kf = KalmanFilter::new(bbox);
    kf.predict();
    let out = kf.bbox();
    assert!((out.x1 - 0.0).abs() < 1.0);
    assert!((out.y1 - 0.0).abs() < 1.0);
}

#[test]
fn kalman_update_converges() {
    let bbox = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let mut kf = KalmanFilter::new(bbox);
    for _ in 0..10 {
        kf.predict();
        kf.update([15.0, 15.0, 10.0, 10.0]);
    }
    let out = kf.bbox();
    let cx = (out.x1 + out.x2) * 0.5;
    let cy = (out.y1 + out.y2) * 0.5;
    assert!((cx - 15.0).abs() < 1.0);
    assert!((cy - 15.0).abs() < 1.0);
}

#[test]
fn kalman_predicted_bbox_no_mutation() {
    let bbox = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let kf = KalmanFilter::new(bbox);
    let pred = kf.predicted_bbox();
    let current = kf.bbox();
    assert!((pred.x1 - current.x1).abs() < 1e-3);
}
