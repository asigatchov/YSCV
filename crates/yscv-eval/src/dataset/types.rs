use serde::Deserialize;
use yscv_detect::{BoundingBox, Detection};
use yscv_track::TrackedDetection;

use crate::{DetectionDatasetFrame, GroundTruthTrack, LabeledBox, TrackingDatasetFrame};

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct BoundingBoxWire {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

impl From<BoundingBoxWire> for BoundingBox {
    fn from(value: BoundingBoxWire) -> Self {
        Self {
            x1: value.x1,
            y1: value.y1,
            x2: value.x2,
            y2: value.y2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct LabeledBoxWire {
    bbox: BoundingBoxWire,
    class_id: usize,
}

impl From<LabeledBoxWire> for LabeledBox {
    fn from(value: LabeledBoxWire) -> Self {
        Self {
            bbox: value.bbox.into(),
            class_id: value.class_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct DetectionWire {
    bbox: BoundingBoxWire,
    score: f32,
    class_id: usize,
}

impl From<DetectionWire> for Detection {
    fn from(value: DetectionWire) -> Self {
        Self {
            bbox: value.bbox.into(),
            score: value.score,
            class_id: value.class_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct DetectionDatasetFrameWire {
    ground_truth: Vec<LabeledBoxWire>,
    predictions: Vec<DetectionWire>,
}

impl DetectionDatasetFrameWire {
    pub(crate) fn into_runtime(self) -> DetectionDatasetFrame {
        DetectionDatasetFrame {
            ground_truth: self.ground_truth.into_iter().map(Into::into).collect(),
            predictions: self.predictions.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct GroundTruthTrackWire {
    object_id: u64,
    bbox: BoundingBoxWire,
    class_id: usize,
}

impl From<GroundTruthTrackWire> for GroundTruthTrack {
    fn from(value: GroundTruthTrackWire) -> Self {
        Self {
            object_id: value.object_id,
            bbox: value.bbox.into(),
            class_id: value.class_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct TrackedDetectionWire {
    track_id: u64,
    detection: DetectionWire,
}

impl From<TrackedDetectionWire> for TrackedDetection {
    fn from(value: TrackedDetectionWire) -> Self {
        Self {
            track_id: value.track_id,
            detection: value.detection.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct TrackingDatasetFrameWire {
    ground_truth: Vec<GroundTruthTrackWire>,
    predictions: Vec<TrackedDetectionWire>,
}

impl TrackingDatasetFrameWire {
    pub(crate) fn into_runtime(self) -> TrackingDatasetFrame {
        TrackingDatasetFrame {
            ground_truth: self.ground_truth.into_iter().map(Into::into).collect(),
            predictions: self.predictions.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct CocoImageWire {
    pub(crate) id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct CocoAnnotationWire {
    pub(crate) image_id: u64,
    pub(crate) category_id: u64,
    pub(crate) bbox: [f32; 4],
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct CocoGroundTruthWire {
    pub(crate) images: Vec<CocoImageWire>,
    pub(crate) annotations: Vec<CocoAnnotationWire>,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) struct CocoPredictionWire {
    pub(crate) image_id: u64,
    pub(crate) category_id: u64,
    pub(crate) bbox: [f32; 4],
    pub(crate) score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct YoloManifestEntry {
    pub(crate) image_id: String,
    pub(crate) width: usize,
    pub(crate) height: usize,
}
