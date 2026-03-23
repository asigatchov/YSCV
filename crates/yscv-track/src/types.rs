use yscv_detect::{BoundingBox, Detection};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Track {
    pub id: u64,
    pub bbox: BoundingBox,
    pub score: f32,
    pub class_id: usize,
    pub age: u64,
    pub hits: u64,
    pub missed_frames: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackedDetection {
    pub track_id: u64,
    pub detection: Detection,
}
