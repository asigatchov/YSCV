pub const CRATE_ID: &str = "yscv-detect";
pub const CLASS_ID_PERSON: usize = 0;
pub const CLASS_ID_FACE: usize = 1;

/// Axis-aligned box in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// One detection with score and class id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub score: f32,
    pub class_id: usize,
}
