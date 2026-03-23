use crate::TrackError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerConfig {
    pub match_iou_threshold: f32,
    pub max_missed_frames: u32,
    pub max_tracks: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            match_iou_threshold: 0.3,
            max_missed_frames: 10,
            max_tracks: 1024,
        }
    }
}

impl TrackerConfig {
    pub fn validate(&self) -> Result<(), TrackError> {
        if !self.match_iou_threshold.is_finite() || !(0.0..=1.0).contains(&self.match_iou_threshold)
        {
            return Err(TrackError::InvalidIouThreshold {
                value: self.match_iou_threshold,
            });
        }
        if self.max_tracks == 0 {
            return Err(TrackError::InvalidMaxTracks { value: 0 });
        }
        Ok(())
    }
}
