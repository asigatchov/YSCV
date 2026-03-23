use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TrackError {
    #[error("invalid IoU threshold: {value}; expected finite value in [0, 1]")]
    InvalidIouThreshold { value: f32 },
    #[error("invalid max_tracks: {value}; expected max_tracks > 0")]
    InvalidMaxTracks { value: usize },
}
