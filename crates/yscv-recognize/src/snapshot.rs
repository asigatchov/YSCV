use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognizerSnapshot {
    pub threshold: f32,
    pub identities: Vec<IdentitySnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentitySnapshot {
    pub id: String,
    pub embedding: Vec<f32>,
}
