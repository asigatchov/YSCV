//! Recognition and identity matching components for yscv.
#![deny(unsafe_code)]

pub const CRATE_ID: &str = "yscv-recognize";

#[path = "error.rs"]
mod error;
#[path = "recognizer.rs"]
mod recognizer;
#[path = "similarity.rs"]
mod similarity;
#[path = "snapshot.rs"]
mod snapshot;
#[path = "types.rs"]
mod types;
#[path = "validate.rs"]
mod validate;
#[path = "vp_tree.rs"]
pub mod vp_tree;

pub use error::RecognizeError;
pub use recognizer::Recognizer;
pub use similarity::{cosine_similarity, cosine_similarity_slice};
pub use snapshot::{IdentitySnapshot, RecognizerSnapshot};
pub use types::{IdentityEmbedding, Recognition};
pub use vp_tree::VpTree;

#[path = "tests.rs"]
#[cfg(test)]
mod tests;
