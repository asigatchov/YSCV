//! Automatic differentiation graph and backward primitives for yscv.
#![deny(unsafe_code)]

pub const CRATE_ID: &str = "yscv-autograd";

#[path = "backward/mod.rs"]
mod backward;
#[path = "checkpoint.rs"]
pub mod checkpoint;
#[path = "error.rs"]
mod error;
#[path = "graph.rs"]
mod graph;
#[path = "node.rs"]
mod node;

pub use error::AutogradError;
pub use graph::Graph;
pub use node::NodeId;

#[path = "tests/mod.rs"]
#[cfg(test)]
mod tests;
