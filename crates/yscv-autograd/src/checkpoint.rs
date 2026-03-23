//! Activation checkpointing for memory-efficient backward passes.
//!
//! During the forward pass, checkpoint segments drop intermediate activations.
//! During backward, those activations are recomputed from the segment inputs,
//! trading compute for memory.

/// Marks a range of node IDs as a checkpoint segment.
/// During backward, activations for these nodes will be recomputed rather than stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSegment {
    /// First node ID in the segment.
    pub start_node: usize,
    /// Last node ID in the segment (inclusive).
    pub end_node: usize,
}

/// Configuration for activation checkpointing.
#[derive(Debug, Clone, Default)]
pub struct CheckpointConfig {
    /// Segments to checkpoint (manually specified).
    pub segments: Vec<CheckpointSegment>,
}

impl CheckpointConfig {
    /// Create checkpointing config that splits the graph into `num_segments` equal segments.
    ///
    /// If `num_nodes` is 0 or `num_segments` is 0, returns an empty config.
    pub fn uniform(num_nodes: usize, num_segments: usize) -> Self {
        if num_nodes == 0 || num_segments == 0 {
            return Self {
                segments: Vec::new(),
            };
        }

        let segment_size = num_nodes / num_segments;
        if segment_size == 0 {
            return Self {
                segments: Vec::new(),
            };
        }

        let mut segments = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let start = i * segment_size;
            let end = if i == num_segments - 1 {
                num_nodes - 1
            } else {
                (i + 1) * segment_size - 1
            };
            segments.push(CheckpointSegment {
                start_node: start,
                end_node: end,
            });
        }

        Self { segments }
    }

    /// Check if a node should have its activation dropped after forward.
    ///
    /// Returns `true` if the node falls within any checkpoint segment.
    pub fn should_checkpoint(&self, node_id: usize) -> bool {
        self.segments
            .iter()
            .any(|seg| node_id >= seg.start_node && node_id <= seg.end_node)
    }
}

/// Given a total number of nodes and a checkpoint config, determine which activations
/// can be freed and which need to be recomputed during backward.
///
/// Returns a `Vec<bool>` of length `num_nodes` where `true` means the activation
/// will be recomputed (freed after forward), and `false` means it is kept in memory.
pub fn plan_recomputation(num_nodes: usize, config: &CheckpointConfig) -> Vec<bool> {
    (0..num_nodes)
        .map(|id| config.should_checkpoint(id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_segments() {
        let config = CheckpointConfig::uniform(10, 2);
        assert_eq!(config.segments.len(), 2);
        // First segment: nodes 0..4
        assert_eq!(config.segments[0].start_node, 0);
        assert_eq!(config.segments[0].end_node, 4);
        // Second segment: nodes 5..9
        assert_eq!(config.segments[1].start_node, 5);
        assert_eq!(config.segments[1].end_node, 9);
    }

    #[test]
    fn test_should_checkpoint() {
        let config = CheckpointConfig {
            segments: vec![
                CheckpointSegment {
                    start_node: 2,
                    end_node: 4,
                },
                CheckpointSegment {
                    start_node: 7,
                    end_node: 9,
                },
            ],
        };
        // Outside any segment
        assert!(!config.should_checkpoint(0));
        assert!(!config.should_checkpoint(1));
        assert!(!config.should_checkpoint(5));
        assert!(!config.should_checkpoint(6));
        assert!(!config.should_checkpoint(10));
        // Inside segments
        assert!(config.should_checkpoint(2));
        assert!(config.should_checkpoint(3));
        assert!(config.should_checkpoint(4));
        assert!(config.should_checkpoint(7));
        assert!(config.should_checkpoint(8));
        assert!(config.should_checkpoint(9));
    }

    #[test]
    fn test_plan_recomputation() {
        let config = CheckpointConfig {
            segments: vec![CheckpointSegment {
                start_node: 1,
                end_node: 3,
            }],
        };
        let plan = plan_recomputation(5, &config);
        assert_eq!(plan, vec![false, true, true, true, false]);
    }

    #[test]
    fn test_empty_config() {
        let config = CheckpointConfig::default();
        assert!(!config.should_checkpoint(0));
        assert!(!config.should_checkpoint(100));
        let plan = plan_recomputation(5, &config);
        assert_eq!(plan, vec![false, false, false, false, false]);
    }

    #[test]
    fn test_single_segment() {
        let config = CheckpointConfig::uniform(8, 1);
        assert_eq!(config.segments.len(), 1);
        assert_eq!(config.segments[0].start_node, 0);
        assert_eq!(config.segments[0].end_node, 7);
        // Every node is checkpointed
        for i in 0..8 {
            assert!(
                config.should_checkpoint(i),
                "node {i} should be checkpointed"
            );
        }
        let plan = plan_recomputation(8, &config);
        assert!(plan.iter().all(|&v| v));
    }
}
