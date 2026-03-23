#[cfg(not(miri))]
use std::sync::atomic::{AtomicU64, Ordering};

use yscv_tensor::Tensor;

use super::{RecognizeError, Recognizer, VpTree, cosine_similarity, cosine_similarity_slice};

#[test]
fn cosine_similarity_is_one_for_identical_vectors() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_similarity_slice_is_one_for_identical_vectors() {
    let sim = cosine_similarity_slice(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn recognize_returns_known_identity_above_threshold() {
    let mut recognizer = Recognizer::new(0.9).unwrap();
    recognizer
        .enroll(
            "alice",
            Tensor::from_vec(vec![3], vec![0.2, 0.3, 0.4]).unwrap(),
        )
        .unwrap();

    let query = Tensor::from_vec(vec![3], vec![0.21, 0.31, 0.39]).unwrap();
    let out = recognizer.recognize(&query).unwrap();
    assert!(out.is_known());
    assert_eq!(out.identity.as_deref(), Some("alice"));
}

#[test]
fn recognize_slice_returns_known_identity_above_threshold() {
    let mut recognizer = Recognizer::new(0.9).unwrap();
    recognizer
        .enroll(
            "alice",
            Tensor::from_vec(vec![3], vec![0.2, 0.3, 0.4]).unwrap(),
        )
        .unwrap();

    let query = [0.21, 0.31, 0.39];
    let out = recognizer.recognize_slice(&query).unwrap();
    assert!(out.is_known());
    assert_eq!(out.identity.as_deref(), Some("alice"));
}

#[test]
fn recognize_returns_unknown_below_threshold() {
    let mut recognizer = Recognizer::new(0.95).unwrap();
    recognizer
        .enroll(
            "alice",
            Tensor::from_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap(),
        )
        .unwrap();
    let query = Tensor::from_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap();
    let out = recognizer.recognize(&query).unwrap();
    assert!(!out.is_known());
}

#[test]
fn enroll_rejects_duplicate_identity() {
    let mut recognizer = Recognizer::new(0.8).unwrap();
    recognizer
        .enroll("alice", Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap())
        .unwrap();
    let err = recognizer
        .enroll("alice", Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap())
        .unwrap_err();
    assert_eq!(
        err,
        RecognizeError::DuplicateIdentity {
            id: "alice".to_string()
        }
    );
}

#[test]
fn recognize_rejects_dimension_mismatch() {
    let mut recognizer = Recognizer::new(0.8).unwrap();
    recognizer
        .enroll("alice", Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap())
        .unwrap();
    let err = recognizer
        .recognize(&Tensor::from_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap())
        .unwrap_err();
    assert_eq!(
        err,
        RecognizeError::EmbeddingDimMismatch {
            expected: 2,
            got: 3
        }
    );
}

#[test]
fn recognize_slice_rejects_non_finite_embedding() {
    let mut recognizer = Recognizer::new(0.8).unwrap();
    recognizer
        .enroll("alice", Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap())
        .unwrap();
    let err = recognizer.recognize_slice(&[1.0, f32::NAN]).unwrap_err();
    assert_eq!(err, RecognizeError::NonFiniteEmbeddingValue { index: 1 });
}

#[test]
fn remove_allows_dimension_reset_after_last_identity() {
    let mut recognizer = Recognizer::new(0.8).unwrap();
    recognizer
        .enroll("alice", Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap())
        .unwrap();
    assert!(recognizer.remove("alice"));
    recognizer
        .enroll(
            "bob",
            Tensor::from_vec(vec![3], vec![0.0, 1.0, 0.0]).unwrap(),
        )
        .unwrap();
    assert_eq!(recognizer.identities().len(), 1);
}

#[test]
fn enroll_or_replace_updates_existing_identity() {
    let mut recognizer = Recognizer::new(0.8).unwrap();
    recognizer
        .enroll("alice", Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap())
        .unwrap();
    recognizer
        .enroll_or_replace("alice", Tensor::from_vec(vec![2], vec![0.5, 0.5]).unwrap())
        .unwrap();

    let query = Tensor::from_vec(vec![2], vec![0.4, 0.6]).unwrap();
    let out = recognizer.recognize(&query).unwrap();
    assert_eq!(out.identity.as_deref(), Some("alice"));
}

#[test]
fn snapshot_json_roundtrip_preserves_state() {
    let mut recognizer = Recognizer::new(0.75).unwrap();
    recognizer
        .enroll(
            "alice",
            Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap(),
        )
        .unwrap();
    recognizer
        .enroll(
            "bob",
            Tensor::from_vec(vec![3], vec![0.3, 0.2, 0.1]).unwrap(),
        )
        .unwrap();

    let json = recognizer.to_json_pretty().unwrap();
    let restored = Recognizer::from_json(&json).unwrap();
    assert_eq!(restored.threshold(), 0.75);
    assert_eq!(restored.identities().len(), 2);
}

#[test]
#[cfg(not(miri))]
fn save_and_load_json_file_roundtrip() {
    let mut recognizer = Recognizer::new(0.8).unwrap();
    recognizer
        .enroll("alice", Tensor::from_vec(vec![2], vec![1.0, 0.0]).unwrap())
        .unwrap();

    static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("yscv-recognizer-{pid}-{unique}.json"));

    recognizer.save_json_file(&path).unwrap();
    let loaded = Recognizer::load_json_file(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.threshold(), recognizer.threshold());
    assert_eq!(loaded.identities().len(), 1);
    assert_eq!(loaded.identities()[0].id, "alice");
}

#[test]
fn vp_tree_empty() {
    let tree = VpTree::new();
    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
    let results = tree.query(&[1.0, 0.0, 0.0], 5);
    assert!(results.is_empty());
}

#[test]
fn vp_tree_single() {
    let tree = VpTree::build(vec![("only".to_string(), vec![1.0, 0.0, 0.0])]);
    assert_eq!(tree.len(), 1);
    assert!(!tree.is_empty());
    let results = tree.query(&[1.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "only");
    assert!(results[0].distance < 1e-6);
}

#[test]
fn vp_tree_knn() {
    // Build 10 embeddings with known structure so we can verify brute-force vs tree.
    let dim = 16;
    let mut entries: Vec<(String, Vec<f32>)> = Vec::new();
    for i in 0..10 {
        let mut emb = vec![0.0f32; dim];
        // Create distinct embeddings by setting different components.
        emb[i] += 1.0;
        emb[(i + 1) % dim] += 0.5;
        entries.push((format!("id_{i}"), emb));
    }

    let tree = VpTree::build(entries.clone());
    assert_eq!(tree.len(), 10);

    let query = vec![
        1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let k = 3;
    let tree_results = tree.query(&query, k);

    // Brute-force: compute all distances, sort, take top k.
    let mut brute: Vec<(String, f32)> = entries
        .iter()
        .map(|(id, emb)| {
            let dot: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = query.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
            let sim = if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na * nb)
            };
            (id.clone(), 1.0 - sim)
        })
        .collect();
    brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    brute.truncate(k);

    assert_eq!(tree_results.len(), k);
    // Compare distances (IDs may differ for equidistant points).
    let tree_dists: Vec<f32> = tree_results.iter().map(|r| r.distance).collect();
    let brute_dists: Vec<f32> = brute.iter().map(|r| r.1).collect();
    for (td, bd) in tree_dists.iter().zip(brute_dists.iter()) {
        assert!(
            (td - bd).abs() < 1e-4,
            "distance mismatch: tree={td}, brute={bd}"
        );
    }
}

#[test]
fn recognizer_build_index() {
    let mut recognizer = Recognizer::new(0.5).unwrap();
    let embeddings = [
        ("alice", vec![1.0, 0.0, 0.0]),
        ("bob", vec![0.0, 1.0, 0.0]),
        ("carol", vec![0.0, 0.0, 1.0]),
        ("dave", vec![0.7, 0.7, 0.0]),
        ("eve", vec![0.0, 0.7, 0.7]),
    ];
    for (id, data) in &embeddings {
        recognizer
            .enroll(*id, Tensor::from_vec(vec![3], data.clone()).unwrap())
            .unwrap();
    }

    recognizer.build_index();

    // Query with something very close to alice.
    let query = Tensor::from_vec(vec![3], vec![0.99, 0.01, 0.0]).unwrap();
    let results = recognizer.search_indexed(&query, 2).unwrap();
    assert!(!results.is_empty());
    // The top result should be alice (highest cosine similarity to [1,0,0]).
    assert_eq!(results[0].identity.as_deref(), Some("alice"));
}
