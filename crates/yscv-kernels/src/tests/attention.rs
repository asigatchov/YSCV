use yscv_tensor::Tensor;

use crate::scaled_dot_product_attention;

#[test]
fn attention_basic() {
    // query [2, 3], key [2, 3], value [2, 4] → output [2, 4]
    let query = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let key = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let value = Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

    let out = scaled_dot_product_attention(&query, &key, &value, None).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    // All values must be finite
    for &v in out.data() {
        assert!(v.is_finite(), "non-finite value in output: {v}");
    }
}

#[test]
fn attention_with_mask() {
    // 2 query positions, 3 key positions, d_k=2, d_v=2
    let query = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let key = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).unwrap();
    let value = Tensor::from_vec(vec![3, 2], vec![10.0, 0.0, 0.0, 10.0, 5.0, 5.0]).unwrap();

    // Mask out position 0 for all queries (large negative → softmax ≈ 0)
    let mask = Tensor::from_vec(vec![2, 3], vec![-1e9, 0.0, 0.0, -1e9, 0.0, 0.0]).unwrap();

    let out = scaled_dot_product_attention(&query, &key, &value, Some(&mask)).unwrap();
    assert_eq!(out.shape(), &[2, 2]);

    // Position 0 of value (10.0, 0.0) should contribute ~0 weight for both queries,
    // so outputs should be close to a mix of value[1] and value[2] only.
    let data = out.data();
    // First query (aligned with key[0] which is masked): output should not be near (10, 0).
    assert!(
        data[0] < 9.0,
        "masked position should not dominate: {}",
        data[0]
    );
    for &v in data {
        assert!(v.is_finite(), "non-finite value in masked output: {v}");
    }
}

#[test]
fn attention_single_head() {
    // Larger example: seq_q=4, seq_k=6, d_k=8, d_v=5 → output [4, 5]
    let seq_q = 4;
    let seq_k = 6;
    let d_k = 8;
    let d_v = 5;

    let q_data: Vec<f32> = (0..seq_q * d_k).map(|i| ((i % 13) as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..seq_k * d_k).map(|i| ((i % 11) as f32) * 0.1).collect();
    let v_data: Vec<f32> = (0..seq_k * d_v).map(|i| ((i % 7) as f32) * 0.1).collect();

    let query = Tensor::from_vec(vec![seq_q, d_k], q_data).unwrap();
    let key = Tensor::from_vec(vec![seq_k, d_k], k_data).unwrap();
    let value = Tensor::from_vec(vec![seq_k, d_v], v_data).unwrap();

    let out = scaled_dot_product_attention(&query, &key, &value, None).unwrap();
    assert_eq!(out.shape(), &[seq_q, d_v]);
    for &v in out.data() {
        assert!(v.is_finite(), "non-finite value in single-head output: {v}");
    }
}

#[test]
fn attention_identity_query_key() {
    // When Q = K with well-separated rows, each query should attend
    // most strongly to its own position (high diagonal weight).
    let d_k = 4;
    let seq = 3;

    // Make rows clearly distinct so self-dot-product dominates
    #[rustfmt::skip]
    let data = vec![
        10.0,  0.0,  0.0,  0.0,
         0.0, 10.0,  0.0,  0.0,
         0.0,  0.0, 10.0,  0.0,
    ];
    let qk = Tensor::from_vec(vec![seq, d_k], data).unwrap();

    // Value = identity-like so we can inspect where the weight goes
    #[rustfmt::skip]
    let v_data = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let value = Tensor::from_vec(vec![seq, seq], v_data).unwrap();

    let out = scaled_dot_product_attention(&qk, &qk, &value, None).unwrap();
    assert_eq!(out.shape(), &[seq, seq]);

    let data = out.data();
    // Diagonal entries should be close to 1 because each query attends mostly to itself
    for i in 0..seq {
        let diag = data[i * seq + i];
        assert!(
            diag > 0.9,
            "diagonal weight at position {i} should be near 1.0, got {diag}"
        );
    }
}
