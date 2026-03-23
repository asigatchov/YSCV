use yscv_tensor::Tensor;

#[test]
fn batched_inference_splits_correctly() {
    let input = Tensor::from_vec(vec![10, 4], vec![1.0; 40]).unwrap();
    let config = crate::DynamicBatchConfig {
        max_batch_size: 3,
        pad_incomplete: false,
    };
    let result = crate::batched_inference(&input, &config, |batch| {
        // Identity function
        Ok(batch.clone())
    })
    .unwrap();
    assert_eq!(result.shape(), &[10, 4]);
    assert_eq!(result.data(), input.data());
}

#[test]
fn batch_collector_flushes_at_capacity() {
    let mut collector = crate::BatchCollector::new(vec![3], 2);
    let s1 = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let s2 = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
    collector.push(&s1).unwrap();
    assert!(!collector.is_ready());
    collector.push(&s2).unwrap();
    assert!(collector.is_ready());
    let batch = collector.flush().unwrap().unwrap();
    assert_eq!(batch.shape(), &[2, 3]);
    assert_eq!(collector.pending(), 0);
}
