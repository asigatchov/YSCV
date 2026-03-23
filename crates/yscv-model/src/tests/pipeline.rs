use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use super::super::SequentialModel;
use super::super::pipeline::InferencePipeline;

#[test]
fn pipeline_identity() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_relu();

    let pipeline = InferencePipeline::new(model);
    let input = Tensor::from_vec(vec![1, 4], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let out = pipeline.run(&input).unwrap();
    assert_eq!(out.shape(), &[1, 4]);
    // ReLU clamps negatives to 0
    let data = out.data();
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[1] - 0.0).abs() < 1e-6);
    assert!((data[2] - 1.0).abs() < 1e-6);
    assert!((data[3] - 2.0).abs() < 1e-6);
}

#[test]
fn pipeline_with_preprocess_postprocess() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_relu();

    let pipeline = InferencePipeline::new(model)
        .with_preprocess(|t| {
            // Scale input by 2
            let scaled: Vec<f32> = t.data().iter().map(|&v| v * 2.0).collect();
            Ok(Tensor::from_vec(t.shape().to_vec(), scaled)?)
        })
        .with_postprocess(|t| {
            // Add 10 to output
            let shifted: Vec<f32> = t.data().iter().map(|&v| v + 10.0).collect();
            Ok(Tensor::from_vec(t.shape().to_vec(), shifted)?)
        });

    let input = Tensor::from_vec(vec![1, 3], vec![-1.0, 0.5, 1.0]).unwrap();
    let out = pipeline.run(&input).unwrap();
    assert_eq!(out.shape(), &[1, 3]);
    let data = out.data();
    // preprocess: [-2.0, 1.0, 2.0]
    // relu:       [0.0, 1.0, 2.0]
    // postprocess:[10.0, 11.0, 12.0]
    assert!((data[0] - 10.0).abs() < 1e-6);
    assert!((data[1] - 11.0).abs() < 1e-6);
    assert!((data[2] - 12.0).abs() < 1e-6);
}

#[test]
fn pipeline_run_batch() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_relu();

    let pipeline = InferencePipeline::new(model);
    let inputs = vec![
        Tensor::from_vec(vec![1, 2], vec![-1.0, 2.0]).unwrap(),
        Tensor::from_vec(vec![1, 2], vec![3.0, -4.0]).unwrap(),
        Tensor::from_vec(vec![1, 2], vec![0.0, 0.5]).unwrap(),
    ];

    let outputs = pipeline.run_batch(&inputs).unwrap();
    assert_eq!(outputs.len(), 3);

    // First: relu([-1, 2]) = [0, 2]
    assert!((outputs[0].data()[0] - 0.0).abs() < 1e-6);
    assert!((outputs[0].data()[1] - 2.0).abs() < 1e-6);

    // Second: relu([3, -4]) = [3, 0]
    assert!((outputs[1].data()[0] - 3.0).abs() < 1e-6);
    assert!((outputs[1].data()[1] - 0.0).abs() < 1e-6);

    // Third: relu([0, 0.5]) = [0, 0.5]
    assert!((outputs[2].data()[0] - 0.0).abs() < 1e-6);
    assert!((outputs[2].data()[1] - 0.5).abs() < 1e-6);
}
