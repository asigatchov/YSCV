use yscv_autograd::{Graph, NodeId};
use yscv_optim::{Adam, AdamW, LearningRate, LrScheduler, RmsProp, Sgd};
use yscv_tensor::Tensor;

use crate::{
    BatchIterOptions, GradientAggregator, ModelError, SequentialModel, SupervisedDataset, bce_loss,
    cross_entropy_loss, hinge_loss, huber_loss, mae_loss, mse_loss, nll_loss,
};

trait GraphOptimizer {
    fn step_graph_node(&mut self, graph: &mut Graph, node: NodeId) -> Result<(), ModelError>;
}

/// Configures supervised-loss function used by train-step and train-epoch helpers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SupervisedLoss {
    #[default]
    Mse,
    Mae,
    Huber {
        delta: f32,
    },
    Hinge {
        margin: f32,
    },
    Bce,
    Nll,
    CrossEntropy,
}

fn build_loss_node(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    loss: SupervisedLoss,
) -> Result<NodeId, ModelError> {
    match loss {
        SupervisedLoss::Mse => mse_loss(graph, prediction, target),
        SupervisedLoss::Mae => mae_loss(graph, prediction, target),
        SupervisedLoss::Huber { delta } => huber_loss(graph, prediction, target, delta),
        SupervisedLoss::Hinge { margin } => hinge_loss(graph, prediction, target, margin),
        SupervisedLoss::Bce => bce_loss(graph, prediction, target),
        SupervisedLoss::Nll => nll_loss(graph, prediction, target),
        SupervisedLoss::CrossEntropy => cross_entropy_loss(graph, prediction, target),
    }
}

impl GraphOptimizer for Sgd {
    fn step_graph_node(&mut self, graph: &mut Graph, node: NodeId) -> Result<(), ModelError> {
        Sgd::step_graph_node(self, graph, node).map_err(Into::into)
    }
}

impl GraphOptimizer for Adam {
    fn step_graph_node(&mut self, graph: &mut Graph, node: NodeId) -> Result<(), ModelError> {
        Adam::step_graph_node(self, graph, node).map_err(Into::into)
    }
}

impl GraphOptimizer for AdamW {
    fn step_graph_node(&mut self, graph: &mut Graph, node: NodeId) -> Result<(), ModelError> {
        AdamW::step_graph_node(self, graph, node).map_err(Into::into)
    }
}

impl GraphOptimizer for RmsProp {
    fn step_graph_node(&mut self, graph: &mut Graph, node: NodeId) -> Result<(), ModelError> {
        RmsProp::step_graph_node(self, graph, node).map_err(Into::into)
    }
}

fn train_step_with_optimizer<O: GraphOptimizer>(
    graph: &mut Graph,
    optimizer: &mut O,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
    loss: SupervisedLoss,
) -> Result<f32, ModelError> {
    let loss_node = build_loss_node(graph, prediction, target, loss)?;
    graph.backward(loss_node)?;

    let loss_value = graph.value(loss_node)?.data()[0];
    for node in trainable_nodes {
        optimizer.step_graph_node(graph, *node)?;
    }
    Ok(loss_value)
}

/// Runs one full train step: loss forward, backward, and SGD updates.
pub fn train_step_sgd(
    graph: &mut Graph,
    optimizer: &mut Sgd,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
) -> Result<f32, ModelError> {
    train_step_sgd_with_loss(
        graph,
        optimizer,
        prediction,
        target,
        trainable_nodes,
        SupervisedLoss::Mse,
    )
}

/// Runs one full train step: configured loss forward, backward, and SGD updates.
pub fn train_step_sgd_with_loss(
    graph: &mut Graph,
    optimizer: &mut Sgd,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
    loss: SupervisedLoss,
) -> Result<f32, ModelError> {
    train_step_with_optimizer(graph, optimizer, prediction, target, trainable_nodes, loss)
}

/// Runs one full train step: loss forward, backward, and Adam updates.
pub fn train_step_adam(
    graph: &mut Graph,
    optimizer: &mut Adam,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
) -> Result<f32, ModelError> {
    train_step_adam_with_loss(
        graph,
        optimizer,
        prediction,
        target,
        trainable_nodes,
        SupervisedLoss::Mse,
    )
}

/// Runs one full train step: configured loss forward, backward, and Adam updates.
pub fn train_step_adam_with_loss(
    graph: &mut Graph,
    optimizer: &mut Adam,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
    loss: SupervisedLoss,
) -> Result<f32, ModelError> {
    train_step_with_optimizer(graph, optimizer, prediction, target, trainable_nodes, loss)
}

/// Runs one full train step: loss forward, backward, and AdamW updates.
pub fn train_step_adamw(
    graph: &mut Graph,
    optimizer: &mut AdamW,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
) -> Result<f32, ModelError> {
    train_step_adamw_with_loss(
        graph,
        optimizer,
        prediction,
        target,
        trainable_nodes,
        SupervisedLoss::Mse,
    )
}

/// Runs one full train step: configured loss forward, backward, and AdamW updates.
pub fn train_step_adamw_with_loss(
    graph: &mut Graph,
    optimizer: &mut AdamW,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
    loss: SupervisedLoss,
) -> Result<f32, ModelError> {
    train_step_with_optimizer(graph, optimizer, prediction, target, trainable_nodes, loss)
}

/// Runs one full train step: loss forward, backward, and RMSProp updates.
pub fn train_step_rmsprop(
    graph: &mut Graph,
    optimizer: &mut RmsProp,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
) -> Result<f32, ModelError> {
    train_step_rmsprop_with_loss(
        graph,
        optimizer,
        prediction,
        target,
        trainable_nodes,
        SupervisedLoss::Mse,
    )
}

/// Runs one full train step: configured loss forward, backward, and RMSProp updates.
pub fn train_step_rmsprop_with_loss(
    graph: &mut Graph,
    optimizer: &mut RmsProp,
    prediction: NodeId,
    target: NodeId,
    trainable_nodes: &[NodeId],
    loss: SupervisedLoss,
) -> Result<f32, ModelError> {
    train_step_with_optimizer(graph, optimizer, prediction, target, trainable_nodes, loss)
}

/// Metrics for one training epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    pub mean_loss: f32,
    pub steps: usize,
}

/// Metrics for one scheduler-driven epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScheduledEpochMetrics {
    pub epoch: usize,
    pub mean_loss: f32,
    pub steps: usize,
    pub learning_rate: f32,
}

/// Epoch-level training controls for batch order and preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochTrainOptions {
    pub batch_size: usize,
    pub batch_iter_options: BatchIterOptions,
}

impl Default for EpochTrainOptions {
    fn default() -> Self {
        Self {
            batch_size: 1,
            batch_iter_options: BatchIterOptions::default(),
        }
    }
}

/// Scheduler-driven epoch training controls.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SchedulerTrainOptions {
    pub epoch_options: EpochTrainOptions,
    pub loss: SupervisedLoss,
}

/// Deterministic one-epoch train loop over sequential mini-batches.
pub fn train_epoch_sgd(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Sgd,
    dataset: &SupervisedDataset,
    batch_size: usize,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_sgd_with_loss(
        graph,
        model,
        optimizer,
        dataset,
        batch_size,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch train loop with configurable supervised loss.
pub fn train_epoch_sgd_with_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Sgd,
    dataset: &SupervisedDataset,
    batch_size: usize,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_sgd_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        EpochTrainOptions {
            batch_size,
            batch_iter_options: BatchIterOptions::default(),
        },
        loss,
    )
}

/// Deterministic one-epoch Adam train loop over sequential mini-batches.
pub fn train_epoch_adam(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Adam,
    dataset: &SupervisedDataset,
    batch_size: usize,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_adam_with_loss(
        graph,
        model,
        optimizer,
        dataset,
        batch_size,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch Adam train loop with configurable supervised loss.
pub fn train_epoch_adam_with_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Adam,
    dataset: &SupervisedDataset,
    batch_size: usize,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_adam_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        EpochTrainOptions {
            batch_size,
            batch_iter_options: BatchIterOptions::default(),
        },
        loss,
    )
}

/// Deterministic one-epoch AdamW train loop over sequential mini-batches.
pub fn train_epoch_adamw(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut AdamW,
    dataset: &SupervisedDataset,
    batch_size: usize,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_adamw_with_loss(
        graph,
        model,
        optimizer,
        dataset,
        batch_size,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch AdamW train loop with configurable supervised loss.
pub fn train_epoch_adamw_with_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut AdamW,
    dataset: &SupervisedDataset,
    batch_size: usize,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_adamw_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        EpochTrainOptions {
            batch_size,
            batch_iter_options: BatchIterOptions::default(),
        },
        loss,
    )
}

/// Deterministic one-epoch RMSProp train loop over sequential mini-batches.
pub fn train_epoch_rmsprop(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut RmsProp,
    dataset: &SupervisedDataset,
    batch_size: usize,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_rmsprop_with_loss(
        graph,
        model,
        optimizer,
        dataset,
        batch_size,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch RMSProp train loop with configurable supervised loss.
pub fn train_epoch_rmsprop_with_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut RmsProp,
    dataset: &SupervisedDataset,
    batch_size: usize,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_rmsprop_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        EpochTrainOptions {
            batch_size,
            batch_iter_options: BatchIterOptions::default(),
        },
        loss,
    )
}

/// Deterministic one-epoch train loop with configurable batch iterator options.
pub fn train_epoch_sgd_with_options(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Sgd,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_sgd_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        options,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch train loop with configurable batch iterator options and loss.
pub fn train_epoch_sgd_with_options_and_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Sgd,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_with_options(graph, model, optimizer, dataset, options, loss)
}

/// Deterministic one-epoch Adam train loop with configurable batch iterator options.
pub fn train_epoch_adam_with_options(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Adam,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_adam_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        options,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch Adam train loop with configurable batch iterator options and loss.
pub fn train_epoch_adam_with_options_and_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Adam,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_with_options(graph, model, optimizer, dataset, options, loss)
}

/// Deterministic one-epoch AdamW train loop with configurable batch iterator options.
pub fn train_epoch_adamw_with_options(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut AdamW,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_adamw_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        options,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch AdamW train loop with configurable batch iterator options and loss.
pub fn train_epoch_adamw_with_options_and_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut AdamW,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_with_options(graph, model, optimizer, dataset, options, loss)
}

/// Deterministic one-epoch RMSProp train loop with configurable batch iterator options.
pub fn train_epoch_rmsprop_with_options(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut RmsProp,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_rmsprop_with_options_and_loss(
        graph,
        model,
        optimizer,
        dataset,
        options,
        SupervisedLoss::Mse,
    )
}

/// Deterministic one-epoch RMSProp train loop with configurable batch iterator options and loss.
pub fn train_epoch_rmsprop_with_options_and_loss(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut RmsProp,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    train_epoch_with_options(graph, model, optimizer, dataset, options, loss)
}

/// Runs multiple SGD epochs and advances scheduler after each epoch.
pub fn train_epochs_sgd_with_scheduler<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Sgd,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: EpochTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_sgd_with_scheduler_and_loss(
        graph,
        model,
        optimizer,
        scheduler,
        dataset,
        epochs,
        SchedulerTrainOptions {
            epoch_options: options,
            loss: SupervisedLoss::Mse,
        },
    )
}

/// Runs multiple SGD epochs with configurable supervised loss and advances scheduler after each epoch.
pub fn train_epochs_sgd_with_scheduler_and_loss<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Sgd,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: SchedulerTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_with_scheduler(graph, model, optimizer, scheduler, dataset, epochs, options)
}

/// Runs multiple Adam epochs and advances scheduler after each epoch.
pub fn train_epochs_adam_with_scheduler<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Adam,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: EpochTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_adam_with_scheduler_and_loss(
        graph,
        model,
        optimizer,
        scheduler,
        dataset,
        epochs,
        SchedulerTrainOptions {
            epoch_options: options,
            loss: SupervisedLoss::Mse,
        },
    )
}

/// Runs multiple Adam epochs with configurable supervised loss and advances scheduler after each epoch.
pub fn train_epochs_adam_with_scheduler_and_loss<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Adam,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: SchedulerTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_with_scheduler(graph, model, optimizer, scheduler, dataset, epochs, options)
}

/// Runs multiple AdamW epochs and advances scheduler after each epoch.
pub fn train_epochs_adamw_with_scheduler<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut AdamW,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: EpochTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_adamw_with_scheduler_and_loss(
        graph,
        model,
        optimizer,
        scheduler,
        dataset,
        epochs,
        SchedulerTrainOptions {
            epoch_options: options,
            loss: SupervisedLoss::Mse,
        },
    )
}

/// Runs multiple AdamW epochs with configurable supervised loss and advances scheduler after each epoch.
pub fn train_epochs_adamw_with_scheduler_and_loss<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut AdamW,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: SchedulerTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_with_scheduler(graph, model, optimizer, scheduler, dataset, epochs, options)
}

/// Runs multiple RMSProp epochs and advances scheduler after each epoch.
pub fn train_epochs_rmsprop_with_scheduler<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut RmsProp,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: EpochTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_rmsprop_with_scheduler_and_loss(
        graph,
        model,
        optimizer,
        scheduler,
        dataset,
        epochs,
        SchedulerTrainOptions {
            epoch_options: options,
            loss: SupervisedLoss::Mse,
        },
    )
}

/// Runs multiple RMSProp epochs with configurable supervised loss and advances scheduler after each epoch.
pub fn train_epochs_rmsprop_with_scheduler_and_loss<S: LrScheduler>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut RmsProp,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: SchedulerTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError> {
    train_epochs_with_scheduler(graph, model, optimizer, scheduler, dataset, epochs, options)
}

fn train_epoch_with_options<O: GraphOptimizer>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut O,
    dataset: &SupervisedDataset,
    options: EpochTrainOptions,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    if dataset.is_empty() {
        return Err(ModelError::EmptyDataset);
    }
    let batches = dataset.batches_with_options(options.batch_size, options.batch_iter_options)?;
    let trainable_nodes = model.trainable_nodes();

    let mut loss_sum = 0.0f32;
    let mut steps = 0usize;
    for batch in batches {
        graph.truncate(model.persistent_node_count())?;

        let input = graph.constant(batch.inputs);
        let target = graph.constant(batch.targets);
        let prediction = model.forward(graph, input)?;
        let loss_value = train_step_with_optimizer(
            graph,
            optimizer,
            prediction,
            target,
            &trainable_nodes,
            loss,
        )?;
        loss_sum += loss_value;
        steps += 1;
    }
    if steps == 0 {
        return Err(ModelError::EmptyDataset);
    }

    Ok(EpochMetrics {
        mean_loss: loss_sum / steps as f32,
        steps,
    })
}

fn train_epochs_with_scheduler<O, S>(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut O,
    scheduler: &mut S,
    dataset: &SupervisedDataset,
    epochs: usize,
    options: SchedulerTrainOptions,
) -> Result<Vec<ScheduledEpochMetrics>, ModelError>
where
    O: GraphOptimizer + LearningRate,
    S: LrScheduler,
{
    if epochs == 0 {
        return Err(ModelError::InvalidEpochCount { epochs });
    }

    let mut all_metrics = Vec::with_capacity(epochs);
    for epoch_index in 0..epochs {
        let epoch_metrics = train_epoch_with_options(
            graph,
            model,
            optimizer,
            dataset,
            options.epoch_options.clone(),
            options.loss,
        )?;
        let learning_rate = scheduler.step(optimizer)?;
        all_metrics.push(ScheduledEpochMetrics {
            epoch: epoch_index + 1,
            mean_loss: epoch_metrics.mean_loss,
            steps: epoch_metrics.steps,
            learning_rate,
        });
    }
    Ok(all_metrics)
}

// ── High-level CNN training and inference helpers ──────────────────

/// Configuration for high-level CNN training.
#[derive(Debug, Clone)]
pub struct CnnTrainConfig {
    pub lr: f32,
    pub batch_size: usize,
    pub loss: SupervisedLoss,
    pub batch_iter_options: BatchIterOptions,
}

impl Default for CnnTrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            batch_size: 16,
            loss: SupervisedLoss::CrossEntropy,
            batch_iter_options: BatchIterOptions::default(),
        }
    }
}

/// One-call CNN training epoch: register params, forward, loss, backward, update, sync.
///
/// Handles the full graph-mode CNN lifecycle for one epoch with SGD.
pub fn train_cnn_epoch_sgd(
    graph: &mut Graph,
    model: &mut SequentialModel,
    dataset: &SupervisedDataset,
    config: &CnnTrainConfig,
) -> Result<EpochMetrics, ModelError> {
    let mut optimizer = yscv_optim::Sgd::new(config.lr)?;
    train_cnn_epoch_with_optimizer(graph, model, dataset, &mut optimizer, config)
}

/// One-call CNN training epoch with Adam optimizer.
pub fn train_cnn_epoch_adam(
    graph: &mut Graph,
    model: &mut SequentialModel,
    dataset: &SupervisedDataset,
    config: &CnnTrainConfig,
) -> Result<EpochMetrics, ModelError> {
    let mut optimizer = Adam::new(config.lr)?;
    train_cnn_epoch_with_optimizer(graph, model, dataset, &mut optimizer, config)
}

/// One-call CNN training epoch with AdamW optimizer.
pub fn train_cnn_epoch_adamw(
    graph: &mut Graph,
    model: &mut SequentialModel,
    dataset: &SupervisedDataset,
    config: &CnnTrainConfig,
) -> Result<EpochMetrics, ModelError> {
    let mut optimizer = AdamW::new(config.lr)?;
    train_cnn_epoch_with_optimizer(graph, model, dataset, &mut optimizer, config)
}

fn train_cnn_epoch_with_optimizer<O: GraphOptimizer>(
    graph: &mut Graph,
    model: &mut SequentialModel,
    dataset: &SupervisedDataset,
    optimizer: &mut O,
    config: &CnnTrainConfig,
) -> Result<EpochMetrics, ModelError> {
    model.register_cnn_params(graph);
    let param_nodes = model.trainable_nodes();
    let persistent = model.persistent_node_count();
    let iter =
        dataset.batches_with_options(config.batch_size, config.batch_iter_options.clone())?;

    let mut total_loss = 0.0f32;
    let mut steps = 0usize;

    for batch in iter {
        graph.truncate(persistent)?;
        let input_node = graph.variable(batch.inputs);
        let target_node = graph.variable(batch.targets);
        let prediction = model.forward(graph, input_node)?;
        let loss_val = train_step_with_optimizer(
            graph,
            optimizer,
            prediction,
            target_node,
            &param_nodes,
            config.loss,
        )?;
        model.sync_cnn_from_graph(graph)?;
        total_loss += loss_val;
        steps += 1;
    }

    Ok(EpochMetrics {
        mean_loss: if steps > 0 {
            total_loss / steps as f32
        } else {
            0.0
        },
        steps,
    })
}

/// Multi-epoch CNN training with configurable optimizer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    Sgd,
    Adam,
    AdamW,
}

/// Runs multiple CNN training epochs, returning per-epoch metrics.
pub fn train_cnn_epochs(
    graph: &mut Graph,
    model: &mut SequentialModel,
    dataset: &SupervisedDataset,
    epochs: usize,
    config: &CnnTrainConfig,
    optimizer_type: OptimizerType,
) -> Result<Vec<EpochMetrics>, ModelError> {
    if epochs == 0 {
        return Err(ModelError::InvalidEpochCount { epochs });
    }
    let mut all = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        let metrics = match optimizer_type {
            OptimizerType::Sgd => train_cnn_epoch_sgd(graph, model, dataset, config)?,
            OptimizerType::Adam => train_cnn_epoch_adam(graph, model, dataset, config)?,
            OptimizerType::AdamW => train_cnn_epoch_adamw(graph, model, dataset, config)?,
        };
        all.push(metrics);
    }
    Ok(all)
}

// ── Gradient accumulation helpers ──────────────────────────────────

/// Scales gradients of the given nodes by a scalar factor.
///
/// Nodes without computed gradients are skipped.
pub fn scale_gradients(graph: &mut Graph, nodes: &[NodeId], scale: f32) -> Result<(), ModelError> {
    for &node in nodes {
        if let Some(grad) = graph.grad_mut(node)? {
            let scaled = grad.scale(scale);
            *grad = scaled;
        }
    }
    Ok(())
}

/// Adds source gradients into the existing gradients of the given nodes.
///
/// For each node, if the node already has a gradient, the corresponding source
/// gradient is added to it element-wise.  If the node has no gradient yet, the
/// source gradient is cloned and set directly.  Source entries that are `None`
/// are skipped.
///
/// `nodes` and `source_grads` must have the same length.
pub fn accumulate_gradients(
    graph: &mut Graph,
    nodes: &[NodeId],
    source_grads: &[Option<Tensor>],
) -> Result<(), ModelError> {
    assert_eq!(
        nodes.len(),
        source_grads.len(),
        "nodes and source_grads must have the same length"
    );
    for (i, &node) in nodes.iter().enumerate() {
        if let Some(src) = &source_grads[i] {
            let existing = graph.grad(node)?;
            let new_grad = match existing {
                Some(current) => current.add(src)?,
                None => src.clone(),
            };
            graph.set_grad(node, new_grad)?;
        }
    }
    Ok(())
}

/// Collects the current gradients for a set of nodes as owned tensors.
///
/// Returns a `Vec` where each entry is `Some(grad.clone())` if a gradient
/// exists for the corresponding node, or `None` otherwise.
pub fn collect_gradients(
    graph: &Graph,
    nodes: &[NodeId],
) -> Result<Vec<Option<Tensor>>, ModelError> {
    let mut grads = Vec::with_capacity(nodes.len());
    for &node in nodes {
        grads.push(graph.grad(node)?.cloned());
    }
    Ok(grads)
}

/// Runs one training step with gradient accumulation across multiple
/// micro-batches.
///
/// This is the SGD variant.  The caller supplies a closure that, given a
/// mutable `Graph` reference, creates a fresh micro-batch forward pass and
/// returns `(prediction_node, target_node)`.  The closure is called
/// `accumulation_steps` times.
///
/// For each micro-batch the loss is scaled by `1 / accumulation_steps` so
/// that the accumulated gradients approximate the gradient over the
/// effective (large) batch.  The optimizer is stepped only once, after all
/// micro-batches.
///
/// Returns the average loss across the micro-batches.
pub fn train_step_sgd_with_accumulation<F>(
    graph: &mut Graph,
    optimizer: &mut Sgd,
    trainable_nodes: &[NodeId],
    accumulation_steps: usize,
    loss_fn: SupervisedLoss,
    mut micro_batch_fn: F,
) -> Result<f32, ModelError>
where
    F: FnMut(&mut Graph) -> Result<(NodeId, NodeId), ModelError>,
{
    train_step_with_accumulation_impl(
        graph,
        optimizer,
        trainable_nodes,
        accumulation_steps,
        loss_fn,
        &mut micro_batch_fn,
    )
}

/// Runs one training step with gradient accumulation across multiple
/// micro-batches using the Adam optimizer.
pub fn train_step_adam_with_accumulation<F>(
    graph: &mut Graph,
    optimizer: &mut Adam,
    trainable_nodes: &[NodeId],
    accumulation_steps: usize,
    loss_fn: SupervisedLoss,
    mut micro_batch_fn: F,
) -> Result<f32, ModelError>
where
    F: FnMut(&mut Graph) -> Result<(NodeId, NodeId), ModelError>,
{
    train_step_with_accumulation_impl(
        graph,
        optimizer,
        trainable_nodes,
        accumulation_steps,
        loss_fn,
        &mut micro_batch_fn,
    )
}

/// Runs one training step with gradient accumulation across multiple
/// micro-batches using the AdamW optimizer.
pub fn train_step_adamw_with_accumulation<F>(
    graph: &mut Graph,
    optimizer: &mut AdamW,
    trainable_nodes: &[NodeId],
    accumulation_steps: usize,
    loss_fn: SupervisedLoss,
    mut micro_batch_fn: F,
) -> Result<f32, ModelError>
where
    F: FnMut(&mut Graph) -> Result<(NodeId, NodeId), ModelError>,
{
    train_step_with_accumulation_impl(
        graph,
        optimizer,
        trainable_nodes,
        accumulation_steps,
        loss_fn,
        &mut micro_batch_fn,
    )
}

/// Runs one training step with gradient accumulation across multiple
/// micro-batches using the RMSProp optimizer.
pub fn train_step_rmsprop_with_accumulation<F>(
    graph: &mut Graph,
    optimizer: &mut RmsProp,
    trainable_nodes: &[NodeId],
    accumulation_steps: usize,
    loss_fn: SupervisedLoss,
    mut micro_batch_fn: F,
) -> Result<f32, ModelError>
where
    F: FnMut(&mut Graph) -> Result<(NodeId, NodeId), ModelError>,
{
    train_step_with_accumulation_impl(
        graph,
        optimizer,
        trainable_nodes,
        accumulation_steps,
        loss_fn,
        &mut micro_batch_fn,
    )
}

#[allow(clippy::type_complexity)]
fn train_step_with_accumulation_impl<O: GraphOptimizer>(
    graph: &mut Graph,
    optimizer: &mut O,
    trainable_nodes: &[NodeId],
    accumulation_steps: usize,
    loss_fn: SupervisedLoss,
    micro_batch_fn: &mut dyn FnMut(&mut Graph) -> Result<(NodeId, NodeId), ModelError>,
) -> Result<f32, ModelError> {
    if accumulation_steps == 0 {
        return Err(ModelError::InvalidAccumulationSteps {
            steps: accumulation_steps,
        });
    }

    let scale = 1.0 / accumulation_steps as f32;
    let mut accumulated: Vec<Option<Tensor>> = vec![None; trainable_nodes.len()];
    let mut total_loss = 0.0f32;

    for _ in 0..accumulation_steps {
        // Zero grads before each micro-batch backward.
        graph.zero_grads();

        let (prediction, target) = micro_batch_fn(graph)?;
        let loss_node = build_loss_node(graph, prediction, target, loss_fn)?;
        graph.backward(loss_node)?;

        let loss_value = graph.value(loss_node)?.data()[0];
        total_loss += loss_value;

        // Collect this micro-batch's gradients and accumulate with scaling.
        for (i, &node) in trainable_nodes.iter().enumerate() {
            if let Some(grad) = graph.grad(node)? {
                let scaled = grad.scale(scale);
                accumulated[i] = Some(match accumulated[i].take() {
                    Some(acc) => acc.add(&scaled)?,
                    None => scaled,
                });
            }
        }
    }

    // Write the accumulated gradients back and step the optimizer.
    for (i, &node) in trainable_nodes.iter().enumerate() {
        if let Some(grad) = accumulated[i].take() {
            graph.set_grad(node, grad)?;
        }
    }

    for &node in trainable_nodes {
        optimizer.step_graph_node(graph, node)?;
    }

    Ok(total_loss / accumulation_steps as f32)
}

/// Batch inference on a SequentialModel (tensor mode, no autograd graph).
pub fn infer_batch(
    model: &SequentialModel,
    input: &yscv_tensor::Tensor,
) -> Result<yscv_tensor::Tensor, ModelError> {
    model.forward_inference(input)
}

/// Runs inference through the autograd graph and returns the output tensor value.
pub fn infer_batch_graph(
    graph: &mut Graph,
    model: &SequentialModel,
    input: yscv_tensor::Tensor,
) -> Result<yscv_tensor::Tensor, ModelError> {
    let persistent = model.persistent_node_count();
    graph.truncate(persistent)?;
    let input_node = graph.variable(input);
    let output_node = model.forward(graph, input_node)?;
    Ok(graph.value(output_node)?.clone())
}

// ── Distributed training epoch ─────────────────────────────────────

/// Train one epoch with distributed gradient synchronization.
///
/// After each batch's backward pass, gradients are collected from the
/// trainable parameter nodes, aggregated across all ranks using the
/// provided [`GradientAggregator`] (e.g. `AllReduceAggregator` or
/// `LocalAggregator` for single-rank), written back, and then the
/// optimizer is stepped.  This is the data-parallel training pattern.
///
/// The caller supplies a closure `train_batch_fn` that, given the graph
/// and a batch index, must:
///   1. Set up the forward pass for the batch (feed inputs, compute
///      prediction).
///   2. Compute the loss and call `graph.backward(loss_node)`.
///   3. Return `Ok(loss_scalar)`.
///
/// The function returns the mean loss across all batches.
#[allow(private_bounds)]
pub fn train_epoch_distributed<F, O: GraphOptimizer>(
    graph: &mut Graph,
    optimizer: &mut O,
    aggregator: &mut dyn GradientAggregator,
    trainable_nodes: &[NodeId],
    num_batches: usize,
    train_batch_fn: &mut F,
) -> Result<EpochMetrics, ModelError>
where
    F: FnMut(&mut Graph, usize) -> Result<f32, ModelError>,
{
    if num_batches == 0 {
        return Err(ModelError::EmptyDataset);
    }

    let mut loss_sum = 0.0f32;

    for batch_idx in 0..num_batches {
        // 1. Run the user's forward+backward pass for this batch.
        let loss_value = train_batch_fn(graph, batch_idx)?;
        loss_sum += loss_value;

        // 2. Collect gradients from the trainable nodes into tensors.
        let mut local_grads = Vec::with_capacity(trainable_nodes.len());
        for &node in trainable_nodes {
            let grad = match graph.grad(node)?.cloned() {
                Some(g) => g,
                None => {
                    // If a node has no gradient, use a zero tensor with
                    // the same shape as the parameter value.
                    let val = graph.value(node)?;
                    Tensor::zeros(val.shape().to_vec())?
                }
            };
            local_grads.push(grad);
        }

        // 3. Aggregate gradients across all ranks.
        let aggregated = aggregator.aggregate(&local_grads)?;

        // 4. Write the aggregated gradients back and step the optimizer.
        for (i, &node) in trainable_nodes.iter().enumerate() {
            graph.set_grad(node, aggregated[i].clone())?;
        }

        for &node in trainable_nodes {
            optimizer.step_graph_node(graph, node)?;
        }
    }

    Ok(EpochMetrics {
        mean_loss: loss_sum / num_batches as f32,
        steps: num_batches,
    })
}

/// Convenience wrapper: train one distributed epoch over a
/// [`SequentialModel`] and [`SupervisedDataset`] with SGD.
///
/// This mirrors [`train_epoch_sgd`] but inserts an aggregation step
/// between backward and optimizer update on every batch.
pub fn train_epoch_distributed_sgd(
    graph: &mut Graph,
    model: &SequentialModel,
    optimizer: &mut Sgd,
    aggregator: &mut dyn GradientAggregator,
    dataset: &SupervisedDataset,
    batch_size: usize,
    loss: SupervisedLoss,
) -> Result<EpochMetrics, ModelError> {
    if dataset.is_empty() {
        return Err(ModelError::EmptyDataset);
    }
    let batches: Vec<_> = dataset
        .batches_with_options(batch_size, BatchIterOptions::default())?
        .collect();
    let trainable_nodes = model.trainable_nodes();
    let persistent = model.persistent_node_count();
    let num_batches = batches.len();

    let mut batch_iter = batches.into_iter();

    train_epoch_distributed(
        graph,
        optimizer,
        aggregator,
        &trainable_nodes,
        num_batches,
        &mut |g, _batch_idx| {
            let batch = batch_iter.next().ok_or(ModelError::EmptyDataset)?;
            g.truncate(persistent)?;
            let input = g.constant(batch.inputs);
            let target = g.constant(batch.targets);
            let prediction = model.forward(g, input)?;
            let loss_node = build_loss_node(g, prediction, target, loss)?;
            g.backward(loss_node)?;
            let loss_value = g.value(loss_node)?.data()[0];
            Ok(loss_value)
        },
    )
}
