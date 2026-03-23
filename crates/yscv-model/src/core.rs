//! Model definitions, losses, checkpoints, and training helpers for yscv.
#![forbid(unsafe_code)]

pub const CRATE_ID: &str = "yscv-model";

#[path = "attention.rs"]
mod attention;
#[path = "augmentation.rs"]
mod augmentation;
#[path = "batch_infer.rs"]
mod batch_infer;
#[path = "blocks.rs"]
mod blocks;
#[path = "callbacks.rs"]
mod callbacks;
#[path = "checkpoint.rs"]
mod checkpoint;
#[path = "checkpoint_state.rs"]
mod checkpoint_state;
#[path = "data_loader.rs"]
mod data_loader;
#[path = "dataset/mod.rs"]
mod dataset;
#[path = "distributed.rs"]
mod distributed;
#[path = "ema.rs"]
mod ema;
#[path = "error.rs"]
mod error;
#[path = "fusion.rs"]
mod fusion;
#[path = "hub.rs"]
mod hub;
#[path = "init.rs"]
mod init;
#[path = "layers/mod.rs"]
mod layers;
#[path = "lora.rs"]
mod lora;
#[path = "loss.rs"]
mod loss;
#[path = "lr_finder.rs"]
mod lr_finder;
#[path = "mixed_precision.rs"]
mod mixed_precision;
#[path = "onnx_export.rs"]
mod onnx_export;
#[path = "pipeline.rs"]
mod pipeline;
#[path = "quantize.rs"]
mod quantize;
#[path = "recurrent.rs"]
mod recurrent;
#[path = "safetensors.rs"]
mod safetensors;
#[path = "sequential.rs"]
mod sequential;
#[path = "tcp_transport.rs"]
pub mod tcp_transport;
#[path = "tensorboard.rs"]
mod tensorboard;
#[path = "train.rs"]
mod train;
#[path = "trainer.rs"]
mod trainer;
#[path = "training_log.rs"]
mod training_log;
#[path = "transform.rs"]
mod transform;
#[path = "transformer_decoder.rs"]
mod transformer_decoder;
#[path = "weight_mapping.rs"]
mod weight_mapping;
#[path = "weights.rs"]
mod weights;
#[path = "zoo.rs"]
mod zoo;

pub use attention::{
    FeedForward, MultiHeadAttention, MultiHeadAttentionConfig, TransformerEncoderBlock,
    generate_causal_mask, generate_padding_mask, scaled_dot_product_attention,
};
pub use augmentation::{ImageAugmentationOp, ImageAugmentationPipeline};
pub use batch_infer::{BatchCollector, DynamicBatchConfig, batched_inference};
pub use blocks::{
    AnchorFreeHead, FpnNeck, MbConvBlock, PatchEmbedding, SqueezeExciteBlock, UNetDecoderStage,
    UNetEncoderStage, VisionTransformer, add_bottleneck_block, add_residual_block,
    build_resnet_feature_extractor, build_simple_cnn_classifier,
};
pub use callbacks::{
    BestModelCheckpoint, EarlyStopping, MetricsLogger, MonitorMode, TrainingCallback,
    train_epochs_with_callbacks,
};
pub use checkpoint::{
    LayerCheckpoint, SequentialCheckpoint, TensorSnapshot, checkpoint_from_json, checkpoint_to_json,
};
pub use checkpoint_state::{
    adam_state_from_map, adam_state_to_map, load_training_checkpoint, save_training_checkpoint,
    sgd_state_from_map, sgd_state_to_map,
};
pub use data_loader::{
    DataLoader, DataLoaderBatch, DataLoaderConfig, DataLoaderIter, RandomSampler,
    SequentialSampler, StreamingDataLoader, WeightedRandomSampler,
};
pub use dataset::{
    Batch, BatchIterOptions, CutMixConfig, DatasetSplit, ImageFolderTargetMode, MiniBatchIter,
    MixUpConfig, SamplingPolicy, SupervisedCsvConfig, SupervisedDataset,
    SupervisedImageFolderConfig, SupervisedImageFolderLoadResult, SupervisedImageManifestConfig,
    SupervisedJsonlConfig, load_supervised_dataset_csv_file, load_supervised_dataset_jsonl_file,
    load_supervised_image_folder_dataset, load_supervised_image_folder_dataset_with_classes,
    load_supervised_image_manifest_csv_file, parse_supervised_dataset_csv,
    parse_supervised_dataset_jsonl, parse_supervised_image_manifest_csv,
};
pub use distributed::{
    AllReduceAggregator, CompressedGradient, DataParallelConfig, DistributedConfig,
    GradientAggregator, InProcessTransport, LocalAggregator, ParameterServer,
    PipelineParallelConfig, PipelineStage, TopKCompressor, Transport, compress_gradients,
    decompress_gradients, distributed_train_step, gather_shards, shard_tensor, split_into_stages,
};
pub use ema::ExponentialMovingAverage;
pub use error::ModelError;
pub use fusion::{fuse_conv_bn, optimize_sequential};
pub use hub::{HubEntry, ModelHub, default_cache_dir};
pub use init::{
    constant, kaiming_normal, kaiming_uniform, orthogonal, xavier_normal, xavier_uniform,
};
pub use layers::{
    AdaptiveAvgPool2dLayer, AdaptiveMaxPool2dLayer, AvgPool2dLayer, BatchNorm2dLayer, Conv1dLayer,
    Conv2dLayer, Conv3dLayer, ConvTranspose2dLayer, DeformableConv2dLayer, DepthwiseConv2dLayer,
    DropoutLayer, EmbeddingLayer, FeedForwardLayer, FlattenLayer, GELULayer, GlobalAvgPool2dLayer,
    GroupNormLayer, GruLayer, InstanceNormLayer, LayerNormLayer, LeakyReLULayer, LinearLayer,
    LstmLayer, MaskHead, MaxPool2dLayer, MishLayer, ModelLayer, MultiHeadAttentionLayer,
    PReLULayer, PixelShuffleLayer, ReLULayer, ResidualBlock, RnnLayer, SeparableConv2dLayer,
    SiLULayer, SigmoidLayer, SoftmaxLayer, TanhLayer, TransformerEncoderLayer, UpsampleLayer,
};
pub use lora::{LoraConfig, LoraLinear};
pub use loss::{
    bce_loss, contrastive_loss, cosine_embedding_loss, cross_entropy_loss, ctc_loss, dice_loss,
    distillation_loss, focal_loss, hinge_loss, huber_loss, kl_div_loss,
    label_smoothing_cross_entropy, mae_loss, mse_loss, nll_loss, smooth_l1_loss, triplet_loss,
};
pub use lr_finder::{LrFinderConfig, LrFinderResult, lr_range_test};
pub use mixed_precision::{
    DynamicLossScaler, MixedPrecisionConfig, cast_params_for_forward, cast_to_master,
    mixed_precision_train_step,
};
pub use onnx_export::{export_sequential_to_onnx, export_sequential_to_onnx_file};
pub use pipeline::InferencePipeline;
pub use quantize::{
    PerChannelQuantResult, PrunedTensor, QuantMode, QuantizedTensor, apply_pruning_mask,
    dequantize_weights, prune_magnitude, quantize_per_channel, quantize_weights, quantized_matmul,
};
pub use recurrent::{
    GruCell, LstmCell, RnnCell, bilstm_forward_sequence, gru_forward_sequence,
    lstm_forward_sequence, rnn_forward_sequence,
};
pub use safetensors::{SafeTensorDType, SafeTensorFile, TensorInfo, load_state_dict};
pub use sequential::SequentialModel;
pub use tcp_transport::{NodeRole, TcpAllReduceAggregator, TcpTransport, loopback_pair};
pub use tensorboard::{TensorBoardCallback, TensorBoardWriter};
pub use train::{
    CnnTrainConfig, EpochMetrics, EpochTrainOptions, OptimizerType, ScheduledEpochMetrics,
    SchedulerTrainOptions, SupervisedLoss, accumulate_gradients, collect_gradients, infer_batch,
    infer_batch_graph, scale_gradients, train_cnn_epoch_adam, train_cnn_epoch_adamw,
    train_cnn_epoch_sgd, train_cnn_epochs, train_epoch_adam, train_epoch_adam_with_loss,
    train_epoch_adam_with_options, train_epoch_adam_with_options_and_loss, train_epoch_adamw,
    train_epoch_adamw_with_loss, train_epoch_adamw_with_options,
    train_epoch_adamw_with_options_and_loss, train_epoch_distributed, train_epoch_distributed_sgd,
    train_epoch_rmsprop, train_epoch_rmsprop_with_loss, train_epoch_rmsprop_with_options,
    train_epoch_rmsprop_with_options_and_loss, train_epoch_sgd, train_epoch_sgd_with_loss,
    train_epoch_sgd_with_options, train_epoch_sgd_with_options_and_loss,
    train_epochs_adam_with_scheduler, train_epochs_adam_with_scheduler_and_loss,
    train_epochs_adamw_with_scheduler, train_epochs_adamw_with_scheduler_and_loss,
    train_epochs_rmsprop_with_scheduler, train_epochs_rmsprop_with_scheduler_and_loss,
    train_epochs_sgd_with_scheduler, train_epochs_sgd_with_scheduler_and_loss, train_step_adam,
    train_step_adam_with_accumulation, train_step_adam_with_loss, train_step_adamw,
    train_step_adamw_with_accumulation, train_step_adamw_with_loss, train_step_rmsprop,
    train_step_rmsprop_with_accumulation, train_step_rmsprop_with_loss, train_step_sgd,
    train_step_sgd_with_accumulation, train_step_sgd_with_loss,
};
pub use trainer::{LossKind, OptimizerKind, TrainResult, Trainer, TrainerConfig};
pub use training_log::TrainingLog;
pub use transform::{
    CenterCrop, Compose, GaussianBlur, Normalize, PermuteDims, RandomHorizontalFlip, Resize,
    ScaleValues, Transform,
};
pub use transformer_decoder::{CrossAttention, TransformerDecoder, TransformerDecoderBlock};
pub use weight_mapping::{remap_state_dict, timm_to_yscv_name};
pub use weights::{inspect_weights, load_weights, save_weights};
pub use zoo::{
    ArchitectureConfig, ModelArchitecture, ModelZoo, build_alexnet, build_classifier,
    build_feature_extractor, build_mobilenet_v2, build_resnet, build_resnet_custom, build_vgg,
};

#[path = "tests/mod.rs"]
#[cfg(test)]
mod tests;
