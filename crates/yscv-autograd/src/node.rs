use yscv_tensor::Tensor;

/// Identifier of one node inside a computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Op {
    Leaf,
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    MatMul2D(NodeId, NodeId),
    Relu(NodeId),
    Neg(NodeId),
    Exp(NodeId),
    Log(NodeId),
    Sqrt(NodeId),
    Sigmoid(NodeId),
    Tanh(NodeId),
    Abs(NodeId),
    Pow(NodeId, NodeId),
    Clamp {
        input: NodeId,
        min_bits: u32,
        max_bits: u32,
    },
    LeakyRelu {
        input: NodeId,
        negative_slope: u32,
    },
    Sum(NodeId),
    Mean(NodeId),
    /// NHWC conv2d: (input, weight, optional bias).
    Conv2dNhwc {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_h: u16,
        stride_w: u16,
    },
    /// NHWC max-pool: stores input id; argmax indices in Node::aux.
    MaxPool2dNhwc {
        input: NodeId,
        kernel_h: u16,
        kernel_w: u16,
        stride_h: u16,
        stride_w: u16,
    },
    /// NHWC average-pool.
    AvgPool2dNhwc {
        input: NodeId,
        kernel_h: u16,
        kernel_w: u16,
        stride_h: u16,
        stride_w: u16,
    },
    /// NHWC batch-normalization (inference mode): (input, gamma, beta, running_mean, running_var).
    BatchNorm2dNhwc {
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        running_mean: NodeId,
        running_var: NodeId,
        epsilon: u32,
    },
    /// Flatten rank-4 NHWC to rank-2 [N, H*W*C].
    Flatten(NodeId),
    /// Softmax along last dimension.
    Softmax(NodeId),
    /// Log-softmax along last dimension.
    LogSoftmax(NodeId),
    /// Transpose (permute). Stores the permutation as indices.
    Transpose2D(NodeId),
    /// Reshape (stores original shape for backward).
    ReshapeView {
        input: NodeId,
    },
    /// Unsqueeze (stores axis for backward).
    UnsqueezeView {
        input: NodeId,
        axis: u16,
    },
    /// Squeeze (stores axis for backward).
    SqueezeView {
        input: NodeId,
        axis: u16,
    },
    /// Concatenation along an axis. Stores (inputs, axis).
    Cat {
        inputs: Vec<NodeId>,
        axis: u16,
    },
    /// Select single index along axis. Stores (input, axis, index).
    Select {
        input: NodeId,
        axis: u16,
        index: u32,
    },
    /// Narrow (slice) along axis. Stores (input, axis, start, len).
    Narrow {
        input: NodeId,
        axis: u16,
        start: u32,
        len: u32,
    },
    /// Gather along axis with index tensor. Index NodeId stored for backward.
    Gather {
        input: NodeId,
        axis: u16,
        index: NodeId,
    },
    /// Scatter-add along axis: out = input.scatter_add(axis, index, src).
    ScatterAdd {
        input: NodeId,
        axis: u16,
        index: NodeId,
        src: NodeId,
    },
    /// Constant-pad along all dimensions. Stores (input, pad_before per dim, pad_after per dim).
    Pad {
        input: NodeId,
        pad_before: Vec<u32>,
        pad_after: Vec<u32>,
    },
    /// Repeat/tile the tensor along each axis.
    Repeat {
        input: NodeId,
        repeats: Vec<u32>,
    },
    /// Sum reduction along a single axis (keeps dims = false).
    SumAxis {
        input: NodeId,
        axis: u16,
    },
    /// Mean reduction along a single axis (keeps dims = false).
    MeanAxis {
        input: NodeId,
        axis: u16,
    },
    /// GELU activation (fast approximation).
    Gelu(NodeId),
    /// SiLU (Swish) activation.
    Silu(NodeId),
    /// Mish activation.
    Mish(NodeId),
    /// NHWC depthwise conv2d: (input, weight, optional bias).
    /// weight shape: [KH, KW, C, 1] (one filter per channel).
    DepthwiseConv2dNhwc {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_h: u16,
        stride_w: u16,
    },
    /// NHWC transposed conv2d: (input, weight, optional bias).
    /// input shape: [N, H, W, C_in], weight shape: [KH, KW, C_out, C_in].
    ConvTranspose2dNhwc {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_h: u16,
        stride_w: u16,
    },
    /// NHWC adaptive average pool 2d.
    AdaptiveAvgPool2dNhwc {
        input: NodeId,
        out_h: u16,
        out_w: u16,
    },
    /// NHWC adaptive max pool 2d (argmax indices stored in aux).
    AdaptiveMaxPool2dNhwc {
        input: NodeId,
        out_h: u16,
        out_w: u16,
    },
    /// Instance normalization (NHWC): per-(N,C) normalization.
    InstanceNormNhwc {
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        eps_bits: u32,
    },
    /// PReLU activation: max(0,x) + alpha * min(0,x).
    /// alpha is per-channel or scalar, stored as a parameter node.
    PRelu {
        input: NodeId,
        alpha: NodeId,
    },
    /// Scatter: write values from `src` into `input` at row positions given by `indices`.
    /// input shape: [N, D], indices shape: [M], src shape: [M, D].
    Scatter {
        input: NodeId,
        indices: NodeId,
        src: NodeId,
    },
    /// Embedding lookup: gather rows from weight matrix at given indices.
    /// weight shape: [vocab_size, embed_dim], indices shape: [seq_len].
    EmbeddingLookup {
        weight: NodeId,
        indices: NodeId,
    },
    /// Layer normalization over the last dimension.
    /// Stores (input, gamma, beta, eps_bits). Aux stores normalized x_hat.
    LayerNorm {
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        eps_bits: u32,
    },
    /// Group normalization (NHWC layout).
    /// Stores (input, gamma, beta, num_groups, eps_bits). Aux stores normalized x_hat.
    GroupNorm {
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        num_groups: u16,
        eps_bits: u32,
    },
    /// NLC 1-D convolution: (input, weight, optional bias).
    /// input shape: [N, L, C_in], weight shape: [K, C_in, C_out].
    Conv1dNlc {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride: u16,
    },
    /// NDHWC 3-D convolution: (input, weight, optional bias).
    /// input shape: [N, D, H, W, C_in], weight shape: [KD, KH, KW, C_in, C_out].
    Conv3dNdhwc {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride_d: u16,
        stride_h: u16,
        stride_w: u16,
    },
    /// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
    /// Q shape: [seq_q, d_k], K shape: [seq_k, d_k], V shape: [seq_k, d_v].
    /// Aux stores the attention weights [seq_q, seq_k] for backward.
    ScaledDotProductAttention {
        query: NodeId,
        key: NodeId,
        value: NodeId,
    },
    /// Pixel shuffle: rearranges [N, H, W, C*r^2] -> [N, H*r, W*r, C].
    PixelShuffle {
        input: NodeId,
        upscale_factor: u16,
    },
    /// Nearest-neighbor upsample: [N, H, W, C] -> [N, H*r, W*r, C].
    UpsampleNearest {
        input: NodeId,
        scale_factor: u16,
    },
    /// Vanilla RNN forward (BPTT). Aux stores all hidden states.
    Rnn {
        input: NodeId,
        w_ih: NodeId,
        w_hh: NodeId,
        bias: NodeId,
    },
    /// LSTM forward (BPTT). Aux stores all hidden + cell states and gate values.
    Lstm {
        input: NodeId,
        w_ih: NodeId,
        w_hh: NodeId,
        bias: NodeId,
    },
    /// GRU forward (BPTT). Aux stores all hidden states and gate values.
    Gru {
        input: NodeId,
        w_ih: NodeId,
        w_hh: NodeId,
        bias_ih: NodeId,
        bias_hh: NodeId,
    },
    /// Deformable conv2d NHWC: (input, weight, offsets, optional bias).
    DeformableConv2dNhwc {
        input: NodeId,
        weight: NodeId,
        offsets: NodeId,
        bias: Option<NodeId>,
        stride: u16,
        padding: u16,
    },
}

/// Auxiliary data stored during forward for certain ops that need it in backward.
#[derive(Debug, Clone)]
pub(crate) enum AuxData {
    /// Max-pool argmax indices (flattened offsets into input spatial*channel plane per output element).
    MaxPoolIndices(Vec<usize>),
    /// Batch-norm: pre-normalized `(input - mean) / sqrt(var + eps)` for gamma gradient.
    BatchNormNormalized(Tensor),
    /// Layer-norm / group-norm: pre-normalized x_hat for gamma gradient.
    NormNormalized(Tensor),
    /// Attention weights: softmax(Q @ K^T / sqrt(d_k)), shape [seq_q, seq_k].
    AttentionWeights(Tensor),
    /// RNN hidden states at each timestep: Vec of [batch, hidden_size].
    RnnHiddenStates(Vec<Tensor>),
    /// LSTM states: (hidden_states, cell_states, gate_values_per_step).
    /// gate_values_per_step: Vec of (i, f, g, o, c_t) per timestep.
    LstmStates {
        hidden_states: Vec<Tensor>,
        cell_states: Vec<Tensor>,
        /// Each element: (i_gate, f_gate, g_gate, o_gate) per timestep.
        gates: Vec<(Tensor, Tensor, Tensor, Tensor)>,
    },
    /// GRU states: (hidden_states, gate_values_per_step).
    /// Each gate element: (r_gate, z_gate, n_candidate) per timestep.
    GruStates {
        hidden_states: Vec<Tensor>,
        gates: Vec<(Tensor, Tensor, Tensor)>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) value: Tensor,
    pub(crate) grad: Option<Tensor>,
    pub(crate) requires_grad: bool,
    pub(crate) op: Op,
    pub(crate) aux: Option<AuxData>,
}
