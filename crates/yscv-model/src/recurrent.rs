use yscv_kernels::matmul_2d;
use yscv_tensor::Tensor;

use crate::ModelError;

/// Vanilla RNN cell: h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b).
#[derive(Debug, Clone)]
pub struct RnnCell {
    pub w_ih: Tensor, // [input_size, hidden_size]
    pub w_hh: Tensor, // [hidden_size, hidden_size]
    pub bias: Tensor, // [hidden_size]
    pub hidden_size: usize,
}

impl RnnCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, ModelError> {
        Ok(Self {
            w_ih: Tensor::from_vec(
                vec![input_size, hidden_size],
                vec![0.0; input_size * hidden_size],
            )?,
            w_hh: Tensor::from_vec(
                vec![hidden_size, hidden_size],
                vec![0.0; hidden_size * hidden_size],
            )?,
            bias: Tensor::from_vec(vec![hidden_size], vec![0.0; hidden_size])?,
            hidden_size,
        })
    }

    /// Forward one timestep: x `[batch, input_size]`, h `[batch, hidden_size]` -> h' `[batch, hidden_size]`.
    pub fn forward(&self, x: &Tensor, h: &Tensor) -> Result<Tensor, ModelError> {
        let xw = matmul_2d(x, &self.w_ih)?;
        let hw = matmul_2d(h, &self.w_hh)?;
        let sum = xw.add(&hw)?;
        let sum = sum.add(&self.bias.unsqueeze(0)?)?;
        let data: Vec<f32> = sum.data().iter().map(|&v| v.tanh()).collect();
        Tensor::from_vec(sum.shape().to_vec(), data).map_err(Into::into)
    }
}

/// LSTM cell: standard gates (input, forget, cell, output).
#[derive(Debug, Clone)]
pub struct LstmCell {
    pub w_ih: Tensor, // [input_size, 4 * hidden_size]
    pub w_hh: Tensor, // [hidden_size, 4 * hidden_size]
    pub bias: Tensor, // [4 * hidden_size]
    pub hidden_size: usize,
}

impl LstmCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, ModelError> {
        let h4 = 4 * hidden_size;
        Ok(Self {
            w_ih: Tensor::from_vec(vec![input_size, h4], vec![0.0; input_size * h4])?,
            w_hh: Tensor::from_vec(vec![hidden_size, h4], vec![0.0; hidden_size * h4])?,
            bias: Tensor::from_vec(vec![h4], vec![0.0; h4])?,
            hidden_size,
        })
    }

    /// Forward one timestep. Returns `(h_new, c_new)`.
    ///
    /// x: `[batch, input_size]`, h: `[batch, hidden_size]`, c: `[batch, hidden_size]`.
    pub fn forward(
        &self,
        x: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor), ModelError> {
        let batch = x.shape()[0];
        let hs = self.hidden_size;

        let gates = {
            let xw = matmul_2d(x, &self.w_ih)?;
            let hw = matmul_2d(h, &self.w_hh)?;
            let g = xw.add(&hw)?;
            g.add(&self.bias.unsqueeze(0)?)?
        };

        let gd = gates.data();
        let cd = c.data();
        let mut h_new = Vec::with_capacity(batch * hs);
        let mut c_new = Vec::with_capacity(batch * hs);

        for b in 0..batch {
            let base = b * 4 * hs;
            for j in 0..hs {
                let i_gate = sigmoid_f32(gd[base + j]);
                let f_gate = sigmoid_f32(gd[base + hs + j]);
                let g_gate = gd[base + 2 * hs + j].tanh();
                let o_gate = sigmoid_f32(gd[base + 3 * hs + j]);
                let c_val = f_gate * cd[b * hs + j] + i_gate * g_gate;
                let h_val = o_gate * c_val.tanh();
                c_new.push(c_val);
                h_new.push(h_val);
            }
        }

        let h_out = Tensor::from_vec(vec![batch, hs], h_new)?;
        let c_out = Tensor::from_vec(vec![batch, hs], c_new)?;
        Ok((h_out, c_out))
    }
}

/// GRU cell: update and reset gates.
#[derive(Debug, Clone)]
pub struct GruCell {
    pub w_ih: Tensor,    // [input_size, 3 * hidden_size]
    pub w_hh: Tensor,    // [hidden_size, 3 * hidden_size]
    pub bias_ih: Tensor, // [3 * hidden_size]
    pub bias_hh: Tensor, // [3 * hidden_size]
    pub hidden_size: usize,
}

impl GruCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, ModelError> {
        let h3 = 3 * hidden_size;
        Ok(Self {
            w_ih: Tensor::from_vec(vec![input_size, h3], vec![0.0; input_size * h3])?,
            w_hh: Tensor::from_vec(vec![hidden_size, h3], vec![0.0; hidden_size * h3])?,
            bias_ih: Tensor::from_vec(vec![h3], vec![0.0; h3])?,
            bias_hh: Tensor::from_vec(vec![h3], vec![0.0; h3])?,
            hidden_size,
        })
    }

    /// Forward one timestep: x `[batch, input_size]`, h `[batch, hidden_size]` -> h' `[batch, hidden_size]`.
    pub fn forward(&self, x: &Tensor, h: &Tensor) -> Result<Tensor, ModelError> {
        let batch = x.shape()[0];
        let hs = self.hidden_size;

        let xw = matmul_2d(x, &self.w_ih)?;
        let xw = xw.add(&self.bias_ih.unsqueeze(0)?)?;
        let hw = matmul_2d(h, &self.w_hh)?;
        let hw = hw.add(&self.bias_hh.unsqueeze(0)?)?;

        let xd = xw.data();
        let hd = hw.data();
        let h_prev = h.data();
        let mut h_new = Vec::with_capacity(batch * hs);

        for b in 0..batch {
            let xb = b * 3 * hs;
            let hb = b * 3 * hs;
            for j in 0..hs {
                let r = sigmoid_f32(xd[xb + j] + hd[hb + j]);
                let z = sigmoid_f32(xd[xb + hs + j] + hd[hb + hs + j]);
                let n = (xd[xb + 2 * hs + j] + r * hd[hb + 2 * hs + j]).tanh();
                let h_val = (1.0 - z) * n + z * h_prev[b * hs + j];
                h_new.push(h_val);
            }
        }

        Tensor::from_vec(vec![batch, hs], h_new).map_err(Into::into)
    }
}

// ── Multi-step sequence wrappers ───────────────────────────────────

/// Runs an RNN cell over a sequence `[batch, seq_len, input_size]`.
///
/// Returns all hidden states `[batch, seq_len, hidden_size]` and final hidden `[batch, hidden_size]`.
pub fn rnn_forward_sequence(
    cell: &RnnCell,
    input: &Tensor,
    h0: Option<&Tensor>,
) -> Result<(Tensor, Tensor), ModelError> {
    let shape = input.shape();
    let (batch, seq_len, _input_size) = (shape[0], shape[1], shape[2]);
    let hs = cell.hidden_size;

    let mut h = match h0 {
        Some(h) => h.clone(),
        None => Tensor::from_vec(vec![batch, hs], vec![0.0; batch * hs])?,
    };

    let mut all_h = Vec::with_capacity(batch * seq_len * hs);

    for t in 0..seq_len {
        let xt = input.narrow(1, t, 1)?;
        let xt = xt.reshape(vec![batch, input.shape()[2]])?;
        h = cell.forward(&xt, &h)?;
        all_h.extend_from_slice(h.data());
    }

    let output = Tensor::from_vec(vec![batch, seq_len, hs], all_h)?;
    Ok((output, h))
}

/// Runs an LSTM cell over a sequence `[batch, seq_len, input_size]`.
///
/// Returns all hidden states `[batch, seq_len, hidden_size]`, final `(h, c)`.
pub fn lstm_forward_sequence(
    cell: &LstmCell,
    input: &Tensor,
    h0: Option<&Tensor>,
    c0: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor), ModelError> {
    let shape = input.shape();
    let (batch, seq_len, _input_size) = (shape[0], shape[1], shape[2]);
    let hs = cell.hidden_size;

    let mut h = match h0 {
        Some(h) => h.clone(),
        None => Tensor::from_vec(vec![batch, hs], vec![0.0; batch * hs])?,
    };
    let mut c = match c0 {
        Some(c) => c.clone(),
        None => Tensor::from_vec(vec![batch, hs], vec![0.0; batch * hs])?,
    };

    let mut all_h = Vec::with_capacity(batch * seq_len * hs);

    for t in 0..seq_len {
        let xt = input.narrow(1, t, 1)?;
        let xt = xt.reshape(vec![batch, input.shape()[2]])?;
        let (h_new, c_new) = cell.forward(&xt, &h, &c)?;
        all_h.extend_from_slice(h_new.data());
        h = h_new;
        c = c_new;
    }

    let output = Tensor::from_vec(vec![batch, seq_len, hs], all_h)?;
    Ok((output, h, c))
}

/// Runs a GRU cell over a sequence `[batch, seq_len, input_size]`.
///
/// Returns all hidden states `[batch, seq_len, hidden_size]` and final hidden `[batch, hidden_size]`.
pub fn gru_forward_sequence(
    cell: &GruCell,
    input: &Tensor,
    h0: Option<&Tensor>,
) -> Result<(Tensor, Tensor), ModelError> {
    let shape = input.shape();
    let (batch, seq_len, _input_size) = (shape[0], shape[1], shape[2]);
    let hs = cell.hidden_size;

    let mut h = match h0 {
        Some(h) => h.clone(),
        None => Tensor::from_vec(vec![batch, hs], vec![0.0; batch * hs])?,
    };

    let mut all_h = Vec::with_capacity(batch * seq_len * hs);

    for t in 0..seq_len {
        let xt = input.narrow(1, t, 1)?;
        let xt = xt.reshape(vec![batch, input.shape()[2]])?;
        h = cell.forward(&xt, &h)?;
        all_h.extend_from_slice(h.data());
    }

    let output = Tensor::from_vec(vec![batch, seq_len, hs], all_h)?;
    Ok((output, h))
}

/// Bidirectional LSTM: runs forward and backward LSTMs, concatenates outputs.
///
/// Returns `[batch, seq_len, 2 * hidden_size]`.
pub fn bilstm_forward_sequence(
    fwd_cell: &LstmCell,
    bwd_cell: &LstmCell,
    input: &Tensor,
) -> Result<Tensor, ModelError> {
    let shape = input.shape();
    let (batch, seq_len, input_size) = (shape[0], shape[1], shape[2]);
    let hs = fwd_cell.hidden_size;

    // Forward pass
    let (fwd_out, _, _) = lstm_forward_sequence(fwd_cell, input, None, None)?;

    // Reverse input along time axis
    let mut rev_data = Vec::with_capacity(batch * seq_len * input_size);
    let in_data = input.data();
    for b in 0..batch {
        for t in (0..seq_len).rev() {
            let start = (b * seq_len + t) * input_size;
            rev_data.extend_from_slice(&in_data[start..start + input_size]);
        }
    }
    let rev_input = Tensor::from_vec(vec![batch, seq_len, input_size], rev_data)?;

    // Backward pass
    let (bwd_out_rev, _, _) = lstm_forward_sequence(bwd_cell, &rev_input, None, None)?;

    // Reverse backward output and concatenate
    let fwd_d = fwd_out.data();
    let bwd_d = bwd_out_rev.data();
    let mut out = Vec::with_capacity(batch * seq_len * 2 * hs);
    for b in 0..batch {
        for t in 0..seq_len {
            let fwd_start = (b * seq_len + t) * hs;
            out.extend_from_slice(&fwd_d[fwd_start..fwd_start + hs]);
            let bwd_t = seq_len - 1 - t;
            let bwd_start = (b * seq_len + bwd_t) * hs;
            out.extend_from_slice(&bwd_d[bwd_start..bwd_start + hs]);
        }
    }

    Tensor::from_vec(vec![batch, seq_len, 2 * hs], out).map_err(Into::into)
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
