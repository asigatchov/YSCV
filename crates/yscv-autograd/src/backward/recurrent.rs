use yscv_tensor::Tensor;

use super::error::AutogradError;
use super::graph::Graph;
use super::node::{AuxData, NodeId};

/// RNN backward (BPTT).
pub(crate) fn rnn_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    index: usize,
    input_id: NodeId,
    w_ih_id: NodeId,
    w_hh_id: NodeId,
    bias_id: NodeId,
) -> Result<(), AutogradError> {
    let hidden_states = match &graph.nodes[index].aux {
        Some(AuxData::RnnHiddenStates(hs)) => hs.clone(),
        _ => {
            return Err(AutogradError::InvalidGradientShape {
                node: index,
                expected: vec![],
                got: vec![],
            });
        }
    };

    let iv = &graph.nodes[input_id.0].value;
    let wih = &graph.nodes[w_ih_id.0].value;
    let whh = &graph.nodes[w_hh_id.0].value;

    let in_shape = iv.shape().to_vec();
    let seq_len = in_shape[0];
    let input_size = in_shape[1];
    let hidden_size = wih.shape()[1];

    let in_data = iv.data().to_vec();
    let wih_data = wih.data().to_vec();
    let whh_data = whh.data().to_vec();
    let up_data = upstream.data();

    let mut grad_input = vec![0.0f32; seq_len * input_size];
    let mut grad_wih = vec![0.0f32; input_size * hidden_size];
    let mut grad_whh = vec![0.0f32; hidden_size * hidden_size];
    let mut grad_bias = vec![0.0f32; hidden_size];
    let mut dh_next = vec![0.0f32; hidden_size];

    for t in (0..seq_len).rev() {
        let h_t = hidden_states[t + 1].data();
        let h_prev = hidden_states[t].data();
        let x_base = t * input_size;

        let mut dh = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dh[j] = up_data[t * hidden_size + j] + dh_next[j];
        }

        let mut d_raw = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            d_raw[j] = dh[j] * (1.0 - h_t[j] * h_t[j]);
        }

        for j in 0..hidden_size {
            grad_bias[j] += d_raw[j];
        }

        for i in 0..input_size {
            for j in 0..hidden_size {
                grad_wih[i * hidden_size + j] += in_data[x_base + i] * d_raw[j];
            }
        }

        for i in 0..hidden_size {
            for j in 0..hidden_size {
                grad_whh[i * hidden_size + j] += h_prev[i] * d_raw[j];
            }
        }

        for i in 0..input_size {
            let mut s = 0.0f32;
            for j in 0..hidden_size {
                s += d_raw[j] * wih_data[i * hidden_size + j];
            }
            grad_input[x_base + i] = s;
        }

        dh_next = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            for j in 0..hidden_size {
                dh_next[i] += d_raw[j] * whh_data[i * hidden_size + j];
            }
        }
    }

    if graph.nodes[input_id.0].requires_grad {
        graph.accumulate_grad(input_id, Tensor::from_vec(in_shape, grad_input)?)?;
    }
    if graph.nodes[w_ih_id.0].requires_grad {
        graph.accumulate_grad(
            w_ih_id,
            Tensor::from_vec(vec![input_size, hidden_size], grad_wih)?,
        )?;
    }
    if graph.nodes[w_hh_id.0].requires_grad {
        graph.accumulate_grad(
            w_hh_id,
            Tensor::from_vec(vec![hidden_size, hidden_size], grad_whh)?,
        )?;
    }
    if graph.nodes[bias_id.0].requires_grad {
        graph.accumulate_grad(bias_id, Tensor::from_vec(vec![hidden_size], grad_bias)?)?;
    }
    Ok(())
}

/// LSTM backward (BPTT).
pub(crate) fn lstm_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    index: usize,
    input_id: NodeId,
    w_ih_id: NodeId,
    w_hh_id: NodeId,
    bias_id: NodeId,
) -> Result<(), AutogradError> {
    let (hidden_states, cell_states, gates) = match &graph.nodes[index].aux {
        Some(AuxData::LstmStates {
            hidden_states,
            cell_states,
            gates,
        }) => (hidden_states.clone(), cell_states.clone(), gates.clone()),
        _ => {
            return Err(AutogradError::InvalidGradientShape {
                node: index,
                expected: vec![],
                got: vec![],
            });
        }
    };

    let iv = &graph.nodes[input_id.0].value;
    let wih = &graph.nodes[w_ih_id.0].value;
    let whh = &graph.nodes[w_hh_id.0].value;

    let in_shape = iv.shape().to_vec();
    let seq_len = in_shape[0];
    let input_size = in_shape[1];
    let h4 = wih.shape()[1];
    let hidden_size = h4 / 4;

    let in_data = iv.data().to_vec();
    let wih_data = wih.data().to_vec();
    let whh_data = whh.data().to_vec();
    let up_data = upstream.data();

    let mut grad_input = vec![0.0f32; seq_len * input_size];
    let mut grad_wih = vec![0.0f32; input_size * h4];
    let mut grad_whh = vec![0.0f32; hidden_size * h4];
    let mut grad_bias = vec![0.0f32; h4];
    let mut dh_next = vec![0.0f32; hidden_size];
    let mut dc_next = vec![0.0f32; hidden_size];

    for t in (0..seq_len).rev() {
        let h_prev = hidden_states[t].data();
        let c_prev = cell_states[t].data();
        let c_t = cell_states[t + 1].data();
        let (ref i_gate, ref f_gate, ref g_gate, ref o_gate) = gates[t];
        let ig = i_gate.data();
        let fg = f_gate.data();
        let gg = g_gate.data();
        let og = o_gate.data();
        let x_base = t * input_size;

        let mut dh = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dh[j] = up_data[t * hidden_size + j] + dh_next[j];
        }

        let mut d_o = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            let tanh_c = c_t[j].tanh();
            d_o[j] = dh[j] * tanh_c * og[j] * (1.0 - og[j]);
        }

        let mut dc = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            let tanh_c = c_t[j].tanh();
            dc[j] = dh[j] * og[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
        }

        let mut d_f = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            d_f[j] = dc[j] * c_prev[j] * fg[j] * (1.0 - fg[j]);
        }

        let mut d_i = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            d_i[j] = dc[j] * gg[j] * ig[j] * (1.0 - ig[j]);
        }

        let mut d_g = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            d_g[j] = dc[j] * ig[j] * (1.0 - gg[j] * gg[j]);
        }

        let mut d_gates = vec![0.0f32; h4];
        for j in 0..hidden_size {
            d_gates[j] = d_i[j];
            d_gates[hidden_size + j] = d_f[j];
            d_gates[2 * hidden_size + j] = d_g[j];
            d_gates[3 * hidden_size + j] = d_o[j];
        }

        for j in 0..h4 {
            grad_bias[j] += d_gates[j];
        }

        for i in 0..input_size {
            for j in 0..h4 {
                grad_wih[i * h4 + j] += in_data[x_base + i] * d_gates[j];
            }
        }

        for i in 0..hidden_size {
            for j in 0..h4 {
                grad_whh[i * h4 + j] += h_prev[i] * d_gates[j];
            }
        }

        for i in 0..input_size {
            let mut s = 0.0f32;
            for j in 0..h4 {
                s += d_gates[j] * wih_data[i * h4 + j];
            }
            grad_input[x_base + i] = s;
        }

        dh_next = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            for j in 0..h4 {
                dh_next[i] += d_gates[j] * whh_data[i * h4 + j];
            }
        }

        dc_next = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dc_next[j] = dc[j] * fg[j];
        }
    }

    if graph.nodes[input_id.0].requires_grad {
        graph.accumulate_grad(input_id, Tensor::from_vec(in_shape, grad_input)?)?;
    }
    if graph.nodes[w_ih_id.0].requires_grad {
        graph.accumulate_grad(w_ih_id, Tensor::from_vec(vec![input_size, h4], grad_wih)?)?;
    }
    if graph.nodes[w_hh_id.0].requires_grad {
        graph.accumulate_grad(w_hh_id, Tensor::from_vec(vec![hidden_size, h4], grad_whh)?)?;
    }
    if graph.nodes[bias_id.0].requires_grad {
        graph.accumulate_grad(bias_id, Tensor::from_vec(vec![h4], grad_bias)?)?;
    }
    Ok(())
}

/// GRU backward (BPTT).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gru_backward(
    graph: &mut Graph,
    upstream: &Tensor,
    index: usize,
    input_id: NodeId,
    w_ih_id: NodeId,
    w_hh_id: NodeId,
    bias_ih_id: NodeId,
    bias_hh_id: NodeId,
) -> Result<(), AutogradError> {
    let (hidden_states, gates) = match &graph.nodes[index].aux {
        Some(AuxData::GruStates {
            hidden_states,
            gates,
        }) => (hidden_states.clone(), gates.clone()),
        _ => {
            return Err(AutogradError::InvalidGradientShape {
                node: index,
                expected: vec![],
                got: vec![],
            });
        }
    };

    let iv = &graph.nodes[input_id.0].value;
    let wih = &graph.nodes[w_ih_id.0].value;
    let whh = &graph.nodes[w_hh_id.0].value;

    let in_shape = iv.shape().to_vec();
    let seq_len = in_shape[0];
    let input_size = in_shape[1];
    let h3 = wih.shape()[1];
    let hidden_size = h3 / 3;

    let in_data = iv.data().to_vec();
    let wih_data = wih.data().to_vec();
    let whh_data = whh.data().to_vec();
    let _bih_data = graph.nodes[bias_ih_id.0].value.data().to_vec();
    let bhh_data = graph.nodes[bias_hh_id.0].value.data().to_vec();
    let up_data = upstream.data();

    let mut grad_input = vec![0.0f32; seq_len * input_size];
    let mut grad_wih = vec![0.0f32; input_size * h3];
    let mut grad_whh = vec![0.0f32; hidden_size * h3];
    let mut grad_bih = vec![0.0f32; h3];
    let mut grad_bhh = vec![0.0f32; h3];
    let mut dh_next = vec![0.0f32; hidden_size];

    for t in (0..seq_len).rev() {
        let h_prev = hidden_states[t].data();
        let (ref r_gate, ref z_gate, ref n_candidate) = gates[t];
        let rg = r_gate.data();
        let zg = z_gate.data();
        let ng = n_candidate.data();
        let x_base = t * input_size;

        let mut h_proj_n = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            let mut sum = bhh_data[2 * hidden_size + j];
            for i in 0..hidden_size {
                sum += h_prev[i] * whh_data[i * h3 + 2 * hidden_size + j];
            }
            h_proj_n[j] = sum;
        }

        let mut dh = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dh[j] = up_data[t * hidden_size + j] + dh_next[j];
        }

        let mut dn = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dn[j] = dh[j] * (1.0 - zg[j]);
        }

        let mut dz = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dz[j] = dh[j] * (h_prev[j] - ng[j]) * zg[j] * (1.0 - zg[j]);
        }

        let mut dn_raw = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dn_raw[j] = dn[j] * (1.0 - ng[j] * ng[j]);
        }

        let mut dr = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            dr[j] = dn_raw[j] * h_proj_n[j] * rg[j] * (1.0 - rg[j]);
        }

        let mut d_x_proj = vec![0.0f32; h3];
        for j in 0..hidden_size {
            d_x_proj[j] = dr[j];
            d_x_proj[hidden_size + j] = dz[j];
            d_x_proj[2 * hidden_size + j] = dn_raw[j];
        }

        let mut d_h_proj = vec![0.0f32; h3];
        for j in 0..hidden_size {
            d_h_proj[j] = dr[j];
            d_h_proj[hidden_size + j] = dz[j];
            d_h_proj[2 * hidden_size + j] = dn_raw[j] * rg[j];
        }

        for j in 0..h3 {
            grad_bih[j] += d_x_proj[j];
            grad_bhh[j] += d_h_proj[j];
        }

        for i in 0..input_size {
            for j in 0..h3 {
                grad_wih[i * h3 + j] += in_data[x_base + i] * d_x_proj[j];
            }
        }

        for i in 0..hidden_size {
            for j in 0..h3 {
                grad_whh[i * h3 + j] += h_prev[i] * d_h_proj[j];
            }
        }

        for i in 0..input_size {
            let mut s = 0.0f32;
            for j in 0..h3 {
                s += d_x_proj[j] * wih_data[i * h3 + j];
            }
            grad_input[x_base + i] = s;
        }

        dh_next = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            let mut s = 0.0f32;
            for j in 0..h3 {
                s += d_h_proj[j] * whh_data[i * h3 + j];
            }
            dh_next[i] = s + dh[i] * zg[i];
        }
    }

    if graph.nodes[input_id.0].requires_grad {
        graph.accumulate_grad(input_id, Tensor::from_vec(in_shape, grad_input)?)?;
    }
    if graph.nodes[w_ih_id.0].requires_grad {
        graph.accumulate_grad(w_ih_id, Tensor::from_vec(vec![input_size, h3], grad_wih)?)?;
    }
    if graph.nodes[w_hh_id.0].requires_grad {
        graph.accumulate_grad(w_hh_id, Tensor::from_vec(vec![hidden_size, h3], grad_whh)?)?;
    }
    if graph.nodes[bias_ih_id.0].requires_grad {
        graph.accumulate_grad(bias_ih_id, Tensor::from_vec(vec![h3], grad_bih)?)?;
    }
    if graph.nodes[bias_hh_id.0].requires_grad {
        graph.accumulate_grad(bias_hh_id, Tensor::from_vec(vec![h3], grad_bhh)?)?;
    }
    Ok(())
}
