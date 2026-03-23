use yscv_tensor::Tensor;

#[test]
fn rnn_cell_forward_produces_correct_shape() {
    let cell = crate::RnnCell::new(4, 8).unwrap();
    let x = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).unwrap();
    let h = Tensor::from_vec(vec![2, 8], vec![0.0; 16]).unwrap();
    let h_new = cell.forward(&x, &h).unwrap();
    assert_eq!(h_new.shape(), &[2, 8]);
}

#[test]
fn lstm_cell_forward_produces_correct_shapes() {
    let cell = crate::LstmCell::new(4, 8).unwrap();
    let x = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).unwrap();
    let h = Tensor::from_vec(vec![2, 8], vec![0.0; 16]).unwrap();
    let c = Tensor::from_vec(vec![2, 8], vec![0.0; 16]).unwrap();
    let (h_new, c_new) = cell.forward(&x, &h, &c).unwrap();
    assert_eq!(h_new.shape(), &[2, 8]);
    assert_eq!(c_new.shape(), &[2, 8]);
}

#[test]
fn gru_cell_forward_produces_correct_shape() {
    let cell = crate::GruCell::new(4, 8).unwrap();
    let x = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).unwrap();
    let h = Tensor::from_vec(vec![2, 8], vec![0.0; 16]).unwrap();
    let h_new = cell.forward(&x, &h).unwrap();
    assert_eq!(h_new.shape(), &[2, 8]);
}

#[test]
fn rnn_forward_sequence_produces_correct_output_shape() {
    let cell = crate::RnnCell::new(3, 5).unwrap();
    let input = Tensor::from_vec(vec![2, 4, 3], vec![0.1; 24]).unwrap();
    let (output, h_final) = crate::rnn_forward_sequence(&cell, &input, None).unwrap();
    assert_eq!(output.shape(), &[2, 4, 5]);
    assert_eq!(h_final.shape(), &[2, 5]);
}

#[test]
fn lstm_forward_sequence_produces_correct_output_shape() {
    let cell = crate::LstmCell::new(3, 5).unwrap();
    let input = Tensor::from_vec(vec![2, 4, 3], vec![0.1; 24]).unwrap();
    let (output, h_final, c_final) =
        crate::lstm_forward_sequence(&cell, &input, None, None).unwrap();
    assert_eq!(output.shape(), &[2, 4, 5]);
    assert_eq!(h_final.shape(), &[2, 5]);
    assert_eq!(c_final.shape(), &[2, 5]);
}

#[test]
fn gru_forward_sequence_produces_correct_output_shape() {
    let cell = crate::GruCell::new(3, 5).unwrap();
    let input = Tensor::from_vec(vec![2, 4, 3], vec![0.1; 24]).unwrap();
    let (output, h_final) = crate::gru_forward_sequence(&cell, &input, None).unwrap();
    assert_eq!(output.shape(), &[2, 4, 5]);
    assert_eq!(h_final.shape(), &[2, 5]);
}

#[test]
fn bilstm_produces_double_hidden_output() {
    let fwd = crate::LstmCell::new(3, 4).unwrap();
    let bwd = crate::LstmCell::new(3, 4).unwrap();
    let input = Tensor::from_vec(vec![1, 5, 3], vec![0.1; 15]).unwrap();
    let output = crate::bilstm_forward_sequence(&fwd, &bwd, &input).unwrap();
    assert_eq!(output.shape(), &[1, 5, 8]);
}
