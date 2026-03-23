use yscv_tensor::Tensor;

use super::ImgProcError;

pub(crate) fn hwc_shape(input: &Tensor) -> Result<(usize, usize, usize), ImgProcError> {
    if input.rank() != 3 {
        return Err(ImgProcError::InvalidImageShape {
            expected_rank: 3,
            got: input.shape().to_vec(),
        });
    }
    Ok((input.shape()[0], input.shape()[1], input.shape()[2]))
}
