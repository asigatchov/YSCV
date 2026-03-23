use super::error::TensorError;
use super::tensor::Tensor;

impl Tensor {
    /// Sum of diagonal elements of a 2D square matrix.
    pub fn trace(&self) -> Result<f32, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("trace requires a 2D tensor, got rank {}", self.rank()),
            });
        }
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        if rows != cols {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: vec![rows, rows],
            });
        }
        let data = self.data();
        let mut sum = 0.0f32;
        for i in 0..rows {
            sum += data[i * cols + i];
        }
        Ok(sum)
    }

    /// Dot product of two 1D tensors.
    pub fn dot(&self, rhs: &Self) -> Result<f32, TensorError> {
        if self.rank() != 1 || rhs.rank() != 1 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!(
                    "dot requires two 1D tensors, got ranks {} and {}",
                    self.rank(),
                    rhs.rank()
                ),
            });
        }
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: rhs.shape().to_vec(),
            });
        }
        let a = self.data();
        let b = rhs.data();
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        Ok(sum)
    }

    /// Cross product of two 3-element 1D tensors.
    pub fn cross(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.rank() != 1 || rhs.rank() != 1 {
            return Err(TensorError::UnsupportedOperation {
                msg: "cross requires two 1D tensors".into(),
            });
        }
        if self.shape()[0] != 3 || rhs.shape()[0] != 3 {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: rhs.shape().to_vec(),
            });
        }
        let a = self.data();
        let b = rhs.data();
        let result = vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        Tensor::from_vec(vec![3], result)
    }

    /// Lp norm of all elements. p=1 for L1, p=2 for L2.
    pub fn norm(&self, p: f32) -> f32 {
        let data = self.data();
        if p == 1.0 {
            data.iter().map(|x| x.abs()).sum()
        } else if p == 2.0 {
            data.iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            data.iter()
                .map(|x| x.abs().powf(p))
                .sum::<f32>()
                .powf(1.0 / p)
        }
    }

    /// Determinant of a square matrix (LU-based).
    pub fn det(&self) -> Result<f32, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("det requires a 2D tensor, got rank {}", self.rank()),
            });
        }
        let n = self.shape()[0];
        if n != self.shape()[1] {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: vec![n, n],
            });
        }

        // Copy data into a working matrix
        let mut a: Vec<f32> = self.data().to_vec();
        let mut sign = 1.0f32;

        for col in 0..n {
            // Partial pivoting: find max in column
            let mut max_row = col;
            let mut max_val = a[col * n + col].abs();
            for row in (col + 1)..n {
                let v = a[row * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return Ok(0.0);
            }
            if max_row != col {
                // Swap rows
                for j in 0..n {
                    a.swap(col * n + j, max_row * n + j);
                }
                sign = -sign;
            }
            let pivot = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                for j in col..n {
                    let val = a[col * n + j];
                    a[row * n + j] -= factor * val;
                }
            }
        }

        let mut det = sign;
        for i in 0..n {
            det *= a[i * n + i];
        }
        Ok(det)
    }

    /// Inverse of a square matrix (Gauss-Jordan elimination).
    pub fn inv(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("inv requires a 2D tensor, got rank {}", self.rank()),
            });
        }
        let n = self.shape()[0];
        if n != self.shape()[1] {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: vec![n, n],
            });
        }

        // Augmented matrix [A | I], stored as n x 2n
        let data = self.data();
        let nn = 2 * n;
        let mut aug = vec![0.0f32; n * nn];
        for i in 0..n {
            for j in 0..n {
                aug[i * nn + j] = data[i * n + j];
            }
            aug[i * nn + n + i] = 1.0;
        }

        // Gauss-Jordan with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col * nn + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * nn + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return Err(TensorError::UnsupportedOperation {
                    msg: "matrix is singular".into(),
                });
            }
            if max_row != col {
                for j in 0..nn {
                    aug.swap(col * nn + j, max_row * nn + j);
                }
            }

            // Scale pivot row
            let pivot = aug[col * nn + col];
            for j in 0..nn {
                aug[col * nn + j] /= pivot;
            }

            // Eliminate column in all other rows
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row * nn + col];
                for j in 0..nn {
                    let val = aug[col * nn + j];
                    aug[row * nn + j] -= factor * val;
                }
            }
        }

        // Extract the right half
        let mut result = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                result[i * n + j] = aug[i * nn + n + j];
            }
        }
        Tensor::from_vec(vec![n, n], result)
    }

    /// Solve linear system Ax = b. self is A, rhs is b.
    pub fn solve(&self, b: &Self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("solve requires A to be 2D, got rank {}", self.rank()),
            });
        }
        let n = self.shape()[0];
        if n != self.shape()[1] {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: vec![n, n],
            });
        }
        if b.rank() != 1 || b.shape()[0] != n {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: b.shape().to_vec(),
            });
        }

        // LU decomposition with partial pivoting
        let data = self.data();
        let mut a = data.to_vec();
        let mut perm: Vec<usize> = (0..n).collect();

        for col in 0..n {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = a[col * n + col].abs();
            for row in (col + 1)..n {
                let v = a[row * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return Err(TensorError::UnsupportedOperation {
                    msg: "matrix is singular".into(),
                });
            }
            if max_row != col {
                for j in 0..n {
                    a.swap(col * n + j, max_row * n + j);
                }
                perm.swap(col, max_row);
            }
            let pivot = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                a[row * n + col] = factor; // store L factor
                for j in (col + 1)..n {
                    let val = a[col * n + j];
                    a[row * n + j] -= factor * val;
                }
            }
        }

        // Apply permutation to b
        let bd = b.data();
        let mut pb = vec![0.0f32; n];
        for i in 0..n {
            pb[i] = bd[perm[i]];
        }

        // Forward substitution (Ly = Pb)
        let mut y = pb;
        for i in 1..n {
            for j in 0..i {
                let l_ij = a[i * n + j];
                y[i] -= l_ij * y[j];
            }
        }

        // Back substitution (Ux = y)
        let mut x = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let u_ij = a[i * n + j];
                x[i] -= u_ij * x[j];
            }
            x[i] /= a[i * n + i];
        }

        Tensor::from_vec(vec![n], x)
    }

    /// QR decomposition via Householder reflections. Returns (Q, R).
    pub fn qr(&self) -> Result<(Self, Self), TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("qr requires a 2D tensor, got rank {}", self.rank()),
            });
        }
        let m = self.shape()[0];
        let n = self.shape()[1];
        let data = self.data();

        // Working copy of the matrix (will become R)
        let mut r = data.to_vec();

        // Q starts as identity
        let mut q = vec![0.0f32; m * m];
        for i in 0..m {
            q[i * m + i] = 1.0;
        }

        let k = m.min(n);
        for j in 0..k {
            // Extract column j from row j..m
            let mut col = vec![0.0f32; m - j];
            for i in j..m {
                col[i - j] = r[i * n + j];
            }

            // Compute the Householder vector
            let norm_col: f32 = col.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_col < 1e-12 {
                continue;
            }
            let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
            col[0] += sign * norm_col;

            let norm_v: f32 = col.iter().map(|x| x * x).sum::<f32>();
            if norm_v < 1e-24 {
                continue;
            }

            // Apply Householder reflection to R: R = R - 2 * v * (v^T * R) / (v^T * v)
            // Only rows j..m, cols j..n
            for jj in j..n {
                let mut dot = 0.0f32;
                for i in j..m {
                    dot += col[i - j] * r[i * n + jj];
                }
                let factor = 2.0 * dot / norm_v;
                for i in j..m {
                    r[i * n + jj] -= factor * col[i - j];
                }
            }

            // Apply Householder reflection to Q: Q = Q - 2 * Q * v * v^T / (v^T * v)
            // Q is m x m, we update all rows of Q, cols j..m
            for i in 0..m {
                let mut dot = 0.0f32;
                for jj in j..m {
                    dot += q[i * m + jj] * col[jj - j];
                }
                let factor = 2.0 * dot / norm_v;
                for jj in j..m {
                    q[i * m + jj] -= factor * col[jj - j];
                }
            }
        }

        let q_tensor = Tensor::from_vec(vec![m, m], q)?;
        let r_tensor = Tensor::from_vec(vec![m, n], r)?;
        Ok((q_tensor, r_tensor))
    }

    /// Cholesky decomposition of a symmetric positive-definite matrix. Returns lower triangular L.
    pub fn cholesky(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("cholesky requires a 2D tensor, got rank {}", self.rank()),
            });
        }
        let n = self.shape()[0];
        if n != self.shape()[1] {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: vec![n, n],
            });
        }
        let data = self.data();
        let mut l = vec![0.0f32; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f32;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let diag = data[i * n + i] - sum;
                    if diag <= 0.0 {
                        return Err(TensorError::UnsupportedOperation {
                            msg: "matrix is not positive definite".into(),
                        });
                    }
                    l[i * n + j] = diag.sqrt();
                } else {
                    l[i * n + j] = (data[i * n + j] - sum) / l[j * n + j];
                }
            }
        }

        Tensor::from_vec(vec![n, n], l)
    }
}
