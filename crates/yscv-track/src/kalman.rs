//! Linear Kalman filter for bounding-box tracking.
//!
//! State vector: `[cx, cy, w, h, vx, vy, vw, vh]` — center position, size,
//! and their velocities.  The measurement is `[cx, cy, w, h]`.
//!
//! All matrices are stored as flat `[f32; N*N]` arrays for simplicity
//! (no external linear-algebra dependency).

use yscv_detect::BoundingBox;

/// Dimension of the state vector.
const STATE_DIM: usize = 8;
/// Dimension of the measurement vector.
const MEAS_DIM: usize = 4;

/// A 2-D Kalman filter for bounding-box tracking.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State estimate `[cx, cy, w, h, vx, vy, vw, vh]`.
    pub(crate) x: [f32; STATE_DIM],
    /// Error covariance matrix (8×8, row-major).
    pub(crate) p: [f32; STATE_DIM * STATE_DIM],
    /// Process noise covariance (8×8).
    q: [f32; STATE_DIM * STATE_DIM],
    /// Measurement noise covariance (4×4).
    r: [f32; MEAS_DIM * MEAS_DIM],
}

impl KalmanFilter {
    /// Create a Kalman filter initialized from a bounding box.
    pub fn new(bbox: BoundingBox) -> Self {
        let cx = (bbox.x1 + bbox.x2) * 0.5;
        let cy = (bbox.y1 + bbox.y2) * 0.5;
        let w = bbox.width();
        let h = bbox.height();

        let mut x = [0.0f32; STATE_DIM];
        x[0] = cx;
        x[1] = cy;
        x[2] = w;
        x[3] = h;
        // velocities start at zero

        // Initial covariance: large uncertainty on velocities.
        let mut p = [0.0f32; STATE_DIM * STATE_DIM];
        for i in 0..4 {
            p[i * STATE_DIM + i] = 10.0;
        }
        for i in 4..8 {
            p[i * STATE_DIM + i] = 100.0;
        }

        // Process noise
        let mut q = [0.0f32; STATE_DIM * STATE_DIM];
        for i in 0..4 {
            q[i * STATE_DIM + i] = 1.0;
        }
        for i in 4..8 {
            q[i * STATE_DIM + i] = 0.01;
        }

        // Measurement noise
        let mut r = [0.0f32; MEAS_DIM * MEAS_DIM];
        for i in 0..MEAS_DIM {
            r[i * MEAS_DIM + i] = 1.0;
        }

        Self { x, p, q, r }
    }

    /// Predict the next state (one time step dt=1).
    #[allow(clippy::needless_range_loop)]
    pub fn predict(&mut self) {
        // x' = F * x  where F is identity + velocity rows
        // x[0..4] += x[4..8]
        for i in 0..4 {
            self.x[i] += self.x[i + 4];
        }

        // P' = F * P * F^T + Q
        // F has 1s on diagonal plus 1s at (i, i+4) for i in 0..4
        let f = transition_matrix();
        let ft = transpose_8x8(&f);
        let fp = mat_mul_8x8(&f, &self.p);
        let fpft = mat_mul_8x8(&fp, &ft);
        for i in 0..STATE_DIM * STATE_DIM {
            self.p[i] = fpft[i] + self.q[i];
        }
    }

    /// Update the filter with a measurement `[cx, cy, w, h]`.
    pub fn update(&mut self, measurement: [f32; MEAS_DIM]) {
        // H is the 4×8 measurement matrix: identity for first 4 cols, zeros for rest.
        // Innovation: y = z - H * x
        let mut y = [0.0f32; MEAS_DIM];
        for i in 0..MEAS_DIM {
            y[i] = measurement[i] - self.x[i];
        }

        // S = H * P * H^T + R  (4×4)
        // Since H selects first 4 rows/cols of P:
        let mut s = [0.0f32; MEAS_DIM * MEAS_DIM];
        for i in 0..MEAS_DIM {
            for j in 0..MEAS_DIM {
                s[i * MEAS_DIM + j] = self.p[i * STATE_DIM + j] + self.r[i * MEAS_DIM + j];
            }
        }

        // K = P * H^T * S^{-1}  (8×4)
        // P * H^T is first 4 columns of P (8×4)
        let s_inv = invert_4x4(&s);
        let mut k = [0.0f32; STATE_DIM * MEAS_DIM];
        for i in 0..STATE_DIM {
            for j in 0..MEAS_DIM {
                let mut sum = 0.0f32;
                for m in 0..MEAS_DIM {
                    sum += self.p[i * STATE_DIM + m] * s_inv[m * MEAS_DIM + j];
                }
                k[i * MEAS_DIM + j] = sum;
            }
        }

        // x = x + K * y
        for i in 0..STATE_DIM {
            let mut sum = 0.0f32;
            for j in 0..MEAS_DIM {
                sum += k[i * MEAS_DIM + j] * y[j];
            }
            self.x[i] += sum;
        }

        // P = (I - K * H) * P
        // K * H is 8×8 where (K*H)[i][j] = K[i][j] for j < 4, 0 otherwise
        let mut kh = [0.0f32; STATE_DIM * STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..MEAS_DIM {
                kh[i * STATE_DIM + j] = k[i * MEAS_DIM + j];
            }
        }
        // I - K*H
        let mut i_kh = [0.0f32; STATE_DIM * STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                i_kh[i * STATE_DIM + j] = if i == j { 1.0 } else { 0.0 } - kh[i * STATE_DIM + j];
            }
        }
        let new_p = mat_mul_8x8(&i_kh, &self.p);
        self.p = new_p;
    }

    /// Get current state as bounding box.
    pub fn bbox(&self) -> BoundingBox {
        let cx = self.x[0];
        let cy = self.x[1];
        let w = self.x[2].max(1e-3);
        let h = self.x[3].max(1e-3);
        BoundingBox {
            x1: cx - w * 0.5,
            y1: cy - h * 0.5,
            x2: cx + w * 0.5,
            y2: cy + h * 0.5,
        }
    }

    /// Get predicted bbox without mutating state.
    pub fn predicted_bbox(&self) -> BoundingBox {
        let cx = self.x[0] + self.x[4];
        let cy = self.x[1] + self.x[5];
        let w = (self.x[2] + self.x[6]).max(1e-3);
        let h = (self.x[3] + self.x[7]).max(1e-3);
        BoundingBox {
            x1: cx - w * 0.5,
            y1: cy - h * 0.5,
            x2: cx + w * 0.5,
            y2: cy + h * 0.5,
        }
    }
}

// ── Small matrix helpers (8×8 and 4×4) ─────────────────────────────

fn transition_matrix() -> [f32; STATE_DIM * STATE_DIM] {
    let mut f = [0.0f32; STATE_DIM * STATE_DIM];
    // Identity
    for i in 0..STATE_DIM {
        f[i * STATE_DIM + i] = 1.0;
    }
    // Position += velocity (dt=1)
    for i in 0..4 {
        f[i * STATE_DIM + i + 4] = 1.0;
    }
    f
}

fn transpose_8x8(a: &[f32; STATE_DIM * STATE_DIM]) -> [f32; STATE_DIM * STATE_DIM] {
    let mut out = [0.0f32; STATE_DIM * STATE_DIM];
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            out[j * STATE_DIM + i] = a[i * STATE_DIM + j];
        }
    }
    out
}

fn mat_mul_8x8(
    a: &[f32; STATE_DIM * STATE_DIM],
    b: &[f32; STATE_DIM * STATE_DIM],
) -> [f32; STATE_DIM * STATE_DIM] {
    let mut out = [0.0f32; STATE_DIM * STATE_DIM];
    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            let mut sum = 0.0f32;
            for k in 0..STATE_DIM {
                sum += a[i * STATE_DIM + k] * b[k * STATE_DIM + j];
            }
            out[i * STATE_DIM + j] = sum;
        }
    }
    out
}

/// Invert a 4×4 matrix using the adjugate method.
#[allow(clippy::needless_range_loop)]
fn invert_4x4(m: &[f32; MEAS_DIM * MEAS_DIM]) -> [f32; MEAS_DIM * MEAS_DIM] {
    let n = MEAS_DIM;
    // Gauss-Jordan elimination on augmented matrix
    let mut aug = [[0.0f32; 8]; 4];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i * n + j];
        }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in col + 1..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            // Singular — return identity as fallback
            let mut result = [0.0f32; MEAS_DIM * MEAS_DIM];
            for i in 0..n {
                result[i * n + i] = 1.0;
            }
            return result;
        }

        let inv_pivot = 1.0 / pivot;
        for j in 0..2 * n {
            aug[col][j] *= inv_pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..2 * n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut result = [0.0f32; MEAS_DIM * MEAS_DIM];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = aug[i][n + j];
        }
    }
    result
}
