//! Hungarian (Kuhn-Munkres) algorithm for optimal assignment.
//!
//! Solves the linear assignment problem on an NxM cost matrix in O(n^3) time.

/// Solve the assignment problem: find the optimal assignment that minimizes total cost.
///
/// `cost_matrix` is NxM where `cost_matrix[i][j]` is the cost of assigning row i to column j.
/// Returns a list of `(row, col)` pairs representing the optimal assignment.
/// The number of pairs equals `min(N, M)`.
pub fn hungarian_assignment(cost_matrix: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n_rows = cost_matrix.len();
    if n_rows == 0 {
        return Vec::new();
    }
    let n_cols = cost_matrix[0].len();
    if n_cols == 0 {
        return Vec::new();
    }

    // We solve using the classic O(n^3) Hungarian algorithm with potentials.
    // We pad the matrix to be square with a large value.
    let n = n_rows.max(n_cols);
    let big = 1e9_f32;

    // Build square cost matrix (1-indexed internally for algorithm convenience).
    let mut cost = vec![vec![0.0_f32; n + 1]; n + 1];
    for i in 0..n {
        for j in 0..n {
            if i < n_rows && j < n_cols {
                cost[i + 1][j + 1] = cost_matrix[i][j];
            } else {
                cost[i + 1][j + 1] = big;
            }
        }
    }

    // u[i] = potential for row i, v[j] = potential for column j
    let mut u = vec![0.0_f32; n + 1];
    let mut v = vec![0.0_f32; n + 1];
    // p[j] = row assigned to column j (0 = unassigned)
    let mut p = vec![0_usize; n + 1];
    // way[j] = column preceding j in the augmenting path
    let mut way = vec![0_usize; n + 1];

    let mut min_v = vec![f32::INFINITY; n + 1];
    let mut used = vec![false; n + 1];
    for i in 1..=n {
        // Try to assign row i.
        p[0] = i;
        let mut j0 = 0_usize; // virtual column
        min_v.fill(f32::INFINITY);
        used.fill(false);

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f32::INFINITY;
            let mut j1 = 0_usize;

            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost[i0][j] - u[i0] - v[j];
                if cur < min_v[j] {
                    min_v[j] = cur;
                    way[j] = j0;
                }
                if min_v[j] < delta {
                    delta = min_v[j];
                    j1 = j;
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        // Update assignment along the augmenting path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Extract results: p[j] = row assigned to column j (1-indexed).
    let mut result = Vec::new();
    for j in 1..=n {
        let row = p[j] - 1;
        let col = j - 1;
        if row < n_rows && col < n_cols {
            result.push((row, col));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hungarian_2x2() {
        // Cost matrix:
        // [1, 2]
        // [3, 0]
        // Optimal: (0,0)=1, (1,1)=0 → total 1
        let cost = vec![vec![1.0, 2.0], vec![3.0, 0.0]];
        let result = hungarian_assignment(&cost);
        assert_eq!(result.len(), 2);
        let total: f32 = result.iter().map(|&(r, c)| cost[r][c]).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_3x3() {
        // Cost matrix:
        // [10, 5, 13]
        // [ 3, 7,  2]
        // [ 6, 8, 12]
        // Optimal: (0,1)=5, (1,2)=2, (2,0)=6 → total 13
        let cost = vec![
            vec![10.0, 5.0, 13.0],
            vec![3.0, 7.0, 2.0],
            vec![6.0, 8.0, 12.0],
        ];
        let result = hungarian_assignment(&cost);
        assert_eq!(result.len(), 3);
        let total: f32 = result.iter().map(|&(r, c)| cost[r][c]).sum();
        assert!((total - 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_non_square() {
        // 2x3 matrix:
        // [1, 2, 0]
        // [3, 4, 1]
        // Optimal: (0,0)=1, (1,2)=1 → total 2
        let cost = vec![vec![1.0, 2.0, 0.0], vec![3.0, 4.0, 1.0]];
        let result = hungarian_assignment(&cost);
        assert_eq!(result.len(), 2);
        let total: f32 = result.iter().map(|&(r, c)| cost[r][c]).sum();
        assert!((total - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_identity_cost() {
        // Identity matrix → optimal is to pick the zeros (off-diagonal).
        // But identity has 1s on diagonal, 0s elsewhere.
        // Actually the identity matrix cost means diagonal costs 1, off-diagonal 0.
        // Optimal: assign each row to any column != row → total 0.
        // For 3x3: possible (0,1),(1,0),(2,2) has cost 0+0+1=1 ...
        // Actually (0,1),(1,2),(2,0) → 0+0+0 = 0
        let cost = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = hungarian_assignment(&cost);
        assert_eq!(result.len(), 3);
        let total: f32 = result.iter().map(|&(r, c)| cost[r][c]).sum();
        assert!((total - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_empty() {
        let cost: Vec<Vec<f32>> = vec![];
        let result = hungarian_assignment(&cost);
        assert!(result.is_empty());
    }
}
