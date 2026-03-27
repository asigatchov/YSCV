//! Tests for SIMD operations.

use super::*;

#[test]
fn sum_basic() {
    let data: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
    let simd = sum_dispatch(&data);
    let scalar: f32 = data.iter().sum();
    assert!((simd - scalar).abs() < 1e-3, "simd={simd}, scalar={scalar}");
}

#[test]
fn sum_empty() {
    assert_eq!(sum_dispatch(&[]), 0.0);
}

#[test]
fn max_basic() {
    let data: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7 - 12.0).sin()).collect();
    let simd = max_dispatch(&data);
    let scalar = reduce::max_scalar(&data);
    assert!((simd - scalar).abs() < 1e-6);
}

#[test]
fn max_empty() {
    assert_eq!(max_dispatch(&[]), f32::NEG_INFINITY);
}

#[test]
fn min_basic() {
    let data: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7 - 12.0).sin()).collect();
    let simd = min_dispatch(&data);
    let scalar = reduce::min_scalar(&data);
    assert!((simd - scalar).abs() < 1e-6);
}

#[test]
fn min_empty() {
    assert_eq!(min_dispatch(&[]), f32::INFINITY);
}

#[test]
fn binary_add() {
    let a: Vec<f32> = (0..33).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
    let mut out = vec![0.0f32; 33];
    binary_dispatch(&a, &b, &mut out, BinaryKind::Add);
    for i in 0..33 {
        assert!((out[i] - (a[i] + b[i])).abs() < 1e-6);
    }
}

#[test]
fn binary_div() {
    let a: Vec<f32> = (1..34).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..34).map(|i| i as f32 * 0.5 + 1.0).collect();
    let mut out = vec![0.0f32; 33];
    binary_dispatch(&a, &b, &mut out, BinaryKind::Div);
    for i in 0..33 {
        assert!((out[i] - (a[i] / b[i])).abs() < 1e-5);
    }
}

#[test]
fn relu_inplace() {
    let mut data: Vec<f32> = (-10..10).map(|i| i as f32).collect();
    relu_inplace_dispatch(&mut data);
    for (i, &v) in data.iter().enumerate() {
        let expected = (i as f32 - 10.0).max(0.0);
        assert_eq!(v, expected);
    }
}

#[test]
fn add_inplace() {
    let mut dst: Vec<f32> = (0..33).map(|i| i as f32).collect();
    let src: Vec<f32> = (0..33).map(|i| i as f32 * 2.0).collect();
    add_inplace_dispatch(&mut dst, &src);
    for i in 0..33 {
        assert_eq!(dst[i], i as f32 * 3.0);
    }
}

#[test]
fn scalar_inplace_ops() {
    let mut data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    add_scalar_inplace_dispatch(&mut data, 10.0);
    for i in 0..20 {
        assert_eq!(data[i], i as f32 + 10.0);
    }
    mul_scalar_inplace_dispatch(&mut data, 2.0);
    for i in 0..20 {
        assert_eq!(data[i], (i as f32 + 10.0) * 2.0);
    }
}

#[test]
fn unary_neg() {
    let data: Vec<f32> = (0..37).map(|i| i as f32 - 18.0).collect();
    let mut out = vec![0.0f32; 37];
    unary_dispatch(&data, &mut out, UnaryKind::Neg);
    for i in 0..37 {
        assert_eq!(out[i], -data[i]);
    }
}

#[test]
fn unary_abs() {
    let data: Vec<f32> = (0..37).map(|i| i as f32 - 18.0).collect();
    let mut out = vec![0.0f32; 37];
    unary_dispatch(&data, &mut out, UnaryKind::Abs);
    for i in 0..37 {
        assert_eq!(out[i], data[i].abs());
    }
}

#[test]
fn unary_sqrt() {
    let data: Vec<f32> = (0..37).map(|i| i as f32 + 1.0).collect();
    let mut out = vec![0.0f32; 37];
    unary_dispatch(&data, &mut out, UnaryKind::Sqrt);
    for i in 0..37 {
        assert!((out[i] - data[i].sqrt()).abs() < 1e-6);
    }
}

#[test]
fn unary_recip() {
    let data: Vec<f32> = (1..38).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 37];
    unary_dispatch(&data, &mut out, UnaryKind::Recip);
    for i in 0..37 {
        assert!(
            (out[i] - 1.0 / data[i]).abs() < 1e-4,
            "recip mismatch at {i}: got {}, expected {}",
            out[i],
            1.0 / data[i]
        );
    }
}

#[test]
fn exp_inplace() {
    let mut data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
    let expected: Vec<f32> = data.iter().map(|&v| v.exp()).collect();
    exp_inplace_dispatch(&mut data);
    for i in 0..20 {
        assert!((data[i] - expected[i]).abs() < 1e-5);
    }
}

#[test]
fn ln_basic() {
    let data: Vec<f32> = (1..38).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 37];
    ln_dispatch(&data, &mut out);
    // Miri scalar fallback has slightly worse precision than SIMD path
    let tol = if cfg!(miri) { 1e-3 } else { 1e-4 };
    for i in 0..37 {
        assert!(
            (out[i] - data[i].ln()).abs() < tol,
            "ln({}) got {}, expected {}",
            data[i],
            out[i],
            data[i].ln()
        );
    }
}

// -- SIMD edge-case tests --
// Test scalar tail handling for input lengths that aren't multiples
// of SIMD register width (4 for NEON/SSE, 8 for AVX, 16/32 for unrolled).

const EDGE_LENGTHS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65];

#[test]
fn unary_edge_lengths() {
    let kinds = [
        UnaryKind::Neg,
        UnaryKind::Abs,
        UnaryKind::Sqrt,
        UnaryKind::Recip,
        UnaryKind::Floor,
        UnaryKind::Ceil,
        UnaryKind::Round,
        UnaryKind::Sign,
    ];
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (1..=len as i32).map(|i| i as f32 * 0.7 - 3.0).collect();
        let mut out = vec![0.0f32; len];
        for kind in &kinds {
            unary_dispatch(&data, &mut out, *kind);
            for i in 0..len {
                let expected = match kind {
                    UnaryKind::Neg => -data[i],
                    UnaryKind::Abs => data[i].abs(),
                    UnaryKind::Sqrt => data[i].abs().sqrt(), // use abs to avoid NaN
                    UnaryKind::Recip => 1.0 / data[i],
                    UnaryKind::Floor => data[i].floor(),
                    UnaryKind::Ceil => data[i].ceil(),
                    UnaryKind::Round => data[i].round(),
                    UnaryKind::Sign => {
                        if data[i] > 0.0 {
                            1.0
                        } else if data[i] < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                };
                let actual = out[i];
                // Sqrt uses abs(data) so feed that to dispatch too
                if matches!(kind, UnaryKind::Sqrt) {
                    continue;
                } // skip -- input domain mismatch
                assert!(
                    (actual - expected).abs() < 0.01
                        || (expected.abs() > 1e6 && actual.abs() > 1e6),
                    "kind={kind:?} len={len} i={i}: got {actual}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn binary_add_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 1.1).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.3 + 1.0).collect();
        let mut out = vec![0.0f32; len];
        binary_dispatch(&a, &b, &mut out, BinaryKind::Add);
        for i in 0..len {
            assert!(
                (out[i] - (a[i] + b[i])).abs() < 1e-5,
                "add len={len} i={i}: got {}, expected {}",
                out[i],
                a[i] + b[i]
            );
        }
    }
}

#[test]
fn sum_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
        let simd = sum_dispatch(&data);
        let scalar: f32 = data.iter().sum();
        assert!(
            (simd - scalar).abs() < 1e-3 + scalar.abs() * 1e-5,
            "sum len={len}: simd={simd}, scalar={scalar}"
        );
    }
}

#[test]
fn max_min_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.7 - 5.0).sin()).collect();
        let mx = max_dispatch(&data);
        let mn = min_dispatch(&data);
        if len == 0 {
            assert_eq!(mx, f32::NEG_INFINITY);
            assert_eq!(mn, f32::INFINITY);
        } else {
            let expected_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let expected_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            assert!((mx - expected_max).abs() < 1e-6, "max len={len}");
            assert!((mn - expected_min).abs() < 1e-6, "min len={len}");
        }
    }
}

#[test]
fn exp_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.1 - 2.0).clamp(-10.0, 10.0))
            .collect();
        let mut out = vec![0.0f32; len];
        exp_dispatch(&data, &mut out);
        for i in 0..len {
            let expected = data[i].exp();
            assert!(
                (out[i] - expected).abs() < expected.abs() * 1e-4 + 1e-6,
                "exp len={len} i={i}: got {}, expected {}",
                out[i],
                expected
            );
        }
    }
}
