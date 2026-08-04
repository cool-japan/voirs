//! Shared numeric utilities for the `statistical` module.
//!
//! These are small, dependency-free helpers (mean/variance/standard
//! deviation, a general-purpose Gauss-Jordan linear-system solver used by
//! [`super::regression`]'s multivariate OLS) reused by several of the
//! `statistical` submodules rather than duplicated in each one.

use crate::{EvaluationError, EvaluationResult};

/// Arithmetic mean of `data`. Returns `0.0` for empty input.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Sample variance of `data` (denominator `n - 1`). Returns `0.0` for fewer
/// than 2 points.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Sample standard deviation of `data` (denominator `n - 1`).
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Pearson correlation coefficient between `x` and `y` (both `f64`).
///
/// Returns an error when the inputs have mismatched or insufficient (`< 2`)
/// length, or when either variable has zero variance (undefined
/// correlation).
pub fn pearson_correlation_f64(x: &[f64], y: &[f64]) -> EvaluationResult<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(EvaluationError::InvalidInput {
            message: "Pearson correlation requires equal-length inputs of at least 2".to_string(),
        }
        .into());
    }
    let mean_x = mean(x);
    let mean_y = mean(y);
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 1e-15 || var_y <= 1e-15 {
        return Err(EvaluationError::InvalidInput {
            message: "Cannot compute correlation: zero variance in input".to_string(),
        }
        .into());
    }
    Ok(cov / (var_x.sqrt() * var_y.sqrt()))
}

/// Solve the dense linear system `a · x = b` via Gauss-Jordan elimination
/// with partial pivoting, where `a` is a square `n x n` matrix stored
/// row-major (`a[i * n + j]` is row `i`, column `j`) and `b` has length `n`.
///
/// Used by [`super::regression`] for small-dimensional ordinary-least-squares
/// normal-equation solves (a handful of predictors), where a general-purpose
/// dense linear-algebra dependency would be disproportionate. Returns an
/// error if the matrix is singular (or near-singular, pivot magnitude below
/// `1e-12`) rather than silently returning a garbage/NaN solution.
pub fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> EvaluationResult<Vec<f64>> {
    if a.len() != n * n || b.len() != n {
        return Err(EvaluationError::InvalidInput {
            message: format!(
                "solve_linear_system: expected a {n}x{n} matrix ({} entries) and a length-{n} \
                 vector, got {} matrix entries and {} vector entries",
                n * n,
                a.len(),
                b.len()
            ),
        }
        .into());
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Augmented matrix [A | b], row-major with n+1 columns.
    let mut aug = vec![0.0f64; n * (n + 1)];
    for row in 0..n {
        aug[row * (n + 1)..row * (n + 1) + n].copy_from_slice(&a[row * n..row * n + n]);
        aug[row * (n + 1) + n] = b[row];
    }

    for pivot_col in 0..n {
        // Partial pivoting: swap in the row with the largest magnitude entry
        // in this column, for numerical stability.
        let mut best_row = pivot_col;
        let mut best_val = aug[pivot_col * (n + 1) + pivot_col].abs();
        for row in (pivot_col + 1)..n {
            let val = aug[row * (n + 1) + pivot_col].abs();
            if val > best_val {
                best_val = val;
                best_row = row;
            }
        }
        if best_val < 1e-12 {
            return Err(EvaluationError::InvalidInput {
                message: "solve_linear_system: matrix is singular or near-singular".to_string(),
            }
            .into());
        }
        if best_row != pivot_col {
            for col in 0..n + 1 {
                aug.swap(pivot_col * (n + 1) + col, best_row * (n + 1) + col);
            }
        }

        let pivot_val = aug[pivot_col * (n + 1) + pivot_col];
        for col in 0..n + 1 {
            aug[pivot_col * (n + 1) + col] /= pivot_val;
        }

        for row in 0..n {
            if row == pivot_col {
                continue;
            }
            let factor = aug[row * (n + 1) + pivot_col];
            if factor == 0.0 {
                continue;
            }
            for col in 0..n + 1 {
                aug[row * (n + 1) + col] -= factor * aug[pivot_col * (n + 1) + col];
            }
        }
    }

    Ok((0..n).map(|row| aug[row * (n + 1) + n]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_and_variance() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(mean(&[2.0, 4.0, 6.0]), 4.0);
        assert_eq!(variance(&[5.0]), 0.0);
        // variance of [2, 4, 6]: mean 4, deviations -2, 0, 2 -> sq sum 8, / (3-1) = 4
        assert!((variance(&[2.0, 4.0, 6.0]) - 4.0).abs() < 1e-9);
        assert!((std_dev(&[2.0, 4.0, 6.0]) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation_f64(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation_f64(&x, &y).unwrap();
        assert!((r - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_correlation_zero_variance_errors() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(pearson_correlation_f64(&x, &y).is_err());
    }

    #[test]
    fn test_solve_linear_system_identity() {
        // Identity matrix: solution should equal b exactly.
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0];
        let x = solve_linear_system(&a, &b, 2).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-9);
        assert!((x[1] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_solve_linear_system_known_solution() {
        // 2x + y = 5
        // x + 3y = 10
        // Solution: x = 1, y = 3
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let x = solve_linear_system(&a, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6, "x = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-6, "y = {}", x[1]);
    }

    #[test]
    fn test_solve_linear_system_singular_errors() {
        // Singular matrix (second row is a multiple of the first).
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let b = vec![3.0, 6.0];
        assert!(solve_linear_system(&a, &b, 2).is_err());
    }

    #[test]
    fn test_solve_linear_system_requires_pivoting() {
        // Without partial pivoting, a naive elimination would divide by the
        // (0,0) zero entry here.
        let a = vec![0.0, 1.0, 1.0, 1.0];
        let b = vec![2.0, 3.0];
        // y = 2 (from row 0: 0x + y = 2), x + y = 3 -> x = 1
        let x = solve_linear_system(&a, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 2.0).abs() < 1e-9);
    }
}
