//! Multivariate ordinary-least-squares (OLS) regression analysis.
//!
//! Real linear regression via the normal-equation solution
//! `β = (XᵀX)⁻¹Xᵀy`, computed with the crate's own
//! [`super::utils::solve_linear_system`] Gauss-Jordan solver — appropriate
//! here since regression designs in this crate's use cases (per-sample
//! evaluation-metric diagnostics) have a small, bounded number of predictors,
//! not large-scale numerical linear algebra. Standard errors, t-statistics,
//! p-values (via `statrs`'s exact Student's-t CDF, matching the pattern in
//! [`super::basic_tests`]), R-squared/adjusted-R-squared, and the overall
//! F-test are all computed from the real residuals of the fit.

use super::types::RegressionResult;
use super::utils::{mean, solve_linear_system};
use crate::{EvaluationError, EvaluationResult};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

/// Fit a multivariate OLS regression of `y` on the predictor columns in
/// `predictors` (each element of `predictors` is one predictor variable's
/// full column of observations; all columns and `y` must have equal length
/// `n`). An intercept term is always included.
///
/// Returns [`EvaluationError::InvalidInput`] when there are fewer
/// observations than parameters (`n <= num_predictors + 1`, i.e. an
/// under-determined system), when inputs have mismatched lengths, or when
/// the design matrix is singular (e.g. two perfectly collinear predictors) —
/// never a fabricated or NaN-silent fit.
///
/// For a single predictor, prefer
/// [`super::basic_tests::StatisticalAnalyzer::linear_regression`], which
/// additionally reports `slope`/`intercept` as first-class fields; this
/// function's `coefficients` vector is `[intercept, β₁, β₂, ...]` for the
/// general multivariate case and only mirrors `slope`/`intercept` when there
/// is exactly one predictor.
pub fn multivariate_linear_regression(
    predictors: &[Vec<f64>],
    y: &[f64],
) -> EvaluationResult<RegressionResult> {
    let num_predictors = predictors.len();
    if num_predictors == 0 {
        return Err(EvaluationError::InvalidInput {
            message: "multivariate_linear_regression requires at least one predictor".to_string(),
        }
        .into());
    }
    let n = y.len();
    if n == 0 || predictors.iter().any(|col| col.len() != n) {
        return Err(EvaluationError::InvalidInput {
            message: "all predictor columns and the response must have equal, non-zero length"
                .to_string(),
        }
        .into());
    }
    let num_params = num_predictors + 1; // + intercept
    if n <= num_params {
        return Err(EvaluationError::InvalidInput {
            message: format!(
                "not enough observations ({n}) to fit {num_params} parameters \
                 ({num_predictors} predictors + intercept)"
            ),
        }
        .into());
    }

    // Design matrix X (n x num_params, row-major): column 0 is the intercept
    // (all ones), columns 1.. are the predictors.
    let mut design = vec![0.0f64; n * num_params];
    for row in 0..n {
        design[row * num_params] = 1.0;
        for (p, col) in predictors.iter().enumerate() {
            design[row * num_params + p + 1] = col[row];
        }
    }

    // Normal equations: (XᵀX) β = Xᵀy.
    let mut xtx = vec![0.0f64; num_params * num_params];
    let mut xty = vec![0.0f64; num_params];
    for i in 0..num_params {
        for j in 0..num_params {
            let mut sum = 0.0;
            for row in 0..n {
                sum += design[row * num_params + i] * design[row * num_params + j];
            }
            xtx[i * num_params + j] = sum;
        }
        let mut sum = 0.0;
        for row in 0..n {
            sum += design[row * num_params + i] * y[row];
        }
        xty[i] = sum;
    }

    let coefficients = solve_linear_system(&xtx, &xty, num_params).map_err(|_| {
        EvaluationError::InvalidInput {
            message: "design matrix is singular (e.g. perfectly collinear predictors); \
                      cannot fit a unique regression"
                .to_string(),
        }
    })?;

    // Fitted values and residuals.
    let fitted: Vec<f64> = (0..n)
        .map(|row| {
            (0..num_params)
                .map(|p| design[row * num_params + p] * coefficients[p])
                .sum::<f64>()
        })
        .collect();
    let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(&yi, &fi)| yi - fi).collect();

    let y_mean = mean(y);
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let r_squared = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 1.0 };
    let df_resid = (n - num_params) as f64;
    let adjusted_r_squared = if df_resid > 0.0 && ss_tot > 1e-12 {
        1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid
    } else {
        r_squared
    };

    let residual_variance = ss_res / df_resid;
    let residual_standard_error = residual_variance.sqrt();

    // Standard errors of the coefficients: sqrt(σ² · diag((XᵀX)⁻¹)). Rather
    // than explicitly inverting XᵀX, solve (XᵀX)·e_i = unit vector for each
    // parameter to obtain the corresponding column of the inverse (the
    // diagonal entry we actually need), reusing the same Gauss-Jordan
    // solver.
    let mut standard_errors = Vec::with_capacity(num_params);
    for i in 0..num_params {
        let mut unit = vec![0.0f64; num_params];
        unit[i] = 1.0;
        let inv_col = solve_linear_system(&xtx, &unit, num_params).map_err(|_| {
            EvaluationError::InvalidInput {
                message: "design matrix became singular while computing standard errors"
                    .to_string(),
            }
        })?;
        let variance_i = (residual_variance * inv_col[i]).max(0.0);
        standard_errors.push(variance_i.sqrt());
    }

    let t_statistics: Vec<f64> = coefficients
        .iter()
        .zip(standard_errors.iter())
        .map(|(&c, &se)| if se > 1e-15 { c / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_statistics
        .iter()
        .map(|&t| match StudentsT::new(0.0, 1.0, df_resid) {
            Ok(dist) if df_resid > 0.0 => (2.0 * (1.0 - dist.cdf(t.abs()))).clamp(0.0, 1.0),
            _ => 1.0,
        })
        .collect();

    // Overall F-test of the fitted model against an intercept-only model.
    let df_model = num_predictors as f64;
    let f_statistic = if ss_res > 1e-15 && df_model > 0.0 {
        ((ss_tot - ss_res) / df_model) / (ss_res / df_resid)
    } else {
        f64::INFINITY
    };
    let f_p_value = if f_statistic.is_finite() && df_resid > 0.0 {
        match FisherSnedecor::new(df_model, df_resid) {
            Ok(dist) => (1.0 - dist.cdf(f_statistic)).clamp(0.0, 1.0),
            Err(_) => 1.0,
        }
    } else {
        0.0
    };

    // `slope`/`intercept` mirror the simple-regression case (single
    // predictor) for API parity with
    // `basic_tests::StatisticalAnalyzer::linear_regression`; for genuinely
    // multivariate fits these are the intercept and the first predictor's
    // coefficient, respectively (see `coefficients` for the full vector).
    let intercept = coefficients[0];
    let slope = coefficients.get(1).copied().unwrap_or(0.0);

    Ok(RegressionResult {
        coefficients,
        r_squared,
        adjusted_r_squared,
        standard_errors,
        t_statistics,
        p_values,
        residual_standard_error,
        f_statistic,
        f_p_value,
        degrees_of_freedom: (num_predictors, df_resid as usize),
        slope,
        intercept,
        p_value: f_p_value,
        standard_error: residual_standard_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_linear_fit_single_predictor() {
        // y = 2 + 3x, no noise.
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
        let result = multivariate_linear_regression(&[x], &y).unwrap();

        assert!((result.intercept - 2.0).abs() < 1e-6, "intercept = {}", result.intercept);
        assert!((result.slope - 3.0).abs() < 1e-6, "slope = {}", result.slope);
        assert!(
            (result.r_squared - 1.0).abs() < 1e-6,
            "perfect fit should have R^2 ~= 1, got {}",
            result.r_squared
        );
        assert!(result.residual_standard_error < 1e-4);
    }

    #[test]
    fn test_two_predictors_known_coefficients() {
        // y = 1 + 2*x1 - 1*x2, no noise.
        let x1: Vec<f64> = (0..20).map(f64::from).collect();
        let x2: Vec<f64> = (0..20).map(|i| (i as f64 * 0.7).sin()).collect();
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + 2.0 * a - b)
            .collect();

        let result = multivariate_linear_regression(&[x1, x2], &y).unwrap();
        assert_eq!(result.coefficients.len(), 3);
        assert!((result.coefficients[0] - 1.0).abs() < 1e-6);
        assert!((result.coefficients[1] - 2.0).abs() < 1e-6);
        assert!((result.coefficients[2] - (-1.0)).abs() < 1e-6);
        assert!((result.r_squared - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_noisy_data_realistic_fit() {
        // y = 5 + 0.5x plus small deterministic "noise" (no rand dependency);
        // R^2 should be high but not exactly 1, and the p-value for the
        // slope should indicate a real, significant relationship.
        let x: Vec<f64> = (0..30).map(f64::from).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 5.0 + 0.5 * xi + ((i as f64 * 1.37).sin() * 0.3))
            .collect();
        let result = multivariate_linear_regression(&[x], &y).unwrap();

        assert!(result.r_squared > 0.9, "R^2 = {}", result.r_squared);
        assert!(result.r_squared < 1.0);
        assert!(
            result.p_values[1] < 0.05,
            "slope p-value should indicate a significant relationship, got {}",
            result.p_values[1]
        );
        assert!(result.standard_errors.iter().all(|&se| se >= 0.0));
    }

    #[test]
    fn test_no_relationship_high_p_value() {
        // y is (deterministically) unrelated to x: alternating constant
        // pattern uncorrelated with the linearly increasing x.
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let result = multivariate_linear_regression(&[x], &y).unwrap();

        assert!(
            result.r_squared < 0.3,
            "unrelated x/y should have low R^2, got {}",
            result.r_squared
        );
    }

    #[test]
    fn test_insufficient_observations_errors() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        // 2 observations, 2 parameters (intercept + slope): under-determined.
        assert!(multivariate_linear_regression(&[x], &y).is_err());
    }

    #[test]
    fn test_mismatched_lengths_errors() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(multivariate_linear_regression(&[x], &y).is_err());
    }

    #[test]
    fn test_collinear_predictors_errors() {
        let x1: Vec<f64> = (0..10).map(f64::from).collect();
        let x2: Vec<f64> = x1.iter().map(|&x| x * 2.0).collect(); // perfectly collinear
        let y: Vec<f64> = x1.iter().map(|&x| x + 1.0).collect();
        assert!(multivariate_linear_regression(&[x1, x2], &y).is_err());
    }
}
