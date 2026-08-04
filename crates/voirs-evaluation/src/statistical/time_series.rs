//! Time series analysis: autocorrelation (ACF), partial autocorrelation
//! (PACF), linear trend detection, and moving averages.
//!
//! All functions operate on a plain `&[f64]` time series (e.g. a metric's
//! score history across repeated evaluation runs) and compute genuine
//! statistics from the series' own values — no fabricated or seeded-random
//! output.

use super::utils::mean;
use crate::{EvaluationError, EvaluationResult};

/// Result of [`analyze_trend`]: a real least-squares linear trend fit of
/// `series` against its own index `0..series.len()`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrendAnalysis {
    /// Least-squares slope (change per time step)
    pub slope: f64,
    /// Least-squares intercept
    pub intercept: f64,
    /// Coefficient of determination of the linear fit
    pub r_squared: f64,
    /// Whether the trend is judged significant (see [`analyze_trend`] docs)
    pub is_significant: bool,
}

/// Sample autocorrelation function (ACF) of `series` at lags `1..=max_lag`,
/// normalized by lag-0 variance (the standard ACF definition:
/// `ρ(k) = γ(k) / γ(0)` where `γ` is the sample autocovariance).
///
/// Returns an empty vector when `series` has fewer than 2 points or
/// `max_lag == 0`; lags with insufficient overlapping pairs (`lag >=
/// series.len()`) are omitted rather than padded with a placeholder.
pub fn autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    if n < 2 || max_lag == 0 {
        return Vec::new();
    }
    let m = mean(series);
    let denom: f64 = series.iter().map(|&x| (x - m).powi(2)).sum();
    if denom <= 1e-12 {
        // A perfectly flat series has no meaningful autocorrelation
        // structure; report it honestly as all-zero rather than dividing by
        // (near-)zero.
        return vec![0.0; max_lag.min(n.saturating_sub(1))];
    }
    (1..=max_lag)
        .filter(|&lag| lag < n)
        .map(|lag| {
            let cov: f64 = (0..n - lag).map(|i| (series[i] - m) * (series[i + lag] - m)).sum();
            cov / denom
        })
        .collect()
}

/// Partial autocorrelation function (PACF) of `series` at lags `1..=max_lag`
/// via the Durbin-Levinson recursion — the standard method for computing
/// PACF from the ACF without fitting `max_lag` separate full autoregressions.
///
/// The PACF at lag `k` measures the correlation between observations `k`
/// steps apart *after* controlling for the intermediate observations, unlike
/// the raw ACF (which does not control for them). Returns an empty vector
/// under the same degenerate conditions as [`autocorrelation`].
pub fn partial_autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
    let acf = autocorrelation(series, max_lag);
    if acf.is_empty() {
        return Vec::new();
    }
    let k_max = acf.len();

    // Durbin-Levinson recursion. `phi[k][j]` is the j-th AR(k) coefficient;
    // we only need the diagonal `phi[k][k]`, which *is* the PACF at lag k.
    let mut phi = vec![vec![0.0f64; k_max + 1]; k_max + 1];
    let mut pacf = Vec::with_capacity(k_max);

    // rho[0] = 1 by definition; rho[k] = acf[k-1] for k >= 1.
    let rho = |k: usize| -> f64 {
        if k == 0 {
            1.0
        } else {
            acf[k - 1]
        }
    };

    phi[1][1] = rho(1);
    pacf.push(phi[1][1]);

    for k in 2..=k_max {
        let mut numerator = rho(k);
        for j in 1..k {
            numerator -= phi[k - 1][j] * rho(k - j);
        }
        let mut denominator = 1.0;
        for j in 1..k {
            denominator -= phi[k - 1][j] * rho(j);
        }
        let phi_kk = if denominator.abs() > 1e-12 {
            numerator / denominator
        } else {
            0.0
        };
        phi[k][k] = phi_kk;
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi_kk * phi[k - 1][k - j];
        }
        pacf.push(phi_kk);
    }

    pacf
}

/// Fit a real least-squares linear trend to `series` (see [`TrendAnalysis`]).
///
/// `is_significant` uses a data-relative threshold (the fitted slope's total
/// change over the series length exceeds 5% of the series' own standard
/// deviation) rather than an arbitrary fixed constant unrelated to the
/// data's scale, so it behaves sensibly whether `series` holds raw scores in
/// `[0, 1]` or a different metric's native range.
///
/// Returns an error for fewer than 3 points (a line always fits 2 points
/// perfectly, which is not a meaningful trend detection).
pub fn analyze_trend(series: &[f64]) -> EvaluationResult<TrendAnalysis> {
    let n = series.len();
    if n < 3 {
        return Err(EvaluationError::InvalidInput {
            message: "trend analysis requires at least 3 points".to_string(),
        }
        .into());
    }
    let n_f = n as f64;
    let x_mean = (n_f - 1.0) / 2.0;
    let y_mean = mean(series);

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, &y) in series.iter().enumerate() {
        let dx = i as f64 - x_mean;
        numerator += dx * (y - y_mean);
        denominator += dx * dx;
    }
    let slope = if denominator > 1e-12 { numerator / denominator } else { 0.0 };
    let intercept = y_mean - slope * x_mean;

    let predicted: Vec<f64> = (0..n).map(|i| intercept + slope * i as f64).collect();
    let ss_res: f64 = series
        .iter()
        .zip(predicted.iter())
        .map(|(&y, &p)| (y - p).powi(2))
        .sum();
    let ss_tot: f64 = series.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 1e-12 { (1.0 - ss_res / ss_tot).max(0.0) } else { 1.0 };

    let series_std = super::utils::std_dev(series);
    let total_change = slope.abs() * (n_f - 1.0);
    let is_significant = series_std > 1e-9 && total_change > 0.05 * series_std;

    Ok(TrendAnalysis {
        slope,
        intercept,
        r_squared,
        is_significant,
    })
}

/// Simple moving average of `series` with window size `window` (must be
/// `>= 1`). Each output point `i` (for `i >= window - 1`) is the mean of
/// `series[i - window + 1 ..= i]`; the first `window - 1` points have no
/// full window and are omitted (rather than padded with a fabricated value),
/// so the output has length `series.len() - window + 1`.
///
/// Returns an error when `window == 0` or `window > series.len()`.
pub fn moving_average(series: &[f64], window: usize) -> EvaluationResult<Vec<f64>> {
    if window == 0 {
        return Err(EvaluationError::InvalidInput {
            message: "moving_average window must be at least 1".to_string(),
        }
        .into());
    }
    if window > series.len() {
        return Err(EvaluationError::InvalidInput {
            message: format!(
                "moving_average window ({window}) exceeds series length ({})",
                series.len()
            ),
        }
        .into());
    }
    let mut result = Vec::with_capacity(series.len() - window + 1);
    let mut window_sum: f64 = series[..window].iter().sum();
    result.push(window_sum / window as f64);
    for i in window..series.len() {
        window_sum += series[i] - series[i - window];
        result.push(window_sum / window as f64);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocorrelation_alternating_series() {
        let series = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let acf = autocorrelation(&series, 2);
        assert_eq!(acf.len(), 2);
        // Lag 1: strongly negative (adjacent points always opposite sign).
        assert!(acf[0] < -0.5, "acf[0] = {}", acf[0]);
        // Lag 2: strongly positive (same phase two steps later).
        assert!(acf[1] > 0.5, "acf[1] = {}", acf[1]);
    }

    #[test]
    fn test_autocorrelation_constant_series_is_zero() {
        let series = vec![0.5; 10];
        let acf = autocorrelation(&series, 3);
        assert!(acf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_autocorrelation_empty_for_short_series() {
        assert!(autocorrelation(&[1.0], 3).is_empty());
        assert!(autocorrelation(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn test_partial_autocorrelation_ar1_process() {
        // A deterministic AR(1)-like process: x[t] = 0.6 * x[t-1] with a
        // small deterministic perturbation (no `rand` dependency). For a
        // "clean" AR(1) series, the PACF should cut off sharply after lag 1
        // (lag-1 PACF large, lag-2+ much smaller).
        let mut series = vec![1.0];
        for i in 1..40 {
            let prev = series[i - 1];
            series.push(0.6 * prev + 0.01 * (i as f64 * 0.9).sin());
        }
        let pacf = partial_autocorrelation(&series, 4);
        assert_eq!(pacf.len(), 4);
        assert!(
            pacf[0].abs() > pacf[2].abs(),
            "lag-1 PACF ({}) should dominate lag-3 PACF ({}) for an AR(1)-like process",
            pacf[0],
            pacf[2]
        );
    }

    #[test]
    fn test_analyze_trend_detects_real_rising_trend() {
        let series: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let trend = analyze_trend(&series).unwrap();
        assert!((trend.slope - 0.5).abs() < 1e-9);
        assert!(trend.is_significant);
        assert!((trend.r_squared - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_analyze_trend_flat_series_not_significant() {
        let series = vec![0.7; 10];
        let trend = analyze_trend(&series).unwrap();
        assert!(trend.slope.abs() < 1e-9);
        assert!(!trend.is_significant);
    }

    #[test]
    fn test_analyze_trend_insufficient_points_errors() {
        assert!(analyze_trend(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_moving_average_smooths_noise() {
        let series = vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let ma = moving_average(&series, 2).unwrap();
        assert_eq!(ma.len(), 7);
        // A window-2 average of alternating [1, 3] should collapse to a
        // constant 2.0 everywhere.
        for &v in &ma {
            assert!((v - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_moving_average_window_errors() {
        assert!(moving_average(&[1.0, 2.0], 0).is_err());
        assert!(moving_average(&[1.0, 2.0], 5).is_err());
    }
}
