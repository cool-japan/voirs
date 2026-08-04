//! Real two-sample statistical significance testing.
//!
//! This module implements a genuine two-sided Welch's t-test (the
//! unequal-variance t-test) used by both
//! [`crate::progress::analytics::ComprehensiveAnalyticsFramework`] and
//! [`crate::progress::core::ComprehensiveAnalyticsFramework`] to report a
//! real, data-dependent p-value and effect size instead of a hardcoded
//! constant.
//!
//! Welch's t-test (rather than the pooled-variance Student's t-test) is
//! used because the two samples being compared (e.g. a user's performance
//! before vs. after a time period, or two different users' score
//! histories) have no reason to share the same variance.

use statrs::distribution::{ContinuousCDF, StudentsT};

/// Minimum number of data points required in *each* sample for a variance
/// (and therefore a t-statistic) to be estimated at all.
const MIN_SAMPLE_SIZE: usize = 2;

/// Real outcome of a [`welch_t_test`] run: a p-value and an effect size,
/// both computed from the actual sample data (never a constant).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchTTestOutcome {
    /// Two-sided p-value from the Welch-Satterthwaite Student's
    /// t-distribution.
    pub p_value: f64,
    /// Cohen's-d-style standardized effect size: the mean difference
    /// divided by the pooled (unweighted-average) standard deviation of
    /// the two samples.
    pub effect_size: f64,
}

/// Perform a real, two-sided Welch's t-test (unequal-variance t-test)
/// between `baseline` and `comparison`.
///
/// Returns `None` when either sample has fewer than [`MIN_SAMPLE_SIZE`]
/// points, since a variance -- and therefore a t-statistic -- cannot be
/// estimated from fewer than two observations. Callers should treat `None`
/// as "not enough data to test" and report an honest neutral result (see
/// [`insufficient_data_result`]) rather than fabricating one.
///
/// The Welch-Satterthwaite equation is used for the degrees of freedom,
/// which correctly handles the common case where the two samples have
/// different variances (unlike the simpler pooled-variance Student's
/// t-test).
#[must_use]
pub fn welch_t_test(baseline: &[f64], comparison: &[f64]) -> Option<WelchTTestOutcome> {
    if baseline.len() < MIN_SAMPLE_SIZE || comparison.len() < MIN_SAMPLE_SIZE {
        return None;
    }

    let n1 = baseline.len() as f64;
    let n2 = comparison.len() as f64;

    let mean1 = baseline.iter().sum::<f64>() / n1;
    let mean2 = comparison.iter().sum::<f64>() / n2;

    let var1 = baseline.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = comparison.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let se_sq = var1 / n1 + var2 / n2;
    let mean_diff = mean2 - mean1;

    // Welch-Satterthwaite degrees of freedom. Falls back to n1 + n2 - 2
    // only in the degenerate zero-variance case, purely to keep
    // `StudentsT::new` well-defined -- the p-value in that branch is
    // driven by the saturated t-statistic below, not by this df.
    let df = if se_sq > 0.0 {
        let numerator = se_sq * se_sq;
        let denominator = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
        if denominator > 0.0 {
            numerator / denominator
        } else {
            (n1 + n2 - 2.0).max(1.0)
        }
    } else {
        (n1 + n2 - 2.0).max(1.0)
    };

    let t_statistic = if se_sq > 0.0 {
        mean_diff / se_sq.sqrt()
    } else if mean_diff.abs() > 1e-9 {
        // Zero variance in both samples but the means differ: a large but
        // finite t keeps the p-value computation well-defined (p -> 0)
        // instead of producing NaN/Inf.
        mean_diff.signum() * 1.0e6
    } else {
        0.0
    };

    let p_value = match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => (2.0 * (1.0 - dist.cdf(t_statistic.abs()))).clamp(0.0, 1.0),
        Err(_) => return None,
    };

    // Cohen's-d-style effect size using the unweighted-average standard
    // deviation of the two samples (the standard choice when variances are
    // not assumed equal).
    let pooled_std = ((var1 + var2) / 2.0).sqrt();
    let effect_size = if pooled_std > 0.0 {
        mean_diff / pooled_std
    } else if mean_diff.abs() > 1e-9 {
        mean_diff.signum() * 10.0
    } else {
        0.0
    };

    Some(WelchTTestOutcome {
        p_value,
        effect_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_are_not_significant() {
        let a = vec![0.5, 0.5, 0.5, 0.5];
        let b = vec![0.5, 0.5, 0.5, 0.5];
        let outcome = welch_t_test(&a, &b).expect("both groups have >= 2 samples");
        assert!((outcome.p_value - 1.0).abs() < 1e-9);
        assert_eq!(outcome.effect_size, 0.0);
    }

    #[test]
    fn clearly_different_samples_are_significant() {
        let a = vec![0.1, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08];
        let b = vec![0.9, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87];
        let outcome = welch_t_test(&a, &b).expect("both groups have >= 2 samples");
        assert!(outcome.p_value < 0.01, "p_value = {}", outcome.p_value);
        assert!(outcome.effect_size > 1.0);
    }

    #[test]
    fn noisy_overlapping_samples_are_not_significant() {
        let a = vec![0.40, 0.55, 0.30, 0.60, 0.45, 0.50];
        let b = vec![0.42, 0.58, 0.33, 0.57, 0.47, 0.52];
        let outcome = welch_t_test(&a, &b).expect("both groups have >= 2 samples");
        assert!(outcome.p_value > 0.05, "p_value = {}", outcome.p_value);
    }

    #[test]
    fn insufficient_samples_return_none() {
        assert!(welch_t_test(&[0.5], &[0.4, 0.6]).is_none());
        assert!(welch_t_test(&[], &[0.4, 0.6]).is_none());
        assert!(welch_t_test(&[0.4, 0.6], &[]).is_none());
    }

    /// The crux of "not hardcoded": a bigger real effect must produce a
    /// smaller (more significant) p-value than a smaller real effect,
    /// given the same baseline.
    #[test]
    fn p_value_varies_with_effect_magnitude() {
        let baseline = vec![0.5, 0.52, 0.48, 0.51, 0.49];
        let small_shift = vec![0.55, 0.57, 0.53, 0.56, 0.54];
        let large_shift = vec![0.95, 0.97, 0.93, 0.96, 0.94];

        let small = welch_t_test(&baseline, &small_shift).unwrap();
        let large = welch_t_test(&baseline, &large_shift).unwrap();

        assert!(
            large.p_value < small.p_value,
            "large shift p={} should be < small shift p={}",
            large.p_value,
            small.p_value
        );
        assert!(large.effect_size.abs() > small.effect_size.abs());
    }

    #[test]
    fn zero_variance_with_mean_difference_is_significant() {
        let a = vec![0.2, 0.2, 0.2];
        let b = vec![0.8, 0.8, 0.8];
        let outcome = welch_t_test(&a, &b).expect("both groups have >= 2 samples");
        assert!(outcome.p_value < 0.01);
        assert!(outcome.effect_size > 0.0);
    }
}
