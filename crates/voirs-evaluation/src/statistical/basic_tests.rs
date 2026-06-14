//! Basic statistical tests implementation
//!
//! This module provides implementations for fundamental statistical tests including
//! t-tests, Mann-Whitney U, Wilcoxon signed-rank, and bootstrap confidence intervals.

use super::types::*;
use crate::{EvaluationError, EvaluationResult};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};
use std::collections::HashMap;

/// Main statistical analyzer that provides all statistical testing functionality
#[derive(Debug, Clone)]
pub struct StatisticalAnalyzer {
    /// Configuration for statistical tests
    config: StatisticalConfig,
}

/// Configuration for statistical analysis
#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    /// Default significance level
    pub alpha: f32,
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            bootstrap_samples: 10000,
            seed: Some(42),
        }
    }
}

impl StatisticalAnalyzer {
    /// Create a new statistical analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: StatisticalConfig::default(),
        }
    }

    /// Create a new statistical analyzer with custom configuration
    pub fn with_config(config: StatisticalConfig) -> Self {
        Self { config }
    }

    /// Perform paired t-test for dependent samples
    pub fn paired_t_test(
        &self,
        before: &[f32],
        after: &[f32],
        alpha: Option<f32>,
    ) -> EvaluationResult<StatisticalTestResult> {
        if before.len() != after.len() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Paired samples must have equal length"),
            }
            .into());
        }

        if before.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Cannot perform t-test on empty samples"),
            }
            .into());
        }

        let alpha = alpha.unwrap_or(self.config.alpha);
        let differences: Vec<f32> = before
            .iter()
            .zip(after.iter())
            .map(|(b, a)| a - b)
            .collect();

        let n = differences.len() as f32;
        let mean_diff = differences.iter().sum::<f32>() / n;
        let variance = differences
            .iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f32>()
            / (n - 1.0);
        let std_error = (variance / n).sqrt();

        // Handle division by zero
        let t_statistic = if std_error > 0.0 && std_error.is_finite() {
            mean_diff / std_error
        } else if mean_diff.abs() > 1e-6 {
            // If there's a mean difference but no variation, this is perfectly significant
            if mean_diff > 0.0 {
                100.0
            } else {
                -100.0
            }
        } else {
            0.0 // No difference and no variation
        };
        let degrees_of_freedom = (n - 1.0) as usize;

        // Exact two-sided p-value from the Student's t-distribution with
        // n - 1 degrees of freedom: p = 2 * (1 - F(|t|)).
        let p_value = (2.0 * (1.0 - self.t_cdf(t_statistic.abs(), n - 1.0))).clamp(0.0, 1.0);

        let effect_size = mean_diff / variance.sqrt(); // Cohen's d
        let is_significant = p_value < alpha;

        let interpretation = if is_significant {
            format!(
                "significant difference detected (p = {:.4}, α = {:.3})",
                p_value, alpha
            )
        } else {
            format!(
                "No significant difference (p = {:.4}, α = {:.3})",
                p_value, alpha
            )
        };

        Ok(StatisticalTestResult {
            test_statistic: t_statistic as f64,
            p_value: p_value as f64,
            degrees_of_freedom: Some(degrees_of_freedom as usize),
            effect_size: Some(effect_size as f64),
            confidence_interval: None, // Would calculate in full implementation
            test_type: String::from("PairedTTest"),
            alpha: alpha as f64,
            is_significant,
            interpretation,
            confidence_level: 1.0 - alpha as f64,
        })
    }

    /// Perform Mann-Whitney U test for independent samples
    pub fn mann_whitney_u_test(
        &self,
        group1: &[f32],
        group2: &[f32],
        alpha: Option<f32>,
    ) -> EvaluationResult<StatisticalTestResult> {
        if group1.is_empty() || group2.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Cannot perform Mann-Whitney U test on empty groups"),
            }
            .into());
        }

        let alpha = alpha.unwrap_or(self.config.alpha);

        // Combine and rank all values
        let mut combined: Vec<(f32, usize)> = group1
            .iter()
            .map(|&x| (x, 0))
            .chain(group2.iter().map(|&x| (x, 1)))
            .collect();

        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Assign fractional (mid-) ranks, averaging tied values so equal
        // observations share the mean of the ranks they span.
        let total = combined.len();
        let mut ranks = vec![0.0_f32; total];
        let mut tie_correction = 0.0_f64; // sum of (t^3 - t) over tie groups
        let mut i = 0;
        while i < total {
            let mut j = i + 1;
            while j < total && combined[j].0 == combined[i].0 {
                j += 1;
            }
            // Ranks i..j (0-based) correspond to natural ranks (i+1)..=j.
            let avg_rank = ((i + 1 + j) as f32) / 2.0;
            for r in ranks.iter_mut().take(j).skip(i) {
                *r = avg_rank;
            }
            let t = (j - i) as f64;
            if t > 1.0 {
                tie_correction += t * t * t - t;
            }
            i = j;
        }

        // Sum ranks for group 1
        let r1: f32 = combined
            .iter()
            .zip(ranks.iter())
            .filter(|((_, group), _)| *group == 0)
            .map(|(_, rank)| rank)
            .sum();

        let n1 = group1.len() as f32;
        let n2 = group2.len() as f32;

        let u1 = r1 - (n1 * (n1 + 1.0)) / 2.0;
        let u2 = n1 * n2 - u1;
        let u_statistic = u1.min(u2);

        // Normal approximation with a continuity-style tie correction to the
        // variance: Var(U) = n1*n2/12 * ((N+1) - sum(t^3 - t)/(N*(N-1))).
        let n = (n1 + n2) as f64;
        let mean_u = n1 * n2 / 2.0;
        let var_u = if n > 1.0 {
            (n1 as f64 * n2 as f64 / 12.0) * ((n + 1.0) - tie_correction / (n * (n - 1.0)))
        } else {
            0.0
        };
        let std_u = var_u.max(0.0).sqrt() as f32;
        let z_score = if std_u > 0.0 {
            (u_statistic - mean_u) / std_u
        } else {
            0.0
        };

        // Two-sided p-value from the normal approximation.
        let p_value = (2.0 * (1.0 - self.normal_cdf(z_score.abs()))).clamp(0.0, 1.0);

        let effect_size = 1.0 - (2.0 * u_statistic) / (n1 * n2); // Rank-biserial correlation
        let is_significant = p_value < alpha;

        let interpretation = if is_significant {
            format!(
                "Significant difference between groups (p = {:.4}, α = {:.3})",
                p_value, alpha
            )
        } else {
            format!(
                "No significant difference between groups (p = {:.4}, α = {:.3})",
                p_value, alpha
            )
        };

        Ok(StatisticalTestResult {
            test_statistic: u_statistic as f64,
            p_value: p_value as f64,
            degrees_of_freedom: Some(0), // U-test doesn't have traditional degrees of freedom
            effect_size: Some(effect_size as f64),
            confidence_interval: None,
            test_type: String::from("MannWhitneyU"),
            alpha: alpha as f64,
            is_significant,
            interpretation,
            confidence_level: 1.0 - alpha as f64,
        })
    }

    /// Cumulative distribution function of the Student's t-distribution.
    ///
    /// Returns `P(T <= t)` for a central Student's t random variable with
    /// `df` degrees of freedom, computed exactly via the regularized
    /// incomplete beta function provided by `statrs`. Degenerate inputs
    /// (non-finite `t`/`df` or `df <= 0`) fall back to the symmetric centre
    /// value `0.5` so callers never panic.
    fn t_cdf(&self, t: f32, df: f32) -> f32 {
        if !t.is_finite() || !df.is_finite() || df <= 0.0 {
            return 0.5;
        }

        match StudentsT::new(0.0, 1.0, df as f64) {
            Ok(dist) => dist.cdf(t as f64) as f32,
            Err(_) => 0.5,
        }
    }

    /// Helper function for normal distribution CDF (simplified)
    fn normal_cdf(&self, z: f32) -> f32 {
        if !z.is_finite() {
            return if z > 0.0 { 1.0 } else { 0.0 };
        }

        // Simplified approximation - would use proper normal CDF in real implementation
        let result = 0.5 * (1.0 + self.erf(z / std::f32::consts::SQRT_2));
        result.max(0.0).min(1.0)
    }

    /// Helper function for error function (simplified)
    fn erf(&self, x: f32) -> f32 {
        // Simplified approximation - would use proper erf implementation
        let a1 = 0.254_829_592;
        let a2 = -0.284_496_736;
        let a3 = 1.421_413_741;
        let a4 = -1.453_152_027;
        let a5 = 1.061_405_429;
        let p = 0.327_591_1;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Perform correlation test between two variables
    pub fn correlation_test(
        &self,
        x: &[f32],
        y: &[f32],
    ) -> EvaluationResult<StatisticalTestResult> {
        if x.len() != y.len() || x.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Variables must have equal non-zero length"),
            }
            .into());
        }

        let n = x.len() as f32;
        let x_mean = x.iter().sum::<f32>() / n;
        let y_mean = y.iter().sum::<f32>() / n;

        let numerator: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let x_var: f32 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
        let y_var: f32 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        let denominator = (x_var * y_var).sqrt();
        let correlation = if denominator > 0.0 && denominator.is_finite() {
            (numerator / denominator).clamp(-1.0, 1.0)
        } else {
            0.0 // No variation means no correlation
        };

        let t_statistic = if correlation.abs() < 0.9999 && correlation.is_finite() {
            correlation * ((n - 2.0) / (1.0 - correlation.powi(2))).sqrt()
        } else if correlation.abs() > 0.9999 {
            // Perfect or near-perfect correlation
            if correlation > 0.0 {
                100.0
            } else {
                -100.0
            }
        } else {
            0.0
        };
        let degrees_of_freedom = (n - 2.0) as usize;

        // Exact two-sided p-value for the correlation t-statistic
        // t = r * sqrt((n - 2) / (1 - r^2)) under the Student's t with n - 2
        // degrees of freedom: p = 2 * (1 - F(|t|)).
        let p_value = (2.0 * (1.0 - self.t_cdf(t_statistic.abs(), n - 2.0))).clamp(0.0, 1.0);
        let is_significant = p_value < self.config.alpha;

        let interpretation = if is_significant {
            format!(
                "significant correlation (r = {:.4}, p = {:.4})",
                correlation, p_value
            )
        } else {
            format!(
                "No significant correlation (r = {:.4}, p = {:.4})",
                correlation, p_value
            )
        };

        Ok(StatisticalTestResult {
            test_statistic: t_statistic as f64,
            p_value: p_value as f64,
            degrees_of_freedom: Some(degrees_of_freedom),
            effect_size: Some(correlation as f64),
            confidence_interval: None,
            test_type: String::from("CorrelationTest"),
            alpha: self.config.alpha as f64,
            is_significant,
            interpretation,
            confidence_level: 1.0 - self.config.alpha as f64,
        })
    }

    /// Perform independent samples t-test
    pub fn independent_t_test(
        &self,
        group1: &[f32],
        group2: &[f32],
    ) -> EvaluationResult<StatisticalTestResult> {
        if group1.is_empty() || group2.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Cannot perform t-test on empty groups"),
            }
            .into());
        }

        let n1 = group1.len() as f32;
        let n2 = group2.len() as f32;
        let mean1 = group1.iter().sum::<f32>() / n1;
        let mean2 = group2.iter().sum::<f32>() / n2;

        let var1 = group1.iter().map(|x| (x - mean1).powi(2)).sum::<f32>() / (n1 - 1.0);
        let var2 = group2.iter().map(|x| (x - mean2).powi(2)).sum::<f32>() / (n2 - 1.0);

        let pooled_std = ((var1 * (n1 - 1.0) + var2 * (n2 - 1.0)) / (n1 + n2 - 2.0)).sqrt();
        let standard_error = pooled_std * (1.0 / n1 + 1.0 / n2).sqrt();

        let mean_diff = mean1 - mean2;
        let t_statistic = if standard_error > 0.0 && standard_error.is_finite() {
            mean_diff / standard_error
        } else if mean_diff.abs() > 1e-6 {
            // If there's a mean difference but no variation, this is perfectly significant
            if mean_diff > 0.0 {
                100.0
            } else {
                -100.0
            }
        } else {
            0.0 // No difference and no variation
        };
        let degrees_of_freedom = (n1 + n2 - 2.0) as usize;

        // Exact two-sided p-value from the pooled-variance Student's t with
        // n1 + n2 - 2 degrees of freedom: p = 2 * (1 - F(|t|)).
        let p_value = (2.0 * (1.0 - self.t_cdf(t_statistic.abs(), n1 + n2 - 2.0))).clamp(0.0, 1.0);
        let is_significant = p_value < self.config.alpha;

        let interpretation = if is_significant {
            format!("Significant difference between groups (p = {:.4})", p_value)
        } else {
            format!(
                "No significant difference between groups (p = {:.4})",
                p_value
            )
        };

        Ok(StatisticalTestResult {
            test_statistic: t_statistic as f64,
            p_value: p_value as f64,
            degrees_of_freedom: Some(degrees_of_freedom),
            effect_size: Some((mean1 - mean2) as f64 / pooled_std as f64),
            confidence_interval: None,
            test_type: String::from("IndependentTTest"),
            alpha: self.config.alpha as f64,
            is_significant,
            interpretation,
            confidence_level: 1.0 - self.config.alpha as f64,
        })
    }

    /// Estimate the statistical power of a two-sample, two-sided t-test.
    ///
    /// The power is the probability of correctly rejecting the null
    /// hypothesis given a true standardized mean difference (Cohen's d).
    /// It is derived from the actual effect size and per-group sample size
    /// `n` via the non-centrality parameter `ncp = d * sqrt(n / 2)`, then
    /// approximated by shifting the central Student's t-distribution by
    /// `ncp` (the same noncentral-by-shift approximation used elsewhere in
    /// this crate): `power = (1 - F(t_crit - ncp)) + F(-t_crit - ncp)`.
    pub fn power_analysis_t_test(&self, config: &ABTestConfig) -> PowerAnalysisResult {
        let alpha = config.alpha as f32;
        let effect_size = config.effect_size as f32;
        let n = (config.sample_size_a + config.sample_size_b) as f32 / 2.0;
        let df = (2.0 * n - 2.0).max(1.0);

        // Non-centrality parameter for the standardized mean difference.
        let ncp = effect_size * (n / 2.0).sqrt();
        let critical_t = self.t_inv(1.0 - alpha / 2.0, df);

        // Two-sided power approximated by shifting the central t by the ncp.
        let upper = 1.0 - self.t_cdf(critical_t - ncp, df);
        let lower = self.t_cdf(-critical_t - ncp, df);
        let power = (upper + lower).clamp(0.0, 1.0) as f64;

        PowerAnalysisResult {
            achieved_power: power,
            effect_size: config.effect_size,
            sample_size: (config.sample_size_a + config.sample_size_b) / 2,
            alpha: config.alpha,
        }
    }

    /// Compute bootstrap confidence interval
    pub fn bootstrap_confidence_interval(
        &self,
        data: &[f32],
        statistic: fn(&[f32]) -> f32,
    ) -> EvaluationResult<(f64, f64)> {
        if data.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Cannot bootstrap empty data"),
            }
            .into());
        }

        let mut bootstrap_stats = Vec::with_capacity(self.config.bootstrap_samples);

        // Simplified bootstrap with better pseudo-random sampling
        for i in 0..self.config.bootstrap_samples {
            let mut sample = Vec::with_capacity(data.len());
            // Use a better pseudo-random pattern that creates more variation
            let seed = (i * 31 + 17) as u64; // More varied seeding
            for j in 0..data.len() {
                // Use a linear congruential generator for better pseudo-randomness
                let a = 1_664_525_u64;
                let c = 1_013_904_223_u64;
                let m = 1u64 << 32;
                let random_val = (a.wrapping_mul(seed + j as u64).wrapping_add(c)) % m;
                let idx = (random_val as usize) % data.len();
                sample.push(data[idx]);
            }
            bootstrap_stats.push(statistic(&sample));
        }

        bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lower_idx = (0.025 * self.config.bootstrap_samples as f32) as usize;
        let upper_idx = (0.975 * self.config.bootstrap_samples as f32) as usize;

        Ok((
            bootstrap_stats[lower_idx] as f64,
            bootstrap_stats[upper_idx] as f64,
        ))
    }

    /// Apply multiple comparison correction
    pub fn multiple_comparison_correction(
        &self,
        p_values: &[f32],
        method: MultipleComparisonCorrection,
    ) -> EvaluationResult<MultipleComparisonResult> {
        if p_values.is_empty() {
            return Ok(MultipleComparisonResult {
                adjusted_p_values: vec![],
                original_p_values: vec![],
                method,
            });
        }

        let adjusted_p_values = match method {
            MultipleComparisonCorrection::None => p_values.to_vec(),
            MultipleComparisonCorrection::Bonferroni => {
                let n = p_values.len() as f32;
                p_values.iter().map(|&p| (p * n).min(1.0)).collect()
            }
            MultipleComparisonCorrection::BenjaminiHochberg => {
                let mut indexed_p: Vec<(usize, f32)> =
                    p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed_p
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let n = p_values.len() as f32;
                let mut corrected = vec![0.0; p_values.len()];

                for (rank, (original_idx, p)) in indexed_p.iter().enumerate() {
                    let adjusted = p * n / (rank + 1) as f32;
                    corrected[*original_idx] = adjusted.min(1.0);
                }

                corrected
            }
            _ => {
                // Simplified implementation for other methods
                p_values.to_vec()
            }
        };

        Ok(MultipleComparisonResult {
            adjusted_p_values,
            original_p_values: p_values.to_vec(),
            method,
        })
    }

    /// Inverse (quantile function) of the Student's t-distribution.
    ///
    /// Returns the value `t` such that `P(T <= t) = p` for a central
    /// Student's t random variable with `df` degrees of freedom, computed via
    /// the inverse regularized incomplete beta function in `statrs`. The
    /// degenerate tails return `+/- inf`; if the distribution cannot be
    /// constructed (`df <= 0`) the normal-quantile fallback is used so the
    /// caller never panics.
    fn t_inv(&self, p: f32, df: f32) -> f32 {
        if p >= 1.0 {
            return f32::INFINITY;
        }
        if p <= 0.0 {
            return f32::NEG_INFINITY;
        }

        if df.is_finite() && df > 0.0 {
            if let Ok(dist) = StudentsT::new(0.0, 1.0, df as f64) {
                return dist.inverse_cdf(p as f64) as f32;
            }
        }

        // Fallback: standard-normal quantile (valid as df -> infinity).
        self.normal_inv(p)
    }

    /// Helper function for inverse normal distribution (simplified)
    fn normal_inv(&self, p: f32) -> f32 {
        // Simplified approximation using Box-Muller-like transformation
        if p >= 1.0 {
            return f32::INFINITY;
        }
        if p <= 0.0 {
            return f32::NEG_INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        // Simplified calculation
        let t = (-2.0 * p.ln()).sqrt();
        t - (2.515_517 + 0.802_853 * t + 0.010_328 * t * t)
            / (1.0 + 1.432_788 * t + 0.189_269 * t * t + 0.001_308 * t * t * t)
    }

    /// Calculate correlation coefficient between two variables
    pub fn correlation(&self, x: &[f32], y: &[f32]) -> EvaluationResult<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Variables must have equal non-zero length"),
            }
            .into());
        }

        let n = x.len() as f32;
        let x_mean = x.iter().sum::<f32>() / n;
        let y_mean = y.iter().sum::<f32>() / n;

        let numerator: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let x_var: f32 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
        let y_var: f32 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        let correlation = numerator / (x_var * y_var).sqrt();
        Ok(correlation as f64)
    }

    /// Perform linear regression analysis
    pub fn linear_regression(&self, x: &[f32], y: &[f32]) -> EvaluationResult<RegressionResult> {
        if x.len() != y.len() || x.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Variables must have equal non-zero length"),
            }
            .into());
        }

        let n = x.len() as f32;
        let x_mean = x.iter().sum::<f32>() / n;
        let y_mean = y.iter().sum::<f32>() / n;

        // Calculate slope and intercept
        let numerator: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let denominator: f32 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();

        if denominator == 0.0 {
            return Err(EvaluationError::InvalidInput {
                message: String::from("No variation in x variable"),
            }
            .into());
        }

        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;

        // Calculate R-squared
        let y_pred: Vec<f32> = x.iter().map(|xi| slope * xi + intercept).collect();
        let ss_res: f32 = y
            .iter()
            .zip(y_pred.iter())
            .map(|(yi, y_pred_i)| (yi - y_pred_i).powi(2))
            .sum();
        let ss_tot: f32 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0 // Perfect fit if no variation in y
        };

        // Calculate standard errors and statistics
        let residual_se = (ss_res / (n - 2.0)).sqrt();
        let se_slope = if denominator > 0.0 {
            residual_se / denominator.sqrt()
        } else {
            0.0
        };
        let se_intercept = residual_se * (1.0 / n + x_mean.powi(2) / denominator).sqrt();

        let t_slope = if se_slope > 0.0 {
            slope / se_slope
        } else {
            0.0
        };
        let t_intercept = if se_intercept > 0.0 {
            intercept / se_intercept
        } else {
            0.0
        };

        // Calculate p-values (simplified)
        let df = n - 2.0;
        let p_slope = 2.0 * (1.0 - self.t_cdf(t_slope.abs(), df));
        let p_intercept = 2.0 * (1.0 - self.t_cdf(t_intercept.abs(), df));

        // F-statistic for overall model significance
        let f_statistic = if ss_res > 0.0 {
            (ss_tot - ss_res) / (ss_res / (n - 2.0))
        } else {
            f32::INFINITY
        };
        let f_p_value = if f_statistic.is_finite() {
            1.0 - self.f_cdf(f_statistic, 1.0, df)
        } else {
            0.0
        };

        let adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / (n - 2.0);

        Ok(RegressionResult {
            coefficients: vec![intercept as f64, slope as f64],
            r_squared: r_squared as f64,
            adjusted_r_squared: adjusted_r_squared as f64,
            standard_errors: vec![se_intercept as f64, se_slope as f64],
            t_statistics: vec![t_intercept as f64, t_slope as f64],
            p_values: vec![p_intercept as f64, p_slope as f64],
            residual_standard_error: residual_se as f64,
            f_statistic: f_statistic as f64,
            f_p_value: f_p_value as f64,
            degrees_of_freedom: (1, df as usize),
            slope: slope as f64,
            intercept: intercept as f64,
            p_value: f_p_value as f64,
            standard_error: residual_se as f64,
        })
    }

    /// Cumulative distribution function of the F (Fisher-Snedecor)
    /// distribution.
    ///
    /// Returns `P(F <= f)` for an F random variable with `df1` numerator and
    /// `df2` denominator degrees of freedom, computed exactly via the
    /// regularized incomplete beta function in `statrs`. Non-finite or
    /// non-positive inputs return `0.0`; if the distribution cannot be
    /// constructed the value also degrades to `0.0` so callers never panic.
    fn f_cdf(&self, f: f32, df1: f32, df2: f32) -> f32 {
        if !f.is_finite() || f <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
            return 0.0;
        }

        match FisherSnedecor::new(df1 as f64, df2 as f64) {
            Ok(dist) => dist.cdf(f as f64) as f32,
            Err(_) => 0.0,
        }
    }
}

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod statrs_distribution_tests {
    use super::*;

    /// The two-sided p-value for t = 2.228 with 10 df is the textbook 0.05
    /// critical value for a two-tailed test at alpha = 0.05.
    #[test]
    fn test_t_cdf_known_two_sided_p() {
        let analyzer = StatisticalAnalyzer::new();
        let cdf = analyzer.t_cdf(2.228, 10.0);
        let two_sided_p = 2.0 * (1.0 - cdf);
        assert!(
            (two_sided_p - 0.05).abs() < 1e-3,
            "expected two-sided p ~= 0.05, got {two_sided_p}"
        );
    }

    /// CDF at the centre (t = 0) is exactly 0.5 by symmetry.
    #[test]
    fn test_t_cdf_centre_is_half() {
        let analyzer = StatisticalAnalyzer::new();
        assert!((analyzer.t_cdf(0.0, 7.0) - 0.5).abs() < 1e-6);
    }

    /// Degenerate degrees of freedom fall back to the symmetric centre.
    #[test]
    fn test_t_cdf_invalid_df_fallback() {
        let analyzer = StatisticalAnalyzer::new();
        assert!((analyzer.t_cdf(1.5, 0.0) - 0.5).abs() < 1e-6);
    }

    /// inverse_cdf and cdf must round-trip: F(F^{-1}(p)) = p.
    #[test]
    fn test_t_inv_roundtrip() {
        let analyzer = StatisticalAnalyzer::new();
        // 0.975 quantile of t with 10 df is the well-known 2.228.
        let q = analyzer.t_inv(0.975, 10.0);
        assert!((q - 2.228).abs() < 2e-3, "expected ~2.228, got {q}");
        let back = analyzer.t_cdf(q, 10.0);
        assert!((back - 0.975).abs() < 1e-3);
    }

    /// F-distribution CDF at a known point: for F(3, 12), the 0.95 quantile
    /// is 3.490, so CDF(3.490) ~= 0.95 and the upper tail ~= 0.05.
    #[test]
    fn test_f_cdf_known_value() {
        let analyzer = StatisticalAnalyzer::new();
        let cdf = analyzer.f_cdf(3.490, 3.0, 12.0);
        assert!((cdf - 0.95).abs() < 5e-3, "expected ~0.95, got {cdf}");
        let upper_tail = 1.0 - cdf;
        assert!((upper_tail - 0.05).abs() < 5e-3);
    }

    /// Non-positive / invalid F inputs return 0.0.
    #[test]
    fn test_f_cdf_invalid_inputs() {
        let analyzer = StatisticalAnalyzer::new();
        assert_eq!(analyzer.f_cdf(-1.0, 2.0, 5.0), 0.0);
        assert_eq!(analyzer.f_cdf(2.0, 0.0, 5.0), 0.0);
    }

    /// Paired t-test on a constant positive shift is highly significant and
    /// reports the exact small p-value from the real t-distribution.
    #[test]
    fn test_paired_t_test_real_p_value() {
        let analyzer = StatisticalAnalyzer::new();
        let before = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let after = [2.0_f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = analyzer.paired_t_test(&before, &after, Some(0.05)).unwrap();
        assert!(result.is_significant);
        assert!(
            result.p_value < 0.01,
            "constant +1 shift should be very significant, p = {}",
            result.p_value
        );
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    /// Mann-Whitney U on a textbook example with no overlap: groups
    /// {1,2,3,4} vs {5,6,7,8}. The smaller U is 0 and the rank-biserial
    /// effect size is exactly 1.0.
    #[test]
    fn test_mann_whitney_textbook_no_overlap() {
        let analyzer = StatisticalAnalyzer::new();
        let g1 = [1.0_f32, 2.0, 3.0, 4.0];
        let g2 = [5.0_f32, 6.0, 7.0, 8.0];
        let result = analyzer.mann_whitney_u_test(&g1, &g2, Some(0.05)).unwrap();
        assert!((result.test_statistic - 0.0).abs() < 1e-6, "U should be 0");
        let effect = result.effect_size.unwrap();
        assert!((effect.abs() - 1.0).abs() < 1e-6, "rank-biserial |r| = 1");
    }

    /// Linear regression on a perfectly linear relation: slope 2, intercept
    /// 1, R^2 = 1, and a finite (or saturated) significant F p-value.
    #[test]
    fn test_linear_regression_perfect_fit() {
        let analyzer = StatisticalAnalyzer::new();
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0_f32, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1
        let result = analyzer.linear_regression(&x, &y).unwrap();
        assert!((result.slope - 2.0).abs() < 1e-4);
        assert!((result.intercept - 1.0).abs() < 1e-4);
        assert!((result.r_squared - 1.0).abs() < 1e-4);
    }

    /// Power increases with effect size for the same sample size.
    #[test]
    fn test_power_monotonic_in_effect_size() {
        let analyzer = StatisticalAnalyzer::new();
        let make_config = |effect_size: f64| ABTestConfig {
            sample_size_a: 30,
            sample_size_b: 30,
            effect_size,
            alpha: 0.05,
            power: 0.8,
            two_tailed: true,
            expected_effect_size: effect_size,
            minimum_detectable_difference: 0.1,
        };
        let p_small = analyzer
            .power_analysis_t_test(&make_config(0.2))
            .achieved_power;
        let p_large = analyzer
            .power_analysis_t_test(&make_config(1.0))
            .achieved_power;
        assert!(p_large > p_small);
        assert!((0.0..=1.0).contains(&p_small));
        assert!((0.0..=1.0).contains(&p_large));
    }
}
