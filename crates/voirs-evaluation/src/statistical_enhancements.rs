//! Statistical Analysis Enhancements
//!
//! This module provides enhanced statistical testing and analysis capabilities:
//! - Robust hypothesis testing with effect size calculations
//! - Non-parametric alternatives for non-normal data
//! - Multiple comparison corrections
//! - Bootstrap confidence intervals
//! - Advanced correlation analysis

use crate::statistical::{CorrelationResult, StatisticalTestResult};

/// Basic t-test result structure
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_freedom: f64,
    /// Whether the result is statistically significant
    pub significant: bool,
}
use crate::EvaluationError;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::*;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::{rngs::StdRng, thread_rng, Random, SeedableRng};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal, StudentsT};
use statrs::statistics::{OrderStatistics, Statistics};
use std::collections::HashMap;

/// Enhanced statistical analyzer with robust testing capabilities
pub struct EnhancedStatisticalAnalyzer {
    /// Bootstrap sample size for confidence intervals
    bootstrap_samples: usize,
    /// Alpha level for significance testing
    alpha: f64,
    /// Random seed for reproducible results
    seed: Option<u64>,
}

impl EnhancedStatisticalAnalyzer {
    /// Create new enhanced statistical analyzer
    pub fn new() -> Self {
        Self {
            bootstrap_samples: 10000,
            alpha: 0.05,
            seed: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(bootstrap_samples: usize, alpha: f64, seed: Option<u64>) -> Self {
        Self {
            bootstrap_samples,
            alpha,
            seed,
        }
    }

    /// Robust two-sample t-test with effect size and confidence intervals
    pub fn robust_t_test(
        &self,
        group1: &[f64],
        group2: &[f64],
    ) -> Result<EnhancedTTestResult, EvaluationError> {
        if group1.is_empty() || group2.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Cannot perform t-test on empty groups".to_string(),
            });
        }

        // Check for normality using Shapiro-Wilk test (simplified)
        let normal1 = self.check_normality(group1)?;
        let normal2 = self.check_normality(group2)?;

        let result = if normal1 && normal2 {
            // Use parametric t-test
            self.welch_t_test(group1, group2)?
        } else {
            // Use non-parametric Mann-Whitney U test
            self.mann_whitney_u_test(group1, group2)?
        };

        // Calculate effect size (Cohen's d)
        let cohens_d = self.calculate_cohens_d(group1, group2)?;

        // Bootstrap confidence intervals
        let (ci_lower, ci_upper) = self.bootstrap_difference_ci(group1, group2)?;

        Ok(EnhancedTTestResult {
            statistic: result.statistic,
            p_value: result.p_value,
            degrees_freedom: result.degrees_freedom,
            effect_size: cohens_d,
            confidence_interval: (ci_lower, ci_upper),
            test_type: if normal1 && normal2 {
                "Welch's t-test".to_string()
            } else {
                "Mann-Whitney U".to_string()
            },
            normality_group1: normal1,
            normality_group2: normal2,
            sample_size_group1: group1.len(),
            sample_size_group2: group2.len(),
        })
    }

    /// Enhanced correlation analysis with multiple methods
    pub fn enhanced_correlation_analysis(
        &self,
        x: &[f64],
        y: &[f64],
    ) -> Result<EnhancedCorrelationResult, EvaluationError> {
        if x.len() != y.len() || x.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Arrays must have equal length and be non-empty".to_string(),
            });
        }

        // Pearson correlation
        let pearson_r = self.pearson_correlation(x, y)?;
        let pearson_p = self.correlation_significance_test(pearson_r, x.len())?;

        // Spearman correlation (rank-based, robust to outliers)
        let spearman_r = self.spearman_correlation(x, y)?;
        let spearman_p = self.correlation_significance_test(spearman_r, x.len())?;

        // Kendall's tau (robust to outliers)
        let kendall_tau = self.kendall_tau(x, y)?;
        let kendall_p = self.correlation_significance_test(kendall_tau, x.len())?;

        // Bootstrap confidence interval for Pearson correlation
        let (ci_lower, ci_upper) = self.bootstrap_correlation_ci(x, y)?;

        Ok(EnhancedCorrelationResult {
            pearson_r,
            pearson_p_value: pearson_p,
            spearman_r,
            spearman_p_value: spearman_p,
            kendall_tau,
            kendall_p_value: kendall_p,
            confidence_interval: (ci_lower, ci_upper),
            sample_size: x.len(),
            outlier_count: self.count_outliers(x, y)?,
        })
    }

    /// Multiple comparison correction using Benjamini-Hochberg procedure
    pub fn benjamini_hochberg_correction(
        &self,
        p_values: &[f64],
    ) -> Result<Vec<f64>, EvaluationError> {
        if p_values.is_empty() {
            return Ok(Vec::new());
        }

        let n = p_values.len() as f64;
        let mut indexed_p: Vec<(usize, f64)> =
            p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by p-value
        indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut adjusted_p = vec![0.0; p_values.len()];

        // Apply Benjamini-Hochberg procedure
        for (rank, &(original_index, p_value)) in indexed_p.iter().enumerate() {
            let rank_f = (rank + 1) as f64;
            let adjusted = p_value * n / rank_f;
            adjusted_p[original_index] = adjusted.min(1.0);
        }

        // Ensure monotonicity
        for i in (0..indexed_p.len() - 1).rev() {
            let curr_idx = indexed_p[i].0;
            let next_idx = indexed_p[i + 1].0;
            adjusted_p[curr_idx] = adjusted_p[curr_idx].min(adjusted_p[next_idx]);
        }

        Ok(adjusted_p)
    }

    /// Power analysis for t-test
    pub fn power_analysis(
        &self,
        effect_size: f64,
        sample_size: usize,
        alpha: f64,
    ) -> Result<f64, EvaluationError> {
        if effect_size < 0.0 || alpha <= 0.0 || alpha >= 1.0 || sample_size == 0 {
            return Err(EvaluationError::InvalidInput {
                message: "Invalid parameters for power analysis".to_string(),
            });
        }

        let df = (sample_size - 1) as f64;
        let t_dist =
            StudentsT::new(0.0, 1.0, df).map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to create t-distribution: {}", e),
                source: None,
            })?;

        // Critical t-value for two-tailed test
        let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        // Non-centrality parameter
        let ncp = effect_size * (sample_size as f64).sqrt();

        // Power calculation (simplified)
        let power: f64 = 1.0 - t_dist.cdf(t_crit - ncp) + t_dist.cdf(-t_crit - ncp);

        Ok(power.max(0.0).min(1.0))
    }

    /// Batch analysis with parallel processing
    pub fn batch_analysis(
        &self,
        datasets: &[(Vec<f64>, Vec<f64>)],
    ) -> Result<Vec<EnhancedTTestResult>, EvaluationError> {
        datasets
            .par_iter()
            .map(|(group1, group2)| self.robust_t_test(group1, group2))
            .collect()
    }

    // Private helper methods

    fn check_normality(&self, data: &[f64]) -> Result<bool, EvaluationError> {
        if data.len() < 3 {
            return Ok(true); // Assume normal for very small samples
        }

        // Simplified normality check using skewness and kurtosis
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(true); // Constant data
        }

        // Calculate skewness
        let skewness = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / data.len() as f64;

        // Calculate kurtosis
        let kurtosis = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>()
            / data.len() as f64
            - 3.0;

        // Simple heuristic: consider normal if skewness and kurtosis are reasonable
        Ok(skewness.abs() < 2.0 && kurtosis.abs() < 7.0)
    }

    fn welch_t_test(&self, group1: &[f64], group2: &[f64]) -> Result<TTestResult, EvaluationError> {
        let n1 = group1.len() as f64;
        let n2 = group2.len() as f64;

        let mean1 = group1.iter().sum::<f64>() / n1;
        let mean2 = group2.iter().sum::<f64>() / n2;

        let var1 = group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2 = group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        let pooled_se = (var1 / n1 + var2 / n2).sqrt();
        let t_statistic = (mean1 - mean2) / pooled_se;

        // Welch-Satterthwaite degrees of freedom
        let df = (var1 / n1 + var2 / n2).powi(2)
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

        let t_dist =
            StudentsT::new(0.0, 1.0, df).map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to create t-distribution: {}", e),
                source: None,
            })?;

        let p_value = 2.0 * (1.0 - t_dist.cdf(t_statistic.abs()));

        Ok(TTestResult {
            statistic: t_statistic,
            p_value,
            degrees_freedom: df,
            significant: p_value < self.alpha,
        })
    }

    fn mann_whitney_u_test(
        &self,
        group1: &[f64],
        group2: &[f64],
    ) -> Result<TTestResult, EvaluationError> {
        // Simplified Mann-Whitney U implementation
        let mut combined: Vec<(f64, usize)> = Vec::new();

        for &x in group1 {
            combined.push((x, 0));
        }
        for &x in group2 {
            combined.push((x, 1));
        }

        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Calculate ranks
        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }

        let u1: f64 = ranks
            .iter()
            .enumerate()
            .filter(|(idx, _)| combined[*idx].1 == 0)
            .map(|(_, &rank)| rank)
            .sum::<f64>()
            - (group1.len() * (group1.len() + 1)) as f64 / 2.0;

        let n1 = group1.len() as f64;
        let n2 = group2.len() as f64;

        // Convert to z-score for large samples
        let mean_u = n1 * n2 / 2.0;
        let var_u = n1 * n2 * (n1 + n2 + 1.0) / 12.0;
        let z = (u1 - mean_u) / var_u.sqrt();

        let normal = Normal::new(0.0, 1.0).map_err(|e| EvaluationError::ProcessingError {
            message: format!("Failed to create normal distribution: {}", e),
            source: None,
        })?;

        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));

        Ok(TTestResult {
            statistic: z,
            p_value,
            degrees_freedom: n1 + n2 - 2.0,
            significant: p_value < self.alpha,
        })
    }

    fn calculate_cohens_d(&self, group1: &[f64], group2: &[f64]) -> Result<f64, EvaluationError> {
        let mean1 = group1.iter().sum::<f64>() / group1.len() as f64;
        let mean2 = group2.iter().sum::<f64>() / group2.len() as f64;

        let var1 =
            group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (group1.len() - 1) as f64;
        let var2 =
            group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (group2.len() - 1) as f64;

        let pooled_sd = ((var1 + var2) / 2.0).sqrt();

        if pooled_sd == 0.0 {
            return Ok(0.0);
        }

        Ok((mean1 - mean2) / pooled_sd)
    }

    fn bootstrap_difference_ci(
        &self,
        group1: &[f64],
        group2: &[f64],
    ) -> Result<(f64, f64), EvaluationError> {
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::seq::SliceRandom;

        let mut rng = if let Some(seed) = self.seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let mut differences = Vec::with_capacity(self.bootstrap_samples);

        for _ in 0..self.bootstrap_samples {
            let sample1: Vec<f64> = (0..group1.len())
                .map(|_| {
                    let idx = rng.gen_range(0..group1.len());
                    group1[idx]
                })
                .collect();
            let sample2: Vec<f64> = (0..group2.len())
                .map(|_| {
                    let idx = rng.gen_range(0..group2.len());
                    group2[idx]
                })
                .collect();

            let mean1 = sample1.iter().sum::<f64>() / sample1.len() as f64;
            let mean2 = sample2.iter().sum::<f64>() / sample2.len() as f64;
            differences.push(mean1 - mean2);
        }

        differences.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha_2 = self.alpha / 2.0;
        let lower_idx = (alpha_2 * self.bootstrap_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha_2) * self.bootstrap_samples as f64) as usize;

        Ok((differences[lower_idx], differences[upper_idx]))
    }

    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64, EvaluationError> {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            num += diff_x * diff_y;
            den_x += diff_x * diff_x;
            den_y += diff_y * diff_y;
        }

        let denominator = (den_x * den_y).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(num / denominator)
        }
    }

    fn spearman_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64, EvaluationError> {
        // Convert to ranks
        let rank_x = self.assign_ranks(x);
        let rank_y = self.assign_ranks(y);

        // Calculate Pearson correlation of ranks
        self.pearson_correlation(&rank_x, &rank_y)
    }

    fn kendall_tau(&self, x: &[f64], y: &[f64]) -> Result<f64, EvaluationError> {
        let n = x.len();
        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let x_diff = x[i] - x[j];
                let y_diff = y[i] - y[j];

                if (x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0) {
                    concordant += 1;
                } else if (x_diff > 0.0 && y_diff < 0.0) || (x_diff < 0.0 && y_diff > 0.0) {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            Ok(0.0)
        } else {
            Ok((concordant - discordant) as f64 / total_pairs as f64)
        }
    }

    fn assign_ranks(&self, data: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> =
            data.iter().enumerate().map(|(i, &x)| (i, x)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = vec![0.0; data.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            while j < indexed.len() && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }

        ranks
    }

    fn correlation_significance_test(&self, r: f64, n: usize) -> Result<f64, EvaluationError> {
        if n < 3 {
            return Ok(1.0); // Cannot test significance with < 3 samples
        }

        let df = (n - 2) as f64;
        let t = r * (df / (1.0 - r * r)).sqrt();

        let t_dist =
            StudentsT::new(0.0, 1.0, df).map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to create t-distribution: {}", e),
                source: None,
            })?;

        Ok(2.0 * (1.0 - t_dist.cdf(t.abs())))
    }

    fn bootstrap_correlation_ci(
        &self,
        x: &[f64],
        y: &[f64],
    ) -> Result<(f64, f64), EvaluationError> {
        use scirs2_core::random::prelude::*;

        let mut rng = if let Some(seed) = self.seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let mut correlations = Vec::with_capacity(self.bootstrap_samples);
        let n = x.len();

        for _ in 0..self.bootstrap_samples {
            // Fisher-Yates shuffle for sampling without replacement
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
            let sample_x: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
            let sample_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

            let r = self.pearson_correlation(&sample_x, &sample_y)?;
            correlations.push(r);
        }

        correlations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha_2 = self.alpha / 2.0;
        let lower_idx = (alpha_2 * self.bootstrap_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha_2) * self.bootstrap_samples as f64) as usize;

        Ok((correlations[lower_idx], correlations[upper_idx]))
    }

    fn count_outliers(&self, x: &[f64], y: &[f64]) -> Result<usize, EvaluationError> {
        // Simple outlier detection using IQR method
        let mut combined: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();
        combined.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = combined.len();
        let q1 = combined[n / 4];
        let q3 = combined[3 * n / 4];
        let iqr = q3 - q1;
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let outliers = combined
            .iter()
            .filter(|&&val| val < lower_bound || val > upper_bound)
            .count();

        Ok(outliers)
    }
}

impl Default for EnhancedStatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced t-test result with effect size and confidence intervals
#[derive(Debug, Clone)]
pub struct EnhancedTTestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_freedom: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Confidence interval for the difference
    pub confidence_interval: (f64, f64),
    /// Type of test used (parametric or non-parametric)
    pub test_type: String,
    /// Whether group 1 data appears normally distributed
    pub normality_group1: bool,
    /// Whether group 2 data appears normally distributed
    pub normality_group2: bool,
    /// Sample size of group 1
    pub sample_size_group1: usize,
    /// Sample size of group 2
    pub sample_size_group2: usize,
}

/// Enhanced correlation result with multiple methods
#[derive(Debug, Clone)]
pub struct EnhancedCorrelationResult {
    /// Pearson correlation coefficient
    pub pearson_r: f64,
    /// P-value for Pearson correlation
    pub pearson_p_value: f64,
    /// Spearman rank correlation coefficient
    pub spearman_r: f64,
    /// P-value for Spearman correlation
    pub spearman_p_value: f64,
    /// Kendall's tau correlation coefficient
    pub kendall_tau: f64,
    /// P-value for Kendall's tau
    pub kendall_p_value: f64,
    /// Confidence interval for Pearson correlation
    pub confidence_interval: (f64, f64),
    /// Sample size
    pub sample_size: usize,
    /// Number of detected outliers
    pub outlier_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_analyzer_creation() {
        let analyzer = EnhancedStatisticalAnalyzer::new();
        assert_eq!(analyzer.bootstrap_samples, 10000);
        assert_eq!(analyzer.alpha, 0.05);
    }

    #[test]
    fn test_robust_t_test() {
        let analyzer = EnhancedStatisticalAnalyzer::with_config(1000, 0.05, Some(42));
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = analyzer.robust_t_test(&group1, &group2).unwrap();
        assert!(result.effect_size.abs() > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_enhanced_correlation() {
        let analyzer = EnhancedStatisticalAnalyzer::with_config(1000, 0.05, Some(42));
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = analyzer.enhanced_correlation_analysis(&x, &y).unwrap();
        assert!((result.pearson_r - 1.0).abs() < 0.01); // Should be perfect correlation
        assert!(result.spearman_r > 0.9);
        assert!(result.kendall_tau > 0.8);
    }

    #[test]
    fn test_benjamini_hochberg_correction() {
        let analyzer = EnhancedStatisticalAnalyzer::new();
        let p_values = vec![0.01, 0.04, 0.03, 0.07, 0.06];
        let adjusted = analyzer.benjamini_hochberg_correction(&p_values).unwrap();

        assert_eq!(adjusted.len(), p_values.len());
        // All adjusted p-values should be >= original p-values
        for (orig, adj) in p_values.iter().zip(adjusted.iter()) {
            assert!(adj >= orig);
        }
    }

    #[test]
    fn test_power_analysis() {
        let analyzer = EnhancedStatisticalAnalyzer::new();
        let power = analyzer.power_analysis(0.5, 30, 0.05).unwrap();
        assert!(power >= 0.0 && power <= 1.0);

        // Larger effect size should give higher power
        let power_large = analyzer.power_analysis(1.0, 30, 0.05).unwrap();
        assert!(power_large > power);
    }
}
