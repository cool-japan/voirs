//! A/B testing and multiple comparisons implementation
//!
//! This module provides implementations for A/B testing and multiple comparison methods.

use super::types::*;
use crate::{EvaluationError, EvaluationResult};
use std::collections::HashMap;

/// A/B test configuration
#[derive(Debug, Clone)]
pub struct ABTestConfig {
    /// Significance level (default: 0.05)
    pub alpha: f32,
    /// Minimum detectable effect (Cohen's d, default: 0.2)
    pub minimum_effect_size: f32,
    /// Statistical power (default: 0.8)
    pub power: f32,
    /// Test type (two-tailed or one-tailed)
    pub test_type: TestType,
    /// Multiple comparison correction method
    pub correction_method: CorrectionMethod,
    /// Bootstrap iterations for confidence intervals (default: 1000)
    pub bootstrap_iterations: usize,
}

/// Test type for hypothesis testing
#[derive(Debug, Clone)]
pub enum TestType {
    /// Two-tailed test (difference in either direction)
    TwoTailed,
    /// One-tailed test (A > B)
    OneTailedGreater,
    /// One-tailed test (A < B)
    OneTailedLess,
}

/// Multiple comparison correction methods
#[derive(Debug, Clone)]
pub enum CorrectionMethod {
    /// No correction
    None,
    /// Bonferroni correction
    Bonferroni,
    /// Benjamini-Hochberg (FDR control)
    BenjaminiHochberg,
    /// Holm-Bonferroni method
    HolmBonferroni,
    /// Šidák correction
    Sidak,
}

/// A/B test result
#[derive(Debug, Clone)]
pub struct ABTestResult {
    /// Sample statistics for group A
    pub group_a_stats: GroupStatistics,
    /// Sample statistics for group B
    pub group_b_stats: GroupStatistics,
    /// Effect size (Cohen's d)
    pub effect_size: f32,
    /// Test statistic
    pub test_statistic: f32,
    /// P-value (raw, before correction)
    pub p_value: f32,
    /// Adjusted p-value (after multiple comparison correction)
    pub adjusted_p_value: f32,
    /// Confidence interval for the difference
    pub confidence_interval: (f32, f32),
    /// Whether the test is statistically significant
    pub is_significant: bool,
    /// Power analysis result
    pub power_analysis: PowerAnalysis,
    /// Test configuration used
    pub config: ABTestConfig,
    /// Interpretation of results
    pub interpretation: String,
}

/// Group statistics
#[derive(Debug, Clone)]
pub struct GroupStatistics {
    /// Group name/identifier
    pub name: String,
    /// Sample size
    pub n: usize,
    /// Mean
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Standard error of the mean
    pub std_error: f32,
    /// Median
    pub median: f32,
    /// 95% confidence interval for the mean
    pub confidence_interval: (f32, f32),
}

/// Power analysis result
#[derive(Debug, Clone)]
pub struct PowerAnalysis {
    /// Observed power
    pub observed_power: f32,
    /// Required sample size for desired power
    pub required_sample_size: usize,
    /// Minimum detectable effect with current sample size
    pub minimum_detectable_effect: f32,
    /// Effect size interpretation
    pub effect_interpretation: String,
}

/// Multiple comparison test result
#[derive(Debug, Clone)]
pub struct MultipleComparisonResult {
    /// Pairwise comparison results
    pub pairwise_comparisons: Vec<PairwiseComparison>,
    /// Overall test result (e.g., one-way ANOVA)
    pub overall_test: OverallTestResult,
    /// Correction method used
    pub correction_method: CorrectionMethod,
    /// Family-wise error rate
    pub family_wise_error_rate: f32,
    /// False discovery rate
    pub false_discovery_rate: f32,
}

/// Pairwise comparison result
#[derive(Debug, Clone)]
pub struct PairwiseComparison {
    /// Group A name
    pub group_a: String,
    /// Group B name  
    pub group_b: String,
    /// Mean difference (A - B)
    pub mean_difference: f32,
    /// Effect size
    pub effect_size: f32,
    /// Raw p-value
    pub raw_p_value: f32,
    /// Adjusted p-value
    pub adjusted_p_value: f32,
    /// Confidence interval for the difference
    pub confidence_interval: (f32, f32),
    /// Whether significant after correction
    pub is_significant: bool,
}

/// Overall test result (e.g., ANOVA)
#[derive(Debug, Clone)]
pub struct OverallTestResult {
    /// Test statistic (F-statistic for ANOVA)
    pub test_statistic: f32,
    /// P-value
    pub p_value: f32,
    /// Degrees of freedom
    pub degrees_freedom: (usize, usize),
    /// Effect size (eta-squared)
    pub effect_size: f32,
    /// Whether significant
    pub is_significant: bool,
}

/// Sequential A/B testing result
#[derive(Debug, Clone)]
pub struct SequentialABResult {
    /// Current test result
    pub current_result: ABTestResult,
    /// Stopping decision
    pub stopping_decision: StoppingDecision,
    /// Probability of superiority (A > B)
    pub probability_superior: f32,
    /// Expected loss if stopping now
    pub expected_loss: f32,
    /// Minimum samples needed for reliable conclusion
    pub min_samples_needed: usize,
}

/// Sequential testing stopping decision
#[derive(Debug, Clone)]
pub enum StoppingDecision {
    /// Continue collecting data
    Continue,
    /// Stop - significant difference detected
    StopSignificant,
    /// Stop - no practical significance
    StopFutility,
    /// Stop - maximum sample size reached
    StopMaxSamples,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            minimum_effect_size: 0.2,
            power: 0.8,
            test_type: TestType::TwoTailed,
            correction_method: CorrectionMethod::None,
            bootstrap_iterations: 1000,
        }
    }
}

/// A/B testing analyzer
pub struct ABTestAnalyzer {
    /// Default configuration
    config: ABTestConfig,
}

impl Default for ABTestAnalyzer {
    fn default() -> Self {
        Self {
            config: ABTestConfig::default(),
        }
    }
}

impl ABTestAnalyzer {
    /// Create new A/B test analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: ABTestConfig) -> Self {
        Self { config }
    }

    /// Perform basic A/B test
    pub fn ab_test(&self, group_a: &[f32], group_b: &[f32]) -> EvaluationResult<ABTestResult> {
        self.ab_test_with_config(group_a, group_b, &self.config)
    }

    /// Perform A/B test with custom configuration
    pub fn ab_test_with_config(
        &self,
        group_a: &[f32],
        group_b: &[f32],
        config: &ABTestConfig,
    ) -> EvaluationResult<ABTestResult> {
        if group_a.is_empty() || group_b.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Groups cannot be empty"),
            }
            .into());
        }

        // Calculate group statistics
        let stats_a = self.calculate_group_statistics("A", group_a, config.alpha)?;
        let stats_b = self.calculate_group_statistics("B", group_b, config.alpha)?;

        // Calculate effect size (Cohen's d)
        let pooled_std = self.calculate_pooled_standard_deviation(group_a, group_b);
        let effect_size = if pooled_std > 0.0 {
            (stats_a.mean - stats_b.mean) / pooled_std
        } else {
            0.0
        };

        // Perform t-test
        let (test_statistic, p_value) = self.perform_t_test(group_a, group_b, &config.test_type)?;

        // Calculate confidence interval for the difference
        let confidence_interval = self.calculate_difference_ci(group_a, group_b, config.alpha)?;

        // Apply multiple comparison correction (if applicable)
        let adjusted_p_value = match config.correction_method {
            CorrectionMethod::None => p_value,
            _ => p_value, // Single comparison doesn't need correction
        };

        let is_significant = adjusted_p_value < config.alpha;

        // Perform power analysis
        let power_analysis = self.power_analysis(group_a, group_b, effect_size, config)?;

        // Generate interpretation
        let interpretation =
            self.generate_interpretation(&stats_a, &stats_b, effect_size, p_value, is_significant);

        Ok(ABTestResult {
            group_a_stats: stats_a,
            group_b_stats: stats_b,
            effect_size,
            test_statistic,
            p_value,
            adjusted_p_value,
            confidence_interval,
            is_significant,
            power_analysis,
            config: config.clone(),
            interpretation,
        })
    }

    /// Perform multiple comparison test across multiple groups
    pub fn multiple_comparison_test(
        &self,
        groups: &HashMap<String, Vec<f32>>,
        correction_method: CorrectionMethod,
    ) -> EvaluationResult<MultipleComparisonResult> {
        if groups.len() < 2 {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Need at least 2 groups for comparison"),
            }
            .into());
        }

        // Perform overall ANOVA test
        let overall_test = self.perform_anova(groups)?;

        // Perform pairwise comparisons
        let mut pairwise_comparisons = Vec::new();
        let group_names: Vec<_> = groups.keys().cloned().collect();

        for i in 0..group_names.len() {
            for j in (i + 1)..group_names.len() {
                let group_a_name = &group_names[i];
                let group_b_name = &group_names[j];
                let group_a_data = &groups[group_a_name];
                let group_b_data = &groups[group_b_name];

                let comparison = self.perform_pairwise_comparison(
                    group_a_name.clone(),
                    group_b_name.clone(),
                    group_a_data,
                    group_b_data,
                )?;

                pairwise_comparisons.push(comparison);
            }
        }

        // Apply multiple comparison correction
        self.apply_multiple_comparison_correction(&mut pairwise_comparisons, &correction_method)?;

        // Calculate error rates
        let family_wise_error_rate = self.calculate_family_wise_error_rate(&pairwise_comparisons);
        let false_discovery_rate = self.calculate_false_discovery_rate(&pairwise_comparisons);

        Ok(MultipleComparisonResult {
            pairwise_comparisons,
            overall_test,
            correction_method,
            family_wise_error_rate,
            false_discovery_rate,
        })
    }

    /// Perform sequential A/B test
    pub fn sequential_ab_test(
        &self,
        group_a: &[f32],
        group_b: &[f32],
        max_samples: usize,
    ) -> EvaluationResult<SequentialABResult> {
        // Perform current A/B test
        let current_result = self.ab_test(group_a, group_b)?;

        // Calculate probability of superiority using Bayesian approach
        let probability_superior = self.calculate_probability_superior(group_a, group_b)?;

        // Calculate expected loss
        let expected_loss = self.calculate_expected_loss(group_a, group_b)?;

        // Determine stopping decision
        let stopping_decision = self.determine_stopping_decision(
            &current_result,
            probability_superior,
            expected_loss,
            group_a.len() + group_b.len(),
            max_samples,
        );

        // Calculate minimum samples needed
        let min_samples_needed = self.calculate_min_samples_needed(&current_result)?;

        Ok(SequentialABResult {
            current_result,
            stopping_decision,
            probability_superior,
            expected_loss,
            min_samples_needed,
        })
    }

    /// Calculate group statistics
    fn calculate_group_statistics(
        &self,
        name: &str,
        data: &[f32],
        alpha: f32,
    ) -> EvaluationResult<GroupStatistics> {
        if data.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Data cannot be empty"),
            }
            .into());
        }

        // Filter out invalid values
        let valid_data: Vec<f32> = data.iter().filter(|&&x| x.is_finite()).copied().collect();

        if valid_data.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: String::from("No valid data points found"),
            }
            .into());
        }

        let n = valid_data.len();
        let mean = valid_data.iter().sum::<f32>() / n as f32;

        // Calculate standard deviation
        let variance =
            valid_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1).max(1) as f32;
        let std_dev = variance.sqrt();
        let std_error = std_dev / (n as f32).sqrt();

        // Calculate median
        let mut sorted_data = valid_data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 0 {
            (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
        } else {
            sorted_data[n / 2]
        };

        // Calculate confidence interval for the mean
        let t_critical = self.get_t_critical(alpha, n - 1);
        let margin_error = t_critical * std_error;
        let confidence_interval = (mean - margin_error, mean + margin_error);

        Ok(GroupStatistics {
            name: name.to_string(),
            n,
            mean,
            std_dev,
            std_error,
            median,
            confidence_interval,
        })
    }

    /// Calculate pooled standard deviation
    fn calculate_pooled_standard_deviation(&self, group_a: &[f32], group_b: &[f32]) -> f32 {
        let n1 = group_a.len() as f32;
        let n2 = group_b.len() as f32;

        let var1 = self.calculate_variance(group_a);
        let var2 = self.calculate_variance(group_b);

        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        pooled_var.sqrt()
    }

    /// Calculate variance
    fn calculate_variance(&self, data: &[f32]) -> f32 {
        if data.len() <= 1 {
            return 0.0;
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32
    }

    /// Perform t-test
    fn perform_t_test(
        &self,
        group_a: &[f32],
        group_b: &[f32],
        test_type: &TestType,
    ) -> EvaluationResult<(f32, f32)> {
        let n1 = group_a.len() as f32;
        let n2 = group_b.len() as f32;

        let mean1 = group_a.iter().sum::<f32>() / n1;
        let mean2 = group_b.iter().sum::<f32>() / n2;

        let var1 = self.calculate_variance(group_a);
        let var2 = self.calculate_variance(group_b);

        // Welch's t-test (unequal variances)
        let se_diff = (var1 / n1 + var2 / n2).sqrt();

        if se_diff == 0.0 {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Cannot perform t-test: standard error is zero"),
            }
            .into());
        }

        let t_stat = (mean1 - mean2) / se_diff;

        // Calculate degrees of freedom (Welch-Satterthwaite equation)
        let df = (var1 / n1 + var2 / n2).powi(2)
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

        // Calculate p-value based on test type
        let p_value = match test_type {
            TestType::TwoTailed => 2.0 * (1.0 - self.t_cdf(t_stat.abs(), df as usize)),
            TestType::OneTailedGreater => 1.0 - self.t_cdf(t_stat, df as usize),
            TestType::OneTailedLess => self.t_cdf(t_stat, df as usize),
        };

        Ok((t_stat, p_value))
    }

    /// Calculate confidence interval for the difference between means
    fn calculate_difference_ci(
        &self,
        group_a: &[f32],
        group_b: &[f32],
        alpha: f32,
    ) -> EvaluationResult<(f32, f32)> {
        let n1 = group_a.len() as f32;
        let n2 = group_b.len() as f32;

        let mean1 = group_a.iter().sum::<f32>() / n1;
        let mean2 = group_b.iter().sum::<f32>() / n2;
        let mean_diff = mean1 - mean2;

        let var1 = self.calculate_variance(group_a);
        let var2 = self.calculate_variance(group_b);
        let se_diff = (var1 / n1 + var2 / n2).sqrt();

        // Calculate degrees of freedom
        let df = (var1 / n1 + var2 / n2).powi(2)
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

        let t_critical = self.get_t_critical(alpha, df as usize);
        let margin_error = t_critical * se_diff;

        Ok((mean_diff - margin_error, mean_diff + margin_error))
    }

    /// Perform power analysis
    fn power_analysis(
        &self,
        group_a: &[f32],
        group_b: &[f32],
        effect_size: f32,
        config: &ABTestConfig,
    ) -> EvaluationResult<PowerAnalysis> {
        let n1 = group_a.len();
        let n2 = group_b.len();

        // Calculate observed power
        let observed_power = self.calculate_power(n1, n2, effect_size, config.alpha);

        // Calculate required sample size for desired power
        let required_sample_size = self.calculate_required_sample_size(
            config.minimum_effect_size,
            config.alpha,
            config.power,
        );

        // Calculate minimum detectable effect with current sample size
        let minimum_detectable_effect =
            self.calculate_minimum_detectable_effect(n1, n2, config.alpha, config.power);

        let effect_interpretation = match effect_size.abs() {
            x if x < 0.2 => "Small effect",
            x if x < 0.5 => "Medium effect",
            x if x < 0.8 => "Large effect",
            _ => "Very large effect",
        }
        .to_string();

        Ok(PowerAnalysis {
            observed_power,
            required_sample_size,
            minimum_detectable_effect,
            effect_interpretation,
        })
    }

    /// Calculate statistical power
    fn calculate_power(&self, n1: usize, n2: usize, effect_size: f32, alpha: f32) -> f32 {
        // Simplified power calculation
        let n_harmonic = 2.0 * (n1 * n2) as f32 / (n1 + n2) as f32;
        let ncp = effect_size * (n_harmonic / 4.0).sqrt(); // Non-centrality parameter

        // Approximate power calculation
        let t_critical = self.get_t_critical(alpha, n1 + n2 - 2);
        let power = 1.0 - self.t_cdf(t_critical - ncp, n1 + n2 - 2);
        power.clamp(0.0, 1.0)
    }

    /// Calculate required sample size
    fn calculate_required_sample_size(&self, effect_size: f32, alpha: f32, power: f32) -> usize {
        // Simplified sample size calculation
        let z_alpha = self.get_z_critical(alpha / 2.0);
        let z_beta = self.get_z_critical(1.0 - power);

        let n = (2.0 * (z_alpha + z_beta).powi(2)) / effect_size.powi(2);
        n.ceil() as usize
    }

    /// Calculate minimum detectable effect
    fn calculate_minimum_detectable_effect(
        &self,
        n1: usize,
        n2: usize,
        alpha: f32,
        power: f32,
    ) -> f32 {
        let z_alpha = self.get_z_critical(alpha / 2.0);
        let z_beta = self.get_z_critical(1.0 - power);
        let n_harmonic = 2.0 * (n1 * n2) as f32 / (n1 + n2) as f32;

        (2.0 * (z_alpha + z_beta).powi(2) / n_harmonic).sqrt()
    }

    /// Perform ANOVA test
    fn perform_anova(
        &self,
        groups: &HashMap<String, Vec<f32>>,
    ) -> EvaluationResult<OverallTestResult> {
        let group_data: Vec<&Vec<f32>> = groups.values().collect();
        let k = group_data.len(); // Number of groups

        if k < 2 {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Need at least 2 groups for ANOVA"),
            }
            .into());
        }

        // Calculate overall statistics
        let all_data: Vec<f32> = group_data.iter().flat_map(|g| g.iter()).copied().collect();
        let n_total = all_data.len();
        let grand_mean = all_data.iter().sum::<f32>() / n_total as f32;

        // Calculate sum of squares
        let mut ss_between = 0.0;
        let mut ss_within = 0.0;

        for group in &group_data {
            let n_group = group.len() as f32;
            let group_mean = group.iter().sum::<f32>() / n_group;

            // Between-group sum of squares
            ss_between += n_group * (group_mean - grand_mean).powi(2);

            // Within-group sum of squares
            for &value in group.iter() {
                ss_within += (value - group_mean).powi(2);
            }
        }

        // Degrees of freedom
        let df_between = k - 1;
        let df_within = n_total - k;

        if df_within == 0 {
            return Err(EvaluationError::InvalidInput {
                message: String::from("Not enough data for ANOVA"),
            }
            .into());
        }

        // Mean squares
        let ms_between = ss_between / df_between as f32;
        let ms_within = ss_within / df_within as f32;

        // F-statistic
        let f_stat = if ms_within > 0.0 {
            ms_between / ms_within
        } else {
            f32::INFINITY
        };

        // Calculate p-value (approximation)
        let p_value = self.f_cdf_complement(f_stat, df_between, df_within);

        // Effect size (eta-squared)
        let eta_squared = ss_between / (ss_between + ss_within);

        let is_significant = p_value < 0.05;

        Ok(OverallTestResult {
            test_statistic: f_stat,
            p_value,
            degrees_freedom: (df_between, df_within),
            effect_size: eta_squared,
            is_significant,
        })
    }

    /// Perform pairwise comparison
    fn perform_pairwise_comparison(
        &self,
        group_a: String,
        group_b: String,
        data_a: &[f32],
        data_b: &[f32],
    ) -> EvaluationResult<PairwiseComparison> {
        let mean_a = data_a.iter().sum::<f32>() / data_a.len() as f32;
        let mean_b = data_b.iter().sum::<f32>() / data_b.len() as f32;
        let mean_difference = mean_a - mean_b;

        // Calculate effect size
        let pooled_std = self.calculate_pooled_standard_deviation(data_a, data_b);
        let effect_size = if pooled_std > 0.0 {
            mean_difference / pooled_std
        } else {
            0.0
        };

        // Perform t-test
        let (_, raw_p_value) = self.perform_t_test(data_a, data_b, &TestType::TwoTailed)?;

        // Calculate confidence interval
        let confidence_interval = self.calculate_difference_ci(data_a, data_b, 0.05)?;

        Ok(PairwiseComparison {
            group_a,
            group_b,
            mean_difference,
            effect_size,
            raw_p_value,
            adjusted_p_value: raw_p_value, // Will be adjusted later
            confidence_interval,
            is_significant: false, // Will be updated after correction
        })
    }

    /// Apply multiple comparison correction
    fn apply_multiple_comparison_correction(
        &self,
        comparisons: &mut [PairwiseComparison],
        correction_method: &CorrectionMethod,
    ) -> EvaluationResult<()> {
        let n_comparisons = comparisons.len();

        match correction_method {
            CorrectionMethod::None => {
                for comparison in comparisons.iter_mut() {
                    comparison.adjusted_p_value = comparison.raw_p_value;
                    comparison.is_significant = comparison.raw_p_value < 0.05;
                }
            }
            CorrectionMethod::Bonferroni => {
                for comparison in comparisons.iter_mut() {
                    comparison.adjusted_p_value =
                        (comparison.raw_p_value * n_comparisons as f32).min(1.0);
                    comparison.is_significant = comparison.adjusted_p_value < 0.05;
                }
            }
            CorrectionMethod::BenjaminiHochberg => {
                // Sort p-values
                let mut indexed_p_values: Vec<(usize, f32)> = comparisons
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.raw_p_value))
                    .collect();
                indexed_p_values
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                // Apply Benjamini-Hochberg correction
                for (rank, &(original_index, _)) in indexed_p_values.iter().enumerate() {
                    let bh_factor = (n_comparisons as f32) / ((rank + 1) as f32);
                    comparisons[original_index].adjusted_p_value =
                        (comparisons[original_index].raw_p_value * bh_factor).min(1.0);
                    comparisons[original_index].is_significant =
                        comparisons[original_index].adjusted_p_value < 0.05;
                }
            }
            CorrectionMethod::HolmBonferroni => {
                // Sort p-values
                let mut indexed_p_values: Vec<(usize, f32)> = comparisons
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.raw_p_value))
                    .collect();
                indexed_p_values
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                // Apply Holm-Bonferroni correction
                for (rank, &(original_index, _)) in indexed_p_values.iter().enumerate() {
                    let holm_factor = (n_comparisons - rank) as f32;
                    comparisons[original_index].adjusted_p_value =
                        (comparisons[original_index].raw_p_value * holm_factor).min(1.0);
                    comparisons[original_index].is_significant =
                        comparisons[original_index].adjusted_p_value < 0.05;
                }
            }
            CorrectionMethod::Sidak => {
                for comparison in comparisons.iter_mut() {
                    comparison.adjusted_p_value =
                        1.0 - (1.0 - comparison.raw_p_value).powf(n_comparisons as f32);
                    comparison.is_significant = comparison.adjusted_p_value < 0.05;
                }
            }
        }

        Ok(())
    }

    /// Calculate family-wise error rate
    fn calculate_family_wise_error_rate(&self, comparisons: &[PairwiseComparison]) -> f32 {
        let significant_count = comparisons.iter().filter(|c| c.is_significant).count();
        significant_count as f32 / comparisons.len() as f32
    }

    /// Calculate false discovery rate
    fn calculate_false_discovery_rate(&self, comparisons: &[PairwiseComparison]) -> f32 {
        let significant_comparisons: Vec<_> =
            comparisons.iter().filter(|c| c.is_significant).collect();
        if significant_comparisons.is_empty() {
            return 0.0;
        }

        // Approximate FDR based on adjusted p-values
        let avg_adjusted_p = significant_comparisons
            .iter()
            .map(|c| c.adjusted_p_value)
            .sum::<f32>()
            / significant_comparisons.len() as f32;

        avg_adjusted_p
    }

    /// Calculate probability of superiority (Bayesian approach)
    fn calculate_probability_superior(
        &self,
        group_a: &[f32],
        group_b: &[f32],
    ) -> EvaluationResult<f32> {
        // Simple frequentist approximation to Bayesian probability
        let (t_stat, _) = self.perform_t_test(group_a, group_b, &TestType::OneTailedGreater)?;
        let df = group_a.len() + group_b.len() - 2;

        // Convert t-statistic to probability
        let prob = 1.0 - self.t_cdf(t_stat, df);
        Ok(prob.clamp(0.0, 1.0))
    }

    /// Calculate expected loss
    fn calculate_expected_loss(&self, group_a: &[f32], group_b: &[f32]) -> EvaluationResult<f32> {
        let mean_a = group_a.iter().sum::<f32>() / group_a.len() as f32;
        let mean_b = group_b.iter().sum::<f32>() / group_b.len() as f32;
        let diff = (mean_a - mean_b).abs();

        let var_a = self.calculate_variance(group_a);
        let var_b = self.calculate_variance(group_b);
        let pooled_var = (var_a + var_b) / 2.0;

        // Normalized expected loss
        let expected_loss = if pooled_var > 0.0 {
            diff / pooled_var.sqrt()
        } else {
            0.0
        };

        Ok(expected_loss)
    }

    /// Determine stopping decision for sequential testing
    fn determine_stopping_decision(
        &self,
        result: &ABTestResult,
        prob_superior: f32,
        expected_loss: f32,
        current_samples: usize,
        max_samples: usize,
    ) -> StoppingDecision {
        // Strong evidence threshold
        if prob_superior > 0.95 || prob_superior < 0.05 {
            return StoppingDecision::StopSignificant;
        }

        // Futility threshold
        if expected_loss < 0.1 && current_samples > max_samples / 2 {
            return StoppingDecision::StopFutility;
        }

        // Maximum samples reached
        if current_samples >= max_samples {
            return StoppingDecision::StopMaxSamples;
        }

        StoppingDecision::Continue
    }

    /// Calculate minimum samples needed
    fn calculate_min_samples_needed(&self, result: &ABTestResult) -> EvaluationResult<usize> {
        let current_power = result.power_analysis.observed_power;
        if current_power >= 0.8 {
            Ok(result.group_a_stats.n + result.group_b_stats.n)
        } else {
            Ok(result.power_analysis.required_sample_size)
        }
    }

    /// Generate interpretation
    fn generate_interpretation(
        &self,
        stats_a: &GroupStatistics,
        stats_b: &GroupStatistics,
        effect_size: f32,
        p_value: f32,
        is_significant: bool,
    ) -> String {
        let direction = if stats_a.mean > stats_b.mean {
            "higher"
        } else {
            "lower"
        };
        let magnitude = match effect_size.abs() {
            x if x < 0.2 => "negligible",
            x if x < 0.5 => "small",
            x if x < 0.8 => "medium",
            _ => "large",
        };

        let significance_text = if is_significant {
            "statistically significant"
        } else {
            "not statistically significant"
        };

        format!(
            "Group {} has a {} mean ({:.3}) compared to Group {} ({:.3}). \
            The difference is {} (p = {:.4}, Cohen's d = {:.3}) with a {} effect size.",
            stats_a.name,
            direction,
            stats_a.mean,
            stats_b.name,
            stats_b.mean,
            significance_text,
            p_value,
            effect_size,
            magnitude
        )
    }

    /// Get t-critical value (approximation)
    fn get_t_critical(&self, alpha: f32, df: usize) -> f32 {
        // Simplified approximation - in practice would use lookup table or more accurate method
        if df > 30 {
            self.get_z_critical(alpha / 2.0)
        } else {
            // Rough approximation
            let base_z = self.get_z_critical(alpha / 2.0);
            base_z * (1.0 + 1.0 / (4.0 * df as f32))
        }
    }

    /// Get z-critical value (approximation)
    fn get_z_critical(&self, alpha: f32) -> f32 {
        // Common critical values
        if alpha <= 0.001 {
            3.291
        } else if alpha <= 0.005 {
            2.807
        } else if alpha <= 0.01 {
            2.576
        } else if alpha <= 0.025 {
            1.96
        } else if alpha <= 0.05 {
            1.645
        } else {
            1.282
        }
    }

    /// Approximate t-distribution CDF
    fn t_cdf(&self, t: f32, df: usize) -> f32 {
        if df > 30 {
            // Use normal approximation for large df
            self.standard_normal_cdf(t)
        } else {
            // Rough approximation for small df
            let normal_p = self.standard_normal_cdf(t);
            let adjustment = 1.0 / (4.0 * df as f32);
            (normal_p * (1.0 + adjustment)).clamp(0.0, 1.0)
        }
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, x: f32) -> f32 {
        if x < -8.0 {
            return 0.0;
        }
        if x > 8.0 {
            return 1.0;
        }

        // Abramowitz & Stegun approximation
        let a1 = 0.254_829_592;
        let a2 = -0.284_496_736;
        let a3 = 1.421_413_741;
        let a4 = -1.453_152_027;
        let a5 = 1.061_405_429;
        let p = 0.327_591_1;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x_abs = x.abs();

        let t = 1.0 / (1.0 + p * x_abs);
        let y = 1.0
            - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs / 2.0).exp();

        0.5 * (1.0 + sign * y)
    }

    /// Approximate F-distribution complement CDF
    fn f_cdf_complement(&self, f: f32, df1: usize, df2: usize) -> f32 {
        if f <= 0.0 {
            return 1.0;
        }

        // Very rough approximation
        // Convert F to chi-square approximation
        let chi_square_approx = f * df1 as f32;

        // Rough p-value approximation
        if chi_square_approx > 20.0 {
            0.0
        } else if chi_square_approx > 10.0 {
            0.01
        } else if chi_square_approx > 5.0 {
            0.05
        } else {
            0.1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ab_test() {
        let analyzer = ABTestAnalyzer::new();
        let group_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = analyzer.ab_test(&group_a, &group_b).unwrap();

        assert_eq!(result.group_a_stats.n, 5);
        assert_eq!(result.group_b_stats.n, 5);
        assert!(result.group_a_stats.mean < result.group_b_stats.mean);
        assert!(result.effect_size.abs() > 0.0);
    }

    #[test]
    fn test_power_analysis() {
        let analyzer = ABTestAnalyzer::new();
        let group_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = analyzer.ab_test(&group_a, &group_b).unwrap();

        assert!(result.power_analysis.observed_power >= 0.0);
        assert!(result.power_analysis.observed_power <= 1.0);
        assert!(result.power_analysis.required_sample_size > 0);
    }

    #[test]
    fn test_multiple_comparison() {
        let analyzer = ABTestAnalyzer::new();
        let mut groups = HashMap::new();
        groups.insert(String::from("A"), vec![1.0, 2.0, 3.0]);
        groups.insert(String::from("B"), vec![3.0, 4.0, 5.0]);
        groups.insert(String::from("C"), vec![5.0, 6.0, 7.0]);

        let result = analyzer
            .multiple_comparison_test(&groups, CorrectionMethod::Bonferroni)
            .unwrap();

        assert_eq!(result.pairwise_comparisons.len(), 3); // 3 choose 2
        assert!(result.overall_test.p_value >= 0.0);
        assert!(result.overall_test.p_value <= 1.0);
    }

    #[test]
    fn test_sequential_testing() {
        let analyzer = ABTestAnalyzer::new();
        let group_a = vec![1.0, 2.0, 3.0];
        let group_b = vec![4.0, 5.0, 6.0];

        let result = analyzer
            .sequential_ab_test(&group_a, &group_b, 100)
            .unwrap();

        assert!(result.probability_superior >= 0.0);
        assert!(result.probability_superior <= 1.0);
        assert!(result.expected_loss >= 0.0);
        assert!(result.min_samples_needed > 0);
    }

    #[test]
    fn test_correction_methods() {
        let analyzer = ABTestAnalyzer::new();
        let mut groups = HashMap::new();
        groups.insert("A".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        groups.insert("B".to_string(), vec![2.0, 3.0, 4.0, 5.0]);
        groups.insert("C".to_string(), vec![3.0, 4.0, 5.0, 6.0]);

        // Test different correction methods
        let corrections = vec![
            CorrectionMethod::None,
            CorrectionMethod::Bonferroni,
            CorrectionMethod::BenjaminiHochberg,
            CorrectionMethod::HolmBonferroni,
        ];

        for correction in corrections {
            let result = analyzer
                .multiple_comparison_test(&groups, correction)
                .unwrap();
            assert_eq!(result.pairwise_comparisons.len(), 3);

            // Check that adjusted p-values are reasonable
            for comparison in &result.pairwise_comparisons {
                assert!(comparison.adjusted_p_value >= 0.0);
                assert!(comparison.adjusted_p_value <= 1.0);
                assert!(comparison.raw_p_value >= 0.0);
                assert!(comparison.raw_p_value <= 1.0);
            }
        }
    }

    #[test]
    fn test_invalid_input_handling() {
        let analyzer = ABTestAnalyzer::new();

        // Empty groups
        let empty_group = vec![];
        let normal_group = vec![1.0, 2.0, 3.0];
        assert!(analyzer.ab_test(&empty_group, &normal_group).is_err());

        // Single group multiple comparison
        let mut single_group = HashMap::new();
        single_group.insert("A".to_string(), vec![1.0, 2.0, 3.0]);
        assert!(analyzer
            .multiple_comparison_test(&single_group, CorrectionMethod::None)
            .is_err());
    }
}
