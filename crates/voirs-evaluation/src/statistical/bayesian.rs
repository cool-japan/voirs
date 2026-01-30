//! Bayesian statistical analysis for TTS evaluation
//!
//! This module provides Bayesian methods for analyzing evaluation metrics,
//! including parameter estimation, hypothesis testing, and model comparison.

use crate::EvaluationError;
use scirs2_core::random::{Normal, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Bayesian A/B test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianABTestResult {
    /// Probability that A is better than B
    pub prob_a_better: f64,
    /// Probability that B is better than A
    pub prob_b_better: f64,
    /// Expected value difference (A - B)
    pub expected_difference: f64,
    /// 95% credible interval for the difference
    pub credible_interval_95: (f64, f64),
    /// 99% credible interval for the difference
    pub credible_interval_99: (f64, f64),
    /// Posterior mean for A
    pub posterior_mean_a: f64,
    /// Posterior mean for B
    pub posterior_mean_b: f64,
    /// Bayes factor (evidence for A > B)
    pub bayes_factor: f64,
}

/// Bayesian parameter estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianEstimation {
    /// Posterior mean
    pub posterior_mean: f64,
    /// Posterior standard deviation
    pub posterior_std: f64,
    /// 95% credible interval
    pub credible_interval_95: (f64, f64),
    /// 99% credible interval
    pub credible_interval_99: (f64, f64),
    /// Mode of the posterior (MAP estimate)
    pub posterior_mode: f64,
}

/// Bayesian model comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianModelComparison {
    /// Model names
    pub model_names: Vec<String>,
    /// Log marginal likelihood for each model
    pub log_marginal_likelihoods: Vec<f64>,
    /// Bayes factors (relative to first model)
    pub bayes_factors: Vec<f64>,
    /// Model probabilities (assuming equal priors)
    pub model_probabilities: Vec<f64>,
    /// Best model index
    pub best_model_index: usize,
}

/// Prior distribution type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PriorType {
    /// Uniform prior (non-informative)
    Uniform,
    /// Normal (Gaussian) prior
    Normal,
    /// Beta prior (for probabilities)
    Beta,
    /// Gamma prior (for positive values)
    Gamma,
}

/// Prior distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorParameters {
    /// Prior type
    pub prior_type: PriorType,
    /// First parameter (e.g., mean for Normal, alpha for Beta)
    pub param1: f64,
    /// Second parameter (e.g., std for Normal, beta for Beta)
    pub param2: f64,
}

/// Bayesian analyzer for TTS evaluation
pub struct BayesianAnalyzer {
    /// Number of MCMC samples
    n_samples: usize,
    /// Random number generator
    rng: scirs2_core::random::ChaCha8Rng,
}

impl BayesianAnalyzer {
    /// Create a new Bayesian analyzer
    #[must_use]
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            rng: scirs2_core::random::ChaCha8Rng::seed_from_u64(fastrand::u64(..)),
        }
    }

    /// Create a default Bayesian analyzer (10,000 samples)
    #[must_use]
    pub fn default() -> Self {
        Self::new(10_000)
    }

    /// Perform Bayesian A/B test comparing two groups
    ///
    /// Uses Beta-Binomial model for proportions or Normal model for continuous metrics
    ///
    /// # Errors
    /// Returns an error if the sample sizes are too small
    pub fn ab_test(
        &mut self,
        group_a: &[f64],
        group_b: &[f64],
    ) -> Result<BayesianABTestResult, EvaluationError> {
        if group_a.is_empty() || group_b.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Groups cannot be empty".to_string(),
            });
        }

        if group_a.len() < 2 || group_b.len() < 2 {
            return Err(EvaluationError::InvalidInput {
                message: "Groups must have at least 2 samples".to_string(),
            });
        }

        // Calculate sample statistics
        let mean_a = group_a.iter().sum::<f64>() / group_a.len() as f64;
        let mean_b = group_b.iter().sum::<f64>() / group_b.len() as f64;

        let var_a =
            group_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (group_a.len() - 1) as f64;
        let var_b =
            group_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (group_b.len() - 1) as f64;

        // Use Normal model with conjugate Normal-Inverse-Gamma prior
        // For simplicity, we'll use posterior approximation
        let posterior_mean_a = mean_a;
        let posterior_mean_b = mean_b;
        let posterior_var_a = var_a / group_a.len() as f64;
        let posterior_var_b = var_b / group_b.len() as f64;

        // Monte Carlo sampling from posterior
        let mut samples_a = Vec::with_capacity(self.n_samples);
        let mut samples_b = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let sample_a = self
                .rng
                .sample(Normal::new(posterior_mean_a, posterior_var_a.sqrt()).unwrap());
            let sample_b = self
                .rng
                .sample(Normal::new(posterior_mean_b, posterior_var_b.sqrt()).unwrap());
            samples_a.push(sample_a);
            samples_b.push(sample_b);
        }

        // Calculate probability that A is better than B
        let mut count_a_better = 0;
        let mut differences = Vec::with_capacity(self.n_samples);

        for i in 0..self.n_samples {
            let diff = samples_a[i] - samples_b[i];
            differences.push(diff);
            if diff > 0.0 {
                count_a_better += 1;
            }
        }

        let prob_a_better = count_a_better as f64 / self.n_samples as f64;
        let prob_b_better = 1.0 - prob_a_better;

        // Calculate expected difference and credible intervals
        let expected_difference = differences.iter().sum::<f64>() / differences.len() as f64;

        let mut sorted_diffs = differences.clone();
        sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ci_95_lower = sorted_diffs[(self.n_samples as f64 * 0.025) as usize];
        let ci_95_upper = sorted_diffs[(self.n_samples as f64 * 0.975) as usize];
        let ci_99_lower = sorted_diffs[(self.n_samples as f64 * 0.005) as usize];
        let ci_99_upper = sorted_diffs[(self.n_samples as f64 * 0.995) as usize];

        // Calculate Bayes factor (simplified)
        // BF = P(D|H1) / P(D|H0)
        // Using Savage-Dickey approximation
        let bayes_factor = if prob_a_better > 0.5 {
            prob_a_better / (1.0 - prob_a_better)
        } else {
            prob_b_better / (1.0 - prob_b_better)
        };

        Ok(BayesianABTestResult {
            prob_a_better,
            prob_b_better,
            expected_difference,
            credible_interval_95: (ci_95_lower, ci_95_upper),
            credible_interval_99: (ci_99_lower, ci_99_upper),
            posterior_mean_a,
            posterior_mean_b,
            bayes_factor,
        })
    }

    /// Estimate parameter with Bayesian inference
    ///
    /// # Errors
    /// Returns an error if the data is empty or invalid
    pub fn estimate_parameter(
        &mut self,
        data: &[f64],
        prior: &PriorParameters,
    ) -> Result<BayesianEstimation, EvaluationError> {
        if data.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Data cannot be empty".to_string(),
            });
        }

        // Calculate sample statistics
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

        // Update prior with likelihood using conjugate updates
        let (posterior_mean, posterior_var) = match prior.prior_type {
            PriorType::Normal => {
                // Normal-Normal conjugate update
                let prior_mean = prior.param1;
                let prior_var = prior.param2.powi(2);
                let likelihood_var = variance / data.len() as f64;

                let post_var = 1.0 / (1.0 / prior_var + data.len() as f64 / variance);
                let post_mean =
                    post_var * (prior_mean / prior_var + data.len() as f64 * mean / variance);

                (post_mean, post_var)
            }
            PriorType::Uniform => {
                // Non-informative prior, posterior equals likelihood
                (mean, variance / data.len() as f64)
            }
            _ => {
                // For other priors, use simple approximation
                (mean, variance / data.len() as f64)
            }
        };

        // Sample from posterior
        let mut samples = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let sample = self
                .rng
                .sample(Normal::new(posterior_mean, posterior_var.sqrt()).unwrap());
            samples.push(sample);
        }

        // Calculate credible intervals
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ci_95_lower = samples[(self.n_samples as f64 * 0.025) as usize];
        let ci_95_upper = samples[(self.n_samples as f64 * 0.975) as usize];
        let ci_99_lower = samples[(self.n_samples as f64 * 0.005) as usize];
        let ci_99_upper = samples[(self.n_samples as f64 * 0.995) as usize];

        // Calculate posterior mode (approximation)
        let posterior_mode = posterior_mean;

        Ok(BayesianEstimation {
            posterior_mean,
            posterior_std: posterior_var.sqrt(),
            credible_interval_95: (ci_95_lower, ci_95_upper),
            credible_interval_99: (ci_99_lower, ci_99_upper),
            posterior_mode,
        })
    }

    /// Compare multiple models using Bayes factors
    ///
    /// # Errors
    /// Returns an error if the input is invalid
    pub fn compare_models(
        &self,
        model_names: Vec<String>,
        log_likelihoods: Vec<f64>,
    ) -> Result<BayesianModelComparison, EvaluationError> {
        if model_names.len() != log_likelihoods.len() {
            return Err(EvaluationError::InvalidInput {
                message: "Number of model names must match number of log likelihoods".to_string(),
            });
        }

        if model_names.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "At least one model must be provided".to_string(),
            });
        }

        // Calculate Bayes factors relative to first model
        let reference_ll = log_likelihoods[0];
        let bayes_factors: Vec<f64> = log_likelihoods
            .iter()
            .map(|ll| (ll - reference_ll).exp())
            .collect();

        // Calculate model probabilities (assuming equal priors)
        let total: f64 = bayes_factors.iter().sum();
        let model_probabilities: Vec<f64> = bayes_factors.iter().map(|bf| bf / total).collect();

        // Find best model
        let best_model_index = model_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        Ok(BayesianModelComparison {
            model_names,
            log_marginal_likelihoods: log_likelihoods,
            bayes_factors,
            model_probabilities,
            best_model_index,
        })
    }

    /// Interpret Bayes factor according to Jeffreys' scale
    #[must_use]
    pub fn interpret_bayes_factor(bf: f64) -> &'static str {
        match bf {
            bf if bf < 1.0 => "Negative (supports alternative)",
            bf if bf < 3.0 => "Barely worth mentioning",
            bf if bf < 10.0 => "Substantial",
            bf if bf < 30.0 => "Strong",
            bf if bf < 100.0 => "Very strong",
            _ => "Decisive",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_analyzer_creation() {
        let analyzer = BayesianAnalyzer::default();
        assert_eq!(analyzer.n_samples, 10_000);
    }

    #[test]
    fn test_bayesian_ab_test() {
        let mut analyzer = BayesianAnalyzer::new(1000);

        // Group A: mean ~= 4.0
        let group_a = vec![3.9, 4.0, 4.1, 4.0, 3.8, 4.2, 4.0, 3.9];

        // Group B: mean ~= 3.5
        let group_b = vec![3.4, 3.5, 3.6, 3.5, 3.4, 3.6, 3.5, 3.4];

        let result = analyzer.ab_test(&group_a, &group_b).unwrap();

        // A should be better than B
        assert!(result.prob_a_better > 0.5);
        assert!(result.expected_difference > 0.0);
        assert!(result.posterior_mean_a > result.posterior_mean_b);
    }

    #[test]
    fn test_bayesian_ab_test_equal_groups() {
        let mut analyzer = BayesianAnalyzer::new(1000);

        let group_a = vec![4.0, 4.1, 3.9, 4.0, 4.1];
        let group_b = vec![4.0, 3.9, 4.1, 4.0, 3.9];

        let result = analyzer.ab_test(&group_a, &group_b).unwrap();

        // Probabilities should be close to 0.5 (relaxed tolerance for small samples)
        assert!((result.prob_a_better - 0.5).abs() < 0.3);
        assert!((result.prob_b_better - 0.5).abs() < 0.3);
    }

    #[test]
    fn test_bayesian_ab_test_empty_input() {
        let mut analyzer = BayesianAnalyzer::new(1000);
        let empty: Vec<f64> = vec![];
        let group_b = vec![1.0, 2.0, 3.0];

        assert!(analyzer.ab_test(&empty, &group_b).is_err());
    }

    #[test]
    fn test_parameter_estimation() {
        let mut analyzer = BayesianAnalyzer::new(1000);

        let data = vec![4.0, 4.1, 3.9, 4.0, 4.2, 3.8, 4.1, 3.9, 4.0];

        let prior = PriorParameters {
            prior_type: PriorType::Uniform,
            param1: 0.0,
            param2: 1.0,
        };

        let result = analyzer.estimate_parameter(&data, &prior).unwrap();

        // Posterior mean should be close to 4.0
        assert!((result.posterior_mean - 4.0).abs() < 0.1);

        // Credible intervals should contain the mean
        assert!(result.credible_interval_95.0 < result.posterior_mean);
        assert!(result.credible_interval_95.1 > result.posterior_mean);
    }

    #[test]
    fn test_parameter_estimation_normal_prior() {
        let mut analyzer = BayesianAnalyzer::new(1000);

        let data = vec![4.0, 4.1, 3.9, 4.0, 4.2];

        let prior = PriorParameters {
            prior_type: PriorType::Normal,
            param1: 3.5, // Prior mean
            param2: 1.0, // Prior std
        };

        let result = analyzer.estimate_parameter(&data, &prior).unwrap();

        // Posterior mean should be between prior mean and data mean
        assert!(result.posterior_mean > 3.5);
        assert!(result.posterior_mean < 4.2);
    }

    #[test]
    fn test_model_comparison() {
        let analyzer = BayesianAnalyzer::default();

        let model_names = vec![
            "Model A".to_string(),
            "Model B".to_string(),
            "Model C".to_string(),
        ];

        // Model B has highest log likelihood
        let log_likelihoods = vec![-100.0, -90.0, -95.0];

        let result = analyzer
            .compare_models(model_names.clone(), log_likelihoods)
            .unwrap();

        assert_eq!(result.best_model_index, 1); // Model B
        assert_eq!(result.model_names[1], "Model B");
        assert!(result.model_probabilities[1] > result.model_probabilities[0]);
        assert!(result.model_probabilities[1] > result.model_probabilities[2]);
    }

    #[test]
    fn test_bayes_factor_interpretation() {
        assert_eq!(
            BayesianAnalyzer::interpret_bayes_factor(0.5),
            "Negative (supports alternative)"
        );
        assert_eq!(
            BayesianAnalyzer::interpret_bayes_factor(2.0),
            "Barely worth mentioning"
        );
        assert_eq!(BayesianAnalyzer::interpret_bayes_factor(5.0), "Substantial");
        assert_eq!(BayesianAnalyzer::interpret_bayes_factor(15.0), "Strong");
        assert_eq!(
            BayesianAnalyzer::interpret_bayes_factor(50.0),
            "Very strong"
        );
        assert_eq!(BayesianAnalyzer::interpret_bayes_factor(150.0), "Decisive");
    }

    #[test]
    fn test_credible_intervals() {
        let mut analyzer = BayesianAnalyzer::new(1000);

        let data = vec![4.0, 4.1, 3.9, 4.0, 4.2, 3.8, 4.1];

        let prior = PriorParameters {
            prior_type: PriorType::Uniform,
            param1: 0.0,
            param2: 1.0,
        };

        let result = analyzer.estimate_parameter(&data, &prior).unwrap();

        // 99% CI should be wider than 95% CI
        let width_95 = result.credible_interval_95.1 - result.credible_interval_95.0;
        let width_99 = result.credible_interval_99.1 - result.credible_interval_99.0;
        assert!(width_99 > width_95);
    }
}
