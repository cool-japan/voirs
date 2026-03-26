//! Differential Privacy mechanisms for privacy-preserving machine learning.
//!
//! This module implements state-of-the-art differential privacy techniques including
//! Gaussian and Laplace mechanisms, gradient clipping, and privacy accounting.

use scirs2_core::ndarray::DataOwned;
use scirs2_core::random::{thread_rng, Distribution, Rng};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use thiserror::Error;

/// Differential privacy error types
#[derive(Debug, Error)]
pub enum DPError {
    /// Invalid epsilon parameter (must be positive)
    #[error("Invalid privacy budget: epsilon must be positive")]
    InvalidEpsilon,

    /// Invalid delta parameter (must be in (0, 1))
    #[error("Invalid delta: must be in (0, 1)")]
    InvalidDelta,

    /// Privacy budget exceeded error with requested and available amounts
    #[error("Privacy budget exceeded: requested {requested}, available {available}")]
    BudgetExceeded {
        /// Requested privacy budget amount
        requested: f32,
        /// Available privacy budget amount
        available: f32,
    },

    /// Invalid sensitivity parameter
    #[error("Invalid sensitivity: must be positive")]
    InvalidSensitivity,

    /// General configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPConfig {
    /// Privacy parameter epsilon (privacy budget)
    pub epsilon: f32,

    /// Privacy parameter delta (failure probability)
    pub delta: f32,

    /// Noise distribution type
    pub noise_distribution: NoiseDistribution,

    /// Gradient clipping norm
    pub clip_norm: f32,

    /// Enable privacy accounting
    pub enable_accounting: bool,

    /// Target delta for privacy amplification
    pub target_delta: Option<f32>,
}

impl Default for DPConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            noise_distribution: NoiseDistribution::Gaussian,
            clip_norm: 1.0,
            enable_accounting: true,
            target_delta: None,
        }
    }
}

/// Noise distribution types for differential privacy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseDistribution {
    /// Gaussian (normal) distribution for (ε, δ)-DP
    Gaussian,
    /// Laplace distribution for ε-DP
    Laplace,
    /// Exponential mechanism
    Exponential,
}

/// Privacy budget tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Total epsilon budget
    pub total_epsilon: f32,

    /// Remaining epsilon budget
    pub remaining_epsilon: f32,

    /// Total delta budget
    pub total_delta: f32,

    /// Remaining delta budget
    pub remaining_delta: f32,

    /// Composition history
    pub composition_history: Vec<PrivacyConsumption>,
}

/// Privacy consumption record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConsumption {
    /// Epsilon consumed
    pub epsilon: f32,

    /// Delta consumed
    pub delta: f32,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Operation description
    pub operation: String,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(epsilon: f32, delta: f32) -> Result<Self, DPError> {
        if epsilon <= 0.0 {
            return Err(DPError::InvalidEpsilon);
        }

        if delta <= 0.0 || delta >= 1.0 {
            return Err(DPError::InvalidDelta);
        }

        Ok(Self {
            total_epsilon: epsilon,
            remaining_epsilon: epsilon,
            total_delta: delta,
            remaining_delta: delta,
            composition_history: Vec::new(),
        })
    }

    /// Consume privacy budget
    pub fn consume(&mut self, epsilon: f32, delta: f32, operation: String) -> Result<(), DPError> {
        if epsilon > self.remaining_epsilon {
            return Err(DPError::BudgetExceeded {
                requested: epsilon,
                available: self.remaining_epsilon,
            });
        }

        if delta > self.remaining_delta {
            return Err(DPError::BudgetExceeded {
                requested: delta,
                available: self.remaining_delta,
            });
        }

        self.remaining_epsilon -= epsilon;
        self.remaining_delta -= delta;

        self.composition_history.push(PrivacyConsumption {
            epsilon,
            delta,
            timestamp: chrono::Utc::now(),
            operation,
        });

        Ok(())
    }

    /// Check if budget is available
    #[must_use]
    pub fn can_consume(&self, epsilon: f32, delta: f32) -> bool {
        epsilon <= self.remaining_epsilon && delta <= self.remaining_delta
    }

    /// Get privacy loss (advanced composition)
    #[must_use]
    pub fn get_composed_privacy_loss(&self) -> (f32, f32) {
        // Advanced composition theorem
        let n = self.composition_history.len() as f32;

        if n == 0.0 {
            return (0.0, 0.0);
        }

        let total_epsilon: f32 = self.composition_history.iter().map(|c| c.epsilon).sum();
        let total_delta: f32 = self.composition_history.iter().map(|c| c.delta).sum();

        // Advanced composition
        let composed_epsilon = ((2.0 * n * total_epsilon.powi(2)).ln() + total_epsilon).sqrt();
        let composed_delta = n * total_delta;

        (composed_epsilon, composed_delta)
    }
}

/// Differential privacy mechanism trait
pub trait DPMechanism {
    /// Add noise to value
    fn add_noise(&self, value: f32, sensitivity: f32) -> f32;

    /// Add noise to vector
    fn add_noise_vec(&self, values: &[f32], sensitivity: f32) -> Vec<f32>;

    /// Get privacy parameters
    fn privacy_params(&self) -> (f32, f32);
}

/// Gaussian mechanism for (ε, δ)-DP
#[derive(Debug, Clone)]
pub struct GaussianMechanism {
    /// Privacy parameter epsilon
    epsilon: f32,

    /// Privacy parameter delta
    delta: f32,

    /// Standard deviation of Gaussian noise
    sigma: f32,
}

impl GaussianMechanism {
    /// Create new Gaussian mechanism
    pub fn new(epsilon: f32, delta: f32) -> Result<Self, DPError> {
        if epsilon <= 0.0 {
            return Err(DPError::InvalidEpsilon);
        }

        if delta <= 0.0 || delta >= 1.0 {
            return Err(DPError::InvalidDelta);
        }

        // Calculate noise scale using analytic Gaussian mechanism
        // σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        let sigma = (2.0 * (1.25 / delta).ln()).sqrt() / epsilon;

        Ok(Self {
            epsilon,
            delta,
            sigma,
        })
    }

    /// Sample from Gaussian distribution
    fn sample_gaussian(&self, mean: f32, std_dev: f32) -> f32 {
        use scirs2_core::random::Rng;
        let mut rng = thread_rng();

        // Box-Muller transform for Gaussian sampling
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std_dev * z0
    }
}

impl DPMechanism for GaussianMechanism {
    fn add_noise(&self, value: f32, sensitivity: f32) -> f32 {
        let noise_scale = self.sigma * sensitivity;
        value + self.sample_gaussian(0.0, noise_scale)
    }

    fn add_noise_vec(&self, values: &[f32], sensitivity: f32) -> Vec<f32> {
        values
            .iter()
            .map(|&v| self.add_noise(v, sensitivity))
            .collect()
    }

    fn privacy_params(&self) -> (f32, f32) {
        (self.epsilon, self.delta)
    }
}

/// Laplace mechanism for ε-DP
#[derive(Debug, Clone)]
pub struct LaplaceMechanism {
    /// Privacy parameter epsilon
    epsilon: f32,

    /// Scale parameter for Laplace distribution
    scale: f32,
}

impl LaplaceMechanism {
    /// Create new Laplace mechanism
    pub fn new(epsilon: f32) -> Result<Self, DPError> {
        if epsilon <= 0.0 {
            return Err(DPError::InvalidEpsilon);
        }

        Ok(Self {
            epsilon,
            scale: 1.0 / epsilon,
        })
    }

    /// Sample from Laplace distribution
    fn sample_laplace(&self, location: f32, scale: f32) -> f32 {
        use scirs2_core::random::Rng;
        let mut rng = thread_rng();

        let u: f32 = rng.random_range(-0.5..0.5);
        location - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

impl DPMechanism for LaplaceMechanism {
    fn add_noise(&self, value: f32, sensitivity: f32) -> f32 {
        let noise_scale = self.scale * sensitivity;
        value + self.sample_laplace(0.0, noise_scale)
    }

    fn add_noise_vec(&self, values: &[f32], sensitivity: f32) -> Vec<f32> {
        values
            .iter()
            .map(|&v| self.add_noise(v, sensitivity))
            .collect()
    }

    fn privacy_params(&self) -> (f32, f32) {
        (self.epsilon, 0.0)
    }
}

/// Main differential privacy interface
pub struct DifferentialPrivacy {
    /// DP configuration
    config: DPConfig,

    /// Privacy budget tracker
    budget: PrivacyBudget,

    /// DP mechanism
    mechanism: Box<dyn DPMechanism + Send + Sync>,
}

impl DifferentialPrivacy {
    /// Create new differential privacy instance
    pub fn new(config: DPConfig) -> Result<Self, DPError> {
        let budget = PrivacyBudget::new(config.epsilon, config.delta)?;

        let mechanism: Box<dyn DPMechanism + Send + Sync> = match config.noise_distribution {
            NoiseDistribution::Gaussian => {
                Box::new(GaussianMechanism::new(config.epsilon, config.delta)?)
            }
            NoiseDistribution::Laplace => Box::new(LaplaceMechanism::new(config.epsilon)?),
            NoiseDistribution::Exponential => {
                // Fallback to Gaussian for now
                Box::new(GaussianMechanism::new(config.epsilon, config.delta)?)
            }
        };

        Ok(Self {
            config,
            budget,
            mechanism,
        })
    }

    /// Clip gradients to bound sensitivity
    #[must_use]
    pub fn clip_gradients(&self, gradients: &[f32]) -> Vec<f32> {
        let norm: f32 = gradients.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > self.config.clip_norm {
            let scale = self.config.clip_norm / norm;
            gradients.iter().map(|&g| g * scale).collect()
        } else {
            gradients.to_vec()
        }
    }

    /// Add DP noise to gradients
    pub fn privatize_gradients(
        &mut self,
        gradients: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, DPError> {
        // Clip gradients first
        let clipped = self.clip_gradients(gradients);

        // Calculate sensitivity (L2 sensitivity for gradient clipping)
        let sensitivity = 2.0 * self.config.clip_norm / batch_size as f32;

        // Add noise
        let privatized = self.mechanism.add_noise_vec(&clipped, sensitivity);

        // Update privacy budget if accounting is enabled
        if self.config.enable_accounting {
            self.budget.consume(
                self.config.epsilon,
                self.config.delta,
                "gradient_privatization".to_string(),
            )?;
        }

        Ok(privatized)
    }

    /// Add DP noise to model parameters
    pub fn privatize_parameters(
        &mut self,
        parameters: &[f32],
        sensitivity: f32,
    ) -> Result<Vec<f32>, DPError> {
        if sensitivity <= 0.0 {
            return Err(DPError::InvalidSensitivity);
        }

        let privatized = self.mechanism.add_noise_vec(parameters, sensitivity);

        if self.config.enable_accounting {
            self.budget.consume(
                self.config.epsilon,
                self.config.delta,
                "parameter_privatization".to_string(),
            )?;
        }

        Ok(privatized)
    }

    /// Get remaining privacy budget
    #[must_use]
    pub fn remaining_budget(&self) -> (f32, f32) {
        (self.budget.remaining_epsilon, self.budget.remaining_delta)
    }

    /// Get total privacy loss (advanced composition)
    #[must_use]
    pub fn total_privacy_loss(&self) -> (f32, f32) {
        self.budget.get_composed_privacy_loss()
    }

    /// Get consumption history
    #[must_use]
    pub fn consumption_history(&self) -> &[PrivacyConsumption] {
        &self.budget.composition_history
    }
}

/// Moments accountant for tighter privacy composition
pub struct MomentsAccountant {
    /// Log moments
    log_moments: Vec<f32>,

    /// Number of moments to track
    num_moments: usize,

    /// Target delta
    target_delta: f32,
}

impl MomentsAccountant {
    /// Create new moments accountant
    #[must_use]
    pub fn new(num_moments: usize, target_delta: f32) -> Self {
        Self {
            log_moments: vec![0.0; num_moments],
            num_moments,
            target_delta,
        }
    }

    /// Accumulate moments from Gaussian mechanism
    pub fn accumulate_gaussian(&mut self, sigma: f32, sensitivity: f32, steps: usize) {
        for lambda in 1..=self.num_moments {
            let lambda_f = lambda as f32;

            // Compute log of λ-th moment
            let log_moment = steps as f32
                * (lambda_f * (lambda_f + 1.0) * sensitivity.powi(2) / (2.0 * sigma.powi(2)));

            self.log_moments[lambda - 1] += log_moment;
        }
    }

    /// Get privacy guarantee (epsilon) for target delta
    #[must_use]
    pub fn get_epsilon(&self) -> f32 {
        let mut min_epsilon = f32::MAX;

        for lambda in 1..=self.num_moments {
            let epsilon = (self.log_moments[lambda - 1] - self.target_delta.ln()) / lambda as f32;
            if epsilon < min_epsilon {
                min_epsilon = epsilon;
            }
        }

        min_epsilon.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_budget_creation() {
        let budget = PrivacyBudget::new(1.0, 1e-5).unwrap();
        assert_eq!(budget.total_epsilon, 1.0);
        assert_eq!(budget.remaining_epsilon, 1.0);
    }

    #[test]
    fn test_privacy_budget_consumption() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5).unwrap();

        assert!(budget.consume(0.5, 5e-6, "test_op".to_string()).is_ok());
        assert!((budget.remaining_epsilon - 0.5).abs() < 1e-6);
        assert!((budget.remaining_delta - 5e-6).abs() < 1e-9);
    }

    #[test]
    fn test_budget_exceeded() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5).unwrap();

        assert!(budget.consume(0.5, 5e-6, "test_op1".to_string()).is_ok());
        let result = budget.consume(0.6, 5e-6, "test_op2".to_string());

        assert!(result.is_err());
    }

    #[test]
    fn test_gaussian_mechanism() {
        let mechanism = GaussianMechanism::new(1.0, 1e-5).unwrap();
        let value = 10.0;
        let sensitivity = 1.0;

        let noisy_value = mechanism.add_noise(value, sensitivity);

        // Noisy value should be different from original
        assert_ne!(noisy_value, value);

        // Should be roughly within a few standard deviations
        assert!((noisy_value - value).abs() < 10.0 * mechanism.sigma);
    }

    #[test]
    fn test_laplace_mechanism() {
        let mechanism = LaplaceMechanism::new(1.0).unwrap();
        let value = 10.0;
        let sensitivity = 1.0;

        let noisy_value = mechanism.add_noise(value, sensitivity);

        assert_ne!(noisy_value, value);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = DPConfig {
            clip_norm: 1.0,
            ..Default::default()
        };

        let dp = DifferentialPrivacy::new(config).unwrap();

        // Test clipping
        let gradients = vec![1.0, 1.0, 1.0, 1.0]; // Norm = 2.0
        let clipped = dp.clip_gradients(&gradients);

        let clipped_norm: f32 = clipped.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_differential_privacy_integration() {
        let config = DPConfig {
            epsilon: 1.0,
            delta: 1e-5,
            clip_norm: 1.0,
            enable_accounting: true,
            ..Default::default()
        };

        let mut dp = DifferentialPrivacy::new(config).unwrap();

        let gradients = vec![0.5, 0.5, 0.5, 0.5];
        let batch_size = 32;

        let result = dp.privatize_gradients(&gradients, batch_size);
        assert!(result.is_ok());

        let (remaining_eps, remaining_delta) = dp.remaining_budget();
        assert!(remaining_eps < 1.0); // Budget consumed
    }

    #[test]
    fn test_moments_accountant() {
        let mut accountant = MomentsAccountant::new(32, 1e-5);

        // Accumulate moments for 10 steps with sigma=4.0, sensitivity=1.0
        // Using higher sigma for stronger privacy
        accountant.accumulate_gaussian(4.0, 1.0, 10);

        let epsilon = accountant.get_epsilon();
        assert!(epsilon > 0.0);
        assert!(epsilon < 10.0); // Should be reasonable with higher sigma
    }
}
