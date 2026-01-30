//! Retry Logic Module
//!
//! Provides retry strategies with configurable backoff algorithms.

use super::definition::{BackoffType, RetryStrategy};
use serde::{Deserialize, Serialize};

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum attempts
    pub max_attempts: usize,
    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
    /// Backoff multiplier
    pub multiplier: f64,
    /// Backoff strategy
    pub strategy: BackoffStrategy,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 60_000,
            multiplier: 2.0,
            strategy: BackoffStrategy::Exponential,
        }
    }
}

/// Backoff strategy enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackoffStrategy {
    /// Fixed delay
    Fixed,
    /// Linear backoff
    Linear,
    /// Exponential backoff
    Exponential,
    /// Exponential with jitter
    ExponentialJitter,
}

/// Retry manager
pub struct RetryManager;

impl RetryManager {
    /// Create new retry manager
    pub fn new() -> Self {
        Self
    }

    /// Calculate delay for retry attempt
    pub fn calculate_delay(&self, strategy: &RetryStrategy, attempt: usize) -> u64 {
        let backoff_strategy = match strategy.backoff {
            BackoffType::Fixed => BackoffStrategy::Fixed,
            BackoffType::Linear => BackoffStrategy::Linear,
            BackoffType::Exponential => BackoffStrategy::Exponential,
            BackoffType::ExponentialJitter => BackoffStrategy::ExponentialJitter,
        };

        let delay = match backoff_strategy {
            BackoffStrategy::Fixed => strategy.initial_delay_ms,
            BackoffStrategy::Linear => strategy.initial_delay_ms * attempt as u64,
            BackoffStrategy::Exponential => {
                let delay = strategy.initial_delay_ms
                    * (strategy.backoff_multiplier.powi(attempt as i32 - 1) as u64);
                delay.min(strategy.max_delay_ms)
            }
            BackoffStrategy::ExponentialJitter => {
                let base_delay = strategy.initial_delay_ms
                    * (strategy.backoff_multiplier.powi(attempt as i32 - 1) as u64);
                // Simple jitter: add 10% of base delay
                let jitter = (base_delay as f64 * 0.1) as u64;
                (base_delay + jitter).min(strategy.max_delay_ms)
            }
        };

        delay
            .max(strategy.initial_delay_ms)
            .min(strategy.max_delay_ms)
    }

    /// Check if should retry
    pub fn should_retry(&self, attempt: usize, max_attempts: usize) -> bool {
        attempt < max_attempts
    }
}

impl Default for RetryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 60_000);
    }

    #[test]
    fn test_retry_manager_creation() {
        let manager = RetryManager::new();
        // Just verify creation works
        assert!(manager.should_retry(1, 3));
    }

    #[test]
    fn test_should_retry() {
        let manager = RetryManager::new();
        assert!(manager.should_retry(1, 3));
        assert!(manager.should_retry(2, 3));
        assert!(!manager.should_retry(3, 3));
        assert!(!manager.should_retry(4, 3));
    }

    #[test]
    fn test_fixed_backoff() {
        let manager = RetryManager::new();
        let strategy = RetryStrategy {
            max_attempts: 3,
            backoff: BackoffType::Fixed,
            initial_delay_ms: 1000,
            max_delay_ms: 60_000,
            backoff_multiplier: 2.0,
        };

        let delay1 = manager.calculate_delay(&strategy, 1);
        let delay2 = manager.calculate_delay(&strategy, 2);
        let delay3 = manager.calculate_delay(&strategy, 3);

        assert_eq!(delay1, 1000);
        assert_eq!(delay2, 1000);
        assert_eq!(delay3, 1000);
    }

    #[test]
    fn test_linear_backoff() {
        let manager = RetryManager::new();
        let strategy = RetryStrategy {
            max_attempts: 3,
            backoff: BackoffType::Linear,
            initial_delay_ms: 1000,
            max_delay_ms: 60_000,
            backoff_multiplier: 2.0,
        };

        let delay1 = manager.calculate_delay(&strategy, 1);
        let delay2 = manager.calculate_delay(&strategy, 2);
        let delay3 = manager.calculate_delay(&strategy, 3);

        assert_eq!(delay1, 1000);
        assert_eq!(delay2, 2000);
        assert_eq!(delay3, 3000);
    }

    #[test]
    fn test_exponential_backoff() {
        let manager = RetryManager::new();
        let strategy = RetryStrategy {
            max_attempts: 4,
            backoff: BackoffType::Exponential,
            initial_delay_ms: 1000,
            max_delay_ms: 60_000,
            backoff_multiplier: 2.0,
        };

        let delay1 = manager.calculate_delay(&strategy, 1);
        let delay2 = manager.calculate_delay(&strategy, 2);
        let delay3 = manager.calculate_delay(&strategy, 3);

        assert_eq!(delay1, 1000);
        assert_eq!(delay2, 2000);
        assert_eq!(delay3, 4000);
    }

    #[test]
    fn test_max_delay_cap() {
        let manager = RetryManager::new();
        let strategy = RetryStrategy {
            max_attempts: 10,
            backoff: BackoffType::Exponential,
            initial_delay_ms: 1000,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        };

        let delay5 = manager.calculate_delay(&strategy, 5);
        assert!(delay5 <= 5000);
    }
}
