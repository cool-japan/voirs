//! API Standards and Guidelines for voirs-cloning
//!
//! This module defines the standardized API patterns that all modules should follow
//! for consistency across the codebase.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Standard API patterns that all modules should implement
pub trait StandardApiPattern {
    /// Configuration type for this component
    type Config: Clone + Default + Serialize + for<'de> Deserialize<'de>;

    /// Builder type for this component (optional)
    type Builder: Default + Clone;

    /// Create new instance with default configuration
    /// Should return Result<Self> if initialization can fail
    fn new() -> Result<Self>
    where
        Self: Sized;

    /// Create new instance with custom configuration
    /// Should always return Result<Self> for consistent error handling
    fn with_config(config: Self::Config) -> Result<Self>
    where
        Self: Sized;

    /// Get builder for this component (if applicable)
    fn builder() -> Self::Builder
    where
        Self: Sized;

    /// Get current configuration
    fn get_config(&self) -> &Self::Config;

    /// Update configuration
    /// Should validate configuration and return error if invalid
    fn update_config(&mut self, config: Self::Config) -> Result<()>;
}

/// Standard builder pattern that all builders should implement
pub trait StandardBuilderPattern<T> {
    /// Create new builder with default settings
    fn new() -> Self;

    /// Build the final instance
    /// Should always return Result<T> for consistent error handling
    fn build(self) -> Result<T>;

    /// Reset builder to default state
    fn reset(&mut self) -> &mut Self;

    /// Validate current builder state
    fn validate(&self) -> Result<()>;
}

/// Standard configuration trait that all config types should implement
pub trait StandardConfig: Clone + Default + Serialize + for<'de> Deserialize<'de> {
    /// Validate configuration parameters
    fn validate(&self) -> Result<()>;

    /// Get configuration name/identifier
    fn name(&self) -> &'static str;

    /// Get configuration version for compatibility checking
    fn version(&self) -> &'static str {
        "1.0.0"
    }

    /// Merge with another configuration, taking non-default values
    fn merge_with(&mut self, other: &Self) -> Result<()>;
}

/// Standard async operation trait for components that perform async operations
#[async_trait::async_trait]
pub trait StandardAsyncOperations {
    /// Initialize component asynchronously
    async fn initialize(&mut self) -> Result<()>;

    /// Cleanup resources
    async fn cleanup(&mut self) -> Result<()>;

    /// Health check for the component
    async fn health_check(&self) -> Result<ComponentHealth>;
}

/// Component health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub is_healthy: bool,
    pub status_message: String,
    pub last_check: std::time::SystemTime,
    pub performance_metrics: Option<std::collections::HashMap<String, f64>>,
}

impl ComponentHealth {
    pub fn healthy(message: &str) -> Self {
        Self {
            is_healthy: true,
            status_message: message.to_string(),
            last_check: std::time::SystemTime::now(),
            performance_metrics: None,
        }
    }

    pub fn unhealthy(message: &str) -> Self {
        Self {
            is_healthy: false,
            status_message: message.to_string(),
            last_check: std::time::SystemTime::now(),
            performance_metrics: None,
        }
    }

    pub fn with_metrics(mut self, metrics: std::collections::HashMap<String, f64>) -> Self {
        self.performance_metrics = Some(metrics);
        self
    }
}

/// Standard error handling patterns
pub mod error_patterns {
    use crate::{Error, Result};

    /// Validate required fields are not empty/default
    pub fn validate_required_string(field_name: &str, value: &str) -> Result<()> {
        if value.trim().is_empty() {
            return Err(Error::Validation(format!("{} cannot be empty", field_name)));
        }
        Ok(())
    }

    /// Validate numeric range
    pub fn validate_range<T: PartialOrd + std::fmt::Display>(
        field_name: &str,
        value: T,
        min: T,
        max: T,
    ) -> Result<()> {
        if value < min || value > max {
            return Err(Error::Validation(format!(
                "{} must be between {} and {}, got {}",
                field_name, min, max, value
            )));
        }
        Ok(())
    }

    /// Validate positive number
    pub fn validate_positive<T: PartialOrd + Default + std::fmt::Display>(
        field_name: &str,
        value: T,
    ) -> Result<()> {
        if value <= T::default() {
            return Err(Error::Validation(format!(
                "{} must be positive, got {}",
                field_name, value
            )));
        }
        Ok(())
    }
}

pub mod naming_conventions {
    //! Standard method naming conventions for methods
    //!
    //! ## Constructor Methods:
    //! - `new()` - Create with default configuration (returns Result<Self>)
    //! - `with_config(config)` - Create with custom configuration (returns Result<Self>)
    //! - `builder()` - Get builder instance (returns Builder)
    //!
    //! ## Configuration Methods:
    //! - `get_config()` - Get current configuration
    //! - `update_config(config)` - Update configuration
    //! - `validate_config(config)` - Validate configuration
    //!
    //! ## Processing Methods:
    //! - `process()` / `process_async()` - Main processing operation
    //! - `extract()` / `extract_async()` - Extract data/features
    //! - `analyze()` / `analyze_async()` - Analyze input
    //! - `generate()` / `generate_async()` - Generate output
    //!
    //! ## State Management:
    //! - `initialize()` / `initialize_async()` - Initialize component
    //! - `cleanup()` / `cleanup_async()` - Cleanup resources
    //! - `reset()` - Reset to initial state
    //! - `start()` / `stop()` - Start/stop operations
    //!
    //! ## Information Methods:
    //! - `get_stats()` - Get statistics
    //! - `get_status()` - Get current status
    //! - `health_check()` - Check component health
    //! - `get_metrics()` - Get performance metrics
    //!
    //! ## Boolean Check Methods:
    //! - `is_*()` - Boolean status checks
    //! - `can_*()` - Capability checks
    //! - `has_*()` - Existence checks
    //! - `supports_*()` - Feature support checks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_component_health_creation() {
        let healthy = ComponentHealth::healthy("All systems operational");
        assert!(healthy.is_healthy);
        assert_eq!(healthy.status_message, "All systems operational");

        let unhealthy = ComponentHealth::unhealthy("System error");
        assert!(!unhealthy.is_healthy);
        assert_eq!(unhealthy.status_message, "System error");
    }

    #[test]
    fn test_component_health_with_metrics() {
        let mut metrics = HashMap::new();
        metrics.insert("cpu_usage".to_string(), 45.5);
        metrics.insert("memory_usage".to_string(), 78.2);

        let health = ComponentHealth::healthy("Running well").with_metrics(metrics.clone());

        assert!(health.performance_metrics.is_some());
        assert_eq!(health.performance_metrics.unwrap(), metrics);
    }

    #[test]
    fn test_error_validation_patterns() {
        use super::error_patterns::*;

        // Test required string validation
        assert!(validate_required_string("name", "valid_name").is_ok());
        assert!(validate_required_string("name", "").is_err());
        assert!(validate_required_string("name", "   ").is_err());

        // Test range validation
        assert!(validate_range("score", 5, 0, 10).is_ok());
        assert!(validate_range("score", -1, 0, 10).is_err());
        assert!(validate_range("score", 11, 0, 10).is_err());

        // Test positive validation
        assert!(validate_positive("count", 5).is_ok());
        assert!(validate_positive("count", 0).is_err());
        assert!(validate_positive("count", -1).is_err());
    }
}
