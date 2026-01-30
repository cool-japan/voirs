// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Serverless and edge computing deployment support
//!
//! This module provides infrastructure for deploying VoiRS Recognizer on serverless
//! platforms (AWS Lambda, Google Cloud Functions, Azure Functions) and edge computing
//! environments. Features include:
//!
//! - Cold start optimization with model preloading
//! - Memory-efficient model deployment
//! - Request batching and connection pooling
//! - Platform-specific adaptors and handlers
//! - Edge-optimized model formats
//! - Automatic scaling support

pub mod aws_lambda;
pub mod azure_functions;
pub mod cold_start;
pub mod edge_optimization;
pub mod gcp_functions;

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Serverless deployment platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerlessPlatform {
    /// AWS Lambda
    AwsLambda,
    /// Google Cloud Functions
    GoogleCloudFunctions,
    /// Azure Functions
    AzureFunctions,
    /// Generic edge computing environment
    GenericEdge,
    /// Cloudflare Workers
    CloudflareWorkers,
    /// Vercel Edge Functions
    VercelEdge,
}

/// Serverless configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessConfig {
    /// Deployment platform
    pub platform: ServerlessPlatform,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Timeout duration
    pub timeout: Duration,
    /// Cold start optimization enabled
    pub cold_start_optimization: bool,
    /// Model preloading enabled
    pub preload_models: bool,
    /// Connection pooling enabled
    pub connection_pooling: bool,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Request batching enabled
    pub request_batching: bool,
    /// Batch timeout
    pub batch_timeout: Duration,
}

impl Default for ServerlessConfig {
    fn default() -> Self {
        Self {
            platform: ServerlessPlatform::GenericEdge,
            memory_limit_mb: 2048,
            timeout: Duration::from_secs(300),
            cold_start_optimization: true,
            preload_models: true,
            connection_pooling: true,
            max_concurrent_requests: 10,
            request_batching: false,
            batch_timeout: Duration::from_millis(100),
        }
    }
}

/// Serverless deployment error
#[derive(Debug, Error)]
pub enum ServerlessError {
    /// Cold start timeout
    #[error("Cold start timeout: {0}")]
    ColdStartTimeout(String),
    /// Memory limit exceeded
    #[error("Memory limit exceeded: used {used}MB, limit {limit}MB")]
    MemoryLimitExceeded {
        /// Used memory in MB
        used: usize,
        /// Memory limit in MB
        limit: usize,
    },
    /// Model loading failed
    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),
    /// Platform initialization error
    #[error("Platform initialization error: {0}")]
    PlatformInitError(String),
    /// Request timeout
    #[error("Request timeout after {0:?}")]
    RequestTimeout(Duration),
    /// Batch processing error
    #[error("Batch processing error: {0}")]
    BatchProcessingError(String),
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Serverless result type
pub type Result<T> = std::result::Result<T, ServerlessError>;

/// Serverless metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessMetrics {
    /// Cold start duration
    pub cold_start_duration: Option<Duration>,
    /// Warm start duration
    pub warm_start_duration: Duration,
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// Request count
    pub request_count: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average latency
    pub average_latency: Duration,
}

impl Default for ServerlessMetrics {
    fn default() -> Self {
        Self {
            cold_start_duration: None,
            warm_start_duration: Duration::from_secs(0),
            memory_usage_mb: 0,
            request_count: 0,
            total_processing_time: Duration::from_secs(0),
            average_latency: Duration::from_secs(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serverless_config_default() {
        let config = ServerlessConfig::default();
        assert_eq!(config.platform, ServerlessPlatform::GenericEdge);
        assert_eq!(config.memory_limit_mb, 2048);
        assert!(config.cold_start_optimization);
        assert!(config.preload_models);
    }

    #[test]
    fn test_serverless_platform_equality() {
        assert_eq!(ServerlessPlatform::AwsLambda, ServerlessPlatform::AwsLambda);
        assert_ne!(
            ServerlessPlatform::AwsLambda,
            ServerlessPlatform::GoogleCloudFunctions
        );
    }

    #[test]
    fn test_serverless_metrics_default() {
        let metrics = ServerlessMetrics::default();
        assert!(metrics.cold_start_duration.is_none());
        assert_eq!(metrics.request_count, 0);
        assert_eq!(metrics.memory_usage_mb, 0);
    }
}
