//! # Integration and Platform Support
//!
//! This module provides seamless integration capabilities for the VoiRS feedback system,
//! including cross-platform compatibility, API frameworks, and ecosystem synchronization.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub mod api;
pub mod browser_extensions;
pub mod ecosystem;
#[cfg(feature = "graphql")]
pub mod graphql;
pub mod lms;
pub mod platform;
pub mod video_conferencing;

// Re-export main types from submodules
pub use api::*;
pub use browser_extensions::*;

// Re-export from ecosystem (includes ConfigValue)
pub use ecosystem::{
    ConfigValue, EcosystemConfig, EcosystemIntegration, EcosystemIntegrationBuilder,
    PerformanceMetrics, SharedState, SyncManager, SyncStatistics, SyncTask, SyncTaskStatus,
    SyncTaskType,
};

pub use lms::*;

// Re-export from platform (exclude ConfigValue to avoid ambiguity - use ecosystem::ConfigValue)
pub use platform::{
    ConflictResolution, CrossPlatformSync, OfflineDataEntry, OfflineStorage, OfflineStorageStats,
    Platform, PlatformCapabilities, PlatformConfig, PlatformFeature, PlatformManager,
    ResourceConstraints, SyncOperation, SyncOperationType, SyncableData,
};

pub use video_conferencing::*;

/// Integration error types
#[derive(Error, Debug)]
pub enum IntegrationError {
    /// API authentication failed
    #[error("Authentication failed: {message}")]
    AuthenticationError {
        /// Error message
        message: String,
    },

    /// API rate limit exceeded
    #[error("Rate limit exceeded: {limit} requests per {window}")]
    RateLimitError {
        /// Request limit
        limit: u32,
        /// Time window
        window: String,
    },

    /// Invalid API request
    #[error("Invalid request: {message}")]
    InvalidRequest {
        /// Error message
        message: String,
    },

    /// External service error
    #[error("External service error: {service} - {message}")]
    ExternalServiceError {
        /// Service name
        service: String,
        /// Error message
        message: String,
    },

    /// Webhook delivery failed
    #[error("Webhook delivery failed: {url} - {message}")]
    WebhookError {
        /// Webhook URL
        url: String,
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Error message
        message: String,
    },

    /// Serialization error
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message
        message: String,
    },
}

/// Result type for integration operations
pub type IntegrationResult<T> = Result<T, IntegrationError>;

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Base URL for API
    pub base_url: String,
    /// API version
    pub api_version: String,
    /// Authentication configuration
    pub auth: AuthConfig,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Webhook configuration
    pub webhooks: WebhookConfig,
    /// Enable CORS
    pub enable_cors: bool,
    /// Allowed origins for CORS
    pub cors_origins: Vec<String>,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Enable request logging
    pub enable_request_logging: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".to_string(),
            api_version: "v1".to_string(),
            auth: AuthConfig {
                auth_type: AuthType::ApiKey,
                api_key: None,
                oauth_token: None,
                jwt_token: None,
                basic_auth: None,
                custom_headers: HashMap::new(),
            },
            rate_limit: RateLimitConfig::default(),
            webhooks: WebhookConfig::default(),
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            request_timeout_seconds: 30,
            enable_request_logging: true,
        }
    }
}

/// API authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// API key
    pub api_key: Option<String>,
    /// OAuth token
    pub oauth_token: Option<String>,
    /// JWT token
    pub jwt_token: Option<String>,
    /// Basic auth credentials
    pub basic_auth: Option<BasicAuth>,
    /// Custom headers
    pub custom_headers: HashMap<String, String>,
}

/// Authentication types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthType {
    /// No authentication
    None,
    /// API key authentication
    ApiKey,
    /// OAuth 2.0
    OAuth,
    /// JWT token
    Jwt,
    /// Basic authentication
    Basic,
    /// Custom authentication
    Custom,
}

/// Basic authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicAuth {
    /// Username
    pub username: String,
    /// Password
    pub password: String,
}

/// API rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Time window in seconds
    pub window_seconds: u32,
    /// Burst allowance
    pub burst_allowance: u32,
    /// Enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            window_seconds: 3600, // 1 hour
            burst_allowance: 100,
            enabled: true,
        }
    }
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Enable webhooks
    pub enabled: bool,
    /// Webhook endpoints
    pub endpoints: Vec<WebhookEndpoint>,
    /// Secret for webhook signing
    pub secret: Option<String>,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Webhook timeout in seconds
    pub timeout_seconds: u64,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoints: Vec::new(),
            secret: None,
            retry_config: RetryConfig::default(),
            timeout_seconds: 10,
        }
    }
}

/// Webhook endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEndpoint {
    /// Endpoint URL
    pub url: String,
    /// Events to listen for
    pub events: Vec<WebhookEvent>,
    /// HTTP method
    pub method: HttpMethod,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Enable this endpoint
    pub enabled: bool,
}

/// Webhook events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WebhookEvent {
    /// User session started
    SessionStarted,
    /// User session ended
    SessionEnded,
    /// Feedback generated
    FeedbackGenerated,
    /// User progress updated
    ProgressUpdated,
    /// Achievement unlocked
    AchievementUnlocked,
    /// Exercise completed
    ExerciseCompleted,
    /// User preferences updated
    PreferencesUpdated,
    /// System health alert
    HealthAlert,
}

/// HTTP methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HttpMethod {
    /// GET method
    Get,
    /// POST method
    Post,
    /// PUT method
    Put,
    /// PATCH method
    Patch,
    /// DELETE method
    Delete,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable exponential backoff
    pub exponential_backoff: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_multiplier: 2.0,
            exponential_backoff: true,
        }
    }
}
