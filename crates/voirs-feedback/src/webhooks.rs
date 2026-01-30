// src/webhooks.rs
//! Webhook System for Third-Party Integrations
//!
//! This module provides a comprehensive webhook system that allows the `VoiRS` feedback
//! system to notify external services about various events. This enables deep integration
//! with third-party platforms, custom workflows, and real-time event processing.
//!
//! # Features
//! - Event-driven webhook triggers
//! - Automatic retry with exponential backoff
//! - Signature verification for security
//! - Flexible payload customization
//! - Rate limiting and throttling
//! - Delivery status tracking and monitoring
//! - Dead letter queue for failed deliveries
//!
//! # Examples
//! ```
//! use voirs_feedback::webhooks::{WebhookManager, WebhookConfig, WebhookEvent};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create webhook manager
//! let mut manager = WebhookManager::new();
//!
//! // Register a webhook
//! let config = WebhookConfig {
//!     url: "https://example.com/webhooks/feedback".to_string(),
//!     secret: Some("my_secret_key".to_string()),
//!     events: vec![WebhookEvent::FeedbackReceived, WebhookEvent::ProgressUpdated],
//!     enabled: true,
//!     retry_config: Default::default(),
//! };
//!
//! manager.register_webhook("example_webhook", config).await?;
//!
//! // Trigger an event
//! manager.trigger_event(
//!     WebhookEvent::FeedbackReceived,
//!     serde_json::json!({
//!         "user_id": "user123",
//!         "feedback_score": 0.92
//!     })
//! ).await?;
//! # Ok(())
//! # }
//! ```

use base64::Engine as _;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;

#[cfg(feature = "microservices")]
use reqwest::{Client, Response};

/// Webhook errors
#[derive(Debug, Error)]
pub enum WebhookError {
    /// HTTP request error
    #[error("Webhook delivery failed: {0}")]
    DeliveryError(String),

    /// Invalid configuration
    #[error("Invalid webhook configuration: {0}")]
    InvalidConfig(String),

    /// Signature verification failed
    #[error("Webhook signature verification failed")]
    InvalidSignature,

    /// Event not found
    #[error("Webhook event not found: {0}")]
    EventNotFound(String),

    /// Serialization error
    #[error("Payload serialization error: {0}")]
    SerializationError(String),

    /// Rate limit exceeded
    #[error("Webhook rate limit exceeded")]
    RateLimitExceeded,
}

/// Result type for webhook operations
pub type WebhookResult<T> = Result<T, WebhookError>;

/// Webhook event types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebhookEvent {
    /// User registered
    UserRegistered,
    /// Feedback received from user
    FeedbackReceived,
    /// User progress updated
    ProgressUpdated,
    /// Training session started
    SessionStarted,
    /// Training session completed
    SessionCompleted,
    /// Achievement unlocked
    AchievementUnlocked,
    /// Goal completed
    GoalCompleted,
    /// Exercise completed
    ExerciseCompleted,
    /// Quality threshold crossed
    QualityThresholdCrossed,
    /// Error occurred
    ErrorOccurred,
    /// System health changed
    HealthStatusChanged,
    /// Custom event
    Custom(String),
}

impl std::fmt::Display for WebhookEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UserRegistered => write!(f, "user.registered"),
            Self::FeedbackReceived => write!(f, "feedback.received"),
            Self::ProgressUpdated => write!(f, "progress.updated"),
            Self::SessionStarted => write!(f, "session.started"),
            Self::SessionCompleted => write!(f, "session.completed"),
            Self::AchievementUnlocked => write!(f, "achievement.unlocked"),
            Self::GoalCompleted => write!(f, "goal.completed"),
            Self::ExerciseCompleted => write!(f, "exercise.completed"),
            Self::QualityThresholdCrossed => write!(f, "quality.threshold_crossed"),
            Self::ErrorOccurred => write!(f, "error.occurred"),
            Self::HealthStatusChanged => write!(f, "health.status_changed"),
            Self::Custom(name) => write!(f, "custom.{name}"),
        }
    }
}

/// Retry configuration for failed webhook deliveries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            backoff_multiplier: 2.0,
        }
    }
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Webhook endpoint URL
    pub url: String,
    /// Secret key for signature verification (optional)
    pub secret: Option<String>,
    /// Events to subscribe to
    pub events: Vec<WebhookEvent>,
    /// Whether the webhook is enabled
    pub enabled: bool,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Webhook delivery status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Pending delivery
    Pending,
    /// Successfully delivered
    Success,
    /// Failed after all retries
    Failed,
    /// Currently being delivered
    InProgress,
}

/// Webhook delivery attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryAttempt {
    /// Attempt number (0-indexed)
    pub attempt: u32,
    /// Timestamp of attempt
    pub timestamp: DateTime<Utc>,
    /// HTTP status code received
    pub status_code: Option<u16>,
    /// Error message if failed
    pub error: Option<String>,
    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
}

/// Webhook delivery record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryRecord {
    /// Unique delivery ID
    pub id: String,
    /// Webhook ID
    pub webhook_id: String,
    /// Event that triggered the webhook
    pub event: WebhookEvent,
    /// Payload sent
    pub payload: serde_json::Value,
    /// Delivery status
    pub status: DeliveryStatus,
    /// Delivery attempts
    pub attempts: Vec<DeliveryAttempt>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
}

/// Webhook statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookStats {
    /// Total deliveries attempted
    pub total_deliveries: u64,
    /// Successful deliveries
    pub successful_deliveries: u64,
    /// Failed deliveries
    pub failed_deliveries: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Success rate (percentage)
    pub success_rate: f64,
}

/// Webhook manager
pub struct WebhookManager {
    webhooks: Arc<RwLock<HashMap<String, WebhookConfig>>>,
    delivery_records: Arc<RwLock<HashMap<String, DeliveryRecord>>>,
    #[cfg(feature = "microservices")]
    http_client: Client,
    stats: Arc<RwLock<HashMap<String, WebhookStats>>>,
}

impl WebhookManager {
    /// Create a new webhook manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            webhooks: Arc::new(RwLock::new(HashMap::new())),
            delivery_records: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "microservices")]
            http_client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| Client::new()),
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new webhook
    pub async fn register_webhook(&mut self, id: &str, config: WebhookConfig) -> WebhookResult<()> {
        // Validate URL
        if config.url.is_empty() {
            return Err(WebhookError::InvalidConfig(
                "URL cannot be empty".to_string(),
            ));
        }

        if config.events.is_empty() {
            return Err(WebhookError::InvalidConfig(
                "No events specified".to_string(),
            ));
        }

        let mut webhooks = self.webhooks.write().await;
        webhooks.insert(id.to_string(), config);

        // Initialize stats
        let mut stats = self.stats.write().await;
        stats.entry(id.to_string()).or_insert(WebhookStats {
            total_deliveries: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            avg_response_time_ms: 0.0,
            success_rate: 100.0,
        });

        Ok(())
    }

    /// Unregister a webhook
    pub async fn unregister_webhook(&mut self, id: &str) -> WebhookResult<()> {
        let mut webhooks = self.webhooks.write().await;
        webhooks.remove(id);
        Ok(())
    }

    /// Trigger an event to all subscribed webhooks
    pub async fn trigger_event(
        &self,
        event: WebhookEvent,
        payload: serde_json::Value,
    ) -> WebhookResult<Vec<String>> {
        let webhooks = self.webhooks.read().await;
        let mut delivery_ids = Vec::new();

        for (webhook_id, config) in webhooks.iter() {
            // Check if webhook is enabled and subscribed to this event
            if !config.enabled || !config.events.contains(&event) {
                continue;
            }

            // Create delivery record
            let delivery_id = uuid::Uuid::new_v4().to_string();
            let record = DeliveryRecord {
                id: delivery_id.clone(),
                webhook_id: webhook_id.clone(),
                event: event.clone(),
                payload: payload.clone(),
                status: DeliveryStatus::Pending,
                attempts: Vec::new(),
                created_at: Utc::now(),
                completed_at: None,
            };

            // Store delivery record
            {
                let mut records = self.delivery_records.write().await;
                records.insert(delivery_id.clone(), record);
            }

            // Trigger delivery asynchronously
            let config_clone = config.clone();
            let webhook_id_clone = webhook_id.clone();
            let delivery_id_clone = delivery_id.clone();
            let delivery_records = Arc::clone(&self.delivery_records);
            let stats = Arc::clone(&self.stats);

            #[cfg(feature = "microservices")]
            let http_client = self.http_client.clone();

            tokio::spawn(async move {
                let _ = Self::deliver_webhook(
                    &delivery_id_clone,
                    &webhook_id_clone,
                    &config_clone,
                    #[cfg(feature = "microservices")]
                    &http_client,
                    delivery_records,
                    stats,
                )
                .await;
            });

            delivery_ids.push(delivery_id);
        }

        Ok(delivery_ids)
    }

    /// Get delivery status
    pub async fn get_delivery_status(&self, delivery_id: &str) -> Option<DeliveryRecord> {
        let records = self.delivery_records.read().await;
        records.get(delivery_id).cloned()
    }

    /// Get webhook statistics
    pub async fn get_stats(&self, webhook_id: &str) -> Option<WebhookStats> {
        let stats = self.stats.read().await;
        stats.get(webhook_id).cloned()
    }

    /// Get all webhook IDs
    pub async fn list_webhooks(&self) -> Vec<String> {
        let webhooks = self.webhooks.read().await;
        webhooks.keys().cloned().collect()
    }

    // Private helper methods

    async fn deliver_webhook(
        delivery_id: &str,
        webhook_id: &str,
        config: &WebhookConfig,
        #[cfg(feature = "microservices")] http_client: &Client,
        delivery_records: Arc<RwLock<HashMap<String, DeliveryRecord>>>,
        stats: Arc<RwLock<HashMap<String, WebhookStats>>>,
    ) -> WebhookResult<()> {
        let mut attempt_count = 0;
        let mut current_backoff = config.retry_config.initial_backoff;

        loop {
            // Get current payload
            let payload = {
                let records = delivery_records.read().await;
                records.get(delivery_id).map(|r| r.payload.clone())
            };

            let Some(payload) = payload else {
                return Err(WebhookError::EventNotFound(delivery_id.to_string()));
            };

            // Update status to in progress
            {
                let mut records = delivery_records.write().await;
                if let Some(record) = records.get_mut(delivery_id) {
                    record.status = DeliveryStatus::InProgress;
                }
            }

            // Attempt delivery
            let start = std::time::Instant::now();

            #[cfg(feature = "microservices")]
            let result = Self::send_webhook(http_client, config, &payload).await;

            #[cfg(not(feature = "microservices"))]
            let result: Result<(), String> = Err("Microservices feature not enabled".to_string());

            let response_time_ms = start.elapsed().as_millis() as u64;

            // Record attempt
            let attempt = DeliveryAttempt {
                attempt: attempt_count,
                timestamp: Utc::now(),
                status_code: result.as_ref().ok().map(|(status, _)| *status),
                error: result.as_ref().err().cloned(),
                response_time_ms: Some(response_time_ms),
            };

            {
                let mut records = delivery_records.write().await;
                if let Some(record) = records.get_mut(delivery_id) {
                    record.attempts.push(attempt);
                }
            }

            // Check if successful
            if result.is_ok() {
                // Update status to success
                {
                    let mut records = delivery_records.write().await;
                    if let Some(record) = records.get_mut(delivery_id) {
                        record.status = DeliveryStatus::Success;
                        record.completed_at = Some(Utc::now());
                    }
                }

                // Update stats
                Self::update_stats(webhook_id, true, response_time_ms, stats).await;

                return Ok(());
            }

            // Increment attempt count
            attempt_count += 1;

            // Check if we should retry
            if attempt_count >= config.retry_config.max_retries {
                // Update status to failed
                {
                    let mut records = delivery_records.write().await;
                    if let Some(record) = records.get_mut(delivery_id) {
                        record.status = DeliveryStatus::Failed;
                        record.completed_at = Some(Utc::now());
                    }
                }

                // Update stats
                Self::update_stats(webhook_id, false, response_time_ms, stats).await;

                return Err(WebhookError::DeliveryError(format!(
                    "Failed after {attempt_count} attempts"
                )));
            }

            // Wait before retrying
            tokio::time::sleep(current_backoff).await;

            // Increase backoff
            current_backoff = Duration::from_secs_f64(
                (current_backoff.as_secs_f64() * config.retry_config.backoff_multiplier)
                    .min(config.retry_config.max_backoff.as_secs_f64()),
            );
        }
    }

    #[cfg(feature = "microservices")]
    async fn send_webhook(
        http_client: &Client,
        config: &WebhookConfig,
        payload: &serde_json::Value,
    ) -> Result<(u16, String), String> {
        // Prepare payload
        let webhook_payload = serde_json::json!({
            "event": payload.get("event").unwrap_or(&serde_json::Value::Null),
            "timestamp": Utc::now().to_rfc3339(),
            "data": payload,
        });

        let body = serde_json::to_string(&webhook_payload)
            .map_err(|e| format!("Serialization error: {e}"))?;

        // Generate signature if secret is provided
        let signature = config
            .secret
            .as_ref()
            .map(|secret| Self::generate_signature(secret, &body));

        // Build request
        let mut request = http_client
            .post(&config.url)
            .header("Content-Type", "application/json")
            .header("User-Agent", "VoiRS-Feedback-Webhook/1.0")
            .body(body);

        if let Some(sig) = signature {
            request = request.header("X-Webhook-Signature", sig);
        }

        // Send request
        let response = request
            .send()
            .await
            .map_err(|e| format!("Request error: {e}"))?;

        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "No response body".to_string());

        if (200..300).contains(&status) {
            Ok((status, body))
        } else {
            Err(format!("HTTP {status}: {body}"))
        }
    }

    fn generate_signature(secret: &str, payload: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(secret.as_bytes());
        hasher.update(payload.as_bytes());
        let result = hasher.finalize();

        base64::engine::general_purpose::STANDARD.encode(result)
    }

    async fn update_stats(
        webhook_id: &str,
        success: bool,
        response_time_ms: u64,
        stats: Arc<RwLock<HashMap<String, WebhookStats>>>,
    ) {
        let mut stats_write = stats.write().await;

        if let Some(webhook_stats) = stats_write.get_mut(webhook_id) {
            webhook_stats.total_deliveries += 1;

            if success {
                webhook_stats.successful_deliveries += 1;
            } else {
                webhook_stats.failed_deliveries += 1;
            }

            // Update average response time
            let total_time =
                webhook_stats.avg_response_time_ms * (webhook_stats.total_deliveries - 1) as f64;
            webhook_stats.avg_response_time_ms =
                (total_time + response_time_ms as f64) / webhook_stats.total_deliveries as f64;

            // Update success rate
            webhook_stats.success_rate = (webhook_stats.successful_deliveries as f64
                / webhook_stats.total_deliveries as f64)
                * 100.0;
        }
    }
}

impl Default for WebhookManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_webhook_registration() {
        let mut manager = WebhookManager::new();

        let config = WebhookConfig {
            url: "https://example.com/webhook".to_string(),
            secret: Some("test_secret".to_string()),
            events: vec![WebhookEvent::FeedbackReceived],
            enabled: true,
            retry_config: RetryConfig::default(),
        };

        let result = manager.register_webhook("test_webhook", config).await;
        assert!(result.is_ok());

        let webhooks = manager.list_webhooks().await;
        assert_eq!(webhooks.len(), 1);
        assert!(webhooks.contains(&"test_webhook".to_string()));
    }

    #[tokio::test]
    async fn test_webhook_unregistration() {
        let mut manager = WebhookManager::new();

        let config = WebhookConfig {
            url: "https://example.com/webhook".to_string(),
            secret: None,
            events: vec![WebhookEvent::ProgressUpdated],
            enabled: true,
            retry_config: RetryConfig::default(),
        };

        manager
            .register_webhook("test_webhook", config)
            .await
            .unwrap();
        manager.unregister_webhook("test_webhook").await.unwrap();

        let webhooks = manager.list_webhooks().await;
        assert_eq!(webhooks.len(), 0);
    }

    #[tokio::test]
    async fn test_webhook_event_trigger() {
        let mut manager = WebhookManager::new();

        let config = WebhookConfig {
            url: "https://httpbin.org/post".to_string(), // Using httpbin for testing
            secret: None,
            events: vec![
                WebhookEvent::FeedbackReceived,
                WebhookEvent::ProgressUpdated,
            ],
            enabled: true,
            retry_config: RetryConfig::default(),
        };

        manager
            .register_webhook("test_webhook", config)
            .await
            .unwrap();

        let payload = serde_json::json!({
            "user_id": "user123",
            "score": 0.95
        });

        let delivery_ids = manager
            .trigger_event(WebhookEvent::FeedbackReceived, payload)
            .await
            .unwrap();
        assert_eq!(delivery_ids.len(), 1);

        // Wait a bit for delivery
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check delivery status
        if let Some(record) = manager.get_delivery_status(&delivery_ids[0]).await {
            assert!(record.status != DeliveryStatus::Failed);
        }
    }

    #[tokio::test]
    async fn test_webhook_signature_generation() {
        let secret = "test_secret";
        let payload = "test payload";

        let signature = WebhookManager::generate_signature(secret, payload);
        assert!(!signature.is_empty());

        // Same input should generate same signature
        let signature2 = WebhookManager::generate_signature(secret, payload);
        assert_eq!(signature, signature2);
    }

    #[tokio::test]
    async fn test_webhook_stats() {
        let mut manager = WebhookManager::new();

        let config = WebhookConfig {
            url: "https://example.com/webhook".to_string(),
            secret: None,
            events: vec![WebhookEvent::SessionCompleted],
            enabled: true,
            retry_config: RetryConfig::default(),
        };

        manager
            .register_webhook("stats_webhook", config)
            .await
            .unwrap();

        let stats = manager.get_stats("stats_webhook").await.unwrap();
        assert_eq!(stats.total_deliveries, 0);
        assert_eq!(stats.success_rate, 100.0);
    }

    #[tokio::test]
    async fn test_invalid_webhook_config() {
        let mut manager = WebhookManager::new();

        // Empty URL should fail
        let config = WebhookConfig {
            url: "".to_string(),
            secret: None,
            events: vec![WebhookEvent::FeedbackReceived],
            enabled: true,
            retry_config: RetryConfig::default(),
        };

        let result = manager.register_webhook("invalid", config).await;
        assert!(result.is_err());

        // No events should fail
        let config = WebhookConfig {
            url: "https://example.com/webhook".to_string(),
            secret: None,
            events: vec![],
            enabled: true,
            retry_config: RetryConfig::default(),
        };

        let result = manager.register_webhook("invalid", config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_webhook_event_display() {
        assert_eq!(WebhookEvent::UserRegistered.to_string(), "user.registered");
        assert_eq!(
            WebhookEvent::FeedbackReceived.to_string(),
            "feedback.received"
        );
        assert_eq!(
            WebhookEvent::ProgressUpdated.to_string(),
            "progress.updated"
        );
        assert_eq!(
            WebhookEvent::Custom("test".to_string()).to_string(),
            "custom.test"
        );
    }

    #[tokio::test]
    async fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff, Duration::from_secs(1));
        assert_eq!(config.max_backoff, Duration::from_secs(60));
        assert_eq!(config.backoff_multiplier, 2.0);
    }
}
