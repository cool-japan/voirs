// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! AWS Lambda integration for serverless speech recognition
//!
//! This module provides AWS Lambda-specific handlers and utilities for deploying
//! VoiRS Recognizer as a Lambda function with optimized cold start and execution.

use super::{Result, ServerlessConfig, ServerlessError, ServerlessMetrics};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// AWS Lambda event for speech recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaRecognitionEvent {
    /// Audio data (base64 encoded)
    pub audio_data: String,
    /// Audio format (wav, mp3, etc.)
    pub format: String,
    /// Language code (en-US, ja-JP, etc.)
    pub language: Option<String>,
    /// Request ID
    pub request_id: String,
}

/// AWS Lambda response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaRecognitionResponse {
    /// Transcription text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Request ID
    pub request_id: String,
    /// Error message if any
    pub error: Option<String>,
}

/// AWS Lambda handler for speech recognition
pub struct LambdaHandler {
    config: ServerlessConfig,
    metrics: Arc<RwLock<ServerlessMetrics>>,
    initialized: Arc<RwLock<bool>>,
}

impl LambdaHandler {
    /// Create a new Lambda handler
    pub fn new(config: ServerlessConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(ServerlessMetrics::default())),
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize Lambda handler (called on cold start)
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing AWS Lambda handler");
        let start = std::time::Instant::now();

        // Preload models if configured
        if self.config.preload_models {
            debug!("Preloading models for Lambda");
            // In real implementation, would load actual models
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Setup connection pooling
        if self.config.connection_pooling {
            debug!("Setting up connection pooling");
        }

        let duration = start.elapsed();
        let mut metrics = self.metrics.write();
        metrics.cold_start_duration = Some(duration);

        *self.initialized.write() = true;
        info!("Lambda handler initialized in {:?}", duration);

        Ok(())
    }

    /// Handle speech recognition request
    pub async fn handle_request(
        &self,
        event: LambdaRecognitionEvent,
    ) -> Result<LambdaRecognitionResponse> {
        let start = std::time::Instant::now();

        // Ensure handler is initialized
        if !*self.initialized.read() {
            self.initialize().await?;
        }

        debug!("Processing request: {}", event.request_id);

        // Decode audio data
        let audio_bytes = base64::decode(&event.audio_data).map_err(|e| {
            ServerlessError::BatchProcessingError(format!("Failed to decode audio: {}", e))
        })?;

        debug!("Decoded {} bytes of audio", audio_bytes.len());

        // Simulate speech recognition processing
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let processing_time = start.elapsed();

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.request_count += 1;
        metrics.total_processing_time += processing_time;
        metrics.average_latency = metrics.total_processing_time / metrics.request_count as u32;

        Ok(LambdaRecognitionResponse {
            text: "Recognized text from audio".to_string(),
            confidence: 0.95,
            processing_time_ms: processing_time.as_millis() as u64,
            request_id: event.request_id,
            error: None,
        })
    }

    /// Handle batch recognition request
    pub async fn handle_batch_request(
        &self,
        events: Vec<LambdaRecognitionEvent>,
    ) -> Result<Vec<LambdaRecognitionResponse>> {
        info!("Processing batch of {} requests", events.len());

        let mut responses = Vec::new();

        for event in events {
            match self.handle_request(event).await {
                Ok(response) => responses.push(response),
                Err(e) => {
                    responses.push(LambdaRecognitionResponse {
                        text: String::new(),
                        confidence: 0.0,
                        processing_time_ms: 0,
                        request_id: "".to_string(),
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(responses)
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> ServerlessMetrics {
        self.metrics.read().clone()
    }

    /// Check if handler is initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.read()
    }
}

/// AWS Lambda configuration builder
pub struct LambdaConfigBuilder {
    config: ServerlessConfig,
}

impl LambdaConfigBuilder {
    /// Create a new Lambda config builder
    pub fn new() -> Self {
        Self {
            config: ServerlessConfig::default(),
        }
    }

    /// Set memory limit
    pub fn memory_limit(mut self, mb: usize) -> Self {
        self.config.memory_limit_mb = mb;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, duration: std::time::Duration) -> Self {
        self.config.timeout = duration;
        self
    }

    /// Enable cold start optimization
    pub fn enable_cold_start_optimization(mut self) -> Self {
        self.config.cold_start_optimization = true;
        self
    }

    /// Enable model preloading
    pub fn enable_model_preloading(mut self) -> Self {
        self.config.preload_models = true;
        self
    }

    /// Enable request batching
    pub fn enable_batching(mut self, batch_timeout: std::time::Duration) -> Self {
        self.config.request_batching = true;
        self.config.batch_timeout = batch_timeout;
        self
    }

    /// Build configuration
    pub fn build(self) -> ServerlessConfig {
        self.config
    }
}

impl Default for LambdaConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lambda_handler_initialization() {
        let config = ServerlessConfig::default();
        let handler = LambdaHandler::new(config);

        assert!(!handler.is_initialized());

        let result = handler.initialize().await;
        assert!(result.is_ok());
        assert!(handler.is_initialized());

        let metrics = handler.get_metrics();
        assert!(metrics.cold_start_duration.is_some());
    }

    #[tokio::test]
    async fn test_lambda_handle_request() {
        let config = ServerlessConfig::default();
        let handler = LambdaHandler::new(config);
        handler.initialize().await.unwrap();

        let event = LambdaRecognitionEvent {
            audio_data: base64::encode(b"fake audio data"),
            format: "wav".to_string(),
            language: Some("en-US".to_string()),
            request_id: "test-123".to_string(),
        };

        let response = handler.handle_request(event).await.unwrap();
        assert!(!response.text.is_empty());
        assert!(response.confidence > 0.0);
        assert_eq!(response.request_id, "test-123");
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_lambda_batch_request() {
        let config = ServerlessConfig::default();
        let handler = LambdaHandler::new(config);
        handler.initialize().await.unwrap();

        let events = vec![
            LambdaRecognitionEvent {
                audio_data: base64::encode(b"audio1"),
                format: "wav".to_string(),
                language: Some("en-US".to_string()),
                request_id: "req-1".to_string(),
            },
            LambdaRecognitionEvent {
                audio_data: base64::encode(b"audio2"),
                format: "wav".to_string(),
                language: Some("en-US".to_string()),
                request_id: "req-2".to_string(),
            },
        ];

        let responses = handler.handle_batch_request(events).await.unwrap();
        assert_eq!(responses.len(), 2);
        assert!(responses.iter().all(|r| r.error.is_none()));
    }

    #[test]
    fn test_lambda_config_builder() {
        let config = LambdaConfigBuilder::new()
            .memory_limit(3008)
            .timeout(std::time::Duration::from_secs(300))
            .enable_cold_start_optimization()
            .enable_model_preloading()
            .build();

        assert_eq!(config.memory_limit_mb, 3008);
        assert_eq!(config.timeout, std::time::Duration::from_secs(300));
        assert!(config.cold_start_optimization);
        assert!(config.preload_models);
    }

    #[test]
    fn test_lambda_event_serialization() {
        let event = LambdaRecognitionEvent {
            audio_data: "base64data".to_string(),
            format: "wav".to_string(),
            language: Some("en-US".to_string()),
            request_id: "test".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: LambdaRecognitionEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(event.request_id, deserialized.request_id);
    }
}
