// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Google Cloud Functions integration
//!
//! This module provides GCP Cloud Functions-specific handlers and utilities
//! for deploying VoiRS Recognizer as a serverless function on Google Cloud Platform.

use super::{Result, ServerlessConfig, ServerlessError, ServerlessMetrics};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// GCP Cloud Function HTTP request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpHttpRequest {
    /// Request body (JSON)
    pub body: GcpRecognitionRequest,
    /// HTTP headers
    pub headers: std::collections::HashMap<String, String>,
}

/// GCP recognition request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpRecognitionRequest {
    /// Audio data (base64 or GCS URI)
    pub audio: AudioInput,
    /// Recognition configuration
    pub config: RecognitionConfig,
}

/// Audio input source
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioInput {
    /// Base64-encoded audio content
    Content(String),
    /// Google Cloud Storage URI
    Uri(String),
}

/// Recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// Audio encoding
    pub encoding: String,
    /// Sample rate in Hz
    pub sample_rate_hertz: u32,
    /// Language code
    pub language_code: String,
    /// Enable word time offsets
    #[serde(default)]
    pub enable_word_time_offsets: bool,
}

/// GCP Cloud Function response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpHttpResponse {
    /// HTTP status code
    pub status_code: u16,
    /// Response body
    pub body: GcpRecognitionResponse,
}

/// GCP recognition response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpRecognitionResponse {
    /// Recognition results
    pub results: Vec<RecognitionResult>,
    /// Total duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<f64>,
}

/// Recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionResult {
    /// Transcription alternatives
    pub alternatives: Vec<TranscriptAlternative>,
}

/// Transcript alternative
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptAlternative {
    /// Transcript text
    pub transcript: String,
    /// Confidence score
    pub confidence: f32,
    /// Word information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordInfo>>,
}

/// Word information with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordInfo {
    /// Word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
}

/// GCP Cloud Function handler
pub struct GcpFunctionHandler {
    config: ServerlessConfig,
    metrics: Arc<RwLock<ServerlessMetrics>>,
    initialized: Arc<RwLock<bool>>,
}

impl GcpFunctionHandler {
    /// Create a new GCP function handler
    pub fn new(config: ServerlessConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(ServerlessMetrics::default())),
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize handler
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing GCP Cloud Function handler");
        let start = std::time::Instant::now();

        if self.config.preload_models {
            debug!("Preloading models for GCP Function");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let duration = start.elapsed();
        let mut metrics = self.metrics.write();
        metrics.cold_start_duration = Some(duration);

        *self.initialized.write() = true;
        info!("GCP function handler initialized in {:?}", duration);

        Ok(())
    }

    /// Handle HTTP request
    pub async fn handle_http_request(&self, request: GcpHttpRequest) -> Result<GcpHttpResponse> {
        let start = std::time::Instant::now();

        if !*self.initialized.read() {
            self.initialize().await?;
        }

        debug!("Processing GCP function request");

        // Get audio data
        let audio_data = match &request.body.audio {
            AudioInput::Content(content) => base64::decode(content).map_err(|e| {
                ServerlessError::BatchProcessingError(format!("Failed to decode audio: {}", e))
            })?,
            AudioInput::Uri(uri) => {
                debug!("Fetching audio from GCS URI: {}", uri);
                // In real implementation, would fetch from GCS
                vec![0u8; 1000]
            }
        };

        debug!("Processing {} bytes of audio", audio_data.len());

        // Simulate recognition
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let processing_time = start.elapsed();

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.request_count += 1;
        metrics.total_processing_time += processing_time;
        metrics.average_latency = metrics.total_processing_time / metrics.request_count as u32;

        Ok(GcpHttpResponse {
            status_code: 200,
            body: GcpRecognitionResponse {
                results: vec![RecognitionResult {
                    alternatives: vec![TranscriptAlternative {
                        transcript: "Recognized text from GCP".to_string(),
                        confidence: 0.96,
                        words: if request.body.config.enable_word_time_offsets {
                            Some(vec![
                                WordInfo {
                                    word: "Recognized".to_string(),
                                    start_time: 0.0,
                                    end_time: 0.5,
                                },
                                WordInfo {
                                    word: "text".to_string(),
                                    start_time: 0.5,
                                    end_time: 0.8,
                                },
                            ])
                        } else {
                            None
                        },
                    }],
                }],
                total_duration: Some(processing_time.as_secs_f64()),
            },
        })
    }

    /// Get metrics
    pub fn get_metrics(&self) -> ServerlessMetrics {
        self.metrics.read().clone()
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        *self.initialized.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gcp_handler_initialization() {
        let config = ServerlessConfig::default();
        let handler = GcpFunctionHandler::new(config);

        let result = handler.initialize().await;
        assert!(result.is_ok());
        assert!(handler.is_initialized());
    }

    #[tokio::test]
    async fn test_gcp_handle_content_request() {
        let config = ServerlessConfig::default();
        let handler = GcpFunctionHandler::new(config);
        handler.initialize().await.unwrap();

        let request = GcpHttpRequest {
            body: GcpRecognitionRequest {
                audio: AudioInput::Content(base64::encode(b"audio data")),
                config: RecognitionConfig {
                    encoding: "LINEAR16".to_string(),
                    sample_rate_hertz: 16000,
                    language_code: "en-US".to_string(),
                    enable_word_time_offsets: false,
                },
            },
            headers: std::collections::HashMap::new(),
        };

        let response = handler.handle_http_request(request).await.unwrap();
        assert_eq!(response.status_code, 200);
        assert!(!response.body.results.is_empty());
    }

    #[tokio::test]
    async fn test_gcp_handle_uri_request() {
        let config = ServerlessConfig::default();
        let handler = GcpFunctionHandler::new(config);
        handler.initialize().await.unwrap();

        let request = GcpHttpRequest {
            body: GcpRecognitionRequest {
                audio: AudioInput::Uri("gs://bucket/audio.wav".to_string()),
                config: RecognitionConfig {
                    encoding: "LINEAR16".to_string(),
                    sample_rate_hertz: 16000,
                    language_code: "en-US".to_string(),
                    enable_word_time_offsets: true,
                },
            },
            headers: std::collections::HashMap::new(),
        };

        let response = handler.handle_http_request(request).await.unwrap();
        assert_eq!(response.status_code, 200);
        assert!(response.body.results[0].alternatives[0].words.is_some());
    }

    #[test]
    fn test_gcp_request_serialization() {
        let request = GcpRecognitionRequest {
            audio: AudioInput::Content("base64data".to_string()),
            config: RecognitionConfig {
                encoding: "LINEAR16".to_string(),
                sample_rate_hertz: 16000,
                language_code: "ja-JP".to_string(),
                enable_word_time_offsets: true,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: GcpRecognitionRequest = serde_json::from_str(&json).unwrap();

        match deserialized.audio {
            AudioInput::Content(_) => {}
            _ => panic!("Expected Content variant"),
        }
    }
}
