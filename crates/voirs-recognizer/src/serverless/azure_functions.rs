// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Azure Functions integration
//!
//! This module provides Azure Functions-specific handlers and utilities
//! for deploying VoiRS Recognizer as a serverless function on Microsoft Azure.

use super::{Result, ServerlessConfig, ServerlessError, ServerlessMetrics};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// Azure Function HTTP request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureHttpRequest {
    /// HTTP method
    pub method: String,
    /// Request URL
    pub url: String,
    /// Request headers
    pub headers: std::collections::HashMap<String, String>,
    /// Request body
    pub body: Option<AzureRecognitionRequest>,
}

/// Azure recognition request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureRecognitionRequest {
    /// Audio configuration
    pub audio_config: AudioConfig,
    /// Speech configuration
    pub speech_config: SpeechConfig,
}

/// Audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Audio source (base64 or Azure Blob URI)
    pub source: AudioSource,
    /// Audio format
    pub format: AudioFormat,
}

/// Audio source
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSource {
    /// Base64-encoded audio
    Base64(String),
    /// Azure Blob Storage URI
    BlobUri(String),
}

/// Audio format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Encoding format (pcm, opus, etc.)
    pub encoding: String,
}

/// Speech configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    /// Language identifier
    pub language: String,
    /// Enable detailed results
    #[serde(default)]
    pub enable_detailed_results: bool,
    /// Enable word-level timing
    #[serde(default)]
    pub enable_word_timing: bool,
    /// Profanity filter level
    #[serde(default)]
    pub profanity_filter: ProfanityFilter,
}

/// Profanity filter level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfanityFilter {
    /// No filtering
    None,
    /// Mask profanity with asterisks
    Masked,
    /// Remove profanity completely
    Removed,
    /// Tag profanity
    Tagged,
}

impl Default for ProfanityFilter {
    fn default() -> Self {
        Self::Masked
    }
}

/// Azure Function HTTP response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureHttpResponse {
    /// HTTP status code
    pub status: u16,
    /// Response headers
    pub headers: std::collections::HashMap<String, String>,
    /// Response body
    pub body: AzureRecognitionResponse,
}

/// Azure recognition response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureRecognitionResponse {
    /// Recognition status
    pub status: RecognitionStatus,
    /// Recognition results
    pub results: Vec<RecognitionResult>,
    /// Offset in ticks (100ns units)
    pub offset: i64,
    /// Duration in ticks
    pub duration: i64,
}

/// Recognition status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecognitionStatus {
    /// Success
    Success,
    /// No speech detected
    NoMatch,
    /// Request cancelled
    Cancelled,
    /// Error occurred
    Error,
}

/// Recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionResult {
    /// Result text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Word-level information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordDetail>>,
}

/// Word-level detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordDetail {
    /// Word text
    pub word: String,
    /// Offset in ticks
    pub offset: i64,
    /// Duration in ticks
    pub duration: i64,
    /// Confidence
    pub confidence: f32,
}

/// Azure Function handler
pub struct AzureFunctionHandler {
    config: ServerlessConfig,
    metrics: Arc<RwLock<ServerlessMetrics>>,
    initialized: Arc<RwLock<bool>>,
}

impl AzureFunctionHandler {
    /// Create a new Azure function handler
    pub fn new(config: ServerlessConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(ServerlessMetrics::default())),
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize handler
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing Azure Function handler");
        let start = std::time::Instant::now();

        if self.config.preload_models {
            debug!("Preloading models for Azure Function");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let duration = start.elapsed();
        let mut metrics = self.metrics.write();
        metrics.cold_start_duration = Some(duration);

        *self.initialized.write() = true;
        info!("Azure function handler initialized in {:?}", duration);

        Ok(())
    }

    /// Handle HTTP request
    pub async fn handle_http_request(
        &self,
        request: AzureHttpRequest,
    ) -> Result<AzureHttpResponse> {
        let start = std::time::Instant::now();

        if !*self.initialized.read() {
            self.initialize().await?;
        }

        let body = request.body.ok_or_else(|| {
            ServerlessError::BatchProcessingError("Missing request body".to_string())
        })?;

        debug!("Processing Azure function request");

        // Get audio data
        let audio_data = match &body.audio_config.source {
            AudioSource::Base64(content) => base64::decode(content).map_err(|e| {
                ServerlessError::BatchProcessingError(format!("Failed to decode audio: {}", e))
            })?,
            AudioSource::BlobUri(uri) => {
                debug!("Fetching audio from Azure Blob: {}", uri);
                // In real implementation, would fetch from Azure Blob Storage
                vec![0u8; 1000]
            }
        };

        debug!("Processing {} bytes of audio", audio_data.len());

        // Simulate recognition
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let processing_time = start.elapsed();
        let offset_ticks = 0i64; // Start offset in 100ns ticks
        let duration_ticks = (processing_time.as_secs_f64() * 10_000_000.0) as i64;

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.request_count += 1;
        metrics.total_processing_time += processing_time;
        metrics.average_latency = metrics.total_processing_time / metrics.request_count as u32;

        let mut headers = std::collections::HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        Ok(AzureHttpResponse {
            status: 200,
            headers,
            body: AzureRecognitionResponse {
                status: RecognitionStatus::Success,
                results: vec![RecognitionResult {
                    text: "Recognized text from Azure".to_string(),
                    confidence: 0.94,
                    words: if body.speech_config.enable_word_timing {
                        Some(vec![
                            WordDetail {
                                word: "Recognized".to_string(),
                                offset: offset_ticks,
                                duration: 5_000_000, // 0.5 seconds in ticks
                                confidence: 0.95,
                            },
                            WordDetail {
                                word: "text".to_string(),
                                offset: 5_000_000,
                                duration: 3_000_000, // 0.3 seconds in ticks
                                confidence: 0.93,
                            },
                        ])
                    } else {
                        None
                    },
                }],
                offset: offset_ticks,
                duration: duration_ticks,
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
    async fn test_azure_handler_initialization() {
        let config = ServerlessConfig::default();
        let handler = AzureFunctionHandler::new(config);

        let result = handler.initialize().await;
        assert!(result.is_ok());
        assert!(handler.is_initialized());
    }

    #[tokio::test]
    async fn test_azure_handle_base64_request() {
        let config = ServerlessConfig::default();
        let handler = AzureFunctionHandler::new(config);
        handler.initialize().await.unwrap();

        let request = AzureHttpRequest {
            method: "POST".to_string(),
            url: "/api/recognize".to_string(),
            headers: std::collections::HashMap::new(),
            body: Some(AzureRecognitionRequest {
                audio_config: AudioConfig {
                    source: AudioSource::Base64(base64::encode(b"audio data")),
                    format: AudioFormat {
                        sample_rate: 16000,
                        channels: 1,
                        encoding: "pcm".to_string(),
                    },
                },
                speech_config: SpeechConfig {
                    language: "en-US".to_string(),
                    enable_detailed_results: true,
                    enable_word_timing: false,
                    profanity_filter: ProfanityFilter::Masked,
                },
            }),
        };

        let response = handler.handle_http_request(request).await.unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body.status, RecognitionStatus::Success);
    }

    #[tokio::test]
    async fn test_azure_handle_blob_uri_request() {
        let config = ServerlessConfig::default();
        let handler = AzureFunctionHandler::new(config);
        handler.initialize().await.unwrap();

        let request = AzureHttpRequest {
            method: "POST".to_string(),
            url: "/api/recognize".to_string(),
            headers: std::collections::HashMap::new(),
            body: Some(AzureRecognitionRequest {
                audio_config: AudioConfig {
                    source: AudioSource::BlobUri(
                        "https://storage.azure.com/container/audio.wav".to_string(),
                    ),
                    format: AudioFormat {
                        sample_rate: 16000,
                        channels: 1,
                        encoding: "pcm".to_string(),
                    },
                },
                speech_config: SpeechConfig {
                    language: "ja-JP".to_string(),
                    enable_detailed_results: true,
                    enable_word_timing: true,
                    profanity_filter: ProfanityFilter::Removed,
                },
            }),
        };

        let response = handler.handle_http_request(request).await.unwrap();
        assert_eq!(response.status, 200);
        assert!(response.body.results[0].words.is_some());
    }

    #[test]
    fn test_profanity_filter() {
        assert_eq!(ProfanityFilter::default(), ProfanityFilter::Masked);
        assert_ne!(ProfanityFilter::None, ProfanityFilter::Masked);
    }

    #[test]
    fn test_azure_request_serialization() {
        let request = AzureRecognitionRequest {
            audio_config: AudioConfig {
                source: AudioSource::Base64("data".to_string()),
                format: AudioFormat {
                    sample_rate: 16000,
                    channels: 1,
                    encoding: "pcm".to_string(),
                },
            },
            speech_config: SpeechConfig {
                language: "en-US".to_string(),
                enable_detailed_results: true,
                enable_word_timing: true,
                profanity_filter: ProfanityFilter::Tagged,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: AzureRecognitionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.speech_config.language, "en-US");
    }
}
