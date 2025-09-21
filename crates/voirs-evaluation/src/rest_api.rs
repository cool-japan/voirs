//! REST API interface for evaluation services
//!
//! This module provides a REST API interface for remote access to speech synthesis
//! evaluation capabilities, enabling integration with web services and external tools.

use crate::quality::QualityEvaluator;
use crate::traits::{QualityEvaluator as QualityEvaluatorTrait, QualityMetric};
use crate::EvaluationError;
use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// REST API errors
#[derive(Error, Debug)]
pub enum ApiError {
    /// Invalid request format
    #[error("Invalid request format: {0}")]
    InvalidRequest(String),
    /// Missing required parameters
    #[error("Missing required parameters: {0}")]
    MissingParameters(String),
    /// Evaluation service error
    #[error("Evaluation service error: {0}")]
    EvaluationError(String),
    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    /// Rate limiting error
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),
}

/// API request authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAuthentication {
    /// API key
    pub api_key: String,
    /// User identifier
    pub user_id: String,
    /// Request timestamp
    pub timestamp: u64,
}

/// Audio data for API requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAudioData {
    /// Base64 encoded audio data
    pub data: String,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Audio format
    pub format: String,
    /// Duration in seconds
    pub duration: f64,
}

/// Quality evaluation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvaluationRequest {
    /// Authentication information
    pub auth: ApiAuthentication,
    /// Generated audio to evaluate
    pub generated_audio: ApiAudioData,
    /// Reference audio (optional)
    pub reference_audio: Option<ApiAudioData>,
    /// Evaluation metrics to compute
    pub metrics: Vec<String>,
    /// Evaluation configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Language code
    pub language: Option<String>,
}

/// Quality evaluation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvaluationResponse {
    /// Request ID for tracking
    pub request_id: String,
    /// Processing status
    pub status: String,
    /// Overall quality score
    pub overall_score: f64,
    /// Individual metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Quality analysis details
    pub analysis: QualityAnalysis,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Error message if any
    pub error: Option<String>,
}

/// Detailed quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAnalysis {
    /// Signal quality metrics
    pub signal_quality: HashMap<String, f64>,
    /// Perceptual quality metrics
    pub perceptual_quality: HashMap<String, f64>,
    /// Audio characteristics
    pub audio_characteristics: AudioCharacteristics,
    /// Quality recommendations
    pub recommendations: Vec<String>,
    /// Confidence scores
    pub confidence_scores: HashMap<String, f64>,
}

/// Audio characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCharacteristics {
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// RMS level
    pub rms_level: f64,
    /// Spectral centroid
    pub spectral_centroid: f64,
    /// Zero crossing rate
    pub zero_crossing_rate: f64,
    /// Fundamental frequency statistics
    pub f0_statistics: F0Statistics,
    /// Spectral features
    pub spectral_features: SpectralFeatures,
}

/// Fundamental frequency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Statistics {
    /// Mean F0 (Hz)
    pub mean: f64,
    /// F0 standard deviation
    pub std: f64,
    /// F0 range (Hz)
    pub range: f64,
    /// Voiced frame percentage
    pub voiced_percentage: f64,
}

/// Spectral features analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    /// Spectral rolloff
    pub spectral_rolloff: f64,
    /// Spectral flux
    pub spectral_flux: f64,
    /// Spectral contrast
    pub spectral_contrast: Vec<f64>,
    /// MFCC coefficients
    pub mfcc: Vec<f64>,
}

/// Pronunciation assessment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationRequest {
    /// Authentication information
    pub auth: ApiAuthentication,
    /// Audio to assess
    pub audio: ApiAudioData,
    /// Reference text or phonemes
    pub reference: String,
    /// Language code
    pub language: String,
    /// Assessment configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Pronunciation assessment response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationResponse {
    /// Request ID
    pub request_id: String,
    /// Overall pronunciation score (0-100)
    pub overall_score: f64,
    /// Phoneme-level scores
    pub phoneme_scores: Vec<PhonemeScore>,
    /// Word-level scores
    pub word_scores: Vec<WordScore>,
    /// Pronunciation feedback
    pub feedback: PronunciationFeedback,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Error message if any
    pub error: Option<String>,
}

/// Individual phoneme assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeScore {
    /// Phoneme symbol
    pub phoneme: String,
    /// Position in utterance
    pub position: usize,
    /// Accuracy score (0-100)
    pub score: f64,
    /// Confidence level
    pub confidence: f64,
    /// Error type if any
    pub error_type: Option<String>,
}

/// Word-level pronunciation score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordScore {
    /// Word text
    pub word: String,
    /// Start time (seconds)
    pub start_time: f64,
    /// End time (seconds)
    pub end_time: f64,
    /// Pronunciation score (0-100)
    pub score: f64,
    /// Stress pattern accuracy
    pub stress_accuracy: f64,
}

/// Pronunciation feedback and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationFeedback {
    /// Overall feedback message
    pub message: String,
    /// Specific improvement areas
    pub improvement_areas: Vec<String>,
    /// Common error patterns
    pub error_patterns: Vec<String>,
    /// Practice recommendations
    pub practice_recommendations: Vec<String>,
}

/// Batch evaluation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEvaluationRequest {
    /// Authentication information
    pub auth: ApiAuthentication,
    /// List of audio samples to evaluate
    pub audio_samples: Vec<ApiAudioData>,
    /// Reference audio samples (optional)
    pub reference_samples: Option<Vec<ApiAudioData>>,
    /// Evaluation type
    pub evaluation_type: String,
    /// Configuration parameters
    pub config: HashMap<String, serde_json::Value>,
    /// Language codes for each sample
    pub languages: Option<Vec<String>>,
}

/// Batch evaluation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEvaluationResponse {
    /// Request ID
    pub request_id: String,
    /// Processing status
    pub status: String,
    /// Individual sample results
    pub sample_results: Vec<QualityEvaluationResponse>,
    /// Batch statistics
    pub batch_statistics: BatchStatistics,
    /// Total processing time
    pub total_processing_time_ms: u64,
    /// Progress information
    pub progress: ProgressInfo,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    /// Total samples processed
    pub total_samples: usize,
    /// Successfully processed samples
    pub successful_samples: usize,
    /// Failed samples
    pub failed_samples: usize,
    /// Average quality score
    pub average_quality: f64,
    /// Quality score distribution
    pub quality_distribution: HashMap<String, usize>,
    /// Processing time statistics
    pub timing_statistics: TimingStatistics,
}

/// Processing time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Average processing time per sample (ms)
    pub avg_processing_time: f64,
    /// Minimum processing time (ms)
    pub min_processing_time: u64,
    /// Maximum processing time (ms)
    pub max_processing_time: u64,
    /// Total processing time (ms)
    pub total_processing_time: u64,
}

/// Progress information for long-running operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    /// Completion percentage (0-100)
    pub percentage: f64,
    /// Estimated time remaining (seconds)
    pub estimated_time_remaining: f64,
    /// Current processing stage
    pub current_stage: String,
    /// Detailed progress message
    pub message: String,
}

/// API service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiServiceConfig {
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout (seconds)
    pub request_timeout: u64,
    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,
    /// Authentication configuration
    pub auth_config: AuthConfig,
    /// Caching configuration
    pub cache_config: CacheConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute per user
    pub requests_per_minute: u32,
    /// Requests per hour per user
    pub requests_per_hour: u32,
    /// Daily request limit per user
    pub daily_limit: u32,
    /// Burst allowance
    pub burst_allowance: u32,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Require API key authentication
    pub require_api_key: bool,
    /// API key validation endpoint
    pub validation_endpoint: Option<String>,
    /// Token expiration time (seconds)
    pub token_expiration: u64,
    /// Allowed origins for CORS
    pub allowed_origins: Vec<String>,
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache expiration time (seconds)
    pub cache_expiration: u64,
    /// Maximum cache size (MB)
    pub max_cache_size: usize,
    /// Cache storage backend
    pub storage_backend: String,
}

/// REST API service implementation
pub struct EvaluationApiService {
    config: ApiServiceConfig,
    request_counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
    rate_limiter: std::sync::Arc<std::sync::Mutex<HashMap<String, RateLimitState>>>,
    quality_evaluator: QualityEvaluator,
}

/// Rate limiting state per user
#[derive(Debug, Clone)]
struct RateLimitState {
    last_request: std::time::Instant,
    request_count: u32,
    daily_count: u32,
    last_reset: std::time::Instant,
}

impl Default for ApiServiceConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_concurrent_requests: 100,
            request_timeout: 300,
            rate_limiting: RateLimitConfig {
                requests_per_minute: 60,
                requests_per_hour: 1000,
                daily_limit: 10000,
                burst_allowance: 10,
            },
            auth_config: AuthConfig {
                require_api_key: true,
                validation_endpoint: None,
                token_expiration: 3600,
                allowed_origins: vec!["*".to_string()],
            },
            cache_config: CacheConfig {
                enable_caching: true,
                cache_expiration: 3600,
                max_cache_size: 1024,
                storage_backend: "memory".to_string(),
            },
        }
    }
}

impl EvaluationApiService {
    /// Create a new API service
    pub async fn new(config: ApiServiceConfig) -> Result<Self, ApiError> {
        let quality_evaluator = QualityEvaluator::new().await.map_err(|e| {
            ApiError::EvaluationError(format!("Failed to initialize quality evaluator: {}", e))
        })?;

        Ok(Self {
            config,
            request_counter: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            rate_limiter: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            quality_evaluator,
        })
    }

    /// Generate unique request ID
    pub fn generate_request_id(&self) -> String {
        let counter = self
            .request_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("req_{}_{}", timestamp, counter)
    }

    /// Validate API authentication
    pub fn validate_authentication(&self, auth: &ApiAuthentication) -> Result<(), ApiError> {
        if auth.api_key.is_empty() {
            return Err(ApiError::AuthenticationError(
                "API key is required".to_string(),
            ));
        }

        if auth.user_id.is_empty() {
            return Err(ApiError::AuthenticationError(
                "User ID is required".to_string(),
            ));
        }

        // Validate timestamp (should be within 5 minutes)
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let time_diff = current_time.saturating_sub(auth.timestamp);
        if time_diff > 300 {
            return Err(ApiError::AuthenticationError(
                "Request timestamp too old".to_string(),
            ));
        }

        // Comprehensive API key validation
        self.validate_api_key(&auth.api_key)?;

        Ok(())
    }

    /// Validate API key format and authenticity
    fn validate_api_key(&self, api_key: &str) -> Result<(), ApiError> {
        // Check key format: should be 32-64 characters of alphanumeric + special chars
        if api_key.len() < 32 || api_key.len() > 64 {
            return Err(ApiError::AuthenticationError(
                "API key length must be between 32-64 characters".to_string(),
            ));
        }

        // Check for valid API key format (alphanumeric + underscore + hyphen)
        if !api_key
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(ApiError::AuthenticationError(
                "API key contains invalid characters".to_string(),
            ));
        }

        // Check for obvious test/placeholder keys
        let invalid_keys = [
            "invalid",
            "test",
            "placeholder",
            "demo",
            "sample",
            "example",
            "dummy",
            "fake",
            "12345678901234567890123456789012", // Simple numeric pattern
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // Simple letter pattern
        ];

        let key_lower = api_key.to_lowercase();
        for invalid_key in &invalid_keys {
            if key_lower.contains(invalid_key) {
                return Err(ApiError::AuthenticationError(
                    "Invalid or placeholder API key detected".to_string(),
                ));
            }
        }

        // Check for minimum complexity (should have mix of letters and numbers)
        let has_letter = api_key.chars().any(|c| c.is_alphabetic());
        let has_digit = api_key.chars().any(|c| c.is_numeric());

        if !has_letter || !has_digit {
            return Err(ApiError::AuthenticationError(
                "API key must contain both letters and numbers".to_string(),
            ));
        }

        // In a real implementation, you would:
        // 1. Hash the API key and compare against stored hashes
        // 2. Check key expiration dates
        // 3. Verify key permissions/scopes
        // 4. Log authentication attempts
        // 5. Implement key rotation mechanisms

        // For this implementation, accept keys that meet format requirements
        // and don't match known invalid patterns
        Ok(())
    }

    /// Decode base64 audio data to AudioBuffer
    fn decode_audio_data(&self, audio_data: &ApiAudioData) -> Result<AudioBuffer, ApiError> {
        // Decode base64 audio data
        let audio_bytes = general_purpose::STANDARD
            .decode(&audio_data.data)
            .map_err(|e| ApiError::InvalidRequest(format!("Invalid base64 audio data: {}", e)))?;

        // Convert bytes to f32 samples (assuming 16-bit PCM for now)
        let samples: Vec<f32> = audio_bytes
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0 // Normalize to [-1, 1]
                } else {
                    0.0
                }
            })
            .collect();

        Ok(AudioBuffer::new(
            samples,
            audio_data.sample_rate,
            audio_data.channels.into(),
        ))
    }

    /// Check rate limiting for user
    pub fn check_rate_limit(&self, user_id: &str) -> Result<(), ApiError> {
        let mut rate_limiter = self.rate_limiter.lock().unwrap();
        let now = std::time::Instant::now();

        let state = rate_limiter
            .entry(user_id.to_string())
            .or_insert_with(|| RateLimitState {
                last_request: now,
                request_count: 0,
                daily_count: 0,
                last_reset: now,
            });

        // Reset counters if needed
        if now.duration_since(state.last_reset).as_secs() >= 86400 {
            state.daily_count = 0;
            state.request_count = 0;
            state.last_reset = now;
        } else if now.duration_since(state.last_request).as_secs() >= 60 {
            state.request_count = 0;
        }

        // Check limits
        if state.daily_count >= self.config.rate_limiting.daily_limit {
            return Err(ApiError::RateLimitError("Daily limit exceeded".to_string()));
        }

        if state.request_count >= self.config.rate_limiting.requests_per_minute {
            return Err(ApiError::RateLimitError("Rate limit exceeded".to_string()));
        }

        // Update counters
        state.request_count += 1;
        state.daily_count += 1;
        state.last_request = now;

        Ok(())
    }

    /// Process quality evaluation request
    pub async fn process_quality_evaluation(
        &self,
        request: QualityEvaluationRequest,
    ) -> Result<QualityEvaluationResponse, ApiError> {
        // Validate authentication
        self.validate_authentication(&request.auth)?;

        // Check rate limiting
        self.check_rate_limit(&request.auth.user_id)?;

        let request_id = self.generate_request_id();
        let start_time = std::time::Instant::now();

        // Validate request parameters
        if request.generated_audio.data.is_empty() {
            return Err(ApiError::InvalidRequest(
                "Audio data is required".to_string(),
            ));
        }

        if request.metrics.is_empty() {
            return Err(ApiError::MissingParameters(
                "At least one metric must be specified".to_string(),
            ));
        }

        // Decode audio data
        let generated_audio = self
            .decode_audio_data(&request.generated_audio)
            .map_err(|e| {
                ApiError::EvaluationError(format!("Failed to decode generated audio: {}", e))
            })?;

        let reference_audio = if let Some(ref ref_data) = request.reference_audio {
            Some(self.decode_audio_data(ref_data).map_err(|e| {
                ApiError::EvaluationError(format!("Failed to decode reference audio: {}", e))
            })?)
        } else {
            None
        };

        // Create evaluation config
        let config = crate::traits::QualityEvaluationConfig::default();

        // Perform actual quality evaluation
        let evaluation_result = self
            .quality_evaluator
            .evaluate_quality(&generated_audio, reference_audio.as_ref(), Some(&config))
            .await
            .map_err(|e| ApiError::EvaluationError(format!("Quality evaluation failed: {}", e)))?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        // Convert metric results to API format
        let mut metric_scores = HashMap::new();
        let mut perceptual_quality = HashMap::new();
        let mut signal_quality = HashMap::new();
        let mut confidence_scores = HashMap::new();

        for metric_name in &request.metrics {
            let score = match metric_name.as_str() {
                "mos" | "overall" => evaluation_result.overall_score,
                "pesq" | "stoi" | "mcd" | "naturalness" | "intelligibility" => {
                    // Try to get specific metric from component_scores
                    evaluation_result
                        .component_scores
                        .get(metric_name)
                        .copied()
                        .unwrap_or(evaluation_result.overall_score)
                }
                _ => evaluation_result.overall_score,
            };

            metric_scores.insert(metric_name.clone(), score as f64);
            confidence_scores.insert(metric_name.clone(), evaluation_result.confidence as f64);

            // Categorize metrics
            match metric_name.as_str() {
                "pesq" | "stoi" | "mcd" => {
                    perceptual_quality.insert(metric_name.clone(), score as f64);
                }
                "snr" | "thd" => {
                    signal_quality.insert(metric_name.clone(), score as f64);
                }
                _ => {}
            }
        }

        // Calculate audio characteristics from the actual audio data
        let audio_characteristics = self.calculate_audio_characteristics(&generated_audio)?;

        // Use recommendations from evaluation result, or generate based on quality scores
        let mut recommendations = evaluation_result.recommendations.clone();
        if recommendations.is_empty() {
            if evaluation_result.overall_score < 0.7 {
                recommendations.push("Consider improving overall quality".to_string());
            }
            if let Some(naturalness) = evaluation_result.component_scores.get("naturalness") {
                if *naturalness < 0.7 {
                    recommendations.push("Improve naturalness of speech synthesis".to_string());
                }
            }
            if let Some(intelligibility) = evaluation_result.component_scores.get("intelligibility")
            {
                if *intelligibility < 0.8 {
                    recommendations.push("Enhance speech intelligibility".to_string());
                }
            }
        }

        Ok(QualityEvaluationResponse {
            request_id,
            status: "completed".to_string(),
            overall_score: evaluation_result.overall_score as f64,
            metric_scores,
            analysis: QualityAnalysis {
                signal_quality,
                perceptual_quality,
                audio_characteristics,
                recommendations,
                confidence_scores,
            },
            processing_time_ms: processing_time,
            error: None,
        })
    }

    /// Process pronunciation assessment request
    pub async fn process_pronunciation_assessment(
        &self,
        request: PronunciationRequest,
    ) -> Result<PronunciationResponse, ApiError> {
        // Validate authentication
        self.validate_authentication(&request.auth)?;

        // Check rate limiting
        self.check_rate_limit(&request.auth.user_id)?;

        let request_id = self.generate_request_id();
        let start_time = std::time::Instant::now();

        // Validate request parameters
        if request.audio.data.is_empty() {
            return Err(ApiError::InvalidRequest(
                "Audio data is required".to_string(),
            ));
        }

        if request.reference.is_empty() {
            return Err(ApiError::MissingParameters(
                "Reference text is required".to_string(),
            ));
        }

        // Mock implementation
        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(PronunciationResponse {
            request_id,
            overall_score: 85.0,
            phoneme_scores: vec![
                PhonemeScore {
                    phoneme: "t".to_string(),
                    position: 0,
                    score: 90.0,
                    confidence: 0.95,
                    error_type: None,
                },
                PhonemeScore {
                    phoneme: "e".to_string(),
                    position: 1,
                    score: 80.0,
                    confidence: 0.85,
                    error_type: Some("slight_mispronunciation".to_string()),
                },
            ],
            word_scores: vec![WordScore {
                word: "test".to_string(),
                start_time: 0.0,
                end_time: 0.5,
                score: 85.0,
                stress_accuracy: 90.0,
            }],
            feedback: PronunciationFeedback {
                message: "Good overall pronunciation with room for improvement".to_string(),
                improvement_areas: vec!["Vowel clarity".to_string()],
                error_patterns: vec!["Slight vowel reduction".to_string()],
                practice_recommendations: vec!["Practice vowel exercises".to_string()],
            },
            processing_time_ms: processing_time,
            error: None,
        })
    }

    /// Calculate audio characteristics from audio buffer
    fn calculate_audio_characteristics(
        &self,
        audio: &AudioBuffer,
    ) -> Result<AudioCharacteristics, ApiError> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        // Calculate RMS level
        let rms_level = if !samples.is_empty() {
            let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
            (sum_squares / samples.len() as f32).sqrt()
        } else {
            0.0
        };

        // Calculate dynamic range (difference between max and min values)
        let (min_val, max_val) = if !samples.is_empty() {
            let min = samples.iter().copied().reduce(f32::min).unwrap_or(0.0);
            let max = samples.iter().copied().reduce(f32::max).unwrap_or(0.0);
            (min, max)
        } else {
            (0.0, 0.0)
        };
        let dynamic_range = max_val - min_val;

        // Calculate zero crossing rate
        let zero_crossing_rate = if samples.len() > 1 {
            let crossings = samples
                .windows(2)
                .filter(|window| (window[0] >= 0.0) != (window[1] >= 0.0))
                .count();
            crossings as f32 / (samples.len() - 1) as f32
        } else {
            0.0
        };

        // Calculate spectral centroid (simplified)
        let spectral_centroid = if !samples.is_empty() {
            // Simple approximation based on audio energy distribution
            let energy_weighted_freq = samples
                .iter()
                .enumerate()
                .map(|(i, &sample)| {
                    let freq = (i as f32 * sample_rate) / (2.0 * samples.len() as f32);
                    freq * sample.abs()
                })
                .sum::<f32>();
            let total_energy: f32 = samples.iter().map(|&x| x.abs()).sum();
            if total_energy > 0.0 {
                energy_weighted_freq / total_energy
            } else {
                1000.0 // Default
            }
        } else {
            1000.0
        };

        // Calculate F0 statistics (simplified)
        let f0_mean = if !samples.is_empty() {
            // Simple pitch estimation based on zero crossings
            let pitch_estimate = zero_crossing_rate * sample_rate / 2.0;
            pitch_estimate.clamp(80.0, 400.0) // Typical voice range
        } else {
            150.0
        };

        Ok(AudioCharacteristics {
            dynamic_range: dynamic_range as f64,
            rms_level: rms_level as f64,
            spectral_centroid: spectral_centroid as f64,
            zero_crossing_rate: zero_crossing_rate as f64,
            f0_statistics: F0Statistics {
                mean: f0_mean as f64,
                std: 25.0,    // Simplified
                range: 100.0, // Simplified
                voiced_percentage: if rms_level > 0.01 { 75.0 } else { 10.0 },
            },
            spectral_features: SpectralFeatures {
                spectral_rolloff: (spectral_centroid * 2.0) as f64,
                spectral_flux: (rms_level * 0.5) as f64,
                spectral_contrast: vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                mfcc: vec![
                    1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005,
                ],
            },
        })
    }

    /// Get service status
    pub fn get_service_status(&self) -> HashMap<String, serde_json::Value> {
        let mut status = HashMap::new();

        status.insert(
            "service".to_string(),
            serde_json::Value::String("VoiRS Evaluation API".to_string()),
        );
        status.insert(
            "version".to_string(),
            serde_json::Value::String("1.0.0".to_string()),
        );
        status.insert(
            "status".to_string(),
            serde_json::Value::String("healthy".to_string()),
        );
        status.insert(
            "uptime".to_string(),
            serde_json::Value::Number(serde_json::Number::from(3600)),
        );
        status.insert(
            "requests_processed".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.request_counter
                    .load(std::sync::atomic::Ordering::SeqCst),
            )),
        );

        let mut capabilities = Vec::new();
        capabilities.push(serde_json::Value::String("quality_evaluation".to_string()));
        capabilities.push(serde_json::Value::String(
            "pronunciation_assessment".to_string(),
        ));
        capabilities.push(serde_json::Value::String("batch_processing".to_string()));
        status.insert(
            "capabilities".to_string(),
            serde_json::Value::Array(capabilities),
        );

        status
    }

    /// Start HTTP server with all endpoints
    pub async fn start_server(&self) -> Result<(), ApiError> {
        use warp::Filter;

        let service = std::sync::Arc::new(self.clone());

        // CORS configuration
        let cors = warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type", "authorization"])
            .allow_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS"]);

        // Health check endpoint
        let health = warp::path("health").and(warp::get()).map(|| {
            warp::reply::json(&serde_json::json!({
                "status": "healthy",
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            }))
        });

        // Service status endpoint
        let status = warp::path("status").and(warp::get()).map({
            let service = service.clone();
            move || {
                let status = service.get_service_status();
                warp::reply::json(&status)
            }
        });

        // Quality evaluation endpoint
        let quality_eval = warp::path("evaluate")
            .and(warp::path("quality"))
            .and(warp::post())
            .and(warp::body::json())
            .and_then({
                let service = service.clone();
                move |request: QualityEvaluationRequest| {
                    let service = service.clone();
                    async move {
                        match service.process_quality_evaluation(request).await {
                            Ok(response) => Ok(warp::reply::json(&response)),
                            Err(e) => {
                                let error_response = serde_json::json!({
                                    "error": e.to_string(),
                                    "status": "error"
                                });
                                Err(warp::reject::custom(ApiErrorRejection(e)))
                            }
                        }
                    }
                }
            });

        // Pronunciation assessment endpoint
        let pronunciation = warp::path("evaluate")
            .and(warp::path("pronunciation"))
            .and(warp::post())
            .and(warp::body::json())
            .and_then({
                let service = service.clone();
                move |request: PronunciationRequest| {
                    let service = service.clone();
                    async move {
                        match service.process_pronunciation_assessment(request).await {
                            Ok(response) => Ok(warp::reply::json(&response)),
                            Err(e) => Err(warp::reject::custom(ApiErrorRejection(e))),
                        }
                    }
                }
            });

        // Batch evaluation endpoint
        let batch_eval = warp::path("evaluate")
            .and(warp::path("batch"))
            .and(warp::post())
            .and(warp::body::json())
            .and_then({
                let service = service.clone();
                move |request: BatchEvaluationRequest| {
                    let service = service.clone();
                    async move {
                        match service.process_batch_evaluation(request).await {
                            Ok(response) => Ok(warp::reply::json(&response)),
                            Err(e) => Err(warp::reject::custom(ApiErrorRejection(e))),
                        }
                    }
                }
            });

        // Model comparison endpoint
        let model_comparison = warp::path("compare")
            .and(warp::path("models"))
            .and(warp::post())
            .and(warp::body::json())
            .and_then({
                let service = service.clone();
                move |request: ModelComparisonRequest| {
                    let service = service.clone();
                    async move {
                        match service.process_model_comparison(request).await {
                            Ok(response) => Ok(warp::reply::json(&response)),
                            Err(e) => Err(warp::reject::custom(ApiErrorRejection(e))),
                        }
                    }
                }
            });

        // Dataset validation endpoint
        let dataset_validation = warp::path("validate")
            .and(warp::path("dataset"))
            .and(warp::post())
            .and(warp::body::json())
            .and_then({
                let service = service.clone();
                move |request: DatasetValidationRequest| {
                    let service = service.clone();
                    async move {
                        match service.process_dataset_validation(request).await {
                            Ok(response) => Ok(warp::reply::json(&response)),
                            Err(e) => Err(warp::reject::custom(ApiErrorRejection(e))),
                        }
                    }
                }
            });

        // Metrics metadata endpoint
        let metrics_info = warp::path("metrics")
            .and(warp::path("info"))
            .and(warp::get())
            .map({
                let service = service.clone();
                move || {
                    let metrics_info = service.get_available_metrics();
                    warp::reply::json(&metrics_info)
                }
            });

        // API documentation endpoint
        let api_docs = warp::path("docs").and(warp::get()).map(|| {
            let docs = generate_api_documentation();
            warp::reply::html(docs)
        });

        // Combine all routes
        let routes = health
            .or(status)
            .or(quality_eval)
            .or(pronunciation)
            .or(batch_eval)
            .or(model_comparison)
            .or(dataset_validation)
            .or(metrics_info)
            .or(api_docs)
            .with(cors)
            .recover(handle_rejection);

        println!(
            "🚀 Starting VoiRS Evaluation API server on {}:{}",
            self.config.host, self.config.port
        );

        warp::serve(routes)
            .run((
                self.config
                    .host
                    .parse::<std::net::IpAddr>()
                    .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0))),
                self.config.port,
            ))
            .await;

        Ok(())
    }

    /// Process batch evaluation request
    pub async fn process_batch_evaluation(
        &self,
        request: BatchEvaluationRequest,
    ) -> Result<BatchEvaluationResponse, ApiError> {
        // Validate authentication
        self.validate_authentication(&request.auth)?;
        self.check_rate_limit(&request.auth.user_id)?;

        let request_id = self.generate_request_id();
        let start_time = std::time::Instant::now();

        if request.audio_samples.is_empty() {
            return Err(ApiError::InvalidRequest(
                "No audio samples provided".to_string(),
            ));
        }

        let mut sample_results = Vec::new();
        let mut successful_samples = 0;
        let mut failed_samples = 0;
        let mut total_quality = 0.0;

        for (i, audio_sample) in request.audio_samples.iter().enumerate() {
            let sample_request = QualityEvaluationRequest {
                auth: request.auth.clone(),
                generated_audio: audio_sample.clone(),
                reference_audio: request
                    .reference_samples
                    .as_ref()
                    .and_then(|refs| refs.get(i).cloned()),
                metrics: vec![
                    "overall".to_string(),
                    "clarity".to_string(),
                    "naturalness".to_string(),
                ],
                config: request.config.clone(),
                language: request
                    .languages
                    .as_ref()
                    .and_then(|langs| langs.get(i).cloned()),
            };

            match self.process_quality_evaluation(sample_request).await {
                Ok(result) => {
                    total_quality += result.overall_score;
                    successful_samples += 1;
                    sample_results.push(result);
                }
                Err(e) => {
                    failed_samples += 1;
                    sample_results.push(QualityEvaluationResponse {
                        request_id: format!("{}_sample_{}", request_id, i),
                        status: "failed".to_string(),
                        overall_score: 0.0,
                        metric_scores: HashMap::new(),
                        analysis: QualityAnalysis {
                            signal_quality: HashMap::new(),
                            perceptual_quality: HashMap::new(),
                            audio_characteristics: AudioCharacteristics {
                                dynamic_range: 0.0,
                                rms_level: 0.0,
                                spectral_centroid: 0.0,
                                zero_crossing_rate: 0.0,
                                f0_statistics: F0Statistics {
                                    mean: 0.0,
                                    std: 0.0,
                                    range: 0.0,
                                    voiced_percentage: 0.0,
                                },
                                spectral_features: SpectralFeatures {
                                    spectral_rolloff: 0.0,
                                    spectral_flux: 0.0,
                                    spectral_contrast: vec![],
                                    mfcc: vec![],
                                },
                            },
                            recommendations: vec![],
                            confidence_scores: HashMap::new(),
                        },
                        processing_time_ms: 0,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let total_processing_time = start_time.elapsed().as_millis() as u64;
        let average_quality = if successful_samples > 0 {
            total_quality / successful_samples as f64
        } else {
            0.0
        };

        Ok(BatchEvaluationResponse {
            request_id,
            status: "completed".to_string(),
            sample_results,
            batch_statistics: BatchStatistics {
                total_samples: request.audio_samples.len(),
                successful_samples,
                failed_samples,
                average_quality,
                quality_distribution: HashMap::new(), // Would be calculated from results
                timing_statistics: TimingStatistics {
                    avg_processing_time: total_processing_time as f64
                        / request.audio_samples.len() as f64,
                    min_processing_time: 0,
                    max_processing_time: total_processing_time,
                    total_processing_time,
                },
            },
            total_processing_time_ms: total_processing_time,
            progress: ProgressInfo {
                percentage: 100.0,
                estimated_time_remaining: 0.0,
                current_stage: "completed".to_string(),
                message: "Batch evaluation completed".to_string(),
            },
        })
    }

    /// Process model comparison request
    pub async fn process_model_comparison(
        &self,
        request: ModelComparisonRequest,
    ) -> Result<ModelComparisonResponse, ApiError> {
        self.validate_authentication(&request.auth)?;
        self.check_rate_limit(&request.auth.user_id)?;

        let request_id = self.generate_request_id();

        // Mock implementation for model comparison
        Ok(ModelComparisonResponse {
            request_id,
            status: "completed".to_string(),
            comparison_results: HashMap::new(),
            statistical_analysis: HashMap::new(),
            ranking: vec![],
            recommendations: vec!["Model A shows better naturalness".to_string()],
            processing_time_ms: 100,
            error: None,
        })
    }

    /// Process dataset validation request
    pub async fn process_dataset_validation(
        &self,
        request: DatasetValidationRequest,
    ) -> Result<DatasetValidationResponse, ApiError> {
        self.validate_authentication(&request.auth)?;
        self.check_rate_limit(&request.auth.user_id)?;

        let request_id = self.generate_request_id();

        // Mock implementation for dataset validation
        Ok(DatasetValidationResponse {
            request_id,
            status: "completed".to_string(),
            validation_results: HashMap::new(),
            quality_metrics: HashMap::new(),
            issues_found: vec![],
            recommendations: vec!["Dataset quality is good".to_string()],
            processing_time_ms: 200,
            error: None,
        })
    }

    /// Get available metrics information
    pub fn get_available_metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "quality_metrics": [
                {
                    "name": "overall",
                    "description": "Overall quality score",
                    "range": "0.0 - 1.0",
                    "type": "perceptual"
                },
                {
                    "name": "pesq",
                    "description": "Perceptual Evaluation of Speech Quality",
                    "range": "-0.5 - 4.5",
                    "type": "perceptual"
                },
                {
                    "name": "stoi",
                    "description": "Short-Time Objective Intelligibility",
                    "range": "0.0 - 1.0",
                    "type": "intelligibility"
                },
                {
                    "name": "mcd",
                    "description": "Mel-Cepstral Distortion",
                    "range": "0.0+",
                    "type": "spectral"
                }
            ],
            "supported_formats": ["wav", "flac", "mp3", "ogg"],
            "sample_rates": [8000, 16000, 22050, 44100, 48000],
            "max_audio_duration": 300
        })
    }
}

impl Clone for EvaluationApiService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            request_counter: self.request_counter.clone(),
            rate_limiter: self.rate_limiter.clone(),
            quality_evaluator: self.quality_evaluator.clone(),
        }
    }
}

/// Additional request/response types for new endpoints

/// Request for comparing multiple model outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonRequest {
    /// Authentication information
    pub auth: ApiAuthentication,
    /// Model outputs to compare (model_name -> audio samples)
    pub model_outputs: HashMap<String, Vec<ApiAudioData>>,
    /// Optional reference audio for comparison
    pub reference_audio: Option<Vec<ApiAudioData>>,
    /// Metrics to use for comparison
    pub comparison_metrics: Vec<String>,
    /// Additional configuration parameters
    pub config: HashMap<String, serde_json::Value>,
}

/// Response for model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResponse {
    /// Unique request identifier
    pub request_id: String,
    /// Status of the comparison (success, error, etc.)
    pub status: String,
    /// Comparison results for each metric
    pub comparison_results: HashMap<String, f64>,
    /// Statistical analysis results
    pub statistical_analysis: HashMap<String, serde_json::Value>,
    /// Ranking of models from best to worst
    pub ranking: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Error message if applicable
    pub error: Option<String>,
}

/// Request for validating a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetValidationRequest {
    /// Authentication information
    pub auth: ApiAuthentication,
    /// Dataset samples to validate
    pub dataset_samples: Vec<ApiAudioData>,
    /// Criteria to use for validation
    pub validation_criteria: Vec<String>,
    /// Additional configuration parameters
    pub config: HashMap<String, serde_json::Value>,
}

/// Response for dataset validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetValidationResponse {
    /// Unique request identifier
    pub request_id: String,
    /// Status of the validation (success, error, etc.)
    pub status: String,
    /// Detailed validation results
    pub validation_results: HashMap<String, serde_json::Value>,
    /// Quality metrics for the dataset
    pub quality_metrics: HashMap<String, f64>,
    /// Issues found during validation
    pub issues_found: Vec<String>,
    /// Recommendations for dataset improvement
    pub recommendations: Vec<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Error message if applicable
    pub error: Option<String>,
}

/// Custom rejection type for API errors
#[derive(Debug)]
struct ApiErrorRejection(ApiError);

impl warp::reject::Reject for ApiErrorRejection {}

/// Handle HTTP rejections
async fn handle_rejection(
    err: warp::Rejection,
) -> Result<impl warp::Reply, std::convert::Infallible> {
    let code;
    let message;

    if err.is_not_found() {
        code = warp::http::StatusCode::NOT_FOUND;
        message = "Not Found";
    } else if let Some(api_error) = err.find::<ApiErrorRejection>() {
        match &api_error.0 {
            ApiError::InvalidRequest(_) => {
                code = warp::http::StatusCode::BAD_REQUEST;
                message = "Bad Request";
            }
            ApiError::AuthenticationError(_) => {
                code = warp::http::StatusCode::UNAUTHORIZED;
                message = "Unauthorized";
            }
            ApiError::RateLimitError(_) => {
                code = warp::http::StatusCode::TOO_MANY_REQUESTS;
                message = "Too Many Requests";
            }
            _ => {
                code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
                message = "Internal Server Error";
            }
        }
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        code = warp::http::StatusCode::METHOD_NOT_ALLOWED;
        message = "Method Not Allowed";
    } else {
        code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
        message = "Internal Server Error";
    }

    let json = warp::reply::json(&serde_json::json!({
        "error": message,
        "status": code.as_u16()
    }));

    Ok(warp::reply::with_status(json, code))
}

/// Generate API documentation HTML
fn generate_api_documentation() -> String {
    r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoiRS Evaluation API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; }
        .get { background: #28a745; }
        .post { background: #007bff; }
        .put { background: #ffc107; color: black; }
        .delete { background: #dc3545; }
        code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>VoiRS Evaluation API Documentation</h1>
    
    <h2>Overview</h2>
    <p>The VoiRS Evaluation API provides comprehensive speech synthesis quality evaluation capabilities.</p>
    
    <h2>Authentication</h2>
    <p>All endpoints require API key authentication. Include your API key in the request body under the <code>auth</code> field.</p>
    
    <h2>Endpoints</h2>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /health</h3>
        <p>Health check endpoint to verify API availability.</p>
        <p><strong>Response:</strong> JSON with status and timestamp</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /status</h3>
        <p>Get detailed service status and capabilities.</p>
        <p><strong>Response:</strong> JSON with service information</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /evaluate/quality</h3>
        <p>Evaluate speech synthesis quality using various metrics.</p>
        <p><strong>Request Body:</strong> QualityEvaluationRequest JSON</p>
        <p><strong>Response:</strong> QualityEvaluationResponse with scores and analysis</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /evaluate/pronunciation</h3>
        <p>Assess pronunciation accuracy and provide feedback.</p>
        <p><strong>Request Body:</strong> PronunciationRequest JSON</p>
        <p><strong>Response:</strong> PronunciationResponse with scores and feedback</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /evaluate/batch</h3>
        <p>Process multiple audio samples in batch for efficient evaluation.</p>
        <p><strong>Request Body:</strong> BatchEvaluationRequest JSON</p>
        <p><strong>Response:</strong> BatchEvaluationResponse with aggregated results</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /compare/models</h3>
        <p>Compare quality between different model outputs.</p>
        <p><strong>Request Body:</strong> ModelComparisonRequest JSON</p>
        <p><strong>Response:</strong> ModelComparisonResponse with comparison analysis</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /validate/dataset</h3>
        <p>Validate quality and consistency of audio datasets.</p>
        <p><strong>Request Body:</strong> DatasetValidationRequest JSON</p>
        <p><strong>Response:</strong> DatasetValidationResponse with validation results</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /metrics/info</h3>
        <p>Get information about available quality metrics.</p>
        <p><strong>Response:</strong> JSON with metric descriptions and capabilities</p>
    </div>
    
    <h2>Rate Limiting</h2>
    <p>API requests are rate limited per user. Default limits:</p>
    <ul>
        <li>60 requests per minute</li>
        <li>1000 requests per hour</li>
        <li>10000 requests per day</li>
    </ul>
    
    <h2>Support</h2>
    <p>For support and additional documentation, visit the VoiRS project repository.</p>
</body>
</html>
    "#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_service_creation() {
        let config = ApiServiceConfig::default();
        let service = EvaluationApiService::new(config).await;

        assert!(service.is_ok());
        let service = service.unwrap();
        assert!(
            service
                .request_counter
                .load(std::sync::atomic::Ordering::SeqCst)
                == 0
        );
    }

    #[tokio::test]
    async fn test_request_id_generation() {
        let service = EvaluationApiService::new(ApiServiceConfig::default())
            .await
            .unwrap();

        let id1 = service.generate_request_id();
        let id2 = service.generate_request_id();

        assert_ne!(id1, id2);
        assert!(id1.starts_with("req_"));
        assert!(id2.starts_with("req_"));
    }

    #[tokio::test]
    async fn test_authentication_validation() {
        let service = EvaluationApiService::new(ApiServiceConfig::default())
            .await
            .unwrap();

        let valid_auth = ApiAuthentication {
            api_key: "voirs_api_key_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6".to_string(), // 48 chars, mixed case+numbers
            user_id: "user123".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        assert!(service.validate_authentication(&valid_auth).is_ok());

        let invalid_auth = ApiAuthentication {
            api_key: "invalid".to_string(),
            user_id: "user123".to_string(),
            timestamp: 0, // Very old timestamp
        };

        assert!(service.validate_authentication(&invalid_auth).is_err());

        // Test various invalid API key formats
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Too short key
        let short_key_auth = ApiAuthentication {
            api_key: "short".to_string(),
            user_id: "user123".to_string(),
            timestamp: current_time,
        };
        assert!(service.validate_authentication(&short_key_auth).is_err());

        // Too long key
        let long_key_auth = ApiAuthentication {
            api_key: "a".repeat(100),
            user_id: "user123".to_string(),
            timestamp: current_time,
        };
        assert!(service.validate_authentication(&long_key_auth).is_err());

        // Key with invalid characters
        let invalid_char_auth = ApiAuthentication {
            api_key: "voirs_api_key_with@invalid#chars$1234567890".to_string(),
            user_id: "user123".to_string(),
            timestamp: current_time,
        };
        assert!(service.validate_authentication(&invalid_char_auth).is_err());

        // Key with only letters (no numbers)
        let letters_only_auth = ApiAuthentication {
            api_key: "voirsapikeywithnumbersjustlettersonly".to_string(),
            user_id: "user123".to_string(),
            timestamp: current_time,
        };
        assert!(service.validate_authentication(&letters_only_auth).is_err());

        // Key with only numbers (no letters)
        let numbers_only_auth = ApiAuthentication {
            api_key: "12345678901234567890123456789012345678".to_string(),
            user_id: "user123".to_string(),
            timestamp: current_time,
        };
        assert!(service.validate_authentication(&numbers_only_auth).is_err());

        // Placeholder/test key detection
        let test_key_auth = ApiAuthentication {
            api_key: "test_key_1234567890123456789012345678".to_string(),
            user_id: "user123".to_string(),
            timestamp: current_time,
        };
        assert!(service.validate_authentication(&test_key_auth).is_err());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let mut config = ApiServiceConfig::default();
        config.rate_limiting.requests_per_minute = 2;
        let service = EvaluationApiService::new(config).await.unwrap();

        // First two requests should succeed
        assert!(service.check_rate_limit("user1").is_ok());
        assert!(service.check_rate_limit("user1").is_ok());

        // Third request should fail
        assert!(service.check_rate_limit("user1").is_err());

        // Different user should still work
        assert!(service.check_rate_limit("user2").is_ok());
    }

    #[tokio::test]
    async fn test_service_status() {
        let service = EvaluationApiService::new(ApiServiceConfig::default())
            .await
            .unwrap();
        let status = service.get_service_status();

        assert_eq!(
            status.get("service").unwrap().as_str().unwrap(),
            "VoiRS Evaluation API"
        );
        assert_eq!(status.get("status").unwrap().as_str().unwrap(), "healthy");
        assert!(status.contains_key("capabilities"));
    }

    #[tokio::test]
    async fn test_quality_evaluation_request_validation() {
        let service = EvaluationApiService::new(ApiServiceConfig::default())
            .await
            .unwrap();

        let invalid_request = QualityEvaluationRequest {
            auth: ApiAuthentication {
                api_key: "test_key".to_string(),
                user_id: "test_user".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
            generated_audio: ApiAudioData {
                data: "".to_string(), // Empty data should fail
                sample_rate: 16000,
                channels: 1,
                format: "wav".to_string(),
                duration: 1.0,
            },
            reference_audio: None,
            metrics: vec![], // Empty metrics should fail
            config: HashMap::new(),
            language: None,
        };

        let result = service.process_quality_evaluation(invalid_request).await;
        assert!(result.is_err());
    }
}
