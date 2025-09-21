//! Stream management and processing

use super::audio_processing::{AudioProcessor, AudioProcessorConfig};
use super::types::*;
use crate::traits::{FeedbackResponse, SessionState};
use crate::FeedbackError;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;
use voirs_sdk::AudioBuffer;

/// Real-time feedback stream for processing audio
pub struct FeedbackStream {
    /// Unique stream identifier
    pub stream_id: Uuid,
    /// Associated user ID
    pub user_id: String,
    /// Stream configuration
    pub config: RealtimeConfig,
    /// Current session state
    pub session_state: Arc<TokioRwLock<SessionState>>,
    /// Audio processor with VAD and feature extraction
    audio_processor: Arc<tokio::sync::Mutex<AudioProcessor>>,
    /// Latency optimizer for fast processing
    latency_optimizer: Arc<tokio::sync::Mutex<LatencyOptimizer>>,
}

impl FeedbackStream {
    /// Create a new feedback stream
    pub fn new(user_id: String, config: RealtimeConfig, session_state: SessionState) -> Self {
        // Create optimized audio processor configuration for low latency
        let audio_config = AudioProcessorConfig {
            frame_size: 256,                // Smaller frame size for lower latency
            always_extract_features: false, // Only extract when voice detected
            vad_config: super::audio_processing::VadConfig {
                energy_threshold: 0.005, // Lower threshold for real-time
                ..Default::default()
            },
            ..Default::default()
        };

        let audio_processor = AudioProcessor::new(audio_config);
        let latency_optimizer = LatencyOptimizer::new();

        Self {
            stream_id: Uuid::new_v4(),
            user_id,
            config,
            session_state: Arc::new(TokioRwLock::new(session_state)),
            audio_processor: Arc::new(tokio::sync::Mutex::new(audio_processor)),
            latency_optimizer: Arc::new(tokio::sync::Mutex::new(latency_optimizer)),
        }
    }

    /// Process audio through the stream with optimized latency
    pub async fn process_audio(
        &self,
        audio: &AudioBuffer,
        text: &str,
    ) -> Result<FeedbackResponse, FeedbackError> {
        let start_time = Instant::now();

        // Fast path: Check if we should use cached results
        let mut latency_optimizer = self.latency_optimizer.lock().await;
        if let Some(cached_response) = latency_optimizer.check_cache(audio, text) {
            latency_optimizer.record_latency(start_time.elapsed());
            return Ok(cached_response);
        }
        drop(latency_optimizer); // Release lock early

        // Process audio with VAD and feature extraction
        let mut audio_processor = self.audio_processor.lock().await;
        let audio_result = audio_processor.process_chunk(audio.samples())?;
        drop(audio_processor); // Release lock early

        // Generate feedback based on audio analysis and text
        let mut feedback_items = Vec::new();

        // Voice activity based feedback
        if audio_result.voice_activity_ratio > 0.8 {
            feedback_items.push(crate::traits::UserFeedback {
                message: "Excellent voice activity! Clear speech detected.".to_string(),
                suggestion: Some("Continue with this level of clarity.".to_string()),
                confidence: audio_result.quality_score,
                score: 0.9,
                priority: 0.8,
                metadata: HashMap::new(),
            });
        } else if audio_result.voice_activity_ratio > 0.5 {
            feedback_items.push(crate::traits::UserFeedback {
                message: "Good voice activity detected.".to_string(),
                suggestion: Some("Try to maintain consistent speaking pace.".to_string()),
                confidence: audio_result.quality_score,
                score: 0.7,
                priority: 0.6,
                metadata: HashMap::new(),
            });
        } else if audio_result.voice_activity_ratio > 0.1 {
            feedback_items.push(crate::traits::UserFeedback {
                message: "Some voice activity detected, but could be clearer.".to_string(),
                suggestion: Some("Speak more clearly and avoid long pauses.".to_string()),
                confidence: audio_result.quality_score,
                score: 0.5,
                priority: 0.7,
                metadata: HashMap::new(),
            });
        }

        // Text-based feedback (fast heuristics)
        if text.len() < 5 {
            feedback_items.push(crate::traits::UserFeedback {
                message: "Try practicing with longer phrases for better results.".to_string(),
                suggestion: Some("Practice with sentences of 5-10 words.".to_string()),
                confidence: 0.9,
                score: 0.6,
                priority: 0.8,
                metadata: HashMap::new(),
            });
        } else if text.len() > 50 {
            feedback_items.push(crate::traits::UserFeedback {
                message: "Great job practicing with longer sentences!".to_string(),
                suggestion: Some("Keep practicing with varied sentence structures.".to_string()),
                confidence: 0.8,
                score: 0.9,
                priority: 0.6,
                metadata: HashMap::new(),
            });
        }

        // Always ensure we have at least one feedback item
        if feedback_items.is_empty() {
            feedback_items.push(crate::traits::UserFeedback {
                message: "Good practice session! Keep up the great work.".to_string(),
                suggestion: Some("Try focusing on consistent practice.".to_string()),
                confidence: 0.7,
                score: 0.75,
                priority: 0.6,
                metadata: HashMap::new(),
            });
        }

        // Generate immediate actions and goals (pre-computed for speed)
        let immediate_actions = vec!["Continue practicing".to_string()];
        let long_term_goals = vec!["Improve pronunciation clarity".to_string()];

        // Create progress indicators based on audio analysis
        let progress_indicators = crate::traits::ProgressIndicators {
            improving_areas: if audio_result.voice_activity_ratio > 0.7 {
                vec!["pronunciation".to_string(), "clarity".to_string()]
            } else {
                vec!["consistency".to_string()]
            },
            attention_areas: if audio_result.voice_activity_ratio < 0.5 {
                vec!["voice clarity".to_string(), "speaking pace".to_string()]
            } else {
                vec!["fluency".to_string()]
            },
            stable_areas: vec!["vocabulary".to_string()],
            overall_trend: (audio_result.voice_activity_ratio - 0.5) * 0.2,
            completion_percentage: audio_result.quality_score.min(1.0),
        };

        // Calculate overall score
        let overall_score = if feedback_items.is_empty() {
            0.5
        } else {
            feedback_items.iter().map(|item| item.score).sum::<f32>() / feedback_items.len() as f32
        };

        let processing_time = start_time.elapsed();

        let response = FeedbackResponse {
            feedback_items,
            overall_score,
            immediate_actions,
            long_term_goals,
            progress_indicators,
            timestamp: chrono::Utc::now(),
            processing_time,
            feedback_type: crate::FeedbackType::Quality,
        };

        // Cache the response for future use
        let mut latency_optimizer = self.latency_optimizer.lock().await;
        latency_optimizer.cache_response(audio, text, response.clone());
        latency_optimizer.record_latency(processing_time);

        Ok(response)
    }

    /// Get stream performance statistics
    pub async fn get_performance_stats(&self) -> LatencyStats {
        let latency_optimizer = self.latency_optimizer.lock().await;
        latency_optimizer.get_stats()
    }

    /// Reset stream performance metrics
    pub async fn reset_performance_stats(&self) {
        let mut latency_optimizer = self.latency_optimizer.lock().await;
        latency_optimizer.reset();
    }
}

// Need to implement Debug manually since AudioProcessor doesn't implement Debug
impl std::fmt::Debug for FeedbackStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedbackStream")
            .field("stream_id", &self.stream_id)
            .field("user_id", &self.user_id)
            .field("config", &self.config)
            .field("audio_processor", &"AudioProcessor { ... }")
            .field("latency_optimizer", &"LatencyOptimizer { ... }")
            .finish()
    }
}

// Need to implement Clone manually since AudioProcessor doesn't implement Clone
impl Clone for FeedbackStream {
    fn clone(&self) -> Self {
        // Create new instances for the non-Clone fields
        let audio_config = AudioProcessorConfig {
            frame_size: 256,
            always_extract_features: false,
            vad_config: super::audio_processing::VadConfig {
                energy_threshold: 0.005,
                ..Default::default()
            },
            ..Default::default()
        };

        let audio_processor = AudioProcessor::new(audio_config);
        let latency_optimizer = LatencyOptimizer::new();

        Self {
            stream_id: self.stream_id,
            user_id: self.user_id.clone(),
            config: self.config.clone(),
            session_state: Arc::clone(&self.session_state),
            audio_processor: Arc::new(tokio::sync::Mutex::new(audio_processor)),
            latency_optimizer: Arc::new(tokio::sync::Mutex::new(latency_optimizer)),
        }
    }
}

/// Latency optimizer for fast audio processing
pub struct LatencyOptimizer {
    /// Cache for recent audio processing results
    response_cache: HashMap<String, CachedResponse>,
    /// Recent latency measurements
    latency_measurements: Vec<Duration>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Cache hit count
    cache_hits: u64,
    /// Total requests
    total_requests: u64,
}

impl LatencyOptimizer {
    /// Create a new latency optimizer
    pub fn new() -> Self {
        Self {
            response_cache: HashMap::new(),
            latency_measurements: Vec::new(),
            max_cache_size: 100,
            cache_hits: 0,
            total_requests: 0,
        }
    }

    /// Check cache for a similar audio/text combination
    pub fn check_cache(&mut self, audio: &AudioBuffer, text: &str) -> Option<FeedbackResponse> {
        self.total_requests += 1;

        let cache_key = self.generate_cache_key(audio, text);

        if let Some(cached) = self.response_cache.get(&cache_key) {
            // Check if cache entry is still valid (within 1 minute)
            if cached.timestamp.elapsed() < Duration::from_secs(60) {
                self.cache_hits += 1;

                // Update timestamp on the response
                let mut response = cached.response.clone();
                response.timestamp = chrono::Utc::now();

                return Some(response);
            } else {
                // Remove expired entry
                self.response_cache.remove(&cache_key);
            }
        }

        None
    }

    /// Cache a response for future use
    pub fn cache_response(&mut self, audio: &AudioBuffer, text: &str, response: FeedbackResponse) {
        let cache_key = self.generate_cache_key(audio, text);

        // Limit cache size
        if self.response_cache.len() >= self.max_cache_size {
            // Remove oldest entry (simple LRU approximation)
            if let Some(oldest_key) = self.response_cache.keys().next().cloned() {
                self.response_cache.remove(&oldest_key);
            }
        }

        self.response_cache.insert(
            cache_key,
            CachedResponse {
                response,
                timestamp: Instant::now(),
            },
        );
    }

    /// Record latency measurement
    pub fn record_latency(&mut self, latency: Duration) {
        self.latency_measurements.push(latency);

        // Keep only recent measurements (last 100)
        if self.latency_measurements.len() > 100 {
            self.latency_measurements.drain(0..50);
        }
    }

    /// Get latency statistics
    pub fn get_stats(&self) -> LatencyStats {
        let avg_latency = if self.latency_measurements.is_empty() {
            Duration::from_millis(0)
        } else {
            let total: Duration = self.latency_measurements.iter().sum();
            total / self.latency_measurements.len() as u32
        };

        let max_latency = self
            .latency_measurements
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::from_millis(0));

        let min_latency = self
            .latency_measurements
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::from_millis(0));

        let cache_hit_rate = if self.total_requests > 0 {
            self.cache_hits as f32 / self.total_requests as f32
        } else {
            0.0
        };

        let p95_latency = if self.latency_measurements.is_empty() {
            Duration::from_millis(0)
        } else {
            let mut sorted = self.latency_measurements.clone();
            sorted.sort();
            let index = (sorted.len() as f32 * 0.95) as usize;
            sorted
                .get(index)
                .copied()
                .unwrap_or(Duration::from_millis(0))
        };

        LatencyStats {
            avg_latency_ms: avg_latency.as_millis() as f32,
            max_latency_ms: max_latency.as_millis() as f32,
            min_latency_ms: min_latency.as_millis() as f32,
            p95_latency_ms: p95_latency.as_millis() as f32,
            cache_hit_rate,
            total_requests: self.total_requests,
            cache_size: self.response_cache.len(),
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.response_cache.clear();
        self.latency_measurements.clear();
        self.cache_hits = 0;
        self.total_requests = 0;
    }

    /// Generate cache key from audio and text
    fn generate_cache_key(&self, audio: &AudioBuffer, text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash audio characteristics (sample rate, length, first/last samples)
        audio.sample_rate().hash(&mut hasher);
        audio.samples().len().hash(&mut hasher);

        // Hash a few samples from beginning and end for audio fingerprint
        if !audio.samples().is_empty() {
            audio.samples()[0].to_bits().hash(&mut hasher);
            if audio.samples().len() > 1 {
                audio.samples()[audio.samples().len() - 1]
                    .to_bits()
                    .hash(&mut hasher);
            }
            if audio.samples().len() > 10 {
                audio.samples()[audio.samples().len() / 2]
                    .to_bits()
                    .hash(&mut hasher);
            }
        }

        // Hash text
        text.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }
}

/// Cached response with timestamp
#[derive(Clone)]
struct CachedResponse {
    response: FeedbackResponse,
    timestamp: Instant,
}

/// Latency performance statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Maximum latency in milliseconds
    pub max_latency_ms: f32,
    /// Minimum latency in milliseconds
    pub min_latency_ms: f32,
    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: f32,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f32,
    /// Total number of requests processed
    pub total_requests: u64,
    /// Current cache size
    pub cache_size: usize,
}
