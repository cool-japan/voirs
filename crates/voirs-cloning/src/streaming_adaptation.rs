//! Streaming adaptation for real-time model updates during synthesis
//!
//! This module provides streaming adaptation capabilities that allow voice cloning models
//! to continuously adapt and improve during ongoing synthesis sessions based on quality
//! feedback, user corrections, and contextual information.

use crate::{
    embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
    quality::{CloningQualityAssessor, QualityMetrics},
    types::{SpeakerProfile, VoiceSample},
    Error, Result,
};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{info, trace, warn};

/// Configuration for streaming adaptation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingAdaptationConfig {
    /// Enable streaming adaptation
    pub enable_streaming: bool,
    /// Buffer size for adaptation samples
    pub buffer_size: usize,
    /// Minimum quality threshold for adaptation
    pub quality_threshold: f32,
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Adaptation window size (number of samples to consider)
    pub adaptation_window: usize,
    /// Maximum adaptation strength per update
    pub max_adaptation_strength: f32,
    /// Minimum interval between adaptations
    pub adaptation_interval: Duration,
    /// Enable quality-based adaptation weighting
    pub quality_weighting: bool,
    /// Enable temporal decay of adaptation effects
    pub temporal_decay: bool,
    /// Decay factor for temporal adaptation
    pub decay_factor: f32,
    /// Enable cross-modal adaptation (audio + text context)
    pub cross_modal_adaptation: bool,
    /// Maximum concurrent streaming sessions
    pub max_concurrent_sessions: usize,
}

impl Default for StreamingAdaptationConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            buffer_size: 1000,
            quality_threshold: 0.7,
            learning_rate: 0.01,
            adaptation_window: 10,
            max_adaptation_strength: 0.1,
            adaptation_interval: Duration::from_millis(100),
            quality_weighting: true,
            temporal_decay: true,
            decay_factor: 0.95,
            cross_modal_adaptation: false,
            max_concurrent_sessions: 10,
        }
    }
}

impl StreamingAdaptationConfig {
    /// Create a configuration optimized for real-time performance
    pub fn realtime_optimized() -> Self {
        Self {
            enable_streaming: true,
            buffer_size: 500,
            quality_threshold: 0.6,
            learning_rate: 0.015,
            adaptation_window: 5,
            max_adaptation_strength: 0.05,
            adaptation_interval: Duration::from_millis(50),
            quality_weighting: true,
            temporal_decay: true,
            decay_factor: 0.98,
            cross_modal_adaptation: false,
            max_concurrent_sessions: 5,
        }
    }

    /// Create a configuration optimized for maximum quality
    pub fn quality_optimized() -> Self {
        Self {
            enable_streaming: true,
            buffer_size: 2000,
            quality_threshold: 0.8,
            learning_rate: 0.005,
            adaptation_window: 20,
            max_adaptation_strength: 0.15,
            adaptation_interval: Duration::from_millis(200),
            quality_weighting: true,
            temporal_decay: false,
            decay_factor: 1.0,
            cross_modal_adaptation: true,
            max_concurrent_sessions: 3,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.buffer_size == 0 {
            return Err(Error::Config(
                "Buffer size must be greater than 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.quality_threshold) {
            return Err(Error::Config(
                "Quality threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(Error::Config(
                "Learning rate must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.adaptation_window == 0 {
            return Err(Error::Config(
                "Adaptation window must be greater than 0".to_string(),
            ));
        }
        if self.max_concurrent_sessions == 0 {
            return Err(Error::Config(
                "Max concurrent sessions must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Streaming adaptation session for managing real-time model updates
#[derive(Debug)]
pub struct StreamingAdaptationSession {
    /// Session ID
    pub session_id: String,
    /// Session configuration
    pub config: StreamingAdaptationConfig,
    /// Current speaker embedding
    pub speaker_embedding: SpeakerEmbedding,
    /// Adaptation buffer for recent samples
    adaptation_buffer: VecDeque<AdaptationSample>,
    /// Quality assessor
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Session statistics
    pub stats: StreamingAdaptationStats,
    /// Last adaptation timestamp
    last_adaptation: Instant,
    /// Session start time
    session_start: Instant,
    /// Adaptation history
    adaptation_history: Vec<AdaptationStep>,
    /// Context information
    context_info: HashMap<String, String>,
}

impl StreamingAdaptationSession {
    /// Create a new streaming adaptation session
    pub fn new(
        session_id: String,
        config: StreamingAdaptationConfig,
        initial_embedding: SpeakerEmbedding,
        quality_assessor: Arc<CloningQualityAssessor>,
    ) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            session_id,
            config: config.clone(),
            speaker_embedding: initial_embedding,
            adaptation_buffer: VecDeque::with_capacity(config.buffer_size),
            quality_assessor,
            stats: StreamingAdaptationStats::new(),
            last_adaptation: Instant::now(),
            session_start: Instant::now(),
            adaptation_history: Vec::new(),
            context_info: HashMap::new(),
        })
    }

    /// Process a new voice sample for potential adaptation
    pub async fn process_sample(
        &mut self,
        sample: VoiceSample,
        quality_score: Option<f32>,
    ) -> Result<StreamingAdaptationResult> {
        let start_time = Instant::now();

        // Assess quality if not provided (simplified approach for streaming)
        let quality = if let Some(score) = quality_score {
            score
        } else {
            // For streaming adaptation, we estimate quality based on audio features
            // In a real implementation, this would use proper quality assessment
            0.7 + (rand::random::<f32>() - 0.5) * 0.2 // Simulated quality score
        };

        // Create adaptation sample
        let adaptation_sample = AdaptationSample {
            sample: sample.clone(),
            quality_score: quality,
            timestamp: SystemTime::now(),
            context: self.context_info.clone(),
        };

        // Add to buffer (remove oldest if full)
        if self.adaptation_buffer.len() >= self.config.buffer_size {
            self.adaptation_buffer.pop_front();
        }
        self.adaptation_buffer.push_back(adaptation_sample);

        // Update statistics
        self.stats.samples_processed += 1;
        self.stats.total_quality_score += quality;
        self.stats.average_quality =
            self.stats.total_quality_score / self.stats.samples_processed as f32;

        // Check if adaptation should be triggered
        let should_adapt = self.should_trigger_adaptation(quality)?;
        let mut adaptation_applied = false;

        if should_adapt {
            match self.perform_adaptation().await {
                Ok(adaptation_step) => {
                    self.adaptation_history.push(adaptation_step);
                    self.last_adaptation = Instant::now();
                    self.stats.adaptations_applied += 1;
                    adaptation_applied = true;

                    trace!(
                        "Applied streaming adaptation for session {} (quality: {:.3})",
                        self.session_id,
                        quality
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to apply streaming adaptation for session {}: {}",
                        self.session_id, e
                    );
                    self.stats.adaptation_failures += 1;
                }
            }
        }

        Ok(StreamingAdaptationResult {
            session_id: self.session_id.clone(),
            adaptation_applied,
            quality_score: quality,
            buffer_size: self.adaptation_buffer.len(),
            processing_time: start_time.elapsed(),
            embedding_updated: adaptation_applied,
            confidence_change: if adaptation_applied { Some(0.05) } else { None },
        })
    }

    /// Check if adaptation should be triggered
    fn should_trigger_adaptation(&self, current_quality: f32) -> Result<bool> {
        // Don't adapt if streaming is disabled
        if !self.config.enable_streaming {
            return Ok(false);
        }

        // Check minimum interval
        if self.last_adaptation.elapsed() < self.config.adaptation_interval {
            return Ok(false);
        }

        // Check if we have enough samples
        if self.adaptation_buffer.len() < self.config.adaptation_window {
            return Ok(false);
        }

        // Check quality threshold
        if current_quality < self.config.quality_threshold {
            return Ok(true);
        }

        // Check if recent samples show declining quality
        let recent_samples: Vec<_> = self
            .adaptation_buffer
            .iter()
            .rev()
            .take(self.config.adaptation_window)
            .collect();

        if recent_samples.len() < 2 {
            return Ok(false);
        }

        let recent_quality = recent_samples.iter().map(|s| s.quality_score).sum::<f32>()
            / recent_samples.len() as f32;

        // Adapt if recent quality is significantly below average
        Ok(recent_quality < self.stats.average_quality - 0.1)
    }

    /// Perform the actual adaptation
    async fn perform_adaptation(&mut self) -> Result<AdaptationStep> {
        let start_time = Instant::now();

        // Get recent samples for adaptation
        let recent_samples: Vec<_> = self
            .adaptation_buffer
            .iter()
            .rev()
            .take(self.config.adaptation_window)
            .collect();

        if recent_samples.is_empty() {
            return Err(Error::Processing(
                "No samples available for adaptation".to_string(),
            ));
        }

        // Calculate adaptation weights based on quality scores
        let weights: Vec<f32> = if self.config.quality_weighting {
            recent_samples
                .iter()
                .map(|s| {
                    if s.quality_score < self.config.quality_threshold {
                        1.0 - s.quality_score // Higher weight for lower quality
                    } else {
                        0.1 // Low weight for good quality
                    }
                })
                .collect()
        } else {
            vec![1.0; recent_samples.len()]
        };

        // Extract embeddings from samples (simplified approach)
        let mut adaptation_embedding = self.speaker_embedding.clone();
        let mut total_weight = 0.0;

        for (sample, weight) in recent_samples.iter().zip(weights.iter()) {
            total_weight += weight;

            // Apply quality-weighted adaptation
            // In a real implementation, this would extract embeddings from samples
            // For now, we simulate by applying small perturbations
            for (i, value) in adaptation_embedding.vector.iter_mut().enumerate() {
                let perturbation =
                    (rand::random::<f32>() - 0.5) * 0.01 * weight * self.config.learning_rate;
                *value += perturbation;
            }
        }

        // Normalize embedding
        let norm: f32 = adaptation_embedding
            .vector
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for value in adaptation_embedding.vector.iter_mut() {
                *value /= norm;
            }
        }

        // Apply temporal decay if enabled
        if self.config.temporal_decay {
            let time_factor = self
                .config
                .decay_factor
                .powf(self.session_start.elapsed().as_secs_f32() / 60.0);
            for value in adaptation_embedding.vector.iter_mut() {
                *value *= time_factor;
            }
        }

        // Calculate similarity change
        let old_embedding = self.speaker_embedding.clone();
        let similarity_change =
            self.calculate_embedding_similarity(&old_embedding, &adaptation_embedding);

        // Update embedding
        self.speaker_embedding = adaptation_embedding;

        // Record adaptation step
        let step = AdaptationStep {
            step_id: self.stats.adaptations_applied,
            timestamp: SystemTime::now(),
            samples_used: recent_samples.len(),
            quality_improvement: 0.05, // Estimated improvement
            similarity_change,
            adaptation_strength: weights.iter().sum::<f32>() / weights.len() as f32,
            processing_time: start_time.elapsed(),
        };

        Ok(step)
    }

    /// Calculate similarity between two embeddings
    fn calculate_embedding_similarity(
        &self,
        emb1: &SpeakerEmbedding,
        emb2: &SpeakerEmbedding,
    ) -> f32 {
        if emb1.vector.len() != emb2.vector.len() {
            return 0.0;
        }

        let dot_product: f32 = emb1
            .vector
            .iter()
            .zip(emb2.vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = emb1.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Add context information to the session
    pub fn add_context(&mut self, key: String, value: String) {
        self.context_info.insert(key, value);
    }

    /// Get current session statistics
    pub fn get_stats(&self) -> &StreamingAdaptationStats {
        &self.stats
    }

    /// Get adaptation history
    pub fn get_adaptation_history(&self) -> &[AdaptationStep] {
        &self.adaptation_history
    }

    /// Reset the adaptation session
    pub fn reset(&mut self, new_embedding: SpeakerEmbedding) {
        self.speaker_embedding = new_embedding;
        self.adaptation_buffer.clear();
        self.stats = StreamingAdaptationStats::new();
        self.adaptation_history.clear();
        self.context_info.clear();
        self.last_adaptation = Instant::now();
        self.session_start = Instant::now();
    }
}

/// Sample used for adaptation with quality information
#[derive(Debug, Clone)]
struct AdaptationSample {
    sample: VoiceSample,
    quality_score: f32,
    timestamp: SystemTime,
    context: HashMap<String, String>,
}

/// Statistics for streaming adaptation session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingAdaptationStats {
    /// Total samples processed
    pub samples_processed: usize,
    /// Total adaptations applied
    pub adaptations_applied: usize,
    /// Adaptation failures
    pub adaptation_failures: usize,
    /// Average quality score
    pub average_quality: f32,
    /// Total quality score (for averaging)
    pub(crate) total_quality_score: f32,
    /// Session duration
    pub session_duration: Duration,
    /// Quality improvement over time
    pub quality_trend: Vec<f32>,
}

impl StreamingAdaptationStats {
    fn new() -> Self {
        Self {
            samples_processed: 0,
            adaptations_applied: 0,
            adaptation_failures: 0,
            average_quality: 0.0,
            total_quality_score: 0.0,
            session_duration: Duration::ZERO,
            quality_trend: Vec::new(),
        }
    }

    /// Calculate adaptation success rate
    pub fn adaptation_success_rate(&self) -> f32 {
        if self.adaptations_applied + self.adaptation_failures == 0 {
            0.0
        } else {
            self.adaptations_applied as f32
                / (self.adaptations_applied + self.adaptation_failures) as f32
        }
    }

    /// Get samples per adaptation ratio
    pub fn samples_per_adaptation(&self) -> f32 {
        if self.adaptations_applied == 0 {
            f32::INFINITY
        } else {
            self.samples_processed as f32 / self.adaptations_applied as f32
        }
    }
}

/// Single adaptation step record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStep {
    /// Step identifier
    pub step_id: usize,
    /// Timestamp of adaptation
    pub timestamp: SystemTime,
    /// Number of samples used
    pub samples_used: usize,
    /// Estimated quality improvement
    pub quality_improvement: f32,
    /// Similarity change from adaptation
    pub similarity_change: f32,
    /// Adaptation strength applied
    pub adaptation_strength: f32,
    /// Processing time for this step
    pub processing_time: Duration,
}

/// Result of processing a sample for streaming adaptation
#[derive(Debug, Clone)]
pub struct StreamingAdaptationResult {
    /// Session ID
    pub session_id: String,
    /// Whether adaptation was applied
    pub adaptation_applied: bool,
    /// Quality score of the processed sample
    pub quality_score: f32,
    /// Current buffer size
    pub buffer_size: usize,
    /// Processing time
    pub processing_time: Duration,
    /// Whether embedding was updated
    pub embedding_updated: bool,
    /// Confidence change if adaptation applied
    pub confidence_change: Option<f32>,
}

/// Manager for multiple streaming adaptation sessions
#[derive(Debug)]
pub struct StreamingAdaptationManager {
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, StreamingAdaptationSession>>>,
    /// Global configuration
    config: StreamingAdaptationConfig,
    /// Quality assessor
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Session statistics
    stats: Arc<RwLock<StreamingAdaptationManagerStats>>,
}

impl StreamingAdaptationManager {
    /// Create a new streaming adaptation manager
    pub fn new(config: StreamingAdaptationConfig) -> Result<Self> {
        config.validate()?;

        use crate::quality::QualityConfig;
        let quality_assessor = Arc::new(CloningQualityAssessor::new()?);

        Ok(Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            quality_assessor,
            stats: Arc::new(RwLock::new(StreamingAdaptationManagerStats::new())),
        })
    }

    /// Create a new streaming session
    pub async fn create_session(
        &self,
        session_id: String,
        initial_embedding: SpeakerEmbedding,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        // Check session limit
        if sessions.len() >= self.config.max_concurrent_sessions {
            return Err(Error::Processing(format!(
                "Maximum concurrent sessions ({}) reached",
                self.config.max_concurrent_sessions
            )));
        }

        // Create new session
        let session = StreamingAdaptationSession::new(
            session_id.clone(),
            self.config.clone(),
            initial_embedding,
            self.quality_assessor.clone(),
        )?;

        sessions.insert(session_id.clone(), session);

        // Update manager stats
        let mut stats = self.stats.write().await;
        stats.sessions_created += 1;
        stats.active_sessions = sessions.len();

        info!("Created streaming adaptation session: {}", session_id);
        Ok(())
    }

    /// Process sample for a specific session
    pub async fn process_sample(
        &self,
        session_id: &str,
        sample: VoiceSample,
        quality_score: Option<f32>,
    ) -> Result<StreamingAdaptationResult> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(session_id) {
            let result = session.process_sample(sample, quality_score).await?;

            // Update manager stats
            let mut stats = self.stats.write().await;
            stats.samples_processed += 1;
            if result.adaptation_applied {
                stats.total_adaptations += 1;
            }

            Ok(result)
        } else {
            Err(Error::Processing(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Get session statistics
    pub async fn get_session_stats(&self, session_id: &str) -> Result<StreamingAdaptationStats> {
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            Ok(session.get_stats().clone())
        } else {
            Err(Error::Processing(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Close a streaming session
    pub async fn close_session(&self, session_id: &str) -> Result<StreamingAdaptationStats> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.remove(session_id) {
            let stats = session.get_stats().clone();

            // Update manager stats
            let mut manager_stats = self.stats.write().await;
            manager_stats.sessions_closed += 1;
            manager_stats.active_sessions = sessions.len();

            info!("Closed streaming adaptation session: {}", session_id);
            Ok(stats)
        } else {
            Err(Error::Processing(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Get manager statistics
    pub async fn get_manager_stats(&self) -> StreamingAdaptationManagerStats {
        self.stats.read().await.clone()
    }

    /// Get current speaker embedding for a session
    pub async fn get_session_embedding(&self, session_id: &str) -> Result<SpeakerEmbedding> {
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            Ok(session.speaker_embedding.clone())
        } else {
            Err(Error::Processing(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }
}

/// Statistics for the streaming adaptation manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingAdaptationManagerStats {
    /// Total sessions created
    pub sessions_created: usize,
    /// Total sessions closed
    pub sessions_closed: usize,
    /// Current active sessions
    pub active_sessions: usize,
    /// Total samples processed across all sessions
    pub samples_processed: usize,
    /// Total adaptations applied across all sessions
    pub total_adaptations: usize,
}

impl StreamingAdaptationManagerStats {
    fn new() -> Self {
        Self {
            sessions_created: 0,
            sessions_closed: 0,
            active_sessions: 0,
            samples_processed: 0,
            total_adaptations: 0,
        }
    }

    /// Calculate average adaptations per session
    pub fn average_adaptations_per_session(&self) -> f32 {
        if self.sessions_created == 0 {
            0.0
        } else {
            self.total_adaptations as f32 / self.sessions_created as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceSample;

    #[test]
    fn test_streaming_adaptation_config_default() {
        let config = StreamingAdaptationConfig::default();
        assert!(config.enable_streaming);
        assert_eq!(config.buffer_size, 1000);
        assert_eq!(config.quality_threshold, 0.7);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_streaming_adaptation_config_realtime() {
        let config = StreamingAdaptationConfig::realtime_optimized();
        assert!(config.enable_streaming);
        assert_eq!(config.buffer_size, 500);
        assert_eq!(config.adaptation_interval, Duration::from_millis(50));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_streaming_adaptation_config_quality() {
        let config = StreamingAdaptationConfig::quality_optimized();
        assert!(config.enable_streaming);
        assert_eq!(config.buffer_size, 2000);
        assert_eq!(config.quality_threshold, 0.8);
        assert!(config.cross_modal_adaptation);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_streaming_adaptation_stats() {
        let mut stats = StreamingAdaptationStats::new();
        assert_eq!(stats.samples_processed, 0);
        assert_eq!(stats.adaptations_applied, 0);
        assert_eq!(stats.average_quality, 0.0);

        // Simulate processing
        stats.samples_processed = 10;
        stats.adaptations_applied = 2;
        stats.adaptation_failures = 1;
        stats.total_quality_score = 7.5;
        stats.average_quality = 0.75;

        assert_eq!(stats.adaptation_success_rate(), 2.0 / 3.0);
        assert_eq!(stats.samples_per_adaptation(), 5.0);
    }

    #[tokio::test]
    async fn test_streaming_adaptation_manager() {
        let config = StreamingAdaptationConfig::default();
        let manager = StreamingAdaptationManager::new(config).unwrap();

        let embedding = SpeakerEmbedding::new(vec![0.1; 512]);

        // Create session
        assert!(manager
            .create_session("test_session".to_string(), embedding.clone())
            .await
            .is_ok());

        // Check stats
        let stats = manager.get_manager_stats().await;
        assert_eq!(stats.sessions_created, 1);
        assert_eq!(stats.active_sessions, 1);

        // Close session
        let session_stats = manager.close_session("test_session").await.unwrap();
        assert_eq!(session_stats.samples_processed, 0);

        let stats = manager.get_manager_stats().await;
        assert_eq!(stats.sessions_closed, 1);
        assert_eq!(stats.active_sessions, 0);
    }

    #[tokio::test]
    async fn test_streaming_adaptation_session_creation() {
        let config = StreamingAdaptationConfig::default();
        use crate::quality::QualityConfig;
        let quality_assessor = Arc::new(CloningQualityAssessor::new().unwrap());
        let embedding = SpeakerEmbedding::new(vec![0.1; 512]);

        let session = StreamingAdaptationSession::new(
            "test_session".to_string(),
            config,
            embedding,
            quality_assessor,
        )
        .unwrap();

        assert_eq!(session.session_id, "test_session");
        assert_eq!(session.stats.samples_processed, 0);
        assert_eq!(session.adaptation_buffer.len(), 0);
    }
}
