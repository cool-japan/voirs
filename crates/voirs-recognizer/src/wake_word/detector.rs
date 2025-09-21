//! Wake word detector implementation
//!
//! Provides the main wake word detection engine with always-on listening,
//! false positive reduction, and energy-efficient processing.

use super::{
    EnergyOptimizer, WakeWordConfig, WakeWordDetection, WakeWordDetector, WakeWordModel,
    WakeWordStats,
};
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::AudioBuffer;

/// Main wake word detector implementation
pub struct WakeWordDetectorImpl {
    /// Detection configuration
    config: WakeWordConfig,
    /// Detection model
    model: Arc<dyn WakeWordModel + Send + Sync>,
    /// Energy optimizer for battery efficiency
    energy_optimizer: EnergyOptimizer,
    /// Detection statistics
    stats: Arc<Mutex<WakeWordStats>>,
    /// Running state
    is_running: Arc<RwLock<bool>>,
    /// Audio buffer for continuous processing
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    /// Detection history for false positive reduction
    detection_history: Arc<Mutex<VecDeque<WakeWordDetection>>>,
    /// Processing timestamps for rate limiting
    processing_times: Arc<Mutex<VecDeque<Instant>>>,
    /// Session start time
    session_start: Instant,
}

impl WakeWordDetectorImpl {
    /// Create a new wake word detector
    pub async fn new(
        config: WakeWordConfig,
        model: Arc<dyn WakeWordModel + Send + Sync>,
    ) -> Result<Self, RecognitionError> {
        let energy_optimizer = EnergyOptimizer::new(config.energy_saving);

        Ok(Self {
            config,
            model,
            energy_optimizer,
            stats: Arc::new(Mutex::new(WakeWordStats::default())),
            is_running: Arc::new(RwLock::new(false)),
            audio_buffer: Arc::new(Mutex::new(VecDeque::new())),
            detection_history: Arc::new(Mutex::new(VecDeque::new())),
            processing_times: Arc::new(Mutex::new(VecDeque::new())),
            session_start: Instant::now(),
        })
    }

    /// Process audio samples and extract features
    async fn extract_features(&self, audio: &AudioBuffer) -> Result<Vec<f32>, RecognitionError> {
        // Extract MFCC features for wake word detection
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();

        // Simple feature extraction (would use more sophisticated methods in production)
        let features = self.extract_mfcc_features(samples, sample_rate as f32)?;

        Ok(features)
    }

    /// Extract MFCC features from audio samples
    fn extract_mfcc_features(
        &self,
        samples: &[f32],
        sample_rate: f32,
    ) -> Result<Vec<f32>, RecognitionError> {
        // Simplified MFCC extraction for demonstration
        // In production, this would use a proper audio processing library

        const N_MFCC: usize = 13;
        const N_FRAMES: usize = 32;

        // Pre-emphasis filter
        let mut emphasized = Vec::with_capacity(samples.len());
        emphasized.push(samples[0]);
        for i in 1..samples.len() {
            emphasized.push(samples[i] - 0.97 * samples[i - 1]);
        }

        // Frame the signal
        let frame_length = ((sample_rate * 0.025) as usize).min(samples.len()); // 25ms frames
        let frame_stride = ((sample_rate * 0.010) as usize).max(1); // 10ms stride

        let mut features = Vec::new();

        for frame_idx in 0..N_FRAMES {
            let start = frame_idx * frame_stride;
            if start + frame_length >= emphasized.len() {
                break;
            }

            // Extract frame
            let frame = &emphasized[start..start + frame_length];

            // Compute energy (simplified)
            let energy: f32 = frame.iter().map(|x| x * x).sum();
            let log_energy = if energy > 0.0 { energy.ln() } else { -10.0 };

            // Add simplified MFCC coefficients
            for i in 0..N_MFCC {
                let coeff = log_energy * ((i as f32 + 1.0) / N_MFCC as f32).cos();
                features.push(coeff);
            }
        }

        // Pad or truncate to fixed size
        features.resize(N_MFCC * N_FRAMES, 0.0);

        Ok(features)
    }

    /// Check for false positives based on detection history
    async fn is_false_positive(&self, detection: &WakeWordDetection) -> bool {
        let history = self.detection_history.lock().unwrap();

        // Check if too many detections in short time window
        let recent_detections = history
            .iter()
            .filter(|d| detection.timestamp.duration_since(d.timestamp) < Duration::from_secs(5))
            .count();

        if recent_detections > 3 {
            return true;
        }

        // Check confidence pattern (multiple low confidence detections)
        let low_confidence_count = history
            .iter()
            .filter(|d| {
                detection.timestamp.duration_since(d.timestamp) < Duration::from_secs(30)
                    && d.confidence < 0.6
            })
            .count();

        if low_confidence_count > 2 && detection.confidence < 0.7 {
            return true;
        }

        false
    }

    /// Update detection statistics
    async fn update_stats(
        &self,
        detection: &WakeWordDetection,
        processing_time: Duration,
        is_false_positive: bool,
    ) {
        let mut stats = self.stats.lock().unwrap();

        stats.total_detections += 1;

        if is_false_positive {
            stats.false_positives += 1;
        } else {
            stats.true_positives += 1;
        }

        // Update rolling averages
        let total = stats.total_detections as f32;
        stats.avg_confidence =
            (stats.avg_confidence * (total - 1.0) + detection.confidence) / total;

        let processing_ms = processing_time.as_millis() as f32;
        stats.avg_processing_time_ms =
            (stats.avg_processing_time_ms * (total - 1.0) + processing_ms) / total;

        // Update detection rate (detections per hour)
        let session_hours = self.session_start.elapsed().as_secs_f32() / 3600.0;
        if session_hours > 0.0 {
            stats.detection_rate = stats.total_detections as f32 / session_hours;
        }

        // Update energy consumption estimate
        stats.energy_consumption += if self.config.energy_saving { 0.1 } else { 1.0 };
    }

    /// Clean up old detection history
    async fn cleanup_history(&self) {
        let mut history = self.detection_history.lock().unwrap();
        let cutoff = Instant::now() - Duration::from_secs(300); // Keep 5 minutes of history

        while let Some(front) = history.front() {
            if front.timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }

        // Limit history size
        while history.len() > 100 {
            history.pop_front();
        }
    }
}

#[async_trait]
impl WakeWordDetector for WakeWordDetectorImpl {
    async fn start_detection(&mut self) -> Result<(), RecognitionError> {
        let mut running = self.is_running.write().await;
        if *running {
            return Ok(()); // Already running
        }

        // Initialize model
        self.model.initialize().await?;

        // Start energy optimization
        self.energy_optimizer.start_optimization().await?;

        *running = true;

        Ok(())
    }

    async fn stop_detection(&mut self) -> Result<(), RecognitionError> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(()); // Already stopped
        }

        // Stop energy optimization
        self.energy_optimizer.stop_optimization().await?;

        *running = false;

        Ok(())
    }

    async fn detect_wake_words(
        &mut self,
        audio: &AudioBuffer,
    ) -> Result<Vec<WakeWordDetection>, RecognitionError> {
        let is_running = *self.is_running.read().await;
        if !is_running {
            return Ok(Vec::new());
        }

        let processing_start = Instant::now();

        // Apply energy optimization
        if self.energy_optimizer.should_skip_processing().await {
            return Ok(Vec::new());
        }

        // Extract features
        let features = self.extract_features(audio).await?;

        // Run detection through model
        let model_results = self.model.detect(&features).await?;

        let mut detections = Vec::new();

        for result in model_results {
            // Check confidence threshold
            if result.confidence < self.config.min_confidence {
                continue;
            }

            // Check if wake word is in our list
            if !self.config.wake_words.contains(&result.word) {
                continue;
            }

            let detection = WakeWordDetection {
                word: result.word.clone(),
                confidence: result.confidence,
                timestamp: Instant::now(),
                start_time: Duration::from_secs_f32(result.start_time),
                end_time: Duration::from_secs_f32(result.end_time),
                false_positive_prob: result.false_positive_prob,
            };

            // Check for false positives
            let is_fp = self.is_false_positive(&detection).await;

            if !is_fp {
                detections.push(detection.clone());
            }

            // Update statistics
            let processing_time = processing_start.elapsed();
            self.update_stats(&detection, processing_time, is_fp).await;

            // Add to history
            {
                let mut history = self.detection_history.lock().unwrap();
                history.push_back(detection);
            }
        }

        // Clean up old history
        self.cleanup_history().await;

        // Update energy optimizer with processing results
        self.energy_optimizer
            .update_processing_result(processing_start.elapsed(), !detections.is_empty())
            .await;

        Ok(detections)
    }

    async fn add_wake_word(&mut self, word: &str) -> Result<(), RecognitionError> {
        if !self.config.wake_words.contains(&word.to_string()) {
            self.config.wake_words.push(word.to_string());

            // Update model if it supports dynamic vocabulary
            if let Err(e) = self.model.add_word(word).await {
                // Remove from config if model update failed
                self.config.wake_words.retain(|w| w != word);
                return Err(e);
            }
        }

        Ok(())
    }

    async fn remove_wake_word(&mut self, word: &str) -> Result<(), RecognitionError> {
        self.config.wake_words.retain(|w| w != word);
        self.model.remove_word(word).await?;
        Ok(())
    }

    async fn update_config(&mut self, config: WakeWordConfig) -> Result<(), RecognitionError> {
        let old_energy_setting = self.config.energy_saving;
        self.config = config;

        // Update energy optimizer if setting changed
        if old_energy_setting != self.config.energy_saving {
            self.energy_optimizer
                .set_energy_saving(self.config.energy_saving)
                .await;
        }

        Ok(())
    }

    async fn get_statistics(&self) -> Result<WakeWordStats, RecognitionError> {
        let stats = self.stats.lock().unwrap();
        Ok(stats.clone())
    }

    fn is_running(&self) -> bool {
        // Use try_read to avoid blocking
        self.is_running
            .try_read()
            .map(|guard| *guard)
            .unwrap_or(false)
    }

    fn get_supported_words(&self) -> Vec<String> {
        self.config.wake_words.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wake_word::models::MockWakeWordModel;

    #[tokio::test]
    async fn test_wake_word_detector_creation() {
        let config = WakeWordConfig::default();
        let model = Arc::new(MockWakeWordModel::new());

        let detector = WakeWordDetectorImpl::new(config, model).await;
        assert!(detector.is_ok());

        let detector = detector.unwrap();
        assert!(!detector.is_running());
        assert_eq!(detector.get_supported_words().len(), 3);
    }

    #[tokio::test]
    async fn test_start_stop_detection() {
        let config = WakeWordConfig::default();
        let model = Arc::new(MockWakeWordModel::new());

        let mut detector = WakeWordDetectorImpl::new(config, model).await.unwrap();

        assert!(!detector.is_running());

        detector.start_detection().await.unwrap();
        assert!(detector.is_running());

        detector.stop_detection().await.unwrap();
        assert!(!detector.is_running());
    }

    #[tokio::test]
    async fn test_add_remove_wake_words() {
        let config = WakeWordConfig::default();
        let model = Arc::new(MockWakeWordModel::new());

        let mut detector = WakeWordDetectorImpl::new(config, model).await.unwrap();

        let initial_count = detector.get_supported_words().len();

        detector.add_wake_word("test").await.unwrap();
        assert_eq!(detector.get_supported_words().len(), initial_count + 1);
        assert!(detector.get_supported_words().contains(&"test".to_string()));

        detector.remove_wake_word("test").await.unwrap();
        assert_eq!(detector.get_supported_words().len(), initial_count);
        assert!(!detector.get_supported_words().contains(&"test".to_string()));
    }

    #[test]
    fn test_mfcc_feature_extraction() {
        let config = WakeWordConfig::default();
        let model = Arc::new(MockWakeWordModel::new());

        // Create a simple test signal
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let detector =
            rt.block_on(async { WakeWordDetectorImpl::new(config, model).await.unwrap() });

        let features = detector.extract_mfcc_features(&samples, 16000.0);
        assert!(features.is_ok());

        let features = features.unwrap();
        assert_eq!(features.len(), 13 * 32); // 13 MFCC coefficients * 32 frames
    }
}
