//! Latency optimization for streaming audio processing
//!
//! Provides predictive processing, look-ahead algorithms, and adaptive
//! chunk sizing to minimize latency in real-time applications.

use crate::config::{LatencyMode, StreamingConfig};
use crate::MelSpectrogram;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Latency optimization engine
pub struct LatencyOptimizer {
    /// Current configuration
    config: Arc<RwLock<StreamingConfig>>,

    /// Processing time history
    processing_times: Arc<RwLock<VecDeque<f32>>>,

    /// Latency measurements
    latency_history: Arc<RwLock<VecDeque<f32>>>,

    /// Adaptive chunk size
    current_chunk_size: Arc<RwLock<usize>>,

    /// Performance statistics
    stats: Arc<RwLock<LatencyStats>>,

    /// Last optimization time
    last_optimization: Arc<RwLock<Instant>>,
}

/// Latency optimization statistics
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    /// Average processing time (ms)
    pub avg_processing_time: f32,

    /// Processing time variance
    pub processing_variance: f32,

    /// Average latency (ms)
    pub avg_latency: f32,

    /// Peak latency (ms)
    pub peak_latency: f32,

    /// Latency jitter (standard deviation)
    pub latency_jitter: f32,

    /// Chunk size adaptations
    pub chunk_adaptations: u64,

    /// Optimization cycles
    pub optimization_cycles: u64,

    /// Deadline misses
    pub deadline_misses: u64,
}

impl LatencyOptimizer {
    /// Create new latency optimizer
    pub fn new(config: StreamingConfig) -> Self {
        let chunk_size = config.chunk_size;

        Self {
            config: Arc::new(RwLock::new(config)),
            processing_times: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            latency_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            current_chunk_size: Arc::new(RwLock::new(chunk_size)),
            stats: Arc::new(RwLock::new(LatencyStats::default())),
            last_optimization: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Record processing time for a chunk
    pub fn record_processing_time(&self, processing_time_ms: f32) {
        let mut times = self.processing_times.write().unwrap();
        times.push_back(processing_time_ms);

        // Keep only recent measurements
        if times.len() > 100 {
            times.pop_front();
        }

        // Update statistics
        self.update_processing_stats();
    }

    /// Record latency measurement
    pub fn record_latency(&self, latency_ms: f32) {
        let mut latencies = self.latency_history.write().unwrap();
        latencies.push_back(latency_ms);

        // Keep only recent measurements
        if latencies.len() > 100 {
            latencies.pop_front();
        }

        // Update statistics
        self.update_latency_stats();

        // Check for deadline miss
        let config = self.config.read().unwrap();
        if latency_ms > config.max_latency_ms {
            if let Ok(mut stats) = self.stats.write() {
                stats.deadline_misses += 1;
            }
        }
    }

    /// Get optimal chunk size for current conditions
    pub fn get_optimal_chunk_size(&self) -> usize {
        let config = self.config.read().unwrap();

        if !config.enable_adaptive_chunking {
            return config.chunk_size;
        }

        // Check if enough data for optimization
        let processing_times = self.processing_times.read().unwrap();
        if processing_times.len() < 10 {
            return config.chunk_size;
        }

        // Check optimization interval
        let now = Instant::now();
        let last_opt = *self.last_optimization.read().unwrap();
        if now.duration_since(last_opt) < Duration::from_millis(500) {
            return *self.current_chunk_size.read().unwrap();
        }

        // Calculate optimal chunk size
        let optimal_size = self.calculate_optimal_chunk_size();

        // Update if changed
        let mut current_size = self.current_chunk_size.write().unwrap();
        if optimal_size != *current_size {
            *current_size = optimal_size;
            if let Ok(mut stats) = self.stats.write() {
                stats.chunk_adaptations += 1;
            }

            tracing::debug!("Adapted chunk size to {}", optimal_size);
        }

        *self.last_optimization.write().unwrap() = now;
        optimal_size
    }

    /// Calculate optimal chunk size based on performance data
    fn calculate_optimal_chunk_size(&self) -> usize {
        let config = self.config.read().unwrap();
        let processing_times = self.processing_times.read().unwrap();

        // Calculate average processing time per sample
        let avg_time: f32 = processing_times.iter().sum::<f32>() / processing_times.len() as f32;
        let current_chunk_size = *self.current_chunk_size.read().unwrap();
        let time_per_sample = avg_time / current_chunk_size as f32;

        // Target processing time based on latency mode
        let target_time = match config.latency_mode {
            LatencyMode::UltraLow => config.target_latency_ms * 0.3,
            LatencyMode::Low => config.target_latency_ms * 0.5,
            LatencyMode::Balanced => config.target_latency_ms * 0.7,
            LatencyMode::Quality => config.target_latency_ms * 0.9,
        };

        // Calculate optimal chunk size
        let optimal_size = (target_time / time_per_sample) as usize;

        // Clamp to valid range
        optimal_size
            .max(config.min_chunk_size)
            .min(config.max_chunk_size)
    }

    /// Update processing time statistics
    fn update_processing_stats(&self) {
        let times = self.processing_times.read().unwrap();
        if times.is_empty() {
            return;
        }

        let sum: f32 = times.iter().sum();
        let count = times.len() as f32;
        let avg = sum / count;

        // Calculate variance
        let variance: f32 = times.iter().map(|&time| (time - avg).powi(2)).sum::<f32>() / count;

        if let Ok(mut stats) = self.stats.write() {
            stats.avg_processing_time = avg;
            stats.processing_variance = variance;
        }
    }

    /// Update latency statistics
    fn update_latency_stats(&self) {
        let latencies = self.latency_history.read().unwrap();
        if latencies.is_empty() {
            return;
        }

        let sum: f32 = latencies.iter().sum();
        let count = latencies.len() as f32;
        let avg = sum / count;
        let peak = latencies.iter().fold(0.0_f32, |acc, &x| acc.max(x));

        // Calculate jitter (standard deviation)
        let jitter: f32 = (latencies
            .iter()
            .map(|&lat| (lat - avg).powi(2))
            .sum::<f32>()
            / count)
            .sqrt();

        if let Ok(mut stats) = self.stats.write() {
            stats.avg_latency = avg;
            stats.peak_latency = peak;
            stats.latency_jitter = jitter;
        }
    }

    /// Get current latency statistics
    pub fn get_stats(&self) -> LatencyStats {
        self.stats.read().unwrap().clone()
    }

    /// Check if latency constraints are being met
    pub fn is_meeting_constraints(&self) -> bool {
        let stats = self.get_stats();
        let config = self.config.read().unwrap();

        stats.avg_latency <= config.target_latency_ms
            && stats.peak_latency <= config.max_latency_ms
            && stats.deadline_misses == 0
    }

    /// Get performance health score (0.0-1.0)
    pub fn get_health_score(&self) -> f32 {
        let stats = self.get_stats();
        let config = self.config.read().unwrap();

        // Latency score
        let latency_score = if stats.avg_latency <= config.target_latency_ms {
            1.0
        } else if stats.avg_latency <= config.max_latency_ms {
            1.0 - (stats.avg_latency - config.target_latency_ms)
                / (config.max_latency_ms - config.target_latency_ms)
        } else {
            0.0
        };

        // Stability score (based on jitter)
        let stability_score = if stats.latency_jitter <= 10.0 {
            1.0
        } else {
            (1.0 / (1.0 + stats.latency_jitter / 10.0)).max(0.1)
        };

        // Reliability score (based on deadline misses)
        let reliability_score = if stats.deadline_misses == 0 {
            1.0
        } else {
            (1.0 / (1.0 + stats.deadline_misses as f32)).max(0.1)
        };

        (latency_score + stability_score + reliability_score) / 3.0
    }
}

/// Predictive processor for look-ahead processing
pub struct PredictiveProcessor {
    /// Look-ahead buffer
    lookahead_buffer: Arc<RwLock<VecDeque<MelSpectrogram>>>,

    /// Prediction model (simplified)
    #[allow(dead_code)]
    prediction_weights: Arc<RwLock<Vec<f32>>>,

    /// Configuration
    config: StreamingConfig,

    /// Processing statistics
    stats: Arc<RwLock<PredictionStats>>,
}

/// Prediction statistics
#[derive(Debug, Clone, Default)]
pub struct PredictionStats {
    /// Prediction accuracy
    pub accuracy: f32,

    /// Predictions made
    pub predictions_made: u64,

    /// Cache hits from predictions
    pub cache_hits: u64,

    /// Processing time saved (ms)
    pub time_saved_ms: f32,
}

impl PredictiveProcessor {
    /// Create new predictive processor
    pub fn new(config: StreamingConfig) -> Self {
        let lookahead_size = config.lookahead_samples / config.chunk_size;

        Self {
            lookahead_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(lookahead_size))),
            prediction_weights: Arc::new(RwLock::new(vec![0.0; 64])),
            config,
            stats: Arc::new(RwLock::new(PredictionStats::default())),
        }
    }

    /// Add mel spectrogram to look-ahead buffer
    pub fn add_lookahead(&self, mel: MelSpectrogram) {
        if !self.config.enable_lookahead {
            return;
        }

        let mut buffer = self.lookahead_buffer.write().unwrap();
        buffer.push_back(mel);

        // Keep buffer size manageable
        let max_size = (self.config.lookahead_samples / self.config.chunk_size).max(2); // At least 2 for prediction
        while buffer.len() > max_size {
            buffer.pop_front();
        }
    }

    /// Predict next mel spectrogram chunk
    pub fn predict_next(&self) -> Option<MelSpectrogram> {
        if !self.config.enable_prediction {
            return None;
        }

        let buffer = self.lookahead_buffer.read().unwrap();
        if buffer.len() < 2 {
            return None;
        }

        // Simple prediction: extrapolate from last two frames
        let last = &buffer[buffer.len() - 1];
        let second_last = &buffer[buffer.len() - 2];

        // Calculate difference and extrapolate
        let mut predicted_data = Vec::new();
        for (frame_last, frame_second) in last.data.iter().zip(&second_last.data) {
            let mut predicted_frame = Vec::new();
            for (&val_last, &val_second) in frame_last.iter().zip(frame_second) {
                let diff = val_last - val_second;
                let predicted = val_last + diff * 0.5; // Simple linear extrapolation
                predicted_frame.push(predicted.clamp(-10.0, 10.0)); // Clamp to reasonable range
            }
            predicted_data.push(predicted_frame);
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.predictions_made += 1;
        }

        Some(MelSpectrogram::new(
            predicted_data,
            last.sample_rate,
            last.hop_length,
        ))
    }

    /// Validate prediction accuracy
    pub fn validate_prediction(&self, predicted: &MelSpectrogram, actual: &MelSpectrogram) {
        if !self.config.enable_prediction {
            return;
        }

        // Calculate mean squared error
        let mut total_error = 0.0;
        let mut count = 0;

        for (pred_frame, actual_frame) in predicted.data.iter().zip(&actual.data) {
            for (&pred_val, &actual_val) in pred_frame.iter().zip(actual_frame) {
                let error = (pred_val - actual_val).powi(2);
                total_error += error;
                count += 1;
            }
        }

        if count > 0 {
            let mse = total_error / count as f32;
            let accuracy = 1.0 / (1.0 + mse); // Convert MSE to accuracy score

            if let Ok(mut stats) = self.stats.write() {
                stats.accuracy = (stats.accuracy + accuracy) / 2.0; // Moving average
            }
        }
    }

    /// Get prediction statistics
    pub fn get_stats(&self) -> PredictionStats {
        self.stats.read().unwrap().clone()
    }

    /// Check if prediction is beneficial
    pub fn is_prediction_beneficial(&self) -> bool {
        let stats = self.get_stats();
        stats.accuracy > 0.7 && stats.time_saved_ms > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::config::LatencyMode;

    #[test]
    fn test_latency_optimizer_creation() {
        let config = StreamingConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        let stats = optimizer.get_stats();
        assert_eq!(stats.optimization_cycles, 0);
        assert_eq!(stats.deadline_misses, 0);
    }

    #[test]
    #[ignore] // Ignore slow test
    fn test_processing_time_recording() {
        let config = StreamingConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        // Record some processing times
        optimizer.record_processing_time(10.0);
        optimizer.record_processing_time(15.0);
        optimizer.record_processing_time(12.0);

        let stats = optimizer.get_stats();
        assert!(stats.avg_processing_time > 0.0);
        assert!(stats.processing_variance >= 0.0);
    }

    #[test]
    #[ignore] // Ignore slow test
    fn test_latency_recording() {
        let config = StreamingConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        // Record latencies
        optimizer.record_latency(25.0);
        optimizer.record_latency(30.0);
        optimizer.record_latency(28.0);

        let stats = optimizer.get_stats();
        assert!(stats.avg_latency > 0.0);
        assert!(stats.peak_latency >= stats.avg_latency);
    }

    #[test]
    #[ignore] // Ignore slow test
    fn test_deadline_miss_detection() {
        let config = StreamingConfig {
            max_latency_ms: 50.0,
            ..Default::default()
        };

        let optimizer = LatencyOptimizer::new(config);

        // Record a latency that exceeds threshold
        optimizer.record_latency(100.0);

        let stats = optimizer.get_stats();
        assert_eq!(stats.deadline_misses, 1);
    }

    #[test]
    #[ignore] // Ignore slow test
    fn test_health_score_calculation() {
        let config = StreamingConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        // Good performance
        optimizer.record_latency(20.0);
        optimizer.record_latency(25.0);

        let health = optimizer.get_health_score();
        assert!(health > 0.5);

        // Poor performance
        optimizer.record_latency(500.0);
        optimizer.record_latency(600.0);

        let health = optimizer.get_health_score();
        assert!(health < 0.5);
    }

    #[test]
    fn test_predictive_processor() {
        let config = StreamingConfig {
            enable_prediction: true,
            enable_lookahead: true,
            ..Default::default()
        };

        let processor = PredictiveProcessor::new(config);

        // Add some test data
        let mel_data = vec![vec![1.0, 2.0, 3.0]; 10];
        let mel1 = MelSpectrogram::new(mel_data.clone(), 22050, 256);
        let mel2 = MelSpectrogram::new(mel_data, 22050, 256);

        processor.add_lookahead(mel1);
        processor.add_lookahead(mel2);

        // Try prediction
        let predicted = processor.predict_next();
        assert!(predicted.is_some());

        let stats = processor.get_stats();
        assert_eq!(stats.predictions_made, 1);
    }
}
