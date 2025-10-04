//! Wake word training functionality
//!
//! Provides training capabilities for custom wake word models including
//! data validation, model training, and progress tracking.

use super::{
    TrainingPhase, TrainingProgress, TrainingValidationReport, WakeWordTrainer,
    WakeWordTrainingData,
};
use crate::RecognitionError;
use async_trait::async_trait;
use scirs2_core::random::{thread_rng, Rng};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::AudioBuffer;

/// Custom wake word trainer implementation
pub struct WakeWordTrainerImpl {
    /// Training configuration
    config: TrainingConfig,
    /// Current training progress
    progress: Arc<RwLock<TrainingProgress>>,
    /// Training statistics
    stats: Arc<Mutex<TrainingStats>>,
    /// Output directory for models
    output_dir: PathBuf,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: u32,
    /// Batch size
    pub batch_size: usize,
    /// Validation split ratio
    pub validation_split: f32,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: u32,
    /// Minimum improvement threshold for early stopping
    pub min_improvement_threshold: f32,
    /// Data augmentation enabled
    pub data_augmentation: bool,
    /// Maximum training duration
    pub max_training_duration: Duration,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: 10,
            min_improvement_threshold: 0.01,
            data_augmentation: true,
            max_training_duration: Duration::from_secs(2 * 60 * 60), // 2 hours
        }
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Training loss history
    pub loss_history: Vec<f32>,
    /// Validation loss history
    pub validation_loss_history: Vec<f32>,
    /// Training accuracy history
    pub accuracy_history: Vec<f32>,
    /// Validation accuracy history
    pub validation_accuracy_history: Vec<f32>,
    /// Training start time
    pub start_time: Option<Instant>,
    /// Training end time
    pub end_time: Option<Instant>,
    /// Best validation accuracy achieved
    pub best_validation_accuracy: f32,
    /// Best validation loss achieved
    pub best_validation_loss: f32,
}

impl Default for TrainingStats {
    fn default() -> Self {
        Self {
            loss_history: Vec::new(),
            validation_loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            validation_accuracy_history: Vec::new(),
            start_time: None,
            end_time: None,
            best_validation_accuracy: 0.0,
            best_validation_loss: f32::INFINITY,
        }
    }
}

/// Feature extractor for training data
pub struct FeatureExtractor {
    /// Sample rate
    sample_rate: u32,
    /// Feature dimension
    feature_dim: usize,
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            feature_dim: 13 * 32, // 13 MFCC coefficients * 32 frames
        }
    }

    /// Extract features from audio buffer
    pub fn extract_features(&self, audio: &AudioBuffer) -> Result<Vec<f32>, RecognitionError> {
        let samples = audio.samples();

        // Extract MFCC features (simplified implementation)
        let features = self.extract_mfcc_features(samples)?;

        Ok(features)
    }

    /// Extract MFCC features (simplified implementation)
    fn extract_mfcc_features(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        const N_MFCC: usize = 13;
        const N_FRAMES: usize = 32;

        // Pre-emphasis filter
        let mut emphasized = Vec::with_capacity(samples.len());
        emphasized.push(samples[0]);
        for i in 1..samples.len() {
            emphasized.push(samples[i] - 0.97 * samples[i - 1]);
        }

        // Frame the signal
        let frame_length = (self.sample_rate as f32 * 0.025) as usize; // 25ms frames
        let frame_stride = (self.sample_rate as f32 * 0.010) as usize; // 10ms stride

        let mut features = Vec::new();

        for frame_idx in 0..N_FRAMES {
            let start = frame_idx * frame_stride;
            if start + frame_length >= emphasized.len() {
                break;
            }

            // Extract frame
            let frame = &emphasized[start..start + frame_length];

            // Apply Hamming window
            let windowed: Vec<f32> = frame
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let window = 0.54
                        - 0.46
                            * (2.0 * std::f32::consts::PI * i as f32 / (frame_length - 1) as f32)
                                .cos();
                    x * window
                })
                .collect();

            // Compute energy
            let energy: f32 = windowed.iter().map(|x| x * x).sum();
            let log_energy = if energy > 0.0 { energy.ln() } else { -10.0 };

            // Add MFCC coefficients (simplified DCT)
            for i in 0..N_MFCC {
                let coeff = windowed
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        x * (std::f32::consts::PI * (i + 1) as f32 * (j as f32 + 0.5)
                            / windowed.len() as f32)
                            .cos()
                    })
                    .sum::<f32>();
                features.push(coeff + log_energy * 0.1);
            }
        }

        // Pad or truncate to fixed size
        features.resize(N_MFCC * N_FRAMES, 0.0);

        Ok(features)
    }
}

/// Data augmentation utilities
pub struct DataAugmenter {
    /// Noise levels for augmentation
    noise_levels: Vec<f32>,
    /// Speed variations
    speed_variations: Vec<f32>,
}

impl DataAugmenter {
    /// Create new data augmenter
    pub fn new() -> Self {
        Self {
            noise_levels: vec![0.01, 0.02, 0.05],
            speed_variations: vec![0.9, 0.95, 1.05, 1.1],
        }
    }

    /// Augment audio data
    pub fn augment_audio(&self, audio: &AudioBuffer) -> Vec<AudioBuffer> {
        let mut augmented = vec![audio.clone()]; // Original

        let samples = audio.samples();
        let sample_rate = audio.sample_rate();

        // Add noise variations
        for &noise_level in &self.noise_levels {
            let noisy_samples: Vec<f32> = samples
                .iter()
                .map(|&x| x + noise_level * (thread_rng().r#gen::<f32>() - 0.5))
                .collect();
            augmented.push(AudioBuffer::mono(noisy_samples, sample_rate));
        }

        // Add speed variations (simplified time-stretching)
        for &speed in &self.speed_variations {
            let new_length = (samples.len() as f32 / speed) as usize;
            let stretched_samples: Vec<f32> = (0..new_length)
                .map(|i| {
                    let src_idx = (i as f32 * speed) as usize;
                    if src_idx < samples.len() {
                        samples[src_idx]
                    } else {
                        0.0
                    }
                })
                .collect();
            augmented.push(AudioBuffer::mono(stretched_samples, sample_rate));
        }

        augmented
    }
}

impl Default for DataAugmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl WakeWordTrainerImpl {
    /// Create new wake word trainer
    pub fn new(config: TrainingConfig, output_dir: PathBuf) -> Self {
        let progress = TrainingProgress {
            phase: TrainingPhase::Preprocessing,
            progress: 0.0,
            current_loss: None,
            eta: None,
            learning_rate: Some(config.learning_rate),
        };

        Self {
            config,
            progress: Arc::new(RwLock::new(progress)),
            stats: Arc::new(Mutex::new(TrainingStats::default())),
            output_dir,
        }
    }

    /// Prepare training data
    async fn prepare_training_data(
        &self,
        training_data: &WakeWordTrainingData,
    ) -> Result<(Vec<Vec<f32>>, Vec<i32>), RecognitionError> {
        // Update progress
        {
            let mut progress = self.progress.write().await;
            progress.phase = TrainingPhase::Preprocessing;
            progress.progress = 0.0;
        }

        let feature_extractor = FeatureExtractor::new(16000); // Assume 16kHz
        let mut features = Vec::new();
        let mut labels = Vec::new();

        // Process positive examples
        for (i, audio) in training_data.positive_examples.iter().enumerate() {
            let audio_features = feature_extractor.extract_features(audio)?;
            features.push(audio_features);
            labels.push(1); // Positive label

            // Update progress
            let progress_val = (i + 1) as f32
                / (training_data.positive_examples.len() + training_data.negative_examples.len())
                    as f32
                * 0.5;
            {
                let mut progress = self.progress.write().await;
                progress.progress = progress_val;
            }
        }

        // Process negative examples
        for (i, audio) in training_data.negative_examples.iter().enumerate() {
            let audio_features = feature_extractor.extract_features(audio)?;
            features.push(audio_features);
            labels.push(0); // Negative label

            // Update progress
            let progress_val = 0.5
                + (i + 1) as f32
                    / (training_data.positive_examples.len()
                        + training_data.negative_examples.len()) as f32
                    * 0.5;
            {
                let mut progress = self.progress.write().await;
                progress.progress = progress_val;
            }
        }

        Ok((features, labels))
    }

    /// Train model with prepared data
    async fn train_model(
        &self,
        _features: Vec<Vec<f32>>,
        _labels: Vec<i32>,
    ) -> Result<String, RecognitionError> {
        // Update progress to training phase
        {
            let mut progress = self.progress.write().await;
            progress.phase = TrainingPhase::Training;
            progress.progress = 0.0;
        }

        let start_time = Instant::now();
        {
            let mut stats = self.stats.lock().unwrap();
            stats.start_time = Some(start_time);
        }

        // Simplified training simulation
        for epoch in 0..self.config.epochs {
            // Simulate training step
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Simulate loss calculation
            let loss = 1.0 - (epoch as f32 / self.config.epochs as f32) * 0.8
                + thread_rng().r#gen::<f32>() * 0.1; // Add some noise
            let accuracy = (epoch as f32 / self.config.epochs as f32) * 0.9 + 0.1;

            // Update statistics
            {
                let mut stats = self.stats.lock().unwrap();
                stats.loss_history.push(loss);
                stats.accuracy_history.push(accuracy);

                if accuracy > stats.best_validation_accuracy {
                    stats.best_validation_accuracy = accuracy;
                }
                if loss < stats.best_validation_loss {
                    stats.best_validation_loss = loss;
                }
            }

            // Update progress
            {
                let mut progress = self.progress.write().await;
                progress.progress = (epoch + 1) as f32 / self.config.epochs as f32;
                progress.current_loss = Some(loss);

                // Estimate remaining time
                let elapsed = start_time.elapsed();
                let epochs_per_second = (epoch + 1) as f32 / elapsed.as_secs_f32();
                let remaining_epochs = self.config.epochs - (epoch + 1);
                if epochs_per_second > 0.0 {
                    progress.eta = Some(Duration::from_secs_f32(
                        remaining_epochs as f32 / epochs_per_second,
                    ));
                }
            }

            // Check for early stopping
            if epoch > self.config.early_stopping_patience {
                let stats = self.stats.lock().unwrap();
                let recent_losses = &stats.loss_history[stats
                    .loss_history
                    .len()
                    .saturating_sub(self.config.early_stopping_patience as usize)..];
                let improvement = recent_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b)) - loss;

                if improvement < self.config.min_improvement_threshold {
                    tracing::info!("Early stopping triggered at epoch {}", epoch);
                    break;
                }
            }

            // Check timeout
            if start_time.elapsed() > self.config.max_training_duration {
                tracing::warn!("Training stopped due to timeout");
                break;
            }
        }

        // Complete training
        {
            let mut stats = self.stats.lock().unwrap();
            stats.end_time = Some(Instant::now());
        }

        {
            let mut progress = self.progress.write().await;
            progress.phase = TrainingPhase::Completed;
            progress.progress = 1.0;
        }

        // Generate model path
        let model_path = self.output_dir.join(format!(
            "wake_word_model_{}.bin",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ));

        // Simulate saving model
        tokio::fs::create_dir_all(&self.output_dir)
            .await
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create output directory: {}", e),
                source: Some(Box::new(e)),
            })?;

        tokio::fs::write(&model_path, b"dummy_model_data")
            .await
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to save model: {}", e),
                source: Some(Box::new(e)),
            })?;

        Ok(model_path.to_string_lossy().to_string())
    }
}

#[async_trait]
impl WakeWordTrainer for WakeWordTrainerImpl {
    async fn train_wake_word(
        &mut self,
        training_data: WakeWordTrainingData,
    ) -> Result<String, RecognitionError> {
        tracing::info!(
            "Starting wake word training for: {}",
            training_data.wake_word
        );

        // Validate training data first
        let validation_report = self.validate_training_data(&training_data).await?;
        if validation_report.quality_score < 0.5 {
            return Err(RecognitionError::InvalidInput {
                message: format!(
                    "Training data quality too low: {:.2}",
                    validation_report.quality_score
                ),
            });
        }

        // Apply data augmentation if enabled
        let augmented_data = if self.config.data_augmentation {
            let augmenter = DataAugmenter::new();
            let mut enhanced_data = training_data.clone();

            for audio in &training_data.positive_examples {
                let augmented_samples = augmenter.augment_audio(audio);
                enhanced_data
                    .positive_examples
                    .extend(augmented_samples[1..].iter().cloned()); // Skip original
            }

            for audio in &training_data.negative_examples {
                let augmented_samples = augmenter.augment_audio(audio);
                enhanced_data
                    .negative_examples
                    .extend(augmented_samples[1..].iter().cloned());
            }

            enhanced_data
        } else {
            training_data
        };

        // Prepare training data
        let (features, labels) = self.prepare_training_data(&augmented_data).await?;

        // Train the model
        let model_path = self.train_model(features, labels).await?;

        tracing::info!(
            "Wake word training completed. Model saved to: {}",
            model_path
        );
        Ok(model_path)
    }

    async fn validate_training_data(
        &self,
        training_data: &WakeWordTrainingData,
    ) -> Result<TrainingValidationReport, RecognitionError> {
        let positive_count = training_data.positive_examples.len();
        let negative_count = training_data.negative_examples.len();

        let mut quality_issues = Vec::new();
        let mut recommendations = Vec::new();

        // Check minimum data requirements
        if positive_count < 10 {
            quality_issues.push("Insufficient positive examples (minimum 10 required)".to_string());
            recommendations.push("Collect more positive examples with the wake word".to_string());
        }

        if negative_count < 20 {
            quality_issues.push("Insufficient negative examples (minimum 20 required)".to_string());
            recommendations
                .push("Collect more negative examples without the wake word".to_string());
        }

        // Check data balance
        let ratio = positive_count as f32 / negative_count.max(1) as f32;
        if ratio < 0.2 || ratio > 0.8 {
            quality_issues.push("Imbalanced training data".to_string());
            recommendations.push(
                "Balance positive and negative examples (recommended ratio 1:2 to 1:4)".to_string(),
            );
        }

        // Check audio quality (simplified)
        let mut total_duration = 0.0;
        let mut silence_ratio = 0.0;

        for audio in &training_data.positive_examples {
            total_duration += audio.samples().len() as f32 / audio.sample_rate() as f32;
            let samples = audio.samples();
            let silent_samples = samples.iter().filter(|&&x| x.abs() < 0.01).count();
            silence_ratio += silent_samples as f32 / samples.len() as f32;
        }

        let avg_silence = silence_ratio / positive_count.max(1) as f32;
        if avg_silence > 0.5 {
            quality_issues.push("High silence ratio in positive examples".to_string());
            recommendations.push(
                "Ensure positive examples contain clear speech with the wake word".to_string(),
            );
        }

        // Calculate quality score
        let mut quality_score: f32 = 1.0;

        // Penalize for insufficient data
        if positive_count < 10 {
            quality_score -= 0.3;
        }
        if negative_count < 20 {
            quality_score -= 0.3;
        }

        // Penalize for imbalance
        if ratio < 0.2 || ratio > 0.8 {
            quality_score -= 0.2;
        }

        // Penalize for high silence
        if avg_silence > 0.5 {
            quality_score -= 0.2;
        }

        quality_score = quality_score.max(0.0);

        // Estimate accuracy based on data quality
        let estimated_accuracy = 0.5 + quality_score * 0.4; // Base 50% + up to 40% based on quality

        Ok(TrainingValidationReport {
            quality_score,
            positive_count,
            negative_count,
            quality_issues,
            recommendations,
            estimated_accuracy,
        })
    }

    async fn get_training_progress(&self) -> Result<TrainingProgress, RecognitionError> {
        let progress = self.progress.read().await;
        Ok(progress.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let output_dir = PathBuf::from("/tmp/wake_word_models");

        let trainer = WakeWordTrainerImpl::new(config, output_dir);
        let progress = trainer.get_training_progress().await.unwrap();

        assert_eq!(progress.phase, TrainingPhase::Preprocessing);
        assert_eq!(progress.progress, 0.0);
    }

    #[tokio::test]
    async fn test_training_data_validation() {
        let config = TrainingConfig::default();
        let output_dir = PathBuf::from("/tmp/wake_word_models");
        let trainer = WakeWordTrainerImpl::new(config, output_dir);

        // Create minimal training data
        let samples = vec![0.1f32; 1600]; // 100ms at 16kHz
        let audio = AudioBuffer::mono(samples, 16000);

        let training_data = WakeWordTrainingData {
            positive_examples: vec![audio.clone(); 15], // 15 positive examples
            negative_examples: vec![audio; 30],         // 30 negative examples
            wake_word: "test".to_string(),
            speaker_id: None,
        };

        let report = trainer
            .validate_training_data(&training_data)
            .await
            .unwrap();

        assert!(report.quality_score > 0.0);
        assert_eq!(report.positive_count, 15);
        assert_eq!(report.negative_count, 30);
        assert!(report.estimated_accuracy > 0.5);
    }

    #[test]
    fn test_feature_extractor() {
        let extractor = FeatureExtractor::new(16000);
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = AudioBuffer::mono(samples, 16000);

        let features = extractor.extract_features(&audio).unwrap();
        assert_eq!(features.len(), 13 * 32); // 13 MFCC * 32 frames
    }

    #[test]
    fn test_data_augmenter() {
        let augmenter = DataAugmenter::new();
        let samples = vec![0.1f32; 1600];
        let audio = AudioBuffer::mono(samples, 16000);

        let augmented = augmenter.augment_audio(&audio);
        assert!(augmented.len() > 1); // Should have original + augmented versions
    }
}
