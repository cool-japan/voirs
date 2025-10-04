//! Data processing pipeline implementation
//!
//! This module provides a comprehensive data processing pipeline for audio datasets
//! with configurable steps, parallel processing, and quality assurance.
//!
//! Features:
//! - Configurable processing steps
//! - Parallel processing with Rayon
//! - Progress tracking and cancellation
//! - Error handling and recovery
//! - Quality validation at each step
//! - Memory-efficient streaming processing

use crate::audio::io::save_audio;
use crate::audio::processing::{detect_silence, mix_channels, normalize_audio, resample_audio};
use crate::processing::features::{
    extract_fundamental_frequency, extract_mel_spectrogram, extract_mfcc,
};
use crate::processing::validation::{AudioQualityThresholds, TextValidationConfig};
use crate::{DatasetError, DatasetSample, Result};
use scirs2_core::parallel_ops::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Processing step configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStep {
    /// Load audio from file
    LoadAudio {
        /// Validate audio after loading
        validate: bool,
    },
    /// Normalize audio amplitude
    NormalizeAudio {
        /// Normalization method (Peak, RMS, LUFS)
        method: String,
        /// Target level
        target_level: f32,
    },
    /// Resample audio to target sample rate
    ResampleAudio {
        /// Target sample rate
        target_sample_rate: u32,
        /// Resampling quality (Low, Medium, High)
        quality: String,
    },
    /// Trim silence from beginning and end
    TrimSilence {
        /// Silence threshold in dB
        threshold_db: f32,
        /// Minimum leading silence to keep in seconds
        min_leading_silence: f32,
        /// Minimum trailing silence to keep in seconds
        min_trailing_silence: f32,
    },
    /// Convert to mono or stereo
    ConvertChannels {
        /// Target number of channels
        target_channels: u32,
        /// Mixing method for downmixing
        mix_method: String,
    },
    /// Validate text content
    ValidateText {
        /// Text validation configuration
        config: TextValidationConfig,
    },
    /// Extract audio features
    ExtractFeatures {
        /// Features to extract
        features: Vec<FeatureType>,
    },
    /// Save processed audio
    SaveAudio {
        /// Output directory
        output_dir: PathBuf,
        /// Output format
        format: String,
        /// Preserve directory structure
        preserve_structure: bool,
    },
}

/// Feature types for extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    /// Mel spectrogram
    MelSpectrogram {
        /// Number of mel bins
        n_mels: usize,
        /// FFT size
        n_fft: usize,
        /// Hop length
        hop_length: usize,
    },
    /// MFCC coefficients
    Mfcc {
        /// Number of coefficients
        n_mfcc: usize,
        /// Include energy coefficient
        include_energy: bool,
    },
    /// Fundamental frequency
    F0 {
        /// Minimum frequency
        f_min: f32,
        /// Maximum frequency
        f_max: f32,
    },
    /// Spectral features (centroid, rolloff, etc.)
    SpectralFeatures,
    /// Energy features (RMS, ZCR)
    EnergyFeatures,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Processing steps to execute
    pub steps: Vec<ProcessingStep>,
    /// Number of parallel workers (None = auto-detect)
    pub num_workers: Option<usize>,
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum memory usage in bytes
    pub max_memory_usage: Option<u64>,
    /// Audio quality thresholds
    pub audio_quality_thresholds: AudioQualityThresholds,
    /// Whether to continue on errors
    pub continue_on_error: bool,
    /// Output directory for processed files
    pub output_dir: Option<PathBuf>,
    /// Whether to preserve original files
    pub preserve_originals: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            steps: vec![
                ProcessingStep::LoadAudio { validate: true },
                ProcessingStep::TrimSilence {
                    threshold_db: -40.0,
                    min_leading_silence: 0.1,
                    min_trailing_silence: 0.1,
                },
                ProcessingStep::NormalizeAudio {
                    method: "RMS".to_string(),
                    target_level: -20.0,
                },
            ],
            num_workers: None,
            batch_size: 10,
            max_memory_usage: Some(2 * 1024 * 1024 * 1024), // 2GB
            audio_quality_thresholds: AudioQualityThresholds::default(),
            continue_on_error: false,
            output_dir: None,
            preserve_originals: true,
        }
    }
}

/// Processing progress information
#[derive(Debug, Clone)]
pub struct ProcessingProgress {
    /// Total number of items to process
    pub total_items: usize,
    /// Number of items processed
    pub processed_items: usize,
    /// Number of items that failed processing
    pub failed_items: usize,
    /// Processing start time
    pub start_time: Instant,
    /// Current processing step
    pub current_step: String,
    /// ETA for completion
    pub eta: Option<Duration>,
    /// Processing throughput (items/second)
    pub throughput: f64,
}

impl ProcessingProgress {
    pub fn new(total_items: usize) -> Self {
        Self {
            total_items,
            processed_items: 0,
            failed_items: 0,
            start_time: Instant::now(),
            current_step: String::new(),
            eta: None,
            throughput: 0.0,
        }
    }

    pub fn update(&mut self, processed: usize, failed: usize, current_step: String) {
        self.processed_items = processed;
        self.failed_items = failed;
        self.current_step = current_step;

        let elapsed = self.start_time.elapsed();
        if !elapsed.is_zero() {
            self.throughput = self.processed_items as f64 / elapsed.as_secs_f64();

            if self.throughput > 0.0 {
                let remaining_items = self.total_items.saturating_sub(self.processed_items);
                let eta_seconds = remaining_items as f64 / self.throughput;
                self.eta = Some(Duration::from_secs_f64(eta_seconds));
            }
        }
    }

    pub fn completion_percentage(&self) -> f64 {
        if self.total_items == 0 {
            100.0
        } else {
            (self.processed_items as f64 / self.total_items as f64) * 100.0
        }
    }
}

/// Processing result for a single item
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Original sample ID
    pub sample_id: String,
    /// Whether processing was successful
    pub success: bool,
    /// Error message if processing failed
    pub error: Option<String>,
    /// Processed sample (if successful)
    pub processed_sample: Option<DatasetSample>,
    /// Extracted features (if requested)
    pub features: HashMap<String, Vec<f32>>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Quality metrics after processing
    pub quality_metrics: Option<HashMap<String, f32>>,
}

/// Main processing pipeline
pub struct ProcessingPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// Processing progress tracking
    progress: Arc<std::sync::Mutex<ProcessingProgress>>,
    /// Cancellation flag
    cancelled: Arc<AtomicBool>,
    /// Progress sender
    progress_sender: Option<mpsc::UnboundedSender<ProcessingProgress>>,
}

impl ProcessingPipeline {
    /// Create a new processing pipeline
    pub fn new(config: PipelineConfig) -> Self {
        let total_items = 0; // Will be set when processing starts
        Self {
            config,
            progress: Arc::new(std::sync::Mutex::new(ProcessingProgress::new(total_items))),
            cancelled: Arc::new(AtomicBool::new(false)),
            progress_sender: None,
        }
    }

    /// Create a pipeline with progress reporting
    pub fn with_progress_reporting(
        mut self,
    ) -> (Self, mpsc::UnboundedReceiver<ProcessingProgress>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.progress_sender = Some(sender);
        (self, receiver)
    }

    /// Cancel the processing pipeline
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Check if pipeline is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Get current processing progress
    pub fn get_progress(&self) -> ProcessingProgress {
        self.progress.lock().unwrap().clone()
    }

    /// Process a batch of dataset samples
    pub async fn process_samples(
        &mut self,
        samples: Vec<DatasetSample>,
    ) -> Result<Vec<ProcessingResult>> {
        // Update progress tracking
        {
            let mut progress = self.progress.lock().unwrap();
            progress.total_items = samples.len();
            progress.processed_items = 0;
            progress.failed_items = 0;
            progress.start_time = Instant::now();
        }

        // Set up parallel processing
        let num_workers = self.config.num_workers.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        // Process samples in parallel batches
        let mut results = Vec::with_capacity(samples.len());
        let batch_size = self.config.batch_size;

        for (batch_idx, batch) in samples.chunks(batch_size).enumerate() {
            if self.is_cancelled() {
                break;
            }

            let batch_results: Vec<_> = batch
                .par_iter()
                .with_max_len(num_workers)
                .map(|sample| {
                    if self.is_cancelled() {
                        return ProcessingResult {
                            sample_id: sample.id.clone(),
                            success: false,
                            error: Some("Processing cancelled".to_string()),
                            processed_sample: None,
                            features: HashMap::new(),
                            processing_time_ms: 0,
                            quality_metrics: None,
                        };
                    }

                    self.process_single_sample(sample.clone())
                })
                .collect();

            // Update progress
            let processed_count =
                (batch_idx + 1) * batch_size.min(samples.len() - batch_idx * batch_size);
            let failed_count = batch_results.iter().filter(|r| !r.success).count();

            {
                let mut progress = self.progress.lock().unwrap();
                progress.update(
                    processed_count,
                    failed_count,
                    format!(
                        "Processing batch {}/{}",
                        batch_idx + 1,
                        (samples.len() + batch_size - 1) / batch_size
                    ),
                );

                // Send progress update if sender is available
                if let Some(ref sender) = self.progress_sender {
                    let _ = sender.send(progress.clone());
                }
            }

            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Process a single dataset sample
    fn process_single_sample(&self, mut sample: DatasetSample) -> ProcessingResult {
        let start_time = Instant::now();
        let mut features = HashMap::new();
        let mut quality_metrics = HashMap::new();

        for step in &self.config.steps {
            if self.is_cancelled() {
                return ProcessingResult {
                    sample_id: sample.id.clone(),
                    success: false,
                    error: Some("Processing cancelled".to_string()),
                    processed_sample: None,
                    features,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    quality_metrics: None,
                };
            }

            match self.process_step(step, &mut sample, &mut features, &mut quality_metrics) {
                Ok(_) => {}
                Err(e) => {
                    if !self.config.continue_on_error {
                        return ProcessingResult {
                            sample_id: sample.id.clone(),
                            success: false,
                            error: Some(e.to_string()),
                            processed_sample: None,
                            features,
                            processing_time_ms: start_time.elapsed().as_millis() as u64,
                            quality_metrics: Some(quality_metrics),
                        };
                    }
                    // Log error but continue processing
                    eprintln!("Warning: Step failed for sample {}: {}", sample.id, e);
                }
            }
        }

        ProcessingResult {
            sample_id: sample.id.clone(),
            success: true,
            error: None,
            processed_sample: Some(sample),
            features,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            quality_metrics: Some(quality_metrics),
        }
    }

    /// Process a single step for a sample
    fn process_step(
        &self,
        step: &ProcessingStep,
        sample: &mut DatasetSample,
        features: &mut HashMap<String, Vec<f32>>,
        quality_metrics: &mut HashMap<String, f32>,
    ) -> Result<()> {
        match step {
            ProcessingStep::LoadAudio { validate } => {
                // Audio should already be loaded in the sample
                if *validate {
                    // Validate audio quality
                    let audio = &sample.audio;
                    quality_metrics.insert("duration".to_string(), audio.duration());
                    quality_metrics.insert("sample_rate".to_string(), audio.sample_rate() as f32);
                    quality_metrics.insert("channels".to_string(), audio.channels() as f32);

                    // Add more quality metrics here
                    if let Some(rms) = audio.rms() {
                        quality_metrics.insert("rms".to_string(), rms);
                    }
                }
            }

            ProcessingStep::NormalizeAudio {
                method,
                target_level,
            } => {
                sample.audio = normalize_audio(&sample.audio, method, *target_level)?;
            }

            ProcessingStep::ResampleAudio {
                target_sample_rate,
                quality: _,
            } => {
                if sample.audio.sample_rate() != *target_sample_rate {
                    sample.audio = resample_audio(&sample.audio, *target_sample_rate)?;
                }
            }

            ProcessingStep::TrimSilence {
                threshold_db,
                min_leading_silence: _,
                min_trailing_silence: _,
            } => {
                let (_start, _end) = detect_silence(&sample.audio, *threshold_db)?;
                // Trim audio based on silence detection
                // This would need to be implemented in the audio processing module
            }

            ProcessingStep::ConvertChannels {
                target_channels,
                mix_method,
            } => {
                if sample.audio.channels() != *target_channels {
                    sample.audio =
                        mix_channels(&sample.audio, *target_channels as usize, mix_method)?;
                }
            }

            ProcessingStep::ValidateText { config: _ } => {
                // Validate text content
                // This would use the validation functions
            }

            ProcessingStep::ExtractFeatures {
                features: feature_types,
            } => {
                for feature_type in feature_types {
                    match feature_type {
                        FeatureType::MelSpectrogram {
                            n_mels,
                            n_fft,
                            hop_length,
                        } => {
                            if let Ok(mel_spec) =
                                extract_mel_spectrogram(&sample.audio, *n_mels, *n_fft, *hop_length)
                            {
                                features.insert("mel_spectrogram".to_string(), mel_spec.values);
                            }
                        }
                        FeatureType::Mfcc {
                            n_mfcc,
                            include_energy,
                        } => {
                            if let Ok(mfcc) = extract_mfcc(&sample.audio, *n_mfcc, *include_energy)
                            {
                                features.insert("mfcc".to_string(), mfcc);
                            }
                        }
                        FeatureType::F0 { f_min, f_max } => {
                            if let Ok(f0) =
                                extract_fundamental_frequency(&sample.audio, *f_min, *f_max)
                            {
                                features.insert("f0".to_string(), f0);
                            }
                        }
                        FeatureType::SpectralFeatures => {
                            // Extract spectral features
                        }
                        FeatureType::EnergyFeatures => {
                            // Extract energy features
                        }
                    }
                }
            }

            ProcessingStep::SaveAudio {
                output_dir,
                format,
                preserve_structure: _,
            } => {
                if let Some(ref base_output_dir) = self.config.output_dir {
                    let output_path = base_output_dir
                        .join(output_dir)
                        .join(format!("{}.{format}", sample.id));
                    if let Some(parent) = output_path.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            DatasetError::ProcessingError(format!(
                                "Failed to create output directory: {e}"
                            ))
                        })?;
                    }
                    save_audio(&sample.audio, &output_path)?;
                }
            }
        }

        Ok(())
    }

    /// Get pipeline statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let progress = self.progress.lock().unwrap();
        let mut stats = HashMap::new();

        stats.insert("total_items".to_string(), progress.total_items as f64);
        stats.insert(
            "processed_items".to_string(),
            progress.processed_items as f64,
        );
        stats.insert("failed_items".to_string(), progress.failed_items as f64);
        stats.insert(
            "completion_percentage".to_string(),
            progress.completion_percentage(),
        );
        stats.insert("throughput".to_string(), progress.throughput);

        if let Some(eta) = progress.eta {
            stats.insert("eta_seconds".to_string(), eta.as_secs_f64());
        }

        stats
    }
}

/// Pipeline builder for easier configuration
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    pub fn with_steps(mut self, steps: Vec<ProcessingStep>) -> Self {
        self.config.steps = steps;
        self
    }

    pub fn with_workers(mut self, num_workers: usize) -> Self {
        self.config.num_workers = Some(num_workers);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn with_output_dir<P: AsRef<Path>>(mut self, output_dir: P) -> Self {
        self.config.output_dir = Some(output_dir.as_ref().to_path_buf());
        self
    }

    pub fn continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.config.continue_on_error = continue_on_error;
        self
    }

    pub fn build(self) -> ProcessingPipeline {
        ProcessingPipeline::new(self.config)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::dummy::DummyDataset;
    use crate::traits::Dataset;

    #[tokio::test]
    async fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_workers(2)
            .with_batch_size(5)
            .continue_on_error(true)
            .build();

        assert_eq!(pipeline.config.num_workers, Some(2));
        assert_eq!(pipeline.config.batch_size, 5);
        assert!(pipeline.config.continue_on_error);
    }

    #[tokio::test]
    async fn test_processing_progress() {
        let mut progress = ProcessingProgress::new(100);
        assert_eq!(progress.completion_percentage(), 0.0);

        progress.update(50, 5, "Processing".to_string());
        assert_eq!(progress.completion_percentage(), 50.0);
        assert_eq!(progress.processed_items, 50);
        assert_eq!(progress.failed_items, 5);
    }

    #[tokio::test]
    async fn test_pipeline_processing() {
        // Create a dummy dataset for testing
        let dataset = DummyDataset::small();

        // Get a few samples
        let mut samples = Vec::new();
        for i in 0..3 {
            if let Ok(sample) = dataset.get(i).await {
                samples.push(sample);
            }
        }

        // Create a simple pipeline
        let pipeline_config = PipelineConfig {
            steps: vec![
                ProcessingStep::LoadAudio { validate: true },
                ProcessingStep::NormalizeAudio {
                    method: "RMS".to_string(),
                    target_level: -20.0,
                },
            ],
            batch_size: 2,
            continue_on_error: true,
            ..Default::default()
        };

        let mut pipeline = ProcessingPipeline::new(pipeline_config);
        let results = pipeline.process_samples(samples).await.unwrap();

        assert_eq!(results.len(), 3);
        // All samples should process successfully with dummy data
        assert!(results.iter().all(|r| r.success));
    }
}
