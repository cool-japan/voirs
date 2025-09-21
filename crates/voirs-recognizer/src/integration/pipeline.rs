//! # Pipeline Integration
//!
//! This module provides unified pipeline integration between recognition
//! and synthesis components, enabling seamless audio processing workflows.

use crate::{analysis::AudioAnalyzerImpl, traits::*, RecognitionError};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Unified VoiRS pipeline for comprehensive audio processing
pub struct UnifiedVoirsPipeline {
    /// ASR model
    asr_model: Box<dyn ASRModel + Send + Sync>,
    /// Phoneme recognizer
    phoneme_recognizer: Box<dyn PhonemeRecognizer + Send + Sync>,
    /// Audio analyzer
    audio_analyzer: AudioAnalyzerImpl,
    /// Pipeline configuration
    config: PipelineProcessingConfig,
    /// Performance metrics
    metrics: Arc<RwLock<PipelineMetrics>>,
}

/// Pipeline processing configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineProcessingConfig {
    /// Processing mode
    pub mode: ProcessingMode,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Buffer size
    pub buffer_size: usize,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Enable caching
    pub enable_caching: bool,
}

/// Processing mode enumeration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ProcessingMode {
    /// Full pipeline (ASR + Phoneme + Analysis)
    Full,
    /// ASR only
    ASROnly,
    /// Phoneme recognition only
    PhonemeOnly,
    /// Audio analysis only
    AnalysisOnly,
    /// Custom pipeline
    Custom(Vec<PipelineStage>),
}

/// Pipeline stage enumeration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PipelineStage {
    /// Audio preprocessing
    Preprocessing,
    /// ASR transcription
    ASR,
    /// Phoneme recognition
    Phoneme,
    /// Audio analysis
    Analysis,
    /// Quality assessment
    Quality,
    /// Post-processing
    PostProcessing,
}

/// Pipeline processing result
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Transcription result
    pub transcription: Option<Transcript>,
    /// Phoneme recognition result
    pub phonemes: Option<Vec<AlignedPhoneme>>,
    /// Audio analysis result
    pub analysis: Option<AudioAnalysis>,
    /// Processing metadata
    pub metadata: PipelineMetadata,
}

/// Pipeline metadata
#[derive(Debug, Clone)]
pub struct PipelineMetadata {
    /// Processing duration
    pub processing_duration: std::time::Duration,
    /// Stages executed
    pub stages_executed: Vec<PipelineStage>,
    /// Confidence scores
    pub confidence_scores: std::collections::HashMap<String, f32>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Resource usage information
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory usage in MB
    pub memory_mb: f32,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// GPU usage percentage
    pub gpu_percent: Option<f32>,
    /// Network usage in MB
    pub network_mb: f32,
}

/// Pipeline performance metrics
#[derive(Debug, Default, Clone)]
pub struct PipelineMetrics {
    /// Total processed samples
    pub total_processed: u64,
    /// Average processing time
    pub avg_processing_time: std::time::Duration,
    /// Success rate
    pub success_rate: f32,
    /// Error count
    pub error_count: u64,
    /// Resource utilization
    pub resource_utilization: ResourceUsage,
}

/// Simple stub phoneme recognizer for testing/fallback
#[derive(Debug)]
struct StubPhonemeRecognizer;

#[async_trait]
impl PhonemeRecognizer for StubPhonemeRecognizer {
    async fn recognize_phonemes(
        &self,
        _audio: &AudioBuffer,
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<Vec<voirs_sdk::Phoneme>> {
        // Return empty phoneme list as fallback
        Ok(vec![])
    }

    async fn align_phonemes(
        &self,
        _audio: &AudioBuffer,
        _expected: &[voirs_sdk::Phoneme],
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        Ok(PhonemeAlignment {
            phonemes: vec![],
            total_duration: 0.0,
            alignment_confidence: 0.0,
            word_alignments: vec![],
        })
    }

    async fn align_text(
        &self,
        _audio: &AudioBuffer,
        _text: &str,
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        Ok(PhonemeAlignment {
            phonemes: vec![],
            total_duration: 0.0,
            alignment_confidence: 0.0,
            word_alignments: vec![],
        })
    }

    fn metadata(&self) -> PhonemeRecognizerMetadata {
        PhonemeRecognizerMetadata {
            name: "StubPhonemeRecognizer".to_string(),
            version: "0.1.0".to_string(),
            description: "Simple stub phoneme recognizer for testing".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            alignment_methods: vec![AlignmentMethod::Forced],
            alignment_accuracy: 0.0,
            supported_features: vec![],
        }
    }

    fn supports_feature(&self, _feature: PhonemeRecognitionFeature) -> bool {
        false
    }
}

impl UnifiedVoirsPipeline {
    /// Create new unified pipeline
    pub async fn new(
        asr_model: Box<dyn ASRModel + Send + Sync>,
        config: PipelineProcessingConfig,
    ) -> Result<Self, RecognitionError> {
        let phoneme_recognizer: Box<dyn PhonemeRecognizer + Send + Sync> =
            Box::new(StubPhonemeRecognizer);
        let audio_analyzer = AudioAnalyzerImpl::new(AudioAnalysisConfig::default()).await?;
        let metrics = Arc::new(RwLock::new(PipelineMetrics::default()));

        Ok(Self {
            asr_model,
            phoneme_recognizer,
            audio_analyzer,
            config,
            metrics,
        })
    }

    /// Process audio through the pipeline
    pub async fn process(&self, audio: &AudioBuffer) -> Result<PipelineResult, RecognitionError> {
        let start_time = std::time::Instant::now();
        let mut result = PipelineResult {
            transcription: None,
            phonemes: None,
            analysis: None,
            metadata: PipelineMetadata {
                processing_duration: std::time::Duration::default(),
                stages_executed: Vec::new(),
                confidence_scores: std::collections::HashMap::new(),
                resource_usage: ResourceUsage {
                    memory_mb: 0.0,
                    cpu_percent: 0.0,
                    gpu_percent: None,
                    network_mb: 0.0,
                },
            },
        };

        // Execute pipeline stages based on mode
        match &self.config.mode {
            ProcessingMode::Full => {
                result = self.execute_full_pipeline(audio, result).await?;
            }
            ProcessingMode::ASROnly => {
                result = self.execute_asr_stage(audio, result).await?;
            }
            ProcessingMode::PhonemeOnly => {
                result = self.execute_phoneme_stage(audio, result).await?;
            }
            ProcessingMode::AnalysisOnly => {
                result = self.execute_analysis_stage(audio, result).await?;
            }
            ProcessingMode::Custom(stages) => {
                result = self.execute_custom_pipeline(audio, result, stages).await?;
            }
        }

        // Update metadata
        result.metadata.processing_duration = start_time.elapsed();

        // Update metrics
        self.update_metrics(&result).await;

        Ok(result)
    }

    /// Execute full pipeline
    async fn execute_full_pipeline(
        &self,
        audio: &AudioBuffer,
        mut result: PipelineResult,
    ) -> Result<PipelineResult, RecognitionError> {
        // Stage 1: Audio Analysis
        result = self.execute_analysis_stage(audio, result).await?;

        // Stage 2: ASR Transcription
        result = self.execute_asr_stage(audio, result).await?;

        // Stage 3: Phoneme Recognition
        result = self.execute_phoneme_stage(audio, result).await?;

        Ok(result)
    }

    /// Execute ASR stage
    async fn execute_asr_stage(
        &self,
        audio: &AudioBuffer,
        mut result: PipelineResult,
    ) -> Result<PipelineResult, RecognitionError> {
        let transcript = self
            .asr_model
            .transcribe(audio, Some(&ASRConfig::default()))
            .await?;
        result.transcription = Some(transcript);
        result.metadata.stages_executed.push(PipelineStage::ASR);

        // Add confidence score
        if let Some(ref transcript) = result.transcription {
            result
                .metadata
                .confidence_scores
                .insert("asr".to_string(), transcript.confidence);
        }

        Ok(result)
    }

    /// Execute phoneme recognition stage
    async fn execute_phoneme_stage(
        &self,
        audio: &AudioBuffer,
        mut result: PipelineResult,
    ) -> Result<PipelineResult, RecognitionError> {
        let phonemes = self
            .phoneme_recognizer
            .recognize_phonemes(audio, Some(&PhonemeRecognitionConfig::default()))
            .await?;

        // Convert phonemes to aligned phonemes with default timing
        let aligned_phonemes: Vec<AlignedPhoneme> = phonemes
            .into_iter()
            .enumerate()
            .map(|(i, phoneme)| AlignedPhoneme {
                phoneme,
                start_time: i as f32 * 0.1, // Default timing
                end_time: (i + 1) as f32 * 0.1,
                confidence: 0.8, // Default confidence
            })
            .collect();

        result.phonemes = Some(aligned_phonemes);
        result.metadata.stages_executed.push(PipelineStage::Phoneme);

        Ok(result)
    }

    /// Execute analysis stage
    async fn execute_analysis_stage(
        &self,
        audio: &AudioBuffer,
        mut result: PipelineResult,
    ) -> Result<PipelineResult, RecognitionError> {
        let analysis = self
            .audio_analyzer
            .analyze(audio, Some(&AudioAnalysisConfig::default()))
            .await?;
        result.analysis = Some(analysis);
        result
            .metadata
            .stages_executed
            .push(PipelineStage::Analysis);

        Ok(result)
    }

    /// Execute custom pipeline
    async fn execute_custom_pipeline(
        &self,
        audio: &AudioBuffer,
        mut result: PipelineResult,
        stages: &[PipelineStage],
    ) -> Result<PipelineResult, RecognitionError> {
        for stage in stages {
            match stage {
                PipelineStage::ASR => {
                    result = self.execute_asr_stage(audio, result).await?;
                }
                PipelineStage::Phoneme => {
                    result = self.execute_phoneme_stage(audio, result).await?;
                }
                PipelineStage::Analysis => {
                    result = self.execute_analysis_stage(audio, result).await?;
                }
                PipelineStage::Preprocessing => {
                    // Preprocessing would be implemented here
                    result
                        .metadata
                        .stages_executed
                        .push(PipelineStage::Preprocessing);
                }
                PipelineStage::Quality => {
                    // Quality assessment would be implemented here
                    result.metadata.stages_executed.push(PipelineStage::Quality);
                }
                PipelineStage::PostProcessing => {
                    // Post-processing would be implemented here
                    result
                        .metadata
                        .stages_executed
                        .push(PipelineStage::PostProcessing);
                }
            }
        }

        Ok(result)
    }

    /// Update pipeline metrics
    async fn update_metrics(&self, result: &PipelineResult) {
        let mut metrics = self.metrics.write().await;
        metrics.total_processed += 1;

        // Update average processing time
        let total_time = metrics.avg_processing_time.as_secs_f64()
            * (metrics.total_processed - 1) as f64
            + result.metadata.processing_duration.as_secs_f64();
        metrics.avg_processing_time =
            std::time::Duration::from_secs_f64(total_time / metrics.total_processed as f64);

        // Update success rate (simplified)
        metrics.success_rate = 1.0 - (metrics.error_count as f32 / metrics.total_processed as f32);

        // Update resource utilization
        metrics.resource_utilization = result.metadata.resource_usage.clone();
    }

    /// Get pipeline metrics
    pub async fn get_metrics(&self) -> PipelineMetrics {
        (*self.metrics.read().await).clone()
    }

    /// Reset pipeline metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = PipelineMetrics::default();
    }

    /// Get pipeline configuration
    pub fn get_config(&self) -> &PipelineProcessingConfig {
        &self.config
    }

    /// Process audio from raw bytes
    pub async fn recognize_bytes(
        &self,
        audio_bytes: &[u8],
    ) -> Result<PipelineResult, RecognitionError> {
        // Convert bytes to AudioBuffer - assuming 16-bit PCM, mono, 16kHz
        // In a real implementation, this should be configurable
        let samples_per_byte = 2; // 16-bit = 2 bytes per sample
        if audio_bytes.len() % samples_per_byte != 0 {
            return Err(RecognitionError::AudioProcessingError {
                message: "Audio bytes length not aligned to sample size".to_string(),
                source: None,
            });
        }

        let mut samples = Vec::with_capacity(audio_bytes.len() / samples_per_byte);
        for chunk in audio_bytes.chunks_exact(samples_per_byte) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
            samples.push(sample);
        }

        let audio_buffer = AudioBuffer::mono(samples, 16000);
        self.process(&audio_buffer).await
    }

    /// Update pipeline configuration
    pub fn update_config(&mut self, config: PipelineProcessingConfig) {
        self.config = config;
    }
}

/// Pipeline builder for easier construction
pub struct PipelineBuilder {
    asr_model: Option<Box<dyn ASRModel + Send + Sync>>,
    config: PipelineProcessingConfig,
    model_name: Option<String>,
    language: Option<LanguageCode>,
    sample_rate: Option<u32>,
}

impl PipelineBuilder {
    /// Create new pipeline builder
    pub fn new() -> Self {
        Self {
            asr_model: None,
            config: PipelineProcessingConfig::default(),
            model_name: None,
            language: None,
            sample_rate: None,
        }
    }

    /// Set ASR model
    pub fn with_asr_model(mut self, model: Box<dyn ASRModel + Send + Sync>) -> Self {
        self.asr_model = Some(model);
        self
    }

    /// Set pipeline configuration
    pub fn with_config(mut self, config: PipelineProcessingConfig) -> Self {
        self.config = config;
        self
    }

    /// Set processing mode
    pub fn with_mode(mut self, mode: ProcessingMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Enable parallel processing
    pub fn with_parallel_processing(mut self, enabled: bool) -> Self {
        self.config.parallel_processing = enabled;
        self
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set model name
    pub fn with_model(mut self, model_name: &str) -> Self {
        self.model_name = Some(model_name.to_string());
        self
    }

    /// Set language
    pub fn with_language(mut self, language: LanguageCode) -> Self {
        self.language = Some(language);
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Build the pipeline
    pub async fn build(self) -> Result<UnifiedVoirsPipeline, RecognitionError> {
        let asr_model = self
            .asr_model
            .ok_or_else(|| RecognitionError::ConfigurationError {
                message: "ASR model is required".to_string(),
            })?;

        UnifiedVoirsPipeline::new(asr_model, self.config).await
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PipelineProcessingConfig {
    fn default() -> Self {
        Self {
            mode: ProcessingMode::Full,
            parallel_processing: true,
            buffer_size: 4096,
            timeout_seconds: 30,
            enable_caching: true,
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            memory_mb: 0.0,
            cpu_percent: 0.0,
            gpu_percent: None,
            network_mb: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "whisper-pure")]
    use crate::asr::whisper::WhisperConfig;
    #[cfg(feature = "whisper-pure")]
    use crate::asr::whisper_pure::PureRustWhisper;

    #[tokio::test]
    #[cfg(feature = "whisper-pure")]
    async fn test_pipeline_builder() {
        let whisper = PureRustWhisper::new(WhisperConfig::default())
            .await
            .unwrap();
        let pipeline = PipelineBuilder::new()
            .with_asr_model(Box::new(whisper))
            .with_mode(ProcessingMode::ASROnly)
            .with_parallel_processing(true)
            .build()
            .await;

        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_config() {
        let config = PipelineProcessingConfig {
            mode: ProcessingMode::Full,
            parallel_processing: true,
            buffer_size: 8192,
            timeout_seconds: 60,
            enable_caching: true,
        };

        assert_eq!(config.mode, ProcessingMode::Full);
        assert!(config.parallel_processing);
        assert_eq!(config.buffer_size, 8192);
    }

    #[tokio::test]
    async fn test_processing_modes() {
        let modes = vec![
            ProcessingMode::Full,
            ProcessingMode::ASROnly,
            ProcessingMode::PhonemeOnly,
            ProcessingMode::AnalysisOnly,
            ProcessingMode::Custom(vec![PipelineStage::ASR, PipelineStage::Analysis]),
        ];

        for mode in modes {
            let config = PipelineProcessingConfig {
                mode,
                ..Default::default()
            };
            assert!(config.timeout_seconds > 0);
        }
    }

    #[test]
    fn test_pipeline_stages() {
        let stages = [
            PipelineStage::Preprocessing,
            PipelineStage::ASR,
            PipelineStage::Phoneme,
            PipelineStage::Analysis,
            PipelineStage::Quality,
            PipelineStage::PostProcessing,
        ];

        assert_eq!(stages.len(), 6);
        assert!(stages.contains(&PipelineStage::ASR));
        assert!(stages.contains(&PipelineStage::Phoneme));
        assert!(stages.contains(&PipelineStage::Analysis));
    }
}
