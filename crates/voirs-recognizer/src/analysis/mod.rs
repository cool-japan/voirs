//! Audio analysis implementations
//!
//! This module provides comprehensive audio analysis capabilities including:
//! - Quality metrics (SNR, THD, spectral features)
//! - Prosody analysis (pitch, rhythm, stress)
//! - Speaker characteristics (gender, age, voice quality)
//! - Emotional analysis

use crate::traits::{
    AnalysisCapability, AudioAnalysis, AudioAnalysisConfig, AudioAnalyzer, AudioAnalyzerMetadata,
    AudioMetric, AudioStream, Emotion, EmotionalAnalysis, ProsodyAnalysis, RecognitionResult,
    SpeakerCharacteristics,
};
use crate::RecognitionError;
use std::sync::Arc;

// Analysis implementations
pub mod prosody;
pub mod quality;
pub mod speaker;
pub mod vad;

pub use prosody::*;
pub use quality::*;
pub use speaker::*;
pub use vad::*;

/// Audio analyzer backend enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum AudioAnalyzerBackend {
    /// Comprehensive analyzer with all features
    Comprehensive {
        /// Enable quality metrics
        quality_metrics: bool,
        /// Enable prosody analysis
        prosody_analysis: bool,
        /// Enable speaker analysis
        speaker_analysis: bool,
        /// Enable emotional analysis
        emotional_analysis: bool,
    },
    /// Quality-focused analyzer
    QualityFocused {
        /// Metrics to compute
        metrics: Vec<AudioMetric>,
    },
    /// Prosody-focused analyzer
    ProsodyFocused {
        /// Prosody features to analyze
        features: Vec<ProsodyFeature>,
    },
    /// Speaker-focused analyzer
    SpeakerFocused {
        /// Speaker features to analyze
        features: Vec<SpeakerFeature>,
    },
}

/// Prosody analysis features
#[derive(Debug, Clone, PartialEq)]
pub enum ProsodyFeature {
    /// Pitch analysis and F0 tracking
    Pitch,
    /// Rhythm and timing analysis
    Rhythm,
    /// Stress pattern detection
    Stress,
    /// Intonation contour analysis
    Intonation,
    /// Speaking rate and tempo analysis
    SpeakingRate,
    /// Pause detection and analysis
    Pauses,
}

/// Speaker analysis features
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakerFeature {
    /// Gender classification
    Gender,
    /// Age estimation
    Age,
    /// Voice quality assessment
    VoiceQuality,
    /// Accent detection and classification
    Accent,
    /// Formant frequency analysis
    Formants,
    /// Fundamental frequency range analysis
    F0Range,
}

/// Main audio analyzer implementation
pub struct AudioAnalyzerImpl {
    /// Quality analyzer
    quality_analyzer: Arc<QualityAnalyzer>,
    /// Prosody analyzer
    prosody_analyzer: Arc<ProsodyAnalyzer>,
    /// Speaker analyzer
    speaker_analyzer: Arc<SpeakerAnalyzer>,
    /// Configuration
    config: AudioAnalysisConfig,
    /// Supported metrics
    supported_metrics: Vec<AudioMetric>,
    /// Metadata
    metadata: AudioAnalyzerMetadata,
}

impl AudioAnalyzerImpl {
    /// Create a new comprehensive audio analyzer
    ///
    /// # Errors
    ///
    /// Returns a `RecognitionError` if any of the component analyzers fail to initialize.
    pub async fn new(config: AudioAnalysisConfig) -> Result<Self, RecognitionError> {
        let quality_analyzer = Arc::new(QualityAnalyzer::new().await?);
        let prosody_analyzer = Arc::new(ProsodyAnalyzer::new().await?);
        let speaker_analyzer = Arc::new(SpeakerAnalyzer::new().await?);

        let supported_metrics = vec![
            AudioMetric::SNR,
            AudioMetric::THD,
            AudioMetric::SpectralCentroid,
            AudioMetric::SpectralRolloff,
            AudioMetric::ZeroCrossingRate,
            AudioMetric::MelFrequencyCepstralCoefficients,
            AudioMetric::ChromaFeatures,
            AudioMetric::SpectralContrast,
            AudioMetric::TonnetzFeatures,
            AudioMetric::RootMeanSquare,
        ];

        let metadata = AudioAnalyzerMetadata {
            name: "Comprehensive Audio Analyzer".to_string(),
            version: "1.0.0".to_string(),
            description:
                "Multi-dimensional audio analysis with quality, prosody, and speaker features"
                    .to_string(),
            supported_metrics: supported_metrics.clone(),
            capabilities: vec![
                AnalysisCapability::QualityMetrics,
                AnalysisCapability::ProsodyAnalysis,
                AnalysisCapability::SpeakerCharacteristics,
                AnalysisCapability::EmotionalAnalysis,
                AnalysisCapability::RealtimeAnalysis,
                AnalysisCapability::BatchProcessing,
                AnalysisCapability::StreamingAnalysis,
            ],
            processing_speed: 2.0, // 2x real-time
        };

        Ok(Self {
            quality_analyzer,
            prosody_analyzer,
            speaker_analyzer,
            config,
            supported_metrics,
            metadata,
        })
    }

    /// Create with specific backend configuration
    ///
    /// # Errors
    ///
    /// Returns a `RecognitionError` if the backend configuration is invalid or initialization fails.
    pub async fn with_backend(backend: AudioAnalyzerBackend) -> Result<Self, RecognitionError> {
        let config = match backend {
            AudioAnalyzerBackend::Comprehensive {
                quality_metrics,
                prosody_analysis,
                speaker_analysis,
                emotional_analysis,
            } => AudioAnalysisConfig {
                quality_metrics,
                prosody_analysis,
                speaker_analysis,
                emotional_analysis,
                ..Default::default()
            },
            AudioAnalyzerBackend::QualityFocused { metrics } => AudioAnalysisConfig {
                quality_metrics: true,
                prosody_analysis: false,
                speaker_analysis: false,
                emotional_analysis: false,
                quality_metrics_list: metrics,
                ..Default::default()
            },
            AudioAnalyzerBackend::ProsodyFocused { .. } => AudioAnalysisConfig {
                quality_metrics: false,
                prosody_analysis: true,
                speaker_analysis: false,
                emotional_analysis: false,
                ..Default::default()
            },
            AudioAnalyzerBackend::SpeakerFocused { .. } => AudioAnalysisConfig {
                quality_metrics: false,
                prosody_analysis: false,
                speaker_analysis: true,
                emotional_analysis: false,
                ..Default::default()
            },
        };

        Self::new(config).await
    }
}

#[async_trait::async_trait]
impl AudioAnalyzer for AudioAnalyzerImpl {
    async fn analyze(
        &self,
        audio: &voirs_sdk::AudioBuffer,
        config: Option<&AudioAnalysisConfig>,
    ) -> RecognitionResult<AudioAnalysis> {
        let config = config.unwrap_or(&self.config);
        let start_time = std::time::Instant::now();

        // Initialize result structure
        let mut quality_metrics = std::collections::HashMap::new();

        // Quality analysis
        if config.quality_metrics {
            let quality_results = self
                .quality_analyzer
                .analyze_quality(audio, &config.quality_metrics_list)
                .await?;
            quality_metrics.extend(quality_results);
        }

        // Prosody analysis
        let prosody = if config.prosody_analysis {
            self.prosody_analyzer.analyze_prosody(audio).await?
        } else {
            ProsodyAnalysis::default()
        };

        // Speaker analysis
        let speaker_characteristics = if config.speaker_analysis {
            self.speaker_analyzer.analyze_speaker(audio).await?
        } else {
            SpeakerCharacteristics::default()
        };

        // Emotional analysis
        let emotional_analysis = if config.emotional_analysis {
            self.speaker_analyzer.analyze_emotion(audio).await?
        } else {
            EmotionalAnalysis::default()
        };

        let processing_duration = start_time.elapsed();

        Ok(AudioAnalysis {
            quality_metrics,
            prosody,
            speaker_characteristics,
            emotional_analysis,
            processing_duration: Some(processing_duration),
        })
    }

    async fn analyze_streaming(
        &self,
        mut audio_stream: AudioStream,
        config: Option<&AudioAnalysisConfig>,
    ) -> RecognitionResult<
        std::pin::Pin<
            Box<dyn tokio_stream::Stream<Item = RecognitionResult<AudioAnalysis>> + Send>,
        >,
    > {
        let config = config.cloned().unwrap_or_else(|| self.config.clone());
        let analyzer = self.clone();

        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            use futures::StreamExt;

            while let Some(audio_chunk) = audio_stream.next().await {
                let analysis_result = analyzer.analyze(&audio_chunk, Some(&config)).await;

                if sender.send(analysis_result).is_err() {
                    break;
                }
            }
        });

        Ok(Box::pin(
            tokio_stream::wrappers::UnboundedReceiverStream::new(receiver),
        ))
    }

    fn supported_metrics(&self) -> Vec<AudioMetric> {
        self.supported_metrics.clone()
    }

    fn metadata(&self) -> AudioAnalyzerMetadata {
        self.metadata.clone()
    }

    fn supports_capability(&self, capability: AnalysisCapability) -> bool {
        self.metadata.capabilities.contains(&capability)
    }
}

impl Clone for AudioAnalyzerImpl {
    fn clone(&self) -> Self {
        Self {
            quality_analyzer: self.quality_analyzer.clone(),
            prosody_analyzer: self.prosody_analyzer.clone(),
            speaker_analyzer: self.speaker_analyzer.clone(),
            config: self.config.clone(),
            supported_metrics: self.supported_metrics.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

// Default implementations for data types

impl Default for EmotionalAnalysis {
    fn default() -> Self {
        Self {
            primary_emotion: Emotion::Neutral,
            emotion_scores: std::collections::HashMap::new(),
            intensity: 0.0,
            valence: 0.0,
            arousal: 0.0,
        }
    }
}

/// Factory function to create audio analyzers
///
/// # Errors
///
/// Returns a `RecognitionError` if the analyzer backend fails to initialize.
pub async fn create_audio_analyzer(
    backend: AudioAnalyzerBackend,
) -> RecognitionResult<Arc<dyn AudioAnalyzer>> {
    let analyzer = AudioAnalyzerImpl::with_backend(backend).await?;
    Ok(Arc::new(analyzer))
}

/// Get recommended analyzer configuration for a specific use case
#[must_use]
pub fn recommended_config_for_use_case(use_case: AnalysisUseCase) -> AudioAnalysisConfig {
    match use_case {
        AnalysisUseCase::QualityAssessment => AudioAnalysisConfig {
            quality_metrics: true,
            prosody_analysis: false,
            speaker_analysis: false,
            emotional_analysis: false,
            quality_metrics_list: vec![
                AudioMetric::SNR,
                AudioMetric::THD,
                AudioMetric::SpectralCentroid,
                AudioMetric::RootMeanSquare,
            ],
            ..Default::default()
        },
        AnalysisUseCase::SpeechEvaluation => AudioAnalysisConfig {
            quality_metrics: true,
            prosody_analysis: true,
            speaker_analysis: false,
            emotional_analysis: false,
            ..Default::default()
        },
        AnalysisUseCase::SpeakerIdentification => AudioAnalysisConfig {
            quality_metrics: false,
            prosody_analysis: false,
            speaker_analysis: true,
            emotional_analysis: false,
            ..Default::default()
        },
        AnalysisUseCase::EmotionRecognition => AudioAnalysisConfig {
            quality_metrics: false,
            prosody_analysis: true,
            speaker_analysis: false,
            emotional_analysis: true,
            ..Default::default()
        },
        AnalysisUseCase::Comprehensive => AudioAnalysisConfig::default(),
    }
}

/// Analysis use cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisUseCase {
    /// Audio quality assessment and metrics
    QualityAssessment,
    /// Speech evaluation and analysis
    SpeechEvaluation,
    /// Speaker identification and characteristics
    SpeakerIdentification,
    /// Emotion recognition and sentiment analysis
    EmotionRecognition,
    /// Comprehensive analysis with all features
    Comprehensive,
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_audio_analyzer_creation() {
        let config = AudioAnalysisConfig::default();
        let analyzer = AudioAnalyzerImpl::new(config).await.unwrap();

        assert!(!analyzer.supported_metrics().is_empty());
        assert!(analyzer.supports_capability(AnalysisCapability::QualityMetrics));
        assert!(analyzer.supports_capability(AnalysisCapability::ProsodyAnalysis));
    }

    #[tokio::test]
    async fn test_comprehensive_analysis() {
        let backend = AudioAnalyzerBackend::Comprehensive {
            quality_metrics: true,
            prosody_analysis: true,
            speaker_analysis: true,
            emotional_analysis: true,
        };

        let analyzer = AudioAnalyzerImpl::with_backend(backend).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let result = analyzer.analyze(&audio, None).await.unwrap();

        // Should have some quality metrics
        assert!(!result.quality_metrics.is_empty());

        // Should have prosody analysis
        assert!(result.prosody.pitch.mean_f0 >= 0.0);

        // Should have processing duration
        assert!(result.processing_duration.is_some());
    }

    #[tokio::test]
    async fn test_quality_focused_analysis() {
        let backend = AudioAnalyzerBackend::QualityFocused {
            metrics: vec![AudioMetric::SNR, AudioMetric::THD],
        };

        let analyzer = AudioAnalyzerImpl::with_backend(backend).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let result = analyzer.analyze(&audio, None).await.unwrap();

        // Should focus on quality metrics
        assert!(!result.quality_metrics.is_empty());

        // Prosody should be default/empty since not requested
        assert_eq!(result.prosody.pitch.mean_f0, 0.0);
    }

    #[tokio::test]
    async fn test_use_case_configs() {
        let quality_config = recommended_config_for_use_case(AnalysisUseCase::QualityAssessment);
        assert!(quality_config.quality_metrics);
        assert!(!quality_config.emotional_analysis);

        let emotion_config = recommended_config_for_use_case(AnalysisUseCase::EmotionRecognition);
        assert!(emotion_config.emotional_analysis);
        assert!(emotion_config.prosody_analysis);

        let comprehensive_config = recommended_config_for_use_case(AnalysisUseCase::Comprehensive);
        assert!(comprehensive_config.quality_metrics);
        assert!(comprehensive_config.prosody_analysis);
        assert!(comprehensive_config.speaker_analysis);
        assert!(comprehensive_config.emotional_analysis);
    }
}
