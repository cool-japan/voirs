//! Multi-speaker voice conversion for singing voice synthesis
//!
//! This module provides capabilities for converting singing voices between different
//! speakers while preserving musical content, timing, and expression.

use crate::ai::StyleTransfer;
use crate::score::MusicalScore;
use crate::types::{SingingRequest, VoiceCharacteristics, VoiceType};
use crate::Error;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod quality;
mod signal;

pub use quality::VoiceQualityMetrics;

use quality::{
    analyze_voice_quality, calculate_audio_quality, calculate_content_preservation,
    calculate_naturalness, calculate_quality_metrics, calculate_speaker_similarity,
};
use signal::{
    estimate_average_f0, extract_formants, extract_speaker_features, formant_conversion,
    get_average_f0_for_voice_type, hybrid_conversion, neural_transfer_conversion,
    spectral_conversion,
};

/// Multi-speaker voice conversion system
#[derive(Debug, Clone)]
pub struct VoiceConverter {
    /// Pre-trained voice embeddings for different speakers
    speaker_embeddings: HashMap<String, SpeakerEmbedding>,
    /// Style transfer engine
    style_transfer: StyleTransfer,
    /// Conversion quality settings
    quality_settings: ConversionQuality,
}

/// Speaker embedding containing voice characteristics and model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Unique speaker identifier
    pub speaker_id: String,
    /// Speaker name or label
    pub speaker_name: String,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
    /// Neural embedding vector
    pub embedding_vector: Vec<f32>,
    /// Supported singing styles for this speaker
    pub supported_styles: Vec<String>,
    /// Average fundamental frequency
    pub avg_f0: f32,
    /// Formant frequencies
    pub formants: Vec<f32>,
    /// Voice quality metrics
    pub quality_metrics: VoiceQualityMetrics,
}

/// Voice conversion quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionQuality {
    /// Conversion method to use
    pub method: ConversionMethod,
    /// Preserve original timing
    pub preserve_timing: bool,
    /// Preserve original pitch contour shape
    pub preserve_pitch_contour: bool,
    /// Conversion strength (0.0-1.0)
    pub conversion_strength: f32,
    /// Enable formant preservation
    pub preserve_formants: bool,
    /// Enable expression preservation
    pub preserve_expression: bool,
}

/// Voice conversion methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversionMethod {
    /// Neural style transfer
    NeuralTransfer,
    /// Spectral envelope conversion
    SpectralConversion,
    /// Formant-based conversion
    FormantConversion,
    /// Hybrid approach combining multiple methods
    Hybrid,
}

/// Voice conversion request
#[derive(Debug, Clone)]
pub struct ConversionRequest {
    /// Source audio or singing request
    pub source: ConversionSource,
    /// Target speaker to convert to
    pub target_speaker: String,
    /// Conversion quality settings
    pub quality: ConversionQuality,
    /// Additional parameters
    pub parameters: HashMap<String, f32>,
}

/// Source for voice conversion
#[derive(Debug, Clone)]
pub enum ConversionSource {
    /// Audio samples with metadata
    Audio {
        /// Audio samples to convert
        samples: Vec<f32>,
        /// Sample rate of the audio in Hz
        sample_rate: u32,
        /// Optional speaker ID for the source audio
        speaker_id: Option<String>,
    },
    /// Singing request to be converted
    SingingRequest(Box<SingingRequest>),
    /// Musical score with source speaker
    Score {
        /// Musical score to synthesize and convert
        score: Box<MusicalScore>,
        /// Source speaker ID for initial synthesis
        source_speaker: String,
    },
}

/// Voice conversion result
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Converted audio samples
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Conversion quality metrics
    pub quality_metrics: ConversionQualityMetrics,
    /// Target speaker embedding used
    pub target_speaker: SpeakerEmbedding,
    /// Conversion parameters applied
    pub applied_parameters: HashMap<String, f32>,
}

/// Quality metrics for conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionQualityMetrics {
    /// Similarity to target speaker (0.0-1.0)
    pub speaker_similarity: f32,
    /// Preservation of musical content (0.0-1.0)
    pub content_preservation: f32,
    /// Audio quality score (0.0-1.0)
    pub audio_quality: f32,
    /// Naturalness score (0.0-1.0)
    pub naturalness: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

impl VoiceConverter {
    /// Creates a new voice converter with default CPU device.
    ///
    /// Initializes a voice converter with an empty speaker embedding database
    /// and default conversion quality settings.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `VoiceConverter` instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the style transfer engine fails to initialize.
    pub fn new() -> Result<Self, Error> {
        let device = Device::Cpu;
        let style_transfer = StyleTransfer::new(device)?;

        Ok(Self {
            speaker_embeddings: HashMap::new(),
            style_transfer,
            quality_settings: ConversionQuality::default(),
        })
    }

    /// Creates a new voice converter with a specified compute device.
    ///
    /// Allows explicit control over the compute device (CPU/GPU) used for
    /// neural network operations in the style transfer engine.
    ///
    /// # Arguments
    ///
    /// * `device` - The compute device to use (CPU or GPU)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `VoiceConverter` instance on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the style transfer engine fails to initialize on the specified device.
    pub fn new_with_device(device: Device) -> Result<Self, Error> {
        let style_transfer = StyleTransfer::new(device)?;

        Ok(Self {
            speaker_embeddings: HashMap::new(),
            style_transfer,
            quality_settings: ConversionQuality::default(),
        })
    }

    /// Adds a speaker embedding to the converter's database.
    ///
    /// Registers a new speaker that can be used as a target for voice conversion.
    /// The embedding must contain a 512-dimensional vector for compatibility.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The speaker embedding to add, containing voice characteristics and neural features
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the speaker was successfully added.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding vector is not exactly 512 dimensions.
    pub fn add_speaker(&mut self, embedding: SpeakerEmbedding) -> Result<(), Error> {
        if embedding.embedding_vector.len() != 512 {
            return Err(Error::Voice(
                "Speaker embedding must have 512 dimensions".to_string(),
            ));
        }

        self.speaker_embeddings
            .insert(embedding.speaker_id.clone(), embedding);
        Ok(())
    }

    /// Removes a speaker from the converter's database.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - The unique identifier of the speaker to remove
    ///
    /// # Returns
    ///
    /// Returns `Some(SpeakerEmbedding)` if the speaker was found and removed,
    /// or `None` if no speaker with the given ID exists.
    pub fn remove_speaker(&mut self, speaker_id: &str) -> Option<SpeakerEmbedding> {
        self.speaker_embeddings.remove(speaker_id)
    }

    /// Lists all available speaker IDs in the converter's database.
    ///
    /// # Returns
    ///
    /// Returns a vector of string slices containing all registered speaker IDs.
    pub fn list_speakers(&self) -> Vec<&str> {
        self.speaker_embeddings.keys().map(|s| s.as_str()).collect()
    }

    /// Retrieves a speaker embedding by its unique identifier.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - The unique identifier of the speaker to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some(&SpeakerEmbedding)` if the speaker exists,
    /// or `None` if no speaker with the given ID is registered.
    pub fn get_speaker(&self, speaker_id: &str) -> Option<&SpeakerEmbedding> {
        self.speaker_embeddings.get(speaker_id)
    }

    /// Converts voice from source to target speaker using the specified request parameters.
    ///
    /// Performs voice conversion by extracting source audio, applying the selected
    /// conversion method, and calculating quality metrics. The conversion preserves
    /// musical content and timing while adapting voice characteristics to match
    /// the target speaker.
    ///
    /// # Arguments
    ///
    /// * `request` - The conversion request containing source audio, target speaker,
    ///   quality settings, and additional parameters
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `ConversionResult` with converted audio,
    /// quality metrics, and applied parameters on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The target speaker ID is not found in the database
    /// - Source audio extraction fails
    /// - The conversion method encounters processing errors
    /// - Quality metric calculation fails
    pub async fn convert_voice(
        &self,
        request: ConversionRequest,
    ) -> Result<ConversionResult, Error> {
        let start_time = std::time::Instant::now();

        let target_speaker = self
            .speaker_embeddings
            .get(&request.target_speaker)
            .ok_or_else(|| {
                Error::Voice(format!(
                    "Unknown target speaker: {}",
                    request.target_speaker
                ))
            })?;

        let (source_audio, sample_rate, source_characteristics) =
            self.extract_source_audio(request.source).await?;

        let converted_audio = match request.quality.method {
            ConversionMethod::NeuralTransfer => neural_transfer_conversion(
                &source_audio,
                &source_characteristics,
                target_speaker,
                &request.quality,
            )?,
            ConversionMethod::SpectralConversion => {
                spectral_conversion(&source_audio, sample_rate, target_speaker, &request.quality)?
            }
            ConversionMethod::FormantConversion => {
                formant_conversion(&source_audio, sample_rate, target_speaker, &request.quality)?
            }
            ConversionMethod::Hybrid => hybrid_conversion(
                &source_audio,
                sample_rate,
                &source_characteristics,
                target_speaker,
                &request.quality,
            )?,
        };

        let quality_metrics = calculate_quality_metrics(
            &source_audio,
            &converted_audio,
            target_speaker,
            start_time.elapsed().as_millis() as f64,
        )?;

        Ok(ConversionResult {
            audio: converted_audio,
            sample_rate,
            quality_metrics,
            target_speaker: target_speaker.clone(),
            applied_parameters: request.parameters,
        })
    }

    async fn extract_source_audio(
        &self,
        source: ConversionSource,
    ) -> Result<(Vec<f32>, u32, VoiceCharacteristics), Error> {
        match source {
            ConversionSource::Audio {
                samples,
                sample_rate,
                speaker_id,
            } => {
                let characteristics = if let Some(id) = speaker_id {
                    self.speaker_embeddings
                        .get(&id)
                        .map(|s| s.voice_characteristics.clone())
                        .unwrap_or_default()
                } else {
                    VoiceCharacteristics::default()
                };
                Ok((samples, sample_rate, characteristics))
            }
            ConversionSource::SingingRequest(boxed_request) => {
                Ok((vec![0.0; 44100], 44100, boxed_request.voice))
            }
            ConversionSource::Score {
                score: _score,
                source_speaker,
            } => {
                let source_characteristics = self
                    .speaker_embeddings
                    .get(&source_speaker)
                    .map(|s| s.voice_characteristics.clone())
                    .unwrap_or_default();

                Ok((vec![0.0; 44100], 44100, source_characteristics))
            }
        }
    }

    /// Creates a speaker embedding from voice samples.
    ///
    /// Analyzes voice samples to extract speaker-specific features including
    /// neural embeddings, fundamental frequency, formants, and quality metrics.
    /// This is a static method that doesn't require a VoiceConverter instance.
    ///
    /// # Arguments
    ///
    /// * `speaker_id` - Unique identifier for the speaker
    /// * `speaker_name` - Human-readable name or label for the speaker
    /// * `voice_samples` - Audio samples of the speaker's voice
    /// * `voice_characteristics` - Known voice characteristics and metadata
    ///
    /// # Returns
    ///
    /// Returns a complete `SpeakerEmbedding` with extracted features and metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction fails or samples are insufficient.
    pub fn create_speaker_embedding(
        speaker_id: String,
        speaker_name: String,
        voice_samples: &[f32],
        voice_characteristics: VoiceCharacteristics,
    ) -> Result<SpeakerEmbedding, Error> {
        let embedding_vector = extract_speaker_features(voice_samples)?;
        let avg_f0 = estimate_average_f0(voice_samples)?;
        let formants = extract_formants(voice_samples)?;
        let quality_metrics = analyze_voice_quality(voice_samples)?;

        Ok(SpeakerEmbedding {
            speaker_id,
            speaker_name,
            voice_characteristics,
            embedding_vector,
            supported_styles: vec!["classical".to_string(), "pop".to_string()],
            avg_f0,
            formants,
            quality_metrics,
        })
    }
}

impl Default for VoiceConverter {
    /// Creates a default voice converter instance.
    ///
    /// # Panics
    ///
    /// Panics if the voice converter initialization fails.
    fn default() -> Self {
        Self::new().expect("Failed to create default VoiceConverter")
    }
}

impl Default for ConversionQuality {
    /// Creates default conversion quality settings.
    ///
    /// Uses hybrid conversion method with high quality preservation:
    /// - Method: Hybrid (neural + spectral)
    /// - Timing preservation: enabled
    /// - Pitch contour preservation: enabled
    /// - Conversion strength: 0.8 (80%)
    /// - Formant preservation: enabled
    /// - Expression preservation: enabled
    fn default() -> Self {
        Self {
            method: ConversionMethod::Hybrid,
            preserve_timing: true,
            preserve_pitch_contour: true,
            conversion_strength: 0.8,
            preserve_formants: true,
            preserve_expression: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_converter_creation() {
        let converter = VoiceConverter::new().expect("Failed to create VoiceConverter");
        assert!(converter.list_speakers().is_empty());
    }

    #[test]
    fn test_speaker_embedding_creation() {
        let voice_samples = vec![0.0; 44100];
        let voice_characteristics = VoiceCharacteristics::for_voice_type(VoiceType::Soprano);

        let embedding = VoiceConverter::create_speaker_embedding(
            "test_speaker".to_string(),
            "Test Speaker".to_string(),
            &voice_samples,
            voice_characteristics,
        );

        assert!(embedding.is_ok());
        let embedding = embedding.unwrap();
        assert_eq!(embedding.speaker_id, "test_speaker");
        assert_eq!(embedding.embedding_vector.len(), 512);
    }

    #[test]
    fn test_add_remove_speaker() {
        let mut converter = VoiceConverter::new().expect("Failed to create VoiceConverter");
        let voice_characteristics = VoiceCharacteristics::for_voice_type(VoiceType::Tenor);

        let embedding = SpeakerEmbedding {
            speaker_id: "tenor1".to_string(),
            speaker_name: "Test Tenor".to_string(),
            voice_characteristics,
            embedding_vector: vec![0.0; 512],
            supported_styles: vec!["classical".to_string()],
            avg_f0: 147.0,
            formants: vec![800.0, 1200.0, 2600.0],
            quality_metrics: VoiceQualityMetrics::default(),
        };

        assert!(converter.add_speaker(embedding.clone()).is_ok());
        assert_eq!(converter.list_speakers().len(), 1);
        assert!(converter.get_speaker("tenor1").is_some());

        let removed = converter.remove_speaker("tenor1");
        assert!(removed.is_some());
        assert_eq!(converter.list_speakers().len(), 0);
    }

    #[test]
    fn test_conversion_quality_default() {
        let quality = ConversionQuality::default();
        assert!(matches!(quality.method, ConversionMethod::Hybrid));
        assert!(quality.preserve_timing);
        assert_eq!(quality.conversion_strength, 0.8);
    }

    #[tokio::test]
    async fn test_voice_conversion_missing_speaker() {
        let converter = VoiceConverter::new().expect("Failed to create VoiceConverter");

        let request = ConversionRequest {
            source: ConversionSource::Audio {
                samples: vec![0.0; 1000],
                sample_rate: 44100,
                speaker_id: None,
            },
            target_speaker: "nonexistent".to_string(),
            quality: ConversionQuality::default(),
            parameters: HashMap::new(),
        };

        let result = converter.convert_voice(request).await;
        assert!(result.is_err());
    }

    fn make_test_speaker(avg_f0: f32, formants: Vec<f32>) -> SpeakerEmbedding {
        SpeakerEmbedding {
            speaker_id: "test".to_string(),
            speaker_name: "Test Speaker".to_string(),
            voice_characteristics: VoiceCharacteristics::default(),
            embedding_vector: vec![0.0; 512],
            supported_styles: vec!["pop".to_string()],
            avg_f0,
            formants,
            quality_metrics: VoiceQualityMetrics::default(),
        }
    }

    #[test]
    fn test_spectral_conversion_length_preserved() {
        let n = 8192usize;
        let source: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let speaker = make_test_speaker(220.0, vec![800.0, 1200.0, 2600.0]);
        let quality = ConversionQuality {
            method: ConversionMethod::SpectralConversion,
            conversion_strength: 1.0,
            ..ConversionQuality::default()
        };

        let result = spectral_conversion(&source, 44100, &speaker, &quality)
            .expect("spectral_conversion should succeed");

        assert_eq!(result.len(), n, "output length must equal input length");
    }

    #[test]
    fn test_spectral_conversion_not_constant_scale() {
        let n = 4096usize;
        let source: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 44100.0).sin() * 0.8)
            .collect();
        let formant_scale_old = 800.0f32 / 1000.0;
        let speaker = make_test_speaker(220.0, vec![800.0, 1200.0, 2600.0]);
        let quality = ConversionQuality {
            method: ConversionMethod::SpectralConversion,
            conversion_strength: 1.0,
            ..ConversionQuality::default()
        };

        let result = spectral_conversion(&source, 44100, &speaker, &quality)
            .expect("spectral_conversion should succeed");

        let differs = result
            .iter()
            .zip(source.iter())
            .any(|(&out, &inp)| (out - inp * formant_scale_old).abs() > 1e-4);
        assert!(
            differs,
            "spectral_conversion must not simply multiply by a constant scale"
        );
    }

    #[test]
    fn test_neural_transfer_shape() {
        let n = 3000usize;
        let source: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 44100.0).sin() * 0.6)
            .collect();
        let source_chars = VoiceCharacteristics {
            f0_mean: 147.0,
            ..VoiceCharacteristics::default()
        };
        let speaker = make_test_speaker(220.0, vec![800.0, 1200.0, 2600.0]);
        let quality = ConversionQuality {
            method: ConversionMethod::NeuralTransfer,
            conversion_strength: 0.8,
            ..ConversionQuality::default()
        };

        let result = neural_transfer_conversion(&source, &source_chars, &speaker, &quality)
            .expect("neural_transfer_conversion should succeed");

        assert_eq!(
            result.len(),
            n,
            "neural_transfer output length must equal input length"
        );
    }

    #[test]
    fn test_f0_estimation_440hz() {
        let sr = 44100usize;
        let duration_samples = sr * 2;
        let samples: Vec<f32> = (0..duration_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.9)
            .collect();

        let f0 = estimate_average_f0(&samples).expect("F0 estimation must succeed");

        assert!(
            (380.0..=500.0).contains(&f0),
            "Expected F0 near 440 Hz, got {f0:.1} Hz"
        );
    }

    #[test]
    fn test_f0_estimation_silence() {
        let samples = vec![0.0f32; 44100 * 2];
        let f0 = estimate_average_f0(&samples).expect("F0 estimation must succeed on silence");
        assert!(
            f0.abs() < 1e-6,
            "Expected 0.0 for silence (no voiced frames), got {f0:.1} Hz"
        );
    }

    #[test]
    fn test_formants_three_values() {
        let sr = 44100usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut s = 0.0f32;
                for h in 1..=20u32 {
                    s += (1.0 / h as f32)
                        * (2.0 * std::f32::consts::PI * 200.0 * h as f32 * t).sin();
                }
                s * 0.1
            })
            .collect();

        let formants = extract_formants(&samples).expect("extract_formants must succeed");

        assert_eq!(formants.len(), 3, "Must return exactly 3 formant values");
        for (idx, &f) in formants.iter().enumerate() {
            assert!(
                f > 50.0,
                "Formant F{} = {:.1} Hz is not above 50 Hz",
                idx + 1,
                f
            );
        }
    }

    #[test]
    fn test_speaker_features_512() {
        let sr = 44100usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();

        let features =
            extract_speaker_features(&samples).expect("extract_speaker_features must succeed");

        assert_eq!(features.len(), 512, "Must return exactly 512 features");
        for (i, &v) in features.iter().enumerate() {
            assert!(v.is_finite(), "Feature[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_speaker_features_varies() {
        let sr = 44100usize;
        let samples_a: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let samples_b: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 700.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();

        let feat_a = extract_speaker_features(&samples_a)
            .expect("extract_speaker_features must succeed for input A");
        let feat_b = extract_speaker_features(&samples_b)
            .expect("extract_speaker_features must succeed for input B");

        let max_diff = feat_a
            .iter()
            .zip(feat_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff > 1e-4,
            "Feature vectors for different inputs must differ (max_diff = {max_diff:.6})"
        );
    }

    #[test]
    fn test_f0_silence() {
        let samples = vec![0.0f32; 44100 * 2];
        let f0 = estimate_average_f0(&samples).expect("F0 estimation must succeed on silence");
        assert!(
            f0.abs() < 1e-6,
            "All-zero input must yield 0.0 (no voiced frames), got {f0:.1} Hz"
        );
    }

    #[test]
    fn test_f0_sine() {
        let sr = 44100usize;
        let target_hz = 220.0f32;
        let samples: Vec<f32> = (0..(sr * 2))
            .map(|i| (2.0 * std::f32::consts::PI * target_hz * i as f32 / sr as f32).sin() * 0.8)
            .collect();

        let f0 = estimate_average_f0(&samples).expect("F0 estimation must succeed for 220 Hz sine");

        assert!(
            (f0 - target_hz).abs() <= 20.0,
            "Expected F0 within ±20 Hz of {target_hz} Hz, got {f0:.1} Hz"
        );
    }

    #[test]
    fn test_formants_shape() {
        let sr = 44100usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.7)
            .collect();

        let formants = extract_formants(&samples).expect("extract_formants must succeed");

        assert_eq!(
            formants.len(),
            3,
            "extract_formants must return exactly 3 formants, got {}",
            formants.len()
        );
    }

    #[test]
    fn test_speaker_features_shape() {
        let sr = 44100usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 330.0 * i as f32 / sr as f32).sin() * 0.6)
            .collect();

        let features =
            extract_speaker_features(&samples).expect("extract_speaker_features must succeed");

        assert_eq!(
            features.len(),
            512,
            "extract_speaker_features must return exactly 512 elements, got {}",
            features.len()
        );
    }

    #[test]
    fn test_spectral_conversion_shape() {
        let n = 5000usize;
        let source: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 250.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let speaker = make_test_speaker(200.0, vec![800.0, 1200.0, 2600.0]);
        let quality = ConversionQuality {
            method: ConversionMethod::SpectralConversion,
            conversion_strength: 0.8,
            ..ConversionQuality::default()
        };

        let result = spectral_conversion(&source, 44100, &speaker, &quality)
            .expect("spectral_conversion must succeed");

        assert_eq!(
            result.len(),
            n,
            "spectral_conversion output length must equal input length: expected {n}, got {}",
            result.len()
        );
    }
}
