//! Core types for voice conversion

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::{Duration, SystemTime};

/// Voice conversion types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ConversionType {
    /// Convert to a specific speaker
    #[default]
    SpeakerConversion,
    /// Transform age characteristics
    AgeTransformation,
    /// Transform gender characteristics
    GenderTransformation,
    /// General pitch shifting
    PitchShift,
    /// Speed/tempo transformation
    SpeedTransformation,
    /// Voice morphing between multiple sources
    VoiceMorphing,
    /// Emotional transformation
    EmotionalTransformation,
    /// Zero-shot conversion to unseen target voices
    ZeroShotConversion,
    /// Pass through with minimal processing (for testing)
    PassThrough,
    /// Custom transformation
    Custom(String),
}

impl ConversionType {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            ConversionType::SpeakerConversion => "speaker_conversion",
            ConversionType::AgeTransformation => "age_transformation",
            ConversionType::GenderTransformation => "gender_transformation",
            ConversionType::PitchShift => "pitch_shift",
            ConversionType::SpeedTransformation => "speed_transformation",
            ConversionType::VoiceMorphing => "voice_morphing",
            ConversionType::EmotionalTransformation => "emotional_transformation",
            ConversionType::ZeroShotConversion => "zero_shot_conversion",
            ConversionType::PassThrough => "pass_through",
            ConversionType::Custom(name) => name,
        }
    }

    /// Parse from string
    pub fn parse_type(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "speaker_conversion" => Some(ConversionType::SpeakerConversion),
            "age_transformation" => Some(ConversionType::AgeTransformation),
            "gender_transformation" => Some(ConversionType::GenderTransformation),
            "pitch_shift" => Some(ConversionType::PitchShift),
            "speed_transformation" => Some(ConversionType::SpeedTransformation),
            "voice_morphing" => Some(ConversionType::VoiceMorphing),
            "emotional_transformation" => Some(ConversionType::EmotionalTransformation),
            "zero_shot_conversion" => Some(ConversionType::ZeroShotConversion),
            "pass_through" => Some(ConversionType::PassThrough),
            _ => Some(ConversionType::Custom(s.to_string())),
        }
    }

    /// Check if conversion type supports real-time processing
    pub fn supports_realtime(&self) -> bool {
        match self {
            ConversionType::PitchShift => true,
            ConversionType::SpeedTransformation => true,
            ConversionType::SpeakerConversion => true,
            ConversionType::VoiceMorphing => false, // Requires complex processing
            ConversionType::AgeTransformation => true,
            ConversionType::GenderTransformation => true,
            ConversionType::EmotionalTransformation => true,
            ConversionType::ZeroShotConversion => false, // Requires complex analysis of unseen voices
            ConversionType::PassThrough => true,         // Fastest possible processing
            ConversionType::Custom(_) => false,          // Conservative default
        }
    }
}

impl FromStr for ConversionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_type(s).ok_or_else(|| format!("Unknown conversion type: {s}"))
    }
}

/// Voice characteristics for conversion targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VoiceCharacteristics {
    /// Fundamental frequency parameters
    pub pitch: PitchCharacteristics,
    /// Temporal characteristics
    pub timing: TimingCharacteristics,
    /// Spectral characteristics
    pub spectral: SpectralCharacteristics,
    /// Voice quality parameters
    pub quality: QualityCharacteristics,
    /// Age group
    pub age_group: Option<AgeGroup>,
    /// Gender
    pub gender: Option<Gender>,
    /// Accent/dialect
    pub accent: Option<String>,
    /// Custom characteristics
    pub custom_params: HashMap<String, f32>,
}

impl VoiceCharacteristics {
    /// Create new voice characteristics
    pub fn new() -> Self {
        Self::default()
    }

    /// Create characteristics for specific age
    pub fn for_age(age_group: AgeGroup) -> Self {
        let mut chars = Self::new();
        chars.age_group = Some(age_group);

        // Adjust characteristics based on age
        match age_group {
            AgeGroup::Child => {
                chars.pitch.mean_f0 = 250.0; // Higher pitch for children
                chars.timing.speaking_rate = 1.1; // Slightly faster
                chars.quality.breathiness = 0.2;
            }
            AgeGroup::Teen => {
                chars.pitch.mean_f0 = 200.0;
                chars.timing.speaking_rate = 1.2; // Faster speech
                chars.quality.roughness = 0.1;
            }
            AgeGroup::YoungAdult => {
                chars.pitch.mean_f0 = 150.0;
                chars.timing.speaking_rate = 1.0; // Normal rate
            }
            AgeGroup::Adult => {
                chars.pitch.mean_f0 = 145.0;
                chars.timing.speaking_rate = 0.98; // Slightly slower than young adult
                chars.quality.stability = 0.85;
            }
            AgeGroup::MiddleAged => {
                chars.pitch.mean_f0 = 140.0;
                chars.timing.speaking_rate = 0.95; // Slightly slower
                chars.quality.stability = 0.9;
            }
            AgeGroup::Senior => {
                chars.pitch.mean_f0 = 130.0;
                chars.timing.speaking_rate = 0.85; // Slower speech
                chars.quality.breathiness = 0.3;
                chars.quality.roughness = 0.2;
            }
            AgeGroup::Unknown => {}
        }

        chars
    }

    /// Create characteristics for specific gender
    pub fn for_gender(gender: Gender) -> Self {
        let mut chars = Self::new();
        chars.gender = Some(gender);

        // Adjust characteristics based on gender
        match gender {
            Gender::Male => {
                chars.pitch.mean_f0 = 120.0; // Lower pitch
                chars.spectral.formant_shift = -0.1; // Lower formants
                chars.quality.roughness = 0.15;
            }
            Gender::Female => {
                chars.pitch.mean_f0 = 200.0; // Higher pitch
                chars.spectral.formant_shift = 0.1; // Higher formants
                chars.quality.breathiness = 0.1;
            }
            Gender::NonBinary | Gender::Other | Gender::Unknown => {
                chars.pitch.mean_f0 = 160.0; // Neutral pitch
            }
        }

        chars
    }

    /// Interpolate between two voice characteristics
    pub fn interpolate(&self, other: &Self, factor: f32) -> Self {
        let t = factor.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        let mut result = self.clone();

        // Interpolate pitch characteristics
        result.pitch.mean_f0 = self.pitch.mean_f0 * inv_t + other.pitch.mean_f0 * t;
        result.pitch.range = self.pitch.range * inv_t + other.pitch.range * t;
        result.pitch.jitter = self.pitch.jitter * inv_t + other.pitch.jitter * t;

        // Interpolate timing characteristics
        result.timing.speaking_rate =
            self.timing.speaking_rate * inv_t + other.timing.speaking_rate * t;
        result.timing.pause_duration =
            self.timing.pause_duration * inv_t + other.timing.pause_duration * t;

        // Interpolate spectral characteristics
        result.spectral.formant_shift =
            self.spectral.formant_shift * inv_t + other.spectral.formant_shift * t;
        result.spectral.brightness =
            self.spectral.brightness * inv_t + other.spectral.brightness * t;

        // Interpolate quality characteristics
        result.quality.breathiness =
            self.quality.breathiness * inv_t + other.quality.breathiness * t;
        result.quality.roughness = self.quality.roughness * inv_t + other.quality.roughness * t;
        result.quality.stability = self.quality.stability * inv_t + other.quality.stability * t;

        // Interpolate custom parameters
        for (key, &value) in &self.custom_params {
            if let Some(&other_value) = other.custom_params.get(key) {
                result
                    .custom_params
                    .insert(key.clone(), value * inv_t + other_value * t);
            }
        }

        result
    }
}

/// Pitch-related characteristics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PitchCharacteristics {
    /// Mean fundamental frequency (Hz)
    pub mean_f0: f32,
    /// Pitch range (semitones)
    pub range: f32,
    /// Pitch jitter (0.0 to 1.0)
    pub jitter: f32,
    /// Pitch stability (0.0 to 1.0)
    pub stability: f32,
}

impl Default for PitchCharacteristics {
    fn default() -> Self {
        Self {
            mean_f0: 150.0,
            range: 12.0, // One octave
            jitter: 0.1,
            stability: 0.8,
        }
    }
}

/// Timing-related characteristics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimingCharacteristics {
    /// Speaking rate (relative to normal)
    pub speaking_rate: f32,
    /// Pause duration scale
    pub pause_duration: f32,
    /// Rhythm regularity (0.0 to 1.0)
    pub rhythm_regularity: f32,
}

impl Default for TimingCharacteristics {
    fn default() -> Self {
        Self {
            speaking_rate: 1.0,
            pause_duration: 1.0,
            rhythm_regularity: 0.7,
        }
    }
}

/// Spectral characteristics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralCharacteristics {
    /// Formant frequency shift (relative)
    pub formant_shift: f32,
    /// Spectral brightness (-1.0 to 1.0)
    pub brightness: f32,
    /// Spectral tilt
    pub spectral_tilt: f32,
    /// Harmonicity
    pub harmonicity: f32,
}

impl Default for SpectralCharacteristics {
    fn default() -> Self {
        Self {
            formant_shift: 0.0,
            brightness: 0.0,
            spectral_tilt: 0.0,
            harmonicity: 0.8,
        }
    }
}

/// Voice quality characteristics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityCharacteristics {
    /// Breathiness (0.0 to 1.0)
    pub breathiness: f32,
    /// Roughness (0.0 to 1.0)
    pub roughness: f32,
    /// Voice stability (0.0 to 1.0)
    pub stability: f32,
    /// Resonance quality (0.0 to 1.0)
    pub resonance: f32,
}

impl Default for QualityCharacteristics {
    fn default() -> Self {
        Self {
            breathiness: 0.1,
            roughness: 0.1,
            stability: 0.8,
            resonance: 0.7,
        }
    }
}

/// Age group classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AgeGroup {
    /// Child (under 12)
    Child,
    /// Teenager (12-19)
    Teen,
    /// Young adult (20-35)
    YoungAdult,
    /// Adult (20-55)
    Adult,
    /// Middle-aged (36-55)
    MiddleAged,
    /// Senior (55+)
    Senior,
    /// Unknown/unclassified
    #[default]
    Unknown,
}

/// Gender classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Gender {
    /// Male voice
    Male,
    /// Female voice
    Female,
    /// Non-binary voice
    NonBinary,
    /// Non-binary/other
    Other,
    /// Unknown/unclassified
    #[default]
    Unknown,
}

/// Conversion target specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversionTarget {
    /// Target voice characteristics
    pub characteristics: VoiceCharacteristics,
    /// Target speaker ID (if applicable)
    pub speaker_id: Option<String>,
    /// Reference audio samples (if available)
    pub reference_samples: Vec<AudioSample>,
    /// Conversion strength (0.0 to 1.0)
    pub strength: f32,
    /// Preserve original characteristics partially
    pub preserve_original: f32,
}

impl ConversionTarget {
    /// Create new conversion target
    pub fn new(characteristics: VoiceCharacteristics) -> Self {
        Self {
            characteristics,
            speaker_id: None,
            reference_samples: Vec::new(),
            strength: 1.0,
            preserve_original: 0.0,
        }
    }

    /// Set target speaker
    pub fn with_speaker_id(mut self, speaker_id: String) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }

    /// Add reference sample
    pub fn with_reference_sample(mut self, sample: AudioSample) -> Self {
        self.reference_samples.push(sample);
        self
    }

    /// Set conversion strength
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set preservation amount
    pub fn with_preservation(mut self, preserve: f32) -> Self {
        self.preserve_original = preserve.clamp(0.0, 1.0);
        self
    }
}

/// Audio sample for reference
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioSample {
    /// Sample ID
    pub id: String,
    /// Audio data (PCM samples)
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Sample metadata
    pub metadata: HashMap<String, String>,
}

impl AudioSample {
    /// Create new audio sample
    pub fn new(id: String, audio: Vec<f32>, sample_rate: u32) -> Self {
        let duration = audio.len() as f32 / sample_rate as f32;
        Self {
            id,
            audio,
            sample_rate,
            duration,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Voice conversion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionRequest {
    /// Request ID
    pub id: String,
    /// Source audio data
    pub source_audio: Vec<f32>,
    /// Source sample rate
    pub source_sample_rate: u32,
    /// Conversion type
    pub conversion_type: ConversionType,
    /// Conversion target
    pub target: ConversionTarget,
    /// Real-time processing flag
    pub realtime: bool,
    /// Quality level (0.0 to 1.0)
    pub quality_level: f32,
    /// Processing parameters
    pub parameters: HashMap<String, f32>,
    /// Request timestamp
    pub timestamp: SystemTime,
}

impl ConversionRequest {
    /// Create new conversion request
    pub fn new(
        id: String,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        conversion_type: ConversionType,
        target: ConversionTarget,
    ) -> Self {
        Self {
            id,
            source_audio,
            source_sample_rate,
            conversion_type,
            target,
            realtime: false,
            quality_level: 0.8,
            parameters: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Enable real-time processing
    pub fn with_realtime(mut self, realtime: bool) -> Self {
        self.realtime = realtime;
        self
    }

    /// Set quality level
    pub fn with_quality_level(mut self, level: f32) -> Self {
        self.quality_level = level.clamp(0.0, 1.0);
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: String, value: f32) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Validate the request
    pub fn validate(&self) -> crate::Result<()> {
        if self.source_audio.is_empty() {
            return Err(crate::Error::Validation {
                message: "Source audio cannot be empty".to_string(),
                field: Some("source_audio".to_string()),
                expected: Some("Non-empty audio data".to_string()),
                actual: Some("Empty audio data".to_string()),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Provide valid audio data".to_string(),
                    "Check audio file loading".to_string(),
                ]),
            });
        }

        if self.source_sample_rate == 0 {
            return Err(crate::Error::Validation {
                message: "Source sample rate must be positive".to_string(),
                field: Some("source_sample_rate".to_string()),
                expected: Some("Positive sample rate".to_string()),
                actual: Some(format!("{}", self.source_sample_rate)),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Set sample rate to a positive value (e.g., 44100, 48000)".to_string(),
                    "Check audio metadata".to_string(),
                ]),
            });
        }

        if self.realtime && !self.conversion_type.supports_realtime() {
            return Err(crate::Error::Validation {
                message: format!(
                    "Conversion type {:?} does not support real-time processing",
                    self.conversion_type
                ),
                field: Some("realtime".to_string()),
                expected: Some("False for non-realtime conversion types".to_string()),
                actual: Some("True".to_string()),
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Set realtime to false".to_string(),
                    "Use a different conversion type that supports real-time processing"
                        .to_string(),
                ]),
            });
        }

        Ok(())
    }

    /// Get source duration in seconds
    pub fn source_duration(&self) -> f32 {
        self.source_audio.len() as f32 / self.source_sample_rate as f32
    }
}

/// Voice conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionResult {
    /// Request ID this result corresponds to
    pub request_id: String,
    /// Converted audio data
    pub converted_audio: Vec<f32>,
    /// Output sample rate
    pub output_sample_rate: u32,
    /// Legacy conversion quality metrics (for compatibility)
    pub quality_metrics: HashMap<String, f32>,
    /// Comprehensive artifact detection results
    pub artifacts: Option<DetectedArtifacts>,
    /// Objective quality assessment results
    pub objective_quality: Option<ObjectiveQualityMetrics>,
    /// Processing time
    pub processing_time: Duration,
    /// Conversion type used
    pub conversion_type: ConversionType,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Result timestamp
    pub timestamp: SystemTime,
}

/// Detected artifacts in conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedArtifacts {
    /// Overall artifact score (0.0 = clean, 1.0 = heavily artifacted)
    pub overall_score: f32,
    /// Individual artifact types and their scores
    pub artifact_types: HashMap<String, f32>,
    /// Number of detected artifact locations
    pub artifact_count: usize,
    /// Quality assessment from artifacts
    pub quality_assessment: QualityAssessment,
}

/// Quality assessment for conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness: f32,
    /// Clarity score (0.0 to 1.0)
    pub clarity: f32,
    /// Consistency score (0.0 to 1.0)
    pub consistency: f32,
    /// Recommended quality adjustments
    pub recommended_adjustments: Vec<QualityAdjustment>,
}

/// Recommended quality adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAdjustment {
    /// Type of adjustment
    pub adjustment_type: String,
    /// Recommended strength (0.0 to 1.0)
    pub strength: f32,
    /// Expected improvement
    pub expected_improvement: f32,
}

/// Objective quality metrics for conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveQualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Spectral similarity score
    pub spectral_similarity: f32,
    /// Temporal consistency score
    pub temporal_consistency: f32,
    /// Prosodic preservation score
    pub prosodic_preservation: f32,
    /// Naturalness score
    pub naturalness: f32,
    /// Perceptual quality score
    pub perceptual_quality: f32,
    /// Signal-to-noise ratio estimate
    pub snr_estimate: f32,
    /// Segmental SNR
    pub segmental_snr: f32,
}

impl ConversionResult {
    /// Create successful result
    pub fn success(
        request_id: String,
        converted_audio: Vec<f32>,
        output_sample_rate: u32,
        processing_time: Duration,
        conversion_type: ConversionType,
    ) -> Self {
        Self {
            request_id,
            converted_audio,
            output_sample_rate,
            quality_metrics: HashMap::new(),
            artifacts: None,
            objective_quality: None,
            processing_time,
            conversion_type,
            success: true,
            error_message: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Create failed result
    pub fn failure(
        request_id: String,
        error_message: String,
        processing_time: Duration,
        conversion_type: ConversionType,
    ) -> Self {
        Self {
            request_id,
            converted_audio: Vec::new(),
            output_sample_rate: 0,
            quality_metrics: HashMap::new(),
            artifacts: None,
            objective_quality: None,
            processing_time,
            conversion_type,
            success: false,
            error_message: Some(error_message),
            timestamp: SystemTime::now(),
        }
    }

    /// Add quality metric
    pub fn with_quality_metric(mut self, name: String, value: f32) -> Self {
        self.quality_metrics.insert(name, value);
        self
    }

    /// Set artifact detection results
    pub fn with_artifacts(mut self, artifacts: DetectedArtifacts) -> Self {
        self.artifacts = Some(artifacts);
        self
    }

    /// Set objective quality metrics
    pub fn with_objective_quality(mut self, quality: ObjectiveQualityMetrics) -> Self {
        self.objective_quality = Some(quality);
        self
    }

    /// Get output duration in seconds
    pub fn output_duration(&self) -> f32 {
        if self.output_sample_rate == 0 {
            return 0.0;
        }
        self.converted_audio.len() as f32 / self.output_sample_rate as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_type_properties() {
        assert!(ConversionType::PitchShift.supports_realtime());
        assert!(ConversionType::SpeakerConversion.supports_realtime());
        assert!(!ConversionType::VoiceMorphing.supports_realtime());

        assert_eq!(ConversionType::PitchShift.as_str(), "pitch_shift");
        assert_eq!(
            ConversionType::from_str("pitch_shift").ok(),
            Some(ConversionType::PitchShift)
        );
    }

    #[test]
    fn test_voice_characteristics_age() {
        let child_chars = VoiceCharacteristics::for_age(AgeGroup::Child);
        let senior_chars = VoiceCharacteristics::for_age(AgeGroup::Senior);

        assert!(child_chars.pitch.mean_f0 > senior_chars.pitch.mean_f0);
        assert!(child_chars.timing.speaking_rate > senior_chars.timing.speaking_rate);
    }

    #[test]
    fn test_voice_characteristics_gender() {
        let male_chars = VoiceCharacteristics::for_gender(Gender::Male);
        let female_chars = VoiceCharacteristics::for_gender(Gender::Female);

        assert!(male_chars.pitch.mean_f0 < female_chars.pitch.mean_f0);
        assert!(male_chars.spectral.formant_shift < female_chars.spectral.formant_shift);
    }

    #[test]
    fn test_voice_characteristics_interpolation() {
        let chars1 = VoiceCharacteristics::for_gender(Gender::Male);
        let chars2 = VoiceCharacteristics::for_gender(Gender::Female);

        let interpolated = chars1.interpolate(&chars2, 0.5);

        let expected_f0 = (chars1.pitch.mean_f0 + chars2.pitch.mean_f0) / 2.0;
        assert!((interpolated.pitch.mean_f0 - expected_f0).abs() < 0.001);
    }

    #[test]
    fn test_conversion_target() {
        let chars = VoiceCharacteristics::for_age(AgeGroup::YoungAdult);
        let target = ConversionTarget::new(chars)
            .with_speaker_id("speaker123".to_string())
            .with_strength(0.8)
            .with_preservation(0.2);

        assert_eq!(target.speaker_id, Some("speaker123".to_string()));
        assert_eq!(target.strength, 0.8);
        assert_eq!(target.preserve_original, 0.2);
    }

    #[test]
    fn test_audio_sample() {
        let audio = vec![0.1, -0.2, 0.3, -0.4];
        let sample = AudioSample::new("test".to_string(), audio.clone(), 16000)
            .with_metadata("quality".to_string(), "high".to_string());

        assert_eq!(sample.audio, audio);
        assert_eq!(sample.sample_rate, 16000);
        assert_eq!(sample.duration, 4.0 / 16000.0);
        assert_eq!(sample.metadata.get("quality"), Some(&"high".to_string()));
    }

    #[test]
    fn test_conversion_request_validation() {
        let chars = VoiceCharacteristics::default();
        let target = ConversionTarget::new(chars);

        // Valid request
        let request = ConversionRequest::new(
            "req1".to_string(),
            vec![0.1, 0.2, 0.3],
            16000,
            ConversionType::PitchShift,
            target.clone(),
        );
        assert!(request.validate().is_ok());

        // Invalid - empty audio
        let invalid_request = ConversionRequest::new(
            "req2".to_string(),
            vec![],
            16000,
            ConversionType::PitchShift,
            target.clone(),
        );
        assert!(invalid_request.validate().is_err());

        // Invalid - realtime not supported
        let realtime_request = ConversionRequest::new(
            "req3".to_string(),
            vec![0.1, 0.2],
            16000,
            ConversionType::VoiceMorphing,
            target,
        )
        .with_realtime(true);
        assert!(realtime_request.validate().is_err());
    }

    #[test]
    fn test_conversion_result() {
        let result = ConversionResult::success(
            "req1".to_string(),
            vec![0.1, 0.2, 0.3, 0.4],
            22050,
            Duration::from_millis(100),
            ConversionType::PitchShift,
        );

        assert!(result.success);
        assert_eq!(result.output_duration(), 4.0 / 22050.0);
    }
}
