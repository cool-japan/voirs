//! Core types for voice cloning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Voice cloning methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloningMethod {
    /// Few-shot adaptation using small amount of target data
    FewShot,
    /// Zero-shot cloning from single embedding
    ZeroShot,
    /// One-shot cloning from single audio sample
    OneShot,
    /// Fine-tuning based approach
    FineTuning,
    /// Voice conversion approach
    VoiceConversion,
    /// Hybrid approach combining multiple methods
    Hybrid,
    /// Cross-lingual voice cloning across different languages
    CrossLingual,
}

impl CloningMethod {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            CloningMethod::FewShot => "few_shot",
            CloningMethod::ZeroShot => "zero_shot",
            CloningMethod::OneShot => "one_shot",
            CloningMethod::FineTuning => "fine_tuning",
            CloningMethod::VoiceConversion => "voice_conversion",
            CloningMethod::Hybrid => "hybrid",
            CloningMethod::CrossLingual => "cross_lingual",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "few_shot" | "few-shot" => Some(CloningMethod::FewShot),
            "zero_shot" | "zero-shot" => Some(CloningMethod::ZeroShot),
            "one_shot" | "one-shot" => Some(CloningMethod::OneShot),
            "fine_tuning" | "fine-tuning" => Some(CloningMethod::FineTuning),
            "voice_conversion" | "voice-conversion" => Some(CloningMethod::VoiceConversion),
            "hybrid" => Some(CloningMethod::Hybrid),
            "cross_lingual" | "cross-lingual" => Some(CloningMethod::CrossLingual),
            _ => None,
        }
    }

    /// Get recommended minimum samples for this method
    pub fn min_samples(&self) -> usize {
        match self {
            CloningMethod::ZeroShot => 0,
            CloningMethod::OneShot => 1,
            CloningMethod::FewShot => 3,
            CloningMethod::VoiceConversion => 5,
            CloningMethod::FineTuning => 10,
            CloningMethod::Hybrid => 5,
            CloningMethod::CrossLingual => 5,
        }
    }

    /// Get recommended maximum samples for this method
    pub fn max_samples(&self) -> usize {
        match self {
            CloningMethod::ZeroShot => 1,
            CloningMethod::OneShot => 1,
            CloningMethod::FewShot => 20,
            CloningMethod::VoiceConversion => 50,
            CloningMethod::FineTuning => 1000,
            CloningMethod::Hybrid => 100,
            CloningMethod::CrossLingual => 30,
        }
    }
}

impl Default for CloningMethod {
    fn default() -> Self {
        CloningMethod::FewShot
    }
}

/// Voice sample data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceSample {
    /// Unique sample ID
    pub id: String,
    /// Audio data (PCM samples)
    pub audio: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Text transcript (if available)
    pub transcript: Option<String>,
    /// Language code (e.g., "en", "ja", "zh")
    pub language: Option<String>,
    /// Duration in seconds
    pub duration: f32,
    /// Audio quality score (0.0 to 1.0)
    pub quality_score: Option<f32>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Recording timestamp
    pub timestamp: SystemTime,
}

impl VoiceSample {
    /// Create a new voice sample
    pub fn new(id: String, audio: Vec<f32>, sample_rate: u32) -> Self {
        let duration = audio.len() as f32 / sample_rate as f32;

        Self {
            id,
            audio,
            sample_rate,
            transcript: None,
            language: None,
            duration,
            quality_score: None,
            metadata: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Set transcript
    pub fn with_transcript(mut self, transcript: String) -> Self {
        self.transcript = Some(transcript);
        self
    }

    /// Set language
    pub fn with_language(mut self, language: String) -> Self {
        self.language = Some(language);
        self
    }

    /// Set quality score
    pub fn with_quality_score(mut self, score: f32) -> Self {
        self.quality_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if sample is valid for cloning
    pub fn is_valid_for_cloning(&self) -> bool {
        !self.audio.is_empty()
            && self.sample_rate > 0
            && self.duration >= 0.5 // At least 0.5 seconds
            && self.duration <= 30.0 // At most 30 seconds
    }

    /// Get audio as normalized floating point samples
    pub fn get_normalized_audio(&self) -> Vec<f32> {
        if self.audio.is_empty() {
            return Vec::new();
        }

        // Find max amplitude
        let max_amplitude = self.audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        if max_amplitude == 0.0 {
            return self.audio.clone();
        }

        // Normalize to [-1.0, 1.0] with 0.95 headroom
        let scale = 0.95 / max_amplitude;
        self.audio.iter().map(|x| x * scale).collect()
    }

    /// Resample audio to target sample rate
    pub fn resample(&self, target_sample_rate: u32) -> crate::Result<VoiceSample> {
        if self.sample_rate == target_sample_rate {
            return Ok(self.clone());
        }

        // Simple linear interpolation resampling
        // In a production system, use a proper resampling library
        let ratio = target_sample_rate as f64 / self.sample_rate as f64;
        let new_length = (self.audio.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f64 / ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index.ceil() as usize).min(self.audio.len() - 1);

            if src_index_floor >= self.audio.len() {
                break;
            }

            let weight = src_index - src_index_floor as f64;
            let sample = if src_index_floor == src_index_ceil {
                self.audio[src_index_floor]
            } else {
                let a = self.audio[src_index_floor];
                let b = self.audio[src_index_ceil];
                a + (b - a) * weight as f32
            };

            resampled.push(sample);
        }

        let mut result = self.clone();
        result.audio = resampled;
        result.sample_rate = target_sample_rate;
        result.duration = result.audio.len() as f32 / target_sample_rate as f32;

        Ok(result)
    }

    /// Trim silence from beginning and end
    pub fn trim_silence(&self, threshold: f32) -> VoiceSample {
        if self.audio.is_empty() {
            return self.clone();
        }

        // Find start of speech
        let start = self
            .audio
            .iter()
            .position(|&x| x.abs() > threshold)
            .unwrap_or(0);

        // Find end of speech
        let end = self
            .audio
            .iter()
            .rposition(|&x| x.abs() > threshold)
            .map(|pos| pos + 1)
            .unwrap_or(self.audio.len());

        if start >= end {
            return self.clone();
        }

        let mut result = self.clone();
        result.audio = self.audio[start..end].to_vec();
        result.duration = result.audio.len() as f32 / self.sample_rate as f32;

        result
    }
}

/// Speaker profile containing voice characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerProfile {
    /// Unique speaker ID
    pub id: String,
    /// Speaker name
    pub name: String,
    /// Speaker characteristics
    pub characteristics: SpeakerCharacteristics,
    /// Voice samples for this speaker
    pub samples: Vec<VoiceSample>,
    /// Speaker embedding vector
    pub embedding: Option<Vec<f32>>,
    /// Supported languages
    pub languages: Vec<String>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub updated_at: SystemTime,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl SpeakerProfile {
    /// Create a new speaker profile
    pub fn new(id: String, name: String) -> Self {
        let now = SystemTime::now();

        Self {
            id,
            name,
            characteristics: SpeakerCharacteristics::default(),
            samples: Vec::new(),
            embedding: None,
            languages: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Add voice sample
    pub fn add_sample(&mut self, sample: VoiceSample) {
        self.samples.push(sample);
        self.updated_at = SystemTime::now();
    }

    /// Set speaker embedding
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
        self.updated_at = SystemTime::now();
    }

    /// Add supported language
    pub fn add_language(&mut self, language: String) {
        if !self.languages.contains(&language) {
            self.languages.push(language);
            self.updated_at = SystemTime::now();
        }
    }

    /// Get total duration of all samples
    pub fn total_duration(&self) -> f32 {
        self.samples.iter().map(|s| s.duration).sum()
    }

    /// Get number of valid samples
    pub fn valid_sample_count(&self) -> usize {
        self.samples
            .iter()
            .filter(|s| s.is_valid_for_cloning())
            .count()
    }

    /// Check if profile has sufficient data for cloning method
    pub fn has_sufficient_data(&self, method: CloningMethod) -> bool {
        let valid_count = self.valid_sample_count();
        valid_count >= method.min_samples()
    }

    /// Get samples for a specific language
    pub fn get_samples_for_language(&self, language: &str) -> Vec<&VoiceSample> {
        self.samples
            .iter()
            .filter(|s| s.language.as_ref().map(|l| l == language).unwrap_or(false))
            .collect()
    }

    /// Update characteristics from samples
    pub fn update_characteristics(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        // Calculate average characteristics from samples
        let mut total_pitch = 0.0;
        let mut total_energy = 0.0;
        let mut sample_count = 0;

        for sample in &self.samples {
            if let Some(pitch) = self.extract_average_pitch(sample) {
                total_pitch += pitch;
                sample_count += 1;
            }

            total_energy += self.calculate_rms_energy(sample);
        }

        if sample_count > 0 {
            self.characteristics.average_pitch = total_pitch / sample_count as f32;
            self.characteristics.average_energy = total_energy / self.samples.len() as f32;
        }

        self.updated_at = SystemTime::now();
    }

    /// Extract average pitch from sample (simplified)
    fn extract_average_pitch(&self, sample: &VoiceSample) -> Option<f32> {
        // Simplified pitch extraction - in reality would use proper pitch detection
        // This is just a placeholder
        if sample.audio.is_empty() {
            return None;
        }

        // Estimate based on zero crossings (very crude)
        let zero_crossings = sample
            .audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();

        let crossing_rate = zero_crossings as f32 / sample.duration;
        Some(crossing_rate * 0.5) // Very rough pitch estimate
    }

    /// Calculate RMS energy
    fn calculate_rms_energy(&self, sample: &VoiceSample) -> f32 {
        if sample.audio.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = sample.audio.iter().map(|x| x * x).sum();
        (sum_squares / sample.audio.len() as f32).sqrt()
    }
}

/// Speaker voice characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCharacteristics {
    /// Average fundamental frequency (Hz)
    pub average_pitch: f32,
    /// Pitch range (semitones)
    pub pitch_range: f32,
    /// Average energy level
    pub average_energy: f32,
    /// Speaking rate (words per minute)
    pub speaking_rate: f32,
    /// Voice quality parameters
    pub voice_quality: VoiceQuality,
    /// Accent/dialect information
    pub accent: Option<String>,
    /// Gender (if determinable)
    pub gender: Option<Gender>,
    /// Age group (if determinable)
    pub age_group: Option<AgeGroup>,
    /// Dynamic adaptive features for learning
    pub adaptive_features: HashMap<String, f32>,
}

impl SpeakerCharacteristics {
    /// Create characteristics with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate similarity to another speaker's characteristics
    pub fn similarity(&self, other: &Self) -> f32 {
        let pitch_sim = 1.0 - (self.average_pitch - other.average_pitch).abs() / 200.0;
        let energy_sim = 1.0 - (self.average_energy - other.average_energy).abs();
        let rate_sim = 1.0 - (self.speaking_rate - other.speaking_rate).abs() / 100.0;

        (pitch_sim + energy_sim + rate_sim) / 3.0
    }
}

impl Default for SpeakerCharacteristics {
    fn default() -> Self {
        Self {
            average_pitch: 150.0, // Default pitch around 150 Hz
            pitch_range: 12.0,    // Default range of one octave
            average_energy: 0.1,  // Default energy level
            speaking_rate: 150.0, // Default speaking rate
            voice_quality: VoiceQuality::default(),
            accent: None,
            gender: None,
            age_group: None,
            adaptive_features: HashMap::new(),
        }
    }
}

/// Voice quality characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceQuality {
    /// Breathiness level (0.0 to 1.0)
    pub breathiness: f32,
    /// Roughness level (0.0 to 1.0)
    pub roughness: f32,
    /// Brightness level (0.0 to 1.0)
    pub brightness: f32,
    /// Warmth level (0.0 to 1.0)
    pub warmth: f32,
    /// Resonance characteristics
    pub resonance: ResonanceProfile,
}

impl Default for VoiceQuality {
    fn default() -> Self {
        Self {
            breathiness: 0.1,
            roughness: 0.1,
            brightness: 0.5,
            warmth: 0.5,
            resonance: ResonanceProfile::default(),
        }
    }
}

/// Resonance profile for voice characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResonanceProfile {
    /// Formant frequencies (Hz)
    pub formants: Vec<f32>,
    /// Formant bandwidths (Hz)
    pub bandwidths: Vec<f32>,
    /// Nasality level (0.0 to 1.0)
    pub nasality: f32,
}

impl Default for ResonanceProfile {
    fn default() -> Self {
        Self {
            formants: vec![700.0, 1200.0, 2500.0], // Typical formants for neutral vowel
            bandwidths: vec![100.0, 120.0, 150.0], // Typical bandwidths
            nasality: 0.1,
        }
    }
}

/// Gender classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gender {
    /// Male voice
    Male,
    /// Female voice
    Female,
    /// Non-binary/other
    Other,
    /// Unknown/unclassified
    Unknown,
}

impl Default for Gender {
    fn default() -> Self {
        Gender::Unknown
    }
}

/// Age group classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeGroup {
    /// Child (under 12)
    Child,
    /// Teenager (12-19)
    Teen,
    /// Young adult (20-35)
    YoungAdult,
    /// Middle-aged (36-55)
    MiddleAged,
    /// Senior (55+)
    Senior,
    /// Unknown/unclassified
    Unknown,
}

impl Default for AgeGroup {
    fn default() -> Self {
        AgeGroup::Unknown
    }
}

/// Speaker data bundle for cloning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerData {
    /// Speaker profile
    pub profile: SpeakerProfile,
    /// Reference samples
    pub reference_samples: Vec<VoiceSample>,
    /// Target text for synthesis
    pub target_text: Option<String>,
    /// Target language
    pub target_language: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl SpeakerData {
    /// Create new speaker data
    pub fn new(profile: SpeakerProfile) -> Self {
        Self {
            profile,
            reference_samples: Vec::new(),
            target_text: None,
            target_language: None,
            context: HashMap::new(),
        }
    }

    /// Add reference sample
    pub fn add_reference_sample(mut self, sample: VoiceSample) -> Self {
        self.reference_samples.push(sample);
        self
    }

    /// Set target text
    pub fn with_target_text(mut self, text: String) -> Self {
        self.target_text = Some(text);
        self
    }

    /// Set target language
    pub fn with_target_language(mut self, language: String) -> Self {
        self.target_language = Some(language);
        self
    }

    /// Validate data for cloning
    pub fn validate(&self, method: CloningMethod) -> crate::Result<()> {
        // Check if we have sufficient samples
        let valid_samples = self
            .reference_samples
            .iter()
            .filter(|s| s.is_valid_for_cloning())
            .count();

        if valid_samples < method.min_samples() {
            return Err(crate::Error::InsufficientData(format!(
                "Need at least {} valid samples for {:?}, got {}",
                method.min_samples(),
                method,
                valid_samples
            )));
        }

        // Check total duration
        let total_duration: f32 = self.reference_samples.iter().map(|s| s.duration).sum();

        let min_duration = match method {
            CloningMethod::ZeroShot => 0.0,
            CloningMethod::OneShot => 1.0,
            CloningMethod::FewShot => 5.0,
            CloningMethod::VoiceConversion => 10.0,
            CloningMethod::FineTuning => 60.0,
            CloningMethod::Hybrid => 10.0,
            CloningMethod::CrossLingual => 15.0,
        };

        if total_duration < min_duration {
            return Err(crate::Error::InsufficientData(format!(
                "Need at least {} seconds of audio for {:?}, got {:.1}",
                min_duration, method, total_duration
            )));
        }

        Ok(())
    }
}

impl Default for SpeakerProfile {
    fn default() -> Self {
        Self {
            id: "default_speaker".to_string(),
            name: "Default Speaker".to_string(),
            characteristics: SpeakerCharacteristics::default(),
            samples: Vec::new(),
            embedding: None,
            languages: vec!["en".to_string()],
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }
}

impl Default for SpeakerData {
    fn default() -> Self {
        Self {
            profile: SpeakerProfile::default(),
            reference_samples: Vec::new(),
            target_text: None,
            target_language: None,
            context: HashMap::new(),
        }
    }
}

/// Voice cloning request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCloneRequest {
    /// Request ID
    pub id: String,
    /// Speaker data
    pub speaker_data: SpeakerData,
    /// Cloning method to use
    pub method: CloningMethod,
    /// Target text to synthesize
    pub text: String,
    /// Target language
    pub language: Option<String>,
    /// Quality level (0.0 to 1.0)
    pub quality_level: f32,
    /// Speed vs quality tradeoff (0.0 = fastest, 1.0 = highest quality)
    pub quality_tradeoff: f32,
    /// Custom parameters
    pub parameters: HashMap<String, f32>,
    /// Request timestamp
    pub timestamp: SystemTime,
}

impl VoiceCloneRequest {
    /// Create new cloning request
    pub fn new(id: String, speaker_data: SpeakerData, method: CloningMethod, text: String) -> Self {
        Self {
            id,
            speaker_data,
            method,
            text,
            language: None,
            quality_level: 0.8,
            quality_tradeoff: 0.7,
            parameters: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Set target language
    pub fn with_language(mut self, language: String) -> Self {
        self.language = Some(language);
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
        if self.text.is_empty() {
            return Err(crate::Error::Validation("Text cannot be empty".to_string()));
        }

        self.speaker_data.validate(self.method)?;

        Ok(())
    }
}

/// Cross-lingual voice cloning information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLingualInfo {
    /// Source language
    pub source_language: String,
    /// Target language
    pub target_language: String,
    /// Phonetic mapping accuracy
    pub phonetic_accuracy: f32,
    /// Language adaptation confidence
    pub adaptation_confidence: f32,
    /// Cross-lingual similarity score
    pub cross_lingual_similarity: f32,
}

/// Voice cloning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCloneResult {
    /// Request ID this result corresponds to
    pub request_id: String,
    /// Generated audio data
    pub audio: Vec<f32>,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f32>,
    /// Similarity score to original speaker
    pub similarity_score: f32,
    /// Processing time
    pub processing_time: Duration,
    /// Method used
    pub method_used: CloningMethod,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Cross-lingual information (if applicable)
    pub cross_lingual_info: Option<CrossLingualInfo>,
    /// Result timestamp
    pub timestamp: SystemTime,
}

impl VoiceCloneResult {
    /// Create successful result
    pub fn success(
        request_id: String,
        audio: Vec<f32>,
        sample_rate: u32,
        similarity_score: f32,
        processing_time: Duration,
        method_used: CloningMethod,
    ) -> Self {
        Self {
            request_id,
            audio,
            sample_rate,
            quality_metrics: HashMap::new(),
            similarity_score,
            processing_time,
            method_used,
            success: true,
            error_message: None,
            cross_lingual_info: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Create successful result with cross-lingual information
    pub fn success_with_cross_lingual(
        request_id: String,
        audio: Vec<f32>,
        sample_rate: u32,
        similarity_score: f32,
        processing_time: Duration,
        method_used: CloningMethod,
        cross_lingual_info: CrossLingualInfo,
    ) -> Self {
        Self {
            request_id,
            audio,
            sample_rate,
            quality_metrics: HashMap::new(),
            similarity_score,
            processing_time,
            method_used,
            success: true,
            error_message: None,
            cross_lingual_info: Some(cross_lingual_info),
            timestamp: SystemTime::now(),
        }
    }

    /// Create failed result
    pub fn failure(
        request_id: String,
        error_message: String,
        processing_time: Duration,
        method_used: CloningMethod,
    ) -> Self {
        Self {
            request_id,
            audio: Vec::new(),
            sample_rate: 0,
            quality_metrics: HashMap::new(),
            similarity_score: 0.0,
            processing_time,
            method_used,
            success: false,
            error_message: Some(error_message),
            cross_lingual_info: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Add quality metric
    pub fn with_quality_metric(mut self, name: String, value: f32) -> Self {
        self.quality_metrics.insert(name, value);
        self
    }

    /// Get audio duration in seconds
    pub fn duration(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.audio.len() as f32 / self.sample_rate as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloning_method_properties() {
        assert_eq!(CloningMethod::ZeroShot.min_samples(), 0);
        assert_eq!(CloningMethod::OneShot.min_samples(), 1);
        assert_eq!(CloningMethod::FewShot.min_samples(), 3);

        assert_eq!(
            CloningMethod::from_str("few-shot"),
            Some(CloningMethod::FewShot)
        );
        assert_eq!(CloningMethod::FewShot.as_str(), "few_shot");
    }

    #[test]
    fn test_voice_sample_creation() {
        let audio = vec![0.1, -0.2, 0.3, -0.4];
        let sample = VoiceSample::new("test".to_string(), audio.clone(), 16000);

        assert_eq!(sample.id, "test");
        assert_eq!(sample.audio, audio);
        assert_eq!(sample.sample_rate, 16000);
        assert_eq!(sample.duration, 4.0 / 16000.0);
    }

    #[test]
    fn test_voice_sample_validation() {
        let mut sample = VoiceSample::new("test".to_string(), vec![0.0; 8000], 16000); // 0.5 seconds
        assert!(sample.is_valid_for_cloning());

        sample.audio = vec![0.0; 1000]; // 0.0625 seconds - too short
        sample.duration = 1000.0 / 16000.0; // Update duration to match new audio length
        assert!(!sample.is_valid_for_cloning());

        sample.audio = vec![0.0; 16000 * 31]; // 31 seconds - too long
        sample.duration = 31.0;
        assert!(!sample.is_valid_for_cloning());
    }

    #[test]
    fn test_speaker_profile() {
        let mut profile = SpeakerProfile::new("speaker1".to_string(), "Test Speaker".to_string());

        let sample = VoiceSample::new("sample1".to_string(), vec![0.0; 16000], 16000);
        profile.add_sample(sample);

        assert_eq!(profile.samples.len(), 1);
        assert_eq!(profile.total_duration(), 1.0);
        assert_eq!(profile.valid_sample_count(), 1);
    }

    #[test]
    fn test_speaker_data_validation() {
        let profile = SpeakerProfile::new("speaker1".to_string(), "Test".to_string());
        let mut speaker_data = SpeakerData::new(profile);

        // Should fail with no samples
        assert!(speaker_data.validate(CloningMethod::FewShot).is_err());

        // Add sufficient samples
        for i in 0..5 {
            let sample = VoiceSample::new(
                format!("sample{}", i),
                vec![0.1; 16000], // 1 second each
                16000,
            );
            speaker_data.reference_samples.push(sample);
        }

        // Should pass now
        assert!(speaker_data.validate(CloningMethod::FewShot).is_ok());
    }

    #[test]
    fn test_voice_clone_request() {
        let profile = SpeakerProfile::new("speaker1".to_string(), "Test".to_string());
        let speaker_data = SpeakerData::new(profile);

        let request = VoiceCloneRequest::new(
            "req1".to_string(),
            speaker_data,
            CloningMethod::FewShot,
            "Hello world".to_string(),
        );

        assert_eq!(request.text, "Hello world");
        assert_eq!(request.method, CloningMethod::FewShot);
    }

    #[test]
    fn test_audio_normalization() {
        let audio = vec![0.5, -1.0, 0.25, -0.75];
        let sample = VoiceSample::new("test".to_string(), audio, 16000);

        let normalized = sample.get_normalized_audio();
        let max_amplitude = normalized.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        assert!(max_amplitude <= 0.95); // Should be normalized with headroom
    }

    #[test]
    fn test_audio_resampling() {
        let audio = vec![0.0, 1.0, 0.0, -1.0]; // 4 samples at 4Hz
        let sample = VoiceSample::new("test".to_string(), audio, 4);

        let resampled = sample.resample(8).unwrap(); // Upsample to 8Hz
        assert_eq!(resampled.sample_rate, 8);
        assert!(resampled.audio.len() > 4); // Should have more samples
    }
}
