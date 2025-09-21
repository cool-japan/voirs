//! Basic forced alignment implementation
//!
//! This module provides a basic forced alignment implementation that can align
//! phonemes or text with audio using dynamic time warping and acoustic models.

use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme};

/// Basic forced alignment model implementation
pub struct ForcedAlignModel {
    /// Model configuration
    config: ForcedAlignConfig,
    /// Model state
    state: Arc<RwLock<ForcedAlignState>>,
    /// Supported languages
    supported_languages: Vec<LanguageCode>,
    /// Model metadata
    metadata: PhonemeRecognizerMetadata,
}

/// Forced alignment configuration
#[derive(Debug, Clone)]
pub struct ForcedAlignConfig {
    /// Path to the acoustic model
    pub model_path: String,
    /// Path to the pronunciation dictionary
    pub dictionary_path: Option<String>,
    /// Frame shift in milliseconds
    pub frame_shift_ms: f32,
    /// Frame length in milliseconds
    pub frame_length_ms: f32,
    /// Beam width for alignment
    pub beam_width: usize,
    /// Minimum phoneme duration in milliseconds
    pub min_phoneme_duration_ms: f32,
    /// Maximum phoneme duration in milliseconds
    pub max_phoneme_duration_ms: f32,
    /// Use GPU if available
    pub use_gpu: bool,
    /// Number of threads
    pub num_threads: usize,
    /// Confidence threshold for accepting alignments
    pub confidence_threshold: f32,
}

impl Default for ForcedAlignConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            dictionary_path: None,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            beam_width: 100,
            min_phoneme_duration_ms: 20.0,
            max_phoneme_duration_ms: 1000.0,
            use_gpu: false,
            num_threads: num_cpus::get(),
            confidence_threshold: 0.3,
        }
    }
}

/// Internal state for forced alignment model
struct ForcedAlignState {
    /// Whether the model is loaded
    loaded: bool,
    /// Model path
    model_path: String,
    /// Dictionary path
    dictionary_path: Option<String>,
    /// Pronunciation dictionary
    dictionary: HashMap<String, Vec<String>>,
    /// Loading time
    load_time: Option<Duration>,
    /// Alignment count
    alignment_count: usize,
    /// Total alignment time
    total_alignment_time: Duration,
}

impl ForcedAlignState {
    fn new(model_path: String, dictionary_path: Option<String>) -> Self {
        Self {
            loaded: false,
            model_path,
            dictionary_path,
            dictionary: HashMap::new(),
            load_time: None,
            alignment_count: 0,
            total_alignment_time: Duration::ZERO,
        }
    }
}

impl ForcedAlignModel {
    /// Create a new forced alignment model
    pub async fn new(
        model_path: String,
        dictionary_path: Option<String>,
    ) -> Result<Self, RecognitionError> {
        // Validate model file exists
        if !Path::new(&model_path).exists() {
            return Err(RecognitionError::ModelLoadError {
                message: format!("Model file not found: {}", model_path),
                source: None,
            });
        }

        // Validate dictionary file if provided
        if let Some(ref dict_path) = dictionary_path {
            if !Path::new(dict_path).exists() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!("Dictionary file not found: {}", dict_path),
                    source: None,
                });
            }
        }

        let config = ForcedAlignConfig {
            model_path: model_path.clone(),
            dictionary_path: dictionary_path.clone(),
            ..Default::default()
        };

        // Basic forced alignment supports common languages
        let supported_languages = vec![
            LanguageCode::EnUs,
            LanguageCode::EnGb,
            LanguageCode::DeDe,
            LanguageCode::FrFr,
            LanguageCode::EsEs,
        ];

        let metadata = PhonemeRecognizerMetadata {
            name: "Basic Forced Alignment".to_string(),
            version: "1.0.0".to_string(),
            description: "Basic forced alignment using dynamic time warping".to_string(),
            supported_languages: supported_languages.clone(),
            alignment_methods: vec![AlignmentMethod::Forced, AlignmentMethod::Hybrid],
            alignment_accuracy: 0.85,
            supported_features: vec![
                PhonemeRecognitionFeature::WordAlignment,
                PhonemeRecognitionFeature::CustomPronunciation,
                PhonemeRecognitionFeature::ConfidenceScoring,
                PhonemeRecognitionFeature::PronunciationAssessment,
            ],
        };

        let state = Arc::new(RwLock::new(ForcedAlignState::new(
            model_path,
            dictionary_path,
        )));

        Ok(Self {
            config,
            state,
            supported_languages,
            metadata,
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: ForcedAlignConfig) -> Result<Self, RecognitionError> {
        Self::new(config.model_path.clone(), config.dictionary_path.clone()).await
    }

    /// Load the model if not already loaded
    async fn ensure_loaded(&self) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;

        if !state.loaded {
            let start_time = Instant::now();

            tracing::info!("Loading forced alignment model: {}", state.model_path);

            // Load acoustic model (placeholder)
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Load pronunciation dictionary if provided
            if let Some(ref dict_path) = state.dictionary_path {
                tracing::info!("Loading pronunciation dictionary: {}", dict_path);
                state.dictionary = self.load_dictionary(dict_path).await?;
            } else {
                // Load default dictionary
                state.dictionary = self.load_default_dictionary().await?;
            }

            state.loaded = true;
            state.load_time = Some(start_time.elapsed());

            tracing::info!(
                "Forced alignment model loaded in {:?}",
                state.load_time.unwrap()
            );
        }

        Ok(())
    }

    /// Load pronunciation dictionary from file
    async fn load_dictionary(
        &self,
        _dict_path: &str,
    ) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
        // Placeholder implementation
        // In a real implementation, this would parse the dictionary file
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut dictionary = HashMap::new();

        // Add some common English words
        dictionary.insert(
            "HELLO".to_string(),
            vec![
                "HH".to_string(),
                "AH".to_string(),
                "L".to_string(),
                "OW".to_string(),
            ],
        );
        dictionary.insert(
            "WORLD".to_string(),
            vec![
                "W".to_string(),
                "ER".to_string(),
                "L".to_string(),
                "D".to_string(),
            ],
        );
        dictionary.insert(
            "TEST".to_string(),
            vec![
                "T".to_string(),
                "EH".to_string(),
                "S".to_string(),
                "T".to_string(),
            ],
        );
        dictionary.insert(
            "SPEECH".to_string(),
            vec![
                "S".to_string(),
                "P".to_string(),
                "IY".to_string(),
                "CH".to_string(),
            ],
        );

        Ok(dictionary)
    }

    /// Load default pronunciation dictionary
    async fn load_default_dictionary(
        &self,
    ) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
        // Use the same default dictionary
        self.load_dictionary("").await
    }

    /// Perform forced alignment using dynamic time warping
    async fn align_with_dtw(
        &self,
        audio: &AudioBuffer,
        phonemes: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        self.ensure_loaded().await?;

        let start_time = Instant::now();

        // Extract acoustic features
        let features = self.extract_features(audio).await?;

        // Perform DTW alignment
        let alignment = self.dtw_align(&features, phonemes, config).await?;

        // Update statistics
        let mut state = self.state.write().await;
        state.alignment_count += 1;
        state.total_alignment_time += start_time.elapsed();

        Ok(alignment)
    }

    /// Extract acoustic features from audio
    async fn extract_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        // Simulate feature extraction (MFCC, etc.)
        tokio::time::sleep(Duration::from_millis(10)).await;

        let samples = audio.samples();
        let frame_length =
            (self.config.frame_length_ms / 1000.0 * audio.sample_rate() as f32) as usize;
        let frame_shift =
            (self.config.frame_shift_ms / 1000.0 * audio.sample_rate() as f32) as usize;

        let mut features = Vec::new();
        let mut start = 0;

        while start + frame_length <= samples.len() {
            let frame = &samples[start..start + frame_length];
            let feature_vector = self.compute_mfcc(frame);
            features.push(feature_vector);
            start += frame_shift;
        }

        Ok(features)
    }

    /// Compute MFCC features for a frame
    fn compute_mfcc(&self, frame: &[f32]) -> Vec<f32> {
        // Simplified MFCC computation (placeholder)
        // In a real implementation, this would compute proper MFCC features
        let mut mfcc = vec![0.0; 13]; // 13 MFCC coefficients

        // Simple energy-based features as placeholder
        let energy: f32 = frame.iter().map(|x| x * x).sum();
        mfcc[0] = energy.ln().max(-10.0);

        // Add some variation based on spectral characteristics
        for i in 1..13 {
            let freq_band = i as f32 / 13.0;
            mfcc[i] = (energy * freq_band).sin() * 0.1;
        }

        mfcc
    }

    /// Perform DTW alignment between features and phonemes
    async fn dtw_align(
        &self,
        features: &[Vec<f32>],
        phonemes: &[Phoneme],
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        // Simulate DTW computation
        tokio::time::sleep(Duration::from_millis(50)).await;

        let frame_duration = self.config.frame_shift_ms / 1000.0;
        let total_duration = features.len() as f32 * frame_duration;

        // Create mock alignment
        let mut aligned_phonemes = Vec::new();
        let mut word_alignments = Vec::new();
        let mut current_time = 0.0;

        // Distribute phonemes evenly across the audio duration
        let phoneme_duration = total_duration / phonemes.len() as f32;

        for (i, phoneme) in phonemes.iter().enumerate() {
            let start_time = current_time;
            let end_time = current_time + phoneme_duration;

            // Calculate confidence based on mock acoustic model
            let confidence = self.calculate_alignment_confidence(features, i, phonemes.len());

            aligned_phonemes.push(AlignedPhoneme {
                phoneme: phoneme.clone(),
                start_time,
                end_time,
                confidence,
            });

            current_time = end_time;
        }

        // Create word alignments (simplified)
        if !aligned_phonemes.is_empty() {
            let word_alignment = WordAlignment {
                word: "test_word".to_string(),
                start_time: aligned_phonemes[0].start_time,
                end_time: aligned_phonemes.last().unwrap().end_time,
                phonemes: aligned_phonemes.clone(),
                confidence: aligned_phonemes.iter().map(|p| p.confidence).sum::<f32>()
                    / aligned_phonemes.len() as f32,
            };
            word_alignments.push(word_alignment);
        }

        let overall_confidence = aligned_phonemes.iter().map(|p| p.confidence).sum::<f32>()
            / aligned_phonemes.len() as f32;

        Ok(PhonemeAlignment {
            phonemes: aligned_phonemes,
            total_duration,
            alignment_confidence: overall_confidence,
            word_alignments,
        })
    }

    /// Calculate alignment confidence
    fn calculate_alignment_confidence(
        &self,
        _features: &[Vec<f32>],
        phoneme_index: usize,
        total_phonemes: usize,
    ) -> f32 {
        // Mock confidence calculation
        // In a real implementation, this would use acoustic model scores
        let base_confidence = 0.85;
        let position_factor = 1.0 - (phoneme_index as f32 / total_phonemes as f32 * 0.1);
        (base_confidence * position_factor).max(0.3).min(1.0)
    }

    /// Convert text to phonemes using dictionary
    async fn text_to_phonemes(
        &self,
        text: &str,
        _language: LanguageCode,
    ) -> Result<Vec<Phoneme>, RecognitionError> {
        self.ensure_loaded().await?;

        let state = self.state.read().await;
        let uppercase_text = text.to_uppercase();
        let words: Vec<&str> = uppercase_text.split_whitespace().collect();
        let mut phonemes = Vec::new();

        for word in words {
            if let Some(word_phonemes) = state.dictionary.get(word) {
                for phoneme_str in word_phonemes {
                    phonemes.push(Phoneme {
                        symbol: phoneme_str.clone(),
                        ipa_symbol: phoneme_str.clone(),
                        stress: 0,
                        syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                        duration_ms: None,
                        confidence: 1.0,
                    });
                }
            } else {
                // Handle unknown words with a simple fallback
                tracing::warn!("Unknown word in dictionary: {}", word);
                for char in word.chars() {
                    phonemes.push(Phoneme {
                        symbol: char.to_string(),
                        ipa_symbol: char.to_string(),
                        stress: 0,
                        syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                        duration_ms: None,
                        confidence: 0.5,
                    });
                }
            }
        }

        Ok(phonemes)
    }

    /// Get model statistics
    pub async fn get_stats(&self) -> ForcedAlignStats {
        let state = self.state.read().await;
        ForcedAlignStats {
            alignment_count: state.alignment_count,
            total_alignment_time: state.total_alignment_time,
            average_alignment_time: if state.alignment_count > 0 {
                state.total_alignment_time / state.alignment_count as u32
            } else {
                Duration::ZERO
            },
            load_time: state.load_time,
            dictionary_size: state.dictionary.len(),
        }
    }

    /// Add word to dictionary
    pub async fn add_word_to_dictionary(
        &self,
        word: String,
        phonemes: Vec<String>,
    ) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;
        state.dictionary.insert(word.to_uppercase(), phonemes);
        Ok(())
    }

    /// Get pronunciation for a word
    pub async fn get_pronunciation(
        &self,
        word: &str,
    ) -> Result<Option<Vec<String>>, RecognitionError> {
        self.ensure_loaded().await?;
        let state = self.state.read().await;
        Ok(state.dictionary.get(&word.to_uppercase()).cloned())
    }
}

/// Forced alignment model statistics
#[derive(Debug, Clone)]
pub struct ForcedAlignStats {
    /// Total number of alignments performed
    pub alignment_count: usize,
    /// Total alignment time
    pub total_alignment_time: Duration,
    /// Average alignment time
    pub average_alignment_time: Duration,
    /// Model load time
    pub load_time: Option<Duration>,
    /// Dictionary size
    pub dictionary_size: usize,
}

#[async_trait]
impl PhonemeRecognizer for ForcedAlignModel {
    async fn recognize_phonemes(
        &self,
        audio: &AudioBuffer,
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<Vec<Phoneme>> {
        // For recognition without expected phonemes, we need to use a different approach
        // This is a simplified implementation
        self.ensure_loaded().await?;

        let features = self.extract_features(audio).await.map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("Feature extraction failed: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        // Mock phoneme recognition based on audio characteristics
        let mut phonemes = Vec::new();
        let num_phonemes = (features.len() / 10).max(1); // Rough estimate

        for i in 0..num_phonemes {
            // Mock phoneme based on position
            let symbol = match i % 5 {
                0 => "AH",
                1 => "L",
                2 => "OW",
                3 => "W",
                _ => "ER",
            };

            phonemes.push(Phoneme {
                symbol: symbol.to_string(),
                ipa_symbol: symbol.to_string(),
                stress: 0, // No stress
                syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                duration_ms: Some(100.0),
                confidence: 0.8,
            });
        }

        Ok(phonemes)
    }

    async fn align_phonemes(
        &self,
        audio: &AudioBuffer,
        expected: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        self.align_with_dtw(audio, expected, config)
            .await
            .map_err(|e| e.into())
    }

    async fn align_text(
        &self,
        audio: &AudioBuffer,
        text: &str,
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        let language = config.map(|c| c.language).unwrap_or(LanguageCode::EnUs);
        let phonemes = self.text_to_phonemes(text, language).await.map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("Text to phoneme conversion failed: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        self.align_phonemes(audio, &phonemes, config).await
    }

    fn metadata(&self) -> PhonemeRecognizerMetadata {
        self.metadata.clone()
    }

    fn supports_feature(&self, feature: PhonemeRecognitionFeature) -> bool {
        self.metadata.supported_features.contains(&feature)
    }
}

impl Clone for ForcedAlignModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: self.state.clone(),
            supported_languages: self.supported_languages.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use voirs_sdk::AudioBuffer;

    fn create_mock_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "mock acoustic model data").unwrap();
        file
    }

    #[tokio::test]
    async fn test_forced_align_model_creation() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        assert_eq!(model.metadata.name, "Basic Forced Alignment");
        assert!(model.supported_languages.contains(&LanguageCode::EnUs));
    }

    #[tokio::test]
    async fn test_forced_align_missing_file() {
        let result = ForcedAlignModel::new("nonexistent.bin".to_string(), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_phoneme_recognition() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1); // 0.1 second of audio

        let result = model.recognize_phonemes(&audio, None).await.unwrap();
        assert!(!result.is_empty());
        assert!(result.iter().all(|p| p.confidence > 0.0));
    }

    #[tokio::test]
    async fn test_phoneme_alignment() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        let phonemes = vec![
            Phoneme {
                symbol: "H".to_string(),
                ipa_symbol: "H".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Onset,
                duration_ms: None,
                confidence: 1.0,
            },
            Phoneme {
                symbol: "AH".to_string(),
                ipa_symbol: "AH".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                duration_ms: None,
                confidence: 1.0,
            },
        ];

        let result = model.align_phonemes(&audio, &phonemes, None).await.unwrap();
        assert_eq!(result.phonemes.len(), 2);
        assert!(result.alignment_confidence > 0.0);
        assert!(result.total_duration > 0.0);
    }

    #[tokio::test]
    async fn test_text_alignment() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        let result = model.align_text(&audio, "HELLO", None).await.unwrap();
        assert!(!result.phonemes.is_empty());
        assert!(result.alignment_confidence > 0.0);
    }

    #[tokio::test]
    async fn test_dictionary_operations() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        // Test getting existing pronunciation
        let pronunciation = model.get_pronunciation("HELLO").await.unwrap();
        assert!(pronunciation.is_some());

        // Test adding new word
        let new_phonemes = vec![
            "T".to_string(),
            "EH".to_string(),
            "S".to_string(),
            "T".to_string(),
        ];
        model
            .add_word_to_dictionary("TESTING".to_string(), new_phonemes.clone())
            .await
            .unwrap();

        let retrieved = model.get_pronunciation("TESTING").await.unwrap();
        assert_eq!(retrieved, Some(new_phonemes));
    }

    #[tokio::test]
    async fn test_features() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        assert!(model.supports_feature(PhonemeRecognitionFeature::WordAlignment));
        assert!(model.supports_feature(PhonemeRecognitionFeature::CustomPronunciation));
        assert!(model.supports_feature(PhonemeRecognitionFeature::ConfidenceScoring));
        assert!(model.supports_feature(PhonemeRecognitionFeature::PronunciationAssessment));
    }

    #[tokio::test]
    async fn test_statistics() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        // Initial stats
        let stats = model.get_stats().await;
        assert_eq!(stats.alignment_count, 0);

        // After alignment
        let phonemes = vec![Phoneme {
            symbol: "H".to_string(),
            ipa_symbol: "H".to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Onset,
            duration_ms: None,
            confidence: 1.0,
        }];

        let _result = model.align_phonemes(&audio, &phonemes, None).await.unwrap();
        let stats = model.get_stats().await;
        assert_eq!(stats.alignment_count, 1);
        assert!(stats.total_alignment_time > Duration::ZERO);
    }
}
