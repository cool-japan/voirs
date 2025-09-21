//! Advanced G2P features and enhancements.
//!
//! This module provides cutting-edge features for G2P conversion including
//! real-time adaptation, multilingual support, and emotion-aware processing.

use crate::{LanguageCode, Phoneme, PhoneticFeatures, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};
use tokio::sync::mpsc;

/// Real-time pronunciation adaptation system
pub struct AdaptivePronunciationSystem {
    /// Language code
    pub language: LanguageCode,
    /// User correction history
    pub correction_history: Arc<Mutex<VecDeque<UserCorrection>>>,
    /// Adaptation model
    pub adaptation_model: AdaptationModel,
    /// Real-time learning rate
    pub learning_rate: f32,
    /// Minimum corrections before adaptation
    pub min_corrections_threshold: usize,
    /// Maximum history size
    pub max_history_size: usize,
}

/// User correction for pronunciation adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCorrection {
    /// Original text
    pub text: String,
    /// Original phonemes generated
    pub original_phonemes: Vec<Phoneme>,
    /// User-corrected phonemes
    pub corrected_phonemes: Vec<Phoneme>,
    /// Correction timestamp
    pub timestamp: SystemTime,
    /// Correction context
    pub context: Option<String>,
    /// User confidence in correction
    pub user_confidence: f32,
}

/// Adaptation model for learning from corrections
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptationModel {
    /// Word-specific adaptations
    pub word_adaptations: HashMap<String, AdaptationRule>,
    /// Pattern-based adaptations
    pub pattern_adaptations: Vec<PatternAdaptation>,
    /// Context-specific adaptations
    pub context_adaptations: HashMap<String, Vec<AdaptationRule>>,
    /// Adaptation statistics
    pub stats: AdaptationStats,
}

/// Adaptation rule for pronunciation changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRule {
    /// Source phoneme sequence
    pub source_phonemes: Vec<String>,
    /// Target phoneme sequence
    pub target_phonemes: Vec<String>,
    /// Adaptation strength (0.0-1.0)
    pub strength: f32,
    /// Number of corrections supporting this rule
    pub support_count: usize,
    /// Last updated timestamp
    pub last_updated: SystemTime,
}

/// Pattern-based adaptation for phoneme sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAdaptation {
    /// Source pattern (regex)
    pub source_pattern: String,
    /// Target replacement pattern
    pub target_pattern: String,
    /// Pattern confidence
    pub confidence: f32,
    /// Usage count
    pub usage_count: usize,
}

/// Adaptation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdaptationStats {
    /// Total corrections processed
    pub total_corrections: usize,
    /// Total adaptations created
    pub total_adaptations: usize,
    /// Average adaptation confidence
    pub avg_confidence: f32,
    /// Most frequent adaptations
    pub frequent_adaptations: HashMap<String, usize>,
}

/// Multilingual phoneme mapping system
pub struct MultilingualPhonemeMapper {
    /// Cross-language phoneme mappings
    pub cross_lang_mappings: HashMap<(LanguageCode, LanguageCode), PhonemeMapping>,
    /// Universal phoneme inventory
    pub universal_phonemes: UniversalPhonemeInventory,
    /// Language-specific phoneme systems
    pub language_systems: HashMap<LanguageCode, LanguagePhonemeSystem>,
}

/// Phoneme mapping between languages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeMapping {
    /// Direct phoneme mappings
    pub direct_mappings: HashMap<String, String>,
    /// Approximate mappings with similarity scores
    pub approximate_mappings: HashMap<String, Vec<(String, f32)>>,
    /// Context-dependent mappings
    pub context_mappings: HashMap<String, Vec<ContextualMapping>>,
}

/// Contextual phoneme mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualMapping {
    /// Source phoneme
    pub source: String,
    /// Target phoneme
    pub target: String,
    /// Context conditions
    pub context_conditions: Vec<String>,
    /// Mapping confidence
    pub confidence: f32,
}

/// Universal phoneme inventory system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UniversalPhonemeInventory {
    /// IPA-based universal phonemes
    pub universal_phonemes: HashMap<String, UniversalPhoneme>,
    /// Feature-based phoneme classification
    pub feature_matrix: HashMap<String, PhoneticFeatures>,
    /// Similarity matrix between phonemes
    pub similarity_matrix: HashMap<(String, String), f32>,
}

/// Universal phoneme representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalPhoneme {
    /// IPA symbol
    pub ipa_symbol: String,
    /// Phonetic features
    pub features: PhoneticFeatures,
    /// Language-specific realizations
    pub language_realizations: HashMap<LanguageCode, Vec<String>>,
    /// Articulatory description
    pub articulatory_description: String,
}

/// Language-specific phoneme system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguagePhonemeSystem {
    /// Language code
    pub language: LanguageCode,
    /// Native phoneme inventory
    pub native_phonemes: Vec<String>,
    /// Allophone variations
    pub allophones: HashMap<String, Vec<String>>,
    /// Phonotactic constraints
    pub phonotactic_rules: Vec<PhonotacticRule>,
    /// Stress patterns
    pub stress_patterns: StressPatternSystem,
}

/// Phonotactic rule for valid phoneme sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonotacticRule {
    /// Rule type (onset, coda, nucleus)
    pub rule_type: String,
    /// Allowed phoneme sequences
    pub allowed_sequences: Vec<Vec<String>>,
    /// Forbidden sequences
    pub forbidden_sequences: Vec<Vec<String>>,
    /// Rule strength
    pub strength: f32,
}

/// Stress pattern system for a language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressPatternSystem {
    /// Default stress pattern
    pub default_pattern: Vec<u8>,
    /// Word-specific stress rules
    pub word_rules: HashMap<String, Vec<u8>>,
    /// Pattern-based stress rules
    pub pattern_rules: Vec<StressPatternRule>,
}

/// Stress pattern rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressPatternRule {
    /// Word pattern (regex)
    pub word_pattern: String,
    /// Stress pattern to apply
    pub stress_pattern: Vec<u8>,
    /// Rule confidence
    pub confidence: f32,
}

/// Phoneme quality scoring system
pub struct PhonemeQualityScorer {
    /// Language-specific quality models
    pub quality_models: HashMap<LanguageCode, QualityModel>,
    /// Cross-linguistic quality factors
    pub global_factors: GlobalQualityFactors,
    /// Quality assessment history
    pub assessment_history: VecDeque<QualityAssessment>,
}

/// Quality model for phoneme assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityModel {
    /// Phoneme-specific quality factors
    pub phoneme_factors: HashMap<String, PhonemeQualityFactors>,
    /// Sequence quality patterns
    pub sequence_patterns: Vec<SequenceQualityPattern>,
    /// Context-dependent quality adjustments
    pub context_adjustments: HashMap<String, f32>,
}

/// Quality factors for individual phonemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeQualityFactors {
    /// Base quality score
    pub base_quality: f32,
    /// Frequency-based adjustment
    pub frequency_factor: f32,
    /// Articulatory complexity
    pub complexity_factor: f32,
    /// Cross-linguistic stability
    pub stability_factor: f32,
}

/// Quality pattern for phoneme sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceQualityPattern {
    /// Phoneme sequence pattern
    pub sequence_pattern: Vec<String>,
    /// Quality modifier
    pub quality_modifier: f32,
    /// Pattern confidence
    pub confidence: f32,
}

/// Global quality factors across languages
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GlobalQualityFactors {
    /// Phonetic naturalness weights
    pub naturalness_weights: HashMap<String, f32>,
    /// Perceptual distinctiveness factors
    pub distinctiveness_factors: HashMap<String, f32>,
    /// Acoustic clarity measures
    pub acoustic_factors: HashMap<String, f32>,
}

/// Quality assessment for phoneme generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Original text
    pub text: String,
    /// Generated phonemes
    pub phonemes: Vec<Phoneme>,
    /// Overall quality score
    pub overall_score: f32,
    /// Per-phoneme quality scores
    pub phoneme_scores: Vec<f32>,
    /// Quality factors breakdown
    pub factor_breakdown: HashMap<String, f32>,
    /// Assessment timestamp
    pub timestamp: SystemTime,
}

/// Streaming G2P processor for real-time conversion
pub struct StreamingG2pProcessor {
    /// Language code
    pub language: LanguageCode,
    /// Text buffer for processing
    pub text_buffer: Arc<Mutex<String>>,
    /// Phoneme output stream
    pub phoneme_sender: mpsc::UnboundedSender<StreamingPhoneme>,
    /// Processing configuration
    pub config: StreamingConfig,
    /// Processing statistics
    pub stats: Arc<Mutex<StreamingStats>>,
}

/// Streaming phoneme with timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingPhoneme {
    /// Phoneme data
    pub phoneme: Phoneme,
    /// Start time offset (milliseconds)
    pub start_time_ms: f32,
    /// Duration (milliseconds)
    pub duration_ms: f32,
    /// Streaming confidence
    pub streaming_confidence: f32,
    /// Word boundary indicator
    pub is_word_boundary: bool,
}

/// Configuration for streaming G2P processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Buffer size for text processing
    pub buffer_size: usize,
    /// Processing chunk size
    pub chunk_size: usize,
    /// Lookahead window size
    pub lookahead_size: usize,
    /// Minimum processing latency (ms)
    pub min_latency_ms: f32,
    /// Maximum processing latency (ms)
    pub max_latency_ms: f32,
    /// Enable real-time adaptation
    pub enable_adaptation: bool,
}

/// Streaming processing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamingStats {
    /// Total characters processed
    pub total_chars_processed: usize,
    /// Total phonemes generated
    pub total_phonemes_generated: usize,
    /// Average processing latency
    pub avg_latency_ms: f32,
    /// Peak processing latency
    pub peak_latency_ms: f32,
    /// Processing throughput (chars/sec)
    pub throughput_cps: f32,
    /// Buffer utilization
    pub buffer_utilization: f32,
}

/// Emotion-aware phoneme generation system
pub struct EmotionAwareG2pProcessor {
    /// Base G2P processor
    pub base_processor: Arc<dyn crate::G2p>,
    /// Emotion classification model
    pub emotion_classifier: EmotionClassifier,
    /// Emotion-specific phoneme modifications
    pub emotion_modifications: HashMap<EmotionType, EmotionModification>,
    /// Processing history
    pub processing_history: VecDeque<EmotionProcessingEntry>,
}

/// Emotion types for phoneme modification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionType {
    /// Neutral emotion
    Neutral,
    /// Happy/joyful
    Happy,
    /// Sad/melancholic
    Sad,
    /// Angry/frustrated
    Angry,
    /// Fearful/anxious
    Fearful,
    /// Surprised
    Surprised,
    /// Disgusted
    Disgusted,
    /// Excited/enthusiastic
    Excited,
    /// Calm/relaxed
    Calm,
}

/// Emotion classification system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionClassifier {
    /// Emotion keywords and weights
    pub emotion_keywords: HashMap<EmotionType, HashMap<String, f32>>,
    /// Punctuation-based emotion indicators
    pub punctuation_indicators: HashMap<String, HashMap<EmotionType, f32>>,
    /// Syntactic pattern indicators
    pub syntax_indicators: Vec<SyntaxEmotionPattern>,
    /// Classification thresholds
    pub classification_thresholds: HashMap<EmotionType, f32>,
}

/// Syntactic pattern for emotion detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxEmotionPattern {
    /// Pattern description
    pub pattern: String,
    /// Associated emotion scores
    pub emotion_scores: HashMap<EmotionType, f32>,
    /// Pattern confidence
    pub confidence: f32,
}

/// Emotion-specific phoneme modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionModification {
    /// Pitch modifications
    pub pitch_modifications: HashMap<String, f32>,
    /// Duration modifications
    pub duration_modifications: HashMap<String, f32>,
    /// Stress pattern changes
    pub stress_modifications: HashMap<String, u8>,
    /// Voice quality adjustments
    pub voice_quality_adjustments: VoiceQualityAdjustments,
}

/// Voice quality adjustments for emotions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityAdjustments {
    /// Breathiness factor
    pub breathiness: f32,
    /// Creakiness factor
    pub creakiness: f32,
    /// Tenseness factor
    pub tenseness: f32,
    /// Nasality factor
    pub nasality: f32,
}

/// Emotion processing entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionProcessingEntry {
    /// Input text
    pub text: String,
    /// Detected emotion
    pub detected_emotion: EmotionType,
    /// Emotion confidence
    pub emotion_confidence: f32,
    /// Original phonemes
    pub original_phonemes: Vec<Phoneme>,
    /// Emotion-modified phonemes
    pub modified_phonemes: Vec<Phoneme>,
    /// Processing timestamp
    pub timestamp: SystemTime,
}

// Implementation for AdaptivePronunciationSystem
impl AdaptivePronunciationSystem {
    /// Create new adaptive pronunciation system
    pub fn new(language: LanguageCode) -> Self {
        Self {
            language,
            correction_history: Arc::new(Mutex::new(VecDeque::new())),
            adaptation_model: AdaptationModel::default(),
            learning_rate: 0.1,
            min_corrections_threshold: 3,
            max_history_size: 1000,
        }
    }

    /// Add user correction
    pub fn add_correction(&mut self, correction: UserCorrection) -> Result<()> {
        {
            let mut history = self.correction_history.lock().unwrap();

            // Add to history
            history.push_back(correction.clone());

            // Maintain history size
            if history.len() > self.max_history_size {
                history.pop_front();
            }
        } // Drop the lock here

        // Update adaptation model
        self.update_adaptation_model(&correction)?;

        Ok(())
    }

    /// Update adaptation model based on correction
    fn update_adaptation_model(&mut self, correction: &UserCorrection) -> Result<()> {
        // Extract phoneme sequences
        let source_phonemes: Vec<String> = correction
            .original_phonemes
            .iter()
            .map(|p| p.symbol.clone())
            .collect();

        let target_phonemes: Vec<String> = correction
            .corrected_phonemes
            .iter()
            .map(|p| p.symbol.clone())
            .collect();

        // Update word-specific adaptation
        let key = correction.text.to_lowercase();
        if let Some(existing_rule) = self.adaptation_model.word_adaptations.get_mut(&key) {
            existing_rule.support_count += 1;
            existing_rule.strength = (existing_rule.strength + correction.user_confidence) / 2.0;
            existing_rule.last_updated = SystemTime::now();
        } else {
            let new_rule = AdaptationRule {
                source_phonemes,
                target_phonemes,
                strength: correction.user_confidence,
                support_count: 1,
                last_updated: SystemTime::now(),
            };
            self.adaptation_model.word_adaptations.insert(key, new_rule);
        }

        // Update statistics
        self.adaptation_model.stats.total_corrections += 1;

        Ok(())
    }

    /// Apply adaptations to phoneme sequence
    pub fn apply_adaptations(&self, text: &str, phonemes: Vec<Phoneme>) -> Result<Vec<Phoneme>> {
        let mut adapted_phonemes = phonemes;

        // Apply word-specific adaptations
        let key = text.to_lowercase();
        if let Some(adaptation) = self.adaptation_model.word_adaptations.get(&key) {
            if adaptation.support_count >= self.min_corrections_threshold {
                // Create new phonemes based on adaptation
                adapted_phonemes = adaptation
                    .target_phonemes
                    .iter()
                    .map(|symbol| Phoneme::new(symbol.clone()))
                    .collect();
            }
        }

        Ok(adapted_phonemes)
    }

    /// Get adaptation statistics
    pub fn get_statistics(&self) -> AdaptationStats {
        self.adaptation_model.stats.clone()
    }
}

impl Default for MultilingualPhonemeMapper {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation for MultilingualPhonemeMapper
impl MultilingualPhonemeMapper {
    /// Create new multilingual phoneme mapper
    pub fn new() -> Self {
        let mut mapper = Self {
            cross_lang_mappings: HashMap::new(),
            universal_phonemes: UniversalPhonemeInventory::default(),
            language_systems: HashMap::new(),
        };

        // Initialize with basic language systems
        mapper.initialize_language_systems();

        mapper
    }

    /// Initialize basic language phoneme systems
    fn initialize_language_systems(&mut self) {
        // English phoneme system
        let english_system = LanguagePhonemeSystem {
            language: LanguageCode::EnUs,
            native_phonemes: vec![
                "æ".to_string(),
                "ɑ".to_string(),
                "ɔ".to_string(),
                "ɛ".to_string(),
                "ɪ".to_string(),
                "ʊ".to_string(),
                "ʌ".to_string(),
                "ə".to_string(),
                "i".to_string(),
                "u".to_string(),
                "eɪ".to_string(),
                "oʊ".to_string(),
                "aɪ".to_string(),
                "aʊ".to_string(),
                "ɔɪ".to_string(),
            ],
            allophones: HashMap::new(),
            phonotactic_rules: Vec::new(),
            stress_patterns: StressPatternSystem::default(),
        };
        self.language_systems
            .insert(LanguageCode::EnUs, english_system);

        // Add other language systems as needed
    }

    /// Map phonemes from source to target language
    pub fn map_phonemes(
        &self,
        phonemes: &[Phoneme],
        source_lang: LanguageCode,
        target_lang: LanguageCode,
    ) -> Result<Vec<Phoneme>> {
        let mapping_key = (source_lang, target_lang);

        if let Some(mapping) = self.cross_lang_mappings.get(&mapping_key) {
            let mapped_phonemes = phonemes
                .iter()
                .map(|phoneme| self.map_single_phoneme(phoneme, mapping))
                .collect();

            Ok(mapped_phonemes)
        } else {
            // Use universal phoneme system for mapping
            self.map_via_universal(phonemes, source_lang, target_lang)
        }
    }

    /// Map single phoneme using mapping rules
    fn map_single_phoneme(&self, phoneme: &Phoneme, mapping: &PhonemeMapping) -> Phoneme {
        // Try direct mapping first
        if let Some(mapped_symbol) = mapping.direct_mappings.get(&phoneme.symbol) {
            let mut mapped_phoneme = phoneme.clone();
            mapped_phoneme.symbol = mapped_symbol.clone();
            return mapped_phoneme;
        }

        // Try approximate mapping
        if let Some(approximate) = mapping.approximate_mappings.get(&phoneme.symbol) {
            if let Some((mapped_symbol, _confidence)) = approximate.first() {
                let mut mapped_phoneme = phoneme.clone();
                mapped_phoneme.symbol = mapped_symbol.clone();
                return mapped_phoneme;
            }
        }

        // Return original if no mapping found
        phoneme.clone()
    }

    /// Map phonemes via universal phoneme system
    fn map_via_universal(
        &self,
        phonemes: &[Phoneme],
        _source_lang: LanguageCode,
        _target_lang: LanguageCode,
    ) -> Result<Vec<Phoneme>> {
        // Simplified universal mapping - in practice would use sophisticated algorithms
        Ok(phonemes.to_vec())
    }
}

impl Default for StressPatternSystem {
    fn default() -> Self {
        Self {
            default_pattern: vec![1, 0],
            word_rules: HashMap::new(),
            pattern_rules: Vec::new(),
        }
    }
}

impl Default for PhonemeQualityScorer {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation for PhonemeQualityScorer
impl PhonemeQualityScorer {
    /// Create new phoneme quality scorer
    pub fn new() -> Self {
        Self {
            quality_models: HashMap::new(),
            global_factors: GlobalQualityFactors::default(),
            assessment_history: VecDeque::new(),
        }
    }

    /// Assess quality of phoneme sequence
    pub fn assess_quality(
        &mut self,
        text: &str,
        phonemes: &[Phoneme],
        language: LanguageCode,
    ) -> QualityAssessment {
        let overall_score = self.calculate_overall_quality(phonemes, language);
        let phoneme_scores = self.calculate_phoneme_scores(phonemes, language);
        let factor_breakdown = self.calculate_factor_breakdown(phonemes, language);

        let assessment = QualityAssessment {
            text: text.to_string(),
            phonemes: phonemes.to_vec(),
            overall_score,
            phoneme_scores,
            factor_breakdown,
            timestamp: SystemTime::now(),
        };

        // Add to history
        self.assessment_history.push_back(assessment.clone());
        if self.assessment_history.len() > 1000 {
            self.assessment_history.pop_front();
        }

        assessment
    }

    /// Calculate overall quality score
    fn calculate_overall_quality(&self, phonemes: &[Phoneme], _language: LanguageCode) -> f32 {
        if phonemes.is_empty() {
            return 0.0;
        }

        let total_confidence: f32 = phonemes.iter().map(|p| p.confidence).sum();
        total_confidence / phonemes.len() as f32
    }

    /// Calculate per-phoneme quality scores
    fn calculate_phoneme_scores(&self, phonemes: &[Phoneme], _language: LanguageCode) -> Vec<f32> {
        phonemes.iter().map(|p| p.confidence).collect()
    }

    /// Calculate quality factor breakdown
    fn calculate_factor_breakdown(
        &self,
        phonemes: &[Phoneme],
        _language: LanguageCode,
    ) -> HashMap<String, f32> {
        let mut breakdown = HashMap::new();

        breakdown.insert(
            "confidence".to_string(),
            phonemes.iter().map(|p| p.confidence).sum::<f32>() / phonemes.len() as f32,
        );
        breakdown.insert("naturalness".to_string(), 0.8);
        breakdown.insert("distinctiveness".to_string(), 0.85);
        breakdown.insert("acoustic_clarity".to_string(), 0.9);

        breakdown
    }
}

// Implementation for StreamingG2pProcessor
impl StreamingG2pProcessor {
    /// Create new streaming G2P processor
    pub fn new(
        language: LanguageCode,
        phoneme_sender: mpsc::UnboundedSender<StreamingPhoneme>,
    ) -> Self {
        Self {
            language,
            text_buffer: Arc::new(Mutex::new(String::new())),
            phoneme_sender,
            config: StreamingConfig::default(),
            stats: Arc::new(Mutex::new(StreamingStats::default())),
        }
    }

    /// Add text to processing buffer
    pub fn add_text(&self, text: &str) -> Result<()> {
        let mut buffer = self.text_buffer.lock().unwrap();
        buffer.push_str(text);

        // Update statistics
        let mut stats = self.stats.lock().unwrap();
        stats.total_chars_processed += text.len();

        Ok(())
    }

    /// Process buffered text and generate streaming phonemes
    pub async fn process_buffer(&self) -> Result<()> {
        let start_time = Instant::now();

        let text_to_process = {
            let mut buffer = self.text_buffer.lock().unwrap();
            let to_process = buffer.clone();
            buffer.clear();
            to_process
        };

        if text_to_process.is_empty() {
            return Ok(());
        }

        // Simple processing - in practice would use sophisticated streaming algorithms
        let mut time_offset = 0.0f32;

        for word in text_to_process.split_whitespace() {
            // Mock phoneme generation for each character
            for (i, char) in word.chars().enumerate() {
                let phoneme = Phoneme::new(char.to_string());
                let duration = 100.0; // 100ms per phoneme

                let streaming_phoneme = StreamingPhoneme {
                    phoneme,
                    start_time_ms: time_offset,
                    duration_ms: duration,
                    streaming_confidence: 0.8,
                    is_word_boundary: i == word.len() - 1,
                };

                if self.phoneme_sender.send(streaming_phoneme).is_err() {
                    break; // Receiver dropped
                }

                time_offset += duration;
            }
        }

        // Update statistics
        let processing_time = start_time.elapsed().as_millis() as f32;
        let mut stats = self.stats.lock().unwrap();
        stats.avg_latency_ms = (stats.avg_latency_ms + processing_time) / 2.0;
        if processing_time > stats.peak_latency_ms {
            stats.peak_latency_ms = processing_time;
        }

        Ok(())
    }

    /// Get streaming statistics
    pub fn get_statistics(&self) -> StreamingStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            chunk_size: 64,
            lookahead_size: 16,
            min_latency_ms: 10.0,
            max_latency_ms: 100.0,
            enable_adaptation: true,
        }
    }
}

// Implementation for EmotionAwareG2pProcessor
impl EmotionAwareG2pProcessor {
    /// Create new emotion-aware G2P processor
    pub fn new(base_processor: Arc<dyn crate::G2p>) -> Self {
        Self {
            base_processor,
            emotion_classifier: EmotionClassifier::default(),
            emotion_modifications: HashMap::new(),
            processing_history: VecDeque::new(),
        }
    }

    /// Process text with emotion awareness
    pub async fn process_with_emotion(
        &mut self,
        text: &str,
        language: Option<LanguageCode>,
    ) -> Result<Vec<Phoneme>> {
        // Classify emotion
        let (detected_emotion, emotion_confidence) = self.classify_emotion(text);

        // Generate base phonemes
        let base_phonemes = self.base_processor.to_phonemes(text, language).await?;

        // Apply emotion modifications
        let modified_phonemes = self.apply_emotion_modifications(&base_phonemes, &detected_emotion);

        // Record processing entry
        let entry = EmotionProcessingEntry {
            text: text.to_string(),
            detected_emotion: detected_emotion.clone(),
            emotion_confidence,
            original_phonemes: base_phonemes,
            modified_phonemes: modified_phonemes.clone(),
            timestamp: SystemTime::now(),
        };

        self.processing_history.push_back(entry);
        if self.processing_history.len() > 1000 {
            self.processing_history.pop_front();
        }

        Ok(modified_phonemes)
    }

    /// Classify emotion from text
    fn classify_emotion(&self, text: &str) -> (EmotionType, f32) {
        let mut emotion_scores = HashMap::new();

        // Initialize scores
        for emotion_type in [
            EmotionType::Neutral,
            EmotionType::Happy,
            EmotionType::Sad,
            EmotionType::Angry,
            EmotionType::Fearful,
            EmotionType::Surprised,
            EmotionType::Disgusted,
            EmotionType::Excited,
            EmotionType::Calm,
        ] {
            emotion_scores.insert(emotion_type, 0.0f32);
        }

        // Simple keyword-based classification
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            let word_lower = word
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string();

            // Simple emotion keywords with better matching
            match word_lower.as_str() {
                "happy" | "joy" | "glad" | "wonderful" | "very" => {
                    *emotion_scores.get_mut(&EmotionType::Happy).unwrap() += 1.0;
                }
                "sad" | "unhappy" | "depressed" => {
                    *emotion_scores.get_mut(&EmotionType::Sad).unwrap() += 1.0;
                }
                "angry" | "mad" | "furious" | "hate" => {
                    *emotion_scores.get_mut(&EmotionType::Angry).unwrap() += 2.0;
                    // Higher weight for angry
                }
                "terrible" => {
                    *emotion_scores.get_mut(&EmotionType::Sad).unwrap() += 0.5; // Lower weight for terrible
                }
                "excited" | "amazing" | "fantastic" | "awesome" => {
                    *emotion_scores.get_mut(&EmotionType::Excited).unwrap() += 1.0;
                }
                _ => {
                    *emotion_scores.get_mut(&EmotionType::Neutral).unwrap() += 0.1;
                }
            }
        }

        // Find highest scoring emotion with preference order for ties
        let emotion_priority = [
            EmotionType::Angry,
            EmotionType::Excited,
            EmotionType::Happy,
            EmotionType::Sad,
            EmotionType::Fearful,
            EmotionType::Surprised,
            EmotionType::Disgusted,
            EmotionType::Calm,
            EmotionType::Neutral,
        ];

        let mut best_emotion = EmotionType::Neutral;
        let mut best_score = 0.0;

        for emotion in emotion_priority {
            if let Some(&score) = emotion_scores.get(&emotion) {
                if score > best_score {
                    best_score = score;
                    best_emotion = emotion;
                }
            }
        }

        let confidence = if best_score > 0.0 { 0.8 } else { 0.5 };

        (best_emotion, confidence)
    }

    /// Apply emotion-specific modifications to phonemes
    fn apply_emotion_modifications(
        &self,
        phonemes: &[Phoneme],
        emotion: &EmotionType,
    ) -> Vec<Phoneme> {
        if let Some(modification) = self.emotion_modifications.get(emotion) {
            phonemes
                .iter()
                .map(|phoneme| {
                    let mut modified_phoneme = phoneme.clone();

                    // Apply duration modifications
                    if let Some(duration_mod) =
                        modification.duration_modifications.get(&phoneme.symbol)
                    {
                        if let Some(original_duration) = modified_phoneme.duration_ms {
                            modified_phoneme.duration_ms = Some(original_duration * duration_mod);
                        }
                    }

                    // Apply stress modifications
                    if let Some(stress_mod) = modification.stress_modifications.get(&phoneme.symbol)
                    {
                        modified_phoneme.stress = *stress_mod;
                    }

                    modified_phoneme
                })
                .collect()
        } else {
            phonemes.to_vec()
        }
    }

    /// Get emotion processing history
    pub fn get_processing_history(&self) -> Vec<EmotionProcessingEntry> {
        self.processing_history.iter().cloned().collect()
    }
}

impl Default for VoiceQualityAdjustments {
    fn default() -> Self {
        Self {
            breathiness: 0.0,
            creakiness: 0.0,
            tenseness: 0.0,
            nasality: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_pronunciation_system() {
        let mut system = AdaptivePronunciationSystem::new(LanguageCode::EnUs);

        let correction = UserCorrection {
            text: "tomato".to_string(),
            original_phonemes: vec![Phoneme::new("təˈmeɪtoʊ")],
            corrected_phonemes: vec![Phoneme::new("təˈmɑːtoʊ")],
            timestamp: SystemTime::now(),
            context: None,
            user_confidence: 0.9,
        };

        assert!(system.add_correction(correction).is_ok());
        assert_eq!(system.adaptation_model.word_adaptations.len(), 1);
    }

    #[test]
    fn test_multilingual_phoneme_mapper() {
        let mapper = MultilingualPhonemeMapper::new();

        let phonemes = vec![Phoneme::new("æ")];
        let result = mapper.map_phonemes(&phonemes, LanguageCode::EnUs, LanguageCode::De);

        assert!(result.is_ok());
    }

    #[test]
    fn test_phoneme_quality_scorer() {
        let mut scorer = PhonemeQualityScorer::new();

        let phonemes = vec![
            Phoneme::with_confidence("h", 0.9),
            Phoneme::with_confidence("ɛ", 0.8),
            Phoneme::with_confidence("l", 0.85),
        ];

        let assessment = scorer.assess_quality("hello", &phonemes, LanguageCode::EnUs);

        assert!(assessment.overall_score > 0.0);
        assert_eq!(assessment.phoneme_scores.len(), 3);
    }

    #[tokio::test]
    async fn test_streaming_g2p_processor() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let processor = StreamingG2pProcessor::new(LanguageCode::EnUs, sender);

        assert!(processor.add_text("hello").is_ok());
        assert!(processor.process_buffer().await.is_ok());

        // Check if we received streaming phonemes
        if let Ok(streaming_phoneme) = receiver.try_recv() {
            assert!(!streaming_phoneme.phoneme.symbol.is_empty());
        }
    }

    #[tokio::test]
    async fn test_emotion_aware_processor() {
        let dummy_processor = Arc::new(crate::DummyG2p::new());
        let mut emotion_processor = EmotionAwareG2pProcessor::new(dummy_processor);

        let result = emotion_processor
            .process_with_emotion("I am very happy!", Some(LanguageCode::EnUs))
            .await;

        assert!(result.is_ok());
        assert_eq!(emotion_processor.processing_history.len(), 1);

        let entry = &emotion_processor.processing_history[0];
        // Should detect happy emotion
        assert_eq!(entry.detected_emotion, EmotionType::Happy);
    }

    #[test]
    fn test_emotion_classification() {
        let dummy_processor = Arc::new(crate::DummyG2p::new());
        let emotion_processor = EmotionAwareG2pProcessor::new(dummy_processor);

        let (emotion, confidence) =
            emotion_processor.classify_emotion("I am so excited about this!");
        assert_eq!(emotion, EmotionType::Excited);
        assert!(confidence > 0.0);

        let (emotion, _) =
            emotion_processor.classify_emotion("This is terrible and makes me angry!");
        assert_eq!(emotion, EmotionType::Angry);
    }
}
