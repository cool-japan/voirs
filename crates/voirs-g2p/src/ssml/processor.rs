//! Advanced SSML processor combining all SSML functionality.

use crate::ssml::accents::{AccentProfile, AccentSystem};
use crate::ssml::context::{ContextAnalysisResult, ContextAnalyzer};
use crate::ssml::dictionary::{DictionaryManager, PronunciationContext};
use crate::ssml::elements::*;
use crate::ssml::simple_parser::SimpleSsmlParser;
use crate::{G2pError, LanguageCode, Phoneme, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Advanced SSML processor with full feature support
pub struct SsmlProcessor {
    /// SSML parser
    parser: SimpleSsmlParser,
    /// Dictionary manager for custom pronunciations
    dictionary_manager: Arc<RwLock<DictionaryManager>>,
    /// Context analyzer
    context_analyzer: Arc<RwLock<ContextAnalyzer>>,
    /// Accent system
    accent_system: Arc<RwLock<AccentSystem>>,
    /// Processor configuration
    config: ProcessorConfig,
    /// Processing statistics
    statistics: ProcessorStatistics,
    /// Custom phoneme overrides
    phoneme_overrides: HashMap<String, Vec<Phoneme>>,
    /// Active processing context
    processing_context: ProcessingContext,
}

/// Processor configuration
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Default language for processing
    pub default_language: LanguageCode,
    /// Enable context-sensitive pronunciation
    pub enable_context_analysis: bool,
    /// Enable regional accent processing
    pub enable_accent_processing: bool,
    /// Enable custom dictionary lookup
    pub enable_dictionary_lookup: bool,
    /// Enable phoneme override processing
    pub enable_phoneme_overrides: bool,
    /// Maximum processing time per element (ms)
    pub max_processing_time_ms: u64,
    /// Enable caching of processing results
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
}

/// Processing statistics
#[derive(Debug, Clone, Default)]
pub struct ProcessorStatistics {
    /// Total elements processed
    pub elements_processed: usize,
    /// Total phonemes generated
    pub phonemes_generated: usize,
    /// Dictionary lookups performed
    pub dictionary_lookups: usize,
    /// Context analyses performed
    pub context_analyses: usize,
    /// Accent transformations applied
    pub accent_transformations: usize,
    /// Processing time breakdown
    pub timing: ProcessingTiming,
    /// Cache statistics
    pub cache_stats: CacheStatistics,
}

/// Processing timing information
#[derive(Debug, Clone, Default)]
pub struct ProcessingTiming {
    /// Total processing time (ms)
    pub total_ms: f64,
    /// Parsing time (ms)
    pub parsing_ms: f64,
    /// Dictionary lookup time (ms)
    pub dictionary_ms: f64,
    /// Context analysis time (ms)
    pub context_ms: f64,
    /// Accent processing time (ms)
    pub accent_ms: f64,
    /// Phoneme generation time (ms)
    pub phoneme_generation_ms: f64,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// Cache hit ratio
    pub hit_ratio: f32,
    /// Cache size
    pub current_size: usize,
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
}

/// Processing context for maintaining state
#[derive(Debug, Clone, Default)]
pub struct ProcessingContext {
    /// Current document language
    pub document_language: Option<LanguageCode>,
    /// Current voice settings
    pub current_voice: Option<VoiceSettings>,
    /// Current prosody settings
    pub current_prosody: Option<ProsodySettings>,
    /// Processing depth
    pub depth: usize,
    /// Current sentence tokens (for context analysis)
    pub current_sentence: Vec<String>,
    /// Current word index in sentence
    pub current_word_index: usize,
}

/// Voice settings from SSML
#[derive(Debug, Clone, Default)]
pub struct VoiceSettings {
    /// Voice name
    pub name: Option<String>,
    /// Voice gender
    pub gender: Option<VoiceGender>,
    /// Voice age
    pub age: Option<String>,
    /// Voice characteristics
    pub characteristics: Option<VoiceCharacteristics>,
}

/// Prosody settings from SSML
#[derive(Debug, Clone, Default)]
pub struct ProsodySettings {
    /// Speaking rate
    pub rate: Option<String>,
    /// Pitch settings
    pub pitch: Option<String>,
    /// Volume settings
    pub volume: Option<String>,
    /// Enhanced prosody parameters
    pub enhanced: Option<EnhancedProsody>,
}

/// Processing result
#[derive(Debug, Clone)]
pub struct SsmlProcessingResult {
    /// Generated phonemes
    pub phonemes: Vec<Phoneme>,
    /// Processing metadata
    pub metadata: ProcessingMetadata,
    /// Applied transformations
    pub transformations: Vec<AppliedTransformation>,
    /// Warnings encountered
    pub warnings: Vec<ProcessingWarning>,
}

/// Processing metadata
#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    /// Original SSML element type
    pub element_type: String,
    /// Language used for processing
    pub language: LanguageCode,
    /// Context information
    pub context: Option<ContextAnalysisResult>,
    /// Dictionary entries used
    pub dictionary_entries: Vec<String>,
    /// Accent profile applied
    pub accent_profile: Option<String>,
    /// Processing time (ms)
    pub processing_time_ms: f64,
}

/// Applied transformation record
#[derive(Debug, Clone)]
pub struct AppliedTransformation {
    /// Transformation type
    pub transformation_type: TransformationType,
    /// Source data
    pub source: String,
    /// Target data
    pub target: String,
    /// Confidence level
    pub confidence: f32,
    /// Applied at position
    pub position: usize,
}

/// Processing warning
#[derive(Debug, Clone)]
pub struct ProcessingWarning {
    /// Warning message
    pub message: String,
    /// Warning type
    pub warning_type: ProcessingWarningType,
    /// Element that caused the warning
    pub element_context: Option<String>,
    /// Suggested action
    pub suggestion: Option<String>,
}

/// Transformation types
#[derive(Debug, Clone)]
pub enum TransformationType {
    /// Dictionary lookup substitution
    DictionarySubstitution,
    /// Context-based modification
    ContextModification,
    /// Accent transformation
    AccentTransformation,
    /// Phoneme override
    PhonemeOverride,
    /// Prosody adjustment
    ProsodyAdjustment,
}

/// Processing warning types
#[derive(Debug, Clone)]
pub enum ProcessingWarningType {
    /// Missing dictionary entry
    MissingDictionaryEntry,
    /// Ambiguous context
    AmbiguousContext,
    /// Accent not available
    AccentNotAvailable,
    /// Invalid phoneme override
    InvalidPhonemeOverride,
    /// Performance issue
    Performance,
}

impl SsmlProcessor {
    /// Create a new SSML processor
    pub fn new() -> Self {
        Self {
            parser: SimpleSsmlParser::new(),
            dictionary_manager: Arc::new(RwLock::new(DictionaryManager::new())),
            context_analyzer: Arc::new(RwLock::new(ContextAnalyzer::new(LanguageCode::EnUs))),
            accent_system: Arc::new(RwLock::new(AccentSystem::new())),
            config: ProcessorConfig::default(),
            statistics: ProcessorStatistics::default(),
            phoneme_overrides: HashMap::new(),
            processing_context: ProcessingContext::default(),
        }
    }

    /// Create processor with custom configuration
    pub fn with_config(config: ProcessorConfig) -> Self {
        let mut processor = Self::new();
        processor.config = config;

        // SimpleSsmlParser uses default configuration
        processor.parser = SimpleSsmlParser::new();

        processor
    }

    /// Process SSML text into phonemes
    pub fn process(&mut self, ssml_text: &str) -> Result<SsmlProcessingResult> {
        let start_time = std::time::Instant::now();

        // Parse SSML
        let parse_start = std::time::Instant::now();
        let element = self.parser.parse(ssml_text)?;
        let parsing_time = parse_start.elapsed().as_millis() as f64;

        // Process the parsed element
        let mut phonemes = Vec::new();
        let mut transformations = Vec::new();
        let mut warnings = Vec::new();
        let mut metadata = ProcessingMetadata {
            element_type: "speak".to_string(),
            language: self.config.default_language,
            context: None,
            dictionary_entries: Vec::new(),
            accent_profile: None,
            processing_time_ms: 0.0,
        };

        self.process_element(
            &element,
            &mut phonemes,
            &mut transformations,
            &mut warnings,
            &mut metadata,
        )?;

        let total_time = start_time.elapsed().as_millis() as f64;
        metadata.processing_time_ms = total_time;

        // Update statistics
        self.update_statistics(parsing_time, total_time, &phonemes, &transformations);

        Ok(SsmlProcessingResult {
            phonemes,
            metadata,
            transformations,
            warnings,
        })
    }

    /// Process a single SSML element
    fn process_element(
        &mut self,
        element: &SsmlElement,
        phonemes: &mut Vec<Phoneme>,
        transformations: &mut Vec<AppliedTransformation>,
        warnings: &mut Vec<ProcessingWarning>,
        metadata: &mut ProcessingMetadata,
    ) -> Result<()> {
        self.processing_context.depth += 1;

        match element {
            SsmlElement::Speak {
                language, content, ..
            } => {
                if let Some(lang) = language {
                    self.processing_context.document_language = Some(*lang);
                    metadata.language = *lang;
                }
                self.process_children(content, phonemes, transformations, warnings, metadata)?;
            }

            SsmlElement::Text(text) => {
                self.process_text(text, phonemes, transformations, warnings, metadata)?;
            }

            SsmlElement::Phoneme {
                ph,
                text,
                metadata: ph_metadata,
                ..
            } => {
                self.process_phoneme_override(ph, text, ph_metadata, phonemes, transformations)?;
            }

            SsmlElement::Lang {
                lang,
                content,
                variant: _,
                accent,
            } => {
                let previous_lang = self.processing_context.document_language;
                self.processing_context.document_language = Some(*lang);
                metadata.language = *lang;

                if let Some(accent_name) = accent {
                    if let Ok(mut accent_system) = self.accent_system.write() {
                        if accent_system.set_active_accent(accent_name).is_ok() {
                            metadata.accent_profile = Some(accent_name.clone());
                        }
                    }
                }

                self.process_children(content, phonemes, transformations, warnings, metadata)?;

                // Restore previous language
                self.processing_context.document_language = previous_lang;
            }

            SsmlElement::Emphasis {
                content,
                level,
                custom_params,
            } => {
                // Process emphasis by modifying stress/prominence
                self.process_children(content, phonemes, transformations, warnings, metadata)?;
                self.apply_emphasis_modification(phonemes, level, custom_params)?;
            }

            SsmlElement::Break {
                time,
                strength,
                custom_timing,
            } => {
                self.process_break(time, strength, custom_timing, phonemes)?;
            }

            SsmlElement::SayAs {
                interpret_as,
                content,
                ..
            } => {
                self.process_say_as(
                    interpret_as,
                    content,
                    phonemes,
                    transformations,
                    warnings,
                    metadata,
                )?;
            }

            SsmlElement::Prosody {
                rate,
                pitch,
                volume,
                content,
                enhanced,
            } => {
                let previous_prosody = self.processing_context.current_prosody.clone();
                self.processing_context.current_prosody = Some(ProsodySettings {
                    rate: rate.clone(),
                    pitch: pitch.clone(),
                    volume: volume.clone(),
                    enhanced: enhanced.clone(),
                });

                self.process_children(content, phonemes, transformations, warnings, metadata)?;

                // Apply prosody modifications
                self.apply_prosody_modifications(phonemes, rate, pitch, volume, enhanced)?;

                // Restore previous prosody
                self.processing_context.current_prosody = previous_prosody;
            }

            SsmlElement::Voice {
                name,
                gender,
                age,
                content,
                characteristics,
            } => {
                let previous_voice = self.processing_context.current_voice.clone();
                self.processing_context.current_voice = Some(VoiceSettings {
                    name: name.clone(),
                    gender: gender.clone(),
                    age: age.clone(),
                    characteristics: characteristics.clone(),
                });

                self.process_children(content, phonemes, transformations, warnings, metadata)?;

                // Restore previous voice
                self.processing_context.current_voice = previous_voice;
            }

            SsmlElement::Mark { name } => {
                // Marks are timing points - create a special phoneme marker
                phonemes.push(self.create_mark_phoneme(name)?);
            }

            SsmlElement::Paragraph { content, prosody } => {
                self.process_children(content, phonemes, transformations, warnings, metadata)?;
                if let Some(para_prosody) = prosody {
                    self.apply_paragraph_prosody(phonemes, para_prosody)?;
                }
            }

            SsmlElement::Sentence { content, prosody } => {
                // Extract sentence tokens for context analysis
                let sentence_text = self.extract_text_from_content(content);
                self.processing_context.current_sentence = sentence_text
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
                self.processing_context.current_word_index = 0;

                self.process_children(content, phonemes, transformations, warnings, metadata)?;

                if let Some(sent_prosody) = prosody {
                    self.apply_sentence_prosody(phonemes, sent_prosody)?;
                }
            }

            SsmlElement::Dictionary {
                ref_name: _,
                scope: _,
            } => {
                // Dictionary references are processed when loading dictionaries
                // This is a placeholder for future implementation
            }
        }

        self.processing_context.depth -= 1;
        Ok(())
    }

    /// Process child elements
    fn process_children(
        &mut self,
        content: &[SsmlElement],
        phonemes: &mut Vec<Phoneme>,
        transformations: &mut Vec<AppliedTransformation>,
        warnings: &mut Vec<ProcessingWarning>,
        metadata: &mut ProcessingMetadata,
    ) -> Result<()> {
        for child in content {
            self.process_element(child, phonemes, transformations, warnings, metadata)?;
        }
        Ok(())
    }

    /// Process text content with full analysis
    fn process_text(
        &mut self,
        text: &str,
        phonemes: &mut Vec<Phoneme>,
        transformations: &mut Vec<AppliedTransformation>,
        _warnings: &mut [ProcessingWarning],
        metadata: &mut ProcessingMetadata,
    ) -> Result<()> {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        for (word_index, word) in words.iter().enumerate() {
            self.processing_context.current_word_index = word_index;

            // Try dictionary lookup first
            let mut word_phonemes = if self.config.enable_dictionary_lookup {
                self.lookup_in_dictionary(word, metadata)?
            } else {
                None
            };

            // Try context analysis if enabled and dictionary didn't provide result
            if word_phonemes.is_none() && self.config.enable_context_analysis {
                word_phonemes = self.analyze_with_context(word, &words, word_index, metadata)?;
            }

            // If still no result, generate basic phonemes (this would call the G2P backend)
            if word_phonemes.is_none() {
                word_phonemes = Some(self.generate_basic_phonemes(word)?);
            }

            if let Some(mut word_phon) = word_phonemes {
                // Apply accent transformations if enabled
                if self.config.enable_accent_processing {
                    word_phon = self.apply_accent_transformations(&word_phon, transformations)?;
                }

                phonemes.extend(word_phon);
            }
        }

        Ok(())
    }

    /// Lookup word in custom dictionaries
    fn lookup_in_dictionary(
        &mut self,
        word: &str,
        metadata: &mut ProcessingMetadata,
    ) -> Result<Option<Vec<Phoneme>>> {
        if let Ok(mut dict_manager) = self.dictionary_manager.write() {
            let context = self.determine_pronunciation_context(word)?;
            let result = dict_manager.lookup(word, context.as_ref());

            if result.is_some() {
                metadata.dictionary_entries.push(word.to_string());
                self.statistics.dictionary_lookups += 1;
            }

            Ok(result)
        } else {
            Ok(None)
        }
    }

    /// Analyze word with context
    fn analyze_with_context(
        &mut self,
        word: &str,
        sentence: &[String],
        word_index: usize,
        metadata: &mut ProcessingMetadata,
    ) -> Result<Option<Vec<Phoneme>>> {
        if let Ok(mut analyzer) = self.context_analyzer.write() {
            let analysis = analyzer.analyze_context(word, sentence, word_index)?;
            metadata.context = Some(analysis);
            self.statistics.context_analyses += 1;

            // Use context to inform pronunciation (simplified)
            // In a full implementation, this would use the context to select
            // appropriate pronunciation variants
            Ok(None) // For now, let basic generation handle it
        } else {
            Ok(None)
        }
    }

    /// Apply accent transformations
    fn apply_accent_transformations(
        &mut self,
        phonemes: &[Phoneme],
        transformations: &mut Vec<AppliedTransformation>,
    ) -> Result<Vec<Phoneme>> {
        if let Ok(mut accent_system) = self.accent_system.write() {
            let result = accent_system.apply_accent(phonemes, None)?;

            // Record transformations
            for (i, (original, modified)) in phonemes.iter().zip(result.iter()).enumerate() {
                if original.symbol != modified.symbol {
                    transformations.push(AppliedTransformation {
                        transformation_type: TransformationType::AccentTransformation,
                        source: original.symbol.clone(),
                        target: modified.symbol.clone(),
                        confidence: modified.confidence,
                        position: i,
                    });
                    self.statistics.accent_transformations += 1;
                }
            }

            Ok(result)
        } else {
            Ok(phonemes.to_vec())
        }
    }

    /// Generate basic phonemes (placeholder - would call actual G2P backend)
    fn generate_basic_phonemes(&mut self, word: &str) -> Result<Vec<Phoneme>> {
        // This would integrate with the actual G2P backends
        // For now, create a placeholder phoneme
        Ok(vec![Phoneme {
            symbol: format!("/{word}/"), // Placeholder
            ipa_symbol: Some(format!("/{word}/")),
            language_notation: None,
            stress: 0,
            syllable_position: crate::SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 0.5, // Low confidence for placeholder
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: true,
            is_syllable_boundary: false,
        }])
    }

    /// Process phoneme override
    fn process_phoneme_override(
        &mut self,
        ph: &str,
        text: &str,
        _metadata: &Option<PhonemeMetadata>,
        phonemes: &mut Vec<Phoneme>,
        transformations: &mut Vec<AppliedTransformation>,
    ) -> Result<()> {
        let override_phonemes = self.parse_phoneme_string(ph)?;

        for (i, phoneme) in override_phonemes.iter().enumerate() {
            transformations.push(AppliedTransformation {
                transformation_type: TransformationType::PhonemeOverride,
                source: text.to_string(),
                target: phoneme.symbol.clone(),
                confidence: 1.0, // Overrides have maximum confidence
                position: phonemes.len() + i,
            });
        }

        phonemes.extend(override_phonemes);
        Ok(())
    }

    /// Parse phoneme string from SSML
    fn parse_phoneme_string(&self, ph: &str) -> Result<Vec<Phoneme>> {
        let symbols: Vec<&str> = ph.split_whitespace().collect();
        let mut phonemes = Vec::new();

        for symbol in symbols {
            phonemes.push(Phoneme {
                symbol: symbol.to_string(),
                ipa_symbol: Some(symbol.to_string()),
                language_notation: Some("SSML-Override".to_string()),
                stress: 0,
                syllable_position: crate::SyllablePosition::Standalone,
                duration_ms: None,
                confidence: 1.0,
                phonetic_features: None,
                custom_features: None,
                is_word_boundary: false,
                is_syllable_boundary: false,
            });
        }

        Ok(phonemes)
    }

    /// Helper methods (simplified implementations)
    fn determine_pronunciation_context(&self, _word: &str) -> Result<Option<PronunciationContext>> {
        // Simplified context determination
        Ok(Some(PronunciationContext::Stressed)) // Placeholder
    }

    fn apply_emphasis_modification(
        &mut self,
        phonemes: &mut [Phoneme],
        level: &EmphasisLevel,
        _custom_params: &Option<EmphasisParams>,
    ) -> Result<()> {
        // Apply emphasis by modifying stress and confidence
        for phoneme in phonemes.iter_mut() {
            match level {
                EmphasisLevel::Strong => {
                    phoneme.stress = phoneme.stress.max(1);
                    phoneme.confidence = (phoneme.confidence * 1.2).min(1.0);
                }
                EmphasisLevel::Moderate => {
                    phoneme.confidence = (phoneme.confidence * 1.1).min(1.0);
                }
                EmphasisLevel::Reduced => {
                    phoneme.confidence *= 0.9;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn process_break(
        &mut self,
        time: &Option<String>,
        _strength: &Option<BreakStrength>,
        _custom_timing: &Option<BreakTiming>,
        phonemes: &mut Vec<Phoneme>,
    ) -> Result<()> {
        // Create a pause phoneme
        let duration = if let Some(time_str) = time {
            self.parse_duration(time_str)?
        } else {
            100.0 // Default 100ms
        };

        phonemes.push(Phoneme {
            symbol: "PAUSE".to_string(),
            ipa_symbol: None,
            language_notation: Some("SSML-Break".to_string()),
            stress: 0,
            syllable_position: crate::SyllablePosition::Standalone,
            duration_ms: Some(duration),
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: true,
            is_syllable_boundary: true,
        });

        Ok(())
    }

    fn parse_duration(&self, time_str: &str) -> Result<f32> {
        // Parse time string like "500ms", "1s", etc.
        if time_str.ends_with("ms") {
            time_str
                .trim_end_matches("ms")
                .parse::<f32>()
                .map_err(|_| G2pError::ConfigError("Invalid duration format".to_string()))
        } else if time_str.ends_with("s") {
            time_str
                .trim_end_matches("s")
                .parse::<f32>()
                .map(|s| s * 1000.0)
                .map_err(|_| G2pError::ConfigError("Invalid duration format".to_string()))
        } else {
            time_str
                .parse::<f32>()
                .map_err(|_| G2pError::ConfigError("Invalid duration format".to_string()))
        }
    }

    #[allow(clippy::ptr_arg)]
    fn process_say_as(
        &mut self,
        interpret_as: &InterpretAs,
        content: &str,
        phonemes: &mut Vec<Phoneme>,
        transformations: &mut Vec<AppliedTransformation>,
        warnings: &mut Vec<ProcessingWarning>,
        metadata: &mut ProcessingMetadata,
    ) -> Result<()> {
        // Process content according to interpretation type
        let processed_text = match interpret_as {
            InterpretAs::Characters => {
                // Spell out each character
                content.chars().map(|c| format!("{c} ")).collect::<String>()
            }
            InterpretAs::Digits => {
                // Spell out each digit
                content
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .map(|c| format!("{c} "))
                    .collect::<String>()
            }
            InterpretAs::Cardinal => {
                // Convert to cardinal number (would need number-to-words)
                content.to_string() // Simplified
            }
            _ => content.to_string(), // Other types would need specific processing
        };

        // Process the interpreted text
        self.process_text(
            &processed_text,
            phonemes,
            transformations,
            warnings,
            metadata,
        )?;
        Ok(())
    }

    fn apply_prosody_modifications(
        &mut self,
        phonemes: &mut [Phoneme],
        rate: &Option<String>,
        _pitch: &Option<String>,
        _volume: &Option<String>,
        _enhanced: &Option<EnhancedProsody>,
    ) -> Result<()> {
        // Apply prosody modifications to phonemes
        if let Some(rate_str) = rate {
            let rate_factor = self.parse_rate_factor(rate_str)?;
            for phoneme in phonemes.iter_mut() {
                if let Some(duration) = phoneme.duration_ms {
                    phoneme.duration_ms = Some(duration / rate_factor);
                }
            }
        }

        // Pitch and volume would be handled by the acoustic model
        // We can add metadata here for downstream processing

        Ok(())
    }

    fn parse_rate_factor(&self, rate_str: &str) -> Result<f32> {
        match rate_str {
            "x-slow" => Ok(0.5),
            "slow" => Ok(0.75),
            "medium" => Ok(1.0),
            "fast" => Ok(1.25),
            "x-fast" => Ok(1.5),
            _ => {
                // Try to parse as percentage or factor
                if rate_str.ends_with('%') {
                    rate_str
                        .trim_end_matches('%')
                        .parse::<f32>()
                        .map(|p| p / 100.0)
                        .map_err(|_| G2pError::ConfigError("Invalid rate format".to_string()))
                } else {
                    rate_str
                        .parse::<f32>()
                        .map_err(|_| G2pError::ConfigError("Invalid rate format".to_string()))
                }
            }
        }
    }

    fn create_mark_phoneme(&self, name: &str) -> Result<Phoneme> {
        Ok(Phoneme {
            symbol: format!("MARK:{name}"),
            ipa_symbol: None,
            language_notation: Some("SSML-Mark".to_string()),
            stress: 0,
            syllable_position: crate::SyllablePosition::Standalone,
            duration_ms: Some(0.0), // Zero duration
            confidence: 1.0,
            phonetic_features: None,
            custom_features: Some({
                let mut features = HashMap::new();
                features.insert("mark_name".to_string(), name.to_string());
                features
            }),
            is_word_boundary: true,
            is_syllable_boundary: false,
        })
    }

    fn extract_text_from_content(&self, content: &[SsmlElement]) -> String {
        let mut text = String::new();
        for element in content {
            match element {
                SsmlElement::Text(t) => text.push_str(t),
                SsmlElement::Phoneme { text: t, .. } => text.push_str(t),
                _ => {
                    // Recursively extract text from other elements
                    // Simplified implementation
                }
            }
        }
        text
    }

    fn apply_paragraph_prosody(
        &mut self,
        _phonemes: &mut [Phoneme],
        _prosody: &ParagraphProsody,
    ) -> Result<()> {
        // Apply paragraph-level prosody modifications
        Ok(())
    }

    fn apply_sentence_prosody(
        &mut self,
        _phonemes: &mut [Phoneme],
        _prosody: &SentenceProsody,
    ) -> Result<()> {
        // Apply sentence-level prosody modifications
        Ok(())
    }

    fn update_statistics(
        &mut self,
        parsing_time: f64,
        total_time: f64,
        phonemes: &[Phoneme],
        transformations: &[AppliedTransformation],
    ) {
        self.statistics.elements_processed += 1;
        self.statistics.phonemes_generated += phonemes.len();
        self.statistics.timing.total_ms += total_time;
        self.statistics.timing.parsing_ms += parsing_time;

        // Update other timing fields based on transformations
        for transformation in transformations {
            match transformation.transformation_type {
                TransformationType::DictionarySubstitution => {
                    self.statistics.timing.dictionary_ms += 1.0; // Simplified
                }
                TransformationType::ContextModification => {
                    self.statistics.timing.context_ms += 1.0; // Simplified
                }
                TransformationType::AccentTransformation => {
                    self.statistics.timing.accent_ms += 1.0; // Simplified
                }
                _ => {}
            }
        }
    }

    /// Public API methods
    /// Add custom phoneme override
    pub fn add_phoneme_override(&mut self, text: String, phonemes: Vec<Phoneme>) {
        self.phoneme_overrides.insert(text, phonemes);
    }

    /// Set active accent
    pub fn set_active_accent(&mut self, accent_name: &str) -> Result<()> {
        if let Ok(mut accent_system) = self.accent_system.write() {
            accent_system.set_active_accent(accent_name)
        } else {
            Err(G2pError::ConfigError(
                "Failed to access accent system".to_string(),
            ))
        }
    }

    /// Load custom accent profile
    pub fn load_accent_profile(&mut self, accent: AccentProfile) -> Result<()> {
        if let Ok(mut accent_system) = self.accent_system.write() {
            accent_system.load_accent(accent);
            Ok(())
        } else {
            Err(G2pError::ConfigError(
                "Failed to access accent system".to_string(),
            ))
        }
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> &ProcessorStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = ProcessorStatistics::default();
    }

    /// Convert SSML element to plain text
    #[allow(clippy::only_used_in_recursion)]
    pub fn to_text(&self, element: &SsmlElement) -> String {
        match element {
            SsmlElement::Speak { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Text(text) => text.clone(),
            SsmlElement::Phoneme { text, .. } => text.clone(),
            SsmlElement::Lang { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Emphasis { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Break { .. } => " ".to_string(),
            SsmlElement::SayAs { content, .. } => content.clone(),
            SsmlElement::Prosody { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Voice { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Mark { .. } => "".to_string(),
            SsmlElement::Paragraph { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Sentence { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Dictionary { .. } => "".to_string(),
        }
    }
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            default_language: LanguageCode::EnUs,
            enable_context_analysis: true,
            enable_accent_processing: true,
            enable_dictionary_lookup: true,
            enable_phoneme_overrides: true,
            max_processing_time_ms: 5000,
            enable_caching: true,
            cache_size_limit: 1000,
        }
    }
}

impl Default for SsmlProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = SsmlProcessor::new();
        assert_eq!(processor.config.default_language, LanguageCode::EnUs);
    }

    #[test]
    fn test_simple_processing() {
        let mut processor = SsmlProcessor::new();
        let ssml = "<speak>Hello world</speak>";
        let result = processor.process(ssml);
        assert!(result.is_ok());

        let processing_result = result.unwrap();
        assert!(!processing_result.phonemes.is_empty());
    }

    #[test]
    fn test_phoneme_override() {
        let mut processor = SsmlProcessor::new();
        let ssml = r#"<speak><phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme></speak>"#;
        let result = processor.process(ssml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_emphasis_processing() {
        let mut processor = SsmlProcessor::new();
        let ssml = r#"<speak><emphasis level="strong">important</emphasis></speak>"#;
        let result = processor.process(ssml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_statistics_tracking() {
        let mut processor = SsmlProcessor::new();
        let ssml = "<speak>Test text</speak>";
        processor.process(ssml).unwrap();

        let stats = processor.get_statistics();
        assert!(stats.elements_processed > 0);
        assert!(stats.phonemes_generated > 0);
    }
}
