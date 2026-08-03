//! Advanced SSML processor combining all SSML functionality.

use crate::backends::RuleBasedG2p;
use crate::ssml::accents::{AccentProfile, AccentSystem};
use crate::ssml::context::{ContextAnalysisResult, ContextAnalyzer};
use crate::ssml::dictionary::DictionaryManager;
use crate::ssml::elements::*;
use crate::ssml::simple_parser::SimpleSsmlParser;
use crate::{G2p, G2pConverter, G2pError, LanguageCode, Phoneme, Result};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
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
    /// Real G2P backend used to phonemize words that have no dictionary entry
    /// or context-specific override (see [`SsmlProcessor::with_g2p_backend`]).
    g2p_backend: Arc<dyn G2p>,
}

/// Build the default G2P backend used by [`SsmlProcessor::new`].
///
/// This wires up a [`G2pConverter`] with a dedicated [`RuleBasedG2p`] instance
/// per supported [`LanguageCode`], so that the `language` hint passed to
/// [`G2p::to_phonemes`] actually selects language-appropriate phonological
/// rules instead of always applying a single fixed language's rules.
fn build_default_g2p_backend(default_language: LanguageCode) -> Arc<dyn G2p> {
    let mut converter = G2pConverter::new();
    for language in [
        LanguageCode::EnUs,
        LanguageCode::EnGb,
        LanguageCode::De,
        LanguageCode::Fr,
        LanguageCode::Es,
        LanguageCode::It,
        LanguageCode::Pt,
        LanguageCode::Ja,
        LanguageCode::ZhCn,
        LanguageCode::Ko,
        LanguageCode::Ru,
        LanguageCode::Ar,
    ] {
        converter.add_backend(language, Box::new(RuleBasedG2p::new(language)));
    }
    converter.set_default_backend(Box::new(RuleBasedG2p::new(default_language)));
    Arc::new(converter)
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
        let config = ProcessorConfig::default();
        let g2p_backend = build_default_g2p_backend(config.default_language);
        Self {
            parser: SimpleSsmlParser::new(),
            dictionary_manager: Arc::new(RwLock::new(DictionaryManager::new())),
            context_analyzer: Arc::new(RwLock::new(ContextAnalyzer::new(LanguageCode::EnUs))),
            accent_system: Arc::new(RwLock::new(AccentSystem::new())),
            config,
            statistics: ProcessorStatistics::default(),
            phoneme_overrides: HashMap::new(),
            processing_context: ProcessingContext::default(),
            g2p_backend,
        }
    }

    /// Create processor with custom configuration
    pub fn with_config(config: ProcessorConfig) -> Self {
        let mut processor = Self::new();
        processor.g2p_backend = build_default_g2p_backend(config.default_language);
        processor.config = config;

        // SimpleSsmlParser uses default configuration
        processor.parser = SimpleSsmlParser::new();

        processor
    }

    /// Inject a custom G2P backend (e.g. a neural or dictionary-based
    /// converter) to use for words that have no dictionary entry or
    /// context-specific override. By default, [`SsmlProcessor::new`] wires up
    /// per-language [`RuleBasedG2p`] instances via a [`G2pConverter`].
    pub fn with_g2p_backend(mut self, backend: Arc<dyn G2p>) -> Self {
        self.g2p_backend = backend;
        self
    }

    /// Process SSML text into phonemes
    pub async fn process(&mut self, ssml_text: &str) -> Result<SsmlProcessingResult> {
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
        )
        .await?;

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

    /// Process a single SSML element.
    ///
    /// This recurses into [`Self::process_children`], which recurses back into
    /// `process_element` for nested content, so (unlike an ordinary `async fn`)
    /// it must return an explicitly boxed future -- Rust cannot compute a
    /// finite size for a directly self-recursive `async fn`'s generated state
    /// machine.
    fn process_element<'a>(
        &'a mut self,
        element: &'a SsmlElement,
        phonemes: &'a mut Vec<Phoneme>,
        transformations: &'a mut Vec<AppliedTransformation>,
        warnings: &'a mut Vec<ProcessingWarning>,
        metadata: &'a mut ProcessingMetadata,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
        Box::pin(async move {
            self.processing_context.depth += 1;

            match element {
                SsmlElement::Speak {
                    language, content, ..
                } => {
                    if let Some(lang) = language {
                        self.processing_context.document_language = Some(*lang);
                        metadata.language = *lang;
                    }
                    self.process_children(content, phonemes, transformations, warnings, metadata)
                        .await?;
                }

                SsmlElement::Text(text) => {
                    self.process_text(text, phonemes, transformations, warnings, metadata)
                        .await?;
                }

                SsmlElement::Phoneme {
                    ph,
                    text,
                    metadata: ph_metadata,
                    ..
                } => {
                    self.process_phoneme_override(
                        ph,
                        text,
                        ph_metadata,
                        phonemes,
                        transformations,
                    )?;
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

                    self.process_children(content, phonemes, transformations, warnings, metadata)
                        .await?;

                    // Restore previous language
                    self.processing_context.document_language = previous_lang;
                }

                SsmlElement::Emphasis {
                    content,
                    level,
                    custom_params,
                } => {
                    // Process emphasis by modifying stress/prominence
                    self.process_children(content, phonemes, transformations, warnings, metadata)
                        .await?;
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
                    )
                    .await?;
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

                    self.process_children(content, phonemes, transformations, warnings, metadata)
                        .await?;

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

                    self.process_children(content, phonemes, transformations, warnings, metadata)
                        .await?;

                    // Restore previous voice
                    self.processing_context.current_voice = previous_voice;
                }

                SsmlElement::Mark { name } => {
                    // Marks are timing points - create a special phoneme marker
                    phonemes.push(self.create_mark_phoneme(name)?);
                }

                SsmlElement::Paragraph { content, prosody } => {
                    self.process_children(content, phonemes, transformations, warnings, metadata)
                        .await?;
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

                    self.process_children(content, phonemes, transformations, warnings, metadata)
                        .await?;

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
        })
    }

    /// Process child elements
    async fn process_children(
        &mut self,
        content: &[SsmlElement],
        phonemes: &mut Vec<Phoneme>,
        transformations: &mut Vec<AppliedTransformation>,
        warnings: &mut Vec<ProcessingWarning>,
        metadata: &mut ProcessingMetadata,
    ) -> Result<()> {
        for child in content {
            self.process_element(child, phonemes, transformations, warnings, metadata)
                .await?;
        }
        Ok(())
    }

    /// Process text content with full analysis
    async fn process_text(
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

            // If still no result, fall back to the real G2P backend.
            if word_phonemes.is_none() {
                word_phonemes = Some(self.generate_basic_phonemes(word).await?);
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

    /// Lookup word in custom dictionaries.
    ///
    /// This first pass has no pronunciation context to work with yet (context
    /// is only known after [`Self::analyze_with_context`] runs), so it performs
    /// a plain, context-free lookup. [`DictionaryManager::lookup`] with `None`
    /// still finds a word's primary pronunciation; a context-aware re-lookup
    /// happens later once real context has actually been analyzed.
    fn lookup_in_dictionary(
        &mut self,
        word: &str,
        metadata: &mut ProcessingMetadata,
    ) -> Result<Option<Vec<Phoneme>>> {
        if let Ok(mut dict_manager) = self.dictionary_manager.write() {
            let result = dict_manager.lookup(word, None);

            if result.is_some() {
                metadata.dictionary_entries.push(word.to_string());
                self.statistics.dictionary_lookups += 1;
            }

            Ok(result)
        } else {
            Ok(None)
        }
    }

    /// Analyze word with context and, if a specific pronunciation context is
    /// identified, use it for a context-aware dictionary re-lookup.
    ///
    /// [`ContextAnalyzer::analyze_context`] performs real analysis and reports
    /// a `primary_context` (e.g. stressed vs. unstressed); this now actually
    /// acts on that result via [`DictionaryManager::lookup`] instead of
    /// discarding it and always deferring to G2P. When no context-specific
    /// entry exists, this honestly returns `None` so the real G2P backend
    /// handles the word.
    fn analyze_with_context(
        &mut self,
        word: &str,
        sentence: &[String],
        word_index: usize,
        metadata: &mut ProcessingMetadata,
    ) -> Result<Option<Vec<Phoneme>>> {
        let Ok(mut analyzer) = self.context_analyzer.write() else {
            return Ok(None);
        };

        let analysis = analyzer.analyze_context(word, sentence, word_index)?;
        let primary_context = analysis.primary_context.clone();
        metadata.context = Some(analysis);
        self.statistics.context_analyses += 1;
        drop(analyzer);

        let Some(context) = primary_context else {
            return Ok(None);
        };

        let Ok(mut dict_manager) = self.dictionary_manager.write() else {
            return Ok(None);
        };

        let result = dict_manager.lookup(word, Some(&context));
        if result.is_some() {
            metadata.dictionary_entries.push(word.to_string());
            self.statistics.dictionary_lookups += 1;
        }

        Ok(result)
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

    /// Generate phonemes for a word that has no dictionary entry or
    /// context-specific override, by calling the processor's real G2P
    /// backend (see [`SsmlProcessor::with_g2p_backend`]).
    ///
    /// The active language is whatever the SSML document currently has in
    /// scope (set by `<speak xml:lang>`/`<lang>`), falling back to
    /// [`ProcessorConfig::default_language`].
    async fn generate_basic_phonemes(&mut self, word: &str) -> Result<Vec<Phoneme>> {
        let language = self
            .processing_context
            .document_language
            .unwrap_or(self.config.default_language);

        self.g2p_backend.to_phonemes(word, Some(language)).await
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
    async fn process_say_as(
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
        )
        .await?;
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

    #[tokio::test]
    async fn test_simple_processing() {
        let mut processor = SsmlProcessor::new();
        let ssml = "<speak>Hello world</speak>";
        let result = processor.process(ssml).await;
        assert!(result.is_ok());

        let processing_result = result.unwrap();
        assert!(!processing_result.phonemes.is_empty());
    }

    #[tokio::test]
    async fn test_phoneme_override() {
        let mut processor = SsmlProcessor::new();
        let ssml = r#"<speak><phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme></speak>"#;
        let result = processor.process(ssml).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_emphasis_processing() {
        let mut processor = SsmlProcessor::new();
        let ssml = r#"<speak><emphasis level="strong">important</emphasis></speak>"#;
        let result = processor.process(ssml).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let mut processor = SsmlProcessor::new();
        let ssml = "<speak>Test text</speak>";
        processor.process(ssml).await.unwrap();

        let stats = processor.get_statistics();
        assert!(stats.elements_processed > 0);
        assert!(stats.phonemes_generated > 0);
    }

    /// Regression test: the OOV fallback must call a real G2P backend and
    /// produce an actual phonemic transcription, not a placeholder literally
    /// equal to `/word/`.
    #[tokio::test]
    async fn test_oov_fallback_uses_real_g2p_not_placeholder() {
        let mut processor = SsmlProcessor::new();
        let ssml = "<speak>zzqvyx</speak>"; // not in any dictionary
        let result = processor.process(ssml).await.unwrap();

        assert!(!result.phonemes.is_empty());
        for phoneme in &result.phonemes {
            assert_ne!(
                phoneme.symbol, "/zzqvyx/",
                "fallback phoneme must not be the fabricated /word/ placeholder"
            );
        }
        // Real rule-based G2P emits one phoneme per matched grapheme, so a
        // 6-letter word should produce more than a single synthesized token.
        assert!(
            result.phonemes.len() > 1,
            "expected a real multi-phoneme transcription, got {:?}",
            result.phonemes
        );
    }

    /// Regression test: different words must produce different phoneme
    /// sequences, proving output genuinely depends on the input word instead
    /// of being a canned/constant result.
    #[tokio::test]
    async fn test_g2p_fallback_output_varies_with_input() {
        let mut processor = SsmlProcessor::new();

        let result_a = processor
            .process("<speak>cat</speak>")
            .await
            .unwrap()
            .phonemes;
        let result_b = processor
            .process("<speak>banana</speak>")
            .await
            .unwrap()
            .phonemes;

        let symbols_a: Vec<&str> = result_a.iter().map(|p| p.symbol.as_str()).collect();
        let symbols_b: Vec<&str> = result_b.iter().map(|p| p.symbol.as_str()).collect();
        assert_ne!(
            symbols_a, symbols_b,
            "phoneme output must vary with input word"
        );
    }

    /// Regression test: a custom G2P backend injected via
    /// `with_g2p_backend` must actually be used instead of the default.
    #[tokio::test]
    async fn test_with_g2p_backend_overrides_default() {
        use crate::{G2pMetadata, Phoneme as PhonemeType};
        use async_trait::async_trait;

        struct StubG2p;

        #[async_trait]
        impl G2p for StubG2p {
            async fn to_phonemes(
                &self,
                _text: &str,
                _lang: Option<LanguageCode>,
            ) -> Result<Vec<PhonemeType>> {
                Ok(vec![PhonemeType {
                    symbol: "STUB".to_string(),
                    ipa_symbol: Some("STUB".to_string()),
                    language_notation: None,
                    stress: 0,
                    syllable_position: crate::SyllablePosition::Standalone,
                    duration_ms: None,
                    confidence: 1.0,
                    phonetic_features: None,
                    custom_features: None,
                    is_word_boundary: true,
                    is_syllable_boundary: false,
                }])
            }

            fn supported_languages(&self) -> Vec<LanguageCode> {
                vec![LanguageCode::EnUs]
            }

            fn metadata(&self) -> G2pMetadata {
                G2pMetadata {
                    name: "stub".to_string(),
                    version: "0.0.0".to_string(),
                    description: "test stub".to_string(),
                    supported_languages: vec![LanguageCode::EnUs],
                    accuracy_scores: HashMap::new(),
                }
            }
        }

        let mut processor = SsmlProcessor::new().with_g2p_backend(Arc::new(StubG2p));
        let result = processor.process("<speak>anything</speak>").await.unwrap();

        assert!(result.phonemes.iter().any(|p| p.symbol == "STUB"));
    }
}
