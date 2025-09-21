//! Model manager for loading and managing pre-trained neural TTS models
//!
//! This module provides utilities for loading, caching, and managing
//! pre-trained VITS and vocoder models from various sources including
//! HuggingFace Hub, local files, and remote URLs.

use crate::backends::{create_default_loader, AcousticModelLoader};
use crate::config::{ModelArchitecture, ModelConfig};
use crate::{AcousticError, AcousticModel, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Model registry for managing pre-trained models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    /// Available models indexed by ID
    pub models: HashMap<String, ModelConfig>,
    /// Model pipelines (acoustic model + vocoder combinations)
    pub pipelines: HashMap<String, PipelineConfig>,
    /// Registry metadata
    pub metadata: RegistryMetadata,
}

/// Pipeline configuration combining acoustic model and vocoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name
    pub name: String,
    /// Pipeline description
    pub description: String,
    /// Acoustic model ID
    pub acoustic_model: String,
    /// Vocoder ID
    pub vocoder: String,
    /// Additional pipeline settings
    pub settings: PipelineSettings,
}

/// Pipeline-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSettings {
    /// Target real-time factor
    pub real_time_factor: f32,
    /// Quality level (0.0-1.0)
    pub quality_level: f32,
    /// Memory usage level (0.0-1.0)
    pub memory_usage: f32,
    /// Whether pipeline supports streaming
    pub supports_streaming: bool,
    /// G2P (Grapheme-to-Phoneme) configuration
    pub g2p_config: G2pConfig,
}

/// Grapheme-to-Phoneme conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pConfig {
    /// G2P engine type
    pub engine: crate::config::G2pEngine,
    /// Language-specific phoneme sets
    pub phoneme_sets: HashMap<LanguageCode, PhonemeSet>,
    /// Pronunciation dictionary paths
    pub dictionaries: HashMap<LanguageCode, String>,
    /// Stress pattern settings
    pub stress_config: StressConfig,
    /// Unknown word handling strategy
    pub unknown_word_strategy: UnknownWordStrategy,
    /// Pronunciation variant preferences
    pub variant_preferences: VariantPreferences,
}

/// Phoneme set definition for a language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeSet {
    /// Phoneme symbols used by the language
    pub symbols: Vec<String>,
    /// Vowel phonemes subset
    pub vowels: Vec<String>,
    /// Consonant phonemes subset
    pub consonants: Vec<String>,
    /// Special symbols (silence, word boundaries, etc.)
    pub special_symbols: Vec<String>,
    /// Stress markers
    pub stress_markers: Vec<String>,
}

/// Stress pattern configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressConfig {
    /// Enable automatic stress prediction
    pub predict_stress: bool,
    /// Primary stress marker
    pub primary_stress_marker: String,
    /// Secondary stress marker
    pub secondary_stress_marker: String,
    /// Stress prediction confidence threshold
    pub confidence_threshold: f32,
}

/// Strategy for handling unknown words
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnknownWordStrategy {
    /// Use fallback pronunciation rules
    FallbackRules,
    /// Generate letter-by-letter phonemes
    LetterByLetter,
    /// Skip unknown words
    Skip,
    /// Use similar word pronunciation
    SimilarWord,
    /// Return error for unknown words
    Error,
}

/// Pronunciation variant preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantPreferences {
    /// Preferred accent/dialect
    pub accent: String,
    /// Formality level preference
    pub formality: FormalityLevel,
    /// Speed optimization preferences
    pub speed_optimized: bool,
    /// Custom pronunciation overrides
    pub custom_pronunciations: HashMap<String, String>,
}

/// Formality level for pronunciation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormalityLevel {
    /// Casual/colloquial pronunciation
    Casual,
    /// Standard/neutral pronunciation
    Standard,
    /// Formal/careful pronunciation
    Formal,
}

/// Registry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    /// Registry version
    pub version: String,
    /// Last updated timestamp
    pub last_updated: String,
    /// Registry description
    pub description: String,
}

/// Pronunciation dictionary entry with multiple pronunciations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationEntry {
    /// Word or token
    pub word: String,
    /// Primary pronunciation
    pub primary: String,
    /// Alternative pronunciations
    pub alternatives: Vec<String>,
    /// Pronunciation variants by accent/formality
    pub variants: HashMap<String, String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Part of speech tag
    pub pos_tag: Option<String>,
}

/// In-memory pronunciation dictionary for fast lookup
#[derive(Debug, Clone)]
pub struct PronunciationDictionary {
    /// Main dictionary entries
    pub entries: HashMap<String, PronunciationEntry>,
    /// Case-insensitive lookup map
    pub case_insensitive_map: HashMap<String, String>,
    /// Language this dictionary is for
    pub language: LanguageCode,
    /// Dictionary metadata
    pub metadata: DictionaryMetadata,
}

/// Dictionary metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryMetadata {
    /// Dictionary name
    pub name: String,
    /// Version
    pub version: String,
    /// Number of entries
    pub entry_count: usize,
    /// Source information
    pub source: String,
    /// License information
    pub license: String,
}

/// Neural TTS model manager
pub struct ModelManager {
    /// Model registry
    registry: ModelRegistry,
    /// Loaded models cache
    loaded_models: Arc<RwLock<HashMap<String, Arc<dyn AcousticModel>>>>,
    /// Model loader
    loader: AcousticModelLoader,
    /// Cache directory
    cache_dir: PathBuf,
    /// Pronunciation dictionaries cache
    pronunciation_dictionaries: Arc<RwLock<HashMap<LanguageCode, PronunciationDictionary>>>,
}

impl ModelManager {
    /// Create new model manager
    pub async fn new() -> Result<Self> {
        let loader = create_default_loader()?;
        let cache_dir = std::env::temp_dir().join("voirs_models");

        // Create cache directory
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| AcousticError::ConfigError(format!("Failed to create cache dir: {e}")))?;

        let registry = ModelRegistry::default();

        Ok(Self {
            registry,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            loader,
            cache_dir,
            pronunciation_dictionaries: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Load model registry from TOML configuration file
    pub async fn load_from_config<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let config_content = std::fs::read_to_string(config_path.as_ref())
            .map_err(|e| AcousticError::ConfigError(format!("Failed to read config: {e}")))?;

        let config: toml::Value = toml::from_str(&config_content)
            .map_err(|e| AcousticError::ConfigError(format!("Failed to parse TOML: {e}")))?;

        let mut manager = Self::new().await?;

        // Parse models section
        if let Some(models_table) = config.get("models").and_then(|v| v.as_table()) {
            for (model_id, model_config) in models_table {
                match ModelConfig::try_from(model_config.clone()) {
                    Ok(config) => {
                        manager.registry.models.insert(model_id.clone(), config);
                        info!("Loaded model configuration: {}", model_id);
                    }
                    Err(e) => {
                        warn!("Failed to parse model config for {}: {}", model_id, e);
                    }
                }
            }
        }

        // Parse pipelines section
        if let Some(pipelines_table) = config.get("pipelines").and_then(|v| v.as_table()) {
            for (pipeline_id, pipeline_config) in pipelines_table {
                match PipelineConfig::try_from(pipeline_config.clone()) {
                    Ok(config) => {
                        manager
                            .registry
                            .pipelines
                            .insert(pipeline_id.clone(), config);
                        info!("Loaded pipeline configuration: {}", pipeline_id);
                    }
                    Err(e) => {
                        warn!("Failed to parse pipeline config for {}: {}", pipeline_id, e);
                    }
                }
            }
        }

        info!(
            "Loaded {} models and {} pipelines from configuration",
            manager.registry.models.len(),
            manager.registry.pipelines.len()
        );

        Ok(manager)
    }

    /// Get model by ID, loading if necessary
    pub async fn get_model(&mut self, model_id: &str) -> Result<Arc<dyn AcousticModel>> {
        // Check if model is already loaded
        {
            let loaded = self.loaded_models.read().await;
            if let Some(model) = loaded.get(model_id) {
                debug!("Using cached model: {}", model_id);
                return Ok(model.clone());
            }
        }

        // Get model configuration
        let model_config = self
            .registry
            .models
            .get(model_id)
            .ok_or_else(|| AcousticError::ModelError(format!("Model not found: {model_id}")))?;

        info!(
            "Loading model: {} ({})",
            model_id, model_config.metadata.name
        );

        // Load the model
        let model = self.loader.load(&model_config.model_path).await?;

        // Cache the loaded model
        {
            let mut loaded = self.loaded_models.write().await;
            loaded.insert(model_id.to_string(), model.clone());
        }

        info!("Successfully loaded and cached model: {}", model_id);
        Ok(model)
    }

    /// Load a complete pipeline (acoustic model + vocoder)
    pub async fn load_pipeline(&mut self, pipeline_id: &str) -> Result<TtsPipeline> {
        // Clone the pipeline config to avoid borrowing issues
        let pipeline_config = self
            .registry
            .pipelines
            .get(pipeline_id)
            .ok_or_else(|| {
                AcousticError::ConfigError(format!("Pipeline not found: {pipeline_id}"))
            })?
            .clone();

        info!(
            "Loading TTS pipeline: {} ({})",
            pipeline_id, pipeline_config.name
        );

        // Load acoustic model
        let acoustic_model = self.get_model(&pipeline_config.acoustic_model).await?;

        // Get model config (clone to avoid borrowing issues)
        let model_config = self
            .registry
            .models
            .get(&pipeline_config.acoustic_model)
            .ok_or_else(|| {
                AcousticError::ConfigError(format!(
                    "Model config not found: {}",
                    pipeline_config.acoustic_model
                ))
            })?
            .clone();

        // Note: In a complete implementation, you would also load the vocoder model here
        // For now, we return a pipeline with just the acoustic model

        let pipeline = TtsPipeline {
            acoustic_model,
            pipeline_config,
            model_config,
            pronunciation_dictionaries: Arc::new(RwLock::new(HashMap::new())),
        };

        info!("Successfully loaded pipeline: {}", pipeline_id);
        Ok(pipeline)
    }

    /// List available models
    pub fn list_models(&self) -> Vec<(&String, &ModelConfig)> {
        self.registry.models.iter().collect()
    }

    /// List available pipelines
    pub fn list_pipelines(&self) -> Vec<(&String, &PipelineConfig)> {
        self.registry.pipelines.iter().collect()
    }

    /// Get models by language
    pub fn get_models_by_language(&self, language: LanguageCode) -> Vec<(&String, &ModelConfig)> {
        self.registry
            .models
            .iter()
            .filter(|(_, config)| config.supported_languages.contains(&language))
            .collect()
    }

    /// Get models by architecture
    pub fn get_models_by_architecture(
        &self,
        architecture: ModelArchitecture,
    ) -> Vec<(&String, &ModelConfig)> {
        self.registry
            .models
            .iter()
            .filter(|(_, config)| config.architecture == architecture)
            .collect()
    }

    /// Load pronunciation dictionary for a specific language
    pub async fn load_pronunciation_dictionary(
        &self,
        language: LanguageCode,
        dict_path: &str,
    ) -> Result<()> {
        info!(
            "Loading pronunciation dictionary for {:?} from: {}",
            language, dict_path
        );

        match PronunciationDictionary::load_from_file(dict_path, language) {
            Ok(dictionary) => {
                let (entry_count, avg_confidence) = dictionary.get_stats();
                info!(
                    "Successfully loaded {} entries (avg confidence: {:.2}) for {:?}",
                    entry_count, avg_confidence, language
                );

                let mut dictionaries = self.pronunciation_dictionaries.write().await;
                dictionaries.insert(language, dictionary);
                Ok(())
            }
            Err(e) => {
                warn!(
                    "Failed to load pronunciation dictionary from {}: {}",
                    dict_path, e
                );
                Err(e)
            }
        }
    }

    /// Check if pronunciation dictionary is loaded for a language
    pub async fn has_pronunciation_dictionary(&self, language: LanguageCode) -> bool {
        let dictionaries = self.pronunciation_dictionaries.read().await;
        dictionaries.contains_key(&language)
    }

    /// Get pronunciation dictionary statistics
    pub async fn get_dictionary_stats(&self, language: LanguageCode) -> Option<(usize, f32)> {
        let dictionaries = self.pronunciation_dictionaries.read().await;
        dictionaries.get(&language).map(|dict| dict.get_stats())
    }

    /// Download model to cache if from remote source
    pub async fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        let model_config = self
            .registry
            .models
            .get(model_id)
            .ok_or_else(|| AcousticError::ModelError(format!("Model not found: {model_id}")))?;

        let model_path = &model_config.model_path;

        // If it's already a local path, return it
        if !model_path.starts_with("http") && !model_path.contains('/') {
            // Assume it's a filename in cache directory
            return Ok(self.cache_dir.join(model_path));
        }

        // For HuggingFace Hub or URL, we would download here
        // This is a simplified implementation
        info!("Model download would be handled here for: {}", model_path);

        Ok(self.cache_dir.join(format!("{model_id}.safetensors")))
    }

    /// Clear model cache
    pub async fn clear_cache(&mut self) {
        let mut loaded = self.loaded_models.write().await;
        loaded.clear();
        info!("Cleared model cache");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> CacheStats {
        let loaded = self.loaded_models.read().await;
        CacheStats {
            loaded_models: loaded.len(),
            cache_size_mb: 0, // Would calculate actual memory usage
        }
    }

    /// Helper method to check if a character is a vowel
    fn is_vowel(ch: Option<char>) -> bool {
        match ch {
            Some(c) => matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y'),
            None => false,
        }
    }

    /// Static version of apply_stress_rules for use in TtsPipeline
    fn apply_stress_rules_static(phonemes: &mut [crate::Phoneme]) {
        use std::collections::HashMap;

        if phonemes.is_empty() {
            return;
        }

        let len = phonemes.len();

        // Simple stress assignment rules for English
        if len == 1 {
            // Single syllable words get primary stress
            let features = phonemes[0].features.get_or_insert_with(HashMap::new);
            features.insert("stress".to_string(), "1".to_string());
        } else if len == 2 {
            // Two syllable words: stress first syllable
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            let features1 = phonemes[1].features.get_or_insert_with(HashMap::new);
            features1.insert("stress".to_string(), "0".to_string());
        } else if len <= 4 {
            // Short words: stress first syllable
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            for phoneme in &mut phonemes[1..] {
                let features = phoneme.features.get_or_insert_with(HashMap::new);
                features.insert("stress".to_string(), "0".to_string());
            }
        } else {
            // Longer words: stress second syllable (rough approximation)
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "0".to_string());
            if len > 1 {
                let features1 = phonemes[1].features.get_or_insert_with(HashMap::new);
                features1.insert("stress".to_string(), "1".to_string());
            }
            for phoneme in &mut phonemes[2..] {
                let features = phoneme.features.get_or_insert_with(HashMap::new);
                features.insert("stress".to_string(), "0".to_string());
            }
        }

        // Add duration based on stress and position
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let base_duration = 0.08; // Base phoneme duration in seconds
            let stress_value = phoneme
                .features
                .as_ref()
                .and_then(|f| f.get("stress"))
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(0);
            let stress_multiplier = match stress_value {
                1 => 1.3, // Primary stress: longer
                0 => 0.9, // Unstressed: shorter
                _ => 1.0, // Default
            };

            // Position-based duration adjustment (final lengthening)
            let position_multiplier = if i == len - 1 { 1.2 } else { 1.0 };

            phoneme.duration = Some(base_duration * stress_multiplier * position_multiplier);
        }
    }

    /// Apply basic stress rules to phonemes
    #[allow(dead_code)]
    fn apply_stress_rules(&self, phonemes: &mut [crate::Phoneme]) {
        if phonemes.is_empty() {
            return;
        }

        let len = phonemes.len();

        // Simple stress assignment rules for English
        if len == 1 {
            // Single syllable words get primary stress
            let features = phonemes[0].features.get_or_insert_with(HashMap::new);
            features.insert("stress".to_string(), "1".to_string());
        } else if len == 2 {
            // Two syllable words: stress first syllable
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            let features1 = phonemes[1].features.get_or_insert_with(HashMap::new);
            features1.insert("stress".to_string(), "0".to_string());
        } else if len <= 4 {
            // Short words: stress first syllable
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            for phoneme in &mut phonemes[1..] {
                let features = phoneme.features.get_or_insert_with(HashMap::new);
                features.insert("stress".to_string(), "0".to_string());
            }
        } else {
            // Longer words: stress second syllable (rough approximation)
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "0".to_string());
            if len > 1 {
                let features1 = phonemes[1].features.get_or_insert_with(HashMap::new);
                features1.insert("stress".to_string(), "1".to_string());
            }
            for phoneme in &mut phonemes[2..] {
                let features = phoneme.features.get_or_insert_with(HashMap::new);
                features.insert("stress".to_string(), "0".to_string());
            }
        }

        // Add duration based on stress and position
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let base_duration = 0.08; // Base phoneme duration in seconds
            let stress_value = phoneme
                .features
                .as_ref()
                .and_then(|f| f.get("stress"))
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(0);
            let stress_multiplier = match stress_value {
                1 => 1.3, // Primary stress: longer
                0 => 0.9, // Unstressed: shorter
                _ => 1.0, // Default
            };

            // Position-based duration adjustment (final lengthening)
            let position_multiplier = if i == len - 1 { 1.2 } else { 1.0 };

            phoneme.duration = Some(base_duration * stress_multiplier * position_multiplier);
        }
    }
}

/// Complete TTS pipeline
pub struct TtsPipeline {
    /// Acoustic model
    acoustic_model: Arc<dyn AcousticModel>,
    /// Pipeline configuration
    pipeline_config: PipelineConfig,
    /// Model configuration
    model_config: ModelConfig,
    /// Pronunciation dictionaries for different languages
    pronunciation_dictionaries: Arc<RwLock<HashMap<LanguageCode, PronunciationDictionary>>>,
}

impl TtsPipeline {
    /// Synthesize speech from text
    pub async fn synthesize(&self, text: &str, language: Option<LanguageCode>) -> Result<Vec<f32>> {
        let target_language = language.unwrap_or(self.model_config.supported_languages[0]);

        info!(
            "Synthesizing with {} pipeline: '{}'",
            self.pipeline_config.name, text
        );

        // Convert text to phonemes using actual G2P integration
        let phonemes = self.text_to_phonemes(text, target_language).await?;

        // Use default synthesis config
        let synthesis_config = crate::SynthesisConfig::default();

        // Generate mel spectrogram using the acoustic model
        let mel_spectrogram = self
            .acoustic_model
            .synthesize(&phonemes, Some(&synthesis_config))
            .await?;

        info!(
            "Generated mel spectrogram: {} frames x {} channels",
            mel_spectrogram.n_frames, mel_spectrogram.n_mels
        );

        // Convert mel to audio using vocoder (placeholder for now)
        // This would integrate with voirs-vocoder crate in a complete implementation
        let duration = mel_spectrogram.n_frames as f32 * 256.0 / 22050.0;
        let samples = (duration * 22050.0) as usize;

        // Generate placeholder audio with basic sine wave for testing
        let mut audio = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = i as f32 / 22050.0;
            let sample = 0.1 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            audio.push(sample);
        }

        Ok(audio)
    }

    /// Convert text to phonemes using G2P configuration
    async fn text_to_phonemes(
        &self,
        text: &str,
        language: LanguageCode,
    ) -> Result<Vec<crate::Phoneme>> {
        info!("Converting text to phonemes for language: {:?}", language);

        let g2p_config = &self.pipeline_config.settings.g2p_config;

        // Get phoneme set for the target language
        let _phoneme_set = g2p_config.phoneme_sets.get(&language).ok_or_else(|| {
            AcousticError::ConfigError(format!(
                "No phoneme set configured for language: {language:?}"
            ))
        })?;

        // Integrate with voirs-g2p crate for actual G2P conversion
        use voirs_g2p::{
            backends::{neural::NeuralG2pBackend, rule_based::RuleBasedG2p},
            G2p as VoirsG2p, LanguageCode as G2pLanguageCode,
        };

        // Convert language code to G2P language code
        let g2p_lang = match language {
            LanguageCode::EnUs | LanguageCode::EnGb => G2pLanguageCode::EnUs,
            LanguageCode::DeDe => G2pLanguageCode::De,
            LanguageCode::FrFr => G2pLanguageCode::Fr,
            LanguageCode::EsEs => G2pLanguageCode::Es,
            LanguageCode::JaJp => G2pLanguageCode::Ja,
            _ => {
                // Fallback to English for unsupported languages
                warn!(
                    "Language {:?} not supported by G2P, falling back to English",
                    language
                );
                G2pLanguageCode::EnUs
            }
        };

        // Create appropriate G2P backend
        let g2p_backend: Box<dyn VoirsG2p> = match g2p_config.engine {
            crate::config::G2pEngine::RuleBased => Box::new(RuleBasedG2p::new(g2p_lang)),
            crate::config::G2pEngine::Neural => {
                match NeuralG2pBackend::new(voirs_g2p::backends::neural::LstmConfig::default()) {
                    Ok(neural_g2p) => Box::new(neural_g2p),
                    Err(e) => {
                        warn!(
                            "Failed to initialize NeuralG2pBackend: {}, falling back to RuleBased",
                            e
                        );
                        Box::new(RuleBasedG2p::new(g2p_lang))
                    }
                }
            }
            crate::config::G2pEngine::Hybrid => {
                // For hybrid, prefer neural with rule-based fallback
                match NeuralG2pBackend::new(voirs_g2p::backends::neural::LstmConfig::default()) {
                    Ok(neural_g2p) => Box::new(neural_g2p),
                    Err(_) => Box::new(RuleBasedG2p::new(g2p_lang)),
                }
            }
        };

        // Convert text to phonemes using the G2P backend
        let g2p_phonemes = g2p_backend.to_phonemes(text, Some(g2p_lang)).await?;

        // Convert voirs-g2p phonemes to acoustic phonemes
        let mut phonemes = Vec::new();
        for g2p_phoneme in g2p_phonemes {
            let mut acoustic_phoneme = crate::Phoneme::new(&g2p_phoneme.symbol);

            // Map additional properties to features HashMap
            let features = acoustic_phoneme.features.get_or_insert_with(HashMap::new);

            if let Some(ipa) = g2p_phoneme.ipa_symbol {
                features.insert("ipa_symbol".to_string(), ipa);
            }

            features.insert("stress".to_string(), g2p_phoneme.stress.to_string());
            features.insert("confidence".to_string(), g2p_phoneme.confidence.to_string());

            if let Some(duration_ms) = g2p_phoneme.duration_ms {
                acoustic_phoneme.duration = Some(duration_ms / 1000.0); // Convert ms to seconds
            }

            phonemes.push(acoustic_phoneme);
        }

        // Step 4: Post-processing and phoneme adjustment
        self.post_process_phonemes(&mut phonemes, g2p_config)?;

        Ok(phonemes)
    }

    /// Preprocess text for G2P conversion
    #[allow(dead_code)]
    fn preprocess_text(&self, text: &str, language: LanguageCode) -> Result<String> {
        let mut normalized = text.to_string();

        // Unicode normalization
        normalized = self.normalize_unicode(&normalized)?;

        // Number expansion
        normalized = self.expand_numbers(&normalized, language)?;

        // Abbreviation expansion
        normalized = self.expand_abbreviations(&normalized, language)?;

        // Special character handling
        normalized = self.handle_special_characters(&normalized)?;

        // Final cleanup
        normalized = self.normalize_whitespace(&normalized)?;

        info!("Preprocessed text: '{}' -> '{}'", text, normalized);
        Ok(normalized)
    }

    /// Normalize Unicode characters
    fn normalize_unicode(&self, text: &str) -> Result<String> {
        // Convert to lowercase and normalize Unicode
        let normalized = text.to_lowercase();

        // Handle common Unicode replacements
        let normalized = normalized
            .replace(['\u{2018}', '\u{2019}'], "'")
            .replace(['"', '"'], "\"")
            .replace(['–', '—'], "-")
            .replace('…', "...");

        Ok(normalized)
    }

    /// Expand numbers to their spoken form
    fn expand_numbers(&self, text: &str, language: LanguageCode) -> Result<String> {
        use regex::Regex;

        let number_regex = Regex::new(r"\b\d+\b").unwrap();
        let mut result = text.to_string();

        // Replace numbers with their spoken equivalents
        result = number_regex
            .replace_all(&result, |caps: &regex::Captures| {
                let number_str = &caps[0];
                if let Ok(num) = number_str.parse::<i32>() {
                    self.number_to_words(num, language)
                } else {
                    number_str.to_string()
                }
            })
            .to_string();

        // Handle ordinal numbers (1st, 2nd, 3rd, etc.)
        let ordinal_regex = Regex::new(r"\b(\d+)(st|nd|rd|th)\b").unwrap();
        result = ordinal_regex
            .replace_all(&result, |caps: &regex::Captures| {
                let number_str = &caps[1];
                if let Ok(num) = number_str.parse::<i32>() {
                    self.ordinal_to_words(num, language)
                } else {
                    caps[0].to_string()
                }
            })
            .to_string();

        Ok(result)
    }

    /// Convert number to words
    #[allow(clippy::only_used_in_recursion)]
    fn number_to_words(&self, num: i32, _language: LanguageCode) -> String {
        match num {
            0 => "zero".to_string(),
            1 => "one".to_string(),
            2 => "two".to_string(),
            3 => "three".to_string(),
            4 => "four".to_string(),
            5 => "five".to_string(),
            6 => "six".to_string(),
            7 => "seven".to_string(),
            8 => "eight".to_string(),
            9 => "nine".to_string(),
            10 => "ten".to_string(),
            11 => "eleven".to_string(),
            12 => "twelve".to_string(),
            13 => "thirteen".to_string(),
            14 => "fourteen".to_string(),
            15 => "fifteen".to_string(),
            16 => "sixteen".to_string(),
            17 => "seventeen".to_string(),
            18 => "eighteen".to_string(),
            19 => "nineteen".to_string(),
            20 => "twenty".to_string(),
            21..=99 => {
                let tens = num / 10;
                let ones = num % 10;
                let tens_word = match tens {
                    2 => "twenty",
                    3 => "thirty",
                    4 => "forty",
                    5 => "fifty",
                    6 => "sixty",
                    7 => "seventy",
                    8 => "eighty",
                    9 => "ninety",
                    _ => "",
                };
                if ones == 0 {
                    tens_word.to_string()
                } else {
                    format!("{} {}", tens_word, self.number_to_words(ones, _language))
                }
            }
            100..=999 => {
                let hundreds = num / 100;
                let remainder = num % 100;
                let hundreds_word =
                    format!("{} hundred", self.number_to_words(hundreds, _language));
                if remainder == 0 {
                    hundreds_word
                } else {
                    format!(
                        "{} {}",
                        hundreds_word,
                        self.number_to_words(remainder, _language)
                    )
                }
            }
            1000..=999999 => {
                let thousands = num / 1000;
                let remainder = num % 1000;
                let thousands_word =
                    format!("{} thousand", self.number_to_words(thousands, _language));
                if remainder == 0 {
                    thousands_word
                } else {
                    format!(
                        "{} {}",
                        thousands_word,
                        self.number_to_words(remainder, _language)
                    )
                }
            }
            _ => num.to_string(), // Fallback for very large numbers
        }
    }

    /// Convert ordinal number to words
    fn ordinal_to_words(&self, num: i32, language: LanguageCode) -> String {
        match num {
            1 => "first".to_string(),
            2 => "second".to_string(),
            3 => "third".to_string(),
            4 => "fourth".to_string(),
            5 => "fifth".to_string(),
            8 => "eighth".to_string(),
            9 => "ninth".to_string(),
            12 => "twelfth".to_string(),
            _ => {
                let base = self.number_to_words(num, language);
                if base.ends_with("y") {
                    format!("{}ieth", &base[..base.len() - 1])
                } else {
                    format!("{base}th")
                }
            }
        }
    }

    /// Expand abbreviations to their full form
    fn expand_abbreviations(&self, text: &str, _language: LanguageCode) -> Result<String> {
        let mut result = text.to_string();

        // Common abbreviations
        let abbreviations = [
            ("dr.", "doctor"),
            ("mr.", "mister"),
            ("mrs.", "misses"),
            ("ms.", "miss"),
            ("prof.", "professor"),
            ("etc.", "et cetera"),
            ("vs.", "versus"),
            ("e.g.", "for example"),
            ("i.e.", "that is"),
            ("a.m.", "a m"),
            ("p.m.", "p m"),
            ("u.s.", "united states"),
            ("u.k.", "united kingdom"),
            ("st.", "saint"),
            ("ave.", "avenue"),
            ("blvd.", "boulevard"),
            ("rd.", "road"),
            ("jr.", "junior"),
            ("sr.", "senior"),
            ("inc.", "incorporated"),
            ("ltd.", "limited"),
            ("co.", "company"),
            ("corp.", "corporation"),
        ];

        for (abbr, expansion) in &abbreviations {
            result = result.replace(abbr, expansion);
        }

        Ok(result)
    }

    /// Handle special characters and punctuation
    fn handle_special_characters(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        // Handle common symbols
        result = result.replace("&", " and ");
        result = result.replace("@", " at ");
        result = result.replace("#", " number ");
        result = result.replace("%", " percent ");
        result = result.replace("$", " dollar ");
        result = result.replace("£", " pound ");
        result = result.replace("€", " euro ");
        result = result.replace("¥", " yen ");
        result = result.replace("+", " plus ");
        result = result.replace("=", " equals ");
        result = result.replace("°", " degree ");

        // Remove remaining punctuation except apostrophes and hyphens
        result = result
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '\'' || *c == '-')
            .collect();

        Ok(result)
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> Result<String> {
        let normalized = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();

        Ok(normalized)
    }

    /// Tokenize text into words
    #[allow(dead_code)]
    fn tokenize_text(&self, text: &str, language: LanguageCode) -> Result<Vec<String>> {
        let mut words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_string())
            .filter(|w| !w.is_empty())
            .collect();

        // Apply language-specific tokenization
        words = self.handle_contractions(words, language)?;
        words = self.handle_compound_words(words, language)?;
        words = self.handle_hyphenated_words(words, language)?;

        debug!("Tokenized into {} words: {:?}", words.len(), words);
        Ok(words)
    }

    /// Handle contractions based on language
    fn handle_contractions(
        &self,
        words: Vec<String>,
        language: LanguageCode,
    ) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for word in words {
            match language {
                LanguageCode::EnUs | LanguageCode::EnGb => {
                    // English contractions
                    match word.as_str() {
                        "can't" => {
                            result.push("can".to_string());
                            result.push("not".to_string());
                        }
                        "won't" => {
                            result.push("will".to_string());
                            result.push("not".to_string());
                        }
                        "don't" => {
                            result.push("do".to_string());
                            result.push("not".to_string());
                        }
                        "doesn't" => {
                            result.push("does".to_string());
                            result.push("not".to_string());
                        }
                        "didn't" => {
                            result.push("did".to_string());
                            result.push("not".to_string());
                        }
                        "isn't" => {
                            result.push("is".to_string());
                            result.push("not".to_string());
                        }
                        "aren't" => {
                            result.push("are".to_string());
                            result.push("not".to_string());
                        }
                        "wasn't" => {
                            result.push("was".to_string());
                            result.push("not".to_string());
                        }
                        "weren't" => {
                            result.push("were".to_string());
                            result.push("not".to_string());
                        }
                        "haven't" => {
                            result.push("have".to_string());
                            result.push("not".to_string());
                        }
                        "hasn't" => {
                            result.push("has".to_string());
                            result.push("not".to_string());
                        }
                        "hadn't" => {
                            result.push("had".to_string());
                            result.push("not".to_string());
                        }
                        "shouldn't" => {
                            result.push("should".to_string());
                            result.push("not".to_string());
                        }
                        "wouldn't" => {
                            result.push("would".to_string());
                            result.push("not".to_string());
                        }
                        "couldn't" => {
                            result.push("could".to_string());
                            result.push("not".to_string());
                        }
                        "mustn't" => {
                            result.push("must".to_string());
                            result.push("not".to_string());
                        }
                        "i'm" => {
                            result.push("i".to_string());
                            result.push("am".to_string());
                        }
                        "you're" => {
                            result.push("you".to_string());
                            result.push("are".to_string());
                        }
                        "he's" => {
                            result.push("he".to_string());
                            result.push("is".to_string());
                        }
                        "she's" => {
                            result.push("she".to_string());
                            result.push("is".to_string());
                        }
                        "it's" => {
                            result.push("it".to_string());
                            result.push("is".to_string());
                        }
                        "we're" => {
                            result.push("we".to_string());
                            result.push("are".to_string());
                        }
                        "they're" => {
                            result.push("they".to_string());
                            result.push("are".to_string());
                        }
                        "i've" => {
                            result.push("i".to_string());
                            result.push("have".to_string());
                        }
                        "you've" => {
                            result.push("you".to_string());
                            result.push("have".to_string());
                        }
                        "we've" => {
                            result.push("we".to_string());
                            result.push("have".to_string());
                        }
                        "they've" => {
                            result.push("they".to_string());
                            result.push("have".to_string());
                        }
                        "i'll" => {
                            result.push("i".to_string());
                            result.push("will".to_string());
                        }
                        "you'll" => {
                            result.push("you".to_string());
                            result.push("will".to_string());
                        }
                        "he'll" => {
                            result.push("he".to_string());
                            result.push("will".to_string());
                        }
                        "she'll" => {
                            result.push("she".to_string());
                            result.push("will".to_string());
                        }
                        "it'll" => {
                            result.push("it".to_string());
                            result.push("will".to_string());
                        }
                        "we'll" => {
                            result.push("we".to_string());
                            result.push("will".to_string());
                        }
                        "they'll" => {
                            result.push("they".to_string());
                            result.push("will".to_string());
                        }
                        "i'd" => {
                            result.push("i".to_string());
                            result.push("would".to_string());
                        }
                        "you'd" => {
                            result.push("you".to_string());
                            result.push("would".to_string());
                        }
                        "he'd" => {
                            result.push("he".to_string());
                            result.push("would".to_string());
                        }
                        "she'd" => {
                            result.push("she".to_string());
                            result.push("would".to_string());
                        }
                        "it'd" => {
                            result.push("it".to_string());
                            result.push("would".to_string());
                        }
                        "we'd" => {
                            result.push("we".to_string());
                            result.push("would".to_string());
                        }
                        "they'd" => {
                            result.push("they".to_string());
                            result.push("would".to_string());
                        }
                        _ => {
                            // Handle generic apostrophe-s contractions
                            if word.ends_with("'s") && word.len() > 2 {
                                let base = &word[..word.len() - 2];
                                result.push(base.to_string());
                                result.push("is".to_string());
                            } else {
                                result.push(word);
                            }
                        }
                    }
                }
                LanguageCode::FrFr => {
                    // French contractions
                    match word.as_str() {
                        "c'est" => {
                            result.push("ce".to_string());
                            result.push("est".to_string());
                        }
                        "n'est" => {
                            result.push("ne".to_string());
                            result.push("est".to_string());
                        }
                        "l'est" => {
                            result.push("le".to_string());
                            result.push("est".to_string());
                        }
                        "d'une" => {
                            result.push("de".to_string());
                            result.push("une".to_string());
                        }
                        "qu'il" => {
                            result.push("que".to_string());
                            result.push("il".to_string());
                        }
                        "qu'elle" => {
                            result.push("que".to_string());
                            result.push("elle".to_string());
                        }
                        _ => {
                            // Handle generic French contractions
                            if word.contains('\'') {
                                let parts: Vec<&str> = word.split('\'').collect();
                                if parts.len() == 2 {
                                    match parts[0] {
                                        "l" => {
                                            result.push("le".to_string());
                                            result.push(parts[1].to_string());
                                        }
                                        "d" => {
                                            result.push("de".to_string());
                                            result.push(parts[1].to_string());
                                        }
                                        "n" => {
                                            result.push("ne".to_string());
                                            result.push(parts[1].to_string());
                                        }
                                        "c" => {
                                            result.push("ce".to_string());
                                            result.push(parts[1].to_string());
                                        }
                                        "qu" => {
                                            result.push("que".to_string());
                                            result.push(parts[1].to_string());
                                        }
                                        _ => result.push(word),
                                    }
                                } else {
                                    result.push(word);
                                }
                            } else {
                                result.push(word);
                            }
                        }
                    }
                }
                _ => {
                    // For other languages, keep as is
                    result.push(word);
                }
            }
        }

        Ok(result)
    }

    /// Handle compound words based on language
    fn handle_compound_words(
        &self,
        words: Vec<String>,
        language: LanguageCode,
    ) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for word in words {
            match language {
                LanguageCode::DeDe => {
                    // German compound word splitting (simplified)
                    // In a real implementation, this would use linguistic rules
                    if word.len() > 10 {
                        // Simple heuristic: split very long German words
                        let parts = self.split_german_compound(&word);
                        result.extend(parts);
                    } else {
                        result.push(word);
                    }
                }
                _ => {
                    // For other languages, keep as is
                    result.push(word);
                }
            }
        }

        Ok(result)
    }

    /// Simple German compound word splitting
    fn split_german_compound(&self, word: &str) -> Vec<String> {
        // This is a very simplified implementation
        // Real compound word splitting would require a dictionary

        // Common German joining morphemes
        let joining_morphemes = ["s", "es", "n", "en", "er"];

        // Try to find reasonable split points
        for i in 3..word.len() - 3 {
            let left = &word[..i];
            let right = &word[i..];

            // Check if there's a joining morpheme
            for morpheme in &joining_morphemes {
                if left.ends_with(morpheme) && right.len() > 3 {
                    return vec![
                        left[..left.len() - morpheme.len()].to_string(),
                        right.to_string(),
                    ];
                }
            }
        }

        // If no good split found, return original word
        vec![word.to_string()]
    }

    /// Handle hyphenated words
    fn handle_hyphenated_words(
        &self,
        words: Vec<String>,
        _language: LanguageCode,
    ) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for word in words {
            if word.contains('-') {
                // Split hyphenated words
                let parts: Vec<String> = word
                    .split('-')
                    .map(|s| s.to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                if parts.len() > 1 {
                    result.extend(parts);
                } else {
                    result.push(word);
                }
            } else {
                result.push(word);
            }
        }

        Ok(result)
    }

    /// Look up word in pronunciation dictionary
    #[allow(dead_code)]
    async fn lookup_dictionary(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
    ) -> Result<Option<String>> {
        // Load and cache pronunciation dictionaries
        // Handle multiple pronunciations per word
        // Support pronunciation variants by accent/formality

        if let Some(dict_path) = g2p_config.dictionaries.get(&language) {
            debug!("Looking up '{}' in dictionary: {}", word, dict_path);

            // Check if dictionary is already loaded
            {
                let dictionaries = self.pronunciation_dictionaries.read().await;
                if let Some(dictionary) = dictionaries.get(&language) {
                    // Use variant preferences if available
                    if let Some(pronunciation) =
                        dictionary.lookup_with_preferences(word, &g2p_config.variant_preferences)
                    {
                        debug!("Found pronunciation for '{}': {}", word, pronunciation);
                        return Ok(Some(pronunciation));
                    } else {
                        debug!("Word '{}' not found in loaded dictionary", word);
                        return Ok(None);
                    }
                }
            }

            // Dictionary not loaded, try to load it
            debug!("Loading pronunciation dictionary from: {}", dict_path);
            match PronunciationDictionary::load_from_file(dict_path, language) {
                Ok(dictionary) => {
                    let pronunciation =
                        dictionary.lookup_with_preferences(word, &g2p_config.variant_preferences);

                    // Cache the loaded dictionary
                    {
                        let mut dictionaries = self.pronunciation_dictionaries.write().await;
                        dictionaries.insert(language, dictionary);
                    }

                    if let Some(pronunciation) = pronunciation {
                        debug!("Found pronunciation for '{}': {}", word, pronunciation);
                        Ok(Some(pronunciation))
                    } else {
                        debug!("Word '{}' not found in newly loaded dictionary", word);
                        Ok(None)
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to load pronunciation dictionary from {}: {}",
                        dict_path, e
                    );
                    // Fallback to built-in common words for testing
                    let pronunciation = match word.to_lowercase().as_str() {
                        "hello" => Some("HH EH1 L OW0".to_string()),
                        "world" => Some("W ER1 L D".to_string()),
                        "the" => Some("DH AH0".to_string()),
                        "and" => Some("AE1 N D".to_string()),
                        "voice" => Some("V OY1 S".to_string()),
                        "synthesis" => Some("S IH1 N TH AH0 S AH0 S".to_string()),
                        "test" => Some("T EH1 S T".to_string()),
                        "example" => Some("IH0 G Z AE1 M P AH0 L".to_string()),
                        _ => None,
                    };
                    Ok(pronunciation)
                }
            }
        } else {
            debug!("No dictionary path configured for language: {:?}", language);
            Ok(None)
        }
    }

    /// Apply G2P model for unknown words
    #[allow(dead_code)]
    fn apply_g2p_model(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        // Implement actual G2P model inference
        // - Load neural G2P models
        // - Apply rule-based fallbacks
        // - Handle model confidence thresholds

        info!(
            "Applying {} G2P model for word: '{}'",
            match g2p_config.engine {
                crate::config::G2pEngine::RuleBased => "rule-based",
                crate::config::G2pEngine::Neural => "neural",
                crate::config::G2pEngine::Hybrid => "hybrid",
            },
            word
        );

        // Try different G2P approaches based on engine configuration
        match g2p_config.engine {
            crate::config::G2pEngine::Neural => {
                // Try neural G2P first
                match self.apply_neural_g2p(word, language, g2p_config, phoneme_set) {
                    Ok(phonemes) if !phonemes.is_empty() => {
                        debug!("Neural G2P succeeded for word: '{}'", word);
                        Ok(phonemes)
                    }
                    Ok(_) | Err(_) => {
                        debug!(
                            "Neural G2P failed, falling back to rule-based for word: '{}'",
                            word
                        );
                        self.apply_rule_based_g2p(word, language, g2p_config, phoneme_set)
                    }
                }
            }
            crate::config::G2pEngine::RuleBased => {
                // Use only rule-based approach
                self.apply_rule_based_g2p(word, language, g2p_config, phoneme_set)
            }
            crate::config::G2pEngine::Hybrid => {
                // Try neural first, then rule-based, then other strategies
                match self.apply_neural_g2p(word, language, g2p_config, phoneme_set) {
                    Ok(phonemes) if !phonemes.is_empty() => Ok(phonemes),
                    Ok(_) | Err(_) => {
                        match self.apply_rule_based_g2p(word, language, g2p_config, phoneme_set) {
                            Ok(phonemes) if !phonemes.is_empty() => Ok(phonemes),
                            Ok(_) | Err(_) => {
                                // Fall back to configured strategy
                                self.apply_unknown_word_strategy(word, g2p_config, phoneme_set)
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply neural G2P model inference
    #[allow(dead_code)]
    fn apply_neural_g2p(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Attempting neural G2P for word: '{}' in language: {:?}",
            word, language
        );

        // Advanced neural-style G2P system with multi-stage processing
        // This implementation provides the foundation for neural G2P integration

        // Stage 1: Character tokenization and preprocessing
        let tokens = self.tokenize_for_neural_g2p(word, language)?;
        debug!("Tokenized '{}' into {} tokens", word, tokens.len());

        // Stage 2: Simulate neural inference with enhanced rules
        let raw_phonemes = self.simulate_neural_inference(&tokens, language, phoneme_set)?;

        // Stage 3: Apply post-processing and confidence filtering
        let filtered_phonemes = self.apply_neural_postprocessing(raw_phonemes, g2p_config)?;

        // Stage 4: Add neural-style confidence scores
        let mut final_phonemes = Vec::new();
        for mut phoneme in filtered_phonemes {
            // Simulate confidence scores based on pattern complexity
            let confidence = self.calculate_phoneme_confidence(&phoneme, word);
            let features = phoneme.features.get_or_insert_with(HashMap::new);
            features.insert("confidence".to_string(), confidence.to_string());
            final_phonemes.push(phoneme);
        }

        if !final_phonemes.is_empty() {
            info!(
                "Neural-style G2P produced {} phonemes for '{}' with avg confidence: {:.2}",
                final_phonemes.len(),
                word,
                final_phonemes
                    .iter()
                    .filter_map(|p| { p.features.as_ref()?.get("confidence")?.parse::<f32>().ok() })
                    .sum::<f32>()
                    / final_phonemes.len() as f32
            );
        }

        Ok(final_phonemes)
    }

    /// Tokenize input for neural G2P processing
    fn tokenize_for_neural_g2p(&self, word: &str, language: LanguageCode) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let normalized_word = word.to_lowercase();

        // Character-level tokenization with special tokens
        tokens.push("<BOS>".to_string()); // Beginning of sequence

        for ch in normalized_word.chars() {
            if ch.is_alphabetic() {
                tokens.push(ch.to_string());
            } else if ch.is_whitespace() {
                tokens.push("<SPACE>".to_string());
            } else {
                tokens.push("<UNK>".to_string()); // Unknown character
            }
        }

        tokens.push("<EOS>".to_string()); // End of sequence

        // Add language-specific preprocessing
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                // English-specific preprocessing (already done above)
            }
            LanguageCode::JaJp => {
                // Japanese would need special handling for hiragana/katakana/kanji
                debug!("Japanese G2P tokenization not fully implemented");
            }
            _ => {
                debug!(
                    "Language-specific tokenization not implemented for {:?}",
                    language
                );
            }
        }

        Ok(tokens)
    }

    /// Simulate neural network inference with sophisticated rules
    fn simulate_neural_inference(
        &self,
        tokens: &[String],
        language: LanguageCode,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let mut phonemes = Vec::new();

        // Skip BOS and EOS tokens
        let content_tokens = &tokens[1..tokens.len().saturating_sub(1)];

        // Use enhanced context-aware processing
        for (i, token) in content_tokens.iter().enumerate() {
            if token == "<SPACE>" {
                // Add word boundary marker
                let mut boundary = crate::Phoneme::new("SIL");
                boundary.duration = Some(0.1);
                phonemes.push(boundary);
                continue;
            }

            if token == "<UNK>" {
                continue; // Skip unknown tokens
            }

            // Get context for better prediction
            let prev_token = if i > 0 {
                content_tokens.get(i - 1)
            } else {
                None
            };
            let next_token = content_tokens.get(i + 1);
            let next2_token = content_tokens.get(i + 2);

            // Apply context-sensitive rules (enhanced version of our fallback rules)
            let context_phonemes = self.get_context_aware_phonemes(
                token,
                prev_token,
                next_token,
                next2_token,
                language,
                phoneme_set,
            )?;

            phonemes.extend(context_phonemes);
        }

        Ok(phonemes)
    }

    /// Get context-aware phonemes using enhanced heuristics
    fn get_context_aware_phonemes(
        &self,
        token: &str,
        prev_token: Option<&String>,
        next_token: Option<&String>,
        next2_token: Option<&String>,
        _language: LanguageCode,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        if token.len() != 1 {
            return Ok(vec![]); // Only handle single characters
        }

        let ch = token.chars().next().unwrap();
        let _prev_ch = prev_token.and_then(|t| t.chars().next());
        let next_ch = next_token.and_then(|t| t.chars().next());
        let _next2_ch = next2_token.and_then(|t| t.chars().next());

        // Use the enhanced fallback rules but with neural-style confidence
        let phoneme_symbols = match ch {
            'a' => {
                if next_ch == Some('r') {
                    vec!["AA", "R"]
                } else if next_ch == Some('i') || next_ch == Some('y') {
                    vec!["EY"]
                } else {
                    vec!["AE"]
                }
            }
            'e' => {
                if next_ch == Some('a') {
                    vec!["IY"]
                } else if next_ch == Some('r') {
                    vec!["ER"]
                } else {
                    vec!["EH"]
                }
            }
            // Add more sophisticated rules here...
            _ => {
                // Fallback to basic mapping
                match ch {
                    'i' => vec!["IH"],
                    'o' => vec!["AA"],
                    'u' => vec!["UH"],
                    'b' => vec!["B"],
                    'c' => vec!["K"],
                    'd' => vec!["D"],
                    'f' => vec!["F"],
                    'g' => vec!["G"],
                    'h' => vec!["HH"],
                    'j' => vec!["JH"],
                    'k' => vec!["K"],
                    'l' => vec!["L"],
                    'm' => vec!["M"],
                    'n' => vec!["N"],
                    'p' => vec!["P"],
                    'r' => vec!["R"],
                    's' => vec!["S"],
                    't' => vec!["T"],
                    'v' => vec!["V"],
                    'w' => vec!["W"],
                    'y' => vec!["Y"],
                    'z' => vec!["Z"],
                    _ => vec!["AH"],
                }
            }
        };

        let mut result = Vec::new();
        for symbol in phoneme_symbols {
            if phoneme_set.symbols.contains(&symbol.to_string()) {
                result.push(crate::Phoneme::new(symbol));
            }
        }

        Ok(result)
    }

    /// Apply neural-style post-processing
    fn apply_neural_postprocessing(
        &self,
        mut phonemes: Vec<crate::Phoneme>,
        _config: &G2pConfig,
    ) -> Result<Vec<crate::Phoneme>> {
        // Remove consecutive duplicate phonemes (neural deduplication)
        phonemes.dedup_by(|a, b| a.symbol == b.symbol);

        // Apply smoothing to durations
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let base_duration = match phoneme.symbol.as_str() {
                // Vowels are typically longer
                "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY"
                | "OW" | "OY" | "UH" | "UW" => 0.12,
                // Consonants are shorter
                _ => 0.08,
            };

            // Add slight variation for naturalness
            let variation = (i as f32 * 0.003) % 0.01;
            phoneme.duration = Some(base_duration + variation);
        }

        Ok(phonemes)
    }

    /// Calculate confidence score for a phoneme based on context
    fn calculate_phoneme_confidence(&self, phoneme: &crate::Phoneme, word: &str) -> f32 {
        let base_confidence: f32 = 0.7; // Base confidence for rule-based system

        // Increase confidence for common phonemes
        let common_bonus: f32 = match phoneme.symbol.as_str() {
            "AH" | "IH" | "T" | "N" | "S" | "R" | "L" => 0.2,
            "AA" | "EH" | "K" | "D" | "M" => 0.1,
            _ => 0.0,
        };

        // Increase confidence for shorter words (easier to predict)
        let length_bonus: f32 = if word.len() <= 4 { 0.1 } else { 0.0 };

        // Ensure confidence stays within bounds
        (base_confidence + common_bonus + length_bonus).clamp(0.1, 1.0)
    }

    /// Apply rule-based G2P system
    #[allow(dead_code)]
    fn apply_rule_based_g2p(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Applying rule-based G2P for word: '{}' in language: {:?}",
            word, language
        );

        // Apply language-specific pronunciation rules
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                self.apply_english_g2p_rules(word, g2p_config, phoneme_set)
            }
            LanguageCode::DeDe => self.apply_german_g2p_rules(word, g2p_config, phoneme_set),
            LanguageCode::FrFr => self.apply_french_g2p_rules(word, g2p_config, phoneme_set),
            LanguageCode::EsEs => self.apply_spanish_g2p_rules(word, g2p_config, phoneme_set),
            LanguageCode::JaJp => self.apply_japanese_g2p_rules(word, g2p_config, phoneme_set),
            _ => {
                // Generic fallback for other languages
                self.apply_generic_g2p_rules(word, g2p_config, phoneme_set)
            }
        }
    }

    /// Apply unknown word strategy
    #[allow(dead_code)]
    fn apply_unknown_word_strategy(
        &self,
        word: &str,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        match g2p_config.unknown_word_strategy {
            UnknownWordStrategy::FallbackRules => self.apply_fallback_rules(word, phoneme_set),
            UnknownWordStrategy::LetterByLetter => {
                self.letter_by_letter_phonemes(word, phoneme_set)
            }
            UnknownWordStrategy::Skip => Ok(vec![]),
            UnknownWordStrategy::SimilarWord => {
                self.find_similar_word_pronunciation(word, phoneme_set)
            }
            UnknownWordStrategy::Error => Err(AcousticError::InputError(format!(
                "Unknown word: '{word}'. No pronunciation available."
            ))),
        }
    }

    /// Enhanced G2P rules that mimic neural network behavior
    #[allow(dead_code)]
    fn enhanced_g2p_rules(
        &self,
        word: &str,
        language: LanguageCode,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        // Apply more sophisticated rules based on context and patterns
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                self.enhanced_english_rules(word, phoneme_set)
            }
            _ => {
                // Fallback to basic rules for other languages
                self.apply_fallback_rules(word, phoneme_set)
            }
        }
    }

    /// Enhanced English G2P rules
    #[allow(dead_code)]
    fn enhanced_english_rules(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let word_lower = word.to_lowercase();
        let mut phonemes = Vec::new();

        // Handle common English patterns
        if word_lower.ends_with("ing") {
            // Handle -ing suffix
            let base = &word_lower[..word_lower.len() - 3];
            phonemes.extend(self.process_english_base(base, phoneme_set)?);
            phonemes.extend(self.create_phonemes(&["IH", "NG"], phoneme_set)?);
        } else if word_lower.ends_with("ed") {
            // Handle -ed suffix
            let base = &word_lower[..word_lower.len() - 2];
            phonemes.extend(self.process_english_base(base, phoneme_set)?);
            phonemes.extend(self.create_phonemes(&["D"], phoneme_set)?);
        } else if word_lower.ends_with("s") && word_lower.len() > 1 {
            // Handle plural -s
            let base = &word_lower[..word_lower.len() - 1];
            phonemes.extend(self.process_english_base(base, phoneme_set)?);
            phonemes.extend(self.create_phonemes(&["S"], phoneme_set)?);
        } else {
            phonemes.extend(self.process_english_base(&word_lower, phoneme_set)?);
        }

        Ok(phonemes)
    }

    /// Process English base word
    #[allow(dead_code)]
    fn process_english_base(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        // Apply more sophisticated English phonetic rules
        let mut result = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let phoneme_symbols = match chars.get(i..i + 2) {
                Some(['c', 'h']) => {
                    i += 2;
                    vec!["CH"]
                }
                Some(['s', 'h']) => {
                    i += 2;
                    vec!["SH"]
                }
                Some(['t', 'h']) => {
                    i += 2;
                    vec!["TH"]
                }
                Some(['p', 'h']) => {
                    i += 2;
                    vec!["F"]
                }
                Some(['g', 'h']) => {
                    i += 2;
                    vec!["G"]
                }
                _ => {
                    let phoneme = self.english_char_to_phoneme(chars[i]);
                    i += 1;
                    vec![phoneme]
                }
            };

            for symbol in phoneme_symbols {
                if phoneme_set.symbols.contains(&symbol.to_string()) {
                    result.push(crate::Phoneme::new(symbol));
                }
            }
        }

        Ok(result)
    }

    /// Convert English character to phoneme
    #[allow(dead_code)]
    fn english_char_to_phoneme(&self, ch: char) -> &'static str {
        match ch {
            'a' => "AE",
            'e' => "EH",
            'i' => "IH",
            'o' => "AO",
            'u' => "UH",
            'b' => "B",
            'c' => "K",
            'd' => "D",
            'f' => "F",
            'g' => "G",
            'h' => "HH",
            'j' => "JH",
            'k' => "K",
            'l' => "L",
            'm' => "M",
            'n' => "N",
            'p' => "P",
            'q' => "K",
            'r' => "R",
            's' => "S",
            't' => "T",
            'v' => "V",
            'w' => "W",
            'x' => "K",
            'y' => "Y",
            'z' => "Z",
            _ => "AH",
        }
    }

    /// Apply English G2P rules
    #[allow(dead_code)]
    fn apply_english_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        self.enhanced_english_rules(word, phoneme_set)
    }

    /// Apply German G2P rules (simplified)
    #[allow(dead_code)]
    fn apply_german_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        // Simplified German phonetic rules
        self.apply_fallback_rules(word, phoneme_set)
    }

    /// Apply French G2P rules (simplified)
    #[allow(dead_code)]
    fn apply_french_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        // Simplified French phonetic rules
        self.apply_fallback_rules(word, phoneme_set)
    }

    /// Apply Spanish G2P rules (simplified)
    #[allow(dead_code)]
    fn apply_spanish_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        // Simplified Spanish phonetic rules
        self.apply_fallback_rules(word, phoneme_set)
    }

    /// Apply Japanese G2P rules with hiragana, katakana, and basic kanji support
    #[allow(dead_code)]
    fn apply_japanese_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!("Applying Japanese G2P rules for: '{}'", word);

        let mut phonemes = Vec::new();

        for ch in word.chars() {
            let (phoneme_symbols, duration) = match ch {
                // Hiragana vowels
                'あ' => (vec!["a"], 0.12),
                'い' => (vec!["i"], 0.10),
                'う' => (vec!["u"], 0.11),
                'え' => (vec!["e"], 0.11),
                'お' => (vec!["o"], 0.12),

                // Hiragana consonants + vowels
                'か' => (vec!["k", "a"], 0.14),
                'き' => (vec!["k", "i"], 0.13),
                'く' => (vec!["k", "u"], 0.13),
                'け' => (vec!["k", "e"], 0.13),
                'こ' => (vec!["k", "o"], 0.14),
                'が' => (vec!["g", "a"], 0.14),
                'ぎ' => (vec!["g", "i"], 0.13),
                'ぐ' => (vec!["g", "u"], 0.13),
                'げ' => (vec!["g", "e"], 0.13),
                'ご' => (vec!["g", "o"], 0.14),

                'さ' => (vec!["s", "a"], 0.14),
                'し' => (vec!["sh", "i"], 0.13),
                'す' => (vec!["s", "u"], 0.13),
                'せ' => (vec!["s", "e"], 0.13),
                'そ' => (vec!["s", "o"], 0.14),
                'ざ' => (vec!["z", "a"], 0.14),
                'じ' => (vec!["zh", "i"], 0.13),
                'ず' => (vec!["z", "u"], 0.13),
                'ぜ' => (vec!["z", "e"], 0.13),
                'ぞ' => (vec!["z", "o"], 0.14),

                'た' => (vec!["t", "a"], 0.14),
                'ち' => (vec!["ch", "i"], 0.13),
                'つ' => (vec!["ts", "u"], 0.13),
                'て' => (vec!["t", "e"], 0.13),
                'と' => (vec!["t", "o"], 0.14),
                'だ' => (vec!["d", "a"], 0.14),
                'ぢ' => (vec!["d", "i"], 0.13),
                'づ' => (vec!["d", "u"], 0.13),
                'で' => (vec!["d", "e"], 0.13),
                'ど' => (vec!["d", "o"], 0.14),

                'な' => (vec!["n", "a"], 0.14),
                'に' => (vec!["n", "i"], 0.13),
                'ぬ' => (vec!["n", "u"], 0.13),
                'ね' => (vec!["n", "e"], 0.13),
                'の' => (vec!["n", "o"], 0.14),

                'は' => (vec!["h", "a"], 0.14),
                'ひ' => (vec!["h", "i"], 0.13),
                'ふ' => (vec!["f", "u"], 0.13),
                'へ' => (vec!["h", "e"], 0.13),
                'ほ' => (vec!["h", "o"], 0.14),
                'ば' => (vec!["b", "a"], 0.14),
                'び' => (vec!["b", "i"], 0.13),
                'ぶ' => (vec!["b", "u"], 0.13),
                'べ' => (vec!["b", "e"], 0.13),
                'ぼ' => (vec!["b", "o"], 0.14),
                'ぱ' => (vec!["p", "a"], 0.14),
                'ぴ' => (vec!["p", "i"], 0.13),
                'ぷ' => (vec!["p", "u"], 0.13),
                'ぺ' => (vec!["p", "e"], 0.13),
                'ぽ' => (vec!["p", "o"], 0.14),

                'ま' => (vec!["m", "a"], 0.14),
                'み' => (vec!["m", "i"], 0.13),
                'む' => (vec!["m", "u"], 0.13),
                'め' => (vec!["m", "e"], 0.13),
                'も' => (vec!["m", "o"], 0.14),

                'や' => (vec!["y", "a"], 0.13),
                'ゆ' => (vec!["y", "u"], 0.13),
                'よ' => (vec!["y", "o"], 0.13),

                'ら' => (vec!["r", "a"], 0.14),
                'り' => (vec!["r", "i"], 0.13),
                'る' => (vec!["r", "u"], 0.13),
                'れ' => (vec!["r", "e"], 0.13),
                'ろ' => (vec!["r", "o"], 0.14),

                'わ' => (vec!["w", "a"], 0.13),
                'ゐ' => (vec!["w", "i"], 0.13),
                'ゑ' => (vec!["w", "e"], 0.13),
                'を' => (vec!["w", "o"], 0.13),
                'ん' => (vec!["N"], 0.10), // moraic nasal

                // Katakana (same phonemes as hiragana)
                'ア' => (vec!["a"], 0.12),
                'イ' => (vec!["i"], 0.10),
                'ウ' => (vec!["u"], 0.11),
                'エ' => (vec!["e"], 0.11),
                'オ' => (vec!["o"], 0.12),

                'カ' => (vec!["k", "a"], 0.14),
                'キ' => (vec!["k", "i"], 0.13),
                'ク' => (vec!["k", "u"], 0.13),
                'ケ' => (vec!["k", "e"], 0.13),
                'コ' => (vec!["k", "o"], 0.14),

                'サ' => (vec!["s", "a"], 0.14),
                'シ' => (vec!["sh", "i"], 0.13),
                'ス' => (vec!["s", "u"], 0.13),
                'セ' => (vec!["s", "e"], 0.13),
                'ソ' => (vec!["s", "o"], 0.14),

                'タ' => (vec!["t", "a"], 0.14),
                'チ' => (vec!["ch", "i"], 0.13),
                'ツ' => (vec!["ts", "u"], 0.13),
                'テ' => (vec!["t", "e"], 0.13),
                'ト' => (vec!["t", "o"], 0.14),

                'ナ' => (vec!["n", "a"], 0.14),
                'ニ' => (vec!["n", "i"], 0.13),
                'ヌ' => (vec!["n", "u"], 0.13),
                'ネ' => (vec!["n", "e"], 0.13),
                'ノ' => (vec!["n", "o"], 0.14),

                'ハ' => (vec!["h", "a"], 0.14),
                'ヒ' => (vec!["h", "i"], 0.13),
                'フ' => (vec!["f", "u"], 0.13),
                'ヘ' => (vec!["h", "e"], 0.13),
                'ホ' => (vec!["h", "o"], 0.14),

                'マ' => (vec!["m", "a"], 0.14),
                'ミ' => (vec!["m", "i"], 0.13),
                'ム' => (vec!["m", "u"], 0.13),
                'メ' => (vec!["m", "e"], 0.13),
                'モ' => (vec!["m", "o"], 0.14),

                'ヤ' => (vec!["y", "a"], 0.13),
                'ユ' => (vec!["y", "u"], 0.13),
                'ヨ' => (vec!["y", "o"], 0.13),

                'ラ' => (vec!["r", "a"], 0.14),
                'リ' => (vec!["r", "i"], 0.13),
                'ル' => (vec!["r", "u"], 0.13),
                'レ' => (vec!["r", "e"], 0.13),
                'ロ' => (vec!["r", "o"], 0.14),

                'ワ' => (vec!["w", "a"], 0.13),
                'ヲ' => (vec!["w", "o"], 0.13),
                'ン' => (vec!["N"], 0.10),

                // Special markers
                'っ' | 'ッ' => (vec!["Q"], 0.08), // geminate (small tsu)
                'ー' => (vec!["LONG"], 0.06),     // long vowel mark

                // Punctuation and pauses
                '。' | '.' => (vec!["sil"], 0.30),
                '、' | ',' => (vec!["sp"], 0.15),
                ' ' => (vec!["sp"], 0.10),

                // Basic ASCII fallback
                c if c.is_ascii_alphabetic() => {
                    // Use English fallback for romaji
                    let symbol = match c.to_ascii_lowercase() {
                        'a' => "a",
                        'e' => "e",
                        'i' => "i",
                        'o' => "o",
                        'u' => "u",
                        'k' => "k",
                        'g' => "g",
                        's' => "s",
                        'z' => "z",
                        't' => "t",
                        'd' => "d",
                        'n' => "n",
                        'h' => "h",
                        'b' => "b",
                        'p' => "p",
                        'm' => "m",
                        'y' => "y",
                        'r' => "r",
                        'w' => "w",
                        'f' => "f",
                        'j' => "j",
                        'l' => "r", // L sounds become R in Japanese
                        'v' => "b", // V sounds become B in Japanese
                        _ => "a",   // Default vowel
                    };
                    (vec![symbol], 0.10)
                }

                // Unknown characters - silent pause
                _ => (vec!["sp"], 0.05),
            };

            // Create phonemes for this character
            for symbol in phoneme_symbols {
                if phoneme_set.symbols.contains(&symbol.to_string()) {
                    let mut phoneme = crate::Phoneme::new(symbol);
                    phoneme.duration = Some(duration);
                    phonemes.push(phoneme);
                }
            }
        }

        // Add final pause if needed
        if !phonemes.is_empty() {
            let mut final_pause = crate::Phoneme::new("sil");
            final_pause.duration = Some(0.20);
            phonemes.push(final_pause);
        }

        debug!("Generated {} phonemes for Japanese text", phonemes.len());
        Ok(phonemes)
    }

    /// Apply generic G2P rules for unsupported languages
    #[allow(dead_code)]
    fn apply_generic_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        self.apply_fallback_rules(word, phoneme_set)
    }

    /// Helper to create phonemes from symbol array
    #[allow(dead_code)]
    fn create_phonemes(
        &self,
        symbols: &[&str],
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let mut phonemes = Vec::new();
        for &symbol in symbols {
            if phoneme_set.symbols.contains(&symbol.to_string()) {
                phonemes.push(crate::Phoneme::new(symbol));
            }
        }
        Ok(phonemes)
    }

    /// Apply advanced language-specific pronunciation rules
    #[allow(dead_code)]
    fn apply_fallback_rules(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!("Applying enhanced fallback rules for: '{}'", word);

        let normalized_word = word.to_lowercase();
        let chars: Vec<char> = normalized_word.chars().collect();
        let mut phonemes = Vec::new();

        // Enhanced English letter-to-sound rules with context sensitivity
        let mut i = 0;
        while i < chars.len() {
            let current_char = chars[i];
            let next_char = chars.get(i + 1).copied();
            let prev_char = if i > 0 {
                chars.get(i - 1).copied()
            } else {
                None
            };
            let next2_char = chars.get(i + 2).copied();

            let phoneme_symbols = match current_char {
                // Vowels with context-sensitive rules
                'a' => {
                    match next_char {
                        Some('r') => {
                            vec!["AA", "R"] // "car" or "far"
                        }
                        Some('i') | Some('y') => vec!["EY"], // "day", "may"
                        Some('u') => vec!["AO"],             // "caught"
                        Some('w') => vec!["AO"],             // "saw"
                        Some('l') if next2_char.map_or(true, |c| !c.is_alphabetic()) => {
                            vec!["AO", "L"]
                        } // "all"
                        _ => {
                            // Check for silent 'e' at end affecting pronunciation
                            if chars.len() > i + 2
                                && chars[chars.len() - 1] == 'e'
                                && chars[chars.len() - 2] != 'e'
                            {
                                vec!["EY"] // "rate" vs "rat"
                            } else {
                                vec!["AE"] // "cat"
                            }
                        }
                    }
                }
                'e' => {
                    match next_char {
                        Some('a') => vec!["IY"], // "eat"
                        Some('e') => vec!["IY"], // "meet"
                        Some('i') => vec!["EY"], // "eight"
                        Some('r') => vec!["ER"], // "her"
                        Some('w') => vec!["UW"], // "new"
                        _ => {
                            if i == chars.len() - 1 {
                                vec![] // Silent 'e' at end
                            } else {
                                vec!["EH"] // "bet"
                            }
                        }
                    }
                }
                'i' => {
                    match next_char {
                        Some('r') => vec!["ER"],                            // "bird"
                        Some('e') => vec!["AY"],                            // "pie"
                        Some('g') if next2_char == Some('h') => vec!["AY"], // "light"
                        _ => {
                            if chars.len() > i + 2 && chars[chars.len() - 1] == 'e' {
                                vec!["AY"] // "bite"
                            } else {
                                vec!["IH"] // "bit"
                            }
                        }
                    }
                }
                'o' => {
                    match next_char {
                        Some('a') => vec!["OW"], // "boat"
                        Some('o') => vec!["UW"], // "soon"
                        Some('u') => vec!["AW"], // "house"
                        Some('w') => vec!["OW"], // "flow"
                        Some('r') => {
                            vec!["AO", "R"] // "for" or "port"
                        }
                        Some('y') => vec!["OY"], // "boy"
                        _ => {
                            if chars.len() > i + 2 && chars[chars.len() - 1] == 'e' {
                                vec!["OW"] // "note"
                            } else {
                                vec!["AA"] // "not"
                            }
                        }
                    }
                }
                'u' => {
                    match next_char {
                        Some('r') => vec!["ER"], // "burn"
                        Some('e') => vec!["UW"], // "true"
                        Some('i') => vec!["UW"], // "fruit"
                        _ => {
                            if chars.len() > i + 2 && chars[chars.len() - 1] == 'e' {
                                vec!["UW"] // "cute"
                            } else {
                                vec!["UH"] // "but"
                            }
                        }
                    }
                }
                'y' => {
                    if i == 0 {
                        vec!["Y"] // "yes"
                    } else if i == chars.len() - 1 {
                        vec!["IY"] // "happy"
                    } else {
                        vec!["IH"] // "symbol"
                    }
                }

                // Consonants with digraph and context handling
                'c' => {
                    match next_char {
                        Some('h') => {
                            i += 1; // Skip the 'h'
                            vec!["CH"] // "chat"
                        }
                        Some('k') => {
                            i += 1; // Skip the 'k'
                            vec!["K"] // "pick"
                        }
                        Some('e') | Some('i') | Some('y') => vec!["S"], // "city", "cell"
                        _ => vec!["K"],                                 // "cat"
                    }
                }
                'g' => {
                    match next_char {
                        Some('h') => {
                            i += 1; // Skip the 'h'
                            if prev_char == Some('u') {
                                vec![] // "laugh" - silent gh
                            } else {
                                vec!["F"] // "cough"
                            }
                        }
                        Some('n') => {
                            if i == chars.len() - 2 {
                                vec!["NG"] // "sing"
                            } else {
                                vec!["G", "N"] // "signal"
                            }
                        }
                        Some('e') | Some('i') | Some('y') => vec!["JH"], // "gem", "gin"
                        _ => vec!["G"],                                  // "go"
                    }
                }
                'p' => {
                    match next_char {
                        Some('h') => {
                            i += 1; // Skip the 'h'
                            vec!["F"] // "phone"
                        }
                        _ => vec!["P"], // "pat"
                    }
                }
                's' => {
                    match next_char {
                        Some('h') => {
                            i += 1; // Skip the 'h'
                            vec!["SH"] // "shop"
                        }
                        Some('s') => vec!["S"], // "pass"
                        _ => {
                            // Check for voiced 's' between vowels
                            if (ModelManager::is_vowel(prev_char)
                                && ModelManager::is_vowel(next_char))
                                || (i == chars.len() - 1 && ModelManager::is_vowel(prev_char))
                            {
                                vec!["Z"] // "rose" or "dogs"
                            } else {
                                vec!["S"] // "sat"
                            }
                        }
                    }
                }
                't' => {
                    match next_char {
                        Some('h') => {
                            i += 1; // Skip the 'h'
                            vec!["TH"] // "think"
                        }
                        Some('i') if next2_char == Some('o') => vec!["SH"], // "nation"
                        _ => vec!["T"],                                     // "top"
                    }
                }
                'w' => {
                    match next_char {
                        Some('h') => {
                            i += 1; // Skip the 'h'
                            vec!["W"] // "when"
                        }
                        Some('r') => {
                            i += 1; // Skip the 'r'
                            vec!["R"] // "write"
                        }
                        _ => vec!["W"], // "win"
                    }
                }
                'x' => vec!["K", "S"], // "box"
                'q' => {
                    if next_char == Some('u') {
                        i += 1; // Skip the 'u'
                        vec!["K", "W"] // "queen"
                    } else {
                        vec!["K"] // "qi"
                    }
                }

                // Simple consonants
                'b' => vec!["B"],
                'd' => vec!["D"],
                'f' => vec!["F"],
                'h' => {
                    if i == 0 || !ModelManager::is_vowel(prev_char) {
                        vec!["HH"] // "hat"
                    } else {
                        vec![] // Silent 'h' in many contexts
                    }
                }
                'j' => vec!["JH"],
                'k' => vec!["K"],
                'l' => vec!["L"],
                'm' => vec!["M"],
                'n' => vec!["N"],
                'r' => vec!["R"],
                'v' => vec!["V"],
                'z' => vec!["Z"],

                // Handle common letter combinations and silent letters
                _ => {
                    if current_char.is_alphabetic() {
                        vec!["AH"] // Default to schwa for unknown letters
                    } else {
                        vec![] // Skip non-alphabetic characters
                    }
                }
            };

            // Add phonemes to result, ensuring they exist in the phoneme set
            for symbol in phoneme_symbols {
                if phoneme_set.symbols.contains(&symbol.to_string()) {
                    phonemes.push(crate::Phoneme::new(symbol));
                } else {
                    // Fallback to a default phoneme if the symbol isn't in the set
                    phonemes.push(crate::Phoneme::new("AH"));
                }
            }

            i += 1;
        }

        // Apply basic syllable stress rules
        ModelManager::apply_stress_rules_static(&mut phonemes);

        debug!(
            "Enhanced G2P produced {} phonemes for '{}'",
            phonemes.len(),
            word
        );

        Ok(phonemes)
    }

    /// Generate enhanced letter-by-letter phonemes with better mapping
    #[allow(dead_code)]
    fn letter_by_letter_phonemes(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Generating enhanced letter-by-letter phonemes for: '{}'",
            word
        );

        let mut phonemes = Vec::new();

        for (i, ch) in word.chars().enumerate() {
            if ch.is_alphabetic() {
                // Map individual letters to reasonable phonetic approximations
                let phoneme_symbol = match ch.to_lowercase().next().unwrap() {
                    'a' => "EY", // Letter name "A"
                    'b' => "B",  // Letter name "bee"
                    'c' => "S",  // Letter name "see"
                    'd' => "D",  // Letter name "dee"
                    'e' => "IY", // Letter name "E"
                    'f' => "F",  // Letter name "eff"
                    'g' => "JH", // Letter name "gee"
                    'h' => "EY", // Letter name "aitch"
                    'i' => "AY", // Letter name "I"
                    'j' => "JH", // Letter name "jay"
                    'k' => "K",  // Letter name "kay"
                    'l' => "L",  // Letter name "ell"
                    'm' => "M",  // Letter name "em"
                    'n' => "N",  // Letter name "en"
                    'o' => "OW", // Letter name "O"
                    'p' => "P",  // Letter name "pee"
                    'q' => "K",  // Letter name "cue"
                    'r' => "R",  // Letter name "arr"
                    's' => "S",  // Letter name "ess"
                    't' => "T",  // Letter name "tee"
                    'u' => "UW", // Letter name "U"
                    'v' => "V",  // Letter name "vee"
                    'w' => "W",  // Letter name "double-u"
                    'x' => "K",  // Letter name "ex"
                    'y' => "W",  // Letter name "why"
                    'z' => "Z",  // Letter name "zee"
                    _ => "AH",   // Default fallback
                };

                // Ensure the phoneme exists in the phoneme set
                let symbol = if phoneme_set.symbols.contains(&phoneme_symbol.to_string()) {
                    phoneme_symbol
                } else {
                    "AH" // Fallback if phoneme not in set
                };

                let mut phoneme = crate::Phoneme::new(symbol);

                // Vary duration slightly for more natural pronunciation
                let base_duration = 0.12; // Slightly longer for letter-by-letter
                let variation = (i as f32 * 0.005) % 0.02; // Small variation
                phoneme.duration = Some(base_duration + variation);

                // Add brief pause between letters
                phonemes.push(phoneme);

                // Add a small pause between letters (except for the last one)
                if i < word.len() - 1 {
                    let mut pause = crate::Phoneme::new("SIL");
                    pause.duration = Some(0.05); // Short pause
                    phonemes.push(pause);
                }
            }
        }

        debug!(
            "Letter-by-letter G2P produced {} phonemes for '{}'",
            phonemes.len(),
            word
        );
        Ok(phonemes)
    }

    /// Find similar word pronunciation
    #[allow(dead_code)]
    fn find_similar_word_pronunciation(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        // Implement phonetic similarity matching
        // - Edit distance algorithms
        // - Phonetic pattern matching
        // - Sound similarity metrics

        debug!("Finding similar word pronunciation for: '{}'", word);

        // Find the most similar known word and use its pronunciation pattern
        let similar_word = self.find_most_similar_word(word);

        if let Some((similar, similarity_score)) = similar_word {
            if similarity_score > 0.7 {
                // High similarity threshold
                debug!(
                    "Found similar word '{}' with similarity {:.2} for '{}'",
                    similar, similarity_score, word
                );

                // Apply phonetic transformation based on the similar word
                return self.apply_phonetic_transformation(word, &similar, phoneme_set);
            }
        }

        // If no similar word found, use enhanced pattern matching
        self.apply_pattern_based_pronunciation(word, phoneme_set)
    }

    /// Find the most similar known word using multiple similarity metrics
    #[allow(dead_code)]
    fn find_most_similar_word(&self, target: &str) -> Option<(String, f32)> {
        let known_words = self.get_known_words();
        let mut best_match: Option<(String, f32)> = None;

        for word in known_words {
            // Calculate multiple similarity metrics
            let edit_distance_sim = self.calculate_edit_distance_similarity(target, &word);
            let phonetic_sim = self.calculate_phonetic_similarity(target, &word);
            let pattern_sim = self.calculate_pattern_similarity(target, &word);

            // Weighted combination of similarities
            let combined_similarity =
                edit_distance_sim * 0.4 + phonetic_sim * 0.4 + pattern_sim * 0.2;

            if let Some((_, current_best)) = &best_match {
                if combined_similarity > *current_best {
                    best_match = Some((word, combined_similarity));
                }
            } else {
                best_match = Some((word, combined_similarity));
            }
        }

        best_match
    }

    /// Calculate edit distance similarity (normalized Levenshtein distance)
    #[allow(dead_code)]
    fn calculate_edit_distance_similarity(&self, s1: &str, s2: &str) -> f32 {
        let distance = self.levenshtein_distance(s1, s2);
        let max_len = s1.len().max(s2.len()) as f32;

        if max_len == 0.0 {
            return 1.0;
        }

        1.0 - (distance as f32 / max_len)
    }

    /// Calculate Levenshtein distance between two strings
    #[allow(dead_code)]
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        #[allow(clippy::needless_range_loop)]
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        // Fill the matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Calculate phonetic similarity using sound patterns
    #[allow(dead_code)]
    fn calculate_phonetic_similarity(&self, s1: &str, s2: &str) -> f32 {
        // Convert words to phonetic representations
        let phonetic1 = self.word_to_phonetic_pattern(s1);
        let phonetic2 = self.word_to_phonetic_pattern(s2);

        // Calculate similarity between phonetic patterns
        self.pattern_similarity(&phonetic1, &phonetic2)
    }

    /// Convert word to phonetic pattern representation
    #[allow(dead_code)]
    fn word_to_phonetic_pattern(&self, word: &str) -> Vec<char> {
        word.chars()
            .map(|c| self.char_to_phonetic_class(c))
            .collect()
    }

    /// Map character to phonetic class
    #[allow(dead_code)]
    fn char_to_phonetic_class(&self, c: char) -> char {
        match c.to_ascii_lowercase() {
            // Vowels
            'a' | 'e' | 'i' | 'o' | 'u' => 'V',
            // Plosives
            'p' | 'b' | 't' | 'd' | 'k' | 'g' => 'P',
            // Fricatives
            'f' | 'v' | 's' | 'z' | 'h' => 'F',
            // Nasals
            'm' | 'n' => 'N',
            // Liquids
            'l' | 'r' => 'L',
            // Glides
            'w' | 'y' => 'G',
            // Other consonants
            _ => 'C',
        }
    }

    /// Calculate pattern similarity between two phonetic patterns
    #[allow(dead_code)]
    fn pattern_similarity(&self, pattern1: &[char], pattern2: &[char]) -> f32 {
        if pattern1.is_empty() && pattern2.is_empty() {
            return 1.0;
        }

        let matching_positions = pattern1
            .iter()
            .zip(pattern2.iter())
            .filter(|(a, b)| a == b)
            .count();

        let max_len = pattern1.len().max(pattern2.len());

        if max_len == 0 {
            return 1.0;
        }

        matching_positions as f32 / max_len as f32
    }

    /// Calculate structural pattern similarity (prefixes, suffixes, etc.)
    #[allow(dead_code)]
    fn calculate_pattern_similarity(&self, s1: &str, s2: &str) -> f32 {
        let mut similarity = 0.0;

        // Check common prefixes
        let prefix_len = self.common_prefix_length(s1, s2);
        similarity += (prefix_len as f32 / s1.len().max(s2.len()) as f32) * 0.5;

        // Check common suffixes
        let suffix_len = self.common_suffix_length(s1, s2);
        similarity += (suffix_len as f32 / s1.len().max(s2.len()) as f32) * 0.5;

        similarity.min(1.0)
    }

    /// Calculate common prefix length
    #[allow(dead_code)]
    fn common_prefix_length(&self, s1: &str, s2: &str) -> usize {
        s1.chars()
            .zip(s2.chars())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Calculate common suffix length
    #[allow(dead_code)]
    fn common_suffix_length(&self, s1: &str, s2: &str) -> usize {
        s1.chars()
            .rev()
            .zip(s2.chars().rev())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Get list of known words for similarity matching
    #[allow(dead_code)]
    fn get_known_words(&self) -> Vec<String> {
        // In a real implementation, this would query the loaded dictionaries
        // For now, return a curated list of common words
        vec![
            "hello".to_string(),
            "world".to_string(),
            "voice".to_string(),
            "speech".to_string(),
            "text".to_string(),
            "synthesis".to_string(),
            "example".to_string(),
            "test".to_string(),
            "word".to_string(),
            "sound".to_string(),
            "language".to_string(),
            "pronunciation".to_string(),
            "phoneme".to_string(),
            "acoustic".to_string(),
            "model".to_string(),
        ]
    }

    /// Apply phonetic transformation based on similar word
    #[allow(dead_code)]
    fn apply_phonetic_transformation(
        &self,
        target: &str,
        similar: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Applying phonetic transformation from '{}' to '{}'",
            similar, target
        );

        // Get pronunciation for similar word (simplified)
        let similar_phonemes = self.enhanced_english_rules(similar, phoneme_set)?;

        // Apply transformation based on character differences
        let transformed = self.transform_pronunciation(target, similar, similar_phonemes)?;

        Ok(transformed)
    }

    /// Transform pronunciation based on character mapping
    #[allow(dead_code)]
    fn transform_pronunciation(
        &self,
        target: &str,
        source: &str,
        source_phonemes: Vec<crate::Phoneme>,
    ) -> Result<Vec<crate::Phoneme>> {
        // Simple approach: if words are similar length, map phonemes proportionally
        let target_len = target.len();
        let _source_len = source.len();

        if target_len == 0 {
            return Ok(vec![]);
        }

        let mut result = Vec::new();

        if source_phonemes.is_empty() {
            return Ok(result);
        }

        // Map phonemes proportionally to character positions
        for (i, _) in target.chars().enumerate() {
            let source_idx = (i * source_phonemes.len()) / target_len;
            if source_idx < source_phonemes.len() {
                result.push(source_phonemes[source_idx].clone());
            }
        }

        Ok(result)
    }

    /// Apply pattern-based pronunciation when no similar word is found
    #[allow(dead_code)]
    fn apply_pattern_based_pronunciation(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!("Applying pattern-based pronunciation for: '{}'", word);

        // Use enhanced rules as fallback
        self.enhanced_english_rules(word, phoneme_set)
    }

    /// Parse pronunciation string into phonemes
    #[allow(dead_code)]
    fn parse_pronunciation(
        &self,
        pronunciation: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let mut phonemes = Vec::new();

        for phoneme_str in pronunciation.split_whitespace() {
            // Extract stress marker if present
            let (symbol, stress_marker) = if phoneme_str.len() > 1 {
                let last_char = phoneme_str.chars().last().unwrap();
                if phoneme_set
                    .stress_markers
                    .iter()
                    .any(|m| m == &last_char.to_string())
                {
                    (
                        &phoneme_str[..phoneme_str.len() - 1],
                        Some(last_char.to_string()),
                    )
                } else {
                    (phoneme_str, None)
                }
            } else {
                (phoneme_str, None)
            };

            if phoneme_set.symbols.contains(&symbol.to_string()) {
                let mut phoneme = crate::Phoneme::new(symbol);
                phoneme.duration = Some(0.08);

                // Add stress information as features
                if let Some(stress) = stress_marker {
                    let features = phoneme.features.get_or_insert_with(HashMap::new);
                    features.insert("stress".to_string(), stress);
                }

                phonemes.push(phoneme);
            }
        }

        Ok(phonemes)
    }

    /// Predict stress patterns for phonemes
    #[allow(dead_code)]
    fn predict_stress(
        &self,
        phonemes: &mut [crate::Phoneme],
        word: &str,
        language: LanguageCode,
        stress_config: &StressConfig,
    ) -> Result<()> {
        // Implement neural stress prediction
        // - Syllable detection
        // - Stress pattern classification
        // - Context-aware stress assignment

        if !stress_config.predict_stress {
            return Ok(());
        }

        debug!(
            "Predicting stress for word: '{}' in language: {:?}",
            word, language
        );

        // Detect syllable boundaries
        let syllable_boundaries = self.detect_syllable_boundaries(phonemes)?;

        // Apply language-specific stress rules
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                self.apply_english_stress_rules(phonemes, word, &syllable_boundaries, stress_config)
            }
            LanguageCode::DeDe => {
                self.apply_german_stress_rules(phonemes, word, &syllable_boundaries, stress_config)
            }
            LanguageCode::FrFr => {
                self.apply_french_stress_rules(phonemes, word, &syllable_boundaries, stress_config)
            }
            LanguageCode::EsEs => {
                self.apply_spanish_stress_rules(phonemes, word, &syllable_boundaries, stress_config)
            }
            _ => {
                self.apply_default_stress_rules(phonemes, word, &syllable_boundaries, stress_config)
            }
        }
    }

    /// Detect syllable boundaries in phoneme sequence
    #[allow(dead_code)]
    fn detect_syllable_boundaries(&self, phonemes: &[crate::Phoneme]) -> Result<Vec<usize>> {
        let mut boundaries = vec![0]; // Start of first syllable

        // Find vowel nuclei and syllable boundaries
        for (i, phoneme) in phonemes.iter().enumerate() {
            if self.is_vowel_phoneme(&phoneme.symbol) {
                // Check if this starts a new syllable
                if i > 0 && !self.is_vowel_phoneme(&phonemes[i - 1].symbol) {
                    // Look ahead for consonant clusters
                    let next_vowel_pos = self.find_next_vowel(phonemes, i + 1);
                    if let Some(next_pos) = next_vowel_pos {
                        let consonants_between = next_pos - i - 1;
                        if consonants_between > 0 {
                            // Apply maximal onset principle
                            let split_point =
                                self.find_syllable_split_point(phonemes, i + 1, next_pos);
                            if split_point > *boundaries.last().unwrap_or(&0) {
                                boundaries.push(split_point);
                            }
                        }
                    }
                }
            }
        }

        debug!("Detected syllable boundaries: {:?}", boundaries);
        Ok(boundaries)
    }

    /// Find the next vowel position
    #[allow(dead_code)]
    fn find_next_vowel(&self, phonemes: &[crate::Phoneme], start: usize) -> Option<usize> {
        for (i, phoneme) in phonemes.iter().enumerate().skip(start) {
            if self.is_vowel_phoneme(&phoneme.symbol) {
                return Some(i);
            }
        }
        None
    }

    /// Find optimal syllable split point using phonotactic rules
    #[allow(dead_code)]
    fn find_syllable_split_point(
        &self,
        phonemes: &[crate::Phoneme],
        start: usize,
        end: usize,
    ) -> usize {
        if start >= end {
            return start;
        }

        let consonant_count = end - start;
        match consonant_count {
            1 => end, // Single consonant goes to next syllable (maximal onset)
            2 => {
                // Check if consonant cluster is valid onset
                if self.is_valid_onset_cluster(&phonemes[start..end]) {
                    end // Both consonants go to next syllable
                } else {
                    start + 1 // Split between consonants
                }
            }
            _ => start + 1, // Complex clusters: split after first consonant
        }
    }

    /// Check if consonant cluster is a valid syllable onset
    #[allow(dead_code)]
    fn is_valid_onset_cluster(&self, consonants: &[crate::Phoneme]) -> bool {
        if consonants.len() != 2 {
            return false;
        }

        let c1 = &consonants[0].symbol;
        let c2 = &consonants[1].symbol;

        // Common English onset clusters
        matches!(
            (c1.as_str(), c2.as_str()),
            ("B", "L")
                | ("B", "R")
                | ("K", "L")
                | ("K", "R")
                | ("G", "L")
                | ("G", "R")
                | ("P", "L")
                | ("P", "R")
                | ("T", "R")
                | ("D", "R")
                | ("F", "L")
                | ("F", "R")
                | ("S", "K")
                | ("S", "P")
                | ("S", "T")
                | ("S", "M")
                | ("S", "N")
                | ("TH", "R")
        )
    }

    /// Apply English stress rules
    #[allow(dead_code)]
    fn apply_english_stress_rules(
        &self,
        phonemes: &mut [crate::Phoneme],
        word: &str,
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        if syllable_boundaries.len() < 2 {
            return Ok(()); // Monosyllabic words typically unstressed
        }

        // Get vowel positions for each syllable
        let syllable_vowels = self.find_syllable_vowels(phonemes, syllable_boundaries)?;

        // Apply English stress patterns based on word structure
        let primary_stress_syllable = self.predict_english_primary_stress(word, &syllable_vowels);

        // Apply primary stress
        if let Some(stress_syllable) = primary_stress_syllable {
            if stress_syllable < syllable_vowels.len() {
                if let Some(vowel_idx) = syllable_vowels[stress_syllable] {
                    let features = phonemes[vowel_idx]
                        .features
                        .get_or_insert_with(HashMap::new);
                    features.insert(
                        "stress".to_string(),
                        stress_config.primary_stress_marker.clone(),
                    );
                    debug!(
                        "Applied primary stress to syllable {} (phoneme {})",
                        stress_syllable, vowel_idx
                    );
                }
            }
        }

        // Apply secondary stress for longer words
        if syllable_vowels.len() > 3 {
            let secondary_stress_syllable = self.predict_english_secondary_stress(
                word,
                &syllable_vowels,
                primary_stress_syllable,
            );
            if let Some(sec_syllable) = secondary_stress_syllable {
                if sec_syllable < syllable_vowels.len() {
                    if let Some(vowel_idx) = syllable_vowels[sec_syllable] {
                        let features = phonemes[vowel_idx]
                            .features
                            .get_or_insert_with(HashMap::new);
                        features.insert(
                            "stress".to_string(),
                            stress_config.secondary_stress_marker.clone(),
                        );
                        debug!(
                            "Applied secondary stress to syllable {} (phoneme {})",
                            sec_syllable, vowel_idx
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Find vowel positions for each syllable
    #[allow(dead_code)]
    fn find_syllable_vowels(
        &self,
        phonemes: &[crate::Phoneme],
        boundaries: &[usize],
    ) -> Result<Vec<Option<usize>>> {
        let mut syllable_vowels = Vec::new();

        for i in 0..boundaries.len() {
            let start = boundaries[i];
            let end = if i + 1 < boundaries.len() {
                boundaries[i + 1]
            } else {
                phonemes.len()
            };

            // Find the vowel in this syllable
            let vowel_pos = (start..end)
                .find(|&idx| idx < phonemes.len() && self.is_vowel_phoneme(&phonemes[idx].symbol));

            syllable_vowels.push(vowel_pos);
        }

        Ok(syllable_vowels)
    }

    /// Predict primary stress position for English words
    #[allow(dead_code)]
    fn predict_english_primary_stress(
        &self,
        word: &str,
        syllable_vowels: &[Option<usize>],
    ) -> Option<usize> {
        let _word_lower = word.to_lowercase();
        let syllable_count = syllable_vowels.len();

        // English stress rules
        if syllable_count == 1 {
            None // Monosyllabic
        } else if syllable_count == 2 {
            // Two-syllable words: typically first syllable for nouns, second for verbs
            // Since we don't have POS info, default to first syllable
            return Some(0);
        } else if syllable_count == 3 {
            // Three-syllable words: antepenultimate stress is common
            return Some(0);
        } else {
            // Longer words: usually second or third syllable from beginning
            return Some(1);
        }
    }

    /// Predict secondary stress position for English words
    #[allow(dead_code)]
    fn predict_english_secondary_stress(
        &self,
        _word: &str,
        syllable_vowels: &[Option<usize>],
        primary: Option<usize>,
    ) -> Option<usize> {
        let syllable_count = syllable_vowels.len();

        if syllable_count <= 3 {
            return None; // Too short for secondary stress
        }

        // Place secondary stress away from primary stress
        match primary {
            Some(0) => Some(syllable_count - 2), // Primary on first, secondary on penultimate
            Some(primary_pos) if primary_pos < syllable_count / 2 => Some(syllable_count - 1), // Primary early, secondary late
            _ => Some(0), // Primary late, secondary early
        }
    }

    /// Apply German stress rules (simplified)
    #[allow(dead_code)]
    fn apply_german_stress_rules(
        &self,
        phonemes: &mut [crate::Phoneme],
        _word: &str,
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        // German typically stresses the first syllable
        self.apply_first_syllable_stress(phonemes, syllable_boundaries, stress_config)
    }

    /// Apply French stress rules (simplified)
    #[allow(dead_code)]
    fn apply_french_stress_rules(
        &self,
        phonemes: &mut [crate::Phoneme],
        _word: &str,
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        // French typically stresses the last syllable
        self.apply_last_syllable_stress(phonemes, syllable_boundaries, stress_config)
    }

    /// Apply Spanish stress rules (simplified)
    #[allow(dead_code)]
    fn apply_spanish_stress_rules(
        &self,
        phonemes: &mut [crate::Phoneme],
        word: &str,
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        // Spanish stress depends on ending: penultimate for vowel/n/s, ultimate otherwise
        let ends_with_vowel_n_s = word
            .to_lowercase()
            .chars()
            .last()
            .map(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'n' | 's'))
            .unwrap_or(false);

        if ends_with_vowel_n_s {
            self.apply_penultimate_syllable_stress(phonemes, syllable_boundaries, stress_config)
        } else {
            self.apply_last_syllable_stress(phonemes, syllable_boundaries, stress_config)
        }
    }

    /// Apply default stress rules for unknown languages
    #[allow(dead_code)]
    fn apply_default_stress_rules(
        &self,
        phonemes: &mut [crate::Phoneme],
        _word: &str,
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        // Default to first syllable stress
        self.apply_first_syllable_stress(phonemes, syllable_boundaries, stress_config)
    }

    /// Apply stress to first syllable
    #[allow(dead_code)]
    fn apply_first_syllable_stress(
        &self,
        phonemes: &mut [crate::Phoneme],
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        if syllable_boundaries.is_empty() {
            return Ok(());
        }

        let syllable_vowels = self.find_syllable_vowels(phonemes, syllable_boundaries)?;
        if let Some(Some(vowel_idx)) = syllable_vowels.first() {
            let features = phonemes[*vowel_idx]
                .features
                .get_or_insert_with(HashMap::new);
            features.insert(
                "stress".to_string(),
                stress_config.primary_stress_marker.clone(),
            );
        }
        Ok(())
    }

    /// Apply stress to last syllable
    #[allow(dead_code)]
    fn apply_last_syllable_stress(
        &self,
        phonemes: &mut [crate::Phoneme],
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        if syllable_boundaries.is_empty() {
            return Ok(());
        }

        let syllable_vowels = self.find_syllable_vowels(phonemes, syllable_boundaries)?;
        if let Some(Some(vowel_idx)) = syllable_vowels.last() {
            let features = phonemes[*vowel_idx]
                .features
                .get_or_insert_with(HashMap::new);
            features.insert(
                "stress".to_string(),
                stress_config.primary_stress_marker.clone(),
            );
        }
        Ok(())
    }

    /// Apply stress to penultimate (second-to-last) syllable
    #[allow(dead_code)]
    fn apply_penultimate_syllable_stress(
        &self,
        phonemes: &mut [crate::Phoneme],
        syllable_boundaries: &[usize],
        stress_config: &StressConfig,
    ) -> Result<()> {
        let syllable_vowels = self.find_syllable_vowels(phonemes, syllable_boundaries)?;
        if syllable_vowels.len() >= 2 {
            let penultimate_idx = syllable_vowels.len() - 2;
            if let Some(Some(vowel_idx)) = syllable_vowels.get(penultimate_idx) {
                let features = phonemes[*vowel_idx]
                    .features
                    .get_or_insert_with(HashMap::new);
                features.insert(
                    "stress".to_string(),
                    stress_config.primary_stress_marker.clone(),
                );
            }
        }
        Ok(())
    }

    /// Post-process phoneme sequence
    fn post_process_phonemes(
        &self,
        phonemes: &mut Vec<crate::Phoneme>,
        g2p_config: &G2pConfig,
    ) -> Result<()> {
        debug!("Post-processing {} phonemes", phonemes.len());

        // 1. Apply phonological rules
        self.apply_phonological_rules(phonemes, g2p_config)?;

        // 2. Apply coarticulation effects
        self.apply_coarticulation_effects(&mut phonemes[..])?;

        // 3. Normalize phoneme durations
        self.normalize_phoneme_durations(&mut phonemes[..], g2p_config)?;

        // 4. Integrate prosody features
        self.integrate_prosody_features(&mut phonemes[..], g2p_config)?;

        // 5. Apply final utility processing
        *phonemes =
            crate::utils::process_phoneme_sequence(phonemes, &crate::SynthesisConfig::default());

        debug!("Post-processing completed for {} phonemes", phonemes.len());
        Ok(())
    }

    /// Apply phonological rules to phoneme sequence
    #[allow(dead_code)]
    fn apply_phonological_rules(
        &self,
        phonemes: &mut Vec<crate::Phoneme>,
        g2p_config: &G2pConfig,
    ) -> Result<()> {
        debug!("Applying phonological rules to {} phonemes", phonemes.len());

        // Apply common phonological rules
        self.apply_assimilation_rules(phonemes)?;
        self.apply_deletion_rules(phonemes)?;
        self.apply_insertion_rules(phonemes)?;
        self.apply_substitution_rules(phonemes)?;

        // Apply language-specific rules based on configuration
        for language in g2p_config.phoneme_sets.keys() {
            match language {
                LanguageCode::EnUs | LanguageCode::EnGb => {
                    self.apply_english_phonological_rules(phonemes)?;
                }
                LanguageCode::DeDe => {
                    self.apply_german_phonological_rules(phonemes)?;
                }
                LanguageCode::FrFr => {
                    self.apply_french_phonological_rules(phonemes)?;
                }
                LanguageCode::EsEs => {
                    self.apply_spanish_phonological_rules(phonemes)?;
                }
                _ => {
                    // Apply default rules for other languages
                    self.apply_default_phonological_rules(phonemes)?;
                }
            }
        }

        Ok(())
    }

    /// Apply coarticulation effects between adjacent phonemes
    #[allow(dead_code)]
    fn apply_coarticulation_effects(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        debug!(
            "Applying coarticulation effects to {} phonemes",
            phonemes.len()
        );

        // Use indices to avoid borrow checker issues
        let len = phonemes.len();
        for i in 0..len {
            // Forward coarticulation (current phoneme affects next)
            if i < len - 1 {
                let (left, right) = phonemes.split_at_mut(i + 1);
                self.apply_forward_coarticulation(&mut left[i], &right[0])?;
            }

            // Backward coarticulation (next phoneme affects current)
            if i > 0 {
                let (left, right) = phonemes.split_at_mut(i);
                self.apply_backward_coarticulation(&mut right[0], &left[i - 1])?;
            }

            // Contextual coarticulation (consider both neighbors)
            if i > 0 && i < len - 1 {
                let prev_symbol = phonemes[i - 1].symbol.clone();
                let next_symbol = phonemes[i + 1].symbol.clone();

                // Create temporary phonemes for the method call
                let prev_phoneme = crate::Phoneme {
                    symbol: prev_symbol,
                    features: phonemes[i - 1].features.clone(),
                    duration: phonemes[i - 1].duration,
                };
                let next_phoneme = crate::Phoneme {
                    symbol: next_symbol,
                    features: phonemes[i + 1].features.clone(),
                    duration: phonemes[i + 1].duration,
                };

                self.apply_contextual_coarticulation(
                    &mut phonemes[i],
                    &prev_phoneme,
                    &next_phoneme,
                )?;
            }
        }

        Ok(())
    }

    /// Normalize phoneme durations based on context
    #[allow(dead_code)]
    fn normalize_phoneme_durations(
        &self,
        phonemes: &mut [crate::Phoneme],
        g2p_config: &G2pConfig,
    ) -> Result<()> {
        debug!("Normalizing durations for {} phonemes", phonemes.len());

        // Calculate base durations
        let base_durations = self.calculate_base_durations(phonemes)?;

        // Collect context data before mutable iteration
        let phonemes_len = phonemes.len();
        let mut context_data = Vec::new();

        for i in 0..phonemes_len {
            let prev_symbol = if i > 0 {
                Some(phonemes[i - 1].symbol.clone())
            } else {
                None
            };
            let next_symbol = if i < phonemes_len - 1 {
                Some(phonemes[i + 1].symbol.clone())
            } else {
                None
            };
            context_data.push((prev_symbol, next_symbol));
        }

        // Apply duration modifications based on context
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let mut duration = base_durations[i];

            // Apply stress-based duration modification
            if let Some(features) = &phoneme.features {
                if let Some(stress) = features.get("stress") {
                    if stress == &g2p_config.stress_config.primary_stress_marker {
                        duration *= 1.2; // Stressed vowels are longer
                    } else if stress == &g2p_config.stress_config.secondary_stress_marker {
                        duration *= 1.1; // Secondary stress slightly longer
                    }
                }
            }

            // Apply position-based duration modification
            if i == 0 {
                duration *= 1.1; // Initial phonemes slightly longer
            } else if i == phonemes_len - 1 {
                duration *= 1.3; // Final phonemes significantly longer
            }

            // Apply phoneme-class based duration modification
            if self.is_vowel_phoneme(&phoneme.symbol) {
                duration *= 1.2; // Vowels are generally longer
            } else if self.is_stop_consonant(&phoneme.symbol) {
                duration *= 0.8; // Stop consonants are shorter
            } else if self.is_fricative(&phoneme.symbol) {
                duration *= 1.1; // Fricatives are longer
            }

            // Apply contextual duration modification
            if let (Some(prev_symbol), Some(next_symbol)) = &context_data[i] {
                let prev_is_vowel = self.is_vowel_phoneme(prev_symbol);
                let next_is_vowel = self.is_vowel_phoneme(next_symbol);

                if prev_is_vowel && next_is_vowel {
                    duration *= 0.9; // Consonants between vowels are shorter
                } else if !prev_is_vowel && !next_is_vowel {
                    duration *= 1.1; // Consonants in clusters are longer
                }
            }

            phoneme.duration = Some(duration);
        }

        Ok(())
    }

    /// Integrate prosody features into phoneme sequence
    #[allow(dead_code)]
    fn integrate_prosody_features(
        &self,
        phonemes: &mut [crate::Phoneme],
        g2p_config: &G2pConfig,
    ) -> Result<()> {
        debug!(
            "Integrating prosody features for {} phonemes",
            phonemes.len()
        );

        // Apply intonation patterns
        self.apply_intonation_patterns(phonemes)?;

        // Apply rhythm adjustments
        self.apply_rhythm_adjustments(phonemes)?;

        // Apply emphasis and focus
        self.apply_emphasis_features(phonemes)?;

        // Apply boundary tones
        self.apply_boundary_tones(phonemes)?;

        // Apply register and voice quality features
        self.apply_voice_quality_features(phonemes, g2p_config)?;

        Ok(())
    }

    /// Check if a phoneme symbol represents a vowel
    #[allow(dead_code)]
    fn is_vowel_phoneme(&self, symbol: &str) -> bool {
        matches!(
            symbol,
            "AA" | "AE"
                | "AH"
                | "AO"
                | "AW"
                | "AY"
                | "EH"
                | "ER"
                | "EY"
                | "IH"
                | "IY"
                | "OW"
                | "OY"
                | "UH"
                | "UW"
        )
    }

    /// Get pipeline information
    pub fn info(&self) -> &PipelineConfig {
        &self.pipeline_config
    }

    /// Get model information
    pub fn model_info(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Check if pipeline supports given language
    pub fn supports_language(&self, language: LanguageCode) -> bool {
        self.model_config.supported_languages.contains(&language)
    }

    /// Apply assimilation rules to phonemes
    #[allow(dead_code)]
    fn apply_assimilation_rules(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Basic assimilation rules implementation
        for i in 0..phonemes.len().saturating_sub(1) {
            let current = &phonemes[i].symbol;
            let next = &phonemes[i + 1].symbol;

            // Nasal assimilation
            if current == "N" && (next == "P" || next == "B") {
                phonemes[i].symbol = "M".to_string();
            } else if current == "N" && (next == "K" || next == "G") {
                phonemes[i].symbol = "NG".to_string();
            }
        }
        Ok(())
    }

    /// Apply deletion rules to phonemes
    #[allow(dead_code)]
    fn apply_deletion_rules(&self, phonemes: &mut Vec<crate::Phoneme>) -> Result<()> {
        // Basic deletion rules implementation
        phonemes.retain(|phoneme| {
            // Remove very short schwas in unstressed positions
            !(phoneme.symbol == "AH" && phoneme.duration.is_some_and(|d| d < 0.05))
        });
        Ok(())
    }

    /// Apply insertion rules to phonemes
    #[allow(dead_code)]
    fn apply_insertion_rules(&self, phonemes: &mut Vec<crate::Phoneme>) -> Result<()> {
        // Basic insertion rules implementation
        let mut insertions = Vec::new();

        for i in 0..phonemes.len().saturating_sub(1) {
            let current = &phonemes[i].symbol;
            let next = &phonemes[i + 1].symbol;

            // Epenthetic vowel insertion
            if current == "S" && (next == "T" || next == "P") {
                insertions.push((
                    i + 1,
                    crate::Phoneme {
                        symbol: "AH".to_string(),
                        duration: Some(0.03),
                        features: None,
                    },
                ));
            }
        }

        for (offset, (index, phoneme)) in insertions.into_iter().enumerate() {
            phonemes.insert(index + offset, phoneme);
        }
        Ok(())
    }

    /// Apply substitution rules to phonemes
    #[allow(dead_code)]
    fn apply_substitution_rules(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Basic substitution rules implementation
        for phoneme in phonemes.iter_mut() {
            match phoneme.symbol.as_str() {
                "AA" if phoneme.duration.is_some_and(|d| d < 0.1) => {
                    phoneme.symbol = "AH".to_string()
                }
                "IY" if phoneme.duration.is_some_and(|d| d < 0.08) => {
                    phoneme.symbol = "IH".to_string()
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Apply English-specific phonological rules
    #[allow(dead_code)]
    fn apply_english_phonological_rules(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // English-specific rules
        for i in 0..phonemes.len().saturating_sub(1) {
            let current = &phonemes[i].symbol;
            let next = &phonemes[i + 1].symbol;

            // English flapping rule
            if current == "T" && self.is_vowel_phoneme(next) {
                phonemes[i].symbol = "DX".to_string();
            }
        }
        Ok(())
    }

    /// Apply German-specific phonological rules
    #[allow(dead_code)]
    fn apply_german_phonological_rules(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // German-specific rules (placeholder)
        for phoneme in phonemes.iter_mut() {
            if phoneme.symbol == "G" {
                phoneme.symbol = "X".to_string(); // Final devoicing
            }
        }
        Ok(())
    }

    /// Apply French-specific phonological rules
    #[allow(dead_code)]
    fn apply_french_phonological_rules(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // French-specific rules (placeholder)
        for phoneme in phonemes.iter_mut() {
            if phoneme.symbol == "R" {
                phoneme.symbol = "GH".to_string(); // Uvular R
            }
        }
        Ok(())
    }

    /// Apply Spanish-specific phonological rules
    #[allow(dead_code)]
    fn apply_spanish_phonological_rules(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Spanish-specific rules (placeholder)
        for phoneme in phonemes.iter_mut() {
            if phoneme.symbol == "B" {
                phoneme.symbol = "V".to_string(); // Spirantization
            }
        }
        Ok(())
    }

    /// Apply default phonological rules
    #[allow(dead_code)]
    fn apply_default_phonological_rules(&self, phonemes: &mut Vec<crate::Phoneme>) -> Result<()> {
        // Default rules for unsupported languages
        self.apply_assimilation_rules(phonemes)?;
        self.apply_deletion_rules(phonemes)?;
        Ok(())
    }

    /// Apply forward coarticulation effects
    #[allow(dead_code)]
    fn apply_forward_coarticulation(
        &self,
        current: &mut crate::Phoneme,
        next: &crate::Phoneme,
    ) -> Result<()> {
        // Anticipatory coarticulation
        if self.is_vowel_phoneme(&current.symbol) && next.symbol == "R" {
            // Store coarticulation effect in features
            let mut features = current.features.clone().unwrap_or_default();
            features.insert("f3_lowering".to_string(), "200".to_string());
            current.features = Some(features);
        }
        Ok(())
    }

    /// Apply backward coarticulation effects
    #[allow(dead_code)]
    fn apply_backward_coarticulation(
        &self,
        current: &mut crate::Phoneme,
        previous: &crate::Phoneme,
    ) -> Result<()> {
        // Carryover coarticulation
        if self.is_vowel_phoneme(&current.symbol) && previous.symbol == "R" {
            // Store coarticulation effect in features
            let mut features = current.features.clone().unwrap_or_default();
            features.insert("f3_lowering".to_string(), "100".to_string());
            current.features = Some(features);
        }
        Ok(())
    }

    /// Apply contextual coarticulation effects
    #[allow(dead_code)]
    fn apply_contextual_coarticulation(
        &self,
        current: &mut crate::Phoneme,
        previous: &crate::Phoneme,
        next: &crate::Phoneme,
    ) -> Result<()> {
        // Contextual effects
        if self.is_vowel_phoneme(&current.symbol) && previous.symbol == "R" && next.symbol == "R" {
            // Store coarticulation effect in features
            let mut features = current.features.clone().unwrap_or_default();
            features.insert("f3_lowering".to_string(), "300".to_string());
            current.features = Some(features);
        }
        Ok(())
    }

    /// Calculate base durations for phonemes
    #[allow(dead_code)]
    fn calculate_base_durations(&self, phonemes: &[crate::Phoneme]) -> Result<Vec<f32>> {
        let mut durations = Vec::new();

        for phoneme in phonemes {
            let base_duration = if self.is_vowel_phoneme(&phoneme.symbol) {
                0.12 // Base vowel duration
            } else {
                0.08 // Base consonant duration
            };
            durations.push(base_duration);
        }

        Ok(durations)
    }

    /// Check if phoneme is a stop consonant
    #[allow(dead_code)]
    fn is_stop_consonant(&self, symbol: &str) -> bool {
        matches!(symbol, "P" | "B" | "T" | "D" | "K" | "G")
    }

    /// Check if phoneme is a fricative
    #[allow(dead_code)]
    fn is_fricative(&self, symbol: &str) -> bool {
        matches!(
            symbol,
            "F" | "V" | "TH" | "DH" | "S" | "Z" | "SH" | "ZH" | "HH"
        )
    }

    /// Apply intonation patterns
    #[allow(dead_code)]
    fn apply_intonation_patterns(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Basic intonation patterns
        let phonemes_len = phonemes.len();
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            if self.is_vowel_phoneme(&phoneme.symbol) {
                let position = i as f32 / phonemes_len as f32;
                let mut features = phoneme.features.clone().unwrap_or_default();
                if position < 0.5 {
                    features.insert("pitch_mult".to_string(), "1.1".to_string());
                // Rising intonation
                } else {
                    features.insert("pitch_mult".to_string(), "0.9".to_string());
                    // Falling intonation
                }
                phoneme.features = Some(features);
            }
        }
        Ok(())
    }

    /// Apply rhythm adjustments
    #[allow(dead_code)]
    fn apply_rhythm_adjustments(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Basic rhythm adjustments
        for phoneme in phonemes.iter_mut() {
            if self.is_vowel_phoneme(&phoneme.symbol) {
                if let Some(duration) = phoneme.duration {
                    phoneme.duration = Some(duration * 1.2); // Lengthen vowels
                }
            }
        }
        Ok(())
    }

    /// Apply emphasis features
    #[allow(dead_code)]
    fn apply_emphasis_features(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Basic emphasis features
        for phoneme in phonemes.iter_mut() {
            let mut features = phoneme.features.clone().unwrap_or_default();
            if let Some(energy_str) = features.get("energy") {
                if let Ok(energy) = energy_str.parse::<f32>() {
                    if energy > 0.8 {
                        features.insert("pitch_mult".to_string(), "1.2".to_string());
                        if let Some(duration) = phoneme.duration {
                            phoneme.duration = Some(duration * 1.1);
                        }
                    }
                }
            }
            phoneme.features = Some(features);
        }
        Ok(())
    }

    /// Apply boundary tones
    #[allow(dead_code)]
    fn apply_boundary_tones(&self, phonemes: &mut [crate::Phoneme]) -> Result<()> {
        // Apply boundary tones at phrase boundaries
        if let Some(first) = phonemes.first_mut() {
            if self.is_vowel_phoneme(&first.symbol) {
                let mut features = first.features.clone().unwrap_or_default();
                features.insert("boundary_tone".to_string(), "high".to_string());
                first.features = Some(features);
            }
        }
        if let Some(last) = phonemes.last_mut() {
            if self.is_vowel_phoneme(&last.symbol) {
                let mut features = last.features.clone().unwrap_or_default();
                features.insert("boundary_tone".to_string(), "low".to_string());
                last.features = Some(features);
            }
        }
        Ok(())
    }

    /// Apply voice quality features
    #[allow(dead_code)]
    fn apply_voice_quality_features(
        &self,
        phonemes: &mut [crate::Phoneme],
        _g2p_config: &G2pConfig,
    ) -> Result<()> {
        // Basic voice quality adjustments
        for phoneme in phonemes.iter_mut() {
            if self.is_vowel_phoneme(&phoneme.symbol) {
                let mut features = phoneme.features.clone().unwrap_or_default();
                features.insert("energy_mult".to_string(), "0.95".to_string());
                phoneme.features = Some(features);
            }
        }
        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of loaded models
    pub loaded_models: usize,
    /// Estimated cache size in MB
    pub cache_size_mb: usize,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            pipelines: HashMap::new(),
            metadata: RegistryMetadata {
                version: "1.0.0".to_string(),
                last_updated: chrono::Utc::now().to_rfc3339(),
                description: "VoiRS Neural TTS Model Registry".to_string(),
            },
        }
    }
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            real_time_factor: 0.3,
            quality_level: 0.8,
            memory_usage: 0.5,
            supports_streaming: true,
            g2p_config: G2pConfig::default(),
        }
    }
}

impl Default for G2pConfig {
    fn default() -> Self {
        let mut phoneme_sets = HashMap::new();

        // English phoneme set (ARPAbet-style)
        phoneme_sets.insert(
            LanguageCode::EnUs,
            PhonemeSet {
                symbols: vec![
                    // Vowels
                    "AA".to_string(),
                    "AE".to_string(),
                    "AH".to_string(),
                    "AO".to_string(),
                    "AW".to_string(),
                    "AY".to_string(),
                    "EH".to_string(),
                    "ER".to_string(),
                    "EY".to_string(),
                    "IH".to_string(),
                    "IY".to_string(),
                    "OW".to_string(),
                    "OY".to_string(),
                    "UH".to_string(),
                    "UW".to_string(),
                    // Consonants
                    "B".to_string(),
                    "CH".to_string(),
                    "D".to_string(),
                    "DH".to_string(),
                    "F".to_string(),
                    "G".to_string(),
                    "HH".to_string(),
                    "JH".to_string(),
                    "K".to_string(),
                    "L".to_string(),
                    "M".to_string(),
                    "N".to_string(),
                    "NG".to_string(),
                    "P".to_string(),
                    "R".to_string(),
                    "S".to_string(),
                    "SH".to_string(),
                    "T".to_string(),
                    "TH".to_string(),
                    "V".to_string(),
                    "W".to_string(),
                    "Y".to_string(),
                    "Z".to_string(),
                    "ZH".to_string(),
                ],
                vowels: vec![
                    "AA".to_string(),
                    "AE".to_string(),
                    "AH".to_string(),
                    "AO".to_string(),
                    "AW".to_string(),
                    "AY".to_string(),
                    "EH".to_string(),
                    "ER".to_string(),
                    "EY".to_string(),
                    "IH".to_string(),
                    "IY".to_string(),
                    "OW".to_string(),
                    "OY".to_string(),
                    "UH".to_string(),
                    "UW".to_string(),
                ],
                consonants: vec![
                    "B".to_string(),
                    "CH".to_string(),
                    "D".to_string(),
                    "DH".to_string(),
                    "F".to_string(),
                    "G".to_string(),
                    "HH".to_string(),
                    "JH".to_string(),
                    "K".to_string(),
                    "L".to_string(),
                    "M".to_string(),
                    "N".to_string(),
                    "NG".to_string(),
                    "P".to_string(),
                    "R".to_string(),
                    "S".to_string(),
                    "SH".to_string(),
                    "T".to_string(),
                    "TH".to_string(),
                    "V".to_string(),
                    "W".to_string(),
                    "Y".to_string(),
                    "Z".to_string(),
                    "ZH".to_string(),
                ],
                special_symbols: vec![
                    "_".to_string(),     // Silence
                    " ".to_string(),     // Word boundary
                    "<pad>".to_string(), // Padding
                    "<unk>".to_string(), // Unknown
                    "<bos>".to_string(), // Beginning of sequence
                    "<eos>".to_string(), // End of sequence
                ],
                stress_markers: vec![
                    "0".to_string(), // Unstressed
                    "1".to_string(), // Primary stress
                    "2".to_string(), // Secondary stress
                ],
            },
        );

        let mut dictionaries = HashMap::new();
        dictionaries.insert(LanguageCode::EnUs, "cmudict.dict".to_string());

        Self {
            engine: crate::config::G2pEngine::Hybrid,
            phoneme_sets,
            dictionaries,
            stress_config: StressConfig::default(),
            unknown_word_strategy: UnknownWordStrategy::FallbackRules,
            variant_preferences: VariantPreferences::default(),
        }
    }
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            predict_stress: true,
            primary_stress_marker: "1".to_string(),
            secondary_stress_marker: "2".to_string(),
            confidence_threshold: 0.7,
        }
    }
}

impl Default for VariantPreferences {
    fn default() -> Self {
        Self {
            accent: "general_american".to_string(),
            formality: FormalityLevel::Standard,
            speed_optimized: false,
            custom_pronunciations: HashMap::new(),
        }
    }
}

// Conversion implementations for TOML parsing
impl TryFrom<toml::Value> for PipelineConfig {
    type Error = AcousticError;

    fn try_from(value: toml::Value) -> Result<Self> {
        let table = value.as_table().ok_or_else(|| {
            AcousticError::ConfigError("Expected table for pipeline config".to_string())
        })?;

        let name = table
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unnamed Pipeline")
            .to_string();

        let description = table
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("No description")
            .to_string();

        let acoustic_model = table
            .get("acoustic_model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AcousticError::ConfigError("Missing acoustic_model".to_string()))?
            .to_string();

        let vocoder = table
            .get("vocoder")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AcousticError::ConfigError("Missing vocoder".to_string()))?
            .to_string();

        Ok(Self {
            name,
            description,
            acoustic_model,
            vocoder,
            settings: PipelineSettings::default(),
        })
    }
}

impl PronunciationDictionary {
    /// Create a new empty pronunciation dictionary
    pub fn new(language: LanguageCode, name: String) -> Self {
        Self {
            entries: HashMap::new(),
            case_insensitive_map: HashMap::new(),
            language,
            metadata: DictionaryMetadata {
                name,
                version: "1.0.0".to_string(),
                entry_count: 0,
                source: "VoiRS Dictionary".to_string(),
                license: "MIT".to_string(),
            },
        }
    }

    /// Load pronunciation dictionary from a file
    /// Supports CMU Pronouncing Dictionary format and other common formats
    pub fn load_from_file<P: AsRef<Path>>(path: P, language: LanguageCode) -> Result<Self> {
        let file = fs::File::open(&path).map_err(|e| {
            AcousticError::ConfigError(format!("Failed to open dictionary file: {e}"))
        })?;
        let reader = BufReader::new(file);

        let mut dictionary = Self::new(
            language,
            path.as_ref()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dictionary")
                .to_string(),
        );

        let mut entry_count = 0;
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                AcousticError::ConfigError(format!("Failed to read line {}: {e}", line_no + 1))
            })?;

            // Skip comments and empty lines
            if line.trim().is_empty() || line.starts_with(";;;") {
                continue;
            }

            // Parse line in CMU format: WORD  phonemes
            if let Some((word, pronunciation)) = dictionary.parse_cmu_line(&line) {
                dictionary.add_entry(word, pronunciation, 1.0)?;
                entry_count += 1;
            }
        }

        dictionary.metadata.entry_count = entry_count;
        info!(
            "Loaded {} entries from pronunciation dictionary: {}",
            entry_count, dictionary.metadata.name
        );

        Ok(dictionary)
    }

    /// Parse CMU Pronouncing Dictionary format line
    fn parse_cmu_line(&self, line: &str) -> Option<(String, String)> {
        // CMU format: WORD(VARIANT)  phonemes
        // Example: HELLO  HH EH1 L OW0
        // Example: HELLO(2)  HH EH0 L OW1

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return None;
        }

        let word_part = parts[0];
        let pronunciation = parts[1..].join(" ");

        // Handle variants like HELLO(2)
        let word = if let Some(paren_pos) = word_part.find('(') {
            word_part[..paren_pos].to_lowercase()
        } else {
            word_part.to_lowercase()
        };

        Some((word, pronunciation))
    }

    /// Add a pronunciation entry to the dictionary
    pub fn add_entry(
        &mut self,
        word: String,
        pronunciation: String,
        confidence: f32,
    ) -> Result<()> {
        let word_lower = word.to_lowercase();

        if let Some(existing) = self.entries.get_mut(&word_lower) {
            // Add as alternative pronunciation
            if !existing.alternatives.contains(&pronunciation) {
                existing.alternatives.push(pronunciation);
            }
        } else {
            // Create new entry
            let entry = PronunciationEntry {
                word: word.clone(),
                primary: pronunciation,
                alternatives: Vec::new(),
                variants: HashMap::new(),
                confidence,
                pos_tag: None,
            };

            self.entries.insert(word_lower.clone(), entry);
            self.case_insensitive_map
                .insert(word_lower.clone(), word_lower);
        }

        Ok(())
    }

    /// Look up pronunciation for a word
    pub fn lookup(&self, word: &str) -> Option<&PronunciationEntry> {
        let word_lower = word.to_lowercase();
        self.entries.get(&word_lower)
    }

    /// Look up pronunciation with variant preferences
    pub fn lookup_with_preferences(
        &self,
        word: &str,
        preferences: &VariantPreferences,
    ) -> Option<String> {
        let entry = self.lookup(word)?;

        // Check custom pronunciations first
        if let Some(custom) = preferences.custom_pronunciations.get(word) {
            return Some(custom.clone());
        }

        // Check accent-specific variants
        if let Some(variant) = entry.variants.get(&preferences.accent) {
            return Some(variant.clone());
        }

        // Check formality-specific variants
        let formality_key = match preferences.formality {
            FormalityLevel::Casual => "casual",
            FormalityLevel::Standard => "standard",
            FormalityLevel::Formal => "formal",
        };

        if let Some(variant) = entry.variants.get(formality_key) {
            return Some(variant.clone());
        }

        // Return primary pronunciation
        Some(entry.primary.clone())
    }

    /// Get multiple pronunciations for a word (primary + alternatives)
    pub fn get_all_pronunciations(&self, word: &str) -> Vec<String> {
        if let Some(entry) = self.lookup(word) {
            let mut pronunciations = vec![entry.primary.clone()];
            pronunciations.extend(entry.alternatives.clone());
            pronunciations
        } else {
            Vec::new()
        }
    }

    /// Check if a word exists in the dictionary
    pub fn contains(&self, word: &str) -> bool {
        self.lookup(word).is_some()
    }

    /// Get dictionary statistics
    pub fn get_stats(&self) -> (usize, f32) {
        let total_entries = self.entries.len();
        let avg_confidence = if total_entries > 0 {
            self.entries.values().map(|e| e.confidence).sum::<f32>() / total_entries as f32
        } else {
            0.0
        };
        (total_entries, avg_confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::default();
        assert!(registry.models.is_empty());
        assert!(registry.pipelines.is_empty());
        assert_eq!(registry.metadata.version, "1.0.0");
    }

    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new().await;
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(manager.registry.models.is_empty());
    }

    #[test]
    fn test_pipeline_settings_default() {
        let settings = PipelineSettings::default();
        assert_eq!(settings.real_time_factor, 0.3);
        assert_eq!(settings.quality_level, 0.8);
        assert!(settings.supports_streaming);
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            loaded_models: 5,
            cache_size_mb: 512,
        };

        assert_eq!(stats.loaded_models, 5);
        assert_eq!(stats.cache_size_mb, 512);
    }
}
