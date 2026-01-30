//! Type definitions for the model manager
//!
//! This module contains all struct definitions and implementations except
//! for TtsPipeline impl methods which are split into separate modules.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::backends::{create_default_loader, AcousticModelLoader};
use crate::{AcousticError, AcousticModel, LanguageCode, ModelArchitecture, ModelConfig, Result};

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
        let file = fs::File::open(&path).map_err(|e| AcousticError::ConfigError {
            message: format!("Failed to open dictionary file: {e}"),
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
            let line = line.map_err(|e| AcousticError::ConfigError {
                message: format!("Failed to read line {}: {e}", line_no + 1),
            })?;
            if line.trim().is_empty() || line.starts_with(";;;") {
                continue;
            }
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
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return None;
        }
        let word_part = parts[0];
        let pronunciation = parts[1..].join(" ");
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
            if !existing.alternatives.contains(&pronunciation) {
                existing.alternatives.push(pronunciation);
            }
        } else {
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
        if let Some(custom) = preferences.custom_pronunciations.get(word) {
            return Some(custom.clone());
        }
        if let Some(variant) = entry.variants.get(&preferences.accent) {
            return Some(variant.clone());
        }
        let formality_key = match preferences.formality {
            FormalityLevel::Casual => "casual",
            FormalityLevel::Standard => "standard",
            FormalityLevel::Formal => "formal",
        };
        if let Some(variant) = entry.variants.get(formality_key) {
            return Some(variant.clone());
        }
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
/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of loaded models
    pub loaded_models: usize,
    /// Estimated cache size in MB
    pub cache_size_mb: usize,
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
        std::fs::create_dir_all(&cache_dir).map_err(|e| AcousticError::ConfigError {
            message: format!("Failed to create cache dir: {e}"),
        })?;
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
        let config_content = std::fs::read_to_string(config_path.as_ref()).map_err(|e| {
            AcousticError::ConfigError {
                message: format!("Failed to read config: {e}"),
            }
        })?;
        let config: toml::Value =
            toml::from_str(&config_content).map_err(|e| AcousticError::ConfigError {
                message: format!("Failed to parse TOML: {e}"),
            })?;
        let mut manager = Self::new().await?;
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
    /// Get reference to the model registry
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }
    /// Get model by ID, loading if necessary
    pub async fn get_model(&mut self, model_id: &str) -> Result<Arc<dyn AcousticModel>> {
        {
            let loaded = self.loaded_models.read().await;
            if let Some(model) = loaded.get(model_id) {
                debug!("Using cached model: {}", model_id);
                return Ok(model.clone());
            }
        }
        let model_config =
            self.registry
                .models
                .get(model_id)
                .ok_or_else(|| AcousticError::ModelError {
                    message: format!("Model not found: {model_id}"),
                })?;
        info!(
            "Loading model: {} ({})",
            model_id, model_config.metadata.name
        );
        let model = self.loader.load(&model_config.model_path).await?;
        {
            let mut loaded = self.loaded_models.write().await;
            loaded.insert(model_id.to_string(), model.clone());
        }
        info!("Successfully loaded and cached model: {}", model_id);
        Ok(model)
    }
    /// Load a complete pipeline (acoustic model + vocoder)
    pub async fn load_pipeline(&mut self, pipeline_id: &str) -> Result<TtsPipeline> {
        let pipeline_config = self
            .registry
            .pipelines
            .get(pipeline_id)
            .ok_or_else(|| AcousticError::ConfigError {
                message: format!("Pipeline not found: {pipeline_id}"),
            })?
            .clone();
        info!(
            "Loading TTS pipeline: {} ({})",
            pipeline_id, pipeline_config.name
        );
        let acoustic_model = self.get_model(&pipeline_config.acoustic_model).await?;
        let model_config = self
            .registry
            .models
            .get(&pipeline_config.acoustic_model)
            .ok_or_else(|| AcousticError::ConfigError {
                message: format!("Model config not found: {}", pipeline_config.acoustic_model),
            })?
            .clone();
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
        let model_config =
            self.registry
                .models
                .get(model_id)
                .ok_or_else(|| AcousticError::ModelError {
                    message: format!("Model not found: {model_id}"),
                })?;
        let model_path = &model_config.model_path;
        if !model_path.starts_with("http") && !model_path.contains('/') {
            return Ok(self.cache_dir.join(model_path));
        }
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
            cache_size_mb: 0,
        }
    }
    /// Helper method to check if a character is a vowel
    pub(crate) fn is_vowel(ch: Option<char>) -> bool {
        match ch {
            Some(c) => matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y'),
            None => false,
        }
    }
    /// Static version of apply_stress_rules for use in TtsPipeline
    pub(crate) fn apply_stress_rules_static(phonemes: &mut [crate::Phoneme]) {
        use std::collections::HashMap;
        if phonemes.is_empty() {
            return;
        }
        let len = phonemes.len();
        if len == 1 {
            let features = phonemes[0].features.get_or_insert_with(HashMap::new);
            features.insert("stress".to_string(), "1".to_string());
        } else if len == 2 {
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            let features1 = phonemes[1].features.get_or_insert_with(HashMap::new);
            features1.insert("stress".to_string(), "0".to_string());
        } else if len <= 4 {
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            for phoneme in &mut phonemes[1..] {
                let features = phoneme.features.get_or_insert_with(HashMap::new);
                features.insert("stress".to_string(), "0".to_string());
            }
        } else {
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
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let base_duration = 0.08;
            let stress_value = phoneme
                .features
                .as_ref()
                .and_then(|f| f.get("stress"))
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(0);
            let stress_multiplier = match stress_value {
                1 => 1.3,
                0 => 0.9,
                _ => 1.0,
            };
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
        if len == 1 {
            let features = phonemes[0].features.get_or_insert_with(HashMap::new);
            features.insert("stress".to_string(), "1".to_string());
        } else if len == 2 {
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            let features1 = phonemes[1].features.get_or_insert_with(HashMap::new);
            features1.insert("stress".to_string(), "0".to_string());
        } else if len <= 4 {
            let features0 = phonemes[0].features.get_or_insert_with(HashMap::new);
            features0.insert("stress".to_string(), "1".to_string());
            for phoneme in &mut phonemes[1..] {
                let features = phoneme.features.get_or_insert_with(HashMap::new);
                features.insert("stress".to_string(), "0".to_string());
            }
        } else {
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
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let base_duration = 0.08;
            let stress_value = phoneme
                .features
                .as_ref()
                .and_then(|f| f.get("stress"))
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(0);
            let stress_multiplier = match stress_value {
                1 => 1.3,
                0 => 0.9,
                _ => 1.0,
            };
            let position_multiplier = if i == len - 1 { 1.2 } else { 1.0 };
            phoneme.duration = Some(base_duration * stress_multiplier * position_multiplier);
        }
    }
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
/// Complete TTS pipeline
pub struct TtsPipeline {
    /// Acoustic model
    pub(crate) acoustic_model: Arc<dyn AcousticModel>,
    /// Pipeline configuration
    pub(crate) pipeline_config: PipelineConfig,
    /// Model configuration
    pub(crate) model_config: ModelConfig,
    /// Pronunciation dictionaries for different languages
    pub(crate) pronunciation_dictionaries:
        Arc<RwLock<HashMap<LanguageCode, PronunciationDictionary>>>,
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
