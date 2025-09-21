//! Custom pronunciation dictionary system for SSML.

use crate::{G2pError, LanguageCode, Phoneme, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Custom pronunciation dictionary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationDictionary {
    /// Dictionary name
    pub name: String,
    /// Target language
    pub language: LanguageCode,
    /// Dictionary version
    pub version: String,
    /// Dictionary entries
    pub entries: HashMap<String, DictionaryEntry>,
    /// Metadata
    pub metadata: DictionaryMetadata,
}

/// Dictionary entry with pronunciation and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    /// Word or phrase
    pub word: String,
    /// Primary pronunciation
    pub pronunciation: Vec<Phoneme>,
    /// Alternative pronunciations
    pub alternatives: Vec<Vec<Phoneme>>,
    /// Context-specific pronunciations
    pub contexts: Vec<ContextualPronunciation>,
    /// Regional variants
    pub regional_variants: HashMap<String, Vec<Phoneme>>,
    /// Metadata
    pub metadata: EntryMetadata,
}

/// Contextual pronunciation for specific usage scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualPronunciation {
    /// Context description
    pub context: PronunciationContext,
    /// Phonemes for this context
    pub phonemes: Vec<Phoneme>,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
    /// Usage frequency
    pub frequency: Option<f32>,
}

/// Pronunciation context types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PronunciationContext {
    /// Beginning of sentence
    SentenceInitial,
    /// End of sentence
    SentenceFinal,
    /// Before specific phoneme
    BeforePhoneme(String),
    /// After specific phoneme
    AfterPhoneme(String),
    /// In specific grammatical position
    PartOfSpeech(PartOfSpeech),
    /// In compound words
    Compound,
    /// Stressed position
    Stressed,
    /// Unstressed position
    Unstressed,
    /// Formal speech style
    Formal,
    /// Casual speech style
    Casual,
    /// Fast speech
    FastSpeech,
    /// Slow speech
    SlowSpeech,
    /// Custom context
    Custom(String),
}

/// Part of speech for context-sensitive pronunciation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Preposition,
    Conjunction,
    Interjection,
    Determiner,
    Pronoun,
    /// Custom part of speech
    Custom(String),
}

/// Dictionary metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryMetadata {
    /// Creation date
    pub created: String,
    /// Last modified date
    pub modified: String,
    /// Author information
    pub author: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Source information
    pub source: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Dictionary statistics
    pub statistics: DictionaryStatistics,
}

/// Entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMetadata {
    /// Word frequency in corpus
    pub frequency: Option<f32>,
    /// Word difficulty level
    pub difficulty: Option<DifficultyLevel>,
    /// Phonetic tags
    pub tags: Vec<String>,
    /// Source information
    pub source: Option<String>,
    /// Confidence in pronunciation
    pub confidence: f32,
    /// Last verified date
    pub verified: Option<String>,
}

/// Dictionary statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DictionaryStatistics {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of entries with alternatives
    pub entries_with_alternatives: usize,
    /// Number of context-sensitive entries
    pub contextual_entries: usize,
    /// Number of regional variants
    pub regional_variants: usize,
    /// Coverage statistics
    pub coverage: Option<CoverageStatistics>,
}

/// Coverage statistics for the dictionary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageStatistics {
    /// Percentage of common words covered
    pub common_words_coverage: f32,
    /// Percentage of corpus covered
    pub corpus_coverage: f32,
    /// Domain-specific coverage
    pub domain_coverage: HashMap<String, f32>,
}

/// Word difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
    /// Custom difficulty (0.0-1.0)
    Custom(f32),
}

/// Dictionary manager for handling multiple dictionaries
pub struct DictionaryManager {
    /// Loaded dictionaries
    dictionaries: HashMap<String, PronunciationDictionary>,
    /// Dictionary priority order
    priority_order: Vec<String>,
    /// Cache for frequent lookups
    lookup_cache: HashMap<String, CachedLookup>,
    /// Cache size limit
    cache_limit: usize,
}

/// Cached lookup result
#[derive(Debug, Clone)]
struct CachedLookup {
    /// Pronunciation result
    pronunciation: Option<Vec<Phoneme>>,
    /// Context used for lookup
    _context: Option<PronunciationContext>,
    /// Dictionary source
    _source: String,
    /// Timestamp
    timestamp: std::time::SystemTime,
}

impl PronunciationDictionary {
    /// Create a new pronunciation dictionary
    pub fn new(name: String, language: LanguageCode) -> Self {
        Self {
            name,
            language,
            version: "1.0.0".to_string(),
            entries: HashMap::new(),
            metadata: DictionaryMetadata::default(),
        }
    }

    /// Add a word with pronunciation
    pub fn add_word(&mut self, word: String, pronunciation: Vec<Phoneme>) -> Result<()> {
        let entry = DictionaryEntry {
            word: word.clone(),
            pronunciation,
            alternatives: Vec::new(),
            contexts: Vec::new(),
            regional_variants: HashMap::new(),
            metadata: EntryMetadata::default(),
        };

        self.entries.insert(word.to_lowercase(), entry);
        self.metadata.statistics.total_entries = self.entries.len();
        Ok(())
    }

    /// Add alternative pronunciation for a word
    pub fn add_alternative(&mut self, word: &str, pronunciation: Vec<Phoneme>) -> Result<()> {
        let key = word.to_lowercase();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.alternatives.push(pronunciation);
            self.update_statistics();
            Ok(())
        } else {
            Err(G2pError::ConfigError(format!(
                "Word '{word}' not found in dictionary"
            )))
        }
    }

    /// Add contextual pronunciation
    pub fn add_contextual_pronunciation(
        &mut self,
        word: &str,
        context: PronunciationContext,
        pronunciation: Vec<Phoneme>,
        confidence: f32,
    ) -> Result<()> {
        let key = word.to_lowercase();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.contexts.push(ContextualPronunciation {
                context,
                phonemes: pronunciation,
                confidence,
                frequency: None,
            });
            self.update_statistics();
            Ok(())
        } else {
            Err(G2pError::ConfigError(format!(
                "Word '{word}' not found in dictionary"
            )))
        }
    }

    /// Add regional variant
    pub fn add_regional_variant(
        &mut self,
        word: &str,
        region: String,
        pronunciation: Vec<Phoneme>,
    ) -> Result<()> {
        let key = word.to_lowercase();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.regional_variants.insert(region, pronunciation);
            self.update_statistics();
            Ok(())
        } else {
            Err(G2pError::ConfigError(format!(
                "Word '{word}' not found in dictionary"
            )))
        }
    }

    /// Look up pronunciation for a word
    pub fn lookup(&self, word: &str) -> Option<&DictionaryEntry> {
        self.entries.get(&word.to_lowercase())
    }

    /// Look up pronunciation with context
    pub fn lookup_with_context(
        &self,
        word: &str,
        context: &PronunciationContext,
    ) -> Option<Vec<Phoneme>> {
        if let Some(entry) = self.lookup(word) {
            // First, check for exact context match
            for contextual in &entry.contexts {
                if contextual.context == *context {
                    return Some(contextual.phonemes.clone());
                }
            }

            // Fall back to primary pronunciation
            Some(entry.pronunciation.clone())
        } else {
            None
        }
    }

    /// Look up regional variant
    pub fn lookup_regional(&self, word: &str, region: &str) -> Option<Vec<Phoneme>> {
        self.lookup(word)?.regional_variants.get(region).cloned()
    }

    /// Get all alternatives for a word
    pub fn get_alternatives(&self, word: &str) -> Vec<Vec<Phoneme>> {
        self.lookup(word)
            .map(|entry| {
                let mut alternatives = vec![entry.pronunciation.clone()];
                alternatives.extend(entry.alternatives.clone());
                alternatives
            })
            .unwrap_or_default()
    }

    /// Update dictionary statistics
    fn update_statistics(&mut self) {
        let mut stats = DictionaryStatistics {
            total_entries: self.entries.len(),
            entries_with_alternatives: 0,
            contextual_entries: 0,
            regional_variants: 0,
            coverage: None,
        };

        for entry in self.entries.values() {
            if !entry.alternatives.is_empty() {
                stats.entries_with_alternatives += 1;
            }
            if !entry.contexts.is_empty() {
                stats.contextual_entries += 1;
            }
            if !entry.regional_variants.is_empty() {
                stats.regional_variants += 1;
            }
        }

        self.metadata.statistics = stats;
    }

    /// Load dictionary from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let dictionary: PronunciationDictionary = serde_json::from_str(&content)
            .map_err(|e| G2pError::ConfigError(format!("Failed to parse dictionary: {e}")))?;
        Ok(dictionary)
    }

    /// Save dictionary to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| G2pError::ConfigError(format!("Failed to serialize dictionary: {e}")))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Merge another dictionary into this one
    pub fn merge(&mut self, other: &PronunciationDictionary) -> Result<()> {
        if self.language != other.language {
            return Err(G2pError::ConfigError(
                "Cannot merge dictionaries of different languages".to_string(),
            ));
        }

        for (key, entry) in &other.entries {
            if self.entries.contains_key(key) {
                // Merge entries if word already exists
                if let Some(existing) = self.entries.get_mut(key) {
                    existing.alternatives.extend(entry.alternatives.clone());
                    existing.contexts.extend(entry.contexts.clone());
                    for (region, pronunciation) in &entry.regional_variants {
                        existing
                            .regional_variants
                            .insert(region.clone(), pronunciation.clone());
                    }
                }
            } else {
                // Add new entry
                self.entries.insert(key.clone(), entry.clone());
            }
        }

        self.update_statistics();
        Ok(())
    }
}

impl DictionaryManager {
    /// Create a new dictionary manager
    pub fn new() -> Self {
        Self {
            dictionaries: HashMap::new(),
            priority_order: Vec::new(),
            lookup_cache: HashMap::new(),
            cache_limit: 1000,
        }
    }

    /// Load dictionary from file
    pub fn load_dictionary<P: AsRef<Path>>(&mut self, name: String, path: P) -> Result<()> {
        let dictionary = PronunciationDictionary::load_from_file(path)?;
        self.dictionaries.insert(name.clone(), dictionary);
        if !self.priority_order.contains(&name) {
            self.priority_order.push(name);
        }
        Ok(())
    }

    /// Add dictionary
    pub fn add_dictionary(&mut self, dictionary: PronunciationDictionary) {
        let name = dictionary.name.clone();
        self.dictionaries.insert(name.clone(), dictionary);
        if !self.priority_order.contains(&name) {
            self.priority_order.push(name);
        }
    }

    /// Set dictionary priority order
    pub fn set_priority_order(&mut self, order: Vec<String>) {
        self.priority_order = order;
    }

    /// Look up word in all dictionaries (priority order)
    pub fn lookup(
        &mut self,
        word: &str,
        context: Option<&PronunciationContext>,
    ) -> Option<Vec<Phoneme>> {
        // Check cache first
        let cache_key = format!("{word}:{context:?}");
        if let Some(cached) = self.lookup_cache.get(&cache_key) {
            // Check if cache entry is still valid (1 hour TTL)
            if cached.timestamp.elapsed().unwrap_or_default().as_secs() < 3600 {
                return cached.pronunciation.clone();
            }
        }

        // Search dictionaries in priority order
        for dict_name in &self.priority_order.clone() {
            if let Some(dictionary) = self.dictionaries.get(dict_name) {
                let result = if let Some(ctx) = context {
                    dictionary.lookup_with_context(word, ctx)
                } else {
                    dictionary
                        .lookup(word)
                        .map(|entry| entry.pronunciation.clone())
                };

                if let Some(pronunciation) = result {
                    // Cache the result
                    self.cache_lookup(
                        cache_key,
                        Some(pronunciation.clone()),
                        context.cloned(),
                        dict_name.clone(),
                    );
                    return Some(pronunciation);
                }
            }
        }

        // Cache negative result
        self.cache_lookup(cache_key, None, context.cloned(), "none".to_string());
        None
    }

    /// Cache lookup result
    fn cache_lookup(
        &mut self,
        key: String,
        pronunciation: Option<Vec<Phoneme>>,
        context: Option<PronunciationContext>,
        source: String,
    ) {
        // Remove oldest entries if cache is full
        if self.lookup_cache.len() >= self.cache_limit {
            let oldest_key = self
                .lookup_cache
                .iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| k.clone());

            if let Some(old_key) = oldest_key {
                self.lookup_cache.remove(&old_key);
            }
        }

        self.lookup_cache.insert(
            key,
            CachedLookup {
                pronunciation,
                _context: context,
                _source: source,
                timestamp: std::time::SystemTime::now(),
            },
        );
    }

    /// Clear lookup cache
    pub fn clear_cache(&mut self) {
        self.lookup_cache.clear();
    }

    /// Get dictionary statistics
    pub fn get_statistics(&self) -> HashMap<String, DictionaryStatistics> {
        self.dictionaries
            .iter()
            .map(|(name, dict)| (name.clone(), dict.metadata.statistics.clone()))
            .collect()
    }

    /// Get loaded dictionary names
    pub fn get_dictionary_names(&self) -> Vec<String> {
        self.dictionaries.keys().cloned().collect()
    }
}

impl Default for DictionaryMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            created: now.clone(),
            modified: now,
            author: None,
            description: None,
            source: None,
            license: None,
            statistics: DictionaryStatistics::default(),
        }
    }
}

impl Default for EntryMetadata {
    fn default() -> Self {
        Self {
            frequency: None,
            difficulty: None,
            tags: Vec::new(),
            source: None,
            confidence: 1.0,
            verified: None,
        }
    }
}

impl Default for DictionaryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LanguageCode, SyllablePosition};

    fn create_test_phoneme(symbol: &str) -> Phoneme {
        Phoneme {
            symbol: symbol.to_string(),
            ipa_symbol: Some(symbol.to_string()),
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    #[test]
    fn test_dictionary_creation() {
        let dict = PronunciationDictionary::new("test".to_string(), LanguageCode::EnUs);
        assert_eq!(dict.name, "test");
        assert_eq!(dict.language, LanguageCode::EnUs);
        assert!(dict.entries.is_empty());
    }

    #[test]
    fn test_add_word() {
        let mut dict = PronunciationDictionary::new("test".to_string(), LanguageCode::EnUs);
        let phonemes = vec![
            create_test_phoneme("h"),
            create_test_phoneme("ɛ"),
            create_test_phoneme("l"),
            create_test_phoneme("oʊ"),
        ];

        dict.add_word("hello".to_string(), phonemes.clone())
            .unwrap();

        let entry = dict.lookup("hello").unwrap();
        assert_eq!(entry.pronunciation, phonemes);
        assert_eq!(dict.metadata.statistics.total_entries, 1);
    }

    #[test]
    fn test_contextual_pronunciation() {
        let mut dict = PronunciationDictionary::new("test".to_string(), LanguageCode::EnUs);
        let normal_phonemes = vec![
            create_test_phoneme("r"),
            create_test_phoneme("i"),
            create_test_phoneme("d"),
        ];
        let past_phonemes = vec![
            create_test_phoneme("r"),
            create_test_phoneme("ɛ"),
            create_test_phoneme("d"),
        ];

        dict.add_word("read".to_string(), normal_phonemes.clone())
            .unwrap();
        dict.add_contextual_pronunciation(
            "read",
            PronunciationContext::PartOfSpeech(PartOfSpeech::Verb),
            past_phonemes.clone(),
            0.9,
        )
        .unwrap();

        let normal = dict.lookup_with_context(
            "read",
            &PronunciationContext::PartOfSpeech(PartOfSpeech::Noun),
        );
        let contextual = dict.lookup_with_context(
            "read",
            &PronunciationContext::PartOfSpeech(PartOfSpeech::Verb),
        );

        assert_eq!(normal.unwrap(), normal_phonemes);
        assert_eq!(contextual.unwrap(), past_phonemes);
    }

    #[test]
    fn test_dictionary_manager() {
        let mut manager = DictionaryManager::new();
        let mut dict = PronunciationDictionary::new("test".to_string(), LanguageCode::EnUs);
        let phonemes = vec![
            create_test_phoneme("t"),
            create_test_phoneme("ɛ"),
            create_test_phoneme("s"),
            create_test_phoneme("t"),
        ];

        dict.add_word("test".to_string(), phonemes.clone()).unwrap();
        manager.add_dictionary(dict);

        let result = manager.lookup("test", None);
        assert_eq!(result.unwrap(), phonemes);
    }

    #[test]
    fn test_regional_variants() {
        let mut dict = PronunciationDictionary::new("test".to_string(), LanguageCode::EnUs);
        let us_phonemes = vec![
            create_test_phoneme("t"),
            create_test_phoneme("oʊ"),
            create_test_phoneme("m"),
            create_test_phoneme("eɪ"),
            create_test_phoneme("t"),
            create_test_phoneme("oʊ"),
        ];
        let uk_phonemes = vec![
            create_test_phoneme("t"),
            create_test_phoneme("ə"),
            create_test_phoneme("m"),
            create_test_phoneme("ɑː"),
            create_test_phoneme("t"),
            create_test_phoneme("əʊ"),
        ];

        dict.add_word("tomato".to_string(), us_phonemes.clone())
            .unwrap();
        dict.add_regional_variant("tomato", "UK".to_string(), uk_phonemes.clone())
            .unwrap();

        let us_pronunciation = dict.lookup("tomato").unwrap().pronunciation.clone();
        let uk_pronunciation = dict.lookup_regional("tomato", "UK").unwrap();

        assert_eq!(us_pronunciation, us_phonemes);
        assert_eq!(uk_pronunciation, uk_phonemes);
    }
}
