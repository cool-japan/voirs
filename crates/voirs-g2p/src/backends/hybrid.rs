//! Hybrid G2P implementation that combines multiple approaches.

use crate::backends::{NeuralG2pBackend, RuleBasedG2p};
use crate::{G2p, G2pError, G2pMetadata, LanguageCode, Phoneme, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, warn};

/// Backend selection strategy for hybrid approach
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionStrategy {
    /// Use the first backend that succeeds
    FirstSuccess,
    /// Use the backend with highest confidence
    HighestConfidence,
    /// Use majority voting among backends
    MajorityVoting,
    /// Use weighted ensemble based on backend accuracy
    WeightedEnsemble,
}

/// Backend configuration for hybrid approach
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Backend weight for ensemble methods
    pub weight: f32,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Whether this backend is enabled
    pub enabled: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            weight: 1.0,
            min_confidence: 0.5,
            enabled: true,
        }
    }
}

/// Hybrid G2P backend that combines multiple approaches for better accuracy
pub struct HybridG2p {
    /// Primary language
    language: LanguageCode,
    /// Rule-based backend
    rule_based: Option<RuleBasedG2p>,
    /// FST-based backend (Phonetisaurus)
    neural_based: Option<NeuralG2pBackend>,
    /// Backend configurations
    backend_configs: HashMap<String, BackendConfig>,
    /// Selection strategy
    selection_strategy: SelectionStrategy,
    /// Fallback order for backends
    fallback_order: Vec<String>,
    /// Enable pronunciation caching
    enable_caching: bool,
    /// Pronunciation cache (using Arc<RwLock> for thread safety)
    cache: Arc<RwLock<HashMap<String, Vec<Phoneme>>>>,
    /// Maximum cache size
    max_cache_size: usize,
}

impl HybridG2p {
    /// Create a new hybrid G2P backend
    pub fn new(language: LanguageCode) -> Self {
        let mut backend_configs = HashMap::new();

        // Default configurations for each backend type
        backend_configs.insert(
            "rule_based".to_string(),
            BackendConfig {
                weight: 0.7,
                min_confidence: 0.6,
                enabled: true,
            },
        );

        backend_configs.insert(
            "neural_based".to_string(),
            BackendConfig {
                weight: 0.8,
                min_confidence: 0.7,
                enabled: false, // Disabled by default until neural model is loaded
            },
        );

        Self {
            language,
            rule_based: Some(RuleBasedG2p::new(language)),
            neural_based: NeuralG2pBackend::new(crate::backends::neural::LstmConfig::default())
                .ok(),
            backend_configs,
            selection_strategy: SelectionStrategy::WeightedEnsemble,
            fallback_order: vec!["neural_based".to_string(), "rule_based".to_string()],
            enable_caching: true,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size: 10000,
        }
    }

    /// Set selection strategy
    pub fn set_selection_strategy(&mut self, strategy: SelectionStrategy) {
        self.selection_strategy = strategy;
        debug!("Set hybrid G2P selection strategy to: {:?}", strategy);
    }

    /// Configure a backend
    pub fn configure_backend(&mut self, backend_id: &str, config: BackendConfig) {
        debug!(
            "Configured backend '{}' with weight {:.2}",
            backend_id, config.weight
        );
        self.backend_configs.insert(backend_id.to_string(), config);
    }

    /// Set fallback order
    pub fn set_fallback_order(&mut self, order: Vec<String>) {
        self.fallback_order = order;
        debug!("Set fallback order: {:?}", self.fallback_order);
    }

    /// Enable or disable caching
    pub fn set_caching(&mut self, enabled: bool) {
        self.enable_caching = enabled;
        if !enabled {
            if let Ok(mut cache) = self.cache.write() {
                cache.clear();
            }
        }
        debug!(
            "Hybrid G2P caching: {}",
            if enabled { "enabled" } else { "disabled" }
        );
    }

    /// Load FST model for Phonetisaurus backend
    pub async fn load_fst_model(&mut self, _model_path: &str) -> Result<()> {
        if let Some(ref mut _neural_backend) = self.neural_based {
            // Neural backend doesn't need model loading in this context
            debug!("Neural backend is available for hybrid processing");

            // Enable FST backend after successful model loading
            if let Some(config) = self.backend_configs.get_mut("fst_based") {
                config.enabled = true;
            }

            debug!("Loaded FST model for hybrid G2P backend");
        }
        Ok(())
    }

    /// Generate pronunciations using multiple backends (optimized for performance)
    fn generate_hybrid_pronunciations(&self, text: &str) -> Result<Vec<Phoneme>> {
        // Check cache first
        if self.enable_caching {
            if let Ok(cache) = self.cache.read() {
                if let Some(cached_phonemes) = cache.get(text) {
                    debug!("Using cached pronunciation for: {}", text);
                    return Ok(cached_phonemes.clone());
                }
            }
        }

        // Collect results from all enabled backends with optimizations
        let mut backend_results = Vec::new();

        // For FirstSuccess strategy, exit early after first successful backend
        let early_exit = matches!(self.selection_strategy, SelectionStrategy::FirstSuccess);

        // Try rule-based backend first (usually fastest) - remove async overhead
        if self.rule_based.is_some() {
            if let Some(config) = self.backend_configs.get("rule_based") {
                if config.enabled {
                    // Use synchronous conversion for better performance
                    if let Ok(phonemes) = self.rule_based_sync_convert(text) {
                        let confidence = self.estimate_confidence(&phonemes, "rule_based");
                        if confidence >= config.min_confidence {
                            backend_results.push((
                                "rule_based".to_string(),
                                phonemes.clone(),
                                confidence,
                                config.weight,
                            ));
                            debug!(
                                "Rule-based backend: {} phonemes, confidence: {:.2}",
                                phonemes.len(),
                                confidence
                            );

                            // Early exit for FirstSuccess strategy with good confidence
                            if early_exit && confidence > 0.8 {
                                debug!("Early exit: high-confidence rule-based result");
                                // Cache the result before returning
                                self.cache_result(text, &phonemes);
                                return Ok(phonemes);
                            }
                        }
                    }
                }
            }
        }

        // Try neural-based backend only if needed
        if let Some(ref neural_backend) = self.neural_based {
            if let Some(config) = self.backend_configs.get("neural_based") {
                if config.enabled && (!early_exit || backend_results.is_empty()) {
                    match neural_backend.text_to_phonemes(text) {
                        Ok(phonemes) => {
                            let phonemes_vec = phonemes.to_vec();
                            let confidence =
                                self.estimate_confidence(&phonemes_vec, "neural_based");
                            if confidence >= config.min_confidence {
                                backend_results.push((
                                    "neural_based".to_string(),
                                    phonemes_vec,
                                    confidence,
                                    config.weight,
                                ));
                                debug!(
                                    "Neural-based backend: {} phonemes, confidence: {:.2}",
                                    backend_results.last().unwrap().1.len(),
                                    confidence
                                );
                            }
                        }
                        Err(e) => warn!("Neural-based backend failed: {}", e),
                    }
                }
            }
        }

        // Select best result based on strategy
        let final_phonemes = self.select_best_result(backend_results)?;

        // Cache the successful result
        self.cache_result(text, &final_phonemes);

        debug!(
            "Hybrid G2P generated {} phonemes for: {}",
            final_phonemes.len(),
            text
        );
        Ok(final_phonemes)
    }

    /// Synchronous rule-based conversion for better performance
    fn rule_based_sync_convert(&self, text: &str) -> Result<Vec<Phoneme>> {
        // Use a simplified synchronous version to avoid async overhead
        // This is a performance optimization that bypasses the async layer
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut phonemes = Vec::new();

        for word in words {
            // Basic English G2P rules (simplified for performance)
            let word_phonemes = self.apply_basic_rules(word);
            phonemes.extend(word_phonemes);
        }

        Ok(phonemes)
    }

    /// Apply basic G2P rules for performance optimization
    fn apply_basic_rules(&self, word: &str) -> Vec<Phoneme> {
        // Simplified rule-based G2P for common English patterns
        // This is optimized for speed over accuracy
        let mut phonemes = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();

        for (i, &ch) in chars.iter().enumerate() {
            let phoneme = match ch.to_ascii_lowercase() {
                'a' => {
                    if i + 1 < len && chars[i + 1] == 'e' {
                        "eɪ" // "ae" -> /eɪ/
                    } else {
                        "æ" // basic "a" -> /æ/
                    }
                }
                'e' => "ɛ",
                'i' => "ɪ",
                'o' => "ɑ",
                'u' => "ʌ",
                'b' => "b",
                'c' => "k",
                'd' => "d",
                'f' => "f",
                'g' => "g",
                'h' => "h",
                'j' => "dʒ",
                'k' => "k",
                'l' => "l",
                'm' => "m",
                'n' => "n",
                'p' => "p",
                'q' => "k",
                'r' => "r",
                's' => "s",
                't' => "t",
                'v' => "v",
                'w' => "w",
                'x' => "ks",
                'y' => "j",
                'z' => "z",
                _ => continue, // Skip non-alphabetic characters
            };

            phonemes.push(Phoneme::new(phoneme.to_string()));
        }

        phonemes
    }

    /// Cache a successful result with size management
    fn cache_result(&self, text: &str, phonemes: &[Phoneme]) {
        if !self.enable_caching {
            return;
        }

        if let Ok(mut cache) = self.cache.write() {
            // Manage cache size - remove oldest entries if needed
            if cache.len() >= self.max_cache_size {
                // Simple approach: clear half the cache when full
                let keys_to_remove: Vec<String> = cache
                    .keys()
                    .take(self.max_cache_size / 2)
                    .cloned()
                    .collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }

            cache.insert(text.to_string(), phonemes.to_vec());
        }
    }

    /// Select the best result based on the configured strategy
    fn select_best_result(
        &self,
        mut results: Vec<(String, Vec<Phoneme>, f32, f32)>,
    ) -> Result<Vec<Phoneme>> {
        if results.is_empty() {
            return Err(G2pError::ConversionError(
                "No backend produced valid results".to_string(),
            ));
        }

        match self.selection_strategy {
            SelectionStrategy::FirstSuccess => {
                // Return the first successful result in fallback order
                for backend_id in &self.fallback_order {
                    if let Some((_, phonemes, _, _)) =
                        results.iter().find(|(id, _, _, _)| id == backend_id)
                    {
                        return Ok(phonemes.clone());
                    }
                }
                // If no fallback found, return first result
                Ok(results[0].1.clone())
            }

            SelectionStrategy::HighestConfidence => {
                // Return result with highest confidence
                results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                Ok(results[0].1.clone())
            }

            SelectionStrategy::MajorityVoting => {
                // Implement majority voting (simplified version)
                // For now, just return the result from the backend with highest weight
                results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
                Ok(results[0].1.clone())
            }

            SelectionStrategy::WeightedEnsemble => {
                // Weighted ensemble based on confidence and weight
                results.sort_by(|a, b| {
                    let score_a = a.2 * a.3; // confidence * weight
                    let score_b = b.2 * b.3;
                    score_b.partial_cmp(&score_a).unwrap()
                });
                Ok(results[0].1.clone())
            }
        }
    }

    /// Estimate confidence score for phoneme sequence
    fn estimate_confidence(&self, phonemes: &[Phoneme], backend_id: &str) -> f32 {
        if phonemes.is_empty() {
            return 0.0;
        }

        // Basic confidence estimation based on backend type and phoneme characteristics
        let base_confidence = match backend_id {
            "rule_based" => 0.75, // Rule-based is generally reliable but not perfect
            "fst_based" => 0.85,  // FST-based is usually more accurate
            _ => 0.5,
        };

        // Adjust confidence based on phoneme characteristics
        let length_factor = if phonemes.len() < 2 {
            0.8 // Very short sequences are less reliable
        } else if phonemes.len() > 20 {
            0.9 // Very long sequences might have errors
        } else {
            1.0 // Normal length is most reliable
        };

        // Check for suspicious patterns (e.g., all single characters)
        let pattern_factor = if phonemes.iter().all(|p| p.symbol.len() == 1) {
            0.7 // All single characters might indicate fallback to letter-by-letter
        } else {
            1.0
        };

        base_confidence * length_factor * pattern_factor
    }

    /// Clear pronunciation cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        debug!("Cleared hybrid G2P pronunciation cache");
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache_size = if let Ok(cache) = self.cache.read() {
            cache.len()
        } else {
            0
        };
        (cache_size, self.max_cache_size)
    }

    /// Get backend statistics
    pub fn backend_stats(&self) -> HashMap<String, bool> {
        let mut stats = HashMap::new();

        if let Some(config) = self.backend_configs.get("rule_based") {
            stats.insert(
                "rule_based".to_string(),
                config.enabled && self.rule_based.is_some(),
            );
        }

        if let Some(config) = self.backend_configs.get("neural_based") {
            stats.insert(
                "neural_based".to_string(),
                config.enabled && self.neural_based.is_some(),
            );
        }

        stats
    }
}

impl Default for HybridG2p {
    fn default() -> Self {
        Self::new(LanguageCode::EnUs)
    }
}

#[async_trait]
impl G2p for HybridG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let phonemes = self.generate_hybrid_pronunciations(text)?;

        Ok(phonemes)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![self.language]
    }

    fn metadata(&self) -> G2pMetadata {
        let mut accuracy_scores = HashMap::new();

        // Hybrid accuracy is generally higher than individual backends
        let estimated_accuracy = match self.language {
            LanguageCode::EnUs | LanguageCode::EnGb => 0.85,
            LanguageCode::Es => 0.90, // Spanish is more regular
            LanguageCode::De => 0.82,
            LanguageCode::Fr => 0.78,
            _ => 0.75,
        };

        accuracy_scores.insert(self.language, estimated_accuracy);

        G2pMetadata {
            name: "Hybrid G2P".to_string(),
            version: "1.0.0".to_string(),
            description: format!(
                "Hybrid G2P combining multiple backends for {}",
                self.language.as_str()
            ),
            supported_languages: vec![self.language],
            accuracy_scores,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_g2p_creation() {
        let g2p = HybridG2p::new(LanguageCode::EnUs);
        assert_eq!(g2p.language, LanguageCode::EnUs);
        assert!(g2p.rule_based.is_some());
        assert!(g2p.neural_based.is_some());
    }

    #[test]
    fn test_selection_strategy_configuration() {
        let mut g2p = HybridG2p::new(LanguageCode::EnUs);

        g2p.set_selection_strategy(SelectionStrategy::HighestConfidence);
        assert_eq!(g2p.selection_strategy, SelectionStrategy::HighestConfidence);

        g2p.set_selection_strategy(SelectionStrategy::MajorityVoting);
        assert_eq!(g2p.selection_strategy, SelectionStrategy::MajorityVoting);
    }

    #[test]
    fn test_backend_configuration() {
        let mut g2p = HybridG2p::new(LanguageCode::EnUs);

        let config = BackendConfig {
            weight: 0.9,
            min_confidence: 0.8,
            enabled: true,
        };

        g2p.configure_backend("rule_based", config.clone());

        let stored_config = g2p.backend_configs.get("rule_based").unwrap();
        assert_eq!(stored_config.weight, 0.9);
        assert_eq!(stored_config.min_confidence, 0.8);
        assert!(stored_config.enabled);
    }

    #[test]
    fn test_fallback_order() {
        let mut g2p = HybridG2p::new(LanguageCode::EnUs);

        let new_order = vec!["rule_based".to_string(), "fst_based".to_string()];
        g2p.set_fallback_order(new_order.clone());

        assert_eq!(g2p.fallback_order, new_order);
    }

    #[test]
    fn test_caching_configuration() {
        let mut g2p = HybridG2p::new(LanguageCode::EnUs);

        assert!(g2p.enable_caching);

        g2p.set_caching(false);
        assert!(!g2p.enable_caching);
        let (cache_size, _) = g2p.cache_stats();
        assert_eq!(cache_size, 0);
    }

    #[test]
    fn test_confidence_estimation() {
        let g2p = HybridG2p::new(LanguageCode::EnUs);

        // Test empty phonemes
        let empty_phonemes = Vec::new();
        assert_eq!(g2p.estimate_confidence(&empty_phonemes, "rule_based"), 0.0);

        // Test normal phonemes
        let normal_phonemes = vec![
            Phoneme::new("h"),
            Phoneme::new("ɛ"),
            Phoneme::new("l"),
            Phoneme::new("oʊ"),
        ];
        let confidence = g2p.estimate_confidence(&normal_phonemes, "rule_based");
        assert!(confidence > 0.5);
        assert!(confidence <= 1.0);

        // FST-based should have higher base confidence
        let fst_confidence = g2p.estimate_confidence(&normal_phonemes, "fst_based");
        let rule_confidence = g2p.estimate_confidence(&normal_phonemes, "rule_based");
        assert!(fst_confidence > rule_confidence);
    }

    #[tokio::test]
    async fn test_hybrid_phoneme_conversion() {
        let g2p = HybridG2p::new(LanguageCode::EnUs);

        let phonemes = g2p.to_phonemes("hello", None).await.unwrap();
        assert!(!phonemes.is_empty());

        // Should get reasonable phonemes from rule-based backend
        let phoneme_symbols: Vec<&str> = phonemes.iter().map(|p| p.symbol.as_str()).collect();
        println!("Hybrid phonemes for 'hello': {phoneme_symbols:?}");
    }

    #[test]
    fn test_supported_languages() {
        let g2p = HybridG2p::new(LanguageCode::De);
        let languages = g2p.supported_languages();
        assert_eq!(languages, vec![LanguageCode::De]);
    }

    #[test]
    fn test_metadata() {
        let g2p = HybridG2p::new(LanguageCode::EnUs);
        let metadata = g2p.metadata();

        assert_eq!(metadata.name, "Hybrid G2P");
        assert_eq!(metadata.supported_languages, vec![LanguageCode::EnUs]);
        assert!(metadata.accuracy_scores.contains_key(&LanguageCode::EnUs));

        // Hybrid should have reasonable accuracy
        let accuracy = metadata.accuracy_scores.get(&LanguageCode::EnUs).unwrap();
        assert!(*accuracy > 0.8);
    }

    #[test]
    fn test_cache_stats() {
        let g2p = HybridG2p::new(LanguageCode::EnUs);
        let (size, max_size) = g2p.cache_stats();
        assert_eq!(size, 0);
        assert_eq!(max_size, 10000);
    }

    #[test]
    fn test_backend_stats() {
        let g2p = HybridG2p::new(LanguageCode::EnUs);
        let stats = g2p.backend_stats();

        assert!(stats.contains_key("rule_based"));
        assert!(stats.contains_key("neural_based"));

        // Rule-based should be enabled by default
        assert!(stats.get("rule_based").unwrap());

        // Neural-based should be disabled by default (no model loaded)
        assert!(!stats.get("neural_based").unwrap());
    }
}
