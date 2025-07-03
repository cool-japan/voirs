//! Neural G2P implementation placeholder.

// TODO: Implement neural G2P backends
// - Transformer-based sequence-to-sequence models
// - Attention mechanisms for alignment
// - Pre-trained model loading
// - Fine-tuning capabilities

pub struct NeuralG2p {
    // Placeholder for neural G2P implementation
}

impl NeuralG2p {
    pub fn new() -> Self {
        Self {}
    }
}

/// Phonetisaurus FST-based G2P backend
pub mod phonetisaurus {
    use crate::{G2p, G2pError, G2pMetadata, LanguageCode, Phoneme, Result};
    use async_trait::async_trait;
    use fst::{Map, MapBuilder};
    use memmap2::Mmap;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader, BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use tracing::{debug, error, info, warn};

    /// Phonetisaurus FST-based G2P backend
    pub struct PhonetisaurusG2p {
        /// Language code
        language: LanguageCode,
        /// FST model for G2P conversion
        fst_model: Option<Arc<Map<Mmap>>>,
        /// Model metadata
        metadata: G2pMetadata,
        /// Model file path
        model_path: Option<PathBuf>,
        /// Pronunciation cache
        cache: HashMap<String, Vec<Phoneme>>,
        /// Maximum number of pronunciation variants to generate
        max_variants: usize,
    }

    impl PhonetisaurusG2p {
        /// Create a new Phonetisaurus G2P backend
        pub fn new(language: LanguageCode) -> Self {
            let metadata = G2pMetadata {
                name: "Phonetisaurus G2P".to_string(),
                version: "1.0.0".to_string(),
                description: format!("FST-based G2P for {}", language.as_str()),
                supported_languages: vec![language],
                accuracy_scores: HashMap::new(),
            };

            Self {
                language,
                fst_model: None,
                metadata,
                model_path: None,
                cache: HashMap::new(),
                max_variants: 3,
            }
        }

        /// Load FST model from file
        pub async fn load_model<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
            let path = model_path.as_ref();
            
            if !path.exists() {
                return Err(G2pError::ModelError(format!("Model file not found: {}", path.display())));
            }

            info!("Loading Phonetisaurus model from: {}", path.display());

            // Memory-map the FST file
            let file = File::open(path)
                .map_err(|e| G2pError::ModelError(format!("Failed to open model file: {}", e)))?;
            
            let mmap = unsafe {
                Mmap::map(&file)
                    .map_err(|e| G2pError::ModelError(format!("Failed to memory-map model file: {}", e)))?
            };

            // Load FST from memory-mapped file
            let fst_map = Map::new(mmap)
                .map_err(|e| G2pError::ModelError(format!("Failed to load FST model: {}", e)))?;

            self.fst_model = Some(Arc::new(fst_map));
            self.model_path = Some(path.to_path_buf());

            info!("Successfully loaded Phonetisaurus model for {}", self.language.as_str());
            Ok(())
        }

        /// Download and load model from URL
        pub async fn load_model_from_url(&mut self, url: &str, cache_dir: Option<&Path>) -> Result<()> {
            let cache_dir = cache_dir.unwrap_or_else(|| Path::new("./models"));
            std::fs::create_dir_all(cache_dir)
                .map_err(|e| G2pError::ModelError(format!("Failed to create cache directory: {}", e)))?;

            let model_filename = format!("{}.fst", self.language.as_str());
            let model_path = cache_dir.join(&model_filename);

            // Check if model already exists
            if model_path.exists() {
                info!("Using cached model: {}", model_path.display());
                return self.load_model(model_path).await;
            }

            info!("Downloading Phonetisaurus model from: {}", url);

            // Download the model
            let response = reqwest::get(url)
                .await
                .map_err(|e| G2pError::ModelError(format!("Failed to download model: {}", e)))?;

            if !response.status().is_success() {
                return Err(G2pError::ModelError(format!("Failed to download model: HTTP {}", response.status())));
            }

            let model_data = response.bytes()
                .await
                .map_err(|e| G2pError::ModelError(format!("Failed to read model data: {}", e)))?;

            // Write to cache
            let mut file = std::fs::File::create(&model_path)
                .map_err(|e| G2pError::ModelError(format!("Failed to create model file: {}", e)))?;
            
            std::io::Write::write_all(&mut file, &model_data)
                .map_err(|e| G2pError::ModelError(format!("Failed to write model file: {}", e)))?;

            info!("Downloaded and cached model: {}", model_path.display());

            // Load the downloaded model
            self.load_model(model_path).await
        }

        /// Build FST model from dictionary file
        pub async fn build_model_from_dict<P: AsRef<Path>>(&mut self, dict_path: P, output_path: P) -> Result<()> {
            let dict_path = dict_path.as_ref();
            let output_path = output_path.as_ref();

            if !dict_path.exists() {
                return Err(G2pError::ModelError(format!("Dictionary file not found: {}", dict_path.display())));
            }

            info!("Building FST model from dictionary: {}", dict_path.display());

            // Read dictionary and build FST
            let mut word_pronunciations = Vec::new();
            let file = File::open(dict_path)
                .map_err(|e| G2pError::ModelError(format!("Failed to open dictionary: {}", e)))?;
            
            let reader = BufReader::new(file);
            for (line_num, line) in reader.lines().enumerate() {
                let line = line.map_err(|e| G2pError::ModelError(format!("Failed to read line {}: {}", line_num, e)))?;
                let line = line.trim();
                
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                // Parse dictionary line (format: WORD PHONEME1 PHONEME2 ...)
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 2 {
                    warn!("Invalid dictionary line {}: {}", line_num, line);
                    continue;
                }

                let word = parts[0].to_lowercase();
                let phonemes = parts[1..].join(" ");
                
                word_pronunciations.push((word, phonemes));
            }

            // Sort by word for FST building
            word_pronunciations.sort_by(|a, b| a.0.cmp(&b.0));

            // Build FST
            let mut temp_file = NamedTempFile::new()
                .map_err(|e| G2pError::ModelError(format!("Failed to create temporary file: {}", e)))?;
            
            let mut builder = MapBuilder::new(BufWriter::new(&mut temp_file))
                .map_err(|e| G2pError::ModelError(format!("Failed to create FST builder: {}", e)))?;

            for (word, phonemes) in word_pronunciations {
                // Use hash of phonemes as value (simple approach)
                let value = phonemes.len() as u64;
                builder.insert(&word, value)
                    .map_err(|e| G2pError::ModelError(format!("Failed to insert '{}' into FST: {}", word, e)))?;
            }

            builder.finish()
                .map_err(|e| G2pError::ModelError(format!("Failed to finish FST: {}", e)))?;

            // Copy to output path
            std::fs::copy(temp_file.path(), output_path)
                .map_err(|e| G2pError::ModelError(format!("Failed to copy model to output: {}", e)))?;

            info!("Built FST model: {}", output_path.display());

            // Load the newly built model
            self.load_model(output_path).await
        }

        /// Generate pronunciations using FST traversal
        fn generate_pronunciations(&self, word: &str) -> Result<Vec<Phoneme>> {
            let fst_model = self.fst_model.as_ref()
                .ok_or_else(|| G2pError::ModelError("No FST model loaded".to_string()))?;

            let word = word.to_lowercase();

            // Check cache first
            if let Some(cached_phonemes) = self.cache.get(&word) {
                return Ok(cached_phonemes.clone());
            }

            // Try FST-based pronunciation generation
            let phonemes = if let Some(_value) = fst_model.get(&word) {
                // Word found in FST - generate pronunciation using character-level traversal
                self.fst_based_pronunciation(&word)?
            } else {
                // Out-of-vocabulary word - try character-level FST traversal
                debug!("OOV word: {}", word);
                match self.character_level_fst_traversal(&word) {
                    Ok(phonemes) if !phonemes.is_empty() => phonemes,
                    _ => {
                        // Fall back to rule-based approach for completely unknown words
                        self.rule_based_fallback(&word)
                    }
                }
            };

            Ok(phonemes)
        }

        /// Generate pronunciation using FST-based approach
        fn fst_based_pronunciation(&self, word: &str) -> Result<Vec<Phoneme>> {
            // In a full implementation, this would use proper FST traversal algorithms
            // For now, we implement a character-by-character pronunciation mapping
            // that simulates FST-based conversion with better accuracy than simple fallback
            
            let mut phonemes = Vec::new();
            let chars: Vec<char> = word.chars().collect();
            let mut i = 0;

            while i < chars.len() {
                let mut matched = false;

                // Try multi-character patterns first (simulating FST paths)
                for pattern_len in (2..=4).rev() {
                    if i + pattern_len <= chars.len() {
                        let pattern: String = chars[i..i + pattern_len].iter().collect();
                        if let Some(phoneme_str) = self.get_fst_phoneme(&pattern) {
                            phonemes.push(Phoneme::new(phoneme_str));
                            i += pattern_len;
                            matched = true;
                            break;
                        }
                    }
                }

                // Single character fallback
                if !matched {
                    let char_str = chars[i].to_string();
                    if let Some(phoneme_str) = self.get_fst_phoneme(&char_str) {
                        phonemes.push(Phoneme::new(phoneme_str));
                    } else {
                        // Use basic phoneme mapping for unknown characters
                        phonemes.push(Phoneme::new(self.basic_char_to_phoneme(chars[i])));
                    }
                    i += 1;
                }
            }

            Ok(phonemes)
        }

        /// Character-level FST traversal for OOV words
        fn character_level_fst_traversal(&self, word: &str) -> Result<Vec<Phoneme>> {
            // Implement a more sophisticated character-level pronunciation
            // This simulates traversing an FST at the character level
            let mut phonemes = Vec::new();
            
            for (i, c) in word.chars().enumerate() {
                let context_left = if i > 0 { Some(word.chars().nth(i - 1).unwrap()) } else { None };
                let context_right = if i < word.len() - 1 { Some(word.chars().nth(i + 1).unwrap()) } else { None };
                
                let phoneme_str = self.context_aware_phoneme(c, context_left, context_right);
                phonemes.push(Phoneme::new(phoneme_str));
            }
            
            Ok(phonemes)
        }

        /// Get phoneme from FST-like mapping (enhanced pronunciation rules)
        fn get_fst_phoneme(&self, pattern: &str) -> Option<String> {
            match self.language {
                LanguageCode::EnUs | LanguageCode::EnGb => {
                    match pattern {
                        // Common English digraphs and trigraphs
                        "tion" => Some("ʃən".to_string()),
                        "sion" => Some("ʒən".to_string()),
                        "ough" => Some("ʌf".to_string()),  // tough, rough
                        "augh" => Some("ɔːf".to_string()), // laugh
                        "eigh" => Some("eɪ".to_string()),  // eight, weigh
                        "ight" => Some("aɪt".to_string()), // light, night
                        "th" => Some("θ".to_string()),     // think
                        "ch" => Some("tʃ".to_string()),    // chair
                        "sh" => Some("ʃ".to_string()),     // ship
                        "ph" => Some("f".to_string()),     // phone
                        "gh" => Some("".to_string()),      // light (silent)
                        "ck" => Some("k".to_string()),     // back
                        "ng" => Some("ŋ".to_string()),     // sing
                        "qu" => Some("kw".to_string()),    // queen
                        "ea" => Some("iː".to_string()),    // eat
                        "ee" => Some("iː".to_string()),    // see
                        "oo" => Some("uː".to_string()),    // moon
                        "ai" => Some("eɪ".to_string()),    // rain
                        "ay" => Some("eɪ".to_string()),    // day
                        "oi" => Some("ɔɪ".to_string()),    // oil
                        "oy" => Some("ɔɪ".to_string()),    // boy
                        "ou" => Some("aʊ".to_string()),    // out
                        "ow" => Some("aʊ".to_string()),    // cow
                        "ar" => Some("ɑːr".to_string()),   // car
                        "er" => Some("ɜːr".to_string()),   // her
                        "ir" => Some("ɜːr".to_string()),   // bird
                        "or" => Some("ɔːr".to_string()),   // for
                        "ur" => Some("ɜːr".to_string()),   // turn
                        _ => None,
                    }
                }
                LanguageCode::De => {
                    match pattern {
                        "sch" => Some("ʃ".to_string()),
                        "ch" => Some("ç".to_string()),
                        "ck" => Some("k".to_string()),
                        "tz" => Some("ts".to_string()),
                        "pf" => Some("pf".to_string()),
                        "ß" => Some("s".to_string()),
                        "ä" => Some("ɛ".to_string()),
                        "ö" => Some("ø".to_string()),
                        "ü" => Some("y".to_string()),
                        _ => None,
                    }
                }
                _ => None,
            }
        }

        /// Context-aware phoneme generation
        fn context_aware_phoneme(&self, c: char, left: Option<char>, right: Option<char>) -> String {
            match self.language {
                LanguageCode::EnUs | LanguageCode::EnGb => {
                    match c {
                        'c' => {
                            // Soft c before e, i, y
                            if matches!(right, Some('e') | Some('i') | Some('y')) {
                                "s".to_string()
                            } else {
                                "k".to_string()
                            }
                        }
                        'g' => {
                            // Soft g before e, i, y (sometimes)
                            if matches!(right, Some('e') | Some('i') | Some('y')) {
                                "dʒ".to_string() // gem, giant
                            } else {
                                "ɡ".to_string()
                            }
                        }
                        's' => {
                            // S between vowels often becomes z
                            if left.map_or(false, |l| "aeiou".contains(l)) &&
                               right.map_or(false, |r| "aeiou".contains(r)) {
                                "z".to_string()
                            } else {
                                "s".to_string()
                            }
                        }
                        'x' => {
                            // X at beginning is often z
                            if left.is_none() {
                                "z".to_string()
                            } else {
                                "ks".to_string()
                            }
                        }
                        _ => self.basic_char_to_phoneme(c),
                    }
                }
                _ => self.basic_char_to_phoneme(c),
            }
        }

        /// Basic character to phoneme mapping
        fn basic_char_to_phoneme(&self, c: char) -> String {
            match c {
                'a' => "æ".to_string(),
                'e' => "ɛ".to_string(),
                'i' => "ɪ".to_string(),
                'o' => "ɑ".to_string(),
                'u' => "ʌ".to_string(),
                'y' => "ɪ".to_string(),
                _ => c.to_string(),
            }
        }

        /// Rule-based fallback for OOV words
        fn rule_based_fallback(&self, word: &str) -> Vec<Phoneme> {
            // Simple character-to-phoneme mapping as fallback
            // In a real implementation, this would use more sophisticated rules
            word.chars()
                .filter(|c| c.is_alphabetic())
                .map(|c| {
                    let phoneme_str = match c {
                        'a' => "æ",
                        'e' => "ɛ", 
                        'i' => "ɪ",
                        'o' => "ɑ",
                        'u' => "ʌ",
                        'y' => "ɪ",
                        _ => return Phoneme::new(c.to_string()),
                    };
                    Phoneme::new(phoneme_str)
                })
                .collect()
        }

        /// Get model information
        pub fn model_info(&self) -> Option<&PathBuf> {
            self.model_path.as_ref()
        }

        /// Set maximum number of pronunciation variants
        pub fn set_max_variants(&mut self, max_variants: usize) {
            self.max_variants = max_variants;
        }

        /// Clear pronunciation cache
        pub fn clear_cache(&mut self) {
            self.cache.clear();
        }

        /// Get cache statistics
        pub fn cache_stats(&self) -> (usize, usize) {
            (self.cache.len(), self.cache.capacity())
        }
    }

    #[async_trait]
    impl G2p for PhonetisaurusG2p {
        async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
            if self.fst_model.is_none() {
                return Err(G2pError::ModelError("No FST model loaded. Call load_model() first.".to_string()));
            }

            let words: Vec<&str> = text.split_whitespace().collect();
            let mut all_phonemes = Vec::new();

            for (i, word) in words.iter().enumerate() {
                let clean_word: String = word.chars()
                    .filter(|c| c.is_alphabetic())
                    .collect();

                if clean_word.is_empty() {
                    continue;
                }

                let word_phonemes = self.generate_pronunciations(&clean_word)?;
                all_phonemes.extend(word_phonemes);

                // Add word boundary (except after last word)
                if i < words.len() - 1 {
                    all_phonemes.push(Phoneme::new(" "));
                }
            }

            debug!("PhonetisaurusG2p: Generated {} phonemes for '{}'", all_phonemes.len(), text);
            Ok(all_phonemes)
        }

        fn supported_languages(&self) -> Vec<LanguageCode> {
            vec![self.language]
        }

        fn metadata(&self) -> G2pMetadata {
            self.metadata.clone()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::Write;
        use tempfile::NamedTempFile;

        #[tokio::test]
        async fn test_phonetisaurus_creation() {
            let g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            assert_eq!(g2p.language, LanguageCode::EnUs);
            assert!(g2p.fst_model.is_none());
        }

        #[tokio::test]
        async fn test_phonetisaurus_without_model() {
            let g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            
            // Should fail without loaded model
            let result = g2p.to_phonemes("hello", None).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_model_loading_nonexistent() {
            let mut g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            
            // Should fail for nonexistent file
            let result = g2p.load_model("nonexistent.fst").await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_build_model_from_dict() {
            let mut g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            
            // Create a temporary dictionary file
            let mut dict_file = NamedTempFile::new().unwrap();
            writeln!(dict_file, "HELLO H EH L OW").unwrap();
            writeln!(dict_file, "WORLD W ER L D").unwrap();
            writeln!(dict_file, "TEST T EH S T").unwrap();
            dict_file.flush().unwrap();
            
            let output_file = NamedTempFile::new().unwrap();
            
            // Build model from dictionary
            let result = g2p.build_model_from_dict(dict_file.path(), output_file.path()).await;
            assert!(result.is_ok());
            
            // Should have model loaded now
            assert!(g2p.fst_model.is_some());
        }

        #[tokio::test]
        async fn test_cache_functionality() {
            let mut g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            
            // Test cache stats
            let (size, capacity) = g2p.cache_stats();
            assert_eq!(size, 0);
            
            // Test cache clearing
            g2p.clear_cache();
            let (size, _) = g2p.cache_stats();
            assert_eq!(size, 0);
        }

        #[tokio::test]
        async fn test_supported_languages() {
            let g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            let languages = g2p.supported_languages();
            assert_eq!(languages, vec![LanguageCode::EnUs]);
        }

        #[tokio::test]
        async fn test_metadata() {
            let g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            let metadata = g2p.metadata();
            assert_eq!(metadata.name, "Phonetisaurus G2P");
            assert_eq!(metadata.supported_languages, vec![LanguageCode::EnUs]);
        }

        #[test]
        fn test_rule_based_fallback() {
            let g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            let phonemes = g2p.rule_based_fallback("cat");
            assert_eq!(phonemes.len(), 3);
            assert_eq!(phonemes[0].symbol, "c"); // 'c' -> 'c' (not mapped)
            assert_eq!(phonemes[1].symbol, "æ"); // 'a' -> 'æ'
            assert_eq!(phonemes[2].symbol, "t"); // 't' -> 't' (not mapped)
        }

        #[test]
        fn test_max_variants() {
            let mut g2p = PhonetisaurusG2p::new(LanguageCode::EnUs);
            g2p.set_max_variants(5);
            assert_eq!(g2p.max_variants, 5);
        }
    }
}