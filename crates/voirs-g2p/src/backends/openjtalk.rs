//! OpenJTalk-based G2P implementation for Japanese.

use crate::{G2p, G2pError, G2pMetadata, LanguageCode, Phoneme, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

// FFI declarations for OpenJTalk
extern "C" {
    // Core OpenJTalk functions
    fn OpenJTalk_initialize() -> c_int;
    fn OpenJTalk_clear() -> c_int;
    fn OpenJTalk_load_voice(voice_path: *const c_char) -> c_int;
    fn OpenJTalk_synthesis(
        text: *const c_char,
        output_wav: *const c_char,
        output_label: *const c_char,
    ) -> c_int;
    fn OpenJTalk_get_phoneme_sequence(text: *const c_char) -> *const c_char;
    fn OpenJTalk_free_string(ptr: *const c_char);
}

// Mock OpenJTalk functions for development/testing when OpenJTalk library is not available
#[cfg(test)]
mod mock_openjtalk {
    use super::*;
    use std::ptr;

    pub unsafe extern "C" fn mock_initialize() -> c_int { 0 }
    pub unsafe extern "C" fn mock_clear() -> c_int { 0 }
    pub unsafe extern "C" fn mock_load_voice(_voice_path: *const c_char) -> c_int { 0 }
    pub unsafe extern "C" fn mock_synthesis(
        _text: *const c_char,
        _output_wav: *const c_char,
        _output_label: *const c_char,
    ) -> c_int { 0 }
    pub unsafe extern "C" fn mock_get_phoneme_sequence(text: *const c_char) -> *const c_char {
        if text.is_null() {
            return ptr::null();
        }
        // Return a mock phoneme sequence
        let mock_result = CString::new("k o n n i ch i w a").unwrap();
        mock_result.into_raw()
    }
    pub unsafe extern "C" fn mock_free_string(_ptr: *const c_char) {}
}

/// OpenJTalk G2P backend for Japanese text-to-phoneme conversion
pub struct OpenJTalkG2p {
    /// Whether OpenJTalk has been initialized
    initialized: bool,
    /// Voice model path
    voice_path: Option<PathBuf>,
    /// Dictionary path
    dict_path: Option<PathBuf>,
    /// Phoneme cache for performance
    cache: HashMap<String, Vec<Phoneme>>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Whether to use mora timing
    use_mora_timing: bool,
    /// Whether to include pitch accent information
    include_pitch_accent: bool,
}

impl OpenJTalkG2p {
    /// Create a new OpenJTalk G2P backend
    pub fn new() -> Self {
        Self {
            initialized: false,
            voice_path: None,
            dict_path: None,
            cache: HashMap::new(),
            max_cache_size: 10000,
            use_mora_timing: true,
            include_pitch_accent: false,
        }
    }

    /// Initialize OpenJTalk with default settings
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing OpenJTalk G2P backend");

        #[cfg(not(test))]
        let result = unsafe { OpenJTalk_initialize() };
        #[cfg(test)]
        let result = unsafe { mock_openjtalk::mock_initialize() };

        if result != 0 {
            return Err(G2pError::ModelError("Failed to initialize OpenJTalk".to_string()));
        }

        self.initialized = true;
        info!("OpenJTalk G2P backend initialized successfully");
        Ok(())
    }

    /// Load voice model from file
    pub async fn load_voice<P: AsRef<Path>>(&mut self, voice_path: P) -> Result<()> {
        if !self.initialized {
            self.initialize().await?;
        }

        let path = voice_path.as_ref();
        if !path.exists() {
            return Err(G2pError::ModelError(format!("Voice file not found: {}", path.display())));
        }

        let path_cstring = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| G2pError::ModelError(format!("Invalid voice path: {}", e)))?;

        info!("Loading OpenJTalk voice from: {}", path.display());

        #[cfg(not(test))]
        let result = unsafe { OpenJTalk_load_voice(path_cstring.as_ptr()) };
        #[cfg(test)]
        let result = unsafe { mock_openjtalk::mock_load_voice(path_cstring.as_ptr()) };

        if result != 0 {
            return Err(G2pError::ModelError(format!("Failed to load voice: {}", path.display())));
        }

        self.voice_path = Some(path.to_path_buf());
        info!("Successfully loaded OpenJTalk voice");
        Ok(())
    }

    /// Set dictionary path
    pub fn set_dictionary_path<P: AsRef<Path>>(&mut self, dict_path: P) {
        self.dict_path = Some(dict_path.as_ref().to_path_buf());
    }

    /// Configure mora timing
    pub fn set_mora_timing(&mut self, use_mora_timing: bool) {
        self.use_mora_timing = use_mora_timing;
    }

    /// Configure pitch accent inclusion
    pub fn set_pitch_accent(&mut self, include_pitch_accent: bool) {
        self.include_pitch_accent = include_pitch_accent;
    }

    /// Convert Japanese text to phonemes
    async fn japanese_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>> {
        if !self.initialized {
            return Err(G2pError::ModelError("OpenJTalk not initialized".to_string()));
        }

        // Check cache first
        if let Some(cached_phonemes) = self.cache.get(text) {
            debug!("Using cached phonemes for: {}", text);
            return Ok(cached_phonemes.clone());
        }

        // Convert text to C string
        let text_cstring = CString::new(text.as_bytes())
            .map_err(|e| G2pError::ConversionError(format!("Invalid text: {}", e)))?;

        debug!("Converting Japanese text to phonemes: {}", text);

        // Get phoneme sequence from OpenJTalk
        #[cfg(not(test))]
        let phoneme_ptr = unsafe { OpenJTalk_get_phoneme_sequence(text_cstring.as_ptr()) };
        #[cfg(test)]
        let phoneme_ptr = unsafe { mock_openjtalk::mock_get_phoneme_sequence(text_cstring.as_ptr()) };

        if phoneme_ptr.is_null() {
            return Err(G2pError::ConversionError("OpenJTalk returned null phoneme sequence".to_string()));
        }

        // Convert C string to Rust string
        let phoneme_sequence = unsafe {
            CStr::from_ptr(phoneme_ptr)
                .to_string_lossy()
                .into_owned()
        };

        // Free the C string
        #[cfg(not(test))]
        unsafe { OpenJTalk_free_string(phoneme_ptr) };
        #[cfg(test)]
        unsafe { mock_openjtalk::mock_free_string(phoneme_ptr) };

        // Parse phoneme sequence
        let phonemes = self.parse_phoneme_sequence(&phoneme_sequence)?;

        debug!("Generated {} phonemes for Japanese text", phonemes.len());
        Ok(phonemes)
    }

    /// Parse OpenJTalk phoneme sequence into Phoneme structs
    fn parse_phoneme_sequence(&self, sequence: &str) -> Result<Vec<Phoneme>> {
        let mut phonemes = Vec::new();
        
        // Split by spaces and process each phoneme
        for phoneme_str in sequence.split_whitespace() {
            if phoneme_str.is_empty() {
                continue;
            }

            // Handle special phonemes
            let processed_phoneme = match phoneme_str {
                "pau" => " ".to_string(), // Pause
                "sil" => "".to_string(),  // Silence (skip)
                _ => self.normalize_japanese_phoneme(phoneme_str),
            };

            if !processed_phoneme.is_empty() {
                let mut phoneme = Phoneme::new(processed_phoneme);
                
                // Add mora timing if enabled
                if self.use_mora_timing {
                    phoneme.duration_ms = Some(self.estimate_mora_duration(phoneme_str) * 1000.0); // Convert to milliseconds
                }

                // Add pitch accent information if enabled
                if self.include_pitch_accent {
                    if let Some(features) = self.extract_pitch_features(phoneme_str) {
                        phoneme.custom_features = Some(features);
                    }
                }

                phonemes.push(phoneme);
            }
        }

        Ok(phonemes)
    }

    /// Normalize Japanese phoneme to IPA or standard representation
    fn normalize_japanese_phoneme(&self, phoneme: &str) -> String {
        // Map OpenJTalk phonemes to IPA or standard representations
        match phoneme {
            // Vowels
            "a" => "a".to_string(),
            "i" => "i".to_string(),
            "u" => "ɯ".to_string(), // Japanese /u/ is unrounded
            "e" => "e".to_string(),
            "o" => "o".to_string(),
            
            // Consonants
            "k" => "k".to_string(),
            "g" => "ɡ".to_string(),
            "s" => "s".to_string(),
            "z" => "z".to_string(),
            "t" => "t".to_string(),
            "d" => "d".to_string(),
            "n" => "n".to_string(),
            "h" => "h".to_string(),
            "b" => "b".to_string(),
            "p" => "p".to_string(),
            "m" => "m".to_string(),
            "y" => "j".to_string(),
            "r" => "ɾ".to_string(), // Japanese tap
            "w" => "w".to_string(),
            
            // Special sounds
            "N" => "ɴ".to_string(),   // Syllabic nasal
            "q" => "ʔ".to_string(),   // Glottal stop (sokuon)
            "ch" => "tʃ".to_string(), // Affricate
            "ts" => "ts".to_string(), // Affricate
            "sh" => "ʃ".to_string(),  // Fricative
            "j" => "dʒ".to_string(),  // Voiced affricate
            "ry" => "ɾj".to_string(), // Palatalized r
            "ky" => "kj".to_string(), // Palatalized k
            "gy" => "ɡj".to_string(), // Palatalized g
            "ny" => "nj".to_string(), // Palatalized n
            "hy" => "hj".to_string(), // Palatalized h
            "by" => "bj".to_string(), // Palatalized b
            "py" => "pj".to_string(), // Palatalized p
            "my" => "mj".to_string(), // Palatalized m
            
            // Long vowels
            "aa" => "aː".to_string(),
            "ii" => "iː".to_string(),
            "uu" => "ɯː".to_string(),
            "ee" => "eː".to_string(),
            "oo" => "oː".to_string(),
            
            // Default: return as-is
            _ => phoneme.to_string(),
        }
    }

    /// Estimate mora duration for timing
    fn estimate_mora_duration(&self, phoneme: &str) -> f32 {
        // Basic mora timing estimation (in milliseconds)
        match phoneme {
            "pau" => 200.0, // Pause
            "q" => 100.0,   // Glottal stop (short)
            "N" => 150.0,   // Syllabic nasal
            _ if phoneme.ends_with("aa") || phoneme.ends_with("ii") || 
                 phoneme.ends_with("uu") || phoneme.ends_with("ee") || 
                 phoneme.ends_with("oo") => 200.0, // Long vowels
            _ => 120.0, // Standard mora
        }
    }

    /// Extract pitch accent features
    fn extract_pitch_features(&self, phoneme: &str) -> Option<HashMap<String, String>> {
        // Basic pitch accent detection (would need more sophisticated implementation)
        let mut features = HashMap::new();
        
        // This is a simplified implementation
        // Real pitch accent would require accent phrase analysis
        if phoneme.len() > 1 && phoneme.chars().next().unwrap().is_uppercase() {
            features.insert("pitch".to_string(), "high".to_string());
        } else {
            features.insert("pitch".to_string(), "low".to_string());
        }
        
        Some(features)
    }

    /// Clear phoneme cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        debug!("Cleared OpenJTalk phoneme cache");
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.max_cache_size)
    }

    /// Set maximum cache size
    pub fn set_max_cache_size(&mut self, max_size: usize) {
        self.max_cache_size = max_size;
        
        // Trim cache if necessary
        if self.cache.len() > max_size {
            let excess = self.cache.len() - max_size;
            let keys_to_remove: Vec<String> = self.cache.keys().take(excess).cloned().collect();
            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
    }
}

impl Default for OpenJTalkG2p {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for OpenJTalkG2p {
    fn drop(&mut self) {
        if self.initialized {
            info!("Cleaning up OpenJTalk G2P backend");
            #[cfg(not(test))]
            unsafe {
                OpenJTalk_clear();
            }
            #[cfg(test)]
            unsafe {
                mock_openjtalk::mock_clear();
            }
        }
    }
}

#[async_trait]
impl G2p for OpenJTalkG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Convert Japanese text to phonemes
        let mut phonemes = self.japanese_to_phonemes(text).await?;

        // Cache the result
        if phonemes.len() <= 1000 && self.cache.len() < self.max_cache_size {
            // Note: We can't mutate self here, so caching would need to be done differently
            // For now, we skip caching in the trait implementation
        }

        debug!("OpenJTalkG2p: Generated {} phonemes for '{}'", phonemes.len(), text);
        Ok(phonemes)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![LanguageCode::Ja]
    }

    fn metadata(&self) -> G2pMetadata {
        let mut accuracy_scores = HashMap::new();
        accuracy_scores.insert(LanguageCode::Ja, 0.90); // OpenJTalk is quite accurate for Japanese

        G2pMetadata {
            name: "OpenJTalk G2P".to_string(),
            version: "1.0.0".to_string(),
            description: "OpenJTalk-based G2P for Japanese with mora timing and pitch accent".to_string(),
            supported_languages: vec![LanguageCode::Ja],
            accuracy_scores,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_openjtalk_creation() {
        let g2p = OpenJTalkG2p::new();
        assert!(!g2p.initialized);
        assert!(g2p.voice_path.is_none());
    }

    #[tokio::test]
    async fn test_openjtalk_initialization() {
        let mut g2p = OpenJTalkG2p::new();
        let result = g2p.initialize().await;
        assert!(result.is_ok());
        assert!(g2p.initialized);
    }

    #[tokio::test]
    async fn test_japanese_phoneme_conversion() {
        let mut g2p = OpenJTalkG2p::new();
        g2p.initialize().await.unwrap();
        
        let phonemes = g2p.to_phonemes("こんにちは", None).await.unwrap();
        assert!(!phonemes.is_empty());
        
        // Check that we get reasonable phonemes
        let phoneme_symbols: Vec<&str> = phonemes.iter().map(|p| p.symbol.as_str()).collect();
        println!("Japanese phonemes: {:?}", phoneme_symbols);
    }

    #[tokio::test]
    async fn test_phoneme_normalization() {
        let g2p = OpenJTalkG2p::new();
        
        // Test vowel normalization
        assert_eq!(g2p.normalize_japanese_phoneme("a"), "a");
        assert_eq!(g2p.normalize_japanese_phoneme("u"), "ɯ"); // Unrounded
        
        // Test consonant normalization
        assert_eq!(g2p.normalize_japanese_phoneme("r"), "ɾ"); // Tap
        assert_eq!(g2p.normalize_japanese_phoneme("N"), "ɴ"); // Syllabic nasal
        
        // Test palatalized sounds
        assert_eq!(g2p.normalize_japanese_phoneme("ky"), "kj");
        assert_eq!(g2p.normalize_japanese_phoneme("ry"), "ɾj");
    }

    #[test]
    fn test_mora_duration_estimation() {
        let g2p = OpenJTalkG2p::new();
        
        assert_eq!(g2p.estimate_mora_duration("pau"), 200.0);
        assert_eq!(g2p.estimate_mora_duration("q"), 100.0);
        assert_eq!(g2p.estimate_mora_duration("aa"), 200.0); // Long vowel
        assert_eq!(g2p.estimate_mora_duration("a"), 120.0);  // Regular mora
    }

    #[test]
    fn test_supported_languages() {
        let g2p = OpenJTalkG2p::new();
        let languages = g2p.supported_languages();
        assert_eq!(languages, vec![LanguageCode::Ja]);
    }

    #[test]
    fn test_metadata() {
        let g2p = OpenJTalkG2p::new();
        let metadata = g2p.metadata();
        
        assert_eq!(metadata.name, "OpenJTalk G2P");
        assert_eq!(metadata.supported_languages, vec![LanguageCode::Ja]);
        assert!(metadata.accuracy_scores.contains_key(&LanguageCode::Ja));
    }

    #[test]
    fn test_cache_functionality() {
        let mut g2p = OpenJTalkG2p::new();
        
        // Test cache stats
        let (size, max_size) = g2p.cache_stats();
        assert_eq!(size, 0);
        assert_eq!(max_size, 10000);
        
        // Test setting max cache size
        g2p.set_max_cache_size(5000);
        let (_, new_max_size) = g2p.cache_stats();
        assert_eq!(new_max_size, 5000);
        
        // Test cache clearing
        g2p.clear_cache();
        let (size, _) = g2p.cache_stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_configuration() {
        let mut g2p = OpenJTalkG2p::new();
        
        // Test mora timing configuration
        g2p.set_mora_timing(false);
        assert!(!g2p.use_mora_timing);
        
        // Test pitch accent configuration
        g2p.set_pitch_accent(true);
        assert!(g2p.include_pitch_accent);
        
        // Test dictionary path setting
        g2p.set_dictionary_path("/path/to/dict");
        assert!(g2p.dict_path.is_some());
    }
}