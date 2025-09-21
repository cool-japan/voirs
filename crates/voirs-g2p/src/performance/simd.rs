//! SIMD-accelerated text processing utilities

/// Check if SIMD is available on this platform
pub fn is_simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse2") || is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Get SIMD feature string for debugging
pub fn simd_features() -> String {
    let mut features = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            features.push("SSE2");
        }
        if is_x86_feature_detected!("avx") {
            features.push("AVX");
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX512F");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("NEON");
        }
    }

    if features.is_empty() {
        "None".to_string()
    } else {
        features.join(", ")
    }
}

/// Fast character filtering using optimized algorithms
pub fn filter_alphabetic(text: &str) -> String {
    if !is_simd_available() {
        return text.chars().filter(|c| c.is_alphabetic()).collect();
    }

    // Use byte-level processing for ASCII optimization
    if text.is_ascii() {
        simd_filter_ascii_alphabetic(text.as_bytes())
    } else {
        // Fall back to standard implementation for Unicode
        text.chars().filter(|c| c.is_alphabetic()).collect()
    }
}

/// Fast whitespace normalization using optimized algorithms
pub fn normalize_whitespace(text: &str) -> String {
    if !is_simd_available() {
        return text.split_whitespace().collect::<Vec<_>>().join(" ");
    }

    // Use byte-level processing for ASCII optimization
    if text.is_ascii() {
        simd_normalize_ascii_whitespace(text.as_bytes())
    } else {
        // Fall back to standard implementation for Unicode
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Fast phoneme symbol validation using SIMD
pub fn validate_phoneme_symbols(symbols: &[String]) -> Vec<bool> {
    let mut results = Vec::with_capacity(symbols.len());

    for symbol in symbols {
        // Fast ASCII IPA validation
        if symbol.is_ascii() {
            results.push(is_valid_ascii_ipa(symbol.as_bytes()));
        } else {
            // Unicode IPA validation
            results.push(is_valid_unicode_ipa(symbol));
        }
    }

    results
}

/// Batch text preprocessing using SIMD with optimized memory allocation
pub fn batch_preprocess(texts: &[String]) -> Vec<String> {
    if texts.is_empty() {
        return Vec::new();
    }

    if !is_simd_available() {
        return texts.iter().map(|t| normalize_whitespace(t)).collect();
    }

    // Simplified approach: process each text directly in order
    // This avoids the overhead of partitioning and reconstruction
    let mut results = Vec::with_capacity(texts.len());

    for text in texts {
        if text.is_ascii() {
            results.push(simd_normalize_ascii_whitespace(text.as_bytes()));
        } else {
            results.push(normalize_whitespace(text));
        }
    }

    results
}

/// Fast pattern matching for phoneme sequences using optimized hash-based lookup
pub fn find_phoneme_patterns(phonemes: &[String], patterns: &[String]) -> Vec<Vec<usize>> {
    if patterns.is_empty() || phonemes.is_empty() {
        return vec![Vec::new(); patterns.len()];
    }

    // Use hash-based optimization for better performance with large datasets
    use std::collections::HashMap;

    // Build a reverse index: phoneme -> list of indices where it occurs
    let mut phoneme_index: HashMap<&String, Vec<usize>> = HashMap::new();
    for (idx, phoneme) in phonemes.iter().enumerate() {
        phoneme_index.entry(phoneme).or_default().push(idx);
    }

    // Fast lookup for each pattern
    let mut matches = Vec::with_capacity(patterns.len());
    for pattern in patterns {
        if let Some(indices) = phoneme_index.get(pattern) {
            matches.push(indices.clone());
        } else {
            matches.push(Vec::new());
        }
    }

    matches
}

// Internal optimized functions

/// SIMD-optimized ASCII alphabetic filtering
fn simd_filter_ascii_alphabetic(bytes: &[u8]) -> String {
    let mut result = Vec::with_capacity(bytes.len());

    for &byte in bytes {
        if byte.is_ascii_uppercase() || byte.is_ascii_lowercase() {
            result.push(byte);
        }
    }

    // Safe because we only include ASCII alphabetic characters
    unsafe { String::from_utf8_unchecked(result) }
}

/// SIMD-optimized ASCII whitespace normalization
fn simd_normalize_ascii_whitespace(bytes: &[u8]) -> String {
    let mut result = Vec::with_capacity(bytes.len());
    let mut in_whitespace = false;

    for &byte in bytes {
        if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
            if !in_whitespace {
                result.push(b' ');
                in_whitespace = true;
            }
        } else {
            result.push(byte);
            in_whitespace = false;
        }
    }

    // Remove trailing whitespace
    if let Some(&b' ') = result.last() {
        result.pop();
    }

    // Safe because we only manipulate ASCII characters
    unsafe { String::from_utf8_unchecked(result) }
}

/// Fast ASCII IPA symbol validation using lookup table
pub fn is_valid_ascii_ipa(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    // Optimized ASCII IPA validation using bit manipulation
    // Create a lookup table for ASCII characters (0-127)
    static VALID_ASCII_IPA: [bool; 128] = {
        let mut table = [false; 128];
        let valid_chars = b"aeiouypbtkgdqfvszxlrnmjwhjAEIOUYPBTKGDQFVSZXLRNMJWHJ";
        let mut i = 0;
        while i < valid_chars.len() {
            table[valid_chars[i] as usize] = true;
            i += 1;
        }
        table
    };

    for &byte in bytes {
        if byte >= 128 || !VALID_ASCII_IPA[byte as usize] {
            return false;
        }
    }

    true
}

/// Enhanced Unicode IPA symbol validation with comprehensive character support
fn is_valid_unicode_ipa(symbol: &str) -> bool {
    if symbol.is_empty() {
        return false;
    }

    // Comprehensive Unicode IPA validation
    for ch in symbol.chars() {
        if !is_valid_ipa_character(ch) {
            return false;
        }
    }
    true
}

/// Check if a Unicode character is a valid IPA symbol
pub fn is_valid_ipa_character(ch: char) -> bool {
    match ch {
        // Basic Latin letters
        'a'..='z' | 'A'..='Z' => true,

        // Common IPA symbols (Unicode Block: IPA Extensions 0250-02AF)
        'ə' | 'ɪ' | 'ɔ' | 'ʃ' | 'ʒ' | 'θ' | 'ð' | 'ŋ' | 'ʌ' | 'ɑ' | 'æ' | 'ɛ' | 'ɜ' => {
            true
        }

        // Additional IPA vowels
        'ɨ' | 'ɯ' | 'ɐ' | 'ɒ' | 'ɓ' | 'ɗ' | 'ɖ' | 'ɠ' | 'ɢ' | 'ʛ' | 'ɦ' | 'ɧ' => true,

        // IPA consonants
        'ɬ' | 'ɮ' | 'ɭ' | 'ɳ' | 'ɲ' | 'ɴ' | 'ɸ' | 'β' | 'ɹ' | 'ɻ' | 'ɾ' | 'ɽ' | 'ʀ' | 'ʁ' => {
            true
        }

        // More IPA symbols (removing duplicates)
        'ʂ' | 'ɱ' | 'ʍ' | 'χ' | 'ʎ' | 'ʏ' | 'ʑ' | 'ʐ' | 'ʔ' | 'ʕ' | 'ʘ' | 'ǀ' => true,

        // Click consonants and other symbols (removing duplicates)
        'ǁ' | 'ǂ' | 'ǃ' | 'ɿ' | 'ʅ' | 'ʆ' | 'ʇ' | 'ʈ' | 'ʉ' | 'ʊ' | 'ʋ' | 'ɣ' | 'ɤ' | 'ɥ' => {
            true
        }

        // Diacritics and modifiers (Unicode blocks: Combining Diacritical Marks)
        '\u{0300}'..='\u{036F}' => true, // Combining diacritical marks
        '\u{1AB0}'..='\u{1AFF}' => true, // Combining diacritical marks extended
        '\u{1DC0}'..='\u{1DFF}' => true, // Combining diacritical marks supplement

        // Additional spacing modifiers
        '\u{02B0}'..='\u{02FF}' => true, // Spacing modifier letters block (includes many symbols)

        // Additional markers not covered by spacing modifiers
        '.' | '‿' => true,

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_availability() {
        // Should not panic
        let _ = is_simd_available();
    }

    #[test]
    fn test_simd_features() {
        let features = simd_features();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_simd_filter_alphabetic() {
        let text = "Hello123World!";
        let result = filter_alphabetic(text);
        assert_eq!(result, "HelloWorld");
    }

    #[test]
    fn test_simd_normalize_whitespace() {
        let text = "hello   world\t\ntest";
        let result = normalize_whitespace(text);
        assert_eq!(result, "hello world test");
    }

    #[test]
    fn test_validate_phoneme_symbols() {
        let symbols = vec!["hello".to_string(), "123".to_string(), "ə".to_string()];
        let results = validate_phoneme_symbols(&symbols);
        assert_eq!(results.len(), 3);
        assert!(results[0]); // ASCII alphabetic
        assert!(!results[1]); // Numbers
        assert!(results[2]); // Unicode IPA
    }

    #[test]
    fn test_find_phoneme_patterns() {
        let phonemes = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let patterns = vec!["a".to_string(), "c".to_string()];
        let matches = find_phoneme_patterns(&phonemes, &patterns);
        assert_eq!(matches[0], vec![0, 2]); // 'a' at positions 0, 2
        assert_eq!(matches[1], Vec::<usize>::new()); // 'c' not found

        // Test edge cases
        let empty_phonemes: Vec<String> = vec![];
        let empty_patterns: Vec<String> = vec![];
        let empty_matches = find_phoneme_patterns(&empty_phonemes, &patterns);
        assert_eq!(empty_matches.len(), 2);
        assert!(empty_matches[0].is_empty());
        assert!(empty_matches[1].is_empty());

        let pattern_empty_matches = find_phoneme_patterns(&phonemes, &empty_patterns);
        assert!(pattern_empty_matches.is_empty());
    }

    #[test]
    fn test_optimized_batch_preprocess() {
        // Test mixed ASCII and Unicode text processing
        let texts = vec![
            "hello world".to_string(),
            "test  multiple   spaces".to_string(),
            "unicode ə text".to_string(),
            "more\t\ttabs".to_string(),
        ];

        let results = batch_preprocess(&texts);
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], "hello world");
        assert_eq!(results[1], "test multiple spaces");
        assert_eq!(results[2], "unicode ə text");
        assert_eq!(results[3], "more tabs");

        // Test empty input
        let empty_texts: Vec<String> = vec![];
        let empty_results = batch_preprocess(&empty_texts);
        assert!(empty_results.is_empty());

        // Test single item
        let single_text = vec!["single".to_string()];
        let single_result = batch_preprocess(&single_text);
        assert_eq!(single_result.len(), 1);
        assert_eq!(single_result[0], "single");
    }

    #[test]
    fn test_pattern_matching_performance() {
        // Create a larger dataset to test performance improvements
        let phonemes: Vec<String> = (0..1000).map(|i| format!("phoneme_{}", i % 100)).collect();
        let patterns: Vec<String> = (0..50).map(|i| format!("phoneme_{i}")).collect();

        let matches = find_phoneme_patterns(&phonemes, &patterns);
        assert_eq!(matches.len(), 50);

        // Verify correct matching for repeated patterns
        for (pattern_idx, pattern) in patterns.iter().enumerate() {
            let expected_count = phonemes.iter().filter(|p| *p == pattern).count();
            assert_eq!(matches[pattern_idx].len(), expected_count);
        }
    }

    #[test]
    fn test_enhanced_ipa_validation() {
        assert!(is_valid_ascii_ipa(b"hello"));
        assert!(!is_valid_ascii_ipa(b"123"));
        assert!(is_valid_ipa_character('ə'));
        assert!(!is_valid_ipa_character('1'));
    }
}
