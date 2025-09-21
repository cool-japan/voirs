//! Unicode normalization and text cleaning for G2P preprocessing.

use crate::Result;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Normalize text using Unicode NFC normalization
pub fn normalize_text(text: &str) -> Result<String> {
    // Normalize to NFC form for consistent processing
    let normalized = text.nfc().collect::<String>();

    // Additional cleaning steps
    let cleaned = clean_text(&normalized)?;

    Ok(cleaned)
}

/// Clean text by removing unwanted characters and fixing encoding issues
fn clean_text(text: &str) -> Result<String> {
    let mut result = String::new();

    for grapheme in text.graphemes(true) {
        match grapheme {
            // Replace common problematic characters
            "‚" | "'" => result.push('\''),
            "\u{201c}" | "\u{201d}" => result.push('"'),
            "—" | "–" => result.push('-'),
            "…" => result.push_str("..."),

            // Keep regular characters
            g if is_valid_text_char(g) => result.push_str(g),

            // Replace other characters with space
            _ => result.push(' '),
        }
    }

    // Collapse multiple spaces
    Ok(collapse_spaces(&result))
}

/// Check if a grapheme is valid for text processing
fn is_valid_text_char(grapheme: &str) -> bool {
    if grapheme.len() == 1 {
        let ch = grapheme.chars().next().unwrap();
        ch.is_alphabetic()
            || ch.is_numeric()
            || ch.is_whitespace()
            || matches!(
                ch,
                '.' | ','
                    | '!'
                    | '?'
                    | ';'
                    | ':'
                    | '\''
                    | '"'
                    | '-'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
            )
    } else {
        // Multi-character graphemes (like accented characters)
        grapheme
            .chars()
            .all(|c| c.is_alphabetic() || c.is_numeric())
    }
}

/// Collapse multiple consecutive spaces into single spaces
fn collapse_spaces(text: &str) -> String {
    let mut result = String::new();
    let mut last_was_space = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(ch);
            last_was_space = false;
        }
    }

    result.trim().to_string()
}

/// Detect script type of text
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScriptType {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Hebrew,
    CJK,
    Hiragana,
    Katakana,
    Hangul,
    Mixed,
    Unknown,
}

/// Detect the primary script type of the text
pub fn detect_script(text: &str) -> ScriptType {
    let mut script_counts = std::collections::HashMap::new();
    let mut total_chars = 0;

    for ch in text.chars() {
        if ch.is_alphabetic() {
            total_chars += 1;

            let script = match ch as u32 {
                // Latin
                0x0041..=0x007A | 0x00C0..=0x00FF | 0x0100..=0x017F => ScriptType::Latin,
                // Cyrillic
                0x0400..=0x04FF => ScriptType::Cyrillic,
                // Greek
                0x0370..=0x03FF => ScriptType::Greek,
                // Arabic
                0x0600..=0x06FF => ScriptType::Arabic,
                // Hebrew
                0x0590..=0x05FF => ScriptType::Hebrew,
                // CJK
                0x4E00..=0x9FFF => ScriptType::CJK,
                // Hiragana
                0x3040..=0x309F => ScriptType::Hiragana,
                // Katakana
                0x30A0..=0x30FF => ScriptType::Katakana,
                // Hangul
                0xAC00..=0xD7AF => ScriptType::Hangul,
                _ => ScriptType::Unknown,
            };

            *script_counts.entry(script).or_insert(0) += 1;
        }
    }

    if total_chars == 0 {
        return ScriptType::Unknown;
    }

    // Check if it's mixed script
    let is_mixed = script_counts.len() > 2;

    let dominant_script = script_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(script, _)| script)
        .unwrap_or(ScriptType::Unknown);

    if is_mixed {
        ScriptType::Mixed
    } else {
        dominant_script
    }
}

/// Filter out emoji and other symbols
pub fn filter_symbols(text: &str) -> String {
    text.chars()
        .filter(|ch| {
            // Keep letters, numbers, and basic punctuation
            ch.is_alphabetic()
                || ch.is_numeric()
                || ch.is_whitespace()
                || matches!(
                    *ch,
                    '.' | ','
                        | '!'
                        | '?'
                        | ';'
                        | ':'
                        | '\''
                        | '"'
                        | '-'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                )
        })
        .collect()
}

/// Detect if text contains RTL (right-to-left) characters
pub fn is_rtl_text(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(ch as u32,
            0x0590..=0x05FF | // Hebrew
            0x0600..=0x06FF | // Arabic
            0x0750..=0x077F | // Arabic Supplement
            0x08A0..=0x08FF   // Arabic Extended-A
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        // Test NFC normalization
        let text = "café"; // This might be composed differently
        let result = normalize_text(text).unwrap();
        assert!(!result.is_empty());

        // Test with special characters
        let text = "\u{201c}Hello World\u{201d} \u{2014} it's great!";
        let result = normalize_text(text).unwrap();
        assert_eq!(result, "\"Hello World\" - it's great!");
    }

    #[test]
    fn test_collapse_spaces() {
        assert_eq!(collapse_spaces("hello    world"), "hello world");
        assert_eq!(collapse_spaces("  hello  world  "), "hello world");
        assert_eq!(collapse_spaces("hello\n\t  world"), "hello world");
    }

    #[test]
    fn test_script_detection() {
        assert_eq!(detect_script("Hello World"), ScriptType::Latin);
        assert_eq!(detect_script("こんにちは"), ScriptType::Hiragana);
        assert_eq!(detect_script("カタカナ"), ScriptType::Katakana);
        assert_eq!(detect_script("한글"), ScriptType::Hangul);
        assert_eq!(detect_script("中文"), ScriptType::CJK);
        assert_eq!(detect_script("Привет"), ScriptType::Cyrillic);
    }

    #[test]
    fn test_filter_symbols() {
        let text = "Hello 👋 World! 🌍";
        let result = filter_symbols(text);
        assert_eq!(result, "Hello  World! ");
    }

    #[test]
    fn test_rtl_detection() {
        assert!(is_rtl_text("שלום"));
        assert!(is_rtl_text("مرحبا"));
        assert!(!is_rtl_text("Hello"));
    }

    #[test]
    fn test_valid_text_char() {
        assert!(is_valid_text_char("a"));
        assert!(is_valid_text_char("A"));
        assert!(is_valid_text_char("1"));
        assert!(is_valid_text_char(" "));
        assert!(is_valid_text_char("."));
        assert!(is_valid_text_char("é"));
        assert!(!is_valid_text_char("👋"));
    }
}
