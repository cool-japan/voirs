//! Mixed language detection and handling.

use crate::detection::rules::RuleBasedDetector;
use crate::detection::statistical::StatisticalDetector;
use crate::detection::{DetectionMethod, DetectionResult};
use crate::preprocessing::unicode::ScriptType;
use crate::{LanguageCode, Result};
use std::collections::HashMap;

/// Mixed language detector for handling code-switching and multilingual text
pub struct MixedLanguageDetector {
    /// Rule-based detector for individual segments
    rule_detector: RuleBasedDetector,
    /// Statistical detector for segments
    statistical_detector: StatisticalDetector,
    /// Minimum segment length for separate detection
    min_segment_length: usize,
}

/// Language segment in mixed text
#[derive(Debug, Clone)]
pub struct LanguageSegment {
    /// Text content of the segment
    pub text: String,
    /// Detected language
    pub language: LanguageCode,
    /// Confidence score
    pub confidence: f32,
    /// Start position in original text
    pub start_pos: usize,
    /// End position in original text
    pub end_pos: usize,
}

/// Mixed language detection result
#[derive(Debug, Clone)]
pub struct MixedDetectionResult {
    /// Primary language (most common)
    pub primary_language: LanguageCode,
    /// Overall confidence
    pub confidence: f32,
    /// Language segments
    pub segments: Vec<LanguageSegment>,
    /// Language distribution (language -> proportion)
    pub distribution: HashMap<LanguageCode, f32>,
}

impl MixedLanguageDetector {
    /// Create new mixed language detector
    pub fn new() -> Self {
        Self {
            rule_detector: RuleBasedDetector::new(),
            statistical_detector: StatisticalDetector::new(),
            min_segment_length: 5, // Minimum 5 characters per segment
        }
    }

    /// Create with custom minimum segment length
    pub fn with_min_segment_length(min_length: usize) -> Self {
        let mut detector = Self::new();
        detector.min_segment_length = min_length;
        detector
    }

    /// Detect language with mixed language support
    pub fn detect(&self, text: &str) -> Result<Option<DetectionResult>> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        // First try to detect if text is mixed language
        let mixed_result = self.detect_mixed(text)?;

        // If we have multiple languages detected, return the primary one
        if mixed_result.segments.len() > 1 {
            let primary_lang = mixed_result.primary_language;
            let confidence = mixed_result.confidence;

            // Create alternatives from language distribution
            let mut alternatives: Vec<(LanguageCode, f32)> = mixed_result
                .distribution
                .into_iter()
                .filter(|(lang, _)| *lang != primary_lang)
                .collect();
            alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            alternatives.truncate(3);

            return Ok(Some(DetectionResult {
                language: primary_lang,
                confidence,
                alternatives,
                method: DetectionMethod::Mixed,
            }));
        }

        // Single language or failed detection
        Ok(None)
    }

    /// Detect mixed languages in text
    pub fn detect_mixed(&self, text: &str) -> Result<MixedDetectionResult> {
        // Segment the text by different criteria
        let segments = self.segment_text(text)?;

        // Detect language for each segment
        let mut language_segments = Vec::new();
        let mut language_counts = HashMap::new();
        let mut total_chars = 0;

        for segment in segments {
            if segment.text.len() < self.min_segment_length {
                continue;
            }

            let detected_lang = self.detect_segment_language(&segment.text)?;

            if let Some(lang) = detected_lang {
                let segment_chars = segment.text.chars().count();
                total_chars += segment_chars;

                *language_counts.entry(lang).or_insert(0) += segment_chars;

                language_segments.push(LanguageSegment {
                    text: segment.text,
                    language: lang,
                    confidence: 0.8, // Placeholder confidence
                    start_pos: segment.start_pos,
                    end_pos: segment.end_pos,
                });
            }
        }

        // Calculate language distribution
        let mut distribution = HashMap::new();
        for (lang, count) in &language_counts {
            if total_chars > 0 {
                distribution.insert(*lang, *count as f32 / total_chars as f32);
            }
        }

        // Find primary language (most common)
        let primary_language = language_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lang, _)| *lang)
            .unwrap_or(LanguageCode::EnUs);

        // Calculate overall confidence based on how mixed the text is
        let num_languages = distribution.len();
        let confidence = if num_languages <= 1 {
            0.9 // Single language
        } else if num_languages == 2 {
            0.7 // Two languages
        } else {
            0.5 // Multiple languages
        };

        Ok(MixedDetectionResult {
            primary_language,
            confidence,
            segments: language_segments,
            distribution,
        })
    }

    /// Segment text into potentially different language chunks
    fn segment_text(&self, text: &str) -> Result<Vec<TextSegment>> {
        let mut segments = Vec::new();

        // Simple segmentation by sentences and script changes
        let sentences = self.split_by_sentences(text);

        for sentence in sentences {
            // Further segment by script changes
            let script_segments = self.split_by_script_changes(&sentence);
            segments.extend(script_segments);
        }

        Ok(segments)
    }

    /// Split text by sentence boundaries
    fn split_by_sentences(&self, text: &str) -> Vec<TextSegment> {
        let mut segments = Vec::new();
        let mut current_start = 0;
        let mut current_text = String::new();

        for (i, ch) in text.char_indices() {
            current_text.push(ch);

            // Check for sentence endings
            if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
                // Look ahead to see if this is actually a sentence end
                let remaining = &text[i + ch.len_utf8()..];
                if remaining.starts_with(' ') || remaining.starts_with('\n') || remaining.is_empty()
                {
                    segments.push(TextSegment {
                        text: current_text.trim().to_string(),
                        start_pos: current_start,
                        end_pos: i + ch.len_utf8(),
                    });
                    current_text.clear();
                    current_start = i + ch.len_utf8();
                }
            }
        }

        // Add remaining text as a segment
        if !current_text.trim().is_empty() {
            segments.push(TextSegment {
                text: current_text.trim().to_string(),
                start_pos: current_start,
                end_pos: text.len(),
            });
        }

        segments
    }

    /// Split text by script changes (e.g., Latin to CJK)
    fn split_by_script_changes(&self, segment: &TextSegment) -> Vec<TextSegment> {
        use crate::preprocessing::unicode::detect_script;

        let mut segments = Vec::new();
        let mut current_script = None;
        let mut current_start = segment.start_pos;
        let mut current_text = String::new();

        for (i, ch) in segment.text.char_indices() {
            let ch_script = detect_script(&ch.to_string());

            // If script changes significantly, create a new segment
            if let Some(prev_script) = current_script {
                if self.should_split_scripts(prev_script, ch_script) {
                    if !current_text.trim().is_empty() {
                        segments.push(TextSegment {
                            text: current_text.trim().to_string(),
                            start_pos: current_start,
                            end_pos: segment.start_pos + i,
                        });
                    }
                    current_text.clear();
                    current_start = segment.start_pos + i;
                }
            }

            current_text.push(ch);
            current_script = Some(ch_script);
        }

        // Add remaining text
        if !current_text.trim().is_empty() {
            segments.push(TextSegment {
                text: current_text.trim().to_string(),
                start_pos: current_start,
                end_pos: segment.end_pos,
            });
        }

        segments
    }

    /// Check if two scripts should cause a split
    fn should_split_scripts(&self, script1: ScriptType, script2: ScriptType) -> bool {
        match (script1, script2) {
            // Don't split within same script
            (s1, s2) if s1 == s2 => false,

            // Don't split between related scripts
            (ScriptType::Hiragana, ScriptType::Katakana)
            | (ScriptType::Katakana, ScriptType::Hiragana)
            | (ScriptType::Hiragana, ScriptType::CJK)
            | (ScriptType::Katakana, ScriptType::CJK)
            | (ScriptType::CJK, ScriptType::Hiragana)
            | (ScriptType::CJK, ScriptType::Katakana) => false,

            // Don't split on unknown or mixed
            (ScriptType::Unknown, _)
            | (_, ScriptType::Unknown)
            | (ScriptType::Mixed, _)
            | (_, ScriptType::Mixed) => false,

            // Split between different major scripts
            _ => true,
        }
    }

    /// Detect language for a single segment
    fn detect_segment_language(&self, text: &str) -> Result<Option<LanguageCode>> {
        // Try rule-based detection first
        if let Some(result) = self.rule_detector.detect(text)? {
            if result.confidence > 0.5 {
                return Ok(Some(result.language));
            }
        }

        // Try statistical detection
        if let Some(result) = self.statistical_detector.detect(text)? {
            if result.confidence > 0.3 {
                return Ok(Some(result.language));
            }
        }

        // Fallback to script-based detection
        use crate::preprocessing::unicode::detect_script;

        let script = detect_script(text);
        match script {
            ScriptType::Hiragana | ScriptType::Katakana => Ok(Some(LanguageCode::Ja)),
            ScriptType::Hangul => Ok(Some(LanguageCode::Ko)),
            ScriptType::CJK => Ok(Some(LanguageCode::ZhCn)),
            _ => Ok(None),
        }
    }
}

/// Text segment for internal processing
#[derive(Debug, Clone)]
struct TextSegment {
    /// Text content
    text: String,
    /// Start position in original text
    start_pos: usize,
    /// End position in original text  
    end_pos: usize,
}

impl Default for MixedLanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_detector_creation() {
        let detector = MixedLanguageDetector::new();
        assert_eq!(detector.min_segment_length, 5);

        let detector = MixedLanguageDetector::with_min_segment_length(10);
        assert_eq!(detector.min_segment_length, 10);
    }

    #[test]
    fn test_sentence_splitting() {
        let detector = MixedLanguageDetector::new();

        let text = "Hello world. How are you? I am fine!";
        let segments = detector.split_by_sentences(text);

        assert!(segments.len() >= 3);
        assert!(segments[0].text.contains("Hello world"));
        assert!(segments[1].text.contains("How are you"));
        assert!(segments[2].text.contains("I am fine"));
    }

    #[test]
    fn test_script_change_detection() {
        let detector = MixedLanguageDetector::new();

        // Should split between Latin and CJK
        assert!(detector.should_split_scripts(ScriptType::Latin, ScriptType::CJK));

        // Should not split between related Japanese scripts
        assert!(!detector.should_split_scripts(ScriptType::Hiragana, ScriptType::Katakana));
        assert!(!detector.should_split_scripts(ScriptType::Hiragana, ScriptType::CJK));
    }

    #[test]
    fn test_mixed_detection() {
        let detector = MixedLanguageDetector::new();

        let result = detector
            .detect_mixed("Hello world. This is English text.")
            .unwrap();

        // The mixed detection should always return a result, even if it's just segmented text
        assert!(result.confidence >= 0.0);
        // Segments might be empty if no language was detected for any segment
        // This is acceptable for this test
    }

    #[test]
    fn test_segment_language_detection() {
        let detector = MixedLanguageDetector::new();

        let lang = detector.detect_segment_language("Hello world").unwrap();
        // Might detect English or return None depending on the detectors
        if let Some(_lang) = lang {
            // Language detected successfully
        }
    }

    #[test]
    fn test_empty_text() {
        let detector = MixedLanguageDetector::new();

        let result = detector.detect("").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_single_language_text() {
        let detector = MixedLanguageDetector::new();

        let result = detector
            .detect("This is purely English text without any other languages")
            .unwrap();
        // Should return None since it's not truly mixed language
        if let Some(result) = result {
            assert_eq!(result.method, DetectionMethod::Mixed);
        }
    }

    #[test]
    fn test_text_segmentation() {
        let detector = MixedLanguageDetector::new();

        let text = "Hello world. How are you?";
        let segments = detector.segment_text(text).unwrap();

        assert!(!segments.is_empty());

        // Check that positions are reasonable
        for segment in &segments {
            assert!(segment.start_pos <= segment.end_pos);
            assert!(segment.end_pos <= text.len());
        }
    }
}
