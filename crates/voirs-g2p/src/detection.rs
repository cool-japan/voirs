//! Language detection for G2P preprocessing.

use crate::preprocessing::unicode::ScriptType;
use crate::{LanguageCode, Result};
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod mixed;
pub mod rules;
pub mod statistical;

/// Language detection confidence threshold
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.75;

/// Language detector with multiple detection strategies
pub struct LanguageDetector {
    /// Rule-based detector
    rule_detector: rules::RuleBasedDetector,
    /// Statistical detector
    statistical_detector: statistical::StatisticalDetector,
    /// Mixed language detector
    mixed_detector: mixed::MixedLanguageDetector,
    /// Confidence threshold for detection
    confidence_threshold: f32,
}

/// Language detection result
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Detected language
    pub language: LanguageCode,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Alternative languages with their scores
    pub alternatives: Vec<(LanguageCode, f32)>,
    /// Detection method used
    pub method: DetectionMethod,
}

/// Detection method used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Script-based detection
    Script,
    /// Rule-based detection
    Rules,
    /// Statistical detection
    Statistical,
    /// Mixed language detection
    Mixed,
    /// Fallback to default
    Fallback,
}

impl LanguageDetector {
    /// Create new language detector
    pub fn new() -> Self {
        Self {
            rule_detector: rules::RuleBasedDetector::new(),
            statistical_detector: statistical::StatisticalDetector::new(),
            mixed_detector: mixed::MixedLanguageDetector::new(),
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        }
    }

    /// Create detector with custom confidence threshold
    pub fn with_threshold(threshold: f32) -> Self {
        let mut detector = Self::new();
        detector.confidence_threshold = threshold;
        detector
    }

    /// Detect language of text
    pub fn detect(&self, text: &str) -> Result<DetectionResult> {
        if text.trim().is_empty() {
            return Ok(DetectionResult {
                language: LanguageCode::EnUs, // Default fallback
                confidence: 0.0,
                alternatives: vec![],
                method: DetectionMethod::Fallback,
            });
        }

        // Try different detection methods in order of reliability

        // 1. Script-based detection (fastest and most reliable for certain scripts)
        if let Some(result) = self.detect_by_script(text)? {
            if result.confidence >= self.confidence_threshold {
                return Ok(result);
            }
        }

        // 2. Rule-based detection
        if let Some(result) = self.rule_detector.detect(text)? {
            if result.confidence >= self.confidence_threshold {
                return Ok(result);
            }
        }

        // 3. Statistical detection
        if let Some(result) = self.statistical_detector.detect(text)? {
            if result.confidence >= self.confidence_threshold {
                return Ok(result);
            }
        }

        // 4. Mixed language detection
        if let Some(result) = self.mixed_detector.detect(text)? {
            if result.confidence >= self.confidence_threshold {
                return Ok(result);
            }
        }

        // 5. Fallback to English
        Ok(DetectionResult {
            language: LanguageCode::EnUs,
            confidence: 0.1,
            alternatives: vec![],
            method: DetectionMethod::Fallback,
        })
    }

    /// Detect language based on script type
    fn detect_by_script(&self, text: &str) -> Result<Option<DetectionResult>> {
        use crate::preprocessing::unicode::detect_script;

        let script = detect_script(text);

        let (language, confidence) = match script {
            ScriptType::Hiragana | ScriptType::Katakana => (LanguageCode::Ja, 0.95),
            ScriptType::Hangul => (LanguageCode::Ko, 0.95),
            ScriptType::CJK => (LanguageCode::ZhCn, 0.80), // Could be Chinese, Japanese, or Korean
            ScriptType::Cyrillic => {
                // Need further analysis to distinguish between Cyrillic languages
                // For now, we'll return None and let other detectors handle it
                return Ok(None);
            }
            ScriptType::Arabic => {
                // Need further analysis to distinguish between Arabic languages
                return Ok(None);
            }
            ScriptType::Latin => {
                // Latin script is used by many languages, needs further analysis
                return Ok(None);
            }
            _ => return Ok(None),
        };

        Ok(Some(DetectionResult {
            language,
            confidence,
            alternatives: vec![],
            method: DetectionMethod::Script,
        }))
    }

    /// Get all possible languages for text (useful for mixed language content)
    pub fn detect_all(&self, text: &str) -> Result<Vec<DetectionResult>> {
        let mut results = Vec::new();

        // Collect results from all detectors
        if let Some(result) = self.detect_by_script(text)? {
            results.push(result);
        }

        if let Some(result) = self.rule_detector.detect(text)? {
            results.push(result);
        }

        if let Some(result) = self.statistical_detector.detect(text)? {
            results.push(result);
        }

        // Sort by confidence
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(results)
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Common language indicators for quick detection
static LANGUAGE_INDICATORS: OnceLock<HashMap<LanguageCode, Vec<&'static str>>> = OnceLock::new();

fn init_language_indicators() -> &'static HashMap<LanguageCode, Vec<&'static str>> {
    LANGUAGE_INDICATORS.get_or_init(|| {
        let mut map = HashMap::new();

        // English indicators
        map.insert(
            LanguageCode::EnUs,
            vec![
                "the", "and", "of", "to", "a", "in", "is", "it", "you", "that", "he", "was", "for",
                "on", "are", "as", "with", "his", "they", "I",
            ],
        );

        // German indicators
        map.insert(
            LanguageCode::De,
            vec![
                "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf",
                "für", "ist", "im", "dem", "nicht", "ein", "eine", "als",
            ],
        );

        // French indicators
        map.insert(
            LanguageCode::Fr,
            vec![
                "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour",
                "dans", "ce", "son", "une", "sur", "avec", "ne", "se",
            ],
        );

        // Spanish indicators
        map.insert(
            LanguageCode::Es,
            vec![
                "el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le",
                "da", "su", "por", "son", "con", "para",
            ],
        );

        // Italian indicators
        map.insert(
            LanguageCode::It,
            vec![
                "il", "di", "che", "e", "la", "per", "un", "in", "con", "del", "da", "è", "le",
                "dei", "nel", "una", "alla", "delle", "non", "sono", "molto", "più", "quando",
                "dove",
            ],
        );

        // Portuguese indicators
        map.insert(
            LanguageCode::Pt,
            vec![
                "o", "de", "e", "a", "que", "do", "da", "em", "um", "para", "é", "com", "não",
                "uma", "os", "no", "se", "na", "por", "mais", "das", "dos", "como", "mas", "foi",
                "ao",
            ],
        );

        // Japanese indicators (hiragana)
        map.insert(
            LanguageCode::Ja,
            vec![
                "の",
                "に",
                "は",
                "を",
                "た",
                "が",
                "で",
                "て",
                "と",
                "し",
                "れ",
                "さ",
                "ある",
                "いる",
                "も",
                "する",
                "から",
                "な",
                "こと",
                "として",
            ],
        );

        // Chinese indicators (simplified)
        map.insert(
            LanguageCode::ZhCn,
            vec![
                "的", "一", "是", "在", "不", "了", "有", "和", "人", "这", "中", "大", "为", "上",
                "个", "国", "我", "以", "要", "他",
            ],
        );

        // Korean indicators
        map.insert(
            LanguageCode::Ko,
            vec![
                "이", "의", "가", "을", "는", "에", "하", "고", "를", "으로", "로", "에서", "와",
                "한", "까지", "도", "만", "부터", "께", "에게",
            ],
        );

        map
    })
}

/// Get language indicators for quick detection
pub fn get_language_indicators(language: LanguageCode) -> Option<&'static Vec<&'static str>> {
    init_language_indicators().get(&language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detector_creation() {
        let detector = LanguageDetector::new();
        assert_eq!(detector.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD);

        let detector = LanguageDetector::with_threshold(0.9);
        assert_eq!(detector.confidence_threshold, 0.9);
    }

    #[test]
    fn test_script_based_detection() {
        let detector = LanguageDetector::new();

        // Test Japanese hiragana
        let result = detector.detect("こんにちは").unwrap();
        assert_eq!(result.language, LanguageCode::Ja);
        assert!(result.confidence > 0.9);
        assert_eq!(result.method, DetectionMethod::Script);

        // Test Korean
        let result = detector.detect("안녕하세요").unwrap();
        assert_eq!(result.language, LanguageCode::Ko);
        assert!(result.confidence > 0.9);
        assert_eq!(result.method, DetectionMethod::Script);
    }

    #[test]
    fn test_empty_text_detection() {
        let detector = LanguageDetector::new();

        let result = detector.detect("").unwrap();
        assert_eq!(result.language, LanguageCode::EnUs);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.method, DetectionMethod::Fallback);
    }

    #[test]
    fn test_language_indicators() {
        let en_indicators = get_language_indicators(LanguageCode::EnUs).unwrap();
        assert!(en_indicators.contains(&"the"));
        assert!(en_indicators.contains(&"and"));

        let de_indicators = get_language_indicators(LanguageCode::De).unwrap();
        assert!(de_indicators.contains(&"der"));
        assert!(de_indicators.contains(&"die"));
    }

    #[test]
    fn test_detect_all() {
        let detector = LanguageDetector::new();

        let results = detector.detect_all("Hello world").unwrap();

        // Results might be empty if no detector can confidently identify the language
        // This is acceptable behavior

        // If we have results, they should be sorted by confidence
        for i in 1..results.len() {
            assert!(results[i - 1].confidence >= results[i].confidence);
        }
    }
}
