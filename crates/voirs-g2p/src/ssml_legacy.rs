//! SSML (Speech Synthesis Markup Language) integration for G2P conversion.

use crate::{G2pError, LanguageCode, Phoneme, Result};
use std::collections::HashMap;
use std::fmt;

/// SSML element types
#[derive(Debug, Clone, PartialEq)]
pub enum SsmlElement {
    /// Root speak element
    Speak {
        language: Option<LanguageCode>,
        content: Vec<SsmlElement>,
    },
    /// Text content
    Text(String),
    /// Phoneme override
    Phoneme {
        alphabet: String,
        ph: String,
        text: String,
    },
    /// Language switching
    Lang {
        lang: LanguageCode,
        content: Vec<SsmlElement>,
    },
    /// Emphasis
    Emphasis {
        level: EmphasisLevel,
        content: Vec<SsmlElement>,
    },
    /// Break/pause
    Break {
        time: Option<String>,
        strength: Option<BreakStrength>,
    },
    /// Say-as for specific pronunciation
    SayAs {
        interpret_as: InterpretAs,
        format: Option<String>,
        content: String,
    },
    /// Prosody control
    Prosody {
        rate: Option<String>,
        pitch: Option<String>,
        volume: Option<String>,
        content: Vec<SsmlElement>,
    },
}

/// Emphasis levels
#[derive(Debug, Clone, PartialEq)]
pub enum EmphasisLevel {
    None,
    Reduced,
    Moderate,
    Strong,
}

impl fmt::Display for EmphasisLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmphasisLevel::None => write!(f, "none"),
            EmphasisLevel::Reduced => write!(f, "reduced"),
            EmphasisLevel::Moderate => write!(f, "moderate"),
            EmphasisLevel::Strong => write!(f, "strong"),
        }
    }
}

/// Break strength levels
#[derive(Debug, Clone, PartialEq)]
pub enum BreakStrength {
    None,
    XWeak,
    Weak,
    Medium,
    Strong,
    XStrong,
}

impl fmt::Display for BreakStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BreakStrength::None => write!(f, "none"),
            BreakStrength::XWeak => write!(f, "x-weak"),
            BreakStrength::Weak => write!(f, "weak"),
            BreakStrength::Medium => write!(f, "medium"),
            BreakStrength::Strong => write!(f, "strong"),
            BreakStrength::XStrong => write!(f, "x-strong"),
        }
    }
}

/// Interpretation types for say-as
#[derive(Debug, Clone, PartialEq)]
pub enum InterpretAs {
    Characters,
    SpellOut,
    Cardinal,
    Ordinal,
    Digits,
    Fraction,
    Unit,
    Date,
    Time,
    Telephone,
    Address,
}

impl fmt::Display for InterpretAs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpretAs::Characters => write!(f, "characters"),
            InterpretAs::SpellOut => write!(f, "spell-out"),
            InterpretAs::Cardinal => write!(f, "cardinal"),
            InterpretAs::Ordinal => write!(f, "ordinal"),
            InterpretAs::Digits => write!(f, "digits"),
            InterpretAs::Fraction => write!(f, "fraction"),
            InterpretAs::Unit => write!(f, "unit"),
            InterpretAs::Date => write!(f, "date"),
            InterpretAs::Time => write!(f, "time"),
            InterpretAs::Telephone => write!(f, "telephone"),
            InterpretAs::Address => write!(f, "address"),
        }
    }
}

/// SSML processor for G2P conversion
pub struct SsmlProcessor {
    /// Custom phoneme overrides
    phoneme_overrides: HashMap<String, Vec<Phoneme>>,
}

impl SsmlProcessor {
    /// Create a new SSML processor
    pub fn new() -> Self {
        Self {
            phoneme_overrides: HashMap::new(),
        }
    }

    /// Add custom phoneme override
    pub fn add_phoneme_override(&mut self, text: String, phonemes: Vec<Phoneme>) {
        self.phoneme_overrides.insert(text, phonemes);
    }

    /// Parse SSML text into structured elements
    pub fn parse_ssml(&self, ssml_text: &str) -> Result<SsmlElement> {
        // Simple XML-like parsing (for now, we'll implement a basic parser)
        // In a full implementation, you'd use a proper XML parser like quick-xml

        if ssml_text.trim().starts_with("<speak") {
            self.parse_speak_element(ssml_text)
        } else {
            // Wrap in speak element if not present
            let wrapped = format!("<speak>{ssml_text}</speak>");
            self.parse_speak_element(&wrapped)
        }
    }

    /// Parse speak element
    fn parse_speak_element(&self, text: &str) -> Result<SsmlElement> {
        // Extract language attribute if present
        let language = self.extract_language_attribute(text)?;

        // Extract content between speak tags
        let content = self.extract_speak_content(text)?;

        // Parse inner content
        let elements = self.parse_inner_content(&content)?;

        Ok(SsmlElement::Speak {
            language,
            content: elements,
        })
    }

    /// Extract language attribute from speak tag
    fn extract_language_attribute(&self, text: &str) -> Result<Option<LanguageCode>> {
        // Simple regex-like extraction for xml:lang attribute
        if let Some(start) = text.find("xml:lang=\"") {
            let start_pos = start + "xml:lang=\"".len();
            if let Some(end_pos) = text[start_pos..].find('"') {
                let lang_str = &text[start_pos..start_pos + end_pos];
                return Ok(Some(self.parse_language_code(lang_str)?));
            }
        }
        Ok(None)
    }

    /// Extract content between speak tags
    fn extract_speak_content(&self, text: &str) -> Result<String> {
        // Find opening speak tag end
        if let Some(start) = text.find('>') {
            // Find closing speak tag
            if let Some(end) = text.rfind("</speak>") {
                if start + 1 < end {
                    return Ok(text[start + 1..end].to_string());
                }
            }
        }
        Err(G2pError::ConfigError("Invalid SSML structure".to_string()))
    }

    /// Parse inner content into SSML elements
    fn parse_inner_content(&self, content: &str) -> Result<Vec<SsmlElement>> {
        let mut elements = Vec::new();
        let mut current_pos = 0;

        while current_pos < content.len() {
            // Find next tag
            if let Some(tag_start) = content[current_pos..].find('<') {
                let absolute_tag_start = current_pos + tag_start;

                // Add text before tag if any
                if tag_start > 0 {
                    let text = content[current_pos..absolute_tag_start].trim().to_string();
                    if !text.is_empty() {
                        elements.push(SsmlElement::Text(text));
                    }
                }

                // Parse the tag
                if let Some(tag_end) = content[absolute_tag_start..].find('>') {
                    let absolute_tag_end = absolute_tag_start + tag_end + 1;
                    let tag_content = &content[absolute_tag_start..absolute_tag_end];

                    if tag_content.starts_with("<phoneme") {
                        let element =
                            self.parse_phoneme_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element.0);
                        current_pos = absolute_tag_end + element.1;
                    } else if tag_content.starts_with("<lang") {
                        let element =
                            self.parse_lang_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element.0);
                        current_pos = absolute_tag_end + element.1;
                    } else if tag_content.starts_with("<emphasis") {
                        let element =
                            self.parse_emphasis_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element.0);
                        current_pos = absolute_tag_end + element.1;
                    } else if tag_content.starts_with("<break") {
                        let element = self.parse_break_element(tag_content)?;
                        elements.push(element);
                        current_pos = absolute_tag_end;
                    } else if tag_content.starts_with("<say-as") {
                        let element =
                            self.parse_say_as_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element.0);
                        current_pos = absolute_tag_end + element.1;
                    } else if tag_content.starts_with("<prosody") {
                        let element =
                            self.parse_prosody_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element.0);
                        current_pos = absolute_tag_end + element.1;
                    } else {
                        // Unknown tag, skip
                        current_pos = absolute_tag_end;
                    }
                } else {
                    // Malformed tag, treat as text
                    let remaining_text = &content[current_pos..];
                    elements.push(SsmlElement::Text(remaining_text.to_string()));
                    break;
                }
            } else {
                // No more tags, add remaining text
                let remaining_text = content[current_pos..].trim().to_string();
                if !remaining_text.is_empty() {
                    elements.push(SsmlElement::Text(remaining_text));
                }
                break;
            }
        }

        Ok(elements)
    }

    /// Parse phoneme element
    fn parse_phoneme_element(&self, tag: &str, remaining: &str) -> Result<(SsmlElement, usize)> {
        // Extract attributes
        let alphabet = self
            .extract_attribute(tag, "alphabet")?
            .unwrap_or_else(|| "ipa".to_string());
        let ph = self.extract_attribute(tag, "ph")?.ok_or_else(|| {
            G2pError::ConfigError("phoneme element requires 'ph' attribute".to_string())
        })?;

        // Find closing tag and extract content
        if let Some(end_pos) = remaining.find("</phoneme>") {
            let text = remaining[..end_pos].to_string();
            let consumed = end_pos + "</phoneme>".len();

            Ok((SsmlElement::Phoneme { alphabet, ph, text }, consumed))
        } else {
            Err(G2pError::ConfigError(
                "Unclosed phoneme element".to_string(),
            ))
        }
    }

    /// Parse lang element
    fn parse_lang_element(&self, tag: &str, remaining: &str) -> Result<(SsmlElement, usize)> {
        let lang_str = self.extract_attribute(tag, "xml:lang")?.ok_or_else(|| {
            G2pError::ConfigError("lang element requires 'xml:lang' attribute".to_string())
        })?;
        let lang = self.parse_language_code(&lang_str)?;

        // Find closing tag and parse content
        if let Some(end_pos) = remaining.find("</lang>") {
            let content_str = &remaining[..end_pos];
            let content = self.parse_inner_content(content_str)?;
            let consumed = end_pos + "</lang>".len();

            Ok((SsmlElement::Lang { lang, content }, consumed))
        } else {
            Err(G2pError::ConfigError("Unclosed lang element".to_string()))
        }
    }

    /// Parse emphasis element
    fn parse_emphasis_element(&self, tag: &str, remaining: &str) -> Result<(SsmlElement, usize)> {
        let level_str = self
            .extract_attribute(tag, "level")?
            .unwrap_or_else(|| "moderate".to_string());
        let level = match level_str.as_str() {
            "none" => EmphasisLevel::None,
            "reduced" => EmphasisLevel::Reduced,
            "moderate" => EmphasisLevel::Moderate,
            "strong" => EmphasisLevel::Strong,
            _ => EmphasisLevel::Moderate,
        };

        // Find closing tag and parse content
        if let Some(end_pos) = remaining.find("</emphasis>") {
            let content_str = &remaining[..end_pos];
            let content = self.parse_inner_content(content_str)?;
            let consumed = end_pos + "</emphasis>".len();

            Ok((SsmlElement::Emphasis { level, content }, consumed))
        } else {
            Err(G2pError::ConfigError(
                "Unclosed emphasis element".to_string(),
            ))
        }
    }

    /// Parse break element (self-closing)
    fn parse_break_element(&self, tag: &str) -> Result<SsmlElement> {
        let time = self.extract_attribute(tag, "time")?;
        let strength_str = self.extract_attribute(tag, "strength")?;
        let strength = strength_str.map(|s| match s.as_str() {
            "none" => BreakStrength::None,
            "x-weak" => BreakStrength::XWeak,
            "weak" => BreakStrength::Weak,
            "medium" => BreakStrength::Medium,
            "strong" => BreakStrength::Strong,
            "x-strong" => BreakStrength::XStrong,
            _ => BreakStrength::Medium,
        });

        Ok(SsmlElement::Break { time, strength })
    }

    /// Parse say-as element
    fn parse_say_as_element(&self, tag: &str, remaining: &str) -> Result<(SsmlElement, usize)> {
        let interpret_str = self
            .extract_attribute(tag, "interpret-as")?
            .ok_or_else(|| {
                G2pError::ConfigError(
                    "say-as element requires 'interpret-as' attribute".to_string(),
                )
            })?;
        let interpret_as = match interpret_str.as_str() {
            "characters" => InterpretAs::Characters,
            "spell-out" => InterpretAs::SpellOut,
            "cardinal" => InterpretAs::Cardinal,
            "ordinal" => InterpretAs::Ordinal,
            "digits" => InterpretAs::Digits,
            "fraction" => InterpretAs::Fraction,
            "unit" => InterpretAs::Unit,
            "date" => InterpretAs::Date,
            "time" => InterpretAs::Time,
            "telephone" => InterpretAs::Telephone,
            "address" => InterpretAs::Address,
            _ => {
                return Err(G2pError::ConfigError(format!(
                    "Unknown interpret-as value: {interpret_str}"
                )))
            }
        };
        let format = self.extract_attribute(tag, "format")?;

        // Find closing tag and extract content
        if let Some(end_pos) = remaining.find("</say-as>") {
            let content = remaining[..end_pos].to_string();
            let consumed = end_pos + "</say-as>".len();

            Ok((
                SsmlElement::SayAs {
                    interpret_as,
                    format,
                    content,
                },
                consumed,
            ))
        } else {
            Err(G2pError::ConfigError("Unclosed say-as element".to_string()))
        }
    }

    /// Parse prosody element
    fn parse_prosody_element(&self, tag: &str, remaining: &str) -> Result<(SsmlElement, usize)> {
        let rate = self.extract_attribute(tag, "rate")?;
        let pitch = self.extract_attribute(tag, "pitch")?;
        let volume = self.extract_attribute(tag, "volume")?;

        // Find closing tag and parse content
        if let Some(end_pos) = remaining.find("</prosody>") {
            let content_str = &remaining[..end_pos];
            let content = self.parse_inner_content(content_str)?;
            let consumed = end_pos + "</prosody>".len();

            Ok((
                SsmlElement::Prosody {
                    rate,
                    pitch,
                    volume,
                    content,
                },
                consumed,
            ))
        } else {
            Err(G2pError::ConfigError(
                "Unclosed prosody element".to_string(),
            ))
        }
    }

    /// Extract attribute value from tag
    fn extract_attribute(&self, tag: &str, attr_name: &str) -> Result<Option<String>> {
        let pattern = format!("{attr_name}=\"");
        if let Some(start) = tag.find(&pattern) {
            let start_pos = start + pattern.len();
            if let Some(end_pos) = tag[start_pos..].find('"') {
                return Ok(Some(tag[start_pos..start_pos + end_pos].to_string()));
            }
        }
        Ok(None)
    }

    /// Parse language code string
    fn parse_language_code(&self, lang: &str) -> Result<LanguageCode> {
        match lang.to_lowercase().as_str() {
            "en-us" | "en_us" | "en" => Ok(LanguageCode::EnUs),
            "en-gb" | "en_gb" => Ok(LanguageCode::EnGb),
            "ja" | "jp" => Ok(LanguageCode::Ja),
            "zh-cn" | "zh_cn" | "zh" => Ok(LanguageCode::ZhCn),
            "ko" | "kr" => Ok(LanguageCode::Ko),
            "de" => Ok(LanguageCode::De),
            "fr" => Ok(LanguageCode::Fr),
            "es" => Ok(LanguageCode::Es),
            _ => Err(G2pError::ConfigError(format!(
                "Unsupported language: {lang}"
            ))),
        }
    }

    /// Convert SSML element to plain text for G2P processing
    #[allow(clippy::only_used_in_recursion)]
    pub fn to_text(&self, element: &SsmlElement) -> String {
        match element {
            SsmlElement::Speak { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Text(text) => text.clone(),
            SsmlElement::Phoneme { text, .. } => text.clone(),
            SsmlElement::Lang { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Emphasis { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Break { .. } => " ".to_string(), // Represent breaks as spaces
            SsmlElement::SayAs { content, .. } => content.clone(),
            SsmlElement::Prosody { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// Apply phoneme overrides from SSML
    pub fn apply_phoneme_overrides(
        &self,
        element: &SsmlElement,
        phonemes: &mut Vec<Phoneme>,
    ) -> Result<()> {
        match element {
            SsmlElement::Phoneme { ph, .. } => {
                // Replace phonemes for this text with the override
                let override_phonemes = self.parse_phoneme_string(ph)?;
                *phonemes = override_phonemes;
            }
            SsmlElement::Speak { content, .. }
            | SsmlElement::Lang { content, .. }
            | SsmlElement::Emphasis { content, .. }
            | SsmlElement::Prosody { content, .. } => {
                for child in content {
                    self.apply_phoneme_overrides(child, phonemes)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Parse phoneme string into individual phonemes
    fn parse_phoneme_string(&self, ph: &str) -> Result<Vec<Phoneme>> {
        // Split phoneme string and create Phoneme objects
        let symbols: Vec<&str> = ph.split_whitespace().collect();
        let mut phonemes = Vec::new();

        for symbol in symbols {
            phonemes.push(Phoneme {
                symbol: symbol.to_string(),
                ipa_symbol: Some(symbol.to_string()),
                language_notation: Some("SSML-Override".to_string()),
                stress: 0,
                syllable_position: crate::SyllablePosition::Standalone,
                duration_ms: None,
                confidence: 1.0, // Override phonemes have maximum confidence
                phonetic_features: None,
                custom_features: None,
                is_word_boundary: false,
                is_syllable_boundary: false,
            });
        }

        Ok(phonemes)
    }
}

impl Default for SsmlProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssml_processor_creation() {
        let processor = SsmlProcessor::new();
        assert!(processor.phoneme_overrides.is_empty());
    }

    #[test]
    fn test_basic_ssml_parsing() {
        let processor = SsmlProcessor::new();
        let ssml = "<speak>Hello world</speak>";
        let result = processor.parse_ssml(ssml).unwrap();

        match result {
            SsmlElement::Speak { content, .. } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    SsmlElement::Text(text) => assert_eq!(text, "Hello world"),
                    _ => panic!("Expected text element"),
                }
            }
            _ => panic!("Expected speak element"),
        }
    }

    #[test]
    fn test_phoneme_override_parsing() {
        let processor = SsmlProcessor::new();
        let ssml = r#"<speak><phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme></speak>"#;
        let result = processor.parse_ssml(ssml).unwrap();

        match result {
            SsmlElement::Speak { content, .. } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    SsmlElement::Phoneme { alphabet, ph, text } => {
                        assert_eq!(alphabet, "ipa");
                        assert_eq!(ph, "təˈmeɪtoʊ");
                        assert_eq!(text, "tomato");
                    }
                    _ => panic!("Expected phoneme element"),
                }
            }
            _ => panic!("Expected speak element"),
        }
    }

    #[test]
    fn test_language_switching() {
        let processor = SsmlProcessor::new();
        let ssml = r#"<speak>Hello <lang xml:lang="ja">こんにちは</lang> world</speak>"#;
        let result = processor.parse_ssml(ssml).unwrap();

        match result {
            SsmlElement::Speak { content, .. } => {
                assert_eq!(content.len(), 3);
                match &content[1] {
                    SsmlElement::Lang { lang, content } => {
                        assert_eq!(*lang, LanguageCode::Ja);
                        assert_eq!(content.len(), 1);
                    }
                    _ => panic!("Expected lang element"),
                }
            }
            _ => panic!("Expected speak element"),
        }
    }

    #[test]
    fn test_emphasis_parsing() {
        let processor = SsmlProcessor::new();
        let ssml = r#"<speak><emphasis level="strong">important</emphasis></speak>"#;
        let result = processor.parse_ssml(ssml).unwrap();

        match result {
            SsmlElement::Speak { content, .. } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    SsmlElement::Emphasis { level, content } => {
                        assert_eq!(*level, EmphasisLevel::Strong);
                        assert_eq!(content.len(), 1);
                    }
                    _ => panic!("Expected emphasis element"),
                }
            }
            _ => panic!("Expected speak element"),
        }
    }

    #[test]
    fn test_break_parsing() {
        let processor = SsmlProcessor::new();
        let ssml = r#"<speak>Hello <break time="1s"/> world</speak>"#;
        let result = processor.parse_ssml(ssml).unwrap();

        match result {
            SsmlElement::Speak { content, .. } => {
                assert_eq!(content.len(), 3);
                match &content[1] {
                    SsmlElement::Break { time, strength: _ } => {
                        assert_eq!(time.as_ref().unwrap(), "1s");
                    }
                    _ => panic!("Expected break element"),
                }
            }
            _ => panic!("Expected speak element"),
        }
    }

    #[test]
    fn test_to_text_conversion() {
        let processor = SsmlProcessor::new();
        let ssml = r#"<speak>Hello <phoneme alphabet="ipa" ph="wɜːrld">world</phoneme></speak>"#;
        let result = processor.parse_ssml(ssml).unwrap();
        let text = processor.to_text(&result);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_phoneme_string_parsing() {
        let processor = SsmlProcessor::new();
        let phonemes = processor.parse_phoneme_string("h ɛ l oʊ").unwrap();
        assert_eq!(phonemes.len(), 4);
        assert_eq!(phonemes[0].symbol, "h");
        assert_eq!(phonemes[1].symbol, "ɛ");
        assert_eq!(phonemes[2].symbol, "l");
        assert_eq!(phonemes[3].symbol, "oʊ");
    }

    #[test]
    fn test_language_code_parsing() {
        let processor = SsmlProcessor::new();

        assert_eq!(
            processor.parse_language_code("en-US").unwrap(),
            LanguageCode::EnUs
        );
        assert_eq!(
            processor.parse_language_code("ja").unwrap(),
            LanguageCode::Ja
        );
        assert_eq!(
            processor.parse_language_code("de").unwrap(),
            LanguageCode::De
        );

        assert!(processor.parse_language_code("invalid").is_err());
    }

    #[test]
    fn test_attribute_extraction() {
        let processor = SsmlProcessor::new();
        let tag = r#"<phoneme alphabet="ipa" ph="test">"#;

        assert_eq!(
            processor
                .extract_attribute(tag, "alphabet")
                .unwrap()
                .unwrap(),
            "ipa"
        );
        assert_eq!(
            processor.extract_attribute(tag, "ph").unwrap().unwrap(),
            "test"
        );
        assert!(processor
            .extract_attribute(tag, "nonexistent")
            .unwrap()
            .is_none());
    }
}
