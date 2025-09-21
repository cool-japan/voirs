//! Simplified SSML parser to avoid complex lifetime issues.

use crate::ssml::elements::*;
use crate::{G2pError, LanguageCode, Result};

/// Simplified SSML parser
pub struct SimpleSsmlParser;

impl SimpleSsmlParser {
    /// Create a new simple parser
    pub fn new() -> Self {
        Self
    }

    /// Parse SSML text into an element tree
    pub fn parse(&self, ssml_text: &str) -> Result<SsmlElement> {
        let trimmed = ssml_text.trim();

        if trimmed.starts_with("<speak") {
            self.parse_speak_element(trimmed)
        } else {
            // Wrap non-SSML text in a speak element
            let wrapped = format!("<speak>{trimmed}</speak>");
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
            version: None,
            content: elements,
        })
    }

    /// Extract language attribute from speak tag
    fn extract_language_attribute(&self, text: &str) -> Result<Option<LanguageCode>> {
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
        if let Some(start) = text.find('>') {
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
                        let (element, consumed) =
                            self.parse_phoneme_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element);
                        current_pos = absolute_tag_end + consumed;
                    } else if tag_content.starts_with("<lang") {
                        let (element, consumed) =
                            self.parse_lang_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element);
                        current_pos = absolute_tag_end + consumed;
                    } else if tag_content.starts_with("<emphasis") {
                        let (element, consumed) =
                            self.parse_emphasis_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element);
                        current_pos = absolute_tag_end + consumed;
                    } else if tag_content.starts_with("<break") {
                        let element = self.parse_break_element(tag_content)?;
                        elements.push(element);
                        current_pos = absolute_tag_end;
                    } else if tag_content.starts_with("<prosody") {
                        let (element, consumed) =
                            self.parse_prosody_element(tag_content, &content[absolute_tag_end..])?;
                        elements.push(element);
                        current_pos = absolute_tag_end + consumed;
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
        let alphabet = self
            .extract_attribute(tag, "alphabet")?
            .unwrap_or_else(|| "ipa".to_string());
        let ph = self.extract_attribute(tag, "ph")?.ok_or_else(|| {
            G2pError::ConfigError("phoneme element requires 'ph' attribute".to_string())
        })?;

        if let Some(end_pos) = remaining.find("</phoneme>") {
            let text = remaining[..end_pos].to_string();
            let consumed = end_pos + "</phoneme>".len();

            Ok((
                SsmlElement::Phoneme {
                    alphabet,
                    ph,
                    text,
                    metadata: None,
                },
                consumed,
            ))
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
        let variant = self.extract_attribute(tag, "variant")?;
        let accent = self.extract_attribute(tag, "accent")?;

        if let Some(end_pos) = remaining.find("</lang>") {
            let content_str = &remaining[..end_pos];
            let content = self.parse_inner_content(content_str)?;
            let consumed = end_pos + "</lang>".len();

            Ok((
                SsmlElement::Lang {
                    lang,
                    content,
                    variant,
                    accent,
                },
                consumed,
            ))
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

        if let Some(end_pos) = remaining.find("</emphasis>") {
            let content_str = &remaining[..end_pos];
            let content = self.parse_inner_content(content_str)?;
            let consumed = end_pos + "</emphasis>".len();

            Ok((
                SsmlElement::Emphasis {
                    level,
                    content,
                    custom_params: None,
                },
                consumed,
            ))
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

        Ok(SsmlElement::Break {
            time,
            strength,
            custom_timing: None,
        })
    }

    /// Parse prosody element
    fn parse_prosody_element(&self, tag: &str, remaining: &str) -> Result<(SsmlElement, usize)> {
        let rate = self.extract_attribute(tag, "rate")?;
        let pitch = self.extract_attribute(tag, "pitch")?;
        let volume = self.extract_attribute(tag, "volume")?;

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
                    enhanced: None,
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

    /// Convert SSML element to plain text
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
            SsmlElement::Break { .. } => " ".to_string(),
            SsmlElement::SayAs { content, .. } => content.clone(),
            SsmlElement::Prosody { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Voice { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Mark { .. } => "".to_string(),
            SsmlElement::Paragraph { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Sentence { content, .. } => content
                .iter()
                .map(|e| self.to_text(e))
                .collect::<Vec<_>>()
                .join(" "),
            SsmlElement::Dictionary { .. } => "".to_string(),
        }
    }
}

impl Default for SimpleSsmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_parsing() {
        let parser = SimpleSsmlParser::new();
        let ssml = "<speak>Hello world</speak>";
        let result = parser.parse(ssml).unwrap();

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
    fn test_phoneme_parsing() {
        let parser = SimpleSsmlParser::new();
        let ssml = r#"<speak><phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme></speak>"#;
        let result = parser.parse(ssml).unwrap();

        match result {
            SsmlElement::Speak { content, .. } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    SsmlElement::Phoneme {
                        alphabet, ph, text, ..
                    } => {
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
    fn test_text_conversion() {
        let parser = SimpleSsmlParser::new();
        let ssml = r#"<speak>Hello <phoneme alphabet="ipa" ph="wɜːrld">world</phoneme></speak>"#;
        let result = parser.parse(ssml).unwrap();
        let text = parser.to_text(&result);
        assert_eq!(text, "Hello world");
    }
}
