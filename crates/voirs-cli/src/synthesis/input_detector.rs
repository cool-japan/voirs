//! Smart text input detection and parsing.
//!
//! This module automatically detects the format of input text and extracts
//! relevant synthesis parameters and content.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::Result;

/// Detected input format
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputFormat {
    /// Plain text without any markup
    PlainText,
    /// SSML (Speech Synthesis Markup Language)
    Ssml,
    /// Markdown with TTS hints
    Markdown,
    /// JSON structured input
    Json,
}

/// Parsed input with extracted metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedInput {
    /// Detected format
    pub format: InputFormat,
    /// Actual text content to synthesize
    pub content: String,
    /// Extracted synthesis parameters
    pub parameters: SynthesisParameters,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Synthesis parameters extracted from input
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthesisParameters {
    /// Voice ID if specified
    pub voice: Option<String>,
    /// Speaking rate if specified
    pub rate: Option<f32>,
    /// Pitch adjustment if specified
    pub pitch: Option<f32>,
    /// Volume adjustment if specified
    pub volume: Option<f32>,
    /// Emotion if specified
    pub emotion: Option<String>,
    /// Language if specified
    pub language: Option<String>,
}

/// Detects the format of input text
pub fn detect_format(input: &str) -> InputFormat {
    let trimmed = input.trim();

    // Check for SSML
    if trimmed.starts_with("<speak") && trimmed.ends_with("</speak>") {
        return InputFormat::Ssml;
    }

    // Check for JSON (simple heuristic)
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        // Try to parse as JSON to confirm
        if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            return InputFormat::Json;
        }
    }

    // Check for Markdown patterns
    if contains_markdown_syntax(trimmed) {
        return InputFormat::Markdown;
    }

    // Default to plain text
    InputFormat::PlainText
}

/// Checks if text contains Markdown syntax
fn contains_markdown_syntax(text: &str) -> bool {
    // Check for common Markdown patterns
    text.contains("# ") // Headers
        || text.contains("## ")
        || text.contains("* ") // Lists
        || text.contains("- ")
        || text.contains("**") // Bold
        || text.contains("*") // Italic
        || text.contains("```") // Code blocks
        || text.contains("[") && text.contains("](") // Links
}

/// Parses input text and extracts synthesis parameters
pub fn parse_input(input: &str) -> Result<ParsedInput> {
    let format = detect_format(input);

    match format {
        InputFormat::PlainText => parse_plain_text(input),
        InputFormat::Ssml => parse_ssml(input),
        InputFormat::Markdown => parse_markdown(input),
        InputFormat::Json => parse_json(input),
    }
}

/// Parses plain text input
fn parse_plain_text(input: &str) -> Result<ParsedInput> {
    Ok(ParsedInput {
        format: InputFormat::PlainText,
        content: input.to_string(),
        parameters: SynthesisParameters::default(),
        metadata: HashMap::new(),
    })
}

/// Parses SSML input
fn parse_ssml(input: &str) -> Result<ParsedInput> {
    // For now, pass SSML through directly
    // Future: Extract attributes from SSML tags for parameters
    let mut parameters = SynthesisParameters::default();
    let mut metadata = HashMap::new();

    // Extract voice from SSML if present
    if let Some(voice_start) = input.find("voice name=\"") {
        if let Some(voice_end) = input[voice_start + 12..].find('"') {
            let voice = &input[voice_start + 12..voice_start + 12 + voice_end];
            parameters.voice = Some(voice.to_string());
        }
    }

    // Extract language from SSML if present
    if let Some(lang_start) = input.find("xml:lang=\"") {
        if let Some(lang_end) = input[lang_start + 10..].find('"') {
            let lang = &input[lang_start + 10..lang_start + 10 + lang_end];
            parameters.language = Some(lang.to_string());
        }
    }

    metadata.insert("original_format".to_string(), "ssml".to_string());

    Ok(ParsedInput {
        format: InputFormat::Ssml,
        content: input.to_string(),
        parameters,
        metadata,
    })
}

/// Parses Markdown input with TTS hints
fn parse_markdown(input: &str) -> Result<ParsedInput> {
    let mut content = String::new();
    let mut parameters = SynthesisParameters::default();
    let mut metadata = HashMap::new();

    // Look for TTS hints in comments (<!-- tts: ... -->)
    let lines: Vec<&str> = input.lines().collect();
    let mut skip_next = false;

    for line in &lines {
        let trimmed = line.trim();

        // Parse TTS hints
        if trimmed.starts_with("<!-- tts:") && trimmed.ends_with("-->") {
            let hint = &trimmed[9..trimmed.len() - 3].trim();
            parse_tts_hint(hint, &mut parameters);
            skip_next = false;
            continue;
        }

        // Skip code blocks
        if trimmed.starts_with("```") {
            skip_next = !skip_next;
            continue;
        }

        if skip_next {
            continue;
        }

        // Remove Markdown formatting for TTS
        let cleaned = clean_markdown_line(line);
        if !cleaned.is_empty() {
            content.push_str(&cleaned);
            content.push(' ');
        }
    }

    metadata.insert("original_format".to_string(), "markdown".to_string());

    Ok(ParsedInput {
        format: InputFormat::Markdown,
        content: content.trim().to_string(),
        parameters,
        metadata,
    })
}

/// Parses TTS hint from Markdown comment
fn parse_tts_hint(hint: &str, parameters: &mut SynthesisParameters) {
    for part in hint.split(',') {
        let kv: Vec<&str> = part.trim().splitn(2, '=').collect();
        if kv.len() == 2 {
            let key = kv[0].trim();
            let value = kv[1].trim();

            match key {
                "voice" => parameters.voice = Some(value.to_string()),
                "rate" => parameters.rate = value.parse().ok(),
                "pitch" => parameters.pitch = value.parse().ok(),
                "volume" => parameters.volume = value.parse().ok(),
                "emotion" => parameters.emotion = Some(value.to_string()),
                "language" => parameters.language = Some(value.to_string()),
                _ => {}
            }
        }
    }
}

/// Cleans Markdown syntax from a line for TTS
fn clean_markdown_line(line: &str) -> String {
    let mut result = line.to_string();

    // Remove headers
    result = result
        .trim_start_matches("# ")
        .trim_start_matches("## ")
        .trim_start_matches("### ")
        .trim_start_matches("#### ")
        .trim_start_matches("##### ")
        .trim_start_matches("###### ")
        .to_string();

    // Remove list markers
    result = result
        .trim_start_matches("* ")
        .trim_start_matches("- ")
        .trim_start_matches("+ ")
        .to_string();

    // Remove bold/italic markers
    result = result.replace("**", "");
    result = result.replace("__", "");

    // Remove links but keep text: [text](url) -> text
    while let Some(start) = result.find('[') {
        if let Some(middle) = result[start..].find("](") {
            if let Some(end) = result[start + middle..].find(')') {
                let text = &result[start + 1..start + middle];
                let before = &result[..start];
                let after = &result[start + middle + end + 1..];
                result = format!("{}{}{}", before, text, after);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    result.trim().to_string()
}

/// Parses JSON structured input
fn parse_json(input: &str) -> Result<ParsedInput> {
    let value: serde_json::Value = serde_json::from_str(input)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Invalid JSON input: {}", e)))?;

    let mut parameters = SynthesisParameters::default();
    let mut metadata = HashMap::new();

    // Extract text content
    let content = if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
        text.to_string()
    } else if let Some(content) = value.get("content").and_then(|v| v.as_str()) {
        content.to_string()
    } else {
        return Err(voirs_sdk::VoirsError::config_error(
            "JSON input must contain 'text' or 'content' field",
        ));
    };

    // Extract synthesis parameters
    if let Some(voice) = value.get("voice").and_then(|v| v.as_str()) {
        parameters.voice = Some(voice.to_string());
    }
    if let Some(rate) = value.get("rate").and_then(|v| v.as_f64()) {
        parameters.rate = Some(rate as f32);
    }
    if let Some(pitch) = value.get("pitch").and_then(|v| v.as_f64()) {
        parameters.pitch = Some(pitch as f32);
    }
    if let Some(volume) = value.get("volume").and_then(|v| v.as_f64()) {
        parameters.volume = Some(volume as f32);
    }
    if let Some(emotion) = value.get("emotion").and_then(|v| v.as_str()) {
        parameters.emotion = Some(emotion.to_string());
    }
    if let Some(language) = value.get("language").and_then(|v| v.as_str()) {
        parameters.language = Some(language.to_string());
    }

    // Extract metadata
    if let Some(obj) = value.as_object() {
        for (key, val) in obj {
            if !matches!(
                key.as_str(),
                "text" | "content" | "voice" | "rate" | "pitch" | "volume" | "emotion" | "language"
            ) {
                if let Some(s) = val.as_str() {
                    metadata.insert(key.clone(), s.to_string());
                }
            }
        }
    }

    metadata.insert("original_format".to_string(), "json".to_string());

    Ok(ParsedInput {
        format: InputFormat::Json,
        content,
        parameters,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_plain_text() {
        let input = "Hello, this is plain text.";
        assert_eq!(detect_format(input), InputFormat::PlainText);
    }

    #[test]
    fn test_detect_ssml() {
        let input = r#"<speak>Hello <break time="500ms"/> world</speak>"#;
        assert_eq!(detect_format(input), InputFormat::Ssml);
    }

    #[test]
    fn test_detect_json() {
        let input = r#"{"text": "Hello world", "voice": "en-US"}"#;
        assert_eq!(detect_format(input), InputFormat::Json);
    }

    #[test]
    fn test_detect_markdown() {
        let input = "# Hello\n\nThis is **bold** text.";
        assert_eq!(detect_format(input), InputFormat::Markdown);
    }

    #[test]
    fn test_parse_plain_text() {
        let input = "Hello world";
        let parsed = parse_input(input).unwrap();
        assert_eq!(parsed.format, InputFormat::PlainText);
        assert_eq!(parsed.content, "Hello world");
    }

    #[test]
    fn test_parse_json() {
        let input = r#"{"text": "Hello world", "voice": "kokoro-en", "rate": 1.2}"#;
        let parsed = parse_input(input).unwrap();
        assert_eq!(parsed.format, InputFormat::Json);
        assert_eq!(parsed.content, "Hello world");
        assert_eq!(parsed.parameters.voice, Some("kokoro-en".to_string()));
        assert_eq!(parsed.parameters.rate, Some(1.2));
    }

    #[test]
    fn test_parse_markdown_with_hints() {
        let input = r#"<!-- tts: voice=kokoro-en, rate=1.1 -->
# Welcome

This is **important** text.
- Item 1
- Item 2"#;
        let parsed = parse_input(input).unwrap();
        assert_eq!(parsed.format, InputFormat::Markdown);
        assert_eq!(parsed.parameters.voice, Some("kokoro-en".to_string()));
        assert_eq!(parsed.parameters.rate, Some(1.1));
        assert!(parsed.content.contains("Welcome"));
        assert!(parsed.content.contains("important"));
    }

    #[test]
    fn test_clean_markdown() {
        let line = "## This is a **bold** heading";
        let cleaned = clean_markdown_line(line);
        assert_eq!(cleaned, "This is a bold heading");
    }
}
