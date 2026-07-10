//! SSML (Speech Synthesis Markup Language) support for VoiRS CLI.

use crate::error::{CliError, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

// Static regex patterns compiled once at first use
static RE_SPEAK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<speak[^>]*>.*</speak>").expect("Invalid SPEAK regex pattern"));
static RE_VOICE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<voice[^>]*>.*</voice>").expect("Invalid VOICE regex pattern"));
static RE_PROSODY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<prosody[^>]*>.*</prosody>").expect("Invalid PROSODY regex pattern"));
static RE_BREAK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<break[^/>]*/>").expect("Invalid BREAK regex pattern"));
static RE_EMPHASIS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"<emphasis[^>]*>.*</emphasis>").expect("Invalid EMPHASIS regex pattern")
});
static RE_SAY_AS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<say-as[^>]*>.*</say-as>").expect("Invalid SAY-AS regex pattern"));
static RE_PHONEME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<phoneme[^>]*>.*</phoneme>").expect("Invalid PHONEME regex pattern"));
static RE_SUB: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<sub[^>]*>.*</sub>").expect("Invalid SUB regex pattern"));
static RE_TAG: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<(/?)(\w+)(?:[^>]*)>").expect("Invalid TAG regex pattern"));
static RE_TAG_REMOVE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<[^>]*>").expect("Invalid TAG_REMOVE regex pattern"));
static RE_WHITESPACE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s+").expect("Invalid WHITESPACE regex pattern"));
static RE_PROSODY_TAG: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"<prosody\s+([^>]+)>"#).expect("Invalid PROSODY_TAG regex pattern"));
static RE_RATE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"rate\s*=\s*["']([^"']+)["']"#).expect("Invalid RATE regex pattern"));
static RE_PITCH: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"pitch\s*=\s*["']([^"']+)["']"#).expect("Invalid PITCH regex pattern")
});
static RE_VOLUME: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"volume\s*=\s*["']([^"']+)["']"#).expect("Invalid VOLUME regex pattern")
});
static RE_VOICE_NAME: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"<voice\s+name\s*=\s*["']([^"']+)["']"#).expect("Invalid VOICE_NAME regex pattern")
});

/// Reference fundamental frequency (in Hz) used to convert an absolute pitch
/// value (e.g. `pitch="400Hz"`) into a semitone shift. Semitones are
/// logarithmic (12 semitones = one octave = a frequency doubling), so the
/// conversion must use `log2`, not a linear Hz difference.
const F0_REF_HZ: f32 = 200.0;

/// SSML validation and processing utilities
pub struct SsmlProcessor {
    /// Regex patterns for SSML validation
    patterns: HashMap<String, &'static Lazy<Regex>>,
}

impl Default for SsmlProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl SsmlProcessor {
    /// Create a new SSML processor
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Reference static regex patterns
        patterns.insert("speak".to_string(), &RE_SPEAK);
        patterns.insert("voice".to_string(), &RE_VOICE);
        patterns.insert("prosody".to_string(), &RE_PROSODY);
        patterns.insert("break".to_string(), &RE_BREAK);
        patterns.insert("emphasis".to_string(), &RE_EMPHASIS);
        patterns.insert("say-as".to_string(), &RE_SAY_AS);
        patterns.insert("phoneme".to_string(), &RE_PHONEME);
        patterns.insert("sub".to_string(), &RE_SUB);

        Self { patterns }
    }

    /// Check if text contains SSML markup
    pub fn is_ssml(&self, text: &str) -> bool {
        text.trim_start().starts_with('<') && text.contains("</")
    }

    /// Validate SSML markup
    pub fn validate(&self, ssml: &str) -> Result<Vec<SsmlValidationIssue>> {
        let mut issues = Vec::new();

        // Basic structure validation
        if !ssml.trim().starts_with("<speak") {
            issues.push(SsmlValidationIssue {
                issue_type: SsmlIssueType::Error,
                message: "SSML must start with <speak> tag".to_string(),
                line: 1,
                column: 1,
                suggestion: Some("Wrap your content in <speak>...</speak> tags".to_string()),
            });
        }

        if !ssml.trim().ends_with("</speak>") {
            let line_count = ssml.lines().count();
            let last_line_len = ssml.lines().last().map(|l| l.len()).unwrap_or(0);

            issues.push(SsmlValidationIssue {
                issue_type: SsmlIssueType::Error,
                message: "SSML must end with </speak> tag".to_string(),
                line: line_count,
                column: last_line_len,
                suggestion: Some("Add closing </speak> tag".to_string()),
            });
        }

        // Tag balance validation
        issues.extend(self.validate_tag_balance(ssml)?);

        // Attribute validation
        issues.extend(self.validate_attributes(ssml)?);

        Ok(issues)
    }

    /// Validate that opening and closing tags are balanced
    fn validate_tag_balance(&self, ssml: &str) -> Result<Vec<SsmlValidationIssue>> {
        let mut issues = Vec::new();
        let mut tag_stack = Vec::new();

        for (line_num, line) in ssml.lines().enumerate() {
            for cap in RE_TAG.captures_iter(line) {
                let is_closing = !cap[1].is_empty();
                let tag_name = &cap[2];

                // Skip self-closing tags
                if line.contains(&format!("<{}", tag_name)) && line.contains("/>") {
                    continue;
                }

                if is_closing {
                    if let Some(last_tag) = tag_stack.pop() {
                        if last_tag != tag_name {
                            issues.push(SsmlValidationIssue {
                                issue_type: SsmlIssueType::Error,
                                message: format!(
                                    "Mismatched closing tag: expected </{}>, found </{}>",
                                    last_tag, tag_name
                                ),
                                line: line_num + 1,
                                column: line.find(&cap[0]).unwrap_or(0) + 1,
                                suggestion: Some(format!("Change to </{}>", last_tag)),
                            });
                        }
                    } else {
                        issues.push(SsmlValidationIssue {
                            issue_type: SsmlIssueType::Error,
                            message: format!("Unexpected closing tag: </{}>", tag_name),
                            line: line_num + 1,
                            column: line.find(&cap[0]).unwrap_or(0) + 1,
                            suggestion: Some(
                                "Remove this closing tag or add matching opening tag".to_string(),
                            ),
                        });
                    }
                } else {
                    tag_stack.push(tag_name.to_string());
                }
            }
        }

        // Check for unclosed tags
        let line_count = ssml.lines().count();
        let last_line_len = ssml.lines().last().map(|l| l.len()).unwrap_or(0);

        for unclosed_tag in tag_stack {
            issues.push(SsmlValidationIssue {
                issue_type: SsmlIssueType::Error,
                message: format!("Unclosed tag: <{}>", unclosed_tag),
                line: line_count,
                column: last_line_len,
                suggestion: Some(format!("Add closing tag: </{}>", unclosed_tag)),
            });
        }

        Ok(issues)
    }

    /// Validate SSML attributes
    fn validate_attributes(&self, ssml: &str) -> Result<Vec<SsmlValidationIssue>> {
        let mut issues = Vec::new();

        for (line_num, line) in ssml.lines().enumerate() {
            if let Some(cap) = RE_PROSODY_TAG.captures(line) {
                let attributes = &cap[1];

                // Validate rate attribute
                if let Some(rate_match) = RE_RATE.captures(attributes) {
                    let rate_value = &rate_match[1];
                    if !self.is_valid_prosody_rate(rate_value) {
                        issues.push(SsmlValidationIssue {
                            issue_type: SsmlIssueType::Warning,
                            message: format!("Invalid prosody rate: '{rate_value}'"),
                            line: line_num + 1,
                            column: line.find(rate_value).unwrap_or(0) + 1,
                            suggestion: Some("Use values like: x-slow, slow, medium, fast, x-fast, or percentage/Hz values".to_string()),
                        });
                    }
                }

                // Validate pitch attribute
                if let Some(pitch_match) = RE_PITCH.captures(attributes) {
                    let pitch_value = &pitch_match[1];
                    if !self.is_valid_prosody_pitch(pitch_value) {
                        issues.push(SsmlValidationIssue {
                            issue_type: SsmlIssueType::Warning,
                            message: format!("Invalid prosody pitch: '{pitch_value}'"),
                            line: line_num + 1,
                            column: line.find(pitch_value).unwrap_or(0) + 1,
                            suggestion: Some("Use values like: x-low, low, medium, high, x-high, or Hz/semitone values".to_string()),
                        });
                    }
                }

                // Validate volume attribute
                if let Some(volume_match) = RE_VOLUME.captures(attributes) {
                    let volume_value = &volume_match[1];
                    if !self.is_valid_prosody_volume(volume_value) {
                        issues.push(SsmlValidationIssue {
                            issue_type: SsmlIssueType::Warning,
                            message: format!("Invalid prosody volume: '{volume_value}'"),
                            line: line_num + 1,
                            column: line.find(volume_value).unwrap_or(0) + 1,
                            suggestion: Some("Use values like: silent, x-soft, soft, medium, loud, x-loud, or dB values".to_string()),
                        });
                    }
                }
            }
        }

        Ok(issues)
    }

    /// Check if prosody rate value is valid
    fn is_valid_prosody_rate(&self, value: &str) -> bool {
        matches!(value, "x-slow" | "slow" | "medium" | "fast" | "x-fast")
            || value.ends_with('%')
            || value.ends_with("Hz")
            || value.parse::<f32>().is_ok()
    }

    /// Check if prosody pitch value is valid
    fn is_valid_prosody_pitch(&self, value: &str) -> bool {
        matches!(value, "x-low" | "low" | "medium" | "high" | "x-high")
            || value.ends_with("Hz")
            || value.ends_with("st")
            || value.starts_with('+')
            || value.starts_with('-')
            || value.parse::<f32>().is_ok()
    }

    /// Check if prosody volume value is valid
    fn is_valid_prosody_volume(&self, value: &str) -> bool {
        matches!(
            value,
            "silent" | "x-soft" | "soft" | "medium" | "loud" | "x-loud"
        ) || value.ends_with("dB")
            || value.starts_with('+')
            || value.starts_with('-')
            || value.parse::<f32>().is_ok()
    }

    /// Convert SSML to plain text (remove markup)
    pub fn to_plain_text(&self, ssml: &str) -> String {
        // Remove SSML tags but keep their content
        let text = RE_TAG_REMOVE.replace_all(ssml, "");

        // Clean up extra whitespace
        let text = RE_WHITESPACE.replace_all(&text, " ");

        text.trim().to_string()
    }

    /// Extract synthesis parameters from SSML
    pub fn extract_synthesis_params(&self, ssml: &str) -> SsmlSynthesisParams {
        let mut params = SsmlSynthesisParams::default();

        // Extract voice parameter
        if let Some(voice_match) = RE_VOICE_NAME.captures(ssml) {
            params.voice = Some(voice_match[1].to_string());
        }

        // Extract prosody parameters (use the first occurrence)
        if let Some(prosody_match) = RE_PROSODY_TAG.captures(ssml) {
            let attributes = &prosody_match[1];

            if let Some(rate_match) = RE_RATE.captures(attributes) {
                params.speaking_rate = self.parse_rate_value(&rate_match[1]);
            }

            if let Some(pitch_match) = RE_PITCH.captures(attributes) {
                params.pitch_shift = self.parse_pitch_value(&pitch_match[1]);
            }

            if let Some(volume_match) = RE_VOLUME.captures(attributes) {
                params.volume_gain = self.parse_volume_value(&volume_match[1]);
            }
        }

        params
    }

    /// Parse rate value to numeric multiplier
    fn parse_rate_value(&self, value: &str) -> Option<f32> {
        match value {
            "x-slow" => Some(0.5),
            "slow" => Some(0.75),
            "medium" => Some(1.0),
            "fast" => Some(1.25),
            "x-fast" => Some(1.5),
            _ => {
                if value.ends_with('%') {
                    value
                        .trim_end_matches('%')
                        .parse::<f32>()
                        .ok()
                        .map(|v| v / 100.0)
                } else {
                    value.parse::<f32>().ok()
                }
            }
        }
    }

    /// Parse pitch value to semitone shift
    fn parse_pitch_value(&self, value: &str) -> Option<f32> {
        match value {
            "x-low" => Some(-6.0),
            "low" => Some(-3.0),
            "medium" => Some(0.0),
            "high" => Some(3.0),
            "x-high" => Some(6.0),
            _ => {
                if value.ends_with("st") {
                    value.trim_end_matches("st").parse::<f32>().ok()
                } else if value.ends_with("Hz") {
                    // Semitones are a logarithmic (base-2) unit: doubling the
                    // frequency is +12 semitones (one octave), so the shift
                    // from a reference frequency must use log2, not a linear
                    // Hz difference (e.g. 400Hz should be +12st relative to
                    // 200Hz, and 100Hz should be -12st).
                    //
                    // NOTE: `src/plugins/voices.rs` (DefaultVoicePlugin::synthesize)
                    // independently converts semitones back to Hz using a 150Hz
                    // reference (`150.0 * 2.0_f32.powf(pitch_shift / 12.0)`).
                    // That's a pre-existing inconsistency with this function's
                    // 200Hz reference; it is not reconciled here since fixing it
                    // is out of scope for this change.
                    value
                        .trim_end_matches("Hz")
                        .parse::<f32>()
                        .ok()
                        .map(|hz| 12.0 * (hz / F0_REF_HZ).log2())
                } else {
                    value.parse::<f32>().ok()
                }
            }
        }
    }

    /// Parse volume value to dB gain
    fn parse_volume_value(&self, value: &str) -> Option<f32> {
        match value {
            "silent" => Some(-60.0),
            "x-soft" => Some(-20.0),
            "soft" => Some(-10.0),
            "medium" => Some(0.0),
            "loud" => Some(6.0),
            "x-loud" => Some(12.0),
            _ => {
                if value.ends_with("dB") {
                    value.trim_end_matches("dB").parse::<f32>().ok()
                } else {
                    value.parse::<f32>().ok()
                }
            }
        }
    }
}

/// SSML validation issue
#[derive(Debug, Clone)]
pub struct SsmlValidationIssue {
    pub issue_type: SsmlIssueType,
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub suggestion: Option<String>,
}

/// Type of SSML validation issue
#[derive(Debug, Clone, PartialEq)]
pub enum SsmlIssueType {
    Error,
    Warning,
    Info,
}

/// Synthesis parameters extracted from SSML
#[derive(Debug, Default)]
pub struct SsmlSynthesisParams {
    pub voice: Option<String>,
    pub speaking_rate: Option<f32>,
    pub pitch_shift: Option<f32>,
    pub volume_gain: Option<f32>,
}

/// SSML processing utilities
pub mod utils {
    use super::*;

    /// Wrap plain text in SSML speak tags
    pub fn wrap_in_speak(text: &str) -> String {
        if text.trim_start().starts_with("<speak") {
            text.to_string()
        } else {
            format!("<speak>{}</speak>", text)
        }
    }

    /// Create SSML with prosody tags
    pub fn with_prosody(
        text: &str,
        rate: Option<f32>,
        pitch: Option<f32>,
        volume: Option<f32>,
    ) -> String {
        let mut prosody_attrs = Vec::new();

        if let Some(rate) = rate {
            prosody_attrs.push(format!(
                "rate=\"{}\"",
                if rate < 1.0 {
                    "slow"
                } else if rate > 1.0 {
                    "fast"
                } else {
                    "medium"
                }
            ));
        }

        if let Some(pitch) = pitch {
            prosody_attrs.push(format!("pitch=\"{}st\"", pitch));
        }

        if let Some(volume) = volume {
            prosody_attrs.push(format!("volume=\"{}dB\"", volume));
        }

        if prosody_attrs.is_empty() {
            wrap_in_speak(text)
        } else {
            wrap_in_speak(&format!(
                "<prosody {}>{}</prosody>",
                prosody_attrs.join(" "),
                text
            ))
        }
    }

    /// Add break (pause) to SSML
    pub fn add_break(time: &str) -> String {
        format!("<break time=\"{}\"/>", time)
    }

    /// Add emphasis to text
    pub fn add_emphasis(text: &str, level: &str) -> String {
        format!("<emphasis level=\"{}\">{}</emphasis>", level, text)
    }
}

/// Process SSML text and return processed text
/// For now, this is a simple implementation that validates SSML and returns the text content
pub fn process_ssml(text: &str) -> crate::error::Result<String> {
    let processor = SsmlProcessor::new();

    // Check if the text is SSML
    if !processor.is_ssml(text) {
        // If not SSML, wrap in speak tags
        return Ok(utils::wrap_in_speak(text));
    }

    // Validate SSML
    let issues = processor.validate(text)?;

    // Check for errors
    let errors: Vec<_> = issues
        .iter()
        .filter(|issue| matches!(issue.issue_type, SsmlIssueType::Error))
        .collect();

    if !errors.is_empty() {
        let error_messages: Vec<String> = errors
            .iter()
            .map(|error| format!("Line {}: {}", error.line, error.message))
            .collect();
        return Err(crate::error::CliError::ValidationError(format!(
            "SSML validation failed:\n{}",
            error_messages.join("\n")
        )));
    }

    // For now, just return the original text since we don't have full SSML processing
    // In a real implementation, this would convert SSML to synthesis parameters
    Ok(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_ssml() {
        let processor = SsmlProcessor::new();

        assert!(processor.is_ssml("<speak>Hello</speak>"));
        assert!(processor.is_ssml("  <voice>Text</voice>"));
        assert!(!processor.is_ssml("Plain text"));
        assert!(!processor.is_ssml("Text with <emphasis> but no closing"));
    }

    #[test]
    fn test_to_plain_text() {
        let processor = SsmlProcessor::new();

        let ssml =
            "<speak><prosody rate=\"slow\">Hello <emphasis>world</emphasis></prosody></speak>";
        let plain = processor.to_plain_text(ssml);
        assert_eq!(plain, "Hello world");
    }

    #[test]
    fn test_wrap_in_speak() {
        assert_eq!(utils::wrap_in_speak("Hello"), "<speak>Hello</speak>");
        assert_eq!(
            utils::wrap_in_speak("<speak>Hello</speak>"),
            "<speak>Hello</speak>"
        );
    }

    #[test]
    fn test_extract_synthesis_params() {
        let processor = SsmlProcessor::new();

        let ssml = r#"<speak><voice name="female-voice"><prosody rate="fast" pitch="high" volume="loud">Hello</prosody></voice></speak>"#;
        let params = processor.extract_synthesis_params(ssml);

        assert_eq!(params.voice, Some("female-voice".to_string()));
        assert_eq!(params.speaking_rate, Some(1.25));
        assert_eq!(params.pitch_shift, Some(3.0));
        assert_eq!(params.volume_gain, Some(6.0));
    }

    #[test]
    fn test_parse_pitch_value_hz_uses_logarithmic_semitone_conversion() {
        let processor = SsmlProcessor::new();

        // 400Hz is exactly one octave above the 200Hz reference, which is
        // +12 semitones -- a linear formula would (incorrectly) give +10.
        let one_octave_up = processor.parse_pitch_value("400Hz").unwrap();
        assert!(
            (one_octave_up - 12.0).abs() < 1e-3,
            "expected +12.0 semitones for 400Hz, got {one_octave_up}"
        );

        // 100Hz is exactly one octave below the 200Hz reference, i.e. -12
        // semitones -- a linear formula would (incorrectly) give -5.
        let one_octave_down = processor.parse_pitch_value("100Hz").unwrap();
        assert!(
            (one_octave_down - (-12.0)).abs() < 1e-3,
            "expected -12.0 semitones for 100Hz, got {one_octave_down}"
        );

        // 200Hz is the reference frequency itself, i.e. no shift.
        let no_shift = processor.parse_pitch_value("200Hz").unwrap();
        assert!(
            no_shift.abs() < 1e-3,
            "expected 0.0 semitones for 200Hz, got {no_shift}"
        );
    }
}
