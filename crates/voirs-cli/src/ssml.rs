//! SSML (Speech Synthesis Markup Language) support for VoiRS CLI.

use std::collections::HashMap;
use regex::Regex;
use crate::error::{CliError, CliResult};

/// SSML validation and processing utilities
pub struct SsmlProcessor {
    /// Regex patterns for SSML validation
    patterns: HashMap<String, Regex>,
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
        
        // Basic SSML tag patterns
        patterns.insert("speak".to_string(), 
            Regex::new(r"<speak[^>]*>.*</speak>").unwrap());
        patterns.insert("voice".to_string(), 
            Regex::new(r"<voice[^>]*>.*</voice>").unwrap());
        patterns.insert("prosody".to_string(), 
            Regex::new(r"<prosody[^>]*>.*</prosody>").unwrap());
        patterns.insert("break".to_string(), 
            Regex::new(r"<break[^/>]*/>").unwrap());
        patterns.insert("emphasis".to_string(), 
            Regex::new(r"<emphasis[^>]*>.*</emphasis>").unwrap());
        patterns.insert("say-as".to_string(), 
            Regex::new(r"<say-as[^>]*>.*</say-as>").unwrap());
        patterns.insert("phoneme".to_string(), 
            Regex::new(r"<phoneme[^>]*>.*</phoneme>").unwrap());
        patterns.insert("sub".to_string(), 
            Regex::new(r"<sub[^>]*>.*</sub>").unwrap());
        
        Self { patterns }
    }
    
    /// Check if text contains SSML markup
    pub fn is_ssml(&self, text: &str) -> bool {
        text.trim_start().starts_with('<') && text.contains("</")
    }
    
    /// Validate SSML markup
    pub fn validate(&self, ssml: &str) -> CliResult<Vec<SsmlValidationIssue>> {
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
            issues.push(SsmlValidationIssue {
                issue_type: SsmlIssueType::Error,
                message: "SSML must end with </speak> tag".to_string(),
                line: ssml.lines().count(),
                column: ssml.lines().last().unwrap_or("").len(),
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
    fn validate_tag_balance(&self, ssml: &str) -> CliResult<Vec<SsmlValidationIssue>> {
        let mut issues = Vec::new();
        let mut tag_stack = Vec::new();
        
        // Simple tag matching regex
        let tag_regex = Regex::new(r"<(/?)(\w+)(?:[^>]*)>").unwrap();
        
        for (line_num, line) in ssml.lines().enumerate() {
            for cap in tag_regex.captures_iter(line) {
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
                                message: format!("Mismatched closing tag: expected </{}>, found </{}>", last_tag, tag_name),
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
                            suggestion: Some("Remove this closing tag or add matching opening tag".to_string()),
                        });
                    }
                } else {
                    tag_stack.push(tag_name.to_string());
                }
            }
        }
        
        // Check for unclosed tags
        for unclosed_tag in tag_stack {
            issues.push(SsmlValidationIssue {
                issue_type: SsmlIssueType::Error,
                message: format!("Unclosed tag: <{}>", unclosed_tag),
                line: ssml.lines().count(),
                column: ssml.lines().last().unwrap_or("").len(),
                suggestion: Some(format!("Add closing tag: </{}>", unclosed_tag)),
            });
        }
        
        Ok(issues)
    }
    
    /// Validate SSML attributes
    fn validate_attributes(&self, ssml: &str) -> CliResult<Vec<SsmlValidationIssue>> {
        let mut issues = Vec::new();
        
        // Prosody attribute validation
        let prosody_regex = Regex::new(r#"<prosody\s+([^>]+)>"#).unwrap();
        for (line_num, line) in ssml.lines().enumerate() {
            if let Some(cap) = prosody_regex.captures(line) {
                let attributes = &cap[1];
                
                // Validate rate attribute
                if let Some(rate_match) = Regex::new(r#"rate\s*=\s*["']([^"']+)["']"#).unwrap().captures(attributes) {
                    let rate_value = &rate_match[1];
                    if !self.is_valid_prosody_rate(rate_value) {
                        issues.push(SsmlValidationIssue {
                            issue_type: SsmlIssueType::Warning,
                            message: format!("Invalid prosody rate: '{}'", rate_value),
                            line: line_num + 1,
                            column: line.find(rate_value).unwrap_or(0) + 1,
                            suggestion: Some("Use values like: x-slow, slow, medium, fast, x-fast, or percentage/Hz values".to_string()),
                        });
                    }
                }
                
                // Validate pitch attribute
                if let Some(pitch_match) = Regex::new(r#"pitch\s*=\s*["']([^"']+)["']"#).unwrap().captures(attributes) {
                    let pitch_value = &pitch_match[1];
                    if !self.is_valid_prosody_pitch(pitch_value) {
                        issues.push(SsmlValidationIssue {
                            issue_type: SsmlIssueType::Warning,
                            message: format!("Invalid prosody pitch: '{}'", pitch_value),
                            line: line_num + 1,
                            column: line.find(pitch_value).unwrap_or(0) + 1,
                            suggestion: Some("Use values like: x-low, low, medium, high, x-high, or Hz/semitone values".to_string()),
                        });
                    }
                }
                
                // Validate volume attribute
                if let Some(volume_match) = Regex::new(r#"volume\s*=\s*["']([^"']+)["']"#).unwrap().captures(attributes) {
                    let volume_value = &volume_match[1];
                    if !self.is_valid_prosody_volume(volume_value) {
                        issues.push(SsmlValidationIssue {
                            issue_type: SsmlIssueType::Warning,
                            message: format!("Invalid prosody volume: '{}'", volume_value),
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
        matches!(value, "x-slow" | "slow" | "medium" | "fast" | "x-fast") ||
        value.ends_with('%') ||
        value.ends_with("Hz") ||
        value.parse::<f32>().is_ok()
    }
    
    /// Check if prosody pitch value is valid
    fn is_valid_prosody_pitch(&self, value: &str) -> bool {
        matches!(value, "x-low" | "low" | "medium" | "high" | "x-high") ||
        value.ends_with("Hz") ||
        value.ends_with("st") ||
        value.starts_with('+') ||
        value.starts_with('-') ||
        value.parse::<f32>().is_ok()
    }
    
    /// Check if prosody volume value is valid
    fn is_valid_prosody_volume(&self, value: &str) -> bool {
        matches!(value, "silent" | "x-soft" | "soft" | "medium" | "loud" | "x-loud") ||
        value.ends_with("dB") ||
        value.starts_with('+') ||
        value.starts_with('-') ||
        value.parse::<f32>().is_ok()
    }
    
    /// Convert SSML to plain text (remove markup)
    pub fn to_plain_text(&self, ssml: &str) -> String {
        let mut text = ssml.to_string();
        
        // Remove SSML tags but keep their content
        let tag_regex = Regex::new(r"<[^>]*>").unwrap();
        text = tag_regex.replace_all(&text, "").to_string();
        
        // Clean up extra whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        text = whitespace_regex.replace_all(&text, " ").to_string();
        
        text.trim().to_string()
    }
    
    /// Extract synthesis parameters from SSML
    pub fn extract_synthesis_params(&self, ssml: &str) -> SsmlSynthesisParams {
        let mut params = SsmlSynthesisParams::default();
        
        // Extract voice parameter
        if let Some(voice_match) = Regex::new(r#"<voice\s+name\s*=\s*["']([^"']+)["']"#).unwrap().captures(ssml) {
            params.voice = Some(voice_match[1].to_string());
        }
        
        // Extract prosody parameters (use the first occurrence)
        if let Some(prosody_match) = Regex::new(r#"<prosody\s+([^>]+)>"#).unwrap().captures(ssml) {
            let attributes = &prosody_match[1];
            
            if let Some(rate_match) = Regex::new(r#"rate\s*=\s*["']([^"']+)["']"#).unwrap().captures(attributes) {
                params.speaking_rate = self.parse_rate_value(&rate_match[1]);
            }
            
            if let Some(pitch_match) = Regex::new(r#"pitch\s*=\s*["']([^"']+)["']"#).unwrap().captures(attributes) {
                params.pitch_shift = self.parse_pitch_value(&pitch_match[1]);
            }
            
            if let Some(volume_match) = Regex::new(r#"volume\s*=\s*["']([^"']+)["']"#).unwrap().captures(attributes) {
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
                    value.trim_end_matches('%').parse::<f32>().ok().map(|v| v / 100.0)
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
                    // Convert Hz to approximate semitones (simplified)
                    value.trim_end_matches("Hz").parse::<f32>().ok().map(|hz| {
                        // Very rough conversion, would need proper pitch detection
                        (hz - 200.0) / 20.0
                    })
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
    pub fn with_prosody(text: &str, rate: Option<f32>, pitch: Option<f32>, volume: Option<f32>) -> String {
        let mut prosody_attrs = Vec::new();
        
        if let Some(rate) = rate {
            prosody_attrs.push(format!("rate=\"{}\"", 
                if rate < 1.0 { "slow" } else if rate > 1.0 { "fast" } else { "medium" }));
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
            wrap_in_speak(&format!("<prosody {}>{}</prosody>", prosody_attrs.join(" "), text))
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
        
        let ssml = "<speak><prosody rate=\"slow\">Hello <emphasis>world</emphasis></prosody></speak>";
        let plain = processor.to_plain_text(ssml);
        assert_eq!(plain, "Hello world");
    }
    
    #[test]
    fn test_wrap_in_speak() {
        assert_eq!(utils::wrap_in_speak("Hello"), "<speak>Hello</speak>");
        assert_eq!(utils::wrap_in_speak("<speak>Hello</speak>"), "<speak>Hello</speak>");
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
}