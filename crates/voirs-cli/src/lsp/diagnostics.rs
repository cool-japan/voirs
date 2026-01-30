//! LSP diagnostics for SSML validation

use super::{Position, Range};
use serde::{Deserialize, Serialize};

/// Diagnostic severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

/// Diagnostic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Range where the diagnostic applies
    pub range: Range,

    /// Severity level
    pub severity: DiagnosticSeverity,

    /// Diagnostic message
    pub message: String,

    /// Source of the diagnostic
    pub source: String,
}

impl Diagnostic {
    /// Create a new diagnostic
    pub fn new(range: Range, severity: DiagnosticSeverity, message: String) -> Self {
        Self {
            range,
            severity,
            message,
            source: "voirs-lsp".to_string(),
        }
    }

    /// Create an error diagnostic
    pub fn error(range: Range, message: String) -> Self {
        Self::new(range, DiagnosticSeverity::Error, message)
    }

    /// Create a warning diagnostic
    pub fn warning(range: Range, message: String) -> Self {
        Self::new(range, DiagnosticSeverity::Warning, message)
    }

    /// Create an information diagnostic
    pub fn info(range: Range, message: String) -> Self {
        Self::new(range, DiagnosticSeverity::Information, message)
    }
}

/// Validate SSML document
pub fn validate_ssml(text: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Check for speak tag
    if !text.contains("<speak>") || !text.contains("</speak>") {
        diagnostics.push(Diagnostic::error(
            Range::new(Position::new(0, 0), Position::new(0, 0)),
            "SSML document must be wrapped in <speak> tags".to_string(),
        ));
    }

    // Check for unclosed tags
    let mut tag_stack = Vec::new();
    for (line_num, line) in text.lines().enumerate() {
        for (col, _) in line.match_indices('<') {
            if let Some(end) = line[col..].find('>') {
                let tag_content = &line[col + 1..col + end];

                if let Some(stripped) = tag_content.strip_prefix('/') {
                    // Closing tag
                    let tag_name = stripped.split_whitespace().next().unwrap_or("");
                    if let Some(last_tag) = tag_stack.pop() {
                        if last_tag != tag_name {
                            diagnostics.push(Diagnostic::error(
                                Range::single_line(
                                    line_num as u32,
                                    col as u32,
                                    (col + end + 1) as u32,
                                ),
                                format!("Mismatched closing tag: expected </{}>", last_tag),
                            ));
                        }
                    } else {
                        diagnostics.push(Diagnostic::error(
                            Range::single_line(line_num as u32, col as u32, (col + end + 1) as u32),
                            format!("Unexpected closing tag: </{}>", tag_name),
                        ));
                    }
                } else if !tag_content.ends_with('/') {
                    // Opening tag (not self-closing)
                    let tag_name = tag_content.split_whitespace().next().unwrap_or("");
                    if !tag_name.is_empty()
                        && !tag_name.starts_with('?')
                        && !tag_name.starts_with('!')
                    {
                        tag_stack.push(tag_name.to_string());
                    }
                }
            }
        }
    }

    // Check for unclosed tags
    if !tag_stack.is_empty() {
        for tag in tag_stack {
            diagnostics.push(Diagnostic::warning(
                Range::new(Position::new(0, 0), Position::new(0, 0)),
                format!("Unclosed tag: <{}>", tag),
            ));
        }
    }

    diagnostics
}

/// Validate voice name
pub fn validate_voice_name(voice: &str) -> Option<Diagnostic> {
    let valid_voices = [
        "kokoro-en",
        "kokoro-ja",
        "kokoro-zh",
        "en-us-male",
        "en-us-female",
    ];

    if !valid_voices.contains(&voice) {
        Some(Diagnostic::warning(
            Range::new(Position::new(0, 0), Position::new(0, 0)),
            format!(
                "Unknown voice: {}. Available voices: {}",
                voice,
                valid_voices.join(", ")
            ),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_ssml_valid() {
        let text = "<speak>Hello, world!</speak>";
        let diagnostics = validate_ssml(text);
        // Should only have warnings about tags, not errors about missing speak
        assert!(
            diagnostics.is_empty()
                || diagnostics
                    .iter()
                    .all(|d| d.severity != DiagnosticSeverity::Error)
        );
    }

    #[test]
    fn test_validate_ssml_no_speak_tag() {
        let text = "Hello, world!";
        let diagnostics = validate_ssml(text);
        assert!(diagnostics
            .iter()
            .any(|d| { d.severity == DiagnosticSeverity::Error && d.message.contains("speak") }));
    }

    #[test]
    fn test_validate_ssml_unclosed_tag() {
        let text = "<speak><voice>Hello, world!</speak>";
        let diagnostics = validate_ssml(text);
        assert!(diagnostics
            .iter()
            .any(|d| d.message.contains("Unclosed") || d.message.contains("Mismatched")));
    }

    #[test]
    fn test_validate_voice_name_valid() {
        let result = validate_voice_name("kokoro-en");
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_voice_name_invalid() {
        let result = validate_voice_name("unknown-voice");
        assert!(result.is_some());
        assert!(result.unwrap().message.contains("Unknown voice"));
    }

    #[test]
    fn test_diagnostic_creation() {
        let range = Range::single_line(5, 10, 20);
        let diag = Diagnostic::error(range, "Test error".to_string());

        assert_eq!(diag.severity, DiagnosticSeverity::Error);
        assert_eq!(diag.message, "Test error");
        assert_eq!(diag.range.start.line, 5);
    }

    #[test]
    fn test_diagnostic_warning() {
        let range = Range::single_line(0, 0, 10);
        let diag = Diagnostic::warning(range, "Test warning".to_string());

        assert_eq!(diag.severity, DiagnosticSeverity::Warning);
    }
}
