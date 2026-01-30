// ! LSP code actions provider
//!
//! Provides quick fixes and refactorings for SSML documents.

use super::{Position, Range};
use serde_json::Value;

/// Code action kind
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum CodeActionKind {
    /// Quick fix for errors
    QuickFix,
    /// Refactor code
    Refactor,
    /// Refactor extract
    RefactorExtract,
    /// Refactor inline
    RefactorInline,
    /// Refactor rewrite
    RefactorRewrite,
    /// Source action
    Source,
    /// Source organize imports
    SourceOrganizeImports,
}

impl CodeActionKind {
    /// Get the LSP string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CodeActionKind::QuickFix => "quickfix",
            CodeActionKind::Refactor => "refactor",
            CodeActionKind::RefactorExtract => "refactor.extract",
            CodeActionKind::RefactorInline => "refactor.inline",
            CodeActionKind::RefactorRewrite => "refactor.rewrite",
            CodeActionKind::Source => "source",
            CodeActionKind::SourceOrganizeImports => "source.organizeImports",
        }
    }
}

/// Get code actions for a document range
pub fn get_code_actions(document_text: &str, range: Range) -> Vec<Value> {
    let mut actions = Vec::new();

    // Extract text in range
    if let Some(selection) = extract_range_text(document_text, range) {
        // Wrap in prosody tag
        if !selection.trim().starts_with('<') {
            actions.push(create_code_action(
                "Wrap in prosody tag",
                CodeActionKind::RefactorRewrite,
                &format!("<prosody rate=\"1.0\">{}</prosody>", selection),
                range,
            ));
        }

        // Wrap in emphasis tag
        if !selection.trim().starts_with('<') {
            actions.push(create_code_action(
                "Add emphasis",
                CodeActionKind::RefactorRewrite,
                &format!("<emphasis level=\"moderate\">{}</emphasis>", selection),
                range,
            ));
        }

        // Wrap in voice tag
        if !selection.trim().starts_with('<') {
            actions.push(create_code_action(
                "Change voice",
                CodeActionKind::RefactorRewrite,
                &format!("<voice name=\"kokoro-en\">{}</voice>", selection),
                range,
            ));
        }
    }

    // Add break before/after
    actions.push(create_insert_action(
        "Insert pause before",
        range.start,
        "<break time=\"500ms\"/>",
    ));

    actions.push(create_insert_action(
        "Insert pause after",
        range.end,
        "<break time=\"500ms\"/>",
    ));

    // Format SSML
    actions.push(create_source_action(
        "Format SSML",
        CodeActionKind::Source,
        "format",
    ));

    // Validate SSML
    actions.push(create_source_action(
        "Validate SSML",
        CodeActionKind::Source,
        "validate",
    ));

    actions
}

/// Create a code action
fn create_code_action(title: &str, kind: CodeActionKind, new_text: &str, range: Range) -> Value {
    serde_json::json!({
        "title": title,
        "kind": kind.as_str(),
        "edit": {
            "changes": {
                "document": [{
                    "range": {
                        "start": {
                            "line": range.start.line,
                            "character": range.start.character
                        },
                        "end": {
                            "line": range.end.line,
                            "character": range.end.character
                        }
                    },
                    "newText": new_text
                }]
            }
        }
    })
}

/// Create an insert action
fn create_insert_action(title: &str, position: Position, text: &str) -> Value {
    serde_json::json!({
        "title": title,
        "kind": CodeActionKind::RefactorRewrite.as_str(),
        "edit": {
            "changes": {
                "document": [{
                    "range": {
                        "start": {
                            "line": position.line,
                            "character": position.character
                        },
                        "end": {
                            "line": position.line,
                            "character": position.character
                        }
                    },
                    "newText": text
                }]
            }
        }
    })
}

/// Create a source action
fn create_source_action(title: &str, kind: CodeActionKind, command: &str) -> Value {
    serde_json::json!({
        "title": title,
        "kind": kind.as_str(),
        "command": {
            "title": title,
            "command": format!("voirs.{}", command),
            "arguments": []
        }
    })
}

/// Extract text from a range
fn extract_range_text(text: &str, range: Range) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();

    if range.start.line == range.end.line {
        // Single line selection
        if let Some(line) = lines.get(range.start.line as usize) {
            let start = range.start.character as usize;
            let end = range.end.character as usize;
            if start < line.len() && end <= line.len() {
                return Some(line[start..end].to_string());
            }
        }
    } else {
        // Multi-line selection
        let mut result = String::new();
        for (i, line) in lines.iter().enumerate() {
            let line_num = i as u32;
            if line_num < range.start.line || line_num > range.end.line {
                continue;
            }

            if line_num == range.start.line {
                // First line
                let start = range.start.character as usize;
                if start < line.len() {
                    result.push_str(&line[start..]);
                    result.push('\n');
                }
            } else if line_num == range.end.line {
                // Last line
                let end = range.end.character as usize;
                if end <= line.len() {
                    result.push_str(&line[..end]);
                }
            } else {
                // Middle lines
                result.push_str(line);
                result.push('\n');
            }
        }
        if !result.is_empty() {
            return Some(result);
        }
    }

    None
}

/// Get quick fixes for common SSML errors
pub fn get_quick_fixes(error_message: &str, range: Range) -> Vec<Value> {
    let mut fixes = Vec::new();

    if error_message.contains("unclosed tag") {
        if let Some(tag_name) = extract_tag_name(error_message) {
            fixes.push(create_code_action(
                &format!("Close <{}> tag", tag_name),
                CodeActionKind::QuickFix,
                &format!("</{}>", tag_name),
                range,
            ));
        }
    }

    if error_message.contains("invalid attribute") {
        fixes.push(create_code_action(
            "Remove invalid attribute",
            CodeActionKind::QuickFix,
            "",
            range,
        ));
    }

    if error_message.contains("invalid voice") {
        fixes.push(create_code_action(
            "Replace with 'kokoro-en'",
            CodeActionKind::QuickFix,
            "kokoro-en",
            range,
        ));
    }

    fixes
}

/// Extract tag name from error message
fn extract_tag_name(message: &str) -> Option<String> {
    // Simple extraction: look for text between '<' and '>'
    if let Some(start) = message.find('<') {
        if let Some(end) = message[start..].find('>') {
            let tag = &message[start + 1..start + end];
            // Remove any attributes or whitespace
            let tag_name = tag.split_whitespace().next()?;
            return Some(tag_name.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_code_actions() {
        let text = "Hello world";
        let range = Range::single_line(0, 0, 11);

        let actions = get_code_actions(text, range);
        assert!(!actions.is_empty());

        // Check for expected actions
        let titles: Vec<&str> = actions
            .iter()
            .map(|a| a["title"].as_str().unwrap_or(""))
            .collect();

        assert!(titles.contains(&"Wrap in prosody tag"));
        assert!(titles.contains(&"Add emphasis"));
        assert!(titles.contains(&"Insert pause before"));
    }

    #[test]
    fn test_extract_range_text_single_line() {
        let text = "Hello world";
        let range = Range::single_line(0, 0, 5);

        let extracted = extract_range_text(text, range);
        assert_eq!(extracted, Some("Hello".to_string()));
    }

    #[test]
    fn test_extract_range_text_multi_line() {
        let text = "Line 1\nLine 2\nLine 3";
        let range = Range::new(Position::new(0, 5), Position::new(1, 4));

        let extracted = extract_range_text(text, range);
        assert!(extracted.is_some());
        assert!(extracted.unwrap().contains("1\nLine"));
    }

    #[test]
    fn test_get_quick_fixes_unclosed_tag() {
        let error = "Error: unclosed tag <speak>";
        let range = Range::single_line(0, 0, 7);

        let fixes = get_quick_fixes(error, range);
        assert!(!fixes.is_empty());
        assert_eq!(fixes[0]["title"].as_str().unwrap(), "Close <speak> tag");
    }

    #[test]
    fn test_extract_tag_name() {
        let message = "unclosed tag <prosody>";
        let tag = extract_tag_name(message);
        assert_eq!(tag, Some("prosody".to_string()));
    }

    #[test]
    fn test_code_action_kinds() {
        assert_eq!(CodeActionKind::QuickFix.as_str(), "quickfix");
        assert_eq!(CodeActionKind::Refactor.as_str(), "refactor");
        assert_eq!(CodeActionKind::Source.as_str(), "source");
    }

    #[test]
    fn test_create_insert_action() {
        let action = create_insert_action("Test insert", Position::new(5, 10), "test text");

        assert_eq!(action["title"].as_str().unwrap(), "Test insert");
        assert_eq!(
            action["edit"]["changes"]["document"][0]["newText"]
                .as_str()
                .unwrap(),
            "test text"
        );
    }
}
