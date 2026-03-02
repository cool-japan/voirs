//! LSP document formatting provider
//!
//! Provides automatic formatting for SSML and VoiRS configuration files.

use super::{Position, Range};
use serde_json::Value;

/// Format a document
pub fn format_document(text: &str, language_id: &str) -> Option<Vec<TextEdit>> {
    match language_id {
        "ssml" | "xml" => format_ssml(text),
        "json" => format_json(text),
        "toml" => format_toml(text),
        _ => None,
    }
}

/// Format a range in a document
pub fn format_range(text: &str, range: Range, language_id: &str) -> Option<Vec<TextEdit>> {
    // Extract range text
    let range_text = extract_range_text(text, range)?;

    // Format the extracted text
    let formatted = match language_id {
        "ssml" | "xml" => format_ssml_text(&range_text)?,
        "json" => format_json_text(&range_text)?,
        _ => return None,
    };

    Some(vec![TextEdit {
        range,
        new_text: formatted,
    }])
}

/// Format SSML document
fn format_ssml(text: &str) -> Option<Vec<TextEdit>> {
    let formatted = format_ssml_text(text)?;

    Some(vec![TextEdit {
        range: Range::new(
            Position::new(0, 0),
            Position::new(u32::MAX, 0), // End of document
        ),
        new_text: formatted,
    }])
}

/// Format SSML text
fn format_ssml_text(text: &str) -> Option<String> {
    // Simple SSML formatter
    let mut result = String::new();
    let mut indent = 0;
    let mut in_tag = false;
    let mut current_tag = String::new();

    for ch in text.chars() {
        match ch {
            '<' => {
                in_tag = true;
                current_tag.clear();

                // Check if closing tag
                if result.ends_with('>') {
                    result.push('\n');
                    result.push_str(&"  ".repeat(indent));
                }

                current_tag.push(ch);
            }
            '>' => {
                in_tag = false;
                current_tag.push(ch);

                // Check if self-closing or closing tag
                if current_tag.contains("</") {
                    // Closing tag
                    indent = indent.saturating_sub(1);
                    // Remove last newline+indent if present
                    result = result.trim_end().to_string();
                } else if !current_tag.contains("/>") {
                    // Opening tag (not self-closing)
                    indent += 1;
                }

                result.push_str(&current_tag);
                current_tag.clear();
            }
            '\n' | '\r' if !in_tag => {
                // Skip whitespace outside tags
                continue;
            }
            c if in_tag => {
                current_tag.push(c);
            }
            c => {
                if !c.is_whitespace()
                    || !result
                        .chars()
                        .last()
                        .is_some_and(|last| last.is_whitespace())
                {
                    result.push(c);
                }
            }
        }
    }

    Some(result.trim().to_string())
}

/// Format JSON document
fn format_json(text: &str) -> Option<Vec<TextEdit>> {
    let formatted = format_json_text(text)?;

    Some(vec![TextEdit {
        range: Range::new(Position::new(0, 0), Position::new(u32::MAX, 0)),
        new_text: formatted,
    }])
}

/// Format JSON text
fn format_json_text(text: &str) -> Option<String> {
    // Try to parse and pretty-print JSON
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    serde_json::to_string_pretty(&value).ok()
}

/// Format TOML document
fn format_toml(text: &str) -> Option<Vec<TextEdit>> {
    // Basic TOML formatting (normalize spacing)
    let mut result = String::new();
    let mut last_was_blank = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if !last_was_blank {
                result.push('\n');
                last_was_blank = true;
            }
        } else {
            result.push_str(trimmed);
            result.push('\n');
            last_was_blank = false;
        }
    }

    Some(vec![TextEdit {
        range: Range::new(Position::new(0, 0), Position::new(u32::MAX, 0)),
        new_text: result.trim().to_string() + "\n",
    }])
}

/// Extract text from range
fn extract_range_text(text: &str, range: Range) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();

    if range.start.line == range.end.line {
        let line = lines.get(range.start.line as usize)?;
        let start = range.start.character as usize;
        let end = range.end.character as usize;
        if start < line.len() && end <= line.len() {
            return Some(line[start..end].to_string());
        }
    } else {
        let mut result = String::new();
        for (i, line) in lines.iter().enumerate() {
            let line_num = i as u32;
            if line_num < range.start.line || line_num > range.end.line {
                continue;
            }

            if line_num == range.start.line {
                let start = range.start.character as usize;
                if start < line.len() {
                    result.push_str(&line[start..]);
                    result.push('\n');
                }
            } else if line_num == range.end.line {
                let end = range.end.character as usize;
                if end <= line.len() {
                    result.push_str(&line[..end]);
                }
            } else {
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

/// Text edit for formatting
#[derive(Debug, Clone)]
pub struct TextEdit {
    /// Range to replace
    pub range: Range,
    /// New text
    pub new_text: String,
}

impl TextEdit {
    /// Convert to LSP JSON format
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "range": {
                "start": {
                    "line": self.range.start.line,
                    "character": self.range.start.character
                },
                "end": {
                    "line": self.range.end.line,
                    "character": self.range.end.character
                }
            },
            "newText": self.new_text
        })
    }
}

/// On-type formatting (triggered by specific characters)
pub fn format_on_type(
    text: &str,
    position: Position,
    ch: char,
    language_id: &str,
) -> Option<Vec<TextEdit>> {
    match language_id {
        "ssml" | "xml" if ch == '>' => {
            // Auto-close tag or auto-indent
            auto_complete_tag(text, position)
        }
        "json" if ch == '}' || ch == ']' => {
            // Auto-indent closing brace
            auto_indent_json(text, position)
        }
        _ => None,
    }
}

/// Auto-complete SSML tag
fn auto_complete_tag(text: &str, position: Position) -> Option<Vec<TextEdit>> {
    let lines: Vec<&str> = text.lines().collect();
    let line = lines.get(position.line as usize)?;

    // Find the most recent opening tag before position
    let before_pos = &line[..position.character as usize];

    // Simple tag matching: find <tag and insert </tag>
    if let Some(tag_start) = before_pos.rfind('<') {
        let tag_part = &before_pos[tag_start + 1..];

        // Skip if it's a closing tag or self-closing
        if tag_part.starts_with('/') || before_pos.ends_with('/') {
            return None;
        }

        // Extract tag name (first word)
        if let Some(tag_name) = tag_part.split_whitespace().next() {
            // Check if tag is not self-closing
            if !["break", "meta", "desc"].contains(&tag_name) {
                let closing_tag = format!("</{}>", tag_name);

                return Some(vec![TextEdit {
                    range: Range::new(position, position),
                    new_text: closing_tag,
                }]);
            }
        }
    }

    None
}

/// Auto-indent JSON closing brace
fn auto_indent_json(_text: &str, position: Position) -> Option<Vec<TextEdit>> {
    // Insert newline with proper indentation
    Some(vec![TextEdit {
        range: Range::new(Position::new(position.line, 0), position),
        new_text: "  ".to_string(), // Basic 2-space indent
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_json_text() {
        let input = r#"{"key":"value","nested":{"a":1}}"#;
        let formatted = format_json_text(input).unwrap();

        assert!(formatted.contains("  \"key\": \"value\""));
        assert!(formatted.contains('\n'));
    }

    #[test]
    fn test_format_ssml_text() {
        let input = "<speak><voice name=\"test\">Hello</voice></speak>";
        let formatted = format_ssml_text(input).unwrap();

        // Should have some structure (not exactly same as input)
        assert!(formatted.contains("<speak>"));
        assert!(formatted.contains("</speak>"));
    }

    #[test]
    fn test_format_document_json() {
        let input = r#"{"a":1,"b":2}"#;
        let edits = format_document(input, "json").unwrap();

        assert_eq!(edits.len(), 1);
        assert!(edits[0].new_text.contains("  \"a\": 1"));
    }

    #[test]
    fn test_format_on_type_tag() {
        let text = "<speak";
        let pos = Position::new(0, 6);

        let edits = format_on_type(text, pos, '>', "ssml");
        assert!(edits.is_some());
    }

    #[test]
    fn test_text_edit_to_json() {
        let edit = TextEdit {
            range: Range::single_line(0, 0, 5),
            new_text: "test".to_string(),
        };

        let json = edit.to_json();
        assert_eq!(json["newText"].as_str().unwrap_or_default(), "test");
        assert_eq!(json["range"]["start"]["line"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_extract_range_text() {
        let text = "Hello\nWorld\nTest";
        let range = Range::new(Position::new(0, 0), Position::new(1, 5));

        let extracted = extract_range_text(text, range).unwrap();
        assert!(extracted.contains("Hello"));
        assert!(extracted.contains("World"));
    }

    #[test]
    fn test_auto_complete_tag_skip_self_closing() {
        let text = "<break/";
        let pos = Position::new(0, 7);

        let edits = auto_complete_tag(text, pos);
        assert!(edits.is_none());
    }

    #[test]
    fn test_format_toml() {
        let input = "[section]\nkey = \"value\"\n\n\n[other]\nkey2 = 123";
        let edits = format_toml(input).unwrap();

        assert_eq!(edits.len(), 1);
        // Should normalize blank lines
        assert!(!edits[0].new_text.contains("\n\n\n"));
    }
}
