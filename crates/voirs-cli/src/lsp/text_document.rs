//! Text document utilities

use super::Position;

/// Get word at position
pub fn get_word_at_position(text: &str, position: Position) -> Option<String> {
    let line = text.lines().nth(position.line as usize)?;
    let chars: Vec<char> = line.chars().collect();

    if position.character as usize >= chars.len() {
        return None;
    }

    // Find word boundaries
    let mut start = position.character as usize;
    let mut end = position.character as usize;

    // Move start backward to word boundary
    while start > 0 && is_word_char(chars[start - 1]) {
        start -= 1;
    }

    // Move end forward to word boundary
    while end < chars.len() && is_word_char(chars[end]) {
        end += 1;
    }

    if start == end {
        return None;
    }

    Some(chars[start..end].iter().collect())
}

/// Check if character is part of a word
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '-' || c == '_'
}

/// Get tag at position (for SSML)
pub fn get_tag_at_position(text: &str, position: Position) -> Option<String> {
    let line = text.lines().nth(position.line as usize)?;

    // Find the nearest opening tag before the position
    let line_up_to_pos = &line[..position.character.min(line.len() as u32) as usize];

    if let Some(last_open) = line_up_to_pos.rfind('<') {
        if let Some(close) = line[last_open..].find('>') {
            let tag_content = &line[last_open + 1..last_open + close];
            let tag_name = tag_content
                .split_whitespace()
                .next()?
                .trim_start_matches('/');
            return Some(tag_name.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_word_at_position() {
        let text = "Hello world test";
        let word = get_word_at_position(text, Position::new(0, 7));
        assert_eq!(word, Some("world".to_string()));
    }

    #[test]
    fn test_get_word_at_position_hyphenated() {
        let text = "kokoro-en test";
        let word = get_word_at_position(text, Position::new(0, 3));
        assert_eq!(word, Some("kokoro-en".to_string()));
    }

    #[test]
    fn test_get_word_at_position_no_word() {
        let text = "   ";
        let word = get_word_at_position(text, Position::new(0, 1));
        assert_eq!(word, None);
    }

    #[test]
    fn test_get_tag_at_position() {
        let text = "<speak>Hello</speak>";
        let tag = get_tag_at_position(text, Position::new(0, 8));
        assert_eq!(tag, Some("speak".to_string()));
    }

    #[test]
    fn test_get_tag_at_position_with_attributes() {
        let text = "<voice name=\"kokoro\">Hello</voice>";
        let tag = get_tag_at_position(text, Position::new(0, 15));
        assert_eq!(tag, Some("voice".to_string()));
    }

    #[test]
    fn test_is_word_char() {
        assert!(is_word_char('a'));
        assert!(is_word_char('Z'));
        assert!(is_word_char('0'));
        assert!(is_word_char('-'));
        assert!(is_word_char('_'));
        assert!(!is_word_char(' '));
        assert!(!is_word_char('<'));
    }
}
