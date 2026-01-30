//! LSP completion provider

use serde_json::Value;

/// Completion item kind
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum CompletionItemKind {
    Text = 1,
    Method = 2,
    Function = 3,
    Constructor = 4,
    Field = 5,
    Variable = 6,
    Class = 7,
    Interface = 8,
    Module = 9,
    Property = 10,
    Unit = 11,
    Value = 12,
    Enum = 13,
    Keyword = 14,
    Snippet = 15,
    Color = 16,
    File = 17,
    Reference = 18,
}

/// Get SSML tag completions
pub fn get_ssml_completions() -> Vec<Value> {
    vec![
        create_completion_item(
            "speak",
            CompletionItemKind::Snippet,
            "SSML root element",
            "<speak>$1</speak>",
        ),
        create_completion_item(
            "voice",
            CompletionItemKind::Snippet,
            "Voice selection element",
            "<voice name=\"${1:kokoro-en}\">$2</voice>",
        ),
        create_completion_item(
            "prosody",
            CompletionItemKind::Snippet,
            "Prosody control (rate, pitch, volume)",
            "<prosody rate=\"${1:1.0}\" pitch=\"${2:0}\" volume=\"${3:0}\">$4</prosody>",
        ),
        create_completion_item(
            "break",
            CompletionItemKind::Snippet,
            "Insert pause",
            "<break time=\"${1:500ms}\"/>",
        ),
        create_completion_item(
            "emphasis",
            CompletionItemKind::Snippet,
            "Emphasis level control",
            "<emphasis level=\"${1:moderate}\">$2</emphasis>",
        ),
        create_completion_item(
            "say-as",
            CompletionItemKind::Snippet,
            "Control interpretation",
            "<say-as interpret-as=\"${1:cardinal}\">$2</say-as>",
        ),
        create_completion_item(
            "phoneme",
            CompletionItemKind::Snippet,
            "Phonetic pronunciation",
            "<phoneme alphabet=\"${1:ipa}\" ph=\"${2:pronunciation}\">$3</phoneme>",
        ),
        create_completion_item(
            "sub",
            CompletionItemKind::Snippet,
            "Substitute pronunciation",
            "<sub alias=\"${1:alias}\">$2</sub>",
        ),
    ]
}

/// Get voice name completions
pub fn get_voice_completions() -> Vec<Value> {
    vec![
        create_simple_completion(
            "kokoro-en",
            CompletionItemKind::Value,
            "Kokoro English voice",
        ),
        create_simple_completion(
            "kokoro-ja",
            CompletionItemKind::Value,
            "Kokoro Japanese voice",
        ),
        create_simple_completion(
            "kokoro-zh",
            CompletionItemKind::Value,
            "Kokoro Chinese voice",
        ),
        create_simple_completion("en-us-male", CompletionItemKind::Value, "English US Male"),
        create_simple_completion(
            "en-us-female",
            CompletionItemKind::Value,
            "English US Female",
        ),
    ]
}

/// Get emphasis level completions
pub fn get_emphasis_levels() -> Vec<Value> {
    vec![
        create_simple_completion("strong", CompletionItemKind::Keyword, "Strong emphasis"),
        create_simple_completion("moderate", CompletionItemKind::Keyword, "Moderate emphasis"),
        create_simple_completion("reduced", CompletionItemKind::Keyword, "Reduced emphasis"),
        create_simple_completion("none", CompletionItemKind::Keyword, "No emphasis"),
    ]
}

/// Get say-as interpretation completions
pub fn get_interpret_as_types() -> Vec<Value> {
    vec![
        create_simple_completion("cardinal", CompletionItemKind::Keyword, "Cardinal number"),
        create_simple_completion("ordinal", CompletionItemKind::Keyword, "Ordinal number"),
        create_simple_completion("characters", CompletionItemKind::Keyword, "Spell out"),
        create_simple_completion("date", CompletionItemKind::Keyword, "Date format"),
        create_simple_completion("time", CompletionItemKind::Keyword, "Time format"),
        create_simple_completion("telephone", CompletionItemKind::Keyword, "Phone number"),
    ]
}

/// Create a completion item with snippet support
fn create_completion_item(
    label: &str,
    kind: CompletionItemKind,
    detail: &str,
    insert_text: &str,
) -> Value {
    serde_json::json!({
        "label": label,
        "kind": kind as u32,
        "detail": detail,
        "insertText": insert_text,
        "insertTextFormat": 2, // Snippet format
        "documentation": {
            "kind": "markdown",
            "value": format!("**{}**\n\n{}", label, detail)
        }
    })
}

/// Create a simple text completion item
fn create_simple_completion(label: &str, kind: CompletionItemKind, detail: &str) -> Value {
    serde_json::json!({
        "label": label,
        "kind": kind as u32,
        "detail": detail,
        "insertText": label,
        "insertTextFormat": 1 // Plain text
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssml_completions() {
        let completions = get_ssml_completions();
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|c| c["label"] == "speak"));
        assert!(completions.iter().any(|c| c["label"] == "voice"));
        assert!(completions.iter().any(|c| c["label"] == "prosody"));
    }

    #[test]
    fn test_voice_completions() {
        let completions = get_voice_completions();
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|c| c["label"] == "kokoro-en"));
    }

    #[test]
    fn test_emphasis_levels() {
        let levels = get_emphasis_levels();
        assert_eq!(levels.len(), 4);
        assert!(levels.iter().any(|l| l["label"] == "strong"));
        assert!(levels.iter().any(|l| l["label"] == "moderate"));
    }

    #[test]
    fn test_interpret_as_types() {
        let types = get_interpret_as_types();
        assert!(!types.is_empty());
        assert!(types.iter().any(|t| t["label"] == "cardinal"));
        assert!(types.iter().any(|t| t["label"] == "date"));
    }
}
