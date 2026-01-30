//! LSP hover provider

/// Get hover information for SSML element
pub fn get_ssml_hover(element: &str) -> Option<String> {
    match element {
        "speak" => Some("**SSML Root Element**\n\nThe root element of an SSML document. All SSML content must be wrapped in this element.".to_string()),
        "voice" => Some("**Voice Selection**\n\nSelects a specific voice for synthesis.\n\n**Attributes:**\n- `name`: Voice identifier (e.g., kokoro-en)".to_string()),
        "prosody" => Some("**Prosody Control**\n\nControls the prosody of speech including rate, pitch, and volume.\n\n**Attributes:**\n- `rate`: Speaking rate (0.5-2.0)\n- `pitch`: Pitch shift in semitones (-12 to +12)\n- `volume`: Volume in dB (-20 to +20)".to_string()),
        "break" => Some("**Pause Insertion**\n\nInserts a pause in speech.\n\n**Attributes:**\n- `time`: Duration (e.g., 500ms, 2s)".to_string()),
        "emphasis" => Some("**Emphasis Control**\n\nControls the emphasis level of text.\n\n**Attributes:**\n- `level`: strong | moderate | reduced | none".to_string()),
        "say-as" => Some("**Interpretation Control**\n\nControls how text is interpreted.\n\n**Attributes:**\n- `interpret-as`: cardinal | ordinal | characters | date | time | telephone".to_string()),
        _ => None,
    }
}

/// Get hover information for voice
pub fn get_voice_hover(voice: &str) -> Option<String> {
    match voice {
        "kokoro-en" => Some("**Kokoro English**\n\nHigh-quality English voice\n\n**Language:** en-US\n**Gender:** Female\n**Features:** Emotion support, prosody control".to_string()),
        "kokoro-ja" => Some("**Kokoro Japanese**\n\nHigh-quality Japanese voice\n\n**Language:** ja-JP\n**Gender:** Female\n**Features:** Emotion support, prosody control".to_string()),
        "kokoro-zh" => Some("**Kokoro Chinese**\n\nHigh-quality Chinese voice\n\n**Language:** zh-CN\n**Gender:** Female\n**Features:** Emotion support, prosody control".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_ssml_hover_speak() {
        let hover = get_ssml_hover("speak");
        assert!(hover.is_some());
        assert!(hover.unwrap().contains("Root Element"));
    }

    #[test]
    fn test_get_ssml_hover_voice() {
        let hover = get_ssml_hover("voice");
        assert!(hover.is_some());
        assert!(hover.unwrap().contains("Voice Selection"));
    }

    #[test]
    fn test_get_ssml_hover_unknown() {
        let hover = get_ssml_hover("unknown-tag");
        assert!(hover.is_none());
    }

    #[test]
    fn test_get_voice_hover() {
        let hover = get_voice_hover("kokoro-en");
        assert!(hover.is_some());
        assert!(hover.unwrap().contains("English"));
    }

    #[test]
    fn test_get_voice_hover_unknown() {
        let hover = get_voice_hover("unknown-voice");
        assert!(hover.is_none());
    }
}
