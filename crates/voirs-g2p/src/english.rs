//! English G2P module providing convenient re-exports and aliases.

pub use crate::rules::EnglishRuleG2p;

/// Alias for EnglishRuleG2p for backward compatibility
pub type EnglishG2p = EnglishRuleG2p;

/// Create a new English G2P converter
pub fn new() -> crate::Result<EnglishG2p> {
    EnglishRuleG2p::new()
}

/// Create a default English G2P converter
pub fn default() -> EnglishG2p {
    EnglishRuleG2p::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{G2p, LanguageCode};

    #[tokio::test]
    async fn test_english_g2p_alias() {
        let g2p = EnglishG2p::new().unwrap();

        let phonemes = g2p
            .to_phonemes("hello", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert!(!phonemes.is_empty());

        let languages = g2p.supported_languages();
        assert!(languages.contains(&LanguageCode::EnUs));
    }
}
