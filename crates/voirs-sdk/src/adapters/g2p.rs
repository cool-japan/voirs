//! G2P adapter for integrating voirs-g2p with the SDK.

use crate::traits::{G2p as SdkG2p, G2pMetadata as SdkG2pMetadata};
use crate::types::{LanguageCode as SdkLanguageCode, Phoneme as SdkPhoneme};
use crate::{Result, VoirsError};
use async_trait::async_trait;
use std::sync::Arc;

/// Adapter that bridges voirs-g2p components to SDK G2p trait
pub struct G2pAdapter {
    inner: Arc<dyn voirs_g2p::G2p>,
}

impl G2pAdapter {
    /// Create new G2P adapter wrapping a voirs-g2p implementation
    pub fn new(g2p: Arc<dyn voirs_g2p::G2p>) -> Self {
        Self { inner: g2p }
    }

    /// Convert SDK LanguageCode to voirs-g2p LanguageCode
    /// Only maps supported languages, defaults to English for unsupported ones
    fn convert_language_code_to_g2p(lang: SdkLanguageCode) -> voirs_g2p::LanguageCode {
        match lang {
            // Directly supported languages
            SdkLanguageCode::EnUs => voirs_g2p::LanguageCode::EnUs,
            SdkLanguageCode::EnGb => voirs_g2p::LanguageCode::EnGb,
            SdkLanguageCode::De | SdkLanguageCode::DeDe => voirs_g2p::LanguageCode::De,
            SdkLanguageCode::Fr | SdkLanguageCode::FrFr => voirs_g2p::LanguageCode::Fr,
            SdkLanguageCode::Es | SdkLanguageCode::EsEs | SdkLanguageCode::EsMx => {
                voirs_g2p::LanguageCode::Es
            }
            SdkLanguageCode::It | SdkLanguageCode::ItIt => voirs_g2p::LanguageCode::It,
            SdkLanguageCode::Pt | SdkLanguageCode::PtBr => voirs_g2p::LanguageCode::Pt,
            SdkLanguageCode::Ja | SdkLanguageCode::JaJp => voirs_g2p::LanguageCode::Ja,
            SdkLanguageCode::ZhCn => voirs_g2p::LanguageCode::ZhCn,
            SdkLanguageCode::Ko | SdkLanguageCode::KoKr => voirs_g2p::LanguageCode::Ko,

            // Unsupported languages - default to English
            _ => voirs_g2p::LanguageCode::EnUs,
        }
    }

    /// Convert voirs-g2p LanguageCode to SDK LanguageCode
    fn convert_language_code_from_g2p(lang: voirs_g2p::LanguageCode) -> SdkLanguageCode {
        match lang {
            voirs_g2p::LanguageCode::EnUs => SdkLanguageCode::EnUs,
            voirs_g2p::LanguageCode::EnGb => SdkLanguageCode::EnGb,
            voirs_g2p::LanguageCode::De => SdkLanguageCode::De,
            voirs_g2p::LanguageCode::Fr => SdkLanguageCode::Fr,
            voirs_g2p::LanguageCode::Es => SdkLanguageCode::Es,
            voirs_g2p::LanguageCode::It => SdkLanguageCode::It,
            voirs_g2p::LanguageCode::Pt => SdkLanguageCode::Pt,
            voirs_g2p::LanguageCode::Ja => SdkLanguageCode::Ja,
            voirs_g2p::LanguageCode::ZhCn => SdkLanguageCode::ZhCn,
            voirs_g2p::LanguageCode::Ko => SdkLanguageCode::Ko,
        }
    }

    /// Convert voirs-g2p Phoneme to SDK Phoneme
    fn convert_phoneme_from_g2p(phoneme: voirs_g2p::Phoneme) -> SdkPhoneme {
        SdkPhoneme {
            symbol: phoneme.symbol.clone(),
            ipa_symbol: phoneme.ipa_symbol.unwrap_or(phoneme.symbol),
            stress: phoneme.stress,
            syllable_position: Self::convert_syllable_position(phoneme.syllable_position),
            duration_ms: phoneme.duration_ms,
            confidence: phoneme.confidence,
        }
    }

    /// Convert voirs-g2p SyllablePosition to SDK SyllablePosition
    fn convert_syllable_position(
        pos: voirs_g2p::SyllablePosition,
    ) -> crate::types::SyllablePosition {
        match pos {
            voirs_g2p::SyllablePosition::Onset => crate::types::SyllablePosition::Onset,
            voirs_g2p::SyllablePosition::Nucleus => crate::types::SyllablePosition::Nucleus,
            voirs_g2p::SyllablePosition::Coda => crate::types::SyllablePosition::Coda,
            voirs_g2p::SyllablePosition::Final => crate::types::SyllablePosition::Coda,
            voirs_g2p::SyllablePosition::Standalone => crate::types::SyllablePosition::Unknown,
        }
    }

    /// Convert SDK G2pMetadata from voirs-g2p G2pMetadata
    fn convert_metadata_from_g2p(metadata: voirs_g2p::G2pMetadata) -> SdkG2pMetadata {
        SdkG2pMetadata {
            name: metadata.name,
            version: metadata.version,
            description: metadata.description,
            supported_languages: metadata
                .supported_languages
                .into_iter()
                .map(Self::convert_language_code_from_g2p)
                .collect(),
            accuracy_scores: metadata
                .accuracy_scores
                .into_iter()
                .map(|(lang, score)| (Self::convert_language_code_from_g2p(lang), score))
                .collect(),
        }
    }
}

#[async_trait]
impl SdkG2p for G2pAdapter {
    async fn to_phonemes(
        &self,
        text: &str,
        lang: Option<SdkLanguageCode>,
    ) -> Result<Vec<SdkPhoneme>> {
        let g2p_lang = lang.map(Self::convert_language_code_to_g2p);

        match self.inner.to_phonemes(text, g2p_lang).await {
            Ok(phonemes) => Ok(phonemes
                .into_iter()
                .map(Self::convert_phoneme_from_g2p)
                .collect()),
            Err(err) => Err(VoirsError::g2p_error(format!(
                "G2P conversion failed: {err}"
            ))),
        }
    }

    fn supported_languages(&self) -> Vec<SdkLanguageCode> {
        self.inner
            .supported_languages()
            .into_iter()
            .map(Self::convert_language_code_from_g2p)
            .collect()
    }

    fn metadata(&self) -> SdkG2pMetadata {
        Self::convert_metadata_from_g2p(self.inner.metadata())
    }

    async fn preprocess(&self, text: &str, lang: Option<SdkLanguageCode>) -> Result<String> {
        // Default implementation: return text as-is
        // The underlying G2P implementation will handle preprocessing internally
        let _ = lang; // Suppress unused parameter warning
        Ok(text.to_string())
    }

    async fn detect_language(&self, text: &str) -> Result<SdkLanguageCode> {
        // Default implementation: return first supported language
        let _ = text; // Suppress unused parameter warning
        self.supported_languages()
            .first()
            .copied()
            .ok_or_else(|| VoirsError::g2p_error("No supported languages"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_g2p::backends::rule_based::RuleBasedG2p;

    #[tokio::test]
    async fn test_g2p_adapter_creation() {
        let g2p = Arc::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::EnUs));
        let adapter = G2pAdapter::new(g2p);

        // Test that adapter can be created and basic functionality works
        let langs = adapter.supported_languages();
        assert!(!langs.is_empty());

        let metadata = adapter.metadata();
        assert!(!metadata.name.is_empty());
    }

    #[tokio::test]
    async fn test_g2p_adapter_conversion() {
        let g2p = Arc::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::EnUs));
        let adapter = G2pAdapter::new(g2p);

        // Test phoneme conversion
        let result = adapter
            .to_phonemes("hello", Some(SdkLanguageCode::EnUs))
            .await;

        assert!(result.is_ok());
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty());

        // Verify phonemes have expected structure
        for phoneme in &phonemes {
            assert!(!phoneme.symbol.is_empty());
            assert!(phoneme.confidence >= 0.0 && phoneme.confidence <= 1.0);
        }
    }
}
