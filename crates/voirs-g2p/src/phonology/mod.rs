//! Phonological process modeling for natural speech production.
//!
//! This module provides sophisticated phonological rule application including:
//! - Assimilation (place, manner, voicing)
//! - Vowel reduction in unstressed syllables
//! - Elision and deletion processes
//! - Liaison and linking
//! - Coarticulation effects
//!
//! # Examples
//!
//! ```no_run
//! use voirs_g2p::phonology::{PhonologicalProcessor, ProcessConfig};
//! use voirs_g2p::{Phoneme, LanguageCode};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let processor = PhonologicalProcessor::new(LanguageCode::EnUs);
//! let input = vec![
//!     Phoneme::new("ɪ".to_string()),
//!     Phoneme::new("n".to_string()),
//!     Phoneme::new("p".to_string()),
//!     Phoneme::new("ʊ".to_string()),
//!     Phoneme::new("t".to_string()),
//! ];
//!
//! // Apply phonological processes (e.g., /n/ -> /m/ before /p/)
//! let output = processor.apply_all_processes(&input)?;
//! # Ok(())
//! # }
//! ```

pub mod processes;
pub mod rules;

pub use processes::{
    AssimilationProcess, ElisionProcess, FinalDevoicingProcess, FortitionProcess, GDroppingProcess,
    HDroppingProcess, LVocalizationProcess, LenitionProcess, LiaisonProcess, NasalizationProcess,
    PalatalizationProcess, PhonologicalProcess, PhonologicalProcessor, ProcessConfig,
    RDroppingProcess, TFlappingProcess, TGlottalingProcess, ThFrontingProcess,
    VoicingAssimilationProcess, VowelDevoicingProcess, VowelReductionProcess,
    YodCoalescenceProcess,
};
pub use rules::{
    AssimilationRule, DeletionRule, PhonologicalRule, ReductionRule, RuleApplication, RuleContext,
};

use crate::{G2pError, LanguageCode, Phoneme, Result};
use serde::{Deserialize, Serialize};

/// Types of phonological processes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessType {
    /// Place assimilation (e.g., /n/ -> /m/ before /p/)
    PlaceAssimilation,
    /// Manner assimilation
    MannerAssimilation,
    /// Voicing assimilation
    VoicingAssimilation,
    /// Vowel reduction in unstressed syllables
    VowelReduction,
    /// Sound deletion (elision)
    Elision,
    /// Linking sounds between words (liaison)
    Liaison,
    /// Nasalization before nasal consonants
    Nasalization,
    /// Palatalization before front vowels
    Palatalization,
    /// Lenition (weakening of consonants)
    Lenition,
    /// Fortition (strengthening of consonants)
    Fortition,
    /// R-dropping in non-rhotic dialects (British RP, New England, etc.)
    RDropping,
    /// T-flapping in American English (/t/ → \[ɾ\])
    TFlapping,
    /// Final devoicing in German (Auslautverhärtung)
    FinalDevoicing,
    /// Vowel devoicing in Japanese (high vowels between voiceless consonants)
    VowelDevoicing,
    /// H-dropping in British English dialects (Cockney, Yorkshire, etc.)
    HDropping,
    /// TH-fronting (/θ/→/f/, /ð/→/v/) in London, Southern US English
    ThFronting,
    /// L-vocalization (syllable-final /l/ → /w/) in London English, Portuguese
    LVocalization,
    /// G-dropping (/ŋ/→/n/) in unstressed syllables (casual English -ing)
    GDropping,
    /// T-glottaling (/t/→/ʔ/) in coda position in British English
    TGlottaling,
    /// Yod-coalescence (/tj/→/tʃ/, /dj/→/dʒ/) in American English
    YodCoalescence,
}

/// Phonological feature for phoneme classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhonologicalFeature {
    /// Voicing feature
    Voiced,
    Voiceless,
    /// Place of articulation
    Bilabial,
    Labiodental,
    Dental,
    Alveolar,
    Postalveolar,
    Palatal,
    Velar,
    Glottal,
    /// Manner of articulation
    Stop,
    Fricative,
    Affricate,
    Nasal,
    Liquid,
    Glide,
    /// Vowel features
    Vowel,
    Front,
    Central,
    Back,
    High,
    Mid,
    Low,
    /// Stress
    Stressed,
    Unstressed,
}

/// Get phonological features for a phoneme
pub fn get_features(phoneme: &str) -> Vec<PhonologicalFeature> {
    use PhonologicalFeature::*;

    match phoneme {
        // Stops
        "p" => vec![Voiceless, Bilabial, Stop],
        "b" => vec![Voiced, Bilabial, Stop],
        "t" => vec![Voiceless, Alveolar, Stop],
        "d" => vec![Voiced, Alveolar, Stop],
        "k" => vec![Voiceless, Velar, Stop],
        "g" => vec![Voiced, Velar, Stop],

        // Fricatives
        "f" => vec![Voiceless, Labiodental, Fricative],
        "v" => vec![Voiced, Labiodental, Fricative],
        "θ" => vec![Voiceless, Dental, Fricative],
        "ð" => vec![Voiced, Dental, Fricative],
        "s" => vec![Voiceless, Alveolar, Fricative],
        "z" => vec![Voiced, Alveolar, Fricative],
        "ʃ" => vec![Voiceless, Postalveolar, Fricative],
        "ʒ" => vec![Voiced, Postalveolar, Fricative],
        "h" => vec![Voiceless, Glottal, Fricative],

        // Affricates
        "tʃ" | "ʧ" => vec![Voiceless, Postalveolar, Affricate],
        "dʒ" | "ʤ" => vec![Voiced, Postalveolar, Affricate],

        // Nasals
        "m" => vec![Voiced, Bilabial, Nasal],
        "n" => vec![Voiced, Alveolar, Nasal],
        "ŋ" => vec![Voiced, Velar, Nasal],

        // Liquids
        "l" => vec![Voiced, Alveolar, Liquid],
        "r" | "ɹ" => vec![Voiced, Alveolar, Liquid],

        // Glides
        "w" => vec![Voiced, Bilabial, Glide],
        "j" | "y" => vec![Voiced, Palatal, Glide],

        // Vowels - High
        "i" | "iː" => vec![Vowel, Front, High],
        "ɪ" => vec![Vowel, Front, High],
        "u" | "uː" => vec![Vowel, Back, High],
        "ʊ" => vec![Vowel, Back, High],

        // Vowels - Mid
        "e" | "eː" => vec![Vowel, Front, Mid],
        "ɛ" => vec![Vowel, Front, Mid],
        "ə" => vec![Vowel, Central, Mid],
        "ʌ" => vec![Vowel, Back, Mid],
        "o" | "oː" => vec![Vowel, Back, Mid],
        "ɔ" | "ɔː" => vec![Vowel, Back, Mid],

        // Vowels - Low
        "æ" => vec![Vowel, Front, Low],
        "a" | "aː" => vec![Vowel, Central, Low],
        "ɑ" | "ɑː" => vec![Vowel, Back, Low],

        // Diphthongs (treat as vowels)
        "aɪ" | "aʊ" | "eɪ" | "oʊ" | "ɔɪ" => vec![Vowel],

        _ => vec![], // Unknown phoneme
    }
}

/// Check if a phoneme has a specific feature
pub fn has_feature(phoneme: &str, feature: PhonologicalFeature) -> bool {
    get_features(phoneme).contains(&feature)
}

/// Find phoneme with similar features except for the changed feature
pub fn find_similar_phoneme(
    phoneme: &str,
    change_feature: PhonologicalFeature,
    target_feature: PhonologicalFeature,
) -> Option<String> {
    let mut features = get_features(phoneme);

    // Remove the old feature if present
    features.retain(|f| *f != change_feature);

    // Add the new feature
    if !features.contains(&target_feature) {
        features.push(target_feature);
    }

    // Find matching phoneme
    match_phoneme_by_features(&features)
}

/// Match a phoneme by its features
fn match_phoneme_by_features(features: &[PhonologicalFeature]) -> Option<String> {
    use PhonologicalFeature::*;

    // Check for specific feature combinations
    if features.contains(&Bilabial) && features.contains(&Nasal) {
        return Some("m".to_string());
    }
    if features.contains(&Alveolar) && features.contains(&Nasal) {
        return Some("n".to_string());
    }
    if features.contains(&Velar) && features.contains(&Nasal) {
        return Some("ŋ".to_string());
    }

    // More sophisticated matching could be added here
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_features() {
        let features = get_features("p");
        assert!(features.contains(&PhonologicalFeature::Voiceless));
        assert!(features.contains(&PhonologicalFeature::Bilabial));
        assert!(features.contains(&PhonologicalFeature::Stop));

        let vowel_features = get_features("i");
        assert!(vowel_features.contains(&PhonologicalFeature::Vowel));
        assert!(vowel_features.contains(&PhonologicalFeature::Front));
        assert!(vowel_features.contains(&PhonologicalFeature::High));
    }

    #[test]
    fn test_has_feature() {
        assert!(has_feature("m", PhonologicalFeature::Nasal));
        assert!(has_feature("m", PhonologicalFeature::Bilabial));
        assert!(!has_feature("m", PhonologicalFeature::Voiceless));

        assert!(has_feature("ə", PhonologicalFeature::Vowel));
        assert!(has_feature("ə", PhonologicalFeature::Central));
    }

    #[test]
    fn test_find_similar_phoneme() {
        // Test place assimilation: alveolar nasal -> bilabial nasal
        let result = find_similar_phoneme(
            "n",
            PhonologicalFeature::Alveolar,
            PhonologicalFeature::Bilabial,
        );
        assert_eq!(result, Some("m".to_string()));
    }
}
