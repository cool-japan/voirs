//! Arabic phoneme inventory and G2P rules.
//!
//! Modern Standard Arabic has 28 consonants and 6 vowels (3 short, 3 long).
//! Notable features include pharyngealization (emphasis) and uvular/pharyngeal consonants.

use super::*;
use crate::LanguageCode;

/// Get the Arabic phoneme inventory
pub fn get_arabic_inventory() -> PhonemeInventory {
    PhonemeInventory {
        language: LanguageCode::Ar,
        consonants: get_arabic_consonants(),
        vowels: get_arabic_vowels(),
        other_sounds: Vec::new(),
        grapheme_mappings: get_arabic_grapheme_mappings(),
    }
}

/// Arabic consonants
fn get_arabic_consonants() -> Vec<PhonemeInfo> {
    vec![
        // Plosives
        PhonemeInfo {
            symbol: "b".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ب".to_string()],
        },
        PhonemeInfo {
            symbol: "t".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ت".to_string()],
        },
        PhonemeInfo {
            symbol: "tˤ".to_string(), // Pharyngealized t
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ط".to_string()],
        },
        PhonemeInfo {
            symbol: "d".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["د".to_string()],
        },
        PhonemeInfo {
            symbol: "dˤ".to_string(), // Pharyngealized d
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ض".to_string()],
        },
        PhonemeInfo {
            symbol: "k".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ك".to_string()],
        },
        PhonemeInfo {
            symbol: "q".to_string(), // Uvular plosive
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Uvular),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ق".to_string()],
        },
        PhonemeInfo {
            symbol: "ʔ".to_string(), // Glottal stop
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Glottal),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ء".to_string(), "أ".to_string(), "إ".to_string()],
        },
        // Fricatives
        PhonemeInfo {
            symbol: "f".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Labiodental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ف".to_string()],
        },
        PhonemeInfo {
            symbol: "θ".to_string(), // Voiceless dental fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Dental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ث".to_string()],
        },
        PhonemeInfo {
            symbol: "ð".to_string(), // Voiced dental fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Dental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ذ".to_string()],
        },
        PhonemeInfo {
            symbol: "ðˤ".to_string(), // Pharyngealized voiced dental fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Dental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ظ".to_string()],
        },
        PhonemeInfo {
            symbol: "s".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["س".to_string()],
        },
        PhonemeInfo {
            symbol: "sˤ".to_string(), // Pharyngealized s
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ص".to_string()],
        },
        PhonemeInfo {
            symbol: "z".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ز".to_string()],
        },
        PhonemeInfo {
            symbol: "ʃ".to_string(), // Postalveolar fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Postalveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ش".to_string()],
        },
        PhonemeInfo {
            symbol: "x".to_string(), // Voiceless velar fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["خ".to_string()],
        },
        PhonemeInfo {
            symbol: "ɣ".to_string(), // Voiced velar fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["غ".to_string()],
        },
        PhonemeInfo {
            symbol: "ħ".to_string(), // Voiceless pharyngeal fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Pharyngeal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ح".to_string()],
        },
        PhonemeInfo {
            symbol: "ʕ".to_string(), // Voiced pharyngeal fricative
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Pharyngeal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ع".to_string()],
        },
        PhonemeInfo {
            symbol: "h".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Glottal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ه".to_string()],
        },
        // Nasals
        PhonemeInfo {
            symbol: "m".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["م".to_string()],
        },
        PhonemeInfo {
            symbol: "n".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ن".to_string()],
        },
        // Liquids
        PhonemeInfo {
            symbol: "l".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::LateralApproximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ل".to_string()],
        },
        PhonemeInfo {
            symbol: "r".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Trill),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ر".to_string()],
        },
        // Approximants
        PhonemeInfo {
            symbol: "w".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Approximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["و".to_string()],
        },
        PhonemeInfo {
            symbol: "j".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Approximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ي".to_string()],
        },
        // Affricate
        PhonemeInfo {
            symbol: "dʒ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Postalveolar),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ج".to_string()],
        },
    ]
}

/// Arabic vowels (short and long)
fn get_arabic_vowels() -> Vec<PhonemeInfo> {
    vec![
        // Short vowels
        PhonemeInfo {
            symbol: "a".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Open),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["َ".to_string()], // Fatha
        },
        PhonemeInfo {
            symbol: "i".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["ِ".to_string()], // Kasra
        },
        PhonemeInfo {
            symbol: "u".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["ُ".to_string()], // Damma
        },
        // Long vowels
        PhonemeInfo {
            symbol: "aː".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Open),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["ا".to_string()], // Alif
        },
        PhonemeInfo {
            symbol: "iː".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["ي".to_string()], // Ya as vowel
        },
        PhonemeInfo {
            symbol: "uː".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["و".to_string()], // Waw as vowel
        },
    ]
}

/// Arabic grapheme to phoneme mappings
fn get_arabic_grapheme_mappings() -> HashMap<String, Vec<String>> {
    let mut mappings = HashMap::new();

    // Consonants
    mappings.insert("ب".to_string(), vec!["b".to_string()]);
    mappings.insert("ت".to_string(), vec!["t".to_string()]);
    mappings.insert("ث".to_string(), vec!["θ".to_string()]);
    mappings.insert("ج".to_string(), vec!["dʒ".to_string()]);
    mappings.insert("ح".to_string(), vec!["ħ".to_string()]);
    mappings.insert("خ".to_string(), vec!["x".to_string()]);
    mappings.insert("د".to_string(), vec!["d".to_string()]);
    mappings.insert("ذ".to_string(), vec!["ð".to_string()]);
    mappings.insert("ر".to_string(), vec!["r".to_string()]);
    mappings.insert("ز".to_string(), vec!["z".to_string()]);
    mappings.insert("س".to_string(), vec!["s".to_string()]);
    mappings.insert("ش".to_string(), vec!["ʃ".to_string()]);
    mappings.insert("ص".to_string(), vec!["sˤ".to_string()]);
    mappings.insert("ض".to_string(), vec!["dˤ".to_string()]);
    mappings.insert("ط".to_string(), vec!["tˤ".to_string()]);
    mappings.insert("ظ".to_string(), vec!["ðˤ".to_string()]);
    mappings.insert("ع".to_string(), vec!["ʕ".to_string()]);
    mappings.insert("غ".to_string(), vec!["ɣ".to_string()]);
    mappings.insert("ف".to_string(), vec!["f".to_string()]);
    mappings.insert("ق".to_string(), vec!["q".to_string()]);
    mappings.insert("ك".to_string(), vec!["k".to_string()]);
    mappings.insert("ل".to_string(), vec!["l".to_string()]);
    mappings.insert("م".to_string(), vec!["m".to_string()]);
    mappings.insert("ن".to_string(), vec!["n".to_string()]);
    mappings.insert("ه".to_string(), vec!["h".to_string()]);
    mappings.insert("و".to_string(), vec!["w".to_string(), "uː".to_string()]);
    mappings.insert("ي".to_string(), vec!["j".to_string(), "iː".to_string()]);
    mappings.insert("ء".to_string(), vec!["ʔ".to_string()]);
    mappings.insert("أ".to_string(), vec!["ʔa".to_string()]);
    mappings.insert("إ".to_string(), vec!["ʔi".to_string()]);
    mappings.insert("ؤ".to_string(), vec!["ʔu".to_string()]);
    mappings.insert("ئ".to_string(), vec!["ʔ".to_string()]);

    // Vowel diacritics
    mappings.insert("َ".to_string(), vec!["a".to_string()]); // Fatha
    mappings.insert("ِ".to_string(), vec!["i".to_string()]); // Kasra
    mappings.insert("ُ".to_string(), vec!["u".to_string()]); // Damma
    mappings.insert("ا".to_string(), vec!["aː".to_string()]); // Alif

    mappings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arabic_consonants() {
        let consonants = get_arabic_consonants();
        assert!(
            consonants.len() >= 28,
            "Arabic should have at least 28 consonants"
        );

        // Check for unique Arabic sounds
        let has_pharyngeal_h = consonants.iter().any(|c| c.symbol == "ħ");
        let has_pharyngeal_ayn = consonants.iter().any(|c| c.symbol == "ʕ");
        let has_uvular_q = consonants.iter().any(|c| c.symbol == "q");
        assert!(
            has_pharyngeal_h && has_pharyngeal_ayn && has_uvular_q,
            "Arabic should have pharyngeal and uvular consonants"
        );
    }

    #[test]
    fn test_arabic_vowels() {
        let vowels = get_arabic_vowels();
        assert_eq!(
            vowels.len(),
            6,
            "Arabic should have 6 vowel phonemes (3 short + 3 long)"
        );

        // Check for short and long vowels
        let has_short_a = vowels.iter().any(|v| v.symbol == "a");
        let has_long_a = vowels.iter().any(|v| v.symbol == "aː");
        assert!(
            has_short_a && has_long_a,
            "Arabic should have both short and long vowels"
        );
    }

    #[test]
    fn test_arabic_pharyngealized_consonants() {
        let consonants = get_arabic_consonants();

        // Test emphatic (pharyngealized) consonants
        let has_emphatic_t = consonants.iter().any(|c| c.symbol == "tˤ");
        let has_emphatic_d = consonants.iter().any(|c| c.symbol == "dˤ");
        let has_emphatic_s = consonants.iter().any(|c| c.symbol == "sˤ");
        assert!(
            has_emphatic_t && has_emphatic_d && has_emphatic_s,
            "Arabic should have emphatic consonants"
        );
    }

    #[test]
    fn test_arabic_grapheme_mappings() {
        let mappings = get_arabic_grapheme_mappings();
        assert!(!mappings.is_empty(), "Arabic should have grapheme mappings");

        // Test some common mappings
        assert!(mappings.contains_key("ب"), "Should map ب");
        assert!(mappings.contains_key("ت"), "Should map ت");

        // Check consonant mapping
        assert_eq!(mappings.get("ب"), Some(&vec!["b".to_string()]));

        // Check that و and ي can be both consonants and vowels
        let waw_phonemes = mappings.get("و").unwrap();
        assert!(
            waw_phonemes.len() > 1,
            "و should map to multiple phonemes (w and uː)"
        );
    }
}
