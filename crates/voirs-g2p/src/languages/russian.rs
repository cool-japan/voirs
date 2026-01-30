//! Russian phoneme inventory and G2P rules.
//!
//! Russian has approximately 37 consonants (including palatalized variants)
//! and 5-6 vowels with stress-dependent pronunciation.

use super::*;
use crate::LanguageCode;

/// Get the Russian phoneme inventory
pub fn get_russian_inventory() -> PhonemeInventory {
    PhonemeInventory {
        language: LanguageCode::Ru,
        consonants: get_russian_consonants(),
        vowels: get_russian_vowels(),
        other_sounds: Vec::new(),
        grapheme_mappings: get_russian_grapheme_mappings(),
    }
}

/// Russian consonants including palatalized variants
fn get_russian_consonants() -> Vec<PhonemeInfo> {
    vec![
        // Plosives
        PhonemeInfo {
            symbol: "p".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["п".to_string()],
        },
        PhonemeInfo {
            symbol: "pʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["пь".to_string(), "пи".to_string()],
        },
        PhonemeInfo {
            symbol: "b".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["б".to_string()],
        },
        PhonemeInfo {
            symbol: "bʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["бь".to_string(), "би".to_string()],
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
            example_graphemes: vec!["т".to_string()],
        },
        PhonemeInfo {
            symbol: "tʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ть".to_string(), "ти".to_string()],
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
            example_graphemes: vec!["д".to_string()],
        },
        PhonemeInfo {
            symbol: "dʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["дь".to_string(), "ди".to_string()],
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
            example_graphemes: vec!["к".to_string()],
        },
        PhonemeInfo {
            symbol: "kʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["кь".to_string(), "ки".to_string()],
        },
        PhonemeInfo {
            symbol: "g".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["г".to_string()],
        },
        PhonemeInfo {
            symbol: "gʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Plosive),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["гь".to_string(), "ги".to_string()],
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
            example_graphemes: vec!["ф".to_string()],
        },
        PhonemeInfo {
            symbol: "fʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Labiodental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["фь".to_string(), "фи".to_string()],
        },
        PhonemeInfo {
            symbol: "v".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Labiodental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["в".to_string()],
        },
        PhonemeInfo {
            symbol: "vʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Labiodental),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["вь".to_string(), "ви".to_string()],
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
            example_graphemes: vec!["с".to_string()],
        },
        PhonemeInfo {
            symbol: "sʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["сь".to_string(), "си".to_string()],
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
            example_graphemes: vec!["з".to_string()],
        },
        PhonemeInfo {
            symbol: "zʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["зь".to_string(), "зи".to_string()],
        },
        PhonemeInfo {
            symbol: "ʃ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Postalveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ш".to_string()],
        },
        PhonemeInfo {
            symbol: "ʒ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Postalveolar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ж".to_string()],
        },
        PhonemeInfo {
            symbol: "ɕ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["щ".to_string()],
        },
        PhonemeInfo {
            symbol: "x".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Velar),
            manner: Some(MannerOfArticulation::Fricative),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["х".to_string()],
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
            example_graphemes: vec!["м".to_string()],
        },
        PhonemeInfo {
            symbol: "mʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Bilabial),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["мь".to_string(), "ми".to_string()],
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
            example_graphemes: vec!["н".to_string()],
        },
        PhonemeInfo {
            symbol: "nʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Nasal),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["нь".to_string(), "ни".to_string()],
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
            example_graphemes: vec!["л".to_string()],
        },
        PhonemeInfo {
            symbol: "lʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::LateralApproximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ль".to_string(), "ли".to_string()],
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
            example_graphemes: vec!["р".to_string()],
        },
        PhonemeInfo {
            symbol: "rʲ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Trill),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["рь".to_string(), "ри".to_string()],
        },
        // Affricates
        PhonemeInfo {
            symbol: "ts".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Alveolar),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ц".to_string()],
        },
        PhonemeInfo {
            symbol: "tɕ".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Affricate),
            voiced: false,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["ч".to_string()],
        },
        // Approximants
        PhonemeInfo {
            symbol: "j".to_string(),
            phoneme_type: PhonemeType::Consonant,
            place: Some(PlaceOfArticulation::Palatal),
            manner: Some(MannerOfArticulation::Approximant),
            voiced: true,
            height: None,
            frontness: None,
            rounded: None,
            example_graphemes: vec!["й".to_string()],
        },
    ]
}

/// Russian vowels
fn get_russian_vowels() -> Vec<PhonemeInfo> {
    vec![
        PhonemeInfo {
            symbol: "a".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Open),
            frontness: Some(VowelFrontness::Central),
            rounded: Some(false),
            example_graphemes: vec!["а".to_string()],
        },
        PhonemeInfo {
            symbol: "e".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::CloseMid),
            frontness: Some(VowelFrontness::Front),
            rounded: Some(false),
            example_graphemes: vec!["э".to_string(), "е".to_string()],
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
            example_graphemes: vec!["и".to_string(), "ы".to_string()],
        },
        PhonemeInfo {
            symbol: "o".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::CloseMid),
            frontness: Some(VowelFrontness::Back),
            rounded: Some(true),
            example_graphemes: vec!["о".to_string()],
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
            example_graphemes: vec!["у".to_string()],
        },
        PhonemeInfo {
            symbol: "ɨ".to_string(),
            phoneme_type: PhonemeType::Vowel,
            place: None,
            manner: None,
            voiced: true,
            height: Some(VowelHeight::Close),
            frontness: Some(VowelFrontness::Central),
            rounded: Some(false),
            example_graphemes: vec!["ы".to_string()],
        },
    ]
}

/// Russian grapheme to phoneme mappings
fn get_russian_grapheme_mappings() -> HashMap<String, Vec<String>> {
    let mut mappings = HashMap::new();

    // Vowels
    mappings.insert("а".to_string(), vec!["a".to_string()]);
    mappings.insert("е".to_string(), vec!["je".to_string(), "e".to_string()]);
    mappings.insert("ё".to_string(), vec!["jo".to_string()]);
    mappings.insert("и".to_string(), vec!["i".to_string()]);
    mappings.insert("о".to_string(), vec!["o".to_string()]);
    mappings.insert("у".to_string(), vec!["u".to_string()]);
    mappings.insert("ы".to_string(), vec!["ɨ".to_string()]);
    mappings.insert("э".to_string(), vec!["e".to_string()]);
    mappings.insert("ю".to_string(), vec!["ju".to_string()]);
    mappings.insert("я".to_string(), vec!["ja".to_string()]);

    // Consonants
    mappings.insert("б".to_string(), vec!["b".to_string()]);
    mappings.insert("в".to_string(), vec!["v".to_string()]);
    mappings.insert("г".to_string(), vec!["g".to_string()]);
    mappings.insert("д".to_string(), vec!["d".to_string()]);
    mappings.insert("ж".to_string(), vec!["ʒ".to_string()]);
    mappings.insert("з".to_string(), vec!["z".to_string()]);
    mappings.insert("й".to_string(), vec!["j".to_string()]);
    mappings.insert("к".to_string(), vec!["k".to_string()]);
    mappings.insert("л".to_string(), vec!["l".to_string()]);
    mappings.insert("м".to_string(), vec!["m".to_string()]);
    mappings.insert("н".to_string(), vec!["n".to_string()]);
    mappings.insert("п".to_string(), vec!["p".to_string()]);
    mappings.insert("р".to_string(), vec!["r".to_string()]);
    mappings.insert("с".to_string(), vec!["s".to_string()]);
    mappings.insert("т".to_string(), vec!["t".to_string()]);
    mappings.insert("ф".to_string(), vec!["f".to_string()]);
    mappings.insert("х".to_string(), vec!["x".to_string()]);
    mappings.insert("ц".to_string(), vec!["ts".to_string()]);
    mappings.insert("ч".to_string(), vec!["tɕ".to_string()]);
    mappings.insert("ш".to_string(), vec!["ʃ".to_string()]);
    mappings.insert("щ".to_string(), vec!["ɕ".to_string()]);

    mappings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_russian_consonants() {
        let consonants = get_russian_consonants();
        assert!(!consonants.is_empty(), "Russian should have consonants");

        // Check for hard/soft pairs
        let has_hard_p = consonants.iter().any(|c| c.symbol == "p");
        let has_soft_p = consonants.iter().any(|c| c.symbol == "pʲ");
        assert!(
            has_hard_p && has_soft_p,
            "Russian should have both hard and soft p"
        );
    }

    #[test]
    fn test_russian_vowels() {
        let vowels = get_russian_vowels();
        assert_eq!(vowels.len(), 6, "Russian should have 6 vowel phonemes");

        // Check for basic vowels
        let has_a = vowels.iter().any(|v| v.symbol == "a");
        let has_i = vowels.iter().any(|v| v.symbol == "i");
        let has_u = vowels.iter().any(|v| v.symbol == "u");
        assert!(
            has_a && has_i && has_u,
            "Russian should have a, i, u vowels"
        );
    }

    #[test]
    fn test_russian_grapheme_mappings() {
        let mappings = get_russian_grapheme_mappings();
        assert!(
            !mappings.is_empty(),
            "Russian should have grapheme mappings"
        );

        // Test some common mappings
        assert!(mappings.contains_key("а"), "Should map а");
        assert!(mappings.contains_key("п"), "Should map п");

        // Check that consonants map correctly
        assert_eq!(mappings.get("п"), Some(&vec!["p".to_string()]));
    }
}
