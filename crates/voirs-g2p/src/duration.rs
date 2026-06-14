//! Rule-based (Klatt-style) phoneme duration model.
//!
//! This module implements a self-contained, deterministic duration model for
//! individual phonemes. It does **not** rely on a trained neural model or any
//! external data file; durations are derived from linguistic rules in the
//! tradition of Klatt (1979), *"Synthesis by rule of segmental durations in
//! English sentences"*.
//!
//! The model assigns each phoneme an inherent (context-free) base duration that
//! depends on its broad articulatory class, then applies multiplicative
//! adjustments for:
//!
//! * **Stress** – stressed syllables are lengthened.
//! * **Phrase / utterance boundaries** – segments immediately before a boundary
//!   undergo pre-pausal lengthening.
//! * **Syllable position** – codas/finals lengthen slightly, onsets shorten.
//! * **Speaking rate** – a global tempo factor scales every duration.
//!
//! The final value is clamped to a perceptually sane range so that no single
//! segment becomes degenerate or unnaturally long.
//!
//! Classification prefers the structured [`PhoneticFeatures`] attached to a
//! [`Phoneme`] when available, and otherwise falls back to symbol-based
//! classification covering IPA, ARPAbet and bare ASCII notations.

use crate::{Phoneme, PhoneticFeatures, SyllablePosition};

/// Minimum admissible phoneme duration in milliseconds.
pub(crate) const MIN_DURATION_MS: f32 = 30.0;

/// Maximum admissible phoneme duration in milliseconds.
pub(crate) const MAX_DURATION_MS: f32 = 300.0;

/// Broad articulatory class used to select an inherent base duration.
///
/// The ordering of the inherent durations (see [`DurationClass::base_duration_ms`])
/// follows the canonical sonority/length hierarchy used by VoiRS:
/// long/tense vowels > short/lax vowels > diphthongs > nasals > liquids >
/// glides > fricatives > affricates > stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurationClass {
    /// Long / tense monophthong vowels (e.g. /iː/, /ɑ/, /u/, /ɔ/).
    LongVowel,
    /// Short / lax monophthong vowels (e.g. /ɪ/, /ʊ/, /ɛ/, /ʌ/, /ə/).
    ShortVowel,
    /// Diphthongs (e.g. /aɪ/, /aʊ/, /ɔɪ/, /eɪ/, /oʊ/).
    Diphthong,
    /// Nasal consonants (e.g. /m/, /n/, /ŋ/).
    Nasal,
    /// Liquids: laterals and rhotics (e.g. /l/, /r/, /ɹ/, /ɾ/).
    Liquid,
    /// Glides / semivowels (e.g. /w/, /j/).
    Glide,
    /// Fricatives (e.g. /f/, /v/, /s/, /z/, /ʃ/, /θ/, /h/).
    Fricative,
    /// Affricates (e.g. /tʃ/, /dʒ/).
    Affricate,
    /// Stops / plosives (e.g. /p/, /b/, /t/, /d/, /k/, /g/).
    Stop,
}

impl DurationClass {
    /// Inherent, context-free base duration of the class in milliseconds.
    ///
    /// These values are Klatt-style inherent durations: vowels carry the most
    /// duration, sonorant consonants (nasals, liquids, glides) sit in the
    /// middle, and obstruents (fricatives, affricates, stops) are shortest.
    pub(crate) fn base_duration_ms(self) -> f32 {
        match self {
            DurationClass::LongVowel => 145.0,
            DurationClass::ShortVowel => 115.0,
            DurationClass::Diphthong => 105.0,
            DurationClass::Nasal => 80.0,
            DurationClass::Liquid => 75.0,
            DurationClass::Glide => 70.0,
            DurationClass::Fricative => 65.0,
            DurationClass::Affricate => 55.0,
            DurationClass::Stop => 45.0,
        }
    }

    /// Whether the class is a vocalic (vowel-like) segment.
    fn is_vocalic(self) -> bool {
        matches!(
            self,
            DurationClass::LongVowel | DurationClass::ShortVowel | DurationClass::Diphthong
        )
    }
}

/// Position of a phoneme relative to a prosodic (phrase/utterance) boundary.
///
/// Used to model pre-pausal (phrase-final) lengthening, where segments are
/// progressively lengthened as they approach a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundaryContext {
    /// The phoneme is the last segment before a phrase / utterance boundary.
    PhraseFinal,
    /// The phoneme is the penultimate segment before a boundary.
    PreFinal,
    /// The phoneme is phrase-medial (no nearby boundary).
    Medial,
}

/// Estimate the duration of a single phoneme using the rule-based model.
///
/// * `phoneme` – the phoneme whose duration is estimated.
/// * `boundary` – the phrase/utterance boundary context for final lengthening.
/// * `speaking_rate` – multiplicative tempo factor where `1.0` is normal speed,
///   values `> 1.0` speed up (shorten) and values `< 1.0` slow down (lengthen).
///   Non-finite or non-positive values are treated as `1.0`.
///
/// The returned value is always within
/// `[MIN_DURATION_MS, MAX_DURATION_MS]`.
pub(crate) fn estimate_phoneme_duration_ms(
    phoneme: &Phoneme,
    boundary: BoundaryContext,
    speaking_rate: f32,
) -> f32 {
    let class = classify_duration_class(phoneme);
    let mut duration = class.base_duration_ms();

    // Stress-conditioned lengthening: stressed segments are longer.
    duration *= stress_factor(phoneme.stress);

    // Pre-pausal (phrase/utterance-final) lengthening.
    duration *= boundary_factor(boundary, class);

    // Syllable-position fine adjustment.
    duration *= syllable_position_factor(&phoneme.syllable_position);

    // Global speaking-rate scaling (faster rate -> shorter durations).
    let rate = if speaking_rate.is_finite() && speaking_rate > 0.0 {
        speaking_rate
    } else {
        1.0
    };
    duration /= rate;

    duration.clamp(MIN_DURATION_MS, MAX_DURATION_MS)
}

/// Multiplicative lengthening factor for a phoneme's stress level.
///
/// `stress` follows the [`Phoneme::stress`] convention:
/// `0` = unstressed, `1` = primary, `2` = secondary, `3` = tertiary.
fn stress_factor(stress: u8) -> f32 {
    match stress {
        1 => 1.40, // primary stress: ~+40%
        2 => 1.20, // secondary stress: ~+20%
        3 => 1.10, // tertiary stress: ~+10%
        _ => 1.0,  // unstressed (or unknown)
    }
}

/// Multiplicative lengthening factor for the boundary context.
///
/// Pre-pausal lengthening affects vocalic segments more strongly than
/// consonants, mirroring the well-documented asymmetry in final lengthening.
fn boundary_factor(boundary: BoundaryContext, class: DurationClass) -> f32 {
    match boundary {
        BoundaryContext::PhraseFinal => {
            if class.is_vocalic() {
                1.50
            } else {
                1.35
            }
        }
        BoundaryContext::PreFinal => {
            if class.is_vocalic() {
                1.18
            } else {
                1.10
            }
        }
        BoundaryContext::Medial => 1.0,
    }
}

/// Multiplicative adjustment for the phoneme's position within its syllable.
fn syllable_position_factor(position: &SyllablePosition) -> f32 {
    match position {
        SyllablePosition::Onset => 0.95,
        SyllablePosition::Nucleus => 1.0,
        SyllablePosition::Coda => 1.05,
        SyllablePosition::Final => 1.10,
        SyllablePosition::Standalone => 1.0,
    }
}

/// Classify a phoneme into a broad [`DurationClass`].
///
/// Structured [`PhoneticFeatures`] are preferred when present; otherwise the
/// phoneme's IPA / ARPAbet / ASCII symbol is used.
pub(crate) fn classify_duration_class(phoneme: &Phoneme) -> DurationClass {
    let symbol = phoneme.ipa_symbol.as_deref().unwrap_or(&phoneme.symbol);

    if let Some(features) = &phoneme.phonetic_features {
        if let Some(class) = classify_from_features(features, symbol) {
            return class;
        }
    }

    classify_from_symbol(symbol)
}

/// Classify using structured phonetic features (manner of articulation).
///
/// Vowel sub-classification (long / short / diphthong) is delegated to the
/// symbol classifier, since the manner alone (`"vowel"`) does not distinguish
/// length.
fn classify_from_features(features: &PhoneticFeatures, symbol: &str) -> Option<DurationClass> {
    let manner = features.manner.as_deref()?.to_ascii_lowercase();
    let class = match manner.as_str() {
        "vowel" | "monophthong" | "diphthong" => {
            // Resolve vowel length from the symbol; fall back to short.
            let resolved = classify_from_symbol(symbol);
            if resolved.is_vocalic() {
                resolved
            } else {
                DurationClass::ShortVowel
            }
        }
        "plosive" | "stop" => DurationClass::Stop,
        "fricative" | "sibilant" => DurationClass::Fricative,
        "affricate" => DurationClass::Affricate,
        "nasal" => DurationClass::Nasal,
        "glide" | "semivowel" => DurationClass::Glide,
        "lateral" | "liquid" | "trill" | "tap" | "flap" | "rhotic" => DurationClass::Liquid,
        "approximant" => {
            // Approximants split into glides (/w/, /j/) and liquids (/l/, /r/).
            match classify_from_symbol(symbol) {
                DurationClass::Glide => DurationClass::Glide,
                _ => DurationClass::Liquid,
            }
        }
        _ => return None,
    };
    Some(class)
}

/// Classify a phoneme purely from its written symbol.
///
/// Handles IPA (including length-marked vowels and multi-character
/// affricates/diphthongs), ARPAbet (upper-case with optional stress digits) and
/// bare ASCII letters.
fn classify_from_symbol(symbol: &str) -> DurationClass {
    let has_length_mark = symbol.contains('ː') || symbol.contains('ˑ');

    // Strip stress, length and separator diacritics for matching.
    let core: String = symbol
        .chars()
        .filter(|c| !matches!(c, 'ˈ' | 'ˌ' | 'ː' | 'ˑ' | '.' | ' ' | '\u{0361}'))
        .collect();

    // ARPAbet (upper-case ASCII, e.g. "AA1", "CH", "NG").
    if let Some(class) = classify_arpabet(&core) {
        return class;
    }

    // Diphthongs: explicit sequences or two leading vowel characters.
    if is_diphthong(&core) {
        return DurationClass::Diphthong;
    }

    // Multi-character affricates (e.g. "tʃ", "dʒ", "ts").
    if is_affricate(&core) {
        return DurationClass::Affricate;
    }

    // Single representative character (ASCII-folded so stray capitals match).
    let first = core.chars().next().unwrap_or(' ').to_ascii_lowercase();
    let mut class = classify_char(first);

    // An explicit length mark promotes a short vowel to a long vowel.
    if has_length_mark && class == DurationClass::ShortVowel {
        class = DurationClass::LongVowel;
    }

    class
}

/// Classify an ARPAbet token, or return `None` if it is not ARPAbet.
fn classify_arpabet(core: &str) -> Option<DurationClass> {
    let letters: String = core.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() || !letters.chars().all(|c| c.is_ascii_uppercase()) {
        return None;
    }

    let class = match letters.as_str() {
        // Diphthongs.
        "AW" | "AY" | "EY" | "OW" | "OY" => DurationClass::Diphthong,
        // Long / tense vowels.
        "AA" | "AE" | "AO" | "ER" | "IY" | "UW" => DurationClass::LongVowel,
        // Short / lax vowels.
        "AH" | "AX" | "EH" | "IH" | "UH" => DurationClass::ShortVowel,
        // Nasals.
        "M" | "N" | "NG" | "EM" | "EN" => DurationClass::Nasal,
        // Liquids.
        "L" | "R" | "EL" | "DX" => DurationClass::Liquid,
        // Glides.
        "W" | "Y" => DurationClass::Glide,
        // Affricates.
        "CH" | "JH" => DurationClass::Affricate,
        // Fricatives.
        "F" | "V" | "TH" | "DH" | "S" | "Z" | "SH" | "ZH" | "HH" => DurationClass::Fricative,
        // Stops / plosives.
        "P" | "B" | "T" | "D" | "K" | "G" | "Q" => DurationClass::Stop,
        _ => return None,
    };
    Some(class)
}

/// Whether `core` denotes a diphthong.
fn is_diphthong(core: &str) -> bool {
    const DIPHTHONGS: &[&str] = &[
        "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ", "əʊ", "ɪə", "eə", "ʊə", "ɛə", "ɔə", "aɪə", "aʊə", "ai", "au",
        "oi", "ei", "ou", "ay", "ey", "oy", "ow", "aw",
    ];
    if DIPHTHONGS.contains(&core) {
        return true;
    }

    // Generic rule: two or more characters whose first two are both vowels.
    let mut chars = core.chars();
    match (chars.next(), chars.next()) {
        (Some(first), Some(second)) => is_vowel_char(first) && is_vowel_char(second),
        _ => false,
    }
}

/// Whether `core` denotes an affricate.
fn is_affricate(core: &str) -> bool {
    const AFFRICATES: &[&str] = &[
        "tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ", "ʈʂ", "ɖʐ", "pf", "cç", "ɟʝ", "kx", "ch", "jh",
    ];
    if AFFRICATES.contains(&core) {
        return true;
    }
    // Single-character affricate ligatures.
    let mut chars = core.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => matches!(c, 'ʧ' | 'ʤ' | 'ʦ' | 'ʣ' | 'ʨ' | 'ʥ'),
        _ => false,
    }
}

/// Whether `c` is a vowel character (IPA or ASCII).
fn is_vowel_char(c: char) -> bool {
    matches!(
        c,
        'a' | 'e'
            | 'i'
            | 'o'
            | 'u'
            | 'y'
            | 'ɑ'
            | 'ɒ'
            | 'ɔ'
            | 'æ'
            | 'ə'
            | 'ɛ'
            | 'ɜ'
            | 'ɝ'
            | 'ɞ'
            | 'ɐ'
            | 'ɘ'
            | 'ɵ'
            | 'ø'
            | 'œ'
            | 'ɶ'
            | 'ɪ'
            | 'ʊ'
            | 'ʉ'
            | 'ɨ'
            | 'ɯ'
            | 'ʌ'
            | 'ɤ'
            | 'ɚ'
    )
}

/// Classify a single (ASCII-folded) phoneme character.
fn classify_char(c: char) -> DurationClass {
    match c {
        // Long / tense vowels (IPA + bare ASCII).
        'i' | 'u' | 'o' | 'a' | 'ɑ' | 'ɒ' | 'ɔ' | 'æ' | 'ɜ' | 'ɝ' | 'ø' | 'œ' | 'ɶ' | 'ʉ' | 'ɨ'
        | 'ɯ' | 'ɤ' | 'e' => DurationClass::LongVowel,

        // Short / lax vowels.
        'ɪ' | 'ʊ' | 'ɛ' | 'ʌ' | 'ə' | 'ɐ' | 'ɘ' | 'ɵ' | 'ɞ' | 'ɚ' | 'y' => {
            DurationClass::ShortVowel
        }

        // Nasals.
        'm' | 'n' | 'ŋ' | 'ɲ' | 'ɳ' | 'ɱ' | 'ɴ' => DurationClass::Nasal,

        // Liquids (laterals + rhotics).
        'l' | 'ɭ' | 'ʎ' | 'ʟ' | 'r' | 'ɹ' | 'ɾ' | 'ɻ' | 'ʀ' | 'ʁ' | 'ɽ' | 'ɺ' | 'ɫ' => {
            DurationClass::Liquid
        }

        // Glides / semivowels.
        'w' | 'j' | 'ɥ' | 'ɰ' | 'ʍ' => DurationClass::Glide,

        // Fricatives.
        'f' | 'v' | 'θ' | 'ð' | 's' | 'z' | 'ʃ' | 'ʒ' | 'ç' | 'ʝ' | 'x' | 'ɣ' | 'χ' | 'h' | 'ɦ'
        | 'ɸ' | 'β' | 'ħ' | 'ʕ' | 'ʜ' | 'ʢ' | 'ɕ' | 'ʑ' => DurationClass::Fricative,

        // Stops / plosives.
        'p' | 'b' | 't' | 'd' | 'k' | 'g' | 'q' | 'ɡ' | 'ʔ' | 'ɢ' | 'ʈ' | 'ɖ' | 'c' | 'ɟ' => {
            DurationClass::Stop
        }

        // Unknown symbols default to a short, neutral duration.
        _ => DurationClass::ShortVowel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phoneme;

    /// Tiny deterministic linear-congruential generator for reproducible
    /// fuzz-style coverage without pulling in an RNG dependency.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u32(&mut self) -> u32 {
            // Constants from Knuth's MMIX LCG.
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        /// Uniform float in `[0, 1)`.
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
        }
    }

    fn phoneme(symbol: &str) -> Phoneme {
        Phoneme::new(symbol)
    }

    fn medial(symbol: &str) -> f32 {
        estimate_phoneme_duration_ms(&phoneme(symbol), BoundaryContext::Medial, 1.0)
    }

    #[test]
    fn vowel_is_longer_than_stop() {
        let long_vowel = medial("ɑ");
        let short_vowel = medial("ɪ");
        let stop = medial("t");

        assert!(
            long_vowel > stop,
            "long vowel {long_vowel} should exceed stop {stop}"
        );
        assert!(
            short_vowel > stop,
            "short vowel {short_vowel} should exceed stop {stop}"
        );
        // Every plosive should be classified as a (short) stop.
        for s in ["p", "b", "t", "d", "k", "g"] {
            assert!(
                medial("i") > medial(s),
                "vowel should exceed stop {s} ({})",
                medial(s)
            );
        }
    }

    #[test]
    fn stressed_is_longer_than_unstressed() {
        for symbol in ["ɑ", "ɪ", "n", "s", "t"] {
            let mut unstressed = phoneme(symbol);
            unstressed.stress = 0;
            let mut stressed = phoneme(symbol);
            stressed.stress = 1;

            let unstressed_ms =
                estimate_phoneme_duration_ms(&unstressed, BoundaryContext::Medial, 1.0);
            let stressed_ms = estimate_phoneme_duration_ms(&stressed, BoundaryContext::Medial, 1.0);

            assert!(
                stressed_ms > unstressed_ms,
                "stressed {symbol} ({stressed_ms}) should exceed unstressed ({unstressed_ms})"
            );
        }
    }

    #[test]
    fn phrase_final_is_longer_than_medial() {
        for symbol in ["ɑ", "ɪ", "n", "s", "t"] {
            let p = phoneme(symbol);
            let final_ms = estimate_phoneme_duration_ms(&p, BoundaryContext::PhraseFinal, 1.0);
            let prefinal_ms = estimate_phoneme_duration_ms(&p, BoundaryContext::PreFinal, 1.0);
            let medial_ms = estimate_phoneme_duration_ms(&p, BoundaryContext::Medial, 1.0);

            assert!(
                final_ms > medial_ms,
                "phrase-final {symbol} ({final_ms}) should exceed medial ({medial_ms})"
            );
            assert!(
                final_ms >= prefinal_ms,
                "phrase-final {symbol} ({final_ms}) should be >= pre-final ({prefinal_ms})"
            );
            assert!(
                prefinal_ms >= medial_ms,
                "pre-final {symbol} ({prefinal_ms}) should be >= medial ({medial_ms})"
            );
        }
    }

    #[test]
    fn faster_speaking_rate_shortens_duration() {
        let p = phoneme("ɑ");
        let normal = estimate_phoneme_duration_ms(&p, BoundaryContext::Medial, 1.0);
        let fast = estimate_phoneme_duration_ms(&p, BoundaryContext::Medial, 2.0);
        let slow = estimate_phoneme_duration_ms(&p, BoundaryContext::Medial, 0.5);

        assert!(fast < normal, "fast {fast} should be < normal {normal}");
        assert!(slow > normal, "slow {slow} should be > normal {normal}");
    }

    #[test]
    fn non_finite_rate_is_treated_as_normal() {
        let p = phoneme("ɑ");
        let normal = estimate_phoneme_duration_ms(&p, BoundaryContext::Medial, 1.0);
        for bad in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let value = estimate_phoneme_duration_ms(&p, BoundaryContext::Medial, bad);
            assert!(
                (value - normal).abs() < 1e-3,
                "rate {bad} should fall back to 1.0"
            );
        }
    }

    #[test]
    fn all_durations_within_clamp_bounds() {
        let symbols = [
            "ɑ", "i", "u", "ɔ", "ɪ", "ʊ", "ɛ", "ʌ", "ə", "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ", "m", "n",
            "ŋ", "l", "r", "ɹ", "w", "j", "f", "v", "s", "z", "ʃ", "θ", "h", "tʃ", "dʒ", "p", "b",
            "t", "d", "k", "g", "AA1", "IY2", "CH", "NG", "hello", ",", "?", "",
        ];
        let boundaries = [
            BoundaryContext::Medial,
            BoundaryContext::PreFinal,
            BoundaryContext::PhraseFinal,
        ];
        let positions = [
            SyllablePosition::Onset,
            SyllablePosition::Nucleus,
            SyllablePosition::Coda,
            SyllablePosition::Final,
            SyllablePosition::Standalone,
        ];

        let mut rng = Lcg::new(0x5eed_1234_abcd_ef01);
        for symbol in symbols {
            for stress in 0u8..=3 {
                for boundary in boundaries {
                    for position in &positions {
                        // Mix in deterministic random rates spanning extremes.
                        let rate = 0.25 + rng.next_f32() * 3.75;
                        let mut p = phoneme(symbol);
                        p.stress = stress;
                        p.syllable_position = position.clone();
                        let duration = estimate_phoneme_duration_ms(&p, boundary, rate);
                        assert!(
                            (MIN_DURATION_MS..=MAX_DURATION_MS).contains(&duration),
                            "duration {duration} out of bounds for {symbol:?} \
                             stress={stress} rate={rate}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn base_duration_ordering_follows_hierarchy() {
        let ordered = [
            DurationClass::LongVowel,
            DurationClass::ShortVowel,
            DurationClass::Diphthong,
            DurationClass::Nasal,
            DurationClass::Liquid,
            DurationClass::Glide,
            DurationClass::Fricative,
            DurationClass::Affricate,
            DurationClass::Stop,
        ];
        for pair in ordered.windows(2) {
            assert!(
                pair[0].base_duration_ms() > pair[1].base_duration_ms(),
                "{:?} base should exceed {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn classification_covers_each_class() {
        assert_eq!(
            classify_duration_class(&phoneme("ɑ")),
            DurationClass::LongVowel
        );
        assert_eq!(
            classify_duration_class(&phoneme("iː")),
            DurationClass::LongVowel
        );
        assert_eq!(
            classify_duration_class(&phoneme("ɪ")),
            DurationClass::ShortVowel
        );
        assert_eq!(
            classify_duration_class(&phoneme("aɪ")),
            DurationClass::Diphthong
        );
        assert_eq!(
            classify_duration_class(&phoneme("eɪ")),
            DurationClass::Diphthong
        );
        assert_eq!(classify_duration_class(&phoneme("m")), DurationClass::Nasal);
        assert_eq!(
            classify_duration_class(&phoneme("l")),
            DurationClass::Liquid
        );
        assert_eq!(
            classify_duration_class(&phoneme("r")),
            DurationClass::Liquid
        );
        assert_eq!(classify_duration_class(&phoneme("w")), DurationClass::Glide);
        assert_eq!(classify_duration_class(&phoneme("j")), DurationClass::Glide);
        assert_eq!(
            classify_duration_class(&phoneme("s")),
            DurationClass::Fricative
        );
        assert_eq!(
            classify_duration_class(&phoneme("θ")),
            DurationClass::Fricative
        );
        assert_eq!(
            classify_duration_class(&phoneme("tʃ")),
            DurationClass::Affricate
        );
        assert_eq!(
            classify_duration_class(&phoneme("dʒ")),
            DurationClass::Affricate
        );
        assert_eq!(classify_duration_class(&phoneme("t")), DurationClass::Stop);
        assert_eq!(classify_duration_class(&phoneme("g")), DurationClass::Stop);
    }

    #[test]
    fn arpabet_classification() {
        assert_eq!(
            classify_duration_class(&phoneme("AA1")),
            DurationClass::LongVowel
        );
        assert_eq!(
            classify_duration_class(&phoneme("IH0")),
            DurationClass::ShortVowel
        );
        assert_eq!(
            classify_duration_class(&phoneme("AY1")),
            DurationClass::Diphthong
        );
        assert_eq!(
            classify_duration_class(&phoneme("NG")),
            DurationClass::Nasal
        );
        assert_eq!(
            classify_duration_class(&phoneme("CH")),
            DurationClass::Affricate
        );
        assert_eq!(
            classify_duration_class(&phoneme("SH")),
            DurationClass::Fricative
        );
        assert_eq!(classify_duration_class(&phoneme("P")), DurationClass::Stop);
    }

    #[test]
    fn structured_features_take_precedence() {
        // A symbol that would classify as a stop, but tagged as a nasal.
        let mut p = phoneme("t");
        p.phonetic_features = Some(PhoneticFeatures::consonant("nasal", "alveolar", true));
        assert_eq!(classify_duration_class(&p), DurationClass::Nasal);

        // Vowel manner: length resolved from the (long) symbol.
        let mut v = phoneme("ɑ");
        v.phonetic_features = Some(PhoneticFeatures::vowel("low", "back", false));
        assert_eq!(classify_duration_class(&v), DurationClass::LongVowel);
    }
}
