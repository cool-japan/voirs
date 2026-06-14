//! IPA / ARPAbet distinctive-feature matrix for phoneme embeddings.
//!
//! This module maps phoneme symbols (both the IPA inventory and the CMU/ARPAbet
//! symbol set) onto a fixed-width vector of standard *distinctive features*. It
//! replaces the previous 15-symbol, four-dimensional placeholder table with a
//! linguistically grounded matrix covering the common cross-linguistic
//! inventory (~80 phoneme classes, ~150 surface spellings once ARPAbet/IPA
//! aliases are counted).
//!
//! # Feature layout (`FEATURE_DIM` = 20)
//!
//! Each phoneme is described by the following fixed dimensions. Binary features
//! use `1.0`/`0.0`; vowel height/backness are graded in `[0.0, 1.0]`.
//!
//! | idx | name                | meaning                                                        |
//! |-----|---------------------|----------------------------------------------------------------|
//! | 0   | `SYLLABIC`          | `1.0` for vowels / syllabic segments, else `0.0`               |
//! | 1   | `CONSONANTAL`       | `1.0` for true consonants; `0.0` for vowels and glides         |
//! | 2   | `SONORANT`          | `1.0` for vowels, glides, liquids, nasals; `0.0` obstruents    |
//! | 3   | `CONTINUANT`        | `1.0` for vowels, glides, liquids, fricatives; `0.0` stops etc |
//! | 4   | `VOICED`            | `1.0` voiced, `0.0` voiceless                                  |
//! | 5   | `MANNER_PLOSIVE`    | `1.0` if a plosive / stop                                      |
//! | 6   | `MANNER_FRICATIVE`  | `1.0` if a fricative                                           |
//! | 7   | `MANNER_AFFRICATE`  | `1.0` if an affricate                                          |
//! | 8   | `MANNER_NASAL`      | `1.0` if a nasal                                               |
//! | 9   | `MANNER_APPROXIMANT`| `1.0` if an approximant (liquid or glide)                      |
//! | 10  | `LATERAL`           | `1.0` if lateral (e.g. /l/)                                    |
//! | 11  | `RHOTIC`            | `1.0` if rhotic (e.g. /ɹ/, /r/, /ɾ/, r-coloured vowels)        |
//! | 12  | `PLACE_LABIAL`      | `1.0` bilabial / labiodental                                   |
//! | 13  | `PLACE_CORONAL`     | `1.0` dental / alveolar / postalveolar / retroflex             |
//! | 14  | `PLACE_DORSAL`      | `1.0` palatal / velar / uvular                                 |
//! | 15  | `PLACE_GLOTTAL`     | `1.0` glottal / pharyngeal                                     |
//! | 16  | `VOWEL_HEIGHT`      | `0.0` open/low … `1.0` close/high (`0.0` for consonants)       |
//! | 17  | `VOWEL_BACKNESS`    | `0.0` front … `1.0` back (`0.0` for consonants)                |
//! | 18  | `VOWEL_ROUNDED`     | `1.0` rounded vowel, else `0.0`                                |
//! | 19  | `VOWEL_TENSE`       | `1.0` tense vowel, `0.0` lax / consonant                       |
//!
//! Unknown symbols map to the documented neutral vector ([`neutral_features`]),
//! which is all-zeros: no distinctive feature is asserted.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Width of the distinctive-feature vector produced for every phoneme.
pub(super) const FEATURE_DIM: usize = 20;

// --- Feature indices (see module docs for the full description). ------------
const F_SYLLABIC: usize = 0;
const F_CONSONANTAL: usize = 1;
const F_SONORANT: usize = 2;
const F_CONTINUANT: usize = 3;
const F_VOICED: usize = 4;
const F_PLOSIVE: usize = 5;
const F_FRICATIVE: usize = 6;
const F_AFFRICATE: usize = 7;
const F_NASAL: usize = 8;
const F_APPROXIMANT: usize = 9;
const F_LATERAL: usize = 10;
const F_RHOTIC: usize = 11;
const F_PLACE_LABIAL: usize = 12;
const F_PLACE_CORONAL: usize = 13;
const F_PLACE_DORSAL: usize = 14;
const F_PLACE_GLOTTAL: usize = 15;
const F_VOWEL_HEIGHT: usize = 16;
const F_VOWEL_BACKNESS: usize = 17;
const F_VOWEL_ROUNDED: usize = 18;
const F_VOWEL_TENSE: usize = 19;

/// Neutral / "unknown phoneme" vector: no feature asserted.
const NEUTRAL: [f32; FEATURE_DIM] = [0.0; FEATURE_DIM];

/// Manner of articulation for consonants.
#[derive(Clone, Copy)]
enum Manner {
    /// Oral stop (e.g. /p/, /b/, /t/, /k/).
    Plosive,
    /// Fricative (e.g. /f/, /s/, /ʃ/).
    Fricative,
    /// Affricate (e.g. /tʃ/, /dʒ/).
    Affricate,
    /// Nasal stop (e.g. /m/, /n/, /ŋ/).
    Nasal,
    /// Lateral approximant (e.g. /l/).
    Lateral,
    /// Rhotic approximant / trill / tap (e.g. /ɹ/, /r/, /ɾ/).
    Rhotic,
    /// Non-syllabic vocoid / glide (e.g. /w/, /j/) — these are `[-consonantal]`.
    Glide,
}

/// Place of articulation for consonants.
#[derive(Clone, Copy)]
enum Place {
    /// Bilabial / labiodental.
    Labial,
    /// Dental / alveolar / postalveolar / retroflex.
    Coronal,
    /// Palatal / velar / uvular.
    Dorsal,
    /// Glottal / pharyngeal.
    Glottal,
    /// Labial-velar double articulation (e.g. /w/).
    Labiovelar,
}

/// Build the distinctive-feature vector for a consonant.
fn consonant(voiced: bool, manner: Manner, place: Place) -> [f32; FEATURE_DIM] {
    let mut f = NEUTRAL;

    // Major class features.
    f[F_CONSONANTAL] = match manner {
        Manner::Glide => 0.0,
        _ => 1.0,
    };
    f[F_SONORANT] = match manner {
        Manner::Plosive | Manner::Fricative | Manner::Affricate => 0.0,
        Manner::Nasal | Manner::Lateral | Manner::Rhotic | Manner::Glide => 1.0,
    };
    f[F_CONTINUANT] = match manner {
        Manner::Fricative | Manner::Lateral | Manner::Rhotic | Manner::Glide => 1.0,
        Manner::Plosive | Manner::Affricate | Manner::Nasal => 0.0,
    };
    if voiced {
        f[F_VOICED] = 1.0;
    }

    // Manner one-hot (+ lateral / rhotic sub-features).
    match manner {
        Manner::Plosive => f[F_PLOSIVE] = 1.0,
        Manner::Fricative => f[F_FRICATIVE] = 1.0,
        Manner::Affricate => f[F_AFFRICATE] = 1.0,
        Manner::Nasal => f[F_NASAL] = 1.0,
        Manner::Lateral => {
            f[F_APPROXIMANT] = 1.0;
            f[F_LATERAL] = 1.0;
        }
        Manner::Rhotic => {
            f[F_APPROXIMANT] = 1.0;
            f[F_RHOTIC] = 1.0;
        }
        Manner::Glide => f[F_APPROXIMANT] = 1.0,
    }

    // Place (multi-hot for the labial-velar double articulation).
    match place {
        Place::Labial => f[F_PLACE_LABIAL] = 1.0,
        Place::Coronal => f[F_PLACE_CORONAL] = 1.0,
        Place::Dorsal => f[F_PLACE_DORSAL] = 1.0,
        Place::Glottal => f[F_PLACE_GLOTTAL] = 1.0,
        Place::Labiovelar => {
            f[F_PLACE_LABIAL] = 1.0;
            f[F_PLACE_DORSAL] = 1.0;
        }
    }

    f
}

/// Build the distinctive-feature vector for a vowel.
///
/// `height` and `backness` are graded in `[0.0, 1.0]` (see the module docs).
fn vowel(
    height: f32,
    backness: f32,
    rounded: bool,
    tense: bool,
    rhotic: bool,
) -> [f32; FEATURE_DIM] {
    let mut f = NEUTRAL;
    f[F_SYLLABIC] = 1.0;
    f[F_SONORANT] = 1.0;
    f[F_CONTINUANT] = 1.0;
    f[F_VOICED] = 1.0;
    f[F_VOWEL_HEIGHT] = height;
    f[F_VOWEL_BACKNESS] = backness;
    if rounded {
        f[F_VOWEL_ROUNDED] = 1.0;
    }
    if tense {
        f[F_VOWEL_TENSE] = 1.0;
    }
    if rhotic {
        f[F_RHOTIC] = 1.0;
    }
    f
}

/// Lazily-initialised lookup table from a (normalised) phoneme symbol to its
/// distinctive-feature vector.
fn feature_table() -> &'static HashMap<&'static str, [f32; FEATURE_DIM]> {
    static TABLE: OnceLock<HashMap<&'static str, [f32; FEATURE_DIM]>> = OnceLock::new();
    TABLE.get_or_init(build_table)
}

/// Construct the static phoneme → feature mapping.
///
/// Each entry lists every surface spelling (ARPAbet upper-case, IPA, ligatures)
/// that should resolve to the same feature vector.
fn build_table() -> HashMap<&'static str, [f32; FEATURE_DIM]> {
    use Manner::{Affricate, Fricative, Glide, Lateral, Nasal, Plosive, Rhotic};
    use Place::{Coronal, Dorsal, Glottal, Labial, Labiovelar};

    let mut m: HashMap<&'static str, [f32; FEATURE_DIM]> = HashMap::new();

    {
        let mut add = |aliases: &[&'static str], v: [f32; FEATURE_DIM]| {
            for &a in aliases {
                m.insert(a, v);
            }
        };

        // ===== Plosives =====
        add(&["P", "p"], consonant(false, Plosive, Labial));
        add(&["B", "b"], consonant(true, Plosive, Labial));
        add(&["T", "t"], consonant(false, Plosive, Coronal));
        add(&["D", "d"], consonant(true, Plosive, Coronal));
        add(&["ʈ"], consonant(false, Plosive, Coronal)); // retroflex
        add(&["ɖ"], consonant(true, Plosive, Coronal));
        add(&["K", "k"], consonant(false, Plosive, Dorsal));
        add(&["G", "g", "ɡ"], consonant(true, Plosive, Dorsal));
        add(&["c"], consonant(false, Plosive, Dorsal)); // palatal
        add(&["ɟ"], consonant(true, Plosive, Dorsal));
        add(&["q"], consonant(false, Plosive, Dorsal)); // uvular
        add(&["ɢ"], consonant(true, Plosive, Dorsal));
        add(&["Q", "ʔ"], consonant(false, Plosive, Glottal)); // glottal stop

        // ===== Fricatives =====
        add(&["F", "f"], consonant(false, Fricative, Labial));
        add(&["V", "v"], consonant(true, Fricative, Labial));
        add(&["ɸ"], consonant(false, Fricative, Labial)); // bilabial
        add(&["β"], consonant(true, Fricative, Labial));
        add(&["TH", "θ"], consonant(false, Fricative, Coronal)); // dental
        add(&["DH", "ð"], consonant(true, Fricative, Coronal));
        add(&["S", "s"], consonant(false, Fricative, Coronal));
        add(&["Z", "z"], consonant(true, Fricative, Coronal));
        add(&["SH", "ʃ", "ʂ"], consonant(false, Fricative, Coronal)); // postalveolar/retroflex
        add(&["ZH", "ʒ", "ʐ"], consonant(true, Fricative, Coronal));
        add(&["ç"], consonant(false, Fricative, Dorsal)); // palatal
        add(&["ʝ"], consonant(true, Fricative, Dorsal));
        add(&["x"], consonant(false, Fricative, Dorsal)); // velar
        add(&["ɣ"], consonant(true, Fricative, Dorsal));
        add(&["χ"], consonant(false, Fricative, Dorsal)); // uvular
        add(&["ʁ"], consonant(true, Fricative, Dorsal));
        add(&["ħ"], consonant(false, Fricative, Glottal)); // pharyngeal
        add(&["ʕ"], consonant(true, Fricative, Glottal));
        add(&["HH", "h"], consonant(false, Fricative, Glottal)); // glottal
        add(&["ɦ"], consonant(true, Fricative, Glottal));

        // ===== Affricates =====
        add(&["CH", "tʃ", "ʧ"], consonant(false, Affricate, Coronal));
        add(&["JH", "dʒ", "ʤ"], consonant(true, Affricate, Coronal));
        add(&["ts", "ʦ"], consonant(false, Affricate, Coronal));
        add(&["dz", "ʣ"], consonant(true, Affricate, Coronal));

        // ===== Nasals =====
        add(&["M", "m"], consonant(true, Nasal, Labial));
        add(&["ɱ"], consonant(true, Nasal, Labial)); // labiodental
        add(&["N", "n"], consonant(true, Nasal, Coronal));
        add(&["ɳ"], consonant(true, Nasal, Coronal)); // retroflex
        add(&["ɲ"], consonant(true, Nasal, Dorsal)); // palatal
        add(&["NG", "ŋ"], consonant(true, Nasal, Dorsal)); // velar
        add(&["ɴ"], consonant(true, Nasal, Dorsal)); // uvular

        // ===== Liquids (laterals + rhotics) =====
        add(&["L", "l"], consonant(true, Lateral, Coronal));
        add(&["ɫ"], consonant(true, Lateral, Coronal)); // velarised ("dark") l
        add(&["ɭ"], consonant(true, Lateral, Coronal)); // retroflex
        add(&["ʎ"], consonant(true, Lateral, Dorsal)); // palatal
        add(&["R", "ɹ", "ɻ", "r", "ɾ"], consonant(true, Rhotic, Coronal));
        add(&["ʀ"], consonant(true, Rhotic, Dorsal)); // uvular trill

        // ===== Glides =====
        add(&["W", "w"], consonant(true, Glide, Labiovelar));
        add(&["ʍ"], consonant(false, Glide, Labiovelar)); // voiceless w
        add(&["ɥ"], consonant(true, Glide, Labiovelar)); // labial-palatal
        add(&["Y", "j"], consonant(true, Glide, Dorsal)); // palatal
        add(&["ɰ"], consonant(true, Glide, Dorsal)); // velar approximant

        // ===== Vowels (height, backness, rounded, tense, rhotic) =====
        // Front unrounded.
        add(&["IY", "i"], vowel(1.00, 0.00, false, true, false));
        add(&["IH", "ɪ"], vowel(0.85, 0.15, false, false, false));
        add(&["EY", "eɪ", "e"], vowel(0.65, 0.00, false, true, false));
        add(&["EH", "ɛ"], vowel(0.35, 0.00, false, false, false));
        add(&["AE", "æ"], vowel(0.20, 0.05, false, false, false));
        add(&["a"], vowel(0.00, 0.20, false, false, false)); // open front
                                                             // Front rounded.
        add(&["y"], vowel(1.00, 0.00, true, true, false));
        add(&["ø"], vowel(0.65, 0.00, true, true, false));
        add(&["œ"], vowel(0.35, 0.00, true, false, false));
        // Central.
        add(&["AH", "ʌ"], vowel(0.40, 0.55, false, false, false));
        add(&["AX", "ə"], vowel(0.50, 0.50, false, false, false)); // schwa
        add(&["ER", "ɝ", "ɜ", "ɚ"], vowel(0.50, 0.50, false, true, true)); // r-coloured
        add(&["ɐ"], vowel(0.20, 0.50, false, false, false));
        add(&["ɨ"], vowel(0.90, 0.50, false, true, false));
        add(&["ʉ"], vowel(0.90, 0.50, true, true, false));
        // Back.
        add(&["UW", "u"], vowel(1.00, 1.00, true, true, false));
        add(&["UH", "ʊ"], vowel(0.85, 0.80, true, false, false));
        add(&["OW", "oʊ", "o"], vowel(0.65, 1.00, true, true, false));
        add(&["AO", "ɔ"], vowel(0.35, 1.00, true, true, false));
        add(&["AA", "ɑ"], vowel(0.00, 1.00, false, true, false));
        add(&["ɒ"], vowel(0.00, 1.00, true, true, false));
        add(&["ɯ"], vowel(1.00, 1.00, false, true, false));
        // Diphthongs (approximated by an averaged nucleus).
        add(&["AY", "aɪ"], vowel(0.35, 0.20, false, true, false));
        add(&["AW", "aʊ"], vowel(0.35, 0.55, true, true, false));
        add(&["OY", "ɔɪ"], vowel(0.45, 0.55, true, true, false));
    }

    m
}

/// Strip stress / length / tie diacritics and ARPAbet stress digits so that
/// surface variants (`AA1`, `iː`, `t͡ʃ`) collapse onto their base symbol.
fn is_strippable(c: char) -> bool {
    c.is_ascii_digit()
        || matches!(
            c,
            '\u{02C8}' // ˈ primary stress
                | '\u{02CC}' // ˌ secondary stress
                | '\u{02D0}' // ː long
                | '\u{02D1}' // ˑ half-long
                | '\u{0361}' // ◌͡ tie bar (above)
                | '\u{035C}' // ◌͜ tie bar (below)
                | '\u{0329}' // ◌̩ syllabic
        )
}

/// Normalise a raw phoneme symbol prior to table lookup.
fn normalize_symbol(raw: &str) -> String {
    raw.trim().chars().filter(|c| !is_strippable(*c)).collect()
}

/// Return the all-zero neutral feature vector used for unknown symbols.
pub(super) fn neutral_features() -> [f32; FEATURE_DIM] {
    NEUTRAL
}

/// Look up the distinctive-feature vector for a single phoneme symbol.
///
/// Accepts both IPA and ARPAbet spellings (case-insensitively for ASCII
/// ARPAbet symbols) and tolerates stress / length / tie diacritics. Unknown
/// symbols return the documented [`neutral_features`] vector.
pub(super) fn phoneme_feature_vector(symbol: &str) -> [f32; FEATURE_DIM] {
    let norm = normalize_symbol(symbol);
    if norm.is_empty() {
        return NEUTRAL;
    }

    let table = feature_table();
    if let Some(v) = table.get(norm.as_str()) {
        return *v;
    }

    // ASCII case folding catches ARPAbet given in the "wrong" case (e.g. `aa`,
    // `hh`) and IPA given upper-cased; non-ASCII codepoints are left untouched.
    let upper = norm.to_ascii_uppercase();
    if upper != norm {
        if let Some(v) = table.get(upper.as_str()) {
            return *v;
        }
    }
    let lower = norm.to_ascii_lowercase();
    if lower != norm {
        if let Some(v) = table.get(lower.as_str()) {
            return *v;
        }
    }

    NEUTRAL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voiced_voiceless_pair_differs_only_in_voicing() {
        let p = phoneme_feature_vector("p");
        let b = phoneme_feature_vector("b");
        for i in 0..FEATURE_DIM {
            if i == F_VOICED {
                assert!(
                    (p[i] - b[i]).abs() > 0.5,
                    "the voicing dimension must distinguish /p/ and /b/"
                );
            } else {
                assert!(
                    (p[i] - b[i]).abs() < f32::EPSILON,
                    "dimension {i} must be identical for /p/ and /b/"
                );
            }
        }
        // ARPAbet upper-case spellings agree with the IPA vectors.
        assert_eq!(p, phoneme_feature_vector("P"));
        assert_eq!(b, phoneme_feature_vector("B"));
    }

    #[test]
    fn alveolar_pair_differs_only_in_voicing() {
        let t = phoneme_feature_vector("T");
        let d = phoneme_feature_vector("D");
        let mut differing = Vec::new();
        for i in 0..FEATURE_DIM {
            if (t[i] - d[i]).abs() > f32::EPSILON {
                differing.push(i);
            }
        }
        assert_eq!(differing, vec![F_VOICED]);
    }

    #[test]
    fn vowel_vs_stop_differ_in_consonantal_and_manner() {
        let vowel_aa = phoneme_feature_vector("AA"); // /ɑ/
        let stop_p = phoneme_feature_vector("P"); // /p/

        // Major class: vowel is [+syllabic, -consonantal], stop is the reverse.
        assert_eq!(vowel_aa[F_SYLLABIC], 1.0);
        assert_eq!(vowel_aa[F_CONSONANTAL], 0.0);
        assert_eq!(stop_p[F_SYLLABIC], 0.0);
        assert_eq!(stop_p[F_CONSONANTAL], 1.0);
        assert_ne!(vowel_aa[F_CONSONANTAL], stop_p[F_CONSONANTAL]);

        // Manner: the stop asserts the plosive manner; the vowel does not.
        assert_eq!(stop_p[F_PLOSIVE], 1.0);
        assert_eq!(vowel_aa[F_PLOSIVE], 0.0);
        assert_ne!(vowel_aa[F_PLOSIVE], stop_p[F_PLOSIVE]);
    }

    #[test]
    fn previously_zero_phonemes_now_have_features() {
        // None of these were present in the old 15-symbol placeholder table.
        for sym in [
            "f", "F", "v", "SH", "ʃ", "CH", "tʃ", "æ", "AE", "l", "L", "w", "W", "r", "R", "NG",
            "ŋ", "HH", "h", "IY", "i", "ER", "θ",
        ] {
            let v = phoneme_feature_vector(sym);
            let energy: f32 = v.iter().map(|x| x.abs()).sum();
            assert!(
                energy > 0.0,
                "symbol {sym:?} should map to a non-zero distinctive-feature vector"
            );
        }
    }

    #[test]
    fn unknown_symbols_return_neutral() {
        for sym in ["QZX", "💥", "", "   ", "#", "42"] {
            assert_eq!(
                phoneme_feature_vector(sym),
                neutral_features(),
                "symbol {sym:?} should fall back to the neutral vector"
            );
        }
    }

    #[test]
    fn neutral_vector_is_all_zero() {
        assert_eq!(neutral_features(), [0.0; FEATURE_DIM]);
    }

    #[test]
    fn arpabet_stress_digits_and_length_are_ignored() {
        assert_eq!(phoneme_feature_vector("AA1"), phoneme_feature_vector("AA"));
        assert_eq!(phoneme_feature_vector("AH0"), phoneme_feature_vector("AH"));
        assert_eq!(phoneme_feature_vector("ER2"), phoneme_feature_vector("ER"));
        assert_eq!(phoneme_feature_vector("iː"), phoneme_feature_vector("i"));
    }

    #[test]
    fn ipa_and_arpabet_spellings_agree() {
        assert_eq!(phoneme_feature_vector("ʃ"), phoneme_feature_vector("SH"));
        assert_eq!(phoneme_feature_vector("ŋ"), phoneme_feature_vector("NG"));
        assert_eq!(phoneme_feature_vector("ɑ"), phoneme_feature_vector("AA"));
        assert_eq!(phoneme_feature_vector("tʃ"), phoneme_feature_vector("CH"));
        assert_eq!(phoneme_feature_vector("dʒ"), phoneme_feature_vector("JH"));
    }

    #[test]
    fn nasal_is_voiced_occlusive_sonorant() {
        let m = phoneme_feature_vector("M");
        assert_eq!(m[F_NASAL], 1.0);
        assert_eq!(m[F_VOICED], 1.0);
        assert_eq!(m[F_SONORANT], 1.0);
        assert_eq!(m[F_CONSONANTAL], 1.0);
        assert_eq!(m[F_CONTINUANT], 0.0); // nasals are [-continuant]
        assert_eq!(m[F_PLACE_LABIAL], 1.0);
    }

    #[test]
    fn glide_is_non_consonantal_sonorant() {
        let w = phoneme_feature_vector("W");
        assert_eq!(w[F_CONSONANTAL], 0.0);
        assert_eq!(w[F_SONORANT], 1.0);
        assert_eq!(w[F_APPROXIMANT], 1.0);
        // /w/ is a labial-velar double articulation.
        assert_eq!(w[F_PLACE_LABIAL], 1.0);
        assert_eq!(w[F_PLACE_DORSAL], 1.0);
    }

    #[test]
    fn liquids_carry_lateral_and_rhotic_subfeatures() {
        let l = phoneme_feature_vector("L");
        let r = phoneme_feature_vector("R");
        assert_eq!(l[F_LATERAL], 1.0);
        assert_eq!(l[F_RHOTIC], 0.0);
        assert_eq!(r[F_RHOTIC], 1.0);
        assert_eq!(r[F_LATERAL], 0.0);
        assert_eq!(l[F_APPROXIMANT], 1.0);
        assert_eq!(r[F_APPROXIMANT], 1.0);
    }

    #[test]
    fn vowel_height_and_backness_are_graded() {
        let close_front = phoneme_feature_vector("IY"); // /i/
        let open_back = phoneme_feature_vector("AA"); // /ɑ/
        assert!(close_front[F_VOWEL_HEIGHT] > open_back[F_VOWEL_HEIGHT]);
        assert!(close_front[F_VOWEL_BACKNESS] < open_back[F_VOWEL_BACKNESS]);
        // Rounded back vowel asserts the roundness feature.
        assert_eq!(phoneme_feature_vector("UW")[F_VOWEL_ROUNDED], 1.0);
        assert_eq!(phoneme_feature_vector("IY")[F_VOWEL_ROUNDED], 0.0);
    }

    #[test]
    fn table_covers_a_broad_inventory() {
        // Sanity check that the matrix is substantially larger than the old
        // 15-symbol placeholder.
        assert!(
            feature_table().len() >= 90,
            "expected a broad IPA/ARPAbet inventory, got {}",
            feature_table().len()
        );
    }
}
