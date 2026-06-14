//! Systematic Japanese grapheme-to-phoneme decomposition for singing lyrics.
//!
//! This module implements the real lyric-to-phoneme conversion used by
//! [`crate::singing`]. It replaces the former ~90-entry hard-coded romaji
//! lookup table with a complete, systematic mora model that decomposes every
//! Japanese mora into its constituent consonant and vowel phonemes.
//!
//! # Why a dedicated table instead of wiring `voirs-g2p`
//!
//! `voirs-acoustic` already depends on `voirs-g2p`, and approach (A) would have
//! been to call its Japanese backend ([`voirs_g2p::backends::JapaneseDictG2p`]).
//! That backend was inspected and rejected for the singing path for three
//! concrete reasons:
//!
//! 1. **Phoneme convention mismatch.** `JapaneseDictG2p` emits *mora-level IPA*
//!    symbols (`う`→`ɯ`, `し`→`ʃi`, `つ`→`tsɯ`, `ら`→`ɾa`, …). The singing
//!    synthesizer consumes a *decomposed ASCII* inventory where every mora is
//!    split into separate consonant + vowel tokens (`さ`→`"s a"`). Wiring the
//!    backend would change the phoneme set of every existing note and break the
//!    downstream singing voice model that expects this ASCII inventory.
//! 2. **Input format.** `MusicalNote::lyrics` carries *romaji* syllables
//!    (`"la"`, `"sa"`, `"ka"`). `JapaneseDictG2p` expects *kana*; romaji input
//!    falls through its character-level romanization and yields garbage.
//! 3. **Sync/async boundary.** The [`voirs_g2p::G2p`] trait is `async`, while
//!    `lyrics_to_phoneme` is a synchronous method that is itself reachable from
//!    inside a Tokio runtime. Bridging it with `block_on` risks the classic
//!    "cannot start a runtime from within a runtime" panic.
//!
//! Approach (B) is therefore the correct one: a systematic, complete mora table
//! covering the full gojūon, dakuten/handakuten, yōon (palatalised きゃ/しゃ/…),
//! sokuon (gemination っ) and long vowels (ー / repeated vowels), accepting both
//! kana (hiragana + katakana) and romaji input.
//!
//! # Phoneme inventory
//!
//! Vowels: `a i u e o`. Consonants: `k g s z t d n h b p f v m r w l`, the
//! affricate/fricative digraphs `sh ch ts` (for し/ち/つ rows), the palatal glide
//! `j` (for や/ゆ/よ and yōon), the moraic nasal `N` (ん) and the geminate
//! closure `q` (a sokuon with no following consonant). Long vowels are realised
//! by repeating the preceding vowel; a sokuon is realised by doubling the
//! following consonant.

/// Decompose a singing-lyric string into a flat sequence of phoneme tokens.
///
/// Accepts kana (hiragana/katakana, including yōon, sokuon and the long-vowel
/// mark `ー`) as well as romaji syllables. Unknown/out-of-vocabulary input is
/// handled gracefully: unparseable characters are skipped, and an input that
/// produces no phonemes at all falls back to a single `"a"` vowel so callers
/// never receive an empty symbol.
pub(crate) fn lyrics_to_phoneme_tokens(lyrics: &str) -> Vec<String> {
    let lowered = lyrics.to_lowercase();
    let hiragana = katakana_to_hiragana(&lowered);
    let chars: Vec<char> = hiragana.chars().collect();
    let len = chars.len();

    let mut tokens: Vec<String> = Vec::new();
    let mut pending_sokuon = false;
    let mut idx = 0;

    while idx < len {
        let current = chars[idx];

        // Long-vowel mark: prolong the most recently emitted vowel.
        if current == 'ー' {
            if pending_sokuon {
                tokens.push("q".to_string());
                pending_sokuon = false;
            }
            if let Some(vowel) = last_vowel(&tokens) {
                tokens.push(vowel);
            }
            idx += 1;
            continue;
        }

        // Sokuon (gemination): remember it and apply to the next mora.
        if current == 'っ' {
            pending_sokuon = true;
            idx += 1;
            continue;
        }

        // Moraic nasal.
        if current == 'ん' {
            emit(&mut tokens, &mut pending_sokuon, &["N"]);
            idx += 1;
            continue;
        }

        // Two-character kana yōon (palatalised mora, e.g. きゃ, しゃ).
        if idx + 1 < len && is_small_y(chars[idx + 1]) {
            let mora: String = chars[idx..idx + 2].iter().collect();
            if let Some(unit) = kana_tokens(&mora) {
                emit(&mut tokens, &mut pending_sokuon, unit);
                idx += 2;
                continue;
            }
        }

        // Single kana mora.
        if let Some(unit) = kana_tokens(&current.to_string()) {
            emit(&mut tokens, &mut pending_sokuon, unit);
            idx += 1;
            continue;
        }

        // Romaji handling.
        if current.is_ascii_alphabetic() {
            // Doubled consonant -> romaji gemination (sokuon), e.g. "katta".
            if is_geminable(current) && idx + 1 < len && chars[idx + 1] == current {
                pending_sokuon = true;
                idx += 1;
                continue;
            }

            // Longest-match romaji syllable (3, then 2, then 1 character).
            let max_len = 3.min(len - idx);
            let mut matched = false;
            for try_len in (1..=max_len).rev() {
                if chars[idx..idx + try_len]
                    .iter()
                    .all(|c| c.is_ascii_alphabetic())
                {
                    let candidate: String = chars[idx..idx + try_len].iter().collect();
                    if let Some(unit) = romaji_tokens(&candidate) {
                        emit(&mut tokens, &mut pending_sokuon, unit);
                        idx += try_len;
                        matched = true;
                        break;
                    }
                }
            }
            if matched {
                continue;
            }

            // A bare 'n' that did not pair with a vowel is the moraic nasal.
            if current == 'n' {
                emit(&mut tokens, &mut pending_sokuon, &["N"]);
                idx += 1;
                continue;
            }

            // Unparseable romaji letter: skip it gracefully.
            idx += 1;
            continue;
        }

        // Any other character (punctuation, whitespace, unknown script): skip.
        idx += 1;
    }

    // A trailing sokuon with nothing to geminate becomes a glottal closure.
    if pending_sokuon {
        tokens.push("q".to_string());
    }

    // Graceful fallback so callers never receive an empty phoneme symbol.
    if tokens.is_empty() {
        tokens.push("a".to_string());
    }

    tokens
}

/// Push a mora's phoneme tokens, applying any pending sokuon (gemination).
///
/// Gemination is realised by doubling the first consonant of the following
/// mora; if that mora is vowel-initial (or there is none) a glottal closure
/// `"q"` is emitted instead.
fn emit(tokens: &mut Vec<String>, pending_sokuon: &mut bool, unit: &[&str]) {
    if *pending_sokuon {
        match unit.first() {
            Some(first) if is_consonant_token(first) => tokens.push(first.to_string()),
            _ => tokens.push("q".to_string()),
        }
        *pending_sokuon = false;
    }
    tokens.extend(unit.iter().map(|t| t.to_string()));
}

/// Return the most recently emitted vowel token, if any (used for long vowels).
fn last_vowel(tokens: &[String]) -> Option<String> {
    tokens.iter().rev().find(|t| is_vowel_token(t)).cloned()
}

/// Convert any katakana in `s` to the equivalent hiragana.
///
/// The katakana block (U+30A1..=U+30F6) maps to hiragana by subtracting 0x60.
/// The half-width long-vowel mark is normalised to its full-width form.
fn katakana_to_hiragana(s: &str) -> String {
    s.chars()
        .map(|c| {
            let code = c as u32;
            if (0x30A1..=0x30F6).contains(&code) {
                char::from_u32(code - 0x60).unwrap_or(c)
            } else if c == 'ｰ' {
                'ー'
            } else {
                c
            }
        })
        .collect()
}

/// True for the small palatalising kana ゃ/ゅ/ょ that form yōon.
fn is_small_y(c: char) -> bool {
    matches!(c, 'ゃ' | 'ゅ' | 'ょ')
}

/// True for vowel phoneme tokens.
fn is_vowel_token(token: &str) -> bool {
    matches!(token, "a" | "i" | "u" | "e" | "o")
}

/// True for consonant phoneme tokens (anything that may be geminated).
fn is_consonant_token(token: &str) -> bool {
    !is_vowel_token(token) && token != "N" && token != "q"
}

/// True for romaji consonants that may be doubled to mark gemination.
///
/// `n` is deliberately excluded: a doubled `n` (e.g. "onna") is a moraic nasal
/// followed by an `n`-row mora, not a sokuon.
fn is_geminable(c: char) -> bool {
    matches!(
        c,
        'k' | 'g'
            | 's'
            | 'z'
            | 't'
            | 'd'
            | 'h'
            | 'b'
            | 'p'
            | 'f'
            | 'v'
            | 'm'
            | 'r'
            | 'l'
            | 'w'
            | 'c'
    )
}

/// Map a single hiragana mora (1 char) or yōon (2 chars) to phoneme tokens.
fn kana_tokens(mora: &str) -> Option<&'static [&'static str]> {
    let unit: &'static [&'static str] = match mora {
        // Bare vowels (including small standalone forms).
        "あ" | "ぁ" => &["a"],
        "い" | "ぃ" => &["i"],
        "う" | "ぅ" => &["u"],
        "え" | "ぇ" => &["e"],
        "お" | "ぉ" => &["o"],
        // K / G series.
        "か" => &["k", "a"],
        "き" => &["k", "i"],
        "く" => &["k", "u"],
        "け" => &["k", "e"],
        "こ" => &["k", "o"],
        "が" => &["g", "a"],
        "ぎ" => &["g", "i"],
        "ぐ" => &["g", "u"],
        "げ" => &["g", "e"],
        "ご" => &["g", "o"],
        // S / Z series (し/じ are palatal).
        "さ" => &["s", "a"],
        "し" => &["sh", "i"],
        "す" => &["s", "u"],
        "せ" => &["s", "e"],
        "そ" => &["s", "o"],
        "ざ" => &["z", "a"],
        "じ" => &["z", "i"],
        "ず" => &["z", "u"],
        "ぜ" => &["z", "e"],
        "ぞ" => &["z", "o"],
        // T / D series (ち/つ are affricates; ぢ/づ merge with じ/ず).
        "た" => &["t", "a"],
        "ち" => &["ch", "i"],
        "つ" => &["ts", "u"],
        "て" => &["t", "e"],
        "と" => &["t", "o"],
        "だ" => &["d", "a"],
        "ぢ" => &["z", "i"],
        "づ" => &["z", "u"],
        "で" => &["d", "e"],
        "ど" => &["d", "o"],
        // N series.
        "な" => &["n", "a"],
        "に" => &["n", "i"],
        "ぬ" => &["n", "u"],
        "ね" => &["n", "e"],
        "の" => &["n", "o"],
        // H / B / P series (ふ is a bilabial fricative).
        "は" => &["h", "a"],
        "ひ" => &["h", "i"],
        "ふ" => &["f", "u"],
        "へ" => &["h", "e"],
        "ほ" => &["h", "o"],
        "ば" => &["b", "a"],
        "び" => &["b", "i"],
        "ぶ" => &["b", "u"],
        "べ" => &["b", "e"],
        "ぼ" => &["b", "o"],
        "ぱ" => &["p", "a"],
        "ぴ" => &["p", "i"],
        "ぷ" => &["p", "u"],
        "ぺ" => &["p", "e"],
        "ぽ" => &["p", "o"],
        // M series.
        "ま" => &["m", "a"],
        "み" => &["m", "i"],
        "む" => &["m", "u"],
        "め" => &["m", "e"],
        "も" => &["m", "o"],
        // Y series (palatal glide).
        "や" => &["j", "a"],
        "ゆ" => &["j", "u"],
        "よ" => &["j", "o"],
        // R series.
        "ら" => &["r", "a"],
        "り" => &["r", "i"],
        "る" => &["r", "u"],
        "れ" => &["r", "e"],
        "ろ" => &["r", "o"],
        // W series and the (near-obsolete) ゐ/ゑ.
        "わ" => &["w", "a"],
        "ゐ" => &["w", "i"],
        "ゑ" => &["w", "e"],
        "を" => &["w", "o"],
        // Extended kana for loanwords.
        "ゔ" => &["v", "u"],
        // Yōon: palatalised morae. The し/ち rows are already palatal, so they
        // take no extra glide; the others insert the palatal glide `j`.
        "きゃ" => &["k", "j", "a"],
        "きゅ" => &["k", "j", "u"],
        "きょ" => &["k", "j", "o"],
        "ぎゃ" => &["g", "j", "a"],
        "ぎゅ" => &["g", "j", "u"],
        "ぎょ" => &["g", "j", "o"],
        "しゃ" => &["sh", "a"],
        "しゅ" => &["sh", "u"],
        "しょ" => &["sh", "o"],
        "じゃ" => &["z", "j", "a"],
        "じゅ" => &["z", "j", "u"],
        "じょ" => &["z", "j", "o"],
        "ちゃ" => &["ch", "a"],
        "ちゅ" => &["ch", "u"],
        "ちょ" => &["ch", "o"],
        "ぢゃ" => &["z", "j", "a"],
        "ぢゅ" => &["z", "j", "u"],
        "ぢょ" => &["z", "j", "o"],
        "にゃ" => &["n", "j", "a"],
        "にゅ" => &["n", "j", "u"],
        "にょ" => &["n", "j", "o"],
        "ひゃ" => &["h", "j", "a"],
        "ひゅ" => &["h", "j", "u"],
        "ひょ" => &["h", "j", "o"],
        "びゃ" => &["b", "j", "a"],
        "びゅ" => &["b", "j", "u"],
        "びょ" => &["b", "j", "o"],
        "ぴゃ" => &["p", "j", "a"],
        "ぴゅ" => &["p", "j", "u"],
        "ぴょ" => &["p", "j", "o"],
        "みゃ" => &["m", "j", "a"],
        "みゅ" => &["m", "j", "u"],
        "みょ" => &["m", "j", "o"],
        "りゃ" => &["r", "j", "a"],
        "りゅ" => &["r", "j", "u"],
        "りょ" => &["r", "j", "o"],
        _ => return None,
    };
    Some(unit)
}

/// Map a romaji syllable to phoneme tokens.
///
/// The single-letter-consonant entries preserve the exact mappings of the
/// former hard-coded table (so `"si"`, `"ti"`, `"tu"`, `"hu"`, `"di"`, … keep
/// their legacy values), while the Hepburn digraphs (`shi`, `chi`, `tsu`, `fu`,
/// `ji`) and yōon (`kya`, `sha`, …) are added for completeness.
fn romaji_tokens(syllable: &str) -> Option<&'static [&'static str]> {
    let unit: &'static [&'static str] = match syllable {
        // Bare vowels.
        "a" => &["a"],
        "i" => &["i"],
        "u" => &["u"],
        "e" => &["e"],
        "o" => &["o"],
        // K / G.
        "ka" => &["k", "a"],
        "ki" => &["k", "i"],
        "ku" => &["k", "u"],
        "ke" => &["k", "e"],
        "ko" => &["k", "o"],
        "ga" => &["g", "a"],
        "gi" => &["g", "i"],
        "gu" => &["g", "u"],
        "ge" => &["g", "e"],
        "go" => &["g", "o"],
        // S / Z (legacy single-letter forms + Hepburn shi/ji).
        "sa" => &["s", "a"],
        "si" => &["s", "i"],
        "su" => &["s", "u"],
        "se" => &["s", "e"],
        "so" => &["s", "o"],
        "shi" => &["sh", "i"],
        "za" => &["z", "a"],
        "zi" => &["z", "i"],
        "zu" => &["z", "u"],
        "ze" => &["z", "e"],
        "zo" => &["z", "o"],
        "ji" => &["z", "i"],
        // T / D (legacy + Hepburn chi/tsu).
        "ta" => &["t", "a"],
        "ti" => &["t", "i"],
        "tu" => &["t", "u"],
        "te" => &["t", "e"],
        "to" => &["t", "o"],
        "chi" => &["ch", "i"],
        "tsu" => &["ts", "u"],
        "da" => &["d", "a"],
        "di" => &["d", "i"],
        "du" => &["d", "u"],
        "de" => &["d", "e"],
        "do" => &["d", "o"],
        // N.
        "na" => &["n", "a"],
        "ni" => &["n", "i"],
        "nu" => &["n", "u"],
        "ne" => &["n", "e"],
        "no" => &["n", "o"],
        // H / B / P (legacy + Hepburn fu).
        "ha" => &["h", "a"],
        "hi" => &["h", "i"],
        "hu" => &["h", "u"],
        "he" => &["h", "e"],
        "ho" => &["h", "o"],
        "fu" => &["f", "u"],
        "ba" => &["b", "a"],
        "bi" => &["b", "i"],
        "bu" => &["b", "u"],
        "be" => &["b", "e"],
        "bo" => &["b", "o"],
        "pa" => &["p", "a"],
        "pi" => &["p", "i"],
        "pu" => &["p", "u"],
        "pe" => &["p", "e"],
        "po" => &["p", "o"],
        // F / V rows (loanwords; preserved from the legacy table).
        "fa" => &["f", "a"],
        "fi" => &["f", "i"],
        "fe" => &["f", "e"],
        "fo" => &["f", "o"],
        "va" => &["v", "a"],
        "vi" => &["v", "i"],
        "vu" => &["v", "u"],
        "ve" => &["v", "e"],
        "vo" => &["v", "o"],
        // M.
        "ma" => &["m", "a"],
        "mi" => &["m", "i"],
        "mu" => &["m", "u"],
        "me" => &["m", "e"],
        "mo" => &["m", "o"],
        // Y (palatal glide; legacy ye/yi preserved).
        "ya" => &["j", "a"],
        "yi" => &["j", "i"],
        "yu" => &["j", "u"],
        "ye" => &["j", "e"],
        "yo" => &["j", "o"],
        // R.
        "ra" => &["r", "a"],
        "ri" => &["r", "i"],
        "ru" => &["r", "u"],
        "re" => &["r", "e"],
        "ro" => &["r", "o"],
        // L row (preserved from the legacy table).
        "la" => &["l", "a"],
        "li" => &["l", "i"],
        "lu" => &["l", "u"],
        "le" => &["l", "e"],
        "lo" => &["l", "o"],
        // W row (legacy wu preserved).
        "wa" => &["w", "a"],
        "wi" => &["w", "i"],
        "wu" => &["w", "u"],
        "we" => &["w", "e"],
        "wo" => &["w", "o"],
        // Yōon (palatalised romaji).
        "kya" => &["k", "j", "a"],
        "kyu" => &["k", "j", "u"],
        "kyo" => &["k", "j", "o"],
        "gya" => &["g", "j", "a"],
        "gyu" => &["g", "j", "u"],
        "gyo" => &["g", "j", "o"],
        "sha" => &["sh", "a"],
        "shu" => &["sh", "u"],
        "sho" => &["sh", "o"],
        "ja" => &["z", "j", "a"],
        "ju" => &["z", "j", "u"],
        "jo" => &["z", "j", "o"],
        "cha" => &["ch", "a"],
        "chu" => &["ch", "u"],
        "cho" => &["ch", "o"],
        "nya" => &["n", "j", "a"],
        "nyu" => &["n", "j", "u"],
        "nyo" => &["n", "j", "o"],
        "hya" => &["h", "j", "a"],
        "hyu" => &["h", "j", "u"],
        "hyo" => &["h", "j", "o"],
        "bya" => &["b", "j", "a"],
        "byu" => &["b", "j", "u"],
        "byo" => &["b", "j", "o"],
        "pya" => &["p", "j", "a"],
        "pyu" => &["p", "j", "u"],
        "pyo" => &["p", "j", "o"],
        "mya" => &["m", "j", "a"],
        "myu" => &["m", "j", "u"],
        "myo" => &["m", "j", "o"],
        "rya" => &["r", "j", "a"],
        "ryu" => &["r", "j", "u"],
        "ryo" => &["r", "j", "o"],
        _ => return None,
    };
    Some(unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbol(lyrics: &str) -> String {
        lyrics_to_phoneme_tokens(lyrics).join(" ")
    }

    #[test]
    fn test_legacy_romaji_morae_preserved() {
        // The exact values from the former hard-coded table must still hold.
        assert_eq!(symbol("la"), "l a");
        assert_eq!(symbol("sa"), "s a");
        assert_eq!(symbol("si"), "s i");
        assert_eq!(symbol("tu"), "t u");
        assert_eq!(symbol("hu"), "h u");
        assert_eq!(symbol("ya"), "j a");
        assert_eq!(symbol("wo"), "w o");
        assert_eq!(symbol("vu"), "v u");
    }

    #[test]
    fn test_single_kana_morae() {
        assert_eq!(symbol("さ"), "s a");
        assert_eq!(symbol("し"), "sh i");
        assert_eq!(symbol("つ"), "ts u");
        assert_eq!(symbol("ふ"), "f u");
        assert_eq!(symbol("ら"), "r a");
        assert_eq!(symbol("を"), "w o");
    }

    #[test]
    fn test_katakana_normalised_to_hiragana() {
        // カ and か must decompose identically.
        assert_eq!(symbol("カ"), symbol("か"));
        assert_eq!(symbol("サクラ"), "s a k u r a");
        assert_eq!(symbol("ヴ"), "v u");
    }

    #[test]
    fn test_multi_mora_kana_lyric() {
        // The canonical example from the task: さくら -> s a k u r a.
        assert_eq!(symbol("さくら"), "s a k u r a");
    }

    #[test]
    fn test_multi_mora_romaji_lyric() {
        // Greedy longest-match must split romaji into morae correctly.
        assert_eq!(symbol("sakura"), "s a k u r a");
    }

    #[test]
    fn test_dakuten_and_handakuten() {
        assert_eq!(symbol("が"), "g a");
        assert_eq!(symbol("ぱ"), "p a");
        assert_eq!(symbol("ど"), "d o");
        assert_eq!(symbol("べ"), "b e");
    }

    #[test]
    fn test_yoon_palatalised() {
        assert_eq!(symbol("きゃ"), "k j a");
        assert_eq!(symbol("しゃ"), "sh a");
        assert_eq!(symbol("ちょ"), "ch o");
        assert_eq!(symbol("りゅ"), "r j u");
        assert_eq!(symbol("じゃ"), "z j a");
        // Romaji yōon must agree.
        assert_eq!(symbol("kya"), "k j a");
        assert_eq!(symbol("sho"), "sh o");
    }

    #[test]
    fn test_moraic_nasal() {
        assert_eq!(symbol("ほん"), "h o N");
        assert_eq!(symbol("にほん"), "n i h o N");
        // Romaji 'n' before a consonant / at end is the moraic nasal.
        assert_eq!(symbol("hon"), "h o N");
        // Doubled 'n' is moraic-nasal + n-row, not gemination.
        assert_eq!(symbol("onna"), "o N n a");
    }

    #[test]
    fn test_sokuon_gemination() {
        // っ doubles the following consonant.
        assert_eq!(symbol("がっこう"), "g a k k o u");
        assert_eq!(symbol("きって"), "k i t t e");
        // Romaji doubled consonants behave the same way.
        assert_eq!(symbol("katta"), "k a t t a");
        // A trailing sokuon with nothing after becomes a glottal closure.
        assert_eq!(symbol("あっ"), "a q");
    }

    #[test]
    fn test_long_vowels() {
        // The ー mark prolongs the previous vowel.
        assert_eq!(symbol("かー"), "k a a");
        assert_eq!(symbol("コーヒー"), "k o o h i i");
        // Repeated vowels are simply two vowel phonemes.
        assert_eq!(symbol("ねえ"), "n e e");
    }

    #[test]
    fn test_oov_degrades_gracefully() {
        // Fully unmappable input must not panic and must yield a fallback vowel.
        assert_eq!(symbol("###"), "a");
        assert_eq!(symbol(""), "a");
        assert_eq!(symbol("123"), "a");
        // Partially parseable garbage yields what it can, without panicking.
        assert!(!lyrics_to_phoneme_tokens("xyz123").is_empty());
    }
}
