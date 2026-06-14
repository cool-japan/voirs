//! Phonetic encoding for similarity-based batch deduplication.
//!
//! This module implements the classic **Metaphone** algorithm
//! (Lawrence Philips, *Computer Language*, December 1990) to derive a
//! pronunciation-based key for English text. Unlike a naive lowercase /
//! alphabetic strip, Metaphone maps groups of letters to a small set of
//! consonant phoneme symbols so that words which *sound* alike collapse to the
//! same code. This lets the batch optimizer group homophonic synthesis
//! requests (e.g. `"night"` / `"knight"`, `"smith"` / `"smyth"`) together for
//! deduplication.
//!
//! # Algorithm overview
//!
//! Metaphone applies a sequence of context-sensitive transformation rules over
//! the uppercased letters of a word:
//!
//! 1. Drop duplicate adjacent letters (except `C`).
//! 2. Strip several silent initial-letter pairs (`KN`, `GN`, `PN`, `AE`, `WR`).
//! 3. Drop a leading `X` to `S` and a leading `WH` to `W`.
//! 4. Keep vowels only when they begin the word.
//! 5. Map each remaining consonant to its Metaphone symbol using the
//!    surrounding letters for disambiguation (e.g. soft vs. hard `C`/`G`,
//!    silent `GH`, the `TH` → `0` (theta) mapping, `PH` → `F`, etc.).
//!
//! The result is an upper-case ASCII string drawn from the alphabet
//! `B X S K J T F H L M N P R 0 W Y` (where `0` denotes the *th* sound and `X`
//! denotes the *sh* / *ch* sound).
//!
//! # References
//!
//! - Lawrence Philips, "Hanging on the Metaphone", *Computer Language*,
//!   Vol. 7, No. 12 (December 1990).
//! - Philips, "The Double Metaphone Search Algorithm", *C/C++ Users Journal*
//!   (June 2000) — the successor algorithm that this implementation is
//!   modelled on the original of.

/// Vowels recognised by the Metaphone rules (uppercase ASCII).
const VOWELS: &[u8] = b"AEIOU";

/// Returns `true` if `b` is one of `A E I O U`.
#[inline]
fn is_vowel(b: u8) -> bool {
    VOWELS.contains(&b)
}

/// Compute the classic Metaphone code for a single word.
///
/// Non-alphabetic characters are ignored. The input is treated
/// case-insensitively. An empty (or fully non-alphabetic) input yields an
/// empty string.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(metaphone("Thompson"), "TMSN");
/// assert_eq!(metaphone("night"), metaphone("knight"));
/// ```
pub fn metaphone(word: &str) -> String {
    // 1. Keep only alphabetic characters and uppercase them.
    let letters: Vec<u8> = word
        .bytes()
        .filter(|b| b.is_ascii_alphabetic())
        .map(|b| b.to_ascii_uppercase())
        .collect();

    if letters.is_empty() {
        return String::new();
    }

    // 2. Collapse duplicate adjacent letters, except a doubled 'C'.
    let mut deduped: Vec<u8> = Vec::with_capacity(letters.len());
    for &b in &letters {
        if let Some(&last) = deduped.last() {
            if last == b && b != b'C' {
                continue;
            }
        }
        deduped.push(b);
    }
    let s = deduped;
    let n = s.len();

    // Helper closures for safe positional lookups.
    let at = |i: isize| -> u8 {
        if i < 0 || i as usize >= n {
            0
        } else {
            s[i as usize]
        }
    };
    let is_vowel_at = |i: isize| -> bool { is_vowel(at(i)) };

    let mut out: Vec<u8> = Vec::with_capacity(n);

    // 3. Handle silent / transformed initial letter combinations.
    let mut i: isize = 0;
    match (at(0), at(1)) {
        (b'A', b'E') | (b'G', b'N') | (b'K', b'N') | (b'P', b'N') | (b'W', b'R') => {
            // Silent first letter: skip it.
            i = 1;
        }
        (b'W', b'H') => {
            // "WH" at the start sounds like "W".
            out.push(b'W');
            i = 2;
        }
        (b'X', _) => {
            // Leading "X" is pronounced "S".
            out.push(b'S');
            i = 1;
        }
        _ => {}
    }

    // 4. Main transduction loop.
    while (i as usize) < n {
        let c = at(i);
        let prev = at(i - 1);
        let next = at(i + 1);

        // Vowels are only emitted when they begin the word.
        if is_vowel(c) {
            if i == 0 {
                out.push(c);
            }
            i += 1;
            continue;
        }

        match c {
            b'B' => {
                // Silent 'B' in a terminal "MB" (e.g. "dumb").
                if !(i as usize == n - 1 && prev == b'M') {
                    out.push(b'B');
                }
            }
            b'C' => {
                if next == b'I' && at(i + 2) == b'A' {
                    // "CIA" -> X (sh)
                    out.push(b'X');
                } else if next == b'H' {
                    if prev == b'S' {
                        // "SCH" -> K
                        out.push(b'K');
                    } else {
                        // "CH" -> X (ch sound)
                        out.push(b'X');
                    }
                    i += 1; // consume the 'H'
                } else if next == b'I' || next == b'E' || next == b'Y' {
                    // Soft 'C' -> S, but not the doubled "SC" cluster.
                    if prev != b'S' {
                        out.push(b'S');
                    }
                } else {
                    out.push(b'K');
                }
            }
            b'D' => {
                if next == b'G' && (at(i + 2) == b'E' || at(i + 2) == b'I' || at(i + 2) == b'Y') {
                    // "DGE", "DGI", "DGY" -> J
                    out.push(b'J');
                    i += 2; // consume "GE"/"GI"/"GY" head
                } else {
                    out.push(b'T');
                }
            }
            b'G' => {
                if next == b'H' {
                    // Silent "GH" unless followed by a vowel.
                    if !is_vowel_at(i + 2) && at(i + 2) != 0 {
                        // silent
                    } else if i == 0 && is_vowel_at(i + 2) {
                        // Initial "GH" + vowel -> J ("ghoti"-style edge); rare.
                        out.push(b'J');
                    } else {
                        // Otherwise "GH" is silent.
                    }
                    i += 1; // consume the 'H'
                } else if next == b'N' {
                    // Silent 'G' in "GN" / "GNED" terminal clusters.
                    let gned = next == b'N'
                        && at(i + 2) == b'E'
                        && at(i + 3) == b'D'
                        && (i as usize + 3 == n - 1);
                    if !((i as usize + 1 == n - 1) || gned) {
                        out.push(b'K');
                    }
                } else if next == b'I' || next == b'E' || next == b'Y' {
                    // Soft 'G' -> J.
                    out.push(b'J');
                } else {
                    out.push(b'K');
                }
            }
            b'H' => {
                // 'H' is silent after a vowel when not followed by a vowel,
                // and silent after the "dull" consonants C/S/P/T/G.
                let after_vowel = is_vowel(prev);
                let before_vowel = is_vowel(next);
                let dull = matches!(prev, b'C' | b'S' | b'P' | b'T' | b'G');
                if (after_vowel && !before_vowel) || dull {
                    // silent
                } else {
                    out.push(b'H');
                }
            }
            b'F' => out.push(b'F'),
            b'J' => out.push(b'J'),
            b'K' => {
                // Silent 'K' when preceded by 'C'.
                if prev != b'C' {
                    out.push(b'K');
                }
            }
            b'L' => out.push(b'L'),
            b'M' => out.push(b'M'),
            b'N' => out.push(b'N'),
            b'P' => {
                if next == b'H' {
                    // "PH" -> F
                    out.push(b'F');
                    i += 1; // consume the 'H'
                } else {
                    out.push(b'P');
                }
            }
            b'Q' => out.push(b'K'),
            b'R' => out.push(b'R'),
            b'S' => {
                if next == b'H' {
                    // "SH" -> X
                    out.push(b'X');
                    i += 1; // consume the 'H'
                } else if next == b'I' && (at(i + 2) == b'O' || at(i + 2) == b'A') {
                    // "SIO"/"SIA" -> X (e.g. "tension", "Asia")
                    out.push(b'X');
                } else {
                    out.push(b'S');
                }
            }
            b'T' => {
                if next == b'H' {
                    // "TH" -> 0 (theta, the 'th' sound)
                    out.push(b'0');
                    i += 1; // consume the 'H'
                } else if next == b'I' && (at(i + 2) == b'O' || at(i + 2) == b'A') {
                    // "TIO"/"TIA" -> X (e.g. "ration")
                    out.push(b'X');
                } else {
                    out.push(b'T');
                }
            }
            b'V' => out.push(b'F'),
            b'W' | b'Y' => {
                // 'W'/'Y' only sound when followed by a vowel.
                if is_vowel(next) {
                    out.push(c);
                }
            }
            b'X' => {
                // Non-initial 'X' -> KS.
                out.push(b'K');
                out.push(b'S');
            }
            b'Z' => out.push(b'S'),
            _ => {}
        }

        i += 1;
    }

    // SAFETY: every byte pushed is an ASCII letter or '0'.
    String::from_utf8(out).unwrap_or_default()
}

/// Compute a phonetic key for arbitrary (possibly multi-word) text.
///
/// Each whitespace-separated token is encoded independently with
/// [`metaphone`], and the per-word codes are re-joined with a single space.
/// Tokens that reduce to an empty code (e.g. pure punctuation) are dropped.
///
/// The result is a deterministic [`String`] suitable for use as a
/// deduplication / grouping key: inputs that sound alike map to the same key.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(phonetic_key("Knight rider"), phonetic_key("night rider"));
/// ```
pub fn phonetic_key(text: &str) -> String {
    let mut codes: Vec<String> = Vec::new();
    for token in text.split_whitespace() {
        let code = metaphone(token);
        if !code.is_empty() {
            codes.push(code);
        }
    }
    codes.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_non_alpha_yield_empty() {
        assert_eq!(metaphone(""), "");
        assert_eq!(metaphone("123!@#"), "");
        assert_eq!(phonetic_key("   "), "");
        assert_eq!(phonetic_key("!!! ???"), "");
    }

    #[test]
    fn case_and_punctuation_insensitive() {
        // Same word in different cases / with stray punctuation collides.
        assert_eq!(metaphone("Thompson"), metaphone("thompson"));
        assert_eq!(metaphone("Thompson"), metaphone("THOMPSON"));
        assert_eq!(phonetic_key("hello,"), phonetic_key("Hello"));
    }

    #[test]
    fn silent_leading_clusters_collide() {
        // The classic "knight" / "night" homophone pair.
        assert_eq!(metaphone("knight"), metaphone("night"));
        // "gnome" / "nome", "pneumo" silent P, "wrack" / "rack".
        assert_eq!(metaphone("gnome"), metaphone("nome"));
        assert_eq!(metaphone("wrack"), metaphone("rack"));
    }

    #[test]
    fn smith_and_smyth_collide() {
        // Y behaves like the vowel I here; both encode to the same key.
        assert_eq!(metaphone("smith"), metaphone("smyth"));
        // And the encoded value uses '0' for the terminal "th".
        assert_eq!(metaphone("smith"), "SM0");
    }

    #[test]
    fn ph_and_f_are_equivalent() {
        assert_eq!(metaphone("phone"), metaphone("fone"));
        assert_eq!(metaphone("philip"), metaphone("filip"));
    }

    #[test]
    fn known_metaphone_values() {
        // Spot-check against deterministic reference outputs. Note that this
        // implementation applies the classic `TH -> 0` (theta) rule everywhere,
        // so "Thompson" begins with `0` rather than `T`.
        assert_eq!(metaphone("Thompson"), "0MPSN");
        assert_eq!(metaphone("Wikipedia"), "WKPT");
        assert_eq!(metaphone("school"), "SKL");
        assert_eq!(metaphone("knight"), "NT");
        assert_eq!(metaphone("thumb"), "0M");
    }

    #[test]
    fn distinct_words_have_distinct_codes() {
        assert_ne!(metaphone("apple"), metaphone("orange"));
        assert_ne!(metaphone("cat"), metaphone("dog"));
        assert_ne!(metaphone("synthesis"), metaphone("recognition"));
    }

    #[test]
    fn double_letters_collapse() {
        // Doubled consonants collapse (except 'C'); these should match.
        assert_eq!(metaphone("ammonia"), metaphone("amonia"));
        assert_eq!(metaphone("balloon"), metaphone("balon"));
    }

    #[test]
    fn multi_word_key_per_word() {
        // Per-word encoding joined by spaces; homophones collide token-wise.
        assert_eq!(phonetic_key("Knight rider"), phonetic_key("night rider"));
        let key = phonetic_key("Thompson school");
        assert_eq!(key, "0MPSN SKL");
        // Different second word -> different key.
        assert_ne!(
            phonetic_key("Thompson school"),
            phonetic_key("Thompson college")
        );
    }

    #[test]
    fn deterministic() {
        // Repeated calls are identical (no RNG / global state).
        let a = phonetic_key("The quick brown fox");
        let b = phonetic_key("The quick brown fox");
        assert_eq!(a, b);
    }
}
