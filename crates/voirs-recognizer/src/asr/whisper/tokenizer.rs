//! Whisper tokenizer implementation with BPE encoding and multilingual support
//!
//! This module provides the tokenizer for Whisper models with support for
//! multiple languages, special tokens, and byte-pair encoding.

use crate::RecognitionError;
use std::collections::HashMap;
use voirs_sdk::LanguageCode;

/// Whisper tokenizer implementation
pub struct WhisperTokenizer {
    /// Vocabulary
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary
    reverse_vocab: HashMap<u32, String>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// BPE encoder
    bpe: BytePairEncoding,
}

/// Special tokens for Whisper
#[derive(Debug, Clone)]
/// Special Tokens
pub struct SpecialTokens {
    /// sot
    pub sot: u32,             // Start of transcript
    /// eot
    pub eot: u32,             // End of transcript
    /// sot prev
    pub sot_prev: u32,        // Start of previous segment
    /// no speech
    pub no_speech: u32,       // No speech
    /// no timestamps
    pub no_timestamps: u32,   // No timestamps
    /// timestamp begin
    pub timestamp_begin: u32, // Beginning of timestamp tokens
    /// language begin
    pub language_begin: u32,  // Beginning of language tokens
}

/// Byte-pair encoding implementation
pub struct BytePairEncoding {
    /// BPE merges
    merges: HashMap<(String, String), u32>,
    /// Vocabulary
    #[allow(dead_code)]
    vocab: HashMap<String, u32>,
}

impl WhisperTokenizer {
    /// new
    pub async fn new() -> Result<Self, RecognitionError> {
        // Create Whisper tokenizer with multilingual vocabulary
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Add basic ASCII characters and common tokens
        Self::initialize_base_vocab(&mut vocab, &mut reverse_vocab);

        let special_tokens = SpecialTokens {
            sot: 50258,             // <|startoftranscript|>
            eot: 50257,             // <|endoftext|>
            sot_prev: 50360,        // <|startofprev|>
            no_speech: 50362,       // <|nospeech|>
            no_timestamps: 50363,   // <|notimestamps|>
            timestamp_begin: 50364, // <|0.00|>
            language_begin: 50259,  // <|en|>
        };

        // Initialize BPE with basic merges
        let bpe = BytePairEncoding::new(&vocab)?;

        Ok(Self {
            vocab,
            reverse_vocab,
            special_tokens,
            bpe,
        })
    }

    fn initialize_base_vocab(
        vocab: &mut HashMap<String, u32>,
        reverse_vocab: &mut HashMap<u32, String>,
    ) {
        // Add ASCII characters (0-255) as byte tokens
        for i in 0..256 {
            let token = format!("Ġ{}", char::from(i as u8));
            vocab.insert(token.clone(), i as u32);
            reverse_vocab.insert(i as u32, token);
        }

        // Add common words and subwords (simplified vocabulary)
        let common_tokens = vec![
            ("the", 256),
            ("and", 257),
            ("to", 258),
            ("of", 259),
            ("a", 260),
            ("in", 261),
            ("is", 262),
            ("it", 263),
            ("you", 264),
            ("that", 265),
            ("he", 266),
            ("was", 267),
            ("for", 268),
            ("on", 269),
            ("are", 270),
            ("as", 271),
            ("with", 272),
            ("his", 273),
            ("they", 274),
            ("I", 275),
            ("at", 276),
            ("be", 277),
            ("this", 278),
            ("have", 279),
            ("from", 280),
            ("or", 281),
            ("one", 282),
            ("had", 283),
            ("by", 284),
            ("word", 285),
            ("but", 286),
            ("not", 287),
            ("what", 288),
            ("all", 289),
            ("were", 290),
            ("we", 291),
            ("when", 292),
            ("your", 293),
            ("can", 294),
            ("said", 295),
            ("there", 296),
            ("each", 297),
            ("which", 298),
            ("do", 299),
            ("how", 300),
        ];

        for (token, id) in common_tokens {
            vocab.insert(token.to_string(), id);
            reverse_vocab.insert(id, token.to_string());
        }

        // Add space token
        vocab.insert("Ġ".to_string(), 301);
        reverse_vocab.insert(301, "Ġ".to_string());

        // Add language tokens
        let languages = vec![
            ("en", 50259),
            ("zh", 50260),
            ("de", 50261),
            ("es", 50262),
            ("ru", 50263),
            ("ko", 50264),
            ("fr", 50265),
            ("ja", 50266),
            ("pt", 50267),
            ("tr", 50268),
            ("pl", 50269),
            ("ca", 50270),
            ("nl", 50271),
            ("ar", 50272),
            ("sv", 50273),
            ("it", 50274),
            ("id", 50275),
            ("hi", 50276),
            ("fi", 50277),
            ("vi", 50278),
        ];

        for (lang, id) in languages {
            let token = format!("<|{lang}|>");
            vocab.insert(token.clone(), id);
            reverse_vocab.insert(id, token);
        }

        // Add special tokens
        let special_tokens_list = vec![
            ("<|endoftext|>", 50257),
            ("<|startoftranscript|>", 50258),
            ("<|translate|>", 50359),
            ("<|transcribe|>", 50360),
            ("<|startoflm|>", 50361),
            ("<|startofprev|>", 50362),
            ("<|nospeech|>", 50363),
            ("<|notimestamps|>", 50364),
        ];

        for (token, id) in special_tokens_list {
            vocab.insert(token.to_string(), id);
            reverse_vocab.insert(id, token.to_string());
        }

        // Add timestamp tokens (0.00 to 30.00 seconds in 0.02s increments)
        for i in 0..=1500 {
            let time = i as f32 * 0.02;
            let token = format!("<|{time:.2}|>");
            let id = 50365 + i;
            vocab.insert(token.clone(), id);
            reverse_vocab.insert(id, token);
        }
    }

    /// decode
    pub fn decode(&self, tokens: &[u32]) -> Result<String, RecognitionError> {
        let mut text = String::new();

        for &token_id in tokens {
            // Skip special tokens in output
            if self.is_special_token(token_id) {
                continue;
            }

            if let Some(token_str) = self.reverse_vocab.get(&token_id) {
                // Handle space token (Ġ represents space in GPT tokenization)
                if token_str.starts_with("Ġ") {
                    if token_str.len() > 1 {
                        text.push(' ');
                        text.push_str(&token_str[1..]);
                    } else {
                        text.push(' ');
                    }
                } else {
                    text.push_str(token_str);
                }
            }
        }

        // Clean up extra spaces and normalize
        text = text.trim().to_string();
        text = text.replace("  ", " "); // Remove double spaces

        Ok(text)
    }

    /// encode
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, RecognitionError> {
        let mut tokens = Vec::new();

        // Add start of transcript token
        tokens.push(self.special_tokens.sot);

        // Add language token (default to English)
        tokens.push(50259); // <|en|>

        // Add transcribe task token
        tokens.push(50360); // <|transcribe|>

        // Add no timestamps token (for simplicity)
        tokens.push(self.special_tokens.no_timestamps);

        // Simple word-based tokenization (in practice, BPE would be used)
        for word in text.split_whitespace() {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                // Apply BPE encoding to unknown words
                let bpe_tokens = self.bpe.encode_word(word);
                for bpe_token in bpe_tokens {
                    if let Some(&token_id) = self.vocab.get(&bpe_token) {
                        tokens.push(token_id);
                    } else {
                        // Fallback: encode as character tokens
                        for ch in bpe_token.chars() {
                            if let Some(&token_id) = self.vocab.get(&ch.to_string()) {
                                tokens.push(token_id);
                            } else {
                                // Unknown character, use a default token
                                tokens.push(0); // UNK token
                            }
                        }
                    }
                }
            }

            // Add space token between words
            if let Some(&space_token) = self.vocab.get("Ġ") {
                tokens.push(space_token);
            }
        }

        // Add end of text token
        tokens.push(self.special_tokens.eot);

        Ok(tokens)
    }

    /// Encode text with specific language and task tokens
    pub fn encode_with_language(
        &self,
        text: &str,
        language: LanguageCode,
        task: WhisperTask,
    ) -> Result<Vec<u32>, RecognitionError> {
        let mut tokens = Vec::new();

        // Add start of transcript token
        tokens.push(self.special_tokens.sot);

        // Add language token
        tokens.push(self.language_token(language));

        // Add task token
        tokens.push(match task {
            WhisperTask::Transcribe => 50360, // <|transcribe|>
            WhisperTask::Translate => 50359,  // <|translate|>
        });

        // Add timestamps or no timestamps token
        tokens.push(self.special_tokens.no_timestamps);

        // Encode the actual text
        self.encode_text_only(text, &mut tokens)?;

        // Add end of text token
        tokens.push(self.special_tokens.eot);

        Ok(tokens)
    }

    /// Encode only the text content without special tokens
    fn encode_text_only(&self, text: &str, tokens: &mut Vec<u32>) -> Result<(), RecognitionError> {
        for word in text.split_whitespace() {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                // Apply BPE encoding to unknown words
                let bpe_tokens = self.bpe.encode_word(word);
                for bpe_token in bpe_tokens {
                    if let Some(&token_id) = self.vocab.get(&bpe_token) {
                        tokens.push(token_id);
                    } else {
                        // Fallback: encode as character tokens
                        for ch in bpe_token.chars() {
                            if let Some(&token_id) = self.vocab.get(&ch.to_string()) {
                                tokens.push(token_id);
                            } else {
                                // Unknown character, use a default token
                                tokens.push(0); // UNK token
                            }
                        }
                    }
                }
            }

            // Add space token between words
            if let Some(&space_token) = self.vocab.get("Ġ") {
                tokens.push(space_token);
            }
        }

        Ok(())
    }

    fn is_special_token(&self, token_id: u32) -> bool {
        token_id == self.special_tokens.sot
            || token_id == self.special_tokens.eot
            || token_id == self.special_tokens.sot_prev
            || token_id == self.special_tokens.no_speech
            || token_id == self.special_tokens.no_timestamps
            || (token_id >= self.special_tokens.timestamp_begin && token_id <= 51865)
            || (token_id >= self.special_tokens.language_begin && token_id <= 50278)
    }

    /// Get the vocabulary size
    #[must_use]
    /// vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token for a specific language
    #[must_use]
    /// language token
    pub fn language_token(&self, language: LanguageCode) -> u32 {
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => 50259, // <|en|>
            LanguageCode::ZhCn => 50260,                      // <|zh|>
            LanguageCode::DeDe => 50261,                      // <|de|>
            LanguageCode::EsEs => 50262,                      // <|es|>
            LanguageCode::JaJp => 50266,                      // <|ja|>
            LanguageCode::KoKr => 50264,                      // <|ko|>
            LanguageCode::FrFr => 50265,                      // <|fr|>
            _ => 50259,                                       // Default to English
        }
    }

    /// Detect language from tokens by finding language token
    #[must_use]
    /// detect language from tokens
    pub fn detect_language_from_tokens(&self, tokens: &[u32]) -> LanguageCode {
        for &token in tokens {
            match token {
                50259 => return LanguageCode::EnUs, // <|en|>
                50260 => return LanguageCode::ZhCn, // <|zh|>
                50261 => return LanguageCode::DeDe, // <|de|>
                50262 => return LanguageCode::EsEs, // <|es|>
                50264 => return LanguageCode::KoKr, // <|ko|>
                50265 => return LanguageCode::FrFr, // <|fr|>
                50266 => return LanguageCode::JaJp, // <|ja|>
                _ => {}
            }
        }
        LanguageCode::EnUs // Default to English if no language token found
    }

    /// Get special tokens
    #[must_use]
    /// special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Check if a token is a timestamp token
    #[must_use]
    /// is timestamp token
    pub fn is_timestamp_token(&self, token_id: u32) -> bool {
        token_id >= self.special_tokens.timestamp_begin && token_id <= 51865
    }

    /// Convert timestamp token to time in seconds
    #[must_use]
    /// timestamp to seconds
    pub fn timestamp_to_seconds(&self, token_id: u32) -> Option<f32> {
        if self.is_timestamp_token(token_id) {
            let offset = token_id - self.special_tokens.timestamp_begin;
            Some(offset as f32 * 0.02)
        } else {
            None
        }
    }

    /// Convert time in seconds to timestamp token
    #[must_use]
    /// seconds to timestamp token
    pub fn seconds_to_timestamp_token(&self, seconds: f32) -> u32 {
        let offset = (seconds / 0.02).round() as u32;
        self.special_tokens.timestamp_begin + offset.min(1500)
    }
}

/// Whisper task types
#[derive(Debug, Clone, Copy)]
/// Whisper Task
pub enum WhisperTask {
    /// Transcribe
    Transcribe,
    /// Translate
    Translate,
}

impl BytePairEncoding {
    fn new(vocab: &HashMap<String, u32>) -> Result<Self, RecognitionError> {
        // Initialize BPE with common merges
        let mut merges = HashMap::new();

        // Add common BPE merges (simplified)
        let common_merges = vec![
            (("t", "h"), 1000),  // "th"
            (("h", "e"), 1001),  // "he"
            (("i", "n"), 1002),  // "in"
            (("e", "r"), 1003),  // "er"
            (("a", "n"), 1004),  // "an"
            (("r", "e"), 1005),  // "re"
            (("n", "d"), 1006),  // "nd"
            (("o", "n"), 1007),  // "on"
            (("e", "n"), 1008),  // "en"
            (("a", "t"), 1009),  // "at"
            (("o", "u"), 1010),  // "ou"
            (("i", "t"), 1011),  // "it"
            (("a", "r"), 1012),  // "ar"
            (("s", "t"), 1013),  // "st"
            (("l", "l"), 1014),  // "ll"
            (("i", "s"), 1015),  // "is"
            (("o", "r"), 1016),  // "or"
            (("e", "d"), 1017),  // "ed"
            (("i", "ng"), 1018), // "ing"
            (("l", "y"), 1019),  // "ly"
        ];

        for ((first, second), rank) in common_merges {
            merges.insert((first.to_string(), second.to_string()), rank);
        }

        Ok(Self {
            merges,
            vocab: vocab.clone(),
        })
    }

    /// Apply BPE encoding to a word
    #[must_use]
    /// encode word
    pub fn encode_word(&self, word: &str) -> Vec<String> {
        if word.len() <= 1 {
            return vec![word.to_string()];
        }

        // Start with individual characters
        let mut word_tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best_merge = None;
            let mut best_rank = u32::MAX;
            let mut best_pos = 0;

            // Find the best merge to apply
            for i in 0..word_tokens.len() - 1 {
                let pair = (&word_tokens[i], &word_tokens[i + 1]);
                if let Some(&rank) = self.merges.get(&(pair.0.clone(), pair.1.clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_merge = Some((pair.0.clone(), pair.1.clone()));
                        best_pos = i;
                    }
                }
            }

            // If no merge found, break
            if best_merge.is_none() {
                break;
            }

            // Apply the best merge
            if let Some((first, second)) = best_merge {
                let merged = first + &second;
                word_tokens[best_pos] = merged;
                word_tokens.remove(best_pos + 1);
            }
        }

        word_tokens
    }

    /// Add a new merge rule
    pub fn add_merge(&mut self, first: String, second: String, rank: u32) {
        self.merges.insert((first, second), rank);
    }

    /// Get all merge rules
    #[must_use]
    /// get merges
    pub fn get_merges(&self) -> &HashMap<(String, String), u32> {
        &self.merges
    }
}
