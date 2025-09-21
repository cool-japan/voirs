//! Part-of-speech tagging functionality for context-aware preprocessing.

use crate::{LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Part-of-speech tags for linguistic analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PosTag {
    /// Noun (person, place, thing)
    Noun,
    /// Verb (action, state)
    Verb,
    /// Adjective (descriptor)
    Adjective,
    /// Adverb (modifies verbs, adjectives)
    Adverb,
    /// Pronoun (substitute for noun)
    Pronoun,
    /// Preposition (relationship)
    Preposition,
    /// Conjunction (connects words)
    Conjunction,
    /// Determiner (article, quantifier)
    Determiner,
    /// Interjection (exclamation)
    Interjection,
    /// Proper noun (names)
    ProperNoun,
    /// Number (cardinal, ordinal)
    Number,
    /// Punctuation
    Punctuation,
    /// Unknown/other
    Other,
}

/// Trait for part-of-speech tagging functionality
pub trait PosTagging: Send + Sync {
    /// Tag entire text and return word-tag pairs
    fn tag_text(&self, text: &str) -> Result<Vec<(String, PosTag)>>;

    /// Tag a single word with context
    fn tag_word(&self, word: &str, context: &[String]) -> Result<PosTag>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageCode>;
}

/// Rule-based POS tagger implementation
#[derive(Debug, Clone)]
pub struct RuleBasedPosTagger {
    /// Language-specific rules
    pub rules: HashMap<LanguageCode, PosTaggingRules>,
    /// Word-to-POS mappings
    pub word_mappings: HashMap<String, PosTag>,
    /// Contextual rules
    pub contextual_rules: Vec<ContextualPosRule>,
}

/// POS tagging rules for a language
#[derive(Debug, Clone)]
pub struct PosTaggingRules {
    /// Suffix-based rules
    pub suffix_rules: HashMap<String, PosTag>,
    /// Prefix-based rules
    pub prefix_rules: HashMap<String, PosTag>,
    /// Pattern-based rules
    pub pattern_rules: Vec<(regex::Regex, PosTag)>,
    /// Common words dictionary
    pub common_words: HashMap<String, PosTag>,
}

/// Contextual POS tagging rule
#[derive(Debug, Clone)]
pub struct ContextualPosRule {
    /// Target word pattern
    pub target_pattern: String,
    /// Preceding context pattern
    pub preceding_pattern: Option<String>,
    /// Following context pattern
    pub following_pattern: Option<String>,
    /// Assigned POS tag
    pub pos_tag: PosTag,
    /// Rule confidence
    pub confidence: f32,
}

impl PosTagging for RuleBasedPosTagger {
    fn tag_text(&self, text: &str) -> Result<Vec<(String, PosTag)>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tagged_words = Vec::new();

        for word in words {
            let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());
            if clean_word.is_empty() {
                continue;
            }

            // Check word mappings first
            if let Some(pos_tag) = self.word_mappings.get(clean_word) {
                tagged_words.push((clean_word.to_string(), pos_tag.clone()));
                continue;
            }

            // Apply rule-based tagging
            let pos_tag = self.apply_rules(clean_word)?;
            tagged_words.push((clean_word.to_string(), pos_tag));
        }

        Ok(tagged_words)
    }

    fn tag_word(&self, word: &str, _context: &[String]) -> Result<PosTag> {
        self.apply_rules(word)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.rules.keys().copied().collect()
    }
}

impl RuleBasedPosTagger {
    /// Create a new rule-based POS tagger with English defaults
    pub fn new() -> Self {
        let mut tagger = Self {
            rules: HashMap::new(),
            word_mappings: HashMap::new(),
            contextual_rules: Vec::new(),
        };

        // Initialize with English rules
        tagger
            .rules
            .insert(LanguageCode::EnUs, Self::create_english_rules());
        tagger.word_mappings = Self::create_english_word_mappings();
        tagger.contextual_rules = Self::create_english_contextual_rules();

        tagger
    }

    /// Create English-specific POS tagging rules
    fn create_english_rules() -> PosTaggingRules {
        let mut suffix_rules = HashMap::new();
        let mut prefix_rules = HashMap::new();
        let mut pattern_rules = Vec::new();
        let common_words = Self::create_english_common_words();

        // Suffix rules for verbs
        suffix_rules.insert("ing".to_string(), PosTag::Verb);
        suffix_rules.insert("ed".to_string(), PosTag::Verb);
        suffix_rules.insert("en".to_string(), PosTag::Verb);
        suffix_rules.insert("ate".to_string(), PosTag::Verb);
        suffix_rules.insert("ize".to_string(), PosTag::Verb);
        suffix_rules.insert("ify".to_string(), PosTag::Verb);

        // Suffix rules for adjectives
        suffix_rules.insert("ful".to_string(), PosTag::Adjective);
        suffix_rules.insert("less".to_string(), PosTag::Adjective);
        suffix_rules.insert("ous".to_string(), PosTag::Adjective);
        suffix_rules.insert("ive".to_string(), PosTag::Adjective);
        suffix_rules.insert("able".to_string(), PosTag::Adjective);
        suffix_rules.insert("ible".to_string(), PosTag::Adjective);
        suffix_rules.insert("er".to_string(), PosTag::Adjective);
        suffix_rules.insert("est".to_string(), PosTag::Adjective);

        // Suffix rules for adverbs
        suffix_rules.insert("ly".to_string(), PosTag::Adverb);
        suffix_rules.insert("ward".to_string(), PosTag::Adverb);
        suffix_rules.insert("wise".to_string(), PosTag::Adverb);

        // Suffix rules for nouns
        suffix_rules.insert("tion".to_string(), PosTag::Noun);
        suffix_rules.insert("sion".to_string(), PosTag::Noun);
        suffix_rules.insert("ment".to_string(), PosTag::Noun);
        suffix_rules.insert("ness".to_string(), PosTag::Noun);
        suffix_rules.insert("ity".to_string(), PosTag::Noun);
        suffix_rules.insert("ism".to_string(), PosTag::Noun);
        suffix_rules.insert("ist".to_string(), PosTag::Noun);
        suffix_rules.insert("ship".to_string(), PosTag::Noun);
        suffix_rules.insert("hood".to_string(), PosTag::Noun);

        // Prefix rules
        prefix_rules.insert("un".to_string(), PosTag::Adjective);
        prefix_rules.insert("re".to_string(), PosTag::Verb);
        prefix_rules.insert("pre".to_string(), PosTag::Adjective);
        prefix_rules.insert("dis".to_string(), PosTag::Verb);

        // Pattern rules
        if let Ok(number_regex) = regex::Regex::new(r"^\d+$") {
            pattern_rules.push((number_regex, PosTag::Number));
        }
        if let Ok(decimal_regex) = regex::Regex::new(r"^\d+\.\d+$") {
            pattern_rules.push((decimal_regex, PosTag::Number));
        }
        if let Ok(ordinal_regex) = regex::Regex::new(r"^\d+(?:st|nd|rd|th)$") {
            pattern_rules.push((ordinal_regex, PosTag::Number));
        }
        if let Ok(punctuation_regex) = regex::Regex::new(r"^[^\w\s]+$") {
            pattern_rules.push((punctuation_regex, PosTag::Punctuation));
        }

        PosTaggingRules {
            suffix_rules,
            prefix_rules,
            pattern_rules,
            common_words,
        }
    }

    /// Create English common words dictionary
    fn create_english_common_words() -> HashMap<String, PosTag> {
        let mut words = HashMap::new();

        // Articles and determiners
        let determiners = vec![
            "the", "a", "an", "this", "that", "these", "those", "some", "any", "all", "each",
            "every", "no", "few", "many", "much", "several",
        ];
        for word in determiners {
            words.insert(word.to_string(), PosTag::Determiner);
        }

        // Pronouns
        let pronouns = vec![
            "I",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "hers",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "theirs",
            "myself",
            "yourself",
            "himself",
            "herself",
            "itself",
            "ourselves",
            "yourselves",
            "themselves",
        ];
        for word in pronouns {
            words.insert(word.to_lowercase(), PosTag::Pronoun);
        }

        // Prepositions
        let prepositions = vec![
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "without",
            "to",
            "from",
            "of",
            "about",
            "under",
            "over",
            "through",
            "between",
            "among",
            "during",
            "before",
            "after",
            "above",
            "below",
            "beside",
            "behind",
            "beyond",
            "within",
            "throughout",
            "across",
            "along",
            "around",
            "against",
            "toward",
            "towards",
            "into",
            "onto",
            "upon",
            "off",
            "out",
            "up",
            "down",
        ];
        for word in prepositions {
            words.insert(word.to_string(), PosTag::Preposition);
        }

        // Conjunctions
        let conjunctions = vec![
            "and", "or", "but", "so", "yet", "nor", "for", "because", "since", "while", "although",
            "though", "unless", "until", "if", "when", "where", "why", "how", "that", "which",
            "who", "whom", "whose", "what",
        ];
        for word in conjunctions {
            words.insert(word.to_string(), PosTag::Conjunction);
        }

        // Common verbs
        let verbs = vec![
            "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "done", "will", "would", "could", "should", "may", "might", "must",
            "can", "shall", "get", "got", "go", "went", "gone", "come", "came", "see", "saw",
            "seen", "know", "knew", "known", "think", "thought", "take", "took", "taken", "give",
            "gave", "given", "find", "found", "tell", "told", "ask", "asked", "work", "worked",
            "seem", "seemed", "feel", "felt", "try", "tried", "leave", "left", "call", "called",
        ];
        for word in verbs {
            words.insert(word.to_string(), PosTag::Verb);
        }

        // Common adjectives
        let adjectives = vec![
            "good",
            "new",
            "first",
            "last",
            "long",
            "great",
            "little",
            "own",
            "other",
            "old",
            "right",
            "big",
            "high",
            "different",
            "small",
            "large",
            "next",
            "early",
            "young",
            "important",
            "few",
            "public",
            "bad",
            "same",
            "able",
            "free",
            "sure",
            "clear",
            "white",
            "red",
            "blue",
            "green",
            "black",
            "true",
            "real",
            "best",
            "better",
            "worse",
            "worst",
            "hot",
            "cold",
            "warm",
            "cool",
            "fast",
            "slow",
            "quick",
            "easy",
            "hard",
            "simple",
            "complex",
        ];
        for word in adjectives {
            words.insert(word.to_string(), PosTag::Adjective);
        }

        // Common adverbs
        let adverbs = vec![
            "very",
            "well",
            "just",
            "now",
            "here",
            "there",
            "then",
            "still",
            "also",
            "only",
            "even",
            "really",
            "never",
            "always",
            "often",
            "sometimes",
            "usually",
            "quite",
            "rather",
            "pretty",
            "too",
            "so",
            "more",
            "most",
            "less",
            "least",
            "much",
            "many",
            "little",
            "few",
            "enough",
            "almost",
            "nearly",
            "hardly",
            "barely",
            "probably",
            "perhaps",
            "maybe",
            "certainly",
            "definitely",
            "absolutely",
            "completely",
            "totally",
            "fully",
            "quickly",
            "slowly",
            "carefully",
            "easily",
            "clearly",
            "simply",
            "directly",
            "recently",
            "finally",
            "already",
            "yet",
            "soon",
            "again",
            "once",
            "twice",
            "today",
            "yesterday",
            "tomorrow",
        ];
        for word in adverbs {
            words.insert(word.to_string(), PosTag::Adverb);
        }

        // Interjections
        let interjections = vec![
            "oh", "ah", "hey", "hi", "hello", "goodbye", "bye", "thanks", "please", "sorry",
            "excuse", "pardon", "wow", "oops", "ouch", "hooray", "yes", "no", "okay", "ok",
        ];
        for word in interjections {
            words.insert(word.to_string(), PosTag::Interjection);
        }

        words
    }

    /// Create English word mappings
    fn create_english_word_mappings() -> HashMap<String, PosTag> {
        Self::create_english_common_words()
    }

    /// Create English contextual rules
    fn create_english_contextual_rules() -> Vec<ContextualPosRule> {
        vec![
            // Verb after modal auxiliary
            ContextualPosRule {
                target_pattern: r"\w+".to_string(),
                preceding_pattern: Some(r"\b(?:can|could|will|would|shall|should|may|might|must)\s+".to_string()),
                following_pattern: None,
                pos_tag: PosTag::Verb,
                confidence: 0.9,
            },
            // Noun after determiner
            ContextualPosRule {
                target_pattern: r"\w+".to_string(),
                preceding_pattern: Some(r"\b(?:the|a|an|this|that|these|those|some|any|all|each|every|no|few|many|much|several)\s+".to_string()),
                following_pattern: None,
                pos_tag: PosTag::Noun,
                confidence: 0.8,
            },
            // Adjective before noun
            ContextualPosRule {
                target_pattern: r"\w+".to_string(),
                preceding_pattern: Some(r"\b(?:the|a|an|this|that|these|those|some|any|all|each|every|no|few|many|much|several)\s+".to_string()),
                following_pattern: Some(r"\s+\w+".to_string()),
                pos_tag: PosTag::Adjective,
                confidence: 0.7,
            },
            // Verb in past tense
            ContextualPosRule {
                target_pattern: r"\w+ed\b".to_string(),
                preceding_pattern: Some(r"\b(?:I|you|he|she|it|we|they|[A-Z][a-z]*)\s+".to_string()),
                following_pattern: None,
                pos_tag: PosTag::Verb,
                confidence: 0.85,
            },
        ]
    }

    /// Apply POS tagging rules to a word
    fn apply_rules(&self, word: &str) -> Result<PosTag> {
        let word_lower = word.to_lowercase();

        // Check word mappings first (highest priority)
        if let Some(pos_tag) = self.word_mappings.get(&word_lower) {
            return Ok(pos_tag.clone());
        }

        // Get English rules (default language)
        let rules = self
            .rules
            .get(&LanguageCode::EnUs)
            .ok_or_else(|| crate::G2pError::ConfigError("No English rules found".to_string()))?;

        // Check pattern rules
        for (pattern, pos_tag) in &rules.pattern_rules {
            if pattern.is_match(word) {
                return Ok(pos_tag.clone());
            }
        }

        // Check for capitalization (potential proper noun)
        if word.chars().next().is_some_and(|c| c.is_uppercase())
            && !self.word_mappings.contains_key(&word_lower)
        {
            return Ok(PosTag::ProperNoun);
        }

        // Check suffix rules (longer suffixes first)
        let mut suffix_matches: Vec<(&String, &PosTag)> = rules
            .suffix_rules
            .iter()
            .filter(|(suffix, _)| word_lower.ends_with(suffix.as_str()))
            .collect();
        suffix_matches.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        if let Some((_, pos_tag)) = suffix_matches.first() {
            return Ok((*pos_tag).clone());
        }

        // Check prefix rules
        for (prefix, pos_tag) in &rules.prefix_rules {
            if word_lower.starts_with(prefix) {
                return Ok(pos_tag.clone());
            }
        }

        // Check common words
        if let Some(pos_tag) = rules.common_words.get(&word_lower) {
            return Ok(pos_tag.clone());
        }

        // Default heuristics
        self.apply_default_heuristics(word)
    }

    /// Apply default heuristics when no rules match
    fn apply_default_heuristics(&self, word: &str) -> Result<PosTag> {
        let word_lower = word.to_lowercase();

        // Single character is likely punctuation
        if word.len() == 1 && word.chars().all(|c| c.is_ascii_punctuation()) {
            return Ok(PosTag::Punctuation);
        }

        // All uppercase words might be acronyms (proper nouns)
        if word.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()) && word.len() > 1 {
            return Ok(PosTag::ProperNoun);
        }

        // Words ending in 's might be plural nouns or possessive
        if word_lower.ends_with('s') && word_lower.len() > 2 {
            return Ok(PosTag::Noun);
        }

        // Words with hyphens are often adjectives
        if word.contains('-') {
            return Ok(PosTag::Adjective);
        }

        // Default to noun (most common POS)
        Ok(PosTag::Noun)
    }

    /// Enhanced tagging with contextual analysis
    pub fn tag_text_with_context(&self, text: &str) -> Result<Vec<(String, PosTag, f32)>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tagged_words = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());
            if clean_word.is_empty() {
                continue;
            }

            // Get context
            let preceding_context = if i > 0 { Some(words[i - 1]) } else { None };
            let following_context = if i + 1 < words.len() {
                Some(words[i + 1])
            } else {
                None
            };

            // Apply contextual rules
            let (pos_tag, confidence) =
                self.apply_contextual_rules(clean_word, preceding_context, following_context)?;

            tagged_words.push((clean_word.to_string(), pos_tag, confidence));
        }

        Ok(tagged_words)
    }

    /// Apply contextual rules to determine POS tag with confidence
    fn apply_contextual_rules(
        &self,
        word: &str,
        preceding: Option<&str>,
        following: Option<&str>,
    ) -> Result<(PosTag, f32)> {
        let word_lower = word.to_lowercase();

        // Check if word exists in mappings (high confidence)
        if let Some(pos_tag) = self.word_mappings.get(&word_lower) {
            return Ok((pos_tag.clone(), 0.95));
        }

        // Apply contextual rules
        let context_before = preceding.map(|w| w.to_lowercase()).unwrap_or_default();
        let context_after = following.map(|w| w.to_lowercase()).unwrap_or_default();

        for rule in &self.contextual_rules {
            let mut matches = true;

            // Check target pattern
            if let Ok(target_regex) = regex::Regex::new(&rule.target_pattern) {
                if !target_regex.is_match(word) {
                    matches = false;
                }
            }

            // Check preceding context
            if matches {
                if let Some(ref preceding_pattern) = rule.preceding_pattern {
                    if let Ok(preceding_regex) = regex::Regex::new(preceding_pattern) {
                        if !preceding_regex.is_match(&context_before) {
                            matches = false;
                        }
                    }
                }
            }

            // Check following context
            if matches {
                if let Some(ref following_pattern) = rule.following_pattern {
                    if let Ok(following_regex) = regex::Regex::new(following_pattern) {
                        if !following_regex.is_match(&context_after) {
                            matches = false;
                        }
                    }
                }
            }

            if matches {
                return Ok((rule.pos_tag.clone(), rule.confidence));
            }
        }

        // Fallback to rule-based tagging
        let pos_tag = self.apply_rules(word)?;
        Ok((pos_tag, 0.6)) // Medium confidence for rule-based tagging
    }

    /// Create language-specific tagger
    pub fn new_for_language(language: LanguageCode) -> Self {
        let mut tagger = Self::new(); // Start with English defaults

        match language {
            LanguageCode::EnUs => {
                // Already initialized with English
            }
            LanguageCode::Es => {
                tagger.add_spanish_rules();
            }
            LanguageCode::Fr => {
                tagger.add_french_rules();
            }
            LanguageCode::De => {
                tagger.add_german_rules();
            }
            _ => {
                // Use English defaults
            }
        }

        tagger
    }

    /// Add Spanish-specific rules
    fn add_spanish_rules(&mut self) {
        // Spanish suffix rules
        let mut spanish_suffixes = HashMap::new();
        spanish_suffixes.insert("ando".to_string(), PosTag::Verb); // gerund
        spanish_suffixes.insert("iendo".to_string(), PosTag::Verb); // gerund
        spanish_suffixes.insert("ado".to_string(), PosTag::Verb); // past participle
        spanish_suffixes.insert("ido".to_string(), PosTag::Verb); // past participle
        spanish_suffixes.insert("mente".to_string(), PosTag::Adverb); // adverb suffix
        spanish_suffixes.insert("ción".to_string(), PosTag::Noun); // noun suffix
        spanish_suffixes.insert("sión".to_string(), PosTag::Noun); // noun suffix

        let spanish_rules = PosTaggingRules {
            suffix_rules: spanish_suffixes,
            prefix_rules: HashMap::new(),
            pattern_rules: Vec::new(),
            common_words: HashMap::new(),
        };

        self.rules.insert(LanguageCode::Es, spanish_rules);
    }

    /// Add French-specific rules
    fn add_french_rules(&mut self) {
        // French suffix rules
        let mut french_suffixes = HashMap::new();
        french_suffixes.insert("ment".to_string(), PosTag::Adverb); // adverb suffix
        french_suffixes.insert("tion".to_string(), PosTag::Noun); // noun suffix
        french_suffixes.insert("ique".to_string(), PosTag::Adjective); // adjective suffix
        french_suffixes.insert("able".to_string(), PosTag::Adjective); // adjective suffix

        let french_rules = PosTaggingRules {
            suffix_rules: french_suffixes,
            prefix_rules: HashMap::new(),
            pattern_rules: Vec::new(),
            common_words: HashMap::new(),
        };

        self.rules.insert(LanguageCode::Fr, french_rules);
    }

    /// Add German-specific rules
    fn add_german_rules(&mut self) {
        // German suffix rules
        let mut german_suffixes = HashMap::new();
        german_suffixes.insert("lich".to_string(), PosTag::Adjective); // adjective suffix
        german_suffixes.insert("heit".to_string(), PosTag::Noun); // noun suffix
        german_suffixes.insert("keit".to_string(), PosTag::Noun); // noun suffix
        german_suffixes.insert("ung".to_string(), PosTag::Noun); // noun suffix

        let german_rules = PosTaggingRules {
            suffix_rules: german_suffixes,
            prefix_rules: HashMap::new(),
            pattern_rules: Vec::new(),
            common_words: HashMap::new(),
        };

        self.rules.insert(LanguageCode::De, german_rules);
    }

    /// Get confidence score for a POS tag assignment
    pub fn get_tag_confidence(&self, word: &str, pos_tag: &PosTag) -> f32 {
        let word_lower = word.to_lowercase();

        // High confidence for exact word matches
        if let Some(mapped_tag) = self.word_mappings.get(&word_lower) {
            return if mapped_tag == pos_tag { 0.95 } else { 0.0 };
        }

        // Medium confidence for rule matches
        if let Ok(rules) = self.rules.get(&LanguageCode::EnUs).ok_or("No rules") {
            // Check suffix rules
            for (suffix, suffix_tag) in &rules.suffix_rules {
                if word_lower.ends_with(suffix) && suffix_tag == pos_tag {
                    return 0.8;
                }
            }

            // Check pattern rules
            for (pattern, pattern_tag) in &rules.pattern_rules {
                if pattern.is_match(word) && pattern_tag == pos_tag {
                    return 0.85;
                }
            }
        }

        // Lower confidence for default assignments
        0.6
    }
}

impl Default for RuleBasedPosTagger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_tagging_basic() {
        let tagger = RuleBasedPosTagger::new();

        // Test basic rules
        assert_eq!(tagger.apply_rules("running").unwrap(), PosTag::Verb);
        assert_eq!(tagger.apply_rules("quickly").unwrap(), PosTag::Adverb);
        assert_eq!(tagger.apply_rules("walked").unwrap(), PosTag::Verb);
        assert_eq!(tagger.apply_rules("better").unwrap(), PosTag::Adjective);
        assert_eq!(tagger.apply_rules("123").unwrap(), PosTag::Number);
        assert_eq!(tagger.apply_rules("!").unwrap(), PosTag::Punctuation);
        assert_eq!(tagger.apply_rules("John").unwrap(), PosTag::ProperNoun);
        assert_eq!(tagger.apply_rules("house").unwrap(), PosTag::Noun);
    }

    #[test]
    fn test_tag_text() {
        let tagger = RuleBasedPosTagger::new();
        let result = tagger.tag_text("John is running quickly").unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], ("John".to_string(), PosTag::ProperNoun));
        assert_eq!(result[1], ("is".to_string(), PosTag::Verb));
        assert_eq!(result[2], ("running".to_string(), PosTag::Verb));
        assert_eq!(result[3], ("quickly".to_string(), PosTag::Adverb));
    }
}
