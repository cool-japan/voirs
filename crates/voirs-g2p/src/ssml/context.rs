//! Context-sensitive pronunciation analysis and processing.

use crate::phonology::{get_features, PhonologicalFeature};
use crate::ssml::dictionary::{PartOfSpeech, PronunciationContext};
use crate::{LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Context analyzer for determining pronunciation context
pub struct ContextAnalyzer {
    /// Language-specific rules
    _language: LanguageCode,
    /// Context rules
    rules: Vec<ContextRule>,
    /// POS tagger cache
    pos_cache: HashMap<String, PartOfSpeech>,
    /// N-gram patterns for context detection
    patterns: HashMap<String, ContextPattern>,
}

/// Context rule for determining pronunciation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRule {
    /// Rule name
    pub name: String,
    /// Rule priority (higher = more important)
    pub priority: u32,
    /// Conditions for applying this rule
    pub conditions: Vec<ContextCondition>,
    /// Resulting context
    pub context: PronunciationContext,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
}

/// Context condition for rule matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextCondition {
    /// Word position in sentence
    Position(PositionCondition),
    /// Preceding words
    PrecedingWords(Vec<String>),
    /// Following words
    FollowingWords(Vec<String>),
    /// Part of speech pattern
    PosPattern(Vec<PartOfSpeech>),
    /// Phonetic environment
    PhoneticEnvironment(PhoneticCondition),
    /// Syntactic structure
    SyntacticStructure(SyntacticCondition),
    /// Semantic context
    SemanticContext(SemanticCondition),
    /// Prosodic context
    ProsodicContext(ProsodicCondition),
    /// Custom condition
    Custom(String, String), // (name, value)
}

/// Position condition in sentence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionCondition {
    SentenceInitial,
    SentenceFinal,
    PhraseInitial,
    PhraseFinal,
    WordInitial,
    WordFinal,
    /// Specific position (0-based index)
    Position(usize),
    /// Relative position (0.0-1.0)
    RelativePosition(f32),
}

/// Phonetic environment condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneticCondition {
    /// Preceding phonemes
    pub preceding: Option<Vec<String>>,
    /// Following phonemes
    pub following: Option<Vec<String>>,
    /// Syllable structure
    pub syllable_structure: Option<SyllableStructure>,
    /// Stress pattern
    pub stress_pattern: Option<StressPattern>,
}

/// Syntactic structure condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyntacticCondition {
    /// Head of phrase
    PhraseHead(PhraseType),
    /// Modifier in phrase
    Modifier(PhraseType),
    /// Complement
    Complement,
    /// Adjunct
    Adjunct,
    /// Coordinate structure
    Coordination,
    /// Subordinate clause
    Subordination,
}

/// Semantic context condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticCondition {
    /// Semantic field
    SemanticField(String),
    /// Named entity type
    NamedEntity(EntityType),
    /// Domain-specific vocabulary
    Domain(String),
    /// Register (formal/informal)
    Register(RegisterLevel),
    /// Emotional content
    Emotion(EmotionType),
}

/// Prosodic context condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicCondition {
    /// Stress level
    pub stress_level: Option<StressLevel>,
    /// Prominence
    pub prominence: Option<ProminenceLevel>,
    /// Boundary strength
    pub boundary: Option<BoundaryStrength>,
    /// Rhythm pattern
    pub rhythm: Option<RhythmPattern>,
}

/// Context pattern for N-gram matching
#[derive(Debug, Clone)]
pub struct ContextPattern {
    /// Pattern tokens
    pub tokens: Vec<PatternToken>,
    /// Target context
    pub context: PronunciationContext,
    /// Pattern confidence
    pub confidence: f32,
    /// Usage frequency
    pub frequency: f32,
}

/// Pattern token for flexible matching
#[derive(Debug, Clone)]
pub enum PatternToken {
    /// Exact word match
    Word(String),
    /// Part of speech match
    Pos(PartOfSpeech),
    /// Phonetic feature match
    PhoneticFeature(String),
    /// Wildcard (any token)
    Wildcard,
    /// Optional token
    Optional(Box<PatternToken>),
    /// Alternative tokens
    Alternative(Vec<PatternToken>),
}

/// Supporting enums and structs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhraseType {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Prepositional,
    Determiner,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Place,
    Organization,
    Date,
    Time,
    Money,
    Percent,
    Misc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegisterLevel {
    VeryFormal,
    Formal,
    Neutral,
    Informal,
    VeryInformal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionType {
    Positive,
    Negative,
    Neutral,
    Excited,
    Calm,
    Angry,
    Sad,
    Happy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressLevel {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProminenceLevel {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryStrength {
    Strong,
    Medium,
    Weak,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RhythmPattern {
    Iambic,
    Trochaic,
    Dactylic,
    Anapestic,
    Irregular,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllableStructure {
    /// Onset complexity
    pub onset: OnsetComplexity,
    /// Nucleus type
    pub nucleus: NucleusType,
    /// Coda complexity
    pub coda: CodaComplexity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnsetComplexity {
    None,
    Simple,
    Complex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NucleusType {
    Monophthong,
    Diphthong,
    Triphthong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodaComplexity {
    None,
    Simple,
    Complex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressPattern {
    /// Primary stress on syllable
    Primary(usize),
    /// Secondary stress on syllable
    Secondary(usize),
    /// Unstressed
    Unstressed,
    /// Complex pattern
    Pattern(Vec<StressLevel>),
}

/// Context analysis result
#[derive(Debug, Clone)]
pub struct ContextAnalysisResult {
    /// Detected contexts with confidence scores
    pub contexts: Vec<(PronunciationContext, f32)>,
    /// Primary context (highest confidence)
    pub primary_context: Option<PronunciationContext>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Analysis metadata
#[derive(Debug, Clone)]
pub struct AnalysisMetadata {
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Rules applied
    pub rules_applied: Vec<String>,
    /// Patterns matched
    pub patterns_matched: Vec<String>,
    /// Confidence level of analysis
    pub overall_confidence: f32,
}

impl ContextAnalyzer {
    /// Create a new context analyzer
    pub fn new(language: LanguageCode) -> Self {
        let mut analyzer = Self {
            _language: language,
            rules: Vec::new(),
            pos_cache: HashMap::new(),
            patterns: HashMap::new(),
        };

        analyzer.load_default_rules();
        analyzer.load_default_patterns();
        analyzer
    }

    /// Analyze context for a word in a sentence
    pub fn analyze_context(
        &mut self,
        word: &str,
        sentence: &[String],
        word_index: usize,
    ) -> Result<ContextAnalysisResult> {
        let start_time = std::time::Instant::now();
        let mut contexts = Vec::new();
        let mut rules_applied = Vec::new();
        let mut patterns_matched = Vec::new();

        // Apply context rules
        let rules = self.rules.clone(); // Clone to avoid borrowing issues
        for rule in &rules {
            if self.evaluate_rule(rule, word, sentence, word_index)? {
                contexts.push((rule.context.clone(), rule.confidence));
                rules_applied.push(rule.name.clone());
            }
        }

        // Apply pattern matching
        for (pattern_name, pattern) in &self.patterns {
            if self.match_pattern(pattern, sentence, word_index) {
                contexts.push((pattern.context.clone(), pattern.confidence));
                patterns_matched.push(pattern_name.clone());
            }
        }

        // Sort by confidence and determine primary context
        contexts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let primary_context = contexts.first().map(|(ctx, _)| ctx.clone());

        let processing_time = start_time.elapsed().as_millis() as f32;
        let overall_confidence = contexts.first().map(|(_, conf)| *conf).unwrap_or(0.0);

        Ok(ContextAnalysisResult {
            contexts,
            primary_context,
            metadata: AnalysisMetadata {
                processing_time_ms: processing_time,
                rules_applied,
                patterns_matched,
                overall_confidence,
            },
        })
    }

    /// Evaluate a context rule
    fn evaluate_rule(
        &self,
        rule: &ContextRule,
        word: &str,
        sentence: &[String],
        word_index: usize,
    ) -> Result<bool> {
        for condition in &rule.conditions {
            if !self.evaluate_condition(condition, word, sentence, word_index)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Evaluate a single condition
    fn evaluate_condition(
        &self,
        condition: &ContextCondition,
        word: &str,
        sentence: &[String],
        word_index: usize,
    ) -> Result<bool> {
        match condition {
            ContextCondition::Position(pos_cond) => {
                self.evaluate_position_condition(pos_cond, sentence, word_index)
            }
            ContextCondition::PrecedingWords(words) => {
                Ok(self.check_preceding_words(words, sentence, word_index))
            }
            ContextCondition::FollowingWords(words) => {
                Ok(self.check_following_words(words, sentence, word_index))
            }
            ContextCondition::PosPattern(pattern) => {
                self.evaluate_pos_pattern(pattern, sentence, word_index)
            }
            ContextCondition::PhoneticEnvironment(phon_cond) => {
                self.evaluate_phonetic_condition(phon_cond, word, sentence, word_index)
            }
            ContextCondition::SyntacticStructure(_) => {
                // Simplified syntactic analysis - would need a proper parser
                Ok(true)
            }
            ContextCondition::SemanticContext(_) => {
                // Simplified semantic analysis - would need semantic models
                Ok(true)
            }
            ContextCondition::ProsodicContext(_) => {
                // Simplified prosodic analysis - would need prosodic models
                Ok(true)
            }
            ContextCondition::Custom(_, _) => {
                // Custom conditions would be user-defined
                Ok(true)
            }
        }
    }

    /// Evaluate position condition
    fn evaluate_position_condition(
        &self,
        condition: &PositionCondition,
        sentence: &[String],
        word_index: usize,
    ) -> Result<bool> {
        match condition {
            PositionCondition::SentenceInitial => Ok(word_index == 0),
            PositionCondition::SentenceFinal => Ok(word_index == sentence.len() - 1),
            PositionCondition::Position(pos) => Ok(word_index == *pos),
            PositionCondition::RelativePosition(rel_pos) => {
                let relative = word_index as f32 / sentence.len() as f32;
                Ok((relative - rel_pos).abs() < 0.1) // 10% tolerance
            }
            _ => Ok(true), // Simplified for other position types
        }
    }

    /// Check preceding words
    fn check_preceding_words(
        &self,
        words: &[String],
        sentence: &[String],
        word_index: usize,
    ) -> bool {
        if word_index < words.len() {
            return false;
        }

        for (i, word) in words.iter().enumerate() {
            let check_index = word_index - words.len() + i;
            if sentence.get(check_index).map(|w| w.to_lowercase()) != Some(word.to_lowercase()) {
                return false;
            }
        }
        true
    }

    /// Check following words
    fn check_following_words(
        &self,
        words: &[String],
        sentence: &[String],
        word_index: usize,
    ) -> bool {
        if word_index + words.len() >= sentence.len() {
            return false;
        }

        for (i, word) in words.iter().enumerate() {
            let check_index = word_index + 1 + i;
            if sentence.get(check_index).map(|w| w.to_lowercase()) != Some(word.to_lowercase()) {
                return false;
            }
        }
        true
    }

    /// Evaluate POS pattern
    fn evaluate_pos_pattern(
        &self,
        pattern: &[PartOfSpeech],
        sentence: &[String],
        word_index: usize,
    ) -> Result<bool> {
        // Simplified POS tagging - in practice would use a proper POS tagger
        for (i, expected_pos) in pattern.iter().enumerate() {
            let check_index = word_index + i;
            if let Some(word) = sentence.get(check_index) {
                let pos = self.simple_pos_tag(word);
                if pos != *expected_pos {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Simple POS tagging (would be replaced with proper tagger)
    fn simple_pos_tag(&self, word: &str) -> PartOfSpeech {
        // Simple rule-based POS tagging (caching removed for simplicity)
        let lower_word = word.to_lowercase();
        match lower_word.as_str() {
            // Articles and determiners
            "the" | "a" | "an" | "this" | "that" | "these" | "those" => PartOfSpeech::Determiner,
            // Common pronouns
            "i" | "you" | "he" | "she" | "it" | "we" | "they" | "me" | "him" | "her" | "us"
            | "them" => PartOfSpeech::Pronoun,
            // Common verbs (simplified)
            "is" | "are" | "was" | "were" | "be" | "been" | "have" | "has" | "had" | "do"
            | "does" | "did" => PartOfSpeech::Verb,
            // Common prepositions
            "in" | "on" | "at" | "by" | "for" | "with" | "to" | "from" | "of" | "about" => {
                PartOfSpeech::Preposition
            }
            // Common conjunctions
            "and" | "or" | "but" | "so" | "yet" | "nor" => PartOfSpeech::Conjunction,
            // Default based on suffixes
            _ => {
                if lower_word.ends_with("ly") {
                    PartOfSpeech::Adverb
                } else if lower_word.ends_with("ing") || lower_word.ends_with("ed") {
                    PartOfSpeech::Verb
                } else if lower_word.ends_with("ful")
                    || lower_word.ends_with("less")
                    || lower_word.ends_with("ous")
                {
                    PartOfSpeech::Adjective
                } else {
                    PartOfSpeech::Noun // Default to noun
                }
            }
        }
    }

    /// Evaluate a phonetic environment condition.
    ///
    /// The condition's [`PhoneticCondition::preceding`] / [`PhoneticCondition::following`]
    /// fields hold the phoneme sequences that must surround the target word. Because
    /// the analyzer is fed raw orthographic tokens (not a phonemic transcription) and
    /// no full G2P backend is wired into it, neighbouring words are approximated to
    /// IPA phonemes via a per-grapheme English letter→IPA map ([`grapheme_to_ipa`]).
    /// The actual comparison is fully feature-based: each expected phoneme is matched
    /// against the corresponding contextual phoneme using the crate's IPA distinctive
    /// feature table ([`crate::phonology::get_features`]) over voicing, place, manner
    /// and the vowel/consonant distinction (see [`phoneme_features_match`]).
    ///
    /// The `preceding` sequence is aligned so that its final element sits immediately
    /// before the word; `following` is aligned so its first element sits immediately
    /// after it. If there is not enough contextual material to fill the requested
    /// sequence, the condition cannot hold and `false` is returned.
    ///
    /// Limitation: [`PhoneticCondition::syllable_structure`] and
    /// [`PhoneticCondition::stress_pattern`] require a syllabifier / lexical-stress
    /// model that is not available to the context analyzer, so they are not evaluated
    /// here (treated as non-constraining) rather than fabricated.
    fn evaluate_phonetic_condition(
        &self,
        condition: &PhoneticCondition,
        _word: &str,
        sentence: &[String],
        word_index: usize,
    ) -> Result<bool> {
        // Preceding phonetic environment: the phonemes ending just before the word.
        if let Some(expected) = &condition.preceding {
            if !expected.is_empty() {
                let actual = collect_preceding_ipa(sentence, word_index, expected.len());
                if actual.len() < expected.len() {
                    // Not enough preceding context to satisfy the requested sequence.
                    return Ok(false);
                }
                for (exp, act) in expected.iter().zip(actual.iter()) {
                    if !phoneme_features_match(exp, act) {
                        return Ok(false);
                    }
                }
            }
        }

        // Following phonetic environment: the phonemes starting just after the word.
        if let Some(expected) = &condition.following {
            if !expected.is_empty() {
                let actual = collect_following_ipa(sentence, word_index, expected.len());
                if actual.len() < expected.len() {
                    return Ok(false);
                }
                for (exp, act) in expected.iter().zip(actual.iter()) {
                    if !phoneme_features_match(exp, act) {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    /// Match a context pattern against the sentence around `word_index`.
    ///
    /// The pattern window is centred on the target word (mirroring the original
    /// heuristic) and matched greedily with a moving cursor so that
    /// [`PatternToken::Optional`] tokens may be skipped: an optional token consumes
    /// the word at the cursor only when its inner pattern matches, otherwise the
    /// token is treated as absent and the cursor stays put. Required tokens must
    /// have a matching word present at the cursor, so a required token that is
    /// missing or mismatched fails the whole pattern.
    fn match_pattern(
        &self,
        pattern: &ContextPattern,
        sentence: &[String],
        word_index: usize,
    ) -> bool {
        if pattern.tokens.is_empty() {
            return false;
        }

        let start_index = word_index.saturating_sub(pattern.tokens.len() / 2);
        let mut cursor = start_index;

        for token in &pattern.tokens {
            match token {
                PatternToken::Optional(inner) => {
                    // Optional: consume the current word only if the inner pattern
                    // matches; otherwise leave the cursor in place (absence is fine).
                    if let Some(word) = sentence.get(cursor) {
                        if self.match_token(inner, word) {
                            cursor += 1;
                        }
                    }
                }
                _ => {
                    // Required: a matching word must be present at the cursor.
                    match sentence.get(cursor) {
                        Some(word) if self.match_token(token, word) => cursor += 1,
                        _ => return false,
                    }
                }
            }
        }

        true
    }

    /// Match a single pattern token against a word.
    ///
    /// Note on [`PatternToken::Optional`]: this entry point is only reached with a
    /// *present* word, so an optional token here matches exactly when its inner
    /// pattern does. The "optional token may be absent" semantics are handled by
    /// [`Self::match_pattern`], which can skip an optional token entirely.
    fn match_token(&self, token: &PatternToken, word: &str) -> bool {
        match token {
            PatternToken::Word(expected) => word.to_lowercase() == expected.to_lowercase(),
            PatternToken::Wildcard => true,
            PatternToken::Pos(expected_pos) => {
                // Matched against the crate's rule-based `simple_pos_tag` heuristic —
                // the only POS information available to the analyzer. There is no
                // statistical/learned POS tagger here, so accuracy is limited to what
                // that lexical+suffix heuristic provides (documented limitation).
                self.simple_pos_tag(word) == *expected_pos
            }
            PatternToken::PhoneticFeature(feature) => word_has_phonetic_feature(word, feature),
            PatternToken::Optional(inner) => self.match_token(inner, word),
            PatternToken::Alternative(alternatives) => {
                alternatives.iter().any(|alt| self.match_token(alt, word))
            }
        }
    }

    /// Load default context rules
    fn load_default_rules(&mut self) {
        // Sentence-initial context rule
        self.rules.push(ContextRule {
            name: "sentence_initial".to_string(),
            priority: 10,
            conditions: vec![ContextCondition::Position(
                PositionCondition::SentenceInitial,
            )],
            context: PronunciationContext::SentenceInitial,
            confidence: 0.9,
        });

        // Sentence-final context rule
        self.rules.push(ContextRule {
            name: "sentence_final".to_string(),
            priority: 10,
            conditions: vec![ContextCondition::Position(PositionCondition::SentenceFinal)],
            context: PronunciationContext::SentenceFinal,
            confidence: 0.9,
        });

        // Stressed position rule (simplified)
        self.rules.push(ContextRule {
            name: "stressed_position".to_string(),
            priority: 5,
            conditions: vec![ContextCondition::Position(
                PositionCondition::RelativePosition(0.3),
            )],
            context: PronunciationContext::Stressed,
            confidence: 0.6,
        });
    }

    /// Load default patterns
    fn load_default_patterns(&mut self) {
        // Pattern for "the" + noun (stressed context)
        self.patterns.insert(
            "the_noun".to_string(),
            ContextPattern {
                tokens: vec![
                    PatternToken::Word("the".to_string()),
                    PatternToken::Pos(PartOfSpeech::Noun),
                ],
                context: PronunciationContext::Stressed,
                confidence: 0.7,
                frequency: 0.8,
            },
        );
    }

    /// Add custom context rule
    pub fn add_rule(&mut self, rule: ContextRule) {
        self.rules.push(rule);
        // Sort by priority
        self.rules.sort_by_key(|b| std::cmp::Reverse(b.priority));
    }

    /// Add custom pattern
    pub fn add_pattern(&mut self, name: String, pattern: ContextPattern) {
        self.patterns.insert(name, pattern);
    }

    /// Clear POS cache
    pub fn clear_pos_cache(&mut self) {
        self.pos_cache.clear();
    }
}

/// Approximate the IPA phoneme produced by a single English grapheme.
///
/// This is a deliberately coarse, deterministic per-letter map used only to derive
/// a phonetic *environment* for context matching when no full G2P transcription is
/// available. Digraphs, silent letters and context-dependent realisations are not
/// modelled; the resulting symbols are looked up in [`crate::phonology::get_features`]
/// for distinctive-feature comparison. Returns `None` for non-alphabetic characters.
fn grapheme_to_ipa(c: char) -> Option<&'static str> {
    let symbol = match c.to_ascii_lowercase() {
        // Vowels (mapped to monophthongs known to the feature table).
        'a' => "a",
        'e' => "e",
        'i' => "i",
        'o' => "o",
        'u' => "u",
        // Consonants.
        'b' => "b",
        'c' | 'k' | 'q' | 'x' => "k",
        'd' => "d",
        'f' => "f",
        'g' => "g",
        'h' => "h",
        'j' => "dʒ",
        'l' => "l",
        'm' => "m",
        'n' => "n",
        'p' => "p",
        'r' => "r",
        's' => "s",
        't' => "t",
        'v' => "v",
        'w' => "w",
        'y' => "j",
        'z' => "z",
        _ => return None,
    };
    Some(symbol)
}

/// Collect up to `n` approximate IPA phonemes immediately preceding `word_index`,
/// returned in left-to-right order (so the final element is adjacent to the word).
fn collect_preceding_ipa(sentence: &[String], word_index: usize, n: usize) -> Vec<&'static str> {
    let mut phonemes = Vec::new();
    for word in &sentence[..word_index.min(sentence.len())] {
        for c in word.chars() {
            if let Some(symbol) = grapheme_to_ipa(c) {
                phonemes.push(symbol);
            }
        }
    }
    let start = phonemes.len().saturating_sub(n);
    phonemes[start..].to_vec()
}

/// Collect up to `n` approximate IPA phonemes immediately following `word_index`,
/// returned in left-to-right order (so the first element is adjacent to the word).
fn collect_following_ipa(sentence: &[String], word_index: usize, n: usize) -> Vec<&'static str> {
    let mut phonemes = Vec::new();
    let start = (word_index + 1).min(sentence.len());
    'outer: for word in &sentence[start..] {
        for c in word.chars() {
            if let Some(symbol) = grapheme_to_ipa(c) {
                phonemes.push(symbol);
                if phonemes.len() >= n {
                    break 'outer;
                }
            }
        }
    }
    phonemes
}

/// First approximate IPA phoneme of a word (its onset), if any letter maps.
fn word_initial_ipa(word: &str) -> Option<&'static str> {
    word.chars().find_map(grapheme_to_ipa)
}

/// Return `true` when `expected` and `actual` agree on the single member of
/// `category` specified by `expected`. If `expected` does not specify any member of
/// the category, the category is not constraining and the function returns `true`.
fn category_matches(
    expected: &[PhonologicalFeature],
    actual: &[PhonologicalFeature],
    category: &[PhonologicalFeature],
) -> bool {
    match category
        .iter()
        .copied()
        .find(|feat| expected.contains(feat))
    {
        Some(expected_feat) => actual.contains(&expected_feat),
        None => true,
    }
}

/// Decide whether an `expected` phoneme symbol matches an `actual` phoneme symbol
/// using IPA distinctive features.
///
/// Comparison rules:
/// * Identical symbols always match.
/// * Otherwise both symbols are resolved to feature sets via
///   [`crate::phonology::get_features`]; an unknown symbol (empty feature set)
///   only matches by exact string (already handled above), so it returns `false`.
/// * A vowel never matches a consonant (and vice versa).
/// * Two consonants match iff they agree on voicing, place and manner.
/// * Two vowels match iff they agree on backness (front/central/back) and height
///   (high/mid/low).
fn phoneme_features_match(expected: &str, actual: &str) -> bool {
    use PhonologicalFeature::*;

    if expected == actual {
        return true;
    }

    let expected_features = get_features(expected);
    let actual_features = get_features(actual);
    if expected_features.is_empty() || actual_features.is_empty() {
        // At least one symbol is outside the feature table; with the exact-match
        // case already excluded above there is no feature-based reason to match.
        return false;
    }

    let expected_vowel = expected_features.contains(&Vowel);
    let actual_vowel = actual_features.contains(&Vowel);
    if expected_vowel != actual_vowel {
        return false;
    }

    if expected_vowel {
        const BACKNESS: [PhonologicalFeature; 3] = [Front, Central, Back];
        const HEIGHT: [PhonologicalFeature; 3] = [High, Mid, Low];
        category_matches(&expected_features, &actual_features, &BACKNESS)
            && category_matches(&expected_features, &actual_features, &HEIGHT)
    } else {
        const VOICING: [PhonologicalFeature; 2] = [Voiced, Voiceless];
        const PLACE: [PhonologicalFeature; 8] = [
            Bilabial,
            Labiodental,
            Dental,
            Alveolar,
            Postalveolar,
            Palatal,
            Velar,
            Glottal,
        ];
        const MANNER: [PhonologicalFeature; 6] = [Stop, Fricative, Affricate, Nasal, Liquid, Glide];
        category_matches(&expected_features, &actual_features, &VOICING)
            && category_matches(&expected_features, &actual_features, &PLACE)
            && category_matches(&expected_features, &actual_features, &MANNER)
    }
}

/// Test whether a word's onset phoneme carries the named IPA feature.
///
/// The feature name is matched case-insensitively against the vowel/consonant
/// class, voicing, manner and place dimensions of [`PhonologicalFeature`]. The
/// word's onset is approximated by its first mappable grapheme. Unknown feature
/// names match nothing (conservative) rather than defaulting to `true`.
fn word_has_phonetic_feature(word: &str, feature: &str) -> bool {
    use PhonologicalFeature::*;

    let Some(initial) = word_initial_ipa(word) else {
        return false;
    };
    let features = get_features(initial);
    if features.is_empty() {
        return false;
    }

    match feature.trim().to_ascii_lowercase().as_str() {
        // Broad class.
        "vowel" => features.contains(&Vowel),
        "consonant" => !features.contains(&Vowel),
        // Voicing.
        "voiced" => features.contains(&Voiced),
        "voiceless" | "unvoiced" => features.contains(&Voiceless),
        // Manner.
        "stop" | "plosive" => features.contains(&Stop),
        "fricative" => features.contains(&Fricative),
        "affricate" => features.contains(&Affricate),
        "nasal" => features.contains(&Nasal),
        "liquid" => features.contains(&Liquid),
        "glide" | "approximant" | "semivowel" => features.contains(&Glide),
        // Place of articulation.
        "bilabial" => features.contains(&Bilabial),
        "labiodental" => features.contains(&Labiodental),
        "dental" => features.contains(&Dental),
        "alveolar" => features.contains(&Alveolar),
        "postalveolar" => features.contains(&Postalveolar),
        "palatal" => features.contains(&Palatal),
        "velar" => features.contains(&Velar),
        "glottal" => features.contains(&Glottal),
        // Vowel backness / height.
        "front" => features.contains(&Front),
        "central" => features.contains(&Central),
        "back" => features.contains(&Back),
        "high" => features.contains(&High),
        "mid" => features.contains(&Mid),
        "low" => features.contains(&Low),
        _ => false,
    }
}

impl Default for ContextAnalyzer {
    fn default() -> Self {
        Self::new(LanguageCode::EnUs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_analyzer_creation() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        assert_eq!(analyzer._language, LanguageCode::EnUs);
        assert!(!analyzer.rules.is_empty());
    }

    #[test]
    fn test_sentence_initial_context() {
        let mut analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec!["Hello".to_string(), "world".to_string()];

        let result = analyzer.analyze_context("Hello", &sentence, 0).unwrap();
        assert!(result.primary_context.is_some());

        if let Some(PronunciationContext::SentenceInitial) = result.primary_context {
            // Expected
        } else {
            panic!("Expected SentenceInitial context");
        }
    }

    #[test]
    fn test_sentence_final_context() {
        let mut analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec!["Hello".to_string(), "world".to_string()];

        let result = analyzer.analyze_context("world", &sentence, 1).unwrap();
        assert!(result.primary_context.is_some());

        if let Some(PronunciationContext::SentenceFinal) = result.primary_context {
            // Expected
        } else {
            panic!("Expected SentenceFinal context");
        }
    }

    #[test]
    fn test_pos_tagging() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);

        assert_eq!(analyzer.simple_pos_tag("the"), PartOfSpeech::Determiner);
        assert_eq!(analyzer.simple_pos_tag("quickly"), PartOfSpeech::Adverb);
        assert_eq!(analyzer.simple_pos_tag("running"), PartOfSpeech::Verb);
        assert_eq!(
            analyzer.simple_pos_tag("beautiful"),
            PartOfSpeech::Adjective
        );
    }

    #[test]
    fn test_preceding_words() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ];
        let words = vec!["the".to_string(), "quick".to_string()];

        assert!(analyzer.check_preceding_words(&words, &sentence, 2));
        assert!(!analyzer.check_preceding_words(&words, &sentence, 1));
    }

    #[test]
    fn test_following_words() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ];
        let words = vec!["brown".to_string(), "fox".to_string()];

        assert!(analyzer.check_following_words(&words, &sentence, 1));
        assert!(!analyzer.check_following_words(&words, &sentence, 2));
    }

    fn phonetic_condition(
        preceding: Option<Vec<&str>>,
        following: Option<Vec<&str>>,
    ) -> PhoneticCondition {
        PhoneticCondition {
            preceding: preceding.map(|v| v.into_iter().map(String::from).collect()),
            following: following.map(|v| v.into_iter().map(String::from).collect()),
            syllable_structure: None,
            stress_pattern: None,
        }
    }

    #[test]
    fn test_grapheme_to_ipa_and_features() {
        // Sanity: graphemes resolve to symbols the feature table understands.
        assert_eq!(grapheme_to_ipa('s'), Some("s"));
        assert_eq!(grapheme_to_ipa('a'), Some("a"));
        assert_eq!(grapheme_to_ipa('1'), None);
        assert!(
            get_features(grapheme_to_ipa('s').unwrap()).contains(&PhonologicalFeature::Fricative)
        );
    }

    #[test]
    fn test_phoneme_features_match() {
        // Identical symbols match.
        assert!(phoneme_features_match("s", "s"));
        // Same place + manner but different voicing must NOT match (/s/ vs /z/).
        assert!(!phoneme_features_match("z", "s"));
        // Same place + voicing but different manner must NOT match (/t/ vs /s/).
        assert!(!phoneme_features_match("t", "s"));
        // Different place must NOT match (/p/ vs /t/).
        assert!(!phoneme_features_match("p", "t"));
        // Vowel vs consonant never match.
        assert!(!phoneme_features_match("a", "t"));
        // Different vowels (backness/height) do not match.
        assert!(!phoneme_features_match("i", "a"));
        // Unknown symbol only matches itself.
        assert!(!phoneme_features_match("@@", "s"));
    }

    #[test]
    fn test_phonetic_condition_matching_context() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        // "cats are good": for the word "are" (index 1) the preceding phoneme is
        // /s/ (end of "cats") and the following phoneme is /g/ (start of "good").
        let sentence = vec!["cats".to_string(), "are".to_string(), "good".to_string()];

        let preceding_match = phonetic_condition(Some(vec!["s"]), None);
        assert!(analyzer
            .evaluate_phonetic_condition(&preceding_match, "are", &sentence, 1)
            .unwrap());

        let following_match = phonetic_condition(None, Some(vec!["g"]));
        assert!(analyzer
            .evaluate_phonetic_condition(&following_match, "are", &sentence, 1)
            .unwrap());
    }

    #[test]
    fn test_phonetic_condition_not_matching_context() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec!["cats".to_string(), "are".to_string(), "good".to_string()];

        // Preceding phoneme is /s/, condition wants /m/ (nasal, voiced) -> no match.
        let nasal = phonetic_condition(Some(vec!["m"]), None);
        assert!(!analyzer
            .evaluate_phonetic_condition(&nasal, "are", &sentence, 1)
            .unwrap());

        // Voicing-only difference: /s/ present, /z/ required -> no match.
        let voiced = phonetic_condition(Some(vec!["z"]), None);
        assert!(!analyzer
            .evaluate_phonetic_condition(&voiced, "are", &sentence, 1)
            .unwrap());

        // Following phoneme is /g/ (voiced velar stop), condition wants /k/
        // (voiceless velar stop) -> voicing differs -> no match.
        let voiceless_follow = phonetic_condition(None, Some(vec!["k"]));
        assert!(!analyzer
            .evaluate_phonetic_condition(&voiceless_follow, "are", &sentence, 1)
            .unwrap());
    }

    #[test]
    fn test_phonetic_condition_insufficient_context() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec!["start".to_string(), "here".to_string()];

        // Word at index 0 has no preceding context, so a preceding requirement
        // cannot be satisfied.
        let needs_preceding = phonetic_condition(Some(vec!["t"]), None);
        assert!(!analyzer
            .evaluate_phonetic_condition(&needs_preceding, "start", &sentence, 0)
            .unwrap());
    }

    #[test]
    fn test_match_token_pos() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        assert!(analyzer.match_token(&PatternToken::Pos(PartOfSpeech::Determiner), "the"));
        // "the" is a determiner, not a noun.
        assert!(!analyzer.match_token(&PatternToken::Pos(PartOfSpeech::Noun), "the"));
        // Unknown word defaults to noun in the rule-based tagger.
        assert!(analyzer.match_token(&PatternToken::Pos(PartOfSpeech::Noun), "apple"));
    }

    #[test]
    fn test_match_token_phonetic_feature() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        // "apple" starts with a vowel.
        assert!(analyzer.match_token(&PatternToken::PhoneticFeature("vowel".to_string()), "apple"));
        assert!(!analyzer.match_token(
            &PatternToken::PhoneticFeature("consonant".to_string()),
            "apple"
        ));
        // "table" starts with a voiceless alveolar stop.
        assert!(analyzer.match_token(&PatternToken::PhoneticFeature("stop".to_string()), "table"));
        assert!(analyzer.match_token(
            &PatternToken::PhoneticFeature("voiceless".to_string()),
            "table"
        ));
        assert!(!analyzer.match_token(&PatternToken::PhoneticFeature("vowel".to_string()), "table"));
        // "milk" starts with a nasal.
        assert!(analyzer.match_token(&PatternToken::PhoneticFeature("nasal".to_string()), "milk"));
        // Unknown feature name matches nothing.
        assert!(!analyzer.match_token(
            &PatternToken::PhoneticFeature("sparkly".to_string()),
            "milk"
        ));
    }

    #[test]
    fn test_match_token_optional_present() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let optional_the = PatternToken::Optional(Box::new(PatternToken::Word("the".to_string())));
        // Present word matching the inner pattern.
        assert!(analyzer.match_token(&optional_the, "the"));
        // Present word that does not match the inner pattern.
        assert!(!analyzer.match_token(&optional_the, "cat"));
    }

    #[test]
    fn test_match_pattern_optional_absent_still_matches() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec!["the".to_string(), "cat".to_string()];

        // [the] [Noun] [optional "please"] — the optional trailing token is absent
        // from the sentence but the pattern still matches.
        let pattern = ContextPattern {
            tokens: vec![
                PatternToken::Word("the".to_string()),
                PatternToken::Pos(PartOfSpeech::Noun),
                PatternToken::Optional(Box::new(PatternToken::Word("please".to_string()))),
            ],
            context: PronunciationContext::Stressed,
            confidence: 0.5,
            frequency: 0.5,
        };
        assert!(analyzer.match_pattern(&pattern, &sentence, 1));
    }

    #[test]
    fn test_match_pattern_required_mismatch_fails() {
        let analyzer = ContextAnalyzer::new(LanguageCode::EnUs);
        let sentence = vec!["the".to_string(), "cat".to_string()];

        // Required second token "dog" does not match "cat" -> pattern fails.
        let pattern = ContextPattern {
            tokens: vec![
                PatternToken::Word("the".to_string()),
                PatternToken::Word("dog".to_string()),
            ],
            context: PronunciationContext::Stressed,
            confidence: 0.5,
            frequency: 0.5,
        };
        assert!(!analyzer.match_pattern(&pattern, &sentence, 1));
    }
}
