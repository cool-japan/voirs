//! Context-sensitive pronunciation analysis and processing.

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

    /// Evaluate phonetic condition
    fn evaluate_phonetic_condition(
        &self,
        condition: &PhoneticCondition,
        _word: &str,
        sentence: &[String],
        word_index: usize,
    ) -> Result<bool> {
        // Simplified phonetic analysis
        // In practice, would need phonetic transcription of the sentence

        if let Some(_preceding) = &condition.preceding {
            // Check if preceding phonetic context matches
            if word_index > 0 {
                let prev_word = &sentence[word_index - 1];
                // Simplified check - just look at word ending
                let _last_char = prev_word.chars().last().unwrap_or(' ');
                // This would be much more sophisticated in practice
            }
        }

        if let Some(_following) = &condition.following {
            // Check if following phonetic context matches
            if word_index + 1 < sentence.len() {
                let next_word = &sentence[word_index + 1];
                // Simplified check - just look at word beginning
                let _first_char = next_word.chars().next().unwrap_or(' ');
                // This would be much more sophisticated in practice
            }
        }

        // For now, return true (would need proper phonetic analysis)
        Ok(true)
    }

    /// Match pattern against sentence
    fn match_pattern(
        &self,
        pattern: &ContextPattern,
        sentence: &[String],
        word_index: usize,
    ) -> bool {
        // Simplified pattern matching
        // In practice, would implement full pattern matching with all token types

        let start_index = word_index.saturating_sub(pattern.tokens.len() / 2);
        let end_index = (word_index + pattern.tokens.len() / 2 + 1).min(sentence.len());

        if end_index - start_index < pattern.tokens.len() {
            return false;
        }

        // Simple token matching (would be much more sophisticated)
        for (i, token) in pattern.tokens.iter().enumerate() {
            if let Some(word) = sentence.get(start_index + i) {
                if !self.match_token(token, word) {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Match a single pattern token
    #[allow(clippy::only_used_in_recursion)]
    fn match_token(&self, token: &PatternToken, word: &str) -> bool {
        match token {
            PatternToken::Word(expected) => word.to_lowercase() == expected.to_lowercase(),
            PatternToken::Wildcard => true,
            PatternToken::Pos(_expected_pos) => {
                // Would need POS tagging here
                true // Simplified
            }
            PatternToken::PhoneticFeature(_) => {
                // Would need phonetic analysis here
                true // Simplified
            }
            PatternToken::Optional(_) => true, // Always matches
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
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
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
}
