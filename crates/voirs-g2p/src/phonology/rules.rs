//! Phonological rule definitions and application logic.
//!
//! This module provides a flexible rule-based system for defining and applying
//! phonological transformations.

use crate::{G2pError, LanguageCode, Phoneme, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Context for rule application
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleContext {
    /// Phonemes before the target
    pub left_context: Vec<String>,
    /// Target phoneme(s)
    pub target: Vec<String>,
    /// Phonemes after the target
    pub right_context: Vec<String>,
    /// Syllable position (onset, nucleus, coda)
    pub syllable_position: Option<SyllablePosition>,
    /// Stress status
    pub is_stressed: Option<bool>,
    /// Word boundary markers
    pub at_word_boundary: bool,
}

/// Position within a syllable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyllablePosition {
    /// Beginning of syllable
    Onset,
    /// Middle of syllable (vowel)
    Nucleus,
    /// End of syllable
    Coda,
}

/// A phonological rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonologicalRule {
    /// Rule name
    pub name: String,
    /// Source pattern to match
    pub source: Vec<String>,
    /// Target replacement
    pub target: Vec<String>,
    /// Left context (environment)
    pub left_context: Option<Vec<String>>,
    /// Right context (environment)
    pub right_context: Option<Vec<String>>,
    /// Required conditions
    pub conditions: Vec<RuleCondition>,
    /// Rule priority (higher = applied first)
    pub priority: u32,
    /// Language this rule applies to
    pub language: Option<LanguageCode>,
}

/// Condition for rule application
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuleCondition {
    /// Syllable position must match
    SyllablePosition(SyllablePosition),
    /// Stress must match
    Stressed(bool),
    /// At word boundary
    WordBoundary,
    /// Not at word boundary
    NotWordBoundary,
    /// Custom condition (feature-based)
    Feature(String, String),
}

/// Result of applying a rule
#[derive(Debug, Clone)]
pub struct RuleApplication {
    /// Position where rule was applied
    pub position: usize,
    /// Original phonemes
    pub original: Vec<Phoneme>,
    /// Replacement phonemes
    pub replacement: Vec<Phoneme>,
    /// Rule that was applied
    pub rule_name: String,
}

impl PhonologicalRule {
    /// Create a new phonological rule
    pub fn new(name: impl Into<String>, source: Vec<String>, target: Vec<String>) -> Self {
        Self {
            name: name.into(),
            source,
            target,
            left_context: None,
            right_context: None,
            conditions: Vec::new(),
            priority: 0,
            language: None,
        }
    }

    /// Set left context
    pub fn with_left_context(mut self, context: Vec<String>) -> Self {
        self.left_context = Some(context);
        self
    }

    /// Set right context
    pub fn with_right_context(mut self, context: Vec<String>) -> Self {
        self.right_context = Some(context);
        self
    }

    /// Add a condition
    pub fn with_condition(mut self, condition: RuleCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set language
    pub fn with_language(mut self, language: LanguageCode) -> Self {
        self.language = Some(language);
        self
    }

    /// Check if rule matches at position
    pub fn matches(&self, phonemes: &[Phoneme], position: usize) -> bool {
        // Check if source pattern matches
        if position + self.source.len() > phonemes.len() {
            return false;
        }

        for (i, source_phoneme) in self.source.iter().enumerate() {
            if phonemes[position + i].symbol != *source_phoneme {
                return false;
            }
        }

        // Check left context if specified
        if let Some(ref left) = self.left_context {
            if position < left.len() {
                return false;
            }
            for (i, context_phoneme) in left.iter().enumerate() {
                let check_pos = position - left.len() + i;
                if phonemes[check_pos].symbol != *context_phoneme {
                    return false;
                }
            }
        }

        // Check right context if specified
        if let Some(ref right) = self.right_context {
            let right_start = position + self.source.len();
            if right_start + right.len() > phonemes.len() {
                return false;
            }
            for (i, context_phoneme) in right.iter().enumerate() {
                if phonemes[right_start + i].symbol != *context_phoneme {
                    return false;
                }
            }
        }

        // Check conditions
        for condition in &self.conditions {
            match condition {
                RuleCondition::Stressed(stressed) => {
                    // stress > 0 means stressed, stress == 0 means unstressed
                    let is_stressed = phonemes[position].stress > 0;
                    if is_stressed != *stressed {
                        return false;
                    }
                }
                RuleCondition::WordBoundary => {
                    if position != 0 && position != phonemes.len() - 1 {
                        return false;
                    }
                }
                RuleCondition::NotWordBoundary => {
                    if position == 0 || position == phonemes.len() - 1 {
                        return false;
                    }
                }
                _ => {} // Other conditions not implemented yet
            }
        }

        true
    }

    /// Apply rule at position
    pub fn apply_at(&self, phonemes: &[Phoneme], position: usize) -> Option<RuleApplication> {
        if !self.matches(phonemes, position) {
            return None;
        }

        let original = phonemes[position..position + self.source.len()].to_vec();
        let replacement = self
            .target
            .iter()
            .map(|s| Phoneme::new(s.clone()))
            .collect();

        Some(RuleApplication {
            position,
            original,
            replacement,
            rule_name: self.name.clone(),
        })
    }
}

/// Predefined assimilation rules
pub struct AssimilationRule;

impl AssimilationRule {
    /// English nasal place assimilation before bilabials
    pub fn nasal_before_bilabial() -> PhonologicalRule {
        PhonologicalRule::new(
            "Nasal place assimilation before bilabials",
            vec!["n".to_string()],
            vec!["m".to_string()],
        )
        .with_right_context(vec!["p".to_string()])
        .with_priority(100)
        .with_language(LanguageCode::EnUs)
    }

    /// English nasal place assimilation before velars
    pub fn nasal_before_velar() -> PhonologicalRule {
        PhonologicalRule::new(
            "Nasal place assimilation before velars",
            vec!["n".to_string()],
            vec!["ŋ".to_string()],
        )
        .with_right_context(vec!["k".to_string()])
        .with_priority(100)
        .with_language(LanguageCode::EnUs)
    }
}

/// Predefined vowel reduction rules
pub struct ReductionRule;

impl ReductionRule {
    /// English unstressed vowel reduction to schwa
    pub fn unstressed_to_schwa() -> PhonologicalRule {
        PhonologicalRule::new(
            "Unstressed vowel reduction to schwa",
            vec!["ʌ".to_string()],
            vec!["ə".to_string()],
        )
        .with_condition(RuleCondition::Stressed(false))
        .with_priority(50)
        .with_language(LanguageCode::EnUs)
    }
}

/// Predefined deletion/elision rules
pub struct DeletionRule;

impl DeletionRule {
    /// English schwa deletion in fast speech
    pub fn schwa_deletion() -> PhonologicalRule {
        PhonologicalRule::new(
            "Schwa deletion in unstressed syllables",
            vec!["ə".to_string()],
            vec![],
        )
        .with_condition(RuleCondition::Stressed(false))
        .with_condition(RuleCondition::NotWordBoundary)
        .with_priority(30)
        .with_language(LanguageCode::EnUs)
    }
}

/// Rule set manager
pub struct RuleSet {
    rules: Vec<PhonologicalRule>,
    language: Option<LanguageCode>,
}

impl RuleSet {
    /// Create a new empty rule set
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            language: None,
        }
    }

    /// Create a rule set for a specific language with default rules
    pub fn for_language(language: LanguageCode) -> Self {
        let mut ruleset = Self::new();
        ruleset.language = Some(language);

        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                ruleset.add_rule(AssimilationRule::nasal_before_bilabial());
                ruleset.add_rule(AssimilationRule::nasal_before_velar());
                ruleset.add_rule(ReductionRule::unstressed_to_schwa());
            }
            _ => {} // Other languages to be added
        }

        ruleset
    }

    /// Add a rule to the set
    pub fn add_rule(&mut self, rule: PhonologicalRule) {
        self.rules.push(rule);
        // Sort by priority (descending)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Apply all rules to a phoneme sequence
    pub fn apply_rules(&self, phonemes: &[Phoneme]) -> Result<Vec<Phoneme>> {
        let mut result = phonemes.to_vec();
        let mut applications = Vec::new();

        // Keep applying rules until no more changes
        loop {
            let mut changed = false;

            for rule in &self.rules {
                for i in 0..result.len() {
                    if let Some(application) = rule.apply_at(&result, i) {
                        applications.push(application.clone());

                        // Apply the replacement
                        let mut new_result = result[..i].to_vec();
                        new_result.extend(application.replacement.clone());
                        new_result.extend(result[i + application.original.len()..].to_vec());

                        result = new_result;
                        changed = true;
                        break; // Start over with new phoneme sequence
                    }
                }

                if changed {
                    break; // Start over with highest priority rule
                }
            }

            if !changed {
                break;
            }
        }

        Ok(result)
    }
}

impl Default for RuleSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_creation() {
        let rule = PhonologicalRule::new("Test rule", vec!["a".to_string()], vec!["b".to_string()])
            .with_priority(10)
            .with_language(LanguageCode::EnUs);

        assert_eq!(rule.name, "Test rule");
        assert_eq!(rule.source, vec!["a".to_string()]);
        assert_eq!(rule.target, vec!["b".to_string()]);
        assert_eq!(rule.priority, 10);
    }

    #[test]
    fn test_rule_matches() {
        let rule = PhonologicalRule::new("Test", vec!["n".to_string()], vec!["m".to_string()])
            .with_right_context(vec!["p".to_string()]);

        let phonemes = vec![Phoneme::new("n".to_string()), Phoneme::new("p".to_string())];

        assert!(rule.matches(&phonemes, 0));

        let phonemes2 = vec![Phoneme::new("n".to_string()), Phoneme::new("t".to_string())];

        assert!(!rule.matches(&phonemes2, 0));
    }

    #[test]
    fn test_rule_application() {
        let rule = AssimilationRule::nasal_before_bilabial();

        let phonemes = vec![
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("n".to_string()),
            Phoneme::new("p".to_string()),
            Phoneme::new("ʊ".to_string()),
            Phoneme::new("t".to_string()),
        ];

        let application = rule.apply_at(&phonemes, 1);
        assert!(application.is_some());

        let app = application.unwrap();
        assert_eq!(app.position, 1);
        assert_eq!(app.original[0].symbol, "n");
        assert_eq!(app.replacement[0].symbol, "m");
    }

    #[test]
    fn test_ruleset_application() {
        let ruleset = RuleSet::for_language(LanguageCode::EnUs);

        let phonemes = vec![
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("n".to_string()),
            Phoneme::new("p".to_string()),
            Phoneme::new("ʊ".to_string()),
            Phoneme::new("t".to_string()),
        ];

        let result = ruleset.apply_rules(&phonemes).unwrap();

        // Should have converted /n/ to /m/ before /p/
        assert_eq!(result[1].symbol, "m");
    }
}
