//! Context-aware understanding module.
//!
//! This module implements context-aware speech recognition that considers
//! environmental factors, conversation history, user profiles, and temporal context
//! to improve recognition accuracy and provide better user experience.

use super::{
    ContextConfig, ContextInformation, DayOfWeek, EnvironmentContext, LocationType,
    MultiModalError, SpeechPatterns, TemporalContext, TimeOfDay, UserProfile,
};
use chrono::{Datelike, Timelike, Utc};
use parking_lot::RwLock;
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

/// Context-aware processor
pub struct ContextAwareProcessor {
    config: ContextConfig,
    conversation_buffer: Arc<RwLock<VecDeque<String>>>,
    user_profile_store: Arc<RwLock<UserProfileStore>>,
    environment_analyzer: Arc<EnvironmentAnalyzer>,
}

impl ContextAwareProcessor {
    /// Create a new context-aware processor
    #[must_use]
    pub fn new(config: ContextConfig) -> Self {
        let conversation_buffer = Arc::new(RwLock::new(VecDeque::with_capacity(
            config.max_history_length,
        )));
        let user_profile_store = Arc::new(RwLock::new(UserProfileStore::new()));
        let environment_analyzer = Arc::new(EnvironmentAnalyzer::new());

        Self {
            config,
            conversation_buffer,
            user_profile_store,
            environment_analyzer,
        }
    }

    /// Process with context
    pub fn process_with_context(
        &self,
        text: &str,
        context: Option<&ContextInformation>,
    ) -> Result<ContextualResult, MultiModalError> {
        // Add to conversation history
        if self.config.enable_conversation_history {
            let mut buffer = self.conversation_buffer.write();
            buffer.push_back(text.to_string());
            if buffer.len() > self.config.max_history_length {
                buffer.pop_front();
            }
        }

        // Analyze context
        let context_score = if let Some(ctx) = context {
            self.analyze_context(ctx, text)?
        } else {
            ContextScore::default()
        };

        // Apply context-based corrections
        let corrected_text = self.apply_context_corrections(text, &context_score)?;

        // Get temporal context
        let temporal = if self.config.enable_temporal_context {
            Some(self.get_current_temporal_context())
        } else {
            None
        };

        Ok(ContextualResult {
            original_text: text.to_string(),
            corrected_text,
            context_score,
            temporal_context: temporal,
            conversation_context: self.get_conversation_context(),
        })
    }

    /// Analyze context information
    fn analyze_context(
        &self,
        context: &ContextInformation,
        text: &str,
    ) -> Result<ContextScore, MultiModalError> {
        let mut score = ContextScore::default();

        // Environment score
        if self.config.enable_environment {
            score.environment = self.environment_analyzer.analyze(&context.environment)?;
        }

        // Conversation history score
        if self.config.enable_conversation_history {
            score.conversation_coherence = self.analyze_conversation_coherence(text)?;
        }

        // User profile score
        if self.config.enable_user_profile {
            if let Some(ref profile) = context.user_profile {
                score.user_profile = self.analyze_user_profile(text, profile)?;
            }
        }

        // Temporal relevance
        if self.config.enable_temporal_context {
            score.temporal_relevance = self.analyze_temporal_context(&context.temporal, text)?;
        }

        // Overall confidence
        score.overall = (score.environment
            + score.conversation_coherence
            + score.user_profile
            + score.temporal_relevance)
            / 4.0;

        Ok(score)
    }

    /// Apply context-based text corrections
    fn apply_context_corrections(
        &self,
        text: &str,
        context_score: &ContextScore,
    ) -> Result<String, MultiModalError> {
        let mut corrected = text.to_string();

        // If context score is high, we might make more aggressive corrections
        if context_score.overall > 0.7 {
            // Apply user-specific pronunciation corrections
            // Apply domain-specific vocabulary
            // Apply conversation-based predictions
        }

        Ok(corrected)
    }

    /// Analyze conversation coherence
    fn analyze_conversation_coherence(&self, text: &str) -> Result<f32, MultiModalError> {
        let buffer = self.conversation_buffer.read();
        if buffer.is_empty() {
            return Ok(0.5); // Neutral score for first utterance
        }

        // Simplified coherence analysis
        // In production, use semantic similarity, topic modeling, etc.
        let mut coherence = 0.7;

        // Apply decay to older context
        let recent_context: Vec<_> = buffer.iter().rev().take(3).collect();
        for (i, prev_text) in recent_context.iter().enumerate() {
            let decay_factor = self.config.context_decay.powi(i as i32);
            // Check for word overlap (simplified)
            let overlap = self.calculate_word_overlap(text, prev_text);
            coherence += overlap * decay_factor;
        }

        Ok((coherence / 2.0).min(1.0))
    }

    /// Calculate word overlap between texts
    fn calculate_word_overlap(&self, text1: &str, text2: &str) -> f32 {
        let words1: Vec<_> = text1.split_whitespace().collect();
        let words2: Vec<_> = text2.split_whitespace().collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let overlap = words1.iter().filter(|w| words2.contains(w)).count();

        overlap as f32 / words1.len().max(words2.len()) as f32
    }

    /// Analyze user profile match
    fn analyze_user_profile(
        &self,
        text: &str,
        profile: &UserProfile,
    ) -> Result<f32, MultiModalError> {
        let mut score: f32 = 0.5;

        // Check for common topics
        for topic in &profile.common_topics {
            if text.to_lowercase().contains(&topic.to_lowercase()) {
                score += 0.1;
            }
        }

        // Check for speech patterns
        if let Some(ref patterns) = profile.speech_patterns {
            for filler in &patterns.filler_words {
                if text.contains(filler) {
                    score += 0.05;
                }
            }
        }

        Ok(score.min(1.0))
    }

    /// Analyze temporal context relevance
    fn analyze_temporal_context(
        &self,
        temporal: &TemporalContext,
        text: &str,
    ) -> Result<f32, MultiModalError> {
        let mut score: f32 = 0.5;

        // Time-based relevance
        if let Some(time) = temporal.time_of_day {
            score += match time {
                TimeOfDay::Morning if text.contains("morning") || text.contains("breakfast") => 0.2,
                TimeOfDay::Afternoon if text.contains("lunch") || text.contains("afternoon") => 0.2,
                TimeOfDay::Evening if text.contains("evening") || text.contains("dinner") => 0.2,
                TimeOfDay::Night if text.contains("night") || text.contains("sleep") => 0.2,
                _ => 0.0,
            };
        }

        Ok(score.min(1.0))
    }

    /// Get current temporal context
    fn get_current_temporal_context(&self) -> TemporalContext {
        let now = Utc::now();
        let hour = now.hour();

        let time_of_day = match hour {
            6..=11 => Some(TimeOfDay::Morning),
            12..=17 => Some(TimeOfDay::Afternoon),
            18..=21 => Some(TimeOfDay::Evening),
            _ => Some(TimeOfDay::Night),
        };

        let day_of_week = match now.weekday() {
            chrono::Weekday::Mon => Some(DayOfWeek::Monday),
            chrono::Weekday::Tue => Some(DayOfWeek::Tuesday),
            chrono::Weekday::Wed => Some(DayOfWeek::Wednesday),
            chrono::Weekday::Thu => Some(DayOfWeek::Thursday),
            chrono::Weekday::Fri => Some(DayOfWeek::Friday),
            chrono::Weekday::Sat => Some(DayOfWeek::Saturday),
            chrono::Weekday::Sun => Some(DayOfWeek::Sunday),
        };

        TemporalContext {
            time_of_day,
            day_of_week,
            session_duration: None,
        }
    }

    /// Get conversation context summary
    fn get_conversation_context(&self) -> Vec<String> {
        let buffer = self.conversation_buffer.read();
        buffer.iter().cloned().collect()
    }

    /// Update user profile
    pub fn update_user_profile(&self, user_id: &str, text: &str) -> Result<(), MultiModalError> {
        let mut store = self.user_profile_store.write();
        store.update(user_id, text);
        Ok(())
    }

    /// Get user profile
    #[must_use]
    pub fn get_user_profile(&self, user_id: &str) -> Option<UserProfile> {
        let store = self.user_profile_store.read();
        store.get(user_id)
    }
}

/// Environment analyzer
pub struct EnvironmentAnalyzer;

impl EnvironmentAnalyzer {
    /// Create new environment analyzer
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Analyze environment context
    pub fn analyze(&self, env: &EnvironmentContext) -> Result<f32, MultiModalError> {
        let mut score: f32 = 0.5;

        // Adjust based on noise level
        if let Some(noise) = env.noise_level {
            score += if noise < 40.0 {
                0.2 // Quiet environment, good
            } else if noise < 60.0 {
                0.1 // Normal
            } else {
                -0.1 // Noisy, more challenging
            };
        }

        // Adjust based on speaker count
        if let Some(count) = env.speaker_count {
            score += if count == 1 {
                0.2 // Single speaker
            } else {
                0.0 // Multiple speakers
            };
        }

        // Location-based adjustments
        if let Some(location) = env.location {
            score += match location {
                LocationType::Office | LocationType::Home => 0.1,
                LocationType::Indoor => 0.05,
                LocationType::Vehicle | LocationType::Outdoor => -0.05,
                LocationType::PublicSpace => -0.1,
            };
        }

        Ok(score.clamp(0.0, 1.0))
    }
}

impl Default for EnvironmentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// User profile store
pub struct UserProfileStore {
    profiles: std::collections::HashMap<String, UserProfile>,
}

impl UserProfileStore {
    /// Create new user profile store
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: std::collections::HashMap::new(),
        }
    }

    /// Get user profile
    #[must_use]
    pub fn get(&self, user_id: &str) -> Option<UserProfile> {
        self.profiles.get(user_id).cloned()
    }

    /// Update user profile with new utterance
    pub fn update(&mut self, user_id: &str, text: &str) {
        let profile = self
            .profiles
            .entry(user_id.to_string())
            .or_insert_with(|| UserProfile {
                user_id: user_id.to_string(),
                language: None,
                speech_patterns: Some(SpeechPatterns {
                    speaking_rate: 150.0,
                    filler_words: Vec::new(),
                    pronunciation_variants: Vec::new(),
                }),
                common_topics: Vec::new(),
                accent: None,
            });

        // Update common topics (simplified)
        let words: Vec<String> = text
            .split_whitespace()
            .map(std::string::ToString::to_string)
            .collect();
        for word in words {
            if word.len() > 4 && !profile.common_topics.contains(&word) {
                profile.common_topics.push(word);
            }
        }

        // Limit topics
        if profile.common_topics.len() > 50 {
            profile.common_topics.drain(0..10);
        }
    }
}

impl Default for UserProfileStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Context score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextScore {
    /// Environment context score
    pub environment: f32,

    /// Conversation coherence score
    pub conversation_coherence: f32,

    /// User profile match score
    pub user_profile: f32,

    /// Temporal relevance score
    pub temporal_relevance: f32,

    /// Overall context score
    pub overall: f32,
}

impl Default for ContextScore {
    fn default() -> Self {
        Self {
            environment: 0.5,
            conversation_coherence: 0.5,
            user_profile: 0.5,
            temporal_relevance: 0.5,
            overall: 0.5,
        }
    }
}

/// Contextual processing result
#[derive(Debug, Clone)]
pub struct ContextualResult {
    /// Original recognized text
    pub original_text: String,

    /// Context-corrected text
    pub corrected_text: String,

    /// Context score breakdown
    pub context_score: ContextScore,

    /// Temporal context at processing time
    pub temporal_context: Option<TemporalContext>,

    /// Recent conversation context
    pub conversation_context: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::ActivityLevel;

    #[test]
    fn test_context_processor_creation() {
        let config = ContextConfig::default();
        let processor = ContextAwareProcessor::new(config);
        assert_eq!(processor.conversation_buffer.read().len(), 0);
    }

    #[test]
    fn test_environment_analyzer() {
        let analyzer = EnvironmentAnalyzer::new();
        let env = EnvironmentContext {
            noise_level: Some(35.0),
            speaker_count: Some(1),
            location: Some(LocationType::Office),
            activity_level: Some(ActivityLevel::Normal),
        };

        let result = analyzer.analyze(&env);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert!(score > 0.5); // Should be positive for good environment
    }

    #[test]
    fn test_user_profile_store() {
        let mut store = UserProfileStore::new();
        store.update("user1", "hello world testing");

        let profile = store.get("user1");
        assert!(profile.is_some());
        let p = profile.unwrap();
        assert_eq!(p.user_id, "user1");
    }

    #[test]
    fn test_temporal_context() {
        let config = ContextConfig::default();
        let processor = ContextAwareProcessor::new(config);
        let temporal = processor.get_current_temporal_context();

        assert!(temporal.time_of_day.is_some());
        assert!(temporal.day_of_week.is_some());
    }

    #[test]
    fn test_word_overlap() {
        let config = ContextConfig::default();
        let processor = ContextAwareProcessor::new(config);

        let text1 = "hello world testing";
        let text2 = "hello testing again";
        let overlap = processor.calculate_word_overlap(text1, text2);

        assert!(overlap > 0.0);
        assert!(overlap <= 1.0);
    }

    #[test]
    fn test_conversation_coherence() {
        let config = ContextConfig::default();
        let processor = ContextAwareProcessor::new(config);

        // Add some context
        processor
            .conversation_buffer
            .write()
            .push_back("hello".to_string());
        processor
            .conversation_buffer
            .write()
            .push_back("how are you".to_string());

        let result = processor.analyze_conversation_coherence("I am fine");
        assert!(result.is_ok());
    }

    #[test]
    fn test_context_score_default() {
        let score = ContextScore::default();
        assert_eq!(score.environment, 0.5);
        assert_eq!(score.overall, 0.5);
    }
}
