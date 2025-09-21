//! # Conversation Context System
//!
//! Context-aware emotional adaptation for maintaining coherent and appropriate
//! emotional expression throughout conversations based on history, relationships,
//! and conversational patterns.

use crate::{
    Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Configuration for conversation context tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    /// Maximum number of conversation turns to track
    pub max_history_size: usize,
    /// Maximum age of conversation context in seconds
    pub max_context_age_secs: u64,
    /// Weight for previous emotion influence (0.0 to 1.0)
    pub emotion_momentum_weight: f32,
    /// Weight for speaker relationship influence
    pub relationship_weight: f32,
    /// Weight for topic context influence
    pub topic_weight: f32,
    /// Enable automatic emotion adaptation
    pub auto_adaptation: bool,
    /// Minimum conversation turns before adaptation kicks in
    pub min_turns_for_adaptation: usize,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_history_size: 50,
            max_context_age_secs: 3600, // 1 hour
            emotion_momentum_weight: 0.3,
            relationship_weight: 0.2,
            topic_weight: 0.2,
            auto_adaptation: true,
            min_turns_for_adaptation: 3,
        }
    }
}

/// Represents a single turn in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Unique identifier for this turn
    pub id: Uuid,
    /// Speaker identifier
    pub speaker_id: String,
    /// Text content of the turn
    pub text: String,
    /// Emotion used for this turn
    pub emotion: Emotion,
    /// Emotion intensity
    pub intensity: EmotionIntensity,
    /// Emotion parameters used
    pub emotion_parameters: EmotionParameters,
    /// Timestamp when this turn occurred
    pub timestamp: std::time::SystemTime,
    /// Duration of this turn in milliseconds
    pub duration_ms: Option<u64>,
    /// Detected topic context
    pub topic_context: TopicContext,
    /// Speaker relationship at this turn
    pub relationship: SpeakerRelationship,
}

/// Topic context categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopicContext {
    /// Casual conversation
    Casual,
    /// Formal discussion
    Formal,
    /// Business/professional context
    Business,
    /// Personal/intimate conversation
    Personal,
    /// Educational context
    Educational,
    /// Entertainment/fun context
    Entertainment,
    /// Emotional/sensitive topics
    Emotional,
    /// Technical/informational
    Technical,
    /// News/current events
    News,
    /// Creative/artistic discussion
    Creative,
    /// Problem-solving context
    ProblemSolving,
    /// Storytelling context
    Storytelling,
}

/// Speaker relationship types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeakerRelationship {
    /// Close friends
    Friend,
    /// Family member
    Family,
    /// Professional colleague
    Colleague,
    /// Superior (boss, teacher, etc.)
    Superior,
    /// Subordinate (employee, student, etc.)
    Subordinate,
    /// Stranger/unknown person
    Stranger,
    /// Romantic partner
    Partner,
    /// Customer/client
    Customer,
    /// Service provider
    ServiceProvider,
    /// Peer/equal
    Peer,
}

/// Conversation statistics and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMetrics {
    /// Total number of turns
    pub total_turns: usize,
    /// Average emotion intensity
    pub avg_intensity: f32,
    /// Most common emotions used
    pub emotion_distribution: HashMap<Emotion, usize>,
    /// Average turn duration
    pub avg_turn_duration_ms: f32,
    /// Dominant topic context
    pub dominant_topic: Option<TopicContext>,
    /// Relationship progression
    pub relationship_progression: Vec<SpeakerRelationship>,
    /// Emotion momentum (trend)
    pub emotion_momentum: EmotionDimensions,
}

/// Conversation context adaptation suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAdaptation {
    /// Suggested emotion for next turn
    pub suggested_emotion: Emotion,
    /// Suggested intensity
    pub suggested_intensity: EmotionIntensity,
    /// Confidence in suggestion (0.0 to 1.0)
    pub confidence: f32,
    /// Reasoning for the adaptation
    pub reasoning: Vec<String>,
    /// Adjusted emotion parameters
    pub adapted_parameters: EmotionParameters,
}

/// Main conversation context manager
#[derive(Debug)]
pub struct ConversationContext {
    /// Configuration
    config: ConversationConfig,
    /// Conversation history
    history: VecDeque<ConversationTurn>,
    /// Current speaker information
    speakers: HashMap<String, SpeakerInfo>,
    /// Current conversation metrics
    metrics: ConversationMetrics,
    /// Current topic context
    current_topic: TopicContext,
    /// Current dominant relationship
    current_relationship: SpeakerRelationship,
    /// Conversation start time
    start_time: std::time::SystemTime,
}

/// Information about a speaker in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker's display name
    pub name: String,
    /// Speaker's preferred emotion style
    pub preferred_emotions: Vec<Emotion>,
    /// Speaker's typical intensity level
    pub typical_intensity: EmotionIntensity,
    /// Relationship to other speakers
    pub relationships: HashMap<String, SpeakerRelationship>,
    /// Speaker's communication style
    pub communication_style: CommunicationStyle,
    /// Number of turns by this speaker
    pub turn_count: usize,
}

/// Communication style characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationStyle {
    /// Expressive and emotional
    Expressive,
    /// Reserved and controlled
    Reserved,
    /// Professional and formal
    Professional,
    /// Casual and relaxed
    Casual,
    /// Analytical and logical
    Analytical,
    /// Empathetic and caring
    Empathetic,
    /// Assertive and direct
    Assertive,
    /// Playful and humorous
    Playful,
}

impl ConversationContext {
    /// Create a new conversation context
    pub fn new() -> Self {
        Self::with_config(ConversationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ConversationConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            speakers: HashMap::new(),
            metrics: ConversationMetrics::default(),
            current_topic: TopicContext::Casual,
            current_relationship: SpeakerRelationship::Peer,
            start_time: std::time::SystemTime::now(),
        }
    }

    /// Add a speaker to the conversation
    pub fn add_speaker(&mut self, speaker_id: String, speaker_info: SpeakerInfo) {
        self.speakers.insert(speaker_id, speaker_info);
    }

    /// Get speaker information
    pub fn get_speaker(&self, speaker_id: &str) -> Option<&SpeakerInfo> {
        self.speakers.get(speaker_id)
    }

    /// Add a conversation turn
    pub fn add_turn(
        &mut self,
        speaker_id: String,
        text: String,
        emotion: Emotion,
        intensity: EmotionIntensity,
        emotion_parameters: EmotionParameters,
    ) -> Result<Uuid> {
        let turn_id = Uuid::new_v4();

        // Detect topic context from text
        let topic_context = self.detect_topic_context(&text);

        // Determine relationship context
        let relationship = self.determine_relationship(&speaker_id);

        let turn = ConversationTurn {
            id: turn_id,
            speaker_id: speaker_id.clone(),
            text,
            emotion,
            intensity,
            emotion_parameters,
            timestamp: std::time::SystemTime::now(),
            duration_ms: None,
            topic_context,
            relationship,
        };

        // Add to history
        self.history.push_back(turn);

        // Maintain history size limit
        while self.history.len() > self.config.max_history_size {
            self.history.pop_front();
        }

        // Update speaker turn count
        if let Some(speaker) = self.speakers.get_mut(&speaker_id) {
            speaker.turn_count += 1;
        }

        // Update metrics
        self.update_metrics();

        // Update current context
        self.current_topic = topic_context;
        self.current_relationship = relationship;

        Ok(turn_id)
    }

    /// Get context-aware emotion adaptation suggestions
    pub fn get_adaptation_suggestion(
        &self,
        speaker_id: &str,
        text: &str,
        base_emotion: Emotion,
        base_intensity: EmotionIntensity,
    ) -> Result<ContextAdaptation> {
        let mut reasoning = Vec::new();
        let mut confidence = 0.5;

        // Get speaker info
        let speaker_info = self.speakers.get(speaker_id);

        // Analyze conversation history for patterns
        let emotion_momentum = self.calculate_emotion_momentum();
        let topic_context = self.detect_topic_context(text);
        let relationship = self.determine_relationship(speaker_id);

        // Start with base emotion and intensity
        let mut suggested_emotion = base_emotion.clone();
        let mut suggested_intensity = base_intensity;

        // Apply emotion momentum if we have enough history
        if self.history.len() >= self.config.min_turns_for_adaptation {
            let momentum_influence = self.config.emotion_momentum_weight;

            if momentum_influence > 0.0 {
                // Check for emotional consistency needs
                if let Some(last_turn) = self.history.back() {
                    let last_emotion = &last_turn.emotion;

                    // If there's strong momentum, suggest maintaining emotional direction
                    if self.should_maintain_emotion_momentum(last_emotion, &base_emotion) {
                        suggested_emotion = last_emotion.clone();
                        reasoning.push(
                            "Maintaining emotional momentum from conversation flow".to_string(),
                        );
                        confidence += 0.2;
                    }
                }
            }
        }

        // Apply speaker-specific adaptations
        if let Some(speaker) = speaker_info {
            // Consider speaker's preferred emotions
            if speaker.preferred_emotions.contains(&base_emotion) {
                confidence += 0.1;
                reasoning.push("Emotion matches speaker's preferred style".to_string());
            }

            // Adjust intensity based on speaker's typical level
            let intensity_diff = (speaker.typical_intensity.value() - base_intensity.value()).abs();
            if intensity_diff > 0.3 {
                suggested_intensity = EmotionIntensity::new(
                    (base_intensity.value() + speaker.typical_intensity.value()) / 2.0,
                );
                reasoning.push("Adjusted intensity to match speaker's typical level".to_string());
                confidence += 0.1;
            }

            // Apply communication style adaptations
            self.apply_communication_style_adaptation(
                speaker.communication_style,
                &mut suggested_emotion,
                &mut suggested_intensity,
                &mut reasoning,
                &mut confidence,
            );
        }

        // Apply topic context adaptations
        self.apply_topic_context_adaptation(
            topic_context,
            &mut suggested_emotion,
            &mut suggested_intensity,
            &mut reasoning,
            &mut confidence,
        );

        // Apply relationship context adaptations
        self.apply_relationship_adaptation(
            relationship,
            speaker_id,
            &mut suggested_emotion,
            &mut suggested_intensity,
            &mut reasoning,
            &mut confidence,
        );

        // Create adapted emotion parameters
        let mut adapted_parameters = EmotionParameters::neutral();
        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(suggested_emotion.clone(), suggested_intensity);
        adapted_parameters.emotion_vector = emotion_vector;

        Ok(ContextAdaptation {
            suggested_emotion,
            suggested_intensity,
            confidence: confidence.min(1.0),
            reasoning,
            adapted_parameters,
        })
    }

    /// Get conversation statistics
    pub fn get_metrics(&self) -> &ConversationMetrics {
        &self.metrics
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.metrics = ConversationMetrics::default();
        self.start_time = std::time::SystemTime::now();
    }

    /// Export conversation history
    pub fn export_history(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.history).map_err(Into::into)
    }

    /// Import conversation history
    pub fn import_history(&mut self, json_data: &str) -> Result<()> {
        let history: VecDeque<ConversationTurn> = serde_json::from_str(json_data)?;
        self.history = history;
        self.update_metrics();
        Ok(())
    }

    /// Detect topic context from text content
    fn detect_topic_context(&self, text: &str) -> TopicContext {
        let text_lower = text.to_lowercase();

        // Business/Professional keywords
        if text_lower.contains("meeting")
            || text_lower.contains("project")
            || text_lower.contains("deadline")
            || text_lower.contains("business")
            || text_lower.contains("report")
            || text_lower.contains("presentation")
        {
            return TopicContext::Business;
        }

        // Emotional keywords
        if text_lower.contains("feel")
            || text_lower.contains("love")
            || text_lower.contains("heart")
            || text_lower.contains("emotional")
            || text_lower.contains("relationship")
            || text_lower.contains("personal")
        {
            return TopicContext::Emotional;
        }

        // Educational keywords
        if text_lower.contains("learn")
            || text_lower.contains("study")
            || text_lower.contains("education")
            || text_lower.contains("teach")
            || text_lower.contains("school")
            || text_lower.contains("university")
        {
            return TopicContext::Educational;
        }

        // Entertainment keywords
        if text_lower.contains("fun")
            || text_lower.contains("game")
            || text_lower.contains("movie")
            || text_lower.contains("music")
            || text_lower.contains("play")
            || text_lower.contains("entertainment")
        {
            return TopicContext::Entertainment;
        }

        // Technical keywords
        if text_lower.contains("code")
            || text_lower.contains("algorithm")
            || text_lower.contains("technical")
            || text_lower.contains("software")
            || text_lower.contains("computer")
            || text_lower.contains("programming")
        {
            return TopicContext::Technical;
        }

        // Problem-solving keywords
        if text_lower.contains("problem")
            || text_lower.contains("solution")
            || text_lower.contains("solve")
            || text_lower.contains("fix")
            || text_lower.contains("help")
            || text_lower.contains("issue")
        {
            return TopicContext::ProblemSolving;
        }

        // Formal keywords
        if text_lower.contains("formal")
            || text_lower.contains("official")
            || text_lower.contains("professional")
            || text_lower.contains("sir")
            || text_lower.contains("madam")
            || text_lower.contains("respectfully")
        {
            return TopicContext::Formal;
        }

        // Default to casual
        TopicContext::Casual
    }

    /// Determine relationship context for a speaker
    fn determine_relationship(&self, speaker_id: &str) -> SpeakerRelationship {
        if let Some(speaker) = self.speakers.get(speaker_id) {
            // Use the most recent relationship from the speaker's relationships
            // For simplicity, we'll return the first relationship or default to Peer
            speaker
                .relationships
                .values()
                .next()
                .copied()
                .unwrap_or(SpeakerRelationship::Peer)
        } else {
            SpeakerRelationship::Stranger
        }
    }

    /// Calculate emotional momentum from conversation history
    fn calculate_emotion_momentum(&self) -> EmotionDimensions {
        if self.history.is_empty() {
            return EmotionDimensions::neutral();
        }

        let recent_turns = self.history.iter().rev().take(5);
        let mut total_valence = 0.0;
        let mut total_arousal = 0.0;
        let mut total_dominance = 0.0;
        let mut count = 0;

        for turn in recent_turns {
            let dims = &turn.emotion_parameters.emotion_vector.dimensions;
            total_valence += dims.valence;
            total_arousal += dims.arousal;
            total_dominance += dims.dominance;
            count += 1;
        }

        if count > 0 {
            EmotionDimensions {
                valence: total_valence / count as f32,
                arousal: total_arousal / count as f32,
                dominance: total_dominance / count as f32,
            }
        } else {
            EmotionDimensions::neutral()
        }
    }

    /// Check if emotion momentum should be maintained
    fn should_maintain_emotion_momentum(
        &self,
        last_emotion: &Emotion,
        current_emotion: &Emotion,
    ) -> bool {
        // Maintain momentum for similar emotions or complementary emotional flow
        match (last_emotion, current_emotion) {
            (Emotion::Happy, Emotion::Excited) => true,
            (Emotion::Excited, Emotion::Happy) => true,
            (Emotion::Sad, Emotion::Melancholic) => true,
            (Emotion::Melancholic, Emotion::Sad) => true,
            (Emotion::Angry, Emotion::Custom(ref name)) if name == "frustrated" => true,
            (Emotion::Calm, Emotion::Tender) => true,
            (Emotion::Tender, Emotion::Calm) => true,
            _ => false,
        }
    }

    /// Apply communication style adaptations
    fn apply_communication_style_adaptation(
        &self,
        style: CommunicationStyle,
        emotion: &mut Emotion,
        intensity: &mut EmotionIntensity,
        reasoning: &mut Vec<String>,
        confidence: &mut f32,
    ) {
        match style {
            CommunicationStyle::Reserved => {
                if intensity.value() > 0.7 {
                    *intensity = EmotionIntensity::new(intensity.value() * 0.8);
                    reasoning
                        .push("Reduced intensity for reserved communication style".to_string());
                    *confidence += 0.1;
                }
            }
            CommunicationStyle::Expressive => {
                if intensity.value() < 0.5 {
                    *intensity = EmotionIntensity::new(intensity.value() * 1.2);
                    reasoning
                        .push("Increased intensity for expressive communication style".to_string());
                    *confidence += 0.1;
                }
            }
            CommunicationStyle::Professional => {
                // Moderate emotions for professional contexts
                if matches!(emotion, Emotion::Excited) {
                    *emotion = Emotion::Confident;
                    reasoning
                        .push("Adapted to confident emotion for professional style".to_string());
                    *confidence += 0.15;
                }
            }
            CommunicationStyle::Playful => {
                // Encourage lighter, more positive emotions
                if matches!(emotion, Emotion::Neutral) {
                    *emotion = Emotion::Happy;
                    reasoning.push("Suggested happier emotion for playful style".to_string());
                    *confidence += 0.1;
                }
            }
            _ => {} // No specific adaptations for other styles
        }
    }

    /// Apply topic context adaptations
    fn apply_topic_context_adaptation(
        &self,
        topic: TopicContext,
        emotion: &mut Emotion,
        intensity: &mut EmotionIntensity,
        reasoning: &mut Vec<String>,
        confidence: &mut f32,
    ) {
        match topic {
            TopicContext::Formal | TopicContext::Business => {
                // Moderate emotions in formal contexts
                if intensity.value() > 0.7 {
                    *intensity = EmotionIntensity::new(0.7);
                    reasoning.push("Moderated intensity for formal context".to_string());
                    *confidence += 0.15;
                }
                if matches!(emotion, Emotion::Excited) {
                    *emotion = Emotion::Confident;
                    reasoning.push("Adapted to confident emotion for business context".to_string());
                    *confidence += 0.1;
                }
            }
            TopicContext::Emotional => {
                // Allow higher emotional intensity
                if intensity.value() < 0.6 && !matches!(emotion, Emotion::Neutral) {
                    *intensity = EmotionIntensity::new(intensity.value() * 1.2);
                    reasoning.push("Increased intensity for emotional context".to_string());
                    *confidence += 0.1;
                }
            }
            TopicContext::Entertainment => {
                // Encourage positive emotions
                if matches!(emotion, Emotion::Neutral) {
                    *emotion = Emotion::Happy;
                    reasoning
                        .push("Suggested positive emotion for entertainment context".to_string());
                    *confidence += 0.1;
                }
            }
            TopicContext::ProblemSolving => {
                // Encourage calm, focused emotions
                if matches!(emotion, Emotion::Excited) {
                    *emotion = Emotion::Confident;
                    reasoning.push("Adapted to focused emotion for problem-solving".to_string());
                    *confidence += 0.1;
                }
            }
            _ => {} // No specific adaptations for other topics
        }
    }

    /// Apply relationship context adaptations
    fn apply_relationship_adaptation(
        &self,
        relationship: SpeakerRelationship,
        _speaker_id: &str,
        emotion: &mut Emotion,
        intensity: &mut EmotionIntensity,
        reasoning: &mut Vec<String>,
        confidence: &mut f32,
    ) {
        match relationship {
            SpeakerRelationship::Superior => {
                // More respectful, moderate emotions
                if intensity.value() > 0.6 {
                    *intensity = EmotionIntensity::new(0.6);
                    reasoning.push("Moderated emotion for superior relationship".to_string());
                    *confidence += 0.1;
                }
            }
            SpeakerRelationship::Family | SpeakerRelationship::Partner => {
                // Allow more emotional expression
                if intensity.value() < 0.4 && !matches!(emotion, Emotion::Neutral) {
                    *intensity = EmotionIntensity::new(intensity.value() * 1.3);
                    reasoning
                        .push("Increased emotional expression for close relationship".to_string());
                    *confidence += 0.1;
                }
            }
            SpeakerRelationship::Stranger => {
                // Conservative, polite emotions
                if intensity.value() > 0.5 {
                    *intensity = EmotionIntensity::new(0.5);
                    reasoning.push("Conservative emotion for stranger relationship".to_string());
                    *confidence += 0.1;
                }
            }
            SpeakerRelationship::Customer => {
                // Professional, helpful emotions
                if matches!(emotion, Emotion::Sad | Emotion::Angry) {
                    *emotion = Emotion::Calm;
                    reasoning.push("Professional demeanor for customer relationship".to_string());
                    *confidence += 0.15;
                }
            }
            _ => {} // No specific adaptations for other relationships
        }
    }

    /// Update conversation metrics
    fn update_metrics(&mut self) {
        self.metrics.total_turns = self.history.len();

        if self.history.is_empty() {
            return;
        }

        // Calculate average intensity
        let total_intensity: f32 = self.history.iter().map(|t| t.intensity.value()).sum();
        self.metrics.avg_intensity = total_intensity / self.history.len() as f32;

        // Calculate emotion distribution
        self.metrics.emotion_distribution.clear();
        for turn in &self.history {
            *self
                .metrics
                .emotion_distribution
                .entry(turn.emotion.clone())
                .or_insert(0) += 1;
        }

        // Calculate average turn duration
        let durations: Vec<u64> = self.history.iter().filter_map(|t| t.duration_ms).collect();
        if !durations.is_empty() {
            self.metrics.avg_turn_duration_ms =
                durations.iter().sum::<u64>() as f32 / durations.len() as f32;
        }

        // Find dominant topic
        let mut topic_counts: HashMap<TopicContext, usize> = HashMap::new();
        for turn in &self.history {
            *topic_counts.entry(turn.topic_context).or_insert(0) += 1;
        }
        self.metrics.dominant_topic = topic_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(topic, _)| topic);

        // Track relationship progression
        self.metrics.relationship_progression =
            self.history.iter().map(|t| t.relationship).collect();

        // Calculate emotion momentum
        self.metrics.emotion_momentum = self.calculate_emotion_momentum();
    }
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConversationMetrics {
    fn default() -> Self {
        Self {
            total_turns: 0,
            avg_intensity: 0.0,
            emotion_distribution: HashMap::new(),
            avg_turn_duration_ms: 0.0,
            dominant_topic: None,
            relationship_progression: Vec::new(),
            emotion_momentum: EmotionDimensions::neutral(),
        }
    }
}

impl Default for SpeakerInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            preferred_emotions: vec![Emotion::Neutral],
            typical_intensity: EmotionIntensity::MEDIUM,
            relationships: HashMap::new(),
            communication_style: CommunicationStyle::Casual,
            turn_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_context_creation() {
        let context = ConversationContext::new();
        assert_eq!(context.history.len(), 0);
        assert_eq!(context.metrics.total_turns, 0);
    }

    #[test]
    fn test_add_speaker() {
        let mut context = ConversationContext::new();
        let speaker_info = SpeakerInfo {
            name: "Alice".to_string(),
            preferred_emotions: vec![Emotion::Happy, Emotion::Calm],
            typical_intensity: EmotionIntensity::MEDIUM,
            relationships: HashMap::new(),
            communication_style: CommunicationStyle::Expressive,
            turn_count: 0,
        };

        context.add_speaker("alice".to_string(), speaker_info);
        assert!(context.get_speaker("alice").is_some());
        assert_eq!(context.get_speaker("alice").unwrap().name, "Alice");
    }

    #[test]
    fn test_add_conversation_turn() {
        let mut context = ConversationContext::new();
        let params = EmotionParameters::neutral();

        let turn_id = context
            .add_turn(
                "speaker1".to_string(),
                "Hello there!".to_string(),
                Emotion::Happy,
                EmotionIntensity::MEDIUM,
                params,
            )
            .unwrap();

        assert!(turn_id != Uuid::nil());
        assert_eq!(context.history.len(), 1);
        assert_eq!(context.metrics.total_turns, 1);
    }

    #[test]
    fn test_topic_context_detection() {
        let context = ConversationContext::new();

        assert_eq!(
            context.detect_topic_context("Let's discuss the business report"),
            TopicContext::Business
        );
        assert_eq!(
            context.detect_topic_context("I feel really emotional about this"),
            TopicContext::Emotional
        );
        assert_eq!(
            context.detect_topic_context("Let's play a fun game"),
            TopicContext::Entertainment
        );
        assert_eq!(
            context.detect_topic_context("I need to learn programming"),
            TopicContext::Educational
        );
        assert_eq!(
            context.detect_topic_context("How are you today?"),
            TopicContext::Casual
        );
    }

    #[test]
    fn test_emotion_momentum_calculation() {
        let mut context = ConversationContext::new();

        // Create proper emotion parameters with positive valence for happy emotion
        let mut params = EmotionParameters::neutral();
        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        params.emotion_vector = emotion_vector;

        // Add several happy turns
        for _ in 0..3 {
            context
                .add_turn(
                    "speaker1".to_string(),
                    "I'm happy!".to_string(),
                    Emotion::Happy,
                    EmotionIntensity::HIGH,
                    params.clone(),
                )
                .unwrap();
        }

        let momentum = context.calculate_emotion_momentum();
        assert!(momentum.valence > 0.0); // Should be positive due to happy emotions
    }

    #[test]
    fn test_adaptation_suggestion() {
        let mut context = ConversationContext::new();

        // Add a speaker with specific preferences
        let mut speaker_info = SpeakerInfo::default();
        speaker_info.name = "Bob".to_string();
        speaker_info.communication_style = CommunicationStyle::Reserved;
        speaker_info.typical_intensity = EmotionIntensity::LOW;
        context.add_speaker("bob".to_string(), speaker_info);

        let adaptation = context
            .get_adaptation_suggestion(
                "bob",
                "I'm excited about this project!",
                Emotion::Excited,
                EmotionIntensity::VERY_HIGH,
            )
            .unwrap();

        // Should suggest lower intensity for reserved speaker
        assert!(adaptation.suggested_intensity.value() < EmotionIntensity::VERY_HIGH.value());
        assert!(!adaptation.reasoning.is_empty());
    }

    #[test]
    fn test_conversation_metrics() {
        let mut context = ConversationContext::new();
        let params = EmotionParameters::neutral();

        // Add multiple turns with different emotions
        context
            .add_turn(
                "s1".to_string(),
                "Hello".to_string(),
                Emotion::Happy,
                EmotionIntensity::MEDIUM,
                params.clone(),
            )
            .unwrap();
        context
            .add_turn(
                "s2".to_string(),
                "Hi there".to_string(),
                Emotion::Happy,
                EmotionIntensity::HIGH,
                params.clone(),
            )
            .unwrap();
        context
            .add_turn(
                "s1".to_string(),
                "How are you?".to_string(),
                Emotion::Calm,
                EmotionIntensity::LOW,
                params,
            )
            .unwrap();

        let metrics = context.get_metrics();
        assert_eq!(metrics.total_turns, 3);
        assert!(metrics.avg_intensity > 0.0);
        assert_eq!(metrics.emotion_distribution.get(&Emotion::Happy), Some(&2));
        assert_eq!(metrics.emotion_distribution.get(&Emotion::Calm), Some(&1));
    }

    #[test]
    fn test_history_export_import() {
        let mut context = ConversationContext::new();
        let params = EmotionParameters::neutral();

        context
            .add_turn(
                "s1".to_string(),
                "Test".to_string(),
                Emotion::Happy,
                EmotionIntensity::MEDIUM,
                params,
            )
            .unwrap();

        let exported = context.export_history().unwrap();
        assert!(!exported.is_empty());

        let mut new_context = ConversationContext::new();
        new_context.import_history(&exported).unwrap();
        assert_eq!(new_context.history.len(), 1);
    }

    #[test]
    fn test_clear_history() {
        let mut context = ConversationContext::new();
        let params = EmotionParameters::neutral();

        context
            .add_turn(
                "s1".to_string(),
                "Test".to_string(),
                Emotion::Happy,
                EmotionIntensity::MEDIUM,
                params,
            )
            .unwrap();
        assert_eq!(context.history.len(), 1);

        context.clear_history();
        assert_eq!(context.history.len(), 0);
        assert_eq!(context.metrics.total_turns, 0);
    }

    #[test]
    fn test_communication_style_adaptations() {
        let mut context = ConversationContext::new();

        let mut reasoning = Vec::new();
        let mut confidence = 0.5;
        let mut emotion = Emotion::Excited;
        let mut intensity = EmotionIntensity::VERY_HIGH;

        context.apply_communication_style_adaptation(
            CommunicationStyle::Reserved,
            &mut emotion,
            &mut intensity,
            &mut reasoning,
            &mut confidence,
        );

        assert!(intensity.value() < EmotionIntensity::VERY_HIGH.value());
        assert!(!reasoning.is_empty());
    }

    #[test]
    fn test_topic_context_adaptations() {
        let mut context = ConversationContext::new();

        let mut reasoning = Vec::new();
        let mut confidence = 0.5;
        let mut emotion = Emotion::Excited;
        let mut intensity = EmotionIntensity::VERY_HIGH;

        context.apply_topic_context_adaptation(
            TopicContext::Formal,
            &mut emotion,
            &mut intensity,
            &mut reasoning,
            &mut confidence,
        );

        assert!(intensity.value() <= 0.7);
        assert!(!reasoning.is_empty());
    }

    #[test]
    fn test_relationship_adaptations() {
        let mut context = ConversationContext::new();

        let mut reasoning = Vec::new();
        let mut confidence = 0.5;
        let mut emotion = Emotion::Happy;
        let mut intensity = EmotionIntensity::VERY_HIGH;

        context.apply_relationship_adaptation(
            SpeakerRelationship::Superior,
            "speaker1",
            &mut emotion,
            &mut intensity,
            &mut reasoning,
            &mut confidence,
        );

        assert!(intensity.value() <= 0.6);
        assert!(!reasoning.is_empty());
    }
}
