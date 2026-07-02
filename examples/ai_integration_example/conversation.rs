//! Conversation and dialogue state management.
//!
//! This module hosts the [`ConversationManager`], which tracks active
//! conversations, their contextual memory, and per-user profiles, along with
//! the supporting message/context/preference data structures.

use crate::personality::{EmotionAnalysis, EmotionalState, PersonalityProfile};
use crate::voice_synthesis::{VoiceMetadata, VoiceSettings};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Conversational AI context and state management
#[allow(dead_code)]
pub struct ConversationManager {
    /// Active conversations
    pub(crate) active_conversations: Arc<RwLock<HashMap<Uuid, Conversation>>>,
    /// Context memory for each conversation
    pub(crate) context_memory: Arc<RwLock<HashMap<Uuid, ConversationContext>>>,
    /// User profiles and preferences
    pub(crate) user_profiles: Arc<RwLock<HashMap<String, UserProfile>>>,
}

#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: Uuid,
    pub user_id: String,
    pub messages: Vec<ConversationMessage>,
    pub created_at: SystemTime,
    pub last_activity: SystemTime,
    pub personality_profile: PersonalityProfile,
    pub voice_settings: VoiceSettings,
}

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub id: Uuid,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: SystemTime,
    pub emotion_analysis: EmotionAnalysis,
    pub voice_metadata: Option<VoiceMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub conversation_id: Uuid,
    pub topic_history: Vec<String>,
    pub emotional_state: EmotionalState,
    pub user_preferences: UserPreferences,
    pub context_memory: Vec<ContextMemoryItem>,
}

#[derive(Debug, Clone)]
pub struct UserProfile {
    pub user_id: String,
    pub name: String,
    pub preferred_voice: String,
    pub personality_preferences: PersonalityPreferences,
    pub interaction_history: InteractionHistory,
    pub accessibility_settings: AccessibilitySettings,
}

#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub preferred_topics: Vec<String>,
    pub conversation_style: String,
    pub response_length: ResponseLength,
    pub formality_level: f32,
}

#[derive(Debug, Clone)]
pub struct ContextMemoryItem {
    pub key: String,
    pub value: String,
    pub importance: f32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct PersonalityPreferences {
    pub preferred_traits: HashMap<String, f32>,
    pub interaction_style: String,
    pub humor_level: f32,
    pub formality: f32,
}

#[derive(Debug, Clone)]
pub struct InteractionHistory {
    pub total_interactions: u64,
    pub favorite_topics: Vec<String>,
    pub interaction_patterns: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct AccessibilitySettings {
    pub speech_rate: f32,
    pub pause_between_sentences: Duration,
    pub pronunciation_help: bool,
    pub context_explanations: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseLength {
    Short,
    Medium,
    Long,
    Adaptive,
}

impl ConversationManager {
    pub(crate) fn new() -> Self {
        Self {
            active_conversations: Arc::new(RwLock::new(HashMap::new())),
            context_memory: Arc::new(RwLock::new(HashMap::new())),
            user_profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}
