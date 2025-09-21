//! Multi-modal feedback delivery

use super::types::*;
use crate::{
    FeedbackError, FeedbackResponse, FeedbackType, ProgressIndicators, UserFeedback,
    UserPreferences,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Multi-modal feedback coordinator
#[derive(Debug, Clone)]
pub struct MultiModalFeedbackManager {
    config: MultiModalConfig,
    active_modalities: Arc<RwLock<HashMap<ModalityType, ModalityState>>>,
    feedback_history: Arc<RwLock<Vec<MultiModalFeedbackEvent>>>,
}

/// Configuration for multi-modal feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    pub visual_enabled: bool,
    pub audio_enabled: bool,
    pub haptic_enabled: bool,
    pub textual_enabled: bool,
    pub gesture_enabled: bool,
    pub synchronization_tolerance_ms: u64,
    pub max_concurrent_modalities: usize,
    pub intensity_levels: IntensityLevels,
}

/// Intensity levels for different feedback types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntensityLevels {
    pub visual: f32,  // 0.0 - 1.0
    pub audio: f32,   // 0.0 - 1.0
    pub haptic: f32,  // 0.0 - 1.0
    pub textual: f32, // 0.0 - 1.0 (verbosity)
    pub gesture: f32, // 0.0 - 1.0 (animation speed)
}

/// Types of feedback modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    Visual,
    Audio,
    Haptic,
    Textual,
    Gesture,
}

/// State of a feedback modality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityState {
    pub active: bool,
    pub intensity: f32,
    pub last_activation: Option<DateTime<Utc>>,
    pub activation_count: u64,
    pub error_count: u64,
}

/// Multi-modal feedback event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalFeedbackEvent {
    pub timestamp: DateTime<Utc>,
    pub modalities: Vec<ModalityActivation>,
    pub coordination_score: f32,
    pub user_response: Option<UserResponse>,
    pub effectiveness_score: Option<f32>,
}

/// Individual modality activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityActivation {
    pub modality: ModalityType,
    pub content: ModalityContent,
    pub timing: ModalityTiming,
    pub intensity: f32,
    pub duration_ms: u64,
}

/// Content for different modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityContent {
    Visual(VisualContent),
    Audio(AudioContent),
    Haptic(HapticContent),
    Textual(TextualContent),
    Gesture(GestureContent),
}

/// Visual feedback content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualContent {
    pub display_type: VisualDisplayType,
    pub colors: Vec<String>,
    pub shapes: Vec<VisualShape>,
    pub animations: Vec<Animation>,
    pub position: Position2D,
    pub size: Size2D,
}

/// Types of visual displays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualDisplayType {
    Indicator,
    Progress,
    Waveform,
    Spectrogram,
    Avatar,
    Text,
    Icon,
}

/// Visual shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualShape {
    pub shape_type: ShapeType,
    pub color: String,
    pub size: f32,
    pub opacity: f32,
}

/// Shape types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShapeType {
    Circle,
    Rectangle,
    Triangle,
    Line,
    Arrow,
    Star,
}

/// Animation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Animation {
    pub animation_type: AnimationType,
    pub duration_ms: u64,
    pub easing: EasingType,
    pub repeat: bool,
}

/// Animation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationType {
    FadeIn,
    FadeOut,
    SlideIn,
    SlideOut,
    Bounce,
    Pulse,
    Rotate,
    Scale,
}

/// Easing functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingType {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
}

/// Audio feedback content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContent {
    pub audio_type: AudioFeedbackType,
    pub frequency: Option<f32>,
    pub volume: f32,
    pub spatialization: Option<Position3D>,
    pub effects: Vec<AudioEffect>,
}

/// Types of audio feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioFeedbackType {
    Tone,
    Chime,
    Beep,
    Voice,
    Music,
    WhiteNoise,
    PinkNoise,
}

/// Audio effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEffect {
    pub effect_type: AudioEffectType,
    pub intensity: f32,
}

/// Audio effect types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioEffectType {
    Reverb,
    Echo,
    Distortion,
    Filter,
    Pitch,
    Speed,
}

/// Haptic feedback content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticContent {
    pub pattern: HapticPattern,
    pub intensity: f32,
    pub frequency: f32,
    pub location: HapticLocation,
}

/// Haptic patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HapticPattern {
    Single,
    Double,
    Triple,
    Continuous,
    Pulse,
    Ramp,
    Wave,
}

/// Haptic feedback locations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HapticLocation {
    Left,
    Right,
    Center,
    All,
    Custom(Position2D),
}

/// Textual feedback content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextualContent {
    pub message: String,
    pub urgency: UrgencyLevel,
    pub formatting: TextFormatting,
    pub display_duration_ms: u64,
}

/// Text formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFormatting {
    pub font_size: f32,
    pub color: String,
    pub background_color: Option<String>,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

/// Gesture feedback content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureContent {
    pub gesture_type: GestureType,
    pub direction: Direction,
    pub speed: f32,
    pub amplitude: f32,
}

/// Gesture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GestureType {
    Point,
    Wave,
    Nod,
    Shake,
    Circle,
    Swipe,
}

/// Directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    Forward,
    Backward,
    Clockwise,
    Counterclockwise,
}

/// Timing information for modality activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityTiming {
    pub start_offset_ms: u64,
    pub duration_ms: u64,
    pub synchronization_priority: SyncPriority,
}

/// Synchronization priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncPriority {
    Primary,
    Secondary,
    Background,
}

/// Position in 2D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position2D {
    pub x: f32,
    pub y: f32,
}

/// Position in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Size in 2D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Size2D {
    pub width: f32,
    pub height: f32,
}

/// Urgency levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// User response to feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserResponse {
    pub modality: ModalityType,
    pub response_time_ms: u64,
    pub accuracy: f32,
    pub satisfaction: f32,
}

impl MultiModalFeedbackManager {
    /// Create a new multi-modal feedback manager
    pub fn new(config: MultiModalConfig) -> Self {
        let mut active_modalities = HashMap::new();

        for modality in [
            ModalityType::Visual,
            ModalityType::Audio,
            ModalityType::Haptic,
            ModalityType::Textual,
            ModalityType::Gesture,
        ] {
            active_modalities.insert(
                modality,
                ModalityState {
                    active: false,
                    intensity: 0.0,
                    last_activation: None,
                    activation_count: 0,
                    error_count: 0,
                },
            );
        }

        Self {
            config,
            active_modalities: Arc::new(RwLock::new(active_modalities)),
            feedback_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Deliver coordinated multi-modal feedback
    pub async fn deliver_multimodal_feedback(
        &self,
        feedback: &FeedbackResponse,
        preferences: &UserPreferences,
    ) -> Result<MultiModalFeedbackEvent, FeedbackError> {
        let start_time = Utc::now();
        let mut activations = Vec::new();

        // Generate appropriate modality activations based on feedback and preferences
        if self.config.visual_enabled && preferences.enable_visual_feedback {
            if let Some(activation) = self.create_visual_activation(feedback).await? {
                activations.push(activation);
            }
        }

        if self.config.audio_enabled && preferences.enable_audio_feedback {
            if let Some(activation) = self.create_audio_activation(feedback).await? {
                activations.push(activation);
            }
        }

        if self.config.haptic_enabled && preferences.enable_haptic_feedback {
            if let Some(activation) = self.create_haptic_activation(feedback).await? {
                activations.push(activation);
            }
        }

        if self.config.textual_enabled && preferences.enable_visual_feedback {
            if let Some(activation) = self.create_textual_activation(feedback).await? {
                activations.push(activation);
            }
        }

        if self.config.gesture_enabled {
            if let Some(activation) = self.create_gesture_activation(feedback).await? {
                activations.push(activation);
            }
        }

        // Synchronize activations
        self.synchronize_activations(&mut activations).await?;

        // Calculate coordination score
        let coordination_score = self.calculate_coordination_score(&activations);

        // Update modality states
        self.update_modality_states(&activations).await?;

        let event = MultiModalFeedbackEvent {
            timestamp: start_time,
            modalities: activations,
            coordination_score,
            user_response: None,
            effectiveness_score: None,
        };

        // Store in history
        let mut history = self.feedback_history.write().await;
        history.push(event.clone());

        // Keep only recent history (last 1000 events)
        if history.len() > 1000 {
            let excess = history.len() - 1000;
            history.drain(0..excess);
        }

        Ok(event)
    }

    /// Create visual feedback activation
    async fn create_visual_activation(
        &self,
        feedback: &FeedbackResponse,
    ) -> Result<Option<ModalityActivation>, FeedbackError> {
        let content = VisualContent {
            display_type: match feedback.feedback_type {
                FeedbackType::Success => VisualDisplayType::Icon,
                FeedbackType::Error => VisualDisplayType::Indicator,
                FeedbackType::Warning => VisualDisplayType::Progress,
                FeedbackType::Info => VisualDisplayType::Text,
                FeedbackType::Quality => VisualDisplayType::Spectrogram,
                FeedbackType::Pronunciation => VisualDisplayType::Waveform,
                FeedbackType::Naturalness => VisualDisplayType::Avatar,
                FeedbackType::Technical => VisualDisplayType::Text,
                FeedbackType::Motivational => VisualDisplayType::Icon,
                FeedbackType::Comparative => VisualDisplayType::Progress,
                FeedbackType::Adaptive => VisualDisplayType::Progress,
            },
            colors: vec![match feedback.feedback_type {
                FeedbackType::Success => "#00FF00".to_string(),
                FeedbackType::Error => "#FF0000".to_string(),
                FeedbackType::Warning => "#FFAA00".to_string(),
                FeedbackType::Info => "#0088FF".to_string(),
                FeedbackType::Quality => "#8A2BE2".to_string(),
                FeedbackType::Pronunciation => "#FF69B4".to_string(),
                FeedbackType::Naturalness => "#32CD32".to_string(),
                FeedbackType::Technical => "#708090".to_string(),
                FeedbackType::Motivational => "#FFD700".to_string(),
                FeedbackType::Comparative => "#4169E1".to_string(),
                FeedbackType::Adaptive => "#00BFFF".to_string(),
            }],
            shapes: vec![VisualShape {
                shape_type: ShapeType::Circle,
                color: "#FFFFFF".to_string(),
                size: 1.0,
                opacity: 0.8,
            }],
            animations: vec![Animation {
                animation_type: match feedback.feedback_type {
                    FeedbackType::Success => AnimationType::Bounce,
                    FeedbackType::Error => AnimationType::Pulse,
                    FeedbackType::Warning => AnimationType::FadeIn,
                    FeedbackType::Info => AnimationType::SlideIn,
                    FeedbackType::Quality => AnimationType::Scale,
                    FeedbackType::Pronunciation => AnimationType::Rotate,
                    FeedbackType::Naturalness => AnimationType::FadeIn,
                    FeedbackType::Technical => AnimationType::SlideIn,
                    FeedbackType::Motivational => AnimationType::Bounce,
                    FeedbackType::Comparative => AnimationType::SlideOut,
                    FeedbackType::Adaptive => AnimationType::FadeIn,
                },
                duration_ms: 500,
                easing: EasingType::EaseInOut,
                repeat: false,
            }],
            position: Position2D { x: 0.5, y: 0.5 },
            size: Size2D {
                width: 100.0,
                height: 100.0,
            },
        };

        Ok(Some(ModalityActivation {
            modality: ModalityType::Visual,
            content: ModalityContent::Visual(content),
            timing: ModalityTiming {
                start_offset_ms: 0,
                duration_ms: 1000,
                synchronization_priority: SyncPriority::Primary,
            },
            intensity: self.config.intensity_levels.visual,
            duration_ms: 1000,
        }))
    }

    /// Create audio feedback activation
    async fn create_audio_activation(
        &self,
        feedback: &FeedbackResponse,
    ) -> Result<Option<ModalityActivation>, FeedbackError> {
        let content = AudioContent {
            audio_type: match feedback.feedback_type {
                FeedbackType::Success => AudioFeedbackType::Chime,
                FeedbackType::Error => AudioFeedbackType::Beep,
                FeedbackType::Warning => AudioFeedbackType::Tone,
                FeedbackType::Info => AudioFeedbackType::Voice,
                FeedbackType::Quality => AudioFeedbackType::Tone,
                FeedbackType::Pronunciation => AudioFeedbackType::Voice,
                FeedbackType::Naturalness => AudioFeedbackType::Music,
                FeedbackType::Technical => AudioFeedbackType::Beep,
                FeedbackType::Motivational => AudioFeedbackType::Chime,
                FeedbackType::Comparative => AudioFeedbackType::Tone,
                FeedbackType::Adaptive => AudioFeedbackType::Tone,
            },
            frequency: Some(match feedback.feedback_type {
                FeedbackType::Success => 800.0,
                FeedbackType::Error => 300.0,
                FeedbackType::Warning => 500.0,
                FeedbackType::Info => 600.0,
                FeedbackType::Quality => 650.0,
                FeedbackType::Pronunciation => 700.0,
                FeedbackType::Naturalness => 750.0,
                FeedbackType::Technical => 400.0,
                FeedbackType::Motivational => 850.0,
                FeedbackType::Comparative => 550.0,
                FeedbackType::Adaptive => 500.0,
            }),
            volume: self.config.intensity_levels.audio,
            spatialization: None,
            effects: vec![],
        };

        Ok(Some(ModalityActivation {
            modality: ModalityType::Audio,
            content: ModalityContent::Audio(content),
            timing: ModalityTiming {
                start_offset_ms: 50, // Slight delay after visual
                duration_ms: 500,
                synchronization_priority: SyncPriority::Secondary,
            },
            intensity: self.config.intensity_levels.audio,
            duration_ms: 500,
        }))
    }

    /// Create haptic feedback activation
    async fn create_haptic_activation(
        &self,
        feedback: &FeedbackResponse,
    ) -> Result<Option<ModalityActivation>, FeedbackError> {
        let content = HapticContent {
            pattern: match feedback.feedback_type {
                FeedbackType::Success => HapticPattern::Double,
                FeedbackType::Error => HapticPattern::Continuous,
                FeedbackType::Warning => HapticPattern::Pulse,
                FeedbackType::Info => HapticPattern::Single,
                FeedbackType::Quality => HapticPattern::Wave,
                FeedbackType::Pronunciation => HapticPattern::Ramp,
                FeedbackType::Naturalness => HapticPattern::Single,
                FeedbackType::Technical => HapticPattern::Triple,
                FeedbackType::Motivational => HapticPattern::Double,
                FeedbackType::Comparative => HapticPattern::Pulse,
                FeedbackType::Adaptive => HapticPattern::Pulse,
            },
            intensity: self.config.intensity_levels.haptic,
            frequency: match feedback.feedback_type {
                FeedbackType::Success => 200.0,
                FeedbackType::Error => 100.0,
                FeedbackType::Warning => 150.0,
                FeedbackType::Info => 250.0,
                FeedbackType::Quality => 175.0,
                FeedbackType::Pronunciation => 225.0,
                FeedbackType::Naturalness => 275.0,
                FeedbackType::Technical => 125.0,
                FeedbackType::Motivational => 300.0,
                FeedbackType::Comparative => 180.0,
                FeedbackType::Adaptive => 150.0,
            },
            location: HapticLocation::Center,
        };

        Ok(Some(ModalityActivation {
            modality: ModalityType::Haptic,
            content: ModalityContent::Haptic(content),
            timing: ModalityTiming {
                start_offset_ms: 100, // After audio
                duration_ms: 300,
                synchronization_priority: SyncPriority::Background,
            },
            intensity: self.config.intensity_levels.haptic,
            duration_ms: 300,
        }))
    }

    /// Create textual feedback activation
    async fn create_textual_activation(
        &self,
        feedback: &FeedbackResponse,
    ) -> Result<Option<ModalityActivation>, FeedbackError> {
        let content = TextualContent {
            message: feedback
                .feedback_items
                .first()
                .map(|item| item.message.clone())
                .unwrap_or_default(),
            urgency: match feedback.feedback_type {
                FeedbackType::Success => UrgencyLevel::Low,
                FeedbackType::Error => UrgencyLevel::High,
                FeedbackType::Warning => UrgencyLevel::Medium,
                FeedbackType::Info => UrgencyLevel::Low,
                FeedbackType::Quality => UrgencyLevel::Medium,
                FeedbackType::Pronunciation => UrgencyLevel::High,
                FeedbackType::Naturalness => UrgencyLevel::Low,
                FeedbackType::Technical => UrgencyLevel::Medium,
                FeedbackType::Motivational => UrgencyLevel::Low,
                FeedbackType::Comparative => UrgencyLevel::Medium,
                FeedbackType::Adaptive => UrgencyLevel::Medium,
            },
            formatting: TextFormatting {
                font_size: 14.0 * self.config.intensity_levels.textual,
                color: match feedback.feedback_type {
                    FeedbackType::Success => "#00AA00".to_string(),
                    FeedbackType::Error => "#AA0000".to_string(),
                    FeedbackType::Warning => "#AA6600".to_string(),
                    FeedbackType::Info => "#0066AA".to_string(),
                    FeedbackType::Quality => "#8A2BE2".to_string(),
                    FeedbackType::Pronunciation => "#FF69B4".to_string(),
                    FeedbackType::Naturalness => "#32CD32".to_string(),
                    FeedbackType::Technical => "#708090".to_string(),
                    FeedbackType::Motivational => "#FFD700".to_string(),
                    FeedbackType::Comparative => "#4169E1".to_string(),
                    FeedbackType::Adaptive => "#00BFFF".to_string(),
                },
                background_color: Some("#FFFFFF".to_string()),
                bold: matches!(
                    feedback.feedback_type,
                    FeedbackType::Error | FeedbackType::Warning
                ),
                italic: false,
                underline: false,
            },
            display_duration_ms: 2000,
        };

        Ok(Some(ModalityActivation {
            modality: ModalityType::Textual,
            content: ModalityContent::Textual(content),
            timing: ModalityTiming {
                start_offset_ms: 0, // Simultaneous with visual
                duration_ms: 2000,
                synchronization_priority: SyncPriority::Primary,
            },
            intensity: self.config.intensity_levels.textual,
            duration_ms: 2000,
        }))
    }

    /// Create gesture feedback activation
    async fn create_gesture_activation(
        &self,
        feedback: &FeedbackResponse,
    ) -> Result<Option<ModalityActivation>, FeedbackError> {
        let content = GestureContent {
            gesture_type: match feedback.feedback_type {
                FeedbackType::Success => GestureType::Nod,
                FeedbackType::Error => GestureType::Shake,
                FeedbackType::Warning => GestureType::Point,
                FeedbackType::Info => GestureType::Wave,
                FeedbackType::Quality => GestureType::Circle,
                FeedbackType::Pronunciation => GestureType::Point,
                FeedbackType::Naturalness => GestureType::Wave,
                FeedbackType::Technical => GestureType::Point,
                FeedbackType::Motivational => GestureType::Nod,
                FeedbackType::Comparative => GestureType::Swipe,
                FeedbackType::Adaptive => GestureType::Swipe,
            },
            direction: match feedback.feedback_type {
                FeedbackType::Success => Direction::Up,
                FeedbackType::Error => Direction::Left,
                FeedbackType::Warning => Direction::Forward,
                FeedbackType::Info => Direction::Right,
                FeedbackType::Quality => Direction::Clockwise,
                FeedbackType::Pronunciation => Direction::Forward,
                FeedbackType::Naturalness => Direction::Right,
                FeedbackType::Technical => Direction::Down,
                FeedbackType::Motivational => Direction::Up,
                FeedbackType::Comparative => Direction::Counterclockwise,
                FeedbackType::Adaptive => Direction::Clockwise,
            },
            speed: self.config.intensity_levels.gesture,
            amplitude: 1.0,
        };

        Ok(Some(ModalityActivation {
            modality: ModalityType::Gesture,
            content: ModalityContent::Gesture(content),
            timing: ModalityTiming {
                start_offset_ms: 200,
                duration_ms: 800,
                synchronization_priority: SyncPriority::Background,
            },
            intensity: self.config.intensity_levels.gesture,
            duration_ms: 800,
        }))
    }

    /// Synchronize modality activations
    async fn synchronize_activations(
        &self,
        activations: &mut [ModalityActivation],
    ) -> Result<(), FeedbackError> {
        // Sort by synchronization priority
        activations.sort_by(|a, b| {
            match (
                &a.timing.synchronization_priority,
                &b.timing.synchronization_priority,
            ) {
                (SyncPriority::Primary, SyncPriority::Primary) => std::cmp::Ordering::Equal,
                (SyncPriority::Primary, _) => std::cmp::Ordering::Less,
                (_, SyncPriority::Primary) => std::cmp::Ordering::Greater,
                (SyncPriority::Secondary, SyncPriority::Secondary) => std::cmp::Ordering::Equal,
                (SyncPriority::Secondary, _) => std::cmp::Ordering::Less,
                (_, SyncPriority::Secondary) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            }
        });

        // Adjust timing for synchronization
        let tolerance = self.config.synchronization_tolerance_ms;
        for activation in activations.iter_mut() {
            if activation.timing.start_offset_ms > tolerance {
                activation.timing.start_offset_ms = tolerance;
            }
        }

        Ok(())
    }

    /// Calculate coordination score for the activations
    fn calculate_coordination_score(&self, activations: &[ModalityActivation]) -> f32 {
        if activations.is_empty() {
            return 0.0;
        }

        let mut score: f32 = 1.0;

        // Penalty for too many simultaneous modalities
        if activations.len() > self.config.max_concurrent_modalities {
            score *= 0.8;
        }

        // Bonus for good timing synchronization
        let max_offset = activations
            .iter()
            .map(|a| a.timing.start_offset_ms)
            .max()
            .unwrap_or(0);

        if max_offset <= self.config.synchronization_tolerance_ms {
            score *= 1.2;
        }

        // Bonus for appropriate intensity distribution
        let avg_intensity =
            activations.iter().map(|a| a.intensity).sum::<f32>() / activations.len() as f32;

        if avg_intensity > 0.3 && avg_intensity < 0.8 {
            score *= 1.1;
        }

        score.min(1.0)
    }

    /// Update modality states after activation
    async fn update_modality_states(
        &self,
        activations: &[ModalityActivation],
    ) -> Result<(), FeedbackError> {
        let mut states = self.active_modalities.write().await;
        let now = Utc::now();

        for activation in activations {
            if let Some(state) = states.get_mut(&activation.modality) {
                state.active = true;
                state.intensity = activation.intensity;
                state.last_activation = Some(now);
                state.activation_count += 1;
            }
        }

        Ok(())
    }

    /// Get modality statistics
    pub async fn get_modality_statistics(&self) -> HashMap<ModalityType, ModalityState> {
        self.active_modalities.read().await.clone()
    }

    /// Get feedback history
    pub async fn get_feedback_history(&self, limit: Option<usize>) -> Vec<MultiModalFeedbackEvent> {
        let history = self.feedback_history.read().await;
        let start_index = if let Some(limit) = limit {
            history.len().saturating_sub(limit)
        } else {
            0
        };
        history[start_index..].to_vec()
    }

    /// Clear feedback history
    pub async fn clear_history(&self) -> Result<(), FeedbackError> {
        let mut history = self.feedback_history.write().await;
        history.clear();
        Ok(())
    }
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            visual_enabled: true,
            audio_enabled: true,
            haptic_enabled: true,
            textual_enabled: true,
            gesture_enabled: false,
            synchronization_tolerance_ms: 100,
            max_concurrent_modalities: 3,
            intensity_levels: IntensityLevels::default(),
        }
    }
}

impl Default for IntensityLevels {
    fn default() -> Self {
        Self {
            visual: 0.7,
            audio: 0.6,
            haptic: 0.5,
            textual: 0.8,
            gesture: 0.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeedbackType;

    #[tokio::test]
    async fn test_multimodal_manager_creation() {
        let manager = MultiModalFeedbackManager::new(MultiModalConfig::default());
        let stats = manager.get_modality_statistics().await;
        assert_eq!(stats.len(), 5);
    }

    #[tokio::test]
    async fn test_feedback_delivery() {
        let manager = MultiModalFeedbackManager::new(MultiModalConfig::default());

        let feedback = FeedbackResponse {
            feedback_type: FeedbackType::Quality,
            feedback_items: vec![UserFeedback {
                message: "Great job!".to_string(),
                suggestion: Some("Keep practicing".to_string()),
                confidence: 0.9,
                score: 0.9,
                priority: 0.8,
                metadata: std::collections::HashMap::new(),
            }],
            overall_score: 0.9,
            immediate_actions: Vec::new(),
            long_term_goals: Vec::new(),
            progress_indicators: ProgressIndicators {
                improving_areas: Vec::new(),
                attention_areas: Vec::new(),
                stable_areas: Vec::new(),
                overall_trend: 0.1,
                completion_percentage: 90.0,
            },
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(50),
        };

        let preferences = UserPreferences::default();

        let event = manager
            .deliver_multimodal_feedback(&feedback, &preferences)
            .await
            .unwrap();

        assert!(!event.modalities.is_empty());
        assert!(event.coordination_score > 0.0);

        let history = manager.get_feedback_history(Some(1)).await;
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_coordination_score_calculation() {
        let manager = MultiModalFeedbackManager::new(MultiModalConfig::default());

        let activations = vec![ModalityActivation {
            modality: ModalityType::Visual,
            content: ModalityContent::Visual(VisualContent {
                display_type: VisualDisplayType::Icon,
                colors: vec!["#00FF00".to_string()],
                shapes: vec![],
                animations: vec![],
                position: Position2D { x: 0.0, y: 0.0 },
                size: Size2D {
                    width: 100.0,
                    height: 100.0,
                },
            }),
            timing: ModalityTiming {
                start_offset_ms: 0,
                duration_ms: 1000,
                synchronization_priority: SyncPriority::Primary,
            },
            intensity: 0.5,
            duration_ms: 1000,
        }];

        let score = manager.calculate_coordination_score(&activations);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[tokio::test]
    async fn test_history_management() {
        let manager = MultiModalFeedbackManager::new(MultiModalConfig::default());

        // Add some events to history
        for i in 0..5 {
            let event = MultiModalFeedbackEvent {
                timestamp: Utc::now(),
                modalities: vec![],
                coordination_score: 0.5,
                user_response: None,
                effectiveness_score: Some(0.8),
            };

            let mut history = manager.feedback_history.write().await;
            history.push(event);
        }

        let history = manager.get_feedback_history(Some(3)).await;
        assert_eq!(history.len(), 3);

        manager.clear_history().await.unwrap();
        let history = manager.get_feedback_history(None).await;
        assert!(history.is_empty());
    }
}
