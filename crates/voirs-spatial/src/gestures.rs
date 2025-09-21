//! Gesture Control System for Spatial Audio
//!
//! This module provides hand and body gesture-based audio interaction capabilities
//! for spatial audio processing, supporting both VR/AR controller input and
//! computer vision-based hand tracking.

use crate::{Error, Position3D, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Supported gesture recognition methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GestureRecognitionMethod {
    /// VR/AR controller-based gesture recognition
    Controller,
    /// Computer vision-based hand tracking
    HandTracking,
    /// IMU-based body gesture recognition
    BodyTracking,
    /// Hybrid approach combining multiple methods
    Hybrid,
}

/// Types of gestures that can be recognized
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GestureType {
    // Hand Gestures
    /// Point gesture for audio source selection
    Point,
    /// Grab gesture for audio source manipulation
    Grab,
    /// Pinch gesture for precise control
    Pinch,
    /// Open palm for area selection
    Palm,
    /// Swipe gesture for navigation
    Swipe,
    /// Rotation gesture for 3D manipulation
    Rotate,
    /// Scale gesture for distance/volume control
    Scale,

    // Body Gestures
    /// Head tilt for spatial orientation
    HeadTilt,
    /// Shoulder shrug for attention
    ShoulderShrug,
    /// Lean forward/backward for engagement
    Lean,
    /// Turn body for spatial navigation
    BodyTurn,

    // Combined Gestures
    /// Two-handed manipulation
    TwoHanded,
    /// Full body spatial positioning
    FullBody,
}

/// Direction information for directional gestures
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GestureDirection {
    /// Leftward direction
    Left,
    /// Rightward direction
    Right,
    /// Upward direction
    Up,
    /// Downward direction
    Down,
    /// Forward direction
    Forward,
    /// Backward direction
    Backward,
    /// Clockwise rotation
    Clockwise,
    /// Counterclockwise rotation
    Counterclockwise,
}

/// Hand information for hand-based gestures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Hand {
    /// Left hand
    Left,
    /// Right hand
    Right,
    /// Both hands
    Both,
}

/// Confidence level for gesture recognition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GestureConfidence {
    /// Overall confidence score (0.0 to 1.0)
    pub score: f32,
    /// Confidence in position accuracy
    pub position_confidence: f32,
    /// Confidence in gesture type recognition
    pub type_confidence: f32,
    /// Temporal stability confidence
    pub temporal_confidence: f32,
}

/// Gesture data containing position and metadata
#[derive(Debug, Clone, PartialEq)]
pub struct GestureData {
    /// Type of gesture
    pub gesture_type: GestureType,
    /// Timestamp when gesture was detected
    pub timestamp: Instant,
    /// 3D position of the gesture
    pub position: Position3D,
    /// Direction (if applicable)
    pub direction: Option<GestureDirection>,
    /// Hand information (if applicable)
    pub hand: Option<Hand>,
    /// Gesture velocity (for dynamic gestures)
    pub velocity: Option<Position3D>,
    /// Confidence in gesture recognition
    pub confidence: GestureConfidence,
    /// Additional gesture-specific parameters
    pub parameters: HashMap<String, f32>,
}

/// Gesture event representing a complete gesture action
#[derive(Debug, Clone, PartialEq)]
pub struct GestureEvent {
    /// Gesture data
    pub data: GestureData,
    /// Duration of the gesture
    pub duration: Duration,
    /// Whether this is a start, update, or end event
    pub event_type: GestureEventType,
}

/// Type of gesture event
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GestureEventType {
    /// Gesture has started
    Start,
    /// Gesture is continuing (update)
    Update,
    /// Gesture has ended
    End,
    /// Single-shot gesture
    Trigger,
}

/// Audio action to be performed based on gesture
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudioAction {
    /// Select an audio source
    SelectSource {
        /// ID of the audio source to select
        source_id: String,
    },
    /// Move an audio source to a position
    MoveSource {
        /// ID of the audio source to move
        source_id: String,
        /// New position for the audio source
        position: Position3D,
    },
    /// Adjust volume
    AdjustVolume {
        /// Optional source ID (None for global volume)
        source_id: Option<String>,
        /// New volume level (0.0-1.0)
        volume: f32,
    },
    /// Adjust spatial parameters
    AdjustSpatial {
        /// ID of the audio source
        source_id: String,
        /// Spatial parameters to adjust (parameter name -> value)
        parameters: HashMap<String, f32>,
    },
    /// Create spatial zone
    CreateZone {
        /// Center position of the zone
        position: Position3D,
        /// Radius of the zone
        radius: f32,
    },
    /// Navigate in 3D space
    Navigate {
        /// Direction of navigation
        direction: GestureDirection,
        /// Speed of navigation
        speed: f32,
    },
    /// Toggle audio effects
    ToggleEffect {
        /// Type of effect to toggle
        effect_type: String,
        /// Whether to enable or disable the effect
        enabled: bool,
    },
}

/// Configuration for gesture recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureConfig {
    /// Recognition method to use
    pub recognition_method: GestureRecognitionMethod,
    /// Minimum confidence threshold for gesture recognition
    pub min_confidence: f32,
    /// Enabled gesture types
    pub enabled_gestures: Vec<GestureType>,
    /// Gesture sensitivity (0.1 = low, 1.0 = high)
    pub sensitivity: f32,
    /// Smoothing factor for gesture tracking (0.0 = no smoothing, 1.0 = maximum)
    pub smoothing: f32,
    /// Maximum distance for gesture recognition (in meters)
    pub max_distance: f32,
    /// Minimum gesture duration to be considered valid
    pub min_duration: Duration,
    /// Maximum time between gesture updates before considering it ended
    pub max_update_interval: Duration,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            recognition_method: GestureRecognitionMethod::Controller,
            min_confidence: 0.7,
            enabled_gestures: vec![
                GestureType::Point,
                GestureType::Grab,
                GestureType::Pinch,
                GestureType::Swipe,
                GestureType::Rotate,
                GestureType::Scale,
            ],
            sensitivity: 0.8,
            smoothing: 0.3,
            max_distance: 5.0,
            min_duration: Duration::from_millis(100),
            max_update_interval: Duration::from_millis(50),
        }
    }
}

/// Gesture recognition and processing system
pub struct GestureController {
    /// Configuration
    config: GestureConfig,
    /// Active gestures being tracked
    active_gestures: HashMap<u32, GestureData>,
    /// Gesture history for temporal analysis
    gesture_history: Vec<GestureEvent>,
    /// Next gesture ID
    next_id: u32,
    /// Gesture-to-action mappings
    action_mappings: HashMap<GestureType, Vec<AudioAction>>,
    /// Smoothing buffers for gesture positions
    position_buffers: HashMap<u32, Vec<Position3D>>,
}

impl GestureController {
    /// Create a new gesture controller
    pub fn new(config: GestureConfig) -> Self {
        Self {
            config,
            active_gestures: HashMap::new(),
            gesture_history: Vec::new(),
            next_id: 1,
            action_mappings: HashMap::new(),
            position_buffers: HashMap::new(),
        }
    }

    /// Create a gesture controller with default configuration
    pub fn with_default_config() -> Self {
        Self::new(GestureConfig::default())
    }

    /// Add a gesture-to-action mapping
    pub fn add_action_mapping(&mut self, gesture_type: GestureType, action: AudioAction) {
        self.action_mappings
            .entry(gesture_type)
            .or_default()
            .push(action);
    }

    /// Process incoming gesture data
    pub fn process_gesture_data(&mut self, raw_data: GestureData) -> Result<Vec<GestureEvent>> {
        // Check if gesture meets minimum confidence threshold
        if raw_data.confidence.score < self.config.min_confidence {
            return Ok(Vec::new());
        }

        // Check if gesture type is enabled
        if !self
            .config
            .enabled_gestures
            .contains(&raw_data.gesture_type)
        {
            return Ok(Vec::new());
        }

        // Apply smoothing to position data
        let smoothed_data = self.apply_smoothing(raw_data)?;

        // Determine if this is a new gesture or update to existing one
        let gesture_id = self.find_or_create_gesture_id(&smoothed_data);

        // Generate appropriate events
        let events = self.generate_gesture_events(gesture_id, smoothed_data)?;

        // Store in history
        for event in &events {
            self.gesture_history.push(event.clone());

            // Limit history size
            if self.gesture_history.len() > 1000 {
                self.gesture_history.remove(0);
            }
        }

        Ok(events)
    }

    /// Apply smoothing to gesture data
    fn apply_smoothing(&mut self, mut data: GestureData) -> Result<GestureData> {
        if self.config.smoothing > 0.0 {
            let gesture_key = self.gesture_type_to_id(&data.gesture_type);
            let buffer = self.position_buffers.entry(gesture_key).or_default();

            buffer.push(data.position);

            // Limit buffer size
            if buffer.len() > 10 {
                buffer.remove(0);
            }

            // Apply exponential moving average
            if buffer.len() > 1 {
                let alpha = 1.0 - self.config.smoothing;
                let prev_pos = &buffer[buffer.len() - 2];

                data.position.x = alpha * data.position.x + (1.0 - alpha) * prev_pos.x;
                data.position.y = alpha * data.position.y + (1.0 - alpha) * prev_pos.y;
                data.position.z = alpha * data.position.z + (1.0 - alpha) * prev_pos.z;
            }
        }

        Ok(data)
    }

    /// Find existing gesture ID or create new one
    fn find_or_create_gesture_id(&mut self, data: &GestureData) -> u32 {
        // Simple approach: look for active gesture of same type within reasonable distance/time
        for (&id, existing) in &self.active_gestures {
            if existing.gesture_type == data.gesture_type {
                let distance = existing.position.distance_to(&data.position);
                let time_diff = data.timestamp.duration_since(existing.timestamp);

                if distance < 0.2 && time_diff < self.config.max_update_interval {
                    return id;
                }
            }
        }

        // Create new gesture ID
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Generate gesture events for a gesture data point
    fn generate_gesture_events(
        &mut self,
        gesture_id: u32,
        data: GestureData,
    ) -> Result<Vec<GestureEvent>> {
        let mut events = Vec::new();

        if let Some(existing) = self.active_gestures.get(&gesture_id) {
            // This is an update to existing gesture
            let duration = data.timestamp.duration_since(existing.timestamp);

            events.push(GestureEvent {
                data: data.clone(),
                duration,
                event_type: GestureEventType::Update,
            });
        } else {
            // This is a new gesture
            events.push(GestureEvent {
                data: data.clone(),
                duration: Duration::from_millis(0),
                event_type: GestureEventType::Start,
            });
        }

        // Update active gestures
        self.active_gestures.insert(gesture_id, data);

        Ok(events)
    }

    /// Process gesture timeout and generate end events
    pub fn process_timeouts(&mut self) -> Vec<GestureEvent> {
        let now = Instant::now();
        let mut ended_gestures = Vec::new();
        let mut events = Vec::new();

        for (&id, data) in &self.active_gestures {
            let time_since_update = now.duration_since(data.timestamp);
            if time_since_update > self.config.max_update_interval {
                ended_gestures.push(id);

                events.push(GestureEvent {
                    data: data.clone(),
                    duration: time_since_update,
                    event_type: GestureEventType::End,
                });
            }
        }

        // Remove ended gestures
        for id in ended_gestures {
            self.active_gestures.remove(&id);
        }

        events
    }

    /// Get actions for a gesture event
    pub fn get_actions_for_gesture(&self, event: &GestureEvent) -> Vec<AudioAction> {
        self.action_mappings
            .get(&event.data.gesture_type)
            .cloned()
            .unwrap_or_default()
    }

    /// Get active gestures
    pub fn get_active_gestures(&self) -> &HashMap<u32, GestureData> {
        &self.active_gestures
    }

    /// Get gesture history
    pub fn get_gesture_history(&self) -> &[GestureEvent] {
        &self.gesture_history
    }

    /// Clear gesture history
    pub fn clear_history(&mut self) {
        self.gesture_history.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: GestureConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &GestureConfig {
        &self.config
    }

    /// Helper function to convert gesture type to numeric ID for buffering
    fn gesture_type_to_id(&self, gesture_type: &GestureType) -> u32 {
        match gesture_type {
            GestureType::Point => 1,
            GestureType::Grab => 2,
            GestureType::Pinch => 3,
            GestureType::Palm => 4,
            GestureType::Swipe => 5,
            GestureType::Rotate => 6,
            GestureType::Scale => 7,
            GestureType::HeadTilt => 8,
            GestureType::ShoulderShrug => 9,
            GestureType::Lean => 10,
            GestureType::BodyTurn => 11,
            GestureType::TwoHanded => 12,
            GestureType::FullBody => 13,
        }
    }
}

/// Gesture builder for easy construction of gesture data
pub struct GestureBuilder {
    gesture_type: GestureType,
    position: Position3D,
    direction: Option<GestureDirection>,
    hand: Option<Hand>,
    velocity: Option<Position3D>,
    confidence: Option<GestureConfidence>,
    parameters: HashMap<String, f32>,
}

impl GestureBuilder {
    /// Create a new gesture builder
    pub fn new(gesture_type: GestureType, position: Position3D) -> Self {
        Self {
            gesture_type,
            position,
            direction: None,
            hand: None,
            velocity: None,
            confidence: None,
            parameters: HashMap::new(),
        }
    }

    /// Set gesture direction
    pub fn direction(mut self, direction: GestureDirection) -> Self {
        self.direction = Some(direction);
        self
    }

    /// Set hand information
    pub fn hand(mut self, hand: Hand) -> Self {
        self.hand = Some(hand);
        self
    }

    /// Set gesture velocity
    pub fn velocity(mut self, velocity: Position3D) -> Self {
        self.velocity = Some(velocity);
        self
    }

    /// Set gesture confidence
    pub fn confidence(mut self, confidence: GestureConfidence) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Add a parameter
    pub fn parameter(mut self, key: String, value: f32) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Build the gesture data
    pub fn build(self) -> GestureData {
        GestureData {
            gesture_type: self.gesture_type,
            timestamp: Instant::now(),
            position: self.position,
            direction: self.direction,
            hand: self.hand,
            velocity: self.velocity,
            confidence: self.confidence.unwrap_or(GestureConfidence {
                score: 0.8,
                position_confidence: 0.8,
                type_confidence: 0.8,
                temporal_confidence: 0.8,
            }),
            parameters: self.parameters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gesture_controller_creation() {
        let controller = GestureController::with_default_config();
        assert_eq!(controller.config.min_confidence, 0.7);
        assert!(controller
            .config
            .enabled_gestures
            .contains(&GestureType::Point));
    }

    #[test]
    fn test_gesture_builder() {
        let gesture = GestureBuilder::new(GestureType::Point, Position3D::new(1.0, 2.0, 3.0))
            .direction(GestureDirection::Forward)
            .hand(Hand::Right)
            .parameter("intensity".to_string(), 0.8)
            .build();

        assert_eq!(gesture.gesture_type, GestureType::Point);
        assert_eq!(gesture.position, Position3D::new(1.0, 2.0, 3.0));
        assert_eq!(gesture.direction, Some(GestureDirection::Forward));
        assert_eq!(gesture.hand, Some(Hand::Right));
        assert_eq!(gesture.parameters.get("intensity"), Some(&0.8));
    }

    #[test]
    fn test_action_mapping() {
        let mut controller = GestureController::with_default_config();
        let action = AudioAction::SelectSource {
            source_id: "test_source".to_string(),
        };

        controller.add_action_mapping(GestureType::Point, action.clone());

        let mappings = controller.action_mappings.get(&GestureType::Point).unwrap();
        assert_eq!(mappings.len(), 1);
    }

    #[test]
    fn test_gesture_confidence() {
        let confidence = GestureConfidence {
            score: 0.9,
            position_confidence: 0.85,
            type_confidence: 0.95,
            temporal_confidence: 0.8,
        };

        assert!(confidence.score > 0.8);
        assert!(confidence.type_confidence > confidence.position_confidence);
    }

    #[test]
    fn test_gesture_processing() {
        let mut controller = GestureController::with_default_config();

        let gesture_data =
            GestureBuilder::new(GestureType::Point, Position3D::new(0.0, 0.0, 1.0)).build();

        let events = controller.process_gesture_data(gesture_data).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, GestureEventType::Start);
        assert_eq!(controller.active_gestures.len(), 1);
    }

    #[test]
    fn test_gesture_timeout() {
        let mut controller = GestureController::with_default_config();
        controller.config.max_update_interval = Duration::from_millis(1);

        let gesture_data =
            GestureBuilder::new(GestureType::Grab, Position3D::new(1.0, 1.0, 1.0)).build();

        // Process initial gesture
        controller.process_gesture_data(gesture_data).unwrap();
        assert_eq!(controller.active_gestures.len(), 1);

        // Wait and process timeouts
        std::thread::sleep(Duration::from_millis(2));
        let timeout_events = controller.process_timeouts();

        assert_eq!(timeout_events.len(), 1);
        assert_eq!(timeout_events[0].event_type, GestureEventType::End);
        assert_eq!(controller.active_gestures.len(), 0);
    }

    #[test]
    fn test_low_confidence_filtering() {
        let mut controller = GestureController::with_default_config();
        controller.config.min_confidence = 0.8;

        let low_confidence_gesture = GestureData {
            gesture_type: GestureType::Point,
            timestamp: Instant::now(),
            position: Position3D::new(0.0, 0.0, 0.0),
            direction: None,
            hand: None,
            velocity: None,
            confidence: GestureConfidence {
                score: 0.5, // Below threshold
                position_confidence: 0.5,
                type_confidence: 0.5,
                temporal_confidence: 0.5,
            },
            parameters: HashMap::new(),
        };

        let events = controller
            .process_gesture_data(low_confidence_gesture)
            .unwrap();
        assert_eq!(events.len(), 0); // Should be filtered out
    }
}
