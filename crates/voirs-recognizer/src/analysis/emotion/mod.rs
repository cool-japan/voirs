//! Emotion and sentiment recognition module
//!
//! This module provides multi-dimensional emotion detection, sentiment polarity analysis,
//! stress and fatigue detection, and mood tracking over time.

use crate::RecognitionError;
use voirs_sdk::AudioBuffer;
use async_trait::async_trait;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub mod detector;
pub mod features;
pub mod models;
pub mod tracking;

pub use detector::*;
pub use features::*;
pub use models::*;
pub use tracking::*;

/// Emotion types supported by the recognition system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionType {
    /// Neutral emotional state
    Neutral,
    /// Happiness and joy
    Happy,
    /// Sadness and sorrow
    Sad,
    /// Anger and irritation
    Angry,
    /// Fear and anxiety
    Fear,
    /// Surprise and amazement
    Surprise,
    /// Disgust and aversion
    Disgust,
    /// Excitement and enthusiasm
    Excited,
    /// Calm and peaceful
    Calm,
    /// Love and affection
    Love,
    /// Stress and tension
    Stressed,
    /// Fatigue and tiredness
    Fatigued,
}

impl EmotionType {
    /// Get all supported emotion types
    pub fn all() -> Vec<EmotionType> {
        vec![
            EmotionType::Neutral,
            EmotionType::Happy,
            EmotionType::Sad,
            EmotionType::Angry,
            EmotionType::Fear,
            EmotionType::Surprise,
            EmotionType::Disgust,
            EmotionType::Excited,
            EmotionType::Calm,
            EmotionType::Love,
            EmotionType::Stressed,
            EmotionType::Fatigued,
        ]
    }

    /// Get human-readable name
    pub fn name(self) -> &'static str {
        match self {
            EmotionType::Neutral => "Neutral",
            EmotionType::Happy => "Happy",
            EmotionType::Sad => "Sad",
            EmotionType::Angry => "Angry",
            EmotionType::Fear => "Fear",
            EmotionType::Surprise => "Surprise",
            EmotionType::Disgust => "Disgust",
            EmotionType::Excited => "Excited",
            EmotionType::Calm => "Calm",
            EmotionType::Love => "Love",
            EmotionType::Stressed => "Stressed",
            EmotionType::Fatigued => "Fatigued",
        }
    }

    /// Get emotion category (primary emotions vs derived emotions)
    pub fn category(self) -> EmotionCategory {
        match self {
            EmotionType::Neutral | EmotionType::Happy | EmotionType::Sad | 
            EmotionType::Angry | EmotionType::Fear | EmotionType::Surprise | 
            EmotionType::Disgust => EmotionCategory::Primary,
            
            EmotionType::Excited | EmotionType::Calm | EmotionType::Love |
            EmotionType::Stressed | EmotionType::Fatigued => EmotionCategory::Derived,
        }
    }
}

/// Emotion categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionCategory {
    /// Primary emotions (basic human emotions)
    Primary,
    /// Derived emotions (combinations or specializations)
    Derived,
}

/// Sentiment polarity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentimentPolarity {
    /// Positive sentiment
    Positive,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment
    Neutral,
}

/// Emotion detection result
#[derive(Debug, Clone)]
pub struct EmotionDetection {
    /// Detected emotion
    pub emotion: EmotionType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Intensity level (0.0 to 1.0)
    pub intensity: f32,
    /// Detection timestamp
    pub timestamp: Instant,
    /// Audio segment start time
    pub start_time: Duration,
    /// Audio segment end time
    pub end_time: Duration,
    /// Raw emotion scores for all emotions
    pub emotion_scores: HashMap<EmotionType, f32>,
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct SentimentAnalysis {
    /// Sentiment polarity
    pub polarity: SentimentPolarity,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Valence score (-1.0 to 1.0, negative to positive)
    pub valence: f32,
    /// Arousal score (0.0 to 1.0, low to high energy)
    pub arousal: f32,
    /// Dominance score (0.0 to 1.0, submissive to dominant)
    pub dominance: f32,
    /// Detection timestamp
    pub timestamp: Instant,
}

/// Multi-dimensional emotion analysis result
#[derive(Debug, Clone)]
pub struct MultiDimensionalEmotion {
    /// Primary emotion detection
    pub primary_emotion: EmotionDetection,
    /// Secondary emotions (if any)
    pub secondary_emotions: Vec<EmotionDetection>,
    /// Sentiment analysis
    pub sentiment: SentimentAnalysis,
    /// Stress level (0.0 to 1.0)
    pub stress_level: f32,
    /// Fatigue level (0.0 to 1.0)
    pub fatigue_level: f32,
    /// Overall emotional complexity score
    pub complexity_score: f32,
}

/// Emotion recognition configuration
#[derive(Debug, Clone)]
pub struct EmotionConfig {
    /// Minimum confidence threshold for emotion detection
    pub min_confidence: f32,
    /// Enable multi-dimensional emotion detection
    pub multi_dimensional: bool,
    /// Enable sentiment analysis
    pub sentiment_analysis: bool,
    /// Enable stress detection
    pub stress_detection: bool,
    /// Enable fatigue detection
    pub fatigue_detection: bool,
    /// Window size for analysis (seconds)
    pub analysis_window_seconds: f32,
    /// Overlap between analysis windows
    pub window_overlap: f32,
    /// Enable emotion tracking over time
    pub enable_tracking: bool,
    /// Smoothing factor for temporal consistency
    pub temporal_smoothing: f32,
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            multi_dimensional: true,
            sentiment_analysis: true,
            stress_detection: true,
            fatigue_detection: true,
            analysis_window_seconds: 3.0,
            window_overlap: 0.5,
            enable_tracking: true,
            temporal_smoothing: 0.3,
        }
    }
}

/// Emotion recognition trait
#[async_trait]
pub trait EmotionRecognizer: Send + Sync {
    /// Detect emotions in audio
    async fn detect_emotions(&mut self, audio: &AudioBuffer) -> Result<Vec<EmotionDetection>, RecognitionError>;
    
    /// Analyze sentiment in audio
    async fn analyze_sentiment(&mut self, audio: &AudioBuffer) -> Result<SentimentAnalysis, RecognitionError>;
    
    /// Perform multi-dimensional emotion analysis
    async fn analyze_multi_dimensional(&mut self, audio: &AudioBuffer) -> Result<MultiDimensionalEmotion, RecognitionError>;
    
    /// Detect stress level
    async fn detect_stress(&mut self, audio: &AudioBuffer) -> Result<f32, RecognitionError>;
    
    /// Detect fatigue level
    async fn detect_fatigue(&mut self, audio: &AudioBuffer) -> Result<f32, RecognitionError>;
    
    /// Update configuration
    async fn update_config(&mut self, config: EmotionConfig) -> Result<(), RecognitionError>;
    
    /// Get current configuration
    fn get_config(&self) -> &EmotionConfig;
    
    /// Get supported emotions
    fn get_supported_emotions(&self) -> Vec<EmotionType>;
}

/// Emotion tracking statistics
#[derive(Debug, Clone)]
pub struct EmotionStats {
    /// Total number of analyses performed
    pub total_analyses: u64,
    /// Emotion distribution over time
    pub emotion_distribution: HashMap<EmotionType, u64>,
    /// Average confidence scores per emotion
    pub avg_confidence: HashMap<EmotionType, f32>,
    /// Sentiment distribution
    pub sentiment_distribution: HashMap<SentimentPolarity, u64>,
    /// Average stress level
    pub avg_stress_level: f32,
    /// Average fatigue level
    pub avg_fatigue_level: f32,
    /// Session start time
    pub session_start: Instant,
    /// Last analysis time
    pub last_analysis: Option<Instant>,
}

impl Default for EmotionStats {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            emotion_distribution: HashMap::new(),
            avg_confidence: HashMap::new(),
            sentiment_distribution: HashMap::new(),
            avg_stress_level: 0.0,
            avg_fatigue_level: 0.0,
            session_start: Instant::now(),
            last_analysis: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_types() {
        let emotions = EmotionType::all();
        assert_eq!(emotions.len(), 12);
        assert!(emotions.contains(&EmotionType::Happy));
        assert!(emotions.contains(&EmotionType::Stressed));
    }

    #[test]
    fn test_emotion_names() {
        assert_eq!(EmotionType::Happy.name(), "Happy");
        assert_eq!(EmotionType::Stressed.name(), "Stressed");
        assert_eq!(EmotionType::Fatigued.name(), "Fatigued");
    }

    #[test]
    fn test_emotion_categories() {
        assert_eq!(EmotionType::Happy.category(), EmotionCategory::Primary);
        assert_eq!(EmotionType::Stressed.category(), EmotionCategory::Derived);
        assert_eq!(EmotionType::Fear.category(), EmotionCategory::Primary);
    }

    #[test]
    fn test_config_default() {
        let config = EmotionConfig::default();
        assert_eq!(config.min_confidence, 0.6);
        assert!(config.multi_dimensional);
        assert!(config.sentiment_analysis);
        assert!(config.stress_detection);
        assert!(config.fatigue_detection);
    }

    #[test]
    fn test_stats_default() {
        let stats = EmotionStats::default();
        assert_eq!(stats.total_analyses, 0);
        assert_eq!(stats.avg_stress_level, 0.0);
        assert_eq!(stats.avg_fatigue_level, 0.0);
        assert!(stats.emotion_distribution.is_empty());
    }
}