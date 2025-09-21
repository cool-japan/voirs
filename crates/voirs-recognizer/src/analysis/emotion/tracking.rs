//! Emotion tracking and mood analysis over time
//!
//! Provides temporal emotion analysis, mood tracking, and emotional state transitions.

use super::{EmotionDetection, EmotionType, SentimentAnalysis, SentimentPolarity};
use crate::RecognitionError;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Emotion tracker for temporal analysis
pub struct EmotionTracker {
    /// Historical emotion detections
    emotion_history: VecDeque<EmotionDetection>,
    /// Historical sentiment analyses
    sentiment_history: VecDeque<SentimentAnalysis>,
    /// Mood tracking window (in seconds)
    tracking_window: Duration,
    /// Maximum history size
    max_history_size: usize,
    /// Current emotional state
    current_state: EmotionalState,
    /// Mood patterns
    mood_patterns: Vec<MoodPattern>,
    /// Transition tracking
    transitions: HashMap<(EmotionType, EmotionType), u32>,
}

impl EmotionTracker {
    /// Create new emotion tracker
    pub fn new() -> Self {
        Self {
            emotion_history: VecDeque::new(),
            sentiment_history: VecDeque::new(),
            tracking_window: Duration::from_secs(300), // 5 minutes
            max_history_size: 1000,
            current_state: EmotionalState::default(),
            mood_patterns: Vec::new(),
            transitions: HashMap::new(),
        }
    }

    /// Create tracker with custom tracking window
    pub fn with_window(window: Duration) -> Self {
        let mut tracker = Self::new();
        tracker.tracking_window = window;
        tracker
    }

    /// Update tracker with new emotion detection
    pub async fn update(&mut self, emotion: &EmotionDetection) {
        // Track emotion transitions
        if let Some(last_emotion) = self.emotion_history.back() {
            if last_emotion.emotion != emotion.emotion {
                let transition = (last_emotion.emotion, emotion.emotion);
                *self.transitions.entry(transition).or_insert(0) += 1;
            }
        }

        // Add to history
        self.emotion_history.push_back(emotion.clone());
        
        // Update current state
        self.update_current_state(emotion).await;
        
        // Analyze mood patterns
        self.analyze_mood_patterns().await;
        
        // Clean up old data
        self.cleanup_history();
    }

    /// Update tracker with sentiment analysis
    pub async fn update_sentiment(&mut self, sentiment: &SentimentAnalysis) {
        self.sentiment_history.push_back(sentiment.clone());
        self.cleanup_history();
    }

    /// Get current emotional state
    pub fn get_current_state(&self) -> &EmotionalState {
        &self.current_state
    }

    /// Get emotion trends over time
    pub fn get_emotion_trends(&self) -> EmotionTrends {
        let window_start = Instant::now() - self.tracking_window;
        
        let recent_emotions: Vec<_> = self.emotion_history.iter()
            .filter(|e| e.timestamp > window_start)
            .collect();

        if recent_emotions.is_empty() {
            return EmotionTrends::default();
        }

        // Calculate emotion distribution
        let mut emotion_counts = HashMap::new();
        let mut confidence_sum = HashMap::new();
        
        for emotion in &recent_emotions {
            *emotion_counts.entry(emotion.emotion).or_insert(0) += 1;
            *confidence_sum.entry(emotion.emotion).or_insert(0.0) += emotion.confidence;
        }

        let mut emotion_distribution = HashMap::new();
        let mut avg_confidence = HashMap::new();
        let total_count = recent_emotions.len() as f32;
        
        for (emotion_type, count) in emotion_counts {
            emotion_distribution.insert(emotion_type, count as f32 / total_count);
            avg_confidence.insert(emotion_type, confidence_sum[&emotion_type] / count as f32);
        }

        // Calculate dominant emotion
        let dominant_emotion = emotion_distribution.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(emotion, _)| *emotion)
            .unwrap_or(EmotionType::Neutral);

        // Calculate emotional stability (variance in emotions)
        let stability = if recent_emotions.len() > 1 {
            let unique_emotions = emotion_counts.len();
            1.0 - (unique_emotions as f32 - 1.0) / (EmotionType::all().len() as f32 - 1.0)
        } else {
            1.0
        };

        // Calculate transition frequency
        let transition_count = self.transitions.values().sum::<u32>() as f32;
        let transition_frequency = if recent_emotions.len() > 1 {
            transition_count / (recent_emotions.len() - 1) as f32
        } else {
            0.0
        };

        EmotionTrends {
            dominant_emotion,
            emotion_distribution,
            avg_confidence,
            emotional_stability: stability.max(0.0).min(1.0),
            transition_frequency,
            analysis_window: self.tracking_window,
            sample_count: recent_emotions.len(),
        }
    }

    /// Get sentiment trends over time
    pub fn get_sentiment_trends(&self) -> SentimentTrends {
        let window_start = Instant::now() - self.tracking_window;
        
        let recent_sentiments: Vec<_> = self.sentiment_history.iter()
            .filter(|s| s.timestamp > window_start)
            .collect();

        if recent_sentiments.is_empty() {
            return SentimentTrends::default();
        }

        // Calculate averages
        let avg_valence = recent_sentiments.iter().map(|s| s.valence).sum::<f32>() / recent_sentiments.len() as f32;
        let avg_arousal = recent_sentiments.iter().map(|s| s.arousal).sum::<f32>() / recent_sentiments.len() as f32;
        let avg_dominance = recent_sentiments.iter().map(|s| s.dominance).sum::<f32>() / recent_sentiments.len() as f32;

        // Calculate variance for stability
        let valence_variance = recent_sentiments.iter()
            .map(|s| (s.valence - avg_valence).powi(2))
            .sum::<f32>() / recent_sentiments.len() as f32;
        
        let sentiment_stability = 1.0 - valence_variance.sqrt().min(1.0);

        // Determine dominant polarity
        let mut polarity_counts = HashMap::new();
        for sentiment in &recent_sentiments {
            *polarity_counts.entry(sentiment.polarity).or_insert(0) += 1;
        }

        let dominant_polarity = polarity_counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(polarity, _)| *polarity)
            .unwrap_or(SentimentPolarity::Neutral);

        SentimentTrends {
            dominant_polarity,
            avg_valence,
            avg_arousal,
            avg_dominance,
            sentiment_stability,
            analysis_window: self.tracking_window,
            sample_count: recent_sentiments.len(),
        }
    }

    /// Get detected mood patterns
    pub fn get_mood_patterns(&self) -> &[MoodPattern] {
        &self.mood_patterns
    }

    /// Get emotion transition matrix
    pub fn get_transition_matrix(&self) -> &HashMap<(EmotionType, EmotionType), u32> {
        &self.transitions
    }

    /// Update current emotional state
    async fn update_current_state(&mut self, emotion: &EmotionDetection) {
        self.current_state.primary_emotion = emotion.emotion;
        self.current_state.confidence = emotion.confidence;
        self.current_state.intensity = emotion.intensity;
        self.current_state.last_update = emotion.timestamp;
        
        // Update duration in current state
        if let Some(state_start) = self.current_state.state_start {
            self.current_state.duration = emotion.timestamp.duration_since(state_start);
        } else {
            self.current_state.state_start = Some(emotion.timestamp);
            self.current_state.duration = Duration::ZERO;
        }

        // Check if we need to reset state start (significant emotion change)
        if emotion.confidence > 0.8 && emotion.emotion != self.current_state.primary_emotion {
            self.current_state.state_start = Some(emotion.timestamp);
            self.current_state.duration = Duration::ZERO;
        }
    }

    /// Analyze mood patterns in recent history
    async fn analyze_mood_patterns(&mut self) {
        let window_start = Instant::now() - self.tracking_window;
        
        let recent_emotions: Vec<_> = self.emotion_history.iter()
            .filter(|e| e.timestamp > window_start)
            .collect();

        if recent_emotions.len() < 3 {
            return; // Need at least 3 samples for pattern analysis
        }

        // Look for cyclical patterns
        self.detect_cyclical_patterns(&recent_emotions);
        
        // Look for escalation patterns
        self.detect_escalation_patterns(&recent_emotions);
        
        // Look for stability patterns
        self.detect_stability_patterns(&recent_emotions);
    }

    /// Detect cyclical emotion patterns
    fn detect_cyclical_patterns(&mut self, emotions: &[&EmotionDetection]) {
        // Simple pattern detection for demonstration
        // In practice, this would use more sophisticated algorithms
        
        if emotions.len() < 6 {
            return;
        }

        // Look for ABA patterns (emotion A -> emotion B -> emotion A)
        for i in 0..(emotions.len() - 2) {
            let emotion_a = emotions[i].emotion;
            let emotion_b = emotions[i + 1].emotion;
            let emotion_c = emotions[i + 2].emotion;

            if emotion_a == emotion_c && emotion_a != emotion_b {
                let pattern = MoodPattern {
                    pattern_type: MoodPatternType::Cyclical,
                    emotions: vec![emotion_a, emotion_b, emotion_c],
                    confidence: 0.7,
                    duration: emotions[i + 2].timestamp.duration_since(emotions[i].timestamp),
                    frequency: 1, // Simplified
                };

                // Only add if not already detected recently
                if !self.mood_patterns.iter().any(|p| 
                    p.pattern_type == MoodPatternType::Cyclical && 
                    p.emotions == pattern.emotions
                ) {
                    self.mood_patterns.push(pattern);
                }
            }
        }
    }

    /// Detect emotion escalation patterns
    fn detect_escalation_patterns(&mut self, emotions: &[&EmotionDetection]) {
        if emotions.len() < 3 {
            return;
        }

        // Look for intensity escalation
        let mut escalating = true;
        for i in 1..emotions.len() {
            if emotions[i].intensity <= emotions[i - 1].intensity {
                escalating = false;
                break;
            }
        }

        if escalating && emotions.len() >= 3 {
            let pattern = MoodPattern {
                pattern_type: MoodPatternType::Escalation,
                emotions: emotions.iter().map(|e| e.emotion).collect(),
                confidence: 0.8,
                duration: emotions.last().unwrap().timestamp.duration_since(emotions[0].timestamp),
                frequency: 1,
            };

            self.mood_patterns.push(pattern);
        }
    }

    /// Detect emotional stability patterns
    fn detect_stability_patterns(&mut self, emotions: &[&EmotionDetection]) {
        if emotions.len() < 5 {
            return;
        }

        // Check if emotions have been stable (same emotion type)
        let first_emotion = emotions[0].emotion;
        let all_same = emotions.iter().all(|e| e.emotion == first_emotion);

        if all_same {
            let avg_confidence = emotions.iter().map(|e| e.confidence).sum::<f32>() / emotions.len() as f32;
            
            if avg_confidence > 0.7 {
                let pattern = MoodPattern {
                    pattern_type: MoodPatternType::Stable,
                    emotions: vec![first_emotion],
                    confidence: avg_confidence,
                    duration: emotions.last().unwrap().timestamp.duration_since(emotions[0].timestamp),
                    frequency: emotions.len(),
                };

                self.mood_patterns.push(pattern);
            }
        }
    }

    /// Clean up old history entries
    fn cleanup_history(&mut self) {
        let cutoff = Instant::now() - self.tracking_window;
        
        // Remove old emotions
        while let Some(front) = self.emotion_history.front() {
            if front.timestamp < cutoff {
                self.emotion_history.pop_front();
            } else {
                break;
            }
        }

        // Remove old sentiments
        while let Some(front) = self.sentiment_history.front() {
            if front.timestamp < cutoff {
                self.sentiment_history.pop_front();
            } else {
                break;
            }
        }

        // Limit history size
        while self.emotion_history.len() > self.max_history_size {
            self.emotion_history.pop_front();
        }
        
        while self.sentiment_history.len() > self.max_history_size {
            self.sentiment_history.pop_front();
        }

        // Clean up old mood patterns
        self.mood_patterns.retain(|pattern| {
            Instant::now().duration_since(self.current_state.last_update) < Duration::from_secs(3600) // Keep for 1 hour
        });
    }
}

impl Default for EmotionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Current emotional state
#[derive(Debug, Clone)]
pub struct EmotionalState {
    /// Primary emotion
    pub primary_emotion: EmotionType,
    /// Confidence in current state
    pub confidence: f32,
    /// Emotional intensity
    pub intensity: f32,
    /// When this state started
    pub state_start: Option<Instant>,
    /// Duration in current state
    pub duration: Duration,
    /// Last update timestamp
    pub last_update: Instant,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            primary_emotion: EmotionType::Neutral,
            confidence: 0.0,
            intensity: 0.0,
            state_start: None,
            duration: Duration::ZERO,
            last_update: Instant::now(),
        }
    }
}

/// Emotion trends analysis
#[derive(Debug, Clone)]
pub struct EmotionTrends {
    /// Most frequent emotion in window
    pub dominant_emotion: EmotionType,
    /// Distribution of emotions (percentages)
    pub emotion_distribution: HashMap<EmotionType, f32>,
    /// Average confidence per emotion
    pub avg_confidence: HashMap<EmotionType, f32>,
    /// Emotional stability (0.0 = very unstable, 1.0 = very stable)
    pub emotional_stability: f32,
    /// Frequency of emotion transitions
    pub transition_frequency: f32,
    /// Analysis window duration
    pub analysis_window: Duration,
    /// Number of samples analyzed
    pub sample_count: usize,
}

impl Default for EmotionTrends {
    fn default() -> Self {
        Self {
            dominant_emotion: EmotionType::Neutral,
            emotion_distribution: HashMap::new(),
            avg_confidence: HashMap::new(),
            emotional_stability: 1.0,
            transition_frequency: 0.0,
            analysis_window: Duration::from_secs(300),
            sample_count: 0,
        }
    }
}

/// Sentiment trends analysis
#[derive(Debug, Clone)]
pub struct SentimentTrends {
    /// Most frequent sentiment polarity
    pub dominant_polarity: SentimentPolarity,
    /// Average valence over time
    pub avg_valence: f32,
    /// Average arousal over time
    pub avg_arousal: f32,
    /// Average dominance over time
    pub avg_dominance: f32,
    /// Sentiment stability
    pub sentiment_stability: f32,
    /// Analysis window duration
    pub analysis_window: Duration,
    /// Number of samples analyzed
    pub sample_count: usize,
}

impl Default for SentimentTrends {
    fn default() -> Self {
        Self {
            dominant_polarity: SentimentPolarity::Neutral,
            avg_valence: 0.0,
            avg_arousal: 0.0,
            avg_dominance: 0.0,
            sentiment_stability: 1.0,
            analysis_window: Duration::from_secs(300),
            sample_count: 0,
        }
    }
}

/// Mood pattern types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoodPatternType {
    /// Cyclical patterns (A->B->A)
    Cyclical,
    /// Escalation patterns (increasing intensity)
    Escalation,
    /// De-escalation patterns (decreasing intensity)
    DeEscalation,
    /// Stable patterns (consistent emotion)
    Stable,
    /// Rapid mood swings
    Volatile,
}

/// Detected mood pattern
#[derive(Debug, Clone)]
pub struct MoodPattern {
    /// Type of pattern
    pub pattern_type: MoodPatternType,
    /// Emotions involved in pattern
    pub emotions: Vec<EmotionType>,
    /// Confidence in pattern detection
    pub confidence: f32,
    /// Duration of pattern
    pub duration: Duration,
    /// Frequency of occurrence
    pub frequency: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_emotion_tracker_creation() {
        let tracker = EmotionTracker::new();
        assert_eq!(tracker.get_current_state().primary_emotion, EmotionType::Neutral);
        assert_eq!(tracker.emotion_history.len(), 0);
    }

    #[tokio::test]
    async fn test_emotion_update() {
        let mut tracker = EmotionTracker::new();
        
        let emotion = EmotionDetection {
            emotion: EmotionType::Happy,
            confidence: 0.8,
            intensity: 0.7,
            timestamp: Instant::now(),
            start_time: Duration::ZERO,
            end_time: Duration::from_secs(1),
            emotion_scores: HashMap::new(),
        };

        tracker.update(&emotion).await;
        
        assert_eq!(tracker.get_current_state().primary_emotion, EmotionType::Happy);
        assert_eq!(tracker.get_current_state().confidence, 0.8);
        assert_eq!(tracker.emotion_history.len(), 1);
    }

    #[tokio::test]
    async fn test_emotion_trends() {
        let mut tracker = EmotionTracker::new();
        let now = Instant::now();
        
        // Add several emotions
        for i in 0..5 {
            let emotion = EmotionDetection {
                emotion: if i % 2 == 0 { EmotionType::Happy } else { EmotionType::Sad },
                confidence: 0.8,
                intensity: 0.7,
                timestamp: now + Duration::from_secs(i),
                start_time: Duration::ZERO,
                end_time: Duration::from_secs(1),
                emotion_scores: HashMap::new(),
            };
            tracker.update(&emotion).await;
        }

        let trends = tracker.get_emotion_trends();
        assert_eq!(trends.sample_count, 5);
        assert!(trends.emotion_distribution.contains_key(&EmotionType::Happy));
        assert!(trends.emotion_distribution.contains_key(&EmotionType::Sad));
    }

    #[tokio::test]
    async fn test_transition_tracking() {
        let mut tracker = EmotionTracker::new();
        let now = Instant::now();
        
        // Create emotion sequence: Happy -> Sad -> Happy
        let emotions = vec![
            (EmotionType::Happy, 0),
            (EmotionType::Sad, 1),
            (EmotionType::Happy, 2),
        ];

        for (emotion_type, offset) in emotions {
            let emotion = EmotionDetection {
                emotion: emotion_type,
                confidence: 0.8,
                intensity: 0.7,
                timestamp: now + Duration::from_secs(offset),
                start_time: Duration::ZERO,
                end_time: Duration::from_secs(1),
                emotion_scores: HashMap::new(),
            };
            tracker.update(&emotion).await;
        }

        let transitions = tracker.get_transition_matrix();
        assert!(transitions.contains_key(&(EmotionType::Happy, EmotionType::Sad)));
        assert!(transitions.contains_key(&(EmotionType::Sad, EmotionType::Happy)));
    }

    #[test]
    fn test_emotional_state_default() {
        let state = EmotionalState::default();
        assert_eq!(state.primary_emotion, EmotionType::Neutral);
        assert_eq!(state.confidence, 0.0);
        assert_eq!(state.duration, Duration::ZERO);
    }

    #[test]
    fn test_trends_default() {
        let trends = EmotionTrends::default();
        assert_eq!(trends.dominant_emotion, EmotionType::Neutral);
        assert_eq!(trends.sample_count, 0);
        
        let sentiment_trends = SentimentTrends::default();
        assert_eq!(sentiment_trends.dominant_polarity, SentimentPolarity::Neutral);
        assert_eq!(sentiment_trends.sample_count, 0);
    }
}