//! Emotion history tracking and analysis system
//!
//! This module provides comprehensive temporal emotion state tracking,
//! analysis capabilities, and historical data management for the emotion
//! processing system.

use crate::types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Historical emotion entry with metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionHistoryEntry {
    /// The emotion state at this point in time
    pub state: EmotionState,
    /// Timestamp when this state was recorded
    pub timestamp: SystemTime,
    /// Duration this emotion state was active (if known)
    pub duration: Option<Duration>,
    /// Context or trigger that caused this emotion change
    pub context: Option<String>,
    /// Confidence score for this emotion recognition (0.0 to 1.0)
    pub confidence: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl EmotionHistoryEntry {
    /// Create a new history entry
    pub fn new(state: EmotionState) -> Self {
        Self {
            state,
            timestamp: SystemTime::now(),
            duration: None,
            context: None,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a new history entry with context
    pub fn with_context(state: EmotionState, context: impl Into<String>) -> Self {
        Self {
            state,
            timestamp: SystemTime::now(),
            duration: None,
            context: Some(context.into()),
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the duration this emotion state was active
    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = Some(duration);
    }

    /// Set the confidence score
    pub fn set_confidence(&mut self, confidence: f32) {
        self.confidence = confidence.clamp(0.0, 1.0);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get the dominant emotion for this entry
    pub fn dominant_emotion(&self) -> Option<(Emotion, EmotionIntensity)> {
        self.state.current.emotion_vector.dominant_emotion()
    }

    /// Get the emotion dimensions for this entry
    pub fn dimensions(&self) -> EmotionDimensions {
        self.state.current.emotion_vector.dimensions
    }

    /// Calculate age of this entry
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.timestamp)
            .unwrap_or_default()
    }

    /// Check if this entry matches the given emotion
    pub fn has_emotion(&self, emotion: &Emotion) -> bool {
        self.state
            .current
            .emotion_vector
            .emotions
            .contains_key(emotion)
    }
}

/// Configuration for emotion history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionHistoryConfig {
    /// Maximum number of entries to keep in memory
    pub max_entries: usize,
    /// Maximum age of entries before automatic cleanup
    pub max_age: Duration,
    /// Whether to automatically track duration of emotion states
    pub track_duration: bool,
    /// Minimum time between history entries (to avoid spam)
    pub min_interval: Duration,
    /// Whether to compress old entries
    pub enable_compression: bool,
    /// Sample rate for compressed entries (keep 1 in N)
    pub compression_rate: usize,
}

impl Default for EmotionHistoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_age: Duration::from_secs(24 * 60 * 60), // 24 hours
            track_duration: true,
            min_interval: Duration::from_millis(100), // 100ms minimum
            enable_compression: true,
            compression_rate: 10, // Keep 1 in 10 old entries
        }
    }
}

/// Statistics about emotion history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionHistoryStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Time span covered by the history
    pub time_span: Duration,
    /// Most frequent emotion
    pub most_frequent_emotion: Option<(Emotion, usize)>,
    /// Average emotion valence
    pub average_valence: f32,
    /// Average emotion arousal
    pub average_arousal: f32,
    /// Average emotion dominance
    pub average_dominance: f32,
    /// Number of emotion transitions
    pub transition_count: usize,
    /// Average duration per emotion state
    pub average_duration: Option<Duration>,
    /// Emotion distribution (emotion -> count)
    pub emotion_distribution: HashMap<String, usize>,
}

/// Emotion pattern detected in history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionPattern {
    /// Description of the pattern
    pub description: String,
    /// Emotions involved in the pattern
    pub emotions: Vec<Emotion>,
    /// Frequency of this pattern
    pub frequency: usize,
    /// Average duration of the pattern
    pub average_duration: Duration,
    /// Confidence in this pattern detection
    pub confidence: f32,
}

/// Emotion transition data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransition {
    /// Source emotion
    pub from: Emotion,
    /// Target emotion
    pub to: Emotion,
    /// Duration of the transition
    pub duration: Duration,
    /// Timestamp of the transition
    pub timestamp: SystemTime,
    /// Context that triggered the transition
    pub context: Option<String>,
}

/// Comprehensive emotion history tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionHistory {
    /// Configuration for this history tracker
    pub config: EmotionHistoryConfig,
    /// Historical entries
    entries: Vec<EmotionHistoryEntry>,
    /// Compressed entries (for long-term storage)
    compressed_entries: Vec<EmotionHistoryEntry>,
    /// Detected emotion transitions
    transitions: Vec<EmotionTransition>,
    /// Last entry timestamp to enforce minimum interval
    last_entry_time: Option<SystemTime>,
}

impl EmotionHistory {
    /// Create a new emotion history tracker
    pub fn new() -> Self {
        Self::with_config(EmotionHistoryConfig::default())
    }

    /// Create a new emotion history tracker with configuration
    pub fn with_config(config: EmotionHistoryConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            compressed_entries: Vec::new(),
            transitions: Vec::new(),
            last_entry_time: None,
        }
    }

    /// Add a new emotion state to the history
    pub fn add_entry(&mut self, mut entry: EmotionHistoryEntry) {
        // Check minimum interval
        if let Some(last_time) = self.last_entry_time {
            if entry
                .timestamp
                .duration_since(last_time)
                .unwrap_or_default()
                < self.config.min_interval
            {
                return; // Skip this entry due to minimum interval
            }
        }

        // Set duration for previous entry if tracking is enabled
        if self.config.track_duration && !self.entries.is_empty() {
            if let Some(last_entry) = self.entries.last_mut() {
                if last_entry.duration.is_none() {
                    let duration = entry
                        .timestamp
                        .duration_since(last_entry.timestamp)
                        .unwrap_or_default();
                    last_entry.set_duration(duration);
                }
            }
        }

        // Detect transitions
        if let Some(last_entry) = self.entries.last() {
            if let (Some(from_emotion), Some(to_emotion)) =
                (last_entry.dominant_emotion(), entry.dominant_emotion())
            {
                if from_emotion.0 != to_emotion.0 {
                    let transition = EmotionTransition {
                        from: from_emotion.0,
                        to: to_emotion.0,
                        duration: entry
                            .timestamp
                            .duration_since(last_entry.timestamp)
                            .unwrap_or_default(),
                        timestamp: entry.timestamp,
                        context: entry.context.clone(),
                    };
                    self.transitions.push(transition);
                }
            }
        }

        // Add the new entry
        let timestamp = entry.timestamp;
        self.entries.push(entry);
        self.last_entry_time = Some(timestamp);

        // Perform maintenance
        self.perform_maintenance();
    }

    /// Add a simple emotion state
    pub fn add_state(&mut self, state: EmotionState) {
        let entry = EmotionHistoryEntry::new(state);
        self.add_entry(entry);
    }

    /// Add an emotion state with context
    pub fn add_state_with_context(&mut self, state: EmotionState, context: impl Into<String>) {
        let entry = EmotionHistoryEntry::with_context(state, context);
        self.add_entry(entry);
    }

    /// Get all entries in chronological order
    pub fn get_entries(&self) -> &[EmotionHistoryEntry] {
        &self.entries
    }

    /// Get entries within a time range
    pub fn get_entries_in_range(
        &self,
        start: SystemTime,
        end: SystemTime,
    ) -> Vec<&EmotionHistoryEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .collect()
    }

    /// Get entries for a specific emotion
    pub fn get_entries_for_emotion(&self, emotion: &Emotion) -> Vec<&EmotionHistoryEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.has_emotion(emotion))
            .collect()
    }

    /// Get recent entries (last N)
    pub fn get_recent_entries(&self, count: usize) -> &[EmotionHistoryEntry] {
        let start = self.entries.len().saturating_sub(count);
        &self.entries[start..]
    }

    /// Get entries from the last duration
    pub fn get_entries_since(&self, duration: Duration) -> Vec<&EmotionHistoryEntry> {
        let cutoff = SystemTime::now() - duration;
        self.entries
            .iter()
            .filter(|entry| entry.timestamp >= cutoff)
            .collect()
    }

    /// Calculate comprehensive statistics
    pub fn calculate_stats(&self) -> EmotionHistoryStats {
        if self.entries.is_empty() {
            return EmotionHistoryStats {
                total_entries: 0,
                time_span: Duration::default(),
                most_frequent_emotion: None,
                average_valence: 0.0,
                average_arousal: 0.0,
                average_dominance: 0.0,
                transition_count: 0,
                average_duration: None,
                emotion_distribution: HashMap::new(),
            };
        }

        let mut emotion_counts: HashMap<String, usize> = HashMap::new();
        let mut total_valence = 0.0;
        let mut total_arousal = 0.0;
        let mut total_dominance = 0.0;
        let mut total_duration = Duration::default();
        let mut duration_count = 0;

        for entry in &self.entries {
            // Count emotions
            if let Some((emotion, _)) = entry.dominant_emotion() {
                *emotion_counts
                    .entry(emotion.as_str().to_string())
                    .or_insert(0) += 1;
            }

            // Sum dimensions
            let dims = entry.dimensions();
            total_valence += dims.valence;
            total_arousal += dims.arousal;
            total_dominance += dims.dominance;

            // Sum durations
            if let Some(duration) = entry.duration {
                total_duration += duration;
                duration_count += 1;
            }
        }

        let count = self.entries.len();
        let most_frequent = emotion_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(emotion, count)| (Emotion::from_str(emotion), *count));

        let time_span = if count > 1 {
            self.entries
                .last()
                .unwrap()
                .timestamp
                .duration_since(self.entries.first().unwrap().timestamp)
                .unwrap_or_default()
        } else {
            Duration::default()
        };

        EmotionHistoryStats {
            total_entries: count,
            time_span,
            most_frequent_emotion: most_frequent,
            average_valence: total_valence / count as f32,
            average_arousal: total_arousal / count as f32,
            average_dominance: total_dominance / count as f32,
            transition_count: self.transitions.len(),
            average_duration: if duration_count > 0 {
                Some(total_duration / duration_count as u32)
            } else {
                None
            },
            emotion_distribution: emotion_counts,
        }
    }

    /// Detect emotion patterns in the history
    pub fn detect_patterns(&self) -> Vec<EmotionPattern> {
        let mut patterns = Vec::new();

        // Simple pattern detection: look for sequences of 3+ emotions that repeat
        if self.entries.len() < 6 {
            // Need at least 2 patterns of length 3
            return patterns;
        }

        let emotions: Vec<Option<Emotion>> = self
            .entries
            .iter()
            .map(|entry| entry.dominant_emotion().map(|(e, _)| e))
            .collect();

        // Look for patterns of length 3-5
        for pattern_length in 3..=5.min(self.entries.len() / 2) {
            let mut pattern_counts: HashMap<Vec<String>, usize> = HashMap::new();

            for window in emotions.windows(pattern_length) {
                if window.iter().all(|e| e.is_some()) {
                    let pattern: Vec<String> = window
                        .iter()
                        .map(|e| e.as_ref().unwrap().as_str().to_string())
                        .collect();
                    *pattern_counts.entry(pattern).or_insert(0) += 1;
                }
            }

            // Convert patterns with frequency >= 2 to EmotionPattern
            for (pattern_emotions, frequency) in pattern_counts {
                if frequency >= 2 {
                    let emotions: Vec<Emotion> = pattern_emotions
                        .iter()
                        .map(|s| Emotion::from_str(s))
                        .collect();

                    patterns.push(EmotionPattern {
                        description: format!("Repeating sequence: {:?}", pattern_emotions),
                        emotions,
                        frequency,
                        average_duration: Duration::from_secs(60), // Placeholder
                        confidence: (frequency as f32).min(1.0),
                    });
                }
            }
        }

        patterns
    }

    /// Get all detected transitions
    pub fn get_transitions(&self) -> &[EmotionTransition] {
        &self.transitions
    }

    /// Get transitions for a specific emotion pair
    pub fn get_transitions_between(&self, from: &Emotion, to: &Emotion) -> Vec<&EmotionTransition> {
        self.transitions
            .iter()
            .filter(|t| &t.from == from && &t.to == to)
            .collect()
    }

    /// Clear all history data
    pub fn clear(&mut self) {
        self.entries.clear();
        self.compressed_entries.clear();
        self.transitions.clear();
        self.last_entry_time = None;
    }

    /// Get the total number of entries (including compressed)
    pub fn total_entries(&self) -> usize {
        self.entries.len() + self.compressed_entries.len()
    }

    /// Export history to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Import history from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save history to file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load history from file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let history = Self::from_json(&json)?;
        Ok(history)
    }

    /// Perform maintenance tasks (cleanup old entries, compression)
    fn perform_maintenance(&mut self) {
        let now = SystemTime::now();

        // Remove entries older than max_age
        self.entries.retain(|entry| {
            now.duration_since(entry.timestamp).unwrap_or_default() <= self.config.max_age
        });

        // Compress old entries if enabled
        if self.config.enable_compression && self.entries.len() > self.config.max_entries {
            let excess = self.entries.len() - self.config.max_entries;
            let to_compress = self.entries.drain(..excess).collect::<Vec<_>>();

            // Compress by keeping every Nth entry
            for (i, entry) in to_compress.into_iter().enumerate() {
                if i % self.config.compression_rate == 0 {
                    self.compressed_entries.push(entry);
                }
            }
        }

        // Limit compressed entries as well
        if self.compressed_entries.len() > self.config.max_entries {
            let excess = self.compressed_entries.len() - self.config.max_entries;
            self.compressed_entries.drain(..excess);
        }

        // Clean up old transitions
        self.transitions.retain(|transition| {
            now.duration_since(transition.timestamp).unwrap_or_default() <= self.config.max_age
        });
    }
}

impl Default for EmotionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmotionParameters, EmotionVector};

    fn create_test_state(emotion: Emotion, intensity: f32) -> EmotionState {
        let mut vector = EmotionVector::new();
        vector.add_emotion(emotion, intensity.into());
        let params = EmotionParameters::new(vector);
        EmotionState::new(params)
    }

    #[test]
    fn test_history_basic_functionality() {
        // Use config with no minimum interval for testing
        let config = EmotionHistoryConfig {
            min_interval: std::time::Duration::from_millis(0),
            ..Default::default()
        };
        let mut history = EmotionHistory::with_config(config);

        let happy_state = create_test_state(Emotion::Happy, 0.8);
        let sad_state = create_test_state(Emotion::Sad, 0.6);

        history.add_state(happy_state);
        std::thread::sleep(std::time::Duration::from_millis(1)); // Ensure different timestamps
        history.add_state(sad_state);

        assert_eq!(history.get_entries().len(), 2);
        assert_eq!(history.get_transitions().len(), 1);

        let transition = &history.get_transitions()[0];
        assert_eq!(transition.from, Emotion::Happy);
        assert_eq!(transition.to, Emotion::Sad);
    }

    #[test]
    fn test_history_stats_calculation() {
        let config = EmotionHistoryConfig {
            min_interval: std::time::Duration::from_millis(0),
            ..Default::default()
        };
        let mut history = EmotionHistory::with_config(config);

        // Add several states with small delays
        history.add_state(create_test_state(Emotion::Happy, 0.8));
        std::thread::sleep(std::time::Duration::from_millis(1));
        history.add_state(create_test_state(Emotion::Happy, 0.7));
        std::thread::sleep(std::time::Duration::from_millis(1));
        history.add_state(create_test_state(Emotion::Sad, 0.6));

        let stats = history.calculate_stats();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.transition_count, 1);
        assert!(stats.most_frequent_emotion.is_some());

        if let Some((emotion, count)) = stats.most_frequent_emotion {
            assert_eq!(emotion, Emotion::Happy);
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_history_filtering() {
        let config = EmotionHistoryConfig {
            min_interval: std::time::Duration::from_millis(0),
            ..Default::default()
        };
        let mut history = EmotionHistory::with_config(config);

        history.add_state(create_test_state(Emotion::Happy, 0.8));
        std::thread::sleep(std::time::Duration::from_millis(1));
        history.add_state(create_test_state(Emotion::Sad, 0.6));
        std::thread::sleep(std::time::Duration::from_millis(1));
        history.add_state(create_test_state(Emotion::Happy, 0.7));

        let happy_entries = history.get_entries_for_emotion(&Emotion::Happy);
        assert_eq!(happy_entries.len(), 2);

        let recent_entries = history.get_recent_entries(2);
        assert_eq!(recent_entries.len(), 2);
    }

    #[test]
    fn test_pattern_detection() {
        let config = EmotionHistoryConfig {
            min_interval: std::time::Duration::from_millis(0),
            ..Default::default()
        };
        let mut history = EmotionHistory::with_config(config);

        // Create a repeating pattern: Happy -> Sad -> Angry
        for i in 0..9 {
            // 3 complete patterns
            let emotion = match i % 3 {
                0 => Emotion::Happy,
                1 => Emotion::Sad,
                _ => Emotion::Angry,
            };
            history.add_state(create_test_state(emotion, 0.8));
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let patterns = history.detect_patterns();
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_history_serialization() {
        let config = EmotionHistoryConfig {
            min_interval: std::time::Duration::from_millis(0),
            ..Default::default()
        };
        let mut history = EmotionHistory::with_config(config);

        history.add_state(create_test_state(Emotion::Happy, 0.8));
        std::thread::sleep(std::time::Duration::from_millis(1));
        history.add_state(create_test_state(Emotion::Sad, 0.6));

        let json = history.to_json().unwrap();
        let loaded_history = EmotionHistory::from_json(&json).unwrap();

        assert_eq!(loaded_history.get_entries().len(), 2);
        assert_eq!(loaded_history.get_transitions().len(), 1);
    }

    #[test]
    fn test_history_maintenance() {
        let config = EmotionHistoryConfig {
            max_entries: 5,
            max_age: Duration::from_secs(1),
            ..Default::default()
        };

        let mut history = EmotionHistory::with_config(config);

        // Add more entries than the limit
        for i in 0..10 {
            history.add_state(create_test_state(Emotion::Happy, 0.5 + i as f32 * 0.1));
        }

        // Should have compressed old entries
        assert!(history.total_entries() <= 15); // Some compression should have occurred
    }
}
