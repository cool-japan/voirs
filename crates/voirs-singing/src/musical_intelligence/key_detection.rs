//! Key detection and analysis

use super::types::{KeyMode, KeyProfile, KeyResult};
use crate::types::NoteEvent;
use crate::Result;
use std::collections::HashMap;

/// Automatic key signature detection system using Krumhansl-Schmuckler algorithm
#[derive(Debug, Clone)]
pub struct KeyDetector {
    /// Key profiles for all major and minor keys (24 total profiles)
    key_profiles: HashMap<String, KeyProfile>,
    /// Analysis window size in beats for segmented key detection
    window_size: f32,
    /// Minimum confidence threshold (0.0-1.0) for accepting key detection results
    min_confidence: f32,
}

impl KeyDetector {
    /// Create a new key detector
    pub fn new() -> Self {
        let mut detector = Self {
            key_profiles: HashMap::new(),
            window_size: 8.0, // 8 beats
            min_confidence: 0.6,
        };

        detector.initialize_key_profiles();
        detector
    }

    /// Initialize key profiles for all major and minor keys using Krumhansl-Schmuckler weights
    fn initialize_key_profiles(&mut self) {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        // Add major key profiles
        for (i, &note) in note_names.iter().enumerate() {
            let mut profile = KeyProfile::major();
            profile.name = format!("{} major", note);

            // Rotate the profile weights to match the key
            let mut rotated_weights = [0.0; 12];
            for (j, weight) in rotated_weights.iter_mut().enumerate() {
                *weight = profile.weights[(j + 12 - i) % 12];
            }
            profile.weights = rotated_weights;

            self.key_profiles.insert(format!("{}_major", note), profile);
        }

        // Add minor key profiles
        for (i, &note) in note_names.iter().enumerate() {
            let mut profile = KeyProfile::minor();
            profile.name = format!("{} minor", note);

            // Rotate the profile weights to match the key
            let mut rotated_weights = [0.0; 12];
            for (j, weight) in rotated_weights.iter_mut().enumerate() {
                *weight = profile.weights[(j + 12 - i) % 12];
            }
            profile.weights = rotated_weights;

            self.key_profiles.insert(format!("{}_minor", note), profile);
        }
    }

    /// Detect key from note events using Krumhansl-Schmuckler algorithm
    ///
    /// # Arguments
    ///
    /// * `note_events` - Musical note events to analyze for key signature
    ///
    /// # Returns
    ///
    /// Detected key with confidence score and alternative key candidates
    ///
    /// # Errors
    ///
    /// Returns error if key detection fails
    pub async fn detect_key(&self, note_events: &[NoteEvent]) -> Result<KeyResult> {
        // Extract pitch class histogram
        let pitch_histogram = self.extract_pitch_histogram(note_events);

        // Find best matching key profile using Krumhansl-Schmuckler algorithm
        let mut best_key = String::new();
        let mut best_confidence = 0.0;
        let mut alternatives = Vec::new();

        for (key_name, profile) in &self.key_profiles {
            let correlation = self.calculate_correlation(&pitch_histogram, &profile.weights);

            if correlation > best_confidence {
                if !best_key.is_empty() {
                    alternatives.push((best_key.clone(), best_confidence));
                }
                best_key = key_name.clone();
                best_confidence = correlation;
            } else if correlation > 0.4 {
                alternatives.push((key_name.clone(), correlation));
            }
        }

        // Sort alternatives by confidence
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        alternatives.truncate(3); // Keep top 3 alternatives

        // Parse key name
        let (root_note, mode) = self.parse_key_name(&best_key);

        Ok(KeyResult {
            key_name: self.format_key_name(&root_note, mode),
            root_note,
            mode,
            confidence: best_confidence,
            alternatives,
        })
    }

    /// Extract pitch class histogram from note events weighted by duration and velocity
    ///
    /// # Arguments
    ///
    /// * `note_events` - Note events to extract histogram from
    ///
    /// # Returns
    ///
    /// Normalized pitch class histogram (12 bins, one per semitone)
    fn extract_pitch_histogram(&self, note_events: &[NoteEvent]) -> [f32; 12] {
        let mut histogram = [0.0; 12];

        for note in note_events {
            let pitch_class = self.frequency_to_pitch_class(note.frequency);
            histogram[pitch_class as usize] += note.duration * note.velocity;
        }

        // Normalize histogram
        let total: f32 = histogram.iter().sum();
        if total > 0.0 {
            for value in &mut histogram {
                *value /= total;
            }
        }

        histogram
    }

    /// Convert frequency to pitch class (0-11, where C=0)
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Pitch class number (0=C, 1=C#, 2=D, etc.)
    fn frequency_to_pitch_class(&self, frequency: f32) -> u8 {
        let a4_freq = 440.0;
        let semitones_from_a4 = 12.0 * (frequency / a4_freq).log2();
        let semitones = semitones_from_a4.round() as i32;
        ((semitones + 9) % 12) as u8 // +9 to make C = 0
    }

    /// Calculate Pearson correlation coefficient between histogram and key profile
    ///
    /// # Arguments
    ///
    /// * `histogram` - Observed pitch class distribution
    /// * `profile` - Expected pitch class distribution for a key
    ///
    /// # Returns
    ///
    /// Correlation coefficient (-1.0 to 1.0, higher means better match)
    fn calculate_correlation(&self, histogram: &[f32; 12], profile: &[f32; 12]) -> f32 {
        // Pearson correlation coefficient
        let hist_mean: f32 = histogram.iter().sum::<f32>() / 12.0;
        let prof_mean: f32 = profile.iter().sum::<f32>() / 12.0;

        let mut numerator = 0.0;
        let mut hist_var = 0.0;
        let mut prof_var = 0.0;

        for i in 0..12 {
            let hist_dev = histogram[i] - hist_mean;
            let prof_dev = profile[i] - prof_mean;

            numerator += hist_dev * prof_dev;
            hist_var += hist_dev * hist_dev;
            prof_var += prof_dev * prof_dev;
        }

        let denominator = (hist_var * prof_var).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Parse internal key name into root note and mode
    ///
    /// # Arguments
    ///
    /// * `key_name` - Internal key name (e.g., "C_major", "A_minor")
    ///
    /// # Returns
    ///
    /// Tuple of (root_note, key_mode)
    fn parse_key_name(&self, key_name: &str) -> (String, KeyMode) {
        if key_name.contains("_major") {
            let root = key_name.replace("_major", "");
            (root, KeyMode::Major)
        } else if key_name.contains("_minor") {
            let root = key_name.replace("_minor", "");
            (root, KeyMode::Minor)
        } else {
            ("C".to_string(), KeyMode::Major)
        }
    }

    /// Format key name for display
    ///
    /// # Arguments
    ///
    /// * `root_note` - Root note (C, D, E, etc.)
    /// * `mode` - Key mode
    ///
    /// # Returns
    ///
    /// Human-readable key name (e.g., "C major", "A minor")
    fn format_key_name(&self, root_note: &str, mode: KeyMode) -> String {
        match mode {
            KeyMode::Major => format!("{} major", root_note),
            KeyMode::Minor => format!("{} minor", root_note),
            _ => format!("{} {:?}", root_note, mode),
        }
    }

    /// Set analysis window size for segmented key detection
    ///
    /// # Arguments
    ///
    /// * `window_size` - Window size in beats (minimum 1.0)
    pub fn set_window_size(&mut self, window_size: f32) {
        self.window_size = window_size.max(1.0);
    }

    /// Set minimum confidence threshold for key detection
    ///
    /// # Arguments
    ///
    /// * `min_confidence` - Minimum confidence (0.0-1.0) required for key detection
    pub fn set_min_confidence(&mut self, min_confidence: f32) {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
    }

    /// Get available key profiles
    ///
    /// # Returns
    ///
    /// Reference to the key profile database (24 profiles for all major and minor keys)
    pub fn key_profiles(&self) -> &HashMap<String, KeyProfile> {
        &self.key_profiles
    }
}

impl Default for KeyDetector {
    fn default() -> Self {
        Self::new()
    }
}
