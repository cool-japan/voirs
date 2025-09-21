//! Scale analysis and detection

use super::types::{ScaleCharacteristics, ScalePattern, ScaleResult};
use crate::types::NoteEvent;
use crate::Result;
use std::collections::HashMap;

/// Scale analysis system
#[derive(Debug, Clone)]
pub struct ScaleAnalyzer {
    /// Scale patterns for recognition
    scale_patterns: HashMap<String, ScalePattern>,
    /// Analysis threshold
    threshold: f32,
}

impl ScaleAnalyzer {
    /// Create a new scale analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            scale_patterns: HashMap::new(),
            threshold: 0.7,
        };

        analyzer.initialize_scale_patterns();
        analyzer
    }

    /// Initialize common scale patterns
    fn initialize_scale_patterns(&mut self) {
        // Major scale
        self.add_scale_pattern(
            "Major",
            vec![0, 2, 4, 5, 7, 9, 11],
            ScaleCharacteristics::major_scale(),
        );

        // Natural minor scale
        self.add_scale_pattern(
            "Natural Minor",
            vec![0, 2, 3, 5, 7, 8, 10],
            ScaleCharacteristics::minor_scale(),
        );

        // Pentatonic major
        self.add_scale_pattern(
            "Pentatonic Major",
            vec![0, 2, 4, 7, 9],
            ScaleCharacteristics {
                note_count: 5,
                brightness: 0.8,
                tension: 0.1,
                contexts: vec!["folk".to_string(), "pop".to_string()],
            },
        );

        // Blues scale
        self.add_scale_pattern(
            "Blues",
            vec![0, 3, 5, 6, 7, 10],
            ScaleCharacteristics {
                note_count: 6,
                brightness: 0.4,
                tension: 0.7,
                contexts: vec!["blues".to_string(), "jazz".to_string()],
            },
        );
    }

    /// Add a scale pattern
    fn add_scale_pattern(
        &mut self,
        name: &str,
        intervals: Vec<u8>,
        characteristics: ScaleCharacteristics,
    ) {
        let pattern = ScalePattern {
            name: name.to_string(),
            intervals,
            characteristics,
        };
        self.scale_patterns.insert(name.to_string(), pattern);
    }

    /// Analyze scales from note events
    pub async fn analyze_scales(&self, note_events: &[NoteEvent]) -> Result<Vec<ScaleResult>> {
        let mut scale_results = Vec::new();

        // Extract pitch classes from note events
        let pitch_classes = self.extract_pitch_classes(note_events);

        // Try each scale pattern
        for pattern in self.scale_patterns.values() {
            if let Some(result) = self.match_scale_pattern(pattern, &pitch_classes) {
                scale_results.push(result);
            }
        }

        // Sort by confidence
        scale_results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(scale_results)
    }

    /// Extract unique pitch classes from note events
    fn extract_pitch_classes(&self, note_events: &[NoteEvent]) -> Vec<u8> {
        let mut pitch_classes = Vec::new();

        for note in note_events {
            let pitch_class = self.frequency_to_pitch_class(note.frequency);
            if !pitch_classes.contains(&pitch_class) {
                pitch_classes.push(pitch_class);
            }
        }

        pitch_classes.sort();
        pitch_classes
    }

    /// Convert frequency to pitch class
    fn frequency_to_pitch_class(&self, frequency: f32) -> u8 {
        let a4_freq = 440.0;
        let semitones_from_a4 = 12.0 * (frequency / a4_freq).log2();
        let semitones = semitones_from_a4.round() as i32;
        ((semitones + 9) % 12) as u8 // +9 to make C = 0
    }

    /// Match scale pattern against pitch classes
    fn match_scale_pattern(
        &self,
        pattern: &ScalePattern,
        pitch_classes: &[u8],
    ) -> Option<ScaleResult> {
        let mut best_confidence = 0.0;
        let mut best_root = 0;

        // Try each possible root note
        for root in 0..12 {
            let confidence = self.calculate_scale_confidence(pattern, pitch_classes, root);
            if confidence > best_confidence {
                best_confidence = confidence;
                best_root = root;
            }
        }

        if best_confidence >= self.threshold {
            let scale_notes = self.generate_scale_notes(pattern, best_root);

            Some(ScaleResult {
                scale_name: pattern.name.clone(),
                root_note: self.pitch_class_to_note_name(best_root),
                intervals: pattern.intervals.clone(),
                confidence: best_confidence,
                characteristics: pattern.characteristics.clone(),
                scale_notes,
            })
        } else {
            None
        }
    }

    /// Calculate confidence for scale pattern match
    fn calculate_scale_confidence(
        &self,
        pattern: &ScalePattern,
        pitch_classes: &[u8],
        root: u8,
    ) -> f32 {
        let expected_pitch_classes: Vec<u8> = pattern
            .intervals
            .iter()
            .map(|&interval| (root + interval) % 12)
            .collect();

        let intersection = pitch_classes
            .iter()
            .filter(|&&pc| expected_pitch_classes.contains(&pc))
            .count();

        let expected_count = expected_pitch_classes.len();
        let actual_count = pitch_classes.len();

        if expected_count == 0 {
            return 0.0;
        }

        // Calculate recall (how many expected notes are present)
        let recall = intersection as f32 / expected_count as f32;

        // Calculate precision (how many actual notes are expected)
        let precision = if actual_count > 0 {
            intersection as f32 / actual_count as f32
        } else {
            0.0
        };

        // F1 score as confidence measure
        if recall + precision > 0.0 {
            2.0 * (recall * precision) / (recall + precision)
        } else {
            0.0
        }
    }

    /// Generate note names for a scale
    fn generate_scale_notes(&self, pattern: &ScalePattern, root: u8) -> Vec<String> {
        pattern
            .intervals
            .iter()
            .map(|&interval| self.pitch_class_to_note_name((root + interval) % 12))
            .collect()
    }

    /// Convert pitch class to note name
    fn pitch_class_to_note_name(&self, pitch_class: u8) -> String {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        note_names[pitch_class as usize % 12].to_string()
    }

    /// Set analysis threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get available scale patterns
    pub fn scale_patterns(&self) -> &HashMap<String, ScalePattern> {
        &self.scale_patterns
    }
}

impl Default for ScaleAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
