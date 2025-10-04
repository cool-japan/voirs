//! Chord recognition and analysis

use super::types::{ChordQuality, ChordResult, ChordTemplate};
use crate::types::NoteEvent;
use crate::Result;
use std::collections::HashMap;

/// Automatic chord detection and recognition system
#[derive(Debug, Clone)]
pub struct ChordRecognizer {
    /// Chord templates for pattern-based recognition (maps chord name to template)
    chord_templates: HashMap<String, ChordTemplate>,
    /// Minimum confidence threshold for chord recognition (0.0-1.0)
    threshold: f32,
    /// Enable detection of jazz chord extensions (7ths, 9ths, etc.)
    enable_extensions: bool,
}

impl ChordRecognizer {
    /// Create a new chord recognizer
    pub fn new() -> Self {
        let mut recognizer = Self {
            chord_templates: HashMap::new(),
            threshold: 0.6,
            enable_extensions: false,
        };

        recognizer.initialize_chord_templates();
        recognizer
    }

    /// Initialize basic chord templates for major, minor, and dominant 7th chords
    fn initialize_chord_templates(&mut self) {
        // Major triads
        self.add_chord_template("C", 0, vec![0, 4, 7], ChordQuality::Major);
        self.add_chord_template("D", 2, vec![0, 4, 7], ChordQuality::Major);
        self.add_chord_template("E", 4, vec![0, 4, 7], ChordQuality::Major);
        self.add_chord_template("F", 5, vec![0, 4, 7], ChordQuality::Major);
        self.add_chord_template("G", 7, vec![0, 4, 7], ChordQuality::Major);
        self.add_chord_template("A", 9, vec![0, 4, 7], ChordQuality::Major);
        self.add_chord_template("B", 11, vec![0, 4, 7], ChordQuality::Major);

        // Minor triads
        self.add_chord_template("Cm", 0, vec![0, 3, 7], ChordQuality::Minor);
        self.add_chord_template("Dm", 2, vec![0, 3, 7], ChordQuality::Minor);
        self.add_chord_template("Em", 4, vec![0, 3, 7], ChordQuality::Minor);
        self.add_chord_template("Fm", 5, vec![0, 3, 7], ChordQuality::Minor);
        self.add_chord_template("Gm", 7, vec![0, 3, 7], ChordQuality::Minor);
        self.add_chord_template("Am", 9, vec![0, 3, 7], ChordQuality::Minor);
        self.add_chord_template("Bm", 11, vec![0, 3, 7], ChordQuality::Minor);

        // Dominant 7th chords
        self.add_chord_template("C7", 0, vec![0, 4, 7, 10], ChordQuality::Dominant7);
        self.add_chord_template("G7", 7, vec![0, 4, 7, 10], ChordQuality::Dominant7);
        self.add_chord_template("D7", 2, vec![0, 4, 7, 10], ChordQuality::Dominant7);
    }

    /// Add a chord template to the recognition database
    ///
    /// # Arguments
    ///
    /// * `name` - Chord name identifier
    /// * `root` - Root note pitch class (0-11)
    /// * `intervals` - Semitone intervals from root
    /// * `quality` - Chord quality type
    fn add_chord_template(
        &mut self,
        name: &str,
        root: u8,
        intervals: Vec<u8>,
        quality: ChordQuality,
    ) {
        let template = ChordTemplate::new(name.to_string(), root, intervals, quality);
        self.chord_templates.insert(name.to_string(), template);
    }

    /// Analyze chords from note events using sliding window approach
    ///
    /// # Arguments
    ///
    /// * `note_events` - Sequence of musical note events to analyze
    ///
    /// # Returns
    ///
    /// Vector of recognized chords with confidence scores
    ///
    /// # Errors
    ///
    /// Returns error if chord recognition fails
    pub async fn analyze_chords(&self, note_events: &[NoteEvent]) -> Result<Vec<ChordResult>> {
        let mut chord_results = Vec::new();

        // Group notes into chord windows (simplified approach)
        let window_size = 2.0; // 2 second windows
        let mut current_time = 0.0;

        while current_time < self.get_total_duration(note_events) {
            let window_notes = self.get_notes_in_window(note_events, current_time, window_size);

            if !window_notes.is_empty() {
                if let Some(chord_result) = self.recognize_chord(&window_notes)? {
                    chord_results.push(chord_result);
                }
            }

            current_time += window_size / 2.0; // 50% overlap
        }

        Ok(chord_results)
    }

    /// Get total duration of note events in seconds
    ///
    /// # Arguments
    ///
    /// * `note_events` - Note events to measure
    ///
    /// # Returns
    ///
    /// Total duration from start to end of all notes
    fn get_total_duration(&self, note_events: &[NoteEvent]) -> f32 {
        note_events
            .iter()
            .map(|note| note.timing_offset + note.duration)
            .fold(0.0, f32::max)
    }

    /// Get notes within a time window
    ///
    /// # Arguments
    ///
    /// * `note_events` - All note events
    /// * `start_time` - Window start time in seconds
    /// * `duration` - Window duration in seconds
    ///
    /// # Returns
    ///
    /// Notes that overlap with the specified time window
    fn get_notes_in_window<'a>(
        &self,
        note_events: &'a [NoteEvent],
        start_time: f32,
        duration: f32,
    ) -> Vec<&'a NoteEvent> {
        note_events
            .iter()
            .filter(|note| {
                let note_start = note.timing_offset;
                let note_end = note.timing_offset + note.duration;
                let window_end = start_time + duration;

                // Note overlaps with window
                note_start < window_end && note_end > start_time
            })
            .collect()
    }

    /// Recognize chord from a collection of notes by template matching
    ///
    /// # Arguments
    ///
    /// * `notes` - Collection of simultaneous or overlapping notes
    ///
    /// # Returns
    ///
    /// Recognized chord result or None if no chord matches above threshold
    ///
    /// # Errors
    ///
    /// Returns error if chord matching encounters issues
    fn recognize_chord(&self, notes: &[&NoteEvent]) -> Result<Option<ChordResult>> {
        if notes.len() < 2 {
            return Ok(None); // Need at least 2 notes for a chord
        }

        // Extract pitch classes
        let pitch_classes = self.extract_pitch_classes(notes);

        // Find best matching chord template
        let mut best_match = None;
        let mut best_confidence = 0.0;

        for template in self.chord_templates.values() {
            let confidence = self.calculate_chord_confidence(template, &pitch_classes);

            if confidence > best_confidence && confidence >= self.threshold {
                best_confidence = confidence;
                best_match = Some(template);
            }
        }

        if let Some(template) = best_match {
            Ok(Some(ChordResult {
                chord_name: template.name.clone(),
                root_note: self.pitch_class_to_note_name(template.root),
                quality: template.quality.clone(),
                confidence: best_confidence,
                inversion: 0, // Simplified - not detecting inversions
                bass_note: None,
                extensions: Vec::new(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Extract unique pitch classes from notes
    ///
    /// # Arguments
    ///
    /// * `notes` - Notes to extract pitch classes from
    ///
    /// # Returns
    ///
    /// Sorted vector of unique pitch classes (0-11)
    fn extract_pitch_classes(&self, notes: &[&NoteEvent]) -> Vec<u8> {
        let mut pitch_classes = Vec::new();

        for note in notes {
            let pitch_class = self.frequency_to_pitch_class(note.frequency);
            if !pitch_classes.contains(&pitch_class) {
                pitch_classes.push(pitch_class);
            }
        }

        pitch_classes.sort();
        pitch_classes
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

    /// Convert pitch class to note name
    ///
    /// # Arguments
    ///
    /// * `pitch_class` - Pitch class (0-11)
    ///
    /// # Returns
    ///
    /// Note name as string (C, C#, D, etc.)
    fn pitch_class_to_note_name(&self, pitch_class: u8) -> String {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        note_names[pitch_class as usize % 12].to_string()
    }

    /// Calculate chord confidence score using Jaccard similarity
    ///
    /// # Arguments
    ///
    /// * `template` - Chord template to match against
    /// * `pitch_classes` - Detected pitch classes from audio
    ///
    /// # Returns
    ///
    /// Confidence score (0.0-1.0) based on pitch class overlap
    fn calculate_chord_confidence(&self, template: &ChordTemplate, pitch_classes: &[u8]) -> f32 {
        if pitch_classes.is_empty() || template.intervals.is_empty() {
            return 0.0;
        }

        // Create expected pitch classes for this chord
        let expected_pitch_classes: Vec<u8> = template
            .intervals
            .iter()
            .map(|&interval| (template.root + interval) % 12)
            .collect();

        // Calculate intersection and union
        let intersection = pitch_classes
            .iter()
            .filter(|&&pc| expected_pitch_classes.contains(&pc))
            .count();

        let union = pitch_classes.len() + expected_pitch_classes.len() - intersection;

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Set recognition threshold for chord detection
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum confidence (0.0-1.0) required for chord recognition
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Enable jazz chord extensions (7ths, 9ths, etc.)
    ///
    /// # Arguments
    ///
    /// * `enable` - True to enable extended chord recognition
    pub fn enable_extensions(&mut self, enable: bool) {
        self.enable_extensions = enable;

        if enable {
            self.add_extended_chords();
        }
    }

    /// Add extended chord templates (maj7, m7) to recognition database
    fn add_extended_chords(&mut self) {
        // Major 7th chords
        self.add_chord_template("Cmaj7", 0, vec![0, 4, 7, 11], ChordQuality::Major7);
        self.add_chord_template("Fmaj7", 5, vec![0, 4, 7, 11], ChordQuality::Major7);
        self.add_chord_template("Gmaj7", 7, vec![0, 4, 7, 11], ChordQuality::Major7);

        // Minor 7th chords
        self.add_chord_template("Dm7", 2, vec![0, 3, 7, 10], ChordQuality::Minor7);
        self.add_chord_template("Em7", 4, vec![0, 3, 7, 10], ChordQuality::Minor7);
        self.add_chord_template("Am7", 9, vec![0, 3, 7, 10], ChordQuality::Minor7);
    }

    /// Get available chord templates
    ///
    /// # Returns
    ///
    /// Reference to the chord template database
    pub fn chord_templates(&self) -> &HashMap<String, ChordTemplate> {
        &self.chord_templates
    }
}

impl Default for ChordRecognizer {
    fn default() -> Self {
        Self::new()
    }
}
