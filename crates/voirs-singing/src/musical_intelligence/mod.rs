//! Musical Intelligence Features
//!
//! This module has been refactored into smaller, more manageable components
//! for better maintainability and organization. It provides AI-powered musical
//! analysis features including chord recognition, key detection, scale analysis,
//! and rhythm pattern analysis for singing voice synthesis.

pub mod chord_recognition;
pub mod core;
pub mod key_detection;
pub mod rhythm_analysis;
pub mod scale_analysis;
pub mod types;

// Re-export main types and traits
pub use core::MusicalIntelligence;

pub use chord_recognition::ChordRecognizer;
pub use key_detection::KeyDetector;
pub use rhythm_analysis::RhythmAnalyzer;
pub use scale_analysis::ScaleAnalyzer;

pub use types::{
    ChordQuality, ChordResult, ChordTemplate, GrooveCharacteristics, KeyMode, KeyProfile,
    KeyResult, MusicalAnalysis, RhythmPattern, RhythmResult, ScaleCharacteristics, ScalePattern,
    ScaleResult,
};

/// Create a default musical intelligence system with all analysis capabilities enabled
///
/// # Returns
///
/// Fully configured musical intelligence system ready for analysis
pub fn create_default_system() -> MusicalIntelligence {
    MusicalIntelligence::default()
}

/// Musical analysis capabilities available in the musical intelligence system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisCapability {
    /// Chord recognition from note events using pattern matching
    ChordRecognition,
    /// Key signature detection using Krumhansl-Schmuckler algorithm
    KeyDetection,
    /// Musical scale analysis and pattern matching
    ScaleAnalysis,
    /// Rhythm and tempo analysis from note timing
    RhythmAnalysis,
    /// Audio onset detection from waveform energy
    OnsetDetection,
    /// Groove analysis including microtiming and syncopation
    GrooveAnalysis,
}

/// Get available analysis capabilities
///
/// # Returns
///
/// Vector of all supported analysis capabilities
pub fn get_analysis_capabilities() -> Vec<AnalysisCapability> {
    vec![
        AnalysisCapability::ChordRecognition,
        AnalysisCapability::KeyDetection,
        AnalysisCapability::ScaleAnalysis,
        AnalysisCapability::RhythmAnalysis,
        AnalysisCapability::OnsetDetection,
        AnalysisCapability::GrooveAnalysis,
    ]
}

/// Check if a specific capability is available
///
/// # Arguments
///
/// * `capability` - The capability to check
///
/// # Returns
///
/// True if the capability is supported, false otherwise
pub fn has_capability(capability: AnalysisCapability) -> bool {
    get_analysis_capabilities().contains(&capability)
}

/// Configuration for musical analysis operations
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Enable chord recognition in analysis pipeline
    pub enable_chord_recognition: bool,
    /// Enable key detection in analysis pipeline
    pub enable_key_detection: bool,
    /// Enable scale analysis in analysis pipeline
    pub enable_scale_analysis: bool,
    /// Enable rhythm analysis in analysis pipeline
    pub enable_rhythm_analysis: bool,
    /// Minimum confidence threshold (0.0-1.0) for accepting analysis results
    pub min_confidence: f32,
}

impl AnalysisConfig {
    /// Full analysis configuration with all features enabled
    ///
    /// # Returns
    ///
    /// Configuration with all analysis capabilities enabled and moderate confidence threshold (0.6)
    pub fn full() -> Self {
        Self {
            enable_chord_recognition: true,
            enable_key_detection: true,
            enable_scale_analysis: true,
            enable_rhythm_analysis: true,
            min_confidence: 0.6,
        }
    }

    /// Harmonic analysis only (chord, key, and scale detection)
    ///
    /// # Returns
    ///
    /// Configuration with harmonic analysis enabled, rhythm disabled, higher confidence threshold (0.7)
    pub fn harmonic_only() -> Self {
        Self {
            enable_chord_recognition: true,
            enable_key_detection: true,
            enable_scale_analysis: true,
            enable_rhythm_analysis: false,
            min_confidence: 0.7,
        }
    }

    /// Rhythm analysis only (tempo, groove, and timing detection)
    ///
    /// # Returns
    ///
    /// Configuration with rhythm analysis enabled, harmonic analysis disabled, lower confidence threshold (0.5)
    pub fn rhythm_only() -> Self {
        Self {
            enable_chord_recognition: false,
            enable_key_detection: false,
            enable_scale_analysis: false,
            enable_rhythm_analysis: true,
            min_confidence: 0.5,
        }
    }

    /// Fast analysis with relaxed thresholds (chord and key only)
    ///
    /// # Returns
    ///
    /// Configuration optimized for speed with chord and key detection only, low confidence threshold (0.5)
    pub fn fast() -> Self {
        Self {
            enable_chord_recognition: true,
            enable_key_detection: true,
            enable_scale_analysis: false,
            enable_rhythm_analysis: false,
            min_confidence: 0.5,
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self::full()
    }
}

/// Analysis utilities
pub mod utils {
    use super::*;

    /// Convert MIDI note number to note name
    ///
    /// # Arguments
    ///
    /// * `midi_note` - MIDI note number (0-127)
    ///
    /// # Returns
    ///
    /// Note name without octave (C, C#, D, etc.)
    pub fn midi_to_note_name(midi_note: u8) -> String {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let note_index = (midi_note % 12) as usize;
        note_names[note_index].to_string()
    }

    /// Convert note name to MIDI note number in middle octave (C4=60)
    ///
    /// # Arguments
    ///
    /// * `note_name` - Note name (C, C#, D, etc.)
    ///
    /// # Returns
    ///
    /// MIDI note number in octave 4, or None if note name is invalid
    pub fn note_name_to_midi(note_name: &str) -> Option<u8> {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        note_names
            .iter()
            .position(|&n| n == note_name)
            .map(|i| i as u8 + 60) // C4 = 60
    }

    /// Convert frequency to MIDI note number
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Closest MIDI note number (0-127)
    pub fn frequency_to_midi(frequency: f32) -> u8 {
        let a4_freq = 440.0;
        let midi_a4 = 69;
        let semitones = 12.0 * (frequency / a4_freq).log2();
        (midi_a4 as f32 + semitones).round().clamp(0.0, 127.0) as u8
    }

    /// Convert MIDI note number to frequency in Hz
    ///
    /// # Arguments
    ///
    /// * `midi_note` - MIDI note number (0-127)
    ///
    /// # Returns
    ///
    /// Frequency in Hz (A4=440Hz)
    pub fn midi_to_frequency(midi_note: u8) -> f32 {
        let a4_freq = 440.0;
        let midi_a4 = 69;
        a4_freq * 2.0_f32.powf((midi_note as f32 - midi_a4 as f32) / 12.0)
    }

    /// Calculate interval between two notes in semitones
    ///
    /// # Arguments
    ///
    /// * `note1` - First MIDI note number
    /// * `note2` - Second MIDI note number
    ///
    /// # Returns
    ///
    /// Interval in semitones (modulo 12)
    pub fn calculate_interval(note1: u8, note2: u8) -> i8 {
        (note2 as i8 - note1 as i8) % 12
    }

    /// Check if an interval is consonant (pleasant-sounding)
    ///
    /// # Arguments
    ///
    /// * `interval` - Interval in semitones
    ///
    /// # Returns
    ///
    /// True if interval is consonant (unison, 3rds, 5th, 6ths), false otherwise
    pub fn is_consonant_interval(interval: u8) -> bool {
        matches!(interval % 12, 0 | 3 | 4 | 7 | 8 | 9) // Unison, minor/major 3rd, 5th, minor/major 6th
    }

    /// Get chord quality from interval pattern
    ///
    /// # Arguments
    ///
    /// * `intervals` - Semitone intervals from root (e.g., [0, 4, 7] for major triad)
    ///
    /// # Returns
    ///
    /// Chord quality if pattern matches a known chord type, None otherwise
    pub fn intervals_to_chord_quality(intervals: &[u8]) -> Option<ChordQuality> {
        match intervals {
            [0, 4, 7] => Some(ChordQuality::Major),
            [0, 3, 7] => Some(ChordQuality::Minor),
            [0, 3, 6] => Some(ChordQuality::Diminished),
            [0, 4, 8] => Some(ChordQuality::Augmented),
            [0, 4, 7, 10] => Some(ChordQuality::Dominant7),
            [0, 4, 7, 11] => Some(ChordQuality::Major7),
            [0, 3, 7, 10] => Some(ChordQuality::Minor7),
            [0, 5, 7] => Some(ChordQuality::Suspended4),
            [0, 2, 7] => Some(ChordQuality::Suspended2),
            _ => None,
        }
    }

    /// Normalize pitch class to 0-11 range using modulo arithmetic
    ///
    /// # Arguments
    ///
    /// * `pitch_class` - Pitch class (can be negative or > 11)
    ///
    /// # Returns
    ///
    /// Normalized pitch class (0-11)
    pub fn normalize_pitch_class(pitch_class: i32) -> u8 {
        pitch_class.rem_euclid(12) as u8
    }
}
